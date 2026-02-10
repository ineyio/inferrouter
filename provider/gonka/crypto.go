package gonka

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"
	"sync"

	"github.com/decred/dcrd/dcrec/secp256k1/v4"
	"github.com/decred/dcrd/dcrec/secp256k1/v4/ecdsa"
)

// keyInfo holds a parsed private key and its derived Cosmos address.
type keyInfo struct {
	privKey *secp256k1.PrivateKey
	address string // bech32 "gonka1..."
}

// keyCache is a thread-safe cache of parsed keys.
type keyCache struct {
	mu    sync.RWMutex
	cache map[string]*keyInfo
}

func newKeyCache() *keyCache {
	return &keyCache{cache: make(map[string]*keyInfo)}
}

// get retrieves or creates a keyInfo for the given hex-encoded private key.
func (c *keyCache) get(hexKey string) (*keyInfo, error) {
	c.mu.RLock()
	ki, ok := c.cache[hexKey]
	c.mu.RUnlock()
	if ok {
		return ki, nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Double-check after write lock.
	if ki, ok := c.cache[hexKey]; ok {
		return ki, nil
	}

	privKey, err := parsePrivateKey(hexKey)
	if err != nil {
		return nil, err
	}

	addr, err := deriveAddress(privKey)
	if err != nil {
		return nil, err
	}

	ki = &keyInfo{privKey: privKey, address: addr}
	c.cache[hexKey] = ki
	return ki, nil
}

// parsePrivateKey decodes a hex string into a secp256k1 private key.
func parsePrivateKey(hexKey string) (*secp256k1.PrivateKey, error) {
	hexKey = strings.TrimPrefix(hexKey, "0x")
	hexKey = strings.TrimPrefix(hexKey, "0X")

	keyBytes, err := hex.DecodeString(hexKey)
	if err != nil {
		return nil, fmt.Errorf("gonka: invalid private key hex: %w", err)
	}
	if len(keyBytes) != 32 {
		return nil, fmt.Errorf("gonka: private key must be 32 bytes, got %d", len(keyBytes))
	}

	privKey := secp256k1.PrivKeyFromBytes(keyBytes)
	if privKey.Key.IsZero() {
		return nil, fmt.Errorf("gonka: private key is zero")
	}

	return privKey, nil
}

// deriveAddress computes the Cosmos bech32 address from a secp256k1 private key.
// Pipeline: compressed pubkey → SHA256 → RIPEMD160 → bech32("gonka").
func deriveAddress(privKey *secp256k1.PrivateKey) (string, error) {
	compressed := privKey.PubKey().SerializeCompressed() // 33 bytes

	sha := sha256.Sum256(compressed)

	rip := ripemd160Sum(sha[:])

	bits5, err := convertBits(rip[:], 8, 5, true)
	if err != nil {
		return "", fmt.Errorf("gonka: convert bits: %w", err)
	}

	return bech32Encode("gonka", bits5)
}

// signRequest produces the ECDSA signature for a Gonka API request.
// Returns the base64-encoded raw signature (r || s, 64 bytes).
func signRequest(privKey *secp256k1.PrivateKey, body []byte, tsNanos int64, transferAddr string) (string, error) {
	// 1. hex(SHA256(body))
	bodyHash := sha256.Sum256(body)
	payloadHex := hex.EncodeToString(bodyHash[:])

	// 2. message = payloadHex + timestamp + transferAddress
	tsStr := strconv.FormatInt(tsNanos, 10)
	message := payloadHex + tsStr + transferAddr

	// 3. SHA256(message)
	digest := sha256.Sum256([]byte(message))

	// 4. ECDSA sign (RFC6979 deterministic, low-S by default in dcrd)
	compactSig := ecdsa.SignCompact(privKey, digest[:], false)
	// compactSig: [recovery_flag, r(32), s(32)] = 65 bytes

	// 5. Extract r || s (skip recovery flag byte)
	rawSig := compactSig[1:65]

	return base64.StdEncoding.EncodeToString(rawSig), nil
}
