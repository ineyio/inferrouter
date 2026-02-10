package gonka

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math/big"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/decred/dcrd/dcrec/secp256k1/v4"
	"github.com/decred/dcrd/dcrec/secp256k1/v4/ecdsa"
	ir "github.com/ineyio/inferrouter"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testKeyHex = "a]0b4ef5f3232b4c1a4b1f5e9c7d83a6e2f1d0c9b8a7968574636251403f2e1d"

func init() {
	// Sanity: ensure testKeyHex is valid. Replace placeholder with deterministic key.
}

// Use a known valid 32-byte hex key for all tests.
const validKeyHex = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

// --- bech32 tests ---

func TestConvertBits_8to5(t *testing.T) {
	// Convert 0x00 0x01 (8-bit) to 5-bit groups with padding.
	out, err := convertBits([]byte{0x00, 0x01}, 8, 5, true)
	require.NoError(t, err)
	assert.NotEmpty(t, out)
}

func TestBech32Encode(t *testing.T) {
	// Known bech32 test: encode "gonka" prefix with zero data.
	data, err := convertBits(make([]byte, 20), 8, 5, true)
	require.NoError(t, err)

	addr, err := bech32Encode("gonka", data)
	require.NoError(t, err)
	assert.True(t, strings.HasPrefix(addr, "gonka1"))
	// All zero address should have a consistent encoding.
	t.Logf("zero address: %s", addr)
}

// --- crypto tests ---

func TestParsePrivateKey_Valid(t *testing.T) {
	key, err := parsePrivateKey(validKeyHex)
	require.NoError(t, err)
	assert.NotNil(t, key)
}

func TestParsePrivateKey_With0xPrefix(t *testing.T) {
	key, err := parsePrivateKey("0x" + validKeyHex)
	require.NoError(t, err)
	assert.NotNil(t, key)
}

func TestParsePrivateKey_InvalidHex(t *testing.T) {
	_, err := parsePrivateKey("not-hex-at-all")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "invalid private key hex")
}

func TestParsePrivateKey_WrongLength(t *testing.T) {
	_, err := parsePrivateKey("0123456789abcdef") // 8 bytes
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "must be 32 bytes")
}

func TestDeriveAddress(t *testing.T) {
	key, err := parsePrivateKey(validKeyHex)
	require.NoError(t, err)

	addr, err := deriveAddress(key)
	require.NoError(t, err)
	assert.True(t, strings.HasPrefix(addr, "gonka1"), "address should start with gonka1, got %s", addr)
	t.Logf("derived address: %s", addr)

	// Same key produces same address.
	addr2, err := deriveAddress(key)
	require.NoError(t, err)
	assert.Equal(t, addr, addr2)
}

func TestSignRequest_Deterministic(t *testing.T) {
	key, err := parsePrivateKey(validKeyHex)
	require.NoError(t, err)

	body := []byte(`{"model":"test","messages":[]}`)
	ts := int64(1700000000000000000)
	addr := "gonka1testaddr"

	sig1, err := signRequest(key, body, ts, addr)
	require.NoError(t, err)

	sig2, err := signRequest(key, body, ts, addr)
	require.NoError(t, err)

	// RFC6979 is deterministic.
	assert.Equal(t, sig1, sig2)
}

func TestSignRequest_LowS(t *testing.T) {
	key, err := parsePrivateKey(validKeyHex)
	require.NoError(t, err)

	halfOrder := new(big.Int).Rsh(secp256k1.S256().N, 1)

	// Sign multiple payloads and verify low-S.
	for i := 0; i < 10; i++ {
		body := []byte(fmt.Sprintf(`{"i":%d}`, i))
		sig, err := signRequest(key, body, int64(i), "gonka1addr")
		require.NoError(t, err)

		raw, err := base64.StdEncoding.DecodeString(sig)
		require.NoError(t, err)
		assert.Len(t, raw, 64, "signature should be 64 bytes (r+s)")

		s := new(big.Int).SetBytes(raw[32:64])
		assert.True(t, s.Cmp(halfOrder) <= 0, "s should be <= halfOrder (low-S)")
	}
}

func TestSignRequest_EmptyBody(t *testing.T) {
	key, err := parsePrivateKey(validKeyHex)
	require.NoError(t, err)

	sig, err := signRequest(key, nil, 1700000000000000000, "gonka1addr")
	require.NoError(t, err)
	assert.NotEmpty(t, sig)
}

// --- key cache tests ---

func TestKeyCache_CachesKey(t *testing.T) {
	cache := newKeyCache()

	ki1, err := cache.get(validKeyHex)
	require.NoError(t, err)

	ki2, err := cache.get(validKeyHex)
	require.NoError(t, err)

	// Same pointer — cache hit.
	assert.Same(t, ki1, ki2)
}

func TestKeyCache_ConcurrentAccess(t *testing.T) {
	cache := newKeyCache()
	var wg sync.WaitGroup

	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ki, err := cache.get(validKeyHex)
			assert.NoError(t, err)
			assert.NotNil(t, ki)
		}()
	}
	wg.Wait()
}

// --- RIPEMD-160 test ---

func TestRipemd160_KnownVector(t *testing.T) {
	// RIPEMD-160("") = 9c1185a5c5e9fc54612808977ee8f548b2258d31
	digest := ripemd160Sum([]byte(""))
	expected := "9c1185a5c5e9fc54612808977ee8f548b2258d31"
	assert.Equal(t, expected, hex.EncodeToString(digest[:]))
}

func TestRipemd160_ABC(t *testing.T) {
	// RIPEMD-160("abc") = 8eb208f7e05d987a9b044a8e98c6b087f15a0bfc
	digest := ripemd160Sum([]byte("abc"))
	expected := "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc"
	assert.Equal(t, expected, hex.EncodeToString(digest[:]))
}

// --- transport tests ---

func TestSigningTransport_SetsHeaders(t *testing.T) {
	fixedNow := time.Date(2025, 1, 15, 12, 0, 0, 0, time.UTC)

	var capturedReq *http.Request
	var capturedBody []byte

	inner := roundTripFunc(func(req *http.Request) (*http.Response, error) {
		capturedReq = req
		capturedBody, _ = io.ReadAll(req.Body)
		return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader("{}"))}, nil
	})

	endpoint := Endpoint{URL: "https://node.test/v1", Address: "gonka1nodeaddr"}
	transport := newSigningTransport(inner, endpoint)
	transport.nowFunc = func() time.Time { return fixedNow }

	body := strings.NewReader(`{"test":true}`)
	req, _ := http.NewRequest("POST", "https://node.test/v1/chat/completions", body)
	req.Header.Set("Authorization", "Bearer "+validKeyHex)

	resp, err := transport.RoundTrip(req)
	require.NoError(t, err)
	resp.Body.Close()

	// Authorization should NOT be "Bearer ..."
	auth := capturedReq.Header.Get("Authorization")
	assert.False(t, strings.HasPrefix(auth, "Bearer"), "Authorization should be signature, not Bearer token")

	// Should be valid base64 (64-byte signature).
	rawSig, err := base64.StdEncoding.DecodeString(auth)
	require.NoError(t, err)
	assert.Len(t, rawSig, 64)

	// X-Requester-Address should be set.
	reqAddr := capturedReq.Header.Get("X-Requester-Address")
	assert.True(t, strings.HasPrefix(reqAddr, "gonka1"))

	// X-Timestamp should be set.
	ts := capturedReq.Header.Get("X-Timestamp")
	assert.NotEmpty(t, ts)
	assert.Equal(t, fmt.Sprintf("%d", fixedNow.UnixNano()), ts)

	// Body should be preserved.
	assert.Equal(t, `{"test":true}`, string(capturedBody))
}

func TestSigningTransport_InvalidKey(t *testing.T) {
	inner := roundTripFunc(func(req *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200}, nil
	})

	transport := newSigningTransport(inner, Endpoint{URL: "https://test/v1", Address: "gonka1addr"})

	req, _ := http.NewRequest("POST", "https://test/v1/chat/completions", strings.NewReader("{}"))
	req.Header.Set("Authorization", "Bearer invalid-hex")

	_, err := transport.RoundTrip(req)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "gonka")
}

func TestSigningTransport_MultipleKeys(t *testing.T) {
	key1 := validKeyHex
	key2 := "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"

	var addresses []string
	inner := roundTripFunc(func(req *http.Request) (*http.Response, error) {
		addresses = append(addresses, req.Header.Get("X-Requester-Address"))
		return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader("{}"))}, nil
	})

	transport := newSigningTransport(inner, Endpoint{URL: "https://test/v1", Address: "gonka1addr"})

	for _, key := range []string{key1, key2} {
		req, _ := http.NewRequest("POST", "https://test/v1/chat/completions", strings.NewReader("{}"))
		req.Header.Set("Authorization", "Bearer "+key)
		resp, err := transport.RoundTrip(req)
		require.NoError(t, err)
		resp.Body.Close()
	}

	assert.Len(t, addresses, 2)
	assert.NotEqual(t, addresses[0], addresses[1], "different keys should produce different addresses")
}

// --- signature verification test ---

func TestSigningTransport_SignatureVerifies(t *testing.T) {
	fixedNow := time.Date(2025, 6, 1, 0, 0, 0, 0, time.UTC)
	endpoint := Endpoint{URL: "https://node.test/v1", Address: "gonka1nodeaddr"}

	var capturedAuth, capturedTS string
	var capturedBody []byte

	inner := roundTripFunc(func(req *http.Request) (*http.Response, error) {
		capturedAuth = req.Header.Get("Authorization")
		capturedTS = req.Header.Get("X-Timestamp")
		capturedBody, _ = io.ReadAll(req.Body)
		return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader("{}"))}, nil
	})

	transport := newSigningTransport(inner, endpoint)
	transport.nowFunc = func() time.Time { return fixedNow }

	bodyStr := `{"model":"test","messages":[{"role":"user","content":"hi"}]}`
	req, _ := http.NewRequest("POST", endpoint.URL+"/chat/completions", strings.NewReader(bodyStr))
	req.Header.Set("Authorization", "Bearer "+validKeyHex)

	resp, err := transport.RoundTrip(req)
	require.NoError(t, err)
	resp.Body.Close()

	// Verify signature manually.
	privKey, _ := parsePrivateKey(validKeyHex)

	bodyHash := sha256.Sum256(capturedBody)
	payloadHex := hex.EncodeToString(bodyHash[:])
	message := payloadHex + capturedTS + endpoint.Address
	digest := sha256.Sum256([]byte(message))

	rawSig, err := base64.StdEncoding.DecodeString(capturedAuth)
	require.NoError(t, err)

	// Parse r, s from raw signature.
	r := new(secp256k1.ModNScalar)
	r.SetByteSlice(rawSig[:32])
	s := new(secp256k1.ModNScalar)
	s.SetByteSlice(rawSig[32:64])

	sig := ecdsa.NewSignature(r, s)
	verified := sig.Verify(digest[:], privKey.PubKey())
	assert.True(t, verified, "signature should verify against the public key")
}

// --- provider tests ---

func newMockServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify Gonka headers.
		auth := r.Header.Get("Authorization")
		if strings.HasPrefix(auth, "Bearer") {
			http.Error(w, "expected signature, got Bearer token", 500)
			return
		}
		if r.Header.Get("X-Requester-Address") == "" {
			http.Error(w, "missing X-Requester-Address", 500)
			return
		}
		if r.Header.Get("X-Timestamp") == "" {
			http.Error(w, "missing X-Timestamp", 500)
			return
		}

		body, _ := io.ReadAll(r.Body)
		var req map[string]interface{}
		json.Unmarshal(body, &req)

		if req["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"id\":\"c-1\",\"model\":\"test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"}}]}\n\n")
			fmt.Fprint(w, "data: {\"id\":\"c-1\",\"model\":\"test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
			return
		}

		json.NewEncoder(w).Encode(map[string]interface{}{
			"id":    "c-1",
			"model": "test-model",
			"choices": []map[string]interface{}{
				{"index": 0, "message": map[string]string{"role": "assistant", "content": "Hello from Gonka!"}, "finish_reason": "stop"},
			},
			"usage": map[string]int64{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
		})
	}))
}

func newTestProvider(t *testing.T, serverURL string) *Provider {
	t.Helper()
	return New(
		WithEndpoint(Endpoint{URL: serverURL, Address: "gonka1testnode"}),
		WithModels("test-model"),
		WithTimeout(5*time.Second),
	)
}

func TestProvider_Name(t *testing.T) {
	p := New(WithEndpoint(Endpoint{URL: "https://test", Address: "gonka1x"}))
	assert.Equal(t, "gonka", p.Name())
}

func TestProvider_NameCustom(t *testing.T) {
	p := New(WithName("gonka-custom"), WithEndpoint(Endpoint{URL: "https://test", Address: "gonka1x"}))
	assert.Equal(t, "gonka-custom", p.Name())
}

func TestProvider_SupportsModel(t *testing.T) {
	p := New(WithEndpoint(Endpoint{URL: "https://test", Address: "gonka1x"}), WithModels("a", "b"))
	assert.True(t, p.SupportsModel("a"))
	assert.True(t, p.SupportsModel("b"))
	assert.False(t, p.SupportsModel("c"))
}

func TestProvider_SupportsModel_NoFilter(t *testing.T) {
	p := New(WithEndpoint(Endpoint{URL: "https://test", Address: "gonka1x"}))
	assert.True(t, p.SupportsModel("anything"))
}

func TestProvider_ChatCompletion(t *testing.T) {
	srv := newMockServer(t)
	defer srv.Close()

	p := newTestProvider(t, srv.URL)

	resp, err := p.ChatCompletion(context.Background(), ir.ProviderRequest{
		Auth:  ir.Auth{APIKey: validKeyHex},
		Model: "test-model",
		Messages: []ir.Message{
			{Role: "user", Content: "hello"},
		},
	})
	require.NoError(t, err)
	assert.Equal(t, "Hello from Gonka!", resp.Content)
	assert.Equal(t, int64(15), resp.Usage.TotalTokens)
	assert.Equal(t, "stop", resp.FinishReason)
}

func TestProvider_ChatCompletionStream(t *testing.T) {
	srv := newMockServer(t)
	defer srv.Close()

	p := newTestProvider(t, srv.URL)

	stream, err := p.ChatCompletionStream(context.Background(), ir.ProviderRequest{
		Auth:  ir.Auth{APIKey: validKeyHex},
		Model: "test-model",
		Messages: []ir.Message{
			{Role: "user", Content: "hello"},
		},
		Stream: true,
	})
	require.NoError(t, err)
	defer stream.Close()

	var content string
	for {
		chunk, err := stream.Next()
		if err != nil {
			break
		}
		for _, c := range chunk.Choices {
			content += c.Delta.Content
		}
	}
	assert.Equal(t, "hello world", content)
}

func TestProvider_HTTPError_401(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		fmt.Fprint(w, "unauthorized")
	}))
	defer srv.Close()

	p := newTestProvider(t, srv.URL)
	_, err := p.ChatCompletion(context.Background(), ir.ProviderRequest{
		Auth:     ir.Auth{APIKey: validKeyHex},
		Model:    "test-model",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})

	assert.ErrorIs(t, err, ir.ErrAuthFailed)
}

func TestProvider_HTTPError_429(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer srv.Close()

	p := newTestProvider(t, srv.URL)
	_, err := p.ChatCompletion(context.Background(), ir.ProviderRequest{
		Auth:     ir.Auth{APIKey: validKeyHex},
		Model:    "test-model",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})

	assert.ErrorIs(t, err, ir.ErrRateLimited)
}

func TestProvider_HTTPError_500(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	p := newTestProvider(t, srv.URL)
	_, err := p.ChatCompletion(context.Background(), ir.ProviderRequest{
		Auth:     ir.Auth{APIKey: validKeyHex},
		Model:    "test-model",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})

	assert.ErrorIs(t, err, ir.ErrProviderUnavailable)
}

// --- helpers ---

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}
