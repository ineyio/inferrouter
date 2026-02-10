package gonka

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Endpoint represents a Gonka inference node.
type Endpoint struct {
	URL     string // HTTP endpoint (e.g. "https://node1.gonka.ai/v1")
	Address string // Cosmos bech32 address of the node (transfer_address in signing)
}

// signingTransport is an http.RoundTripper that intercepts requests,
// reads the private key from the Authorization header, signs the body,
// and sets Gonka-specific headers.
type signingTransport struct {
	base     http.RoundTripper
	keys     *keyCache
	endpoint Endpoint
	nowFunc  func() time.Time
}

func newSigningTransport(base http.RoundTripper, endpoint Endpoint) *signingTransport {
	return &signingTransport{
		base:     base,
		keys:     newKeyCache(),
		endpoint: endpoint,
	}
}

func (t *signingTransport) now() time.Time {
	if t.nowFunc != nil {
		return t.nowFunc()
	}
	return time.Now()
}

// RoundTrip implements http.RoundTripper.
func (t *signingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// 1. Extract hex private key from Authorization: Bearer <key>
	authHeader := req.Header.Get("Authorization")
	if !strings.HasPrefix(authHeader, "Bearer ") {
		return nil, fmt.Errorf("gonka: missing Bearer authorization header")
	}
	hexKey := strings.TrimSpace(strings.TrimPrefix(authHeader, "Bearer "))

	// 2. Parse key + derive address (cached)
	ki, err := t.keys.get(hexKey)
	if err != nil {
		return nil, fmt.Errorf("gonka: %w", err)
	}

	// 3. Read request body
	var body []byte
	if req.Body != nil {
		body, err = io.ReadAll(req.Body)
		req.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("gonka: read request body: %w", err)
		}
	}

	// 4. Sign
	tsNanos := t.now().UnixNano()
	signature, err := signRequest(ki.privKey, body, tsNanos, t.endpoint.Address)
	if err != nil {
		return nil, fmt.Errorf("gonka: sign: %w", err)
	}

	// 5. Clone request and set Gonka headers
	clone := req.Clone(req.Context())
	clone.Header.Set("Authorization", signature)
	clone.Header.Set("X-Requester-Address", ki.address)
	clone.Header.Set("X-Timestamp", strconv.FormatInt(tsNanos, 10))

	// 6. Restore body
	clone.Body = io.NopCloser(bytes.NewReader(body))
	clone.ContentLength = int64(len(body))

	return t.base.RoundTrip(clone)
}
