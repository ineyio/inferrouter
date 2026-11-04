package openaicompat

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"syscall"
	"testing"
	"time"

	ir "github.com/ineyio/inferrouter"
	"github.com/ineyio/inferrouter/quota"
)

func chatRequest() ir.ProviderRequest {
	return ir.ProviderRequest{
		Auth:     ir.Auth{APIKey: "k"},
		Model:    "m",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	}
}

type failingTransport struct{ err error }

func (t failingTransport) RoundTrip(*http.Request) (*http.Response, error) { return nil, t.err }

func TestTransportCauseIsReachable(t *testing.T) {
	t.Run("refused", func(t *testing.T) {
		srv := httptest.NewServer(http.NotFoundHandler())
		srv.Close()
		_, err := New("x", srv.URL).ChatCompletion(context.Background(), chatRequest())
		if !errors.Is(err, ir.ErrProviderUnavailable) || !errors.Is(err, syscall.ECONNREFUSED) {
			t.Errorf("err = %v, want both the sentinel and ECONNREFUSED", err)
		}
	})

	t.Run("dns", func(t *testing.T) {
		client := &http.Client{Transport: failingTransport{err: &net.DNSError{Err: "no such host", Name: "gw"}}}
		_, err := New("x", "http://gw", WithHTTPClient(client)).ChatCompletion(context.Background(), chatRequest())
		var dnsErr *net.DNSError
		if !errors.Is(err, ir.ErrProviderUnavailable) || !errors.As(err, &dnsErr) {
			t.Errorf("err = %v, want both the sentinel and *net.DNSError", err)
		}
	})

	t.Run("deadline", func(t *testing.T) {
		release := make(chan struct{})
		srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { <-release }))
		defer srv.Close()
		defer close(release)
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
		defer cancel()
		_, err := New("x", srv.URL).ChatCompletion(ctx, chatRequest())
		if !errors.Is(err, ir.ErrProviderUnavailable) || !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("err = %v, want both the sentinel and DeadlineExceeded", err)
		}
	})

	t.Run("canceled", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := New("x", "http://gw").ChatCompletion(ctx, chatRequest())
		if !errors.Is(err, context.Canceled) {
			t.Errorf("err = %v, want Canceled", err)
		}
	})

	t.Run("stream shares the path", func(t *testing.T) {
		srv := httptest.NewServer(http.NotFoundHandler())
		srv.Close()
		_, err := New("x", srv.URL).ChatCompletionStream(context.Background(), chatRequest())
		if !errors.Is(err, syscall.ECONNREFUSED) {
			t.Errorf("err = %v, want ECONNREFUSED", err)
		}
	})
}

// Callers store this text; wrapping with %w must not change a byte of it.
func TestErrorTextIsUnchanged(t *testing.T) {
	cause := errors.New("dial tcp: boom")
	client := &http.Client{Transport: failingTransport{err: cause}}
	_, err := New("x", "http://gw", WithHTTPClient(client)).ChatCompletion(context.Background(), chatRequest())
	want := `inferrouter: provider unavailable: Post "http://gw/chat/completions": dial tcp: boom`
	if err == nil || err.Error() != want {
		t.Errorf("transport text = %q, want %q", err, want)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte("slot busy"))
	}))
	defer srv.Close()
	_, err = New("x", srv.URL).ChatCompletion(context.Background(), chatRequest())
	if want := "inferrouter: rate limited by provider: slot busy"; err == nil || err.Error() != want {
		t.Errorf("429 text = %q, want %q", err, want)
	}
}

func TestRateLimitCarriesRetryAfter(t *testing.T) {
	now := time.Now()
	tests := []struct {
		name   string
		header string
		want   time.Duration
	}{
		{"absent", "", 0},
		{"seconds", "7", 7 * time.Second},
		{"zero", "0", 0},
		{"negative", "-5", 0},
		{"over the cap", strconv.Itoa(int(time.Hour.Seconds())), maxRetryAfter},
		{"garbage", "soon", 0},
		{"past date", now.Add(-time.Hour).UTC().Format(http.TimeFormat), 0},
		{"far date", now.Add(time.Hour).UTC().Format(http.TimeFormat), maxRetryAfter},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if tt.header != "" {
					w.Header().Set("Retry-After", tt.header)
				}
				w.WriteHeader(http.StatusTooManyRequests)
			}))
			defer srv.Close()

			_, err := New("x", srv.URL).ChatCompletion(context.Background(), chatRequest())
			var limited *ir.RateLimitedError
			if !errors.As(err, &limited) {
				t.Fatalf("err = %v, want *RateLimitedError", err)
			}
			if !errors.Is(err, ir.ErrRateLimited) {
				t.Errorf("err = %v, want Is ErrRateLimited", err)
			}
			if limited.RetryAfter != tt.want {
				t.Errorf("RetryAfter = %v, want %v", limited.RetryAfter, tt.want)
			}
		})
	}

	near := now.Add(30 * time.Second)
	if got := retryAfter(near.UTC().Format(http.TimeFormat), now); got <= 0 || got > 30*time.Second {
		t.Errorf("near date = %v", got)
	}
}

func TestNotFoundIsModelNotFoundAndWalkContinues(t *testing.T) {
	gone := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "no such model", http.StatusNotFound)
	}))
	defer gone.Close()
	var served bool
	alive := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		served = true
		_, _ = w.Write([]byte(`{"id":"1","model":"m","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer alive.Close()

	_, err := New("a", gone.URL).ChatCompletion(context.Background(), chatRequest())
	if !errors.Is(err, ir.ErrModelNotFound) {
		t.Fatalf("err = %v, want ErrModelNotFound", err)
	}
	if errors.Is(err, ir.ErrProviderUnavailable) {
		t.Errorf("err = %v is still ErrProviderUnavailable", err)
	}
	if ir.IsFatal(err) || !ir.IsRetryable(err) {
		t.Errorf("IsFatal=%v IsRetryable=%v, want a walk that goes on", ir.IsFatal(err), ir.IsRetryable(err))
	}

	router, err := ir.NewRouter(ir.Config{
		AllowPaid: true,
		Models: []ir.ModelMapping{{Alias: "cheap", Models: []ir.ModelRef{
			{Provider: "a", Model: "m"}, {Provider: "b", Model: "m"},
		}}},
		Accounts: []ir.AccountConfig{
			{Provider: "a", ID: "a", Auth: ir.Auth{APIKey: "k"}, QuotaUnit: ir.QuotaDollars, PaidEnabled: true},
			{Provider: "b", ID: "b", Auth: ir.Auth{APIKey: "k"}, QuotaUnit: ir.QuotaDollars, PaidEnabled: true},
		},
	}, []ir.Provider{New("a", gone.URL), New("b", alive.URL)}, ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := router.ChatCompletion(context.Background(), ir.ChatRequest{
		Model:    "cheap",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("walk stopped: %v", err)
	}
	if !served || resp.Routing.AccountID != "b" {
		t.Errorf("served=%v routing=%+v", served, resp.Routing)
	}
}
