package openaicompat

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	ir "github.com/ineyio/inferrouter"
)

// callerEndRouter puts the real adapter on a gateway whose first call refuses,
// so the count starts at one and a false success shows by wiping it. Later calls
// go to next.
func callerEndRouter(t *testing.T, budget time.Duration, next http.HandlerFunc) (*ir.Router, func() int) {
	t.Helper()
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		next(w, r)
	}))
	t.Cleanup(srv.Close)

	r, err := ir.NewRouter(ir.Config{
		AllowPaid:      true,
		DefaultModel:   "ladder",
		AttemptTimeout: budget,
		Models:         []ir.ModelMapping{{Alias: "ladder", Models: []ir.ModelRef{{Provider: "gw", Model: "m"}}}},
		Accounts:       []ir.AccountConfig{{Provider: "gw", ID: "gw", QuotaUnit: ir.QuotaTokens, PaidEnabled: true}},
	}, []ir.Provider{New("gw", srv.URL)})
	if err != nil {
		t.Fatal(err)
	}

	req := ir.ChatRequest{Messages: []ir.Message{{Role: "user", Content: "hi"}}}
	if _, err := r.ChatCompletion(context.Background(), req); err == nil {
		t.Fatal("the first call must refuse")
	}
	failures := func() int { return r.AccountsSnapshot(context.Background())[0].FailuresInWindow }
	if failures() != 1 {
		t.Fatalf("the refusal must count, got %d", failures())
	}
	return r, failures
}

// hang reads the body first: until it is read the server does not watch the
// connection, r.Context() never ends, and srv.Close waits for ever.
func hang(_ http.ResponseWriter, r *http.Request) {
	_, _ = io.Copy(io.Discard, r.Body)
	<-r.Context().Done()
}

// halfAnswer sends one chunk of a stream and hangs.
func halfAnswer(w http.ResponseWriter, r *http.Request) {
	_, _ = io.Copy(io.Discard, r.Body)
	w.Header().Set("Content-Type", "text/event-stream")
	fmt.Fprint(w, `data: {"id":"r","model":"m","choices":[{"index":0,"delta":{"content":"tok"}}]}`+"\n\n")
	w.(http.Flusher).Flush()
	<-r.Context().Done()
}

// CE-I3 and CE-I4 on the real adapter: the test double of the root package only
// repeats what this code is assumed to do with a context, and it does not do the
// obvious thing: its stream reports a broken read as EOF.
func TestCallerEnd_RealAdapter(t *testing.T) {
	req := ir.ChatRequest{Messages: []ir.Message{{Role: "user", Content: "hi"}}}

	t.Run("attempt timeout counts", func(t *testing.T) {
		r, failures := callerEndRouter(t, 20*time.Millisecond, hang)
		if _, err := r.ChatCompletion(context.Background(), req); err == nil {
			t.Fatal("expected a timeout")
		}
		if failures() != 2 {
			t.Fatalf("failures = %d, want 2", failures())
		}
	})

	t.Run("cancel does not count", func(t *testing.T) {
		r, failures := callerEndRouter(t, time.Hour, hang)
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
		defer cancel()
		ctx, stop := context.WithCancel(ctx)
		time.AfterFunc(5*time.Millisecond, stop)
		if _, err := r.ChatCompletion(ctx, req); err == nil {
			t.Fatal("expected the caller's end")
		}
		if failures() != 1 {
			t.Fatalf("failures = %d, want 1", failures())
		}
	})

	t.Run("stream watchdog counts", func(t *testing.T) {
		r, failures := callerEndRouter(t, 20*time.Millisecond, hang)
		if _, err := r.ChatCompletionStream(context.Background(), req); err == nil {
			t.Fatal("expected the watchdog")
		}
		if failures() != 2 {
			t.Fatalf("failures = %d, want 2", failures())
		}
	})

	t.Run("visitor leaving a stream is neither a failure nor a success", func(t *testing.T) {
		r, failures := callerEndRouter(t, time.Hour, halfAnswer)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		s, err := r.ChatCompletionStream(ctx, req)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := s.Next(); err != nil {
			t.Fatal(err)
		}
		cancel()
		for {
			if _, err := s.Next(); err != nil {
				break
			}
		}
		_ = s.Close()
		if failures() != 1 {
			t.Fatalf("failures = %d, want 1: the broken read must not read as a finished answer", failures())
		}
	})
}
