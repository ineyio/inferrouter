package inferrouter

import (
	"context"
	"errors"
	"io"
	"sync"
	"testing"
)

// blockingProvider is a Provider double whose ChatCompletion signals start
// and blocks until released — used to observe in-flight counts mid-request.
type blockingProvider struct {
	name    string
	started chan struct{} // receives one value when ChatCompletion begins
	release chan struct{} // ChatCompletion returns when closed
	err     error
}

func (p *blockingProvider) Name() string              { return p.name }
func (p *blockingProvider) SupportsModel(string) bool { return true }
func (p *blockingProvider) SupportsMultimodal() bool  { return false }

func (p *blockingProvider) ChatCompletion(context.Context, ProviderRequest) (ProviderResponse, error) {
	if p.started != nil {
		p.started <- struct{}{}
	}
	if p.release != nil {
		<-p.release
	}
	if p.err != nil {
		return ProviderResponse{}, p.err
	}
	return ProviderResponse{Content: "ok", Usage: Usage{TotalTokens: 1}}, nil
}

func (p *blockingProvider) ChatCompletionStream(context.Context, ProviderRequest) (ProviderStream, error) {
	if p.err != nil {
		return nil, p.err
	}
	return &eofStream{}, nil
}

// eofStream is an immediately-exhausted ProviderStream.
type eofStream struct{}

func (s *eofStream) Next() (StreamChunk, error) { return StreamChunk{}, io.EOF }
func (s *eofStream) Close() error               { return nil }

func inflightTestConfig(providerName string) Config {
	return Config{
		DefaultModel: "m",
		AllowPaid:    true,
		Accounts: []AccountConfig{{
			Provider:           providerName,
			ID:                 "acc-1",
			QuotaUnit:          QuotaTokens,
			PaidEnabled:        true,
			CostPerInputToken:  0.001,
			CostPerOutputToken: 0.001,
		}},
	}
}

func inflightTestRequest() ChatRequest {
	return ChatRequest{Messages: []Message{{Role: "user", Content: "hi"}}}
}

func TestInflightTrackerIncDecGet(t *testing.T) {
	tr := NewInflightTracker()
	if got := tr.Get("a"); got != 0 {
		t.Fatalf("fresh counter = %d, want 0", got)
	}
	tr.Inc("a")
	tr.Inc("a")
	tr.Inc("b")
	if got := tr.Get("a"); got != 2 {
		t.Errorf("a = %d, want 2", got)
	}
	if got := tr.Get("b"); got != 1 {
		t.Errorf("b = %d, want 1", got)
	}
	tr.Dec("a")
	if got := tr.Get("a"); got != 1 {
		t.Errorf("a after Dec = %d, want 1", got)
	}
}

func TestInflightTrackerConcurrent(t *testing.T) {
	tr := NewInflightTracker()
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			tr.Inc("acc")
			tr.Dec("acc")
		}()
	}
	wg.Wait()
	if got := tr.Get("acc"); got != 0 {
		t.Errorf("after balanced Inc/Dec = %d, want 0", got)
	}
}

func TestRouterTracksInflightDuringChat(t *testing.T) {
	p := &blockingProvider{
		name:    "slow",
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
	r, err := NewRouter(inflightTestConfig("slow"), []Provider{p})
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}

	done := make(chan error, 1)
	go func() {
		_, err := r.ChatCompletion(context.Background(), inflightTestRequest())
		done <- err
	}()

	<-p.started
	if got := r.inflight.Get("acc-1"); got != 1 {
		t.Errorf("inflight during request = %d, want 1", got)
	}

	close(p.release)
	if err := <-done; err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}
	if got := r.inflight.Get("acc-1"); got != 0 {
		t.Errorf("inflight after success = %d, want 0", got)
	}
}

func TestRouterDecrementsInflightOnChatError(t *testing.T) {
	p := &blockingProvider{name: "broken", err: ErrProviderUnavailable}
	r, err := NewRouter(inflightTestConfig("broken"), []Provider{p})
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}

	_, err = r.ChatCompletion(context.Background(), inflightTestRequest())
	if err == nil {
		t.Fatal("expected error from broken provider")
	}
	if got := r.inflight.Get("acc-1"); got != 0 {
		t.Errorf("inflight after failure = %d, want 0", got)
	}
}

func TestRouterStreamDecrementsInflightOnCloseOnly(t *testing.T) {
	p := &blockingProvider{name: "streamy"}
	r, err := NewRouter(inflightTestConfig("streamy"), []Provider{p})
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}

	stream, err := r.ChatCompletionStream(context.Background(), inflightTestRequest())
	if err != nil {
		t.Fatalf("ChatCompletionStream: %v", err)
	}

	if got := r.inflight.Get("acc-1"); got != 1 {
		t.Errorf("inflight while stream open = %d, want 1", got)
	}

	if _, err := stream.Next(); !errors.Is(err, io.EOF) {
		t.Fatalf("Next = %v, want EOF", err)
	}
	if got := r.inflight.Get("acc-1"); got != 1 {
		t.Errorf("inflight after EOF but before Close = %d, want 1", got)
	}

	_ = stream.Close()
	if got := r.inflight.Get("acc-1"); got != 0 {
		t.Errorf("inflight after Close = %d, want 0", got)
	}

	// Double Close must not double-decrement.
	_ = stream.Close()
	if got := r.inflight.Get("acc-1"); got != 0 {
		t.Errorf("inflight after second Close = %d, want 0", got)
	}
}

func TestRouterDecrementsInflightOnStreamSetupError(t *testing.T) {
	p := &blockingProvider{name: "nostream", err: ErrProviderUnavailable}
	r, err := NewRouter(inflightTestConfig("nostream"), []Provider{p})
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}

	if _, err := r.ChatCompletionStream(context.Background(), inflightTestRequest()); err == nil {
		t.Fatal("expected stream setup error")
	}
	if got := r.inflight.Get("acc-1"); got != 0 {
		t.Errorf("inflight after stream setup failure = %d, want 0", got)
	}
}
