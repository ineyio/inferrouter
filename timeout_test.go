package inferrouter_test

import (
	"context"
	"errors"
	"io"
	"sync"
	"testing"
	"time"

	ir "github.com/ineyio/inferrouter"
	"github.com/ineyio/inferrouter/quota"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// hangingProvider accepts the request and never answers — it returns only when
// the attempt context is cancelled.
type hangingProvider struct {
	name string
	log  *attemptLog
}

func (p *hangingProvider) Name() string              { return p.name }
func (p *hangingProvider) SupportsModel(string) bool { return true }
func (p *hangingProvider) SupportsMultimodal() bool  { return false }

func (p *hangingProvider) ChatCompletion(ctx context.Context, _ ir.ProviderRequest) (ir.ProviderResponse, error) {
	p.log.record(p.name)
	<-ctx.Done()
	return ir.ProviderResponse{}, ctx.Err()
}

func (p *hangingProvider) ChatCompletionStream(ctx context.Context, _ ir.ProviderRequest) (ir.ProviderStream, error) {
	p.log.record(p.name)
	<-ctx.Done()
	return nil, ctx.Err()
}

func hangingStep(log *attemptLog, name string) *hangingProvider {
	return &hangingProvider{name: name, log: log}
}

// L4 (R5): a step that hangs burns its own budget, not the caller's. Without a
// per-attempt bound the first hung step consumes the whole deadline and the
// remaining steps are never tried — the ladder collapses to one rung.
func TestPerAttemptTimeout(t *testing.T) {
	log := &attemptLog{}
	cfg := ladderConfig("hangs", "answers")
	cfg.AttemptTimeout = 50 * time.Millisecond

	r, err := ir.NewRouter(cfg,
		[]ir.Provider{hangingStep(log, "hangs"), servingStep(log, "answers")},
		ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
	require.NoError(t, err)

	// The caller's budget is enough for two attempts but not for one hang.
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	start := time.Now()
	resp, err := r.ChatCompletion(ctx, ir.ChatRequest{
		Model:    "ladder",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})
	elapsed := time.Since(start)

	require.NoError(t, err)
	assert.Equal(t, []string{"hangs", "answers"}, log.snapshot(),
		"the hung step must be abandoned and the next one tried")
	assert.Equal(t, "answers", resp.Routing.AccountID[:7])
	assert.Less(t, elapsed, time.Second, "the walk must not wait out the caller's deadline")
}

// Per-account override wins over the global setting.
func TestPerAttemptTimeout_AccountOverride(t *testing.T) {
	log := &attemptLog{}
	cfg := ladderConfig("hangs", "answers")
	cfg.AttemptTimeout = 5 * time.Second // global: far too generous
	cfg.Accounts[0].AttemptTimeout = 50 * time.Millisecond

	r, err := ir.NewRouter(cfg,
		[]ir.Provider{hangingStep(log, "hangs"), servingStep(log, "answers")},
		ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	start := time.Now()
	_, err = r.ChatCompletion(ctx, ir.ChatRequest{
		Model:    "ladder",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})

	require.NoError(t, err)
	assert.Less(t, time.Since(start), time.Second)
	assert.Equal(t, []string{"hangs", "answers"}, log.snapshot())
}

// Zero means "no separate budget": the attempt may use the caller's deadline,
// which is the behaviour every caller had before this setting existed.
func TestPerAttemptTimeout_ZeroUsesCallerBudget(t *testing.T) {
	log := &attemptLog{}
	cfg := ladderConfig("hangs", "answers") // AttemptTimeout unset

	r, err := ir.NewRouter(cfg,
		[]ir.Provider{hangingStep(log, "hangs"), servingStep(log, "answers")},
		ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	_, err = r.ChatCompletion(ctx, ir.ChatRequest{
		Model:    "ladder",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})

	require.Error(t, err, "the caller's deadline is the only bound, and the first step ate it")
	assert.True(t, errors.Is(err, context.DeadlineExceeded), "got %v", err)
	assert.Equal(t, []string{"hangs"}, log.snapshot(),
		"with the caller's budget spent, the walk stops rather than retrying into a dead context")
}

// blockingStream opens immediately and then emits chunks slowly, outliving any
// sane per-attempt budget.
type blockingStream struct {
	chunks int
	mu     sync.Mutex
	ctx    context.Context
}

func (s *blockingStream) Next() (ir.StreamChunk, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ctx.Err() != nil {
		return ir.StreamChunk{}, s.ctx.Err()
	}
	if s.chunks >= 3 {
		return ir.StreamChunk{}, io.EOF
	}
	s.chunks++
	time.Sleep(30 * time.Millisecond)
	return ir.StreamChunk{Choices: []ir.StreamDelta{{Delta: ir.Delta{Content: "tok"}}}}, nil
}

func (s *blockingStream) Close() error { return nil }

type slowGeneratingProvider struct {
	name    string
	lastCtx context.Context
}

func (p *slowGeneratingProvider) Name() string              { return p.name }
func (p *slowGeneratingProvider) SupportsModel(string) bool { return true }
func (p *slowGeneratingProvider) SupportsMultimodal() bool  { return false }

func (p *slowGeneratingProvider) ChatCompletion(ctx context.Context, _ ir.ProviderRequest) (ir.ProviderResponse, error) {
	return ir.ProviderResponse{}, ir.ErrProviderUnavailable
}

func (p *slowGeneratingProvider) ChatCompletionStream(ctx context.Context, _ ir.ProviderRequest) (ir.ProviderStream, error) {
	p.lastCtx = ctx
	return &blockingStream{ctx: ctx}, nil
}

// R5 for streams: the budget covers opening the stream. Once open, a long
// generation is not interrupted — the deadline that guards the attempt must
// not become a deadline on the answer.
func TestPerAttemptTimeout_StreamSurvivesAfterOpen(t *testing.T) {
	prov := &slowGeneratingProvider{name: "streamer"}
	cfg := ladderConfig("streamer")
	cfg.AttemptTimeout = 40 * time.Millisecond // shorter than the generation

	r, err := ir.NewRouter(cfg, []ir.Provider{prov},
		ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
	require.NoError(t, err)

	stream, err := r.ChatCompletionStream(context.Background(), ir.ChatRequest{
		Model:    "ladder",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})
	require.NoError(t, err)

	var got int
	for {
		chunk, err := stream.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		require.NoError(t, err, "generation must outlive the per-attempt budget")
		if len(chunk.Choices) > 0 {
			got++
		}
	}
	require.NoError(t, stream.Close())
	assert.Equal(t, 3, got, "every chunk arrives despite the attempt budget being long past")
}

// Closing the stream releases the attempt context; leaving it live would leak
// a cancel func per stream.
func TestStream_CloseReleasesAttemptContext(t *testing.T) {
	prov := &slowGeneratingProvider{name: "streamer"}
	cfg := ladderConfig("streamer")

	r, err := ir.NewRouter(cfg, []ir.Provider{prov},
		ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
	require.NoError(t, err)

	stream, err := r.ChatCompletionStream(context.Background(), ir.ChatRequest{
		Model:    "ladder",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})
	require.NoError(t, err)

	require.NoError(t, prov.lastCtx.Err(), "context is live while the stream is open")
	require.NoError(t, stream.Close())
	assert.Error(t, prov.lastCtx.Err(), "Close must release it")
}

// L7 (R6): the stream reports which step served it. Previously ChatCompletionStream
// returned no RoutingInfo at all and callers guessed from chunk metadata.
func TestStream_ReturnsRoutingInfo(t *testing.T) {
	log := &attemptLog{}
	failing := failingStep(log, "step-one")
	prov := &slowGeneratingProvider{name: "step-two"}

	cfg := ladderConfig("step-one", "step-two")
	r, err := ir.NewRouter(cfg, []ir.Provider{failing, prov},
		ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
	require.NoError(t, err)

	stream, err := r.ChatCompletionStream(context.Background(), ir.ChatRequest{
		Model:    "ladder",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})
	require.NoError(t, err)
	defer stream.Close()

	routing := stream.Routing()
	assert.Equal(t, "step-two", routing.Provider)
	assert.Equal(t, "step-two-acc", routing.AccountID)
	assert.Equal(t, "ladder-model", routing.Model)
	assert.Equal(t, 2, routing.Attempts, "the first step failed before this one answered")
	assert.True(t, routing.Free, "the last step of ladderConfig is the free one")
}
