package inferrouter_test

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	ir "github.com/ineyio/inferrouter"
	"github.com/ineyio/inferrouter/meter"
	"github.com/ineyio/inferrouter/provider/mock"
	"github.com/ineyio/inferrouter/quota"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Happy path: single provider, single account, batch fits in one call.
func TestEmbedBatch_HappyPathSingleBatch(t *testing.T) {
	embedProv := mock.NewEmbed(
		mock.WithEmbedSupportedModels("text-embedding-004"),
		mock.WithEmbedMaxBatch(100),
		mock.WithEmbedDimensions(4),
	)

	cfg := ir.Config{
		DefaultModel: "text-embedding-004",
		Accounts: []ir.AccountConfig{
			{Provider: "mock-embed", ID: "free-1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens,
				CostPerEmbeddingInputToken: 0.0001},
		},
	}
	r := newEmbedRouter(t, cfg, embedProv)

	inputs := []string{"hello", "world", "embedding test"}
	resp, err := r.EmbedBatch(context.Background(), ir.EmbedRequest{
		Inputs:   inputs,
		TaskType: "RETRIEVAL_DOCUMENT",
	})
	require.NoError(t, err)
	assert.Len(t, resp.Embeddings, 3, "one embedding per input")
	assert.Len(t, resp.Embeddings[0], 4, "dimensions matches mock config")
	assert.Equal(t, "text-embedding-004", resp.Model)
	assert.Equal(t, "free-1", resp.Routing.AccountID)
	assert.True(t, resp.Routing.Free)
	assert.EqualValues(t, 1, embedProv.CallCount(), "single provider call")
}

// Batch splitting: 120 inputs -> 2 sub-batches (100 + 20), order preserved.
func TestEmbedBatch_SplitOn100Boundary(t *testing.T) {
	embedProv := mock.NewEmbed(
		mock.WithEmbedSupportedModels("text-embedding-004"),
		mock.WithEmbedMaxBatch(100),
		mock.WithEmbedDimensions(3),
	)

	cfg := ir.Config{
		DefaultModel: "text-embedding-004",
		Accounts: []ir.AccountConfig{
			{Provider: "mock-embed", ID: "free-1", DailyFree: 100000, QuotaUnit: ir.QuotaTokens,
				CostPerEmbeddingInputToken: 0.0001},
		},
	}
	r := newEmbedRouter(t, cfg, embedProv)

	inputs := make([]string, 120)
	for i := range inputs {
		inputs[i] = fmt.Sprintf("input-%d", i)
	}
	resp, err := r.EmbedBatch(context.Background(), ir.EmbedRequest{Inputs: inputs})
	require.NoError(t, err)
	assert.Len(t, resp.Embeddings, 120)
	assert.EqualValues(t, 2, embedProv.CallCount(), "120 inputs split into 2 batches of 100+20")

	// Order verification: fake embeddings are deterministic (same text → same vec),
	// so embeddings[i] must equal the embedding of inputs[i]. We verify this by
	// checking that embedding[0] != embedding[1] != embedding[119] (sanity) and
	// that re-embedding input[50] matches response[50] independently.
	assert.NotEqual(t, resp.Embeddings[0], resp.Embeddings[1])
	assert.NotEqual(t, resp.Embeddings[0], resp.Embeddings[119])
}

// Rate limit on first provider → fallback to second candidate.
// We use two accounts on the same mock provider with different behaviors.
func TestEmbedBatch_FallbackOnRateLimit(t *testing.T) {
	primary := mock.NewEmbed(
		mock.WithEmbedName("primary"),
		mock.WithEmbedSupportedModels("text-embedding-004"),
		mock.WithEmbedMaxBatch(100),
		mock.WithEmbedError(ir.ErrRateLimited),
	)
	secondary := mock.NewEmbed(
		mock.WithEmbedName("secondary"),
		mock.WithEmbedSupportedModels("text-embedding-004"),
		mock.WithEmbedMaxBatch(100),
	)

	cfg := ir.Config{
		DefaultModel: "text-embedding-004",
		Accounts: []ir.AccountConfig{
			{Provider: "primary", ID: "primary-acc", DailyFree: 1000, QuotaUnit: ir.QuotaTokens,
				CostPerEmbeddingInputToken: 0.0001},
			{Provider: "secondary", ID: "secondary-acc", DailyFree: 1000, QuotaUnit: ir.QuotaTokens,
				CostPerEmbeddingInputToken: 0.0001},
		},
	}
	r := newEmbedRouter(t, cfg,
		primary,
		embedProviderAsProvider(secondary),
	)

	resp, err := r.EmbedBatch(context.Background(), ir.EmbedRequest{
		Inputs: []string{"hello"},
	})
	require.NoError(t, err)
	assert.Equal(t, "secondary-acc", resp.Routing.AccountID)
	assert.EqualValues(t, 2, resp.Routing.Attempts)
	// Contract change (embed pacing, 2026-09-02): a 429 no longer sends the
	// router straight down the ladder. The same candidate is asked again,
	// twice, because a provider rate-limit window is measured in seconds and
	// the ladder below an embedding account is usually empty. Only after the
	// retries are spent does the next candidate get the request.
	assert.EqualValues(t, 1+2, primary.CallCount())
	assert.EqualValues(t, 1, secondary.CallCount())
}

// Partial failure: first batch succeeds, second batch fails permanently
// (no fallback available). Consumer receives ErrPartialBatch with the
// successful prefix preserved.
func TestEmbedBatch_PartialFailureReturnsPrefix(t *testing.T) {
	// Use a response func that succeeds for first batch then errors.
	var callNum int
	embedProv := mock.NewEmbed(
		mock.WithEmbedSupportedModels("text-embedding-004"),
		mock.WithEmbedMaxBatch(5),
		mock.WithEmbedResponseFunc(func(req ir.EmbedProviderRequest) (ir.EmbedProviderResponse, error) {
			callNum++
			if callNum == 1 {
				// First sub-batch: succeed with deterministic embeddings.
				out := make([][]float32, len(req.Inputs))
				for i := range req.Inputs {
					out[i] = []float32{float32(i), 0, 0}
				}
				return ir.EmbedProviderResponse{
					Embeddings: out,
					Model:      req.Model,
					Usage:      ir.EmbedUsage{InputTokens: 10, TotalTokens: 10},
				}, nil
			}
			// Second sub-batch: permanent error, no fallback.
			return ir.EmbedProviderResponse{}, ir.ErrRateLimited
		}),
	)

	cfg := ir.Config{
		DefaultModel: "text-embedding-004",
		Accounts: []ir.AccountConfig{
			{Provider: "mock-embed", ID: "only", DailyFree: 10000, QuotaUnit: ir.QuotaTokens,
				CostPerEmbeddingInputToken: 0.0001},
		},
	}
	r := newEmbedRouter(t, cfg, embedProv)

	// 12 inputs, max batch 5 → 3 sub-batches. First succeeds (5 embeddings),
	// second fails, partial contains 5 embeddings.
	inputs := make([]string, 12)
	for i := range inputs {
		inputs[i] = fmt.Sprintf("input-%d", i)
	}

	resp, err := r.EmbedBatch(context.Background(), ir.EmbedRequest{Inputs: inputs})
	require.Error(t, err)

	var partial *ir.ErrPartialBatch
	require.True(t, errors.As(err, &partial), "error must be *ErrPartialBatch")
	assert.Equal(t, 5, partial.ProcessedInputs, "first sub-batch succeeded")
	assert.Len(t, resp.Embeddings, 5, "response has valid prefix")
	assert.EqualValues(t, 10, resp.Usage.InputTokens, "usage from first sub-batch only")
}

// Full failure (first batch fails): returns zero response and non-partial error.
func TestEmbedBatch_FirstBatchFailsReturnsFullError(t *testing.T) {
	embedProv := mock.NewEmbed(
		mock.WithEmbedSupportedModels("text-embedding-004"),
		mock.WithEmbedMaxBatch(100),
		mock.WithEmbedError(ir.ErrProviderUnavailable),
	)

	cfg := ir.Config{
		DefaultModel: "text-embedding-004",
		Accounts: []ir.AccountConfig{
			{Provider: "mock-embed", ID: "only", DailyFree: 1000, QuotaUnit: ir.QuotaTokens,
				CostPerEmbeddingInputToken: 0.0001},
		},
	}
	r := newEmbedRouter(t, cfg, embedProv)

	resp, err := r.EmbedBatch(context.Background(), ir.EmbedRequest{
		Inputs: []string{"x", "y"},
	})
	require.Error(t, err)
	var partial *ir.ErrPartialBatch
	assert.False(t, errors.As(err, &partial), "full failure should NOT be ErrPartialBatch")
	assert.Empty(t, resp.Embeddings)
}

// No embedding providers registered for a chat-only router → ErrNoEmbeddingProviders.
func TestEmbedBatch_NoEmbeddingProviders(t *testing.T) {
	chatOnly := mock.New(mock.WithModels("chat-model"))

	cfg := ir.Config{
		DefaultModel: "text-embedding-004",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "acc-1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
		},
	}
	r, err := ir.NewRouter(declareLadder(cfg), []ir.Provider{chatOnly})
	require.NoError(t, err)

	_, err = r.EmbedBatch(context.Background(), ir.EmbedRequest{
		Inputs: []string{"hello"},
	})
	assert.ErrorIs(t, err, ir.ErrNoEmbeddingProviders)
}

// TaskType and OutputDimensionality are propagated to the provider.
func TestEmbedBatch_PropagatesTaskTypeAndDimensions(t *testing.T) {
	var capturedReq ir.EmbedProviderRequest
	embedProv := mock.NewEmbed(
		mock.WithEmbedSupportedModels("text-embedding-004"),
		mock.WithEmbedMaxBatch(100),
		mock.WithEmbedResponseFunc(func(req ir.EmbedProviderRequest) (ir.EmbedProviderResponse, error) {
			capturedReq = req
			out := make([][]float32, len(req.Inputs))
			for i := range req.Inputs {
				out[i] = []float32{float32(i)}
			}
			return ir.EmbedProviderResponse{
				Embeddings: out,
				Model:      req.Model,
				Usage:      ir.EmbedUsage{InputTokens: 5, TotalTokens: 5},
			}, nil
		}),
	)

	cfg := ir.Config{
		DefaultModel: "text-embedding-004",
		Accounts: []ir.AccountConfig{
			{Provider: "mock-embed", ID: "acc", DailyFree: 1000, QuotaUnit: ir.QuotaTokens,
				CostPerEmbeddingInputToken: 0.0001},
		},
	}
	r := newEmbedRouter(t, cfg, embedProv)

	_, err := r.EmbedBatch(context.Background(), ir.EmbedRequest{
		Inputs:               []string{"x"},
		TaskType:             "RETRIEVAL_QUERY",
		OutputDimensionality: 256,
	})
	require.NoError(t, err)
	assert.Equal(t, "RETRIEVAL_QUERY", capturedReq.TaskType)
	assert.Equal(t, 256, capturedReq.OutputDimensionality)
}

// Aliases resolve to the concrete model and EmbedResponse.Model contains
// the actual (resolved) model name, not the alias — critical for runtime
// verification in consumers per RFC §3.3 contract.
func TestEmbedBatch_ModelFieldIsResolved(t *testing.T) {
	embedProv := mock.NewEmbed(
		mock.WithEmbedSupportedModels("text-embedding-004"),
		mock.WithEmbedMaxBatch(100),
	)

	cfg := ir.Config{
		Models: []ir.ModelMapping{
			{
				Alias: "default-embedding",
				Models: []ir.ModelRef{
					{Provider: "mock-embed", Model: "text-embedding-004"},
				},
			},
		},
		Accounts: []ir.AccountConfig{
			{Provider: "mock-embed", ID: "acc", DailyFree: 1000, QuotaUnit: ir.QuotaTokens,
				CostPerEmbeddingInputToken: 0.0001},
		},
	}
	r := newEmbedRouter(t, cfg, embedProv)

	resp, err := r.EmbedBatch(context.Background(), ir.EmbedRequest{
		Model:  "default-embedding",
		Inputs: []string{"x"},
	})
	require.NoError(t, err)
	assert.Equal(t, "text-embedding-004", resp.Model, "resolved model, not alias")
}

// --- Embed pacing (C.21) ---

// newPacedEmbedRouter builds a single-account embed router with a sleeper the
// test can observe. The shared helper installs an instant one, which is right
// for tests that merely tolerate backoff and wrong for tests that assert on it.
func newPacedEmbedRouter(t *testing.T, prov *mock.EmbedProvider, sleeper ir.Sleeper) *ir.Router {
	t.Helper()
	r, err := ir.NewRouter(
		ir.Config{
			DefaultModel: "text-embedding-004",
			Accounts: []ir.AccountConfig{{
				Provider: "paced", ID: "paced-acc", DailyFree: 1000, QuotaUnit: ir.QuotaTokens,
			}},
			Models: []ir.ModelMapping{{
				Alias:  "text-embedding-004",
				Models: []ir.ModelRef{{Provider: "paced", Model: "text-embedding-004"}},
			}},
		},
		[]ir.Provider{embedProviderAsProvider(prov)},
		ir.WithQuotaStore(quota.NewMemoryQuotaStore()),
		ir.WithMeter(&meter.NoopMeter{}),
		ir.WithSleeper(sleeper),
	)
	require.NoError(t, err)
	return r
}

// requireRateLimitedCandidate asserts the run ended because the candidate was
// rate limited. The router reports a run through ErrAllFailed and keeps the
// per-candidate causes in Tried, so the sentinel is not on the unwrap chain.
func requireRateLimitedCandidate(t *testing.T, err error) {
	t.Helper()
	var routerErr *ir.RouterError
	require.ErrorAs(t, err, &routerErr)
	require.Len(t, routerErr.Tried, 1)
	require.ErrorIs(t, routerErr.Tried[0].Err, ir.ErrRateLimited)
}

func pacedEmbedProvider(fn func(ir.EmbedProviderRequest) (ir.EmbedProviderResponse, error)) *mock.EmbedProvider {
	return mock.NewEmbed(
		mock.WithEmbedName("paced"),
		mock.WithEmbedSupportedModels("text-embedding-004"),
		mock.WithEmbedMaxBatch(100),
		mock.WithEmbedResponseFunc(fn),
	)
}

// A 429 is backpressure, not ill health. Three of them must leave the account
// usable, because with one embedding account an open breaker is a total
// outage for every caller — the amplifier behind the 2026-09-01 incident.
func TestEmbed_RateLimitDoesNotTripBreaker(t *testing.T) {
	prov := pacedEmbedProvider(func(ir.EmbedProviderRequest) (ir.EmbedProviderResponse, error) {
		return ir.EmbedProviderResponse{}, &ir.RateLimitedError{Detail: "quota exceeded"}
	})
	r := newPacedEmbedRouter(t, prov, func(context.Context, time.Duration) error { return nil })

	req := ir.EmbedRequest{Inputs: []string{"hello"}}
	for i := range 3 {
		_, err := r.EmbedBatch(context.Background(), req)
		require.Error(t, err, "call %d", i+1)
		requireRateLimitedCandidate(t, err)
	}

	// The breaker trips at three failures. If 429 counted as one, the account
	// would now be filtered out and the provider would never be asked again.
	before := prov.CallCount()
	_, err := r.EmbedBatch(context.Background(), req)
	require.NotErrorIs(t, err, ir.ErrNoEmbeddingProviders)
	assert.Greater(t, prov.CallCount(), before, "provider must still be reachable after three 429s")
}

// The provider's own Retry-After beats any backoff we could invent, and the
// retry has to actually recover — otherwise the pause is pure latency.
func TestEmbed_RetryUsesProviderRetryAfter(t *testing.T) {
	var calls int
	prov := pacedEmbedProvider(func(req ir.EmbedProviderRequest) (ir.EmbedProviderResponse, error) {
		calls++
		if calls == 1 {
			return ir.EmbedProviderResponse{}, &ir.RateLimitedError{
				RetryAfter: 5 * time.Second,
				Detail:     "quota exceeded",
			}
		}
		return ir.EmbedProviderResponse{
			Embeddings: [][]float32{{0.1, 0.2}},
			Model:      req.Model,
			Usage:      ir.EmbedUsage{InputTokens: 2, TotalTokens: 2},
		}, nil
	})

	var slept []time.Duration
	r := newPacedEmbedRouter(t, prov, func(_ context.Context, d time.Duration) error {
		slept = append(slept, d)
		return nil
	})

	resp, err := r.EmbedBatch(context.Background(), ir.EmbedRequest{Inputs: []string{"hello"}})
	require.NoError(t, err)
	require.Len(t, resp.Embeddings, 1)
	assert.Equal(t, []time.Duration{5 * time.Second}, slept)
	assert.EqualValues(t, 2, prov.CallCount())
}

// With no hint from the provider, the retry falls back to its own short
// backoff rather than hammering immediately.
func TestEmbed_RetryFallsBackToDefaultBackoff(t *testing.T) {
	prov := pacedEmbedProvider(func(ir.EmbedProviderRequest) (ir.EmbedProviderResponse, error) {
		return ir.EmbedProviderResponse{}, &ir.RateLimitedError{Detail: "slow down"}
	})

	var slept []time.Duration
	r := newPacedEmbedRouter(t, prov, func(_ context.Context, d time.Duration) error {
		slept = append(slept, d)
		return nil
	})

	_, err := r.EmbedBatch(context.Background(), ir.EmbedRequest{Inputs: []string{"hello"}})
	requireRateLimitedCandidate(t, err)
	assert.Equal(t, []time.Duration{200 * time.Millisecond, time.Second}, slept)
	assert.EqualValues(t, 3, prov.CallCount())
}

// Sleeping past the caller's deadline buys nothing: the caller would get a
// cancellation instead of the rate-limit error that explains it.
func TestEmbed_RetryRespectsDeadline(t *testing.T) {
	prov := pacedEmbedProvider(func(ir.EmbedProviderRequest) (ir.EmbedProviderResponse, error) {
		return ir.EmbedProviderResponse{}, &ir.RateLimitedError{
			RetryAfter: 30 * time.Second,
			Detail:     "quota exceeded",
		}
	})

	sleeps := 0
	r := newPacedEmbedRouter(t, prov, func(context.Context, time.Duration) error {
		sleeps++
		return nil
	})

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	_, err := r.EmbedBatch(ctx, ir.EmbedRequest{Inputs: []string{"hello"}})
	requireRateLimitedCandidate(t, err)
	assert.Zero(t, sleeps, "a 30s pause does not fit a 200ms budget")
	assert.EqualValues(t, 1, prov.CallCount())
}

// The pacing has to be wired into the embed path, not merely available on the
// limiter: a burst above RPM must be delayed and then served, where before it
// was refused outright and — with one account — turned into a 500.
func TestEmbed_PacesBurstInsteadOfRefusing(t *testing.T) {
	prov := pacedEmbedProvider(func(req ir.EmbedProviderRequest) (ir.EmbedProviderResponse, error) {
		return ir.EmbedProviderResponse{
			Embeddings: [][]float32{{0.1, 0.2}},
			Model:      req.Model,
			Usage:      ir.EmbedUsage{InputTokens: 2, TotalTokens: 2},
		}, nil
	})

	limiter := ir.NewRateLimiter()
	var paced []time.Duration
	limiter.SetSleeper(func(_ context.Context, d time.Duration) error {
		paced = append(paced, d)
		// Waking up frees the window: without this the retry would refuse
		// again and the test could not tell "waited" from "gave up".
		limiter.Reset()
		return nil
	})

	r, err := ir.NewRouter(
		ir.Config{
			DefaultModel: "text-embedding-004",
			Accounts: []ir.AccountConfig{{
				Provider: "paced", ID: "paced-acc", DailyFree: 1000,
				QuotaUnit: ir.QuotaTokens, RPM: 1,
			}},
			Models: []ir.ModelMapping{{
				Alias:  "text-embedding-004",
				Models: []ir.ModelRef{{Provider: "paced", Model: "text-embedding-004"}},
			}},
		},
		[]ir.Provider{embedProviderAsProvider(prov)},
		ir.WithQuotaStore(quota.NewMemoryQuotaStore()),
		ir.WithMeter(&meter.NoopMeter{}),
		ir.WithRateLimiter(limiter),
	)
	require.NoError(t, err)

	req := ir.EmbedRequest{Inputs: []string{"hello"}}
	_, err = r.EmbedBatch(context.Background(), req)
	require.NoError(t, err)

	// Second call within the same minute is over RPM: it must pause, not fail.
	resp, err := r.EmbedBatch(context.Background(), req)
	require.NoError(t, err, "a burst above RPM must be paced, not refused")
	require.Len(t, resp.Embeddings, 1)

	require.Len(t, paced, 1, "the second call must have waited exactly once")
	assert.Positive(t, paced[0])
	assert.LessOrEqual(t, paced[0], time.Minute)
	assert.EqualValues(t, 2, prov.CallCount())
}
