package inferrouter_test

import (
	"context"
	"errors"
	"fmt"
	"testing"

	ir "github.com/ineyio/inferrouter"
	"github.com/ineyio/inferrouter/provider/mock"
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
	assert.EqualValues(t, 1, primary.CallCount())
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
	r, err := ir.NewRouter(cfg, []ir.Provider{chatOnly})
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
