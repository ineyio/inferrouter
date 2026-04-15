package inferrouter_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	ir "github.com/ineyio/inferrouter"
	"github.com/ineyio/inferrouter/meter"
	"github.com/ineyio/inferrouter/provider/mock"
	"github.com/ineyio/inferrouter/quota"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Test: EstimateEmbedTokens returns rough character-based estimate.
func TestEstimateEmbedTokens_BasicBatch(t *testing.T) {
	inputs := []string{
		strings.Repeat("a", 400), // ~100 tokens
		strings.Repeat("b", 200), // ~50 tokens
	}
	got := ir.EstimateEmbedTokens(inputs)
	assert.Equal(t, int64(150), got)
}

func TestEstimateEmbedTokens_Empty(t *testing.T) {
	assert.Equal(t, int64(0), ir.EstimateEmbedTokens(nil))
	assert.Equal(t, int64(0), ir.EstimateEmbedTokens([]string{}))
	assert.Equal(t, int64(0), ir.EstimateEmbedTokens([]string{""}))
}

// Test: Embed rejects empty inputs.
func TestEmbed_EmptyInputs(t *testing.T) {
	embedProv := mock.NewEmbed(mock.WithEmbedSupportedModels("test-embed"))

	cfg := ir.Config{
		DefaultModel: "test-embed",
		Accounts: []ir.AccountConfig{
			{Provider: "mock-embed", ID: "free-1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens,
				CostPerEmbeddingInputToken: 0.0001},
		},
	}
	r := newEmbedRouter(t, cfg, embedProv)

	_, err := r.Embed(context.Background(), ir.EmbedRequest{})
	assert.ErrorIs(t, err, ir.ErrInvalidRequest)
}

// Test: Embed returns ErrBatchTooLarge when inputs exceed provider MaxBatchSize.
func TestEmbed_BatchTooLarge(t *testing.T) {
	embedProv := mock.NewEmbed(
		mock.WithEmbedSupportedModels("test-embed"),
		mock.WithEmbedMaxBatch(5),
	)

	cfg := ir.Config{
		DefaultModel: "test-embed",
		Accounts: []ir.AccountConfig{
			{Provider: "mock-embed", ID: "free-1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens,
				CostPerEmbeddingInputToken: 0.0001},
		},
	}
	r := newEmbedRouter(t, cfg, embedProv)

	inputs := make([]string, 10)
	for i := range inputs {
		inputs[i] = "test"
	}
	_, err := r.Embed(context.Background(), ir.EmbedRequest{Inputs: inputs})
	assert.ErrorIs(t, err, ir.ErrBatchTooLarge)
}

// Test: NewRouter rejects embedding alias with multiple models (single-model invariant).
// This is the core correctness guarantee from qarap review — cross-model fallback
// silently corrupts RAG retrieval.
func TestNewRouter_RejectsCrossModelEmbeddingAlias(t *testing.T) {
	embedProv := mock.NewEmbed(
		mock.WithEmbedSupportedModels("model-a", "model-b"),
	)

	cfg := ir.Config{
		Models: []ir.ModelMapping{
			{
				Alias: "bad-embed",
				Models: []ir.ModelRef{
					{Provider: "mock-embed", Model: "model-a"},
					{Provider: "mock-embed", Model: "model-b"}, // cross-model: forbidden
				},
			},
		},
		Accounts: []ir.AccountConfig{
			{Provider: "mock-embed", ID: "acc-1", DailyFree: 100, QuotaUnit: ir.QuotaTokens,
				CostPerEmbeddingInputToken: 0.0001},
		},
	}

	_, err := ir.NewRouter(cfg, []ir.Provider{embedProviderAsProvider(embedProv)})
	require.Error(t, err)
	assert.ErrorIs(t, err, ir.ErrInvalidConfig)
	assert.Contains(t, err.Error(), "bad-embed")
	assert.Contains(t, err.Error(), "exactly one model")
}

// Test: Single-model alias is accepted (happy path — "multi-account fallback
// on the same model" pattern from RFC §3.6).
func TestNewRouter_AcceptsSingleModelEmbeddingAlias(t *testing.T) {
	embedProv := mock.NewEmbed(
		mock.WithEmbedSupportedModels("text-embedding-004"),
	)

	cfg := ir.Config{
		Models: []ir.ModelMapping{
			{
				Alias: "text-embedding-004",
				Models: []ir.ModelRef{
					{Provider: "mock-embed", Model: "text-embedding-004"},
				},
			},
		},
		Accounts: []ir.AccountConfig{
			// Two accounts on same provider = multi-account fallback, allowed.
			{Provider: "mock-embed", ID: "free-a", DailyFree: 100, QuotaUnit: ir.QuotaTokens,
				CostPerEmbeddingInputToken: 0.0001},
			{Provider: "mock-embed", ID: "free-b", DailyFree: 100, QuotaUnit: ir.QuotaTokens,
				CostPerEmbeddingInputToken: 0.0001},
		},
	}

	_, err := ir.NewRouter(cfg, []ir.Provider{embedProviderAsProvider(embedProv)})
	assert.NoError(t, err)
}

// Test: Chat-only aliases with multiple models still work (invariant only
// applies to aliases that reference embedding models).
func TestNewRouter_ChatAliasMultiModelUnaffected(t *testing.T) {
	chatProv := mock.New(mock.WithModels("chat-a", "chat-b"))

	cfg := ir.Config{
		Models: []ir.ModelMapping{
			{
				Alias: "chat-alias",
				Models: []ir.ModelRef{
					{Provider: "mock", Model: "chat-a"},
					{Provider: "mock", Model: "chat-b"},
				},
			},
		},
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "acc-1", DailyFree: 100, QuotaUnit: ir.QuotaTokens},
		},
	}
	_, err := ir.NewRouter(cfg, []ir.Provider{chatProv})
	assert.NoError(t, err)
}

// Test: ErrPartialBatch unwrap + As behavior.
func TestErrPartialBatch_ErrorsAs(t *testing.T) {
	cause := errors.New("boom")
	err := &ir.ErrPartialBatch{ProcessedInputs: 42, Cause: cause}

	var partial *ir.ErrPartialBatch
	assert.True(t, errors.As(err, &partial))
	assert.Equal(t, 42, partial.ProcessedInputs)

	assert.True(t, errors.Is(err, cause))
	assert.Contains(t, err.Error(), "42")
}

// Test: Embed config rejects negative cost.
func TestConfig_NegativeEmbeddingCost(t *testing.T) {
	cfg := ir.Config{
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "acc-1", DailyFree: 100, QuotaUnit: ir.QuotaTokens,
				CostPerEmbeddingInputToken: -1},
		},
	}
	err := cfg.Validate()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "cost_per_embedding_input_token")
}

// --- helpers ---

// newEmbedRouter creates a router with an embedding provider. We need to
// declare the provider both as Provider and EmbeddingProvider; the mock
// EmbedProvider is NOT a chat Provider, so we wrap it.
func newEmbedRouter(t *testing.T, cfg ir.Config, embedProv *mock.EmbedProvider, extra ...ir.Provider) *ir.Router {
	t.Helper()
	qs := quota.NewMemoryQuotaStore()
	providers := append([]ir.Provider{embedProviderAsProvider(embedProv)}, extra...)
	r, err := ir.NewRouter(cfg, providers,
		ir.WithQuotaStore(qs),
		ir.WithMeter(&meter.NoopMeter{}),
	)
	require.NoError(t, err)
	return r
}

// embedProviderAsProvider adapts an EmbedProvider (which only implements
// EmbeddingProvider) to the chat Provider interface as a non-functional
// stub. Router only looks at the chat Provider.Name() for account matching;
// the chat methods are never called for embed-only paths in these tests.
func embedProviderAsProvider(ep *mock.EmbedProvider) ir.Provider {
	return &embedOnlyStub{inner: ep}
}

type embedOnlyStub struct {
	inner *mock.EmbedProvider
}

func (s *embedOnlyStub) Name() string                { return s.inner.Name() }
func (s *embedOnlyStub) SupportsModel(string) bool   { return false }
func (s *embedOnlyStub) SupportsMultimodal() bool    { return false }
func (s *embedOnlyStub) ChatCompletion(context.Context, ir.ProviderRequest) (ir.ProviderResponse, error) {
	return ir.ProviderResponse{}, errors.New("mock-embed does not support chat")
}
func (s *embedOnlyStub) ChatCompletionStream(context.Context, ir.ProviderRequest) (ir.ProviderStream, error) {
	return nil, errors.New("mock-embed does not support chat")
}

// Embed delegates to the inner EmbedProvider. This implements
// EmbeddingProvider on the stub so type-assertion in NewRouter picks it up.
func (s *embedOnlyStub) Embed(ctx context.Context, req ir.EmbedProviderRequest) (ir.EmbedProviderResponse, error) {
	return s.inner.Embed(ctx, req)
}
func (s *embedOnlyStub) SupportsEmbeddingModel(m string) bool { return s.inner.SupportsEmbeddingModel(m) }
func (s *embedOnlyStub) MaxBatchSize() int                    { return s.inner.MaxBatchSize() }
