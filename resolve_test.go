package inferrouter_test

import (
	"context"
	"errors"
	"testing"

	ir "github.com/ineyio/inferrouter"
	"github.com/ineyio/inferrouter/provider/mock"
	"github.com/ineyio/inferrouter/quota"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// L2 (R2): a name that is not a declared alias is rejected outright. Before
// strict resolution it fell through to "try this name against every provider",
// and openaicompat's empty whitelist accepted anything — so a typo in a ladder
// name quietly widened the candidate set to every account (RFC §3.4).
func TestRouter_UnknownAliasFailsClosed(t *testing.T) {
	log := &attemptLog{}
	cfg := ladderConfig("step-one", "step-two")
	r, err := ir.NewRouter(cfg,
		[]ir.Provider{failingStep(log, "step-one"), servingStep(log, "step-two")},
		ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
	require.NoError(t, err)

	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{
		Model:    "laddr", // typo
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})

	require.Error(t, err)
	assert.True(t, errors.Is(err, ir.ErrUnknownAlias), "got %v", err)
	assert.Contains(t, err.Error(), "laddr", "the offending name belongs in the message")
	assert.Empty(t, log.snapshot(), "no account may be attempted for an undeclared name")
	assert.True(t, ir.IsFatal(err), "a config error must not be retried against other candidates")
}

// A bare model name is not a ladder either, however plausible it looks.
func TestRouter_DirectModelNameIsNotAnAlias(t *testing.T) {
	log := &attemptLog{}
	cfg := ladderConfig("step-one", "step-two")
	r, err := ir.NewRouter(cfg,
		[]ir.Provider{failingStep(log, "step-one"), servingStep(log, "step-two")},
		ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
	require.NoError(t, err)

	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{
		Model:    "ladder-model", // the model the steps actually serve
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})

	assert.True(t, errors.Is(err, ir.ErrUnknownAlias), "got %v", err)
	assert.Empty(t, log.snapshot())
}

// Neither a request model nor a default is a configuration error, not a
// silent walk over every account.
func TestRouter_NoModelAndNoDefault(t *testing.T) {
	log := &attemptLog{}
	cfg := ladderConfig("step-one")
	cfg.DefaultModel = ""
	r, err := ir.NewRouter(cfg, []ir.Provider{servingStep(log, "step-one")},
		ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
	require.NoError(t, err)

	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})

	assert.True(t, errors.Is(err, ir.ErrUnknownAlias), "got %v", err)
	assert.Empty(t, log.snapshot())
}

// DefaultModel names a ladder, and an empty request model uses it.
func TestRouter_DefaultModelResolvesToLadder(t *testing.T) {
	log := &attemptLog{}
	cfg := ladderConfig("step-one")
	cfg.DefaultModel = "ladder"
	r, err := ir.NewRouter(cfg, []ir.Provider{servingStep(log, "step-one")},
		ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
	require.NoError(t, err)

	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})
	require.NoError(t, err)
	assert.Equal(t, []string{"step-one"}, log.snapshot())
}

// Embeddings resolve through the same strict path — one resolver, one rule.
func TestEmbed_UnknownAliasFailsClosed(t *testing.T) {
	embedProv := mock.NewEmbed(
		mock.WithEmbedName("mock-embed"),
		mock.WithEmbedSupportedModels("text-embedding-004"),
		mock.WithEmbedMaxBatch(100),
	)
	cfg := ir.Config{
		DefaultModel: "embed",
		Models: []ir.ModelMapping{{
			Alias:  "embed",
			Models: []ir.ModelRef{{Provider: "mock-embed", Model: "text-embedding-004"}},
		}},
		Accounts: []ir.AccountConfig{
			{Provider: "mock-embed", ID: "acc", DailyFree: 1000, QuotaUnit: ir.QuotaTokens,
				CostPerEmbeddingInputToken: 0.0001},
		},
	}
	r := newEmbedRouter(t, cfg, embedProv)

	_, err := r.EmbedBatch(context.Background(), ir.EmbedRequest{
		Model:  "text-embedding-004", // the model, not the declared alias
		Inputs: []string{"hello"},
	})

	assert.True(t, errors.Is(err, ir.ErrUnknownAlias), "got %v", err)
	assert.EqualValues(t, 0, embedProv.CallCount())
}

// Multi-account fallback on one model is legal for embeddings even when the
// accounts sit behind different provider entries: the RAG guarantee is one
// MODEL per alias, and with strict resolution a ladder is the only way to
// express that fallback at all.
func TestEmbed_MultiProviderSameModelAllowed(t *testing.T) {
	primary := mock.NewEmbed(
		mock.WithEmbedName("primary"),
		mock.WithEmbedSupportedModels("text-embedding-004"),
		mock.WithEmbedMaxBatch(100),
	)
	secondary := mock.NewEmbed(
		mock.WithEmbedName("secondary"),
		mock.WithEmbedSupportedModels("text-embedding-004"),
		mock.WithEmbedMaxBatch(100),
	)
	cfg := ir.Config{
		DefaultModel: "embed",
		Models: []ir.ModelMapping{{Alias: "embed", Models: []ir.ModelRef{
			{Provider: "primary", Model: "text-embedding-004"},
			{Provider: "secondary", Model: "text-embedding-004"},
		}}},
		Accounts: []ir.AccountConfig{
			{Provider: "primary", ID: "primary-acc", DailyFree: 1000, QuotaUnit: ir.QuotaTokens,
				CostPerEmbeddingInputToken: 0.0001},
			{Provider: "secondary", ID: "secondary-acc", DailyFree: 1000, QuotaUnit: ir.QuotaTokens,
				CostPerEmbeddingInputToken: 0.0001},
		},
	}

	r := newEmbedRouter(t, cfg, primary, embedProviderAsProvider(secondary))
	_, err := r.EmbedBatch(context.Background(), ir.EmbedRequest{Inputs: []string{"hello"}})
	require.NoError(t, err)
}

// Two DIFFERENT models under one embedding alias stay rejected — that is the
// RAG-correctness rule the check exists for.
func TestEmbed_CrossModelAliasStillRejected(t *testing.T) {
	embedProv := mock.NewEmbed(
		mock.WithEmbedName("mock-embed"),
		mock.WithEmbedSupportedModels("text-embedding-004", "gemini-embedding-001"),
		mock.WithEmbedMaxBatch(100),
	)
	cfg := ir.Config{
		DefaultModel: "embed",
		Models: []ir.ModelMapping{{Alias: "embed", Models: []ir.ModelRef{
			{Provider: "mock-embed", Model: "text-embedding-004"},
			{Provider: "mock-embed", Model: "gemini-embedding-001"},
		}}},
		Accounts: []ir.AccountConfig{
			{Provider: "mock-embed", ID: "acc", DailyFree: 1000, QuotaUnit: ir.QuotaTokens,
				CostPerEmbeddingInputToken: 0.0001},
		},
	}

	_, err := ir.NewRouter(cfg, []ir.Provider{embedProviderAsProvider(embedProv)})
	require.Error(t, err)
	assert.True(t, errors.Is(err, ir.ErrInvalidConfig), "got %v", err)
}

// L6 (§7 e): qarap branches its diagnostics on these sentinels and on the
// RouterError fields. They are a published contract — renaming one or changing
// the wrap chain still compiles here and degrades diagnostics there.
func TestErrorTaxonomyContract(t *testing.T) {
	t.Run("ErrNoCandidates reachable", func(t *testing.T) {
		// A declared ladder whose provider is not registered: nothing to try.
		cfg := ir.Config{
			DefaultModel: "ladder",
			Models: []ir.ModelMapping{{Alias: "ladder", Models: []ir.ModelRef{
				{Provider: "absent", Model: "m"},
			}}},
			Accounts: []ir.AccountConfig{
				{Provider: "other", ID: "acc", DailyFree: 10, QuotaUnit: ir.QuotaRequests},
			},
		}
		r, err := ir.NewRouter(cfg, []ir.Provider{mock.New(mock.WithName("other"))},
			ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
		require.NoError(t, err)

		_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{
			Messages: []ir.Message{{Role: "user", Content: "hi"}},
		})
		assert.True(t, errors.Is(err, ir.ErrNoCandidates), "got %v", err)
	})

	t.Run("ErrAllFailed unwraps through RouterError", func(t *testing.T) {
		log := &attemptLog{}
		cfg := ladderConfig("step-one", "step-two")
		cfg.Accounts[1].DailyFree = 0 // both steps paid so both are attempted
		cfg.Accounts[1].PaidEnabled = true

		r, err := ir.NewRouter(cfg,
			[]ir.Provider{failingStep(log, "step-one"), failingStep(log, "step-two")},
			ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
		require.NoError(t, err)

		_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{
			Model:    "ladder",
			Messages: []ir.Message{{Role: "user", Content: "hi"}},
		})
		require.Error(t, err)

		var routerErr *ir.RouterError
		require.True(t, errors.As(err, &routerErr), "err must carry RouterError: %v", err)
		assert.True(t, errors.Is(routerErr.Unwrap(), ir.ErrAllFailed))
		assert.Equal(t, 2, routerErr.Attempts)
		assert.Len(t, routerErr.Tried, 2, "every attempted step is reported")
		assert.Equal(t, "step-one", routerErr.Tried[0].Provider)
		assert.Equal(t, "step-one-acc", routerErr.Tried[0].AccountID)
		assert.Equal(t, "ladder-model", routerErr.Tried[0].Model)
	})

	t.Run("classification helpers", func(t *testing.T) {
		assert.True(t, ir.IsFatal(ir.ErrAuthFailed))
		assert.True(t, ir.IsFatal(ir.ErrInvalidRequest))
		assert.True(t, ir.IsRetryable(ir.ErrRateLimited))
		assert.True(t, ir.IsRetryable(ir.ErrProviderUnavailable))
		assert.True(t, ir.IsRetryable(ir.ErrQuotaExceeded))
		assert.True(t, ir.IsRetryable(ir.ErrRPMExceeded))
		assert.False(t, ir.IsRetryable(ir.ErrMultimodalUnavailable),
			"callers degrade on this one deliberately, they don't retry it")
	})
}
