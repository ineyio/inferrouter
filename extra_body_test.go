package inferrouter_test

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	ir "github.com/ineyio/inferrouter"
	"github.com/ineyio/inferrouter/provider/mock"
	"github.com/ineyio/inferrouter/provider/openaicompat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func extraBodyConfig(refs ...ir.ModelRef) ir.Config {
	cfg := ir.Config{
		AllowPaid:    true,
		DefaultModel: "ladder",
		Models:       []ir.ModelMapping{{Alias: "ladder", Models: refs}},
	}
	seen := map[string]bool{}
	for _, ref := range refs {
		if !seen[ref.Provider] {
			seen[ref.Provider] = true
			cfg.Accounts = append(cfg.Accounts, ir.AccountConfig{
				Provider: ref.Provider, ID: ref.Provider + "-acc", QuotaUnit: ir.QuotaTokens,
			})
		}
	}
	return cfg
}

// The list is written out here, not read from the package: a key dropped from
// the code must fail this test.
func TestExtraBody_ReservedKeysAreRefused(t *testing.T) {
	for _, key := range []string{"model", "messages", "stream", "stream_options", "max_tokens", "max_completion_tokens"} {
		t.Run(key, func(t *testing.T) {
			cfg := extraBodyConfig(ir.ModelRef{Provider: "p", Model: "m", ExtraBody: map[string]any{key: 1}})

			err := cfg.Validate()
			require.Error(t, err)
			assert.Contains(t, err.Error(), key)

			_, err = ir.NewRouter(cfg, []ir.Provider{mock.New(mock.WithName("p"))})
			assert.Error(t, err, "a config built in code goes through NewRouter, not LoadConfig")
		})
	}
}

func TestExtraBody_OtherFieldsPassValidate(t *testing.T) {
	cfg := extraBodyConfig(ir.ModelRef{Provider: "p", Model: "m", ExtraBody: map[string]any{
		"temperature":          0.1,
		"reasoning_effort":     "none",
		"chat_template_kwargs": map[string]any{"enable_thinking": false},
	}})
	assert.NoError(t, cfg.Validate())
}

func TestExtraBody_UnserialisableMapIsRefusedAtStartup(t *testing.T) {
	for name, value := range map[string]any{
		"NaN":     math.NaN(),
		"nested":  map[string]any{"inner": math.Inf(1)},
		"channel": make(chan int),
	} {
		t.Run(name, func(t *testing.T) {
			cfg := extraBodyConfig(ir.ModelRef{Provider: "p", Model: "m", ExtraBody: map[string]any{"field": value}})
			assert.Error(t, cfg.Validate())
		})
	}
}

func TestLoadConfig_ExtraBody(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	require.NoError(t, os.WriteFile(path, []byte(`
accounts:
  - provider: gate
    id: gate-1
    quota_unit: tokens
models:
  - alias: extract
    models:
      - provider: gate
        model: m1
        extra_body:
          reasoning_effort: low
          chat_template_kwargs: {enable_thinking: false}
      - provider: gate
        model: m2
`), 0o600))

	cfg, err := ir.LoadConfig(path)
	require.NoError(t, err)
	refs := cfg.Models[0].Models
	assert.Equal(t, map[string]any{
		"reasoning_effort":     "low",
		"chat_template_kwargs": map[string]any{"enable_thinking": false},
	}, refs[0].ExtraBody)
	assert.Nil(t, refs[1].ExtraBody)
}

func TestLoadConfig_ExtraBodyWithReservedKeyFails(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	require.NoError(t, os.WriteFile(path, []byte(`
accounts:
  - {provider: gate, id: gate-1, quota_unit: tokens}
models:
  - alias: extract
    models:
      - {provider: gate, model: m1, extra_body: {max_tokens: 10}}
`), 0o600))

	_, err := ir.LoadConfig(path)
	assert.ErrorContains(t, err, "max_tokens")
}

// reversePolicy turns the walk around, as a latency sort may.
type reversePolicy struct{}

func (reversePolicy) Select(c []ir.Candidate) []ir.Candidate {
	out := make([]ir.Candidate, 0, len(c))
	for i := len(c) - 1; i >= 0; i-- {
		out = append(out, c[i])
	}
	return out
}

// bodyLog records, per attempt, the model asked for and the extra_body it came
// with, and refuses every attempt so that the walk visits each step.
type bodyLog struct {
	mu   sync.Mutex
	seen []ir.ProviderRequest
}

func (l *bodyLog) provider(name string) *mock.Provider {
	return mock.New(mock.WithName(name), mock.WithModels("m1", "m2"),
		mock.WithResponseFunc(func(req ir.ProviderRequest) (ir.ProviderResponse, error) {
			l.mu.Lock()
			defer l.mu.Unlock()
			l.seen = append(l.seen, req)
			return ir.ProviderResponse{}, ir.ErrProviderUnavailable
		}))
}

func TestExtraBody_TravelsWithItsStepUnderAPolicy(t *testing.T) {
	log := &bodyLog{}
	low := map[string]any{"reasoning_effort": "low"}
	none := map[string]any{"reasoning_effort": "none"}
	cfg := extraBodyConfig(
		ir.ModelRef{Provider: "a", Model: "m1", ExtraBody: low},
		ir.ModelRef{Provider: "b", Model: "m2", ExtraBody: none},
		ir.ModelRef{Provider: "b", Model: "m1"},
	)
	r, err := ir.NewRouter(cfg, []ir.Provider{log.provider("a"), log.provider("b")}, ir.WithPolicy(reversePolicy{}))
	require.NoError(t, err)

	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{Messages: []ir.Message{{Role: "user", Content: "hi"}}})
	require.Error(t, err)

	require.Len(t, log.seen, 3)
	assert.Equal(t, "m1", log.seen[0].Model)
	assert.Nil(t, log.seen[0].ExtraBody, "reversed: the last step, which has none, goes first")
	assert.Equal(t, none, log.seen[1].ExtraBody)
	assert.Equal(t, low, log.seen[2].ExtraBody)
}

// One model on one gateway, asked two ways, is two steps of a ladder.
func TestExtraBody_SameModelTwiceWithDifferentFields(t *testing.T) {
	log := &bodyLog{}
	cfg := extraBodyConfig(
		ir.ModelRef{Provider: "a", Model: "m1", ExtraBody: map[string]any{"reasoning_effort": "none"}},
		ir.ModelRef{Provider: "a", Model: "m1", ExtraBody: map[string]any{"reasoning_effort": "low"}},
	)
	r, err := ir.NewRouter(cfg, []ir.Provider{log.provider("a")})
	require.NoError(t, err)

	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{Messages: []ir.Message{{Role: "user", Content: "hi"}}})
	require.Error(t, err)

	require.Len(t, log.seen, 2)
	assert.Equal(t, "none", log.seen[0].ExtraBody["reasoning_effort"])
	assert.Equal(t, "low", log.seen[1].ExtraBody["reasoning_effort"])
}

func TestExtraBody_RoutingCopiesWhatTheAdapterReports(t *testing.T) {
	p := mock.New(mock.WithName("a"), mock.WithModels("m1"),
		mock.WithResponseFunc(func(ir.ProviderRequest) (ir.ProviderResponse, error) {
			return ir.ProviderResponse{Content: "ok", ExtraBodyApplied: []string{"from_adapter"}}, nil
		}))
	cfg := extraBodyConfig(ir.ModelRef{Provider: "a", Model: "m1", ExtraBody: map[string]any{"in_config": 1}})
	r, err := ir.NewRouter(cfg, []ir.Provider{p})
	require.NoError(t, err)

	resp, err := r.ChatCompletion(context.Background(), ir.ChatRequest{Messages: []ir.Message{{Role: "user", Content: "hi"}}})
	require.NoError(t, err)
	assert.Equal(t, []string{"from_adapter"}, resp.Routing.ExtraBody, "the adapter's report, not the config")
}

// wrappedProvider is a caller's wrapper that embeds the interface, as qarap's
// do. Embedding promotes only the methods of ir.Provider.
type wrappedProvider struct{ ir.Provider }

// embedMock is a provider with an embedding path, for the warning about it.
type embedMock struct{ *mock.Provider }

func (embedMock) SupportsEmbeddingModel(model string) bool { return model == "embed-model" }
func (embedMock) MaxBatchSize() int                        { return 10 }
func (embedMock) Embed(context.Context, ir.EmbedProviderRequest) (ir.EmbedProviderResponse, error) {
	return ir.EmbedProviderResponse{}, nil
}

func TestExtraBody_ConfigWarnings(t *testing.T) {
	extra := map[string]any{"reasoning_effort": "low"}
	gateway := openaicompat.New("gate", "http://127.0.0.1:1")

	tests := []struct {
		name     string
		provider ir.Provider
		ref      ir.ModelRef
		want     string
	}{
		{"adapter that sends", gateway, ir.ModelRef{Provider: "gate", Model: "m1", ExtraBody: extra}, ""},
		{"adapter that cannot", mock.New(mock.WithName("gate")), ir.ModelRef{Provider: "gate", Model: "m1", ExtraBody: extra}, "does not send"},
		{"wrapper without forwarding", wrappedProvider{gateway}, ir.ModelRef{Provider: "gate", Model: "m1", ExtraBody: extra}, "does not send"},
		{"no extra_body", mock.New(mock.WithName("gate")), ir.ModelRef{Provider: "gate", Model: "m1"}, ""},
		{"embedding step", embedMock{mock.New(mock.WithName("gate"))}, ir.ModelRef{Provider: "gate", Model: "embed-model", ExtraBody: extra}, "embeddings"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r, err := ir.NewRouter(extraBodyConfig(tt.ref), []ir.Provider{tt.provider})
			require.NoError(t, err)

			warnings := strings.Join(r.ConfigWarnings(), "\n")
			if tt.want == "" {
				assert.Empty(t, warnings)
				return
			}
			assert.Contains(t, warnings, tt.want)
			assert.Contains(t, warnings, "gate/"+tt.ref.Model, "the warning names the step")
		})
	}
}
