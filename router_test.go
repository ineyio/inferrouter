package inferrouter_test

import (
	"context"
	"errors"
	"io"
	"sync"
	"testing"
	"time"

	ir "github.com/ineyio/inferrouter"
	"github.com/ineyio/inferrouter/meter"
	"github.com/ineyio/inferrouter/policy"
	"github.com/ineyio/inferrouter/provider/mock"
	"github.com/ineyio/inferrouter/quota"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestRouter(t *testing.T, cfg ir.Config, providers []ir.Provider) *ir.Router {
	t.Helper()
	qs := quota.NewMemoryQuotaStore()
	r, err := ir.NewRouter(cfg, providers,
		ir.WithQuotaStore(qs),
		ir.WithPolicy(&policy.FreeFirstPolicy{}),
		ir.WithMeter(&meter.NoopMeter{}),
	)
	require.NoError(t, err)
	return r
}

// Test 1: Free candidate selected when quota available
func TestFreeCandidate_SelectedWhenQuotaAvailable(t *testing.T) {
	mockProv := mock.New(mock.WithModels("test-model"))

	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "free-1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
		},
	}

	r := newTestRouter(t, cfg, []ir.Provider{mockProv})

	resp, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hello"}},
	})
	require.NoError(t, err)
	assert.Equal(t, "free-1", resp.Routing.AccountID)
	assert.True(t, resp.Routing.Free)
	assert.Equal(t, "Hello from mock provider", resp.Choices[0].Message.Content)
}

// Test 2: Fallback to next free on ErrQuotaExceeded
func TestFallback_ToNextFreeOnQuotaExceeded(t *testing.T) {
	mockProv := mock.New(mock.WithModels("test-model"))

	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "free-1", DailyFree: 1, QuotaUnit: ir.QuotaTokens},    // almost exhausted
			{Provider: "mock", ID: "free-2", DailyFree: 1000, QuotaUnit: ir.QuotaTokens}, // plenty
		},
	}

	r := newTestRouter(t, cfg, []ir.Provider{mockProv})

	resp, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hello"}},
	})
	require.NoError(t, err)
	// free-1 has quota 1 but estimated tokens > 1, so it should fail and fallback to free-2
	assert.Equal(t, "free-2", resp.Routing.AccountID)
}

// Test 3: Error when all free exhausted + AllowPaid=false
func TestError_AllFreeExhausted_NoPaid(t *testing.T) {
	mockProv := mock.New(mock.WithModels("test-model"))

	cfg := ir.Config{
		AllowPaid:    false,
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			// DailyFree=0 → no free quota, not a free candidate
			{Provider: "mock", ID: "free-1", DailyFree: 0, QuotaUnit: ir.QuotaTokens},
		},
	}

	r := newTestRouter(t, cfg, []ir.Provider{mockProv})

	_, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hello"}},
	})
	assert.ErrorIs(t, err, ir.ErrNoCandidates)
}

// Test 4: Paid fallback when free exhausted + AllowPaid=true
func TestPaidFallback_WhenFreeExhausted(t *testing.T) {
	mockProv := mock.New(mock.WithModels("test-model"))

	cfg := ir.Config{
		AllowPaid:    true,
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "free-1", DailyFree: 0, QuotaUnit: ir.QuotaTokens},
			{Provider: "mock", ID: "paid-1", DailyFree: 0, QuotaUnit: ir.QuotaTokens, PaidEnabled: true, CostPerToken: 0.001},
		},
	}

	r := newTestRouter(t, cfg, []ir.Provider{mockProv})

	resp, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hello"}},
	})
	require.NoError(t, err)
	assert.Equal(t, "paid-1", resp.Routing.AccountID)
	assert.False(t, resp.Routing.Free)
}

// Test 5: Reservation prevents race (concurrent goroutines)
func TestReservation_PreventsRace(t *testing.T) {
	mockProv := mock.New(mock.WithModels("test-model"))

	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "free-1", DailyFree: 10000, QuotaUnit: ir.QuotaTokens},
		},
	}

	r := newTestRouter(t, cfg, []ir.Provider{mockProv})

	var wg sync.WaitGroup
	errs := make([]error, 20)

	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			_, errs[idx] = r.ChatCompletion(context.Background(), ir.ChatRequest{
				Messages: []ir.Message{{Role: "user", Content: "hello"}},
			})
		}(i)
	}

	wg.Wait()

	successCount := 0
	for _, err := range errs {
		if err == nil {
			successCount++
		}
	}
	assert.Greater(t, successCount, 0)
}

// Test 6: IdempotencyKey dedup
func TestIdempotencyKey_Dedup(t *testing.T) {
	qs := quota.NewMemoryQuotaStore()
	qs.SetQuota("acc-1", 1000, ir.QuotaTokens)

	ctx := context.Background()

	// First reservation succeeds.
	_, err := qs.Reserve(ctx, "acc-1", 100, ir.QuotaTokens, "key-1")
	require.NoError(t, err)

	// Same key → error.
	_, err = qs.Reserve(ctx, "acc-1", 100, ir.QuotaTokens, "key-1")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "duplicate")
}

// Test 7: Circuit breaker opens after failures
func TestCircuitBreaker_OpensAfterFailures(t *testing.T) {
	ht := ir.NewHealthTracker()

	assert.Equal(t, ir.HealthHealthy, ht.GetHealth("acc-1"))

	// 3 failures → unhealthy
	ht.RecordFailure("acc-1")
	ht.RecordFailure("acc-1")
	ht.RecordFailure("acc-1")

	assert.Equal(t, ir.HealthUnhealthy, ht.GetHealth("acc-1"))
}

// Test 8: Circuit breaker half-open recovery
func TestCircuitBreaker_HalfOpenRecovery(t *testing.T) {
	ht := ir.NewHealthTracker()

	ht.RecordFailure("acc-1")
	ht.RecordFailure("acc-1")
	ht.RecordFailure("acc-1")
	assert.Equal(t, ir.HealthUnhealthy, ht.GetHealth("acc-1"))

	// After success, should go healthy.
	ht.RecordSuccess("acc-1")
	assert.Equal(t, ir.HealthHealthy, ht.GetHealth("acc-1"))
}

// Test 9: Model aliasing resolves correctly
func TestModelAliasing_ResolvesCorrectly(t *testing.T) {
	geminiProv := mock.New(mock.WithName("gemini"), mock.WithModels("gemini-2.0-flash"))
	grokProv := mock.New(mock.WithName("grok"), mock.WithModels("grok-3"))

	cfg := ir.Config{
		DefaultModel: "fast",
		Models: []ir.ModelMapping{
			{
				Alias: "fast",
				Models: []ir.ModelRef{
					{Provider: "gemini", Model: "gemini-2.0-flash"},
					{Provider: "grok", Model: "grok-3"},
				},
			},
		},
		Accounts: []ir.AccountConfig{
			{Provider: "gemini", ID: "gemini-1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
			{Provider: "grok", ID: "grok-1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
		},
	}

	r := newTestRouter(t, cfg, []ir.Provider{geminiProv, grokProv})

	resp, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
		Model:    "fast",
		Messages: []ir.Message{{Role: "user", Content: "hello"}},
	})
	require.NoError(t, err)
	// Should resolve to one of the aliased models.
	assert.Contains(t, []string{"gemini-2.0-flash", "grok-3"}, resp.Routing.Model)
}

// Test 10: Streaming passthrough works
func TestStreaming_Passthrough(t *testing.T) {
	mockProv := mock.New(mock.WithModels("test-model"))

	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "free-1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
		},
	}

	r := newTestRouter(t, cfg, []ir.Provider{mockProv})

	stream, err := r.ChatCompletionStream(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hello"}},
	})
	require.NoError(t, err)

	var chunks []ir.StreamChunk
	for {
		chunk, err := stream.Next()
		if err == io.EOF {
			break
		}
		require.NoError(t, err)
		chunks = append(chunks, chunk)
	}
	require.NoError(t, stream.Close())

	assert.Greater(t, len(chunks), 0)
}

// Test 11: Provider error mapping (retryable vs fatal)
func TestProviderErrorMapping(t *testing.T) {
	failProv := mock.New(
		mock.WithName("fail"),
		mock.WithModels("test-model"),
		mock.WithError(ir.ErrRateLimited),
	)
	goodProv := mock.New(
		mock.WithName("good"),
		mock.WithModels("test-model"),
	)

	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "fail", ID: "fail-1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
			{Provider: "good", ID: "good-1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
		},
	}

	r := newTestRouter(t, cfg, []ir.Provider{failProv, goodProv})

	resp, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hello"}},
	})
	require.NoError(t, err)
	assert.Equal(t, "good-1", resp.Routing.AccountID)
}

// Test: Fatal error stops retrying
func TestFatalError_StopsRetrying(t *testing.T) {
	failProv := mock.New(
		mock.WithModels("test-model"),
		mock.WithError(ir.ErrAuthFailed),
	)

	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "acc-1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
			{Provider: "mock", ID: "acc-2", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
		},
	}

	r := newTestRouter(t, cfg, []ir.Provider{failProv})

	_, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hello"}},
	})
	require.Error(t, err)
	assert.ErrorIs(t, err, ir.ErrAuthFailed)

	// Should have stopped after first attempt.
	var routerErr *ir.RouterError
	assert.ErrorAs(t, err, &routerErr)
	assert.Equal(t, 1, routerErr.Attempts)
}

// Test: Quota auto-initialization from config
func TestQuotaAutoInit_FromConfig(t *testing.T) {
	mockProv := mock.New(mock.WithModels("test-model"))

	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "auto-1", DailyFree: 500, QuotaUnit: ir.QuotaRequests},
		},
	}

	// Don't call SetQuota manually — NewRouter should do it.
	qs := quota.NewMemoryQuotaStore()
	r, err := ir.NewRouter(cfg, []ir.Provider{mockProv},
		ir.WithQuotaStore(qs),
		ir.WithPolicy(&policy.FreeFirstPolicy{}),
	)
	require.NoError(t, err)

	resp, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hello"}},
	})
	require.NoError(t, err)
	assert.Equal(t, "auto-1", resp.Routing.AccountID)
	assert.True(t, resp.Routing.Free)

	// Verify quota was consumed.
	remaining, err := qs.Remaining(context.Background(), "auto-1")
	require.NoError(t, err)
	assert.Equal(t, int64(499), remaining) // 500 - 1 request
}

// Test: EstimateTokens
func TestEstimateTokens(t *testing.T) {
	msgs := []ir.Message{
		{Role: "user", Content: "Hello, how are you?"}, // 19 chars → ~4 tokens + 4 overhead
	}
	tokens := ir.EstimateTokens(msgs)
	assert.Greater(t, tokens, int64(0))
}

// Test: Config validation
func TestConfig_Validate(t *testing.T) {
	t.Run("empty accounts", func(t *testing.T) {
		cfg := ir.Config{}
		err := cfg.Validate()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "at least one account")
	})

	t.Run("missing provider", func(t *testing.T) {
		cfg := ir.Config{
			Accounts: []ir.AccountConfig{{ID: "a", QuotaUnit: ir.QuotaTokens}},
		}
		err := cfg.Validate()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "provider is required")
	})

	t.Run("duplicate id", func(t *testing.T) {
		cfg := ir.Config{
			Accounts: []ir.AccountConfig{
				{Provider: "mock", ID: "a", QuotaUnit: ir.QuotaTokens},
				{Provider: "mock", ID: "a", QuotaUnit: ir.QuotaTokens},
			},
		}
		err := cfg.Validate()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "duplicate")
	})

	t.Run("valid config", func(t *testing.T) {
		cfg := ir.Config{
			Accounts: []ir.AccountConfig{
				{Provider: "mock", ID: "a", QuotaUnit: ir.QuotaTokens},
			},
		}
		err := cfg.Validate()
		assert.NoError(t, err)
	})
}

// Test: HealthState String()
func TestHealthState_String(t *testing.T) {
	assert.Equal(t, "healthy", ir.HealthHealthy.String())
	assert.Equal(t, "unhealthy", ir.HealthUnhealthy.String())
	assert.Equal(t, "half-open", ir.HealthHalfOpen.String())
}

// Test: Error helpers
func TestErrorHelpers(t *testing.T) {
	assert.True(t, ir.IsFatal(ir.ErrAuthFailed))
	assert.True(t, ir.IsFatal(ir.ErrInvalidRequest))
	assert.False(t, ir.IsFatal(ir.ErrRateLimited))

	assert.True(t, ir.IsRetryable(ir.ErrRateLimited))
	assert.True(t, ir.IsRetryable(ir.ErrProviderUnavailable))
	assert.False(t, ir.IsRetryable(ir.ErrAuthFailed))
}

// --- New tests: Config Validation ---

func TestConfigValidation_NegativeValues(t *testing.T) {
	tests := []struct {
		name        string
		cfg         ir.Config
		errContains string
	}{
		{
			name: "negative daily_free",
			cfg: ir.Config{
				Accounts: []ir.AccountConfig{
					{Provider: "mock", ID: "a", DailyFree: -100, QuotaUnit: ir.QuotaTokens},
				},
			},
			errContains: "daily_free must be >= 0",
		},
		{
			name: "negative max_daily_spend",
			cfg: ir.Config{
				Accounts: []ir.AccountConfig{
					{Provider: "mock", ID: "a", MaxDailySpend: -10.5, QuotaUnit: ir.QuotaTokens},
				},
			},
			errContains: "max_daily_spend must be >= 0",
		},
		{
			name: "negative cost_per_token",
			cfg: ir.Config{
				Accounts: []ir.AccountConfig{
					{Provider: "mock", ID: "a", CostPerToken: -0.001, QuotaUnit: ir.QuotaTokens},
				},
			},
			errContains: "cost_per_token must be >= 0",
		},
		{
			name: "negative cost_per_input_token",
			cfg: ir.Config{
				Accounts: []ir.AccountConfig{
					{Provider: "mock", ID: "a", CostPerInputToken: -0.001, QuotaUnit: ir.QuotaTokens},
				},
			},
			errContains: "cost_per_input_token must be >= 0",
		},
		{
			name: "negative cost_per_output_token",
			cfg: ir.Config{
				Accounts: []ir.AccountConfig{
					{Provider: "mock", ID: "a", CostPerOutputToken: -0.001, QuotaUnit: ir.QuotaTokens},
				},
			},
			errContains: "cost_per_output_token must be >= 0",
		},
		{
			name: "paid_enabled without cost",
			cfg: ir.Config{
				Accounts: []ir.AccountConfig{
					{Provider: "mock", ID: "a", PaidEnabled: true, QuotaUnit: ir.QuotaTokens},
				},
			},
			errContains: "paid_enabled requires cost configuration",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.cfg.Validate()
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.errContains)
		})
	}
}

func TestConfigValidation_PaidEnabled_WithCost(t *testing.T) {
	t.Run("old format", func(t *testing.T) {
		cfg := ir.Config{
			Accounts: []ir.AccountConfig{
				{Provider: "mock", ID: "a", PaidEnabled: true, CostPerToken: 0.001, QuotaUnit: ir.QuotaTokens},
			},
		}
		assert.NoError(t, cfg.Validate())
	})

	t.Run("new format", func(t *testing.T) {
		cfg := ir.Config{
			Accounts: []ir.AccountConfig{
				{Provider: "mock", ID: "a", PaidEnabled: true, CostPerInputToken: 0.001, CostPerOutputToken: 0.003, QuotaUnit: ir.QuotaTokens},
			},
		}
		assert.NoError(t, cfg.Validate())
	})
}

// --- New tests: NormalizeCosts ---

func TestNormalizeCosts_BackwardCompat(t *testing.T) {
	cfg := ir.Config{
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "a", CostPerToken: 0.002, QuotaUnit: ir.QuotaTokens},
		},
	}
	cfg.NormalizeCosts()
	assert.Equal(t, 0.002, cfg.Accounts[0].CostPerInputToken)
	assert.Equal(t, 0.002, cfg.Accounts[0].CostPerOutputToken)
}

func TestNormalizeCosts_NoOverwrite(t *testing.T) {
	cfg := ir.Config{
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "a", CostPerToken: 0.002, CostPerInputToken: 0.001, CostPerOutputToken: 0.003, QuotaUnit: ir.QuotaTokens},
		},
	}
	cfg.NormalizeCosts()
	// New fields already set — should not be overwritten.
	assert.Equal(t, 0.001, cfg.Accounts[0].CostPerInputToken)
	assert.Equal(t, 0.003, cfg.Accounts[0].CostPerOutputToken)
}

// --- New tests: SpendTracker ---

func TestSpendTracker_RecordAndGet(t *testing.T) {
	st := ir.NewSpendTracker()

	st.RecordSpend("acc1", 1.5)
	st.RecordSpend("acc1", 0.5)
	st.RecordSpend("acc2", 3.0)

	assert.InDelta(t, 2.0, st.GetSpend("acc1"), 0.001)
	assert.InDelta(t, 3.0, st.GetSpend("acc2"), 0.001)
	assert.Equal(t, 0.0, st.GetSpend("acc3"))
}

// --- New tests: max_daily_spend enforcement ---

func TestMaxDailySpend_Enforcement(t *testing.T) {
	mockProv := mock.New(mock.WithModels("test-model"))

	cfg := ir.Config{
		AllowPaid:    true,
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{
				Provider:       "mock",
				ID:             "paid-1",
				PaidEnabled:    true,
				CostPerToken:   0.001, // $0.001/token
				MaxDailySpend:  0.01,  // $0.01/day
				QuotaUnit:      ir.QuotaTokens,
			},
		},
	}

	qs := quota.NewMemoryQuotaStore()
	st := ir.NewSpendTracker()

	r, err := ir.NewRouter(cfg, []ir.Provider{mockProv},
		ir.WithQuotaStore(qs),
		ir.WithPolicy(&policy.FreeFirstPolicy{}),
		ir.WithSpendTracker(st),
	)
	require.NoError(t, err)

	// First request succeeds. Mock returns 30 total tokens × $0.001 = $0.03.
	// This exceeds $0.01 limit, so second request should fail.
	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hello"}},
	})
	require.NoError(t, err)

	// Second request should fail — spend limit exceeded.
	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hello"}},
	})
	assert.ErrorIs(t, err, ir.ErrNoCandidates)
}

// --- New tests: Separate input/output pricing ---

func TestSeparateInputOutputPricing(t *testing.T) {
	// Mock returns Usage{PromptTokens: 10, CompletionTokens: 20, TotalTokens: 30}
	mockProv := mock.New(mock.WithModels("test-model"))

	cfg := ir.Config{
		AllowPaid:    true,
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{
				Provider:           "mock",
				ID:                 "paid-1",
				PaidEnabled:        true,
				CostPerInputToken:  0.001, // $0.001/token input
				CostPerOutputToken: 0.003, // $0.003/token output
				QuotaUnit:          ir.QuotaTokens,
			},
		},
	}

	qs := quota.NewMemoryQuotaStore()
	st := ir.NewSpendTracker()

	r, err := ir.NewRouter(cfg, []ir.Provider{mockProv},
		ir.WithQuotaStore(qs),
		ir.WithPolicy(&policy.FreeFirstPolicy{}),
		ir.WithSpendTracker(st),
	)
	require.NoError(t, err)

	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hello"}},
	})
	require.NoError(t, err)

	// Expected: 10*0.001 + 20*0.003 = 0.01 + 0.06 = 0.07
	assert.InDelta(t, 0.07, st.GetSpend("paid-1"), 0.0001)
}

// --- New tests: Configurable Circuit Breaker ---

func TestHealthTracker_CustomConfig(t *testing.T) {
	cfg := ir.HealthConfig{
		FailureThreshold: 2,
		FailureWindow:    1 * time.Second,
		UnhealthyPeriod:  100 * time.Millisecond,
	}

	ht := ir.NewHealthTrackerWithConfig(cfg)

	ht.RecordFailure("acc1")
	assert.Equal(t, ir.HealthHealthy, ht.GetHealth("acc1"))

	ht.RecordFailure("acc1")
	assert.Equal(t, ir.HealthUnhealthy, ht.GetHealth("acc1"))

	time.Sleep(150 * time.Millisecond)
	assert.Equal(t, ir.HealthHalfOpen, ht.GetHealth("acc1"))

	ht.RecordSuccess("acc1")
	assert.Equal(t, ir.HealthHealthy, ht.GetHealth("acc1"))
}

// --- New tests: BlendedCost ---

func TestBlendedCost(t *testing.T) {
	t.Run("old format", func(t *testing.T) {
		c := ir.Candidate{CostPerToken: 0.003}
		assert.InDelta(t, 0.003, c.BlendedCost(), 0.0001)
	})

	t.Run("new format", func(t *testing.T) {
		c := ir.Candidate{CostPerInputToken: 0.001, CostPerOutputToken: 0.003}
		// (0.001 + 2*0.003) / 3 = 0.007/3 ≈ 0.002333
		assert.InDelta(t, 0.002333, c.BlendedCost(), 0.0001)
	})

	t.Run("zero cost", func(t *testing.T) {
		c := ir.Candidate{}
		assert.Equal(t, 0.0, c.BlendedCost())
	})
}

// --- New tests: Stream quota error logging ---

func TestStream_QuotaCommitError_Reported(t *testing.T) {
	mockProv := mock.New(mock.WithModels("test-model"))
	spy := &meterSpy{}
	failQS := &failingCommitQuotaStore{}

	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "free-1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
		},
	}

	r, err := ir.NewRouter(cfg, []ir.Provider{mockProv},
		ir.WithQuotaStore(failQS),
		ir.WithMeter(spy),
	)
	require.NoError(t, err)

	stream, err := r.ChatCompletionStream(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hello"}},
	})
	require.NoError(t, err)

	for {
		_, err := stream.Next()
		if err != nil {
			break
		}
	}
	_ = stream.Close()

	// Meter should report the quota failure.
	assert.False(t, spy.lastResult.Success)
	require.NotNil(t, spy.lastResult.Error)
	assert.Contains(t, spy.lastResult.Error.Error(), "quota operation failed")
}

// --- Bug fix tests ---

// Test: ErrAllFailed includes per-candidate errors
func TestAllFailed_IncludesTriedCandidates(t *testing.T) {
	failProv1 := mock.New(
		mock.WithName("prov1"),
		mock.WithModels("test-model"),
		mock.WithError(ir.ErrRateLimited),
	)
	failProv2 := mock.New(
		mock.WithName("prov2"),
		mock.WithModels("test-model"),
		mock.WithError(ir.ErrProviderUnavailable),
	)

	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "prov1", ID: "acc-1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
			{Provider: "prov2", ID: "acc-2", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
		},
	}

	r := newTestRouter(t, cfg, []ir.Provider{failProv1, failProv2})

	_, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hello"}},
	})
	require.Error(t, err)
	assert.ErrorIs(t, err, ir.ErrAllFailed)

	var routerErr *ir.RouterError
	require.ErrorAs(t, err, &routerErr)
	assert.Equal(t, 2, routerErr.Attempts)
	assert.Len(t, routerErr.Tried, 2)

	// Verify per-candidate error details.
	assert.Equal(t, "prov1", routerErr.Tried[0].Provider)
	assert.Equal(t, "acc-1", routerErr.Tried[0].AccountID)
	assert.ErrorIs(t, routerErr.Tried[0].Err, ir.ErrRateLimited)

	assert.Equal(t, "prov2", routerErr.Tried[1].Provider)
	assert.Equal(t, "acc-2", routerErr.Tried[1].AccountID)
	assert.ErrorIs(t, routerErr.Tried[1].Err, ir.ErrProviderUnavailable)

	// Error string should contain provider info.
	assert.Contains(t, err.Error(), "prov1")
	assert.Contains(t, err.Error(), "prov2")
	assert.Contains(t, err.Error(), "acc-1")
	assert.Contains(t, err.Error(), "acc-2")
}

// Test: ErrAllFailed includes quota reserve failures
func TestAllFailed_IncludesQuotaErrors(t *testing.T) {
	mockProv := mock.New(mock.WithModels("test-model"))

	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "acc-1", DailyFree: 1, QuotaUnit: ir.QuotaTokens}, // too little quota
		},
	}

	r := newTestRouter(t, cfg, []ir.Provider{mockProv})

	_, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hello"}},
	})
	require.Error(t, err)

	var routerErr *ir.RouterError
	require.ErrorAs(t, err, &routerErr)
	assert.Len(t, routerErr.Tried, 1)
	assert.Equal(t, "acc-1", routerErr.Tried[0].AccountID)
}

// Test: HealthTracker.Reset clears all state
func TestHealthTracker_Reset(t *testing.T) {
	ht := ir.NewHealthTracker()

	// Trip circuit breakers.
	for i := 0; i < 3; i++ {
		ht.RecordFailure("acc-1")
		ht.RecordFailure("acc-2")
	}
	assert.Equal(t, ir.HealthUnhealthy, ht.GetHealth("acc-1"))
	assert.Equal(t, ir.HealthUnhealthy, ht.GetHealth("acc-2"))

	// Reset all.
	ht.Reset()
	assert.Equal(t, ir.HealthHealthy, ht.GetHealth("acc-1"))
	assert.Equal(t, ir.HealthHealthy, ht.GetHealth("acc-2"))
}

// Test: HealthTracker.ResetAccount clears single account
func TestHealthTracker_ResetAccount(t *testing.T) {
	ht := ir.NewHealthTracker()

	for i := 0; i < 3; i++ {
		ht.RecordFailure("acc-1")
		ht.RecordFailure("acc-2")
	}
	assert.Equal(t, ir.HealthUnhealthy, ht.GetHealth("acc-1"))
	assert.Equal(t, ir.HealthUnhealthy, ht.GetHealth("acc-2"))

	// Reset only acc-1.
	ht.ResetAccount("acc-1")
	assert.Equal(t, ir.HealthHealthy, ht.GetHealth("acc-1"))
	assert.Equal(t, ir.HealthUnhealthy, ht.GetHealth("acc-2"))
}

// --- Fix #1: Stream Close returns quota error, not body error ---

func TestStream_Close_ReturnsQuotaError(t *testing.T) {
	mockProv := mock.New(mock.WithModels("test-model"))
	failQS := &failingCommitQuotaStore{}

	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "free-1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
		},
	}

	r, err := ir.NewRouter(cfg, []ir.Provider{mockProv},
		ir.WithQuotaStore(failQS),
	)
	require.NoError(t, err)

	stream, err := r.ChatCompletionStream(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hello"}},
	})
	require.NoError(t, err)

	// Drain the stream.
	for {
		_, err := stream.Next()
		if err != nil {
			break
		}
	}

	// Close should return quota error, not nil.
	closeErr := stream.Close()
	require.Error(t, closeErr)
	assert.Contains(t, closeErr.Error(), "quota operation failed")
}

// --- Fix #2: Commit/Rollback errors reported through Meter ---

func TestCommitError_ReportedViaMeter(t *testing.T) {
	mockProv := mock.New(mock.WithModels("test-model"))
	spy := &meterSpy{}
	failQS := &failingCommitQuotaStore{}

	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "free-1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
		},
	}

	r, err := ir.NewRouter(cfg, []ir.Provider{mockProv},
		ir.WithQuotaStore(failQS),
		ir.WithMeter(spy),
	)
	require.NoError(t, err)

	// ChatCompletion succeeds at provider level but commit fails.
	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hello"}},
	})
	// Response is still returned to caller (provider succeeded).
	require.NoError(t, err)

	// But meter should report the commit failure.
	assert.False(t, spy.lastResult.Success)
	require.NotNil(t, spy.lastResult.Error)
	assert.Contains(t, spy.lastResult.Error.Error(), "quota commit failed")
}

func TestRollbackError_ReportedViaMeter(t *testing.T) {
	failProv := mock.New(
		mock.WithModels("test-model"),
		mock.WithError(ir.ErrRateLimited),
	)
	spy := &meterSpy{}
	failRollbackQS := &failingRollbackQuotaStore{}

	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "free-1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
		},
	}

	r, err := ir.NewRouter(cfg, []ir.Provider{failProv},
		ir.WithQuotaStore(failRollbackQS),
		ir.WithMeter(spy),
	)
	require.NoError(t, err)

	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hello"}},
	})
	require.Error(t, err)

	// Meter should report rollback failure in the error.
	assert.False(t, spy.lastResult.Success)
	require.NotNil(t, spy.lastResult.Error)
	assert.Contains(t, spy.lastResult.Error.Error(), "rollback failed")
}

// --- Fix #3: Remaining() fail-open for free tier ---

func TestRemainingError_FailOpen_FreeTier(t *testing.T) {
	mockProv := mock.New(mock.WithModels("test-model"))
	failRemainQS := &failingRemainingQuotaStore{}

	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "free-1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
		},
	}

	r, err := ir.NewRouter(cfg, []ir.Provider{mockProv},
		ir.WithQuotaStore(failRemainQS),
	)
	require.NoError(t, err)

	// Even though Remaining() fails, the account should still be tried (fail-open).
	resp, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hello"}},
	})
	require.NoError(t, err)
	assert.Equal(t, "free-1", resp.Routing.AccountID)
	assert.True(t, resp.Routing.Free)
}

// --- Fix #4/#5: SetQuota error propagation ---

func TestNewRouter_SetQuotaError_Propagated(t *testing.T) {
	mockProv := mock.New(mock.WithModels("test-model"))
	failInitQS := &failingSetQuotaStore{}

	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "free-1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
		},
	}

	_, err := ir.NewRouter(cfg, []ir.Provider{mockProv},
		ir.WithQuotaStore(failInitQS),
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "init quota")
	assert.Contains(t, err.Error(), "free-1")
}

// --- Test helpers ---

type meterSpy struct {
	lastRoute  ir.RouteEvent
	lastResult ir.ResultEvent
}

func (m *meterSpy) OnRoute(e ir.RouteEvent)   { m.lastRoute = e }
func (m *meterSpy) OnResult(e ir.ResultEvent)  { m.lastResult = e }

type failingCommitQuotaStore struct{}

func (f *failingCommitQuotaStore) Reserve(_ context.Context, accountID string, amount int64, unit ir.QuotaUnit, _ string) (ir.Reservation, error) {
	return ir.Reservation{ID: "test", AccountID: accountID, Amount: amount, Unit: unit}, nil
}
func (f *failingCommitQuotaStore) Commit(context.Context, ir.Reservation, int64) error {
	return errors.New("commit failed")
}
func (f *failingCommitQuotaStore) Rollback(context.Context, ir.Reservation) error { return nil }
func (f *failingCommitQuotaStore) Remaining(context.Context, string) (int64, error) {
	return 1000, nil
}

type failingRollbackQuotaStore struct{}

func (f *failingRollbackQuotaStore) Reserve(_ context.Context, accountID string, amount int64, unit ir.QuotaUnit, _ string) (ir.Reservation, error) {
	return ir.Reservation{ID: "test", AccountID: accountID, Amount: amount, Unit: unit}, nil
}
func (f *failingRollbackQuotaStore) Commit(context.Context, ir.Reservation, int64) error {
	return nil
}
func (f *failingRollbackQuotaStore) Rollback(context.Context, ir.Reservation) error {
	return errors.New("rollback failed")
}
func (f *failingRollbackQuotaStore) Remaining(context.Context, string) (int64, error) {
	return 1000, nil
}

type failingRemainingQuotaStore struct{}

func (f *failingRemainingQuotaStore) Reserve(_ context.Context, accountID string, amount int64, unit ir.QuotaUnit, _ string) (ir.Reservation, error) {
	return ir.Reservation{ID: "test", AccountID: accountID, Amount: amount, Unit: unit}, nil
}
func (f *failingRemainingQuotaStore) Commit(context.Context, ir.Reservation, int64) error {
	return nil
}
func (f *failingRemainingQuotaStore) Rollback(context.Context, ir.Reservation) error { return nil }
func (f *failingRemainingQuotaStore) Remaining(context.Context, string) (int64, error) {
	return 0, errors.New("redis timeout")
}

type failingSetQuotaStore struct{}

func (f *failingSetQuotaStore) Reserve(_ context.Context, accountID string, amount int64, unit ir.QuotaUnit, _ string) (ir.Reservation, error) {
	return ir.Reservation{ID: "test", AccountID: accountID, Amount: amount, Unit: unit}, nil
}
func (f *failingSetQuotaStore) Commit(context.Context, ir.Reservation, int64) error { return nil }
func (f *failingSetQuotaStore) Rollback(context.Context, ir.Reservation) error      { return nil }
func (f *failingSetQuotaStore) Remaining(context.Context, string) (int64, error)    { return 1000, nil }
func (f *failingSetQuotaStore) SetQuota(string, int64, ir.QuotaUnit) error {
	return errors.New("postgres connection refused")
}

// --- RPM Rate Limiting Tests ---

func TestRPM_SkipsRateLimitedCandidate(t *testing.T) {
	mockProv := mock.New(mock.WithModels("test-model"))

	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "limited", DailyFree: 1000, QuotaUnit: ir.QuotaTokens, RPM: 1},
			{Provider: "mock", ID: "unlimited", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
		},
	}

	r := newTestRouter(t, cfg, []ir.Provider{mockProv})

	// First request uses "limited" (free-first sorts by remaining, both equal).
	resp1, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hello"}},
	})
	require.NoError(t, err)
	assert.Equal(t, "limited", resp1.Routing.AccountID)

	// Second request — "limited" is at RPM cap, falls back to "unlimited".
	resp2, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hello again"}},
	})
	require.NoError(t, err)
	assert.Equal(t, "unlimited", resp2.Routing.AccountID)
}

func TestRPM_AllCandidatesLimited(t *testing.T) {
	mockProv := mock.New(mock.WithModels("test-model"))

	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "acc1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens, RPM: 1},
		},
	}

	r := newTestRouter(t, cfg, []ir.Provider{mockProv})

	// First request succeeds.
	_, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hello"}},
	})
	require.NoError(t, err)

	// Second request — all candidates RPM-limited.
	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hello again"}},
	})
	require.Error(t, err)

	var routerErr *ir.RouterError
	require.True(t, errors.As(err, &routerErr))
	require.Len(t, routerErr.Tried, 1)
	assert.True(t, errors.Is(routerErr.Tried[0].Err, ir.ErrRPMExceeded))
	assert.True(t, ir.IsRetryable(ir.ErrRPMExceeded))
}

func TestRPM_ZeroMeansUnlimited(t *testing.T) {
	mockProv := mock.New(mock.WithModels("test-model"))

	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "acc1", DailyFree: 10000, QuotaUnit: ir.QuotaTokens, RPM: 0},
		},
	}

	r := newTestRouter(t, cfg, []ir.Provider{mockProv})

	// Send many requests — all should succeed (no RPM limit).
	for i := 0; i < 50; i++ {
		_, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
			Messages: []ir.Message{{Role: "user", Content: "hello"}},
		})
		require.NoError(t, err)
	}
}

func TestRPM_NegativeRejected(t *testing.T) {
	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "acc1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens, RPM: -1},
		},
	}

	err := cfg.Validate()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "rpm must be >= 0")
}

func TestRPM_StreamingSameAsSync(t *testing.T) {
	mockProv := mock.New(mock.WithModels("test-model"))

	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "limited", DailyFree: 1000, QuotaUnit: ir.QuotaTokens, RPM: 1},
			{Provider: "mock", ID: "unlimited", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
		},
	}

	r := newTestRouter(t, cfg, []ir.Provider{mockProv})

	// First stream uses "limited".
	stream1, err := r.ChatCompletionStream(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hello"}},
	})
	require.NoError(t, err)
	// Drain and close stream.
	for {
		_, err := stream1.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			break
		}
	}
	stream1.Close()

	// Second stream — "limited" is at RPM cap, falls back to "unlimited".
	stream2, err := r.ChatCompletionStream(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hello again"}},
	})
	require.NoError(t, err)
	stream2.Close()
}
