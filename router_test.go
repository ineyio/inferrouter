package inferrouter_test

import (
	"context"
	"io"
	"sync"
	"testing"

	ir "github.com/ineyio/inferrouter"
	"github.com/ineyio/inferrouter/meter"
	"github.com/ineyio/inferrouter/policy"
	"github.com/ineyio/inferrouter/provider/mock"
	"github.com/ineyio/inferrouter/quota"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestRouter(t *testing.T, cfg ir.Config, providers []ir.Provider, qs ir.QuotaStore) *ir.Router {
	t.Helper()
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

	qs := quota.NewMemoryQuotaStore()
	qs.SetQuota("free-1", 1000, ir.QuotaTokens)

	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "free-1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
		},
	}

	r := newTestRouter(t, cfg, []ir.Provider{mockProv}, qs)

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

	qs := quota.NewMemoryQuotaStore()
	qs.SetQuota("free-1", 1, ir.QuotaTokens)   // almost exhausted
	qs.SetQuota("free-2", 1000, ir.QuotaTokens) // plenty

	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "free-1", DailyFree: 1, QuotaUnit: ir.QuotaTokens},
			{Provider: "mock", ID: "free-2", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
		},
	}

	r := newTestRouter(t, cfg, []ir.Provider{mockProv}, qs)

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

	qs := quota.NewMemoryQuotaStore()
	qs.SetQuota("free-1", 0, ir.QuotaTokens)

	cfg := ir.Config{
		AllowPaid:    false,
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "free-1", DailyFree: 1, QuotaUnit: ir.QuotaTokens},
		},
	}

	r := newTestRouter(t, cfg, []ir.Provider{mockProv}, qs)

	_, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hello"}},
	})
	assert.ErrorIs(t, err, ir.ErrNoCandidates)
}

// Test 4: Paid fallback when free exhausted + AllowPaid=true
func TestPaidFallback_WhenFreeExhausted(t *testing.T) {
	mockProv := mock.New(mock.WithModels("test-model"))

	qs := quota.NewMemoryQuotaStore()
	qs.SetQuota("free-1", 0, ir.QuotaTokens) // exhausted

	cfg := ir.Config{
		AllowPaid:    true,
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "free-1", DailyFree: 1, QuotaUnit: ir.QuotaTokens},
			{Provider: "mock", ID: "paid-1", DailyFree: 0, QuotaUnit: ir.QuotaTokens, PaidEnabled: true},
		},
	}

	r := newTestRouter(t, cfg, []ir.Provider{mockProv}, qs)

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

	qs := quota.NewMemoryQuotaStore()
	qs.SetQuota("free-1", 10000, ir.QuotaTokens)

	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "free-1", DailyFree: 10000, QuotaUnit: ir.QuotaTokens},
		},
	}

	r := newTestRouter(t, cfg, []ir.Provider{mockProv}, qs)

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

	qs := quota.NewMemoryQuotaStore()
	qs.SetQuota("gemini-1", 1000, ir.QuotaTokens)
	qs.SetQuota("grok-1", 1000, ir.QuotaTokens)

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

	r := newTestRouter(t, cfg, []ir.Provider{geminiProv, grokProv}, qs)

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

	qs := quota.NewMemoryQuotaStore()
	qs.SetQuota("free-1", 1000, ir.QuotaTokens)

	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "free-1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
		},
	}

	r := newTestRouter(t, cfg, []ir.Provider{mockProv}, qs)

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

	qs := quota.NewMemoryQuotaStore()
	qs.SetQuota("fail-1", 1000, ir.QuotaTokens)
	qs.SetQuota("good-1", 1000, ir.QuotaTokens)

	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "fail", ID: "fail-1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
			{Provider: "good", ID: "good-1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
		},
	}

	r := newTestRouter(t, cfg, []ir.Provider{failProv, goodProv}, qs)

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

	qs := quota.NewMemoryQuotaStore()
	qs.SetQuota("acc-1", 1000, ir.QuotaTokens)
	qs.SetQuota("acc-2", 1000, ir.QuotaTokens)

	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "acc-1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
			{Provider: "mock", ID: "acc-2", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
		},
	}

	r := newTestRouter(t, cfg, []ir.Provider{failProv}, qs)

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
