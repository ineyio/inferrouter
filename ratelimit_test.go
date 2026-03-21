package inferrouter

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRateLimiter_RPM_AllowUnderLimit(t *testing.T) {
	rl := NewRateLimiter()
	rl.SetAccountDefault("acc1", Limits{RPM: 5})

	for i := 0; i < 5; i++ {
		assert.True(t, rl.Allow("acc1", "model-a"), "request %d should be allowed", i+1)
	}
}

func TestRateLimiter_RPM_BlockAtLimit(t *testing.T) {
	rl := NewRateLimiter()
	rl.SetAccountDefault("acc1", Limits{RPM: 3})

	assert.True(t, rl.Allow("acc1", "model-a"))
	assert.True(t, rl.Allow("acc1", "model-a"))
	assert.True(t, rl.Allow("acc1", "model-a"))
	assert.False(t, rl.Allow("acc1", "model-a"), "4th request should be blocked")
}

func TestRateLimiter_RPM_SlidingWindowExpiry(t *testing.T) {
	rl := NewRateLimiter()
	rl.SetAccountDefault("acc1", Limits{RPM: 2})

	now := time.Now()
	rl.now = func() time.Time { return now }

	assert.True(t, rl.Allow("acc1", "m"))
	assert.True(t, rl.Allow("acc1", "m"))
	assert.False(t, rl.Allow("acc1", "m"))

	// Advance 61s — old requests expire.
	rl.now = func() time.Time { return now.Add(61 * time.Second) }
	assert.True(t, rl.Allow("acc1", "m"))
}

func TestRateLimiter_RPH(t *testing.T) {
	rl := NewRateLimiter()
	rl.SetModelLimits("acc1", "m", Limits{RPH: 3})

	now := time.Now()
	rl.now = func() time.Time { return now }

	assert.True(t, rl.Allow("acc1", "m"))
	assert.True(t, rl.Allow("acc1", "m"))
	assert.True(t, rl.Allow("acc1", "m"))
	assert.False(t, rl.Allow("acc1", "m"), "RPH limit reached")

	// Advance 61 minutes — RPH window expires.
	rl.now = func() time.Time { return now.Add(61 * time.Minute) }
	assert.True(t, rl.Allow("acc1", "m"))
}

func TestRateLimiter_RPD(t *testing.T) {
	rl := NewRateLimiter()
	rl.SetModelLimits("acc1", "m", Limits{RPD: 5})

	now := time.Now()
	rl.now = func() time.Time { return now }

	for i := 0; i < 5; i++ {
		assert.True(t, rl.Allow("acc1", "m"))
	}
	assert.False(t, rl.Allow("acc1", "m"), "RPD limit reached")

	// Advance 25 hours — RPD window expires.
	rl.now = func() time.Time { return now.Add(25 * time.Hour) }
	assert.True(t, rl.Allow("acc1", "m"))
}

func TestRateLimiter_MultiWindow_RPM_Triggers_First(t *testing.T) {
	rl := NewRateLimiter()
	rl.SetModelLimits("acc1", "m", Limits{RPM: 2, RPH: 100, RPD: 1000})

	assert.True(t, rl.Allow("acc1", "m"))
	assert.True(t, rl.Allow("acc1", "m"))
	assert.False(t, rl.Allow("acc1", "m"), "RPM should trigger before RPH/RPD")
}

func TestRateLimiter_ModelSpecific_IndependentLimits(t *testing.T) {
	rl := NewRateLimiter()
	rl.SetModelLimits("acc1", "gpt-oss-120b", Limits{RPM: 1})
	rl.SetModelLimits("acc1", "qwen-3-235b", Limits{RPM: 1})

	// Each model has its own budget.
	assert.True(t, rl.Allow("acc1", "gpt-oss-120b"))
	assert.False(t, rl.Allow("acc1", "gpt-oss-120b"), "gpt at limit")

	assert.True(t, rl.Allow("acc1", "qwen-3-235b"), "qwen should be independent")
	assert.False(t, rl.Allow("acc1", "qwen-3-235b"), "qwen at limit")
}

func TestRateLimiter_AccountDefault_Fallback(t *testing.T) {
	rl := NewRateLimiter()
	rl.SetModelLimits("acc1", "model-a", Limits{RPM: 1})
	rl.SetAccountDefault("acc1", Limits{RPM: 2})

	// model-a uses its specific limit (1).
	assert.True(t, rl.Allow("acc1", "model-a"))
	assert.False(t, rl.Allow("acc1", "model-a"))

	// model-b falls back to account default (2).
	assert.True(t, rl.Allow("acc1", "model-b"))
	assert.True(t, rl.Allow("acc1", "model-b"))
	assert.False(t, rl.Allow("acc1", "model-b"))
}

func TestRateLimiter_UnknownAccount_Unlimited(t *testing.T) {
	rl := NewRateLimiter()
	rl.SetModelLimits("acc1", "m", Limits{RPM: 1})

	// Unknown account has no limits.
	for i := 0; i < 100; i++ {
		assert.True(t, rl.Allow("unknown", "m"))
	}
}

func TestRateLimiter_BackwardCompat_SetLimit(t *testing.T) {
	rl := NewRateLimiter()
	rl.SetLimit("acc1", 2) // old API

	assert.True(t, rl.Allow("acc1", "any-model"))
	assert.True(t, rl.Allow("acc1", "any-model"))
	assert.False(t, rl.Allow("acc1", "any-model"))
}

func TestRateLimiter_Reset(t *testing.T) {
	rl := NewRateLimiter()
	rl.SetAccountDefault("acc1", Limits{RPM: 1})

	assert.True(t, rl.Allow("acc1", "m"))
	assert.False(t, rl.Allow("acc1", "m"))

	rl.Reset()
	assert.True(t, rl.Allow("acc1", "m"))
}

func TestRateLimiter_ResetAccount(t *testing.T) {
	rl := NewRateLimiter()
	rl.SetAccountDefault("acc1", Limits{RPM: 1})
	rl.SetAccountDefault("acc2", Limits{RPM: 1})

	assert.True(t, rl.Allow("acc1", "m"))
	assert.True(t, rl.Allow("acc2", "m"))

	rl.ResetAccount("acc1")

	assert.True(t, rl.Allow("acc1", "m"), "acc1 reset")
	assert.False(t, rl.Allow("acc2", "m"), "acc2 still limited")
}

func TestRateLimiter_ConcurrentAccess(t *testing.T) {
	rl := NewRateLimiter()
	rl.SetAccountDefault("acc1", Limits{RPM: 100})

	var allowed atomic.Int64
	var wg sync.WaitGroup

	for i := 0; i < 200; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if rl.Allow("acc1", "m") {
				allowed.Add(1)
			}
		}()
	}

	wg.Wait()
	require.Equal(t, int64(100), allowed.Load())
}

func TestRateLimiter_CerebrasScenario(t *testing.T) {
	// Simulate: gpt-oss-120b hits RPD, fall back to qwen-3-235b.
	rl := NewRateLimiter()
	rl.SetModelLimits("cerebras-free", "gpt-oss-120b", Limits{RPM: 30, RPD: 5})
	rl.SetModelLimits("cerebras-free", "qwen-3-235b", Limits{RPM: 30, RPD: 5})

	// Exhaust gpt-oss-120b RPD.
	for i := 0; i < 5; i++ {
		assert.True(t, rl.Allow("cerebras-free", "gpt-oss-120b"))
	}
	assert.False(t, rl.Allow("cerebras-free", "gpt-oss-120b"), "gpt RPD exhausted")

	// qwen-3-235b still has budget.
	assert.True(t, rl.Allow("cerebras-free", "qwen-3-235b"), "qwen should be independent")
}
