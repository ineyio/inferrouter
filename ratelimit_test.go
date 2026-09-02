package inferrouter

import (
	"context"
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

// --- Pacing (Wait) ---
//
// Wait is the embedding path's answer to a burst: an embedding caller can
// absorb a pause, but cannot absorb a missing vector. These tests drive it
// with a fake clock and a fake sleeper, so the overlap they describe is
// forced rather than hoped for.

// Wait must block until the minute window frees a slot — and then let the
// request through. Both halves matter: a Wait that always blocked would pass
// the first assertion alone.
func TestRateLimiter_Wait_BlocksUntilMinuteWindowFrees(t *testing.T) {
	base := time.Date(2026, 9, 2, 10, 0, 0, 0, time.UTC)
	now := base

	rl := NewRateLimiter()
	rl.now = func() time.Time { return now }

	var slept []time.Duration
	rl.sleep = func(_ context.Context, d time.Duration) error {
		slept = append(slept, d)
		now = now.Add(d) // waking up means the clock really moved
		return nil
	}
	rl.SetAccountDefault("acc", Limits{RPM: 2})

	require.NoError(t, rl.Wait(context.Background(), "acc", "model"))
	now = now.Add(10 * time.Second)
	require.NoError(t, rl.Wait(context.Background(), "acc", "model"))

	// Third request is early, not over budget.
	require.NoError(t, rl.Wait(context.Background(), "acc", "model"))

	require.Len(t, slept, 1, "third request must pause exactly once")
	// The oldest slot was taken at base, so the window frees at base+1m —
	// 50s after the clock reached base+10s.
	assert.Equal(t, 50*time.Second, slept[0])
}

// A day window that is spent cannot be waited out inside any request's
// lifetime, so Wait must refuse immediately and say which window refused it.
func TestRateLimiter_Wait_DoesNotSleepOnDailyWindow(t *testing.T) {
	rl := NewRateLimiter()
	rl.now = func() time.Time { return time.Date(2026, 9, 2, 10, 0, 0, 0, time.UTC) }

	sleeps := 0
	rl.sleep = func(context.Context, time.Duration) error {
		sleeps++
		return nil
	}
	rl.SetAccountDefault("acc", Limits{RPM: 10, RPD: 1})

	require.NoError(t, rl.Wait(context.Background(), "acc", "model"))

	err := rl.Wait(context.Background(), "acc", "model")
	require.ErrorIs(t, err, ErrRateWindowExhausted)
	assert.NotErrorIs(t, err, ErrRPMExceeded, "a day refusal must not claim to be a minute refusal")
	assert.Contains(t, err.Error(), "day window")
	assert.Zero(t, sleeps, "an exhausted day allowance must not park the caller")
}

// A cancelled context ends the wait, rather than the wait outliving the
// request that asked for it.
func TestRateLimiter_Wait_HonoursContextCancellation(t *testing.T) {
	rl := NewRateLimiter()
	rl.now = func() time.Time { return time.Date(2026, 9, 2, 10, 0, 0, 0, time.UTC) }
	rl.sleep = realSleep
	rl.SetAccountDefault("acc", Limits{RPM: 1})

	require.NoError(t, rl.Wait(context.Background(), "acc", "model"))

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	assert.ErrorIs(t, rl.Wait(ctx, "acc", "model"), context.Canceled)
}

// A background worker's context has no deadline of its own, so Wait carries
// its own ceiling: under sustained contention it gives up rather than parking
// the worker forever.
func TestRateLimiter_Wait_GivesUpAfterMaxRounds(t *testing.T) {
	rl := NewRateLimiter()
	// Clock never advances, so the window never frees.
	rl.now = func() time.Time { return time.Date(2026, 9, 2, 10, 0, 0, 0, time.UTC) }

	sleeps := 0
	rl.sleep = func(context.Context, time.Duration) error {
		sleeps++
		return nil
	}
	rl.SetAccountDefault("acc", Limits{RPM: 1})

	require.NoError(t, rl.Wait(context.Background(), "acc", "model"))
	assert.ErrorIs(t, rl.Wait(context.Background(), "acc", "model"), ErrRPMExceeded)
	assert.Equal(t, maxWaitRounds, sleeps)
}
