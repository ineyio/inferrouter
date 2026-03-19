package inferrouter

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRateLimiter_AllowUnderLimit(t *testing.T) {
	rl := NewRateLimiter()
	rl.SetLimit("acc1", 5)

	for i := 0; i < 5; i++ {
		assert.True(t, rl.Allow("acc1"), "request %d should be allowed", i+1)
	}
}

func TestRateLimiter_BlockAtLimit(t *testing.T) {
	rl := NewRateLimiter()
	rl.SetLimit("acc1", 3)

	assert.True(t, rl.Allow("acc1"))
	assert.True(t, rl.Allow("acc1"))
	assert.True(t, rl.Allow("acc1"))
	assert.False(t, rl.Allow("acc1"), "4th request should be blocked")
	assert.False(t, rl.Allow("acc1"), "5th request should also be blocked")
}

func TestRateLimiter_SlidingWindowExpiry(t *testing.T) {
	rl := NewRateLimiter()
	rl.SetLimit("acc1", 2)

	now := time.Now()
	rl.now = func() time.Time { return now }

	assert.True(t, rl.Allow("acc1"))
	assert.True(t, rl.Allow("acc1"))
	assert.False(t, rl.Allow("acc1"), "at limit")

	// Advance time by 61 seconds — old requests should expire.
	rl.now = func() time.Time { return now.Add(61 * time.Second) }

	assert.True(t, rl.Allow("acc1"), "should be allowed after window expires")
	assert.True(t, rl.Allow("acc1"))
	assert.False(t, rl.Allow("acc1"), "at limit again")
}

func TestRateLimiter_PartialExpiry(t *testing.T) {
	rl := NewRateLimiter()
	rl.SetLimit("acc1", 3)

	now := time.Now()
	rl.now = func() time.Time { return now }

	assert.True(t, rl.Allow("acc1")) // t=0

	// Advance 30s, add another
	rl.now = func() time.Time { return now.Add(30 * time.Second) }
	assert.True(t, rl.Allow("acc1")) // t=30s

	// Advance 40s more (t=70s) — first request expired, second still valid
	rl.now = func() time.Time { return now.Add(70 * time.Second) }
	assert.True(t, rl.Allow("acc1"))  // t=70s (window has t=30s and t=70s)
	assert.True(t, rl.Allow("acc1"))  // t=70s (window has t=30s, t=70s, t=70s)
	assert.False(t, rl.Allow("acc1")) // at limit
}

func TestRateLimiter_UnknownAccountAllowed(t *testing.T) {
	rl := NewRateLimiter()
	rl.SetLimit("acc1", 1)

	// Unknown account has no limit → always allowed.
	assert.True(t, rl.Allow("unknown"))
	assert.True(t, rl.Allow("unknown"))
	assert.True(t, rl.Allow("unknown"))
}

func TestRateLimiter_MultipleAccounts(t *testing.T) {
	rl := NewRateLimiter()
	rl.SetLimit("acc1", 1)
	rl.SetLimit("acc2", 2)

	assert.True(t, rl.Allow("acc1"))
	assert.False(t, rl.Allow("acc1"), "acc1 at limit")

	assert.True(t, rl.Allow("acc2"))
	assert.True(t, rl.Allow("acc2"))
	assert.False(t, rl.Allow("acc2"), "acc2 at limit")
}

func TestRateLimiter_Reset(t *testing.T) {
	rl := NewRateLimiter()
	rl.SetLimit("acc1", 1)

	assert.True(t, rl.Allow("acc1"))
	assert.False(t, rl.Allow("acc1"))

	rl.Reset()

	assert.True(t, rl.Allow("acc1"), "should be allowed after reset")
}

func TestRateLimiter_ResetAccount(t *testing.T) {
	rl := NewRateLimiter()
	rl.SetLimit("acc1", 1)
	rl.SetLimit("acc2", 1)

	assert.True(t, rl.Allow("acc1"))
	assert.True(t, rl.Allow("acc2"))
	assert.False(t, rl.Allow("acc1"))
	assert.False(t, rl.Allow("acc2"))

	rl.ResetAccount("acc1")

	assert.True(t, rl.Allow("acc1"), "acc1 should be allowed after reset")
	assert.False(t, rl.Allow("acc2"), "acc2 should still be limited")
}

func TestRateLimiter_ConcurrentAccess(t *testing.T) {
	rl := NewRateLimiter()
	rl.SetLimit("acc1", 100)

	var allowed atomic.Int64
	var wg sync.WaitGroup

	for i := 0; i < 200; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if rl.Allow("acc1") {
				allowed.Add(1)
			}
		}()
	}

	wg.Wait()

	require.Equal(t, int64(100), allowed.Load(), "exactly 100 requests should be allowed")
}
