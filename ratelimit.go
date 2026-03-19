package inferrouter

import (
	"sync"
	"time"
)

// RateLimiter enforces per-account requests-per-minute limits using a sliding window.
// Thread-safe. Memory usage is O(RPM) per account (stores recent request timestamps).
type RateLimiter struct {
	mu       sync.Mutex
	accounts map[string]*rpmWindow
	now      func() time.Time // injectable for testing
}

type rpmWindow struct {
	limit int
	times []time.Time
}

// NewRateLimiter creates a new RateLimiter with no account limits configured.
func NewRateLimiter() *RateLimiter {
	return &RateLimiter{
		accounts: make(map[string]*rpmWindow),
		now:      time.Now,
	}
}

// SetLimit configures the RPM limit for an account.
// Called from NewRouter() during initialization.
func (rl *RateLimiter) SetLimit(accountID string, rpm int) {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	rl.accounts[accountID] = &rpmWindow{
		limit: rpm,
		times: make([]time.Time, 0, rpm),
	}
}

// Allow checks if a request is permitted under the account's RPM limit.
// Returns true and records the request if under limit.
// Returns false if at or over limit.
// Accounts without a configured limit are always allowed.
func (rl *RateLimiter) Allow(accountID string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	w, ok := rl.accounts[accountID]
	if !ok {
		return true // no limit configured
	}

	now := rl.now()
	cutoff := now.Add(-time.Minute)

	// Prune expired entries (older than 1 minute).
	valid := w.times[:0]
	for _, t := range w.times {
		if t.After(cutoff) {
			valid = append(valid, t)
		}
	}
	w.times = valid

	if len(w.times) >= w.limit {
		return false
	}

	w.times = append(w.times, now)
	return true
}

// Reset clears all rate limiter state.
func (rl *RateLimiter) Reset() {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	for id, w := range rl.accounts {
		rl.accounts[id] = &rpmWindow{
			limit: w.limit,
			times: make([]time.Time, 0, w.limit),
		}
	}
}

// ResetAccount clears rate limiter state for a single account.
func (rl *RateLimiter) ResetAccount(accountID string) {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	if w, ok := rl.accounts[accountID]; ok {
		rl.accounts[accountID] = &rpmWindow{
			limit: w.limit,
			times: make([]time.Time, 0, w.limit),
		}
	}
}
