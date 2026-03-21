package inferrouter

import (
	"sync"
	"time"
)

// Limits defines rate limits for a provider account or model.
// Zero values mean unlimited for that window.
type Limits struct {
	RPM int `yaml:"rpm"` // requests per minute
	RPH int `yaml:"rph"` // requests per hour
	RPD int `yaml:"rpd"` // requests per day
}

// IsZero returns true if no limits are configured.
func (l Limits) IsZero() bool {
	return l.RPM == 0 && l.RPH == 0 && l.RPD == 0
}

// RateLimiter enforces per-(account, model) rate limits using sliding windows.
// Thread-safe. Supports RPM, RPH, and RPD simultaneously.
//
// Lookup order: model-specific limits first, then account-level defaults.
// This allows Cerebras-style configs where each model has independent limits,
// and simpler configs where one RPM applies to all models on an account.
type RateLimiter struct {
	mu              sync.Mutex
	windows         map[string]*multiWindow // key: "accountID:model"
	accountDefaults map[string]Limits       // key: accountID
	now             func() time.Time
}

type multiWindow struct {
	limits Limits
	times  []time.Time
}

// NewRateLimiter creates a new RateLimiter.
func NewRateLimiter() *RateLimiter {
	return &RateLimiter{
		windows:         make(map[string]*multiWindow),
		accountDefaults: make(map[string]Limits),
		now:             time.Now,
	}
}

// SetModelLimits configures rate limits for a specific (account, model) pair.
func (rl *RateLimiter) SetModelLimits(accountID, model string, limits Limits) {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	key := accountID + ":" + model
	rl.windows[key] = &multiWindow{
		limits: limits,
		times:  make([]time.Time, 0, max(limits.RPM, 16)),
	}
}

// SetAccountDefault configures fallback rate limits for models without explicit limits.
func (rl *RateLimiter) SetAccountDefault(accountID string, limits Limits) {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	rl.accountDefaults[accountID] = limits
}

// SetLimit is a convenience method for backward compatibility.
// Equivalent to SetAccountDefault with RPM only.
func (rl *RateLimiter) SetLimit(accountID string, rpm int) {
	rl.SetAccountDefault(accountID, Limits{RPM: rpm})
}

// Allow checks if a request is permitted for the given (account, model) pair.
// Checks model-specific limits first. If none configured, falls back to account defaults.
// Returns true and records the request if under all limits.
// Returns false if any limit is exceeded.
func (rl *RateLimiter) Allow(accountID, model string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	key := accountID + ":" + model
	w, ok := rl.windows[key]
	if !ok {
		// No model-specific limits — check account defaults.
		defaults, hasDefault := rl.accountDefaults[accountID]
		if !hasDefault || defaults.IsZero() {
			return true // no limits configured
		}
		// Lazily create window from account defaults.
		w = &multiWindow{
			limits: defaults,
			times:  make([]time.Time, 0, max(defaults.RPM, 16)),
		}
		rl.windows[key] = w
	}

	return w.allow(rl.now())
}

// allow checks all time windows and records the request if permitted.
// Must be called with rl.mu held.
func (w *multiWindow) allow(now time.Time) bool {
	// Prune entries older than the longest window (24h for RPD).
	maxWindow := time.Minute
	if w.limits.RPH > 0 {
		maxWindow = time.Hour
	}
	if w.limits.RPD > 0 {
		maxWindow = 24 * time.Hour
	}

	cutoff := now.Add(-maxWindow)
	valid := w.times[:0]
	for _, t := range w.times {
		if t.After(cutoff) {
			valid = append(valid, t)
		}
	}
	w.times = valid

	// Check RPD (24h window).
	if w.limits.RPD > 0 && len(w.times) >= w.limits.RPD {
		return false
	}

	// Check RPH (1h window).
	if w.limits.RPH > 0 {
		hourCutoff := now.Add(-time.Hour)
		hourCount := 0
		for _, t := range w.times {
			if t.After(hourCutoff) {
				hourCount++
			}
		}
		if hourCount >= w.limits.RPH {
			return false
		}
	}

	// Check RPM (1min window).
	if w.limits.RPM > 0 {
		minCutoff := now.Add(-time.Minute)
		minCount := 0
		for _, t := range w.times {
			if t.After(minCutoff) {
				minCount++
			}
		}
		if minCount >= w.limits.RPM {
			return false
		}
	}

	w.times = append(w.times, now)
	return true
}

// Reset clears all rate limiter state (preserves configured limits).
func (rl *RateLimiter) Reset() {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	for key, w := range rl.windows {
		rl.windows[key] = &multiWindow{
			limits: w.limits,
			times:  make([]time.Time, 0, max(w.limits.RPM, 16)),
		}
	}
}

// ResetAccount clears state for all models under an account.
func (rl *RateLimiter) ResetAccount(accountID string) {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	for key, w := range rl.windows {
		if len(key) > len(accountID) && key[:len(accountID)+1] == accountID+":" {
			rl.windows[key] = &multiWindow{
				limits: w.limits,
				times:  make([]time.Time, 0, max(w.limits.RPM, 16)),
			}
		}
	}
}
