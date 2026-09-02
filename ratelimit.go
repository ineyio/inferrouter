package inferrouter

import (
	"context"
	"fmt"
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

// limitWindow names the window a request breached. A refusal is not one fact
// but two — that the request was refused, and by which window — because the
// answers differ by orders of magnitude: a minute window frees a slot in
// hundreds of milliseconds, a day window in hours. Wait can only pace the
// first kind.
type limitWindow int

const (
	windowNone limitWindow = iota
	windowMinute
	windowHour
	windowDay
)

func (w limitWindow) String() string {
	switch w {
	case windowMinute:
		return "minute"
	case windowHour:
		return "hour"
	case windowDay:
		return "day"
	default:
		return "none"
	}
}

// Sleeper pauses for d and returns nil, or returns ctx.Err() if the context
// ends first. Injectable so that pacing can be tested without real time.
type Sleeper func(ctx context.Context, d time.Duration) error

// realSleep is the default Sleeper: a timer the context can cut short.
func realSleep(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			return nil
		}
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// maxWaitRounds caps how many times Wait sleeps before giving up. The minute
// window frees a slot within a minute, so two rounds is already generous; the
// cap exists because a background worker's context has no deadline of its own
// and an unbounded loop would park it forever under sustained contention.
const maxWaitRounds = 2

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
	sleep           Sleeper
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
		sleep:           realSleep,
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

// SetSleeper overrides how Wait pauses. Production code never calls it; it
// exists so that pacing can be exercised in tests without spending the wall
// clock the pacing is measured in.
func (rl *RateLimiter) SetSleeper(s Sleeper) {
	if s == nil {
		return
	}
	rl.mu.Lock()
	defer rl.mu.Unlock()
	rl.sleep = s
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
	ok, _, _ := rl.check(accountID, model)
	return ok
}

// Wait is Allow for callers that can afford to be paced instead of refused.
//
// A refusal from the minute window means the caller is early, not over budget:
// it sleeps until the oldest request occupying a slot ages out and tries
// again. A refusal from the hour or day window means the allowance is spent,
// and no reachable amount of waiting brings it back, so Wait returns
// ErrRateWindowExhausted immediately rather than parking the caller for hours.
//
// Waiting always respects ctx and never holds the lock while it sleeps.
func (rl *RateLimiter) Wait(ctx context.Context, accountID, model string) error {
	for round := 0; ; round++ {
		if err := ctx.Err(); err != nil {
			return err
		}

		ok, window, retryAt := rl.check(accountID, model)
		if ok {
			return nil
		}
		if window != windowMinute {
			return fmt.Errorf("%w: %s window", ErrRateWindowExhausted, window)
		}
		if round >= maxWaitRounds {
			return ErrRPMExceeded
		}

		if err := rl.sleeper()(ctx, retryAt.Sub(rl.now())); err != nil {
			return err
		}
	}
}

// sleeper reads the current Sleeper under the lock, so that SetSleeper stays
// safe to call alongside a Wait already in flight.
func (rl *RateLimiter) sleeper() Sleeper {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	return rl.sleep
}

// check resolves the window for (account, model), records the request if it
// fits, and reports which window refused it otherwise.
func (rl *RateLimiter) check(accountID, model string) (bool, limitWindow, time.Time) {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	key := accountID + ":" + model
	w, ok := rl.windows[key]
	if !ok {
		// No model-specific limits — check account defaults.
		defaults, hasDefault := rl.accountDefaults[accountID]
		if !hasDefault || defaults.IsZero() {
			return true, windowNone, time.Time{} // no limits configured
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
//
// Windows are checked widest first, and the widest breached window is the one
// reported: when the day allowance is spent, freeing a minute slot unblocks
// nothing, so naming the minute window would send a waiter to sleep for a
// refusal that sleeping cannot fix.
func (w *multiWindow) allow(now time.Time) (bool, limitWindow, time.Time) {
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
		return false, windowDay, w.slotFreesAt(w.limits.RPD, 24*time.Hour)
	}

	// Check RPH (1h window).
	if w.limits.RPH > 0 && countAfter(w.times, now.Add(-time.Hour)) >= w.limits.RPH {
		return false, windowHour, w.slotFreesAt(w.limits.RPH, time.Hour)
	}

	// Check RPM (1min window).
	if w.limits.RPM > 0 && countAfter(w.times, now.Add(-time.Minute)) >= w.limits.RPM {
		return false, windowMinute, w.slotFreesAt(w.limits.RPM, time.Minute)
	}

	w.times = append(w.times, now)
	return true, windowNone, time.Time{}
}

// slotFreesAt returns the moment a window of length d and limit n frees a
// slot: the n-th newest request is the oldest one still occupying one, so the
// window has room again once it ages out.
func (w *multiWindow) slotFreesAt(n int, d time.Duration) time.Time {
	if n <= 0 || len(w.times) < n {
		return time.Time{}
	}
	return w.times[len(w.times)-n].Add(d)
}

// countAfter counts timestamps strictly newer than cutoff.
func countAfter(times []time.Time, cutoff time.Time) int {
	count := 0
	for _, t := range times {
		if t.After(cutoff) {
			count++
		}
	}
	return count
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
