package inferrouter

import (
	"sync"
	"time"
)

// HealthConfig configures circuit breaker behavior.
type HealthConfig struct {
	FailureThreshold int           // failures to trip circuit (default: 3)
	FailureWindow    time.Duration // window for counting failures (default: 5min)
	UnhealthyPeriod  time.Duration // cooldown before half-open (default: 30s)
}

// DefaultHealthConfig returns the default circuit breaker settings.
func DefaultHealthConfig() HealthConfig {
	return HealthConfig{
		FailureThreshold: 3,
		FailureWindow:    5 * time.Minute,
		UnhealthyPeriod:  30 * time.Second,
	}
}

// HealthTracker tracks per-account health using a circuit breaker pattern.
type HealthTracker struct {
	cfg      HealthConfig
	mu       sync.RWMutex
	accounts map[string]*accountHealth
}

type accountHealth struct {
	state       HealthState
	failures    []time.Time // sliding window of failure timestamps
	unhealthyAt time.Time   // when state transitioned to unhealthy
}

// NewHealthTracker creates a new HealthTracker with default config.
func NewHealthTracker() *HealthTracker {
	return NewHealthTrackerWithConfig(DefaultHealthConfig())
}

// NewHealthTrackerWithConfig creates a new HealthTracker with custom config.
func NewHealthTrackerWithConfig(cfg HealthConfig) *HealthTracker {
	return &HealthTracker{
		cfg:      cfg,
		accounts: make(map[string]*accountHealth),
	}
}

// GetHealth returns the current health state for an account.
func (h *HealthTracker) GetHealth(accountID string) HealthState {
	h.mu.Lock()
	defer h.mu.Unlock()

	ah, ok := h.accounts[accountID]
	if !ok {
		return HealthHealthy
	}

	// Check if unhealthy period has elapsed → transition to half-open.
	ah.state = h.effectiveState(ah, time.Now())

	return ah.state
}

// effectiveState is the one rule for half-open, shared by GetHealth, which
// stores the transition, and Peek, which must not.
func (h *HealthTracker) effectiveState(ah *accountHealth, now time.Time) HealthState {
	if ah.state == HealthUnhealthy && now.Sub(ah.unhealthyAt) >= h.cfg.UnhealthyPeriod {
		return HealthHalfOpen
	}
	return ah.state
}

// HealthStatus is a read-only view of one account's circuit breaker.
type HealthStatus struct {
	State HealthState

	// FailuresInWindow counts failures still inside the failure window. The
	// breaker does not clear them when it trips, so one failure in half-open
	// reopens it.
	FailuresInWindow int

	// UnhealthySince is when the breaker last tripped; zero while Healthy.
	UnhealthySince time.Time
}

// Peek reports an account's breaker without changing it. GetHealth stores the
// transition to half-open, so an observer calling it would change what the
// next caller sees.
func (h *HealthTracker) Peek(accountID string) HealthStatus {
	h.mu.RLock()
	defer h.mu.RUnlock()

	ah, ok := h.accounts[accountID]
	if !ok {
		return HealthStatus{State: HealthHealthy}
	}

	now := time.Now()
	status := HealthStatus{State: h.effectiveState(ah, now)}
	if status.State != HealthHealthy {
		status.UnhealthySince = ah.unhealthyAt
	}
	cutoff := now.Add(-h.cfg.FailureWindow)
	for _, t := range ah.failures {
		if t.After(cutoff) {
			status.FailuresInWindow++
		}
	}
	return status
}

// RecordSuccess records a successful request for an account.
func (h *HealthTracker) RecordSuccess(accountID string) {
	h.mu.Lock()
	defer h.mu.Unlock()

	ah := h.getOrCreate(accountID)
	ah.state = HealthHealthy
	ah.failures = ah.failures[:0]
}

// RecordFailure records a failed request for an account.
func (h *HealthTracker) RecordFailure(accountID string) {
	h.mu.Lock()
	defer h.mu.Unlock()

	ah := h.getOrCreate(accountID)
	if ah.state == HealthUnhealthy {
		return
	}

	now := time.Now()

	// Prune old failures outside the window.
	cutoff := now.Add(-h.cfg.FailureWindow)
	valid := ah.failures[:0]
	for _, t := range ah.failures {
		if t.After(cutoff) {
			valid = append(valid, t)
		}
	}
	ah.failures = append(valid, now)

	// Check threshold.
	if len(ah.failures) >= h.cfg.FailureThreshold {
		ah.state = HealthUnhealthy
		ah.unhealthyAt = now
	}
}

// Reset clears health state for all accounts, returning them to healthy.
func (h *HealthTracker) Reset() {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.accounts = make(map[string]*accountHealth)
}

// ResetAccount clears health state for a single account.
func (h *HealthTracker) ResetAccount(accountID string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	delete(h.accounts, accountID)
}

func (h *HealthTracker) getOrCreate(accountID string) *accountHealth {
	ah, ok := h.accounts[accountID]
	if !ok {
		ah = &accountHealth{state: HealthHealthy}
		h.accounts[accountID] = ah
	}
	return ah
}
