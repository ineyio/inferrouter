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
	h.mu.RLock()
	ah, ok := h.accounts[accountID]
	h.mu.RUnlock()

	if !ok {
		return HealthHealthy
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	// Check if unhealthy period has elapsed → transition to half-open.
	if ah.state == HealthUnhealthy && time.Since(ah.unhealthyAt) >= h.cfg.UnhealthyPeriod {
		ah.state = HealthHalfOpen
	}

	return ah.state
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

func (h *HealthTracker) getOrCreate(accountID string) *accountHealth {
	ah, ok := h.accounts[accountID]
	if !ok {
		ah = &accountHealth{state: HealthHealthy}
		h.accounts[accountID] = ah
	}
	return ah
}
