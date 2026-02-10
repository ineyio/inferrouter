package inferrouter

import (
	"sync"
	"time"
)

const (
	healthFailureThreshold = 3
	healthFailureWindow    = 5 * time.Minute
	healthUnhealthyPeriod  = 30 * time.Second
)

// HealthTracker tracks per-account health using a circuit breaker pattern.
type HealthTracker struct {
	mu       sync.RWMutex
	accounts map[string]*accountHealth
}

type accountHealth struct {
	state      HealthState
	failures   []time.Time // sliding window of failure timestamps
	unhealthyAt time.Time  // when state transitioned to unhealthy
}

// NewHealthTracker creates a new HealthTracker.
func NewHealthTracker() *HealthTracker {
	return &HealthTracker{
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
	if ah.state == HealthUnhealthy && time.Since(ah.unhealthyAt) >= healthUnhealthyPeriod {
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
	cutoff := now.Add(-healthFailureWindow)
	valid := ah.failures[:0]
	for _, t := range ah.failures {
		if t.After(cutoff) {
			valid = append(valid, t)
		}
	}
	ah.failures = append(valid, now)

	// Check threshold.
	if len(ah.failures) >= healthFailureThreshold {
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
