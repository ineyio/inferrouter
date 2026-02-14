package inferrouter

import (
	"sync"
	"time"
)

// SpendTracker tracks per-account dollar spend with daily reset.
type SpendTracker struct {
	mu       sync.Mutex
	accounts map[string]*accountSpend
	resetDay int // day of year for last reset
}

type accountSpend struct {
	amount float64
}

// NewSpendTracker creates a new SpendTracker.
func NewSpendTracker() *SpendTracker {
	return &SpendTracker{
		accounts: make(map[string]*accountSpend),
		resetDay: time.Now().UTC().YearDay(),
	}
}

// RecordSpend records dollar spend for an account.
func (s *SpendTracker) RecordSpend(accountID string, dollars float64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.checkReset()

	as, ok := s.accounts[accountID]
	if !ok {
		as = &accountSpend{}
		s.accounts[accountID] = as
	}
	as.amount += dollars
}

// GetSpend returns the current daily spend for an account.
func (s *SpendTracker) GetSpend(accountID string) float64 {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.checkReset()

	as, ok := s.accounts[accountID]
	if !ok {
		return 0
	}
	return as.amount
}

// checkReset resets all spend if day has changed. Must be called with lock held.
func (s *SpendTracker) checkReset() {
	today := time.Now().UTC().YearDay()
	if today != s.resetDay {
		s.accounts = make(map[string]*accountSpend)
		s.resetDay = today
	}
}

// calculateSpend computes the dollar cost for a request.
func calculateSpend(c Candidate, usage Usage) float64 {
	if c.CostPerInputToken > 0 || c.CostPerOutputToken > 0 {
		return float64(usage.PromptTokens)*c.CostPerInputToken +
			float64(usage.CompletionTokens)*c.CostPerOutputToken
	}
	if c.CostPerToken > 0 {
		return float64(usage.TotalTokens) * c.CostPerToken
	}
	return 0
}
