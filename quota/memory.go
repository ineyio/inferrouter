package quota

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/ineyio/inferrouter"
)

// MemoryQuotaStore is an in-memory QuotaStore with daily reset.
type MemoryQuotaStore struct {
	mu       sync.RWMutex
	accounts map[string]*accountQuota
	seen     map[string]bool // idempotency key dedup
}

type accountQuota struct {
	DailyLimit int64
	Used       int64
	Reserved   int64
	Unit       inferrouter.QuotaUnit
	ResetAt    time.Time
}

var _ inferrouter.QuotaStore = (*MemoryQuotaStore)(nil)

// NewMemoryQuotaStore creates a new in-memory quota store.
func NewMemoryQuotaStore() *MemoryQuotaStore {
	return &MemoryQuotaStore{
		accounts: make(map[string]*accountQuota),
		seen:     make(map[string]bool),
	}
}

// SetQuota configures the daily quota for an account.
func (s *MemoryQuotaStore) SetQuota(accountID string, dailyLimit int64, unit inferrouter.QuotaUnit) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.accounts[accountID] = &accountQuota{
		DailyLimit: dailyLimit,
		Unit:       unit,
		ResetAt:    nextMidnightUTC(),
	}
}

// Reserve attempts to reserve quota. Returns ErrQuotaExceeded if insufficient.
func (s *MemoryQuotaStore) Reserve(_ context.Context, accountID string, amount int64, unit inferrouter.QuotaUnit, idempotencyKey string) (inferrouter.Reservation, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Idempotency check.
	if idempotencyKey != "" && s.seen[idempotencyKey] {
		return inferrouter.Reservation{}, fmt.Errorf("inferrouter: duplicate idempotency key %q", idempotencyKey)
	}

	aq, ok := s.accounts[accountID]
	if !ok {
		// No quota configured — unlimited.
		return inferrouter.Reservation{
			ID:        uuid.New().String(),
			AccountID: accountID,
			Amount:    amount,
			Unit:      unit,
		}, nil
	}

	s.maybeReset(aq)

	available := aq.DailyLimit - aq.Used - aq.Reserved
	if amount > available {
		return inferrouter.Reservation{}, inferrouter.ErrQuotaExceeded
	}

	aq.Reserved += amount

	if idempotencyKey != "" {
		s.seen[idempotencyKey] = true
	}

	return inferrouter.Reservation{
		ID:        uuid.New().String(),
		AccountID: accountID,
		Amount:    amount,
		Unit:      unit,
	}, nil
}

// Commit finalizes a reservation with actual usage.
func (s *MemoryQuotaStore) Commit(_ context.Context, res inferrouter.Reservation, actualAmount int64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	aq, ok := s.accounts[res.AccountID]
	if !ok {
		return nil
	}

	aq.Reserved -= res.Amount
	aq.Used += actualAmount
	return nil
}

// Rollback releases a reservation.
func (s *MemoryQuotaStore) Rollback(_ context.Context, res inferrouter.Reservation) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	aq, ok := s.accounts[res.AccountID]
	if !ok {
		return nil
	}

	aq.Reserved -= res.Amount
	return nil
}

// Remaining returns the remaining free quota for an account.
func (s *MemoryQuotaStore) Remaining(_ context.Context, accountID string) (int64, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	aq, ok := s.accounts[accountID]
	if !ok {
		return 0, nil
	}

	available := aq.DailyLimit - aq.Used - aq.Reserved
	if available < 0 {
		return 0, nil
	}
	return available, nil
}

func (s *MemoryQuotaStore) maybeReset(aq *accountQuota) {
	now := time.Now().UTC()
	if now.After(aq.ResetAt) {
		aq.Used = 0
		aq.Reserved = 0
		aq.ResetAt = nextMidnightUTC()
	}
}

func nextMidnightUTC() time.Time {
	now := time.Now().UTC()
	return time.Date(now.Year(), now.Month(), now.Day()+1, 0, 0, 0, 0, time.UTC)
}
