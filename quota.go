package inferrouter

import "context"

// QuotaStore manages per-account quota reservations.
type QuotaStore interface {
	// Reserve attempts to reserve quota for a request. Returns a Reservation on success.
	Reserve(ctx context.Context, accountID string, amount int64, unit QuotaUnit, idempotencyKey string) (Reservation, error)

	// Commit finalizes a reservation with the actual usage.
	Commit(ctx context.Context, reservation Reservation, actualAmount int64) error

	// Rollback releases a reservation that was not used.
	Rollback(ctx context.Context, reservation Reservation) error

	// Remaining returns the remaining free quota for an account.
	Remaining(ctx context.Context, accountID string) (int64, error)
}

// Reservation represents a reserved quota allocation.
type Reservation struct {
	ID        string
	AccountID string
	Amount    int64
	Unit      QuotaUnit
}

// QuotaUnit defines how quota is measured.
type QuotaUnit string

const (
	QuotaTokens   QuotaUnit = "tokens"
	QuotaRequests QuotaUnit = "requests"
	QuotaDollars  QuotaUnit = "dollars"
)
