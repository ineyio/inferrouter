// Package postgres provides a PostgreSQL-backed QuotaStore for inferrouter.
//
// Quota state is stored in PostgreSQL tables with transactional Reserve/Commit/Rollback.
// This makes it safe for multi-instance deployments and provides durability across restarts.
package postgres

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ineyio/inferrouter"
)

// Store is a PostgreSQL-backed QuotaStore.
type Store struct {
	pool        *pgxpool.Pool
	tablePrefix string
}

var (
	_ inferrouter.QuotaStore       = (*Store)(nil)
	_ inferrouter.QuotaInitializer = (*Store)(nil)
)

// Option configures Store.
type Option func(*Store)

// WithTablePrefix sets the table name prefix (default "inferrouter_").
func WithTablePrefix(prefix string) Option {
	return func(s *Store) { s.tablePrefix = prefix }
}

// New creates a new PostgreSQL-backed QuotaStore.
func New(pool *pgxpool.Pool, opts ...Option) *Store {
	s := &Store{
		pool:        pool,
		tablePrefix: "inferrouter_",
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

func (s *Store) quotasTable() string      { return s.tablePrefix + "quotas" }
func (s *Store) idempotencyTable() string { return s.tablePrefix + "idempotency" }

// EnsureSchema creates the required tables if they don't exist.
func (s *Store) EnsureSchema(ctx context.Context) error {
	q := fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS %s (
			account_id TEXT PRIMARY KEY,
			daily_limit BIGINT NOT NULL,
			used BIGINT NOT NULL DEFAULT 0,
			reserved BIGINT NOT NULL DEFAULT 0,
			unit TEXT NOT NULL DEFAULT 'tokens',
			reset_at TIMESTAMPTZ NOT NULL
		);
		CREATE TABLE IF NOT EXISTS %s (
			key TEXT PRIMARY KEY,
			created_at TIMESTAMPTZ NOT NULL DEFAULT now()
		);
	`, s.quotasTable(), s.idempotencyTable())
	_, err := s.pool.Exec(ctx, q)
	if err != nil {
		return fmt.Errorf("inferrouter/postgres: ensure schema: %w", err)
	}
	return nil
}

// Reserve attempts to reserve quota for a request.
func (s *Store) Reserve(ctx context.Context, accountID string, amount int64, unit inferrouter.QuotaUnit, idempotencyKey string) (inferrouter.Reservation, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return inferrouter.Reservation{}, fmt.Errorf("inferrouter/postgres: begin tx: %w", err)
	}
	defer tx.Rollback(ctx)

	// 1. Idempotency check.
	if idempotencyKey != "" {
		var inserted bool
		err = tx.QueryRow(ctx,
			fmt.Sprintf(`INSERT INTO %s (key) VALUES ($1) ON CONFLICT DO NOTHING RETURNING true`, s.idempotencyTable()),
			idempotencyKey,
		).Scan(&inserted)
		if err == pgx.ErrNoRows {
			return inferrouter.Reservation{}, fmt.Errorf("inferrouter: duplicate idempotency key %q", idempotencyKey)
		}
		if err != nil {
			return inferrouter.Reservation{}, fmt.Errorf("inferrouter/postgres: idem check: %w", err)
		}
	}

	now := time.Now().UTC()
	nextMidnight := nextMidnightUTC(now)

	// 2. Lazy daily reset.
	_, err = tx.Exec(ctx,
		fmt.Sprintf(`UPDATE %s SET used = 0, reserved = 0, reset_at = $1 WHERE account_id = $2 AND reset_at <= $3`,
			s.quotasTable()),
		nextMidnight, accountID, now,
	)
	if err != nil {
		return inferrouter.Reservation{}, fmt.Errorf("inferrouter/postgres: daily reset: %w", err)
	}

	// 3. Atomic reserve: update only if enough available.
	var reserved bool
	err = tx.QueryRow(ctx,
		fmt.Sprintf(`UPDATE %s SET reserved = reserved + $1
			WHERE account_id = $2 AND (daily_limit - used - reserved) >= $1
			RETURNING true`, s.quotasTable()),
		amount, accountID,
	).Scan(&reserved)

	if err == pgx.ErrNoRows {
		// Check if account exists at all.
		var exists bool
		err = tx.QueryRow(ctx,
			fmt.Sprintf(`SELECT true FROM %s WHERE account_id = $1`, s.quotasTable()),
			accountID,
		).Scan(&exists)

		if err == pgx.ErrNoRows {
			// Account not found — unlimited. Remove idempotency key if inserted.
			if err := tx.Commit(ctx); err != nil {
				return inferrouter.Reservation{}, fmt.Errorf("inferrouter/postgres: commit unlimited: %w", err)
			}
			return inferrouter.Reservation{
				ID:        uuid.New().String(),
				AccountID: accountID,
				Amount:    amount,
				Unit:      unit,
			}, nil
		}
		if err != nil {
			return inferrouter.Reservation{}, fmt.Errorf("inferrouter/postgres: check exists: %w", err)
		}

		// Account exists but insufficient quota. Rollback idem key.
		return inferrouter.Reservation{}, inferrouter.ErrQuotaExceeded
	}
	if err != nil {
		return inferrouter.Reservation{}, fmt.Errorf("inferrouter/postgres: reserve: %w", err)
	}

	if err := tx.Commit(ctx); err != nil {
		return inferrouter.Reservation{}, fmt.Errorf("inferrouter/postgres: commit: %w", err)
	}

	return inferrouter.Reservation{
		ID:        uuid.New().String(),
		AccountID: accountID,
		Amount:    amount,
		Unit:      unit,
	}, nil
}

// Commit finalizes a reservation with the actual usage.
func (s *Store) Commit(ctx context.Context, res inferrouter.Reservation, actualAmount int64) error {
	_, err := s.pool.Exec(ctx,
		fmt.Sprintf(`UPDATE %s SET reserved = reserved - $1, used = used + $2 WHERE account_id = $3`,
			s.quotasTable()),
		res.Amount, actualAmount, res.AccountID,
	)
	if err != nil {
		return fmt.Errorf("inferrouter/postgres: commit: %w", err)
	}
	return nil
}

// Rollback releases a reservation that was not used.
func (s *Store) Rollback(ctx context.Context, res inferrouter.Reservation) error {
	_, err := s.pool.Exec(ctx,
		fmt.Sprintf(`UPDATE %s SET reserved = reserved - $1 WHERE account_id = $2`,
			s.quotasTable()),
		res.Amount, res.AccountID,
	)
	if err != nil {
		return fmt.Errorf("inferrouter/postgres: rollback: %w", err)
	}
	return nil
}

// Remaining returns the remaining free quota for an account.
func (s *Store) Remaining(ctx context.Context, accountID string) (int64, error) {
	var dailyLimit, used, reserved int64
	var resetAt time.Time

	err := s.pool.QueryRow(ctx,
		fmt.Sprintf(`SELECT daily_limit, used, reserved, reset_at FROM %s WHERE account_id = $1`,
			s.quotasTable()),
		accountID,
	).Scan(&dailyLimit, &used, &reserved, &resetAt)

	if err == pgx.ErrNoRows {
		return 0, nil
	}
	if err != nil {
		return 0, fmt.Errorf("inferrouter/postgres: remaining: %w", err)
	}

	// Lazy reset check (read-only).
	now := time.Now().UTC()
	if now.After(resetAt) || now.Equal(resetAt) {
		used = 0
		reserved = 0
	}

	available := dailyLimit - used - reserved
	if available < 0 {
		return 0, nil
	}
	return available, nil
}

// SetQuota configures the daily quota for an account (upsert).
func (s *Store) SetQuota(accountID string, dailyLimit int64, unit inferrouter.QuotaUnit) {
	ctx := context.Background()
	nextMidnight := nextMidnightUTC(time.Now().UTC())
	_, _ = s.pool.Exec(ctx,
		fmt.Sprintf(`INSERT INTO %s (account_id, daily_limit, unit, reset_at)
			VALUES ($1, $2, $3, $4)
			ON CONFLICT (account_id) DO UPDATE SET daily_limit = $2, unit = $3`,
			s.quotasTable()),
		accountID, dailyLimit, string(unit), nextMidnight,
	)
}

// CleanupIdempotency removes expired idempotency keys.
func (s *Store) CleanupIdempotency(ctx context.Context, olderThan time.Duration) (int64, error) {
	cutoff := time.Now().UTC().Add(-olderThan)
	tag, err := s.pool.Exec(ctx,
		fmt.Sprintf(`DELETE FROM %s WHERE created_at < $1`, s.idempotencyTable()),
		cutoff,
	)
	if err != nil {
		return 0, fmt.Errorf("inferrouter/postgres: cleanup idempotency: %w", err)
	}
	return tag.RowsAffected(), nil
}

func nextMidnightUTC(now time.Time) time.Time {
	return time.Date(now.Year(), now.Month(), now.Day()+1, 0, 0, 0, 0, time.UTC)
}
