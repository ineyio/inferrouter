//go:build integration

package postgres_test

import (
	"context"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ineyio/inferrouter"
	quotapg "github.com/ineyio/inferrouter/quota/postgres"
)

func newTestPool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	dsn := os.Getenv("DATABASE_URL")
	if dsn == "" {
		dsn = "postgres://localhost:5432/inferrouter_test?sslmode=disable"
	}
	pool, err := pgxpool.New(context.Background(), dsn)
	if err != nil {
		t.Fatalf("pgxpool: %v", err)
	}
	if err := pool.Ping(context.Background()); err != nil {
		t.Fatalf("postgres not available: %v", err)
	}
	t.Cleanup(func() { pool.Close() })
	return pool
}

func newTestStore(t *testing.T, pool *pgxpool.Pool) *quotapg.Store {
	t.Helper()
	// Use a unique prefix per test to avoid collisions.
	prefix := fmt.Sprintf("test_%s_", t.Name())
	s := quotapg.New(pool, quotapg.WithTablePrefix(prefix))

	ctx := context.Background()
	if err := s.EnsureSchema(ctx); err != nil {
		t.Fatalf("ensure schema: %v", err)
	}
	t.Cleanup(func() {
		pool.Exec(ctx, fmt.Sprintf("DROP TABLE IF EXISTS %squotas, %sidempotency", prefix, prefix))
	})
	return s
}

func TestReserveAndCommit(t *testing.T) {
	pool := newTestPool(t)
	store := newTestStore(t, pool)
	ctx := context.Background()

	store.SetQuota("acct1", 1000, inferrouter.QuotaTokens)

	res, err := store.Reserve(ctx, "acct1", 100, inferrouter.QuotaTokens, "")
	if err != nil {
		t.Fatalf("reserve: %v", err)
	}
	if res.AccountID != "acct1" || res.Amount != 100 {
		t.Fatalf("unexpected reservation: %+v", res)
	}

	err = store.Commit(ctx, res, 80)
	if err != nil {
		t.Fatalf("commit: %v", err)
	}

	remaining, err := store.Remaining(ctx, "acct1")
	if err != nil {
		t.Fatalf("remaining: %v", err)
	}
	if remaining != 920 {
		t.Fatalf("expected remaining=920, got %d", remaining)
	}
}

func TestReserveExceeded(t *testing.T) {
	pool := newTestPool(t)
	store := newTestStore(t, pool)
	ctx := context.Background()

	store.SetQuota("acct1", 100, inferrouter.QuotaTokens)

	_, err := store.Reserve(ctx, "acct1", 101, inferrouter.QuotaTokens, "")
	if err != inferrouter.ErrQuotaExceeded {
		t.Fatalf("expected ErrQuotaExceeded, got %v", err)
	}
}

func TestRollback(t *testing.T) {
	pool := newTestPool(t)
	store := newTestStore(t, pool)
	ctx := context.Background()

	store.SetQuota("acct1", 100, inferrouter.QuotaTokens)

	res, err := store.Reserve(ctx, "acct1", 60, inferrouter.QuotaTokens, "")
	if err != nil {
		t.Fatalf("reserve: %v", err)
	}

	err = store.Rollback(ctx, res)
	if err != nil {
		t.Fatalf("rollback: %v", err)
	}

	remaining, err := store.Remaining(ctx, "acct1")
	if err != nil {
		t.Fatalf("remaining: %v", err)
	}
	if remaining != 100 {
		t.Fatalf("expected remaining=100 after rollback, got %d", remaining)
	}
}

func TestIdempotencyDedup(t *testing.T) {
	pool := newTestPool(t)
	store := newTestStore(t, pool)
	ctx := context.Background()

	store.SetQuota("acct1", 1000, inferrouter.QuotaTokens)

	_, err := store.Reserve(ctx, "acct1", 10, inferrouter.QuotaTokens, "key-1")
	if err != nil {
		t.Fatalf("first reserve: %v", err)
	}

	_, err = store.Reserve(ctx, "acct1", 10, inferrouter.QuotaTokens, "key-1")
	if err == nil {
		t.Fatal("expected duplicate error, got nil")
	}
}

func TestUnknownAccountUnlimited(t *testing.T) {
	pool := newTestPool(t)
	store := newTestStore(t, pool)
	ctx := context.Background()

	res, err := store.Reserve(ctx, "unknown", 999999, inferrouter.QuotaTokens, "")
	if err != nil {
		t.Fatalf("expected unlimited for unknown account, got: %v", err)
	}
	if res.AccountID != "unknown" {
		t.Fatalf("unexpected account: %s", res.AccountID)
	}
}

func TestDailyReset(t *testing.T) {
	pool := newTestPool(t)
	store := newTestStore(t, pool)
	ctx := context.Background()

	store.SetQuota("acct1", 100, inferrouter.QuotaTokens)

	// Use up all quota.
	res, err := store.Reserve(ctx, "acct1", 100, inferrouter.QuotaTokens, "")
	if err != nil {
		t.Fatalf("reserve: %v", err)
	}
	_ = store.Commit(ctx, res, 100)

	// Manually set reset_at to the past.
	prefix := fmt.Sprintf("test_%s_", t.Name())
	_, err = pool.Exec(ctx,
		fmt.Sprintf(`UPDATE %squotas SET reset_at = $1 WHERE account_id = 'acct1'`, prefix),
		time.Now().UTC().Add(-time.Hour),
	)
	if err != nil {
		t.Fatalf("set reset_at: %v", err)
	}

	// Should now succeed (reset triggers).
	_, err = store.Reserve(ctx, "acct1", 50, inferrouter.QuotaTokens, "")
	if err != nil {
		t.Fatalf("expected reserve after reset, got: %v", err)
	}
}

func TestConcurrentReserves(t *testing.T) {
	pool := newTestPool(t)
	store := newTestStore(t, pool)
	ctx := context.Background()

	store.SetQuota("acct1", 100, inferrouter.QuotaRequests)

	var wg sync.WaitGroup
	var successCount atomic.Int64

	for i := range 20 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, err := store.Reserve(ctx, "acct1", 1, inferrouter.QuotaRequests, "")
			if err == nil {
				successCount.Add(1)
			}
		}(i)
	}

	wg.Wait()

	if successCount.Load() != 20 {
		t.Fatalf("expected 20 successful reserves, got %d", successCount.Load())
	}

	remaining, err := store.Remaining(ctx, "acct1")
	if err != nil {
		t.Fatalf("remaining: %v", err)
	}
	if remaining != 80 {
		t.Fatalf("expected remaining=80, got %d", remaining)
	}
}

func TestConcurrentReservesNoOverAllocation(t *testing.T) {
	pool := newTestPool(t)
	store := newTestStore(t, pool)
	ctx := context.Background()

	store.SetQuota("acct1", 10, inferrouter.QuotaRequests)

	var wg sync.WaitGroup
	var successCount atomic.Int64

	for i := range 20 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, err := store.Reserve(ctx, "acct1", 1, inferrouter.QuotaRequests, "")
			if err == nil {
				successCount.Add(1)
			}
		}(i)
	}

	wg.Wait()

	if successCount.Load() != 10 {
		t.Fatalf("expected exactly 10 successful reserves, got %d", successCount.Load())
	}
}

func TestRemainingCorrectness(t *testing.T) {
	pool := newTestPool(t)
	store := newTestStore(t, pool)
	ctx := context.Background()

	store.SetQuota("acct1", 500, inferrouter.QuotaTokens)

	remaining, err := store.Remaining(ctx, "acct1")
	if err != nil {
		t.Fatalf("remaining: %v", err)
	}
	if remaining != 500 {
		t.Fatalf("expected 500, got %d", remaining)
	}

	res, _ := store.Reserve(ctx, "acct1", 200, inferrouter.QuotaTokens, "")

	remaining, _ = store.Remaining(ctx, "acct1")
	if remaining != 300 {
		t.Fatalf("expected 300 after reserve, got %d", remaining)
	}

	_ = store.Commit(ctx, res, 150)

	remaining, _ = store.Remaining(ctx, "acct1")
	if remaining != 350 {
		t.Fatalf("expected 350 after commit, got %d", remaining)
	}
}

func TestTablePrefixIsolation(t *testing.T) {
	pool := newTestPool(t)
	ctx := context.Background()

	s1 := quotapg.New(pool, quotapg.WithTablePrefix("test_iso1_"))
	s2 := quotapg.New(pool, quotapg.WithTablePrefix("test_iso2_"))

	if err := s1.EnsureSchema(ctx); err != nil {
		t.Fatalf("ensure schema s1: %v", err)
	}
	if err := s2.EnsureSchema(ctx); err != nil {
		t.Fatalf("ensure schema s2: %v", err)
	}
	t.Cleanup(func() {
		pool.Exec(ctx, "DROP TABLE IF EXISTS test_iso1_quotas, test_iso1_idempotency, test_iso2_quotas, test_iso2_idempotency")
	})

	s1.SetQuota("acct1", 100, inferrouter.QuotaTokens)
	s2.SetQuota("acct1", 200, inferrouter.QuotaTokens)

	r1, _ := s1.Remaining(ctx, "acct1")
	r2, _ := s2.Remaining(ctx, "acct1")

	if r1 != 100 {
		t.Fatalf("s1 expected 100, got %d", r1)
	}
	if r2 != 200 {
		t.Fatalf("s2 expected 200, got %d", r2)
	}
}

func TestCleanupIdempotency(t *testing.T) {
	pool := newTestPool(t)
	store := newTestStore(t, pool)
	ctx := context.Background()

	store.SetQuota("acct1", 1000, inferrouter.QuotaTokens)

	// Create some idempotency keys.
	for i := range 5 {
		_, err := store.Reserve(ctx, "acct1", 1, inferrouter.QuotaTokens, fmt.Sprintf("cleanup-key-%d", i))
		if err != nil {
			t.Fatalf("reserve %d: %v", i, err)
		}
	}

	// Cleanup with 0 duration should remove all.
	deleted, err := store.CleanupIdempotency(ctx, 0)
	if err != nil {
		t.Fatalf("cleanup: %v", err)
	}
	if deleted != 5 {
		t.Fatalf("expected 5 deleted, got %d", deleted)
	}
}
