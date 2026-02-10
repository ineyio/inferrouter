//go:build integration

package redis_test

import (
	"context"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	goredis "github.com/redis/go-redis/v9"

	"github.com/ineyio/inferrouter"
	quotaredis "github.com/ineyio/inferrouter/quota/redis"
)

func newTestClient(t *testing.T) *goredis.Client {
	t.Helper()
	addr := os.Getenv("REDIS_ADDR")
	if addr == "" {
		addr = "localhost:6379"
	}
	client := goredis.NewClient(&goredis.Options{Addr: addr})
	ctx := context.Background()
	if err := client.Ping(ctx).Err(); err != nil {
		t.Fatalf("redis not available at %s: %v", addr, err)
	}
	t.Cleanup(func() { client.Close() })
	return client
}

func newTestStore(t *testing.T, client *goredis.Client) *quotaredis.Store {
	t.Helper()
	// Use a unique prefix per test to avoid collisions.
	prefix := "test:" + t.Name() + ":"
	s := quotaredis.New(client, quotaredis.WithKeyPrefix(prefix))
	t.Cleanup(func() {
		ctx := context.Background()
		iter := client.Scan(ctx, 0, prefix+"*", 100).Iterator()
		for iter.Next(ctx) {
			client.Del(ctx, iter.Val())
		}
	})
	return s
}

func TestReserveAndCommit(t *testing.T) {
	client := newTestClient(t)
	store := newTestStore(t, client)
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
	// 1000 - 80 used = 920
	if remaining != 920 {
		t.Fatalf("expected remaining=920, got %d", remaining)
	}
}

func TestReserveExceeded(t *testing.T) {
	client := newTestClient(t)
	store := newTestStore(t, client)
	ctx := context.Background()

	store.SetQuota("acct1", 100, inferrouter.QuotaTokens)

	_, err := store.Reserve(ctx, "acct1", 101, inferrouter.QuotaTokens, "")
	if err != inferrouter.ErrQuotaExceeded {
		t.Fatalf("expected ErrQuotaExceeded, got %v", err)
	}
}

func TestRollback(t *testing.T) {
	client := newTestClient(t)
	store := newTestStore(t, client)
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
	client := newTestClient(t)
	store := newTestStore(t, client)
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
	client := newTestClient(t)
	store := newTestStore(t, client)
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
	client := newTestClient(t)
	store := newTestStore(t, client)
	ctx := context.Background()

	store.SetQuota("acct1", 100, inferrouter.QuotaTokens)

	// Use up all quota.
	res, err := store.Reserve(ctx, "acct1", 100, inferrouter.QuotaTokens, "")
	if err != nil {
		t.Fatalf("reserve: %v", err)
	}
	_ = store.Commit(ctx, res, 100)

	// Manually set reset_at to the past to simulate daily reset.
	prefix := "test:" + t.Name() + ":"
	client.HSet(ctx, prefix+"acct1", "reset_at", time.Now().UTC().Add(-time.Hour).Unix())

	// Should now succeed (reset triggers).
	_, err = store.Reserve(ctx, "acct1", 50, inferrouter.QuotaTokens, "")
	if err != nil {
		t.Fatalf("expected reserve after reset, got: %v", err)
	}
}

func TestConcurrentReserves(t *testing.T) {
	client := newTestClient(t)
	store := newTestStore(t, client)
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

	// All 20 should succeed (limit is 100).
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
	client := newTestClient(t)
	store := newTestStore(t, client)
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
	client := newTestClient(t)
	store := newTestStore(t, client)
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
	// 500 - 150 used = 350
	if remaining != 350 {
		t.Fatalf("expected 350 after commit, got %d", remaining)
	}
}

func TestKeyPrefixIsolation(t *testing.T) {
	client := newTestClient(t)
	ctx := context.Background()

	s1 := quotaredis.New(client, quotaredis.WithKeyPrefix("test:iso1:"))
	s2 := quotaredis.New(client, quotaredis.WithKeyPrefix("test:iso2:"))
	t.Cleanup(func() {
		iter := client.Scan(ctx, 0, "test:iso*", 100).Iterator()
		for iter.Next(ctx) {
			client.Del(ctx, iter.Val())
		}
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
