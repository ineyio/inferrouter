// Package redis provides a Redis-backed QuotaStore for inferrouter.
//
// Quota state is stored in Redis hashes with atomic Lua scripts for
// Reserve/Commit/Rollback. This makes it safe for multi-instance deployments.
package redis

import (
	"context"
	"fmt"
	"strconv"
	"time"

	"github.com/google/uuid"
	goredis "github.com/redis/go-redis/v9"

	"github.com/ineyio/inferrouter"
)

// Store is a Redis-backed QuotaStore.
type Store struct {
	client    goredis.Cmdable
	keyPrefix string
}

var (
	_ inferrouter.QuotaStore      = (*Store)(nil)
	_ inferrouter.QuotaInitializer = (*Store)(nil)
)

// Option configures Store.
type Option func(*Store)

// WithKeyPrefix sets the Redis key prefix (default "inferrouter:quota:").
func WithKeyPrefix(prefix string) Option {
	return func(s *Store) { s.keyPrefix = prefix }
}

// New creates a new Redis-backed QuotaStore.
// The client must be a connected *goredis.Client or *goredis.ClusterClient.
func New(client goredis.Cmdable, opts ...Option) *Store {
	s := &Store{
		client:    client,
		keyPrefix: "inferrouter:quota:",
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

func (s *Store) accountKey(accountID string) string {
	return s.keyPrefix + accountID
}

func (s *Store) idemKey(key string) string {
	return s.keyPrefix + "idem:" + key
}

// reserveScript is a Lua script for atomic reserve.
// KEYS[1] = account hash key
// KEYS[2] = idempotency key
// ARGV[1] = amount
// ARGV[2] = now (unix seconds)
// ARGV[3] = next_midnight (unix seconds)
// ARGV[4] = has_idem ("1" or "0")
//
// Returns:
//
//	1  = reserved OK
//	0  = quota exceeded
//	-1 = duplicate idempotency key
//	-2 = account not found (unlimited)
var reserveScript = goredis.NewScript(`
local account_key = KEYS[1]
local idem_key = KEYS[2]
local amount = tonumber(ARGV[1])
local now = tonumber(ARGV[2])
local next_midnight = tonumber(ARGV[3])
local has_idem = ARGV[4]

-- Idempotency check
if has_idem == "1" then
    local set = redis.call("SET", idem_key, "1", "NX", "EX", 86400)
    if not set then
        return -1
    end
end

-- Check account exists
local daily_limit = redis.call("HGET", account_key, "daily_limit")
if not daily_limit then
    return -2
end
daily_limit = tonumber(daily_limit)

-- Lazy daily reset
local reset_at = tonumber(redis.call("HGET", account_key, "reset_at") or "0")
if now >= reset_at then
    redis.call("HSET", account_key, "used", "0", "reserved", "0", "reset_at", tostring(next_midnight))
end

local used = tonumber(redis.call("HGET", account_key, "used") or "0")
local reserved = tonumber(redis.call("HGET", account_key, "reserved") or "0")
local available = daily_limit - used - reserved

if amount > available then
    -- Rollback idempotency key on failure
    if has_idem == "1" then
        redis.call("DEL", idem_key)
    end
    return 0
end

redis.call("HINCRBY", account_key, "reserved", amount)
return 1
`)

// commitScript atomically commits a reservation.
// KEYS[1] = account hash key
// ARGV[1] = reserved_amount (to release from reserved)
// ARGV[2] = actual_amount (to add to used)
var commitScript = goredis.NewScript(`
local account_key = KEYS[1]
if redis.call("EXISTS", account_key) == 0 then
    return 1
end
redis.call("HINCRBY", account_key, "reserved", -tonumber(ARGV[1]))
redis.call("HINCRBY", account_key, "used", tonumber(ARGV[2]))
return 1
`)

// rollbackScript atomically rolls back a reservation.
// KEYS[1] = account hash key
// ARGV[1] = amount
var rollbackScript = goredis.NewScript(`
local account_key = KEYS[1]
if redis.call("EXISTS", account_key) == 0 then
    return 1
end
redis.call("HINCRBY", account_key, "reserved", -tonumber(ARGV[1]))
return 1
`)

// Reserve attempts to reserve quota for a request.
func (s *Store) Reserve(ctx context.Context, accountID string, amount int64, unit inferrouter.QuotaUnit, idempotencyKey string) (inferrouter.Reservation, error) {
	now := time.Now().UTC()
	nextMidnight := nextMidnightUTC(now)

	hasIdem := "0"
	idemK := s.idemKey("_noop")
	if idempotencyKey != "" {
		hasIdem = "1"
		idemK = s.idemKey(idempotencyKey)
	}

	result, err := reserveScript.Run(ctx, s.client,
		[]string{s.accountKey(accountID), idemK},
		amount, now.Unix(), nextMidnight.Unix(), hasIdem,
	).Int64()
	if err != nil {
		return inferrouter.Reservation{}, fmt.Errorf("inferrouter/redis: reserve: %w", err)
	}

	switch result {
	case 1:
		return inferrouter.Reservation{
			ID:        uuid.New().String(),
			AccountID: accountID,
			Amount:    amount,
			Unit:      unit,
		}, nil
	case 0:
		return inferrouter.Reservation{}, inferrouter.ErrQuotaExceeded
	case -1:
		return inferrouter.Reservation{}, fmt.Errorf("inferrouter: duplicate idempotency key %q", idempotencyKey)
	case -2:
		// Account not found — unlimited.
		return inferrouter.Reservation{
			ID:        uuid.New().String(),
			AccountID: accountID,
			Amount:    amount,
			Unit:      unit,
		}, nil
	default:
		return inferrouter.Reservation{}, fmt.Errorf("inferrouter/redis: unexpected reserve result: %d", result)
	}
}

// Commit finalizes a reservation with the actual usage.
func (s *Store) Commit(ctx context.Context, res inferrouter.Reservation, actualAmount int64) error {
	_, err := commitScript.Run(ctx, s.client,
		[]string{s.accountKey(res.AccountID)},
		res.Amount, actualAmount,
	).Result()
	if err != nil {
		return fmt.Errorf("inferrouter/redis: commit: %w", err)
	}
	return nil
}

// Rollback releases a reservation that was not used.
func (s *Store) Rollback(ctx context.Context, res inferrouter.Reservation) error {
	_, err := rollbackScript.Run(ctx, s.client,
		[]string{s.accountKey(res.AccountID)},
		res.Amount,
	).Result()
	if err != nil {
		return fmt.Errorf("inferrouter/redis: rollback: %w", err)
	}
	return nil
}

// Remaining returns the remaining free quota for an account.
func (s *Store) Remaining(ctx context.Context, accountID string) (int64, error) {
	vals, err := s.client.HMGet(ctx, s.accountKey(accountID), "daily_limit", "used", "reserved", "reset_at").Result()
	if err != nil {
		return 0, fmt.Errorf("inferrouter/redis: remaining: %w", err)
	}

	// Account not found.
	if vals[0] == nil {
		return 0, nil
	}

	dailyLimit, _ := strconv.ParseInt(vals[0].(string), 10, 64)
	used, _ := strconv.ParseInt(vals[1].(string), 10, 64)
	reserved, _ := strconv.ParseInt(vals[2].(string), 10, 64)
	resetAt, _ := strconv.ParseInt(vals[3].(string), 10, 64)

	// Lazy reset check (read-only, don't write).
	now := time.Now().UTC().Unix()
	if now >= resetAt {
		used = 0
		reserved = 0
	}

	available := dailyLimit - used - reserved
	if available < 0 {
		return 0, nil
	}
	return available, nil
}

// SetQuota configures the daily quota for an account.
func (s *Store) SetQuota(accountID string, dailyLimit int64, unit inferrouter.QuotaUnit) {
	ctx := context.Background()
	key := s.accountKey(accountID)
	now := time.Now().UTC()
	nextMidnight := nextMidnightUTC(now)

	// Only set if not already existing (preserve current used/reserved).
	// Use HSETNX for daily_limit to avoid resetting an active account.
	exists, _ := s.client.Exists(ctx, key).Result()
	if exists == 0 {
		s.client.HSet(ctx, key,
			"daily_limit", dailyLimit,
			"used", 0,
			"reserved", 0,
			"unit", string(unit),
			"reset_at", nextMidnight.Unix(),
		)
	} else {
		// Update limit and unit, preserve used/reserved.
		s.client.HSet(ctx, key,
			"daily_limit", dailyLimit,
			"unit", string(unit),
		)
	}
}

func nextMidnightUTC(now time.Time) time.Time {
	return time.Date(now.Year(), now.Month(), now.Day()+1, 0, 0, 0, 0, time.UTC)
}
