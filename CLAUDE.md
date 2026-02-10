# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is inferrouter

A Go library (not a standalone service) for smart LLM request routing that maximizes free tier usage across multiple providers and accounts. It routes requests through a candidate selection pipeline with quota management, circuit breaking, and policy-based sorting.

## Commands

```bash
go test ./...                          # Run all tests
go test -run TestName ./...            # Run a single test
go test -race ./...                    # Run tests with race detector
go build ./...                         # Verify compilation
go mod tidy                            # Clean up dependencies
```

No Makefile, no linter configured, no CI config. Module: `github.com/ineyio/inferrouter`.

## Architecture

### Routing Flow

```
ChatCompletion(req)
  → resolveModel() (alias or direct)
  → EstimateTokens()
  → buildCandidates() (Provider × Account × Model tuples)
  → filterCandidates() (remove unhealthy, paid if disallowed)
  → policy.Select() (sort by priority)
  → Loop candidates: Reserve → Execute → Commit/Rollback
```

Fatal errors (`ErrAuthFailed`, `ErrInvalidRequest`) stop the loop immediately. Retryable errors (`ErrRateLimited`, `ErrProviderUnavailable`, `ErrQuotaExceeded`) try the next candidate.

### Core Interfaces (all in root package)

| Interface | Purpose | Implementations |
|-----------|---------|----------------|
| `Provider` | LLM API adapter | `provider/openaicompat` (OpenAI, Grok, Cerebras), `provider/gemini`, `provider/mock` |
| `Policy` | Candidate sorting strategy | `policy.FreeFirstPolicy`, `policy.CostFirstPolicy` |
| `QuotaStore` | Reserve/Commit/Rollback quota | `quota.MemoryQuotaStore`, `quota/redis.Store`, `quota/postgres.Store` |
| `Meter` | Observability events | `meter.NoopMeter`, `meter.LogMeter` |

### Key Patterns

- **Reservation workflow**: Reserve (with idempotency key) → Execute → Commit (actual usage) or Rollback. Prevents double-charging on retries.
- **Circuit breaker** (`health.go`): Per-account. 3 failures in 5min → Unhealthy. After 30s → HalfOpen. Success → Healthy.
- **Streaming** (`stream.go`): `RouterStream` wraps provider stream; commits quota on `Close()`. Uses `context.Background()` for cleanup to avoid cancelled context issues.
- **QuotaInitializer**: If QuotaStore implements this optional interface, `NewRouter()` auto-initializes quotas from config.
- **NoopQuotaStore/NoopMeter**: Default no-ops when not configured — allows running without quota tracking.

### Config

YAML with `${ENV_VAR}` expansion. Defines accounts (provider, auth, daily free quota, cost), model aliases (map alias → list of provider/model pairs), and global settings (AllowPaid, DefaultModel).

## Conventions

- Go 1.23+, standard library preferred, minimal dependencies
- No ORM, no CGO, no HTTP server (library only)
- Errors use sentinel values with `IsFatal()`/`IsRetryable()` classification
- `RouterError` wraps errors with provider/account/model context
- Tests use `provider/mock` with configurable behavior (latency, errors, call counting)
- Quota stores: Memory (dev), Redis (multi-instance, Lua scripts), PostgreSQL (durable, ACID)
