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
  → messagesHaveMedia(req.Messages) → hasMedia bool (precomputed once)
  → resolveModel() (alias or direct)
  → EstimateTokens()  (handles text + per-modality byte heuristics)
  → buildCandidates() (Provider × Account × Model tuples; per-modality cost rates normalized via resolveModalityCost fallback)
  → filterCandidates(allowPaid, needMultimodal) (unhealthy, paid, spend-cap, multimodal-capable)
  → len==0 → ErrMultimodalUnavailable (if hasMedia) else ErrNoCandidates
  → policy.Select() (sort by priority)
  → Loop candidates: Reserve → Execute → Commit/Rollback
```

Fatal errors (`ErrAuthFailed`, `ErrInvalidRequest`) stop the loop immediately. Retryable errors (`ErrRateLimited`, `ErrProviderUnavailable`, `ErrQuotaExceeded`) try the next candidate. `ErrMultimodalUnavailable` is neither — callers are expected to catch it explicitly and degrade (e.g. strip media, retry via text alias).

### Core Interfaces (all in root package)

| Interface | Purpose | Implementations |
|-----------|---------|----------------|
| `Provider` | LLM API adapter (`Name`, `SupportsModel`, `SupportsMultimodal`, `ChatCompletion`, `ChatCompletionStream`) | `provider/openaicompat` (OpenAI, Grok, Cerebras — text-only), `provider/gemini` (multimodal), `provider/gonka`, `provider/mock` |
| `Policy` | Candidate sorting strategy | `policy.FreeFirstPolicy`, `policy.CostFirstPolicy` |
| `QuotaStore` | Reserve/Commit/Rollback quota | `quota.MemoryQuotaStore`, `quota/redis.Store`, `quota/postgres.Store` |
| `Meter` | Observability events | `meter.NoopMeter`, `meter.LogMeter` |

### Multimodal types

- **`Message.Parts []Part`** — multi-part content for image/audio/video. Non-nil `Parts` takes precedence over legacy `Content string`.
- **`Part{Type, Text, MIMEType, Data []byte}`** — caller passes raw bytes, providers handle base64 encoding internally.
- **`Usage.InputBreakdown *InputTokenBreakdown`** — per-modality (Text/Audio/Image/Video) split of PromptTokens. Gemini populates this from `promptTokensDetails[]`. Nil for text-only providers.
- **`Usage.CachedTokens int64`** — subset of PromptTokens served from context cache. **Observability-only**, not subtracted from cost (avoids double-counting the server-side discount).
- **`ProviderRequest.HasMedia bool`** — precomputed by router so providers don't re-walk messages on the streaming path.

### Key Patterns

- **Reservation workflow**: Reserve (with idempotency key) → Execute → Commit (actual usage) or Rollback. Prevents double-charging on retries.
- **Circuit breaker** (`health.go`): Per-account. 3 failures in 5min → Unhealthy. After 30s → HalfOpen. Success → Healthy.
- **Streaming** (`stream.go`): `RouterStream` wraps provider stream; commits quota on `Close()`. Uses `context.Background()` for cleanup to avoid cancelled context issues.
- **QuotaInitializer**: If QuotaStore implements this optional interface, `NewRouter()` auto-initializes quotas from config.
- **NoopQuotaStore/NoopMeter**: Default no-ops when not configured — allows running without quota tracking.
- **Pre-normalized modality costs**: `buildCandidates` resolves zero `CostPerAudio/Image/VideoInputToken` to `CostPerInputToken` via `resolveModalityCost`, so `calculateSpend` can multiply without fallback branches.

### Config

YAML with `${ENV_VAR}` expansion. Defines accounts (provider, auth, daily free quota, cost), model aliases (map alias → list of provider/model pairs), and global settings (AllowPaid, DefaultModel).

## Conventions

- Go 1.23+, standard library preferred, minimal dependencies
- No ORM, no CGO, no HTTP server (library only)
- Errors use sentinel values with `IsFatal()`/`IsRetryable()` classification
- `RouterError` wraps errors with provider/account/model context
- Tests use `provider/mock` with configurable behavior (latency, errors, call counting)
- Quota stores: Memory (dev), Redis (multi-instance, Lua scripts), PostgreSQL (durable, ACID)
