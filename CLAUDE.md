# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is inferrouter

A Go library (not a standalone service) that executes an **ordered ladder** of LLM endpoints: an alias lists its steps in order of preference, and the router attempts them in that order, skipping only what cannot serve the request. Quota management, circuit breaking, rate limits and policy-based reordering are available around that core; only health and capability filtering are unconditional.

The order is the contract. If you find yourself adding something that reorders candidates by cost, load or free/paid without the caller having asked for it, that is the defect this core was rewritten to remove — `order_test.go` pins it, including the mutation that reintroduces a free-first default.

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

### Routing Flow (chat)

```
ChatCompletion(req)
  → messagesHaveMedia(req.Messages) → hasMedia bool (precomputed once)
  → resolveModel() (STRICT: alias only; unknown name → ErrUnknownAlias, fatal)
  → EstimateTokens()  (handles text + per-modality byte heuristics)
  → buildCandidates() (ref-major: for each ladder step, its provider's accounts
                       in cfg.Accounts order; cost rates normalized via resolveModalityCost)
  → filterCandidates(allowPaid, needMultimodal) (unhealthy, paid, spend-cap, multimodal-capable)
  → len==0 → ErrMultimodalUnavailable (if hasMedia) else ErrNoCandidates
  → policy.Select() ONLY if WithPolicy was given; otherwise config order stands
  → Loop candidates: ctx.Err() check → Reserve → Execute (under AttemptTimeout) → Commit/Rollback
```

Streaming follows the same walk. The attempt budget covers opening the stream (a watchdog cancels a stall); once open, the stream runs on the caller's context and releases it on `Close()`. `RouterStream.Routing()` reports the serving step, mirroring `ChatResponse.Routing`.

### Routing Flow (embeddings)

```
EmbedBatch(req)
  → resolveModel() (STRICT: same resolver as chat — one rule for both paths)
  → splitIntoBatches(inputs, MaxBatchSize) (100 for Gemini)
  → For each sub-batch:
      EstimateEmbedTokens(batch)
      → buildEmbedCandidates() (EmbeddingProvider × Account × Model; separate from chat map)
      → filterEmbedCandidates(allowPaid) (unhealthy, spend-cap)
      → len==0 → ErrNoEmbeddingProviders
      → no reordering: declared order stands (no Policy on this path)
      → Loop candidates: Wait (pace) → Reserve → Embed (retry on 429) → Commit/Rollback
  → Consolidate: concat Embeddings (order preserved), sum Usage
  → Partial failure: first failed sub-batch stops loop,
    return EmbedResponse{prefix} + *ErrPartialBatch{ProcessedInputs}
```

**Single-model alias invariant for embeddings** (RFC §3.6): any alias that references an embedding model must name exactly one distinct **model** — several steps pointing at the same model are the recommended multi-account fallback and are allowed. Cross-model fallback is rejected at `NewRouter` with `ErrInvalidConfig` because embedding vector spaces are not compatible between different models — routing index and query to different models silently corrupts RAG retrieval. Multi-account fallback on the same model is the reliability pattern.

**Pacing and 429 on the embed path.** Where chat *skips* a rate-limited candidate, embeddings
*wait* for one: a caller can absorb a few hundred milliseconds but not a missing vector, and an
embedding alias names one model, so the ladder below is usually empty. `RateLimiter.Wait` sleeps
until the minute window frees a slot (at most `maxWaitRounds` times, because a worker's context
has no deadline of its own); an exhausted hour or day window returns `ErrRateWindowExhausted`
immediately, since no reachable amount of waiting brings a spent allowance back.

A provider 429 is retried against the *same* candidate, up to `embedRateLimitRetries` times,
inside the same reservation — one logical request stays one reservation and one rate-limiter slot.
The pause is the provider's own: `gemini` parses `Retry-After` and the `google.rpc` RetryInfo
detail into `*RateLimitedError`, and the longer of the two wins, capped at `maxRetryAfter`.
A pause that would outlive the caller's deadline is not taken at all.

A 429 also does **not** feed the circuit breaker on this path. It is backpressure from a healthy
account, and with a single embedding account, counting it as ill health converts a seconds-long
provider window into a 30-second outage for everyone — the amplifier behind the 2026-09-01
incident. Chat keeps the old behaviour: `settleFailure` is untouched, and `order_test.go` still
pins the ladder's skip semantics.

Fatal errors (`ErrAuthFailed`, `ErrInvalidRequest`) stop the loop immediately. Retryable errors (`ErrRateLimited`, `ErrProviderUnavailable`, `ErrQuotaExceeded`) try the next candidate. `ErrMultimodalUnavailable` is neither — callers are expected to catch it explicitly and degrade (e.g. strip media, retry via text alias).

### Core Interfaces (all in root package)

| Interface | Purpose | Implementations |
|-----------|---------|----------------|
| `Provider` | LLM API adapter (`Name`, `SupportsModel`, `SupportsMultimodal`, `ChatCompletion`, `ChatCompletionStream`) | `provider/openaicompat` (OpenAI, Grok, Cerebras — text-only), `provider/gemini` (multimodal), `provider/gonka`, `provider/mock` |
| `EmbeddingProvider` | **Optional** capability for text embeddings (`Name`, `SupportsEmbeddingModel`, `Embed`, `MaxBatchSize`). Discovered via type assertion at `NewRouter`, so chat-only providers are unaffected. | `provider/gemini` (text-embedding-004, gemini-embedding-001), `provider/mock` (mock.NewEmbed) |
| `Policy` | **Optional** candidate reordering. Absent by default — config order wins | `policy.FreeFirstPolicy`, `policy.CostFirstPolicy`, `policy.LeastBusyPolicy` (fewest in-flight first — spreads concurrent requests across a pool of equivalent gateways; fed by the router's `InflightTracker`) |
| `QuotaStore` | Reserve/Commit/Rollback quota | `quota.MemoryQuotaStore`, `quota/redis.Store`, `quota/postgres.Store` |
| `Meter` | Observability events | `meter.NoopMeter`, `meter.LogMeter` |

### Multimodal types

- **`Message.Parts []Part`** — multi-part content for image/audio/video. Non-nil `Parts` takes precedence over legacy `Content string`.
- **`Part{Type, Text, MIMEType, Data []byte}`** — caller passes raw bytes, providers handle base64 encoding internally.
- **`Usage.InputBreakdown *InputTokenBreakdown`** — per-modality (Text/Audio/Image/Video) split of PromptTokens. Gemini populates this from `promptTokensDetails[]`. Nil for text-only providers.
- **`Usage.CachedTokens int64`** — subset of PromptTokens served from context cache. **Observability-only**, not subtracted from cost (avoids double-counting the server-side discount).
- **`ProviderRequest.HasMedia bool`** — precomputed by router so providers don't re-walk messages on the streaming path.

### Key Patterns

- **Cost is metadata**: an account may bill without publishing a price. Validation accepts it; only accounts with `daily_free > 0` get a local quota (no allowance means the provider enforces its own limits, not that the budget is zero). What such an account loses is spend tracking — `Router.ConfigWarnings()` returns that as data, since the library has no logger. A `max_daily_spend` with no rate can never trigger and says so there.
- **Reservation workflow**: Reserve (with idempotency key) → Execute → Commit (actual usage) or Rollback. Prevents double-charging on retries.
- **Circuit breaker** (`health.go`): Per-account. 3 failures in 5min → Unhealthy. After 30s → HalfOpen. Success → Healthy. On the embed path a 429 is not a failure (see above).
- **Streaming** (`stream.go`): `RouterStream` wraps provider stream; commits quota on `Close()`. Uses `context.Background()` for cleanup to avoid cancelled context issues.
- **QuotaInitializer**: If QuotaStore implements this optional interface, `NewRouter()` auto-initializes quotas from config.
- **NoopQuotaStore/NoopMeter**: Default no-ops when not configured — allows running without quota tracking.
- **Pre-normalized modality costs**: `buildCandidates` resolves zero `CostPerAudio/Image/VideoInputToken` to `CostPerInputToken` via `resolveModalityCost`, so `calculateSpend` can multiply without fallback branches.

### Config

YAML with `${ENV_VAR}` expansion. Defines accounts (provider, auth, daily free quota, cost, `attempt_timeout`), model aliases — the ladders, mapping alias → ordered list of provider/model pairs — and global settings (`AllowPaid`, `DefaultModel`, `AttemptTimeout`). `DefaultModel` must itself name a declared alias.

**Gateway pools**: `AccountConfig.BaseURL` (`base_url` in YAML) declares an OpenAI-compatible endpoint per account; `openaicompat.FromAccounts(cfg.Accounts)` builds one provider per distinct provider name from those entries — a pool of gateways (e.g. Gonka resellers like GonkaGate) is pure config, no code per gateway. By default the pool is walked in declaration order; add `policy.LeastBusyPolicy` when the gateways really are interchangeable and you want concurrent requests spread across them instead of funnelling into the first. In-flight counts are process-local and best-effort: a simultaneous burst may cluster its first wave before any counter increments. See `examples/gateway-pool/`.

## Conventions

- Go 1.23+, standard library preferred, minimal dependencies
- No ORM, no CGO, no HTTP server (library only)
- Errors use sentinel values with `IsFatal()`/`IsRetryable()` classification
- `RouterError` wraps errors with provider/account/model context
- Tests use `provider/mock` with configurable behavior (latency, errors, call counting)
- Quota stores: Memory (dev), Redis (multi-instance, Lua scripts), PostgreSQL (durable, ACID)
