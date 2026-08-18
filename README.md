# inferrouter

LLM request router that walks an ordered ladder of endpoints: it tries them in the order you declared, skips the ones that are unhealthy or cannot serve the request, and reports which one answered.

## Why?

A ladder is an ordered preference — "this model first, that gateway if it is down, this paid one as a last resort" — and the order usually encodes something the router cannot infer: quality, latency, which budget the traffic is allowed to touch. So the order you write is the order that runs. Nothing reorders it unless you ask for a policy, and the only steps that get skipped are ones that cannot serve the request at all.

Free tiers still matter — several providers offer generous ones, and `FreeFirstPolicy` exists for pooling them. It is now something you opt into rather than a rule built into routing.

## Install

```bash
go get github.com/ineyio/inferrouter
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    ir "github.com/ineyio/inferrouter"
    "github.com/ineyio/inferrouter/provider/gemini"
    "github.com/ineyio/inferrouter/quota"
)

func main() {
    qs := quota.NewMemoryQuotaStore()

    cfg := ir.Config{
        DefaultModel: "gemini-2.5-flash-lite",
        Accounts: []ir.AccountConfig{
            {
                Provider: "gemini", ID: "gemini-free",
                Auth: ir.Auth{APIKey: "your-key"},
                DailyFree: 1000, QuotaUnit: ir.QuotaRequests,
            },
        },
    }

    router, err := ir.NewRouter(cfg, []ir.Provider{gemini.New()}, ir.WithQuotaStore(qs))
    if err != nil {
        log.Fatal(err)
    }

    resp, err := router.ChatCompletion(context.Background(), ir.ChatRequest{
        Messages: []ir.Message{{Role: "user", Content: "Hello!"}},
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(resp.Choices[0].Message.Content)
    // Routed to: gemini/gemini-free (free=true)
}
```

## Multi-Provider Setup

```go
providers := []ir.Provider{
    gemini.New(),
    openaicompat.NewGrok(),
    openaicompat.NewOpenAI(),
}

cfg := ir.Config{
    AllowPaid:    true,
    DefaultModel: "fast",
    Models: []ir.ModelMapping{
        {
            Alias: "fast",
            Models: []ir.ModelRef{
                {Provider: "gemini", Model: "gemini-2.5-flash-lite"},
                {Provider: "grok", Model: "grok-3-fast"},
                {Provider: "openai", Model: "gpt-4o-mini"},
            },
        },
    },
    Accounts: []ir.AccountConfig{
        {Provider: "gemini", ID: "gemini-1", Auth: ir.Auth{APIKey: key1}, DailyFree: 1500, QuotaUnit: ir.QuotaRequests},
        {Provider: "gemini", ID: "gemini-2", Auth: ir.Auth{APIKey: key2}, DailyFree: 1500, QuotaUnit: ir.QuotaRequests},
        {Provider: "grok",   ID: "grok-free", Auth: ir.Auth{APIKey: key3}, DailyFree: 5000000, QuotaUnit: ir.QuotaTokens},
        {Provider: "openai", ID: "openai-paid", Auth: ir.Auth{APIKey: key4}, QuotaUnit: ir.QuotaTokens, PaidEnabled: true},
    },
}
```

## YAML Config

Config can be loaded from YAML with environment variable expansion:

```yaml
allow_paid: true
default_model: "fast"
models:
  - alias: "fast"
    models:
      - provider: gemini
        model: gemini-2.5-flash-lite
      - provider: grok
        model: grok-3-fast
accounts:
  - provider: gemini
    id: gemini-1
    auth: { api_key: "${GEMINI_KEY_1}" }
    daily_free: 1500
    quota_unit: requests
  - provider: grok
    id: grok-free
    auth: { api_key: "${GROK_API_KEY}" }
    daily_free: 5000000
    quota_unit: tokens
```

```go
cfg, err := ir.LoadConfig("config.yaml")
```

## Model Aliasing

Define aliases that map to different models per provider:

```yaml
models:
  - alias: "fast"
    models:
      - { provider: gemini, model: gemini-2.5-flash-lite }
      - { provider: grok, model: grok-3-fast }
      - { provider: openai, model: gpt-4o-mini }
  - alias: "smart"
    models:
      - { provider: gemini, model: gemini-2.5-pro }
      - { provider: openai, model: gpt-4o }
```

Then use `Model: "fast"` in requests — the router resolves to the right model per provider.

## Adding Providers

Most providers have OpenAI-compatible APIs and work with the universal adapter:

```go
import "github.com/ineyio/inferrouter/provider/openaicompat"

// Pre-built constructors:
openaicompat.NewOpenAI()
openaicompat.NewGrok()
openaicompat.NewCerebro()

// Or any OpenAI-compatible provider:
openaicompat.New("together", "https://api.together.xyz/v1")
openaicompat.New("ollama", "http://localhost:11434/v1")
```

Gemini has its own adapter due to a non-standard API:

```go
import "github.com/ineyio/inferrouter/provider/gemini"

gemini.New()
```

## Routing Policies

By default there is no policy: candidates are attempted in the order the alias lists its steps, and within a step in the order the accounts are declared. A policy is a deliberate reordering, useful when the steps really are interchangeable:

- **FreeFirstPolicy**: free candidates first (most remaining quota), then paid (cheapest)
- **CostFirstPolicy**: all candidates by cost ascending
- **LeastBusyPolicy**: fewest in-flight requests first — for a pool of equivalent gateways

```go
import "github.com/ineyio/inferrouter/policy"

ir.WithPolicy(&policy.FreeFirstPolicy{})
ir.WithPolicy(&policy.LeastBusyPolicy{})
```

## Per-attempt time budget

`AttemptTimeout` bounds one attempt rather than the whole walk, so a hung step cannot spend the budget its successors need. Set it globally or per account; zero means an attempt may use the caller's entire deadline.

```yaml
attempt_timeout: 60s
accounts:
  - id: slow-gateway
    attempt_timeout: 120s   # this one is known to be slower
```

On the streaming path the budget covers opening the stream. Once the first chunk is on its way the generation runs on the caller's deadline alone — a slow answer from a healthy step is not a reason to move on.

## Streaming

```go
stream, err := router.ChatCompletionStream(ctx, ir.ChatRequest{
    Messages: []ir.Message{{Role: "user", Content: "Hello!"}},
})
if err != nil {
    log.Fatal(err)
}
defer stream.Close()

for {
    chunk, err := stream.Next()
    if err == io.EOF {
        break
    }
    if err != nil {
        log.Fatal(err)
    }
    for _, c := range chunk.Choices {
        fmt.Print(c.Delta.Content)
    }
}
```

## Multimodal (image / audio / video)

Providers that support multimodal input (currently `provider/gemini`) accept media parts alongside text. Pass raw bytes via `Message.Parts` — the provider handles base64 encoding internally.

```go
resp, err := router.ChatCompletion(ctx, ir.ChatRequest{
    Model: "multimodal",
    Messages: []ir.Message{
        {
            Role: "user",
            Parts: []ir.Part{
                {Type: ir.PartText, Text: "What's in this photo?"},
                {Type: ir.PartImage, MIMEType: "image/jpeg", Data: photoBytes},
            },
        },
    },
})
```

When a request carries media, the router automatically filters candidates to providers whose `SupportsMultimodal()` returns true. If none are available (all filtered out or circuit-broken), it returns `ErrMultimodalUnavailable` — callers can catch this sentinel and degrade gracefully:

```go
if errors.Is(err, ir.ErrMultimodalUnavailable) {
    // Strip media, retry with a text-only alias
    return retryWithStrippedMedia(ctx, req)
}
```

### Per-modality cost and usage

`Usage.InputBreakdown` splits prompt tokens by modality for providers that report it (Gemini via `promptTokensDetails[]`):

```go
resp, _ := router.ChatCompletion(ctx, req)
if b := resp.Usage.InputBreakdown; b != nil {
    fmt.Printf("text=%d image=%d audio=%d video=%d\n", b.Text, b.Image, b.Audio, b.Video)
}
// Observability-only — does NOT reduce cost (providers already price cached
// content server-side).
fmt.Println("cached tokens:", resp.Usage.CachedTokens)
```

Configure per-modality rates in the account (zero values fall back to the text input rate):

```yaml
accounts:
  - provider: gemini
    id: gemini-paid
    auth: { api_key: "${GEMINI_API_KEY}" }
    quota_unit: tokens
    paid_enabled: true
    cost_per_input_token:       0.0000001  # $0.10 / 1M  (text/image/video baseline)
    cost_per_output_token:      0.0000004  # $0.40 / 1M
    cost_per_audio_input_token: 0.0000003  # $0.30 / 1M  (audio has a higher rate)
    max_daily_spend: 0.50
```

### LogMeter fields

`LogMeter.OnResult` emits `text_tokens`, `audio_tokens`, `image_tokens`, `video_tokens`, and `cached_tokens` only when non-zero. Text-only providers (Cerebras, OpenAI) see zero diff in their log shape.

## Embeddings

Providers that support text embedding implement the optional `EmbeddingProvider` interface. The router discovers this capability via type assertion at `NewRouter` time — no config flag required. Currently supported: `provider/gemini` with `text-embedding-004` and `gemini-embedding-001`.

```go
resp, err := router.EmbedBatch(ctx, ir.EmbedRequest{
    Model:    "text-embedding-004",
    Inputs:   []string{"first chunk", "second chunk", "third chunk"},
    TaskType: "RETRIEVAL_DOCUMENT", // RETRIEVAL_QUERY for queries
})
if err != nil {
    return err
}
// resp.Embeddings[i] is the vector for req.Inputs[i]
for i, vec := range resp.Embeddings {
    store(inputs[i], vec)
}
```

`EmbedBatch` automatically splits large input lists into sub-batches of at most `MaxBatchSize()` per the selected provider (Gemini = 100). On partial failure it returns `*ErrPartialBatch` alongside a valid prefix of embeddings, so consumers can checkpoint and resume:

```go
resp, err := router.EmbedBatch(ctx, req)
var partial *ir.ErrPartialBatch
if errors.As(err, &partial) {
    persist(resp.Embeddings) // len == partial.ProcessedInputs
    return retryWith(req.Inputs[partial.ProcessedInputs:])
}
```

### Config

Add a single-model alias per embedding model. **Cross-model fallback in embedding aliases is forbidden** — embedding vector spaces are not compatible between different models, so routing an index and a query to different models would silently corrupt retrieval quality. The router rejects such configs at `NewRouter` time with `ErrInvalidConfig`. Reliability should come from multiple accounts on the same model:

```yaml
models:
  - alias: text-embedding-004
    models:
      - provider: gemini
        model: text-embedding-004

accounts:
  - provider: gemini
    id: gemini-free
    auth: { api_key: "${GEMINI_FREE_KEY}" }
    daily_free: 1500          # free tier RPM budget
    quota_unit: requests
    cost_per_embedding_input_token: 0 # free

  - provider: gemini
    id: gemini-paid
    auth: { api_key: "${GEMINI_PAID_KEY}" }
    paid_enabled: true
    quota_unit: tokens
    cost_per_embedding_input_token: 0.00000003  # $0.025 / 1M input (text-embedding-004)
    max_daily_spend: 0.50
```

The embedding path follows the same rules: steps are attempted in declared order, the circuit breaker skips a tripped account, and an alias may never mix two different embedding models — several steps naming the *same* model are exactly the multi-account fallback you want.

If no provider implements `EmbeddingProvider` for the requested model, `EmbedBatch` returns `ErrNoEmbeddingProviders` — symmetric to `ErrMultimodalUnavailable` on the chat path.

### Genkit consumers

Inferrouter is a Go library. Consumers using [Genkit](https://firebase.google.com/docs/genkit) register models via `genkit.DefineModel` / `genkit.DefineEmbedder`. This library does not ship a Genkit-native `Embedder` adapter — consumers are responsible for wrapping `Router.EmbedBatch` in their own `DefineEmbedder` registration. See qarap's `pkg/inferrouterplugin/embed.go` as a reference implementation (if published).

## Quota Stores

The default `MemoryQuotaStore` is in-memory and doesn't survive restarts. For production, use Redis or PostgreSQL.

### Redis QuotaStore

```bash
go get github.com/ineyio/inferrouter/quota/redis
```

```go
import (
    goredis "github.com/redis/go-redis/v9"
    quotaredis "github.com/ineyio/inferrouter/quota/redis"
)

client := goredis.NewClient(&goredis.Options{Addr: "localhost:6379"})
qs := quotaredis.New(client)
// Optional: quotaredis.New(client, quotaredis.WithKeyPrefix("myapp:quota:"))

router, _ := ir.NewRouter(cfg, providers, ir.WithQuotaStore(qs))
```

Quota state is stored in Redis hashes with atomic Lua scripts. Safe for multi-instance deployments.

### PostgreSQL QuotaStore

```bash
go get github.com/ineyio/inferrouter/quota/postgres
```

```go
import (
    "github.com/jackc/pgx/v5/pgxpool"
    quotapg "github.com/ineyio/inferrouter/quota/postgres"
)

pool, _ := pgxpool.New(ctx, "postgres://localhost:5432/mydb")
qs := quotapg.New(pool)
// Optional: quotapg.New(pool, quotapg.WithTablePrefix("myapp_"))

qs.EnsureSchema(ctx) // creates tables if not exist

router, _ := ir.NewRouter(cfg, providers, ir.WithQuotaStore(qs))
```

Durable quota state with transactional Reserve. Call `CleanupIdempotency(ctx, 24*time.Hour)` periodically to prune old keys.

## How It Works

1. **Resolve model** — strict alias lookup; a name that is not a declared alias is `ErrUnknownAlias`, never an attempt against every provider
2. **Build candidates** — for each step of the ladder, the accounts serving that step's provider, in declaration order; per-modality cost rates pre-resolved from account config
3. **Filter** — remove unhealthy (circuit breaker), remove paid if AllowPaid=false, and (when request has media) drop providers whose SupportsMultimodal=false. Filtering skips steps; it never reorders them
4. **Policy.Select** — only if a policy was configured
5. **Loop**: Reserve quota -> Execute (under the attempt budget) -> Commit (success) / Rollback+next (failure)
6. **Error classification**: fatal (400, 401) -> return immediately, retryable (429, 5xx) -> try next. Multimodal requests with no capable candidate return `ErrMultimodalUnavailable` instead of the generic `ErrNoCandidates`.

## License

MIT
