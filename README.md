# inferrouter

Smart LLM request router that maximizes free tier usage across multiple providers and accounts.

## Why?

Most LLM providers offer free tiers (Gemini 1500 RPD, Grok 5M tokens/day, etc.). By combining N accounts x M providers, you get significant free throughput. inferrouter automates the routing — picking free candidates first, falling back to paid when needed.

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

- **FreeFirstPolicy** (default): Free candidates first (most remaining quota), then paid (cheapest)
- **CostFirstPolicy**: All candidates by cost ascending (free naturally first)

```go
import "github.com/ineyio/inferrouter/policy"

ir.WithPolicy(&policy.FreeFirstPolicy{})
ir.WithPolicy(&policy.CostFirstPolicy{})
```

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

1. **Resolve model** — alias lookup or direct match
2. **Build candidates** — Provider x Account x Model (filtered by SupportsModel); per-modality cost rates pre-resolved from account config
3. **Filter** — remove unhealthy (circuit breaker), remove paid if AllowPaid=false, and (when request has media) drop providers whose SupportsMultimodal=false
4. **Policy.Select** — order candidates by priority
5. **Loop**: Reserve quota -> Execute -> Commit (success) / Rollback+next (failure)
6. **Error classification**: fatal (400, 401) -> return immediately, retryable (429, 5xx) -> try next. Multimodal requests with no capable candidate return `ErrMultimodalUnavailable` instead of the generic `ErrNoCandidates`.

## License

MIT
