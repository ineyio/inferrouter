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
    qs.SetQuota("gemini-free", 1500, ir.QuotaRequests)

    cfg := ir.Config{
        DefaultModel: "gemini-2.0-flash",
        Accounts: []ir.AccountConfig{
            {
                Provider: "gemini", ID: "gemini-free",
                Auth: ir.Auth{APIKey: "your-key"},
                DailyFree: 1500, QuotaUnit: ir.QuotaRequests,
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
                {Provider: "gemini", Model: "gemini-2.0-flash"},
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
        model: gemini-2.0-flash
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
      - { provider: gemini, model: gemini-2.0-flash }
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

## How It Works

1. **Resolve model** — alias lookup or direct match
2. **Build candidates** — Provider x Account x Model (filtered by SupportsModel)
3. **Filter** — remove unhealthy (circuit breaker), remove paid if AllowPaid=false
4. **Policy.Select** — order candidates by priority
5. **Loop**: Reserve quota -> Execute -> Commit (success) / Rollback+next (failure)
6. **Error classification**: fatal (400, 401) -> return immediately, retryable (429, 5xx) -> try next

## License

MIT
