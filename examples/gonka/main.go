package main

import (
	"context"
	"fmt"
	"log"
	"os"

	ir "github.com/ineyio/inferrouter"
	"github.com/ineyio/inferrouter/meter"
	"github.com/ineyio/inferrouter/policy"
	"github.com/ineyio/inferrouter/provider/gemini"
	"github.com/ineyio/inferrouter/provider/gonka"
	"github.com/ineyio/inferrouter/quota"
)

func main() {
	// Gonka private key — hex-encoded secp256k1 key.
	gonkaKey := os.Getenv("GONKA_PRIVATE_KEY")
	if gonkaKey == "" {
		log.Fatal("GONKA_PRIVATE_KEY is required")
	}

	providers := []ir.Provider{
		// Gonka: cheap ($1.5/1B tokens), slow, decentralized.
		gonka.New(
			gonka.WithEndpoint(gonka.Endpoint{
				URL:     os.Getenv("GONKA_ENDPOINT_URL"),     // e.g. "https://node1.gonka.ai/v1"
				Address: os.Getenv("GONKA_ENDPOINT_ADDRESS"), // Cosmos bech32 address of the node
			}),
			gonka.WithModels("Qwen/QwQ-32B", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"),
		),
		// Gemini: free tier fallback.
		gemini.New(),
	}

	qs := quota.NewMemoryQuotaStore()

	cfg := ir.Config{
		AllowPaid:    true,
		DefaultModel: "fast",
		Models: []ir.ModelMapping{
			{
				Alias: "fast",
				Models: []ir.ModelRef{
					{Provider: "gonka", Model: "Qwen/QwQ-32B"},
					{Provider: "gemini", Model: "gemini-2.0-flash"},
				},
			},
			{
				Alias: "smart",
				Models: []ir.ModelRef{
					{Provider: "gonka", Model: "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"},
					{Provider: "gemini", Model: "gemini-2.5-pro"},
				},
			},
		},
		Accounts: []ir.AccountConfig{
			{
				Provider:     "gonka",
				ID:           "gonka-bulk",
				Auth:         ir.Auth{APIKey: gonkaKey}, // hex private key goes here
				DailyFree:    0,
				QuotaUnit:    ir.QuotaTokens,
				PaidEnabled:  true,
				CostPerToken: 0.0000000015, // $1.5 / 1B tokens
			},
			{
				Provider:  "gemini",
				ID:        "gemini-free",
				Auth:      ir.Auth{APIKey: os.Getenv("GEMINI_API_KEY")},
				DailyFree: 1500,
				QuotaUnit: ir.QuotaRequests,
			},
		},
	}

	// CostFirstPolicy: routes to cheapest first (Gonka at $0.0000000015/token wins).
	// FreeFirstPolicy: routes to free Gemini first, then cheap Gonka.
	router, err := ir.NewRouter(cfg, providers,
		ir.WithPolicy(&policy.CostFirstPolicy{}),
		ir.WithQuotaStore(qs),
		ir.WithMeter(meter.NewLogMeter(nil)),
	)
	if err != nil {
		log.Fatal(err)
	}

	resp, err := router.ChatCompletion(context.Background(), ir.ChatRequest{
		Model:    "fast",
		Messages: []ir.Message{{Role: "user", Content: "Explain quantum computing in one sentence."}},
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Response: %s\n", resp.Choices[0].Message.Content)
	fmt.Printf("Routed to: %s/%s model=%s (free=%v, attempts=%d)\n",
		resp.Routing.Provider, resp.Routing.AccountID, resp.Routing.Model,
		resp.Routing.Free, resp.Routing.Attempts)
	fmt.Printf("Tokens: %d prompt + %d completion = %d total\n",
		resp.Usage.PromptTokens, resp.Usage.CompletionTokens, resp.Usage.TotalTokens)
}
