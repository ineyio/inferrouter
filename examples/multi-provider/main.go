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
	"github.com/ineyio/inferrouter/provider/openaicompat"
	"github.com/ineyio/inferrouter/quota"
)

func main() {
	// Set up providers.
	providers := []ir.Provider{
		gemini.New(),
		openaicompat.NewGrok(),
		openaicompat.NewOpenAI(),
	}

	// Quota store — NewRouter auto-initializes limits from AccountConfig.DailyFree.
	qs := quota.NewMemoryQuotaStore()

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
			{
				Provider:  "gemini",
				ID:        "gemini-1",
				Auth:      ir.Auth{APIKey: os.Getenv("GEMINI_KEY_1")},
				DailyFree: 1500,
				QuotaUnit: ir.QuotaRequests,
			},
			{
				Provider:  "gemini",
				ID:        "gemini-2",
				Auth:      ir.Auth{APIKey: os.Getenv("GEMINI_KEY_2")},
				DailyFree: 1500,
				QuotaUnit: ir.QuotaRequests,
			},
			{
				Provider:  "grok",
				ID:        "grok-free",
				Auth:      ir.Auth{APIKey: os.Getenv("GROK_API_KEY")},
				DailyFree: 5_000_000,
				QuotaUnit: ir.QuotaTokens,
			},
			{
				Provider:    "openai",
				ID:          "openai-paid",
				Auth:        ir.Auth{APIKey: os.Getenv("OPENAI_KEY")},
				DailyFree:   0,
				QuotaUnit:   ir.QuotaTokens,
				PaidEnabled: true,
			},
		},
	}

	router, err := ir.NewRouter(cfg, providers,
		ir.WithPolicy(&policy.FreeFirstPolicy{}),
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
}
