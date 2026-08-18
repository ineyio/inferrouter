package main

import (
	"context"
	"fmt"
	"log"
	"os"

	ir "github.com/ineyio/inferrouter"
	"github.com/ineyio/inferrouter/provider/gemini"
	"github.com/ineyio/inferrouter/quota"
)

func main() {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Fatal("GEMINI_API_KEY is required")
	}

	// Quota store — NewRouter auto-initializes limits from AccountConfig.DailyFree.
	qs := quota.NewMemoryQuotaStore()

	// Every request resolves through a declared alias (ladder). Even a
	// single-step one is spelled out — an undeclared name is an error, never
	// an attempt against every configured provider.
	cfg := ir.Config{
		DefaultModel: "fast",
		Models: []ir.ModelMapping{
			{
				Alias:  "fast",
				Models: []ir.ModelRef{{Provider: "gemini", Model: "gemini-2.0-flash"}},
			},
		},
		Accounts: []ir.AccountConfig{
			{
				Provider:  "gemini",
				ID:        "gemini-free",
				Auth:      ir.Auth{APIKey: apiKey},
				DailyFree: 1500,
				QuotaUnit: ir.QuotaRequests,
			},
		},
	}

	router, err := ir.NewRouter(cfg,
		[]ir.Provider{gemini.New()},
		ir.WithQuotaStore(qs),
	)
	if err != nil {
		log.Fatal(err)
	}

	resp, err := router.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{
			{Role: "user", Content: "What is the capital of France? Answer in one word."},
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Response: %s\n", resp.Choices[0].Message.Content)
	fmt.Printf("Routed to: %s/%s (free=%v, attempts=%d)\n",
		resp.Routing.Provider, resp.Routing.AccountID,
		resp.Routing.Free, resp.Routing.Attempts)
	fmt.Printf("Tokens: %d prompt + %d completion = %d total\n",
		resp.Usage.PromptTokens, resp.Usage.CompletionTokens, resp.Usage.TotalTokens)
}
