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

	// Set up quota store with free daily limit.
	qs := quota.NewMemoryQuotaStore()
	qs.SetQuota("gemini-free", 1500, ir.QuotaRequests)

	cfg := ir.Config{
		DefaultModel: "gemini-2.0-flash",
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
