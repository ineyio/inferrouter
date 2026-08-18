// Example: a pool of OpenAI-compatible gateways (Gonka resellers) declared
// entirely via AccountConfig.BaseURL, with LeastBusyPolicy spreading
// concurrent tasks across the pool.
//
// Gonka gateways are cheap but slow (~50-75 tok/s per stream), so throughput
// comes from parallelism: N concurrent requests land on N different gateways.
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"sync"
	"time"

	ir "github.com/ineyio/inferrouter"
	"github.com/ineyio/inferrouter/policy"
	"github.com/ineyio/inferrouter/provider/openaicompat"
)

func main() {
	const model = "qwen/qwen3-235b-a22b-instruct-2507-fp8"

	cfg := ir.Config{
		AllowPaid:    true,
		DefaultModel: "pool",
		// One step per gateway, in preference order. Without a policy the
		// router walks them exactly as listed; LeastBusyPolicy below opts into
		// load-spreading instead.
		Models: []ir.ModelMapping{
			{
				Alias:  "pool",
				Models: []ir.ModelRef{{Provider: "gonkagate", Model: model}},
			},
		},
		Accounts: []ir.AccountConfig{
			{
				Provider:           "gonkagate",
				ID:                 "gonkagate-main",
				BaseURL:            "https://api.gonkagate.com/v1",
				Auth:               ir.Auth{APIKey: os.Getenv("GONKAGATE_API_KEY")},
				QuotaUnit:          ir.QuotaDollars,
				PaidEnabled:        true,
				CostPerInputToken:  0.0000000004, // ~$0.0004 / 1M — check current pricing
				CostPerOutputToken: 0.0000000004,
				MaxDailySpend:      1.0,
			},
			// More Gonka gateways drop in here as plain config entries —
			// same model alias, LeastBusyPolicy spreads load across them.
		},
	}

	// One provider per distinct gateway, built straight from config.
	providers, err := openaicompat.FromAccounts(cfg.Accounts,
		// Slow decentralized backends: one shared generous timeout.
		openaicompat.WithHTTPClient(&http.Client{Timeout: 120 * time.Second}),
	)
	if err != nil {
		log.Fatal(err)
	}

	router, err := ir.NewRouter(cfg, providers,
		ir.WithPolicy(&policy.LeastBusyPolicy{}),
	)
	if err != nil {
		log.Fatal(err)
	}

	// Fan out independent tasks; the pool absorbs them in parallel.
	tasks := []string{
		"Extract the price entity from: <span class=\"cost\">$42.50</span>",
		"Extract the address entity from: <p>221B Baker Street, London</p>",
		"Extract the person name from: <h1>Dr. Jane Goodall</h1>",
	}

	var wg sync.WaitGroup
	for i, task := range tasks {
		wg.Add(1)
		go func(n int, prompt string) {
			defer wg.Done()
			resp, err := router.ChatCompletion(context.Background(), ir.ChatRequest{
				Messages: []ir.Message{{Role: "user", Content: prompt}},
			})
			if err != nil {
				log.Printf("task %d: %v", n, err)
				return
			}
			fmt.Printf("task %d via %s/%s: %s\n",
				n, resp.Routing.Provider, resp.Routing.AccountID,
				resp.Choices[0].Message.Content)
		}(i, task)
	}
	wg.Wait()
}
