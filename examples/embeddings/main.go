package main

import (
	"context"
	"errors"
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

	qs := quota.NewMemoryQuotaStore()

	cfg := ir.Config{
		// Single-model alias. Multi-model aliases are rejected at NewRouter
		// because cross-model fallback silently breaks RAG — embedding
		// vector spaces are not compatible between different models.
		Models: []ir.ModelMapping{
			{
				Alias: "text-embedding-004",
				Models: []ir.ModelRef{
					{Provider: "gemini", Model: "text-embedding-004"},
				},
			},
		},
		Accounts: []ir.AccountConfig{
			{
				Provider:                   "gemini",
				ID:                         "gemini-free",
				Auth:                       ir.Auth{APIKey: apiKey},
				DailyFree:                  1500, // free tier RPM budget
				QuotaUnit:                  ir.QuotaRequests,
				CostPerEmbeddingInputToken: 0,
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

	// Index a batch of "documents" with RETRIEVAL_DOCUMENT task type.
	docs := []string{
		"The capital of France is Paris.",
		"Berlin is the capital of Germany.",
		"Madrid is the capital of Spain.",
	}

	resp, err := router.EmbedBatch(context.Background(), ir.EmbedRequest{
		Model:    "text-embedding-004",
		Inputs:   docs,
		TaskType: "RETRIEVAL_DOCUMENT",
	})
	var partial *ir.ErrPartialBatch
	if errors.As(err, &partial) {
		// resp.Embeddings[0..partial.ProcessedInputs-1] is a valid prefix.
		fmt.Printf("Partial failure after %d inputs: %v\n", partial.ProcessedInputs, partial.Cause)
	} else if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Indexed %d documents with %s\n", len(resp.Embeddings), resp.Model)
	for i, vec := range resp.Embeddings {
		fmt.Printf("  doc[%d] = %d-dim vector (first 3: %v...)\n", i, len(vec), vec[:3])
	}

	// Query with a different task type (RETRIEVAL_QUERY) for asymmetric retrieval.
	query := "Which city is France's capital?"
	qResp, err := router.EmbedBatch(context.Background(), ir.EmbedRequest{
		Model:    "text-embedding-004",
		Inputs:   []string{query},
		TaskType: "RETRIEVAL_QUERY",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\nQuery vector: %d-dim (first 3: %v...)\n",
		len(qResp.Embeddings[0]), qResp.Embeddings[0][:3])

	fmt.Printf("\nRouted to: %s/%s (free=%v)\n",
		qResp.Routing.Provider, qResp.Routing.AccountID, qResp.Routing.Free)
	fmt.Printf("Total input tokens across all calls: %d\n", resp.Usage.InputTokens+qResp.Usage.InputTokens)
}
