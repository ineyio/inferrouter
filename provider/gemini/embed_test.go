package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ir "github.com/ineyio/inferrouter"
)

func TestSupportsEmbeddingModel(t *testing.T) {
	p := New()
	if !p.SupportsEmbeddingModel("text-embedding-004") {
		t.Error("should accept text-embedding-004")
	}
	if !p.SupportsEmbeddingModel("gemini-embedding-001") {
		t.Error("should accept gemini-embedding-001")
	}
	if p.SupportsEmbeddingModel("gemini-2.0-flash") {
		t.Error("should reject chat models")
	}
	if p.SupportsEmbeddingModel("text-embedding-3-small") {
		t.Error("should reject non-gemini embedding models")
	}
}

func TestMaxBatchSize(t *testing.T) {
	if New().MaxBatchSize() != 100 {
		t.Errorf("MaxBatchSize should be 100, got %d", New().MaxBatchSize())
	}
}

func TestEmbed_HappyPath(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify URL shape.
		if !strings.Contains(r.URL.Path, "models/text-embedding-004:batchEmbedContents") {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.URL.Query().Get("key") != "test-api-key" {
			t.Errorf("missing/wrong key query param: %q", r.URL.Query().Get("key"))
		}

		var body geminiEmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if len(body.Requests) != 3 {
			t.Errorf("expected 3 requests, got %d", len(body.Requests))
		}
		// Each sub-request must carry its model path (Gemini quirk).
		for i, req := range body.Requests {
			if req.Model != "models/text-embedding-004" {
				t.Errorf("req[%d].Model = %q, want models/text-embedding-004", i, req.Model)
			}
			if req.TaskType != "RETRIEVAL_DOCUMENT" {
				t.Errorf("req[%d].TaskType = %q, want RETRIEVAL_DOCUMENT", i, req.TaskType)
			}
		}

		resp := geminiEmbedResponse{
			Embeddings: []geminiEmbedding{
				{Values: []float32{0.1, 0.2, 0.3}},
				{Values: []float32{0.4, 0.5, 0.6}},
				{Values: []float32{0.7, 0.8, 0.9}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	p := New(WithBaseURL(srv.URL))
	out, err := p.Embed(context.Background(), ir.EmbedProviderRequest{
		Auth:     ir.Auth{APIKey: "test-api-key"},
		Model:    "text-embedding-004",
		Inputs:   []string{"alpha", "beta", "gamma"},
		TaskType: "RETRIEVAL_DOCUMENT",
	})
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}
	if len(out.Embeddings) != 3 {
		t.Errorf("got %d embeddings, want 3", len(out.Embeddings))
	}
	if len(out.Embeddings[0]) != 3 || out.Embeddings[0][0] != 0.1 {
		t.Errorf("embedding[0] = %v, want [0.1 0.2 0.3]", out.Embeddings[0])
	}
	if out.Model != "text-embedding-004" {
		t.Errorf("Model = %q, want text-embedding-004", out.Model)
	}
	if out.Usage.InputTokens <= 0 {
		t.Errorf("Usage.InputTokens should be estimated, got %d", out.Usage.InputTokens)
	}
}

func TestEmbed_OutputDimensionality(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body geminiEmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if body.Requests[0].OutputDimensionality == nil || *body.Requests[0].OutputDimensionality != 256 {
			t.Errorf("OutputDimensionality = %v, want 256", body.Requests[0].OutputDimensionality)
		}
		json.NewEncoder(w).Encode(geminiEmbedResponse{
			Embeddings: []geminiEmbedding{{Values: make([]float32, 256)}},
		})
	}))
	defer srv.Close()

	p := New(WithBaseURL(srv.URL))
	_, err := p.Embed(context.Background(), ir.EmbedProviderRequest{
		Auth:                 ir.Auth{APIKey: "k"},
		Model:                "text-embedding-004",
		Inputs:               []string{"hello"},
		OutputDimensionality: 256,
	})
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}
}

func TestEmbed_OmitsOutputDimensionalityWhenZero(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Raw body check — verify OutputDimensionality is not serialized
		// when zero (omitempty via pointer).
		raw, _ := io.ReadAll(r.Body)
		if strings.Contains(string(raw), "outputDimensionality") {
			t.Errorf("outputDimensionality should be omitted, body: %s", string(raw))
		}
		json.NewEncoder(w).Encode(geminiEmbedResponse{
			Embeddings: []geminiEmbedding{{Values: []float32{1}}},
		})
	}))
	defer srv.Close()

	p := New(WithBaseURL(srv.URL))
	_, err := p.Embed(context.Background(), ir.EmbedProviderRequest{
		Auth:   ir.Auth{APIKey: "k"},
		Model:  "text-embedding-004",
		Inputs: []string{"x"},
	})
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}
}

func TestEmbed_ErrorMapping(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		wantErr    error
	}{
		{"rate limited", http.StatusTooManyRequests, ir.ErrRateLimited},
		{"unauthorized", http.StatusUnauthorized, ir.ErrAuthFailed},
		{"forbidden", http.StatusForbidden, ir.ErrAuthFailed},
		{"bad request", http.StatusBadRequest, ir.ErrInvalidRequest},
		{"server error", http.StatusInternalServerError, ir.ErrProviderUnavailable},
		{"bad gateway", http.StatusBadGateway, ir.ErrProviderUnavailable},
	}
	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tc.statusCode)
				w.Write([]byte(`{"error": {"message": "test error"}}`))
			}))
			defer srv.Close()

			p := New(WithBaseURL(srv.URL))
			_, err := p.Embed(context.Background(), ir.EmbedProviderRequest{
				Auth:   ir.Auth{APIKey: "k"},
				Model:  "text-embedding-004",
				Inputs: []string{"x"},
			})
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("err = %v, want wrapped %v", err, tc.wantErr)
			}
		})
	}
}

func TestEmbed_ResponseSizeMismatch(t *testing.T) {
	// Gemini returns fewer embeddings than inputs — must error.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(geminiEmbedResponse{
			Embeddings: []geminiEmbedding{{Values: []float32{1}}},
		})
	}))
	defer srv.Close()

	p := New(WithBaseURL(srv.URL))
	_, err := p.Embed(context.Background(), ir.EmbedProviderRequest{
		Auth:   ir.Auth{APIKey: "k"},
		Model:  "text-embedding-004",
		Inputs: []string{"x", "y", "z"}, // 3 inputs
	})
	if err == nil {
		t.Fatal("expected size-mismatch error")
	}
	if !strings.Contains(err.Error(), "size mismatch") {
		t.Errorf("err = %v, want contains 'size mismatch'", err)
	}
}

func TestEmbed_ContextCancellation(t *testing.T) {
	// Block handler until context canceled.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	defer srv.Close()

	p := New(WithBaseURL(srv.URL))

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel before the call even starts

	_, err := p.Embed(ctx, ir.EmbedProviderRequest{
		Auth:   ir.Auth{APIKey: "k"},
		Model:  "text-embedding-004",
		Inputs: []string{"x"},
	})
	if err == nil {
		t.Fatal("expected error on canceled context")
	}
}

func TestEmbed_MalformedResponseBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("not json"))
	}))
	defer srv.Close()

	p := New(WithBaseURL(srv.URL))
	_, err := p.Embed(context.Background(), ir.EmbedProviderRequest{
		Auth:   ir.Auth{APIKey: "k"},
		Model:  "text-embedding-004",
		Inputs: []string{"x"},
	})
	if err == nil {
		t.Fatal("expected decode error")
	}
	if !strings.Contains(err.Error(), "decode") {
		t.Errorf("err = %v, want contains 'decode'", err)
	}
}
