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
	// text-embedding-004 was removed from v1beta in 2026-04 and is
	// intentionally absent from the whitelist (see NOTE in embed.go) —
	// rejecting it surfaces ErrNoEmbeddingProviders at router level
	// instead of a late HTTP 404.
	if p.SupportsEmbeddingModel("text-embedding-004") {
		t.Error("should reject text-embedding-004 (removed from v1beta 2026-04)")
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
	// embedding-2 was preview in April 2026 and is stable now; the whitelist
	// comment used to say otherwise.
	if !p.SupportsEmbeddingModel("gemini-embedding-2") {
		t.Error("should accept gemini-embedding-2")
	}
}

// TestBuildEmbedRequest_TaskTypeByModelCapability pins the pair, not the half
// that motivated the change.
//
// gemini-embedding-2 takes no taskType field, so the task rides inside the
// text. Asserting only that the field is gone would pass for a build that
// dropped the task altogether — losing the instruction quietly, which is a
// retrieval-quality regression with nothing to see in a log. The positive twin
// (001 still sends the field and leaves the text alone) is what makes the
// negative half mean anything.
func TestBuildEmbedRequest_TaskTypeByModelCapability(t *testing.T) {
	const input = "What is the meaning of life?"

	t.Run("field form keeps text and sends taskType", func(t *testing.T) {
		out, err := buildEmbedRequest(ir.EmbedProviderRequest{
			Model:    "gemini-embedding-001",
			Inputs:   []string{input},
			TaskType: "RETRIEVAL_QUERY",
		})
		if err != nil {
			t.Fatalf("buildEmbedRequest: %v", err)
		}
		if got := out.Requests[0].TaskType; got != "RETRIEVAL_QUERY" {
			t.Errorf("TaskType = %q, want RETRIEVAL_QUERY", got)
		}
		if got := out.Requests[0].Content.Parts[0].Text; got != input {
			t.Errorf("text = %q, want it untouched (%q)", got, input)
		}
	})

	t.Run("prompt form drops the field and prefixes the text", func(t *testing.T) {
		out, err := buildEmbedRequest(ir.EmbedProviderRequest{
			Model:    "gemini-embedding-2",
			Inputs:   []string{input},
			TaskType: "RETRIEVAL_QUERY",
		})
		if err != nil {
			t.Fatalf("buildEmbedRequest: %v", err)
		}
		if got := out.Requests[0].TaskType; got != "" {
			t.Errorf("TaskType = %q, want empty (the model has no such field)", got)
		}
		want := "task: search result | query: " + input
		if got := out.Requests[0].Content.Parts[0].Text; got != want {
			t.Errorf("text = %q, want %q", got, want)
		}
	})

	t.Run("prompt form documents carry structure, not a task", func(t *testing.T) {
		out, err := buildEmbedRequest(ir.EmbedProviderRequest{
			Model:    "gemini-embedding-2",
			Inputs:   []string{input},
			TaskType: "RETRIEVAL_DOCUMENT",
		})
		if err != nil {
			t.Fatalf("buildEmbedRequest: %v", err)
		}
		want := "title: none | text: " + input
		if got := out.Requests[0].Content.Parts[0].Text; got != want {
			t.Errorf("text = %q, want %q", got, want)
		}
	})

	t.Run("prompt form without a task leaves the text alone", func(t *testing.T) {
		out, err := buildEmbedRequest(ir.EmbedProviderRequest{
			Model:  "gemini-embedding-2",
			Inputs: []string{input},
		})
		if err != nil {
			t.Fatalf("buildEmbedRequest: %v", err)
		}
		if got := out.Requests[0].Content.Parts[0].Text; got != input {
			t.Errorf("text = %q, want it untouched (%q)", got, input)
		}
	})
}

// TestBuildEmbedRequest_OnePartPerInput guards the difference between N vectors
// and one.
//
// gemini-embedding-2 aggregates the parts of a SINGLE content into one joint
// embedding. Moving the inputs into one content is a two-character edit that
// still answers 200 and still decodes — and it would return one vector for a
// whole batch of chunks. Embed's response-size check catches that for a batch,
// but not for a single input, and a retrieve call is always a single input.
// Hence the assertion is on the request shape, not on the answer.
func TestBuildEmbedRequest_OnePartPerInput(t *testing.T) {
	inputs := []string{"alpha", "beta", "gamma"}

	for _, model := range []string{"gemini-embedding-001", "gemini-embedding-2"} {
		out, err := buildEmbedRequest(ir.EmbedProviderRequest{
			Model:    model,
			Inputs:   inputs,
			TaskType: "RETRIEVAL_DOCUMENT",
		})
		if err != nil {
			t.Fatalf("%s: buildEmbedRequest: %v", model, err)
		}
		if len(out.Requests) != len(inputs) {
			t.Fatalf("%s: %d requests for %d inputs — inputs must not share a request",
				model, len(out.Requests), len(inputs))
		}
		for i, r := range out.Requests {
			if len(r.Content.Parts) != 1 {
				t.Errorf("%s: request[%d] has %d parts, want exactly 1 — parts of one content are aggregated into a single vector",
					model, i, len(r.Content.Parts))
			}
			if !strings.Contains(r.Content.Parts[0].Text, inputs[i]) {
				t.Errorf("%s: request[%d] text %q does not carry input %q",
					model, i, r.Content.Parts[0].Text, inputs[i])
			}
		}
	}
}

// TestBuildEmbedRequest_UnknownTaskTypeOnPromptModel — an unmapped task must
// fail, not embed bare. Embedding it bare succeeds, costs money and returns a
// vector from a different distribution than the corpus it gets compared to.
func TestBuildEmbedRequest_UnknownTaskTypeOnPromptModel(t *testing.T) {
	_, err := buildEmbedRequest(ir.EmbedProviderRequest{
		Model:    "gemini-embedding-2",
		Inputs:   []string{"x"},
		TaskType: "NO_SUCH_TASK",
	})
	if !errors.Is(err, ir.ErrInvalidRequest) {
		t.Fatalf("err = %v, want wrapped %v", err, ir.ErrInvalidRequest)
	}
}

// TestBuildEmbedRequest_UnknownModelRefused — the zero value of embedModelCaps
// is the field form, so a missing map entry would otherwise make an unknown
// model behave exactly like 001. The router asks SupportsEmbeddingModel first,
// which is precisely why this refusal must not rest on it.
func TestBuildEmbedRequest_UnknownModelRefused(t *testing.T) {
	_, err := buildEmbedRequest(ir.EmbedProviderRequest{
		Model:  "text-embedding-004",
		Inputs: []string{"x"},
	})
	if !errors.Is(err, ir.ErrModelNotFound) {
		t.Fatalf("err = %v, want wrapped %v", err, ir.ErrModelNotFound)
	}
}

// TestEmbed2TaskPrefixes_ShapeAndCriticalMembers checks the carrier for a
// property no member may break, and separately names the two members the engine
// actually sends: a map is a poor witness to its own completeness, and deleting
// a row would take the assertion with it.
func TestEmbed2TaskPrefixes_ShapeAndCriticalMembers(t *testing.T) {
	for task, prefix := range embed2TaskPrefixes {
		if prefix == "" {
			t.Errorf("%s: empty prefix", task)
		}
		if !strings.HasSuffix(prefix, " ") {
			t.Errorf("%s: prefix %q must end with a space or it glues onto the first word", task, prefix)
		}
	}
	// VES sends exactly these two. Named here rather than left to the loop
	// above, which passes on an empty map.
	for _, task := range []string{"RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT"} {
		if _, ok := embed2TaskPrefixes[task]; !ok {
			t.Errorf("%s has no prompt form; VES sends it on every index and retrieve", task)
		}
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
		if !strings.Contains(r.URL.Path, "models/gemini-embedding-001:batchEmbedContents") {
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
			if req.Model != "models/gemini-embedding-001" {
				t.Errorf("req[%d].Model = %q, want models/gemini-embedding-001", i, req.Model)
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
		Model:    "gemini-embedding-001",
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
	if out.Model != "gemini-embedding-001" {
		t.Errorf("Model = %q, want gemini-embedding-001", out.Model)
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
		Model:                "gemini-embedding-001",
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
		Model:  "gemini-embedding-001",
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
				Model:  "gemini-embedding-001",
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
		Model:  "gemini-embedding-001",
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
		Model:  "gemini-embedding-001",
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
		Model:  "gemini-embedding-001",
		Inputs: []string{"x"},
	})
	if err == nil {
		t.Fatal("expected decode error")
	}
	if !strings.Contains(err.Error(), "decode") {
		t.Errorf("err = %v, want contains 'decode'", err)
	}
}
