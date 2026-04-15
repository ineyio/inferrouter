package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/ineyio/inferrouter"
)

// EmbeddingProvider interface compliance.
var _ inferrouter.EmbeddingProvider = (*Provider)(nil)

// Known supported embedding models. Extending this set is safe — adding a new
// model identifier does not change vector space compatibility (each model is
// its own namespace per RFC §3.6).
var supportedEmbedModels = map[string]struct{}{
	"text-embedding-004":   {},
	"gemini-embedding-001": {},
}

// geminiEmbedMaxBatch is the hard limit on the batchEmbedContents endpoint.
// See https://ai.google.dev/api/embeddings#method:-models.batchembedcontents
const geminiEmbedMaxBatch = 100

// SupportsEmbeddingModel reports whether this provider can handle the given
// embedding model. Embedding and chat model namespaces are disjoint in
// Gemini, so this is a dedicated whitelist independent of SupportsModel.
func (p *Provider) SupportsEmbeddingModel(model string) bool {
	_, ok := supportedEmbedModels[model]
	return ok
}

// MaxBatchSize returns the maximum number of inputs accepted in one
// batchEmbedContents call.
func (p *Provider) MaxBatchSize() int { return geminiEmbedMaxBatch }

// Embed calls the Gemini batchEmbedContents endpoint for a batch of inputs.
// The router guarantees len(req.Inputs) <= MaxBatchSize() before calling.
//
// Single-endpoint simplicity: we always use batchEmbedContents even for
// a single input (RFC §4.2 design decision — one code path vs two).
func (p *Provider) Embed(ctx context.Context, req inferrouter.EmbedProviderRequest) (inferrouter.EmbedProviderResponse, error) {
	body := buildEmbedRequest(req)
	url := fmt.Sprintf("%s/models/%s:batchEmbedContents?key=%s", p.baseURL, req.Model, req.Auth.APIKey)

	httpResp, err := p.doEmbedRequest(ctx, url, body)
	if err != nil {
		return inferrouter.EmbedProviderResponse{}, err
	}
	defer httpResp.Body.Close()

	if err := mapHTTPError(httpResp); err != nil {
		return inferrouter.EmbedProviderResponse{}, err
	}

	var resp geminiEmbedResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return inferrouter.EmbedProviderResponse{}, fmt.Errorf("inferrouter: decode gemini embed response: %w", err)
	}

	if len(resp.Embeddings) != len(req.Inputs) {
		return inferrouter.EmbedProviderResponse{}, fmt.Errorf(
			"inferrouter: gemini embed response size mismatch: got %d, want %d",
			len(resp.Embeddings), len(req.Inputs))
	}

	embeddings := make([][]float32, len(resp.Embeddings))
	for i, e := range resp.Embeddings {
		embeddings[i] = e.Values
	}

	// Gemini embedding endpoints do not return token counts. Estimate with
	// the same heuristic as router-side pre-reservation so Commit lines up
	// with Reserve. Accuracy ~±20% is acceptable per RFC §4.4 — consumer
	// billing uses aggregated credit buckets, not per-token precision.
	var estimatedTokens int64
	for _, in := range req.Inputs {
		estimatedTokens += int64(len(in)) / 4
	}

	return inferrouter.EmbedProviderResponse{
		Embeddings: embeddings,
		Model:      req.Model,
		Usage: inferrouter.EmbedUsage{
			InputTokens: estimatedTokens,
			TotalTokens: estimatedTokens,
		},
	}, nil
}

// --- Gemini embed API types ---

type geminiEmbedRequest struct {
	Requests []geminiEmbedSingleRequest `json:"requests"`
}

type geminiEmbedSingleRequest struct {
	// Model is repeated per request per Gemini API quirk (even though
	// the URL already contains the model name).
	Model                string             `json:"model"`
	Content              geminiEmbedContent `json:"content"`
	TaskType             string             `json:"taskType,omitempty"`
	OutputDimensionality *int               `json:"outputDimensionality,omitempty"`
}

type geminiEmbedContent struct {
	Parts []geminiEmbedPart `json:"parts"`
}

type geminiEmbedPart struct {
	Text string `json:"text"`
}

type geminiEmbedResponse struct {
	Embeddings []geminiEmbedding `json:"embeddings"`
}

type geminiEmbedding struct {
	Values []float32 `json:"values"`
}

func buildEmbedRequest(req inferrouter.EmbedProviderRequest) geminiEmbedRequest {
	modelPath := "models/" + req.Model
	out := geminiEmbedRequest{
		Requests: make([]geminiEmbedSingleRequest, len(req.Inputs)),
	}
	var outDims *int
	if req.OutputDimensionality > 0 {
		d := req.OutputDimensionality
		outDims = &d
	}
	for i, input := range req.Inputs {
		out.Requests[i] = geminiEmbedSingleRequest{
			Model: modelPath,
			Content: geminiEmbedContent{
				Parts: []geminiEmbedPart{{Text: input}},
			},
			TaskType:             req.TaskType,
			OutputDimensionality: outDims,
		}
	}
	return out
}

// doEmbedRequest is analogous to doRequest for chat, but typed for embed.
// Sharing a lower-level helper would require generics or interface{}; the
// duplication is small and keeps both paths easy to read.
func (p *Provider) doEmbedRequest(ctx context.Context, url string, body geminiEmbedRequest) (*http.Response, error) {
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("inferrouter: marshal gemini embed request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("inferrouter: create gemini embed request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, inferrouter.ErrProviderUnavailable
	}
	return resp, nil
}
