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

// taskForm is how a model wants the task of an embedding expressed.
//
// It is a property of the endpoint, not of this adapter: two models behind the
// same provider disagree about it, so the answer cannot live in one flag on the
// provider.
type taskForm int

const (
	// taskFormField — the model takes the task in the taskType request field.
	taskFormField taskForm = iota
	// taskFormPrompt — the model has no taskType field; the same information
	// is written as an instruction into the text being embedded.
	taskFormPrompt
)

// embedModelCaps describes what one embedding model accepts.
type embedModelCaps struct {
	taskForm taskForm
}

// Known supported embedding models and how each expresses the task of an
// embedding. Extending this set is safe — adding a new model identifier does
// not change vector space compatibility (each model is its own namespace per
// RFC §3.6).
//
// Verified against the Gemini API embeddings reference on 2026-09-09, and for
// 001 additionally against a live ListModels call on 2026-04-15:
//
//   - gemini-embedding-001: GA, native 3072 dims, supports outputDimensionality
//     for Matryoshka truncation down to any [1..3072]. Takes taskType as a
//     request field.
//   - gemini-embedding-2: GA (stable; the reference's "latest update" is April
//     2026), multimodal, 8192-token input window. **The taskType field is not
//     supported**, and sending it is not merely ignored — the task has to be
//     carried as an instruction inside the text instead (see
//     embed2TaskPrefixes). That difference is the whole reason this whitelist
//     maps to capabilities rather than being a set.
//
// An earlier revision of this comment called embedding-2 "gemini-embedding-2-preview:
// preview, not recommended for production". That was true in April 2026 and is
// not any more; it is called out rather than quietly deleted because it is the
// line the next person reads when deciding whether the model may be switched on.
//
// NOTE: text-embedding-004 was listed in the original VES RFC as the target
// model but has been removed from v1beta as of 2026-04. It is intentionally
// not in this whitelist — callers trying to use it will get an explicit
// ErrNoEmbeddingProviders at router level rather than a late HTTP 404.
var supportedEmbedModels = map[string]embedModelCaps{
	"gemini-embedding-001": {taskForm: taskFormField},
	"gemini-embedding-2":   {taskForm: taskFormPrompt},
}

// embed2TaskPrefixes translates the task vocabulary of gemini-embedding-001
// into the prompt prefixes gemini-embedding-2 was trained on.
//
// The strings are quoted verbatim from the "Task types with Embeddings 2"
// section of the Gemini API embeddings reference rather than paraphrased: the
// model was trained on these exact forms, and a synonym does not fail — it
// quietly retrieves worse, which is the kind of regression nothing reports.
//
// Retrieval is asymmetric: the query carries the task, while the document
// carries structure only. A document with no title uses the literal "none",
// which the reference prescribes explicitly. VES chunks have no title (an index
// request carries text, namespace and metadata, none of which is a document
// title), so RETRIEVAL_DOCUMENT is the constant form below; giving titles their
// own slot would change the embedded text, hence the vector space, hence would
// be a new model version rather than an edit here.
//
// Every prefix ends with a space so concatenation cannot glue the instruction
// onto the first word. TestEmbed2TaskPrefixes_ShapeAndCriticalMembers pins both
// that property and the two members the engine actually sends.
var embed2TaskPrefixes = map[string]string{
	"RETRIEVAL_QUERY":      "task: search result | query: ",
	"RETRIEVAL_DOCUMENT":   "title: none | text: ",
	"QUESTION_ANSWERING":   "task: question answering | query: ",
	"FACT_VERIFICATION":    "task: fact checking | query: ",
	"CODE_RETRIEVAL_QUERY": "task: code retrieval | query: ",
	"CLASSIFICATION":       "task: classification | query: ",
	"CLUSTERING":           "task: clustering | query: ",
	"SEMANTIC_SIMILARITY":  "task: sentence similarity | query: ",
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
	body, err := buildEmbedRequest(req)
	if err != nil {
		return inferrouter.EmbedProviderResponse{}, err
	}
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

// embedTaskForModel renders one input the way the given model wants the task
// expressed: either as text left alone plus a taskType field, or as text
// carrying the instruction and no field at all.
//
// An unknown task type on a prompt-form model is refused rather than embedded
// bare. Embedding it bare would succeed, cost money and return a vector from a
// different distribution than the corpus it will be compared against — a defect
// with no error attached to it.
func embedTaskForModel(caps embedModelCaps, taskType, input string) (text, field string, err error) {
	if caps.taskForm == taskFormField {
		return input, taskType, nil
	}
	if taskType == "" {
		// Nothing was asked for. The reference recommends against prefixing
		// multimodal input, so an absent task is a legitimate request rather
		// than an omission to repair.
		return input, "", nil
	}
	prefix, ok := embed2TaskPrefixes[taskType]
	if !ok {
		return "", "", fmt.Errorf(
			"%w: task type %q has no prompt form, and this model takes no taskType field",
			inferrouter.ErrInvalidRequest, taskType)
	}
	return prefix + input, "", nil
}

// buildEmbedRequest turns one provider request into the Gemini wire shape.
//
// Exactly one part per element of Requests, and one element per input. The
// distinction matters more than it looks: parts inside a SINGLE content are
// aggregated by gemini-embedding-2 into ONE vector for all of them. Both shapes
// answer 200, and the collapsed one destroys a corpus silently — the response
// size check in Embed catches it for a batch, but 1 == 1 for a single input,
// which is every retrieve call.
func buildEmbedRequest(req inferrouter.EmbedProviderRequest) (geminiEmbedRequest, error) {
	caps, ok := supportedEmbedModels[req.Model]
	if !ok {
		// The router asks SupportsEmbeddingModel before dispatching here, so
		// this is unreachable through it. It is still an explicit refusal
		// rather than a zero-value default, because the zero value is
		// taskFormField and would silently treat an unknown model like 001.
		return geminiEmbedRequest{}, fmt.Errorf(
			"%w: gemini has no embedding capabilities recorded for model %q",
			inferrouter.ErrModelNotFound, req.Model)
	}

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
		text, field, err := embedTaskForModel(caps, req.TaskType, input)
		if err != nil {
			return geminiEmbedRequest{}, err
		}
		out.Requests[i] = geminiEmbedSingleRequest{
			Model: modelPath,
			Content: geminiEmbedContent{
				Parts: []geminiEmbedPart{{Text: text}},
			},
			TaskType:             field,
			OutputDimensionality: outDims,
		}
	}
	return out, nil
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
