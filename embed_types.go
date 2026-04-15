package inferrouter

// EmbedRequest is the public API request for text embeddings.
//
// Unlike ChatRequest, embeddings are pure input — no temperature, no streaming,
// no multimodal media parts. Inputs is a batch of texts to embed in a single
// logical request; the router may split this across provider API calls if the
// batch exceeds the selected provider's MaxBatchSize.
type EmbedRequest struct {
	// Model is the alias or concrete model name (e.g. "text-embedding-004").
	Model string

	// Inputs is the batch of texts to embed. Order is preserved in the response.
	Inputs []string

	// TaskType influences embedding quality for some models. For
	// text-embedding-004, valid values include:
	//   RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT, SEMANTIC_SIMILARITY,
	//   CLASSIFICATION, CLUSTERING, QUESTION_ANSWERING,
	//   FACT_VERIFICATION, CODE_RETRIEVAL_QUERY.
	//
	// An empty string defers to the provider default. Consumers doing RAG
	// should set this explicitly — indexing uses RETRIEVAL_DOCUMENT and
	// queries use RETRIEVAL_QUERY (asymmetric retrieval).
	TaskType string

	// OutputDimensionality optionally truncates the embedding to a smaller
	// size (Matryoshka representation). 0 means native size of the model.
	// text-embedding-004 supports [1..768]. Ignored by providers that
	// don't support truncation.
	OutputDimensionality int
}

// EmbedResponse is the public API response.
//
// Invariants (contract for consumers):
//
//   - len(Embeddings) equals the number of successfully processed inputs.
//     On happy path that equals len(req.Inputs). On ErrPartialBatch it
//     equals ErrPartialBatch.ProcessedInputs (see errors.go).
//   - Embeddings[i] corresponds to req.Inputs[i] for i < len(Embeddings).
//     Order is preserved strictly — consumers may use indices directly.
//   - Model holds the actual resolved model (not alias). Consumers may
//     compare resp.Model against their expected model as a last-line
//     defense against configuration drift.
type EmbedResponse struct {
	Embeddings [][]float32
	Model      string
	Usage      EmbedUsage
	Routing    RoutingInfo
}

// EmbedUsage tracks embedding-specific usage.
//
// Embeddings have only input tokens — no completion tokens, no cached
// tokens, no per-modality breakdown. Reusing the chat Usage type would
// force zero-filled fields that obscure the semantic difference.
type EmbedUsage struct {
	InputTokens int64
	TotalTokens int64 // == InputTokens for embeddings
}

// EmbedProviderRequest is what the router passes to an EmbeddingProvider adapter.
//
// The router guarantees len(Inputs) <= provider.MaxBatchSize() before calling
// Embed — providers do not need to split internally.
type EmbedProviderRequest struct {
	Auth                 Auth
	Model                string
	Inputs               []string
	TaskType             string
	OutputDimensionality int
}

// EmbedProviderResponse is what an EmbeddingProvider adapter returns.
//
// Embeddings must be in the same order as EmbedProviderRequest.Inputs.
type EmbedProviderResponse struct {
	Embeddings [][]float32
	Model      string
	Usage      EmbedUsage
}
