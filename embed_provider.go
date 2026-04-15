package inferrouter

import "context"

// EmbeddingProvider is an OPTIONAL capability interface.
//
// Providers implement this interface if they support text embedding.
// A single Provider struct may implement both Provider (for chat) and
// EmbeddingProvider (for embeddings), or only one of them. The router
// discovers embedding capability via type assertion at NewRouter time.
//
// Chat-only providers (e.g. openaicompat in its current form, gonka) do
// not implement this interface — this is honest via compile-time absence,
// not via a runtime "return ErrNotSupported" stub.
//
// See RFC docs/proposals/inferrouter-embeddings.md §3.1.
type EmbeddingProvider interface {
	// Name returns the provider identifier. For providers that implement
	// both Provider and EmbeddingProvider, this must match Provider.Name().
	Name() string

	// SupportsEmbeddingModel reports whether this provider can handle the
	// given embedding model name (e.g. "text-embedding-004").
	SupportsEmbeddingModel(model string) bool

	// Embed generates embeddings for a batch of input texts.
	//
	// The router guarantees len(req.Inputs) <= MaxBatchSize() before
	// calling. Providers do not need to split internally.
	//
	// The returned Embeddings must preserve the order of req.Inputs.
	Embed(ctx context.Context, req EmbedProviderRequest) (EmbedProviderResponse, error)

	// MaxBatchSize returns the maximum number of inputs the provider
	// accepts in a single Embed call.
	//
	// Known values:
	//   - Gemini text-embedding-004 / gemini-embedding-001: 100
	//   - OpenAI text-embedding-3-small / text-embedding-3-large: 2048
	MaxBatchSize() int
}
