package mock

import (
	"context"
	"sync/atomic"
	"time"

	"github.com/ineyio/inferrouter"
)

// EmbedProvider is a mock embedding provider for testing.
//
// Kept as a separate struct (not merged into the existing chat Provider)
// so that chat-only tests and embed-only tests stay independent and so
// callers can compose providers however they like. When a test wants a
// single struct implementing BOTH chat and embed, use DualProvider below.
type EmbedProvider struct {
	name              string
	models            []string
	maxBatch          int
	latency           time.Duration
	callCount         atomic.Int64
	staticErr         error
	tokensPerInput    int64
	embedFn           func(inferrouter.EmbedProviderRequest) (inferrouter.EmbedProviderResponse, error)
	dimensionsDefault int
}

var _ inferrouter.EmbeddingProvider = (*EmbedProvider)(nil)

// EmbedOption configures a mock EmbedProvider.
type EmbedOption func(*EmbedProvider)

// NewEmbed creates a mock embedding provider with the given options.
func NewEmbed(opts ...EmbedOption) *EmbedProvider {
	p := &EmbedProvider{
		name:              "mock-embed",
		models:            []string{"mock-embedding"},
		maxBatch:          100,
		tokensPerInput:    4,
		dimensionsDefault: 8,
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

// WithEmbedName sets the provider name.
func WithEmbedName(name string) EmbedOption {
	return func(p *EmbedProvider) { p.name = name }
}

// WithEmbedSupportedModels sets the list of supported embedding models.
func WithEmbedSupportedModels(models ...string) EmbedOption {
	return func(p *EmbedProvider) { p.models = models }
}

// WithEmbedMaxBatch sets the maximum batch size (default 100).
func WithEmbedMaxBatch(n int) EmbedOption {
	return func(p *EmbedProvider) { p.maxBatch = n }
}

// WithEmbedLatency adds simulated latency to each Embed call.
func WithEmbedLatency(d time.Duration) EmbedOption {
	return func(p *EmbedProvider) { p.latency = d }
}

// WithEmbedError makes the provider always return this error.
func WithEmbedError(err error) EmbedOption {
	return func(p *EmbedProvider) { p.staticErr = err }
}

// WithEmbedResponseFunc sets a custom response function for full control
// over the returned embeddings/usage.
func WithEmbedResponseFunc(fn func(inferrouter.EmbedProviderRequest) (inferrouter.EmbedProviderResponse, error)) EmbedOption {
	return func(p *EmbedProvider) { p.embedFn = fn }
}

// WithEmbedDimensions sets the default embedding dimensions when using the
// deterministic fake embeddings (no responseFunc set).
func WithEmbedDimensions(n int) EmbedOption {
	return func(p *EmbedProvider) { p.dimensionsDefault = n }
}

// WithEmbedTokensPerInput sets how many tokens each input counts for in
// the reported Usage. Default 4.
func WithEmbedTokensPerInput(n int64) EmbedOption {
	return func(p *EmbedProvider) { p.tokensPerInput = n }
}

func (p *EmbedProvider) Name() string { return p.name }

func (p *EmbedProvider) MaxBatchSize() int { return p.maxBatch }

func (p *EmbedProvider) SupportsEmbeddingModel(model string) bool {
	for _, m := range p.models {
		if m == model {
			return true
		}
	}
	return false
}

func (p *EmbedProvider) Embed(ctx context.Context, req inferrouter.EmbedProviderRequest) (inferrouter.EmbedProviderResponse, error) {
	if p.latency > 0 {
		select {
		case <-time.After(p.latency):
		case <-ctx.Done():
			return inferrouter.EmbedProviderResponse{}, ctx.Err()
		}
	}

	p.callCount.Add(1)

	if p.staticErr != nil {
		return inferrouter.EmbedProviderResponse{}, p.staticErr
	}

	if p.embedFn != nil {
		return p.embedFn(req)
	}

	// Deterministic fake embeddings: hash of input text into a fixed-size
	// float32 slice. Same input always yields same vector (useful for
	// consumer tests that check ordering and stability).
	dims := p.dimensionsDefault
	if req.OutputDimensionality > 0 && req.OutputDimensionality < dims {
		dims = req.OutputDimensionality
	}
	embeddings := make([][]float32, len(req.Inputs))
	for i, input := range req.Inputs {
		embeddings[i] = fakeEmbedding(input, dims)
	}

	return inferrouter.EmbedProviderResponse{
		Embeddings: embeddings,
		Model:      req.Model,
		Usage: inferrouter.EmbedUsage{
			InputTokens: int64(len(req.Inputs)) * p.tokensPerInput,
			TotalTokens: int64(len(req.Inputs)) * p.tokensPerInput,
		},
	}, nil
}

// CallCount returns the number of Embed calls made to this provider.
func (p *EmbedProvider) CallCount() int64 { return p.callCount.Load() }

// fakeEmbedding produces a deterministic fixed-length vector from a text.
// Not cryptographically meaningful — just reproducible for assertions.
func fakeEmbedding(text string, dims int) []float32 {
	vec := make([]float32, dims)
	var h uint32 = 2166136261
	for i := 0; i < len(text); i++ {
		h ^= uint32(text[i])
		h *= 16777619
	}
	for i := range vec {
		h ^= h >> 13
		h *= 16777619
		// Map uint32 to [-1, 1] float32.
		vec[i] = float32(int32(h))/float32(1<<31) - 0.5
	}
	return vec
}
