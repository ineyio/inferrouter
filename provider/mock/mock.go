package mock

import (
	"context"
	"io"
	"sync/atomic"
	"time"

	"github.com/ineyio/inferrouter"
)

// Provider is a mock LLM provider for testing.
type Provider struct {
	name        string
	models      []string
	latency     time.Duration
	failAfter   int
	callCount   atomic.Int64
	staticErr   error
	usage       inferrouter.Usage
	responseFunc func(inferrouter.ProviderRequest) (inferrouter.ProviderResponse, error)
}

var _ inferrouter.Provider = (*Provider)(nil)

// Option configures a mock Provider.
type Option func(*Provider)

// New creates a mock provider with the given options.
func New(opts ...Option) *Provider {
	p := &Provider{
		name:   "mock",
		models: []string{"mock-model"},
		usage: inferrouter.Usage{
			PromptTokens:     10,
			CompletionTokens: 20,
			TotalTokens:      30,
		},
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

// WithName sets the provider name.
func WithName(name string) Option {
	return func(p *Provider) { p.name = name }
}

// WithModels sets supported models.
func WithModels(models ...string) Option {
	return func(p *Provider) { p.models = models }
}

// WithLatency adds simulated latency to each call.
func WithLatency(d time.Duration) Option {
	return func(p *Provider) { p.latency = d }
}

// WithFailAfter makes the provider fail after N successful calls.
func WithFailAfter(n int) Option {
	return func(p *Provider) { p.failAfter = n }
}

// WithError makes the provider always return this error.
func WithError(err error) Option {
	return func(p *Provider) { p.staticErr = err }
}

// WithUsage sets the usage returned by the mock.
func WithUsage(u inferrouter.Usage) Option {
	return func(p *Provider) { p.usage = u }
}

// WithResponseFunc sets a custom response function.
func WithResponseFunc(fn func(inferrouter.ProviderRequest) (inferrouter.ProviderResponse, error)) Option {
	return func(p *Provider) { p.responseFunc = fn }
}

func (p *Provider) Name() string { return p.name }

func (p *Provider) SupportsModel(model string) bool {
	for _, m := range p.models {
		if m == model {
			return true
		}
	}
	return false
}

func (p *Provider) ChatCompletion(ctx context.Context, req inferrouter.ProviderRequest) (inferrouter.ProviderResponse, error) {
	if p.latency > 0 {
		select {
		case <-time.After(p.latency):
		case <-ctx.Done():
			return inferrouter.ProviderResponse{}, ctx.Err()
		}
	}

	count := p.callCount.Add(1)

	if p.staticErr != nil {
		return inferrouter.ProviderResponse{}, p.staticErr
	}

	if p.failAfter > 0 && int(count) > p.failAfter {
		return inferrouter.ProviderResponse{}, inferrouter.ErrProviderUnavailable
	}

	if p.responseFunc != nil {
		return p.responseFunc(req)
	}

	return inferrouter.ProviderResponse{
		ID:           "mock-response-id",
		Content:      "Hello from mock provider",
		FinishReason: "stop",
		Usage:        p.usage,
		Model:        req.Model,
	}, nil
}

func (p *Provider) ChatCompletionStream(ctx context.Context, req inferrouter.ProviderRequest) (inferrouter.ProviderStream, error) {
	resp, err := p.ChatCompletion(ctx, req)
	if err != nil {
		return nil, err
	}

	return &mockStream{
		chunks: []inferrouter.StreamChunk{
			{
				ID:    resp.ID,
				Model: resp.Model,
				Choices: []inferrouter.StreamDelta{
					{Index: 0, Delta: inferrouter.Delta{Role: "assistant"}},
				},
			},
			{
				ID:    resp.ID,
				Model: resp.Model,
				Choices: []inferrouter.StreamDelta{
					{Index: 0, Delta: inferrouter.Delta{Content: resp.Content}},
				},
			},
			{
				ID:    resp.ID,
				Model: resp.Model,
				Choices: []inferrouter.StreamDelta{
					{Index: 0, FinishReason: "stop"},
				},
				Usage: &resp.Usage,
			},
		},
	}, nil
}

// CallCount returns the number of calls made to the provider.
func (p *Provider) CallCount() int64 { return p.callCount.Load() }

type mockStream struct {
	chunks []inferrouter.StreamChunk
	index  int
}

func (s *mockStream) Next() (inferrouter.StreamChunk, error) {
	if s.index >= len(s.chunks) {
		return inferrouter.StreamChunk{}, io.EOF
	}
	chunk := s.chunks[s.index]
	s.index++
	return chunk, nil
}

func (s *mockStream) Close() error { return nil }
