package gonka

import (
	"context"
	"net/http"
	"time"

	"github.com/ineyio/inferrouter"
	"github.com/ineyio/inferrouter/provider/openaicompat"
)

// Provider is the Gonka decentralized AI compute network adapter.
// It composes openaicompat.Provider with ECDSA request signing.
//
// Auth.APIKey is used to pass the hex-encoded secp256k1 private key.
// The signing transport reads it from the Authorization header,
// replaces it with the ECDSA signature, and adds Gonka-specific headers.
type Provider struct {
	inner *openaicompat.Provider
}

var _ inferrouter.Provider = (*Provider)(nil)

// Option configures the Gonka provider.
type Option func(*config)

type config struct {
	name      string
	models    []string
	endpoint  Endpoint
	timeout   time.Duration
	transport http.RoundTripper
	nowFunc   func() time.Time
}

// WithName sets the provider name (default: "gonka").
func WithName(name string) Option {
	return func(c *config) { c.name = name }
}

// WithModels sets the list of supported models.
func WithModels(models ...string) Option {
	return func(c *config) { c.models = models }
}

// WithEndpoint sets the Gonka node endpoint.
func WithEndpoint(e Endpoint) Option {
	return func(c *config) { c.endpoint = e }
}

// WithTimeout sets the HTTP client timeout.
// Default is 120s — generous for Gonka's P2P network.
func WithTimeout(d time.Duration) Option {
	return func(c *config) { c.timeout = d }
}

// WithBaseTransport sets the underlying HTTP transport (before signing).
func WithBaseTransport(rt http.RoundTripper) Option {
	return func(c *config) { c.transport = rt }
}

// withNowFunc is unexported — used in tests for deterministic timestamps.
func withNowFunc(fn func() time.Time) Option {
	return func(c *config) { c.nowFunc = fn }
}

// New creates a new Gonka provider.
func New(opts ...Option) *Provider {
	cfg := &config{
		name:    "gonka",
		timeout: 120 * time.Second,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	base := cfg.transport
	if base == nil {
		base = http.DefaultTransport
	}

	signing := newSigningTransport(base, cfg.endpoint)
	if cfg.nowFunc != nil {
		signing.nowFunc = cfg.nowFunc
	}

	httpClient := &http.Client{
		Transport: signing,
		Timeout:   cfg.timeout,
	}

	innerOpts := []openaicompat.Option{
		openaicompat.WithHTTPClient(httpClient),
	}
	if len(cfg.models) > 0 {
		innerOpts = append(innerOpts, openaicompat.WithModels(cfg.models...))
	}

	inner := openaicompat.New(cfg.name, cfg.endpoint.URL, innerOpts...)

	return &Provider{inner: inner}
}

func (p *Provider) Name() string { return p.inner.Name() }

func (p *Provider) SupportsModel(model string) bool { return p.inner.SupportsModel(model) }

func (p *Provider) ChatCompletion(ctx context.Context, req inferrouter.ProviderRequest) (inferrouter.ProviderResponse, error) {
	return p.inner.ChatCompletion(ctx, req)
}

func (p *Provider) ChatCompletionStream(ctx context.Context, req inferrouter.ProviderRequest) (inferrouter.ProviderStream, error) {
	return p.inner.ChatCompletionStream(ctx, req)
}
