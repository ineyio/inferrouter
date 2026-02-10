package inferrouter

import "context"

// Provider is the interface that LLM provider adapters must implement.
type Provider interface {
	// Name returns the provider identifier (e.g. "gemini", "openai", "grok").
	Name() string

	// SupportsModel returns true if this provider can handle the given model.
	SupportsModel(model string) bool

	// ChatCompletion performs a synchronous chat completion.
	ChatCompletion(ctx context.Context, req ProviderRequest) (ProviderResponse, error)

	// ChatCompletionStream performs a streaming chat completion.
	ChatCompletionStream(ctx context.Context, req ProviderRequest) (ProviderStream, error)
}

// Auth holds authentication credentials for a provider account.
type Auth struct {
	APIKey string `yaml:"api_key" json:"api_key"`
}

// ProviderRequest is the request sent to a provider adapter.
type ProviderRequest struct {
	Auth     Auth
	Model    string
	Messages []Message

	Temperature *float64
	MaxTokens   *int
	TopP        *float64
	Stop        []string
	Stream      bool
}

// ProviderResponse is the response from a provider adapter.
type ProviderResponse struct {
	ID           string
	Content      string
	FinishReason string
	Usage        Usage
	Model        string
}

// ProviderStream is the interface for streaming responses.
type ProviderStream interface {
	// Next returns the next chunk. Returns io.EOF when done.
	Next() (StreamChunk, error)

	// Close releases resources and signals completion.
	Close() error
}
