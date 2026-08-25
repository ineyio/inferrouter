package inferrouter

import "context"

// Provider is the interface that LLM provider adapters must implement.
type Provider interface {
	// Name returns the provider identifier (e.g. "gemini", "openai", "grok").
	Name() string

	// SupportsModel returns true if this provider can handle the given model.
	SupportsModel(model string) bool

	// SupportsMultimodal reports whether this provider accepts media parts
	// (image/audio/video) in messages. Text-only providers return false.
	SupportsMultimodal() bool

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

	// HasMedia is precomputed by the router so providers don't need to
	// rewalk Messages/Parts (important on the streaming path where buildUsage
	// fires per chunk).
	HasMedia bool

	// ResponseFormat is the caller's output constraint, or nil.
	//
	// An adapter that cannot express it leaves it alone and reports
	// StructuredOutputApplied false — dropping it silently is allowed, lying
	// about it is not. Ignoring is the right default: the alternative is an
	// error that would take a whole gateway out of a ladder over a field the
	// caller asked for as an improvement, not as a condition.
	ResponseFormat *ResponseFormat
}

// ProviderResponse is the response from a provider adapter.
type ProviderResponse struct {
	ID           string
	Content      string
	FinishReason string
	Usage        Usage
	Model        string

	// StructuredOutputApplied reports that this adapter serialised
	// ProviderRequest.ResponseFormat into the request it sent.
	//
	// Only the code that writes the field may set it — an adapter that never
	// looks at ResponseFormat leaves the zero value, which is the truth about
	// it. The router copies this to RoutingInfo.StructuredOutput; nothing
	// else may.
	StructuredOutputApplied bool
}

// ProviderStream is the interface for streaming responses.
type ProviderStream interface {
	// Next returns the next chunk. Returns io.EOF when done.
	Next() (StreamChunk, error)

	// Close releases resources and signals completion.
	Close() error
}
