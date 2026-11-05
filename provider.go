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

// ExtraBodySender is implemented by adapters that put ProviderRequest.ExtraBody
// on the wire. A step with extra_body on an adapter without it gets a line in
// Router.ConfigWarnings: the author of the config wrote that field for that
// step and should hear at startup that it goes nowhere.
//
// A wrapper around a Provider must forward this method. A wrapper that does
// not is reported as unable to send, which is the safe way to be wrong.
type ExtraBodySender interface {
	SendsExtraBody() bool
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

	// Reasoning is the caller's thinking budget, or nil. Same rule as
	// ResponseFormat: an adapter that cannot express it leaves it alone and
	// reports ReasoningApplied false. Dropping it silently is allowed,
	// lying about it is not.
	Reasoning *ReasoningConfig

	// ExtraBody is ModelRef.ExtraBody of this step, or nil. Read-only: the map
	// is shared by every request through the step. An adapter that puts it on
	// the wire implements ExtraBodySender and reports ExtraBodyApplied.
	ExtraBody map[string]any
}

// ProviderResponse is the response from a provider adapter.
type ProviderResponse struct {
	ID           string
	Content      string
	FinishReason string
	Usage        Usage
	Model        string

	// ExtraBodyApplied names, sorted, the ProviderRequest.ExtraBody fields this
	// adapter put into the request it sent. Copied by the router to
	// RoutingInfo.ExtraBody.
	ExtraBodyApplied []string

	// StructuredOutputApplied reports that this adapter serialised
	// ProviderRequest.ResponseFormat into the request it sent.
	//
	// Only the code that writes the field may set it — an adapter that never
	// looks at ResponseFormat leaves the zero value, which is the truth about
	// it. The router copies this to RoutingInfo.StructuredOutput; nothing
	// else may.
	StructuredOutputApplied bool

	// ReasoningApplied reports that this adapter serialised
	// ProviderRequest.Reasoning into the request it sent. Copied by the
	// router to RoutingInfo.Reasoning; nothing else may set it.
	ReasoningApplied bool

	// ReasoningSummary is the thinking the model reported about itself,
	// already separated from Content by the adapter that knows the wire
	// shape. Empty when not asked for, not returned, or not separable.
	ReasoningSummary string
}

// ProviderStream is the interface for streaming responses.
type ProviderStream interface {
	// Next returns the next chunk. Returns io.EOF when done.
	Next() (StreamChunk, error)

	// Close releases resources and signals completion.
	Close() error
}
