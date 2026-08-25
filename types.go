package inferrouter

import "encoding/json"

// ChatRequest represents a chat completion request.
type ChatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature *float64  `json:"temperature,omitempty"`
	MaxTokens   *int      `json:"max_tokens,omitempty"`
	TopP        *float64  `json:"top_p,omitempty"`
	Stream      bool      `json:"stream,omitempty"`
	Stop        []string  `json:"stop,omitempty"`

	// ResponseFormat asks the serving endpoint to constrain its own output —
	// the provider-side counterpart of validating an answer after the fact.
	//
	// It is a request, never a guarantee. A ladder step is a gateway, and
	// gateways differ: one honours the field, the next ignores it, a third
	// rejects the request outright. Whether it reached the wire is reported
	// back on RoutingInfo.StructuredOutput, so a caller can tell "asked and
	// applied" from "asked and dropped" — a caller that needs the answer to
	// actually satisfy a schema still has to check it.
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
}

// ResponseFormat is the output constraint asked of the endpoint.
//
// Shaped after the OpenAI chat-completions field, because that is the wire
// format every gateway this library speaks to derives from.
type ResponseFormat struct {
	// Type is the mode: "json_object" promises valid JSON, "json_schema"
	// promises a shape. Adapters pass it through rather than validating it —
	// the vocabulary belongs to the endpoint, and a library that policed it
	// would go stale the first time a provider added a mode.
	Type string `json:"type"`

	// JSONSchema carries the shape for Type "json_schema".
	JSONSchema *JSONSchemaSpec `json:"json_schema,omitempty"`
}

// JSONSchemaSpec is the schema half of a "json_schema" response format.
type JSONSchemaSpec struct {
	// Name labels the schema for the endpoint. Not resolvable and not an
	// address: nothing dereferences it.
	Name string `json:"name"`

	// Strict opts into the provider's restricted schema subset (closed
	// objects, every property required). Off by default because a schema
	// authored against full JSON Schema is usually outside that subset, and a
	// gateway answers that mismatch with a 400 — turning a schema our own
	// validator accepts into a request that cannot be sent at all.
	Strict bool `json:"strict,omitempty"`

	// Schema is the schema itself, passed to the endpoint byte for byte. The
	// caller owns its contents; this library neither compiles nor rewrites it.
	Schema json.RawMessage `json:"schema"`
}

// Message represents a chat message.
//
// For text-only messages, set Content. For multimodal messages (image/audio/video),
// set Parts. If Parts is non-empty, it takes precedence over Content.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content,omitempty"`
	Parts   []Part `json:"parts,omitempty"`
}

// PartType identifies the kind of content in a Part.
type PartType string

const (
	PartText  PartType = "text"
	PartImage PartType = "image"
	PartAudio PartType = "audio"
	PartVideo PartType = "video"
)

// Part is a single content element in a multimodal Message.
//
// For Type=PartText, set Text. For media parts, set MIMEType and Data (raw bytes).
// Provider adapters handle base64 encoding internally — callers pass raw bytes.
type Part struct {
	Type     PartType `json:"type"`
	Text     string   `json:"text,omitempty"`
	MIMEType string   `json:"mime_type,omitempty"`
	Data     []byte   `json:"data,omitempty"`
}

// IsMedia reports whether this part carries non-text media.
func (p Part) IsMedia() bool {
	return p.Type == PartImage || p.Type == PartAudio || p.Type == PartVideo
}

// ChatResponse represents a chat completion response.
type ChatResponse struct {
	ID      string   `json:"id"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
	Model   string   `json:"model"`
	Routing RoutingInfo
}

// Choice represents a single completion choice.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// Usage represents token usage information.
type Usage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`

	// CachedTokens is the subset of PromptTokens served from provider-side
	// context cache. Orthogonal to modality. Observability-only — not
	// subtracted from cost calculation (providers already price cached
	// tokens server-side; subtracting would double-count the discount).
	CachedTokens int64 `json:"cached_tokens,omitempty"`

	// InputBreakdown splits PromptTokens by modality. Nil for providers
	// that don't report it. When non-nil, Text+Audio+Image+Video == PromptTokens.
	InputBreakdown *InputTokenBreakdown `json:"input_breakdown,omitempty"`
}

// InputTokenBreakdown splits PromptTokens by modality.
type InputTokenBreakdown struct {
	Text  int64 `json:"text"`
	Audio int64 `json:"audio"`
	Image int64 `json:"image"`
	Video int64 `json:"video"`
}

// RoutingInfo describes which provider/account served the request.
type RoutingInfo struct {
	Provider  string
	AccountID string
	Model     string
	Attempts  int
	Free      bool

	// StructuredOutput reports whether the serving adapter actually put
	// ChatRequest.ResponseFormat on the wire.
	//
	// It is set by the adapter that serialised the field, never by the router
	// from the fact that the caller asked: an adapter that does not know the
	// field would otherwise be credited with honouring it, and the caller
	// would read a promise nobody made. False therefore covers both "not
	// asked" and "asked, but this endpoint's adapter cannot send it".
	//
	// It says the request carried the constraint — not that the model obeyed.
	StructuredOutput bool
}

// StreamChunk represents a single chunk in a streaming response.
type StreamChunk struct {
	ID      string        `json:"id"`
	Choices []StreamDelta `json:"choices"`
	Model   string        `json:"model"`
	Usage   *Usage        `json:"usage,omitempty"`
}

// StreamDelta represents a delta in a streaming choice.
type StreamDelta struct {
	Index        int    `json:"index"`
	Delta        Delta  `json:"delta"`
	FinishReason string `json:"finish_reason,omitempty"`
}

// Delta represents incremental content in a stream.
type Delta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// IntPtr returns a pointer to the given int.
func IntPtr(v int) *int { return &v }

// Float64Ptr returns a pointer to the given float64.
func Float64Ptr(v float64) *float64 { return &v }
