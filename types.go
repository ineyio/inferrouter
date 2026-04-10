package inferrouter

// ChatRequest represents a chat completion request.
type ChatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature *float64  `json:"temperature,omitempty"`
	MaxTokens   *int      `json:"max_tokens,omitempty"`
	TopP        *float64  `json:"top_p,omitempty"`
	Stream      bool      `json:"stream,omitempty"`
	Stop        []string  `json:"stop,omitempty"`
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
