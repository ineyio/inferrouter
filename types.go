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
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
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
