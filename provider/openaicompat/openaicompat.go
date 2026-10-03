package openaicompat

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/ineyio/inferrouter"
)

// Provider is a universal OpenAI-compatible API adapter.
// Works with OpenAI, Grok/xAI, Cerebras, Together, Ollama, and others.
type Provider struct {
	name       string
	baseURL    string
	httpClient *http.Client
	models     []string

	// maxCompletionTokens sends the output ceiling as max_completion_tokens.
	// It is a property of the endpoint, not of the model: OpenAI refuses
	// max_tokens with a 400 on every model that thinks and accepts the new
	// spelling on every chat model, while gateways (Gonka resellers,
	// Cerebras, Grok) speak the old one.
	maxCompletionTokens bool
}

var _ inferrouter.Provider = (*Provider)(nil)

// Option configures the provider.
type Option func(*Provider)

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(c *http.Client) Option {
	return func(p *Provider) { p.httpClient = c }
}

// WithModels sets the list of supported models.
func WithModels(models ...string) Option {
	return func(p *Provider) { p.models = models }
}

// WithMaxCompletionTokens sends ProviderRequest.MaxTokens as
// max_completion_tokens instead of max_tokens. NewOpenAI and FromAccounts set
// it for api.openai.com themselves; an endpoint elsewhere that wants the new
// spelling opts in here.
func WithMaxCompletionTokens() Option {
	return func(p *Provider) { p.maxCompletionTokens = true }
}

// New creates a new OpenAI-compatible provider.
func New(name, baseURL string, opts ...Option) *Provider {
	p := &Provider{
		name:       name,
		baseURL:    strings.TrimRight(baseURL, "/"),
		httpClient: http.DefaultClient,
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

// openAIBaseURL is OpenAI's own endpoint. Anything served from its host takes
// max_completion_tokens, however the provider was constructed.
const openAIBaseURL = "https://api.openai.com/v1"

// NewOpenAI creates a provider for OpenAI.
func NewOpenAI(opts ...Option) *Provider {
	return New("openai", openAIBaseURL, append([]Option{WithMaxCompletionTokens()}, opts...)...)
}

// NewGrok creates a provider for Grok/xAI.
func NewGrok(opts ...Option) *Provider {
	return New("grok", "https://api.x.ai/v1", opts...)
}

// NewCerebro creates a provider for Cerebras.
func NewCerebro(opts ...Option) *Provider {
	return New("cerebro", "https://api.cerebras.ai/v1", opts...)
}

func (p *Provider) Name() string { return p.name }

// SupportsMultimodal returns false. OpenAI-compatible backends support a
// content[] array format for vision, but this provider does not yet serialize
// Parts into that format — advertising true would silently strip media.
func (p *Provider) SupportsMultimodal() bool { return false }

func (p *Provider) SupportsModel(model string) bool {
	if len(p.models) == 0 {
		return true // no filter → accept all
	}
	for _, m := range p.models {
		if m == model {
			return true
		}
	}
	return false
}

// apiRequest is the OpenAI chat completion request format.
type apiRequest struct {
	Model       string       `json:"model"`
	Messages    []apiMessage `json:"messages"`
	Temperature *float64     `json:"temperature,omitempty"`
	MaxTokens   *int         `json:"max_tokens,omitempty"`

	// max_completion_tokens is the same ceiling in the spelling OpenAI
	// requires. At most one of the two is set: which one is the endpoint's
	// choice (Provider.maxCompletionTokens), never the caller's.
	MaxCompletionTokens *int `json:"max_completion_tokens,omitempty"`

	TopP   *float64 `json:"top_p,omitempty"`
	Stream bool     `json:"stream,omitempty"`
	Stop   []string `json:"stop,omitempty"`

	// stream_options is the only way to be told what a streamed answer cost.
	// Without it an OpenAI-compatible gateway ends the stream with no usage
	// chunk at all, and every number downstream reads zero: the quota commit
	// on Close, the spend tracker, and any cap built on them. That is not a
	// missing metric, it is a cap that never fires — so the ask travels on
	// every stream, and the pointer is nil on the unary path where the field
	// has no meaning.
	StreamOptions *apiStreamOptions `json:"stream_options,omitempty"`

	// omitempty on a pointer, so a request without a format carries no
	// response_format key at all. A present-but-null key is a different thing
	// to a gateway than an absent one, and older endpoints reject it.
	ResponseFormat *inferrouter.ResponseFormat `json:"response_format,omitempty"`

	// reasoning_effort is the OpenAI-side spelling of a thinking budget.
	// Same pointer-with-omitempty reason as above: an endpoint that has
	// never heard of the field must receive no key, not a null one.
	//
	// Only the effort travels. There is no portable spelling for "hand back
	// the thinking" here — endpoints that return it do so in their own
	// field — so IncludeSummary is not claimed and not reported.
	ReasoningEffort *string `json:"reasoning_effort,omitempty"`
}

// apiStreamOptions carries the usage ask. A struct rather than a bare bool
// because this is the endpoint's own spelling, and the object is where the
// OpenAI API puts any later stream-scoped flag.
type apiStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type apiMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// apiResponse is the OpenAI chat completion response format.
type apiResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int        `json:"index"`
		Message      apiMessage `json:"message"`
		FinishReason string     `json:"finish_reason"`
	} `json:"choices"`
	Usage apiUsage `json:"usage"`
}

// apiUsage is the usage object, identical on the unary body and on the final
// SSE chunk.
type apiUsage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`

	// Absent on gateways that do not report it — and absent reads as zero
	// thinking, which is what such a gateway is telling us.
	CompletionTokensDetails *struct {
		ReasoningTokens int64 `json:"reasoning_tokens"`
	} `json:"completion_tokens_details,omitempty"`
}

// toUsage maps the OpenAI accounting onto ours, and the two disagree on one
// point: OpenAI counts thinking INSIDE completion_tokens and breaks it out in
// completion_tokens_details, while inferrouter.Usage keeps ReasoningTokens
// apart from CompletionTokens (Gemini's shape) and calculateSpend adds the
// two. Copying both numbers as they come would bill the thinking twice, so
// the thinking is moved, not copied.
//
// A reasoning count larger than completion_tokens cannot be a breakdown of
// it; such an endpoint reports thinking separately already, and is taken at
// its word rather than driven negative.
func (u apiUsage) toUsage() inferrouter.Usage {
	out := inferrouter.Usage{
		PromptTokens:     u.PromptTokens,
		CompletionTokens: u.CompletionTokens,
		TotalTokens:      u.TotalTokens,
	}
	if u.CompletionTokensDetails == nil {
		return out
	}
	r := u.CompletionTokensDetails.ReasoningTokens
	out.ReasoningTokens = r
	if r <= u.CompletionTokens {
		out.CompletionTokens -= r
	}
	return out
}

// apiStreamChunk is a single SSE chunk.
type apiStreamChunk struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Role    string `json:"role,omitempty"`
			Content string `json:"content,omitempty"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason,omitempty"`
	} `json:"choices"`
	Usage *apiUsage `json:"usage,omitempty"`
}

func (p *Provider) ChatCompletion(ctx context.Context, req inferrouter.ProviderRequest) (inferrouter.ProviderResponse, error) {
	body := p.buildRequest(req, false)

	httpResp, err := p.doRequest(ctx, req.Auth, body)
	if err != nil {
		return inferrouter.ProviderResponse{}, err
	}
	defer httpResp.Body.Close()

	if err := mapHTTPError(httpResp); err != nil {
		return inferrouter.ProviderResponse{}, err
	}

	var resp apiResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return inferrouter.ProviderResponse{}, fmt.Errorf("inferrouter: decode response: %w", err)
	}

	if len(resp.Choices) == 0 {
		return inferrouter.ProviderResponse{}, fmt.Errorf("inferrouter: empty choices in response")
	}

	return inferrouter.ProviderResponse{
		ID:           resp.ID,
		Content:      resp.Choices[0].Message.Content,
		FinishReason: resp.Choices[0].FinishReason,
		Model:        resp.Model,
		// Read off the body we built, not off the argument: this is the claim
		// "the constraint went out on the wire", and the body is the wire. The
		// endpoint accepted it — whether it then honoured it is not knowable
		// from here, and is not what this field says.
		StructuredOutputApplied: body.ResponseFormat != nil,
		ReasoningApplied:        body.ReasoningEffort != nil,
		Usage:                   resp.Usage.toUsage(),
	}, nil
}

func (p *Provider) ChatCompletionStream(ctx context.Context, req inferrouter.ProviderRequest) (inferrouter.ProviderStream, error) {
	body := p.buildRequest(req, true)

	httpResp, err := p.doRequest(ctx, req.Auth, body)
	if err != nil {
		return nil, err
	}

	if err := mapHTTPError(httpResp); err != nil {
		httpResp.Body.Close()
		return nil, err
	}

	return &sseStream{
		reader: bufio.NewReader(httpResp.Body),
		body:   httpResp.Body,
	}, nil
}

func (p *Provider) buildRequest(req inferrouter.ProviderRequest, stream bool) apiRequest {
	msgs := make([]apiMessage, len(req.Messages))
	for i, m := range req.Messages {
		msgs[i] = apiMessage{Role: m.Role, Content: m.Content}
	}
	body := apiRequest{
		Model:       req.Model,
		Messages:    msgs,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      stream,
		Stop:        req.Stop,
		// Only on the streaming path: on a unary call the endpoint reports
		// usage in the body anyway, and an unasked-for key is one more thing
		// an older gateway can refuse.
		StreamOptions: streamOptions(stream),
		// Passed through unchanged, schema bytes included: the caller wrote
		// that schema and is the one who will be told whether it held.
		ResponseFormat:  req.ResponseFormat,
		ReasoningEffort: reasoningEffort(req.Reasoning),
	}
	if p.maxCompletionTokens {
		body.MaxCompletionTokens = req.MaxTokens
	} else {
		body.MaxTokens = req.MaxTokens
	}
	return body
}

// streamOptions asks for the usage chunk, and only when there is a stream to
// ask about.
func streamOptions(stream bool) *apiStreamOptions {
	if !stream {
		return nil
	}
	return &apiStreamOptions{IncludeUsage: true}
}

// reasoningEffort maps the router-level ask onto the OpenAI field, and sends
// nothing when the caller only asked for a summary: half a request answered
// silently is what ReasoningApplied exists to prevent.
func reasoningEffort(r *inferrouter.ReasoningConfig) *string {
	if r == nil || r.Effort == "" {
		return nil
	}
	e := r.Effort
	return &e
}

func (p *Provider) doRequest(ctx context.Context, auth inferrouter.Auth, body apiRequest) (*http.Response, error) {
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("inferrouter: marshal request: %w", err)
	}

	url := p.baseURL + "/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("inferrouter: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+auth.APIKey)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		// Keep the transport error in the chain: "connection refused", a DNS
		// failure and a deadline are three different outages, and this text is
		// the only trace of which one it was once the attempt is recorded.
		return nil, fmt.Errorf("%w: %v", inferrouter.ErrProviderUnavailable, err)
	}

	return resp, nil
}

func mapHTTPError(resp *http.Response) error {
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}

	// Best-effort body read for diagnostics.
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1024))
	resp.Body.Close()

	detail := ""
	if err == nil && len(body) > 0 {
		detail = string(body)
	} else {
		detail = http.StatusText(resp.StatusCode)
	}

	switch resp.StatusCode {
	case http.StatusTooManyRequests:
		return fmt.Errorf("%w: %s", inferrouter.ErrRateLimited, detail)
	case http.StatusUnauthorized, http.StatusForbidden:
		return fmt.Errorf("%w: %s", inferrouter.ErrAuthFailed, detail)
	case http.StatusBadRequest:
		return fmt.Errorf("%w: %s", inferrouter.ErrInvalidRequest, detail)
	default:
		return fmt.Errorf("%w: HTTP %d: %s", inferrouter.ErrProviderUnavailable, resp.StatusCode, detail)
	}
}

// sseStream parses Server-Sent Events from an HTTP response body.
type sseStream struct {
	reader    *bufio.Reader
	body      io.ReadCloser
	parseErrs int // consecutive parse errors
}

func (s *sseStream) Next() (inferrouter.StreamChunk, error) {
	for {
		line, err := s.reader.ReadString('\n')
		if err != nil {
			return inferrouter.StreamChunk{}, io.EOF
		}

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			return inferrouter.StreamChunk{}, io.EOF
		}

		var chunk apiStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			s.parseErrs++
			if s.parseErrs >= 3 {
				return inferrouter.StreamChunk{}, fmt.Errorf("inferrouter: %d consecutive malformed SSE chunks: %w", s.parseErrs, err)
			}
			continue
		}
		s.parseErrs = 0

		result := inferrouter.StreamChunk{
			ID:    chunk.ID,
			Model: chunk.Model,
		}

		for _, c := range chunk.Choices {
			result.Choices = append(result.Choices, inferrouter.StreamDelta{
				Index:        c.Index,
				Delta:        inferrouter.Delta{Role: c.Delta.Role, Content: c.Delta.Content},
				FinishReason: c.FinishReason,
			})
		}

		if chunk.Usage != nil {
			u := chunk.Usage.toUsage()
			result.Usage = &u
		}

		return result, nil
	}
}

func (s *sseStream) Close() error {
	return s.body.Close()
}
