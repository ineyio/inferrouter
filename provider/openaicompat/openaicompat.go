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

// NewOpenAI creates a provider for OpenAI.
func NewOpenAI(opts ...Option) *Provider {
	return New("openai", "https://api.openai.com/v1", opts...)
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
	TopP        *float64     `json:"top_p,omitempty"`
	Stream      bool         `json:"stream,omitempty"`
	Stop        []string     `json:"stop,omitempty"`
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
	Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
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
	Usage *struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage,omitempty"`
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
		Usage: inferrouter.Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
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
	return apiRequest{
		Model:       req.Model,
		Messages:    msgs,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		TopP:        req.TopP,
		Stream:      stream,
		Stop:        req.Stop,
	}
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
		return nil, inferrouter.ErrProviderUnavailable
	}

	return resp, nil
}

func mapHTTPError(resp *http.Response) error {
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}

	// Read body for error context, but don't fail if we can't.
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
	resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusTooManyRequests:
		return inferrouter.ErrRateLimited
	case http.StatusUnauthorized, http.StatusForbidden:
		return inferrouter.ErrAuthFailed
	case http.StatusBadRequest:
		return fmt.Errorf("%w: %s", inferrouter.ErrInvalidRequest, string(body))
	default:
		return inferrouter.ErrProviderUnavailable
	}
}

// sseStream parses Server-Sent Events from an HTTP response body.
type sseStream struct {
	reader *bufio.Reader
	body   io.ReadCloser
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
			continue // skip malformed chunks
		}

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
			result.Usage = &inferrouter.Usage{
				PromptTokens:     chunk.Usage.PromptTokens,
				CompletionTokens: chunk.Usage.CompletionTokens,
				TotalTokens:      chunk.Usage.TotalTokens,
			}
		}

		return result, nil
	}
}

func (s *sseStream) Close() error {
	return s.body.Close()
}
