package gemini

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

const defaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"

// Provider is the Gemini API adapter.
type Provider struct {
	baseURL    string
	httpClient *http.Client
	models     []string
}

var _ inferrouter.Provider = (*Provider)(nil)

// Option configures the provider.
type Option func(*Provider)

// WithBaseURL sets a custom base URL.
func WithBaseURL(url string) Option {
	return func(p *Provider) { p.baseURL = strings.TrimRight(url, "/") }
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(c *http.Client) Option {
	return func(p *Provider) { p.httpClient = c }
}

// WithModels sets the list of supported models.
func WithModels(models ...string) Option {
	return func(p *Provider) { p.models = models }
}

// New creates a new Gemini provider.
func New(opts ...Option) *Provider {
	p := &Provider{
		baseURL:    defaultBaseURL,
		httpClient: http.DefaultClient,
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

func (p *Provider) Name() string { return "gemini" }

func (p *Provider) SupportsModel(model string) bool {
	if len(p.models) == 0 {
		return true
	}
	for _, m := range p.models {
		if m == model {
			return true
		}
	}
	return false
}

// Gemini API types.
type geminiRequest struct {
	Contents         []geminiContent        `json:"contents"`
	GenerationConfig *geminiGenerationConfig `json:"generationConfig,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text string `json:"text"`
}

type geminiGenerationConfig struct {
	Temperature   *float64 `json:"temperature,omitempty"`
	MaxOutputTokens *int   `json:"maxOutputTokens,omitempty"`
	TopP          *float64 `json:"topP,omitempty"`
	StopSequences []string `json:"stopSequences,omitempty"`
}

type geminiResponse struct {
	Candidates []struct {
		Content      geminiContent `json:"content"`
		FinishReason string        `json:"finishReason"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount     int64 `json:"promptTokenCount"`
		CandidatesTokenCount int64 `json:"candidatesTokenCount"`
		TotalTokenCount      int64 `json:"totalTokenCount"`
	} `json:"usageMetadata"`
	ModelVersion string `json:"modelVersion"`
}

func (p *Provider) ChatCompletion(ctx context.Context, req inferrouter.ProviderRequest) (inferrouter.ProviderResponse, error) {
	body := p.buildRequest(req)
	url := fmt.Sprintf("%s/models/%s:generateContent?key=%s", p.baseURL, req.Model, req.Auth.APIKey)

	httpResp, err := p.doRequest(ctx, url, body)
	if err != nil {
		return inferrouter.ProviderResponse{}, err
	}
	defer httpResp.Body.Close()

	if err := mapHTTPError(httpResp); err != nil {
		return inferrouter.ProviderResponse{}, err
	}

	var resp geminiResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return inferrouter.ProviderResponse{}, fmt.Errorf("inferrouter: decode gemini response: %w", err)
	}

	if len(resp.Candidates) == 0 {
		return inferrouter.ProviderResponse{}, fmt.Errorf("inferrouter: empty candidates in gemini response")
	}

	content := ""
	if len(resp.Candidates[0].Content.Parts) > 0 {
		content = resp.Candidates[0].Content.Parts[0].Text
	}

	return inferrouter.ProviderResponse{
		ID:           "",
		Content:      content,
		FinishReason: strings.ToLower(resp.Candidates[0].FinishReason),
		Model:        req.Model,
		Usage: inferrouter.Usage{
			PromptTokens:     resp.UsageMetadata.PromptTokenCount,
			CompletionTokens: resp.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      resp.UsageMetadata.TotalTokenCount,
		},
	}, nil
}

func (p *Provider) ChatCompletionStream(ctx context.Context, req inferrouter.ProviderRequest) (inferrouter.ProviderStream, error) {
	body := p.buildRequest(req)
	url := fmt.Sprintf("%s/models/%s:streamGenerateContent?alt=sse&key=%s", p.baseURL, req.Model, req.Auth.APIKey)

	httpResp, err := p.doRequest(ctx, url, body)
	if err != nil {
		return nil, err
	}

	if err := mapHTTPError(httpResp); err != nil {
		httpResp.Body.Close()
		return nil, err
	}

	return &geminiStream{
		reader: bufio.NewReader(httpResp.Body),
		body:   httpResp.Body,
		model:  req.Model,
	}, nil
}

func (p *Provider) buildRequest(req inferrouter.ProviderRequest) geminiRequest {
	var contents []geminiContent
	for _, m := range req.Messages {
		role := m.Role
		if role == "assistant" {
			role = "model"
		}
		contents = append(contents, geminiContent{
			Role:  role,
			Parts: []geminiPart{{Text: m.Content}},
		})
	}

	gr := geminiRequest{Contents: contents}

	if req.Temperature != nil || req.MaxTokens != nil || req.TopP != nil || len(req.Stop) > 0 {
		gr.GenerationConfig = &geminiGenerationConfig{
			Temperature:   req.Temperature,
			MaxOutputTokens: req.MaxTokens,
			TopP:          req.TopP,
			StopSequences: req.Stop,
		}
	}

	return gr
}

func (p *Provider) doRequest(ctx context.Context, url string, body geminiRequest) (*http.Response, error) {
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("inferrouter: marshal gemini request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("inferrouter: create gemini request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

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

type geminiStream struct {
	reader *bufio.Reader
	body   io.ReadCloser
	model  string
}

func (s *geminiStream) Next() (inferrouter.StreamChunk, error) {
	for {
		line, err := s.reader.ReadString('\n')
		if err != nil {
			return inferrouter.StreamChunk{}, io.EOF
		}

		line = strings.TrimSpace(line)
		if line == "" || !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")

		var resp geminiResponse
		if err := json.Unmarshal([]byte(data), &resp); err != nil {
			continue
		}

		chunk := inferrouter.StreamChunk{
			Model: s.model,
		}

		if len(resp.Candidates) > 0 && len(resp.Candidates[0].Content.Parts) > 0 {
			chunk.Choices = []inferrouter.StreamDelta{
				{
					Index: 0,
					Delta: inferrouter.Delta{Content: resp.Candidates[0].Content.Parts[0].Text},
				},
			}
			if resp.Candidates[0].FinishReason != "" {
				chunk.Choices[0].FinishReason = strings.ToLower(resp.Candidates[0].FinishReason)
			}
		}

		if resp.UsageMetadata.TotalTokenCount > 0 {
			chunk.Usage = &inferrouter.Usage{
				PromptTokens:     resp.UsageMetadata.PromptTokenCount,
				CompletionTokens: resp.UsageMetadata.CandidatesTokenCount,
				TotalTokens:      resp.UsageMetadata.TotalTokenCount,
			}
		}

		return chunk, nil
	}
}

func (s *geminiStream) Close() error {
	return s.body.Close()
}
