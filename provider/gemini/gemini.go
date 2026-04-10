package gemini

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
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
	logger     *slog.Logger
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

// WithLogger sets a logger for warnings (e.g. missing promptTokensDetails).
// If not set, slog.Default() is used.
func WithLogger(l *slog.Logger) Option {
	return func(p *Provider) { p.logger = l }
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
	if p.logger == nil {
		p.logger = slog.Default()
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

// SupportsMultimodal reports that Gemini accepts media parts (image/audio/video).
func (p *Provider) SupportsMultimodal() bool { return true }

// Gemini API types.
type geminiRequest struct {
	Contents         []geminiContent         `json:"contents"`
	GenerationConfig *geminiGenerationConfig `json:"generationConfig,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text       string            `json:"text,omitempty"`
	InlineData *geminiInlineData `json:"inline_data,omitempty"`
}

type geminiInlineData struct {
	MIMEType string `json:"mime_type"`
	Data     string `json:"data"` // base64-encoded
}

type geminiGenerationConfig struct {
	Temperature     *float64 `json:"temperature,omitempty"`
	MaxOutputTokens *int     `json:"maxOutputTokens,omitempty"`
	TopP            *float64 `json:"topP,omitempty"`
	StopSequences   []string `json:"stopSequences,omitempty"`
}

type geminiTokenDetail struct {
	Modality   string `json:"modality"`
	TokenCount int64  `json:"tokenCount"`
}

type geminiUsageMetadata struct {
	PromptTokenCount        int64               `json:"promptTokenCount"`
	CandidatesTokenCount    int64               `json:"candidatesTokenCount"`
	TotalTokenCount         int64               `json:"totalTokenCount"`
	CachedContentTokenCount int64               `json:"cachedContentTokenCount"`
	PromptTokensDetails     []geminiTokenDetail `json:"promptTokensDetails"`
}

type geminiResponse struct {
	Candidates []struct {
		Content      geminiContent `json:"content"`
		FinishReason string        `json:"finishReason"`
	} `json:"candidates"`
	UsageMetadata geminiUsageMetadata `json:"usageMetadata"`
	ModelVersion  string              `json:"modelVersion"`
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
		Usage:        p.buildUsage(resp.UsageMetadata, req),
	}, nil
}

// buildUsage maps Gemini usageMetadata to inferrouter.Usage.
//
// When promptTokensDetails is absent for a text-only request, we synthesize
// {Text: PromptTokens} so callers have a single non-nil breakdown code path.
// When absent for a multimodal request it's left nil and a warning is logged
// — that combination signals Gemini API drift and the caller should notice.
//
// CachedTokens is copied as-is; it's a subset of PromptTokens that must NOT
// reduce the cost (Google already priced it server-side; subtracting would
// double-count).
func (p *Provider) buildUsage(meta geminiUsageMetadata, req inferrouter.ProviderRequest) inferrouter.Usage {
	u := inferrouter.Usage{
		PromptTokens:     meta.PromptTokenCount,
		CompletionTokens: meta.CandidatesTokenCount,
		TotalTokens:      meta.TotalTokenCount,
		CachedTokens:     meta.CachedContentTokenCount,
	}

	if len(meta.PromptTokensDetails) > 0 {
		u.InputBreakdown = p.buildBreakdown(meta.PromptTokensDetails)
		return u
	}

	if !req.HasMedia {
		u.InputBreakdown = &inferrouter.InputTokenBreakdown{Text: meta.PromptTokenCount}
		return u
	}

	p.logger.Warn("gemini response missing promptTokensDetails for multimodal request",
		"prompt_tokens", meta.PromptTokenCount,
		"model", req.Model,
	)
	return u
}

// buildBreakdown converts Gemini's per-modality details into our struct.
// Unknown modalities (e.g. a future DOCUMENT type) fold into Text with a warning.
func (p *Provider) buildBreakdown(details []geminiTokenDetail) *inferrouter.InputTokenBreakdown {
	b := &inferrouter.InputTokenBreakdown{}
	for _, d := range details {
		switch strings.ToUpper(d.Modality) {
		case "TEXT":
			b.Text += d.TokenCount
		case "AUDIO":
			b.Audio += d.TokenCount
		case "IMAGE":
			b.Image += d.TokenCount
		case "VIDEO":
			b.Video += d.TokenCount
		default:
			p.logger.Warn("gemini: unknown modality in promptTokensDetails, folding into Text",
				"modality", d.Modality, "tokens", d.TokenCount)
			b.Text += d.TokenCount
		}
	}
	return b
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
		req:    req,
		prov:   p,
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
			Parts: buildParts(m),
		})
	}

	gr := geminiRequest{Contents: contents}

	if req.Temperature != nil || req.MaxTokens != nil || req.TopP != nil || len(req.Stop) > 0 {
		gr.GenerationConfig = &geminiGenerationConfig{
			Temperature:     req.Temperature,
			MaxOutputTokens: req.MaxTokens,
			TopP:            req.TopP,
			StopSequences:   req.Stop,
		}
	}

	return gr
}

// buildParts maps inferrouter.Message to Gemini parts. If m.Parts is empty,
// falls back to m.Content as a single text part (legacy path).
func buildParts(m inferrouter.Message) []geminiPart {
	if len(m.Parts) == 0 {
		return []geminiPart{{Text: m.Content}}
	}
	parts := make([]geminiPart, 0, len(m.Parts))
	for _, p := range m.Parts {
		switch p.Type {
		case inferrouter.PartText:
			parts = append(parts, geminiPart{Text: p.Text})
		case inferrouter.PartImage, inferrouter.PartAudio, inferrouter.PartVideo:
			parts = append(parts, geminiPart{
				InlineData: &geminiInlineData{
					MIMEType: p.MIMEType,
					Data:     base64.StdEncoding.EncodeToString(p.Data),
				},
			})
		}
	}
	return parts
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

type geminiStream struct {
	reader    *bufio.Reader
	body      io.ReadCloser
	model     string
	req       inferrouter.ProviderRequest
	prov      *Provider
	parseErrs int // consecutive parse errors
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
			s.parseErrs++
			if s.parseErrs >= 3 {
				return inferrouter.StreamChunk{}, fmt.Errorf("inferrouter: %d consecutive malformed SSE chunks: %w", s.parseErrs, err)
			}
			continue
		}
		s.parseErrs = 0

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
			u := s.prov.buildUsage(resp.UsageMetadata, s.req)
			chunk.Usage = &u
		}

		return chunk, nil
	}
}

func (s *geminiStream) Close() error {
	return s.body.Close()
}
