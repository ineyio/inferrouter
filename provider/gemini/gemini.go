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
	"strconv"
	"strings"
	"time"

	"github.com/ineyio/inferrouter"
)

const defaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"

const (
	// errorBodyLimit is how much of an error response we read. Google puts its
	// RetryInfo in error.details, at the tail of the body, so the old 1KB was
	// enough for a diagnostic string but not enough to see the one number in
	// the response that is actionable.
	errorBodyLimit = 8192

	// errorDetailLimit is how much of that body reaches the error message —
	// unchanged, so error text stays the size callers already log.
	errorDetailLimit = 1024

	// maxRetryAfter caps a provider-supplied delay. A provider is free to name
	// an absurd number, and a caller parked on it is indistinguishable from a
	// hang. Observed Gemini values sit around 30s, so the cap is set above
	// them rather than through them.
	maxRetryAfter = 60 * time.Second
)

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
	body, err := io.ReadAll(io.LimitReader(resp.Body, errorBodyLimit))
	resp.Body.Close()

	detail := ""
	if err == nil && len(body) > 0 {
		detail = string(body)
		if len(detail) > errorDetailLimit {
			detail = detail[:errorDetailLimit]
		}
	} else {
		detail = http.StatusText(resp.StatusCode)
	}

	switch resp.StatusCode {
	case http.StatusTooManyRequests:
		return &inferrouter.RateLimitedError{
			RetryAfter: parseRetryAfter(resp.Header, body, time.Now()),
			Detail:     detail,
		}
	case http.StatusUnauthorized, http.StatusForbidden:
		return fmt.Errorf("%w: %s", inferrouter.ErrAuthFailed, detail)
	case http.StatusBadRequest:
		return fmt.Errorf("%w: %s", inferrouter.ErrInvalidRequest, detail)
	default:
		return fmt.Errorf("%w: HTTP %d: %s", inferrouter.ErrProviderUnavailable, resp.StatusCode, detail)
	}
}

// parseRetryAfter extracts how long the provider wants us to wait from the two
// places Google puts it: the standard Retry-After header and a google.rpc
// RetryInfo detail in the JSON body. They can disagree, so the longer one
// wins — under-waiting just earns another 429.
//
// Zero means the provider said nothing, which is not the same as "retry now":
// the consumer falls back to its own backoff.
func parseRetryAfter(h http.Header, body []byte, now time.Time) time.Duration {
	delay := headerRetryAfter(h.Get("Retry-After"), now)
	if fromBody := bodyRetryDelay(body); fromBody > delay {
		delay = fromBody
	}

	if delay < 0 {
		return 0
	}
	if delay > maxRetryAfter {
		return maxRetryAfter
	}
	return delay
}

// headerRetryAfter reads a Retry-After header, which RFC 9110 allows to be
// either a count of seconds or an HTTP-date.
func headerRetryAfter(value string, now time.Time) time.Duration {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0
	}

	if seconds, err := strconv.Atoi(value); err == nil {
		if seconds <= 0 {
			return 0
		}
		return time.Duration(seconds) * time.Second
	}

	if when, err := http.ParseTime(value); err == nil {
		if d := when.Sub(now); d > 0 {
			return d
		}
	}
	return 0
}

// bodyRetryDelay reads google.rpc.RetryInfo out of an error body. The detail
// is matched by @type suffix rather than by position, because Google orders
// details freely and adds new kinds without notice.
//
// A truncated or unparseable body yields zero, never an error: this is a hint,
// and losing it costs a default backoff, not a request.
func bodyRetryDelay(body []byte) time.Duration {
	var payload struct {
		Error struct {
			Details []struct {
				Type       string `json:"@type"`
				RetryDelay string `json:"retryDelay"`
			} `json:"details"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &payload) != nil {
		return 0
	}

	for _, detail := range payload.Error.Details {
		if !strings.Contains(detail.Type, "RetryInfo") {
			continue
		}
		if d, err := time.ParseDuration(detail.RetryDelay); err == nil && d > 0 {
			return d
		}
	}
	return 0
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
