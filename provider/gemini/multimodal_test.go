package gemini

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ir "github.com/ineyio/inferrouter"
)

func TestSupportsMultimodal(t *testing.T) {
	p := New()
	if !p.SupportsMultimodal() {
		t.Fatal("gemini provider should support multimodal")
	}
}

func TestBuildPartsLegacyContent(t *testing.T) {
	// Backward compat: no Parts → single text part from Content.
	parts := buildParts(ir.Message{Role: "user", Content: "hi"})
	if len(parts) != 1 {
		t.Fatalf("len(parts) = %d, want 1", len(parts))
	}
	if parts[0].Text != "hi" {
		t.Errorf("parts[0].Text = %q, want %q", parts[0].Text, "hi")
	}
	if parts[0].InlineData != nil {
		t.Errorf("parts[0].InlineData = %v, want nil", parts[0].InlineData)
	}
}

func TestBuildPartsMultimodal(t *testing.T) {
	photoBytes := []byte{0xFF, 0xD8, 0xFF, 0xE0} // fake JPEG header
	audioBytes := []byte{0x4F, 0x67, 0x67, 0x53} // fake OGG header

	parts := buildParts(ir.Message{
		Role: "user",
		Parts: []ir.Part{
			{Type: ir.PartText, Text: "describe this"},
			{Type: ir.PartImage, MIMEType: "image/jpeg", Data: photoBytes},
			{Type: ir.PartAudio, MIMEType: "audio/ogg", Data: audioBytes},
		},
	})

	if len(parts) != 3 {
		t.Fatalf("len(parts) = %d, want 3", len(parts))
	}

	if parts[0].Text != "describe this" || parts[0].InlineData != nil {
		t.Errorf("parts[0] = %+v, want text part", parts[0])
	}

	if parts[1].InlineData == nil {
		t.Fatal("parts[1].InlineData is nil")
	}
	if parts[1].InlineData.MIMEType != "image/jpeg" {
		t.Errorf("parts[1].MIMEType = %q", parts[1].InlineData.MIMEType)
	}
	wantImg := base64.StdEncoding.EncodeToString(photoBytes)
	if parts[1].InlineData.Data != wantImg {
		t.Errorf("parts[1] base64 mismatch")
	}

	if parts[2].InlineData == nil || parts[2].InlineData.MIMEType != "audio/ogg" {
		t.Errorf("parts[2] = %+v", parts[2])
	}
	wantAudio := base64.StdEncoding.EncodeToString(audioBytes)
	if parts[2].InlineData.Data != wantAudio {
		t.Errorf("parts[2] base64 mismatch")
	}
}

func TestBuildRequestWithMultimodalPart(t *testing.T) {
	p := New()
	req := p.buildRequest(ir.ProviderRequest{
		Messages: []ir.Message{
			{
				Role: "user",
				Parts: []ir.Part{
					{Type: ir.PartText, Text: "what's this?"},
					{Type: ir.PartImage, MIMEType: "image/png", Data: []byte{1, 2, 3}},
				},
			},
		},
	})
	if len(req.Contents) != 1 {
		t.Fatalf("contents len = %d", len(req.Contents))
	}
	if len(req.Contents[0].Parts) != 2 {
		t.Fatalf("parts len = %d", len(req.Contents[0].Parts))
	}
	// Verify wire format via JSON marshal — should produce inline_data field.
	raw, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(raw), `"inline_data"`) {
		t.Errorf("wire format missing inline_data: %s", raw)
	}
	if !strings.Contains(string(raw), `"mime_type":"image/png"`) {
		t.Errorf("wire format missing mime_type: %s", raw)
	}
}

func TestBuildUsageWithPromptTokensDetails(t *testing.T) {
	p := New()
	req := ir.ProviderRequest{
		HasMedia: true,
		Messages: []ir.Message{
			{Parts: []ir.Part{
				{Type: ir.PartText, Text: "hi"},
				{Type: ir.PartImage, MIMEType: "image/jpeg", Data: []byte{1}},
			}},
		},
	}
	meta := geminiUsageMetadata{
		PromptTokenCount:     1234,
		CandidatesTokenCount: 200,
		TotalTokenCount:      1434,
		PromptTokensDetails: []geminiTokenDetail{
			{Modality: "TEXT", TokenCount: 100},
			{Modality: "IMAGE", TokenCount: 560},
			{Modality: "AUDIO", TokenCount: 574},
		},
	}
	u := p.buildUsage(meta, req)

	if u.PromptTokens != 1234 || u.CompletionTokens != 200 || u.TotalTokens != 1434 {
		t.Errorf("base tokens = %+v", u)
	}
	if u.InputBreakdown == nil {
		t.Fatal("InputBreakdown is nil, want populated")
	}
	b := u.InputBreakdown
	if b.Text != 100 || b.Image != 560 || b.Audio != 574 || b.Video != 0 {
		t.Errorf("breakdown = %+v", b)
	}
	if sum := b.Text + b.Audio + b.Image + b.Video; sum != u.PromptTokens {
		t.Errorf("breakdown sum %d != PromptTokens %d", sum, u.PromptTokens)
	}
}

func TestBuildUsageCachedTokens(t *testing.T) {
	p := New()
	req := ir.ProviderRequest{Messages: []ir.Message{{Content: "hi"}}}
	meta := geminiUsageMetadata{
		PromptTokenCount:        500,
		CandidatesTokenCount:    100,
		TotalTokenCount:         600,
		CachedContentTokenCount: 300,
		PromptTokensDetails: []geminiTokenDetail{
			{Modality: "TEXT", TokenCount: 500},
		},
	}
	u := p.buildUsage(meta, req)
	if u.CachedTokens != 300 {
		t.Errorf("CachedTokens = %d, want 300", u.CachedTokens)
	}
	// CachedTokens is orthogonal: still present in PromptTokens (not subtracted).
	if u.PromptTokens != 500 {
		t.Errorf("PromptTokens should not be reduced by cache: %d", u.PromptTokens)
	}
}

func TestBuildUsageTextOnlyFallback(t *testing.T) {
	// Text-only request (HasMedia=false) without promptTokensDetails must
	// synthesize {Text: PromptTokens} so callers have a single code path.
	p := New()
	req := ir.ProviderRequest{
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	}
	meta := geminiUsageMetadata{
		PromptTokenCount:     42,
		CandidatesTokenCount: 10,
		TotalTokenCount:      52,
	}
	u := p.buildUsage(meta, req)
	if u.InputBreakdown == nil {
		t.Fatal("text-only fallback should synthesize breakdown, got nil")
	}
	if u.InputBreakdown.Text != 42 || u.InputBreakdown.Audio != 0 {
		t.Errorf("fallback breakdown = %+v", u.InputBreakdown)
	}
}

func TestBuildUsageMultimodalAnomaly(t *testing.T) {
	// Multimodal request + no promptTokensDetails = API drift signal:
	// leave InputBreakdown nil, log a warning, do not crash.
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, nil))
	p := New(WithLogger(logger))

	req := ir.ProviderRequest{
		HasMedia: true,
		Messages: []ir.Message{
			{Parts: []ir.Part{
				{Type: ir.PartImage, MIMEType: "image/jpeg", Data: []byte{1}},
			}},
		},
	}
	meta := geminiUsageMetadata{
		PromptTokenCount:     1234,
		CandidatesTokenCount: 200,
		TotalTokenCount:      1434,
	}
	u := p.buildUsage(meta, req)

	if u.InputBreakdown != nil {
		t.Errorf("multimodal anomaly should leave InputBreakdown nil, got %+v", u.InputBreakdown)
	}
	if u.PromptTokens != 1234 {
		t.Errorf("base tokens should still be populated")
	}
	if !strings.Contains(buf.String(), "missing promptTokensDetails") {
		t.Errorf("expected warning in log, got: %s", buf.String())
	}
}

func TestBuildBreakdownUnknownModality(t *testing.T) {
	var buf bytes.Buffer
	p := New(WithLogger(slog.New(slog.NewTextHandler(&buf, nil))))
	details := []geminiTokenDetail{
		{Modality: "TEXT", TokenCount: 100},
		{Modality: "DOCUMENT", TokenCount: 50},
	}
	b := p.buildBreakdown(details)
	if b.Text != 150 {
		t.Errorf("Text = %d, want 150 (100 + 50 folded)", b.Text)
	}
	if !strings.Contains(buf.String(), "unknown modality") {
		t.Errorf("expected warning for unknown modality, got: %s", buf.String())
	}
}

func TestChatCompletionMultimodalHappyPath(t *testing.T) {
	var gotBody geminiRequest

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatalf("decode: %v", err)
		}
		_, _ = w.Write([]byte(`{
			"candidates":[{"content":{"role":"model","parts":[{"text":"looks like a cat"}]},"finishReason":"STOP"}],
			"usageMetadata":{
				"promptTokenCount":660,
				"candidatesTokenCount":40,
				"totalTokenCount":700,
				"promptTokensDetails":[
					{"modality":"TEXT","tokenCount":100},
					{"modality":"IMAGE","tokenCount":560}
				]
			}
		}`))
	}))
	defer srv.Close()

	p := New(WithBaseURL(srv.URL))
	resp, err := p.ChatCompletion(context.Background(), ir.ProviderRequest{
		Auth:  ir.Auth{APIKey: "k"},
		Model: "gemini-2.5-flash-lite",
		Messages: []ir.Message{
			{Role: "user", Parts: []ir.Part{
				{Type: ir.PartText, Text: "what's in this?"},
				{Type: ir.PartImage, MIMEType: "image/jpeg", Data: []byte{1, 2, 3}},
			}},
		},
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}

	// Request should have carried inline_data.
	if len(gotBody.Contents[0].Parts) != 2 {
		t.Errorf("wire parts len = %d", len(gotBody.Contents[0].Parts))
	}
	if gotBody.Contents[0].Parts[1].InlineData == nil {
		t.Error("image part not wired as inline_data")
	}

	if resp.Content != "looks like a cat" {
		t.Errorf("content = %q", resp.Content)
	}
	if resp.Usage.InputBreakdown == nil {
		t.Fatal("InputBreakdown nil")
	}
	if resp.Usage.InputBreakdown.Text != 100 || resp.Usage.InputBreakdown.Image != 560 {
		t.Errorf("breakdown = %+v", resp.Usage.InputBreakdown)
	}
}

func TestChatCompletionStreamCarriesBreakdown(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":660,\"candidatesTokenCount\":40,\"totalTokenCount\":700,\"promptTokensDetails\":[{\"modality\":\"TEXT\",\"tokenCount\":100},{\"modality\":\"IMAGE\",\"tokenCount\":560}]}}\n\n"))
	}))
	defer srv.Close()

	p := New(WithBaseURL(srv.URL))
	stream, err := p.ChatCompletionStream(context.Background(), ir.ProviderRequest{
		Auth:  ir.Auth{APIKey: "k"},
		Model: "gemini-2.5-flash-lite",
		Messages: []ir.Message{
			{Parts: []ir.Part{{Type: ir.PartImage, MIMEType: "image/jpeg", Data: []byte{1}}}},
		},
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()

	var lastUsage *ir.Usage
	for {
		c, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if c.Usage != nil {
			lastUsage = c.Usage
		}
	}

	if lastUsage == nil || lastUsage.InputBreakdown == nil {
		t.Fatal("stream chunk did not carry InputBreakdown")
	}
	if lastUsage.InputBreakdown.Image != 560 {
		t.Errorf("stream breakdown Image = %d, want 560", lastUsage.InputBreakdown.Image)
	}
}
