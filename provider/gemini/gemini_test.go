package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ir "github.com/ineyio/inferrouter"
)

func TestName(t *testing.T) {
	if New().Name() != "gemini" {
		t.Fatal("Name should be gemini")
	}
}

func TestSupportsModel(t *testing.T) {
	p := New()
	if !p.SupportsModel("gemini-2.0-flash") {
		t.Fatal("unfiltered should accept anything")
	}
	p = New(WithModels("gemini-2.0-flash"))
	if !p.SupportsModel("gemini-2.0-flash") {
		t.Fatal("should accept configured model")
	}
	if p.SupportsModel("gemini-1.5-pro") {
		t.Fatal("should reject non-configured model")
	}
}

func TestBuildRequestRoleRemap(t *testing.T) {
	p := New()
	temp := 0.2
	max := 128
	req := p.buildRequest(ir.ProviderRequest{
		Model: "gemini-2.0-flash",
		Messages: []ir.Message{
			{Role: "user", Content: "hi"},
			{Role: "assistant", Content: "yo"},
			{Role: "user", Content: "more"},
		},
		Temperature: &temp,
		MaxTokens:   &max,
		Stop:        []string{"END"},
	})
	if len(req.Contents) != 3 {
		t.Fatalf("contents len = %d", len(req.Contents))
	}
	if req.Contents[1].Role != "model" {
		t.Errorf("assistant should be remapped to model, got %q", req.Contents[1].Role)
	}
	if req.Contents[0].Role != "user" || req.Contents[2].Role != "user" {
		t.Errorf("user roles = %q, %q", req.Contents[0].Role, req.Contents[2].Role)
	}
	if req.Contents[1].Parts[0].Text != "yo" {
		t.Errorf("parts text = %q", req.Contents[1].Parts[0].Text)
	}
	if req.GenerationConfig == nil || *req.GenerationConfig.Temperature != 0.2 {
		t.Errorf("generationConfig = %+v", req.GenerationConfig)
	}
	if req.GenerationConfig.MaxOutputTokens == nil || *req.GenerationConfig.MaxOutputTokens != 128 {
		t.Errorf("maxOutputTokens = %+v", req.GenerationConfig.MaxOutputTokens)
	}
	if len(req.GenerationConfig.StopSequences) != 1 || req.GenerationConfig.StopSequences[0] != "END" {
		t.Errorf("stopSequences = %v", req.GenerationConfig.StopSequences)
	}
}

func TestBuildRequestNoGenerationConfig(t *testing.T) {
	p := New()
	req := p.buildRequest(ir.ProviderRequest{
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})
	if req.GenerationConfig != nil {
		t.Errorf("generationConfig should be nil when no params set, got %+v", req.GenerationConfig)
	}
}

func TestChatCompletionHappyPath(t *testing.T) {
	var gotURL string
	var gotBody geminiRequest

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotURL = r.URL.String()
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatalf("decode: %v", err)
		}
		_, _ = w.Write([]byte(`{
			"candidates":[{"content":{"role":"model","parts":[{"text":"hello"}]},"finishReason":"STOP"}],
			"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15},
			"modelVersion":"gemini-2.0-flash"
		}`))
	}))
	defer srv.Close()

	p := New(WithBaseURL(srv.URL))
	resp, err := p.ChatCompletion(context.Background(), ir.ProviderRequest{
		Auth:     ir.Auth{APIKey: "test-key"},
		Model:    "gemini-2.0-flash",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}

	if !strings.Contains(gotURL, "/models/gemini-2.0-flash:generateContent") {
		t.Errorf("url = %q", gotURL)
	}
	if !strings.Contains(gotURL, "key=test-key") {
		t.Errorf("url should carry api key, got %q", gotURL)
	}
	if len(gotBody.Contents) != 1 || gotBody.Contents[0].Parts[0].Text != "hi" {
		t.Errorf("body = %+v", gotBody)
	}

	if resp.Content != "hello" {
		t.Errorf("content = %q", resp.Content)
	}
	if resp.FinishReason != "stop" {
		t.Errorf("finishReason = %q (should be lowercased)", resp.FinishReason)
	}
	if resp.Model != "gemini-2.0-flash" {
		t.Errorf("model = %q", resp.Model)
	}
	if resp.Usage.PromptTokens != 10 || resp.Usage.CompletionTokens != 5 || resp.Usage.TotalTokens != 15 {
		t.Errorf("usage = %+v", resp.Usage)
	}
}

func TestChatCompletionEmptyCandidates(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"candidates":[],"usageMetadata":{}}`))
	}))
	defer srv.Close()

	p := New(WithBaseURL(srv.URL))
	_, err := p.ChatCompletion(context.Background(), ir.ProviderRequest{
		Auth:     ir.Auth{APIKey: "k"},
		Model:    "m",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})
	if err == nil || !strings.Contains(err.Error(), "empty candidates") {
		t.Errorf("expected empty candidates error, got %v", err)
	}
}

func TestChatCompletionErrorMapping(t *testing.T) {
	cases := []struct {
		name   string
		status int
		want   error
	}{
		{"429", http.StatusTooManyRequests, ir.ErrRateLimited},
		{"401", http.StatusUnauthorized, ir.ErrAuthFailed},
		{"403", http.StatusForbidden, ir.ErrAuthFailed},
		{"400", http.StatusBadRequest, ir.ErrInvalidRequest},
		{"500", http.StatusInternalServerError, ir.ErrProviderUnavailable},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				http.Error(w, `{"error":"nope"}`, tc.status)
			}))
			defer srv.Close()

			p := New(WithBaseURL(srv.URL))
			_, err := p.ChatCompletion(context.Background(), ir.ProviderRequest{
				Auth:     ir.Auth{APIKey: "k"},
				Model:    "m",
				Messages: []ir.Message{{Role: "user", Content: "hi"}},
			})
			if !errors.Is(err, tc.want) {
				t.Errorf("err = %v, want Is=%v", err, tc.want)
			}
		})
	}
}

func TestChatCompletionNetworkError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	srv.Close()

	p := New(WithBaseURL(srv.URL))
	_, err := p.ChatCompletion(context.Background(), ir.ProviderRequest{
		Auth:     ir.Auth{APIKey: "k"},
		Model:    "m",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})
	if !errors.Is(err, ir.ErrProviderUnavailable) {
		t.Errorf("err = %v, want ErrProviderUnavailable", err)
	}
}

func TestChatCompletionStreamHappyPath(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.String(), "streamGenerateContent") {
			t.Errorf("url = %q", r.URL.String())
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"hel\"}]},\"finishReason\":\"\"}],\"usageMetadata\":{}}\n\n"))
		_, _ = w.Write([]byte("data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"lo\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":2,\"candidatesTokenCount\":3,\"totalTokenCount\":5}}\n\n"))
	}))
	defer srv.Close()

	p := New(WithBaseURL(srv.URL))
	stream, err := p.ChatCompletionStream(context.Background(), ir.ProviderRequest{
		Auth:     ir.Auth{APIKey: "k"},
		Model:    "gemini-2.0-flash",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()

	var chunks []ir.StreamChunk
	for {
		c, err := stream.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		chunks = append(chunks, c)
	}

	if len(chunks) != 2 {
		t.Fatalf("got %d chunks, want 2", len(chunks))
	}
	if chunks[0].Choices[0].Delta.Content != "hel" {
		t.Errorf("chunk0 content = %q", chunks[0].Choices[0].Delta.Content)
	}
	if chunks[0].Model != "gemini-2.0-flash" {
		t.Errorf("chunk0 model = %q (should mirror request model)", chunks[0].Model)
	}
	if chunks[1].Choices[0].FinishReason != "stop" {
		t.Errorf("chunk1 finish = %q (want lowercased)", chunks[1].Choices[0].FinishReason)
	}
	if chunks[1].Usage == nil || chunks[1].Usage.TotalTokens != 5 {
		t.Errorf("usage = %+v", chunks[1].Usage)
	}
}

func TestChatCompletionStreamToleratesMalformed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("data: {bad1\n\n"))
		_, _ = w.Write([]byte("data: {bad2\n\n"))
		_, _ = w.Write([]byte("data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]}}]}\n\n"))
	}))
	defer srv.Close()

	p := New(WithBaseURL(srv.URL))
	stream, err := p.ChatCompletionStream(context.Background(), ir.ProviderRequest{
		Auth:     ir.Auth{APIKey: "k"},
		Model:    "m",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()

	c, err := stream.Next()
	if err != nil {
		t.Fatalf("Next: %v", err)
	}
	if c.Choices[0].Delta.Content != "ok" {
		t.Errorf("content = %q", c.Choices[0].Delta.Content)
	}
}

func TestChatCompletionStreamFailsOnPersistentMalformed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("data: {bad1\n\n"))
		_, _ = w.Write([]byte("data: {bad2\n\n"))
		_, _ = w.Write([]byte("data: {bad3\n\n"))
	}))
	defer srv.Close()

	p := New(WithBaseURL(srv.URL))
	stream, err := p.ChatCompletionStream(context.Background(), ir.ProviderRequest{
		Auth:     ir.Auth{APIKey: "k"},
		Model:    "m",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()

	_, err = stream.Next()
	if err == nil || !strings.Contains(err.Error(), "malformed SSE") {
		t.Errorf("expected malformed SSE error, got %v", err)
	}
}

func TestChatCompletionStreamErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "auth", http.StatusUnauthorized)
	}))
	defer srv.Close()

	p := New(WithBaseURL(srv.URL))
	_, err := p.ChatCompletionStream(context.Background(), ir.ProviderRequest{
		Auth:     ir.Auth{APIKey: "k"},
		Model:    "m",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})
	if !errors.Is(err, ir.ErrAuthFailed) {
		t.Errorf("err = %v, want ErrAuthFailed", err)
	}
}
