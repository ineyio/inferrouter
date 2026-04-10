package openaicompat

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

func TestSupportsModel(t *testing.T) {
	p := New("test", "http://x")
	if !p.SupportsModel("anything") {
		t.Fatal("expected unfiltered provider to accept any model")
	}

	p = New("test", "http://x", WithModels("gpt-4o", "gpt-4o-mini"))
	if !p.SupportsModel("gpt-4o") {
		t.Fatal("expected gpt-4o to be supported")
	}
	if p.SupportsModel("gpt-3.5") {
		t.Fatal("expected gpt-3.5 to be rejected")
	}
}

func TestPresetConstructors(t *testing.T) {
	cases := []struct {
		name       string
		p          *Provider
		wantName   string
		wantPrefix string
	}{
		{"openai", NewOpenAI(), "openai", "https://api.openai.com"},
		{"grok", NewGrok(), "grok", "https://api.x.ai"},
		{"cerebro", NewCerebro(), "cerebro", "https://api.cerebras.ai"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.p.Name() != tc.wantName {
				t.Errorf("Name() = %q, want %q", tc.p.Name(), tc.wantName)
			}
			if !strings.HasPrefix(tc.p.baseURL, tc.wantPrefix) {
				t.Errorf("baseURL = %q, want prefix %q", tc.p.baseURL, tc.wantPrefix)
			}
		})
	}
}

func TestBaseURLTrailingSlashTrimmed(t *testing.T) {
	p := New("x", "http://example.com/v1/")
	if p.baseURL != "http://example.com/v1" {
		t.Errorf("baseURL = %q, want trimmed", p.baseURL)
	}
}

func TestChatCompletionHappyPath(t *testing.T) {
	var gotBody apiRequest
	var gotAuth, gotContentType, gotPath string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotContentType = r.Header.Get("Content-Type")
		gotPath = r.URL.Path
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatalf("decode: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "resp-1",
			"model": "gpt-4o-mini",
			"choices": [{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],
			"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
		}`))
	}))
	defer srv.Close()

	p := New("openai", srv.URL)
	temp := 0.7
	maxTok := 256
	resp, err := p.ChatCompletion(context.Background(), ir.ProviderRequest{
		Auth:        ir.Auth{APIKey: "sk-test"},
		Model:       "gpt-4o-mini",
		Messages:    []ir.Message{{Role: "user", Content: "hi"}},
		Temperature: &temp,
		MaxTokens:   &maxTok,
		Stop:        []string{"STOP"},
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}

	if gotAuth != "Bearer sk-test" {
		t.Errorf("Authorization = %q", gotAuth)
	}
	if gotContentType != "application/json" {
		t.Errorf("Content-Type = %q", gotContentType)
	}
	if gotPath != "/chat/completions" {
		t.Errorf("path = %q", gotPath)
	}
	if gotBody.Model != "gpt-4o-mini" {
		t.Errorf("body model = %q", gotBody.Model)
	}
	if gotBody.Stream {
		t.Error("non-stream request should have stream=false")
	}
	if len(gotBody.Messages) != 1 || gotBody.Messages[0].Content != "hi" {
		t.Errorf("body messages = %+v", gotBody.Messages)
	}
	if gotBody.Temperature == nil || *gotBody.Temperature != 0.7 {
		t.Errorf("body temperature = %v", gotBody.Temperature)
	}
	if gotBody.MaxTokens == nil || *gotBody.MaxTokens != 256 {
		t.Errorf("body max_tokens = %v", gotBody.MaxTokens)
	}
	if len(gotBody.Stop) != 1 || gotBody.Stop[0] != "STOP" {
		t.Errorf("body stop = %v", gotBody.Stop)
	}

	if resp.ID != "resp-1" || resp.Content != "hello" || resp.FinishReason != "stop" {
		t.Errorf("resp = %+v", resp)
	}
	if resp.Usage.PromptTokens != 10 || resp.Usage.CompletionTokens != 5 || resp.Usage.TotalTokens != 15 {
		t.Errorf("usage = %+v", resp.Usage)
	}
	if resp.Model != "gpt-4o-mini" {
		t.Errorf("resp.Model = %q", resp.Model)
	}
}

func TestChatCompletionErrorMapping(t *testing.T) {
	cases := []struct {
		name     string
		status   int
		wantSent error
	}{
		{"429", http.StatusTooManyRequests, ir.ErrRateLimited},
		{"401", http.StatusUnauthorized, ir.ErrAuthFailed},
		{"403", http.StatusForbidden, ir.ErrAuthFailed},
		{"400", http.StatusBadRequest, ir.ErrInvalidRequest},
		{"500", http.StatusInternalServerError, ir.ErrProviderUnavailable},
		{"502", http.StatusBadGateway, ir.ErrProviderUnavailable},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				http.Error(w, `{"error":"oops"}`, tc.status)
			}))
			defer srv.Close()

			p := New("x", srv.URL)
			_, err := p.ChatCompletion(context.Background(), ir.ProviderRequest{
				Auth:     ir.Auth{APIKey: "k"},
				Model:    "m",
				Messages: []ir.Message{{Role: "user", Content: "hi"}},
			})
			if !errors.Is(err, tc.wantSent) {
				t.Errorf("err = %v, want Is=%v", err, tc.wantSent)
			}
		})
	}
}

func TestChatCompletionEmptyChoices(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"x","model":"m","choices":[],"usage":{}}`))
	}))
	defer srv.Close()

	p := New("x", srv.URL)
	_, err := p.ChatCompletion(context.Background(), ir.ProviderRequest{
		Auth:     ir.Auth{APIKey: "k"},
		Model:    "m",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})
	if err == nil || !strings.Contains(err.Error(), "empty choices") {
		t.Errorf("expected empty choices error, got %v", err)
	}
}

func TestChatCompletionNetworkError(t *testing.T) {
	// Point at an unrouteable address. http.DefaultClient has no timeout;
	// use a closed server to get an immediate connection refused.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	srv.Close()

	p := New("x", srv.URL)
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
		var body apiRequest
		_ = json.NewDecoder(r.Body).Decode(&body)
		if !body.Stream {
			t.Error("stream request should have stream=true")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		// Normal chunk, then usage chunk, then [DONE].
		_, _ = w.Write([]byte("data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hel\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3,\"total_tokens\":5}}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	p := New("openai", srv.URL)
	stream, err := p.ChatCompletionStream(context.Background(), ir.ProviderRequest{
		Auth:     ir.Auth{APIKey: "k"},
		Model:    "m",
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
	if chunks[0].Choices[0].Delta.Content != "hel" || chunks[0].Choices[0].Delta.Role != "assistant" {
		t.Errorf("chunk0 = %+v", chunks[0])
	}
	if chunks[1].Choices[0].Delta.Content != "lo" || chunks[1].Choices[0].FinishReason != "stop" {
		t.Errorf("chunk1 = %+v", chunks[1])
	}
	if chunks[1].Usage == nil || chunks[1].Usage.TotalTokens != 5 {
		t.Errorf("usage = %+v", chunks[1].Usage)
	}
}

func TestChatCompletionStreamToleratesMalformed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Two malformed, one good — parseErrs counter must reset.
		_, _ = w.Write([]byte("data: {not-json\n\n"))
		_, _ = w.Write([]byte("data: also{bad\n\n"))
		_, _ = w.Write([]byte("data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	p := New("x", srv.URL)
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
	if _, err := stream.Next(); !errors.Is(err, io.EOF) {
		t.Errorf("expected EOF, got %v", err)
	}
}

func TestChatCompletionStreamFailsOnPersistentMalformed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("data: {bad1\n\n"))
		_, _ = w.Write([]byte("data: {bad2\n\n"))
		_, _ = w.Write([]byte("data: {bad3\n\n"))
	}))
	defer srv.Close()

	p := New("x", srv.URL)
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
		t.Errorf("expected malformed error, got %v", err)
	}
}

func TestChatCompletionStreamSkipsCommentsAndEmpty(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// SSE keep-alive comments and blank lines must be ignored.
		_, _ = w.Write([]byte(": keep-alive\n"))
		_, _ = w.Write([]byte("\n"))
		_, _ = w.Write([]byte("data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	p := New("x", srv.URL)
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
	if c.Choices[0].Delta.Content != "x" {
		t.Errorf("content = %q", c.Choices[0].Delta.Content)
	}
}

func TestChatCompletionStreamErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "slow down", http.StatusTooManyRequests)
	}))
	defer srv.Close()

	p := New("x", srv.URL)
	_, err := p.ChatCompletionStream(context.Background(), ir.ProviderRequest{
		Auth:     ir.Auth{APIKey: "k"},
		Model:    "m",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})
	if !errors.Is(err, ir.ErrRateLimited) {
		t.Errorf("err = %v, want ErrRateLimited", err)
	}
}
