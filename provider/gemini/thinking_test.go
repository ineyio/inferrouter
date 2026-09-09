package gemini

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ineyio/inferrouter"
)

// captureBody serves a fixed response and keeps the request it was given.
func captureBody(t *testing.T, respJSON string) (*Provider, *geminiRequest) {
	t.Helper()
	var got geminiRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Errorf("request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, respJSON)
	}))
	t.Cleanup(srv.Close)
	return New(WithBaseURL(srv.URL), WithHTTPClient(srv.Client())), &got
}

const answerOnly = `{"candidates":[{"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP"}],
"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":2,"totalTokenCount":12}}`

// Ничего не просили — на проводе не должно появиться ни ключа. Пустой
// thinkingConfig это не «по умолчанию», это другой запрос.
func TestThinking_AbsentWhenNotAsked(t *testing.T) {
	p, got := captureBody(t, answerOnly)
	resp, err := p.ChatCompletion(context.Background(), inferrouter.ProviderRequest{
		Model: "gemini-3.5-flash-lite", Messages: []inferrouter.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.GenerationConfig != nil && got.GenerationConfig.ThinkingConfig != nil {
		t.Error("thinkingConfig ушёл в запрос, которого о нём не просили")
	}
	if resp.ReasoningApplied {
		t.Error("ReasoningApplied истинен без запроса — отчёт врёт в сторону, которая дороже")
	}
}

func TestThinking_LevelAndSummaryReachTheWire(t *testing.T) {
	p, got := captureBody(t, answerOnly)
	resp, err := p.ChatCompletion(context.Background(), inferrouter.ProviderRequest{
		Model:     "gemini-3.5-flash-lite",
		Messages:  []inferrouter.Message{{Role: "user", Content: "hi"}},
		Reasoning: &inferrouter.ReasoningConfig{Effort: "low", IncludeSummary: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	tc := got.GenerationConfig.ThinkingConfig
	if tc == nil || tc.ThinkingLevel != "low" || !tc.IncludeThoughts {
		t.Fatalf("thinkingConfig на проводе: %+v", tc)
	}
	if !resp.ReasoningApplied {
		t.Error("отправили и не признались")
	}
}

// Ответ приходит НЕСКОЛЬКИМИ частями, а при includeThoughts мысль идёт
// ПЕРВОЙ. Прежний читатель брал Parts[0] — то есть отдавал вызывающему
// размышление вместо ответа и терял хвост настоящего.
func TestThinking_ThoughtIsNotTheAnswer(t *testing.T) {
	const withThought = `{"candidates":[{"content":{"parts":[
		{"text":"сначала подумаю","thought":true},
		{"text":"Привет, "},
		{"text":"как дела?"}]},"finishReason":"STOP"}],
	"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"thoughtsTokenCount":77,"totalTokenCount":92}}`

	p, _ := captureBody(t, withThought)
	resp, err := p.ChatCompletion(context.Background(), inferrouter.ProviderRequest{
		Model:     "gemini-3.5-flash-lite",
		Messages:  []inferrouter.Message{{Role: "user", Content: "hi"}},
		Reasoning: &inferrouter.ReasoningConfig{Effort: "low", IncludeSummary: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "Привет, как дела?" {
		t.Errorf("ответ = %q — мысль просочилась в него или хвост потерян", resp.Content)
	}
	if resp.ReasoningSummary != "сначала подумаю" {
		t.Errorf("сводка мышления = %q", resp.ReasoningSummary)
	}
	if resp.Usage.ReasoningTokens != 77 {
		t.Errorf("thoughtsTokenCount не доехал: %d", resp.Usage.ReasoningTokens)
	}
	if strings.Contains(resp.Content, "подумаю") {
		t.Error("мысль внутри ответа")
	}
}

// Тот же дефект на пути без мышления: многочастный ответ терял всё, кроме
// первой части, и это было так с самого начала.
func TestThinking_MultipartAnswerIsJoined(t *testing.T) {
	const multi = `{"candidates":[{"content":{"parts":[{"text":"раз "},{"text":"два"}]},"finishReason":"STOP"}],
	"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2,"totalTokenCount":5}}`
	p, _ := captureBody(t, multi)
	resp, err := p.ChatCompletion(context.Background(), inferrouter.ProviderRequest{
		Model: "gemini-3.5-flash-lite", Messages: []inferrouter.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "раз два" {
		t.Errorf("ответ = %q", resp.Content)
	}
}
