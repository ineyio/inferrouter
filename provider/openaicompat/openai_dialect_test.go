package openaicompat

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	ir "github.com/ineyio/inferrouter"
)

// Two places where OpenAI itself and the gateways that copy its API disagree
// (qarap, 2026-10-03, on gpt-6-luna): the spelling of the output ceiling, and
// whether thinking sits inside completion_tokens.

// ceilingKeys marshals the body the provider builds and reports which of the
// two ceiling keys reached the wire. Read off the JSON, not the struct: the
// endpoint reads the JSON.
func ceilingKeys(t *testing.T, p *Provider) map[string]any {
	t.Helper()
	limit := 512
	raw, err := json.Marshal(p.buildRequest(ir.ProviderRequest{
		Model:     "m",
		Messages:  []ir.Message{{Role: ir.RoleUser, Content: "hi"}},
		MaxTokens: &limit,
	}, false))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return body
}

// TestOutputCeilingSpellingFollowsTheEndpoint pins the dialect per endpoint,
// both ways: OpenAI answers max_tokens with a 400 on a thinking model, and a
// gateway that has never heard of max_completion_tokens would drop the
// ceiling silently. The config row is the one a constructor-based rule
// misses — an account with base_url at api.openai.com goes through New.
func TestOutputCeilingSpellingFollowsTheEndpoint(t *testing.T) {
	fromConfig := func(baseURL string) *Provider {
		ps, err := FromAccounts([]ir.AccountConfig{{Provider: "x", ID: "x-1", BaseURL: baseURL}})
		if err != nil || len(ps) != 1 {
			t.Fatalf("FromAccounts(%q) = %v, %v", baseURL, ps, err)
		}
		return ps[0].(*Provider)
	}

	cases := []struct {
		name    string
		p       *Provider
		wantNew bool
	}{
		{"NewOpenAI", NewOpenAI(), true},
		{"config base_url at api.openai.com", fromConfig("https://api.openai.com/v1"), true},
		{"opt-in option on a gateway", New("g", "https://gw.example/v1", WithMaxCompletionTokens()), true},
		{"gateway via New", New("gonkagg", "https://proxy.gonka.gg/v1"), false},
		{"gateway via config", fromConfig("https://api.gonka24.com/v1"), false},
		{"Gemini OpenAI-compatible endpoint", fromConfig("https://generativelanguage.googleapis.com/v1beta/openai"), false},
		{"NewGrok", NewGrok(), false},
		{"NewCerebro", NewCerebro(), false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			body := ceilingKeys(t, tc.p)
			want, other := "max_tokens", "max_completion_tokens"
			if tc.wantNew {
				want, other = other, want
			}
			if body[want] != float64(512) {
				t.Errorf("%s = %#v, want 512", want, body[want])
			}
			if v, present := body[other]; present {
				t.Errorf("%s present (%#v): exactly one ceiling key must travel", other, v)
			}
		})
	}
}

// TestNoCeilingSendsNeitherKey is the twin: a caller that did not ask for a
// ceiling must not get one in either spelling.
func TestNoCeilingSendsNeitherKey(t *testing.T) {
	for _, p := range []*Provider{NewOpenAI(), New("g", "https://gw.example/v1")} {
		raw, _ := json.Marshal(p.buildRequest(ir.ProviderRequest{Model: "m"}, false))
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		for _, k := range []string{"max_tokens", "max_completion_tokens"} {
			if v, present := body[k]; present {
				t.Errorf("%s: %s = %#v without a ceiling asked", p.name, k, v)
			}
		}
	}
}

// openAIUsage is the usage object exactly as qarap measured it on gpt-6-luna
// with reasoning_effort=medium: 22 completion tokens, 10 of them thinking.
const openAIUsage = `{"prompt_tokens":17,"completion_tokens":22,"total_tokens":39,` +
	`"completion_tokens_details":{"reasoning_tokens":10,"audio_tokens":0}}`

// checkMovedNotCopied is the accounting identity types.go promises:
// ReasoningTokens apart from CompletionTokens, and the three parts summing to
// the endpoint's own total. Copying reasoning_tokens without subtracting it
// passes a "ReasoningTokens == 10" check and fails this one — and that is the
// mutation that bills thinking twice in calculateSpend.
func checkMovedNotCopied(t *testing.T, u ir.Usage) {
	t.Helper()
	if u.ReasoningTokens != 10 {
		t.Errorf("ReasoningTokens = %d, want 10", u.ReasoningTokens)
	}
	if u.CompletionTokens != 12 {
		t.Errorf("CompletionTokens = %d, want 12 (22 minus the thinking)", u.CompletionTokens)
	}
	if got := u.PromptTokens + u.CompletionTokens + u.ReasoningTokens; got != u.TotalTokens {
		t.Errorf("prompt+completion+reasoning = %d, total = %d: thinking counted twice or lost", got, u.TotalTokens)
	}
}

func TestUnaryUsageMovesThinkingOutOfCompletion(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"id":"r1","model":"gpt-6-luna","choices":[{"index":0,`+
			`"message":{"role":"assistant","content":"Париж"},"finish_reason":"stop"}],"usage":`+openAIUsage+`}`)
	}))
	defer srv.Close()

	resp, err := New("openai", srv.URL).ChatCompletion(context.Background(), ir.ProviderRequest{
		Auth: ir.Auth{APIKey: "k"}, Model: "gpt-6-luna",
		Messages: []ir.Message{{Role: ir.RoleUser, Content: "Столица Франции?"}},
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}
	checkMovedNotCopied(t, resp.Usage)
}

func TestStreamUsageMovesThinkingOutOfCompletion(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Париж\"},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = io.WriteString(w, "data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[],\"usage\":"+openAIUsage+"}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	stream, err := New("openai", srv.URL).ChatCompletionStream(context.Background(), ir.ProviderRequest{
		Auth: ir.Auth{APIKey: "k"}, Model: "m",
		Messages: []ir.Message{{Role: ir.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()

	var usage *ir.Usage
	for {
		c, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if c.Usage != nil {
			usage = c.Usage
		}
	}
	if usage == nil {
		t.Fatal("no usage chunk")
	}
	checkMovedNotCopied(t, *usage)
}

// TestUsageWithoutDetailsIsUnchanged: a gateway that does not report the
// breakdown keeps today's numbers — absence is zero thinking, not a reason to
// touch completion_tokens.
func TestUsageWithoutDetailsIsUnchanged(t *testing.T) {
	var u apiUsage
	if err := json.Unmarshal([]byte(`{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}`), &u); err != nil {
		t.Fatal(err)
	}
	got := u.toUsage()
	if got != (ir.Usage{PromptTokens: 5, CompletionTokens: 7, TotalTokens: 12}) {
		t.Errorf("usage = %+v", got)
	}
}

// TestReasoningLargerThanCompletionIsNotABreakdown: such an endpoint already
// reports thinking apart; subtracting would drive CompletionTokens negative
// and under-bill the answer.
func TestReasoningLargerThanCompletionIsNotABreakdown(t *testing.T) {
	var u apiUsage
	if err := json.Unmarshal([]byte(`{"prompt_tokens":5,"completion_tokens":7,"total_tokens":42,`+
		`"completion_tokens_details":{"reasoning_tokens":30}}`), &u); err != nil {
		t.Fatal(err)
	}
	got := u.toUsage()
	if got.CompletionTokens != 7 || got.ReasoningTokens != 30 {
		t.Errorf("usage = %+v, want completion 7 and reasoning 30 as reported", got)
	}
}
