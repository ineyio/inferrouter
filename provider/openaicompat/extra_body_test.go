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

const okAnswer = `{"id":"r","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`

// captureServer returns the raw body of the last request.
func captureServer(t *testing.T, answer string) (*Provider, *[]byte) {
	t.Helper()
	var got []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(answer))
	}))
	t.Cleanup(srv.Close)
	return New("gate", srv.URL), &got
}

func decodeBody(t *testing.T, raw []byte) map[string]any {
	t.Helper()
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("body is not JSON: %v\n%s", err, raw)
	}
	return body
}

func TestExtraBody_ReachesTheWire(t *testing.T) {
	p, got := captureServer(t, okAnswer)
	temp := 0.7

	resp, err := p.ChatCompletion(context.Background(), ir.ProviderRequest{
		Model:       "m",
		Messages:    []ir.Message{{Role: "user", Content: "hi"}},
		Temperature: &temp,
		Reasoning:   &ir.ReasoningConfig{Effort: "high"},
		ExtraBody: map[string]any{
			"temperature":          0.1,
			"reasoning_effort":     "none",
			"chat_template_kwargs": map[string]any{"enable_thinking": false},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	body := decodeBody(t, *got)
	if body["temperature"] != 0.1 {
		t.Errorf("temperature = %v, the step must win", body["temperature"])
	}
	if body["reasoning_effort"] != "none" {
		t.Errorf("reasoning_effort = %v, the step must win", body["reasoning_effort"])
	}
	nested, _ := body["chat_template_kwargs"].(map[string]any)
	if nested["enable_thinking"] != false {
		t.Errorf("nested field lost: %v", body["chat_template_kwargs"])
	}
	if body["model"] != "m" {
		t.Errorf("the envelope must survive the merge: model = %v", body["model"])
	}
	if msgs, _ := body["messages"].([]any); len(msgs) != 1 {
		t.Errorf("the envelope must survive the merge: messages = %v", body["messages"])
	}
	want := []string{"chat_template_kwargs", "reasoning_effort", "temperature"}
	if len(resp.ExtraBodyApplied) != len(want) {
		t.Fatalf("ExtraBodyApplied = %v, want %v", resp.ExtraBodyApplied, want)
	}
	for i := range want {
		if resp.ExtraBodyApplied[i] != want[i] {
			t.Errorf("ExtraBodyApplied = %v, want sorted %v", resp.ExtraBodyApplied, want)
		}
	}
}

// Through a float64 this integer would come out as ...992.
func TestExtraBody_LargeIntegerIsNotRounded(t *testing.T) {
	body, err := apiRequest{Model: "m", extra: map[string]any{"seed": int64(9007199254740993)}}.wire()
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		t.Fatal(err)
	}
	if string(fields["seed"]) != "9007199254740993" {
		t.Errorf("seed on the wire = %s", fields["seed"])
	}
}

func TestExtraBody_ReachesTheWireOnAStream(t *testing.T) {
	p, got := captureServer(t, "data: [DONE]\n\n")

	stream, err := p.ChatCompletionStream(context.Background(), ir.ProviderRequest{
		Model:     "m",
		Messages:  []ir.Message{{Role: "user", Content: "hi"}},
		ExtraBody: map[string]any{"reasoning_effort": "none"},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()

	body := decodeBody(t, *got)
	if body["reasoning_effort"] != "none" {
		t.Errorf("reasoning_effort = %v", body["reasoning_effort"])
	}
	if body["stream"] != true || body["stream_options"] == nil {
		t.Errorf("the stream envelope must survive the merge: %s", *got)
	}
}

func TestBuildRequest_NoExtraBodySameBytes(t *testing.T) {
	temp := 0.5
	for name, extra := range map[string]map[string]any{"nil": nil, "empty": {}} {
		t.Run(name, func(t *testing.T) {
			body := New("gate", "http://x").buildRequest(ir.ProviderRequest{
				Model:       "m",
				Messages:    []ir.Message{{Role: "user", Content: "hi"}},
				Temperature: &temp,
				ExtraBody:   extra,
			}, false)

			wire, err := body.wire()
			if err != nil {
				t.Fatal(err)
			}
			want, err := json.Marshal(body)
			if err != nil {
				t.Fatal(err)
			}
			if string(wire) != string(want) {
				t.Errorf("wire  %s\nwant  %s", wire, want)
			}
			const literal = `{"model":"m","messages":[{"role":"user","content":"hi"}],"temperature":0.5}`
			if string(wire) != literal {
				t.Errorf("wire  %s\nwant  %s", wire, literal)
			}
		})
	}
}

func TestExtraBody_AppliedReportsWhatWentOut(t *testing.T) {
	format := &ir.ResponseFormat{Type: "json_object"}
	effort := &ir.ReasoningConfig{Effort: "low"}

	tests := []struct {
		name                 string
		req                  ir.ProviderRequest
		wantReasoning, wantS bool
	}{
		{name: "neither"},
		{name: "asked in the request", req: ir.ProviderRequest{Reasoning: effort, ResponseFormat: format}, wantReasoning: true, wantS: true},
		{name: "only in extra_body", req: ir.ProviderRequest{ExtraBody: map[string]any{
			"reasoning_effort": "none", "response_format": map[string]any{"type": "json_object"},
		}}, wantReasoning: true, wantS: true},
		{name: "null in extra_body removes the ask", req: ir.ProviderRequest{
			Reasoning: effort, ResponseFormat: format,
			ExtraBody: map[string]any{"reasoning_effort": nil, "response_format": nil},
		}},
		{name: "other extra fields change nothing", req: ir.ProviderRequest{
			ExtraBody: map[string]any{"top_k": 3},
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, _ := captureServer(t, okAnswer)
			tt.req.Model = "m"
			tt.req.Messages = []ir.Message{{Role: "user", Content: "hi"}}

			resp, err := p.ChatCompletion(context.Background(), tt.req)
			if err != nil {
				t.Fatal(err)
			}
			if resp.ReasoningApplied != tt.wantReasoning {
				t.Errorf("ReasoningApplied = %v, want %v", resp.ReasoningApplied, tt.wantReasoning)
			}
			if resp.StructuredOutputApplied != tt.wantS {
				t.Errorf("StructuredOutputApplied = %v, want %v", resp.StructuredOutputApplied, tt.wantS)
			}
		})
	}
}

// The whole path through a real router and a real adapter.
func TestExtraBody_ThroughARealRouter(t *testing.T) {
	p, got := captureServer(t, okAnswer)
	r, err := ir.NewRouter(ir.Config{
		AllowPaid:    true,
		DefaultModel: "ladder",
		Models: []ir.ModelMapping{{Alias: "ladder", Models: []ir.ModelRef{
			{Provider: "gate", Model: "m", ExtraBody: map[string]any{"reasoning_effort": "low"}},
		}}},
		Accounts: []ir.AccountConfig{{Provider: "gate", ID: "gate-1", QuotaUnit: ir.QuotaTokens}},
	}, []ir.Provider{p})
	if err != nil {
		t.Fatal(err)
	}
	if w := r.ConfigWarnings(); len(w) != 0 {
		t.Errorf("warnings: %v", w)
	}

	resp, err := r.ChatCompletion(context.Background(), ir.ChatRequest{Messages: []ir.Message{{Role: "user", Content: "hi"}}})
	if err != nil {
		t.Fatal(err)
	}

	if body := decodeBody(t, *got); body["reasoning_effort"] != "low" {
		t.Errorf("wire: %s", *got)
	}
	if !resp.Routing.Reasoning {
		t.Error("Routing.Reasoning must report what went out")
	}
	if len(resp.Routing.ExtraBody) != 1 || resp.Routing.ExtraBody[0] != "reasoning_effort" {
		t.Errorf("Routing.ExtraBody = %v", resp.Routing.ExtraBody)
	}
}
