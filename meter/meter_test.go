package meter

import (
	"bytes"
	"errors"
	"log/slog"
	"strings"
	"testing"
	"time"

	ir "github.com/ineyio/inferrouter"
)

func TestNoopMeterAcceptsEvents(t *testing.T) {
	var m ir.Meter = &NoopMeter{}
	m.OnRoute(ir.RouteEvent{Provider: "p", AccountID: "a", Model: "m", Free: true, AttemptNum: 1, EstimatedIn: 42})
	m.OnResult(ir.ResultEvent{Provider: "p", AccountID: "a", Success: true, Duration: time.Millisecond, Usage: ir.Usage{TotalTokens: 10}})
	m.OnResult(ir.ResultEvent{Provider: "p", AccountID: "a", Success: false, Error: errors.New("x")})
}

func newCapturingLogger(buf *bytes.Buffer) *slog.Logger {
	return slog.New(slog.NewTextHandler(buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
}

func TestLogMeterNilLoggerUsesDefault(t *testing.T) {
	m := NewLogMeter(nil)
	if m.Logger == nil {
		t.Fatal("nil logger should be replaced with slog.Default()")
	}
	// Should not panic when events fire.
	m.OnRoute(ir.RouteEvent{Provider: "p"})
	m.OnResult(ir.ResultEvent{Provider: "p", Success: true})
}

func TestLogMeterOnRouteFields(t *testing.T) {
	var buf bytes.Buffer
	m := NewLogMeter(newCapturingLogger(&buf))

	m.OnRoute(ir.RouteEvent{
		Provider:    "openai",
		AccountID:   "acc-1",
		Model:       "gpt-4o",
		Free:        true,
		AttemptNum:  2,
		EstimatedIn: 123,
	})

	out := buf.String()
	checks := []string{
		"level=INFO",
		`msg=route`,
		`provider=openai`,
		`account=acc-1`,
		`model=gpt-4o`,
		`free=true`,
		`attempt=2`,
		`estimated_tokens=123`,
	}
	for _, want := range checks {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in:\n%s", want, out)
		}
	}
}

func TestLogMeterOnResultSuccessFields(t *testing.T) {
	var buf bytes.Buffer
	m := NewLogMeter(newCapturingLogger(&buf))

	m.OnResult(ir.ResultEvent{
		Provider:   "gemini",
		AccountID:  "gem-1",
		Model:      "gemini-2.0-flash",
		Free:       true,
		Success:    true,
		Duration:   150 * time.Millisecond,
		Usage:      ir.Usage{PromptTokens: 10, CompletionTokens: 20, TotalTokens: 30},
		DollarCost: 0.0042,
	})

	out := buf.String()
	checks := []string{
		"level=INFO",
		`msg=result`,
		`provider=gemini`,
		`account=gem-1`,
		`model=gemini-2.0-flash`,
		`duration_ms=150`,
		`prompt_tokens=10`,
		`completion_tokens=20`,
		`dollar_cost=0.0042`,
	}
	for _, want := range checks {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in:\n%s", want, out)
		}
	}
}

func TestLogMeterOnResultTextOnlyZeroDiff(t *testing.T) {
	// Q9 backward-compat: a text-only result (no CachedTokens, nil InputBreakdown)
	// must NOT emit any new fields. Existing parsers see the same log shape.
	var buf bytes.Buffer
	m := NewLogMeter(newCapturingLogger(&buf))

	m.OnResult(ir.ResultEvent{
		Provider:   "cerebras",
		AccountID:  "cerebras-free",
		Model:      "qwen-3",
		Free:       true,
		Success:    true,
		Duration:   100 * time.Millisecond,
		Usage:      ir.Usage{PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150},
		DollarCost: 0,
	})

	out := buf.String()
	// New fields must be absent.
	for _, forbidden := range []string{"cached_tokens", "text_tokens", "audio_tokens", "image_tokens", "video_tokens"} {
		if strings.Contains(out, forbidden) {
			t.Errorf("text-only log must not contain %q, got: %s", forbidden, out)
		}
	}
	// Existing fields still present.
	for _, want := range []string{"prompt_tokens=100", "completion_tokens=50", "dollar_cost=0"} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in: %s", want, out)
		}
	}
}

func TestLogMeterOnResultMultimodalWithBreakdown(t *testing.T) {
	var buf bytes.Buffer
	m := NewLogMeter(newCapturingLogger(&buf))

	m.OnResult(ir.ResultEvent{
		Provider:  "gemini",
		AccountID: "gemini-free",
		Model:     "gemini-2.5-flash-lite",
		Free:      true,
		Success:   true,
		Duration:  200 * time.Millisecond,
		Usage: ir.Usage{
			PromptTokens:     1234,
			CompletionTokens: 200,
			TotalTokens:      1434,
			CachedTokens:     300,
			InputBreakdown: &ir.InputTokenBreakdown{
				Text:  100,
				Audio: 574,
				Image: 560,
			},
		},
	})

	out := buf.String()
	checks := []string{
		"cached_tokens=300",
		"text_tokens=100",
		"audio_tokens=574",
		"image_tokens=560",
		"video_tokens=0",
	}
	for _, want := range checks {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in: %s", want, out)
		}
	}
}

func TestLogMeterOnResultCachedTokensOmittedWhenZero(t *testing.T) {
	var buf bytes.Buffer
	m := NewLogMeter(newCapturingLogger(&buf))

	// InputBreakdown present but CachedTokens=0 → cached_tokens should be absent.
	m.OnResult(ir.ResultEvent{
		Provider:  "gemini",
		AccountID: "gemini-free",
		Model:     "gemini-2.5-flash-lite",
		Success:   true,
		Duration:  100 * time.Millisecond,
		Usage: ir.Usage{
			PromptTokens:     100,
			CompletionTokens: 50,
			InputBreakdown:   &ir.InputTokenBreakdown{Text: 100},
		},
	})

	out := buf.String()
	if strings.Contains(out, "cached_tokens") {
		t.Errorf("cached_tokens must be omitted when zero, got: %s", out)
	}
	if !strings.Contains(out, "text_tokens=100") {
		t.Errorf("text_tokens should be present: %s", out)
	}
}

func TestLogMeterOnResultErrorFields(t *testing.T) {
	var buf bytes.Buffer
	m := NewLogMeter(newCapturingLogger(&buf))

	m.OnResult(ir.ResultEvent{
		Provider:  "grok",
		AccountID: "grok-1",
		Model:     "grok-3",
		Success:   false,
		Duration:  50 * time.Millisecond,
		Error:     errors.New("rate limited"),
	})

	out := buf.String()
	checks := []string{
		"level=WARN",
		`msg=result_error`,
		`provider=grok`,
		`account=grok-1`,
		`model=grok-3`,
		`duration_ms=50`,
		"rate limited",
	}
	for _, want := range checks {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in:\n%s", want, out)
		}
	}
}
