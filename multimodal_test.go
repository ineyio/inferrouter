package inferrouter_test

import (
	"context"
	"errors"
	"testing"

	ir "github.com/ineyio/inferrouter"
	"github.com/ineyio/inferrouter/provider/mock"
	"github.com/ineyio/inferrouter/quota"
)

// newMultimodalTestRouter builds a minimal Router with the given providers.
func newMultimodalTestRouter(t *testing.T, providers ...ir.Provider) *ir.Router {
	t.Helper()

	accounts := make([]ir.AccountConfig, len(providers))
	for i, p := range providers {
		accounts[i] = ir.AccountConfig{
			Provider:  p.Name(),
			ID:        p.Name() + "-acc",
			DailyFree: 1000,
			QuotaUnit: ir.QuotaRequests,
		}
	}

	cfg := ir.Config{
		DefaultModel: "mock-model",
		Accounts:     accounts,
	}
	r, err := ir.NewRouter(cfg, providers, ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}
	return r
}

func TestMultimodalRequestUnavailableWhenNoCapableProvider(t *testing.T) {
	// Only a text-only provider in the pool.
	textOnly := mock.New(mock.WithName("text-only")) // default multimodal=false
	r := newMultimodalTestRouter(t, textOnly)

	_, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Parts: []ir.Part{
			{Type: ir.PartText, Text: "describe"},
			{Type: ir.PartImage, MIMEType: "image/jpeg", Data: []byte{1, 2, 3}},
		}}},
	})

	if !errors.Is(err, ir.ErrMultimodalUnavailable) {
		t.Errorf("err = %v, want ErrMultimodalUnavailable", err)
	}
	if textOnly.CallCount() != 0 {
		t.Errorf("text-only provider should not be called, got %d", textOnly.CallCount())
	}
}

func TestMultimodalRequestRoutesToCapableProvider(t *testing.T) {
	textOnly := mock.New(mock.WithName("text-only"))
	mmCapable := mock.New(mock.WithName("vision"), mock.WithMultimodal(true))

	r := newMultimodalTestRouter(t, textOnly, mmCapable)

	resp, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Parts: []ir.Part{
			{Type: ir.PartImage, MIMEType: "image/jpeg", Data: []byte{1}},
		}}},
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}

	if textOnly.CallCount() != 0 {
		t.Errorf("text-only provider should not be called, got %d", textOnly.CallCount())
	}
	if mmCapable.CallCount() != 1 {
		t.Errorf("vision provider call count = %d, want 1", mmCapable.CallCount())
	}
	if resp.Routing.Provider != "vision" {
		t.Errorf("routed to %q, want vision", resp.Routing.Provider)
	}
}

func TestTextOnlyRequestUsesAnyProvider(t *testing.T) {
	// Text-only request should NOT filter out text-only providers.
	textOnly := mock.New(mock.WithName("text-only"))
	r := newMultimodalTestRouter(t, textOnly)

	_, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}
	if textOnly.CallCount() != 1 {
		t.Errorf("call count = %d, want 1", textOnly.CallCount())
	}
}

func TestErrMultimodalUnavailableClassification(t *testing.T) {
	// Not retryable (no candidates to cycle through) and not fatal
	// (caller may still degrade gracefully by stripping media).
	if ir.IsRetryable(ir.ErrMultimodalUnavailable) {
		t.Error("ErrMultimodalUnavailable must not be retryable")
	}
	if ir.IsFatal(ir.ErrMultimodalUnavailable) {
		t.Error("ErrMultimodalUnavailable must not be fatal")
	}
}

func TestMultimodalStreamUnavailableWhenNoCapableProvider(t *testing.T) {
	// ChatCompletionStream must also surface ErrMultimodalUnavailable —
	// it shares prepareRoute with ChatCompletion but is a separate public API.
	textOnly := mock.New(mock.WithName("text-only"))
	r := newMultimodalTestRouter(t, textOnly)

	_, err := r.ChatCompletionStream(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Parts: []ir.Part{
			{Type: ir.PartImage, MIMEType: "image/jpeg", Data: []byte{1, 2, 3}},
		}}},
	})

	if !errors.Is(err, ir.ErrMultimodalUnavailable) {
		t.Errorf("stream err = %v, want ErrMultimodalUnavailable", err)
	}
	if textOnly.CallCount() != 0 {
		t.Errorf("text-only provider should not be called, got %d", textOnly.CallCount())
	}
}

func TestProviderRequestHasMediaPropagation(t *testing.T) {
	// Verify router sets ProviderRequest.HasMedia correctly for both
	// text-only and multimodal requests, so providers can rely on the flag
	// instead of re-walking Messages.
	cases := []struct {
		name     string
		req      ir.ChatRequest
		wantFlag bool
	}{
		{
			name:     "text-only via Content",
			req:      ir.ChatRequest{Messages: []ir.Message{{Role: "user", Content: "hi"}}},
			wantFlag: false,
		},
		{
			name: "text-only via Parts",
			req: ir.ChatRequest{Messages: []ir.Message{
				{Role: "user", Parts: []ir.Part{{Type: ir.PartText, Text: "hi"}}},
			}},
			wantFlag: false,
		},
		{
			name: "multimodal",
			req: ir.ChatRequest{Messages: []ir.Message{
				{Role: "user", Parts: []ir.Part{
					{Type: ir.PartText, Text: "what"},
					{Type: ir.PartImage, MIMEType: "image/jpeg", Data: []byte{1}},
				}},
			}},
			wantFlag: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var seenFlag bool
			prov := mock.New(
				mock.WithName("probe"),
				mock.WithMultimodal(true),
				mock.WithResponseFunc(func(req ir.ProviderRequest) (ir.ProviderResponse, error) {
					seenFlag = req.HasMedia
					return ir.ProviderResponse{
						ID:      "x",
						Content: "ok",
						Usage:   ir.Usage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
					}, nil
				}),
			)
			r := newMultimodalTestRouter(t, prov)

			if _, err := r.ChatCompletion(context.Background(), tc.req); err != nil {
				t.Fatalf("ChatCompletion: %v", err)
			}
			if seenFlag != tc.wantFlag {
				t.Errorf("HasMedia = %v, want %v", seenFlag, tc.wantFlag)
			}
		})
	}
}

func TestMultimodalWithBreakdownFunc(t *testing.T) {
	// End-to-end: mock provider with WithInputBreakdownFunc returns a
	// deterministic breakdown, which should propagate into ChatResponse.Usage.
	imageBytes := []byte{1, 2, 3, 4}
	vision := mock.New(
		mock.WithName("vision"),
		mock.WithMultimodal(true),
		mock.WithInputBreakdownFunc(func(req ir.ProviderRequest) ir.InputTokenBreakdown {
			var b ir.InputTokenBreakdown
			for _, m := range req.Messages {
				for _, p := range m.Parts {
					switch p.Type {
					case ir.PartText:
						b.Text += 10
					case ir.PartImage:
						b.Image += 560
					case ir.PartAudio:
						b.Audio += 300
					}
				}
			}
			return b
		}),
	)

	r := newMultimodalTestRouter(t, vision)

	resp, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Parts: []ir.Part{
			{Type: ir.PartText, Text: "what's this"},
			{Type: ir.PartImage, MIMEType: "image/jpeg", Data: imageBytes},
		}}},
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}
	if resp.Usage.InputBreakdown == nil {
		t.Fatal("InputBreakdown nil")
	}
	b := resp.Usage.InputBreakdown
	if b.Text != 10 || b.Image != 560 || b.Audio != 0 {
		t.Errorf("breakdown = %+v", b)
	}
	if resp.Usage.PromptTokens != 570 {
		t.Errorf("PromptTokens = %d, want 570", resp.Usage.PromptTokens)
	}
}
