package inferrouter

import (
	"math"
	"testing"
)

const eps = 1e-12

func approxEqual(a, b float64) bool {
	return math.Abs(a-b) < eps
}

func TestCalculateSpendLegacyFlatRate(t *testing.T) {
	// Old-style: no InputBreakdown, single CostPerToken.
	c := Candidate{CostPerToken: 0.001}
	u := Usage{PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150}
	got := calculateSpend(c, u)
	want := 0.150
	if !approxEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestCalculateSpendLegacyInputOutput(t *testing.T) {
	// Old-style: input + output split, no breakdown.
	c := Candidate{CostPerInputToken: 0.00001, CostPerOutputToken: 0.00004}
	u := Usage{PromptTokens: 1000, CompletionTokens: 200}
	got := calculateSpend(c, u)
	want := 1000*0.00001 + 200*0.00004
	if !approxEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestCalculateSpendWithBreakdownTextOnly(t *testing.T) {
	// Breakdown present but all text — should equal old formula.
	c := Candidate{CostPerInputToken: 0.00001, CostPerOutputToken: 0.00004}
	u := Usage{
		PromptTokens:     1000,
		CompletionTokens: 200,
		InputBreakdown:   &InputTokenBreakdown{Text: 1000},
	}
	got := calculateSpend(c, u)
	want := 1000*0.00001 + 200*0.00004
	if !approxEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestCalculateSpendPerModality(t *testing.T) {
	// Gemini 2.5 Flash-Lite paid rates.
	c := Candidate{
		CostPerInputToken:      0.0000001, // $0.10 / 1M (text baseline)
		CostPerOutputToken:     0.0000004, // $0.40 / 1M
		CostPerAudioInputToken: 0.0000003, // $0.30 / 1M
		CostPerImageInputToken: 0.0000001, // $0.10 / 1M
	}
	u := Usage{
		PromptTokens:     1234,
		CompletionTokens: 200,
		InputBreakdown: &InputTokenBreakdown{
			Text:  100,
			Audio: 574,
			Image: 560,
		},
	}
	got := calculateSpend(c, u)
	want := 100*0.0000001 + 574*0.0000003 + 560*0.0000001 + 200*0.0000004
	if !approxEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestCalculateSpendModalityFallbackToText(t *testing.T) {
	// Per-modality rates on Candidate are pre-resolved in buildCandidates,
	// so here we construct the Candidate as buildCandidates would have:
	// zero audio/image/video rates filled with CostPerInputToken.
	textRate := 0.00001
	c := Candidate{
		CostPerInputToken:      textRate,
		CostPerOutputToken:     0.00004,
		CostPerAudioInputToken: textRate,
		CostPerImageInputToken: textRate,
		CostPerVideoInputToken: textRate,
	}
	u := Usage{
		PromptTokens:     1000,
		CompletionTokens: 100,
		InputBreakdown: &InputTokenBreakdown{
			Text:  400,
			Audio: 300,
			Image: 300,
		},
	}
	got := calculateSpend(c, u)
	want := 1000*textRate + 100*0.00004
	if !approxEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestResolveModalityCost(t *testing.T) {
	if resolveModalityCost(0.5, 0.1) != 0.5 {
		t.Error("specific should be used when > 0")
	}
	if resolveModalityCost(0, 0.1) != 0.1 {
		t.Error("fallback should be used when specific is 0")
	}
	if resolveModalityCost(0, 0) != 0 {
		t.Error("both zero should return zero")
	}
}

func TestCalculateSpendCachedTokensNotSubtracted(t *testing.T) {
	// CachedTokens is observability-only. Providers already apply the cache
	// discount server-side via a lower promptTokenCount, so subtracting it
	// here would double-count.
	c := Candidate{
		CostPerInputToken:  0.0000001,
		CostPerOutputToken: 0.0000004,
	}
	uWithCache := Usage{
		PromptTokens:     1000,
		CompletionTokens: 200,
		CachedTokens:     500, // orthogonal observability signal
		InputBreakdown:   &InputTokenBreakdown{Text: 1000},
	}
	uNoCache := Usage{
		PromptTokens:     1000,
		CompletionTokens: 200,
		InputBreakdown:   &InputTokenBreakdown{Text: 1000},
	}
	if calculateSpend(c, uWithCache) != calculateSpend(c, uNoCache) {
		t.Errorf("CachedTokens must not affect spend: cached=%v, nocache=%v",
			calculateSpend(c, uWithCache), calculateSpend(c, uNoCache))
	}
}

func TestCalculateSpendZeroConfig(t *testing.T) {
	// No cost configured anywhere → zero.
	c := Candidate{}
	u := Usage{PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150}
	if got := calculateSpend(c, u); got != 0 {
		t.Errorf("got %v, want 0", got)
	}
}
