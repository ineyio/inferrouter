package inferrouter

import (
	"context"
	"testing"
)

// testProvider is a minimal internal Provider double. Lives in-package to
// avoid the mock package import cycle issue when testing lowercase helpers.
type testProvider struct {
	name       string
	multimodal bool
}

func (p *testProvider) Name() string              { return p.name }
func (p *testProvider) SupportsModel(string) bool { return true }
func (p *testProvider) SupportsMultimodal() bool  { return p.multimodal }
func (p *testProvider) ChatCompletion(context.Context, ProviderRequest) (ProviderResponse, error) {
	return ProviderResponse{}, nil
}
func (p *testProvider) ChatCompletionStream(context.Context, ProviderRequest) (ProviderStream, error) {
	return nil, nil
}

// --- filterCandidates ---

func TestFilterCandidatesDropsUnhealthy(t *testing.T) {
	text := &testProvider{name: "text"}
	in := []Candidate{
		{Provider: text, AccountID: "a", Free: true, Health: HealthHealthy},
		{Provider: text, AccountID: "b", Free: true, Health: HealthUnhealthy},
		{Provider: text, AccountID: "c", Free: true, Health: HealthHalfOpen},
	}
	out := filterCandidates(in, false, false)
	if len(out) != 2 {
		t.Fatalf("len = %d, want 2 (healthy + half-open)", len(out))
	}
	for _, c := range out {
		if c.AccountID == "b" {
			t.Error("unhealthy candidate survived filter")
		}
	}
}

func TestFilterCandidatesDropsPaidWhenDisallowed(t *testing.T) {
	text := &testProvider{name: "text"}
	in := []Candidate{
		{Provider: text, AccountID: "free", Free: true},
		{Provider: text, AccountID: "paid", Free: false},
	}
	out := filterCandidates(in, false, false)
	if len(out) != 1 || out[0].AccountID != "free" {
		t.Errorf("got %+v, want only free", ids(out))
	}
}

func TestFilterCandidatesAllowsPaidWhenEnabled(t *testing.T) {
	text := &testProvider{name: "text"}
	in := []Candidate{
		{Provider: text, AccountID: "free", Free: true},
		{Provider: text, AccountID: "paid", Free: false},
	}
	out := filterCandidates(in, true, false)
	if len(out) != 2 {
		t.Errorf("expected both, got %v", ids(out))
	}
}

func TestFilterCandidatesDropsPaidOverSpendCap(t *testing.T) {
	text := &testProvider{name: "text"}
	in := []Candidate{
		{Provider: text, AccountID: "paid-under", Free: false, MaxDailySpend: 10.0, CurrentSpend: 5.0},
		{Provider: text, AccountID: "paid-over", Free: false, MaxDailySpend: 10.0, CurrentSpend: 10.0},
		{Provider: text, AccountID: "paid-unlimited", Free: false, MaxDailySpend: 0, CurrentSpend: 1e9},
	}
	out := filterCandidates(in, true, false)
	got := ids(out)
	if len(got) != 2 || got[0] != "paid-under" || got[1] != "paid-unlimited" {
		t.Errorf("got %v, want [paid-under paid-unlimited]", got)
	}
}

func TestFilterCandidatesNeedMultimodal(t *testing.T) {
	text := &testProvider{name: "text", multimodal: false}
	vision := &testProvider{name: "vision", multimodal: true}
	in := []Candidate{
		{Provider: text, AccountID: "text-a", Free: true},
		{Provider: vision, AccountID: "vision-a", Free: true},
		{Provider: text, AccountID: "text-b", Free: true},
	}
	out := filterCandidates(in, false, true)
	if len(out) != 1 || out[0].AccountID != "vision-a" {
		t.Errorf("got %v, want only vision-a", ids(out))
	}
}

func TestFilterCandidatesNeedMultimodalEmpty(t *testing.T) {
	// No multimodal-capable providers → empty result, caller distinguishes error.
	text := &testProvider{name: "text", multimodal: false}
	in := []Candidate{
		{Provider: text, AccountID: "a", Free: true},
		{Provider: text, AccountID: "b", Free: true},
	}
	out := filterCandidates(in, false, true)
	if len(out) != 0 {
		t.Errorf("got %v, want empty", ids(out))
	}
}

func TestFilterCandidatesTextOnlyRequestPassesTextProviders(t *testing.T) {
	// needMultimodal=false must NOT drop text-only providers.
	text := &testProvider{name: "text", multimodal: false}
	in := []Candidate{{Provider: text, AccountID: "a", Free: true}}
	out := filterCandidates(in, false, false)
	if len(out) != 1 {
		t.Errorf("text provider dropped for text-only request")
	}
}

func ids(cs []Candidate) []string {
	out := make([]string, len(cs))
	for i, c := range cs {
		out[i] = c.AccountID
	}
	return out
}

// --- buildCandidates: modality cost resolution ---

func TestBuildCandidatesResolvesModalityCostFallback(t *testing.T) {
	// When audio/image/video rates are zero, buildCandidates must fill them
	// with CostPerInputToken. This is the architectural guarantee that
	// calculateSpend can safely multiply without null checks.
	provs := map[string]Provider{"gemini": &testProvider{name: "gemini", multimodal: true}}
	cfg := Config{
		DefaultModel: "gemini-2.5-flash-lite",
		Accounts: []AccountConfig{
			{
				Provider:           "gemini",
				ID:                 "gemini-paid",
				QuotaUnit:          QuotaTokens,
				PaidEnabled:        true,
				CostPerInputToken:  0.001, // only text rate set
				CostPerOutputToken: 0.004,
				// Audio/image/video explicitly omitted → must fall back to 0.001.
			},
		},
	}
	cands, err := buildCandidates(
		context.Background(),
		cfg,
		provs,
		&noopQuotaStore{},
		NewHealthTracker(),
		NewSpendTracker(),
		"gemini-2.5-flash-lite",
	)
	if err != nil {
		t.Fatalf("buildCandidates: %v", err)
	}
	if len(cands) != 1 {
		t.Fatalf("got %d candidates, want 1", len(cands))
	}
	c := cands[0]
	if c.CostPerAudioInputToken != 0.001 {
		t.Errorf("audio fallback = %v, want 0.001", c.CostPerAudioInputToken)
	}
	if c.CostPerImageInputToken != 0.001 {
		t.Errorf("image fallback = %v, want 0.001", c.CostPerImageInputToken)
	}
	if c.CostPerVideoInputToken != 0.001 {
		t.Errorf("video fallback = %v, want 0.001", c.CostPerVideoInputToken)
	}
}

func TestBuildCandidatesExplicitModalityRatesWin(t *testing.T) {
	// When a specific modality rate is set, it overrides the text fallback.
	provs := map[string]Provider{"gemini": &testProvider{name: "gemini", multimodal: true}}
	cfg := Config{
		DefaultModel: "gemini-2.5-flash-lite",
		Accounts: []AccountConfig{
			{
				Provider:               "gemini",
				ID:                     "gemini-paid",
				QuotaUnit:              QuotaTokens,
				PaidEnabled:            true,
				CostPerInputToken:      0.0000001, // $0.10 / 1M text
				CostPerOutputToken:     0.0000004,
				CostPerAudioInputToken: 0.0000003, // $0.30 / 1M audio — overrides
			},
		},
	}
	cands, err := buildCandidates(
		context.Background(),
		cfg,
		provs,
		&noopQuotaStore{},
		NewHealthTracker(),
		NewSpendTracker(),
		"gemini-2.5-flash-lite",
	)
	if err != nil {
		t.Fatalf("buildCandidates: %v", err)
	}
	c := cands[0]
	if c.CostPerAudioInputToken != 0.0000003 {
		t.Errorf("audio rate = %v, want explicit 0.0000003", c.CostPerAudioInputToken)
	}
	// Image and video not set → fall back to text input rate.
	if c.CostPerImageInputToken != 0.0000001 {
		t.Errorf("image fallback = %v, want 0.0000001", c.CostPerImageInputToken)
	}
}
