package inferrouter

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func validAccount() AccountConfig {
	return AccountConfig{
		Provider:  "mock",
		ID:        "mock-acc",
		QuotaUnit: QuotaTokens,
		DailyFree: 1000,
	}
}

func TestLoadConfigParsesBaseURL(t *testing.T) {
	yamlCfg := `
default_model: qwen/qwen3-235b-a22b-instruct-2507-fp8
accounts:
  - provider: gonkagate
    id: gonkagate-main
    base_url: https://api.gonkagate.com/v1
    auth:
      api_key: gp-test
    quota_unit: tokens
    daily_free: 1000
  - provider: codeprov
    id: code-constructed
    quota_unit: tokens
    daily_free: 1000
`
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(yamlCfg), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if got := cfg.Accounts[0].BaseURL; got != "https://api.gonkagate.com/v1" {
		t.Errorf("BaseURL = %q, want gonkagate endpoint", got)
	}
	if got := cfg.Accounts[1].BaseURL; got != "" {
		t.Errorf("BaseURL = %q, want empty for code-constructed provider", got)
	}
}

func TestConfigValidateRequiresAccount(t *testing.T) {
	cfg := Config{}
	if err := cfg.Validate(); err == nil {
		t.Fatal("expected error for empty accounts")
	}
}

func TestConfigValidateRejectsNegativeCosts(t *testing.T) {
	cases := []struct {
		name  string
		mut   func(*AccountConfig)
		field string
	}{
		{"CostPerToken", func(a *AccountConfig) { a.CostPerToken = -0.1 }, "cost_per_token"},
		{"CostPerInputToken", func(a *AccountConfig) { a.CostPerInputToken = -0.1 }, "cost_per_input_token"},
		{"CostPerOutputToken", func(a *AccountConfig) { a.CostPerOutputToken = -0.1 }, "cost_per_output_token"},
		{"CostPerAudioInputToken", func(a *AccountConfig) { a.CostPerAudioInputToken = -0.1 }, "cost_per_audio_input_token"},
		{"CostPerImageInputToken", func(a *AccountConfig) { a.CostPerImageInputToken = -0.1 }, "cost_per_image_input_token"},
		{"CostPerVideoInputToken", func(a *AccountConfig) { a.CostPerVideoInputToken = -0.1 }, "cost_per_video_input_token"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			acc := validAccount()
			tc.mut(&acc)
			cfg := Config{Accounts: []AccountConfig{acc}}
			err := cfg.Validate()
			if err == nil {
				t.Fatalf("expected error for negative %s", tc.field)
			}
			if !strings.Contains(err.Error(), tc.field) {
				t.Errorf("error should mention %q, got: %v", tc.field, err)
			}
		})
	}
}

func TestConfigValidateAcceptsZeroCosts(t *testing.T) {
	// Zero rates must be accepted — they trigger the fallback path in
	// buildCandidates (zero audio rate → falls back to text input rate).
	acc := validAccount()
	acc.CostPerInputToken = 0.0000001
	acc.CostPerOutputToken = 0.0000004
	// Audio/image/video explicitly zero.
	cfg := Config{Accounts: []AccountConfig{acc}}
	if err := cfg.Validate(); err != nil {
		t.Errorf("zero modality rates should be accepted: %v", err)
	}
}

func TestConfigValidateRejectsNegativeDailyFree(t *testing.T) {
	acc := validAccount()
	acc.DailyFree = -1
	cfg := Config{Accounts: []AccountConfig{acc}}
	if err := cfg.Validate(); err == nil {
		t.Error("expected error for negative daily_free")
	}
}

func TestConfigValidateRejectsDuplicateIDs(t *testing.T) {
	acc1 := validAccount()
	acc2 := validAccount() // same ID
	cfg := Config{Accounts: []AccountConfig{acc1, acc2}}
	if err := cfg.Validate(); err == nil {
		t.Error("expected error for duplicate account IDs")
	}
}

func TestConfigValidateRequiresQuotaUnit(t *testing.T) {
	acc := validAccount()
	acc.QuotaUnit = ""
	cfg := Config{Accounts: []AccountConfig{acc}}
	if err := cfg.Validate(); err == nil {
		t.Error("expected error for missing quota_unit")
	}
}

func TestConfigValidatePaidEnabledWithoutCostIsValid(t *testing.T) {
	// A billable account whose price is not published is a normal state, not a
	// configuration error: two of our three Gonka gateways publish no rate. The
	// consequence (spend is not tracked) is reported by Router.ConfigWarnings.
	acc := validAccount()
	acc.PaidEnabled = true
	// No cost configured.
	cfg := Config{Accounts: []AccountConfig{acc}}
	if err := cfg.Validate(); err != nil {
		t.Errorf("paid account without a price must validate, got: %v", err)
	}
}

func TestConfigValidateValidMultimodalConfig(t *testing.T) {
	// The full qarap-shaped config from the multimodal proposal §5.4 should validate.
	cfg := Config{
		DefaultModel: "multimodal",
		Accounts: []AccountConfig{
			{
				Provider:  "gemini",
				ID:        "gemini-free",
				QuotaUnit: QuotaRequests,
				DailyFree: 1000,
			},
			{
				Provider:               "gemini",
				ID:                     "gemini-paid",
				QuotaUnit:              QuotaTokens,
				PaidEnabled:            true,
				CostPerInputToken:      0.0000001,
				CostPerOutputToken:     0.0000004,
				CostPerAudioInputToken: 0.0000003,
				CostPerImageInputToken: 0.0000001,
				CostPerVideoInputToken: 0.0000001,
				MaxDailySpend:          0.50,
			},
		},
		Models: []ModelMapping{
			{
				Alias: "multimodal",
				Models: []ModelRef{
					{Provider: "gemini", Model: "gemini-2.5-flash-lite"},
				},
			},
		},
	}
	if err := cfg.Validate(); err != nil {
		t.Errorf("valid multimodal config rejected: %v", err)
	}
}

func TestNormalizeCostsLegacyFallback(t *testing.T) {
	// When only legacy CostPerToken is set, Normalize should populate the
	// new per-direction fields. Regression guard for backward compat.
	cfg := Config{
		Accounts: []AccountConfig{
			{
				Provider:     "mock",
				ID:           "mock-acc",
				QuotaUnit:    QuotaTokens,
				CostPerToken: 0.002,
			},
		},
	}
	cfg.NormalizeCosts()
	got := cfg.Accounts[0]
	if got.CostPerInputToken != 0.002 {
		t.Errorf("CostPerInputToken = %v, want 0.002", got.CostPerInputToken)
	}
	if got.CostPerOutputToken != 0.002 {
		t.Errorf("CostPerOutputToken = %v, want 0.002", got.CostPerOutputToken)
	}
}
