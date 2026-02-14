package inferrouter

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// Config is the top-level router configuration.
type Config struct {
	AllowPaid    bool            `yaml:"allow_paid"`
	DefaultModel string          `yaml:"default_model"`
	Models       []ModelMapping  `yaml:"models"`
	Accounts     []AccountConfig `yaml:"accounts"`
}

// ModelMapping defines a model alias.
type ModelMapping struct {
	Alias    string     `yaml:"alias"`
	Models   []ModelRef `yaml:"models"`
}

// ModelRef references a specific provider model.
type ModelRef struct {
	Provider string `yaml:"provider"`
	Model    string `yaml:"model"`
}

// AccountConfig configures a single provider account.
type AccountConfig struct {
	Provider     string    `yaml:"provider"`
	ID           string    `yaml:"id"`
	Auth         Auth      `yaml:"auth"`
	DailyFree    int64     `yaml:"daily_free"`
	QuotaUnit    QuotaUnit `yaml:"quota_unit"`
	PaidEnabled  bool      `yaml:"paid_enabled"`
	MaxDailySpend float64  `yaml:"max_daily_spend"`

	// Deprecated: use CostPerInputToken and CostPerOutputToken instead.
	CostPerToken float64 `yaml:"cost_per_token"`

	CostPerInputToken  float64 `yaml:"cost_per_input_token"`
	CostPerOutputToken float64 `yaml:"cost_per_output_token"`
}

// LoadConfig reads and parses a YAML config file.
// Environment variables in the format ${VAR} are expanded before parsing.
func LoadConfig(path string) (Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("inferrouter: read config: %w", err)
	}

	expanded := os.ExpandEnv(string(data))

	var cfg Config
	if err := yaml.Unmarshal([]byte(expanded), &cfg); err != nil {
		return Config{}, fmt.Errorf("inferrouter: parse config: %w", err)
	}

	if err := cfg.Validate(); err != nil {
		return Config{}, err
	}

	return cfg, nil
}

// Validate checks the config for required fields and consistency.
func (c Config) Validate() error {
	if len(c.Accounts) == 0 {
		return fmt.Errorf("inferrouter: config: at least one account is required")
	}

	ids := make(map[string]bool, len(c.Accounts))
	for i, acc := range c.Accounts {
		if acc.Provider == "" {
			return fmt.Errorf("inferrouter: config: account[%d]: provider is required", i)
		}
		if acc.ID == "" {
			return fmt.Errorf("inferrouter: config: account[%d]: id is required", i)
		}
		if ids[acc.ID] {
			return fmt.Errorf("inferrouter: config: duplicate account id %q", acc.ID)
		}
		ids[acc.ID] = true

		if acc.QuotaUnit == "" {
			return fmt.Errorf("inferrouter: config: account[%d] (%s): quota_unit is required", i, acc.ID)
		}
		if acc.QuotaUnit != QuotaTokens && acc.QuotaUnit != QuotaRequests && acc.QuotaUnit != QuotaDollars {
			return fmt.Errorf("inferrouter: config: account[%d] (%s): invalid quota_unit %q", i, acc.ID, acc.QuotaUnit)
		}

		if acc.DailyFree < 0 {
			return fmt.Errorf("inferrouter: config: account[%d] (%s): daily_free must be >= 0", i, acc.ID)
		}
		if acc.MaxDailySpend < 0 {
			return fmt.Errorf("inferrouter: config: account[%d] (%s): max_daily_spend must be >= 0", i, acc.ID)
		}
		if acc.CostPerToken < 0 {
			return fmt.Errorf("inferrouter: config: account[%d] (%s): cost_per_token must be >= 0", i, acc.ID)
		}
		if acc.CostPerInputToken < 0 {
			return fmt.Errorf("inferrouter: config: account[%d] (%s): cost_per_input_token must be >= 0", i, acc.ID)
		}
		if acc.CostPerOutputToken < 0 {
			return fmt.Errorf("inferrouter: config: account[%d] (%s): cost_per_output_token must be >= 0", i, acc.ID)
		}
		if acc.PaidEnabled {
			hasCost := acc.CostPerToken > 0 || acc.CostPerInputToken > 0 || acc.CostPerOutputToken > 0
			if !hasCost {
				return fmt.Errorf("inferrouter: config: account[%d] (%s): paid_enabled requires cost configuration", i, acc.ID)
			}
		}
	}

	for i, m := range c.Models {
		if m.Alias == "" {
			return fmt.Errorf("inferrouter: config: models[%d]: alias is required", i)
		}
		if len(m.Models) == 0 {
			return fmt.Errorf("inferrouter: config: models[%d] (%s): at least one model ref is required", i, m.Alias)
		}
	}

	return nil
}

// NormalizeCosts applies backward compatibility for cost fields.
// If CostPerToken is set and the new fields are zero, it is used for both.
func (c *Config) NormalizeCosts() {
	for i := range c.Accounts {
		acc := &c.Accounts[i]
		if acc.CostPerToken > 0 && acc.CostPerInputToken == 0 && acc.CostPerOutputToken == 0 {
			acc.CostPerInputToken = acc.CostPerToken
			acc.CostPerOutputToken = acc.CostPerToken
		}
	}
}
