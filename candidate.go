package inferrouter

import "context"

// resolveModel resolves a model name through aliases and defaults.
// Returns a list of (provider, model) pairs to try.
func resolveModel(cfg Config, requestModel string) []ModelRef {
	model := requestModel
	if model == "" {
		model = cfg.DefaultModel
	}

	// Check aliases.
	for _, m := range cfg.Models {
		if m.Alias == model {
			return m.Models
		}
	}

	// Direct model name — will be tried against all providers.
	return nil
}

// buildCandidates creates the list of possible candidates for a request.
func buildCandidates(
	ctx context.Context,
	cfg Config,
	providers map[string]Provider,
	quotaStore QuotaStore,
	health *HealthTracker,
	spend *SpendTracker,
	requestModel string,
) ([]Candidate, error) {
	refs := resolveModel(cfg, requestModel)
	var candidates []Candidate

	for _, acc := range cfg.Accounts {
		prov, ok := providers[acc.Provider]
		if !ok {
			continue
		}

		models := modelsForAccount(refs, acc, prov, requestModel, cfg)

		for _, model := range models {
			remaining, _ := quotaStore.Remaining(ctx, acc.ID)
			free := acc.DailyFree > 0 && remaining > 0

			c := Candidate{
				Provider:           prov,
				AccountID:          acc.ID,
				Auth:               acc.Auth,
				Model:              model,
				Free:               free,
				Remaining:          remaining,
				QuotaUnit:          acc.QuotaUnit,
				Health:             health.GetHealth(acc.ID),
				CostPerToken:       acc.CostPerToken,
				CostPerInputToken:  acc.CostPerInputToken,
				CostPerOutputToken: acc.CostPerOutputToken,
				MaxDailySpend:      acc.MaxDailySpend,
				CurrentSpend:       spend.GetSpend(acc.ID),
			}
			candidates = append(candidates, c)
		}
	}

	return candidates, nil
}

// modelsForAccount returns the models to try for a given account.
func modelsForAccount(refs []ModelRef, acc AccountConfig, prov Provider, requestModel string, cfg Config) []string {
	// If we have explicit refs from alias resolution, filter for this provider.
	if len(refs) > 0 {
		var models []string
		for _, ref := range refs {
			if ref.Provider == acc.Provider {
				models = append(models, ref.Model)
			}
		}
		return models
	}

	// Direct model name — check if provider supports it.
	model := requestModel
	if model == "" {
		model = cfg.DefaultModel
	}
	if model != "" && prov.SupportsModel(model) {
		return []string{model}
	}

	return nil
}

// filterCandidates removes unhealthy candidates and enforces paid/spend limits.
func filterCandidates(candidates []Candidate, allowPaid bool) []Candidate {
	var filtered []Candidate
	for _, c := range candidates {
		if c.Health == HealthUnhealthy {
			continue
		}
		if !c.Free && !allowPaid {
			continue
		}
		// Skip paid candidates that exceeded their daily spend limit.
		if !c.Free && c.MaxDailySpend > 0 && c.CurrentSpend >= c.MaxDailySpend {
			continue
		}
		filtered = append(filtered, c)
	}
	return filtered
}
