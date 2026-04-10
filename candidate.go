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
			remaining, remainErr := quotaStore.Remaining(ctx, acc.ID)
			// Fail-open: if we can't check remaining quota, assume free if configured.
			// Reserve() will enforce the actual limit.
			free := acc.DailyFree > 0 && (remaining > 0 || remainErr != nil)

			c := Candidate{
				Provider:               prov,
				AccountID:              acc.ID,
				Auth:                   acc.Auth,
				Model:                  model,
				Free:                   free,
				Remaining:              remaining,
				QuotaUnit:              acc.QuotaUnit,
				Health:                 health.GetHealth(acc.ID),
				CostPerToken:           acc.CostPerToken,
				CostPerInputToken:      acc.CostPerInputToken,
				CostPerOutputToken:     acc.CostPerOutputToken,
				CostPerAudioInputToken: resolveModalityCost(acc.CostPerAudioInputToken, acc.CostPerInputToken),
				CostPerImageInputToken: resolveModalityCost(acc.CostPerImageInputToken, acc.CostPerInputToken),
				CostPerVideoInputToken: resolveModalityCost(acc.CostPerVideoInputToken, acc.CostPerInputToken),
				MaxDailySpend:          acc.MaxDailySpend,
				CurrentSpend:           spend.GetSpend(acc.ID),
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

// filterCandidates removes unhealthy candidates, enforces paid/spend limits,
// and (when needMultimodal is true) drops providers that don't advertise
// multimodal support.
func filterCandidates(candidates []Candidate, allowPaid, needMultimodal bool) []Candidate {
	filtered := make([]Candidate, 0, len(candidates))
	for _, c := range candidates {
		if c.Health == HealthUnhealthy {
			continue
		}
		if !c.Free && !allowPaid {
			continue
		}
		if !c.Free && c.MaxDailySpend > 0 && c.CurrentSpend >= c.MaxDailySpend {
			continue
		}
		if needMultimodal && !c.Provider.SupportsMultimodal() {
			continue
		}
		filtered = append(filtered, c)
	}
	return filtered
}

// resolveModalityCost returns the specific per-modality rate if configured,
// otherwise falls back to the text input rate as a baseline.
func resolveModalityCost(specific, fallback float64) float64 {
	if specific > 0 {
		return specific
	}
	return fallback
}
