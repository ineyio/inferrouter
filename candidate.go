package inferrouter

import (
	"context"
	"fmt"
)

// resolveModel resolves a model name to the ordered steps of its ladder.
//
// Resolution is strict: the name must be a declared alias. A name that is not
// declared is a configuration error (ErrUnknownAlias), not a request to try
// that name against every provider — the old fallback let a typo in a ladder
// name widen the candidate set to every account (RFC inferrouter-purpose §3.4).
func resolveModel(cfg Config, requestModel string) ([]ModelRef, error) {
	model := requestModel
	if model == "" {
		model = cfg.DefaultModel
	}
	if model == "" {
		return nil, fmt.Errorf("%w: request names no model and config has no default_model", ErrUnknownAlias)
	}

	for _, m := range cfg.Models {
		if m.Alias == model {
			return m.Models, nil
		}
	}

	return nil, fmt.Errorf("%w: %q", ErrUnknownAlias, model)
}

// buildCandidates creates the list of possible candidates for a request.
//
// Candidate order IS the attempt order: the alias (ladder) lists its steps in
// order of preference, and that order is preserved here. The loop is therefore
// ref-major — for each step, every account serving that step's provider, in
// config order. An account is reached only through a step that names its
// provider, so a step the operator put second never overtakes the first.
//
// Two accounts of the same provider within one step (e.g. a free and a paid
// key for the same endpoint) are attempted in cfg.Accounts order.
func buildCandidates(
	ctx context.Context,
	cfg Config,
	providers map[string]Provider,
	quotaStore QuotaStore,
	health *HealthTracker,
	spend *SpendTracker,
	inflight *InflightTracker,
	requestModel string,
) ([]Candidate, error) {
	refs, err := resolveModel(cfg, requestModel)
	if err != nil {
		return nil, err
	}

	var candidates []Candidate
	for _, ref := range refs {
		for _, acc := range cfg.Accounts {
			if acc.Provider != ref.Provider {
				continue
			}
			prov, ok := providers[acc.Provider]
			if !ok {
				continue
			}
			candidates = append(candidates, newCandidate(ctx, acc, prov, ref.Model, quotaStore, health, spend, inflight))
		}
	}

	return candidates, nil
}

// newCandidate assembles one (account, provider, model) candidate with its
// live quota, health, load and cost metadata.
func newCandidate(
	ctx context.Context,
	acc AccountConfig,
	prov Provider,
	model string,
	quotaStore QuotaStore,
	health *HealthTracker,
	spend *SpendTracker,
	inflight *InflightTracker,
) Candidate {
	remaining, remainErr := quotaStore.Remaining(ctx, acc.ID)
	// Fail-open: if we can't check remaining quota, assume free if configured.
	// Reserve() will enforce the actual limit.
	free := acc.DailyFree > 0 && (remaining > 0 || remainErr != nil)

	return Candidate{
		Provider:               prov,
		AccountID:              acc.ID,
		Auth:                   acc.Auth,
		Model:                  model,
		Free:                   free,
		Remaining:              remaining,
		QuotaUnit:              acc.QuotaUnit,
		Health:                 health.GetHealth(acc.ID),
		Inflight:               inflight.Get(acc.ID),
		CostPerToken:           acc.CostPerToken,
		CostPerInputToken:      acc.CostPerInputToken,
		CostPerOutputToken:     acc.CostPerOutputToken,
		CostPerAudioInputToken: resolveModalityCost(acc.CostPerAudioInputToken, acc.CostPerInputToken),
		CostPerImageInputToken: resolveModalityCost(acc.CostPerImageInputToken, acc.CostPerInputToken),
		CostPerVideoInputToken: resolveModalityCost(acc.CostPerVideoInputToken, acc.CostPerInputToken),
		MaxDailySpend:          acc.MaxDailySpend,
		CurrentSpend:           spend.GetSpend(acc.ID),
	}
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
