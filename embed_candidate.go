package inferrouter

import "context"

// EstimateEmbedTokens provides a rough token count estimate for an embedding
// batch. Used only for quota pre-reservation sizing; the actual token count
// is committed on the successful call (providers typically don't return
// per-input token counts for embeddings, so we commit with the estimate).
//
// Uses the same character-per-token heuristic as chat (~4 chars/token).
// Per-request overhead is omitted since embeddings have no system prompts
// or role scaffolding.
func EstimateEmbedTokens(inputs []string) int64 {
	var total int64
	for _, in := range inputs {
		total += int64(len(in)) / charsPerTextToken
	}
	return total
}

// EmbedCandidate is a possible (provider, account, model) tuple for an
// embedding request. Symmetric to Candidate for chat, but references
// EmbeddingProvider and uses the embedding-specific cost field.
type EmbedCandidate struct {
	Provider      EmbeddingProvider
	AccountID     string
	Auth          Auth
	Model         string
	Free          bool
	Remaining     int64
	QuotaUnit     QuotaUnit
	Health        HealthState
	Cost          float64 // CostPerEmbeddingInputToken
	MaxDailySpend float64
	CurrentSpend  float64
}

// buildEmbedCandidates creates the list of possible embedding candidates for
// a request. Symmetric to buildCandidates for chat, but filters by
// EmbeddingProvider capability and embedding-specific cost field.
//
// Note: all candidates returned for a given model MUST be using the same
// model — this invariant is enforced at NewRouter startup by
// validateEmbeddingAliases (single-model alias rule from RFC §3.6).
func buildEmbedCandidates(
	ctx context.Context,
	cfg Config,
	embedProviders map[string]EmbeddingProvider,
	quotaStore QuotaStore,
	health *HealthTracker,
	spend *SpendTracker,
	requestModel string,
) []EmbedCandidate {
	refs := resolveModel(cfg, requestModel)
	var candidates []EmbedCandidate

	for _, acc := range cfg.Accounts {
		prov, ok := embedProviders[acc.Provider]
		if !ok {
			continue // provider does not implement EmbeddingProvider
		}

		models := embedModelsForAccount(refs, acc, prov, requestModel, cfg)
		for _, model := range models {
			// Account must have either free quota or a non-zero embedding cost
			// to be a valid candidate. Zero-cost paid accounts are explicitly
			// treated as "embeddings disabled" per config spec.
			if acc.DailyFree == 0 && acc.CostPerEmbeddingInputToken == 0 {
				continue
			}

			remaining, remainErr := quotaStore.Remaining(ctx, acc.ID)
			// Fail-open: assume free if we can't check.
			free := acc.DailyFree > 0 && (remaining > 0 || remainErr != nil)

			candidates = append(candidates, EmbedCandidate{
				Provider:      prov,
				AccountID:     acc.ID,
				Auth:          acc.Auth,
				Model:         model,
				Free:          free,
				Remaining:     remaining,
				QuotaUnit:     acc.QuotaUnit,
				Health:        health.GetHealth(acc.ID),
				Cost:          acc.CostPerEmbeddingInputToken,
				MaxDailySpend: acc.MaxDailySpend,
				CurrentSpend:  spend.GetSpend(acc.ID),
			})
		}
	}

	return candidates
}

// embedModelsForAccount returns the embedding models to try for a given
// account. Mirrors modelsForAccount but checks SupportsEmbeddingModel instead
// of SupportsModel.
func embedModelsForAccount(refs []ModelRef, acc AccountConfig, prov EmbeddingProvider, requestModel string, cfg Config) []string {
	if len(refs) > 0 {
		var models []string
		for _, ref := range refs {
			if ref.Provider == acc.Provider && prov.SupportsEmbeddingModel(ref.Model) {
				models = append(models, ref.Model)
			}
		}
		return models
	}

	// Direct model name (no alias).
	model := requestModel
	if model == "" {
		model = cfg.DefaultModel
	}
	if model != "" && prov.SupportsEmbeddingModel(model) {
		return []string{model}
	}
	return nil
}

// filterEmbedCandidates removes unhealthy candidates and enforces paid/spend
// limits. Symmetric to filterCandidates for chat; no multimodal dimension.
func filterEmbedCandidates(candidates []EmbedCandidate, allowPaid bool) []EmbedCandidate {
	filtered := make([]EmbedCandidate, 0, len(candidates))
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
		filtered = append(filtered, c)
	}
	return filtered
}
