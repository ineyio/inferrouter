package inferrouter

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
)

// validateEmbeddingAliases enforces the single-model invariant for embedding
// aliases (RFC §3.6). An alias whose entries reference any embedding model
// must contain exactly one entry.
//
// Cross-model fallback in embedding aliases is a correctness bug: embedding
// vector spaces are not compatible between models, so routing a query to
// model A and an index chunk to model B would produce ~random cosine scores
// that silently land in a realistic range. Fail-fast at startup avoids a
// silent quality regression in production.
func validateEmbeddingAliases(cfg Config, embedProviders map[string]EmbeddingProvider) error {
	for _, alias := range cfg.Models {
		containsEmbedding := false
		for _, ref := range alias.Models {
			prov, ok := embedProviders[ref.Provider]
			if !ok {
				continue
			}
			if prov.SupportsEmbeddingModel(ref.Model) {
				containsEmbedding = true
				break
			}
		}
		if containsEmbedding && len(alias.Models) > 1 {
			return fmt.Errorf("%w: embedding alias %q must contain exactly one model entry "+
				"(got %d); cross-model fallback breaks RAG correctness — use multi-account "+
				"fallback on the same model instead",
				ErrInvalidConfig, alias.Alias, len(alias.Models))
		}
	}
	return nil
}

// prepareEmbedRoute builds, filters, and orders embedding candidates.
// Symmetric to prepareRoute for chat, but uses EmbeddingProvider capability
// and simpler free-first ordering (inline — no separate Policy type for
// embeddings in Phase 1).
func (r *Router) prepareEmbedRoute(ctx context.Context, requestModel string) ([]EmbedCandidate, error) {
	candidates := buildEmbedCandidates(ctx, r.cfg, r.embedProviders, r.quotaStore, r.health, r.spend, requestModel)
	candidates = filterEmbedCandidates(candidates, r.cfg.AllowPaid)
	if len(candidates) == 0 {
		return nil, ErrNoEmbeddingProviders
	}

	// Free-first inline ordering. Embedding path does not currently have
	// a pluggable Policy — if custom ordering is needed, parallel the chat
	// Policy interface at that point.
	free := candidates[:0:0]
	paid := make([]EmbedCandidate, 0, len(candidates))
	for _, c := range candidates {
		if c.Free {
			free = append(free, c)
		} else {
			paid = append(paid, c)
		}
	}
	return append(free, paid...), nil
}

// acquireEmbed attempts RPM check and quota reservation for an embed candidate.
func (r *Router) acquireEmbed(ctx context.Context, c EmbedCandidate, estimatedTokens int64) (Reservation, *CandidateError) {
	if !r.rateLimiter.Allow(c.AccountID, c.Model) {
		return Reservation{}, &CandidateError{
			Provider: c.Provider.Name(), AccountID: c.AccountID, Model: c.Model,
			Err: ErrRPMExceeded,
		}
	}

	reserveAmount := estimatedTokens
	if c.QuotaUnit == QuotaRequests {
		reserveAmount = 1
	}

	reservation, err := r.quotaStore.Reserve(ctx, c.AccountID, reserveAmount, c.QuotaUnit, uuid.New().String())
	if err != nil {
		return Reservation{}, &CandidateError{
			Provider: c.Provider.Name(), AccountID: c.AccountID, Model: c.Model,
			Err: err,
		}
	}
	return reservation, nil
}

// settleEmbedFailure handles rollback, health tracking, and metering after
// an embedding provider error. Symmetric to settleFailure for chat.
func (r *Router) settleEmbedFailure(ctx context.Context, c EmbedCandidate, reservation Reservation, providerErr error, duration time.Duration, attempt int) (*RouterError, CandidateError) {
	rollbackErr := r.quotaStore.Rollback(ctx, reservation)
	r.health.RecordFailure(c.AccountID)

	resultErr := providerErr
	if rollbackErr != nil {
		resultErr = fmt.Errorf("%w (rollback failed: %v)", providerErr, rollbackErr)
	}

	r.meter.OnResult(ResultEvent{
		Provider:  c.Provider.Name(),
		AccountID: c.AccountID,
		Model:     c.Model,
		Free:      c.Free,
		Success:   false,
		Duration:  duration,
		Error:     resultErr,
	})

	ce := CandidateError{
		Provider: c.Provider.Name(), AccountID: c.AccountID, Model: c.Model,
		Err: providerErr,
	}

	if IsFatal(providerErr) {
		return &RouterError{
			Err:       providerErr,
			Provider:  c.Provider.Name(),
			AccountID: c.AccountID,
			Model:     c.Model,
			Attempts:  attempt + 1,
		}, ce
	}
	return nil, ce
}

// settleEmbedSuccess handles quota commit, health tracking, spend recording,
// and metering after a successful embedding provider response.
func (r *Router) settleEmbedSuccess(ctx context.Context, c EmbedCandidate, reservation Reservation, usage EmbedUsage, duration time.Duration) {
	actualTokens := usage.TotalTokens
	if c.QuotaUnit == QuotaRequests {
		actualTokens = 1
	}
	commitErr := r.quotaStore.Commit(ctx, reservation, actualTokens)
	r.health.RecordSuccess(c.AccountID)

	dollarCost := float64(usage.InputTokens) * c.Cost
	if dollarCost > 0 {
		r.spend.RecordSpend(c.AccountID, dollarCost)
	}

	var meterErr error
	if commitErr != nil {
		meterErr = fmt.Errorf("quota commit failed: %w", commitErr)
	}

	// Reuse chat Usage type for the meter event (embedding fills only input
	// tokens). Meter consumers that care about embedding-vs-chat distinction
	// can inspect Model.
	r.meter.OnResult(ResultEvent{
		Provider:   c.Provider.Name(),
		AccountID:  c.AccountID,
		Model:      c.Model,
		Free:       c.Free,
		Success:    commitErr == nil,
		Duration:   duration,
		Usage:      Usage{PromptTokens: usage.InputTokens, TotalTokens: usage.TotalTokens},
		Error:      meterErr,
		DollarCost: dollarCost,
	})
}

// buildEmbedProviderRequest creates the request to send to an EmbeddingProvider
// adapter for a single (pre-split) batch.
func buildEmbedProviderRequest(c EmbedCandidate, req EmbedRequest, inputs []string) EmbedProviderRequest {
	return EmbedProviderRequest{
		Auth:                 c.Auth,
		Model:                c.Model,
		Inputs:               inputs,
		TaskType:             req.TaskType,
		OutputDimensionality: req.OutputDimensionality,
	}
}

// --- Public API ---

// Embed performs a synchronous embedding request against a single candidate
// batch. This is the low-level escape hatch for callers that manage their
// own batching. For automatic batch splitting (which you probably want),
// use EmbedBatch.
//
// Returns ErrBatchTooLarge if len(req.Inputs) exceeds any available
// provider's MaxBatchSize — callers should switch to EmbedBatch instead.
func (r *Router) Embed(ctx context.Context, req EmbedRequest) (EmbedResponse, error) {
	if len(req.Inputs) == 0 {
		return EmbedResponse{}, fmt.Errorf("%w: empty inputs", ErrInvalidRequest)
	}

	ordered, err := r.prepareEmbedRoute(ctx, req.Model)
	if err != nil {
		return EmbedResponse{}, err
	}

	// Validate batch size against the best (first) candidate. If that
	// provider cannot fit the batch, require EmbedBatch — we don't
	// silently route to a different provider whose MaxBatchSize happens
	// to fit, because the user's intent with Embed is "use one provider call".
	if len(req.Inputs) > ordered[0].Provider.MaxBatchSize() {
		return EmbedResponse{}, fmt.Errorf("%w: %d inputs > provider max %d (use EmbedBatch)",
			ErrBatchTooLarge, len(req.Inputs), ordered[0].Provider.MaxBatchSize())
	}

	estimatedTokens := EstimateEmbedTokens(req.Inputs)
	resp, _, err := r.embedOnce(ctx, ordered, req, req.Inputs, estimatedTokens)
	return resp, err
}

// EmbedBatch performs an embedding request with automatic batch splitting.
// req.Inputs is split into sub-batches of at most MaxBatchSize (per the
// first candidate provider) and each sub-batch goes through the full
// candidate selection + reservation workflow.
//
// Happy path: returns EmbedResponse with Embeddings of len(req.Inputs),
// Usage summed across sub-batches, Routing reflecting the LAST successful
// sub-batch (typically all sub-batches route the same way).
//
// Partial failure path: returns EmbedResponse with a valid prefix of
// Embeddings (the successfully processed portion) AND a non-nil
// *ErrPartialBatch error. Consumer pattern:
//
//	resp, err := router.EmbedBatch(ctx, req)
//	var partial *ErrPartialBatch
//	if errors.As(err, &partial) {
//	    persist(resp.Embeddings) // valid prefix, len == partial.ProcessedInputs
//	    return retryWith(req.Inputs[partial.ProcessedInputs:])
//	}
//
// Full failure path (no successful sub-batches): returns zero-value
// EmbedResponse with a non-*ErrPartialBatch error (RouterError or sentinel).
func (r *Router) EmbedBatch(ctx context.Context, req EmbedRequest) (EmbedResponse, error) {
	if len(req.Inputs) == 0 {
		return EmbedResponse{}, fmt.Errorf("%w: empty inputs", ErrInvalidRequest)
	}

	ordered, err := r.prepareEmbedRoute(ctx, req.Model)
	if err != nil {
		return EmbedResponse{}, err
	}

	// Split request against the first candidate's MaxBatchSize. If that
	// candidate fails and the next one has a smaller MaxBatchSize, we
	// would need to re-split — but in practice candidates for a given
	// model have the same MaxBatchSize (all Gemini providers = 100), so
	// this is fine for Phase 1. A future optimization could re-split per
	// candidate on failure.
	maxBatch := ordered[0].Provider.MaxBatchSize()
	chunks := splitIntoBatches(req.Inputs, maxBatch)

	var (
		allEmbeddings = make([][]float32, 0, len(req.Inputs))
		totalUsage    EmbedUsage
		lastRouting   RoutingInfo
		lastModel     string
	)

	for chunkIdx, chunkInputs := range chunks {
		estimatedTokens := EstimateEmbedTokens(chunkInputs)
		resp, routing, err := r.embedOnce(ctx, ordered, req, chunkInputs, estimatedTokens)
		if err != nil {
			// Some chunks may have already succeeded. Return partial
			// result so the consumer can persist valid embeddings and
			// retry with the remainder.
			if chunkIdx > 0 {
				return EmbedResponse{
					Embeddings: allEmbeddings,
					Model:      lastModel,
					Usage:      totalUsage,
					Routing:    lastRouting,
				}, &ErrPartialBatch{ProcessedInputs: len(allEmbeddings), Cause: err}
			}
			// First chunk failed — full failure, no partial result.
			return EmbedResponse{}, err
		}
		allEmbeddings = append(allEmbeddings, resp.Embeddings...)
		totalUsage.InputTokens += resp.Usage.InputTokens
		totalUsage.TotalTokens += resp.Usage.TotalTokens
		lastRouting = routing
		lastModel = resp.Model
	}

	return EmbedResponse{
		Embeddings: allEmbeddings,
		Model:      lastModel,
		Usage:      totalUsage,
		Routing:    lastRouting,
	}, nil
}

// embedOnce executes a single embed call against the ordered candidate list
// with full Reserve → Execute → Commit/Rollback workflow. Retries across
// candidates on retryable errors; returns a fatal RouterError or ErrAllFailed
// wrapper if no candidate succeeds.
//
// The second return value is the RoutingInfo of the successful candidate,
// so EmbedBatch can surface it without re-reading response fields.
func (r *Router) embedOnce(ctx context.Context, ordered []EmbedCandidate, req EmbedRequest, inputs []string, estimatedTokens int64) (EmbedResponse, RoutingInfo, error) {
	var tried []CandidateError
	for attempt, c := range ordered {
		reservation, skip := r.acquireEmbed(ctx, c, estimatedTokens)
		if skip != nil {
			tried = append(tried, *skip)
			continue
		}

		r.meter.OnRoute(RouteEvent{
			Provider:    c.Provider.Name(),
			AccountID:   c.AccountID,
			Model:       c.Model,
			Free:        c.Free,
			AttemptNum:  attempt + 1,
			EstimatedIn: estimatedTokens,
		})

		start := time.Now()
		provResp, err := c.Provider.Embed(ctx, buildEmbedProviderRequest(c, req, inputs))
		duration := time.Since(start)

		if err != nil {
			fatal, ce := r.settleEmbedFailure(ctx, c, reservation, err, duration, attempt)
			if fatal != nil {
				return EmbedResponse{}, RoutingInfo{}, fatal
			}
			tried = append(tried, ce)
			continue
		}

		r.settleEmbedSuccess(ctx, c, reservation, provResp.Usage, duration)

		routing := RoutingInfo{
			Provider:  c.Provider.Name(),
			AccountID: c.AccountID,
			Model:     c.Model,
			Attempts:  attempt + 1,
			Free:      c.Free,
		}
		return EmbedResponse{
			Embeddings: provResp.Embeddings,
			Model:      provResp.Model,
			Usage:      provResp.Usage,
			Routing:    routing,
		}, routing, nil
	}

	return EmbedResponse{}, RoutingInfo{}, allFailedError(tried, len(ordered))
}

// splitIntoBatches splits inputs into sub-slices of at most maxBatch length.
// Preserves order.
func splitIntoBatches(inputs []string, maxBatch int) [][]string {
	if maxBatch <= 0 {
		return [][]string{inputs}
	}
	n := len(inputs)
	if n <= maxBatch {
		return [][]string{inputs}
	}
	batches := make([][]string, 0, (n+maxBatch-1)/maxBatch)
	for i := 0; i < n; i += maxBatch {
		end := i + maxBatch
		if end > n {
			end = n
		}
		batches = append(batches, inputs[i:end])
	}
	return batches
}
