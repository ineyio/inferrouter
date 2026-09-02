package inferrouter

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
)

// embedRateLimitRetries is how many extra attempts a single candidate gets
// after the provider answers 429.
//
// With one embedding account configured — the normal case, since vector
// spaces forbid cross-model fallback — "try the next candidate" is not a
// recovery. A 429 window at the provider is measured in seconds, so pausing
// and asking the same account again is the only move that can still succeed.
const embedRateLimitRetries = 2

// embedRetryBackoff is the pause used when the provider does not say how long
// to wait. A provider-supplied RetryAfter always wins over these.
var embedRetryBackoff = [embedRateLimitRetries]time.Duration{
	200 * time.Millisecond,
	time.Second,
}

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
		models := make(map[string]struct{}, len(alias.Models))
		for _, ref := range alias.Models {
			models[ref.Model] = struct{}{}
			prov, ok := embedProviders[ref.Provider]
			if !ok {
				continue
			}
			if prov.SupportsEmbeddingModel(ref.Model) {
				containsEmbedding = true
			}
		}
		// The guarantee is one MODEL, not one step: vector spaces are tied to
		// the model, not to the endpoint serving it. Several steps naming the
		// same model are the recommended multi-account fallback — and since
		// resolution is strict, a ladder is the only way to express it.
		if containsEmbedding && len(models) > 1 {
			return fmt.Errorf("%w: embedding alias %q must reference exactly one model "+
				"(got %d distinct); cross-model fallback breaks RAG correctness — use "+
				"multi-account fallback on the same model instead",
				ErrInvalidConfig, alias.Alias, len(models))
		}
	}
	return nil
}

// prepareEmbedRoute builds and filters embedding candidates.
// Symmetric to prepareRoute for chat: config order is the attempt order (R1),
// so accounts are tried as the alias and cfg.Accounts declare them. Embeddings
// have no pluggable Policy — if deliberate reordering is ever needed here,
// parallel the chat Policy interface at that point.
func (r *Router) prepareEmbedRoute(ctx context.Context, requestModel string) ([]EmbedCandidate, error) {
	candidates, err := buildEmbedCandidates(ctx, r.cfg, r.embedProviders, r.quotaStore, r.health, r.spend, requestModel)
	if err != nil {
		return nil, err
	}

	candidates = filterEmbedCandidates(candidates, r.cfg.AllowPaid)
	if len(candidates) == 0 {
		return nil, ErrNoEmbeddingProviders
	}

	return candidates, nil
}

// acquireEmbed paces the request against the account's rate limits and
// reserves quota for an embed candidate.
//
// Unlike the chat path, which skips a rate-limited candidate and moves down
// the ladder, this waits: an embedding caller can absorb a few hundred
// milliseconds, but cannot absorb an empty block of memory, and the ladder
// below is usually empty anyway.
func (r *Router) acquireEmbed(ctx context.Context, c EmbedCandidate, estimatedTokens int64) (Reservation, *CandidateError) {
	if err := r.rateLimiter.Wait(ctx, c.AccountID, c.Model); err != nil {
		return Reservation{}, &CandidateError{
			Provider: c.Provider.Name(), AccountID: c.AccountID, Model: c.Model,
			Err: err,
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

	// A 429 is backpressure, not ill health: the account is answering, and
	// answering correctly. Counting it as a failure trips the breaker after
	// three of them, and with a single embedding account that turns a
	// seconds-long provider window into a 30-second outage for every caller —
	// the same amplifier that killed 187 requests in the gonka incident.
	if !errors.Is(providerErr, ErrRateLimited) {
		r.health.RecordFailure(c.AccountID)
	}

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
		provResp, err := r.embedWithRetry(ctx, c, buildEmbedProviderRequest(c, req, inputs))
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

// embedWithRetry calls one candidate, retrying that same candidate on a
// provider 429 after the delay the provider asked for (or a short backoff if
// it asked for nothing).
//
// The retry happens inside the caller's reservation: one logical request stays
// one reservation and one rate-limiter slot, so a retry cannot double-charge
// quota or double-count against our own pacing.
func (r *Router) embedWithRetry(ctx context.Context, c EmbedCandidate, preq EmbedProviderRequest) (EmbedProviderResponse, error) {
	for attempt := 0; ; attempt++ {
		resp, err := c.Provider.Embed(ctx, preq)
		if err == nil || attempt >= embedRateLimitRetries || !errors.Is(err, ErrRateLimited) {
			return resp, err
		}

		delay := embedRetryBackoff[attempt]
		var rateLimited *RateLimitedError
		if errors.As(err, &rateLimited) && rateLimited.RetryAfter > 0 {
			delay = rateLimited.RetryAfter
		}

		// Sleeping past the caller's deadline buys nothing: the context would
		// cancel mid-pause and the caller would get the cancellation instead
		// of the rate-limit error that explains it.
		if !fitsDeadline(ctx, delay) {
			return resp, err
		}
		if sleepErr := r.sleep(ctx, delay); sleepErr != nil {
			return resp, err
		}
	}
}

// fitsDeadline reports whether a pause of d still leaves the context alive.
// A context without a deadline fits anything.
func fitsDeadline(ctx context.Context, d time.Duration) bool {
	deadline, ok := ctx.Deadline()
	if !ok {
		return true
	}
	return time.Now().Add(d).Before(deadline)
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
