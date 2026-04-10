package inferrouter

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
)

// Router routes LLM requests across multiple providers and accounts.
type Router struct {
	cfg         Config
	providers   map[string]Provider
	policy      Policy
	quotaStore  QuotaStore
	meter       Meter
	health      *HealthTracker
	spend       *SpendTracker
	rateLimiter *RateLimiter
}

// Option configures a Router.
type Option func(*Router)

// WithPolicy sets the routing policy.
func WithPolicy(p Policy) Option {
	return func(r *Router) { r.policy = p }
}

// WithQuotaStore sets the quota store.
func WithQuotaStore(qs QuotaStore) Option {
	return func(r *Router) { r.quotaStore = qs }
}

// WithMeter sets the meter.
func WithMeter(m Meter) Option {
	return func(r *Router) { r.meter = m }
}

// WithHealthTracker sets the health tracker.
func WithHealthTracker(h *HealthTracker) Option {
	return func(r *Router) { r.health = h }
}

// WithHealthConfig sets health tracker configuration.
func WithHealthConfig(cfg HealthConfig) Option {
	return func(r *Router) { r.health = NewHealthTrackerWithConfig(cfg) }
}

// WithSpendTracker sets the spend tracker.
func WithSpendTracker(s *SpendTracker) Option {
	return func(r *Router) { r.spend = s }
}

// WithRateLimiter sets a custom rate limiter.
func WithRateLimiter(rl *RateLimiter) Option {
	return func(r *Router) { r.rateLimiter = rl }
}

// NewRouter creates a new Router with the given config and providers.
// Default components (FreeFirstPolicy, MemoryQuotaStore, NoopMeter) are used
// unless overridden via options.
func NewRouter(cfg Config, providers []Provider, opts ...Option) (*Router, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	if len(providers) == 0 {
		return nil, fmt.Errorf("inferrouter: at least one provider is required")
	}

	provMap := make(map[string]Provider, len(providers))
	for _, p := range providers {
		provMap[p.Name()] = p
	}

	cfg.NormalizeCosts()

	r := &Router{
		cfg:       cfg,
		providers: provMap,
		health:    NewHealthTracker(),
		spend:     NewSpendTracker(),
	}

	for _, opt := range opts {
		opt(r)
	}

	// Apply defaults after options.
	if r.policy == nil {
		r.policy = &defaultFreeFirstPolicy{}
	}
	if r.quotaStore == nil {
		r.quotaStore = &noopQuotaStore{}
	}
	if r.meter == nil {
		r.meter = &noopMeter{}
	}
	if r.rateLimiter == nil {
		r.rateLimiter = NewRateLimiter()
	}

	// Initialize rate limits from config.
	for _, acc := range cfg.Accounts {
		// Per-model limits take priority.
		for model, limits := range acc.ModelLimits {
			r.rateLimiter.SetModelLimits(acc.ID, model, limits)
		}
		// Account-level RPM as fallback for models without explicit limits.
		if acc.RPM > 0 {
			r.rateLimiter.SetAccountDefault(acc.ID, Limits{RPM: acc.RPM})
		}
	}

	// Initialize quota limits from config if the store supports it.
	if init, ok := r.quotaStore.(QuotaInitializer); ok {
		for _, acc := range cfg.Accounts {
			if acc.DailyFree > 0 || !acc.PaidEnabled {
				if err := init.SetQuota(acc.ID, acc.DailyFree, acc.QuotaUnit); err != nil {
					return nil, fmt.Errorf("inferrouter: init quota for %q: %w", acc.ID, err)
				}
			}
		}
	}

	return r, nil
}

// --- Domain phases of a routing request ---

// prepareRoute resolves the model, builds, filters, and orders candidates.
// When needMultimodal is true and the filter empties the list, the more
// specific ErrMultimodalUnavailable is returned instead of ErrNoCandidates.
func (r *Router) prepareRoute(ctx context.Context, requestModel string, needMultimodal bool) ([]Candidate, error) {
	candidates, err := buildCandidates(ctx, r.cfg, r.providers, r.quotaStore, r.health, r.spend, requestModel)
	if err != nil {
		return nil, err
	}

	candidates = filterCandidates(candidates, r.cfg.AllowPaid, needMultimodal)
	if len(candidates) == 0 {
		if needMultimodal {
			return nil, ErrMultimodalUnavailable
		}
		return nil, ErrNoCandidates
	}

	return r.policy.Select(candidates), nil
}

// acquire attempts RPM check and quota reservation for a candidate.
// Returns the reservation on success, or a CandidateError if the candidate should be skipped.
func (r *Router) acquire(ctx context.Context, c Candidate, estimatedTokens int64) (Reservation, *CandidateError) {
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

// settleFailure handles rollback, health tracking, and metering after a provider error.
// Returns a RouterError if the error is fatal (caller should return immediately),
// or a CandidateError to append to the tried list.
func (r *Router) settleFailure(ctx context.Context, c Candidate, reservation Reservation, providerErr error, duration time.Duration, attempt int) (*RouterError, CandidateError) {
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

// settleSuccess handles quota commit, health tracking, spend recording, and metering
// after a successful provider response.
func (r *Router) settleSuccess(ctx context.Context, c Candidate, reservation Reservation, usage Usage, duration time.Duration) {
	actualTokens := usage.TotalTokens
	if c.QuotaUnit == QuotaRequests {
		actualTokens = 1
	}
	commitErr := r.quotaStore.Commit(ctx, reservation, actualTokens)
	r.health.RecordSuccess(c.AccountID)

	dollarCost := calculateSpend(c, usage)
	if dollarCost > 0 {
		r.spend.RecordSpend(c.AccountID, dollarCost)
	}

	var meterErr error
	if commitErr != nil {
		meterErr = fmt.Errorf("quota commit failed: %w", commitErr)
	}

	r.meter.OnResult(ResultEvent{
		Provider:   c.Provider.Name(),
		AccountID:  c.AccountID,
		Model:      c.Model,
		Free:       c.Free,
		Success:    commitErr == nil,
		Duration:   duration,
		Usage:      usage,
		Error:      meterErr,
		DollarCost: dollarCost,
	})
}

// messagesHaveMedia reports whether any message carries non-text parts.
func messagesHaveMedia(msgs []Message) bool {
	for _, m := range msgs {
		for _, p := range m.Parts {
			if p.IsMedia() {
				return true
			}
		}
	}
	return false
}

// buildProviderRequest creates the request to send to the provider.
func buildProviderRequest(c Candidate, req ChatRequest, stream, hasMedia bool) ProviderRequest {
	return ProviderRequest{
		Auth:        c.Auth,
		Model:       c.Model,
		Messages:    req.Messages,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		TopP:        req.TopP,
		Stop:        req.Stop,
		Stream:      stream,
		HasMedia:    hasMedia,
	}
}

func allFailedError(tried []CandidateError, total int) error {
	if len(tried) > 0 {
		return &RouterError{
			Err:      ErrAllFailed,
			Attempts: total,
			Tried:    tried,
		}
	}
	return ErrNoCandidates
}

// --- Public API ---

// ChatCompletion performs a synchronous chat completion with automatic routing.
func (r *Router) ChatCompletion(ctx context.Context, req ChatRequest) (ChatResponse, error) {
	estimatedTokens := EstimateTokens(req.Messages)
	hasMedia := messagesHaveMedia(req.Messages)

	ordered, err := r.prepareRoute(ctx, req.Model, hasMedia)
	if err != nil {
		return ChatResponse{}, err
	}

	var tried []CandidateError
	for attempt, c := range ordered {
		reservation, skip := r.acquire(ctx, c, estimatedTokens)
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
		resp, err := c.Provider.ChatCompletion(ctx, buildProviderRequest(c, req, false, hasMedia))
		duration := time.Since(start)

		if err != nil {
			fatal, ce := r.settleFailure(ctx, c, reservation, err, duration, attempt)
			if fatal != nil {
				return ChatResponse{}, fatal
			}
			tried = append(tried, ce)
			continue
		}

		r.settleSuccess(ctx, c, reservation, resp.Usage, duration)

		return ChatResponse{
			ID:    resp.ID,
			Model: resp.Model,
			Choices: []Choice{{
				Index:        0,
				Message:      Message{Role: "assistant", Content: resp.Content},
				FinishReason: resp.FinishReason,
			}},
			Usage: resp.Usage,
			Routing: RoutingInfo{
				Provider:  c.Provider.Name(),
				AccountID: c.AccountID,
				Model:     c.Model,
				Attempts:  attempt + 1,
				Free:      c.Free,
			},
		}, nil
	}

	return ChatResponse{}, allFailedError(tried, len(ordered))
}

// ChatCompletionStream performs a streaming chat completion with automatic routing.
func (r *Router) ChatCompletionStream(ctx context.Context, req ChatRequest) (*RouterStream, error) {
	estimatedTokens := EstimateTokens(req.Messages)
	hasMedia := messagesHaveMedia(req.Messages)

	ordered, err := r.prepareRoute(ctx, req.Model, hasMedia)
	if err != nil {
		return nil, err
	}

	var tried []CandidateError
	for attempt, c := range ordered {
		reservation, skip := r.acquire(ctx, c, estimatedTokens)
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

		stream, err := c.Provider.ChatCompletionStream(ctx, buildProviderRequest(c, req, true, hasMedia))
		if err != nil {
			fatal, ce := r.settleFailure(ctx, c, reservation, err, 0, attempt)
			if fatal != nil {
				return nil, fatal
			}
			tried = append(tried, ce)
			continue
		}

		return &RouterStream{
			inner:       stream,
			reservation: reservation,
			quotaStore:  r.quotaStore,
			meter:       r.meter,
			health:      r.health,
			spend:       r.spend,
			candidate:   c,
			startTime:   time.Now(),
		}, nil
	}

	return nil, allFailedError(tried, len(ordered))
}

// defaultFreeFirstPolicy is an inline free-first policy to avoid import cycles.
type defaultFreeFirstPolicy struct{}

func (p *defaultFreeFirstPolicy) Select(candidates []Candidate) []Candidate {
	var free, paid []Candidate
	for _, c := range candidates {
		if c.Free {
			free = append(free, c)
		} else {
			paid = append(paid, c)
		}
	}
	return append(free, paid...)
}

// noopQuotaStore is a quota store that allows everything (no limits).
type noopQuotaStore struct{}

func (s *noopQuotaStore) Reserve(_ context.Context, accountID string, amount int64, unit QuotaUnit, _ string) (Reservation, error) {
	return Reservation{ID: uuid.New().String(), AccountID: accountID, Amount: amount, Unit: unit}, nil
}
func (s *noopQuotaStore) Commit(context.Context, Reservation, int64) error { return nil }
func (s *noopQuotaStore) Rollback(context.Context, Reservation) error      { return nil }
func (s *noopQuotaStore) Remaining(context.Context, string) (int64, error) { return 0, nil }

// noopMeter is a meter that does nothing.
type noopMeter struct{}

func (m *noopMeter) OnRoute(RouteEvent)   {}
func (m *noopMeter) OnResult(ResultEvent) {}
