package inferrouter

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
)

// Router routes LLM requests across multiple providers and accounts.
type Router struct {
	cfg        Config
	providers  map[string]Provider
	policy     Policy
	quotaStore QuotaStore
	meter      Meter
	health     *HealthTracker
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

// NewRouter creates a new Router with the given config and providers.
// Default components (FreeFirstPolicy, MemoryQuotaStore, NoopMeter) are used
// unless overridden via options.
func NewRouter(cfg Config, providers []Provider, opts ...Option) (*Router, error) {
	if len(providers) == 0 {
		return nil, fmt.Errorf("inferrouter: at least one provider is required")
	}

	provMap := make(map[string]Provider, len(providers))
	for _, p := range providers {
		provMap[p.Name()] = p
	}

	r := &Router{
		cfg:       cfg,
		providers: provMap,
		health:    NewHealthTracker(),
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

	// Initialize quota limits from config if the store supports it.
	// All accounts get a quota entry. Paid accounts without DailyFree get unlimited (no entry).
	if init, ok := r.quotaStore.(QuotaInitializer); ok {
		for _, acc := range cfg.Accounts {
			if acc.DailyFree > 0 || !acc.PaidEnabled {
				init.SetQuota(acc.ID, acc.DailyFree, acc.QuotaUnit)
			}
		}
	}

	return r, nil
}

// ChatCompletion performs a synchronous chat completion with automatic routing.
func (r *Router) ChatCompletion(ctx context.Context, req ChatRequest) (ChatResponse, error) {
	model := req.Model
	if model == "" {
		model = r.cfg.DefaultModel
	}

	estimatedTokens := EstimateTokens(req.Messages)

	candidates, err := buildCandidates(ctx, r.cfg, r.providers, r.quotaStore, r.health, req.Model)
	if err != nil {
		return ChatResponse{}, err
	}

	candidates = filterCandidates(candidates, r.cfg.AllowPaid)
	if len(candidates) == 0 {
		return ChatResponse{}, ErrNoCandidates
	}

	ordered := r.policy.Select(candidates)

	var lastErr error
	for attempt, c := range ordered {
		idempotencyKey := uuid.New().String()

		reserveAmount := estimatedTokens
		if c.QuotaUnit == QuotaRequests {
			reserveAmount = 1
		}

		reservation, err := r.quotaStore.Reserve(ctx, c.AccountID, reserveAmount, c.QuotaUnit, idempotencyKey)
		if err != nil {
			lastErr = err
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

		provReq := ProviderRequest{
			Auth:        c.Auth,
			Model:       c.Model,
			Messages:    req.Messages,
			Temperature: req.Temperature,
			MaxTokens:   req.MaxTokens,
			TopP:        req.TopP,
			Stop:        req.Stop,
		}

		start := time.Now()
		resp, err := c.Provider.ChatCompletion(ctx, provReq)
		duration := time.Since(start)

		if err != nil {
			_ = r.quotaStore.Rollback(ctx, reservation)
			r.health.RecordFailure(c.AccountID)
			r.meter.OnResult(ResultEvent{
				Provider:  c.Provider.Name(),
				AccountID: c.AccountID,
				Model:     c.Model,
				Free:      c.Free,
				Success:   false,
				Duration:  duration,
				Error:     err,
			})

			if IsFatal(err) {
				return ChatResponse{}, &RouterError{
					Err:       err,
					Provider:  c.Provider.Name(),
					AccountID: c.AccountID,
					Model:     c.Model,
					Attempts:  attempt + 1,
				}
			}

			lastErr = err
			continue
		}

		// Success.
		actualTokens := resp.Usage.TotalTokens
		if c.QuotaUnit == QuotaRequests {
			actualTokens = 1
		}
		_ = r.quotaStore.Commit(ctx, reservation, actualTokens)
		r.health.RecordSuccess(c.AccountID)
		r.meter.OnResult(ResultEvent{
			Provider:  c.Provider.Name(),
			AccountID: c.AccountID,
			Model:     c.Model,
			Free:      c.Free,
			Success:   true,
			Duration:  duration,
			Usage:     resp.Usage,
		})

		return ChatResponse{
			ID:    resp.ID,
			Model: resp.Model,
			Choices: []Choice{
				{
					Index:        0,
					Message:      Message{Role: "assistant", Content: resp.Content},
					FinishReason: resp.FinishReason,
				},
			},
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

	if lastErr != nil {
		return ChatResponse{}, &RouterError{
			Err:      ErrAllFailed,
			Attempts: len(ordered),
		}
	}
	return ChatResponse{}, ErrNoCandidates
}

// ChatCompletionStream performs a streaming chat completion with automatic routing.
func (r *Router) ChatCompletionStream(ctx context.Context, req ChatRequest) (*RouterStream, error) {
	model := req.Model
	if model == "" {
		model = r.cfg.DefaultModel
	}

	estimatedTokens := EstimateTokens(req.Messages)

	candidates, err := buildCandidates(ctx, r.cfg, r.providers, r.quotaStore, r.health, req.Model)
	if err != nil {
		return nil, err
	}

	candidates = filterCandidates(candidates, r.cfg.AllowPaid)
	if len(candidates) == 0 {
		return nil, ErrNoCandidates
	}

	ordered := r.policy.Select(candidates)

	var lastErr error
	for attempt, c := range ordered {
		idempotencyKey := uuid.New().String()

		reserveAmount := estimatedTokens
		if c.QuotaUnit == QuotaRequests {
			reserveAmount = 1
		}

		reservation, err := r.quotaStore.Reserve(ctx, c.AccountID, reserveAmount, c.QuotaUnit, idempotencyKey)
		if err != nil {
			lastErr = err
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

		provReq := ProviderRequest{
			Auth:        c.Auth,
			Model:       c.Model,
			Messages:    req.Messages,
			Temperature: req.Temperature,
			MaxTokens:   req.MaxTokens,
			TopP:        req.TopP,
			Stop:        req.Stop,
			Stream:      true,
		}

		stream, err := c.Provider.ChatCompletionStream(ctx, provReq)
		if err != nil {
			_ = r.quotaStore.Rollback(ctx, reservation)
			r.health.RecordFailure(c.AccountID)

			if IsFatal(err) {
				return nil, &RouterError{
					Err:       err,
					Provider:  c.Provider.Name(),
					AccountID: c.AccountID,
					Model:     c.Model,
					Attempts:  attempt + 1,
				}
			}

			lastErr = err
			continue
		}

		return &RouterStream{
			inner:       stream,
			reservation: reservation,
			quotaStore:  r.quotaStore,
			meter:       r.meter,
			health:      r.health,
			candidate:   c,
			startTime:   time.Now(),
		}, nil
	}

	if lastErr != nil {
		return nil, &RouterError{
			Err:      ErrAllFailed,
			Attempts: len(ordered),
		}
	}
	return nil, ErrNoCandidates
}

// defaultFreeFirstPolicy is an inline free-first policy to avoid import cycles.
type defaultFreeFirstPolicy struct{}

func (p *defaultFreeFirstPolicy) Select(candidates []Candidate) []Candidate {
	// Simple: free first, then paid. Within each group, preserve order.
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
func (s *noopQuotaStore) Commit(context.Context, Reservation, int64) error   { return nil }
func (s *noopQuotaStore) Rollback(context.Context, Reservation) error        { return nil }
func (s *noopQuotaStore) Remaining(context.Context, string) (int64, error)   { return 0, nil }

// noopMeter is a meter that does nothing.
type noopMeter struct{}

func (m *noopMeter) OnRoute(RouteEvent)   {}
func (m *noopMeter) OnResult(ResultEvent) {}
