package inferrouter

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"
)

type snapshotProvider struct{ name string }

func (p snapshotProvider) Name() string              { return p.name }
func (p snapshotProvider) SupportsModel(string) bool { return true }
func (p snapshotProvider) SupportsMultimodal() bool  { return false }
func (p snapshotProvider) ChatCompletion(context.Context, ProviderRequest) (ProviderResponse, error) {
	return ProviderResponse{}, errors.New("not called")
}
func (p snapshotProvider) ChatCompletionStream(context.Context, ProviderRequest) (ProviderStream, error) {
	return nil, errors.New("not called")
}

// remainingStore answers Remaining with a fixed result and nothing else.
type remainingStore struct {
	*noopQuotaStore
	remaining int64
	err       error
}

func (s remainingStore) Remaining(context.Context, string) (int64, error) {
	return s.remaining, s.err
}

func snapshotRouter(t *testing.T, acc AccountConfig, allowPaid bool, store QuotaStore, opts ...Option) *Router {
	t.Helper()
	acc.Provider, acc.ID, acc.Auth = "gw", "gw", Auth{APIKey: "k"}
	if acc.QuotaUnit == "" {
		acc.QuotaUnit = QuotaDollars
	}
	r, err := NewRouter(Config{
		AllowPaid: allowPaid,
		Accounts:  []AccountConfig{acc},
		Models:    []ModelMapping{{Alias: "cheap", Models: []ModelRef{{Provider: "gw", Model: "m"}}}},
	}, []Provider{snapshotProvider{name: "gw"}}, append([]Option{WithQuotaStore(store)}, opts...)...)
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}
	return r
}

// The report and the filter must give one answer on every combination, half-open
// included. Asked of a real router: prepareRoute is what a request goes through.
func TestSnapshot_AgreesWithFilter(t *testing.T) {
	type health struct {
		name   string
		trip   bool
		period time.Duration
		want   HealthState
	}
	healths := []health{
		{"healthy", false, time.Hour, HealthHealthy},
		{"unhealthy", true, time.Hour, HealthUnhealthy},
		{"half-open", true, 0, HealthHalfOpen},
	}

	for _, h := range healths {
		for _, free := range []bool{false, true} {
			for _, allowPaid := range []bool{false, true} {
				for _, capped := range []bool{false, true} {
					name := fmt.Sprintf("%s/free=%v/allowPaid=%v/capped=%v", h.name, free, allowPaid, capped)
					t.Run(name, func(t *testing.T) {
						acc := AccountConfig{MaxDailySpend: 1, PaidEnabled: !free}
						var store QuotaStore = remainingStore{}
						if free {
							acc.DailyFree, acc.QuotaUnit = 10, QuotaTokens
							store = remainingStore{remaining: 5}
						}
						r := snapshotRouter(t, acc, allowPaid, store, WithHealthConfig(HealthConfig{
							FailureThreshold: 1, FailureWindow: time.Hour, UnhealthyPeriod: h.period,
						}))
						if h.trip {
							r.health.RecordFailure("gw")
						}
						if capped {
							r.spend.RecordSpend("gw", 2)
						}

						status := r.AccountsSnapshot(context.Background())[0]
						if status.Health != h.want {
							t.Fatalf("Health = %v, want %v", status.Health, h.want)
						}

						_, err := r.prepareRoute(context.Background(), "cheap", false)
						dropped := errors.Is(err, ErrNoCandidates)
						if err != nil && !dropped {
							t.Fatalf("prepareRoute: %v", err)
						}
						if dropped != (status.Dropped != DropNone) {
							t.Errorf("filter dropped=%v, snapshot says %q", dropped, status.Dropped)
						}
					})
				}
			}
		}
	}
}

func TestDropReason_NamesEachCause(t *testing.T) {
	tests := []struct {
		name      string
		health    HealthState
		free      bool
		allowPaid bool
		maxSpend  float64
		spend     float64
		want      DropReason
	}{
		{"healthy paid", HealthHealthy, false, true, 1, 0, DropNone},
		{"half-open is offered", HealthHalfOpen, false, true, 1, 0, DropNone},
		{"breaker", HealthUnhealthy, false, true, 1, 0, DropBreakerOpen},
		{"breaker on a free account", HealthUnhealthy, true, true, 0, 0, DropBreakerOpen},
		{"paid not allowed", HealthHealthy, false, false, 0, 0, DropPaidNotAllowed},
		{"cap reached", HealthHealthy, false, true, 1, 1, DropSpendCap},
		{"cap below", HealthHealthy, false, true, 1, 0.99, DropNone},
		{"no cap", HealthHealthy, false, true, 0, 100, DropNone},
		{"cap does not bind a free call", HealthHealthy, true, true, 1, 2, DropNone},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := dropReason(tt.health, tt.free, tt.allowPaid, tt.maxSpend, tt.spend); got != tt.want {
				t.Errorf("dropReason = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestSnapshot_ReportsTheAccount(t *testing.T) {
	r := snapshotRouter(t, AccountConfig{MaxDailySpend: 0.25, PaidEnabled: true}, true, remainingStore{},
		WithHealthConfig(HealthConfig{FailureThreshold: 2, FailureWindow: time.Hour, UnhealthyPeriod: time.Hour}))
	r.spend.RecordSpend("gw", 0.1)
	r.inflight.Inc("gw")
	r.health.RecordFailure("gw")

	got := r.AccountsSnapshot(context.Background())
	if len(got) != 1 {
		t.Fatalf("got %d accounts", len(got))
	}
	s := got[0]
	if s.AccountID != "gw" || s.Provider != "gw" || s.Free {
		t.Errorf("identity: %+v", s)
	}
	if s.CurrentSpend != 0.1 || s.MaxDailySpend != 0.25 || s.Inflight != 1 {
		t.Errorf("load and spend: %+v", s)
	}
	if s.Health != HealthHealthy || s.FailuresInWindow != 1 || !s.UnhealthySince.IsZero() || s.Dropped != DropNone {
		t.Errorf("one failure under the threshold: %+v", s)
	}

	r.health.RecordFailure("gw")
	s = r.AccountsSnapshot(context.Background())[0]
	if s.Health != HealthUnhealthy || s.FailuresInWindow != 2 || s.UnhealthySince.IsZero() || s.Dropped != DropBreakerOpen {
		t.Errorf("tripped: %+v", s)
	}
}

// A store that cannot answer keeps the same fail-open reading as a real walk.
func TestSnapshot_FreeFailsOpenLikeTheWalk(t *testing.T) {
	store := remainingStore{err: errors.New("store down")}
	r := snapshotRouter(t, AccountConfig{DailyFree: 10, QuotaUnit: QuotaTokens}, false, store)
	s := r.AccountsSnapshot(context.Background())[0]
	if !s.Free || s.Dropped != DropNone {
		t.Errorf("store error must read as free: %+v", s)
	}
}

func TestPeek_DoesNotTransition(t *testing.T) {
	h := NewHealthTrackerWithConfig(HealthConfig{FailureThreshold: 1, FailureWindow: time.Hour, UnhealthyPeriod: 0})
	h.RecordFailure("gw")

	if got := h.Peek("gw").State; got != HealthHalfOpen {
		t.Fatalf("Peek = %v, want half-open by the same rule as GetHealth", got)
	}
	if stored := h.accounts["gw"].state; stored != HealthUnhealthy {
		t.Fatalf("Peek stored %v", stored)
	}
	if got := h.GetHealth("gw"); got != HealthHalfOpen {
		t.Fatalf("GetHealth = %v", got)
	}
	if stored := h.accounts["gw"].state; stored != HealthHalfOpen {
		t.Fatalf("GetHealth did not store the transition: %v", stored)
	}
}

func TestPeek_UnknownAccountIsHealthyAndNotCreated(t *testing.T) {
	h := NewHealthTracker()
	if got := h.Peek("gw"); got.State != HealthHealthy || got.FailuresInWindow != 0 {
		t.Errorf("Peek = %+v", got)
	}
	if _, ok := h.accounts["gw"]; ok {
		t.Error("Peek created an account")
	}
}

func TestPeek_WindowAndRecovery(t *testing.T) {
	h := NewHealthTrackerWithConfig(HealthConfig{FailureThreshold: 3, FailureWindow: time.Hour, UnhealthyPeriod: time.Hour})
	h.RecordFailure("gw")
	h.accounts["gw"].failures[0] = time.Now().Add(-2 * time.Hour)
	h.RecordFailure("gw")
	h.accounts["gw"].failures = append(h.accounts["gw"].failures, time.Now().Add(-2*time.Hour))
	if got := h.Peek("gw").FailuresInWindow; got != 1 {
		t.Errorf("FailuresInWindow = %d, want old failures not counted", got)
	}

	h.RecordFailure("gw")
	h.RecordFailure("gw")
	if h.Peek("gw").UnhealthySince.IsZero() {
		t.Fatal("tripped breaker has no time")
	}
	h.RecordSuccess("gw")
	if got := h.Peek("gw"); got.State != HealthHealthy || !got.UnhealthySince.IsZero() || got.FailuresInWindow != 0 {
		t.Errorf("after success: %+v", got)
	}
}
