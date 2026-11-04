package inferrouter

import (
	"context"
	"time"
)

// DropReason says why the router would not offer an account as a candidate.
// A half-open account is not dropped: the router offers it, and that call is
// the probe.
type DropReason string

// The reasons an account drops out of a walk.
const (
	DropNone           DropReason = ""
	DropBreakerOpen    DropReason = "breaker_open"
	DropPaidNotAllowed DropReason = "paid_not_allowed"
	DropSpendCap       DropReason = "spend_cap"
)

// dropReason is the one rule for dropping an account, shared by filterCandidates
// and AccountsSnapshot so that what is reported is what the router does.
func dropReason(health HealthState, free, allowPaid bool, maxDailySpend, currentSpend float64) DropReason {
	switch {
	case health == HealthUnhealthy:
		return DropBreakerOpen
	case !free && !allowPaid:
		return DropPaidNotAllowed
	case !free && maxDailySpend > 0 && currentSpend >= maxDailySpend:
		return DropSpendCap
	}
	return DropNone
}

// AccountStatus is the state of one account as the router sees it now.
type AccountStatus struct {
	AccountID string
	Provider  string
	Free      bool

	Health           HealthState
	FailuresInWindow int
	UnhealthySince   time.Time // when the breaker last tripped; zero while Healthy

	CurrentSpend  float64
	MaxDailySpend float64
	Inflight      int64

	// Dropped is why a walk would skip this account. DropNone means it would
	// be offered, half-open included.
	Dropped DropReason
}

// AccountsSnapshot reports every chat account in config order. It changes
// nothing: a half-open transition seen here is not stored, so observing the
// router does not alter what the next request gets.
//
// It answers for the account, not for a request: a multimodal request also
// drops text-only providers, and an RPM limit or a refused reservation skips a
// candidate during the walk. Neither appears here.
func (r *Router) AccountsSnapshot(ctx context.Context) []AccountStatus {
	out := make([]AccountStatus, 0, len(r.cfg.Accounts))
	for _, acc := range r.cfg.Accounts {
		if _, ok := r.providers[acc.Provider]; !ok {
			continue
		}
		_, free := freeQuota(ctx, acc, r.quotaStore)
		health := r.health.Peek(acc.ID)
		spend := r.spend.GetSpend(acc.ID)
		out = append(out, AccountStatus{
			AccountID:        acc.ID,
			Provider:         acc.Provider,
			Free:             free,
			Health:           health.State,
			FailuresInWindow: health.FailuresInWindow,
			UnhealthySince:   health.UnhealthySince,
			CurrentSpend:     spend,
			MaxDailySpend:    acc.MaxDailySpend,
			Inflight:         r.inflight.Get(acc.ID),
			Dropped:          dropReason(health.State, free, r.cfg.AllowPaid, acc.MaxDailySpend, spend),
		})
	}
	return out
}
