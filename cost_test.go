package inferrouter_test

import (
	"context"
	"testing"

	ir "github.com/ineyio/inferrouter"
	"github.com/ineyio/inferrouter/quota"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// pricelessLadder is a single-step ladder whose account bills but publishes no
// rate — the shape of two of our three Gonka gateways.
func pricelessLadder() ir.Config {
	return ir.Config{
		AllowPaid:    true,
		DefaultModel: "ladder",
		Models: []ir.ModelMapping{{Alias: "ladder", Models: []ir.ModelRef{
			{Provider: "priced-out", Model: "ladder-model"},
		}}},
		Accounts: []ir.AccountConfig{
			{Provider: "priced-out", ID: "no-price", QuotaUnit: ir.QuotaTokens, PaidEnabled: true},
		},
	}
}

// L3 (R7): a billable account with no known price starts, routes, and serves.
// Before this it was unusable in two different ways: Validate refused
// paid_enabled without a cost, and dropping paid_enabled handed the account a
// zero daily quota that rejected every reservation — alive in the candidate
// list, dead at Reserve, with nothing in the error to say why.
func TestPaidAccountWithoutCost(t *testing.T) {
	log := &attemptLog{}
	cfg := pricelessLadder()

	r, err := ir.NewRouter(cfg, []ir.Provider{servingStep(log, "priced-out")},
		ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
	require.NoError(t, err, "a price is metadata, not a precondition for starting")

	resp, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})
	require.NoError(t, err)
	assert.Equal(t, []string{"priced-out"}, log.snapshot())
	assert.Equal(t, "no-price", resp.Routing.AccountID)
}

// The same account without paid_enabled is equally usable: no free allowance
// means "limits live at the provider", not "budget of zero".
func TestAccountWithoutQuotaOrPriceStillRoutes(t *testing.T) {
	log := &attemptLog{}
	cfg := pricelessLadder()
	cfg.Accounts[0].PaidEnabled = false

	r, err := ir.NewRouter(cfg, []ir.Provider{servingStep(log, "priced-out")},
		ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
	require.NoError(t, err)

	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})
	require.NoError(t, err, "an account with no local quota must not be unreservable")
	assert.Equal(t, []string{"priced-out"}, log.snapshot())
}

// An unknowable price is accepted, but the consequence is stated rather than
// left for the operator to discover from a spend counter stuck at zero.
func TestConfigWarnings_PricelessAccounts(t *testing.T) {
	t.Run("paid without price: spend not tracked", func(t *testing.T) {
		r, err := ir.NewRouter(pricelessLadder(), []ir.Provider{servingStep(&attemptLog{}, "priced-out")},
			ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
		require.NoError(t, err)

		warnings := r.ConfigWarnings()
		require.Len(t, warnings, 1)
		assert.Contains(t, warnings[0], "no-price")
		assert.Contains(t, warnings[0], "not tracked")
	})

	t.Run("spend cap without price cannot bind", func(t *testing.T) {
		cfg := pricelessLadder()
		cfg.Accounts[0].MaxDailySpend = 10

		r, err := ir.NewRouter(cfg, []ir.Provider{servingStep(&attemptLog{}, "priced-out")},
			ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
		require.NoError(t, err)

		warnings := r.ConfigWarnings()
		require.Len(t, warnings, 1, "a cap that cannot trigger is the warning that matters")
		assert.Contains(t, warnings[0], "max_daily_spend")
		assert.Contains(t, warnings[0], "never trigger")
	})

	t.Run("priced account is quiet", func(t *testing.T) {
		cfg := pricelessLadder()
		cfg.Accounts[0].CostPerInputToken = 0.001
		cfg.Accounts[0].CostPerOutputToken = 0.002
		cfg.Accounts[0].MaxDailySpend = 10

		r, err := ir.NewRouter(cfg, []ir.Provider{servingStep(&attemptLog{}, "priced-out")},
			ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
		require.NoError(t, err)
		assert.Empty(t, r.ConfigWarnings())
	})
}

// A free allowance still produces a real local quota — relaxing the zero-quota
// rule must not relax quota enforcement where a limit was actually declared.
func TestFreeQuotaStillEnforced(t *testing.T) {
	log := &attemptLog{}
	cfg := pricelessLadder()
	cfg.Accounts[0].DailyFree = 1 // one token: the request cannot fit
	cfg.Accounts[0].PaidEnabled = false

	r, err := ir.NewRouter(cfg, []ir.Provider{servingStep(log, "priced-out")},
		ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
	require.NoError(t, err)

	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "a much longer prompt than one token"}},
	})
	require.Error(t, err)
	assert.Empty(t, log.snapshot(), "the declared allowance must still stop the request")
}
