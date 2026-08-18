package inferrouter_test

import (
	"context"
	"errors"
	"sync"
	"testing"

	ir "github.com/ineyio/inferrouter"
	"github.com/ineyio/inferrouter/policy"
	"github.com/ineyio/inferrouter/provider/mock"
	"github.com/ineyio/inferrouter/quota"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// attemptLog records the order in which providers were attempted.
type attemptLog struct {
	mu    sync.Mutex
	order []string
}

func (l *attemptLog) record(name string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.order = append(l.order, name)
}

func (l *attemptLog) snapshot() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]string(nil), l.order...)
}

// failingStep is a provider that records the attempt and reports a retryable
// error, so the router moves on to the next step of the ladder.
func failingStep(log *attemptLog, name string) *mock.Provider {
	return mock.New(
		mock.WithName(name),
		mock.WithModels("ladder-model"),
		mock.WithResponseFunc(func(ir.ProviderRequest) (ir.ProviderResponse, error) {
			log.record(name)
			return ir.ProviderResponse{}, ir.ErrProviderUnavailable
		}),
	)
}

// servingStep is a provider that records the attempt and answers successfully.
func servingStep(log *attemptLog, name string) *mock.Provider {
	return mock.New(
		mock.WithName(name),
		mock.WithModels("ladder-model"),
		mock.WithResponseFunc(func(ir.ProviderRequest) (ir.ProviderResponse, error) {
			log.record(name)
			return ir.ProviderResponse{
				Content: "ok",
				Model:   "ladder-model",
				Usage:   ir.Usage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
			}, nil
		}),
	)
}

// ladderConfig declares an alias whose steps are listed in the given order.
// The LAST account is the free one: under free-first ordering it would jump to
// the head of the queue, which is exactly what must not happen.
//
// Paid accounts carry PaidEnabled + a cost because today's library refuses to
// start otherwise, and a paid account without PaidEnabled gets a zero daily
// quota and dies at Reserve (RFC §3.2). Step 3 lifts both constraints; this
// helper gets simpler then.
func ladderConfig(stepNames ...string) ir.Config {
	accounts := make([]ir.AccountConfig, 0, len(stepNames))
	refs := make([]ir.ModelRef, 0, len(stepNames))
	for i, name := range stepNames {
		acc := ir.AccountConfig{
			Provider:  name,
			ID:        name + "-acc",
			QuotaUnit: ir.QuotaRequests,
		}
		if i == len(stepNames)-1 {
			acc.DailyFree = 1000
		} else {
			acc.PaidEnabled = true
			acc.CostPerInputToken = 0.001
			acc.CostPerOutputToken = 0.001
		}
		accounts = append(accounts, acc)
		refs = append(refs, ir.ModelRef{Provider: name, Model: "ladder-model"})
	}

	return ir.Config{
		AllowPaid: true,
		Models:    []ir.ModelMapping{{Alias: "ladder", Models: refs}},
		Accounts:  accounts,
	}
}

// L1: without an explicit policy the attempt order IS the config order.
// Mutation check: restoring a free-first default (or any policy) makes the
// free last step overtake the paid ones and this test fails.
func TestRouter_DefaultOrderFollowsConfig(t *testing.T) {
	log := &attemptLog{}
	first := failingStep(log, "step-one")
	second := failingStep(log, "step-two")
	third := servingStep(log, "step-three") // the free one

	cfg := ladderConfig("step-one", "step-two", "step-three")
	r, err := ir.NewRouter(cfg, []ir.Provider{first, second, third},
		ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
	require.NoError(t, err)

	resp, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
		Model:    "ladder",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})
	require.NoError(t, err)

	assert.Equal(t, []string{"step-one", "step-two", "step-three"}, log.snapshot(),
		"attempts must follow the declared ladder order, not free/paid priority")
	assert.Equal(t, "step-three", resp.Routing.Provider)
	assert.Equal(t, 3, resp.Routing.Attempts)
}

// The declared order is a property of the alias, not of the account list:
// reversing the steps reverses the attempts while cfg.Accounts stays put.
func TestRouter_OrderFollowsAliasNotAccountList(t *testing.T) {
	log := &attemptLog{}
	a := servingStep(log, "alpha")
	b := failingStep(log, "beta")

	cfg := ladderConfig("alpha", "beta")
	// Ladder says beta first, alpha second — the opposite of cfg.Accounts.
	cfg.Models = []ir.ModelMapping{{Alias: "ladder", Models: []ir.ModelRef{
		{Provider: "beta", Model: "ladder-model"},
		{Provider: "alpha", Model: "ladder-model"},
	}}}

	r, err := ir.NewRouter(cfg, []ir.Provider{a, b},
		ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
	require.NoError(t, err)

	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{
		Model:    "ladder",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})
	require.NoError(t, err)

	assert.Equal(t, []string{"beta", "alpha"}, log.snapshot())
}

// Two accounts of one provider inside a single step are attempted in
// cfg.Accounts order (qarap's free/paid Cerebras keys shape).
func TestRouter_AccountsWithinStepFollowConfigOrder(t *testing.T) {
	var served []string
	var mu sync.Mutex
	prov := mock.New(
		mock.WithName("dual"),
		mock.WithModels("ladder-model"),
		mock.WithResponseFunc(func(req ir.ProviderRequest) (ir.ProviderResponse, error) {
			mu.Lock()
			served = append(served, req.Auth.APIKey)
			mu.Unlock()
			return ir.ProviderResponse{}, ir.ErrProviderUnavailable
		}),
	)

	cfg := ir.Config{
		AllowPaid: true,
		Models: []ir.ModelMapping{{Alias: "ladder", Models: []ir.ModelRef{
			{Provider: "dual", Model: "ladder-model"},
		}}},
		Accounts: []ir.AccountConfig{
			{Provider: "dual", ID: "paid-key", QuotaUnit: ir.QuotaRequests,
				Auth: ir.Auth{APIKey: "paid"}, PaidEnabled: true, CostPerInputToken: 0.001},
			{Provider: "dual", ID: "free-key", QuotaUnit: ir.QuotaRequests,
				Auth: ir.Auth{APIKey: "free"}, DailyFree: 1000},
		},
	}

	r, err := ir.NewRouter(cfg, []ir.Provider{prov},
		ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
	require.NoError(t, err)

	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{
		Model:    "ladder",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})
	require.Error(t, err)

	mu.Lock()
	defer mu.Unlock()
	assert.Equal(t, []string{"paid", "free"}, served,
		"accounts within one step keep cfg.Accounts order regardless of free/paid")
}

// A policy is still honoured when asked for explicitly — reordering did not
// disappear, it stopped being the default.
func TestRouter_ExplicitPolicyStillReorders(t *testing.T) {
	log := &attemptLog{}
	paid := failingStep(log, "paid-step")
	free := servingStep(log, "free-step")

	cfg := ladderConfig("paid-step", "free-step")
	r, err := ir.NewRouter(cfg, []ir.Provider{paid, free},
		ir.WithQuotaStore(quota.NewMemoryQuotaStore()),
		ir.WithPolicy(&policy.FreeFirstPolicy{}))
	require.NoError(t, err)

	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{
		Model:    "ladder",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})
	require.NoError(t, err)

	assert.Equal(t, []string{"free-step"}, log.snapshot(),
		"FreeFirstPolicy must still pull the free step to the front")
}

// L5 (R8): a step that cannot serve the requested modality is skipped like an
// unhealthy one — it must not fail the whole ladder, and it must not shift the
// remaining steps out of their declared order.
func TestRouter_MultimodalStepSkipped(t *testing.T) {
	log := &attemptLog{}
	textOnly := failingStep(log, "text-step") // would fail if ever called
	capable := mock.New(
		mock.WithName("vision-step"),
		mock.WithModels("ladder-model"),
		mock.WithMultimodal(true),
		mock.WithResponseFunc(func(ir.ProviderRequest) (ir.ProviderResponse, error) {
			log.record("vision-step")
			return ir.ProviderResponse{Content: "ok", Model: "ladder-model"}, nil
		}),
	)

	cfg := ladderConfig("text-step", "vision-step")
	r, err := ir.NewRouter(cfg, []ir.Provider{textOnly, capable},
		ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
	require.NoError(t, err)

	resp, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
		Model: "ladder",
		Messages: []ir.Message{{Role: "user", Parts: []ir.Part{
			{Type: ir.PartText, Text: "describe"},
			{Type: ir.PartImage, MIMEType: "image/jpeg", Data: []byte{1, 2, 3}},
		}}},
	})
	require.NoError(t, err)

	assert.Equal(t, []string{"vision-step"}, log.snapshot(),
		"the text-only step is skipped, not attempted and not fatal")
	assert.Equal(t, "vision-step", resp.Routing.Provider)
}

// No step can serve the modality → the specific sentinel, not ErrNoCandidates.
func TestRouter_MultimodalNoCapableStep(t *testing.T) {
	log := &attemptLog{}
	cfg := ladderConfig("text-one", "text-two")
	r, err := ir.NewRouter(cfg,
		[]ir.Provider{failingStep(log, "text-one"), failingStep(log, "text-two")},
		ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
	require.NoError(t, err)

	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{
		Model: "ladder",
		Messages: []ir.Message{{Role: "user", Parts: []ir.Part{
			{Type: ir.PartImage, MIMEType: "image/jpeg", Data: []byte{1}},
		}}},
	})

	assert.True(t, errors.Is(err, ir.ErrMultimodalUnavailable), "got %v", err)
	assert.Empty(t, log.snapshot(), "no step may be attempted")
}
