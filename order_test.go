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
// The paid steps declare no price — a billable account without a published
// rate is a supported state, and one of the shapes this ordering has to work
// for.
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

// declareLadder gives a pre-strict-resolution test config the alias its
// DefaultModel now needs: one step per distinct provider, in account order.
// Resolution is strict (R2), so every request must name a declared ladder;
// these tests predate that, and their intent — "route this model across these
// accounts" — is exactly a ladder listing each account's provider.
func declareLadder(cfg ir.Config) ir.Config {
	if len(cfg.Models) > 0 || cfg.DefaultModel == "" {
		return cfg
	}
	seen := map[string]bool{}
	var refs []ir.ModelRef
	for _, acc := range cfg.Accounts {
		if seen[acc.Provider] {
			continue
		}
		seen[acc.Provider] = true
		refs = append(refs, ir.ModelRef{Provider: acc.Provider, Model: cfg.DefaultModel})
	}
	cfg.Models = []ir.ModelMapping{{Alias: cfg.DefaultModel, Models: refs}}
	return cfg
}

// rejectingStep is a provider that records the attempt and refuses the request
// outright (HTTP 400 at the wire), the way a reseller does when the model the
// step is configured with has left its line-up.
func rejectingStep(log *attemptLog, name string) *mock.Provider {
	return mock.New(
		mock.WithName(name),
		mock.WithModels("ladder-model"),
		mock.WithResponseFunc(func(ir.ProviderRequest) (ir.ProviderResponse, error) {
			log.record(name)
			return ir.ProviderResponse{}, ir.ErrInvalidRequest
		}),
	)
}

// A step that rejects the request does not end the walk: the refusal belongs to
// that gateway, and the next step is asked the same question.
//
// Live case: proxy.gonka.gg answered "unsupported model" for the model its step
// names while four steps behind it were healthy; every chunk that hit it died
// with attempts=1 (qarap's rerun, pjob_16347b80c56441d6, 2026-09-04).
//
// Mutation: put ErrInvalidRequest back into IsFatal. The walk then stops at
// step-one and this goes red.
func TestRouter_InvalidRequestOnOneStepDoesNotEndTheWalk(t *testing.T) {
	log := &attemptLog{}
	cfg := ladderConfig("step-one", "step-two")
	r, err := ir.NewRouter(cfg,
		[]ir.Provider{rejectingStep(log, "step-one"), servingStep(log, "step-two")},
		ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
	require.NoError(t, err)

	resp, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
		Model:    "ladder",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})
	require.NoError(t, err, "the second step served, so the caller gets an answer")
	assert.Equal(t, "step-two", resp.Routing.Provider)
	assert.Equal(t, 2, resp.Routing.Attempts, "the refusal is counted as an attempt, not hidden")
	assert.Equal(t, []string{"step-one", "step-two"}, log.snapshot())
}

// The other half: when every step rejects, the caller still learns that every
// step was asked — ErrAllFailed with each refusal listed, not the first 400
// dressed up as the ladder's verdict.
func TestRouter_InvalidRequestOnEveryStepReportsAllTried(t *testing.T) {
	log := &attemptLog{}
	cfg := ladderConfig("step-one", "step-two")
	r, err := ir.NewRouter(cfg,
		[]ir.Provider{rejectingStep(log, "step-one"), rejectingStep(log, "step-two")},
		ir.WithQuotaStore(quota.NewMemoryQuotaStore()))
	require.NoError(t, err)

	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{
		Model:    "ladder",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})
	require.Error(t, err)
	var routerErr *ir.RouterError
	require.True(t, errors.As(err, &routerErr), "err must carry RouterError: %v", err)
	assert.True(t, errors.Is(routerErr.Unwrap(), ir.ErrAllFailed))
	assert.Len(t, routerErr.Tried, 2)
	for _, ce := range routerErr.Tried {
		assert.True(t, errors.Is(ce.Err, ir.ErrInvalidRequest), "each refusal keeps its own cause: %v", ce.Err)
	}
	assert.Equal(t, []string{"step-one", "step-two"}, log.snapshot())
}
