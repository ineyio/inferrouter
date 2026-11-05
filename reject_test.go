package inferrouter_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"sync"
	"testing"
	"time"

	ir "github.com/ineyio/inferrouter"
	"github.com/ineyio/inferrouter/provider/mock"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var errUnfit = errors.New("cut off without JSON")

func rejectEverything(ir.ChatResponse) error { return errUnfit }

var hello = []ir.Message{{Role: "user", Content: "hi"}}

// quotaSpy records what the router did with each reservation.
type quotaSpy struct {
	mu        sync.Mutex
	committed []int64
	rolled    int
}

func (q *quotaSpy) Reserve(_ context.Context, id string, amount int64, unit ir.QuotaUnit, _ string) (ir.Reservation, error) {
	return ir.Reservation{ID: "r", AccountID: id, Amount: amount, Unit: unit}, nil
}
func (q *quotaSpy) Commit(_ context.Context, _ ir.Reservation, actual int64) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.committed = append(q.committed, actual)
	return nil
}
func (q *quotaSpy) Rollback(context.Context, ir.Reservation) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.rolled++
	return nil
}
func (q *quotaSpy) Remaining(context.Context, string) (int64, error) { return 0, nil }

// resultLog keeps every ResultEvent; the spy in router_test keeps only the last.
type resultLog struct {
	mu     sync.Mutex
	events []ir.ResultEvent
}

func (m *resultLog) OnRoute(ir.RouteEvent) {}
func (m *resultLog) OnResult(e ir.ResultEvent) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.events = append(m.events, e)
}

func answering(log *attemptLog, name, content string) *mock.Provider {
	return mock.New(mock.WithName(name), mock.WithModels("ladder-model"),
		mock.WithResponseFunc(func(ir.ProviderRequest) (ir.ProviderResponse, error) {
			log.record(name)
			return ir.ProviderResponse{
				ID: "id-" + name, Content: content, FinishReason: "length", Model: "ladder-model",
				Usage: ir.Usage{PromptTokens: 100, CompletionTokens: 40, ReasoningTokens: 10, TotalTokens: 150},
			}, nil
		}))
}

func ladderOf(names ...string) ir.Config {
	cfg := ir.Config{AllowPaid: true, DefaultModel: "ladder", Models: []ir.ModelMapping{{Alias: "ladder"}}}
	for _, name := range names {
		cfg.Models[0].Models = append(cfg.Models[0].Models, ir.ModelRef{Provider: name, Model: "ladder-model"})
		cfg.Accounts = append(cfg.Accounts, ir.AccountConfig{
			Provider: name, ID: name, QuotaUnit: ir.QuotaTokens, PaidEnabled: true,
			CostPerInputToken: 0.01, CostPerOutputToken: 0.1,
		})
	}
	return cfg
}

func TestRejected_WalkGoesToTheNextStepInOrder(t *testing.T) {
	log := &attemptLog{}
	r, err := ir.NewRouter(ladderOf("first", "second", "third"), []ir.Provider{
		answering(log, "first", "bad"), answering(log, "second", "good"), answering(log, "third", "never"),
	})
	require.NoError(t, err)

	var saw []string
	resp, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: hello,
		ValidateResponse: func(c ir.ChatResponse) error {
			saw = append(saw, c.Choices[0].Message.Content)
			if c.Choices[0].Message.Content == "bad" {
				return errUnfit
			}
			return nil
		},
	})
	require.NoError(t, err)

	assert.Equal(t, []string{"first", "second"}, log.snapshot())
	assert.Equal(t, []string{"bad", "good"}, saw)
	assert.Equal(t, "good", resp.Choices[0].Message.Content)
	assert.Equal(t, "second", resp.Routing.Provider)
	assert.Equal(t, 2, resp.Routing.Attempts)
}

func TestRejected_FunctionSeesTheWholeAnswer(t *testing.T) {
	log := &attemptLog{}
	r, err := ir.NewRouter(ladderOf("only"), []ir.Provider{answering(log, "only", "text")})
	require.NoError(t, err)

	var got ir.ChatResponse
	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: hello,
		ValidateResponse: func(c ir.ChatResponse) error {
			got = c
			return nil
		},
	})
	require.NoError(t, err)

	assert.Equal(t, "length", got.Choices[0].FinishReason)
	assert.Equal(t, int64(150), got.Usage.TotalTokens)
	assert.Equal(t, "only", got.Routing.Provider)
	assert.Equal(t, 1, got.Routing.Attempts)
}

func TestRejected_TokensCountAsSpent(t *testing.T) {
	log := &attemptLog{}
	quotas := &quotaSpy{}
	meter := &resultLog{}
	spend := ir.NewSpendTracker()
	r, err := ir.NewRouter(ladderOf("first", "second"), []ir.Provider{
		answering(log, "first", "bad"), answering(log, "second", "bad"),
	}, ir.WithQuotaStore(quotas), ir.WithMeter(meter), ir.WithSpendTracker(spend))
	require.NoError(t, err)

	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{Messages: hello, ValidateResponse: rejectEverything})
	require.Error(t, err)

	assert.Equal(t, []int64{150, 150}, quotas.committed, "actual usage, once per attempt")
	assert.Zero(t, quotas.rolled, "the tokens were spent")

	const cost = 100*0.01 + (40+10)*0.1
	assert.InDelta(t, cost, spend.GetSpend("first"), 1e-9)
	assert.InDelta(t, cost, spend.GetSpend("second"), 1e-9)

	require.Len(t, meter.events, 2)
	for _, e := range meter.events {
		assert.False(t, e.Success, "a meter that does not know Rejected must read a failure")
		assert.True(t, e.Rejected)
		assert.Equal(t, int64(150), e.Usage.TotalTokens)
		assert.InDelta(t, cost, e.DollarCost, 1e-9)
		assert.ErrorIs(t, e.Error, ir.ErrResponseRejected)
		assert.ErrorIs(t, e.Error, errUnfit)
	}
}

func TestRejected_OtherOutcomesAreNotMarked(t *testing.T) {
	log := &attemptLog{}
	meter := &resultLog{}
	r, err := ir.NewRouter(ladderOf("down", "up"), []ir.Provider{
		failingStep(log, "down"), answering(log, "up", "good"),
	}, ir.WithMeter(meter))
	require.NoError(t, err)

	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: hello, ValidateResponse: func(ir.ChatResponse) error { return nil },
	})
	require.NoError(t, err)

	require.Len(t, meter.events, 2)
	assert.False(t, meter.events[0].Rejected)
	assert.False(t, meter.events[0].Success)
	assert.False(t, meter.events[1].Rejected)
	assert.True(t, meter.events[1].Success)
}

func TestRejected_NeverTripsTheBreaker(t *testing.T) {
	log := &attemptLog{}
	r, err := ir.NewRouter(ladderOf("only"), []ir.Provider{answering(log, "only", "bad")})
	require.NoError(t, err)

	for range 10 {
		_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{Messages: hello, ValidateResponse: rejectEverything})
		require.ErrorIs(t, err, ir.ErrAllFailed)
	}

	status := r.AccountsSnapshot(context.Background())[0]
	assert.Equal(t, ir.HealthHealthy, status.Health)
	assert.Zero(t, status.FailuresInWindow)
	assert.Len(t, log.snapshot(), 10, "every call still reaches the step")
}

func TestRejected_DoesNotWipeTheFailureCount(t *testing.T) {
	fail := true
	p := mock.New(mock.WithName("only"), mock.WithModels("ladder-model"),
		mock.WithResponseFunc(func(ir.ProviderRequest) (ir.ProviderResponse, error) {
			if fail {
				return ir.ProviderResponse{}, ir.ErrProviderUnavailable
			}
			return ir.ProviderResponse{Content: "bad"}, nil
		}))
	r, err := ir.NewRouter(ladderOf("only"), []ir.Provider{p})
	require.NoError(t, err)

	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{Messages: hello})
	require.Error(t, err)
	fail = false
	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{Messages: hello, ValidateResponse: rejectEverything})
	require.ErrorIs(t, err, ir.ErrAllFailed)

	assert.Equal(t, 1, r.AccountsSnapshot(context.Background())[0].FailuresInWindow)
}

func TestRejected_DoesNotCloseAnOpenBreaker(t *testing.T) {
	fail := true
	p := mock.New(mock.WithName("only"), mock.WithModels("ladder-model"),
		mock.WithResponseFunc(func(ir.ProviderRequest) (ir.ProviderResponse, error) {
			if fail {
				return ir.ProviderResponse{}, ir.ErrProviderUnavailable
			}
			return ir.ProviderResponse{Content: "bad"}, nil
		}))
	r, err := ir.NewRouter(ladderOf("only"), []ir.Provider{p}, ir.WithHealthConfig(ir.HealthConfig{
		FailureThreshold: 1, FailureWindow: time.Minute, UnhealthyPeriod: time.Millisecond,
	}))
	require.NoError(t, err)

	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{Messages: hello})
	require.Error(t, err)
	time.Sleep(5 * time.Millisecond)
	require.Equal(t, ir.HealthHalfOpen, r.AccountsSnapshot(context.Background())[0].Health)

	fail = false
	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{Messages: hello, ValidateResponse: rejectEverything})
	require.ErrorIs(t, err, ir.ErrAllFailed, "half-open is offered, so the step was attempted")

	assert.Equal(t, ir.HealthHalfOpen, r.AccountsSnapshot(context.Background())[0].Health)
}

func TestRejected_AllStepsGiveTheReasons(t *testing.T) {
	log := &attemptLog{}
	r, err := ir.NewRouter(ladderOf("first", "second"), []ir.Provider{
		answering(log, "first", "bad"), answering(log, "second", "bad"),
	})
	require.NoError(t, err)

	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{Messages: hello, ValidateResponse: rejectEverything})
	require.ErrorIs(t, err, ir.ErrAllFailed)

	var routerErr *ir.RouterError
	require.ErrorAs(t, err, &routerErr)
	require.Len(t, routerErr.Tried, 2)
	for i, want := range []string{"first", "second"} {
		assert.Equal(t, want, routerErr.Tried[i].AccountID)
		assert.ErrorIs(t, routerErr.Tried[i].Err, ir.ErrResponseRejected)
		assert.ErrorIs(t, routerErr.Tried[i].Err, errUnfit)
		assert.True(t, ir.IsRetryable(routerErr.Tried[i].Err))
		assert.False(t, ir.IsFatal(routerErr.Tried[i].Err))
	}
	assert.Contains(t, err.Error(), errUnfit.Error())
}

func TestRejected_ContextEndKeepsTheAttempts(t *testing.T) {
	log := &attemptLog{}
	r, err := ir.NewRouter(ladderOf("first", "second"), []ir.Provider{
		answering(log, "first", "bad"), answering(log, "second", "never"),
	})
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	_, err = r.ChatCompletion(ctx, ir.ChatRequest{
		Messages: hello,
		ValidateResponse: func(ir.ChatResponse) error {
			cancel()
			return errUnfit
		},
	})

	require.ErrorIs(t, err, context.Canceled)
	assert.NotErrorIs(t, err, ir.ErrAllFailed)
	assert.NotContains(t, err.Error(), "all ", "the walk stopped, the candidates did not all fail")
	assert.Equal(t, []string{"first"}, log.snapshot())

	var routerErr *ir.RouterError
	require.ErrorAs(t, err, &routerErr)
	require.Len(t, routerErr.Tried, 1)
	assert.ErrorIs(t, routerErr.Tried[0].Err, errUnfit)
	assert.Equal(t, 1, routerErr.Attempts)
}

func TestWalk_ContextEndBeforeAnyAttemptIsTheBareError(t *testing.T) {
	log := &attemptLog{}
	r, err := ir.NewRouter(ladderOf("first"), []ir.Provider{answering(log, "first", "x")})
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = r.ChatCompletion(ctx, ir.ChatRequest{Messages: hello})
	assert.Equal(t, context.Canceled, err)

	_, err = r.ChatCompletionStream(ctx, ir.ChatRequest{Messages: hello})
	assert.Equal(t, context.Canceled, err)
}

func TestStream_ContextEndKeepsTheFailedAttempts(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	down := mock.New(mock.WithName("down"), mock.WithModels("ladder-model"), mock.WithError(ir.ErrProviderUnavailable))
	cancelling := &cancelOnStream{Provider: down, cancel: cancel}
	log := &attemptLog{}
	r, err := ir.NewRouter(ladderOf("down", "second"), []ir.Provider{cancelling, answering(log, "second", "never")})
	require.NoError(t, err)

	_, err = r.ChatCompletionStream(ctx, ir.ChatRequest{Messages: hello})

	require.ErrorIs(t, err, context.Canceled)
	var routerErr *ir.RouterError
	require.ErrorAs(t, err, &routerErr)
	require.Len(t, routerErr.Tried, 1)
	assert.ErrorIs(t, routerErr.Tried[0].Err, ir.ErrProviderUnavailable)
}

// cancelOnStream fails to open and ends the caller's context on the way out.
type cancelOnStream struct {
	*mock.Provider
	cancel context.CancelFunc
}

func (p *cancelOnStream) ChatCompletionStream(context.Context, ir.ProviderRequest) (ir.ProviderStream, error) {
	p.cancel()
	return nil, ir.ErrProviderUnavailable
}

func TestStream_RefusesAValidator(t *testing.T) {
	log := &attemptLog{}
	p := answering(log, "only", "x")
	r, err := ir.NewRouter(ladderOf("only"), []ir.Provider{p})
	require.NoError(t, err)

	stream, err := r.ChatCompletionStream(context.Background(), ir.ChatRequest{Messages: hello, ValidateResponse: rejectEverything})

	assert.Nil(t, stream)
	assert.ErrorIs(t, err, ir.ErrInvalidRequest)
	assert.Zero(t, p.CallCount(), "refused before any step is attempted")
}

func TestStream_WithoutValidatorStillOpens(t *testing.T) {
	log := &attemptLog{}
	r, err := ir.NewRouter(ladderOf("only"), []ir.Provider{answering(log, "only", "x")})
	require.NoError(t, err)

	stream, err := r.ChatCompletionStream(context.Background(), ir.ChatRequest{Messages: hello})
	require.NoError(t, err)
	defer stream.Close()
	_, err = stream.Next()
	if err != nil {
		assert.ErrorIs(t, err, io.EOF)
	}
}

func TestChatRequest_ValidatorDoesNotBreakMarshalling(t *testing.T) {
	data, err := json.Marshal(ir.ChatRequest{Model: "m", ValidateResponse: rejectEverything})
	require.NoError(t, err)
	assert.NotContains(t, string(data), "Validate")
}

// commitFailingQuota refuses every commit, as a store that lost its connection
// after the reservation would.
type commitFailingQuota struct{ quotaSpy }

func (*commitFailingQuota) Commit(context.Context, ir.Reservation, int64) error {
	return errors.New("store gone")
}

func TestRejected_QuotaCommitFailureIsVisibleAndTheWalkGoesOn(t *testing.T) {
	log := &attemptLog{}
	meter := &resultLog{}
	r, err := ir.NewRouter(ladderOf("first", "second"), []ir.Provider{
		answering(log, "first", "bad"), answering(log, "second", "bad"),
	}, ir.WithQuotaStore(&commitFailingQuota{}), ir.WithMeter(meter))
	require.NoError(t, err)

	_, err = r.ChatCompletion(context.Background(), ir.ChatRequest{Messages: hello, ValidateResponse: rejectEverything})
	require.ErrorIs(t, err, ir.ErrAllFailed)

	assert.Equal(t, []string{"first", "second"}, log.snapshot())
	require.Len(t, meter.events, 2)
	assert.ErrorIs(t, meter.events[0].Error, ir.ErrResponseRejected)
	assert.Contains(t, meter.events[0].Error.Error(), "store gone")
}
