package inferrouter_test

import (
	"context"
	"errors"
	"io"
	"sync"
	"testing"
	"time"

	ir "github.com/ineyio/inferrouter"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// scriptedStep is a step whose every behaviour the test writes out, context
// included: mock.Provider does not look at its context once a stream is open.
type scriptedStep struct {
	name   string
	calls  int
	mu     sync.Mutex
	chat   func(ctx context.Context) (ir.ProviderResponse, error)
	stream func(ctx context.Context) (ir.ProviderStream, error)
	embed  func(ctx context.Context) (ir.EmbedProviderResponse, error)
}

func (p *scriptedStep) Name() string              { return p.name }
func (p *scriptedStep) SupportsModel(string) bool { return true }
func (p *scriptedStep) SupportsMultimodal() bool  { return false }
func (p *scriptedStep) count() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.calls++
}
func (p *scriptedStep) callCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.calls
}
func (p *scriptedStep) ChatCompletion(ctx context.Context, _ ir.ProviderRequest) (ir.ProviderResponse, error) {
	p.count()
	return p.chat(ctx)
}
func (p *scriptedStep) ChatCompletionStream(ctx context.Context, _ ir.ProviderRequest) (ir.ProviderStream, error) {
	p.count()
	return p.stream(ctx)
}
func (p *scriptedStep) SupportsEmbeddingModel(string) bool { return true }
func (p *scriptedStep) MaxBatchSize() int                  { return 100 }
func (p *scriptedStep) Embed(ctx context.Context, _ ir.EmbedProviderRequest) (ir.EmbedProviderResponse, error) {
	p.count()
	return p.embed(ctx)
}

// hangs answers only when its context ends, as a hung gateway does.
func hangs(ctx context.Context) (ir.ProviderResponse, error) {
	<-ctx.Done()
	return ir.ProviderResponse{}, ctx.Err()
}

func answers(context.Context) (ir.ProviderResponse, error) {
	return ir.ProviderResponse{Content: "ok", Usage: ir.Usage{TotalTokens: 7}}, nil
}

func refuses(context.Context) (ir.ProviderResponse, error) {
	return ir.ProviderResponse{}, ir.ErrProviderUnavailable
}

// stepRouter is one account with the given attempt budget and a threshold high
// enough that the count itself can be read.
func stepRouter(t *testing.T, budget time.Duration, step *scriptedStep, opts ...ir.Option) *ir.Router {
	t.Helper()
	cfg := ladderOf(step.name)
	cfg.AttemptTimeout = budget
	r, err := ir.NewRouter(cfg, []ir.Provider{step}, opts...)
	require.NoError(t, err)
	return r
}

func failuresOf(r *ir.Router) int {
	return r.AccountsSnapshot(context.Background())[0].FailuresInWindow
}

func healthOf(r *ir.Router) ir.HealthState {
	return r.AccountsSnapshot(context.Background())[0].Health
}

func chatOnce(ctx context.Context, r *ir.Router) error {
	_, err := r.ChatCompletion(ctx, ir.ChatRequest{Messages: hello})
	return err
}

// oneRealFailure leaves the account with one counted failure, so a later success
// would show by wiping it.
func oneRealFailure(t *testing.T, step *scriptedStep, r *ir.Router) {
	t.Helper()
	was := step.chat
	step.chat = refuses
	require.Error(t, chatOnce(context.Background(), r))
	step.chat = was
	require.Equal(t, 1, failuresOf(r))
}

// CE-I1: the caller going away says nothing about the gateway, whatever the
// attempt budget, and that is neither a failure nor a success. Three of them
// were enough to open the breaker on a healthy gateway (RISK-177).
func TestCallerEnd_CancelIsNeverCounted(t *testing.T) {
	for _, budget := range []time.Duration{0, time.Hour} {
		t.Run(budget.String(), func(t *testing.T) {
			step := &scriptedStep{name: "only", chat: hangs}
			r := stepRouter(t, budget, step)
			oneRealFailure(t, step, r)

			for i := 0; i < 3; i++ {
				ctx, cancel := context.WithCancel(context.Background())
				step.chat = func(ctx context.Context) (ir.ProviderResponse, error) {
					cancel()
					return hangs(ctx)
				}
				require.Error(t, chatOnce(ctx, r))
			}

			assert.Equal(t, 1, failuresOf(r), "not a failure, and not a success that wipes the count")
			assert.Equal(t, ir.HealthHealthy, healthOf(r))
		})
	}
}

// CE-I2: with an attempt budget a hung gateway has another way to show itself,
// so the caller's own deadline is not counted. Without one nothing else would
// ever see the hang, so it is (owner's decision, 2026-11-06).
func TestCallerEnd_DeadlineFollowsTheBudget(t *testing.T) {
	for _, tc := range []struct {
		name    string
		global  time.Duration
		account time.Duration
		want    int
	}{
		{"no budget counts", 0, 0, 1},
		{"global budget does not", time.Hour, 0, 0},
		{"account budget does not", 0, time.Hour, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			step := &scriptedStep{name: "only", chat: hangs}
			cfg := ladderOf(step.name)
			cfg.AttemptTimeout = tc.global
			cfg.Accounts[0].AttemptTimeout = tc.account
			r, err := ir.NewRouter(cfg, []ir.Provider{step})
			require.NoError(t, err)

			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
			defer cancel()
			require.Error(t, chatOnce(ctx, r))

			assert.Equal(t, tc.want, failuresOf(r))
		})
	}
}

// CE-I3: while the caller is alive every failure counts. The stream's watchdog is
// the case that a test of the error alone would get wrong: it ends the attempt
// with context.Canceled, and that is a hung gateway.
func TestCallerEnd_AttemptTimeoutCounts(t *testing.T) {
	t.Run("attempt timeout", func(t *testing.T) {
		step := &scriptedStep{name: "only", chat: hangs}
		r := stepRouter(t, 5*time.Millisecond, step)
		ctx, cancel := context.WithTimeout(context.Background(), time.Hour)
		defer cancel()
		require.Error(t, chatOnce(ctx, r))
		assert.Equal(t, 1, failuresOf(r))
	})

	t.Run("gateway refusal", func(t *testing.T) {
		step := &scriptedStep{name: "only", chat: refuses}
		r := stepRouter(t, time.Hour, step)
		require.Error(t, chatOnce(context.Background(), r))
		assert.Equal(t, 1, failuresOf(r))
	})

	t.Run("a context error from the gateway with the caller alive", func(t *testing.T) {
		step := &scriptedStep{name: "only", chat: func(context.Context) (ir.ProviderResponse, error) {
			return ir.ProviderResponse{}, context.Canceled
		}}
		r := stepRouter(t, 0, step)
		require.Error(t, chatOnce(context.Background(), r))
		assert.Equal(t, 1, failuresOf(r), "only the caller's own context decides")
	})

	t.Run("stream watchdog", func(t *testing.T) {
		step := &scriptedStep{name: "only", stream: func(ctx context.Context) (ir.ProviderStream, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		}}
		r := stepRouter(t, 5*time.Millisecond, step)
		_, err := r.ChatCompletionStream(context.Background(), ir.ChatRequest{Messages: hello})
		require.Error(t, err)
		assert.Equal(t, 1, failuresOf(r))
	})
}

// CE-I3: the watchdog fired, the adapter opened the stream all the same, and the
// body dies on the attempt's context. The caller is alive, so the gateway answers
// for it: the stream must hold the caller's context, not the attempt's.
func TestCallerEnd_AttemptTimeoutCounts_LateOpening(t *testing.T) {
	step := &scriptedStep{name: "only", stream: func(ctx context.Context) (ir.ProviderStream, error) {
		<-ctx.Done()
		return &scriptedStream{ctx: ctx, end: contextEnd}, nil
	}}
	r := stepRouter(t, 5*time.Millisecond, step)
	s, err := r.ChatCompletionStream(context.Background(), ir.ChatRequest{Messages: hello})
	require.NoError(t, err)
	require.Error(t, drain(s))
	_ = s.Close()
	assert.Equal(t, 1, failuresOf(r))
}

// scriptedStream gives chunks and then the error its script names, reading the
// context at that moment as a real body does.
type scriptedStream struct {
	ctx    context.Context
	chunks int
	end    func(ctx context.Context) error
}

func (s *scriptedStream) Next() (ir.StreamChunk, error) {
	if s.chunks > 0 {
		s.chunks--
		return ir.StreamChunk{Usage: &ir.Usage{TotalTokens: 3}}, nil
	}
	return ir.StreamChunk{}, s.end(s.ctx)
}
func (s *scriptedStream) Close() error { return nil }

func streamEnding(end func(ctx context.Context) error, chunks int) func(ctx context.Context) (ir.ProviderStream, error) {
	return func(ctx context.Context) (ir.ProviderStream, error) {
		return &scriptedStream{ctx: ctx, chunks: chunks, end: end}, nil
	}
}

func contextEnd(ctx context.Context) error {
	<-ctx.Done()
	return ctx.Err()
}

// eofOnContextEnd is what openaicompat's sseStream does: any read error is EOF.
func eofOnContextEnd(ctx context.Context) error {
	<-ctx.Done()
	return io.EOF
}

func drain(s *ir.RouterStream) error {
	for {
		if _, err := s.Next(); err != nil {
			return err
		}
	}
}

// CE-I4: after opening, the attempt budget no longer runs, and the rule is the
// same one: a long answer on a short budget is a healthy gateway.
func TestCallerEnd_Stream(t *testing.T) {
	open := func(t *testing.T, budget time.Duration, end func(context.Context) error, ctx context.Context) (*ir.Router, *scriptedStep, *ir.RouterStream) {
		t.Helper()
		step := &scriptedStep{name: "only", chat: refuses, stream: streamEnding(end, 1)}
		r := stepRouter(t, budget, step)
		require.Error(t, chatOnce(context.Background(), r))
		require.Equal(t, 1, failuresOf(r))
		s, err := r.ChatCompletionStream(ctx, ir.ChatRequest{Messages: hello})
		require.NoError(t, err)
		return r, step, s
	}

	for _, tc := range []struct {
		name string
		end  func(context.Context) error
	}{{"context error", contextEnd}, {"swallowed as EOF", eofOnContextEnd}} {
		t.Run("cancel, "+tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			r, _, s := open(t, 0, tc.end, ctx)
			_, err := s.Next()
			require.NoError(t, err)
			cancel()
			require.Error(t, drain(s))
			_ = s.Close()
			assert.Equal(t, 1, failuresOf(r), "neither a failure nor a success")
		})

		t.Run("deadline with a budget, "+tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
			defer cancel()
			r, _, s := open(t, time.Hour, tc.end, ctx)
			require.Error(t, drain(s))
			_ = s.Close()
			assert.Equal(t, 1, failuresOf(r))
		})

		t.Run("deadline without a budget, "+tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
			defer cancel()
			r, _, s := open(t, 0, tc.end, ctx)
			require.Error(t, drain(s))
			_ = s.Close()
			assert.Equal(t, 2, failuresOf(r))
		})
	}

	t.Run("closed without EOF after the caller left", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		r, _, s := open(t, 0, func(context.Context) error { return io.EOF }, ctx)
		cancel()
		require.NoError(t, s.Close())
		assert.Equal(t, 1, failuresOf(r), "an unread answer has not shown the gateway healthy")
	})

	t.Run("EOF and only then the caller's end is a success", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		r, _, s := open(t, 0, func(context.Context) error { return io.EOF }, ctx)
		require.ErrorIs(t, drain(s), io.EOF)
		cancel()
		require.NoError(t, s.Close())
		assert.Equal(t, 0, failuresOf(r))
	})

	t.Run("a failure before the caller's end counts", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		r, _, s := open(t, 0, func(context.Context) error { return ir.ErrProviderUnavailable }, ctx)
		require.ErrorIs(t, drain(s), ir.ErrProviderUnavailable)
		cancel()
		require.Error(t, s.Close())
		assert.Equal(t, 2, failuresOf(r))
	})

	t.Run("closed without EOF with the caller alive stays a success", func(t *testing.T) {
		r, _, s := open(t, 0, func(context.Context) error { return io.EOF }, context.Background())
		require.NoError(t, s.Close())
		assert.Equal(t, 0, failuresOf(r))
	})

	t.Run("cancel while opening", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		step := &scriptedStep{name: "only", stream: func(ctx context.Context) (ir.ProviderStream, error) {
			cancel()
			<-ctx.Done()
			return nil, ctx.Err()
		}}
		r := stepRouter(t, 0, step)
		_, err := r.ChatCompletionStream(ctx, ir.ChatRequest{Messages: hello})
		require.Error(t, err)
		assert.Equal(t, 0, failuresOf(r))
	})
}

func embedRouter(t *testing.T, budget time.Duration, step *scriptedStep, opts ...ir.Option) *ir.Router {
	t.Helper()
	cfg := declareLadder(ir.Config{
		AllowPaid:      true,
		AttemptTimeout: budget,
		DefaultModel:   "text-embedding-004",
		Accounts: []ir.AccountConfig{
			{Provider: step.name, ID: step.name, QuotaUnit: ir.QuotaTokens, PaidEnabled: true, CostPerEmbeddingInputToken: 0.01},
		},
	})
	opts = append(opts, ir.WithSleeper(func(context.Context, time.Duration) error { return nil }))
	r, err := ir.NewRouter(cfg, []ir.Provider{step}, opts...)
	require.NoError(t, err)
	return r
}

func embedOnce(ctx context.Context, r *ir.Router) error {
	_, err := r.EmbedBatch(ctx, ir.EmbedRequest{Inputs: []string{"hi"}})
	return err
}

func embedHangs(ctx context.Context) (ir.EmbedProviderResponse, error) {
	<-ctx.Done()
	return ir.EmbedProviderResponse{}, ctx.Err()
}

// CE-I5: this path has no attempt timeout at all, so it is the no-budget case
// whatever the configuration says (owner's decision, 2026-11-07).
func TestCallerEnd_Embed(t *testing.T) {
	t.Run("cancel does not count", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		step := &scriptedStep{name: "only", embed: func(ctx context.Context) (ir.EmbedProviderResponse, error) {
			cancel()
			return embedHangs(ctx)
		}}
		r := embedRouter(t, 0, step)
		require.Error(t, embedOnce(ctx, r))
		assert.Equal(t, 0, failuresOf(r))
	})

	for _, budget := range []time.Duration{0, time.Hour} {
		t.Run("deadline counts, configured budget "+budget.String(), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
			defer cancel()
			step := &scriptedStep{name: "only", embed: embedHangs}
			r := embedRouter(t, budget, step)
			require.Error(t, embedOnce(ctx, r))
			assert.Equal(t, 1, failuresOf(r))
		})
	}

	t.Run("a refusal counts, a 429 does not", func(t *testing.T) {
		step := &scriptedStep{name: "only", embed: func(context.Context) (ir.EmbedProviderResponse, error) {
			return ir.EmbedProviderResponse{}, ir.ErrProviderUnavailable
		}}
		r := embedRouter(t, 0, step)
		require.Error(t, embedOnce(context.Background(), r))
		assert.Equal(t, 1, failuresOf(r))

		step.embed = func(context.Context) (ir.EmbedProviderResponse, error) {
			return ir.EmbedProviderResponse{}, ir.ErrRateLimited
		}
		require.Error(t, embedOnce(context.Background(), r))
		assert.Equal(t, 1, failuresOf(r))
	})
}

// CE-I6: a probe the caller ended has told nothing: half-open stays, the count
// stays, and the next caller is offered the account. There is no slot to free:
// half-open is visible to every caller at once.
func TestCallerEnd_HalfOpenProbe(t *testing.T) {
	step := &scriptedStep{name: "only", chat: refuses}
	r := stepRouter(t, time.Hour, step, ir.WithHealthConfig(ir.HealthConfig{
		FailureThreshold: 1, FailureWindow: time.Minute, UnhealthyPeriod: time.Millisecond,
	}))
	require.Error(t, chatOnce(context.Background(), r))
	time.Sleep(5 * time.Millisecond)
	require.Equal(t, ir.HealthHalfOpen, healthOf(r))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	step.chat = func(ctx context.Context) (ir.ProviderResponse, error) {
		cancel()
		return hangs(ctx)
	}
	before := step.callCount()
	require.Error(t, chatOnce(ctx, r))
	require.Equal(t, before+1, step.callCount(), "the probe was made")

	assert.Equal(t, ir.HealthHalfOpen, healthOf(r), "a failure would reopen it, a success would close it")
	assert.Equal(t, 1, failuresOf(r))

	step.chat = answers
	require.NoError(t, chatOnce(context.Background(), r), "the next caller is offered the account")
	assert.Equal(t, ir.HealthHealthy, healthOf(r))
}

type ctxKey struct{}

// strictQuota refuses a dead context, as quota/postgres and quota/redis do by
// passing it to their query, and keeps the reservations a refusal leaves behind.
type strictQuota struct {
	mu       sync.Mutex
	reserved int64
	used     int64
	settled  []context.Context
}

func (q *strictQuota) Reserve(_ context.Context, id string, amount int64, unit ir.QuotaUnit, _ string) (ir.Reservation, error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.reserved += amount
	return ir.Reservation{ID: "r", AccountID: id, Amount: amount, Unit: unit}, nil
}
func (q *strictQuota) Commit(ctx context.Context, res ir.Reservation, actual int64) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.settled = append(q.settled, ctx)
	if err := ctx.Err(); err != nil {
		return err
	}
	q.reserved -= res.Amount
	q.used += actual
	return nil
}
func (q *strictQuota) Rollback(ctx context.Context, res ir.Reservation) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.settled = append(q.settled, ctx)
	if err := ctx.Err(); err != nil {
		return err
	}
	q.reserved -= res.Amount
	return nil
}
func (q *strictQuota) Remaining(context.Context, string) (int64, error) { return 0, nil }

// CE-I7: the accounting of an attempt outlives the caller (RISK-188), with the
// caller's values and a bound of its own.
func TestCallerEnd_QuotaSettles(t *testing.T) {
	check := func(t *testing.T, q *strictQuota, wantUsed int64) {
		t.Helper()
		assert.Zero(t, q.reserved, "nothing stays reserved")
		assert.Equal(t, wantUsed, q.used)
		require.Len(t, q.settled, 1)
		assert.Equal(t, "v", q.settled[0].Value(ctxKey{}), "values of the caller stay")
		_, bounded := q.settled[0].Deadline()
		assert.True(t, bounded, "a hung store does not hold the caller for ever")
	}
	caller := func() (context.Context, context.CancelFunc) {
		return context.WithCancel(context.WithValue(context.Background(), ctxKey{}, "v"))
	}

	t.Run("chat rollback", func(t *testing.T) {
		ctx, cancel := caller()
		defer cancel()
		q := &strictQuota{}
		step := &scriptedStep{name: "only", chat: func(ctx context.Context) (ir.ProviderResponse, error) {
			cancel()
			return hangs(ctx)
		}}
		r := stepRouter(t, time.Hour, step, ir.WithQuotaStore(q))
		require.Error(t, chatOnce(ctx, r))
		check(t, q, 0)
	})

	t.Run("chat commit", func(t *testing.T) {
		ctx, cancel := caller()
		defer cancel()
		q := &strictQuota{}
		log := &resultLog{}
		step := &scriptedStep{name: "only", chat: func(context.Context) (ir.ProviderResponse, error) {
			cancel()
			return answers(nil)
		}}
		r := stepRouter(t, time.Hour, step, ir.WithQuotaStore(q), ir.WithMeter(log))
		require.NoError(t, chatOnce(ctx, r), "a delivered answer is an answer")
		check(t, q, 7)
		require.Len(t, log.events, 1)
		assert.True(t, log.events[0].Success, "the commit did not fail")
	})

	t.Run("embed rollback", func(t *testing.T) {
		ctx, cancel := caller()
		defer cancel()
		q := &strictQuota{}
		step := &scriptedStep{name: "only", embed: func(ctx context.Context) (ir.EmbedProviderResponse, error) {
			cancel()
			return embedHangs(ctx)
		}}
		r := embedRouter(t, 0, step, ir.WithQuotaStore(q))
		require.Error(t, embedOnce(ctx, r))
		check(t, q, 0)
	})

	t.Run("embed commit", func(t *testing.T) {
		ctx, cancel := caller()
		defer cancel()
		q := &strictQuota{}
		step := &scriptedStep{name: "only", embed: func(context.Context) (ir.EmbedProviderResponse, error) {
			cancel()
			return ir.EmbedProviderResponse{Embeddings: [][]float32{{1}}, Usage: ir.EmbedUsage{TotalTokens: 7}}, nil
		}}
		r := embedRouter(t, 0, step, ir.WithQuotaStore(q))
		require.NoError(t, embedOnce(ctx, r))
		check(t, q, 7)
	})
}

// CE-I8 and CE-I9: an attempt kept out of the breaker is still seen, and the walk
// goes as it did: the caller's end stops it where it stood.
func TestCallerEnd_VisibleAndWalkStops(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	first := &scriptedStep{name: "first", chat: func(ctx context.Context) (ir.ProviderResponse, error) {
		cancel()
		return hangs(ctx)
	}}
	second := &scriptedStep{name: "second", chat: answers}
	log := &resultLog{}
	r, err := ir.NewRouter(ladderOf("first", "second"), []ir.Provider{first, second}, ir.WithMeter(log))
	require.NoError(t, err)

	err = chatOnce(ctx, r)

	require.ErrorIs(t, err, context.Canceled)
	var routerErr *ir.RouterError
	require.ErrorAs(t, err, &routerErr)
	require.Len(t, routerErr.Tried, 1)
	assert.Equal(t, "first", routerErr.Tried[0].AccountID)
	assert.True(t, errors.Is(routerErr.Tried[0].Err, context.Canceled))

	require.Len(t, log.events, 1)
	assert.False(t, log.events[0].Success)
	assert.ErrorIs(t, log.events[0].Error, context.Canceled)

	assert.Zero(t, second.callCount())
}
