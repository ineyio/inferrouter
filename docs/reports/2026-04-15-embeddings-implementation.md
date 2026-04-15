# Embeddings Implementation Report

**Date:** 2026-04-15
**Scope:** inferrouter library — text embedding support as an optional capability interface
**Proposal:** `iney/docs/proposals/inferrouter-embeddings.md` (Approved v2, qarap review 2026-04-15)
**Stats:** 14 files touched (9 new, 5 modified), ~1733 lines of new code, 25 new test functions (31 cases with subtests)

---

## Контекст

Vector Embedding Service (VES) в iney engine — первая платная RAG-фича, её первый клиент — qarap platform. VES Phase 0 требовал text embedding API в inferrouter, которого не было: библиотека поддерживала только chat completion (text + multimodal). Gemini `text-embedding-004` — primary модель, 768 dimensions, multilingual, free tier 1500 RPM.

Параллельно proposal прошёл lightweight review у qarap team, которые поймали один **критический correctness bug**: cross-model fallback в embedding aliases. Embedding vector spaces несовместимы между разными моделями — cosine similarity между векторами `text-embedding-004` и `gemini-embedding-001` — случайный шум, который попадает в реалистичный `0.6–0.8` диапазон и автоматически не детектится. Чистое silent failure в продакшене.

Это решение фиксируется на уровне архитектуры через single-model alias invariant с fail-fast валидацией при `NewRouter`.

---

## Архитектурные решения

### 1. Optional `EmbeddingProvider` interface, не расширение `Provider`

Отвергнут вариант добавить `Embed`/`SupportsEmbedding` в существующий `Provider` interface:
- Chat-only провайдеры (openaicompat, gonka) были бы вынуждены писать `return nil, ErrNotSupported` — runtime lie вместо honest compile-time capability
- Миграционный цикл для внешних имплементаций `Provider`, если они есть
- Interface pollution: half of the methods не работают для половины провайдеров

Принят **optional capability interface**:

```go
type EmbeddingProvider interface {
    Name() string
    SupportsEmbeddingModel(model string) bool
    Embed(ctx context.Context, req EmbedProviderRequest) (EmbedProviderResponse, error)
    MaxBatchSize() int
}
```

Провайдер может имплементировать один, другой, или оба интерфейса. `Router.NewRouter` делает type assertion при регистрации и сохраняет двойную map:

```go
type Router struct {
    providers      map[string]Provider
    embedProviders map[string]EmbeddingProvider // populated via type-assert
    // ...
}

// In NewRouter:
for _, p := range providers {
    provMap[p.Name()] = p
    if ep, ok := p.(EmbeddingProvider); ok {
        embedProvMap[ep.Name()] = ep
    }
}
```

Chat-only провайдеры — zero cost: отсутствуют в `embedProviders`, не фигурируют в `buildEmbedCandidates`, `ErrNoEmbeddingProviders` возвращается симметрично `ErrMultimodalUnavailable`.

### 2. Single-model alias invariant (qarap correctness fix)

**Ключевое correctness решение всей реализации.** Embedding alias в конфиге обязан содержать ровно один `ModelRef`. Cross-model fallback отвергнут fail-fast при `NewRouter` через `validateEmbeddingAliases()`:

```go
func validateEmbeddingAliases(cfg Config, embedProviders map[string]EmbeddingProvider) error {
    for _, alias := range cfg.Models {
        containsEmbedding := false
        for _, ref := range alias.Models {
            prov, ok := embedProviders[ref.Provider]
            if ok && prov.SupportsEmbeddingModel(ref.Model) {
                containsEmbedding = true
                break
            }
        }
        if containsEmbedding && len(alias.Models) > 1 {
            return fmt.Errorf("%w: embedding alias %q must contain exactly one model entry...",
                ErrInvalidConfig, alias.Alias)
        }
    }
    return nil
}
```

**Важное свойство:** валидация знает про embedding capabilities только после регистрации providers — поэтому вызов `validateEmbeddingAliases` находится **после** применения options в `NewRouter`, не в `Config.Validate()` (где нет доступа к providers). Chat aliases с multiple models остаются валидными — инвариант активируется только если хотя бы один entry — embedding model.

Reliability для embeddings строится через **multi-account fallback на одну модель**, а не multi-model: free-first policy выбирает лучший аккаунт, circuit breaker per-account, vector space consistent.

### 3. `EmbedBatch` с auto-split и `ErrPartialBatch` multi-return

`Router.Embed` (single call) и `Router.EmbedBatch` (auto-split) — два уровня API:

- `Embed` — low-level escape hatch для caller'ов, которые сами управляют батчингом. Возвращает `ErrBatchTooLarge`, если input превышает `MaxBatchSize()` первого кандидата.
- `EmbedBatch` — default choice: принимает любое количество inputs, делит на sub-batches по `MaxBatchSize`, каждый sub-batch через полный reserve→execute→commit workflow.

**Partial failure pattern** — самый тонкий момент контракта:

```go
resp, err := router.EmbedBatch(ctx, req)
var partial *ErrPartialBatch
if errors.As(err, &partial) {
    // resp.Embeddings содержит valid prefix len == partial.ProcessedInputs
    persist(resp.Embeddings)
    return retryWith(req.Inputs[partial.ProcessedInputs:])
}
```

`EmbedBatch` возвращает **непустой** `EmbedResponse` **одновременно** с `*ErrPartialBatch` — это не Go-идиоматично, но оправдано: потеря успешной работы = реальные деньги на inferrouter API + дубликаты в vector_chunks (которые ON CONFLICT спасёт, но повторный call не вернёт). Multi-return с валидным prefix — это checkpoint+resume primitive для VES worker.

Quota handling:
- Successful sub-batches → `Commit` на estimated tokens
- Failing sub-batch → `Rollback`
- Unattempted remainder → не резервируется (loop останавливается)
- Net: consumer платит только за successful work, симметрично VES Reserve/Finalize refund delta

Implementation: `embedOnce` — helper, инкапсулирующий один sub-batch call через candidate retry loop. `EmbedBatch` вызывает его для каждого chunk и консолидирует результаты. На первом падении после успешных возвращает `ErrPartialBatch`. На падении первого sub-batch'а — full error без partial response.

### 4. Отдельный `EmbedUsage` type

Переиспользование chat `Usage` отвергнуто:

```go
type Usage struct {
    PromptTokens     int64
    CompletionTokens int64
    TotalTokens      int64
    CachedTokens     int64
    InputBreakdown   *InputTokenBreakdown // per-modality
}
```

У embeddings нет completion tokens, cached tokens (Gemini embedding endpoints не возвращают cache metadata), modality breakdown (text only). Zero-filled поля = семантический мусор в логах и dashboards.

```go
type EmbedUsage struct {
    InputTokens int64
    TotalTokens int64 // == InputTokens для embeddings
}
```

Стоимость — одна new field в `AccountConfig`:

```go
CostPerEmbeddingInputToken float64 `yaml:"cost_per_embedding_input_token"`
```

**Почему отдельное поле, а не reuse `CostPerInputToken`:** Gemini `text-embedding-004` стоит ~$0.025/1M input tokens, а chat Gemini 2.5 — ~$0.30/1M. Reuse привёл бы к 12× завышению стоимости embeddings и поломке free-first policy для аккаунта, который предлагает бесплатные embeddings и платный chat (частый случай на Google Cloud). Zero = embeddings отключены для аккаунта.

### 5. Gemini provider: single endpoint, heuristic token estimation

Gemini API имеет два endpoint'а:
- `POST /v1beta/models/{model}:embedContent` для single input
- `POST /v1beta/models/{model}:batchEmbedContents` для batch

Реализация использует **только batch endpoint**, даже для N=1. Один code path, меньше багов в request/response parsing, меньше тестов. Overhead одного request в batch-обёртке пренебрежимо мал.

**Gemini quirk:** каждый sub-request внутри batch body обязан повторять полное имя модели (`"models/text-embedding-004"`), даже при том что URL уже содержит model — иначе API возвращает 400. Этот factoid задокументирован в коде.

**Token estimation:** endpoints НЕ возвращают token count в response body (в отличие от chat). Provider использует тот же `len(text) / 4` heuristic что и router-side `EstimateEmbedTokens`, чтобы Reserve и Commit совпадали в quota store. Accuracy ~±20% приемлема — consumers (VES, qarap) не биллят end-юзеров по per-token precision.

Альтернатива — отдельный `countTokens` call через `:countTokens` endpoint — отвергнута: удвоение latency на каждый batch не оправдано для 20% precision gain. Если в будущем понадобится — добавить опциональный config flag, не default.

### 6. Mock provider: отдельный `EmbedProvider` struct, deterministic fake embeddings

В `provider/mock` реализован параллельный `EmbedProvider` struct (не встроен в существующий chat `Provider`), чтобы:
- Chat-only тесты остались полностью независимыми
- Embed-only тесты не тянули за собой chat expectations
- Каждый struct composes однозначную capability

`fakeEmbedding(text, dims)` — deterministic FNV-based hash → fixed-length `[]float32` в `[-1, 1]`. Same text → same vector — reproducible assertions в consumer тестах.

Опции следуют паттерну chat mock:
- `WithEmbedSupportedModels`
- `WithEmbedMaxBatch`
- `WithEmbedLatency`
- `WithEmbedError` (static error)
- `WithEmbedResponseFunc` (полный контроль)
- `WithEmbedDimensions`
- `WithEmbedTokensPerInput`

---

## Deviations from RFC / tactical decisions

### `HealthState` vs `HealthStatus` — naming catch

Первый draft `embed_candidate.go` писал `Health HealthStatus` — я предположил имя без проверки. Type на самом деле — `HealthState`. Тривиальная правка, поймана первым `go build`.

Lesson: для нового кода, опирающегося на существующие типы, стоит сразу делать быстрый `grep` по точному имени даже если оно кажется очевидным.

### Embedding candidate cost filter

В `buildEmbedCandidates` добавлен небольшой filter, не описанный в RFC явно:

```go
if acc.DailyFree == 0 && acc.CostPerEmbeddingInputToken == 0 {
    continue
}
```

Semantics: аккаунт с нулевым free quota И нулевой embedding cost считается "embeddings отключены на уровне аккаунта". Это позволяет конфигурировать одного провайдера с двумя аккаунтами — один для chat (без `CostPerEmbeddingInputToken`), другой для embeddings — без того чтобы chat аккаунт пытался принимать embed requests.

### Free-first policy inline, не отдельный `EmbedPolicy` interface

В RFC §3.4 упомянут `policy.Select()` как переиспользование chat policy. На практике chat `Policy.Select([]Candidate)` не принимает `[]EmbedCandidate` — это два разных типа. Варианты:

1. Сделать `Policy` generic — крупный refactor chat path
2. Конвертировать `EmbedCandidate` в dummy `Candidate` для сортировки — грязно
3. Inline free-first прямо в `prepareEmbedRoute` — выбрано

```go
free := candidates[:0:0]
paid := make([]EmbedCandidate, 0, len(candidates))
for _, c := range candidates {
    if c.Free {
        free = append(free, c)
    } else {
        paid = append(paid, c)
    }
}
return append(free, paid...), nil
```

Если в будущем появится запрос на кастомные embedding policies (например, cost-first), это станет триггером для generic `Policy[T]` — сейчас YAGNI.

### `EmbedResponse.Routing` содержит routing **последнего** sub-batch

В multi-batch сценарии sub-batches могут теоретически routing-иться на разные accounts (если primary упал на первом, secondary продолжил). `EmbedResponse.Routing` содержит `RoutingInfo` последнего успешного sub-batch'а, не aggregate. На practice это один и тот же аккаунт почти всегда (фейлы редкие), а aggregate `RoutingInfo` — более сложный контракт без реального use case.

Задокументировано в `EmbedBatch` docstring.

### `embedOnlyStub` в тестах

Мелкое friction point: `NewRouter` принимает `[]Provider`, не два списка. Чтобы зарегистрировать mock `EmbedProvider` (который имплементит только `EmbeddingProvider`), тесты используют wrapper `embedOnlyStub`, который имплементит `Provider` c no-op chat methods.

Это не идеально — честнее было бы сделать `NewRouter` принимать отдельный слайс `[]EmbeddingProvider` или сам детектировать dual-capability через одну коллекцию. Но это изменение сигнатуры public API, которое затронет всех существующих пользователей. Для Phase 1 stub в тестовом helper'е — приемлемый trade-off. Production провайдеры (Gemini) имплементируют ОБА интерфейса на одном struct, так что этой проблемы у real code нет.

---

## Coverage

### Core unit tests (`embed_core_test.go`, 9 tests)

- `TestEstimateEmbedTokens_BasicBatch` — heuristic math
- `TestEstimateEmbedTokens_Empty` — nil, empty slice, empty string
- `TestEmbed_EmptyInputs` — `ErrInvalidRequest` wrapping
- `TestEmbed_BatchTooLarge` — `Embed` (не `EmbedBatch`) падает когда inputs > MaxBatchSize
- `TestNewRouter_RejectsCrossModelEmbeddingAlias` — **correctness invariant, core тест всей работы**
- `TestNewRouter_AcceptsSingleModelEmbeddingAlias` — single-model + multi-account happy path
- `TestNewRouter_ChatAliasMultiModelUnaffected` — chat aliases с multiple models остаются валидными
- `TestErrPartialBatch_ErrorsAs` — `errors.As` unwrapping, `errors.Is` на Cause
- `TestConfig_NegativeEmbeddingCost` — validation error message

### Router tests (`embed_router_test.go`, 8 tests)

- `TestEmbedBatch_HappyPathSingleBatch` — 3 inputs, 1 provider, 1 call
- `TestEmbedBatch_SplitOn100Boundary` — 120 inputs → 2 sub-batches, order preserved
- `TestEmbedBatch_FallbackOnRateLimit` — primary `ErrRateLimited` → secondary succeeds
- `TestEmbedBatch_PartialFailureReturnsPrefix` — 12 inputs, batch 5, fail on 2nd sub-batch, `ProcessedInputs=5`, valid prefix
- `TestEmbedBatch_FirstBatchFailsReturnsFullError` — fail на первом → full error, no partial
- `TestEmbedBatch_NoEmbeddingProviders` — chat-only router → `ErrNoEmbeddingProviders`
- `TestEmbedBatch_PropagatesTaskTypeAndDimensions` — `RETRIEVAL_QUERY`, `OutputDimensionality=256` → provider receives
- `TestEmbedBatch_ModelFieldIsResolved` — `default-embedding` alias → `resp.Model == "text-embedding-004"`

### Gemini provider tests (`provider/gemini/embed_test.go`, 9 tests с subtests)

- `TestSupportsEmbeddingModel` — whitelist (accept 004/001, reject chat и OpenAI embeddings)
- `TestMaxBatchSize` — returns 100
- `TestEmbed_HappyPath` — httptest server проверяет URL shape, API key, per-request model field, task type, decodes response
- `TestEmbed_OutputDimensionality` — serialized as `outputDimensionality`
- `TestEmbed_OmitsOutputDimensionalityWhenZero` — pointer + omitempty, raw body regex check
- `TestEmbed_ErrorMapping` — 6 subtests: 429, 401, 403, 400, 500, 502 → правильные sentinels
- `TestEmbed_ResponseSizeMismatch` — Gemini вернул меньше embeddings чем inputs → error
- `TestEmbed_ContextCancellation` — cancelled context → err
- `TestEmbed_MalformedResponseBody` — invalid JSON → decode error

### Не покрыто (намеренно)

- **Real Gemini API smoke test** — RFC §7 Phase 2 упоминает manual smoke test на 3 RU текстах. Не в CI (требует live API key), будет выполнен отдельно при интеграции в VES
- **qarap RU Wikipedia benchmark fixtures** — ожидается contribution от qarap team (RFC §10.2)
- **Heartbeat / retry backoff для embedding requests** — переиспользуется rate limiter chat path без изменений, не требует отдельных тестов embedding stack'а
- **LogMeter embedding events** — существующий meter принимает embedding operations как chat Usage (settleEmbedSuccess конвертирует `EmbedUsage` → `Usage{PromptTokens: InputTokens}`), без отдельных тестов meter side

### Verification gates

- `go test -race ./...` — всё зелёное
- `go vet ./...` — чисто
- `go build ./...` — включая `examples/embeddings/` пример
- Ни один существующий chat тест не сломался

---

## Files changed

**New (9 files, ~1733 lines):**

| File | Lines | Purpose |
|------|-------|---------|
| `embed_types.go` | 82 | Public API types с invariants в docstrings |
| `embed_provider.go` | 41 | `EmbeddingProvider` optional interface |
| `embed_candidate.go` | 137 | `EmbedCandidate`, `buildEmbedCandidates`, `filterEmbedCandidates`, `EstimateEmbedTokens` |
| `embed_router.go` | 370 | `Router.Embed`, `Router.EmbedBatch`, helpers, `validateEmbeddingAliases` |
| `embed_core_test.go` | 228 | 9 unit tests + test helpers (embedOnlyStub) |
| `embed_router_test.go` | 288 | 8 integration tests |
| `provider/gemini/embed.go` | 168 | Gemini `batchEmbedContents` implementation |
| `provider/gemini/embed_test.go` | 254 | 9 tests с httptest (14 cases включая subtests) |
| `provider/mock/embed.go` | 165 | Mock `EmbedProvider` с deterministic fake vectors |

**Modified (5 files):**

| File | Change |
|------|--------|
| `errors.go` | +`ErrNoEmbeddingProviders`, +`ErrBatchTooLarge`, +`ErrInvalidConfig`, +`ErrPartialBatch` struct с multi-return docs |
| `config.go` | +`CostPerEmbeddingInputToken` в `AccountConfig`, +validation |
| `router.go` | +`embedProviders` map в `Router` struct, +type-assertion регистрация в `NewRouter`, +вызов `validateEmbeddingAliases` |
| `README.md` | +секция `## Embeddings` с usage, config, Genkit note |
| `CLAUDE.md` | +`EmbeddingProvider` в interfaces table, +routing flow для embeddings, +single-model invariant explanation |

**New example:**

- `examples/embeddings/main.go` — runnable demo с Gemini: index 3 docs + query, demonstrates alias + task type + partial failure handling

---

## Что прошло хорошо

- **RFC как точный план.** После двух раундов ревью (qarap + prior внутренних) RFC оказался детальным настолько, что реализация шла линейно без архитектурных surprises. Ни одно решение не потребовало pause и пересмотра. Все 13 tasks закрыты за один session.
- **Existing inferrouter infrastructure reusable почти полностью.** Reservation workflow, circuit breaker, rate limiter, spend tracker, config loader, meter — ничего не потребовало модификации. Embedding path — параллельный track, который подключается type-assertion'ом.
- **Тесты по первому запуску почти все зелёные.** Единственный bug — assertion `Contains(..., "single")` против error message с фразой "exactly one model entry". Правка на 1 строку.
- **qarap review добавил реальную ценность.** Single-model invariant — это именно тот класс ошибок, который domain experts ловят лучше general reviewers. Claude и Gemini в предыдущих раундах это упустили.

## Что стоит улучшить в future

- **`NewRouter` signature.** Приём одного `[]Provider` + type-assertion discovery EmbeddingProvider работает, но test friction с `embedOnlyStub` показывает limit этого подхода. Если появится третья capability (reranker, image generation?), стоит либо перейти на `NewRouter(cfg, providers []AnyProvider, opts...)` с Generic wrapping, либо на `NewRouter(cfg, ChatProviders: [...], EmbedProviders: [...])` explicit structure. Не critical сейчас.
- **Policy для embedding path.** Сейчас inline free-first в `prepareEmbedRoute`. Если появится запрос на cost-first или другую strategy для embeddings — настанет момент generalize `Policy` interface (или дублировать его для embed, который наименее желательно).
- **LogMeter не знает про embedding operations явно.** `settleEmbedSuccess` эмитит `ResultEvent` с `Usage{PromptTokens: InputTokens}`, поэтому existing meters работают, но не различают "embedding call, 500 tokens" vs "chat call with 500 prompt tokens". Если observability требует этого — добавить поле `Operation Operation` в `ResultEvent` (enum `chat | embed`). Не блокер, backlog.
- **Token count precision.** Heuristic ±20% приемлем, но при масштабе ledger возможен дрифт. Optional опциональный точный `countTokens` call через config flag (`precise_embedding_billing: true`) — nice-to-have если VES начнёт видеть budget mismatches в продакшне.

---

## Next steps

1. **Commit inferrouter changes** — review and commit as one cohesive feature branch
2. **Tag release** — bump version (semver minor, new feature additive — `v1.x.0`)
3. **Bump `go.mod` в iney engine** когда начнётся VES Phase 0, чтобы engine мог импортировать новый API
4. **Phase 2 smoke test** — manual test с real Gemini API key на 3 RU Wikipedia статьях (Пушкин, Наполеон, WWII) как подготовка к VES Phase 1 exit criteria
5. **qarap fixtures** — ждём их PR с RU benchmark test cases для `provider/gemini/embed_test.go`

---

**Implementation status: DONE.** Готово к интеграции в VES Phase 0.
