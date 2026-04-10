# Multimodal Implementation Report

**Date:** 2026-04-11
**Scope:** inferrouter library — image / audio / video input support
**Proposal:** `docs/proposals/multimodal-2026-04-11.md` (approved by qarap team without changes)
**Stats:** 18 files changed, ~1100 insertions / ~50 deletions, 45+ new tests

---

## Контекст

qarap platform требовала мультимодальность для своих платных тарифов (Starter/Pro/Team) — это main upgrade trigger из Free. Без image/voice/video в InferRouter у них нет product differentiation для продажи paid планов. Запрос пришёл как `qarap/docs/inferrouter-multimodal-requirements.md` с требованиями R1–R9 + R5.1 (per-modality usage return).

Primary провайдер — Gemini 2.5 Flash-Lite (paid $0.10 text / $0.30 audio / 1M tokens). Gemini 2.0 Flash исключён как deprecated (shutdown 2026-06-01).

---

## Архитектурные решения

### 1. `Message.Parts` вместо отдельного метода

Отвергнут вариант `ChatCompletionMultimodal(req MultimodalRequest)`:
- Дублирование Reserve → Execute → Commit в `router.go`
- Двойной set тестов
- Ломает симметрию провайдерского интерфейса

Принят multi-part content в `Message`:

```go
type Message struct {
    Role    string
    Content string // legacy text shortcut
    Parts   []Part // if non-empty, takes precedence
}

type Part struct {
    Type     PartType // text | image | audio | video
    Text     string
    MIMEType string
    Data     []byte   // raw bytes, provider handles base64
}
```

**Backward compat 100%:** существующий код с `Content: "hi"` работает без изменений. Nil `Parts` → legacy path.

### 2. Capability filter вместо duck-typing по моделям

`Provider` interface расширен методом `SupportsMultimodal() bool`. Gemini → `true`, openaicompat/gonka → `false` честно. Когда в запросе media, `filterCandidates(allowPaid, needMultimodal)` дропает text-only кандидатов. Если после фильтра 0 кандидатов — возвращаем специфичный `ErrMultimodalUnavailable` вместо generic `ErrNoCandidates`.

`ErrMultimodalUnavailable` классифицирован как **не** retryable (перебирать нечего) и **не** fatal (caller может degrade через strip-media retry).

### 3. Per-modality cost tracking

`Usage` расширен:

```go
type Usage struct {
    PromptTokens     int64
    CompletionTokens int64
    TotalTokens      int64
    CachedTokens     int64                 // observability only
    InputBreakdown   *InputTokenBreakdown  // nil for text-only providers
}

type InputTokenBreakdown struct {
    Text, Audio, Image, Video int64
}
```

`AccountConfig` получил три новых поля: `CostPerAudioInputToken`, `CostPerImageInputToken`, `CostPerVideoInputToken`. **Нормализация на уровне `Candidate`:** при `buildCandidates` нулевые rate для модальности заполняются `CostPerInputToken` (text rate как baseline) через `resolveModalityCost`. Это значит `calculateSpend` просто умножает без fallback-ветвей — чистая hot path.

**Критичный инвариант:** `CachedTokens` **не** вычитается из стоимости. Google применяет cache discount server-side через сниженный `promptTokenCount`, вычитание привело бы к double-count. Поле существует только для observability (qarap логирует в dashboards).

### 4. Gemini provider — `buildUsage` с тремя путями

```go
if len(meta.PromptTokensDetails) > 0 {
    u.InputBreakdown = p.buildBreakdown(meta.PromptTokensDetails)
} else if !req.HasMedia {
    u.InputBreakdown = &InputTokenBreakdown{Text: meta.PromptTokenCount} // synthesize
} else {
    // Multimodal + no details = API drift signal
    p.logger.Warn("gemini response missing promptTokensDetails for multimodal request")
    // leave nil, caller detects
}
```

**Гарантия:** для text-only Gemini запросов `InputBreakdown` **всегда** non-nil. Это упрощает qarap код — один code path на чтение usage, без `if breakdown == nil { treat as text }`.

Unknown modality (future `DOCUMENT`) fold into `Text` с warning — graceful degradation при API drift.

### 5. `ProviderRequest.HasMedia` — precomputed flag

Вместо того чтобы каждый провайдер re-walk'ал `Messages` для определения "multimodal или нет", router вычисляет флаг один раз в `buildProviderRequest` и прокидывает в `ProviderRequest.HasMedia`. Критично для streaming пути, где `buildUsage` вызывается per chunk.

---

## Self-review findings и фиксы

Перед финальным merge прошёл self-review через 3 параллельных агента (reuse / quality / efficiency). Нашли 9 правок, в том числе **критический correctness bug**:

### Bug: `EstimateTokens` игнорировал `Parts`

**Было:** функция проходила только по `m.Content`. Multimodal запросы получали near-zero token estimate → квота резервировалась крошечная → 20 MB audio blob проходил квоту как 0 токенов.

**Стало:** per-modality byte heuristics (`tokensPerImage=560`, `audioBytesPerToken=1000`, `videoBytesPerToken=500`) + константы вынесены в top-level. Regression-guard тестом `TestEstimateTokensLargeMedia` (20 MB audio → >20k tokens).

### Trap: `openaicompat.WithMultimodal`

**Было:** опция advertised multimodal capability, но `buildRequest` всё равно сериализовал только текст. Candidate выбирался для multimodal запроса → media silently стриппались → фиктивный ответ от OpenAI на пустой запрос.

**Стало:** опция удалена целиком. `SupportsMultimodal() bool { return false }` захардкожено с комментарием почему. Когда OpenAI content[] array format будет реально имплементирован — опция вернётся.

### Duplicated helpers

- `providerRequestHasMedia` в gemini.go дублировал `requestHasMedia` в router.go. **Фикс:** оба удалены, `messagesHaveMedia(msgs []Message)` — единственный helper. Провайдер читает готовый `ProviderRequest.HasMedia`.
- `filterMultimodalCapable` был вторым allocating pass после `filterCandidates`. **Фикс:** фьюз в один filter с параметром `needMultimodal bool`.
- `costOrFallback` helper в spend.go → заменён нормализацией на уровне `Candidate` в `buildCandidates`.
- `buildBreakdown` package-function → метод `(p *Provider) buildBreakdown` для консистентности с `buildUsage`.

### Comments cleanup

Удалены все `§N.M` references на proposal (прогниют при переименовании разделов) и narration comments ("New fields added...", "Existing fields..."). Оставлены только non-obvious WHY-комментарии с объяснением инвариантов.

---

## Тестовое покрытие

**Всего:** 225 тестов зелёные на `go test -race -count=1 ./...`.

### Новые файлы

| Файл | Тестов | Что покрывает |
|---|---|---|
| `types_test.go` | 4 | `Part.IsMedia`, `messagesHaveMedia`, `Usage` zero-value, invariant sum |
| `estimate_test.go` | 10 | Per-modality estimation, legacy Content path, 20 MB audio guard |
| `config_test.go` | 9 | Validation для 6 cost-полей, full multimodal config из proposal §5.4 |
| `candidate_test.go` | 9 | `filterCandidates` все 5 ветвей, `buildCandidates` modality fallback |
| `spend_test.go` | 7 | `calculateSpend` per-modality, `CachedTokens` не вычитается |
| `multimodal_test.go` | 7 | Router capability filter, `ErrMultimodalUnavailable` (sync + stream), `HasMedia` propagation, end-to-end через mock с breakdown |
| `provider/gemini/multimodal_test.go` | 11 | `inline_data` encoding, `promptTokensDetails` parsing, three-way `buildUsage` fallback, cached tokens, unknown modality, streaming breakdown |
| `meter/meter_test.go` | (+3) | Zero-diff для text-only логов (Q9 guarantee), multimodal fields, `cached_tokens` omitted when zero |

### Критичные инварианты gated тестом

| Инвариант | Тест |
|---|---|
| 20 MB audio → token estimate > 20k | `TestEstimateTokensLargeMedia` |
| `Parts` берут верх над `Content` | `TestEstimateTokensPartsIgnoreContent` |
| Zero modality rate → text fallback в `Candidate` | `TestBuildCandidatesResolvesModalityCostFallback` |
| `CachedTokens` не влияет на spend | `TestCalculateSpendCachedTokensNotSubtracted` |
| Gemini text-only запрос получает синтезированный `InputBreakdown` | `TestBuildUsageTextOnlyFallback` |
| Gemini multimodal без details → nil + warning | `TestBuildUsageMultimodalAnomaly` |
| Streaming path surfaces `ErrMultimodalUnavailable` | `TestMultimodalStreamUnavailableWhenNoCapableProvider` |
| `HasMedia` корректно прокинут в provider | `TestProviderRequestHasMediaPropagation` |
| Cerebras text-only логи = zero diff | `TestLogMeterOnResultTextOnlyZeroDiff` |

---

## Что отложено (явно out of MVP)

Все эти пункты зафиксированы в `docs/proposals/multimodal-2026-04-11.md` §16.4 с обоснованием:

- **Streaming base64 encoding** — 20 MB audio blob даёт ~27 MB альтернативной строки во время сериализации. Для qarap MVP (4 GB VPS, единицы concurrent multimodal users) не блокер. Revisit при scale.
- **Context caching reconciliation hook** — qarap на free tier, где Gemini caching недоступен (`CachedTokens` всегда 0 для них). Follow-up когда перейдут на paid.
- **OpenAI-compat vision (`content[]` array)** — вернуть `WithMultimodal` опцию когда будет реальная имплементация.
- **DOCUMENT и другие новые modality types** — текущий fallback "unknown → Text + warning" достаточно.

---

## Backward compatibility

**Zero breaking changes для существующих пользователей:**

- Legacy `Content string` path не тронут — `TestEstimateTokensLegacyContent`, legacy тесты router'а проходят.
- YAML config с только text fields валидируется как раньше — `TestNormalizeCostsLegacyFallback`.
- `LogMeter` для text-only провайдеров (Cerebras, OpenAI) даёт bit-for-bit идентичные логи — `TestLogMeterOnResultTextOnlyZeroDiff` (Q9 обещание).
- `calculateSpend` legacy path (`InputBreakdown == nil`) работает идентично старой формуле.

**Technically breaking для кастомных `Provider` реализаций:**

- Добавлен метод `SupportsMultimodal() bool` в `Provider` interface. Внешние имплементации (если есть) должны добавить его. Migration: одна строка `func (p *T) SupportsMultimodal() bool { return false }`.
- Добавлено поле `HasMedia bool` в `ProviderRequest`. Опционально к чтению — нулевое значение безопасно.

Semver bump: **minor** (pre-1.0 library).

---

## Связанные документы

- **Proposal:** `docs/proposals/multimodal-2026-04-11.md` (approved без правок)
- **qarap requirements:** `qarap/docs/inferrouter-multimodal-requirements.md` (R1-R9 + R5.1)
- **Smart-routing follow-up:** `docs/proposals/smart-routing-2026-03-20.md` — fair-распределение запросов между моделями одного провайдера. Полностью backward compatible с текущей multimodal архитектурой, можно делать отдельным PR когда квота станет constraint.
