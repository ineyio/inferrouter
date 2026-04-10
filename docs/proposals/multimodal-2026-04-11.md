# Proposal: Multimodal Support (image / audio / video input)

**Date:** 2026-04-11
**Status:** ✅ **IMPLEMENTED** — see `docs/reports/2026-04-11-multimodal-implementation.md`
**Author:** inferrouter team (Claude, Artur Zagidullin)
**Reviewed by:** qarap team (2026-04-11)
**Blocks:** qarap v0.5 Block 2 (media pipeline)
**Source request:** `qarap/docs/inferrouter-multimodal-requirements.md` (R1–R9, R5.1)

**Changelog:**
- 2026-04-11 — Initial draft
- 2026-04-11 — Updated after qarap added R5.1 (mandatory per-modality usage return) and open questions Q8–Q9
- 2026-04-11 — **APPROVED** by qarap team without changes. All R1–R9, R5.1, Q1–Q9 accepted. R9 marked as already-implemented (see §16). Sync point: day 7 (mock ready) for parallel qarap integration.
- 2026-04-11 — **IMPLEMENTED.** 18 files, 225 tests green, 9 self-review fixes including a critical `EstimateTokens` correctness bug. Full implementation report: `docs/reports/2026-04-11-multimodal-implementation.md`. `provider/mock` with `WithMultimodal` + `WithInputBreakdownFunc` ready for qarap parallel integration.

---

## 1. Context

qarap требует от inferrouter единый inference-слой для текстовых и мультимодальных запросов (изображения, voice, видео). Первичный провайдер — Gemini 2.5 Flash-Lite через существующий `provider/gemini`. Streaming, tool use, embeddings и image generation явно вне scope.

Цель этого документа — зафиксировать дизайн API, изменения внутренних типов, расширение cost tracking и план тестов до начала имплементации, чтобы обе команды согласовали контракт.

---

## 2. Goals / Non-goals

### Goals

- Единый API `ChatCompletion` принимает и текст, и media без отдельного метода.
- 100% backward compatibility: существующий код с `Message.Content string` работает без изменений.
- Gemini provider поддерживает `inline_data` parts + парсит `promptTokensDetails[]` для разбивки по модальностям.
- Per-modality cost tracking (text / audio / image / video input) в `spend.go`.
- **Per-modality usage прокидывается обратно в `ChatResponse.Usage`** чтобы qarap Genkit plugin мог строить `InferenceUsage` для biling reconciliation (R5.1).
- Явный sentinel `ErrMultimodalUnavailable` для fail-fast graceful degradation на стороне qarap.
- Расширенный `provider/mock` для CI-тестов без расхода real Gemini quota.

### Non-goals

- Streaming multimodal (qarap не использует streaming — R7 non-req).
- Function calling / tool use.
- Автоматическое определение модальности — alias выбирает вызывающая сторона.
- Video / image **output** — только understanding (input).
- Новый провайдер-тип помимо Gemini в MVP.

---

## 3. API design

### 3.1 Расширение `Message` через multi-part content

**Текущее состояние** (`types.go:15-18`):

```go
type Message struct {
    Role    string `json:"role"`
    Content string `json:"content"`
}
```

**Новое состояние:**

```go
type Message struct {
    Role    string `json:"role"`
    Content string `json:"content,omitempty"` // legacy: text-only shortcut
    Parts   []Part `json:"parts,omitempty"`   // if set, takes precedence over Content
}

type Part struct {
    Type     PartType `json:"type"`              // "text" | "image" | "audio" | "video"
    Text     string   `json:"text,omitempty"`    // when Type == PartText
    MIMEType string   `json:"mime_type,omitempty"` // e.g. "image/jpeg", "audio/ogg"
    Data     []byte   `json:"data,omitempty"`    // raw bytes; provider handles base64
}

type PartType string

const (
    PartText  PartType = "text"
    PartImage PartType = "image"
    PartAudio PartType = "audio"
    PartVideo PartType = "video"
)
```

**Contract:**

- Если `Parts != nil`, провайдер использует `Parts` и игнорирует `Content`.
- Если `Parts == nil`, провайдер использует `Content` как единственный text part (текущее поведение).
- `Data` — raw bytes. Base64-кодирование происходит **внутри** провайдер-адаптера. Вызывающий код никогда не кодирует сам.
- Размер blob не валидируется библиотекой — это политика Telegram / провайдера, qarap enforce'ит на своей стороне (20 MB).

**Использование на стороне qarap (ожидаемое):**

```go
resp, err := router.ChatCompletion(ctx, inferrouter.ChatRequest{
    Model: "multimodal",
    Messages: []inferrouter.Message{
        {
            Role: "user",
            Parts: []inferrouter.Part{
                {Type: inferrouter.PartText, Text: "What's in this photo?"},
                {Type: inferrouter.PartImage, MIMEType: "image/jpeg", Data: photoBytes},
            },
        },
    },
})
```

### 3.2 Почему не отдельный метод

Альтернатива — `ChatCompletionMultimodal(req MultimodalRequest)`. Отвергнута:

- Дублирование code path в `router.go` (Reserve → Execute → Commit логика).
- Двойная поверхность для тестов и circuit breaker hooks.
- Genkit plugin на стороне qarap всё равно строит `Message` из Genkit `Part` — маппинг 1:1 с нашим `Part` естественен.
- Нарушает single-method симметрию с существующими провайдерами.

Multi-part content — стандартный подход OpenAI vision и нативный формат Gemini. Один метод, один code path.

---

## 4. Provider capability filter + R8 (`ErrMultimodalUnavailable`)

### 4.1 Расширение интерфейса `Provider`

```go
type Provider interface {
    Name() string
    SupportsModel(model string) bool
    SupportsMultimodal() bool // NEW
    ChatCompletion(ctx context.Context, req ProviderRequest) (ProviderResponse, error)
    ChatCompletionStream(ctx context.Context, req ProviderRequest) (ProviderStream, error)
}
```

**Значения по провайдерам:**

| Provider | `SupportsMultimodal()` |
|---|---|
| `provider/gemini` | `true` |
| `provider/openaicompat` | `false` по умолчанию, опция `WithMultimodal(true)` для совместимых моделей |
| `provider/gonka` | `false` |
| `provider/mock` | конфигурируемо через `WithMultimodal(bool)` |

### 4.2 Фильтрация в router

В `router.ChatCompletion` добавляется:

```go
hasMedia := requestHasMedia(req) // true если любой Message.Parts содержит не-text part

if hasMedia {
    candidates = filterMultimodalCapable(candidates)
    if len(candidates) == 0 {
        return nil, ErrMultimodalUnavailable
    }
}
```

### 4.3 Sentinel error

В `errors.go`:

```go
var ErrMultimodalUnavailable = errors.New("inferrouter: no multimodal-capable candidates available")
```

- **Не** попадает под `IsRetryable` — перебирать нечего, нужен user-visible fail-fast.
- **Не** попадает под `IsFatal` (в смысле auth/invalid) — это operational, qarap может отреагировать `<stripped_media>` fallback.
- qarap ловит через `errors.Is(err, inferrouter.ErrMultimodalUnavailable)`.

---

## 5. Cost tracking per modality (R5)

### 5.1 Расширение `Usage`

**Текущее** (`types.go:37-41`):

```go
type Usage struct {
    PromptTokens     int64
    CompletionTokens int64
    TotalTokens      int64
}
```

**Новое:**

```go
type Usage struct {
    PromptTokens     int64
    CompletionTokens int64
    TotalTokens      int64

    // CachedTokens — subset of PromptTokens served from provider-side context cache.
    // Orthogonal to modality. 0 if provider doesn't support caching or no cache hit.
    CachedTokens int64 `json:"cached_tokens,omitempty"`

    // InputBreakdown — per-modality breakdown of PromptTokens.
    // nil for providers that don't report it (cerebras, openai text-only).
    InputBreakdown *InputTokenBreakdown `json:"input_breakdown,omitempty"`
}

type InputTokenBreakdown struct {
    Text  int64 `json:"text"`
    Audio int64 `json:"audio"`
    Image int64 `json:"image"`
    Video int64 `json:"video"`
}
```

**Инвариант:** если `InputBreakdown != nil`, то `Text + Audio + Image + Video == PromptTokens`. Провайдер обязан обеспечивать эту сумму. Провайдеры, которые не знают разбивку (cerebras, openai text-only), оставляют поле `nil` — поведение не меняется.

**Почему `CachedTokens` отдельное поле, а не внутри `InputBreakdown`:**

- Кэш ортогонален модальности — cached content может быть text, image или смесь.
- Gemini отдаёт `cachedContentTokenCount` на верхнем уровне `usageMetadata`, не внутри `promptTokensDetails[]`.
- Поле полезно и для text-only провайдеров с caching (Anthropic prompt caching, OpenAI prompt caching), что пригодится когда добавим соответствующие провайдеры.
- qarap R5.1 явно требует `CachedInputTokens` отдельно от modality-specific полей.

### 5.2 Расширение `AccountConfig`

Добавляемые поля (`config.go`):

```go
type AccountConfig struct {
    // ... existing fields

    CostPerInputToken       float64 `yaml:"cost_per_input_token"`       // text input (existing)
    CostPerOutputToken      float64 `yaml:"cost_per_output_token"`      // existing
    CostPerAudioInputToken  float64 `yaml:"cost_per_audio_input_token"`  // NEW
    CostPerImageInputToken  float64 `yaml:"cost_per_image_input_token"`  // NEW
    CostPerVideoInputToken  float64 `yaml:"cost_per_video_input_token"`  // NEW
}
```

`Validate()` добавляет non-negative проверки для новых полей. Fallback: если specific-modality cost `== 0`, используется `CostPerInputToken` (текстовая ставка как baseline).

### 5.3 `calculateSpend` — новая формула

```go
func calculateSpend(c Candidate, usage Usage) float64 {
    output := float64(usage.CompletionTokens) * c.CostPerOutputToken

    if usage.InputBreakdown != nil {
        b := usage.InputBreakdown
        input := float64(b.Text)*c.CostPerInputToken +
            float64(b.Audio)*costOrFallback(c.CostPerAudioInputToken, c.CostPerInputToken) +
            float64(b.Image)*costOrFallback(c.CostPerImageInputToken, c.CostPerInputToken) +
            float64(b.Video)*costOrFallback(c.CostPerVideoInputToken, c.CostPerInputToken)
        return input + output
    }

    // Legacy path: no breakdown → use flat input rate
    if c.CostPerInputToken > 0 || c.CostPerOutputToken > 0 {
        return float64(usage.PromptTokens)*c.CostPerInputToken + output
    }
    if c.CostPerToken > 0 {
        return float64(usage.TotalTokens) * c.CostPerToken
    }
    return 0
}
```

### 5.4 Пример YAML (от qarap, адаптированный)

```yaml
accounts:
  - provider: gemini
    id: gemini-free
    auth:
      api_key: "${GEMINI_API_KEY}"
    daily_free: 1000
    quota_unit: requests

  - provider: gemini
    id: gemini-paid
    auth:
      api_key: "${GEMINI_API_KEY}"  # same key
    quota_unit: tokens
    paid_enabled: true
    cost_per_input_token:       0.0000001  # $0.10 / 1M  (text)
    cost_per_output_token:      0.0000004  # $0.40 / 1M
    cost_per_audio_input_token: 0.0000003  # $0.30 / 1M
    cost_per_image_input_token: 0.0000001  # $0.10 / 1M
    cost_per_video_input_token: 0.0000001  # $0.10 / 1M
    max_daily_spend: 0.50

models:
  - alias: "multimodal"
    models:
      - provider: gemini
        model: gemini-2.5-flash-lite
```

**Почему два аккаунта на один ключ:** inferrouter trackает free/paid как per-account state. Free-first policy сама перенаправит на paid, когда `gemini-free` RPD исчерпан. Gemini на бэкенде биллит paid автоматически — для нас это просто второй аккаунт с `paid_enabled: true`.

**Caveat:** circuit breaker per-account. Если free зашоркается из-за 5xx от Gemini, paid на том же ключе с большой вероятностью тоже (общий remote). Это ожидаемое поведение — fail-fast лучше, чем слепой fallback.

### 5.5 R5.1: return shape contract для qarap plugin (mandatory)

qarap `inference_metrics` path требует per-request breakdown для billing reconciliation. Контракт возврата данных из библиотеки:

```go
// ChatResponse возвращается вызывающему коду (qarap plugin).
// Usage — существующее поле, расширенное полями из §5.1.
resp, err := router.ChatCompletion(ctx, req)
if err != nil { /* ... */ }

// qarap plugin строит InferenceUsage из resp.Usage + metadata роутинга:
usage := InferenceUsage{
    TextInputTokens:   int(textOrZero(resp.Usage.InputBreakdown)),
    TextOutputTokens:  int(resp.Usage.CompletionTokens),
    AudioInputTokens:  int(audioOrZero(resp.Usage.InputBreakdown)),
    ImageInputTokens:  int(imageOrZero(resp.Usage.InputBreakdown)),
    VideoInputTokens:  int(videoOrZero(resp.Usage.InputBreakdown)),
    CachedInputTokens: int(resp.Usage.CachedTokens),
    ProviderID:        resp.Routing.Provider,
    ModelAlias:        req.Model,
    CostUSD:           ... , // вычисляется из meter event или отдельным методом
}
```

**Гарантии inferrouter:**

1. Для Gemini 2.5 запросов с media, `Usage.InputBreakdown != nil` и содержит точные числа из `promptTokensDetails[]`.
2. Для Gemini запросов без media (text-only) — `InputBreakdown` **всё равно** заполняется: `{Text: PromptTokens, Audio:0, Image:0, Video:0}`. Это упрощает qarap-сторону: один code path на чтение usage, а не `if nil`.
3. Если Gemini API в ответе не содержит `promptTokensDetails` (редкий случай: ошибка API, legacy endpoint, частичный ответ) — `InputBreakdown` остаётся `nil`, qarap plugin считает это text-only fallback (`TextInputTokens = PromptTokens`, остальные = 0). **Библиотека не крашится**, логгирует warning через meter: `slog.Warn("gemini response missing promptTokensDetails", ...)`.
4. `CachedTokens` = `usageMetadata.cachedContentTokenCount` от Gemini, или 0 если отсутствует.
5. Для text-only провайдеров (cerebras, openai) `InputBreakdown = nil` и `CachedTokens = 0` (пока не добавим caching-aware провайдеры). qarap plugin fallback-ит к treat-as-text-only.

**Почему не делать `InputBreakdown` always-non-nil для всех провайдеров:**

- Ненужная аллокация на hot path для cerebras/openai.
- Явный `nil` — корректный сигнал "этот провайдер не знает модальности", qarap может при желании отличать Gemini fallback от cerebras-default.
- Для Gemini гарантия (2) выше достаточна — у qarap единый code path для Gemini responses.

---

## 6. Gemini provider changes

### 6.1 Request encoding

Текущий `buildRequest` (`provider/gemini/gemini.go:168-193`) генерирует:

```json
{"contents": [{"role": "user", "parts": [{"text": "..."}]}]}
```

Новая логика:

```go
func (p *Provider) buildRequest(req inferrouter.ProviderRequest) geminiRequest {
    var contents []geminiContent
    for _, m := range req.Messages {
        role := m.Role
        if role == "assistant" {
            role = "model"
        }
        contents = append(contents, geminiContent{
            Role:  role,
            Parts: buildParts(m),
        })
    }
    // ...
}

func buildParts(m inferrouter.Message) []geminiPart {
    if len(m.Parts) == 0 {
        return []geminiPart{{Text: m.Content}}
    }
    parts := make([]geminiPart, 0, len(m.Parts))
    for _, p := range m.Parts {
        switch p.Type {
        case inferrouter.PartText:
            parts = append(parts, geminiPart{Text: p.Text})
        case inferrouter.PartImage, inferrouter.PartAudio, inferrouter.PartVideo:
            parts = append(parts, geminiPart{
                InlineData: &geminiInlineData{
                    MIMEType: p.MIMEType,
                    Data:     base64.StdEncoding.EncodeToString(p.Data),
                },
            })
        }
    }
    return parts
}
```

Расширение struct:

```go
type geminiPart struct {
    Text       string            `json:"text,omitempty"`
    InlineData *geminiInlineData `json:"inline_data,omitempty"`
}

type geminiInlineData struct {
    MIMEType string `json:"mime_type"`
    Data     string `json:"data"` // base64
}
```

### 6.2 Response parsing — `promptTokensDetails` + `cachedContentTokenCount`

Gemini 2.5 возвращает:

```json
{
  "usageMetadata": {
    "promptTokenCount": 1234,
    "candidatesTokenCount": 200,
    "totalTokenCount": 1434,
    "cachedContentTokenCount": 0,
    "promptTokensDetails": [
      {"modality": "TEXT",  "tokenCount": 100},
      {"modality": "IMAGE", "tokenCount": 560},
      {"modality": "AUDIO", "tokenCount": 574}
    ]
  }
}
```

Парсим и заполняем `Usage.InputBreakdown` + `Usage.CachedTokens`:

```go
type geminiTokenDetail struct {
    Modality   string `json:"modality"`
    TokenCount int64  `json:"tokenCount"`
}

// В geminiResponse.UsageMetadata добавляем:
PromptTokensDetails     []geminiTokenDetail `json:"promptTokensDetails"`
CachedContentTokenCount int64               `json:"cachedContentTokenCount"`
```

**Логика заполнения (contract §5.5):**

```go
usage := inferrouter.Usage{
    PromptTokens:     meta.PromptTokenCount,
    CompletionTokens: meta.CandidatesTokenCount,
    TotalTokens:      meta.TotalTokenCount,
    CachedTokens:     meta.CachedContentTokenCount,
}

if len(meta.PromptTokensDetails) > 0 {
    usage.InputBreakdown = buildBreakdown(meta.PromptTokensDetails)
} else if !requestHasMedia(req) {
    // Text-only запрос без promptTokensDetails — безопасный fallback: весь prompt = text.
    usage.InputBreakdown = &inferrouter.InputTokenBreakdown{Text: meta.PromptTokenCount}
} else {
    // Multimodal запрос без promptTokensDetails — аномалия, логгируем.
    logger.Warn("gemini response missing promptTokensDetails for multimodal request",
        "prompt_tokens", meta.PromptTokenCount)
    // InputBreakdown оставляем nil — qarap plugin это обработает.
}
```

Маппинг `TEXT → Text`, `IMAGE → Image`, `AUDIO → Audio`, `VIDEO → Video`. Unknown modality (теоретически — новые типы в будущих API) → складывается в `Text` + лог warning.

**Риск:** Google может переименовать поля в будущих версиях API. Закрываем интеграционным тестом (п. 9.3).

### 6.3 Streaming

Gemini streaming endpoint (`streamGenerateContent`) отдаёт тот же формат usage в последнем chunk. Изменения в `geminiStream.Next()` минимальны — дополнительно парсим `promptTokensDetails` в финальный chunk. Так как qarap streaming не использует, это low-priority, но не ломать существующее.

---

## 7. Meter (observability)

`ResultEvent.Usage` теперь может содержать `InputBreakdown` и `CachedTokens`. Расширяем `LogMeter.OnResult` (файл `meter/log.go`):

```go
attrs := []any{
    "provider", e.Provider, "account", e.AccountID, "model", e.Model,
    "duration_ms", e.Duration.Milliseconds(),
    "prompt_tokens", e.Usage.PromptTokens,
    "completion_tokens", e.Usage.CompletionTokens,
    "dollar_cost", e.DollarCost,
}
if e.Usage.CachedTokens > 0 {
    attrs = append(attrs, "cached_tokens", e.Usage.CachedTokens)
}
if b := e.Usage.InputBreakdown; b != nil {
    attrs = append(attrs,
        "text_tokens", b.Text,
        "audio_tokens", b.Audio,
        "image_tokens", b.Image,
        "video_tokens", b.Video,
    )
}
m.Logger.Info("result", attrs...)
```

**Backward compatibility обещание (для Q9):**

- Существующие поля — `provider`, `account`, `model`, `free`, `duration_ms`, `prompt_tokens`, `completion_tokens`, `dollar_cost` — **не меняются ни в именовании, ни в типе**. Текущие парсеры логов qarap продолжают работать.
- Новые поля добавляются **только когда они не-нулевые** (`cached_tokens` > 0, `InputBreakdown != nil`). Это значит:
  - Cerebras text-only лог не получает ни одного нового поля → полный zero-diff для существующих дашбордов.
  - Gemini text-only лог получает `text_tokens=N, audio_tokens=0, image_tokens=0, video_tokens=0` (т.к. §6.2 всегда заполняет `InputBreakdown` для Gemini).
  - Gemini multimodal лог — полная разбивка.
- Одна строка лога на запрос как сейчас. Grafana dashboards могут строить breakdown по новым полям, старые запросы продолжают работать.

---

## 8. Mock provider

`provider/mock.Provider` расширяется:

- `WithMultimodal(true)` — `SupportsMultimodal()` возвращает `true`.
- Новая опция `WithInputBreakdownFunc(func(req ProviderRequest) InputTokenBreakdown)` — позволяет тестам задавать детерминированную разбивку.
- По умолчанию при `Parts != nil` возвращает канонический ответ `"mock multimodal response"` с фейковой разбивкой (например, 100 text + 560 image tokens для каждого image part).

Это позволит qarap гонять CI полностью без внешних вызовов и без расхода real Gemini quota.

---

## 9. Test plan

### 9.1 Unit tests (inferrouter)

- `types_test.go` — `requestHasMedia()`, Part validation helpers.
- `router_test.go` — новые сценарии:
  - multimodal request → только multimodal-capable кандидаты в loop.
  - multimodal request + Gemini unhealthy → `ErrMultimodalUnavailable`.
  - text-only request → существующее поведение не меняется.
  - mixed alias с текстовой и multimodal моделью, media request → text-only модель отфильтрована.
- `spend_test.go` — `calculateSpend` с `InputBreakdown`:
  - Только text → равно старой формуле.
  - Audio + text → раздельные ставки применены.
  - Unknown modality fallback на text rate.
  - `InputBreakdown == nil` + `PromptTokens > 0` → legacy path, не крашится.
- `errors_test.go` — `ErrMultimodalUnavailable` не retryable и не fatal.

### 9.2 Provider tests

- `provider/gemini/gemini_test.go`:
  - `buildRequest` с image part → `inline_data` с base64.
  - `buildRequest` с audio + text parts в одном сообщении → порядок parts сохранён.
  - ChatCompletion с mock HTTP server возвращающим `promptTokensDetails` → `Usage.InputBreakdown` заполнен корректно, сумма полей равна `PromptTokens`.
  - ChatCompletion с `cachedContentTokenCount > 0` в response → `Usage.CachedTokens` заполнен.
  - **Text-only Gemini запрос без `promptTokensDetails` в response → `InputBreakdown = {Text: PromptTokens}` (fallback из §6.2).**
  - **Multimodal Gemini запрос без `promptTokensDetails` в response → `InputBreakdown == nil` + warning в логе, нет паники (§5.5 гарантия 3).**
  - Unknown modality в response → fallback в Text + нет паники.
  - Backward compat: сообщение с `Content: "hi"` без `Parts` → тот же wire format что раньше.
- `provider/mock/mock_test.go` (новый) — покрывает multimodal опции.

### 9.3 Integration test (manual, один раз)

Отдельный файл `provider/gemini/integration_test.go` с build tag `//go:build integration`:

- Реальный `GEMINI_API_KEY` из env.
- Один запрос с тестовой картинкой (вшитой в testdata/).
- Проверяем что `promptTokensDetails[]` действительно парсится (защита от future API drift).
- Не запускается в CI, запускается вручную перед релизом.

### 9.4 End-to-end с qarap

После мержа — совместная проверка одного реального message flow через qarap bot на staging: photo → `GetFile` → `Part{Image}` → `router.ChatCompletion` → Gemini → response. До этого — qarap работает против расширенного mock.

---

## 10. Migration / backwards compatibility

- **Zero breaking changes для существующих пользователей.** `Content string` остаётся working. Nil `Parts` → legacy path.
- **Provider interface breaks:** добавление `SupportsMultimodal() bool` технически ломает кастомные реализации `Provider` на стороне пользователей. Смягчение: все наши встроенные провайдеры обновляются в одном PR. Semver bump: **minor** (library pre-1.0), либо **major** если pинем 1.0 раньше.
- **Config:** новые YAML поля опциональны. Старые конфиги валидны.
- **QuotaStore / Meter interfaces:** не меняются.
- **`Usage` struct:** расширение новым опциональным полем (`*InputTokenBreakdown`) — не ломает JSON сериализацию существующих клиентов.

---

## 11. Ответы на открытые вопросы qarap

| # | Вопрос qarap | Ответ |
|---|---|---|
| Q1 | Gemini free tier точные лимиты | Мы не хардкодим числа в библиотеке. Вы конфигурируете `daily_free` по вашему плану в AI Studio; наш `QuotaStore` trackает сам. 429 от Gemini → circuit breaker. |
| Q2 | API shape — new method vs extended | Расширяем `Message` через `Parts []Part`. Один метод `ChatCompletion`, один code path (п. 3). |
| Q3 | Error taxonomy | `ErrMultimodalUnavailable` sentinel, не retryable, не fatal. Ловится через `errors.Is`. |
| Q4 | Cost observability | `Usage.InputBreakdown` + `Usage.CachedTokens` + расширенные поля в `LogMeter`. Один event per request. |
| Q5 | Mock provider для CI | `provider/mock` расширяется (п. 8). Zero real API calls в вашем CI. |
| Q6 | Timeline | ~1–1.5 недели dev work, +2–3 дня на review/integration (п. 12). |
| Q7 | Model upgrade path | Вы владеете `config/inferrouter.yaml`. Мы гарантируем стабильность API для Gemini моделей. |
| **Q8** | **Парсинг `promptTokensDetails[]`** | **Да, в плане (§6.2). Это ядро R5.1 — без парсинга разбивки всё остальное теряет смысл. Дополнительно парсим `cachedContentTokenCount` → `Usage.CachedTokens`. Для text-only Gemini запросов `InputBreakdown` всегда заполняется фоллбэком `{Text: PromptTokens}`, чтобы у qarap был единый code path на чтение usage. Если Gemini по какой-то причине не вернул details на multimodal запрос — `nil` + warning через `meter`, no crash (§5.5 гарантия 3).** |
| **Q9** | **Backward-compat формат логов cerebras** | **Текущий shape `LogMeter.OnResult` (§7): `provider, account, model, free, duration_ms, prompt_tokens, completion_tokens, dollar_cost`. Новые поля (`cached_tokens`, `text_tokens`, `audio_tokens`, `image_tokens`, `video_tokens`) добавляются **только когда не-нулевые**. Cerebras text-only лог = zero diff → ваши существующие парсеры продолжают работать без изменений. Gemini text-only лог получит `text_tokens=N` с нулями для аудио/видео/картинок (т.к. §6.2 всегда заполняет InputBreakdown для Gemini). Gemini multimodal — полная разбивка. Никакой отдельной схемы для multimodal.** |

---

## 12. Timeline

| День | Работа |
|---|---|
| 1–2 | `Message.Parts` + `Part` + `PartType` + `Usage.InputBreakdown`; миграция всех существующих тестов на zero-change path; `requestHasMedia` helper. |
| 3–4 | Gemini provider: `buildParts` с base64 + `geminiInlineData`; парсинг `promptTokensDetails`; unit тесты. |
| 5 | `calculateSpend` per-modality + `AccountConfig` new fields + `Validate` + `config_test.go`. |
| 6 | Router filter by capability + `ErrMultimodalUnavailable` + router тесты. |
| 7 | `provider/mock` расширение + `meter/log.go` расширение + финальный `go test -race ./...`. |
| 8–9 | Self-review, integration test с реальным ключом, PR. |
| +2–3 | Ревью qarap, мерж, совместный end-to-end test. |

**Критический путь для qarap:** после дня 7 (mock готов) qarap может начать параллельную интеграцию против `provider/mock` не дожидаясь финального мержа.

---

## 13. Risks

| Риск | Митигация |
|---|---|
| Google меняет формат `promptTokensDetails[]` | Integration test на реальном ключе в pre-release checklist. Fallback на flat `PromptTokens` если поле отсутствует. |
| `Provider` interface break ломает сторонних пользователей | Pre-1.0 semver, changelog с migration note. Все встроенные провайдеры обновлены в одном PR. |
| Circuit breaker зашорчивает free и paid на одном ключе одновременно | Документируем как expected behavior. Рекомендуем qarap catch'ить `ErrMultimodalUnavailable` и fall back на `<stripped_media>` + `fast` alias (это и так их план). |
| Media bytes в памяти: большие blob'ы × много одновременных запросов → OOM | Библиотека не буферизует, но base64 encoding удваивает размер в памяти на момент сериализации. Документируем. qarap cap 20 MB/blob на своей стороне — приемлемо. |
| Двойной account на один API key путает квоты | Явный пример в README + валидация логгирует warning если два аккаунта одного provider с одинаковым `api_key`. |
| **R5.1 billing reconciliation зависит от точности `InputBreakdown`** | **Unit тест на инвариант `Text+Audio+Image+Video == PromptTokens`. Integration test на реальном ключе проверяет что Gemini действительно отдаёт details для всех трёх модальностей (text/image/audio). Если Google начнёт терять точность — qarap заметит расхождение между `CostUSD` и billing, мы в ответ добавим reconciliation hook.** |
| **Кэширование (`CachedTokens`) может дважды считать один и тот же токен** | **Контракт Gemini: `cachedContentTokenCount` — это **subset** от `promptTokenCount` (не addition). Документируем это в godoc `Usage.CachedTokens`. `calculateSpend` использует только `InputBreakdown` + `CompletionTokens`, НЕ учитывает `CachedTokens` отдельно — иначе был бы double-count. qarap на своей стороне использует `CachedTokens` только для observability, не для cost. |

---

## 14. Action items

- [ ] qarap team: review, confirm API shape (`Message.Parts`)
- [ ] qarap team: confirm YAML schema (п. 5.4) подходит под их конфиг-флоу
- [ ] inferrouter team: начать имплементацию после approval
- [ ] joint: интеграционный тест с реальным Gemini ключом перед мержем
- [ ] joint: e2e на qarap staging после мержа

---

## 15. References

- qarap requirements: `qarap/docs/inferrouter-multimodal-requirements.md`
- Gemini API docs: https://ai.google.dev/api/generate-content (`inline_data`, `promptTokensDetails`)
- Existing inferrouter:
  - `types.go` — `Message`, `Usage`
  - `provider/gemini/gemini.go` — provider implementation
  - `spend.go` — cost calculation
  - `errors.go` — sentinel error definitions
  - `config.go` — `AccountConfig`

---

**Contact:** inferrouter team — Artur, via GitHub issues on this repo.

---

## 16. Approval notes & scope adjustments

### 16.1 R9 already implemented — removed from scope

qarap reviewers verified that `max_tokens` parameter is **already wired through** в текущем коде:

- `types.go:8` — `ChatRequest.MaxTokens *int`
- `router.go:252` — pass-through в `buildProviderRequest`
- `provider/gemini/gemini.go:186` — mapping в `generationConfig.maxOutputTokens`

**Scope adjustment:** R9 вычёркивается из плана работ. Timeline в §12 **не меняется** (R9 изначально оценивался в 0 дней).

### 16.2 Parallel work coordination

qarap стартует параллельно со дня 1 по своему плану (Block 0 + Block 1 + Block 1.5 + text-часть Block 2). **Единственная точка синхронизации — конец дня 7 (`provider/mock` готов).** С этого момента qarap начинает media-интеграцию против нашего расширенного mock без ожидания финального merge.

### 16.3 Joint integration test (pre-merge gate)

Перед мержем — 30-минутный созвон для двух сценариев:

1. Real multimodal request: тестовый Telegram bot → `bot.GetFile` → blob → `router.ChatCompletion` → Gemini → response. Проверка что фото действительно улетает и `promptTokensDetails[]` парсится как ожидается.
2. Staging e2e: voice (30s OGG) → проверка `Usage.InputBreakdown.Audio > 0` и корректного per-modality cost.

### 16.4 Deferred / out-of-MVP topics

Явно зафиксировано в approval letter:

- **Streaming base64 encoding** (§13 memory doubling risk) — qarap running on 4 GB VPS, expected concurrent multimodal load единицы на старте. Revisit при scale to hundreds of concurrent multimodal users. Not MVP.
- **Context caching reconciliation hook** — qarap в MVP кэш не использует (free tier Gemini 2.5 вообще не поддерживает caching). `Usage.CachedTokens` логируется только для observability/anomaly detection, не для billing. Reconciliation hook для cache discount — follow-up discussion когда qarap перейдёт на paid.
- **DOCUMENT / future modality types** — текущий fallback "unknown modality → Text + warning" принят. Если Google добавит новую модальность, qarap увидит warnings в логах и закажет новый case.

### 16.5 Что qarap обновил у себя

- `qarap/docs/qarap-concept-draft.md` → v0.5.2 (accepted API shape, YAML из §5.4, cache cost caveat, 6 новых entries в "Принятые решения", changelog).
- `qarap/docs/inferrouter-multimodal-requirements.md` — будет отмечен "all requirements closed" после старта нашей работы.
- Исправлен YAML config shape: `daily_free: 1000, quota_unit: requests` (было ошибочно `1000000, tokens` — copy-paste из Cerebras).
