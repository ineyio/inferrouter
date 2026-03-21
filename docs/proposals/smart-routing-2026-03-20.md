# Proposal: Smart Routing — эвристики из production-опыта iney

**Date:** 2026-03-20
**Status:** Draft
**Author:** Artur Zagidullin, Claude
**Context:** Максимизация free tier Cerebras для iney (discover) и qarap (chat bots)

---

## Motivation

InferRouter сейчас роутит запросы по простому принципу: free first → try next on failure. Это работает, но не оптимально:

- При наличии 3 моделей с независимыми лимитами (14.4K RPD каждая = 43.2K/день) router использует модели последовательно, а не параллельно. Первая модель выгорает, потом вторая, потом третья. К вечеру все лимиты исчерпаны, хотя при равномерном распределении хватило бы на весь день.

- При провайдерском outage router перебирает все модели одного провайдера, сжигая RPM budget каждой. Cascade burn-through по аналогии с cascade cooldown в SOCKS5-пуле iney.

- После рестарта RateLimiter пустой — burst запросов выжигает весь RPM за секунду.

- Локальные счётчики дрейфуют от реальности. Cerebras отдаёт точные remaining в headers каждого ответа — мы их игнорируем.

Все четыре проблемы имеют аналоги в production-опыте iney, где они уже решены.

---

## 1. Remaining-Aware Routing

### Проблема

Policy сортирует кандидатов по `Remaining` из QuotaStore (daily token quota). Но не видит RPD headroom из RateLimiter. Модель с 12K/14.4K spent идёт первой — потому что в alias она первая.

### Аналог в iney

Budget Manager распределяет запросы fair между клиентами. Клиент с меньшим расходом получает приоритет → все клиенты доживают до конца дня.

### Решение

RateLimiter экспонирует оставшийся бюджет:

```go
// Remaining returns how many requests are available in each window.
func (rl *RateLimiter) Remaining(accountID, model string) Limits
```

Candidate получает новое поле `RPDRemaining int`. Policy учитывает его при сортировке — модель с большим RPD headroom идёт первой.

```go
// В FreeFirstPolicy.Select():
// Среди free-кандидатов сортировать по RPDRemaining DESC (больше запас → выше приоритет).
// Это распределяет нагрузку равномерно: если gpt потратил 10K, а qwen 2K → qwen идёт первым.
```

### Эффект

Вместо последовательного выгорания (gpt → qwen → llama) все три модели тратят бюджет равномерно. При равномерном трафике каждая модель доживает до конца дня.

### Effort

~30 LOC: `Remaining()` method + поле в Candidate + tweak в policy sort.

---

## 2. Rate Limit Header Feedback

### Проблема

Локальный RateLimiter считает запросы приблизительно. Реальный бюджет может отличаться:
- Другие клиенты того же Cerebras org тратят бюджет параллельно
- После рестарта локальный счётчик обнуляется, но провайдер помнит реальный расход
- Clock drift между нашим sliding window и token bucket Cerebras

### Аналог в iney

Все решения в Budget Manager основаны на реальных метриках, не на предположениях. ASN budget корректируется по фактическому использованию.

### Решение

**Шаг 1:** Расширить `ProviderResponse` для передачи rate limit данных от провайдера:

```go
type ProviderResponse struct {
    // ... existing ...
    RateLimits *RateLimitHeaders // parsed from response headers, nil if absent
}

type RateLimitHeaders struct {
    RemainingRequestsDay  int
    RemainingTokensMinute int
    ResetRequestsDay      time.Duration // seconds until daily reset
    ResetTokensMinute     time.Duration // seconds until per-minute reset
}
```

**Шаг 2:** В `openaicompat` provider — парсить `x-ratelimit-*` headers из HTTP response.

**Шаг 3:** В `settleSuccess()` — если `resp.RateLimits` не nil, обновить RateLimiter реальными данными:

```go
func (rl *RateLimiter) SyncFromProvider(accountID, model string, remaining Limits)
```

Это заменяет локальный approximation точными данными от провайдера. Дрейф корректируется с каждым запросом.

### Effort

~40 LOC: struct + header parser в openaicompat + sync метод в RateLimiter.

---

## 3. Provider-Level Health (Cascade Protection)

### Проблема

Cerebras outage. Router пробует `gpt-oss-120b` → 429. Пробует `qwen-3-235b` → 429. Пробует `llama3.1-8b` → 429. За один HTTP-запрос сожжено 3 RPM слота. За минуту — 90 слотов (30 × 3 модели). Все бюджеты выгорели, хотя проблема была на стороне провайдера.

### Аналог в iney

StickyProxyDialer изолирует blast radius. Если один endpoint плохой — не раскидываем ошибки по всему пулу. `pinFailed` flag переключает на fallback только при proxy-level ошибках, не при target-level.

### Решение

Добавить provider-level health поверх существующего account-level:

```go
type HealthTracker struct {
    // ... existing per-account ...
    providers map[string]*providerHealth // NEW: per-provider aggregate
}
```

Логика: если ≥N аккаунтов одного провайдера unhealthy за короткий период → пометить провайдер целиком как unhealthy. `filterCandidates()` отсеивает все кандидаты этого провайдера.

Различие между provider-level и model-level ошибками:
- HTTP 429 от конкретной модели → model-level (не cascade)
- HTTP 503, network timeout, connection refused → provider-level (cascade risk)

### Effort

~50 LOC: extend HealthTracker + classify errors в provider adapter.

---

## 4. Cold Start Protection

### Проблема

Рестарт сервиса. RateLimiter пустой. 100 накопившихся запросов уходят burst'ом — 30 в первую секунду, 70 получают `ErrRPMExceeded`. Провайдер может интерпретировать burst как abuse. Также: если provider headers feedback ещё не получен, локальный счётчик не знает реальный расход.

### Аналог в iney

Budget Manager `cold_start_ratio: 0.1` — после рестарта token bucket стартует на 10% ёмкости. Рампируется до 100% за первую минуту. Защита от burst + время для синхронизации с реальным состоянием.

### Решение

```go
type Limits struct {
    RPM           int     `yaml:"rpm"`
    RPH           int     `yaml:"rph"`
    RPD           int     `yaml:"rpd"`
    ColdStartRatio float64 `yaml:"cold_start_ratio"` // 0.0-1.0, default 1.0 (no protection)
}
```

`Allow()` в первую минуту после `SetLimit`/`SetModelLimits` применяет `RPM * ColdStartRatio` как effective limit. После первой минуты — полный RPM.

Или проще: после получения первого response с rate limit headers — синхронизация заменяет cold start автоматически (решение 2 покрывает этот кейс).

### Effort

~15 LOC: timestamp в `multiWindow` + adjusted limit в `allow()`.

---

## Порядок реализации

| Phase | Что | Зависит от | Impact |
|-------|-----|-----------|--------|
| 1 | Remaining-aware routing | — | Равномерное распределение → +2-3x free usage lifespan |
| 2 | Header feedback | — | Точный бюджет, устраняет drift |
| 3 | Provider-level health | — | Cascade protection |
| 4 | Cold start | Phase 2 (headers делают cold start менее критичным) | Burst protection |

Phase 1 и 2 независимы, можно параллелить.

---

## Open Questions

| # | Вопрос | Предварительно |
|---|--------|----------------|
| 1 | Нужен ли TPM (tokens per minute) tracking? Cerebras лимитирует 60K TPM на free | Не на MVP. RPM — bottleneck сейчас. TPM добавляется в ту же структуру Limits позже |
| 2 | Как обрабатывать header feedback при multi-instance? Два инстанса видят разные remaining | Не актуально (single binary). При multi-instance — Redis-backed RateLimiter |
| 3 | Должен ли cold_start_ratio быть в Limits или отдельным параметром? | В Limits — проще конфигурация, одно место для всех rate limit настроек |
| 4 | Policy должна учитывать RPD remaining или RPH remaining? | RPD — основной bottleneck. RPH = 900 достаточно, RPD = 14.4K — узкое место |
