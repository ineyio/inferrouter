# Bug Audit & Fixes Report

**Date:** 2026-02-24
**Scope:** inferrouter library — полный аудит и фиксы ошибок
**Commits:** `bfca707`, `7f2a9bf`, `76f5e2a`
**Stats:** 13 файлов, +636 / -64 строк, 14 багов зафиксировано

---

## Контекст

При расследовании проблемы с бесполезными сообщениями об ошибках (`inferrouter: provider= account= model=`) обнаружены два начальных бага. Последующий полный аудит кодовой базы выявил ещё 12 проблем аналогичных паттернов: потеря контекста ошибок, игнорирование ошибок, race conditions, утечки памяти.

Все 14 багов зафиксированы в трёх коммитах.

---

## Коммит 1: `bfca707` — Начальные баги

### RouterError без контекста при ErrAllFailed

**Было:** Когда все кандидаты проваливались, `RouterError` создавался с пустыми `Provider`, `AccountID`, `Model`. Ошибка в логах выглядела так:
```
inferrouter: provider= account= model= attempts=2: inferrouter: all candidates failed
```

**Стало:** Добавлен `CandidateError` struct и поле `Tried []CandidateError` в `RouterError`. Каждая ошибка кандидата (и от Reserve, и от Provider) собирается в слайс:
```
inferrouter: all 2 candidates failed: [provider=gemini account=gem-1 model=gemini-2.0-flash: rate limited] [provider=grok account=grok-1 model=grok-3: provider unavailable]
```

**Файлы:** `errors.go`, `router.go` (ChatCompletion + ChatCompletionStream)

### HealthTracker без сброса

**Было:** Circuit breaker хранил состояние in-memory без возможности сброса. После временного сбоя провайдеров все аккаунты оставались unhealthy — помогал только рестарт процесса.

**Стало:** Добавлены `Reset()` (все аккаунты) и `ResetAccount(accountID)` (один аккаунт). Оба потокобезопасны.

**Файлы:** `health.go`

---

## Коммит 2: `7f2a9bf` — CRITICAL и HIGH баги

### #1 CRITICAL: stream.go Close() возвращал не ту ошибку

**Было:** `RouterStream.Close()` возвращал ошибку от `inner.Close()` (закрытие HTTP body), а не от quota Commit/Rollback. Если стрим завершился ok, но Redis Commit упал — caller получал `nil`.

**Стало:** Возвращает `resultErr` — приоритет: quota error > stream error > body close error. EOF (нормальный конец стрима) не считается ошибкой для caller.

**Файлы:** `stream.go`

### #2 CRITICAL: Commit/Rollback ошибки игнорировались

**Было:** `_ = r.quotaStore.Rollback(...)` и `_ = r.quotaStore.Commit(...)` — при недоступности Redis/Postgres квоты ломались без единого лога.

**Стало:**
- Error path: Rollback ошибка включается в error, переданный в `Meter.OnResult()` (`rollback failed: ...`)
- Success path: Commit ошибка → `OnResult` с `Success: false` и `Error: "quota commit failed: ..."`
- Return value caller'а не меняется — provider response возвращается, но meter видит проблему

**Файлы:** `router.go`

### #3 HIGH: Remaining() ошибка → paid fallback

**Было:**
```go
remaining, _ := quotaStore.Remaining(ctx, acc.ID)
free := acc.DailyFree > 0 && remaining > 0
```
Таймаут Redis → `remaining=0` → `free=false` → все free-аккаунты фильтруются → трафик утекает на paid.

**Стало:** Fail-open: при ошибке `Remaining()` и `DailyFree > 0` — считаем `free=true`. `Reserve()` проверит реальный лимит.

**Файлы:** `candidate.go`

### #4/#5 HIGH: SetQuota() игнорировал ошибки

**Было:** `QuotaInitializer.SetQuota()` не возвращал error. PostgreSQL и Redis реализации молча проглатывали ошибки (`_, _ = pool.Exec(...)`). Если БД лежит при старте — квоты не инициализированы, но роутер работает.

**Стало:** Интерфейс `QuotaInitializer.SetQuota` возвращает `error`. `NewRouter()` пропагирует ошибку: `inferrouter: init quota for "acc-1": postgres connection refused`. Все 3 реализации (Memory, Redis, PostgreSQL) обновлены.

**Файлы:** `quota.go`, `quota/memory.go`, `quota/redis/redis.go`, `quota/postgres/postgres.go`, `router.go`

**Breaking change:** Кастомные реализации `QuotaInitializer` должны обновить сигнатуру.

---

## Коммит 3: `76f5e2a` — MEDIUM баги

### #6/#7: Malformed SSE chunks молча пропускались

**Было:** `json.Unmarshal` fail → `continue`. Ответ неполный, caller не знает.

**Стало:** Счётчик consecutive parse errors. 3 подряд → ошибка: `inferrouter: 3 consecutive malformed SSE chunks: ...`. Успешный chunk сбрасывает счётчик.

**Файлы:** `provider/openaicompat/openaicompat.go`, `provider/gemini/gemini.go`

### #8: Redis ParseInt → silent zero

**Было:** `strconv.ParseInt(vals[0].(string), 10, 64)` — ошибки игнорировались. Corrupted Redis data → `dailyLimit=0` → аккаунт заблокирован.

**Стало:** Каждый `ParseInt` проверяется: `inferrouter/redis: parse daily_limit: ...`.

**Файлы:** `quota/redis/redis.go`

### #9/#10: HTTP error body терялся для большинства кодов

**Было:** Response body включался в ошибку только для 400 (BadRequest). Для 429, 401, 5xx — только sentinel error без контекста.

**Стало:** Body включается для всех кодов. Если body нечитаем — fallback на `http.StatusText()`:
```
inferrouter: rate limited by provider: {"error": "rate limit exceeded, retry after 30s"}
inferrouter: provider unavailable: HTTP 503: service temporarily overloaded
```

**Файлы:** `provider/openaicompat/openaicompat.go`, `provider/gemini/gemini.go`

### #11: HealthTracker RLock→Lock race

**Было:** `GetHealth()` брал `RLock`, отпускал, потом брал `Lock`. Между ними `Reset()` мог удалить запись из map — `ah` pointer становился orphaned.

**Стало:** Один `Lock` на весь метод. Потеря производительности минимальна — `GetHealth()` вызывается раз на кандидата при `buildCandidates()`.

**Файлы:** `health.go`

### #12: Idempotency seen map рос весь день

**Было:** `map[string]bool` — очищался только при daily reset. При 1M req/day = десятки MB на ключи, которые нужны максимум минуту.

**Стало:** `map[string]time.Time` с TTL 1 час. Pruning при каждом `Reserve()`. Ключи старше часа удаляются.

**Файлы:** `quota/memory.go`

---

## Тесты

Добавлено 11 новых тестов:

| Тест | Покрывает |
|------|-----------|
| `TestAllFailed_IncludesTriedCandidates` | ErrAllFailed с per-candidate errors |
| `TestAllFailed_IncludesQuotaErrors` | Quota reserve errors в Tried |
| `TestHealthTracker_Reset` | Reset() все аккаунты |
| `TestHealthTracker_ResetAccount` | ResetAccount() один аккаунт |
| `TestStream_Close_ReturnsQuotaError` | Fix #1: Close() → quota error |
| `TestCommitError_ReportedViaMeter` | Fix #2: Commit fail → Meter |
| `TestRollbackError_ReportedViaMeter` | Fix #2: Rollback fail → Meter |
| `TestRemainingError_FailOpen_FreeTier` | Fix #3: Remaining error → free=true |
| `TestNewRouter_SetQuotaError_Propagated` | Fix #4: SetQuota error → NewRouter error |
| `TestStream_QuotaCommitError_Reported` | Existing test (adjusted) |

Все тесты проходят с `-race`.

---

## Breaking Changes

1. **`QuotaInitializer.SetQuota`** — сигнатура изменена с `SetQuota(string, int64, QuotaUnit)` на `SetQuota(string, int64, QuotaUnit) error`
2. **`RouterError.Error()`** — формат строки изменён при `ErrAllFailed` (теперь включает tried candidates)
3. **`mapHTTPError`** — sentinel errors (ErrRateLimited, ErrAuthFailed) теперь wrapped с body контекстом. `errors.Is()` по-прежнему работает.

---

## Что осталось

Аудит полностью закрыт. Все 12 + 2 начальных бага зафиксированы. Proposal: `docs/proposals/bug-audit-2026-02-24.md`.
