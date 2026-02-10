package meter

import (
	"log/slog"

	"github.com/ineyio/inferrouter"
)

// LogMeter logs routing events using slog.
type LogMeter struct {
	Logger *slog.Logger
}

var _ inferrouter.Meter = (*LogMeter)(nil)

// NewLogMeter creates a LogMeter with the given logger.
// If logger is nil, slog.Default() is used.
func NewLogMeter(logger *slog.Logger) *LogMeter {
	if logger == nil {
		logger = slog.Default()
	}
	return &LogMeter{Logger: logger}
}

func (m *LogMeter) OnRoute(e inferrouter.RouteEvent) {
	m.Logger.Info("route",
		"provider", e.Provider,
		"account", e.AccountID,
		"model", e.Model,
		"free", e.Free,
		"attempt", e.AttemptNum,
		"estimated_tokens", e.EstimatedIn,
	)
}

func (m *LogMeter) OnResult(e inferrouter.ResultEvent) {
	if e.Success {
		m.Logger.Info("result",
			"provider", e.Provider,
			"account", e.AccountID,
			"model", e.Model,
			"free", e.Free,
			"duration_ms", e.Duration.Milliseconds(),
			"prompt_tokens", e.Usage.PromptTokens,
			"completion_tokens", e.Usage.CompletionTokens,
		)
	} else {
		m.Logger.Warn("result_error",
			"provider", e.Provider,
			"account", e.AccountID,
			"model", e.Model,
			"free", e.Free,
			"duration_ms", e.Duration.Milliseconds(),
			"error", e.Error,
		)
	}
}
