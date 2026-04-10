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
		attrs := make([]any, 0, 18)
		attrs = append(attrs,
			"provider", e.Provider,
			"account", e.AccountID,
			"model", e.Model,
			"free", e.Free,
			"duration_ms", e.Duration.Milliseconds(),
			"prompt_tokens", e.Usage.PromptTokens,
			"completion_tokens", e.Usage.CompletionTokens,
			"dollar_cost", e.DollarCost,
		)
		// Emit multimodal fields only when non-zero so text-only providers
		// produce no log-shape change for existing parsers.
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
