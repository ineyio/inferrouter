package meter

import "github.com/ineyio/inferrouter"

// NoopMeter is a meter that does nothing.
type NoopMeter struct{}

var _ inferrouter.Meter = (*NoopMeter)(nil)

func (m *NoopMeter) OnRoute(inferrouter.RouteEvent)   {}
func (m *NoopMeter) OnResult(inferrouter.ResultEvent) {}
