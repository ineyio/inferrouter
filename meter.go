package inferrouter

import "time"

// Meter observes routing events for monitoring/logging.
type Meter interface {
	// OnRoute is called when a routing decision is made.
	OnRoute(event RouteEvent)

	// OnResult is called when a provider returns a result.
	OnResult(event ResultEvent)
}

// RouteEvent describes a routing decision.
type RouteEvent struct {
	Provider    string
	AccountID   string
	Model       string
	Free        bool
	AttemptNum  int
	EstimatedIn int64
}

// ResultEvent describes the outcome of a provider call.
type ResultEvent struct {
	Provider    string
	AccountID   string
	Model       string
	Free        bool
	Success     bool
	Duration    time.Duration
	Usage       Usage
	Error       error
}
