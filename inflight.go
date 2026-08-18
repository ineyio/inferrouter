package inferrouter

import (
	"sync"
	"sync/atomic"
)

// InflightTracker counts requests currently executing per account.
// With slow backends (seconds per request) the in-flight count is the
// strongest signal for spreading concurrent requests across a pool of
// gateways — see policy.LeastBusyPolicy.
//
// Best-effort by design: counters are process-local and advisory. They
// influence candidate ordering but never block a request, so the tracker
// is fail-open and needs no persistence.
type InflightTracker struct {
	counters sync.Map // accountID → *atomic.Int64
}

// NewInflightTracker creates an empty tracker.
func NewInflightTracker() *InflightTracker {
	return &InflightTracker{}
}

func (t *InflightTracker) counter(accountID string) *atomic.Int64 {
	if c, ok := t.counters.Load(accountID); ok {
		return c.(*atomic.Int64)
	}
	c, _ := t.counters.LoadOrStore(accountID, new(atomic.Int64))
	return c.(*atomic.Int64)
}

// Inc registers the start of a request for the account.
func (t *InflightTracker) Inc(accountID string) {
	t.counter(accountID).Add(1)
}

// Dec registers the end of a request for the account.
func (t *InflightTracker) Dec(accountID string) {
	t.counter(accountID).Add(-1)
}

// Get returns the current in-flight count for the account.
func (t *InflightTracker) Get(accountID string) int64 {
	return t.counter(accountID).Load()
}
