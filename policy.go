package inferrouter

// Policy selects and orders candidates for a given request.
type Policy interface {
	// Select orders candidates by priority. Returns ordered slice (highest priority first).
	Select(candidates []Candidate) []Candidate
}

// Candidate represents a possible route for a request.
type Candidate struct {
	Provider  Provider
	AccountID string
	Auth      Auth
	Model     string
	Free      bool
	Remaining int64     // remaining quota (tokens or requests)
	QuotaUnit QuotaUnit // unit of the quota
	Health    HealthState

	// CostPerToken is the cost in dollars per token for paid accounts.
	CostPerToken float64
}

// HealthState describes the health of a provider account.
type HealthState int

const (
	HealthHealthy  HealthState = iota
	HealthUnhealthy
	HealthHalfOpen
)

func (h HealthState) String() string {
	switch h {
	case HealthHealthy:
		return "healthy"
	case HealthUnhealthy:
		return "unhealthy"
	case HealthHalfOpen:
		return "half-open"
	default:
		return "unknown"
	}
}
