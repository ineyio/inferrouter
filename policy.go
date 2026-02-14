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

	// Deprecated: use CostPerInputToken/CostPerOutputToken.
	CostPerToken float64

	CostPerInputToken  float64
	CostPerOutputToken float64

	MaxDailySpend float64 // max daily dollar spend (0 = unlimited)
	CurrentSpend  float64 // current daily dollar spend
}

// BlendedCost returns an estimated cost per token for sorting.
// Assumes ~3:1 input:output ratio typical for chat.
func (c Candidate) BlendedCost() float64 {
	if c.CostPerInputToken > 0 || c.CostPerOutputToken > 0 {
		return (c.CostPerInputToken + 2*c.CostPerOutputToken) / 3
	}
	return c.CostPerToken
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
