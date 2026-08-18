package policy

import (
	"sort"

	"github.com/ineyio/inferrouter"
)

// LeastBusyPolicy spreads concurrent requests across accounts by preferring
// the candidate with the fewest in-flight requests. Designed for pools of
// slow gateways (e.g. Gonka resellers at 3-20s per request), where a
// deterministic sort would funnel every parallel request to the same
// endpoint.
//
// Ordering: free before paid; within each group fewest in-flight first;
// ties broken by most remaining quota (free) or lowest blended cost (paid).
//
// Best-effort: a burst of simultaneous requests may read the same in-flight
// counts before any of them increments, so the very first wave can cluster
// on one account. Counters become accurate as soon as requests are
// dispatched, which is what matters at multi-second latencies.
type LeastBusyPolicy struct{}

var _ inferrouter.Policy = (*LeastBusyPolicy)(nil)

// Select orders candidates: free first, then least busy, then quota/cost.
func (p *LeastBusyPolicy) Select(candidates []inferrouter.Candidate) []inferrouter.Candidate {
	result := make([]inferrouter.Candidate, len(candidates))
	copy(result, candidates)

	sort.SliceStable(result, func(i, j int) bool {
		ci, cj := result[i], result[j]

		// Free before paid.
		if ci.Free != cj.Free {
			return ci.Free
		}

		// Least busy first.
		if ci.Inflight != cj.Inflight {
			return ci.Inflight < cj.Inflight
		}

		if ci.Free {
			// Among free: most remaining first.
			return ci.Remaining > cj.Remaining
		}

		// Among paid: cheapest first.
		return ci.BlendedCost() < cj.BlendedCost()
	})

	return result
}
