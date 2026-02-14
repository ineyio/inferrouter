package policy

import (
	"sort"

	"github.com/ineyio/inferrouter"
)

// CostFirstPolicy prioritizes candidates by cost (cheapest first).
// Free candidates (cost=0) naturally come first.
type CostFirstPolicy struct{}

var _ inferrouter.Policy = (*CostFirstPolicy)(nil)

// Select orders candidates by cost per token ascending.
func (p *CostFirstPolicy) Select(candidates []inferrouter.Candidate) []inferrouter.Candidate {
	result := make([]inferrouter.Candidate, len(candidates))
	copy(result, candidates)

	sort.SliceStable(result, func(i, j int) bool {
		return result[i].BlendedCost() < result[j].BlendedCost()
	})

	return result
}
