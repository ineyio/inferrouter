package policy

import (
	"sort"

	"github.com/ineyio/inferrouter"
)

// FreeFirstPolicy prioritizes free candidates (sorted by remaining DESC),
// then paid candidates (sorted by cost ASC).
type FreeFirstPolicy struct{}

var _ inferrouter.Policy = (*FreeFirstPolicy)(nil)

// Select orders candidates: free first (most remaining), then paid (cheapest).
func (p *FreeFirstPolicy) Select(candidates []inferrouter.Candidate) []inferrouter.Candidate {
	result := make([]inferrouter.Candidate, len(candidates))
	copy(result, candidates)

	sort.SliceStable(result, func(i, j int) bool {
		ci, cj := result[i], result[j]

		// Free before paid.
		if ci.Free != cj.Free {
			return ci.Free
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
