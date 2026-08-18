package policy

import (
	"math"
	"testing"

	ir "github.com/ineyio/inferrouter"
)

func ids(cands []ir.Candidate) []string {
	out := make([]string, len(cands))
	for i, c := range cands {
		out[i] = c.AccountID
	}
	return out
}

func TestFreeFirstPrefersFreeOverPaid(t *testing.T) {
	p := &FreeFirstPolicy{}
	in := []ir.Candidate{
		{AccountID: "paid-cheap", Free: false, CostPerToken: 0.00001},
		{AccountID: "free-low", Free: true, Remaining: 10},
		{AccountID: "paid-expensive", Free: false, CostPerToken: 0.001},
		{AccountID: "free-high", Free: true, Remaining: 500},
	}
	got := ids(p.Select(in))
	want := []string{"free-high", "free-low", "paid-cheap", "paid-expensive"}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("position %d: got %q, want %q (full=%v)", i, got[i], want[i], got)
		}
	}
}

func TestFreeFirstSortsFreeByRemainingDesc(t *testing.T) {
	p := &FreeFirstPolicy{}
	in := []ir.Candidate{
		{AccountID: "a", Free: true, Remaining: 100},
		{AccountID: "b", Free: true, Remaining: 300},
		{AccountID: "c", Free: true, Remaining: 200},
	}
	got := ids(p.Select(in))
	want := []string{"b", "c", "a"}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("pos %d: got %q, want %q", i, got[i], want[i])
		}
	}
}

func TestFreeFirstSortsPaidByCostAsc(t *testing.T) {
	p := &FreeFirstPolicy{}
	in := []ir.Candidate{
		{AccountID: "exp", Free: false, CostPerInputToken: 0.002, CostPerOutputToken: 0.006},
		{AccountID: "mid", Free: false, CostPerInputToken: 0.001, CostPerOutputToken: 0.003},
		{AccountID: "cheap", Free: false, CostPerToken: 0.0005},
	}
	got := ids(p.Select(in))
	want := []string{"cheap", "mid", "exp"}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("pos %d: got %q, want %q", i, got[i], want[i])
		}
	}
}

func TestFreeFirstStableForTies(t *testing.T) {
	p := &FreeFirstPolicy{}
	in := []ir.Candidate{
		{AccountID: "a", Free: true, Remaining: 100},
		{AccountID: "b", Free: true, Remaining: 100},
		{AccountID: "c", Free: true, Remaining: 100},
	}
	got := ids(p.Select(in))
	want := []string{"a", "b", "c"}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("stable sort broken: got %v, want %v", got, want)
		}
	}
}

func TestFreeFirstDoesNotMutateInput(t *testing.T) {
	p := &FreeFirstPolicy{}
	in := []ir.Candidate{
		{AccountID: "paid", Free: false, CostPerToken: 0.001},
		{AccountID: "free", Free: true, Remaining: 10},
	}
	_ = p.Select(in)
	if in[0].AccountID != "paid" || in[1].AccountID != "free" {
		t.Errorf("input mutated: %v", ids(in))
	}
}

func TestLeastBusyPrefersFewestInflight(t *testing.T) {
	p := &LeastBusyPolicy{}
	in := []ir.Candidate{
		{AccountID: "busy", Free: true, Remaining: 500, Inflight: 3},
		{AccountID: "idle", Free: true, Remaining: 100, Inflight: 0},
		{AccountID: "mid", Free: true, Remaining: 300, Inflight: 1},
	}
	got := ids(p.Select(in))
	want := []string{"idle", "mid", "busy"}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("pos %d: got %q, want %q (full=%v)", i, got[i], want[i], got)
		}
	}
}

func TestLeastBusyFreeBeforePaidRegardlessOfLoad(t *testing.T) {
	p := &LeastBusyPolicy{}
	in := []ir.Candidate{
		{AccountID: "paid-idle", Free: false, CostPerToken: 0.001, Inflight: 0},
		{AccountID: "free-busy", Free: true, Remaining: 10, Inflight: 5},
	}
	got := ids(p.Select(in))
	if got[0] != "free-busy" {
		t.Errorf("got %v, want free candidate first even when busy", got)
	}
}

func TestLeastBusyTiesFallBackToRemainingAndCost(t *testing.T) {
	p := &LeastBusyPolicy{}
	in := []ir.Candidate{
		{AccountID: "free-low", Free: true, Remaining: 10, Inflight: 1},
		{AccountID: "free-high", Free: true, Remaining: 500, Inflight: 1},
		{AccountID: "paid-exp", Free: false, CostPerToken: 0.01, Inflight: 2},
		{AccountID: "paid-cheap", Free: false, CostPerToken: 0.0001, Inflight: 2},
	}
	got := ids(p.Select(in))
	want := []string{"free-high", "free-low", "paid-cheap", "paid-exp"}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("pos %d: got %q, want %q (full=%v)", i, got[i], want[i], got)
		}
	}
}

func TestLeastBusyDoesNotMutateInput(t *testing.T) {
	p := &LeastBusyPolicy{}
	in := []ir.Candidate{
		{AccountID: "busy", Free: true, Inflight: 9},
		{AccountID: "idle", Free: true, Inflight: 0},
	}
	_ = p.Select(in)
	if in[0].AccountID != "busy" || in[1].AccountID != "idle" {
		t.Errorf("input mutated: %v", ids(in))
	}
}

func TestCostFirstOrdersByCost(t *testing.T) {
	p := &CostFirstPolicy{}
	in := []ir.Candidate{
		{AccountID: "b", CostPerToken: 0.001},
		{AccountID: "free", Free: true, CostPerToken: 0},
		{AccountID: "a", CostPerToken: 0.0001},
		{AccountID: "c", CostPerInputToken: 0.002, CostPerOutputToken: 0.004},
	}
	got := ids(p.Select(in))
	want := []string{"free", "a", "b", "c"}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("pos %d: got %q, want %q (full=%v)", i, got[i], want[i], got)
		}
	}
}

func TestBlendedCost(t *testing.T) {
	// New-style: blended = (in + 2*out) / 3
	c := ir.Candidate{CostPerInputToken: 0.001, CostPerOutputToken: 0.004}
	got := c.BlendedCost()
	want := (0.001 + 2*0.004) / 3
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("BlendedCost = %v, want %v", got, want)
	}

	// Legacy fallback: CostPerToken
	c = ir.Candidate{CostPerToken: 0.002}
	if c.BlendedCost() != 0.002 {
		t.Errorf("legacy BlendedCost = %v, want 0.002", c.BlendedCost())
	}
}

func TestEmptyInput(t *testing.T) {
	if got := (&FreeFirstPolicy{}).Select(nil); len(got) != 0 {
		t.Errorf("FreeFirst on nil = %v", got)
	}
	if got := (&CostFirstPolicy{}).Select(nil); len(got) != 0 {
		t.Errorf("CostFirst on nil = %v", got)
	}
	if got := (&LeastBusyPolicy{}).Select(nil); len(got) != 0 {
		t.Errorf("LeastBusy on nil = %v", got)
	}
}
