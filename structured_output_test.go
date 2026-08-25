package inferrouter_test

import (
	"context"
	"encoding/json"
	"testing"

	ir "github.com/ineyio/inferrouter"
	"github.com/ineyio/inferrouter/provider/mock"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// SO-1: Routing.StructuredOutput is a report from the adapter, not an echo of
// the request.
//
// Both halves are asserted against the SAME asked-for format, because only the
// pair carries the claim. An adapter that ignores the field must report false
// while the caller asked true — that is the case a router deriving the flag
// from `req.ResponseFormat != nil` would get wrong, and it is exactly the shape
// of an endpoint that quietly drops the constraint and answers free text.
func TestStructuredOutput_AppliedIsFactNotIntent(t *testing.T) {
	format := &ir.ResponseFormat{
		Type: "json_schema",
		JSONSchema: &ir.JSONSchemaSpec{
			Name:   "output_schema",
			Schema: json.RawMessage(`{"type":"object"}`),
		},
	}

	cases := []struct {
		name string
		// honours reports what the adapter did with the field it received.
		honours bool
		want    bool
	}{
		{"adapter serialises the field", true, true},
		{"adapter ignores the field", false, false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var sawFormat bool
			prov := mock.New(
				mock.WithModels("test-model"),
				mock.WithResponseFunc(func(req ir.ProviderRequest) (ir.ProviderResponse, error) {
					sawFormat = req.ResponseFormat != nil
					return ir.ProviderResponse{
						ID:      "1",
						Content: "ok",
						Model:   "test-model",
						// An adapter that cannot express the constraint leaves
						// this alone; one that sent it sets it.
						StructuredOutputApplied: tc.honours && req.ResponseFormat != nil,
					}, nil
				}),
			)

			cfg := ir.Config{
				DefaultModel: "test-model",
				Accounts: []ir.AccountConfig{
					{Provider: "mock", ID: "acct-1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
				},
			}
			r := newTestRouter(t, cfg, []ir.Provider{prov})

			resp, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
				Messages:       []ir.Message{{Role: "user", Content: "hello"}},
				ResponseFormat: format,
			})
			require.NoError(t, err)

			// The router must hand the format down regardless of what the
			// adapter then does with it: without this, the false case below
			// would pass for the wrong reason.
			assert.True(t, sawFormat, "router did not pass ResponseFormat to the provider")
			assert.Equal(t, tc.want, resp.Routing.StructuredOutput)
		})
	}
}

// A request that asks for nothing reports nothing, and the provider is told
// nothing — the default path stays exactly as it was.
func TestStructuredOutput_AbsentByDefault(t *testing.T) {
	var sawFormat bool
	prov := mock.New(
		mock.WithModels("test-model"),
		mock.WithResponseFunc(func(req ir.ProviderRequest) (ir.ProviderResponse, error) {
			sawFormat = req.ResponseFormat != nil
			return ir.ProviderResponse{ID: "1", Content: "ok", Model: "test-model"}, nil
		}),
	)

	cfg := ir.Config{
		DefaultModel: "test-model",
		Accounts: []ir.AccountConfig{
			{Provider: "mock", ID: "acct-1", DailyFree: 1000, QuotaUnit: ir.QuotaTokens},
		},
	}
	r := newTestRouter(t, cfg, []ir.Provider{prov})

	resp, err := r.ChatCompletion(context.Background(), ir.ChatRequest{
		Messages: []ir.Message{{Role: "user", Content: "hello"}},
	})
	require.NoError(t, err)
	assert.False(t, sawFormat, "provider received a ResponseFormat nobody asked for")
	assert.False(t, resp.Routing.StructuredOutput)
}
