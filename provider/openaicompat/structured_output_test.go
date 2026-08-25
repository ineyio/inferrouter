package openaicompat

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	ir "github.com/ineyio/inferrouter"
)

// captureBody stands up an endpoint that records the exact bytes it was sent
// and answers a minimal completion.
//
// The bytes matter, not a decoded struct: the claims these tests make are about
// what a gateway receives, and decoding would erase the difference between an
// absent key and a null one — which is the whole of TestBuildRequest_OmitsResponseFormat.
func captureBody(t *testing.T) (*Provider, *[]byte) {
	t.Helper()
	var seen []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
		}
		seen = b
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"1","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(srv.Close)
	return New("test", srv.URL), &seen
}

// SO-2: no format asked → the key is not on the wire at all.
func TestBuildRequest_OmitsResponseFormat(t *testing.T) {
	p, seen := captureBody(t)

	resp, err := p.ChatCompletion(context.Background(), ir.ProviderRequest{
		Model:    "m",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}

	if bytes.Contains(*seen, []byte("response_format")) {
		t.Errorf("request carries response_format when none was asked for: %s", *seen)
	}
	if resp.StructuredOutputApplied {
		t.Error("StructuredOutputApplied is true for a request that carried no format")
	}
}

// SO-3: the schema reaches the gateway byte for byte, under json_schema, and
// strict stays off.
//
// The fixture schema is deliberately awkward — an unusual key order and a key
// this library has no opinion about — so that any re-marshalling on the way out
// shows up as a difference instead of being normalised into looking correct.
func TestBuildRequest_CarriesClientSchemaVerbatim(t *testing.T) {
	p, seen := captureBody(t)

	schema := json.RawMessage(`{"type":"object","x-client-note":"keep me","properties":{"facts":{"type":"array"}}}`)

	resp, err := p.ChatCompletion(context.Background(), ir.ProviderRequest{
		Model:    "m",
		Messages: []ir.Message{{Role: "user", Content: "hi"}},
		ResponseFormat: &ir.ResponseFormat{
			Type: "json_schema",
			JSONSchema: &ir.JSONSchemaSpec{
				Name:   "output_schema",
				Schema: schema,
			},
		},
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}

	if !resp.StructuredOutputApplied {
		t.Error("StructuredOutputApplied is false although the format went out")
	}

	var body struct {
		ResponseFormat *struct {
			Type       string `json:"type"`
			JSONSchema *struct {
				Name   string          `json:"name"`
				Strict *bool           `json:"strict"`
				Schema json.RawMessage `json:"schema"`
			} `json:"json_schema"`
		} `json:"response_format"`
	}
	if err := json.Unmarshal(*seen, &body); err != nil {
		t.Fatalf("decode captured body: %v (%s)", err, *seen)
	}
	if body.ResponseFormat == nil || body.ResponseFormat.JSONSchema == nil {
		t.Fatalf("response_format missing from the request: %s", *seen)
	}
	if got := body.ResponseFormat.Type; got != "json_schema" {
		t.Errorf("response_format.type = %q, want json_schema", got)
	}
	if got := body.ResponseFormat.JSONSchema.Name; got != "output_schema" {
		t.Errorf("json_schema.name = %q, want output_schema", got)
	}
	if s := body.ResponseFormat.JSONSchema.Strict; s != nil && *s {
		t.Error("strict was sent as true — a client schema outside the provider's restricted subset would then 400")
	}
	if got := string(body.ResponseFormat.JSONSchema.Schema); got != string(schema) {
		t.Errorf("schema was rewritten on the way out:\n got: %s\nwant: %s", got, schema)
	}
}
