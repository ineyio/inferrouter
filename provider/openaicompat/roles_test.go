package openaicompat

import (
	"encoding/json"
	"strings"
	"testing"

	ir "github.com/ineyio/inferrouter"
)

// The other half of the role contract. Gemini has to move a system message to
// a field of its own; an OpenAI-shaped gateway keeps it in the list, and this
// adapter's job is to carry every role of the vocabulary through untouched.
//
// It walks ir.Roles rather than three literals, so a role added to the
// vocabulary without a place here fails a test instead of a request.
func TestBuildRequestCarriesEveryRoleVerbatim(t *testing.T) {
	p := New("test", "https://example.invalid")

	msgs := make([]ir.Message, 0, len(ir.Roles))
	for _, role := range ir.Roles {
		msgs = append(msgs, ir.Message{Role: role, Content: "текст " + role})
	}

	req := p.buildRequest(ir.ProviderRequest{Model: "m", Messages: msgs}, false)

	if len(req.Messages) != len(ir.Roles) {
		t.Fatalf("messages len = %d, want %d", len(req.Messages), len(ir.Roles))
	}
	for i, role := range ir.Roles {
		if req.Messages[i].Role != role {
			t.Errorf("messages[%d].Role = %q, want %q verbatim", i, req.Messages[i].Role, role)
		}
		if req.Messages[i].Content != "текст "+role {
			t.Errorf("messages[%d].Content = %q", i, req.Messages[i].Content)
		}
	}

	// And on the wire, where the gateway reads it.
	raw, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for _, role := range ir.Roles {
		if !strings.Contains(string(raw), `"role":"`+role+`"`) {
			t.Errorf("wire format lost role %q: %s", role, raw)
		}
	}
}
