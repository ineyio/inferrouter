package openaicompat

import (
	"strings"
	"testing"

	"github.com/ineyio/inferrouter"
)

func TestFromAccountsBuildsOneProviderPerName(t *testing.T) {
	accounts := []inferrouter.AccountConfig{
		{Provider: "gonkagate", ID: "gg-1", BaseURL: "https://api.gonkagate.com/v1"},
		{Provider: "gonkagate", ID: "gg-2", BaseURL: "https://api.gonkagate.com/v1"}, // second account, same gateway
		{Provider: "othergate", ID: "og-1", BaseURL: "https://api.othergate.io/v1"},
	}
	providers, err := FromAccounts(accounts)
	if err != nil {
		t.Fatalf("FromAccounts: %v", err)
	}
	if len(providers) != 2 {
		t.Fatalf("len = %d, want 2 (accounts dedup to one provider per name)", len(providers))
	}
	if providers[0].Name() != "gonkagate" || providers[1].Name() != "othergate" {
		t.Errorf("order = [%s %s], want first-appearance order [gonkagate othergate]",
			providers[0].Name(), providers[1].Name())
	}
}

func TestFromAccountsSkipsAccountsWithoutBaseURL(t *testing.T) {
	accounts := []inferrouter.AccountConfig{
		{Provider: "cerebras", ID: "cb-1"}, // constructed in code, no base_url
		{Provider: "gonkagate", ID: "gg-1", BaseURL: "https://api.gonkagate.com/v1"},
	}
	providers, err := FromAccounts(accounts)
	if err != nil {
		t.Fatalf("FromAccounts: %v", err)
	}
	if len(providers) != 1 || providers[0].Name() != "gonkagate" {
		t.Errorf("got %d providers, want only gonkagate", len(providers))
	}
}

func TestFromAccountsRejectsConflictingBaseURLs(t *testing.T) {
	accounts := []inferrouter.AccountConfig{
		{Provider: "gonkagate", ID: "gg-1", BaseURL: "https://api.gonkagate.com/v1"},
		{Provider: "gonkagate", ID: "gg-2", BaseURL: "https://other.example.com/v1"},
	}
	_, err := FromAccounts(accounts)
	if err == nil {
		t.Fatal("expected conflict error")
	}
	if !strings.Contains(err.Error(), "conflicting base URLs") {
		t.Errorf("error = %v, want mention of conflicting base URLs", err)
	}
}

func TestFromAccountsEmpty(t *testing.T) {
	providers, err := FromAccounts(nil)
	if err != nil {
		t.Fatalf("FromAccounts(nil): %v", err)
	}
	if len(providers) != 0 {
		t.Errorf("len = %d, want 0", len(providers))
	}
}

func TestFromAccountsTrimsTrailingSlashViaNew(t *testing.T) {
	// New() trims trailing slashes; pool must inherit that behavior.
	accounts := []inferrouter.AccountConfig{
		{Provider: "gonkagate", ID: "gg-1", BaseURL: "https://api.gonkagate.com/v1/"},
	}
	providers, err := FromAccounts(accounts)
	if err != nil {
		t.Fatalf("FromAccounts: %v", err)
	}
	p, ok := providers[0].(*Provider)
	if !ok {
		t.Fatalf("provider type = %T, want *Provider", providers[0])
	}
	if p.baseURL != "https://api.gonkagate.com/v1" {
		t.Errorf("baseURL = %q, want trailing slash trimmed", p.baseURL)
	}
}
