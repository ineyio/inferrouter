package openaicompat

import (
	"fmt"

	"github.com/ineyio/inferrouter"
)

// FromAccounts builds one Provider per distinct provider name among the
// accounts that set BaseURL, so a pool of OpenAI-compatible gateways can be
// declared entirely in config:
//
//	accounts:
//	  - provider: gonkagate
//	    id: gonkagate-main
//	    base_url: https://api.gonkagate.com/v1
//	    auth: {api_key: ${GONKAGATE_API_KEY}}
//	    ...
//
//	providers, err := openaicompat.FromAccounts(cfg.Accounts)
//	router, err := inferrouter.NewRouter(cfg, providers,
//	    inferrouter.WithPolicy(&policy.LeastBusyPolicy{}))
//
// Accounts without BaseURL are skipped (their providers are constructed in
// code). Multiple accounts may share one provider name and BaseURL — they
// become separate accounts on the same provider. Conflicting BaseURLs for
// the same provider name are a configuration error.
//
// opts apply to every provider in the pool (e.g. a shared WithHTTPClient
// with a generous timeout for slow gateways).
func FromAccounts(accounts []inferrouter.AccountConfig, opts ...Option) ([]inferrouter.Provider, error) {
	urls := make(map[string]string) // provider name → base URL
	var order []string              // deterministic provider order

	for _, acc := range accounts {
		if acc.BaseURL == "" {
			continue
		}
		if existing, ok := urls[acc.Provider]; ok {
			if existing != acc.BaseURL {
				return nil, fmt.Errorf("inferrouter: provider %q has conflicting base URLs: %q and %q",
					acc.Provider, existing, acc.BaseURL)
			}
			continue
		}
		urls[acc.Provider] = acc.BaseURL
		order = append(order, acc.Provider)
	}

	providers := make([]inferrouter.Provider, 0, len(order))
	for _, name := range order {
		providers = append(providers, New(name, urls[name], opts...))
	}
	return providers, nil
}
