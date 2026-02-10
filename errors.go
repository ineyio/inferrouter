package inferrouter

import (
	"errors"
	"fmt"
)

// Sentinel errors.
var (
	ErrNoCandidates       = errors.New("inferrouter: no candidates available")
	ErrNoFreeQuota        = errors.New("inferrouter: no free quota remaining")
	ErrQuotaExceeded      = errors.New("inferrouter: quota exceeded")
	ErrRateLimited        = errors.New("inferrouter: rate limited by provider")
	ErrAuthFailed         = errors.New("inferrouter: authentication failed")
	ErrInvalidRequest     = errors.New("inferrouter: invalid request")
	ErrProviderUnavailable = errors.New("inferrouter: provider unavailable")
	ErrModelNotFound      = errors.New("inferrouter: model not found")
	ErrAllFailed          = errors.New("inferrouter: all candidates failed")
)

// RouterError wraps an error with routing context.
type RouterError struct {
	Err       error
	Provider  string
	AccountID string
	Model     string
	Attempts  int
}

func (e *RouterError) Error() string {
	return fmt.Sprintf("inferrouter: provider=%s account=%s model=%s attempts=%d: %v",
		e.Provider, e.AccountID, e.Model, e.Attempts, e.Err)
}

func (e *RouterError) Unwrap() error {
	return e.Err
}

// IsFatal returns true if the error should not be retried with another candidate.
func IsFatal(err error) bool {
	return errors.Is(err, ErrAuthFailed) || errors.Is(err, ErrInvalidRequest)
}

// IsRetryable returns true if the error can be retried with another candidate.
func IsRetryable(err error) bool {
	return errors.Is(err, ErrRateLimited) ||
		errors.Is(err, ErrProviderUnavailable) ||
		errors.Is(err, ErrQuotaExceeded)
}
