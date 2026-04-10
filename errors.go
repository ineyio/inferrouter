package inferrouter

import (
	"errors"
	"fmt"
)

// Sentinel errors.
var (
	ErrNoCandidates        = errors.New("inferrouter: no candidates available")
	ErrNoFreeQuota         = errors.New("inferrouter: no free quota remaining")
	ErrQuotaExceeded       = errors.New("inferrouter: quota exceeded")
	ErrRateLimited         = errors.New("inferrouter: rate limited by provider")
	ErrAuthFailed          = errors.New("inferrouter: authentication failed")
	ErrInvalidRequest      = errors.New("inferrouter: invalid request")
	ErrProviderUnavailable = errors.New("inferrouter: provider unavailable")
	ErrModelNotFound       = errors.New("inferrouter: model not found")
	ErrAllFailed           = errors.New("inferrouter: all candidates failed")
	ErrRPMExceeded         = errors.New("inferrouter: requests per minute limit exceeded")

	// ErrMultimodalUnavailable is returned when a request contains media parts
	// but no multimodal-capable candidate is available (all filtered out or
	// unhealthy). Not retryable with text-only fallback — callers should catch
	// this explicitly and either degrade (strip media) or fail the request.
	ErrMultimodalUnavailable = errors.New("inferrouter: no multimodal-capable candidates available")
)

// CandidateError records the error from a single candidate attempt.
type CandidateError struct {
	Provider  string
	AccountID string
	Model     string
	Err       error
}

func (e *CandidateError) Error() string {
	return fmt.Sprintf("provider=%s account=%s model=%s: %v",
		e.Provider, e.AccountID, e.Model, e.Err)
}

func (e *CandidateError) Unwrap() error {
	return e.Err
}

// RouterError wraps an error with routing context.
type RouterError struct {
	Err       error
	Provider  string
	AccountID string
	Model     string
	Attempts  int
	Tried     []CandidateError // per-candidate errors (populated on ErrAllFailed)
}

func (e *RouterError) Error() string {
	if len(e.Tried) > 0 {
		msg := fmt.Sprintf("inferrouter: all %d candidates failed:", e.Attempts)
		for _, t := range e.Tried {
			msg += fmt.Sprintf(" [%s]", t.Error())
		}
		return msg
	}
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
		errors.Is(err, ErrQuotaExceeded) ||
		errors.Is(err, ErrRPMExceeded)
}
