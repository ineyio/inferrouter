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

	// ErrNoEmbeddingProviders is returned by Router.Embed/EmbedBatch when no
	// configured provider implements EmbeddingProvider for the requested model.
	// Symmetric to ErrMultimodalUnavailable — a specific failure mode distinct
	// from generic ErrNoCandidates.
	ErrNoEmbeddingProviders = errors.New("inferrouter: no embedding providers for model")

	// ErrBatchTooLarge is returned by Router.Embed (single-call path, NOT
	// EmbedBatch) when len(req.Inputs) exceeds the selected provider's
	// MaxBatchSize. Callers should use EmbedBatch for automatic splitting.
	ErrBatchTooLarge = errors.New("inferrouter: batch exceeds provider max size")

	// ErrInvalidConfig is returned by NewRouter for structural config problems
	// that cannot be expressed via YAML schema alone (e.g. embedding alias
	// with multiple models — see RFC §3.6 single-model invariant).
	ErrInvalidConfig = errors.New("inferrouter: invalid config")
)

// ErrPartialBatch is returned by Router.EmbedBatch when the operation
// successfully processed some inputs before encountering an unrecoverable
// error on a later batch.
//
// Contract (critical for consumer correctness):
//
//   - ProcessedInputs is the exact count of successfully processed inputs
//     from the start of req.Inputs. Ordering is preserved.
//   - The accompanying EmbedResponse (returned via multi-return alongside
//     this error) contains Embeddings[0..ProcessedInputs-1] — valid vectors
//     for req.Inputs[0..ProcessedInputs-1], in original order.
//   - Usage reflects actual tokens consumed on the successful part only.
//   - Quota reservations for successful batches are COMMITTED; the failing
//     batch's reservation is ROLLED BACK; unattempted remainder is not
//     reserved. Consumer pays only for successful work.
//
// Consumer retry pattern:
//
//	resp, err := router.EmbedBatch(ctx, req)
//	var partial *ErrPartialBatch
//	if errors.As(err, &partial) {
//	    persist(resp.Embeddings) // valid prefix
//	    return retryWith(req.Inputs[partial.ProcessedInputs:])
//	}
type ErrPartialBatch struct {
	ProcessedInputs int
	Cause           error
}

func (e *ErrPartialBatch) Error() string {
	return fmt.Sprintf("inferrouter: partial batch failure after %d inputs: %v",
		e.ProcessedInputs, e.Cause)
}

func (e *ErrPartialBatch) Unwrap() error { return e.Cause }

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
