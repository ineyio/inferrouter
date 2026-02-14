package inferrouter

import (
	"context"
	"errors"
	"fmt"
	"io"
	"time"
)

// RouterStream wraps a ProviderStream with quota commit on close.
type RouterStream struct {
	inner       ProviderStream
	reservation Reservation
	quotaStore  QuotaStore
	meter       Meter
	health      *HealthTracker
	spend       *SpendTracker
	candidate   Candidate
	startTime   time.Time
	totalUsage  Usage
	closed      bool
	streamErr   error // first error encountered during streaming
}

// Next returns the next chunk from the stream.
func (s *RouterStream) Next() (StreamChunk, error) {
	chunk, err := s.inner.Next()
	if err != nil {
		if s.streamErr == nil {
			s.streamErr = err
		}
		return chunk, err
	}

	// Track usage from the final chunk.
	if chunk.Usage != nil {
		s.totalUsage = *chunk.Usage
	}

	return chunk, nil
}

// Close releases the stream and commits quota.
func (s *RouterStream) Close() error {
	if s.closed {
		return nil
	}
	s.closed = true

	err := s.inner.Close()
	duration := time.Since(s.startTime)

	// io.EOF is the normal end of stream, not an error.
	isSuccess := s.streamErr == nil || errors.Is(s.streamErr, io.EOF)

	var quotaErr error
	if isSuccess {
		actualTokens := s.totalUsage.TotalTokens
		if s.candidate.QuotaUnit == QuotaRequests {
			actualTokens = 1
		}
		quotaErr = s.quotaStore.Commit(context.Background(), s.reservation, actualTokens)
		s.health.RecordSuccess(s.candidate.AccountID)
	} else {
		quotaErr = s.quotaStore.Rollback(context.Background(), s.reservation)
		s.health.RecordFailure(s.candidate.AccountID)
	}

	var dollarCost float64
	if isSuccess {
		dollarCost = calculateSpend(s.candidate, s.totalUsage)
		if dollarCost > 0 {
			s.spend.RecordSpend(s.candidate.AccountID, dollarCost)
		}
	}

	resultErr := s.streamErr
	if quotaErr != nil && (resultErr == nil || errors.Is(resultErr, io.EOF)) {
		resultErr = fmt.Errorf("quota operation failed: %w", quotaErr)
	}

	s.meter.OnResult(ResultEvent{
		Provider:   s.candidate.Provider.Name(),
		AccountID:  s.candidate.AccountID,
		Model:      s.candidate.Model,
		Free:       s.candidate.Free,
		Success:    isSuccess && quotaErr == nil,
		Duration:   duration,
		Usage:      s.totalUsage,
		Error:      resultErr,
		DollarCost: dollarCost,
	})

	return err
}
