package inferrouter

import (
	"context"
	"errors"
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

	if isSuccess {
		actualTokens := s.totalUsage.TotalTokens
		if s.candidate.QuotaUnit == QuotaRequests {
			actualTokens = 1
		}
		_ = s.quotaStore.Commit(context.Background(), s.reservation, actualTokens)
		s.health.RecordSuccess(s.candidate.AccountID)
	} else {
		_ = s.quotaStore.Rollback(context.Background(), s.reservation)
		s.health.RecordFailure(s.candidate.AccountID)
	}

	s.meter.OnResult(ResultEvent{
		Provider:  s.candidate.Provider.Name(),
		AccountID: s.candidate.AccountID,
		Model:     s.candidate.Model,
		Free:      s.candidate.Free,
		Success:   isSuccess,
		Duration:  duration,
		Usage:     s.totalUsage,
		Error:     s.streamErr,
	})

	return err
}
