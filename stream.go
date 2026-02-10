package inferrouter

import (
	"context"
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
}

// Next returns the next chunk from the stream.
func (s *RouterStream) Next() (StreamChunk, error) {
	chunk, err := s.inner.Next()
	if err != nil {
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

	// Commit quota with actual usage.
	actualTokens := s.totalUsage.TotalTokens
	if s.candidate.QuotaUnit == QuotaRequests {
		actualTokens = 1
	}
	_ = s.quotaStore.Commit(context.Background(), s.reservation, actualTokens)

	s.health.RecordSuccess(s.candidate.AccountID)
	s.meter.OnResult(ResultEvent{
		Provider:  s.candidate.Provider.Name(),
		AccountID: s.candidate.AccountID,
		Model:     s.candidate.Model,
		Free:      s.candidate.Free,
		Success:   true,
		Duration:  time.Since(s.startTime),
		Usage:     s.totalUsage,
	})

	return err
}
