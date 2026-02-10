package inferrouter

// EstimateTokens provides a rough token count estimate for messages.
// Uses the approximation: ~4 chars per token + overhead per message.
func EstimateTokens(messages []Message) int64 {
	var total int64
	for _, m := range messages {
		// ~4 chars per token
		total += int64(len(m.Content)) / 4
		// overhead per message (role, formatting)
		total += 4
	}
	// base overhead for the request
	total += 3
	return total
}
