package inferrouter

// Rough per-modality token estimates calibrated against Gemini 2.5 behavior.
// Used only for quota pre-reservation sizing; actual usage is read back from
// the provider response and committed via QuotaStore.Commit.
const (
	charsPerTextToken  = 4
	tokensPerImage     = 560
	audioBytesPerToken = 1000 // ~32 tokens/sec for 32 kbps OGG
	videoBytesPerToken = 500
	perMessageOverhead = 4
	perRequestOverhead = 3
)

// EstimateTokens provides a rough token count estimate for messages.
// Handles both legacy Content strings and multi-part messages including
// image/audio/video; for media parts, byte-size heuristics are used.
func EstimateTokens(messages []Message) int64 {
	var total int64
	for _, m := range messages {
		if len(m.Parts) > 0 {
			for _, p := range m.Parts {
				total += estimatePartTokens(p)
			}
		} else {
			total += int64(len(m.Content)) / charsPerTextToken
		}
		total += perMessageOverhead
	}
	total += perRequestOverhead
	return total
}

func estimatePartTokens(p Part) int64 {
	switch p.Type {
	case PartText:
		return int64(len(p.Text)) / charsPerTextToken
	case PartImage:
		return tokensPerImage
	case PartAudio:
		return int64(len(p.Data)) / audioBytesPerToken
	case PartVideo:
		return int64(len(p.Data)) / videoBytesPerToken
	}
	return 0
}
