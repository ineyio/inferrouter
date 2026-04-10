package inferrouter

import "testing"

func TestEstimateTokensEmpty(t *testing.T) {
	// Only base per-request overhead.
	if got := EstimateTokens(nil); got != perRequestOverhead {
		t.Errorf("empty = %d, want %d", got, perRequestOverhead)
	}
}

func TestEstimateTokensLegacyContent(t *testing.T) {
	// Regression guard for the pre-multimodal path: every Message with
	// Content gets len(Content)/4 + perMessageOverhead, plus per-request overhead.
	msgs := []Message{
		{Role: "user", Content: "hello world"}, // 11 chars / 4 = 2
		{Role: "assistant", Content: "hi"},     // 2 / 4 = 0
	}
	got := EstimateTokens(msgs)
	want := int64(2+perMessageOverhead) + int64(0+perMessageOverhead) + perRequestOverhead
	if got != want {
		t.Errorf("got %d, want %d", got, want)
	}
}

func TestEstimateTokensTextPart(t *testing.T) {
	// A single text part should behave identically to legacy Content.
	msgs := []Message{
		{Role: "user", Parts: []Part{{Type: PartText, Text: "hello world"}}},
	}
	got := EstimateTokens(msgs)
	want := int64(2+perMessageOverhead) + perRequestOverhead
	if got != want {
		t.Errorf("got %d, want %d", got, want)
	}
}

func TestEstimateTokensImagePart(t *testing.T) {
	msgs := []Message{
		{Role: "user", Parts: []Part{
			{Type: PartImage, MIMEType: "image/jpeg", Data: []byte{1, 2, 3}},
		}},
	}
	got := EstimateTokens(msgs)
	want := int64(tokensPerImage+perMessageOverhead) + perRequestOverhead
	if got != want {
		t.Errorf("got %d, want %d", got, want)
	}
}

func TestEstimateTokensAudioPart(t *testing.T) {
	// 16000 bytes / audioBytesPerToken = 16 tokens.
	audioData := make([]byte, 16000)
	msgs := []Message{
		{Role: "user", Parts: []Part{
			{Type: PartAudio, MIMEType: "audio/ogg", Data: audioData},
		}},
	}
	got := EstimateTokens(msgs)
	want := int64(16+perMessageOverhead) + perRequestOverhead
	if got != want {
		t.Errorf("got %d, want %d", got, want)
	}
}

func TestEstimateTokensVideoPart(t *testing.T) {
	videoData := make([]byte, 5000) // 5000 / videoBytesPerToken = 10
	msgs := []Message{
		{Role: "user", Parts: []Part{
			{Type: PartVideo, MIMEType: "video/mp4", Data: videoData},
		}},
	}
	got := EstimateTokens(msgs)
	want := int64(10+perMessageOverhead) + perRequestOverhead
	if got != want {
		t.Errorf("got %d, want %d", got, want)
	}
}

func TestEstimateTokensMixedParts(t *testing.T) {
	// Text (12 chars = 3) + image (560) + audio (2000 bytes = 2) in one message.
	msgs := []Message{
		{Role: "user", Parts: []Part{
			{Type: PartText, Text: "describe me?"},
			{Type: PartImage, MIMEType: "image/jpeg", Data: []byte{1}},
			{Type: PartAudio, MIMEType: "audio/ogg", Data: make([]byte, 2000)},
		}},
	}
	got := EstimateTokens(msgs)
	want := int64(3+tokensPerImage+2+perMessageOverhead) + perRequestOverhead
	if got != want {
		t.Errorf("got %d, want %d", got, want)
	}
}

func TestEstimateTokensPartsIgnoreContent(t *testing.T) {
	// When Parts is non-empty, Content is not counted — Parts takes precedence.
	// This matches the Message contract documented in types.go.
	msgs := []Message{
		{
			Role:    "user",
			Content: "this very long content should be ignored because Parts wins",
			Parts:   []Part{{Type: PartText, Text: "hi"}},
		},
	}
	got := EstimateTokens(msgs)
	// Only the Parts text counts: 2/4 = 0.
	want := int64(0+perMessageOverhead) + perRequestOverhead
	if got != want {
		t.Errorf("got %d, want %d", got, want)
	}
}

func TestEstimateTokensLargeMedia(t *testing.T) {
	// A 20 MB voice file should land around 20_000 tokens, not near zero.
	// This is the guard against the bug the review caught.
	twentyMB := make([]byte, 20*1024*1024)
	msgs := []Message{
		{Role: "user", Parts: []Part{
			{Type: PartAudio, MIMEType: "audio/ogg", Data: twentyMB},
		}},
	}
	got := EstimateTokens(msgs)
	if got < 20000 {
		t.Errorf("20 MB audio estimated at %d tokens, expected ~20k", got)
	}
}

func TestEstimatePartTokensUnknownType(t *testing.T) {
	// Defensive: unknown part type returns 0, does not panic.
	p := Part{Type: PartType("unknown"), Text: "ignored"}
	if got := estimatePartTokens(p); got != 0 {
		t.Errorf("unknown part = %d, want 0", got)
	}
}
