package inferrouter

import "testing"

func TestPartIsMedia(t *testing.T) {
	cases := []struct {
		part Part
		want bool
	}{
		{Part{Type: PartText, Text: "hi"}, false},
		{Part{Type: PartImage, MIMEType: "image/jpeg", Data: []byte{1}}, true},
		{Part{Type: PartAudio, MIMEType: "audio/ogg", Data: []byte{1}}, true},
		{Part{Type: PartVideo, MIMEType: "video/mp4", Data: []byte{1}}, true},
		{Part{}, false},
	}
	for _, tc := range cases {
		if got := tc.part.IsMedia(); got != tc.want {
			t.Errorf("Part{Type=%q}.IsMedia() = %v, want %v", tc.part.Type, got, tc.want)
		}
	}
}

func TestMessagesHaveMedia(t *testing.T) {
	cases := []struct {
		name string
		msgs []Message
		want bool
	}{
		{
			name: "legacy Content only",
			msgs: []Message{{Role: "user", Content: "hi"}},
			want: false,
		},
		{
			name: "Parts with text only",
			msgs: []Message{{Role: "user", Parts: []Part{{Type: PartText, Text: "hi"}}}},
			want: false,
		},
		{
			name: "Parts with image",
			msgs: []Message{{Role: "user", Parts: []Part{
				{Type: PartText, Text: "describe this"},
				{Type: PartImage, MIMEType: "image/jpeg", Data: []byte{1, 2, 3}},
			}}},
			want: true,
		},
		{
			name: "media in second message",
			msgs: []Message{
				{Role: "user", Content: "hi"},
				{Role: "user", Parts: []Part{{Type: PartAudio, MIMEType: "audio/ogg", Data: []byte{1}}}},
			},
			want: true,
		},
		{
			name: "empty",
			msgs: nil,
			want: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := messagesHaveMedia(tc.msgs); got != tc.want {
				t.Errorf("messagesHaveMedia() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestUsageZeroValueBackwardCompat(t *testing.T) {
	// Zero-value Usage must still have working fields — no nil deref on new ones.
	u := Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}
	if u.CachedTokens != 0 {
		t.Errorf("zero Usage.CachedTokens = %d, want 0", u.CachedTokens)
	}
	if u.InputBreakdown != nil {
		t.Errorf("zero Usage.InputBreakdown = %v, want nil", u.InputBreakdown)
	}
}

func TestInputTokenBreakdownInvariant(t *testing.T) {
	// Text+Audio+Image+Video must equal PromptTokens when breakdown is non-nil.
	u := Usage{
		PromptTokens: 1234,
		InputBreakdown: &InputTokenBreakdown{
			Text:  100,
			Audio: 574,
			Image: 560,
			Video: 0,
		},
	}
	b := u.InputBreakdown
	sum := b.Text + b.Audio + b.Image + b.Video
	if sum != u.PromptTokens {
		t.Errorf("breakdown sum %d != PromptTokens %d", sum, u.PromptTokens)
	}
}
