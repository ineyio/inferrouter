package gemini_test

import (
	"context"
	"fmt"
	"math"
	"os"
	"sort"
	"testing"
	"time"

	ir "github.com/ineyio/inferrouter"
	"github.com/ineyio/inferrouter/provider/gemini"
	"github.com/ineyio/inferrouter/quota"
)

// TestSmoke_RuWikipediaRetrieval exercises the full embedding path against
// the real Gemini API. It is a manual smoke test, not part of CI — auto-skips
// if GEMINI_API_KEY is not set in the environment.
//
// What it verifies end-to-end:
//   - Live API contract matches our request/response marshaling
//   - Auth query param propagation works
//   - Batch endpoint returns correctly-sized embeddings array
//   - Task type propagation (RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY produces
//     different vectors for the same text — asymmetric retrieval)
//   - Cosine similarity retrieval quality on RU text is good enough to
//     distinguish between clearly different topics
//
// Run manually:
//
//	GEMINI_API_KEY=... go test -v -run TestSmoke_RuWikipediaRetrieval ./provider/gemini/
//
// Expected quality baseline: the top-1 retrieval result for each RU query
// must match its intended document. The test fails if any query retrieves
// the wrong document as #1.
func TestSmoke_RuWikipediaRetrieval(t *testing.T) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		t.Skip("GEMINI_API_KEY not set; skipping real-API smoke test")
	}

	// Short RU Wikipedia-style fragments on three clearly distinct topics.
	// Chosen so each pair is topically non-overlapping, making retrieval
	// misses unambiguous rather than edge-case.
	docs := []struct {
		name string
		text string
	}{
		{
			"pushkin",
			"Александр Сергеевич Пушкин (1799–1837) — русский поэт, драматург и прозаик, " +
				"заложивший основы русского реалистического направления, литературный критик " +
				"и теоретик литературы, историк, публицист. Пушкин считается одним из " +
				"самых авторитетных литературных деятелей первой трети XIX века.",
		},
		{
			"napoleon",
			"Наполеон I Бонапарт (1769–1821) — император французов в 1804–1814 и 1815 годах, " +
				"французский полководец и государственный деятель, заложивший основы " +
				"современного французского государства. Начал службу в артиллерии " +
				"и быстро продвинулся в период Великой французской революции.",
		},
		{
			"wwii",
			"Вторая мировая война (1939–1945) — война двух мировых военно-политических " +
				"коалиций, ставшая самой масштабной в истории человечества. В ней " +
				"участвовали 62 государства из 74 существовавших на тот момент. Боевые " +
				"действия велись на территории Европы, Азии, Африки, а также в Мировом океане.",
		},
	}

	// RU queries with the expected top-1 document name.
	queries := []struct {
		query       string
		expectedDoc string
	}{
		{"кто такой Пушкин", "pushkin"},
		{"император Франции в начале XIX века", "napoleon"},
		{"сколько стран участвовало во Второй мировой", "wwii"},
		{"русский поэт XIX века", "pushkin"},
		{"крупнейшая война в истории", "wwii"},
	}

	// Build router with a single real Gemini account.
	cfg := ir.Config{
		Models: []ir.ModelMapping{
			{
				Alias: "gemini-embedding-001",
				Models: []ir.ModelRef{
					{Provider: "gemini", Model: "gemini-embedding-001"},
				},
			},
		},
		Accounts: []ir.AccountConfig{
			{
				Provider:                   "gemini",
				ID:                         "gemini-smoke",
				Auth:                       ir.Auth{APIKey: apiKey},
				DailyFree:                  1500,
				QuotaUnit:                  ir.QuotaRequests,
				CostPerEmbeddingInputToken: 0,
			},
		},
	}
	qs := quota.NewMemoryQuotaStore()
	router, err := ir.NewRouter(cfg, []ir.Provider{gemini.New()}, ir.WithQuotaStore(qs))
	if err != nil {
		t.Fatalf("NewRouter failed: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Index documents with RETRIEVAL_DOCUMENT task type.
	docTexts := make([]string, len(docs))
	for i, d := range docs {
		docTexts[i] = d.text
	}
	docResp, err := router.EmbedBatch(ctx, ir.EmbedRequest{
		Model:    "gemini-embedding-001",
		Inputs:   docTexts,
		TaskType: "RETRIEVAL_DOCUMENT",
	})
	if err != nil {
		t.Fatalf("EmbedBatch (docs) failed: %v", err)
	}
	if len(docResp.Embeddings) != len(docs) {
		t.Fatalf("expected %d doc embeddings, got %d", len(docs), len(docResp.Embeddings))
	}
	dims := len(docResp.Embeddings[0])
	t.Logf("indexed %d docs, %d dims, routed=%s/%s",
		len(docs), dims, docResp.Routing.Provider, docResp.Routing.AccountID)

	// Embed queries with RETRIEVAL_QUERY task type (asymmetric retrieval).
	queryTexts := make([]string, len(queries))
	for i, q := range queries {
		queryTexts[i] = q.query
	}
	queryResp, err := router.EmbedBatch(ctx, ir.EmbedRequest{
		Model:    "gemini-embedding-001",
		Inputs:   queryTexts,
		TaskType: "RETRIEVAL_QUERY",
	})
	if err != nil {
		t.Fatalf("EmbedBatch (queries) failed: %v", err)
	}
	if len(queryResp.Embeddings) != len(queries) {
		t.Fatalf("expected %d query embeddings, got %d", len(queries), len(queryResp.Embeddings))
	}

	// For each query, compute cosine similarity against all docs and verify
	// that the top-1 match is the expected document.
	type scored struct {
		docName string
		score   float32
	}
	t.Log("retrieval results:")
	var failures int
	for qi, q := range queries {
		qVec := queryResp.Embeddings[qi]
		ranked := make([]scored, len(docs))
		for di, d := range docs {
			ranked[di] = scored{
				docName: d.name,
				score:   cosineSim(qVec, docResp.Embeddings[di]),
			}
		}
		sort.Slice(ranked, func(a, b int) bool { return ranked[a].score > ranked[b].score })

		top := ranked[0]
		status := "OK"
		if top.docName != q.expectedDoc {
			status = "FAIL"
			failures++
		}
		t.Logf("  [%s] %q → top1=%s (%.3f) top2=%s (%.3f) top3=%s (%.3f) expected=%s",
			status, q.query,
			ranked[0].docName, ranked[0].score,
			ranked[1].docName, ranked[1].score,
			ranked[2].docName, ranked[2].score,
			q.expectedDoc)
	}

	if failures > 0 {
		t.Errorf("%d/%d queries retrieved wrong top-1 document", failures, len(queries))
	}

	// Also verify that doc[i] embedded as RETRIEVAL_DOCUMENT is NOT identical
	// to doc[i] embedded as RETRIEVAL_QUERY — Gemini applies an asymmetric
	// transformation, so the vectors should differ meaningfully. If they were
	// identical, our TaskType propagation would be broken somewhere.
	asymResp, err := router.EmbedBatch(ctx, ir.EmbedRequest{
		Model:    "gemini-embedding-001",
		Inputs:   []string{docs[0].text},
		TaskType: "RETRIEVAL_QUERY",
	})
	if err != nil {
		t.Fatalf("asymmetric check failed: %v", err)
	}
	asymSim := cosineSim(docResp.Embeddings[0], asymResp.Embeddings[0])
	t.Logf("asymmetric check: cosine(doc as DOCUMENT, doc as QUERY) = %.4f", asymSim)
	if asymSim > 0.9999 {
		t.Errorf("RETRIEVAL_DOCUMENT and RETRIEVAL_QUERY produced identical vectors (sim=%.4f); "+
			"TaskType propagation may be broken", asymSim)
	}
}

func cosineSim(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(normA) * math.Sqrt(normB)))
}

// Also provide an explicit skip message when the var is obviously absent,
// so running `go test -v ./provider/gemini/` prints something useful.
func TestMain(m *testing.M) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		// Don't block CI; smoke tests opt-in via env.
		fmt.Fprintln(os.Stderr, "note: GEMINI_API_KEY not set; smoke tests will skip")
	}
	m.Run()
}
