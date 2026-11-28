package admission

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"testing"
	"time"
)

type temporalCorpus struct {
	Profile string `json:"profile"`
	Version string `json:"version"`
	Revocation string `json:"revocation_effective_at"`
	Cases []struct {
		ID string `json:"id"`
		Earliest string `json:"capture_earliest"`
		Latest string `json:"capture_latest"`
		Expected Verdict `json:"expected"`
	} `json:"cases"`
}

const temporalCorpusSHA256 = "a799044cabc52cc49f07218adfc61b217a27aa12c33a0c5c9094f3d58d3def85"

func TestFrozenTemporalBoundaryCorpus(t *testing.T) {
	data, err := os.ReadFile("fixtures/TEMPORAL-BOUNDARIES-v1.json")
	if err != nil { t.Fatal(err) }
	sum := sha256.Sum256(data)
	if got := hex.EncodeToString(sum[:]); got != temporalCorpusSHA256 { t.Fatalf("temporal corpus digest=%s", got) }
	var corpus temporalCorpus
	if err := json.Unmarshal(data, &corpus); err != nil { t.Fatal(err) }
	if corpus.Profile != "mon-temporal-boundaries-v1" || corpus.Version != "1.0.0" || len(corpus.Cases) != 7 { t.Fatalf("invalid temporal corpus: %+v", corpus) }
	revoke, err := time.Parse(time.RFC3339Nano, corpus.Revocation)
	if err != nil { t.Fatal(err) }
	seen := map[string]bool{}
	for _, tc := range corpus.Cases {
		if tc.ID == "" || seen[tc.ID] { t.Fatalf("duplicate/empty temporal vector ID: %q", tc.ID) }
		seen[tc.ID] = true
		lo, err := time.Parse(time.RFC3339Nano, tc.Earliest); if err != nil { t.Fatal(err) }
		hi, err := time.Parse(time.RFC3339Nano, tc.Latest); if err != nil { t.Fatal(err) }
		got, err := AppraiseCapture(Interval{Earliest: lo, Latest: hi}, &revoke)
		if err != nil { t.Fatalf("%s: %v", tc.ID, err) }
		if got.Verdict != tc.Expected { t.Fatalf("%s: got %s want %s (%s)", tc.ID, got.Verdict, tc.Expected, got.Reason) }
	}
}
