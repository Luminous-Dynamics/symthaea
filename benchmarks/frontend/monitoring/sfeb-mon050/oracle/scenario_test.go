package impact

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"testing"
)

type scenarioCorpus struct {
	Profile string `json:"profile"`
	Version string `json:"version"`
	Counts struct { Total int `json:"total"` } `json:"counts"`
	Graph Graph `json:"graph"`
	Scenarios []struct {
		ID string `json:"id"`
		Name string `json:"name"`
		Event Event `json:"event"`
		ExpectedAffected []string `json:"expected_affected_claims"`
		ExpectedUnaffected []string `json:"expected_unaffected_claims"`
	} `json:"scenarios"`
}
const frozenScenarioCorpusSHA256 = "0d71b5c55f5ed3c5cab6a399c8fb709f0c6cfbf93beeef87a7fa7ec2379ad312"
func TestFrozenScenarioCorpusKnownAnswers(t *testing.T) {
	data,err:=os.ReadFile("../fixtures/SCENARIOS-v1.json");if err!=nil{t.Fatal(err)}
	sum:=sha256.Sum256(data);if got:=hex.EncodeToString(sum[:]);got!=frozenScenarioCorpusSHA256{t.Fatalf("scenario corpus digest mismatch: %s",got)}
	var corpus scenarioCorpus;if err:=json.Unmarshal(data,&corpus);err!=nil{t.Fatal(err)}
	if corpus.Profile!="mon-dependency-impact-corpus-v1"||corpus.Version!="1.0.0"||corpus.Counts.Total!=6||len(corpus.Scenarios)!=6{t.Fatalf("invalid corpus header/cardinality: %+v",corpus)}
	seen:=map[string]bool{};for _,s:=range corpus.Scenarios{if s.ID==""||s.Name==""||seen[s.ID]{t.Fatalf("duplicate/empty scenario: %+v",s)};seen[s.ID]=true
		f:=Fixture{Profile:fixtureProfile,ID:s.ID,Graph:corpus.Graph,Event:s.Event,ExpectedAffected:s.ExpectedAffected,ExpectedUnaffected:s.ExpectedUnaffected}
		r,err:=Evaluate(f);if err!=nil{t.Fatalf("%s: %v",s.ID,err)};if r.Verdict!="qualified"||!r.ExactSet{t.Fatalf("%s failed known answer: %+v",s.ID,r)}
	}
}
