package minimize

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"reflect"
	"testing"
	"time"
)

const scenarioDigest = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
const boundaryDigest = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
const codeRevision = "0123456789abcdef0123456789abcdef01234567"
var evaluatedAt = time.Date(2026, 11, 27, 20, 0, 0, 0, time.UTC)

func testDigest(s string) string {
	d := sha256.Sum256([]byte(s))
	return hex.EncodeToString(d[:])
}
func makeObservation(ids []string, verdict Verdict, signature string, revision string, complete bool) Observation {
	canonical, ok := canonicalFaults(ids)
	if !ok { panic("invalid test fault IDs") }
	return Observation{
		FaultIDs: canonical,
		FaultSetSHA256: FaultSetDigest(canonical),
		ScenarioSHA256: scenarioDigest,
		RecoveryBoundarySHA256: boundaryDigest,
		CodeRevision: revision,
		ObservedAt: evaluatedAt.Add(-time.Minute),
		Verdict: verdict,
		FailureSignatureSHA256: signature,
		OutcomeReceiptSHA256: testDigest("receipt:"+FaultSetDigest(canonical)+":"+string(verdict)+":"+signature),
		Complete: complete,
	}
}
func baseInput(faults []string, observations []Observation) Input {
	return Input{
		Profile: Profile, Version: Version, TrialID: "trial-039-001",
		Mode: OfflineFixture, EvaluatedAt: evaluatedAt,
		ScenarioSHA256: scenarioDigest, RecoveryBoundarySHA256: boundaryDigest,
		CodeRevision: codeRevision, InitialFaultIDs: faults, Observations: observations,
	}
}
func allSubsets(ids []string, outcome func([]string)(Verdict,string)) []Observation {
	canonical, ok := canonicalFaults(ids)
	if !ok { panic("invalid subset fixture") }
	observations := make([]Observation, 0, 1<<len(canonical))
	for mask:=0; mask<(1<<len(canonical)); mask++ {
		subset:=[]string{}
		for i,id:=range canonical { if mask&(1<<i)!=0 { subset=append(subset,id) } }
		v,sig:=outcome(subset)
		observations=append(observations,makeObservation(subset,v,sig,codeRevision,true))
	}
	return observations
}
func contains(ids []string, id string) bool { for _,v:=range ids { if v==id { return true } }; return false }
func failureAAndC(ids []string)(Verdict,string) {
	sig:=testDigest("signature:failure-A-and-C")
	if contains(ids,"fault-a")&&contains(ids,"fault-c"){return Blocked,sig}
	return Qualified,testDigest("signature:clean")
}

func TestExhaustiveSearchFindsGlobalCandidateMinimum(t *testing.T) {
	ids:=[]string{"fault-a","fault-b","fault-c"}
	in:=baseInput([]string{"fault-c","fault-a","fault-b"},allSubsets(ids,failureAAndC))
	got:=Minimize(in)
	if got.FullFaultSetVerdict!=Blocked||got.MinimalityKind!="global_candidate_minimum"||!got.EvidenceComplete {
		t.Fatalf("expected complete global minimum: %+v",got)
	}
	if !reflect.DeepEqual(got.MinimalFaultIDs,[]string{"fault-a","fault-c"}) {
		t.Fatalf("wrong minimal fault set: %v",got.MinimalFaultIDs)
	}
	if got.CampaignVerdict!=Indeterminate||got.ProductionActionPermitted {
		t.Fatalf("offline counterfactual result cannot authorize production: %+v",got)
	}
	if len(got.Steps)==0||!got.Steps[0].TargetFailurePreserved {
		t.Fatalf("baseline full fault set must preserve the target failure: %+v",got.Steps)
	}
}
func TestDifferentFailureSignatureIsNotTheSameFailure(t *testing.T) {
	ids:=[]string{"fault-a","fault-b","fault-c"}
	observations:=allSubsets(ids,failureAAndC)
	for i:=range observations{
		if reflect.DeepEqual(observations[i].FaultIDs,[]string{"fault-a","fault-b"}){
			observations[i].Verdict=Blocked
			observations[i].FailureSignatureSHA256=testDigest("signature:different-block")
			observations[i].OutcomeReceiptSHA256=testDigest("different-receipt")
		}
	}
	got:=Minimize(baseInput(ids,observations))
	if !reflect.DeepEqual(got.MinimalFaultIDs,[]string{"fault-a","fault-c"}) {
		t.Fatalf("a different blocked signature was incorrectly retained: %+v",got)
	}
}
func TestMissingSmallerSubsetEvidenceCannotClaimGlobalMinimum(t *testing.T) {
	ids:=[]string{"fault-a","fault-b","fault-c"}
	observations:=allSubsets(ids,failureAAndC)
	filtered:=observations[:0]
	for _,o:=range observations{if len(o.FaultIDs)!=0{filtered=append(filtered,o)}}
	got:=Minimize(baseInput(ids,filtered))
	if got.MinimalityKind!="not_established"||got.EvidenceComplete||got.CampaignVerdict!=Indeterminate {
		t.Fatalf("missing empty-set replay must withhold global-minimum claim: %+v",got)
	}
	if !reflect.DeepEqual(got.MinimalFaultIDs,[]string{"fault-a","fault-c"}) {
		t.Fatalf("expected best observed candidate to be retained with qualification withheld: %v",got.MinimalFaultIDs)
	}
}
func TestLargeSetReportsOnlyOneMinimal(t *testing.T) {
	ids:=make([]string,13)
	for i:=range ids{ids[i]=fmt.Sprintf("fault-%02d",i)}
	sig:=testDigest("signature:single-essential-fault")
	observations:=[]Observation{}
	for n:=len(ids);n>=0;n--{
		subset:=append([]string(nil),ids[:n]...)
		v,s:=Qualified,testDigest("signature:clean")
		if contains(subset,"fault-00"){v,s=Blocked,sig}
		observations=append(observations,makeObservation(subset,v,s,codeRevision,true))
	}
	got:=Minimize(baseInput(ids,observations))
	if got.MinimalityKind!="one_minimal"||!got.EvidenceComplete {
		t.Fatalf("large set must report 1-minimal, not global minimum: %+v",got)
	}
	if !reflect.DeepEqual(got.MinimalFaultIDs,[]string{"fault-00"}) {
		t.Fatalf("unexpected 1-minimal result: %v",got.MinimalFaultIDs)
	}
}
func TestIncompleteReplayRowDoesNotCountAsNoFailure(t *testing.T) {
	ids:=[]string{"fault-a","fault-b","fault-c"}
	observations:=allSubsets(ids,failureAAndC)
	for i:=range observations{if len(observations[i].FaultIDs)==0{observations[i].Complete=false}}
	got:=Minimize(baseInput(ids,observations))
	if got.MinimalityKind!="not_established"||got.EvidenceComplete {
		t.Fatalf("incomplete subset must not be treated as a safe result: %+v",got)
	}
}
func TestStaleCodeRevisionBlocksEvidence(t *testing.T) {
	ids:=[]string{"fault-a"}
	o:=makeObservation(ids,Blocked,testDigest("failure"),"ffffffffffffffffffffffffffffffffffffffff",true)
	got:=Minimize(baseInput(ids,[]Observation{o}))
	if got.CampaignVerdict!=Blocked||got.EvidenceComplete {
		t.Fatalf("stale code revision must block: %+v",got)
	}
}
func TestDuplicateFaultSetObservationBlocks(t *testing.T) {
	ids:=[]string{"fault-a"}
	o:=makeObservation(ids,Blocked,testDigest("failure"),codeRevision,true)
	got:=Minimize(baseInput(ids,[]Observation{o,o}))
	if got.CampaignVerdict!=Blocked {
		t.Fatalf("duplicate observations must block: %+v",got)
	}
}
func TestDuplicateInitialFaultIDBlocks(t *testing.T) {
	o:=makeObservation([]string{"fault-a"},Blocked,testDigest("failure"),codeRevision,true)
	got:=Minimize(baseInput([]string{"fault-a","fault-a"},[]Observation{o}))
	if got.CampaignVerdict!=Blocked {
		t.Fatalf("duplicate initial fault ID must block: %+v",got)
	}
}
func TestFaultSetDigestIsOrderIndependent(t *testing.T) {
	if FaultSetDigest([]string{"b","a"})!=FaultSetDigest([]string{"a","b"}) {
		t.Fatal("fault set digest must canonicalize order")
	}
}
func TestRuntimeCandidateRemainsCampaignIndeterminate(t *testing.T) {
	ids:=[]string{"fault-a","fault-b","fault-c"}
	in:=baseInput(ids,allSubsets(ids,failureAAndC))
	in.Mode=RuntimeCandidate
	got:=Minimize(in)
	if got.MinimizationKind!="global_candidate_minimum"||got.CampaignVerdict!=Indeterminate||got.ProductionActionPermitted {
		t.Fatalf("runtime-candidate minimization cannot self-authorize: %+v",got)
	}
}
func TestExpectedFailureSignatureMismatchBlocks(t *testing.T) {
	ids:=[]string{"fault-a","fault-b","fault-c"}
	in:=baseInput(ids,allSubsets(ids,failureAAndC))
	in.ExpectedFailureSignatureSHA256=testDigest("wrong-signature")
	got:=Minimize(in)
	if got.CampaignVerdict!=Blocked {
		t.Fatalf("unexpected target signature must block: %+v",got)
	}
}
func TestInitialQualifiedResultIsNotApplicable(t *testing.T) {
	ids:=[]string{"fault-a"}
	o:=makeObservation(ids,Qualified,testDigest("clean"),codeRevision,true)
	got:=Minimize(baseInput(ids,[]Observation{o}))
	if got.MinimalityKind!="not_applicable"||got.CampaignVerdict!=Indeterminate {
		t.Fatalf("no failure to minimize: %+v",got)
	}
}
func TestFaultLimitBlocksBeforeSubsetSearch(t *testing.T) {
	ids:=make([]string,MaxFaults+1)
	for i:=range ids{ids[i]=fmt.Sprintf("fault-%03d",i)}
	got:=Minimize(baseInput(ids,nil))
	if got.CampaignVerdict!=Blocked {
		t.Fatalf("excessive fault count must block: %+v",got)
	}
}
func TestObservationMustBePinnedToRecoveryBoundary(t *testing.T) {
	ids:=[]string{"fault-a"}
	o:=makeObservation(ids,Blocked,testDigest("failure"),codeRevision,true)
	o.RecoveryBoundarySHA256=testDigest("other-boundary")
	got:=Minimize(baseInput(ids,[]Observation{o}))
	if got.CampaignVerdict!=Blocked {
		t.Fatalf("cross-boundary observation must block: %+v",got)
	}
}
func TestObservationAfterEvaluationTimeIsRejected(t *testing.T) {
	ids:=[]string{"fault-a"}
	o:=makeObservation(ids,Blocked,testDigest("failure"),codeRevision,true)
	o.ObservedAt=evaluatedAt.Add(time.Second)
	got:=Minimize(baseInput(ids,[]Observation{o}))
	if got.CampaignVerdict!=Blocked {
		t.Fatalf("future observation must be rejected: %+v",got)
	}
}

func BenchmarkExhaustiveTwelveFaults(b *testing.B) {
	ids:=[]string{}
	for i:=0;i<12;i++{ids=append(ids,fmt.Sprintf("fault-%02d",i))}
	sig:=testDigest("signature:two-causes")
	observations:=allSubsets(ids,func(subset []string)(Verdict,string){
		if contains(subset,"fault-00")&&contains(subset,"fault-11"){return Blocked,sig}
		return Qualified,testDigest("clean")
	})
	in:=baseInput(ids,observations)
	b.ReportAllocs();b.ResetTimer()
	for i:=0;i<b.N;i++{if Minimize(in).MinimalityKind!="global_candidate_minimum"{b.Fatal("exhaustive benchmark failed")}}
}
func BenchmarkOneMinimalSixtyFourFaults(b *testing.B) {
	ids:=make([]string,64);for i:=range ids{ids[i]=fmt.Sprintf("fault-%02d",i)}
	sig:=testDigest("signature:single-cause")
	observations:=[]Observation{}
	for n:=len(ids);n>=0;n--{
		subset:=append([]string(nil),ids[:n]...)
		v,s:=Qualified,testDigest("clean");if contains(subset,"fault-00"){v,s=Blocked,sig}
		observations=append(observations,makeObservation(subset,v,s,codeRevision,true))
	}
	in:=baseInput(ids,observations)
	b.ReportAllocs();b.ResetTimer()
	for i:=0;i<b.N;i++{if Minimize(in).MinimalityKind!="one_minimal"{b.Fatal("1-minimal benchmark failed")}}
}


func TestEmptyFaultSetUsesCanonicalJSONArrayDigest(t *testing.T) {
	const expected = "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
	if got := FaultSetDigest([]string{}); got != expected {
		t.Fatalf("empty fault set digest mismatch: got %s want %s", got, expected)
	}
}
