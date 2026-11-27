package replay

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"testing"
	"time"
)

const scenarioSHA="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
const boundarySHA="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
const faultSetSHA="f5fc0a015d906c648d3e4306703dd343d2aa2bcb255e732ba3ea0bf40659ec11"
const revision="0123456789abcdef0123456789abcdef01234567"
const targetSignature="30100eb79964719053a37a1ef7fe2be0e9290f7639284a848565a99dc77ffa5e"
const otherSignature="2222222222222222222222222222222222222222222222222222222222222222"
var now=time.Date(2026,11,27,20,0,0,0,time.UTC)

func testDigest(s string)string{d:=sha256.Sum256([]byte(s));return hex.EncodeToString(d[:])}
func observation(n int, verdict Verdict, signature string, complete bool)Observation{
	o:=Observation{
		ReplayID:fmt.Sprintf("replay-%03d",n+1),
		TrialID:"trial-040-001",
		Seed:int64(1000+n),
		FaultSetSHA256:faultSetSHA,
		ScenarioSHA256:scenarioSHA,
		RecoveryBoundarySHA256:boundarySHA,
		CodeRevision:revision,
		ObservedAt:now.Add(-time.Duration(30-n)*time.Second),
		Verdict:verdict,
		FailureSignatureSHA256:signature,
		Complete:complete,
	}
	o.OutcomeReceiptSHA256=ObservationReceiptDigest(o)
	return o
}
func fixture(n int)Input{
	observations:=make([]Observation,n)
	for i:=range observations{observations[i]=observation(i,Blocked,targetSignature,true)}
	return Input{
		Profile:Profile,Version:Version,TrialID:"trial-040-001",Mode:OfflineFixture,
		EvaluatedAt:now,ScenarioSHA256:scenarioSHA,RecoveryBoundarySHA256:boundarySHA,
		CodeRevision:revision,FaultIDs:[]string{"fault-a","fault-c"},FaultSetSHA256:faultSetSHA,TargetFailureSignatureSHA256:targetSignature,
		RequiredRuns:n,MinimumReliability:0.90,ConfidenceLevel:0.95,Observations:observations,
	}
}

func TestTwentyNineOfTwentyNineExceedsNinetyPercentLowerBound(t *testing.T){
	in:=fixture(29);got:=Evaluate(in)
	if got.ReproducibilityVerdict!=Qualified||!got.EvidenceComplete{t.Fatalf("29/29 should qualify the preregistered reproducibility threshold: %+v",got)}
	if got.OneSidedConfidenceLowerBound<0.90||got.OneSidedConfidenceLowerBound>=0.903{t.Fatalf("unexpected exact lower bound: %.12f",got.OneSidedConfidenceLowerBound)}
	if got.CampaignVerdict!=Indeterminate||got.ProductionUsePermitted{t.Fatalf("offline evidence cannot authorize production: %+v",got)}
}
func TestTwentyEightOfTwentyNineFailsNinetyPercentLowerBound(t *testing.T){
	in:=fixture(29);in.Observations[28]=observation(28,Qualified,testDigest("clean"),true)
	got:=Evaluate(in)
	if got.ReproducibilityVerdict!=Blocked||got.OneSidedConfidenceLowerBound>=0.90{t.Fatalf("28/29 must not qualify 90%% at 95%% one-sided confidence: %+v",got)}
}
func TestMissingRunIsIndeterminateNotAQualifiedSmallerDenominator(t *testing.T){
	in:=fixture(28);in.RequiredRuns=29
	got:=Evaluate(in)
	if got.ReproducibilityVerdict!=Indeterminate||got.EvidenceComplete{t.Fatalf("missing replay must hold qualification: %+v",got)}
}
func TestIncompleteObservationDoesNotCountAsCompleteEvidence(t *testing.T){
	in:=fixture(29);in.Observations[0]=observation(0,Blocked,targetSignature,false)
	got:=Evaluate(in)
	if got.ReproducibilityVerdict!=Indeterminate||got.EvidenceComplete{t.Fatalf("incomplete replay should withhold qualification: %+v",got)}
}
func TestIndeterminateReplayOutcomeHoldsEvenWhenAllReceiptsPresent(t *testing.T){
	in:=fixture(29);in.Observations[0]=observation(0,Indeterminate,testDigest("unknown"),true)
	got:=Evaluate(in)
	if got.ReproducibilityVerdict!=Indeterminate||!got.EvidenceComplete{t.Fatalf("complete but indeterminate replay is not a target match: %+v",got)}
}
func TestOtherBlockedSignatureIsNotATargetMatch(t *testing.T){
	in:=fixture(29);in.Observations[0]=observation(0,Blocked,otherSignature,true)
	got:=Evaluate(in)
	if got.ReproducibilityVerdict!=Blocked||got.OtherBlockedOutcomes!=1||got.TargetSignatureMatches!=28{t.Fatalf("different blocked signature must count as a non-match: %+v",got)}
}
func TestDuplicateReplayIDBlocks(t *testing.T){
	in:=fixture(29);in.Observations[1].ReplayID=in.Observations[0].ReplayID
	in.Observations[1].OutcomeReceiptSHA256=ObservationReceiptDigest(in.Observations[1])
	got:=Evaluate(in)
	if got.ReproducibilityVerdict!=Blocked{t.Fatalf("duplicate replay identity must block: %+v",got)}
}
func TestDuplicateSeedBlocks(t *testing.T){
	in:=fixture(29);in.Observations[1].Seed=in.Observations[0].Seed
	in.Observations[1].OutcomeReceiptSHA256=ObservationReceiptDigest(in.Observations[1])
	got:=Evaluate(in)
	if got.ReproducibilityVerdict!=Blocked{t.Fatalf("duplicate random seed must block: %+v",got)}
}
func TestCrossBoundaryObservationBlocks(t *testing.T){
	in:=fixture(29);in.Observations[0].RecoveryBoundarySHA256=testDigest("other")
	in.Observations[0].OutcomeReceiptSHA256=ObservationReceiptDigest(in.Observations[0])
	got:=Evaluate(in)
	if got.ReproducibilityVerdict!=Blocked{t.Fatalf("cross-boundary replay must block: %+v",got)}
}
func TestReceiptMutationBlocks(t *testing.T){
	in:=fixture(29);in.Observations[0].Complete=false
	got:=Evaluate(in)
	if got.ReproducibilityVerdict!=Blocked{t.Fatalf("stale receipt digest after content mutation must block: %+v",got)}
}
func TestObservationAfterEvaluationTimeBlocks(t *testing.T){
	in:=fixture(29);in.Observations[0].ObservedAt=now.Add(time.Second)
	in.Observations[0].OutcomeReceiptSHA256=ObservationReceiptDigest(in.Observations[0])
	got:=Evaluate(in)
	if got.ReproducibilityVerdict!=Blocked{t.Fatalf("future replay observation must block: %+v",got)}
}
func TestRuntimeCandidateRemainsCampaignIndeterminate(t *testing.T){
	in:=fixture(29);in.Mode=RuntimeCandidate
	got:=Evaluate(in)
	if got.ReproducibilityVerdict!=Qualified||got.CampaignVerdict!=Indeterminate||got.ProductionUsePermitted{t.Fatalf("runtime candidate cannot self-authorize: %+v",got)}
}
func TestOneSidedClopperPearsonKnownAnswers(t *testing.T){
	lower29:=OneSidedClopperPearsonLower(29,29,0.95)
	lower28:=OneSidedClopperPearsonLower(28,29,0.95)
	if lower29<0.9018||lower29>0.9019{t.Fatalf("29/29 exact lower bound unexpected: %.12f",lower29)}
	if lower28>=0.90||lower28<0.846||lower28>0.847{t.Fatalf("28/29 exact lower bound unexpected: %.12f",lower28)}
}
func TestConfidenceBoundIsMonotonicInSuccessCount(t *testing.T){
	prev:=0.0
	for k:=0;k<=29;k++{now:=OneSidedClopperPearsonLower(k,29,0.95);if now<prev{t.Fatalf("lower bound decreased at %d: %f < %f",k,now,prev)};prev=now}
}
func TestExcessReplayCountBlocks(t *testing.T){
	in:=fixture(29);in.Observations=append(in.Observations,observation(29,Blocked,targetSignature,true))
	got:=Evaluate(in)
	if got.ReproducibilityVerdict!=Blocked{t.Fatalf("unpreregistered extra replay must block: %+v",got)}
}
func TestInvalidConfidenceParametersBlock(t *testing.T){
	in:=fixture(29);in.ConfidenceLevel=1.0
	got:=Evaluate(in)
	if got.ReproducibilityVerdict!=Blocked{t.Fatalf("invalid confidence parameter must block: %+v",got)}
}
func TestTenOfTenCannotClaimNinetyPercentNinetyFiveConfidence(t *testing.T){
	in:=fixture(10);in.MinimumReliability=0.90;in.ConfidenceLevel=0.95
	got:=Evaluate(in)
	if got.ReproducibilityVerdict!=Blocked{t.Fatalf("10/10 lower bound is below 0.9: %+v",got)}
}
func BenchmarkEvaluateTwentyNineReplays(b *testing.B){
	in:=fixture(29);b.ReportAllocs();b.ResetTimer()
	for i:=0;i<b.N;i++{if Evaluate(in).ReproducibilityVerdict!=Qualified{b.Fatal("29-replay fixture failed")}}
}
func BenchmarkEvaluateTwoHundredReplays(b *testing.B){
	in:=fixture(200);b.ReportAllocs();b.ResetTimer()
	for i:=0;i<b.N;i++{_ = Evaluate(in)}
}


func TestMissingFaultIDListBlocks(t *testing.T) {
	in:=fixture(29)
	in.FaultIDs=nil
	got:=Evaluate(in)
	if got.ReproducibilityVerdict!=Blocked {
		t.Fatalf("omitted fault list must block: %+v",got)
	}
}

func TestFaultIDListMustMatchDigest(t *testing.T) {
	in:=fixture(29)
	in.FaultIDs=[]string{"fault-a","fault-b"}
	got:=Evaluate(in)
	if got.ReproducibilityVerdict!=Blocked {
		t.Fatalf("fault-list/digest mismatch must block: %+v",got)
	}
}
