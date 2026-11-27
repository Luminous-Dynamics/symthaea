package continuity

import (
	"testing"
	"time"
)

var t0=time.Date(2026,11,27,12,0,0,0,time.UTC)
func digest(ch byte)string{b:=make([]byte,64);for i:=range b{b[i]=ch};return string(b)}
func baseInput()Input{
	b:=DependencyBinding{DependencyID:"pg.cluster",Revision:1,Digest:digest('a')}
	c:=Claim{ClaimID:"recovery-claim-1",EvidenceID:"bundle-1",EvidenceGrade:GradeRuntimeIndependent,ClosureComplete:true,ObservedAt:t0.Add(-time.Hour),ExpiresAt:t0.Add(time.Hour),Support:[]DependencyBinding{b}}
	l:=Ledger{Complete:true,CapturedAt:t0,ExpiresAt:t0.Add(time.Hour),Events:[]Event{}}
	s:=DependencyState{DependencyID:b.DependencyID,Revision:1,Digest:b.Digest}
	return Input{Profile:Profile,Version:Version,Now:t0,Ledger:l,CurrentStates:[]DependencyState{s},Claims:[]Claim{c}}
}
func appendEvent(in *Input,e Event){
	e.Sequence=uint64(len(in.Ledger.Events)+1)
	if len(in.Ledger.Events)>0{e.PreviousDigest=in.Ledger.Events[len(in.Ledger.Events)-1].Digest}
	e=SealEvent(e);in.Ledger.Events=append(in.Ledger.Events,e);in.Ledger.ThroughSequence=e.Sequence;in.Ledger.HeadDigest=e.Digest
}
func resultFor(t *testing.T,in Input)ClaimResult{t.Helper();out:=Evaluate(in);if len(out.Claims)!=1{t.Fatalf("expected one claim: %+v",out)};return out.Claims[0]}
func TestUnchangedClaimRetainedDespiteUnrelatedEvent(t *testing.T){
	in:=baseInput();appendEvent(&in,Event{EventID:"topic-retention-update",DependencyID:"kafka.topic.retention",Kind:DependencyChanged,BeforeRevision:1,AfterRevision:2,BeforeDigest:digest('b'),AfterDigest:digest('c'),OccurredAt:t0.Add(-time.Minute)})
	in.CurrentStates=append(in.CurrentStates,DependencyState{DependencyID:"kafka.topic.retention",Revision:2,Digest:digest('c')})
	if got:=resultFor(t,in);got.Disposition!=Retain{t.Fatalf("unrelated change should retain: %+v",got)}
}
func TestSupportedDependencyChangeRequiresRevalidation(t *testing.T){
	in:=baseInput();appendEvent(&in,Event{EventID:"connector-config-change",DependencyID:"pg.cluster",Kind:DependencyChanged,BeforeRevision:1,AfterRevision:2,BeforeDigest:digest('a'),AfterDigest:digest('b'),OccurredAt:t0.Add(-time.Minute)})
	in.CurrentStates[0].Revision=2;in.CurrentStates[0].Digest=digest('b')
	got:=resultFor(t,in);if got.Disposition!=Revalidate||len(got.AffectedSequences)!=1{t.Fatalf("supported change should revalidate: %+v",got)}
}
func TestVerificationMethodChangeRequiresRevalidation(t *testing.T){
	in:=baseInput();appendEvent(&in,Event{EventID:"verifier-v2",DependencyID:"restore.verifier",Kind:VerificationChanged,BeforeRevision:1,AfterRevision:2,BeforeDigest:digest('a'),AfterDigest:digest('b'),OccurredAt:t0.Add(-time.Minute)})
	in.Claims[0].Support=append(in.Claims[0].Support,DependencyBinding{DependencyID:"restore.verifier",Revision:1,Digest:digest('a')})
	in.CurrentStates=append(in.CurrentStates,DependencyState{DependencyID:"restore.verifier",Revision:2,Digest:digest('b')})
	if got:=resultFor(t,in);got.Disposition!=Revalidate{t.Fatalf("verifier change should revalidate: %+v",got)}
}
func TestAuthorityRevocationInvalidates(t *testing.T){
	in:=baseInput();appendEvent(&in,Event{EventID:"signer-revoked",DependencyID:"pg.cluster",Kind:AuthorityRevoked,OccurredAt:t0.Add(-time.Minute)})
	if got:=resultFor(t,in);got.Disposition!=Invalidate{t.Fatalf("revocation should invalidate: %+v",got)}
}
func TestContradictionAndScopeChangeInvalidate(t *testing.T){
	for _,kind:=range []EventKind{Contradiction,ScopeChanged}{in:=baseInput();appendEvent(&in,Event{EventID:"hard-change",DependencyID:"pg.cluster",Kind:kind,OccurredAt:t0.Add(-time.Minute)});if got:=resultFor(t,in);got.Disposition!=Invalidate{t.Fatalf("%s should invalidate: %+v",kind,got)}}
}
func TestUnknownEventIsUnresolved(t *testing.T){
	in:=baseInput();appendEvent(&in,Event{EventID:"unknown-impact",DependencyID:"pg.cluster",Kind:UnknownChange,OccurredAt:t0.Add(-time.Minute)})
	if got:=resultFor(t,in);got.Disposition!=Unresolved{t.Fatalf("unknown impact should abstain: %+v",got)}
}
func TestGlobalUnknownEventIsUnresolved(t *testing.T){
	in:=baseInput();appendEvent(&in,Event{EventID:"unknown-global",Kind:UnknownChange,OccurredAt:t0.Add(-time.Minute)})
	if got:=resultFor(t,in);got.Disposition!=Unresolved{t.Fatalf("global unknown should abstain: %+v",got)}
}
func TestIncompleteLedgerCannotRetain(t *testing.T){
	in:=baseInput();in.Ledger.Complete=false
	if got:=resultFor(t,in);got.Disposition!=Unresolved{t.Fatalf("incomplete ledger should abstain: %+v",got)}
}
func TestSequenceGapCannotRetain(t *testing.T){
	in:=baseInput();appendEvent(&in,Event{EventID:"e1",DependencyID:"other",Kind:UnknownChange,OccurredAt:t0.Add(-time.Minute)})
	in.Ledger.Events[0].Sequence=2;in.Ledger.Events[0]=SealEvent(in.Ledger.Events[0]);in.Ledger.ThroughSequence=2;in.Ledger.HeadDigest=in.Ledger.Events[0].Digest
	if got:=resultFor(t,in);got.Disposition!=Unresolved{t.Fatalf("gap should abstain: %+v",got)}
}
func TestBrokenDigestChainInvalidates(t *testing.T){
	in:=baseInput();appendEvent(&in,Event{EventID:"e1",DependencyID:"other",Kind:UnknownChange,OccurredAt:t0.Add(-time.Minute)});in.Ledger.Events[0].Digest=digest('f')
	if got:=resultFor(t,in);got.Disposition!=Invalidate{t.Fatalf("corrupt chain should invalidate: %+v",got)}
}
func TestWrongClaimWatermarkDigestInvalidates(t *testing.T){
	in:=baseInput();appendEvent(&in,Event{EventID:"e1",DependencyID:"other",Kind:UnknownChange,OccurredAt:t0.Add(-time.Minute)})
	in.Claims[0].ObservedThroughSequence=1;in.Claims[0].ObservedThroughDigest=digest('f')
	if got:=resultFor(t,in);got.Disposition!=Invalidate{t.Fatalf("wrong watermark should invalidate: %+v",got)}
}
func TestClaimWatermarkEventMustPrecedeClaimObservation(t *testing.T){
	in:=baseInput();appendEvent(&in,Event{EventID:"event-after-observation",DependencyID:"other",Kind:UnknownChange,OccurredAt:t0.Add(time.Minute)})
	in.Claims[0].ObservedThroughSequence=1;in.Claims[0].ObservedThroughDigest=in.Ledger.HeadDigest
	in.Now=t0.Add(2*time.Minute);in.Ledger.CapturedAt=in.Now
	if got:=resultFor(t,in);got.Disposition!=Unresolved{t.Fatalf("watermark event was not observable at claim time: %+v",got)}
}
func TestClaimAtCurrentLedgerWatermarkCanBeRetained(t *testing.T){
	in:=baseInput();appendEvent(&in,Event{EventID:"known-unrelated-event",DependencyID:"other",Kind:UnknownChange,OccurredAt:t0.Add(-2*time.Minute)})
	in.Claims[0].ObservedThroughSequence=1;in.Claims[0].ObservedThroughDigest=in.Ledger.HeadDigest;in.Claims[0].ObservedAt=t0.Add(-time.Minute);in.Ledger.CapturedAt=t0
	if got:=resultFor(t,in);got.Disposition!=Retain{t.Fatalf("claim at verified watermark should retain: %+v",got)}
}
func TestCurrentStateDriftWithoutEventIsUnresolved(t *testing.T){
	in:=baseInput();in.CurrentStates[0].Revision=2;in.CurrentStates[0].Digest=digest('b')
	if got:=resultFor(t,in);got.Disposition!=Unresolved{t.Fatalf("untracked drift should abstain: %+v",got)}
}
func TestCurrentStateMustMatchEventTail(t *testing.T){
	in:=baseInput();appendEvent(&in,Event{EventID:"dep-change",DependencyID:"pg.cluster",Kind:DependencyChanged,BeforeRevision:1,AfterRevision:2,BeforeDigest:digest('a'),AfterDigest:digest('b'),OccurredAt:t0.Add(-time.Minute)})
	in.CurrentStates[0].Revision=3;in.CurrentStates[0].Digest=digest('c')
	if got:=resultFor(t,in);got.Disposition!=Unresolved{t.Fatalf("unmatched tail should abstain: %+v",got)}
}
func TestExpiredClaimRequiresRevalidation(t *testing.T){
	in:=baseInput();in.Claims[0].ExpiresAt=t0
	if got:=resultFor(t,in);got.Disposition!=Revalidate{t.Fatalf("expired claim should revalidate: %+v",got)}
}
func TestMissingStateAndIncompleteClosureAreUnresolved(t *testing.T){
	in:=baseInput();in.CurrentStates=nil
	if got:=resultFor(t,in);got.Disposition!=Unresolved{t.Fatalf("missing state should abstain: %+v",got)}
	in=baseInput();in.Claims[0].ClosureComplete=false
	if got:=resultFor(t,in);got.Disposition!=Unresolved{t.Fatalf("incomplete closure should abstain: %+v",got)}
}
func TestNonRuntimeEvidenceCannotBeRetained(t *testing.T){
	in:=baseInput();in.Claims[0].EvidenceGrade="synthetic"
	if got:=resultFor(t,in);got.Disposition!=Unresolved{t.Fatalf("synthetic evidence should abstain: %+v",got)}
}
func TestDuplicateClaimIDsInvalidateEveryAmbiguousClaim(t *testing.T){
	in:=baseInput();in.Claims=append(in.Claims,in.Claims[0]);out:=Evaluate(in)
	if len(out.Claims)!=2{t.Fatalf("unexpected result count: %d",len(out.Claims))}
	for _,claim:=range out.Claims{if claim.Disposition!=Invalidate{t.Fatalf("both ambiguous claims must invalidate: %+v",claim)}}
}
func TestEventAfterLedgerCaptureInvalidates(t *testing.T){
	in:=baseInput();appendEvent(&in,Event{EventID:"late-event",DependencyID:"other",Kind:UnknownChange,OccurredAt:t0.Add(time.Second)})
	in.Ledger.CapturedAt=t0
	if got:=resultFor(t,in);got.Disposition!=Invalidate{t.Fatalf("post-capture event should invalidate: %+v",got)}
}
