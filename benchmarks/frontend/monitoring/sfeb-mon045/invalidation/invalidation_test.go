package invalidation

import (
 "fmt"
 "testing"
 "time"
)

var at=time.Date(2026,11,28,12,30,0,0,time.UTC)
func testHash(s string)string{return digestJSON(s)}
func node(id string,kind NodeKind)Node{
 return Node{ID:id,Kind:kind,Digest:testHash("digest-"+id),DeclaredVerdict:Qualified,IssuerID:"trusted-test-issuer",
  IssuedAt:at.Add(-time.Hour),ValidUntil:at.Add(time.Hour),Authenticated:true}
}
func edge(from,to string,relation Relation)Edge{return Edge{PrerequisiteID:from,DependentID:to,Relation:relation,ExpectedSHA256:testHash("digest-"+from)}}
func healthyInput()Input{
 nodes:=[]Node{
  node("db-source-truth",Evidence),node("schema-scope",Scope),node("connector-capability",Capability),
  node("conformance-report",Evidence),node("recovery-policy",Policy),node("recovery-claim",Claim),
  node("telemetry-source",Evidence),node("telemetry-claim",Claim),
 }
 edges:=[]Edge{
  edge("db-source-truth","connector-capability",Requires),
  edge("conformance-report","connector-capability",Requires),
  edge("schema-scope","recovery-claim",Requires),
  edge("connector-capability","recovery-claim",Requires),
  edge("recovery-policy","recovery-claim",Requires),
  edge("telemetry-source","telemetry-claim",Requires),
 }
 return Input{Profile:Profile,Version:Version,Mode:OfflineFixture,EvaluatedAt:at,AllowedFutureSkewSeconds:5,Nodes:nodes,Edges:edges,RootClaimIDs:[]string{"recovery-claim","telemetry-claim"}}
}
func findClaim(r Result,id string)*ClaimResult{for i:=range r.Claims{if r.Claims[i].ClaimID==id{return &r.Claims[i]}};return nil}
func TestHealthyDependencyGraphQualifiesRootClaimsOfflineOnly(t *testing.T){
 got:=Evaluate(healthyInput())
 if got.GraphVerdict!=Qualified{t.Fatalf("healthy graph should qualify internally: %+v",got)}
 if len(got.ImpactedClaimIDs)!=0||len(got.UnaffectedClaimIDs)!=2{t.Fatalf("unexpected scope: %+v",got)}
 if got.CampaignVerdict!=Indeterminate||got.ProductionActionPermitted{t.Fatalf("offline graph cannot authorize promotion: %+v",got)}
}
func TestRevocationInvalidatesOnlyDependentClaim(t *testing.T){
 in:=healthyInput();revoked:=at.Add(-time.Minute)
 for i:=range in.Nodes{if in.Nodes[i].ID=="db-source-truth"{in.Nodes[i].RevokedAt=&revoked}}
 got:=Evaluate(in)
 if got.GraphVerdict!=Blocked{t.Fatalf("revoked required evidence must block dependent graph claim: %+v",got)}
 if len(got.ImpactedClaimIDs)!=1||got.ImpactedClaimIDs[0]!="recovery-claim"{t.Fatalf("revocation escaped its dependency closure: %+v",got)}
 if len(got.UnaffectedClaimIDs)!=1||got.UnaffectedClaimIDs[0]!="telemetry-claim"{t.Fatalf("unrelated claim must remain qualified: %+v",got)}
 claim:=findClaim(got,"recovery-claim")
 if claim==nil||len(claim.InvalidatingNodeIDs)==0||claim.InvalidatingNodeIDs[0]!="connector-capability"&&claim.InvalidatingNodeIDs[0]!="db-source-truth"{t.Fatalf("missing invalidation trace: %+v",claim)}
}
func TestTransitiveInvalidationTraversesIntermediateClaims(t *testing.T){
 in:=healthyInput()
 // The existing graph already contains source -> capability -> recovery claim.
 rev:=at.Add(-time.Minute)
 for i:=range in.Nodes{if in.Nodes[i].ID=="db-source-truth"{in.Nodes[i].RevokedAt=&rev}}
 got:=Evaluate(in)
 cap:=nodeResult(got,"connector-capability")
 if cap==nil||cap.EffectiveVerdict!=Blocked{t.Fatalf("intermediate capability should be invalidated: %+v",cap)}
 claim:=findClaim(got,"recovery-claim")
 if claim==nil||claim.Verdict!=Blocked{t.Fatalf("downstream recovery claim must be invalidated: %+v",claim)}
}
func nodeResult(r Result,id string)*NodeResult{for i:=range r.NodeResults{if r.NodeResults[i].ID==id{return &r.NodeResults[i]}};return nil}
func TestExpiryHoldsDependentClaimIndeterminate(t *testing.T){
 in:=healthyInput()
 for i:=range in.Nodes{if in.Nodes[i].ID=="schema-scope"{in.Nodes[i].ValidUntil=at.Add(-time.Second)}}
 got:=Evaluate(in)
 claim:=findClaim(got,"recovery-claim")
 if claim==nil||claim.Verdict!=Indeterminate{t.Fatalf("expired dependency must hold claim: %+v",claim)}
 if got.GraphVerdict!=Indeterminate{t.Fatalf("expiry is unknown/indeterminate, not fabricated success: %+v",got)}
}
func TestMissingRequiredPrerequisiteHolds(t *testing.T){
 in:=healthyInput();in.Edges=append(in.Edges,edge("missing-artifact","recovery-claim",Requires))
 got:=Evaluate(in)
 claim:=findClaim(got,"recovery-claim")
 if claim==nil||claim.Verdict!=Indeterminate{t.Fatalf("missing evidence must hold claim: %+v",claim)}
}
func TestRequiredDependencyDigestMismatchBlocks(t *testing.T){
 in:=healthyInput()
 for i:=range in.Edges{if in.Edges[i].PrerequisiteID=="schema-scope"{in.Edges[i].ExpectedSHA256=testHash("old-schema")}}
 if got:=Evaluate(in);findClaim(got,"recovery-claim").Verdict!=Blocked{t.Fatalf("stale dependency pin must block: %+v",got)}
}
func TestOptionalContextCannotDowngradeRequiredClaim(t *testing.T){
 in:=healthyInput()
 optional:=node("optional-dashboard",Evidence);optional.Authenticated=false;optional.DeclaredVerdict=Indeterminate
 in.Nodes=append(in.Nodes,optional)
 in.Edges=append(in.Edges,Edge{PrerequisiteID:"optional-dashboard",DependentID:"recovery-claim",Relation:Context,ExpectedSHA256:optional.Digest})
 got:=Evaluate(in)
 if got.GraphVerdict!=Qualified||findClaim(got,"recovery-claim").Verdict!=Qualified{t.Fatalf("optional context must not invalidate required claim: %+v",got)}
 nr:=nodeResult(got,"recovery-claim")
 if nr==nil||len(nr.OptionalContext)!=1{t.Fatalf("optional context should be visible: %+v",nr)}
}
func TestOptionalContextDigestDriftIsVisibleButNotBlocking(t *testing.T){
 in:=healthyInput()
 in.Edges=append(in.Edges,Edge{PrerequisiteID:"telemetry-source",DependentID:"recovery-claim",Relation:Context,ExpectedSHA256:testHash("old-telemetry")})
 got:=Evaluate(in)
 if got.GraphVerdict!=Qualified{t.Fatalf("optional context digest drift must not block required claim: %+v",got)}
 nr:=nodeResult(got,"recovery-claim")
 found:=false;for _,r:=range nr.ReasonCodes{if r=="optional_context_digest_mismatch:telemetry-source"{found=true}}
 if !found{t.Fatalf("optional digest drift not reported: %+v",nr)}
}
func TestUnauthenticatedRequiredEvidenceHolds(t *testing.T){
 in:=healthyInput()
 for i:=range in.Nodes{if in.Nodes[i].ID=="db-source-truth"{in.Nodes[i].Authenticated=false}}
 got:=Evaluate(in)
 if findClaim(got,"recovery-claim").Verdict!=Indeterminate||got.GraphVerdict!=Indeterminate{t.Fatalf("unauthenticated evidence cannot support claim: %+v",got)}
}
func TestRevocationAfterEvaluationTimeDoesNotInvalidateYet(t *testing.T){
 in:=healthyInput();future:=at.Add(time.Minute)
 for i:=range in.Nodes{if in.Nodes[i].ID=="db-source-truth"{in.Nodes[i].RevokedAt=&future}}
 if got:=Evaluate(in);findClaim(got,"recovery-claim").Verdict!=Qualified{t.Fatalf("future revocation must not be applied early: %+v",got)}
}
func TestRevocationBeforeIssuanceBlocksAsMalformedHistory(t *testing.T){
 in:=healthyInput();early:=at.Add(-2*time.Hour)
 for i:=range in.Nodes{if in.Nodes[i].ID=="db-source-truth"{in.Nodes[i].RevokedAt=&early}}
 if got:=Evaluate(in);findClaim(got,"recovery-claim").Verdict!=Blocked{t.Fatalf("revocation preceding issue must block: %+v",got)}
}
func TestFutureEvidenceBeyondSkewBlocks(t *testing.T){
 in:=healthyInput()
 for i:=range in.Nodes{if in.Nodes[i].ID=="schema-scope"{in.Nodes[i].IssuedAt=at.Add(time.Minute);in.Nodes[i].ValidUntil=at.Add(2*time.Hour)}}
 if got:=Evaluate(in);findClaim(got,"recovery-claim").Verdict!=Blocked{t.Fatalf("far-future evidence must block: %+v",got)}
}
func TestCycleBlocksWholeGraph(t *testing.T){
 in:=healthyInput();in.Edges=append(in.Edges,edge("recovery-claim","schema-scope",Requires))
 if got:=Evaluate(in);got.GraphVerdict!=Blocked{t.Fatalf("cycle must block: %+v",got)}
}
func TestDuplicateNodeBlocks(t *testing.T){
 in:=healthyInput();in.Nodes=append(in.Nodes,in.Nodes[0])
 if got:=Evaluate(in);got.GraphVerdict!=Blocked{t.Fatalf("duplicate node ID must block: %+v",got)}
}
func TestDuplicateEdgeBlocks(t *testing.T){
 in:=healthyInput();in.Edges=append(in.Edges,in.Edges[0])
 if got:=Evaluate(in);got.GraphVerdict!=Blocked{t.Fatalf("duplicate edge must block: %+v",got)}
}
func TestWrongRootKindBlocks(t *testing.T){
 in:=healthyInput();in.RootClaimIDs=[]string{"db-source-truth"}
 if got:=Evaluate(in);got.GraphVerdict!=Blocked{t.Fatalf("evidence cannot be promoted as a claim root: %+v",got)}
}
func TestMissingRootClaimHolds(t *testing.T){
 in:=healthyInput();in.RootClaimIDs=append(in.RootClaimIDs,"claim-not-produced")
 if got:=Evaluate(in);got.GraphVerdict!=Indeterminate{t.Fatalf("missing root claim must hold: %+v",got)}
}
func TestUnrelatedInvalidNodeDoesNotInvalidateRootClaims(t *testing.T){
 in:=healthyInput();orphan:=node("unrelated-revoked-record",Evidence);rev:=at.Add(-time.Second);orphan.RevokedAt=&rev
 in.Nodes=append(in.Nodes,orphan)
 got:=Evaluate(in)
 if got.GraphVerdict!=Qualified||len(got.ImpactedClaimIDs)!=0{t.Fatalf("orphan evidence must not downgrade unrelated claims: %+v",got)}
 if !contains(got.InvalidatedNodeIDs,"unrelated-revoked-record"){t.Fatalf("orphan invalidation must remain observable: %+v",got.InvalidatedNodeIDs)}
}
func TestGraphDigestIsInputOrderIndependent(t *testing.T){
 in:=healthyInput();a:=GraphDigest(in.Nodes,in.Edges,in.RootClaimIDs)
 for i,j:=0,len(in.Nodes)-1;i<j;i,j=i+1,j-1{in.Nodes[i],in.Nodes[j]=in.Nodes[j],in.Nodes[i]}
 for i,j:=0,len(in.Edges)-1;i<j;i,j=i+1,j-1{in.Edges[i],in.Edges[j]=in.Edges[j],in.Edges[i]}
 b:=GraphDigest(in.Nodes,in.Edges,in.RootClaimIDs)
 if a!=b{t.Fatalf("canonical graph digest depends on input ordering: %s != %s",a,b)}
}
func TestMissingExpectedDependencyDigestHolds(t *testing.T){
 in:=healthyInput();for i:=range in.Edges{if in.Edges[i].PrerequisiteID=="schema-scope"{in.Edges[i].ExpectedSHA256=""}}
 if got:=Evaluate(in);findClaim(got,"recovery-claim").Verdict!=Indeterminate{t.Fatalf("missing pinned digest must hold: %+v",got)}
}
func TestAllRequiredParentsMustQualify(t *testing.T){
 in:=healthyInput()
 for i:=range in.Nodes{if in.Nodes[i].ID=="conformance-report"{in.Nodes[i].DeclaredVerdict=Blocked}}
 got:=Evaluate(in)
 if findClaim(got,"recovery-claim").Verdict!=Blocked{t.Fatalf("one failing required parent blocks the claim: %+v",got)}
}
func TestRootClaimOwnFailureIsNotMaskedByHealthyParents(t *testing.T){
 in:=healthyInput();for i:=range in.Nodes{if in.Nodes[i].ID=="recovery-claim"{in.Nodes[i].DeclaredVerdict=Blocked}}
 if got:=Evaluate(in);findClaim(got,"recovery-claim").Verdict!=Blocked{t.Fatalf("claim's own failure must not be masked: %+v",got)}
}
func TestOptionalMissingContextDoesNotHold(t *testing.T){
 in:=healthyInput();in.Edges=append(in.Edges,Edge{PrerequisiteID:"missing-optional-context",DependentID:"recovery-claim",Relation:Context})
 if got:=Evaluate(in);got.GraphVerdict!=Qualified{t.Fatalf("missing optional context cannot hold required claim: %+v",got)}
}
func BenchmarkEvaluate1000NodeEvidenceGraph(b *testing.B){
 in:=Input{Profile:Profile,Version:Version,Mode:OfflineFixture,EvaluatedAt:at,AllowedFutureSkewSeconds:5}
 root:=node("root-claim",Claim);in.Nodes=append(in.Nodes,root);in.RootClaimIDs=[]string{"root-claim"}
 for i:=0;i<999;i++{id:=fmt.Sprintf("evidence-%04d",i);n:=node(id,Evidence);in.Nodes=append(in.Nodes,n);in.Edges=append(in.Edges,edge(id,"root-claim",Requires))}
 b.ReportAllocs();b.ResetTimer()
 for i:=0;i<b.N;i++{_ = Evaluate(in)}
}
