package minimize

import (
 "fmt"
 "testing"
 "time"
)

func tournamentFixture() (Input,[][]string) {
 ids:=[]string{"a","b","c","d","e","f","g","h","i","j","k","l","m"}
 now:=time.Date(2026,11,28,12,0,0,0,time.UTC)
 scenario:=digestJSON("mon039-tournament-scenario")
 boundary:=digestJSON("mon039-tournament-boundary")
 revision:="0123456789abcdef0123456789abcdef01234567"
 target:=digestJSON("target-signature")
 other:=digestJSON("other-signature")
 obs:=make([]Observation,0,1<<len(ids))
 for mask:=0;mask<(1<<len(ids));mask++{
  subset:=make([]string,0,len(ids))
  present:=map[string]bool{}
  for bit,id:=range ids{if mask&(1<<bit)!=0{subset=append(subset,id);present[id]=true}}
  targetPreserved:=(present["a"]&&present["b"])||(present["k"]&&present["l"]&&present["m"])
  verdict:=Qualified;signature:=other
  if targetPreserved{verdict=Blocked;signature=target}
  o:=Observation{FaultIDs:subset,TrialID:"trial-mon039-tournament",FaultSetSHA256:FaultSetDigest(subset),ScenarioSHA256:scenario,RecoveryBoundarySHA256:boundary,CodeRevision:revision,ObservedAt:now.Add(-time.Minute),Verdict:verdict,FailureSignatureSHA256:signature,Complete:true}
  o.OutcomeReceiptSHA256=ObservationReceiptDigest(o)
  obs=append(obs,o)
 }
 in:=Input{Profile:Profile,Version:Version,TrialID:"trial-mon039-tournament",Mode:OfflineFixture,EvaluatedAt:now,
  ScenarioSHA256:scenario,RecoveryBoundarySHA256:boundary,CodeRevision:revision,
  InitialFaultIDs:ids,ExpectedFailureSignatureSHA256:target,Observations:obs}
 orderA:=[]string{"a","b","c","d","e","f","g","h","i","j","k","l","m"}
 orderB:=[]string{"k","l","m","a","b","c","d","e","f","g","h","i","j"}
 return in,[][]string{orderA,orderB}
}
func TestExplicitReductionOrdersFindDifferentOneMinimalSets(t *testing.T){
 in,orders:=tournamentFixture()
 a:=MinimizeInOrder(in,orders[0])
 b:=MinimizeInOrder(in,orders[1])
 if !a.EvidenceComplete||a.MinimalityKind!="one_minimal"||len(a.MinimalFaultIDs)!=3{t.Fatalf("first deletion order should find three-fault cause: %+v",a)}
 if !b.EvidenceComplete||b.MinimalityKind!="one_minimal"||len(b.MinimalFaultIDs)!=2{t.Fatalf("second deletion order should find two-fault cause: %+v",b)}
 if !sameIDs(a.MinimalFaultIDs,[]string{"k","l","m"}){t.Fatalf("unexpected first cause: %v",a.MinimalFaultIDs)}
 if !sameIDs(b.MinimalFaultIDs,[]string{"a","b"}){t.Fatalf("unexpected second cause: %v",b.MinimalFaultIDs)}
}
func TestTournamentSelectsBestObservedWithoutGlobalClaim(t *testing.T){
 in,orders:=tournamentFixture()
 got:=MinimizeTournament(in,orders)
 if got.TournamentVerdict!=Indeterminate||got.CampaignVerdict!=Indeterminate||got.ProductionActionPermitted{t.Fatalf("offline tournament must never authorize: %+v",got)}
 if !got.EvidenceComplete||got.OrdersAttempted!=2||len(got.Orders)!=2{t.Fatalf("both tested orders should be complete: %+v",got)}
 if got.BestObservedFaultCount!=2||!sameIDs(got.BestObservedFaultIDs,[]string{"a","b"}){t.Fatalf("best observed set not selected: %+v",got)}
 if got.BestObservedGuarantee!="best_observed_one_minimal_across_tested_orders"||got.GlobalMinimumClaimed{t.Fatalf("must not claim global minimum for >12 faults: %+v",got)}
 if got.TotalSubsetEvaluations<=0{t.Fatalf("subset evaluation count missing: %+v",got)}
}
func TestIncompleteOrderingDoesNotDisappearFromTournament(t *testing.T){
 in,orders:=tournamentFixture()
 missing:=append([]string(nil),in.InitialFaultIDs[1:]...)
 digest:=FaultSetDigest(missing)
 filtered:=make([]Observation,0,len(in.Observations)-1)
 for _,o:=range in.Observations{if o.FaultSetSHA256!=digest{filtered=append(filtered,o)}}
 in.Observations=filtered
 got:=MinimizeTournament(in,orders)
 if got.EvidenceComplete||got.TournamentVerdict!=Indeterminate{t.Fatalf("incomplete order must hold whole tournament qualification: %+v",got)}
 if got.BestObservedFaultCount!=2{t.Fatalf("best complete observed order may still be reported descriptively: %+v",got)}
}
func TestInvalidOrDuplicatePermutationBlocks(t *testing.T){
 in,orders:=tournamentFixture()
 bad:=append([]string(nil),orders[0]...);bad[0]=bad[1]
 result:=MinimizeInOrder(in,bad)
 if result.CampaignVerdict!=Blocked{t.Fatalf("duplicate/missing fault order must block: %+v",result)}
 got:=MinimizeTournament(in,[][]string{orders[0],orders[0]})
 if got.TournamentVerdict!=Blocked{t.Fatalf("duplicate order must block: %+v",got)}
}
func TestSortedReductionOrdersAreBoundedAndUnique(t *testing.T){
 in,_:=tournamentFixture()
 orders:=SortedReductionOrders(in.InitialFaultIDs)
 if len(orders)<2||len(orders)>MaxReductionOrders{t.Fatalf("unexpected number of orders: %d",len(orders))}
 seen:=map[string]bool{}
 for _,order:=range orders{
  if !sameSetOrder(order,in.InitialFaultIDs){t.Fatalf("not a fault permutation: %v",order)}
  d:=orderDigest(order);if seen[d]{t.Fatalf("duplicate deterministic order: %v",order)};seen[d]=true
 }
}
func BenchmarkMinimizeTournamentEightOrders(b *testing.B){
 in,_:=tournamentFixture()
 orders:=SortedReductionOrders(in.InitialFaultIDs)
 b.ReportAllocs();b.ResetTimer()
 for i:=0;i<b.N;i++{_ = MinimizeTournament(in,orders)}
}
func ExampleSortedReductionOrders(){
 orders:=SortedReductionOrders([]string{"a","b","c"})
 fmt.Println(len(orders)>1)
 // Output: true
}
