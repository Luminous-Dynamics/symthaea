// Package minimize adds bounded, deterministic deletion-order tournaments.
// Results are best-observed across tested orders, never a global minimum claim
// for fault universes larger than the exhaustive v1 threshold.
package minimize

import (
 "sort"
 "strings"
)

const MaxReductionOrders = 8

type OrderedReduction struct {
 Order []string `json:"order"`
 OrderSHA256 string `json:"order_sha256"`
 MinimalFaultIDs []string `json:"minimal_fault_ids"`
 MinimalFaultSetSHA256 string `json:"minimal_fault_set_sha256"`
 MinimalityKind string `json:"minimality_kind"`
 SubsetEvaluations int `json:"subset_evaluations"`
 EvidenceComplete bool `json:"evidence_complete"`
 CampaignVerdict Verdict `json:"campaign_verdict"`
 ReasonCodes []string `json:"reason_codes"`
}
type TournamentResult struct {
 Profile string `json:"profile"`
 Version string `json:"version"`
 TrialID string `json:"trial_id"`
 TournamentVerdict Verdict `json:"tournament_verdict"`
 CampaignVerdict Verdict `json:"campaign_verdict"`
 ProductionActionPermitted bool `json:"production_action_permitted"`
 InitialFaultSetSHA256 string `json:"initial_fault_set_sha256"`
 OrdersAttempted int `json:"orders_attempted"`
 TotalSubsetEvaluations int `json:"total_subset_evaluations"`
 EvidenceComplete bool `json:"evidence_complete"`
 BestObservedFaultIDs []string `json:"best_observed_fault_ids"`
 BestObservedFaultSetSHA256 string `json:"best_observed_fault_set_sha256"`
 BestObservedFaultCount int `json:"best_observed_fault_count"`
 BestObservedGuarantee string `json:"best_observed_guarantee"`
 GlobalMinimumClaimed bool `json:"global_minimum_claimed"`
 Orders []OrderedReduction `json:"orders"`
 ReasonCodes []string `json:"reason_codes"`
}

func containsID(ids []string,target string)bool{for _,id:=range ids{if id==target{return true}};return false}
func sameSetOrder(order,initial []string)bool{
 if len(order)!=len(initial){return false}
 seen:=map[string]struct{}{}
 for _,id:=range order{if !validID(id){return false};if _,ok:=seen[id];ok{return false};seen[id]=struct{}{}}
 for _,id:=range initial{if _,ok:=seen[id];!ok{return false}}
 return true
}
func orderDigest(order []string)string{return digestJSON(order)}
func defaultDeletionOrder(initial []string)[]string{
 out:=append([]string(nil),initial...)
 for i,j:=0,len(out)-1;i<j;i,j=i+1,j-1{out[i],out[j]=out[j],out[i]}
 return out
}
func indexID(ids []string,target string)int{for i,id:=range ids{if id==target{return i}};return -1}
func lexicographicallyLess(a,b []string)bool{return strings.Join(a,"\x00")<strings.Join(b,"\x00")}

// MinimizeInOrder performs the existing MON-039 semantic reduction with an
// explicit deletion permutation. For <=12 faults the exhaustive search remains
// authoritative and order-independent. For larger sets the result is only
// one_minimal under the tested deterministic deletion order.
func MinimizeInOrder(in Input, deletionOrder []string)Result{
 out:=Result{
  Profile:"mon039-counterfactual-fault-reduction-result-v1",Version:Version,
  TrialID:in.TrialID,Mode:in.Mode,MinimalFaultIDs:[]string{},
  MinimalityKind:"not_established",EvidenceComplete:false,
  CampaignVerdict:Indeterminate,ProductionActionPermitted:false,
  Steps:[]EvaluationStep{},ReasonCodes:[]string{},
 }
 f:=newFlags()
 index:=validateInput(in,f)
 if f.blocked{out.CampaignVerdict=Blocked;out.ReasonCodes=f.codes();return out}
 initial,ok:=canonicalFaults(in.InitialFaultIDs)
 if !ok||len(initial)>MaxFaults{f.block("initial_fault_set_invalid");out.CampaignVerdict=Blocked;out.ReasonCodes=f.codes();return out}
 order:=append([]string(nil),deletionOrder...)
 if len(order)==0{order=defaultDeletionOrder(initial)}
 if !sameSetOrder(order,initial){f.block("reduction_order_not_exact_fault_permutation");out.CampaignVerdict=Blocked;out.ReasonCodes=f.codes();return out}

 evaluations:=0
 incomplete:=false
 lookup:=func(ids []string)(Observation,bool){
  if evaluations>=MaxEvaluations{incomplete=true;f.hold("evaluation_limit_reached");return Observation{},false}
  canonical,valid:=canonicalFaults(ids)
  if !valid{f.block("candidate_fault_set_invalid");return Observation{},false}
  digest:=digestJSON(canonical)
  o,exists:=index[digest]
  if !exists{incomplete=true;f.hold("subset_observation_missing:"+digest);return Observation{},false}
  if !o.Complete{incomplete=true;f.hold("subset_observation_incomplete:"+digest);return Observation{},false}
  if !sameIDs(o.FaultIDs,canonical){f.block("fault_set_canonicalization_mismatch");return Observation{},false}
  evaluations++
  target:=sameFailure(o,out.FullFaultSetVerdict,out.TargetFailureSignatureSHA256)
  out.Steps=append(out.Steps,EvaluationStep{Sequence:evaluations,FaultIDs:canonical,FaultSetSHA256:digest,Verdict:o.Verdict,FailureSignatureSHA256:o.FailureSignatureSHA256,OutcomeReceiptSHA256:o.OutcomeReceiptSHA256,TargetFailurePreserved:target})
  return o,true
 }
 full,exists:=lookup(initial)
 if !exists{
  out.SubsetEvaluations=evaluations;out.EvidenceComplete=false;out.CampaignVerdict=f.verdict();out.ReasonCodes=f.codes();return out
 }
 out.FullFaultSetVerdict=full.Verdict
 out.TargetFailureSignatureSHA256=full.FailureSignatureSHA256
 if len(out.Steps)>0{out.Steps[0].TargetFailurePreserved=true}
 if full.Verdict!=Blocked{
  f.hold("initial_fault_set_did_not_reproduce_blocked_failure")
  out.MinimalityKind="not_applicable";out.SubsetEvaluations=evaluations;out.EvidenceComplete=true
  out.CampaignVerdict=Indeterminate;out.ReasonCodes=f.codes();return out
 }
 if in.ExpectedFailureSignatureSHA256!=""&&in.ExpectedFailureSignatureSHA256!=full.FailureSignatureSHA256{
  f.block("initial_failure_signature_mismatch");out.SubsetEvaluations=evaluations;out.CampaignVerdict=Blocked;out.ReasonCodes=f.codes();return out
 }

 current:=append([]string(nil),initial...)
 if len(initial)<=ExhaustiveFaultLimit{
  found:=false;var candidate []string
  for size:=0;size<len(initial);size++{
   combinations(initial,size,func(ids []string)bool{
    o,ok:=lookup(ids);if !ok{return false}
    if sameFailure(o,out.FullFaultSetVerdict,out.TargetFailureSignatureSHA256){found=true;candidate=append([]string(nil),ids...);return true}
    return false
   })
   if found{break}
   if f.blocked||evaluations>=MaxEvaluations{break}
  }
  if found{current=candidate;if !incomplete&&!f.blocked{out.MinimalityKind="global_candidate_minimum"}else{out.MinimalityKind="not_established"}}else if !incomplete&&!f.blocked{current=initial;out.MinimalityKind="global_candidate_minimum"}
 }else{
  changed:=true
  for changed&&!f.blocked&&!incomplete{
   changed=false
   for _,faultID:=range order{
    position:=indexID(current,faultID)
    if position<0{continue}
    candidate:=make([]string,0,len(current)-1)
    candidate=append(candidate,current[:position]...)
    candidate=append(candidate,current[position+1:]...)
    o,ok:=lookup(candidate)
    if !ok{incomplete=true;break}
    if sameFailure(o,out.FullFaultSetVerdict,out.TargetFailureSignatureSHA256){current=candidate;changed=true;break}
   }
  }
  if !incomplete&&!f.blocked&&!changed{out.MinimalityKind="one_minimal"}
 }
 out.MinimalFaultIDs=append([]string(nil),current...)
 out.MinimalFaultSetSHA256=digestJSON(current)
 out.SubsetEvaluations=evaluations
 out.EvidenceComplete=!incomplete&&!f.blocked
 if incomplete||f.blocked{out.MinimalityKind="not_established";f.hold("minimality_not_established")}
 if in.Mode==OfflineFixture{f.hold("offline_fixture_is_not_causal_proof")}
 if in.Mode==RuntimeCandidate{f.hold("runtime_replay_authentication_not_implemented")}
 out.CampaignVerdict=f.verdict();out.ReasonCodes=f.codes()
 return out
}

// MinimizeTournament repeats the exact same trial/scope/oracle under a bounded
// set of explicit deletion orders. It selects the smallest complete observed
// result and reports only best-observed 1-minimality for >12 faults.
func MinimizeTournament(in Input, orders [][]string)TournamentResult{
 out:=TournamentResult{
  Profile:"mon039-multi-order-tournament-result-v1",Version:Version,TrialID:in.TrialID,
  TournamentVerdict:Indeterminate,CampaignVerdict:Indeterminate,ProductionActionPermitted:false,
  OrdersAttempted:len(orders),TotalSubsetEvaluations:0,EvidenceComplete:false,
  BestObservedFaultIDs:[]string{},BestObservedFaultSetSHA256:"",BestObservedFaultCount:-1,
  BestObservedGuarantee:"not_established",GlobalMinimumClaimed:false,
  Orders:[]OrderedReduction{},ReasonCodes:[]string{},
 }
 f:=newFlags()
 initial,ok:=canonicalFaults(in.InitialFaultIDs)
 if !ok{f.block("initial_fault_set_invalid")}else{out.InitialFaultSetSHA256=digestJSON(initial)}
 if len(orders)<1||len(orders)>MaxReductionOrders{f.block("reduction_order_count_out_of_range")}
 if f.blocked{out.TournamentVerdict=Blocked;out.ReasonCodes=f.codes();return out}

 allComplete:=true
 seenOrders:=map[string]struct{}{}
 for index,order:=range orders{
  if !sameSetOrder(order,initial){f.block("reduction_order_not_exact_fault_permutation");continue}
  orderSHA:=orderDigest(order)
  if _,duplicate:=seenOrders[orderSHA];duplicate{f.block("reduction_order_duplicated");continue}
  seenOrders[orderSHA]=struct{}{}
  result:=MinimizeInOrder(in,order)
  out.TotalSubsetEvaluations+=result.SubsetEvaluations
  allComplete=allComplete&&result.EvidenceComplete
  out.Orders=append(out.Orders,OrderedReduction{
   Order:append([]string(nil),order...),OrderSHA256:orderDigest(order),
   MinimalFaultIDs:append([]string(nil),result.MinimalFaultIDs...),
   MinimalFaultSetSHA256:result.MinimalFaultSetSHA256,MinimalityKind:result.MinimalityKind,
   SubsetEvaluations:result.SubsetEvaluations,EvidenceComplete:result.EvidenceComplete,
   CampaignVerdict:result.CampaignVerdict,ReasonCodes:append([]string(nil),result.ReasonCodes...),
  })
  if result.CampaignVerdict==Blocked{f.block("ordered_reduction_blocked:"+strings.Join(result.ReasonCodes,","))}
  if result.EvidenceComplete&&(result.MinimalityKind=="one_minimal"||result.MinimalityKind=="global_candidate_minimum"){
   candidate:=result.MinimalFaultIDs
   if out.BestObservedFaultCount<0||len(candidate)<out.BestObservedFaultCount||(len(candidate)==out.BestObservedFaultCount&&lexicographicallyLess(candidate,out.BestObservedFaultIDs)){
    out.BestObservedFaultIDs=append([]string(nil),candidate...)
    out.BestObservedFaultSetSHA256=result.MinimalFaultSetSHA256
    out.BestObservedFaultCount=len(candidate)
   }
  }else{allComplete=false}
  _=index
 }
 out.EvidenceComplete=allComplete&&len(out.Orders)==len(orders)&&!f.blocked
 if out.BestObservedFaultCount>=0{
  if len(initial)<=ExhaustiveFaultLimit{out.BestObservedGuarantee="global_candidate_minimum_within_finite_fault_universe"}else{out.BestObservedGuarantee="best_observed_one_minimal_across_tested_orders"}
 }
 if !out.EvidenceComplete{f.hold("one_or_more_orderings_not_fully_established")}
 if len(initial)>ExhaustiveFaultLimit&&out.EvidenceComplete{f.hold("best_observed_is_not_global_minimum")}
 if len(initial)<=ExhaustiveFaultLimit&&out.EvidenceComplete&&out.BestObservedGuarantee=="global_candidate_minimum_within_finite_fault_universe"{out.GlobalMinimumClaimed=true}
 out.TournamentVerdict=f.verdict()
 out.CampaignVerdict=Indeterminate
 out.ProductionActionPermitted=false
 out.ReasonCodes=f.codes()
 return out
}

func SortedReductionOrders(initial []string)[][]string{
 ids,ok:=canonicalFaults(initial);if !ok{return nil}
 orders:=[][]string{}
 seen:=map[string]struct{}{}
 add:=func(order []string){key:=orderDigest(order);if _,exists:=seen[key];exists{return};seen[key]=struct{}{};orders=append(orders,order)}
 add(defaultDeletionOrder(ids))
 add(append([]string(nil),ids...))
 // Deterministic rotations add alternative deletion schedules without random state.
 for shift:=1;shift<len(ids)&&len(orders)<MaxReductionOrders;shift++{
  rotated:=append([]string(nil),ids[shift:]...)
  rotated=append(rotated,ids[:shift]...)
  add(rotated)
 }
 return orders
}

// Keep ordering stable in JSON/report generation.
func SortOrderResults(results []OrderedReduction){sort.Slice(results,func(i,j int)bool{return results[i].OrderSHA256<results[j].OrderSHA256})}
