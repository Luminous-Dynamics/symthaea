// Package invalidation evaluates typed evidence dependencies as a DAG.
// It computes scoped downstream invalidation. It never authenticates the
// supplied trust snapshot or authorizes production mutations.
package invalidation

import (
 "container/heap"
 "crypto/sha256"
 "encoding/hex"
 "encoding/json"
 "fmt"
 "sort"
 "strings"
 "time"
)

const (
 Profile = "mon045-evidence-dependency-input-v1"
 Version = "1.0.0"
 MaxNodes = 10000
 MaxEdges = 100000
 MaxRoots = 1000
 MaxIDLength = 512
)

type Verdict string
const (
 Qualified Verdict = "qualified"
 Blocked Verdict = "blocked"
 Indeterminate Verdict = "indeterminate"
)
type Mode string
const (
 OfflineFixture Mode = "offline_fixture"
 RuntimeCandidate Mode = "runtime_candidate"
)
type NodeKind string
const (
 Evidence NodeKind = "evidence"
 Capability NodeKind = "capability"
 Scope NodeKind = "scope"
 Policy NodeKind = "policy"
 Claim NodeKind = "claim"
)
type Relation string
const (
 Requires Relation = "requires"
 Context Relation = "context"
)
type Node struct {
 ID string `json:"id"`
 Kind NodeKind `json:"kind"`
 Digest string `json:"digest"`
 DeclaredVerdict Verdict `json:"declared_verdict"`
 IssuerID string `json:"issuer_id"`
 IssuedAt time.Time `json:"issued_at"`
 ValidUntil time.Time `json:"valid_until"`
 RevokedAt *time.Time `json:"revoked_at,omitempty"`
 Authenticated bool `json:"authenticated"`
}
type Edge struct {
 PrerequisiteID string `json:"prerequisite_id"`
 DependentID string `json:"dependent_id"`
 Relation Relation `json:"relation"`
 ExpectedSHA256 string `json:"expected_sha256"`
}
type Input struct {
 Profile string `json:"profile"`
 Version string `json:"version"`
 Mode Mode `json:"mode"`
 EvaluatedAt time.Time `json:"evaluated_at"`
 AllowedFutureSkewSeconds int64 `json:"allowed_future_skew_seconds"`
 Nodes []Node `json:"nodes"`
 Edges []Edge `json:"edges"`
 RootClaimIDs []string `json:"root_claim_ids"`
}
type NodeResult struct {
 ID string `json:"id"`
 Kind NodeKind `json:"kind"`
 EffectiveVerdict Verdict `json:"effective_verdict"`
 OwnVerdict Verdict `json:"own_verdict"`
 RequiredPrerequisites []string `json:"required_prerequisites"`
 OptionalContext []string `json:"optional_context"`
 ReasonCodes []string `json:"reason_codes"`
}
type ClaimResult struct {
 ClaimID string `json:"claim_id"`
 Verdict Verdict `json:"verdict"`
 DependencyNodeIDs []string `json:"dependency_node_ids"`
 InvalidatingNodeIDs []string `json:"invalidating_node_ids"`
 DependencyTraceSHA256 string `json:"dependency_trace_sha256"`
 ReasonCodes []string `json:"reason_codes"`
}
type Result struct {
 Profile string `json:"profile"`
 Version string `json:"version"`
 GraphVerdict Verdict `json:"graph_verdict"`
 CampaignVerdict Verdict `json:"campaign_verdict"`
 ProductionActionPermitted bool `json:"production_action_permitted"`
 GraphSHA256 string `json:"graph_sha256"`
 NodeCount int `json:"node_count"`
 EdgeCount int `json:"edge_count"`
 TopologicalOrder []string `json:"topological_order"`
 Claims []ClaimResult `json:"claims"`
 ImpactedClaimIDs []string `json:"impacted_claim_ids"`
 UnaffectedClaimIDs []string `json:"unaffected_claim_ids"`
 InvalidatedNodeIDs []string `json:"invalidated_node_ids"`
 NodeResults []NodeResult `json:"node_results"`
 ReasonCodes []string `json:"reason_codes"`
}
type flags struct { blocked,unknown bool; codes map[string]struct{} }
func newFlags()*flags{return &flags{codes:map[string]struct{}{}}}
func(f *flags)block(s string){f.blocked=true;f.codes[s]=struct{}{}}
func(f *flags)hold(s string){f.unknown=true;f.codes[s]=struct{}{}}
func(f *flags)verdict()Verdict{if f.blocked{return Blocked};if f.unknown{return Indeterminate};return Qualified}
func(f *flags)sorted()[]string{out:=make([]string,0,len(f.codes));for s:=range f.codes{out=append(out,s)};sort.Strings(out);return out}
func validID(s string)bool{return s!=""&&len(s)<=MaxIDLength&&!strings.ContainsAny(s," \t\r\n")}
func validDigest(s string)bool{if len(s)!=64||s!=strings.ToLower(s){return false};_,e:=hex.DecodeString(s);return e==nil}
func validKind(k NodeKind)bool{switch k{case Evidence,Capability,Scope,Policy,Claim:return true};return false}
func validVerdict(v Verdict)bool{switch v{case Qualified,Blocked,Indeterminate:return true};return false}
func nodeOwnEvaluation(n Node,at time.Time,skew int64)(Verdict,[]string){
 f:=newFlags()
 if at.IsZero(){f.hold("evaluation_time_missing")}
 if !validID(n.ID)||!validKind(n.Kind)||!validDigest(n.Digest)||!validID(n.IssuerID){f.block("node_identity_or_digest_invalid")}
 if !validVerdict(n.DeclaredVerdict){f.block("node_declared_verdict_invalid")}
 if n.IssuedAt.IsZero()||n.ValidUntil.IsZero()||!n.ValidUntil.After(n.IssuedAt){f.hold("node_validity_window_invalid")}
 if !at.IsZero()&&!n.IssuedAt.IsZero()&&n.IssuedAt.After(at.Add(time.Duration(skew)*time.Second)){f.block("node_issued_in_future")}
 if !at.IsZero()&&!n.ValidUntil.IsZero()&&!n.ValidUntil.After(at){f.hold("node_expired")}
 if n.RevokedAt!=nil&&!at.IsZero()&&!n.RevokedAt.After(at){f.block("node_revoked")}
 if n.RevokedAt!=nil&&!n.IssuedAt.IsZero()&&n.RevokedAt.Before(n.IssuedAt){f.block("node_revocation_precedes_issue")}
 if !n.Authenticated{f.hold("node_issuer_not_authenticated")}
 if n.DeclaredVerdict==Blocked{f.block("node_declared_blocked")}
 if n.DeclaredVerdict==Indeterminate{f.hold("node_declared_indeterminate")}
 return f.verdict(),f.sorted()
}
type stringHeap []string
func(h stringHeap)Len()int{return len(h)}
func(h stringHeap)Less(i,j int)bool{return h[i]<h[j]}
func(h stringHeap)Swap(i,j int){h[i],h[j]=h[j],h[i]}
func(h *stringHeap)Push(x any){*h=append(*h,x.(string))}
func(h *stringHeap)Pop()any{old:=*h;n:=len(old);x:=old[n-1];*h=old[:n-1];return x}

func GraphDigest(nodes []Node,edges []Edge,roots []string)string{
 ns:=append([]Node(nil),nodes...)
 for i:=range ns{
  ns[i].IssuedAt=ns[i].IssuedAt.Round(0).UTC()
  ns[i].ValidUntil=ns[i].ValidUntil.Round(0).UTC()
  if ns[i].RevokedAt!=nil{revoked:=ns[i].RevokedAt.Round(0).UTC();ns[i].RevokedAt=&revoked}
 }
 sort.Slice(ns,func(i,j int)bool{return ns[i].ID<ns[j].ID})
 es:=append([]Edge(nil),edges...)
 sort.Slice(es,func(i,j int)bool{
  if es[i].PrerequisiteID!=es[j].PrerequisiteID{return es[i].PrerequisiteID<es[j].PrerequisiteID}
  if es[i].DependentID!=es[j].DependentID{return es[i].DependentID<es[j].DependentID}
  return es[i].Relation<es[j].Relation
 })
 rs:=append([]string(nil),roots...);sort.Strings(rs)
 payload:=struct{Nodes []Node `json:"nodes"`;Edges []Edge `json:"edges"`;Roots []string `json:"root_claim_ids"`}{ns,es,rs}
 b,_:=json.Marshal(payload);h:=sha256.Sum256(b);return hex.EncodeToString(h[:])
}
func resultCombine(a,b Verdict)Verdict{if a==Blocked||b==Blocked{return Blocked};if a==Indeterminate||b==Indeterminate{return Indeterminate};return Qualified}

func topological(nodes map[string]Node,edges []Edge)([]string,bool){
 indegree:=map[string]int{};adj:=map[string][]string{}
 for id:=range nodes{indegree[id]=0}
 for _,e:=range edges{
  if _,ok:=nodes[e.PrerequisiteID];!ok{continue}
  if _,ok:=nodes[e.DependentID];!ok{continue}
  adj[e.PrerequisiteID]=append(adj[e.PrerequisiteID],e.DependentID)
  indegree[e.DependentID]++
 }
 ready:=&stringHeap{};heap.Init(ready)
 for id,d:=range indegree{if d==0{heap.Push(ready,id)}}
 order:=make([]string,0,len(nodes))
 for ready.Len()>0{
  id:=heap.Pop(ready).(string);order=append(order,id)
  for _,dependent:=range adj[id]{indegree[dependent]--;if indegree[dependent]==0{heap.Push(ready,dependent)}}
 }
 return order,len(order)==len(nodes)
}
func appendUnique(xs []string,s string)[]string{for _,x:=range xs{if x==s{return xs}};return append(xs,s)}

func Evaluate(in Input)Result{
 out:=Result{
  Profile:"mon045-evidence-dependency-result-v1",Version:Version,
  GraphVerdict:Indeterminate,CampaignVerdict:Indeterminate,ProductionActionPermitted:false,
  GraphSHA256:GraphDigest(in.Nodes,in.Edges,in.RootClaimIDs),
  NodeCount:len(in.Nodes),EdgeCount:len(in.Edges),
  TopologicalOrder:[]string{},Claims:[]ClaimResult{},ImpactedClaimIDs:[]string{},
  UnaffectedClaimIDs:[]string{},InvalidatedNodeIDs:[]string{},NodeResults:[]NodeResult{},ReasonCodes:[]string{},
 }
 f:=newFlags()
 if in.Profile!=Profile||in.Version!=Version{f.block("profile_version_invalid")}
 if in.Mode!=OfflineFixture&&in.Mode!=RuntimeCandidate{f.block("mode_invalid")}
 if in.EvaluatedAt.IsZero(){f.hold("evaluation_time_missing")}
 if in.AllowedFutureSkewSeconds<0||in.AllowedFutureSkewSeconds>300{f.block("future_skew_out_of_range")}
 if len(in.Nodes)==0||len(in.Nodes)>MaxNodes{f.block("node_count_out_of_range")}
 if len(in.Edges)>MaxEdges{f.block("edge_count_out_of_range")}
 if len(in.RootClaimIDs)==0||len(in.RootClaimIDs)>MaxRoots{f.block("root_claim_count_out_of_range")}
 if f.blocked{out.GraphVerdict=Blocked;out.ReasonCodes=f.sorted();return out}

 nodes:=map[string]Node{}
 for _,n:=range in.Nodes{
  if !validID(n.ID){f.block("node_id_invalid");continue}
  if _,ok:=nodes[n.ID];ok{f.block("node_id_duplicated:"+n.ID);continue}
  nodes[n.ID]=n
 }
 if f.blocked{out.GraphVerdict=Blocked;out.ReasonCodes=f.sorted();return out}

 edgeKeys:=map[string]struct{}{}
 for _,e:=range in.Edges{
  if !validID(e.PrerequisiteID)||!validID(e.DependentID){f.block("edge_identity_invalid");continue}
  if e.PrerequisiteID==e.DependentID{f.block("self_dependency:"+e.PrerequisiteID)}
  if _,ok:=nodes[e.DependentID];!ok{f.block("dependent_node_missing:"+e.DependentID)}
  if e.Relation!=Requires&&e.Relation!=Context{f.block("edge_relation_invalid")}
  if e.Relation==Requires&&!validDigest(e.ExpectedSHA256){/* the dependent claim is held during evaluation */}
  key:=e.PrerequisiteID+"\x00"+e.DependentID+"\x00"+string(e.Relation)
  if _,ok:=edgeKeys[key];ok{f.block("edge_duplicated:"+e.PrerequisiteID+":"+e.DependentID)}
  edgeKeys[key]=struct{}{}
 }
 roots:=map[string]struct{}{}
 for _,id:=range in.RootClaimIDs{
  if !validID(id){f.block("root_claim_id_invalid");continue}
  if _,ok:=roots[id];ok{f.block("root_claim_duplicated:"+id)}
  roots[id]=struct{}{}
  n,ok:=nodes[id];if !ok{f.hold("root_claim_missing:"+id)}else if n.Kind!=Claim{f.block("root_node_is_not_claim:"+id)}
 }
 if f.blocked{out.GraphVerdict=Blocked;out.ReasonCodes=f.sorted();return out}

 order,acyclic:=topological(nodes,in.Edges)
 if !acyclic{f.block("dependency_cycle_detected");out.GraphVerdict=Blocked;out.ReasonCodes=f.sorted();return out}
 out.TopologicalOrder=append([]string(nil),order...)

 inbound:=map[string][]Edge{}
 for _,e:=range in.Edges{inbound[e.DependentID]=append(inbound[e.DependentID],e)}
 for id:=range inbound{
  sort.Slice(inbound[id],func(i,j int)bool{
   if inbound[id][i].PrerequisiteID!=inbound[id][j].PrerequisiteID{return inbound[id][i].PrerequisiteID<inbound[id][j].PrerequisiteID}
   return inbound[id][i].Relation<inbound[id][j].Relation
  })
 }
 effective:=map[string]Verdict{}
 reasons:=map[string][]string{}
 nodeResults:=map[string]NodeResult{}
 for _,id:=range order{
  n:=nodes[id]
  own,ownCodes:=nodeOwnEvaluation(n,in.EvaluatedAt,in.AllowedFutureSkewSeconds)
  combined:=own
  required:=[]string{};optional:=[]string{};codes:=append([]string(nil),ownCodes...)
  for _,e:=range inbound[id]{
   dependency,exists:=nodes[e.PrerequisiteID]
   if !exists{
    if e.Relation==Requires{combined=resultCombine(combined,Indeterminate);codes=appendUnique(codes,"required_prerequisite_missing:"+e.PrerequisiteID)}
    else{optional=appendUnique(optional,e.PrerequisiteID)}
    continue
   }
   depVerdict:=effective[e.PrerequisiteID]
   digestMatches:=e.ExpectedSHA256==""||e.ExpectedSHA256==dependency.Digest
   if e.Relation==Requires{
    required=append(required,e.PrerequisiteID)
    if !validDigest(e.ExpectedSHA256){combined=resultCombine(combined,Indeterminate);codes=appendUnique(codes,"required_dependency_digest_unverifiable:"+e.PrerequisiteID)}
    else if !digestMatches{combined=Blocked;codes=appendUnique(codes,"required_dependency_digest_mismatch:"+e.PrerequisiteID)}
    combined=resultCombine(combined,depVerdict)
    if depVerdict!=Qualified{codes=appendUnique(codes,"required_dependency_not_qualified:"+e.PrerequisiteID)}
   }else{
    optional=append(optional,e.PrerequisiteID)
    if !digestMatches{codes=appendUnique(codes,"optional_context_digest_mismatch:"+e.PrerequisiteID)}
    if depVerdict!=Qualified{codes=appendUnique(codes,"optional_context_not_qualified:"+e.PrerequisiteID)}
   }
  }
  sort.Strings(required);sort.Strings(optional);sort.Strings(codes)
  effective[id]=combined;reasons[id]=codes
  nodeResults[id]=NodeResult{ID:id,Kind:n.Kind,OwnVerdict:own,EffectiveVerdict:combined,RequiredPrerequisites:required,OptionalContext:optional,ReasonCodes:codes}
  if combined!=Qualified{out.InvalidatedNodeIDs=append(out.InvalidatedNodeIDs,id)}
  // Node failures are recorded locally. Only requested root claims contribute
  // to GraphVerdict; unrelated evidence remains visible without global blast radius.
 }
 for _,id:=range order{out.NodeResults=append(out.NodeResults,nodeResults[id])}
 sort.Strings(out.InvalidatedNodeIDs)

 claimIDs:=make([]string,0,len(in.RootClaimIDs));for id:=range roots{claimIDs=append(claimIDs,id)};sort.Strings(claimIDs)
 for _,claimID:=range claimIDs{
  visited:=map[string]struct{}{}
  stack:=[]string{claimID}
  invalidating:=[]string{}
  for len(stack)>0{
   current:=stack[len(stack)-1];stack=stack[:len(stack)-1]
   for _,edge:=range inbound[current]{
    if edge.Relation!=Requires{continue}
    dep:=edge.PrerequisiteID
    if _,ok:=visited[dep];ok{continue};visited[dep]=struct{}{}
    stack=append(stack,dep)
    if effective[dep]!=Qualified{invalidating=appendUnique(invalidating,dep)}
   }
  }
  deps:=make([]string,0,len(visited));for dep:=range visited{deps=append(deps,dep)};sort.Strings(deps);sort.Strings(invalidating)
  trace:=[]struct{ID string `json:"id"`;Digest string `json:"digest"`;Verdict Verdict `json:"verdict"`}{}
  for _,dep:=range deps{n:=nodes[dep];trace=append(trace,struct{ID string `json:"id"`;Digest string `json:"digest"`;Verdict Verdict `json:"verdict"`}{dep,n.Digest,effective[dep]})}
  traceDigest:=digestJSON(trace)
  claimReasons:=append([]string(nil),reasons[claimID]...)
  for _,dep:=range invalidating{for _,code:=range reasons[dep]{claimReasons=appendUnique(claimReasons,dep+":"+code)}}
  claimVerdict,claimExists:=effective[claimID]
  if !claimExists{claimVerdict=Indeterminate;claimReasons=appendUnique(claimReasons,"root_claim_missing:"+claimID)}
  sort.Strings(claimReasons)
  cr:=ClaimResult{ClaimID:claimID,Verdict:claimVerdict,DependencyNodeIDs:deps,InvalidatingNodeIDs:invalidating,DependencyTraceSHA256:traceDigest,ReasonCodes:claimReasons}
  out.Claims=append(out.Claims,cr)
  if cr.Verdict==Qualified{out.UnaffectedClaimIDs=append(out.UnaffectedClaimIDs,claimID)}else{out.ImpactedClaimIDs=append(out.ImpactedClaimIDs,claimID)}
  f.codes["root_claim:"+claimID+":"+string(cr.Verdict)]=struct{}{}
  if cr.Verdict==Blocked{f.block("root_claim_blocked:"+claimID)}else if cr.Verdict==Indeterminate{f.hold("root_claim_indeterminate:"+claimID)}
 }
 sort.Strings(out.ImpactedClaimIDs);sort.Strings(out.UnaffectedClaimIDs)
 out.GraphVerdict=f.verdict()
 // A correct DAG evaluation over caller-supplied statuses is not yet an
 // authenticated recovery campaign. Trust bootstrap, source adapters and
 // mutation-boundary enforcement remain external.
 f.hold("graph_inputs_and_trust_snapshot_not_independently_authenticated")
 if in.Mode==RuntimeCandidate{f.hold("runtime_graph_adapter_not_implemented")}
 out.CampaignVerdict=Indeterminate
 out.ProductionActionPermitted=false
 out.ReasonCodes=f.sorted()
 return out
}
func digestJSON(v any)string{b,e:=json.Marshal(v);if e!=nil{return ""};h:=sha256.Sum256(b);return hex.EncodeToString(h[:])}
func FormatGraphDigest(nodes []Node,edges []Edge,roots []string)string{return fmt.Sprintf("%s",GraphDigest(nodes,edges,roots))}
