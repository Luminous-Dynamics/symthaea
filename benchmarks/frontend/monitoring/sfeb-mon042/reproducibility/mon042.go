// Package reproducibility evaluates preregistered replay schedules at the
// dependency-cluster level. It does not infer independence from distinct seeds.
package reproducibility

import (
 "crypto/sha256"
 "encoding/hex"
 "encoding/json"
 "fmt"
 "math"
 "sort"
 "strings"
 "time"
)

const (
 Profile = "mon042-dependence-aware-input-v1"
 Version = "1.0.0"
 MaxUnits = 1000
 MaxRunsPerUnit = 200
 MaxDomains = 64
 MaxIDLength = 256
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

type Observation struct {
 UnitID string `json:"unit_id"`
 ReplayID string `json:"replay_id"`
 TrialID string `json:"trial_id"`
 ScenarioSHA256 string `json:"scenario_sha256"`
 RecoveryBoundarySHA256 string `json:"recovery_boundary_sha256"`
 FaultSetSHA256 string `json:"fault_set_sha256"`
 CodeRevision string `json:"code_revision"`
 Seed int64 `json:"seed"`
 Verdict Verdict `json:"verdict"`
 FailureSignatureSHA256 string `json:"failure_signature_sha256"`
 Complete bool `json:"complete"`
 OutcomeReceiptSHA256 string `json:"outcome_receipt_sha256"`
}
type ExecutionUnit struct {
 UnitID string `json:"unit_id"`
 StratumID string `json:"stratum_id"`
 DomainInventoryComplete bool `json:"domain_inventory_complete"`
 DependencyDomains map[string]string `json:"dependency_domains"`
 DomainManifestSHA256 string `json:"domain_manifest_sha256"`
 RequiredRuns int `json:"required_runs"`
 Observations []Observation `json:"observations"`
}
type Input struct {
 Profile string `json:"profile"`
 Version string `json:"version"`
 Mode Mode `json:"mode"`
 EvaluatedAt time.Time `json:"evaluated_at"`
 TrialID string `json:"trial_id"`
 ScenarioSHA256 string `json:"scenario_sha256"`
 RecoveryBoundarySHA256 string `json:"recovery_boundary_sha256"`
 FaultSetSHA256 string `json:"fault_set_sha256"`
 CodeRevision string `json:"code_revision"`
 TargetFailureSignatureSHA256 string `json:"target_failure_signature_sha256"`
 RequiredSeeds []int64 `json:"required_seeds"`
 RequiredDomainClasses []string `json:"required_domain_classes"`
 ExpectedUnitCount int `json:"expected_unit_count"`
 RequiredRunsPerUnit int `json:"required_runs_per_unit"`
 RequiredIndependentClusters int `json:"required_independent_clusters"`
 MinimumReliability float64 `json:"minimum_reliability"`
 ConfidenceLevel float64 `json:"confidence_level"`
 Units []ExecutionUnit `json:"units"`
}
type ClusterResult struct {
 ClusterID string `json:"cluster_id"`
 UnitIDs []string `json:"unit_ids"`
 ExecutionUnitCount int `json:"execution_unit_count"`
 SharedDomains []string `json:"shared_domains"`
 RequiredRuns int `json:"required_runs"`
 ObservedRuns int `json:"observed_runs"`
 ExactMatches int `json:"exact_matches"`
 ClusterSuccess bool `json:"cluster_success"`
}
type Result struct {
 Profile string `json:"profile"`
 Version string `json:"version"`
 TrialID string `json:"trial_id"`
 DesignVerdict Verdict `json:"design_verdict"`
 StatisticalVerdict Verdict `json:"statistical_verdict"`
 CampaignVerdict Verdict `json:"campaign_verdict"`
 ProductionUsePermitted bool `json:"production_use_permitted"`
 ExpectedUnitCount int `json:"expected_unit_count"`
 ObservedUnitCount int `json:"observed_unit_count"`
 RequiredIndependentClusters int `json:"required_independent_clusters"`
 ObservedIndependentClusters int `json:"observed_independent_clusters"`
 SuccessfulClusters int `json:"successful_clusters"`
 FailedClusters int `json:"failed_clusters"`
 ObservedClusterSuccessRate float64 `json:"observed_cluster_success_rate"`
 OneSidedConfidenceLowerBound float64 `json:"one_sided_confidence_lower_bound"`
 ConfidenceLevel float64 `json:"confidence_level"`
 MinimumReliability float64 `json:"minimum_reliability"`
 DomainInventoryAuthenticated bool `json:"domain_inventory_authenticated"`
 Clusters []ClusterResult `json:"clusters"`
 ReasonCodes []string `json:"reason_codes"`
}

type flags struct { blocked, unknown bool; codes map[string]struct{} }
func newFlags()*flags { return &flags{codes:map[string]struct{}{}} }
func (f *flags) block(s string){f.blocked=true;f.codes[s]=struct{}{}}
func (f *flags) hold(s string){f.unknown=true;f.codes[s]=struct{}{}}
func (f *flags) verdict()Verdict{if f.blocked{return Blocked};if f.unknown{return Indeterminate};return Qualified}
func (f *flags) sorted()[]string{out:=make([]string,0,len(f.codes));for s:=range f.codes{out=append(out,s)};sort.Strings(out);return out}

func validID(s string)bool{return s!=""&&len(s)<=MaxIDLength&&!strings.ContainsAny(s," \t\r\n")}
func validDigest(s string)bool{if len(s)!=64||s!=strings.ToLower(s){return false};_,e:=hex.DecodeString(s);return e==nil}
func digestJSON(v any)string{b,e:=json.Marshal(v);if e!=nil{return ""};h:=sha256.Sum256(b);return hex.EncodeToString(h[:])}
func ObservationReceiptDigest(o Observation)string{o.OutcomeReceiptSHA256="";return digestJSON(o)}
func DomainManifestDigest(unitID string,domains map[string]string)string{return digestJSON(struct{UnitID string `json:"unit_id"`;Domains map[string]string `json:"domains"`}{unitID,domains})}
func validRevision(s string)bool{if len(s)!=40&&len(s)!=64{return false};if s!=strings.ToLower(s){return false};_,e:=hex.DecodeString(s);return e==nil}

type disjointSet struct{ parent, rank []int }
func newDisjointSet(n int)*disjointSet{p:=make([]int,n);r:=make([]int,n);for i:=range p{p[i]=i};return &disjointSet{p,r}}
func(d *disjointSet)find(x int)int{if d.parent[x]!=x{d.parent[x]=d.find(d.parent[x])};return d.parent[x]}
func(d *disjointSet)union(a,b int){ra,rb:=d.find(a),d.find(b);if ra==rb{return};if d.rank[ra]<d.rank[rb]{ra,rb=rb,ra};d.parent[rb]=ra;if d.rank[ra]==d.rank[rb]{d.rank[ra]++}}

type clusterBuild struct{ indices []int; shared map[string]struct{} }

func OneSidedClopperPearsonLower(successes,trials int,confidence float64)float64{
 if trials<=0||successes<0||successes>trials||confidence<=0||confidence>=1{return 0}
 if successes==0{return 0}
 alpha:=1-confidence
 if successes==trials{return math.Pow(alpha,1/float64(trials))}
 lo,hi:=0.0,float64(successes)/float64(trials)
 for i:=0;i<100;i++{mid:=(lo+hi)/2;if binomialTail(trials,successes,mid)<alpha{lo=mid}else{hi=mid}}
 return (lo+hi)/2
}
func binomialTail(n,k int,p float64)float64{
 if k<=0{return 1};if k>n{return 0};if p<=0{return 0};if p>=1{return 1}
 terms:=make([]float64,0,n-k+1);maxLog:=math.Inf(-1);lnN,_:=math.Lgamma(float64(n+1))
 for x:=k;x<=n;x++{lnX,_:=math.Lgamma(float64(x+1));lnRest,_:=math.Lgamma(float64(n-x+1));v:=lnN-lnX-lnRest+float64(x)*math.Log(p)+float64(n-x)*math.Log1p(-p);terms=append(terms,v);if v>maxLog{maxLog=v}}
 sum:=0.0;for _,v:=range terms{sum+=math.Exp(v-maxLog)};result:=math.Exp(maxLog)*sum;if result>1{return 1};if result<0{return 0};return result
}

// Evaluate clusters units that share any declared required failure domain.
// A cluster succeeds only when every preregistered observation in every
// member unit is complete and matches the exact target signature. Missing or
// incomplete runs are conservative cluster failures, never removed from N.
func Evaluate(in Input)Result{
 out:=Result{Profile:"mon042-dependence-aware-result-v1",Version:Version,TrialID:in.TrialID,DesignVerdict:Indeterminate,StatisticalVerdict:Indeterminate,CampaignVerdict:Indeterminate,ProductionUsePermitted:false,ExpectedUnitCount:in.ExpectedUnitCount,ObservedUnitCount:len(in.Units),RequiredIndependentClusters:in.RequiredIndependentClusters,ConfidenceLevel:in.ConfidenceLevel,MinimumReliability:in.MinimumReliability,DomainInventoryAuthenticated:false,Clusters:[]ClusterResult{},ReasonCodes:[]string{}}
 f:=newFlags()
 if in.Profile!=Profile||in.Version!=Version{f.block("profile_version_invalid")}
 if in.Mode!=OfflineFixture&&in.Mode!=RuntimeCandidate{f.block("mode_invalid")}
 if !validID(in.TrialID){f.block("trial_id_invalid")}
 if in.EvaluatedAt.IsZero(){f.hold("evaluation_time_missing")}
 if !validDigest(in.TargetFailureSignatureSHA256){f.block("target_signature_digest_invalid")}
 if !validDigest(in.ScenarioSHA256){f.block("scenario_digest_invalid")}
 if !validDigest(in.RecoveryBoundarySHA256){f.block("recovery_boundary_digest_invalid")}
 if !validDigest(in.FaultSetSHA256){f.block("fault_set_digest_invalid")}
 if !validRevision(in.CodeRevision){f.block("code_revision_invalid")}
 if in.ExpectedUnitCount<1||in.ExpectedUnitCount>MaxUnits{f.block("expected_unit_count_out_of_range")}
 if in.RequiredIndependentClusters<1||in.RequiredIndependentClusters>MaxUnits{f.block("required_cluster_count_out_of_range")}
 if in.RequiredRunsPerUnit<1||in.RequiredRunsPerUnit>MaxRunsPerUnit{f.block("required_runs_per_unit_out_of_range")}
 if len(in.RequiredSeeds)!=in.RequiredRunsPerUnit{f.block("required_seed_count_mismatch")}
 requiredSeeds:=map[int64]struct{}{}
 for _,seed:=range in.RequiredSeeds{if seed<0{f.block("required_seed_invalid")};if _,ok:=requiredSeeds[seed];ok{f.block("required_seed_duplicated")};requiredSeeds[seed]=struct{}{}}
 if in.MinimumReliability<=0||in.MinimumReliability>1||math.IsNaN(in.MinimumReliability){f.block("minimum_reliability_invalid")}
 if in.ConfidenceLevel<=0||in.ConfidenceLevel>=1||math.IsNaN(in.ConfidenceLevel){f.block("confidence_level_invalid")}
 if len(in.RequiredDomainClasses)==0||len(in.RequiredDomainClasses)>MaxDomains{f.block("required_domain_classes_out_of_range")}
 if len(in.Units)>MaxUnits{f.block("unit_limit_exceeded")}
 if len(in.Units)<in.ExpectedUnitCount{f.hold("preregistered_unit_count_incomplete")}
 if len(in.Units)>in.ExpectedUnitCount{f.block("unit_count_exceeds_preregistered_design")}
 if f.blocked{out.DesignVerdict=Blocked;out.CampaignVerdict=Indeterminate;out.ReasonCodes=f.sorted();return out}

 required:=map[string]struct{}{}
 for _,name:=range in.RequiredDomainClasses{if !validID(name){f.block("domain_class_invalid");continue};if _,ok:=required[name];ok{f.block("domain_class_duplicated")};required[name]=struct{}{}}
 if f.blocked{out.DesignVerdict=Blocked;out.ReasonCodes=f.sorted();return out}
 if len(in.Units)==0{f.hold("execution_units_missing");out.DesignVerdict=f.verdict();out.ReasonCodes=f.sorted();return out}

 seenUnits:=map[string]struct{}{};globalReplayIDs:=map[string]struct{}{};stratum:=""
 for i:=range in.Units{
  u:=&in.Units[i]
  if !validID(u.UnitID)||!validID(u.StratumID){f.block("unit_identity_invalid")}
  if _,ok:=seenUnits[u.UnitID];ok{f.block("unit_id_duplicated")};seenUnits[u.UnitID]=struct{}{}
  if i==0{stratum=u.StratumID}else if u.StratumID!=stratum{f.block("mixed_strata_require_separate_analysis")}
  if !u.DomainInventoryComplete{f.hold("domain_inventory_incomplete:"+u.UnitID)}
  if len(u.DependencyDomains)>MaxDomains{f.block("unit_domain_limit_exceeded:"+u.UnitID)}
  for name:=range required{value,ok:=u.DependencyDomains[name];if !ok||!validID(value){f.hold("required_domain_missing:"+u.UnitID+":"+name)}}
  if !validDigest(u.DomainManifestSHA256){f.hold("domain_manifest_digest_missing:"+u.UnitID)}else if u.DomainManifestSHA256!=DomainManifestDigest(u.UnitID,u.DependencyDomains){f.block("domain_manifest_digest_mismatch:"+u.UnitID)}
  if u.RequiredRuns!=in.RequiredRunsPerUnit{f.block("unit_preregistered_run_count_mismatch:"+u.UnitID)}
  if len(u.Observations)>in.RequiredRunsPerUnit{f.block("unit_observation_count_exceeds_preregistered:"+u.UnitID)}
  if len(u.Observations)<in.RequiredRunsPerUnit{f.hold("unit_observation_count_incomplete:"+u.UnitID)}
  replayIDs,seeds:=map[string]struct{}{},map[int64]struct{}{}
  for _,o:=range u.Observations{
   if !validID(o.ReplayID)||o.Seed<0||!validVerdict(o.Verdict)||!validDigest(o.FailureSignatureSHA256){f.block("observation_identity_or_result_invalid:"+u.UnitID)}
   if _,ok:=replayIDs[o.ReplayID];ok{f.block("replay_id_duplicated_within_unit:"+u.UnitID)};replayIDs[o.ReplayID]=struct{}{}
   if _,ok:=globalReplayIDs[o.ReplayID];ok{f.block("replay_id_duplicated_across_units:"+u.UnitID)};globalReplayIDs[o.ReplayID]=struct{}{}
   if _,ok:=seeds[o.Seed];ok{f.block("seed_duplicated_within_unit:"+u.UnitID)};seeds[o.Seed]=struct{}{}
   if _,ok:=requiredSeeds[o.Seed];!ok{f.block("observation_seed_not_preregistered:"+u.UnitID)}
   if o.UnitID!=u.UnitID||o.TrialID!=in.TrialID||o.ScenarioSHA256!=in.ScenarioSHA256||o.RecoveryBoundarySHA256!=in.RecoveryBoundarySHA256||o.FaultSetSHA256!=in.FaultSetSHA256||o.CodeRevision!=in.CodeRevision{f.block("observation_scope_mismatch:"+u.UnitID)}
   if !validDigest(o.OutcomeReceiptSHA256){f.block("observation_receipt_digest_invalid:"+u.UnitID)}else if o.OutcomeReceiptSHA256!=ObservationReceiptDigest(o){f.block("observation_receipt_digest_mismatch:"+u.UnitID)}
   if !o.Complete{f.hold("observation_incomplete:"+u.UnitID)}
  }
  if len(seeds)!=len(requiredSeeds){f.hold("unit_seed_schedule_incomplete:"+u.UnitID)}
 }
 if f.blocked{out.DesignVerdict=Blocked;out.ReasonCodes=f.sorted();return out}
 domainComplete:=true
 for _,u:=range in.Units{if !u.DomainInventoryComplete||!validDigest(u.DomainManifestSHA256){domainComplete=false};for name:=range required{if !validID(u.DependencyDomains[name]){domainComplete=false}}}
 if !domainComplete{out.DesignVerdict=Indeterminate;out.ReasonCodes=f.sorted();return out}

 ds:=newDisjointSet(len(in.Units));firstByDomain:=map[string]int{}
 for i,u:=range in.Units{for name:=range required{token:=name+"\x00"+u.DependencyDomains[name];if previous,ok:=firstByDomain[token];ok{ds.union(i,previous)}else{firstByDomain[token]=i}}}
 builds:=map[int]*clusterBuild{}
 for i:=range in.Units{root:=ds.find(i);if builds[root]==nil{builds[root]=&clusterBuild{indices:[]int{},shared:map[string]struct{}{}}};builds[root].indices=append(builds[root].indices,i)}
 for _,build:=range builds{
  counts:=map[string]map[string]int{}
  for _,idx:=range build.indices{for name:=range required{if counts[name]==nil{counts[name]=map[string]int{}};counts[name][in.Units[idx].DependencyDomains[name]]++}}
  for name,values:=range counts{for value,n:=range values{if n>1{build.shared[name+"="+value]=struct{}{}}}}
 }
 ordered:=make([]*clusterBuild,0,len(builds));for _,b:=range builds{ordered=append(ordered,b)}
 sort.Slice(ordered,func(i,j int)bool{return in.Units[ordered[i].indices[0]].UnitID<in.Units[ordered[j].indices[0]].UnitID})
 clusterSuccesses:=0
 for index,build:=range ordered{
  ids:=make([]string,0,len(build.indices));success:=true;requiredRuns,observedRuns,matches:=0,0,0
  for _,unitIndex:=range build.indices{
   u:=in.Units[unitIndex];ids=append(ids,u.UnitID);requiredRuns+=in.RequiredRunsPerUnit
   if len(u.Observations)!=in.RequiredRunsPerUnit{success=false}
   for _,o:=range u.Observations{observedRuns++;if o.Complete&&o.Verdict==Blocked&&o.FailureSignatureSHA256==in.TargetFailureSignatureSHA256{matches++}else{success=false}}
  }
  sort.Strings(ids);shared:=make([]string,0,len(build.shared));for s:=range build.shared{shared=append(shared,s)};sort.Strings(shared)
  out.Clusters=append(out.Clusters,ClusterResult{ClusterID:fmt.Sprintf("cluster-%03d",index+1),UnitIDs:ids,ExecutionUnitCount:len(build.indices),SharedDomains:shared,RequiredRuns:requiredRuns,ObservedRuns:observedRuns,ExactMatches:matches,ClusterSuccess:success})
  if success{clusterSuccesses++}
 }
 out.ObservedIndependentClusters=len(ordered);out.SuccessfulClusters=clusterSuccesses;out.FailedClusters=len(ordered)-clusterSuccesses
 if len(ordered)>0{out.ObservedClusterSuccessRate=float64(clusterSuccesses)/float64(len(ordered))}
 if len(ordered)>in.RequiredIndependentClusters{f.block("independent_cluster_count_exceeds_preregistered_design")}else if len(ordered)<in.RequiredIndependentClusters{f.hold("independent_cluster_count_below_preregistered_target")}
 if len(in.Units)!=in.ExpectedUnitCount{f.hold("execution_unit_frame_not_complete")}
 uniformClusterSize:=true
 if len(ordered)>1{baseSize:=len(ordered[0].indices);for _,build:=range ordered[1:]{if len(build.indices)!=baseSize{uniformClusterSize=false}}}
 if !uniformClusterSize{f.hold("unequal_dependency_cluster_sizes_require_stratified_analysis")}
 if len(ordered)==in.RequiredIndependentClusters&&len(in.Units)==in.ExpectedUnitCount&&uniformClusterSize{
  out.OneSidedConfidenceLowerBound=OneSidedClopperPearsonLower(clusterSuccesses,len(ordered),in.ConfidenceLevel)
  if out.OneSidedConfidenceLowerBound>=in.MinimumReliability&&!f.blocked&&!f.unknown{out.StatisticalVerdict=Qualified}else if !f.unknown&&out.OneSidedConfidenceLowerBound<in.MinimumReliability{out.StatisticalVerdict=Blocked;f.block("cluster_reproducibility_below_preregistered_threshold")}else{out.StatisticalVerdict=Indeterminate}
 }else{out.StatisticalVerdict=Indeterminate}
 // DesignVerdict describes the supplied dependency graph and preregistered
 // geometry only. It does not authenticate the domain evidence.
 out.DesignVerdict=f.verdict()
 // Domain inventories and outcome receipts are only hash-bound here. MON-041
 // must authenticate origins, key state, and nonce/attestation ledgers.
 f.hold("dependency_domain_custody_not_authenticated")
 if in.Mode==RuntimeCandidate{f.hold("runtime_independence_attestation_adapter_not_implemented")}
 out.CampaignVerdict=Indeterminate;out.ProductionUsePermitted=false
 out.DomainInventoryAuthenticated=false;out.ReasonCodes=f.sorted()
 return out
}
