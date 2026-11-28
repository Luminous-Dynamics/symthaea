package reproducibility

import (
 "crypto/sha256"
 "encoding/hex"
 "fmt"
 "testing"
 "time"
)

var fixtureNow=time.Date(2026,11,28,10,0,0,0,time.UTC)
var targetSHA=testSHA("target-failure")
var otherSHA=testSHA("other-failure")
var requiredDomains=[]string{"executor","host","hypervisor","storage","network","clock","identity","orchestrator","control_plane","credential_authority","region","external_service"}

func testSHA(s string)string{h:=sha256.Sum256([]byte(s));return hex.EncodeToString(h[:])}
func validInput(n int)Input{
 in:=Input{Profile:Profile,Version:Version,Mode:OfflineFixture,EvaluatedAt:fixtureNow,TrialID:"mon042-trial-001",ScenarioSHA256:testSHA("scenario-042"),RecoveryBoundarySHA256:testSHA("boundary-042"),FaultSetSHA256:testSHA("fault-set-042"),CodeRevision:"0123456789abcdef0123456789abcdef01234567",TargetFailureSignatureSHA256:targetSHA,RequiredSeeds:[]int64{1000},RequiredDomainClasses:append([]string(nil),requiredDomains...),ExpectedUnitCount:n,RequiredRunsPerUnit:1,RequiredIndependentClusters:n,MinimumReliability:0.90,ConfidenceLevel:0.95,Units:make([]ExecutionUnit,n)}
 for i:=0;i<n;i++{
  domains:=map[string]string{}
  for _,name:=range requiredDomains{domains[name]=fmt.Sprintf("%s-%03d",name,i)}
  unitID:=fmt.Sprintf("unit-%03d",i)
  o:=Observation{UnitID:unitID,ReplayID:fmt.Sprintf("replay-%03d",i),TrialID:in.TrialID,ScenarioSHA256:in.ScenarioSHA256,RecoveryBoundarySHA256:in.RecoveryBoundarySHA256,FaultSetSHA256:in.FaultSetSHA256,CodeRevision:in.CodeRevision,Seed:1000,Verdict:Blocked,FailureSignatureSHA256:targetSHA,Complete:true}
  o.OutcomeReceiptSHA256=ObservationReceiptDigest(o)
  u:=ExecutionUnit{UnitID:unitID,StratumID:"workload-profile-v1",DomainInventoryComplete:true,DependencyDomains:domains,DomainManifestSHA256:DomainManifestDigest(unitID,domains),RequiredRuns:1,Observations:[]Observation{o}}
  in.Units[i]=u
 }
 return in
}
func Test29IndependentClustersMeetNinetyPercentTarget(t *testing.T){
 got:=Evaluate(validInput(29))
 if got.StatisticalVerdict!=Qualified||got.DesignVerdict!=Qualified{t.Fatalf("29/29 independent cluster schedule should qualify its declared design: %+v",got)}
 if got.ObservedIndependentClusters!=29||got.SuccessfulClusters!=29||got.FailedClusters!=0{t.Fatalf("wrong cluster accounting: %+v",got)}
 if got.OneSidedConfidenceLowerBound<0.9018||got.OneSidedConfidenceLowerBound>0.9019{t.Fatalf("wrong CP cluster bound %.12f",got.OneSidedConfidenceLowerBound)}
 if got.CampaignVerdict!=Indeterminate||got.ProductionUsePermitted||got.DomainInventoryAuthenticated{t.Fatalf("domain snapshots are not independently authenticated: %+v",got)}
}
func Test28Of29ClustersDoesNotMeetNinetyPercentTarget(t *testing.T){
 in:=validInput(29);in.Units[28].Observations[0].FailureSignatureSHA256=otherSHA;in.Units[28].Observations[0].OutcomeReceiptSHA256=ObservationReceiptDigest(in.Units[28].Observations[0])
 got:=Evaluate(in)
 if got.StatisticalVerdict!=Blocked||got.OneSidedConfidenceLowerBound>=0.90{t.Fatalf("28/29 clusters must fail: %+v",got)}
 if got.SuccessfulClusters!=28||got.FailedClusters!=1{t.Fatalf("wrong cluster outcomes: %+v",got)}
}
func TestSharedHostCollapsesTwoUnitsIntoOneCluster(t *testing.T){
 in:=validInput(29);in.Units[1].DependencyDomains["host"]=in.Units[0].DependencyDomains["host"];in.Units[1].DomainManifestSHA256=DomainManifestDigest(in.Units[1].UnitID,in.Units[1].DependencyDomains)
 got:=Evaluate(in)
 if got.ObservedIndependentClusters!=28||got.StatisticalVerdict!=Indeterminate{t.Fatalf("shared host must reduce independent N: %+v",got)}
 found:=false;for _,c:=range got.Clusters{if len(c.UnitIDs)==2{for _,d:=range c.SharedDomains{if d=="host=host-000"{found=true}}}}
 if !found{t.Fatalf("shared host edge not reported: %+v",got.Clusters)}
}
func TestTransitiveSharedDomainsFormOneDependenceCluster(t *testing.T){
 in:=validInput(3);in.RequiredIndependentClusters=1
 in.Units[1].DependencyDomains["host"]=in.Units[0].DependencyDomains["host"]
 in.Units[2].DependencyDomains["storage"]=in.Units[1].DependencyDomains["storage"]
 for i:=range in.Units{in.Units[i].DomainManifestSHA256=DomainManifestDigest(in.Units[i].UnitID,in.Units[i].DependencyDomains)}
 got:=Evaluate(in)
 if got.ObservedIndependentClusters!=1||len(got.Clusters)!=1||len(got.Clusters[0].UnitIDs)!=3{t.Fatalf("transitive dependency graph not clustered: %+v",got)}
 shared:=map[string]bool{};for _,s:=range got.Clusters[0].SharedDomains{shared[s]=true}
 if !shared["host=host-000"]||!shared["storage=storage-001"]{t.Fatalf("shared edges missing: %+v",got.Clusters[0])}
}
func TestIncompleteReplayRemainsInDenominatorAsClusterFailure(t *testing.T){
 in:=validInput(29);in.Units[0].Observations[0].Complete=false;in.Units[0].Observations[0].OutcomeReceiptSHA256=ObservationReceiptDigest(in.Units[0].Observations[0])
 got:=Evaluate(in)
 if got.ObservedIndependentClusters!=29||got.FailedClusters!=1||got.SuccessfulClusters!=28{t.Fatalf("incomplete cluster must not disappear from denominator: %+v",got)}
 if got.StatisticalVerdict!=Indeterminate{t.Fatalf("incomplete observation must hold inference: %+v",got)}
}
func TestMissingRunDoesNotShrinkIndependentClusterCount(t *testing.T){
 in:=validInput(29);in.Units[0].Observations=nil
 got:=Evaluate(in)
 if got.ObservedIndependentClusters!=29||got.FailedClusters!=1{t.Fatalf("missing unit observation must not reduce N: %+v",got)}
 if got.StatisticalVerdict!=Indeterminate{t.Fatalf("missing observation must hold: %+v",got)}
}
func TestMissingCriticalDomainHoldsDesign(t *testing.T){
 in:=validInput(29);delete(in.Units[0].DependencyDomains,"control_plane");in.Units[0].DomainManifestSHA256=DomainManifestDigest(in.Units[0].UnitID,in.Units[0].DependencyDomains)
 got:=Evaluate(in)
 if got.DesignVerdict!=Indeterminate||got.StatisticalVerdict!=Indeterminate{t.Fatalf("missing critical domain must hold: %+v",got)}
}
func TestDomainManifestTamperingBlocks(t *testing.T){
 in:=validInput(29);in.Units[0].DependencyDomains["host"]="foreign-host"
 if got:=Evaluate(in);got.DesignVerdict!=Blocked{t.Fatalf("domain digest mismatch must block: %+v",got)}
}
func TestDuplicateUnitIDBlocks(t *testing.T){
 in:=validInput(29);in.Units[1].UnitID=in.Units[0].UnitID
 if got:=Evaluate(in);got.DesignVerdict!=Blocked{t.Fatalf("duplicate unit id must block: %+v",got)}
}
func TestMixedStrataMustBeAnalyzedSeparately(t *testing.T){
 in:=validInput(29);in.Units[1].StratumID="workload-profile-v2"
 if got:=Evaluate(in);got.DesignVerdict!=Blocked{t.Fatalf("mixed strata must block pooled estimate: %+v",got)}
}
func TestDuplicateSeedWithinUnitBlocks(t *testing.T){
 in:=validInput(29);in.RequiredRunsPerUnit=2;in.RequiredSeeds=[]int64{1000,1001};in.RequiredIndependentClusters=29
 for i:=range in.Units{in.Units[i].RequiredRuns=2;o:=in.Units[i].Observations[0];o.ReplayID=o.ReplayID+"-2";o.Seed=1000;o.OutcomeReceiptSHA256=ObservationReceiptDigest(o);in.Units[i].Observations=append(in.Units[i].Observations,o)}
 if got:=Evaluate(in);got.DesignVerdict!=Blocked{t.Fatalf("duplicate within-unit seeds must block: %+v",got)}
}
func TestReceiptTamperingBlocks(t *testing.T){
 in:=validInput(29);in.Units[0].Observations[0].FailureSignatureSHA256=otherSHA
 if got:=Evaluate(in);got.DesignVerdict!=Blocked{t.Fatalf("stale outcome receipt must block: %+v",got)}
}
func TestPreregisteredUnitCountCannotBeExpanded(t *testing.T){
 in:=validInput(2);in.Units=append(in.Units,in.Units[0]);if got:=Evaluate(in);got.DesignVerdict!=Blocked{t.Fatalf("extra unit must block optional stopping: %+v",got)}
}
func TestMissingTrustProvenanceNeverQualifiesCampaign(t *testing.T){
 in:=validInput(29);got:=Evaluate(in)
 if got.CampaignVerdict!=Indeterminate||got.ProductionUsePermitted{t.Fatalf("hash-only manifests cannot authorize campaign: %+v",got)}
}

func TestUnequalDependencyClusterSizesRequireSeparateStratification(t *testing.T){
 in:=validInput(3);in.RequiredIndependentClusters=2
 in.Units[1].DependencyDomains["host"]=in.Units[0].DependencyDomains["host"]
 in.Units[1].DomainManifestSHA256=DomainManifestDigest(in.Units[1].UnitID,in.Units[1].DependencyDomains)
 got:=Evaluate(in)
 if got.ObservedIndependentClusters!=2||got.StatisticalVerdict!=Indeterminate{t.Fatalf("unequal cluster sizes must not be pooled into one binomial interval: %+v",got)}
 found:=false;for _,r:=range got.ReasonCodes{if r=="unequal_dependency_cluster_sizes_require_stratified_analysis"{found=true}}
 if !found{t.Fatalf("unequal cluster-size reason missing: %+v",got.ReasonCodes)}
}
func TestReplayMustBindUnitAndSharedExperimentScope(t *testing.T){
 in:=validInput(29);in.Units[0].Observations[0].CodeRevision="ffffffffffffffffffffffffffffffffffffffff"
 if got:=Evaluate(in);got.DesignVerdict!=Blocked{t.Fatalf("replay scope drift must block: %+v",got)}
}
func TestReplayIDsAreUniqueAcrossUnits(t *testing.T){
 in:=validInput(29);in.Units[1].Observations[0].ReplayID=in.Units[0].Observations[0].ReplayID
 in.Units[1].Observations[0].OutcomeReceiptSHA256=ObservationReceiptDigest(in.Units[1].Observations[0])
 if got:=Evaluate(in);got.DesignVerdict!=Blocked{t.Fatalf("global replay ID collision must block: %+v",got)}
}

func TestClopperPearsonKnownAnswers(t *testing.T){
 if v:=OneSidedClopperPearsonLower(29,29,0.95);v<0.9018||v>0.9019{t.Fatalf("29/29 CP bound %.12f",v)}
 if v:=OneSidedClopperPearsonLower(28,29,0.95);v<0.846||v>0.847{t.Fatalf("28/29 CP bound %.12f",v)}
}
func TestConfidenceBoundMonotone(t *testing.T){
 prev:=0.0;for k:=0;k<=29;k++{v:=OneSidedClopperPearsonLower(k,29,.95);if v<prev{t.Fatalf("bound decreases at %d",k)};prev=v}
}
func BenchmarkEvaluate29IndependentClusters(b *testing.B){
 in:=validInput(29);b.ReportAllocs();b.ResetTimer();for i:=0;i<b.N;i++{_ = Evaluate(in)}
}
