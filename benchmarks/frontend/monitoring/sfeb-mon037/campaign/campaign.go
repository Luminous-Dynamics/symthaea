// Package campaign composes versioned MON recovery outputs with independent
// source-event truth. It is an offline decision evaluator, never a traffic controller.
package campaign

import (
 "crypto/sha256"
 "encoding/hex"
 "encoding/json"
 "sort"
 "strings"
 "time"
)

const (
 Profile="mon037-unified-recovery-campaign-input-v1"
 Version="1.0.0"
 MaxEvents=100000
 MaxTransactions=10000
 MaxSurfaces=10000
)

type Verdict string
const (
 Qualified Verdict="qualified"
 Blocked Verdict="blocked"
 Indeterminate Verdict="indeterminate"
)
type Mode string
const (
 OfflineFixture Mode="offline_fixture"
 RuntimeCandidate Mode="runtime_candidate"
)

type RecoveryBoundary struct {
 IncidentID string `json:"incident_id"`
 ClaimID string `json:"claim_id"`
 SourceClusterID string `json:"source_cluster_id"`
 Database string `json:"database"`
 SourceEpoch string `json:"source_epoch"`
 TimelineID string `json:"timeline_id"`
 SnapshotID string `json:"snapshot_id"`
 TerminalPosition string `json:"terminal_position"`
 ScopeDigest string `json:"scope_digest"`
 InventoryDigest string `json:"inventory_digest"`
}
type Event struct {
 EventID string `json:"event_id"`
 TransactionID string `json:"transaction_id"`
 Topic string `json:"topic"`
 DataCollection string `json:"data_collection"`
 Operation string `json:"operation"`
 PayloadSHA256 string `json:"payload_sha256"`
 TotalOrder int `json:"total_order"`
 CollectionOrder int `json:"collection_order"`
 Partition int `json:"partition"`
 Offset int64 `json:"offset"`
}
type TopicPartition struct {
 Topic string `json:"topic"`
 PartitionCount int `json:"partition_count"`
}
type TransactionProof struct {
 TransactionID string `json:"transaction_id"`
 ExpectedEventCount int `json:"expected_event_count"`
 BeginObserved bool `json:"begin_observed"`
 EndObserved bool `json:"end_observed"`
 EndEventCount int `json:"end_event_count"`
 DataCollections map[string]int `json:"data_collections"`
}
type OffsetProof struct {
 Profile string `json:"profile"`
 Version string `json:"version"`
 Target string `json:"target"`
 TargetReached bool `json:"target_reached"`
 CapturedAt time.Time `json:"captured_at"`
 OffsetsSHA256 string `json:"offsets_sha256"`
 BoundarySHA256 string `json:"boundary_sha256"`
 Offset struct {
  LSN string `json:"lsn"`
  LSNProc string `json:"lsn_proc"`
  LSNCommit string `json:"lsn_commit"`
  MessageType string `json:"messageType"`
  EventsProcessed int `json:"lsn_events_processed"`
  TxID string `json:"txId"`
 } `json:"offset"`
}
type Surface struct {
 ID string `json:"id"`
 Disposition string `json:"disposition"`
 SourceSHA256 string `json:"source_sha256,omitempty"`
 RestoredSHA256 string `json:"restored_sha256,omitempty"`
 JustificationSHA256 string `json:"justification_sha256,omitempty"`
 ExclusionManifestSHA256 string `json:"exclusion_manifest_sha256,omitempty"`
}
type Components struct {
 OffsetProof json.RawMessage `json:"mon033_offset_proof,omitempty"`
 TimelineResult json.RawMessage `json:"mon034_timeline_result"`
 ObserverResult json.RawMessage `json:"mon035_observer_result"`
 ContinuityResult json.RawMessage `json:"mon036_continuity_result"`
 ProofGraphResult json.RawMessage `json:"mon024_proof_graph_result"`
}
type Input struct {
 Profile string `json:"profile"`
 Version string `json:"version"`
 Boundary RecoveryBoundary `json:"recovery_boundary"`
 SourceTruthBoundarySHA256 string `json:"source_truth_boundary_sha256"`
 ObservedCaptureBoundarySHA256 string `json:"observed_capture_boundary_sha256"`
 TransactionReportBoundarySHA256 string `json:"transaction_report_boundary_sha256"`
 SurfaceInventoryBoundarySHA256 string `json:"surface_inventory_boundary_sha256"`
 Mon033BoundarySHA256 string `json:"mon033_boundary_sha256,omitempty"`
 Mon034BoundarySHA256 string `json:"mon034_boundary_sha256"`
 Mon035BoundarySHA256 string `json:"mon035_boundary_sha256"`
 Mon036BoundarySHA256 string `json:"mon036_boundary_sha256"`
 Mon024BoundarySHA256 string `json:"mon024_boundary_sha256"`
 TrialID string `json:"trial_id"`
 ScenarioID string `json:"scenario_id"`
 Mode Mode `json:"mode"`
 EvaluatedAt time.Time `json:"evaluated_at"`
 OffsetTargetRequired bool `json:"offset_target_required"`
 OffsetTarget string `json:"offset_target,omitempty"`
 RequiredClaimIDs []string `json:"required_claim_ids"`
 RequiredSurfaceIDs []string `json:"required_surface_ids"`
 SurfaceInventoryComplete bool `json:"surface_inventory_complete"`
 SurfaceInventorySHA256 string `json:"surface_inventory_sha256"`
 RequiredSurfaceIDsSHA256 string `json:"required_surface_ids_sha256"`
 SurfaceAssertionsSHA256 string `json:"surface_assertions_sha256"`
 SourceTruthComplete bool `json:"source_truth_complete"`
 SourceTruthManifestSHA256 string `json:"source_truth_manifest_sha256"`
 SourceEventsSHA256 string `json:"source_events_sha256"`
 ObservedCaptureComplete bool `json:"observed_capture_complete"`
 ObservedCaptureManifestSHA256 string `json:"observed_capture_manifest_sha256"`
 ObservedEventsSHA256 string `json:"observed_events_sha256"`
 TopicPartitionsSHA256 string `json:"topic_partitions_sha256"`
 TransactionReportComplete bool `json:"transaction_report_complete"`
 TransactionReportSHA256 string `json:"transaction_report_sha256"`
 TransactionProofsSHA256 string `json:"transaction_proofs_sha256"`
 SourceEvents []Event `json:"source_events"`
 ObservedEvents []Event `json:"observed_events"`
 TopicPartitions []TopicPartition `json:"topic_partitions"`
 Transactions []TransactionProof `json:"transactions"`
 ExactReplayDeduplicationProven bool `json:"exact_replay_deduplication_proven"`
 DeduplicationProofSHA256 string `json:"deduplication_proof_sha256,omitempty"`
 DeduplicationCoverageSHA256 string `json:"deduplication_coverage_sha256,omitempty"`
 SurfaceScope string `json:"surface_scope"`
 Surfaces []Surface `json:"surfaces"`
 RestoreChecks map[string]string `json:"restore_checks"`
 TrafficCurrentlyEnabled bool `json:"traffic_currently_enabled"`
 Components Components `json:"components"`
}
type Reconciliation struct {
 ExpectedEvents int `json:"expected_events"`
 ObservedEvents int `json:"observed_events"`
 UniqueObservedEvents int `json:"unique_observed_events"`
 DuplicateReplays int `json:"duplicate_replays"`
 MissingEventIDs []string `json:"missing_event_ids"`
 ExtraEventIDs []string `json:"extra_event_ids"`
 ConflictingEventIDs []string `json:"conflicting_event_ids"`
 MismatchedEventIDs []string `json:"mismatched_event_ids"`
 OrderingViolations []string `json:"ordering_violations"`
 TransactionViolations []string `json:"transaction_violations"`
 SurfaceCounts map[string]int `json:"surface_counts"`
}
type ComponentStatus struct {
 Name string `json:"name"`
 Verdict Verdict `json:"verdict"`
 ReasonCodes []string `json:"reason_codes"`
}
type Result struct {
 Profile string `json:"profile"`
 Version string `json:"version"`
 TrialID string `json:"trial_id"`
 ScenarioID string `json:"scenario_id"`
 Mode Mode `json:"mode"`
 ComponentVerdict Verdict `json:"component_verdict"`
 CampaignVerdict Verdict `json:"campaign_verdict"`
 TrafficActivationPermitted bool `json:"traffic_activation_permitted"`
 Components []ComponentStatus `json:"components"`
 Reconciliation Reconciliation `json:"reconciliation"`
 ReasonCodes []string `json:"reason_codes"`
}

type flags struct { blocked bool; unknown bool; reasons map[string]struct{} }
func newFlags()*flags { return &flags{reasons:map[string]struct{}{}} }
func(f *flags) block(s string){f.blocked=true;f.reasons[s]=struct{}{}}
func(f *flags) hold(s string){f.unknown=true;f.reasons[s]=struct{}{}}
func(f *flags) codes()[]string{out:=make([]string,0,len(f.reasons));for s:=range f.reasons{out=append(out,s)};sort.Strings(out);return out}
func(f *flags) verdict()Verdict{if f.blocked{return Blocked};if f.unknown{return Indeterminate};return Qualified}
func merge(dst,src *flags,prefix string){dst.blocked=dst.blocked||src.blocked;dst.unknown=dst.unknown||src.unknown;for s:=range src.reasons{dst.reasons[prefix+":"+s]=struct{}{}}}
func validID(s string)bool{return s!=""&&len(s)<=256&&!strings.ContainsAny(s," \t\r\n")}
func validDigest(s string)bool{if len(s)!=64{return false};_,err:=hex.DecodeString(s);return err==nil&&s==strings.ToLower(s)}
func contains(v []string,w string)bool{for _,s:=range v{if s==w{return true}};return false}
func canonicalDigest(value any) string {
 raw,err:=json.Marshal(value)
 if err!=nil{return ""}
 sum:=sha256.Sum256(raw)
 return hex.EncodeToString(sum[:])
}
func sourceEventsDigest(events []Event) string {
 copyEvents:=append([]Event(nil),events...)
 sort.Slice(copyEvents,func(i,j int)bool{
  a,b:=copyEvents[i],copyEvents[j]
  ka:=[]string{a.EventID,a.TransactionID,a.Topic,a.DataCollection,a.Operation,a.PayloadSHA256}
  kb:=[]string{b.EventID,b.TransactionID,b.Topic,b.DataCollection,b.Operation,b.PayloadSHA256}
  for n:=range ka{if ka[n]!=kb[n]{return ka[n]<kb[n]}}
  if a.TotalOrder!=b.TotalOrder{return a.TotalOrder<b.TotalOrder}
  if a.CollectionOrder!=b.CollectionOrder{return a.CollectionOrder<b.CollectionOrder}
  if a.Partition!=b.Partition{return a.Partition<b.Partition}
  return a.Offset<b.Offset
 })
 return canonicalDigest(copyEvents)
}
func observedEventsDigest(events []Event) string {
 copyEvents:=append([]Event(nil),events...)
 sort.Slice(copyEvents,func(i,j int)bool{
  a,b:=copyEvents[i],copyEvents[j]
  ka:=[]string{a.Topic,a.EventID,a.TransactionID,a.DataCollection,a.Operation,a.PayloadSHA256}
  kb:=[]string{b.Topic,b.EventID,b.TransactionID,b.DataCollection,b.Operation,b.PayloadSHA256}
  if a.Partition!=b.Partition{return a.Partition<b.Partition}
  if a.Offset!=b.Offset{return a.Offset<b.Offset}
  for n:=range ka{if ka[n]!=kb[n]{return ka[n]<kb[n]}}
  if a.TotalOrder!=b.TotalOrder{return a.TotalOrder<b.TotalOrder}
  return a.CollectionOrder<b.CollectionOrder
 })
 return canonicalDigest(copyEvents)
}
func transactionProofsDigest(proofs []TransactionProof) string {
 copyProofs:=append([]TransactionProof(nil),proofs...)
 sort.Slice(copyProofs,func(i,j int)bool{if copyProofs[i].TransactionID!=copyProofs[j].TransactionID{return copyProofs[i].TransactionID<copyProofs[j].TransactionID};return canonicalDigest(copyProofs[i])<canonicalDigest(copyProofs[j])})
 return canonicalDigest(copyProofs)
}
func surfaceIDsDigest(ids []string) string {
 copyIDs:=append([]string(nil),ids...)
 sort.Strings(copyIDs)
 return canonicalDigest(copyIDs)
}
func topicPartitionsDigest(values []TopicPartition) string {
 copyValues:=append([]TopicPartition(nil),values...)
 sort.Slice(copyValues,func(i,j int)bool{if copyValues[i].Topic!=copyValues[j].Topic{return copyValues[i].Topic<copyValues[j].Topic};return copyValues[i].PartitionCount<copyValues[j].PartitionCount})
 return canonicalDigest(copyValues)
}
func surfaceAssertionsDigest(values []Surface) string {
 copyValues:=append([]Surface(nil),values...)
 sort.Slice(copyValues,func(i,j int)bool{if copyValues[i].ID!=copyValues[j].ID{return copyValues[i].ID<copyValues[j].ID};if copyValues[i].Disposition!=copyValues[j].Disposition{return copyValues[i].Disposition<copyValues[j].Disposition};if copyValues[i].SourceSHA256!=copyValues[j].SourceSHA256{return copyValues[i].SourceSHA256<copyValues[j].SourceSHA256};if copyValues[i].RestoredSHA256!=copyValues[j].RestoredSHA256{return copyValues[i].RestoredSHA256<copyValues[j].RestoredSHA256};return copyValues[i].JustificationSHA256<copyValues[j].JustificationSHA256})
 return canonicalDigest(copyValues)
}
// DeriveFixtureRecordSetDigests is an authoring helper for public offline fixtures only.
// It must not be used by runtime adapters: runtime manifests must arrive pre-bound and
// independently authenticated by the caller/custody verifier.
func recoveryBoundaryDigest(b RecoveryBoundary) string { return canonicalDigest(b) }
func recoveryBoundaryComplete(b RecoveryBoundary) bool {
 return validID(b.IncidentID)&&validID(b.ClaimID)&&validID(b.SourceClusterID)&&validID(b.Database)&&
  validID(b.SourceEpoch)&&validID(b.TimelineID)&&validID(b.SnapshotID)&&validID(b.TerminalPosition)&&
  validDigest(b.ScopeDigest)&&validDigest(b.InventoryDigest)
}
func bindBoundary(label,actual,expected string,f *flags) {
 if !validDigest(actual) { f.hold(label+"_boundary_binding_missing"); return }
 if actual!=expected { f.block(label+"_recovery_boundary_mismatch") }
}
func checkBoundaryBindings(in Input,f *flags) {
 if !recoveryBoundaryComplete(in.Boundary) { f.hold("recovery_boundary_incomplete"); return }
 requiredClaims:=map[string]struct{}{}
 for _,id:=range in.RequiredClaimIDs {
  if !validID(id) { f.block("required_claim_id_invalid"); continue }
  if _,ok:=requiredClaims[id];ok { f.block("duplicate_required_claim_id:"+id) }
  requiredClaims[id]=struct{}{}
 }
 if _,ok:=requiredClaims[in.Boundary.ClaimID];!ok {
  f.block("recovery_boundary_claim_not_in_required_claims")
 }
 expected:=recoveryBoundaryDigest(in.Boundary)
 bindBoundary("source_truth",in.SourceTruthBoundarySHA256,expected,f)
 bindBoundary("observed_capture",in.ObservedCaptureBoundarySHA256,expected,f)
 bindBoundary("transaction_report",in.TransactionReportBoundarySHA256,expected,f)
 bindBoundary("surface_inventory",in.SurfaceInventoryBoundarySHA256,expected,f)
 bindBoundary("mon034",in.Mon034BoundarySHA256,expected,f)
 bindBoundary("mon035",in.Mon035BoundarySHA256,expected,f)
 bindBoundary("mon036",in.Mon036BoundarySHA256,expected,f)
 bindBoundary("mon024",in.Mon024BoundarySHA256,expected,f)
 if in.OffsetTargetRequired { bindBoundary("mon033",in.Mon033BoundarySHA256,expected,f) }
}
func bindEnvelopeBoundary(raw json.RawMessage, digest string) (json.RawMessage, bool) {
 if len(raw)==0 || !validDigest(digest) { return raw,false }
 var envelope map[string]json.RawMessage
 if err:=json.Unmarshal(raw,&envelope);err!=nil || envelope==nil { return raw,false }
 value,err:=json.Marshal(digest);if err!=nil{return raw,false}
 envelope["boundary_sha256"]=value
 encoded,err:=json.Marshal(envelope);if err!=nil{return raw,false}
 return encoded,true
}
func deriveComponentBoundaryBindings(in *Input,digest string) bool {
 var ok bool
 if in.Components.TimelineResult,ok=bindEnvelopeBoundary(in.Components.TimelineResult,digest);!ok{return false}
 if in.Components.ObserverResult,ok=bindEnvelopeBoundary(in.Components.ObserverResult,digest);!ok{return false}
 if in.Components.ContinuityResult,ok=bindEnvelopeBoundary(in.Components.ContinuityResult,digest);!ok{return false}
 if in.Components.ProofGraphResult,ok=bindEnvelopeBoundary(in.Components.ProofGraphResult,digest);!ok{return false}
 if in.OffsetTargetRequired {
  var proof OffsetProof
  if err:=json.Unmarshal(in.Components.OffsetProof,&proof);err!=nil||len(in.Components.OffsetProof)==0{return false}
  proof.BoundarySHA256=digest
  encoded,err:=json.Marshal(proof);if err!=nil{return false}
  in.Components.OffsetProof=encoded
 }
 return true
}
func DeriveFixtureRecordSetDigests(in *Input) bool {
 if in==nil||in.Mode!=OfflineFixture{return false}
 in.SourceEventsSHA256=sourceEventsDigest(in.SourceEvents)
 in.ObservedEventsSHA256=observedEventsDigest(in.ObservedEvents)
 in.TransactionProofsSHA256=transactionProofsDigest(in.Transactions)
 in.TopicPartitionsSHA256=topicPartitionsDigest(in.TopicPartitions)
 in.RequiredSurfaceIDsSHA256=surfaceIDsDigest(in.RequiredSurfaceIDs)
 in.SurfaceAssertionsSHA256=surfaceAssertionsDigest(in.Surfaces)
 if !recoveryBoundaryComplete(in.Boundary){return false}
 digest:=recoveryBoundaryDigest(in.Boundary)
 in.SourceTruthBoundarySHA256=digest;in.ObservedCaptureBoundarySHA256=digest;in.TransactionReportBoundarySHA256=digest;in.SurfaceInventoryBoundarySHA256=digest
 in.Mon034BoundarySHA256=digest;in.Mon035BoundarySHA256=digest;in.Mon036BoundarySHA256=digest;in.Mon024BoundarySHA256=digest
 if in.OffsetTargetRequired { in.Mon033BoundarySHA256=digest }
 if !deriveComponentBoundaryBindings(in,digest){return false}
 return true
}
func checkRecordManifest(label string,complete bool,artifactSHA,recordsSHA,expectedSHA string,f *flags) {
 if !complete||!validDigest(artifactSHA)||!validDigest(recordsSHA) {
  f.hold(label+"_manifest_missing_or_incomplete")
  return
 }
 if recordsSHA!=expectedSHA {
  f.block(label+"_record_set_digest_mismatch")
 }
}
func sameEvent(a,b Event)bool{return a.EventID==b.EventID&&a.TransactionID==b.TransactionID&&a.Topic==b.Topic&&a.DataCollection==b.DataCollection&&a.Operation==b.Operation&&a.PayloadSHA256==b.PayloadSHA256&&a.TotalOrder==b.TotalOrder&&a.CollectionOrder==b.CollectionOrder}
func decode(raw json.RawMessage,dst any,name string,f *flags){if len(raw)==0{f.hold(name+"_missing");return};if err:=json.Unmarshal(raw,dst);err!=nil{f.block(name+"_invalid_json")}}
func status(name string,f *flags)ComponentStatus{return ComponentStatus{Name:name,Verdict:f.verdict(),ReasonCodes:f.codes()}}
func checkVerdict(v,name string,f *flags){switch v{case "qualified":case "blocked":f.block(name+"_blocked");case "indeterminate":f.hold(name+"_indeterminate");default:f.hold(name+"_verdict_invalid")}}

type envelope34 struct {
 Profile string `json:"profile"`
 Version string `json:"version"`
 BoundarySHA256 string `json:"boundary_sha256"`
 Result struct { Verdict string `json:"verdict"` } `json:"result"`
}
type envelope35 struct {
 Profile string `json:"profile"`
 Version string `json:"version"`
 BoundarySHA256 string `json:"boundary_sha256"`
 Result struct {
  Verdict string `json:"verdict"`
  Independent bool `json:"independent"`
  SharedDomains []struct {Class string `json:"class"`;DomainID string `json:"domain_id"`} `json:"shared_domains"`
  ObserverAssessments []struct {ObserverID string `json:"observer_id"`;Eligible bool `json:"eligible"`} `json:"observer_assessments"`
 } `json:"result"`
}
type envelope36 struct {
 Profile string `json:"profile"`
 Version string `json:"version"`
 BoundarySHA256 string `json:"boundary_sha256"`
 LedgerIntegrityValid bool `json:"ledger_integrity_valid"`
 LedgerComplete bool `json:"ledger_complete"`
 Claims []struct {ClaimID string `json:"claim_id"`;Disposition string `json:"disposition"`} `json:"claims"`
}
type envelope24 struct {BoundarySHA256 string `json:"boundary_sha256"`;Verdict string `json:"verdict"`;MayEnableTraffic bool `json:"may_enable_traffic"`}

func Evaluate(in Input)Result{
 out:=Result{
  Profile:"mon037-unified-recovery-campaign-result-v1",Version:Version,TrialID:in.TrialID,ScenarioID:in.ScenarioID,Mode:in.Mode,
  Components:[]ComponentStatus{},
  Reconciliation:Reconciliation{MissingEventIDs:[]string{},ExtraEventIDs:[]string{},ConflictingEventIDs:[]string{},MismatchedEventIDs:[]string{},OrderingViolations:[]string{},TransactionViolations:[]string{},SurfaceCounts:map[string]int{}},
  ReasonCodes:[]string{},
 }
 global:=newFlags()
 if in.Profile!=Profile||in.Version!=Version{global.block("unsupported_profile_version")}
 if !validID(in.TrialID)||!validID(in.ScenarioID){global.block("trial_identity_invalid")}
 if in.Mode!=OfflineFixture&&in.Mode!=RuntimeCandidate{global.block("unsupported_mode")}
 if in.EvaluatedAt.IsZero(){global.hold("evaluation_time_missing")}
 if len(in.SourceEvents)>MaxEvents||len(in.ObservedEvents)>MaxEvents{global.block("event_limit_exceeded")}
 if len(in.Transactions)>MaxTransactions||len(in.Surfaces)>MaxSurfaces{global.block("transaction_surface_limit_exceeded")}
 names:=[]string{"mon033_offset","mon034_timeline","mon035_observers","mon036_continuity","cdc_reconciliation","transaction_closure","surface_coverage","restore_readiness","mon024_proof_graph"}
 fs:=map[string]*flags{};for _,n:=range names{fs[n]=newFlags()}
 fs["shared_recovery_boundary"]=newFlags();checkBoundaryBindings(in,fs["shared_recovery_boundary"])
 checkOffset(in,fs["mon033_offset"]);checkTimeline(in,fs["mon034_timeline"]);checkObservers(in,fs["mon035_observers"]);checkContinuity(in,fs["mon036_continuity"])
 reconcileEvents(in,&out.Reconciliation,fs["cdc_reconciliation"],fs["transaction_closure"])
 checkSurfaces(in,&out.Reconciliation,fs["surface_coverage"]);checkRestore(in,fs["restore_readiness"])
 out.Components=append(out.Components,status("shared_recovery_boundary",fs["shared_recovery_boundary"]))
 for _,n:=range names[:8]{out.Components=append(out.Components,status(n,fs[n]))}
 domain:=newFlags();merge(domain,fs["shared_recovery_boundary"],"shared_recovery_boundary");for _,n:=range names[:8]{merge(domain,fs[n],n)};out.ComponentVerdict=domain.verdict()
 checkProofGraph(in,out.ComponentVerdict,fs["mon024_proof_graph"]);out.Components=append(out.Components,status("mon024_proof_graph",fs["mon024_proof_graph"]))
 final:=newFlags();merge(final,global,"input");merge(final,fs["shared_recovery_boundary"],"shared_recovery_boundary");for _,n:=range names{merge(final,fs[n],n)}
 if in.Mode==OfflineFixture{final.hold("offline_fixture_cannot_qualify_live_recovery")}
 if in.Mode==RuntimeCandidate{final.hold("runtime_attestation_adapter_not_implemented")}
 out.CampaignVerdict=final.verdict()
 if in.TrafficCurrentlyEnabled&&out.CampaignVerdict!=Qualified{out.CampaignVerdict=Blocked;final.block("traffic_enabled_without_qualified_campaign")}
 out.TrafficActivationPermitted=false;out.ReasonCodes=final.codes();return out
}
func checkOffset(in Input,f *flags){
 if !in.OffsetTargetRequired{return};if in.OffsetTarget==""{f.hold("offset_target_missing");return}
 var p OffsetProof;decode(in.Components.OffsetProof,&p,"offset_proof",f)
 if p.Profile!="mon033-postgres-offset-target-v1"||p.Version!="1.0.0"{f.block("offset_profile_mismatch");return}
 if !validDigest(p.BoundarySHA256)||p.BoundarySHA256!=recoveryBoundaryDigest(in.Boundary){f.block("offset_recovery_boundary_mismatch");return}
 if p.Target!=in.OffsetTarget{f.block("offset_target_mismatch");return};if !p.TargetReached{f.hold("exact_offset_target_not_observed");return}
 if !validDigest(p.OffsetsSHA256)||p.CapturedAt.IsZero()||p.CapturedAt.After(in.EvaluatedAt){f.block("offset_digest_or_time_invalid");return}
 if p.Offset.LSN==""||p.Offset.LSNProc==""||p.Offset.LSNCommit==""{f.hold("offset_lsn_fields_missing");return}
 switch p.Target{case "lsn_events_processed_1":if p.Offset.EventsProcessed!=1||!contains([]string{"INSERT","UPDATE","DELETE","TRUNCATE"},p.Offset.MessageType){f.block("offset_predicate_mismatch")}
 case "transactional_message":if p.Offset.MessageType!="MESSAGE"||p.Offset.TxID==""{f.block("message_offset_predicate_mismatch")}
 default:f.block("unsupported_offset_target")}
}
func checkTimeline(in Input,f *flags){var e envelope34;decode(in.Components.TimelineResult,&e,"timeline_result",f);if e.Profile!="mon034-postgres-timeline-ancestry-result-v1"||e.Version!="1.0.0"{f.hold("timeline_profile_mismatch");return};bindBoundary("mon034",e.BoundarySHA256,recoveryBoundaryDigest(in.Boundary),f);checkVerdict(e.Result.Verdict,"timeline",f)}
func checkObservers(in Input,f *flags){
 var e envelope35;decode(in.Components.ObserverResult,&e,"observer_result",f);if e.Profile!="mon035-observer-independence-result-v1"||e.Version!="1.0.0"{f.hold("observer_profile_mismatch");return}
 bindBoundary("mon035",e.BoundarySHA256,recoveryBoundaryDigest(in.Boundary),f);checkVerdict(e.Result.Verdict,"observers",f);if !e.Result.Independent{if e.Result.Verdict=="qualified"{f.block("observer_qualified_without_independence")}else{f.hold("observer_independence_unproven")}}
 if len(e.Result.SharedDomains)>0{f.block("shared_observer_domain")};if len(e.Result.ObserverAssessments)!=2{f.hold("observer_count_not_two")};for _,a:=range e.Result.ObserverAssessments{if !a.Eligible{f.hold("observer_inventory_ineligible:"+a.ObserverID)}}
}
func checkContinuity(in Input,f *flags){
 var e envelope36;decode(in.Components.ContinuityResult,&e,"continuity_result",f);if e.Profile!="mon036-recovery-evidence-continuity-result-v1"||e.Version!="1.0.0"{f.hold("continuity_profile_mismatch");return}
 bindBoundary("mon036",e.BoundarySHA256,recoveryBoundaryDigest(in.Boundary),f)
 if !e.LedgerIntegrityValid{f.block("continuity_ledger_integrity_failed")};if !e.LedgerComplete{f.hold("continuity_ledger_incomplete")}
 claims:=map[string][]string{};for _,c:=range e.Claims{claims[c.ClaimID]=append(claims[c.ClaimID],c.Disposition)}
 required:=in.RequiredClaimIDs;if len(required)==0{for id:=range claims{required=append(required,id)};sort.Strings(required)}
 if len(required)==0{f.hold("required_continuity_claims_missing")}
 for _,id:=range required{ds:=claims[id];if len(ds)!=1{f.hold("claim_missing_or_ambiguous:"+id);continue};switch ds[0]{case "retain":case "revalidate":f.hold("claim_revalidation_required:"+id);case "invalidate":f.block("claim_invalidated:"+id);case "unresolved":f.hold("claim_unresolved:"+id);default:f.hold("claim_disposition_invalid:"+id)}}
}
func reconcileEvents(in Input,r *Reconciliation,cdc,txn *flags){
 r.ExpectedEvents=len(in.SourceEvents);r.ObservedEvents=len(in.ObservedEvents);if len(in.SourceEvents)==0{cdc.hold("source_truth_missing")};checkRecordManifest("source_truth",in.SourceTruthComplete,in.SourceTruthManifestSHA256,in.SourceEventsSHA256,sourceEventsDigest(in.SourceEvents),cdc);checkRecordManifest("observed_capture",in.ObservedCaptureComplete,in.ObservedCaptureManifestSHA256,in.ObservedEventsSHA256,observedEventsDigest(in.ObservedEvents),cdc);checkRecordManifest("topic_partition_inventory",in.ObservedCaptureComplete,in.ObservedCaptureManifestSHA256,in.TopicPartitionsSHA256,topicPartitionsDigest(in.TopicPartitions),cdc);checkRecordManifest("transaction_report",in.TransactionReportComplete,in.TransactionReportSHA256,in.TransactionProofsSHA256,transactionProofsDigest(in.Transactions),txn)
 want:=map[string]Event{};for _,e:=range in.SourceEvents{if !validID(e.EventID)||!validID(e.TransactionID)||!validID(e.Topic)||!validID(e.DataCollection)||!validDigest(e.PayloadSHA256)||e.TotalOrder<1||e.CollectionOrder<1{cdc.block("source_event_malformed");continue};if _,ok:=want[e.EventID];ok{cdc.block("duplicate_source_event_id:"+e.EventID)};want[e.EventID]=e}
 got:=map[string][]Event{};unique:=map[string]Event{};exactReplayIDs:=[]string{};for _,e:=range in.ObservedEvents{if !validID(e.EventID)||!validID(e.TransactionID)||!validID(e.Topic)||!validID(e.DataCollection)||!validDigest(e.PayloadSHA256)||e.TotalOrder<1||e.CollectionOrder<1||e.Partition<0||e.Offset<0{cdc.block("observed_event_malformed");continue};got[e.EventID]=append(got[e.EventID],e);if _,ok:=unique[e.EventID];!ok{unique[e.EventID]=e}}
 r.UniqueObservedEvents=len(unique)
 for id,w:=range want{records,ok:=got[id];if !ok{r.MissingEventIDs=append(r.MissingEventIDs,id);continue};for _,a:=range records{if !sameEvent(w,a){r.MismatchedEventIDs=append(r.MismatchedEventIDs,id);break}}}
 for id,records:=range got{if _,ok:=want[id];!ok{r.ExtraEventIDs=append(r.ExtraEventIDs,id)};if len(records)>1{exact:=true;for _,a:=range records[1:]{if !sameEvent(records[0],a){exact=false}};if exact{r.DuplicateReplays+=len(records)-1;exactReplayIDs=append(exactReplayIDs,id)}else{r.ConflictingEventIDs=append(r.ConflictingEventIDs,id)}}}
 for _,list:=range [][]string{r.MissingEventIDs,r.ExtraEventIDs,r.ConflictingEventIDs,r.MismatchedEventIDs}{sort.Strings(list)}
 sort.Strings(exactReplayIDs)
 if len(exactReplayIDs)>0 {
  if !in.ExactReplayDeduplicationProven||!validDigest(in.DeduplicationProofSHA256)||!validDigest(in.DeduplicationCoverageSHA256) {
   cdc.hold("exact_replay_not_proven_idempotent")
  } else if in.DeduplicationCoverageSHA256!=canonicalDigest(exactReplayIDs) {
   cdc.block("deduplication_proof_event_coverage_mismatch")
  }
 }
 if len(r.MissingEventIDs)>0{cdc.block("missing_events")};if len(r.ExtraEventIDs)>0{cdc.block("extra_events")};if len(r.ConflictingEventIDs)>0{cdc.block("conflicting_event_ids")};if len(r.MismatchedEventIDs)>0{cdc.block("event_payload_mismatch")}
 parts:=map[string]int{};for _,tp:=range in.TopicPartitions{if !validID(tp.Topic)||tp.PartitionCount<1{cdc.hold("topic_partition_inventory_invalid");continue};if _,ok:=parts[tp.Topic];ok{cdc.block("duplicate_topic_partition_inventory:"+tp.Topic)};parts[tp.Topic]=tp.PartitionCount}
 for _,e:=range in.SourceEvents{if _,ok:=parts[e.Topic];!ok{cdc.hold("topic_partition_inventory_missing:"+e.Topic)}};for topic,n:=range parts{if n!=1{cdc.hold("multi_partition_order_unsupported:"+topic)}}
 type topicPartitionKey struct{topic string;partition int}
 byTopicPartition:=map[topicPartitionKey][]Event{}
 for _,e:=range unique{key:=topicPartitionKey{topic:e.Topic,partition:e.Partition};byTopicPartition[key]=append(byTopicPartition[key],e)}
 for key,records:=range byTopicPartition{
  sort.Slice(records,func(i,j int)bool{return records[i].Offset<records[j].Offset})
  if declared,ok:=parts[key.topic];ok&&declared==1&&key.partition!=0{r.OrderingViolations=append(r.OrderingViolations,key.topic+":nonzero_partition")}
  for i:=1;i<len(records);i++{if records[i].Offset<=records[i-1].Offset{r.OrderingViolations=append(r.OrderingViolations,key.topic+":partition_offset_not_monotonic")}}
 }
 if len(r.OrderingViolations)>0{sort.Strings(r.OrderingViolations);cdc.block("per_topic_order_violation")}
 if len(in.Transactions)==0{txn.hold("transaction_proofs_missing")}
 txProofs:=map[string]struct{}{};byTx:=map[string][]Event{};for _,e:=range unique{byTx[e.TransactionID]=append(byTx[e.TransactionID],e)}
 for _,p:=range in.Transactions{if !validID(p.TransactionID)||p.ExpectedEventCount<=0{txn.block("transaction_proof_malformed");continue};if _,ok:=txProofs[p.TransactionID];ok{txn.block("duplicate_transaction_proof:"+p.TransactionID);continue};txProofs[p.TransactionID]=struct{}{};records:=byTx[p.TransactionID]
 if len(records)!=p.ExpectedEventCount{r.TransactionViolations=append(r.TransactionViolations,p.TransactionID+":event_count")};if !p.BeginObserved||!p.EndObserved{r.TransactionViolations=append(r.TransactionViolations,p.TransactionID+":begin_end")};if p.EndEventCount!=p.ExpectedEventCount{r.TransactionViolations=append(r.TransactionViolations,p.TransactionID+":end_count")}
 order:=make([]int,0,len(records));rowCollectionOrders:=map[string][]int{};rowCollectionCounts:=map[string]int{};rowCount:=0;messageCount:=0
 for _,e:=range records{
  order=append(order,e.TotalOrder)
  if e.Operation=="MESSAGE"{messageCount++;continue}
  rowCount++;rowCollectionCounts[e.DataCollection]++;rowCollectionOrders[e.DataCollection]=append(rowCollectionOrders[e.DataCollection],e.CollectionOrder)
 }
 sort.Ints(order);for i,n:=range order{if n!=i+1{r.TransactionViolations=append(r.TransactionViolations,p.TransactionID+":total_order");break}}
 for collection,orders:=range rowCollectionOrders{sort.Ints(orders);for i,n:=range orders{if n!=i+1{r.TransactionViolations=append(r.TransactionViolations,p.TransactionID+":row_collection_order:"+collection);break}}}
 declaredRowCount,rowDeclared:=p.DataCollections["public.ledger"]
 declaredMessageCount:=0;declaredMessageLabels:=0;malformedCollectionCount:=false
 for label,n:=range p.DataCollections{if n<0{malformedCollectionCount=true};if label!="public.ledger"{declaredMessageCount+=n;if n>0{declaredMessageLabels++}}}
 if malformedCollectionCount||!rowDeclared||len(rowCollectionCounts)!=1||rowCollectionCounts["public.ledger"]!=rowCount||declaredRowCount!=rowCount||declaredMessageCount!=messageCount||(messageCount>0&&declaredMessageLabels==0)||(messageCount==0&&declaredMessageLabels>0){
  r.TransactionViolations=append(r.TransactionViolations,p.TransactionID+":data_collections")
 }}
 for id:=range byTx{if _,ok:=txProofs[id];!ok{r.TransactionViolations=append(r.TransactionViolations,id+":proof_missing")}};if len(r.TransactionViolations)>0{sort.Strings(r.TransactionViolations);txn.block("transaction_not_closed")}
}
func checkSurfaces(in Input,r *Reconciliation,f *flags){
 if in.SurfaceScope!="full_state"&&in.SurfaceScope!="row_stream"{f.hold("surface_scope_invalid")};if len(in.Surfaces)==0{f.hold("surface_inventory_missing")};if len(in.RequiredSurfaceIDs)==0{f.hold("required_surface_inventory_missing")};checkRecordManifest("surface_inventory",in.SurfaceInventoryComplete,in.SurfaceInventorySHA256,in.RequiredSurfaceIDsSHA256,surfaceIDsDigest(in.RequiredSurfaceIDs),f);checkRecordManifest("surface_assertions",in.SurfaceInventoryComplete,in.SurfaceInventorySHA256,in.SurfaceAssertionsSHA256,surfaceAssertionsDigest(in.Surfaces),f)
 required:=map[string]struct{}{};for _,id:=range in.RequiredSurfaceIDs{if !validID(id){f.block("required_surface_id_invalid");continue};if _,ok:=required[id];ok{f.block("duplicate_required_surface_id:"+id)};required[id]=struct{}{}}
 seen:=map[string]struct{}{};for _,s:=range in.Surfaces{if !validID(s.ID){f.block("surface_id_invalid");continue};if _,ok:=seen[s.ID];ok{f.block("duplicate_surface:"+s.ID);continue};seen[s.ID]=struct{}{};if _,ok:=required[s.ID];!ok{f.block("surface_not_in_required_inventory:"+s.ID)};r.SurfaceCounts[s.Disposition]++
 switch s.Disposition{case "covered":if !validDigest(s.SourceSHA256)||!validDigest(s.RestoredSHA256){f.hold("surface_digest_missing:"+s.ID)}else if s.SourceSHA256!=s.RestoredSHA256{f.block("surface_digest_mismatch:"+s.ID)}
 case "not_applicable":if !validDigest(s.JustificationSHA256){f.hold("surface_na_unproven:"+s.ID)}
 case "excluded":if in.SurfaceScope=="full_state"{f.block("full_state_exclusion:"+s.ID)}else if !validDigest(s.ExclusionManifestSHA256){f.hold("row_stream_exclusion_manifest_missing:"+s.ID)}
 case "unknown":f.hold("surface_unknown:"+s.ID)
 default:f.block("surface_disposition_invalid:"+s.ID)}}
 for id:=range required{if _,ok:=seen[id];!ok{f.hold("required_surface_missing:"+id)}}
}
var restoreKeys=[]string{"artifact_validated","database_recovered","dependencies_ready","application_read_probe","application_write_readback","write_probe_rollback","application_invariants","old_primary_fenced","stale_endpoints_fenced","independent_service_acceptance","activation_order_verified","traffic_disabled_until_ready"}
func checkRestore(in Input,f *flags){if in.RestoreChecks==nil{f.hold("restore_checks_missing");return};for _,k:=range restoreKeys{switch in.RestoreChecks[k]{case "passed":case "failed":f.block("restore_failed:"+k);case "unknown","":f.hold("restore_unproven:"+k);default:f.block("restore_state_invalid:"+k)}}}
func checkProofGraph(in Input,domain Verdict,f *flags){var e envelope24;decode(in.Components.ProofGraphResult,&e,"proof_graph",f);bindBoundary("mon024",e.BoundarySHA256,recoveryBoundaryDigest(in.Boundary),f)
 switch e.Verdict{case "qualified":if !e.MayEnableTraffic{f.hold("proof_graph_not_activation_eligible")};if domain!=Qualified{f.block("proof_graph_promoted_nonqualified_components")};if in.Mode==OfflineFixture{f.block("synthetic_proof_promoted_to_production")}
case "blocked":f.block("proof_graph_blocked");case "indeterminate":f.hold("proof_graph_indeterminate");default:f.hold("proof_graph_verdict_invalid")};if e.MayEnableTraffic&&e.Verdict!="qualified"{f.block("proof_graph_activation_flag_inconsistent")}}
