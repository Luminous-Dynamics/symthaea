// Package capability evaluates whether a version/configuration-bound capture
// capability covers every item in an independently declared recovery scope.
// It is an offline evaluator: it never authenticates evidence or authorizes production.
package capability

import (
 "crypto/sha256"
 "encoding/hex"
 "encoding/json"
 "fmt"
 "sort"
 "strings"
 "time"
)

const (
 Profile = "mon044-capture-capability-input-v1"
 Version = "1.0.0"
 MaxItems = 100000
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
type CapabilityStatus string
const (
 Supported CapabilityStatus = "supported"
 Conditional CapabilityStatus = "conditional"
 Unsupported CapabilityStatus = "unsupported"
 Excluded CapabilityStatus = "excluded"
 NotApplicable CapabilityStatus = "not_applicable"
 Unknown CapabilityStatus = "unknown"
)
type EvidenceStatus string
const (
 EvidencePassed = "passed"
 EvidenceFailed = "failed"
 EvidenceMissing = "missing"
)

type ConnectorIdentity struct {
 ConnectorID string `json:"connector_id"`
 ConnectorVersion string `json:"connector_version"`
 DistributionSHA256 string `json:"distribution_sha256"`
 DatabaseEngine string `json:"database_engine"`
 DatabaseVersion string `json:"database_version"`
 LogicalDecodingPlugin string `json:"logical_decoding_plugin"`
 BrokerVersion string `json:"broker_version"`
 ClientVersion string `json:"client_version"`
 ConfigurationSHA256 string `json:"configuration_sha256"`
 PublicationSHA256 string `json:"publication_sha256"`
 SchemaHistorySHA256 string `json:"schema_history_sha256"`
 SnapshotMode string `json:"snapshot_mode"`
 TransactionMetadataEnabled bool `json:"transaction_metadata_enabled"`
 TransactionMetadataEpoch string `json:"transaction_metadata_epoch"`
 OperationFiltersSHA256 string `json:"operation_filters_sha256"`
 ColumnPolicySHA256 string `json:"column_policy_sha256"`
 TopicRoutingSHA256 string `json:"topic_routing_sha256"`
}
type RequiredItem struct {
 ItemID string `json:"item_id"`
 StateDomain string `json:"state_domain"`
 ObjectID string `json:"object_id"`
 Operation string `json:"operation"`
 Required bool `json:"required"`
 ScopeSHA256 string `json:"scope_sha256"`
}
type ConformanceEvidence struct {
 CapabilityIdentitySHA256 string `json:"capability_identity_sha256"`
 TestProfileSHA256 string `json:"test_profile_sha256"`
 ReportSHA256 string `json:"report_sha256"`
 Status EvidenceStatus `json:"status"`
}
type PrerequisiteEvidence struct {
 PrerequisiteID string `json:"prerequisite_id"`
 EvidenceSHA256 string `json:"evidence_sha256"`
 Status EvidenceStatus `json:"status"`
}
type CapabilityItem struct {
 ItemID string `json:"item_id"`
 ScopeSHA256 string `json:"scope_sha256"`
 Status CapabilityStatus `json:"status"`
 Conformance *ConformanceEvidence `json:"conformance,omitempty"`
 RequiredPrerequisites []string `json:"required_prerequisites,omitempty"`
 PrerequisiteEvidence []PrerequisiteEvidence `json:"prerequisite_evidence,omitempty"`
 AuthorityReference string `json:"authority_reference,omitempty"`
 AuthorityDocumentSHA256 string `json:"authority_document_sha256,omitempty"`
 JustificationSHA256 string `json:"justification_sha256,omitempty"`
}
type Manifest struct {
 Identity ConnectorIdentity `json:"identity"`
 Items []CapabilityItem `json:"items"`
}
type Input struct {
 Profile string `json:"profile"`
 Version string `json:"version"`
 Mode Mode `json:"mode"`
 EvaluatedAt time.Time `json:"evaluated_at"`
 TrialID string `json:"trial_id"`
 CapabilityIdentitySHA256 string `json:"capability_identity_sha256"`
 CapabilityManifestSHA256 string `json:"capability_manifest_sha256"`
 RequiredScopeSHA256 string `json:"required_scope_sha256"`
 RequiredScopeSourceID string `json:"required_scope_source_id"`
 RequiredScopeProvenanceSHA256 string `json:"required_scope_provenance_sha256"`
 RequiredItems []RequiredItem `json:"required_items"`
 Manifest Manifest `json:"manifest"`
}
type ItemResult struct {
 ItemID string `json:"item_id"`
 Status CapabilityStatus `json:"status"`
 Required bool `json:"required"`
 Verdict Verdict `json:"verdict"`
 ReasonCodes []string `json:"reason_codes"`
}
type Result struct {
 Profile string `json:"profile"`
 Version string `json:"version"`
 TrialID string `json:"trial_id"`
 CapabilityVerdict Verdict `json:"capability_verdict"`
 CoverageVerdict Verdict `json:"coverage_verdict"`
 CampaignVerdict Verdict `json:"campaign_verdict"`
 ProductionActionPermitted bool `json:"production_action_permitted"`
 RequiredItemCount int `json:"required_item_count"`
 CoveredRequiredCount int `json:"covered_required_count"`
 UnsupportedRequired []string `json:"unsupported_required"`
 ExcludedRequired []string `json:"excluded_required"`
 UnknownRequired []string `json:"unknown_required"`
 MissingRequired []string `json:"missing_required"`
 OutOfScopeDeclarations []string `json:"out_of_scope_declarations"`
 OptionalNegativeSpace []string `json:"optional_negative_space"`
 Items []ItemResult `json:"items"`
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
func digest(v any)string{b,e:=json.Marshal(v);if e!=nil{return ""};h:=sha256.Sum256(b);return hex.EncodeToString(h[:])}
func IdentityDigest(v ConnectorIdentity)string{return digest(v)}
func RequiredScopeDigest(v []RequiredItem)string{
 c:=append([]RequiredItem(nil),v...)
 sort.Slice(c,func(i,j int)bool{return c[i].ItemID<c[j].ItemID})
 return digest(c)
}
func ManifestDigest(v Manifest)string{
 c:=v
 c.Items=make([]CapabilityItem,len(v.Items));copy(c.Items,v.Items)
 for i:=range c.Items{
  c.Items[i].RequiredPrerequisites=append([]string(nil),v.Items[i].RequiredPrerequisites...)
  sort.Strings(c.Items[i].RequiredPrerequisites)
  c.Items[i].PrerequisiteEvidence=append([]PrerequisiteEvidence(nil),v.Items[i].PrerequisiteEvidence...)
  sort.Slice(c.Items[i].PrerequisiteEvidence,func(a,b int)bool{return c.Items[i].PrerequisiteEvidence[a].PrerequisiteID<c.Items[i].PrerequisiteEvidence[b].PrerequisiteID})
 }
 sort.Slice(c.Items,func(i,j int)bool{return c.Items[i].ItemID<c.Items[j].ItemID})
 return digest(c)
}
func validateIdentity(v ConnectorIdentity,f *flags){
 if !validID(v.ConnectorID)||!validID(v.ConnectorVersion)||!validID(v.DatabaseEngine)||!validID(v.DatabaseVersion)||!validID(v.LogicalDecodingPlugin)||!validID(v.BrokerVersion)||!validID(v.ClientVersion)||!validID(v.SnapshotMode){f.block("capability_identity_fields_incomplete")}
 for name,d:=range map[string]string{"distribution":v.DistributionSHA256,"configuration":v.ConfigurationSHA256,"publication":v.PublicationSHA256,"schema_history":v.SchemaHistorySHA256,"operation_filters":v.OperationFiltersSHA256,"column_policy":v.ColumnPolicySHA256,"topic_routing":v.TopicRoutingSHA256}{if !validDigest(d){f.block("capability_identity_digest_invalid:"+name)}}
 if v.TransactionMetadataEnabled&&!validID(v.TransactionMetadataEpoch){f.block("transaction_metadata_activation_epoch_missing")}
}
func validStatus(s CapabilityStatus)bool{
 switch s{case Supported,Conditional,Unsupported,Excluded,NotApplicable,Unknown:return true};return false
}
func validEvidence(e *ConformanceEvidence,identity string,f *flags,prefix string){
 if e==nil{f.hold(prefix+"_conformance_evidence_missing");return}
 if e.CapabilityIdentitySHA256!=identity{f.block(prefix+"_conformance_identity_mismatch")}
 if !validDigest(e.TestProfileSHA256)||!validDigest(e.ReportSHA256){f.hold(prefix+"_conformance_digest_missing")}
 switch e.Status{case EvidencePassed:case EvidenceFailed:f.block(prefix+"_conformance_failed");case EvidenceMissing:f.hold(prefix+"_conformance_result_missing");default:f.block(prefix+"_conformance_status_invalid")}
}
func itemIDMap(items []CapabilityItem,f *flags)map[string]CapabilityItem{
 out:=map[string]CapabilityItem{}
 for _,item:=range items{
  if !validID(item.ItemID){f.block("capability_item_id_invalid");continue}
  if _,ok:=out[item.ItemID];ok{f.block("capability_item_duplicated:"+item.ItemID);continue}
  out[item.ItemID]=item
 }
 return out
}
func prereqCheck(item CapabilityItem,f *flags){
 if len(item.RequiredPrerequisites)==0{f.block("conditional_prerequisite_list_missing:"+item.ItemID);return}
 req:=map[string]struct{}{}
 for _,id:=range item.RequiredPrerequisites{
  if !validID(id){f.block("conditional_prerequisite_id_invalid:"+item.ItemID);continue}
  if _,ok:=req[id];ok{f.block("conditional_prerequisite_duplicated:"+item.ItemID)}
  req[id]=struct{}{}
 }
 seen:=map[string]struct{}{}
 for _,e:=range item.PrerequisiteEvidence{
  if _,ok:=req[e.PrerequisiteID];!ok{f.block("unexpected_prerequisite_evidence:"+item.ItemID);continue}
  if _,ok:=seen[e.PrerequisiteID];ok{f.block("duplicate_prerequisite_evidence:"+item.ItemID);continue}
  seen[e.PrerequisiteID]=struct{}{}
  if !validDigest(e.EvidenceSHA256){f.hold("prerequisite_evidence_digest_missing:"+item.ItemID+":"+e.PrerequisiteID)}
  switch e.Status{case EvidencePassed:case EvidenceFailed:f.block("prerequisite_failed:"+item.ItemID+":"+e.PrerequisiteID);case EvidenceMissing:f.hold("prerequisite_not_established:"+item.ItemID+":"+e.PrerequisiteID);default:f.block("prerequisite_status_invalid:"+item.ItemID)}
 }
 for id:=range req{if _,ok:=seen[id];!ok{f.hold("prerequisite_evidence_missing:"+item.ItemID+":"+id)}}
}

func Evaluate(in Input)Result{
 out:=Result{Profile:"mon044-capture-capability-result-v1",Version:Version,TrialID:in.TrialID,CapabilityVerdict:Indeterminate,CoverageVerdict:Indeterminate,CampaignVerdict:Indeterminate,ProductionActionPermitted:false,RequiredItemCount:len(in.RequiredItems),UnsupportedRequired:[]string{},ExcludedRequired:[]string{},UnknownRequired:[]string{},MissingRequired:[]string{},OutOfScopeDeclarations:[]string{},OptionalNegativeSpace:[]string{},Items:[]ItemResult{},ReasonCodes:[]string{}}
 f:=newFlags()
 coverage:=newFlags()
 if in.Profile!=Profile||in.Version!=Version{f.block("profile_version_invalid")}
 if in.Mode!=OfflineFixture&&in.Mode!=RuntimeCandidate{f.block("mode_invalid")}
 if !validID(in.TrialID){f.block("trial_id_invalid")}
 if in.EvaluatedAt.IsZero(){f.hold("evaluation_time_missing")}
 if len(in.RequiredItems)==0{f.block("required_scope_missing")}
 if len(in.RequiredItems)>MaxItems||len(in.Manifest.Items)>MaxItems{f.block("item_limit_exceeded")}
 validateIdentity(in.Manifest.Identity,f)
 if !validID(in.RequiredScopeSourceID){f.block("required_scope_source_missing")}
 if !validDigest(in.RequiredScopeProvenanceSHA256){f.hold("required_scope_provenance_missing")}
 identity:=IdentityDigest(in.Manifest.Identity)
 if !validDigest(in.CapabilityIdentitySHA256){f.block("capability_identity_digest_invalid")}else if in.CapabilityIdentitySHA256!=identity{f.block("capability_identity_digest_mismatch")}
 if !validDigest(in.CapabilityManifestSHA256){f.block("capability_manifest_digest_invalid")}else if in.CapabilityManifestSHA256!=ManifestDigest(in.Manifest){f.block("capability_manifest_digest_mismatch")}
 if !validDigest(in.RequiredScopeSHA256){f.block("required_scope_digest_invalid")}else if in.RequiredScopeSHA256!=RequiredScopeDigest(in.RequiredItems){f.block("required_scope_digest_mismatch")}
 if f.blocked{out.CapabilityVerdict=Blocked;out.CoverageVerdict=Blocked;coverage.block("capability_or_scope_contract_invalid");out.ReasonCodes=f.sorted();return out}
 if f.unknown{coverage.hold("capability_or_scope_provenance_incomplete")}

 required:=map[string]RequiredItem{}
 for _,item:=range in.RequiredItems{
  if !validID(item.ItemID)||!validID(item.StateDomain)||!validID(item.ObjectID)||!validID(item.Operation)||!validDigest(item.ScopeSHA256){f.block("required_scope_item_invalid");continue}
  if _,ok:=required[item.ItemID];ok{f.block("required_scope_item_duplicated:"+item.ItemID);continue}
  required[item.ItemID]=item
 }
 declared:=itemIDMap(in.Manifest.Items,f)
 if f.blocked{out.CapabilityVerdict=Blocked;out.CoverageVerdict=Blocked;coverage.block("required_scope_or_manifest_invalid");out.ReasonCodes=f.sorted();return out}

 for id,req:=range required{
  item,ok:=declared[id]
  if !ok{
   f.hold("required_capability_item_missing:"+id)
   coverage.hold("required_capability_item_missing:"+id)
   out.MissingRequired=append(out.MissingRequired,id)
   out.Items=append(out.Items,ItemResult{ItemID:id,Required:req.Required,Status:Unknown,Verdict:Indeterminate,ReasonCodes:[]string{"required_capability_item_missing"}})
   continue
  }
  local:=newFlags()
  if item.ScopeSHA256!=req.ScopeSHA256{local.block("capability_scope_mismatch")}
  if !validStatus(item.Status){local.block("capability_status_invalid")}
  switch item.Status{
  case Supported:
   validEvidence(item.Conformance,identity,local,id)
  case Conditional:
   validEvidence(item.Conformance,identity,local,id)
   prereqCheck(item,local)
  case Unsupported:
   if !validID(item.AuthorityReference)||!validDigest(item.AuthorityDocumentSHA256){local.hold("unsupported_classification_provenance_missing")}
   if req.Required{local.block("unsupported_required_capability");out.UnsupportedRequired=append(out.UnsupportedRequired,id)}else{out.OptionalNegativeSpace=append(out.OptionalNegativeSpace,id)}
  case Excluded:
   if !validID(item.AuthorityReference)||!validDigest(item.AuthorityDocumentSHA256){local.hold("exclusion_provenance_missing")}
   if req.Required{local.block("required_scope_excluded");out.ExcludedRequired=append(out.ExcludedRequired,id)}else{out.OptionalNegativeSpace=append(out.OptionalNegativeSpace,id)}
  case NotApplicable:
   if !validDigest(item.JustificationSHA256){local.hold("not_applicable_justification_missing")}
  case Unknown:
   local.hold("capability_status_unknown")
   if req.Required{out.UnknownRequired=append(out.UnknownRequired,id)}else{out.OptionalNegativeSpace=append(out.OptionalNegativeSpace,id)}
  }
  verdict:=local.verdict()
  if req.Required&&verdict==Qualified{out.CoveredRequiredCount++}
  if req.Required&&verdict==Indeterminate{out.UnknownRequired=appendUnique(out.UnknownRequired,id)}
  if !req.Required&&verdict!=Qualified{out.OptionalNegativeSpace=appendUnique(out.OptionalNegativeSpace,id)}
  for _,code:=range local.sorted(){
   if local.blocked{f.block(id+":"+code)}else{f.hold(id+":"+code)}
   if req.Required{if local.blocked{coverage.block(id+":"+code)}else{coverage.hold(id+":"+code)}}
  }
  out.Items=append(out.Items,ItemResult{ItemID:id,Status:item.Status,Required:req.Required,Verdict:verdict,ReasonCodes:local.sorted()})
 }
 for id:=range declared{
  if _,ok:=required[id];!ok{out.OutOfScopeDeclarations=append(out.OutOfScopeDeclarations,id)}
 }
 sort.Strings(out.OutOfScopeDeclarations)
 sort.Strings(out.UnsupportedRequired);sort.Strings(out.ExcludedRequired);sort.Strings(out.UnknownRequired);sort.Strings(out.MissingRequired);sort.Strings(out.OptionalNegativeSpace)
 sort.Slice(out.Items,func(i,j int)bool{return out.Items[i].ItemID<out.Items[j].ItemID})
 out.CapabilityVerdict=f.verdict()
 out.CoverageVerdict=coverage.verdict()
 // Offline evidence digests cannot authenticate the capability issuer, required
 // scope source, conformance runner or source capture. MON-041 is still required.
 f.hold("capability_and_scope_provenance_not_authenticated")
 if in.Mode==RuntimeCandidate{f.hold("runtime_capability_attestation_adapter_not_implemented")}
 out.CampaignVerdict=Indeterminate
 out.ProductionActionPermitted=false
 out.ReasonCodes=f.sorted()
 return out
}
func appendUnique(items []string,value string)[]string{for _,s:=range items{if s==value{return items}};return append(items,value)}
func FormatEvaluationError(err error)string{return fmt.Sprint(err)}
