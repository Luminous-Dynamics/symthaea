// Package custody verifies signed, claim-bound evidence receipts.
// Offline prototype only: trust-store bootstrap and atomic nonce consumption remain external.
package custody

import (
 "crypto/ed25519"
 "crypto/sha256"
 "encoding/base64"
 "encoding/hex"
 "encoding/json"
 "sort"
 "strings"
 "time"
)

const (
 Profile = "mon041-custody-input-v1"
 Version = "1.0.0"
 ReceiptProfile = "mon041-evidence-receipt-v1"
 MaxReceipts = 1000
 MaxAgeSeconds int64 = 604800
 MaxSkewSeconds int64 = 300
 signingDomain = "LUMINOUS-DYNAMICS-MON041-RECEIPT-V1\x00"
)
type Verdict string
const ( Qualified Verdict="qualified"; Blocked Verdict="blocked"; Indeterminate Verdict="indeterminate" )
type Mode string
const ( OfflineFixture Mode="offline_fixture"; RuntimeCandidate Mode="runtime_candidate" )
type Receipt struct {
 Profile string `json:"profile"`
 Version string `json:"version"`
 ReceiptID string `json:"receipt_id"`
 SessionID string `json:"session_id"`
 Nonce string `json:"nonce"`
 ClaimID string `json:"claim_id"`
 TrialID string `json:"trial_id"`
 RecoveryBoundarySHA256 string `json:"recovery_boundary_sha256"`
 SourceID string `json:"source_id"`
 EvidenceClass string `json:"evidence_class"`
 CapturePrincipal string `json:"capture_principal"`
 CapturedAt time.Time `json:"captured_at"`
 ExpiresAt time.Time `json:"expires_at"`
 Sequence uint64 `json:"sequence"`
 PreviousReceiptSHA256 string `json:"previous_receipt_sha256,omitempty"`
 ArtifactSHA256 string `json:"artifact_sha256"`
 ArtifactBytes int64 `json:"artifact_bytes"`
 MediaType string `json:"media_type"`
 SignerKeyID string `json:"signer_key_id"`
 SignatureBase64 string `json:"signature_base64"`
}
type TrustedKey struct {
 KeyID string `json:"key_id"`
 Principal string `json:"principal"`
 PublicKeyBase64 string `json:"public_key_base64"`
 NotBefore time.Time `json:"not_before"`
 NotAfter time.Time `json:"not_after"`
 RevokedAt *time.Time `json:"revoked_at,omitempty"`
}
type Input struct {
 Profile string `json:"profile"`
 Version string `json:"version"`
 Mode Mode `json:"mode"`
 EvaluatedAt time.Time `json:"evaluated_at"`
 ClaimID string `json:"claim_id"`
 TrialID string `json:"trial_id"`
 RecoveryBoundarySHA256 string `json:"recovery_boundary_sha256"`
 RequiredEvidenceClasses []string `json:"required_evidence_classes"`
 ExpectedSources map[string]string `json:"expected_sources"`
 IssuedNonces []string `json:"issued_nonces"`
 UsedNonces []string `json:"used_nonces"`
 TrustedKeys []TrustedKey `json:"trusted_keys"`
 Receipts []Receipt `json:"receipts"`
 MaxReceiptAgeSeconds int64 `json:"max_receipt_age_seconds"`
 AllowedClockSkewSeconds int64 `json:"allowed_clock_skew_seconds"`
}
type ReceiptResult struct {
 ReceiptID string `json:"receipt_id"`
 EvidenceClass string `json:"evidence_class"`
 Verdict Verdict `json:"verdict"`
 ReasonCodes []string `json:"reason_codes"`
 ReceiptSHA256 string `json:"receipt_sha256,omitempty"`
}
type Result struct {
 Profile string `json:"profile"`
 Version string `json:"version"`
 CustodyVerdict Verdict `json:"custody_verdict"`
 CampaignVerdict Verdict `json:"campaign_verdict"`
 ProductionActionPermitted bool `json:"production_action_permitted"`
 ReceiptCount int `json:"receipt_count"`
 AuthenticatedCount int `json:"authenticated_count"`
 AcceptedEvidenceClasses []string `json:"accepted_evidence_classes"`
 Receipts []ReceiptResult `json:"receipts"`
 ReasonCodes []string `json:"reason_codes"`
}
type flags struct { blocked, unknown bool; codes map[string]struct{} }
func newFlags()*flags { return &flags{codes:map[string]struct{}{}} }
func (f *flags) block(s string){ f.blocked=true; f.codes[s]=struct{}{} }
func (f *flags) hold(s string){ f.unknown=true; f.codes[s]=struct{}{} }
func (f *flags) verdict()Verdict { if f.blocked{return Blocked}; if f.unknown{return Indeterminate}; return Qualified }
func (f *flags) sorted()[]string { out:=make([]string,0,len(f.codes)); for s:=range f.codes{out=append(out,s)}; sort.Strings(out); return out }
func validID(s string)bool{return s!=""&&len(s)<=256&&!strings.ContainsAny(s," \t\r\n")}
func validDigest(s string)bool{if len(s)!=64||s!=strings.ToLower(s){return false};_,e:=hex.DecodeString(s);return e==nil}
func validNonce(s string)bool{if len(s)!=64||s!=strings.ToLower(s){return false};b,e:=hex.DecodeString(s);return e==nil&&len(b)==32}

// CanonicalSigningBytes uses domain separation and deterministic JSON encoding.
func CanonicalSigningBytes(r Receipt)[]byte{r.SignatureBase64="";b,_:=json.Marshal(r);return append([]byte(signingDomain),b...)}
func Digest(r Receipt)string{b,_:=json.Marshal(r);h:=sha256.Sum256(b);return hex.EncodeToString(h[:])}
func publicKey(s string)(ed25519.PublicKey,bool){b,e:=base64.StdEncoding.DecodeString(s);if e!=nil||len(b)!=ed25519.PublicKeySize{return nil,false};return ed25519.PublicKey(b),true}

func Evaluate(in Input)Result{
 out:=Result{Profile:"mon041-custody-result-v1",Version:Version,CustodyVerdict:Indeterminate,CampaignVerdict:Indeterminate,ProductionActionPermitted:false,ReceiptCount:len(in.Receipts),AcceptedEvidenceClasses:[]string{},Receipts:[]ReceiptResult{},ReasonCodes:[]string{}}
 f:=newFlags()
 if in.Profile!=Profile||in.Version!=Version{f.block("profile_version_invalid")}
 if in.Mode!=OfflineFixture&&in.Mode!=RuntimeCandidate{f.block("mode_invalid")}
 if in.EvaluatedAt.IsZero(){f.hold("evaluation_time_missing")}
 if !validID(in.ClaimID)||!validID(in.TrialID){f.block("claim_or_trial_identity_invalid")}
 if !validDigest(in.RecoveryBoundarySHA256){f.block("recovery_boundary_digest_invalid")}
 if in.MaxReceiptAgeSeconds<=0||in.MaxReceiptAgeSeconds>MaxAgeSeconds{f.block("max_receipt_age_out_of_range")}
 if in.AllowedClockSkewSeconds<0||in.AllowedClockSkewSeconds>MaxSkewSeconds{f.block("clock_skew_out_of_range")}
 if len(in.Receipts)==0{f.hold("receipts_missing")}
 if len(in.Receipts)>MaxReceipts{f.block("receipt_limit_exceeded")}
 if len(in.RequiredEvidenceClasses)==0{f.block("required_evidence_classes_missing")}
 if in.IssuedNonces==nil{f.hold("issued_nonce_registry_missing")}
 if in.UsedNonces==nil{f.hold("used_nonce_registry_missing")}
 if len(in.TrustedKeys)==0{f.hold("trust_store_missing")}
 if f.blocked{out.CustodyVerdict=Blocked;out.ReasonCodes=f.sorted();return out}
 keys:=map[string]struct{pub ed25519.PublicKey; principal string; from,to time.Time; revokedAt *time.Time}{}
 for _,k:=range in.TrustedKeys{
  if !validID(k.KeyID)||!validID(k.Principal){f.block("trusted_key_identity_invalid");continue}
  if _,ok:=keys[k.KeyID];ok{f.block("trusted_key_id_duplicated");continue}
  pub,ok:=publicKey(k.PublicKeyBase64);if !ok{f.block("trusted_public_key_invalid");continue}
  if k.NotBefore.IsZero()||k.NotAfter.IsZero()||!k.NotAfter.After(k.NotBefore){f.block("trusted_key_validity_invalid");continue}
  keys[k.KeyID]=struct{pub ed25519.PublicKey;principal string;from,to time.Time;revokedAt *time.Time}{pub,k.Principal,k.NotBefore,k.NotAfter,k.RevokedAt}
 }
 issued:=map[string]struct{}{};for _,n:=range in.IssuedNonces{if !validNonce(n){f.block("issued_nonce_invalid");continue};if _,ok:=issued[n];ok{f.block("issued_nonce_duplicated")};issued[n]=struct{}{}}
 used:=map[string]struct{}{};for _,n:=range in.UsedNonces{if !validNonce(n){f.block("used_nonce_invalid");continue};if _,ok:=used[n];ok{f.block("used_nonce_duplicated")};used[n]=struct{}{}}
 required:=map[string]struct{}{}
 for _,c:=range in.RequiredEvidenceClasses{
  if !validID(c){f.block("required_evidence_class_invalid");continue}
  if _,ok:=required[c];ok{f.block("required_evidence_class_duplicated")};required[c]=struct{}{}
  if s,ok:=in.ExpectedSources[c];!ok||!validID(s){f.hold("expected_source_missing:"+c)}
 }
 seenIDs,seenNonces,seenClasses:=map[string]struct{}{},map[string]struct{}{},map[string]struct{}{}
 valid:=[]Receipt{}
 for _,r:=range in.Receipts{
  rf:=newFlags()
  if r.Profile!=ReceiptProfile||r.Version!=Version{rf.block("receipt_profile_version_invalid")}
  if !validID(r.ReceiptID)||!validID(r.SessionID)||!validID(r.SourceID)||!validID(r.EvidenceClass)||!validID(r.CapturePrincipal)||!validID(r.SignerKeyID){rf.block("receipt_identity_invalid")}
  if _,ok:=seenIDs[r.ReceiptID];ok{rf.block("receipt_id_duplicated")};seenIDs[r.ReceiptID]=struct{}{}
  if !validNonce(r.Nonce){rf.block("receipt_nonce_invalid")}else{if _,ok:=issued[r.Nonce];!ok{rf.block("receipt_nonce_not_issued")};if _,ok:=used[r.Nonce];ok{rf.block("receipt_nonce_replayed")};if _,ok:=seenNonces[r.Nonce];ok{rf.block("receipt_nonce_duplicated")};seenNonces[r.Nonce]=struct{}{}}
  if r.ClaimID!=in.ClaimID||r.TrialID!=in.TrialID||r.RecoveryBoundarySHA256!=in.RecoveryBoundarySHA256{rf.block("receipt_claim_scope_mismatch")}
  if s,ok:=in.ExpectedSources[r.EvidenceClass];ok&&s!=r.SourceID{rf.block("receipt_source_scope_mismatch")}
  if _,ok:=required[r.EvidenceClass];!ok{rf.block("receipt_evidence_class_unexpected")}
  if _,ok:=seenClasses[r.EvidenceClass];ok{rf.block("receipt_evidence_class_duplicated")};seenClasses[r.EvidenceClass]=struct{}{}
  if !validDigest(r.ArtifactSHA256)||r.ArtifactBytes<=0||strings.TrimSpace(r.MediaType)==""{rf.block("receipt_artifact_descriptor_invalid")}
  if r.CapturedAt.IsZero()||r.ExpiresAt.IsZero()||!r.ExpiresAt.After(r.CapturedAt){rf.block("receipt_time_window_invalid")}
  if !in.EvaluatedAt.IsZero()&&!r.CapturedAt.IsZero()&&r.CapturedAt.After(in.EvaluatedAt.Add(time.Duration(in.AllowedClockSkewSeconds)*time.Second)){rf.block("receipt_captured_in_future")}
  if !in.EvaluatedAt.IsZero()&&!r.ExpiresAt.IsZero()&&!r.ExpiresAt.After(in.EvaluatedAt){rf.block("receipt_expired")}
  if !in.EvaluatedAt.IsZero()&&!r.CapturedAt.IsZero()&&r.CapturedAt.Before(in.EvaluatedAt.Add(-time.Duration(in.MaxReceiptAgeSeconds)*time.Second)){rf.block("receipt_stale")}
  k,ok:=keys[r.SignerKeyID]
  if !ok{rf.hold("signer_key_not_in_trust_store")}else{
   if k.revokedAt!=nil&&!k.revokedAt.After(r.CapturedAt){rf.block("signer_key_revoked_at_capture")}
   if k.principal!=r.CapturePrincipal{rf.block("signer_principal_mismatch")}
   if r.CapturedAt.Before(k.from)||r.CapturedAt.After(k.to){rf.block("signer_key_not_valid_at_capture")}
   sig,e:=base64.StdEncoding.DecodeString(r.SignatureBase64);if e!=nil||len(sig)!=ed25519.SignatureSize||!ed25519.Verify(k.pub,CanonicalSigningBytes(r),sig){rf.block("receipt_signature_invalid")}
  }
  codes:=rf.sorted();out.Receipts=append(out.Receipts,ReceiptResult{ReceiptID:r.ReceiptID,EvidenceClass:r.EvidenceClass,Verdict:rf.verdict(),ReasonCodes:codes,ReceiptSHA256:Digest(r)})
  if rf.verdict()==Qualified{out.AuthenticatedCount++;valid=append(valid,r)}
  for _,c:=range codes{if rf.blocked{f.block(c)}else{f.hold(c)}}
 }
 for c:=range required{if _,ok:=seenClasses[c];!ok{f.hold("required_evidence_class_missing:"+c)}}
 validateChains(in.Receipts,f)
 for _,r:=range valid{out.AcceptedEvidenceClasses=append(out.AcceptedEvidenceClasses,r.EvidenceClass)}
 sort.Strings(out.AcceptedEvidenceClasses)
 out.CustodyVerdict=f.verdict()
 // The trust-store snapshot and atomic nonce ledger are not independently attested.
 out.CampaignVerdict=Indeterminate;out.ProductionActionPermitted=false
 f.hold("offline_trust_store_and_nonce_ledger_not_independently_attested")
 if in.Mode==RuntimeCandidate{f.hold("runtime_capture_adapter_not_implemented")}
 out.ReasonCodes=f.sorted();return out
}
func validateChains(rs []Receipt,f *flags){
 groups:=map[string][]Receipt{};for _,r:=range rs{groups[r.SessionID]=append(groups[r.SessionID],r)}
 for _,g:=range groups{sort.Slice(g,func(i,j int)bool{return g[i].Sequence<g[j].Sequence});for i,r:=range g{
  if r.CapturePrincipal!=g[0].CapturePrincipal||r.SignerKeyID!=g[0].SignerKeyID{f.block("session_signer_changed");break}
  if r.Sequence!=uint64(i+1){f.block("receipt_sequence_gap");break}
  if i==0{if r.PreviousReceiptSHA256!=""{f.block("first_receipt_has_parent");break}}else if r.PreviousReceiptSHA256!=Digest(g[i-1]){f.block("receipt_chain_link_mismatch");break}
 }}
}
