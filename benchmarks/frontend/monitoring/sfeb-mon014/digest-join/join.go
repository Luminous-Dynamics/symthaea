// Package crossprofile joins already-verified evidence across profiles.
// It is not a cryptographic verifier and defines no receipt format.
package crossprofile

import (
 "encoding/hex"
 "strings"
)

type VerificationState string
const (
 Verified VerificationState = "verified"
 Missing VerificationState = "missing"
 Invalid VerificationState = "invalid"
)

type DigestContext struct {
 Profile, Version, PayloadType, Canonicalization, Encoding, Purpose, Subject string
}
type DigestRef struct {
 Algorithm, Value string
 Context DigestContext
 Verification VerificationState
}
type DigestRelation string
const (
 DigestEqual DigestRelation = "equal"
 DigestDifferent DigestRelation = "different"
 DigestIncomparable DigestRelation = "incomparable"
 DigestIndeterminate DigestRelation = "indeterminate"
 DigestInvalid DigestRelation = "invalid"
)

// CompareDigestRefs treats a digest as a typed reference, not a bare hex string.
// Equal/Different require verified SHA-256 values and identical profile context.
func CompareDigestRefs(a,b DigestRef) DigestRelation {
 if a.Verification==Invalid || b.Verification==Invalid { return DigestInvalid }
 if a.Verification!=Verified || b.Verification!=Verified { return DigestIndeterminate }
 if !validSHA256(a.Value) || !validSHA256(b.Value) { return DigestInvalid }
 if !validContext(a.Context) || !validContext(b.Context) { return DigestIndeterminate }
 if !strings.EqualFold(a.Algorithm,"sha-256") || !strings.EqualFold(b.Algorithm,"sha-256") { return DigestIncomparable }
 if a.Context!=b.Context { return DigestIncomparable }
 if a.Value==b.Value { return DigestEqual }
 return DigestDifferent
}
func validContext(c DigestContext) bool {
 return strings.TrimSpace(c.Profile)!="" && strings.TrimSpace(c.Version)!="" &&
 strings.TrimSpace(c.PayloadType)!="" && strings.TrimSpace(c.Canonicalization)!="" &&
 strings.TrimSpace(c.Encoding)!="" && strings.TrimSpace(c.Purpose)!="" && strings.TrimSpace(c.Subject)!=""
}
func validSHA256(s string) bool {
 if len(s)!=64 || strings.ToLower(s)!=s { return false }
 b,err:=hex.DecodeString(s); return err==nil && len(b)==32
}

type EffectMode string
const (
 NotApplicable EffectMode = "not_applicable"
 DispatchedUnconfirmed EffectMode = "dispatched_unconfirmed"
 Confirmed EffectMode = "confirmed"
)
type EffectAttestation string
const (
 GateExecuted EffectAttestation = "gate_executed"
 RuntimeClaimed EffectAttestation = "runtime_claimed"
 HostServedObserved EffectAttestation = "host_served_observed"
)
type ProducerGrade string
const (
 NoExecutionClaim ProducerGrade = "no_execution_claim"
 RuntimeAssertion ProducerGrade = "runtime_assertion"
 GateObserved ProducerGrade = "gate_observed_effect_boundary"
)
type CapsuleAssessment struct {
 Valid bool
 EffectMode EffectMode
 Attestation EffectAttestation
 Grade ProducerGrade
 UnknownAttestation bool
 ResponseDigestPresent bool
 Findings []string
}

// AssessCapsuleEffect checks the AAC-06 effect-mode/attestation relationship.
// Signature, response digest bytes and SCITT registration are verified upstream.
func AssessCapsuleEffect(mode EffectMode, attestation string, responseDigestPresent bool) CapsuleAssessment {
 r:=CapsuleAssessment{EffectMode:mode,Attestation:EffectAttestation(attestation),Grade:NoExecutionClaim,ResponseDigestPresent:responseDigestPresent}
 switch mode {
 case NotApplicable:
  if attestation!="" { r.Findings=append(r.Findings,"attestation_present_for_not_applicable_effect"); return r }
  r.Valid=true; return r
 case DispatchedUnconfirmed:
  if attestation=="" { r.Findings=append(r.Findings,"attestation_required_after_dispatch"); return r }
 case Confirmed:
  if attestation=="" { r.Findings=append(r.Findings,"attestation_required_for_confirmed_effect"); return r }
  if !responseDigestPresent { r.Findings=append(r.Findings,"response_digest_required_for_confirmed_effect"); return r }
 default:
  r.Findings=append(r.Findings,"unknown_effect_mode"); return r
 }
 switch EffectAttestation(attestation) {
 case GateExecuted: r.Grade=GateObserved
 case RuntimeClaimed,HostServedObserved: r.Grade=RuntimeAssertion
 default:
  r.UnknownAttestation=true; r.Grade=RuntimeAssertion
  r.Findings=append(r.Findings,"unknown_attestation_not_graded_above_runtime_claimed")
 }
 r.Valid=true; return r
}

type SinkOutcome string
const (
 SinkCommitted SinkOutcome = "committed"
 SinkRefused SinkOutcome = "refused"
 SinkUnknown SinkOutcome = "unresolved"
)
type Independence string
const (
 IndependenceVerified Independence = "independent"
 SharedControlRoot Independence = "shared_control_root"
 IndependenceUnknown Independence = "uncheckable"
)
type AuthorityAtCommit string
const (
 AuthorityValid AuthorityAtCommit = "valid"
 AuthorityInvalid AuthorityAtCommit = "invalid"
 AuthorityUncheckable AuthorityAtCommit = "uncheckable"
)
type SinkEvidence struct {
 Verification VerificationState
 Outcome SinkOutcome
 DestinationID,OperationID,PrincipalID,TenantID,ResourceID string
 RequestDigest DigestRef
 ResultRevision int64
 StateDigest DigestRef
}
type ObserverEvidence struct {
 Verification VerificationState
 DestinationID,OperationID,PrincipalID,TenantID,ResourceID string
 Revision int64
 StateDigest DigestRef
 Independence Independence
}
type ProducerEvidence struct {
 Verification VerificationState
 DestinationID,OperationID,PrincipalID,TenantID,ResourceID string
 RequestDigest DigestRef
 Assessment CapsuleAssessment
 ClaimsSuccess bool
}
type DigitalEffectInput struct {
 Producer ProducerEvidence
 Sink SinkEvidence
 Observer ObserverEvidence
 AuthorityAtCommit AuthorityAtCommit
}
type DigitalEffectResult string
const (
 QualifiedDigitalEffect DigitalEffectResult = "qualified_digital_effect"
 RefusedWithoutEffect DigitalEffectResult = "refused_without_effect"
 EffectObservedReadbackMissing DigitalEffectResult = "effect_observed_readback_missing"
 EffectObservedNotIndependent DigitalEffectResult = "effect_observed_not_independent"
 EffectObservedAuthorityInvalid DigitalEffectResult = "effect_observed_authority_invalid"
 EffectObservedAuthorityUnknown DigitalEffectResult = "effect_observed_authority_uncheckable"
 IndeterminateEffect DigitalEffectResult = "indeterminate"
 ContradictedEffect DigitalEffectResult = "contradicted"
)
type DigitalEffectReport struct {
 Result DigitalEffectResult
 RequestBinding,StateBinding DigestRelation
 Findings []string
}

// QualifyDigitalEffect checks the normalized cross-profile join. A qualified
// result means digital destination state only; it does not assert physical effect.
func QualifyDigitalEffect(in DigitalEffectInput) DigitalEffectReport {
 r:=DigitalEffectReport{Result:IndeterminateEffect,RequestBinding:DigestIndeterminate,StateBinding:DigestIndeterminate}
 if in.Producer.Verification==Invalid || in.Sink.Verification==Invalid || in.Observer.Verification==Invalid {
  return finding(r,ContradictedEffect,"upstream_verifier_reported_invalid_evidence")
 }
 if in.Producer.Verification!=Verified || in.Sink.Verification!=Verified {
  return finding(r,IndeterminateEffect,"producer_or_sink_not_verified")
 }
 if !in.Producer.Assessment.Valid { return finding(r,IndeterminateEffect,"action_capsule_effect_binding_invalid") }
 if !sameScope(in.Producer.DestinationID,in.Producer.OperationID,in.Producer.PrincipalID,in.Producer.TenantID,in.Producer.ResourceID,
  in.Sink.DestinationID,in.Sink.OperationID,in.Sink.PrincipalID,in.Sink.TenantID,in.Sink.ResourceID) {
  return finding(r,ContradictedEffect,"producer_sink_operation_scope_mismatch")
 }
 r.RequestBinding=CompareDigestRefs(in.Producer.RequestDigest,in.Sink.RequestDigest)
 switch r.RequestBinding {
 case DigestEqual:
 case DigestDifferent: return finding(r,ContradictedEffect,"producer_sink_request_digest_mismatch")
 case DigestIncomparable: return finding(r,IndeterminateEffect,"request_digest_contexts_incomparable")
 case DigestInvalid: return finding(r,ContradictedEffect,"request_digest_invalid")
 default: return finding(r,IndeterminateEffect,"request_digest_unverified_or_incomplete")
 }
 if in.Sink.Outcome==SinkRefused {
  if in.Producer.ClaimsSuccess { return finding(r,ContradictedEffect,"producer_success_conflicts_with_sink_refusal") }
  return finding(r,RefusedWithoutEffect,"destination_refused_mutation")
 }
 if in.Sink.Outcome!=SinkCommitted { return finding(r,IndeterminateEffect,"sink_outcome_unresolved") }
 if in.Sink.ResultRevision<0 { return finding(r,ContradictedEffect,"sink_revision_invalid") }
 if in.Observer.Verification==Missing { return finding(r,EffectObservedReadbackMissing,"independent_readback_missing") }
 if in.Observer.Verification!=Verified { return finding(r,IndeterminateEffect,"observer_verification_unavailable") }
 if !sameResourceOperation(in.Sink.DestinationID,in.Sink.OperationID,in.Sink.TenantID,in.Sink.ResourceID,
  in.Observer.DestinationID,in.Observer.OperationID,in.Observer.TenantID,in.Observer.ResourceID) ||
  strings.TrimSpace(in.Observer.PrincipalID)=="" || in.Observer.PrincipalID==in.Sink.PrincipalID {
  return finding(r,ContradictedEffect,"observer_identity_or_operation_scope_mismatch")
 }
 if in.Observer.Revision!=in.Sink.ResultRevision { return finding(r,ContradictedEffect,"observer_revision_mismatch") }
 r.StateBinding=CompareDigestRefs(in.Sink.StateDigest,in.Observer.StateDigest)
 switch r.StateBinding {
 case DigestEqual:
 case DigestDifferent: return finding(r,ContradictedEffect,"observer_state_digest_mismatch")
 case DigestIncomparable: return finding(r,IndeterminateEffect,"state_digest_contexts_incomparable")
 case DigestInvalid: return finding(r,ContradictedEffect,"state_digest_invalid")
 default: return finding(r,EffectObservedReadbackMissing,"exact_state_digest_readback_unverified")
 }
 if in.Observer.Independence==SharedControlRoot { return finding(r,EffectObservedNotIndependent,"observer_shares_required_control_root") }
 if in.Observer.Independence!=IndependenceVerified { return finding(r,EffectObservedNotIndependent,"observer_independence_uncheckable") }
 switch in.AuthorityAtCommit {
 case AuthorityValid: return finding(r,QualifiedDigitalEffect,"digital_commit_and_independent_readback_qualified")
 case AuthorityInvalid: return finding(r,EffectObservedAuthorityInvalid,"effect_occurred_but_authority_invalid_at_commit")
 default: return finding(r,EffectObservedAuthorityUnknown,"effect_observed_but_authority_at_commit_uncheckable")
 }
}
func finding(r DigitalEffectReport,result DigitalEffectResult,reason string) DigitalEffectReport {
 r.Result=result; r.Findings=append(r.Findings,reason); return r
}
func sameScope(ad,ao,ap,at,ar,bd,bo,bp,bt,br string) bool {
 return ad!="" && ao!="" && ap!="" && at!="" && ar!="" && ad==bd && ao==bo && ap==bp && at==bt && ar==br
}
func sameResourceOperation(ad,ao,at,ar,bd,bo,bt,br string) bool {
 return ad!="" && ao!="" && at!="" && ar!="" && ad==bd && ao==bo && at==bt && ar==br
}
