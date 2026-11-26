// Package crossprofile reconciles profile-qualified evidence without inventing
// a replacement receipt format or a cryptographic verifier. All Verified states
// are inputs from maintained upstream verifiers/adapters.
package crossprofile

import (
	"encoding/hex"
	"strings"
)

type VerificationState string

const (
	Verified VerificationState = "verified"
	Missing  VerificationState = "missing"
	Invalid  VerificationState = "invalid"
)

// DigestContext declares exactly what was hashed and how. A hexadecimal value
// alone is never enough to compare digests across profile boundaries.
type DigestContext struct {
	Profile          string
	Version          string
	PayloadType      string
	Canonicalization string
	Encoding         string
	Purpose          string
	Subject          string
}

type DigestRef struct {
	Algorithm    string
	Value        string
	Context      DigestContext
	Verification VerificationState
}

type DigestRelation string

const (
	DigestEqual         DigestRelation = "equal"
	DigestDifferent     DigestRelation = "different"
	DigestIncomparable  DigestRelation = "incomparable"
	DigestIndeterminate DigestRelation = "indeterminate"
	DigestInvalid       DigestRelation = "invalid"
)

// CompareDigestRefs returns Equal or Different only when the algorithm and
// every declared digest-context field agree and both references were verified.
// Equal-looking hex in different profiles or canonicalizations is Incomparable.
func CompareDigestRefs(a, b DigestRef) DigestRelation {
	if a.Verification == Invalid || b.Verification == Invalid {
		return DigestInvalid
	}
	if a.Verification != Verified || b.Verification != Verified {
		return DigestIndeterminate
	}
	if !validSHA256(a.Value) || !validSHA256(b.Value) {
		return DigestInvalid
	}
	if !validContext(a.Context) || !validContext(b.Context) {
		return DigestIndeterminate
	}
	if !strings.EqualFold(a.Algorithm, b.Algorithm) ||
		!strings.EqualFold(a.Algorithm, "sha-256") {
		return DigestIncomparable
	}
	if a.Context != b.Context {
		return DigestIncomparable
	}
	if strings.EqualFold(a.Value, b.Value) {
		return DigestEqual
	}
	return DigestDifferent
}

func validContext(c DigestContext) bool {
	return strings.TrimSpace(c.Profile) != "" &&
		strings.TrimSpace(c.Version) != "" &&
		strings.TrimSpace(c.PayloadType) != "" &&
		strings.TrimSpace(c.Canonicalization) != "" &&
		strings.TrimSpace(c.Encoding) != "" &&
		strings.TrimSpace(c.Purpose) != "" &&
		strings.TrimSpace(c.Subject) != ""
}

func validSHA256(value string) bool {
	if len(value) != 64 || strings.ToLower(value) != value {
		return false
	}
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == 32
}

type EffectMode string

const (
	NotApplicable         EffectMode = "not_applicable"
	DispatchedUnconfirmed EffectMode = "dispatched_unconfirmed"
	Confirmed             EffectMode = "confirmed"
)

type EffectAttestation string

const (
	GateExecuted       EffectAttestation = "gate_executed"
	RuntimeClaimed     EffectAttestation = "runtime_claimed"
	HostServedObserved EffectAttestation = "host_served_observed"
)

type ProducerGrade string

const (
	NoExecutionClaim ProducerGrade = "no_execution_claim"
	RuntimeAssertion ProducerGrade = "runtime_assertion"
	GateObserved     ProducerGrade = "gate_observed_effect_boundary"
)

type CapsuleAssessment struct {
	Valid                 bool
	EffectMode            EffectMode
	Attestation           EffectAttestation
	Grade                 ProducerGrade
	UnknownAttestation    bool
	ResponseDigestPresent bool
	Findings              []string
}

// AssessCapsuleEffect applies the Agent Action Capsule -06 effect-mode and
// attestation relationship. It does not verify the Capsule signature, recompute
// its response digest, verify SCITT registration, or prove sink state.
func AssessCapsuleEffect(mode EffectMode, attestation string, responseDigestPresent bool) CapsuleAssessment {
	out := CapsuleAssessment{
		EffectMode:            mode,
		Attestation:           EffectAttestation(attestation),
		Grade:                 NoExecutionClaim,
		ResponseDigestPresent: responseDigestPresent,
	}
	switch mode {
	case NotApplicable:
		if attestation != "" {
			out.Findings = append(out.Findings, "effect_attestation_must_be_absent_when_not_applicable")
			return out
		}
		out.Valid = true
		return out
	case DispatchedUnconfirmed:
		if attestation == "" {
			out.Findings = append(out.Findings, "effect_attestation_required_after_dispatch")
			return out
		}
	case Confirmed:
		if attestation == "" {
			out.Findings = append(out.Findings, "effect_attestation_required_for_confirmed_effect")
			return out
		}
		if !responseDigestPresent {
			out.Findings = append(out.Findings, "confirmed_effect_requires_response_digest")
			return out
		}
	default:
		out.Findings = append(out.Findings, "unknown_effect_mode")
		return out
	}

	switch EffectAttestation(attestation) {
	case GateExecuted:
		out.Grade = GateObserved
	case RuntimeClaimed, HostServedObserved:
		out.Grade = RuntimeAssertion
	default:
		// Unknown registry values are informational, never rejected and never
		// graded above runtime_claimed.
		out.UnknownAttestation = true
		out.Grade = RuntimeAssertion
		out.Findings = append(out.Findings, "unknown_effect_attestation_graded_no_higher_than_runtime_claimed")
	}
	out.Valid = true
	return out
}

type SinkOutcome string

const (
	SinkCommitted SinkOutcome = "committed"
	SinkRefused   SinkOutcome = "refused"
	SinkUnknown   SinkOutcome = "unresolved"
)

type Independence string

const (
	IndependenceVerified Independence = "independent"
	SharedControlRoot    Independence = "shared_control_root"
	IndependenceUnknown  Independence = "uncheckable"
)

type AuthorityAtCommit string

const (
	AuthorityValid       AuthorityAtCommit = "valid"
	AuthorityInvalid     AuthorityAtCommit = "invalid"
	AuthorityUncheckable AuthorityAtCommit = "uncheckable"
)

type SinkEvidence struct {
	Verification   VerificationState
	Outcome        SinkOutcome
	DestinationID  string
	OperationID    string
	PrincipalID    string
	TenantID       string
	ResourceID     string
	RequestDigest  DigestRef
	ResultRevision int64
	StateDigest    DigestRef
}

type ObserverEvidence struct {
	Verification  VerificationState
	DestinationID string
	OperationID   string
	PrincipalID   string
	TenantID      string
	ResourceID    string
	Revision      int64
	StateDigest   DigestRef
	Independence  Independence
}

type ProducerEvidence struct {
	Verification  VerificationState
	DestinationID string
	OperationID   string
	PrincipalID   string
	TenantID      string
	ResourceID    string
	RequestDigest DigestRef
	Assessment    CapsuleAssessment
	ClaimsSuccess bool
}

type DigitalEffectResult string

const (
	QualifiedDigitalEffect         DigitalEffectResult = "qualified_digital_effect"
	RefusedWithoutEffect           DigitalEffectResult = "refused_without_effect"
	EffectObservedReadbackMissing  DigitalEffectResult = "effect_observed_readback_missing"
	EffectObservedNotIndependent   DigitalEffectResult = "effect_observed_not_independent"
	EffectObservedAuthorityInvalid DigitalEffectResult = "effect_observed_authority_invalid"
	EffectObservedAuthorityUnknown DigitalEffectResult = "effect_observed_authority_uncheckable"
	IndeterminateEffect            DigitalEffectResult = "indeterminate"
	ContradictedEffect              DigitalEffectResult = "contradicted"
)

type DigitalEffectInput struct {
	Producer          ProducerEvidence
	Sink              SinkEvidence
	Observer           ObserverEvidence
	AuthorityAtCommit AuthorityAtCommit
}

type DigitalEffectReport struct {
	Result         DigitalEffectResult
	RequestBinding DigestRelation
	StateBinding   DigestRelation
	Findings       []string
}

// QualifyDigitalEffect is a normalized semantic join, not a verifier. It
// requires verified producer, destination and observer evidence; exact
// operation/scope identity; comparable request and state digest contexts;
// exact revision readback; independent observer control roots; and valid
// authority at the recorded commit time. A pass is limited to a digital
// destination effect, never a physical effect.
func QualifyDigitalEffect(in DigitalEffectInput) DigitalEffectReport {
	out := DigitalEffectReport{
		Result:         IndeterminateEffect,
		RequestBinding: DigestIndeterminate,
		StateBinding:   DigestIndeterminate,
	}
	if in.Producer.Verification == Invalid || in.Sink.Verification == Invalid ||
		in.Observer.Verification == Invalid {
		return withFinding(out, ContradictedEffect, "one_or_more_profile_verifiers_reported_invalid_evidence")
	}
	if in.Producer.Verification != Verified || in.Sink.Verification != Verified {
		return withFinding(out, IndeterminateEffect, "producer_or_sink_evidence_not_verified")
	}
	if !in.Producer.Assessment.Valid {
		return withFinding(out, IndeterminateEffect, "agent_action_capsule_effect_binding_not_valid")
	}

	// A producer-confirmed success and an authoritative sink refusal cannot
	// both describe the same bound operation.
	if in.Sink.Outcome == SinkRefused {
		if in.Producer.ClaimsSuccess {
			return withFinding(out, ContradictedEffect, "producer_success_conflicts_with_sink_refusal")
		}
		return withFinding(out, RefusedWithoutEffect, "destination_refused_mutation")
	}
	if in.Sink.Outcome != SinkCommitted {
		return withFinding(out, IndeterminateEffect, "destination_outcome_unresolved")
	}

	if !sameOperationScope(in.Producer.DestinationID, in.Producer.OperationID,
		in.Producer.PrincipalID, in.Producer.TenantID, in.Producer.ResourceID,
		in.Sink.DestinationID, in.Sink.OperationID, in.Sink.PrincipalID,
		in.Sink.TenantID, in.Sink.ResourceID) {
		return withFinding(out, ContradictedEffect, "producer_and_sink_operation_scope_mismatch")
	}

	out.RequestBinding = CompareDigestRefs(in.Producer.RequestDigest, in.Sink.RequestDigest)
	switch out.RequestBinding {
	case DigestEqual:
	case DigestDifferent:
		return withFinding(out, ContradictedEffect, "producer_and_sink_request_digest_mismatch")
	case DigestIncomparable:
		return withFinding(out, IndeterminateEffect, "request_digest_contexts_not_comparable")
	case DigestInvalid:
		return withFinding(out, ContradictedEffect, "request_digest_evidence_invalid")
	default:
		return withFinding(out, IndeterminateEffect, "request_digest_evidence_unverified_or_incomplete")
	}

	if in.Sink.ResultRevision < 0 {
		return withFinding(out, ContradictedEffect, "sink_result_revision_invalid")
	}
	if in.Observer.Verification == Missing {
		return withFinding(out, EffectObservedReadbackMissing, "independent_readback_not_verified")
	}
	if in.Observer.Verification != Verified {
		return withFinding(out, IndeterminateEffect, "observer_verification_state_unknown")
	}
	if !sameResourceOperation(in.Sink.DestinationID, in.Sink.OperationID,
		in.Sink.TenantID, in.Sink.ResourceID,
		in.Observer.DestinationID, in.Observer.OperationID,
		in.Observer.TenantID, in.Observer.ResourceID) ||
		strings.TrimSpace(in.Observer.PrincipalID) == "" {
		return withFinding(out, ContradictedEffect, "sink_and_observer_operation_scope_mismatch")
	}
	if in.Observer.Revision != in.Sink.ResultRevision {
		return withFinding(out, ContradictedEffect, "observer_revision_differs_from_sink_result_revision")
	}
	out.StateBinding = CompareDigestRefs(in.Sink.StateDigest, in.Observer.StateDigest)
	switch out.StateBinding {
	case DigestEqual:
	case DigestDifferent:
		return withFinding(out, ContradictedEffect, "observer_state_digest_differs_from_sink_result")
	case DigestIncomparable:
		return withFinding(out, IndeterminateEffect, "state_digest_contexts_not_comparable")
	case DigestInvalid:
		return withFinding(out, ContradictedEffect, "state_digest_evidence_invalid")
	default:
		return withFinding(out, EffectObservedReadbackMissing, "exact_state_digest_readback_unverified")
	}

	if in.Observer.Independence == SharedControlRoot {
		return withFinding(out, EffectObservedNotIndependent, "observer_shares_a_required_control_root_with_sink")
	}
	if in.Observer.Independence != IndependenceVerified {
		return withFinding(out, EffectObservedNotIndependent, "observer_independence_uncheckable")
	}
	switch in.AuthorityAtCommit {
	case AuthorityValid:
		return withFinding(out, QualifiedDigitalEffect, "exact_digital_commit_and_independent_readback_qualified")
	case AuthorityInvalid:
		return withFinding(out, EffectObservedAuthorityInvalid, "effect_occurred_but_authority_was_invalid_at_commit")
	default:
		return withFinding(out, EffectObservedAuthorityUnknown, "effect_observed_but_authority_at_commit_is_uncheckable")
	}
}

func withFinding(out DigitalEffectReport, result DigitalEffectResult, finding string) DigitalEffectReport {
	out.Result = result
	out.Findings = append(out.Findings, finding)
	return out
}

func sameOperationScope(ad, ao, ap, at, ar, bd, bo, bp, bt, br string) bool {
	return ad != "" && ao != "" && ap != "" && at != "" && ar != "" &&
		ad == bd && ao == bo && ap == bp && at == bt && ar == br
}

func sameResourceOperation(ad, ao, at, ar, bd, bo, bt, br string) bool {
	return ad != "" && ao != "" && at != "" && ar != "" &&
		ad == bd && ao == bo && at == bt && ar == br
}
