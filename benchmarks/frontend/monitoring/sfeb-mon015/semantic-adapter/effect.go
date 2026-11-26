// Package assurance consumes normalized verification results from maintained
// AAC, OAR, COSE and SCITT implementations. It does not verify cryptography.
package assurance

import (
	"strings"
)

type Verification string

const (
	Verified Verification = "verified"
	Missing  Verification = "missing"
	Invalid  Verification = "invalid"
)

type DigestEvidence struct {
	Present   bool
	Valid     bool // upstream verifier validated encoding and digest binding
	Algorithm string
	Value     string
	Context   DigestContext
}

type DigestContext struct {
	Profile, Version, PayloadType, Canonicalization, Encoding, Purpose, Subject string
}

type DigestRelation string

const (
	DigestEqual         DigestRelation = "equal"
	DigestDifferent     DigestRelation = "different"
	DigestIncomparable  DigestRelation = "incomparable"
	DigestIndeterminate DigestRelation = "indeterminate"
	DigestInvalid       DigestRelation = "invalid"
)

func CompareDigest(a, b DigestEvidence) DigestRelation {
	if (a.Present && !a.Valid) || (b.Present && !b.Valid) {
		return DigestInvalid
	}
	if !a.Present || !b.Present {
		return DigestIndeterminate
	}
	if !validSHA256(a.Value) || !validSHA256(b.Value) {
		return DigestInvalid
	}
	if a.Algorithm != "sha-256" || b.Algorithm != "sha-256" {
		return DigestIncomparable
	}
	if !validDigestContext(a.Context) || !validDigestContext(b.Context) {
		return DigestIndeterminate
	}
	if a.Context != b.Context {
		return DigestIncomparable
	}
	if a.Value == b.Value {
		return DigestEqual
	}
	return DigestDifferent
}

func validDigestContext(c DigestContext) bool {
	return strings.TrimSpace(c.Profile) != "" && strings.TrimSpace(c.Version) != "" &&
		strings.TrimSpace(c.PayloadType) != "" && strings.TrimSpace(c.Canonicalization) != "" &&
		strings.TrimSpace(c.Encoding) != "" && strings.TrimSpace(c.Purpose) != "" &&
		strings.TrimSpace(c.Subject) != ""
}

func validSHA256(s string) bool {
	if len(s) != 64 || strings.ToLower(s) != s {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			return false
		}
	}
	return true
}

type EffectStatus string

const (
	Planned    EffectStatus = "planned"
	Dispatched EffectStatus = "dispatched"
	Confirmed  EffectStatus = "confirmed"
	Failed     EffectStatus = "failed"
	Reverted   EffectStatus = "reverted"
)

type EffectMode string

const (
	NotApplicable         EffectMode = "not_applicable"
	DispatchedUnconfirmed EffectMode = "dispatched_unconfirmed"
	EffectConfirmed       EffectMode = "confirmed"
)

type AttestationGrade string

const (
	NoExecutionClaim AttestationGrade = "no_execution_claim"
	RuntimeClaimed   AttestationGrade = "runtime_claimed_or_host_observed"
	GateExecuted     AttestationGrade = "gate_executed"
)

type EffectRecord struct {
	Status                     EffectStatus
	RequestDigest              DigestEvidence
	ResponseDigest             DigestEvidence
	EffectAttestation          string
	ReversalCorrelationPresent bool // external_ref / decision_id
	ClaimedMode                EffectMode
}

type EffectAssessment struct {
	Valid              bool
	DerivedMode        EffectMode
	ClaimedMode        EffectMode
	Grade              AttestationGrade
	UnknownAttestation bool
	Overclaim          bool
	Findings           []string
}

// AssessEffect derives effect_mode from the five AAC-06 Effect Record states.
// Request-digest SHOULD/MAY semantics are preserved: planned may bind a request;
// dispatched recommends one but absence is not upgraded to a hard failure.
func AssessEffect(e EffectRecord) EffectAssessment {
	r := EffectAssessment{Valid: true, ClaimedMode: e.ClaimedMode, Grade: NoExecutionClaim}
	if e.RequestDigest.Present && (!e.RequestDigest.Valid || !validSHA256(e.RequestDigest.Value)) {
		r.Valid = false
		r.Findings = append(r.Findings, "present_request_digest_invalid")
	}
	if e.ResponseDigest.Present && (!e.ResponseDigest.Valid || !validSHA256(e.ResponseDigest.Value)) {
		r.Valid = false
		r.Findings = append(r.Findings, "present_response_digest_invalid")
	}
	switch e.Status {
	case Planned:
		r.DerivedMode = NotApplicable
		if e.ResponseDigest.Present {
			r.Valid = false
			r.Findings = append(r.Findings, "planned_response_digest_must_be_absent")
		}
		if e.EffectAttestation != "" {
			r.Valid = false
			r.Findings = append(r.Findings, "planned_effect_attestation_must_be_absent")
		}
	case Dispatched:
		r.DerivedMode = DispatchedUnconfirmed
		if !e.RequestDigest.Present {
			r.Findings = append(r.Findings, "request_digest_should_be_present_after_dispatch")
		}
		if e.ResponseDigest.Present {
			r.Valid = false
			r.Findings = append(r.Findings, "dispatched_response_digest_must_be_absent")
		}
	case Confirmed:
		r.DerivedMode = EffectConfirmed
		if !e.ResponseDigest.Present || !e.ResponseDigest.Valid || !validSHA256(e.ResponseDigest.Value) {
			r.Valid = false
			r.DerivedMode = DispatchedUnconfirmed
			r.Findings = append(r.Findings, "confirmed_requires_well_formed_actual_response_digest")
		}
	case Failed, Reverted:
		r.DerivedMode = DispatchedUnconfirmed
		if e.ResponseDigest.Present && (!e.ResponseDigest.Valid || !validSHA256(e.ResponseDigest.Value)) {
			r.Valid = false
		}
		if e.Status == Reverted && !e.ReversalCorrelationPresent {
			r.Findings = append(r.Findings, "reverted_effect_should_reference_original_commit")
		}
	default:
		r.Valid = false
		r.Findings = append(r.Findings, "unknown_effect_status")
		return r
	}

	if r.DerivedMode == NotApplicable {
		r.Grade = NoExecutionClaim
	} else {
		if e.EffectAttestation == "" {
			r.Valid = false
			r.Findings = append(r.Findings, "effect_attestation_required_after_dispatch")
		} else {
			switch e.EffectAttestation {
			case "gate_executed":
				r.Grade = GateExecuted
			case "runtime_claimed", "host_served_observed":
				r.Grade = RuntimeClaimed
			default:
				r.Grade = RuntimeClaimed
				r.UnknownAttestation = true
				r.Findings = append(r.Findings, "unknown_effect_attestation_informational_never_grades_up")
			}
		}
	}

	if e.ClaimedMode != "" {
		claimRank, claimOK := effectRank(e.ClaimedMode)
		derivedRank, derivedOK := effectRank(r.DerivedMode)
		if !claimOK || !derivedOK {
			r.Valid = false
			r.Findings = append(r.Findings, "unknown_effect_mode_claim")
		} else if claimRank > derivedRank {
			r.Valid = false
			r.Overclaim = true
			r.Findings = append(r.Findings, "effect_mode_overclaim")
		}
	}
	return r
}

func effectRank(m EffectMode) (int, bool) {
	switch m {
	case NotApplicable:
		return 0, true
	case DispatchedUnconfirmed:
		return 1, true
	case EffectConfirmed:
		return 2, true
	default:
		return 0, false
	}
}

type Orthogonality string

const (
	OrthogonalPass          Orthogonality = "pass"
	OrthogonalFail          Orthogonality = "fail"
	OrthogonalInformational Orthogonality = "informational"
)

// CheckVerdictEffect implements the AAC-06 explicit pairings plus two MON
// defensive checks: executed cannot assert no dispatch; timeout cannot claim
// a confirmed effect. Unknown registry verdicts remain informational.
func CheckVerdictEffect(verdict string, mode EffectMode) (Orthogonality, string) {
	known := map[string]bool{"executed": true, "blocked": true, "hitl_dispatched": true, "denied": true, "timeout": true, "errored": true, "engine_failure": true, "deferred": true, "needs_decision": true, "expired": true, "escalated": true, "resolved": true, "epoch_boundary": true}
	never := map[string]bool{"blocked": true, "hitl_dispatched": true, "denied": true, "engine_failure": true, "deferred": true, "needs_decision": true, "expired": true, "escalated": true, "resolved": true, "epoch_boundary": true}
	if !known[verdict] {
		return OrthogonalInformational, "unknown_verdict_class_informational"
	}
	if _, ok := effectRank(mode); !ok {
		return OrthogonalFail, "known_verdict_requires_known_effect_mode"
	}
	if never[verdict] && mode != NotApplicable {
		return OrthogonalFail, "never_dispatch_verdict_requires_not_applicable"
	}
	if verdict == "errored" && mode != DispatchedUnconfirmed {
		return OrthogonalFail, "errored_requires_dispatched_unconfirmed"
	}
	if verdict == "timeout" && mode == EffectConfirmed {
		return OrthogonalFail, "timeout_does_not_establish_confirmed_effect"
	}
	if verdict == "executed" && mode == NotApplicable {
		return OrthogonalFail, "executed_cannot_assert_no_dispatch"
	}
	return OrthogonalPass, ""
}

type AttestationLevel string

const (
	AttestationSelf     AttestationLevel = "self_attested"
	AttestationAnchored AttestationLevel = "anchored"
)

type LedgerLevel string

const (
	LedgerStandalone LedgerLevel = "standalone"
	LedgerChained    LedgerLevel = "chained"
	LedgerAnchored   LedgerLevel = "anchored"
)

type CrossPartyLevel string

const (
	CrossPartyUnilateral   CrossPartyLevel = "unilateral_fallback"
	CrossPartyAcknowledged CrossPartyLevel = "acknowledged_receipt"
	CrossPartyBilateral    CrossPartyLevel = "full_bilateral"
)

type CrossPartyEvidence struct {
	BlockPresent              bool
	Malformed                 bool
	InitiatorRefValid         bool
	CounterpartyRefVerified   bool
	CorrelatorVerified        bool
	SubstantiveCoSignedResult bool
}

type AssuranceEvidence struct {
	CapsuleIDRegistrationReceiptValid bool
	OwnChainBlockIntact               bool
	ChainRootInclusionProofValid      bool
	CrossParty                        CrossPartyEvidence
}

type AssuranceClaims struct {
	Attestation AttestationLevel
	Effect      EffectMode
	Ledger      LedgerLevel
	CrossParty  CrossPartyLevel // empty when the optional block is absent
}

type AssuranceAssessment struct {
	Valid                             bool
	Derived                           AssuranceClaims
	Overclaims, Underclaims, Findings []string
}

func ReconcileAssurance(claims AssuranceClaims, effect EffectAssessment, evidence AssuranceEvidence) AssuranceAssessment {
	r := AssuranceAssessment{Valid: effect.Valid}
	r.Derived.Effect = effect.DerivedMode
	if evidence.CapsuleIDRegistrationReceiptValid {
		r.Derived.Attestation = AttestationAnchored
	} else {
		r.Derived.Attestation = AttestationSelf
	}
	switch {
	case evidence.OwnChainBlockIntact && evidence.ChainRootInclusionProofValid:
		r.Derived.Ledger = LedgerAnchored
	case evidence.OwnChainBlockIntact:
		r.Derived.Ledger = LedgerChained
	default:
		r.Derived.Ledger = LedgerStandalone
	}
	if evidence.CrossParty.BlockPresent {
		if evidence.CrossParty.Malformed || !evidence.CrossParty.InitiatorRefValid {
			r.Valid = false
			r.Findings = append(r.Findings, "cross_party_block_malformed")
		} else if !evidence.CrossParty.CounterpartyRefVerified || !evidence.CrossParty.CorrelatorVerified {
			r.Derived.CrossParty = CrossPartyUnilateral
		} else if evidence.CrossParty.SubstantiveCoSignedResult {
			r.Derived.CrossParty = CrossPartyBilateral
		} else {
			r.Derived.CrossParty = CrossPartyAcknowledged
		}
	} else if claims.CrossParty != "" {
		r.Overclaims = append(r.Overclaims, "cross_party_without_block")
	}
	if evidence.ChainRootInclusionProofValid && !evidence.OwnChainBlockIntact {
		r.Valid = false
		r.Findings = append(r.Findings, "chain_root_proof_without_intact_chain_block")
	}
	compare("attestation", rankAttestation(claims.Attestation), rankAttestation(r.Derived.Attestation), &r)
	compare("effect", effectRankInt(claims.Effect), effectRankInt(r.Derived.Effect), &r)
	compare("ledger", rankLedger(claims.Ledger), rankLedger(r.Derived.Ledger), &r)
	if evidence.CrossParty.BlockPresent {
		compare("cross_party", rankCross(claims.CrossParty), rankCross(r.Derived.CrossParty), &r)
	}
	if len(r.Overclaims) > 0 {
		r.Valid = false
		r.Findings = append(r.Findings, "assurance_overclaim")
	}
	return r
}
func compare(name string, c, d int, r *AssuranceAssessment) {
	if c < 0 || d < 0 {
		r.Valid = false
		r.Findings = append(r.Findings, "unknown_assurance_claim:"+name)
		return
	}
	if c > d {
		r.Overclaims = append(r.Overclaims, name)
	} else if c < d {
		r.Underclaims = append(r.Underclaims, name)
	}
}
func rankAttestation(x AttestationLevel) int {
	switch x {
	case AttestationSelf:
		return 0
	case AttestationAnchored:
		return 1
	default:
		return -1
	}
}
func effectRankInt(x EffectMode) int {
	n, ok := effectRank(x)
	if !ok {
		return -1
	}
	return n
}
func rankLedger(x LedgerLevel) int {
	switch x {
	case LedgerStandalone:
		return 0
	case LedgerChained:
		return 1
	case LedgerAnchored:
		return 2
	default:
		return -1
	}
}
func rankCross(x CrossPartyLevel) int {
	switch x {
	case CrossPartyUnilateral:
		return 0
	case CrossPartyAcknowledged:
		return 1
	case CrossPartyBilateral:
		return 2
	default:
		return -1
	}
}

func ScopeEqual(a, b OperationScope) bool {
	return a == b && a.Destination != "" && a.Operation != "" && a.Principal != "" && a.Tenant != "" && a.Resource != ""
}

// SameDestinationOperationScope binds the observer to the same destination,
// operation, tenant and resource without requiring the observer and writer to
// have the same principal. Principal separation is checked independently.
func SameDestinationOperationScope(a, b OperationScope) bool {
	return a.Destination != "" && a.Operation != "" && a.Tenant != "" && a.Resource != "" &&
		a.Destination == b.Destination && a.Operation == b.Operation && a.Tenant == b.Tenant && a.Resource == b.Resource
}

type OperationScope struct{ Destination, Operation, Principal, Tenant, Resource string }

type SinkOutcome string

const (
	SinkOutcomeUnresolved SinkOutcome = "unresolved"
	SinkOutcomeCommitted  SinkOutcome = "committed"
	SinkOutcomeRefused    SinkOutcome = "refused"
)

type DigitalEffectInput struct {
	ProducerVerification, SinkVerification, ObserverVerification Verification
	ProducerScope, SinkScope, ObserverScope                      OperationScope
	ProducerRequest, SinkRequest                                 DigestEvidence
	Effect                                                       EffectAssessment
	SinkOutcome                                                  SinkOutcome
	SinkRevision, ObserverRevision                               int64
	SinkState, ObserverState                                     DigestEvidence
	ObserverPrincipal                                            string
	ObserverIndependent                                          bool
	AuthorityAtCommit                                            string // valid / invalid / uncheckable
}
type DigitalEffectResult string

const (
	QualifiedDigital       DigitalEffectResult = "qualified_digital_effect"
	Indeterminate          DigitalEffectResult = "indeterminate"
	Contradicted           DigitalEffectResult = "contradicted"
	ReadbackMissing        DigitalEffectResult = "effect_observed_readback_missing"
	NotIndependent         DigitalEffectResult = "effect_observed_not_independent"
	AuthorityInvalidResult DigitalEffectResult = "effect_observed_authority_invalid"
	AuthorityUnknownResult DigitalEffectResult = "effect_observed_authority_uncheckable"
	Refused                DigitalEffectResult = "refused_without_effect"
)

type DigitalEffectReport struct {
	Result                          DigitalEffectResult
	RequestRelation, DigestRelation DigestRelation
	Finding                         string
}

func QualifyDigitalEffect(in DigitalEffectInput) DigitalEffectReport {
	r := DigitalEffectReport{Result: Indeterminate, RequestRelation: DigestIndeterminate, DigestRelation: DigestIndeterminate}
	if in.ProducerVerification == Invalid || in.SinkVerification == Invalid || in.ObserverVerification == Invalid {
		r.Result = Contradicted
		r.Finding = "upstream_invalid"
		return r
	}
	if in.ProducerVerification != Verified || in.SinkVerification != Verified || !in.Effect.Valid {
		r.Finding = "producer_or_sink_unverified"
		return r
	}
	if !ScopeEqual(in.ProducerScope, in.SinkScope) {
		r.Result = Contradicted
		r.Finding = "producer_sink_scope_mismatch"
		return r
	}
	r.RequestRelation = CompareDigest(in.ProducerRequest, in.SinkRequest)
	switch r.RequestRelation {
	case DigestEqual:
	case DigestDifferent:
		r.Result = Contradicted
		r.Finding = "request_digest_mismatch"
		return r
	case DigestInvalid:
		r.Result = Contradicted
		r.Finding = "request_digest_invalid"
		return r
	default:
		r.Finding = "request_digest_incomparable_or_missing"
		return r
	}
	switch in.SinkOutcome {
	case SinkOutcomeRefused:
		if in.Effect.DerivedMode == EffectConfirmed {
			r.Result = Contradicted
			r.Finding = "producer_confirmed_effect_conflicts_with_sink_refusal"
		} else {
			r.Result = Refused
			r.Finding = "sink_refused_mutation"
		}
		return r
	case SinkOutcomeCommitted:
	case SinkOutcomeUnresolved, "":
		r.Result = Indeterminate
		r.Finding = "sink_outcome_unresolved"
		return r
	default:
		r.Result = Contradicted
		r.Finding = "unknown_sink_outcome"
		return r
	}
	if in.Effect.DerivedMode == NotApplicable {
		r.Result = Contradicted
		r.Finding = "sink_commit_conflicts_with_no_effect_claim"
		return r
	}
	if in.ObserverVerification == Missing {
		r.Result = ReadbackMissing
		r.Finding = "observer_missing"
		return r
	}
	if in.ObserverVerification != Verified || !SameDestinationOperationScope(in.SinkScope, in.ObserverScope) {
		r.Result = Contradicted
		r.Finding = "observer_scope_mismatch"
		return r
	}
	if in.ObserverPrincipal == "" || in.ObserverScope.Principal == "" ||
		in.ObserverPrincipal != in.ObserverScope.Principal || in.ObserverPrincipal == in.SinkScope.Principal {
		r.Result = Contradicted
		r.Finding = "observer_principal_not_distinct_or_unbound"
		return r
	}
	if in.ObserverRevision != in.SinkRevision {
		r.Result = Contradicted
		r.Finding = "observer_revision_mismatch"
		return r
	}
	r.DigestRelation = CompareDigest(in.SinkState, in.ObserverState)
	switch r.DigestRelation {
	case DigestEqual:
	case DigestDifferent, DigestInvalid:
		r.Result = Contradicted
		r.Finding = "state_digest_mismatch_or_invalid"
		return r
	default:
		r.Finding = "state_digest_incomparable_or_missing"
		return r
	}
	if !in.ObserverIndependent {
		r.Result = NotIndependent
		r.Finding = "observer_control_roots_not_independent"
		return r
	}
	switch in.AuthorityAtCommit {
	case "valid":
		r.Result = QualifiedDigital
		r.Finding = "digital_state_qualified"
	case "invalid":
		r.Result = AuthorityInvalidResult
		r.Finding = "authority_invalid_at_commit"
	default:
		r.Result = AuthorityUnknownResult
		r.Finding = "authority_at_commit_uncheckable"
	}
	return r
}
