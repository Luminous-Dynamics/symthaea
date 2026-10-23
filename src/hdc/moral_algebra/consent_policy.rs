// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Scope-bound consent policy checks, deliberately separate from moral verdicts.
//!
//! This evaluates one consent-specific precondition. It does not grant overall
//! permission to act, authenticate records, verify signatures, or replace
//! authorization infrastructure. Callers must populate verification fields only
//! after validating a trusted record, and must still run all other policy gates.
//! Scope IDs are canonical identifiers; exact matching is intentional.

use super::primitives::ConsentState;

/// Whether the caller's policy says this operation requires consent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsentRequirement {
    /// This gate imposes no consent requirement; this is not permission to act.
    NotRequired,
    /// A matching affirmative and currently valid record is required.
    Required,
    /// Policy lookup failed or the action's consent requirement is ambiguous.
    /// This must block; it must never be silently mapped to NotRequired.
    Unresolved,
}

/// Canonical identity of the operation being assessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConsentRequest<'a> {
    /// Stable identifier for the person/data subject affected.
    pub subject_id: &'a str,
    /// Stable identifier for the requested action, not free-form prose.
    pub action_id: &'a str,
    /// Stable identifier for the declared purpose.
    pub purpose_id: &'a str,
}

/// A consent record after a trusted adapter has checked its provenance.
///
/// The adapter must set the verification flags from actual validation, not
/// natural-language parser output. This crate supplies no cryptographic
/// verification; this data structure is not itself a security boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConsentEvidence<'a> {
    /// State recorded by the trusted source.
    pub state: ConsentState,
    /// Stable identifier of the affected subject.
    pub subject_id: &'a str,
    /// Stable identifier of the covered action.
    pub action_id: &'a str,
    /// Stable identifier of the consented purpose.
    pub purpose_id: &'a str,
    /// Opaque identifier for the stored record.
    pub record_id: &'a str,
    /// True only after record integrity/authenticity validation.
    pub record_verified: bool,
    /// True only after validating the consenting subject's identity.
    pub identity_verified: bool,
    /// True only after validating authority for this operation.
    pub authority_verified: bool,
    /// Inclusive start of validity, in Unix seconds.
    pub valid_from_unix_s: u64,
    /// Exclusive end of validity; None means no expiry was supplied.
    pub expires_at_unix_s: Option<u64>,
    /// Revocation status from the authoritative record source.
    pub revoked: bool,
}

/// Why a consent-required action did not satisfy this gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsentBlockReason {
    RequirementUnresolved,
    MissingEvidence,
    UnknownState,
    ImpliedState,
    ExplicitlyDenied,
    ExplicitlyAbsent,
    InvalidRequestScope,
    EmptyEvidenceScope,
    RecordUnverified,
    IdentityUnverified,
    AuthorityUnverified,
    SubjectMismatch,
    ActionMismatch,
    PurposeMismatch,
    NotYetValid,
    Expired,
    InvalidValidityWindow,
    Revoked,
}

/// Outcome of the consent-specific precondition, not an overall action permit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsentGateOutcome {
    /// Consent is not required by this gate; other policies still apply.
    NotRequired,
    /// This consent-specific precondition is satisfied; other policies still apply.
    Satisfied,
    /// Consent was not established for this exact request.
    Blocked(ConsentBlockReason),
}

/// Fail closed when required evidence is unknown, assumed, unverified, out of
/// scope, not yet valid, expired, or revoked. The clock must be trusted. A Given
/// label by itself can never satisfy this policy.
pub fn evaluate_consent_gate(
    requirement: ConsentRequirement,
    request: ConsentRequest<'_>,
    evidence: Option<&ConsentEvidence<'_>>,
    now_unix_s: u64,
) -> ConsentGateOutcome {
    match requirement {
        ConsentRequirement::NotRequired => return ConsentGateOutcome::NotRequired,
        ConsentRequirement::Required => {},
        ConsentRequirement::Unresolved => {
            return ConsentGateOutcome::Blocked(ConsentBlockReason::RequirementUnresolved);
        }
    }
    let Some(evidence) = evidence else {
        return ConsentGateOutcome::Blocked(ConsentBlockReason::MissingEvidence);
    };

    match evidence.state {
        ConsentState::Given => {}
        ConsentState::Denied => return ConsentGateOutcome::Blocked(ConsentBlockReason::ExplicitlyDenied),
        ConsentState::Absent => return ConsentGateOutcome::Blocked(ConsentBlockReason::ExplicitlyAbsent),
        ConsentState::Implied => return ConsentGateOutcome::Blocked(ConsentBlockReason::ImpliedState),
        ConsentState::Unknown => return ConsentGateOutcome::Blocked(ConsentBlockReason::UnknownState),
    }

    if request.subject_id.trim().is_empty()
        || request.action_id.trim().is_empty()
        || request.purpose_id.trim().is_empty()
    {
        return ConsentGateOutcome::Blocked(ConsentBlockReason::InvalidRequestScope);
    }
    if evidence.subject_id.trim().is_empty()
        || evidence.action_id.trim().is_empty()
        || evidence.purpose_id.trim().is_empty()
        || evidence.record_id.trim().is_empty()
    {
        return ConsentGateOutcome::Blocked(ConsentBlockReason::EmptyEvidenceScope);
    }
    if !evidence.record_verified {
        return ConsentGateOutcome::Blocked(ConsentBlockReason::RecordUnverified);
    }
    if !evidence.identity_verified {
        return ConsentGateOutcome::Blocked(ConsentBlockReason::IdentityUnverified);
    }
    if !evidence.authority_verified {
        return ConsentGateOutcome::Blocked(ConsentBlockReason::AuthorityUnverified);
    }
    if evidence.revoked {
        return ConsentGateOutcome::Blocked(ConsentBlockReason::Revoked);
    }
    if evidence.subject_id != request.subject_id {
        return ConsentGateOutcome::Blocked(ConsentBlockReason::SubjectMismatch);
    }
    if evidence.action_id != request.action_id {
        return ConsentGateOutcome::Blocked(ConsentBlockReason::ActionMismatch);
    }
    if evidence.purpose_id != request.purpose_id {
        return ConsentGateOutcome::Blocked(ConsentBlockReason::PurposeMismatch);
    }
    if evidence.expires_at_unix_s.is_some_and(|expiry| expiry <= evidence.valid_from_unix_s) {
        return ConsentGateOutcome::Blocked(ConsentBlockReason::InvalidValidityWindow);
    }
    if now_unix_s < evidence.valid_from_unix_s {
        return ConsentGateOutcome::Blocked(ConsentBlockReason::NotYetValid);
    }
    if evidence.expires_at_unix_s.is_some_and(|expiry| now_unix_s >= expiry) {
        return ConsentGateOutcome::Blocked(ConsentBlockReason::Expired);
    }

    ConsentGateOutcome::Satisfied
}

/// Evaluate the consent precondition at the point of use and invoke a protected
/// action only when this consent-specific gate is satisfied or the policy
/// explicitly says consent is not required.
///
/// Callers must fetch fresh evidence and a trusted current time immediately
/// before calling this function, and must complete all *other* authorization,
/// safety, and legal checks before invoking it. Parser output must never be
/// represented as verified evidence. This synchronous helper inserts no
/// `await` between checking consent and invoking the closure, but it cannot
/// eliminate races in an external system: resources with concurrent revocation
/// or mutation must additionally enforce revision/transaction checks themselves.
///
/// A blocked result never invokes `action`.
pub fn execute_after_consent_check<T>(
    requirement: ConsentRequirement,
    request: ConsentRequest<'_>,
    evidence: Option<&ConsentEvidence<'_>>,
    now_unix_s: u64,
    action: impl FnOnce() -> T,
) -> Result<T, ConsentBlockReason> {
    match evaluate_consent_gate(requirement, request, evidence, now_unix_s) {
        ConsentGateOutcome::Blocked(reason) => Err(reason),
        ConsentGateOutcome::Satisfied | ConsentGateOutcome::NotRequired => Ok(action()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_800_000_000;

    fn request() -> ConsentRequest<'static> {
        ConsentRequest {
            subject_id: "subject:alice",
            action_id: "action:export-health-record",
            purpose_id: "purpose:care-coordination",
        }
    }

    fn evidence(state: ConsentState) -> ConsentEvidence<'static> {
        ConsentEvidence {
            state,
            subject_id: "subject:alice",
            action_id: "action:export-health-record",
            purpose_id: "purpose:care-coordination",
            record_id: "consent:record-17",
            record_verified: true,
            identity_verified: true,
            authority_verified: true,
            valid_from_unix_s: NOW - 60,
            expires_at_unix_s: Some(NOW + 60),
            revoked: false,
        }
    }

    #[test]
    fn unknown_implied_absent_and_denied_never_satisfy_required_consent() {
        for (state, reason) in [
            (ConsentState::Unknown, ConsentBlockReason::UnknownState),
            (ConsentState::Implied, ConsentBlockReason::ImpliedState),
            (ConsentState::Absent, ConsentBlockReason::ExplicitlyAbsent),
            (ConsentState::Denied, ConsentBlockReason::ExplicitlyDenied),
        ] {
            assert_eq!(
                evaluate_consent_gate(ConsentRequirement::Required, request(), Some(&evidence(state)), NOW),
                ConsentGateOutcome::Blocked(reason)
            );
        }
    }

    #[test]
    fn missing_evidence_and_unverified_records_fail_closed() {
        assert_eq!(
            evaluate_consent_gate(ConsentRequirement::Required, request(), None, NOW),
            ConsentGateOutcome::Blocked(ConsentBlockReason::MissingEvidence)
        );
        let mut e = evidence(ConsentState::Given);
        e.record_verified = false;
        assert_eq!(
            evaluate_consent_gate(ConsentRequirement::Required, request(), Some(&e), NOW),
            ConsentGateOutcome::Blocked(ConsentBlockReason::RecordUnverified)
        );
        e.record_verified = true;
        e.identity_verified = false;
        assert_eq!(
            evaluate_consent_gate(ConsentRequirement::Required, request(), Some(&e), NOW),
            ConsentGateOutcome::Blocked(ConsentBlockReason::IdentityUnverified)
        );
        e.identity_verified = true;
        e.authority_verified = false;
        assert_eq!(
            evaluate_consent_gate(ConsentRequirement::Required, request(), Some(&e), NOW),
            ConsentGateOutcome::Blocked(ConsentBlockReason::AuthorityUnverified)
        );
    }

    #[test]
    fn matching_current_verified_record_satisfies_only_this_gate() {
        assert_eq!(
            evaluate_consent_gate(ConsentRequirement::Required, request(), Some(&evidence(ConsentState::Given)), NOW),
            ConsentGateOutcome::Satisfied
        );
    }

    #[test]
    fn subject_action_and_purpose_must_match_exactly() {
        let e = evidence(ConsentState::Given);
        let mut r = request();
        r.subject_id = "subject:bob";
        assert_eq!(
            evaluate_consent_gate(ConsentRequirement::Required, r, Some(&e), NOW),
            ConsentGateOutcome::Blocked(ConsentBlockReason::SubjectMismatch)
        );
        let mut r = request();
        r.action_id = "action:publish-record";
        assert_eq!(
            evaluate_consent_gate(ConsentRequirement::Required, r, Some(&e), NOW),
            ConsentGateOutcome::Blocked(ConsentBlockReason::ActionMismatch)
        );
        let mut r = request();
        r.purpose_id = "purpose:advertising";
        assert_eq!(
            evaluate_consent_gate(ConsentRequirement::Required, r, Some(&e), NOW),
            ConsentGateOutcome::Blocked(ConsentBlockReason::PurposeMismatch)
        );
    }

    #[test]
    fn future_expired_invalid_and_revoked_records_fail_closed() {
        let mut e = evidence(ConsentState::Given);
        e.valid_from_unix_s = NOW + 1;
        assert_eq!(
            evaluate_consent_gate(ConsentRequirement::Required, request(), Some(&e), NOW),
            ConsentGateOutcome::Blocked(ConsentBlockReason::NotYetValid)
        );
        let mut e = evidence(ConsentState::Given);
        e.expires_at_unix_s = Some(NOW);
        assert_eq!(
            evaluate_consent_gate(ConsentRequirement::Required, request(), Some(&e), NOW),
            ConsentGateOutcome::Blocked(ConsentBlockReason::Expired)
        );
        let mut e = evidence(ConsentState::Given);
        e.expires_at_unix_s = Some(e.valid_from_unix_s);
        assert_eq!(
            evaluate_consent_gate(ConsentRequirement::Required, request(), Some(&e), NOW),
            ConsentGateOutcome::Blocked(ConsentBlockReason::InvalidValidityWindow)
        );
        let mut e = evidence(ConsentState::Given);
        e.revoked = true;
        assert_eq!(
            evaluate_consent_gate(ConsentRequirement::Required, request(), Some(&e), NOW),
            ConsentGateOutcome::Blocked(ConsentBlockReason::Revoked)
        );
    }

    #[test]
    fn unresolved_requirement_blocks_instead_of_defaulting_open() {
        assert_eq!(
            evaluate_consent_gate(
                ConsentRequirement::Unresolved,
                request(),
                None,
                NOW
            ),
            ConsentGateOutcome::Blocked(ConsentBlockReason::RequirementUnresolved)
        );
    }

    #[test]
    fn malformed_identifiers_are_rejected_and_not_required_is_not_a_permit() {
        let mut r = request();
        r.action_id = "  ";
        assert_eq!(
            evaluate_consent_gate(ConsentRequirement::Required, r, Some(&evidence(ConsentState::Given)), NOW),
            ConsentGateOutcome::Blocked(ConsentBlockReason::InvalidRequestScope)
        );
        let mut e = evidence(ConsentState::Given);
        e.record_id = "";
        assert_eq!(
            evaluate_consent_gate(ConsentRequirement::Required, request(), Some(&e), NOW),
            ConsentGateOutcome::Blocked(ConsentBlockReason::EmptyEvidenceScope)
        );
        assert_eq!(
            evaluate_consent_gate(ConsentRequirement::NotRequired, request(), None, NOW),
            ConsentGateOutcome::NotRequired
        );
    }
    #[test]
    fn blocked_point_of_use_gate_never_invokes_protected_action() {
        let mut invoked = false;
        let result = execute_after_consent_check(
            ConsentRequirement::Required,
            request(),
            Some(&evidence(ConsentState::Unknown)),
            NOW,
            || {
                invoked = true;
                7
            },
        );

        assert_eq!(result, Err(ConsentBlockReason::UnknownState));
        assert!(!invoked, "a blocked consent gate must not run the action closure");
    }

    #[test]
    fn satisfied_point_of_use_gate_invokes_protected_action_exactly_once() {
        let invocations = std::cell::Cell::new(0);
        let result = execute_after_consent_check(
            ConsentRequirement::Required,
            request(),
            Some(&evidence(ConsentState::Given)),
            NOW,
            || {
                invocations.set(invocations.get() + 1);
                "executed"
            },
        );

        assert_eq!(result, Ok("executed"));
        assert_eq!(invocations.get(), 1);
    }

}
