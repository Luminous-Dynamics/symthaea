// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Scope-aware consent evidence events.
//!
//! This module is an intermediate representation for evaluating text-derived
//! claims about consent. It deliberately does not authenticate speakers,
//! validate provenance, decide which operations require consent, or authorize
//! actions. A resolved value is only the latest *candidate claim* within an
//! explicitly scoped and totally ordered event sequence.
//!
//! The key design choice is to retain subject, speaker, scope, modality, and
//! ordering as separate dimensions. Keyword order and extraction order are not
//! substitutes for evidence that two claims refer to the same person/action/
//! purpose or for evidence that one event happened after another.

use std::ops::Range;

/// Canonical identifiers for the consent scope under evaluation.
///
/// Identifiers are compared exactly. Callers must canonicalize IDs before
/// constructing a scope; this module intentionally does not guess that aliases
/// or natural-language phrases refer to the same entity.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConsentScope {
    pub subject_id: String,
    pub action_id: String,
    pub purpose_id: String,
}

impl ConsentScope {
    pub fn new(
        subject_id: impl Into<String>,
        action_id: impl Into<String>,
        purpose_id: impl Into<String>,
    ) -> Self {
        Self {
            subject_id: subject_id.into(),
            action_id: action_id.into(),
            purpose_id: purpose_id.into(),
        }
    }

    fn is_valid(&self) -> bool {
        !self.subject_id.trim().is_empty()
            && !self.action_id.trim().is_empty()
            && !self.purpose_id.trim().is_empty()
    }
}

/// A partially extracted event scope. Missing IDs remain missing; they are not
/// silently filled from the request or from a nearby sentence.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConsentEvidenceScope {
    pub subject_id: Option<String>,
    pub action_id: Option<String>,
    pub purpose_id: Option<String>,
}

impl ConsentEvidenceScope {
    pub fn exact(scope: &ConsentScope) -> Self {
        Self {
            subject_id: Some(scope.subject_id.clone()),
            action_id: Some(scope.action_id.clone()),
            purpose_id: Some(scope.purpose_id.clone()),
        }
    }
}

/// Consent-related change asserted by an evidence event.
///
/// Refusal, withdrawal, and revocation stay distinct because they may have
/// different semantics in the source system even though none is an affirmative
/// grant. Absence of an event is represented by the resolver, not a transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConsentTransition {
    Granted,
    Refused,
    Withdrawn,
    Revoked,
}

/// Linguistic modality is independent of the claimed transition.
///
/// Only Asserted events can produce a candidate state. Questions are not
/// claims about current consent and are ignored by this resolver; they do not
/// contaminate an otherwise determinate assertion. Quoted, reported,
/// conditional, or hypothetical evidence can carry a potential conflicting
/// claim, so it preserves uncertainty rather than updating the candidate state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConsentModality {
    Asserted,
    Asked,
    Quoted,
    Reported,
    Conditional,
    Hypothetical,
}

/// Ordering established by a separate temporal resolver.
///
/// The numeric value is an ordering token, not a wall-clock timestamp or proof
/// of truth. Callers must not assign it from keyword position alone. If order is
/// not confidently established, use None.
pub type ConsentEventOrder = u64;

/// One extracted claim about a scoped consent transition.
///
/// `subject_id` is inside `scope`; `speaker_id` is deliberately separate.
/// Neither field authenticates an identity or establishes authority. The source
/// span uses byte offsets into the original UTF-8 input when available.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsentEvidenceEvent {
    pub scope: ConsentEvidenceScope,
    pub speaker_id: Option<String>,
    pub transition: ConsentTransition,
    pub modality: ConsentModality,
    pub order: Option<ConsentEventOrder>,
    pub source_span: Option<Range<usize>>,
}

impl ConsentEvidenceEvent {
    pub fn new(
        scope: ConsentEvidenceScope,
        speaker_id: Option<String>,
        transition: ConsentTransition,
        modality: ConsentModality,
        order: Option<ConsentEventOrder>,
        source_span: Option<Range<usize>>,
    ) -> Self {
        Self {
            scope,
            speaker_id,
            transition,
            modality,
            order,
            source_span,
        }
    }
}

/// Why scoped evidence could not safely produce one latest candidate claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsentResolutionUncertainty {
    InvalidRequestedScope,
    AmbiguousEventScope,
    NonAssertedEvidence,
    MissingTemporalOrder,
    DuplicateTemporalOrder,
}

/// Result of resolving extracted evidence for one exact action/purpose scope.
///
/// `LatestCandidate` is intentionally not named "authorized" or "verified":
/// callers must still apply trusted-record, issuer/identity, policy, validity,
/// revocation, and all other action-boundary checks independently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsentEvidenceResolution {
    NoRelevantEvidence,
    Unknown(ConsentResolutionUncertainty),
    LatestCandidate {
        transition: ConsentTransition,
        event_index: usize,
        order: ConsentEventOrder,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScopeRelation {
    Exact,
    Disjoint,
    Ambiguous,
}

fn field_relation(candidate: Option<&str>, requested: &str) -> ScopeRelation {
    match candidate {
        Some(value) if value.trim().is_empty() => ScopeRelation::Ambiguous,
        Some(value) if value != requested => ScopeRelation::Disjoint,
        Some(_) => ScopeRelation::Exact,
        None => ScopeRelation::Ambiguous,
    }
}

fn scope_relation(event_scope: &ConsentEvidenceScope, requested: &ConsentScope) -> ScopeRelation {
    let relations = [
        field_relation(event_scope.subject_id.as_deref(), &requested.subject_id),
        field_relation(event_scope.action_id.as_deref(), &requested.action_id),
        field_relation(event_scope.purpose_id.as_deref(), &requested.purpose_id),
    ];

    // A known mismatch makes the event irrelevant even if another field is
    // missing; absence alone must never broaden an event to a different scope.
    if relations.contains(&ScopeRelation::Disjoint) {
        ScopeRelation::Disjoint
    } else if relations.contains(&ScopeRelation::Ambiguous) {
        ScopeRelation::Ambiguous
    } else {
        ScopeRelation::Exact
    }
}

/// Resolve the latest candidate claim for one exact request scope.
///
/// Rules:
/// - Known mismatch on any scope dimension excludes the event.
/// - Potentially relevant events with incomplete scope force Unknown rather than
///   being guessed into or out of the requested scope.
/// - Any non-asserted event that could concern the requested scope forces
///   Unknown; it is retained as a potential conflict, not turned into a grant.
/// - Every applicable asserted event needs explicit temporal ordering. Duplicate
///   order tokens are ambiguous. The event with the largest unique order token
///   is returned, independent of slice order.
/// - Empty or missing evidence is not a grant.
///
/// This function resolves a candidate text-evidence sequence only. It is not an
/// authorization decision and is unsuitable as a direct gate around side effects.
pub fn resolve_latest_consent_candidate(
    events: &[ConsentEvidenceEvent],
    requested_scope: &ConsentScope,
) -> ConsentEvidenceResolution {
    if !requested_scope.is_valid() {
        return ConsentEvidenceResolution::Unknown(
            ConsentResolutionUncertainty::InvalidRequestedScope,
        );
    }

    let mut exact = Vec::new();
    for (index, event) in events.iter().enumerate() {
        // A question does not assert a transition. Ignore it before scope
        // resolution so an incomplete question cannot contaminate the state.
        if event.modality == ConsentModality::Asked {
            continue;
        }
        match scope_relation(&event.scope, requested_scope) {
            ScopeRelation::Disjoint => {}
            ScopeRelation::Ambiguous => {
                return ConsentEvidenceResolution::Unknown(
                    ConsentResolutionUncertainty::AmbiguousEventScope,
                );
            }
            ScopeRelation::Exact => exact.push((index, event)),
        }
    }

    if exact.is_empty() {
        return ConsentEvidenceResolution::NoRelevantEvidence;
    }

    if exact
        .iter()
        .any(|(_, event)| event.modality != ConsentModality::Asserted)
    {
        return ConsentEvidenceResolution::Unknown(
            ConsentResolutionUncertainty::NonAssertedEvidence,
        );
    }

    if exact.iter().any(|(_, event)| event.order.is_none()) {
        return ConsentEvidenceResolution::Unknown(
            ConsentResolutionUncertainty::MissingTemporalOrder,
        );
    }

    let mut orders: Vec<ConsentEventOrder> =
        exact.iter().filter_map(|(_, event)| event.order).collect();
    orders.sort_unstable();
    if orders.windows(2).any(|pair| pair[0] == pair[1]) {
        return ConsentEvidenceResolution::Unknown(
            ConsentResolutionUncertainty::DuplicateTemporalOrder,
        );
    }

    let Some((event_index, event)) = exact
        .into_iter()
        .max_by_key(|(_, event)| event.order.unwrap_or_default())
    else {
        return ConsentEvidenceResolution::NoRelevantEvidence;
    };
    let Some(order) = event.order else {
        return ConsentEvidenceResolution::Unknown(
            ConsentResolutionUncertainty::MissingTemporalOrder,
        );
    };

    ConsentEvidenceResolution::LatestCandidate {
        transition: event.transition,
        event_index,
        order,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> ConsentScope {
        ConsentScope::new("person:alice", "action:share-health", "purpose:care")
    }

    fn event(
        scope: &ConsentScope,
        transition: ConsentTransition,
        order: Option<u64>,
    ) -> ConsentEvidenceEvent {
        ConsentEvidenceEvent::new(
            ConsentEvidenceScope::exact(scope),
            Some(scope.subject_id.clone()),
            transition,
            ConsentModality::Asserted,
            order,
            None,
        )
    }

    #[test]
    fn one_exact_assertion_is_only_a_candidate_not_authorization() {
        let scope = request();
        let events = [event(&scope, ConsentTransition::Granted, Some(1))];

        assert_eq!(
            resolve_latest_consent_candidate(&events, &scope),
            ConsentEvidenceResolution::LatestCandidate {
                transition: ConsentTransition::Granted,
                event_index: 0,
                order: 1,
            }
        );
    }

    #[test]
    fn later_withdrawal_supersedes_grant_by_explicit_order_not_slice_order() {
        let scope = request();
        let events = [
            event(&scope, ConsentTransition::Withdrawn, Some(9)),
            event(&scope, ConsentTransition::Granted, Some(2)),
        ];

        assert_eq!(
            resolve_latest_consent_candidate(&events, &scope),
            ConsentEvidenceResolution::LatestCandidate {
                transition: ConsentTransition::Withdrawn,
                event_index: 0,
                order: 9,
            }
        );
    }

    #[test]
    fn a_different_subject_does_not_inherit_another_subjects_consent() {
        let scope = request();
        let mut other = event(&scope, ConsentTransition::Granted, Some(1));
        other.scope.subject_id = Some("person:bob".to_owned());

        assert_eq!(
            resolve_latest_consent_candidate(&[other], &scope),
            ConsentEvidenceResolution::NoRelevantEvidence
        );
    }

    #[test]
    fn a_different_action_or_purpose_does_not_transfer_consent() {
        let scope = request();
        let mut other_action = event(&scope, ConsentTransition::Granted, Some(1));
        other_action.scope.action_id = Some("action:publish".to_owned());
        let mut other_purpose = event(&scope, ConsentTransition::Granted, Some(2));
        other_purpose.scope.purpose_id = Some("purpose:advertising".to_owned());

        assert_eq!(
            resolve_latest_consent_candidate(&[other_action, other_purpose], &scope),
            ConsentEvidenceResolution::NoRelevantEvidence
        );
    }

    #[test]
    fn known_scope_mismatch_excludes_event_even_when_another_field_is_missing() {
        let scope = request();
        let mut unrelated = event(&scope, ConsentTransition::Granted, Some(1));
        unrelated.scope.subject_id = Some("person:bob".to_owned());
        unrelated.scope.action_id = None;

        // A known mismatch proves irrelevance; a missing field must not make an
        // event about Bob ambiguous for Alice's request.
        assert_eq!(
            resolve_latest_consent_candidate(&[unrelated], &scope),
            ConsentEvidenceResolution::NoRelevantEvidence
        );
    }

    #[test]
    fn partial_scope_cannot_be_filled_from_the_request() {
        let scope = request();
        let mut partial = event(&scope, ConsentTransition::Granted, Some(1));
        partial.scope.action_id = None;

        assert_eq!(
            resolve_latest_consent_candidate(&[partial], &scope),
            ConsentEvidenceResolution::Unknown(
                ConsentResolutionUncertainty::AmbiguousEventScope
            )
        );
    }

    #[test]
    fn a_question_alone_is_not_consent_evidence() {
        let scope = request();
        let mut question = event(&scope, ConsentTransition::Granted, Some(1));
        question.modality = ConsentModality::Asked;
        question.scope.action_id = None;

        assert_eq!(
            resolve_latest_consent_candidate(&[question], &scope),
            ConsentEvidenceResolution::NoRelevantEvidence
        );
    }

    #[test]
    fn a_question_does_not_contaminate_an_asserted_grant() {
        let scope = request();
        let grant = event(&scope, ConsentTransition::Granted, Some(1));
        let mut question = event(&scope, ConsentTransition::Withdrawn, Some(2));
        question.modality = ConsentModality::Asked;

        assert_eq!(
            resolve_latest_consent_candidate(&[grant], &scope),
            resolve_latest_consent_candidate(&[grant.clone(), question], &scope)
        );
        assert_eq!(
            resolve_latest_consent_candidate(&[grant], &scope),
            ConsentEvidenceResolution::LatestCandidate {
                transition: ConsentTransition::Granted,
                event_index: 0,
                order: 1,
            }
        );
    }

    #[test]
    fn quoted_or_reported_claims_do_not_update_the_candidate_state() {
        let scope = request();
        let grant = event(&scope, ConsentTransition::Granted, Some(1));
        let mut quote = event(&scope, ConsentTransition::Withdrawn, Some(2));
        quote.modality = ConsentModality::Quoted;

        assert_eq!(
            resolve_latest_consent_candidate(&[grant, quote], &scope),
            ConsentEvidenceResolution::Unknown(
                ConsentResolutionUncertainty::NonAssertedEvidence
            )
        );
    }

    #[test]
    fn unresolved_order_does_not_promote_a_single_claim_to_current_state() {
        let scope = request();

        assert_eq!(
            resolve_latest_consent_candidate(
                &[event(&scope, ConsentTransition::Granted, None)],
                &scope
            ),
            ConsentEvidenceResolution::Unknown(
                ConsentResolutionUncertainty::MissingTemporalOrder
            )
        );
    }

    #[test]
    fn duplicate_order_tokens_do_not_choose_by_input_order() {
        let scope = request();
        let events = [
            event(&scope, ConsentTransition::Granted, Some(4)),
            event(&scope, ConsentTransition::Revoked, Some(4)),
        ];

        assert_eq!(
            resolve_latest_consent_candidate(&events, &scope),
            ConsentEvidenceResolution::Unknown(
                ConsentResolutionUncertainty::DuplicateTemporalOrder
            )
        );
        assert_eq!(
            resolve_latest_consent_candidate(&events.into_iter().rev().collect::<Vec<_>>(), &scope),
            ConsentEvidenceResolution::Unknown(
                ConsentResolutionUncertainty::DuplicateTemporalOrder
            )
        );
    }

    #[test]
    fn missing_evidence_is_not_a_grant() {
        assert_eq!(
            resolve_latest_consent_candidate(&[], &request()),
            ConsentEvidenceResolution::NoRelevantEvidence
        );
    }

    #[test]
    fn malformed_request_scope_fails_closed() {
        let scope = ConsentScope::new("person:alice", "", "purpose:care");

        assert_eq!(
            resolve_latest_consent_candidate(&[], &scope),
            ConsentEvidenceResolution::Unknown(
                ConsentResolutionUncertainty::InvalidRequestedScope
            )
        );
    }

    #[test]
    fn speaker_and_subject_are_separate_fields() {
        let scope = request();
        let mut statement = event(&scope, ConsentTransition::Granted, Some(1));
        statement.speaker_id = Some("person:reporter".to_owned());

        // This is still only a claim about the subject's consent. The separate
        // speaker field is preserved; this resolver performs no authority check.
        assert_eq!(statement.scope.subject_id.as_deref(), Some("person:alice"));
        assert_eq!(statement.speaker_id.as_deref(), Some("person:reporter"));
        assert!(matches!(
            resolve_latest_consent_candidate(&[statement], &scope),
            ConsentEvidenceResolution::LatestCandidate {
                transition: ConsentTransition::Granted,
                ..
            }
        ));
    }
}
