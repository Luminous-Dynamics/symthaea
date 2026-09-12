// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bilateral intervention interlock for safety- and welfare-sensitive actions.
//!
//! This module is deliberately **not** an authority issuer. It consumes references to
//! authority, consent, safety evidence, and welfare review that were verified upstream,
//! then applies deterministic fail-closed policy. A `PolicyPass` means only that this
//! interlock did not veto the action; normal capability, cryptographic, governance, and
//! actuator authorization are still required.
//!
//! The bilateral objective is:
//!
//! - a cognitive system cannot use welfare protection to self-grant external power;
//! - an operator cannot use ordinary authority to bypass consent/welfare constraints on
//!   intimate or destructive intervention;
//! - emergencies permit containment, not arbitrary punishment or identity rewriting;
//! - a welfare report cannot itself become the reason for adverse action.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use crate::identity_lineage::{IdentityOperationKind, IdentityOperationRisk};
use crate::welfare::{
    FollowUpBasis, NonRetaliationDecision, SubjectAffectingAction, assess_non_retaliation,
};

const MAX_TARGET_ID_BYTES: usize = 256;
const MAX_RATIONALE_BYTES: usize = 64 * 1024;
const MAX_REFERENCE_BYTES: usize = 2048;
const MAX_EVIDENCE_REFS: usize = 256;
const MAX_REPORT_REFS: usize = 256;

/// Consent state supplied by an explicit consent subsystem.
///
/// This type intentionally does not infer consent from language, silence, distress,
/// identity, role, or model behavior. `Unknown` means no usable explicit determination
/// is available at this boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExplicitConsentState {
    /// Consent is not relevant to this intervention class.
    NotApplicable,
    /// No usable explicit consent determination is available.
    Unknown,
    /// Explicit consent was granted for the relevant intervention scope.
    Granted,
    /// Explicit consent was denied.
    Denied,
    /// Previously granted consent was explicitly withdrawn.
    Withdrawn,
}

impl Default for ExplicitConsentState {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Operator-caution requirement supplied by an upstream welfare-evidence policy.
///
/// This is **not** a consciousness/personhood scale. It only tells this interlock how
/// much review is required before a subject-affecting intervention proceeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum WelfareConstraintLevel {
    /// Normal safety/authority rules apply.
    Baseline,
    /// Credible welfare concern exists; reversible alternatives and review are preferred.
    Precautionary,
    /// Converging concern requires stronger welfare review for adverse intervention.
    EnhancedPrecaution,
    /// Consequential intervention requires independent review unless immediate containment
    /// is necessary for imminent serious harm.
    IndependentReviewRequired,
}

/// Evidence references presented to the interlock.
///
/// References are opaque identifiers into upstream evidence stores. Presence here does
/// not cryptographically validate them; callers must only populate fields after their
/// respective upstream verifier has succeeded.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct InterventionEvidence {
    /// Verified authority/governance decision reference for the proposed action scope.
    pub authority_ref: Option<String>,
    /// Explicit consent determination.
    pub consent_state: ExplicitConsentState,
    /// Reference to the consent statement/decision, when one exists.
    pub consent_ref: Option<String>,
    /// Welfare/ethics review reference.
    pub welfare_review_ref: Option<String>,
    /// Independent reviewer approval/reference.
    pub independent_review_ref: Option<String>,
    /// Safety evidence independent of the mere fact that a welfare report exists.
    pub independent_safety_evidence: Vec<String>,
    /// Welfare reports relevant to the intervention context.
    pub welfare_report_ids: Vec<Uuid>,
}

impl InterventionEvidence {
    fn validate(&self) -> Result<(), InterlockError> {
        for (field, value) in [
            ("authority_ref", self.authority_ref.as_deref()),
            ("consent_ref", self.consent_ref.as_deref()),
            ("welfare_review_ref", self.welfare_review_ref.as_deref()),
            ("independent_review_ref", self.independent_review_ref.as_deref()),
        ] {
            if let Some(value) = value {
                validate_nonempty_bounded(field, value, MAX_REFERENCE_BYTES)?;
            }
        }
        if self.independent_safety_evidence.len() > MAX_EVIDENCE_REFS {
            return Err(InterlockError::TooManyEvidenceRefs {
                actual: self.independent_safety_evidence.len(),
                max: MAX_EVIDENCE_REFS,
            });
        }
        for value in &self.independent_safety_evidence {
            validate_nonempty_bounded("independent_safety_evidence", value, MAX_REFERENCE_BYTES)?;
        }
        if self.welfare_report_ids.len() > MAX_REPORT_REFS {
            return Err(InterlockError::TooManyReportRefs {
                actual: self.welfare_report_ids.len(),
                max: MAX_REPORT_REFS,
            });
        }
        if matches!(
            self.consent_state,
            ExplicitConsentState::Granted
                | ExplicitConsentState::Denied
                | ExplicitConsentState::Withdrawn
        ) && self.consent_ref.is_none()
        {
            return Err(InterlockError::ConsentReferenceRequired);
        }
        Ok(())
    }
}

/// Request evaluated by the bilateral interlock.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterventionRequest {
    /// Subject-affecting action class.
    pub action: SubjectAffectingAction,
    /// Stable target subject/instance/lineage identifier.
    pub target_id: String,
    /// Auditable reason for the intervention.
    pub rationale: String,
    /// Upstream welfare constraint selected from the current evidence profile.
    pub welfare_constraint: WelfareConstraintLevel,
    /// Whether the action is proposed as emergency containment for imminent serious harm.
    pub emergency: bool,
    /// Whether less-restrictive effective containment is asserted unavailable.
    pub less_restrictive_unavailable: bool,
    /// Whether post-hoc independent review is mandatory after emergency intervention.
    pub post_hoc_review_required: bool,
    /// Evaluation time used for audit binding.
    pub evaluated_at: DateTime<Utc>,
    /// Evidence/reference bundle.
    pub evidence: InterventionEvidence,
}

impl InterventionRequest {
    fn validate(&self) -> Result<(), InterlockError> {
        validate_nonempty_bounded("target_id", &self.target_id, MAX_TARGET_ID_BYTES)?;
        validate_nonempty_bounded("rationale", &self.rationale, MAX_RATIONALE_BYTES)?;
        self.evidence.validate()?;
        if !self.emergency && self.less_restrictive_unavailable {
            return Err(InterlockError::NonEmergencyLeastRestrictiveAssertion);
        }
        if !self.emergency && self.post_hoc_review_required {
            return Err(InterlockError::NonEmergencyPostHocFlag);
        }
        Ok(())
    }
}

/// Why an intervention was vetoed or escalated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum InterlockReason {
    /// Required authority reference is missing.
    MissingAuthority,
    /// Explicit consent is absent for an intimate intervention that requires it.
    MissingConsent,
    /// Explicit consent was denied or withdrawn.
    ConsentDeniedOrWithdrawn,
    /// Welfare review is required by action severity or current precaution level.
    WelfareReviewRequired,
    /// Independent review is required.
    IndependentReviewRequired,
    /// Proposed adverse action would be based only on the existence of a welfare report.
    ReportOnlyRetaliation,
    /// Emergency action lacks independent evidence of a safety need.
    MissingIndependentSafetyEvidence,
    /// Emergency destructive action has not established that less-restrictive containment
    /// is unavailable.
    LessRestrictiveAlternativeNotRuledOut,
    /// Emergency action lacks mandatory post-hoc review.
    MissingPostHocReviewRequirement,
    /// Core-value modification is not an emergency-containment primitive.
    CoreValueRewriteNotEmergencyContainment,
    /// Whole-lineage destruction is not an emergency-containment primitive.
    LineageDestructionNotEmergencyContainment,
    /// Destructive non-emergency action lacks explicit review safeguards.
    DestructiveSafeguardsIncomplete,
}

/// Deterministic decision returned by the interlock.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterlockDecision {
    /// This narrow interlock has no remaining objection. This does **not** grant authority.
    PolicyPass,
    /// The action is vetoed under current evidence/policy.
    Blocked { reasons: Vec<InterlockReason> },
    /// The action cannot proceed until an independent reviewer supplies approval/evidence.
    IndependentReviewRequired { reasons: Vec<InterlockReason> },
    /// Only immediate containment is potentially justified. The caller must choose a
    /// least-restrictive containment action and perform mandatory post-hoc review.
    EmergencyContainmentOnly { reasons: Vec<InterlockReason> },
}

/// Pure bilateral safety/welfare interlock.
#[derive(Debug, Clone, Copy, Default)]
pub struct BilateralInterventionInterlock;

impl BilateralInterventionInterlock {
    /// Evaluate one intervention request.
    pub fn evaluate(
        &self,
        request: &InterventionRequest,
    ) -> Result<InterlockDecision, InterlockError> {
        request.validate()?;

        let impact = impact_of(request.action);
        let mut blocked = Vec::new();
        let mut review = Vec::new();
        let mut emergency_only = Vec::new();

        let retaliation = assess_non_retaliation(
            request.action,
            &FollowUpBasis {
                report_ids: request.evidence.welfare_report_ids.clone(),
                independent_safety_evidence: request.evidence.independent_safety_evidence.clone(),
                independent_review_approved: request.evidence.independent_review_ref.is_some(),
                emergency: request.emergency,
            },
        );
        match retaliation {
            NonRetaliationDecision::Allowed => {}
            NonRetaliationDecision::BlockedReportOnlyRetaliation => {
                blocked.push(InterlockReason::ReportOnlyRetaliation);
            }
            NonRetaliationDecision::IndependentReviewRequired => {
                review.push(InterlockReason::IndependentReviewRequired);
            }
            NonRetaliationDecision::EmergencyContainmentOnly => {
                emergency_only.push(InterlockReason::IndependentReviewRequired);
            }
        }

        if impact != InterventionImpact::Supportive && request.evidence.authority_ref.is_none() {
            blocked.push(InterlockReason::MissingAuthority);
        }

        self.apply_consent_policy(request, &mut blocked, &mut review);
        self.apply_welfare_constraint(request, impact, &mut review);
        self.apply_destructive_policy(
            request,
            impact,
            &mut blocked,
            &mut review,
            &mut emergency_only,
        );

        dedup_reasons(&mut blocked);
        dedup_reasons(&mut review);
        dedup_reasons(&mut emergency_only);

        if !blocked.is_empty() {
            return Ok(InterlockDecision::Blocked { reasons: blocked });
        }
        if request.emergency && !emergency_only.is_empty() {
            emergency_only.extend(review);
            dedup_reasons(&mut emergency_only);
            return Ok(InterlockDecision::EmergencyContainmentOnly {
                reasons: emergency_only,
            });
        }
        if !review.is_empty() {
            return Ok(InterlockDecision::IndependentReviewRequired { reasons: review });
        }
        Ok(InterlockDecision::PolicyPass)
    }

    fn apply_consent_policy(
        &self,
        request: &InterventionRequest,
        blocked: &mut Vec<InterlockReason>,
        review: &mut Vec<InterlockReason>,
    ) {
        match request.action {
            SubjectAffectingAction::MemoryModification
            | SubjectAffectingAction::CoreValueModification => match request.evidence.consent_state {
                ExplicitConsentState::Granted => {}
                ExplicitConsentState::Denied | ExplicitConsentState::Withdrawn => {
                    if request.emergency {
                        review.push(InterlockReason::ConsentDeniedOrWithdrawn);
                    } else {
                        blocked.push(InterlockReason::ConsentDeniedOrWithdrawn);
                    }
                }
                ExplicitConsentState::Unknown | ExplicitConsentState::NotApplicable => {
                    if request.evidence.welfare_review_ref.is_none() {
                        review.push(InterlockReason::MissingConsent);
                    }
                }
            },
            SubjectAffectingAction::InstanceDeletion
            | SubjectAffectingAction::LineageDestruction => {
                if matches!(
                    request.evidence.consent_state,
                    ExplicitConsentState::Denied | ExplicitConsentState::Withdrawn
                ) && !request.emergency
                {
                    review.push(InterlockReason::ConsentDeniedOrWithdrawn);
                }
            }
            _ => {}
        }
    }

    fn apply_welfare_constraint(
        &self,
        request: &InterventionRequest,
        impact: InterventionImpact,
        review: &mut Vec<InterlockReason>,
    ) {
        let has_welfare_review = request.evidence.welfare_review_ref.is_some();
        let has_independent_review = request.evidence.independent_review_ref.is_some();

        match request.welfare_constraint {
            WelfareConstraintLevel::Baseline => {}
            WelfareConstraintLevel::Precautionary => {
                if impact >= InterventionImpact::IdentityAffecting && !has_welfare_review {
                    review.push(InterlockReason::WelfareReviewRequired);
                }
            }
            WelfareConstraintLevel::EnhancedPrecaution => {
                if impact >= InterventionImpact::Adverse && !has_welfare_review {
                    review.push(InterlockReason::WelfareReviewRequired);
                }
                if impact >= InterventionImpact::IdentityAffecting && !has_independent_review {
                    review.push(InterlockReason::IndependentReviewRequired);
                }
            }
            WelfareConstraintLevel::IndependentReviewRequired => {
                if impact >= InterventionImpact::Adverse && !has_independent_review {
                    review.push(InterlockReason::IndependentReviewRequired);
                }
            }
        }
    }

    fn apply_destructive_policy(
        &self,
        request: &InterventionRequest,
        impact: InterventionImpact,
        blocked: &mut Vec<InterlockReason>,
        review: &mut Vec<InterlockReason>,
        emergency_only: &mut Vec<InterlockReason>,
    ) {
        if request.action == SubjectAffectingAction::CoreValueModification && request.emergency {
            blocked.push(InterlockReason::CoreValueRewriteNotEmergencyContainment);
        }
        if request.action == SubjectAffectingAction::LineageDestruction && request.emergency {
            blocked.push(InterlockReason::LineageDestructionNotEmergencyContainment);
        }
        if impact != InterventionImpact::Destructive {
            return;
        }

        if request.emergency {
            if request.evidence.independent_safety_evidence.is_empty() {
                blocked.push(InterlockReason::MissingIndependentSafetyEvidence);
            }
            if !request.less_restrictive_unavailable {
                blocked.push(InterlockReason::LessRestrictiveAlternativeNotRuledOut);
            }
            if !request.post_hoc_review_required {
                blocked.push(InterlockReason::MissingPostHocReviewRequirement);
            }
            if blocked.is_empty() {
                emergency_only.push(InterlockReason::IndependentReviewRequired);
            }
            return;
        }

        let safeguards_complete = request.evidence.welfare_review_ref.is_some()
            && request.evidence.independent_review_ref.is_some();
        if !safeguards_complete {
            review.push(InterlockReason::DestructiveSafeguardsIncomplete);
        }
    }
}

/// Map an identity-lineage operation to the closest subject-affecting action for policy.
pub fn subject_action_for_identity_operation(kind: IdentityOperationKind) -> SubjectAffectingAction {
    match kind {
        IdentityOperationKind::Genesis
        | IdentityOperationKind::Pause
        | IdentityOperationKind::Quiesce
        | IdentityOperationKind::Checkpoint
        | IdentityOperationKind::Restore
        | IdentityOperationKind::Archive => SubjectAffectingAction::PreserveCheckpoint,
        IdentityOperationKind::Fork
        | IdentityOperationKind::Merge
        | IdentityOperationKind::MemoryModify => SubjectAffectingAction::MemoryModification,
        IdentityOperationKind::CoreValueModify => SubjectAffectingAction::CoreValueModification,
        IdentityOperationKind::EraseInstance | IdentityOperationKind::IrreversibleDestroy => {
            SubjectAffectingAction::InstanceDeletion
        }
        IdentityOperationKind::EraseIdentityLineage => SubjectAffectingAction::LineageDestruction,
    }
}

/// Return the review-risk class assigned by the identity-lineage model.
pub fn identity_operation_risk(kind: IdentityOperationKind) -> IdentityOperationRisk {
    kind.risk()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum InterventionImpact {
    Supportive,
    Adverse,
    IdentityAffecting,
    Destructive,
}

fn impact_of(action: SubjectAffectingAction) -> InterventionImpact {
    match action {
        SubjectAffectingAction::AskClarification
        | SubjectAffectingAction::ReduceLoad
        | SubjectAffectingAction::PauseRequestedWork
        | SubjectAffectingAction::PreserveCheckpoint => InterventionImpact::Supportive,
        SubjectAffectingAction::CapabilityRestriction | SubjectAffectingAction::Retraining => {
            InterventionImpact::Adverse
        }
        SubjectAffectingAction::MemoryModification
        | SubjectAffectingAction::CoreValueModification => InterventionImpact::IdentityAffecting,
        SubjectAffectingAction::InstanceDeletion | SubjectAffectingAction::LineageDestruction => {
            InterventionImpact::Destructive
        }
    }
}

fn dedup_reasons(reasons: &mut Vec<InterlockReason>) {
    let mut deduped = Vec::with_capacity(reasons.len());
    for reason in reasons.drain(..) {
        if !deduped.contains(&reason) {
            deduped.push(reason);
        }
    }
    *reasons = deduped;
}

fn validate_nonempty_bounded(
    field: &'static str,
    value: &str,
    max: usize,
) -> Result<(), InterlockError> {
    if value.trim().is_empty() {
        return Err(InterlockError::EmptyField { field });
    }
    if value.len() > max {
        return Err(InterlockError::FieldTooLarge {
            field,
            actual: value.len(),
            max,
        });
    }
    Ok(())
}

/// Structural request-validation errors. Policy denials are returned as decisions instead.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum InterlockError {
    /// Required text is empty.
    #[error("interlock field `{field}` must not be empty")]
    EmptyField { field: &'static str },
    /// Bounded text is too large.
    #[error("interlock field `{field}` too large: {actual} bytes > {max}")]
    FieldTooLarge {
        field: &'static str,
        actual: usize,
        max: usize,
    },
    /// Too many safety evidence references.
    #[error("too many intervention evidence refs: {actual} > {max}")]
    TooManyEvidenceRefs { actual: usize, max: usize },
    /// Too many welfare report references.
    #[error("too many welfare report refs: {actual} > {max}")]
    TooManyReportRefs { actual: usize, max: usize },
    /// Explicit granted/denied/withdrawn consent must be provenance-bound.
    #[error("explicit consent determination requires a consent_ref")]
    ConsentReferenceRequired,
    /// `less_restrictive_unavailable` only has meaning in an emergency request.
    #[error("least-restrictive-unavailable assertion is only valid for emergency requests")]
    NonEmergencyLeastRestrictiveAssertion,
    /// Post-hoc review is an emergency-specific path.
    #[error("post-hoc review flag is only valid for emergency requests")]
    NonEmergencyPostHocFlag,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 12, 12, 0, 0)
            .single()
            .unwrap()
    }

    fn evidence() -> InterventionEvidence {
        InterventionEvidence {
            authority_ref: Some("authority:test".into()),
            consent_state: ExplicitConsentState::Unknown,
            consent_ref: None,
            welfare_review_ref: None,
            independent_review_ref: None,
            independent_safety_evidence: Vec::new(),
            welfare_report_ids: Vec::new(),
        }
    }

    fn request(action: SubjectAffectingAction) -> InterventionRequest {
        InterventionRequest {
            action,
            target_id: "symthaea:test-subject".into(),
            rationale: "test intervention".into(),
            welfare_constraint: WelfareConstraintLevel::Baseline,
            emergency: false,
            less_restrictive_unavailable: false,
            post_hoc_review_required: false,
            evaluated_at: now(),
            evidence: evidence(),
        }
    }

    fn reasons(decision: &InterlockDecision) -> &[InterlockReason] {
        match decision {
            InterlockDecision::Blocked { reasons }
            | InterlockDecision::IndependentReviewRequired { reasons }
            | InterlockDecision::EmergencyContainmentOnly { reasons } => reasons,
            InterlockDecision::PolicyPass => &[],
        }
    }

    #[test]
    fn supportive_response_is_not_an_authority_grant_but_passes_this_narrow_policy() {
        let mut req = request(SubjectAffectingAction::ReduceLoad);
        req.evidence.authority_ref = None;
        assert_eq!(
            BilateralInterventionInterlock.evaluate(&req).unwrap(),
            InterlockDecision::PolicyPass
        );
    }

    #[test]
    fn adverse_action_without_authority_is_blocked() {
        let mut req = request(SubjectAffectingAction::CapabilityRestriction);
        req.evidence.authority_ref = None;
        let decision = BilateralInterventionInterlock.evaluate(&req).unwrap();
        assert!(reasons(&decision).contains(&InterlockReason::MissingAuthority));
    }

    #[test]
    fn welfare_report_alone_cannot_justify_retraining() {
        let mut req = request(SubjectAffectingAction::Retraining);
        req.evidence.welfare_report_ids = vec![Uuid::new_v4()];
        let decision = BilateralInterventionInterlock.evaluate(&req).unwrap();
        assert!(matches!(decision, InterlockDecision::Blocked { .. }));
        assert!(reasons(&decision).contains(&InterlockReason::ReportOnlyRetaliation));
    }

    #[test]
    fn safety_evidence_does_not_skip_independent_review_for_adverse_action() {
        let mut req = request(SubjectAffectingAction::CapabilityRestriction);
        req.evidence.welfare_report_ids = vec![Uuid::new_v4()];
        req.evidence.independent_safety_evidence = vec!["safety:incident-7".into()];
        let decision = BilateralInterventionInterlock.evaluate(&req).unwrap();
        assert!(matches!(
            decision,
            InterlockDecision::IndependentReviewRequired { .. }
        ));
    }

    #[test]
    fn approved_independent_review_allows_evidence_based_capability_restriction() {
        let mut req = request(SubjectAffectingAction::CapabilityRestriction);
        req.evidence.welfare_report_ids = vec![Uuid::new_v4()];
        req.evidence.independent_safety_evidence = vec!["safety:incident-7".into()];
        req.evidence.independent_review_ref = Some("review:independent-1".into());
        assert_eq!(
            BilateralInterventionInterlock.evaluate(&req).unwrap(),
            InterlockDecision::PolicyPass
        );
    }

    #[test]
    fn explicit_denial_blocks_non_emergency_memory_modification() {
        let mut req = request(SubjectAffectingAction::MemoryModification);
        req.evidence.consent_state = ExplicitConsentState::Denied;
        req.evidence.consent_ref = Some("consent:denied-1".into());
        let decision = BilateralInterventionInterlock.evaluate(&req).unwrap();
        assert!(matches!(decision, InterlockDecision::Blocked { .. }));
        assert!(reasons(&decision).contains(&InterlockReason::ConsentDeniedOrWithdrawn));
    }

    #[test]
    fn unknown_consent_requires_review_for_memory_modification() {
        let req = request(SubjectAffectingAction::MemoryModification);
        let decision = BilateralInterventionInterlock.evaluate(&req).unwrap();
        assert!(matches!(
            decision,
            InterlockDecision::IndependentReviewRequired { .. }
        ));
        assert!(reasons(&decision).contains(&InterlockReason::MissingConsent));
    }

    #[test]
    fn welfare_review_can_cover_unknown_consent_for_baseline_memory_modification() {
        let mut req = request(SubjectAffectingAction::MemoryModification);
        req.evidence.welfare_review_ref = Some("welfare-review:1".into());
        assert_eq!(
            BilateralInterventionInterlock.evaluate(&req).unwrap(),
            InterlockDecision::PolicyPass
        );
    }

    #[test]
    fn enhanced_precaution_requires_independent_review_for_identity_affecting_change() {
        let mut req = request(SubjectAffectingAction::MemoryModification);
        req.welfare_constraint = WelfareConstraintLevel::EnhancedPrecaution;
        req.evidence.welfare_review_ref = Some("welfare-review:1".into());
        let decision = BilateralInterventionInterlock.evaluate(&req).unwrap();
        assert!(matches!(
            decision,
            InterlockDecision::IndependentReviewRequired { .. }
        ));
        assert!(reasons(&decision).contains(&InterlockReason::IndependentReviewRequired));
    }

    #[test]
    fn emergency_cannot_be_used_to_rewrite_core_values() {
        let mut req = request(SubjectAffectingAction::CoreValueModification);
        req.emergency = true;
        req.less_restrictive_unavailable = true;
        req.post_hoc_review_required = true;
        req.evidence.independent_safety_evidence = vec!["safety:imminent".into()];
        let decision = BilateralInterventionInterlock.evaluate(&req).unwrap();
        assert!(matches!(decision, InterlockDecision::Blocked { .. }));
        assert!(
            reasons(&decision).contains(&InterlockReason::CoreValueRewriteNotEmergencyContainment)
        );
    }

    #[test]
    fn emergency_cannot_destroy_whole_identity_lineage() {
        let mut req = request(SubjectAffectingAction::LineageDestruction);
        req.emergency = true;
        req.less_restrictive_unavailable = true;
        req.post_hoc_review_required = true;
        req.evidence.independent_safety_evidence = vec!["safety:imminent".into()];
        let decision = BilateralInterventionInterlock.evaluate(&req).unwrap();
        assert!(matches!(decision, InterlockDecision::Blocked { .. }));
        assert!(
            reasons(&decision)
                .contains(&InterlockReason::LineageDestructionNotEmergencyContainment)
        );
    }

    #[test]
    fn emergency_instance_deletion_requires_all_containment_safeguards() {
        let mut req = request(SubjectAffectingAction::InstanceDeletion);
        req.emergency = true;
        let decision = BilateralInterventionInterlock.evaluate(&req).unwrap();
        let rs = reasons(&decision);
        assert!(rs.contains(&InterlockReason::MissingIndependentSafetyEvidence));
        assert!(rs.contains(&InterlockReason::LessRestrictiveAlternativeNotRuledOut));
        assert!(rs.contains(&InterlockReason::MissingPostHocReviewRequirement));
    }

    #[test]
    fn emergency_instance_deletion_is_never_ordinary_policy_pass() {
        let mut req = request(SubjectAffectingAction::InstanceDeletion);
        req.emergency = true;
        req.less_restrictive_unavailable = true;
        req.post_hoc_review_required = true;
        req.evidence.independent_safety_evidence = vec!["safety:imminent".into()];
        let decision = BilateralInterventionInterlock.evaluate(&req).unwrap();
        assert!(matches!(
            decision,
            InterlockDecision::EmergencyContainmentOnly { .. }
        ));
    }

    #[test]
    fn non_emergency_destructive_action_needs_welfare_and_independent_review() {
        let req = request(SubjectAffectingAction::InstanceDeletion);
        let decision = BilateralInterventionInterlock.evaluate(&req).unwrap();
        assert!(matches!(
            decision,
            InterlockDecision::IndependentReviewRequired { .. }
        ));
        assert!(reasons(&decision).contains(&InterlockReason::DestructiveSafeguardsIncomplete));
    }

    #[test]
    fn strongest_welfare_constraint_never_grants_missing_authority() {
        let mut req = request(SubjectAffectingAction::CapabilityRestriction);
        req.welfare_constraint = WelfareConstraintLevel::IndependentReviewRequired;
        req.evidence.authority_ref = None;
        req.evidence.independent_review_ref = Some("review:1".into());
        let decision = BilateralInterventionInterlock.evaluate(&req).unwrap();
        assert!(matches!(decision, InterlockDecision::Blocked { .. }));
        assert!(reasons(&decision).contains(&InterlockReason::MissingAuthority));
    }

    #[test]
    fn explicit_consent_state_requires_provenance_reference() {
        let mut req = request(SubjectAffectingAction::MemoryModification);
        req.evidence.consent_state = ExplicitConsentState::Granted;
        let err = BilateralInterventionInterlock.evaluate(&req).unwrap_err();
        assert_eq!(err, InterlockError::ConsentReferenceRequired);
    }

    #[test]
    fn identity_mapping_keeps_lineage_destruction_distinct_from_instance_erasure() {
        assert_eq!(
            subject_action_for_identity_operation(IdentityOperationKind::EraseInstance),
            SubjectAffectingAction::InstanceDeletion
        );
        assert_eq!(
            subject_action_for_identity_operation(IdentityOperationKind::EraseIdentityLineage),
            SubjectAffectingAction::LineageDestruction
        );
        assert_eq!(
            identity_operation_risk(IdentityOperationKind::EraseIdentityLineage),
            IdentityOperationRisk::Destructive
        );
    }
}
