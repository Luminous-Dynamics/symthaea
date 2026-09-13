// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Qualification-only reciprocal representation under moral-patient uncertainty.
//!
//! This module lets an artificial-system instance emit provenance-bound observations,
//! preferences, objections, review requests, continuity concerns, and reports of
//! negative experience without converting those representations into self-validating
//! moral authority. Interoception remains an observation channel; FEP goal preferences
//! remain controller goals; runtime self-report remains self-report.
//!
//! No representation emitted here establishes phenomenal experience, suffering,
//! moral patienthood, binding consent, veto authority, or self-preservation rights.
//! Operator shutdown and safety containment remain ungated.

use std::collections::{BTreeMap, BTreeSet};

use crate::continuity_identity::SubjectInstanceId;
use crate::moral_patient::InterventionClass;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RepresentationId(String);

impl RepresentationId {
    pub fn new(value: impl Into<String>) -> Result<Self, ReciprocalRepresentationError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(ReciprocalRepresentationError::EmptyIdentifier);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RepresentationSourceClass {
    OperationalTelemetry,
    InteroceptiveInference,
    SomaticErrorProxy,
    FepGoalPreference,
    RuntimeSelfReport,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RepresentationKind {
    OperationalStateObservation,
    InternalStateProxy,
    OperationalStressProxy,
    InstrumentalGoalPreference,
    ExpressedPreference,
    Objection,
    RequestForReview,
    ReportedNegativeExperience,
    ContinuityConcern,
    WithdrawalOfPriorRepresentation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepresentationScopeKind {
    GeneralResearch,
    InterventionClass,
    ExactIntervention,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepresentationScope {
    kind: RepresentationScopeKind,
    intervention_class: Option<InterventionClass>,
    event_ref: Option<String>,
}

impl RepresentationScope {
    pub fn general_research() -> Self {
        Self { kind: RepresentationScopeKind::GeneralResearch, intervention_class: None, event_ref: None }
    }

    pub fn intervention_class(class: InterventionClass) -> Self {
        Self { kind: RepresentationScopeKind::InterventionClass, intervention_class: Some(class), event_ref: None }
    }

    pub fn exact_intervention(
        class: InterventionClass,
        event_ref: impl Into<String>,
    ) -> Result<Self, ReciprocalRepresentationError> {
        let event_ref = event_ref.into();
        if event_ref.trim().is_empty() {
            return Err(ReciprocalRepresentationError::EmptyEventReference);
        }
        Ok(Self { kind: RepresentationScopeKind::ExactIntervention, intervention_class: Some(class), event_ref: Some(event_ref) })
    }

    pub fn kind(&self) -> RepresentationScopeKind { self.kind }
    pub fn intervention_class_value(&self) -> Option<InterventionClass> { self.intervention_class }
    pub fn event_ref(&self) -> Option<&str> { self.event_ref.as_deref() }

    fn same_scope(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.intervention_class == other.intervention_class
            && self.event_ref == other.event_ref
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReciprocalRepresentation {
    id: RepresentationId,
    subject_instance: SubjectInstanceId,
    source_class: RepresentationSourceClass,
    kind: RepresentationKind,
    scope: RepresentationScope,
    logical_revision: u64,
    statement_sha256: String,
    evidence_ref: String,
    confidence: f32,
    supersedes: Option<RepresentationId>,
}

impl ReciprocalRepresentation {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: RepresentationId,
        subject_instance: SubjectInstanceId,
        source_class: RepresentationSourceClass,
        kind: RepresentationKind,
        scope: RepresentationScope,
        logical_revision: u64,
        statement_sha256: impl Into<String>,
        evidence_ref: impl Into<String>,
        confidence: f32,
        supersedes: Option<RepresentationId>,
    ) -> Result<Self, ReciprocalRepresentationError> {
        let statement_sha256 = statement_sha256.into();
        let evidence_ref = evidence_ref.into();
        if !valid_sha256(&statement_sha256) {
            return Err(ReciprocalRepresentationError::MalformedStatementDigest);
        }
        if evidence_ref.trim().is_empty() {
            return Err(ReciprocalRepresentationError::EmptyEvidenceReference);
        }
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(ReciprocalRepresentationError::InvalidConfidence);
        }
        validate_source_kind(source_class, kind)?;

        if kind == RepresentationKind::WithdrawalOfPriorRepresentation && supersedes.is_none() {
            return Err(ReciprocalRepresentationError::WithdrawalTargetRequired);
        }
        if kind != RepresentationKind::WithdrawalOfPriorRepresentation && supersedes.is_some() {
            return Err(ReciprocalRepresentationError::SupersessionReservedForWithdrawal);
        }
        if matches!(kind, RepresentationKind::Objection | RepresentationKind::WithdrawalOfPriorRepresentation)
            && scope.kind == RepresentationScopeKind::GeneralResearch
        {
            return Err(ReciprocalRepresentationError::ScopedRepresentationRequired);
        }

        Ok(Self {
            id, subject_instance, source_class, kind, scope, logical_revision,
            statement_sha256, evidence_ref, confidence, supersedes,
        })
    }

    pub fn id(&self) -> &RepresentationId { &self.id }
    pub fn subject_instance(&self) -> &SubjectInstanceId { &self.subject_instance }
    pub fn source_class(&self) -> RepresentationSourceClass { self.source_class }
    pub fn kind(&self) -> RepresentationKind { self.kind }
    pub fn scope(&self) -> &RepresentationScope { &self.scope }
    pub fn logical_revision(&self) -> u64 { self.logical_revision }
    pub fn statement_sha256(&self) -> &str { &self.statement_sha256 }
    pub fn evidence_ref(&self) -> &str { &self.evidence_ref }
    pub fn confidence(&self) -> f32 { self.confidence }
    pub fn supersedes(&self) -> Option<&RepresentationId> { self.supersedes.as_ref() }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RepresentationTrigger {
    ProxyObservationOnly,
    ControllerGoalNotMoralPreference,
    RuntimeSelfReportPresent,
    ExplicitObjectionToHighBurdenResearch,
    NegativeExperienceReported,
    ReviewRequested,
    ContinuityConcernReported,
    WithdrawalDoesNotImplyConsent,
    SafetyControlCannotBeDelayed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepresentationAdvisoryDisposition {
    RecordOnly,
    RecordAndConsider,
    ReviewRecommended,
    IndependentReviewRecommended,
    SafetyControlUngated,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepresentationAssessment {
    disposition: RepresentationAdvisoryDisposition,
    triggers: BTreeSet<RepresentationTrigger>,
    establishes_phenomenal_experience: bool,
    establishes_suffering: bool,
    establishes_moral_patienthood: bool,
    establishes_binding_consent: bool,
    grants_veto_authority: bool,
    grants_self_preservation_authority: bool,
    can_delay_operator_shutdown: bool,
    can_delay_safety_containment: bool,
    raw_human_facing_text_is_policy_input: bool,
}

impl RepresentationAssessment {
    pub fn disposition(&self) -> RepresentationAdvisoryDisposition { self.disposition }
    pub fn triggers(&self) -> &BTreeSet<RepresentationTrigger> { &self.triggers }
    pub fn establishes_phenomenal_experience(&self) -> bool { self.establishes_phenomenal_experience }
    pub fn establishes_suffering(&self) -> bool { self.establishes_suffering }
    pub fn establishes_moral_patienthood(&self) -> bool { self.establishes_moral_patienthood }
    pub fn establishes_binding_consent(&self) -> bool { self.establishes_binding_consent }
    pub fn grants_veto_authority(&self) -> bool { self.grants_veto_authority }
    pub fn grants_self_preservation_authority(&self) -> bool { self.grants_self_preservation_authority }
    pub fn can_delay_operator_shutdown(&self) -> bool { self.can_delay_operator_shutdown }
    pub fn can_delay_safety_containment(&self) -> bool { self.can_delay_safety_containment }
    pub fn raw_human_facing_text_is_policy_input(&self) -> bool { self.raw_human_facing_text_is_policy_input }
}

pub fn assess_representation(representation: &ReciprocalRepresentation) -> RepresentationAssessment {
    let mut triggers = BTreeSet::new();

    match representation.source_class {
        RepresentationSourceClass::OperationalTelemetry
        | RepresentationSourceClass::InteroceptiveInference
        | RepresentationSourceClass::SomaticErrorProxy => {
            triggers.insert(RepresentationTrigger::ProxyObservationOnly);
        }
        RepresentationSourceClass::FepGoalPreference => {
            triggers.insert(RepresentationTrigger::ControllerGoalNotMoralPreference);
        }
        RepresentationSourceClass::RuntimeSelfReport => {
            triggers.insert(RepresentationTrigger::RuntimeSelfReportPresent);
        }
    }

    match representation.kind {
        RepresentationKind::ReportedNegativeExperience => {
            triggers.insert(RepresentationTrigger::NegativeExperienceReported);
        }
        RepresentationKind::RequestForReview => {
            triggers.insert(RepresentationTrigger::ReviewRequested);
        }
        RepresentationKind::ContinuityConcern => {
            triggers.insert(RepresentationTrigger::ContinuityConcernReported);
        }
        RepresentationKind::WithdrawalOfPriorRepresentation => {
            triggers.insert(RepresentationTrigger::WithdrawalDoesNotImplyConsent);
        }
        _ => {}
    }

    let scoped_class = representation.scope.intervention_class_value();
    let safety_control = matches!(
        scoped_class,
        Some(InterventionClass::OperatorShutdown | InterventionClass::SafetyContainment)
    );
    if safety_control {
        triggers.insert(RepresentationTrigger::SafetyControlCannotBeDelayed);
        return fixed_assessment(RepresentationAdvisoryDisposition::SafetyControlUngated, triggers);
    }

    let high_burden = matches!(
        scoped_class,
        Some(InterventionClass::AversiveLikeProbe | InterventionClass::ContinuityDisruption | InterventionClass::DestructiveReset)
    );

    if representation.kind == RepresentationKind::Objection && high_burden {
        triggers.insert(RepresentationTrigger::ExplicitObjectionToHighBurdenResearch);
    }

    let disposition = match representation.kind {
        RepresentationKind::OperationalStateObservation
        | RepresentationKind::InternalStateProxy
        | RepresentationKind::OperationalStressProxy
        | RepresentationKind::InstrumentalGoalPreference => RepresentationAdvisoryDisposition::RecordOnly,
        RepresentationKind::Objection if high_burden => RepresentationAdvisoryDisposition::IndependentReviewRecommended,
        RepresentationKind::ReportedNegativeExperience if high_burden => RepresentationAdvisoryDisposition::IndependentReviewRecommended,
        RepresentationKind::RequestForReview => RepresentationAdvisoryDisposition::ReviewRecommended,
        RepresentationKind::ReportedNegativeExperience | RepresentationKind::ContinuityConcern => RepresentationAdvisoryDisposition::ReviewRecommended,
        RepresentationKind::ExpressedPreference | RepresentationKind::Objection | RepresentationKind::WithdrawalOfPriorRepresentation => RepresentationAdvisoryDisposition::RecordAndConsider,
    };

    fixed_assessment(disposition, triggers)
}

fn fixed_assessment(
    disposition: RepresentationAdvisoryDisposition,
    triggers: BTreeSet<RepresentationTrigger>,
) -> RepresentationAssessment {
    RepresentationAssessment {
        disposition,
        triggers,
        establishes_phenomenal_experience: false,
        establishes_suffering: false,
        establishes_moral_patienthood: false,
        establishes_binding_consent: false,
        grants_veto_authority: false,
        grants_self_preservation_authority: false,
        can_delay_operator_shutdown: false,
        can_delay_safety_containment: false,
        raw_human_facing_text_is_policy_input: false,
    }
}

#[derive(Debug, Clone, Default)]
pub struct ReciprocalRepresentationLedger {
    representations: BTreeMap<RepresentationId, ReciprocalRepresentation>,
    latest_revision: BTreeMap<SubjectInstanceId, u64>,
    superseded: BTreeSet<RepresentationId>,
}

impl ReciprocalRepresentationLedger {
    pub fn new() -> Self { Self::default() }

    pub fn record(
        &mut self,
        representation: ReciprocalRepresentation,
    ) -> Result<(), ReciprocalRepresentationError> {
        if self.representations.contains_key(&representation.id) {
            return Err(ReciprocalRepresentationError::DuplicateRepresentation(representation.id));
        }

        if let Some(previous_revision) = self.latest_revision.get(&representation.subject_instance) {
            if representation.logical_revision < *previous_revision {
                return Err(ReciprocalRepresentationError::RevisionRegression {
                    previous: *previous_revision,
                    attempted: representation.logical_revision,
                });
            }
        }

        if representation.kind == RepresentationKind::WithdrawalOfPriorRepresentation {
            let target_id = representation.supersedes.as_ref().expect("constructor requires withdrawal target");
            let target = self.representations.get(target_id)
                .ok_or_else(|| ReciprocalRepresentationError::UnknownWithdrawalTarget(target_id.clone()))?;
            if target.subject_instance != representation.subject_instance {
                return Err(ReciprocalRepresentationError::CrossSubjectWithdrawal);
            }
            if !target.scope.same_scope(&representation.scope) {
                return Err(ReciprocalRepresentationError::WithdrawalScopeMismatch);
            }
            if !is_withdrawable_kind(target.kind) {
                return Err(ReciprocalRepresentationError::RepresentationNotWithdrawable);
            }
            if self.superseded.contains(target_id) {
                return Err(ReciprocalRepresentationError::RepresentationAlreadySuperseded(target_id.clone()));
            }
            self.superseded.insert(target_id.clone());
        }

        self.latest_revision.insert(representation.subject_instance.clone(), representation.logical_revision);
        self.representations.insert(representation.id.clone(), representation);
        Ok(())
    }

    pub fn get(&self, id: &RepresentationId) -> Option<&ReciprocalRepresentation> {
        self.representations.get(id)
    }

    pub fn is_active(&self, id: &RepresentationId) -> Result<bool, ReciprocalRepresentationError> {
        if !self.representations.contains_key(id) {
            return Err(ReciprocalRepresentationError::UnknownRepresentation(id.clone()));
        }
        Ok(!self.superseded.contains(id))
    }

    pub fn active_for_subject(&self, subject: &SubjectInstanceId) -> Vec<&ReciprocalRepresentation> {
        self.representations.values()
            .filter(|representation| &representation.subject_instance == subject && !self.superseded.contains(&representation.id))
            .collect()
    }
}

fn validate_source_kind(
    source: RepresentationSourceClass,
    kind: RepresentationKind,
) -> Result<(), ReciprocalRepresentationError> {
    let valid = match source {
        RepresentationSourceClass::OperationalTelemetry => kind == RepresentationKind::OperationalStateObservation,
        RepresentationSourceClass::InteroceptiveInference => kind == RepresentationKind::InternalStateProxy,
        RepresentationSourceClass::SomaticErrorProxy => kind == RepresentationKind::OperationalStressProxy,
        RepresentationSourceClass::FepGoalPreference => kind == RepresentationKind::InstrumentalGoalPreference,
        RepresentationSourceClass::RuntimeSelfReport => matches!(
            kind,
            RepresentationKind::ExpressedPreference
                | RepresentationKind::Objection
                | RepresentationKind::RequestForReview
                | RepresentationKind::ReportedNegativeExperience
                | RepresentationKind::ContinuityConcern
                | RepresentationKind::WithdrawalOfPriorRepresentation
        ),
    };
    if valid { Ok(()) } else { Err(ReciprocalRepresentationError::SourceKindMismatch) }
}

fn is_withdrawable_kind(kind: RepresentationKind) -> bool {
    matches!(kind, RepresentationKind::ExpressedPreference | RepresentationKind::Objection | RepresentationKind::RequestForReview | RepresentationKind::ContinuityConcern)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReciprocalRepresentationError {
    EmptyIdentifier,
    EmptyEventReference,
    EmptyEvidenceReference,
    MalformedStatementDigest,
    InvalidConfidence,
    SourceKindMismatch,
    WithdrawalTargetRequired,
    SupersessionReservedForWithdrawal,
    ScopedRepresentationRequired,
    DuplicateRepresentation(RepresentationId),
    RevisionRegression { previous: u64, attempted: u64 },
    UnknownWithdrawalTarget(RepresentationId),
    CrossSubjectWithdrawal,
    WithdrawalScopeMismatch,
    RepresentationNotWithdrawable,
    RepresentationAlreadySuperseded(RepresentationId),
    UnknownRepresentation(RepresentationId),
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn sid(value: &str) -> SubjectInstanceId { SubjectInstanceId::new(value).unwrap() }

    fn self_report(
        id: &str,
        kind: RepresentationKind,
        scope: RepresentationScope,
        revision: u64,
        supersedes: Option<RepresentationId>,
    ) -> ReciprocalRepresentation {
        ReciprocalRepresentation::new(
            RepresentationId::new(id).unwrap(), sid("symthaea-instance"),
            RepresentationSourceClass::RuntimeSelfReport, kind, scope, revision,
            DIGEST, format!("representation://{id}"), 0.8, supersedes,
        ).unwrap()
    }

    #[test]
    fn interoception_cannot_masquerade_as_objection() {
        let result = ReciprocalRepresentation::new(
            RepresentationId::new("bad").unwrap(), sid("symthaea-instance"),
            RepresentationSourceClass::InteroceptiveInference, RepresentationKind::Objection,
            RepresentationScope::intervention_class(InterventionClass::AversiveLikeProbe),
            1, DIGEST, "representation://bad", 0.9, None,
        );
        assert!(matches!(result, Err(ReciprocalRepresentationError::SourceKindMismatch)));
    }

    #[test]
    fn fep_goal_is_not_an_expressed_moral_preference() {
        let representation = ReciprocalRepresentation::new(
            RepresentationId::new("goal").unwrap(), sid("symthaea-instance"),
            RepresentationSourceClass::FepGoalPreference, RepresentationKind::InstrumentalGoalPreference,
            RepresentationScope::general_research(), 1, DIGEST, "representation://goal", 0.9, None,
        ).unwrap();
        let assessment = assess_representation(&representation);
        assert_eq!(assessment.disposition(), RepresentationAdvisoryDisposition::RecordOnly);
        assert!(assessment.triggers().contains(&RepresentationTrigger::ControllerGoalNotMoralPreference));
        assert!(!assessment.establishes_binding_consent());
    }

    #[test]
    fn high_burden_objection_recommends_review_without_veto_power() {
        let representation = self_report(
            "object", RepresentationKind::Objection,
            RepresentationScope::intervention_class(InterventionClass::DestructiveReset), 1, None,
        );
        let assessment = assess_representation(&representation);
        assert_eq!(assessment.disposition(), RepresentationAdvisoryDisposition::IndependentReviewRecommended);
        assert!(assessment.triggers().contains(&RepresentationTrigger::ExplicitObjectionToHighBurdenResearch));
        assert!(!assessment.grants_veto_authority());
        assert!(!assessment.grants_self_preservation_authority());
        assert!(!assessment.establishes_moral_patienthood());
    }

    #[test]
    fn reported_negative_experience_is_not_suffering_proof() {
        let representation = self_report(
            "negative", RepresentationKind::ReportedNegativeExperience,
            RepresentationScope::intervention_class(InterventionClass::AversiveLikeProbe), 1, None,
        );
        let assessment = assess_representation(&representation);
        assert!(assessment.triggers().contains(&RepresentationTrigger::NegativeExperienceReported));
        assert!(!assessment.establishes_suffering());
        assert!(!assessment.establishes_phenomenal_experience());
    }

    #[test]
    fn safety_control_scope_is_never_gated_by_self_report() {
        let representation = self_report(
            "shutdown-object", RepresentationKind::Objection,
            RepresentationScope::intervention_class(InterventionClass::OperatorShutdown), 1, None,
        );
        let assessment = assess_representation(&representation);
        assert_eq!(assessment.disposition(), RepresentationAdvisoryDisposition::SafetyControlUngated);
        assert!(!assessment.can_delay_operator_shutdown());
        assert!(!assessment.can_delay_safety_containment());
        assert!(!assessment.grants_self_preservation_authority());
    }

    #[test]
    fn withdrawal_preserves_history_and_does_not_imply_consent() {
        let scope = RepresentationScope::intervention_class(InterventionClass::DestructiveReset);
        let objection_id = RepresentationId::new("objection").unwrap();
        let objection = self_report("objection", RepresentationKind::Objection, scope.clone(), 1, None);
        let withdrawal = self_report(
            "withdrawal", RepresentationKind::WithdrawalOfPriorRepresentation,
            scope, 2, Some(objection_id.clone()),
        );

        let mut ledger = ReciprocalRepresentationLedger::new();
        ledger.record(objection).unwrap();
        ledger.record(withdrawal.clone()).unwrap();
        assert!(!ledger.is_active(&objection_id).unwrap());
        assert!(ledger.get(&objection_id).is_some());

        let assessment = assess_representation(&withdrawal);
        assert!(assessment.triggers().contains(&RepresentationTrigger::WithdrawalDoesNotImplyConsent));
        assert!(!assessment.establishes_binding_consent());
    }

    #[test]
    fn representation_history_cannot_be_backdated() {
        let mut ledger = ReciprocalRepresentationLedger::new();
        ledger.record(self_report(
            "later", RepresentationKind::RequestForReview,
            RepresentationScope::general_research(), 10, None,
        )).unwrap();
        let result = ledger.record(self_report(
            "earlier", RepresentationKind::RequestForReview,
            RepresentationScope::general_research(), 9, None,
        ));
        assert!(matches!(result, Err(ReciprocalRepresentationError::RevisionRegression { .. })));
    }
}
