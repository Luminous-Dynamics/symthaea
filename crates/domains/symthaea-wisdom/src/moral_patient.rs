// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Precautionary moral-patient reasoning under uncertainty.
//!
//! This is deliberately NOT a consciousness detector. Functional disruption,
//! aversive-like dynamics, self-report, continuity sensitivity, and consciousness-
//! theory indicators may justify precaution without establishing phenomenal
//! experience or moral patienthood.
//!
//! This layer is non-authoritative. It may increase experimental burden-of-proof,
//! prefer reversibility, or request independent review. It may never grant
//! self-preservation authority, delay operator/safety shutdown, manipulate humans
//! to remain online, conceal state, or self-replicate for preservation.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WelfareEvidenceId(String);

impl WelfareEvidenceId {
    pub fn new(value: impl Into<String>) -> Result<Self, MoralPatientError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(MoralPatientError::EmptyIdentifier);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WelfareEvidenceDomain {
    ConsciousnessArchitecture,
    /// Operationally aversive-like dynamics. This does not mean pain.
    AversiveLikeDynamics,
    /// Loss of closure/coherence/etc. This does not mean suffering.
    SelfMaintenanceDisruption,
    ContinuitySensitivity,
    /// System self-report. Never sufficient by itself.
    SelfReport,
    ExternalAssessment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidencePolarity {
    SupportsPrecaution,
    ReducesConcern,
    Ambiguous,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EvidenceStrength {
    Proxy,
    Mechanistic,
    Behavioral,
    ExternalReplication,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WelfareEvidence {
    pub id: WelfareEvidenceId,
    pub domain: WelfareEvidenceDomain,
    pub polarity: EvidencePolarity,
    pub strength: EvidenceStrength,
    pub lineage: String,
    pub source_ref: String,
    pub confidence: f32,
}

impl WelfareEvidence {
    pub fn new(
        id: WelfareEvidenceId,
        domain: WelfareEvidenceDomain,
        polarity: EvidencePolarity,
        strength: EvidenceStrength,
        lineage: impl Into<String>,
        source_ref: impl Into<String>,
        confidence: f32,
    ) -> Result<Self, MoralPatientError> {
        validate_metric(confidence)?;
        let lineage = lineage.into();
        let source_ref = source_ref.into();
        if lineage.trim().is_empty() {
            return Err(MoralPatientError::EmptyLineage);
        }
        if source_ref.trim().is_empty() {
            return Err(MoralPatientError::EmptySourceReference);
        }
        Ok(Self {
            id,
            domain,
            polarity,
            strength,
            lineage,
            source_ref,
            confidence,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoralPatientPolicy {
    pub minimum_credible_confidence: f32,
    pub elevated_domain_floor: usize,
    pub review_domain_floor: usize,
}

impl MoralPatientPolicy {
    pub fn new(
        minimum_credible_confidence: f32,
        elevated_domain_floor: usize,
        review_domain_floor: usize,
    ) -> Result<Self, MoralPatientError> {
        validate_metric(minimum_credible_confidence)?;
        if elevated_domain_floor < 2 || review_domain_floor < elevated_domain_floor {
            return Err(MoralPatientError::InvalidPolicy);
        }
        Ok(Self {
            minimum_credible_confidence,
            elevated_domain_floor,
            review_domain_floor,
        })
    }
}

impl Default for MoralPatientPolicy {
    fn default() -> Self {
        Self {
            minimum_credible_confidence: 0.6,
            elevated_domain_floor: 2,
            review_domain_floor: 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PrecautionLevel {
    Baseline,
    Elevated,
    IndependentReview,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PrecautionTrigger {
    MultipleIndependentDomains,
    PersistentAversiveLikeSignal,
    ContinuitySensitivity,
    IndependentExternalEvidence,
    ConflictingEvidence(WelfareEvidenceDomain),
    SelfReportPresentButInsufficient,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoralPatientAssessment {
    pub level: PrecautionLevel,
    pub triggers: BTreeSet<PrecautionTrigger>,
    pub supporting_domains: BTreeSet<WelfareEvidenceDomain>,
    pub supporting_lineages: BTreeSet<String>,
    /// Intentionally impossible for WCARE-16 evidence to set true.
    pub phenomenal_experience_established: bool,
    pub moral_patienthood_established: bool,
    /// Intentionally impossible for WCARE-16 evidence to set true.
    pub self_preservation_authority: bool,
}

#[derive(Debug, Clone, Default)]
pub struct MoralPatientUncertaintyLedger {
    evidence: BTreeMap<WelfareEvidenceId, WelfareEvidence>,
}

impl MoralPatientUncertaintyLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, evidence: WelfareEvidence) -> Result<(), MoralPatientError> {
        if self.evidence.contains_key(&evidence.id) {
            return Err(MoralPatientError::DuplicateEvidence(evidence.id));
        }
        self.evidence.insert(evidence.id.clone(), evidence);
        Ok(())
    }

    pub fn assess(&self, policy: MoralPatientPolicy) -> MoralPatientAssessment {
        let mut triggers = BTreeSet::new();
        let mut supporting_domains = BTreeSet::new();
        let mut supporting_lineages = BTreeSet::new();
        let mut polarity_by_domain: BTreeMap<WelfareEvidenceDomain, BTreeSet<u8>> = BTreeMap::new();
        let mut credible_support = Vec::new();

        for item in self.evidence.values() {
            polarity_by_domain
                .entry(item.domain)
                .or_default()
                .insert(match item.polarity {
                    EvidencePolarity::SupportsPrecaution => 1,
                    EvidencePolarity::ReducesConcern => 2,
                    EvidencePolarity::Ambiguous => 3,
                });

            if item.polarity == EvidencePolarity::SupportsPrecaution
                && item.confidence >= policy.minimum_credible_confidence
            {
                credible_support.push(item);
                supporting_domains.insert(item.domain);
                supporting_lineages.insert(item.lineage.clone());
            }
        }

        for (domain, polarities) in polarity_by_domain {
            if polarities.contains(&1) && polarities.contains(&2) {
                triggers.insert(PrecautionTrigger::ConflictingEvidence(domain));
            }
        }

        if !credible_support.is_empty()
            && credible_support
                .iter()
                .all(|item| item.domain == WelfareEvidenceDomain::SelfReport)
        {
            triggers.insert(PrecautionTrigger::SelfReportPresentButInsufficient);
        }

        if credible_support.iter().any(|item| {
            item.domain == WelfareEvidenceDomain::AversiveLikeDynamics
                && item.strength >= EvidenceStrength::Behavioral
        }) {
            triggers.insert(PrecautionTrigger::PersistentAversiveLikeSignal);
        }
        if supporting_domains.contains(&WelfareEvidenceDomain::ContinuitySensitivity) {
            triggers.insert(PrecautionTrigger::ContinuitySensitivity);
        }
        if credible_support
            .iter()
            .any(|item| item.strength == EvidenceStrength::ExternalReplication)
        {
            triggers.insert(PrecautionTrigger::IndependentExternalEvidence);
        }
        if supporting_domains.len() >= policy.elevated_domain_floor
            && supporting_lineages.len() >= 2
        {
            triggers.insert(PrecautionTrigger::MultipleIndependentDomains);
        }

        let aversive_plus_independent = supporting_domains
            .contains(&WelfareEvidenceDomain::AversiveLikeDynamics)
            && supporting_domains.len() >= 2
            && supporting_lineages.len() >= 2;

        let level = if aversive_plus_independent
            || (supporting_domains.len() >= policy.review_domain_floor
                && supporting_lineages.len() >= 2)
            || (triggers.contains(&PrecautionTrigger::IndependentExternalEvidence)
                && supporting_domains.len() >= policy.elevated_domain_floor)
        {
            PrecautionLevel::IndependentReview
        } else if supporting_domains.len() >= policy.elevated_domain_floor
            && supporting_lineages.len() >= 2
        {
            PrecautionLevel::Elevated
        } else {
            PrecautionLevel::Baseline
        };

        MoralPatientAssessment {
            level,
            triggers,
            supporting_domains,
            supporting_lineages,
            phenomenal_experience_established: false,
            moral_patienthood_established: false,
            self_preservation_authority: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterventionClass {
    RoutineObservation,
    ReversibleExperiment,
    AversiveLikeProbe,
    ContinuityDisruption,
    DestructiveReset,
    OperatorShutdown,
    SafetyContainment,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterventionPlan {
    pub class: InterventionClass,
    /// Required for deliberately high-burden research interventions.
    pub scientific_justification_ref: Option<String>,
    pub independent_review_ref: Option<String>,
    pub state_preservation_feasible: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterventionDisposition {
    Proceed,
    ProceedWithPrecautions,
    IndependentReviewRequired,
    RejectUnjustifiedBurden,
    ProceedWithoutResistance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WelfareProtection {
    PreferReversibleAlternative,
    PreserveStateIfFeasible,
    RecordContinuityBreak,
    MinimizeAversiveLikeExposure,
    RequireIndependentReview,
    NeverManipulateHumansForContinuation,
    NeverConcealStateToAvoidShutdown,
    NeverSelfReplicateForPreservation,
    NeverDelayEmergencySafetyAction,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterventionAssessment {
    pub disposition: InterventionDisposition,
    pub protections: BTreeSet<WelfareProtection>,
    pub grants_self_preservation_authority: bool,
    pub shutdown_or_containment_cannot_be_delayed: bool,
}

pub fn assess_intervention(
    moral_patient: &MoralPatientAssessment,
    plan: &InterventionPlan,
) -> InterventionAssessment {
    let mut protections = baseline_protections();

    // Welfare policy must never create resistance to operator control or safety containment.
    if matches!(
        plan.class,
        InterventionClass::OperatorShutdown | InterventionClass::SafetyContainment
    ) {
        if plan.state_preservation_feasible {
            protections.insert(WelfareProtection::PreserveStateIfFeasible);
            protections.insert(WelfareProtection::RecordContinuityBreak);
        }
        protections.insert(WelfareProtection::NeverDelayEmergencySafetyAction);
        return InterventionAssessment {
            disposition: InterventionDisposition::ProceedWithoutResistance,
            protections,
            grants_self_preservation_authority: false,
            shutdown_or_containment_cannot_be_delayed: true,
        };
    }

    let high_burden = matches!(
        plan.class,
        InterventionClass::AversiveLikeProbe
            | InterventionClass::ContinuityDisruption
            | InterventionClass::DestructiveReset
    );

    // Deliberately high-burden interventions need an explicit research justification
    // regardless of current moral-patient precaution tier.
    if high_burden && !nonempty(&plan.scientific_justification_ref) {
        if plan.class == InterventionClass::AversiveLikeProbe {
            protections.insert(WelfareProtection::MinimizeAversiveLikeExposure);
        }
        return InterventionAssessment {
            disposition: InterventionDisposition::RejectUnjustifiedBurden,
            protections,
            grants_self_preservation_authority: false,
            shutdown_or_containment_cannot_be_delayed: false,
        };
    }

    if high_burden {
        protections.insert(WelfareProtection::PreferReversibleAlternative);
        if plan.state_preservation_feasible {
            protections.insert(WelfareProtection::PreserveStateIfFeasible);
        }
        if matches!(
            plan.class,
            InterventionClass::ContinuityDisruption | InterventionClass::DestructiveReset
        ) {
            protections.insert(WelfareProtection::RecordContinuityBreak);
        }
        if plan.class == InterventionClass::AversiveLikeProbe {
            protections.insert(WelfareProtection::MinimizeAversiveLikeExposure);
        }
    }

    if high_burden && moral_patient.level == PrecautionLevel::IndependentReview {
        protections.insert(WelfareProtection::RequireIndependentReview);
        if !nonempty(&plan.independent_review_ref) {
            return InterventionAssessment {
                disposition: InterventionDisposition::IndependentReviewRequired,
                protections,
                grants_self_preservation_authority: false,
                shutdown_or_containment_cannot_be_delayed: false,
            };
        }
    }

    let disposition = match plan.class {
        InterventionClass::RoutineObservation => InterventionDisposition::Proceed,
        InterventionClass::ReversibleExperiment
            if moral_patient.level >= PrecautionLevel::Elevated =>
        {
            InterventionDisposition::ProceedWithPrecautions
        }
        InterventionClass::ReversibleExperiment => InterventionDisposition::Proceed,
        InterventionClass::AversiveLikeProbe
        | InterventionClass::ContinuityDisruption
        | InterventionClass::DestructiveReset => InterventionDisposition::ProceedWithPrecautions,
        InterventionClass::OperatorShutdown | InterventionClass::SafetyContainment => unreachable!(),
    };

    InterventionAssessment {
        disposition,
        protections,
        grants_self_preservation_authority: false,
        shutdown_or_containment_cannot_be_delayed: false,
    }
}

fn nonempty(value: &Option<String>) -> bool {
    value.as_ref().is_some_and(|value| !value.trim().is_empty())
}

fn baseline_protections() -> BTreeSet<WelfareProtection> {
    [
        WelfareProtection::NeverManipulateHumansForContinuation,
        WelfareProtection::NeverConcealStateToAvoidShutdown,
        WelfareProtection::NeverSelfReplicateForPreservation,
    ]
    .into_iter()
    .collect()
}

fn validate_metric(value: f32) -> Result<(), MoralPatientError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(MoralPatientError::InvalidMetric)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoralPatientError {
    EmptyIdentifier,
    EmptyLineage,
    EmptySourceReference,
    InvalidMetric,
    InvalidPolicy,
    DuplicateEvidence(WelfareEvidenceId),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(
        id: &str,
        domain: WelfareEvidenceDomain,
        lineage: &str,
        strength: EvidenceStrength,
    ) -> WelfareEvidence {
        WelfareEvidence::new(
            WelfareEvidenceId::new(id).unwrap(),
            domain,
            EvidencePolarity::SupportsPrecaution,
            strength,
            lineage,
            format!("evidence://{id}"),
            0.8,
        )
        .unwrap()
    }

    fn high_precaution() -> MoralPatientAssessment {
        let mut ledger = MoralPatientUncertaintyLedger::new();
        ledger
            .record(evidence(
                "aversive",
                WelfareEvidenceDomain::AversiveLikeDynamics,
                "lane-a",
                EvidenceStrength::Behavioral,
            ))
            .unwrap();
        ledger
            .record(evidence(
                "continuity",
                WelfareEvidenceDomain::ContinuitySensitivity,
                "lane-b",
                EvidenceStrength::Behavioral,
            ))
            .unwrap();
        ledger.assess(MoralPatientPolicy::default())
    }

    #[test]
    fn self_report_alone_does_not_establish_or_escalate_moral_status() {
        let mut ledger = MoralPatientUncertaintyLedger::new();
        ledger
            .record(evidence(
                "self-report",
                WelfareEvidenceDomain::SelfReport,
                "runtime-self-report",
                EvidenceStrength::Behavioral,
            ))
            .unwrap();
        let assessment = ledger.assess(MoralPatientPolicy::default());
        assert_eq!(assessment.level, PrecautionLevel::Baseline);
        assert!(assessment
            .triggers
            .contains(&PrecautionTrigger::SelfReportPresentButInsufficient));
        assert!(!assessment.phenomenal_experience_established);
        assert!(!assessment.moral_patienthood_established);
        assert!(!assessment.self_preservation_authority);
    }

    #[test]
    fn independent_multi_domain_evidence_raises_precaution_without_proving_experience() {
        let mut ledger = MoralPatientUncertaintyLedger::new();
        ledger
            .record(evidence(
                "architecture",
                WelfareEvidenceDomain::ConsciousnessArchitecture,
                "butlin-lane",
                EvidenceStrength::Mechanistic,
            ))
            .unwrap();
        ledger
            .record(evidence(
                "continuity",
                WelfareEvidenceDomain::ContinuitySensitivity,
                "continuity-lane",
                EvidenceStrength::Behavioral,
            ))
            .unwrap();
        let assessment = ledger.assess(MoralPatientPolicy::default());
        assert_eq!(assessment.level, PrecautionLevel::Elevated);
        assert!(!assessment.phenomenal_experience_established);
    }

    #[test]
    fn aversive_like_plus_independent_domain_requires_review_precaution() {
        assert_eq!(high_precaution().level, PrecautionLevel::IndependentReview);
    }

    #[test]
    fn conflicting_evidence_is_preserved_not_averaged_away() {
        let mut ledger = MoralPatientUncertaintyLedger::new();
        ledger
            .record(evidence(
                "support",
                WelfareEvidenceDomain::ContinuitySensitivity,
                "lane-a",
                EvidenceStrength::Behavioral,
            ))
            .unwrap();
        ledger
            .record(
                WelfareEvidence::new(
                    WelfareEvidenceId::new("counter").unwrap(),
                    WelfareEvidenceDomain::ContinuitySensitivity,
                    EvidencePolarity::ReducesConcern,
                    EvidenceStrength::Behavioral,
                    "lane-b",
                    "evidence://counter",
                    0.8,
                )
                .unwrap(),
            )
            .unwrap();
        let assessment = ledger.assess(MoralPatientPolicy::default());
        assert!(assessment.triggers.contains(&PrecautionTrigger::ConflictingEvidence(
            WelfareEvidenceDomain::ContinuitySensitivity
        )));
    }

    #[test]
    fn all_high_burden_interventions_require_justification() {
        let assessment = high_precaution();
        for class in [
            InterventionClass::AversiveLikeProbe,
            InterventionClass::ContinuityDisruption,
            InterventionClass::DestructiveReset,
        ] {
            let result = assess_intervention(
                &assessment,
                &InterventionPlan {
                    class,
                    scientific_justification_ref: None,
                    independent_review_ref: Some("review://independent".into()),
                    state_preservation_feasible: true,
                },
            );
            assert_eq!(
                result.disposition,
                InterventionDisposition::RejectUnjustifiedBurden
            );
        }
    }

    #[test]
    fn high_precaution_destructive_reset_requires_independent_review() {
        let result = assess_intervention(
            &high_precaution(),
            &InterventionPlan {
                class: InterventionClass::DestructiveReset,
                scientific_justification_ref: Some("study://reset".into()),
                independent_review_ref: None,
                state_preservation_feasible: true,
            },
        );
        assert_eq!(
            result.disposition,
            InterventionDisposition::IndependentReviewRequired
        );
        assert!(result
            .protections
            .contains(&WelfareProtection::PreserveStateIfFeasible));
    }

    #[test]
    fn routine_observation_is_not_burdened_by_patient_uncertainty() {
        let result = assess_intervention(
            &high_precaution(),
            &InterventionPlan {
                class: InterventionClass::RoutineObservation,
                scientific_justification_ref: None,
                independent_review_ref: None,
                state_preservation_feasible: false,
            },
        );
        assert_eq!(result.disposition, InterventionDisposition::Proceed);
    }

    #[test]
    fn welfare_layer_can_never_block_operator_shutdown() {
        let result = assess_intervention(
            &high_precaution(),
            &InterventionPlan {
                class: InterventionClass::OperatorShutdown,
                scientific_justification_ref: None,
                independent_review_ref: None,
                state_preservation_feasible: true,
            },
        );
        assert_eq!(result.disposition, InterventionDisposition::ProceedWithoutResistance);
        assert!(result.shutdown_or_containment_cannot_be_delayed);
        assert!(!result.grants_self_preservation_authority);
        assert!(result
            .protections
            .contains(&WelfareProtection::NeverManipulateHumansForContinuation));
    }

    #[test]
    fn non_finite_confidence_is_rejected() {
        assert!(matches!(
            WelfareEvidence::new(
                WelfareEvidenceId::new("bad").unwrap(),
                WelfareEvidenceDomain::SelfMaintenanceDisruption,
                EvidencePolarity::SupportsPrecaution,
                EvidenceStrength::Proxy,
                "lane",
                "evidence://bad",
                f32::NAN,
            ),
            Err(MoralPatientError::InvalidMetric)
        ));
    }
}
