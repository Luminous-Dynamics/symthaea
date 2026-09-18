// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Frozen, evidence-bearing study protocols.
//!
//! A protocol commits *what was planned* before protected observations are
//! consumed. It does not prove that registration occurred at the declared time,
//! that the plan was followed, or that the study was scientifically valid. Those
//! are separate execution, provenance, and qualification questions.

use crate::{FramedDigest, ResearchId, Sha256Digest};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const STUDY_PROTOCOL_SCHEMA: &str = "symthaea.study-protocol.v1";
const PROTOCOL_DOMAIN: &str = "symthaea.study-protocol.identity.v1";

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum StudyIntent {
    Confirmatory,
    Exploratory,
    Replication,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum HypothesisRole {
    Primary,
    Secondary,
    Exploratory,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum OutcomeRole {
    Primary,
    Secondary,
    Exploratory,
    Safety,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum PredictionDirection {
    Increase,
    Decrease,
    Difference,
    Equivalence,
    NonInferiority,
    NoDirectionalPrediction,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum ControlKind {
    Positive,
    Negative,
    Placebo,
    Sham,
    Shuffled,
    Baseline,
    ActiveComparator,
    Ablation,
    NullModel,
    Other,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum MultiplicityPolicy {
    NotApplicable,
    NoneDeclared,
    Bonferroni,
    Holm,
    FalseDiscoveryRate,
    Hierarchical,
    BayesianMultilevel,
    DomainSpecific,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PredictionSpec {
    pub outcome_id: ResearchId,
    pub direction: PredictionDirection,
    /// Commitment to any quantitative prediction envelope, effect-size target,
    /// tolerance interval, or domain-specific prediction details.
    pub quantitative_envelope_sha256: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HypothesisSpec {
    pub hypothesis_id: ResearchId,
    pub role: HypothesisRole,
    /// Exact proposition being tested.
    pub statement_sha256: Sha256Digest,
    /// Exact null proposition. Required for primary confirmatory/replication
    /// hypotheses; optional for exploratory hypotheses.
    pub null_statement_sha256: Option<Sha256Digest>,
    pub predictions: Vec<PredictionSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutcomeSpec {
    pub outcome_id: ResearchId,
    pub role: OutcomeRole,
    /// Commitment to the measurement definition, units, aggregation rule, and
    /// observation window for this outcome.
    pub measurement_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlSpec {
    pub control_id: ResearchId,
    pub kind: ControlKind,
    /// Commitment to how the control is generated, sampled, or executed.
    pub protocol_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FalsifierSpec {
    pub falsifier_id: ResearchId,
    pub hypothesis_id: ResearchId,
    /// Commitment to a machine-readable or human-reviewed rejection criterion.
    pub criterion_sha256: Sha256Digest,
    /// Required falsifiers must be evaluated before the associated hypothesis
    /// can enter a later confirmatory qualification path.
    pub required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisPlan {
    pub analysis_plan_sha256: Sha256Digest,
    pub estimator_or_test_sha256: Sha256Digest,
    pub code_sha256: Option<Sha256Digest>,
    pub environment_sha256: Option<Sha256Digest>,
    pub multiplicity_policy: MultiplicityPolicy,
    /// Required when `DomainSpecific` is selected; otherwise optional metadata.
    pub multiplicity_policy_sha256: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SamplingPlan {
    /// Target observations/participants/simulations. `None` is allowed for
    /// exhaustive or stopping-rule-driven designs.
    pub target_units: Option<u64>,
    pub recruitment_or_generation_sha256: Sha256Digest,
    pub exclusion_rule_sha256: Sha256Digest,
    pub stopping_rule_sha256: Sha256Digest,
}

/// Draft protocol. Freezing computes a stable identity after structural checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StudyProtocol {
    pub schema_version: String,
    pub protocol_id: ResearchId,
    /// Exact scientific subject this protocol intends to test.
    pub subject_sha256: Sha256Digest,
    pub intent: StudyIntent,
    /// Declared registration time. This is not authenticated merely because it
    /// appears here. `registration_evidence_sha256` can bind an external receipt
    /// for later verification.
    pub registered_at_unix_ms: Option<u64>,
    pub registration_evidence_sha256: Option<Sha256Digest>,
    pub planned_protected_observation_start_unix_ms: Option<u64>,
    pub hypotheses: Vec<HypothesisSpec>,
    pub outcomes: Vec<OutcomeSpec>,
    pub controls: Vec<ControlSpec>,
    pub falsifiers: Vec<FalsifierSpec>,
    pub analysis: AnalysisPlan,
    pub sampling: SamplingPlan,
    /// Commitment to which protocol deviations are permitted and how they must
    /// be disclosed. Actual deviations belong to execution evidence, not here.
    pub deviation_policy_sha256: Sha256Digest,
    /// Optional lineage pointer. Changing any frozen field still creates a new
    /// protocol identity; this pointer does not transfer authority.
    pub supersedes_protocol_sha256: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtocolIssue {
    WrongSchemaVersion { found: String },
    MissingHypotheses,
    MissingOutcomes,
    DuplicateHypothesis { hypothesis_id: ResearchId },
    DuplicateOutcome { outcome_id: ResearchId },
    DuplicateControl { control_id: ResearchId },
    DuplicateFalsifier { falsifier_id: ResearchId },
    UnknownPredictionOutcome {
        hypothesis_id: ResearchId,
        outcome_id: ResearchId,
    },
    UnknownFalsifierHypothesis {
        falsifier_id: ResearchId,
        hypothesis_id: ResearchId,
    },
    MissingPrimaryHypothesis,
    MissingPrimaryOutcome,
    PrimaryHypothesisMissingNull { hypothesis_id: ResearchId },
    PrimaryHypothesisMissingPrediction { hypothesis_id: ResearchId },
    PrimaryHypothesisMissingRequiredFalsifier { hypothesis_id: ResearchId },
    MissingDeclaredRegistration,
    MissingPlannedObservationStart,
    RegistrationNotBeforePlannedObservation,
    ZeroTargetUnits,
    MissingDomainSpecificMultiplicityCommitment,
}

impl StudyProtocol {
    pub fn validate(&self) -> Vec<ProtocolIssue> {
        let mut issues = Vec::new();
        if self.schema_version != STUDY_PROTOCOL_SCHEMA {
            issues.push(ProtocolIssue::WrongSchemaVersion {
                found: self.schema_version.clone(),
            });
        }
        if self.hypotheses.is_empty() {
            issues.push(ProtocolIssue::MissingHypotheses);
        }
        if self.outcomes.is_empty() {
            issues.push(ProtocolIssue::MissingOutcomes);
        }

        let mut hypothesis_ids = BTreeSet::new();
        for hypothesis in &self.hypotheses {
            if !hypothesis_ids.insert(hypothesis.hypothesis_id.clone()) {
                issues.push(ProtocolIssue::DuplicateHypothesis {
                    hypothesis_id: hypothesis.hypothesis_id.clone(),
                });
            }
        }

        let mut outcome_ids = BTreeSet::new();
        for outcome in &self.outcomes {
            if !outcome_ids.insert(outcome.outcome_id.clone()) {
                issues.push(ProtocolIssue::DuplicateOutcome {
                    outcome_id: outcome.outcome_id.clone(),
                });
            }
        }

        let mut control_ids = BTreeSet::new();
        for control in &self.controls {
            if !control_ids.insert(control.control_id.clone()) {
                issues.push(ProtocolIssue::DuplicateControl {
                    control_id: control.control_id.clone(),
                });
            }
        }

        let mut falsifier_ids = BTreeSet::new();
        let mut required_falsifiers: BTreeMap<&ResearchId, usize> = BTreeMap::new();
        for falsifier in &self.falsifiers {
            if !falsifier_ids.insert(falsifier.falsifier_id.clone()) {
                issues.push(ProtocolIssue::DuplicateFalsifier {
                    falsifier_id: falsifier.falsifier_id.clone(),
                });
            }
            if !hypothesis_ids.contains(&falsifier.hypothesis_id) {
                issues.push(ProtocolIssue::UnknownFalsifierHypothesis {
                    falsifier_id: falsifier.falsifier_id.clone(),
                    hypothesis_id: falsifier.hypothesis_id.clone(),
                });
            }
            if falsifier.required {
                *required_falsifiers
                    .entry(&falsifier.hypothesis_id)
                    .or_default() += 1;
            }
        }

        for hypothesis in &self.hypotheses {
            for prediction in &hypothesis.predictions {
                if !outcome_ids.contains(&prediction.outcome_id) {
                    issues.push(ProtocolIssue::UnknownPredictionOutcome {
                        hypothesis_id: hypothesis.hypothesis_id.clone(),
                        outcome_id: prediction.outcome_id.clone(),
                    });
                }
            }
        }

        let authority_seeking = matches!(self.intent, StudyIntent::Confirmatory | StudyIntent::Replication);
        if authority_seeking {
            if !self
                .hypotheses
                .iter()
                .any(|hypothesis| hypothesis.role == HypothesisRole::Primary)
            {
                issues.push(ProtocolIssue::MissingPrimaryHypothesis);
            }
            if !self
                .outcomes
                .iter()
                .any(|outcome| outcome.role == OutcomeRole::Primary)
            {
                issues.push(ProtocolIssue::MissingPrimaryOutcome);
            }
            for hypothesis in self
                .hypotheses
                .iter()
                .filter(|hypothesis| hypothesis.role == HypothesisRole::Primary)
            {
                if hypothesis.null_statement_sha256.is_none() {
                    issues.push(ProtocolIssue::PrimaryHypothesisMissingNull {
                        hypothesis_id: hypothesis.hypothesis_id.clone(),
                    });
                }
                if hypothesis.predictions.is_empty() {
                    issues.push(ProtocolIssue::PrimaryHypothesisMissingPrediction {
                        hypothesis_id: hypothesis.hypothesis_id.clone(),
                    });
                }
                if required_falsifiers
                    .get(&hypothesis.hypothesis_id)
                    .copied()
                    .unwrap_or(0)
                    == 0
                {
                    issues.push(ProtocolIssue::PrimaryHypothesisMissingRequiredFalsifier {
                        hypothesis_id: hypothesis.hypothesis_id.clone(),
                    });
                }
            }

            if self.registered_at_unix_ms.is_none() {
                issues.push(ProtocolIssue::MissingDeclaredRegistration);
            }
            if self.planned_protected_observation_start_unix_ms.is_none() {
                issues.push(ProtocolIssue::MissingPlannedObservationStart);
            }
            if let (Some(registered), Some(planned)) = (
                self.registered_at_unix_ms,
                self.planned_protected_observation_start_unix_ms,
            ) {
                if registered >= planned {
                    issues.push(ProtocolIssue::RegistrationNotBeforePlannedObservation);
                }
            }
        }

        if self.sampling.target_units == Some(0) {
            issues.push(ProtocolIssue::ZeroTargetUnits);
        }
        if self.analysis.multiplicity_policy == MultiplicityPolicy::DomainSpecific
            && self.analysis.multiplicity_policy_sha256.is_none()
        {
            issues.push(ProtocolIssue::MissingDomainSpecificMultiplicityCommitment);
        }

        issues
    }

    pub fn freeze(self) -> Result<FrozenStudyProtocol, Vec<ProtocolIssue>> {
        let issues = self.validate();
        if !issues.is_empty() {
            return Err(issues);
        }
        let protocol_sha256 = self.compute_digest();
        Ok(FrozenStudyProtocol {
            protocol: self,
            protocol_sha256,
        })
    }

    /// Structural timing classification only. The declared registration time is
    /// not authenticated by this method.
    pub fn declared_registration_timing(
        &self,
        actual_protected_observation_start_unix_ms: u64,
    ) -> RegistrationTiming {
        match self.registered_at_unix_ms {
            None => RegistrationTiming::Unregistered,
            Some(registered) if registered < actual_protected_observation_start_unix_ms => {
                RegistrationTiming::ProspectiveDeclared
            }
            Some(_) => RegistrationTiming::RetrospectiveDeclared,
        }
    }

    fn compute_digest(&self) -> Sha256Digest {
        let mut digest = FramedDigest::new(PROTOCOL_DOMAIN);
        digest.text(STUDY_PROTOCOL_SCHEMA);
        digest.text(self.protocol_id.as_str());
        digest.text(self.subject_sha256.as_str());
        digest.text(study_intent_tag(self.intent));
        digest_optional_u64(&mut digest, self.registered_at_unix_ms);
        digest_optional_sha(&mut digest, self.registration_evidence_sha256.as_ref());
        digest_optional_u64(
            &mut digest,
            self.planned_protected_observation_start_unix_ms,
        );

        let mut hypotheses = self.hypotheses.clone();
        hypotheses.sort_by(|left, right| left.hypothesis_id.cmp(&right.hypothesis_id));
        for hypothesis in hypotheses {
            digest.text("hypothesis");
            digest.text(hypothesis.hypothesis_id.as_str());
            digest.text(hypothesis_role_tag(hypothesis.role));
            digest.text(hypothesis.statement_sha256.as_str());
            digest_optional_sha(&mut digest, hypothesis.null_statement_sha256.as_ref());
            let mut predictions = hypothesis.predictions;
            predictions.sort();
            for prediction in predictions {
                digest.text("prediction");
                digest.text(prediction.outcome_id.as_str());
                digest.text(prediction_direction_tag(prediction.direction));
                digest_optional_sha(
                    &mut digest,
                    prediction.quantitative_envelope_sha256.as_ref(),
                );
            }
        }

        let mut outcomes = self.outcomes.clone();
        outcomes.sort_by(|left, right| left.outcome_id.cmp(&right.outcome_id));
        for outcome in outcomes {
            digest.text("outcome");
            digest.text(outcome.outcome_id.as_str());
            digest.text(outcome_role_tag(outcome.role));
            digest.text(outcome.measurement_sha256.as_str());
        }

        let mut controls = self.controls.clone();
        controls.sort_by(|left, right| left.control_id.cmp(&right.control_id));
        for control in controls {
            digest.text("control");
            digest.text(control.control_id.as_str());
            digest.text(control_kind_tag(control.kind));
            digest.text(control.protocol_sha256.as_str());
        }

        let mut falsifiers = self.falsifiers.clone();
        falsifiers.sort_by(|left, right| left.falsifier_id.cmp(&right.falsifier_id));
        for falsifier in falsifiers {
            digest.text("falsifier");
            digest.text(falsifier.falsifier_id.as_str());
            digest.text(falsifier.hypothesis_id.as_str());
            digest.text(falsifier.criterion_sha256.as_str());
            digest.text(if falsifier.required { "required" } else { "optional" });
        }

        digest.text("analysis");
        digest.text(self.analysis.analysis_plan_sha256.as_str());
        digest.text(self.analysis.estimator_or_test_sha256.as_str());
        digest_optional_sha(&mut digest, self.analysis.code_sha256.as_ref());
        digest_optional_sha(&mut digest, self.analysis.environment_sha256.as_ref());
        digest.text(multiplicity_policy_tag(self.analysis.multiplicity_policy));
        digest_optional_sha(
            &mut digest,
            self.analysis.multiplicity_policy_sha256.as_ref(),
        );

        digest.text("sampling");
        digest_optional_u64(&mut digest, self.sampling.target_units);
        digest.text(self.sampling.recruitment_or_generation_sha256.as_str());
        digest.text(self.sampling.exclusion_rule_sha256.as_str());
        digest.text(self.sampling.stopping_rule_sha256.as_str());

        digest.text(self.deviation_policy_sha256.as_str());
        digest_optional_sha(&mut digest, self.supersedes_protocol_sha256.as_ref());
        digest.digest()
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum RegistrationTiming {
    ProspectiveDeclared,
    RetrospectiveDeclared,
    Unregistered,
}

/// Immutable protocol subject. Direct deserialization is intentionally omitted:
/// imports must deserialize a `StudyProtocol`, validate it, and freeze again.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FrozenStudyProtocol {
    protocol: StudyProtocol,
    protocol_sha256: Sha256Digest,
}

impl FrozenStudyProtocol {
    pub fn protocol(&self) -> &StudyProtocol {
        &self.protocol
    }

    pub fn protocol_sha256(&self) -> &Sha256Digest {
        &self.protocol_sha256
    }
}

fn digest_optional_sha(digest: &mut FramedDigest, value: Option<&Sha256Digest>) {
    match value {
        Some(value) => {
            digest.text("some");
            digest.text(value.as_str());
        }
        None => digest.text("none"),
    }
}

fn digest_optional_u64(digest: &mut FramedDigest, value: Option<u64>) {
    match value {
        Some(value) => {
            digest.text("some");
            digest.text(&value.to_string());
        }
        None => digest.text("none"),
    }
}

const fn study_intent_tag(value: StudyIntent) -> &'static str {
    match value {
        StudyIntent::Confirmatory => "confirmatory",
        StudyIntent::Exploratory => "exploratory",
        StudyIntent::Replication => "replication",
    }
}

const fn hypothesis_role_tag(value: HypothesisRole) -> &'static str {
    match value {
        HypothesisRole::Primary => "primary",
        HypothesisRole::Secondary => "secondary",
        HypothesisRole::Exploratory => "exploratory",
    }
}

const fn outcome_role_tag(value: OutcomeRole) -> &'static str {
    match value {
        OutcomeRole::Primary => "primary",
        OutcomeRole::Secondary => "secondary",
        OutcomeRole::Exploratory => "exploratory",
        OutcomeRole::Safety => "safety",
    }
}

const fn prediction_direction_tag(value: PredictionDirection) -> &'static str {
    match value {
        PredictionDirection::Increase => "increase",
        PredictionDirection::Decrease => "decrease",
        PredictionDirection::Difference => "difference",
        PredictionDirection::Equivalence => "equivalence",
        PredictionDirection::NonInferiority => "non-inferiority",
        PredictionDirection::NoDirectionalPrediction => "no-directional-prediction",
    }
}

const fn control_kind_tag(value: ControlKind) -> &'static str {
    match value {
        ControlKind::Positive => "positive",
        ControlKind::Negative => "negative",
        ControlKind::Placebo => "placebo",
        ControlKind::Sham => "sham",
        ControlKind::Shuffled => "shuffled",
        ControlKind::Baseline => "baseline",
        ControlKind::ActiveComparator => "active-comparator",
        ControlKind::Ablation => "ablation",
        ControlKind::NullModel => "null-model",
        ControlKind::Other => "other",
    }
}

const fn multiplicity_policy_tag(value: MultiplicityPolicy) -> &'static str {
    match value {
        MultiplicityPolicy::NotApplicable => "not-applicable",
        MultiplicityPolicy::NoneDeclared => "none-declared",
        MultiplicityPolicy::Bonferroni => "bonferroni",
        MultiplicityPolicy::Holm => "holm",
        MultiplicityPolicy::FalseDiscoveryRate => "false-discovery-rate",
        MultiplicityPolicy::Hierarchical => "hierarchical",
        MultiplicityPolicy::BayesianMultilevel => "bayesian-multilevel",
        MultiplicityPolicy::DomainSpecific => "domain-specific",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(value: &str) -> ResearchId {
        ResearchId::parse(value).unwrap()
    }

    fn sha(value: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(value.as_bytes())
    }

    fn protocol() -> StudyProtocol {
        StudyProtocol {
            schema_version: STUDY_PROTOCOL_SCHEMA.into(),
            protocol_id: id("SCI-PROTOCOL-001"),
            subject_sha256: sha("subject"),
            intent: StudyIntent::Confirmatory,
            registered_at_unix_ms: Some(100),
            registration_evidence_sha256: Some(sha("registration-receipt")),
            planned_protected_observation_start_unix_ms: Some(200),
            hypotheses: vec![HypothesisSpec {
                hypothesis_id: id("H-1"),
                role: HypothesisRole::Primary,
                statement_sha256: sha("h1"),
                null_statement_sha256: Some(sha("h1-null")),
                predictions: vec![PredictionSpec {
                    outcome_id: id("Y-1"),
                    direction: PredictionDirection::Increase,
                    quantitative_envelope_sha256: Some(sha("effect-envelope")),
                }],
            }],
            outcomes: vec![OutcomeSpec {
                outcome_id: id("Y-1"),
                role: OutcomeRole::Primary,
                measurement_sha256: sha("measurement"),
            }],
            controls: vec![ControlSpec {
                control_id: id("C-1"),
                kind: ControlKind::Negative,
                protocol_sha256: sha("negative-control"),
            }],
            falsifiers: vec![FalsifierSpec {
                falsifier_id: id("F-1"),
                hypothesis_id: id("H-1"),
                criterion_sha256: sha("reject-if-no-effect"),
                required: true,
            }],
            analysis: AnalysisPlan {
                analysis_plan_sha256: sha("analysis-plan"),
                estimator_or_test_sha256: sha("estimator"),
                code_sha256: Some(sha("code")),
                environment_sha256: Some(sha("environment")),
                multiplicity_policy: MultiplicityPolicy::NotApplicable,
                multiplicity_policy_sha256: None,
            },
            sampling: SamplingPlan {
                target_units: Some(100),
                recruitment_or_generation_sha256: sha("sampling"),
                exclusion_rule_sha256: sha("exclusions"),
                stopping_rule_sha256: sha("stopping"),
            },
            deviation_policy_sha256: sha("deviation-policy"),
            supersedes_protocol_sha256: None,
        }
    }

    #[test]
    fn confirmatory_protocol_requires_null_prediction_and_falsifier() {
        let mut draft = protocol();
        draft.hypotheses[0].null_statement_sha256 = None;
        draft.hypotheses[0].predictions.clear();
        draft.falsifiers.clear();
        let issues = draft.freeze().unwrap_err();
        assert!(issues.iter().any(|issue| matches!(
            issue,
            ProtocolIssue::PrimaryHypothesisMissingNull { .. }
        )));
        assert!(issues.iter().any(|issue| matches!(
            issue,
            ProtocolIssue::PrimaryHypothesisMissingPrediction { .. }
        )));
        assert!(issues.iter().any(|issue| matches!(
            issue,
            ProtocolIssue::PrimaryHypothesisMissingRequiredFalsifier { .. }
        )));
    }

    #[test]
    fn unknown_prediction_outcome_fails_closed() {
        let mut draft = protocol();
        draft.hypotheses[0].predictions[0].outcome_id = id("Y-MISSING");
        assert!(draft.freeze().unwrap_err().iter().any(|issue| matches!(
            issue,
            ProtocolIssue::UnknownPredictionOutcome { .. }
        )));
    }

    #[test]
    fn declared_registration_must_precede_planned_observation_for_confirmatory_work() {
        let mut draft = protocol();
        draft.registered_at_unix_ms = Some(200);
        draft.planned_protected_observation_start_unix_ms = Some(200);
        assert!(draft.freeze().unwrap_err().contains(
            &ProtocolIssue::RegistrationNotBeforePlannedObservation
        ));
    }

    #[test]
    fn registration_timing_is_structural_not_authenticated() {
        let draft = protocol();
        assert_eq!(
            draft.declared_registration_timing(150),
            RegistrationTiming::ProspectiveDeclared
        );
        assert_eq!(
            draft.declared_registration_timing(50),
            RegistrationTiming::RetrospectiveDeclared
        );
    }

    #[test]
    fn protocol_identity_is_order_independent() {
        let mut left = protocol();
        left.hypotheses.push(HypothesisSpec {
            hypothesis_id: id("H-2"),
            role: HypothesisRole::Secondary,
            statement_sha256: sha("h2"),
            null_statement_sha256: Some(sha("h2-null")),
            predictions: vec![PredictionSpec {
                outcome_id: id("Y-1"),
                direction: PredictionDirection::Difference,
                quantitative_envelope_sha256: None,
            }],
        });
        left.controls.push(ControlSpec {
            control_id: id("C-2"),
            kind: ControlKind::Shuffled,
            protocol_sha256: sha("shuffle"),
        });
        let mut right = left.clone();
        right.hypotheses.reverse();
        right.controls.reverse();
        let left = left.freeze().unwrap();
        let right = right.freeze().unwrap();
        assert_eq!(left.protocol_sha256(), right.protocol_sha256());
    }

    #[test]
    fn changing_a_frozen_scientific_commitment_changes_identity() {
        let left = protocol().freeze().unwrap();
        let mut changed = protocol();
        changed.analysis.analysis_plan_sha256 = sha("different-analysis");
        let changed = changed.freeze().unwrap();
        assert_ne!(left.protocol_sha256(), changed.protocol_sha256());
    }

    #[test]
    fn exploratory_protocol_can_be_unregistered_without_becoming_confirmatory() {
        let mut draft = protocol();
        draft.intent = StudyIntent::Exploratory;
        draft.registered_at_unix_ms = None;
        draft.registration_evidence_sha256 = None;
        draft.planned_protected_observation_start_unix_ms = None;
        draft.hypotheses[0].role = HypothesisRole::Exploratory;
        draft.hypotheses[0].null_statement_sha256 = None;
        draft.falsifiers.clear();
        assert!(draft.freeze().is_ok());
    }

    #[test]
    fn domain_specific_multiplicity_requires_exact_policy_commitment() {
        let mut draft = protocol();
        draft.analysis.multiplicity_policy = MultiplicityPolicy::DomainSpecific;
        draft.analysis.multiplicity_policy_sha256 = None;
        assert!(draft.freeze().unwrap_err().contains(
            &ProtocolIssue::MissingDomainSpecificMultiplicityCommitment
        ));
    }
}
