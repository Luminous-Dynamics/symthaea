// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Domain-neutral contracts for evidence-bearing scientific and engineering discovery.
//!
//! This crate intentionally contains **contracts, validation, and deterministic
//! review artifacts only**. It does not run experiments, invoke external solvers,
//! synthesize materials, place orders, or otherwise act on a candidate.
//!
//! Existing domain implementations (for example the scientific-method engine,
//! Spark expected-information-gain planner, Pareto optimizers, and simulation
//! bridge) can adapt to these types without being moved or rewritten first.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

/// Explicit capability boundary repeated on every review certificate.
pub const CAPABILITY_CLASSIFICATION: &str =
    "COMPUTATIONAL DISCOVERY RECORD ONLY -- not an execution, synthesis, procurement, deployment, or operating instruction. Human review is required before any real-world action.";

/// Stable identifier for a candidate under evaluation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CandidateId(pub String);

impl CandidateId {
    pub fn new(id: impl Into<String>) -> Result<Self, DiscoveryError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(DiscoveryError::Invalid("candidate id cannot be empty".into()));
        }
        Ok(Self(id))
    }
}

/// Where a candidate entered the discovery process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateOrigin {
    UserProposed,
    Generated {
        generator: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        version: Option<String>,
    },
    Imported {
        source: String,
    },
}

/// Domain-neutral candidate representation.
///
/// `specification` deliberately uses a sorted map so certificates and digests
/// remain deterministic across independent construction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Candidate {
    pub id: CandidateId,
    pub kind: String,
    #[serde(default)]
    pub specification: BTreeMap<String, String>,
    pub origin: CandidateOrigin,
}

impl Candidate {
    pub fn new(
        id: CandidateId,
        kind: impl Into<String>,
        origin: CandidateOrigin,
    ) -> Result<Self, DiscoveryError> {
        let kind = kind.into();
        if kind.trim().is_empty() {
            return Err(DiscoveryError::Invalid("candidate kind cannot be empty".into()));
        }
        Ok(Self {
            id,
            kind,
            specification: BTreeMap::new(),
            origin,
        })
    }

    pub fn with_spec(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.specification.insert(key.into(), value.into());
        self
    }
}

/// Closed interval for a prediction or observation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Interval {
    pub lower: f64,
    pub upper: f64,
}

impl Interval {
    pub fn new(lower: f64, upper: f64) -> Result<Self, DiscoveryError> {
        if !lower.is_finite() || !upper.is_finite() {
            return Err(DiscoveryError::Invalid(
                "interval bounds must be finite".into(),
            ));
        }
        if lower > upper {
            return Err(DiscoveryError::Invalid(
                "interval lower bound cannot exceed upper bound".into(),
            ));
        }
        Ok(Self { lower, upper })
    }
}

/// Uncertainty decomposition shared by discovery records.
///
/// This mirrors the existing engineering distinction between model/knowledge
/// uncertainty and irreducible variability while remaining solver-agnostic.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UncertaintyEstimate {
    /// Model/knowledge uncertainty, from 0 (known) to 1 (unknown).
    pub epistemic: f64,
    /// Noise/irreducible variability, from 0 (deterministic) to 1 (high noise).
    pub aleatoric: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval: Option<Interval>,
}

impl UncertaintyEstimate {
    pub fn new(epistemic: f64, aleatoric: f64) -> Result<Self, DiscoveryError> {
        let value = Self {
            epistemic,
            aleatoric,
            interval: None,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn certain() -> Self {
        Self {
            epistemic: 0.0,
            aleatoric: 0.0,
            interval: None,
        }
    }

    pub fn with_interval(mut self, interval: Interval) -> Self {
        self.interval = Some(interval);
        self
    }

    pub fn total(&self) -> f64 {
        (self.epistemic + self.aleatoric).clamp(0.0, 1.0)
    }

    pub fn validate(&self) -> Result<(), DiscoveryError> {
        for (name, value) in [
            ("epistemic uncertainty", self.epistemic),
            ("aleatoric uncertainty", self.aleatoric),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(DiscoveryError::Invalid(format!(
                    "{name} must be finite and in [0, 1]"
                )));
            }
        }
        if let Some(interval) = self.interval {
            Interval::new(interval.lower, interval.upper)?;
        }
        Ok(())
    }
}

/// Broad evidence class. The class describes *what happened*, not how much
/// confidence a caller wishes to assign to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    Literature,
    Dataset,
    Heuristic,
    SurrogateModel,
    AnalyticalModel,
    FirstPrinciplesSimulation,
    ExternalSimulation,
    Experiment,
    IndependentReplication,
    DeviceValidation,
    FieldObservation,
}

/// Stable pointer to evidence or an evidence-bearing artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRef {
    pub id: String,
    pub kind: EvidenceKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl EvidenceRef {
    pub fn validate(&self) -> Result<(), DiscoveryError> {
        if self.id.trim().is_empty() {
            return Err(DiscoveryError::Invalid("evidence id cannot be empty".into()));
        }
        if self.uri.as_deref().is_none_or(str::is_empty)
            && self.digest.as_deref().is_none_or(str::is_empty)
        {
            return Err(DiscoveryError::Invalid(format!(
                "evidence {:?} requires a URI or digest",
                self.id
            )));
        }
        Ok(())
    }
}

/// Approximate evidence/model fidelity. The rank is used only to choose among
/// multiple predictions of the *same metric and unit*; it is not a claim that
/// every higher tier is universally more accurate than every lower tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FidelityLevel {
    Heuristic,
    Surrogate,
    Analytical,
    FirstPrinciples,
    ExternalSimulation,
    Experiment,
    IndependentReplication,
    DeviceValidated,
    FieldValidated,
}

impl FidelityLevel {
    pub const fn rank(self) -> u8 {
        match self {
            FidelityLevel::Heuristic => 0,
            FidelityLevel::Surrogate => 1,
            FidelityLevel::Analytical => 2,
            FidelityLevel::FirstPrinciples => 3,
            FidelityLevel::ExternalSimulation => 4,
            FidelityLevel::Experiment => 5,
            FidelityLevel::IndependentReplication => 6,
            FidelityLevel::DeviceValidated => 7,
            FidelityLevel::FieldValidated => 8,
        }
    }
}

/// Provenance of the model or method that produced a prediction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelProvenance {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub implementation_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_digest: Option<String>,
}

impl ModelProvenance {
    pub fn named(name: impl Into<String>) -> Result<Self, DiscoveryError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(DiscoveryError::Invalid("model name cannot be empty".into()));
        }
        Ok(Self {
            name,
            version: None,
            implementation_digest: None,
            input_digest: None,
            output_digest: None,
        })
    }
}

/// One scalar prediction with uncertainty and evidence lineage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Prediction {
    pub metric: String,
    pub value: f64,
    pub unit: String,
    pub uncertainty: UncertaintyEstimate,
    pub fidelity: FidelityLevel,
    pub model: ModelProvenance,
    #[serde(default)]
    pub assumptions: Vec<String>,
    #[serde(default)]
    pub evidence: Vec<EvidenceRef>,
}

impl Prediction {
    pub fn validate(&self) -> Result<(), DiscoveryError> {
        if self.metric.trim().is_empty() || self.unit.trim().is_empty() {
            return Err(DiscoveryError::Invalid(
                "prediction requires a metric and unit".into(),
            ));
        }
        if !self.value.is_finite() {
            return Err(DiscoveryError::Invalid(format!(
                "prediction {:?} must be finite",
                self.metric
            )));
        }
        self.uncertainty.validate()?;
        if self.model.name.trim().is_empty() {
            return Err(DiscoveryError::Invalid("prediction model name cannot be empty".into()));
        }
        for evidence in &self.evidence {
            evidence.validate()?;
        }
        Ok(())
    }
}

/// Optimization direction for an objective.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveDirection {
    Maximize,
    Minimize,
    Target { value: f64, tolerance: f64 },
}

/// An optimization objective. Objectives are kept separate rather than being
/// collapsed into one weighted score, so Pareto methods can operate upstream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Objective {
    pub metric: String,
    pub unit: String,
    pub direction: ObjectiveDirection,
}

/// Hard bound for a candidate metric.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintBound {
    AtLeast(f64),
    AtMost(f64),
    Between { min: f64, max: f64 },
}

impl ConstraintBound {
    fn validate(self) -> Result<(), DiscoveryError> {
        match self {
            ConstraintBound::AtLeast(v) | ConstraintBound::AtMost(v) => {
                if !v.is_finite() {
                    return Err(DiscoveryError::Invalid(
                        "constraint bound must be finite".into(),
                    ));
                }
            }
            ConstraintBound::Between { min, max } => {
                if !min.is_finite() || !max.is_finite() || min > max {
                    return Err(DiscoveryError::Invalid(
                        "constraint range requires finite ordered bounds".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn contains(self, value: f64) -> bool {
        match self {
            ConstraintBound::AtLeast(min) => value >= min,
            ConstraintBound::AtMost(max) => value <= max,
            ConstraintBound::Between { min, max } => value >= min && value <= max,
        }
    }
}

/// Hard feasibility constraint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Constraint {
    pub metric: String,
    pub unit: String,
    pub bound: ConstraintBound,
}

impl Constraint {
    pub fn validate(&self) -> Result<(), DiscoveryError> {
        if self.metric.trim().is_empty() || self.unit.trim().is_empty() {
            return Err(DiscoveryError::Invalid(
                "constraint requires a metric and unit".into(),
            ));
        }
        self.bound.validate()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintStatus {
    Satisfied,
    Violated,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstraintAssessment {
    pub constraint: Constraint,
    pub status: ConstraintStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prediction_value: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prediction_fidelity: Option<FidelityLevel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Feasibility {
    Feasible,
    Infeasible,
    Unknown,
}

/// Candidate evaluation at one point in the discovery loop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Evaluation {
    pub candidate_id: CandidateId,
    #[serde(default)]
    pub objectives: Vec<Objective>,
    #[serde(default)]
    pub constraints: Vec<Constraint>,
    #[serde(default)]
    pub predictions: Vec<Prediction>,
    /// Populated by a Pareto adapter when available. `None` means unranked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pareto_rank: Option<usize>,
}

impl Evaluation {
    pub fn validate(&self) -> Result<(), DiscoveryError> {
        for prediction in &self.predictions {
            prediction.validate()?;
        }
        for objective in &self.objectives {
            if objective.metric.trim().is_empty() || objective.unit.trim().is_empty() {
                return Err(DiscoveryError::Invalid(
                    "objective requires a metric and unit".into(),
                ));
            }
            if let ObjectiveDirection::Target { value, tolerance } = objective.direction {
                if !value.is_finite() || !tolerance.is_finite() || tolerance < 0.0 {
                    return Err(DiscoveryError::Invalid(
                        "target objective requires finite value and non-negative tolerance".into(),
                    ));
                }
            }
        }
        for constraint in &self.constraints {
            constraint.validate()?;
        }
        Ok(())
    }

    /// Evaluate hard constraints using the highest-fidelity prediction for the
    /// matching metric and unit. Ties at the highest fidelity are rejected as
    /// ambiguous rather than silently selecting by insertion order.
    pub fn assess_constraints(&self) -> Result<Vec<ConstraintAssessment>, DiscoveryError> {
        self.validate()?;
        let mut assessments = Vec::with_capacity(self.constraints.len());

        for constraint in &self.constraints {
            let same_metric: Vec<&Prediction> = self
                .predictions
                .iter()
                .filter(|prediction| prediction.metric == constraint.metric)
                .collect();

            if same_metric.is_empty() {
                assessments.push(ConstraintAssessment {
                    constraint: constraint.clone(),
                    status: ConstraintStatus::Unknown,
                    prediction_value: None,
                    prediction_fidelity: None,
                });
                continue;
            }

            let matching_unit: Vec<&Prediction> = same_metric
                .iter()
                .copied()
                .filter(|prediction| prediction.unit == constraint.unit)
                .collect();

            if matching_unit.is_empty() {
                let found: BTreeSet<String> = same_metric
                    .iter()
                    .map(|prediction| prediction.unit.clone())
                    .collect();
                return Err(DiscoveryError::UnitMismatch {
                    metric: constraint.metric.clone(),
                    expected: constraint.unit.clone(),
                    found: found.into_iter().collect(),
                });
            }

            let max_rank = matching_unit
                .iter()
                .map(|prediction| prediction.fidelity.rank())
                .max()
                .expect("matching_unit is non-empty");
            let highest: Vec<&Prediction> = matching_unit
                .into_iter()
                .filter(|prediction| prediction.fidelity.rank() == max_rank)
                .collect();

            if highest.len() != 1 {
                return Err(DiscoveryError::AmbiguousPrediction {
                    metric: constraint.metric.clone(),
                    unit: constraint.unit.clone(),
                    fidelity: highest[0].fidelity,
                    count: highest.len(),
                });
            }

            let prediction = highest[0];
            assessments.push(ConstraintAssessment {
                constraint: constraint.clone(),
                status: if constraint.bound.contains(prediction.value) {
                    ConstraintStatus::Satisfied
                } else {
                    ConstraintStatus::Violated
                },
                prediction_value: Some(prediction.value),
                prediction_fidelity: Some(prediction.fidelity),
            });
        }

        Ok(assessments)
    }

    pub fn feasibility(&self) -> Result<Feasibility, DiscoveryError> {
        let assessments = self.assess_constraints()?;
        if assessments
            .iter()
            .any(|assessment| assessment.status == ConstraintStatus::Violated)
        {
            return Ok(Feasibility::Infeasible);
        }
        if assessments
            .iter()
            .any(|assessment| assessment.status == ConstraintStatus::Unknown)
        {
            return Ok(Feasibility::Unknown);
        }
        Ok(Feasibility::Feasible)
    }
}

/// Generic resource estimate, e.g. USD, GPU-hours, samples, or staff-hours.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceEstimate {
    pub amount: f64,
    pub unit: String,
}

impl ResourceEstimate {
    pub fn validate(&self) -> Result<(), DiscoveryError> {
        if !self.amount.is_finite() || self.amount < 0.0 || self.unit.trim().is_empty() {
            return Err(DiscoveryError::Invalid(
                "resource estimate requires a finite non-negative amount and unit".into(),
            ));
        }
        Ok(())
    }
}

/// A proposed next observation, simulation, or experiment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentProposal {
    pub id: String,
    pub candidate_id: CandidateId,
    pub question: String,
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_information_gain_bits: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_resource: Option<ResourceEstimate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_duration_seconds: Option<f64>,
    #[serde(default)]
    pub intended_evidence: Vec<EvidenceKind>,
}

impl ExperimentProposal {
    pub fn validate(&self) -> Result<(), DiscoveryError> {
        if self.id.trim().is_empty() || self.question.trim().is_empty() || self.method.trim().is_empty() {
            return Err(DiscoveryError::Invalid(
                "experiment proposal requires id, question, and method".into(),
            ));
        }
        if let Some(bits) = self.expected_information_gain_bits {
            if !bits.is_finite() || bits < 0.0 {
                return Err(DiscoveryError::Invalid(
                    "expected information gain must be finite and non-negative".into(),
                ));
            }
        }
        if let Some(resource) = &self.estimated_resource {
            resource.validate()?;
        }
        if let Some(seconds) = self.estimated_duration_seconds {
            if !seconds.is_finite() || seconds < 0.0 {
                return Err(DiscoveryError::Invalid(
                    "estimated duration must be finite and non-negative".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Deterministic, human-reviewable output of a discovery run.
///
/// No timestamp or random field is generated internally. If a caller needs
/// time metadata it should live in an external evidence artifact referenced by
/// `evidence_lineage`, preserving digest stability of identical logical runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscoveryCertificate {
    pub schema_version: u16,
    pub capability_classification: String,
    pub run_id: String,
    pub candidate: Candidate,
    #[serde(default)]
    pub evaluations: Vec<Evaluation>,
    #[serde(default)]
    pub evidence_lineage: Vec<EvidenceRef>,
    #[serde(default)]
    pub unresolved_questions: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recommended_next_test: Option<ExperimentProposal>,
}

impl DiscoveryCertificate {
    pub fn new(run_id: impl Into<String>, candidate: Candidate) -> Result<Self, DiscoveryError> {
        let run_id = run_id.into();
        if run_id.trim().is_empty() {
            return Err(DiscoveryError::Invalid("run id cannot be empty".into()));
        }
        Ok(Self {
            schema_version: 1,
            capability_classification: CAPABILITY_CLASSIFICATION.to_string(),
            run_id,
            candidate,
            evaluations: Vec::new(),
            evidence_lineage: Vec::new(),
            unresolved_questions: Vec::new(),
            recommended_next_test: None,
        })
    }

    pub fn validate(&self) -> Result<(), DiscoveryError> {
        if self.schema_version != 1 {
            return Err(DiscoveryError::Invalid(format!(
                "unsupported discovery certificate schema version {}",
                self.schema_version
            )));
        }
        if self.capability_classification != CAPABILITY_CLASSIFICATION {
            return Err(DiscoveryError::Invalid(
                "certificate capability classification was altered".into(),
            ));
        }
        if self.run_id.trim().is_empty() {
            return Err(DiscoveryError::Invalid("run id cannot be empty".into()));
        }
        for evaluation in &self.evaluations {
            if evaluation.candidate_id != self.candidate.id {
                return Err(DiscoveryError::CandidateMismatch {
                    expected: self.candidate.id.0.clone(),
                    found: evaluation.candidate_id.0.clone(),
                });
            }
            evaluation.validate()?;
        }
        for evidence in &self.evidence_lineage {
            evidence.validate()?;
        }
        if self
            .unresolved_questions
            .iter()
            .any(|question| question.trim().is_empty())
        {
            return Err(DiscoveryError::Invalid(
                "unresolved questions cannot contain empty entries".into(),
            ));
        }
        if let Some(proposal) = &self.recommended_next_test {
            if proposal.candidate_id != self.candidate.id {
                return Err(DiscoveryError::CandidateMismatch {
                    expected: self.candidate.id.0.clone(),
                    found: proposal.candidate_id.0.clone(),
                });
            }
            proposal.validate()?;
        }
        Ok(())
    }

    pub fn to_json_pretty(&self) -> Result<String, DiscoveryError> {
        self.validate()?;
        serde_json::to_string_pretty(self).map_err(DiscoveryError::Serialization)
    }

    /// Stable BLAKE3 digest of the compact JSON representation.
    pub fn digest_blake3(&self) -> Result<String, DiscoveryError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(DiscoveryError::Serialization)?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }
}

/// Generates candidates without prescribing how generation is implemented.
pub trait CandidateGenerator: Send + Sync {
    fn name(&self) -> &'static str;
    fn generate(&self, budget: usize) -> Result<Vec<Candidate>, DiscoveryError>;
}

/// Evaluates a candidate using one or more domain-specific models.
pub trait CandidateEvaluator: Send + Sync {
    fn name(&self) -> &'static str;
    fn evaluate(&self, candidate: &Candidate) -> Result<Evaluation, DiscoveryError>;
}

/// Selects the next information-gathering action. Existing expected-information-
/// gain planners can implement this trait without changing their internals.
pub trait ExperimentSelector: Send + Sync {
    fn name(&self) -> &'static str;
    fn select_next(
        &self,
        candidate: &Candidate,
        evaluations: &[Evaluation],
    ) -> Result<Option<ExperimentProposal>, DiscoveryError>;
}

#[derive(Debug, Error)]
pub enum DiscoveryError {
    #[error("invalid discovery record: {0}")]
    Invalid(String),
    #[error("unit mismatch for metric {metric:?}: expected {expected:?}, found {found:?}")]
    UnitMismatch {
        metric: String,
        expected: String,
        found: Vec<String>,
    },
    #[error(
        "ambiguous prediction for metric {metric:?} in unit {unit:?}: {count} values at fidelity {fidelity:?}"
    )]
    AmbiguousPrediction {
        metric: String,
        unit: String,
        fidelity: FidelityLevel,
        count: usize,
    },
    #[error("candidate mismatch: expected {expected:?}, found {found:?}")]
    CandidateMismatch { expected: String, found: String },
    #[error("failed to serialize discovery certificate: {0}")]
    Serialization(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(name: &str) -> ModelProvenance {
        ModelProvenance::named(name).unwrap()
    }

    fn candidate() -> Candidate {
        Candidate::new(
            CandidateId::new("candidate-001").unwrap(),
            "toy_material",
            CandidateOrigin::Generated {
                generator: "unit-test".into(),
                version: Some("1".into()),
            },
        )
        .unwrap()
        .with_spec("formula", "AB2")
        .with_spec("phase", "alpha")
    }

    fn prediction(value: f64, fidelity: FidelityLevel) -> Prediction {
        Prediction {
            metric: "efficiency".into(),
            value,
            unit: "fraction".into(),
            uncertainty: UncertaintyEstimate::new(0.1, 0.02).unwrap(),
            fidelity,
            model: model("toy-model"),
            assumptions: vec!["steady state".into()],
            evidence: vec![],
        }
    }

    #[test]
    fn uncertainty_rejects_invalid_values() {
        assert!(UncertaintyEstimate::new(-0.1, 0.0).is_err());
        assert!(UncertaintyEstimate::new(0.1, 1.1).is_err());
        assert!(Interval::new(2.0, 1.0).is_err());
    }

    #[test]
    fn constraint_uses_unique_highest_fidelity_prediction() {
        let c = candidate();
        let evaluation = Evaluation {
            candidate_id: c.id.clone(),
            objectives: vec![],
            constraints: vec![Constraint {
                metric: "efficiency".into(),
                unit: "fraction".into(),
                bound: ConstraintBound::AtLeast(0.80),
            }],
            predictions: vec![
                prediction(0.75, FidelityLevel::Surrogate),
                prediction(0.86, FidelityLevel::ExternalSimulation),
            ],
            pareto_rank: None,
        };
        let assessed = evaluation.assess_constraints().unwrap();
        assert_eq!(assessed[0].status, ConstraintStatus::Satisfied);
        assert_eq!(assessed[0].prediction_value, Some(0.86));
        assert_eq!(evaluation.feasibility().unwrap(), Feasibility::Feasible);
    }

    #[test]
    fn missing_metric_keeps_feasibility_unknown() {
        let c = candidate();
        let evaluation = Evaluation {
            candidate_id: c.id,
            objectives: vec![],
            constraints: vec![Constraint {
                metric: "cycle_life".into(),
                unit: "cycles".into(),
                bound: ConstraintBound::AtLeast(1000.0),
            }],
            predictions: vec![prediction(0.9, FidelityLevel::Analytical)],
            pareto_rank: None,
        };
        assert_eq!(evaluation.feasibility().unwrap(), Feasibility::Unknown);
    }

    #[test]
    fn equal_fidelity_duplicates_are_not_silently_selected() {
        let c = candidate();
        let evaluation = Evaluation {
            candidate_id: c.id,
            objectives: vec![],
            constraints: vec![Constraint {
                metric: "efficiency".into(),
                unit: "fraction".into(),
                bound: ConstraintBound::AtLeast(0.8),
            }],
            predictions: vec![
                prediction(0.82, FidelityLevel::Surrogate),
                prediction(0.84, FidelityLevel::Surrogate),
            ],
            pareto_rank: None,
        };
        assert!(matches!(
            evaluation.assess_constraints(),
            Err(DiscoveryError::AmbiguousPrediction { .. })
        ));
    }

    #[test]
    fn certificate_rejects_cross_candidate_evidence() {
        let c = candidate();
        let mut cert = DiscoveryCertificate::new("run-1", c).unwrap();
        cert.evaluations.push(Evaluation {
            candidate_id: CandidateId::new("different-candidate").unwrap(),
            objectives: vec![],
            constraints: vec![],
            predictions: vec![],
            pareto_rank: None,
        });
        assert!(matches!(
            cert.validate(),
            Err(DiscoveryError::CandidateMismatch { .. })
        ));
    }

    #[test]
    fn certificate_json_and_digest_are_deterministic() {
        let build = || {
            let c = candidate();
            let mut cert = DiscoveryCertificate::new("run-deterministic", c.clone()).unwrap();
            cert.evaluations.push(Evaluation {
                candidate_id: c.id.clone(),
                objectives: vec![Objective {
                    metric: "efficiency".into(),
                    unit: "fraction".into(),
                    direction: ObjectiveDirection::Maximize,
                }],
                constraints: vec![],
                predictions: vec![prediction(0.86, FidelityLevel::ExternalSimulation)],
                pareto_rank: Some(0),
            });
            cert.unresolved_questions
                .push("does the result replicate experimentally?".into());
            cert.recommended_next_test = Some(ExperimentProposal {
                id: "exp-1".into(),
                candidate_id: c.id,
                question: "Does measured efficiency agree with the simulation?".into(),
                method: "independent bench measurement".into(),
                expected_information_gain_bits: Some(0.7),
                estimated_resource: Some(ResourceEstimate {
                    amount: 2.0,
                    unit: "sample".into(),
                }),
                estimated_duration_seconds: Some(3600.0),
                intended_evidence: vec![EvidenceKind::Experiment],
            });
            cert
        };

        let a = build();
        let b = build();
        assert_eq!(a.to_json_pretty().unwrap(), b.to_json_pretty().unwrap());
        assert_eq!(a.digest_blake3().unwrap(), b.digest_blake3().unwrap());
        assert!(
            a.to_json_pretty()
                .unwrap()
                .contains("COMPUTATIONAL DISCOVERY RECORD ONLY")
        );
    }
}
