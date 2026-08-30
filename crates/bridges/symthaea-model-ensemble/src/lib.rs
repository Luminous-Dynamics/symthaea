//! Evidence-preserving comparison of multiple models for one scenario outcome.
//!
//! This crate deliberately does not average models into a synthetic truth.
//! Each prediction retains model identity, applicability, calibration evidence,
//! assumptions, provenance, and uncertainty. Agreement/disagreement is computed
//! only under an explicit comparison policy supplied by the caller.

use std::collections::HashSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

use symthaea_scenario_outcomes::{OutcomeEstimate, OutcomeSourceRef};

pub type Result<T> = std::result::Result<T, EnsembleError>;

#[derive(Debug, Clone, PartialEq)]
pub enum EnsembleError {
    EmptyField(&'static str),
    NonFinite { field: &'static str, value: f64 },
    InvalidProbability { field: &'static str, value: f64 },
    InvalidCoverage { observed: f64, nominal: f64 },
    InvalidSampleCount,
    MissingPrediction,
    DuplicateModel(String),
    ScenarioMismatch { expected: String, got: String },
    DimensionMismatch { expected: String, got: String },
    UnitMismatch { expected: String, got: String },
    InvalidTolerance(f64),
}

impl Display for EnsembleError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "{field} must not be empty"),
            Self::NonFinite { field, value } => write!(f, "{field} must be finite, got {value}"),
            Self::InvalidProbability { field, value } => {
                write!(f, "{field} must be in [0, 1], got {value}")
            }
            Self::InvalidCoverage { observed, nominal } => write!(
                f,
                "coverage values must be probabilities, got observed={observed}, nominal={nominal}"
            ),
            Self::InvalidSampleCount => write!(f, "calibration sample count must be greater than zero"),
            Self::MissingPrediction => write!(f, "an ensemble requires at least one model prediction"),
            Self::DuplicateModel(id) => write!(f, "duplicate model prediction {id}"),
            Self::ScenarioMismatch { expected, got } => {
                write!(f, "scenario mismatch: expected {expected}, got {got}")
            }
            Self::DimensionMismatch { expected, got } => {
                write!(f, "outcome dimension mismatch: expected {expected}, got {got}")
            }
            Self::UnitMismatch { expected, got } => {
                write!(f, "outcome unit mismatch: expected {expected}, got {got}")
            }
            Self::InvalidTolerance(value) => {
                write!(f, "comparison tolerance must be finite and non-negative, got {value}")
            }
        }
    }
}

impl Error for EnsembleError {}

fn non_empty(value: &str, field: &'static str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(EnsembleError::EmptyField(field));
    }
    Ok(())
}

fn finite(value: f64, field: &'static str) -> Result<()> {
    if !value.is_finite() {
        return Err(EnsembleError::NonFinite { field, value });
    }
    Ok(())
}

fn probability(value: f64, field: &'static str) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(EnsembleError::InvalidProbability { field, value });
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Physical,
    Causal,
    Statistical,
    Learned,
    DigitalTwin,
    Symtropy,
    ExpertRule,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnsembleModelRef {
    pub id: String,
    pub version: String,
    pub family: ModelFamily,
    pub artifact_digest: Option<String>,
}

impl EnsembleModelRef {
    pub fn new(id: impl Into<String>, version: impl Into<String>, family: ModelFamily) -> Result<Self> {
        let id = id.into();
        let version = version.into();
        non_empty(&id, "model id")?;
        non_empty(&version, "model version")?;
        Ok(Self {
            id,
            version,
            family,
            artifact_digest: None,
        })
    }

    pub fn with_artifact_digest(mut self, digest: impl Into<String>) -> Result<Self> {
        let digest = digest.into();
        non_empty(&digest, "model artifact digest")?;
        self.artifact_digest = Some(digest);
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplicabilityStatus {
    InDistribution,
    NearBoundary,
    OutOfDistribution,
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ApplicabilityAssessment {
    pub status: ApplicabilityStatus,
    /// Optional support score. This is model-specific evidence, not a universal probability.
    pub support: Option<f64>,
    pub note: Option<String>,
}

impl ApplicabilityAssessment {
    pub fn new(status: ApplicabilityStatus, support: Option<f64>, note: Option<String>) -> Result<Self> {
        if let Some(value) = support {
            probability(value, "applicability support")?;
        }
        Ok(Self { status, support, note })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationSummary {
    pub sample_count: usize,
    pub mean_absolute_error: Option<f64>,
    pub brier_score: Option<f64>,
    pub interval_nominal_coverage: Option<f64>,
    pub interval_observed_coverage: Option<f64>,
    pub validation_dataset_digest: String,
}

impl CalibrationSummary {
    pub fn new(sample_count: usize, validation_dataset_digest: impl Into<String>) -> Result<Self> {
        if sample_count == 0 {
            return Err(EnsembleError::InvalidSampleCount);
        }
        let validation_dataset_digest = validation_dataset_digest.into();
        non_empty(&validation_dataset_digest, "validation dataset digest")?;
        Ok(Self {
            sample_count,
            mean_absolute_error: None,
            brier_score: None,
            interval_nominal_coverage: None,
            interval_observed_coverage: None,
            validation_dataset_digest,
        })
    }

    pub fn with_mean_absolute_error(mut self, value: f64) -> Result<Self> {
        finite(value, "mean absolute error")?;
        if value < 0.0 {
            return Err(EnsembleError::NonFinite {
                field: "non-negative mean absolute error",
                value,
            });
        }
        self.mean_absolute_error = Some(value);
        Ok(self)
    }

    pub fn with_brier_score(mut self, value: f64) -> Result<Self> {
        probability(value, "brier score")?;
        self.brier_score = Some(value);
        Ok(self)
    }

    pub fn with_interval_coverage(mut self, nominal: f64, observed: f64) -> Result<Self> {
        if probability(nominal, "nominal interval coverage").is_err()
            || probability(observed, "observed interval coverage").is_err()
        {
            return Err(EnsembleError::InvalidCoverage { observed, nominal });
        }
        self.interval_nominal_coverage = Some(nominal);
        self.interval_observed_coverage = Some(observed);
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelPrediction {
    pub model: EnsembleModelRef,
    pub scenario_id: String,
    pub dimension_id: String,
    pub estimate: OutcomeEstimate,
    pub applicability: ApplicabilityAssessment,
    pub calibration: Option<CalibrationSummary>,
    pub assumptions: Vec<String>,
    pub sources: Vec<OutcomeSourceRef>,
}

impl ModelPrediction {
    pub fn new(
        model: EnsembleModelRef,
        scenario_id: impl Into<String>,
        dimension_id: impl Into<String>,
        estimate: OutcomeEstimate,
        applicability: ApplicabilityAssessment,
    ) -> Result<Self> {
        let scenario_id = scenario_id.into();
        let dimension_id = dimension_id.into();
        non_empty(&scenario_id, "scenario id")?;
        non_empty(&dimension_id, "dimension id")?;
        Ok(Self {
            model,
            scenario_id,
            dimension_id,
            estimate,
            applicability,
            calibration: None,
            assumptions: Vec::new(),
            sources: Vec::new(),
        })
    }

    pub fn with_calibration(mut self, calibration: CalibrationSummary) -> Self {
        self.calibration = Some(calibration);
        self
    }

    pub fn with_assumption(mut self, assumption: impl Into<String>) -> Result<Self> {
        let assumption = assumption.into();
        non_empty(&assumption, "model assumption")?;
        self.assumptions.push(assumption);
        Ok(self)
    }

    pub fn with_source(mut self, source: OutcomeSourceRef) -> Self {
        self.sources.push(source);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AgreementPolicy {
    pub absolute_point_tolerance: f64,
    pub require_interval_overlap: bool,
}

impl AgreementPolicy {
    pub fn new(absolute_point_tolerance: f64, require_interval_overlap: bool) -> Result<Self> {
        if !absolute_point_tolerance.is_finite() || absolute_point_tolerance < 0.0 {
            return Err(EnsembleError::InvalidTolerance(absolute_point_tolerance));
        }
        Ok(Self {
            absolute_point_tolerance,
            require_interval_overlap,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairwiseAgreementClass {
    Agreement,
    MaterialDisagreement,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairwiseModelComparison {
    pub left_model_id: String,
    pub right_model_id: String,
    pub absolute_point_gap: f64,
    pub intervals_overlap: Option<bool>,
    pub class: PairwiseAgreementClass,
}

fn interval_overlap(left: &OutcomeEstimate, right: &OutcomeEstimate) -> Option<bool> {
    match (left.lower, left.upper, right.lower, right.upper) {
        (Some(ll), Some(lu), Some(rl), Some(ru)) => Some(ll <= ru && rl <= lu),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelEnsembleReport {
    pub scenario_id: String,
    pub dimension_id: String,
    pub unit: String,
    pub predictions: Vec<ModelPrediction>,
}

impl ModelEnsembleReport {
    pub fn new(predictions: Vec<ModelPrediction>) -> Result<Self> {
        let Some(first) = predictions.first() else {
            return Err(EnsembleError::MissingPrediction);
        };

        let scenario_id = first.scenario_id.clone();
        let dimension_id = first.dimension_id.clone();
        let unit = first.estimate.unit.clone();
        let mut model_ids = HashSet::new();

        for prediction in &predictions {
            if prediction.scenario_id != scenario_id {
                return Err(EnsembleError::ScenarioMismatch {
                    expected: scenario_id,
                    got: prediction.scenario_id.clone(),
                });
            }
            if prediction.dimension_id != dimension_id {
                return Err(EnsembleError::DimensionMismatch {
                    expected: dimension_id,
                    got: prediction.dimension_id.clone(),
                });
            }
            if prediction.estimate.unit != unit {
                return Err(EnsembleError::UnitMismatch {
                    expected: unit,
                    got: prediction.estimate.unit.clone(),
                });
            }
            if !model_ids.insert(prediction.model.id.clone()) {
                return Err(EnsembleError::DuplicateModel(prediction.model.id.clone()));
            }
        }

        Ok(Self {
            scenario_id,
            dimension_id,
            unit,
            predictions,
        })
    }

    /// Models that explicitly claim in-distribution applicability. Unknown and
    /// OOD models remain visible in the full report rather than being deleted.
    pub fn in_distribution_predictions(&self) -> Vec<&ModelPrediction> {
        self.predictions
            .iter()
            .filter(|prediction| {
                prediction.applicability.status == ApplicabilityStatus::InDistribution
            })
            .collect()
    }

    pub fn pairwise_comparisons(&self, policy: AgreementPolicy) -> Vec<PairwiseModelComparison> {
        let mut comparisons = Vec::new();
        for left_index in 0..self.predictions.len() {
            for right_index in (left_index + 1)..self.predictions.len() {
                let left = &self.predictions[left_index];
                let right = &self.predictions[right_index];
                let absolute_point_gap = (left.estimate.point - right.estimate.point).abs();
                let intervals_overlap = interval_overlap(&left.estimate, &right.estimate);
                let point_agrees = absolute_point_gap <= policy.absolute_point_tolerance;
                let interval_agrees = if policy.require_interval_overlap {
                    intervals_overlap == Some(true)
                } else {
                    true
                };
                let class = if point_agrees && interval_agrees {
                    PairwiseAgreementClass::Agreement
                } else {
                    PairwiseAgreementClass::MaterialDisagreement
                };
                comparisons.push(PairwiseModelComparison {
                    left_model_id: left.model.id.clone(),
                    right_model_id: right.model.id.clone(),
                    absolute_point_gap,
                    intervals_overlap,
                    class,
                });
            }
        }
        comparisons
    }

    pub fn has_material_disagreement(&self, policy: AgreementPolicy) -> bool {
        self.pairwise_comparisons(policy)
            .iter()
            .any(|comparison| comparison.class == PairwiseAgreementClass::MaterialDisagreement)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(id: &str) -> EnsembleModelRef {
        EnsembleModelRef::new(id, "1.0", ModelFamily::Physical).unwrap()
    }

    fn applicability(status: ApplicabilityStatus) -> ApplicabilityAssessment {
        ApplicabilityAssessment::new(status, Some(0.8), None).unwrap()
    }

    fn prediction(id: &str, point: f64, lower: f64, upper: f64) -> ModelPrediction {
        ModelPrediction::new(
            model(id),
            "scenario-1",
            "wetland-area",
            OutcomeEstimate::interval(point, lower, upper, "ha").unwrap(),
            applicability(ApplicabilityStatus::InDistribution),
        )
        .unwrap()
    }

    #[test]
    fn ensemble_preserves_individual_predictions_instead_of_averaging() {
        let report = ModelEnsembleReport::new(vec![
            prediction("hydrology-a", 800.0, 760.0, 840.0),
            prediction("hydrology-b", 620.0, 590.0, 650.0),
        ])
        .unwrap();

        assert_eq!(report.predictions.len(), 2);
        assert_eq!(report.predictions[0].estimate.point, 800.0);
        assert_eq!(report.predictions[1].estimate.point, 620.0);
    }

    #[test]
    fn material_disagreement_requires_an_explicit_policy() {
        let report = ModelEnsembleReport::new(vec![
            prediction("a", 800.0, 760.0, 840.0),
            prediction("b", 620.0, 590.0, 650.0),
        ])
        .unwrap();
        let policy = AgreementPolicy::new(25.0, true).unwrap();

        assert!(report.has_material_disagreement(policy));
    }

    #[test]
    fn overlapping_intervals_can_still_disagree_when_point_gap_exceeds_policy() {
        let report = ModelEnsembleReport::new(vec![
            prediction("a", 100.0, 80.0, 120.0),
            prediction("b", 118.0, 95.0, 140.0),
        ])
        .unwrap();
        let policy = AgreementPolicy::new(10.0, true).unwrap();
        let comparison = &report.pairwise_comparisons(policy)[0];

        assert_eq!(comparison.intervals_overlap, Some(true));
        assert_eq!(comparison.class, PairwiseAgreementClass::MaterialDisagreement);
    }

    #[test]
    fn out_of_distribution_model_remains_visible_but_not_in_eligible_subset() {
        let mut ood = prediction("ood-model", 900.0, 850.0, 950.0);
        ood.applicability = applicability(ApplicabilityStatus::OutOfDistribution);
        let report = ModelEnsembleReport::new(vec![prediction("in-model", 800.0, 760.0, 840.0), ood])
            .unwrap();

        assert_eq!(report.predictions.len(), 2);
        assert_eq!(report.in_distribution_predictions().len(), 1);
        assert_eq!(report.in_distribution_predictions()[0].model.id, "in-model");
    }

    #[test]
    fn calibration_keeps_validation_lineage() {
        let calibration = CalibrationSummary::new(250, "sha256:validation-set")
            .unwrap()
            .with_mean_absolute_error(3.5)
            .unwrap()
            .with_interval_coverage(0.9, 0.86)
            .unwrap();
        let prediction = prediction("calibrated", 100.0, 90.0, 110.0).with_calibration(calibration);

        assert_eq!(
            prediction.calibration.unwrap().validation_dataset_digest,
            "sha256:validation-set"
        );
    }

    #[test]
    fn ensemble_rejects_incompatible_units() {
        let left = prediction("a", 100.0, 90.0, 110.0);
        let right = ModelPrediction::new(
            model("b"),
            "scenario-1",
            "wetland-area",
            OutcomeEstimate::point(0.8, "fraction").unwrap(),
            applicability(ApplicabilityStatus::InDistribution),
        )
        .unwrap();

        assert!(matches!(
            ModelEnsembleReport::new(vec![left, right]),
            Err(EnsembleError::UnitMismatch { .. })
        ));
    }
}
