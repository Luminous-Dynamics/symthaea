// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-fidelity materials evaluation with explicit applicability and calibration.
//!
//! Fidelity, cost, prediction uncertainty, and scientific authority are deliberately
//! separate concepts. A cheap surrogate can be useful for acquisition while remaining
//! a surrogate; an expensive calculation does not gain experimental authority merely
//! because it was expensive; and an out-of-domain result cannot silently enter an
//! ordinary campaign ranking.

use crate::conditioned_property::{
    ConditionedPropertyError, ConditionedPropertyObservation, PropertyArtifactRef,
    PropertyObservationMethod, PropertyUncertainty,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

const SHA256_HEX_LEN: usize = 64;

/// Broad evaluator family. This is descriptive methodology, not MAT-001 authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvaluationMethodClass {
    /// Database or literature lookup.
    DatabaseOrLiterature,
    /// Statistical, ML, or other surrogate prediction.
    Surrogate,
    /// First-principles electronic-structure calculation.
    Dft,
    /// Convex-hull or decomposition analysis.
    ConvexHull,
    /// Phonon/DFPT or other dynamical-stability calculation.
    Phonon,
    /// CALPHAD or other equilibrium/phase-diagram model.
    Calphad,
    /// Atomistic molecular dynamics or interatomic-potential simulation.
    Atomistic,
    /// Irradiation/displacement/transmutation calculation.
    Irradiation,
    /// Micromagnetic calculation.
    Micromagnetics,
    /// Continuum/FEM/transport/mechanics calculation.
    Continuum,
    /// Physical experiment/measurement.
    Experiment,
    /// Explicit other method family.
    Other(String),
}

/// How the evaluation result entered Symthaea.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvaluationOrigin {
    /// Imported result produced elsewhere.
    ExternalImport,
    /// Calculation reproduced locally from bound code and inputs.
    LocalReproduction,
    /// Statistical/ML/surrogate prediction.
    SurrogatePrediction,
    /// Physical measurement on a real specimen/device.
    PhysicalMeasurement,
}

/// Applicability-domain state for one evaluator/subject/property combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApplicabilityState {
    /// Evaluator declares this point inside its validated/calibrated domain.
    InDomain,
    /// Point lies near a declared applicability boundary.
    NearBoundary,
    /// Point lies outside the evaluator's declared applicability domain.
    OutOfDomain,
    /// Applicability has not been established.
    Unknown,
}

/// Source-bound applicability assessment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApplicabilityAssessment {
    /// Stable applicability-domain definition/version.
    pub domain_id: String,
    /// Domain state for this exact evaluation.
    pub state: ApplicabilityState,
    /// Optional model-specific distance/score. Its semantics belong to `domain_id`.
    pub score: Option<f64>,
    /// Optional source artifact defining or calculating the domain state.
    pub artifact: Option<PropertyArtifactRef>,
}

/// Exact evaluator/model/code identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluatorRef {
    /// Stable evaluator identifier.
    pub evaluator_id: String,
    /// Evaluator/model/code version.
    pub version: String,
    /// SHA-256 of exact executable/model/workflow artifact.
    pub artifact_sha256: String,
}

/// Resources consumed by one evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct EvaluationResourceCost {
    /// Compute core-hours.
    pub compute_core_hours: f64,
    /// Wall-clock hours.
    pub wall_time_hours: f64,
    /// Material/feedstock mass consumed, kg.
    pub material_mass_kg: f64,
    /// Optional direct monetary cost.
    pub direct_cost: Option<f64>,
    /// Currency paired with `direct_cost`.
    pub currency: Option<String>,
}

/// One paired approximate/reference value used for empirical cross-fidelity calibration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CalibrationPair {
    /// Approximate/cheaper-fidelity value.
    pub approximate: f64,
    /// Reference/higher-fidelity value.
    pub reference: f64,
}

/// Empirical relationship between one fidelity and a named reference fidelity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CrossFidelityCalibration {
    /// Stable calibration set/version.
    pub calibration_id: String,
    /// Fidelity being calibrated.
    pub approximate_fidelity_id: String,
    /// Reference fidelity.
    pub reference_fidelity_id: String,
    /// Number of paired observations.
    pub sample_count: u32,
    /// Mean absolute error in the property's native unit.
    pub mean_absolute_error: f64,
    /// Root mean squared error in the property's native unit.
    pub root_mean_squared_error: f64,
    /// Mean signed error `(approximate - reference)`.
    pub mean_bias: f64,
    /// Optional empirical coverage fraction for declared uncertainty intervals.
    pub coverage_fraction: Option<f64>,
    /// Exact calibration dataset/result artifact.
    pub artifact: PropertyArtifactRef,
}

impl CrossFidelityCalibration {
    /// Validate calibration metadata and statistics.
    pub fn validate(&self) -> Result<(), MultiFidelityError> {
        nonempty("calibration_id", &self.calibration_id)?;
        nonempty("approximate_fidelity_id", &self.approximate_fidelity_id)?;
        nonempty("reference_fidelity_id", &self.reference_fidelity_id)?;
        if self.sample_count == 0 {
            return Err(MultiFidelityError::EmptyCalibrationSet);
        }
        nonnegative("mean_absolute_error", self.mean_absolute_error)?;
        nonnegative("root_mean_squared_error", self.root_mean_squared_error)?;
        finite("mean_bias", self.mean_bias)?;
        if let Some(coverage) = self.coverage_fraction {
            finite("coverage_fraction", coverage)?;
            if !(0.0..=1.0).contains(&coverage) {
                return Err(MultiFidelityError::FractionOutOfRange {
                    field: "coverage_fraction",
                    value: coverage,
                });
            }
        }
        artifact(&self.artifact)
    }
}

/// One complete multi-fidelity evaluation record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiFidelityEvaluation {
    /// Deterministic result identity.
    pub evaluation_id: String,
    /// Evaluator identity.
    pub evaluator: EvaluatorRef,
    /// Descriptive method class.
    pub method_class: EvaluationMethodClass,
    /// Origin of the result.
    pub origin: EvaluationOrigin,
    /// Stable fidelity/method identifier used by campaigns/search memory.
    pub fidelity_id: String,
    /// Exact conditioned property result.
    pub observation: ConditionedPropertyObservation,
    /// Applicability assessment for this result.
    pub applicability: ApplicabilityAssessment,
    /// Optional empirical relationship to a reference fidelity.
    pub calibration: Option<CrossFidelityCalibration>,
    /// Bound input/reference artifacts beyond those already carried by the observation.
    pub input_artifacts: Vec<PropertyArtifactRef>,
    /// Resources consumed by the evaluation.
    pub cost: EvaluationResourceCost,
}

impl MultiFidelityEvaluation {
    /// Construct a validated evaluation and derive its deterministic identity.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        evaluator: EvaluatorRef,
        method_class: EvaluationMethodClass,
        origin: EvaluationOrigin,
        fidelity_id: String,
        observation: ConditionedPropertyObservation,
        applicability: ApplicabilityAssessment,
        calibration: Option<CrossFidelityCalibration>,
        input_artifacts: Vec<PropertyArtifactRef>,
        cost: EvaluationResourceCost,
    ) -> Result<Self, MultiFidelityError> {
        let mut result = Self {
            evaluation_id: String::new(),
            evaluator,
            method_class,
            origin,
            fidelity_id,
            observation,
            applicability,
            calibration,
            input_artifacts,
            cost,
        };
        result.validate_without_id()?;
        result.evaluation_id = result.derived_identity()?;
        Ok(result)
    }

    /// Validate stored content including deterministic identity consistency.
    pub fn validate(&self) -> Result<(), MultiFidelityError> {
        self.validate_without_id()?;
        if self.evaluation_id != self.derived_identity()? {
            return Err(MultiFidelityError::EvaluationIdentityMismatch);
        }
        Ok(())
    }

    /// Whether this result may participate in normal acquisition/ranking logic.
    ///
    /// `OutOfDomain` is a hard exclusion. `Unknown` remains visible to callers but
    /// is not silently converted into an in-domain confidence estimate.
    pub fn may_inform_acquisition(&self) -> Result<bool, MultiFidelityError> {
        self.validate()?;
        Ok(self.applicability.state != ApplicabilityState::OutOfDomain)
    }

    /// Whether this result is explicitly established as inside the evaluator's domain.
    pub fn is_explicitly_in_domain(&self) -> Result<bool, MultiFidelityError> {
        self.validate()?;
        Ok(self.applicability.state == ApplicabilityState::InDomain)
    }

    /// Descriptive scientific origin retained without mapping to MAT-001 authority.
    pub fn origin(&self) -> EvaluationOrigin {
        self.origin
    }

    fn derived_identity(&self) -> Result<String, MultiFidelityError> {
        Ok(format!(
            "materials-evaluation:v1|comparison={}|evaluator={}|version={}|artifact={}|fidelity={}|origin={}|result={}",
            token(&self.observation.direct_comparison_key()?),
            token(&self.evaluator.evaluator_id),
            token(&self.evaluator.version),
            self.evaluator.artifact_sha256.to_ascii_lowercase(),
            token(&self.fidelity_id),
            origin_key(self.origin),
            self.observation.artifact.artifact_sha256.to_ascii_lowercase()
        ))
    }

    fn validate_without_id(&self) -> Result<(), MultiFidelityError> {
        nonempty("evaluator_id", &self.evaluator.evaluator_id)?;
        nonempty("evaluator version", &self.evaluator.version)?;
        sha256(&self.evaluator.artifact_sha256)?;
        nonempty("fidelity_id", &self.fidelity_id)?;
        self.observation.validate()?;
        validate_origin_method(self.origin, &self.observation.method)?;
        validate_applicability(&self.applicability)?;
        if let Some(calibration) = &self.calibration {
            calibration.validate()?;
            if calibration.approximate_fidelity_id != self.fidelity_id {
                return Err(MultiFidelityError::CalibrationFidelityMismatch);
            }
        }
        let mut seen = HashSet::new();
        for input in &self.input_artifacts {
            artifact(input)?;
            let key = (&input.source_id, &input.artifact_sha256);
            if !seen.insert(key) {
                return Err(MultiFidelityError::DuplicateInputArtifact);
            }
        }
        validate_cost(&self.cost)
    }
}

/// Policy used to interpret declared uncertainties when checking disagreement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContradictionPolicy {
    /// Number of standard uncertainties used to form an interval for `Standard` uncertainty.
    pub sigma_multiplier: f64,
    /// Optional absolute tolerance used when one or both observations have unknown uncertainty.
    pub unknown_uncertainty_tolerance: Option<f64>,
    /// Additional absolute gap allowed between known uncertainty intervals.
    pub interval_gap_tolerance: f64,
}

/// Relationship between two evaluations of a directly comparable property.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EvaluationAgreement {
    /// Subject/property/unit/conditions differ, so no direct comparison was attempted.
    NotDirectlyComparable,
    /// Directly comparable, but quantitative uncertainty was insufficient for contradiction testing.
    IndeterminateUncertainty,
    /// Declared uncertainty/tolerance regions overlap or touch.
    Consistent {
        /// Signed difference `left - right`.
        difference: f64,
    },
    /// Declared uncertainty/tolerance regions are disjoint beyond the policy tolerance.
    Contradictory {
        /// Signed difference `left - right`.
        difference: f64,
        /// Positive uncovered gap between uncertainty/tolerance regions.
        interval_gap: f64,
    },
}

/// Assess disagreement without averaging or choosing a preferred fidelity.
pub fn assess_evaluation_agreement(
    left: &MultiFidelityEvaluation,
    right: &MultiFidelityEvaluation,
    policy: ContradictionPolicy,
) -> Result<EvaluationAgreement, MultiFidelityError> {
    left.validate()?;
    right.validate()?;
    validate_contradiction_policy(policy)?;
    if !left
        .observation
        .is_directly_comparable_with(&right.observation)?
    {
        return Ok(EvaluationAgreement::NotDirectlyComparable);
    }

    let left_interval = uncertainty_interval(
        left.observation.value,
        &left.observation.uncertainty,
        policy,
    )?;
    let right_interval = uncertainty_interval(
        right.observation.value,
        &right.observation.uncertainty,
        policy,
    )?;
    let difference = left.observation.value - right.observation.value;
    let (Some((left_low, left_high)), Some((right_low, right_high))) =
        (left_interval, right_interval)
    else {
        return Ok(EvaluationAgreement::IndeterminateUncertainty);
    };

    let gap = if left_high < right_low {
        right_low - left_high
    } else if right_high < left_low {
        left_low - right_high
    } else {
        0.0
    };

    if gap > policy.interval_gap_tolerance {
        Ok(EvaluationAgreement::Contradictory {
            difference,
            interval_gap: gap,
        })
    } else {
        Ok(EvaluationAgreement::Consistent { difference })
    }
}

/// Compute empirical error statistics for a cheaper/approximate fidelity against a named reference.
pub fn calibrate_against_reference(
    calibration_id: String,
    approximate_fidelity_id: String,
    reference_fidelity_id: String,
    pairs: &[CalibrationPair],
    coverage_fraction: Option<f64>,
    artifact_ref: PropertyArtifactRef,
) -> Result<CrossFidelityCalibration, MultiFidelityError> {
    nonempty("calibration_id", &calibration_id)?;
    nonempty("approximate_fidelity_id", &approximate_fidelity_id)?;
    nonempty("reference_fidelity_id", &reference_fidelity_id)?;
    if pairs.is_empty() {
        return Err(MultiFidelityError::EmptyCalibrationSet);
    }
    let mut abs_sum = 0.0;
    let mut squared_sum = 0.0;
    let mut signed_sum = 0.0;
    for pair in pairs {
        finite("calibration approximate", pair.approximate)?;
        finite("calibration reference", pair.reference)?;
        let error = pair.approximate - pair.reference;
        abs_sum += error.abs();
        squared_sum += error * error;
        signed_sum += error;
    }
    let n = pairs.len() as f64;
    let calibration = CrossFidelityCalibration {
        calibration_id,
        approximate_fidelity_id,
        reference_fidelity_id,
        sample_count: pairs.len() as u32,
        mean_absolute_error: abs_sum / n,
        root_mean_squared_error: (squared_sum / n).sqrt(),
        mean_bias: signed_sum / n,
        coverage_fraction,
        artifact: artifact_ref,
    };
    calibration.validate()?;
    Ok(calibration)
}

fn validate_origin_method(
    origin: EvaluationOrigin,
    method: &PropertyObservationMethod,
) -> Result<(), MultiFidelityError> {
    let compatible = matches!(
        (origin, method),
        (
            EvaluationOrigin::ExternalImport,
            PropertyObservationMethod::DatabaseImported { .. }
                | PropertyObservationMethod::LiteratureReported { .. }
        ) | (
            EvaluationOrigin::LocalReproduction,
            PropertyObservationMethod::Calculation { .. }
        ) | (
            EvaluationOrigin::SurrogatePrediction,
            PropertyObservationMethod::ModelPrediction { .. }
        ) | (
            EvaluationOrigin::PhysicalMeasurement,
            PropertyObservationMethod::Experiment { .. }
        )
    );
    if compatible {
        Ok(())
    } else {
        Err(MultiFidelityError::OriginMethodMismatch)
    }
}

fn validate_applicability(value: &ApplicabilityAssessment) -> Result<(), MultiFidelityError> {
    nonempty("applicability domain_id", &value.domain_id)?;
    if let Some(score) = value.score {
        finite("applicability score", score)?;
    }
    if let Some(artifact_ref) = &value.artifact {
        artifact(artifact_ref)?;
    }
    Ok(())
}

fn validate_cost(cost: &EvaluationResourceCost) -> Result<(), MultiFidelityError> {
    nonnegative("compute_core_hours", cost.compute_core_hours)?;
    nonnegative("wall_time_hours", cost.wall_time_hours)?;
    nonnegative("material_mass_kg", cost.material_mass_kg)?;
    match (cost.direct_cost, cost.currency.as_deref()) {
        (Some(value), Some(currency)) => {
            nonnegative("direct_cost", value)?;
            nonempty("currency", currency)
        }
        (None, None) => Ok(()),
        _ => Err(MultiFidelityError::IncompleteMonetaryCost),
    }
}

fn validate_contradiction_policy(policy: ContradictionPolicy) -> Result<(), MultiFidelityError> {
    nonnegative("sigma_multiplier", policy.sigma_multiplier)?;
    nonnegative("interval_gap_tolerance", policy.interval_gap_tolerance)?;
    if let Some(value) = policy.unknown_uncertainty_tolerance {
        nonnegative("unknown_uncertainty_tolerance", value)?;
    }
    Ok(())
}

fn uncertainty_interval(
    value: f64,
    uncertainty: &PropertyUncertainty,
    policy: ContradictionPolicy,
) -> Result<Option<(f64, f64)>, MultiFidelityError> {
    match uncertainty {
        PropertyUncertainty::Unknown => Ok(policy
            .unknown_uncertainty_tolerance
            .map(|tolerance| (value - tolerance, value + tolerance))),
        PropertyUncertainty::Standard { sigma } => {
            nonnegative("sigma", *sigma)?;
            let width = policy.sigma_multiplier * sigma;
            Ok(Some((value - width, value + width)))
        }
        PropertyUncertainty::Interval { lower, upper, .. } => Ok(Some((*lower, *upper))),
    }
}

fn artifact(value: &PropertyArtifactRef) -> Result<(), MultiFidelityError> {
    nonempty("artifact source_id", &value.source_id)?;
    sha256(&value.artifact_sha256)
}

fn sha256(value: &str) -> Result<(), MultiFidelityError> {
    if value.len() != SHA256_HEX_LEN || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Err(MultiFidelityError::InvalidSha256)
    } else {
        Ok(())
    }
}

fn nonempty(field: &'static str, value: &str) -> Result<(), MultiFidelityError> {
    if value.trim().is_empty() {
        Err(MultiFidelityError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn finite(field: &'static str, value: f64) -> Result<(), MultiFidelityError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(MultiFidelityError::NonFiniteValue { field, value })
    }
}

fn nonnegative(field: &'static str, value: f64) -> Result<(), MultiFidelityError> {
    finite(field, value)?;
    if value < 0.0 {
        Err(MultiFidelityError::NegativeValue { field, value })
    } else {
        Ok(())
    }
}

fn origin_key(origin: EvaluationOrigin) -> &'static str {
    match origin {
        EvaluationOrigin::ExternalImport => "external-import",
        EvaluationOrigin::LocalReproduction => "local-reproduction",
        EvaluationOrigin::SurrogatePrediction => "surrogate-prediction",
        EvaluationOrigin::PhysicalMeasurement => "physical-measurement",
    }
}

fn token(value: &str) -> String {
    let mut output = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.') {
            output.push(byte as char);
        } else {
            output.push_str(&format!("%{byte:02X}"));
        }
    }
    output
}

/// Multi-fidelity validation/calibration/comparison failure.
#[derive(Debug, Clone, PartialEq)]
pub enum MultiFidelityError {
    /// Required text field was empty.
    EmptyField(&'static str),
    /// Numeric value was NaN or infinite.
    NonFiniteValue {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// Numeric value was negative where non-negative was required.
    NegativeValue {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// Fraction-like value was outside [0,1].
    FractionOutOfRange {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// SHA-256 binding was malformed.
    InvalidSha256,
    /// Direct cost/currency were not supplied together.
    IncompleteMonetaryCost,
    /// Observation method did not match the claimed result origin.
    OriginMethodMismatch,
    /// Stored identity did not match the exact result definition.
    EvaluationIdentityMismatch,
    /// Same input artifact was repeated.
    DuplicateInputArtifact,
    /// Calibration contained no paired observations.
    EmptyCalibrationSet,
    /// Calibration was attached to an evaluation with a different approximate fidelity ID.
    CalibrationFidelityMismatch,
    /// Underlying MAT-008 observation was invalid.
    ConditionedProperty(ConditionedPropertyError),
}

impl From<ConditionedPropertyError> for MultiFidelityError {
    fn from(value: ConditionedPropertyError) -> Self {
        Self::ConditionedProperty(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conditioned_property::{PropertyConditions, PropertyEvidenceClass};

    const A64: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B64: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C64: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn artifact_ref(source: &str, hash: &str) -> PropertyArtifactRef {
        PropertyArtifactRef {
            source_id: source.to_string(),
            artifact_sha256: hash.to_string(),
        }
    }

    fn model_observation(value: f64, uncertainty: PropertyUncertainty) -> ConditionedPropertyObservation {
        ConditionedPropertyObservation {
            subject_identity: "material-subject:v1|Fe-Co-X-fixture".to_string(),
            property_id: "formation_energy".to_string(),
            value,
            unit: "eV/atom".to_string(),
            uncertainty,
            conditions: PropertyConditions {
                temperature_k: Some(0.0_f64.max(1.0)),
                ..Default::default()
            },
            method: PropertyObservationMethod::ModelPrediction {
                model_id: "cgcnn-fixture".to_string(),
                model_sha256: B64.to_string(),
                applicability_domain_id: Some("soap-loco-v1".to_string()),
            },
            artifact: artifact_ref("prediction", C64),
        }
    }

    fn surrogate(value: f64, state: ApplicabilityState, uncertainty: PropertyUncertainty) -> MultiFidelityEvaluation {
        MultiFidelityEvaluation::new(
            EvaluatorRef {
                evaluator_id: "cgcnn-fixture".to_string(),
                version: "1".to_string(),
                artifact_sha256: B64.to_string(),
            },
            EvaluationMethodClass::Surrogate,
            EvaluationOrigin::SurrogatePrediction,
            "ml-cgcnn".to_string(),
            model_observation(value, uncertainty),
            ApplicabilityAssessment {
                domain_id: "soap-loco-v1".to_string(),
                state,
                score: Some(0.2),
                artifact: Some(artifact_ref("ood-basis", A64)),
            },
            None,
            vec![],
            EvaluationResourceCost {
                compute_core_hours: 0.01,
                wall_time_hours: 0.001,
                material_mass_kg: 0.0,
                direct_cost: None,
                currency: None,
            },
        )
        .unwrap()
    }

    fn local_dft(value: f64, uncertainty: PropertyUncertainty) -> MultiFidelityEvaluation {
        let observation = ConditionedPropertyObservation {
            subject_identity: "material-subject:v1|Fe-Co-X-fixture".to_string(),
            property_id: "formation_energy".to_string(),
            value,
            unit: "eV/atom".to_string(),
            uncertainty,
            conditions: PropertyConditions {
                temperature_k: Some(1.0),
                ..Default::default()
            },
            method: PropertyObservationMethod::Calculation {
                method_id: "DFT-PBE".to_string(),
                code_id: "qe-fixture".to_string(),
                input_sha256: A64.to_string(),
                output_sha256: C64.to_string(),
            },
            artifact: artifact_ref("local-dft", C64),
        };
        MultiFidelityEvaluation::new(
            EvaluatorRef {
                evaluator_id: "qe-dft".to_string(),
                version: "7.x-fixture".to_string(),
                artifact_sha256: A64.to_string(),
            },
            EvaluationMethodClass::Dft,
            EvaluationOrigin::LocalReproduction,
            "dft-pbe".to_string(),
            observation,
            ApplicabilityAssessment {
                domain_id: "dft-method-domain-v1".to_string(),
                state: ApplicabilityState::InDomain,
                score: None,
                artifact: None,
            },
            None,
            vec![],
            EvaluationResourceCost {
                compute_core_hours: 16.0,
                wall_time_hours: 2.0,
                material_mass_kg: 0.0,
                direct_cost: None,
                currency: None,
            },
        )
        .unwrap()
    }

    #[test]
    fn out_of_domain_result_is_excluded_from_normal_acquisition() {
        let result = surrogate(-0.2, ApplicabilityState::OutOfDomain, PropertyUncertainty::Unknown);
        assert!(!result.may_inform_acquisition().unwrap());
        assert!(!result.is_explicitly_in_domain().unwrap());
    }

    #[test]
    fn cheap_biased_fidelity_can_be_calibrated_without_becoming_reference_evidence() {
        let calibration = calibrate_against_reference(
            "mag-fixture-cal-v1".to_string(),
            "ml-cgcnn".to_string(),
            "dft-pbe".to_string(),
            &[
                CalibrationPair { approximate: 8.0, reference: 10.0 },
                CalibrationPair { approximate: 18.0, reference: 20.0 },
                CalibrationPair { approximate: 28.0, reference: 30.0 },
            ],
            None,
            artifact_ref("calibration", A64),
        )
        .unwrap();
        assert_eq!(calibration.sample_count, 3);
        assert!((calibration.mean_absolute_error - 2.0).abs() < 1e-12);
        assert!((calibration.root_mean_squared_error - 2.0).abs() < 1e-12);
        assert!((calibration.mean_bias + 2.0).abs() < 1e-12);

        let low = surrogate(-0.2, ApplicabilityState::InDomain, PropertyUncertainty::Unknown);
        let high = local_dft(-0.18, PropertyUncertainty::Unknown);
        assert!(low.may_inform_acquisition().unwrap());
        assert_eq!(low.origin(), EvaluationOrigin::SurrogatePrediction);
        assert_eq!(high.origin(), EvaluationOrigin::LocalReproduction);
        assert_eq!(low.observation.method.evidence_class(), PropertyEvidenceClass::Predicted);
        assert_eq!(high.observation.method.evidence_class(), PropertyEvidenceClass::Calculated);
    }

    #[test]
    fn origin_method_mismatch_is_rejected() {
        let mut observation = model_observation(-0.2, PropertyUncertainty::Unknown);
        observation.method = PropertyObservationMethod::DatabaseImported {
            provider_id: "materials-project".to_string(),
            record_id: "mp-fixture".to_string(),
        };
        let result = MultiFidelityEvaluation::new(
            EvaluatorRef {
                evaluator_id: "external-dft".to_string(),
                version: "1".to_string(),
                artifact_sha256: A64.to_string(),
            },
            EvaluationMethodClass::Dft,
            EvaluationOrigin::LocalReproduction,
            "dft-pbe".to_string(),
            observation,
            ApplicabilityAssessment {
                domain_id: "fixture".to_string(),
                state: ApplicabilityState::Unknown,
                score: None,
                artifact: None,
            },
            None,
            vec![],
            EvaluationResourceCost::default(),
        );
        assert_eq!(result.unwrap_err(), MultiFidelityError::OriginMethodMismatch);
    }

    #[test]
    fn contradiction_is_explicit_and_never_averaged() {
        let left = surrogate(
            -0.30,
            ApplicabilityState::InDomain,
            PropertyUncertainty::Interval {
                lower: -0.31,
                upper: -0.29,
                confidence_fraction: Some(0.95),
            },
        );
        let right = local_dft(
            -0.20,
            PropertyUncertainty::Interval {
                lower: -0.21,
                upper: -0.19,
                confidence_fraction: Some(0.95),
            },
        );
        let agreement = assess_evaluation_agreement(
            &left,
            &right,
            ContradictionPolicy {
                sigma_multiplier: 2.0,
                unknown_uncertainty_tolerance: None,
                interval_gap_tolerance: 0.0,
            },
        )
        .unwrap();
        assert!(matches!(agreement, EvaluationAgreement::Contradictory { .. }));
    }

    #[test]
    fn unknown_uncertainty_is_indeterminate_without_explicit_policy_tolerance() {
        let left = surrogate(-0.30, ApplicabilityState::InDomain, PropertyUncertainty::Unknown);
        let right = local_dft(-0.20, PropertyUncertainty::Unknown);
        let agreement = assess_evaluation_agreement(
            &left,
            &right,
            ContradictionPolicy {
                sigma_multiplier: 2.0,
                unknown_uncertainty_tolerance: None,
                interval_gap_tolerance: 0.0,
            },
        )
        .unwrap();
        assert_eq!(agreement, EvaluationAgreement::IndeterminateUncertainty);
    }

    #[test]
    fn differing_conditions_are_not_directly_compared() {
        let left = surrogate(-0.30, ApplicabilityState::InDomain, PropertyUncertainty::Unknown);
        let mut right = local_dft(-0.30, PropertyUncertainty::Unknown);
        right.observation.conditions.temperature_k = Some(300.0);
        right.evaluation_id = right.derived_identity().unwrap();
        let agreement = assess_evaluation_agreement(
            &left,
            &right,
            ContradictionPolicy {
                sigma_multiplier: 2.0,
                unknown_uncertainty_tolerance: Some(0.01),
                interval_gap_tolerance: 0.0,
            },
        )
        .unwrap();
        assert_eq!(agreement, EvaluationAgreement::NotDirectlyComparable);
    }
}
