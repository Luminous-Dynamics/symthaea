// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Full-semantic sealing for multi-fidelity materials evaluations.
//!
//! `MultiFidelityEvaluation::evaluation_id` is intentionally treated here as a
//! record identifier, not by itself as a complete authority seal. This module
//! binds every authority-bearing field needed by later evidence promotion:
//! value, uncertainty, method details, applicability/OOD state, calibration,
//! input artifacts, and resource cost.
//!
//! The seal does not increase scientific authority. It only makes mutation of
//! an already-recorded evaluation machine-detectable at the promotion boundary.

use crate::conditioned_property::{
    ConditionedPropertyError, PropertyArtifactRef, PropertyObservationMethod, PropertyUncertainty,
};
use crate::multi_fidelity::{
    ApplicabilityAssessment, ApplicabilityState, CrossFidelityCalibration,
    EvaluationMethodClass, EvaluationOrigin, EvaluationResourceCost, MultiFidelityError,
    MultiFidelityEvaluation,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

/// Canonical full-semantic seal for one exact MAT-011 evaluation.
///
/// The `semantic_identity` is deliberately a canonical string rather than a
/// cryptographic digest so this crate does not add a hashing dependency merely
/// to close the identity boundary. Callers may content-address the serialized
/// seal as a separate artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultiFidelityEvaluationSeal {
    /// The MAT-011 evaluation record identifier this seal was created from.
    pub evaluation_id: String,
    /// Canonical projection of all authority-bearing evaluation semantics.
    pub semantic_identity: String,
}

impl MultiFidelityEvaluationSeal {
    /// Create a seal from a validated exact evaluation.
    pub fn from_evaluation(
        evaluation: &MultiFidelityEvaluation,
    ) -> Result<Self, EvaluationSealError> {
        evaluation.validate()?;
        Ok(Self {
            evaluation_id: evaluation.evaluation_id.clone(),
            semantic_identity: canonical_semantic_identity(evaluation)?,
        })
    }

    /// Require this stored seal to describe the supplied evaluation exactly.
    pub fn validate_against(
        &self,
        evaluation: &MultiFidelityEvaluation,
    ) -> Result<(), EvaluationSealError> {
        evaluation.validate()?;
        if self.evaluation_id != evaluation.evaluation_id
            || self.semantic_identity != canonical_semantic_identity(evaluation)?
        {
            return Err(EvaluationSealError::SealMismatch);
        }
        Ok(())
    }
}

/// Build the full canonical semantic identity for an evaluation.
pub fn canonical_semantic_identity(
    evaluation: &MultiFidelityEvaluation,
) -> Result<String, EvaluationSealError> {
    evaluation.validate()?;

    let comparison_key = evaluation.observation.direct_comparison_key()?;
    let inputs = canonical_input_artifacts(&evaluation.input_artifacts);

    Ok(format!(
        concat!(
            "materials-evaluation-seal:v1",
            "|record={}",
            "|comparison={}",
            "|value={}",
            "|uncertainty={}",
            "|observation_method={}",
            "|result_artifact={}",
            "|evaluator={}",
            "|method_class={}",
            "|origin={}",
            "|fidelity={}",
            "|applicability={}",
            "|calibration={}",
            "|inputs=[{}]",
            "|cost={}"
        ),
        token(&evaluation.evaluation_id),
        token(&comparison_key),
        float_key(evaluation.observation.value),
        uncertainty_key(&evaluation.observation.uncertainty),
        observation_method_key(&evaluation.observation.method),
        artifact_key(&evaluation.observation.artifact),
        evaluator_key(evaluation),
        method_class_key(&evaluation.method_class),
        origin_key(evaluation.origin),
        token(&evaluation.fidelity_id),
        applicability_key(&evaluation.applicability),
        calibration_key(evaluation.calibration.as_ref()),
        inputs,
        cost_key(&evaluation.cost),
    ))
}

fn evaluator_key(evaluation: &MultiFidelityEvaluation) -> String {
    format!(
        "{}:{}:{}",
        token(&evaluation.evaluator.evaluator_id),
        token(&evaluation.evaluator.version),
        evaluation.evaluator.artifact_sha256.to_ascii_lowercase(),
    )
}

fn uncertainty_key(uncertainty: &PropertyUncertainty) -> String {
    match uncertainty {
        PropertyUncertainty::Unknown => "unknown".to_string(),
        PropertyUncertainty::Standard { sigma } => {
            format!("standard:{}", float_key(*sigma))
        }
        PropertyUncertainty::Interval {
            lower,
            upper,
            confidence_fraction,
        } => format!(
            "interval:{}:{}:{}",
            float_key(*lower),
            float_key(*upper),
            optional_float_key(*confidence_fraction),
        ),
    }
}

fn observation_method_key(method: &PropertyObservationMethod) -> String {
    match method {
        PropertyObservationMethod::Experiment {
            method_id,
            instrument_id,
            calibration,
        } => format!(
            "experiment:{}:{}:{}",
            token(method_id),
            optional_string_key(instrument_id.as_deref()),
            optional_artifact_key(calibration.as_ref()),
        ),
        PropertyObservationMethod::Calculation {
            method_id,
            code_id,
            input_sha256,
            output_sha256,
        } => format!(
            "calculation:{}:{}:{}:{}",
            token(method_id),
            token(code_id),
            input_sha256.to_ascii_lowercase(),
            output_sha256.to_ascii_lowercase(),
        ),
        PropertyObservationMethod::LiteratureReported {
            publication_id,
            locator,
        } => format!(
            "literature:{}:{}",
            token(publication_id),
            token(locator),
        ),
        PropertyObservationMethod::DatabaseImported {
            provider_id,
            record_id,
        } => format!(
            "database:{}:{}",
            token(provider_id),
            token(record_id),
        ),
        PropertyObservationMethod::ModelPrediction {
            model_id,
            model_sha256,
            applicability_domain_id,
        } => format!(
            "model:{}:{}:{}",
            token(model_id),
            model_sha256.to_ascii_lowercase(),
            optional_string_key(applicability_domain_id.as_deref()),
        ),
    }
}

fn method_class_key(method: &EvaluationMethodClass) -> String {
    match method {
        EvaluationMethodClass::DatabaseOrLiterature => "database-or-literature".to_string(),
        EvaluationMethodClass::Surrogate => "surrogate".to_string(),
        EvaluationMethodClass::Dft => "dft".to_string(),
        EvaluationMethodClass::ConvexHull => "convex-hull".to_string(),
        EvaluationMethodClass::Phonon => "phonon".to_string(),
        EvaluationMethodClass::Calphad => "calphad".to_string(),
        EvaluationMethodClass::Atomistic => "atomistic".to_string(),
        EvaluationMethodClass::Irradiation => "irradiation".to_string(),
        EvaluationMethodClass::Micromagnetics => "micromagnetics".to_string(),
        EvaluationMethodClass::Continuum => "continuum".to_string(),
        EvaluationMethodClass::Experiment => "experiment".to_string(),
        EvaluationMethodClass::Other(value) => format!("other:{}", token(value)),
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

fn applicability_key(applicability: &ApplicabilityAssessment) -> String {
    format!(
        "{}:{}:{}:{}",
        token(&applicability.domain_id),
        applicability_state_key(applicability.state),
        optional_float_key(applicability.score),
        optional_artifact_key(applicability.artifact.as_ref()),
    )
}

fn applicability_state_key(state: ApplicabilityState) -> &'static str {
    match state {
        ApplicabilityState::InDomain => "in-domain",
        ApplicabilityState::NearBoundary => "near-boundary",
        ApplicabilityState::OutOfDomain => "out-of-domain",
        ApplicabilityState::Unknown => "unknown",
    }
}

fn calibration_key(calibration: Option<&CrossFidelityCalibration>) -> String {
    let Some(calibration) = calibration else {
        return "-".to_string();
    };

    let coverage = optional_float_key(calibration.coverage_fraction);
    let artifact = artifact_key(&calibration.artifact);
    format!(
        "{}:{}:{}:{}:{}:{}:{}:{}:{}",
        token(&calibration.calibration_id),
        token(&calibration.approximate_fidelity_id),
        token(&calibration.reference_fidelity_id),
        calibration.sample_count,
        float_key(calibration.mean_absolute_error),
        float_key(calibration.root_mean_squared_error),
        float_key(calibration.mean_bias),
        coverage,
        artifact,
    )
}

fn canonical_input_artifacts(artifacts: &[PropertyArtifactRef]) -> String {
    let mut keys = artifacts.iter().map(artifact_key).collect::<Vec<_>>();
    keys.sort();
    keys.join(",")
}

fn cost_key(cost: &EvaluationResourceCost) -> String {
    format!(
        "{}:{}:{}:{}:{}",
        float_key(cost.compute_core_hours),
        float_key(cost.wall_time_hours),
        float_key(cost.material_mass_kg),
        optional_float_key(cost.direct_cost),
        optional_string_key(cost.currency.as_deref()),
    )
}

fn artifact_key(artifact: &PropertyArtifactRef) -> String {
    format!(
        "{}:{}",
        token(&artifact.source_id),
        artifact.artifact_sha256.to_ascii_lowercase(),
    )
}

fn optional_artifact_key(artifact: Option<&PropertyArtifactRef>) -> String {
    artifact.map(artifact_key).unwrap_or_else(|| "-".to_string())
}

fn optional_string_key(value: Option<&str>) -> String {
    value.map(token).unwrap_or_else(|| "-".to_string())
}

fn optional_float_key(value: Option<f64>) -> String {
    value.map(float_key).unwrap_or_else(|| "-".to_string())
}

fn float_key(value: f64) -> String {
    format!("{:016x}", value.to_bits())
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

/// Failure while constructing or validating a full-semantic evaluation seal.
#[derive(Debug, Clone, PartialEq)]
pub enum EvaluationSealError {
    /// The underlying MAT-011 evaluation did not validate.
    Evaluation(MultiFidelityError),
    /// The underlying MAT-008 observation did not validate.
    Observation(ConditionedPropertyError),
    /// Stored seal no longer matches the exact evaluation semantics.
    SealMismatch,
}

impl fmt::Display for EvaluationSealError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Evaluation(error) => write!(formatter, "invalid MAT-011 evaluation: {error:?}"),
            Self::Observation(error) => write!(formatter, "invalid MAT-008 observation: {error:?}"),
            Self::SealMismatch => {
                formatter.write_str("stored evaluation seal does not match exact evaluation semantics")
            }
        }
    }
}

impl Error for EvaluationSealError {}

impl From<MultiFidelityError> for EvaluationSealError {
    fn from(value: MultiFidelityError) -> Self {
        Self::Evaluation(value)
    }
}

impl From<ConditionedPropertyError> for EvaluationSealError {
    fn from(value: ConditionedPropertyError) -> Self {
        Self::Observation(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conditioned_property::{ConditionedPropertyObservation, PropertyConditions};
    use crate::multi_fidelity::EvaluatorRef;

    const A64: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B64: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C64: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const D64: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    fn artifact(source_id: &str, hash: &str) -> PropertyArtifactRef {
        PropertyArtifactRef {
            source_id: source_id.to_string(),
            artifact_sha256: hash.to_string(),
        }
    }

    fn local_dft(value: f64) -> MultiFidelityEvaluation {
        MultiFidelityEvaluation::new(
            EvaluatorRef {
                evaluator_id: "qe".to_string(),
                version: "7.x".to_string(),
                artifact_sha256: A64.to_string(),
            },
            EvaluationMethodClass::Dft,
            EvaluationOrigin::LocalReproduction,
            "dft-pbe".to_string(),
            ConditionedPropertyObservation {
                subject_identity: "material-subject:v1|fixture".to_string(),
                property_id: "formation_energy".to_string(),
                value,
                unit: "eV/atom".to_string(),
                uncertainty: PropertyUncertainty::Standard { sigma: 0.01 },
                conditions: PropertyConditions {
                    temperature_k: Some(1.0),
                    ..Default::default()
                },
                method: PropertyObservationMethod::Calculation {
                    method_id: "DFT-PBE".to_string(),
                    code_id: "qe".to_string(),
                    input_sha256: B64.to_string(),
                    output_sha256: C64.to_string(),
                },
                artifact: artifact("normalized-result", D64),
            },
            ApplicabilityAssessment {
                domain_id: "dft-domain-v1".to_string(),
                state: ApplicabilityState::InDomain,
                score: None,
                artifact: None,
            },
            None,
            vec![artifact("structure-input", B64)],
            EvaluationResourceCost {
                compute_core_hours: 8.0,
                wall_time_hours: 1.0,
                material_mass_kg: 0.0,
                direct_cost: None,
                currency: None,
            },
        )
        .unwrap()
    }

    fn surrogate(value: f64) -> MultiFidelityEvaluation {
        MultiFidelityEvaluation::new(
            EvaluatorRef {
                evaluator_id: "ml-fixture".to_string(),
                version: "1".to_string(),
                artifact_sha256: A64.to_string(),
            },
            EvaluationMethodClass::Surrogate,
            EvaluationOrigin::SurrogatePrediction,
            "ml-fixture".to_string(),
            ConditionedPropertyObservation {
                subject_identity: "material-subject:v1|fixture".to_string(),
                property_id: "formation_energy".to_string(),
                value,
                unit: "eV/atom".to_string(),
                uncertainty: PropertyUncertainty::Unknown,
                conditions: PropertyConditions {
                    temperature_k: Some(1.0),
                    ..Default::default()
                },
                method: PropertyObservationMethod::ModelPrediction {
                    model_id: "ml-fixture".to_string(),
                    model_sha256: A64.to_string(),
                    applicability_domain_id: Some("ml-domain-v1".to_string()),
                },
                artifact: artifact("prediction", D64),
            },
            ApplicabilityAssessment {
                domain_id: "ml-domain-v1".to_string(),
                state: ApplicabilityState::InDomain,
                score: Some(0.2),
                artifact: Some(artifact("ood-basis", C64)),
            },
            None,
            vec![artifact("structure-input", B64)],
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

    #[test]
    fn seal_survives_serialization_roundtrip() {
        let evaluation = local_dft(-0.2);
        let seal = MultiFidelityEvaluationSeal::from_evaluation(&evaluation).unwrap();
        let serialized = serde_json::to_string(&seal).unwrap();
        let restored: MultiFidelityEvaluationSeal = serde_json::from_str(&serialized).unwrap();
        restored.validate_against(&evaluation).unwrap();
    }

    #[test]
    fn numeric_value_mutation_breaks_full_semantic_seal() {
        let evaluation = local_dft(-0.2);
        let seal = MultiFidelityEvaluationSeal::from_evaluation(&evaluation).unwrap();
        let mut mutated = evaluation.clone();
        mutated.observation.value = -0.1;
        assert!(seal.validate_against(&mutated).is_err());
    }

    #[test]
    fn applicability_mutation_breaks_full_semantic_seal() {
        let evaluation = local_dft(-0.2);
        let seal = MultiFidelityEvaluationSeal::from_evaluation(&evaluation).unwrap();
        let mut mutated = evaluation.clone();
        mutated.applicability.state = ApplicabilityState::OutOfDomain;
        assert!(seal.validate_against(&mutated).is_err());
    }

    #[test]
    fn resource_cost_mutation_breaks_full_semantic_seal() {
        let evaluation = local_dft(-0.2);
        let seal = MultiFidelityEvaluationSeal::from_evaluation(&evaluation).unwrap();
        let mut mutated = evaluation.clone();
        mutated.cost.compute_core_hours = 80.0;
        assert!(seal.validate_against(&mutated).is_err());
    }

    #[test]
    fn input_artifact_mutation_breaks_full_semantic_seal() {
        let evaluation = local_dft(-0.2);
        let seal = MultiFidelityEvaluationSeal::from_evaluation(&evaluation).unwrap();
        let mut mutated = evaluation.clone();
        mutated.input_artifacts[0].artifact_sha256 = C64.to_string();
        assert!(seal.validate_against(&mutated).is_err());
    }

    #[test]
    fn exact_input_artifact_order_is_semantically_irrelevant() {
        let mut evaluation = local_dft(-0.2);
        evaluation.input_artifacts.push(artifact("second-input", C64));
        let seal = MultiFidelityEvaluationSeal::from_evaluation(&evaluation).unwrap();
        evaluation.input_artifacts.reverse();
        seal.validate_against(&evaluation).unwrap();
    }

    #[test]
    fn numerically_equal_surrogate_and_local_dft_remain_distinct() {
        let dft = local_dft(-0.2);
        let model = surrogate(-0.2);
        let dft_seal = MultiFidelityEvaluationSeal::from_evaluation(&dft).unwrap();
        let model_seal = MultiFidelityEvaluationSeal::from_evaluation(&model).unwrap();
        assert_ne!(dft_seal.semantic_identity, model_seal.semantic_identity);
    }
}
