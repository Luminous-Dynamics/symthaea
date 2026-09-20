// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Authority firewall between exploratory materials intelligence and scientific evaluation.
//!
//! Advisory predictions may propose, rank, or prioritize material candidates. They are
//! deliberately not MAT-008 conditioned-property observations and cannot create a
//! scientific-evaluation reference. The scientific side of this boundary requires an
//! exact MAT-011 evaluation plus a MAT-011B full-semantic seal that validates against it.
//!
//! ```text
//! advisory prediction -> acquisition feature -> evaluation request
//!                                      != scientific evidence
//!
//! MAT-011 evaluation + valid MAT-011B seal -> ScientificEvaluationRef
//!                                      != evidence-stage promotion by itself
//! ```

use crate::conditioned_property::PropertyArtifactRef;
use crate::evaluation_seal::{EvaluationSealError, MultiFidelityEvaluationSeal};
use crate::multi_fidelity::{
    EvaluationMethodClass, EvaluationOrigin, MultiFidelityError, MultiFidelityEvaluation,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

const SHA256_HEX_LEN: usize = 64;

/// How an advisory prediction was produced.
///
/// These classes are candidate-generation metadata only. None is scientific evidence authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdvisoryPredictionKind {
    /// Hyperdimensional similarity or associative retrieval.
    HdcSimilarity,
    /// Hand-authored or algorithmic heuristic/rule system.
    Heuristic,
    /// Broad engineering estimate or preset-based approximation.
    EngineeringEstimate,
    /// Learned/statistical model used before the MAT-011 evidence boundary.
    LearnedModel,
    /// Explicit additional advisory family.
    Other(String),
}

/// Provenance for an advisory-only prediction.
///
/// This is intentionally distinct from MAT-008's scientific-property artifact semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdvisorySourceRef {
    /// Stable source/model/algorithm identifier.
    pub source_id: String,
    /// Exact source/model/configuration artifact SHA-256.
    pub artifact_sha256: String,
}

/// Exploratory property-like prediction that may inform search but carries no scientific authority.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdvisoryMaterialPrediction {
    /// Deterministic advisory-record identity.
    pub advisory_id: String,
    /// Exact MAT-007 subject identity or proposed-subject identity.
    pub subject_identity: String,
    /// Property/feature identifier being estimated.
    pub property_id: String,
    /// Advisory numeric estimate.
    pub value: f64,
    /// Explicit unit for the advisory estimate.
    pub unit: String,
    /// Optional dimensionless advisory score in [0,1]. This is not quantitative uncertainty.
    pub advisory_score: Option<f64>,
    /// Advisory prediction family.
    pub kind: AdvisoryPredictionKind,
    /// Exact advisory source/model provenance.
    pub source: AdvisorySourceRef,
}

impl AdvisoryMaterialPrediction {
    /// Construct and validate an advisory prediction, deriving its deterministic identity.
    pub fn new(
        subject_identity: String,
        property_id: String,
        value: f64,
        unit: String,
        advisory_score: Option<f64>,
        kind: AdvisoryPredictionKind,
        source: AdvisorySourceRef,
    ) -> Result<Self, AuthorityFirewallError> {
        let mut result = Self {
            advisory_id: String::new(),
            subject_identity,
            property_id,
            value,
            unit,
            advisory_score,
            kind,
            source,
        };
        result.validate_without_id()?;
        result.advisory_id = result.derived_identity();
        Ok(result)
    }

    /// Validate stored content and deterministic identity.
    pub fn validate(&self) -> Result<(), AuthorityFirewallError> {
        self.validate_without_id()?;
        if self.advisory_id != self.derived_identity() {
            return Err(AuthorityFirewallError::AdvisoryIdentityMismatch);
        }
        Ok(())
    }

    /// Produce an acquisition/search feature without changing scientific authority.
    pub fn as_acquisition_feature(&self) -> Result<AcquisitionFeature, AuthorityFirewallError> {
        self.validate()?;
        Ok(AcquisitionFeature {
            advisory_id: self.advisory_id.clone(),
            subject_identity: self.subject_identity.clone(),
            feature_id: self.property_id.clone(),
            value: self.value,
            unit: self.unit.clone(),
            advisory_score: self.advisory_score,
        })
    }

    fn validate_without_id(&self) -> Result<(), AuthorityFirewallError> {
        nonempty("subject_identity", &self.subject_identity)?;
        nonempty("property_id", &self.property_id)?;
        nonempty("unit", &self.unit)?;
        finite("advisory value", self.value)?;
        if let Some(score) = self.advisory_score {
            finite("advisory_score", score)?;
            if !(0.0..=1.0).contains(&score) {
                return Err(AuthorityFirewallError::AdvisoryScoreOutOfRange(score));
            }
        }
        validate_source(&self.source)?;
        if let AdvisoryPredictionKind::Other(name) = &self.kind {
            nonempty("advisory kind", name)?;
        }
        Ok(())
    }

    fn derived_identity(&self) -> String {
        format!(
            "materials-advisory:v1|subject={}|property={}|value={}|unit={}|score={}|kind={}|source={}:{}",
            token(&self.subject_identity),
            token(&self.property_id),
            float_key(self.value),
            token(&self.unit),
            optional_float_key(self.advisory_score),
            advisory_kind_key(&self.kind),
            token(&self.source.source_id),
            self.source.artifact_sha256.to_ascii_lowercase(),
        )
    }
}

/// Search/acquisition feature derived from advisory intelligence.
///
/// This type is intentionally not accepted by MAT-008 or the scientific-evaluation reference API.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcquisitionFeature {
    /// Exact advisory record from which this feature was derived.
    pub advisory_id: String,
    /// Subject this feature refers to.
    pub subject_identity: String,
    /// Stable feature/property identifier.
    pub feature_id: String,
    /// Advisory feature value.
    pub value: f64,
    /// Explicit unit.
    pub unit: String,
    /// Optional advisory score; never scientific uncertainty.
    pub advisory_score: Option<f64>,
}

impl AcquisitionFeature {
    /// Validate the acquisition feature as advisory-only data.
    pub fn validate(&self) -> Result<(), AuthorityFirewallError> {
        nonempty("advisory_id", &self.advisory_id)?;
        nonempty("subject_identity", &self.subject_identity)?;
        nonempty("feature_id", &self.feature_id)?;
        nonempty("unit", &self.unit)?;
        finite("acquisition feature value", self.value)?;
        if let Some(score) = self.advisory_score {
            finite("advisory_score", score)?;
            if !(0.0..=1.0).contains(&score) {
                return Err(AuthorityFirewallError::AdvisoryScoreOutOfRange(score));
            }
        }
        Ok(())
    }
}

/// Exact sealed reference to one MAT-011 scientific evaluation.
///
/// Construction requires the complete evaluation object and a MAT-011B seal that validates
/// against it. This reference still does not grant a materials evidence stage: MAT-016B must
/// separately prove that the evaluation's method, origin, property, and capability justify the
/// requested stage transition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificEvaluationRef {
    /// MAT-011 record identifier.
    pub evaluation_id: String,
    /// Full-semantic MAT-011B identity.
    pub semantic_identity: String,
    /// Exact subject identity.
    pub subject_identity: String,
    /// Exact property identifier.
    pub property_id: String,
    /// Evaluation origin retained for capability admission.
    pub origin: EvaluationOrigin,
    /// Method class retained for capability admission.
    pub method_class: EvaluationMethodClass,
    /// Exact normalized result/source artifact.
    pub result_artifact: PropertyArtifactRef,
}

impl ScientificEvaluationRef {
    /// Create a reference only from an exact evaluation with a matching full-semantic seal.
    pub fn from_sealed_evaluation(
        evaluation: &MultiFidelityEvaluation,
        seal: &MultiFidelityEvaluationSeal,
    ) -> Result<Self, AuthorityFirewallError> {
        evaluation.validate()?;
        seal.validate_against(evaluation)?;
        Ok(Self {
            evaluation_id: evaluation.evaluation_id.clone(),
            semantic_identity: seal.semantic_identity.clone(),
            subject_identity: evaluation.observation.subject_identity.clone(),
            property_id: evaluation.observation.property_id.clone(),
            origin: evaluation.origin,
            method_class: evaluation.method_class.clone(),
            result_artifact: evaluation.observation.artifact.clone(),
        })
    }

    /// Revalidate this stored reference against the complete evaluation and seal.
    pub fn validate_against(
        &self,
        evaluation: &MultiFidelityEvaluation,
        seal: &MultiFidelityEvaluationSeal,
    ) -> Result<(), AuthorityFirewallError> {
        evaluation.validate()?;
        seal.validate_against(evaluation)?;
        let expected = Self::from_sealed_evaluation(evaluation, seal)?;
        if self != &expected {
            return Err(AuthorityFirewallError::ScientificReferenceMismatch);
        }
        Ok(())
    }
}

fn validate_source(source: &AdvisorySourceRef) -> Result<(), AuthorityFirewallError> {
    nonempty("advisory source_id", &source.source_id)?;
    sha256(&source.artifact_sha256)
}

fn sha256(value: &str) -> Result<(), AuthorityFirewallError> {
    if value.len() != SHA256_HEX_LEN || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Err(AuthorityFirewallError::InvalidSha256)
    } else {
        Ok(())
    }
}

fn nonempty(field: &'static str, value: &str) -> Result<(), AuthorityFirewallError> {
    if value.trim().is_empty() {
        Err(AuthorityFirewallError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn finite(field: &'static str, value: f64) -> Result<(), AuthorityFirewallError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(AuthorityFirewallError::NonFiniteValue { field, value })
    }
}

fn advisory_kind_key(kind: &AdvisoryPredictionKind) -> String {
    match kind {
        AdvisoryPredictionKind::HdcSimilarity => "hdc-similarity".to_string(),
        AdvisoryPredictionKind::Heuristic => "heuristic".to_string(),
        AdvisoryPredictionKind::EngineeringEstimate => "engineering-estimate".to_string(),
        AdvisoryPredictionKind::LearnedModel => "learned-model".to_string(),
        AdvisoryPredictionKind::Other(value) => format!("other:{}", token(value)),
    }
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

/// Validation failure at the advisory/scientific authority firewall.
#[derive(Debug, Clone, PartialEq)]
pub enum AuthorityFirewallError {
    /// Required text field was empty.
    EmptyField(&'static str),
    /// Numeric value was NaN or infinite.
    NonFiniteValue {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// Advisory score was outside [0,1].
    AdvisoryScoreOutOfRange(f64),
    /// SHA-256 binding was malformed.
    InvalidSha256,
    /// Stored advisory identity did not match its exact advisory semantics.
    AdvisoryIdentityMismatch,
    /// Stored scientific reference did not match the exact sealed evaluation.
    ScientificReferenceMismatch,
    /// Underlying MAT-011 evaluation validation failed.
    Evaluation(MultiFidelityError),
    /// MAT-011B semantic-seal validation failed.
    Seal(EvaluationSealError),
}

impl fmt::Display for AuthorityFirewallError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(formatter, "empty authority-firewall field: {field}"),
            Self::NonFiniteValue { field, value } => {
                write!(formatter, "non-finite authority-firewall value {field}={value}")
            }
            Self::AdvisoryScoreOutOfRange(value) => {
                write!(formatter, "advisory score outside [0,1]: {value}")
            }
            Self::InvalidSha256 => formatter.write_str("invalid SHA-256 binding"),
            Self::AdvisoryIdentityMismatch => {
                formatter.write_str("stored advisory identity does not match advisory semantics")
            }
            Self::ScientificReferenceMismatch => formatter.write_str(
                "stored scientific evaluation reference does not match exact sealed evaluation",
            ),
            Self::Evaluation(error) => write!(formatter, "invalid MAT-011 evaluation: {error:?}"),
            Self::Seal(error) => write!(formatter, "invalid MAT-011B evaluation seal: {error}"),
        }
    }
}

impl Error for AuthorityFirewallError {}

impl From<MultiFidelityError> for AuthorityFirewallError {
    fn from(value: MultiFidelityError) -> Self {
        Self::Evaluation(value)
    }
}

impl From<EvaluationSealError> for AuthorityFirewallError {
    fn from(value: EvaluationSealError) -> Self {
        Self::Seal(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conditioned_property::{
        ConditionedPropertyObservation, PropertyConditions, PropertyObservationMethod,
        PropertyUncertainty,
    };
    use crate::evaluation_seal::MultiFidelityEvaluationSeal;
    use crate::multi_fidelity::{
        ApplicabilityAssessment, ApplicabilityState, EvaluationResourceCost, EvaluatorRef,
    };

    const A64: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B64: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C64: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn artifact(source: &str, hash: &str) -> PropertyArtifactRef {
        PropertyArtifactRef {
            source_id: source.to_string(),
            artifact_sha256: hash.to_string(),
        }
    }

    fn advisory(value: f64) -> AdvisoryMaterialPrediction {
        AdvisoryMaterialPrediction::new(
            "material-subject:v1|fixture".to_string(),
            "formation_energy".to_string(),
            value,
            "eV/atom".to_string(),
            Some(0.9),
            AdvisoryPredictionKind::HdcSimilarity,
            AdvisorySourceRef {
                source_id: "hdc-screen-v1".to_string(),
                artifact_sha256: A64.to_string(),
            },
        )
        .unwrap()
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
                artifact: artifact("dft-result", C64),
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

    #[test]
    fn advisory_prediction_roundtrips_without_becoming_scientific_evidence() {
        let prediction = advisory(-0.2);
        let json = serde_json::to_string(&prediction).unwrap();
        let restored: AdvisoryMaterialPrediction = serde_json::from_str(&json).unwrap();
        restored.validate().unwrap();
        assert_eq!(prediction, restored);
    }

    #[test]
    fn acquisition_feature_is_advisory_projection_only() {
        let prediction = advisory(-0.2);
        let feature = prediction.as_acquisition_feature().unwrap();
        feature.validate().unwrap();
        assert_eq!(feature.advisory_id, prediction.advisory_id);
        assert_eq!(feature.advisory_score, Some(0.9));
    }

    #[test]
    fn scientific_reference_requires_matching_full_semantic_seal() {
        let evaluation = local_dft(-0.2);
        let seal = MultiFidelityEvaluationSeal::from_evaluation(&evaluation).unwrap();
        let reference = ScientificEvaluationRef::from_sealed_evaluation(&evaluation, &seal).unwrap();
        reference.validate_against(&evaluation, &seal).unwrap();
    }

    #[test]
    fn scientific_reference_rejects_mutated_evaluation_even_when_old_record_id_survives() {
        let evaluation = local_dft(-0.2);
        let seal = MultiFidelityEvaluationSeal::from_evaluation(&evaluation).unwrap();
        let reference = ScientificEvaluationRef::from_sealed_evaluation(&evaluation, &seal).unwrap();
        let mut mutated = evaluation.clone();
        mutated.applicability.state = ApplicabilityState::OutOfDomain;
        assert!(reference.validate_against(&mutated, &seal).is_err());
    }

    #[test]
    fn equal_advisory_and_dft_numbers_remain_different_authority_types() {
        let prediction = advisory(-0.2);
        let evaluation = local_dft(-0.2);
        let seal = MultiFidelityEvaluationSeal::from_evaluation(&evaluation).unwrap();
        let reference = ScientificEvaluationRef::from_sealed_evaluation(&evaluation, &seal).unwrap();
        assert_eq!(prediction.value, evaluation.observation.value);
        assert_ne!(prediction.advisory_id, reference.semantic_identity);
    }

    #[test]
    fn advisory_score_is_not_property_uncertainty() {
        let prediction = advisory(-0.2);
        assert_eq!(prediction.advisory_score, Some(0.9));
        let evaluation = local_dft(-0.2);
        assert!(matches!(
            evaluation.observation.uncertainty,
            PropertyUncertainty::Standard { .. }
        ));
    }
}
