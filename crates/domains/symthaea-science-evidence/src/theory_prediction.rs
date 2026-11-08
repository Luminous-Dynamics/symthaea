// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit quantitative prediction coverage and pairwise comparability.
//!
//! Missing predictions are never converted into zeroes, poor fits, or
//! artificial likelihoods. Pairwise normalized separation is only reported
//! when both models predict the same operational observable in the same
//! declared domain and the measurement noise is calibrated as Gaussian.
//! This is a screening diagnostic, not EIG, EVSI, a Bayes factor, or evidence
//! that either theory is true.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

pub const THEORY_PREDICTION_SCHEMA: &str = "symthaea-theory-prediction-v1";
pub const OTHER_UNRESOLVED_MODEL_ID: &str = "OTHER_UNRESOLVED";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum PredictionDisposition {
    Quantitative,
    QualitativeOnly,
    NotDerived,
    OutsideValidatedDomain,
    Unresolved,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum TheoryModelRole {
    NamedTheory,
    /// An explicit open-world state. No arbitrary predictive likelihood is
    /// assigned by this ledger.
    OtherUnresolved,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum MeasurementNoiseModel {
    Gaussian,
    NonGaussian,
    Unspecified,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum NoiseCalibrationStatus {
    SyntheticKnownAnswer,
    EmpiricallyCalibrated,
    Uncalibrated,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TheoryModel {
    pub theory_id: String,
    pub family: String,
    /// Content-addressed model equations/assumptions/specification artifact.
    pub model_spec_artifact: super::ArtifactDigest,
    /// Full source commit associated with this model version.
    pub source_revision_sha: String,
    pub role: TheoryModelRole,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservableContract {
    pub observable_id: String,
    pub units: String,
    pub operational_definition: String,
    pub domain_id: String,
    pub noise_model: MeasurementNoiseModel,
    /// One-sigma measurement uncertainty in the declared units.
    pub measurement_sigma: Option<f64>,
    pub noise_calibration: NoiseCalibrationStatus,
    pub calibration_reference: Option<String>,
}

/// One explicit theory/observable prediction status. A numeric value is
/// permitted only for Quantitative records.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TheoryPredictionRecord {
    pub theory_id: String,
    pub observable_id: String,
    pub disposition: PredictionDisposition,
    pub scope_id: String,
    pub units: String,
    pub predicted_value: Option<f64>,
    pub derivation_reference: Option<String>,
    pub notes: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TheoryPredictionLedger {
    pub schema: String,
    pub models: Vec<TheoryModel>,
    pub observables: Vec<ObservableContract>,
    /// Exactly one record is required for every model/observable pair.
    pub predictions: Vec<TheoryPredictionRecord>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct PredictionCoverage {
    pub total_model_observable_pairs: usize,
    pub quantitative: usize,
    pub qualitative_only: usize,
    pub not_derived: usize,
    pub outside_validated_domain: usize,
    pub unresolved: usize,
}

impl PredictionCoverage {
    pub fn quantitative_fraction(&self) -> f64 {
        if self.total_model_observable_pairs == 0 {
            0.0
        } else {
            self.quantitative as f64 / self.total_model_observable_pairs as f64
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ComparisonBlocker {
    LeftNotQuantitative(PredictionDisposition),
    RightNotQuantitative(PredictionDisposition),
    LeftDomainMismatch,
    RightDomainMismatch,
    UnitsMismatch,
    NonGaussianNoise,
    NoiseNotCalibrated,
    MissingMeasurementSigma,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ModelPredictionComparison {
    Comparable {
        left_theory_id: String,
        right_theory_id: String,
        observable_id: String,
        left_predicted_value: f64,
        right_predicted_value: f64,
        absolute_difference: f64,
        measurement_sigma: f64,
        normalized_separation: f64,
    },
    NotComparable {
        left_theory_id: String,
        right_theory_id: String,
        observable_id: String,
        blockers: Vec<ComparisonBlocker>,
    },
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum TheoryPredictionError {
    #[error("unsupported theory-prediction schema: {0}")]
    UnsupportedSchema(String),
    #[error("{0} must not be empty")]
    EmptyField(&'static str),
    #[error("theory model list must not be empty")]
    NoModels,
    #[error("observable contract list must not be empty")]
    NoObservables,
    #[error("theory prediction matrix must not be empty")]
    NoPredictions,
    #[error("invalid theory model ID at index {0}")]
    InvalidModelId(usize),
    #[error("duplicate theory model ID: {0}")]
    DuplicateModelId(String),
    #[error("OTHER_UNRESOLVED must use the OtherUnresolved role")]
    InvalidUnresolvedModelRole,
    #[error("exactly one OTHER_UNRESOLVED model is required")]
    MissingOrDuplicateUnresolvedModel,
    #[error("model {theory_id} has an invalid or empty specification artifact")]
    InvalidModelSpecificationArtifact { theory_id: String },
    #[error("model {theory_id} has an invalid full source commit SHA")]
    InvalidModelSourceRevision { theory_id: String },
    #[error("invalid observable ID at index {0}")]
    InvalidObservableId(usize),
    #[error("duplicate observable ID: {0}")]
    DuplicateObservableId(String),
    #[error("measurement sigma for observable {observable_id} must be finite and positive")]
    InvalidMeasurementSigma { observable_id: String },
    #[error("Gaussian observable {observable_id} requires a measurement sigma")]
    GaussianNoiseRequiresSigma { observable_id: String },
    #[error("observable {observable_id} has no calibration reference")]
    MissingCalibrationReference { observable_id: String },
    #[error("prediction references unknown theory model: {0}")]
    UnknownTheoryModel(String),
    #[error("prediction references unknown observable: {0}")]
    UnknownObservable(String),
    #[error("duplicate prediction for theory {theory_id}, observable {observable_id}")]
    DuplicatePrediction {
        theory_id: String,
        observable_id: String,
    },
    #[error("missing prediction status for theory {theory_id}, observable {observable_id}")]
    MissingPrediction {
        theory_id: String,
        observable_id: String,
    },
    #[error("quantitative prediction for {theory_id}/{observable_id} has no finite value")]
    QuantitativePredictionMissingValue {
        theory_id: String,
        observable_id: String,
    },
    #[error(
        "non-quantitative prediction for {theory_id}/{observable_id} must not carry a numeric value"
    )]
    NonQuantitativePredictionHasValue {
        theory_id: String,
        observable_id: String,
    },
    #[error("quantitative prediction for {theory_id}/{observable_id} has no derivation reference")]
    QuantitativePredictionMissingDerivation {
        theory_id: String,
        observable_id: String,
    },
    #[error(
        "quantitative prediction for {theory_id}/{observable_id} uses units different from its observable"
    )]
    PredictionUnitsMismatch {
        theory_id: String,
        observable_id: String,
    },
    #[error(
        "quantitative prediction for {theory_id}/{observable_id} is outside the declared observable domain"
    )]
    QuantitativePredictionOutsideDomain {
        theory_id: String,
        observable_id: String,
    },
    #[error("OTHER_UNRESOLVED must remain explicitly unresolved for observable {observable_id}")]
    UnresolvedModelMustNotInventPrediction { observable_id: String },
    #[error("theory model not found: {0}")]
    ModelNotFound(String),
    #[error("observable not found: {0}")]
    ObservableNotFound(String),
    #[error("cannot compare a theory with itself: {0}")]
    SelfComparison(String),
}

impl TheoryPredictionLedger {
    pub fn validate(&self) -> Result<(), TheoryPredictionError> {
        if self.schema != THEORY_PREDICTION_SCHEMA {
            return Err(TheoryPredictionError::UnsupportedSchema(
                self.schema.clone(),
            ));
        }
        if self.models.is_empty() {
            return Err(TheoryPredictionError::NoModels);
        }
        if self.observables.is_empty() {
            return Err(TheoryPredictionError::NoObservables);
        }
        if self.predictions.is_empty() {
            return Err(TheoryPredictionError::NoPredictions);
        }

        let mut model_ids = BTreeSet::new();
        let mut unresolved_count = 0usize;
        for (index, model) in self.models.iter().enumerate() {
            if model.theory_id.trim().is_empty() {
                return Err(TheoryPredictionError::InvalidModelId(index));
            }
            require_nonempty("theory model family", &model.family)?;
            if !model_ids.insert(model.theory_id.as_str()) {
                return Err(TheoryPredictionError::DuplicateModelId(
                    model.theory_id.clone(),
                ));
            }
            if model.model_spec_artifact.validate().is_err() {
                return Err(TheoryPredictionError::InvalidModelSpecificationArtifact {
                    theory_id: model.theory_id.clone(),
                });
            }
            if !is_git_sha(&model.source_revision_sha) {
                return Err(TheoryPredictionError::InvalidModelSourceRevision {
                    theory_id: model.theory_id.clone(),
                });
            }
            match model.role {
                TheoryModelRole::NamedTheory if model.theory_id == OTHER_UNRESOLVED_MODEL_ID => {
                    return Err(TheoryPredictionError::InvalidUnresolvedModelRole);
                }
                TheoryModelRole::OtherUnresolved => {
                    if model.theory_id != OTHER_UNRESOLVED_MODEL_ID {
                        return Err(TheoryPredictionError::InvalidUnresolvedModelRole);
                    }
                    unresolved_count += 1;
                }
                TheoryModelRole::NamedTheory => {}
            }
        }
        if unresolved_count != 1 {
            return Err(TheoryPredictionError::MissingOrDuplicateUnresolvedModel);
        }

        let mut observable_ids = BTreeSet::new();
        for (index, observable) in self.observables.iter().enumerate() {
            if observable.observable_id.trim().is_empty() {
                return Err(TheoryPredictionError::InvalidObservableId(index));
            }
            require_nonempty("observable units", &observable.units)?;
            require_nonempty(
                "observable operational definition",
                &observable.operational_definition,
            )?;
            require_nonempty("observable domain ID", &observable.domain_id)?;
            if !observable_ids.insert(observable.observable_id.as_str()) {
                return Err(TheoryPredictionError::DuplicateObservableId(
                    observable.observable_id.clone(),
                ));
            }
            if let Some(sigma) = observable.measurement_sigma {
                if !sigma.is_finite() || sigma <= 0.0 {
                    return Err(TheoryPredictionError::InvalidMeasurementSigma {
                        observable_id: observable.observable_id.clone(),
                    });
                }
            }
            if observable.noise_model == MeasurementNoiseModel::Gaussian
                && observable.measurement_sigma.is_none()
            {
                return Err(TheoryPredictionError::GaussianNoiseRequiresSigma {
                    observable_id: observable.observable_id.clone(),
                });
            }
            if observable.noise_calibration != NoiseCalibrationStatus::Uncalibrated {
                let reference = observable
                    .calibration_reference
                    .as_deref()
                    .unwrap_or_default();
                require_nonempty("noise calibration reference", reference).map_err(|_| {
                    TheoryPredictionError::MissingCalibrationReference {
                        observable_id: observable.observable_id.clone(),
                    }
                })?;
            }
        }

        let mut observed = BTreeSet::new();
        for record in &self.predictions {
            let model = self
                .models
                .iter()
                .find(|item| item.theory_id == record.theory_id)
                .ok_or_else(|| {
                    TheoryPredictionError::UnknownTheoryModel(record.theory_id.clone())
                })?;
            let observable = self
                .observables
                .iter()
                .find(|item| item.observable_id == record.observable_id)
                .ok_or_else(|| {
                    TheoryPredictionError::UnknownObservable(record.observable_id.clone())
                })?;
            let key = (record.theory_id.as_str(), record.observable_id.as_str());
            if !observed.insert(key) {
                return Err(TheoryPredictionError::DuplicatePrediction {
                    theory_id: record.theory_id.clone(),
                    observable_id: record.observable_id.clone(),
                });
            }
            require_nonempty("prediction scope ID", &record.scope_id)?;
            require_nonempty("prediction units", &record.units)?;
            require_nonempty("prediction notes", &record.notes)?;

            if model.role == TheoryModelRole::OtherUnresolved {
                if record.disposition != PredictionDisposition::Unresolved
                    || record.predicted_value.is_some()
                    || record.derivation_reference.is_some()
                {
                    return Err(
                        TheoryPredictionError::UnresolvedModelMustNotInventPrediction {
                            observable_id: record.observable_id.clone(),
                        },
                    );
                }
                continue;
            }

            match record.disposition {
                PredictionDisposition::Quantitative => {
                    let value = record.predicted_value.ok_or_else(|| {
                        TheoryPredictionError::QuantitativePredictionMissingValue {
                            theory_id: record.theory_id.clone(),
                            observable_id: record.observable_id.clone(),
                        }
                    })?;
                    if !value.is_finite() {
                        return Err(TheoryPredictionError::QuantitativePredictionMissingValue {
                            theory_id: record.theory_id.clone(),
                            observable_id: record.observable_id.clone(),
                        });
                    }
                    if record
                        .derivation_reference
                        .as_deref()
                        .unwrap_or_default()
                        .trim()
                        .is_empty()
                    {
                        return Err(
                            TheoryPredictionError::QuantitativePredictionMissingDerivation {
                                theory_id: record.theory_id.clone(),
                                observable_id: record.observable_id.clone(),
                            },
                        );
                    }
                    if record.units != observable.units {
                        return Err(TheoryPredictionError::PredictionUnitsMismatch {
                            theory_id: record.theory_id.clone(),
                            observable_id: record.observable_id.clone(),
                        });
                    }
                    if record.scope_id != observable.domain_id {
                        return Err(TheoryPredictionError::QuantitativePredictionOutsideDomain {
                            theory_id: record.theory_id.clone(),
                            observable_id: record.observable_id.clone(),
                        });
                    }
                }
                _ => {
                    if record.predicted_value.is_some() {
                        return Err(TheoryPredictionError::NonQuantitativePredictionHasValue {
                            theory_id: record.theory_id.clone(),
                            observable_id: record.observable_id.clone(),
                        });
                    }
                    if record.disposition == PredictionDisposition::QualitativeOnly
                        && record
                            .derivation_reference
                            .as_deref()
                            .unwrap_or_default()
                            .trim()
                            .is_empty()
                    {
                        return Err(
                            TheoryPredictionError::QuantitativePredictionMissingDerivation {
                                theory_id: record.theory_id.clone(),
                                observable_id: record.observable_id.clone(),
                            },
                        );
                    }
                }
            }
        }

        let expected = self.models.len() * self.observables.len();
        if observed.len() != expected {
            for model in &self.models {
                for observable in &self.observables {
                    if !observed
                        .contains(&(model.theory_id.as_str(), observable.observable_id.as_str()))
                    {
                        return Err(TheoryPredictionError::MissingPrediction {
                            theory_id: model.theory_id.clone(),
                            observable_id: observable.observable_id.clone(),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Report how much of the theory/observable matrix is quantitative.
    /// Coverage is not evidence that the predictions are correct.
    pub fn coverage(&self) -> Result<PredictionCoverage, TheoryPredictionError> {
        self.validate()?;
        let mut report = PredictionCoverage {
            total_model_observable_pairs: self.models.len() * self.observables.len(),
            ..PredictionCoverage::default()
        };
        for prediction in &self.predictions {
            match prediction.disposition {
                PredictionDisposition::Quantitative => report.quantitative += 1,
                PredictionDisposition::QualitativeOnly => report.qualitative_only += 1,
                PredictionDisposition::NotDerived => report.not_derived += 1,
                PredictionDisposition::OutsideValidatedDomain => {
                    report.outside_validated_domain += 1
                }
                PredictionDisposition::Unresolved => report.unresolved += 1,
            }
        }
        Ok(report)
    }

    /// Compare two models on the same observable. The result is eligible only
    /// for two quantitative predictions and calibrated Gaussian measurement
    /// noise. The separation in measurement-sigma units is not an EIG or Bayes
    /// factor; downstream likelihood construction is a separate step.
    pub fn compare_models(
        &self,
        left_theory_id: &str,
        right_theory_id: &str,
        observable_id: &str,
    ) -> Result<ModelPredictionComparison, TheoryPredictionError> {
        self.validate()?;
        if left_theory_id == right_theory_id {
            return Err(TheoryPredictionError::SelfComparison(
                left_theory_id.to_string(),
            ));
        }
        if !self
            .models
            .iter()
            .any(|model| model.theory_id == left_theory_id)
        {
            return Err(TheoryPredictionError::ModelNotFound(
                left_theory_id.to_string(),
            ));
        }
        if !self
            .models
            .iter()
            .any(|model| model.theory_id == right_theory_id)
        {
            return Err(TheoryPredictionError::ModelNotFound(
                right_theory_id.to_string(),
            ));
        }
        let observable = self
            .observables
            .iter()
            .find(|item| item.observable_id == observable_id)
            .ok_or_else(|| TheoryPredictionError::ObservableNotFound(observable_id.to_string()))?;
        let left = self
            .predictions
            .iter()
            .find(|record| {
                record.theory_id == left_theory_id && record.observable_id == observable_id
            })
            .ok_or_else(|| TheoryPredictionError::MissingPrediction {
                theory_id: left_theory_id.to_string(),
                observable_id: observable_id.to_string(),
            })?;
        let right = self
            .predictions
            .iter()
            .find(|record| {
                record.theory_id == right_theory_id && record.observable_id == observable_id
            })
            .ok_or_else(|| TheoryPredictionError::MissingPrediction {
                theory_id: right_theory_id.to_string(),
                observable_id: observable_id.to_string(),
            })?;

        let mut blockers = Vec::new();
        if left.disposition != PredictionDisposition::Quantitative {
            blockers.push(ComparisonBlocker::LeftNotQuantitative(left.disposition));
        }
        if right.disposition != PredictionDisposition::Quantitative {
            blockers.push(ComparisonBlocker::RightNotQuantitative(right.disposition));
        }
        if left.scope_id != observable.domain_id {
            blockers.push(ComparisonBlocker::LeftDomainMismatch);
        }
        if right.scope_id != observable.domain_id {
            blockers.push(ComparisonBlocker::RightDomainMismatch);
        }
        if left.units != observable.units || right.units != observable.units {
            blockers.push(ComparisonBlocker::UnitsMismatch);
        }
        if observable.noise_model != MeasurementNoiseModel::Gaussian {
            blockers.push(ComparisonBlocker::NonGaussianNoise);
        }
        if observable.noise_calibration == NoiseCalibrationStatus::Uncalibrated {
            blockers.push(ComparisonBlocker::NoiseNotCalibrated);
        }
        let Some(sigma) = observable.measurement_sigma else {
            blockers.push(ComparisonBlocker::MissingMeasurementSigma);
            return Ok(ModelPredictionComparison::NotComparable {
                left_theory_id: left_theory_id.to_string(),
                right_theory_id: right_theory_id.to_string(),
                observable_id: observable_id.to_string(),
                blockers,
            });
        };
        if !blockers.is_empty() {
            return Ok(ModelPredictionComparison::NotComparable {
                left_theory_id: left_theory_id.to_string(),
                right_theory_id: right_theory_id.to_string(),
                observable_id: observable_id.to_string(),
                blockers,
            });
        }

        let left_value = left
            .predicted_value
            .expect("validated quantitative prediction");
        let right_value = right
            .predicted_value
            .expect("validated quantitative prediction");
        let absolute_difference = (left_value - right_value).abs();
        Ok(ModelPredictionComparison::Comparable {
            left_theory_id: left_theory_id.to_string(),
            right_theory_id: right_theory_id.to_string(),
            observable_id: observable_id.to_string(),
            left_predicted_value: left_value,
            right_predicted_value: right_value,
            absolute_difference,
            measurement_sigma: sigma,
            normalized_separation: absolute_difference / sigma,
        })
    }
}

fn require_nonempty(field: &'static str, value: &str) -> Result<(), TheoryPredictionError> {
    if value.trim().is_empty() {
        Err(TheoryPredictionError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn is_git_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_lower_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha() -> String {
        "1111111111111111111111111111111111111111".to_string()
    }

    fn model(id: &str, role: TheoryModelRole, marker: u8) -> TheoryModel {
        let specification = format!("model-specification:{id}:version-1");
        TheoryModel {
            theory_id: id.to_string(),
            family: format!("family-{id}"),
            model_spec_artifact: super::super::ArtifactDigest::from_bytes(
                format!("tests/fixtures/model-spec-{marker:02x}.txt"),
                specification.as_bytes(),
            ),
            source_revision_sha: sha(),
            role,
        }
    }

    fn observable(
        id: &str,
        units: &str,
        domain: &str,
        noise_model: MeasurementNoiseModel,
        sigma: Option<f64>,
        calibration: NoiseCalibrationStatus,
    ) -> ObservableContract {
        ObservableContract {
            observable_id: id.to_string(),
            units: units.to_string(),
            operational_definition: format!("operational definition for {id}"),
            domain_id: domain.to_string(),
            noise_model,
            measurement_sigma: sigma,
            noise_calibration: calibration,
            calibration_reference: if calibration == NoiseCalibrationStatus::Uncalibrated {
                None
            } else {
                Some("fixture-noise-calibration-v1".to_string())
            },
        }
    }

    fn prediction(
        theory_id: &str,
        observable_id: &str,
        disposition: PredictionDisposition,
        scope_id: &str,
        units: &str,
        value: Option<f64>,
        derivation: Option<&str>,
    ) -> TheoryPredictionRecord {
        TheoryPredictionRecord {
            theory_id: theory_id.to_string(),
            observable_id: observable_id.to_string(),
            disposition,
            scope_id: scope_id.to_string(),
            units: units.to_string(),
            predicted_value: value,
            derivation_reference: derivation.map(str::to_string),
            notes: format!("{theory_id} status for {observable_id}"),
        }
    }

    fn fixture() -> TheoryPredictionLedger {
        let models = vec![
            model("causal-set", TheoryModelRole::NamedTheory, b'a'),
            model("minkowski-continuum", TheoryModelRole::NamedTheory, b'b'),
            model("chain-theory", TheoryModelRole::NamedTheory, b'c'),
            model(
                OTHER_UNRESOLVED_MODEL_ID,
                TheoryModelRole::OtherUnresolved,
                b'd',
            ),
        ];
        let observables = vec![
            observable(
                "ordering-fraction-1p1",
                "fraction",
                "poisson-sprinkling-minkowski-1p1",
                MeasurementNoiseModel::Gaussian,
                Some(0.01),
                NoiseCalibrationStatus::SyntheticKnownAnswer,
            ),
            observable(
                "inverse-alpha-candidate",
                "dimensionless",
                "low-energy-fine-structure-constant-definition",
                MeasurementNoiseModel::Unspecified,
                None,
                NoiseCalibrationStatus::Uncalibrated,
            ),
            observable(
                "photon-propagator",
                "declared-amplitude-units",
                "fixed-momentum-scattering-protocol",
                MeasurementNoiseModel::Unspecified,
                None,
                NoiseCalibrationStatus::Uncalibrated,
            ),
        ];
        let predictions = vec![
            prediction(
                "causal-set",
                "ordering-fraction-1p1",
                PredictionDisposition::Quantitative,
                "poisson-sprinkling-minkowski-1p1",
                "fraction",
                Some(0.5),
                Some("Myrheim-Meyer expected ordering fraction, d=2"),
            ),
            prediction(
                "minkowski-continuum",
                "ordering-fraction-1p1",
                PredictionDisposition::Quantitative,
                "poisson-sprinkling-minkowski-1p1",
                "fraction",
                Some(0.5),
                Some("continuum Alexandrov-interval ordering reference"),
            ),
            prediction(
                "chain-theory",
                "ordering-fraction-1p1",
                PredictionDisposition::NotDerived,
                "qudit-entanglement-network",
                "fraction",
                None,
                None,
            ),
            prediction(
                OTHER_UNRESOLVED_MODEL_ID,
                "ordering-fraction-1p1",
                PredictionDisposition::Unresolved,
                "open-world",
                "fraction",
                None,
                None,
            ),
            prediction(
                "chain-theory",
                "inverse-alpha-candidate",
                PredictionDisposition::Quantitative,
                "low-energy-fine-structure-constant-definition",
                "dimensionless",
                Some(137.035_999_189_566),
                Some("Chain Theory candidate readout; Photon Conjecture not inferred"),
            ),
            prediction(
                "causal-set",
                "inverse-alpha-candidate",
                PredictionDisposition::NotDerived,
                "causal-set-minkowski",
                "dimensionless",
                None,
                None,
            ),
            prediction(
                "minkowski-continuum",
                "inverse-alpha-candidate",
                PredictionDisposition::NotDerived,
                "standard-model-parameter-input",
                "dimensionless",
                None,
                None,
            ),
            prediction(
                OTHER_UNRESOLVED_MODEL_ID,
                "inverse-alpha-candidate",
                PredictionDisposition::Unresolved,
                "open-world",
                "dimensionless",
                None,
                None,
            ),
            prediction(
                "chain-theory",
                "photon-propagator",
                PredictionDisposition::NotDerived,
                "qudit-entanglement-network",
                "declared-amplitude-units",
                None,
                None,
            ),
            prediction(
                "causal-set",
                "photon-propagator",
                PredictionDisposition::OutsideValidatedDomain,
                "causal-set-link-propagator-not-implemented",
                "declared-amplitude-units",
                None,
                None,
            ),
            prediction(
                "minkowski-continuum",
                "photon-propagator",
                PredictionDisposition::QualitativeOnly,
                "fixed-momentum-scattering-protocol",
                "declared-amplitude-units",
                None,
                Some("QED contains a photon propagator; this observable protocol is not pinned"),
            ),
            prediction(
                OTHER_UNRESOLVED_MODEL_ID,
                "photon-propagator",
                PredictionDisposition::Unresolved,
                "open-world",
                "declared-amplitude-units",
                None,
                None,
            ),
        ];
        TheoryPredictionLedger {
            schema: THEORY_PREDICTION_SCHEMA.to_string(),
            models,
            observables,
            predictions,
        }
    }

    #[test]
    fn ledger_requires_complete_model_observable_matrix_and_explicit_unknown() {
        let mut ledger = fixture();
        assert!(ledger.validate().is_ok());
        ledger.predictions.pop();
        assert_eq!(
            ledger.validate(),
            Err(TheoryPredictionError::MissingPrediction {
                theory_id: OTHER_UNRESOLVED_MODEL_ID.to_string(),
                observable_id: "photon-propagator".to_string(),
            })
        );
        let mut ledger = fixture();
        ledger
            .models
            .retain(|model| model.role != TheoryModelRole::OtherUnresolved);
        assert_eq!(
            ledger.validate(),
            Err(TheoryPredictionError::MissingOrDuplicateUnresolvedModel)
        );
    }

    #[test]
    fn identical_quantitative_predictions_are_not_discriminating() {
        let comparison = fixture()
            .compare_models("causal-set", "minkowski-continuum", "ordering-fraction-1p1")
            .unwrap();
        assert_eq!(
            comparison,
            ModelPredictionComparison::Comparable {
                left_theory_id: "causal-set".to_string(),
                right_theory_id: "minkowski-continuum".to_string(),
                observable_id: "ordering-fraction-1p1".to_string(),
                left_predicted_value: 0.5,
                right_predicted_value: 0.5,
                absolute_difference: 0.0,
                measurement_sigma: 0.01,
                normalized_separation: 0.0,
            }
        );
    }

    #[test]
    fn missing_propagator_prediction_is_not_comparable_not_zero() {
        let comparison = fixture()
            .compare_models("chain-theory", "minkowski-continuum", "photon-propagator")
            .unwrap();
        let ModelPredictionComparison::NotComparable { blockers, .. } = comparison else {
            panic!("a missing prediction must block quantitative comparison");
        };
        assert!(blockers.contains(&ComparisonBlocker::LeftNotQuantitative(
            PredictionDisposition::NotDerived
        )));
        assert!(blockers.contains(&ComparisonBlocker::RightNotQuantitative(
            PredictionDisposition::QualitativeOnly
        )));
        assert!(blockers.contains(&ComparisonBlocker::NonGaussianNoise));
        assert!(blockers.contains(&ComparisonBlocker::NoiseNotCalibrated));
    }

    #[test]
    fn uncalibrated_candidate_number_cannot_enter_likelihood_comparison() {
        let comparison = fixture()
            .compare_models(
                "chain-theory",
                "minkowski-continuum",
                "inverse-alpha-candidate",
            )
            .unwrap();
        let ModelPredictionComparison::NotComparable { blockers, .. } = comparison else {
            panic!("uncalibrated measurement noise must block comparison");
        };
        assert!(blockers.contains(&ComparisonBlocker::RightNotQuantitative(
            PredictionDisposition::NotDerived
        )));
        assert!(blockers.contains(&ComparisonBlocker::NonGaussianNoise));
        assert!(blockers.contains(&ComparisonBlocker::NoiseNotCalibrated));
    }

    #[test]
    fn coverage_keeps_non_quantitative_statuses_separate() {
        let report = fixture().coverage().unwrap();
        assert_eq!(report.total_model_observable_pairs, 12);
        assert_eq!(report.quantitative, 3);
        assert_eq!(report.not_derived, 4);
        assert_eq!(report.outside_validated_domain, 1);
        assert_eq!(report.qualitative_only, 1);
        assert_eq!(report.unresolved, 3);
        assert!((report.quantitative_fraction() - 0.25).abs() < 1.0e-12);
    }

    #[test]
    fn non_quantitative_status_cannot_carry_a_numeric_value() {
        let mut ledger = fixture();
        let record = ledger
            .predictions
            .iter_mut()
            .find(|record| {
                record.theory_id == "chain-theory" && record.observable_id == "photon-propagator"
            })
            .unwrap();
        record.predicted_value = Some(0.0);
        assert_eq!(
            ledger.validate(),
            Err(TheoryPredictionError::NonQuantitativePredictionHasValue {
                theory_id: "chain-theory".to_string(),
                observable_id: "photon-propagator".to_string(),
            })
        );
    }

    #[test]
    fn quantitative_prediction_requires_derivation() {
        let mut ledger = fixture();
        let record = ledger
            .predictions
            .iter_mut()
            .find(|record| {
                record.theory_id == "chain-theory"
                    && record.observable_id == "inverse-alpha-candidate"
            })
            .unwrap();
        record.derivation_reference = None;
        assert_eq!(
            ledger.validate(),
            Err(
                TheoryPredictionError::QuantitativePredictionMissingDerivation {
                    theory_id: "chain-theory".to_string(),
                    observable_id: "inverse-alpha-candidate".to_string(),
                }
            )
        );
    }

    #[test]
    fn unresolved_model_cannot_be_given_an_arbitrary_quantitative_prediction() {
        let mut ledger = fixture();
        let record = ledger
            .predictions
            .iter_mut()
            .find(|record| {
                record.theory_id == OTHER_UNRESOLVED_MODEL_ID
                    && record.observable_id == "ordering-fraction-1p1"
            })
            .unwrap();
        record.disposition = PredictionDisposition::Quantitative;
        record.predicted_value = Some(0.0);
        record.derivation_reference = Some("convenience zero".to_string());
        assert_eq!(
            ledger.validate(),
            Err(
                TheoryPredictionError::UnresolvedModelMustNotInventPrediction {
                    observable_id: "ordering-fraction-1p1".to_string(),
                }
            )
        );
    }

    #[test]
    fn malformed_noise_calibration_fails_closed() {
        let mut ledger = fixture();
        ledger.observables[0].calibration_reference = None;
        assert_eq!(
            ledger.validate(),
            Err(TheoryPredictionError::MissingCalibrationReference {
                observable_id: "ordering-fraction-1p1".to_string(),
            })
        );
        let mut ledger = fixture();
        ledger.observables[0].measurement_sigma = Some(f64::NAN);
        assert_eq!(
            ledger.validate(),
            Err(TheoryPredictionError::InvalidMeasurementSigma {
                observable_id: "ordering-fraction-1p1".to_string(),
            })
        );
    }

    #[test]
    fn duplicate_and_unknown_model_observables_are_rejected() {
        let mut ledger = fixture();
        ledger.predictions.push(ledger.predictions[0].clone());
        assert!(matches!(
            ledger.validate(),
            Err(TheoryPredictionError::DuplicatePrediction { .. })
        ));
        let mut ledger = fixture();
        ledger.predictions[0].theory_id = "unknown-theory".to_string();
        assert_eq!(
            ledger.validate(),
            Err(TheoryPredictionError::UnknownTheoryModel(
                "unknown-theory".to_string()
            ))
        );
    }
}
