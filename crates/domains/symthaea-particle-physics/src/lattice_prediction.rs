// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Blinded prediction and experiment-selection contracts for lattice physics.
//!
//! The purpose of this module is epistemic discipline, not numerical solving.
//! Predictions are frozen with explicit uncertainty and lineage before a held-
//! out result is revealed. Candidate calculations can then be ranked using
//! externally supplied information-gain and compute-cost estimates.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictionMethod {
    /// Direct output of a validated lattice calculation.
    FirstPrinciplesLattice,
    /// Learned emulator/surrogate whose calibration must be established elsewhere.
    ValidatedSurrogate,
    /// Closed-form or independently tabulated reference calculation.
    AnalyticReference,
    /// HDC / qualitative / hypothesis-generating estimate only.
    ExploratoryHeuristic,
}

impl PredictionMethod {
    pub fn is_first_principles(self) -> bool {
        matches!(self, Self::FirstPrinciplesLattice)
    }

    pub fn can_be_called_lattice_output(self) -> bool {
        matches!(self, Self::FirstPrinciplesLattice)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredictionTarget {
    /// Stable observable key, e.g. `glueball::0++::mass_ratio_to_string_tension`.
    pub observable: String,
    /// Stable ensemble / parameter-point identifier.
    pub ensemble_id: String,
    /// Unit label, or `dimensionless`.
    pub units: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredictionLineage {
    pub code_revision: String,
    pub model_revision: String,
    pub configuration_digest: String,
    pub training_data_digest: Option<String>,
    pub evidence_lineage: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindBenchmarkManifest {
    pub training_ids: Vec<String>,
    pub validation_ids: Vec<String>,
    pub held_out_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PredictionProtocolError {
    EmptyHeldOutSet,
    DuplicateDatasetId(String),
    DatasetLeakage(String),
    EmptyField(&'static str),
    NonFiniteValue,
    InvalidUncertainty(f64),
    RevealIdMismatch { expected: String, actual: String },
    InvalidInformationGain(f64),
    InvalidComputeCost(f64),
}

impl BlindBenchmarkManifest {
    pub fn validate(&self) -> Result<(), PredictionProtocolError> {
        if self.held_out_ids.is_empty() {
            return Err(PredictionProtocolError::EmptyHeldOutSet);
        }

        for ids in [&self.training_ids, &self.validation_ids, &self.held_out_ids] {
            let mut local = BTreeSet::new();
            for id in ids {
                if !local.insert(id) {
                    return Err(PredictionProtocolError::DuplicateDatasetId(id.clone()));
                }
            }
        }

        let training: BTreeSet<_> = self.training_ids.iter().collect();
        let validation: BTreeSet<_> = self.validation_ids.iter().collect();
        let held_out: BTreeSet<_> = self.held_out_ids.iter().collect();

        for id in training.intersection(&validation) {
            return Err(PredictionProtocolError::DatasetLeakage((*id).clone()));
        }
        for id in training.intersection(&held_out) {
            return Err(PredictionProtocolError::DatasetLeakage((*id).clone()));
        }
        for id in validation.intersection(&held_out) {
            return Err(PredictionProtocolError::DatasetLeakage((*id).clone()));
        }
        Ok(())
    }

    pub fn is_held_out(&self, id: &str) -> bool {
        self.held_out_ids.iter().any(|candidate| candidate == id)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrozenPrediction {
    pub prediction_id: String,
    pub held_out_id: String,
    pub target: PredictionTarget,
    pub method: PredictionMethod,
    pub value: f64,
    /// One-standard-deviation predictive uncertainty in `target.units`.
    pub uncertainty: f64,
    pub lineage: PredictionLineage,
    /// RFC3339 timestamp supplied by the caller / evidence system.
    pub frozen_at: String,
}

impl FrozenPrediction {
    pub fn validate(
        &self,
        benchmark: &BlindBenchmarkManifest,
    ) -> Result<(), PredictionProtocolError> {
        benchmark.validate()?;
        for (value, name) in [
            (&self.prediction_id, "prediction_id"),
            (&self.held_out_id, "held_out_id"),
            (&self.target.observable, "observable"),
            (&self.target.ensemble_id, "ensemble_id"),
            (&self.target.units, "units"),
            (&self.lineage.code_revision, "code_revision"),
            (&self.lineage.model_revision, "model_revision"),
            (&self.lineage.configuration_digest, "configuration_digest"),
            (&self.lineage.evidence_lineage, "evidence_lineage"),
            (&self.frozen_at, "frozen_at"),
        ] {
            if value.trim().is_empty() {
                return Err(PredictionProtocolError::EmptyField(name));
            }
        }
        if !benchmark.is_held_out(&self.held_out_id) {
            return Err(PredictionProtocolError::DatasetLeakage(
                self.held_out_id.clone(),
            ));
        }
        if !self.value.is_finite() {
            return Err(PredictionProtocolError::NonFiniteValue);
        }
        if !self.uncertainty.is_finite() || self.uncertainty <= 0.0 {
            return Err(PredictionProtocolError::InvalidUncertainty(
                self.uncertainty,
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredictionReveal {
    pub held_out_id: String,
    pub observed_value: f64,
    pub observed_uncertainty: f64,
    pub source_id: String,
    pub revealed_at: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PredictionScore {
    pub absolute_error: f64,
    pub combined_uncertainty: f64,
    /// `absolute_error / sqrt(sigma_pred^2 + sigma_obs^2)`.
    pub normalized_residual: f64,
}

pub fn score_prediction(
    prediction: &FrozenPrediction,
    reveal: &PredictionReveal,
) -> Result<PredictionScore, PredictionProtocolError> {
    if reveal.held_out_id != prediction.held_out_id {
        return Err(PredictionProtocolError::RevealIdMismatch {
            expected: prediction.held_out_id.clone(),
            actual: reveal.held_out_id.clone(),
        });
    }
    if !reveal.observed_value.is_finite() {
        return Err(PredictionProtocolError::NonFiniteValue);
    }
    if !reveal.observed_uncertainty.is_finite() || reveal.observed_uncertainty <= 0.0 {
        return Err(PredictionProtocolError::InvalidUncertainty(
            reveal.observed_uncertainty,
        ));
    }
    let absolute_error = (prediction.value - reveal.observed_value).abs();
    let combined_uncertainty =
        (prediction.uncertainty.powi(2) + reveal.observed_uncertainty.powi(2)).sqrt();
    Ok(PredictionScore {
        absolute_error,
        combined_uncertainty,
        normalized_residual: absolute_error / combined_uncertainty,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateCalculation {
    pub calculation_id: String,
    /// Estimated information gain supplied by an external model/analysis.
    pub expected_information_gain_nats: f64,
    /// Positive normalized cost estimate, e.g. GPU-hours or core-hours.
    pub estimated_compute_cost: f64,
    pub rationale: String,
}

impl CandidateCalculation {
    pub fn information_per_cost(&self) -> Result<f64, PredictionProtocolError> {
        if !self.expected_information_gain_nats.is_finite()
            || self.expected_information_gain_nats < 0.0
        {
            return Err(PredictionProtocolError::InvalidInformationGain(
                self.expected_information_gain_nats,
            ));
        }
        if !self.estimated_compute_cost.is_finite() || self.estimated_compute_cost <= 0.0 {
            return Err(PredictionProtocolError::InvalidComputeCost(
                self.estimated_compute_cost,
            ));
        }
        Ok(self.expected_information_gain_nats / self.estimated_compute_cost)
    }
}

/// Rank externally estimated candidate calculations by information gained per
/// unit compute cost. This function does not infer the information gain itself.
pub fn rank_candidate_calculations(
    candidates: &[CandidateCalculation],
) -> Result<Vec<CandidateCalculation>, PredictionProtocolError> {
    let mut ranked = candidates.to_vec();
    for candidate in &ranked {
        candidate.information_per_cost()?;
    }
    ranked.sort_by(|a, b| {
        let a_score = a.information_per_cost().expect("validated score");
        let b_score = b.information_per_cost().expect("validated score");
        b_score.total_cmp(&a_score)
    });
    Ok(ranked)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> BlindBenchmarkManifest {
        BlindBenchmarkManifest {
            training_ids: vec!["train-a".into(), "train-b".into()],
            validation_ids: vec!["validation-a".into()],
            held_out_ids: vec!["holdout-a".into()],
        }
    }

    fn prediction() -> FrozenPrediction {
        FrozenPrediction {
            prediction_id: "pred-001".into(),
            held_out_id: "holdout-a".into(),
            target: PredictionTarget {
                observable: "glueball::0++::mass_ratio".into(),
                ensemble_id: "pure-su3-beta-x-volume-y".into(),
                units: "dimensionless".into(),
            },
            method: PredictionMethod::ValidatedSurrogate,
            value: 1.50,
            uncertainty: 0.10,
            lineage: PredictionLineage {
                code_revision: "abc123".into(),
                model_revision: "surrogate-v1".into(),
                configuration_digest: "sha256:config".into(),
                training_data_digest: Some("sha256:training".into()),
                evidence_lineage: "evidence-root-1".into(),
            },
            frozen_at: "2026-09-11T00:00:00Z".into(),
        }
    }

    #[test]
    fn manifest_rejects_heldout_leakage() {
        let mut m = manifest();
        m.training_ids.push("holdout-a".into());
        assert!(matches!(
            m.validate(),
            Err(PredictionProtocolError::DatasetLeakage(id)) if id == "holdout-a"
        ));
    }

    #[test]
    fn frozen_prediction_requires_positive_uncertainty() {
        let mut p = prediction();
        p.uncertainty = 0.0;
        assert!(matches!(
            p.validate(&manifest()),
            Err(PredictionProtocolError::InvalidUncertainty(0.0))
        ));
    }

    #[test]
    fn heuristic_cannot_be_called_lattice_output() {
        assert!(!PredictionMethod::ExploratoryHeuristic.can_be_called_lattice_output());
        assert!(PredictionMethod::FirstPrinciplesLattice.can_be_called_lattice_output());
    }

    #[test]
    fn scoring_combines_prediction_and_observation_uncertainty() {
        let p = prediction();
        p.validate(&manifest()).unwrap();
        let reveal = PredictionReveal {
            held_out_id: "holdout-a".into(),
            observed_value: 1.60,
            observed_uncertainty: 0.10,
            source_id: "reference-lattice-run".into(),
            revealed_at: "2026-09-12T00:00:00Z".into(),
        };
        let score = score_prediction(&p, &reveal).unwrap();
        assert!((score.absolute_error - 0.10).abs() < 1e-12);
        assert!((score.combined_uncertainty - 2.0_f64.sqrt() * 0.10).abs() < 1e-12);
        assert!((score.normalized_residual - 1.0 / 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn experiment_ranking_is_information_per_cost_only() {
        let candidates = vec![
            CandidateCalculation {
                calculation_id: "expensive".into(),
                expected_information_gain_nats: 8.0,
                estimated_compute_cost: 8.0,
                rationale: "high gain, high cost".into(),
            },
            CandidateCalculation {
                calculation_id: "efficient".into(),
                expected_information_gain_nats: 4.0,
                estimated_compute_cost: 2.0,
                rationale: "best gain per cost".into(),
            },
        ];
        let ranked = rank_candidate_calculations(&candidates).unwrap();
        assert_eq!(ranked[0].calculation_id, "efficient");
    }
}
