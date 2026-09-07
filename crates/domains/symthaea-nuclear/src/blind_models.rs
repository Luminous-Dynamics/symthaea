// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Lineage-bound models and model-parliament reports for blind nuclear mass work.
//!
//! The central invariant is that a trainable model used in a blind protocol must
//! be fitted from the exact `BlindValidationSplit::training` membership.  The
//! resulting receipt retains those coordinates so later evidence cannot silently
//! substitute a fuller AME corpus.
//!
//! This module intentionally starts with simple, auditable baselines rather than
//! wrapping `MlMassPredictor::new()`: that constructor currently trains on the
//! full measured corpus and therefore cannot satisfy a structural holdout.  The
//! existing RF remains useful exploratory code, but it is not admitted here until
//! it exposes an explicit training-set constructor.
//!
//! The "parliament" reports disagreement; it does not vote a truth into existence.
//! Shared declared dependencies remain visible and no winner/consensus field is
//! produced.

use crate::blind_validation::{
    BlindHoldout, BlindValidationError, BlindValidationReport, BlindValidationSplit,
};
use crate::duflo_zuker::dz_binding_energy;
use crate::mass_formula::SemiEmpiricalMassFormula;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

/// Stable nucleus coordinate used in training receipts.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct NuclearCoordinate {
    pub z: u16,
    pub n: u16,
}

/// Exact membership retained for one blind-model fit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindTrainingReceipt {
    pub protocol: BlindHoldout,
    pub protocol_label: String,
    pub source: String,
    pub nuclei: Vec<NuclearCoordinate>,
}

impl BlindTrainingReceipt {
    pub fn from_split(split: &BlindValidationSplit) -> Result<Self, BlindModelError> {
        split.validate().map_err(BlindModelError::BlindValidation)?;

        let mut seen = BTreeSet::new();
        let mut nuclei = Vec::with_capacity(split.training.len());
        for nucleus in &split.training {
            if !nucleus.binding_energy_mev.is_finite() {
                return Err(BlindModelError::NonFiniteTrainingEnergy {
                    z: nucleus.z,
                    n: nucleus.n,
                });
            }
            let coordinate = NuclearCoordinate {
                z: nucleus.z,
                n: nucleus.n,
            };
            if !seen.insert(coordinate) {
                return Err(BlindModelError::DuplicateTrainingNucleus {
                    z: nucleus.z,
                    n: nucleus.n,
                });
            }
            nuclei.push(coordinate);
        }

        // `contains` uses binary search, so the receipt—not merely the canonical
        // AME split constructor—owns the ordering invariant. This also makes
        // caller-constructed but otherwise valid splits safe and deterministic.
        nuclei.sort_unstable();

        Ok(Self {
            protocol: split.protocol,
            protocol_label: split.protocol.label(),
            source: "AME2020 measured-only blind training partition".to_string(),
            nuclei,
        })
    }

    pub fn n_training(&self) -> usize {
        self.nuclei.len()
    }

    pub fn contains(&self, z: u16, n: u16) -> bool {
        self.nuclei.binary_search(&NuclearCoordinate { z, n }).is_ok()
    }
}

/// Scientific dependencies declared by a model implementation.
///
/// These are lineage descriptors, not an evidence-strength ordering.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum NuclearModelDependency {
    SemiEmpiricalMassFormulaDefaults,
    DufloZuker10,
    Ame2020MeasuredTrainingPartition,
    CoordinateNearestNeighborResiduals,
}

/// A prediction with the model lineage required to interpret it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NuclearMassPredictionEvidence {
    pub model_id: String,
    pub z: u16,
    pub n: u16,
    pub binding_energy_mev: f64,
    /// Within-training residual dispersion. This is deliberately named a proxy:
    /// it is not a calibrated predictive uncertainty interval.
    pub residual_dispersion_proxy_mev: Option<f64>,
    /// Euclidean distance in raw (Z,N) coordinates to the nearest training point.
    /// This is an OOD geometry diagnostic, not a probability.
    pub nearest_training_distance: Option<f64>,
    pub dependencies: Vec<NuclearModelDependency>,
    pub training_receipt: Option<BlindTrainingReceipt>,
}

impl NuclearMassPredictionEvidence {
    fn validate(&self) -> Result<(), BlindModelError> {
        if self.model_id.trim().is_empty() {
            return Err(BlindModelError::EmptyModelId);
        }
        if !self.binding_energy_mev.is_finite() {
            return Err(BlindModelError::NonFinitePrediction {
                model_id: self.model_id.clone(),
                z: self.z,
                n: self.n,
            });
        }
        if self
            .residual_dispersion_proxy_mev
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(BlindModelError::InvalidDispersionProxy {
                model_id: self.model_id.clone(),
            });
        }
        if self
            .nearest_training_distance
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(BlindModelError::InvalidTrainingDistance {
                model_id: self.model_id.clone(),
            });
        }
        Ok(())
    }
}

/// DZ10 plus one mean residual learned only from the blind training partition.
#[derive(Debug, Clone)]
pub struct MeanDzResidualModel {
    correction_mev: f64,
    residual_rms_mev: f64,
    receipt: BlindTrainingReceipt,
}

impl MeanDzResidualModel {
    pub fn fit(split: &BlindValidationSplit) -> Result<Self, BlindModelError> {
        let receipt = BlindTrainingReceipt::from_split(split)?;
        let residuals: Vec<f64> = split
            .training
            .iter()
            .map(|nucleus| {
                nucleus.binding_energy_mev - dz_binding_energy(nucleus.z, nucleus.n)
            })
            .collect();

        let n = residuals.len() as f64;
        let correction_mev = residuals.iter().sum::<f64>() / n;
        let residual_rms_mev = (residuals
            .iter()
            .map(|residual| (residual - correction_mev).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();

        if !correction_mev.is_finite() || !residual_rms_mev.is_finite() {
            return Err(BlindModelError::NonFiniteFitStatistic {
                model_id: "DZ10+mean-training-residual".to_string(),
            });
        }

        Ok(Self {
            correction_mev,
            residual_rms_mev,
            receipt,
        })
    }

    pub fn predict(
        &self,
        z: u16,
        n: u16,
    ) -> Result<NuclearMassPredictionEvidence, BlindModelError> {
        let prediction = NuclearMassPredictionEvidence {
            model_id: "DZ10+mean-training-residual".to_string(),
            z,
            n,
            binding_energy_mev: dz_binding_energy(z, n) + self.correction_mev,
            residual_dispersion_proxy_mev: Some(self.residual_rms_mev),
            nearest_training_distance: None,
            dependencies: vec![
                NuclearModelDependency::DufloZuker10,
                NuclearModelDependency::Ame2020MeasuredTrainingPartition,
            ],
            training_receipt: Some(self.receipt.clone()),
        };
        prediction.validate()?;
        Ok(prediction)
    }

    pub fn training_receipt(&self) -> &BlindTrainingReceipt {
        &self.receipt
    }
}

#[derive(Debug, Clone)]
struct ResidualTrainingPoint {
    coordinate: NuclearCoordinate,
    residual_mev: f64,
}

/// DZ10 plus the mean residual of the `k` nearest training nuclei in (Z,N).
///
/// This is intentionally a transparent local baseline, not a claim that raw
/// coordinate distance is the optimal nuclear similarity metric. The returned
/// neighbor residual spread and nearest-coordinate distance remain diagnostics,
/// not calibrated uncertainty.
#[derive(Debug, Clone)]
pub struct KNearestDzResidualModel {
    k: usize,
    points: Vec<ResidualTrainingPoint>,
    receipt: BlindTrainingReceipt,
}

impl KNearestDzResidualModel {
    pub fn fit(split: &BlindValidationSplit, k: usize) -> Result<Self, BlindModelError> {
        let receipt = BlindTrainingReceipt::from_split(split)?;
        if k == 0 || k > split.training.len() {
            return Err(BlindModelError::InvalidNeighborCount {
                requested: k,
                available: split.training.len(),
            });
        }

        let points = split
            .training
            .iter()
            .map(|nucleus| ResidualTrainingPoint {
                coordinate: NuclearCoordinate {
                    z: nucleus.z,
                    n: nucleus.n,
                },
                residual_mev: nucleus.binding_energy_mev
                    - dz_binding_energy(nucleus.z, nucleus.n),
            })
            .collect();

        Ok(Self { k, points, receipt })
    }

    pub fn predict(
        &self,
        z: u16,
        n: u16,
    ) -> Result<NuclearMassPredictionEvidence, BlindModelError> {
        let mut neighbors: Vec<(f64, f64)> = self
            .points
            .iter()
            .map(|point| {
                let dz = f64::from(z.abs_diff(point.coordinate.z));
                let dn = f64::from(n.abs_diff(point.coordinate.n));
                ((dz * dz + dn * dn).sqrt(), point.residual_mev)
            })
            .collect();
        neighbors.sort_by(|left, right| left.0.total_cmp(&right.0));

        let selected = &neighbors[..self.k];
        let correction = selected.iter().map(|(_, residual)| residual).sum::<f64>()
            / self.k as f64;
        let dispersion = (selected
            .iter()
            .map(|(_, residual)| (residual - correction).powi(2))
            .sum::<f64>()
            / self.k as f64)
            .sqrt();
        let nearest_distance = selected[0].0;

        let prediction = NuclearMassPredictionEvidence {
            model_id: format!("DZ10+{0}NN-training-residual", self.k),
            z,
            n,
            binding_energy_mev: dz_binding_energy(z, n) + correction,
            residual_dispersion_proxy_mev: Some(dispersion),
            nearest_training_distance: Some(nearest_distance),
            dependencies: vec![
                NuclearModelDependency::DufloZuker10,
                NuclearModelDependency::Ame2020MeasuredTrainingPartition,
                NuclearModelDependency::CoordinateNearestNeighborResiduals,
            ],
            training_receipt: Some(self.receipt.clone()),
        };
        prediction.validate()?;
        Ok(prediction)
    }

    pub fn training_receipt(&self) -> &BlindTrainingReceipt {
        &self.receipt
    }
}

pub fn dz10_prediction(
    z: u16,
    n: u16,
) -> Result<NuclearMassPredictionEvidence, BlindModelError> {
    let prediction = NuclearMassPredictionEvidence {
        model_id: "DZ10".to_string(),
        z,
        n,
        binding_energy_mev: dz_binding_energy(z, n),
        residual_dispersion_proxy_mev: None,
        nearest_training_distance: None,
        dependencies: vec![NuclearModelDependency::DufloZuker10],
        training_receipt: None,
    };
    prediction.validate()?;
    Ok(prediction)
}

pub fn semf_default_prediction(
    z: u16,
    n: u16,
) -> Result<NuclearMassPredictionEvidence, BlindModelError> {
    let a = z.checked_add(n).ok_or(BlindModelError::MassNumberOverflow { z, n })?;
    let prediction = NuclearMassPredictionEvidence {
        model_id: "SEMF-default".to_string(),
        z,
        n,
        binding_energy_mev: SemiEmpiricalMassFormula::default().binding_energy(a, z),
        residual_dispersion_proxy_mev: None,
        nearest_training_distance: None,
        dependencies: vec![NuclearModelDependency::SemiEmpiricalMassFormulaDefaults],
        training_receipt: None,
    };
    prediction.validate()?;
    Ok(prediction)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeclaredDependencyRelation {
    SharedDeclaredDependencies(Vec<NuclearModelDependency>),
    DeclaredDisjointWithinVocabulary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairwiseModelDependencyRelation {
    pub left_model_id: String,
    pub right_model_id: String,
    pub relation: DeclaredDependencyRelation,
}

fn pairwise_dependency_relations(
    models: &[(String, Vec<NuclearModelDependency>)],
) -> Vec<PairwiseModelDependencyRelation> {
    let mut relations = Vec::new();
    for left_index in 0..models.len() {
        for right_index in (left_index + 1)..models.len() {
            let (left_id, left_dependencies) = &models[left_index];
            let (right_id, right_dependencies) = &models[right_index];
            let right_set: BTreeSet<_> = right_dependencies.iter().cloned().collect();
            let shared: Vec<_> = left_dependencies
                .iter()
                .filter(|dependency| right_set.contains(*dependency))
                .cloned()
                .collect();
            let relation = if shared.is_empty() {
                DeclaredDependencyRelation::DeclaredDisjointWithinVocabulary
            } else {
                DeclaredDependencyRelation::SharedDeclaredDependencies(shared)
            };
            relations.push(PairwiseModelDependencyRelation {
                left_model_id: left_id.clone(),
                right_model_id: right_id.clone(),
                relation,
            });
        }
    }
    relations
}

/// Target-level disagreement report.
///
/// `DeclaredDisjointWithinVocabulary` is deliberately weaker than independent
/// replication: undeclared common assumptions or calibration ancestry may exist.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NuclearPredictionParliament {
    pub z: u16,
    pub n: u16,
    pub predictions: Vec<NuclearMassPredictionEvidence>,
    pub mean_binding_energy_mev: f64,
    pub model_stddev_mev: f64,
    pub min_binding_energy_mev: f64,
    pub max_binding_energy_mev: f64,
    pub spread_mev: f64,
    pub dependency_relations: Vec<PairwiseModelDependencyRelation>,
}

impl NuclearPredictionParliament {
    pub fn new(
        z: u16,
        n: u16,
        predictions: Vec<NuclearMassPredictionEvidence>,
    ) -> Result<Self, BlindModelError> {
        if predictions.len() < 2 {
            return Err(BlindModelError::TooFewParliamentMembers);
        }

        let mut model_ids = BTreeSet::new();
        for prediction in &predictions {
            prediction.validate()?;
            if prediction.z != z || prediction.n != n {
                return Err(BlindModelError::PredictionTargetMismatch {
                    model_id: prediction.model_id.clone(),
                });
            }
            if !model_ids.insert(prediction.model_id.clone()) {
                return Err(BlindModelError::DuplicateModelId(
                    prediction.model_id.clone(),
                ));
            }
        }

        let n_models = predictions.len() as f64;
        let mean_binding_energy_mev = predictions
            .iter()
            .map(|prediction| prediction.binding_energy_mev)
            .sum::<f64>()
            / n_models;
        let model_stddev_mev = (predictions
            .iter()
            .map(|prediction| {
                (prediction.binding_energy_mev - mean_binding_energy_mev).powi(2)
            })
            .sum::<f64>()
            / n_models)
            .sqrt();
        let min_binding_energy_mev = predictions
            .iter()
            .map(|prediction| prediction.binding_energy_mev)
            .fold(f64::INFINITY, f64::min);
        let max_binding_energy_mev = predictions
            .iter()
            .map(|prediction| prediction.binding_energy_mev)
            .fold(f64::NEG_INFINITY, f64::max);
        let spread_mev = max_binding_energy_mev - min_binding_energy_mev;
        let models = predictions
            .iter()
            .map(|prediction| {
                (prediction.model_id.clone(), prediction.dependencies.clone())
            })
            .collect::<Vec<_>>();

        Ok(Self {
            z,
            n,
            predictions,
            mean_binding_energy_mev,
            model_stddev_mev,
            min_binding_energy_mev,
            max_binding_energy_mev,
            spread_mev,
            dependency_relations: pairwise_dependency_relations(&models),
        })
    }
}

/// Build a four-member parliament using only models whose training lineage can
/// be represented by this module.
pub fn blind_prediction_parliament(
    split: &BlindValidationSplit,
    z: u16,
    n: u16,
    k: usize,
) -> Result<NuclearPredictionParliament, BlindModelError> {
    split.validate().map_err(BlindModelError::BlindValidation)?;
    let mean_model = MeanDzResidualModel::fit(split)?;
    let nearest_model = KNearestDzResidualModel::fit(split, k)?;
    NuclearPredictionParliament::new(
        z,
        n,
        vec![
            semf_default_prediction(z, n)?,
            dz10_prediction(z, n)?,
            mean_model.predict(z, n)?,
            nearest_model.predict(z, n)?,
        ],
    )
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlindTournamentEntry {
    pub model_id: String,
    pub dependencies: Vec<NuclearModelDependency>,
    pub training_receipt: Option<BlindTrainingReceipt>,
    pub report: BlindValidationReport,
}

/// Performance evidence for multiple models on the exact same frozen holdout.
/// There is intentionally no winner/ranking field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlindModelTournament {
    pub protocol: BlindHoldout,
    pub protocol_label: String,
    pub entries: Vec<BlindTournamentEntry>,
    pub dependency_relations: Vec<PairwiseModelDependencyRelation>,
}

pub fn evaluate_blind_tournament(
    split: &BlindValidationSplit,
    k: usize,
) -> Result<BlindModelTournament, BlindModelError> {
    split.validate().map_err(BlindModelError::BlindValidation)?;
    let mean_model = MeanDzResidualModel::fit(split)?;
    let nearest_model = KNearestDzResidualModel::fit(split, k)?;
    let semf = SemiEmpiricalMassFormula::default();

    let semf_report = split
        .evaluate("SEMF-default", |z, n| {
            semf.binding_energy(z.saturating_add(n), z)
        })
        .map_err(BlindModelError::BlindValidation)?;
    let dz_report = split
        .evaluate("DZ10", dz_binding_energy)
        .map_err(BlindModelError::BlindValidation)?;
    let mean_report = split
        .evaluate("DZ10+mean-training-residual", |z, n| {
            dz_binding_energy(z, n) + mean_model.correction_mev
        })
        .map_err(BlindModelError::BlindValidation)?;
    let nearest_report = split
        .evaluate(format!("DZ10+{k}NN-training-residual"), |z, n| {
            nearest_model
                .predict(z, n)
                .map(|prediction| prediction.binding_energy_mev)
                .unwrap_or(f64::NAN)
        })
        .map_err(BlindModelError::BlindValidation)?;

    let entries = vec![
        BlindTournamentEntry {
            model_id: "SEMF-default".to_string(),
            dependencies: vec![NuclearModelDependency::SemiEmpiricalMassFormulaDefaults],
            training_receipt: None,
            report: semf_report,
        },
        BlindTournamentEntry {
            model_id: "DZ10".to_string(),
            dependencies: vec![NuclearModelDependency::DufloZuker10],
            training_receipt: None,
            report: dz_report,
        },
        BlindTournamentEntry {
            model_id: "DZ10+mean-training-residual".to_string(),
            dependencies: vec![
                NuclearModelDependency::DufloZuker10,
                NuclearModelDependency::Ame2020MeasuredTrainingPartition,
            ],
            training_receipt: Some(mean_model.training_receipt().clone()),
            report: mean_report,
        },
        BlindTournamentEntry {
            model_id: format!("DZ10+{k}NN-training-residual"),
            dependencies: vec![
                NuclearModelDependency::DufloZuker10,
                NuclearModelDependency::Ame2020MeasuredTrainingPartition,
                NuclearModelDependency::CoordinateNearestNeighborResiduals,
            ],
            training_receipt: Some(nearest_model.training_receipt().clone()),
            report: nearest_report,
        },
    ];
    let models = entries
        .iter()
        .map(|entry| (entry.model_id.clone(), entry.dependencies.clone()))
        .collect::<Vec<_>>();

    Ok(BlindModelTournament {
        protocol: split.protocol,
        protocol_label: split.protocol.label(),
        entries,
        dependency_relations: pairwise_dependency_relations(&models),
    })
}

#[derive(Debug, Clone, PartialEq)]
pub enum BlindModelError {
    BlindValidation(BlindValidationError),
    DuplicateTrainingNucleus { z: u16, n: u16 },
    NonFiniteTrainingEnergy { z: u16, n: u16 },
    InvalidNeighborCount { requested: usize, available: usize },
    NonFiniteFitStatistic { model_id: String },
    NonFinitePrediction { model_id: String, z: u16, n: u16 },
    InvalidDispersionProxy { model_id: String },
    InvalidTrainingDistance { model_id: String },
    MassNumberOverflow { z: u16, n: u16 },
    EmptyModelId,
    TooFewParliamentMembers,
    DuplicateModelId(String),
    PredictionTargetMismatch { model_id: String },
}

impl fmt::Display for BlindModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BlindValidation(error) => write!(f, "blind validation failed: {error}"),
            Self::DuplicateTrainingNucleus { z, n } => {
                write!(f, "duplicate blind-training nucleus Z={z}, N={n}")
            }
            Self::NonFiniteTrainingEnergy { z, n } => {
                write!(f, "non-finite blind-training energy at Z={z}, N={n}")
            }
            Self::InvalidNeighborCount {
                requested,
                available,
            } => write!(
                f,
                "invalid nearest-neighbor count {requested}; training partition has {available} nuclei"
            ),
            Self::NonFiniteFitStatistic { model_id } => {
                write!(f, "model {model_id} produced a non-finite fit statistic")
            }
            Self::NonFinitePrediction { model_id, z, n } => write!(
                f,
                "model {model_id} produced a non-finite prediction at Z={z}, N={n}"
            ),
            Self::InvalidDispersionProxy { model_id } => {
                write!(f, "model {model_id} produced an invalid dispersion proxy")
            }
            Self::InvalidTrainingDistance { model_id } => {
                write!(f, "model {model_id} produced an invalid training distance")
            }
            Self::MassNumberOverflow { z, n } => {
                write!(f, "mass number overflow for Z={z}, N={n}")
            }
            Self::EmptyModelId => write!(f, "model id must not be empty"),
            Self::TooFewParliamentMembers => {
                write!(f, "a model parliament requires at least two members")
            }
            Self::DuplicateModelId(model_id) => {
                write!(f, "duplicate model id in parliament: {model_id}")
            }
            Self::PredictionTargetMismatch { model_id } => write!(
                f,
                "model {model_id} prediction targets a different nucleus than the parliament"
            ),
        }
    }
}

impl std::error::Error for BlindModelError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn frontier_split() -> BlindValidationSplit {
        BlindValidationSplit::from_ame2020(BlindHoldout::ProtonFrontier {
            train_z_max: 82,
        })
        .unwrap()
    }

    #[test]
    fn training_receipt_is_exactly_the_split_training_membership() {
        let split = frontier_split();
        let receipt = BlindTrainingReceipt::from_split(&split).unwrap();
        assert_eq!(receipt.n_training(), split.training.len());
        assert!(split
            .training
            .iter()
            .all(|nucleus| receipt.contains(nucleus.z, nucleus.n)));
        assert!(split
            .holdout
            .iter()
            .all(|nucleus| !receipt.contains(nucleus.z, nucleus.n)));
    }

    #[test]
    fn training_receipt_contains_is_safe_for_unsorted_valid_split() {
        let mut split = frontier_split();
        split.training.reverse();
        split.validate().unwrap();

        let receipt = BlindTrainingReceipt::from_split(&split).unwrap();
        assert!(receipt.nuclei.windows(2).all(|window| window[0] <= window[1]));
        assert!(split
            .training
            .iter()
            .all(|nucleus| receipt.contains(nucleus.z, nucleus.n)));
    }

    #[test]
    fn trainable_models_never_receive_holdout_membership() {
        let split = frontier_split();
        let mean = MeanDzResidualModel::fit(&split).unwrap();
        let nearest = KNearestDzResidualModel::fit(&split, 5).unwrap();
        for nucleus in &split.holdout {
            assert!(!mean.training_receipt().contains(nucleus.z, nucleus.n));
            assert!(!nearest.training_receipt().contains(nucleus.z, nucleus.n));
        }
    }

    #[test]
    fn nearest_neighbor_model_rejects_invalid_k() {
        let split = frontier_split();
        assert!(matches!(
            KNearestDzResidualModel::fit(&split, 0),
            Err(BlindModelError::InvalidNeighborCount { .. })
        ));
        assert!(matches!(
            KNearestDzResidualModel::fit(&split, split.training.len() + 1),
            Err(BlindModelError::InvalidNeighborCount { .. })
        ));
    }

    #[test]
    fn blind_tournament_is_same_holdout_and_has_no_hidden_training_leak() {
        let split = frontier_split();
        let tournament = evaluate_blind_tournament(&split, 5).unwrap();
        assert_eq!(tournament.entries.len(), 4);
        for entry in &tournament.entries {
            assert_eq!(entry.report.protocol, split.protocol);
            assert_eq!(entry.report.n_holdout, split.holdout.len());
            if let Some(receipt) = &entry.training_receipt {
                assert_eq!(receipt.n_training(), split.training.len());
                assert!(split
                    .holdout
                    .iter()
                    .all(|nucleus| !receipt.contains(nucleus.z, nucleus.n)));
            }
        }
    }

    #[test]
    fn parliament_preserves_shared_model_ancestry() {
        let split = frontier_split();
        let parliament = blind_prediction_parliament(&split, 114, 184, 5).unwrap();
        assert_eq!(parliament.predictions.len(), 4);
        assert!(parliament.spread_mev.is_finite() && parliament.spread_mev >= 0.0);
        assert!(parliament.dependency_relations.iter().any(|relation| {
            matches!(
                &relation.relation,
                DeclaredDependencyRelation::SharedDeclaredDependencies(shared)
                    if shared.contains(&NuclearModelDependency::DufloZuker10)
            )
        }));
    }

    #[test]
    fn semf_and_dz_are_only_declared_disjoint_not_independent() {
        let parliament = NuclearPredictionParliament::new(
            82,
            126,
            vec![
                semf_default_prediction(82, 126).unwrap(),
                dz10_prediction(82, 126).unwrap(),
            ],
        )
        .unwrap();
        assert_eq!(
            parliament.dependency_relations[0].relation,
            DeclaredDependencyRelation::DeclaredDisjointWithinVocabulary
        );
    }
}
