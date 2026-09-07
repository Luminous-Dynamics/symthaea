// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-qualified admission for learned nuclear-mass models.
//!
//! This layer extends the structural blind-validation work with two stronger
//! invariants:
//!
//! 1. a learned model is fitted only from the exact `BlindValidationSplit::training`
//!    corpus; and
//! 2. its receipt retains not just (Z,N) membership but the exact floating-point
//!    binding-energy bits and training configuration used by the fit.
//!
//! The existing exploratory `MlMassPredictor::new()` and
//! `HdcMassPredictor::new()` constructors are deliberately not used here.
//! Random-Forest admission goes through `fit_measured_with_config`; HDC admission
//! goes through `train_on` with the already-validated measured-only split.
//!
//! RF tree spread and raw (Z,N) distance are diagnostics, not calibrated
//! uncertainty. HDC receives no uncertainty field merely because it is a learned
//! model. Model disagreement is evidence to inspect, not a vote that creates truth.

use crate::blind_models::{
    BlindModelError, BlindModelTournament, BlindTrainingReceipt, NuclearModelDependency,
    evaluate_blind_tournament,
};
use crate::blind_validation::{
    BlindHoldout, BlindValidationError, BlindValidationReport, BlindValidationSplit,
};
use crate::hdc_mass::HdcMassPredictor;
use crate::ml_mass::{MlMassConfig, MlMassFitError, MlMassPredictor};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

/// Exact training value carried by a learned-model receipt.
///
/// `binding_energy_mev_bits` preserves the exact `f64` training datum rather
/// than relying on display formatting or a rounded decimal serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LearnedTrainingDatum {
    pub z: u16,
    pub n: u16,
    pub binding_energy_mev_bits: u64,
}

impl LearnedTrainingDatum {
    pub fn binding_energy_mev(self) -> f64 {
        f64::from_bits(self.binding_energy_mev_bits)
    }
}

/// Learned architecture admitted by this blind layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LearnedNuclearAlgorithm {
    RandomForestDzResidual,
    HdcLtcGluDzResidual,
}

/// Exact training configuration retained with the fit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LearnedTrainingConfig {
    RandomForest(MlMassConfig),
    HdcLtcGlu {
        epochs: usize,
        seed_offset: u64,
        /// Stable architecture identity for the current HDC implementation.
        /// The qualification workflow ratchets the implementation constants that
        /// this label denotes.
        architecture_id: String,
    },
}

/// Value-level learned-model training receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LearnedTrainingReceipt {
    pub coordinate_receipt: BlindTrainingReceipt,
    pub algorithm: LearnedNuclearAlgorithm,
    pub config: LearnedTrainingConfig,
    pub training_data: Vec<LearnedTrainingDatum>,
}

impl LearnedTrainingReceipt {
    fn from_split(
        split: &BlindValidationSplit,
        algorithm: LearnedNuclearAlgorithm,
        config: LearnedTrainingConfig,
    ) -> Result<Self, LearnedBlindError> {
        split.validate().map_err(LearnedBlindError::BlindValidation)?;
        let coordinate_receipt =
            BlindTrainingReceipt::from_split(split).map_err(LearnedBlindError::BlindModel)?;

        let training_data = split
            .training
            .iter()
            .map(|nucleus| LearnedTrainingDatum {
                z: nucleus.z,
                n: nucleus.n,
                binding_energy_mev_bits: nucleus.binding_energy_mev.to_bits(),
            })
            .collect::<Vec<_>>();

        let receipt = Self {
            coordinate_receipt,
            algorithm,
            config,
            training_data,
        };
        receipt.validate(split)?;
        Ok(receipt)
    }

    fn validate(&self, split: &BlindValidationSplit) -> Result<(), LearnedBlindError> {
        if self.training_data.len() != split.training.len()
            || self.coordinate_receipt.n_training() != split.training.len()
        {
            return Err(LearnedBlindError::TrainingReceiptLengthMismatch);
        }

        for (datum, nucleus) in self.training_data.iter().zip(&split.training) {
            if datum.z != nucleus.z
                || datum.n != nucleus.n
                || datum.binding_energy_mev_bits != nucleus.binding_energy_mev.to_bits()
            {
                return Err(LearnedBlindError::TrainingReceiptValueMismatch {
                    z: nucleus.z,
                    n: nucleus.n,
                });
            }
            if !self.coordinate_receipt.contains(datum.z, datum.n) {
                return Err(LearnedBlindError::TrainingReceiptCoordinateMismatch {
                    z: datum.z,
                    n: datum.n,
                });
            }
        }

        if split
            .holdout
            .iter()
            .any(|nucleus| self.coordinate_receipt.contains(nucleus.z, nucleus.n))
        {
            return Err(LearnedBlindError::HoldoutEnteredTrainingReceipt);
        }

        Ok(())
    }

    pub fn n_training(&self) -> usize {
        self.training_data.len()
    }

    pub fn contains(&self, z: u16, n: u16) -> bool {
        self.coordinate_receipt.contains(z, n)
    }
}

/// Scientific/model dependencies of a learned blind prediction.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum LearnedModelDependency {
    DufloZuker10,
    Ame2020MeasuredTrainingPartition,
    FrdmDeformationLookup,
    PhysicsMotivatedRfFeatureMapV1,
    RandomForestResidualRegressor,
    NuclearStateEncoder,
    HdcLtcUnifiedNeuron,
    GluResidualDecoder,
}

impl LearnedModelDependency {
    fn lineage_id(&self) -> &'static str {
        match self {
            Self::DufloZuker10 => "DufloZuker10",
            Self::Ame2020MeasuredTrainingPartition => "Ame2020MeasuredTrainingPartition",
            Self::FrdmDeformationLookup => "FrdmDeformationLookup",
            Self::PhysicsMotivatedRfFeatureMapV1 => "PhysicsMotivatedRfFeatureMapV1",
            Self::RandomForestResidualRegressor => "RandomForestResidualRegressor",
            Self::NuclearStateEncoder => "NuclearStateEncoder",
            Self::HdcLtcUnifiedNeuron => "HdcLtcUnifiedNeuron",
            Self::GluResidualDecoder => "GluResidualDecoder",
        }
    }
}

fn base_dependency_id(dependency: &NuclearModelDependency) -> &'static str {
    match dependency {
        NuclearModelDependency::SemiEmpiricalMassFormulaDefaults => {
            "SemiEmpiricalMassFormulaDefaults"
        }
        NuclearModelDependency::DufloZuker10 => "DufloZuker10",
        NuclearModelDependency::Ame2020MeasuredTrainingPartition => {
            "Ame2020MeasuredTrainingPartition"
        }
        NuclearModelDependency::CoordinateNearestNeighborResiduals => {
            "CoordinateNearestNeighborResiduals"
        }
    }
}

/// Prediction produced by one learned model admitted through a value-level
/// training receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LearnedNuclearPredictionEvidence {
    pub model_id: String,
    pub algorithm: LearnedNuclearAlgorithm,
    pub z: u16,
    pub n: u16,
    pub binding_energy_mev: f64,
    /// Optional ensemble dispersion from the model itself. This is explicitly a
    /// proxy, not a calibrated predictive interval.
    pub ensemble_dispersion_proxy_mev: Option<f64>,
    /// Raw Euclidean distance in (Z,N) to the closest training datum.
    pub nearest_training_distance: f64,
    pub dependencies: Vec<LearnedModelDependency>,
    pub training_receipt: LearnedTrainingReceipt,
}

impl LearnedNuclearPredictionEvidence {
    fn validate(&self) -> Result<(), LearnedBlindError> {
        if self.model_id.trim().is_empty() {
            return Err(LearnedBlindError::EmptyModelId);
        }
        if !self.binding_energy_mev.is_finite() {
            return Err(LearnedBlindError::NonFinitePrediction {
                model_id: self.model_id.clone(),
                z: self.z,
                n: self.n,
            });
        }
        if self
            .ensemble_dispersion_proxy_mev
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(LearnedBlindError::InvalidDispersionProxy {
                model_id: self.model_id.clone(),
            });
        }
        if !self.nearest_training_distance.is_finite() || self.nearest_training_distance < 0.0 {
            return Err(LearnedBlindError::InvalidTrainingDistance {
                model_id: self.model_id.clone(),
            });
        }
        Ok(())
    }
}

fn nearest_training_distance(receipt: &LearnedTrainingReceipt, z: u16, n: u16) -> f64 {
    receipt
        .training_data
        .iter()
        .map(|datum| {
            let dz = f64::from(z.abs_diff(datum.z));
            let dn = f64::from(n.abs_diff(datum.n));
            (dz * dz + dn * dn).sqrt()
        })
        .fold(f64::INFINITY, f64::min)
}

/// Random-Forest residual model fitted only from the blind training partition.
pub struct BlindRfMassModel {
    predictor: MlMassPredictor,
    receipt: LearnedTrainingReceipt,
}

impl BlindRfMassModel {
    pub fn fit(
        split: &BlindValidationSplit,
        config: MlMassConfig,
    ) -> Result<Self, LearnedBlindError> {
        let receipt = LearnedTrainingReceipt::from_split(
            split,
            LearnedNuclearAlgorithm::RandomForestDzResidual,
            LearnedTrainingConfig::RandomForest(config),
        )?;
        let predictor = MlMassPredictor::fit_measured_with_config(&split.training, config)
            .map_err(LearnedBlindError::MlMassFit)?;
        Ok(Self { predictor, receipt })
    }

    pub fn predict(
        &self,
        z: u16,
        n: u16,
    ) -> Result<LearnedNuclearPredictionEvidence, LearnedBlindError> {
        let prediction = self.predictor.predict(z, n);
        let evidence = LearnedNuclearPredictionEvidence {
            model_id: "DZ10+RF-blind-residual".to_string(),
            algorithm: LearnedNuclearAlgorithm::RandomForestDzResidual,
            z,
            n,
            binding_energy_mev: prediction.binding_energy,
            ensemble_dispersion_proxy_mev: Some(prediction.uncertainty),
            nearest_training_distance: nearest_training_distance(&self.receipt, z, n),
            dependencies: vec![
                LearnedModelDependency::DufloZuker10,
                LearnedModelDependency::Ame2020MeasuredTrainingPartition,
                LearnedModelDependency::FrdmDeformationLookup,
                LearnedModelDependency::PhysicsMotivatedRfFeatureMapV1,
                LearnedModelDependency::RandomForestResidualRegressor,
            ],
            training_receipt: self.receipt.clone(),
        };
        evidence.validate()?;
        Ok(evidence)
    }

    pub fn training_receipt(&self) -> &LearnedTrainingReceipt {
        &self.receipt
    }
}

/// HDC/LTC/GLU residual model fitted only from the blind training partition.
pub struct BlindHdcMassModel {
    predictor: HdcMassPredictor,
    receipt: LearnedTrainingReceipt,
}

impl BlindHdcMassModel {
    pub const ARCHITECTURE_ID: &'static str =
        "symthaea-hdc-ltc-glu-v1:heads=1:evolve-dt=0.01,0.1,1.0";

    pub fn fit(
        split: &BlindValidationSplit,
        epochs: usize,
        seed_offset: u64,
    ) -> Result<Self, LearnedBlindError> {
        if epochs == 0 {
            return Err(LearnedBlindError::InvalidHdcEpochs);
        }
        split.validate().map_err(LearnedBlindError::BlindValidation)?;
        let receipt = LearnedTrainingReceipt::from_split(
            split,
            LearnedNuclearAlgorithm::HdcLtcGluDzResidual,
            LearnedTrainingConfig::HdcLtcGlu {
                epochs,
                seed_offset,
                architecture_id: Self::ARCHITECTURE_ID.to_string(),
            },
        )?;

        // `split.validate()` establishes measured-only membership before the
        // existing explicit-corpus HDC constructor receives the training slice.
        let predictor = HdcMassPredictor::train_on(&split.training, epochs, seed_offset);
        Ok(Self { predictor, receipt })
    }

    pub fn predict(
        &self,
        z: u16,
        n: u16,
    ) -> Result<LearnedNuclearPredictionEvidence, LearnedBlindError> {
        let prediction = self.predictor.predict(z, n);
        let evidence = LearnedNuclearPredictionEvidence {
            model_id: "DZ10+HDC-LTC-GLU-blind-residual".to_string(),
            algorithm: LearnedNuclearAlgorithm::HdcLtcGluDzResidual,
            z,
            n,
            binding_energy_mev: prediction.binding_energy,
            // The current HDC predictor does not expose a calibrated ensemble or
            // posterior spread. Do not manufacture one here.
            ensemble_dispersion_proxy_mev: None,
            nearest_training_distance: nearest_training_distance(&self.receipt, z, n),
            dependencies: vec![
                LearnedModelDependency::DufloZuker10,
                LearnedModelDependency::Ame2020MeasuredTrainingPartition,
                LearnedModelDependency::FrdmDeformationLookup,
                LearnedModelDependency::NuclearStateEncoder,
                LearnedModelDependency::HdcLtcUnifiedNeuron,
                LearnedModelDependency::GluResidualDecoder,
            ],
            training_receipt: self.receipt.clone(),
        };
        evidence.validate()?;
        Ok(evidence)
    }

    pub fn training_receipt(&self) -> &LearnedTrainingReceipt {
        &self.receipt
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LearnedBlindTournamentEntry {
    pub model_id: String,
    pub algorithm: LearnedNuclearAlgorithm,
    pub dependencies: Vec<LearnedModelDependency>,
    pub training_receipt: LearnedTrainingReceipt,
    pub report: BlindValidationReport,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnifiedDependencyRelation {
    SharedDeclaredDependencies(Vec<String>),
    DeclaredDisjointWithinVocabulary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnifiedPairwiseDependencyRelation {
    pub left_model_id: String,
    pub right_model_id: String,
    pub relation: UnifiedDependencyRelation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnifiedModelLineage {
    pub model_id: String,
    pub dependencies: Vec<String>,
}

fn unified_dependency_relations(
    models: &[UnifiedModelLineage],
) -> Vec<UnifiedPairwiseDependencyRelation> {
    let mut relations = Vec::new();
    for left_index in 0..models.len() {
        for right_index in (left_index + 1)..models.len() {
            let left = &models[left_index];
            let right = &models[right_index];
            let right_set: BTreeSet<_> = right.dependencies.iter().cloned().collect();
            let shared = left
                .dependencies
                .iter()
                .filter(|dependency| right_set.contains(*dependency))
                .cloned()
                .collect::<Vec<_>>();
            let relation = if shared.is_empty() {
                UnifiedDependencyRelation::DeclaredDisjointWithinVocabulary
            } else {
                UnifiedDependencyRelation::SharedDeclaredDependencies(shared)
            };
            relations.push(UnifiedPairwiseDependencyRelation {
                left_model_id: left.model_id.clone(),
                right_model_id: right.model_id.clone(),
                relation,
            });
        }
    }
    relations
}

fn base_lineage(tournament: &BlindModelTournament) -> Vec<UnifiedModelLineage> {
    tournament
        .entries
        .iter()
        .map(|entry| UnifiedModelLineage {
            model_id: entry.model_id.clone(),
            dependencies: entry
                .dependencies
                .iter()
                .map(base_dependency_id)
                .map(str::to_string)
                .collect(),
        })
        .collect()
}

fn learned_lineage(entries: &[LearnedBlindTournamentEntry]) -> Vec<UnifiedModelLineage> {
    entries
        .iter()
        .map(|entry| UnifiedModelLineage {
            model_id: entry.model_id.clone(),
            dependencies: entry
                .dependencies
                .iter()
                .map(LearnedModelDependency::lineage_id)
                .map(str::to_string)
                .collect(),
        })
        .collect()
}

/// Same-holdout tournament combining the transparent #642 baselines with
/// explicitly fitted RF and HDC models.
///
/// There is deliberately no winner, ranking, consensus, confidence or
/// qualification field. The report exists to expose behavior and ancestry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LearnedBlindTournament {
    pub protocol: BlindHoldout,
    pub protocol_label: String,
    pub base: BlindModelTournament,
    pub learned_entries: Vec<LearnedBlindTournamentEntry>,
    pub unified_lineage: Vec<UnifiedModelLineage>,
    pub dependency_relations: Vec<UnifiedPairwiseDependencyRelation>,
}

pub fn evaluate_learned_blind_tournament(
    split: &BlindValidationSplit,
    nearest_neighbor_k: usize,
    rf_config: MlMassConfig,
    hdc_epochs: usize,
    hdc_seed_offset: u64,
) -> Result<LearnedBlindTournament, LearnedBlindError> {
    split.validate().map_err(LearnedBlindError::BlindValidation)?;
    let base = evaluate_blind_tournament(split, nearest_neighbor_k)
        .map_err(LearnedBlindError::BlindModel)?;
    let rf = BlindRfMassModel::fit(split, rf_config)?;
    let hdc = BlindHdcMassModel::fit(split, hdc_epochs, hdc_seed_offset)?;

    let rf_report = split
        .evaluate("DZ10+RF-blind-residual", |z, n| {
            rf.predict(z, n)
                .map(|prediction| prediction.binding_energy_mev)
                .unwrap_or(f64::NAN)
        })
        .map_err(LearnedBlindError::BlindValidation)?;
    let hdc_report = split
        .evaluate("DZ10+HDC-LTC-GLU-blind-residual", |z, n| {
            hdc.predict(z, n)
                .map(|prediction| prediction.binding_energy_mev)
                .unwrap_or(f64::NAN)
        })
        .map_err(LearnedBlindError::BlindValidation)?;

    let rf_example = rf.predict(split.holdout[0].z, split.holdout[0].n)?;
    let hdc_example = hdc.predict(split.holdout[0].z, split.holdout[0].n)?;
    let learned_entries = vec![
        LearnedBlindTournamentEntry {
            model_id: rf_example.model_id,
            algorithm: rf_example.algorithm,
            dependencies: rf_example.dependencies,
            training_receipt: rf.training_receipt().clone(),
            report: rf_report,
        },
        LearnedBlindTournamentEntry {
            model_id: hdc_example.model_id,
            algorithm: hdc_example.algorithm,
            dependencies: hdc_example.dependencies,
            training_receipt: hdc.training_receipt().clone(),
            report: hdc_report,
        },
    ];

    let mut unified_lineage = base_lineage(&base);
    unified_lineage.extend(learned_lineage(&learned_entries));
    let dependency_relations = unified_dependency_relations(&unified_lineage);

    Ok(LearnedBlindTournament {
        protocol: split.protocol,
        protocol_label: split.protocol.label(),
        base,
        learned_entries,
        unified_lineage,
        dependency_relations,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub enum LearnedBlindError {
    BlindValidation(BlindValidationError),
    BlindModel(BlindModelError),
    MlMassFit(MlMassFitError),
    InvalidHdcEpochs,
    TrainingReceiptLengthMismatch,
    TrainingReceiptValueMismatch { z: u16, n: u16 },
    TrainingReceiptCoordinateMismatch { z: u16, n: u16 },
    HoldoutEnteredTrainingReceipt,
    EmptyModelId,
    NonFinitePrediction { model_id: String, z: u16, n: u16 },
    InvalidDispersionProxy { model_id: String },
    InvalidTrainingDistance { model_id: String },
}

impl fmt::Display for LearnedBlindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BlindValidation(error) => write!(f, "blind validation failed: {error}"),
            Self::BlindModel(error) => write!(f, "blind model failed: {error}"),
            Self::MlMassFit(error) => write!(f, "RF fit failed: {error}"),
            Self::InvalidHdcEpochs => write!(f, "HDC blind fit requires at least one epoch"),
            Self::TrainingReceiptLengthMismatch => {
                write!(f, "learned training receipt length differs from frozen split")
            }
            Self::TrainingReceiptValueMismatch { z, n } => write!(
                f,
                "learned training receipt value differs from frozen split at Z={z}, N={n}"
            ),
            Self::TrainingReceiptCoordinateMismatch { z, n } => write!(
                f,
                "learned training receipt coordinate is absent from coordinate receipt at Z={z}, N={n}"
            ),
            Self::HoldoutEnteredTrainingReceipt => {
                write!(f, "blind holdout coordinate entered learned training receipt")
            }
            Self::EmptyModelId => write!(f, "learned model id must not be empty"),
            Self::NonFinitePrediction { model_id, z, n } => write!(
                f,
                "learned model {model_id} produced a non-finite prediction at Z={z}, N={n}"
            ),
            Self::InvalidDispersionProxy { model_id } => {
                write!(f, "learned model {model_id} produced an invalid dispersion proxy")
            }
            Self::InvalidTrainingDistance { model_id } => {
                write!(f, "learned model {model_id} produced an invalid training distance")
            }
        }
    }
}

impl std::error::Error for LearnedBlindError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ame2020::ame2020_reference_nuclei;

    fn tiny_chain_split() -> BlindValidationSplit {
        let measured = ame2020_reference_nuclei()
            .into_iter()
            .filter(|nucleus| nucleus.is_measured)
            .collect::<Vec<_>>();
        let training = measured
            .iter()
            .filter(|nucleus| nucleus.z != 50 && nucleus.z >= 3)
            .take(10)
            .cloned()
            .collect::<Vec<_>>();
        let holdout = measured
            .iter()
            .filter(|nucleus| nucleus.z == 50)
            .take(2)
            .cloned()
            .collect::<Vec<_>>();
        let split = BlindValidationSplit {
            protocol: BlindHoldout::IsotopicChain { z: 50 },
            training,
            holdout,
        };
        split.validate().unwrap();
        split
    }

    #[test]
    fn learned_receipt_binds_exact_training_values_and_excludes_holdout() {
        let split = tiny_chain_split();
        let receipt = LearnedTrainingReceipt::from_split(
            &split,
            LearnedNuclearAlgorithm::RandomForestDzResidual,
            LearnedTrainingConfig::RandomForest(MlMassConfig::default()),
        )
        .unwrap();

        assert_eq!(receipt.n_training(), split.training.len());
        for (datum, nucleus) in receipt.training_data.iter().zip(&split.training) {
            assert_eq!(datum.z, nucleus.z);
            assert_eq!(datum.n, nucleus.n);
            assert_eq!(datum.binding_energy_mev_bits, nucleus.binding_energy_mev.to_bits());
            assert_eq!(datum.binding_energy_mev().to_bits(), nucleus.binding_energy_mev.to_bits());
        }
        assert!(split
            .holdout
            .iter()
            .all(|nucleus| !receipt.contains(nucleus.z, nucleus.n)));
    }

    #[test]
    fn rf_blind_fit_uses_explicit_partition_and_labels_spread_as_proxy() {
        let split = tiny_chain_split();
        let model = BlindRfMassModel::fit(
            &split,
            MlMassConfig {
                n_trees: 8,
                max_depth: 4,
                min_samples: 2,
                seed: 91,
            },
        )
        .unwrap();
        let target = &split.holdout[0];
        let prediction = model.predict(target.z, target.n).unwrap();

        assert!(prediction.binding_energy_mev.is_finite());
        assert!(prediction
            .ensemble_dispersion_proxy_mev
            .is_some_and(|value| value.is_finite() && value >= 0.0));
        assert!(!model.training_receipt().contains(target.z, target.n));
        assert!(matches!(
            &model.training_receipt().config,
            LearnedTrainingConfig::RandomForest(MlMassConfig { seed: 91, .. })
        ));
    }

    #[test]
    fn hdc_blind_fit_uses_explicit_partition_without_inventing_uncertainty() {
        let split = tiny_chain_split();
        let model = BlindHdcMassModel::fit(&split, 1, 17).unwrap();
        let target = &split.holdout[0];
        let prediction = model.predict(target.z, target.n).unwrap();

        assert!(prediction.binding_energy_mev.is_finite());
        assert_eq!(prediction.ensemble_dispersion_proxy_mev, None);
        assert!(!model.training_receipt().contains(target.z, target.n));
        assert!(matches!(
            &model.training_receipt().config,
            LearnedTrainingConfig::HdcLtcGlu {
                epochs: 1,
                seed_offset: 17,
                architecture_id,
            } if architecture_id == BlindHdcMassModel::ARCHITECTURE_ID
        ));
    }

    #[test]
    fn learned_tournament_preserves_shared_dz_and_training_ancestry() {
        let split = tiny_chain_split();
        let tournament = evaluate_learned_blind_tournament(
            &split,
            3,
            MlMassConfig {
                n_trees: 8,
                max_depth: 4,
                min_samples: 2,
                seed: 123,
            },
            1,
            23,
        )
        .unwrap();

        assert_eq!(tournament.learned_entries.len(), 2);
        assert_eq!(tournament.unified_lineage.len(), 6);
        assert!(tournament.dependency_relations.iter().any(|relation| {
            relation.left_model_id.contains("RF")
                && relation.right_model_id.contains("HDC")
                && matches!(
                    &relation.relation,
                    UnifiedDependencyRelation::SharedDeclaredDependencies(shared)
                        if shared.contains(&"DufloZuker10".to_string())
                            && shared.contains(&"Ame2020MeasuredTrainingPartition".to_string())
                )
        }));
        for entry in &tournament.learned_entries {
            assert_eq!(entry.report.protocol, split.protocol);
            assert_eq!(entry.report.n_training, split.training.len());
            assert_eq!(entry.report.n_holdout, split.holdout.len());
        }
    }
}
