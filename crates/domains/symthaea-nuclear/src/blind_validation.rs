// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Leakage-resistant validation protocols for nuclear mass prediction.
//!
//! Random k-fold cross-validation is useful for interpolation, but it is a weak
//! proxy for the scientific question posed by superheavy and neutron-rich
//! prediction: can a model extrapolate into a region deliberately absent from
//! its calibration set?
//!
//! This module freezes several structurally hostile holdout shapes over the
//! measured portion of AME2020. Estimated/extrapolated AME entries are excluded
//! from both training and adjudication by construction.

use crate::ame2020::ame2020_reference_nuclei;
use crate::discovery::MeasuredNucleus;
use crate::duflo_zuker::dz_binding_energy;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

/// A deliberately structured blind holdout.
///
/// These variants are scientific partitions, not difficulty rankings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlindHoldout {
    /// Train only through `train_z_max`; hold out all measured nuclei above it.
    ProtonFrontier { train_z_max: u16 },
    /// Train only through `train_n_max`; hold out all measured nuclei above it.
    NeutronFrontier { train_n_max: u16 },
    /// Hold out an entire isotopic chain (fixed proton number Z).
    IsotopicChain { z: u16 },
    /// Hold out an entire isotone (fixed neutron number N).
    Isotone { n: u16 },
    /// Hold out every measured nucleus at or above a declared heavy-element Z.
    SuperheavyRegion { z_min: u16 },
    /// Hold out a rectangular shell neighborhood around a declared center.
    ShellWindow {
        center_z: u16,
        center_n: u16,
        radius_z: u16,
        radius_n: u16,
    },
}

impl BlindHoldout {
    fn is_holdout(self, nucleus: &MeasuredNucleus) -> bool {
        match self {
            Self::ProtonFrontier { train_z_max } => nucleus.z > train_z_max,
            Self::NeutronFrontier { train_n_max } => nucleus.n > train_n_max,
            Self::IsotopicChain { z } => nucleus.z == z,
            Self::Isotone { n } => nucleus.n == n,
            Self::SuperheavyRegion { z_min } => nucleus.z >= z_min,
            Self::ShellWindow {
                center_z,
                center_n,
                radius_z,
                radius_n,
            } => {
                nucleus.z.abs_diff(center_z) <= radius_z
                    && nucleus.n.abs_diff(center_n) <= radius_n
            }
        }
    }

    pub fn label(self) -> String {
        match self {
            Self::ProtonFrontier { train_z_max } => {
                format!("proton-frontier:z>{train_z_max}")
            }
            Self::NeutronFrontier { train_n_max } => {
                format!("neutron-frontier:n>{train_n_max}")
            }
            Self::IsotopicChain { z } => format!("isotopic-chain:z={z}"),
            Self::Isotone { n } => format!("isotone:n={n}"),
            Self::SuperheavyRegion { z_min } => format!("superheavy:z>={z_min}"),
            Self::ShellWindow {
                center_z,
                center_n,
                radius_z,
                radius_n,
            } => format!(
                "shell-window:z={center_z}±{radius_z},n={center_n}±{radius_n}"
            ),
        }
    }
}

/// Frozen training/holdout membership for one blind validation protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlindValidationSplit {
    pub protocol: BlindHoldout,
    pub training: Vec<MeasuredNucleus>,
    pub holdout: Vec<MeasuredNucleus>,
}

impl BlindValidationSplit {
    /// Build from the measured subset of AME2020.
    pub fn from_ame2020(protocol: BlindHoldout) -> Result<Self, BlindValidationError> {
        let measured = ame2020_reference_nuclei()
            .into_iter()
            .filter(|nucleus| nucleus.is_measured);

        let (mut holdout, mut training): (Vec<_>, Vec<_>) =
            measured.partition(|nucleus| protocol.is_holdout(nucleus));

        training.sort_by_key(|nucleus| (nucleus.z, nucleus.n));
        holdout.sort_by_key(|nucleus| (nucleus.z, nucleus.n));

        let split = Self {
            protocol,
            training,
            holdout,
        };
        split.validate()?;
        Ok(split)
    }

    /// Re-check the no-leakage invariants carried by the split.
    pub fn validate(&self) -> Result<(), BlindValidationError> {
        if self.training.is_empty() {
            return Err(BlindValidationError::EmptyTrainingSet);
        }
        if self.holdout.is_empty() {
            return Err(BlindValidationError::EmptyHoldoutSet);
        }
        if self
            .training
            .iter()
            .chain(self.holdout.iter())
            .any(|nucleus| !nucleus.is_measured)
        {
            return Err(BlindValidationError::EstimatedMassEnteredBlindProtocol);
        }
        if self
            .training
            .iter()
            .any(|nucleus| self.protocol.is_holdout(nucleus))
            || self
                .holdout
                .iter()
                .any(|nucleus| !self.protocol.is_holdout(nucleus))
        {
            return Err(BlindValidationError::PartitionRuleViolation);
        }

        let training_ids: BTreeSet<_> = self
            .training
            .iter()
            .map(|nucleus| (nucleus.z, nucleus.n))
            .collect();
        if self
            .holdout
            .iter()
            .any(|nucleus| training_ids.contains(&(nucleus.z, nucleus.n)))
        {
            return Err(BlindValidationError::TrainHoldoutOverlap);
        }

        Ok(())
    }

    /// Evaluate any already-fitted or training-free predictor on the frozen
    /// holdout. The caller owns model fitting; this function owns only blind
    /// membership and adjudication metrics.
    ///
    /// For trainable models, callers must train exclusively on `self.training`
    /// before supplying the predictor closure. `evaluate_dz10` below is safe
    /// immediately because DZ10 is parameterized independently of this split.
    pub fn evaluate<F>(
        &self,
        method: impl Into<String>,
        mut predict_binding_energy: F,
    ) -> Result<BlindValidationReport, BlindValidationError>
    where
        F: FnMut(u16, u16) -> f64,
    {
        self.validate()?;

        let mut errors = Vec::with_capacity(self.holdout.len());
        for nucleus in &self.holdout {
            let predicted = predict_binding_energy(nucleus.z, nucleus.n);
            if !predicted.is_finite() {
                return Err(BlindValidationError::NonFinitePrediction {
                    z: nucleus.z,
                    n: nucleus.n,
                });
            }
            errors.push(predicted - nucleus.binding_energy_mev);
        }

        let n = errors.len() as f64;
        let bias_mev = errors.iter().sum::<f64>() / n;
        let mae_mev = errors.iter().map(|error| error.abs()).sum::<f64>() / n;
        let rms_mev = (errors.iter().map(|error| error * error).sum::<f64>() / n).sqrt();
        let max_abs_error_mev = errors.iter().map(|error| error.abs()).fold(0.0, f64::max);

        Ok(BlindValidationReport {
            protocol: self.protocol,
            protocol_label: self.protocol.label(),
            method: method.into(),
            n_training: self.training.len(),
            n_holdout: self.holdout.len(),
            bias_mev,
            mae_mev,
            rms_mev,
            max_abs_error_mev,
        })
    }

    /// Training-free DZ10 baseline on exactly the same frozen holdout.
    pub fn evaluate_dz10(&self) -> Result<BlindValidationReport, BlindValidationError> {
        self.evaluate("DZ10", dz_binding_energy)
    }
}

/// Blind extrapolation metrics. No pass/fail threshold is embedded here:
/// qualification policy must preregister its own acceptable bounds rather than
/// choosing them after seeing a result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlindValidationReport {
    pub protocol: BlindHoldout,
    pub protocol_label: String,
    pub method: String,
    pub n_training: usize,
    pub n_holdout: usize,
    pub bias_mev: f64,
    pub mae_mev: f64,
    pub rms_mev: f64,
    pub max_abs_error_mev: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindValidationError {
    EmptyTrainingSet,
    EmptyHoldoutSet,
    EstimatedMassEnteredBlindProtocol,
    PartitionRuleViolation,
    TrainHoldoutOverlap,
    NonFinitePrediction { z: u16, n: u16 },
}

impl fmt::Display for BlindValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTrainingSet => write!(f, "blind validation training set is empty"),
            Self::EmptyHoldoutSet => write!(f, "blind validation holdout set is empty"),
            Self::EstimatedMassEnteredBlindProtocol => write!(
                f,
                "AME estimated/extrapolated mass entered a measured-only blind protocol"
            ),
            Self::PartitionRuleViolation => {
                write!(f, "blind validation split violates its declared partition rule")
            }
            Self::TrainHoldoutOverlap => {
                write!(f, "a nucleus appears in both training and blind holdout")
            }
            Self::NonFinitePrediction { z, n } => {
                write!(f, "predictor returned non-finite binding energy at Z={z}, N={n}")
            }
        }
    }
}

impl std::error::Error for BlindValidationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proton_frontier_is_measured_only_and_leakage_free() {
        let split = BlindValidationSplit::from_ame2020(BlindHoldout::ProtonFrontier {
            train_z_max: 82,
        })
        .unwrap();

        assert!(split.training.iter().all(|nucleus| nucleus.is_measured));
        assert!(split.holdout.iter().all(|nucleus| nucleus.is_measured));
        assert!(split.training.iter().all(|nucleus| nucleus.z <= 82));
        assert!(split.holdout.iter().all(|nucleus| nucleus.z > 82));
        assert!(!split.training.is_empty());
        assert!(!split.holdout.is_empty());
    }

    #[test]
    fn whole_isotopic_chain_is_absent_from_training() {
        let split = BlindValidationSplit::from_ame2020(BlindHoldout::IsotopicChain { z: 50 })
            .unwrap();

        assert!(split.training.iter().all(|nucleus| nucleus.z != 50));
        assert!(split.holdout.iter().all(|nucleus| nucleus.z == 50));
    }

    #[test]
    fn shell_window_is_a_real_region_holdout() {
        let split = BlindValidationSplit::from_ame2020(BlindHoldout::ShellWindow {
            center_z: 82,
            center_n: 126,
            radius_z: 4,
            radius_n: 6,
        })
        .unwrap();

        assert!(split.holdout.iter().all(|nucleus| {
            nucleus.z.abs_diff(82) <= 4 && nucleus.n.abs_diff(126) <= 6
        }));
        assert!(split.training.iter().all(|nucleus| {
            nucleus.z.abs_diff(82) > 4 || nucleus.n.abs_diff(126) > 6
        }));
    }

    #[test]
    fn dz10_can_be_adjudicated_on_frontier_holdout() {
        let split = BlindValidationSplit::from_ame2020(BlindHoldout::ProtonFrontier {
            train_z_max: 82,
        })
        .unwrap();
        let report = split.evaluate_dz10().unwrap();

        assert_eq!(report.n_training, split.training.len());
        assert_eq!(report.n_holdout, split.holdout.len());
        assert!(report.rms_mev.is_finite() && report.rms_mev >= 0.0);
        assert!(report.mae_mev.is_finite() && report.mae_mev >= 0.0);
        assert!(report.max_abs_error_mev.is_finite());
    }

    #[test]
    fn nonfinite_predictor_fails_closed() {
        let split = BlindValidationSplit::from_ame2020(BlindHoldout::IsotopicChain { z: 50 })
            .unwrap();
        let error = split
            .evaluate("bad-model", |_z, _n| f64::NAN)
            .unwrap_err();
        assert!(matches!(
            error,
            BlindValidationError::NonFinitePrediction { .. }
        ));
    }
}
