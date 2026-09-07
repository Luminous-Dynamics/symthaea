// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Transparent heuristic ranking for candidate nuclear measurements.
//!
//! This module does not treat Random-Forest tree disagreement as calibrated
//! predictive uncertainty, does not claim a causal reduction in posterior
//! entropy, and does not infer experimental accessibility from `(Z,N)` alone.
//! It provides an auditable acquisition proxy whose components are exposed in
//! every candidate record so downstream experiment planners can decide whether
//! the heuristic is useful.

use crate::discovery::MeasuredNucleus;
use crate::ml_mass::{MlMassConfig, MlMassFitError, MlMassPredictor};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

/// Heuristic shell anchors used only as one optional ranking feature.
const SHELL_ANCHOR_Z: &[u16] = &[2, 8, 20, 28, 50, 82, 114, 126];
const SHELL_ANCHOR_N: &[u16] = &[2, 8, 20, 28, 50, 82, 126, 184];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasurementRegion {
    pub z_min: u16,
    pub z_max: u16,
    pub n_min: u16,
    pub n_max: u16,
}

impl MeasurementRegion {
    fn validate(self) -> Result<(), AcquisitionError> {
        if self.z_min == 0 || self.z_min > self.z_max || self.n_min > self.n_max {
            return Err(AcquisitionError::InvalidMeasurementRegion);
        }
        Ok(())
    }

    fn contains(self, z: u16, n: u16) -> bool {
        z >= self.z_min && z <= self.z_max && n >= self.n_min && n <= self.n_max
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AcquisitionScoreConfig {
    /// Multiplier on ln(1 + locally unmeasured coordinates).
    pub neighbor_log_weight: f64,
    /// Multiplier on the shell-anchor proximity proxy.
    pub shell_proximity_weight: f64,
    pub neighbor_radius_z: u16,
    pub neighbor_radius_n: u16,
    /// Optional predicted B/A screen. This is a caller-declared heuristic filter,
    /// not a statement of physical existence or experimental accessibility.
    pub min_predicted_ba_mev: Option<f64>,
}

impl AcquisitionScoreConfig {
    fn validate(self) -> Result<(), AcquisitionError> {
        if !self.neighbor_log_weight.is_finite()
            || self.neighbor_log_weight < 0.0
            || !self.shell_proximity_weight.is_finite()
            || self.shell_proximity_weight < 0.0
        {
            return Err(AcquisitionError::InvalidScoreWeight);
        }
        if self.neighbor_radius_z == 0 && self.neighbor_radius_n == 0 {
            return Err(AcquisitionError::EmptyNeighborWindow);
        }
        if self
            .min_predicted_ba_mev
            .is_some_and(|value| !value.is_finite())
        {
            return Err(AcquisitionError::InvalidPredictedBaScreen);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasurementFeasibilityScope {
    /// No beamline, reaction-channel, yield, detector, target, safety, or facility
    /// feasibility has been evaluated by this ranking layer.
    NotEvaluated,
    /// A caller points to external feasibility evidence. This module stores the
    /// identity but does not independently verify the facility/experiment claim.
    ExternallyDeclared { evidence_id: String },
}

impl MeasurementFeasibilityScope {
    fn validate(&self) -> Result<(), AcquisitionError> {
        match self {
            Self::NotEvaluated => Ok(()),
            Self::ExternallyDeclared { evidence_id } if !evidence_id.trim().is_empty() => Ok(()),
            Self::ExternallyDeclared { .. } => Err(AcquisitionError::EmptyFeasibilityEvidenceId),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasurementAcquisitionRequest {
    pub region: MeasurementRegion,
    pub score: AcquisitionScoreConfig,
    pub batch_size: usize,
    pub min_batch_manhattan_distance: u16,
    pub feasibility: MeasurementFeasibilityScope,
}

impl MeasurementAcquisitionRequest {
    fn validate(&self) -> Result<(), AcquisitionError> {
        self.region.validate()?;
        self.score.validate()?;
        self.feasibility.validate()?;
        if self.batch_size == 0 {
            return Err(AcquisitionError::ZeroBatchSize);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AcquisitionTrainingDatum {
    pub z: u16,
    pub n: u16,
    pub binding_energy_mev_bits: u64,
}

impl From<&MeasuredNucleus> for AcquisitionTrainingDatum {
    fn from(nucleus: &MeasuredNucleus) -> Self {
        Self {
            z: nucleus.z,
            n: nucleus.n,
            binding_energy_mev_bits: nucleus.binding_energy_mev.to_bits(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcquisitionTrainingReceipt {
    pub rf_config: MlMassConfig,
    pub exact_measured_training: Vec<AcquisitionTrainingDatum>,
}

impl AcquisitionTrainingReceipt {
    fn from_training(training: &[MeasuredNucleus], rf_config: MlMassConfig) -> Self {
        let mut exact_measured_training = training
            .iter()
            .map(AcquisitionTrainingDatum::from)
            .collect::<Vec<_>>();
        exact_measured_training.sort_unstable();
        Self {
            rf_config,
            exact_measured_training,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcquisitionInterpretation {
    HeuristicPriorityOnly,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasurementCandidatePriority {
    pub z: u16,
    pub n: u16,
    pub predicted_binding_energy_mev: f64,
    pub predicted_ba_mev: f64,
    /// Standard deviation across RF tree outputs. This remains an ensemble
    /// dispersion proxy, not a calibrated standard deviation of predictive error.
    pub tree_dispersion_proxy_mev: f64,
    /// Number of coordinates in the caller-declared local window that are absent
    /// from the supplied measured training corpus.
    pub unmeasured_neighbor_count_proxy: usize,
    /// 1/(1+d) to the nearest declared shell anchor in either Z or N.
    pub shell_proximity_proxy: f64,
    /// Transparent heuristic score:
    /// tree_spread * (1 + w_n ln(1+neighbors)) * (1 + w_s shell_proxy).
    pub acquisition_score_proxy: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasurementPriorityReport {
    pub request: MeasurementAcquisitionRequest,
    pub training_receipt: AcquisitionTrainingReceipt,
    pub interpretation: AcquisitionInterpretation,
    pub ranked_candidates: Vec<MeasurementCandidatePriority>,
    pub diverse_batch: Vec<MeasurementCandidatePriority>,
    pub note: String,
}

pub fn rank_measurement_candidates(
    training: &[MeasuredNucleus],
    rf_config: MlMassConfig,
    request: MeasurementAcquisitionRequest,
) -> Result<MeasurementPriorityReport, AcquisitionError> {
    request.validate()?;

    // The explicit-corpus constructor performs measured-only, duplicate, finite,
    // coordinate, and RF-configuration validation before ranking begins.
    let predictor = MlMassPredictor::fit_measured_with_config(training, rf_config)
        .map_err(AcquisitionError::MlMassFit)?;
    let training_receipt = AcquisitionTrainingReceipt::from_training(training, rf_config);
    let measured = training
        .iter()
        .map(|nucleus| (nucleus.z, nucleus.n))
        .collect::<BTreeSet<_>>();

    let mut ranked_candidates = Vec::new();
    for z in request.region.z_min..=request.region.z_max {
        for n in request.region.n_min..=request.region.n_max {
            if measured.contains(&(z, n)) {
                continue;
            }
            let prediction = predictor.predict(z, n);
            if !prediction.binding_energy.is_finite()
                || !prediction.ba.is_finite()
                || !prediction.uncertainty.is_finite()
                || prediction.uncertainty < 0.0
            {
                return Err(AcquisitionError::NonFiniteModelOutput { z, n });
            }
            if request
                .score
                .min_predicted_ba_mev
                .is_some_and(|minimum| prediction.ba < minimum)
            {
                continue;
            }

            let unmeasured_neighbor_count_proxy = count_unmeasured_neighbors(
                z,
                n,
                &measured,
                request.region,
                request.score.neighbor_radius_z,
                request.score.neighbor_radius_n,
            );
            let shell_proximity_proxy = shell_proximity_proxy(z, n);
            let neighbor_factor = 1.0
                + request.score.neighbor_log_weight
                    * (unmeasured_neighbor_count_proxy as f64 + 1.0).ln();
            let shell_factor =
                1.0 + request.score.shell_proximity_weight * shell_proximity_proxy;
            let acquisition_score_proxy = prediction.uncertainty * neighbor_factor * shell_factor;
            if !acquisition_score_proxy.is_finite() || acquisition_score_proxy < 0.0 {
                return Err(AcquisitionError::NonFiniteAcquisitionScore { z, n });
            }

            ranked_candidates.push(MeasurementCandidatePriority {
                z,
                n,
                predicted_binding_energy_mev: prediction.binding_energy,
                predicted_ba_mev: prediction.ba,
                tree_dispersion_proxy_mev: prediction.uncertainty,
                unmeasured_neighbor_count_proxy,
                shell_proximity_proxy,
                acquisition_score_proxy,
            });
        }
    }

    ranked_candidates.sort_by(|left, right| {
        right
            .acquisition_score_proxy
            .total_cmp(&left.acquisition_score_proxy)
            .then(left.z.cmp(&right.z))
            .then(left.n.cmp(&right.n))
    });

    let diverse_batch = select_diverse_batch(
        &ranked_candidates,
        request.batch_size,
        request.min_batch_manhattan_distance,
    );

    Ok(MeasurementPriorityReport {
        request,
        training_receipt,
        interpretation: AcquisitionInterpretation::HeuristicPriorityOnly,
        ranked_candidates,
        diverse_batch,
        note: "ranking uses RF tree-dispersion, local unmeasured-coordinate density, and shell-anchor proximity proxies; it does not establish predictive calibration, expected scientific value, experimental feasibility, or causal information value".to_string(),
    })
}

fn count_unmeasured_neighbors(
    z: u16,
    n: u16,
    measured: &BTreeSet<(u16, u16)>,
    region: MeasurementRegion,
    radius_z: u16,
    radius_n: u16,
) -> usize {
    let z_lo = z.saturating_sub(radius_z).max(region.z_min);
    let z_hi = z.saturating_add(radius_z).min(region.z_max);
    let n_lo = n.saturating_sub(radius_n).max(region.n_min);
    let n_hi = n.saturating_add(radius_n).min(region.n_max);
    let mut count = 0;
    for zz in z_lo..=z_hi {
        for nn in n_lo..=n_hi {
            if (zz, nn) != (z, n) && !measured.contains(&(zz, nn)) {
                count += 1;
            }
        }
    }
    count
}

fn shell_proximity_proxy(z: u16, n: u16) -> f64 {
    let min_z = SHELL_ANCHOR_Z
        .iter()
        .map(|anchor| z.abs_diff(*anchor))
        .min()
        .unwrap_or(u16::MAX);
    let min_n = SHELL_ANCHOR_N
        .iter()
        .map(|anchor| n.abs_diff(*anchor))
        .min()
        .unwrap_or(u16::MAX);
    let distance = min_z.min(min_n);
    1.0 / (1.0 + f64::from(distance))
}

pub fn select_diverse_batch(
    ranked_candidates: &[MeasurementCandidatePriority],
    batch_size: usize,
    min_manhattan_distance: u16,
) -> Vec<MeasurementCandidatePriority> {
    let mut selected = Vec::new();
    for candidate in ranked_candidates {
        if selected.len() >= batch_size {
            break;
        }
        let too_close = selected.iter().any(|existing: &MeasurementCandidatePriority| {
            candidate.z.abs_diff(existing.z) + candidate.n.abs_diff(existing.n)
                < min_manhattan_distance
        });
        if !too_close {
            selected.push(candidate.clone());
        }
    }
    selected
}

#[derive(Debug, Clone, PartialEq)]
pub enum AcquisitionError {
    InvalidMeasurementRegion,
    InvalidScoreWeight,
    EmptyNeighborWindow,
    InvalidPredictedBaScreen,
    EmptyFeasibilityEvidenceId,
    ZeroBatchSize,
    MlMassFit(MlMassFitError),
    NonFiniteModelOutput { z: u16, n: u16 },
    NonFiniteAcquisitionScore { z: u16, n: u16 },
}

impl fmt::Display for AcquisitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMeasurementRegion => write!(f, "invalid measurement-priority region"),
            Self::InvalidScoreWeight => write!(f, "acquisition heuristic weights must be finite and non-negative"),
            Self::EmptyNeighborWindow => write!(f, "acquisition neighbor window must span at least one axis"),
            Self::InvalidPredictedBaScreen => write!(f, "predicted B/A screen must be finite when present"),
            Self::EmptyFeasibilityEvidenceId => write!(f, "external feasibility evidence id must not be empty"),
            Self::ZeroBatchSize => write!(f, "measurement-priority batch size must be greater than zero"),
            Self::MlMassFit(error) => write!(f, "measurement-priority RF fit failed: {error}"),
            Self::NonFiniteModelOutput { z, n } => write!(f, "RF produced invalid ranking output at Z={z}, N={n}"),
            Self::NonFiniteAcquisitionScore { z, n } => write!(f, "acquisition proxy became invalid at Z={z}, N={n}"),
        }
    }
}

impl std::error::Error for AcquisitionError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn nucleus(z: u16, n: u16, be: f64) -> MeasuredNucleus {
        MeasuredNucleus {
            z,
            n,
            binding_energy_mev: be,
            is_measured: true,
        }
    }

    fn training() -> Vec<MeasuredNucleus> {
        vec![
            nucleus(6, 6, 92.1),
            nucleus(6, 7, 97.1),
            nucleus(7, 7, 104.7),
            nucleus(8, 8, 127.6),
            nucleus(9, 10, 147.8),
            nucleus(10, 10, 160.6),
        ]
    }

    fn request() -> MeasurementAcquisitionRequest {
        MeasurementAcquisitionRequest {
            region: MeasurementRegion {
                z_min: 6,
                z_max: 10,
                n_min: 6,
                n_max: 11,
            },
            score: AcquisitionScoreConfig {
                neighbor_log_weight: 1.0,
                shell_proximity_weight: 0.2,
                neighbor_radius_z: 1,
                neighbor_radius_n: 1,
                min_predicted_ba_mev: None,
            },
            batch_size: 5,
            min_batch_manhattan_distance: 2,
            feasibility: MeasurementFeasibilityScope::NotEvaluated,
        }
    }

    #[test]
    fn ranking_is_explicit_corpus_and_heuristic_only() {
        let training = training();
        let report = rank_measurement_candidates(
            &training,
            MlMassConfig {
                n_trees: 6,
                max_depth: 3,
                min_samples: 1,
                seed: 31,
            },
            request(),
        )
        .unwrap();

        assert_eq!(report.interpretation, AcquisitionInterpretation::HeuristicPriorityOnly);
        assert_eq!(report.training_receipt.exact_measured_training.len(), training.len());
        assert!(!report.ranked_candidates.is_empty());
        let measured = training
            .iter()
            .map(|nucleus| (nucleus.z, nucleus.n))
            .collect::<BTreeSet<_>>();
        assert!(report
            .ranked_candidates
            .iter()
            .all(|candidate| !measured.contains(&(candidate.z, candidate.n))));
        assert!(report.ranked_candidates.iter().all(|candidate| {
            candidate.tree_dispersion_proxy_mev.is_finite()
                && candidate.tree_dispersion_proxy_mev >= 0.0
                && candidate.acquisition_score_proxy.is_finite()
                && candidate.acquisition_score_proxy >= 0.0
        }));
    }

    #[test]
    fn diverse_batch_obeys_declared_manhattan_spacing() {
        let report = rank_measurement_candidates(
            &training(),
            MlMassConfig {
                n_trees: 4,
                max_depth: 3,
                min_samples: 1,
                seed: 7,
            },
            request(),
        )
        .unwrap();
        for i in 0..report.diverse_batch.len() {
            for j in (i + 1)..report.diverse_batch.len() {
                let left = &report.diverse_batch[i];
                let right = &report.diverse_batch[j];
                assert!(
                    left.z.abs_diff(right.z) + left.n.abs_diff(right.n) >= 2,
                    "batch spacing violated for ({},{}) and ({},{})",
                    left.z,
                    left.n,
                    right.z,
                    right.n
                );
            }
        }
    }

    #[test]
    fn external_feasibility_is_identity_only_and_requires_evidence_id() {
        let mut req = request();
        req.feasibility = MeasurementFeasibilityScope::ExternallyDeclared {
            evidence_id: String::new(),
        };
        assert!(matches!(
            rank_measurement_candidates(
                &training(),
                MlMassConfig {
                    n_trees: 4,
                    max_depth: 3,
                    min_samples: 1,
                    seed: 9,
                },
                req,
            ),
            Err(AcquisitionError::EmptyFeasibilityEvidenceId)
        ));
    }

    #[test]
    fn invalid_heuristic_weights_fail_closed() {
        let mut req = request();
        req.score.shell_proximity_weight = -0.1;
        assert!(matches!(
            rank_measurement_candidates(
                &training(),
                MlMassConfig {
                    n_trees: 4,
                    max_depth: 3,
                    min_samples: 1,
                    seed: 9,
                },
                req,
            ),
            Err(AcquisitionError::InvalidScoreWeight)
        ));
    }
}
