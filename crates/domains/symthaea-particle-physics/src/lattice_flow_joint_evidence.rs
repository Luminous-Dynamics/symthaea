// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence wrapper for production joint blocked-jackknife scale estimates.
//!
//! This layer binds the executable resampler result to the pre-existing
//! same-ensemble flow-curve evidence contract. It verifies that the central
//! curve, configuration population and resampling geometry all refer to the
//! same subject before promoting the uncertainty into a scale-evidence record.

use crate::lattice_flow_joint_jackknife::{
    JOINT_BLOCKED_JACKKNIFE_ID, JointBlockedJackknifeScale, JointFlowScaleKind,
};
use crate::lattice_flow_scale_evidence::{
    FlowEnergyEvidenceCurve, FlowScaleEstimateEvidence, FlowScaleEvidenceError,
    FlowScaleKind, bind_flow_scale_estimate,
};

#[derive(Debug, Clone, PartialEq)]
pub struct JointJackknifeScaleEstimateEvidence {
    pub scale: FlowScaleEstimateEvidence,
    pub resampling_method_id: &'static str,
    pub configuration_count: usize,
    pub block_size: usize,
    pub block_count: usize,
    pub replicate_count: usize,
    pub replicate_mean: f64,
    pub replicate_estimates: Vec<f64>,
    pub resampling_artifact_digest: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum JointJackknifeEvidenceError {
    Scale(FlowScaleEvidenceError),
    ResamplingMethodMismatch,
    InvalidBlockGeometry,
    ReplicateCountMismatch {
        expected: usize,
        actual: usize,
    },
    ConfigurationCountMismatch {
        curve: usize,
        resampling: usize,
    },
    MeanCurveLengthMismatch {
        curve: usize,
        resampling: usize,
    },
    MeanCurvePointMismatch {
        index: usize,
    },
    CentralEstimateMismatch {
        curve: f64,
        resampling: f64,
    },
    EmptyResamplingArtifactDigest,
}

impl From<FlowScaleEvidenceError> for JointJackknifeEvidenceError {
    fn from(value: FlowScaleEvidenceError) -> Self {
        Self::Scale(value)
    }
}

fn scale_kind(kind: JointFlowScaleKind) -> FlowScaleKind {
    match kind {
        JointFlowScaleKind::T0Like => FlowScaleKind::T0Like,
        JointFlowScaleKind::W0Like => FlowScaleKind::W0Like,
    }
}

fn approximately_equal(left: f64, right: f64) -> bool {
    let tolerance = 1.0e-12 * (1.0 + left.abs().max(right.abs()));
    (left - right).abs() <= tolerance
}

/// Bind an executable joint blocked-jackknife result to the frozen curve
/// evidence from which its central scale is supposed to have been derived.
pub fn bind_joint_jackknife_scale_evidence(
    curve: &FlowEnergyEvidenceCurve,
    resampling: &JointBlockedJackknifeScale,
    curve_artifact_digest: impl Into<String>,
    joint_resampling_evidence_id: impl Into<String>,
    resampling_artifact_digest: impl Into<String>,
) -> Result<JointJackknifeScaleEstimateEvidence, JointJackknifeEvidenceError> {
    if resampling.method_id != JOINT_BLOCKED_JACKKNIFE_ID {
        return Err(JointJackknifeEvidenceError::ResamplingMethodMismatch);
    }
    if resampling.block_size == 0
        || resampling.block_count < 2
        || resampling
            .block_size
            .checked_mul(resampling.block_count)
            != Some(resampling.configuration_count)
    {
        return Err(JointJackknifeEvidenceError::InvalidBlockGeometry);
    }
    if resampling.replicate_estimates.len() != resampling.block_count {
        return Err(JointJackknifeEvidenceError::ReplicateCountMismatch {
            expected: resampling.block_count,
            actual: resampling.replicate_estimates.len(),
        });
    }

    let kind = scale_kind(resampling.kind);
    let scale = bind_flow_scale_estimate(
        curve,
        kind,
        resampling.target,
        resampling.standard_error,
        curve_artifact_digest,
        joint_resampling_evidence_id,
    )?;

    let curve_configuration_count = curve.points[0].retained_configurations;
    if curve_configuration_count != resampling.configuration_count {
        return Err(JointJackknifeEvidenceError::ConfigurationCountMismatch {
            curve: curve_configuration_count,
            resampling: resampling.configuration_count,
        });
    }
    if curve.points.len() != resampling.ensemble_mean_curve.len() {
        return Err(JointJackknifeEvidenceError::MeanCurveLengthMismatch {
            curve: curve.points.len(),
            resampling: resampling.ensemble_mean_curve.len(),
        });
    }
    for (index, (curve_point, resampled_point)) in curve
        .points
        .iter()
        .zip(&resampling.ensemble_mean_curve)
        .enumerate()
    {
        if !approximately_equal(curve_point.flow_time, resampled_point.flow_time)
            || !approximately_equal(
                curve_point.ensemble_mean_energy,
                resampled_point.ensemble_mean_energy,
            )
        {
            return Err(JointJackknifeEvidenceError::MeanCurvePointMismatch { index });
        }
    }
    if !approximately_equal(scale.estimate, resampling.central_estimate) {
        return Err(JointJackknifeEvidenceError::CentralEstimateMismatch {
            curve: scale.estimate,
            resampling: resampling.central_estimate,
        });
    }

    let resampling_artifact_digest = resampling_artifact_digest.into();
    if resampling_artifact_digest.trim().is_empty() {
        return Err(JointJackknifeEvidenceError::EmptyResamplingArtifactDigest);
    }

    Ok(JointJackknifeScaleEstimateEvidence {
        scale,
        resampling_method_id: JOINT_BLOCKED_JACKKNIFE_ID,
        configuration_count: resampling.configuration_count,
        block_size: resampling.block_size,
        block_count: resampling.block_count,
        replicate_count: resampling.replicate_estimates.len(),
        replicate_mean: resampling.replicate_mean,
        replicate_estimates: resampling.replicate_estimates.clone(),
        resampling_artifact_digest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_flow_energy::{CLOVER_FLOW_ENERGY_ID, EnsembleFlowEnergyPoint};
    use crate::lattice_flow_scale_evidence::FlowEnergyEvidencePoint;

    fn point(flow_time: f64, dimensionless: f64) -> FlowEnergyEvidencePoint {
        FlowEnergyEvidencePoint {
            flow_time,
            ensemble_mean_energy: dimensionless / (flow_time * flow_time),
            standard_error: 0.01,
            effective_sample_size: 10.0,
            retained_configurations: 12,
            independent_chain_count: 2,
            energy_operator_id: CLOVER_FLOW_ENERGY_ID.into(),
            flow_implementation_id: "wilson_action_staple_rk3_v1".into(),
            flow_step_size: 0.1,
            ensemble_manifest_digest: "ensemble-sha256".into(),
            statistics_evidence_id: format!("stats-{flow_time}"),
            output_artifact_digest: format!("artifact-{flow_time}"),
        }
    }

    fn curve() -> FlowEnergyEvidenceCurve {
        FlowEnergyEvidenceCurve {
            points: vec![
                point(0.1, 0.10),
                point(0.2, 0.18),
                point(0.3, 0.26),
                point(0.4, 0.34),
            ],
        }
    }

    fn result() -> JointBlockedJackknifeScale {
        JointBlockedJackknifeScale {
            method_id: JOINT_BLOCKED_JACKKNIFE_ID,
            kind: JointFlowScaleKind::T0Like,
            target: 0.30,
            central_estimate: 0.35,
            replicate_estimates: vec![0.34, 0.345, 0.355, 0.36],
            replicate_mean: 0.35,
            standard_error: 0.019_364_916_731_037_084,
            ensemble_mean_curve: vec![
                EnsembleFlowEnergyPoint {
                    flow_time: 0.1,
                    ensemble_mean_energy: 0.10 / 0.01,
                },
                EnsembleFlowEnergyPoint {
                    flow_time: 0.2,
                    ensemble_mean_energy: 0.18 / 0.04,
                },
                EnsembleFlowEnergyPoint {
                    flow_time: 0.3,
                    ensemble_mean_energy: 0.26 / 0.09,
                },
                EnsembleFlowEnergyPoint {
                    flow_time: 0.4,
                    ensemble_mean_energy: 0.34 / 0.16,
                },
            ],
            configuration_count: 12,
            block_size: 3,
            block_count: 4,
        }
    }

    #[test]
    fn binds_resampling_geometry_and_replicates_to_scale_evidence() {
        let evidence = bind_joint_jackknife_scale_evidence(
            &curve(),
            &result(),
            "curve-sha256",
            "jackknife-evidence-id",
            "jackknife-artifact-sha256",
        )
        .unwrap();
        assert!((evidence.scale.estimate - 0.35).abs() < 1.0e-15);
        assert_eq!(evidence.resampling_method_id, JOINT_BLOCKED_JACKKNIFE_ID);
        assert_eq!(evidence.configuration_count, 12);
        assert_eq!(evidence.block_size, 3);
        assert_eq!(evidence.block_count, 4);
        assert_eq!(evidence.replicate_count, 4);
        assert_eq!(evidence.replicate_estimates, vec![0.34, 0.345, 0.355, 0.36]);
    }

    #[test]
    fn mismatched_population_fails_closed() {
        let mut resampling = result();
        resampling.configuration_count = 15;
        resampling.block_size = 5;
        assert!(matches!(
            bind_joint_jackknife_scale_evidence(
                &curve(),
                &resampling,
                "curve-sha256",
                "jackknife-evidence-id",
                "jackknife-artifact-sha256",
            ),
            Err(JointJackknifeEvidenceError::ConfigurationCountMismatch {
                curve: 12,
                resampling: 15,
            })
        ));
    }

    #[test]
    fn mismatched_mean_curve_fails_closed() {
        let mut resampling = result();
        resampling.ensemble_mean_curve[2].ensemble_mean_energy *= 1.01;
        assert!(matches!(
            bind_joint_jackknife_scale_evidence(
                &curve(),
                &resampling,
                "curve-sha256",
                "jackknife-evidence-id",
                "jackknife-artifact-sha256",
            ),
            Err(JointJackknifeEvidenceError::MeanCurvePointMismatch { index: 2 })
        ));
    }
}
