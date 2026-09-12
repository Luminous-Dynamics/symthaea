// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bearing ensemble flow curves and scale estimates.
//!
//! A scale extracted from `t^2<E(t)>` is a derived ensemble statistic. Values at
//! different flow times are correlated because they are normally measured on
//! the same retained gauge configurations. This module therefore refuses to
//! synthesize a scale uncertainty from independent pointwise error bars. A
//! promoted scale estimate must bind an externally produced **joint resampling**
//! evidence record (e.g. blocked jackknife/bootstrap over whole configuration
//! trajectories).

use crate::lattice_flow_energy::{
    CLOVER_FLOW_ENERGY_ID, EnsembleFlowEnergyPoint, FlowEnergyError,
    t0_like_from_ensemble_mean, w0_like_from_ensemble_mean,
};

#[derive(Debug, Clone, PartialEq)]
pub struct FlowEnergyEvidencePoint {
    pub flow_time: f64,
    pub ensemble_mean_energy: f64,
    pub standard_error: f64,
    pub effective_sample_size: f64,
    pub retained_configurations: usize,
    pub independent_chain_count: usize,
    pub energy_operator_id: String,
    pub flow_implementation_id: String,
    pub flow_step_size: f64,
    pub ensemble_manifest_digest: String,
    pub statistics_evidence_id: String,
    pub output_artifact_digest: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlowEnergyEvidenceCurve {
    pub points: Vec<FlowEnergyEvidencePoint>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowScaleKind {
    T0Like,
    W0Like,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlowScaleEstimateEvidence {
    pub kind: FlowScaleKind,
    pub target: f64,
    /// `t0` for T0Like, `w0` for W0Like, both in raw lattice units.
    pub estimate: f64,
    /// Must come from a joint resampling of the complete correlated flow curve.
    pub standard_error: f64,
    pub energy_operator_id: String,
    pub flow_implementation_id: String,
    pub flow_step_size: f64,
    pub ensemble_manifest_digest: String,
    pub curve_artifact_digest: String,
    pub joint_resampling_evidence_id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FlowScaleEvidenceError {
    Energy(FlowEnergyError),
    TooFewPoints { required: usize, actual: usize },
    NonFinitePoint { index: usize },
    InvalidFlowStepSize { index: usize, value: f64 },
    FlowTimeOffStepGrid {
        index: usize,
        flow_time: f64,
        flow_step_size: f64,
    },
    InvalidPointStandardError { index: usize, value: f64 },
    InvalidEffectiveSampleSize { index: usize, value: f64 },
    InvalidRetainedConfigurations { index: usize, value: usize },
    InvalidIndependentChainCount { index: usize },
    EmptyField { index: Option<usize>, field: &'static str },
    EnergyOperatorMismatch { index: usize },
    FlowImplementationMismatch { index: usize },
    FlowStepSizeMismatch { index: usize },
    RetainedConfigurationsMismatch { index: usize },
    IndependentChainCountMismatch { index: usize },
    EnsembleManifestMismatch { index: usize },
    NonIncreasingFlowTime { previous: f64, current: f64 },
    InvalidScaleStandardError(f64),
}

impl From<FlowEnergyError> for FlowScaleEvidenceError {
    fn from(value: FlowEnergyError) -> Self {
        Self::Energy(value)
    }
}

fn require_nonempty(
    value: &str,
    index: Option<usize>,
    field: &'static str,
) -> Result<(), FlowScaleEvidenceError> {
    if value.trim().is_empty() {
        Err(FlowScaleEvidenceError::EmptyField { index, field })
    } else {
        Ok(())
    }
}

impl FlowEnergyEvidenceCurve {
    pub fn validate(&self, required: usize) -> Result<(), FlowScaleEvidenceError> {
        let required = required.max(1);
        if self.points.len() < required {
            return Err(FlowScaleEvidenceError::TooFewPoints {
                required,
                actual: self.points.len(),
            });
        }

        let first = &self.points[0];
        require_nonempty(&first.energy_operator_id, Some(0), "energy_operator_id")?;
        require_nonempty(&first.flow_implementation_id, Some(0), "flow_implementation_id")?;
        require_nonempty(
            &first.ensemble_manifest_digest,
            Some(0),
            "ensemble_manifest_digest",
        )?;
        if first.energy_operator_id != CLOVER_FLOW_ENERGY_ID {
            return Err(FlowScaleEvidenceError::EnergyOperatorMismatch { index: 0 });
        }

        for (index, point) in self.points.iter().enumerate() {
            if !point.flow_time.is_finite()
                || !point.ensemble_mean_energy.is_finite()
                || !point.flow_step_size.is_finite()
            {
                return Err(FlowScaleEvidenceError::NonFinitePoint { index });
            }
            if point.flow_time <= 0.0 {
                return Err(FlowScaleEvidenceError::Energy(
                    FlowEnergyError::NonPositiveFlowTime {
                        index,
                        value: point.flow_time,
                    },
                ));
            }
            if point.ensemble_mean_energy < 0.0 {
                return Err(FlowScaleEvidenceError::Energy(
                    FlowEnergyError::NegativeMeanEnergy {
                        index,
                        value: point.ensemble_mean_energy,
                    },
                ));
            }
            if point.flow_step_size <= 0.0 {
                return Err(FlowScaleEvidenceError::InvalidFlowStepSize {
                    index,
                    value: point.flow_step_size,
                });
            }
            let step_count = point.flow_time / point.flow_step_size;
            let nearest_step_count = step_count.round();
            let grid_tolerance = 1.0e-10 * (1.0 + step_count.abs());
            if nearest_step_count < 1.0 || (step_count - nearest_step_count).abs() > grid_tolerance {
                return Err(FlowScaleEvidenceError::FlowTimeOffStepGrid {
                    index,
                    flow_time: point.flow_time,
                    flow_step_size: point.flow_step_size,
                });
            }
            if !point.standard_error.is_finite() || point.standard_error < 0.0 {
                return Err(FlowScaleEvidenceError::InvalidPointStandardError {
                    index,
                    value: point.standard_error,
                });
            }
            if !point.effective_sample_size.is_finite() || point.effective_sample_size <= 0.0 {
                return Err(FlowScaleEvidenceError::InvalidEffectiveSampleSize {
                    index,
                    value: point.effective_sample_size,
                });
            }
            if point.retained_configurations < 2 {
                return Err(FlowScaleEvidenceError::InvalidRetainedConfigurations {
                    index,
                    value: point.retained_configurations,
                });
            }
            if point.independent_chain_count == 0 {
                return Err(FlowScaleEvidenceError::InvalidIndependentChainCount { index });
            }
            require_nonempty(&point.energy_operator_id, Some(index), "energy_operator_id")?;
            require_nonempty(
                &point.flow_implementation_id,
                Some(index),
                "flow_implementation_id",
            )?;
            require_nonempty(
                &point.ensemble_manifest_digest,
                Some(index),
                "ensemble_manifest_digest",
            )?;
            require_nonempty(
                &point.statistics_evidence_id,
                Some(index),
                "statistics_evidence_id",
            )?;
            require_nonempty(
                &point.output_artifact_digest,
                Some(index),
                "output_artifact_digest",
            )?;

            if point.energy_operator_id != first.energy_operator_id {
                return Err(FlowScaleEvidenceError::EnergyOperatorMismatch { index });
            }
            if point.flow_implementation_id != first.flow_implementation_id {
                return Err(FlowScaleEvidenceError::FlowImplementationMismatch { index });
            }
            if point.flow_step_size.to_bits() != first.flow_step_size.to_bits() {
                return Err(FlowScaleEvidenceError::FlowStepSizeMismatch { index });
            }
            if point.retained_configurations != first.retained_configurations {
                return Err(FlowScaleEvidenceError::RetainedConfigurationsMismatch { index });
            }
            if point.independent_chain_count != first.independent_chain_count {
                return Err(FlowScaleEvidenceError::IndependentChainCountMismatch { index });
            }
            if point.ensemble_manifest_digest != first.ensemble_manifest_digest {
                return Err(FlowScaleEvidenceError::EnsembleManifestMismatch { index });
            }
            if index > 0 && point.flow_time <= self.points[index - 1].flow_time {
                return Err(FlowScaleEvidenceError::NonIncreasingFlowTime {
                    previous: self.points[index - 1].flow_time,
                    current: point.flow_time,
                });
            }
        }
        Ok(())
    }

    pub fn mean_points(&self) -> Vec<EnsembleFlowEnergyPoint> {
        self.points
            .iter()
            .map(|point| EnsembleFlowEnergyPoint {
                flow_time: point.flow_time,
                ensemble_mean_energy: point.ensemble_mean_energy,
            })
            .collect()
    }
}

/// Bind a central scale estimate to a **joint** uncertainty analysis.
///
/// The central value is computed from the frozen ensemble means. The uncertainty
/// is deliberately supplied rather than reconstructed from the pointwise
/// standard errors: a correct uncertainty analysis must preserve correlation
/// across flow times by resampling whole retained configurations/blocks.
pub fn bind_flow_scale_estimate(
    curve: &FlowEnergyEvidenceCurve,
    kind: FlowScaleKind,
    target: f64,
    standard_error: f64,
    curve_artifact_digest: impl Into<String>,
    joint_resampling_evidence_id: impl Into<String>,
) -> Result<FlowScaleEstimateEvidence, FlowScaleEvidenceError> {
    let required = match kind {
        FlowScaleKind::T0Like => 2,
        FlowScaleKind::W0Like => 4,
    };
    curve.validate(required)?;
    if !standard_error.is_finite() || standard_error < 0.0 {
        return Err(FlowScaleEvidenceError::InvalidScaleStandardError(
            standard_error,
        ));
    }

    let curve_artifact_digest = curve_artifact_digest.into();
    let joint_resampling_evidence_id = joint_resampling_evidence_id.into();
    require_nonempty(&curve_artifact_digest, None, "curve_artifact_digest")?;
    require_nonempty(
        &joint_resampling_evidence_id,
        None,
        "joint_resampling_evidence_id",
    )?;

    let mean_points = curve.mean_points();
    let estimate = match kind {
        FlowScaleKind::T0Like => t0_like_from_ensemble_mean(&mean_points, target)?,
        FlowScaleKind::W0Like => w0_like_from_ensemble_mean(&mean_points, target)?,
    };
    let first = &curve.points[0];

    Ok(FlowScaleEstimateEvidence {
        kind,
        target,
        estimate,
        standard_error,
        energy_operator_id: first.energy_operator_id.clone(),
        flow_implementation_id: first.flow_implementation_id.clone(),
        flow_step_size: first.flow_step_size,
        ensemble_manifest_digest: first.ensemble_manifest_digest.clone(),
        curve_artifact_digest,
        joint_resampling_evidence_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(flow_time: f64, dimensionless: f64) -> FlowEnergyEvidencePoint {
        FlowEnergyEvidencePoint {
            flow_time,
            ensemble_mean_energy: dimensionless / (flow_time * flow_time),
            standard_error: 0.01,
            effective_sample_size: 42.0,
            retained_configurations: 64,
            independent_chain_count: 4,
            energy_operator_id: CLOVER_FLOW_ENERGY_ID.into(),
            flow_implementation_id: "wilson_action_staple_rk3_v1".into(),
            flow_step_size: 0.001,
            ensemble_manifest_digest: "ensemble-sha256".into(),
            statistics_evidence_id: format!("stats-{flow_time}"),
            output_artifact_digest: format!("artifact-{flow_time}"),
        }
    }

    fn t0_curve() -> FlowEnergyEvidenceCurve {
        FlowEnergyEvidenceCurve {
            points: vec![
                point(0.10, 0.10),
                point(0.20, 0.18),
                point(0.30, 0.26),
                point(0.40, 0.34),
                point(0.50, 0.42),
                point(0.60, 0.50),
            ],
        }
    }

    #[test]
    fn scale_estimate_requires_joint_resampling_evidence() {
        let evidence = bind_flow_scale_estimate(
            &t0_curve(),
            FlowScaleKind::T0Like,
            0.30,
            0.004,
            "curve-sha256",
            "blocked-jackknife-evidence",
        )
        .unwrap();
        assert!((evidence.estimate - 0.35).abs() < 1.0e-15);
        assert_eq!(evidence.standard_error, 0.004);
        assert_eq!(evidence.energy_operator_id, CLOVER_FLOW_ENERGY_ID);
        assert_eq!(
            evidence.joint_resampling_evidence_id,
            "blocked-jackknife-evidence"
        );
    }

    #[test]
    fn empty_joint_resampling_lineage_fails_closed() {
        assert!(matches!(
            bind_flow_scale_estimate(
                &t0_curve(),
                FlowScaleKind::T0Like,
                0.30,
                0.004,
                "curve-sha256",
                "",
            ),
            Err(FlowScaleEvidenceError::EmptyField {
                index: None,
                field: "joint_resampling_evidence_id"
            })
        ));
    }

    #[test]
    fn empty_curve_fails_even_when_caller_requests_zero_points() {
        let curve = FlowEnergyEvidenceCurve { points: vec![] };
        assert!(matches!(
            curve.validate(0),
            Err(FlowScaleEvidenceError::TooFewPoints {
                required: 1,
                actual: 0
            })
        ));
    }

    #[test]
    fn mixing_ensemble_manifests_is_rejected() {
        let mut curve = t0_curve();
        curve.points[3].ensemble_manifest_digest = "different-ensemble".into();
        assert!(matches!(
            curve.validate(2),
            Err(FlowScaleEvidenceError::EnsembleManifestMismatch { index: 3 })
        ));
    }

    #[test]
    fn mixing_integrators_is_rejected() {
        let mut curve = t0_curve();
        curve.points[2].flow_implementation_id = "lie_euler_reference_v1".into();
        assert!(matches!(
            curve.validate(2),
            Err(FlowScaleEvidenceError::FlowImplementationMismatch { index: 2 })
        ));
    }

    #[test]
    fn mixing_flow_step_sizes_is_rejected_even_at_valid_times() {
        let mut curve = t0_curve();
        curve.points[4].flow_step_size = 0.0005;
        assert!(matches!(
            curve.validate(2),
            Err(FlowScaleEvidenceError::FlowStepSizeMismatch { index: 4 })
        ));
    }

    #[test]
    fn mixing_retained_population_is_rejected() {
        let mut curve = t0_curve();
        curve.points[1].retained_configurations = 63;
        assert!(matches!(
            curve.validate(2),
            Err(FlowScaleEvidenceError::RetainedConfigurationsMismatch { index: 1 })
        ));
        let mut curve = t0_curve();
        curve.points[5].independent_chain_count = 3;
        assert!(matches!(
            curve.validate(2),
            Err(FlowScaleEvidenceError::IndependentChainCountMismatch { index: 5 })
        ));
    }

    #[test]
    fn off_grid_flow_time_is_rejected() {
        let mut curve = t0_curve();
        curve.points[2].flow_time = 0.300_5;
        assert!(matches!(
            curve.validate(2),
            Err(FlowScaleEvidenceError::FlowTimeOffStepGrid { index: 2, .. })
        ));
    }
}
