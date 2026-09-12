// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Block-size stability analysis for nonlinear gradient-flow scale estimates.
//!
//! A single blocked-jackknife result can have mathematically correct resampling
//! semantics while still depending materially on the chosen block length. This
//! module therefore recomputes the full nonlinear scale estimator at multiple
//! chain-local block sizes and assesses only those candidates that independently
//! satisfy the declared LQCD-018F block-adequacy policy.
//!
//! No universal plateau tolerance is encoded here. The caller declares how many
//! adequate block sizes must participate and the maximum tolerated relative
//! changes in the central estimate and standard error.

use crate::lattice_flow_block_adequacy::BlockAdequacyAssessment;
use crate::lattice_flow_joint_jackknife::{
    JointBlockedJackknifeInput, JointBlockedJackknifeScale, JointFlowScaleKind,
    JointJackknifeError, joint_blocked_jackknife_scale,
};

pub const FLOW_SCALE_BLOCK_STABILITY_ID: &str = "flow_scale_block_stability_v1";

#[derive(Debug, Clone, PartialEq)]
pub struct FlowScaleBlockScan {
    pub kind: JointFlowScaleKind,
    pub target: f64,
    pub configuration_count: usize,
    pub independent_chain_count: usize,
    pub results: Vec<JointBlockedJackknifeScale>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlowScaleBlockStabilityPolicy {
    /// Number of the largest adequate block sizes used to define the plateau.
    pub plateau_point_count: usize,
    /// Symmetric relative change tolerated between adjacent plateau SE values.
    pub maximum_relative_standard_error_change: f64,
    /// Symmetric relative change tolerated between adjacent central estimates.
    pub maximum_relative_central_estimate_change: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlowScaleBlockStabilityAssessment {
    pub assessment_id: &'static str,
    pub plateau_point_count: usize,
    pub maximum_relative_standard_error_change: f64,
    pub maximum_relative_central_estimate_change: f64,
    pub admissible_block_sizes: Vec<usize>,
    pub plateau_block_sizes: Vec<usize>,
    pub maximum_observed_relative_standard_error_change: Option<f64>,
    pub maximum_observed_relative_central_estimate_change: Option<f64>,
    pub enough_admissible_points: bool,
    pub meets_uncertainty_plateau: bool,
    pub meets_central_estimate_stability: bool,
    pub meets_declared_policy: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FlowScaleBlockStabilityError {
    Jackknife(JointJackknifeError),
    EmptyBlockSizeScan,
    InvalidBlockSizeCandidate { index: usize, value: usize },
    NonIncreasingBlockSize { previous: usize, current: usize },
    InconsistentScanSubject,
    InconsistentCentralEstimate,
    InvalidPolicyPlateauPointCount(usize),
    InvalidPolicyRelativeStandardErrorChange(f64),
    InvalidPolicyRelativeCentralEstimateChange(f64),
    AdequacyCountMismatch { scan: usize, adequacy: usize },
    AdequacyGeometryMismatch { index: usize },
}

impl From<JointJackknifeError> for FlowScaleBlockStabilityError {
    fn from(value: JointJackknifeError) -> Self {
        Self::Jackknife(value)
    }
}

fn approximately_equal(left: f64, right: f64) -> bool {
    let tolerance = 1.0e-12 * (1.0 + left.abs().max(right.abs()));
    (left - right).abs() <= tolerance
}

fn symmetric_relative_change(left: f64, right: f64) -> f64 {
    let denominator = left.abs() + right.abs();
    if denominator == 0.0 {
        0.0
    } else {
        2.0 * (left - right).abs() / denominator
    }
}

/// Recompute the complete joint blocked-jackknife scale estimate for every
/// declared block size. Each candidate is validated by the chain-aware 018E
/// resampler, so a block can never straddle independent-chain boundaries.
pub fn joint_jackknife_block_size_scan(
    input: &JointBlockedJackknifeInput,
    kind: JointFlowScaleKind,
    target: f64,
    block_sizes: &[usize],
) -> Result<FlowScaleBlockScan, FlowScaleBlockStabilityError> {
    if block_sizes.is_empty() {
        return Err(FlowScaleBlockStabilityError::EmptyBlockSizeScan);
    }
    for (index, block_size) in block_sizes.iter().copied().enumerate() {
        if block_size == 0 {
            return Err(FlowScaleBlockStabilityError::InvalidBlockSizeCandidate {
                index,
                value: block_size,
            });
        }
        if index > 0 && block_size <= block_sizes[index - 1] {
            return Err(FlowScaleBlockStabilityError::NonIncreasingBlockSize {
                previous: block_sizes[index - 1],
                current: block_size,
            });
        }
    }

    let mut results = Vec::with_capacity(block_sizes.len());
    for block_size in block_sizes.iter().copied() {
        let mut candidate = input.clone();
        candidate.block_size = block_size;
        results.push(joint_blocked_jackknife_scale(&candidate, kind, target)?);
    }

    let first = &results[0];
    for result in &results[1..] {
        if result.kind != first.kind
            || result.target.to_bits() != first.target.to_bits()
            || result.configuration_count != first.configuration_count
            || result.independent_chain_count != first.independent_chain_count
        {
            return Err(FlowScaleBlockStabilityError::InconsistentScanSubject);
        }
        if !approximately_equal(result.central_estimate, first.central_estimate) {
            return Err(FlowScaleBlockStabilityError::InconsistentCentralEstimate);
        }
    }

    Ok(FlowScaleBlockScan {
        kind,
        target,
        configuration_count: first.configuration_count,
        independent_chain_count: first.independent_chain_count,
        results,
    })
}

/// Assess the largest independently adequate block sizes for a stable uncertainty
/// plateau. Low block sizes may legitimately fail 018F and are excluded rather
/// than being allowed to make the plateau easier to satisfy.
pub fn assess_flow_scale_block_stability(
    scan: &FlowScaleBlockScan,
    adequacy: &[BlockAdequacyAssessment],
    policy: &FlowScaleBlockStabilityPolicy,
) -> Result<FlowScaleBlockStabilityAssessment, FlowScaleBlockStabilityError> {
    if policy.plateau_point_count < 2 {
        return Err(FlowScaleBlockStabilityError::InvalidPolicyPlateauPointCount(
            policy.plateau_point_count,
        ));
    }
    if !policy.maximum_relative_standard_error_change.is_finite()
        || policy.maximum_relative_standard_error_change < 0.0
    {
        return Err(
            FlowScaleBlockStabilityError::InvalidPolicyRelativeStandardErrorChange(
                policy.maximum_relative_standard_error_change,
            ),
        );
    }
    if !policy.maximum_relative_central_estimate_change.is_finite()
        || policy.maximum_relative_central_estimate_change < 0.0
    {
        return Err(
            FlowScaleBlockStabilityError::InvalidPolicyRelativeCentralEstimateChange(
                policy.maximum_relative_central_estimate_change,
            ),
        );
    }
    if scan.results.len() != adequacy.len() {
        return Err(FlowScaleBlockStabilityError::AdequacyCountMismatch {
            scan: scan.results.len(),
            adequacy: adequacy.len(),
        });
    }

    let mut admissible_indices = Vec::new();
    for (index, (result, assessment)) in scan.results.iter().zip(adequacy).enumerate() {
        if result.block_size != assessment.block_size
            || result.block_count != assessment.block_count
            || result.configuration_count != assessment.configuration_count
        {
            return Err(FlowScaleBlockStabilityError::AdequacyGeometryMismatch { index });
        }
        if assessment.meets_declared_policy {
            admissible_indices.push(index);
        }
    }

    let admissible_block_sizes = admissible_indices
        .iter()
        .map(|index| scan.results[*index].block_size)
        .collect::<Vec<_>>();
    let enough_admissible_points = admissible_indices.len() >= policy.plateau_point_count;
    let plateau_indices = if enough_admissible_points {
        admissible_indices[admissible_indices.len() - policy.plateau_point_count..].to_vec()
    } else {
        admissible_indices.clone()
    };
    let plateau_block_sizes = plateau_indices
        .iter()
        .map(|index| scan.results[*index].block_size)
        .collect::<Vec<_>>();

    let mut maximum_se_change: Option<f64> = None;
    let mut maximum_central_change: Option<f64> = None;
    for pair in plateau_indices.windows(2) {
        let left = &scan.results[pair[0]];
        let right = &scan.results[pair[1]];
        let se_change = symmetric_relative_change(left.standard_error, right.standard_error);
        let central_change =
            symmetric_relative_change(left.central_estimate, right.central_estimate);
        maximum_se_change = Some(maximum_se_change.map_or(se_change, |value| value.max(se_change)));
        maximum_central_change = Some(
            maximum_central_change.map_or(central_change, |value| value.max(central_change)),
        );
    }

    let meets_uncertainty_plateau = enough_admissible_points
        && maximum_se_change.is_some_and(|change| {
            change <= policy.maximum_relative_standard_error_change
        });
    let meets_central_estimate_stability = enough_admissible_points
        && maximum_central_change.is_some_and(|change| {
            change <= policy.maximum_relative_central_estimate_change
        });
    let meets_declared_policy =
        meets_uncertainty_plateau && meets_central_estimate_stability;

    Ok(FlowScaleBlockStabilityAssessment {
        assessment_id: FLOW_SCALE_BLOCK_STABILITY_ID,
        plateau_point_count: policy.plateau_point_count,
        maximum_relative_standard_error_change: policy.maximum_relative_standard_error_change,
        maximum_relative_central_estimate_change: policy.maximum_relative_central_estimate_change,
        admissible_block_sizes,
        plateau_block_sizes,
        maximum_observed_relative_standard_error_change: maximum_se_change,
        maximum_observed_relative_central_estimate_change: maximum_central_change,
        enough_admissible_points,
        meets_uncertainty_plateau,
        meets_central_estimate_stability,
        meets_declared_policy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_flow_block_adequacy::FLOW_BLOCK_ADEQUACY_POLICY_ID;
    use crate::lattice_flow_joint_jackknife::FlowEnergyTrajectory;

    const TIMES: [f64; 6] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    const DIMENSIONLESS: [[f64; 6]; 12] = [
        [0.1120,0.1921,0.2542,0.3343,0.4064,0.4865],
        [0.1140,0.1901,0.2562,0.3323,0.4084,0.4845],
        [0.1160,0.1881,0.2582,0.3303,0.4104,0.4825],
        [0.1160,0.1987,0.2634,0.3461,0.4208,0.5035],
        [0.1180,0.1967,0.2654,0.3441,0.4228,0.5015],
        [0.1200,0.1947,0.2674,0.3421,0.4248,0.4995],
        [0.1198,0.2056,0.2734,0.3592,0.4370,0.5228],
        [0.1218,0.2036,0.2754,0.3572,0.4390,0.5208],
        [0.1238,0.2016,0.2774,0.3552,0.4410,0.5188],
        [0.1242,0.2116,0.2810,0.3684,0.4478,0.5352],
        [0.1262,0.2096,0.2830,0.3664,0.4498,0.5332],
        [0.1282,0.2076,0.2850,0.3644,0.4518,0.5312],
    ];

    fn input() -> JointBlockedJackknifeInput {
        JointBlockedJackknifeInput {
            flow_times: TIMES.to_vec(),
            trajectories: DIMENSIONLESS.iter().enumerate().map(|(index, values)| {
                FlowEnergyTrajectory {
                    chain_id: "chain-0".into(),
                    chain_position: index,
                    configuration_id: format!("cfg-{index:02}"),
                    energy_by_flow_time: values.iter().zip(TIMES).map(|(value, time)| {
                        value / (time * time)
                    }).collect(),
                }
            }).collect(),
            block_size: 1,
        }
    }

    fn adequacy(result: &JointBlockedJackknifeScale, pass: bool) -> BlockAdequacyAssessment {
        BlockAdequacyAssessment {
            policy_id: FLOW_BLOCK_ADEQUACY_POLICY_ID,
            minimum_tau_multiple: 2.0,
            minimum_block_count: 2,
            required_observable_ids: vec!["flow-energy".into()],
            supplied_observable_ids: vec!["flow-energy".into()],
            configuration_count: result.configuration_count,
            block_size: result.block_size,
            block_count: result.block_count,
            slowest_observable_id: "flow-energy".into(),
            slowest_tau_int: 1.0,
            required_block_size: 2,
            tau_window_complete: true,
            meets_tau_multiple: pass,
            meets_minimum_block_count: true,
            meets_declared_policy: pass,
            statistics_evidence_ids: vec!["stats-flow-energy".into()],
        }
    }

    #[test]
    fn block_scan_recomputes_nonlinear_uncertainty_at_each_size() {
        let scan = joint_jackknife_block_size_scan(
            &input(), JointFlowScaleKind::T0Like, 0.30, &[1, 2, 3, 4, 6],
        ).unwrap();
        let expected = [
            0.004_212_888_395_644_406,
            0.006_104_555_527_510_23,
            0.008_054_420_481_627_626,
            0.009_086_301_895_189_922,
            0.012_732_198_384_543_647,
        ];
        for (result, expected) in scan.results.iter().zip(expected) {
            assert!((result.standard_error - expected).abs() < 5.0e-15);
            assert!((result.central_estimate - 0.3375).abs() < 4.0e-15);
        }
    }

    #[test]
    fn stability_uses_only_largest_adequate_block_sizes() {
        let scan = joint_jackknife_block_size_scan(
            &input(), JointFlowScaleKind::T0Like, 0.30, &[1, 2, 3, 4, 6],
        ).unwrap();
        let adequate = scan.results.iter().enumerate().map(|(index, result)| {
            adequacy(result, index >= 2)
        }).collect::<Vec<_>>();
        let assessment = assess_flow_scale_block_stability(
            &scan,
            &adequate,
            &FlowScaleBlockStabilityPolicy {
                plateau_point_count: 3,
                maximum_relative_standard_error_change: 0.35,
                maximum_relative_central_estimate_change: 1.0e-10,
            },
        ).unwrap();
        assert_eq!(assessment.admissible_block_sizes, vec![3, 4, 6]);
        assert_eq!(assessment.plateau_block_sizes, vec![3, 4, 6]);
        assert!(assessment.enough_admissible_points);
        assert!(assessment.meets_central_estimate_stability);
        assert!(!assessment.meets_uncertainty_plateau);
        assert!(!assessment.meets_declared_policy);
    }

    #[test]
    fn caller_can_record_a_looser_plateau_policy_without_hiding_it() {
        let scan = joint_jackknife_block_size_scan(
            &input(), JointFlowScaleKind::T0Like, 0.30, &[1, 2, 3, 4, 6],
        ).unwrap();
        let adequate = scan.results.iter().enumerate().map(|(index, result)| {
            adequacy(result, index >= 2)
        }).collect::<Vec<_>>();
        let assessment = assess_flow_scale_block_stability(
            &scan,
            &adequate,
            &FlowScaleBlockStabilityPolicy {
                plateau_point_count: 3,
                maximum_relative_standard_error_change: 0.40,
                maximum_relative_central_estimate_change: 1.0e-10,
            },
        ).unwrap();
        assert!(assessment.meets_declared_policy);
        assert!(assessment.maximum_observed_relative_standard_error_change.unwrap() > 0.30);
    }
}
