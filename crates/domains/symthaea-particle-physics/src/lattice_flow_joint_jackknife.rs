// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Joint blocked jackknife for correlated gradient-flow scale trajectories.
//!
//! Each retained gauge configuration is one complete row across all declared
//! flow times. Every jackknife replicate removes the same contiguous row block
//! at every flow time, preserving cross-flow covariance. Independent Markov
//! chains remain explicit: blocks are formed within one chain only and are never
//! allowed to straddle a chain boundary.
//!
//! The block size is caller supplied and must be justified independently from
//! autocorrelation evidence; this module deliberately does not infer it.

use std::collections::BTreeSet;

use crate::lattice_flow_energy::{
    EnsembleFlowEnergyPoint, FlowEnergyError, t0_like_from_ensemble_mean,
    w0_like_from_ensemble_mean,
};

pub const JOINT_BLOCKED_JACKKNIFE_ID: &str = "joint_blocked_jackknife_v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointFlowScaleKind {
    T0Like,
    W0Like,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlowEnergyTrajectory {
    /// Stable identity of the independent Markov chain that produced this row.
    pub chain_id: String,
    /// Zero-based retained ordering inside `chain_id`.
    pub chain_position: usize,
    /// Stable identity of this retained configuration.
    pub configuration_id: String,
    /// Per-configuration clover energy E_i(t), one value per declared flow time.
    pub energy_by_flow_time: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JointBlockedJackknifeInput {
    pub flow_times: Vec<f64>,
    /// Rows must be grouped by chain, and each chain must be ordered by
    /// consecutive `chain_position` values starting at zero.
    pub trajectories: Vec<FlowEnergyTrajectory>,
    pub block_size: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JointBlockedJackknifeScale {
    pub method_id: &'static str,
    pub kind: JointFlowScaleKind,
    pub target: f64,
    pub central_estimate: f64,
    pub replicate_estimates: Vec<f64>,
    pub replicate_mean: f64,
    pub standard_error: f64,
    pub ensemble_mean_curve: Vec<EnsembleFlowEnergyPoint>,
    pub configuration_count: usize,
    pub independent_chain_count: usize,
    pub block_size: usize,
    pub block_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum JointJackknifeError {
    Energy(FlowEnergyError),
    InvalidBlockSize(usize),
    TooFewBlocks(usize),
    TooFewFlowTimes { required: usize, actual: usize },
    NonFiniteFlowTime { index: usize, value: f64 },
    NonPositiveFlowTime { index: usize, value: f64 },
    NonIncreasingFlowTime { previous: f64, current: f64 },
    EmptyChainId { index: usize },
    ChainReappeared { index: usize, chain_id: String },
    ChainPositionMismatch { index: usize, expected: usize, actual: usize },
    PartialTrailingChainBlock {
        chain_id: String,
        configurations: usize,
        block_size: usize,
    },
    EmptyConfigurationId { index: usize },
    DuplicateConfigurationId { index: usize },
    TrajectoryWidthMismatch { index: usize, expected: usize, actual: usize },
    NonFiniteEnergy { configuration: usize, flow_index: usize, value: f64 },
    NegativeEnergy { configuration: usize, flow_index: usize, value: f64 },
    NonFiniteMean { flow_index: usize },
    NonFiniteJackknifeStatistic,
}

impl From<FlowEnergyError> for JointJackknifeError {
    fn from(value: FlowEnergyError) -> Self {
        Self::Energy(value)
    }
}

impl JointBlockedJackknifeInput {
    fn validated_blocks(
        &self,
        kind: JointFlowScaleKind,
    ) -> Result<(Vec<(usize, usize)>, usize), JointJackknifeError> {
        let required_flow_times = match kind {
            JointFlowScaleKind::T0Like => 2,
            JointFlowScaleKind::W0Like => 4,
        };
        if self.flow_times.len() < required_flow_times {
            return Err(JointJackknifeError::TooFewFlowTimes {
                required: required_flow_times,
                actual: self.flow_times.len(),
            });
        }
        if self.block_size == 0 {
            return Err(JointJackknifeError::InvalidBlockSize(0));
        }
        for (index, &flow_time) in self.flow_times.iter().enumerate() {
            if !flow_time.is_finite() {
                return Err(JointJackknifeError::NonFiniteFlowTime { index, value: flow_time });
            }
            if flow_time <= 0.0 {
                return Err(JointJackknifeError::NonPositiveFlowTime { index, value: flow_time });
            }
            if index > 0 && flow_time <= self.flow_times[index - 1] {
                return Err(JointJackknifeError::NonIncreasingFlowTime {
                    previous: self.flow_times[index - 1],
                    current: flow_time,
                });
            }
        }

        let mut configuration_ids = BTreeSet::new();
        let mut closed_chains: BTreeSet<&str> = BTreeSet::new();
        let mut current_chain: Option<&str> = None;
        let mut current_chain_start = 0usize;
        let mut expected_chain_position = 0usize;
        let mut blocks = Vec::new();
        let mut chain_count = 0usize;

        let mut close_chain = |chain_id: &str, start: usize, end: usize| -> Result<(), JointJackknifeError> {
            let count = end - start;
            if count == 0 || count % self.block_size != 0 {
                return Err(JointJackknifeError::PartialTrailingChainBlock {
                    chain_id: chain_id.to_string(),
                    configurations: count,
                    block_size: self.block_size,
                });
            }
            for block_start in (start..end).step_by(self.block_size) {
                blocks.push((block_start, block_start + self.block_size));
            }
            Ok(())
        };

        for (index, trajectory) in self.trajectories.iter().enumerate() {
            let chain_id = trajectory.chain_id.trim();
            if chain_id.is_empty() {
                return Err(JointJackknifeError::EmptyChainId { index });
            }
            if current_chain != Some(chain_id) {
                if let Some(previous) = current_chain {
                    close_chain(previous, current_chain_start, index)?;
                    closed_chains.insert(previous);
                }
                if closed_chains.contains(chain_id) {
                    return Err(JointJackknifeError::ChainReappeared {
                        index,
                        chain_id: chain_id.to_string(),
                    });
                }
                current_chain = Some(chain_id);
                current_chain_start = index;
                expected_chain_position = 0;
                chain_count += 1;
            }
            if trajectory.chain_position != expected_chain_position {
                return Err(JointJackknifeError::ChainPositionMismatch {
                    index,
                    expected: expected_chain_position,
                    actual: trajectory.chain_position,
                });
            }
            expected_chain_position += 1;

            if trajectory.configuration_id.trim().is_empty() {
                return Err(JointJackknifeError::EmptyConfigurationId { index });
            }
            if !configuration_ids.insert(trajectory.configuration_id.as_str()) {
                return Err(JointJackknifeError::DuplicateConfigurationId { index });
            }
            if trajectory.energy_by_flow_time.len() != self.flow_times.len() {
                return Err(JointJackknifeError::TrajectoryWidthMismatch {
                    index,
                    expected: self.flow_times.len(),
                    actual: trajectory.energy_by_flow_time.len(),
                });
            }
            for (flow_index, &energy) in trajectory.energy_by_flow_time.iter().enumerate() {
                if !energy.is_finite() {
                    return Err(JointJackknifeError::NonFiniteEnergy {
                        configuration: index,
                        flow_index,
                        value: energy,
                    });
                }
                if energy < 0.0 {
                    return Err(JointJackknifeError::NegativeEnergy {
                        configuration: index,
                        flow_index,
                        value: energy,
                    });
                }
            }
        }
        if let Some(last_chain) = current_chain {
            close_chain(last_chain, current_chain_start, self.trajectories.len())?;
        }
        if blocks.len() < 2 {
            return Err(JointJackknifeError::TooFewBlocks(blocks.len()));
        }
        Ok((blocks, chain_count))
    }

    pub fn validate(&self, kind: JointFlowScaleKind) -> Result<(), JointJackknifeError> {
        self.validated_blocks(kind).map(|_| ())
    }
}

fn mean_curve_excluding(
    input: &JointBlockedJackknifeInput,
    excluded: Option<(usize, usize)>,
) -> Result<Vec<EnsembleFlowEnergyPoint>, JointJackknifeError> {
    let excluded_len = excluded.map_or(0, |(start, end)| end - start);
    let retained = input.trajectories.len() - excluded_len;
    let mut points = Vec::with_capacity(input.flow_times.len());
    for (flow_index, &flow_time) in input.flow_times.iter().enumerate() {
        let mut sum = 0.0;
        for (configuration, trajectory) in input.trajectories.iter().enumerate() {
            if excluded.is_some_and(|(start, end)| configuration >= start && configuration < end) {
                continue;
            }
            sum += trajectory.energy_by_flow_time[flow_index];
            if !sum.is_finite() {
                return Err(JointJackknifeError::NonFiniteMean { flow_index });
            }
        }
        let mean = sum / retained as f64;
        if !mean.is_finite() {
            return Err(JointJackknifeError::NonFiniteMean { flow_index });
        }
        points.push(EnsembleFlowEnergyPoint { flow_time, ensemble_mean_energy: mean });
    }
    Ok(points)
}

fn scale_from_curve(
    kind: JointFlowScaleKind,
    points: &[EnsembleFlowEnergyPoint],
    target: f64,
) -> Result<f64, JointJackknifeError> {
    let estimate = match kind {
        JointFlowScaleKind::T0Like => t0_like_from_ensemble_mean(points, target)?,
        JointFlowScaleKind::W0Like => w0_like_from_ensemble_mean(points, target)?,
    };
    if !estimate.is_finite() {
        return Err(JointJackknifeError::NonFiniteJackknifeStatistic);
    }
    Ok(estimate)
}

fn jackknife_summary(replicates: &[f64]) -> Result<(f64, f64), JointJackknifeError> {
    if replicates.len() < 2 || replicates.iter().any(|value| !value.is_finite()) {
        return Err(JointJackknifeError::NonFiniteJackknifeStatistic);
    }
    let replicate_mean = replicates.iter().sum::<f64>() / replicates.len() as f64;
    let square_sum = replicates
        .iter()
        .map(|value| (value - replicate_mean).powi(2))
        .sum::<f64>();
    let variance = (replicates.len() - 1) as f64 / replicates.len() as f64 * square_sum;
    let standard_error = variance.sqrt();
    if !replicate_mean.is_finite() || !standard_error.is_finite() {
        return Err(JointJackknifeError::NonFiniteJackknifeStatistic);
    }
    Ok((replicate_mean, standard_error))
}

/// Delete one complete contiguous block from one chain per replicate and
/// recompute the full nonlinear flow-scale estimator on every retained curve.
pub fn joint_blocked_jackknife_scale(
    input: &JointBlockedJackknifeInput,
    kind: JointFlowScaleKind,
    target: f64,
) -> Result<JointBlockedJackknifeScale, JointJackknifeError> {
    let (blocks, independent_chain_count) = input.validated_blocks(kind)?;
    let ensemble_mean_curve = mean_curve_excluding(input, None)?;
    let central_estimate = scale_from_curve(kind, &ensemble_mean_curve, target)?;
    let mut replicate_estimates = Vec::with_capacity(blocks.len());
    for block in &blocks {
        let replicate_curve = mean_curve_excluding(input, Some(*block))?;
        replicate_estimates.push(scale_from_curve(kind, &replicate_curve, target)?);
    }
    let (replicate_mean, standard_error) = jackknife_summary(&replicate_estimates)?;
    Ok(JointBlockedJackknifeScale {
        method_id: JOINT_BLOCKED_JACKKNIFE_ID,
        kind,
        target,
        central_estimate,
        replicate_estimates,
        replicate_mean,
        standard_error,
        ensemble_mean_curve,
        configuration_count: input.trajectories.len(),
        independent_chain_count,
        block_size: input.block_size,
        block_count: blocks.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TIMES: [f64; 6] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    const DIMENSIONLESS: [[f64; 6]; 12] = [
        [0.1120, 0.1921, 0.2542, 0.3343, 0.4064, 0.4865],
        [0.1140, 0.1901, 0.2562, 0.3323, 0.4084, 0.4845],
        [0.1160, 0.1881, 0.2582, 0.3303, 0.4104, 0.4825],
        [0.1160, 0.1987, 0.2634, 0.3461, 0.4208, 0.5035],
        [0.1180, 0.1967, 0.2654, 0.3441, 0.4228, 0.5015],
        [0.1200, 0.1947, 0.2674, 0.3421, 0.4248, 0.4995],
        [0.1198, 0.2056, 0.2734, 0.3592, 0.4370, 0.5228],
        [0.1218, 0.2036, 0.2754, 0.3572, 0.4390, 0.5208],
        [0.1238, 0.2016, 0.2774, 0.3552, 0.4410, 0.5188],
        [0.1242, 0.2116, 0.2810, 0.3684, 0.4478, 0.5352],
        [0.1262, 0.2096, 0.2830, 0.3664, 0.4498, 0.5332],
        [0.1282, 0.2076, 0.2850, 0.3644, 0.4518, 0.5312],
    ];

    fn oracle_input() -> JointBlockedJackknifeInput {
        JointBlockedJackknifeInput {
            flow_times: TIMES.to_vec(),
            trajectories: DIMENSIONLESS
                .iter()
                .enumerate()
                .map(|(index, values)| FlowEnergyTrajectory {
                    chain_id: "chain-0".into(),
                    chain_position: index,
                    configuration_id: format!("cfg-{index:02}"),
                    energy_by_flow_time: values
                        .iter()
                        .zip(TIMES)
                        .map(|(dimensionless, flow_time)| *dimensionless / (flow_time * flow_time))
                        .collect(),
                })
                .collect(),
            block_size: 3,
        }
    }

    fn assert_vector_close(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < tolerance, "{actual} != {expected}");
        }
    }

    #[test]
    fn t0_joint_jackknife_matches_independent_oracle() {
        let result = joint_blocked_jackknife_scale(&oracle_input(), JointFlowScaleKind::T0Like, 0.30).unwrap();
        assert_eq!(result.independent_chain_count, 1);
        assert_eq!(result.configuration_count, 12);
        assert_eq!(result.block_size, 3);
        assert_eq!(result.block_count, 4);
        assert!((result.central_estimate - 0.3375).abs() < 3.0e-15);
        assert_vector_close(
            &result.replicate_estimates,
            &[0.331_242_312_423_124_19, 0.335_391_628_677_994_16,
              0.340_050_377_833_753_14, 0.343_533_389_687_235_79],
            4.0e-15,
        );
        assert!((result.standard_error - 0.008_054_420_481_627_626).abs() < 4.0e-15);
    }

    #[test]
    fn w0_joint_jackknife_matches_independent_oracle() {
        let result = joint_blocked_jackknife_scale(&oracle_input(), JointFlowScaleKind::W0Like, 0.32).unwrap();
        assert!((result.central_estimate - 0.632_455_532_033_675_9).abs() < 3.0e-15);
        assert_vector_close(
            &result.replicate_estimates,
            &[0.628_172_116_290_049_76, 0.631_018_005_580_742_17,
              0.634_840_665_068_951_76, 0.636_983_593_239_096_26],
            4.0e-15,
        );
        assert!((result.standard_error - 0.005_889_670_897_688_516).abs() < 4.0e-15);
    }

    #[test]
    fn multiple_chains_keep_block_boundaries_inside_each_chain() {
        let mut input = oracle_input();
        for (index, trajectory) in input.trajectories.iter_mut().enumerate() {
            if index < 6 {
                trajectory.chain_id = "chain-a".into();
                trajectory.chain_position = index;
            } else {
                trajectory.chain_id = "chain-b".into();
                trajectory.chain_position = index - 6;
            }
        }
        let result = joint_blocked_jackknife_scale(&input, JointFlowScaleKind::T0Like, 0.30).unwrap();
        assert_eq!(result.independent_chain_count, 2);
        assert_eq!(result.block_count, 4);
        assert_eq!(result.replicate_estimates.len(), 4);
    }

    #[test]
    fn chain_reappearance_fails_closed() {
        let mut input = oracle_input();
        input.trajectories[3].chain_id = "chain-b".into();
        input.trajectories[3].chain_position = 0;
        input.trajectories[4].chain_id = "chain-0".into();
        input.trajectories[4].chain_position = 3;
        assert!(matches!(
            input.validate(JointFlowScaleKind::T0Like),
            Err(JointJackknifeError::ChainReappeared { index: 4, .. })
        ));
    }

    #[test]
    fn partial_chain_block_fails_closed() {
        let mut input = oracle_input();
        input.trajectories.pop();
        assert!(matches!(
            input.validate(JointFlowScaleKind::T0Like),
            Err(JointJackknifeError::PartialTrailingChainBlock {
                configurations: 11,
                block_size: 3,
                ..
            })
        ));
    }

    #[test]
    fn duplicate_configuration_ids_fail_closed() {
        let mut input = oracle_input();
        input.trajectories[1].configuration_id = input.trajectories[0].configuration_id.clone();
        assert!(matches!(
            input.validate(JointFlowScaleKind::T0Like),
            Err(JointJackknifeError::DuplicateConfigurationId { index: 1 })
        ));
    }
}
