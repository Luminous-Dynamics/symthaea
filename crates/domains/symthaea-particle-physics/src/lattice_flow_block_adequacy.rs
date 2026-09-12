// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence contract for joint-jackknife block-size adequacy.
//!
//! Autocorrelation estimation remains external (for example LQCD-003/#1629).
//! This module consumes qualified `tau_int` evidence and evaluates the already-
//! bound resampling geometry against an explicit caller-declared policy.

use std::collections::{BTreeMap, BTreeSet};

use crate::lattice_flow_joint_evidence::JointJackknifeScaleEstimateEvidence;
use crate::lattice_flow_joint_jackknife::JOINT_BLOCKED_JACKKNIFE_ID;

pub const FLOW_BLOCK_ADEQUACY_POLICY_ID: &str = "flow_block_adequacy_v1";

#[derive(Debug, Clone, PartialEq)]
pub struct AutocorrelationEvidence {
    pub observable_id: String,
    pub tau_int: f64,
    pub retained_sample_count: usize,
    pub positive_lag_count: usize,
    pub max_lag: usize,
    pub ensemble_manifest_digest: String,
    pub statistics_evidence_id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlockAdequacyPolicy {
    pub minimum_tau_multiple: f64,
    pub minimum_block_count: usize,
    pub required_observable_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlockAdequacyAssessment {
    pub policy_id: &'static str,
    pub minimum_tau_multiple: f64,
    pub minimum_block_count: usize,
    pub required_observable_ids: Vec<String>,
    pub supplied_observable_ids: Vec<String>,
    pub configuration_count: usize,
    pub independent_chain_count: usize,
    pub block_size: usize,
    pub block_count: usize,
    pub slowest_observable_id: String,
    pub slowest_tau_int: f64,
    pub required_block_size: usize,
    pub tau_window_complete: bool,
    pub meets_tau_multiple: bool,
    pub meets_minimum_block_count: bool,
    pub meets_declared_policy: bool,
    pub statistics_evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BlockAdequacyError {
    WrongResamplingMethod,
    InvalidBoundResamplingGeometry,
    InvalidPolicyTauMultiple(f64),
    InvalidPolicyMinimumBlockCount(usize),
    EmptyRequiredObservableSet,
    EmptyRequiredObservableId { index: usize },
    DuplicateRequiredObservableId { observable_id: String },
    EmptyAutocorrelationEvidence,
    EmptyObservableId { index: usize },
    DuplicateObservableId { observable_id: String },
    EmptyStatisticsEvidenceId { index: usize },
    EmptyEnsembleManifestDigest { index: usize },
    EnsembleManifestMismatch { index: usize },
    RetainedSampleCountMismatch { index: usize, expected: usize, actual: usize },
    InvalidTauInt { index: usize, value: f64 },
    InvalidMaxLag { index: usize, max_lag: usize, retained_sample_count: usize },
    PositiveLagCountExceedsMaxLag { index: usize, positive_lag_count: usize, max_lag: usize },
    MissingRequiredObservable { observable_id: String },
    RequiredBlockSizeOverflow,
}

fn validate_policy(policy: &BlockAdequacyPolicy) -> Result<BTreeSet<&str>, BlockAdequacyError> {
    if !policy.minimum_tau_multiple.is_finite() || policy.minimum_tau_multiple <= 0.0 {
        return Err(BlockAdequacyError::InvalidPolicyTauMultiple(policy.minimum_tau_multiple));
    }
    if policy.minimum_block_count < 2 {
        return Err(BlockAdequacyError::InvalidPolicyMinimumBlockCount(policy.minimum_block_count));
    }
    if policy.required_observable_ids.is_empty() {
        return Err(BlockAdequacyError::EmptyRequiredObservableSet);
    }
    let mut required = BTreeSet::new();
    for (index, observable_id) in policy.required_observable_ids.iter().enumerate() {
        let id = observable_id.trim();
        if id.is_empty() {
            return Err(BlockAdequacyError::EmptyRequiredObservableId { index });
        }
        if !required.insert(id) {
            return Err(BlockAdequacyError::DuplicateRequiredObservableId { observable_id: id.into() });
        }
    }
    Ok(required)
}

pub fn assess_flow_block_adequacy(
    scale: &JointJackknifeScaleEstimateEvidence,
    autocorrelation: &[AutocorrelationEvidence],
    policy: &BlockAdequacyPolicy,
) -> Result<BlockAdequacyAssessment, BlockAdequacyError> {
    if scale.resampling_method_id != JOINT_BLOCKED_JACKKNIFE_ID {
        return Err(BlockAdequacyError::WrongResamplingMethod);
    }
    if scale.block_size == 0
        || scale.block_count < 2
        || scale.independent_chain_count == 0
        || scale.independent_chain_count > scale.block_count
        || scale.block_size.checked_mul(scale.block_count) != Some(scale.configuration_count)
        || scale.replicate_count != scale.block_count
        || scale.replicate_estimates.len() != scale.replicate_count
    {
        return Err(BlockAdequacyError::InvalidBoundResamplingGeometry);
    }
    let required = validate_policy(policy)?;
    if autocorrelation.is_empty() {
        return Err(BlockAdequacyError::EmptyAutocorrelationEvidence);
    }

    let expected_manifest = scale.scale.ensemble_manifest_digest.as_str();
    let mut by_observable: BTreeMap<&str, &AutocorrelationEvidence> = BTreeMap::new();
    let mut tau_window_complete = true;
    for (index, evidence) in autocorrelation.iter().enumerate() {
        let id = evidence.observable_id.trim();
        if id.is_empty() { return Err(BlockAdequacyError::EmptyObservableId { index }); }
        if by_observable.insert(id, evidence).is_some() {
            return Err(BlockAdequacyError::DuplicateObservableId { observable_id: id.into() });
        }
        if evidence.statistics_evidence_id.trim().is_empty() {
            return Err(BlockAdequacyError::EmptyStatisticsEvidenceId { index });
        }
        if evidence.ensemble_manifest_digest.trim().is_empty() {
            return Err(BlockAdequacyError::EmptyEnsembleManifestDigest { index });
        }
        if evidence.ensemble_manifest_digest != expected_manifest {
            return Err(BlockAdequacyError::EnsembleManifestMismatch { index });
        }
        if evidence.retained_sample_count != scale.configuration_count {
            return Err(BlockAdequacyError::RetainedSampleCountMismatch {
                index, expected: scale.configuration_count, actual: evidence.retained_sample_count,
            });
        }
        if !evidence.tau_int.is_finite() || evidence.tau_int < 0.5 {
            return Err(BlockAdequacyError::InvalidTauInt { index, value: evidence.tau_int });
        }
        if evidence.max_lag == 0 || evidence.max_lag >= evidence.retained_sample_count {
            return Err(BlockAdequacyError::InvalidMaxLag {
                index, max_lag: evidence.max_lag,
                retained_sample_count: evidence.retained_sample_count,
            });
        }
        if evidence.positive_lag_count > evidence.max_lag {
            return Err(BlockAdequacyError::PositiveLagCountExceedsMaxLag {
                index, positive_lag_count: evidence.positive_lag_count, max_lag: evidence.max_lag,
            });
        }
        tau_window_complete &= evidence.positive_lag_count < evidence.max_lag;
    }
    for id in required.iter().copied() {
        if !by_observable.contains_key(id) {
            return Err(BlockAdequacyError::MissingRequiredObservable { observable_id: id.into() });
        }
    }

    let slowest = by_observable.values().copied().max_by(|a, b| a.tau_int.total_cmp(&b.tau_int))
        .ok_or(BlockAdequacyError::EmptyAutocorrelationEvidence)?;
    let required_f64 = (policy.minimum_tau_multiple * slowest.tau_int).ceil();
    if !required_f64.is_finite() || required_f64 > usize::MAX as f64 {
        return Err(BlockAdequacyError::RequiredBlockSizeOverflow);
    }
    let required_block_size = required_f64.max(1.0) as usize;
    let meets_tau_multiple = scale.block_size >= required_block_size;
    let meets_minimum_block_count = scale.block_count >= policy.minimum_block_count;

    let mut required_observable_ids = required.into_iter().map(str::to_string).collect::<Vec<_>>();
    required_observable_ids.sort();
    let mut supplied_observable_ids = by_observable.keys().map(|id| (*id).to_string()).collect::<Vec<_>>();
    supplied_observable_ids.sort();
    let mut statistics_evidence_ids = autocorrelation.iter().map(|e| e.statistics_evidence_id.clone()).collect::<Vec<_>>();
    statistics_evidence_ids.sort();

    Ok(BlockAdequacyAssessment {
        policy_id: FLOW_BLOCK_ADEQUACY_POLICY_ID,
        minimum_tau_multiple: policy.minimum_tau_multiple,
        minimum_block_count: policy.minimum_block_count,
        required_observable_ids,
        supplied_observable_ids,
        configuration_count: scale.configuration_count,
        independent_chain_count: scale.independent_chain_count,
        block_size: scale.block_size,
        block_count: scale.block_count,
        slowest_observable_id: slowest.observable_id.clone(),
        slowest_tau_int: slowest.tau_int,
        required_block_size,
        tau_window_complete,
        meets_tau_multiple,
        meets_minimum_block_count,
        meets_declared_policy: tau_window_complete && meets_tau_multiple && meets_minimum_block_count,
        statistics_evidence_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_flow_scale_evidence::{FlowScaleEstimateEvidence, FlowScaleKind};

    fn bound(block_size: usize, block_count: usize, chains: usize) -> JointJackknifeScaleEstimateEvidence {
        JointJackknifeScaleEstimateEvidence {
            scale: FlowScaleEstimateEvidence {
                kind: FlowScaleKind::T0Like, target: 0.30, estimate: 0.35, standard_error: 0.01,
                energy_operator_id: "symthaea_clover_energy_v1".into(),
                flow_implementation_id: "wilson_action_staple_rk3_v1".into(), flow_step_size: 0.001,
                ensemble_manifest_digest: "ensemble-sha256".into(), curve_artifact_digest: "curve-sha256".into(),
                joint_resampling_evidence_id: "jackknife-evidence".into(),
            },
            resampling_method_id: JOINT_BLOCKED_JACKKNIFE_ID,
            configuration_count: block_size * block_count,
            independent_chain_count: chains,
            block_size, block_count, replicate_count: block_count,
            replicate_mean: 0.35, replicate_estimates: vec![0.35; block_count],
            resampling_artifact_digest: "jackknife-artifact".into(),
        }
    }

    fn ac(id: &str, tau: f64, positive: usize, max_lag: usize, n: usize) -> AutocorrelationEvidence {
        AutocorrelationEvidence {
            observable_id: id.into(), tau_int: tau, retained_sample_count: n,
            positive_lag_count: positive, max_lag,
            ensemble_manifest_digest: "ensemble-sha256".into(),
            statistics_evidence_id: format!("stats-{id}"),
        }
    }

    fn policy() -> BlockAdequacyPolicy {
        BlockAdequacyPolicy {
            minimum_tau_multiple: 2.0, minimum_block_count: 4,
            required_observable_ids: vec!["flow-energy".into(), "topological-q".into()],
        }
    }

    #[test]
    fn declared_policy_passes_only_with_complete_slow_mode_coverage() {
        let scale = bound(6, 4, 2);
        let evidence = [ac("flow-energy", 1.4, 3, 12, 24), ac("topological-q", 2.2, 5, 12, 24)];
        let result = assess_flow_block_adequacy(&scale, &evidence, &policy()).unwrap();
        assert_eq!(result.required_block_size, 5);
        assert_eq!(result.slowest_observable_id, "topological-q");
        assert!(result.meets_declared_policy);
    }

    #[test]
    fn undersized_or_window_truncated_blocks_do_not_pass() {
        let small = bound(4, 6, 2);
        let evidence = [ac("flow-energy", 1.4, 3, 12, 24), ac("topological-q", 2.2, 5, 12, 24)];
        assert!(!assess_flow_block_adequacy(&small, &evidence, &policy()).unwrap().meets_declared_policy);

        let scale = bound(6, 4, 2);
        let truncated = [ac("flow-energy", 1.4, 3, 12, 24), ac("topological-q", 2.2, 12, 12, 24)];
        let result = assess_flow_block_adequacy(&scale, &truncated, &policy()).unwrap();
        assert!(!result.tau_window_complete);
        assert!(!result.meets_declared_policy);
    }

    #[test]
    fn required_observable_cannot_be_omitted() {
        let scale = bound(6, 4, 2);
        let evidence = [ac("flow-energy", 1.4, 3, 12, 24)];
        assert!(matches!(
            assess_flow_block_adequacy(&scale, &evidence, &policy()),
            Err(BlockAdequacyError::MissingRequiredObservable { observable_id }) if observable_id == "topological-q"
        ));
    }

    #[test]
    fn ensemble_population_and_bound_geometry_are_verified() {
        let scale = bound(6, 4, 2);
        let mut evidence = vec![ac("flow-energy", 1.4, 3, 12, 24), ac("topological-q", 2.2, 5, 12, 24)];
        evidence[0].ensemble_manifest_digest = "other".into();
        assert!(matches!(assess_flow_block_adequacy(&scale, &evidence, &policy()), Err(BlockAdequacyError::EnsembleManifestMismatch { index: 0 })));

        let mut forged = scale.clone(); forged.replicate_count = 3;
        assert!(matches!(assess_flow_block_adequacy(&forged, &[], &policy()), Err(BlockAdequacyError::InvalidBoundResamplingGeometry)));
    }
}
