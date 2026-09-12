// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Policy-bound ensemble-promotion assessment for gradient-flow scale evidence.
//!
//! This module composes already-produced diagnostics. It does not infer burn-in,
//! estimate R-hat/ESS, count topology transitions, or choose universal thresholds.
//! A caller must declare the acceptance policy explicitly. The result records each
//! gate separately so `meets_declared_policy` cannot be mistaken for a theorem of
//! equilibrium, ergodicity, or physical adequacy.

use std::collections::{BTreeMap, BTreeSet};

use crate::lattice_flow_joint_evidence::JointJackknifeScaleEstimateEvidence;
use crate::lattice_flow_joint_jackknife::JOINT_BLOCKED_JACKKNIFE_ID;
use crate::lattice_flow_scale_stability::{
    FLOW_SCALE_BLOCK_STABILITY_ID, FlowScaleBlockStabilityAssessment,
};

pub const ENSEMBLE_SCALE_PROMOTION_ASSESSMENT_ID: &str =
    "ensemble_scale_promotion_assessment_v1";

#[derive(Debug, Clone, PartialEq)]
pub struct ObservableEquilibriumEvidence {
    pub observable_id: String,
    pub retained_sample_count: usize,
    pub independent_chain_count: usize,
    pub nominal_burn_in: usize,
    pub comparison_burn_in: usize,
    /// Absolute shift between declared burn-in choices, in caller-defined
    /// combined-standard-error units.
    pub burn_in_normalized_shift: f64,
    /// Conservative rank/folded multi-chain R-hat summary.
    pub rank_normalized_max_r_hat: f64,
    pub effective_sample_size: f64,
    pub ensemble_manifest_digest: String,
    pub burn_in_evidence_id: String,
    pub convergence_evidence_id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TopologyPromotionEvidence {
    pub observable_id: String,
    pub retained_sample_count: usize,
    pub independent_chain_count: usize,
    pub zero_variance_observed: bool,
    pub tau_window_complete: bool,
    pub effective_sample_size: Option<f64>,
    pub sector_change_count: usize,
    pub ensemble_manifest_digest: String,
    pub topology_evidence_id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnsembleScalePromotionPolicy {
    pub maximum_burn_in_normalized_shift: f64,
    pub maximum_rank_normalized_r_hat: f64,
    pub minimum_effective_sample_size: f64,
    pub minimum_topology_effective_sample_size: f64,
    pub minimum_topology_sector_changes: usize,
    pub required_observable_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnsembleScalePromotionAssessment {
    pub assessment_id: &'static str,
    pub ensemble_manifest_digest: String,
    pub configuration_count: usize,
    pub independent_chain_count: usize,
    pub selected_block_size: usize,
    pub block_stability_evidence_id: String,
    pub required_observable_ids: Vec<String>,
    pub supplied_observable_ids: Vec<String>,
    pub worst_burn_in_observable_id: String,
    pub maximum_observed_burn_in_normalized_shift: f64,
    pub worst_r_hat_observable_id: String,
    pub maximum_observed_rank_normalized_r_hat: f64,
    pub worst_ess_observable_id: String,
    pub minimum_observed_effective_sample_size: f64,
    pub topology_observable_id: String,
    pub topology_effective_sample_size: Option<f64>,
    pub topology_sector_change_count: usize,
    pub selected_block_is_on_plateau: bool,
    pub block_stability_satisfied: bool,
    pub burn_in_sensitivity_satisfied: bool,
    pub multi_chain_r_hat_satisfied: bool,
    pub effective_sample_size_satisfied: bool,
    pub topology_nonzero_variance: bool,
    pub topology_tau_window_complete: bool,
    pub topology_effective_sample_size_satisfied: bool,
    pub topology_sector_changes_satisfied: bool,
    pub meets_declared_policy: bool,
    pub evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EnsembleScalePromotionError {
    WrongResamplingMethod,
    InvalidScaleEvidenceGeometry,
    WrongBlockStabilityAssessment,
    InvalidBlockStabilityRecord,
    EmptyBlockStabilityEvidenceId,
    EmptyEnsembleManifestDigest,
    InvalidMaximumBurnInShift(f64),
    InvalidMaximumRHat(f64),
    InvalidMinimumEffectiveSampleSize(f64),
    InvalidMinimumTopologyEffectiveSampleSize(f64),
    EmptyRequiredObservableSet,
    EmptyRequiredObservableId { index: usize },
    DuplicateRequiredObservableId { observable_id: String },
    EmptyObservableEvidence,
    EmptyObservableId { index: usize },
    DuplicateObservableId { observable_id: String },
    MissingRequiredObservable { observable_id: String },
    ObservableManifestMismatch { index: usize },
    ObservablePopulationMismatch { index: usize },
    InvalidBurnInComparison { index: usize },
    InvalidBurnInShift { index: usize, value: f64 },
    InvalidRHat { index: usize, value: f64 },
    InvalidEffectiveSampleSize { index: usize, value: f64 },
    EmptyEvidenceId { index: Option<usize>, field: &'static str },
    TopologyManifestMismatch,
    TopologyPopulationMismatch,
    InvalidTopologyEffectiveSampleSize(f64),
}

fn require_nonempty(
    value: &str,
    index: Option<usize>,
    field: &'static str,
) -> Result<(), EnsembleScalePromotionError> {
    if value.trim().is_empty() {
        Err(EnsembleScalePromotionError::EmptyEvidenceId { index, field })
    } else {
        Ok(())
    }
}

fn validate_policy(
    policy: &EnsembleScalePromotionPolicy,
) -> Result<BTreeSet<&str>, EnsembleScalePromotionError> {
    if !policy.maximum_burn_in_normalized_shift.is_finite()
        || policy.maximum_burn_in_normalized_shift < 0.0
    {
        return Err(EnsembleScalePromotionError::InvalidMaximumBurnInShift(
            policy.maximum_burn_in_normalized_shift,
        ));
    }
    if !policy.maximum_rank_normalized_r_hat.is_finite()
        || policy.maximum_rank_normalized_r_hat <= 0.0
    {
        return Err(EnsembleScalePromotionError::InvalidMaximumRHat(
            policy.maximum_rank_normalized_r_hat,
        ));
    }
    if !policy.minimum_effective_sample_size.is_finite()
        || policy.minimum_effective_sample_size <= 0.0
    {
        return Err(EnsembleScalePromotionError::InvalidMinimumEffectiveSampleSize(
            policy.minimum_effective_sample_size,
        ));
    }
    if !policy.minimum_topology_effective_sample_size.is_finite()
        || policy.minimum_topology_effective_sample_size <= 0.0
    {
        return Err(EnsembleScalePromotionError::InvalidMinimumTopologyEffectiveSampleSize(
            policy.minimum_topology_effective_sample_size,
        ));
    }
    if policy.required_observable_ids.is_empty() {
        return Err(EnsembleScalePromotionError::EmptyRequiredObservableSet);
    }
    let mut required = BTreeSet::new();
    for (index, observable_id) in policy.required_observable_ids.iter().enumerate() {
        let observable_id = observable_id.trim();
        if observable_id.is_empty() {
            return Err(EnsembleScalePromotionError::EmptyRequiredObservableId { index });
        }
        if !required.insert(observable_id) {
            return Err(EnsembleScalePromotionError::DuplicateRequiredObservableId {
                observable_id: observable_id.to_string(),
            });
        }
    }
    Ok(required)
}

fn validate_scale_subject(
    scale: &JointJackknifeScaleEstimateEvidence,
) -> Result<(), EnsembleScalePromotionError> {
    let valid_geometry = scale.configuration_count > 0
        && scale.independent_chain_count > 0
        && scale.independent_chain_count <= scale.block_count
        && scale.block_size > 0
        && scale.block_count >= 2
        && scale.block_size.checked_mul(scale.block_count) == Some(scale.configuration_count)
        && scale.replicate_count == scale.block_count
        && scale.replicate_estimates.len() == scale.replicate_count;
    if scale.resampling_method_id != JOINT_BLOCKED_JACKKNIFE_ID {
        return Err(EnsembleScalePromotionError::WrongResamplingMethod);
    }
    if !valid_geometry {
        return Err(EnsembleScalePromotionError::InvalidScaleEvidenceGeometry);
    }
    if scale.scale.ensemble_manifest_digest.trim().is_empty() {
        return Err(EnsembleScalePromotionError::EmptyEnsembleManifestDigest);
    }
    Ok(())
}

fn validate_block_stability(
    stability: &FlowScaleBlockStabilityAssessment,
) -> Result<(), EnsembleScalePromotionError> {
    let consistent = stability.assessment_id == FLOW_SCALE_BLOCK_STABILITY_ID
        && stability.plateau_point_count >= 2
        && stability.meets_declared_policy
            == (stability.enough_admissible_points
                && stability.meets_uncertainty_plateau
                && stability.meets_central_estimate_stability)
        && (!stability.enough_admissible_points
            || stability.plateau_block_sizes.len() == stability.plateau_point_count);
    if stability.assessment_id != FLOW_SCALE_BLOCK_STABILITY_ID {
        return Err(EnsembleScalePromotionError::WrongBlockStabilityAssessment);
    }
    if !consistent {
        return Err(EnsembleScalePromotionError::InvalidBlockStabilityRecord);
    }
    Ok(())
}

pub fn assess_ensemble_scale_promotion(
    scale_evidence: &JointJackknifeScaleEstimateEvidence,
    block_stability: &FlowScaleBlockStabilityAssessment,
    block_stability_evidence_id: &str,
    observables: &[ObservableEquilibriumEvidence],
    topology: &TopologyPromotionEvidence,
    policy: &EnsembleScalePromotionPolicy,
) -> Result<EnsembleScalePromotionAssessment, EnsembleScalePromotionError> {
    validate_scale_subject(scale_evidence)?;
    validate_block_stability(block_stability)?;
    if block_stability_evidence_id.trim().is_empty() {
        return Err(EnsembleScalePromotionError::EmptyBlockStabilityEvidenceId);
    }
    let required = validate_policy(policy)?;
    if observables.is_empty() {
        return Err(EnsembleScalePromotionError::EmptyObservableEvidence);
    }

    let manifest = scale_evidence.scale.ensemble_manifest_digest.as_str();
    let configuration_count = scale_evidence.configuration_count;
    let independent_chain_count = scale_evidence.independent_chain_count;
    let mut by_observable = BTreeMap::new();
    let mut worst_burn_in: Option<&ObservableEquilibriumEvidence> = None;
    let mut worst_r_hat: Option<&ObservableEquilibriumEvidence> = None;
    let mut worst_ess: Option<&ObservableEquilibriumEvidence> = None;
    let mut evidence_ids = vec![block_stability_evidence_id.to_string()];

    for (index, evidence) in observables.iter().enumerate() {
        let observable_id = evidence.observable_id.trim();
        if observable_id.is_empty() {
            return Err(EnsembleScalePromotionError::EmptyObservableId { index });
        }
        if by_observable.insert(observable_id, evidence).is_some() {
            return Err(EnsembleScalePromotionError::DuplicateObservableId {
                observable_id: observable_id.to_string(),
            });
        }
        if evidence.ensemble_manifest_digest != manifest {
            return Err(EnsembleScalePromotionError::ObservableManifestMismatch { index });
        }
        if evidence.retained_sample_count != configuration_count
            || evidence.independent_chain_count != independent_chain_count
        {
            return Err(EnsembleScalePromotionError::ObservablePopulationMismatch { index });
        }
        if evidence.comparison_burn_in <= evidence.nominal_burn_in {
            return Err(EnsembleScalePromotionError::InvalidBurnInComparison { index });
        }
        if !evidence.burn_in_normalized_shift.is_finite()
            || evidence.burn_in_normalized_shift < 0.0
        {
            return Err(EnsembleScalePromotionError::InvalidBurnInShift {
                index,
                value: evidence.burn_in_normalized_shift,
            });
        }
        if !evidence.rank_normalized_max_r_hat.is_finite()
            || evidence.rank_normalized_max_r_hat <= 0.0
        {
            return Err(EnsembleScalePromotionError::InvalidRHat {
                index,
                value: evidence.rank_normalized_max_r_hat,
            });
        }
        if !evidence.effective_sample_size.is_finite()
            || evidence.effective_sample_size <= 0.0
        {
            return Err(EnsembleScalePromotionError::InvalidEffectiveSampleSize {
                index,
                value: evidence.effective_sample_size,
            });
        }
        require_nonempty(&evidence.burn_in_evidence_id, Some(index), "burn_in_evidence_id")?;
        require_nonempty(&evidence.convergence_evidence_id, Some(index), "convergence_evidence_id")?;
        evidence_ids.push(evidence.burn_in_evidence_id.clone());
        evidence_ids.push(evidence.convergence_evidence_id.clone());

        if worst_burn_in.is_none_or(|current| evidence.burn_in_normalized_shift > current.burn_in_normalized_shift) {
            worst_burn_in = Some(evidence);
        }
        if worst_r_hat.is_none_or(|current| evidence.rank_normalized_max_r_hat > current.rank_normalized_max_r_hat) {
            worst_r_hat = Some(evidence);
        }
        if worst_ess.is_none_or(|current| evidence.effective_sample_size < current.effective_sample_size) {
            worst_ess = Some(evidence);
        }
    }

    for observable_id in required.iter().copied() {
        if !by_observable.contains_key(observable_id) {
            return Err(EnsembleScalePromotionError::MissingRequiredObservable {
                observable_id: observable_id.to_string(),
            });
        }
    }

    if topology.ensemble_manifest_digest != manifest {
        return Err(EnsembleScalePromotionError::TopologyManifestMismatch);
    }
    if topology.retained_sample_count != configuration_count
        || topology.independent_chain_count != independent_chain_count
    {
        return Err(EnsembleScalePromotionError::TopologyPopulationMismatch);
    }
    require_nonempty(&topology.observable_id, None, "topology_observable_id")?;
    require_nonempty(&topology.topology_evidence_id, None, "topology_evidence_id")?;
    if let Some(ess) = topology.effective_sample_size {
        if !ess.is_finite() || ess <= 0.0 {
            return Err(EnsembleScalePromotionError::InvalidTopologyEffectiveSampleSize(ess));
        }
    }
    evidence_ids.push(topology.topology_evidence_id.clone());
    evidence_ids.sort();
    evidence_ids.dedup();

    let worst_burn_in = worst_burn_in.expect("non-empty observables validated above");
    let worst_r_hat = worst_r_hat.expect("non-empty observables validated above");
    let worst_ess = worst_ess.expect("non-empty observables validated above");

    let selected_block_is_on_plateau = block_stability
        .plateau_block_sizes
        .contains(&scale_evidence.block_size);
    let block_stability_satisfied =
        block_stability.meets_declared_policy && selected_block_is_on_plateau;
    let burn_in_sensitivity_satisfied = worst_burn_in.burn_in_normalized_shift
        <= policy.maximum_burn_in_normalized_shift;
    let multi_chain_r_hat_satisfied = worst_r_hat.rank_normalized_max_r_hat
        <= policy.maximum_rank_normalized_r_hat;
    let effective_sample_size_satisfied =
        worst_ess.effective_sample_size >= policy.minimum_effective_sample_size;
    let topology_nonzero_variance = !topology.zero_variance_observed;
    let topology_tau_window_complete = topology.tau_window_complete;
    let topology_effective_sample_size_satisfied = topology
        .effective_sample_size
        .is_some_and(|ess| ess >= policy.minimum_topology_effective_sample_size);
    let topology_sector_changes_satisfied =
        topology.sector_change_count >= policy.minimum_topology_sector_changes;

    let meets_declared_policy = block_stability_satisfied
        && burn_in_sensitivity_satisfied
        && multi_chain_r_hat_satisfied
        && effective_sample_size_satisfied
        && topology_nonzero_variance
        && topology_tau_window_complete
        && topology_effective_sample_size_satisfied
        && topology_sector_changes_satisfied;

    let mut required_observable_ids = required.into_iter().map(str::to_string).collect::<Vec<_>>();
    required_observable_ids.sort();
    let mut supplied_observable_ids = by_observable.keys().map(|value| (*value).to_string()).collect::<Vec<_>>();
    supplied_observable_ids.sort();

    Ok(EnsembleScalePromotionAssessment {
        assessment_id: ENSEMBLE_SCALE_PROMOTION_ASSESSMENT_ID,
        ensemble_manifest_digest: manifest.to_string(),
        configuration_count,
        independent_chain_count,
        selected_block_size: scale_evidence.block_size,
        block_stability_evidence_id: block_stability_evidence_id.to_string(),
        required_observable_ids,
        supplied_observable_ids,
        worst_burn_in_observable_id: worst_burn_in.observable_id.clone(),
        maximum_observed_burn_in_normalized_shift: worst_burn_in.burn_in_normalized_shift,
        worst_r_hat_observable_id: worst_r_hat.observable_id.clone(),
        maximum_observed_rank_normalized_r_hat: worst_r_hat.rank_normalized_max_r_hat,
        worst_ess_observable_id: worst_ess.observable_id.clone(),
        minimum_observed_effective_sample_size: worst_ess.effective_sample_size,
        topology_observable_id: topology.observable_id.clone(),
        topology_effective_sample_size: topology.effective_sample_size,
        topology_sector_change_count: topology.sector_change_count,
        selected_block_is_on_plateau,
        block_stability_satisfied,
        burn_in_sensitivity_satisfied,
        multi_chain_r_hat_satisfied,
        effective_sample_size_satisfied,
        topology_nonzero_variance,
        topology_tau_window_complete,
        topology_effective_sample_size_satisfied,
        topology_sector_changes_satisfied,
        meets_declared_policy,
        evidence_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_flow_scale_evidence::{FlowScaleEstimateEvidence, FlowScaleKind};

    fn scale() -> JointJackknifeScaleEstimateEvidence {
        JointJackknifeScaleEstimateEvidence {
            scale: FlowScaleEstimateEvidence {
                kind: FlowScaleKind::T0Like,
                target: 0.30,
                estimate: 0.35,
                standard_error: 0.01,
                energy_operator_id: "symthaea_clover_energy_v1".into(),
                flow_implementation_id: "wilson_action_staple_rk3_v1".into(),
                flow_step_size: 0.001,
                ensemble_manifest_digest: "ensemble-sha256".into(),
                curve_artifact_digest: "curve-sha256".into(),
                joint_resampling_evidence_id: "jackknife-evidence".into(),
            },
            resampling_method_id: JOINT_BLOCKED_JACKKNIFE_ID,
            configuration_count: 24,
            independent_chain_count: 2,
            block_size: 6,
            block_count: 4,
            replicate_count: 4,
            replicate_mean: 0.35,
            replicate_estimates: vec![0.34, 0.345, 0.355, 0.36],
            resampling_artifact_digest: "jackknife-artifact".into(),
        }
    }

    fn stability(pass: bool) -> FlowScaleBlockStabilityAssessment {
        FlowScaleBlockStabilityAssessment {
            assessment_id: FLOW_SCALE_BLOCK_STABILITY_ID,
            plateau_point_count: 3,
            maximum_relative_standard_error_change: 0.2,
            maximum_relative_central_estimate_change: 0.01,
            admissible_block_sizes: vec![4, 6, 8],
            plateau_block_sizes: vec![4, 6, 8],
            maximum_observed_relative_standard_error_change: Some(0.1),
            maximum_observed_relative_central_estimate_change: Some(0.0),
            enough_admissible_points: true,
            meets_uncertainty_plateau: pass,
            meets_central_estimate_stability: true,
            meets_declared_policy: pass,
        }
    }

    fn observable(id: &str, burn: f64, rhat: f64, ess: f64) -> ObservableEquilibriumEvidence {
        ObservableEquilibriumEvidence {
            observable_id: id.into(),
            retained_sample_count: 24,
            independent_chain_count: 2,
            nominal_burn_in: 100,
            comparison_burn_in: 150,
            burn_in_normalized_shift: burn,
            rank_normalized_max_r_hat: rhat,
            effective_sample_size: ess,
            ensemble_manifest_digest: "ensemble-sha256".into(),
            burn_in_evidence_id: format!("burn-{id}"),
            convergence_evidence_id: format!("conv-{id}"),
        }
    }

    fn topology() -> TopologyPromotionEvidence {
        TopologyPromotionEvidence {
            observable_id: "topological-q".into(),
            retained_sample_count: 24,
            independent_chain_count: 2,
            zero_variance_observed: false,
            tau_window_complete: true,
            effective_sample_size: Some(80.0),
            sector_change_count: 12,
            ensemble_manifest_digest: "ensemble-sha256".into(),
            topology_evidence_id: "topology-evidence".into(),
        }
    }

    fn policy() -> EnsembleScalePromotionPolicy {
        EnsembleScalePromotionPolicy {
            maximum_burn_in_normalized_shift: 1.0,
            maximum_rank_normalized_r_hat: 1.05,
            minimum_effective_sample_size: 50.0,
            minimum_topology_effective_sample_size: 40.0,
            minimum_topology_sector_changes: 4,
            required_observable_ids: vec!["plaquette".into(), "flow-energy".into()],
        }
    }

    #[test]
    fn declared_policy_can_pass_without_becoming_an_equilibrium_theorem() {
        let evidence = [
            observable("plaquette", 0.4, 1.01, 120.0),
            observable("flow-energy", 0.6, 1.03, 90.0),
        ];
        let assessment = assess_ensemble_scale_promotion(
            &scale(), &stability(true), "block-stability-evidence", &evidence, &topology(), &policy(),
        ).unwrap();
        assert!(assessment.meets_declared_policy);
        assert!(assessment.selected_block_is_on_plateau);
        assert_eq!(assessment.worst_burn_in_observable_id, "flow-energy");
    }

    #[test]
    fn optional_slow_observable_can_only_make_assessment_stricter() {
        let evidence = [
            observable("plaquette", 0.4, 1.01, 120.0),
            observable("flow-energy", 0.6, 1.03, 90.0),
            observable("wilson-loop", 0.3, 1.08, 35.0),
        ];
        let assessment = assess_ensemble_scale_promotion(
            &scale(), &stability(true), "block-stability-evidence", &evidence, &topology(), &policy(),
        ).unwrap();
        assert!(!assessment.multi_chain_r_hat_satisfied);
        assert!(!assessment.effective_sample_size_satisfied);
        assert!(!assessment.meets_declared_policy);
    }

    #[test]
    fn frozen_topology_blocks_promotion_even_when_local_observables_look_good() {
        let evidence = [
            observable("plaquette", 0.2, 1.01, 150.0),
            observable("flow-energy", 0.3, 1.02, 120.0),
        ];
        let mut topo = topology();
        topo.zero_variance_observed = true;
        topo.effective_sample_size = None;
        topo.sector_change_count = 0;
        let assessment = assess_ensemble_scale_promotion(
            &scale(), &stability(true), "block-stability-evidence", &evidence, &topo, &policy(),
        ).unwrap();
        assert!(!assessment.topology_nonzero_variance);
        assert!(!assessment.topology_effective_sample_size_satisfied);
        assert!(!assessment.topology_sector_changes_satisfied);
        assert!(!assessment.meets_declared_policy);
    }

    #[test]
    fn selected_scale_block_must_belong_to_stability_plateau() {
        let evidence = [
            observable("plaquette", 0.2, 1.01, 150.0),
            observable("flow-energy", 0.3, 1.02, 120.0),
        ];
        let mut block = stability(true);
        block.plateau_block_sizes = vec![4, 8, 12];
        let assessment = assess_ensemble_scale_promotion(
            &scale(), &block, "block-stability-evidence", &evidence, &topology(), &policy(),
        ).unwrap();
        assert!(!assessment.selected_block_is_on_plateau);
        assert!(!assessment.block_stability_satisfied);
        assert!(!assessment.meets_declared_policy);
    }

    #[test]
    fn population_mismatch_fails_closed() {
        let mut evidence = vec![
            observable("plaquette", 0.2, 1.01, 150.0),
            observable("flow-energy", 0.3, 1.02, 120.0),
        ];
        evidence[0].retained_sample_count = 23;
        assert!(matches!(
            assess_ensemble_scale_promotion(
                &scale(), &stability(true), "block-stability-evidence", &evidence, &topology(), &policy(),
            ),
            Err(EnsembleScalePromotionError::ObservablePopulationMismatch { index: 0 })
        ));
    }
}
