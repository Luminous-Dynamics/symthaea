// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Predeclared qualification-policy manifest for lattice scale evidence.
//!
//! Explicit thresholds are not enough if they can be chosen after looking at the
//! result. This module freezes the complete qualification policy before analysis
//! and later verifies that block adequacy, block stability and ensemble-promotion
//! assessments were interpreted under that exact policy.
//!
//! A positive binding remains a policy-compliance statement, not proof of
//! equilibrium, ergodicity, topology completeness or physical scale setting.

use std::collections::BTreeSet;

use crate::lattice_ensemble_promotion::{
    ENSEMBLE_SCALE_PROMOTION_ASSESSMENT_ID, EnsembleScalePromotionAssessment,
    EnsembleScalePromotionPolicy,
};
use crate::lattice_flow_block_adequacy::{
    FLOW_BLOCK_ADEQUACY_POLICY_ID, BlockAdequacyAssessment, BlockAdequacyPolicy,
};
use crate::lattice_flow_joint_evidence::JointJackknifeScaleEstimateEvidence;
use crate::lattice_flow_scale_evidence::FlowScaleKind;
use crate::lattice_flow_scale_stability::{
    FLOW_SCALE_BLOCK_STABILITY_ID, FlowScaleBlockStabilityAssessment,
    FlowScaleBlockStabilityPolicy,
};

pub const QUALIFICATION_POLICY_MANIFEST_ID: &str = "lqcd_qualification_policy_manifest_v1";

#[derive(Debug, Clone, PartialEq)]
pub struct EnsembleQualificationPolicyManifest {
    pub scale_kind: FlowScaleKind,
    pub scale_target: f64,
    pub block_adequacy_policy: BlockAdequacyPolicy,
    pub block_stability_policy: FlowScaleBlockStabilityPolicy,
    pub promotion_policy: EnsembleScalePromotionPolicy,
    pub code_revision: String,
    pub configuration_digest: String,
    pub freeze_evidence_id: String,
    pub freeze_timestamp_unix_ns: u128,
    pub frozen_policy_artifact_digest: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PredeclaredQualificationBinding {
    pub manifest_id: &'static str,
    pub canonical_policy_material: String,
    pub frozen_policy_artifact_digest: String,
    pub freeze_evidence_id: String,
    pub freeze_timestamp_unix_ns: u128,
    pub analysis_started_unix_ns: u128,
    pub predeclared_before_analysis: bool,
    pub code_revision: String,
    pub configuration_digest: String,
    pub ensemble_manifest_digest: String,
    pub selected_block_size: usize,
    pub block_adequacy_policy_satisfied: bool,
    pub block_stability_policy_satisfied: bool,
    pub promotion_policy_satisfied: bool,
    pub meets_predeclared_policy: bool,
    pub evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum QualificationPreregistrationError {
    InvalidScaleTarget(f64),
    InvalidBlockAdequacyPolicy,
    InvalidBlockStabilityPolicy,
    InvalidPromotionPolicy,
    EmptyPolicyObservableSet { policy: &'static str },
    EmptyPolicyObservableId { policy: &'static str, index: usize },
    DuplicatePolicyObservableId { policy: &'static str, observable_id: String },
    EmptyField(&'static str),
    InvalidFreezeTimestamp,
    AnalysisNotAfterFreeze,
    ScaleKindMismatch,
    ScaleTargetMismatch,
    EnsembleManifestMismatch,
    SelectedBlockGeometryMismatch,
    BlockAdequacyPolicyMismatch,
    BlockStabilityPolicyMismatch,
    PromotionPolicyMismatch,
    PromotionAssessmentPolicyMismatch,
    EvidenceLineageMismatch,
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), QualificationPreregistrationError> {
    if value.trim().is_empty() {
        Err(QualificationPreregistrationError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn sorted_policy_ids(
    ids: &[String],
    policy: &'static str,
) -> Result<Vec<String>, QualificationPreregistrationError> {
    if ids.is_empty() {
        return Err(QualificationPreregistrationError::EmptyPolicyObservableSet { policy });
    }
    let mut seen = BTreeSet::new();
    for (index, id) in ids.iter().enumerate() {
        let id = id.trim();
        if id.is_empty() {
            return Err(QualificationPreregistrationError::EmptyPolicyObservableId { policy, index });
        }
        if !seen.insert(id) {
            return Err(QualificationPreregistrationError::DuplicatePolicyObservableId {
                policy,
                observable_id: id.to_string(),
            });
        }
    }
    Ok(seen.into_iter().map(str::to_string).collect())
}

fn encode_strings(ids: &[String]) -> String {
    let mut ids = ids.to_vec();
    ids.sort();
    ids.into_iter()
        .map(|value| format!("{}:{}", value.len(), value))
        .collect::<Vec<_>>()
        .join("|")
}

fn scale_kind_id(kind: FlowScaleKind) -> &'static str {
    match kind {
        FlowScaleKind::T0Like => "t0_like",
        FlowScaleKind::W0Like => "w0_like",
    }
}

impl EnsembleQualificationPolicyManifest {
    pub fn validate(&self) -> Result<(), QualificationPreregistrationError> {
        if !self.scale_target.is_finite() || self.scale_target <= 0.0 {
            return Err(QualificationPreregistrationError::InvalidScaleTarget(self.scale_target));
        }
        if !self.block_adequacy_policy.minimum_tau_multiple.is_finite()
            || self.block_adequacy_policy.minimum_tau_multiple <= 0.0
            || self.block_adequacy_policy.minimum_block_count < 2
        {
            return Err(QualificationPreregistrationError::InvalidBlockAdequacyPolicy);
        }
        sorted_policy_ids(&self.block_adequacy_policy.required_observable_ids, "block_adequacy")?;

        if self.block_stability_policy.plateau_point_count < 2
            || !self.block_stability_policy.maximum_relative_standard_error_change.is_finite()
            || self.block_stability_policy.maximum_relative_standard_error_change < 0.0
            || !self.block_stability_policy.maximum_relative_central_estimate_change.is_finite()
            || self.block_stability_policy.maximum_relative_central_estimate_change < 0.0
        {
            return Err(QualificationPreregistrationError::InvalidBlockStabilityPolicy);
        }

        if !self.promotion_policy.maximum_burn_in_normalized_shift.is_finite()
            || self.promotion_policy.maximum_burn_in_normalized_shift < 0.0
            || !self.promotion_policy.maximum_rank_normalized_r_hat.is_finite()
            || self.promotion_policy.maximum_rank_normalized_r_hat <= 0.0
            || !self.promotion_policy.minimum_effective_sample_size.is_finite()
            || self.promotion_policy.minimum_effective_sample_size <= 0.0
            || !self.promotion_policy.minimum_topology_effective_sample_size.is_finite()
            || self.promotion_policy.minimum_topology_effective_sample_size <= 0.0
        {
            return Err(QualificationPreregistrationError::InvalidPromotionPolicy);
        }
        sorted_policy_ids(&self.promotion_policy.required_observable_ids, "promotion")?;

        require_nonempty(&self.code_revision, "code_revision")?;
        require_nonempty(&self.configuration_digest, "configuration_digest")?;
        require_nonempty(&self.freeze_evidence_id, "freeze_evidence_id")?;
        require_nonempty(&self.frozen_policy_artifact_digest, "frozen_policy_artifact_digest")?;
        if self.freeze_timestamp_unix_ns == 0 {
            return Err(QualificationPreregistrationError::InvalidFreezeTimestamp);
        }
        Ok(())
    }

    /// Canonical, hashable manifest material. Floating-point policy values use
    /// IEEE-754 bit patterns rather than locale/formatter-dependent decimal text.
    /// Observable sets are sorted and length-prefixed.
    pub fn canonical_material(&self) -> Result<String, QualificationPreregistrationError> {
        self.validate()?;
        Ok([
            format!("schema={QUALIFICATION_POLICY_MANIFEST_ID}"),
            format!("scale_kind={}", scale_kind_id(self.scale_kind)),
            format!("scale_target_bits={:016x}", self.scale_target.to_bits()),
            format!(
                "block_tau_multiple_bits={:016x}",
                self.block_adequacy_policy.minimum_tau_multiple.to_bits()
            ),
            format!("block_min_count={}", self.block_adequacy_policy.minimum_block_count),
            format!(
                "block_required={}",
                encode_strings(&self.block_adequacy_policy.required_observable_ids)
            ),
            format!("plateau_points={}", self.block_stability_policy.plateau_point_count),
            format!(
                "plateau_se_bits={:016x}",
                self.block_stability_policy.maximum_relative_standard_error_change.to_bits()
            ),
            format!(
                "plateau_central_bits={:016x}",
                self.block_stability_policy.maximum_relative_central_estimate_change.to_bits()
            ),
            format!(
                "burn_shift_bits={:016x}",
                self.promotion_policy.maximum_burn_in_normalized_shift.to_bits()
            ),
            format!(
                "rhat_bits={:016x}",
                self.promotion_policy.maximum_rank_normalized_r_hat.to_bits()
            ),
            format!(
                "ess_bits={:016x}",
                self.promotion_policy.minimum_effective_sample_size.to_bits()
            ),
            format!(
                "topology_ess_bits={:016x}",
                self.promotion_policy.minimum_topology_effective_sample_size.to_bits()
            ),
            format!(
                "topology_sector_changes={}",
                self.promotion_policy.minimum_topology_sector_changes
            ),
            format!(
                "promotion_required={}",
                encode_strings(&self.promotion_policy.required_observable_ids)
            ),
            format!("code_revision={}:{}", self.code_revision.len(), self.code_revision),
            format!(
                "configuration_digest={}:{}",
                self.configuration_digest.len(), self.configuration_digest
            ),
            format!(
                "freeze_evidence={}:{}",
                self.freeze_evidence_id.len(), self.freeze_evidence_id
            ),
            format!("freeze_timestamp_unix_ns={}", self.freeze_timestamp_unix_ns),
        ]
        .join("\n"))
    }
}

fn same_id_set(left: &[String], right: &[String]) -> bool {
    let mut left = left.iter().map(|value| value.trim()).collect::<Vec<_>>();
    let mut right = right.iter().map(|value| value.trim()).collect::<Vec<_>>();
    left.sort_unstable();
    right.sort_unstable();
    left == right
}

fn promotion_matches_manifest(
    assessment: &EnsembleScalePromotionAssessment,
    policy: &EnsembleScalePromotionPolicy,
) -> bool {
    let burn = assessment.maximum_observed_burn_in_normalized_shift
        <= policy.maximum_burn_in_normalized_shift;
    let rhat = assessment.maximum_observed_rank_normalized_r_hat
        <= policy.maximum_rank_normalized_r_hat;
    let ess = assessment.minimum_observed_effective_sample_size
        >= policy.minimum_effective_sample_size;
    let topology_ess = assessment
        .topology_effective_sample_size
        .is_some_and(|value| value >= policy.minimum_topology_effective_sample_size);
    let topology_changes = assessment.topology_sector_change_count
        >= policy.minimum_topology_sector_changes;
    let overall = assessment.block_stability_satisfied
        && burn
        && rhat
        && ess
        && assessment.topology_nonzero_variance
        && assessment.topology_tau_window_complete
        && topology_ess
        && topology_changes;
    assessment.burn_in_sensitivity_satisfied == burn
        && assessment.multi_chain_r_hat_satisfied == rhat
        && assessment.effective_sample_size_satisfied == ess
        && assessment.topology_effective_sample_size_satisfied == topology_ess
        && assessment.topology_sector_changes_satisfied == topology_changes
        && assessment.meets_declared_policy == overall
        && same_id_set(&assessment.required_observable_ids, &policy.required_observable_ids)
}

pub fn bind_predeclared_qualification_policy(
    manifest: &EnsembleQualificationPolicyManifest,
    scale: &JointJackknifeScaleEstimateEvidence,
    block_adequacy: &BlockAdequacyAssessment,
    block_stability: &FlowScaleBlockStabilityAssessment,
    promotion: &EnsembleScalePromotionAssessment,
    analysis_started_unix_ns: u128,
    block_adequacy_evidence_id: &str,
    block_stability_evidence_id: &str,
    promotion_evidence_id: &str,
) -> Result<PredeclaredQualificationBinding, QualificationPreregistrationError> {
    manifest.validate()?;
    if analysis_started_unix_ns <= manifest.freeze_timestamp_unix_ns {
        return Err(QualificationPreregistrationError::AnalysisNotAfterFreeze);
    }
    require_nonempty(block_adequacy_evidence_id, "block_adequacy_evidence_id")?;
    require_nonempty(block_stability_evidence_id, "block_stability_evidence_id")?;
    require_nonempty(promotion_evidence_id, "promotion_evidence_id")?;

    if scale.scale.kind != manifest.scale_kind {
        return Err(QualificationPreregistrationError::ScaleKindMismatch);
    }
    if scale.scale.target.to_bits() != manifest.scale_target.to_bits() {
        return Err(QualificationPreregistrationError::ScaleTargetMismatch);
    }
    if promotion.ensemble_manifest_digest != scale.scale.ensemble_manifest_digest {
        return Err(QualificationPreregistrationError::EnsembleManifestMismatch);
    }
    if block_adequacy.block_size != scale.block_size
        || block_adequacy.block_count != scale.block_count
        || block_adequacy.configuration_count != scale.configuration_count
        || promotion.selected_block_size != scale.block_size
        || promotion.configuration_count != scale.configuration_count
        || promotion.independent_chain_count != scale.independent_chain_count
    {
        return Err(QualificationPreregistrationError::SelectedBlockGeometryMismatch);
    }

    let adequacy_policy_matches = block_adequacy.policy_id == FLOW_BLOCK_ADEQUACY_POLICY_ID
        && block_adequacy.minimum_tau_multiple.to_bits()
            == manifest.block_adequacy_policy.minimum_tau_multiple.to_bits()
        && block_adequacy.minimum_block_count == manifest.block_adequacy_policy.minimum_block_count
        && same_id_set(
            &block_adequacy.required_observable_ids,
            &manifest.block_adequacy_policy.required_observable_ids,
        );
    if !adequacy_policy_matches {
        return Err(QualificationPreregistrationError::BlockAdequacyPolicyMismatch);
    }

    let stability_policy_matches = block_stability.assessment_id == FLOW_SCALE_BLOCK_STABILITY_ID
        && block_stability.plateau_point_count == manifest.block_stability_policy.plateau_point_count
        && block_stability.maximum_relative_standard_error_change.to_bits()
            == manifest
                .block_stability_policy
                .maximum_relative_standard_error_change
                .to_bits()
        && block_stability.maximum_relative_central_estimate_change.to_bits()
            == manifest
                .block_stability_policy
                .maximum_relative_central_estimate_change
                .to_bits();
    if !stability_policy_matches {
        return Err(QualificationPreregistrationError::BlockStabilityPolicyMismatch);
    }

    if promotion.assessment_id != ENSEMBLE_SCALE_PROMOTION_ASSESSMENT_ID
        || !same_id_set(
            &promotion.required_observable_ids,
            &manifest.promotion_policy.required_observable_ids,
        )
    {
        return Err(QualificationPreregistrationError::PromotionPolicyMismatch);
    }
    if !promotion_matches_manifest(promotion, &manifest.promotion_policy) {
        return Err(QualificationPreregistrationError::PromotionAssessmentPolicyMismatch);
    }
    if promotion.block_stability_evidence_id != block_stability_evidence_id {
        return Err(QualificationPreregistrationError::EvidenceLineageMismatch);
    }

    let mut evidence_ids = vec![
        manifest.freeze_evidence_id.clone(),
        block_adequacy_evidence_id.to_string(),
        block_stability_evidence_id.to_string(),
        promotion_evidence_id.to_string(),
        scale.scale.joint_resampling_evidence_id.clone(),
        scale.resampling_artifact_digest.clone(),
    ];
    evidence_ids.extend(promotion.evidence_ids.iter().cloned());
    evidence_ids.sort();
    evidence_ids.dedup();

    let block_adequacy_policy_satisfied = block_adequacy.meets_declared_policy;
    let block_stability_policy_satisfied = block_stability.meets_declared_policy;
    let promotion_policy_satisfied = promotion.meets_declared_policy;
    let meets_predeclared_policy = block_adequacy_policy_satisfied
        && block_stability_policy_satisfied
        && promotion_policy_satisfied;

    Ok(PredeclaredQualificationBinding {
        manifest_id: QUALIFICATION_POLICY_MANIFEST_ID,
        canonical_policy_material: manifest.canonical_material()?,
        frozen_policy_artifact_digest: manifest.frozen_policy_artifact_digest.clone(),
        freeze_evidence_id: manifest.freeze_evidence_id.clone(),
        freeze_timestamp_unix_ns: manifest.freeze_timestamp_unix_ns,
        analysis_started_unix_ns,
        predeclared_before_analysis: true,
        code_revision: manifest.code_revision.clone(),
        configuration_digest: manifest.configuration_digest.clone(),
        ensemble_manifest_digest: scale.scale.ensemble_manifest_digest.clone(),
        selected_block_size: scale.block_size,
        block_adequacy_policy_satisfied,
        block_stability_policy_satisfied,
        promotion_policy_satisfied,
        meets_predeclared_policy,
        evidence_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_ensemble_promotion::ENSEMBLE_SCALE_PROMOTION_ASSESSMENT_ID;
    use crate::lattice_flow_joint_jackknife::JOINT_BLOCKED_JACKKNIFE_ID;
    use crate::lattice_flow_scale_evidence::FlowScaleEstimateEvidence;

    fn manifest() -> EnsembleQualificationPolicyManifest {
        EnsembleQualificationPolicyManifest {
            scale_kind: FlowScaleKind::T0Like,
            scale_target: 0.30,
            block_adequacy_policy: BlockAdequacyPolicy {
                minimum_tau_multiple: 2.0,
                minimum_block_count: 4,
                required_observable_ids: vec!["topological-q".into(), "flow-energy".into()],
            },
            block_stability_policy: FlowScaleBlockStabilityPolicy {
                plateau_point_count: 3,
                maximum_relative_standard_error_change: 0.20,
                maximum_relative_central_estimate_change: 0.01,
            },
            promotion_policy: EnsembleScalePromotionPolicy {
                maximum_burn_in_normalized_shift: 1.0,
                maximum_rank_normalized_r_hat: 1.05,
                minimum_effective_sample_size: 50.0,
                minimum_topology_effective_sample_size: 40.0,
                minimum_topology_sector_changes: 4,
                required_observable_ids: vec!["flow-energy".into(), "plaquette".into()],
            },
            code_revision: "exact-head-sha".into(),
            configuration_digest: "config-sha256".into(),
            freeze_evidence_id: "freeze-receipt".into(),
            freeze_timestamp_unix_ns: 1000,
            frozen_policy_artifact_digest: "policy-sha256".into(),
        }
    }

    fn scale() -> JointJackknifeScaleEstimateEvidence {
        JointJackknifeScaleEstimateEvidence {
            scale: FlowScaleEstimateEvidence {
                kind: FlowScaleKind::T0Like,
                target: 0.30,
                estimate: 0.35,
                standard_error: 0.013_693_063_937_629_136,
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

    fn adequacy() -> BlockAdequacyAssessment {
        BlockAdequacyAssessment {
            policy_id: FLOW_BLOCK_ADEQUACY_POLICY_ID,
            minimum_tau_multiple: 2.0,
            minimum_block_count: 4,
            required_observable_ids: vec!["flow-energy".into(), "topological-q".into()],
            supplied_observable_ids: vec!["flow-energy".into(), "topological-q".into()],
            configuration_count: 24,
            block_size: 6,
            block_count: 4,
            slowest_observable_id: "topological-q".into(),
            slowest_tau_int: 2.2,
            required_block_size: 5,
            tau_window_complete: true,
            meets_tau_multiple: true,
            meets_minimum_block_count: true,
            meets_declared_policy: true,
            statistics_evidence_ids: vec!["stats-flow".into(), "stats-topology".into()],
        }
    }

    fn stability() -> FlowScaleBlockStabilityAssessment {
        FlowScaleBlockStabilityAssessment {
            assessment_id: FLOW_SCALE_BLOCK_STABILITY_ID,
            plateau_point_count: 3,
            maximum_relative_standard_error_change: 0.20,
            maximum_relative_central_estimate_change: 0.01,
            admissible_block_sizes: vec![4, 6, 8],
            plateau_block_sizes: vec![4, 6, 8],
            maximum_observed_relative_standard_error_change: Some(0.10),
            maximum_observed_relative_central_estimate_change: Some(0.0),
            enough_admissible_points: true,
            meets_uncertainty_plateau: true,
            meets_central_estimate_stability: true,
            meets_declared_policy: true,
        }
    }

    fn promotion() -> EnsembleScalePromotionAssessment {
        EnsembleScalePromotionAssessment {
            assessment_id: ENSEMBLE_SCALE_PROMOTION_ASSESSMENT_ID,
            ensemble_manifest_digest: "ensemble-sha256".into(),
            configuration_count: 24,
            independent_chain_count: 2,
            selected_block_size: 6,
            block_stability_evidence_id: "stability-evidence".into(),
            required_observable_ids: vec!["plaquette".into(), "flow-energy".into()],
            supplied_observable_ids: vec!["plaquette".into(), "flow-energy".into()],
            worst_burn_in_observable_id: "flow-energy".into(),
            maximum_observed_burn_in_normalized_shift: 0.6,
            worst_r_hat_observable_id: "flow-energy".into(),
            maximum_observed_rank_normalized_r_hat: 1.03,
            worst_ess_observable_id: "flow-energy".into(),
            minimum_observed_effective_sample_size: 90.0,
            topology_observable_id: "topological-q".into(),
            topology_effective_sample_size: Some(80.0),
            topology_sector_change_count: 12,
            selected_block_is_on_plateau: true,
            block_stability_satisfied: true,
            burn_in_sensitivity_satisfied: true,
            multi_chain_r_hat_satisfied: true,
            effective_sample_size_satisfied: true,
            topology_nonzero_variance: true,
            topology_tau_window_complete: true,
            topology_effective_sample_size_satisfied: true,
            topology_sector_changes_satisfied: true,
            meets_declared_policy: true,
            evidence_ids: vec!["burn-flow".into(), "conv-flow".into(), "topology".into()],
        }
    }

    #[test]
    fn canonical_material_is_order_invariant_for_observable_sets() {
        let a = manifest();
        let mut b = a.clone();
        b.block_adequacy_policy.required_observable_ids.reverse();
        b.promotion_policy.required_observable_ids.reverse();
        assert_eq!(a.canonical_material().unwrap(), b.canonical_material().unwrap());
    }

    #[test]
    fn frozen_policy_binds_before_analysis() {
        let binding = bind_predeclared_qualification_policy(
            &manifest(), &scale(), &adequacy(), &stability(), &promotion(), 2000,
            "adequacy-evidence", "stability-evidence", "promotion-evidence",
        ).unwrap();
        assert!(binding.predeclared_before_analysis);
        assert!(binding.meets_predeclared_policy);
        assert_eq!(binding.selected_block_size, 6);
    }

    #[test]
    fn post_hoc_freeze_is_rejected() {
        assert!(matches!(
            bind_predeclared_qualification_policy(
                &manifest(), &scale(), &adequacy(), &stability(), &promotion(), 1000,
                "adequacy-evidence", "stability-evidence", "promotion-evidence",
            ),
            Err(QualificationPreregistrationError::AnalysisNotAfterFreeze)
        ));
    }

    #[test]
    fn changing_threshold_after_result_is_detected() {
        let mut changed = manifest();
        changed.promotion_policy.maximum_rank_normalized_r_hat = 1.02;
        assert!(matches!(
            bind_predeclared_qualification_policy(
                &changed, &scale(), &adequacy(), &stability(), &promotion(), 2000,
                "adequacy-evidence", "stability-evidence", "promotion-evidence",
            ),
            Err(QualificationPreregistrationError::PromotionAssessmentPolicyMismatch)
        ));
    }

    #[test]
    fn selected_block_policy_mismatch_is_detected() {
        let mut changed = manifest();
        changed.block_adequacy_policy.minimum_tau_multiple = 3.0;
        assert!(matches!(
            bind_predeclared_qualification_policy(
                &changed, &scale(), &adequacy(), &stability(), &promotion(), 2000,
                "adequacy-evidence", "stability-evidence", "promotion-evidence",
            ),
            Err(QualificationPreregistrationError::BlockAdequacyPolicyMismatch)
        ));
    }
}
