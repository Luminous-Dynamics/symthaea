// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact-head qualification receipt for a lattice scale analysis candidate.
//!
//! Passing a preregistered diagnostic policy is still not the same thing as
//! having an executable-qualified analysis subject. This module binds the
//! predeclared-policy result to exact-head CI evidence and immutable analysis
//! artifacts before producing a narrowly scoped analysis-candidate receipt.
//!
//! The resulting receipt is **not** a physical-QCD result and does not establish
//! finite-volume, continuum, experimental or phenomenological validity.

use crate::lattice_flow_joint_evidence::JointJackknifeScaleEstimateEvidence;
use crate::lattice_qualification_preregistration::{
    QUALIFICATION_POLICY_MANIFEST_ID, PredeclaredQualificationBinding,
};

pub const QUALIFIED_SCALE_ANALYSIS_CANDIDATE_SCOPE: &str =
    "qualified_scale_analysis_candidate_only_v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactHeadCiConclusion {
    Passed,
    Failed,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExactHeadCiEvidence {
    pub subject_revision: String,
    pub source_tree_digest: String,
    pub ci_run_id: u64,
    pub ci_receipt_digest: String,
    pub toolchain_digest: String,
    pub build_profile: String,
    pub conclusion: ExactHeadCiConclusion,
    pub qualification_timestamp_unix_ns: u128,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScaleAnalysisArtifactLineage {
    pub retained_trajectory_artifact_digest: String,
    pub flow_curve_artifact_digest: String,
    pub resampling_artifact_digest: String,
    pub block_adequacy_artifact_digest: String,
    pub block_stability_artifact_digest: String,
    pub promotion_artifact_digest: String,
    pub preregistration_binding_artifact_digest: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QualifiedScaleAnalysisCandidate {
    pub authority_scope: &'static str,
    pub subject_revision: String,
    pub source_tree_digest: String,
    pub ensemble_manifest_digest: String,
    pub configuration_digest: String,
    pub selected_block_size: usize,
    pub frozen_policy_artifact_digest: String,
    pub ci_run_id: u64,
    pub ci_receipt_digest: String,
    pub qualification_timestamp_unix_ns: u128,
    pub artifact_lineage: ScaleAnalysisArtifactLineage,
    pub evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AnalysisAuthorityError {
    WrongPreregistrationManifest,
    PolicyNotPredeclared,
    PredeclaredPolicyNotSatisfied,
    EmptyField(&'static str),
    InvalidCiRunId,
    CiDidNotPass(ExactHeadCiConclusion),
    SubjectRevisionMismatch,
    QualificationNotAfterAnalysis,
    EnsembleManifestMismatch,
    SelectedBlockMismatch,
    ResamplingArtifactMismatch,
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), AnalysisAuthorityError> {
    if value.trim().is_empty() {
        Err(AnalysisAuthorityError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn validate_ci(ci: &ExactHeadCiEvidence) -> Result<(), AnalysisAuthorityError> {
    require_nonempty(&ci.subject_revision, "subject_revision")?;
    require_nonempty(&ci.source_tree_digest, "source_tree_digest")?;
    require_nonempty(&ci.ci_receipt_digest, "ci_receipt_digest")?;
    require_nonempty(&ci.toolchain_digest, "toolchain_digest")?;
    require_nonempty(&ci.build_profile, "build_profile")?;
    if ci.ci_run_id == 0 {
        return Err(AnalysisAuthorityError::InvalidCiRunId);
    }
    if ci.conclusion != ExactHeadCiConclusion::Passed {
        return Err(AnalysisAuthorityError::CiDidNotPass(ci.conclusion));
    }
    Ok(())
}

fn validate_artifacts(lineage: &ScaleAnalysisArtifactLineage) -> Result<(), AnalysisAuthorityError> {
    require_nonempty(
        &lineage.retained_trajectory_artifact_digest,
        "retained_trajectory_artifact_digest",
    )?;
    require_nonempty(&lineage.flow_curve_artifact_digest, "flow_curve_artifact_digest")?;
    require_nonempty(&lineage.resampling_artifact_digest, "resampling_artifact_digest")?;
    require_nonempty(
        &lineage.block_adequacy_artifact_digest,
        "block_adequacy_artifact_digest",
    )?;
    require_nonempty(
        &lineage.block_stability_artifact_digest,
        "block_stability_artifact_digest",
    )?;
    require_nonempty(&lineage.promotion_artifact_digest, "promotion_artifact_digest")?;
    require_nonempty(
        &lineage.preregistration_binding_artifact_digest,
        "preregistration_binding_artifact_digest",
    )?;
    Ok(())
}

/// Bind a preregistered, policy-satisfying scale analysis to exact-head CI and
/// immutable artifacts. Construction proves lineage consistency only; it does
/// not elevate the result to a physical lattice-QCD claim.
pub fn bind_qualified_scale_analysis_candidate(
    preregistration: &PredeclaredQualificationBinding,
    scale: &JointJackknifeScaleEstimateEvidence,
    ci: &ExactHeadCiEvidence,
    artifacts: &ScaleAnalysisArtifactLineage,
    ci_evidence_id: &str,
) -> Result<QualifiedScaleAnalysisCandidate, AnalysisAuthorityError> {
    if preregistration.manifest_id != QUALIFICATION_POLICY_MANIFEST_ID {
        return Err(AnalysisAuthorityError::WrongPreregistrationManifest);
    }
    if !preregistration.predeclared_before_analysis {
        return Err(AnalysisAuthorityError::PolicyNotPredeclared);
    }
    if !preregistration.meets_predeclared_policy {
        return Err(AnalysisAuthorityError::PredeclaredPolicyNotSatisfied);
    }
    require_nonempty(ci_evidence_id, "ci_evidence_id")?;
    validate_ci(ci)?;
    validate_artifacts(artifacts)?;

    if ci.subject_revision != preregistration.code_revision {
        return Err(AnalysisAuthorityError::SubjectRevisionMismatch);
    }
    if ci.qualification_timestamp_unix_ns <= preregistration.analysis_started_unix_ns {
        return Err(AnalysisAuthorityError::QualificationNotAfterAnalysis);
    }
    if scale.scale.ensemble_manifest_digest != preregistration.ensemble_manifest_digest {
        return Err(AnalysisAuthorityError::EnsembleManifestMismatch);
    }
    if scale.block_size != preregistration.selected_block_size {
        return Err(AnalysisAuthorityError::SelectedBlockMismatch);
    }
    if artifacts.resampling_artifact_digest != scale.resampling_artifact_digest {
        return Err(AnalysisAuthorityError::ResamplingArtifactMismatch);
    }

    let mut evidence_ids = preregistration.evidence_ids.clone();
    evidence_ids.push(ci_evidence_id.to_string());
    evidence_ids.push(preregistration.freeze_evidence_id.clone());
    evidence_ids.sort();
    evidence_ids.dedup();

    Ok(QualifiedScaleAnalysisCandidate {
        authority_scope: QUALIFIED_SCALE_ANALYSIS_CANDIDATE_SCOPE,
        subject_revision: ci.subject_revision.clone(),
        source_tree_digest: ci.source_tree_digest.clone(),
        ensemble_manifest_digest: preregistration.ensemble_manifest_digest.clone(),
        configuration_digest: preregistration.configuration_digest.clone(),
        selected_block_size: preregistration.selected_block_size,
        frozen_policy_artifact_digest: preregistration.frozen_policy_artifact_digest.clone(),
        ci_run_id: ci.ci_run_id,
        ci_receipt_digest: ci.ci_receipt_digest.clone(),
        qualification_timestamp_unix_ns: ci.qualification_timestamp_unix_ns,
        artifact_lineage: artifacts.clone(),
        evidence_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_flow_scale_evidence::{FlowScaleEstimateEvidence, FlowScaleKind};
    use crate::lattice_flow_joint_jackknife::JOINT_BLOCKED_JACKKNIFE_ID;

    fn preregistration() -> PredeclaredQualificationBinding {
        PredeclaredQualificationBinding {
            manifest_id: QUALIFICATION_POLICY_MANIFEST_ID,
            canonical_policy_material: "frozen-policy".into(),
            frozen_policy_artifact_digest: "policy-sha256".into(),
            freeze_evidence_id: "freeze-evidence".into(),
            freeze_timestamp_unix_ns: 1_000,
            analysis_started_unix_ns: 2_000,
            predeclared_before_analysis: true,
            code_revision: "exact-head-sha".into(),
            configuration_digest: "config-sha256".into(),
            ensemble_manifest_digest: "ensemble-sha256".into(),
            selected_block_size: 6,
            block_adequacy_policy_satisfied: true,
            block_stability_policy_satisfied: true,
            promotion_policy_satisfied: true,
            meets_predeclared_policy: true,
            evidence_ids: vec!["promotion-evidence".into()],
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
            resampling_artifact_digest: "resampling-sha256".into(),
        }
    }

    fn ci() -> ExactHeadCiEvidence {
        ExactHeadCiEvidence {
            subject_revision: "exact-head-sha".into(),
            source_tree_digest: "tree-sha256".into(),
            ci_run_id: 42,
            ci_receipt_digest: "ci-receipt-sha256".into(),
            toolchain_digest: "toolchain-sha256".into(),
            build_profile: "rust-1.96-release".into(),
            conclusion: ExactHeadCiConclusion::Passed,
            qualification_timestamp_unix_ns: 3_000,
        }
    }

    fn artifacts() -> ScaleAnalysisArtifactLineage {
        ScaleAnalysisArtifactLineage {
            retained_trajectory_artifact_digest: "trajectory-sha256".into(),
            flow_curve_artifact_digest: "curve-sha256".into(),
            resampling_artifact_digest: "resampling-sha256".into(),
            block_adequacy_artifact_digest: "adequacy-sha256".into(),
            block_stability_artifact_digest: "stability-sha256".into(),
            promotion_artifact_digest: "promotion-sha256".into(),
            preregistration_binding_artifact_digest: "prereg-sha256".into(),
        }
    }

    #[test]
    fn exact_head_passed_ci_yields_analysis_candidate_scope_only() {
        let receipt = bind_qualified_scale_analysis_candidate(
            &preregistration(), &scale(), &ci(), &artifacts(), "ci-evidence",
        ).unwrap();
        assert_eq!(receipt.authority_scope, QUALIFIED_SCALE_ANALYSIS_CANDIDATE_SCOPE);
        assert_eq!(receipt.subject_revision, "exact-head-sha");
    }

    #[test]
    fn failed_ci_cannot_be_promoted() {
        let mut ci = ci();
        ci.conclusion = ExactHeadCiConclusion::Failed;
        assert!(matches!(
            bind_qualified_scale_analysis_candidate(
                &preregistration(), &scale(), &ci, &artifacts(), "ci-evidence",
            ),
            Err(AnalysisAuthorityError::CiDidNotPass(ExactHeadCiConclusion::Failed))
        ));
    }

    #[test]
    fn wrong_exact_head_cannot_reuse_another_ci_receipt() {
        let mut ci = ci();
        ci.subject_revision = "different-head".into();
        assert!(matches!(
            bind_qualified_scale_analysis_candidate(
                &preregistration(), &scale(), &ci, &artifacts(), "ci-evidence",
            ),
            Err(AnalysisAuthorityError::SubjectRevisionMismatch)
        ));
    }

    #[test]
    fn ci_must_postdate_analysis_subject() {
        let mut ci = ci();
        ci.qualification_timestamp_unix_ns = 2_000;
        assert!(matches!(
            bind_qualified_scale_analysis_candidate(
                &preregistration(), &scale(), &ci, &artifacts(), "ci-evidence",
            ),
            Err(AnalysisAuthorityError::QualificationNotAfterAnalysis)
        ));
    }

    #[test]
    fn artifact_lineage_must_match_bound_resampling() {
        let mut artifacts = artifacts();
        artifacts.resampling_artifact_digest = "other-resampling".into();
        assert!(matches!(
            bind_qualified_scale_analysis_candidate(
                &preregistration(), &scale(), &ci(), &artifacts, "ci-evidence",
            ),
            Err(AnalysisAuthorityError::ResamplingArtifactMismatch)
        ));
    }
}
