// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Selection-root execution identity for resource shadow campaigns.
//!
//! `selection root != realization root != execution root != scientific result root`.
//! V1 defines the selection root only: exact source/dependency/toolchain selections,
//! platform/CPU policy, build recipe, and process-environment policy. It cannot
//! claim a realized closure, an executed benchmark, or scientific qualification.

#![deny(unsafe_code)]

use blake3::Hasher;
use symthaea_resource_shadow_campaign::{CampaignManifest, CampaignManifestId};
use thiserror::Error;

pub const RESOURCE_CAMPAIGN_EXECUTION_PROFILE_V1: &str =
    "symthaea.resource-campaign-execution-profile.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExecutionSelectionProfileId([u8; 32]);
impl ExecutionSelectionProfileId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExecutionBoundCampaignId([u8; 32]);
impl ExecutionBoundCampaignId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionQualificationState {
    SelectionBoundOnly,
}

/// Opaque digests are supplied by the surrounding reproducibility system. This
/// type commits to those bytes but does not authenticate or realize their content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionSelectionProfile {
    source_revision: String,
    workspace_tree_commitment: [u8; 32],
    dependency_lock_commitment: [u8; 32],
    rust_toolchain_commitment: [u8; 32],
    nix_selection_commitment: [u8; 32],
    target_triple: String,
    cpu_feature_profile: String,
    build_recipe_commitment: [u8; 32],
    process_environment_commitment: [u8; 32],
    profile_id: ExecutionSelectionProfileId,
}

impl ExecutionSelectionProfile {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source_revision: impl Into<String>,
        workspace_tree_commitment: [u8; 32],
        dependency_lock_commitment: [u8; 32],
        rust_toolchain_commitment: [u8; 32],
        nix_selection_commitment: [u8; 32],
        target_triple: impl Into<String>,
        cpu_feature_profile: impl Into<String>,
        build_recipe_commitment: [u8; 32],
        process_environment_commitment: [u8; 32],
    ) -> Result<Self, ExecutionProfileError> {
        let source_revision = source_revision.into();
        let target_triple = target_triple.into();
        let cpu_feature_profile = cpu_feature_profile.into();
        if source_revision.trim().is_empty() { return Err(ExecutionProfileError::BlankSourceRevision); }
        if target_triple.trim().is_empty() { return Err(ExecutionProfileError::BlankTargetTriple); }
        if cpu_feature_profile.trim().is_empty() { return Err(ExecutionProfileError::BlankCpuFeatureProfile); }
        for (field, value) in [
            ("workspace_tree", workspace_tree_commitment),
            ("dependency_lock", dependency_lock_commitment),
            ("rust_toolchain", rust_toolchain_commitment),
            ("nix_selection", nix_selection_commitment),
            ("build_recipe", build_recipe_commitment),
            ("process_environment", process_environment_commitment),
        ] {
            if value == [0; 32] { return Err(ExecutionProfileError::ZeroCommitment { field }); }
        }
        let profile_id = ExecutionSelectionProfileId(hash_profile(
            &source_revision,
            &target_triple,
            &cpu_feature_profile,
            [
                workspace_tree_commitment,
                dependency_lock_commitment,
                rust_toolchain_commitment,
                nix_selection_commitment,
                build_recipe_commitment,
                process_environment_commitment,
            ],
        ));
        Ok(Self {
            source_revision,
            workspace_tree_commitment,
            dependency_lock_commitment,
            rust_toolchain_commitment,
            nix_selection_commitment,
            target_triple,
            cpu_feature_profile,
            build_recipe_commitment,
            process_environment_commitment,
            profile_id,
        })
    }

    pub fn source_revision(&self) -> &str { &self.source_revision }
    pub fn workspace_tree_commitment(&self) -> &[u8; 32] { &self.workspace_tree_commitment }
    pub fn dependency_lock_commitment(&self) -> &[u8; 32] { &self.dependency_lock_commitment }
    pub fn rust_toolchain_commitment(&self) -> &[u8; 32] { &self.rust_toolchain_commitment }
    pub fn nix_selection_commitment(&self) -> &[u8; 32] { &self.nix_selection_commitment }
    pub fn target_triple(&self) -> &str { &self.target_triple }
    pub fn cpu_feature_profile(&self) -> &str { &self.cpu_feature_profile }
    pub fn build_recipe_commitment(&self) -> &[u8; 32] { &self.build_recipe_commitment }
    pub fn process_environment_commitment(&self) -> &[u8; 32] { &self.process_environment_commitment }
    pub fn profile_id(&self) -> ExecutionSelectionProfileId { self.profile_id }
    pub fn qualification_state(&self) -> ExecutionQualificationState {
        ExecutionQualificationState::SelectionBoundOnly
    }
}

/// Campaign + execution selection. This prevents the same campaign manifest from
/// floating across source/toolchain/platform selections, but remains selection-only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionBoundCampaign {
    campaign: CampaignManifest,
    selection_profile: ExecutionSelectionProfile,
    binding_id: ExecutionBoundCampaignId,
}

impl ExecutionBoundCampaign {
    pub fn new(campaign: CampaignManifest, selection_profile: ExecutionSelectionProfile) -> Self {
        let binding_id = ExecutionBoundCampaignId(hash_binding(
            campaign.manifest_id(), selection_profile.profile_id(),
        ));
        Self { campaign, selection_profile, binding_id }
    }
    pub fn campaign(&self) -> &CampaignManifest { &self.campaign }
    pub fn selection_profile(&self) -> &ExecutionSelectionProfile { &self.selection_profile }
    pub fn binding_id(&self) -> ExecutionBoundCampaignId { self.binding_id }
    pub fn qualification_state(&self) -> ExecutionQualificationState {
        ExecutionQualificationState::SelectionBoundOnly
    }
}

/// A later realized receipt must satisfy all of these independently; this constant
/// is a contract declaration only and does not constitute evidence that any item
/// has been captured or verified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FutureRealizationRequirements {
    pub recursive_runtime_closure: bool,
    pub per_object_content_hash: bool,
    pub per_object_reference_set: bool,
    pub executable_content_hash: bool,
    pub exact_version_output_hash: bool,
    pub platform_identity: bool,
    pub runner_identity: bool,
    pub process_environment_identity: bool,
}
impl FutureRealizationRequirements {
    pub const V1: Self = Self {
        recursive_runtime_closure: true,
        per_object_content_hash: true,
        per_object_reference_set: true,
        executable_content_hash: true,
        exact_version_output_hash: true,
        platform_identity: true,
        runner_identity: true,
        process_environment_identity: true,
    };
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ExecutionProfileError {
    #[error("source revision must not be blank")] BlankSourceRevision,
    #[error("target triple must not be blank")] BlankTargetTriple,
    #[error("CPU feature profile must not be blank")] BlankCpuFeatureProfile,
    #[error("execution selection field {field} has an unset all-zero commitment")]
    ZeroCommitment { field: &'static str },
}

fn hash_profile(
    source_revision: &str,
    target_triple: &str,
    cpu_feature_profile: &str,
    commitments: [[u8; 32]; 6],
) -> [u8; 32] {
    let mut hasher = Hasher::new();
    frame(&mut hasher, RESOURCE_CAMPAIGN_EXECUTION_PROFILE_V1.as_bytes());
    frame(&mut hasher, source_revision.as_bytes());
    for commitment in commitments { hasher.update(&commitment); }
    frame(&mut hasher, target_triple.as_bytes());
    frame(&mut hasher, cpu_feature_profile.as_bytes());
    *hasher.finalize().as_bytes()
}

fn hash_binding(campaign_id: CampaignManifestId, profile_id: ExecutionSelectionProfileId) -> [u8; 32] {
    let mut hasher = Hasher::new();
    frame(&mut hasher, b"symthaea.resource-campaign.execution-binding.v1");
    hasher.update(campaign_id.as_bytes());
    hasher.update(profile_id.as_bytes());
    *hasher.finalize().as_bytes()
}

fn frame(hasher: &mut Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_operations_research::ObjectiveDirection;
    use symthaea_resource_objective_metric::{ObjectiveMetric, ObjectiveStatistic};
    use symthaea_resource_shadow_benchmark::EvaluationObjective;
    use symthaea_resource_shadow_campaign::{CampaignManifest, CampaignScenarioSpec};

    fn d(byte: u8) -> [u8; 32] { [byte; 32] }
    fn profile() -> ExecutionSelectionProfile {
        ExecutionSelectionProfile::new(
            "git:deadbeef", d(1), d(2), d(3), d(4),
            "x86_64-unknown-linux-gnu", "x86_64-baseline-v1", d(5), d(6),
        ).unwrap()
    }
    fn campaign(seed: u64) -> CampaignManifest {
        let metric = ObjectiveMetric::new(
            "cost", "cost.realized.usd_micro.v1", "currency.usd_micro.v1",
            ObjectiveStatistic::CandidateTotal,
        ).unwrap();
        CampaignManifest::new(
            "campaign",
            vec![EvaluationObjective::new(metric, ObjectiveDirection::Minimize)],
            vec![CampaignScenarioSpec {
                scenario_id: "scenario-a".into(), seed,
                planning_input_commitment: d(10), normalization_profile_commitment: d(11),
                hdc_preference_profile_commitment: d(12), outcome_protocol_commitment: d(13),
            }],
        ).unwrap()
    }

    #[test]
    fn profile_can_only_claim_selection_binding() {
        assert_eq!(profile().qualification_state(), ExecutionQualificationState::SelectionBoundOnly);
        assert!(FutureRealizationRequirements::V1.recursive_runtime_closure);
        assert!(FutureRealizationRequirements::V1.runner_identity);
    }

    #[test]
    fn platform_and_environment_change_profile_identity() {
        let base = profile();
        let platform = ExecutionSelectionProfile::new(
            "git:deadbeef", d(1), d(2), d(3), d(4),
            "aarch64-unknown-linux-gnu", "aarch64-baseline-v1", d(5), d(6),
        ).unwrap();
        let environment = ExecutionSelectionProfile::new(
            "git:deadbeef", d(1), d(2), d(3), d(4),
            "x86_64-unknown-linux-gnu", "x86_64-baseline-v1", d(5), d(99),
        ).unwrap();
        assert_ne!(base.profile_id(), platform.profile_id());
        assert_ne!(base.profile_id(), environment.profile_id());
    }

    #[test]
    fn unset_selection_commitment_fails_closed() {
        assert!(matches!(
            ExecutionSelectionProfile::new(
                "git:deadbeef", [0; 32], d(2), d(3), d(4),
                "x86_64-unknown-linux-gnu", "baseline", d(5), d(6),
            ),
            Err(ExecutionProfileError::ZeroCommitment { field: "workspace_tree" })
        ));
    }

    #[test]
    fn binding_commits_campaign_and_execution_roots() {
        let a = ExecutionBoundCampaign::new(campaign(7), profile());
        let b = ExecutionBoundCampaign::new(campaign(8), profile());
        assert_ne!(a.binding_id(), b.binding_id());

        let changed_source = ExecutionSelectionProfile::new(
            "git:other", d(1), d(2), d(3), d(4),
            "x86_64-unknown-linux-gnu", "x86_64-baseline-v1", d(5), d(6),
        ).unwrap();
        let c = ExecutionBoundCampaign::new(campaign(7), changed_source);
        assert_ne!(a.binding_id(), c.binding_id());
    }
}
