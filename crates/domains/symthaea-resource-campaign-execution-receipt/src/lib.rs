// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Structural execution-receipt matching for resource shadow campaigns.
//!
//! A receipt claim may describe a realized closure and execution environment, but
//! this crate does not query Nix, recompute closure hashes, inspect a runner, or
//! execute a benchmark. Its strongest positive theorem is only:
//!
//! `receipt claim is structurally complete and matches one exact selection profile`.
//!
//! Independent verification of the claimed content remains a later evidence layer.

#![deny(unsafe_code)]

use blake3::Hasher;
use symthaea_resource_campaign_execution_profile::{
    ExecutionSelectionProfile, ExecutionSelectionProfileId,
};
use thiserror::Error;

pub const RESOURCE_CAMPAIGN_EXECUTION_RECEIPT_V1: &str =
    "symthaea.resource-campaign-execution-receipt.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExecutionReceiptId([u8; 32]);
impl ExecutionReceiptId {
    pub fn as_bytes(&self) -> &[u8; 32] { &self.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiptQualificationState {
    StructurallyMatchedOnly,
}

/// Untrusted external claim describing one realized execution capsule.
/// Commitment bytes are opaque until a later verifier checks the referenced data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionReceiptClaim {
    pub selection_profile_id: ExecutionSelectionProfileId,
    pub recursive_runtime_closure_commitment: [u8; 32],
    pub closure_content_census_commitment: [u8; 32],
    pub closure_reference_graph_commitment: [u8; 32],
    pub executable_content_commitment: [u8; 32],
    pub exact_version_output_commitment: [u8; 32],
    pub target_triple: String,
    pub cpu_feature_profile: String,
    pub platform_identity_commitment: [u8; 32],
    pub runner_identity_commitment: [u8; 32],
    pub process_environment_commitment: [u8; 32],
    pub nix_version: String,
    pub evidence_ref: String,
}

/// Positive structural/correlation witness. This is intentionally not named
/// `VerifiedExecutionReceipt`: no external content has been independently checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileMatchedExecutionReceipt {
    selection_profile: ExecutionSelectionProfile,
    claim: ExecutionReceiptClaim,
    receipt_id: ExecutionReceiptId,
}

impl ProfileMatchedExecutionReceipt {
    pub fn selection_profile(&self) -> &ExecutionSelectionProfile { &self.selection_profile }
    pub fn claim(&self) -> &ExecutionReceiptClaim { &self.claim }
    pub fn receipt_id(&self) -> ExecutionReceiptId { self.receipt_id }
    pub fn qualification_state(&self) -> ReceiptQualificationState {
        ReceiptQualificationState::StructurallyMatchedOnly
    }
}

pub fn match_execution_receipt(
    selection_profile: &ExecutionSelectionProfile,
    claim: ExecutionReceiptClaim,
) -> Result<ProfileMatchedExecutionReceipt, ExecutionReceiptError> {
    validate_claim_shape(&claim)?;

    if claim.selection_profile_id != selection_profile.profile_id() {
        return Err(ExecutionReceiptError::SelectionProfileMismatch);
    }
    if claim.target_triple != selection_profile.target_triple() {
        return Err(ExecutionReceiptError::TargetTripleMismatch {
            expected: selection_profile.target_triple().to_owned(),
            actual: claim.target_triple,
        });
    }
    if claim.cpu_feature_profile != selection_profile.cpu_feature_profile() {
        return Err(ExecutionReceiptError::CpuFeatureProfileMismatch {
            expected: selection_profile.cpu_feature_profile().to_owned(),
            actual: claim.cpu_feature_profile,
        });
    }
    if claim.process_environment_commitment != *selection_profile.process_environment_commitment() {
        return Err(ExecutionReceiptError::ProcessEnvironmentMismatch);
    }

    let receipt_id = ExecutionReceiptId(hash_receipt(&claim));
    Ok(ProfileMatchedExecutionReceipt {
        selection_profile: selection_profile.clone(),
        claim,
        receipt_id,
    })
}

fn validate_claim_shape(claim: &ExecutionReceiptClaim) -> Result<(), ExecutionReceiptError> {
    if claim.target_triple.trim().is_empty() {
        return Err(ExecutionReceiptError::BlankTargetTriple);
    }
    if claim.cpu_feature_profile.trim().is_empty() {
        return Err(ExecutionReceiptError::BlankCpuFeatureProfile);
    }
    if claim.nix_version.trim().is_empty() {
        return Err(ExecutionReceiptError::BlankNixVersion);
    }
    if claim.evidence_ref.trim().is_empty() {
        return Err(ExecutionReceiptError::BlankEvidenceRef);
    }
    for (field, value) in [
        ("recursive_runtime_closure", claim.recursive_runtime_closure_commitment),
        ("closure_content_census", claim.closure_content_census_commitment),
        ("closure_reference_graph", claim.closure_reference_graph_commitment),
        ("executable_content", claim.executable_content_commitment),
        ("exact_version_output", claim.exact_version_output_commitment),
        ("platform_identity", claim.platform_identity_commitment),
        ("runner_identity", claim.runner_identity_commitment),
        ("process_environment", claim.process_environment_commitment),
    ] {
        if value == [0; 32] {
            return Err(ExecutionReceiptError::ZeroCommitment { field });
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ExecutionReceiptError {
    #[error("receipt target triple must not be blank")] BlankTargetTriple,
    #[error("receipt CPU feature profile must not be blank")] BlankCpuFeatureProfile,
    #[error("receipt Nix version must not be blank")] BlankNixVersion,
    #[error("receipt evidence ref must not be blank")] BlankEvidenceRef,
    #[error("receipt field {field} has an unset all-zero commitment")]
    ZeroCommitment { field: &'static str },
    #[error("receipt selection-profile identity does not match the selected profile")]
    SelectionProfileMismatch,
    #[error("receipt target triple {actual} does not match selected {expected}")]
    TargetTripleMismatch { expected: String, actual: String },
    #[error("receipt CPU feature profile {actual} does not match selected {expected}")]
    CpuFeatureProfileMismatch { expected: String, actual: String },
    #[error("receipt process-environment commitment does not match selected environment")]
    ProcessEnvironmentMismatch,
}

fn hash_receipt(claim: &ExecutionReceiptClaim) -> [u8; 32] {
    let mut hasher = Hasher::new();
    frame(&mut hasher, RESOURCE_CAMPAIGN_EXECUTION_RECEIPT_V1.as_bytes());
    hasher.update(claim.selection_profile_id.as_bytes());
    for commitment in [
        claim.recursive_runtime_closure_commitment,
        claim.closure_content_census_commitment,
        claim.closure_reference_graph_commitment,
        claim.executable_content_commitment,
        claim.exact_version_output_commitment,
    ] {
        hasher.update(&commitment);
    }
    frame(&mut hasher, claim.target_triple.as_bytes());
    frame(&mut hasher, claim.cpu_feature_profile.as_bytes());
    hasher.update(&claim.platform_identity_commitment);
    hasher.update(&claim.runner_identity_commitment);
    hasher.update(&claim.process_environment_commitment);
    frame(&mut hasher, claim.nix_version.as_bytes());
    frame(&mut hasher, claim.evidence_ref.as_bytes());
    *hasher.finalize().as_bytes()
}

fn frame(hasher: &mut Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(byte: u8) -> [u8; 32] { [byte; 32] }
    fn profile() -> ExecutionSelectionProfile {
        ExecutionSelectionProfile::new(
            "git:deadbeef", d(1), d(2), d(3), d(4),
            "x86_64-unknown-linux-gnu", "x86_64-baseline-v1", d(5), d(6),
        ).unwrap()
    }
    fn claim(profile: &ExecutionSelectionProfile) -> ExecutionReceiptClaim {
        ExecutionReceiptClaim {
            selection_profile_id: profile.profile_id(),
            recursive_runtime_closure_commitment: d(10),
            closure_content_census_commitment: d(11),
            closure_reference_graph_commitment: d(12),
            executable_content_commitment: d(13),
            exact_version_output_commitment: d(14),
            target_triple: profile.target_triple().into(),
            cpu_feature_profile: profile.cpu_feature_profile().into(),
            platform_identity_commitment: d(15),
            runner_identity_commitment: d(16),
            process_environment_commitment: *profile.process_environment_commitment(),
            nix_version: "nix 2.x".into(),
            evidence_ref: "evidence://execution-receipt".into(),
        }
    }

    #[test]
    fn complete_matching_claim_is_only_structurally_matched() {
        let profile = profile();
        let receipt = match_execution_receipt(&profile, claim(&profile)).unwrap();
        assert_eq!(receipt.qualification_state(), ReceiptQualificationState::StructurallyMatchedOnly);
        assert_eq!(receipt.selection_profile(), &profile);
    }

    #[test]
    fn profile_target_and_environment_mismatches_fail_closed() {
        let profile = profile();
        let mut wrong_target = claim(&profile);
        wrong_target.target_triple = "aarch64-unknown-linux-gnu".into();
        assert!(matches!(
            match_execution_receipt(&profile, wrong_target),
            Err(ExecutionReceiptError::TargetTripleMismatch { .. })
        ));

        let mut wrong_env = claim(&profile);
        wrong_env.process_environment_commitment = d(99);
        assert!(matches!(
            match_execution_receipt(&profile, wrong_env),
            Err(ExecutionReceiptError::ProcessEnvironmentMismatch)
        ));
    }

    #[test]
    fn incomplete_claim_fails_before_matching() {
        let profile = profile();
        let mut incomplete = claim(&profile);
        incomplete.closure_reference_graph_commitment = [0; 32];
        assert!(matches!(
            match_execution_receipt(&profile, incomplete),
            Err(ExecutionReceiptError::ZeroCommitment { field: "closure_reference_graph" })
        ));
    }

    #[test]
    fn claimed_closure_and_runner_change_receipt_identity() {
        let profile = profile();
        let base = match_execution_receipt(&profile, claim(&profile)).unwrap();
        let mut changed_closure = claim(&profile);
        changed_closure.recursive_runtime_closure_commitment = d(80);
        let changed_closure = match_execution_receipt(&profile, changed_closure).unwrap();
        assert_ne!(base.receipt_id(), changed_closure.receipt_id());

        let mut changed_runner = claim(&profile);
        changed_runner.runner_identity_commitment = d(81);
        let changed_runner = match_execution_receipt(&profile, changed_runner).unwrap();
        assert_ne!(base.receipt_id(), changed_runner.receipt_id());
    }
}
