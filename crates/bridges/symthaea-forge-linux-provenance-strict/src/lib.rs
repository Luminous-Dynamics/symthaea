// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Strict promotion gate over the read-only Bubblewrap Nix provenance observations.
//!
//! The v1 provenance collector intentionally preserves observations even when a fixed Nix command
//! fails. That is useful evidence, but parseable stdout from a failed command must not satisfy a
//! stronger provenance proposition. This crate therefore adds a separate v2 gate rather than
//! changing the meaning of the v1 receipt.
//!
//! A strict binding requires all six fixed observations to have exited successfully:
//! `nix --version`, `nix-store --version`, `nix store verify`, `nix-store --query --hash`,
//! `--deriver`, and `--references`. It also requires the already-recorded non-empty NAR hash and a
//! known derivation. Exact tool/output observations remain content-addressed by the v1 collector.
//!
//! This still does not independently prove Nix-tool semantics or Bubblewrap semantic correctness.

use serde::Serialize;
use symthaea_algorithms::ContentId;
use symthaea_forge_linux_provenance::{
    BubblewrapNixProvenanceError, BubblewrapNixProvenancePolicy,
    BubblewrapNixProvenanceReceipt, FixedCommandObservation, ObservationState,
    SemanticCorrectnessStatus,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StrictNixProvenanceError {
    #[error(transparent)]
    Upstream(#[from] BubblewrapNixProvenanceError),
    #[error("required fixed Nix provenance command failed: {command} exited {exit_code}")]
    RequiredCommandFailed {
        command: &'static str,
        exit_code: i32,
    },
    #[error("strict Nix provenance requires a successful store verification")]
    VerificationDidNotPass,
    #[error("strict Nix provenance requires a non-empty NAR hash")]
    MissingNarHash,
    #[error("strict Nix provenance requires a known derivation")]
    UnknownDeriver,
    #[error("strict Nix provenance binding does not match supplied evidence")]
    ScopeMismatch,
    #[error("strict Nix provenance binding identity does not match canonical fields")]
    IdentityMismatch,
}

/// Deliberate nonclaim retained by the stricter gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum StrictSemanticCorrectnessStatus {
    NotEstablishedV2,
}

/// Stronger Nix provenance proposition issued only after every fixed observation command succeeds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StrictBubblewrapNixProvenanceBindingV2 {
    id: ContentId,
    policy_id: ContentId,
    receipt_id: ContentId,
    bubblewrap_artifact_id: ContentId,
    store_output_path: String,
    nar_hash: String,
    deriver: String,
    references_id: ContentId,
    nix_version_observation_id: ContentId,
    nix_store_version_observation_id: ContentId,
    store_verify_observation_id: ContentId,
    query_hash_observation_id: ContentId,
    query_deriver_observation_id: ContentId,
    query_references_observation_id: ContentId,
    bubblewrap_semantic_correctness: StrictSemanticCorrectnessStatus,
    nix_tool_semantic_correctness: StrictSemanticCorrectnessStatus,
}

impl StrictBubblewrapNixProvenanceBindingV2 {
    pub fn issue(
        policy: &BubblewrapNixProvenancePolicy,
        receipt: &BubblewrapNixProvenanceReceipt,
    ) -> Result<Self, StrictNixProvenanceError> {
        receipt.validate_for(policy)?;
        if receipt.bubblewrap_semantic_correctness()
            != SemanticCorrectnessStatus::NotEstablishedV1
            || receipt.nix_tool_semantic_correctness()
                != SemanticCorrectnessStatus::NotEstablishedV1
        {
            return Err(StrictNixProvenanceError::ScopeMismatch);
        }
        require_success("nix --version", receipt.nix_version())?;
        require_success("nix-store --version", receipt.nix_store_version())?;
        require_success("nix store verify", receipt.store_verify())?;
        require_success("nix-store --query --hash", receipt.query_hash())?;
        require_success("nix-store --query --deriver", receipt.query_deriver())?;
        require_success("nix-store --query --references", receipt.query_references())?;

        if receipt.verification_state() != ObservationState::ObservedPassed {
            return Err(StrictNixProvenanceError::VerificationDidNotPass);
        }
        if receipt.nar_hash().trim().is_empty() {
            return Err(StrictNixProvenanceError::MissingNarHash);
        }
        let deriver = receipt
            .deriver()
            .filter(|value| !value.trim().is_empty())
            .ok_or(StrictNixProvenanceError::UnknownDeriver)?
            .to_string();

        let references_id = derive_references_id(receipt.references());
        let id = derive_binding_id(
            policy.id(),
            receipt.id(),
            receipt.bubblewrap_artifact_id(),
            receipt.store_output_path(),
            receipt.nar_hash(),
            &deriver,
            &references_id,
            receipt.nix_version().id(),
            receipt.nix_store_version().id(),
            receipt.store_verify().id(),
            receipt.query_hash().id(),
            receipt.query_deriver().id(),
            receipt.query_references().id(),
        );
        Ok(Self {
            id,
            policy_id: policy.id().clone(),
            receipt_id: receipt.id().clone(),
            bubblewrap_artifact_id: receipt.bubblewrap_artifact_id().clone(),
            store_output_path: receipt.store_output_path().to_string(),
            nar_hash: receipt.nar_hash().to_string(),
            deriver,
            references_id,
            nix_version_observation_id: receipt.nix_version().id().clone(),
            nix_store_version_observation_id: receipt.nix_store_version().id().clone(),
            store_verify_observation_id: receipt.store_verify().id().clone(),
            query_hash_observation_id: receipt.query_hash().id().clone(),
            query_deriver_observation_id: receipt.query_deriver().id().clone(),
            query_references_observation_id: receipt.query_references().id().clone(),
            bubblewrap_semantic_correctness: StrictSemanticCorrectnessStatus::NotEstablishedV2,
            nix_tool_semantic_correctness: StrictSemanticCorrectnessStatus::NotEstablishedV2,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn policy_id(&self) -> &ContentId { &self.policy_id }
    pub fn receipt_id(&self) -> &ContentId { &self.receipt_id }
    pub fn bubblewrap_artifact_id(&self) -> &ContentId { &self.bubblewrap_artifact_id }
    pub fn store_output_path(&self) -> &str { &self.store_output_path }
    pub fn nar_hash(&self) -> &str { &self.nar_hash }
    pub fn deriver(&self) -> &str { &self.deriver }
    pub fn references_id(&self) -> &ContentId { &self.references_id }
    pub fn bubblewrap_semantic_correctness(&self) -> StrictSemanticCorrectnessStatus {
        self.bubblewrap_semantic_correctness
    }
    pub fn nix_tool_semantic_correctness(&self) -> StrictSemanticCorrectnessStatus {
        self.nix_tool_semantic_correctness
    }

    pub fn validate_for(
        &self,
        policy: &BubblewrapNixProvenancePolicy,
        receipt: &BubblewrapNixProvenanceReceipt,
    ) -> Result<(), StrictNixProvenanceError> {
        let rebuilt = Self::issue(policy, receipt)?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(StrictNixProvenanceError::IdentityMismatch)
        }
    }
}

fn require_success(
    command: &'static str,
    observation: &FixedCommandObservation,
) -> Result<(), StrictNixProvenanceError> {
    if observation.exit_code() == 0 {
        Ok(())
    } else {
        Err(StrictNixProvenanceError::RequiredCommandFailed {
            command,
            exit_code: observation.exit_code(),
        })
    }
}

fn derive_references_id(references: &[String]) -> ContentId {
    let mut parts = vec![(references.len() as u64).to_be_bytes().to_vec()];
    for reference in references {
        parts.push((reference.len() as u64).to_be_bytes().to_vec());
        parts.push(reference.as_bytes().to_vec());
    }
    ContentId::derive(
        "symthaea.forge-bubblewrap-nix-references.v2",
        parts.iter().map(Vec::as_slice),
    )
}

#[allow(clippy::too_many_arguments)]
fn derive_binding_id(
    policy_id: &ContentId,
    receipt_id: &ContentId,
    artifact_id: &ContentId,
    store_output_path: &str,
    nar_hash: &str,
    deriver: &str,
    references_id: &ContentId,
    nix_version_id: &ContentId,
    nix_store_version_id: &ContentId,
    store_verify_id: &ContentId,
    query_hash_id: &ContentId,
    query_deriver_id: &ContentId,
    query_references_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-bubblewrap-nix-provenance-binding.v2",
        [
            policy_id.as_str().as_bytes(),
            receipt_id.as_str().as_bytes(),
            artifact_id.as_str().as_bytes(),
            store_output_path.as_bytes(),
            nar_hash.as_bytes(),
            deriver.as_bytes(),
            references_id.as_str().as_bytes(),
            nix_version_id.as_str().as_bytes(),
            nix_store_version_id.as_str().as_bytes(),
            store_verify_id.as_str().as_bytes(),
            query_hash_id.as_str().as_bytes(),
            query_deriver_id.as_str().as_bytes(),
            query_references_id.as_str().as_bytes(),
            b"all-fixed-commands-exit-zero",
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn references_identity_is_order_sensitive_and_length_framed() {
        let a = derive_references_id(&["/nix/store/a".into(), "/nix/store/b".into()]);
        let b = derive_references_id(&["/nix/store/b".into(), "/nix/store/a".into()]);
        assert_ne!(a, b);
    }

    #[test]
    fn strict_semantic_correctness_remains_unestablished() {
        assert_eq!(
            StrictSemanticCorrectnessStatus::NotEstablishedV2,
            StrictSemanticCorrectnessStatus::NotEstablishedV2
        );
        assert_eq!(
            SemanticCorrectnessStatus::NotEstablishedV1,
            SemanticCorrectnessStatus::NotEstablishedV1
        );
    }
}
