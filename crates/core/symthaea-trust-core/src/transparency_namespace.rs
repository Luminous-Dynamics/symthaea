// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Stable transparency-log namespace authority across delegated trust-root rotation.
//!
//! The namespace identity is stable and root-independent after creation. Authority
//! over that namespace is root-specific and must be explicitly transferred across
//! a valid dual-root transition while preserving append-only Merkle continuity.
//! This module does not establish global log consistency or witness independence.

use serde::Serialize;

use crate::{
    AuthenticatedTransparencyCheckpoint, AuthorizedTrustRoleAttestation, AuthorizedTrustState,
    FramedDigest, FrozenTrustRoot, RootRoleQuorumProof, Sha256Digest,
    TransparencyConsistencyProof, TrustRole, TrustRootTransitionContract, TrustSnapshot,
    authorize_root_transition, verify_transparency_consistency,
};

pub const TRANSPARENCY_LOG_NAMESPACE_SCHEMA: &str = "symthaea.transparency-log-namespace.v1";
const NAMESPACE_IDENTITY_DOMAIN: &str = "symthaea.transparency-log-namespace.identity.v1";
const NAMESPACE_AUTHORITY_DOMAIN: &str = "symthaea.transparency-log-namespace-authority.identity.v1";

/// Caller-declared stable namespace identity. Possession of this value grants no
/// authority until a root-authorized TransparencyLog role binds it to a checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TransparencyLogNamespace {
    schema_version: String,
    namespace_seed_sha256: Sha256Digest,
    namespace_sha256: Sha256Digest,
}

impl TransparencyLogNamespace {
    pub fn new(namespace_seed_sha256: Sha256Digest) -> Self {
        let mut digest = FramedDigest::new(NAMESPACE_IDENTITY_DOMAIN);
        digest.text(TRANSPARENCY_LOG_NAMESPACE_SCHEMA);
        digest.text(namespace_seed_sha256.as_str());
        let namespace_sha256 = digest.digest();
        Self {
            schema_version: TRANSPARENCY_LOG_NAMESPACE_SCHEMA.into(),
            namespace_seed_sha256,
            namespace_sha256,
        }
    }

    pub fn namespace_seed_sha256(&self) -> &Sha256Digest { &self.namespace_seed_sha256 }
    pub fn namespace_sha256(&self) -> &Sha256Digest { &self.namespace_sha256 }
    pub const fn authority_established(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransparencyNamespaceAuthorizationError {
    WrongRole,
    RootAuthorityMismatch,
    SubjectMismatch,
    PayloadMismatch,
    ContextMustBeEmpty,
}

/// Root-specific authority over one stable transparency namespace.
///
/// Serializable for retained evidence but intentionally not deserializable into
/// authority. `namespace_sha256` remains stable while `namespace_authority_sha256`
/// changes whenever authority transfers to a new root.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AuthorizedTransparencyLogNamespace {
    namespace_sha256: Sha256Digest,
    root_authority_sha256: Sha256Digest,
    trust_snapshot_authority_sha256: Sha256Digest,
    log_role_authority_sha256: Sha256Digest,
    anchor_checkpoint_sha256: Sha256Digest,
    anchor_tree_size: u64,
    anchor_root_sha256: Sha256Digest,
    predecessor_namespace_authority_sha256: Option<Sha256Digest>,
    root_transition_sha256: Option<Sha256Digest>,
    namespace_authority_sha256: Sha256Digest,
}

impl AuthorizedTransparencyLogNamespace {
    pub fn namespace_sha256(&self) -> &Sha256Digest { &self.namespace_sha256 }
    pub fn root_authority_sha256(&self) -> &Sha256Digest { &self.root_authority_sha256 }
    pub fn trust_snapshot_authority_sha256(&self) -> &Sha256Digest {
        &self.trust_snapshot_authority_sha256
    }
    pub fn log_role_authority_sha256(&self) -> &Sha256Digest { &self.log_role_authority_sha256 }
    pub fn anchor_checkpoint_sha256(&self) -> &Sha256Digest { &self.anchor_checkpoint_sha256 }
    pub fn anchor_tree_size(&self) -> u64 { self.anchor_tree_size }
    pub fn anchor_root_sha256(&self) -> &Sha256Digest { &self.anchor_root_sha256 }
    pub fn predecessor_namespace_authority_sha256(&self) -> Option<&Sha256Digest> {
        self.predecessor_namespace_authority_sha256.as_ref()
    }
    pub fn root_transition_sha256(&self) -> Option<&Sha256Digest> {
        self.root_transition_sha256.as_ref()
    }
    pub fn namespace_authority_sha256(&self) -> &Sha256Digest { &self.namespace_authority_sha256 }
    pub const fn namespace_authority_established(&self) -> bool { true }
    pub const fn global_log_consistency_established(&self) -> bool { false }
}

pub fn authorize_transparency_log_namespace(
    namespace: &TransparencyLogNamespace,
    checkpoint: &AuthenticatedTransparencyCheckpoint,
    log_authority: &AuthorizedTrustRoleAttestation,
) -> Result<AuthorizedTransparencyLogNamespace, TransparencyNamespaceAuthorizationError> {
    if log_authority.role() != TrustRole::TransparencyLog {
        return Err(TransparencyNamespaceAuthorizationError::WrongRole);
    }
    if log_authority.root_authority_sha256() != checkpoint.root_authority_sha256() {
        return Err(TransparencyNamespaceAuthorizationError::RootAuthorityMismatch);
    }
    if log_authority.subject_sha256() != namespace.namespace_sha256() {
        return Err(TransparencyNamespaceAuthorizationError::SubjectMismatch);
    }
    if log_authority.payload_sha256() != checkpoint.checkpoint_sha256() {
        return Err(TransparencyNamespaceAuthorizationError::PayloadMismatch);
    }
    if log_authority.context_sha256().is_some() {
        return Err(TransparencyNamespaceAuthorizationError::ContextMustBeEmpty);
    }

    let namespace_authority_sha256 = namespace_authority_digest(
        namespace.namespace_sha256(),
        log_authority,
        checkpoint,
        None,
        None,
    );
    Ok(AuthorizedTransparencyLogNamespace {
        namespace_sha256: namespace.namespace_sha256().clone(),
        root_authority_sha256: log_authority.root_authority_sha256().clone(),
        trust_snapshot_authority_sha256: log_authority.trust_snapshot_authority_sha256().clone(),
        log_role_authority_sha256: log_authority.authority_sha256().clone(),
        anchor_checkpoint_sha256: checkpoint.checkpoint_sha256().clone(),
        anchor_tree_size: checkpoint.statement().tree_size(),
        anchor_root_sha256: checkpoint.statement().root_sha256().clone(),
        predecessor_namespace_authority_sha256: None,
        root_transition_sha256: None,
        namespace_authority_sha256,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransparencyNamespaceTransferError {
    PreviousStateMismatch,
    HandoffCheckpointWrongRoot,
    HandoffTreeRollback,
    HandoffSameSizeRootMismatch,
    MissingAnchorConsistencyProof,
    AnchorConsistencySizeMismatch,
    AnchorConsistencyRootMismatch,
    InvalidAnchorConsistencyProof,
    HandoffAfterTransition,
    RootTransitionRejected,
    NextCheckpointWrongRoot,
    NextCheckpointBeforeTransition,
    CrossTransitionSizeMismatch,
    CrossTransitionRootMismatch,
    InvalidCrossTransitionConsistencyProof,
    WrongRole,
    NextRoleRootMismatch,
    NextRoleSnapshotMismatch,
    SubjectMismatch,
    PayloadMismatch,
    ContextMismatch,
}

#[allow(clippy::too_many_arguments)]
pub fn transfer_transparency_log_namespace(
    previous: &AuthorizedTransparencyLogNamespace,
    handoff_checkpoint: &AuthenticatedTransparencyCheckpoint,
    anchor_to_handoff: Option<&TransparencyConsistencyProof>,
    previous_state: &AuthorizedTrustState,
    previous_root: &FrozenTrustRoot,
    next_root: &FrozenTrustRoot,
    next_snapshot: &TrustSnapshot,
    transition_at_unix_s: u64,
    transition_contract: &TrustRootTransitionContract,
    old_root_proof: &RootRoleQuorumProof,
    new_root_proof: &RootRoleQuorumProof,
    next_checkpoint: &AuthenticatedTransparencyCheckpoint,
    cross_transition_consistency: &TransparencyConsistencyProof,
    next_log_authority: &AuthorizedTrustRoleAttestation,
) -> Result<AuthorizedTransparencyLogNamespace, TransparencyNamespaceTransferError> {
    if previous.root_authority_sha256() != previous_state.root().authority_sha256()
        || previous_state.root().root_sha256() != previous_root.root_sha256()
    {
        return Err(TransparencyNamespaceTransferError::PreviousStateMismatch);
    }
    if handoff_checkpoint.root_authority_sha256() != previous.root_authority_sha256() {
        return Err(TransparencyNamespaceTransferError::HandoffCheckpointWrongRoot);
    }

    let handoff_size = handoff_checkpoint.statement().tree_size();
    let handoff_root = handoff_checkpoint.statement().root_sha256();
    if handoff_size < previous.anchor_tree_size() {
        return Err(TransparencyNamespaceTransferError::HandoffTreeRollback);
    }
    if handoff_size == previous.anchor_tree_size() {
        if handoff_root != previous.anchor_root_sha256() {
            return Err(TransparencyNamespaceTransferError::HandoffSameSizeRootMismatch);
        }
    } else {
        let proof = anchor_to_handoff
            .ok_or(TransparencyNamespaceTransferError::MissingAnchorConsistencyProof)?;
        if proof.first_tree_size() != previous.anchor_tree_size()
            || proof.second_tree_size() != handoff_size
        {
            return Err(TransparencyNamespaceTransferError::AnchorConsistencySizeMismatch);
        }
        if proof.first_root_sha256() != previous.anchor_root_sha256()
            || proof.second_root_sha256() != handoff_root
        {
            return Err(TransparencyNamespaceTransferError::AnchorConsistencyRootMismatch);
        }
        verify_transparency_consistency(proof)
            .map_err(|_| TransparencyNamespaceTransferError::InvalidAnchorConsistencyProof)?;
    }

    let (_, handoff_latest) = handoff_checkpoint.statement().consensus_interval();
    if handoff_latest > transition_at_unix_s {
        return Err(TransparencyNamespaceTransferError::HandoffAfterTransition);
    }

    let next_state = authorize_root_transition(
        previous_state,
        previous_root,
        next_root,
        next_snapshot,
        transition_at_unix_s,
        transition_contract,
        old_root_proof,
        new_root_proof,
    )
    .map_err(|_| TransparencyNamespaceTransferError::RootTransitionRejected)?;

    if next_checkpoint.root_authority_sha256() != next_state.root().authority_sha256() {
        return Err(TransparencyNamespaceTransferError::NextCheckpointWrongRoot);
    }
    let (next_earliest, _) = next_checkpoint.statement().consensus_interval();
    if next_earliest < transition_at_unix_s {
        return Err(TransparencyNamespaceTransferError::NextCheckpointBeforeTransition);
    }

    if cross_transition_consistency.first_tree_size() != handoff_size
        || cross_transition_consistency.second_tree_size() != next_checkpoint.statement().tree_size()
    {
        return Err(TransparencyNamespaceTransferError::CrossTransitionSizeMismatch);
    }
    if cross_transition_consistency.first_root_sha256() != handoff_root
        || cross_transition_consistency.second_root_sha256()
            != next_checkpoint.statement().root_sha256()
    {
        return Err(TransparencyNamespaceTransferError::CrossTransitionRootMismatch);
    }
    verify_transparency_consistency(cross_transition_consistency)
        .map_err(|_| TransparencyNamespaceTransferError::InvalidCrossTransitionConsistencyProof)?;

    if next_log_authority.role() != TrustRole::TransparencyLog {
        return Err(TransparencyNamespaceTransferError::WrongRole);
    }
    if next_log_authority.root_authority_sha256() != next_state.root().authority_sha256() {
        return Err(TransparencyNamespaceTransferError::NextRoleRootMismatch);
    }
    if next_log_authority.trust_snapshot_authority_sha256()
        != next_state.snapshot().authority_sha256()
    {
        return Err(TransparencyNamespaceTransferError::NextRoleSnapshotMismatch);
    }
    if next_log_authority.subject_sha256() != previous.namespace_sha256() {
        return Err(TransparencyNamespaceTransferError::SubjectMismatch);
    }
    if next_log_authority.payload_sha256() != next_checkpoint.checkpoint_sha256() {
        return Err(TransparencyNamespaceTransferError::PayloadMismatch);
    }
    if next_log_authority.context_sha256() != Some(previous.namespace_authority_sha256()) {
        return Err(TransparencyNamespaceTransferError::ContextMismatch);
    }

    let namespace_authority_sha256 = namespace_authority_digest(
        previous.namespace_sha256(),
        next_log_authority,
        next_checkpoint,
        Some(previous.namespace_authority_sha256()),
        Some(transition_contract.transition_sha256()),
    );
    Ok(AuthorizedTransparencyLogNamespace {
        namespace_sha256: previous.namespace_sha256().clone(),
        root_authority_sha256: next_log_authority.root_authority_sha256().clone(),
        trust_snapshot_authority_sha256: next_log_authority.trust_snapshot_authority_sha256().clone(),
        log_role_authority_sha256: next_log_authority.authority_sha256().clone(),
        anchor_checkpoint_sha256: next_checkpoint.checkpoint_sha256().clone(),
        anchor_tree_size: next_checkpoint.statement().tree_size(),
        anchor_root_sha256: next_checkpoint.statement().root_sha256().clone(),
        predecessor_namespace_authority_sha256: Some(previous.namespace_authority_sha256().clone()),
        root_transition_sha256: Some(transition_contract.transition_sha256().clone()),
        namespace_authority_sha256,
    })
}

fn namespace_authority_digest(
    namespace_sha256: &Sha256Digest,
    log_authority: &AuthorizedTrustRoleAttestation,
    checkpoint: &AuthenticatedTransparencyCheckpoint,
    predecessor_namespace_authority_sha256: Option<&Sha256Digest>,
    root_transition_sha256: Option<&Sha256Digest>,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(NAMESPACE_AUTHORITY_DOMAIN);
    digest.text(namespace_sha256.as_str());
    digest.text(log_authority.root_authority_sha256().as_str());
    digest.text(log_authority.trust_snapshot_authority_sha256().as_str());
    digest.text(log_authority.authority_sha256().as_str());
    digest.text(checkpoint.checkpoint_sha256().as_str());
    digest.text(&checkpoint.statement().tree_size().to_string());
    digest.text(checkpoint.statement().root_sha256().as_str());
    digest.optional_sha(predecessor_namespace_authority_sha256);
    digest.optional_sha(root_transition_sha256);
    digest.digest()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(value: &str) -> Sha256Digest { Sha256Digest::of_bytes(value.as_bytes()) }

    #[test]
    fn stable_namespace_identity_depends_only_on_seed() {
        let left = TransparencyLogNamespace::new(sha("stable-log-seed"));
        let right = TransparencyLogNamespace::new(sha("stable-log-seed"));
        assert_eq!(left.namespace_sha256(), right.namespace_sha256());
        assert!(!left.authority_established());
    }

    #[test]
    fn different_namespace_seeds_do_not_alias() {
        let left = TransparencyLogNamespace::new(sha("log-a"));
        let right = TransparencyLogNamespace::new(sha("log-b"));
        assert_ne!(left.namespace_sha256(), right.namespace_sha256());
    }
}
