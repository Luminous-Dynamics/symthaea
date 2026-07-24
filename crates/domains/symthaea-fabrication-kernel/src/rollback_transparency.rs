// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transparency-log publication and witnessed verification of rollback authority.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::release_rollback::AuthorizedReleaseRollback;
use crate::transparency::{
    TransparencyError, TransparencyInclusionProof, TransparencyLog, digest_transparency_entry,
    verify_transparency_inclusion,
};
use crate::transparency_checkpoint::VerifiedTransparencyCheckpoint;
use crate::transparency_witness::VerifiedTransparencyWitnessQuorum;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RollbackTransparencyError {
    Transparency(TransparencyError),
    MissingEntry,
    EntryMismatch,
    CheckpointMismatch,
    WitnessMismatch,
}

#[derive(Debug, Clone)]
pub struct VerifiedRollbackTransparency {
    rollback_digest: Sha256Digest,
    inclusion_digest: Sha256Digest,
    checkpoint_digest: Sha256Digest,
    witness_quorum_digest: Sha256Digest,
}

impl VerifiedRollbackTransparency {
    pub fn rollback_digest(&self) -> Sha256Digest {
        self.rollback_digest
    }
    pub fn inclusion_digest(&self) -> Sha256Digest {
        self.inclusion_digest
    }
    pub fn checkpoint_digest(&self) -> Sha256Digest {
        self.checkpoint_digest
    }
    pub fn witness_quorum_digest(&self) -> Sha256Digest {
        self.witness_quorum_digest
    }
}

pub fn publish_release_rollback(
    log: &mut TransparencyLog,
    rollback: &AuthorizedReleaseRollback,
    recorded_at_unix_s: u64,
) -> Result<TransparencyInclusionProof, RollbackTransparencyError> {
    log.append(
        recorded_at_unix_s,
        "release-rollback",
        rollback.rollback_digest(),
    )
    .map_err(RollbackTransparencyError::Transparency)?;
    log.inclusion_proof(log.entries.len().saturating_sub(1))
        .map_err(RollbackTransparencyError::Transparency)
}

pub fn verify_release_rollback_transparency(
    log: &TransparencyLog,
    rollback: &AuthorizedReleaseRollback,
    proof: &TransparencyInclusionProof,
    checkpoint: &VerifiedTransparencyCheckpoint,
    witness_quorum: &VerifiedTransparencyWitnessQuorum,
) -> Result<VerifiedRollbackTransparency, RollbackTransparencyError> {
    verify_transparency_inclusion(proof).map_err(RollbackTransparencyError::Transparency)?;
    let entry = log
        .entries
        .get(proof.leaf_index as usize)
        .ok_or(RollbackTransparencyError::MissingEntry)?;
    if entry.kind != "release-rollback"
        || entry.subject_digest != rollback.rollback_digest()
        || digest_transparency_entry(entry).map_err(RollbackTransparencyError::Transparency)?
            != proof.leaf_digest
    {
        return Err(RollbackTransparencyError::EntryMismatch);
    }
    if checkpoint.checkpoint().log_size != proof.tree_size
        || checkpoint.checkpoint().root_digest != proof.root_digest
    {
        return Err(RollbackTransparencyError::CheckpointMismatch);
    }
    if witness_quorum.checkpoint_digest() != checkpoint.checkpoint_digest() {
        return Err(RollbackTransparencyError::WitnessMismatch);
    }
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.rollback-inclusion-digest.v1\0");
    hasher.update(&proof.leaf_digest.0);
    hasher.update(&proof.root_digest.0);
    hasher.update(&proof.tree_size.to_le_bytes());
    hasher.update(&proof.leaf_index.to_le_bytes());
    Ok(VerifiedRollbackTransparency {
        rollback_digest: rollback.rollback_digest(),
        inclusion_digest: hasher.finalize(),
        checkpoint_digest: checkpoint.checkpoint_digest(),
        witness_quorum_digest: witness_quorum.witness_quorum_digest(),
    })
}
