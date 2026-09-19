// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Authenticated transparency checkpoints over structural Merkle tree heads.
//!
//! A checkpoint statement is authority-independent and binds an exact tree head,
//! predecessor checkpoint, and authenticated bounded time. A TransparencyLog
//! role attestation signs that statement. Only after the exact root authority,
//! tree head, statement, and predecessor context all match can the private
//! `AuthenticatedTransparencyCheckpoint` capability be minted.
//!
//! Checkpoint tracking is append-only and requires a compact Merkle consistency
//! proof whenever tree size grows. Same-size refreshes are allowed only for the
//! same root and only when explicitly chained to the previous checkpoint.

use serde::Serialize;

use crate::{
    AuthorizedTrustRoleAttestation, FramedDigest, Sha256Digest, TransparencyConsistencyProof,
    TransparencyTreeHead, TrustRole, TrustedTime, verify_transparency_consistency,
};

const CHECKPOINT_STATEMENT_DOMAIN: &str = "symthaea.transparency-checkpoint-statement.identity.v1";
const AUTHENTICATED_CHECKPOINT_DOMAIN: &str = "symthaea.authenticated-transparency-checkpoint.identity.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TransparencyCheckpointStatement {
    tree_head_sha256: Sha256Digest,
    tree_size: u64,
    root_sha256: Sha256Digest,
    previous_checkpoint_sha256: Option<Sha256Digest>,
    trusted_time_authority_sha256: Sha256Digest,
    consensus_earliest_unix_s: u64,
    consensus_latest_unix_s: u64,
    statement_sha256: Sha256Digest,
}

impl TransparencyCheckpointStatement {
    pub fn new(
        head: &TransparencyTreeHead,
        previous_checkpoint_sha256: Option<Sha256Digest>,
        trusted_time: &TrustedTime,
    ) -> Self {
        let (earliest, latest) = trusted_time.consensus_interval();
        let statement_sha256 = checkpoint_statement_digest(
            head.tree_head_sha256(),
            head.tree_size(),
            head.root_sha256(),
            previous_checkpoint_sha256.as_ref(),
            trusted_time.authority_sha256(),
            earliest,
            latest,
        );
        Self {
            tree_head_sha256: head.tree_head_sha256().clone(),
            tree_size: head.tree_size(),
            root_sha256: head.root_sha256().clone(),
            previous_checkpoint_sha256,
            trusted_time_authority_sha256: trusted_time.authority_sha256().clone(),
            consensus_earliest_unix_s: earliest,
            consensus_latest_unix_s: latest,
            statement_sha256,
        }
    }

    pub fn tree_head_sha256(&self) -> &Sha256Digest { &self.tree_head_sha256 }
    pub fn tree_size(&self) -> u64 { self.tree_size }
    pub fn root_sha256(&self) -> &Sha256Digest { &self.root_sha256 }
    pub fn previous_checkpoint_sha256(&self) -> Option<&Sha256Digest> {
        self.previous_checkpoint_sha256.as_ref()
    }
    pub fn trusted_time_authority_sha256(&self) -> &Sha256Digest {
        &self.trusted_time_authority_sha256
    }
    pub fn consensus_interval(&self) -> (u64, u64) {
        (self.consensus_earliest_unix_s, self.consensus_latest_unix_s)
    }
    pub fn statement_sha256(&self) -> &Sha256Digest { &self.statement_sha256 }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransparencyCheckpointAuthenticationError {
    TreeHeadMismatch,
    TrustedTimeMismatch,
    WrongRole,
    RootAuthorityMismatch,
    SubjectMismatch,
    PayloadMismatch,
    ContextMismatch,
}

/// Authenticated operator checkpoint. Serializable for evidence retention and
/// intentionally not deserializable into authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AuthenticatedTransparencyCheckpoint {
    statement: TransparencyCheckpointStatement,
    log_role_authority_sha256: Sha256Digest,
    root_authority_sha256: Sha256Digest,
    trust_snapshot_authority_sha256: Sha256Digest,
    checkpoint_sha256: Sha256Digest,
}

impl AuthenticatedTransparencyCheckpoint {
    pub fn statement(&self) -> &TransparencyCheckpointStatement { &self.statement }
    pub fn log_role_authority_sha256(&self) -> &Sha256Digest { &self.log_role_authority_sha256 }
    pub fn root_authority_sha256(&self) -> &Sha256Digest { &self.root_authority_sha256 }
    pub fn trust_snapshot_authority_sha256(&self) -> &Sha256Digest {
        &self.trust_snapshot_authority_sha256
    }
    pub fn checkpoint_sha256(&self) -> &Sha256Digest { &self.checkpoint_sha256 }
    pub const fn log_operator_authority_established(&self) -> bool { true }
    pub const fn global_log_consistency_established(&self) -> bool { false }
}

pub fn authenticate_transparency_checkpoint(
    statement: TransparencyCheckpointStatement,
    head: &TransparencyTreeHead,
    trusted_time: &TrustedTime,
    log_authority: &AuthorizedTrustRoleAttestation,
) -> Result<AuthenticatedTransparencyCheckpoint, TransparencyCheckpointAuthenticationError> {
    if statement.tree_head_sha256() != head.tree_head_sha256()
        || statement.tree_size() != head.tree_size()
        || statement.root_sha256() != head.root_sha256()
    {
        return Err(TransparencyCheckpointAuthenticationError::TreeHeadMismatch);
    }
    let time_interval = trusted_time.consensus_interval();
    if statement.trusted_time_authority_sha256() != trusted_time.authority_sha256()
        || statement.consensus_interval() != time_interval
    {
        return Err(TransparencyCheckpointAuthenticationError::TrustedTimeMismatch);
    }
    if log_authority.role() != TrustRole::TransparencyLog {
        return Err(TransparencyCheckpointAuthenticationError::WrongRole);
    }
    if log_authority.root_authority_sha256() != trusted_time.root_authority_sha256() {
        return Err(TransparencyCheckpointAuthenticationError::RootAuthorityMismatch);
    }
    if log_authority.subject_sha256() != statement.tree_head_sha256() {
        return Err(TransparencyCheckpointAuthenticationError::SubjectMismatch);
    }
    if log_authority.payload_sha256() != statement.statement_sha256() {
        return Err(TransparencyCheckpointAuthenticationError::PayloadMismatch);
    }
    if log_authority.context_sha256() != statement.previous_checkpoint_sha256() {
        return Err(TransparencyCheckpointAuthenticationError::ContextMismatch);
    }

    let checkpoint_sha256 = authenticated_checkpoint_digest(
        &statement,
        log_authority.authority_sha256(),
        log_authority.root_authority_sha256(),
        log_authority.trust_snapshot_authority_sha256(),
    );
    Ok(AuthenticatedTransparencyCheckpoint {
        statement,
        log_role_authority_sha256: log_authority.authority_sha256().clone(),
        root_authority_sha256: log_authority.root_authority_sha256().clone(),
        trust_snapshot_authority_sha256: log_authority.trust_snapshot_authority_sha256().clone(),
        checkpoint_sha256,
    })
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct TransparencyCheckpointTracker {
    latest_tree_size: Option<u64>,
    latest_root_sha256: Option<Sha256Digest>,
    latest_checkpoint_sha256: Option<Sha256Digest>,
    latest_time_earliest_unix_s: Option<u64>,
    latest_time_latest_unix_s: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransparencyCheckpointTrackingError {
    GenesisHasPredecessor,
    TreeSizeRollback { latest: u64, proposed: u64 },
    SameSizeEquivocation,
    PreviousCheckpointMismatch,
    TimeRegressed,
    MissingConsistencyProof,
    ConsistencyProofSizeMismatch,
    ConsistencyProofRootMismatch,
    InvalidConsistencyProof,
}

impl TransparencyCheckpointTracker {
    pub fn accept(
        &mut self,
        checkpoint: &AuthenticatedTransparencyCheckpoint,
        consistency_proof: Option<&TransparencyConsistencyProof>,
    ) -> Result<(), TransparencyCheckpointTrackingError> {
        let statement = checkpoint.statement();
        let (earliest, latest) = statement.consensus_interval();
        let Some(previous_size) = self.latest_tree_size else {
            if statement.previous_checkpoint_sha256().is_some() {
                return Err(TransparencyCheckpointTrackingError::GenesisHasPredecessor);
            }
            self.record(checkpoint);
            return Ok(());
        };

        let previous_root = self.latest_root_sha256.as_ref().expect("tracker size implies root");
        let previous_checkpoint = self
            .latest_checkpoint_sha256
            .as_ref()
            .expect("tracker size implies checkpoint");
        let previous_earliest = self
            .latest_time_earliest_unix_s
            .expect("tracker size implies time lower bound");
        let previous_latest = self
            .latest_time_latest_unix_s
            .expect("tracker size implies time upper bound");

        if statement.tree_size() < previous_size {
            return Err(TransparencyCheckpointTrackingError::TreeSizeRollback {
                latest: previous_size,
                proposed: statement.tree_size(),
            });
        }
        if earliest < previous_earliest || latest < previous_latest {
            return Err(TransparencyCheckpointTrackingError::TimeRegressed);
        }

        if statement.tree_size() == previous_size {
            if statement.root_sha256() != previous_root {
                return Err(TransparencyCheckpointTrackingError::SameSizeEquivocation);
            }
            if checkpoint.checkpoint_sha256() == previous_checkpoint {
                return Ok(());
            }
            if statement.previous_checkpoint_sha256() != Some(previous_checkpoint) {
                return Err(TransparencyCheckpointTrackingError::PreviousCheckpointMismatch);
            }
            self.record(checkpoint);
            return Ok(());
        }

        if statement.previous_checkpoint_sha256() != Some(previous_checkpoint) {
            return Err(TransparencyCheckpointTrackingError::PreviousCheckpointMismatch);
        }
        let Some(proof) = consistency_proof else {
            return Err(TransparencyCheckpointTrackingError::MissingConsistencyProof);
        };
        if proof.first_tree_size() != previous_size
            || proof.second_tree_size() != statement.tree_size()
        {
            return Err(TransparencyCheckpointTrackingError::ConsistencyProofSizeMismatch);
        }
        if proof.first_root_sha256() != previous_root
            || proof.second_root_sha256() != statement.root_sha256()
        {
            return Err(TransparencyCheckpointTrackingError::ConsistencyProofRootMismatch);
        }
        verify_transparency_consistency(proof)
            .map_err(|_| TransparencyCheckpointTrackingError::InvalidConsistencyProof)?;
        self.record(checkpoint);
        Ok(())
    }

    pub fn latest_checkpoint_sha256(&self) -> Option<&Sha256Digest> {
        self.latest_checkpoint_sha256.as_ref()
    }

    pub fn latest_tree_size(&self) -> Option<u64> { self.latest_tree_size }

    /// In-memory monotonic tracking is not durable currentness. The embedding
    /// system must persist/anchor tracker state and later compare witnessed views.
    pub const fn global_currentness_established(&self) -> bool { false }

    fn record(&mut self, checkpoint: &AuthenticatedTransparencyCheckpoint) {
        let statement = checkpoint.statement();
        let (earliest, latest) = statement.consensus_interval();
        self.latest_tree_size = Some(statement.tree_size());
        self.latest_root_sha256 = Some(statement.root_sha256().clone());
        self.latest_checkpoint_sha256 = Some(checkpoint.checkpoint_sha256().clone());
        self.latest_time_earliest_unix_s = Some(earliest);
        self.latest_time_latest_unix_s = Some(latest);
    }
}

fn checkpoint_statement_digest(
    tree_head_sha256: &Sha256Digest,
    tree_size: u64,
    root_sha256: &Sha256Digest,
    previous_checkpoint_sha256: Option<&Sha256Digest>,
    trusted_time_authority_sha256: &Sha256Digest,
    earliest: u64,
    latest: u64,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(CHECKPOINT_STATEMENT_DOMAIN);
    digest.text(tree_head_sha256.as_str());
    digest.text(&tree_size.to_string());
    digest.text(root_sha256.as_str());
    digest.optional_sha(previous_checkpoint_sha256);
    digest.text(trusted_time_authority_sha256.as_str());
    digest.text(&earliest.to_string());
    digest.text(&latest.to_string());
    digest.digest()
}

fn authenticated_checkpoint_digest(
    statement: &TransparencyCheckpointStatement,
    log_role_authority_sha256: &Sha256Digest,
    root_authority_sha256: &Sha256Digest,
    trust_snapshot_authority_sha256: &Sha256Digest,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(AUTHENTICATED_CHECKPOINT_DOMAIN);
    digest.text(statement.statement_sha256.as_str());
    digest.text(log_role_authority_sha256.as_str());
    digest.text(root_authority_sha256.as_str());
    digest.text(trust_snapshot_authority_sha256.as_str());
    digest.digest()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(value: &str) -> Sha256Digest { Sha256Digest::of_bytes(value.as_bytes()) }

    fn fake_checkpoint(
        tree_size: u64,
        root: &str,
        previous: Option<Sha256Digest>,
        earliest: u64,
        latest: u64,
    ) -> AuthenticatedTransparencyCheckpoint {
        let statement = TransparencyCheckpointStatement {
            tree_head_sha256: sha(&format!("head-{tree_size}-{root}")),
            tree_size,
            root_sha256: sha(root),
            previous_checkpoint_sha256: previous,
            trusted_time_authority_sha256: sha(&format!("time-{earliest}-{latest}")),
            consensus_earliest_unix_s: earliest,
            consensus_latest_unix_s: latest,
            statement_sha256: sha(&format!("statement-{tree_size}-{root}-{earliest}-{latest}")),
        };
        let checkpoint_sha256 = authenticated_checkpoint_digest(
            &statement,
            &sha("log-authority"),
            &sha("root-authority"),
            &sha("snapshot-authority"),
        );
        AuthenticatedTransparencyCheckpoint {
            statement,
            log_role_authority_sha256: sha("log-authority"),
            root_authority_sha256: sha("root-authority"),
            trust_snapshot_authority_sha256: sha("snapshot-authority"),
            checkpoint_sha256,
        }
    }

    #[test]
    fn tracker_rejects_same_size_split_view() {
        let first = fake_checkpoint(4, "root-a", None, 100, 110);
        let mut tracker = TransparencyCheckpointTracker::default();
        tracker.accept(&first, None).unwrap();
        let second = fake_checkpoint(
            4,
            "root-b",
            Some(first.checkpoint_sha256().clone()),
            111,
            120,
        );
        assert_eq!(
            tracker.accept(&second, None),
            Err(TransparencyCheckpointTrackingError::SameSizeEquivocation)
        );
    }

    #[test]
    fn same_size_refresh_requires_explicit_chain() {
        let first = fake_checkpoint(4, "root-a", None, 100, 110);
        let mut tracker = TransparencyCheckpointTracker::default();
        tracker.accept(&first, None).unwrap();
        let refresh = fake_checkpoint(4, "root-a", None, 111, 120);
        assert_eq!(
            tracker.accept(&refresh, None),
            Err(TransparencyCheckpointTrackingError::PreviousCheckpointMismatch)
        );
    }

    #[test]
    fn checkpoint_time_cannot_regress() {
        let first = fake_checkpoint(4, "root-a", None, 100, 110);
        let mut tracker = TransparencyCheckpointTracker::default();
        tracker.accept(&first, None).unwrap();
        let regressed = fake_checkpoint(
            4,
            "root-a",
            Some(first.checkpoint_sha256().clone()),
            99,
            120,
        );
        assert_eq!(
            tracker.accept(&regressed, None),
            Err(TransparencyCheckpointTrackingError::TimeRegressed)
        );
    }
}
