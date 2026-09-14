// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pure anti-rollback checkpoint-chain mechanics for EUREKA-002 V2 qualifier
//! governance state.
//!
//! This module binds active qualifier-authority currentness and signer-policy
//! state into one monotonic append-only checkpoint chain. It deliberately does
//! not implement durable storage/fsync, admit a real root, or grant execution
//! authority. A later persistence tranche must durably store these snapshots
//! before they can serve as a real high-water mark.

#![allow(dead_code)]

use super::v2_qualifier_authority::V2QualifierAuthorityRecord;
use super::v2_qualifier_currentness::{
    V2QualifierAuthorityLineage, V2QualifierCurrentnessError,
};
use super::v2_qualifier_signer_policy::V2QualifierSignerPolicy;

pub(super) const V2_QUALIFIER_ANTI_ROLLBACK_CHECKPOINT_SCHEMA: &str =
    "EUREKA.002.V2.QUALIFIER_ANTI_ROLLBACK_CHECKPOINT.v1";

/// A serializable logical high-water checkpoint.
///
/// It is not itself proof that these bytes were ever durably persisted. The
/// persistence theorem is deliberately separate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct V2QualifierCheckpointSnapshot {
    checkpoint_sequence: u64,
    predecessor_checkpoint_commitment: Option<[u8; 32]>,
    genesis_authority_commitment: [u8; 32],
    authority_sequence: u64,
    authority_commitment: [u8; 32],
    currentness_commitment: [u8; 32],
    signer_policy_sequence: u64,
    signer_policy_commitment: [u8; 32],
    commitment: [u8; 32],
}

impl V2QualifierCheckpointSnapshot {
    pub(super) const fn checkpoint_sequence(&self) -> u64 {
        self.checkpoint_sequence
    }

    pub(super) const fn predecessor_checkpoint_commitment(&self) -> Option<[u8; 32]> {
        self.predecessor_checkpoint_commitment
    }

    pub(super) const fn genesis_authority_commitment(&self) -> [u8; 32] {
        self.genesis_authority_commitment
    }

    pub(super) const fn authority_sequence(&self) -> u64 {
        self.authority_sequence
    }

    pub(super) const fn authority_commitment(&self) -> [u8; 32] {
        self.authority_commitment
    }

    pub(super) const fn currentness_commitment(&self) -> [u8; 32] {
        self.currentness_commitment
    }

    pub(super) const fn signer_policy_sequence(&self) -> u64 {
        self.signer_policy_sequence
    }

    pub(super) const fn signer_policy_commitment(&self) -> [u8; 32] {
        self.signer_policy_commitment
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }

    pub(super) fn verify_internal_commitment(&self) -> bool {
        checkpoint_commitment(
            self.checkpoint_sequence,
            self.predecessor_checkpoint_commitment,
            self.genesis_authority_commitment,
            self.authority_sequence,
            self.authority_commitment,
            self.currentness_commitment,
            self.signer_policy_sequence,
            self.signer_policy_commitment,
        ) == self.commitment
    }
}

/// Move-only logical checkpoint chain.
///
/// Exact authority continuation is delegated to `V2QualifierAuthorityLineage`;
/// exact signer-policy continuation is checked against the current policy.
/// Therefore a caller cannot advance this chain merely by presenting a larger
/// sequence number or a sibling fork.
#[derive(Debug)]
pub(super) struct V2QualifierCheckpointChain {
    lineage: V2QualifierAuthorityLineage,
    signer_policy: V2QualifierSignerPolicy,
    checkpoint_sequence: u64,
    predecessor_checkpoint_commitment: Option<[u8; 32]>,
    commitment: [u8; 32],
}

impl V2QualifierCheckpointChain {
    /// Construct mechanical checkpoint genesis only.
    ///
    /// This does not admit the supplied qualifier/signer roots as trusted. A
    /// future ceremony must supply already-admitted roots before using the same
    /// checkpoint mechanics as a durable security boundary.
    pub(super) fn mechanical_genesis(
        lineage: V2QualifierAuthorityLineage,
        signer_policy: V2QualifierSignerPolicy,
    ) -> Result<Self, V2QualifierAntiRollbackError> {
        if lineage.current_sequence() != 1
            || lineage.genesis_commitment() != lineage.current_authority_commitment()
            || signer_policy.sequence() != 1
            || signer_policy.predecessor_commitment().is_some()
        {
            return Err(V2QualifierAntiRollbackError::NotGenesis);
        }

        let checkpoint_sequence = 1;
        let commitment = checkpoint_commitment(
            checkpoint_sequence,
            None,
            lineage.genesis_commitment(),
            lineage.current_sequence(),
            lineage.current_authority_commitment(),
            lineage.commitment(),
            signer_policy.sequence(),
            signer_policy.commitment(),
        );
        Ok(Self {
            lineage,
            signer_policy,
            checkpoint_sequence,
            predecessor_checkpoint_commitment: None,
            commitment,
        })
    }

    /// Advance exactly one authority/currentness transition while preserving
    /// signer policy. Sibling/stale authority successors are rejected by the
    /// underlying active-lineage state.
    pub(super) fn advance_authority(
        self,
        successor: V2QualifierAuthorityRecord,
    ) -> Result<Self, V2QualifierAntiRollbackError> {
        let next_checkpoint_sequence = self
            .checkpoint_sequence
            .checked_add(1)
            .ok_or(V2QualifierAntiRollbackError::CheckpointSequenceExhausted)?;
        let previous_checkpoint_commitment = self.commitment;
        let Self {
            lineage,
            signer_policy,
            ..
        } = self;
        let lineage = lineage
            .advance(successor)
            .map_err(V2QualifierAntiRollbackError::AuthorityTransition)?;

        let commitment = checkpoint_commitment(
            next_checkpoint_sequence,
            Some(previous_checkpoint_commitment),
            lineage.genesis_commitment(),
            lineage.current_sequence(),
            lineage.current_authority_commitment(),
            lineage.commitment(),
            signer_policy.sequence(),
            signer_policy.commitment(),
        );
        Ok(Self {
            lineage,
            signer_policy,
            checkpoint_sequence: next_checkpoint_sequence,
            predecessor_checkpoint_commitment: Some(previous_checkpoint_commitment),
            commitment,
        })
    }

    /// Advance exactly one signer-policy transition while preserving authority
    /// currentness. A successor from an older policy or with a skipped sequence
    /// cannot become the next checkpoint.
    pub(super) fn advance_signer_policy(
        self,
        successor: V2QualifierSignerPolicy,
    ) -> Result<Self, V2QualifierAntiRollbackError> {
        let next_checkpoint_sequence = self
            .checkpoint_sequence
            .checked_add(1)
            .ok_or(V2QualifierAntiRollbackError::CheckpointSequenceExhausted)?;
        if successor.predecessor_commitment() != Some(self.signer_policy.commitment()) {
            return Err(V2QualifierAntiRollbackError::StaleSignerPolicy);
        }
        let expected_signer_sequence = self
            .signer_policy
            .sequence()
            .checked_add(1)
            .ok_or(V2QualifierAntiRollbackError::SignerPolicySequenceExhausted)?;
        if successor.sequence() != expected_signer_sequence {
            return Err(V2QualifierAntiRollbackError::WrongSignerPolicySequence);
        }

        let previous_checkpoint_commitment = self.commitment;
        let Self { lineage, .. } = self;
        let commitment = checkpoint_commitment(
            next_checkpoint_sequence,
            Some(previous_checkpoint_commitment),
            lineage.genesis_commitment(),
            lineage.current_sequence(),
            lineage.current_authority_commitment(),
            lineage.commitment(),
            successor.sequence(),
            successor.commitment(),
        );
        Ok(Self {
            lineage,
            signer_policy: successor,
            checkpoint_sequence: next_checkpoint_sequence,
            predecessor_checkpoint_commitment: Some(previous_checkpoint_commitment),
            commitment,
        })
    }

    pub(super) fn snapshot(&self) -> V2QualifierCheckpointSnapshot {
        V2QualifierCheckpointSnapshot {
            checkpoint_sequence: self.checkpoint_sequence,
            predecessor_checkpoint_commitment: self.predecessor_checkpoint_commitment,
            genesis_authority_commitment: self.lineage.genesis_commitment(),
            authority_sequence: self.lineage.current_sequence(),
            authority_commitment: self.lineage.current_authority_commitment(),
            currentness_commitment: self.lineage.commitment(),
            signer_policy_sequence: self.signer_policy.sequence(),
            signer_policy_commitment: self.signer_policy.commitment(),
            commitment: self.commitment,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2QualifierRecoveryDisposition {
    ExactHighWater,
}

/// Compare a recovered snapshot against an independently durable high-water
/// snapshot.
///
/// Higher sequence is *not* accepted merely for being larger. The missing
/// transitions must be replayed/verified through the chain mechanics first.
pub(super) fn classify_recovery_snapshot(
    high_water: &V2QualifierCheckpointSnapshot,
    candidate: &V2QualifierCheckpointSnapshot,
) -> Result<V2QualifierRecoveryDisposition, V2QualifierAntiRollbackError> {
    if !high_water.verify_internal_commitment() || !candidate.verify_internal_commitment() {
        return Err(V2QualifierAntiRollbackError::CorruptCheckpoint);
    }
    if candidate.genesis_authority_commitment != high_water.genesis_authority_commitment {
        return Err(V2QualifierAntiRollbackError::WrongGenesisAnchor);
    }
    if candidate.checkpoint_sequence < high_water.checkpoint_sequence {
        return Err(V2QualifierAntiRollbackError::RollbackDetected);
    }
    if candidate.checkpoint_sequence == high_water.checkpoint_sequence {
        if candidate.commitment == high_water.commitment {
            return Ok(V2QualifierRecoveryDisposition::ExactHighWater);
        }
        return Err(V2QualifierAntiRollbackError::EquivocationDetected);
    }
    Err(V2QualifierAntiRollbackError::UnprovenFutureState)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2QualifierAntiRollbackError {
    NotGenesis,
    CheckpointSequenceExhausted,
    SignerPolicySequenceExhausted,
    StaleSignerPolicy,
    WrongSignerPolicySequence,
    AuthorityTransition(V2QualifierCurrentnessError),
    CorruptCheckpoint,
    WrongGenesisAnchor,
    RollbackDetected,
    EquivocationDetected,
    UnprovenFutureState,
}

#[allow(clippy::too_many_arguments)]
fn checkpoint_commitment(
    checkpoint_sequence: u64,
    predecessor_checkpoint_commitment: Option<[u8; 32]>,
    genesis_authority_commitment: [u8; 32],
    authority_sequence: u64,
    authority_commitment: [u8; 32],
    currentness_commitment: [u8; 32],
    signer_policy_sequence: u64,
    signer_policy_commitment: [u8; 32],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        V2_QUALIFIER_ANTI_ROLLBACK_CHECKPOINT_SCHEMA.as_bytes(),
    );
    bytes.extend_from_slice(&checkpoint_sequence.to_le_bytes());
    match predecessor_checkpoint_commitment {
        Some(predecessor) => {
            bytes.push(1);
            bytes.extend_from_slice(&predecessor);
        }
        None => bytes.push(0),
    }
    bytes.extend_from_slice(&genesis_authority_commitment);
    bytes.extend_from_slice(&authority_sequence.to_le_bytes());
    bytes.extend_from_slice(&authority_commitment);
    bytes.extend_from_slice(&currentness_commitment);
    bytes.extend_from_slice(&signer_policy_sequence.to_le_bytes());
    bytes.extend_from_slice(&signer_policy_commitment);
    *blake3::hash(&bytes).as_bytes()
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarks::eureka::v2_qualifier_authority::{
        V2QualifierAuthorityProfile, V2QualifierAuthorityRecord,
    };
    use crate::benchmarks::eureka::v2_qualifier_currentness::V2QualifierAuthorityLineage;
    use crate::benchmarks::eureka::v2_qualifier_signer_policy::{
        V2GovernanceSigner, V2QualifierSignerPolicy,
    };

    fn authority_profile(workflow: char, contract: char) -> V2QualifierAuthorityProfile {
        V2QualifierAuthorityProfile::current_from_hex(
            &workflow.to_string().repeat(64),
            &contract.to_string().repeat(64),
        )
        .unwrap()
    }

    fn signer(principal: &str, key: char) -> V2GovernanceSigner {
        V2GovernanceSigner::from_hex(
            principal,
            "ssh-ed25519",
            &key.to_string().repeat(64),
        )
        .unwrap()
    }

    fn roots() -> (
        V2QualifierAuthorityRecord,
        V2QualifierAuthorityLineage,
        V2QualifierSignerPolicy,
    ) {
        let record = V2QualifierAuthorityRecord::genesis(1, authority_profile('b', 'c')).unwrap();
        let lineage = V2QualifierAuthorityLineage::activate_genesis(record.clone()).unwrap();
        let policy = V2QualifierSignerPolicy::genesis(
            1,
            1,
            vec![signer("root@example.invalid", 'a')],
        )
        .unwrap();
        (record, lineage, policy)
    }

    #[test]
    fn mechanical_genesis_is_deterministic_and_binds_both_security_domains() {
        let (_, lineage_a, policy_a) = roots();
        let (_, lineage_b, policy_b) = roots();
        let first = V2QualifierCheckpointChain::mechanical_genesis(lineage_a, policy_a).unwrap();
        let second = V2QualifierCheckpointChain::mechanical_genesis(lineage_b, policy_b).unwrap();
        let snapshot = first.snapshot();

        assert_eq!(snapshot.checkpoint_sequence(), 1);
        assert_eq!(snapshot.authority_sequence(), 1);
        assert_eq!(snapshot.signer_policy_sequence(), 1);
        assert_eq!(snapshot.predecessor_checkpoint_commitment(), None);
        assert!(snapshot.verify_internal_commitment());
        assert_eq!(snapshot.commitment(), second.snapshot().commitment());
        assert_ne!(snapshot.commitment(), [0_u8; 32]);
    }

    #[test]
    fn authority_transition_advances_checkpoint_and_rejects_stale_sibling() {
        let (genesis, lineage, policy) = roots();
        let left = V2QualifierAuthorityRecord::rotate(
            &genesis,
            genesis.commitment(),
            2,
            authority_profile('d', 'c'),
        )
        .unwrap();
        let right = V2QualifierAuthorityRecord::rotate(
            &genesis,
            genesis.commitment(),
            2,
            authority_profile('b', 'd'),
        )
        .unwrap();

        let chain = V2QualifierCheckpointChain::mechanical_genesis(lineage, policy).unwrap();
        let genesis_checkpoint = chain.snapshot().commitment();
        let chain = chain.advance_authority(left).unwrap();
        let snapshot = chain.snapshot();
        assert_eq!(snapshot.checkpoint_sequence(), 2);
        assert_eq!(snapshot.authority_sequence(), 2);
        assert_eq!(snapshot.signer_policy_sequence(), 1);
        assert_eq!(
            snapshot.predecessor_checkpoint_commitment(),
            Some(genesis_checkpoint)
        );
        assert!(snapshot.verify_internal_commitment());
        assert_eq!(
            chain.advance_authority(right).unwrap_err(),
            V2QualifierAntiRollbackError::AuthorityTransition(
                V2QualifierCurrentnessError::StalePredecessor
            )
        );
    }

    #[test]
    fn signer_policy_transition_advances_checkpoint_and_rejects_stale_sibling() {
        let (_, lineage, policy) = roots();
        let left = V2QualifierSignerPolicy::rotate(
            &policy,
            policy.commitment(),
            2,
            1,
            vec![signer("left@example.invalid", 'b')],
        )
        .unwrap();
        let right = V2QualifierSignerPolicy::rotate(
            &policy,
            policy.commitment(),
            2,
            1,
            vec![signer("right@example.invalid", 'c')],
        )
        .unwrap();

        let chain = V2QualifierCheckpointChain::mechanical_genesis(lineage, policy).unwrap();
        let genesis_checkpoint = chain.snapshot().commitment();
        let chain = chain.advance_signer_policy(left).unwrap();
        let snapshot = chain.snapshot();
        assert_eq!(snapshot.checkpoint_sequence(), 2);
        assert_eq!(snapshot.authority_sequence(), 1);
        assert_eq!(snapshot.signer_policy_sequence(), 2);
        assert_eq!(
            snapshot.predecessor_checkpoint_commitment(),
            Some(genesis_checkpoint)
        );
        assert_eq!(
            chain.advance_signer_policy(right).unwrap_err(),
            V2QualifierAntiRollbackError::StaleSignerPolicy
        );
    }

    #[test]
    fn recovery_rejects_rollback_equivocation_corruption_and_unproven_future() {
        let (genesis, lineage, policy) = roots();
        let authority_successor = V2QualifierAuthorityRecord::rotate(
            &genesis,
            genesis.commitment(),
            2,
            authority_profile('d', 'c'),
        )
        .unwrap();
        let signer_successor = V2QualifierSignerPolicy::rotate(
            &policy,
            policy.commitment(),
            2,
            1,
            vec![signer("next@example.invalid", 'b')],
        )
        .unwrap();

        let genesis_chain = V2QualifierCheckpointChain::mechanical_genesis(lineage, policy).unwrap();
        let old = genesis_chain.snapshot();
        let authority_chain = genesis_chain.advance_authority(authority_successor).unwrap();
        let high_water = authority_chain.snapshot();
        let future = authority_chain
            .advance_signer_policy(signer_successor)
            .unwrap()
            .snapshot();

        assert_eq!(
            classify_recovery_snapshot(&high_water, &high_water),
            Ok(V2QualifierRecoveryDisposition::ExactHighWater)
        );
        assert_eq!(
            classify_recovery_snapshot(&high_water, &old).unwrap_err(),
            V2QualifierAntiRollbackError::RollbackDetected
        );
        assert_eq!(
            classify_recovery_snapshot(&high_water, &future).unwrap_err(),
            V2QualifierAntiRollbackError::UnprovenFutureState
        );

        let mut equivocation = high_water;
        equivocation.authority_commitment = [9_u8; 32];
        equivocation.commitment = checkpoint_commitment(
            equivocation.checkpoint_sequence,
            equivocation.predecessor_checkpoint_commitment,
            equivocation.genesis_authority_commitment,
            equivocation.authority_sequence,
            equivocation.authority_commitment,
            equivocation.currentness_commitment,
            equivocation.signer_policy_sequence,
            equivocation.signer_policy_commitment,
        );
        assert_eq!(
            classify_recovery_snapshot(&high_water, &equivocation).unwrap_err(),
            V2QualifierAntiRollbackError::EquivocationDetected
        );

        let mut corrupt = high_water;
        corrupt.commitment[0] ^= 1;
        assert_eq!(
            classify_recovery_snapshot(&high_water, &corrupt).unwrap_err(),
            V2QualifierAntiRollbackError::CorruptCheckpoint
        );
    }

    #[test]
    fn recovery_rejects_different_genesis_even_if_checkpoint_sequence_matches() {
        let (_, lineage, policy) = roots();
        let high_water = V2QualifierCheckpointChain::mechanical_genesis(lineage, policy)
            .unwrap()
            .snapshot();
        let mut alien = high_water;
        alien.genesis_authority_commitment = [7_u8; 32];
        alien.commitment = checkpoint_commitment(
            alien.checkpoint_sequence,
            alien.predecessor_checkpoint_commitment,
            alien.genesis_authority_commitment,
            alien.authority_sequence,
            alien.authority_commitment,
            alien.currentness_commitment,
            alien.signer_policy_sequence,
            alien.signer_policy_commitment,
        );
        assert_eq!(
            classify_recovery_snapshot(&high_water, &alien).unwrap_err(),
            V2QualifierAntiRollbackError::WrongGenesisAnchor
        );
    }

    #[test]
    fn checkpoint_sequence_exhaustion_fails_closed() {
        let (genesis, lineage, policy) = roots();
        let successor = V2QualifierAuthorityRecord::rotate(
            &genesis,
            genesis.commitment(),
            2,
            authority_profile('d', 'c'),
        )
        .unwrap();
        let mut chain = V2QualifierCheckpointChain::mechanical_genesis(lineage, policy).unwrap();
        chain.checkpoint_sequence = u64::MAX;
        assert_eq!(
            chain.advance_authority(successor).unwrap_err(),
            V2QualifierAntiRollbackError::CheckpointSequenceExhausted
        );
    }

    #[test]
    fn source_has_no_persistence_claim_real_root_or_execution_surface() {
        let production = include_str!("v2_qualifier_anti_rollback.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for forbidden in [
            "std::fs",
            "File::create",
            "fsync",
            "V2AdmittedQualifierRoot",
            "V2CanaryAuthorization",
            "predict_ticket",
            "reveal(",
            "score_consequence",
        ] {
            assert!(
                !production.contains(forbidden),
                "forbidden anti-rollback surface: {forbidden}"
            );
        }
        assert!(production.contains("UnprovenFutureState"));
        assert!(production.contains("EquivocationDetected"));
    }
}
