// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pure anti-rollback checkpoint mechanics for EUREKA-002 V2 qualifier
//! authority.
//!
//! Authentic signatures and valid predecessor chains do not by themselves prove
//! that a presented authority state is the newest state a verifier has already
//! accepted. This module defines a canonical high-water checkpoint and forces
//! authority/policy advancement through the exact state already represented by
//! that checkpoint.
//!
//! This is test-only mechanics. It does not provide durable filesystem storage,
//! admit a real governance root, or grant execution authority. A real ceremony
//! must persist `persisted_bytes()` atomically outside mutable repository state
//! and bind the resulting durable high-water mark into permit mint/execution.
//!
//! The checkpoint commitment is an integrity identifier, not authorization.
//! Parsing a self-consistent checkpoint can detect corruption and compare
//! relative height, but an externally presented successor becomes semantically
//! current only after replay through `advance_authority` or
//! `advance_signer_policy` (plus the future signed-admission theorem).

#![allow(dead_code)]

use super::v2_qualifier_authority::V2QualifierAuthorityRecord;
use super::v2_qualifier_currentness::{
    V2QualifierAuthorityLineage, V2QualifierCurrentnessError,
};
use super::v2_qualifier_signer_policy::V2QualifierSignerPolicy;

pub(super) const V2_QUALIFIER_ANTI_ROLLBACK_CHECKPOINT_SCHEMA: &str =
    "EUREKA.002.V2.QUALIFIER_ANTI_ROLLBACK_CHECKPOINT.v1";
pub(super) const V2_QUALIFIER_ANTI_ROLLBACK_CHECKPOINT_COMMITMENT_REVISION: &str =
    "EUREKA.002.V2.QUALIFIER_ANTI_ROLLBACK_CHECKPOINT_COMMITMENT.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2QualifierAntiRollbackCheckpoint {
    genesis_authority_commitment: [u8; 32],
    authority_sequence: u64,
    authority_commitment: [u8; 32],
    currentness_commitment: [u8; 32],
    signer_policy_sequence: u64,
    signer_policy_commitment: [u8; 32],
    predecessor_checkpoint_commitment: Option<[u8; 32]>,
    commitment: [u8; 32],
}

impl V2QualifierAntiRollbackCheckpoint {
    fn genesis(
        lineage: &V2QualifierAuthorityLineage,
        signer_policy: &V2QualifierSignerPolicy,
    ) -> Result<Self, V2QualifierAntiRollbackError> {
        if lineage.current_sequence() != 1 || signer_policy.sequence() != 1 {
            return Err(V2QualifierAntiRollbackError::NotGenesisState);
        }
        Ok(Self::from_current(lineage, signer_policy, None))
    }

    fn successor(
        lineage: &V2QualifierAuthorityLineage,
        signer_policy: &V2QualifierSignerPolicy,
        predecessor_checkpoint_commitment: [u8; 32],
    ) -> Self {
        Self::from_current(
            lineage,
            signer_policy,
            Some(predecessor_checkpoint_commitment),
        )
    }

    fn from_current(
        lineage: &V2QualifierAuthorityLineage,
        signer_policy: &V2QualifierSignerPolicy,
        predecessor_checkpoint_commitment: Option<[u8; 32]>,
    ) -> Self {
        let mut checkpoint = Self {
            genesis_authority_commitment: lineage.genesis_commitment(),
            authority_sequence: lineage.current_sequence(),
            authority_commitment: lineage.current_authority_commitment(),
            currentness_commitment: lineage.commitment(),
            signer_policy_sequence: signer_policy.sequence(),
            signer_policy_commitment: signer_policy.commitment(),
            predecessor_checkpoint_commitment,
            commitment: [0_u8; 32],
        };
        checkpoint.commitment = checkpoint_commitment(&checkpoint);
        checkpoint
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

    pub(super) const fn predecessor_checkpoint_commitment(&self) -> Option<[u8; 32]> {
        self.predecessor_checkpoint_commitment
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }

    /// Canonical durable representation. The final field self-authenticates all
    /// preceding fields through a domain-separated BLAKE3 commitment.
    pub(super) fn persisted_bytes(&self) -> Vec<u8> {
        let mut out = String::from_utf8(self.canonical_body_bytes())
            .expect("checkpoint canonical body is ASCII");
        push_field(&mut out, "checkpoint_commitment", &hex32(self.commitment));
        out.into_bytes()
    }

    fn canonical_body_bytes(&self) -> Vec<u8> {
        let mut out = String::new();
        push_field(
            &mut out,
            "checkpoint_schema_revision",
            V2_QUALIFIER_ANTI_ROLLBACK_CHECKPOINT_SCHEMA,
        );
        push_field(
            &mut out,
            "genesis_authority_commitment",
            &hex32(self.genesis_authority_commitment),
        );
        push_field(
            &mut out,
            "authority_sequence",
            &self.authority_sequence.to_string(),
        );
        push_field(
            &mut out,
            "authority_commitment",
            &hex32(self.authority_commitment),
        );
        push_field(
            &mut out,
            "currentness_commitment",
            &hex32(self.currentness_commitment),
        );
        push_field(
            &mut out,
            "signer_policy_sequence",
            &self.signer_policy_sequence.to_string(),
        );
        push_field(
            &mut out,
            "signer_policy_commitment",
            &hex32(self.signer_policy_commitment),
        );
        push_field(
            &mut out,
            "predecessor_checkpoint_commitment",
            &optional_hex(self.predecessor_checkpoint_commitment),
        );
        out.into_bytes()
    }

    pub(super) fn parse_persisted(
        input: &[u8],
    ) -> Result<Self, V2QualifierAntiRollbackError> {
        let text = std::str::from_utf8(input)
            .map_err(|_| V2QualifierAntiRollbackError::InvalidUtf8)?;
        if !text.ends_with('\n') {
            return Err(V2QualifierAntiRollbackError::MissingTrailingNewline);
        }
        let body = &text[..text.len() - 1];
        let lines: Vec<&str> = body.split('\n').collect();
        if lines.len() != 9 {
            return Err(V2QualifierAntiRollbackError::WrongFieldCount);
        }

        if field(lines[0], "checkpoint_schema_revision")?
            != V2_QUALIFIER_ANTI_ROLLBACK_CHECKPOINT_SCHEMA
        {
            return Err(V2QualifierAntiRollbackError::UnsupportedSchema);
        }
        let genesis_authority_commitment =
            decode_hex32(field(lines[1], "genesis_authority_commitment")?)?;
        let authority_sequence = canonical_u64(field(lines[2], "authority_sequence")?)?;
        let authority_commitment = decode_hex32(field(lines[3], "authority_commitment")?)?;
        let currentness_commitment = decode_hex32(field(lines[4], "currentness_commitment")?)?;
        let signer_policy_sequence =
            canonical_u64(field(lines[5], "signer_policy_sequence")?)?;
        let signer_policy_commitment =
            decode_hex32(field(lines[6], "signer_policy_commitment")?)?;
        let predecessor_checkpoint_commitment = optional_commitment(field(
            lines[7],
            "predecessor_checkpoint_commitment",
        )?)?;
        let claimed_commitment = decode_hex32(field(lines[8], "checkpoint_commitment")?)?;

        if authority_sequence == 0 || signer_policy_sequence == 0 {
            return Err(V2QualifierAntiRollbackError::InvalidCheckpointShape);
        }
        let is_genesis = authority_sequence == 1 && signer_policy_sequence == 1;
        if is_genesis != predecessor_checkpoint_commitment.is_none()
            || (authority_sequence == 1
                && authority_commitment != genesis_authority_commitment)
        {
            return Err(V2QualifierAntiRollbackError::InvalidCheckpointShape);
        }

        let mut checkpoint = Self {
            genesis_authority_commitment,
            authority_sequence,
            authority_commitment,
            currentness_commitment,
            signer_policy_sequence,
            signer_policy_commitment,
            predecessor_checkpoint_commitment,
            commitment: [0_u8; 32],
        };
        checkpoint.commitment = checkpoint_commitment(&checkpoint);
        if checkpoint.commitment != claimed_commitment {
            return Err(V2QualifierAntiRollbackError::CommitmentMismatch);
        }
        if checkpoint.persisted_bytes() != input {
            return Err(V2QualifierAntiRollbackError::NonCanonicalEncoding);
        }
        Ok(checkpoint)
    }
}

/// Move-only high-water guard. It does not become durable until the caller
/// atomically persists `checkpoint().persisted_bytes()` outside mutable Git
/// state. Keeping this type move-only makes state transitions explicit and
/// prevents an old in-memory guard from being silently reused after advance.
#[derive(Debug)]
pub(super) struct V2QualifierAntiRollbackGuard {
    checkpoint: V2QualifierAntiRollbackCheckpoint,
}

impl V2QualifierAntiRollbackGuard {
    pub(super) fn activate_genesis(
        lineage: &V2QualifierAuthorityLineage,
        signer_policy: &V2QualifierSignerPolicy,
    ) -> Result<Self, V2QualifierAntiRollbackError> {
        Ok(Self {
            checkpoint: V2QualifierAntiRollbackCheckpoint::genesis(lineage, signer_policy)?,
        })
    }

    pub(super) fn recover_exact(
        persisted: &[u8],
        lineage: &V2QualifierAuthorityLineage,
        signer_policy: &V2QualifierSignerPolicy,
    ) -> Result<Self, V2QualifierAntiRollbackError> {
        let checkpoint = V2QualifierAntiRollbackCheckpoint::parse_persisted(persisted)?;
        let guard = Self { checkpoint };
        guard.verify_current(lineage, signer_policy)?;
        Ok(guard)
    }

    pub(super) fn checkpoint(&self) -> &V2QualifierAntiRollbackCheckpoint {
        &self.checkpoint
    }

    pub(super) fn verify_current(
        &self,
        lineage: &V2QualifierAuthorityLineage,
        signer_policy: &V2QualifierSignerPolicy,
    ) -> Result<(), V2QualifierAntiRollbackError> {
        let checkpoint = &self.checkpoint;
        if checkpoint.genesis_authority_commitment != lineage.genesis_commitment()
            || checkpoint.authority_sequence != lineage.current_sequence()
            || checkpoint.authority_commitment != lineage.current_authority_commitment()
            || checkpoint.currentness_commitment != lineage.commitment()
            || checkpoint.signer_policy_sequence != signer_policy.sequence()
            || checkpoint.signer_policy_commitment != signer_policy.commitment()
        {
            return Err(V2QualifierAntiRollbackError::CurrentStateMismatch);
        }
        Ok(())
    }

    /// Advance authority only by consuming the exact lineage represented by the
    /// high-water state and invoking that lineage's own predecessor-checked
    /// transition. A forked successor cannot be accepted from a sequence number
    /// alone.
    pub(super) fn advance_authority(
        self,
        lineage: V2QualifierAuthorityLineage,
        successor_authority: V2QualifierAuthorityRecord,
        signer_policy: &V2QualifierSignerPolicy,
    ) -> Result<(Self, V2QualifierAuthorityLineage), V2QualifierAntiRollbackError> {
        self.verify_current(&lineage, signer_policy)?;
        let predecessor_checkpoint = self.checkpoint.commitment;
        let next_lineage = lineage
            .advance(successor_authority)
            .map_err(V2QualifierAntiRollbackError::AuthorityTransition)?;
        let next_checkpoint = V2QualifierAntiRollbackCheckpoint::successor(
            &next_lineage,
            signer_policy,
            predecessor_checkpoint,
        );
        Ok((
            Self {
                checkpoint: next_checkpoint,
            },
            next_lineage,
        ))
    }

    /// Advance signer policy only from the exact policy represented by this
    /// checkpoint. Signed governance authorization remains a separate theorem;
    /// this method only preserves predecessor/currentness mechanics.
    pub(super) fn advance_signer_policy(
        self,
        lineage: &V2QualifierAuthorityLineage,
        current_policy: &V2QualifierSignerPolicy,
        successor_policy: V2QualifierSignerPolicy,
    ) -> Result<(Self, V2QualifierSignerPolicy), V2QualifierAntiRollbackError> {
        self.verify_current(lineage, current_policy)?;
        let expected_sequence = current_policy
            .sequence()
            .checked_add(1)
            .ok_or(V2QualifierAntiRollbackError::SequenceExhausted)?;
        if successor_policy.sequence() != expected_sequence {
            return Err(V2QualifierAntiRollbackError::SignerPolicyWrongSequence);
        }
        if successor_policy.predecessor_commitment() != Some(current_policy.commitment()) {
            return Err(V2QualifierAntiRollbackError::SignerPolicyWrongPredecessor);
        }
        if successor_policy.commitment() == current_policy.commitment() {
            return Err(V2QualifierAntiRollbackError::SignerPolicyNoChange);
        }

        let predecessor_checkpoint = self.checkpoint.commitment;
        let next_checkpoint = V2QualifierAntiRollbackCheckpoint::successor(
            lineage,
            &successor_policy,
            predecessor_checkpoint,
        );
        Ok((
            Self {
                checkpoint: next_checkpoint,
            },
            successor_policy,
        ))
    }

    /// Classify a parsed candidate relative to the durable high-water mark.
    ///
    /// `StructurallyImmediateSuccessor` is intentionally *not* an authorization
    /// result. An unkeyed self-consistent checkpoint can be manufactured by an
    /// attacker. Semantic activation of that candidate still requires replay of
    /// the corresponding authority/policy transition through this guard and the
    /// future signed-admission checks.
    pub(super) fn classify_candidate(
        &self,
        candidate: &V2QualifierAntiRollbackCheckpoint,
    ) -> Result<V2CheckpointCandidateDisposition, V2QualifierAntiRollbackError> {
        let durable = &self.checkpoint;
        if candidate.genesis_authority_commitment != durable.genesis_authority_commitment {
            return Err(V2QualifierAntiRollbackError::WrongGenesis);
        }
        if candidate == durable {
            return Ok(V2CheckpointCandidateDisposition::ExactCurrent);
        }
        if candidate.authority_sequence < durable.authority_sequence
            || candidate.signer_policy_sequence < durable.signer_policy_sequence
        {
            return Err(V2QualifierAntiRollbackError::RollbackDetected);
        }

        let authority_delta = candidate
            .authority_sequence
            .checked_sub(durable.authority_sequence)
            .ok_or(V2QualifierAntiRollbackError::RollbackDetected)?;
        let policy_delta = candidate
            .signer_policy_sequence
            .checked_sub(durable.signer_policy_sequence)
            .ok_or(V2QualifierAntiRollbackError::RollbackDetected)?;

        if authority_delta == 0 && policy_delta == 0 {
            return Err(V2QualifierAntiRollbackError::EquivocationDetected);
        }
        if candidate.predecessor_checkpoint_commitment != Some(durable.commitment)
            || !matches!((authority_delta, policy_delta), (1, 0) | (0, 1))
        {
            return Err(V2QualifierAntiRollbackError::SkippedTransition);
        }

        match (authority_delta, policy_delta) {
            (1, 0) => {
                if candidate.signer_policy_commitment != durable.signer_policy_commitment
                    || candidate.authority_commitment == durable.authority_commitment
                    || candidate.currentness_commitment == durable.currentness_commitment
                {
                    return Err(V2QualifierAntiRollbackError::InvalidSuccessorShape);
                }
            }
            (0, 1) => {
                if candidate.authority_commitment != durable.authority_commitment
                    || candidate.currentness_commitment != durable.currentness_commitment
                    || candidate.signer_policy_commitment == durable.signer_policy_commitment
                {
                    return Err(V2QualifierAntiRollbackError::InvalidSuccessorShape);
                }
            }
            _ => return Err(V2QualifierAntiRollbackError::SkippedTransition),
        }

        Ok(V2CheckpointCandidateDisposition::StructurallyImmediateSuccessor)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2CheckpointCandidateDisposition {
    ExactCurrent,
    StructurallyImmediateSuccessor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2QualifierAntiRollbackError {
    NotGenesisState,
    CurrentStateMismatch,
    AuthorityTransition(V2QualifierCurrentnessError),
    SequenceExhausted,
    SignerPolicyWrongSequence,
    SignerPolicyWrongPredecessor,
    SignerPolicyNoChange,
    InvalidUtf8,
    MissingTrailingNewline,
    WrongFieldCount,
    WrongField(&'static str),
    UnsupportedSchema,
    InvalidNumber,
    InvalidHex,
    InvalidCheckpointShape,
    CommitmentMismatch,
    NonCanonicalEncoding,
    WrongGenesis,
    RollbackDetected,
    EquivocationDetected,
    SkippedTransition,
    InvalidSuccessorShape,
}

fn checkpoint_commitment(checkpoint: &V2QualifierAntiRollbackCheckpoint) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        V2_QUALIFIER_ANTI_ROLLBACK_CHECKPOINT_COMMITMENT_REVISION.as_bytes(),
    );
    encode_bytes(&mut bytes, &checkpoint.canonical_body_bytes());
    *blake3::hash(&bytes).as_bytes()
}

fn field<'a>(
    line: &'a str,
    expected_key: &'static str,
) -> Result<&'a str, V2QualifierAntiRollbackError> {
    let Some((key, value)) = line.split_once('=') else {
        return Err(V2QualifierAntiRollbackError::WrongField(expected_key));
    };
    if key != expected_key || value.is_empty() || value.contains('=') {
        return Err(V2QualifierAntiRollbackError::WrongField(expected_key));
    }
    Ok(value)
}

fn canonical_u64(value: &str) -> Result<u64, V2QualifierAntiRollbackError> {
    if value.is_empty()
        || !value.bytes().all(|byte| byte.is_ascii_digit())
        || (value.len() > 1 && value.starts_with('0'))
    {
        return Err(V2QualifierAntiRollbackError::InvalidNumber);
    }
    let parsed = value
        .parse::<u64>()
        .map_err(|_| V2QualifierAntiRollbackError::InvalidNumber)?;
    if parsed.to_string() != value {
        return Err(V2QualifierAntiRollbackError::InvalidNumber);
    }
    Ok(parsed)
}

fn optional_commitment(
    value: &str,
) -> Result<Option<[u8; 32]>, V2QualifierAntiRollbackError> {
    if value == "none" {
        Ok(None)
    } else {
        decode_hex32(value).map(Some)
    }
}

fn decode_hex32(value: &str) -> Result<[u8; 32], V2QualifierAntiRollbackError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(V2QualifierAntiRollbackError::InvalidHex);
    }
    let mut output = [0_u8; 32];
    for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_nibble(chunk[0]).ok_or(V2QualifierAntiRollbackError::InvalidHex)?;
        let low = hex_nibble(chunk[1]).ok_or(V2QualifierAntiRollbackError::InvalidHex)?;
        output[index] = (high << 4) | low;
    }
    Ok(output)
}

const fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn optional_hex(value: Option<[u8; 32]>) -> String {
    value.map(hex32).unwrap_or_else(|| "none".to_string())
}

fn hex32(value: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in value {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

fn push_field(out: &mut String, key: &str, value: &str) {
    out.push_str(key);
    out.push('=');
    out.push_str(value);
    out.push('\n');
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarks::eureka::v2_qualifier_authority::V2QualifierAuthorityProfile;
    use crate::benchmarks::eureka::v2_qualifier_signer_policy::V2GovernanceSigner;

    fn profile(workflow: char, contract: char) -> V2QualifierAuthorityProfile {
        V2QualifierAuthorityProfile::current_from_hex(
            &workflow.to_string().repeat(64),
            &contract.to_string().repeat(64),
        )
        .unwrap()
    }

    fn authority_genesis(workflow: char, contract: char) -> V2QualifierAuthorityRecord {
        V2QualifierAuthorityRecord::genesis(1, profile(workflow, contract)).unwrap()
    }

    fn lineage(workflow: char, contract: char) -> V2QualifierAuthorityLineage {
        V2QualifierAuthorityLineage::activate_genesis(authority_genesis(workflow, contract)).unwrap()
    }

    fn signer(principal: &str, key: char) -> V2GovernanceSigner {
        V2GovernanceSigner::from_hex(
            principal,
            "ssh-ed25519",
            &key.to_string().repeat(64),
        )
        .unwrap()
    }

    fn policy(key: char) -> V2QualifierSignerPolicy {
        V2QualifierSignerPolicy::genesis(
            1,
            1,
            vec![signer("governance@example.invalid", key)],
        )
        .unwrap()
    }

    #[test]
    fn genesis_checkpoint_is_deterministic_canonical_and_round_trips() {
        let lineage_a = lineage('b', 'c');
        let lineage_b = lineage('b', 'c');
        let policy_a = policy('a');
        let policy_b = policy('a');
        let first = V2QualifierAntiRollbackGuard::activate_genesis(&lineage_a, &policy_a).unwrap();
        let second = V2QualifierAntiRollbackGuard::activate_genesis(&lineage_b, &policy_b).unwrap();

        assert_eq!(first.checkpoint(), second.checkpoint());
        assert_eq!(first.checkpoint().authority_sequence(), 1);
        assert_eq!(first.checkpoint().signer_policy_sequence(), 1);
        assert_eq!(first.checkpoint().predecessor_checkpoint_commitment(), None);
        assert_ne!(first.checkpoint().commitment(), [0_u8; 32]);

        let bytes = first.checkpoint().persisted_bytes();
        let parsed = V2QualifierAntiRollbackCheckpoint::parse_persisted(&bytes).unwrap();
        assert_eq!(&parsed, first.checkpoint());
        assert_eq!(parsed.persisted_bytes(), bytes);
    }

    #[test]
    fn authority_advance_consumes_exact_current_lineage_and_ratchets_checkpoint() {
        let genesis = authority_genesis('b', 'c');
        let successor = V2QualifierAuthorityRecord::rotate(
            &genesis,
            genesis.commitment(),
            2,
            profile('d', 'c'),
        )
        .unwrap();
        let lineage = V2QualifierAuthorityLineage::activate_genesis(genesis).unwrap();
        let policy = policy('a');
        let guard = V2QualifierAntiRollbackGuard::activate_genesis(&lineage, &policy).unwrap();
        let predecessor_checkpoint = guard.checkpoint().commitment();

        let (guard, lineage) = guard
            .advance_authority(lineage, successor, &policy)
            .unwrap();
        assert_eq!(guard.checkpoint().authority_sequence(), 2);
        assert_eq!(guard.checkpoint().signer_policy_sequence(), 1);
        assert_eq!(
            guard.checkpoint().predecessor_checkpoint_commitment(),
            Some(predecessor_checkpoint)
        );
        guard.verify_current(&lineage, &policy).unwrap();
    }

    #[test]
    fn sibling_fork_cannot_be_smuggled_after_current_head_advances() {
        let genesis = authority_genesis('b', 'c');
        let left = V2QualifierAuthorityRecord::rotate(
            &genesis,
            genesis.commitment(),
            2,
            profile('d', 'c'),
        )
        .unwrap();
        let right = V2QualifierAuthorityRecord::rotate(
            &genesis,
            genesis.commitment(),
            2,
            profile('b', 'd'),
        )
        .unwrap();
        let lineage = V2QualifierAuthorityLineage::activate_genesis(genesis).unwrap();
        let policy = policy('a');
        let guard = V2QualifierAntiRollbackGuard::activate_genesis(&lineage, &policy).unwrap();
        let (guard, lineage) = guard.advance_authority(lineage, left, &policy).unwrap();

        assert_eq!(
            guard.advance_authority(lineage, right, &policy).unwrap_err(),
            V2QualifierAntiRollbackError::AuthorityTransition(
                V2QualifierCurrentnessError::StalePredecessor
            )
        );
    }

    #[test]
    fn signer_policy_advance_requires_exact_predecessor_and_next_sequence() {
        let lineage = lineage('b', 'c');
        let policy1 = policy('a');
        let policy2 = V2QualifierSignerPolicy::rotate(
            &policy1,
            policy1.commitment(),
            2,
            1,
            vec![signer("replacement@example.invalid", 'b')],
        )
        .unwrap();
        let policy3 = V2QualifierSignerPolicy::rotate(
            &policy2,
            policy2.commitment(),
            3,
            1,
            vec![signer("third@example.invalid", 'c')],
        )
        .unwrap();

        let guard = V2QualifierAntiRollbackGuard::activate_genesis(&lineage, &policy1).unwrap();
        assert_eq!(
            guard
                .advance_signer_policy(&lineage, &policy1, policy3)
                .unwrap_err(),
            V2QualifierAntiRollbackError::SignerPolicyWrongSequence
        );

        let guard = V2QualifierAntiRollbackGuard::activate_genesis(&lineage, &policy1).unwrap();
        let predecessor_checkpoint = guard.checkpoint().commitment();
        let (guard, policy2) = guard
            .advance_signer_policy(&lineage, &policy1, policy2)
            .unwrap();
        assert_eq!(guard.checkpoint().authority_sequence(), 1);
        assert_eq!(guard.checkpoint().signer_policy_sequence(), 2);
        assert_eq!(
            guard.checkpoint().predecessor_checkpoint_commitment(),
            Some(predecessor_checkpoint)
        );
        guard.verify_current(&lineage, &policy2).unwrap();
    }

    #[test]
    fn durable_high_water_rejects_rollback_and_same_height_equivocation() {
        let genesis = authority_genesis('b', 'c');
        let left = V2QualifierAuthorityRecord::rotate(
            &genesis,
            genesis.commitment(),
            2,
            profile('d', 'c'),
        )
        .unwrap();
        let right = V2QualifierAuthorityRecord::rotate(
            &genesis,
            genesis.commitment(),
            2,
            profile('b', 'd'),
        )
        .unwrap();
        let policy = policy('a');

        let left_lineage = V2QualifierAuthorityLineage::activate_genesis(genesis).unwrap();
        let left_genesis_guard =
            V2QualifierAntiRollbackGuard::activate_genesis(&left_lineage, &policy).unwrap();
        let old_checkpoint = left_genesis_guard.checkpoint().clone();
        let (left_guard, _) = left_genesis_guard
            .advance_authority(left_lineage, left, &policy)
            .unwrap();
        assert_eq!(
            left_guard.classify_candidate(&old_checkpoint).unwrap_err(),
            V2QualifierAntiRollbackError::RollbackDetected
        );

        let right_lineage = lineage('b', 'c');
        let right_guard =
            V2QualifierAntiRollbackGuard::activate_genesis(&right_lineage, &policy).unwrap();
        let (right_guard, _) = right_guard
            .advance_authority(right_lineage, right, &policy)
            .unwrap();
        assert_eq!(
            left_guard
                .classify_candidate(right_guard.checkpoint())
                .unwrap_err(),
            V2QualifierAntiRollbackError::EquivocationDetected
        );
    }

    #[test]
    fn structural_recovery_position_does_not_skip_intermediate_checkpoints() {
        let genesis = authority_genesis('b', 'c');
        let a2 = V2QualifierAuthorityRecord::rotate(
            &genesis,
            genesis.commitment(),
            2,
            profile('d', 'c'),
        )
        .unwrap();
        let a3 = V2QualifierAuthorityRecord::rotate(
            &a2,
            a2.commitment(),
            3,
            profile('e', 'c'),
        )
        .unwrap();
        let lineage1 = V2QualifierAuthorityLineage::activate_genesis(genesis).unwrap();
        let policy = policy('a');
        let guard1 = V2QualifierAntiRollbackGuard::activate_genesis(&lineage1, &policy).unwrap();
        let (guard2, lineage2) = guard1
            .advance_authority(lineage1, a2, &policy)
            .unwrap();
        let checkpoint2 = guard2.checkpoint().clone();
        let (guard3, _) = guard2
            .advance_authority(lineage2, a3, &policy)
            .unwrap();

        let genesis_lineage = lineage('b', 'c');
        let genesis_guard =
            V2QualifierAntiRollbackGuard::activate_genesis(&genesis_lineage, &policy).unwrap();
        assert_eq!(
            genesis_guard.classify_candidate(&checkpoint2).unwrap(),
            V2CheckpointCandidateDisposition::StructurallyImmediateSuccessor
        );
        assert_eq!(
            genesis_guard
                .classify_candidate(guard3.checkpoint())
                .unwrap_err(),
            V2QualifierAntiRollbackError::SkippedTransition
        );
    }

    #[test]
    fn persisted_corruption_noncanonical_numbers_and_state_mismatch_fail_closed() {
        let lineage = lineage('b', 'c');
        let policy = policy('a');
        let guard = V2QualifierAntiRollbackGuard::activate_genesis(&lineage, &policy).unwrap();
        let bytes = guard.checkpoint().persisted_bytes();

        let corrupted = std::str::from_utf8(&bytes)
            .unwrap()
            .replacen(
                "checkpoint_schema_revision=EUREKA.002.V2.QUALIFIER_ANTI_ROLLBACK_CHECKPOINT.v1",
                "checkpoint_schema_revision=EUREKA.002.V2.QUALIFIER_ANTI_ROLLBACK_CHECKPOINT.v9",
                1,
            );
        assert_eq!(
            V2QualifierAntiRollbackCheckpoint::parse_persisted(corrupted.as_bytes()).unwrap_err(),
            V2QualifierAntiRollbackError::UnsupportedSchema
        );

        let noncanonical = std::str::from_utf8(&bytes)
            .unwrap()
            .replacen("authority_sequence=1\n", "authority_sequence=01\n", 1);
        assert_eq!(
            V2QualifierAntiRollbackCheckpoint::parse_persisted(noncanonical.as_bytes())
                .unwrap_err(),
            V2QualifierAntiRollbackError::InvalidNumber
        );

        V2QualifierAntiRollbackGuard::recover_exact(&bytes, &lineage, &policy).unwrap();
        let different_policy = policy('b');
        assert_eq!(
            V2QualifierAntiRollbackGuard::recover_exact(&bytes, &lineage, &different_policy)
                .unwrap_err(),
            V2QualifierAntiRollbackError::CurrentStateMismatch
        );
    }

    #[test]
    fn anti_rollback_source_has_no_execution_or_real_root_surface() {
        let production = include_str!("v2_qualifier_anti_rollback.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for forbidden in [
            "V2AdmittedQualifierRoot",
            "V2CanaryAuthorization",
            "predict_ticket",
            "reveal(",
            "score_consequence",
            "SystemTime",
            "UNIX_EPOCH",
            "fs::write",
        ] {
            assert!(
                !production.contains(forbidden),
                "forbidden anti-rollback production surface: {forbidden}"
            );
        }
        assert!(production.contains("persisted_bytes"));
        assert!(production.contains("RollbackDetected"));
        assert!(production.contains("EquivocationDetected"));
        assert!(production.contains("StructurallyImmediateSuccessor"));
    }
}
