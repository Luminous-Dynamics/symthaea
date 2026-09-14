// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical external high-water witness mechanics for EUREKA-002 V2.
//!
//! Local crash-consistent persistence cannot detect rollback of the entire
//! local state directory to an older authentic snapshot. This test-only module
//! defines canonical witness bytes and fail-closed comparison mechanics that a
//! future independently retained publication domain can carry.
//!
//! A witness constructed here is still only a *local witness statement* until
//! an independent publication/acceptance receipt exists. This module performs
//! no network publication, chooses no provider, admits no root, and grants no
//! execution authority.

#![allow(dead_code)]

use super::v2_durably_authorized_transition::V2DurablyAuthorizedAuthorityTransition;
use super::v2_qualifier_admission_manifest::V2_REPOSITORY_IDENTITY;
use super::v2_qualifier_anti_rollback::V2QualifierAntiRollbackCheckpoint;
use std::cmp::Ordering;

pub(super) const V2_EXTERNAL_HIGH_WATER_WITNESS_SCHEMA: &str =
    "EUREKA.002.V2.EXTERNAL_HIGH_WATER_WITNESS.v1";
pub(super) const V2_EXTERNAL_HIGH_WATER_WITNESS_COMMITMENT_REVISION: &str =
    "EUREKA.002.V2.EXTERNAL_HIGH_WATER_WITNESS_COMMITMENT.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2ExternalHighWaterWitness {
    witness_sequence: u64,
    predecessor_witness_commitment: Option<[u8; 32]>,
    genesis_authority_commitment: [u8; 32],
    authority_sequence: u64,
    authority_commitment: [u8; 32],
    currentness_commitment: [u8; 32],
    signer_policy_sequence: u64,
    signer_policy_commitment: [u8; 32],
    checkpoint_commitment: [u8; 32],
    durable_transition_commitment: Option<[u8; 32]>,
    commitment: [u8; 32],
}

impl V2ExternalHighWaterWitness {
    /// Construct witness genesis from an exact qualifier genesis checkpoint.
    ///
    /// This does not publish anything externally. It only creates canonical
    /// bytes suitable for later independently retained publication.
    pub(super) fn genesis(
        checkpoint: &V2QualifierAntiRollbackCheckpoint,
    ) -> Result<Self, V2ExternalHighWaterWitnessError> {
        if checkpoint.authority_sequence() != 1
            || checkpoint.signer_policy_sequence() != 1
            || checkpoint.predecessor_checkpoint_commitment().is_some()
            || checkpoint.genesis_authority_commitment() != checkpoint.authority_commitment()
        {
            return Err(V2ExternalHighWaterWitnessError::InvalidGenesisCheckpoint);
        }

        Ok(Self::new(
            1,
            None,
            checkpoint,
            None,
        ))
    }

    /// Construct the exact next witness from a signed-governance-authorized,
    /// locally durable authority transition.
    ///
    /// Witness publication is intentionally one transition at a time. A caller
    /// cannot jump directly from witness A1 to local A3 and call that externally
    /// witnessed; every accepted witness successor has an exact predecessor.
    pub(super) fn successor_authority(
        previous: &Self,
        durable: &V2DurablyAuthorizedAuthorityTransition,
    ) -> Result<Self, V2ExternalHighWaterWitnessError> {
        let transition = durable.transition();
        let predecessor = transition.predecessor_checkpoint();
        let successor = transition.successor_checkpoint();

        if predecessor.commitment() != previous.checkpoint_commitment
            || predecessor.genesis_authority_commitment()
                != previous.genesis_authority_commitment
            || predecessor.authority_sequence() != previous.authority_sequence
            || predecessor.authority_commitment() != previous.authority_commitment
            || predecessor.currentness_commitment() != previous.currentness_commitment
            || predecessor.signer_policy_sequence() != previous.signer_policy_sequence
            || predecessor.signer_policy_commitment() != previous.signer_policy_commitment
        {
            return Err(V2ExternalHighWaterWitnessError::PredecessorWitnessMismatch);
        }

        if successor.predecessor_checkpoint_commitment()
            != Some(previous.checkpoint_commitment)
            || successor.genesis_authority_commitment()
                != previous.genesis_authority_commitment
            || successor.authority_sequence()
                != previous
                    .authority_sequence
                    .checked_add(1)
                    .ok_or(V2ExternalHighWaterWitnessError::SequenceExhausted)?
            || successor.authority_commitment() == previous.authority_commitment
            || successor.currentness_commitment() == previous.currentness_commitment
            || successor.signer_policy_sequence() != previous.signer_policy_sequence
            || successor.signer_policy_commitment() != previous.signer_policy_commitment
        {
            return Err(V2ExternalHighWaterWitnessError::InvalidAuthoritySuccessor);
        }

        let witness_sequence = previous
            .witness_sequence
            .checked_add(1)
            .ok_or(V2ExternalHighWaterWitnessError::SequenceExhausted)?;

        Ok(Self::new(
            witness_sequence,
            Some(previous.commitment),
            successor,
            Some(durable.commitment()),
        ))
    }

    fn new(
        witness_sequence: u64,
        predecessor_witness_commitment: Option<[u8; 32]>,
        checkpoint: &V2QualifierAntiRollbackCheckpoint,
        durable_transition_commitment: Option<[u8; 32]>,
    ) -> Self {
        let mut witness = Self {
            witness_sequence,
            predecessor_witness_commitment,
            genesis_authority_commitment: checkpoint.genesis_authority_commitment(),
            authority_sequence: checkpoint.authority_sequence(),
            authority_commitment: checkpoint.authority_commitment(),
            currentness_commitment: checkpoint.currentness_commitment(),
            signer_policy_sequence: checkpoint.signer_policy_sequence(),
            signer_policy_commitment: checkpoint.signer_policy_commitment(),
            checkpoint_commitment: checkpoint.commitment(),
            durable_transition_commitment,
            commitment: [0_u8; 32],
        };
        witness.commitment = witness_commitment(&witness);
        witness
    }

    pub(super) const fn witness_sequence(&self) -> u64 {
        self.witness_sequence
    }

    pub(super) const fn predecessor_witness_commitment(&self) -> Option<[u8; 32]> {
        self.predecessor_witness_commitment
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

    pub(super) const fn checkpoint_commitment(&self) -> [u8; 32] {
        self.checkpoint_commitment
    }

    pub(super) const fn durable_transition_commitment(&self) -> Option<[u8; 32]> {
        self.durable_transition_commitment
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }

    /// Canonical provider-neutral bytes. Human timestamps, URLs, PR numbers and
    /// publication-provider identifiers deliberately do not participate in the
    /// witness identity.
    pub(super) fn persisted_bytes(&self) -> Vec<u8> {
        let mut out = String::from_utf8(self.canonical_body_bytes())
            .expect("witness canonical body is ASCII");
        push_field(&mut out, "witness_commitment", &hex32(self.commitment));
        out.into_bytes()
    }

    fn canonical_body_bytes(&self) -> Vec<u8> {
        let mut out = String::new();
        push_field(
            &mut out,
            "witness_schema_revision",
            V2_EXTERNAL_HIGH_WATER_WITNESS_SCHEMA,
        );
        push_field(&mut out, "repository", V2_REPOSITORY_IDENTITY);
        push_field(
            &mut out,
            "witness_sequence",
            &self.witness_sequence.to_string(),
        );
        push_field(
            &mut out,
            "predecessor_witness_commitment",
            &optional_hex(self.predecessor_witness_commitment),
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
            "checkpoint_commitment",
            &hex32(self.checkpoint_commitment),
        );
        push_field(
            &mut out,
            "durable_transition_commitment",
            &optional_hex(self.durable_transition_commitment),
        );
        out.into_bytes()
    }

    pub(super) fn parse_persisted(
        input: &[u8],
    ) -> Result<Self, V2ExternalHighWaterWitnessError> {
        let text = std::str::from_utf8(input)
            .map_err(|_| V2ExternalHighWaterWitnessError::InvalidUtf8)?;
        if !text.ends_with('\n') {
            return Err(V2ExternalHighWaterWitnessError::MissingTrailingNewline);
        }
        let lines: Vec<&str> = text[..text.len() - 1].split('\n').collect();
        if lines.len() != 13 {
            return Err(V2ExternalHighWaterWitnessError::WrongFieldCount);
        }

        if field(lines[0], "witness_schema_revision")?
            != V2_EXTERNAL_HIGH_WATER_WITNESS_SCHEMA
        {
            return Err(V2ExternalHighWaterWitnessError::UnsupportedSchema);
        }
        if field(lines[1], "repository")? != V2_REPOSITORY_IDENTITY {
            return Err(V2ExternalHighWaterWitnessError::WrongRepository);
        }

        let witness_sequence = canonical_u64(field(lines[2], "witness_sequence")?)?;
        let predecessor_witness_commitment = optional_commitment(field(
            lines[3],
            "predecessor_witness_commitment",
        )?)?;
        let genesis_authority_commitment =
            decode_hex32(field(lines[4], "genesis_authority_commitment")?)?;
        let authority_sequence = canonical_u64(field(lines[5], "authority_sequence")?)?;
        let authority_commitment = decode_hex32(field(lines[6], "authority_commitment")?)?;
        let currentness_commitment = decode_hex32(field(lines[7], "currentness_commitment")?)?;
        let signer_policy_sequence =
            canonical_u64(field(lines[8], "signer_policy_sequence")?)?;
        let signer_policy_commitment =
            decode_hex32(field(lines[9], "signer_policy_commitment")?)?;
        let checkpoint_commitment = decode_hex32(field(lines[10], "checkpoint_commitment")?)?;
        let durable_transition_commitment = optional_commitment(field(
            lines[11],
            "durable_transition_commitment",
        )?)?;
        let claimed_commitment = decode_hex32(field(lines[12], "witness_commitment")?)?;

        if witness_sequence == 0 || authority_sequence == 0 || signer_policy_sequence == 0 {
            return Err(V2ExternalHighWaterWitnessError::InvalidWitnessShape);
        }

        let witness_genesis = witness_sequence == 1;
        if witness_genesis != predecessor_witness_commitment.is_none()
            || witness_genesis != durable_transition_commitment.is_none()
            || (witness_genesis
                && (authority_sequence != 1
                    || signer_policy_sequence != 1
                    || authority_commitment != genesis_authority_commitment))
        {
            return Err(V2ExternalHighWaterWitnessError::InvalidWitnessShape);
        }

        let mut witness = Self {
            witness_sequence,
            predecessor_witness_commitment,
            genesis_authority_commitment,
            authority_sequence,
            authority_commitment,
            currentness_commitment,
            signer_policy_sequence,
            signer_policy_commitment,
            checkpoint_commitment,
            durable_transition_commitment,
            commitment: [0_u8; 32],
        };
        witness.commitment = witness_commitment(&witness);
        if witness.commitment != claimed_commitment {
            return Err(V2ExternalHighWaterWitnessError::CommitmentMismatch);
        }
        if witness.persisted_bytes() != input {
            return Err(V2ExternalHighWaterWitnessError::NonCanonicalEncoding);
        }
        Ok(witness)
    }

    /// Compare one local high-water checkpoint against this externally accepted
    /// witness identity.
    ///
    /// `UnwitnessedAhead` is intentionally restricted to an exact immediate
    /// successor. A local state multiple transitions ahead needs an explicit
    /// chain proof and is fail-closed here as `AheadRequiresChain`.
    pub(super) fn classify_local_checkpoint(
        &self,
        local: &V2QualifierAntiRollbackCheckpoint,
    ) -> V2ExternalHighWaterDisposition {
        if local.genesis_authority_commitment() != self.genesis_authority_commitment {
            return V2ExternalHighWaterDisposition::Incomparable(
                V2ExternalHighWaterIncomparableReason::WrongGenesis,
            );
        }

        let authority_order = local.authority_sequence().cmp(&self.authority_sequence);
        let policy_order = local
            .signer_policy_sequence()
            .cmp(&self.signer_policy_sequence);

        match (authority_order, policy_order) {
            (Ordering::Equal, Ordering::Equal) => {
                if local.authority_commitment() == self.authority_commitment
                    && local.currentness_commitment() == self.currentness_commitment
                    && local.signer_policy_commitment() == self.signer_policy_commitment
                    && local.commitment() == self.checkpoint_commitment
                {
                    V2ExternalHighWaterDisposition::ExactWitnessed
                } else {
                    V2ExternalHighWaterDisposition::EquivocationDetected
                }
            }
            (Ordering::Less | Ordering::Equal, Ordering::Less | Ordering::Equal) => {
                V2ExternalHighWaterDisposition::RollbackDetected
            }
            (Ordering::Greater | Ordering::Equal, Ordering::Greater | Ordering::Equal) => {
                self.classify_ahead(local)
            }
            _ => V2ExternalHighWaterDisposition::Incomparable(
                V2ExternalHighWaterIncomparableReason::CrossedSequenceVector,
            ),
        }
    }

    fn classify_ahead(
        &self,
        local: &V2QualifierAntiRollbackCheckpoint,
    ) -> V2ExternalHighWaterDisposition {
        let authority_delta = local.authority_sequence() - self.authority_sequence;
        let policy_delta = local.signer_policy_sequence() - self.signer_policy_sequence;
        if !matches!((authority_delta, policy_delta), (1, 0) | (0, 1)) {
            return V2ExternalHighWaterDisposition::Incomparable(
                V2ExternalHighWaterIncomparableReason::AheadRequiresChain,
            );
        }
        if local.predecessor_checkpoint_commitment() != Some(self.checkpoint_commitment) {
            return V2ExternalHighWaterDisposition::Incomparable(
                V2ExternalHighWaterIncomparableReason::WrongCheckpointPredecessor,
            );
        }

        match (authority_delta, policy_delta) {
            (1, 0)
                if local.authority_commitment() != self.authority_commitment
                    && local.currentness_commitment() != self.currentness_commitment
                    && local.signer_policy_commitment() == self.signer_policy_commitment =>
            {
                V2ExternalHighWaterDisposition::UnwitnessedAhead
            }
            (0, 1)
                if local.authority_commitment() == self.authority_commitment
                    && local.currentness_commitment() == self.currentness_commitment
                    && local.signer_policy_commitment() != self.signer_policy_commitment =>
            {
                V2ExternalHighWaterDisposition::UnwitnessedAhead
            }
            _ => V2ExternalHighWaterDisposition::Incomparable(
                V2ExternalHighWaterIncomparableReason::InvalidImmediateSuccessorShape,
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2ExternalHighWaterDisposition {
    ExactWitnessed,
    RollbackDetected,
    UnwitnessedAhead,
    EquivocationDetected,
    Incomparable(V2ExternalHighWaterIncomparableReason),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2ExternalHighWaterIncomparableReason {
    WrongGenesis,
    CrossedSequenceVector,
    AheadRequiresChain,
    WrongCheckpointPredecessor,
    InvalidImmediateSuccessorShape,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2ExternalHighWaterWitnessError {
    InvalidGenesisCheckpoint,
    PredecessorWitnessMismatch,
    InvalidAuthoritySuccessor,
    SequenceExhausted,
    InvalidUtf8,
    MissingTrailingNewline,
    WrongFieldCount,
    WrongField(&'static str),
    UnsupportedSchema,
    WrongRepository,
    InvalidNumber,
    InvalidHex,
    InvalidWitnessShape,
    CommitmentMismatch,
    NonCanonicalEncoding,
}

fn witness_commitment(witness: &V2ExternalHighWaterWitness) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        V2_EXTERNAL_HIGH_WATER_WITNESS_COMMITMENT_REVISION.as_bytes(),
    );
    encode_bytes(&mut bytes, &witness.canonical_body_bytes());
    *blake3::hash(&bytes).as_bytes()
}

fn field<'a>(
    line: &'a str,
    expected_key: &'static str,
) -> Result<&'a str, V2ExternalHighWaterWitnessError> {
    let Some((key, value)) = line.split_once('=') else {
        return Err(V2ExternalHighWaterWitnessError::WrongField(expected_key));
    };
    if key != expected_key || value.is_empty() || value.contains('=') {
        return Err(V2ExternalHighWaterWitnessError::WrongField(expected_key));
    }
    Ok(value)
}

fn canonical_u64(value: &str) -> Result<u64, V2ExternalHighWaterWitnessError> {
    if value.is_empty()
        || !value.bytes().all(|byte| byte.is_ascii_digit())
        || (value.len() > 1 && value.starts_with('0'))
    {
        return Err(V2ExternalHighWaterWitnessError::InvalidNumber);
    }
    let parsed = value
        .parse::<u64>()
        .map_err(|_| V2ExternalHighWaterWitnessError::InvalidNumber)?;
    if parsed.to_string() != value {
        return Err(V2ExternalHighWaterWitnessError::InvalidNumber);
    }
    Ok(parsed)
}

fn optional_commitment(
    value: &str,
) -> Result<Option<[u8; 32]>, V2ExternalHighWaterWitnessError> {
    if value == "none" {
        Ok(None)
    } else {
        decode_hex32(value).map(Some)
    }
}

fn decode_hex32(value: &str) -> Result<[u8; 32], V2ExternalHighWaterWitnessError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(V2ExternalHighWaterWitnessError::InvalidHex);
    }
    let mut output = [0_u8; 32];
    for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_nibble(chunk[0]).ok_or(V2ExternalHighWaterWitnessError::InvalidHex)?;
        let low = hex_nibble(chunk[1]).ok_or(V2ExternalHighWaterWitnessError::InvalidHex)?;
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

fn push_field(output: &mut String, key: &str, value: &str) {
    output.push_str(key);
    output.push('=');
    output.push_str(value);
    output.push('\n');
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarks::eureka::v2_openssh_admission_verifier::V2OpenSshAdmissionSignature;
    use crate::benchmarks::eureka::v2_qualifier_anti_rollback::V2QualifierAntiRollbackGuard;
    use crate::benchmarks::eureka::v2_qualifier_authority::{
        V2QualifierAuthorityProfile, V2QualifierAuthorityRecord,
    };
    use crate::benchmarks::eureka::v2_qualifier_checkpoint_store::V2QualifierCheckpointStore;
    use crate::benchmarks::eureka::v2_qualifier_currentness::V2QualifierAuthorityLineage;
    use crate::benchmarks::eureka::v2_qualifier_signer_policy::{
        V2GovernanceSigner, V2QualifierSignerPolicy,
    };
    use crate::benchmarks::eureka::v2_signed_governance_authorization::authorize_signed_authority_rotation;
    use crate::benchmarks::eureka::v2_authorized_anti_rollback_transition::advance_authority_with_signed_governance;
    use crate::benchmarks::eureka::v2_durably_authorized_transition::persist_authorized_authority_transition;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);
    const MANIFEST: &[u8] = include_bytes!("fixtures/eureka-v2-signed-transition-rotation.env");
    const ALPHA_PUB: &str = include_str!("fixtures/eureka-v2-signed-transition-alpha.pub");
    const BETA_PUB: &str = include_str!("fixtures/eureka-v2-signed-transition-beta.pub");
    const ALPHA_SIG: &[u8] = include_bytes!("fixtures/eureka-v2-signed-transition-alpha.sig");
    const BETA_SIG: &[u8] = include_bytes!("fixtures/eureka-v2-signed-transition-beta.sig");

    fn profile(workflow: char, contract: char) -> V2QualifierAuthorityProfile {
        V2QualifierAuthorityProfile::current_from_hex(
            &workflow.to_string().repeat(64),
            &contract.to_string().repeat(64),
        )
        .unwrap()
    }

    fn signer(principal: &str, hash: &str) -> V2GovernanceSigner {
        V2GovernanceSigner::from_hex(principal, "ssh-ed25519", hash).unwrap()
    }

    fn policy() -> V2QualifierSignerPolicy {
        V2QualifierSignerPolicy::genesis(
            1,
            2,
            vec![
                signer(
                    "fixture-transition-alpha@example.invalid",
                    "98c9c52c03393af5fabb34d1a41a60a87073525c52f5fb7ca0a51c7de931da2d",
                ),
                signer(
                    "fixture-transition-beta@example.invalid",
                    "70535ce88cf2b7575de01e7e98559e95fa66d18dd7317f9411500f3930dcfbd7",
                ),
            ],
        )
        .unwrap()
    }

    fn signatures() -> [V2OpenSshAdmissionSignature<'static>; 2] {
        [
            V2OpenSshAdmissionSignature::new(
                "fixture-transition-alpha@example.invalid",
                ALPHA_PUB,
                ALPHA_SIG,
            ),
            V2OpenSshAdmissionSignature::new(
                "fixture-transition-beta@example.invalid",
                BETA_PUB,
                BETA_SIG,
            ),
        ]
    }

    fn predecessor() -> V2QualifierAuthorityRecord {
        V2QualifierAuthorityRecord::genesis(1, profile('c', 'd')).unwrap()
    }

    fn successor(predecessor: &V2QualifierAuthorityRecord) -> V2QualifierAuthorityRecord {
        V2QualifierAuthorityRecord::rotate(
            predecessor,
            predecessor.commitment(),
            2,
            profile('e', 'd'),
        )
        .unwrap()
    }

    fn transition_state() -> (
        V2QualifierAntiRollbackCheckpoint,
        V2QualifierAntiRollbackCheckpoint,
        V2DurablyAuthorizedAuthorityTransition,
        PathBuf,
    ) {
        let predecessor = predecessor();
        let successor = successor(&predecessor);
        let lineage = V2QualifierAuthorityLineage::activate_genesis(predecessor.clone()).unwrap();
        let policy = policy();
        let guard = V2QualifierAntiRollbackGuard::activate_genesis(&lineage, &policy).unwrap();
        let genesis = guard.checkpoint().clone();
        let authorization = authorize_signed_authority_rotation(
            MANIFEST,
            &policy,
            &signatures(),
            &lineage,
            &predecessor,
            &successor,
        )
        .unwrap();
        let (transition, _, _) = advance_authority_with_signed_governance(
            guard,
            lineage,
            &policy,
            successor,
            authorization,
        )
        .unwrap();
        let successor_checkpoint = transition.successor_checkpoint().clone();

        let id = TEST_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "symthaea-eureka-v2-external-witness-{}-{id}",
            std::process::id()
        ));
        fs::create_dir(&dir).unwrap();
        let store = V2QualifierCheckpointStore::new(dir.join("high-water.env")).unwrap();
        store.persist_genesis(&genesis).unwrap();
        let durable = persist_authorized_authority_transition(&store, transition).unwrap();
        (genesis, successor_checkpoint, durable, dir)
    }

    #[cfg(unix)]
    #[test]
    fn genesis_and_successor_witness_are_canonical_and_exactly_chained() {
        let (genesis_checkpoint, successor_checkpoint, durable, dir) = transition_state();
        let genesis = V2ExternalHighWaterWitness::genesis(&genesis_checkpoint).unwrap();
        let successor =
            V2ExternalHighWaterWitness::successor_authority(&genesis, &durable).unwrap();

        assert_eq!(genesis.witness_sequence(), 1);
        assert_eq!(genesis.predecessor_witness_commitment(), None);
        assert_eq!(genesis.durable_transition_commitment(), None);
        assert_eq!(genesis.checkpoint_commitment(), genesis_checkpoint.commitment());

        assert_eq!(successor.witness_sequence(), 2);
        assert_eq!(
            successor.predecessor_witness_commitment(),
            Some(genesis.commitment())
        );
        assert_eq!(successor.checkpoint_commitment(), successor_checkpoint.commitment());
        assert_eq!(
            successor.durable_transition_commitment(),
            Some(durable.commitment())
        );
        assert_eq!(
            V2ExternalHighWaterWitness::parse_persisted(&successor.persisted_bytes()).unwrap(),
            successor
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn exact_rollback_and_immediate_unwitnessed_ahead_are_distinct() {
        let (genesis_checkpoint, successor_checkpoint, durable, dir) = transition_state();
        let genesis = V2ExternalHighWaterWitness::genesis(&genesis_checkpoint).unwrap();
        assert_eq!(
            genesis.classify_local_checkpoint(&genesis_checkpoint),
            V2ExternalHighWaterDisposition::ExactWitnessed
        );
        assert_eq!(
            genesis.classify_local_checkpoint(&successor_checkpoint),
            V2ExternalHighWaterDisposition::UnwitnessedAhead
        );

        let successor =
            V2ExternalHighWaterWitness::successor_authority(&genesis, &durable).unwrap();
        assert_eq!(
            successor.classify_local_checkpoint(&genesis_checkpoint),
            V2ExternalHighWaterDisposition::RollbackDetected
        );
        assert_eq!(
            successor.classify_local_checkpoint(&successor_checkpoint),
            V2ExternalHighWaterDisposition::ExactWitnessed
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn same_vector_different_successor_is_equivocation() {
        let genesis_authority = predecessor();
        let sibling = V2QualifierAuthorityRecord::rotate(
            &genesis_authority,
            genesis_authority.commitment(),
            2,
            profile('f', 'd'),
        )
        .unwrap();
        let lineage = V2QualifierAuthorityLineage::activate_genesis(genesis_authority.clone()).unwrap();
        let policy = policy();
        let guard = V2QualifierAntiRollbackGuard::activate_genesis(&lineage, &policy).unwrap();
        let genesis_checkpoint = guard.checkpoint().clone();
        let (sibling_guard, _) = guard.advance_authority(lineage, sibling, &policy).unwrap();
        let sibling_checkpoint = sibling_guard.checkpoint().clone();

        let intended_authority = successor(&genesis_authority);
        let intended_lineage =
            V2QualifierAuthorityLineage::activate_genesis(genesis_authority).unwrap();
        let intended_guard =
            V2QualifierAntiRollbackGuard::activate_genesis(&intended_lineage, &policy).unwrap();
        let (intended_guard, _) = intended_guard
            .advance_authority(intended_lineage, intended_authority, &policy)
            .unwrap();
        let intended_checkpoint = intended_guard.checkpoint().clone();

        let mut witness = V2ExternalHighWaterWitness::genesis(&genesis_checkpoint).unwrap();
        witness.witness_sequence = 2;
        witness.predecessor_witness_commitment = Some(witness.commitment);
        witness.authority_sequence = intended_checkpoint.authority_sequence();
        witness.authority_commitment = intended_checkpoint.authority_commitment();
        witness.currentness_commitment = intended_checkpoint.currentness_commitment();
        witness.checkpoint_commitment = intended_checkpoint.commitment();
        witness.durable_transition_commitment = Some([7_u8; 32]);
        witness.commitment = witness_commitment(&witness);

        assert_eq!(
            witness.classify_local_checkpoint(&sibling_checkpoint),
            V2ExternalHighWaterDisposition::EquivocationDetected
        );
    }

    #[test]
    fn crossed_authority_and_policy_vectors_are_incomparable() {
        let authority = predecessor();
        let successor_authority = successor(&authority);
        let lineage = V2QualifierAuthorityLineage::activate_genesis(authority.clone()).unwrap();
        let policy = policy();
        let guard = V2QualifierAntiRollbackGuard::activate_genesis(&lineage, &policy).unwrap();
        let genesis_checkpoint = guard.checkpoint().clone();
        let (authority_guard, _) = guard
            .advance_authority(lineage, successor_authority, &policy)
            .unwrap();
        let authority_checkpoint = authority_guard.checkpoint().clone();

        let policy2 = V2QualifierSignerPolicy::rotate(
            &policy,
            policy.commitment(),
            2,
            1,
            vec![signer("policy-two@example.invalid", &"a".repeat(64))],
        )
        .unwrap();
        let lineage = V2QualifierAuthorityLineage::activate_genesis(authority).unwrap();
        let guard = V2QualifierAntiRollbackGuard::activate_genesis(&lineage, &policy).unwrap();
        let (policy_guard, _) = guard
            .advance_signer_policy(&lineage, &policy, policy2)
            .unwrap();
        let policy_checkpoint = policy_guard.checkpoint().clone();

        let mut witness = V2ExternalHighWaterWitness::genesis(&genesis_checkpoint).unwrap();
        witness.witness_sequence = 2;
        witness.predecessor_witness_commitment = Some(witness.commitment);
        witness.authority_sequence = authority_checkpoint.authority_sequence();
        witness.authority_commitment = authority_checkpoint.authority_commitment();
        witness.currentness_commitment = authority_checkpoint.currentness_commitment();
        witness.checkpoint_commitment = authority_checkpoint.commitment();
        witness.durable_transition_commitment = Some([8_u8; 32]);
        witness.commitment = witness_commitment(&witness);

        assert_eq!(
            witness.classify_local_checkpoint(&policy_checkpoint),
            V2ExternalHighWaterDisposition::Incomparable(
                V2ExternalHighWaterIncomparableReason::CrossedSequenceVector
            )
        );
    }

    #[test]
    fn multi_step_ahead_requires_explicit_chain_proof() {
        let a1 = predecessor();
        let a2 = successor(&a1);
        let a3 = V2QualifierAuthorityRecord::rotate(
            &a2,
            a2.commitment(),
            3,
            profile('f', 'd'),
        )
        .unwrap();
        let lineage = V2QualifierAuthorityLineage::activate_genesis(a1).unwrap();
        let policy = policy();
        let guard = V2QualifierAntiRollbackGuard::activate_genesis(&lineage, &policy).unwrap();
        let genesis_checkpoint = guard.checkpoint().clone();
        let (guard, lineage) = guard.advance_authority(lineage, a2, &policy).unwrap();
        let (guard, _) = guard.advance_authority(lineage, a3, &policy).unwrap();
        let a3_checkpoint = guard.checkpoint().clone();
        let witness = V2ExternalHighWaterWitness::genesis(&genesis_checkpoint).unwrap();

        assert_eq!(
            witness.classify_local_checkpoint(&a3_checkpoint),
            V2ExternalHighWaterDisposition::Incomparable(
                V2ExternalHighWaterIncomparableReason::AheadRequiresChain
            )
        );
    }

    #[test]
    fn parser_rejects_noncanonical_or_self_inconsistent_witness_bytes() {
        let authority = predecessor();
        let lineage = V2QualifierAuthorityLineage::activate_genesis(authority).unwrap();
        let policy = policy();
        let guard = V2QualifierAntiRollbackGuard::activate_genesis(&lineage, &policy).unwrap();
        let witness = V2ExternalHighWaterWitness::genesis(guard.checkpoint()).unwrap();
        let bytes = witness.persisted_bytes();
        assert_eq!(
            V2ExternalHighWaterWitness::parse_persisted(&bytes).unwrap(),
            witness
        );

        let mut missing_newline = bytes.clone();
        missing_newline.pop();
        assert_eq!(
            V2ExternalHighWaterWitness::parse_persisted(&missing_newline).unwrap_err(),
            V2ExternalHighWaterWitnessError::MissingTrailingNewline
        );

        let text = String::from_utf8(bytes).unwrap();
        let noncanonical = text.replacen("witness_sequence=1\n", "witness_sequence=01\n", 1);
        assert_eq!(
            V2ExternalHighWaterWitness::parse_persisted(noncanonical.as_bytes()).unwrap_err(),
            V2ExternalHighWaterWitnessError::InvalidNumber
        );
    }

    #[test]
    fn witness_source_has_no_publication_root_permit_clock_or_execution_surface() {
        let production = include_str!("v2_external_high_water_witness.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for forbidden in [
            "reqwest",
            "SystemTime",
            "UNIX_EPOCH",
            "github.com",
            "V2AdmittedQualifierRoot",
            "V2CanaryAuthorization",
            "predict_ticket",
            "reveal(",
            "score_consequence",
            "execution_authority_granted=true",
            "PRIVATE KEY",
        ] {
            assert!(
                !production.contains(forbidden),
                "forbidden external-witness surface: {forbidden}"
            );
        }
        assert!(production.contains("UnwitnessedAhead"));
        assert!(production.contains("AheadRequiresChain"));
        assert!(production.contains("V2DurablyAuthorizedAuthorityTransition"));
    }
}
