// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Provider-neutral publication envelope for EUREKA-002 V2 external witnesses.
//!
//! This canonical envelope is the object a future transparency-log adapter may
//! sign/publish. It binds the exact witness bytes, checkpoint bytes and, for a
//! successor, the independently verified witness/checkpoint successor proof.
//!
//! The envelope is deliberately **not external evidence**. Its canonical bytes
//! carry `externality_verified=false` and `execution_authority_granted=false`.
//! Only a future provider-specific verifier may elevate one exact envelope into
//! externally included evidence.

#![allow(dead_code)]

use super::v2_external_high_water_witness::{
    V2ExternalHighWaterDisposition, V2ExternalHighWaterWitness,
};
use super::v2_external_high_water_witness_chain::V2ExternalHighWaterSuccessorProof;
use super::v2_qualifier_admission_manifest::V2_REPOSITORY_IDENTITY;
use super::v2_qualifier_anti_rollback::V2QualifierAntiRollbackCheckpoint;

pub(super) const V2_EXTERNAL_WITNESS_PUBLICATION_ENVELOPE_SCHEMA: &str =
    "EUREKA.002.V2.EXTERNAL_WITNESS_PUBLICATION_ENVELOPE.v1";
pub(super) const V2_EXTERNAL_WITNESS_PUBLICATION_ENVELOPE_COMMITMENT_REVISION: &str =
    "EUREKA.002.V2.EXTERNAL_WITNESS_PUBLICATION_ENVELOPE_COMMITMENT.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2ExternalWitnessPublicationEnvelope {
    publication_sequence: u64,
    predecessor_publication_commitment: Option<[u8; 32]>,
    witness_sequence: u64,
    witness_commitment: [u8; 32],
    witness_bytes_blake3: [u8; 32],
    witness_bytes_len: u64,
    checkpoint_commitment: [u8; 32],
    checkpoint_bytes_blake3: [u8; 32],
    checkpoint_bytes_len: u64,
    successor_proof_commitment: Option<[u8; 32]>,
    durable_transition_commitment: Option<[u8; 32]>,
    commitment: [u8; 32],
}

impl V2ExternalWitnessPublicationEnvelope {
    pub(super) fn genesis(
        witness: &V2ExternalHighWaterWitness,
        checkpoint: &V2QualifierAntiRollbackCheckpoint,
    ) -> Result<Self, V2ExternalWitnessPublicationEnvelopeError> {
        if witness.witness_sequence() != 1
            || witness.predecessor_witness_commitment().is_some()
            || witness.durable_transition_commitment().is_some()
            || checkpoint.predecessor_checkpoint_commitment().is_some()
            || witness.classify_local_checkpoint(checkpoint)
                != V2ExternalHighWaterDisposition::ExactWitnessed
        {
            return Err(V2ExternalWitnessPublicationEnvelopeError::InvalidGenesisPayload);
        }

        Ok(Self::new(1, None, witness, checkpoint, None))
    }

    pub(super) fn successor(
        previous_envelope: &Self,
        previous_witness: &V2ExternalHighWaterWitness,
        previous_checkpoint: &V2QualifierAntiRollbackCheckpoint,
        previous_successor_proof: Option<&V2ExternalHighWaterSuccessorProof>,
        witness: &V2ExternalHighWaterWitness,
        checkpoint: &V2QualifierAntiRollbackCheckpoint,
        successor_proof: &V2ExternalHighWaterSuccessorProof,
    ) -> Result<Self, V2ExternalWitnessPublicationEnvelopeError> {
        previous_envelope
            .verify_bound_payload(
                previous_witness,
                previous_checkpoint,
                previous_successor_proof,
            )
            .map_err(|_| V2ExternalWitnessPublicationEnvelopeError::SuccessorBindingMismatch)?;

        if previous_checkpoint.commitment() != previous_witness.checkpoint_commitment()
            || previous_envelope.witness_commitment != previous_witness.commitment()
            || previous_envelope.checkpoint_commitment != previous_checkpoint.commitment()
            || witness.predecessor_witness_commitment() != Some(previous_witness.commitment())
            || witness.classify_local_checkpoint(checkpoint)
                != V2ExternalHighWaterDisposition::ExactWitnessed
            || successor_proof.predecessor_witness_commitment() != previous_witness.commitment()
            || successor_proof.successor_witness_commitment() != witness.commitment()
            || successor_proof.predecessor_checkpoint_commitment()
                != previous_checkpoint.commitment()
            || successor_proof.successor_checkpoint_commitment() != checkpoint.commitment()
            || Some(successor_proof.durable_transition_commitment())
                != witness.durable_transition_commitment()
        {
            return Err(V2ExternalWitnessPublicationEnvelopeError::SuccessorBindingMismatch);
        }

        let publication_sequence = previous_envelope
            .publication_sequence
            .checked_add(1)
            .ok_or(V2ExternalWitnessPublicationEnvelopeError::SequenceExhausted)?;
        let expected_witness_sequence = previous_witness
            .witness_sequence()
            .checked_add(1)
            .ok_or(V2ExternalWitnessPublicationEnvelopeError::SequenceExhausted)?;
        if witness.witness_sequence() != expected_witness_sequence
            || publication_sequence != witness.witness_sequence()
        {
            return Err(V2ExternalWitnessPublicationEnvelopeError::SuccessorBindingMismatch);
        }

        Ok(Self::new(
            publication_sequence,
            Some(previous_envelope.commitment),
            witness,
            checkpoint,
            Some(successor_proof.commitment()),
        ))
    }

    fn new(
        publication_sequence: u64,
        predecessor_publication_commitment: Option<[u8; 32]>,
        witness: &V2ExternalHighWaterWitness,
        checkpoint: &V2QualifierAntiRollbackCheckpoint,
        successor_proof_commitment: Option<[u8; 32]>,
    ) -> Self {
        let witness_bytes = witness.persisted_bytes();
        let checkpoint_bytes = checkpoint.persisted_bytes();
        let mut envelope = Self {
            publication_sequence,
            predecessor_publication_commitment,
            witness_sequence: witness.witness_sequence(),
            witness_commitment: witness.commitment(),
            witness_bytes_blake3: *blake3::hash(&witness_bytes).as_bytes(),
            witness_bytes_len: witness_bytes.len() as u64,
            checkpoint_commitment: checkpoint.commitment(),
            checkpoint_bytes_blake3: *blake3::hash(&checkpoint_bytes).as_bytes(),
            checkpoint_bytes_len: checkpoint_bytes.len() as u64,
            successor_proof_commitment,
            durable_transition_commitment: witness.durable_transition_commitment(),
            commitment: [0_u8; 32],
        };
        envelope.commitment = envelope_commitment(&envelope);
        envelope
    }

    pub(super) const fn publication_sequence(&self) -> u64 {
        self.publication_sequence
    }

    pub(super) const fn predecessor_publication_commitment(&self) -> Option<[u8; 32]> {
        self.predecessor_publication_commitment
    }

    pub(super) const fn witness_sequence(&self) -> u64 {
        self.witness_sequence
    }

    pub(super) const fn witness_commitment(&self) -> [u8; 32] {
        self.witness_commitment
    }

    pub(super) const fn witness_bytes_blake3(&self) -> [u8; 32] {
        self.witness_bytes_blake3
    }

    pub(super) const fn checkpoint_commitment(&self) -> [u8; 32] {
        self.checkpoint_commitment
    }

    pub(super) const fn checkpoint_bytes_blake3(&self) -> [u8; 32] {
        self.checkpoint_bytes_blake3
    }

    pub(super) const fn successor_proof_commitment(&self) -> Option<[u8; 32]> {
        self.successor_proof_commitment
    }

    pub(super) const fn durable_transition_commitment(&self) -> Option<[u8; 32]> {
        self.durable_transition_commitment
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }

    pub(super) fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = String::from_utf8(self.canonical_body_bytes())
            .expect("publication envelope canonical body is ASCII");
        push_field(&mut out, "envelope_commitment", &hex32(self.commitment));
        out.into_bytes()
    }

    fn canonical_body_bytes(&self) -> Vec<u8> {
        let mut out = String::new();
        push_field(
            &mut out,
            "envelope_schema_revision",
            V2_EXTERNAL_WITNESS_PUBLICATION_ENVELOPE_SCHEMA,
        );
        push_field(&mut out, "repository", V2_REPOSITORY_IDENTITY);
        push_field(
            &mut out,
            "publication_sequence",
            &self.publication_sequence.to_string(),
        );
        push_field(
            &mut out,
            "predecessor_publication_commitment",
            &optional_hex(self.predecessor_publication_commitment),
        );
        push_field(
            &mut out,
            "witness_sequence",
            &self.witness_sequence.to_string(),
        );
        push_field(
            &mut out,
            "witness_commitment",
            &hex32(self.witness_commitment),
        );
        push_field(
            &mut out,
            "witness_bytes_blake3",
            &hex32(self.witness_bytes_blake3),
        );
        push_field(
            &mut out,
            "witness_bytes_len",
            &self.witness_bytes_len.to_string(),
        );
        push_field(
            &mut out,
            "checkpoint_commitment",
            &hex32(self.checkpoint_commitment),
        );
        push_field(
            &mut out,
            "checkpoint_bytes_blake3",
            &hex32(self.checkpoint_bytes_blake3),
        );
        push_field(
            &mut out,
            "checkpoint_bytes_len",
            &self.checkpoint_bytes_len.to_string(),
        );
        push_field(
            &mut out,
            "successor_proof_commitment",
            &optional_hex(self.successor_proof_commitment),
        );
        push_field(
            &mut out,
            "durable_transition_commitment",
            &optional_hex(self.durable_transition_commitment),
        );
        push_field(&mut out, "externality_verified", "false");
        push_field(&mut out, "execution_authority_granted", "false");
        out.into_bytes()
    }

    pub(super) fn parse_canonical(
        input: &[u8],
    ) -> Result<Self, V2ExternalWitnessPublicationEnvelopeError> {
        let text = std::str::from_utf8(input)
            .map_err(|_| V2ExternalWitnessPublicationEnvelopeError::InvalidUtf8)?;
        if !text.ends_with('\n') {
            return Err(V2ExternalWitnessPublicationEnvelopeError::MissingTrailingNewline);
        }
        let lines: Vec<&str> = text[..text.len() - 1].split('\n').collect();
        if lines.len() != 16 {
            return Err(V2ExternalWitnessPublicationEnvelopeError::WrongFieldCount);
        }
        if field(lines[0], "envelope_schema_revision")?
            != V2_EXTERNAL_WITNESS_PUBLICATION_ENVELOPE_SCHEMA
        {
            return Err(V2ExternalWitnessPublicationEnvelopeError::UnsupportedSchema);
        }
        if field(lines[1], "repository")? != V2_REPOSITORY_IDENTITY {
            return Err(V2ExternalWitnessPublicationEnvelopeError::WrongRepository);
        }

        let publication_sequence = canonical_u64(field(lines[2], "publication_sequence")?)?;
        let predecessor_publication_commitment = optional_commitment(field(
            lines[3],
            "predecessor_publication_commitment",
        )?)?;
        let witness_sequence = canonical_u64(field(lines[4], "witness_sequence")?)?;
        let witness_commitment = decode_hex32(field(lines[5], "witness_commitment")?)?;
        let witness_bytes_blake3 = decode_hex32(field(lines[6], "witness_bytes_blake3")?)?;
        let witness_bytes_len = canonical_u64(field(lines[7], "witness_bytes_len")?)?;
        let checkpoint_commitment = decode_hex32(field(lines[8], "checkpoint_commitment")?)?;
        let checkpoint_bytes_blake3 = decode_hex32(field(lines[9], "checkpoint_bytes_blake3")?)?;
        let checkpoint_bytes_len = canonical_u64(field(lines[10], "checkpoint_bytes_len")?)?;
        let successor_proof_commitment = optional_commitment(field(
            lines[11],
            "successor_proof_commitment",
        )?)?;
        let durable_transition_commitment = optional_commitment(field(
            lines[12],
            "durable_transition_commitment",
        )?)?;
        if field(lines[13], "externality_verified")? != "false" {
            return Err(V2ExternalWitnessPublicationEnvelopeError::ExternalityClaimed);
        }
        if field(lines[14], "execution_authority_granted")? != "false" {
            return Err(V2ExternalWitnessPublicationEnvelopeError::ExecutionAuthorityClaimed);
        }
        let claimed_commitment = decode_hex32(field(lines[15], "envelope_commitment")?)?;

        if publication_sequence == 0
            || witness_sequence == 0
            || publication_sequence != witness_sequence
            || witness_bytes_len == 0
            || checkpoint_bytes_len == 0
        {
            return Err(V2ExternalWitnessPublicationEnvelopeError::InvalidEnvelopeShape);
        }
        let genesis = publication_sequence == 1;
        if genesis != predecessor_publication_commitment.is_none()
            || genesis != successor_proof_commitment.is_none()
            || genesis != durable_transition_commitment.is_none()
        {
            return Err(V2ExternalWitnessPublicationEnvelopeError::InvalidEnvelopeShape);
        }

        let mut envelope = Self {
            publication_sequence,
            predecessor_publication_commitment,
            witness_sequence,
            witness_commitment,
            witness_bytes_blake3,
            witness_bytes_len,
            checkpoint_commitment,
            checkpoint_bytes_blake3,
            checkpoint_bytes_len,
            successor_proof_commitment,
            durable_transition_commitment,
            commitment: [0_u8; 32],
        };
        envelope.commitment = envelope_commitment(&envelope);
        if envelope.commitment != claimed_commitment {
            return Err(V2ExternalWitnessPublicationEnvelopeError::CommitmentMismatch);
        }
        if envelope.canonical_bytes() != input {
            return Err(V2ExternalWitnessPublicationEnvelopeError::NonCanonicalEncoding);
        }
        Ok(envelope)
    }

    pub(super) fn verify_bound_payload(
        &self,
        witness: &V2ExternalHighWaterWitness,
        checkpoint: &V2QualifierAntiRollbackCheckpoint,
        successor_proof: Option<&V2ExternalHighWaterSuccessorProof>,
    ) -> Result<(), V2ExternalWitnessPublicationEnvelopeError> {
        let witness_bytes = witness.persisted_bytes();
        let checkpoint_bytes = checkpoint.persisted_bytes();
        if self.witness_sequence != witness.witness_sequence()
            || self.witness_commitment != witness.commitment()
            || self.witness_bytes_len != witness_bytes.len() as u64
            || self.witness_bytes_blake3 != *blake3::hash(&witness_bytes).as_bytes()
            || self.checkpoint_commitment != checkpoint.commitment()
            || self.checkpoint_bytes_len != checkpoint_bytes.len() as u64
            || self.checkpoint_bytes_blake3 != *blake3::hash(&checkpoint_bytes).as_bytes()
            || witness.classify_local_checkpoint(checkpoint)
                != V2ExternalHighWaterDisposition::ExactWitnessed
            || self.durable_transition_commitment != witness.durable_transition_commitment()
        {
            return Err(V2ExternalWitnessPublicationEnvelopeError::PayloadMismatch);
        }

        match (self.publication_sequence, successor_proof) {
            (1, None) if self.successor_proof_commitment.is_none() => Ok(()),
            (1, _) => Err(V2ExternalWitnessPublicationEnvelopeError::InvalidGenesisPayload),
            (_, Some(proof))
                if self.successor_proof_commitment == Some(proof.commitment())
                    && proof.successor_witness_commitment() == witness.commitment()
                    && proof.successor_checkpoint_commitment() == checkpoint.commitment()
                    && Some(proof.durable_transition_commitment())
                        == witness.durable_transition_commitment() =>
            {
                Ok(())
            }
            _ => Err(V2ExternalWitnessPublicationEnvelopeError::PayloadMismatch),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2ExternalWitnessPublicationEnvelopeError {
    InvalidGenesisPayload,
    SuccessorBindingMismatch,
    SequenceExhausted,
    InvalidUtf8,
    MissingTrailingNewline,
    WrongFieldCount,
    WrongField(&'static str),
    UnsupportedSchema,
    WrongRepository,
    InvalidNumber,
    InvalidHex,
    InvalidEnvelopeShape,
    ExternalityClaimed,
    ExecutionAuthorityClaimed,
    CommitmentMismatch,
    NonCanonicalEncoding,
    PayloadMismatch,
}

fn envelope_commitment(envelope: &V2ExternalWitnessPublicationEnvelope) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        V2_EXTERNAL_WITNESS_PUBLICATION_ENVELOPE_COMMITMENT_REVISION.as_bytes(),
    );
    encode_bytes(&mut bytes, &envelope.canonical_body_bytes());
    *blake3::hash(&bytes).as_bytes()
}

fn field<'a>(
    line: &'a str,
    expected_key: &'static str,
) -> Result<&'a str, V2ExternalWitnessPublicationEnvelopeError> {
    let Some((key, value)) = line.split_once('=') else {
        return Err(V2ExternalWitnessPublicationEnvelopeError::WrongField(expected_key));
    };
    if key != expected_key || value.is_empty() || value.contains('=') {
        return Err(V2ExternalWitnessPublicationEnvelopeError::WrongField(expected_key));
    }
    Ok(value)
}

fn canonical_u64(value: &str) -> Result<u64, V2ExternalWitnessPublicationEnvelopeError> {
    if value.is_empty()
        || !value.bytes().all(|byte| byte.is_ascii_digit())
        || (value.len() > 1 && value.starts_with('0'))
    {
        return Err(V2ExternalWitnessPublicationEnvelopeError::InvalidNumber);
    }
    let parsed = value
        .parse::<u64>()
        .map_err(|_| V2ExternalWitnessPublicationEnvelopeError::InvalidNumber)?;
    if parsed.to_string() != value {
        return Err(V2ExternalWitnessPublicationEnvelopeError::InvalidNumber);
    }
    Ok(parsed)
}

fn optional_commitment(
    value: &str,
) -> Result<Option<[u8; 32]>, V2ExternalWitnessPublicationEnvelopeError> {
    if value == "none" {
        Ok(None)
    } else {
        decode_hex32(value).map(Some)
    }
}

fn decode_hex32(value: &str) -> Result<[u8; 32], V2ExternalWitnessPublicationEnvelopeError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(V2ExternalWitnessPublicationEnvelopeError::InvalidHex);
    }
    let mut output = [0_u8; 32];
    for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_nibble(chunk[0]).ok_or(V2ExternalWitnessPublicationEnvelopeError::InvalidHex)?;
        let low = hex_nibble(chunk[1]).ok_or(V2ExternalWitnessPublicationEnvelopeError::InvalidHex)?;
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
    use crate::benchmarks::eureka::v2_authorized_anti_rollback_transition::advance_authority_with_signed_governance;
    use crate::benchmarks::eureka::v2_durably_authorized_transition::persist_authorized_authority_transition;
    use crate::benchmarks::eureka::v2_external_high_water_witness::V2ExternalHighWaterWitness;
    use crate::benchmarks::eureka::v2_external_high_water_witness_chain::verify_external_witness_successor;
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
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

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

    fn policy() -> V2QualifierSignerPolicy {
        let alpha = V2GovernanceSigner::from_hex(
            "fixture-transition-alpha@example.invalid",
            "ssh-ed25519",
            "98c9c52c03393af5fabb34d1a41a60a87073525c52f5fb7ca0a51c7de931da2d",
        )
        .unwrap();
        let beta = V2GovernanceSigner::from_hex(
            "fixture-transition-beta@example.invalid",
            "ssh-ed25519",
            "70535ce88cf2b7575de01e7e98559e95fa66d18dd7317f9411500f3930dcfbd7",
        )
        .unwrap();
        V2QualifierSignerPolicy::genesis(1, 2, vec![alpha, beta]).unwrap()
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

    fn genesis_checkpoint() -> V2QualifierAntiRollbackCheckpoint {
        let authority = V2QualifierAuthorityRecord::genesis(1, profile('c', 'd')).unwrap();
        let lineage = V2QualifierAuthorityLineage::activate_genesis(authority).unwrap();
        let policy = policy();
        V2QualifierAntiRollbackGuard::activate_genesis(&lineage, &policy)
            .unwrap()
            .checkpoint()
            .clone()
    }

    #[test]
    fn genesis_envelope_is_canonical_non_external_and_payload_bound() {
        let checkpoint = genesis_checkpoint();
        let witness = V2ExternalHighWaterWitness::genesis(&checkpoint).unwrap();
        let envelope = V2ExternalWitnessPublicationEnvelope::genesis(&witness, &checkpoint).unwrap();
        envelope.verify_bound_payload(&witness, &checkpoint, None).unwrap();

        assert_eq!(envelope.publication_sequence(), 1);
        assert_eq!(envelope.predecessor_publication_commitment(), None);
        assert_eq!(envelope.witness_commitment(), witness.commitment());
        assert_eq!(envelope.checkpoint_commitment(), checkpoint.commitment());
        assert_eq!(
            V2ExternalWitnessPublicationEnvelope::parse_canonical(&envelope.canonical_bytes()).unwrap(),
            envelope
        );
        let text = String::from_utf8(envelope.canonical_bytes()).unwrap();
        assert!(text.contains("externality_verified=false\n"));
        assert!(text.contains("execution_authority_granted=false\n"));
    }

    #[cfg(unix)]
    #[test]
    fn successor_envelope_binds_previous_payload_envelope_witness_checkpoint_and_proof() {
        let predecessor = V2QualifierAuthorityRecord::genesis(1, profile('c', 'd')).unwrap();
        let successor = V2QualifierAuthorityRecord::rotate(
            &predecessor,
            predecessor.commitment(),
            2,
            profile('e', 'd'),
        )
        .unwrap();
        let lineage = V2QualifierAuthorityLineage::activate_genesis(predecessor.clone()).unwrap();
        let policy = policy();
        let guard = V2QualifierAntiRollbackGuard::activate_genesis(&lineage, &policy).unwrap();
        let genesis_checkpoint = guard.checkpoint().clone();
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

        let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "symthaea-eureka-v2-publication-envelope-{}-{id}",
            std::process::id()
        ));
        fs::create_dir(&dir).unwrap();
        let store = V2QualifierCheckpointStore::new(dir.join("high-water.env")).unwrap();
        store.persist_genesis(&genesis_checkpoint).unwrap();
        let durable = persist_authorized_authority_transition(&store, transition).unwrap();

        let genesis_witness = V2ExternalHighWaterWitness::genesis(&genesis_checkpoint).unwrap();
        let genesis_envelope =
            V2ExternalWitnessPublicationEnvelope::genesis(&genesis_witness, &genesis_checkpoint)
                .unwrap();
        let successor_witness =
            V2ExternalHighWaterWitness::successor_authority(&genesis_witness, &durable).unwrap();
        let proof = verify_external_witness_successor(
            &genesis_witness,
            &successor_witness,
            &successor_checkpoint,
        )
        .unwrap();

        let successor_envelope = V2ExternalWitnessPublicationEnvelope::successor(
            &genesis_envelope,
            &genesis_witness,
            &genesis_checkpoint,
            None,
            &successor_witness,
            &successor_checkpoint,
            &proof,
        )
        .unwrap();
        successor_envelope
            .verify_bound_payload(&successor_witness, &successor_checkpoint, Some(&proof))
            .unwrap();

        assert_eq!(successor_envelope.publication_sequence(), 2);
        assert_eq!(successor_envelope.witness_sequence(), 2);
        assert_eq!(
            successor_envelope.predecessor_publication_commitment(),
            Some(genesis_envelope.commitment())
        );
        assert_eq!(successor_envelope.successor_proof_commitment(), Some(proof.commitment()));
        assert_eq!(
            successor_envelope.durable_transition_commitment(),
            successor_witness.durable_transition_commitment()
        );
        assert_eq!(
            V2ExternalWitnessPublicationEnvelope::parse_canonical(
                &successor_envelope.canonical_bytes()
            )
            .unwrap(),
            successor_envelope
        );

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn parser_rejects_externality_execution_escalation_sequence_drift_and_noncanonical_numbers() {
        let checkpoint = genesis_checkpoint();
        let witness = V2ExternalHighWaterWitness::genesis(&checkpoint).unwrap();
        let envelope = V2ExternalWitnessPublicationEnvelope::genesis(&witness, &checkpoint).unwrap();
        let text = String::from_utf8(envelope.canonical_bytes()).unwrap();

        let external = text.replacen("externality_verified=false", "externality_verified=true", 1);
        assert_eq!(
            V2ExternalWitnessPublicationEnvelope::parse_canonical(external.as_bytes()).unwrap_err(),
            V2ExternalWitnessPublicationEnvelopeError::ExternalityClaimed
        );
        let execution = text.replacen(
            "execution_authority_granted=false",
            "execution_authority_granted=true",
            1,
        );
        assert_eq!(
            V2ExternalWitnessPublicationEnvelope::parse_canonical(execution.as_bytes()).unwrap_err(),
            V2ExternalWitnessPublicationEnvelopeError::ExecutionAuthorityClaimed
        );
        let sequence_drift = text.replacen("witness_sequence=1", "witness_sequence=2", 1);
        assert_eq!(
            V2ExternalWitnessPublicationEnvelope::parse_canonical(sequence_drift.as_bytes())
                .unwrap_err(),
            V2ExternalWitnessPublicationEnvelopeError::InvalidEnvelopeShape
        );
        let leading_zero = text.replacen("publication_sequence=1", "publication_sequence=01", 1);
        assert_eq!(
            V2ExternalWitnessPublicationEnvelope::parse_canonical(leading_zero.as_bytes())
                .unwrap_err(),
            V2ExternalWitnessPublicationEnvelopeError::InvalidNumber
        );
    }

    #[test]
    fn publication_envelope_source_has_no_network_provider_signing_or_execution_surface() {
        let production = include_str!("v2_external_witness_publication_envelope.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for forbidden in [
            "reqwest",
            "rekor",
            "sigstore",
            "SystemTime",
            "UNIX_EPOCH",
            "ssh-keygen -Y sign",
            "PRIVATE KEY",
            "V2AdmittedQualifierRoot",
            "V2CanaryAuthorization",
            "predict_ticket",
            "reveal(",
            "score_consequence",
            "externality_verified=true",
            "execution_authority_granted=true",
            "allow(clippy::too_many_arguments)",
            "panic!(\"previous checkpoint bytes",
        ] {
            assert!(
                !production.contains(forbidden),
                "forbidden publication-envelope surface: {forbidden}"
            );
        }
        assert!(production.contains("externality_verified\", \"false"));
        assert!(production.contains("execution_authority_granted\", \"false"));
    }
}
