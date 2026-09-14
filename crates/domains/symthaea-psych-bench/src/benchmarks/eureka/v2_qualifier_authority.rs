// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pure trust-policy mechanics for EUREKA-002 V2 qualification evidence.
//!
//! This module intentionally contains no real admitted workflow/contract hash.
//! It qualifies authority-record construction, append-only rotation, and strict
//! receipt admission only. Trusted qualification evidence is still not an
//! execution permit.

#![allow(dead_code)]

use super::v2_qualification_receipt::{
    V2QualificationReceipt, V2_QUALIFICATION_CLAIM_SCOPE,
    V2_QUALIFICATION_COMMAND_CONTRACT_REVISION, V2_QUALIFICATION_RECEIPT_SCHEMA,
    V2_QUALIFICATION_REVISION,
};

pub(super) const V2_QUALIFIER_AUTHORITY_SCHEMA: &str =
    "EUREKA.002.V2.TRUSTED_QUALIFIER_AUTHORITY.v1";
pub(super) const V2_TRUSTED_QUALIFICATION_EVIDENCE_REVISION: &str =
    "EUREKA.002.V2.TRUSTED_QUALIFICATION_EVIDENCE.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2QualifierAuthorityProfile {
    receipt_schema_revision: String,
    qualification_revision: String,
    command_contract_revision: String,
    claim_scope: String,
    workflow_sha256: [u8; 32],
    command_contract_sha256: [u8; 32],
}

impl V2QualifierAuthorityProfile {
    pub(super) fn current_from_hex(
        workflow_sha256: &str,
        command_contract_sha256: &str,
    ) -> Result<Self, V2QualifierAuthorityError> {
        Ok(Self {
            receipt_schema_revision: V2_QUALIFICATION_RECEIPT_SCHEMA.to_string(),
            qualification_revision: V2_QUALIFICATION_REVISION.to_string(),
            command_contract_revision: V2_QUALIFICATION_COMMAND_CONTRACT_REVISION.to_string(),
            claim_scope: V2_QUALIFICATION_CLAIM_SCOPE.to_string(),
            workflow_sha256: decode_sha256(workflow_sha256)?,
            command_contract_sha256: decode_sha256(command_contract_sha256)?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2QualifierAuthorityRecord {
    sequence: u64,
    profile: V2QualifierAuthorityProfile,
    predecessor_commitment: Option<[u8; 32]>,
    commitment: [u8; 32],
}

impl V2QualifierAuthorityRecord {
    pub(super) fn genesis(
        claimed_sequence: u64,
        profile: V2QualifierAuthorityProfile,
    ) -> Result<Self, V2QualifierAuthorityError> {
        if claimed_sequence != 1 {
            return Err(V2QualifierAuthorityError::WrongSequence);
        }
        let commitment = authority_commitment(claimed_sequence, &profile, None);
        Ok(Self {
            sequence: claimed_sequence,
            profile,
            predecessor_commitment: None,
            commitment,
        })
    }

    pub(super) fn rotate(
        predecessor: &Self,
        claimed_predecessor_commitment: [u8; 32],
        claimed_sequence: u64,
        next_profile: V2QualifierAuthorityProfile,
    ) -> Result<Self, V2QualifierAuthorityError> {
        if claimed_predecessor_commitment != predecessor.commitment {
            return Err(V2QualifierAuthorityError::WrongPredecessor);
        }
        let expected_sequence = predecessor
            .sequence
            .checked_add(1)
            .ok_or(V2QualifierAuthorityError::SequenceExhausted)?;
        if claimed_sequence != expected_sequence {
            return Err(V2QualifierAuthorityError::WrongSequence);
        }
        if next_profile == predecessor.profile {
            return Err(V2QualifierAuthorityError::NoOpRotation);
        }

        let commitment = authority_commitment(
            claimed_sequence,
            &next_profile,
            Some(claimed_predecessor_commitment),
        );
        Ok(Self {
            sequence: claimed_sequence,
            profile: next_profile,
            predecessor_commitment: Some(claimed_predecessor_commitment),
            commitment,
        })
    }

    pub(super) fn verify_receipt(
        &self,
        receipt: &V2QualificationReceipt,
    ) -> Result<V2TrustedQualificationEvidence, V2QualifierAuthorityError> {
        if self.profile.receipt_schema_revision != V2_QUALIFICATION_RECEIPT_SCHEMA
            || self.profile.qualification_revision != V2_QUALIFICATION_REVISION
            || self.profile.command_contract_revision
                != V2_QUALIFICATION_COMMAND_CONTRACT_REVISION
            || self.profile.claim_scope != V2_QUALIFICATION_CLAIM_SCOPE
        {
            return Err(V2QualifierAuthorityError::UnsupportedAuthorityProfile);
        }

        let receipt_lock = decode_sha256(receipt.cargo_lock_sha256())?;
        let receipt_workflow = decode_sha256(receipt.workflow_sha256())?;
        let receipt_contract = decode_sha256(receipt.command_contract_sha256())?;
        if receipt_workflow != self.profile.workflow_sha256 {
            return Err(V2QualifierAuthorityError::WorkflowNotTrusted);
        }
        if receipt_contract != self.profile.command_contract_sha256 {
            return Err(V2QualifierAuthorityError::ContractNotTrusted);
        }

        let mut evidence = V2TrustedQualificationEvidence {
            authority_commitment: self.commitment,
            github_run_id: receipt.github_run_id(),
            github_run_attempt: receipt.github_run_attempt(),
            subject_head: receipt.subject_head().to_string(),
            subject_tree: receipt.subject_tree().to_string(),
            cargo_lock_sha256: receipt_lock,
            workflow_sha256: receipt_workflow,
            command_contract_sha256: receipt_contract,
            rustc_version: receipt.rustc_version().to_string(),
            cargo_version: receipt.cargo_version().to_string(),
            commitment: [0_u8; 32],
        };
        evidence.commitment = trusted_evidence_commitment(&evidence);
        Ok(evidence)
    }

    pub(super) const fn sequence(&self) -> u64 {
        self.sequence
    }

    pub(super) const fn predecessor_commitment(&self) -> Option<[u8; 32]> {
        self.predecessor_commitment
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

/// Trusted build/test/lint evidence only. Deliberately not Clone and contains no
/// canary/HeldOut execution capability. It retains every qualification fact the
/// downstream one-shot permit theorem needs, so permit issuance never needs to
/// bypass trust and re-accept a raw qualification receipt.
#[derive(Debug)]
pub(super) struct V2TrustedQualificationEvidence {
    authority_commitment: [u8; 32],
    github_run_id: u64,
    github_run_attempt: u32,
    subject_head: String,
    subject_tree: String,
    cargo_lock_sha256: [u8; 32],
    workflow_sha256: [u8; 32],
    command_contract_sha256: [u8; 32],
    rustc_version: String,
    cargo_version: String,
    commitment: [u8; 32],
}

impl V2TrustedQualificationEvidence {
    pub(super) const fn authority_commitment(&self) -> [u8; 32] {
        self.authority_commitment
    }

    pub(super) fn subject_head(&self) -> &str {
        &self.subject_head
    }

    pub(super) fn subject_tree(&self) -> &str {
        &self.subject_tree
    }

    pub(super) const fn cargo_lock_sha256(&self) -> [u8; 32] {
        self.cargo_lock_sha256
    }

    pub(super) fn rustc_version(&self) -> &str {
        &self.rustc_version
    }

    pub(super) fn cargo_version(&self) -> &str {
        &self.cargo_version
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2QualifierAuthorityError {
    InvalidHash,
    WrongSequence,
    SequenceExhausted,
    WrongPredecessor,
    NoOpRotation,
    UnsupportedAuthorityProfile,
    WorkflowNotTrusted,
    ContractNotTrusted,
}

fn authority_commitment(
    sequence: u64,
    profile: &V2QualifierAuthorityProfile,
    predecessor_commitment: Option<[u8; 32]>,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_QUALIFIER_AUTHORITY_SCHEMA.as_bytes());
    bytes.extend_from_slice(&sequence.to_le_bytes());
    encode_bytes(&mut bytes, profile.receipt_schema_revision.as_bytes());
    encode_bytes(&mut bytes, profile.qualification_revision.as_bytes());
    encode_bytes(&mut bytes, profile.command_contract_revision.as_bytes());
    encode_bytes(&mut bytes, profile.claim_scope.as_bytes());
    bytes.extend_from_slice(&profile.workflow_sha256);
    bytes.extend_from_slice(&profile.command_contract_sha256);
    match predecessor_commitment {
        Some(predecessor) => {
            bytes.push(1);
            bytes.extend_from_slice(&predecessor);
        }
        None => bytes.push(0),
    }
    *blake3::hash(&bytes).as_bytes()
}

fn trusted_evidence_commitment(evidence: &V2TrustedQualificationEvidence) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        V2_TRUSTED_QUALIFICATION_EVIDENCE_REVISION.as_bytes(),
    );
    bytes.extend_from_slice(&evidence.authority_commitment);
    bytes.extend_from_slice(&evidence.github_run_id.to_le_bytes());
    bytes.extend_from_slice(&evidence.github_run_attempt.to_le_bytes());
    encode_bytes(&mut bytes, evidence.subject_head.as_bytes());
    encode_bytes(&mut bytes, evidence.subject_tree.as_bytes());
    bytes.extend_from_slice(&evidence.cargo_lock_sha256);
    bytes.extend_from_slice(&evidence.workflow_sha256);
    bytes.extend_from_slice(&evidence.command_contract_sha256);
    encode_bytes(&mut bytes, evidence.rustc_version.as_bytes());
    encode_bytes(&mut bytes, evidence.cargo_version.as_bytes());
    *blake3::hash(&bytes).as_bytes()
}

fn decode_sha256(value: &str) -> Result<[u8; 32], V2QualifierAuthorityError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(V2QualifierAuthorityError::InvalidHash);
    }
    let mut output = [0_u8; 32];
    for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_nibble(chunk[0]).ok_or(V2QualifierAuthorityError::InvalidHash)?;
        let low = hex_nibble(chunk[1]).ok_or(V2QualifierAuthorityError::InvalidHash)?;
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

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(workflow: char, contract: char) -> V2QualifierAuthorityProfile {
        V2QualifierAuthorityProfile::current_from_hex(
            &workflow.to_string().repeat(64),
            &contract.to_string().repeat(64),
        )
        .unwrap()
    }

    fn valid_receipt(workflow: char, contract: char) -> V2QualificationReceipt {
        valid_receipt_with_details(
            workflow,
            contract,
            '2',
            'a',
            "rustc 1.96.0 fixture",
            "cargo 1.96.0 fixture",
        )
    }

    fn valid_receipt_with_details(
        workflow: char,
        contract: char,
        tree: char,
        lock: char,
        rustc_version: &str,
        cargo_version: &str,
    ) -> V2QualificationReceipt {
        let head = "1".repeat(40);
        let tree = tree.to_string().repeat(40);
        let lock = lock.to_string().repeat(64);
        let workflow = workflow.to_string().repeat(64);
        let contract = contract.to_string().repeat(64);
        let raw = format!(
            "receipt_schema_revision={V2_QUALIFICATION_RECEIPT_SCHEMA}\n\
qualification_revision={V2_QUALIFICATION_REVISION}\n\
command_contract_revision={V2_QUALIFICATION_COMMAND_CONTRACT_REVISION}\n\
repository=Luminous-Dynamics/symthaea\n\
event=pull_request\n\
github_run_id=42\n\
github_run_attempt=1\n\
github_workflow_ref=Luminous-Dynamics/symthaea/.github/workflows/eureka-v2-backend-qualification.yml@refs/pull/1/merge\n\
expected_subject_head={head}\n\
subject_head={head}\n\
subject_tree={tree}\n\
cargo_lock_sha256={lock}\n\
workflow_sha256={workflow}\n\
command_contract_sha256={contract}\n\
rustc_version={rustc_version}\n\
cargo_version={cargo_version}\n\
checkout_clean_before=true\n\
claim_scope={V2_QUALIFICATION_CLAIM_SCOPE}\n\
execution_authority_granted=false\n\
real_canary_executed=false\n\
heldout_executed=false\n\
confirmatory_evidence_minted=false\n\
postflight_head={head}\n\
postflight_tree={tree}\n\
postflight_cargo_lock_sha256={lock}\n\
postflight_workflow_sha256={workflow}\n\
postflight_command_contract_sha256={contract}\n\
checkout_clean_after=true\n\
qualification_result=PASS\n"
        );
        V2QualificationReceipt::parse_and_verify(&raw).unwrap()
    }

    #[test]
    fn fixture_genesis_is_deterministic_and_binds_hashes() {
        let first = V2QualifierAuthorityRecord::genesis(1, profile('b', 'c')).unwrap();
        let second = V2QualifierAuthorityRecord::genesis(1, profile('b', 'c')).unwrap();
        let changed_workflow = V2QualifierAuthorityRecord::genesis(1, profile('d', 'c')).unwrap();
        let changed_contract = V2QualifierAuthorityRecord::genesis(1, profile('b', 'd')).unwrap();
        assert_eq!(first.commitment(), second.commitment());
        assert_ne!(first.commitment(), changed_workflow.commitment());
        assert_ne!(first.commitment(), changed_contract.commitment());
        assert_ne!(first.commitment(), [0_u8; 32]);
    }

    #[test]
    fn malformed_hashes_and_wrong_genesis_sequence_fail_closed() {
        assert_eq!(
            V2QualifierAuthorityProfile::current_from_hex("ABC", &"c".repeat(64)),
            Err(V2QualifierAuthorityError::InvalidHash)
        );
        assert_eq!(
            V2QualifierAuthorityRecord::genesis(2, profile('b', 'c')),
            Err(V2QualifierAuthorityError::WrongSequence)
        );
    }

    #[test]
    fn rotation_requires_exact_predecessor_next_sequence_and_real_change() {
        let genesis = V2QualifierAuthorityRecord::genesis(1, profile('b', 'c')).unwrap();
        assert_eq!(
            V2QualifierAuthorityRecord::rotate(
                &genesis,
                genesis.commitment(),
                2,
                profile('b', 'c'),
            ),
            Err(V2QualifierAuthorityError::NoOpRotation)
        );
        assert_eq!(
            V2QualifierAuthorityRecord::rotate(&genesis, [9_u8; 32], 2, profile('d', 'c')),
            Err(V2QualifierAuthorityError::WrongPredecessor)
        );
        assert_eq!(
            V2QualifierAuthorityRecord::rotate(
                &genesis,
                genesis.commitment(),
                3,
                profile('d', 'c'),
            ),
            Err(V2QualifierAuthorityError::WrongSequence)
        );

        let successor = V2QualifierAuthorityRecord::rotate(
            &genesis,
            genesis.commitment(),
            2,
            profile('d', 'c'),
        )
        .unwrap();
        assert_eq!(successor.sequence(), 2);
        assert_eq!(successor.predecessor_commitment(), Some(genesis.commitment()));
    }

    #[test]
    fn rotation_sequence_exhaustion_fails_closed() {
        let exhausted_profile = profile('b', 'c');
        let predecessor_commitment = Some([7_u8; 32]);
        let exhausted = V2QualifierAuthorityRecord {
            sequence: u64::MAX,
            commitment: authority_commitment(
                u64::MAX,
                &exhausted_profile,
                predecessor_commitment,
            ),
            profile: exhausted_profile,
            predecessor_commitment,
        };

        assert_eq!(
            V2QualifierAuthorityRecord::rotate(
                &exhausted,
                exhausted.commitment(),
                u64::MAX,
                profile('d', 'c'),
            ),
            Err(V2QualifierAuthorityError::SequenceExhausted)
        );
    }

    #[test]
    fn sibling_rotations_from_one_predecessor_are_observable_forks() {
        let genesis = V2QualifierAuthorityRecord::genesis(1, profile('b', 'c')).unwrap();
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
        assert_ne!(left.commitment(), right.commitment());
        assert_eq!(left.predecessor_commitment(), right.predecessor_commitment());
    }

    #[test]
    fn trusted_receipt_admission_requires_exact_authority_hashes() {
        let authority = V2QualifierAuthorityRecord::genesis(1, profile('b', 'c')).unwrap();
        let receipt = valid_receipt('b', 'c');
        let evidence = authority.verify_receipt(&receipt).unwrap();
        assert_eq!(evidence.authority_commitment(), authority.commitment());
        assert_eq!(evidence.subject_head(), "1".repeat(40));
        assert_eq!(evidence.subject_tree(), "2".repeat(40));
        assert_eq!(evidence.cargo_lock_sha256(), [0xaa_u8; 32]);
        assert_eq!(evidence.rustc_version(), "rustc 1.96.0 fixture");
        assert_eq!(evidence.cargo_version(), "cargo 1.96.0 fixture");
        assert_ne!(evidence.commitment(), [0_u8; 32]);

        let wrong_workflow = V2QualifierAuthorityRecord::genesis(1, profile('d', 'c')).unwrap();
        assert_eq!(
            wrong_workflow.verify_receipt(&receipt).unwrap_err(),
            V2QualifierAuthorityError::WorkflowNotTrusted
        );
        let wrong_contract = V2QualifierAuthorityRecord::genesis(1, profile('b', 'd')).unwrap();
        assert_eq!(
            wrong_contract.verify_receipt(&receipt).unwrap_err(),
            V2QualifierAuthorityError::ContractNotTrusted
        );
    }

    #[test]
    fn trusted_evidence_commitment_binds_permit_relevant_qualification_facts() {
        let authority = V2QualifierAuthorityRecord::genesis(1, profile('b', 'c')).unwrap();
        let baseline = authority
            .verify_receipt(&valid_receipt('b', 'c'))
            .unwrap();
        let changed_tree = authority
            .verify_receipt(&valid_receipt_with_details(
                'b',
                'c',
                '3',
                'a',
                "rustc 1.96.0 fixture",
                "cargo 1.96.0 fixture",
            ))
            .unwrap();
        let changed_lock = authority
            .verify_receipt(&valid_receipt_with_details(
                'b',
                'c',
                '2',
                'd',
                "rustc 1.96.0 fixture",
                "cargo 1.96.0 fixture",
            ))
            .unwrap();
        let changed_rustc = authority
            .verify_receipt(&valid_receipt_with_details(
                'b',
                'c',
                '2',
                'a',
                "rustc 1.96.1 fixture",
                "cargo 1.96.0 fixture",
            ))
            .unwrap();
        let changed_cargo = authority
            .verify_receipt(&valid_receipt_with_details(
                'b',
                'c',
                '2',
                'a',
                "rustc 1.96.0 fixture",
                "cargo 1.96.1 fixture",
            ))
            .unwrap();

        for changed in [changed_tree, changed_lock, changed_rustc, changed_cargo] {
            assert_ne!(baseline.commitment(), changed.commitment());
        }
    }

    #[test]
    fn same_receipt_under_distinct_valid_authority_lineages_has_distinct_evidence_identity() {
        let genesis = V2QualifierAuthorityRecord::genesis(1, profile('b', 'c')).unwrap();
        let away = V2QualifierAuthorityRecord::rotate(
            &genesis,
            genesis.commitment(),
            2,
            profile('d', 'c'),
        )
        .unwrap();
        let returned = V2QualifierAuthorityRecord::rotate(
            &away,
            away.commitment(),
            3,
            profile('b', 'c'),
        )
        .unwrap();
        let receipt = valid_receipt('b', 'c');

        let genesis_evidence = genesis.verify_receipt(&receipt).unwrap();
        let returned_evidence = returned.verify_receipt(&receipt).unwrap();

        assert_ne!(genesis.commitment(), returned.commitment());
        assert_ne!(
            genesis_evidence.authority_commitment(),
            returned_evidence.authority_commitment()
        );
        assert_ne!(genesis_evidence.commitment(), returned_evidence.commitment());
    }

    #[test]
    fn authority_source_has_no_execution_capability_or_real_trust_root() {
        let production = include_str!("v2_qualifier_authority.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for forbidden in [
            "predict_ticket",
            "reveal(",
            "score_consequence",
            "V2CanaryAuthorization",
            "V2RealHeldOutPairedFreeze",
            "const TRUSTED_WORKFLOW",
            "const TRUSTED_CONTRACT",
        ] {
            assert!(!production.contains(forbidden), "forbidden authority surface: {forbidden}");
        }
    }
}
