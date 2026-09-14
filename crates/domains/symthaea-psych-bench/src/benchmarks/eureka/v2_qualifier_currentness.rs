// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Active-head/currentness mechanics for EUREKA-002 V2 qualifier authority.
//!
//! A structurally valid authority record may remain useful historical evidence
//! after rotation. It must not thereby remain current authority for elevating a
//! new qualification receipt. This module separates lineage/currentness from
//! record validity without admitting any real qualifier root.

#![allow(dead_code)]

use super::v2_qualification_receipt::V2QualificationReceipt;
use super::v2_qualifier_authority::{
    V2QualifierAuthorityError, V2QualifierAuthorityRecord, V2TrustedQualificationEvidence,
};

pub(super) const V2_QUALIFIER_CURRENTNESS_REVISION: &str =
    "EUREKA.002.V2.QUALIFIER_AUTHORITY_CURRENTNESS.v1";
pub(super) const V2_CURRENT_TRUSTED_QUALIFICATION_EVIDENCE_REVISION: &str =
    "EUREKA.002.V2.CURRENT_TRUSTED_QUALIFICATION_EVIDENCE.v1";

/// Move-only active authority state.
///
/// The generic mechanism contains no real root. A future root-admission tranche
/// must supply the admitted genesis record used to create the real lineage.
#[derive(Debug)]
pub(super) struct V2QualifierAuthorityLineage {
    genesis_commitment: [u8; 32],
    current: V2QualifierAuthorityRecord,
    commitment: [u8; 32],
}

impl V2QualifierAuthorityLineage {
    /// Activate a structurally valid genesis as the active head of a lineage.
    ///
    /// This method establishes *mechanical* currentness only. Supplying the real
    /// genesis remains the separate human/governance decision in the root-
    /// admission tranche.
    pub(super) fn activate_genesis(
        genesis: V2QualifierAuthorityRecord,
    ) -> Result<Self, V2QualifierCurrentnessError> {
        if genesis.sequence() != 1 || genesis.predecessor_commitment().is_some() {
            return Err(V2QualifierCurrentnessError::NotGenesis);
        }

        let genesis_commitment = genesis.commitment();
        let commitment = lineage_commitment(
            genesis_commitment,
            genesis.sequence(),
            genesis.commitment(),
            None,
        );
        Ok(Self {
            genesis_commitment,
            current: genesis,
            commitment,
        })
    }

    /// Advance the active head exactly once from the exact current predecessor.
    ///
    /// A sibling successor constructed from an older predecessor remains an
    /// observable historical fork candidate, but cannot become active through
    /// this transition once the lineage has already advanced.
    pub(super) fn advance(
        self,
        successor: V2QualifierAuthorityRecord,
    ) -> Result<Self, V2QualifierCurrentnessError> {
        if successor.predecessor_commitment() != Some(self.current.commitment()) {
            return Err(V2QualifierCurrentnessError::StalePredecessor);
        }

        let expected_sequence = self
            .current
            .sequence()
            .checked_add(1)
            .ok_or(V2QualifierCurrentnessError::SequenceExhausted)?;
        if successor.sequence() != expected_sequence {
            return Err(V2QualifierCurrentnessError::WrongSequence);
        }

        let commitment = lineage_commitment(
            self.genesis_commitment,
            successor.sequence(),
            successor.commitment(),
            Some(self.commitment),
        );
        Ok(Self {
            genesis_commitment: self.genesis_commitment,
            current: successor,
            commitment,
        })
    }

    /// Elevate a receipt only through the exact currently active authority.
    ///
    /// The returned wrapper is deliberately distinct from ordinary trusted
    /// qualification evidence. A future permit issuer should consume this
    /// currentness-bound type (or an exact serialized equivalent) and compare
    /// its currentness identity to the then-active admitted lineage.
    pub(super) fn verify_current_receipt(
        &self,
        receipt: &V2QualificationReceipt,
    ) -> Result<V2CurrentTrustedQualificationEvidence, V2QualifierAuthorityError> {
        let evidence = self.current.verify_receipt(receipt)?;
        let commitment = current_evidence_commitment(self.commitment, &evidence);
        Ok(V2CurrentTrustedQualificationEvidence {
            currentness_commitment: self.commitment,
            evidence,
            commitment,
        })
    }

    pub(super) const fn current_sequence(&self) -> u64 {
        self.current.sequence()
    }

    pub(super) const fn current_authority_commitment(&self) -> [u8; 32] {
        self.current.commitment()
    }

    pub(super) const fn genesis_commitment(&self) -> [u8; 32] {
        self.genesis_commitment
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

/// Trusted qualification evidence plus proof of the active-head state through
/// which it was elevated. This remains build/test/lint evidence only; it is not
/// an execution permit.
#[derive(Debug)]
pub(super) struct V2CurrentTrustedQualificationEvidence {
    currentness_commitment: [u8; 32],
    evidence: V2TrustedQualificationEvidence,
    commitment: [u8; 32],
}

impl V2CurrentTrustedQualificationEvidence {
    pub(super) const fn currentness_commitment(&self) -> [u8; 32] {
        self.currentness_commitment
    }

    pub(super) const fn authority_commitment(&self) -> [u8; 32] {
        self.evidence.authority_commitment()
    }

    pub(super) const fn qualification_evidence_commitment(&self) -> [u8; 32] {
        self.evidence.commitment()
    }

    pub(super) fn subject_head(&self) -> &str {
        self.evidence.subject_head()
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2QualifierCurrentnessError {
    NotGenesis,
    StalePredecessor,
    WrongSequence,
    SequenceExhausted,
}

fn lineage_commitment(
    genesis_commitment: [u8; 32],
    active_sequence: u64,
    active_authority_commitment: [u8; 32],
    previous_lineage_commitment: Option<[u8; 32]>,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_QUALIFIER_CURRENTNESS_REVISION.as_bytes());
    bytes.extend_from_slice(&genesis_commitment);
    bytes.extend_from_slice(&active_sequence.to_le_bytes());
    bytes.extend_from_slice(&active_authority_commitment);
    match previous_lineage_commitment {
        Some(previous) => {
            bytes.push(1);
            bytes.extend_from_slice(&previous);
        }
        None => bytes.push(0),
    }
    *blake3::hash(&bytes).as_bytes()
}

fn current_evidence_commitment(
    currentness_commitment: [u8; 32],
    evidence: &V2TrustedQualificationEvidence,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        V2_CURRENT_TRUSTED_QUALIFICATION_EVIDENCE_REVISION.as_bytes(),
    );
    bytes.extend_from_slice(&currentness_commitment);
    bytes.extend_from_slice(&evidence.authority_commitment());
    bytes.extend_from_slice(&evidence.commitment());
    *blake3::hash(&bytes).as_bytes()
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarks::eureka::v2_qualification_receipt::{
        V2_QUALIFICATION_CLAIM_SCOPE, V2_QUALIFICATION_COMMAND_CONTRACT_REVISION,
        V2_QUALIFICATION_RECEIPT_SCHEMA, V2_QUALIFICATION_REVISION,
    };
    use crate::benchmarks::eureka::v2_qualifier_authority::V2QualifierAuthorityProfile;

    fn profile(workflow: char, contract: char) -> V2QualifierAuthorityProfile {
        V2QualifierAuthorityProfile::current_from_hex(
            &workflow.to_string().repeat(64),
            &contract.to_string().repeat(64),
        )
        .unwrap()
    }

    fn receipt(workflow: char, contract: char) -> V2QualificationReceipt {
        let head = "1".repeat(40);
        let tree = "2".repeat(40);
        let lock = "a".repeat(64);
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
rustc_version=rustc 1.96.0 fixture\n\
cargo_version=cargo 1.96.0 fixture\n\
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
    fn genesis_activation_is_deterministic_and_nonzero() {
        let first = V2QualifierAuthorityLineage::activate_genesis(
            V2QualifierAuthorityRecord::genesis(1, profile('b', 'c')).unwrap(),
        )
        .unwrap();
        let second = V2QualifierAuthorityLineage::activate_genesis(
            V2QualifierAuthorityRecord::genesis(1, profile('b', 'c')).unwrap(),
        )
        .unwrap();

        assert_eq!(first.current_sequence(), 1);
        assert_eq!(first.genesis_commitment(), first.current_authority_commitment());
        assert_eq!(first.commitment(), second.commitment());
        assert_ne!(first.commitment(), [0_u8; 32]);
    }

    #[test]
    fn non_genesis_record_cannot_be_activated_as_lineage_root() {
        let genesis = V2QualifierAuthorityRecord::genesis(1, profile('b', 'c')).unwrap();
        let successor = V2QualifierAuthorityRecord::rotate(
            &genesis,
            genesis.commitment(),
            2,
            profile('d', 'c'),
        )
        .unwrap();

        assert_eq!(
            V2QualifierAuthorityLineage::activate_genesis(successor).unwrap_err(),
            V2QualifierCurrentnessError::NotGenesis
        );
    }

    #[test]
    fn accepted_successor_stales_sibling_fork_for_current_activation() {
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

        let lineage = V2QualifierAuthorityLineage::activate_genesis(genesis).unwrap();
        let old_lineage_commitment = lineage.commitment();
        let lineage = lineage.advance(left).unwrap();
        assert_eq!(lineage.current_sequence(), 2);
        assert_ne!(lineage.commitment(), old_lineage_commitment);
        assert_eq!(
            lineage.advance(right).unwrap_err(),
            V2QualifierCurrentnessError::StalePredecessor
        );
    }

    #[test]
    fn current_receipt_elevation_uses_only_active_authority() {
        let genesis = V2QualifierAuthorityRecord::genesis(1, profile('b', 'c')).unwrap();
        let successor = V2QualifierAuthorityRecord::rotate(
            &genesis,
            genesis.commitment(),
            2,
            profile('d', 'c'),
        )
        .unwrap();
        let lineage = V2QualifierAuthorityLineage::activate_genesis(genesis)
            .unwrap()
            .advance(successor)
            .unwrap();

        assert_eq!(
            lineage.verify_current_receipt(&receipt('b', 'c')).unwrap_err(),
            V2QualifierAuthorityError::WorkflowNotTrusted
        );
        let evidence = lineage.verify_current_receipt(&receipt('d', 'c')).unwrap();
        assert_eq!(
            evidence.authority_commitment(),
            lineage.current_authority_commitment()
        );
        assert_eq!(evidence.currentness_commitment(), lineage.commitment());
        assert_eq!(evidence.subject_head(), "1".repeat(40));
        assert_ne!(evidence.qualification_evidence_commitment(), [0_u8; 32]);
        assert_ne!(evidence.commitment(), [0_u8; 32]);
    }

    #[test]
    fn same_receipt_after_return_to_profile_gets_new_currentness_identity() {
        let genesis = V2QualifierAuthorityRecord::genesis(1, profile('b', 'c')).unwrap();
        let genesis_lineage = V2QualifierAuthorityLineage::activate_genesis(
            V2QualifierAuthorityRecord::genesis(1, profile('b', 'c')).unwrap(),
        )
        .unwrap();
        let genesis_evidence = genesis_lineage
            .verify_current_receipt(&receipt('b', 'c'))
            .unwrap();

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
        let returned_lineage = V2QualifierAuthorityLineage::activate_genesis(genesis)
            .unwrap()
            .advance(away)
            .unwrap()
            .advance(returned)
            .unwrap();
        let returned_evidence = returned_lineage
            .verify_current_receipt(&receipt('b', 'c'))
            .unwrap();

        assert_ne!(
            genesis_evidence.currentness_commitment(),
            returned_evidence.currentness_commitment()
        );
        assert_ne!(genesis_evidence.commitment(), returned_evidence.commitment());
    }

    #[test]
    fn currentness_source_has_no_execution_surface_or_real_root() {
        let source = include_str!("v2_qualifier_currentness.rs")
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
            "const TRUSTED_ENVIRONMENT",
        ] {
            assert!(!source.contains(forbidden), "forbidden currentness surface: {forbidden}");
        }
    }
}
