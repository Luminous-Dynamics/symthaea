// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Successor-proof mechanics for EUREKA-002 V2 external high-water witnesses.
//!
//! A canonical witness statement commits to the exact checkpoint identity but
//! intentionally does not duplicate the checkpoint's full canonical bytes.
//! External acceptance must therefore retain/provide the exact checkpoint bytes
//! (or an equivalent strict parsed checkpoint) and replay this proof against the
//! previous accepted witness.

#![allow(dead_code)]

use super::v2_external_high_water_witness::V2ExternalHighWaterWitness;
use super::v2_qualifier_anti_rollback::V2QualifierAntiRollbackCheckpoint;

pub(super) const V2_EXTERNAL_HIGH_WATER_SUCCESSOR_PROOF_REVISION: &str =
    "EUREKA.002.V2.EXTERNAL_HIGH_WATER_SUCCESSOR_PROOF.v1";

#[derive(Debug)]
pub(super) struct V2ExternalHighWaterSuccessorProof {
    predecessor_witness_commitment: [u8; 32],
    successor_witness_commitment: [u8; 32],
    predecessor_checkpoint_commitment: [u8; 32],
    successor_checkpoint_commitment: [u8; 32],
    durable_transition_commitment: [u8; 32],
    commitment: [u8; 32],
}

impl V2ExternalHighWaterSuccessorProof {
    pub(super) const fn predecessor_witness_commitment(&self) -> [u8; 32] {
        self.predecessor_witness_commitment
    }

    pub(super) const fn successor_witness_commitment(&self) -> [u8; 32] {
        self.successor_witness_commitment
    }

    pub(super) const fn predecessor_checkpoint_commitment(&self) -> [u8; 32] {
        self.predecessor_checkpoint_commitment
    }

    pub(super) const fn successor_checkpoint_commitment(&self) -> [u8; 32] {
        self.successor_checkpoint_commitment
    }

    pub(super) const fn durable_transition_commitment(&self) -> [u8; 32] {
        self.durable_transition_commitment
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2ExternalHighWaterSuccessorProofError {
    WitnessSequenceMismatch,
    WitnessPredecessorMismatch,
    GenesisMismatch,
    MissingDurableTransition,
    CheckpointIdentityMismatch,
    CheckpointPredecessorMismatch,
    SequenceVectorMismatch,
    StateVectorMismatch,
    SequenceExhausted,
}

pub(super) fn verify_external_witness_successor(
    previous: &V2ExternalHighWaterWitness,
    successor: &V2ExternalHighWaterWitness,
    successor_checkpoint: &V2QualifierAntiRollbackCheckpoint,
) -> Result<V2ExternalHighWaterSuccessorProof, V2ExternalHighWaterSuccessorProofError> {
    let expected_witness_sequence = previous
        .witness_sequence()
        .checked_add(1)
        .ok_or(V2ExternalHighWaterSuccessorProofError::SequenceExhausted)?;
    if successor.witness_sequence() != expected_witness_sequence {
        return Err(V2ExternalHighWaterSuccessorProofError::WitnessSequenceMismatch);
    }
    if successor.predecessor_witness_commitment() != Some(previous.commitment()) {
        return Err(V2ExternalHighWaterSuccessorProofError::WitnessPredecessorMismatch);
    }
    if successor.genesis_authority_commitment() != previous.genesis_authority_commitment()
        || successor_checkpoint.genesis_authority_commitment()
            != previous.genesis_authority_commitment()
    {
        return Err(V2ExternalHighWaterSuccessorProofError::GenesisMismatch);
    }

    let durable_transition_commitment = successor
        .durable_transition_commitment()
        .ok_or(V2ExternalHighWaterSuccessorProofError::MissingDurableTransition)?;

    if successor.checkpoint_commitment() != successor_checkpoint.commitment()
        || successor.authority_sequence() != successor_checkpoint.authority_sequence()
        || successor.authority_commitment() != successor_checkpoint.authority_commitment()
        || successor.currentness_commitment() != successor_checkpoint.currentness_commitment()
        || successor.signer_policy_sequence() != successor_checkpoint.signer_policy_sequence()
        || successor.signer_policy_commitment()
            != successor_checkpoint.signer_policy_commitment()
    {
        return Err(V2ExternalHighWaterSuccessorProofError::CheckpointIdentityMismatch);
    }
    if successor_checkpoint.predecessor_checkpoint_commitment()
        != Some(previous.checkpoint_commitment())
    {
        return Err(V2ExternalHighWaterSuccessorProofError::CheckpointPredecessorMismatch);
    }

    let authority_delta = successor
        .authority_sequence()
        .checked_sub(previous.authority_sequence())
        .ok_or(V2ExternalHighWaterSuccessorProofError::SequenceVectorMismatch)?;
    let policy_delta = successor
        .signer_policy_sequence()
        .checked_sub(previous.signer_policy_sequence())
        .ok_or(V2ExternalHighWaterSuccessorProofError::SequenceVectorMismatch)?;
    if !matches!((authority_delta, policy_delta), (1, 0) | (0, 1)) {
        return Err(V2ExternalHighWaterSuccessorProofError::SequenceVectorMismatch);
    }

    let state_shape_valid = match (authority_delta, policy_delta) {
        (1, 0) => {
            successor.authority_commitment() != previous.authority_commitment()
                && successor.currentness_commitment() != previous.currentness_commitment()
                && successor.signer_policy_commitment() == previous.signer_policy_commitment()
        }
        (0, 1) => {
            successor.authority_commitment() == previous.authority_commitment()
                && successor.currentness_commitment() == previous.currentness_commitment()
                && successor.signer_policy_commitment() != previous.signer_policy_commitment()
        }
        _ => false,
    };
    if !state_shape_valid {
        return Err(V2ExternalHighWaterSuccessorProofError::StateVectorMismatch);
    }

    let mut proof = V2ExternalHighWaterSuccessorProof {
        predecessor_witness_commitment: previous.commitment(),
        successor_witness_commitment: successor.commitment(),
        predecessor_checkpoint_commitment: previous.checkpoint_commitment(),
        successor_checkpoint_commitment: successor_checkpoint.commitment(),
        durable_transition_commitment,
        commitment: [0_u8; 32],
    };
    proof.commitment = successor_proof_commitment(&proof);
    Ok(proof)
}

fn successor_proof_commitment(proof: &V2ExternalHighWaterSuccessorProof) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        V2_EXTERNAL_HIGH_WATER_SUCCESSOR_PROOF_REVISION.as_bytes(),
    );
    bytes.extend_from_slice(&proof.predecessor_witness_commitment);
    bytes.extend_from_slice(&proof.successor_witness_commitment);
    bytes.extend_from_slice(&proof.predecessor_checkpoint_commitment);
    bytes.extend_from_slice(&proof.successor_checkpoint_commitment);
    bytes.extend_from_slice(&proof.durable_transition_commitment);
    *blake3::hash(&bytes).as_bytes()
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

    #[cfg(unix)]
    #[test]
    fn parsed_witness_successor_is_bound_to_exact_checkpoint_chain_edge() {
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
            "symthaea-eureka-v2-witness-chain-{}-{id}",
            std::process::id()
        ));
        fs::create_dir(&dir).unwrap();
        let store = V2QualifierCheckpointStore::new(dir.join("high-water.env")).unwrap();
        store.persist_genesis(&genesis_checkpoint).unwrap();
        let durable = persist_authorized_authority_transition(&store, transition).unwrap();

        let previous = V2ExternalHighWaterWitness::genesis(&genesis_checkpoint).unwrap();
        let successor_witness =
            V2ExternalHighWaterWitness::successor_authority(&previous, &durable).unwrap();
        let reparsed = V2ExternalHighWaterWitness::parse_persisted(
            &successor_witness.persisted_bytes(),
        )
        .unwrap();

        let proof = verify_external_witness_successor(
            &previous,
            &reparsed,
            &successor_checkpoint,
        )
        .unwrap();
        assert_eq!(proof.predecessor_witness_commitment(), previous.commitment());
        assert_eq!(proof.successor_witness_commitment(), reparsed.commitment());
        assert_eq!(
            proof.predecessor_checkpoint_commitment(),
            genesis_checkpoint.commitment()
        );
        assert_eq!(
            proof.successor_checkpoint_commitment(),
            successor_checkpoint.commitment()
        );
        assert_eq!(proof.durable_transition_commitment(), durable.commitment());
        assert_ne!(proof.commitment(), [0_u8; 32]);

        let sibling = V2QualifierAuthorityRecord::rotate(
            &predecessor,
            predecessor.commitment(),
            2,
            profile('f', 'd'),
        )
        .unwrap();
        let sibling_lineage =
            V2QualifierAuthorityLineage::activate_genesis(predecessor).unwrap();
        let sibling_guard =
            V2QualifierAntiRollbackGuard::activate_genesis(&sibling_lineage, &policy).unwrap();
        let (sibling_guard, _) = sibling_guard
            .advance_authority(sibling_lineage, sibling, &policy)
            .unwrap();
        assert_eq!(
            verify_external_witness_successor(
                &previous,
                &reparsed,
                sibling_guard.checkpoint(),
            )
            .unwrap_err(),
            V2ExternalHighWaterSuccessorProofError::CheckpointIdentityMismatch
        );

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn successor_proof_source_has_no_publication_or_execution_authority() {
        let production = include_str!("v2_external_high_water_witness_chain.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for forbidden in [
            "reqwest",
            "SystemTime",
            "remove_file",
            "V2AdmittedQualifierRoot",
            "V2CanaryAuthorization",
            "predict_ticket",
            "reveal(",
            "score_consequence",
            "execution_authority_granted=true",
        ] {
            assert!(
                !production.contains(forbidden),
                "forbidden witness-chain surface: {forbidden}"
            );
        }
        assert!(production.contains("predecessor_checkpoint_commitment"));
        assert!(production.contains("V2_EXTERNAL_HIGH_WATER_SUCCESSOR_PROOF_REVISION"));
    }
}
