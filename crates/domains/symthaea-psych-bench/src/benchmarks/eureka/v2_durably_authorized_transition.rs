// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Durable composition of signed-governance-authorized authority transitions.
//!
//! `V2AuthorizedAuthorityTransition` proves governance/currentness composition
//! but is still only in-memory. This module requires the #3148 crash-conscious
//! store to commit that transition's exact predecessor -> successor checkpoint
//! pair before producing a stronger durable-transition object.

#![allow(dead_code)]

use super::v2_authorized_anti_rollback_transition::V2AuthorizedAuthorityTransition;
use super::v2_qualifier_checkpoint_store::{
    V2CheckpointStoreError, V2DurableCheckpointCommitReceipt, V2QualifierCheckpointStore,
};

pub(super) const V2_DURABLY_AUTHORIZED_AUTHORITY_TRANSITION_REVISION: &str =
    "EUREKA.002.V2.DURABLY_AUTHORIZED_AUTHORITY_TRANSITION.v1";

/// Signed-governance-authorized transition whose exact successor checkpoint has
/// crossed the crash-conscious durable-store boundary.
///
/// This still is not an admitted root or execution permit. The durable object
/// proves composition of governance authorization + currentness + local durable
/// high-water storage only.
#[derive(Debug)]
pub(super) struct V2DurablyAuthorizedAuthorityTransition {
    transition: V2AuthorizedAuthorityTransition,
    durable_receipt: V2DurableCheckpointCommitReceipt,
    commitment: [u8; 32],
}

impl V2DurablyAuthorizedAuthorityTransition {
    pub(super) fn transition(&self) -> &V2AuthorizedAuthorityTransition {
        &self.transition
    }

    pub(super) fn durable_receipt(&self) -> &V2DurableCheckpointCommitReceipt {
        &self.durable_receipt
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2DurablyAuthorizedTransitionError {
    Store(V2CheckpointStoreError),
    TransitionShapeMismatch,
    DurableReceiptMismatch,
    DurableReloadMismatch,
}

/// Persist exactly the checkpoint pair already bound into a signed-governance
/// authorized transition.
pub(super) fn persist_authorized_authority_transition(
    store: &V2QualifierCheckpointStore,
    transition: V2AuthorizedAuthorityTransition,
) -> Result<V2DurablyAuthorizedAuthorityTransition, V2DurablyAuthorizedTransitionError> {
    let predecessor = transition.predecessor_checkpoint();
    let successor = transition.successor_checkpoint();
    if successor.predecessor_checkpoint_commitment() != Some(predecessor.commitment())
        || successor.commitment() == predecessor.commitment()
        || transition.authorization().predecessor_authority_commitment()
            != predecessor.authority_commitment()
        || transition.authorization().predecessor_currentness_commitment()
            != predecessor.currentness_commitment()
        || transition.authorization().authority_sequence() != successor.authority_sequence()
        || transition.authorization().authority_commitment() != successor.authority_commitment()
    {
        return Err(V2DurablyAuthorizedTransitionError::TransitionShapeMismatch);
    }

    let transition_commitment = transition.commitment();
    let authorization_commitment = transition.authorization().commitment();
    let receipt = store
        .persist_successor(predecessor, successor)
        .map_err(V2DurablyAuthorizedTransitionError::Store)?;

    if receipt.predecessor_checkpoint_commitment() != Some(predecessor.commitment())
        || receipt.checkpoint_commitment() != successor.commitment()
    {
        return Err(V2DurablyAuthorizedTransitionError::DurableReceiptMismatch);
    }

    let durable = store
        .load()
        .map_err(V2DurablyAuthorizedTransitionError::Store)?
        .ok_or(V2DurablyAuthorizedTransitionError::DurableReloadMismatch)?;
    if &durable != successor {
        return Err(V2DurablyAuthorizedTransitionError::DurableReloadMismatch);
    }

    let mut result = V2DurablyAuthorizedAuthorityTransition {
        transition,
        durable_receipt: receipt,
        commitment: [0_u8; 32],
    };
    result.commitment = durable_transition_commitment(
        &result,
        transition_commitment,
        authorization_commitment,
    );
    Ok(result)
}

fn durable_transition_commitment(
    durable: &V2DurablyAuthorizedAuthorityTransition,
    transition_commitment: [u8; 32],
    authorization_commitment: [u8; 32],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        V2_DURABLY_AUTHORIZED_AUTHORITY_TRANSITION_REVISION.as_bytes(),
    );
    bytes.extend_from_slice(&transition_commitment);
    bytes.extend_from_slice(&authorization_commitment);
    bytes.extend_from_slice(&durable.transition.predecessor_checkpoint().commitment());
    bytes.extend_from_slice(&durable.transition.successor_checkpoint().commitment());
    bytes.extend_from_slice(&durable.durable_receipt.commitment());
    bytes.extend_from_slice(&durable.durable_receipt.persisted_byte_len().to_le_bytes());
    *blake3::hash(&bytes).as_bytes()
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::super::v2_authorized_anti_rollback_transition::{
        V2AuthorizedAuthorityTransition, advance_authority_with_signed_governance,
    };
    use super::super::v2_openssh_admission_verifier::V2OpenSshAdmissionSignature;
    use super::super::v2_qualifier_anti_rollback::V2QualifierAntiRollbackGuard;
    use super::super::v2_qualifier_authority::{
        V2QualifierAuthorityProfile, V2QualifierAuthorityRecord,
    };
    use super::super::v2_qualifier_currentness::V2QualifierAuthorityLineage;
    use super::super::v2_qualifier_signer_policy::{
        V2GovernanceSigner, V2QualifierSignerPolicy,
    };
    use super::super::v2_signed_governance_authorization::authorize_signed_authority_rotation;
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);
    const MANIFEST: &[u8] = include_bytes!("fixtures/eureka-v2-signed-transition-rotation.env");
    const ALPHA_PUB: &str = include_str!("fixtures/eureka-v2-signed-transition-alpha.pub");
    const BETA_PUB: &str = include_str!("fixtures/eureka-v2-signed-transition-beta.pub");
    const ALPHA_SIG: &[u8] = include_bytes!("fixtures/eureka-v2-signed-transition-alpha.sig");
    const BETA_SIG: &[u8] = include_bytes!("fixtures/eureka-v2-signed-transition-beta.sig");
    const ALPHA_PRINCIPAL: &str = "fixture-transition-alpha@example.invalid";
    const BETA_PRINCIPAL: &str = "fixture-transition-beta@example.invalid";
    const ALPHA_KEY_SHA256: &str =
        "98c9c52c03393af5fabb34d1a41a60a87073525c52f5fb7ca0a51c7de931da2d";
    const BETA_KEY_SHA256: &str =
        "70535ce88cf2b7575de01e7e98559e95fa66d18dd7317f9411500f3930dcfbd7";

    fn policy() -> V2QualifierSignerPolicy {
        let alpha = V2GovernanceSigner::from_hex(
            ALPHA_PRINCIPAL,
            "ssh-ed25519",
            ALPHA_KEY_SHA256,
        )
        .unwrap();
        let beta = V2GovernanceSigner::from_hex(
            BETA_PRINCIPAL,
            "ssh-ed25519",
            BETA_KEY_SHA256,
        )
        .unwrap();
        V2QualifierSignerPolicy::genesis(1, 2, vec![beta, alpha]).unwrap()
    }

    fn signatures() -> [V2OpenSshAdmissionSignature<'static>; 2] {
        [
            V2OpenSshAdmissionSignature::new(ALPHA_PRINCIPAL, ALPHA_PUB, ALPHA_SIG),
            V2OpenSshAdmissionSignature::new(BETA_PRINCIPAL, BETA_PUB, BETA_SIG),
        ]
    }

    fn predecessor() -> V2QualifierAuthorityRecord {
        V2QualifierAuthorityRecord::genesis(
            1,
            V2QualifierAuthorityProfile::current_from_hex(&"c".repeat(64), &"d".repeat(64))
                .unwrap(),
        )
        .unwrap()
    }

    fn successor(predecessor: &V2QualifierAuthorityRecord) -> V2QualifierAuthorityRecord {
        V2QualifierAuthorityRecord::rotate(
            predecessor,
            predecessor.commitment(),
            2,
            V2QualifierAuthorityProfile::current_from_hex(&"e".repeat(64), &"d".repeat(64))
                .unwrap(),
        )
        .unwrap()
    }

    fn authorized_transition() -> V2AuthorizedAuthorityTransition {
        let predecessor = predecessor();
        let successor = successor(&predecessor);
        let lineage = V2QualifierAuthorityLineage::activate_genesis(predecessor.clone()).unwrap();
        let policy = policy();
        let guard = V2QualifierAntiRollbackGuard::activate_genesis(&lineage, &policy).unwrap();
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
        transition
    }

    fn test_store() -> (PathBuf, V2QualifierCheckpointStore) {
        let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "symthaea-eureka-v2-durable-authorized-{}-{id}",
            std::process::id()
        ));
        fs::create_dir(&dir).unwrap();
        let store = V2QualifierCheckpointStore::new(dir.join("high-water.env")).unwrap();
        (dir, store)
    }

    #[cfg(unix)]
    #[test]
    fn signed_authorized_transition_becomes_durable_as_one_bound_object() {
        let transition = authorized_transition();
        let predecessor = transition.predecessor_checkpoint().clone();
        let successor = transition.successor_checkpoint().clone();
        let authorization_commitment = transition.authorization().commitment();
        let transition_commitment = transition.commitment();
        let (dir, store) = test_store();
        store.persist_genesis(&predecessor).unwrap();

        let durable = persist_authorized_authority_transition(&store, transition).unwrap();
        assert_eq!(
            durable.transition().predecessor_checkpoint(),
            &predecessor
        );
        assert_eq!(durable.transition().successor_checkpoint(), &successor);
        assert_eq!(
            durable.transition().authorization().commitment(),
            authorization_commitment
        );
        assert_eq!(durable.transition().commitment(), transition_commitment);
        assert_eq!(
            durable.durable_receipt().checkpoint_commitment(),
            successor.commitment()
        );
        assert_eq!(store.load().unwrap().unwrap(), successor);
        assert_ne!(durable.durable_receipt().commitment(), [0_u8; 32]);
        assert_ne!(durable.commitment(), [0_u8; 32]);

        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn durable_store_high_water_mismatch_blocks_authorized_transition_and_leaves_lock() {
        let transition = authorized_transition();
        let expected_predecessor_commitment = transition.predecessor_checkpoint().commitment();
        let intended_successor_commitment = transition.successor_checkpoint().commitment();
        let policy = policy();
        let different_authority = V2QualifierAuthorityRecord::genesis(
            1,
            V2QualifierAuthorityProfile::current_from_hex(&"9".repeat(64), &"8".repeat(64))
                .unwrap(),
        )
        .unwrap();
        let different_lineage =
            V2QualifierAuthorityLineage::activate_genesis(different_authority).unwrap();
        let different_guard =
            V2QualifierAntiRollbackGuard::activate_genesis(&different_lineage, &policy).unwrap();
        let (dir, store) = test_store();
        store.persist_genesis(different_guard.checkpoint()).unwrap();

        assert_eq!(
            persist_authorized_authority_transition(&store, transition).unwrap_err(),
            V2DurablyAuthorizedTransitionError::Store(
                V2CheckpointStoreError::HighWaterMismatch
            )
        );
        let lock = store.inspect_writer_lock().unwrap().unwrap();
        assert_eq!(
            lock.expected_checkpoint_commitment(),
            Some(expected_predecessor_commitment)
        );
        assert_eq!(lock.intended_checkpoint_commitment(), intended_successor_commitment);
        assert_ne!(store.load().unwrap().unwrap().commitment(), intended_successor_commitment);

        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn missing_durable_predecessor_fails_closed_with_recovery_lock() {
        let transition = authorized_transition();
        let expected_predecessor_commitment = transition.predecessor_checkpoint().commitment();
        let intended_successor_commitment = transition.successor_checkpoint().commitment();
        let (dir, store) = test_store();

        assert_eq!(
            persist_authorized_authority_transition(&store, transition).unwrap_err(),
            V2DurablyAuthorizedTransitionError::Store(V2CheckpointStoreError::MissingCheckpoint)
        );
        let lock = store.inspect_writer_lock().unwrap().unwrap();
        assert_eq!(
            lock.expected_checkpoint_commitment(),
            Some(expected_predecessor_commitment)
        );
        assert_eq!(lock.intended_checkpoint_commitment(), intended_successor_commitment);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn durable_authorized_source_has_no_root_permit_or_execution_surface() {
        let production = include_str!("v2_durably_authorized_transition.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for forbidden in [
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
                "forbidden durable-authorized surface: {forbidden}"
            );
        }
        assert!(production.contains("persist_successor"));
        assert!(production.contains("transition.authorization().commitment()"));
        assert!(production.contains("durable_receipt.commitment()"));
    }
}
