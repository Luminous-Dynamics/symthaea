// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Witness-scoped negative rollback authority for hardened fabrication upgrades.
//!
//! This crate deliberately consumes only `QuorumObservedUpgradeOperationalHeadV1`. A raw
//! `FabricationUpgradeOperationalState`, caller-selected rollback option, or scalar current time is
//! insufficient. The resulting capability therefore means only that the exact witnessed head view
//! from the upstream currentness theorem contains no durable automatic-rollback digest.

#![deny(unsafe_code)]

use serde::Serialize;
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_upgrade_operational_head::{
    QuorumObservedUpgradeOperationalHeadIdV1, QuorumObservedUpgradeOperationalHeadV1,
};
use symthaea_fabrication_witness_authority::WitnessAuthorityRegistryIdV1;
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, OperationalClockBasisIdV1,
};

pub const WITNESSED_NO_ROLLBACK_UPGRADE_HEAD_SCHEMA: &str =
    "symthaea.fabrication.witnessed-no-rollback-upgrade-head.v1";

const NO_ROLLBACK_HEAD_DOMAIN: &[u8] =
    b"symthaea.fabrication.witnessed-no-rollback-upgrade-head.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WitnessedNoRollbackUpgradeHeadIdV1(Sha256Digest);

impl WitnessedNoRollbackUpgradeHeadIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque negative capability: the exact upstream witnessed operational head has no durable
/// automatic-rollback digest.
///
/// This is intentionally scoped to one exact witnessed transparency/checkpoint view. It does not
/// claim that no later checkpoint or operational head exists outside that view.
#[derive(Debug, Clone)]
#[must_use]
pub struct WitnessedNoRollbackUpgradeHeadV1 {
    id: WitnessedNoRollbackUpgradeHeadIdV1,
    observed_head_id: QuorumObservedUpgradeOperationalHeadIdV1,
    state_digest: Sha256Digest,
    state_generation: u64,
    handoff_digest: Sha256Digest,
    publication_digest: Sha256Digest,
    publication_entry_sequence: u64,
    transparency_log_digest: Sha256Digest,
    transparency_log_size: u64,
    transparency_root_digest: Sha256Digest,
    checkpoint_digest: Sha256Digest,
    witness_registry_id: WitnessAuthorityRegistryIdV1,
    witness_registry_sequence: u64,
    exact_verifier_set_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
}

impl WitnessedNoRollbackUpgradeHeadV1 {
    pub fn id(&self) -> WitnessedNoRollbackUpgradeHeadIdV1 {
        self.id
    }
    pub fn observed_head_id(&self) -> QuorumObservedUpgradeOperationalHeadIdV1 {
        self.observed_head_id
    }
    pub fn state_digest(&self) -> Sha256Digest {
        self.state_digest
    }
    pub fn state_generation(&self) -> u64 {
        self.state_generation
    }
    pub fn handoff_digest(&self) -> Sha256Digest {
        self.handoff_digest
    }
    pub fn publication_digest(&self) -> Sha256Digest {
        self.publication_digest
    }
    pub fn publication_entry_sequence(&self) -> u64 {
        self.publication_entry_sequence
    }
    pub fn transparency_log_digest(&self) -> Sha256Digest {
        self.transparency_log_digest
    }
    pub fn transparency_log_size(&self) -> u64 {
        self.transparency_log_size
    }
    pub fn transparency_root_digest(&self) -> Sha256Digest {
        self.transparency_root_digest
    }
    pub fn checkpoint_digest(&self) -> Sha256Digest {
        self.checkpoint_digest
    }
    pub fn witness_registry_id(&self) -> WitnessAuthorityRegistryIdV1 {
        self.witness_registry_id
    }
    pub fn witness_registry_sequence(&self) -> u64 {
        self.witness_registry_sequence
    }
    pub fn exact_verifier_set_digest(&self) -> Sha256Digest {
        self.exact_verifier_set_digest
    }
    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }
    pub fn containment_state_digest(&self) -> Sha256Digest {
        self.containment_state_digest
    }
    pub fn compromise_tracker_digest(&self) -> Sha256Digest {
        self.compromise_tracker_digest
    }
    pub fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.clock_envelope_id
    }
    pub fn operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.operational_basis_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WitnessedNoRollbackUpgradeHeadError {
    DurableRollbackObserved(Sha256Digest),
    Encoding(String),
}

#[derive(Debug, Clone, Serialize)]
struct NoRollbackHeadCommitment {
    schema: &'static str,
    observed_head_id: String,
    state_digest: String,
    state_generation: u64,
    handoff_digest: String,
    publication_digest: String,
    publication_entry_sequence: u64,
    transparency_log_digest: String,
    transparency_log_size: u64,
    transparency_root_digest: String,
    checkpoint_digest: String,
    witness_registry_id: String,
    witness_registry_sequence: u64,
    exact_verifier_set_digest: String,
    trust_snapshot_digest: String,
    containment_state_digest: String,
    compromise_tracker_digest: String,
    clock_envelope_id: String,
    operational_basis_id: String,
}

/// Derive a negative rollback capability from one exact opaque witnessed operational head.
///
/// A raw operational state is deliberately not accepted here. If rollback is durable in the exact
/// observed head, derivation fails closed and preserves that rollback digest in the error.
pub fn derive_witnessed_no_rollback_upgrade_head_v1(
    observed: &QuorumObservedUpgradeOperationalHeadV1,
) -> Result<WitnessedNoRollbackUpgradeHeadV1, WitnessedNoRollbackUpgradeHeadError> {
    if let Some(rollback_digest) = observed.automatic_rollback_digest() {
        return Err(WitnessedNoRollbackUpgradeHeadError::DurableRollbackObserved(
            rollback_digest,
        ));
    }

    let commitment = NoRollbackHeadCommitment {
        schema: WITNESSED_NO_ROLLBACK_UPGRADE_HEAD_SCHEMA,
        observed_head_id: observed.id().to_hex(),
        state_digest: observed.state_digest().to_hex(),
        state_generation: observed.state_generation(),
        handoff_digest: observed.handoff_digest().to_hex(),
        publication_digest: observed.publication_digest().to_hex(),
        publication_entry_sequence: observed.publication_entry_sequence(),
        transparency_log_digest: observed.transparency_log_digest().to_hex(),
        transparency_log_size: observed.transparency_log_size(),
        transparency_root_digest: observed.transparency_root_digest().to_hex(),
        checkpoint_digest: observed.checkpoint_digest().to_hex(),
        witness_registry_id: observed.witness_registry_id().to_hex(),
        witness_registry_sequence: observed.witness_registry_sequence(),
        exact_verifier_set_digest: observed.exact_verifier_set_digest().to_hex(),
        trust_snapshot_digest: observed.trust_snapshot_digest().to_hex(),
        containment_state_digest: observed.containment_state_digest().to_hex(),
        compromise_tracker_digest: observed.compromise_tracker_digest().to_hex(),
        clock_envelope_id: observed.clock_envelope_id().to_hex(),
        operational_basis_id: observed.operational_basis_id().to_hex(),
    };
    let id = WitnessedNoRollbackUpgradeHeadIdV1(hash_serializable(
        NO_ROLLBACK_HEAD_DOMAIN,
        &commitment,
    )?);

    Ok(WitnessedNoRollbackUpgradeHeadV1 {
        id,
        observed_head_id: observed.id(),
        state_digest: observed.state_digest(),
        state_generation: observed.state_generation(),
        handoff_digest: observed.handoff_digest(),
        publication_digest: observed.publication_digest(),
        publication_entry_sequence: observed.publication_entry_sequence(),
        transparency_log_digest: observed.transparency_log_digest(),
        transparency_log_size: observed.transparency_log_size(),
        transparency_root_digest: observed.transparency_root_digest(),
        checkpoint_digest: observed.checkpoint_digest(),
        witness_registry_id: observed.witness_registry_id(),
        witness_registry_sequence: observed.witness_registry_sequence(),
        exact_verifier_set_digest: observed.exact_verifier_set_digest(),
        trust_snapshot_digest: observed.trust_snapshot_digest(),
        containment_state_digest: observed.containment_state_digest(),
        compromise_tracker_digest: observed.compromise_tracker_digest(),
        clock_envelope_id: observed.clock_envelope_id(),
        operational_basis_id: observed.operational_basis_id(),
    })
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, WitnessedNoRollbackUpgradeHeadError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| WitnessedNoRollbackUpgradeHeadError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commitment_identity_changes_with_witnessed_state() {
        let first = NoRollbackHeadCommitment {
            schema: WITNESSED_NO_ROLLBACK_UPGRADE_HEAD_SCHEMA,
            observed_head_id: "11".repeat(32),
            state_digest: "22".repeat(32),
            state_generation: 4,
            handoff_digest: "33".repeat(32),
            publication_digest: "44".repeat(32),
            publication_entry_sequence: 8,
            transparency_log_digest: "55".repeat(32),
            transparency_log_size: 9,
            transparency_root_digest: "66".repeat(32),
            checkpoint_digest: "77".repeat(32),
            witness_registry_id: "88".repeat(32),
            witness_registry_sequence: 1,
            exact_verifier_set_digest: "99".repeat(32),
            trust_snapshot_digest: "aa".repeat(32),
            containment_state_digest: "bb".repeat(32),
            compromise_tracker_digest: "cc".repeat(32),
            clock_envelope_id: "dd".repeat(32),
            operational_basis_id: "ee".repeat(32),
        };
        let mut second = first.clone();
        second.state_generation = 5;
        assert_ne!(
            hash_serializable(NO_ROLLBACK_HEAD_DOMAIN, &first).unwrap(),
            hash_serializable(NO_ROLLBACK_HEAD_DOMAIN, &second).unwrap()
        );
    }
}
