// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! External verification receipts for trust-store attestation evidence.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_assurance_trust_store::{TrustStoreCheckpoint, TrustStoreProfile};
use symthaea_assurance_trust_store_recovery::TrustStoreRecoveryCommit;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointAttestationVerificationReceipt {
    pub receipt_id: String,
    pub checkpoint_digest: String,
    pub logical_store_id: String,
    pub trust_store_ref: String,
    pub monotonic_counter_ref: String,
    pub counter_epoch: String,
    pub counter_value: u64,
    pub attestation_ref: String,
    pub verified_by_ref: String,
    pub verification_ref: String,
    pub verified_at_ms: u64,
    pub evidence_refs: Vec<String>,
}

impl CheckpointAttestationVerificationReceipt {
    pub fn validate_for(
        &self,
        profile: &TrustStoreProfile,
        checkpoint: &TrustStoreCheckpoint,
    ) -> bool {
        profile.validate()
            && checkpoint.validate(profile)
            && !self.receipt_id.trim().is_empty()
            && self.checkpoint_digest == checkpoint.checkpoint_digest()
            && self.logical_store_id == profile.store_id
            && self.trust_store_ref == profile.trust_store_ref
            && self.monotonic_counter_ref == profile.monotonic_counter_ref
            && self.counter_epoch == checkpoint.counter_epoch
            && self.counter_value == checkpoint.counter_value
            && self.attestation_ref == checkpoint.attestation_ref
            && !self.verified_by_ref.trim().is_empty()
            && self.verified_by_ref != profile.trust_store_ref
            && profile
                .hardware_instance_ref
                .as_ref()
                .is_none_or(|hardware| self.verified_by_ref.as_str() != hardware.as_str())
            && !self.verification_ref.trim().is_empty()
            && self.verified_at_ms >= checkpoint.recorded_at_ms
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryAttestationVerificationReceipt {
    pub receipt_id: String,
    pub recovery_commit_id: String,
    pub logical_store_id: String,
    pub replacement_trust_store_ref: String,
    pub replacement_counter_epoch: String,
    pub replacement_attestation_ref: String,
    pub verified_by_ref: String,
    pub verification_ref: String,
    pub verified_at_ms: u64,
    pub evidence_refs: Vec<String>,
}

impl RecoveryAttestationVerificationReceipt {
    pub fn validate_for(
        &self,
        replacement_profile: &TrustStoreProfile,
        commit: &TrustStoreRecoveryCommit,
    ) -> bool {
        replacement_profile.validate()
            && commit.validate()
            && !self.receipt_id.trim().is_empty()
            && self.recovery_commit_id == commit.commit_id
            && self.logical_store_id == replacement_profile.store_id
            && self.logical_store_id == commit.logical_store_id
            && self.replacement_trust_store_ref == replacement_profile.trust_store_ref
            && self.replacement_trust_store_ref == commit.replacement_trust_store_ref
            && self.replacement_counter_epoch == replacement_profile.initial_counter_epoch
            && self.replacement_counter_epoch == commit.replacement_counter_epoch
            && self.replacement_attestation_ref == commit.replacement_attestation_ref
            && !self.verified_by_ref.trim().is_empty()
            && self.verified_by_ref != replacement_profile.trust_store_ref
            && replacement_profile
                .hardware_instance_ref
                .as_ref()
                .is_none_or(|hardware| self.verified_by_ref.as_str() != hardware.as_str())
            && !self.verification_ref.trim().is_empty()
            && self.verified_at_ms >= commit.committed_at_ms
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_assurance_trust_store::TrustStoreBackendKind;

    fn profile() -> TrustStoreProfile {
        TrustStoreProfile {
            schema_version: "1".into(),
            store_id: "policy-root".into(),
            trust_store_ref: "trust-store:tpm-a".into(),
            backend_kind: TrustStoreBackendKind::HardwareMonotonicCounter,
            hardware_instance_ref: Some("hardware:tpm-a".into()),
            monotonic_counter_ref: "counter:nv-a".into(),
            initial_counter_epoch: "epoch:a".into(),
            minimum_independent_recovery_approvals: 2,
            evidence_refs: vec!["review:profile".into()],
        }
    }

    fn checkpoint() -> TrustStoreCheckpoint {
        TrustStoreCheckpoint {
            schema_version: "1".into(),
            checkpoint_id: "checkpoint:1".into(),
            store_id: "policy-root".into(),
            store_revision: 1,
            counter_epoch: "epoch:a".into(),
            counter_value: 7,
            anchor_revision: 1,
            anchor_digest: "blake3:anchor".into(),
            policy_tip_revision: 1,
            predecessor_checkpoint_digest: None,
            recorded_at_ms: 1_000,
            attestation_ref: "attestation:checkpoint-1".into(),
            independent_verification_ref: "verification:checkpoint-source".into(),
            evidence_refs: vec!["audit:checkpoint".into()],
        }
    }

    fn checkpoint_receipt() -> CheckpointAttestationVerificationReceipt {
        let profile = profile();
        let checkpoint = checkpoint();
        CheckpointAttestationVerificationReceipt {
            receipt_id: "attestation-verification:1".into(),
            checkpoint_digest: checkpoint.checkpoint_digest(),
            logical_store_id: profile.store_id.clone(),
            trust_store_ref: profile.trust_store_ref.clone(),
            monotonic_counter_ref: profile.monotonic_counter_ref.clone(),
            counter_epoch: checkpoint.counter_epoch.clone(),
            counter_value: checkpoint.counter_value,
            attestation_ref: checkpoint.attestation_ref.clone(),
            verified_by_ref: "verifier:hardware-attestation".into(),
            verification_ref: "verification:run-1".into(),
            verified_at_ms: 1_100,
            evidence_refs: vec!["audit:attestation-verification".into()],
        }
    }

    #[test]
    fn checkpoint_attestation_must_bind_exact_checkpoint_and_profile() {
        let profile = profile();
        let checkpoint = checkpoint();
        let receipt = checkpoint_receipt();
        assert!(receipt.validate_for(&profile, &checkpoint));
        assert!(!receipt.grants_physical_authority());
    }

    #[test]
    fn trust_store_cannot_verify_its_own_checkpoint_attestation() {
        let profile = profile();
        let checkpoint = checkpoint();
        let mut receipt = checkpoint_receipt();
        receipt.verified_by_ref = profile.trust_store_ref.clone();
        assert!(!receipt.validate_for(&profile, &checkpoint));
    }

    #[test]
    fn checkpoint_digest_substitution_fails() {
        let profile = profile();
        let checkpoint = checkpoint();
        let mut receipt = checkpoint_receipt();
        receipt.checkpoint_digest = "blake3:other".into();
        assert!(!receipt.validate_for(&profile, &checkpoint));
    }

    #[test]
    fn recovery_attestation_must_bind_exact_replacement() {
        let replacement = TrustStoreProfile {
            trust_store_ref: "trust-store:tpm-b".into(),
            hardware_instance_ref: Some("hardware:tpm-b".into()),
            monotonic_counter_ref: "counter:nv-b".into(),
            initial_counter_epoch: "epoch:b".into(),
            evidence_refs: vec!["review:replacement".into()],
            ..profile()
        };
        let commit = TrustStoreRecoveryCommit {
            schema_version: "1".into(),
            commit_id: "recovery-commit:1".into(),
            authorization_id: "recovery:1".into(),
            logical_store_id: "policy-root".into(),
            previous_checkpoint_digest: "blake3:old-tip".into(),
            backup_digest: "blake3:backup".into(),
            replacement_trust_store_ref: replacement.trust_store_ref.clone(),
            replacement_counter_epoch: replacement.initial_counter_epoch.clone(),
            first_counter_value: 1,
            restored_anchor_revision: 3,
            restored_anchor_digest: "blake3:anchor-3".into(),
            restored_policy_tip_revision: 3,
            replacement_attestation_ref: "attestation:replacement".into(),
            independent_verification_ref: "verification:recovery-source".into(),
            committed_at_ms: 5_000,
            evidence_refs: vec!["audit:recovery".into()],
        };
        let receipt = RecoveryAttestationVerificationReceipt {
            receipt_id: "replacement-verification:1".into(),
            recovery_commit_id: commit.commit_id.clone(),
            logical_store_id: commit.logical_store_id.clone(),
            replacement_trust_store_ref: commit.replacement_trust_store_ref.clone(),
            replacement_counter_epoch: commit.replacement_counter_epoch.clone(),
            replacement_attestation_ref: commit.replacement_attestation_ref.clone(),
            verified_by_ref: "verifier:hardware-attestation".into(),
            verification_ref: "verification:replacement".into(),
            verified_at_ms: 5_100,
            evidence_refs: vec!["audit:replacement-verification".into()],
        };
        assert!(receipt.validate_for(&replacement, &commit));
        assert!(!receipt.grants_physical_authority());
    }
}
