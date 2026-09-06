// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::time::{SystemTime, UNIX_EPOCH};

use ed25519_dalek::{Signature, VerifyingKey};
use symthaea_authority::Digest32;
use symthaea_iot_actuation_guard_two_phase_protocol::{
    SemanticReservationChallengeV1, device_attestation_result_digest,
};
use symthaea_iot_authority::DeviceRuntimeState;
use symthaea_iot_posture::DeviceAttestationResultV1;

use crate::{
    DEVICE_REALITY_ED25519_ALGORITHM, DEVICE_REALITY_ED25519_SIGNATURE_LEN,
    DeviceRealityError, DeviceRealityPolicyV1, DeviceRealityTrustHead,
    DeviceRealityTrustRegistry,
};

/// Guard-owned fixed device-reality verification state.
#[derive(Debug)]
pub struct GuardDeviceRealityState {
    policy: DeviceRealityPolicyV1,
    anchored_policy_digest: Digest32,
    trust_registry: DeviceRealityTrustRegistry,
    anchored_trust_head: DeviceRealityTrustHead,
}

impl GuardDeviceRealityState {
    /// Construct only when both loaded local objects match independently retained
    /// anchors. Neither policy nor trust state is accepted from phase-2 IPC.
    pub fn new(
        policy: DeviceRealityPolicyV1,
        anchored_policy_digest: Digest32,
        trust_registry: DeviceRealityTrustRegistry,
        anchored_trust_head: DeviceRealityTrustHead,
    ) -> Result<Self, DeviceRealityError> {
        if policy.digest()? != anchored_policy_digest {
            return Err(DeviceRealityError::AnchoredPolicyDigestMismatch);
        }
        if trust_registry.head() != anchored_trust_head {
            return Err(DeviceRealityError::AnchoredTrustHeadMismatch);
        }
        Ok(Self {
            policy,
            anchored_policy_digest,
            trust_registry,
            anchored_trust_head,
        })
    }

    pub const fn anchored_policy_digest(&self) -> Digest32 {
        self.anchored_policy_digest
    }

    pub const fn anchored_trust_head(&self) -> DeviceRealityTrustHead {
        self.anchored_trust_head
    }

    /// Verify one exact challenge-bound attestation using the guard's local wall clock
    /// and fixed Ed25519 implementation. There is no signature-provider/key argument.
    pub fn verify(
        &self,
        result: DeviceAttestationResultV1,
        challenge: &SemanticReservationChallengeV1,
    ) -> Result<VerifiedPostReservationDeviceReality, DeviceRealityError> {
        self.verify_at(result, challenge, system_unix_ms()?)
    }

    fn verify_at(
        &self,
        result: DeviceAttestationResultV1,
        challenge: &SemanticReservationChallengeV1,
        now_unix_ms: u64,
    ) -> Result<VerifiedPostReservationDeviceReality, DeviceRealityError> {
        if self.policy.digest()? != self.anchored_policy_digest {
            return Err(DeviceRealityError::AnchoredPolicyDigestMismatch);
        }
        if self.trust_registry.head() != self.anchored_trust_head {
            return Err(DeviceRealityError::AnchoredTrustHeadMismatch);
        }

        challenge
            .validate()
            .map_err(|_| DeviceRealityError::InvalidReservationChallenge)?;
        if !challenge.is_fresh_at(now_unix_ms) {
            return Err(DeviceRealityError::ReservationChallengeNotFresh);
        }
        if challenge.device != self.policy.device {
            return Err(DeviceRealityError::AttestationDeviceMismatch);
        }
        // Any verifier-trust generation advance invalidates outstanding older
        // reservation challenges. A newly trusted key therefore cannot retroactively
        // answer a challenge created before that trust generation existed.
        if challenge.persisted_at_unix_ms < self.trust_registry.snapshot().issued_at_unix_ms {
            return Err(DeviceRealityError::AttestationPredatesCurrentTrustGeneration);
        }

        result
            .body
            .validate_structure()
            .map_err(|_| DeviceRealityError::InvalidAttestationResult)?;
        if result.signature.len() != DEVICE_REALITY_ED25519_SIGNATURE_LEN {
            return Err(DeviceRealityError::InvalidAttestationSignatureLength);
        }
        if result.body.algorithm != DEVICE_REALITY_ED25519_ALGORITHM {
            return Err(DeviceRealityError::AttestationAlgorithmMismatch);
        }
        if result.body.device != challenge.device || result.body.device != self.policy.device {
            return Err(DeviceRealityError::AttestationDeviceMismatch);
        }
        let challenge_digest = challenge
            .digest()
            .map_err(|_| DeviceRealityError::InvalidReservationChallenge)?;
        if result.body.challenge_digest != challenge_digest {
            return Err(DeviceRealityError::AttestationChallengeMismatch);
        }
        if !self
            .policy
            .allowed_verifier_ids
            .contains(&result.body.verifier_id)
        {
            return Err(DeviceRealityError::AttestationVerifierDenied);
        }
        if !self
            .policy
            .accepted_reference_values
            .contains(&result.body.reference_values_digest)
        {
            return Err(DeviceRealityError::AttestationReferenceValuesDenied);
        }
        if result.body.appraisal_policy_digest != self.policy.exact_appraisal_policy_digest {
            return Err(DeviceRealityError::AttestationAppraisalPolicyMismatch);
        }

        // Posture v0.1 records whole seconds. Treat the start of that second as the
        // conservative lower bound rather than claiming sub-second ordering it cannot
        // prove. This may reject a valid result created later in the persistence second,
        // but cannot admit a result whose represented interval may predate persistence.
        let appraised_lower_bound_unix_ms = seconds_to_millis(result.body.appraised_at_unix_s)?;
        let expires_at_unix_ms = seconds_to_millis(result.body.expires_at_unix_s)?;
        if appraised_lower_bound_unix_ms < challenge.persisted_at_unix_ms {
            return Err(DeviceRealityError::AttestationPredatesSemanticPersistence);
        }
        if expires_at_unix_ms > challenge.expires_at_unix_ms
            || now_unix_ms < appraised_lower_bound_unix_ms
            || now_unix_ms >= expires_at_unix_ms
        {
            return Err(DeviceRealityError::AttestationNotFreshForReservation);
        }
        let lifetime_ms = expires_at_unix_ms
            .checked_sub(appraised_lower_bound_unix_ms)
            .ok_or(DeviceRealityError::AttestationNotFreshForReservation)?;
        if lifetime_ms == 0 || lifetime_ms > self.policy.max_result_lifetime_ms {
            return Err(DeviceRealityError::AttestationLifetimeExceedsPolicy);
        }
        if appraised_lower_bound_unix_ms < self.trust_registry.snapshot().issued_at_unix_ms {
            return Err(DeviceRealityError::AttestationPredatesCurrentTrustGeneration);
        }

        let trusted_key = self.trust_registry.exact_active_key(
            &result.body,
            appraised_lower_bound_unix_ms,
            now_unix_ms,
        )?;
        if lifetime_ms > trusted_key.max_result_lifetime_ms {
            return Err(DeviceRealityError::AttestationLifetimeExceedsPolicy);
        }

        let message = result
            .body
            .signature_message()
            .map_err(|_| DeviceRealityError::InvalidAttestationResult)?;
        let signature = Signature::try_from(result.signature.as_slice())
            .map_err(|_| DeviceRealityError::InvalidAttestationSignatureLength)?;
        let verifying_key = VerifyingKey::from_bytes(&trusted_key.public_key)
            .map_err(|_| DeviceRealityError::InvalidVerifierPublicKey)?;
        verifying_key
            .verify_strict(&message, &signature)
            .map_err(|_| DeviceRealityError::InvalidAttestationSignature)?;

        let result_digest = result
            .body
            .digest()
            .map_err(|_| DeviceRealityError::InvalidAttestationResult)?;
        let attestation_object_digest = device_attestation_result_digest(&result)
            .map_err(|_| DeviceRealityError::AttestationObjectCommitmentFailed)?;
        let key_id = trusted_key.key_id.clone();
        let key_digest = trusted_key.digest()?;
        let verifier_id = trusted_key.verifier_id.clone();
        let runtime = DeviceRuntimeState {
            running_firmware: result.body.running_firmware,
            last_accepted_sequence: result.body.last_accepted_sequence,
            observations: result.body.observations.clone(),
        };

        Ok(VerifiedPostReservationDeviceReality {
            result,
            runtime,
            result_digest,
            attestation_object_digest,
            challenge_digest,
            verifier_id,
            key_id,
            key_digest,
            trust_head: self.anchored_trust_head,
            policy_digest: self.anchored_policy_digest,
            verified_at_unix_ms: now_unix_ms,
        })
    }
}

/// Opaque local proof that one exact post-reservation device-attestation result passed
/// fixed cryptography, current concrete-key trust and exact guard-owned appraisal policy.
#[derive(Debug)]
pub struct VerifiedPostReservationDeviceReality {
    result: DeviceAttestationResultV1,
    runtime: DeviceRuntimeState,
    result_digest: Digest32,
    attestation_object_digest: Digest32,
    challenge_digest: Digest32,
    verifier_id: String,
    key_id: String,
    key_digest: Digest32,
    trust_head: DeviceRealityTrustHead,
    policy_digest: Digest32,
    verified_at_unix_ms: u64,
}

impl VerifiedPostReservationDeviceReality {
    /// Trusted runtime projection derived from the exact signed body. This crate never
    /// accepts a standalone `DeviceRuntimeState` as verification input.
    pub fn runtime_state(&self) -> &DeviceRuntimeState {
        &self.runtime
    }

    /// Exact retained attestation object for later controller/JIT correlation.
    pub fn attestation_result(&self) -> &DeviceAttestationResultV1 {
        &self.result
    }

    pub const fn result_digest(&self) -> Digest32 {
        self.result_digest
    }

    pub const fn attestation_object_digest(&self) -> Digest32 {
        self.attestation_object_digest
    }

    pub const fn challenge_digest(&self) -> Digest32 {
        self.challenge_digest
    }

    pub fn verifier_id(&self) -> &str {
        &self.verifier_id
    }

    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    pub const fn key_digest(&self) -> Digest32 {
        self.key_digest
    }

    pub const fn trust_head(&self) -> DeviceRealityTrustHead {
        self.trust_head
    }

    pub const fn policy_digest(&self) -> Digest32 {
        self.policy_digest
    }

    pub const fn verified_at_unix_ms(&self) -> u64 {
        self.verified_at_unix_ms
    }
}

fn system_unix_ms() -> Result<u64, DeviceRealityError> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| DeviceRealityError::SystemClockBeforeUnixEpoch)?;
    u64::try_from(elapsed.as_millis()).map_err(|_| DeviceRealityError::SystemClockOverflow)
}

fn seconds_to_millis(value: u64) -> Result<u64, DeviceRealityError> {
    value
        .checked_mul(1_000)
        .ok_or(DeviceRealityError::TimeConversionOverflow)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use ed25519_dalek::{Signer, SigningKey};
    use symthaea_authority::ResourceRef;
    use symthaea_iot_actuation_guard_two_phase_protocol::{
        SEMANTIC_RESERVATION_CHALLENGE_SCHEMA_VERSION, SemanticReservationChallengeV1,
        device_attestation_result_digest,
    };
    use symthaea_iot_device_protocol::DeviceSemanticHead;
    use symthaea_iot_posture::{
        DEVICE_ATTESTATION_RESULT_SCHEMA_VERSION, DeviceAttestationResultBodyV1,
    };
    use symthaea_iot_transport_receipt::TransportTrustHead;

    use super::*;
    use crate::{
        DEVICE_REALITY_POLICY_SCHEMA_VERSION, DEVICE_REALITY_TRUST_SCHEMA_VERSION,
        DeviceRealityTrustSnapshotV1, DeviceRealityVerifierKeyStatus, DeviceRealityVerifierKeyV1,
    };

    fn d(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn challenge() -> SemanticReservationChallengeV1 {
        SemanticReservationChallengeV1 {
            schema_version: SEMANTIC_RESERVATION_CHALLENGE_SCHEMA_VERSION,
            nonce: [0xA5; 32],
            admission_request_digest: d(1),
            envelope_digest: d(2),
            transport_receipt_digest: d(3),
            device: ResourceRef("iot:valve:72".into()),
            transport_trust_head: TransportTrustHead {
                sequence: 4,
                digest: d(4),
            },
            semantic_head: DeviceSemanticHead {
                generation: 8,
                digest: d(5),
            },
            persisted_at_unix_ms: 10_000,
            expires_at_unix_ms: 15_000,
        }
    }

    fn signing_key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    fn policy() -> DeviceRealityPolicyV1 {
        DeviceRealityPolicyV1 {
            schema_version: DEVICE_REALITY_POLICY_SCHEMA_VERSION,
            device: ResourceRef("iot:valve:72".into()),
            allowed_verifier_ids: BTreeSet::from(["verifier:fleet-a".into()]),
            accepted_reference_values: BTreeSet::from([d(0x32)]),
            exact_appraisal_policy_digest: d(0x33),
            max_result_lifetime_ms: 3_000,
        }
    }

    fn key(
        signing_key: &SigningKey,
        status: DeviceRealityVerifierKeyStatus,
        not_after_unix_ms: u64,
    ) -> DeviceRealityVerifierKeyV1 {
        DeviceRealityVerifierKeyV1 {
            verifier_id: "verifier:fleet-a".into(),
            key_id: "device-key-1".into(),
            algorithm: DEVICE_REALITY_ED25519_ALGORITHM.into(),
            public_key: signing_key.verifying_key().to_bytes(),
            status,
            not_before_unix_ms: 5_000,
            not_after_unix_ms,
            max_result_lifetime_ms: 3_000,
        }
    }

    fn registry(
        signing_key: &SigningKey,
        status: DeviceRealityVerifierKeyStatus,
        not_after_unix_ms: u64,
    ) -> DeviceRealityTrustRegistry {
        DeviceRealityTrustRegistry::genesis(DeviceRealityTrustSnapshotV1 {
            schema_version: DEVICE_REALITY_TRUST_SCHEMA_VERSION,
            sequence: 1,
            issued_at_unix_ms: 9_000,
            expires_at_unix_ms: 20_000,
            previous_snapshot_digest: None,
            keys: vec![key(signing_key, status, not_after_unix_ms)],
        })
        .unwrap()
    }

    fn signed_result(
        signing_key: &SigningKey,
        challenge: &SemanticReservationChallengeV1,
    ) -> DeviceAttestationResultV1 {
        let body = DeviceAttestationResultBodyV1 {
            schema_version: DEVICE_ATTESTATION_RESULT_SCHEMA_VERSION,
            verifier_id: "verifier:fleet-a".into(),
            key_id: "device-key-1".into(),
            algorithm: DEVICE_REALITY_ED25519_ALGORITHM.into(),
            device: challenge.device.clone(),
            challenge_digest: challenge.digest().unwrap(),
            appraised_at_unix_s: 11,
            expires_at_unix_s: 14,
            evidence_digest: d(0x31),
            reference_values_digest: d(0x32),
            appraisal_policy_digest: d(0x33),
            running_firmware: d(0x34),
            last_accepted_sequence: Some(7),
            observations: BTreeMap::from([("pressure_x100".into(), 20_000)]),
        };
        let signature = signing_key.sign(&body.signature_message().unwrap()).to_bytes();
        DeviceAttestationResultV1 {
            body,
            signature: signature.to_vec(),
        }
    }

    fn state(signing_key: &SigningKey) -> GuardDeviceRealityState {
        state_with_key(signing_key, DeviceRealityVerifierKeyStatus::Active, 20_000)
    }

    fn state_with_key(
        signing_key: &SigningKey,
        status: DeviceRealityVerifierKeyStatus,
        not_after_unix_ms: u64,
    ) -> GuardDeviceRealityState {
        let policy = policy();
        let policy_digest = policy.digest().unwrap();
        let registry = registry(signing_key, status, not_after_unix_ms);
        let head = registry.head();
        GuardDeviceRealityState::new(policy, policy_digest, registry, head).unwrap()
    }

    #[test]
    fn real_ed25519_challenge_bound_device_reality_passes() {
        let signing_key = signing_key(0x61);
        let challenge = challenge();
        let result = signed_result(&signing_key, &challenge);
        let expected_object_digest = device_attestation_result_digest(&result).unwrap();
        let verified = state(&signing_key)
            .verify_at(result, &challenge, 12_000)
            .unwrap();

        assert_eq!(verified.runtime_state().running_firmware, d(0x34));
        assert_eq!(verified.runtime_state().last_accepted_sequence, Some(7));
        assert_eq!(verified.attestation_object_digest(), expected_object_digest);
        assert_eq!(verified.challenge_digest(), challenge.digest().unwrap());
        assert_eq!(verified.verifier_id(), "verifier:fleet-a");
        assert_eq!(verified.key_id(), "device-key-1");
        assert_ne!(verified.key_digest(), Digest32([0; 32]));
    }

    #[test]
    fn signed_observation_mutation_is_rejected() {
        let signing_key = signing_key(0x61);
        let challenge = challenge();
        let mut result = signed_result(&signing_key, &challenge);
        result.body.observations.insert("pressure_x100".into(), 20_001);
        assert!(matches!(
            state(&signing_key).verify_at(result, &challenge, 12_000),
            Err(DeviceRealityError::InvalidAttestationSignature)
        ));
    }

    #[test]
    fn wrong_challenge_reference_values_and_policy_fail_closed() {
        let signing_key = signing_key(0x61);
        let challenge = challenge();
        let mut other = challenge.clone();
        other.nonce[0] ^= 1;
        let result = signed_result(&signing_key, &challenge);
        assert!(matches!(
            state(&signing_key).verify_at(result, &other, 12_000),
            Err(DeviceRealityError::AttestationChallengeMismatch)
        ));

        let mut wrong_reference = signed_result(&signing_key, &challenge);
        wrong_reference.body.reference_values_digest = d(0x99);
        assert!(matches!(
            state(&signing_key).verify_at(wrong_reference, &challenge, 12_000),
            Err(DeviceRealityError::AttestationReferenceValuesDenied)
        ));

        let mut wrong_policy = signed_result(&signing_key, &challenge);
        wrong_policy.body.appraisal_policy_digest = d(0x98);
        assert!(matches!(
            state(&signing_key).verify_at(wrong_policy, &challenge, 12_000),
            Err(DeviceRealityError::AttestationAppraisalPolicyMismatch)
        ));
    }

    #[test]
    fn revoked_key_fails_closed() {
        let signing_key = signing_key(0x61);
        let challenge = challenge();
        let result = signed_result(&signing_key, &challenge);
        assert!(matches!(
            state_with_key(
                &signing_key,
                DeviceRealityVerifierKeyStatus::Revoked,
                20_000,
            )
            .verify_at(result, &challenge, 12_000),
            Err(DeviceRealityError::NoActiveVerifierKey)
        ));
    }

    #[test]
    fn natural_key_expiry_fails_while_challenge_and_attestation_are_live() {
        let signing_key = signing_key(0x61);
        let challenge = challenge();
        let result = signed_result(&signing_key, &challenge);
        assert!(matches!(
            state_with_key(
                &signing_key,
                DeviceRealityVerifierKeyStatus::Active,
                12_000,
            )
            .verify_at(result, &challenge, 12_500),
            Err(DeviceRealityError::VerifierKeyNotActive)
        ));
    }

    #[test]
    fn challenge_predating_current_trust_generation_is_rejected() {
        let signing_key = signing_key(0x61);
        let mut challenge = challenge();
        challenge.persisted_at_unix_ms = 8_000;
        challenge.expires_at_unix_ms = 13_000;
        let result = signed_result(&signing_key, &challenge);
        assert!(matches!(
            state(&signing_key).verify_at(result, &challenge, 12_000),
            Err(DeviceRealityError::AttestationPredatesCurrentTrustGeneration)
        ));
    }

    #[test]
    fn independently_anchored_policy_and_trust_are_required() {
        let signing_key = signing_key(0x61);
        let policy = policy();
        let policy_mismatch_registry = registry(
            &signing_key,
            DeviceRealityVerifierKeyStatus::Active,
            20_000,
        );
        let head = policy_mismatch_registry.head();
        assert!(matches!(
            GuardDeviceRealityState::new(
                policy.clone(),
                d(0xFE),
                policy_mismatch_registry,
                head,
            ),
            Err(DeviceRealityError::AnchoredPolicyDigestMismatch)
        ));

        let policy_digest = policy.digest().unwrap();
        let registry = registry(
            &signing_key,
            DeviceRealityVerifierKeyStatus::Active,
            20_000,
        );
        let wrong_head = DeviceRealityTrustHead {
            sequence: registry.head().sequence,
            digest: d(0xFD),
        };
        assert!(matches!(
            GuardDeviceRealityState::new(policy, policy_digest, registry, wrong_head),
            Err(DeviceRealityError::AnchoredTrustHeadMismatch)
        ));
    }
}
