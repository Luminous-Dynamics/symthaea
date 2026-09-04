// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::time::{SystemTime, UNIX_EPOCH};

use ed25519_dalek::{Signature, VerifyingKey};
use symthaea_authority::Digest32;
use symthaea_iot_actuation_guard_admission_challenge::{
    AdmissionRealityChallengeV1, DecodedAdmissionDeviceRealityEvidence,
};
use symthaea_iot_actuation_guard_admission_reservation::AdmissionReservationHead;
use symthaea_iot_authority::DeviceRuntimeState;
use symthaea_iot_posture::DeviceAttestationResultV1;
use symthaea_iot_transport_receipt::TransportTrustHead;

use crate::{
    DEVICE_REALITY_ED25519_ALGORITHM, DEVICE_REALITY_ED25519_SIGNATURE_LEN,
    DeviceRealityError, DeviceRealityPolicyV1, DeviceRealityTrustHead,
    DeviceRealityTrustRegistry,
};

/// Guard-owned fixed verifier for device reality bound to a crash-durable admission
/// reservation challenge.
///
/// This is the corrected successor to the historical semantic-reservation verifier.
/// Policy and concrete-key trust remain local to the privileged process; neither is
/// supplied by the device response.
#[derive(Debug)]
pub struct GuardAdmissionDeviceRealityState {
    policy: DeviceRealityPolicyV1,
    anchored_policy_digest: Digest32,
    trust_registry: DeviceRealityTrustRegistry,
    anchored_trust_head: DeviceRealityTrustHead,
}

impl GuardAdmissionDeviceRealityState {
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

    /// Verify one canonically decoded admission-bound device appraisal using fixed
    /// Ed25519 and guard-local current time.
    ///
    /// The caller cannot supply a verifier, public key, policy, trust head, runtime
    /// projection or relying-party clock.
    pub fn verify_admission_evidence(
        &self,
        evidence: DecodedAdmissionDeviceRealityEvidence,
        challenge: &AdmissionRealityChallengeV1,
    ) -> Result<VerifiedAdmissionDeviceReality, DeviceRealityError> {
        self.verify_admission_evidence_at(evidence, challenge, system_unix_ms()?)
    }

    fn verify_admission_evidence_at(
        &self,
        evidence: DecodedAdmissionDeviceRealityEvidence,
        challenge: &AdmissionRealityChallengeV1,
        now_unix_ms: u64,
    ) -> Result<VerifiedAdmissionDeviceReality, DeviceRealityError> {
        if self.policy.digest()? != self.anchored_policy_digest {
            return Err(DeviceRealityError::AnchoredPolicyDigestMismatch);
        }
        if self.trust_registry.head() != self.anchored_trust_head {
            return Err(DeviceRealityError::AnchoredTrustHeadMismatch);
        }

        challenge
            .validate()
            .map_err(|_| DeviceRealityError::InvalidAdmissionChallenge)?;
        if now_unix_ms < challenge.issued_at_unix_ms()
            || now_unix_ms >= challenge.expires_at_unix_ms()
        {
            return Err(DeviceRealityError::AdmissionChallengeNotFresh);
        }
        if *challenge.device() != self.policy.device {
            return Err(DeviceRealityError::AttestationDeviceMismatch);
        }
        if challenge.reservation_persisted_at_unix_ms()
            < self.trust_registry.snapshot().issued_at_unix_ms
        {
            return Err(DeviceRealityError::AttestationPredatesCurrentTrustGeneration);
        }

        let attestation_object_digest = evidence.attestation_object_digest();
        let response_digest = evidence.response_digest();
        if attestation_object_digest == Digest32([0; 32]) || response_digest == Digest32([0; 32]) {
            return Err(DeviceRealityError::AttestationObjectCommitmentFailed);
        }
        let result = evidence.into_result();
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
        if result.body.device != *challenge.device() || result.body.device != self.policy.device {
            return Err(DeviceRealityError::AttestationDeviceMismatch);
        }

        let challenge_digest = challenge
            .digest()
            .map_err(|_| DeviceRealityError::InvalidAdmissionChallenge)?;
        if result.body.challenge_digest != challenge_digest {
            return Err(DeviceRealityError::AttestationAdmissionChallengeMismatch);
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

        // Device posture v0.1 represents appraisal time in whole seconds. Treat the
        // beginning of that second as the conservative lower bound; never infer finer
        // ordering than the signed evidence represents.
        let appraised_lower_bound_unix_ms = seconds_to_millis(result.body.appraised_at_unix_s)?;
        let expires_at_unix_ms = seconds_to_millis(result.body.expires_at_unix_s)?;
        if appraised_lower_bound_unix_ms < challenge.reservation_persisted_at_unix_ms()
            || appraised_lower_bound_unix_ms < challenge.issued_at_unix_ms()
        {
            return Err(DeviceRealityError::AttestationPredatesAdmissionReservation);
        }
        if expires_at_unix_ms > challenge.expires_at_unix_ms()
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
        let runtime = DeviceRuntimeState {
            running_firmware: result.body.running_firmware,
            last_accepted_sequence: result.body.last_accepted_sequence,
            observations: result.body.observations.clone(),
        };
        let verifier_id = trusted_key.verifier_id.clone();
        let key_id = trusted_key.key_id.clone();
        let key_digest = trusted_key.digest()?;

        Ok(VerifiedAdmissionDeviceReality {
            result,
            runtime,
            result_digest,
            attestation_object_digest,
            response_digest,
            challenge_digest,
            reservation_head: challenge.reservation_head(),
            envelope_digest: challenge.envelope_digest(),
            config_digest: challenge.config_digest(),
            transport_receipt_digest: challenge.transport_receipt_digest(),
            transport_trust_head: challenge.transport_trust_head(),
            verifier_id,
            key_id,
            key_digest,
            trust_head: self.anchored_trust_head,
            policy_digest: self.anchored_policy_digest,
            verified_at_unix_ms: now_unix_ms,
        })
    }
}

/// Opaque proof that the exact reservation-bound signed device appraisal passed fixed
/// cryptography, current concrete-key trust and guard-owned appraisal policy.
///
/// This proof is not semantic acceptance and grants no actuator/HAL authority.
#[derive(Debug)]
pub struct VerifiedAdmissionDeviceReality {
    result: DeviceAttestationResultV1,
    runtime: DeviceRuntimeState,
    result_digest: Digest32,
    attestation_object_digest: Digest32,
    response_digest: Digest32,
    challenge_digest: Digest32,
    reservation_head: AdmissionReservationHead,
    envelope_digest: Digest32,
    config_digest: Digest32,
    transport_receipt_digest: Digest32,
    transport_trust_head: TransportTrustHead,
    verifier_id: String,
    key_id: String,
    key_digest: Digest32,
    trust_head: DeviceRealityTrustHead,
    policy_digest: Digest32,
    verified_at_unix_ms: u64,
}

impl VerifiedAdmissionDeviceReality {
    pub fn runtime_state(&self) -> &DeviceRuntimeState {
        &self.runtime
    }

    pub fn attestation_result(&self) -> &DeviceAttestationResultV1 {
        &self.result
    }

    pub const fn result_digest(&self) -> Digest32 {
        self.result_digest
    }

    pub const fn attestation_object_digest(&self) -> Digest32 {
        self.attestation_object_digest
    }

    pub const fn response_digest(&self) -> Digest32 {
        self.response_digest
    }

    pub const fn challenge_digest(&self) -> Digest32 {
        self.challenge_digest
    }

    pub const fn reservation_head(&self) -> AdmissionReservationHead {
        self.reservation_head
    }

    pub const fn envelope_digest(&self) -> Digest32 {
        self.envelope_digest
    }

    pub const fn config_digest(&self) -> Digest32 {
        self.config_digest
    }

    pub const fn transport_receipt_digest(&self) -> Digest32 {
        self.transport_receipt_digest
    }

    pub const fn transport_trust_head(&self) -> TransportTrustHead {
        self.transport_trust_head
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
