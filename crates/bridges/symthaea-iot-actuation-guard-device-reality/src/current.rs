// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Current device-reality fencing for one already authenticated admission-bound appraisal.
//!
//! The earlier admission verifier proves that one exact signed appraisal was trusted at its
//! verification boundary. This module answers a different question immediately before a later
//! physical attempt: does that exact proof still agree with independently anchored current
//! device-reality policy, current verifier trust, the exact verifier key, and every natural
//! expiry boundary?
//!
//! Success remains non-authorizing. The returned fence borrows both this guard-owned current
//! state and the exact historical proof so a higher-level actuation linearization boundary can
//! keep the checked state pinned for one attempt.

use std::time::{SystemTime, UNIX_EPOCH};

use ed25519_dalek::{Signature, VerifyingKey};
use symthaea_authority::Digest32;

use crate::{
    DEVICE_REALITY_ED25519_ALGORITHM, DEVICE_REALITY_ED25519_SIGNATURE_LEN, DeviceRealityError,
    DeviceRealityPolicyV1, DeviceRealityTrustHead, DeviceRealityTrustRegistry,
    VerifiedAdmissionDeviceReality,
};

/// Guard-owned current device-reality state used only to fence an existing admission-bound proof.
#[derive(Debug)]
pub struct CurrentAdmissionDeviceRealityGuard {
    policy: DeviceRealityPolicyV1,
    anchored_policy_digest: Digest32,
    trust_registry: DeviceRealityTrustRegistry,
    anchored_trust_head: DeviceRealityTrustHead,
}

impl CurrentAdmissionDeviceRealityGuard {
    /// Bind independently retained current device-reality policy and verifier trust.
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

    /// Re-establish current device-reality trust using guard-local wall time and fixed Ed25519.
    ///
    /// The caller cannot supply current time, a verifier, a public key, policy, or trust state.
    pub fn fence_current<'a>(
        &'a self,
        proof: &'a VerifiedAdmissionDeviceReality,
    ) -> Result<CurrentAdmissionDeviceRealityFence<'a>, DeviceRealityError> {
        self.fence_current_at(proof, system_unix_ms()?)
    }

    fn fence_current_at<'a>(
        &'a self,
        proof: &'a VerifiedAdmissionDeviceReality,
        now_unix_ms: u64,
    ) -> Result<CurrentAdmissionDeviceRealityFence<'a>, DeviceRealityError> {
        self.policy.validate()?;
        if self.policy.digest()? != self.anchored_policy_digest {
            return Err(DeviceRealityError::AnchoredPolicyDigestMismatch);
        }
        if self.trust_registry.head() != self.anchored_trust_head {
            return Err(DeviceRealityError::AnchoredTrustHeadMismatch);
        }
        if proof.policy_digest() != self.anchored_policy_digest {
            return Err(DeviceRealityError::CurrentDeviceRealityProofPolicyMismatch);
        }
        if proof.trust_head() != self.anchored_trust_head {
            return Err(DeviceRealityError::CurrentDeviceRealityProofTrustHeadMismatch);
        }
        if now_unix_ms < proof.verified_at_unix_ms() {
            return Err(DeviceRealityError::CurrentDeviceRealityClockRegressed);
        }

        let result = proof.attestation_result();
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
        if result.body.device != self.policy.device {
            return Err(DeviceRealityError::AttestationDeviceMismatch);
        }
        if result.body.verifier_id != proof.verifier_id() || result.body.key_id != proof.key_id() {
            return Err(DeviceRealityError::CurrentDeviceRealityVerifierKeyMismatch);
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

        let result_digest = result
            .body
            .digest()
            .map_err(|_| DeviceRealityError::InvalidAttestationResult)?;
        if result_digest != proof.result_digest() {
            return Err(DeviceRealityError::CurrentDeviceRealityProofCommitmentMismatch);
        }

        let appraised_lower_bound_unix_ms = seconds_to_millis(result.body.appraised_at_unix_s)?;
        let attestation_expires_at_unix_ms = seconds_to_millis(result.body.expires_at_unix_s)?;
        if now_unix_ms < appraised_lower_bound_unix_ms
            || now_unix_ms >= attestation_expires_at_unix_ms
        {
            return Err(DeviceRealityError::CurrentDeviceRealityWindowElapsed);
        }
        if appraised_lower_bound_unix_ms < self.trust_registry.snapshot().issued_at_unix_ms {
            return Err(DeviceRealityError::AttestationPredatesCurrentTrustGeneration);
        }

        let lifetime_ms = attestation_expires_at_unix_ms
            .checked_sub(appraised_lower_bound_unix_ms)
            .ok_or(DeviceRealityError::CurrentDeviceRealityWindowElapsed)?;
        if lifetime_ms == 0 || lifetime_ms > self.policy.max_result_lifetime_ms {
            return Err(DeviceRealityError::AttestationLifetimeExceedsPolicy);
        }

        // Reuse the crate-private exact selector. The later generic actuation layer therefore never
        // learns the verifier-key lifecycle rules and cannot accidentally fork them.
        let current_key = self.trust_registry.exact_active_key(
            &result.body,
            appraised_lower_bound_unix_ms,
            now_unix_ms,
        )?;
        if lifetime_ms > current_key.max_result_lifetime_ms {
            return Err(DeviceRealityError::AttestationLifetimeExceedsPolicy);
        }
        if current_key.verifier_id != proof.verifier_id()
            || current_key.key_id != proof.key_id()
            || current_key.digest()? != proof.key_digest()
        {
            return Err(DeviceRealityError::CurrentDeviceRealityVerifierKeyMismatch);
        }

        let message = result
            .body
            .signature_message()
            .map_err(|_| DeviceRealityError::InvalidAttestationResult)?;
        let signature = Signature::try_from(result.signature.as_slice())
            .map_err(|_| DeviceRealityError::InvalidAttestationSignatureLength)?;
        let verifying_key = VerifyingKey::from_bytes(&current_key.public_key)
            .map_err(|_| DeviceRealityError::InvalidVerifierPublicKey)?;
        verifying_key
            .verify_strict(&message, &signature)
            .map_err(|_| DeviceRealityError::InvalidAttestationSignature)?;

        let verifier_key_not_after_unix_ms = current_key.not_after_unix_ms;
        let trust_snapshot_expires_at_unix_ms = self.trust_registry.snapshot().expires_at_unix_ms;
        let valid_until_unix_ms = attestation_expires_at_unix_ms
            .min(verifier_key_not_after_unix_ms)
            .min(trust_snapshot_expires_at_unix_ms);
        if now_unix_ms >= valid_until_unix_ms {
            return Err(DeviceRealityError::CurrentDeviceRealityWindowElapsed);
        }

        Ok(CurrentAdmissionDeviceRealityFence {
            _guard: self,
            proof,
            fenced_at_unix_ms: now_unix_ms,
            attestation_expires_at_unix_ms,
            verifier_key_not_after_unix_ms,
            trust_snapshot_expires_at_unix_ms,
            valid_until_unix_ms,
        })
    }
}

/// Borrowed proof that one exact admission-bound device appraisal remains current now.
///
/// This object is neither authority nor a portable lease. It exists only while the exact current
/// guard state and exact historical proof remain borrowed.
#[derive(Debug)]
pub struct CurrentAdmissionDeviceRealityFence<'a> {
    _guard: &'a CurrentAdmissionDeviceRealityGuard,
    proof: &'a VerifiedAdmissionDeviceReality,
    fenced_at_unix_ms: u64,
    attestation_expires_at_unix_ms: u64,
    verifier_key_not_after_unix_ms: u64,
    trust_snapshot_expires_at_unix_ms: u64,
    valid_until_unix_ms: u64,
}

impl<'a> CurrentAdmissionDeviceRealityFence<'a> {
    pub const fn proof(&self) -> &'a VerifiedAdmissionDeviceReality {
        self.proof
    }

    pub const fn fenced_at_unix_ms(&self) -> u64 {
        self.fenced_at_unix_ms
    }

    pub const fn attestation_expires_at_unix_ms(&self) -> u64 {
        self.attestation_expires_at_unix_ms
    }

    pub const fn verifier_key_not_after_unix_ms(&self) -> u64 {
        self.verifier_key_not_after_unix_ms
    }

    pub const fn trust_snapshot_expires_at_unix_ms(&self) -> u64 {
        self.trust_snapshot_expires_at_unix_ms
    }

    /// Earliest exclusive natural-expiry boundary across attestation, exact key and trust snapshot.
    pub const fn valid_until_unix_ms(&self) -> u64 {
        self.valid_until_unix_ms
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
