// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use symthaea_authority::{Digest32, ResourceRef};
use symthaea_iot_actuation_guard_admission_reservation::AdmissionReservationHead;
use symthaea_iot_actuation_guard_device_reality::DeviceRealityTrustHead;
use symthaea_iot_actuation_guard_semantic_persistence::PersistedSemanticAcceptance;
use symthaea_iot_device_protocol::DeviceSemanticHead;
use symthaea_iot_transport_receipt::TransportTrustHead;

use crate::{
    MAX_POST_SEMANTIC_CONTROLLER_CHALLENGE_LIFETIME_MS, POST_SEMANTIC_CONTROLLER_CHALLENGE_DOMAIN,
    POST_SEMANTIC_CONTROLLER_CHALLENGE_SCHEMA_VERSION, PostSemanticControllerError,
    update_digest, update_string, valid_device,
};

/// Outbound privileged challenge proving that semantic safety was durably committed before
/// a hardware controller is allowed to produce consequential interlock evidence.
///
/// Fields are private and the type intentionally does not implement `Deserialize`.
/// The only production constructor requires opaque `PersistedSemanticAcceptance` and reads
/// OS entropy + trusted process wall time internally.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PostSemanticControllerChallengeV1 {
    schema_version: u16,
    nonce: [u8; 32],
    admission_reservation_head: AdmissionReservationHead,
    admission_challenge_digest: Digest32,
    envelope_digest: Digest32,
    transport_receipt_digest: Digest32,
    transport_trust_head: TransportTrustHead,
    config_digest: Digest32,
    device: ResourceRef,
    device_attestation_object_digest: Digest32,
    device_reality_result_digest: Digest32,
    device_reality_trust_head: DeviceRealityTrustHead,
    device_reality_policy_digest: Digest32,
    device_reality_key_digest: Digest32,
    semantic_head: DeviceSemanticHead,
    admission_persisted_at_unix_ms: u64,
    device_reality_verified_at_unix_ms: u64,
    semantic_persisted_at_unix_ms: u64,
    device_reality_expires_at_unix_ms: u64,
    effect_deadline_unix_ms: u64,
    issued_at_unix_ms: u64,
    expires_at_unix_ms: u64,
}

impl PostSemanticControllerChallengeV1 {
    /// Issue one controller challenge from the exact opaque semantic-persistence proof.
    ///
    /// The challenge cannot outlive either the authenticated device-reality result or the
    /// original physical-effect deadline. If either window has already closed, issuance
    /// fails and no controller evidence can enter the current physical-effect lineage.
    pub fn issue_from_persisted_semantic_acceptance(
        acceptance: &PersistedSemanticAcceptance,
    ) -> Result<Self, PostSemanticControllerError> {
        let mut nonce = [0u8; 32];
        getrandom::getrandom(&mut nonce).map_err(|_| PostSemanticControllerError::EntropyUnavailable)?;
        let issued_at_unix_ms = system_unix_ms()?;

        let device_reality = acceptance.device_reality();
        let admission = acceptance.admission_reservation();
        let device_reality_expires_at_unix_ms = device_reality
            .attestation_result()
            .body
            .expires_at_unix_s
            .checked_mul(1_000)
            .ok_or(PostSemanticControllerError::TimeOverflow)?;
        let effect_deadline_unix_ms = admission
            .envelope()
            .send_not_after_unix_s
            .checked_mul(1_000)
            .ok_or(PostSemanticControllerError::TimeOverflow)?;
        let nominal_expiry = issued_at_unix_ms
            .checked_add(MAX_POST_SEMANTIC_CONTROLLER_CHALLENGE_LIFETIME_MS)
            .ok_or(PostSemanticControllerError::TimeOverflow)?;
        let expires_at_unix_ms = nominal_expiry
            .min(device_reality_expires_at_unix_ms)
            .min(effect_deadline_unix_ms);

        let challenge = Self {
            schema_version: POST_SEMANTIC_CONTROLLER_CHALLENGE_SCHEMA_VERSION,
            nonce,
            admission_reservation_head: admission.head(),
            admission_challenge_digest: device_reality.challenge_digest(),
            envelope_digest: acceptance.envelope_digest(),
            transport_receipt_digest: admission.transport_receipt_digest(),
            transport_trust_head: admission.transport_trust_head(),
            config_digest: device_reality.config_digest(),
            device: admission.envelope().command.device.clone(),
            device_attestation_object_digest: acceptance.device_attestation_object_digest(),
            device_reality_result_digest: device_reality.result_digest(),
            device_reality_trust_head: device_reality.trust_head(),
            device_reality_policy_digest: device_reality.policy_digest(),
            device_reality_key_digest: device_reality.key_digest(),
            semantic_head: acceptance.device_head(),
            admission_persisted_at_unix_ms: admission.persisted_at_unix_ms(),
            device_reality_verified_at_unix_ms: device_reality.verified_at_unix_ms(),
            semantic_persisted_at_unix_ms: acceptance.semantic_persisted_at_unix_ms(),
            device_reality_expires_at_unix_ms,
            effect_deadline_unix_ms,
            issued_at_unix_ms,
            expires_at_unix_ms,
        };
        challenge.validate()?;
        Ok(challenge)
    }

    pub fn validate(&self) -> Result<(), PostSemanticControllerError> {
        if self.schema_version != POST_SEMANTIC_CONTROLLER_CHALLENGE_SCHEMA_VERSION {
            return Err(PostSemanticControllerError::UnsupportedChallengeSchema);
        }
        if self.nonce == [0; 32] {
            return Err(PostSemanticControllerError::ZeroChallengeNonce);
        }
        if self.admission_reservation_head.generation == 0
            || self.admission_reservation_head.digest == Digest32([0; 32])
            || self.admission_challenge_digest == Digest32([0; 32])
            || self.envelope_digest == Digest32([0; 32])
            || self.transport_receipt_digest == Digest32([0; 32])
            || self.transport_trust_head.sequence == 0
            || self.transport_trust_head.digest == Digest32([0; 32])
            || self.config_digest == Digest32([0; 32])
            || self.device_attestation_object_digest == Digest32([0; 32])
            || self.device_reality_result_digest == Digest32([0; 32])
            || self.device_reality_trust_head.sequence == 0
            || self.device_reality_trust_head.digest == Digest32([0; 32])
            || self.device_reality_policy_digest == Digest32([0; 32])
            || self.device_reality_key_digest == Digest32([0; 32])
            || self.semantic_head.generation == 0
            || self.semantic_head.digest == Digest32([0; 32])
        {
            return Err(PostSemanticControllerError::ZeroSecurityCommitment);
        }
        if !valid_device(&self.device) {
            return Err(PostSemanticControllerError::InvalidDeviceIdentity);
        }
        if self.admission_persisted_at_unix_ms == 0
            || self.device_reality_verified_at_unix_ms < self.admission_persisted_at_unix_ms
            || self.semantic_persisted_at_unix_ms < self.device_reality_verified_at_unix_ms
            || self.issued_at_unix_ms < self.semantic_persisted_at_unix_ms
        {
            return Err(PostSemanticControllerError::InvalidChallengeOrdering);
        }
        if self.issued_at_unix_ms >= self.device_reality_expires_at_unix_ms
            || self.issued_at_unix_ms >= self.effect_deadline_unix_ms
            || self.expires_at_unix_ms > self.device_reality_expires_at_unix_ms
            || self.expires_at_unix_ms > self.effect_deadline_unix_ms
        {
            return Err(PostSemanticControllerError::InvalidChallengeWindow);
        }
        let lifetime = self
            .expires_at_unix_ms
            .checked_sub(self.issued_at_unix_ms)
            .ok_or(PostSemanticControllerError::InvalidChallengeWindow)?;
        if lifetime == 0 || lifetime > MAX_POST_SEMANTIC_CONTROLLER_CHALLENGE_LIFETIME_MS {
            return Err(PostSemanticControllerError::InvalidChallengeWindow);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest32, PostSemanticControllerError> {
        self.validate()?;
        let mut h = blake3::Hasher::new();
        h.update(POST_SEMANTIC_CONTROLLER_CHALLENGE_DOMAIN);
        h.update(&self.schema_version.to_be_bytes());
        h.update(&self.nonce);
        h.update(&self.admission_reservation_head.generation.to_be_bytes());
        update_digest(&mut h, self.admission_reservation_head.digest);
        update_digest(&mut h, self.admission_challenge_digest);
        update_digest(&mut h, self.envelope_digest);
        update_digest(&mut h, self.transport_receipt_digest);
        h.update(&self.transport_trust_head.sequence.to_be_bytes());
        update_digest(&mut h, self.transport_trust_head.digest);
        update_digest(&mut h, self.config_digest);
        update_string(&mut h, &self.device.0);
        update_digest(&mut h, self.device_attestation_object_digest);
        update_digest(&mut h, self.device_reality_result_digest);
        h.update(&self.device_reality_trust_head.sequence.to_be_bytes());
        update_digest(&mut h, self.device_reality_trust_head.digest);
        update_digest(&mut h, self.device_reality_policy_digest);
        update_digest(&mut h, self.device_reality_key_digest);
        h.update(&self.semantic_head.generation.to_be_bytes());
        update_digest(&mut h, self.semantic_head.digest);
        h.update(&self.admission_persisted_at_unix_ms.to_be_bytes());
        h.update(&self.device_reality_verified_at_unix_ms.to_be_bytes());
        h.update(&self.semantic_persisted_at_unix_ms.to_be_bytes());
        h.update(&self.device_reality_expires_at_unix_ms.to_be_bytes());
        h.update(&self.effect_deadline_unix_ms.to_be_bytes());
        h.update(&self.issued_at_unix_ms.to_be_bytes());
        h.update(&self.expires_at_unix_ms.to_be_bytes());
        Ok(Digest32(*h.finalize().as_bytes()))
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, PostSemanticControllerError> {
        self.validate()?;
        bincode::serialize(self).map_err(PostSemanticControllerError::Encoding)
    }

    pub const fn admission_reservation_head(&self) -> AdmissionReservationHead {
        self.admission_reservation_head
    }

    pub const fn envelope_digest(&self) -> Digest32 {
        self.envelope_digest
    }

    pub const fn transport_trust_head(&self) -> TransportTrustHead {
        self.transport_trust_head
    }

    pub const fn semantic_head(&self) -> DeviceSemanticHead {
        self.semantic_head
    }

    pub fn device(&self) -> &ResourceRef {
        &self.device
    }

    pub const fn device_attestation_object_digest(&self) -> Digest32 {
        self.device_attestation_object_digest
    }

    pub const fn semantic_persisted_at_unix_ms(&self) -> u64 {
        self.semantic_persisted_at_unix_ms
    }

    pub const fn issued_at_unix_ms(&self) -> u64 {
        self.issued_at_unix_ms
    }

    pub const fn expires_at_unix_ms(&self) -> u64 {
        self.expires_at_unix_ms
    }

    pub const fn effect_deadline_unix_ms(&self) -> u64 {
        self.effect_deadline_unix_ms
    }

    pub const fn device_reality_expires_at_unix_ms(&self) -> u64 {
        self.device_reality_expires_at_unix_ms
    }

    #[cfg(test)]
    pub(crate) fn fixture() -> Self {
        Self {
            schema_version: POST_SEMANTIC_CONTROLLER_CHALLENGE_SCHEMA_VERSION,
            nonce: [0xA5; 32],
            admission_reservation_head: AdmissionReservationHead {
                generation: 2,
                digest: Digest32([1; 32]),
            },
            admission_challenge_digest: Digest32([2; 32]),
            envelope_digest: Digest32([3; 32]),
            transport_receipt_digest: Digest32([4; 32]),
            transport_trust_head: TransportTrustHead {
                sequence: 5,
                digest: Digest32([5; 32]),
            },
            config_digest: Digest32([6; 32]),
            device: ResourceRef("iot:valve:72".into()),
            device_attestation_object_digest: Digest32([7; 32]),
            device_reality_result_digest: Digest32([8; 32]),
            device_reality_trust_head: DeviceRealityTrustHead {
                sequence: 3,
                digest: Digest32([9; 32]),
            },
            device_reality_policy_digest: Digest32([10; 32]),
            device_reality_key_digest: Digest32([11; 32]),
            semantic_head: DeviceSemanticHead {
                generation: 4,
                digest: Digest32([12; 32]),
            },
            admission_persisted_at_unix_ms: 10_000,
            device_reality_verified_at_unix_ms: 11_000,
            semantic_persisted_at_unix_ms: 12_000,
            device_reality_expires_at_unix_ms: 16_000,
            effect_deadline_unix_ms: 16_000,
            issued_at_unix_ms: 12_100,
            expires_at_unix_ms: 14_100,
        }
    }

    #[cfg(test)]
    pub(crate) fn test_set_semantic_head(&mut self, head: DeviceSemanticHead) {
        self.semantic_head = head;
    }
}

fn system_unix_ms() -> Result<u64, PostSemanticControllerError> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| PostSemanticControllerError::SystemClockBeforeUnixEpoch)?;
    u64::try_from(elapsed.as_millis()).map_err(|_| PostSemanticControllerError::TimeOverflow)
}
