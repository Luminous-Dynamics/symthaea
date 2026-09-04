// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use symthaea_authority::{Digest32, ResourceRef};
use symthaea_iot_actuation_guard_admission_reservation::{
    AdmissionReservationHead, PersistedAdmissionReservation,
};
use symthaea_iot_transport_receipt::TransportTrustHead;

use crate::{
    ADMISSION_REALITY_CHALLENGE_SCHEMA_VERSION, AdmissionChallengeError, CHALLENGE_DOMAIN,
    MAX_ADMISSION_REALITY_CHALLENGE_LIFETIME_MS, update_digest, update_string, valid_device,
};

/// Privileged outbound challenge proving which exact crash-durable admission reservation
/// the next device appraisal must answer.
///
/// Fields are private and this type intentionally does **not** implement `Deserialize`.
/// Guard-side Rust code therefore cannot reconstruct it from caller bytes; construction
/// requires opaque persisted-reservation proof. Production issuance also obtains its nonce
/// and wall time internally rather than accepting caller-selected freshness inputs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AdmissionRealityChallengeV1 {
    schema_version: u16,
    nonce: [u8; 32],
    reservation_head: AdmissionReservationHead,
    envelope_digest: Digest32,
    transport_receipt_digest: Digest32,
    transport_trust_head: TransportTrustHead,
    config_digest: Digest32,
    device: ResourceRef,
    reservation_persisted_at_unix_ms: u64,
    issued_at_unix_ms: u64,
    effect_deadline_unix_ms: u64,
    expires_at_unix_ms: u64,
}

impl AdmissionRealityChallengeV1 {
    /// Issue one short-lived challenge from the exact opaque durable reservation.
    ///
    /// OS entropy and current wall time are read inside this function. Expiry is clipped
    /// to the already-authenticated physical-effect deadline; challenge issuance can never
    /// extend the command's existing authority/freshness window.
    pub fn issue_from_persisted_reservation(
        reservation: &PersistedAdmissionReservation,
    ) -> Result<Self, AdmissionChallengeError> {
        let mut nonce = [0u8; 32];
        getrandom::getrandom(&mut nonce).map_err(|_| AdmissionChallengeError::EntropyUnavailable)?;
        let issued_at_unix_ms = system_unix_ms()?;
        let effect_deadline_unix_ms = reservation
            .envelope()
            .send_not_after_unix_s
            .checked_mul(1_000)
            .ok_or(AdmissionChallengeError::TimeOverflow)?;
        let nominal_expiry = issued_at_unix_ms
            .checked_add(MAX_ADMISSION_REALITY_CHALLENGE_LIFETIME_MS)
            .ok_or(AdmissionChallengeError::TimeOverflow)?;
        let expires_at_unix_ms = nominal_expiry.min(effect_deadline_unix_ms);
        Self::from_parts(
            reservation,
            nonce,
            issued_at_unix_ms,
            expires_at_unix_ms,
            effect_deadline_unix_ms,
        )
    }

    fn from_parts(
        reservation: &PersistedAdmissionReservation,
        nonce: [u8; 32],
        issued_at_unix_ms: u64,
        expires_at_unix_ms: u64,
        effect_deadline_unix_ms: u64,
    ) -> Result<Self, AdmissionChallengeError> {
        let challenge = Self {
            schema_version: ADMISSION_REALITY_CHALLENGE_SCHEMA_VERSION,
            nonce,
            reservation_head: reservation.head(),
            envelope_digest: reservation.envelope_digest(),
            transport_receipt_digest: reservation.transport_receipt_digest(),
            transport_trust_head: reservation.transport_trust_head(),
            config_digest: reservation.checkpoint().config_digest,
            device: reservation.envelope().command.device.clone(),
            reservation_persisted_at_unix_ms: reservation.persisted_at_unix_ms(),
            issued_at_unix_ms,
            effect_deadline_unix_ms,
            expires_at_unix_ms,
        };
        challenge.validate()?;
        Ok(challenge)
    }

    pub fn validate(&self) -> Result<(), AdmissionChallengeError> {
        if self.schema_version != ADMISSION_REALITY_CHALLENGE_SCHEMA_VERSION {
            return Err(AdmissionChallengeError::UnsupportedChallengeSchema);
        }
        if self.nonce == [0; 32] {
            return Err(AdmissionChallengeError::ZeroChallengeNonce);
        }
        if self.reservation_head.generation == 0
            || self.reservation_head.digest == Digest32([0; 32])
            || self.envelope_digest == Digest32([0; 32])
            || self.transport_receipt_digest == Digest32([0; 32])
            || self.transport_trust_head.sequence == 0
            || self.transport_trust_head.digest == Digest32([0; 32])
            || self.config_digest == Digest32([0; 32])
        {
            return Err(AdmissionChallengeError::ZeroSecurityCommitment);
        }
        if !valid_device(&self.device) {
            return Err(AdmissionChallengeError::InvalidDeviceIdentity);
        }
        if self.issued_at_unix_ms < self.reservation_persisted_at_unix_ms
            || self.effect_deadline_unix_ms < self.issued_at_unix_ms
            || self.expires_at_unix_ms > self.effect_deadline_unix_ms
        {
            return Err(AdmissionChallengeError::InvalidChallengeOrdering);
        }
        let lifetime = self
            .expires_at_unix_ms
            .checked_sub(self.issued_at_unix_ms)
            .ok_or(AdmissionChallengeError::InvalidChallengeWindow)?;
        if lifetime == 0 || lifetime > MAX_ADMISSION_REALITY_CHALLENGE_LIFETIME_MS {
            return Err(AdmissionChallengeError::InvalidChallengeWindow);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest32, AdmissionChallengeError> {
        self.validate()?;
        let mut h = blake3::Hasher::new();
        h.update(CHALLENGE_DOMAIN);
        h.update(&self.schema_version.to_be_bytes());
        h.update(&self.nonce);
        h.update(&self.reservation_head.generation.to_be_bytes());
        update_digest(&mut h, self.reservation_head.digest);
        update_digest(&mut h, self.envelope_digest);
        update_digest(&mut h, self.transport_receipt_digest);
        h.update(&self.transport_trust_head.sequence.to_be_bytes());
        update_digest(&mut h, self.transport_trust_head.digest);
        update_digest(&mut h, self.config_digest);
        update_string(&mut h, &self.device.0);
        h.update(&self.reservation_persisted_at_unix_ms.to_be_bytes());
        h.update(&self.issued_at_unix_ms.to_be_bytes());
        h.update(&self.effect_deadline_unix_ms.to_be_bytes());
        h.update(&self.expires_at_unix_ms.to_be_bytes());
        Ok(Digest32(*h.finalize().as_bytes()))
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, AdmissionChallengeError> {
        self.validate()?;
        bincode::serialize(self).map_err(AdmissionChallengeError::Encoding)
    }

    pub fn is_fresh_at(&self, now_unix_ms: u64) -> bool {
        now_unix_ms >= self.issued_at_unix_ms && now_unix_ms < self.expires_at_unix_ms
    }

    pub const fn reservation_head(&self) -> AdmissionReservationHead {
        self.reservation_head
    }

    pub const fn envelope_digest(&self) -> Digest32 {
        self.envelope_digest
    }

    pub const fn transport_receipt_digest(&self) -> Digest32 {
        self.transport_receipt_digest
    }

    pub const fn transport_trust_head(&self) -> TransportTrustHead {
        self.transport_trust_head
    }

    pub const fn config_digest(&self) -> Digest32 {
        self.config_digest
    }

    pub fn device(&self) -> &ResourceRef {
        &self.device
    }

    pub const fn reservation_persisted_at_unix_ms(&self) -> u64 {
        self.reservation_persisted_at_unix_ms
    }

    pub const fn issued_at_unix_ms(&self) -> u64 {
        self.issued_at_unix_ms
    }

    pub const fn effect_deadline_unix_ms(&self) -> u64 {
        self.effect_deadline_unix_ms
    }

    pub const fn expires_at_unix_ms(&self) -> u64 {
        self.expires_at_unix_ms
    }

    #[cfg(test)]
    pub(crate) fn fixture() -> Self {
        Self {
            schema_version: ADMISSION_REALITY_CHALLENGE_SCHEMA_VERSION,
            nonce: [0xA5; 32],
            reservation_head: AdmissionReservationHead {
                generation: 4,
                digest: Digest32([1; 32]),
            },
            envelope_digest: Digest32([2; 32]),
            transport_receipt_digest: Digest32([3; 32]),
            transport_trust_head: TransportTrustHead {
                sequence: 5,
                digest: Digest32([4; 32]),
            },
            config_digest: Digest32([5; 32]),
            device: ResourceRef("iot:valve:72".into()),
            reservation_persisted_at_unix_ms: 10_000,
            issued_at_unix_ms: 10_100,
            effect_deadline_unix_ms: 14_000,
            expires_at_unix_ms: 13_000,
        }
    }

    #[cfg(test)]
    pub(crate) fn test_set_reservation_digest(&mut self, digest: Digest32) {
        self.reservation_head.digest = digest;
    }
}

fn system_unix_ms() -> Result<u64, AdmissionChallengeError> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| AdmissionChallengeError::SystemClockBeforeUnixEpoch)?;
    u64::try_from(elapsed.as_millis()).map_err(|_| AdmissionChallengeError::TimeOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn challenge_names_admission_reservation_not_semantic_state() {
        let challenge = AdmissionRealityChallengeV1::fixture();
        challenge.validate().unwrap();
        assert_eq!(challenge.reservation_head().generation, 4);
        assert_eq!(challenge.config_digest(), Digest32([5; 32]));
        assert_eq!(challenge.device().0, "iot:valve:72");
    }

    #[test]
    fn zero_nonce_and_post_deadline_challenges_fail_closed() {
        let mut zero = AdmissionRealityChallengeV1::fixture();
        zero.nonce = [0; 32];
        assert!(matches!(
            zero.validate(),
            Err(AdmissionChallengeError::ZeroChallengeNonce)
        ));

        let mut late = AdmissionRealityChallengeV1::fixture();
        late.expires_at_unix_ms = late.effect_deadline_unix_ms + 1;
        assert!(matches!(
            late.validate(),
            Err(AdmissionChallengeError::InvalidChallengeOrdering)
        ));
    }

    #[test]
    fn reservation_commitment_changes_challenge_digest() {
        let a = AdmissionRealityChallengeV1::fixture();
        let mut b = a.clone();
        b.test_set_reservation_digest(Digest32([0x99; 32]));
        assert_ne!(a.digest().unwrap(), b.digest().unwrap());
    }
}
