// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::{BTreeMap, BTreeSet};

use ed25519_dalek::VerifyingKey;
use serde::{Deserialize, Serialize};
use symthaea_authority::Digest32;
use symthaea_iot_posture::DeviceAttestationResultBodyV1;

use crate::{
    DEVICE_REALITY_ED25519_ALGORITHM, DeviceRealityError, MAX_DEVICE_REALITY_RESULT_LIFETIME_MS,
    valid_id,
};

pub const DEVICE_REALITY_TRUST_SCHEMA_VERSION: u16 = 1;
pub const MAX_DEVICE_REALITY_VERIFIER_KEYS: usize = 4096;

const DEVICE_REALITY_KEY_DOMAIN: &[u8] = b"symthaea-iot-device-reality-verifier-key-v1\0";
const DEVICE_REALITY_TRUST_DOMAIN: &[u8] = b"symthaea-iot-device-reality-trust-v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceRealityVerifierKeyStatus {
    Active,
    Retired,
    Revoked,
}

impl DeviceRealityVerifierKeyStatus {
    const fn tag(self) -> u8 {
        match self {
            Self::Active => 0,
            Self::Retired => 1,
            Self::Revoked => 2,
        }
    }

    const fn transition_allowed(self, next: Self) -> bool {
        match self {
            Self::Active => true,
            Self::Retired => matches!(next, Self::Retired | Self::Revoked),
            Self::Revoked => matches!(next, Self::Revoked),
        }
    }
}

/// One exact device-appraisal verifier key committed inside anti-rollback trust state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceRealityVerifierKeyV1 {
    pub verifier_id: String,
    pub key_id: String,
    pub algorithm: String,
    pub public_key: [u8; 32],
    pub status: DeviceRealityVerifierKeyStatus,
    pub not_before_unix_ms: u64,
    pub not_after_unix_ms: u64,
    pub max_result_lifetime_ms: u64,
}

impl DeviceRealityVerifierKeyV1 {
    pub fn validate(&self) -> Result<(), DeviceRealityError> {
        if !valid_id(&self.verifier_id) || !valid_id(&self.key_id) {
            return Err(DeviceRealityError::InvalidVerifierKeyIdentity);
        }
        if self.algorithm != DEVICE_REALITY_ED25519_ALGORITHM {
            return Err(DeviceRealityError::UnsupportedVerifierAlgorithm);
        }
        if self.public_key == [0; 32] || VerifyingKey::from_bytes(&self.public_key).is_err() {
            return Err(DeviceRealityError::InvalidVerifierPublicKey);
        }
        if self.not_before_unix_ms >= self.not_after_unix_ms {
            return Err(DeviceRealityError::InvalidVerifierKeyWindow);
        }
        if self.max_result_lifetime_ms == 0
            || self.max_result_lifetime_ms > MAX_DEVICE_REALITY_RESULT_LIFETIME_MS
        {
            return Err(DeviceRealityError::InvalidVerifierKeyResultLifetime);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest32, DeviceRealityError> {
        self.validate()?;
        let mut h = blake3::Hasher::new();
        h.update(DEVICE_REALITY_KEY_DOMAIN);
        update_string(&mut h, &self.verifier_id);
        update_string(&mut h, &self.key_id);
        update_string(&mut h, &self.algorithm);
        h.update(&self.public_key);
        h.update(&[self.status.tag()]);
        h.update(&self.not_before_unix_ms.to_be_bytes());
        h.update(&self.not_after_unix_ms.to_be_bytes());
        h.update(&self.max_result_lifetime_ms.to_be_bytes());
        Ok(Digest32(*h.finalize().as_bytes()))
    }

    fn identity(&self) -> (&str, &str) {
        (&self.verifier_id, &self.key_id)
    }

    fn immutable_identity_eq(&self, other: &Self) -> bool {
        self.verifier_id == other.verifier_id
            && self.key_id == other.key_id
            && self.algorithm == other.algorithm
            && self.public_key == other.public_key
            && self.not_before_unix_ms == other.not_before_unix_ms
    }

    pub(crate) fn active_at(&self, unix_ms: u64) -> bool {
        self.status == DeviceRealityVerifierKeyStatus::Active
            && unix_ms >= self.not_before_unix_ms
            && unix_ms < self.not_after_unix_ms
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceRealityTrustSnapshotV1 {
    pub schema_version: u16,
    pub sequence: u64,
    pub issued_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
    pub previous_snapshot_digest: Option<Digest32>,
    pub keys: Vec<DeviceRealityVerifierKeyV1>,
}

impl DeviceRealityTrustSnapshotV1 {
    pub fn validate(&self) -> Result<(), DeviceRealityError> {
        if self.schema_version != DEVICE_REALITY_TRUST_SCHEMA_VERSION {
            return Err(DeviceRealityError::UnsupportedTrustSchema);
        }
        if self.sequence == 0 {
            return Err(DeviceRealityError::TrustSequenceZero);
        }
        if self.issued_at_unix_ms >= self.expires_at_unix_ms {
            return Err(DeviceRealityError::InvalidTrustWindow);
        }
        if self.keys.is_empty() || self.keys.len() > MAX_DEVICE_REALITY_VERIFIER_KEYS {
            return Err(DeviceRealityError::InvalidTrustKeyCount);
        }
        if self.sequence == 1 && self.previous_snapshot_digest.is_some() {
            return Err(DeviceRealityError::GenesisHasPredecessor);
        }
        if self.sequence > 1 && self.previous_snapshot_digest.is_none() {
            return Err(DeviceRealityError::SuccessorMissingPredecessor);
        }
        let mut identities = BTreeSet::new();
        for key in &self.keys {
            key.validate()?;
            if !identities.insert((key.verifier_id.clone(), key.key_id.clone())) {
                return Err(DeviceRealityError::DuplicateVerifierKeyIdentity);
            }
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest32, DeviceRealityError> {
        self.validate()?;
        let mut keys = self.keys.iter().collect::<Vec<_>>();
        keys.sort_by(|a, b| a.identity().cmp(&b.identity()));

        let mut h = blake3::Hasher::new();
        h.update(DEVICE_REALITY_TRUST_DOMAIN);
        h.update(&self.schema_version.to_be_bytes());
        h.update(&self.sequence.to_be_bytes());
        h.update(&self.issued_at_unix_ms.to_be_bytes());
        h.update(&self.expires_at_unix_ms.to_be_bytes());
        match self.previous_snapshot_digest {
            Some(digest) => {
                h.update(&[1]);
                update_digest(&mut h, digest);
            }
            None => {
                h.update(&[0]);
            }
        }
        h.update(&(keys.len() as u32).to_be_bytes());
        for key in keys {
            update_digest(&mut h, key.digest()?);
        }
        Ok(Digest32(*h.finalize().as_bytes()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceRealityTrustHead {
    pub sequence: u64,
    pub digest: Digest32,
}

/// Anti-rollback trust registry whose head must be retained independently from the
/// serialized snapshot itself.
#[derive(Debug)]
pub struct DeviceRealityTrustRegistry {
    snapshot: DeviceRealityTrustSnapshotV1,
    head: DeviceRealityTrustHead,
}

impl DeviceRealityTrustRegistry {
    pub fn genesis(snapshot: DeviceRealityTrustSnapshotV1) -> Result<Self, DeviceRealityError> {
        snapshot.validate()?;
        if snapshot.sequence != 1 || snapshot.previous_snapshot_digest.is_some() {
            return Err(DeviceRealityError::NotGenesis);
        }
        let head = DeviceRealityTrustHead {
            sequence: snapshot.sequence,
            digest: snapshot.digest()?,
        };
        Ok(Self { snapshot, head })
    }

    pub fn successor(
        &self,
        snapshot: DeviceRealityTrustSnapshotV1,
    ) -> Result<Self, DeviceRealityError> {
        snapshot.validate()?;
        let expected = self
            .head
            .sequence
            .checked_add(1)
            .ok_or(DeviceRealityError::TrustSequenceOverflow)?;
        if snapshot.sequence != expected {
            return Err(DeviceRealityError::TrustSequenceNotNext {
                expected,
                proposed: snapshot.sequence,
            });
        }
        if snapshot.previous_snapshot_digest != Some(self.head.digest) {
            return Err(DeviceRealityError::TrustPredecessorMismatch);
        }
        if snapshot.issued_at_unix_ms < self.snapshot.issued_at_unix_ms {
            return Err(DeviceRealityError::TrustIssuedAtRegressed);
        }
        validate_successor(&self.snapshot, &snapshot)?;
        let head = DeviceRealityTrustHead {
            sequence: snapshot.sequence,
            digest: snapshot.digest()?,
        };
        Ok(Self { snapshot, head })
    }

    pub fn restore(
        snapshot: DeviceRealityTrustSnapshotV1,
        trusted_head: DeviceRealityTrustHead,
    ) -> Result<Self, DeviceRealityError> {
        snapshot.validate()?;
        let head = DeviceRealityTrustHead {
            sequence: snapshot.sequence,
            digest: snapshot.digest()?,
        };
        if head != trusted_head {
            return Err(DeviceRealityError::TrustedHeadMismatch);
        }
        Ok(Self { snapshot, head })
    }

    pub const fn head(&self) -> DeviceRealityTrustHead {
        self.head
    }

    pub fn snapshot(&self) -> &DeviceRealityTrustSnapshotV1 {
        &self.snapshot
    }

    pub(crate) fn exact_active_key(
        &self,
        body: &DeviceAttestationResultBodyV1,
        appraised_lower_bound_unix_ms: u64,
        now_unix_ms: u64,
    ) -> Result<&DeviceRealityVerifierKeyV1, DeviceRealityError> {
        if now_unix_ms < self.snapshot.issued_at_unix_ms
            || now_unix_ms >= self.snapshot.expires_at_unix_ms
        {
            return Err(DeviceRealityError::TrustSnapshotNotFresh);
        }
        let key = self
            .snapshot
            .keys
            .iter()
            .find(|key| {
                key.verifier_id == body.verifier_id
                    && key.key_id == body.key_id
                    && key.algorithm == body.algorithm
                    && key.status == DeviceRealityVerifierKeyStatus::Active
            })
            .ok_or(DeviceRealityError::NoActiveVerifierKey)?;
        if !key.active_at(appraised_lower_bound_unix_ms) || !key.active_at(now_unix_ms) {
            return Err(DeviceRealityError::VerifierKeyNotActive);
        }
        Ok(key)
    }
}

fn validate_successor(
    previous: &DeviceRealityTrustSnapshotV1,
    next: &DeviceRealityTrustSnapshotV1,
) -> Result<(), DeviceRealityError> {
    let next_map = next
        .keys
        .iter()
        .map(|key| ((key.verifier_id.as_str(), key.key_id.as_str()), key))
        .collect::<BTreeMap<_, _>>();

    for old in &previous.keys {
        let identity = (old.verifier_id.as_str(), old.key_id.as_str());
        let new = next_map
            .get(&identity)
            .ok_or(DeviceRealityError::TrustedKeyDeleted)?;
        if !old.immutable_identity_eq(new) {
            return Err(DeviceRealityError::TrustedKeyMutated);
        }
        if !old.status.transition_allowed(new.status) {
            return Err(DeviceRealityError::TrustedKeyReactivated);
        }
        if new.not_after_unix_ms > old.not_after_unix_ms {
            return Err(DeviceRealityError::TrustedKeyExpiryExtended);
        }
        if new.max_result_lifetime_ms > old.max_result_lifetime_ms {
            return Err(DeviceRealityError::TrustedKeyResultLifetimeExtended);
        }
    }
    Ok(())
}

fn update_string(h: &mut blake3::Hasher, value: &str) {
    h.update(&(value.len() as u32).to_be_bytes());
    h.update(value.as_bytes());
}

fn update_digest(h: &mut blake3::Hasher, Digest32(value): Digest32) {
    h.update(&value);
}

#[cfg(test)]
mod tests {
    use ed25519_dalek::SigningKey;

    use super::*;

    fn signing_key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    fn key(
        key_id: &str,
        signing_key: &SigningKey,
        status: DeviceRealityVerifierKeyStatus,
    ) -> DeviceRealityVerifierKeyV1 {
        DeviceRealityVerifierKeyV1 {
            verifier_id: "verifier:fleet-a".into(),
            key_id: key_id.into(),
            algorithm: DEVICE_REALITY_ED25519_ALGORITHM.into(),
            public_key: signing_key.verifying_key().to_bytes(),
            status,
            not_before_unix_ms: 5_000,
            not_after_unix_ms: 20_000,
            max_result_lifetime_ms: 3_000,
        }
    }

    fn registry() -> DeviceRealityTrustRegistry {
        let signing_key = signing_key(0x61);
        DeviceRealityTrustRegistry::genesis(DeviceRealityTrustSnapshotV1 {
            schema_version: DEVICE_REALITY_TRUST_SCHEMA_VERSION,
            sequence: 1,
            issued_at_unix_ms: 5_000,
            expires_at_unix_ms: 20_000,
            previous_snapshot_digest: None,
            keys: vec![key(
                "device-key-1",
                &signing_key,
                DeviceRealityVerifierKeyStatus::Active,
            )],
        })
        .unwrap()
    }

    #[test]
    fn public_key_cannot_mutate_under_existing_key_id() {
        let base = registry();
        let other = signing_key(0x62);
        let mut mutated = base.snapshot().keys[0].clone();
        mutated.public_key = other.verifying_key().to_bytes();
        let next = DeviceRealityTrustSnapshotV1 {
            schema_version: DEVICE_REALITY_TRUST_SCHEMA_VERSION,
            sequence: 2,
            issued_at_unix_ms: 6_000,
            expires_at_unix_ms: 20_000,
            previous_snapshot_digest: Some(base.head().digest),
            keys: vec![mutated],
        };
        assert!(matches!(
            base.successor(next),
            Err(DeviceRealityError::TrustedKeyMutated)
        ));
    }

    #[test]
    fn existing_key_expiry_and_result_lifetime_cannot_expand() {
        let base = registry();
        let mut extended_expiry = base.snapshot().keys[0].clone();
        extended_expiry.not_after_unix_ms += 1;
        let next = DeviceRealityTrustSnapshotV1 {
            schema_version: DEVICE_REALITY_TRUST_SCHEMA_VERSION,
            sequence: 2,
            issued_at_unix_ms: 6_000,
            expires_at_unix_ms: 21_000,
            previous_snapshot_digest: Some(base.head().digest),
            keys: vec![extended_expiry],
        };
        assert!(matches!(
            base.successor(next),
            Err(DeviceRealityError::TrustedKeyExpiryExtended)
        ));

        let base = registry();
        let mut widened_lifetime = base.snapshot().keys[0].clone();
        widened_lifetime.max_result_lifetime_ms += 1;
        let next = DeviceRealityTrustSnapshotV1 {
            schema_version: DEVICE_REALITY_TRUST_SCHEMA_VERSION,
            sequence: 2,
            issued_at_unix_ms: 6_000,
            expires_at_unix_ms: 20_000,
            previous_snapshot_digest: Some(base.head().digest),
            keys: vec![widened_lifetime],
        };
        assert!(matches!(
            base.successor(next),
            Err(DeviceRealityError::TrustedKeyResultLifetimeExtended)
        ));
    }

    #[test]
    fn retired_or_revoked_key_cannot_reactivate() {
        let base = registry();
        let mut retired_key = base.snapshot().keys[0].clone();
        retired_key.status = DeviceRealityVerifierKeyStatus::Retired;
        retired_key.not_after_unix_ms = 18_000;
        let retired = base
            .successor(DeviceRealityTrustSnapshotV1 {
                schema_version: DEVICE_REALITY_TRUST_SCHEMA_VERSION,
                sequence: 2,
                issued_at_unix_ms: 6_000,
                expires_at_unix_ms: 20_000,
                previous_snapshot_digest: Some(base.head().digest),
                keys: vec![retired_key],
            })
            .unwrap();

        let mut reactivated = retired.snapshot().keys[0].clone();
        reactivated.status = DeviceRealityVerifierKeyStatus::Active;
        let next = DeviceRealityTrustSnapshotV1 {
            schema_version: DEVICE_REALITY_TRUST_SCHEMA_VERSION,
            sequence: 3,
            issued_at_unix_ms: 7_000,
            expires_at_unix_ms: 20_000,
            previous_snapshot_digest: Some(retired.head().digest),
            keys: vec![reactivated],
        };
        assert!(matches!(
            retired.successor(next),
            Err(DeviceRealityError::TrustedKeyReactivated)
        ));
    }

    #[test]
    fn rotation_requires_a_new_key_id_and_preserves_old_identity() {
        let base = registry();
        let mut old = base.snapshot().keys[0].clone();
        old.status = DeviceRealityVerifierKeyStatus::Retired;
        old.not_after_unix_ms = 18_000;
        let new_signing_key = signing_key(0x62);
        let mut new = key(
            "device-key-2",
            &new_signing_key,
            DeviceRealityVerifierKeyStatus::Active,
        );
        new.not_before_unix_ms = 6_000;

        let rotated = base
            .successor(DeviceRealityTrustSnapshotV1 {
                schema_version: DEVICE_REALITY_TRUST_SCHEMA_VERSION,
                sequence: 2,
                issued_at_unix_ms: 6_000,
                expires_at_unix_ms: 20_000,
                previous_snapshot_digest: Some(base.head().digest),
                keys: vec![old, new],
            })
            .unwrap();
        assert_eq!(rotated.head().sequence, 2);
        assert_eq!(rotated.snapshot().keys.len(), 2);
    }
}
