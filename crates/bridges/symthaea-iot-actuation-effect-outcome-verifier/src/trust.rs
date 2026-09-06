// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::{BTreeMap, BTreeSet};

use ed25519_dalek::VerifyingKey;
use symthaea_authority::Digest32;

use crate::{
    EFFECT_OUTCOME_ED25519_ALGORITHM, EffectOutcomeError, MAX_EFFECT_OUTCOME_EVIDENCE_LIFETIME_MS,
    MAX_EFFECT_OUTCOME_ID_BYTES, PhysicalEffectOutcomeEvidenceBodyV1, valid_id,
};

pub const EFFECT_OUTCOME_TRUST_SCHEMA_VERSION: u16 = 1;
pub const MAX_EFFECT_OUTCOME_VERIFIER_KEYS: usize = 4096;

const EFFECT_OUTCOME_KEY_DOMAIN: &[u8] = b"symthaea-iot-effect-outcome-verifier-key-v1\0";
const EFFECT_OUTCOME_TRUST_DOMAIN: &[u8] = b"symthaea-iot-effect-outcome-trust-v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectOutcomeVerifierKeyStatus {
    Active,
    Retired,
    Revoked,
}

impl EffectOutcomeVerifierKeyStatus {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectOutcomeVerifierKeyV1 {
    pub verifier_id: String,
    pub key_id: String,
    pub algorithm: String,
    pub public_key: [u8; 32],
    pub status: EffectOutcomeVerifierKeyStatus,
    pub not_before_unix_ms: u64,
    pub not_after_unix_ms: u64,
    pub max_evidence_lifetime_ms: u64,
}

impl EffectOutcomeVerifierKeyV1 {
    pub fn validate(&self) -> Result<(), EffectOutcomeError> {
        if !valid_id(&self.verifier_id, MAX_EFFECT_OUTCOME_ID_BYTES)
            || !valid_id(&self.key_id, MAX_EFFECT_OUTCOME_ID_BYTES)
        {
            return Err(EffectOutcomeError::InvalidVerifierKeyIdentity);
        }
        if self.algorithm != EFFECT_OUTCOME_ED25519_ALGORITHM {
            return Err(EffectOutcomeError::UnsupportedVerifierAlgorithm);
        }
        if self.public_key == [0; 32] || VerifyingKey::from_bytes(&self.public_key).is_err() {
            return Err(EffectOutcomeError::InvalidVerifierPublicKey);
        }
        if self.not_before_unix_ms >= self.not_after_unix_ms {
            return Err(EffectOutcomeError::InvalidVerifierKeyWindow);
        }
        if self.max_evidence_lifetime_ms == 0
            || self.max_evidence_lifetime_ms > MAX_EFFECT_OUTCOME_EVIDENCE_LIFETIME_MS
        {
            return Err(EffectOutcomeError::InvalidVerifierKeyEvidenceLifetime);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest32, EffectOutcomeError> {
        self.validate()?;
        let mut h = blake3::Hasher::new();
        h.update(EFFECT_OUTCOME_KEY_DOMAIN);
        update_string(&mut h, &self.verifier_id);
        update_string(&mut h, &self.key_id);
        update_string(&mut h, &self.algorithm);
        h.update(&self.public_key);
        h.update(&[self.status.tag()]);
        h.update(&self.not_before_unix_ms.to_be_bytes());
        h.update(&self.not_after_unix_ms.to_be_bytes());
        h.update(&self.max_evidence_lifetime_ms.to_be_bytes());
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

    fn active_at(&self, unix_ms: u64) -> bool {
        self.status == EffectOutcomeVerifierKeyStatus::Active
            && unix_ms >= self.not_before_unix_ms
            && unix_ms < self.not_after_unix_ms
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectOutcomeTrustSnapshotV1 {
    pub schema_version: u16,
    pub sequence: u64,
    pub issued_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
    pub previous_snapshot_digest: Option<Digest32>,
    pub keys: Vec<EffectOutcomeVerifierKeyV1>,
}

impl EffectOutcomeTrustSnapshotV1 {
    pub fn validate(&self) -> Result<(), EffectOutcomeError> {
        if self.schema_version != EFFECT_OUTCOME_TRUST_SCHEMA_VERSION {
            return Err(EffectOutcomeError::UnsupportedTrustSchema);
        }
        if self.sequence == 0 {
            return Err(EffectOutcomeError::TrustSequenceZero);
        }
        if self.issued_at_unix_ms >= self.expires_at_unix_ms {
            return Err(EffectOutcomeError::InvalidTrustWindow);
        }
        if self.keys.is_empty() || self.keys.len() > MAX_EFFECT_OUTCOME_VERIFIER_KEYS {
            return Err(EffectOutcomeError::InvalidTrustKeyCount);
        }
        if self.sequence == 1 && self.previous_snapshot_digest.is_some() {
            return Err(EffectOutcomeError::GenesisHasPredecessor);
        }
        if self.sequence > 1 && self.previous_snapshot_digest.is_none() {
            return Err(EffectOutcomeError::SuccessorMissingPredecessor);
        }
        let mut identities = BTreeSet::new();
        for key in &self.keys {
            key.validate()?;
            if !identities.insert((key.verifier_id.clone(), key.key_id.clone())) {
                return Err(EffectOutcomeError::DuplicateVerifierKeyIdentity);
            }
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest32, EffectOutcomeError> {
        self.validate()?;
        let mut keys = self.keys.iter().collect::<Vec<_>>();
        keys.sort_by(|a, b| a.identity().cmp(&b.identity()));
        let mut h = blake3::Hasher::new();
        h.update(EFFECT_OUTCOME_TRUST_DOMAIN);
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectOutcomeTrustHead {
    pub sequence: u64,
    pub digest: Digest32,
}

#[derive(Debug)]
pub struct EffectOutcomeTrustRegistry {
    snapshot: EffectOutcomeTrustSnapshotV1,
    head: EffectOutcomeTrustHead,
}

impl EffectOutcomeTrustRegistry {
    pub fn genesis(snapshot: EffectOutcomeTrustSnapshotV1) -> Result<Self, EffectOutcomeError> {
        snapshot.validate()?;
        if snapshot.sequence != 1 || snapshot.previous_snapshot_digest.is_some() {
            return Err(EffectOutcomeError::NotGenesis);
        }
        let head = EffectOutcomeTrustHead {
            sequence: snapshot.sequence,
            digest: snapshot.digest()?,
        };
        Ok(Self { snapshot, head })
    }

    pub fn successor(
        &self,
        snapshot: EffectOutcomeTrustSnapshotV1,
    ) -> Result<Self, EffectOutcomeError> {
        snapshot.validate()?;
        let expected = self
            .head
            .sequence
            .checked_add(1)
            .ok_or(EffectOutcomeError::TrustSequenceOverflow)?;
        if snapshot.sequence != expected {
            return Err(EffectOutcomeError::TrustSequenceNotNext {
                expected,
                proposed: snapshot.sequence,
            });
        }
        if snapshot.previous_snapshot_digest != Some(self.head.digest) {
            return Err(EffectOutcomeError::TrustPredecessorMismatch);
        }
        if snapshot.issued_at_unix_ms < self.snapshot.issued_at_unix_ms {
            return Err(EffectOutcomeError::TrustIssuedAtRegressed);
        }
        validate_successor(&self.snapshot, &snapshot)?;
        let head = EffectOutcomeTrustHead {
            sequence: snapshot.sequence,
            digest: snapshot.digest()?,
        };
        Ok(Self { snapshot, head })
    }

    pub fn restore(
        snapshot: EffectOutcomeTrustSnapshotV1,
        trusted_head: EffectOutcomeTrustHead,
    ) -> Result<Self, EffectOutcomeError> {
        snapshot.validate()?;
        let head = EffectOutcomeTrustHead {
            sequence: snapshot.sequence,
            digest: snapshot.digest()?,
        };
        if head != trusted_head {
            return Err(EffectOutcomeError::TrustedHeadMismatch);
        }
        Ok(Self { snapshot, head })
    }

    pub const fn head(&self) -> EffectOutcomeTrustHead {
        self.head
    }

    pub fn snapshot(&self) -> &EffectOutcomeTrustSnapshotV1 {
        &self.snapshot
    }

    pub(crate) fn exact_active_key(
        &self,
        body: &PhysicalEffectOutcomeEvidenceBodyV1,
        now_unix_ms: u64,
    ) -> Result<&EffectOutcomeVerifierKeyV1, EffectOutcomeError> {
        if now_unix_ms < self.snapshot.issued_at_unix_ms
            || now_unix_ms >= self.snapshot.expires_at_unix_ms
        {
            return Err(EffectOutcomeError::TrustSnapshotNotFresh);
        }
        let key = self
            .snapshot
            .keys
            .iter()
            .find(|key| {
                key.verifier_id == body.verifier_id
                    && key.key_id == body.key_id
                    && key.algorithm == body.algorithm
                    && key.status == EffectOutcomeVerifierKeyStatus::Active
            })
            .ok_or(EffectOutcomeError::NoActiveVerifierKey)?;
        if !key.active_at(body.evidence_issued_at_unix_ms) || !key.active_at(now_unix_ms) {
            return Err(EffectOutcomeError::VerifierKeyNotActive);
        }
        Ok(key)
    }
}

fn validate_successor(
    previous: &EffectOutcomeTrustSnapshotV1,
    next: &EffectOutcomeTrustSnapshotV1,
) -> Result<(), EffectOutcomeError> {
    let next_map = next
        .keys
        .iter()
        .map(|key| ((key.verifier_id.as_str(), key.key_id.as_str()), key))
        .collect::<BTreeMap<_, _>>();

    for old in &previous.keys {
        let identity = (old.verifier_id.as_str(), old.key_id.as_str());
        let new = next_map
            .get(&identity)
            .ok_or(EffectOutcomeError::TrustedKeyDeleted)?;
        if !old.immutable_identity_eq(new) {
            return Err(EffectOutcomeError::TrustedKeyMutated);
        }
        if !old.status.transition_allowed(new.status) {
            return Err(EffectOutcomeError::TrustedKeyReactivated);
        }
        if new.not_after_unix_ms > old.not_after_unix_ms {
            return Err(EffectOutcomeError::TrustedKeyExpiryExtended);
        }
        if new.max_evidence_lifetime_ms > old.max_evidence_lifetime_ms {
            return Err(EffectOutcomeError::TrustedKeyEvidenceLifetimeExtended);
        }
    }
    Ok(())
}

fn update_string(h: &mut blake3::Hasher, value: &str) {
    h.update(&(value.len() as u32).to_be_bytes());
    h.update(value.as_bytes());
}

fn update_digest(h: &mut blake3::Hasher, Digest32(bytes): Digest32) {
    h.update(&bytes);
}
