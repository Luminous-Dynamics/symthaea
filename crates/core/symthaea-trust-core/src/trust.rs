// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Domain-neutral lifecycle-governed trust snapshots.
//!
//! Cryptographic validity is necessary but not sufficient authority. A key must
//! also be known, active, purpose-authorized, inside its validity window, and
//! evaluated against the current snapshot accepted by a monotonic tracker.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::attestation::{SignatureAlgorithm, digest_signature_algorithm};
use crate::identity::{FramedDigest, Sha256Digest, TrustUsage};

pub const TRUST_SNAPSHOT_SCHEMA: &str = "symthaea.trust-snapshot.v1";
const TRUST_SNAPSHOT_DIGEST_DOMAIN: &str = "symthaea.trust-snapshot.identity.v1";
pub const MAX_TRUST_KEYS: usize = 4096;
pub const MAX_KEY_ID_BYTES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyLifecycleStatus {
    Active,
    Retired,
    Revoked,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyTrustRecord {
    pub algorithm: SignatureAlgorithm,
    pub key_id: String,
    pub not_before_unix_s: u64,
    pub not_after_unix_s: Option<u64>,
    pub status: KeyLifecycleStatus,
    pub usages: BTreeSet<TrustUsage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustSnapshot {
    pub schema_version: String,
    pub sequence: u64,
    pub issued_at_unix_s: u64,
    pub expires_at_unix_s: u64,
    pub keys: Vec<KeyTrustRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustSnapshotError {
    UnsupportedSchema,
    SequenceZero,
    InvalidSnapshotWindow,
    EmptySnapshot,
    TooManyKeys { actual: usize, maximum: usize },
    InvalidAlgorithm,
    EmptyKeyId,
    NonCanonicalKeyId { key_id: String },
    KeyIdTooLong { actual: usize, maximum: usize },
    EmptyUsages { key_id: String },
    InvalidKeyWindow { key_id: String },
    DuplicateKey { algorithm: SignatureAlgorithm, key_id: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyEligibility {
    Eligible,
    Unknown,
    NotYetValid,
    Expired,
    Retired,
    Revoked,
    UsageNotAllowed,
}

impl TrustSnapshot {
    pub fn new(
        sequence: u64,
        issued_at_unix_s: u64,
        expires_at_unix_s: u64,
        keys: Vec<KeyTrustRecord>,
    ) -> Result<Self, TrustSnapshotError> {
        let mut snapshot = Self {
            schema_version: TRUST_SNAPSHOT_SCHEMA.into(),
            sequence,
            issued_at_unix_s,
            expires_at_unix_s,
            keys,
        };
        snapshot.canonicalize();
        snapshot.validate()?;
        Ok(snapshot)
    }

    pub fn canonicalize(&mut self) {
        self.keys.sort_by(|left, right| {
            left.algorithm
                .cmp(&right.algorithm)
                .then(left.key_id.cmp(&right.key_id))
        });
    }

    pub fn validate(&self) -> Result<(), TrustSnapshotError> {
        if self.schema_version != TRUST_SNAPSHOT_SCHEMA {
            return Err(TrustSnapshotError::UnsupportedSchema);
        }
        if self.sequence == 0 {
            return Err(TrustSnapshotError::SequenceZero);
        }
        if self.issued_at_unix_s >= self.expires_at_unix_s {
            return Err(TrustSnapshotError::InvalidSnapshotWindow);
        }
        if self.keys.is_empty() {
            return Err(TrustSnapshotError::EmptySnapshot);
        }
        if self.keys.len() > MAX_TRUST_KEYS {
            return Err(TrustSnapshotError::TooManyKeys {
                actual: self.keys.len(),
                maximum: MAX_TRUST_KEYS,
            });
        }

        let mut identities = BTreeSet::new();
        for key in &self.keys {
            if !key.algorithm.is_canonical() {
                return Err(TrustSnapshotError::InvalidAlgorithm);
            }
            let canonical_key_id = key.key_id.trim();
            if canonical_key_id.is_empty() {
                return Err(TrustSnapshotError::EmptyKeyId);
            }
            if canonical_key_id != key.key_id {
                return Err(TrustSnapshotError::NonCanonicalKeyId {
                    key_id: key.key_id.clone(),
                });
            }
            if key.key_id.len() > MAX_KEY_ID_BYTES {
                return Err(TrustSnapshotError::KeyIdTooLong {
                    actual: key.key_id.len(),
                    maximum: MAX_KEY_ID_BYTES,
                });
            }
            if key.usages.is_empty() {
                return Err(TrustSnapshotError::EmptyUsages {
                    key_id: key.key_id.clone(),
                });
            }
            if key
                .not_after_unix_s
                .is_some_and(|not_after| not_after <= key.not_before_unix_s)
            {
                return Err(TrustSnapshotError::InvalidKeyWindow {
                    key_id: key.key_id.clone(),
                });
            }
            if !identities.insert((key.algorithm.clone(), key.key_id.clone())) {
                return Err(TrustSnapshotError::DuplicateKey {
                    algorithm: key.algorithm.clone(),
                    key_id: key.key_id.clone(),
                });
            }
        }
        Ok(())
    }

    pub fn is_fresh_at(&self, unix_s: u64) -> bool {
        unix_s >= self.issued_at_unix_s && unix_s < self.expires_at_unix_s
    }

    pub fn key_eligibility(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        usage: &TrustUsage,
        unix_s: u64,
    ) -> KeyEligibility {
        let Some(key) = self
            .keys
            .iter()
            .find(|key| &key.algorithm == algorithm && key.key_id == key_id)
        else {
            return KeyEligibility::Unknown;
        };
        if unix_s < key.not_before_unix_s {
            return KeyEligibility::NotYetValid;
        }
        if key
            .not_after_unix_s
            .is_some_and(|not_after| unix_s >= not_after)
        {
            return KeyEligibility::Expired;
        }
        match key.status {
            KeyLifecycleStatus::Revoked => return KeyEligibility::Revoked,
            KeyLifecycleStatus::Retired => return KeyEligibility::Retired,
            KeyLifecycleStatus::Active => {}
        }
        if !key.usages.contains(usage) {
            return KeyEligibility::UsageNotAllowed;
        }
        KeyEligibility::Eligible
    }

    pub fn digest(&self) -> Result<Sha256Digest, TrustSnapshotError> {
        self.validate()?;
        let mut canonical = self.clone();
        canonical.canonicalize();
        let mut digest = FramedDigest::new(TRUST_SNAPSHOT_DIGEST_DOMAIN);
        digest.text(TRUST_SNAPSHOT_SCHEMA);
        digest.text(&canonical.sequence.to_string());
        digest.text(&canonical.issued_at_unix_s.to_string());
        digest.text(&canonical.expires_at_unix_s.to_string());
        for key in canonical.keys {
            digest.text("key");
            digest_signature_algorithm(&mut digest, &key.algorithm);
            digest.text(&key.key_id);
            digest.text(&key.not_before_unix_s.to_string());
            match key.not_after_unix_s {
                Some(value) => {
                    digest.text("not-after");
                    digest.text(&value.to_string());
                }
                None => digest.text("no-not-after"),
            }
            digest.text(key_lifecycle_tag(key.status));
            for usage in key.usages {
                digest.text("usage");
                digest.text(usage.as_str());
            }
        }
        Ok(digest.digest())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TrustSnapshotTracker {
    latest_sequence: Option<u64>,
    latest_issued_at_unix_s: Option<u64>,
    latest_digest: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustSnapshotTrackingError {
    InvalidSnapshot(TrustSnapshotError),
    SequenceRollback { latest: u64, proposed: u64 },
    SequenceCollision { sequence: u64 },
    IssuedAtRegressed { latest: u64, proposed: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustSnapshotCurrentnessError {
    InvalidSnapshot(TrustSnapshotError),
    NoAcceptedSnapshot,
    SequenceMismatch { accepted: u64, presented: u64 },
    DigestMismatch { sequence: u64 },
}

impl TrustSnapshotTracker {
    /// Advance the accepted trust state monotonically. The caller is responsible
    /// for persisting this tracker (or equivalent trusted state) across process
    /// restarts; resetting trusted state is outside this in-memory primitive.
    pub fn accept(
        &mut self,
        snapshot: &TrustSnapshot,
    ) -> Result<Sha256Digest, TrustSnapshotTrackingError> {
        let digest = snapshot
            .digest()
            .map_err(TrustSnapshotTrackingError::InvalidSnapshot)?;
        if let Some(latest) = self.latest_sequence {
            if snapshot.sequence < latest {
                return Err(TrustSnapshotTrackingError::SequenceRollback {
                    latest,
                    proposed: snapshot.sequence,
                });
            }
            if snapshot.sequence == latest {
                if self.latest_digest.as_ref() == Some(&digest) {
                    return Ok(digest);
                }
                return Err(TrustSnapshotTrackingError::SequenceCollision {
                    sequence: snapshot.sequence,
                });
            }
        }
        if let Some(latest) = self.latest_issued_at_unix_s {
            if snapshot.issued_at_unix_s < latest {
                return Err(TrustSnapshotTrackingError::IssuedAtRegressed {
                    latest,
                    proposed: snapshot.issued_at_unix_s,
                });
            }
        }
        self.latest_sequence = Some(snapshot.sequence);
        self.latest_issued_at_unix_s = Some(snapshot.issued_at_unix_s);
        self.latest_digest = Some(digest.clone());
        Ok(digest)
    }

    /// Prove that `snapshot` is exactly the current accepted snapshot, not merely
    /// a valid/fresh older snapshot. Authority verification should use this gate
    /// so a still-time-valid pre-revocation snapshot cannot be replayed after a
    /// newer snapshot has been accepted.
    pub fn require_current(
        &self,
        snapshot: &TrustSnapshot,
    ) -> Result<Sha256Digest, TrustSnapshotCurrentnessError> {
        let digest = snapshot
            .digest()
            .map_err(TrustSnapshotCurrentnessError::InvalidSnapshot)?;
        let accepted_sequence = self
            .latest_sequence
            .ok_or(TrustSnapshotCurrentnessError::NoAcceptedSnapshot)?;
        if snapshot.sequence != accepted_sequence {
            return Err(TrustSnapshotCurrentnessError::SequenceMismatch {
                accepted: accepted_sequence,
                presented: snapshot.sequence,
            });
        }
        if self.latest_digest.as_ref() != Some(&digest) {
            return Err(TrustSnapshotCurrentnessError::DigestMismatch {
                sequence: snapshot.sequence,
            });
        }
        Ok(digest)
    }

    pub fn latest_sequence(&self) -> Option<u64> {
        self.latest_sequence
    }

    pub fn latest_digest(&self) -> Option<&Sha256Digest> {
        self.latest_digest.as_ref()
    }
}

const fn key_lifecycle_tag(value: KeyLifecycleStatus) -> &'static str {
    match value {
        KeyLifecycleStatus::Active => "active",
        KeyLifecycleStatus::Retired => "retired",
        KeyLifecycleStatus::Revoked => "revoked",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(value: &str) -> TrustUsage {
        TrustUsage::parse(value).unwrap()
    }

    fn active_key(key_id: &str) -> KeyTrustRecord {
        KeyTrustRecord {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: key_id.into(),
            not_before_unix_s: 100,
            not_after_unix_s: Some(900),
            status: KeyLifecycleStatus::Active,
            usages: BTreeSet::from([usage("science.qualification")]),
        }
    }

    #[test]
    fn canonical_digest_is_order_independent() {
        let left = TrustSnapshot::new(
            7,
            100,
            1_000,
            vec![active_key("b"), active_key("a")],
        )
        .unwrap();
        let right = TrustSnapshot::new(
            7,
            100,
            1_000,
            vec![active_key("a"), active_key("b")],
        )
        .unwrap();
        assert_eq!(left.digest().unwrap(), right.digest().unwrap());
    }

    #[test]
    fn revocation_overrides_time_and_usage() {
        let mut key = active_key("reviewer");
        key.status = KeyLifecycleStatus::Revoked;
        let snapshot = TrustSnapshot::new(1, 100, 1_000, vec![key]).unwrap();
        assert_eq!(
            snapshot.key_eligibility(
                &SignatureAlgorithm::Ed25519,
                "reviewer",
                &usage("science.qualification"),
                500,
            ),
            KeyEligibility::Revoked
        );
    }

    #[test]
    fn usage_is_fail_closed() {
        let snapshot = TrustSnapshot::new(1, 100, 1_000, vec![active_key("reviewer")]).unwrap();
        assert_eq!(
            snapshot.key_eligibility(
                &SignatureAlgorithm::Ed25519,
                "reviewer",
                &usage("fabrication.release"),
                500,
            ),
            KeyEligibility::UsageNotAllowed
        );
    }

    #[test]
    fn tracker_rejects_rollback_and_collision() {
        let first = TrustSnapshot::new(10, 100, 1_000, vec![active_key("a")]).unwrap();
        let rollback = TrustSnapshot::new(9, 200, 1_100, vec![active_key("a")]).unwrap();
        let collision = TrustSnapshot::new(10, 100, 1_000, vec![active_key("b")]).unwrap();
        let mut tracker = TrustSnapshotTracker::default();
        tracker.accept(&first).unwrap();
        assert!(matches!(
            tracker.accept(&rollback),
            Err(TrustSnapshotTrackingError::SequenceRollback { .. })
        ));
        assert!(matches!(
            tracker.accept(&collision),
            Err(TrustSnapshotTrackingError::SequenceCollision { sequence: 10 })
        ));
    }

    #[test]
    fn older_but_still_valid_snapshot_is_not_current_after_advance() {
        let first = TrustSnapshot::new(1, 100, 1_000, vec![active_key("a")]).unwrap();
        let mut revoked = active_key("a");
        revoked.status = KeyLifecycleStatus::Revoked;
        let second = TrustSnapshot::new(2, 200, 1_000, vec![revoked]).unwrap();
        let mut tracker = TrustSnapshotTracker::default();
        tracker.accept(&first).unwrap();
        tracker.accept(&second).unwrap();
        assert!(matches!(
            tracker.require_current(&first),
            Err(TrustSnapshotCurrentnessError::SequenceMismatch {
                accepted: 2,
                presented: 1,
            })
        ));
        assert_eq!(tracker.require_current(&second).unwrap(), second.digest().unwrap());
    }

    #[test]
    fn custom_algorithm_cannot_alias_builtin_snapshot_identity() {
        let mut custom_key = active_key("same");
        custom_key.algorithm = SignatureAlgorithm::Other("ed25519".into());
        let builtin = TrustSnapshot::new(1, 100, 1_000, vec![active_key("same")]).unwrap();
        let custom = TrustSnapshot::new(1, 100, 1_000, vec![custom_key]).unwrap();
        assert_ne!(builtin.digest().unwrap(), custom.digest().unwrap());
    }
}
