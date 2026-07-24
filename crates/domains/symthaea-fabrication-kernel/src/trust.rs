// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bounded key-lifecycle snapshots for fabrication authority.
//!
//! Cryptographic validity is not sufficient authority by itself. A signer must
//! also be known, active for the requested purpose, inside its validity window,
//! and evaluated against a fresh, sequence-numbered trust snapshot.

use crate::attestation::SignatureAlgorithm;
use crate::crypto_digest::{Sha256, Sha256Digest};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const TRUST_SNAPSHOT_SCHEMA: &str = "symthaea.fabrication.trust-snapshot.v1";
pub const MAX_TRUST_KEYS: usize = 4096;
pub const MAX_TRUST_KEY_ID_BYTES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum KeyUsage {
    FabricationManifest,
    MachineSession,
    MachineTelemetry,
    OperatorCommand,
    GatewayConsensus,
    IncidentEvidence,
    ReleaseCertification,
    TrustRotation,
    RecoveryAuthorization,
    AuditAnchor,
    ThresholdCeremony,
    GatewayMembership,
    TransparencyLog,
    ReleasePromotion,
    ArtifactProvenance,
    TransparencyWitness,
    GatewayDecommission,
    ReleaseRollback,
    RegionalQuorum,
    SignerCompromise,
    WitnessGossip,
    GatewayTombstone,
    RolloutRevocation,
    PostRollbackRequalification,
    ClockAuthority,
    PolicyMigration,
    UpgradeHandoff,
    EvidenceCompaction,
    UpgradeProbation,
    AutomaticRollback,
    HardwareReauthorization,
    EvidenceRetention,
    KeyContinuity,
    ClockContinuity,
}

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
    pub usages: BTreeSet<KeyUsage>,
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
    TooManyKeys {
        actual: usize,
        maximum: usize,
    },
    InvalidAlgorithm(String),
    EmptyKeyId,
    NonCanonicalKeyId(String),
    KeyIdTooLong {
        actual: usize,
        maximum: usize,
    },
    EmptyUsages(String),
    InvalidKeyWindow(String),
    DuplicateKey {
        algorithm: SignatureAlgorithm,
        key_id: String,
    },
    Encoding(String),
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
            (&left.algorithm, left.key_id.as_str()).cmp(&(&right.algorithm, right.key_id.as_str()))
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
                return Err(TrustSnapshotError::InvalidAlgorithm(format!(
                    "{:?}",
                    key.algorithm
                )));
            }
            let key_id = key.key_id.trim();
            if key_id.is_empty() {
                return Err(TrustSnapshotError::EmptyKeyId);
            }
            if key_id != key.key_id {
                return Err(TrustSnapshotError::NonCanonicalKeyId(key.key_id.clone()));
            }
            if key_id.len() > MAX_TRUST_KEY_ID_BYTES {
                return Err(TrustSnapshotError::KeyIdTooLong {
                    actual: key_id.len(),
                    maximum: MAX_TRUST_KEY_ID_BYTES,
                });
            }
            if key.usages.is_empty() {
                return Err(TrustSnapshotError::EmptyUsages(key.key_id.clone()));
            }
            if key
                .not_after_unix_s
                .is_some_and(|not_after| not_after <= key.not_before_unix_s)
            {
                return Err(TrustSnapshotError::InvalidKeyWindow(key.key_id.clone()));
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
        usage: KeyUsage,
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
        if !key.usages.contains(&usage) {
            return KeyEligibility::UsageNotAllowed;
        }
        KeyEligibility::Eligible
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

impl TrustSnapshotTracker {
    pub fn accept(
        &mut self,
        snapshot: &TrustSnapshot,
    ) -> Result<Sha256Digest, TrustSnapshotTrackingError> {
        snapshot
            .validate()
            .map_err(TrustSnapshotTrackingError::InvalidSnapshot)?;
        let digest =
            digest_trust_snapshot(snapshot).map_err(TrustSnapshotTrackingError::InvalidSnapshot)?;
        if let Some(latest) = self.latest_sequence {
            if snapshot.sequence < latest {
                return Err(TrustSnapshotTrackingError::SequenceRollback {
                    latest,
                    proposed: snapshot.sequence,
                });
            }
            if snapshot.sequence == latest {
                if self.latest_digest == Some(digest) {
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
        self.latest_digest = Some(digest);
        Ok(digest)
    }

    pub fn latest_sequence(&self) -> Option<u64> {
        self.latest_sequence
    }

    pub fn latest_digest(&self) -> Option<Sha256Digest> {
        self.latest_digest
    }
}

pub fn canonical_trust_snapshot_bytes(
    snapshot: &TrustSnapshot,
) -> Result<Vec<u8>, TrustSnapshotError> {
    let mut canonical = snapshot.clone();
    canonical.canonicalize();
    canonical.validate()?;
    serde_json::to_vec(&canonical).map_err(|error| TrustSnapshotError::Encoding(error.to_string()))
}

pub fn digest_trust_snapshot(snapshot: &TrustSnapshot) -> Result<Sha256Digest, TrustSnapshotError> {
    let bytes = canonical_trust_snapshot_bytes(snapshot)?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.trust-snapshot-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn active_key(key_id: &str) -> KeyTrustRecord {
        KeyTrustRecord {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: key_id.into(),
            not_before_unix_s: 100,
            not_after_unix_s: Some(900),
            status: KeyLifecycleStatus::Active,
            usages: BTreeSet::from([KeyUsage::FabricationManifest]),
        }
    }

    #[test]
    fn canonical_digest_is_independent_of_input_order() {
        let left =
            TrustSnapshot::new(7, 100, 1_000, vec![active_key("b"), active_key("a")]).unwrap();
        let right =
            TrustSnapshot::new(7, 100, 1_000, vec![active_key("a"), active_key("b")]).unwrap();
        assert_eq!(
            digest_trust_snapshot(&left).unwrap(),
            digest_trust_snapshot(&right).unwrap()
        );
    }

    #[test]
    fn revocation_overrides_valid_time_and_usage() {
        let mut key = active_key("release");
        key.status = KeyLifecycleStatus::Revoked;
        let snapshot = TrustSnapshot::new(1, 100, 1_000, vec![key]).unwrap();
        assert_eq!(
            snapshot.key_eligibility(
                &SignatureAlgorithm::Ed25519,
                "release",
                KeyUsage::FabricationManifest,
                500,
            ),
            KeyEligibility::Revoked
        );
    }

    #[test]
    fn validity_windows_are_fail_closed() {
        let snapshot = TrustSnapshot::new(1, 100, 1_000, vec![active_key("release")]).unwrap();
        assert_eq!(
            snapshot.key_eligibility(
                &SignatureAlgorithm::Ed25519,
                "release",
                KeyUsage::FabricationManifest,
                99,
            ),
            KeyEligibility::NotYetValid
        );
        assert_eq!(
            snapshot.key_eligibility(
                &SignatureAlgorithm::Ed25519,
                "release",
                KeyUsage::FabricationManifest,
                900,
            ),
            KeyEligibility::Expired
        );
    }

    #[test]
    fn duplicate_identities_are_rejected() {
        assert!(matches!(
            TrustSnapshot::new(1, 100, 1_000, vec![active_key("same"), active_key("same")]),
            Err(TrustSnapshotError::DuplicateKey { .. })
        ));
    }

    #[test]
    fn long_lived_keys_can_span_snapshot_rollovers() {
        let mut key = active_key("long-lived");
        key.not_before_unix_s = 10;
        key.not_after_unix_s = Some(10_000);
        let snapshot = TrustSnapshot::new(2, 1_000, 2_000, vec![key]).unwrap();
        assert_eq!(
            snapshot.key_eligibility(
                &SignatureAlgorithm::Ed25519,
                "long-lived",
                KeyUsage::FabricationManifest,
                1_500,
            ),
            KeyEligibility::Eligible
        );
    }

    #[test]
    fn tracker_rejects_rollback_and_sequence_collision() {
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
    fn whitespace_ambiguous_key_ids_are_rejected() {
        let mut key = active_key(" release ");
        key.key_id = " release ".into();
        assert!(matches!(
            TrustSnapshot::new(1, 100, 1_000, vec![key]),
            Err(TrustSnapshotError::NonCanonicalKeyId(_))
        ));
    }
}
