// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generic lifecycle-governed key trust.
//!
//! Protocol identifiers intentionally preserve the existing fabrication V1
//! wire identity while the implementation is extracted into a reusable crate.

use crate::digest::{Sha256Digest, domain_hash};
use crate::signature::SignatureAlgorithm;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const TRUST_SNAPSHOT_SCHEMA: &str = "symthaea.fabrication.trust-snapshot.v1";
const TRUST_SNAPSHOT_DIGEST_DOMAIN: &[u8] = b"symthaea.fabrication.trust-snapshot-digest.v1\0";
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
    TooManyKeys { actual: usize, maximum: usize },
    InvalidAlgorithm(String),
    EmptyKeyId,
    NonCanonicalKeyId(String),
    KeyIdTooLong { actual: usize, maximum: usize },
    EmptyUsages(String),
    InvalidKeyWindow(String),
    DuplicateKey { algorithm: SignatureAlgorithm, key_id: String },
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
                return Err(TrustSnapshotError::InvalidAlgorithm(format!("{:?}", key.algorithm)));
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
    Ok(domain_hash(TRUST_SNAPSHOT_DIGEST_DOMAIN, &bytes))
}
