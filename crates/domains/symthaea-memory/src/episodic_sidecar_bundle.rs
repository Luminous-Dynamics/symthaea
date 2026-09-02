// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Paired persistence for episodic epistemic sidecars.
//!
//! The replay store itself remains owned by the existing persistence machinery. This
//! bundle binds all epistemic sidecars to an externally computed replay-snapshot SHA-256
//! so provenance cannot be restored against a different replay snapshot by accident.

use crate::{
    EpisodicComponentProvenanceIndex, EpisodicOccurrenceIndex, EpisodicProvenanceIndex,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

pub const EPISODIC_SIDECAR_BUNDLE_SCHEMA: &str = "symthaea.episodic-sidecar-bundle.v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicSidecarBundle {
    schema: String,
    replay_snapshot_sha256: String,
    provenance: EpisodicProvenanceIndex,
    component_provenance: EpisodicComponentProvenanceIndex,
    occurrences: EpisodicOccurrenceIndex,
}

impl EpisodicSidecarBundle {
    pub fn new(
        replay_snapshot_sha256: impl Into<String>,
        provenance: EpisodicProvenanceIndex,
        component_provenance: EpisodicComponentProvenanceIndex,
        occurrences: EpisodicOccurrenceIndex,
    ) -> Result<Self, EpisodicSidecarBundleError> {
        let replay_snapshot_sha256 = replay_snapshot_sha256.into();
        validate_sha256(&replay_snapshot_sha256)?;
        Ok(Self {
            schema: EPISODIC_SIDECAR_BUNDLE_SCHEMA.into(),
            replay_snapshot_sha256,
            provenance,
            component_provenance,
            occurrences,
        })
    }

    pub fn replay_snapshot_sha256(&self) -> &str {
        &self.replay_snapshot_sha256
    }

    pub fn provenance(&self) -> &EpisodicProvenanceIndex {
        &self.provenance
    }

    pub fn component_provenance(&self) -> &EpisodicComponentProvenanceIndex {
        &self.component_provenance
    }

    pub fn occurrences(&self) -> &EpisodicOccurrenceIndex {
        &self.occurrences
    }

    pub fn into_indices(
        self,
    ) -> (
        EpisodicProvenanceIndex,
        EpisodicComponentProvenanceIndex,
        EpisodicOccurrenceIndex,
    ) {
        (self.provenance, self.component_provenance, self.occurrences)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, EpisodicSidecarBundleError> {
        bincode::serialize(self).map_err(|error| EpisodicSidecarBundleError::Serialization(error.to_string()))
    }

    pub fn bundle_sha256(&self) -> Result<String, EpisodicSidecarBundleError> {
        Ok(bytes_sha256(&self.to_bytes()?))
    }

    /// Decode only after verifying the exact externally expected bundle digest.
    pub fn from_bytes(
        bytes: &[u8],
        expected_bundle_sha256: &str,
    ) -> Result<Self, EpisodicSidecarBundleError> {
        validate_sha256(expected_bundle_sha256)?;
        let actual = bytes_sha256(bytes);
        if actual != expected_bundle_sha256 {
            return Err(EpisodicSidecarBundleError::BundleDigestMismatch {
                expected: expected_bundle_sha256.to_string(),
                got: actual,
            });
        }
        let bundle: Self = bincode::deserialize(bytes)
            .map_err(|error| EpisodicSidecarBundleError::Serialization(error.to_string()))?;
        if bundle.schema != EPISODIC_SIDECAR_BUNDLE_SCHEMA {
            return Err(EpisodicSidecarBundleError::SchemaMismatch(bundle.schema));
        }
        validate_sha256(&bundle.replay_snapshot_sha256)?;
        Ok(bundle)
    }

    /// Verify this sidecar set belongs to the replay snapshot the caller actually loaded.
    pub fn verify_replay_snapshot(
        &self,
        actual_replay_snapshot_sha256: &str,
    ) -> Result<(), EpisodicSidecarBundleError> {
        validate_sha256(actual_replay_snapshot_sha256)?;
        if self.replay_snapshot_sha256 != actual_replay_snapshot_sha256 {
            return Err(EpisodicSidecarBundleError::ReplaySnapshotMismatch {
                expected: self.replay_snapshot_sha256.clone(),
                got: actual_replay_snapshot_sha256.to_string(),
            });
        }
        Ok(())
    }
}

pub fn bytes_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn validate_sha256(value: &str) -> Result<(), EpisodicSidecarBundleError> {
    if value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        Ok(())
    } else {
        Err(EpisodicSidecarBundleError::InvalidSha256(value.to_string()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpisodicSidecarBundleError {
    InvalidSha256(String),
    BundleDigestMismatch { expected: String, got: String },
    ReplaySnapshotMismatch { expected: String, got: String },
    SchemaMismatch(String),
    Serialization(String),
}

impl fmt::Display for EpisodicSidecarBundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSha256(value) => write!(f, "expected lowercase SHA-256, got {value}"),
            Self::BundleDigestMismatch { expected, got } => write!(
                f,
                "episodic sidecar bundle digest mismatch: expected {expected}, got {got}"
            ),
            Self::ReplaySnapshotMismatch { expected, got } => write!(
                f,
                "episodic sidecar replay snapshot mismatch: expected {expected}, got {got}"
            ),
            Self::SchemaMismatch(schema) => write!(f, "unsupported episodic sidecar schema {schema}"),
            Self::Serialization(error) => write!(f, "episodic sidecar serialization error: {error}"),
        }
    }
}

impl std::error::Error for EpisodicSidecarBundleError {}

#[cfg(test)]
mod tests {
    use super::*;

    const REPLAY: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn bundle() -> EpisodicSidecarBundle {
        EpisodicSidecarBundle::new(
            REPLAY,
            EpisodicProvenanceIndex::default(),
            EpisodicComponentProvenanceIndex::default(),
            EpisodicOccurrenceIndex::default(),
        )
        .unwrap()
    }

    #[test]
    fn binary_round_trip_is_digest_bound() {
        let bundle = bundle();
        let bytes = bundle.to_bytes().unwrap();
        let digest = bytes_sha256(&bytes);
        let restored = EpisodicSidecarBundle::from_bytes(&bytes, &digest).unwrap();
        assert_eq!(restored.replay_snapshot_sha256(), REPLAY);
        assert_eq!(restored.bundle_sha256().unwrap(), digest);
    }

    #[test]
    fn byte_tamper_rejects_before_deserialization() {
        let bundle = bundle();
        let mut bytes = bundle.to_bytes().unwrap();
        let digest = bytes_sha256(&bytes);
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        assert!(matches!(
            EpisodicSidecarBundle::from_bytes(&bytes, &digest),
            Err(EpisodicSidecarBundleError::BundleDigestMismatch { .. })
        ));
    }

    #[test]
    fn replay_snapshot_substitution_rejects() {
        let bundle = bundle();
        let other = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        assert!(matches!(
            bundle.verify_replay_snapshot(other),
            Err(EpisodicSidecarBundleError::ReplaySnapshotMismatch { .. })
        ));
    }

    #[test]
    fn invalid_snapshot_digest_fails_closed() {
        assert!(EpisodicSidecarBundle::new(
            "not-a-hash",
            EpisodicProvenanceIndex::default(),
            EpisodicComponentProvenanceIndex::default(),
            EpisodicOccurrenceIndex::default(),
        )
        .is_err());
    }
}
