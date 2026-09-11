// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Durable-artifact receipts for legacy-computing qualification sources.
//!
//! A content digest proves identity only if the exact bytes can still be obtained.
//! This module binds source snapshots to retained-artifact metadata without storing
//! copyrighted vendor documents, credentials, or the document bytes in this crate.

use crate::legacy_computing::{LegacyComputingErrorV1, LegacyComputingPackV1};
use crate::standards_registry::{SourceCaptureV1, SourceSnapshotIdV1};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const LEGACY_SOURCE_ARTIFACT_LEDGER_SCHEMA_V1: &str =
    "symthaea-it-legacy-source-artifact-ledger-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LegacyArtifactStorageClassV1 {
    /// Retained in a controlled benchmark/evidence store.
    PrivateEvidenceStore,
    /// Retained in an organization-controlled archival store.
    OrganizationArchive,
    /// Public immutable artifact store where redistribution is permitted.
    PublicImmutableArchive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LegacyArtifactAccessPolicyV1 {
    /// Only the evaluator/evidence service may dereference the artifact.
    EvaluatorOnly,
    /// Authorized local operators may dereference it; not solver-visible by default.
    AuthorizedLocal,
    /// Artifact is public and redistribution/use policy permits direct access.
    Public,
}

/// Receipt for one exact retained source artifact.
///
/// `artifact_locator` is an opaque locator, not necessarily a URL. It must never
/// contain credentials or bearer tokens. Runtime storage adapters own retrieval.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacySourceArtifactRefV1 {
    pub snapshot_id: SourceSnapshotIdV1,
    pub content_algorithm: String,
    pub content_digest: String,
    pub byte_length: u64,
    pub media_type: String,
    pub retrieved_at_unix_ms: u64,
    pub artifact_locator: String,
    pub storage_class: LegacyArtifactStorageClassV1,
    pub access_policy: LegacyArtifactAccessPolicyV1,
    /// Digest of the external store's own retention/ingest receipt when available.
    /// This is an integrity/audit binding, not proof that the store is trustworthy.
    pub retention_receipt_digest: Option<String>,
}

impl LegacySourceArtifactRefV1 {
    pub fn validate_against_pack(
        &self,
        pack: &LegacyComputingPackV1,
    ) -> Result<(), LegacySourceArtifactErrorV1> {
        require_nonempty(&self.snapshot_id.0, "snapshot id")?;
        require_nonempty(&self.content_algorithm, "content algorithm")?;
        require_nonempty(&self.content_digest, "content digest")?;
        require_nonempty(&self.media_type, "media type")?;
        require_nonempty(&self.artifact_locator, "artifact locator")?;
        if self.byte_length == 0 {
            return Err(LegacySourceArtifactErrorV1::InvalidField(
                "artifact byte length must be non-zero".into(),
            ));
        }
        if self.retrieved_at_unix_ms == 0 {
            return Err(LegacySourceArtifactErrorV1::InvalidField(
                "artifact retrieval timestamp must be non-zero".into(),
            ));
        }
        if looks_secret_bearing(&self.artifact_locator) {
            // Deliberately do not echo the rejected locator: doing so could put the
            // credential we detected into logs, traces, or user-visible errors.
            return Err(LegacySourceArtifactErrorV1::SecretBearingLocator);
        }
        if let Some(receipt) = &self.retention_receipt_digest {
            validate_hex_digest(receipt, "retention receipt digest")?;
        }

        let snapshot = pack
            .sources
            .snapshot(&self.snapshot_id)
            .ok_or_else(|| LegacySourceArtifactErrorV1::UnknownSnapshot(self.snapshot_id.clone()))?;
        let SourceCaptureV1::ContentDigest { algorithm, digest } = &snapshot.capture else {
            return Err(LegacySourceArtifactErrorV1::SnapshotNotContentDigestBound(
                self.snapshot_id.clone(),
            ));
        };
        if !algorithm.eq_ignore_ascii_case(self.content_algorithm.trim())
            || digest.to_ascii_lowercase() != self.content_digest.trim().to_ascii_lowercase()
        {
            return Err(LegacySourceArtifactErrorV1::ContentDigestMismatch(
                self.snapshot_id.clone(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacySourceArtifactLedgerV1 {
    pub schema_version: String,
    artifacts: BTreeMap<SourceSnapshotIdV1, LegacySourceArtifactRefV1>,
}

impl Default for LegacySourceArtifactLedgerV1 {
    fn default() -> Self {
        Self {
            schema_version: LEGACY_SOURCE_ARTIFACT_LEDGER_SCHEMA_V1.into(),
            artifacts: BTreeMap::new(),
        }
    }
}

impl LegacySourceArtifactLedgerV1 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn artifact(
        &self,
        snapshot_id: &SourceSnapshotIdV1,
    ) -> Option<&LegacySourceArtifactRefV1> {
        self.artifacts.get(snapshot_id)
    }

    pub fn artifacts(&self) -> impl Iterator<Item = &LegacySourceArtifactRefV1> {
        self.artifacts.values()
    }

    /// Exact replay is idempotent; a source snapshot can never be rebound to a
    /// different retained artifact through the same V1 ledger identity.
    pub fn register(
        &mut self,
        pack: &LegacyComputingPackV1,
        artifact: LegacySourceArtifactRefV1,
    ) -> Result<bool, LegacySourceArtifactErrorV1> {
        self.validate_schema()?;
        pack.validate()?;
        artifact.validate_against_pack(pack)?;
        if let Some(existing) = self.artifacts.get(&artifact.snapshot_id) {
            if existing == &artifact {
                return Ok(false);
            }
            return Err(LegacySourceArtifactErrorV1::ArtifactIdentityConflict(
                artifact.snapshot_id,
            ));
        }
        self.artifacts.insert(artifact.snapshot_id.clone(), artifact);
        Ok(true)
    }

    pub fn validate_against_pack(
        &self,
        pack: &LegacyComputingPackV1,
    ) -> Result<(), LegacySourceArtifactErrorV1> {
        self.validate_schema()?;
        pack.validate()?;
        for artifact in self.artifacts.values() {
            artifact.validate_against_pack(pack)?;
        }
        Ok(())
    }

    fn validate_schema(&self) -> Result<(), LegacySourceArtifactErrorV1> {
        if self.schema_version != LEGACY_SOURCE_ARTIFACT_LEDGER_SCHEMA_V1 {
            return Err(LegacySourceArtifactErrorV1::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacySourceArtifactReadinessV1 {
    pub required_snapshots: usize,
    pub retained_artifacts: usize,
    pub missing_artifacts: BTreeSet<SourceSnapshotIdV1>,
    pub missing_retention_receipts: BTreeSet<SourceSnapshotIdV1>,
    /// True only when every source is content-digest-bound and retained as an
    /// exact matching artifact. Retention receipts are separately reported; V1
    /// does not require them because some trusted stores expose no receipt API.
    pub qualification_artifact_ready: bool,
}

pub fn assess_legacy_source_artifact_readiness_v1(
    pack: &LegacyComputingPackV1,
    ledger: &LegacySourceArtifactLedgerV1,
) -> Result<LegacySourceArtifactReadinessV1, LegacySourceArtifactErrorV1> {
    pack.validate()?;
    ledger.validate_against_pack(pack)?;

    let mut missing_artifacts = BTreeSet::new();
    let mut missing_retention_receipts = BTreeSet::new();
    let mut required_snapshots = 0usize;
    let mut retained_artifacts = 0usize;
    let mut all_content_bound = true;

    for snapshot in pack.sources.snapshots() {
        required_snapshots += 1;
        if !matches!(&snapshot.capture, SourceCaptureV1::ContentDigest { .. }) {
            all_content_bound = false;
        }
        match ledger.artifact(&snapshot.id) {
            Some(artifact) => {
                artifact.validate_against_pack(pack)?;
                retained_artifacts += 1;
                if artifact.retention_receipt_digest.is_none() {
                    missing_retention_receipts.insert(snapshot.id.clone());
                }
            }
            None => {
                missing_artifacts.insert(snapshot.id.clone());
            }
        }
    }

    let qualification_artifact_ready = required_snapshots > 0
        && all_content_bound
        && missing_artifacts.is_empty()
        && retained_artifacts == required_snapshots;

    Ok(LegacySourceArtifactReadinessV1 {
        required_snapshots,
        retained_artifacts,
        missing_artifacts,
        missing_retention_receipts,
        qualification_artifact_ready,
    })
}

fn looks_secret_bearing(locator: &str) -> bool {
    let lower = locator.to_ascii_lowercase();
    [
        "token=",
        "access_token=",
        "sig=",
        "signature=",
        "x-amz-signature=",
        "password=",
        "passwd=",
        "secret=",
        "bearer ",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), LegacySourceArtifactErrorV1> {
    if value.trim().is_empty() {
        Err(LegacySourceArtifactErrorV1::InvalidField(field.into()))
    } else {
        Ok(())
    }
}

fn validate_hex_digest(
    value: &str,
    field: &'static str,
) -> Result<(), LegacySourceArtifactErrorV1> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(LegacySourceArtifactErrorV1::InvalidField(format!(
            "{field} must be exactly 64 hexadecimal characters"
        )));
    }
    Ok(())
}

#[derive(Debug)]
pub enum LegacySourceArtifactErrorV1 {
    LegacyPack(LegacyComputingErrorV1),
    UnsupportedSchema(String),
    InvalidField(String),
    UnknownSnapshot(SourceSnapshotIdV1),
    SnapshotNotContentDigestBound(SourceSnapshotIdV1),
    ContentDigestMismatch(SourceSnapshotIdV1),
    ArtifactIdentityConflict(SourceSnapshotIdV1),
    /// Locator was rejected because it appears credential-bearing. The locator
    /// itself is intentionally not retained in the error value.
    SecretBearingLocator,
}

impl fmt::Display for LegacySourceArtifactErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LegacyPack(err) => write!(f, "invalid legacy computing pack: {err}"),
            Self::UnsupportedSchema(schema) => {
                write!(f, "unsupported legacy artifact ledger schema {schema}")
            }
            Self::InvalidField(message) => write!(f, "invalid legacy artifact field: {message}"),
            Self::UnknownSnapshot(id) => write!(f, "unknown legacy source snapshot {}", id.0),
            Self::SnapshotNotContentDigestBound(id) => write!(
                f,
                "legacy source snapshot {} is not content-digest-bound",
                id.0
            ),
            Self::ContentDigestMismatch(id) => write!(
                f,
                "retained artifact digest does not match source snapshot {}",
                id.0
            ),
            Self::ArtifactIdentityConflict(id) => {
                write!(f, "retained artifact identity was rebound for {}", id.0)
            }
            Self::SecretBearingLocator => {
                write!(f, "legacy artifact locator appears to contain credentials")
            }
        }
    }
}

impl Error for LegacySourceArtifactErrorV1 {}

impl From<LegacyComputingErrorV1> for LegacySourceArtifactErrorV1 {
    fn from(value: LegacyComputingErrorV1) -> Self {
        Self::LegacyPack(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seed_legacy_computing_pack_v1;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    #[test]
    fn current_metadata_only_pack_cannot_register_fake_archived_artifacts() {
        let pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        let snapshot = pack.sources.snapshots().next().unwrap();
        let mut ledger = LegacySourceArtifactLedgerV1::new();
        let artifact = LegacySourceArtifactRefV1 {
            snapshot_id: snapshot.id.clone(),
            content_algorithm: "sha256".into(),
            content_digest: digest('a'),
            byte_length: 100,
            media_type: "text/html".into(),
            retrieved_at_unix_ms: 1_800_000_000_001,
            artifact_locator: "evidence://legacy/source-1".into(),
            storage_class: LegacyArtifactStorageClassV1::PrivateEvidenceStore,
            access_policy: LegacyArtifactAccessPolicyV1::EvaluatorOnly,
            retention_receipt_digest: None,
        };
        assert!(matches!(
            ledger.register(&pack, artifact),
            Err(LegacySourceArtifactErrorV1::SnapshotNotContentDigestBound(_))
        ));
    }

    #[test]
    fn current_seed_reports_every_source_as_missing_retained_artifact() {
        let pack = seed_legacy_computing_pack_v1(1_800_000_000_000).unwrap();
        let ledger = LegacySourceArtifactLedgerV1::new();
        let readiness = assess_legacy_source_artifact_readiness_v1(&pack, &ledger).unwrap();
        assert_eq!(readiness.required_snapshots, pack.sources.snapshots().count());
        assert_eq!(readiness.retained_artifacts, 0);
        assert_eq!(readiness.missing_artifacts.len(), readiness.required_snapshots);
        assert!(!readiness.qualification_artifact_ready);
    }

    #[test]
    fn secret_bearing_locator_is_detected_without_echoing_it() {
        let locator = "https://archive.invalid/file?token=secret";
        assert!(looks_secret_bearing(locator));
        assert_eq!(
            LegacySourceArtifactErrorV1::SecretBearingLocator.to_string(),
            "legacy artifact locator appears to contain credentials"
        );
        assert!(!LegacySourceArtifactErrorV1::SecretBearingLocator
            .to_string()
            .contains("secret"));
    }
}
