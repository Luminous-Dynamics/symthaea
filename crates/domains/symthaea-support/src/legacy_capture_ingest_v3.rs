// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Revision-bound offline capture ingress for legacy qualification sources.
//!
//! V3 binds the external capture result to one exact immutable source revision
//! from `LegacySourceCapturePlanV3`. Wire receipts are inert until an injected
//! verifier accepts their external attestation; the verified runtime wrapper is
//! intentionally non-Serde and privately constructible.
//!
//! ```text
//! logical document != source revision
//! serialized receipt != verified receipt
//! verified capture != qualification selection
//! qualification selection != verified claim/procedure
//! ```

use crate::legacy_computing::{LegacyComputingErrorV1, LegacyComputingPackV1};
use crate::legacy_qualification_source_ledger_v3::LegacyQualificationSourceSelectionV3;
use crate::legacy_source_artifacts::{
    LegacyArtifactAccessPolicyV1, LegacyArtifactStorageClassV1,
    LegacySourceArtifactErrorV1, LegacySourceArtifactLedgerV1, LegacySourceArtifactRefV1,
};
use crate::legacy_source_capture_plan::{LegacyCaptureRetentionV2};
use crate::legacy_source_capture_plan_v3::{
    LegacySourceCapturePlanV3, LegacySourceRevisionCaptureRequestV3,
    LEGACY_SOURCE_CAPTURE_PLAN_SCHEMA_V3,
};
use crate::standards_registry::{
    SourceCaptureV1, SourceSnapshotIdV1, StandardsRegistryErrorV1, TechnicalSourceSnapshotV1,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

pub const LEGACY_EXTERNAL_REVISION_CAPTURE_RECEIPT_SCHEMA_V3: &str =
    "symthaea-it-legacy-external-revision-capture-receipt-v3";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyExternalRevisionCaptureReceiptV3 {
    pub schema_version: String,
    /// Exact historical revision being frozen.
    pub original_snapshot_id: SourceSnapshotIdV1,
    /// Exact typed source-revision commitment from the V3 plan.
    pub source_revision_blake3: String,
    /// BLAKE3 over the complete V3 capture request.
    pub capture_request_blake3: String,
    pub capture_service_id: String,
    pub content_algorithm: String,
    pub content_digest: String,
    pub byte_length: u64,
    pub media_type: String,
    pub fetched_at_unix_ms: u64,
    pub issued_at_unix_ms: u64,
    pub artifact_locator: String,
    pub retention_receipt_sha256: String,
    pub external_attestation_sha256: String,
    /// BLAKE3 over every semantic receipt field above.
    pub receipt_commitment_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegacyRevisionCaptureTrustEvidenceV3 {
    pub verifier_id: String,
    pub trust_policy_sha256: String,
    pub signature_suite: String,
    pub verified_attestation_sha256: String,
}

pub trait LegacyRevisionCaptureReceiptVerifierV3 {
    fn verify(
        &self,
        receipt: &LegacyExternalRevisionCaptureReceiptV3,
        receipt_commitment_blake3: &str,
    ) -> Result<LegacyRevisionCaptureTrustEvidenceV3, String>;
}

#[derive(Debug, Clone)]
pub struct VerifiedLegacyRevisionCaptureReceiptV3 {
    receipt: LegacyExternalRevisionCaptureReceiptV3,
    request: LegacySourceRevisionCaptureRequestV3,
    trust: LegacyRevisionCaptureTrustEvidenceV3,
}

impl VerifiedLegacyRevisionCaptureReceiptV3 {
    pub fn receipt(&self) -> &LegacyExternalRevisionCaptureReceiptV3 {
        &self.receipt
    }

    pub fn request(&self) -> &LegacySourceRevisionCaptureRequestV3 {
        &self.request
    }

    pub fn trust(&self) -> &LegacyRevisionCaptureTrustEvidenceV3 {
        &self.trust
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyRevisionCaptureIngestResultV3 {
    pub original_snapshot_id: SourceSnapshotIdV1,
    pub qualifying_snapshot_id: SourceSnapshotIdV1,
    pub snapshot_inserted: bool,
    pub artifact_inserted: bool,
    pub verifier_id: String,
    pub trust_policy_sha256: String,
    /// Candidate only. Selection remains a separate explicit ledger transition.
    pub selection_candidate: LegacyQualificationSourceSelectionV3,
}

pub fn legacy_revision_capture_request_commitment_v3(
    request: &LegacySourceRevisionCaptureRequestV3,
) -> Result<String, LegacyRevisionCaptureIngestErrorV3> {
    let encoded = serde_json::to_vec(request)
        .map_err(|err| LegacyRevisionCaptureIngestErrorV3::Serialization(err.to_string()))?;
    let mut hasher = blake3::Hasher::new();
    frame(
        &mut hasher,
        b"schema",
        LEGACY_SOURCE_CAPTURE_PLAN_SCHEMA_V3.as_bytes(),
    );
    frame(&mut hasher, b"capture_request", &encoded);
    Ok(hasher.finalize().to_hex().to_string())
}

pub fn legacy_external_revision_capture_receipt_commitment_v3(
    receipt: &LegacyExternalRevisionCaptureReceiptV3,
) -> Result<String, LegacyRevisionCaptureIngestErrorV3> {
    #[derive(Serialize)]
    struct Commitment<'a> {
        schema_version: &'a str,
        original_snapshot_id: &'a SourceSnapshotIdV1,
        source_revision_blake3: &'a str,
        capture_request_blake3: &'a str,
        capture_service_id: &'a str,
        content_algorithm: &'a str,
        content_digest: &'a str,
        byte_length: u64,
        media_type: &'a str,
        fetched_at_unix_ms: u64,
        issued_at_unix_ms: u64,
        artifact_locator: &'a str,
        retention_receipt_sha256: &'a str,
        external_attestation_sha256: &'a str,
    }

    let encoded = serde_json::to_vec(&Commitment {
        schema_version: &receipt.schema_version,
        original_snapshot_id: &receipt.original_snapshot_id,
        source_revision_blake3: &receipt.source_revision_blake3,
        capture_request_blake3: &receipt.capture_request_blake3,
        capture_service_id: &receipt.capture_service_id,
        content_algorithm: &receipt.content_algorithm,
        content_digest: &receipt.content_digest,
        byte_length: receipt.byte_length,
        media_type: &receipt.media_type,
        fetched_at_unix_ms: receipt.fetched_at_unix_ms,
        issued_at_unix_ms: receipt.issued_at_unix_ms,
        artifact_locator: &receipt.artifact_locator,
        retention_receipt_sha256: &receipt.retention_receipt_sha256,
        external_attestation_sha256: &receipt.external_attestation_sha256,
    })
    .map_err(|err| LegacyRevisionCaptureIngestErrorV3::Serialization(err.to_string()))?;

    let mut hasher = blake3::Hasher::new();
    frame(
        &mut hasher,
        b"schema",
        LEGACY_EXTERNAL_REVISION_CAPTURE_RECEIPT_SCHEMA_V3.as_bytes(),
    );
    frame(&mut hasher, b"receipt", &encoded);
    Ok(hasher.finalize().to_hex().to_string())
}

pub fn verify_legacy_revision_capture_receipt_v3<
    V: LegacyRevisionCaptureReceiptVerifierV3,
>(
    plan: &LegacySourceCapturePlanV3,
    receipt: LegacyExternalRevisionCaptureReceiptV3,
    verifier: &V,
    now_unix_ms: u64,
    max_receipt_age_ms: u64,
    max_future_skew_ms: u64,
) -> Result<VerifiedLegacyRevisionCaptureReceiptV3, LegacyRevisionCaptureIngestErrorV3> {
    validate_receipt_shape(&receipt)?;
    if plan.schema_version != LEGACY_SOURCE_CAPTURE_PLAN_SCHEMA_V3 {
        return Err(
            LegacyRevisionCaptureIngestErrorV3::UnsupportedCapturePlanSchema(
                plan.schema_version.clone(),
            ),
        );
    }
    if now_unix_ms == 0 || max_receipt_age_ms == 0 {
        return Err(LegacyRevisionCaptureIngestErrorV3::InvalidField(
            "verification clock and maximum receipt age must be non-zero".into(),
        ));
    }

    let request = plan
        .requests
        .iter()
        .find(|request| request.original_snapshot_id == receipt.original_snapshot_id)
        .ok_or_else(|| {
            LegacyRevisionCaptureIngestErrorV3::UnknownCaptureRequest(
                receipt.original_snapshot_id.clone(),
            )
        })?
        .clone();
    if !request.capture_needed {
        return Err(LegacyRevisionCaptureIngestErrorV3::CaptureNotNeeded(
            receipt.original_snapshot_id.clone(),
        ));
    }
    if !request.capture_ready {
        return Err(LegacyRevisionCaptureIngestErrorV3::CaptureRequestNotReady(
            receipt.original_snapshot_id.clone(),
        ));
    }
    if request.source_revision_blake3 != receipt.source_revision_blake3.to_ascii_lowercase() {
        return Err(LegacyRevisionCaptureIngestErrorV3::SourceRevisionMismatch);
    }
    let expected_request = legacy_revision_capture_request_commitment_v3(&request)?;
    if expected_request != receipt.capture_request_blake3.to_ascii_lowercase() {
        return Err(LegacyRevisionCaptureIngestErrorV3::CaptureRequestCommitmentMismatch);
    }
    if !request
        .content_digest_algorithm
        .eq_ignore_ascii_case(receipt.content_algorithm.trim())
    {
        return Err(LegacyRevisionCaptureIngestErrorV3::UnexpectedDigestAlgorithm(
            receipt.content_algorithm.clone(),
        ));
    }

    let future_limit = now_unix_ms
        .checked_add(max_future_skew_ms)
        .ok_or_else(|| LegacyRevisionCaptureIngestErrorV3::InvalidField("clock overflow".into()))?;
    if receipt.fetched_at_unix_ms > future_limit || receipt.issued_at_unix_ms > future_limit {
        return Err(LegacyRevisionCaptureIngestErrorV3::ReceiptFromFuture);
    }
    if receipt.fetched_at_unix_ms > receipt.issued_at_unix_ms {
        return Err(LegacyRevisionCaptureIngestErrorV3::CaptureAfterReceiptIssued);
    }
    if now_unix_ms >= receipt.issued_at_unix_ms
        && now_unix_ms - receipt.issued_at_unix_ms > max_receipt_age_ms
    {
        return Err(LegacyRevisionCaptureIngestErrorV3::ReceiptTooOld);
    }

    let expected_receipt = legacy_external_revision_capture_receipt_commitment_v3(&receipt)?;
    if expected_receipt != receipt.receipt_commitment_blake3.to_ascii_lowercase() {
        return Err(LegacyRevisionCaptureIngestErrorV3::ReceiptCommitmentMismatch);
    }
    let trust = verifier
        .verify(&receipt, &expected_receipt)
        .map_err(LegacyRevisionCaptureIngestErrorV3::ExternalVerificationRejected)?;
    validate_trust_evidence(&receipt, &trust)?;

    Ok(VerifiedLegacyRevisionCaptureReceiptV3 {
        receipt,
        request,
        trust,
    })
}

pub fn ingest_verified_legacy_revision_capture_v3(
    pack: &mut LegacyComputingPackV1,
    artifacts: &mut LegacySourceArtifactLedgerV1,
    verified: &VerifiedLegacyRevisionCaptureReceiptV3,
) -> Result<LegacyRevisionCaptureIngestResultV3, LegacyRevisionCaptureIngestErrorV3> {
    pack.validate()?;
    artifacts.validate_against_pack(pack)?;

    let original = pack
        .sources
        .snapshot(&verified.receipt.original_snapshot_id)
        .ok_or_else(|| {
            LegacyRevisionCaptureIngestErrorV3::UnknownOriginalSnapshot(
                verified.receipt.original_snapshot_id.clone(),
            )
        })?
        .clone();
    if verified.request.source_revision_blake3 != verified.receipt.source_revision_blake3 {
        return Err(LegacyRevisionCaptureIngestErrorV3::SourceRevisionMismatch);
    }

    let qualifying_snapshot_id = derived_content_snapshot_id(&verified.receipt);
    let qualifying = TechnicalSourceSnapshotV1 {
        id: qualifying_snapshot_id.clone(),
        document_id: original.document_id.clone(),
        version: original.version.clone(),
        lifecycle: original.lifecycle,
        authority: original.authority,
        stability: original.stability,
        published_at_unix_ms: original.published_at_unix_ms,
        source_updated_at_unix_ms: original.source_updated_at_unix_ms,
        fetched_at_unix_ms: verified.receipt.fetched_at_unix_ms,
        capture: SourceCaptureV1::ContentDigest {
            algorithm: verified.receipt.content_algorithm.trim().to_ascii_lowercase(),
            digest: verified.receipt.content_digest.trim().to_ascii_lowercase(),
        },
        relations: original.relations.clone(),
    };

    let (storage_class, access_policy) = artifact_policy(verified.request.retention);
    let artifact = LegacySourceArtifactRefV1 {
        snapshot_id: qualifying_snapshot_id.clone(),
        content_algorithm: verified.receipt.content_algorithm.trim().to_ascii_lowercase(),
        content_digest: verified.receipt.content_digest.trim().to_ascii_lowercase(),
        byte_length: verified.receipt.byte_length,
        media_type: verified.receipt.media_type.trim().to_string(),
        retrieved_at_unix_ms: verified.receipt.fetched_at_unix_ms,
        artifact_locator: verified.receipt.artifact_locator.clone(),
        storage_class,
        access_policy,
        retention_receipt_digest: Some(
            verified
                .receipt
                .retention_receipt_sha256
                .trim()
                .to_ascii_lowercase(),
        ),
    };

    let mut next_pack = pack.clone();
    let mut next_artifacts = artifacts.clone();
    let snapshot_inserted = next_pack.sources.register_snapshot(qualifying)?;
    next_pack.validate()?;
    let artifact_inserted = next_artifacts.register(&next_pack, artifact)?;
    next_artifacts.validate_against_pack(&next_pack)?;
    *pack = next_pack;
    *artifacts = next_artifacts;

    let selection_candidate = LegacyQualificationSourceSelectionV3 {
        original_snapshot_id: verified.receipt.original_snapshot_id.clone(),
        source_revision_blake3: verified.receipt.source_revision_blake3.clone(),
        qualifying_snapshot_id: qualifying_snapshot_id.clone(),
        content_algorithm: verified.receipt.content_algorithm.trim().to_ascii_lowercase(),
        content_digest: verified.receipt.content_digest.trim().to_ascii_lowercase(),
        selected_at_unix_ms: verified.receipt.issued_at_unix_ms,
    };

    Ok(LegacyRevisionCaptureIngestResultV3 {
        original_snapshot_id: verified.receipt.original_snapshot_id.clone(),
        qualifying_snapshot_id,
        snapshot_inserted,
        artifact_inserted,
        verifier_id: verified.trust.verifier_id.clone(),
        trust_policy_sha256: verified.trust.trust_policy_sha256.clone(),
        selection_candidate,
    })
}

fn validate_receipt_shape(
    receipt: &LegacyExternalRevisionCaptureReceiptV3,
) -> Result<(), LegacyRevisionCaptureIngestErrorV3> {
    if receipt.schema_version != LEGACY_EXTERNAL_REVISION_CAPTURE_RECEIPT_SCHEMA_V3 {
        return Err(LegacyRevisionCaptureIngestErrorV3::UnsupportedReceiptSchema(
            receipt.schema_version.clone(),
        ));
    }
    require_nonempty(&receipt.original_snapshot_id.0, "original snapshot id")?;
    require_blake3(&receipt.source_revision_blake3, "source revision digest")?;
    require_blake3(&receipt.capture_request_blake3, "capture request digest")?;
    require_nonempty(&receipt.capture_service_id, "capture service id")?;
    if !receipt.content_algorithm.trim().eq_ignore_ascii_case("sha256") {
        return Err(LegacyRevisionCaptureIngestErrorV3::UnexpectedDigestAlgorithm(
            receipt.content_algorithm.clone(),
        ));
    }
    require_sha256(&receipt.content_digest, "content digest")?;
    if receipt.byte_length == 0 {
        return Err(LegacyRevisionCaptureIngestErrorV3::InvalidField(
            "captured byte length must be non-zero".into(),
        ));
    }
    require_nonempty(&receipt.media_type, "media type")?;
    if receipt.fetched_at_unix_ms == 0 || receipt.issued_at_unix_ms == 0 {
        return Err(LegacyRevisionCaptureIngestErrorV3::InvalidField(
            "capture timestamps must be non-zero".into(),
        ));
    }
    require_nonempty(&receipt.artifact_locator, "artifact locator")?;
    if looks_secret_bearing(&receipt.artifact_locator) {
        return Err(LegacyRevisionCaptureIngestErrorV3::SecretBearingArtifactLocator);
    }
    require_sha256(&receipt.retention_receipt_sha256, "retention receipt digest")?;
    require_sha256(&receipt.external_attestation_sha256, "external attestation digest")?;
    require_blake3(&receipt.receipt_commitment_blake3, "receipt commitment")?;
    Ok(())
}

fn validate_trust_evidence(
    receipt: &LegacyExternalRevisionCaptureReceiptV3,
    trust: &LegacyRevisionCaptureTrustEvidenceV3,
) -> Result<(), LegacyRevisionCaptureIngestErrorV3> {
    require_nonempty(&trust.verifier_id, "verifier id")?;
    require_sha256(&trust.trust_policy_sha256, "trust policy digest")?;
    require_nonempty(&trust.signature_suite, "signature suite")?;
    require_sha256(&trust.verified_attestation_sha256, "verified attestation digest")?;
    if trust.verified_attestation_sha256.to_ascii_lowercase()
        != receipt.external_attestation_sha256.to_ascii_lowercase()
    {
        return Err(LegacyRevisionCaptureIngestErrorV3::AttestationDigestMismatch);
    }
    Ok(())
}

fn artifact_policy(
    retention: LegacyCaptureRetentionV2,
) -> (LegacyArtifactStorageClassV1, LegacyArtifactAccessPolicyV1) {
    match retention {
        LegacyCaptureRetentionV2::PrivateEvidenceStore => (
            LegacyArtifactStorageClassV1::PrivateEvidenceStore,
            LegacyArtifactAccessPolicyV1::EvaluatorOnly,
        ),
        LegacyCaptureRetentionV2::OrganizationArchive => (
            LegacyArtifactStorageClassV1::OrganizationArchive,
            LegacyArtifactAccessPolicyV1::AuthorizedLocal,
        ),
        LegacyCaptureRetentionV2::PublicImmutableArchive => (
            LegacyArtifactStorageClassV1::PublicImmutableArchive,
            LegacyArtifactAccessPolicyV1::Public,
        ),
    }
}

fn derived_content_snapshot_id(
    receipt: &LegacyExternalRevisionCaptureReceiptV3,
) -> SourceSnapshotIdV1 {
    SourceSnapshotIdV1(format!(
        "{}:content:{}:{}:{}",
        receipt.original_snapshot_id.0,
        receipt.content_algorithm.trim().to_ascii_lowercase(),
        receipt.content_digest.trim().to_ascii_lowercase(),
        receipt.fetched_at_unix_ms
    ))
}

fn frame(hasher: &mut blake3::Hasher, label: &[u8], value: &[u8]) {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn require_nonempty(
    value: &str,
    field: &'static str,
) -> Result<(), LegacyRevisionCaptureIngestErrorV3> {
    if value.trim().is_empty() {
        return Err(LegacyRevisionCaptureIngestErrorV3::InvalidField(format!(
            "{field} must be non-empty"
        )));
    }
    Ok(())
}

fn require_sha256(
    value: &str,
    field: &'static str,
) -> Result<(), LegacyRevisionCaptureIngestErrorV3> {
    let value = value.trim();
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(LegacyRevisionCaptureIngestErrorV3::InvalidField(format!(
            "{field} must be a 64-character hex SHA-256"
        )));
    }
    Ok(())
}

fn require_blake3(
    value: &str,
    field: &'static str,
) -> Result<(), LegacyRevisionCaptureIngestErrorV3> {
    let value = value.trim();
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(LegacyRevisionCaptureIngestErrorV3::InvalidField(format!(
            "{field} must be a 64-character hex BLAKE3 digest"
        )));
    }
    Ok(())
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

#[derive(Debug)]
pub enum LegacyRevisionCaptureIngestErrorV3 {
    LegacyPack(LegacyComputingErrorV1),
    Standards(StandardsRegistryErrorV1),
    Artifact(LegacySourceArtifactErrorV1),
    UnsupportedCapturePlanSchema(String),
    UnsupportedReceiptSchema(String),
    InvalidField(String),
    Serialization(String),
    UnknownCaptureRequest(SourceSnapshotIdV1),
    CaptureNotNeeded(SourceSnapshotIdV1),
    CaptureRequestNotReady(SourceSnapshotIdV1),
    SourceRevisionMismatch,
    CaptureRequestCommitmentMismatch,
    UnexpectedDigestAlgorithm(String),
    ReceiptFromFuture,
    CaptureAfterReceiptIssued,
    ReceiptTooOld,
    ReceiptCommitmentMismatch,
    ExternalVerificationRejected(String),
    AttestationDigestMismatch,
    SecretBearingArtifactLocator,
    UnknownOriginalSnapshot(SourceSnapshotIdV1),
}

impl fmt::Display for LegacyRevisionCaptureIngestErrorV3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LegacyPack(err) => write!(f, "legacy revision ingest pack error: {err}"),
            Self::Standards(err) => write!(f, "legacy revision ingest registry error: {err}"),
            Self::Artifact(err) => write!(f, "legacy revision ingest artifact error: {err}"),
            Self::UnsupportedCapturePlanSchema(schema) => {
                write!(f, "unsupported legacy capture-plan schema {schema}")
            }
            Self::UnsupportedReceiptSchema(schema) => {
                write!(f, "unsupported legacy revision receipt schema {schema}")
            }
            Self::InvalidField(message) => write!(f, "invalid legacy revision receipt: {message}"),
            Self::Serialization(message) => write!(f, "legacy revision serialization failed: {message}"),
            Self::UnknownCaptureRequest(id) => write!(f, "no capture request exists for source revision {}", id.0),
            Self::CaptureNotNeeded(id) => write!(f, "source revision {} is already content-bound", id.0),
            Self::CaptureRequestNotReady(id) => write!(f, "capture request for source revision {} is not ready", id.0),
            Self::SourceRevisionMismatch => write!(f, "receipt does not bind the exact source revision"),
            Self::CaptureRequestCommitmentMismatch => write!(f, "receipt does not bind the exact capture request"),
            Self::UnexpectedDigestAlgorithm(value) => write!(f, "unexpected digest algorithm {value}"),
            Self::ReceiptFromFuture => write!(f, "receipt timestamp exceeds allowed future skew"),
            Self::CaptureAfterReceiptIssued => write!(f, "capture occurred after receipt issuance"),
            Self::ReceiptTooOld => write!(f, "capture receipt exceeds maximum accepted age"),
            Self::ReceiptCommitmentMismatch => write!(f, "capture receipt commitment mismatch"),
            Self::ExternalVerificationRejected(message) => write!(f, "external capture attestation rejected: {message}"),
            Self::AttestationDigestMismatch => write!(f, "verifier accepted a different attestation digest"),
            Self::SecretBearingArtifactLocator => write!(f, "artifact locator appears to contain secret-bearing material"),
            Self::UnknownOriginalSnapshot(id) => write!(f, "unknown original source revision {}", id.0),
        }
    }
}

impl Error for LegacyRevisionCaptureIngestErrorV3 {}

impl From<LegacyComputingErrorV1> for LegacyRevisionCaptureIngestErrorV3 {
    fn from(value: LegacyComputingErrorV1) -> Self {
        Self::LegacyPack(value)
    }
}

impl From<StandardsRegistryErrorV1> for LegacyRevisionCaptureIngestErrorV3 {
    fn from(value: StandardsRegistryErrorV1) -> Self {
        Self::Standards(value)
    }
}

impl From<LegacySourceArtifactErrorV1> for LegacyRevisionCaptureIngestErrorV3 {
    fn from(value: LegacySourceArtifactErrorV1) -> Self {
        Self::Artifact(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        build_legacy_five_platform_portfolio_v1,
        plan_legacy_qualification_source_captures_v3,
    };

    const CONTENT: &str =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const RETENTION: &str =
        "1111111111111111111111111111111111111111111111111111111111111111";
    const ATTESTATION: &str =
        "2222222222222222222222222222222222222222222222222222222222222222";
    const POLICY: &str =
        "3333333333333333333333333333333333333333333333333333333333333333";

    struct AcceptVerifier;
    impl LegacyRevisionCaptureReceiptVerifierV3 for AcceptVerifier {
        fn verify(
            &self,
            receipt: &LegacyExternalRevisionCaptureReceiptV3,
            _commitment: &str,
        ) -> Result<LegacyRevisionCaptureTrustEvidenceV3, String> {
            Ok(LegacyRevisionCaptureTrustEvidenceV3 {
                verifier_id: "test-v3-verifier".into(),
                trust_policy_sha256: POLICY.into(),
                signature_suite: "test-only".into(),
                verified_attestation_sha256: receipt.external_attestation_sha256.clone(),
            })
        }
    }

    fn pack_and_plan() -> (LegacyComputingPackV1, LegacySourceCapturePlanV3) {
        let pack = build_legacy_five_platform_portfolio_v1(1_800_000_000_000)
            .unwrap()
            .0;
        let plan = plan_legacy_qualification_source_captures_v3(&pack).unwrap();
        (pack, plan)
    }

    fn receipt(plan: &LegacySourceCapturePlanV3) -> LegacyExternalRevisionCaptureReceiptV3 {
        let request = plan
            .requests
            .iter()
            .find(|request| request.capture_needed && request.capture_ready)
            .unwrap();
        let mut value = LegacyExternalRevisionCaptureReceiptV3 {
            schema_version: LEGACY_EXTERNAL_REVISION_CAPTURE_RECEIPT_SCHEMA_V3.into(),
            original_snapshot_id: request.original_snapshot_id.clone(),
            source_revision_blake3: request.source_revision_blake3.clone(),
            capture_request_blake3: legacy_revision_capture_request_commitment_v3(request).unwrap(),
            capture_service_id: "capture-service:test-v3".into(),
            content_algorithm: "sha256".into(),
            content_digest: CONTENT.into(),
            byte_length: 42_000,
            media_type: "application/pdf".into(),
            fetched_at_unix_ms: 1_800_000_001_000,
            issued_at_unix_ms: 1_800_000_002_000,
            artifact_locator: "evidence://legacy-v3/source.pdf".into(),
            retention_receipt_sha256: RETENTION.into(),
            external_attestation_sha256: ATTESTATION.into(),
            receipt_commitment_blake3: String::new(),
        };
        value.receipt_commitment_blake3 =
            legacy_external_revision_capture_receipt_commitment_v3(&value).unwrap();
        value
    }

    #[test]
    fn verified_capture_returns_selection_for_exact_original_revision() {
        let (mut pack, plan) = pack_and_plan();
        let value = receipt(&plan);
        let original_id = value.original_snapshot_id.clone();
        let revision = value.source_revision_blake3.clone();
        let verified = verify_legacy_revision_capture_receipt_v3(
            &plan,
            value,
            &AcceptVerifier,
            1_800_000_003_000,
            60_000,
            1_000,
        )
        .unwrap();
        let mut artifacts = LegacySourceArtifactLedgerV1::new();
        let result = ingest_verified_legacy_revision_capture_v3(
            &mut pack,
            &mut artifacts,
            &verified,
        )
        .unwrap();
        assert_eq!(result.original_snapshot_id, original_id);
        assert_eq!(result.selection_candidate.original_snapshot_id, original_id);
        assert_eq!(result.selection_candidate.source_revision_blake3, revision);
        assert!(result.snapshot_inserted && result.artifact_inserted);
    }

    #[test]
    fn source_revision_substitution_is_rejected() {
        let (_pack, plan) = pack_and_plan();
        let mut value = receipt(&plan);
        value.source_revision_blake3 = "a".repeat(64);
        value.receipt_commitment_blake3 =
            legacy_external_revision_capture_receipt_commitment_v3(&value).unwrap();
        let error = verify_legacy_revision_capture_receipt_v3(
            &plan,
            value,
            &AcceptVerifier,
            1_800_000_003_000,
            60_000,
            1_000,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            LegacyRevisionCaptureIngestErrorV3::SourceRevisionMismatch
        ));
    }

    #[test]
    fn exact_verified_replay_is_idempotent() {
        let (mut pack, plan) = pack_and_plan();
        let verified = verify_legacy_revision_capture_receipt_v3(
            &plan,
            receipt(&plan),
            &AcceptVerifier,
            1_800_000_003_000,
            60_000,
            1_000,
        )
        .unwrap();
        let mut artifacts = LegacySourceArtifactLedgerV1::new();
        let first = ingest_verified_legacy_revision_capture_v3(
            &mut pack,
            &mut artifacts,
            &verified,
        )
        .unwrap();
        let second = ingest_verified_legacy_revision_capture_v3(
            &mut pack,
            &mut artifacts,
            &verified,
        )
        .unwrap();
        assert!(first.snapshot_inserted && first.artifact_inserted);
        assert!(!second.snapshot_inserted && !second.artifact_inserted);
        assert_eq!(first.qualifying_snapshot_id, second.qualifying_snapshot_id);
    }

    #[test]
    fn receipt_tamper_is_detected_before_external_trust_promotion() {
        let (_pack, plan) = pack_and_plan();
        let mut value = receipt(&plan);
        value.byte_length += 1;
        let error = verify_legacy_revision_capture_receipt_v3(
            &plan,
            value,
            &AcceptVerifier,
            1_800_000_003_000,
            60_000,
            1_000,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            LegacyRevisionCaptureIngestErrorV3::ReceiptCommitmentMismatch
        ));
    }
}
