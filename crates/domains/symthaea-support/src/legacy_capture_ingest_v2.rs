// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Offline ingress for externally captured legacy qualification sources.
//!
//! This module deliberately does not fetch vendor documentation, hold network
//! credentials, implement signature algorithms, or accept serialized receipts as
//! trusted facts. An external evidence service performs the fetch/retention step.
//! Its serialized receipt is inert until an injected verifier validates the
//! external attestation and returns a non-Serde verified wrapper.
//!
//! The ingest path is transactional at the in-memory model boundary: it applies
//! the content snapshot and retained-artifact receipt to clones first, and only
//! commits both objects when every invariant succeeds.
//!
//! Core non-equivalences:
//!
//! ```text
//! capture request != capture result
//! serialized receipt != verified receipt
//! verifier acceptance != source competence
//! content snapshot != claim verification
//! ingest success != qualification selection
//! ```

use crate::legacy_computing::{LegacyComputingErrorV1, LegacyComputingPackV1};
use crate::legacy_qualification_source_ledger_v2::LegacyQualificationSourceSelectionV2;
use crate::legacy_source_artifacts::{
    LegacyArtifactAccessPolicyV1, LegacyArtifactStorageClassV1,
    LegacySourceArtifactErrorV1, LegacySourceArtifactLedgerV1, LegacySourceArtifactRefV1,
};
use crate::legacy_source_capture_plan::{
    LegacyCaptureRetentionV2, LegacySourceCapturePlanV2, LegacySourceCaptureRequestV2,
    LEGACY_SOURCE_CAPTURE_PLAN_SCHEMA_V2,
};
use crate::standards_registry::{
    SourceCaptureV1, SourceDocumentIdV1, SourceSnapshotIdV1, StandardsRegistryErrorV1,
    TechnicalSourceSnapshotV1,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

pub const LEGACY_EXTERNAL_CAPTURE_RECEIPT_SCHEMA_V2: &str =
    "symthaea-it-legacy-external-capture-receipt-v2";

/// Serialized output from an external capture/retention service.
///
/// This object is evidence *input*, not trust. `verify_legacy_capture_receipt_v2`
/// must successfully pass it through a caller-supplied verifier before it can be
/// ingested.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyExternalCaptureReceiptV2 {
    pub schema_version: String,
    pub document_id: SourceDocumentIdV1,
    /// Historical snapshot whose metadata/version/lifecycle the exact byte
    /// capture is intended to preserve.
    pub basis_snapshot_id: SourceSnapshotIdV1,
    /// Binds the result to the exact V2 acquisition request.
    pub capture_request_blake3: String,
    pub capture_service_id: String,
    pub content_algorithm: String,
    pub content_digest: String,
    pub byte_length: u64,
    pub media_type: String,
    pub fetched_at_unix_ms: u64,
    pub issued_at_unix_ms: u64,
    /// Opaque location in the evidence store. Credentials/tokens are forbidden.
    pub artifact_locator: String,
    /// Digest of the external evidence store's ingest/retention receipt.
    pub retention_receipt_sha256: String,
    /// Digest of the external signed/attested envelope. Cryptographic validation
    /// remains the responsibility of `LegacyCaptureReceiptVerifierV2`.
    pub external_attestation_sha256: String,
    /// BLAKE3 commitment over every preceding semantic receipt field.
    pub receipt_commitment_blake3: String,
}

/// Trust evidence returned by the injected verifier.
///
/// This says which verifier/policy accepted the external attestation; it does not
/// grant operational authority or prove platform competence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegacyCaptureTrustEvidenceV2 {
    pub verifier_id: String,
    pub trust_policy_sha256: String,
    pub signature_suite: String,
    pub verified_attestation_sha256: String,
}

/// External trust boundary. Production implementations may delegate to Xenia,
/// an operator signer, an HSM-backed verifier, or another explicitly configured
/// trust service. `symthaea-support` owns none of those keys or algorithms.
pub trait LegacyCaptureReceiptVerifierV2 {
    fn verify(
        &self,
        receipt: &LegacyExternalCaptureReceiptV2,
        receipt_commitment_blake3: &str,
    ) -> Result<LegacyCaptureTrustEvidenceV2, String>;
}

/// Runtime proof that a serialized receipt passed the configured external trust
/// boundary. It is intentionally not Serialize/Deserialize and its fields are
/// private so callers cannot recreate it from wire data.
#[derive(Debug, Clone)]
pub struct VerifiedLegacyCaptureReceiptV2 {
    receipt: LegacyExternalCaptureReceiptV2,
    request: LegacySourceCaptureRequestV2,
    trust: LegacyCaptureTrustEvidenceV2,
}

impl VerifiedLegacyCaptureReceiptV2 {
    pub fn receipt(&self) -> &LegacyExternalCaptureReceiptV2 {
        &self.receipt
    }

    pub fn trust(&self) -> &LegacyCaptureTrustEvidenceV2 {
        &self.trust
    }

    pub fn request(&self) -> &LegacySourceCaptureRequestV2 {
        &self.request
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyCaptureIngestResultV2 {
    pub qualifying_snapshot_id: SourceSnapshotIdV1,
    pub snapshot_inserted: bool,
    pub artifact_inserted: bool,
    pub verifier_id: String,
    pub trust_policy_sha256: String,
    /// Candidate only. Ingest never writes the qualification-source ledger;
    /// selection remains a distinct explicit transition.
    pub selection_candidate: LegacyQualificationSourceSelectionV2,
}

/// Commitment to one exact capture-plan request. The request contains only
/// provenance/acquisition metadata, never source bytes.
pub fn legacy_capture_request_commitment_v2(
    request: &LegacySourceCaptureRequestV2,
) -> Result<String, LegacyCaptureIngestErrorV2> {
    let encoded = serde_json::to_vec(request)
        .map_err(|err| LegacyCaptureIngestErrorV2::Serialization(err.to_string()))?;
    let mut hasher = blake3::Hasher::new();
    frame(
        &mut hasher,
        b"schema",
        LEGACY_SOURCE_CAPTURE_PLAN_SCHEMA_V2.as_bytes(),
    );
    frame(&mut hasher, b"capture_request", &encoded);
    Ok(hasher.finalize().to_hex().to_string())
}

/// Commitment to the semantic receipt fields, excluding the commitment field
/// itself. External attestation systems should sign/bind this value.
pub fn legacy_external_capture_receipt_commitment_v2(
    receipt: &LegacyExternalCaptureReceiptV2,
) -> Result<String, LegacyCaptureIngestErrorV2> {
    #[derive(Serialize)]
    struct Commitment<'a> {
        schema_version: &'a str,
        document_id: &'a SourceDocumentIdV1,
        basis_snapshot_id: &'a SourceSnapshotIdV1,
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
        document_id: &receipt.document_id,
        basis_snapshot_id: &receipt.basis_snapshot_id,
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
    .map_err(|err| LegacyCaptureIngestErrorV2::Serialization(err.to_string()))?;

    let mut hasher = blake3::Hasher::new();
    frame(
        &mut hasher,
        b"schema",
        LEGACY_EXTERNAL_CAPTURE_RECEIPT_SCHEMA_V2.as_bytes(),
    );
    frame(&mut hasher, b"receipt", &encoded);
    Ok(hasher.finalize().to_hex().to_string())
}

pub fn verify_legacy_capture_receipt_v2<V: LegacyCaptureReceiptVerifierV2>(
    plan: &LegacySourceCapturePlanV2,
    receipt: LegacyExternalCaptureReceiptV2,
    verifier: &V,
    now_unix_ms: u64,
    max_receipt_age_ms: u64,
    max_future_skew_ms: u64,
) -> Result<VerifiedLegacyCaptureReceiptV2, LegacyCaptureIngestErrorV2> {
    validate_receipt_shape(&receipt)?;
    if plan.schema_version != LEGACY_SOURCE_CAPTURE_PLAN_SCHEMA_V2 {
        return Err(LegacyCaptureIngestErrorV2::UnsupportedCapturePlanSchema(
            plan.schema_version.clone(),
        ));
    }
    if now_unix_ms == 0 {
        return Err(LegacyCaptureIngestErrorV2::InvalidField(
            "verification clock must be non-zero".into(),
        ));
    }
    if max_receipt_age_ms == 0 {
        return Err(LegacyCaptureIngestErrorV2::InvalidField(
            "maximum receipt age must be non-zero".into(),
        ));
    }

    let request = plan
        .requests
        .iter()
        .find(|request| request.document_id == receipt.document_id)
        .ok_or_else(|| LegacyCaptureIngestErrorV2::UnknownCaptureRequest(receipt.document_id.clone()))?
        .clone();
    if !request.capture_ready {
        return Err(LegacyCaptureIngestErrorV2::CaptureRequestNotReady(
            receipt.document_id.clone(),
        ));
    }
    if !request
        .metadata_snapshot_ids
        .contains(&receipt.basis_snapshot_id)
    {
        return Err(LegacyCaptureIngestErrorV2::BasisSnapshotOutsideRequest(
            receipt.basis_snapshot_id.clone(),
        ));
    }

    let expected_request = legacy_capture_request_commitment_v2(&request)?;
    if expected_request != receipt.capture_request_blake3.trim().to_ascii_lowercase() {
        return Err(LegacyCaptureIngestErrorV2::CaptureRequestCommitmentMismatch);
    }
    if !request
        .content_digest_algorithm
        .eq_ignore_ascii_case(receipt.content_algorithm.trim())
    {
        return Err(LegacyCaptureIngestErrorV2::UnexpectedDigestAlgorithm(
            receipt.content_algorithm.clone(),
        ));
    }

    let future_limit = now_unix_ms
        .checked_add(max_future_skew_ms)
        .ok_or_else(|| LegacyCaptureIngestErrorV2::InvalidField("clock overflow".into()))?;
    if receipt.issued_at_unix_ms > future_limit || receipt.fetched_at_unix_ms > future_limit {
        return Err(LegacyCaptureIngestErrorV2::ReceiptFromFuture);
    }
    if receipt.fetched_at_unix_ms > receipt.issued_at_unix_ms {
        return Err(LegacyCaptureIngestErrorV2::CaptureAfterReceiptIssued);
    }
    if now_unix_ms >= receipt.issued_at_unix_ms
        && now_unix_ms - receipt.issued_at_unix_ms > max_receipt_age_ms
    {
        return Err(LegacyCaptureIngestErrorV2::ReceiptTooOld);
    }

    let expected_receipt = legacy_external_capture_receipt_commitment_v2(&receipt)?;
    if expected_receipt != receipt.receipt_commitment_blake3.trim().to_ascii_lowercase() {
        return Err(LegacyCaptureIngestErrorV2::ReceiptCommitmentMismatch);
    }

    let trust = verifier
        .verify(&receipt, &expected_receipt)
        .map_err(LegacyCaptureIngestErrorV2::ExternalVerificationRejected)?;
    validate_trust_evidence(&receipt, &trust)?;

    Ok(VerifiedLegacyCaptureReceiptV2 {
        receipt,
        request,
        trust,
    })
}

/// Atomically add the new content-bound snapshot and retained-artifact receipt to
/// the in-memory legacy pack/artifact ledger. Qualification selection remains a
/// separate explicit step using the returned candidate.
pub fn ingest_verified_legacy_capture_v2(
    pack: &mut LegacyComputingPackV1,
    artifacts: &mut LegacySourceArtifactLedgerV1,
    verified: &VerifiedLegacyCaptureReceiptV2,
) -> Result<LegacyCaptureIngestResultV2, LegacyCaptureIngestErrorV2> {
    pack.validate()?;
    artifacts.validate_against_pack(pack)?;

    let basis = pack
        .sources
        .snapshot(&verified.receipt.basis_snapshot_id)
        .ok_or_else(|| {
            LegacyCaptureIngestErrorV2::UnknownBasisSnapshot(
                verified.receipt.basis_snapshot_id.clone(),
            )
        })?
        .clone();
    if basis.document_id != verified.receipt.document_id {
        return Err(LegacyCaptureIngestErrorV2::BasisDocumentMismatch);
    }

    let snapshot_id = derived_content_snapshot_id(&verified.receipt);
    let snapshot = TechnicalSourceSnapshotV1 {
        id: snapshot_id.clone(),
        document_id: basis.document_id.clone(),
        version: basis.version.clone(),
        lifecycle: basis.lifecycle,
        authority: basis.authority,
        stability: basis.stability,
        published_at_unix_ms: basis.published_at_unix_ms,
        source_updated_at_unix_ms: basis.source_updated_at_unix_ms,
        fetched_at_unix_ms: verified.receipt.fetched_at_unix_ms,
        capture: SourceCaptureV1::ContentDigest {
            algorithm: verified.receipt.content_algorithm.trim().to_ascii_lowercase(),
            digest: verified.receipt.content_digest.trim().to_ascii_lowercase(),
        },
        relations: basis.relations.clone(),
    };

    let (storage_class, access_policy) = artifact_policy(verified.request.retention);
    let artifact = LegacySourceArtifactRefV1 {
        snapshot_id: snapshot_id.clone(),
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

    // Preflight/commit on clones so a failed artifact insertion cannot leave a
    // newly registered source snapshot behind.
    let mut next_pack = pack.clone();
    let mut next_artifacts = artifacts.clone();
    let snapshot_inserted = next_pack.sources.register_snapshot(snapshot)?;
    next_pack.validate()?;
    let artifact_inserted = next_artifacts.register(&next_pack, artifact)?;
    next_artifacts.validate_against_pack(&next_pack)?;

    *pack = next_pack;
    *artifacts = next_artifacts;

    let selection_candidate = LegacyQualificationSourceSelectionV2 {
        document_id: verified.receipt.document_id.clone(),
        qualifying_snapshot_id: snapshot_id.clone(),
        content_algorithm: verified.receipt.content_algorithm.trim().to_ascii_lowercase(),
        content_digest: verified.receipt.content_digest.trim().to_ascii_lowercase(),
        selected_at_unix_ms: verified.receipt.issued_at_unix_ms,
    };

    Ok(LegacyCaptureIngestResultV2 {
        qualifying_snapshot_id: snapshot_id,
        snapshot_inserted,
        artifact_inserted,
        verifier_id: verified.trust.verifier_id.clone(),
        trust_policy_sha256: verified.trust.trust_policy_sha256.clone(),
        selection_candidate,
    })
}

fn validate_receipt_shape(
    receipt: &LegacyExternalCaptureReceiptV2,
) -> Result<(), LegacyCaptureIngestErrorV2> {
    if receipt.schema_version != LEGACY_EXTERNAL_CAPTURE_RECEIPT_SCHEMA_V2 {
        return Err(LegacyCaptureIngestErrorV2::UnsupportedReceiptSchema(
            receipt.schema_version.clone(),
        ));
    }
    require_nonempty(&receipt.document_id.0, "document id")?;
    require_nonempty(&receipt.basis_snapshot_id.0, "basis snapshot id")?;
    require_blake3(&receipt.capture_request_blake3, "capture request commitment")?;
    require_nonempty(&receipt.capture_service_id, "capture service id")?;
    if !receipt.content_algorithm.trim().eq_ignore_ascii_case("sha256") {
        return Err(LegacyCaptureIngestErrorV2::UnexpectedDigestAlgorithm(
            receipt.content_algorithm.clone(),
        ));
    }
    require_sha256(&receipt.content_digest, "content digest")?;
    if receipt.byte_length == 0 {
        return Err(LegacyCaptureIngestErrorV2::InvalidField(
            "captured byte length must be non-zero".into(),
        ));
    }
    require_nonempty(&receipt.media_type, "media type")?;
    if receipt.fetched_at_unix_ms == 0 || receipt.issued_at_unix_ms == 0 {
        return Err(LegacyCaptureIngestErrorV2::InvalidField(
            "capture timestamps must be non-zero".into(),
        ));
    }
    require_nonempty(&receipt.artifact_locator, "artifact locator")?;
    if looks_secret_bearing(&receipt.artifact_locator) {
        return Err(LegacyCaptureIngestErrorV2::SecretBearingArtifactLocator);
    }
    require_sha256(
        &receipt.retention_receipt_sha256,
        "retention receipt digest",
    )?;
    require_sha256(
        &receipt.external_attestation_sha256,
        "external attestation digest",
    )?;
    require_blake3(&receipt.receipt_commitment_blake3, "receipt commitment")?;
    Ok(())
}

fn validate_trust_evidence(
    receipt: &LegacyExternalCaptureReceiptV2,
    trust: &LegacyCaptureTrustEvidenceV2,
) -> Result<(), LegacyCaptureIngestErrorV2> {
    require_nonempty(&trust.verifier_id, "verifier id")?;
    require_sha256(&trust.trust_policy_sha256, "trust policy digest")?;
    require_nonempty(&trust.signature_suite, "signature suite")?;
    require_sha256(
        &trust.verified_attestation_sha256,
        "verified attestation digest",
    )?;
    if trust.verified_attestation_sha256.to_ascii_lowercase()
        != receipt.external_attestation_sha256.to_ascii_lowercase()
    {
        return Err(LegacyCaptureIngestErrorV2::AttestationDigestMismatch);
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

fn derived_content_snapshot_id(receipt: &LegacyExternalCaptureReceiptV2) -> SourceSnapshotIdV1 {
    SourceSnapshotIdV1(format!(
        "{}:content:{}:{}:{}",
        receipt.basis_snapshot_id.0,
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

fn require_nonempty(value: &str, field: &'static str) -> Result<(), LegacyCaptureIngestErrorV2> {
    if value.trim().is_empty() {
        return Err(LegacyCaptureIngestErrorV2::InvalidField(format!(
            "{field} must be non-empty"
        )));
    }
    Ok(())
}

fn require_sha256(value: &str, field: &'static str) -> Result<(), LegacyCaptureIngestErrorV2> {
    let value = value.trim();
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(LegacyCaptureIngestErrorV2::InvalidField(format!(
            "{field} must be a 64-character hex SHA-256"
        )));
    }
    Ok(())
}

fn require_blake3(value: &str, field: &'static str) -> Result<(), LegacyCaptureIngestErrorV2> {
    let value = value.trim();
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(LegacyCaptureIngestErrorV2::InvalidField(format!(
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
pub enum LegacyCaptureIngestErrorV2 {
    LegacyPack(LegacyComputingErrorV1),
    Standards(StandardsRegistryErrorV1),
    Artifact(LegacySourceArtifactErrorV1),
    UnsupportedCapturePlanSchema(String),
    UnsupportedReceiptSchema(String),
    InvalidField(String),
    Serialization(String),
    UnknownCaptureRequest(SourceDocumentIdV1),
    CaptureRequestNotReady(SourceDocumentIdV1),
    BasisSnapshotOutsideRequest(SourceSnapshotIdV1),
    CaptureRequestCommitmentMismatch,
    UnexpectedDigestAlgorithm(String),
    ReceiptFromFuture,
    CaptureAfterReceiptIssued,
    ReceiptTooOld,
    ReceiptCommitmentMismatch,
    ExternalVerificationRejected(String),
    AttestationDigestMismatch,
    SecretBearingArtifactLocator,
    UnknownBasisSnapshot(SourceSnapshotIdV1),
    BasisDocumentMismatch,
}

impl fmt::Display for LegacyCaptureIngestErrorV2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LegacyPack(err) => write!(f, "legacy capture ingest pack error: {err}"),
            Self::Standards(err) => write!(f, "legacy capture ingest registry error: {err}"),
            Self::Artifact(err) => write!(f, "legacy capture ingest artifact error: {err}"),
            Self::UnsupportedCapturePlanSchema(schema) => {
                write!(f, "unsupported legacy capture-plan schema {schema}")
            }
            Self::UnsupportedReceiptSchema(schema) => {
                write!(f, "unsupported legacy external capture receipt schema {schema}")
            }
            Self::InvalidField(message) => write!(f, "invalid legacy capture field: {message}"),
            Self::Serialization(message) => write!(f, "legacy capture serialization failed: {message}"),
            Self::UnknownCaptureRequest(id) => {
                write!(f, "no capture request exists for logical document {}", id.0)
            }
            Self::CaptureRequestNotReady(id) => {
                write!(f, "capture request for logical document {} is not ready", id.0)
            }
            Self::BasisSnapshotOutsideRequest(id) => {
                write!(f, "basis snapshot {} is outside the capture request", id.0)
            }
            Self::CaptureRequestCommitmentMismatch => {
                write!(f, "capture result does not bind the exact capture request")
            }
            Self::UnexpectedDigestAlgorithm(value) => {
                write!(f, "unexpected capture digest algorithm {value}")
            }
            Self::ReceiptFromFuture => write!(f, "capture receipt timestamp exceeds allowed future skew"),
            Self::CaptureAfterReceiptIssued => {
                write!(f, "capture fetch timestamp is after receipt issuance")
            }
            Self::ReceiptTooOld => write!(f, "capture receipt exceeds maximum accepted age"),
            Self::ReceiptCommitmentMismatch => write!(f, "capture receipt commitment mismatch"),
            Self::ExternalVerificationRejected(message) => {
                write!(f, "external capture attestation rejected: {message}")
            }
            Self::AttestationDigestMismatch => {
                write!(f, "verifier accepted a different external attestation digest")
            }
            Self::SecretBearingArtifactLocator => write!(
                f,
                "artifact locator appears to contain secret-bearing material"
            ),
            Self::UnknownBasisSnapshot(id) => write!(f, "unknown capture basis snapshot {}", id.0),
            Self::BasisDocumentMismatch => {
                write!(f, "capture basis snapshot belongs to the wrong logical document")
            }
        }
    }
}

impl Error for LegacyCaptureIngestErrorV2 {}

impl From<LegacyComputingErrorV1> for LegacyCaptureIngestErrorV2 {
    fn from(value: LegacyComputingErrorV1) -> Self {
        Self::LegacyPack(value)
    }
}

impl From<StandardsRegistryErrorV1> for LegacyCaptureIngestErrorV2 {
    fn from(value: StandardsRegistryErrorV1) -> Self {
        Self::Standards(value)
    }
}

impl From<LegacySourceArtifactErrorV1> for LegacyCaptureIngestErrorV2 {
    fn from(value: LegacySourceArtifactErrorV1) -> Self {
        Self::Artifact(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        build_legacy_five_platform_portfolio_v1,
        plan_legacy_qualification_source_captures_v2,
    };

    const CONTENT: &str =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const RECEIPT: &str =
        "1111111111111111111111111111111111111111111111111111111111111111";
    const ATTESTATION: &str =
        "2222222222222222222222222222222222222222222222222222222222222222";
    const POLICY: &str =
        "3333333333333333333333333333333333333333333333333333333333333333";

    struct AcceptVerifier;
    impl LegacyCaptureReceiptVerifierV2 for AcceptVerifier {
        fn verify(
            &self,
            receipt: &LegacyExternalCaptureReceiptV2,
            _receipt_commitment_blake3: &str,
        ) -> Result<LegacyCaptureTrustEvidenceV2, String> {
            Ok(LegacyCaptureTrustEvidenceV2 {
                verifier_id: "test-verifier".into(),
                trust_policy_sha256: POLICY.into(),
                signature_suite: "test-only".into(),
                verified_attestation_sha256: receipt.external_attestation_sha256.clone(),
            })
        }
    }

    struct RejectVerifier;
    impl LegacyCaptureReceiptVerifierV2 for RejectVerifier {
        fn verify(
            &self,
            _receipt: &LegacyExternalCaptureReceiptV2,
            _receipt_commitment_blake3: &str,
        ) -> Result<LegacyCaptureTrustEvidenceV2, String> {
            Err("untrusted signer".into())
        }
    }

    fn pack_and_plan() -> (LegacyComputingPackV1, LegacySourceCapturePlanV2) {
        let pack = build_legacy_five_platform_portfolio_v1(1_800_000_000_000)
            .unwrap()
            .0;
        let plan = plan_legacy_qualification_source_captures_v2(&pack).unwrap();
        (pack, plan)
    }

    fn receipt(plan: &LegacySourceCapturePlanV2) -> LegacyExternalCaptureReceiptV2 {
        let request = plan.requests.iter().find(|request| request.capture_ready).unwrap();
        let mut value = LegacyExternalCaptureReceiptV2 {
            schema_version: LEGACY_EXTERNAL_CAPTURE_RECEIPT_SCHEMA_V2.into(),
            document_id: request.document_id.clone(),
            basis_snapshot_id: request.metadata_snapshot_ids.iter().next().unwrap().clone(),
            capture_request_blake3: legacy_capture_request_commitment_v2(request).unwrap(),
            capture_service_id: "capture-service:test".into(),
            content_algorithm: "sha256".into(),
            content_digest: CONTENT.into(),
            byte_length: 42_000,
            media_type: "application/pdf".into(),
            fetched_at_unix_ms: 1_800_000_001_000,
            issued_at_unix_ms: 1_800_000_002_000,
            artifact_locator: "evidence://legacy/test/source.pdf".into(),
            retention_receipt_sha256: RECEIPT.into(),
            external_attestation_sha256: ATTESTATION.into(),
            receipt_commitment_blake3: String::new(),
        };
        value.receipt_commitment_blake3 =
            legacy_external_capture_receipt_commitment_v2(&value).unwrap();
        value
    }

    #[test]
    fn serialized_receipt_requires_external_verifier_acceptance() {
        let (_pack, plan) = pack_and_plan();
        let error = verify_legacy_capture_receipt_v2(
            &plan,
            receipt(&plan),
            &RejectVerifier,
            1_800_000_003_000,
            60_000,
            1_000,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            LegacyCaptureIngestErrorV2::ExternalVerificationRejected(_)
        ));
    }

    #[test]
    fn request_or_receipt_tampering_fails_before_ingest() {
        let (_pack, plan) = pack_and_plan();
        let mut value = receipt(&plan);
        value.byte_length += 1;
        let error = verify_legacy_capture_receipt_v2(
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
            LegacyCaptureIngestErrorV2::ReceiptCommitmentMismatch
        ));
    }

    #[test]
    fn verified_ingest_preserves_metadata_history_and_registers_exact_artifact() {
        let (mut pack, plan) = pack_and_plan();
        let metadata_before = pack
            .sources
            .snapshots()
            .filter(|snapshot| matches!(&snapshot.capture, SourceCaptureV1::MetadataOnly))
            .count();
        let verified = verify_legacy_capture_receipt_v2(
            &plan,
            receipt(&plan),
            &AcceptVerifier,
            1_800_000_003_000,
            60_000,
            1_000,
        )
        .unwrap();
        let mut artifacts = LegacySourceArtifactLedgerV1::new();
        let result = ingest_verified_legacy_capture_v2(&mut pack, &mut artifacts, &verified).unwrap();
        assert!(result.snapshot_inserted);
        assert!(result.artifact_inserted);
        assert_eq!(
            pack.sources
                .snapshots()
                .filter(|snapshot| matches!(&snapshot.capture, SourceCaptureV1::MetadataOnly))
                .count(),
            metadata_before
        );
        let snapshot = pack.sources.snapshot(&result.qualifying_snapshot_id).unwrap();
        assert!(matches!(&snapshot.capture, SourceCaptureV1::ContentDigest { .. }));
        let artifact = artifacts.artifact(&result.qualifying_snapshot_id).unwrap();
        assert_eq!(artifact.content_digest, CONTENT);
        assert_eq!(artifact.retention_receipt_digest.as_deref(), Some(RECEIPT));
    }

    #[test]
    fn exact_verified_replay_is_idempotent() {
        let (mut pack, plan) = pack_and_plan();
        let verified = verify_legacy_capture_receipt_v2(
            &plan,
            receipt(&plan),
            &AcceptVerifier,
            1_800_000_003_000,
            60_000,
            1_000,
        )
        .unwrap();
        let mut artifacts = LegacySourceArtifactLedgerV1::new();
        let first = ingest_verified_legacy_capture_v2(&mut pack, &mut artifacts, &verified).unwrap();
        let second = ingest_verified_legacy_capture_v2(&mut pack, &mut artifacts, &verified).unwrap();
        assert!(first.snapshot_inserted && first.artifact_inserted);
        assert!(!second.snapshot_inserted && !second.artifact_inserted);
        assert_eq!(first.qualifying_snapshot_id, second.qualifying_snapshot_id);
    }

    #[test]
    fn secret_bearing_artifact_locator_is_rejected_without_echoing_secret() {
        let (_pack, plan) = pack_and_plan();
        let mut value = receipt(&plan);
        value.artifact_locator = "https://store.example/evidence?token=super-secret".into();
        value.receipt_commitment_blake3 =
            legacy_external_capture_receipt_commitment_v2(&value).unwrap();
        let error = verify_legacy_capture_receipt_v2(
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
            LegacyCaptureIngestErrorV2::SecretBearingArtifactLocator
        ));
        assert!(!error.to_string().contains("super-secret"));
    }
}
