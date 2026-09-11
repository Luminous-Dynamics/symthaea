// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Public handoff boundary for V3 legacy capture ingest.
//!
//! Receipt verification and ingest can be separated in time. This wrapper
//! therefore recomputes the exact source-revision commitment against the current
//! pack immediately before mutation. The lower-level ingest implementation stays
//! crate-private through a private module and is not re-exported.

use crate::legacy_computing::LegacyComputingPackV1;
use crate::legacy_source_artifacts::LegacySourceArtifactLedgerV1;
use crate::legacy_source_capture_plan_v3::legacy_source_revision_commitment_v3;

pub use crate::legacy_capture_ingest_v3::{
    legacy_external_revision_capture_receipt_commitment_v3,
    legacy_revision_capture_request_commitment_v3,
    verify_legacy_revision_capture_receipt_v3,
    LegacyExternalRevisionCaptureReceiptV3, LegacyRevisionCaptureIngestErrorV3,
    LegacyRevisionCaptureIngestResultV3, LegacyRevisionCaptureReceiptVerifierV3,
    LegacyRevisionCaptureTrustEvidenceV3, VerifiedLegacyRevisionCaptureReceiptV3,
    LEGACY_EXTERNAL_REVISION_CAPTURE_RECEIPT_SCHEMA_V3,
};

/// Revalidate the exact source revision against the current pack immediately
/// before the transactional snapshot/artifact mutation.
pub fn ingest_verified_legacy_revision_capture_v3(
    pack: &mut LegacyComputingPackV1,
    artifacts: &mut LegacySourceArtifactLedgerV1,
    verified: &VerifiedLegacyRevisionCaptureReceiptV3,
) -> Result<LegacyRevisionCaptureIngestResultV3, LegacyRevisionCaptureIngestErrorV3> {
    pack.validate()?;
    let original_id = &verified.receipt().original_snapshot_id;
    let original = pack.sources.snapshot(original_id).ok_or_else(|| {
        LegacyRevisionCaptureIngestErrorV3::UnknownOriginalSnapshot(original_id.clone())
    })?;
    let document = pack.sources.document(&original.document_id).ok_or_else(|| {
        LegacyRevisionCaptureIngestErrorV3::InvalidField(format!(
            "current pack is missing logical document {} for source revision {}",
            original.document_id.0, original.id.0
        ))
    })?;
    let current_revision = legacy_source_revision_commitment_v3(document, original)
        .map_err(|err| LegacyRevisionCaptureIngestErrorV3::InvalidField(err.to_string()))?;
    if current_revision != verified.request().source_revision_blake3
        || current_revision != verified.receipt().source_revision_blake3
    {
        return Err(LegacyRevisionCaptureIngestErrorV3::SourceRevisionMismatch);
    }

    crate::legacy_capture_ingest_v3::ingest_verified_legacy_revision_capture_v3(
        pack, artifacts, verified,
    )
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
                verifier_id: "handoff-test".into(),
                trust_policy_sha256: POLICY.into(),
                signature_suite: "test-only".into(),
                verified_attestation_sha256: receipt.external_attestation_sha256.clone(),
            })
        }
    }

    #[test]
    fn handoff_revalidates_current_pack_revision() {
        let mut pack = build_legacy_five_platform_portfolio_v1(1_800_000_000_000)
            .unwrap()
            .0;
        let plan = plan_legacy_qualification_source_captures_v3(&pack).unwrap();
        let request = plan
            .requests
            .iter()
            .find(|request| request.capture_needed && request.capture_ready)
            .unwrap();
        let mut receipt = LegacyExternalRevisionCaptureReceiptV3 {
            schema_version: LEGACY_EXTERNAL_REVISION_CAPTURE_RECEIPT_SCHEMA_V3.into(),
            original_snapshot_id: request.original_snapshot_id.clone(),
            source_revision_blake3: request.source_revision_blake3.clone(),
            capture_request_blake3: legacy_revision_capture_request_commitment_v3(request).unwrap(),
            capture_service_id: "capture-service:test".into(),
            content_algorithm: "sha256".into(),
            content_digest: CONTENT.into(),
            byte_length: 100,
            media_type: "application/octet-stream".into(),
            fetched_at_unix_ms: 1_800_000_001_000,
            issued_at_unix_ms: 1_800_000_002_000,
            artifact_locator: "evidence://handoff/source".into(),
            retention_receipt_sha256: RETENTION.into(),
            external_attestation_sha256: ATTESTATION.into(),
            receipt_commitment_blake3: String::new(),
        };
        receipt.receipt_commitment_blake3 =
            legacy_external_revision_capture_receipt_commitment_v3(&receipt).unwrap();
        let verified = verify_legacy_revision_capture_receipt_v3(
            &plan,
            receipt,
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
        assert_eq!(
            result.original_snapshot_id,
            verified.receipt().original_snapshot_id
        );
    }
}
