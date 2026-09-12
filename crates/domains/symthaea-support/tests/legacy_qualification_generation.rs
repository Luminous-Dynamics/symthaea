// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_support::legacy_qualification_profile_v3::{
    assess_legacy_qualification_generation_v1, assess_legacy_qualification_profile_v3,
    initial_legacy_qualification_manifest_v1, LegacyQualificationManifestErrorV1,
    LegacyQualificationProfileErrorV3,
};
use symthaea_support::{
    build_legacy_five_platform_portfolio_v1, exhaustive_legacy_qualification_profile_v1,
    legacy_source_revision_commitment_v3, LegacyArtifactAccessPolicyV1,
    LegacyArtifactStorageClassV1, LegacyQualificationSourceLedgerV3,
    LegacyQualificationSourceSelectionV3, LegacySourceArtifactLedgerV1,
    LegacySourceArtifactRefV1, SourceCaptureV1, SourceSnapshotIdV1,
    TechnicalSourceSnapshotV1,
};

#[test]
fn full_registry_profile_still_rejects_retroactive_metadata_only_capture() {
    let (mut pack, matrix, _) =
        build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
    let original_id = SourceSnapshotIdV1("ibm:aix-os-management@7.3".into());
    let original = pack.sources.snapshot(&original_id).unwrap().clone();
    let qualifying_id = SourceSnapshotIdV1(
        "ibm:aix-os-management@7.3:integration-retroactive-content".into(),
    );
    let digest = "7".repeat(64);

    pack.sources
        .register_snapshot(TechnicalSourceSnapshotV1 {
            id: qualifying_id.clone(),
            document_id: original.document_id.clone(),
            version: original.version.clone(),
            lifecycle: original.lifecycle,
            authority: original.authority,
            stability: original.stability,
            published_at_unix_ms: original.published_at_unix_ms,
            source_updated_at_unix_ms: original.source_updated_at_unix_ms,
            fetched_at_unix_ms: original.fetched_at_unix_ms + 100,
            capture: SourceCaptureV1::ContentDigest {
                algorithm: "sha256".into(),
                digest: digest.clone(),
            },
            relations: original.relations.clone(),
        })
        .unwrap();

    let mut artifacts = LegacySourceArtifactLedgerV1::new();
    artifacts
        .register(
            &pack,
            LegacySourceArtifactRefV1 {
                snapshot_id: qualifying_id.clone(),
                content_algorithm: "sha256".into(),
                content_digest: digest.clone(),
                byte_length: 1,
                media_type: "application/octet-stream".into(),
                retrieved_at_unix_ms: original.fetched_at_unix_ms + 100,
                artifact_locator: "evidence://continuity/integration-regression".into(),
                storage_class: LegacyArtifactStorageClassV1::PrivateEvidenceStore,
                access_policy: LegacyArtifactAccessPolicyV1::EvaluatorOnly,
                retention_receipt_digest: Some("8".repeat(64)),
            },
        )
        .unwrap();

    let document = pack.sources.document(&original.document_id).unwrap();
    let mut source_ledger = LegacyQualificationSourceLedgerV3::new();
    source_ledger
        .register_selection(
            &pack,
            &artifacts,
            LegacyQualificationSourceSelectionV3 {
                original_snapshot_id: original.id.clone(),
                source_revision_blake3: legacy_source_revision_commitment_v3(document, &original)
                    .unwrap(),
                qualifying_snapshot_id: qualifying_id,
                content_algorithm: "sha256".into(),
                content_digest: digest,
                selected_at_unix_ms: original.fetched_at_unix_ms + 101,
            },
        )
        .unwrap();

    let profile = exhaustive_legacy_qualification_profile_v1();
    let error = assess_legacy_qualification_profile_v3(
        &pack,
        &profile,
        &matrix,
        &artifacts,
        &source_ledger,
    )
    .unwrap_err();

    assert!(matches!(
        error,
        LegacyQualificationProfileErrorV3::Continuity(_)
    ));
}

#[test]
fn generation_one_cannot_silently_shrink_the_active_registry() {
    let (pack, matrix, _) =
        build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
    let profile = exhaustive_legacy_qualification_profile_v1();
    let artifacts = LegacySourceArtifactLedgerV1::new();
    let source_ledger = LegacyQualificationSourceLedgerV3::new();
    let mut manifest = initial_legacy_qualification_manifest_v1(&pack).unwrap();
    manifest.claim_ids.pop_first();

    let error = assess_legacy_qualification_generation_v1(
        &pack,
        &profile,
        &matrix,
        &artifacts,
        &source_ledger,
        None,
        &manifest,
    )
    .unwrap_err();

    assert!(matches!(
        error,
        LegacyQualificationProfileErrorV3::Manifest(
            LegacyQualificationManifestErrorV1::NonCanonicalInitialManifest
        )
    ));
}
