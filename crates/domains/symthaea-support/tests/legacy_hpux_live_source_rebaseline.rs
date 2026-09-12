// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use symthaea_support::{
    assess_legacy_qualification_manifest_source_readiness_v1,
    build_legacy_five_platform_portfolio_v1, initial_legacy_qualification_manifest_v1,
    plan_legacy_qualification_manifest_captures_v1, LegacyQualificationSourceLedgerV3,
    LegacySourceArtifactLedgerV1, SourceCaptureV1, SourceSnapshotIdV1, TechnicalClaimIdV1,
};

const FETCHED_AT_UNIX_MS: u64 = 1_800_000_000_000;
const HPUX_INSTALL_SNAPSHOT: &str = "hpe:hpux-install-update@2025-05";
const HPUX_INSTALL_DOCUMENT: &str = "hpe:hpux-install-update";
const HPUX_INSTALL_LOCATOR: &str =
    "https://support.hpe.com/hpesc/public/api/document/dp00006246en_us";

#[test]
fn hpux_may_2025_guide_cannot_progress_past_capture_without_exact_bytes() {
    let (pack, _, _) = build_legacy_five_platform_portfolio_v1(FETCHED_AT_UNIX_MS).unwrap();
    let manifest = initial_legacy_qualification_manifest_v1(&pack).unwrap();
    let plan = plan_legacy_qualification_manifest_captures_v1(&pack, &manifest).unwrap();

    let snapshot_id = SourceSnapshotIdV1(HPUX_INSTALL_SNAPSHOT.into());
    let request = plan
        .requests
        .iter()
        .find(|request| request.original_snapshot_id == snapshot_id)
        .expect("HP-UX May 2025 install/update guide must be in the active capture plan");

    assert_eq!(request.document_id.0, HPUX_INSTALL_DOCUMENT);
    assert_eq!(request.canonical_locator.as_deref(), Some(HPUX_INSTALL_LOCATOR));
    assert_eq!(request.observed_version.as_deref(), Some("2025-05"));
    assert!(request.capture_needed);
    assert!(request.capture_ready);
    assert!(!request.existing_content_bound);

    assert!(request
        .claim_ids
        .contains(&TechnicalClaimIdV1("legacy:hpux:ignite-recovery-context".into())));
    assert!(request.claim_ids.contains(&TechnicalClaimIdV1(
        "legacy:hpux:software-distributor-verification".into()
    )));

    // The seeded historical observation remains metadata-only. Live metadata or
    // a failed raw-byte fetch must not be converted into a content identity.
    let snapshot = pack.sources.snapshot(&snapshot_id).unwrap();
    assert!(matches!(snapshot.capture, SourceCaptureV1::MetadataOnly));

    let artifacts = LegacySourceArtifactLedgerV1::new();
    let source_ledger = LegacyQualificationSourceLedgerV3::new();
    let readiness = assess_legacy_qualification_manifest_source_readiness_v1(
        &pack,
        &artifacts,
        &source_ledger,
        &manifest,
    )
    .unwrap();

    assert!(!readiness.source_evidence_ready);
    assert!(readiness.missing_source_selections.contains(&snapshot_id));
    assert!(readiness.missing_claim_receipts.contains(&TechnicalClaimIdV1(
        "legacy:hpux:ignite-recovery-context".into()
    )));
    assert!(readiness.missing_claim_receipts.contains(&TechnicalClaimIdV1(
        "legacy:hpux:software-distributor-verification".into()
    )));
}

#[test]
fn hpux_install_guide_current_procedure_fanout_is_visible_for_precision_followup() {
    let (pack, _, _) = build_legacy_five_platform_portfolio_v1(FETCHED_AT_UNIX_MS).unwrap();
    let manifest = initial_legacy_qualification_manifest_v1(&pack).unwrap();
    let plan = plan_legacy_qualification_manifest_captures_v1(&pack, &manifest).unwrap();

    let request = plan
        .requests
        .iter()
        .find(|request| request.original_snapshot_id.0 == HPUX_INSTALL_SNAPSHOT)
        .unwrap();

    // These two procedures genuinely depend on the installation/update guide
    // and must continue to appear after procedure-source bindings are narrowed.
    assert!(request
        .procedure_ids
        .contains("legacy:hpux:ignite-recovery-triage"));
    assert!(request
        .procedure_ids
        .contains("legacy:hpux:software-update-triage"));

    // V1 HP-UX enrichment currently binds every HP-UX procedure to every HP-UX
    // source snapshot. Do not treat this count as semantic precision; the test
    // intentionally records the over-broad fanout so a future source-precision
    // tranche can narrow it without losing the two genuinely dependent paths.
    assert!(request.procedure_ids.len() >= 2);
}
