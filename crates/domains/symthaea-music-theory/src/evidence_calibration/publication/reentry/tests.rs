// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use super::*;
use crate::evidence_calibration::*;
use crate::evidence_calibration::publication::incident_response_tests::{
    DigestVerifier, authorized_recovery_bundle, digest, signer,
};

pub(super) fn accepted_post_recovery_certification(
) -> CalibrationPublicationPostRecoveryCertification {
    let recovery_bundle = authorized_recovery_bundle();
    let recovered_anchor = build_calibration_publication_recovered_policy_anchor(&recovery_bundle)
        .expect("recovered anchor");
    let response = build_calibration_publication_incident_response_package(
        recovery_bundle.clone(),
        recovered_anchor.clone(),
        &DigestVerifier,
        &DigestVerifier,
        &DigestVerifier,
        &DigestVerifier,
        &DigestVerifier,
    )
    .expect("incident response");
    let (selected_catalog, selected_checkpoint) = calibration_publication_catalog_lineage_terminal(
        &recovery_bundle.plan.selected_lineage,
    );
    let selected_catalog = selected_catalog.clone();
    let selected_checkpoint = selected_checkpoint.clone();
    let next_checkpoint = build_calibration_publication_catalog_checkpoint(
        &selected_catalog,
        Some(&selected_checkpoint),
        6,
    )
    .expect("fresh checkpoint");
    let consistency = build_calibration_publication_catalog_consistency_proof(
        &selected_catalog,
        &selected_checkpoint,
        &selected_catalog,
        &next_checkpoint,
    )
    .expect("consistency");
    let witness_payload = build_calibration_publication_checkpoint_witness_payload(
        &next_checkpoint,
        6,
    );
    let witness = build_calibration_signed_publication_checkpoint_witness(
        witness_payload.clone(),
        signer("new-witness"),
        &digest(&witness_payload.canonical_bytes()),
    );
    let witness_policy = recovered_anchor.recovered_policy_ledger.epochs[0].policy.clone();
    let witness_set = build_calibration_publication_checkpoint_witness_set(
        &next_checkpoint,
        witness_policy,
        vec![witness],
    );
    let head = build_calibration_publication_catalog_head_bundle(
        selected_catalog.clone(),
        next_checkpoint.clone(),
        Some(CalibrationPublicationCatalogHeadPredecessor {
            catalog: selected_catalog.clone(),
            checkpoint: selected_checkpoint.clone(),
            consistency_proof: consistency,
        }),
        witness_set,
        None,
        Vec::new(),
        &DigestVerifier,
    )
    .expect("post-recovery head");
    let lineage = build_calibration_publication_catalog_lineage_chain(
        selected_catalog.clone(),
        selected_checkpoint.clone(),
        vec![(selected_catalog, next_checkpoint)],
    )
    .expect("post-recovery lineage");
    let continuity = build_calibration_publication_continuity_bundle(
        head,
        recovered_anchor.recovered_policy_ledger,
        lineage,
        None,
        Vec::new(),
        &DigestVerifier,
        &DigestVerifier,
        &DigestVerifier,
    )
    .expect("post-recovery continuity");
    let authority_ledger = build_calibration_publication_recovery_authority_genesis(
        recovery_bundle.recovery_authority_policy.clone(),
        selected_checkpoint,
        recovery_bundle.plan.recovery_epoch,
    )
    .expect("authority genesis");
    build_calibration_publication_post_recovery_certification(
        response,
        continuity,
        authority_ledger,
        0,
        7,
        &DigestVerifier,
        &DigestVerifier,
        &DigestVerifier,
        &DigestVerifier,
        &DigestVerifier,
        &DigestVerifier,
    )
    .expect("post-recovery certification")
}

#[test]
fn fresh_checkpoint_reentry_is_accepted() {
    let certification = accepted_post_recovery_certification();
    let report = verify_calibration_publication_post_recovery_certification(
        &certification,
        &DigestVerifier,
        &DigestVerifier,
        &DigestVerifier,
        &DigestVerifier,
        &DigestVerifier,
        &DigestVerifier,
    );
    assert!(report.accepted(), "{:?}", report.issues);
    assert!(report.fresh_checkpoint_confirmed);
}

#[test]
fn minimum_catalog_advance_is_enforced() {
    let mut certification = accepted_post_recovery_certification();
    certification.minimum_additional_catalog_events = 1;
    certification.certification_sha256 =
        calibration_publication_post_recovery_certification_sha256(&certification);
    let report = verify_calibration_publication_post_recovery_certification(
        &certification,
        &DigestVerifier,
        &DigestVerifier,
        &DigestVerifier,
        &DigestVerifier,
        &DigestVerifier,
        &DigestVerifier,
    );
    assert!(!report.accepted());
    assert!(report.issues.iter().any(|issue| {
        issue.code
            == CalibrationPublicationPostRecoveryIssueCode::InsufficientAdditionalCatalogEvents
    }));
}

#[test]
fn replacing_the_recovered_policy_anchor_is_detected_after_rehash() {
    let mut certification = accepted_post_recovery_certification();
    certification
        .continuity_bundle
        .witness_policy_ledger
        .epochs[0]
        .epoch_sha256 = "11".repeat(32);
    certification.continuity_bundle.witness_policy_ledger.ledger_sha256 =
        calibration_publication_witness_policy_ledger_sha256(
            &certification.continuity_bundle.witness_policy_ledger,
        );
    certification.continuity_bundle.bundle_sha256 =
        calibration_publication_continuity_bundle_sha256(&certification.continuity_bundle);
    certification.certification_sha256 =
        calibration_publication_post_recovery_certification_sha256(&certification);
    let report = audit_calibration_publication_post_recovery_certification(&certification);
    assert!(!report.valid());
    assert!(report.issues.iter().any(|issue| {
        issue.code == CalibrationPublicationPostRecoveryIssueCode::RecoveredPolicyAnchorMismatch
            || issue.code == CalibrationPublicationPostRecoveryIssueCode::ContinuityInvalid
    }));
}
