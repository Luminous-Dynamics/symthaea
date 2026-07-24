// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use super::*;
use crate::evidence_calibration::{
    CALIBRATION_PUBLICATION_CATALOG_CHECKPOINT_VERSION,
    CalibrationSignerIdentity,
    calibration_publication_catalog_checkpoint_sha256,
    build_calibration_publication_recovery_authority_policy,
};

struct AcceptVerifier;

impl CalibrationPublicationRecoveryAuthorityRotationVerifier for AcceptVerifier {
    type Error = &'static str;

    fn verify(
        &self,
        _payload: &[u8],
        _signer: &CalibrationSignerIdentity,
        _signature: &[u8],
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

fn checkpoint(event_count: u64, issued_epoch: u64) -> CalibrationPublicationCatalogCheckpoint {
    let mut checkpoint = CalibrationPublicationCatalogCheckpoint {
        checkpoint_version: CALIBRATION_PUBLICATION_CATALOG_CHECKPOINT_VERSION.into(),
        catalog_id: "catalog".into(),
        authority_id: "authority".into(),
        catalog_version: "catalog-v1".into(),
        catalog_sha256: format!("{:064x}", event_count + 1),
        record_count: event_count,
        event_count,
        head_event_sha256: Some(format!("{:064x}", event_count + 2)),
        previous_checkpoint_sha256: None,
        issued_epoch,
        checkpoint_sha256: String::new(),
    };
    checkpoint.checkpoint_sha256 = calibration_publication_catalog_checkpoint_sha256(&checkpoint);
    checkpoint
}

fn policy(threshold: u64, keys: &[&str]) -> CalibrationPublicationRecoveryAuthorityPolicy {
    build_calibration_publication_recovery_authority_policy(
        threshold,
        keys.iter().map(|value| (*value).to_string()).collect(),
    )
    .expect("policy")
}

fn signer(key_id: &str) -> CalibrationSignerIdentity {
    CalibrationSignerIdentity {
        key_id: key_id.into(),
        algorithm: "test".into(),
        issuer: None,
    }
}

#[test]
fn recovery_authority_genesis_is_active() {
    let ledger = build_calibration_publication_recovery_authority_genesis(
        policy(1, &["a"]),
        checkpoint(1, 1),
        1,
    )
    .expect("genesis");
    let active = active_calibration_publication_recovery_authority_epoch(
        &ledger,
        &checkpoint(4, 4),
    )
    .expect("active authority");
    assert_eq!(active.ordinal, 0);
    assert!(audit_calibration_publication_recovery_authority_ledger(&ledger).valid());
}

#[test]
fn recovery_authority_rotation_requires_both_quorums() {
    let mut ledger = build_calibration_publication_recovery_authority_genesis(
        policy(1, &["a"]),
        checkpoint(1, 1),
        1,
    )
    .expect("genesis");
    let (epoch, payload) = plan_calibration_publication_recovery_authority_rotation(
        &ledger,
        checkpoint(3, 3),
        policy(1, &["b"]),
        3,
    )
    .expect("plan");
    let outgoing = build_calibration_signed_publication_recovery_authority_rotation(
        payload.clone(), signer("a"), &[1],
    );
    let incoming = build_calibration_signed_publication_recovery_authority_rotation(
        payload.clone(), signer("b"), &[2],
    );
    let set = build_calibration_publication_recovery_authority_rotation_set(
        &payload,
        &ledger.epochs[0].policy,
        &epoch.policy,
        vec![outgoing],
        vec![incoming],
    );
    append_calibration_publication_recovery_authority_rotation(
        &mut ledger,
        epoch,
        set,
        &AcceptVerifier,
    )
    .expect("append");
    let report = verify_calibration_publication_recovery_authority_ledger(
        &ledger,
        &AcceptVerifier,
    );
    assert!(report.accepted(), "{:?}", report.issues);
    assert_eq!(
        active_calibration_publication_recovery_authority_epoch(&ledger, &checkpoint(5, 5))
            .expect("active")
            .ordinal,
        1
    );
}

#[test]
fn recovery_authority_rotation_rolls_back_on_missing_incoming_quorum() {
    let mut ledger = build_calibration_publication_recovery_authority_genesis(
        policy(1, &["a"]),
        checkpoint(1, 1),
        1,
    )
    .expect("genesis");
    let (epoch, payload) = plan_calibration_publication_recovery_authority_rotation(
        &ledger,
        checkpoint(2, 2),
        policy(1, &["b"]),
        2,
    )
    .expect("plan");
    let outgoing = build_calibration_signed_publication_recovery_authority_rotation(
        payload.clone(), signer("a"), &[1],
    );
    let set = build_calibration_publication_recovery_authority_rotation_set(
        &payload,
        &ledger.epochs[0].policy,
        &epoch.policy,
        vec![outgoing],
        Vec::new(),
    );
    assert!(append_calibration_publication_recovery_authority_rotation(
        &mut ledger,
        epoch,
        set,
        &AcceptVerifier,
    )
    .is_err());
    assert_eq!(ledger.epochs.len(), 1);
}

#[test]
fn recovery_authority_noop_rotation_is_rejected() {
    let unchanged = policy(1, &["a"]);
    let ledger = build_calibration_publication_recovery_authority_genesis(
        unchanged.clone(), checkpoint(1, 1), 1,
    )
    .expect("genesis");
    assert!(matches!(
        plan_calibration_publication_recovery_authority_rotation(
            &ledger,
            checkpoint(2, 2),
            unchanged,
            2,
        ),
        Err(CalibrationPublicationRecoveryAuthorityError::PolicyUnchanged)
    ));
}

#[test]
fn recovery_authority_ledger_tampering_is_detected() {
    let mut ledger = build_calibration_publication_recovery_authority_genesis(
        policy(1, &["a"]), checkpoint(1, 1), 1,
    )
    .expect("genesis");
    ledger.ledger_sha256 = "00".repeat(32);
    let report = audit_calibration_publication_recovery_authority_ledger(&ledger);
    assert!(!report.valid());
    assert!(report.issues.iter().any(|issue| {
        issue.code == CalibrationPublicationRecoveryAuthorityIssueCode::LedgerSha256Mismatch
    }));
}
