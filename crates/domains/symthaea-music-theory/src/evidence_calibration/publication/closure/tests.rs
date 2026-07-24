// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use super::*;
use crate::evidence_calibration::*;
use crate::evidence_calibration::publication::incident_response_tests::{
    DigestVerifier, digest, signer,
};
use crate::evidence_calibration::publication::reentry::tests::accepted_post_recovery_certification;

pub(super) fn accepted_incident_closure_bundle(
) -> CalibrationPublicationIncidentClosureBundle {
    let certification = accepted_post_recovery_certification();
    let quarantine = certification
        .incident_response_package
        .recovery_bundle
        .quarantine_ledger
        .clone();
    let policy = build_calibration_publication_incident_closure_policy(
        0,
        None,
        None,
        true,
        false,
    )
    .expect("closure policy");
    let plan = plan_calibration_publication_incident_closure(
        &certification,
        &quarantine,
        &policy,
        8,
    )
    .expect("closure plan");
    let authority_statement = build_calibration_signed_publication_incident_closure_statement(
        &plan,
        CalibrationPublicationIncidentClosureSignerRole::RecoveryAuthority,
        signer("recovery-authority"),
        &digest(&plan.canonical_bytes()),
    );
    let witness_statement = build_calibration_signed_publication_incident_closure_statement(
        &plan,
        CalibrationPublicationIncidentClosureSignerRole::RecoveredWitness,
        signer("new-witness"),
        &digest(&plan.canonical_bytes()),
    );
    let head = &certification.continuity_bundle.head_bundle.checkpoint;
    let authority_policy = &active_calibration_publication_recovery_authority_epoch(
        &certification.recovery_authority_ledger,
        head,
    )
    .expect("active authority")
    .policy;
    let witness_policy = &active_calibration_publication_witness_policy_epoch(
        &certification.continuity_bundle.witness_policy_ledger,
        head,
    )
    .expect("active witness policy")
    .policy;
    let authorization = build_calibration_publication_incident_closure_authorization_set(
        &plan,
        authority_policy,
        witness_policy,
        vec![authority_statement],
        vec![witness_statement],
    );
    build_calibration_publication_incident_closure_bundle(
        certification,
        quarantine,
        policy,
        plan,
        authorization,
        &DigestVerifier,
        &DigestVerifier,
        &DigestVerifier,
        &DigestVerifier,
        &DigestVerifier,
        &DigestVerifier,
        &DigestVerifier,
    )
    .expect("incident closure")
}

#[test]
fn operational_closure_requires_recovered_authority_and_witness() {
    let bundle = accepted_incident_closure_bundle();
    let report = verify_calibration_publication_incident_closure_bundle(
        &bundle,
        &DigestVerifier,
        &DigestVerifier,
        &DigestVerifier,
        &DigestVerifier,
        &DigestVerifier,
        &DigestVerifier,
        &DigestVerifier,
    );
    assert!(report.accepted(), "{:?}", report.issues);
    assert!(report.operationally_closed);
}

#[test]
fn active_observer_quarantine_can_be_forbidden_by_policy() {
    let mut bundle = accepted_incident_closure_bundle();
    bundle.closure_policy.require_no_active_observer_quarantines = true;
    bundle.closure_policy.policy_sha256 =
        calibration_publication_incident_closure_policy_sha256(&bundle.closure_policy);
    bundle.plan.closure_policy_sha256 = bundle.closure_policy.policy_sha256.clone();
    bundle.plan.plan_sha256 = calibration_publication_incident_closure_plan_sha256(&bundle.plan);
    for statement in bundle
        .authorization_set
        .recovery_authority_statements
        .iter_mut()
        .chain(bundle.authorization_set.recovered_witness_statements.iter_mut())
    {
        statement.plan_sha256 = bundle.plan.plan_sha256.clone();
        statement.signature_hex = hex(&digest(&bundle.plan.canonical_bytes()));
        statement.envelope_sha256 =
            calibration_signed_publication_incident_closure_statement_sha256(statement);
    }
    bundle.authorization_set.plan_sha256 = bundle.plan.plan_sha256.clone();
    bundle.authorization_set.set_sha256 =
        calibration_publication_incident_closure_authorization_set_sha256(
            &bundle.authorization_set,
        );
    bundle.bundle_sha256 = calibration_publication_incident_closure_bundle_sha256(&bundle);
    let report = verify_calibration_publication_incident_closure_bundle(
        &bundle,
        &DigestVerifier,
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
            == CalibrationPublicationIncidentClosureIssueCode::ActiveObserverQuarantineForbidden
    }));
}

#[test]
fn closure_quarantine_history_must_extend_recovery_history() {
    let mut bundle = accepted_incident_closure_bundle();
    bundle.closure_quarantine_ledger.decisions.clear();
    bundle.closure_quarantine_ledger.ledger_sha256 =
        calibration_publication_quarantine_ledger_sha256(&bundle.closure_quarantine_ledger);
    bundle.plan.closure_quarantine_ledger_sha256 =
        bundle.closure_quarantine_ledger.ledger_sha256.clone();
    bundle.plan.plan_sha256 = calibration_publication_incident_closure_plan_sha256(&bundle.plan);
    bundle.authorization_set.plan_sha256 = bundle.plan.plan_sha256.clone();
    bundle.authorization_set.set_sha256 =
        calibration_publication_incident_closure_authorization_set_sha256(
            &bundle.authorization_set,
        );
    bundle.bundle_sha256 = calibration_publication_incident_closure_bundle_sha256(&bundle);
    let report = audit_calibration_publication_incident_closure_bundle(&bundle);
    assert!(!report.valid());
    assert!(report.issues.iter().any(|issue| {
        issue.code
            == CalibrationPublicationIncidentClosureIssueCode::QuarantineLedgerNotAnExtension
    }));
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}
