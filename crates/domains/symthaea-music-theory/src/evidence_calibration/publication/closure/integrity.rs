// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;

use crate::evidence_calibration::{
    CalibrationPublicationCheckpointWitnessPolicy,
    CalibrationPublicationCheckpointWitnessVerifier,
    CalibrationPublicationPostRecoveryCertification,
    CalibrationPublicationGossipVerifier,
    CalibrationPublicationQuarantineVerifier,
    CalibrationPublicationRecoveryAuthorityRotationVerifier,
    CalibrationPublicationRecoveryVerifier,
    CalibrationPublicationWitnessPolicyRotationVerifier,
    CalibrationSignerIdentity,
    active_calibration_publication_quarantines,
    active_calibration_publication_recovery_authority_epoch,
    active_calibration_publication_witness_policy_epoch,
    audit_calibration_publication_post_recovery_certification,
    verify_calibration_publication_post_recovery_certification,
    audit_calibration_publication_quarantine_ledger,
    verify_calibration_publication_quarantine_ledger,
};
use crate::evidence_calibration::sha256::{Sha256, hex as sha256_hex};

use super::model::*;

pub fn build_calibration_publication_incident_closure_policy(
    minimum_additional_catalog_events: u64,
    minimum_recovery_authority_signers: Option<u64>,
    minimum_recovered_witness_signers: Option<u64>,
    require_no_active_witness_quarantines: bool,
    require_no_active_observer_quarantines: bool,
) -> Result<CalibrationPublicationIncidentClosurePolicy, CalibrationPublicationIncidentClosureError> {
    if minimum_recovery_authority_signers == Some(0)
        || minimum_recovered_witness_signers == Some(0)
    {
        return Err(CalibrationPublicationIncidentClosureError::InvalidPolicy);
    }
    let mut policy = CalibrationPublicationIncidentClosurePolicy {
        policy_version: CALIBRATION_PUBLICATION_INCIDENT_CLOSURE_POLICY_VERSION.into(),
        minimum_additional_catalog_events,
        minimum_recovery_authority_signers,
        minimum_recovered_witness_signers,
        require_no_active_witness_quarantines,
        require_no_active_observer_quarantines,
        policy_sha256: String::new(),
    };
    policy.policy_sha256 = calibration_publication_incident_closure_policy_sha256(&policy);
    Ok(policy)
}

pub fn plan_calibration_publication_incident_closure(
    certification: &CalibrationPublicationPostRecoveryCertification,
    closure_quarantine_ledger: &crate::evidence_calibration::CalibrationPublicationQuarantineLedger,
    policy: &CalibrationPublicationIncidentClosurePolicy,
    closure_epoch: u64,
) -> Result<CalibrationPublicationIncidentClosurePlan, CalibrationPublicationIncidentClosureError> {
    if !audit_calibration_publication_post_recovery_certification(certification).valid() {
        return Err(CalibrationPublicationIncidentClosureError::InvalidCertification);
    }
    if !audit_calibration_publication_quarantine_ledger(closure_quarantine_ledger).valid() {
        return Err(CalibrationPublicationIncidentClosureError::InvalidQuarantine);
    }
    if !closure_policy_structurally_valid(policy) {
        return Err(CalibrationPublicationIncidentClosureError::InvalidPolicy);
    }
    let head = &certification.continuity_bundle.head_bundle.checkpoint;
    let active_authority = active_calibration_publication_recovery_authority_epoch(
        &certification.recovery_authority_ledger,
        head,
    )
    .ok_or(CalibrationPublicationIncidentClosureError::InvalidCertification)?;
    let active_witness = active_calibration_publication_witness_policy_epoch(
        &certification.continuity_bundle.witness_policy_ledger,
        head,
    )
    .ok_or(CalibrationPublicationIncidentClosureError::InvalidCertification)?;
    let active = active_calibration_publication_quarantines(closure_quarantine_ledger, closure_epoch);
    let mut witness_keys = active
        .iter()
        .filter(|entry| entry.scope.includes_witness())
        .map(|entry| entry.key_id.clone())
        .collect::<Vec<_>>();
    let mut observer_keys = active
        .iter()
        .filter(|entry| entry.scope.includes_observer())
        .map(|entry| entry.key_id.clone())
        .collect::<Vec<_>>();
    witness_keys.sort();
    witness_keys.dedup();
    observer_keys.sort();
    observer_keys.dedup();
    let mut plan = CalibrationPublicationIncidentClosurePlan {
        plan_version: CALIBRATION_PUBLICATION_INCIDENT_CLOSURE_PLAN_VERSION.into(),
        catalog_id: head.catalog_id.clone(),
        authority_id: head.authority_id.clone(),
        post_recovery_certification_sha256: certification.certification_sha256.clone(),
        closure_policy_sha256: policy.policy_sha256.clone(),
        closure_quarantine_ledger_sha256: closure_quarantine_ledger.ledger_sha256.clone(),
        active_recovery_authority_epoch_sha256: active_authority.epoch_sha256.clone(),
        active_witness_policy_epoch_sha256: active_witness.epoch_sha256.clone(),
        active_witness_quarantine_key_ids: witness_keys,
        active_observer_quarantine_key_ids: observer_keys,
        closure_epoch,
        disposition: CalibrationPublicationIncidentClosureDisposition::OperationallyClosed,
        plan_sha256: String::new(),
    };
    plan.plan_sha256 = calibration_publication_incident_closure_plan_sha256(&plan);
    Ok(plan)
}

pub fn build_calibration_signed_publication_incident_closure_statement(
    plan: &CalibrationPublicationIncidentClosurePlan,
    role: CalibrationPublicationIncidentClosureSignerRole,
    signer: CalibrationSignerIdentity,
    signature: &[u8],
) -> CalibrationSignedPublicationIncidentClosureStatement {
    let mut statement = CalibrationSignedPublicationIncidentClosureStatement {
        envelope_version: CALIBRATION_SIGNED_PUBLICATION_INCIDENT_CLOSURE_STATEMENT_VERSION.into(),
        plan_sha256: plan.plan_sha256.clone(),
        role,
        signer,
        signature_hex: encode_hex(signature),
        envelope_sha256: String::new(),
    };
    statement.envelope_sha256 =
        calibration_signed_publication_incident_closure_statement_sha256(&statement);
    statement
}

pub fn build_calibration_publication_incident_closure_authorization_set(
    plan: &CalibrationPublicationIncidentClosurePlan,
    recovery_authority_policy: &crate::evidence_calibration::CalibrationPublicationRecoveryAuthorityPolicy,
    recovered_witness_policy: &CalibrationPublicationCheckpointWitnessPolicy,
    recovery_authority_statements: Vec<CalibrationSignedPublicationIncidentClosureStatement>,
    recovered_witness_statements: Vec<CalibrationSignedPublicationIncidentClosureStatement>,
) -> CalibrationPublicationIncidentClosureAuthorizationSet {
    let mut set = CalibrationPublicationIncidentClosureAuthorizationSet {
        set_version: CALIBRATION_PUBLICATION_INCIDENT_CLOSURE_AUTHORIZATION_SET_VERSION.into(),
        plan_sha256: plan.plan_sha256.clone(),
        recovery_authority_policy_sha256: recovery_authority_policy.policy_sha256.clone(),
        recovered_witness_policy_sha256: recovered_witness_policy.policy_sha256.clone(),
        recovery_authority_statements,
        recovered_witness_statements,
        set_sha256: String::new(),
    };
    set.set_sha256 = calibration_publication_incident_closure_authorization_set_sha256(&set);
    set
}

#[allow(clippy::too_many_arguments)]
pub fn build_calibration_publication_incident_closure_bundle<HV, RV, GV, QV, RecV, ARV, CV>(
    post_recovery_certification: CalibrationPublicationPostRecoveryCertification,
    closure_quarantine_ledger: crate::evidence_calibration::CalibrationPublicationQuarantineLedger,
    closure_policy: CalibrationPublicationIncidentClosurePolicy,
    plan: CalibrationPublicationIncidentClosurePlan,
    authorization_set: CalibrationPublicationIncidentClosureAuthorizationSet,
    head_verifier: &HV,
    rotation_verifier: &RV,
    gossip_verifier: &GV,
    quarantine_verifier: &QV,
    recovery_verifier: &RecV,
    authority_rotation_verifier: &ARV,
    closure_verifier: &CV,
) -> Result<CalibrationPublicationIncidentClosureBundle, CalibrationPublicationIncidentClosureError>
where
    HV: CalibrationPublicationCheckpointWitnessVerifier,
    RV: CalibrationPublicationWitnessPolicyRotationVerifier,
    GV: CalibrationPublicationGossipVerifier,
    QV: CalibrationPublicationQuarantineVerifier,
    RecV: CalibrationPublicationRecoveryVerifier,
    ARV: CalibrationPublicationRecoveryAuthorityRotationVerifier,
    CV: CalibrationPublicationIncidentClosureVerifier,
{
    let mut bundle = CalibrationPublicationIncidentClosureBundle {
        bundle_version: CALIBRATION_PUBLICATION_INCIDENT_CLOSURE_BUNDLE_VERSION.into(),
        post_recovery_certification,
        closure_quarantine_ledger,
        closure_policy,
        plan,
        authorization_set,
        limitations: calibration_publication_incident_closure_required_limitations(),
        bundle_sha256: String::new(),
    };
    bundle.bundle_sha256 = calibration_publication_incident_closure_bundle_sha256(&bundle);
    let report = verify_calibration_publication_incident_closure_bundle(
        &bundle,
        head_verifier,
        rotation_verifier,
        gossip_verifier,
        quarantine_verifier,
        recovery_verifier,
        authority_rotation_verifier,
        closure_verifier,
    );
    if !report.accepted() {
        return Err(CalibrationPublicationIncidentClosureError::InvalidClosure {
            issues: report.issues.len(),
        });
    }
    Ok(bundle)
}

pub fn audit_calibration_publication_incident_closure_bundle(
    bundle: &CalibrationPublicationIncidentClosureBundle,
) -> CalibrationPublicationIncidentClosureAuditReport {
    audit_inner::<NeverHeadVerifier, NeverRotationVerifier, NeverGossipVerifier, NeverQuarantineVerifier, NeverRecoveryVerifier, NeverAuthorityVerifier, NeverClosureVerifier>(
        bundle, None, None, None, None, None, None, None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn verify_calibration_publication_incident_closure_bundle<HV, RV, GV, QV, RecV, ARV, CV>(
    bundle: &CalibrationPublicationIncidentClosureBundle,
    head_verifier: &HV,
    rotation_verifier: &RV,
    gossip_verifier: &GV,
    quarantine_verifier: &QV,
    recovery_verifier: &RecV,
    authority_rotation_verifier: &ARV,
    closure_verifier: &CV,
) -> CalibrationPublicationIncidentClosureAuditReport
where
    HV: CalibrationPublicationCheckpointWitnessVerifier,
    RV: CalibrationPublicationWitnessPolicyRotationVerifier,
    GV: CalibrationPublicationGossipVerifier,
    QV: CalibrationPublicationQuarantineVerifier,
    RecV: CalibrationPublicationRecoveryVerifier,
    ARV: CalibrationPublicationRecoveryAuthorityRotationVerifier,
    CV: CalibrationPublicationIncidentClosureVerifier,
{
    audit_inner(
        bundle,
        Some(head_verifier),
        Some(rotation_verifier),
        Some(gossip_verifier),
        Some(quarantine_verifier),
        Some(recovery_verifier),
        Some(authority_rotation_verifier),
        Some(closure_verifier),
    )
}

pub fn calibration_publication_incident_closure_required_limitations(
) -> Vec<CalibrationPublicationIncidentClosureLimitation> {
    vec![
        CalibrationPublicationIncidentClosureLimitation::OperationalClosureIsNotUniversalResolution,
        CalibrationPublicationIncidentClosureLimitation::SelectedBranchOnly,
        CalibrationPublicationIncidentClosureLimitation::WithheldForksMayRemainUnknown,
        CalibrationPublicationIncidentClosureLimitation::ExternalVerifiersDefineAuthentication,
        CalibrationPublicationIncidentClosureLimitation::QuarantineDoesNotEstablishFault,
        CalibrationPublicationIncidentClosureLimitation::FutureEvidenceMayReopenTheIncident,
    ]
}

pub fn calibration_publication_incident_closure_policy_sha256(
    policy: &CalibrationPublicationIncidentClosurePolicy,
) -> String {
    let mut hash = Sha256::new();
    hash.update(CLOSURE_POLICY_DOMAIN);
    hash_field(&mut hash, &policy.policy_version);
    hash.update(&policy.minimum_additional_catalog_events.to_le_bytes());
    hash_optional_u64(&mut hash, policy.minimum_recovery_authority_signers);
    hash_optional_u64(&mut hash, policy.minimum_recovered_witness_signers);
    hash.update(&[u8::from(policy.require_no_active_witness_quarantines)]);
    hash.update(&[u8::from(policy.require_no_active_observer_quarantines)]);
    sha256_hex(&hash.finalize())
}

pub fn calibration_publication_incident_closure_plan_sha256(
    plan: &CalibrationPublicationIncidentClosurePlan,
) -> String {
    let mut hash = Sha256::new();
    hash.update(&plan.canonical_bytes());
    sha256_hex(&hash.finalize())
}

pub fn calibration_signed_publication_incident_closure_statement_sha256(
    statement: &CalibrationSignedPublicationIncidentClosureStatement,
) -> String {
    let mut hash = Sha256::new();
    hash.update(CLOSURE_ENVELOPE_DOMAIN);
    hash_field(&mut hash, &statement.envelope_version);
    hash_field(&mut hash, &statement.plan_sha256);
    hash.update(&[closure_role_code(statement.role)]);
    hash_field(&mut hash, &statement.signer.key_id);
    hash_field(&mut hash, &statement.signer.algorithm);
    hash_optional_field(&mut hash, statement.signer.issuer.as_deref());
    hash_field(&mut hash, &statement.signature_hex);
    sha256_hex(&hash.finalize())
}

pub fn calibration_publication_incident_closure_authorization_set_sha256(
    set: &CalibrationPublicationIncidentClosureAuthorizationSet,
) -> String {
    let mut hash = Sha256::new();
    hash.update(CLOSURE_SET_DOMAIN);
    hash_field(&mut hash, &set.set_version);
    hash_field(&mut hash, &set.plan_sha256);
    hash_field(&mut hash, &set.recovery_authority_policy_sha256);
    hash_field(&mut hash, &set.recovered_witness_policy_sha256);
    hash_statements(&mut hash, &set.recovery_authority_statements);
    hash_statements(&mut hash, &set.recovered_witness_statements);
    sha256_hex(&hash.finalize())
}

pub fn calibration_publication_incident_closure_bundle_sha256(
    bundle: &CalibrationPublicationIncidentClosureBundle,
) -> String {
    let mut hash = Sha256::new();
    hash.update(CLOSURE_BUNDLE_DOMAIN);
    hash_field(&mut hash, &bundle.bundle_version);
    hash_field(
        &mut hash,
        &bundle.post_recovery_certification.certification_sha256,
    );
    hash_field(&mut hash, &bundle.closure_quarantine_ledger.ledger_sha256);
    hash_field(&mut hash, &bundle.closure_policy.policy_sha256);
    hash_field(&mut hash, &bundle.plan.plan_sha256);
    hash_field(&mut hash, &bundle.authorization_set.set_sha256);
    let limitations = bundle.limitations.iter().copied().collect::<BTreeSet<_>>();
    hash.update(&(limitations.len() as u64).to_le_bytes());
    for limitation in limitations {
        hash.update(&[closure_limitation_code(limitation)]);
    }
    sha256_hex(&hash.finalize())
}

#[allow(clippy::too_many_arguments)]
fn audit_inner<HV, RV, GV, QV, RecV, ARV, CV>(
    bundle: &CalibrationPublicationIncidentClosureBundle,
    head_verifier: Option<&HV>,
    rotation_verifier: Option<&RV>,
    gossip_verifier: Option<&GV>,
    quarantine_verifier: Option<&QV>,
    recovery_verifier: Option<&RecV>,
    authority_rotation_verifier: Option<&ARV>,
    closure_verifier: Option<&CV>,
) -> CalibrationPublicationIncidentClosureAuditReport
where
    HV: CalibrationPublicationCheckpointWitnessVerifier,
    RV: CalibrationPublicationWitnessPolicyRotationVerifier,
    GV: CalibrationPublicationGossipVerifier,
    QV: CalibrationPublicationQuarantineVerifier,
    RecV: CalibrationPublicationRecoveryVerifier,
    ARV: CalibrationPublicationRecoveryAuthorityRotationVerifier,
    CV: CalibrationPublicationIncidentClosureVerifier,
{
    let mut report = CalibrationPublicationIncidentClosureAuditReport {
        audit_version: CALIBRATION_PUBLICATION_INCIDENT_CLOSURE_AUDIT_VERSION.into(),
        structurally_valid: true,
        post_recovery_accepted: false,
        quarantine_authenticated: false,
        recovery_authority_authenticated: false,
        recovered_witnesses_authenticated: false,
        operationally_closed: false,
        issues: Vec::new(),
    };
    if bundle.bundle_version != CALIBRATION_PUBLICATION_INCIDENT_CLOSURE_BUNDLE_VERSION {
        closure_issue(&mut report, CalibrationPublicationIncidentClosureIssueCode::BundleVersionMismatch, None, "incident-closure bundle version mismatch");
    }
    let certification_report = match (
        head_verifier, rotation_verifier, gossip_verifier, quarantine_verifier,
        recovery_verifier, authority_rotation_verifier,
    ) {
        (Some(head), Some(rotation), Some(gossip), Some(quarantine), Some(recovery), Some(authority)) => {
            verify_calibration_publication_post_recovery_certification(
                &bundle.post_recovery_certification,
                head, rotation, gossip, quarantine, recovery, authority,
            )
        }
        _ => audit_calibration_publication_post_recovery_certification(
            &bundle.post_recovery_certification,
        ),
    };
    if !certification_report.valid() {
        closure_issue(&mut report, CalibrationPublicationIncidentClosureIssueCode::PostRecoveryCertificationInvalid, None, "post-recovery certification is invalid");
    }
    report.post_recovery_accepted = certification_report.accepted();
    if closure_verifier.is_some() && !report.post_recovery_accepted {
        closure_issue(&mut report, CalibrationPublicationIncidentClosureIssueCode::PostRecoveryCertificationAuthenticationFailed, None, "post-recovery certification failed authenticated acceptance");
    }

    let quarantine_report = match quarantine_verifier {
        Some(verifier) => verify_calibration_publication_quarantine_ledger(
            &bundle.closure_quarantine_ledger,
            verifier,
        ),
        None => audit_calibration_publication_quarantine_ledger(
            &bundle.closure_quarantine_ledger,
        ),
    };
    if !quarantine_report.valid() {
        closure_issue(&mut report, CalibrationPublicationIncidentClosureIssueCode::QuarantineLedgerInvalid, None, "closure quarantine ledger is invalid");
    }
    report.quarantine_authenticated = quarantine_report.accepted();
    if quarantine_verifier.is_some() && !report.quarantine_authenticated {
        closure_issue(&mut report, CalibrationPublicationIncidentClosureIssueCode::QuarantineAuthenticationFailed, None, "closure quarantine ledger failed authentication");
    }
    let recovery_quarantine = &bundle
        .post_recovery_certification
        .incident_response_package
        .recovery_bundle
        .quarantine_ledger;
    if !quarantine_is_prefix(recovery_quarantine, &bundle.closure_quarantine_ledger) {
        closure_issue(&mut report, CalibrationPublicationIncidentClosureIssueCode::QuarantineLedgerNotAnExtension, None, "closure quarantine ledger is not an append-only extension of recovery containment");
    }
    if !closure_policy_structurally_valid(&bundle.closure_policy) {
        closure_issue(&mut report, CalibrationPublicationIncidentClosureIssueCode::ClosurePolicyInvalid, None, "incident-closure policy is invalid");
    }

    let plan = &bundle.plan;
    let head = &bundle.post_recovery_certification.continuity_bundle.head_bundle.checkpoint;
    if plan.plan_version != CALIBRATION_PUBLICATION_INCIDENT_CLOSURE_PLAN_VERSION {
        closure_issue(&mut report, CalibrationPublicationIncidentClosureIssueCode::PlanVersionMismatch, None, "incident-closure plan version mismatch");
    }
    if plan.catalog_id != head.catalog_id || plan.authority_id != head.authority_id {
        closure_issue(&mut report, CalibrationPublicationIncidentClosureIssueCode::PlanIdentityMismatch, None, "incident-closure plan identity mismatch");
    }
    if plan.post_recovery_certification_sha256
        != bundle.post_recovery_certification.certification_sha256
    {
        closure_issue(&mut report, CalibrationPublicationIncidentClosureIssueCode::PlanCertificationMismatch, None, "incident-closure plan does not bind the certification");
    }
    if plan.closure_policy_sha256 != bundle.closure_policy.policy_sha256 {
        closure_issue(&mut report, CalibrationPublicationIncidentClosureIssueCode::PlanPolicyMismatch, None, "incident-closure plan does not bind the policy");
    }
    if plan.closure_quarantine_ledger_sha256 != bundle.closure_quarantine_ledger.ledger_sha256 {
        closure_issue(&mut report, CalibrationPublicationIncidentClosureIssueCode::PlanQuarantineMismatch, None, "incident-closure plan does not bind the quarantine ledger");
    }
    if plan.closure_epoch < bundle.post_recovery_certification.certification_epoch {
        closure_issue(&mut report, CalibrationPublicationIncidentClosureIssueCode::ClosureBeforeCertification, None, "incident closure predates post-recovery certification");
    }
    if bundle.post_recovery_certification.additional_catalog_events
        < bundle.closure_policy.minimum_additional_catalog_events
    {
        closure_issue(&mut report, CalibrationPublicationIncidentClosureIssueCode::AdditionalCatalogEventsBelowPolicy, None, "post-recovery catalog advance is below closure policy");
    }

    let active_authority = active_calibration_publication_recovery_authority_epoch(
        &bundle.post_recovery_certification.recovery_authority_ledger,
        head,
    );
    let active_witness = active_calibration_publication_witness_policy_epoch(
        &bundle.post_recovery_certification.continuity_bundle.witness_policy_ledger,
        head,
    );
    let recovery_authority_policy = active_authority.map(|epoch| &epoch.policy);
    let recovered_witness_policy = active_witness.map(|epoch| &epoch.policy);
    if active_authority.map(|epoch| epoch.epoch_sha256.as_str())
        != Some(plan.active_recovery_authority_epoch_sha256.as_str())
    {
        closure_issue(&mut report, CalibrationPublicationIncidentClosureIssueCode::ActiveAuthorityEpochMismatch, None, "incident-closure plan names the wrong active recovery-authority epoch");
    }
    if active_witness.map(|epoch| epoch.epoch_sha256.as_str())
        != Some(plan.active_witness_policy_epoch_sha256.as_str())
    {
        closure_issue(&mut report, CalibrationPublicationIncidentClosureIssueCode::ActiveWitnessEpochMismatch, None, "incident-closure plan names the wrong active witness-policy epoch");
    }

    let active = active_calibration_publication_quarantines(
        &bundle.closure_quarantine_ledger,
        plan.closure_epoch,
    );
    let (witness_keys, observer_keys) = quarantine_key_summaries(&active);
    if plan.active_witness_quarantine_key_ids != witness_keys
        || plan.active_observer_quarantine_key_ids != observer_keys
    {
        closure_issue(&mut report, CalibrationPublicationIncidentClosureIssueCode::ActiveQuarantineSummaryMismatch, None, "incident-closure quarantine summary is incorrect");
    }
    if bundle.closure_policy.require_no_active_witness_quarantines && !witness_keys.is_empty() {
        closure_issue(&mut report, CalibrationPublicationIncidentClosureIssueCode::ActiveWitnessQuarantineForbidden, None, "closure policy forbids active witness quarantines");
    }
    if bundle.closure_policy.require_no_active_observer_quarantines && !observer_keys.is_empty() {
        closure_issue(&mut report, CalibrationPublicationIncidentClosureIssueCode::ActiveObserverQuarantineForbidden, None, "closure policy forbids active observer quarantines");
    }

    audit_authorization_set(
        &mut report,
        bundle,
        recovery_authority_policy,
        recovered_witness_policy,
        closure_verifier,
    );
    if plan.plan_sha256 != calibration_publication_incident_closure_plan_sha256(plan) {
        closure_issue(&mut report, CalibrationPublicationIncidentClosureIssueCode::PlanSha256Mismatch, None, "incident-closure plan SHA-256 mismatch");
    }
    let limitations = bundle.limitations.iter().copied().collect::<BTreeSet<_>>();
    if limitations.len() != bundle.limitations.len() {
        closure_issue(&mut report, CalibrationPublicationIncidentClosureIssueCode::DuplicateLimitation, None, "incident-closure bundle contains duplicate limitations");
    }
    for required in calibration_publication_incident_closure_required_limitations() {
        if !limitations.contains(&required) {
            closure_issue(&mut report, CalibrationPublicationIncidentClosureIssueCode::MissingLimitation, None, "incident-closure bundle omits a mandatory limitation");
        }
    }
    if bundle.bundle_sha256 != calibration_publication_incident_closure_bundle_sha256(bundle) {
        closure_issue(&mut report, CalibrationPublicationIncidentClosureIssueCode::BundleSha256Mismatch, None, "incident-closure bundle SHA-256 mismatch");
    }
    report.operationally_closed = !report.issues.iter().any(|issue| matches!(
        issue.code,
        CalibrationPublicationIncidentClosureIssueCode::PostRecoveryCertificationInvalid
            | CalibrationPublicationIncidentClosureIssueCode::QuarantineLedgerInvalid
            | CalibrationPublicationIncidentClosureIssueCode::QuarantineLedgerNotAnExtension
            | CalibrationPublicationIncidentClosureIssueCode::ClosurePolicyInvalid
            | CalibrationPublicationIncidentClosureIssueCode::ClosureBeforeCertification
            | CalibrationPublicationIncidentClosureIssueCode::AdditionalCatalogEventsBelowPolicy
            | CalibrationPublicationIncidentClosureIssueCode::ActiveWitnessQuarantineForbidden
            | CalibrationPublicationIncidentClosureIssueCode::ActiveObserverQuarantineForbidden
    ));
    report.structurally_valid = report.issues.iter().all(|issue| matches!(
        issue.code,
        CalibrationPublicationIncidentClosureIssueCode::PostRecoveryCertificationAuthenticationFailed
            | CalibrationPublicationIncidentClosureIssueCode::QuarantineAuthenticationFailed
            | CalibrationPublicationIncidentClosureIssueCode::RecoveryAuthoritySignatureRejected
            | CalibrationPublicationIncidentClosureIssueCode::RecoveredWitnessSignatureRejected
            | CalibrationPublicationIncidentClosureIssueCode::RecoveryAuthorityThresholdNotMet
            | CalibrationPublicationIncidentClosureIssueCode::RecoveredWitnessThresholdNotMet
    ));
    report
}

fn audit_authorization_set<CV: CalibrationPublicationIncidentClosureVerifier>(
    report: &mut CalibrationPublicationIncidentClosureAuditReport,
    bundle: &CalibrationPublicationIncidentClosureBundle,
    recovery_authority_policy: Option<&crate::evidence_calibration::CalibrationPublicationRecoveryAuthorityPolicy>,
    recovered_witness_policy: Option<&CalibrationPublicationCheckpointWitnessPolicy>,
    verifier: Option<&CV>,
) {
    let set = &bundle.authorization_set;
    if set.set_version != CALIBRATION_PUBLICATION_INCIDENT_CLOSURE_AUTHORIZATION_SET_VERSION {
        closure_issue(report, CalibrationPublicationIncidentClosureIssueCode::AuthorizationSetVersionMismatch, None, "incident-closure authorization-set version mismatch");
    }
    if set.plan_sha256 != bundle.plan.plan_sha256 {
        closure_issue(report, CalibrationPublicationIncidentClosureIssueCode::AuthorizationPlanMismatch, None, "incident-closure authorization set does not bind the plan");
    }
    if recovery_authority_policy.map(|policy| policy.policy_sha256.as_str())
        != Some(set.recovery_authority_policy_sha256.as_str())
        || recovered_witness_policy.map(|policy| policy.policy_sha256.as_str())
            != Some(set.recovered_witness_policy_sha256.as_str())
    {
        closure_issue(report, CalibrationPublicationIncidentClosureIssueCode::AuthorizationPolicyMismatch, None, "incident-closure authorization set binds the wrong active policies");
    }
    audit_statements(report, &bundle.plan, CalibrationPublicationIncidentClosureSignerRole::RecoveryAuthority, &set.recovery_authority_statements);
    audit_statements(report, &bundle.plan, CalibrationPublicationIncidentClosureSignerRole::RecoveredWitness, &set.recovered_witness_statements);
    if let (Some(verifier), Some(authority_policy), Some(witness_policy)) = (
        verifier, recovery_authority_policy, recovered_witness_policy,
    ) {
        let authority_count = authenticate_statements(
            report,
            &bundle.plan,
            &set.recovery_authority_statements,
            &authority_policy.accepted_key_ids,
            CalibrationPublicationIncidentClosureIssueCode::UnacceptedRecoveryAuthority,
            CalibrationPublicationIncidentClosureIssueCode::RecoveryAuthoritySignatureRejected,
            verifier,
        );
        let witness_count = authenticate_statements(
            report,
            &bundle.plan,
            &set.recovered_witness_statements,
            &witness_policy.accepted_key_ids,
            CalibrationPublicationIncidentClosureIssueCode::UnacceptedRecoveredWitness,
            CalibrationPublicationIncidentClosureIssueCode::RecoveredWitnessSignatureRejected,
            verifier,
        );
        let authority_threshold = bundle
            .closure_policy
            .minimum_recovery_authority_signers
            .unwrap_or(authority_policy.minimum_distinct_authorizers)
            .max(authority_policy.minimum_distinct_authorizers);
        let witness_threshold = bundle
            .closure_policy
            .minimum_recovered_witness_signers
            .unwrap_or(witness_policy.minimum_distinct_witnesses)
            .max(witness_policy.minimum_distinct_witnesses);
        report.recovery_authority_authenticated = authority_count >= authority_threshold;
        report.recovered_witnesses_authenticated = witness_count >= witness_threshold;
        if !report.recovery_authority_authenticated {
            closure_issue(report, CalibrationPublicationIncidentClosureIssueCode::RecoveryAuthorityThresholdNotMet, None, "recovery-authority closure threshold not met");
        }
        if !report.recovered_witnesses_authenticated {
            closure_issue(report, CalibrationPublicationIncidentClosureIssueCode::RecoveredWitnessThresholdNotMet, None, "recovered-witness closure threshold not met");
        }
    }
    if set.set_sha256 != calibration_publication_incident_closure_authorization_set_sha256(set) {
        closure_issue(report, CalibrationPublicationIncidentClosureIssueCode::AuthorizationSetSha256Mismatch, None, "incident-closure authorization-set SHA-256 mismatch");
    }
}

fn audit_statements(
    report: &mut CalibrationPublicationIncidentClosureAuditReport,
    plan: &CalibrationPublicationIncidentClosurePlan,
    expected_role: CalibrationPublicationIncidentClosureSignerRole,
    statements: &[CalibrationSignedPublicationIncidentClosureStatement],
) {
    for statement in statements {
        if statement.envelope_version != CALIBRATION_SIGNED_PUBLICATION_INCIDENT_CLOSURE_STATEMENT_VERSION {
            closure_issue(report, CalibrationPublicationIncidentClosureIssueCode::StatementVersionMismatch, Some(statement.signer.key_id.clone()), "incident-closure statement version mismatch");
        }
        if statement.plan_sha256 != plan.plan_sha256 {
            closure_issue(report, CalibrationPublicationIncidentClosureIssueCode::AuthorizationPlanMismatch, Some(statement.signer.key_id.clone()), "incident-closure statement does not bind the plan");
        }
        if statement.role != expected_role {
            closure_issue(report, CalibrationPublicationIncidentClosureIssueCode::StatementRoleMismatch, Some(statement.signer.key_id.clone()), "incident-closure statement role mismatch");
        }
        if statement.signer.key_id.is_empty() || statement.signer.algorithm.is_empty() {
            closure_issue(report, CalibrationPublicationIncidentClosureIssueCode::EmptySignerIdentity, Some(statement.signer.key_id.clone()), "incident-closure signer identity is empty");
        }
        if decode_hex(&statement.signature_hex).is_none() {
            closure_issue(report, CalibrationPublicationIncidentClosureIssueCode::InvalidSignatureHex, Some(statement.signer.key_id.clone()), "incident-closure signature is invalid hex");
        }
        if statement.envelope_sha256 != calibration_signed_publication_incident_closure_statement_sha256(statement) {
            closure_issue(report, CalibrationPublicationIncidentClosureIssueCode::StatementSha256Mismatch, Some(statement.signer.key_id.clone()), "incident-closure statement SHA-256 mismatch");
        }
    }
}

fn authenticate_statements<CV: CalibrationPublicationIncidentClosureVerifier>(
    report: &mut CalibrationPublicationIncidentClosureAuditReport,
    plan: &CalibrationPublicationIncidentClosurePlan,
    statements: &[CalibrationSignedPublicationIncidentClosureStatement],
    accepted_keys: &[String],
    unaccepted_code: CalibrationPublicationIncidentClosureIssueCode,
    rejected_code: CalibrationPublicationIncidentClosureIssueCode,
    verifier: &CV,
) -> u64 {
    let accepted = accepted_keys.iter().cloned().collect::<BTreeSet<_>>();
    let mut seen = BTreeSet::new();
    let mut authenticated = 0u64;
    for statement in statements {
        let key_id = statement.signer.key_id.clone();
        if !seen.insert(key_id.clone()) {
            closure_issue(report, CalibrationPublicationIncidentClosureIssueCode::DuplicateSigner, Some(key_id), "duplicate incident-closure signer");
            continue;
        }
        if !accepted.contains(&key_id) {
            closure_issue(report, unaccepted_code, Some(key_id), "incident-closure signer is not accepted by the active policy");
            continue;
        }
        let Some(signature) = decode_hex(&statement.signature_hex) else { continue };
        if verifier.verify(&plan.canonical_bytes(), &statement.signer, &signature).is_err() {
            closure_issue(report, rejected_code, Some(key_id), "external verifier rejected incident-closure signature");
            continue;
        }
        authenticated += 1;
    }
    authenticated
}

fn quarantine_is_prefix(
    prior: &crate::evidence_calibration::CalibrationPublicationQuarantineLedger,
    later: &crate::evidence_calibration::CalibrationPublicationQuarantineLedger,
) -> bool {
    prior.catalog_id == later.catalog_id
        && prior.authority_id == later.authority_id
        && prior.policy == later.policy
        && prior.decisions.len() <= later.decisions.len()
        && prior.decisions.iter().zip(later.decisions.iter()).all(|(left, right)| left == right)
}

fn quarantine_key_summaries(
    active: &[crate::evidence_calibration::CalibrationPublicationActiveQuarantine],
) -> (Vec<String>, Vec<String>) {
    let mut witness = active.iter()
        .filter(|entry| entry.scope.includes_witness())
        .map(|entry| entry.key_id.clone()).collect::<Vec<_>>();
    let mut observer = active.iter()
        .filter(|entry| entry.scope.includes_observer())
        .map(|entry| entry.key_id.clone()).collect::<Vec<_>>();
    witness.sort(); witness.dedup(); observer.sort(); observer.dedup();
    (witness, observer)
}

fn closure_policy_structurally_valid(policy: &CalibrationPublicationIncidentClosurePolicy) -> bool {
    policy.policy_version == CALIBRATION_PUBLICATION_INCIDENT_CLOSURE_POLICY_VERSION
        && policy.minimum_recovery_authority_signers != Some(0)
        && policy.minimum_recovered_witness_signers != Some(0)
        && policy.policy_sha256 == calibration_publication_incident_closure_policy_sha256(policy)
}

fn hash_statements(hash: &mut Sha256, statements: &[CalibrationSignedPublicationIncidentClosureStatement]) {
    let mut identities = statements.iter().map(|statement| statement.envelope_sha256.clone()).collect::<Vec<_>>();
    identities.sort();
    hash.update(&(identities.len() as u64).to_le_bytes());
    for identity in identities { hash_field(hash, &identity); }
}

fn hash_field(hash: &mut Sha256, value: &str) {
    hash.update(&(value.len() as u64).to_le_bytes());
    hash.update(value.as_bytes());
}
fn hash_optional_field(hash: &mut Sha256, value: Option<&str>) {
    match value { None => hash.update(&[0]), Some(value) => { hash.update(&[1]); hash_field(hash, value); } }
}
fn hash_optional_u64(hash: &mut Sha256, value: Option<u64>) {
    match value { None => hash.update(&[0]), Some(value) => { hash.update(&[1]); hash.update(&value.to_le_bytes()); } }
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes { output.push(HEX[(byte >> 4) as usize] as char); output.push(HEX[(byte & 0x0f) as usize] as char); }
    output
}
fn decode_hex(value: &str) -> Option<Vec<u8>> {
    if value.len() % 2 != 0 { return None; }
    let mut output = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) { output.push((hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?); }
    Some(output)
}
fn hex_nibble(value: u8) -> Option<u8> { match value { b'0'..=b'9' => Some(value-b'0'), b'a'..=b'f' => Some(value-b'a'+10), b'A'..=b'F' => Some(value-b'A'+10), _ => None } }

const fn closure_role_code(value: CalibrationPublicationIncidentClosureSignerRole) -> u8 {
    match value { CalibrationPublicationIncidentClosureSignerRole::RecoveryAuthority => 0, CalibrationPublicationIncidentClosureSignerRole::RecoveredWitness => 1 }
}
const fn closure_limitation_code(value: CalibrationPublicationIncidentClosureLimitation) -> u8 {
    match value {
        CalibrationPublicationIncidentClosureLimitation::OperationalClosureIsNotUniversalResolution => 0,
        CalibrationPublicationIncidentClosureLimitation::SelectedBranchOnly => 1,
        CalibrationPublicationIncidentClosureLimitation::WithheldForksMayRemainUnknown => 2,
        CalibrationPublicationIncidentClosureLimitation::ExternalVerifiersDefineAuthentication => 3,
        CalibrationPublicationIncidentClosureLimitation::QuarantineDoesNotEstablishFault => 4,
        CalibrationPublicationIncidentClosureLimitation::FutureEvidenceMayReopenTheIncident => 5,
    }
}

fn closure_issue(
    report: &mut CalibrationPublicationIncidentClosureAuditReport,
    code: CalibrationPublicationIncidentClosureIssueCode,
    signer_key_id: Option<String>,
    detail: impl Into<String>,
) {
    report.issues.push(CalibrationPublicationIncidentClosureIssue { code, signer_key_id, detail: detail.into() });
}

struct NeverHeadVerifier;
impl CalibrationPublicationCheckpointWitnessVerifier for NeverHeadVerifier { type Error=&'static str; fn verify(&self,_:&[u8],_:&CalibrationSignerIdentity,_:&[u8])->Result<(),Self::Error>{Err("not attempted")} }
struct NeverRotationVerifier;
impl CalibrationPublicationWitnessPolicyRotationVerifier for NeverRotationVerifier { type Error=&'static str; fn verify(&self,_:&[u8],_:&CalibrationSignerIdentity,_:&[u8])->Result<(),Self::Error>{Err("not attempted")} }
struct NeverGossipVerifier;
impl CalibrationPublicationGossipVerifier for NeverGossipVerifier { type Error=&'static str; fn verify(&self,_:&[u8],_:&CalibrationSignerIdentity,_:&[u8])->Result<(),Self::Error>{Err("not attempted")} }
struct NeverQuarantineVerifier;
impl CalibrationPublicationQuarantineVerifier for NeverQuarantineVerifier { type Error=&'static str; fn verify(&self,_:&[u8],_:&CalibrationSignerIdentity,_:&[u8])->Result<(),Self::Error>{Err("not attempted")} }
struct NeverRecoveryVerifier;
impl CalibrationPublicationRecoveryVerifier for NeverRecoveryVerifier { type Error=&'static str; fn verify(&self,_:&[u8],_:&CalibrationSignerIdentity,_:&[u8])->Result<(),Self::Error>{Err("not attempted")} }
struct NeverAuthorityVerifier;
impl CalibrationPublicationRecoveryAuthorityRotationVerifier for NeverAuthorityVerifier { type Error=&'static str; fn verify(&self,_:&[u8],_:&CalibrationSignerIdentity,_:&[u8])->Result<(),Self::Error>{Err("not attempted")} }
struct NeverClosureVerifier;
impl CalibrationPublicationIncidentClosureVerifier for NeverClosureVerifier { type Error=&'static str; fn verify(&self,_:&[u8],_:&CalibrationSignerIdentity,_:&[u8])->Result<(),Self::Error>{Err("not attempted")} }
