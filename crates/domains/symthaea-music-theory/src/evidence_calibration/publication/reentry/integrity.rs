// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;

use crate::evidence_calibration::{
    CalibrationPublicationCheckpointWitnessVerifier,
    CalibrationPublicationGossipVerifier,
    CalibrationPublicationQuarantineVerifier,
    CalibrationPublicationRecoveryAuthorityRotationVerifier,
    CalibrationPublicationRecoveryVerifier,
    CalibrationPublicationWitnessPolicyRotationVerifier,
    active_calibration_publication_recovery_authority_epoch,
    audit_calibration_publication_continuity_bundle,
    audit_calibration_publication_incident_response_package,
    audit_calibration_publication_recovery_authority_ledger,
    calibration_publication_catalog_lineage_checkpoint_sha256s,
    calibration_publication_catalog_lineage_terminal,
    verify_calibration_publication_continuity_bundle,
    verify_calibration_publication_incident_response_package,
    verify_calibration_publication_recovery_authority_ledger,
};
use crate::evidence_calibration::sha256::{Sha256, hex as sha256_hex};

use super::model::*;

#[allow(clippy::too_many_arguments)]
pub fn build_calibration_publication_post_recovery_certification<HV, RV, GV, QV, RecV, ARV>(
    incident_response_package: crate::evidence_calibration::CalibrationPublicationIncidentResponsePackage,
    continuity_bundle: crate::evidence_calibration::CalibrationPublicationContinuityBundle,
    recovery_authority_ledger: crate::evidence_calibration::CalibrationPublicationRecoveryAuthorityLedger,
    minimum_additional_catalog_events: u64,
    certification_epoch: u64,
    head_verifier: &HV,
    rotation_verifier: &RV,
    gossip_verifier: &GV,
    quarantine_verifier: &QV,
    recovery_verifier: &RecV,
    authority_rotation_verifier: &ARV,
) -> Result<CalibrationPublicationPostRecoveryCertification, CalibrationPublicationPostRecoveryError>
where
    HV: CalibrationPublicationCheckpointWitnessVerifier,
    RV: CalibrationPublicationWitnessPolicyRotationVerifier,
    GV: CalibrationPublicationGossipVerifier,
    QV: CalibrationPublicationQuarantineVerifier,
    RecV: CalibrationPublicationRecoveryVerifier,
    ARV: CalibrationPublicationRecoveryAuthorityRotationVerifier,
{
    let (_, selected_checkpoint) = calibration_publication_catalog_lineage_terminal(
        &incident_response_package.recovery_bundle.plan.selected_lineage,
    );
    let head_checkpoint = &continuity_bundle.head_bundle.checkpoint;
    let selected_recovery_checkpoint_sha256 = selected_checkpoint.checkpoint_sha256.clone();
    let head_checkpoint_sha256 = head_checkpoint.checkpoint_sha256.clone();
    let additional_catalog_events = head_checkpoint
        .event_count
        .saturating_sub(selected_checkpoint.event_count);
    let active_recovery_authority_epoch_sha256 =
        active_calibration_publication_recovery_authority_epoch(
            &recovery_authority_ledger,
            head_checkpoint,
        )
        .map(|epoch| epoch.epoch_sha256.clone())
        .unwrap_or_default();
    let mut certification = CalibrationPublicationPostRecoveryCertification {
        certification_version: CALIBRATION_PUBLICATION_POST_RECOVERY_CERTIFICATION_VERSION.into(),
        incident_response_package,
        continuity_bundle,
        recovery_authority_ledger,
        active_recovery_authority_epoch_sha256,
        selected_recovery_checkpoint_sha256,
        head_checkpoint_sha256,
        minimum_additional_catalog_events,
        additional_catalog_events,
        certification_epoch,
        limitations: calibration_publication_post_recovery_required_limitations(),
        certification_sha256: String::new(),
    };
    certification.certification_sha256 =
        calibration_publication_post_recovery_certification_sha256(&certification);
    let report = verify_calibration_publication_post_recovery_certification(
        &certification,
        head_verifier,
        rotation_verifier,
        gossip_verifier,
        quarantine_verifier,
        recovery_verifier,
        authority_rotation_verifier,
    );
    if !report.accepted() {
        return Err(CalibrationPublicationPostRecoveryError::InvalidCertification {
            issues: report.issues.len(),
        });
    }
    Ok(certification)
}

pub fn audit_calibration_publication_post_recovery_certification(
    certification: &CalibrationPublicationPostRecoveryCertification,
) -> CalibrationPublicationPostRecoveryAuditReport {
    audit_inner::<NeverHeadVerifier, NeverRotationVerifier, NeverGossipVerifier, NeverQuarantineVerifier, NeverRecoveryVerifier, NeverAuthorityVerifier>(
        certification,
        None,
        None,
        None,
        None,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn verify_calibration_publication_post_recovery_certification<HV, RV, GV, QV, RecV, ARV>(
    certification: &CalibrationPublicationPostRecoveryCertification,
    head_verifier: &HV,
    rotation_verifier: &RV,
    gossip_verifier: &GV,
    quarantine_verifier: &QV,
    recovery_verifier: &RecV,
    authority_rotation_verifier: &ARV,
) -> CalibrationPublicationPostRecoveryAuditReport
where
    HV: CalibrationPublicationCheckpointWitnessVerifier,
    RV: CalibrationPublicationWitnessPolicyRotationVerifier,
    GV: CalibrationPublicationGossipVerifier,
    QV: CalibrationPublicationQuarantineVerifier,
    RecV: CalibrationPublicationRecoveryVerifier,
    ARV: CalibrationPublicationRecoveryAuthorityRotationVerifier,
{
    audit_inner(
        certification,
        Some(head_verifier),
        Some(rotation_verifier),
        Some(gossip_verifier),
        Some(quarantine_verifier),
        Some(recovery_verifier),
        Some(authority_rotation_verifier),
    )
}

pub fn calibration_publication_post_recovery_required_limitations(
) -> Vec<CalibrationPublicationPostRecoveryLimitation> {
    vec![
        CalibrationPublicationPostRecoveryLimitation::SelectedBranchOnly,
        CalibrationPublicationPostRecoveryLimitation::FreshWitnessingIsNotGlobalConsensus,
        CalibrationPublicationPostRecoveryLimitation::ExternalVerifiersDefineAuthentication,
        CalibrationPublicationPostRecoveryLimitation::WithheldForksMayRemainUnknown,
        CalibrationPublicationPostRecoveryLimitation::OperationalReentryIsNotIncidentClosure,
    ]
}

pub fn calibration_publication_post_recovery_certification_sha256(
    certification: &CalibrationPublicationPostRecoveryCertification,
) -> String {
    let mut hash = Sha256::new();
    hash.update(POST_RECOVERY_CERTIFICATION_DOMAIN);
    hash_field(&mut hash, &certification.certification_version);
    hash_field(
        &mut hash,
        &certification.incident_response_package.package_sha256,
    );
    hash_field(&mut hash, &certification.continuity_bundle.bundle_sha256);
    hash_field(
        &mut hash,
        &certification.recovery_authority_ledger.ledger_sha256,
    );
    hash_field(
        &mut hash,
        &certification.active_recovery_authority_epoch_sha256,
    );
    hash_field(
        &mut hash,
        &certification.selected_recovery_checkpoint_sha256,
    );
    hash_field(&mut hash, &certification.head_checkpoint_sha256);
    hash.update(&certification.minimum_additional_catalog_events.to_le_bytes());
    hash.update(&certification.additional_catalog_events.to_le_bytes());
    hash.update(&certification.certification_epoch.to_le_bytes());
    let limitations = certification
        .limitations
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    hash.update(&(limitations.len() as u64).to_le_bytes());
    for limitation in limitations {
        hash.update(&[limitation_code(limitation)]);
    }
    sha256_hex(&hash.finalize())
}

#[allow(clippy::too_many_arguments)]
fn audit_inner<HV, RV, GV, QV, RecV, ARV>(
    certification: &CalibrationPublicationPostRecoveryCertification,
    head_verifier: Option<&HV>,
    rotation_verifier: Option<&RV>,
    gossip_verifier: Option<&GV>,
    quarantine_verifier: Option<&QV>,
    recovery_verifier: Option<&RecV>,
    authority_rotation_verifier: Option<&ARV>,
) -> CalibrationPublicationPostRecoveryAuditReport
where
    HV: CalibrationPublicationCheckpointWitnessVerifier,
    RV: CalibrationPublicationWitnessPolicyRotationVerifier,
    GV: CalibrationPublicationGossipVerifier,
    QV: CalibrationPublicationQuarantineVerifier,
    RecV: CalibrationPublicationRecoveryVerifier,
    ARV: CalibrationPublicationRecoveryAuthorityRotationVerifier,
{
    let mut report = CalibrationPublicationPostRecoveryAuditReport {
        audit_version: CALIBRATION_PUBLICATION_POST_RECOVERY_CERTIFICATION_AUDIT_VERSION.into(),
        structurally_valid: true,
        incident_response_authorized: false,
        continuity_accepted: false,
        recovery_authority_rotations_authenticated: false,
        fresh_checkpoint_confirmed: false,
        issues: Vec::new(),
    };
    if certification.certification_version
        != CALIBRATION_PUBLICATION_POST_RECOVERY_CERTIFICATION_VERSION
    {
        post_recovery_issue(
            &mut report,
            CalibrationPublicationPostRecoveryIssueCode::CertificationVersionMismatch,
            "post-recovery certification version mismatch",
        );
    }

    let response_report = match (
        head_verifier,
        rotation_verifier,
        gossip_verifier,
        quarantine_verifier,
        recovery_verifier,
    ) {
        (Some(head), Some(rotation), Some(gossip), Some(quarantine), Some(recovery)) => {
            verify_calibration_publication_incident_response_package(
                &certification.incident_response_package,
                head,
                rotation,
                gossip,
                quarantine,
                recovery,
            )
        }
        _ => audit_calibration_publication_incident_response_package(
            &certification.incident_response_package,
        ),
    };
    if !response_report.valid() {
        post_recovery_issue(
            &mut report,
            CalibrationPublicationPostRecoveryIssueCode::IncidentResponseInvalid,
            "incident-response package is invalid",
        );
    }
    report.incident_response_authorized = response_report.accepted();
    if recovery_verifier.is_some() && !report.incident_response_authorized {
        post_recovery_issue(
            &mut report,
            CalibrationPublicationPostRecoveryIssueCode::IncidentResponseAuthenticationFailed,
            "incident-response package failed authenticated recovery audit",
        );
    }

    let continuity_report = match (head_verifier, rotation_verifier, gossip_verifier) {
        (Some(head), Some(rotation), Some(gossip)) => {
            verify_calibration_publication_continuity_bundle(
                &certification.continuity_bundle,
                head,
                rotation,
                gossip,
            )
        }
        _ => audit_calibration_publication_continuity_bundle(
            &certification.continuity_bundle,
        ),
    };
    if !continuity_report.valid() {
        post_recovery_issue(
            &mut report,
            CalibrationPublicationPostRecoveryIssueCode::ContinuityInvalid,
            "post-recovery continuity bundle is invalid",
        );
    }
    report.continuity_accepted = continuity_report.accepted();
    if continuity_report.conflict_detected {
        post_recovery_issue(
            &mut report,
            CalibrationPublicationPostRecoveryIssueCode::ContinuityConflictDetected,
            "post-recovery continuity contains unresolved conflict evidence",
        );
    }
    if head_verifier.is_some() && !report.continuity_accepted {
        post_recovery_issue(
            &mut report,
            CalibrationPublicationPostRecoveryIssueCode::ContinuityAuthenticationFailed,
            "post-recovery continuity failed authenticated acceptance",
        );
    }

    let authority_report = match authority_rotation_verifier {
        Some(verifier) => verify_calibration_publication_recovery_authority_ledger(
            &certification.recovery_authority_ledger,
            verifier,
        ),
        None => audit_calibration_publication_recovery_authority_ledger(
            &certification.recovery_authority_ledger,
        ),
    };
    if !authority_report.valid() {
        post_recovery_issue(
            &mut report,
            CalibrationPublicationPostRecoveryIssueCode::RecoveryAuthorityLedgerInvalid,
            "recovery-authority ledger is invalid",
        );
    }
    report.recovery_authority_rotations_authenticated = authority_report.accepted();
    if authority_rotation_verifier.is_some()
        && !report.recovery_authority_rotations_authenticated
    {
        post_recovery_issue(
            &mut report,
            CalibrationPublicationPostRecoveryIssueCode::RecoveryAuthorityAuthenticationFailed,
            "recovery-authority rotations failed authentication",
        );
    }

    let recovery_bundle = &certification.incident_response_package.recovery_bundle;
    let recovered_anchor = &certification.incident_response_package.recovered_policy_anchor;
    let (_, selected_checkpoint) = calibration_publication_catalog_lineage_terminal(
        &recovery_bundle.plan.selected_lineage,
    );
    let head_checkpoint = &certification.continuity_bundle.head_bundle.checkpoint;
    if selected_checkpoint.catalog_id != head_checkpoint.catalog_id
        || selected_checkpoint.authority_id != head_checkpoint.authority_id
        || certification.recovery_authority_ledger.catalog_id != head_checkpoint.catalog_id
        || certification.recovery_authority_ledger.authority_id != head_checkpoint.authority_id
    {
        post_recovery_issue(
            &mut report,
            CalibrationPublicationPostRecoveryIssueCode::IdentityMismatch,
            "incident response, continuity, and authority ledger identities differ",
        );
    }
    if certification.selected_recovery_checkpoint_sha256
        != selected_checkpoint.checkpoint_sha256
    {
        post_recovery_issue(
            &mut report,
            CalibrationPublicationPostRecoveryIssueCode::SelectedCheckpointMismatch,
            "stored selected-recovery checkpoint identity is incorrect",
        );
    }
    if certification.head_checkpoint_sha256 != head_checkpoint.checkpoint_sha256 {
        post_recovery_issue(
            &mut report,
            CalibrationPublicationPostRecoveryIssueCode::HeadCheckpointMismatch,
            "stored post-recovery head checkpoint identity is incorrect",
        );
    }
    let additional = head_checkpoint
        .event_count
        .saturating_sub(selected_checkpoint.event_count);
    if certification.additional_catalog_events != additional {
        post_recovery_issue(
            &mut report,
            CalibrationPublicationPostRecoveryIssueCode::AdditionalEventCountMismatch,
            "stored post-recovery event count does not match the checkpoints",
        );
    }
    if head_checkpoint.checkpoint_sha256 == selected_checkpoint.checkpoint_sha256
        || head_checkpoint.issued_epoch <= recovery_bundle.plan.recovery_epoch
        || head_checkpoint.event_count < selected_checkpoint.event_count
    {
        post_recovery_issue(
            &mut report,
            CalibrationPublicationPostRecoveryIssueCode::FreshCheckpointMissing,
            "post-recovery checkpoint is not fresh relative to the selected recovery checkpoint",
        );
    }
    if additional < certification.minimum_additional_catalog_events {
        post_recovery_issue(
            &mut report,
            CalibrationPublicationPostRecoveryIssueCode::InsufficientAdditionalCatalogEvents,
            "post-recovery catalog did not advance by the required number of events",
        );
    }
    if certification.certification_epoch < head_checkpoint.issued_epoch {
        post_recovery_issue(
            &mut report,
            CalibrationPublicationPostRecoveryIssueCode::CertificationBeforeHead,
            "certification epoch predates the post-recovery checkpoint",
        );
    }

    let recovered_genesis = recovered_anchor.recovered_policy_ledger.epochs.first();
    let continuity_genesis = certification.continuity_bundle.witness_policy_ledger.epochs.first();
    if recovered_genesis.map(|epoch| epoch.epoch_sha256.as_str())
        != continuity_genesis.map(|epoch| epoch.epoch_sha256.as_str())
    {
        post_recovery_issue(
            &mut report,
            CalibrationPublicationPostRecoveryIssueCode::RecoveredPolicyAnchorMismatch,
            "post-recovery continuity does not begin with the recovered witness policy",
        );
    }
    if certification.continuity_bundle.policy_lineage.anchor_checkpoint.checkpoint_sha256
        != selected_checkpoint.checkpoint_sha256
    {
        post_recovery_issue(
            &mut report,
            CalibrationPublicationPostRecoveryIssueCode::RecoveryLineageAnchorMismatch,
            "post-recovery policy lineage is not anchored at the selected recovery checkpoint",
        );
    }

    let active_at_selected = active_calibration_publication_recovery_authority_epoch(
        &certification.recovery_authority_ledger,
        selected_checkpoint,
    );
    if active_at_selected.map(|epoch| epoch.policy.policy_sha256.as_str())
        != Some(recovery_bundle.recovery_authority_policy.policy_sha256.as_str())
    {
        post_recovery_issue(
            &mut report,
            CalibrationPublicationPostRecoveryIssueCode::ActiveRecoveryAuthorityMismatch,
            "recovery bundle was not authorized by the authority policy active at the selected checkpoint",
        );
    }
    let active_at_head = active_calibration_publication_recovery_authority_epoch(
        &certification.recovery_authority_ledger,
        head_checkpoint,
    );
    match active_at_head {
        None => post_recovery_issue(
            &mut report,
            CalibrationPublicationPostRecoveryIssueCode::ActiveRecoveryAuthorityMissing,
            "no recovery-authority policy is active at the post-recovery head",
        ),
        Some(epoch) if epoch.epoch_sha256 != certification.active_recovery_authority_epoch_sha256 => {
            post_recovery_issue(
                &mut report,
                CalibrationPublicationPostRecoveryIssueCode::ActiveRecoveryAuthorityMismatch,
                "stored active recovery-authority epoch does not match the post-recovery head",
            );
        }
        Some(_) => {}
    }
    let lineage_checkpoints = calibration_publication_catalog_lineage_checkpoint_sha256s(
        &certification.continuity_bundle.policy_lineage,
    )
    .into_iter()
    .collect::<BTreeSet<_>>();
    for epoch in &certification.recovery_authority_ledger.epochs {
        if epoch.activation_checkpoint.event_count >= selected_checkpoint.event_count
            && !lineage_checkpoints.contains(&epoch.activation_checkpoint.checkpoint_sha256)
        {
            post_recovery_issue(
                &mut report,
                CalibrationPublicationPostRecoveryIssueCode::RecoveryAuthorityActivationMissingFromLineage,
                "a post-recovery authority-policy activation is absent from the exact catalog lineage",
            );
        }
    }

    report.fresh_checkpoint_confirmed = !report.issues.iter().any(|issue| matches!(
        issue.code,
        CalibrationPublicationPostRecoveryIssueCode::FreshCheckpointMissing
            | CalibrationPublicationPostRecoveryIssueCode::InsufficientAdditionalCatalogEvents
            | CalibrationPublicationPostRecoveryIssueCode::CertificationBeforeHead
            | CalibrationPublicationPostRecoveryIssueCode::SelectedCheckpointMismatch
            | CalibrationPublicationPostRecoveryIssueCode::HeadCheckpointMismatch
            | CalibrationPublicationPostRecoveryIssueCode::AdditionalEventCountMismatch
    ));

    let limitations = certification
        .limitations
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if limitations.len() != certification.limitations.len() {
        post_recovery_issue(
            &mut report,
            CalibrationPublicationPostRecoveryIssueCode::DuplicateLimitation,
            "post-recovery certification contains duplicate limitations",
        );
    }
    for required in calibration_publication_post_recovery_required_limitations() {
        if !limitations.contains(&required) {
            post_recovery_issue(
                &mut report,
                CalibrationPublicationPostRecoveryIssueCode::MissingLimitation,
                "post-recovery certification omits a mandatory limitation",
            );
        }
    }
    if certification.certification_sha256
        != calibration_publication_post_recovery_certification_sha256(certification)
    {
        post_recovery_issue(
            &mut report,
            CalibrationPublicationPostRecoveryIssueCode::CertificationSha256Mismatch,
            "post-recovery certification SHA-256 mismatch",
        );
    }
    report.structurally_valid = report.issues.iter().all(|issue| matches!(
        issue.code,
        CalibrationPublicationPostRecoveryIssueCode::IncidentResponseAuthenticationFailed
            | CalibrationPublicationPostRecoveryIssueCode::ContinuityAuthenticationFailed
            | CalibrationPublicationPostRecoveryIssueCode::RecoveryAuthorityAuthenticationFailed
    ));
    report
}

fn hash_field(hash: &mut Sha256, value: &str) {
    hash.update(&(value.len() as u64).to_le_bytes());
    hash.update(value.as_bytes());
}

const fn limitation_code(value: CalibrationPublicationPostRecoveryLimitation) -> u8 {
    match value {
        CalibrationPublicationPostRecoveryLimitation::SelectedBranchOnly => 0,
        CalibrationPublicationPostRecoveryLimitation::FreshWitnessingIsNotGlobalConsensus => 1,
        CalibrationPublicationPostRecoveryLimitation::ExternalVerifiersDefineAuthentication => 2,
        CalibrationPublicationPostRecoveryLimitation::WithheldForksMayRemainUnknown => 3,
        CalibrationPublicationPostRecoveryLimitation::OperationalReentryIsNotIncidentClosure => 4,
    }
}

fn post_recovery_issue(
    report: &mut CalibrationPublicationPostRecoveryAuditReport,
    code: CalibrationPublicationPostRecoveryIssueCode,
    detail: impl Into<String>,
) {
    report.issues.push(CalibrationPublicationPostRecoveryIssue {
        code,
        detail: detail.into(),
    });
}

struct NeverHeadVerifier;
impl CalibrationPublicationCheckpointWitnessVerifier for NeverHeadVerifier {
    type Error = &'static str;
    fn verify(
        &self,
        _: &[u8],
        _: &crate::evidence_calibration::CalibrationSignerIdentity,
        _: &[u8],
    ) -> Result<(), Self::Error> { Err("not attempted") }
}
struct NeverRotationVerifier;
impl CalibrationPublicationWitnessPolicyRotationVerifier for NeverRotationVerifier {
    type Error = &'static str;
    fn verify(
        &self,
        _: &[u8],
        _: &crate::evidence_calibration::CalibrationSignerIdentity,
        _: &[u8],
    ) -> Result<(), Self::Error> { Err("not attempted") }
}
struct NeverGossipVerifier;
impl CalibrationPublicationGossipVerifier for NeverGossipVerifier {
    type Error = &'static str;
    fn verify(
        &self,
        _: &[u8],
        _: &crate::evidence_calibration::CalibrationSignerIdentity,
        _: &[u8],
    ) -> Result<(), Self::Error> { Err("not attempted") }
}
struct NeverQuarantineVerifier;
impl CalibrationPublicationQuarantineVerifier for NeverQuarantineVerifier {
    type Error = &'static str;
    fn verify(
        &self,
        _: &[u8],
        _: &crate::evidence_calibration::CalibrationSignerIdentity,
        _: &[u8],
    ) -> Result<(), Self::Error> { Err("not attempted") }
}
struct NeverRecoveryVerifier;
impl CalibrationPublicationRecoveryVerifier for NeverRecoveryVerifier {
    type Error = &'static str;
    fn verify(
        &self,
        _: &[u8],
        _: &crate::evidence_calibration::CalibrationSignerIdentity,
        _: &[u8],
    ) -> Result<(), Self::Error> { Err("not attempted") }
}
struct NeverAuthorityVerifier;
impl CalibrationPublicationRecoveryAuthorityRotationVerifier for NeverAuthorityVerifier {
    type Error = &'static str;
    fn verify(
        &self,
        _: &[u8],
        _: &crate::evidence_calibration::CalibrationSignerIdentity,
        _: &[u8],
    ) -> Result<(), Self::Error> { Err("not attempted") }
}
