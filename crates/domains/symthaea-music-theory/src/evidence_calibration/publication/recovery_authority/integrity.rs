// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;

use crate::evidence_calibration::{
    CALIBRATION_PUBLICATION_CATALOG_CHECKPOINT_VERSION,
    CalibrationPublicationCatalogCheckpoint,
    CalibrationPublicationRecoveryAuthorityPolicy,
    CalibrationSignerIdentity,
    calibration_publication_catalog_checkpoint_sha256,
    calibration_publication_recovery_authority_policy_sha256,
};
use crate::evidence_calibration::sha256::{Sha256, hex as sha256_hex};

use super::model::*;

pub fn build_calibration_publication_recovery_authority_genesis(
    policy: CalibrationPublicationRecoveryAuthorityPolicy,
    checkpoint: CalibrationPublicationCatalogCheckpoint,
    issued_epoch: u64,
) -> Result<CalibrationPublicationRecoveryAuthorityLedger, CalibrationPublicationRecoveryAuthorityError> {
    if !policy_structurally_valid(&policy) || !checkpoint_structurally_valid(&checkpoint) {
        return Err(CalibrationPublicationRecoveryAuthorityError::InvalidCheckpoint);
    }
    if issued_epoch < checkpoint.issued_epoch {
        return Err(CalibrationPublicationRecoveryAuthorityError::InvalidCheckpoint);
    }
    let mut epoch = CalibrationPublicationRecoveryAuthorityEpoch {
        epoch_version: CALIBRATION_PUBLICATION_RECOVERY_AUTHORITY_EPOCH_VERSION.into(),
        ordinal: 0,
        policy,
        activation_checkpoint: checkpoint.clone(),
        previous_epoch_sha256: None,
        issued_epoch,
        epoch_sha256: String::new(),
    };
    epoch.epoch_sha256 = calibration_publication_recovery_authority_epoch_sha256(&epoch);
    let mut ledger = CalibrationPublicationRecoveryAuthorityLedger {
        ledger_version: CALIBRATION_PUBLICATION_RECOVERY_AUTHORITY_LEDGER_VERSION.into(),
        catalog_id: checkpoint.catalog_id,
        authority_id: checkpoint.authority_id,
        epochs: vec![epoch],
        rotations: Vec::new(),
        ledger_sha256: String::new(),
    };
    ledger.ledger_sha256 = calibration_publication_recovery_authority_ledger_sha256(&ledger);
    let audit = audit_calibration_publication_recovery_authority_ledger(&ledger);
    if !audit.valid() {
        return Err(CalibrationPublicationRecoveryAuthorityError::InvalidLedger {
            issues: audit.issues.len(),
        });
    }
    Ok(ledger)
}

pub fn plan_calibration_publication_recovery_authority_rotation(
    ledger: &CalibrationPublicationRecoveryAuthorityLedger,
    activation_checkpoint: CalibrationPublicationCatalogCheckpoint,
    incoming_policy: CalibrationPublicationRecoveryAuthorityPolicy,
    issued_epoch: u64,
) -> Result<(
    CalibrationPublicationRecoveryAuthorityEpoch,
    CalibrationPublicationRecoveryAuthorityRotationPayload,
), CalibrationPublicationRecoveryAuthorityError> {
    let audit = audit_calibration_publication_recovery_authority_ledger(ledger);
    if !audit.valid() {
        return Err(CalibrationPublicationRecoveryAuthorityError::InvalidLedger {
            issues: audit.issues.len(),
        });
    }
    if !checkpoint_structurally_valid(&activation_checkpoint) || !policy_structurally_valid(&incoming_policy) {
        return Err(CalibrationPublicationRecoveryAuthorityError::InvalidCheckpoint);
    }
    if activation_checkpoint.catalog_id != ledger.catalog_id
        || activation_checkpoint.authority_id != ledger.authority_id
    {
        return Err(CalibrationPublicationRecoveryAuthorityError::CatalogIdentityMismatch);
    }
    let previous = ledger.epochs.last().ok_or(
        CalibrationPublicationRecoveryAuthorityError::InvalidLedger { issues: 1 },
    )?;
    if activation_checkpoint.event_count <= previous.activation_checkpoint.event_count {
        return Err(CalibrationPublicationRecoveryAuthorityError::ActivationCountRegression);
    }
    if issued_epoch < activation_checkpoint.issued_epoch || issued_epoch < previous.issued_epoch {
        return Err(CalibrationPublicationRecoveryAuthorityError::ActivationCountRegression);
    }
    if incoming_policy.policy_sha256 == previous.policy.policy_sha256 {
        return Err(CalibrationPublicationRecoveryAuthorityError::PolicyUnchanged);
    }
    let mut epoch = CalibrationPublicationRecoveryAuthorityEpoch {
        epoch_version: CALIBRATION_PUBLICATION_RECOVERY_AUTHORITY_EPOCH_VERSION.into(),
        ordinal: previous.ordinal + 1,
        policy: incoming_policy,
        activation_checkpoint: activation_checkpoint.clone(),
        previous_epoch_sha256: Some(previous.epoch_sha256.clone()),
        issued_epoch,
        epoch_sha256: String::new(),
    };
    epoch.epoch_sha256 = calibration_publication_recovery_authority_epoch_sha256(&epoch);
    let mut payload = CalibrationPublicationRecoveryAuthorityRotationPayload {
        payload_version: CALIBRATION_PUBLICATION_RECOVERY_AUTHORITY_ROTATION_PAYLOAD_VERSION.into(),
        catalog_id: ledger.catalog_id.clone(),
        authority_id: ledger.authority_id.clone(),
        rotation_ordinal: epoch.ordinal,
        from_epoch_sha256: previous.epoch_sha256.clone(),
        to_epoch_sha256: epoch.epoch_sha256.clone(),
        activation_checkpoint_sha256: activation_checkpoint.checkpoint_sha256,
        issued_epoch,
        payload_sha256: String::new(),
    };
    payload.payload_sha256 = calibration_publication_recovery_authority_rotation_payload_sha256(&payload);
    Ok((epoch, payload))
}

pub fn build_calibration_signed_publication_recovery_authority_rotation(
    payload: CalibrationPublicationRecoveryAuthorityRotationPayload,
    signer: CalibrationSignerIdentity,
    signature: &[u8],
) -> CalibrationSignedPublicationRecoveryAuthorityRotation {
    let mut envelope = CalibrationSignedPublicationRecoveryAuthorityRotation {
        envelope_version: CALIBRATION_SIGNED_PUBLICATION_RECOVERY_AUTHORITY_ROTATION_VERSION.into(),
        payload,
        signer,
        signature_hex: encode_hex(signature),
        envelope_sha256: String::new(),
    };
    envelope.envelope_sha256 = calibration_signed_publication_recovery_authority_rotation_sha256(&envelope);
    envelope
}

pub fn build_calibration_publication_recovery_authority_rotation_set(
    payload: &CalibrationPublicationRecoveryAuthorityRotationPayload,
    outgoing_policy: &CalibrationPublicationRecoveryAuthorityPolicy,
    incoming_policy: &CalibrationPublicationRecoveryAuthorityPolicy,
    outgoing_statements: Vec<CalibrationSignedPublicationRecoveryAuthorityRotation>,
    incoming_statements: Vec<CalibrationSignedPublicationRecoveryAuthorityRotation>,
) -> CalibrationPublicationRecoveryAuthorityRotationSet {
    let mut set = CalibrationPublicationRecoveryAuthorityRotationSet {
        set_version: CALIBRATION_PUBLICATION_RECOVERY_AUTHORITY_ROTATION_SET_VERSION.into(),
        payload_sha256: payload.payload_sha256.clone(),
        outgoing_policy_sha256: outgoing_policy.policy_sha256.clone(),
        incoming_policy_sha256: incoming_policy.policy_sha256.clone(),
        outgoing_statements,
        incoming_statements,
        set_sha256: String::new(),
    };
    set.set_sha256 = calibration_publication_recovery_authority_rotation_set_sha256(&set);
    set
}

pub fn append_calibration_publication_recovery_authority_rotation<
    V: CalibrationPublicationRecoveryAuthorityRotationVerifier,
>(
    ledger: &mut CalibrationPublicationRecoveryAuthorityLedger,
    epoch: CalibrationPublicationRecoveryAuthorityEpoch,
    rotation: CalibrationPublicationRecoveryAuthorityRotationSet,
    verifier: &V,
) -> Result<(), CalibrationPublicationRecoveryAuthorityError> {
    let existing = audit_calibration_publication_recovery_authority_ledger(ledger);
    if !existing.valid() {
        return Err(CalibrationPublicationRecoveryAuthorityError::InvalidLedger {
            issues: existing.issues.len(),
        });
    }
    let mut candidate = ledger.clone();
    candidate.epochs.push(epoch);
    candidate.rotations.push(rotation);
    candidate.ledger_sha256 = calibration_publication_recovery_authority_ledger_sha256(&candidate);
    let audit = verify_calibration_publication_recovery_authority_ledger(&candidate, verifier);
    if !audit.accepted() {
        return Err(CalibrationPublicationRecoveryAuthorityError::InvalidRotation {
            issues: audit.issues.len(),
        });
    }
    *ledger = candidate;
    Ok(())
}

pub fn active_calibration_publication_recovery_authority_epoch<'a>(
    ledger: &'a CalibrationPublicationRecoveryAuthorityLedger,
    checkpoint: &CalibrationPublicationCatalogCheckpoint,
) -> Option<&'a CalibrationPublicationRecoveryAuthorityEpoch> {
    if ledger.catalog_id != checkpoint.catalog_id || ledger.authority_id != checkpoint.authority_id {
        return None;
    }
    ledger
        .epochs
        .iter()
        .rev()
        .find(|epoch| epoch.activation_checkpoint.event_count <= checkpoint.event_count)
}

pub fn audit_calibration_publication_recovery_authority_ledger(
    ledger: &CalibrationPublicationRecoveryAuthorityLedger,
) -> CalibrationPublicationRecoveryAuthorityAuditReport {
    audit_inner(ledger, None::<&NeverVerifier>)
}

pub fn verify_calibration_publication_recovery_authority_ledger<
    V: CalibrationPublicationRecoveryAuthorityRotationVerifier,
>(
    ledger: &CalibrationPublicationRecoveryAuthorityLedger,
    verifier: &V,
) -> CalibrationPublicationRecoveryAuthorityAuditReport {
    audit_inner(ledger, Some(verifier))
}

pub fn calibration_publication_recovery_authority_epoch_sha256(
    epoch: &CalibrationPublicationRecoveryAuthorityEpoch,
) -> String {
    let mut hash = Sha256::new();
    hash.update(AUTHORITY_EPOCH_DOMAIN);
    hash_field(&mut hash, &epoch.epoch_version);
    hash.update(&epoch.ordinal.to_le_bytes());
    hash_field(&mut hash, &epoch.policy.policy_sha256);
    hash_field(&mut hash, &epoch.activation_checkpoint.checkpoint_sha256);
    hash_optional_field(&mut hash, epoch.previous_epoch_sha256.as_deref());
    hash.update(&epoch.issued_epoch.to_le_bytes());
    sha256_hex(&hash.finalize())
}

pub fn calibration_publication_recovery_authority_rotation_payload_sha256(
    payload: &CalibrationPublicationRecoveryAuthorityRotationPayload,
) -> String {
    let mut hash = Sha256::new();
    hash.update(&payload.canonical_bytes());
    sha256_hex(&hash.finalize())
}

pub fn calibration_signed_publication_recovery_authority_rotation_sha256(
    envelope: &CalibrationSignedPublicationRecoveryAuthorityRotation,
) -> String {
    let mut hash = Sha256::new();
    hash.update(AUTHORITY_ROTATION_ENVELOPE_DOMAIN);
    hash_field(&mut hash, &envelope.envelope_version);
    hash_field(&mut hash, &envelope.payload.payload_sha256);
    hash_field(&mut hash, &envelope.signer.key_id);
    hash_field(&mut hash, &envelope.signer.algorithm);
    hash_optional_field(&mut hash, envelope.signer.issuer.as_deref());
    hash_field(&mut hash, &envelope.signature_hex);
    sha256_hex(&hash.finalize())
}

pub fn calibration_publication_recovery_authority_rotation_set_sha256(
    set: &CalibrationPublicationRecoveryAuthorityRotationSet,
) -> String {
    let mut hash = Sha256::new();
    hash.update(AUTHORITY_ROTATION_SET_DOMAIN);
    hash_field(&mut hash, &set.set_version);
    hash_field(&mut hash, &set.payload_sha256);
    hash_field(&mut hash, &set.outgoing_policy_sha256);
    hash_field(&mut hash, &set.incoming_policy_sha256);
    hash_envelopes(&mut hash, &set.outgoing_statements);
    hash_envelopes(&mut hash, &set.incoming_statements);
    sha256_hex(&hash.finalize())
}

pub fn calibration_publication_recovery_authority_ledger_sha256(
    ledger: &CalibrationPublicationRecoveryAuthorityLedger,
) -> String {
    let mut hash = Sha256::new();
    hash.update(AUTHORITY_LEDGER_DOMAIN);
    hash_field(&mut hash, &ledger.ledger_version);
    hash_field(&mut hash, &ledger.catalog_id);
    hash_field(&mut hash, &ledger.authority_id);
    hash.update(&(ledger.epochs.len() as u64).to_le_bytes());
    for epoch in &ledger.epochs {
        hash_field(&mut hash, &epoch.epoch_sha256);
    }
    hash.update(&(ledger.rotations.len() as u64).to_le_bytes());
    for rotation in &ledger.rotations {
        hash_field(&mut hash, &rotation.set_sha256);
    }
    sha256_hex(&hash.finalize())
}

fn audit_inner<V: CalibrationPublicationRecoveryAuthorityRotationVerifier>(
    ledger: &CalibrationPublicationRecoveryAuthorityLedger,
    verifier: Option<&V>,
) -> CalibrationPublicationRecoveryAuthorityAuditReport {
    let mut report = CalibrationPublicationRecoveryAuthorityAuditReport {
        audit_version: CALIBRATION_PUBLICATION_RECOVERY_AUTHORITY_LEDGER_AUDIT_VERSION.into(),
        structurally_valid: true,
        authenticated_rotations: 0,
        total_rotations: ledger.rotations.len() as u64,
        rotations_authenticated: verifier.is_some() || ledger.rotations.is_empty(),
        issues: Vec::new(),
    };
    if ledger.ledger_version != CALIBRATION_PUBLICATION_RECOVERY_AUTHORITY_LEDGER_VERSION {
        authority_issue(&mut report, CalibrationPublicationRecoveryAuthorityIssueCode::LedgerVersionMismatch, None, None, "recovery-authority ledger version mismatch");
    }
    if ledger.catalog_id.is_empty() || ledger.authority_id.is_empty() {
        authority_issue(&mut report, CalibrationPublicationRecoveryAuthorityIssueCode::EmptyIdentity, None, None, "catalog or authority identity is empty");
    }
    if ledger.epochs.is_empty() {
        authority_issue(&mut report, CalibrationPublicationRecoveryAuthorityIssueCode::MissingGenesis, None, None, "recovery-authority ledger has no genesis epoch");
    }
    if ledger.rotations.len().saturating_add(1) != ledger.epochs.len() && !ledger.epochs.is_empty() {
        authority_issue(&mut report, CalibrationPublicationRecoveryAuthorityIssueCode::EpochRotationCountMismatch, None, None, "epoch and rotation counts are inconsistent");
    }
    for (index, epoch) in ledger.epochs.iter().enumerate() {
        let ordinal = index as u64;
        if epoch.epoch_version != CALIBRATION_PUBLICATION_RECOVERY_AUTHORITY_EPOCH_VERSION {
            authority_issue(&mut report, CalibrationPublicationRecoveryAuthorityIssueCode::EpochVersionMismatch, Some(ordinal), None, "recovery-authority epoch version mismatch");
        }
        if epoch.ordinal != ordinal {
            authority_issue(&mut report, CalibrationPublicationRecoveryAuthorityIssueCode::EpochOrdinalMismatch, Some(ordinal), None, "recovery-authority epoch ordinal mismatch");
        }
        let expected_previous = if index == 0 { None } else { Some(ledger.epochs[index - 1].epoch_sha256.clone()) };
        if epoch.previous_epoch_sha256 != expected_previous {
            authority_issue(&mut report, CalibrationPublicationRecoveryAuthorityIssueCode::PreviousEpochMismatch, Some(ordinal), None, "recovery-authority epoch predecessor mismatch");
        }
        if !policy_structurally_valid(&epoch.policy) {
            authority_issue(&mut report, CalibrationPublicationRecoveryAuthorityIssueCode::PolicyInvalid, Some(ordinal), None, "recovery-authority policy is invalid");
        }
        if !checkpoint_structurally_valid(&epoch.activation_checkpoint) {
            authority_issue(&mut report, CalibrationPublicationRecoveryAuthorityIssueCode::CheckpointInvalid, Some(ordinal), None, "activation checkpoint is invalid");
        }
        if epoch.activation_checkpoint.catalog_id != ledger.catalog_id
            || epoch.activation_checkpoint.authority_id != ledger.authority_id
        {
            authority_issue(&mut report, CalibrationPublicationRecoveryAuthorityIssueCode::CheckpointIdentityMismatch, Some(ordinal), None, "activation checkpoint identity mismatch");
        }
        if epoch.issued_epoch < epoch.activation_checkpoint.issued_epoch {
            authority_issue(&mut report, CalibrationPublicationRecoveryAuthorityIssueCode::EpochBeforeCheckpoint, Some(ordinal), None, "recovery-authority epoch predates its checkpoint");
        }
        if index > 0 {
            let previous = &ledger.epochs[index - 1];
            if epoch.activation_checkpoint.event_count <= previous.activation_checkpoint.event_count
                || epoch.issued_epoch < previous.issued_epoch
            {
                authority_issue(&mut report, CalibrationPublicationRecoveryAuthorityIssueCode::ActivationCountRegression, Some(ordinal), None, "recovery-authority activation does not advance monotonically");
            }
            if epoch.policy.policy_sha256 == previous.policy.policy_sha256 {
                authority_issue(&mut report, CalibrationPublicationRecoveryAuthorityIssueCode::PolicyUnchanged, Some(ordinal), None, "recovery-authority policy is unchanged");
            }
        }
        if epoch.epoch_sha256 != calibration_publication_recovery_authority_epoch_sha256(epoch) {
            authority_issue(&mut report, CalibrationPublicationRecoveryAuthorityIssueCode::EpochSha256Mismatch, Some(ordinal), None, "recovery-authority epoch SHA-256 mismatch");
        }
    }
    for (index, rotation) in ledger.rotations.iter().enumerate() {
        let ordinal = (index + 1) as u64;
        let Some(outgoing_epoch) = ledger.epochs.get(index) else { continue };
        let Some(incoming_epoch) = ledger.epochs.get(index + 1) else { continue };
        if rotation.set_version != CALIBRATION_PUBLICATION_RECOVERY_AUTHORITY_ROTATION_SET_VERSION {
            authority_issue(&mut report, CalibrationPublicationRecoveryAuthorityIssueCode::RotationSetVersionMismatch, Some(ordinal), None, "recovery-authority rotation-set version mismatch");
        }
        if rotation.outgoing_policy_sha256 != outgoing_epoch.policy.policy_sha256
            || rotation.incoming_policy_sha256 != incoming_epoch.policy.policy_sha256
        {
            authority_issue(&mut report, CalibrationPublicationRecoveryAuthorityIssueCode::RotationPolicyMismatch, Some(ordinal), None, "recovery-authority rotation policy mismatch");
        }
        let statements = rotation.outgoing_statements.iter().chain(rotation.incoming_statements.iter());
        for statement in statements {
            audit_statement(&mut report, ordinal, statement, incoming_epoch, rotation);
        }
        if rotation.set_sha256 != calibration_publication_recovery_authority_rotation_set_sha256(rotation) {
            authority_issue(&mut report, CalibrationPublicationRecoveryAuthorityIssueCode::RotationSetSha256Mismatch, Some(ordinal), None, "recovery-authority rotation-set SHA-256 mismatch");
        }
        if let Some(verifier) = verifier {
            let outgoing_count = authenticate_statements(
                &mut report,
                ordinal,
                &rotation.outgoing_statements,
                &outgoing_epoch.policy.accepted_key_ids,
                CalibrationPublicationRecoveryAuthorityIssueCode::UnacceptedOutgoingSigner,
                CalibrationPublicationRecoveryAuthorityIssueCode::OutgoingSignatureRejected,
                verifier,
            );
            let incoming_count = authenticate_statements(
                &mut report,
                ordinal,
                &rotation.incoming_statements,
                &incoming_epoch.policy.accepted_key_ids,
                CalibrationPublicationRecoveryAuthorityIssueCode::UnacceptedIncomingSigner,
                CalibrationPublicationRecoveryAuthorityIssueCode::IncomingSignatureRejected,
                verifier,
            );
            let outgoing_ok = outgoing_count >= outgoing_epoch.policy.minimum_distinct_authorizers;
            let incoming_ok = incoming_count >= incoming_epoch.policy.minimum_distinct_authorizers;
            if !outgoing_ok {
                authority_issue(&mut report, CalibrationPublicationRecoveryAuthorityIssueCode::OutgoingThresholdNotMet, Some(ordinal), None, "outgoing recovery-authority threshold not met");
            }
            if !incoming_ok {
                authority_issue(&mut report, CalibrationPublicationRecoveryAuthorityIssueCode::IncomingThresholdNotMet, Some(ordinal), None, "incoming recovery-authority threshold not met");
            }
            if outgoing_ok && incoming_ok {
                report.authenticated_rotations += 1;
            }
        }
    }
    if verifier.is_some() {
        report.rotations_authenticated = report.authenticated_rotations == report.total_rotations;
    }
    if ledger.ledger_sha256 != calibration_publication_recovery_authority_ledger_sha256(ledger) {
        authority_issue(&mut report, CalibrationPublicationRecoveryAuthorityIssueCode::LedgerSha256Mismatch, None, None, "recovery-authority ledger SHA-256 mismatch");
    }
    report.structurally_valid = report.issues.iter().all(|issue| matches!(
        issue.code,
        CalibrationPublicationRecoveryAuthorityIssueCode::OutgoingSignatureRejected
            | CalibrationPublicationRecoveryAuthorityIssueCode::IncomingSignatureRejected
            | CalibrationPublicationRecoveryAuthorityIssueCode::OutgoingThresholdNotMet
            | CalibrationPublicationRecoveryAuthorityIssueCode::IncomingThresholdNotMet
    ));
    report
}

fn audit_statement(
    report: &mut CalibrationPublicationRecoveryAuthorityAuditReport,
    ordinal: u64,
    statement: &CalibrationSignedPublicationRecoveryAuthorityRotation,
    incoming_epoch: &CalibrationPublicationRecoveryAuthorityEpoch,
    rotation: &CalibrationPublicationRecoveryAuthorityRotationSet,
) {
    if statement.envelope_version != CALIBRATION_SIGNED_PUBLICATION_RECOVERY_AUTHORITY_ROTATION_VERSION {
        authority_issue(report, CalibrationPublicationRecoveryAuthorityIssueCode::RotationEnvelopeVersionMismatch, Some(ordinal), Some(statement.signer.key_id.clone()), "recovery-authority rotation envelope version mismatch");
    }
    let payload = &statement.payload;
    if payload.payload_version != CALIBRATION_PUBLICATION_RECOVERY_AUTHORITY_ROTATION_PAYLOAD_VERSION {
        authority_issue(report, CalibrationPublicationRecoveryAuthorityIssueCode::RotationPayloadVersionMismatch, Some(ordinal), Some(statement.signer.key_id.clone()), "recovery-authority rotation payload version mismatch");
    }
    if payload.rotation_ordinal != ordinal {
        authority_issue(report, CalibrationPublicationRecoveryAuthorityIssueCode::RotationOrdinalMismatch, Some(ordinal), Some(statement.signer.key_id.clone()), "recovery-authority rotation ordinal mismatch");
    }
    if payload.catalog_id != incoming_epoch.activation_checkpoint.catalog_id
        || payload.authority_id != incoming_epoch.activation_checkpoint.authority_id
    {
        authority_issue(report, CalibrationPublicationRecoveryAuthorityIssueCode::RotationIdentityMismatch, Some(ordinal), Some(statement.signer.key_id.clone()), "recovery-authority rotation identity mismatch");
    }
    if payload.to_epoch_sha256 != incoming_epoch.epoch_sha256
        || incoming_epoch.previous_epoch_sha256.as_deref() != Some(payload.from_epoch_sha256.as_str())
    {
        authority_issue(report, CalibrationPublicationRecoveryAuthorityIssueCode::RotationEpochMismatch, Some(ordinal), Some(statement.signer.key_id.clone()), "recovery-authority rotation epoch mismatch");
    }
    if payload.activation_checkpoint_sha256 != incoming_epoch.activation_checkpoint.checkpoint_sha256 {
        authority_issue(report, CalibrationPublicationRecoveryAuthorityIssueCode::RotationCheckpointMismatch, Some(ordinal), Some(statement.signer.key_id.clone()), "recovery-authority rotation checkpoint mismatch");
    }
    if payload.payload_sha256 != rotation.payload_sha256
        || payload.payload_sha256 != calibration_publication_recovery_authority_rotation_payload_sha256(payload)
    {
        authority_issue(report, CalibrationPublicationRecoveryAuthorityIssueCode::RotationPayloadSha256Mismatch, Some(ordinal), Some(statement.signer.key_id.clone()), "recovery-authority rotation payload SHA-256 mismatch");
    }
    if statement.signer.key_id.is_empty() || statement.signer.algorithm.is_empty() {
        authority_issue(report, CalibrationPublicationRecoveryAuthorityIssueCode::EmptySignerIdentity, Some(ordinal), Some(statement.signer.key_id.clone()), "recovery-authority rotation signer is empty");
    }
    if decode_hex(&statement.signature_hex).is_none() {
        authority_issue(report, CalibrationPublicationRecoveryAuthorityIssueCode::InvalidSignatureHex, Some(ordinal), Some(statement.signer.key_id.clone()), "recovery-authority rotation signature is invalid hex");
    }
    if statement.envelope_sha256 != calibration_signed_publication_recovery_authority_rotation_sha256(statement) {
        authority_issue(report, CalibrationPublicationRecoveryAuthorityIssueCode::RotationEnvelopeSha256Mismatch, Some(ordinal), Some(statement.signer.key_id.clone()), "recovery-authority rotation envelope SHA-256 mismatch");
    }
}

fn authenticate_statements<V: CalibrationPublicationRecoveryAuthorityRotationVerifier>(
    report: &mut CalibrationPublicationRecoveryAuthorityAuditReport,
    ordinal: u64,
    statements: &[CalibrationSignedPublicationRecoveryAuthorityRotation],
    accepted: &[String],
    unaccepted_code: CalibrationPublicationRecoveryAuthorityIssueCode,
    rejected_code: CalibrationPublicationRecoveryAuthorityIssueCode,
    verifier: &V,
) -> u64 {
    let accepted = accepted.iter().cloned().collect::<BTreeSet<_>>();
    let mut seen = BTreeSet::new();
    let mut authenticated = 0u64;
    for statement in statements {
        let key_id = statement.signer.key_id.clone();
        if !seen.insert(key_id.clone()) {
            authority_issue(report, CalibrationPublicationRecoveryAuthorityIssueCode::DuplicateRotationSigner, Some(ordinal), Some(key_id), "duplicate recovery-authority rotation signer");
            continue;
        }
        if !accepted.contains(&key_id) {
            authority_issue(report, unaccepted_code, Some(ordinal), Some(key_id), "recovery-authority signer is not accepted by the relevant policy");
            continue;
        }
        let Some(signature) = decode_hex(&statement.signature_hex) else { continue };
        if verifier.verify(&statement.payload.canonical_bytes(), &statement.signer, &signature).is_err() {
            authority_issue(report, rejected_code, Some(ordinal), Some(key_id), "external verifier rejected recovery-authority rotation signature");
            continue;
        }
        authenticated += 1;
    }
    authenticated
}

fn policy_structurally_valid(policy: &CalibrationPublicationRecoveryAuthorityPolicy) -> bool {
    if policy.policy_version != crate::evidence_calibration::CALIBRATION_PUBLICATION_RECOVERY_AUTHORITY_POLICY_VERSION
        || policy.minimum_distinct_authorizers == 0
        || policy.accepted_key_ids.is_empty()
        || policy.minimum_distinct_authorizers > policy.accepted_key_ids.len() as u64
        || policy.accepted_key_ids.iter().any(String::is_empty)
    {
        return false;
    }
    let mut canonical = policy.accepted_key_ids.clone();
    canonical.sort();
    canonical.dedup();
    canonical == policy.accepted_key_ids
        && policy.policy_sha256 == calibration_publication_recovery_authority_policy_sha256(policy)
}

fn checkpoint_structurally_valid(checkpoint: &CalibrationPublicationCatalogCheckpoint) -> bool {
    checkpoint.checkpoint_version == CALIBRATION_PUBLICATION_CATALOG_CHECKPOINT_VERSION
        && checkpoint.checkpoint_sha256
            == calibration_publication_catalog_checkpoint_sha256(checkpoint)
        && !checkpoint.catalog_id.trim().is_empty()
        && !checkpoint.authority_id.trim().is_empty()
}

fn hash_envelopes(
    hash: &mut Sha256,
    envelopes: &[CalibrationSignedPublicationRecoveryAuthorityRotation],
) {
    let mut identities = envelopes.iter().map(|value| value.envelope_sha256.clone()).collect::<Vec<_>>();
    identities.sort();
    hash.update(&(identities.len() as u64).to_le_bytes());
    for identity in identities {
        hash_field(hash, &identity);
    }
}

fn hash_field(hash: &mut Sha256, value: &str) {
    hash.update(&(value.len() as u64).to_le_bytes());
    hash.update(value.as_bytes());
}

fn hash_optional_field(hash: &mut Sha256, value: Option<&str>) {
    match value {
        None => hash.update(&[0]),
        Some(value) => {
            hash.update(&[1]);
            hash_field(hash, value);
        }
    }
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn decode_hex(value: &str) -> Option<Vec<u8>> {
    if value.len() % 2 != 0 { return None; }
    let bytes = value.as_bytes();
    let mut output = Vec::with_capacity(value.len() / 2);
    for pair in bytes.chunks_exact(2) {
        output.push((hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?);
    }
    Some(output)
}

fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

fn authority_issue(
    report: &mut CalibrationPublicationRecoveryAuthorityAuditReport,
    code: CalibrationPublicationRecoveryAuthorityIssueCode,
    ordinal: Option<u64>,
    signer_key_id: Option<String>,
    detail: impl Into<String>,
) {
    report.issues.push(CalibrationPublicationRecoveryAuthorityIssue {
        code,
        ordinal,
        signer_key_id,
        detail: detail.into(),
    });
}

struct NeverVerifier;
impl CalibrationPublicationRecoveryAuthorityRotationVerifier for NeverVerifier {
    type Error = &'static str;
    fn verify(&self, _: &[u8], _: &CalibrationSignerIdentity, _: &[u8]) -> Result<(), Self::Error> {
        Err("not attempted")
    }
}
