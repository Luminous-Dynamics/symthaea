// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Construction, authentication, conflict extraction, and integrity auditing.

use std::collections::{BTreeMap, BTreeSet};

use crate::evidence_calibration::{
    CALIBRATION_PUBLICATION_CATALOG_CHECKPOINT_VERSION,
    CalibrationPublicationCatalogCheckpoint, CalibrationSignerIdentity,
    calibration_publication_catalog_checkpoint_sha256,
};
use crate::evidence_calibration::sha256::{Sha256, hex as sha256_hex};

use super::model::*;

pub fn build_calibration_publication_gossip_payload(
    observer_id: impl Into<String>,
    checkpoint: CalibrationPublicationCatalogCheckpoint,
    previous_observed_checkpoint_sha256: Option<String>,
    witness_policy_epoch_sha256: impl Into<String>,
    observed_epoch: u64,
) -> CalibrationPublicationGossipPayload {
    let mut payload = CalibrationPublicationGossipPayload {
        payload_version: CALIBRATION_PUBLICATION_GOSSIP_PAYLOAD_VERSION.into(),
        observer_id: observer_id.into(),
        checkpoint,
        previous_observed_checkpoint_sha256,
        witness_policy_epoch_sha256: witness_policy_epoch_sha256.into(),
        observed_epoch,
        payload_sha256: String::new(),
    };
    payload.payload_sha256 = calibration_publication_gossip_payload_sha256(&payload);
    payload
}

pub fn build_calibration_signed_publication_gossip(
    payload: CalibrationPublicationGossipPayload,
    signer: CalibrationSignerIdentity,
    signature: &[u8],
) -> CalibrationSignedPublicationGossip {
    let mut envelope = CalibrationSignedPublicationGossip {
        envelope_version: CALIBRATION_SIGNED_PUBLICATION_GOSSIP_VERSION.into(),
        payload,
        signer,
        signature_hex: encode_hex(signature),
        envelope_sha256: String::new(),
    };
    envelope.envelope_sha256 = calibration_signed_publication_gossip_sha256(&envelope);
    envelope
}

pub fn build_calibration_publication_gossip_ledger(
    catalog_id: impl Into<String>,
    authority_id: impl Into<String>,
) -> CalibrationPublicationGossipLedger {
    let mut ledger = CalibrationPublicationGossipLedger {
        ledger_version: CALIBRATION_PUBLICATION_GOSSIP_LEDGER_VERSION.into(),
        catalog_id: catalog_id.into(),
        authority_id: authority_id.into(),
        statements: Vec::new(),
        ledger_sha256: String::new(),
    };
    ledger.ledger_sha256 = calibration_publication_gossip_ledger_sha256(&ledger);
    ledger
}

pub fn record_calibration_publication_gossip_statement<
    V: CalibrationPublicationGossipVerifier,
>(
    ledger: &mut CalibrationPublicationGossipLedger,
    statement: CalibrationSignedPublicationGossip,
    verifier: &V,
) -> Result<(), CalibrationPublicationGossipError> {
    let existing = audit_calibration_publication_gossip_ledger(ledger);
    if !existing.integrity_valid() {
        return Err(CalibrationPublicationGossipError::InvalidLedger {
            issues: existing.issues.len(),
        });
    }
    if statement.payload.checkpoint.catalog_id != ledger.catalog_id
        || statement.payload.checkpoint.authority_id != ledger.authority_id
    {
        return Err(CalibrationPublicationGossipError::IdentityMismatch);
    }
    if ledger
        .statements
        .iter()
        .any(|existing| existing.envelope_sha256 == statement.envelope_sha256)
    {
        return Err(CalibrationPublicationGossipError::DuplicateStatement);
    }
    let mut candidate = ledger.clone();
    candidate.statements.push(statement);
    candidate.ledger_sha256 = calibration_publication_gossip_ledger_sha256(&candidate);
    let verification = verify_calibration_publication_gossip_ledger(&candidate, verifier);
    if !verification.integrity_valid() {
        return Err(CalibrationPublicationGossipError::InvalidStatement {
            issues: verification.issues.len(),
        });
    }
    if !verification.signatures_authenticated {
        return Err(CalibrationPublicationGossipError::SignatureRejected);
    }
    *ledger = candidate;
    Ok(())
}

pub fn audit_calibration_publication_gossip_ledger(
    ledger: &CalibrationPublicationGossipLedger,
) -> CalibrationPublicationGossipAuditReport {
    audit_gossip_inner(ledger, None::<&NeverVerifier>)
}

pub fn verify_calibration_publication_gossip_ledger<V: CalibrationPublicationGossipVerifier>(
    ledger: &CalibrationPublicationGossipLedger,
    verifier: &V,
) -> CalibrationPublicationGossipAuditReport {
    audit_gossip_inner(ledger, Some(verifier))
}

pub fn build_calibration_publication_gossip_conflict_proof(
    kind: CalibrationPublicationGossipConflictKind,
    first: CalibrationSignedPublicationGossip,
    second: CalibrationSignedPublicationGossip,
) -> Result<CalibrationPublicationGossipConflictProof, CalibrationPublicationGossipError> {
    if !statements_conflict(kind, &first, &second) {
        return Err(CalibrationPublicationGossipError::NoConflict);
    }
    let mut proof = CalibrationPublicationGossipConflictProof {
        proof_version: CALIBRATION_PUBLICATION_GOSSIP_CONFLICT_PROOF_VERSION.into(),
        kind,
        first,
        second,
        proof_sha256: String::new(),
    };
    canonicalize_pair(&mut proof.first, &mut proof.second);
    proof.proof_sha256 = calibration_publication_gossip_conflict_proof_sha256(&proof);
    Ok(proof)
}

pub fn extract_calibration_publication_gossip_conflict_proofs(
    ledger: &CalibrationPublicationGossipLedger,
) -> Vec<CalibrationPublicationGossipConflictProof> {
    let mut proofs = Vec::new();
    for left_index in 0..ledger.statements.len() {
        for right_index in (left_index + 1)..ledger.statements.len() {
            let first = &ledger.statements[left_index];
            let second = &ledger.statements[right_index];
            for kind in [
                CalibrationPublicationGossipConflictKind::ObserverRollback,
                CalibrationPublicationGossipConflictKind::ObserverEquivocation,
                CalibrationPublicationGossipConflictKind::AuthorityEquivocation,
                CalibrationPublicationGossipConflictKind::CheckpointFork,
            ] {
                if let Ok(proof) = build_calibration_publication_gossip_conflict_proof(
                    kind,
                    first.clone(),
                    second.clone(),
                ) {
                    proofs.push(proof);
                }
            }
        }
    }
    proofs.sort_by(|left, right| {
        left.kind
            .cmp(&right.kind)
            .then_with(|| left.proof_sha256.cmp(&right.proof_sha256))
    });
    proofs.dedup_by(|left, right| left.proof_sha256 == right.proof_sha256);
    proofs
}

pub fn audit_calibration_publication_gossip_conflict_proof(
    proof: &CalibrationPublicationGossipConflictProof,
) -> CalibrationPublicationGossipConflictAuditReport {
    let mut report = CalibrationPublicationGossipConflictAuditReport {
        audit_version: CALIBRATION_PUBLICATION_GOSSIP_CONFLICT_AUDIT_VERSION.into(),
        issues: Vec::new(),
    };
    if proof.proof_version != CALIBRATION_PUBLICATION_GOSSIP_CONFLICT_PROOF_VERSION {
        conflict_issue(&mut report, CalibrationPublicationGossipConflictIssueCode::ProofVersionMismatch, "gossip conflict-proof version mismatch");
    }
    if !statement_structurally_valid(&proof.first) {
        conflict_issue(&mut report, CalibrationPublicationGossipConflictIssueCode::FirstStatementInvalid, "first gossip statement is structurally invalid");
    }
    if !statement_structurally_valid(&proof.second) {
        conflict_issue(&mut report, CalibrationPublicationGossipConflictIssueCode::SecondStatementInvalid, "second gossip statement is structurally invalid");
    }
    if !statements_conflict(proof.kind, &proof.first, &proof.second) {
        conflict_issue(&mut report, CalibrationPublicationGossipConflictIssueCode::ConflictKindMismatch, "statements do not prove the declared conflict kind");
    }
    if proof.proof_sha256 != calibration_publication_gossip_conflict_proof_sha256(proof) {
        conflict_issue(&mut report, CalibrationPublicationGossipConflictIssueCode::ProofSha256Mismatch, "gossip conflict-proof SHA-256 mismatch");
    }
    report
}

pub fn calibration_publication_gossip_payload_sha256(
    payload: &CalibrationPublicationGossipPayload,
) -> String {
    let mut hash = Sha256::new();
    hash.update(&payload.canonical_bytes());
    sha256_hex(&hash.finalize())
}

pub fn calibration_signed_publication_gossip_sha256(
    envelope: &CalibrationSignedPublicationGossip,
) -> String {
    let mut hash = Sha256::new();
    hash.update(GOSSIP_ENVELOPE_DOMAIN);
    hash_field(&mut hash, &envelope.envelope_version);
    hash_field(&mut hash, &envelope.payload.payload_sha256);
    hash_field(&mut hash, &envelope.signer.key_id);
    hash_field(&mut hash, &envelope.signer.algorithm);
    hash_optional_field(&mut hash, envelope.signer.issuer.as_deref());
    hash_field(&mut hash, &envelope.signature_hex);
    sha256_hex(&hash.finalize())
}

pub fn calibration_publication_gossip_ledger_sha256(
    ledger: &CalibrationPublicationGossipLedger,
) -> String {
    let mut hash = Sha256::new();
    hash.update(GOSSIP_LEDGER_DOMAIN);
    hash_field(&mut hash, &ledger.ledger_version);
    hash_field(&mut hash, &ledger.catalog_id);
    hash_field(&mut hash, &ledger.authority_id);
    hash.update(&(ledger.statements.len() as u64).to_le_bytes());
    for statement in &ledger.statements {
        hash_field(&mut hash, &statement.envelope_sha256);
    }
    sha256_hex(&hash.finalize())
}

pub fn calibration_publication_gossip_conflict_proof_sha256(
    proof: &CalibrationPublicationGossipConflictProof,
) -> String {
    let mut hash = Sha256::new();
    hash.update(GOSSIP_CONFLICT_DOMAIN);
    hash_field(&mut hash, &proof.proof_version);
    hash.update(&[conflict_kind_code(proof.kind)]);
    let mut identities = [
        proof.first.envelope_sha256.as_str(),
        proof.second.envelope_sha256.as_str(),
    ];
    identities.sort();
    hash_field(&mut hash, identities[0]);
    hash_field(&mut hash, identities[1]);
    sha256_hex(&hash.finalize())
}

fn audit_gossip_inner<V: CalibrationPublicationGossipVerifier>(
    ledger: &CalibrationPublicationGossipLedger,
    verifier: Option<&V>,
) -> CalibrationPublicationGossipAuditReport {
    let mut report = CalibrationPublicationGossipAuditReport {
        audit_version: CALIBRATION_PUBLICATION_GOSSIP_AUDIT_VERSION.into(),
        structurally_valid: true,
        signatures_authenticated: ledger.statements.is_empty(),
        authenticated_statements: 0,
        rollback_detected: false,
        observer_equivocation_detected: false,
        authority_equivocation_detected: false,
        fork_detected: false,
        issues: Vec::new(),
    };
    if ledger.ledger_version != CALIBRATION_PUBLICATION_GOSSIP_LEDGER_VERSION {
        gossip_issue(&mut report, CalibrationPublicationGossipIssueCode::LedgerVersionMismatch, None, None, "gossip-ledger version mismatch");
    }
    if ledger.catalog_id.trim().is_empty() || ledger.authority_id.trim().is_empty() {
        gossip_issue(&mut report, CalibrationPublicationGossipIssueCode::EmptyIdentity, None, None, "catalog and authority identities must not be empty");
    }
    let mut envelope_ids = BTreeSet::new();
    let mut observer_checkpoints = BTreeSet::new();
    let mut authenticated = 0u64;
    for statement in &ledger.statements {
        let observer = statement.payload.observer_id.clone();
        let checkpoint_sha = statement.payload.checkpoint.checkpoint_sha256.clone();
        if statement.payload.payload_version != CALIBRATION_PUBLICATION_GOSSIP_PAYLOAD_VERSION {
            gossip_issue(&mut report, CalibrationPublicationGossipIssueCode::PayloadVersionMismatch, Some(observer.clone()), Some(checkpoint_sha.clone()), "gossip-payload version mismatch");
        }
        if statement.envelope_version != CALIBRATION_SIGNED_PUBLICATION_GOSSIP_VERSION {
            gossip_issue(&mut report, CalibrationPublicationGossipIssueCode::EnvelopeVersionMismatch, Some(observer.clone()), Some(checkpoint_sha.clone()), "signed gossip-envelope version mismatch");
        }
        if observer.trim().is_empty() {
            gossip_issue(&mut report, CalibrationPublicationGossipIssueCode::EmptyObserverIdentity, Some(observer.clone()), Some(checkpoint_sha.clone()), "observer identity must not be empty");
        }
        if statement.signer.key_id.trim().is_empty() || statement.signer.algorithm.trim().is_empty() {
            gossip_issue(&mut report, CalibrationPublicationGossipIssueCode::EmptySignerIdentity, Some(observer.clone()), Some(checkpoint_sha.clone()), "gossip signer identity and algorithm must not be empty");
        }
        let signature = match decode_hex(&statement.signature_hex) {
            Some(value) if !value.is_empty() => value,
            _ => {
                gossip_issue(&mut report, CalibrationPublicationGossipIssueCode::InvalidSignatureHex, Some(observer.clone()), Some(checkpoint_sha.clone()), "gossip signature is not non-empty hexadecimal");
                Vec::new()
            }
        };
        let checkpoint = &statement.payload.checkpoint;
        if checkpoint.checkpoint_version != CALIBRATION_PUBLICATION_CATALOG_CHECKPOINT_VERSION {
            gossip_issue(&mut report, CalibrationPublicationGossipIssueCode::CheckpointVersionMismatch, Some(observer.clone()), Some(checkpoint_sha.clone()), "gossiped checkpoint version mismatch");
        }
        if checkpoint.checkpoint_sha256 != calibration_publication_catalog_checkpoint_sha256(checkpoint) {
            gossip_issue(&mut report, CalibrationPublicationGossipIssueCode::CheckpointSha256Mismatch, Some(observer.clone()), Some(checkpoint_sha.clone()), "gossiped checkpoint SHA-256 mismatch");
        }
        if checkpoint.catalog_id != ledger.catalog_id || checkpoint.authority_id != ledger.authority_id {
            gossip_issue(&mut report, CalibrationPublicationGossipIssueCode::CheckpointIdentityMismatch, Some(observer.clone()), Some(checkpoint_sha.clone()), "gossiped checkpoint identity differs from ledger identity");
        }
        if !is_sha256(&statement.payload.witness_policy_epoch_sha256) {
            gossip_issue(&mut report, CalibrationPublicationGossipIssueCode::InvalidPolicyEpochSha256, Some(observer.clone()), Some(checkpoint_sha.clone()), "witness-policy epoch identity is not canonical SHA-256");
        }
        if statement.payload.payload_sha256 != calibration_publication_gossip_payload_sha256(&statement.payload) {
            gossip_issue(&mut report, CalibrationPublicationGossipIssueCode::PayloadSha256Mismatch, Some(observer.clone()), Some(checkpoint_sha.clone()), "gossip-payload SHA-256 mismatch");
        }
        if statement.envelope_sha256 != calibration_signed_publication_gossip_sha256(statement) {
            gossip_issue(&mut report, CalibrationPublicationGossipIssueCode::EnvelopeSha256Mismatch, Some(observer.clone()), Some(checkpoint_sha.clone()), "signed gossip-envelope SHA-256 mismatch");
        }
        if !envelope_ids.insert(statement.envelope_sha256.clone()) {
            gossip_issue(&mut report, CalibrationPublicationGossipIssueCode::DuplicateStatement, Some(observer.clone()), Some(checkpoint_sha.clone()), "duplicate signed gossip statement");
        }
        if !observer_checkpoints.insert((observer.clone(), checkpoint_sha.clone())) {
            gossip_issue(&mut report, CalibrationPublicationGossipIssueCode::DuplicateObserverCheckpoint, Some(observer.clone()), Some(checkpoint_sha.clone()), "observer reported the same checkpoint more than once");
        }
        if let Some(verifier) = verifier {
            if !signature.is_empty()
                && verifier
                    .verify(&statement.payload.canonical_bytes(), &statement.signer, &signature)
                    .is_ok()
            {
                authenticated += 1;
            } else {
                gossip_issue(&mut report, CalibrationPublicationGossipIssueCode::SignatureRejected, Some(observer), Some(checkpoint_sha), "external gossip signature verifier rejected the statement");
            }
        }
    }
    report.authenticated_statements = authenticated;
    report.signatures_authenticated = verifier.is_some()
        && authenticated == ledger.statements.len() as u64;
    if ledger.statements.is_empty() {
        report.signatures_authenticated = true;
    }
    detect_conflicts(ledger, &mut report);
    if ledger.ledger_sha256 != calibration_publication_gossip_ledger_sha256(ledger) {
        gossip_issue(&mut report, CalibrationPublicationGossipIssueCode::LedgerSha256Mismatch, None, None, "gossip-ledger SHA-256 mismatch");
    }
    report.structurally_valid = report.issues.iter().all(|issue| {
        matches!(
            issue.code,
            CalibrationPublicationGossipIssueCode::ObserverRollback
                | CalibrationPublicationGossipIssueCode::ObserverEquivocation
                | CalibrationPublicationGossipIssueCode::AuthorityEquivocation
                | CalibrationPublicationGossipIssueCode::CheckpointFork
        )
    });
    report
}

fn detect_conflicts(
    ledger: &CalibrationPublicationGossipLedger,
    report: &mut CalibrationPublicationGossipAuditReport,
) {
    let mut last_by_observer: BTreeMap<&str, &CalibrationSignedPublicationGossip> = BTreeMap::new();
    let mut by_height: BTreeMap<u64, &CalibrationSignedPublicationGossip> = BTreeMap::new();
    let mut by_parent: BTreeMap<&str, &CalibrationSignedPublicationGossip> = BTreeMap::new();
    for statement in &ledger.statements {
        let observer = statement.payload.observer_id.as_str();
        if let Some(previous) = last_by_observer.get(observer).copied() {
            if statement.payload.observed_epoch < previous.payload.observed_epoch {
                gossip_issue(report, CalibrationPublicationGossipIssueCode::ObserverEpochRegression, Some(observer.into()), Some(statement.payload.checkpoint.checkpoint_sha256.clone()), "observer logical epoch regressed");
            }
            if statement.payload.previous_observed_checkpoint_sha256.as_deref()
                != Some(previous.payload.checkpoint.checkpoint_sha256.as_str())
            {
                gossip_issue(report, CalibrationPublicationGossipIssueCode::PreviousObservationMismatch, Some(observer.into()), Some(statement.payload.checkpoint.checkpoint_sha256.clone()), "observer chain does not name its preceding observation");
            }
            if statement.payload.checkpoint.event_count < previous.payload.checkpoint.event_count {
                report.rollback_detected = true;
                gossip_issue(report, CalibrationPublicationGossipIssueCode::ObserverRollback, Some(observer.into()), Some(statement.payload.checkpoint.checkpoint_sha256.clone()), "observer later reported a lower catalog height");
            }
            if statement.payload.checkpoint.event_count == previous.payload.checkpoint.event_count
                && statement.payload.checkpoint.checkpoint_sha256
                    != previous.payload.checkpoint.checkpoint_sha256
            {
                report.observer_equivocation_detected = true;
                gossip_issue(report, CalibrationPublicationGossipIssueCode::ObserverEquivocation, Some(observer.into()), Some(statement.payload.checkpoint.checkpoint_sha256.clone()), "observer reported different checkpoints at the same height");
            }
        } else if statement.payload.previous_observed_checkpoint_sha256.is_some() {
            gossip_issue(report, CalibrationPublicationGossipIssueCode::PreviousObservationMismatch, Some(observer.into()), Some(statement.payload.checkpoint.checkpoint_sha256.clone()), "first observation for an observer must not name an unknown predecessor");
        }
        last_by_observer.insert(observer, statement);

        let height = statement.payload.checkpoint.event_count;
        if let Some(existing) = by_height.get(&height).copied() {
            if existing.payload.checkpoint.checkpoint_sha256
                != statement.payload.checkpoint.checkpoint_sha256
            {
                report.authority_equivocation_detected = true;
                gossip_issue(report, CalibrationPublicationGossipIssueCode::AuthorityEquivocation, Some(observer.into()), Some(statement.payload.checkpoint.checkpoint_sha256.clone()), "different checkpoints were observed at the same catalog height");
            }
        } else {
            by_height.insert(height, statement);
        }

        if let Some(parent) = statement.payload.checkpoint.previous_checkpoint_sha256.as_deref() {
            if let Some(existing) = by_parent.get(parent).copied() {
                if existing.payload.checkpoint.checkpoint_sha256
                    != statement.payload.checkpoint.checkpoint_sha256
                {
                    report.fork_detected = true;
                    gossip_issue(report, CalibrationPublicationGossipIssueCode::CheckpointFork, Some(observer.into()), Some(statement.payload.checkpoint.checkpoint_sha256.clone()), "two observed checkpoints name the same predecessor");
                }
            } else {
                by_parent.insert(parent, statement);
            }
        }
    }
}

fn statement_structurally_valid(statement: &CalibrationSignedPublicationGossip) -> bool {
    statement.envelope_version == CALIBRATION_SIGNED_PUBLICATION_GOSSIP_VERSION
        && statement.payload.payload_version == CALIBRATION_PUBLICATION_GOSSIP_PAYLOAD_VERSION
        && !statement.payload.observer_id.trim().is_empty()
        && !statement.signer.key_id.trim().is_empty()
        && !statement.signer.algorithm.trim().is_empty()
        && statement.payload.checkpoint.checkpoint_version
            == CALIBRATION_PUBLICATION_CATALOG_CHECKPOINT_VERSION
        && statement.payload.checkpoint.checkpoint_sha256
            == calibration_publication_catalog_checkpoint_sha256(&statement.payload.checkpoint)
        && is_sha256(&statement.payload.witness_policy_epoch_sha256)
        && statement.payload.payload_sha256
            == calibration_publication_gossip_payload_sha256(&statement.payload)
        && statement.envelope_sha256 == calibration_signed_publication_gossip_sha256(statement)
        && decode_hex(&statement.signature_hex).is_some_and(|value| !value.is_empty())
}

fn statements_conflict(
    kind: CalibrationPublicationGossipConflictKind,
    first: &CalibrationSignedPublicationGossip,
    second: &CalibrationSignedPublicationGossip,
) -> bool {
    if first.payload.checkpoint.catalog_id != second.payload.checkpoint.catalog_id
        || first.payload.checkpoint.authority_id != second.payload.checkpoint.authority_id
        || first.envelope_sha256 == second.envelope_sha256
    {
        return false;
    }
    match kind {
        CalibrationPublicationGossipConflictKind::ObserverRollback => {
            first.payload.observer_id == second.payload.observer_id
                && ((first.payload.observed_epoch < second.payload.observed_epoch
                    && first.payload.checkpoint.event_count > second.payload.checkpoint.event_count)
                    || (second.payload.observed_epoch < first.payload.observed_epoch
                        && second.payload.checkpoint.event_count > first.payload.checkpoint.event_count))
        }
        CalibrationPublicationGossipConflictKind::ObserverEquivocation => {
            first.payload.observer_id == second.payload.observer_id
                && first.payload.checkpoint.event_count == second.payload.checkpoint.event_count
                && first.payload.checkpoint.checkpoint_sha256
                    != second.payload.checkpoint.checkpoint_sha256
        }
        CalibrationPublicationGossipConflictKind::AuthorityEquivocation => {
            first.payload.checkpoint.event_count == second.payload.checkpoint.event_count
                && first.payload.checkpoint.checkpoint_sha256
                    != second.payload.checkpoint.checkpoint_sha256
        }
        CalibrationPublicationGossipConflictKind::CheckpointFork => {
            first.payload.checkpoint.previous_checkpoint_sha256.is_some()
                && first.payload.checkpoint.previous_checkpoint_sha256
                    == second.payload.checkpoint.previous_checkpoint_sha256
                && first.payload.checkpoint.checkpoint_sha256
                    != second.payload.checkpoint.checkpoint_sha256
        }
    }
}

fn canonicalize_pair(
    first: &mut CalibrationSignedPublicationGossip,
    second: &mut CalibrationSignedPublicationGossip,
) {
    if second.envelope_sha256 < first.envelope_sha256 {
        std::mem::swap(first, second);
    }
}

fn conflict_kind_code(kind: CalibrationPublicationGossipConflictKind) -> u8 {
    match kind {
        CalibrationPublicationGossipConflictKind::ObserverRollback => 0,
        CalibrationPublicationGossipConflictKind::ObserverEquivocation => 1,
        CalibrationPublicationGossipConflictKind::AuthorityEquivocation => 2,
        CalibrationPublicationGossipConflictKind::CheckpointFork => 3,
    }
}

fn gossip_issue(
    report: &mut CalibrationPublicationGossipAuditReport,
    code: CalibrationPublicationGossipIssueCode,
    observer_id: Option<String>,
    checkpoint_sha256: Option<String>,
    detail: impl Into<String>,
) {
    report.issues.push(CalibrationPublicationGossipIssue {
        code,
        observer_id,
        checkpoint_sha256,
        detail: detail.into(),
    });
}

fn conflict_issue(
    report: &mut CalibrationPublicationGossipConflictAuditReport,
    code: CalibrationPublicationGossipConflictIssueCode,
    detail: impl Into<String>,
) {
    report.issues.push(CalibrationPublicationGossipConflictIssue {
        code,
        detail: detail.into(),
    });
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
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
    if value.len() % 2 != 0 {
        return None;
    }
    let bytes = value.as_bytes();
    let mut output = Vec::with_capacity(bytes.len() / 2);
    let mut index = 0;
    while index < bytes.len() {
        output.push((hex_nibble(bytes[index])? << 4) | hex_nibble(bytes[index + 1])?);
        index += 2;
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

struct NeverVerifier;

impl CalibrationPublicationGossipVerifier for NeverVerifier {
    type Error = &'static str;

    fn verify(
        &self,
        _payload: &[u8],
        _signer: &CalibrationSignerIdentity,
        _signature: &[u8],
    ) -> Result<(), Self::Error> {
        Err("verification not attempted")
    }
}
