// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-journal reconciliation for submission and audit evidence.

use crate::audit::{AuditAction, AuditAppendError, AuditJournal};
use crate::crypto_digest::Sha256Digest;
use crate::submission_ledger::{
    SubmissionLedger, SubmissionLedgerAction, SubmissionLedgerError, SubmissionLedgerEvent,
    digest_submission_ledger_event,
};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconciliationError {
    AuditNotIntact(String),
    SubmissionLedgerNotIntact(String),
    SubmissionEvent(SubmissionLedgerError),
    Audit(AuditAppendError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubmissionAuditMismatch {
    MissingAuditRecord {
        submission_sequence: u64,
        event_digest: Sha256Digest,
    },
    DuplicateAuditRecord {
        submission_sequence: u64,
        event_digest: Sha256Digest,
        count: usize,
    },
    OrphanAuditRecord {
        audit_sequence: u64,
        event_digest: Sha256Digest,
    },
    TimestampMismatch {
        submission_sequence: u64,
        audit_sequence: u64,
        expected_unix_s: u64,
        actual_unix_s: u64,
    },
    ActionMismatch {
        submission_sequence: u64,
        audit_sequence: u64,
    },
    SubjectMismatch {
        submission_sequence: u64,
        audit_sequence: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubmissionAuditReconciliationReport {
    pub submission_event_count: usize,
    pub matched_event_count: usize,
    pub mismatches: Vec<SubmissionAuditMismatch>,
}

impl SubmissionAuditReconciliationReport {
    pub fn reconciled(&self) -> bool {
        self.mismatches.is_empty() && self.matched_event_count == self.submission_event_count
    }
}

pub fn append_submission_event_audit(
    journal: &mut AuditJournal,
    actor: impl Into<String>,
    event: &SubmissionLedgerEvent,
) -> Result<Sha256Digest, ReconciliationError> {
    let event_digest =
        digest_submission_ledger_event(event).map_err(ReconciliationError::SubmissionEvent)?;
    journal
        .append(
            event.timestamp_unix_ms / 1_000,
            actor,
            audit_action_for(&event.action),
            event.manifest_digest,
            Some(event_digest),
        )
        .map_err(ReconciliationError::Audit)
}

pub fn reconcile_submission_audit(
    ledger: &SubmissionLedger,
    journal: &AuditJournal,
) -> Result<SubmissionAuditReconciliationReport, ReconciliationError> {
    let ledger_report = ledger.verify();
    if !ledger_report.intact() {
        return Err(ReconciliationError::SubmissionLedgerNotIntact(format!(
            "{:?}",
            ledger_report.violations
        )));
    }
    let audit_report = journal.verify();
    if !audit_report.intact() {
        return Err(ReconciliationError::AuditNotIntact(format!(
            "{:?}",
            audit_report.violations
        )));
    }

    let mut relevant_audit = BTreeMap::<Sha256Digest, Vec<_>>::new();
    for event in &journal.events {
        if is_submission_audit_action(&event.action) {
            if let Some(details) = event.details_digest {
                relevant_audit.entry(details).or_default().push(event);
            }
        }
    }

    let mut mismatches = Vec::new();
    let mut matched_event_count = 0;
    let mut expected_digests = BTreeMap::<Sha256Digest, u64>::new();
    for event in &ledger.events {
        let event_digest =
            digest_submission_ledger_event(event).map_err(ReconciliationError::SubmissionEvent)?;
        expected_digests.insert(event_digest, event.sequence);
        let matches = relevant_audit
            .get(&event_digest)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        if matches.is_empty() {
            mismatches.push(SubmissionAuditMismatch::MissingAuditRecord {
                submission_sequence: event.sequence,
                event_digest,
            });
            continue;
        }
        if matches.len() > 1 {
            mismatches.push(SubmissionAuditMismatch::DuplicateAuditRecord {
                submission_sequence: event.sequence,
                event_digest,
                count: matches.len(),
            });
            continue;
        }
        let audit = matches[0];
        let expected_action = audit_action_for(&event.action);
        if audit.action != expected_action {
            mismatches.push(SubmissionAuditMismatch::ActionMismatch {
                submission_sequence: event.sequence,
                audit_sequence: audit.sequence,
            });
        }
        if audit.subject_digest != event.manifest_digest {
            mismatches.push(SubmissionAuditMismatch::SubjectMismatch {
                submission_sequence: event.sequence,
                audit_sequence: audit.sequence,
            });
        }
        let expected_unix_s = event.timestamp_unix_ms / 1_000;
        if audit.timestamp_unix_s != expected_unix_s {
            mismatches.push(SubmissionAuditMismatch::TimestampMismatch {
                submission_sequence: event.sequence,
                audit_sequence: audit.sequence,
                expected_unix_s,
                actual_unix_s: audit.timestamp_unix_s,
            });
        }
        if audit.action == expected_action
            && audit.subject_digest == event.manifest_digest
            && audit.timestamp_unix_s == expected_unix_s
        {
            matched_event_count += 1;
        }
    }

    for (digest, events) in relevant_audit {
        if !expected_digests.contains_key(&digest) {
            for event in events {
                mismatches.push(SubmissionAuditMismatch::OrphanAuditRecord {
                    audit_sequence: event.sequence,
                    event_digest: digest,
                });
            }
        }
    }

    Ok(SubmissionAuditReconciliationReport {
        submission_event_count: ledger.events.len(),
        matched_event_count,
        mismatches,
    })
}

fn audit_action_for(action: &SubmissionLedgerAction) -> AuditAction {
    match action {
        SubmissionLedgerAction::Prepared => AuditAction::SubmissionPrepared,
        SubmissionLedgerAction::Acknowledged { .. } => AuditAction::SubmissionAcknowledged,
        SubmissionLedgerAction::Uncertain { .. } => AuditAction::SubmissionUncertain,
        SubmissionLedgerAction::Reconciled { .. } => AuditAction::SubmissionReconciled,
        SubmissionLedgerAction::Abandoned { .. } => AuditAction::SubmissionAbandoned,
    }
}

fn is_submission_audit_action(action: &AuditAction) -> bool {
    matches!(
        action,
        AuditAction::SubmissionPrepared
            | AuditAction::SubmissionAcknowledged
            | AuditAction::SubmissionUncertain
            | AuditAction::SubmissionReconciled
            | AuditAction::SubmissionAbandoned
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;
    use crate::submission_ledger::SubmissionIntent;

    fn intent<'a>(request_id: &'a str) -> SubmissionIntent<'a> {
        SubmissionIntent {
            request_id,
            manifest_digest: sha256(b"manifest"),
            machine_id: "machine-1",
            session_digest: sha256(b"session"),
            session_sequence: 1,
        }
    }

    #[test]
    fn exact_cross_journal_evidence_reconciles() {
        let mut ledger = SubmissionLedger::default();
        ledger.prepare(100_000, intent("request-1")).unwrap();
        ledger
            .acknowledge(101_000, intent("request-1"), "job-1")
            .unwrap();
        let mut audit = AuditJournal::default();
        for event in &ledger.events {
            append_submission_event_audit(&mut audit, "gateway", event).unwrap();
        }
        let report = reconcile_submission_audit(&ledger, &audit).unwrap();
        assert!(report.reconciled());
        assert_eq!(report.matched_event_count, 2);
    }

    #[test]
    fn missing_and_orphan_records_are_detected() {
        let mut ledger = SubmissionLedger::default();
        ledger.prepare(100_000, intent("request-1")).unwrap();
        let mut audit = AuditJournal::default();
        audit
            .append(
                100,
                "gateway",
                AuditAction::SubmissionPrepared,
                sha256(b"manifest"),
                Some(sha256(b"orphan")),
            )
            .unwrap();
        let report = reconcile_submission_audit(&ledger, &audit).unwrap();
        assert!(report.mismatches.iter().any(|mismatch| matches!(
            mismatch,
            SubmissionAuditMismatch::MissingAuditRecord { .. }
        )));
        assert!(
            report.mismatches.iter().any(|mismatch| matches!(
                mismatch,
                SubmissionAuditMismatch::OrphanAuditRecord { .. }
            ))
        );
    }
}
