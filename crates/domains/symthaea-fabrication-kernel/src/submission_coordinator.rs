// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Persist-before-dispatch coordination for governed print submissions.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::printer_control::PrinterApi;
use crate::submission::{
    GovernedAuthorizedPrintJob, GovernedSubmittedJobReceipt, SubmissionError,
    submit_governed_authorized_job,
};
use crate::submission_ledger::{
    SubmissionDisposition, SubmissionIntent, SubmissionLedger, SubmissionLedgerError,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubmissionContext {
    pub request_id: String,
    pub manifest_digest: Sha256Digest,
    pub machine_id: String,
    pub session_digest: Sha256Digest,
    pub session_sequence: u64,
}

impl SubmissionContext {
    fn as_intent(&self) -> SubmissionIntent<'_> {
        SubmissionIntent {
            request_id: &self.request_id,
            manifest_digest: self.manifest_digest,
            machine_id: &self.machine_id,
            session_digest: self.session_digest,
            session_sequence: self.session_sequence,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoordinatedSubmissionReceipt {
    pub receipt: GovernedSubmittedJobReceipt,
    pub request_id: String,
    pub ledger_head: Sha256Digest,
    pub ledger_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoordinatedSubmissionError {
    Ledger(SubmissionLedgerError),
    PersistenceBeforeDispatch(String),
    RejectedBeforeDispatch {
        source: SubmissionError,
        ledger_head: Sha256Digest,
    },
    SubmissionUncertain {
        source: SubmissionError,
        ledger_head: Sha256Digest,
    },
    OutcomePersistenceFailed {
        source: Option<SubmissionError>,
        acknowledged_receipt: Option<GovernedSubmittedJobReceipt>,
        in_memory_disposition: SubmissionDisposition,
        reason: String,
    },
}

/// Persist the intent before physical dispatch and persist the outcome after it.
///
/// `persist` must durably replace its previous snapshot before returning `Ok`.
/// The function never retries a printer call. A printer-layer error is recorded
/// as `Uncertain`, because a timeout does not prove that the machine rejected
/// the job.
#[allow(clippy::too_many_arguments)]
pub fn coordinate_governed_submission<P>(
    ledger: &mut SubmissionLedger,
    persist: &mut P,
    printer: &mut dyn PrinterApi,
    request_id: impl Into<String>,
    job: GovernedAuthorizedPrintJob,
    active_machine_id: &str,
    active_session_nonce: &str,
    now_unix_ms: u64,
) -> Result<CoordinatedSubmissionReceipt, CoordinatedSubmissionError>
where
    P: FnMut(&SubmissionLedger) -> Result<(), String>,
{
    let context = SubmissionContext {
        request_id: request_id.into(),
        manifest_digest: job.manifest_digest(),
        machine_id: job.machine_id().to_string(),
        session_digest: job.session_digest(),
        session_sequence: job.session_sequence(),
    };
    ledger
        .prepare(now_unix_ms, context.as_intent())
        .map_err(CoordinatedSubmissionError::Ledger)?;
    persist(ledger).map_err(CoordinatedSubmissionError::PersistenceBeforeDispatch)?;

    match submit_governed_authorized_job(
        printer,
        job,
        active_machine_id,
        active_session_nonce,
        now_unix_ms / 1_000,
    ) {
        Ok(receipt) => {
            let ledger_head = ledger
                .acknowledge(
                    now_unix_ms,
                    context.as_intent(),
                    receipt.submission.printer_job_id.clone(),
                )
                .map_err(CoordinatedSubmissionError::Ledger)?;
            if let Err(reason) = persist(ledger) {
                return Err(CoordinatedSubmissionError::OutcomePersistenceFailed {
                    source: None,
                    acknowledged_receipt: Some(receipt),
                    in_memory_disposition: ledger
                        .status(&context.request_id)
                        .unwrap_or(SubmissionDisposition::Prepared),
                    reason,
                });
            }
            let ledger_digest = ledger
                .digest()
                .map_err(CoordinatedSubmissionError::Ledger)?;
            Ok(CoordinatedSubmissionReceipt {
                receipt,
                request_id: context.request_id,
                ledger_head,
                ledger_digest,
            })
        }
        Err(source @ SubmissionError::Printer(_)) => {
            let error_digest = digest_submission_error(&source);
            let ledger_head = ledger
                .mark_uncertain(now_unix_ms, context.as_intent(), error_digest)
                .map_err(CoordinatedSubmissionError::Ledger)?;
            if let Err(reason) = persist(ledger) {
                return Err(CoordinatedSubmissionError::OutcomePersistenceFailed {
                    source: Some(source),
                    acknowledged_receipt: None,
                    in_memory_disposition: SubmissionDisposition::Uncertain,
                    reason,
                });
            }
            Err(CoordinatedSubmissionError::SubmissionUncertain {
                source,
                ledger_head,
            })
        }
        Err(source) => {
            let reason_digest = digest_submission_error(&source);
            let ledger_head = ledger
                .abandon(now_unix_ms, context.as_intent(), reason_digest)
                .map_err(CoordinatedSubmissionError::Ledger)?;
            if let Err(reason) = persist(ledger) {
                return Err(CoordinatedSubmissionError::OutcomePersistenceFailed {
                    source: Some(source),
                    acknowledged_receipt: None,
                    in_memory_disposition: SubmissionDisposition::Abandoned,
                    reason,
                });
            }
            Err(CoordinatedSubmissionError::RejectedBeforeDispatch {
                source,
                ledger_head,
            })
        }
    }
}

pub fn reconcile_uncertain_submission<P>(
    ledger: &mut SubmissionLedger,
    persist: &mut P,
    context: &SubmissionContext,
    printer_job_id: impl Into<String>,
    timestamp_unix_ms: u64,
) -> Result<Sha256Digest, CoordinatedSubmissionError>
where
    P: FnMut(&SubmissionLedger) -> Result<(), String>,
{
    let head = ledger
        .reconcile(timestamp_unix_ms, context.as_intent(), printer_job_id)
        .map_err(CoordinatedSubmissionError::Ledger)?;
    persist(ledger).map_err(
        |reason| CoordinatedSubmissionError::OutcomePersistenceFailed {
            source: None,
            acknowledged_receipt: None,
            in_memory_disposition: ledger
                .status(&context.request_id)
                .unwrap_or(SubmissionDisposition::Uncertain),
            reason,
        },
    )?;
    Ok(head)
}

pub fn digest_submission_error(error: &SubmissionError) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.submission-error.v1\0");
    match error {
        SubmissionError::MachineIdentityChanged { authorized, active } => {
            hasher.update(&[0]);
            append_string(&mut hasher, authorized);
            append_string(&mut hasher, active);
        }
        SubmissionError::SessionExpired { authorized, active } => {
            hasher.update(&[1]);
            append_string(&mut hasher, authorized);
            append_string(&mut hasher, active);
        }
        SubmissionError::Printer(message) => {
            hasher.update(&[2]);
            append_string(&mut hasher, message);
        }
    }
    hasher.finalize()
}

fn append_string(hasher: &mut Sha256, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;

    #[test]
    fn error_digest_is_variant_and_content_sensitive() {
        let left = SubmissionError::Printer("timeout".into());
        let right = SubmissionError::Printer("connection reset".into());
        let deterministic = SubmissionError::SessionExpired {
            authorized: "old".into(),
            active: "new".into(),
        };
        assert_ne!(
            digest_submission_error(&left),
            digest_submission_error(&right)
        );
        assert_ne!(
            digest_submission_error(&left),
            digest_submission_error(&deterministic)
        );
        assert_eq!(
            digest_submission_error(&left),
            digest_submission_error(&left)
        );
    }

    #[test]
    fn reconciliation_requires_an_uncertain_request() {
        let context = SubmissionContext {
            request_id: "request-1".into(),
            manifest_digest: sha256(b"manifest"),
            machine_id: "machine-1".into(),
            session_digest: sha256(b"session"),
            session_sequence: 1,
        };
        let mut ledger = SubmissionLedger::default();
        ledger.prepare(100, context.as_intent()).unwrap();
        let mut persist = |_ledger: &SubmissionLedger| Ok(());
        assert!(matches!(
            reconcile_uncertain_submission(&mut ledger, &mut persist, &context, "job-1", 101,),
            Err(CoordinatedSubmissionError::Ledger(
                SubmissionLedgerError::InvalidTransition { .. }
            ))
        ));
    }
}
