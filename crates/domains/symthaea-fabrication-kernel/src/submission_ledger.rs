// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tamper-evident, crash-recoverable submission intent ledger.
//!
//! A governed gateway persists `Prepared` before contacting a printer. A
//! transport error becomes `Uncertain`, never an implicit permission to retry.
//! Only external reconciliation may resolve an uncertain physical outcome.

use crate::crypto_digest::{Sha256, Sha256Digest};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const SUBMISSION_LEDGER_SCHEMA: &str = "symthaea.fabrication.submission-ledger.v1";
pub const MAX_SUBMISSION_EVENTS: usize = 1_000_000;
pub const MAX_SUBMISSION_ID_BYTES: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubmissionLedgerAction {
    Prepared,
    Acknowledged { printer_job_id: String },
    Uncertain { error_digest: Sha256Digest },
    Reconciled { printer_job_id: String },
    Abandoned { reason_digest: Sha256Digest },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubmissionLedgerEvent {
    pub sequence: u64,
    pub timestamp_unix_ms: u64,
    pub request_id: String,
    pub manifest_digest: Sha256Digest,
    pub machine_id: String,
    pub session_digest: Sha256Digest,
    pub session_sequence: u64,
    pub action: SubmissionLedgerAction,
    pub previous_hash: Option<Sha256Digest>,
    pub record_hash: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubmissionLedger {
    pub schema_version: String,
    pub events: Vec<SubmissionLedgerEvent>,
}

impl Default for SubmissionLedger {
    fn default() -> Self {
        Self {
            schema_version: SUBMISSION_LEDGER_SCHEMA.into(),
            events: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubmissionDisposition {
    Prepared,
    Acknowledged { printer_job_id: String },
    Uncertain,
    Reconciled { printer_job_id: String },
    Abandoned,
}

impl SubmissionDisposition {
    pub fn terminal(&self) -> bool {
        matches!(
            self,
            Self::Acknowledged { .. } | Self::Reconciled { .. } | Self::Abandoned
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubmissionLedgerError {
    UnsupportedSchema,
    CapacityExceeded,
    InvalidIdentifier(&'static str),
    ZeroSessionSequence,
    DuplicateRequest(String),
    UnknownRequest(String),
    InvalidTransition {
        request_id: String,
        from: SubmissionDisposition,
        action: &'static str,
    },
    RequestContextMismatch(String),
    TimestampRegressed {
        previous: u64,
        current: u64,
    },
    SequenceOverflow,
    VerificationFailed(Vec<SubmissionLedgerViolation>),
    Encoding(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubmissionLedgerViolation {
    UnsupportedSchema,
    SequenceMismatch {
        index: usize,
        actual: u64,
        expected: u64,
    },
    TimestampRegressed {
        index: usize,
    },
    InvalidIdentifier {
        index: usize,
        field: &'static str,
    },
    ZeroSessionSequence {
        index: usize,
    },
    PreviousHashMismatch {
        index: usize,
    },
    RecordHashMismatch {
        index: usize,
    },
    DuplicatePrepared {
        index: usize,
    },
    MissingPrepared {
        index: usize,
    },
    ContextChanged {
        index: usize,
    },
    InvalidTransition {
        index: usize,
    },
    Encoding {
        index: usize,
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubmissionLedgerVerificationReport {
    pub violations: Vec<SubmissionLedgerViolation>,
    pub verified_head: Option<Sha256Digest>,
}

impl SubmissionLedgerVerificationReport {
    pub fn intact(&self) -> bool {
        self.violations.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct SubmissionIntent<'a> {
    pub request_id: &'a str,
    pub manifest_digest: Sha256Digest,
    pub machine_id: &'a str,
    pub session_digest: Sha256Digest,
    pub session_sequence: u64,
}

#[derive(Debug, Clone)]
struct RequestState {
    manifest_digest: Sha256Digest,
    machine_id: String,
    session_digest: Sha256Digest,
    session_sequence: u64,
    disposition: SubmissionDisposition,
}

impl SubmissionLedger {
    pub fn prepare(
        &mut self,
        timestamp_unix_ms: u64,
        intent: SubmissionIntent<'_>,
    ) -> Result<Sha256Digest, SubmissionLedgerError> {
        self.require_intact()?;
        validate_intent(&intent)?;
        if self.status(intent.request_id).is_some() {
            return Err(SubmissionLedgerError::DuplicateRequest(
                intent.request_id.to_string(),
            ));
        }
        self.append(timestamp_unix_ms, intent, SubmissionLedgerAction::Prepared)
    }

    pub fn acknowledge(
        &mut self,
        timestamp_unix_ms: u64,
        intent: SubmissionIntent<'_>,
        printer_job_id: impl Into<String>,
    ) -> Result<Sha256Digest, SubmissionLedgerError> {
        let printer_job_id = printer_job_id.into();
        if !canonical(&printer_job_id) {
            return Err(SubmissionLedgerError::InvalidIdentifier("printer_job_id"));
        }
        self.transition(
            timestamp_unix_ms,
            intent,
            SubmissionLedgerAction::Acknowledged { printer_job_id },
            "acknowledge",
            |state| matches!(state, SubmissionDisposition::Prepared),
        )
    }

    pub fn mark_uncertain(
        &mut self,
        timestamp_unix_ms: u64,
        intent: SubmissionIntent<'_>,
        error_digest: Sha256Digest,
    ) -> Result<Sha256Digest, SubmissionLedgerError> {
        self.transition(
            timestamp_unix_ms,
            intent,
            SubmissionLedgerAction::Uncertain { error_digest },
            "mark_uncertain",
            |state| matches!(state, SubmissionDisposition::Prepared),
        )
    }

    pub fn reconcile(
        &mut self,
        timestamp_unix_ms: u64,
        intent: SubmissionIntent<'_>,
        printer_job_id: impl Into<String>,
    ) -> Result<Sha256Digest, SubmissionLedgerError> {
        let printer_job_id = printer_job_id.into();
        if !canonical(&printer_job_id) {
            return Err(SubmissionLedgerError::InvalidIdentifier("printer_job_id"));
        }
        self.transition(
            timestamp_unix_ms,
            intent,
            SubmissionLedgerAction::Reconciled { printer_job_id },
            "reconcile",
            |state| matches!(state, SubmissionDisposition::Uncertain),
        )
    }

    pub fn abandon(
        &mut self,
        timestamp_unix_ms: u64,
        intent: SubmissionIntent<'_>,
        reason_digest: Sha256Digest,
    ) -> Result<Sha256Digest, SubmissionLedgerError> {
        self.transition(
            timestamp_unix_ms,
            intent,
            SubmissionLedgerAction::Abandoned { reason_digest },
            "abandon",
            |state| {
                matches!(
                    state,
                    SubmissionDisposition::Prepared | SubmissionDisposition::Uncertain
                )
            },
        )
    }

    pub fn status(&self, request_id: &str) -> Option<SubmissionDisposition> {
        replay_states(&self.events).ok().and_then(|states| {
            states
                .get(request_id)
                .map(|state| state.disposition.clone())
        })
    }

    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    pub fn head(&self) -> Option<Sha256Digest> {
        self.events.last().map(|event| event.record_hash)
    }

    pub fn digest(&self) -> Result<Sha256Digest, SubmissionLedgerError> {
        self.require_intact()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| SubmissionLedgerError::Encoding(error.to_string()))?;
        let mut hasher = Sha256::new();
        hasher.update(b"symthaea.fabrication.submission-ledger-digest.v1\0");
        hasher.update(&bytes);
        Ok(hasher.finalize())
    }

    pub fn verify(&self) -> SubmissionLedgerVerificationReport {
        let mut violations = Vec::new();
        if self.schema_version != SUBMISSION_LEDGER_SCHEMA {
            violations.push(SubmissionLedgerViolation::UnsupportedSchema);
        }
        let mut states = BTreeMap::<String, RequestState>::new();
        let mut previous_hash = None;
        let mut previous_timestamp = None;
        let mut verified_head = None;
        for (index, event) in self.events.iter().enumerate() {
            let expected_sequence = index as u64 + 1;
            if event.sequence != expected_sequence {
                violations.push(SubmissionLedgerViolation::SequenceMismatch {
                    index,
                    actual: event.sequence,
                    expected: expected_sequence,
                });
            }
            if previous_timestamp.is_some_and(|value| event.timestamp_unix_ms < value) {
                violations.push(SubmissionLedgerViolation::TimestampRegressed { index });
            }
            for (field, value) in [
                ("request_id", event.request_id.as_str()),
                ("machine_id", event.machine_id.as_str()),
            ] {
                if !canonical(value) {
                    violations.push(SubmissionLedgerViolation::InvalidIdentifier { index, field });
                }
            }
            if event.session_sequence == 0 {
                violations.push(SubmissionLedgerViolation::ZeroSessionSequence { index });
            }
            if event.previous_hash != previous_hash {
                violations.push(SubmissionLedgerViolation::PreviousHashMismatch { index });
            }
            match hash_event_fields(event) {
                Ok(expected) if expected == event.record_hash => verified_head = Some(expected),
                Ok(_) => violations.push(SubmissionLedgerViolation::RecordHashMismatch { index }),
                Err(error) => violations.push(SubmissionLedgerViolation::Encoding {
                    index,
                    reason: format!("{error:?}"),
                }),
            }
            apply_event(index, event, &mut states, &mut violations);
            previous_hash = Some(event.record_hash);
            previous_timestamp = Some(event.timestamp_unix_ms);
        }
        SubmissionLedgerVerificationReport {
            violations,
            verified_head,
        }
    }

    pub fn verify_successor_of(&self, previous: &Self) -> Result<(), SubmissionLedgerError> {
        previous.require_intact()?;
        self.require_intact()?;
        if self.events.len() < previous.events.len()
            || self.events[..previous.events.len()] != previous.events
        {
            return Err(SubmissionLedgerError::VerificationFailed(vec![
                SubmissionLedgerViolation::InvalidTransition { index: 0 },
            ]));
        }
        Ok(())
    }

    fn transition<F>(
        &mut self,
        timestamp_unix_ms: u64,
        intent: SubmissionIntent<'_>,
        action: SubmissionLedgerAction,
        action_name: &'static str,
        allowed: F,
    ) -> Result<Sha256Digest, SubmissionLedgerError>
    where
        F: FnOnce(&SubmissionDisposition) -> bool,
    {
        self.require_intact()?;
        validate_intent(&intent)?;
        let states = replay_states(&self.events)?;
        let Some(state) = states.get(intent.request_id) else {
            return Err(SubmissionLedgerError::UnknownRequest(
                intent.request_id.to_string(),
            ));
        };
        if !same_context(state, &intent) {
            return Err(SubmissionLedgerError::RequestContextMismatch(
                intent.request_id.to_string(),
            ));
        }
        if !allowed(&state.disposition) {
            return Err(SubmissionLedgerError::InvalidTransition {
                request_id: intent.request_id.to_string(),
                from: state.disposition.clone(),
                action: action_name,
            });
        }
        self.append(timestamp_unix_ms, intent, action)
    }

    fn append(
        &mut self,
        timestamp_unix_ms: u64,
        intent: SubmissionIntent<'_>,
        action: SubmissionLedgerAction,
    ) -> Result<Sha256Digest, SubmissionLedgerError> {
        if self.events.len() >= MAX_SUBMISSION_EVENTS {
            return Err(SubmissionLedgerError::CapacityExceeded);
        }
        if let Some(previous) = self.events.last() {
            if timestamp_unix_ms < previous.timestamp_unix_ms {
                return Err(SubmissionLedgerError::TimestampRegressed {
                    previous: previous.timestamp_unix_ms,
                    current: timestamp_unix_ms,
                });
            }
        }
        let sequence = u64::try_from(self.events.len())
            .ok()
            .and_then(|value| value.checked_add(1))
            .ok_or(SubmissionLedgerError::SequenceOverflow)?;
        let previous_hash = self.head();
        let mut event = SubmissionLedgerEvent {
            sequence,
            timestamp_unix_ms,
            request_id: intent.request_id.to_string(),
            manifest_digest: intent.manifest_digest,
            machine_id: intent.machine_id.to_string(),
            session_digest: intent.session_digest,
            session_sequence: intent.session_sequence,
            action,
            previous_hash,
            record_hash: Sha256Digest([0; 32]),
        };
        event.record_hash = hash_event_fields(&event)?;
        let result = event.record_hash;
        self.events.push(event);
        Ok(result)
    }

    fn require_intact(&self) -> Result<(), SubmissionLedgerError> {
        if self.events.len() > MAX_SUBMISSION_EVENTS {
            return Err(SubmissionLedgerError::CapacityExceeded);
        }
        let report = self.verify();
        if report.intact() {
            Ok(())
        } else {
            Err(SubmissionLedgerError::VerificationFailed(report.violations))
        }
    }
}

fn validate_intent(intent: &SubmissionIntent<'_>) -> Result<(), SubmissionLedgerError> {
    if !canonical(intent.request_id) {
        return Err(SubmissionLedgerError::InvalidIdentifier("request_id"));
    }
    if !canonical(intent.machine_id) {
        return Err(SubmissionLedgerError::InvalidIdentifier("machine_id"));
    }
    if intent.session_sequence == 0 {
        return Err(SubmissionLedgerError::ZeroSessionSequence);
    }
    Ok(())
}

fn same_context(state: &RequestState, intent: &SubmissionIntent<'_>) -> bool {
    state.manifest_digest == intent.manifest_digest
        && state.machine_id == intent.machine_id
        && state.session_digest == intent.session_digest
        && state.session_sequence == intent.session_sequence
}

fn replay_states(
    events: &[SubmissionLedgerEvent],
) -> Result<BTreeMap<String, RequestState>, SubmissionLedgerError> {
    let mut states = BTreeMap::new();
    let mut violations = Vec::new();
    for (index, event) in events.iter().enumerate() {
        apply_event(index, event, &mut states, &mut violations);
    }
    if violations.is_empty() {
        Ok(states)
    } else {
        Err(SubmissionLedgerError::VerificationFailed(violations))
    }
}

fn apply_event(
    index: usize,
    event: &SubmissionLedgerEvent,
    states: &mut BTreeMap<String, RequestState>,
    violations: &mut Vec<SubmissionLedgerViolation>,
) {
    match &event.action {
        SubmissionLedgerAction::Prepared => {
            if states.contains_key(&event.request_id) {
                violations.push(SubmissionLedgerViolation::DuplicatePrepared { index });
            } else {
                states.insert(
                    event.request_id.clone(),
                    RequestState {
                        manifest_digest: event.manifest_digest,
                        machine_id: event.machine_id.clone(),
                        session_digest: event.session_digest,
                        session_sequence: event.session_sequence,
                        disposition: SubmissionDisposition::Prepared,
                    },
                );
            }
        }
        action => {
            let Some(state) = states.get_mut(&event.request_id) else {
                violations.push(SubmissionLedgerViolation::MissingPrepared { index });
                return;
            };
            if state.manifest_digest != event.manifest_digest
                || state.machine_id != event.machine_id
                || state.session_digest != event.session_digest
                || state.session_sequence != event.session_sequence
            {
                violations.push(SubmissionLedgerViolation::ContextChanged { index });
                return;
            }
            let next = match (state.disposition.clone(), action) {
                (
                    SubmissionDisposition::Prepared,
                    SubmissionLedgerAction::Acknowledged { printer_job_id },
                ) => Some(SubmissionDisposition::Acknowledged {
                    printer_job_id: printer_job_id.clone(),
                }),
                (SubmissionDisposition::Prepared, SubmissionLedgerAction::Uncertain { .. }) => {
                    Some(SubmissionDisposition::Uncertain)
                }
                (
                    SubmissionDisposition::Uncertain,
                    SubmissionLedgerAction::Reconciled { printer_job_id },
                ) => Some(SubmissionDisposition::Reconciled {
                    printer_job_id: printer_job_id.clone(),
                }),
                (
                    SubmissionDisposition::Prepared | SubmissionDisposition::Uncertain,
                    SubmissionLedgerAction::Abandoned { .. },
                ) => Some(SubmissionDisposition::Abandoned),
                _ => None,
            };
            if let Some(next) = next {
                state.disposition = next;
            } else {
                violations.push(SubmissionLedgerViolation::InvalidTransition { index });
            }
        }
    }
}

#[derive(Serialize)]
struct SubmissionEventBody<'a> {
    sequence: u64,
    timestamp_unix_ms: u64,
    request_id: &'a str,
    manifest_digest: Sha256Digest,
    machine_id: &'a str,
    session_digest: Sha256Digest,
    session_sequence: u64,
    action: &'a SubmissionLedgerAction,
    previous_hash: Option<Sha256Digest>,
}

fn hash_event_fields(event: &SubmissionLedgerEvent) -> Result<Sha256Digest, SubmissionLedgerError> {
    let body = SubmissionEventBody {
        sequence: event.sequence,
        timestamp_unix_ms: event.timestamp_unix_ms,
        request_id: &event.request_id,
        manifest_digest: event.manifest_digest,
        machine_id: &event.machine_id,
        session_digest: event.session_digest,
        session_sequence: event.session_sequence,
        action: &event.action,
        previous_hash: event.previous_hash,
    };
    let bytes = serde_json::to_vec(&body)
        .map_err(|error| SubmissionLedgerError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.submission-ledger-event.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

/// Digest one intact ledger event for cross-journal reconciliation.
pub fn digest_submission_ledger_event(
    event: &SubmissionLedgerEvent,
) -> Result<Sha256Digest, SubmissionLedgerError> {
    let expected = hash_event_fields(event)?;
    if expected != event.record_hash {
        return Err(SubmissionLedgerError::VerificationFailed(vec![
            SubmissionLedgerViolation::RecordHashMismatch { index: 0 },
        ]));
    }
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.submission-ledger-evidence.v1\0");
    hasher.update(&event.record_hash.0);
    Ok(hasher.finalize())
}

fn canonical(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= MAX_SUBMISSION_ID_BYTES
        && !value.chars().any(char::is_control)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;

    fn intent<'a>(request_id: &'a str) -> SubmissionIntent<'a> {
        SubmissionIntent {
            request_id,
            manifest_digest: sha256(b"manifest"),
            machine_id: "machine-1",
            session_digest: sha256(b"session"),
            session_sequence: 4,
        }
    }

    #[test]
    fn uncertain_outcome_cannot_be_acknowledged_or_retried() {
        let mut ledger = SubmissionLedger::default();
        ledger.prepare(100, intent("request-1")).unwrap();
        ledger
            .mark_uncertain(101, intent("request-1"), sha256(b"timeout"))
            .unwrap();
        assert_eq!(
            ledger.status("request-1"),
            Some(SubmissionDisposition::Uncertain)
        );
        assert!(matches!(
            ledger.acknowledge(102, intent("request-1"), "job-1"),
            Err(SubmissionLedgerError::InvalidTransition { .. })
        ));
        ledger.reconcile(103, intent("request-1"), "job-1").unwrap();
        assert_eq!(
            ledger.status("request-1"),
            Some(SubmissionDisposition::Reconciled {
                printer_job_id: "job-1".into()
            })
        );
    }

    #[test]
    fn context_substitution_fails_closed() {
        let mut ledger = SubmissionLedger::default();
        ledger.prepare(100, intent("request-1")).unwrap();
        let mut wrong = intent("request-1");
        wrong.machine_id = "machine-2";
        assert!(matches!(
            ledger.acknowledge(101, wrong, "job-1"),
            Err(SubmissionLedgerError::RequestContextMismatch(_))
        ));
    }

    #[test]
    fn event_tampering_breaks_chain_verification() {
        let mut ledger = SubmissionLedger::default();
        ledger.prepare(100, intent("request-1")).unwrap();
        ledger
            .acknowledge(101, intent("request-1"), "job-1")
            .unwrap();
        ledger.events[0].machine_id = "machine-2".into();
        assert!(!ledger.verify().intact());
        assert!(ledger.digest().is_err());
    }
}
