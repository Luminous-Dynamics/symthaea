// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tamper-evident audit journals for fabrication authority transitions.

use crate::crypto_digest::{Sha256, Sha256Digest};
use serde::{Deserialize, Serialize};

pub const AUDIT_JOURNAL_SCHEMA: &str = "symthaea.fabrication.audit-journal.v1";
pub const MAX_AUDIT_EVENTS: usize = 1_000_000;
pub const MAX_AUDIT_ACTOR_BYTES: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditAction {
    AttestationVerified,
    ReleaseAuthorized,
    MachineSessionAccepted,
    MachineSessionConsumed,
    JobAuthorized,
    JobSubmitted,
    SubmissionPrepared,
    SubmissionAcknowledged,
    SubmissionUncertain,
    SubmissionReconciled,
    SubmissionAbandoned,
    TelemetryVerified,
    OperatorCommandVerified,
    OperatorCommandApplied,
    GatewayStateCommitted,
    ExecutionPaused,
    ExecutionCancelled,
    EmergencyStopped,
    FaultInjectionVerified,
    AuditAnchored,
    TrustRotationAuthorized,
    TrustRotationActivated,
    RecoveryAuthorized,
    RecoveryRejected,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditEvent {
    pub sequence: u64,
    pub timestamp_unix_s: u64,
    pub actor: String,
    pub action: AuditAction,
    pub subject_digest: Sha256Digest,
    pub details_digest: Option<Sha256Digest>,
    pub previous_hash: Option<Sha256Digest>,
    pub record_hash: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditJournal {
    pub schema_version: String,
    pub events: Vec<AuditEvent>,
}

impl Default for AuditJournal {
    fn default() -> Self {
        Self {
            schema_version: AUDIT_JOURNAL_SCHEMA.into(),
            events: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditAppendError {
    UnsupportedSchema,
    CapacityExceeded,
    EmptyActor,
    ActorTooLong { actual: usize, maximum: usize },
    TimestampRegressed { previous: u64, current: u64 },
    SequenceOverflow,
    Encoding(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditViolation {
    UnsupportedSchema,
    SequenceMismatch {
        index: usize,
        actual: u64,
        expected: u64,
    },
    TimestampRegressed {
        index: usize,
    },
    InvalidActor {
        index: usize,
    },
    PreviousHashMismatch {
        index: usize,
    },
    RecordHashMismatch {
        index: usize,
    },
    Encoding {
        index: usize,
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditVerificationReport {
    pub violations: Vec<AuditViolation>,
    pub verified_head: Option<Sha256Digest>,
}

impl AuditVerificationReport {
    pub fn intact(&self) -> bool {
        self.violations.is_empty()
    }
}

impl AuditJournal {
    pub fn append(
        &mut self,
        timestamp_unix_s: u64,
        actor: impl Into<String>,
        action: AuditAction,
        subject_digest: Sha256Digest,
        details_digest: Option<Sha256Digest>,
    ) -> Result<Sha256Digest, AuditAppendError> {
        if self.schema_version != AUDIT_JOURNAL_SCHEMA {
            return Err(AuditAppendError::UnsupportedSchema);
        }
        if self.events.len() >= MAX_AUDIT_EVENTS {
            return Err(AuditAppendError::CapacityExceeded);
        }
        let actor = actor.into();
        if actor.trim().is_empty() {
            return Err(AuditAppendError::EmptyActor);
        }
        if actor.len() > MAX_AUDIT_ACTOR_BYTES {
            return Err(AuditAppendError::ActorTooLong {
                actual: actor.len(),
                maximum: MAX_AUDIT_ACTOR_BYTES,
            });
        }
        if let Some(previous) = self.events.last() {
            if timestamp_unix_s < previous.timestamp_unix_s {
                return Err(AuditAppendError::TimestampRegressed {
                    previous: previous.timestamp_unix_s,
                    current: timestamp_unix_s,
                });
            }
        }
        let sequence = u64::try_from(self.events.len())
            .ok()
            .and_then(|value| value.checked_add(1))
            .ok_or(AuditAppendError::SequenceOverflow)?;
        let previous_hash = self.events.last().map(|event| event.record_hash);
        let record_hash = hash_event_fields(
            sequence,
            timestamp_unix_s,
            &actor,
            &action,
            subject_digest,
            details_digest,
            previous_hash,
        )?;
        self.events.push(AuditEvent {
            sequence,
            timestamp_unix_s,
            actor,
            action,
            subject_digest,
            details_digest,
            previous_hash,
            record_hash,
        });
        Ok(record_hash)
    }

    pub fn head(&self) -> Option<Sha256Digest> {
        self.events.last().map(|event| event.record_hash)
    }

    pub fn verify(&self) -> AuditVerificationReport {
        let mut violations = Vec::new();
        if self.schema_version != AUDIT_JOURNAL_SCHEMA {
            violations.push(AuditViolation::UnsupportedSchema);
        }
        let mut previous_timestamp = None;
        let mut previous_hash = None;
        let mut verified_head = None;
        for (index, event) in self.events.iter().enumerate() {
            let expected_sequence = index as u64 + 1;
            if event.sequence != expected_sequence {
                violations.push(AuditViolation::SequenceMismatch {
                    index,
                    actual: event.sequence,
                    expected: expected_sequence,
                });
            }
            if previous_timestamp.is_some_and(|previous| event.timestamp_unix_s < previous) {
                violations.push(AuditViolation::TimestampRegressed { index });
            }
            if event.actor.trim().is_empty() || event.actor.len() > MAX_AUDIT_ACTOR_BYTES {
                violations.push(AuditViolation::InvalidActor { index });
            }
            if event.previous_hash != previous_hash {
                violations.push(AuditViolation::PreviousHashMismatch { index });
            }
            match hash_event_fields(
                event.sequence,
                event.timestamp_unix_s,
                &event.actor,
                &event.action,
                event.subject_digest,
                event.details_digest,
                event.previous_hash,
            ) {
                Ok(expected) if expected == event.record_hash => verified_head = Some(expected),
                Ok(_) => violations.push(AuditViolation::RecordHashMismatch { index }),
                Err(error) => violations.push(AuditViolation::Encoding {
                    index,
                    reason: format!("{error:?}"),
                }),
            }
            previous_timestamp = Some(event.timestamp_unix_s);
            previous_hash = Some(event.record_hash);
        }
        AuditVerificationReport {
            violations,
            verified_head,
        }
    }
}

#[derive(Serialize)]
struct AuditEventBody<'a> {
    sequence: u64,
    timestamp_unix_s: u64,
    actor: &'a str,
    action: &'a AuditAction,
    subject_digest: Sha256Digest,
    details_digest: Option<Sha256Digest>,
    previous_hash: Option<Sha256Digest>,
}

fn hash_event_fields(
    sequence: u64,
    timestamp_unix_s: u64,
    actor: &str,
    action: &AuditAction,
    subject_digest: Sha256Digest,
    details_digest: Option<Sha256Digest>,
    previous_hash: Option<Sha256Digest>,
) -> Result<Sha256Digest, AuditAppendError> {
    let body = AuditEventBody {
        sequence,
        timestamp_unix_s,
        actor,
        action,
        subject_digest,
        details_digest,
        previous_hash,
    };
    let bytes =
        serde_json::to_vec(&body).map_err(|error| AuditAppendError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.audit-event.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

/// Recompute one event record hash from its canonical fields.
pub fn compute_audit_event_hash(event: &AuditEvent) -> Result<Sha256Digest, AuditAppendError> {
    hash_event_fields(
        event.sequence,
        event.timestamp_unix_s,
        &event.actor,
        &event.action,
        event.subject_digest,
        event.details_digest,
        event.previous_hash,
    )
}

pub fn digest_audit_journal(journal: &AuditJournal) -> Result<Sha256Digest, AuditAppendError> {
    let report = journal.verify();
    if !report.intact() {
        return Err(AuditAppendError::Encoding(format!(
            "audit journal is not intact: {:?}",
            report.violations
        )));
    }
    let bytes = serde_json::to_vec(journal)
        .map_err(|error| AuditAppendError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.audit-journal-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;

    #[test]
    fn append_builds_a_verified_hash_chain() {
        let subject = sha256(b"manifest");
        let mut journal = AuditJournal::default();
        let first = journal
            .append(
                100,
                "operator",
                AuditAction::AttestationVerified,
                subject,
                None,
            )
            .unwrap();
        let second = journal
            .append(
                101,
                "operator",
                AuditAction::JobAuthorized,
                subject,
                Some(first),
            )
            .unwrap();
        assert_eq!(journal.head(), Some(second));
        assert!(journal.verify().intact());
    }

    #[test]
    fn content_tampering_breaks_the_record_hash() {
        let mut journal = AuditJournal::default();
        journal
            .append(
                100,
                "operator",
                AuditAction::JobSubmitted,
                sha256(b"job"),
                None,
            )
            .unwrap();
        journal.events[0].actor = "intruder".into();
        assert!(
            journal.verify().violations.iter().any(|violation| matches!(
                violation,
                AuditViolation::RecordHashMismatch { index: 0 }
            ))
        );
    }

    #[test]
    fn deletion_breaks_sequence_and_previous_hash_evidence() {
        let mut journal = AuditJournal::default();
        for timestamp in 100..103 {
            journal
                .append(
                    timestamp,
                    "operator",
                    AuditAction::Other("step".into()),
                    sha256(&[timestamp as u8]),
                    None,
                )
                .unwrap();
        }
        journal.events.remove(1);
        let report = journal.verify();
        assert!(report.violations.iter().any(|violation| matches!(
            violation,
            AuditViolation::SequenceMismatch { index: 1, .. }
        )));
        assert!(report.violations.iter().any(|violation| matches!(
            violation,
            AuditViolation::PreviousHashMismatch { index: 1 }
        )));
    }

    #[test]
    fn timestamps_cannot_regress() {
        let mut journal = AuditJournal::default();
        journal
            .append(
                100,
                "operator",
                AuditAction::JobAuthorized,
                sha256(b"job"),
                None,
            )
            .unwrap();
        assert!(matches!(
            journal.append(
                99,
                "operator",
                AuditAction::JobSubmitted,
                sha256(b"job"),
                None
            ),
            Err(AuditAppendError::TimestampRegressed { .. })
        ));
    }
}
