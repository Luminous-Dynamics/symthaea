// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Append-only registry and resolution ledger for signed incident evidence.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::incident::VerifiedIncidentBundle;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const INCIDENT_LEDGER_SCHEMA: &str = "symthaea.fabrication.incident-ledger.v1";
pub const MAX_INCIDENT_LEDGER_EVENTS: usize = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IncidentLedgerAction {
    Registered,
    Resolved,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IncidentLedgerEvent {
    pub sequence: u64,
    pub timestamp_unix_s: u64,
    pub action: IncidentLedgerAction,
    pub incident_digest: Sha256Digest,
    pub resolution_digest: Option<Sha256Digest>,
    pub previous_hash: Option<Sha256Digest>,
    pub record_hash: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IncidentLedger {
    pub schema_version: String,
    pub events: Vec<IncidentLedgerEvent>,
}

impl Default for IncidentLedger {
    fn default() -> Self {
        Self {
            schema_version: INCIDENT_LEDGER_SCHEMA.into(),
            events: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IncidentLedgerError {
    UnsupportedSchema,
    CapacityExceeded,
    TimestampRegressed { previous: u64, current: u64 },
    DuplicateIncident(Sha256Digest),
    UnknownIncident(Sha256Digest),
    AlreadyResolved(Sha256Digest),
    ResolutionDigestRequired,
    ResolutionDigestForbidden,
    SequenceOverflow,
    VerificationFailed(Vec<IncidentLedgerViolation>),
    Encoding(String),
    EvidenceRollback(&'static str),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IncidentLedgerViolation {
    UnsupportedSchema,
    SequenceMismatch {
        index: usize,
        actual: u64,
        expected: u64,
    },
    TimestampRegressed {
        index: usize,
    },
    PreviousHashMismatch {
        index: usize,
    },
    RecordHashMismatch {
        index: usize,
    },
    DuplicateIncident {
        index: usize,
    },
    UnknownIncidentResolution {
        index: usize,
    },
    DuplicateResolution {
        index: usize,
    },
    ResolutionDigestRequired {
        index: usize,
    },
    ResolutionDigestForbidden {
        index: usize,
    },
    Encoding {
        index: usize,
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IncidentLedgerVerificationReport {
    pub violations: Vec<IncidentLedgerViolation>,
    pub open_incidents: Vec<Sha256Digest>,
    pub resolved_incidents: Vec<Sha256Digest>,
    pub verified_head: Option<Sha256Digest>,
}

impl IncidentLedgerVerificationReport {
    pub fn intact(&self) -> bool {
        self.violations.is_empty()
    }
}

impl IncidentLedger {
    pub fn register(
        &mut self,
        timestamp_unix_s: u64,
        incident: &VerifiedIncidentBundle,
    ) -> Result<Sha256Digest, IncidentLedgerError> {
        self.append(
            timestamp_unix_s,
            IncidentLedgerAction::Registered,
            incident.bundle_digest(),
            None,
        )
    }

    pub fn resolve(
        &mut self,
        timestamp_unix_s: u64,
        incident_digest: Sha256Digest,
        resolution_digest: Sha256Digest,
    ) -> Result<Sha256Digest, IncidentLedgerError> {
        self.append(
            timestamp_unix_s,
            IncidentLedgerAction::Resolved,
            incident_digest,
            Some(resolution_digest),
        )
    }

    pub fn append(
        &mut self,
        timestamp_unix_s: u64,
        action: IncidentLedgerAction,
        incident_digest: Sha256Digest,
        resolution_digest: Option<Sha256Digest>,
    ) -> Result<Sha256Digest, IncidentLedgerError> {
        if self.schema_version != INCIDENT_LEDGER_SCHEMA {
            return Err(IncidentLedgerError::UnsupportedSchema);
        }
        if self.events.len() >= MAX_INCIDENT_LEDGER_EVENTS {
            return Err(IncidentLedgerError::CapacityExceeded);
        }
        if let Some(previous) = self.events.last() {
            if timestamp_unix_s < previous.timestamp_unix_s {
                return Err(IncidentLedgerError::TimestampRegressed {
                    previous: previous.timestamp_unix_s,
                    current: timestamp_unix_s,
                });
            }
        }
        let state = replay_state(&self.events).map_err(IncidentLedgerError::VerificationFailed)?;
        match action {
            IncidentLedgerAction::Registered => {
                if resolution_digest.is_some() {
                    return Err(IncidentLedgerError::ResolutionDigestForbidden);
                }
                if state.contains_key(&incident_digest) {
                    return Err(IncidentLedgerError::DuplicateIncident(incident_digest));
                }
            }
            IncidentLedgerAction::Resolved => {
                if resolution_digest.is_none() {
                    return Err(IncidentLedgerError::ResolutionDigestRequired);
                }
                match state.get(&incident_digest) {
                    None => return Err(IncidentLedgerError::UnknownIncident(incident_digest)),
                    Some(true) => {
                        return Err(IncidentLedgerError::AlreadyResolved(incident_digest));
                    }
                    Some(false) => {}
                }
            }
        }
        let sequence = u64::try_from(self.events.len())
            .ok()
            .and_then(|value| value.checked_add(1))
            .ok_or(IncidentLedgerError::SequenceOverflow)?;
        let previous_hash = self.events.last().map(|event| event.record_hash);
        let record_hash = hash_event_fields(
            sequence,
            timestamp_unix_s,
            action,
            incident_digest,
            resolution_digest,
            previous_hash,
        )?;
        self.events.push(IncidentLedgerEvent {
            sequence,
            timestamp_unix_s,
            action,
            incident_digest,
            resolution_digest,
            previous_hash,
            record_hash,
        });
        Ok(record_hash)
    }

    pub fn verify(&self) -> IncidentLedgerVerificationReport {
        let mut violations = Vec::new();
        if self.schema_version != INCIDENT_LEDGER_SCHEMA {
            violations.push(IncidentLedgerViolation::UnsupportedSchema);
        }
        let mut previous_hash = None;
        let mut previous_timestamp = 0;
        let mut state = BTreeMap::new();
        for (index, event) in self.events.iter().enumerate() {
            let expected_sequence = (index as u64).saturating_add(1);
            if event.sequence != expected_sequence {
                violations.push(IncidentLedgerViolation::SequenceMismatch {
                    index,
                    actual: event.sequence,
                    expected: expected_sequence,
                });
            }
            if index > 0 && event.timestamp_unix_s < previous_timestamp {
                violations.push(IncidentLedgerViolation::TimestampRegressed { index });
            }
            if event.previous_hash != previous_hash {
                violations.push(IncidentLedgerViolation::PreviousHashMismatch { index });
            }
            match event.action {
                IncidentLedgerAction::Registered => {
                    if event.resolution_digest.is_some() {
                        violations
                            .push(IncidentLedgerViolation::ResolutionDigestForbidden { index });
                    }
                    if state.insert(event.incident_digest, false).is_some() {
                        violations.push(IncidentLedgerViolation::DuplicateIncident { index });
                    }
                }
                IncidentLedgerAction::Resolved => {
                    if event.resolution_digest.is_none() {
                        violations
                            .push(IncidentLedgerViolation::ResolutionDigestRequired { index });
                    }
                    match state.get_mut(&event.incident_digest) {
                        None => violations
                            .push(IncidentLedgerViolation::UnknownIncidentResolution { index }),
                        Some(resolved) if *resolved => {
                            violations.push(IncidentLedgerViolation::DuplicateResolution { index })
                        }
                        Some(resolved) => *resolved = true,
                    }
                }
            }
            match hash_event_fields(
                event.sequence,
                event.timestamp_unix_s,
                event.action,
                event.incident_digest,
                event.resolution_digest,
                event.previous_hash,
            ) {
                Ok(expected) if expected != event.record_hash => {
                    violations.push(IncidentLedgerViolation::RecordHashMismatch { index })
                }
                Err(error) => violations.push(IncidentLedgerViolation::Encoding {
                    index,
                    reason: format!("{error:?}"),
                }),
                Ok(_) => {}
            }
            previous_timestamp = event.timestamp_unix_s;
            previous_hash = Some(event.record_hash);
        }
        let open_incidents = state
            .iter()
            .filter_map(|(digest, resolved)| (!*resolved).then_some(*digest))
            .collect();
        let resolved_incidents = state
            .iter()
            .filter_map(|(digest, resolved)| (*resolved).then_some(*digest))
            .collect();
        IncidentLedgerVerificationReport {
            violations,
            open_incidents,
            resolved_incidents,
            verified_head: previous_hash,
        }
    }

    pub fn unresolved_digests(&self) -> Result<Vec<Sha256Digest>, IncidentLedgerError> {
        let report = self.verify();
        if !report.intact() {
            return Err(IncidentLedgerError::VerificationFailed(report.violations));
        }
        Ok(report.open_incidents)
    }

    pub fn head(&self) -> Option<Sha256Digest> {
        self.events.last().map(|event| event.record_hash)
    }

    pub fn digest(&self) -> Result<Sha256Digest, IncidentLedgerError> {
        let report = self.verify();
        if !report.intact() {
            return Err(IncidentLedgerError::VerificationFailed(report.violations));
        }
        let bytes = serde_json::to_vec(self)
            .map_err(|error| IncidentLedgerError::Encoding(error.to_string()))?;
        let mut hasher = Sha256::new();
        hasher.update(b"symthaea.fabrication.incident-ledger-digest.v1\0");
        hasher.update(&bytes);
        Ok(hasher.finalize())
    }

    pub fn verify_successor_of(&self, previous: &Self) -> Result<(), IncidentLedgerError> {
        let previous_report = previous.verify();
        if !previous_report.intact() {
            return Err(IncidentLedgerError::VerificationFailed(
                previous_report.violations,
            ));
        }
        let current_report = self.verify();
        if !current_report.intact() {
            return Err(IncidentLedgerError::VerificationFailed(
                current_report.violations,
            ));
        }
        if self.events.len() < previous.events.len()
            || self.events[..previous.events.len()] != previous.events
        {
            return Err(IncidentLedgerError::EvidenceRollback(
                "incident ledger prefix was removed or changed",
            ));
        }
        Ok(())
    }
}

fn replay_state(
    events: &[IncidentLedgerEvent],
) -> Result<BTreeMap<Sha256Digest, bool>, Vec<IncidentLedgerViolation>> {
    let ledger = IncidentLedger {
        schema_version: INCIDENT_LEDGER_SCHEMA.into(),
        events: events.to_vec(),
    };
    let report = ledger.verify();
    if !report.intact() {
        return Err(report.violations);
    }
    let resolved: BTreeSet<_> = report.resolved_incidents.into_iter().collect();
    let mut state = BTreeMap::new();
    for digest in report.open_incidents {
        state.insert(digest, false);
    }
    for digest in resolved {
        state.insert(digest, true);
    }
    Ok(state)
}

pub fn digest_incident_ledger(
    ledger: &IncidentLedger,
) -> Result<Sha256Digest, IncidentLedgerError> {
    let report = ledger.verify();
    if !report.intact() {
        return Err(IncidentLedgerError::VerificationFailed(report.violations));
    }
    let bytes = serde_json::to_vec(ledger)
        .map_err(|error| IncidentLedgerError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.incident-ledger-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[derive(Serialize)]
struct IncidentLedgerEventBody {
    sequence: u64,
    timestamp_unix_s: u64,
    action: IncidentLedgerAction,
    incident_digest: Sha256Digest,
    resolution_digest: Option<Sha256Digest>,
    previous_hash: Option<Sha256Digest>,
}

fn hash_event_fields(
    sequence: u64,
    timestamp_unix_s: u64,
    action: IncidentLedgerAction,
    incident_digest: Sha256Digest,
    resolution_digest: Option<Sha256Digest>,
    previous_hash: Option<Sha256Digest>,
) -> Result<Sha256Digest, IncidentLedgerError> {
    let bytes = serde_json::to_vec(&IncidentLedgerEventBody {
        sequence,
        timestamp_unix_s,
        action,
        incident_digest,
        resolution_digest,
        previous_hash,
    })
    .map_err(|error| IncidentLedgerError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.incident-ledger-event.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;

    #[test]
    fn duplicate_or_unknown_transitions_fail_closed() {
        let mut ledger = IncidentLedger::default();
        let incident = sha256(b"incident");
        ledger
            .append(100, IncidentLedgerAction::Registered, incident, None)
            .unwrap();
        assert!(matches!(
            ledger.append(101, IncidentLedgerAction::Registered, incident, None),
            Err(IncidentLedgerError::DuplicateIncident(_))
        ));
        assert!(matches!(
            ledger.resolve(102, sha256(b"unknown"), sha256(b"resolution")),
            Err(IncidentLedgerError::UnknownIncident(_))
        ));
    }

    #[test]
    fn unresolved_set_is_derived_from_append_only_evidence() {
        let mut ledger = IncidentLedger::default();
        let first = sha256(b"first");
        let second = sha256(b"second");
        ledger
            .append(100, IncidentLedgerAction::Registered, first, None)
            .unwrap();
        ledger
            .append(101, IncidentLedgerAction::Registered, second, None)
            .unwrap();
        ledger.resolve(102, first, sha256(b"fixed")).unwrap();
        assert_eq!(ledger.unresolved_digests().unwrap(), vec![second]);
    }
}
