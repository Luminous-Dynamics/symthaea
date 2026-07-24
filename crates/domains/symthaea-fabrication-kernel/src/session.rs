// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Persistent anti-replay tracking for timed machine sessions.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::machine::{NegotiatedMachine, TimedMachineSession, digest_timed_machine_session};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const SESSION_TRACKER_SCHEMA: &str = "symthaea.fabrication.session-tracker.v1";
pub const MAX_TRACKED_MACHINES: usize = 1024;
pub const MAX_CONSUMED_NONCES_PER_MACHINE: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct MachineSessionState {
    latest_sequence: u64,
    latest_digest: Sha256Digest,
    accepted_nonce: String,
    expires_at_unix_s: u64,
    consumed_nonces: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineSessionTracker {
    pub schema_version: String,
    machines: BTreeMap<String, MachineSessionState>,
}

impl Default for MachineSessionTracker {
    fn default() -> Self {
        Self {
            schema_version: SESSION_TRACKER_SCHEMA.into(),
            machines: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionTrackingError {
    UnsupportedSchema,
    InvalidSession(String),
    InvalidMachineId,
    InvalidNonce,
    CapacityExceeded,
    SequenceRollback {
        latest: u64,
        proposed: u64,
    },
    SequenceCollision {
        sequence: u64,
    },
    NonceReuse(String),
    SessionNotAccepted,
    SessionSuperseded,
    SessionExpired {
        now_unix_s: u64,
        expires_at_unix_s: u64,
    },
    LegacySessionNotAllowed,
    InvalidTrackerState(String),
    TrackerRollback(&'static str),
    Encoding(String),
}

#[derive(Debug, Clone)]
pub struct MachineSessionLease {
    machine_id: String,
    session_nonce: String,
    session_sequence: u64,
    session_digest: Sha256Digest,
    expires_at_unix_s: u64,
}

impl MachineSessionLease {
    pub fn machine_id(&self) -> &str {
        &self.machine_id
    }
    pub fn session_nonce(&self) -> &str {
        &self.session_nonce
    }
    pub fn session_sequence(&self) -> u64 {
        self.session_sequence
    }
    pub fn session_digest(&self) -> Sha256Digest {
        self.session_digest
    }
    pub fn expires_at_unix_s(&self) -> u64 {
        self.expires_at_unix_s
    }
}

impl MachineSessionTracker {
    /// Accept a fresh signed/authenticated capability advertisement after its
    /// transport authenticity has been established by the caller.
    pub fn accept(
        &mut self,
        session: &TimedMachineSession,
    ) -> Result<Sha256Digest, SessionTrackingError> {
        self.validate()?;
        let machine_id = session.session.capabilities.machine_id.as_str();
        let nonce = session.session.session_nonce.as_str();
        if !canonical(machine_id) {
            return Err(SessionTrackingError::InvalidMachineId);
        }
        if !canonical(nonce) {
            return Err(SessionTrackingError::InvalidNonce);
        }
        let digest =
            digest_timed_machine_session(session).map_err(SessionTrackingError::InvalidSession)?;
        if !self.machines.contains_key(machine_id) && self.machines.len() >= MAX_TRACKED_MACHINES {
            return Err(SessionTrackingError::CapacityExceeded);
        }
        if let Some(state) = self.machines.get(machine_id) {
            if session.session_sequence < state.latest_sequence {
                return Err(SessionTrackingError::SequenceRollback {
                    latest: state.latest_sequence,
                    proposed: session.session_sequence,
                });
            }
            if session.session_sequence == state.latest_sequence {
                if state.latest_digest == digest && state.accepted_nonce == nonce {
                    return Ok(digest);
                }
                return Err(SessionTrackingError::SequenceCollision {
                    sequence: session.session_sequence,
                });
            }
            if state.consumed_nonces.contains(nonce) || state.accepted_nonce == nonce {
                return Err(SessionTrackingError::NonceReuse(nonce.to_string()));
            }
        }
        let consumed_nonces = self
            .machines
            .get(machine_id)
            .map(|state| state.consumed_nonces.clone())
            .unwrap_or_default();
        self.machines.insert(
            machine_id.to_string(),
            MachineSessionState {
                latest_sequence: session.session_sequence,
                latest_digest: digest,
                accepted_nonce: nonce.to_string(),
                expires_at_unix_s: session.expires_at_unix_s,
                consumed_nonces,
            },
        );
        Ok(digest)
    }

    /// Consume a negotiated session exactly once and issue an in-memory lease.
    pub fn consume(
        &mut self,
        machine: &NegotiatedMachine,
        now_unix_s: u64,
    ) -> Result<MachineSessionLease, SessionTrackingError> {
        self.validate()?;
        let Some(window) = machine.session_window() else {
            return Err(SessionTrackingError::LegacySessionNotAllowed);
        };
        let Some(state) = self.machines.get_mut(machine.machine_id()) else {
            return Err(SessionTrackingError::SessionNotAccepted);
        };
        if state.latest_sequence != window.sequence
            || state.latest_digest != window.digest
            || state.accepted_nonce != machine.session_nonce()
        {
            return Err(SessionTrackingError::SessionSuperseded);
        }
        if now_unix_s < window.issued_at_unix_s || now_unix_s >= window.expires_at_unix_s {
            return Err(SessionTrackingError::SessionExpired {
                now_unix_s,
                expires_at_unix_s: window.expires_at_unix_s,
            });
        }
        if state.consumed_nonces.contains(machine.session_nonce()) {
            return Err(SessionTrackingError::NonceReuse(
                machine.session_nonce().to_string(),
            ));
        }
        if state.consumed_nonces.len() >= MAX_CONSUMED_NONCES_PER_MACHINE {
            return Err(SessionTrackingError::CapacityExceeded);
        }
        state
            .consumed_nonces
            .insert(machine.session_nonce().to_string());
        Ok(MachineSessionLease {
            machine_id: machine.machine_id().to_string(),
            session_nonce: machine.session_nonce().to_string(),
            session_sequence: window.sequence,
            session_digest: window.digest,
            expires_at_unix_s: window.expires_at_unix_s,
        })
    }

    pub fn contains_consumed_nonce(&self, machine_id: &str, nonce: &str) -> bool {
        self.machines
            .get(machine_id)
            .is_some_and(|state| state.consumed_nonces.contains(nonce))
    }

    pub fn digest(&self) -> Result<Sha256Digest, SessionTrackingError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| SessionTrackingError::Encoding(error.to_string()))?;
        let mut hasher = Sha256::new();
        hasher.update(b"symthaea.fabrication.session-tracker-digest.v1\0");
        hasher.update(&bytes);
        Ok(hasher.finalize())
    }

    pub fn verify_successor_of(&self, previous: &Self) -> Result<(), SessionTrackingError> {
        previous.validate()?;
        self.validate()?;
        for (machine_id, previous_state) in &previous.machines {
            let Some(current) = self.machines.get(machine_id) else {
                return Err(SessionTrackingError::TrackerRollback(
                    "tracked machine was removed",
                ));
            };
            if current.latest_sequence < previous_state.latest_sequence {
                return Err(SessionTrackingError::TrackerRollback(
                    "latest session sequence regressed",
                ));
            }
            if current.latest_sequence == previous_state.latest_sequence
                && (current.latest_digest != previous_state.latest_digest
                    || current.accepted_nonce != previous_state.accepted_nonce
                    || current.expires_at_unix_s != previous_state.expires_at_unix_s)
            {
                return Err(SessionTrackingError::TrackerRollback(
                    "same session sequence was substituted",
                ));
            }
            if !current
                .consumed_nonces
                .is_superset(&previous_state.consumed_nonces)
            {
                return Err(SessionTrackingError::TrackerRollback(
                    "consumed nonce evidence was removed",
                ));
            }
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), SessionTrackingError> {
        if self.schema_version != SESSION_TRACKER_SCHEMA {
            return Err(SessionTrackingError::UnsupportedSchema);
        }
        if self.machines.len() > MAX_TRACKED_MACHINES {
            return Err(SessionTrackingError::CapacityExceeded);
        }
        for (machine_id, state) in &self.machines {
            if !canonical(machine_id) {
                return Err(SessionTrackingError::InvalidTrackerState(
                    "non-canonical machine identity".into(),
                ));
            }
            if state.latest_sequence == 0 {
                return Err(SessionTrackingError::InvalidTrackerState(
                    "zero session sequence".into(),
                ));
            }
            if !canonical(&state.accepted_nonce) {
                return Err(SessionTrackingError::InvalidTrackerState(
                    "non-canonical accepted nonce".into(),
                ));
            }
            if state.consumed_nonces.len() > MAX_CONSUMED_NONCES_PER_MACHINE {
                return Err(SessionTrackingError::CapacityExceeded);
            }
            if state.consumed_nonces.iter().any(|nonce| !canonical(nonce)) {
                return Err(SessionTrackingError::InvalidTrackerState(
                    "non-canonical consumed nonce".into(),
                ));
            }
        }
        Ok(())
    }
}

fn canonical(value: &str) -> bool {
    !value.trim().is_empty() && value == value.trim() && value.len() <= 256
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::machine::{
        MachineCapabilities, MachineProfile, MachineSession, TimedMachineSession,
        negotiate_machine_profile_at,
    };

    fn timed(sequence: u64, nonce: &str) -> (MachineProfile, TimedMachineSession) {
        let profile = MachineProfile::default();
        let session = TimedMachineSession::new(
            MachineSession {
                session_nonce: nonce.into(),
                capabilities: MachineCapabilities::from_profile("machine-1", &profile),
            },
            sequence,
            100,
            200,
        );
        (profile, session)
    }

    #[test]
    fn accepted_session_can_be_consumed_only_once() {
        let (profile, session) = timed(1, "nonce-1");
        let machine = negotiate_machine_profile_at(&profile, session.clone(), 150).unwrap();
        let mut tracker = MachineSessionTracker::default();
        tracker.accept(&session).unwrap();
        tracker.consume(&machine, 150).unwrap();
        assert!(matches!(
            tracker.consume(&machine, 150),
            Err(SessionTrackingError::NonceReuse(_))
        ));
    }

    #[test]
    fn sequence_rollback_and_nonce_reuse_are_rejected() {
        let (_, first) = timed(2, "nonce-2");
        let (_, rollback) = timed(1, "nonce-1");
        let (_, reused) = timed(3, "nonce-2");
        let mut tracker = MachineSessionTracker::default();
        tracker.accept(&first).unwrap();
        assert!(matches!(
            tracker.accept(&rollback),
            Err(SessionTrackingError::SequenceRollback { .. })
        ));
        assert!(matches!(
            tracker.accept(&reused),
            Err(SessionTrackingError::NonceReuse(_))
        ));
    }

    #[test]
    fn tracker_digest_changes_when_nonce_is_consumed() {
        let (profile, session) = timed(1, "nonce-1");
        let machine = negotiate_machine_profile_at(&profile, session.clone(), 150).unwrap();
        let mut tracker = MachineSessionTracker::default();
        tracker.accept(&session).unwrap();
        let before = tracker.digest().unwrap();
        tracker.consume(&machine, 150).unwrap();
        assert_ne!(before, tracker.digest().unwrap());
    }
}
