// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Persistent anti-replay and terminal-state tracking for verified operator commands.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::operator_command::{OperatorCommandKind, VerifiedOperatorCommand};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const OPERATOR_COMMAND_TRACKER_SCHEMA: &str =
    "symthaea.fabrication.operator-command-tracker.v1";
pub const MAX_OPERATOR_COMMAND_STREAMS: usize = 100_000;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
struct OperatorCommandStreamId {
    manifest_digest: Sha256Digest,
    machine_id: String,
    session_digest: Sha256Digest,
    printer_job_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperatorExecutionState {
    Running,
    Paused,
    Cancelled,
    EmergencyStopped,
}

impl OperatorExecutionState {
    pub fn terminal(self) -> bool {
        matches!(self, Self::Cancelled | Self::EmergencyStopped)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct OperatorCommandStreamState {
    latest_sequence: u64,
    latest_digest: Sha256Digest,
    latest_issued_at_unix_ms: u64,
    execution_state: OperatorExecutionState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorCommandTracker {
    pub schema_version: String,
    streams: BTreeMap<OperatorCommandStreamId, OperatorCommandStreamState>,
}

impl Default for OperatorCommandTracker {
    fn default() -> Self {
        Self {
            schema_version: OPERATOR_COMMAND_TRACKER_SCHEMA.into(),
            streams: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperatorCommandTrackingError {
    UnsupportedSchema,
    CapacityExceeded,
    SequenceRollback { latest: u64, proposed: u64 },
    SequenceCollision { sequence: u64 },
    IssuedAtRegressed { latest: u64, proposed: u64 },
    ResumeWithoutPause,
    TerminalState { state: OperatorExecutionState },
    Encoding(String),
    EvidenceRollback(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppliedOperatorCommand {
    pub command_digest: Sha256Digest,
    pub execution_state: OperatorExecutionState,
    pub idempotent_replay: bool,
}

impl OperatorCommandTracker {
    pub fn validate(&self) -> Result<(), OperatorCommandTrackingError> {
        if self.schema_version != OPERATOR_COMMAND_TRACKER_SCHEMA {
            return Err(OperatorCommandTrackingError::UnsupportedSchema);
        }
        if self.streams.len() > MAX_OPERATOR_COMMAND_STREAMS {
            return Err(OperatorCommandTrackingError::CapacityExceeded);
        }
        if self
            .streams
            .values()
            .any(|state| state.latest_sequence == 0)
        {
            return Err(OperatorCommandTrackingError::SequenceRollback {
                latest: 1,
                proposed: 0,
            });
        }
        Ok(())
    }

    pub fn apply(
        &mut self,
        verified: &VerifiedOperatorCommand,
    ) -> Result<AppliedOperatorCommand, OperatorCommandTrackingError> {
        self.validate()?;
        let command = verified.command();
        let stream_id = OperatorCommandStreamId {
            manifest_digest: command.manifest_digest,
            machine_id: command.machine_id.clone(),
            session_digest: command.session_digest,
            printer_job_id: command.printer_job_id.clone(),
        };
        if !self.streams.contains_key(&stream_id)
            && self.streams.len() >= MAX_OPERATOR_COMMAND_STREAMS
        {
            return Err(OperatorCommandTrackingError::CapacityExceeded);
        }
        if let Some(current) = self.streams.get(&stream_id) {
            if command.command_sequence < current.latest_sequence {
                return Err(OperatorCommandTrackingError::SequenceRollback {
                    latest: current.latest_sequence,
                    proposed: command.command_sequence,
                });
            }
            if command.command_sequence == current.latest_sequence {
                if verified.command_digest() == current.latest_digest {
                    return Ok(AppliedOperatorCommand {
                        command_digest: current.latest_digest,
                        execution_state: current.execution_state,
                        idempotent_replay: true,
                    });
                }
                return Err(OperatorCommandTrackingError::SequenceCollision {
                    sequence: command.command_sequence,
                });
            }
            if command.issued_at_unix_ms < current.latest_issued_at_unix_ms {
                return Err(OperatorCommandTrackingError::IssuedAtRegressed {
                    latest: current.latest_issued_at_unix_ms,
                    proposed: command.issued_at_unix_ms,
                });
            }
            if current.execution_state.terminal() {
                return Err(OperatorCommandTrackingError::TerminalState {
                    state: current.execution_state,
                });
            }
            if command.kind == OperatorCommandKind::Resume
                && current.execution_state != OperatorExecutionState::Paused
            {
                return Err(OperatorCommandTrackingError::ResumeWithoutPause);
            }
        } else if command.kind == OperatorCommandKind::Resume {
            return Err(OperatorCommandTrackingError::ResumeWithoutPause);
        }

        let execution_state = match command.kind {
            OperatorCommandKind::Pause => OperatorExecutionState::Paused,
            OperatorCommandKind::Resume => OperatorExecutionState::Running,
            OperatorCommandKind::Cancel => OperatorExecutionState::Cancelled,
            OperatorCommandKind::EmergencyStop => OperatorExecutionState::EmergencyStopped,
        };
        self.streams.insert(
            stream_id,
            OperatorCommandStreamState {
                latest_sequence: command.command_sequence,
                latest_digest: verified.command_digest(),
                latest_issued_at_unix_ms: command.issued_at_unix_ms,
                execution_state,
            },
        );
        Ok(AppliedOperatorCommand {
            command_digest: verified.command_digest(),
            execution_state,
            idempotent_replay: false,
        })
    }

    pub fn stream_state(
        &self,
        manifest_digest: Sha256Digest,
        machine_id: &str,
        session_digest: Sha256Digest,
        printer_job_id: &str,
    ) -> Option<OperatorExecutionState> {
        self.streams
            .get(&OperatorCommandStreamId {
                manifest_digest,
                machine_id: machine_id.to_string(),
                session_digest,
                printer_job_id: printer_job_id.to_string(),
            })
            .map(|state| state.execution_state)
    }

    pub fn digest(&self) -> Result<Sha256Digest, OperatorCommandTrackingError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| OperatorCommandTrackingError::Encoding(error.to_string()))?;
        let mut hasher = Sha256::new();
        hasher.update(b"symthaea.fabrication.operator-command-tracker-digest.v1\0");
        hasher.update(&bytes);
        Ok(hasher.finalize())
    }

    pub fn verify_successor_of(&self, previous: &Self) -> Result<(), OperatorCommandTrackingError> {
        previous.validate()?;
        self.validate()?;
        for (stream_id, prior) in &previous.streams {
            let Some(current) = self.streams.get(stream_id) else {
                return Err(OperatorCommandTrackingError::EvidenceRollback(
                    "operator command stream disappeared",
                ));
            };
            if current.latest_sequence < prior.latest_sequence {
                return Err(OperatorCommandTrackingError::EvidenceRollback(
                    "operator command sequence regressed",
                ));
            }
            if current.latest_sequence == prior.latest_sequence && current != prior {
                return Err(OperatorCommandTrackingError::EvidenceRollback(
                    "operator command stream was substituted at the same sequence",
                ));
            }
            if prior.execution_state.terminal() && current.execution_state != prior.execution_state
            {
                return Err(OperatorCommandTrackingError::EvidenceRollback(
                    "terminal operator state was cleared",
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attestation::SignatureAlgorithm;
    use crate::crypto_digest::sha256;
    use crate::operator_command::{
        OperatorCommand, OperatorCommandExpectation, OperatorCommandPolicy, OperatorCommandSigner,
        OperatorCommandVerifier, sign_operator_command, verify_operator_command,
    };
    use crate::trust::{KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot};
    use std::collections::BTreeSet;

    struct Provider;
    impl OperatorCommandSigner for Provider {
        fn algorithm(&self) -> SignatureAlgorithm {
            SignatureAlgorithm::Ed25519
        }
        fn key_id(&self) -> &str {
            "operator"
        }
        fn sign_operator_command(&self, message: &[u8]) -> Result<Vec<u8>, String> {
            Ok(sha256(message).0.to_vec())
        }
    }
    impl OperatorCommandVerifier for Provider {
        fn verify_operator_command(
            &self,
            _algorithm: &SignatureAlgorithm,
            _key_id: &str,
            message: &[u8],
            signature: &[u8],
        ) -> Result<bool, String> {
            Ok(signature == sha256(message).0.as_slice())
        }
    }

    fn verified(sequence: u64, kind: OperatorCommandKind) -> VerifiedOperatorCommand {
        let snapshot = TrustSnapshot::new(
            1,
            100,
            1_000,
            vec![KeyTrustRecord {
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "operator".into(),
                not_before_unix_s: 100,
                not_after_unix_s: None,
                status: KeyLifecycleStatus::Active,
                usages: BTreeSet::from([KeyUsage::OperatorCommand]),
            }],
        )
        .unwrap();
        let command = OperatorCommand::new(
            sha256(b"manifest"),
            "machine",
            sha256(b"session"),
            "job",
            sequence,
            500_000 + sequence,
            510_000 + sequence,
            kind,
            "test command",
        )
        .unwrap();
        let signed = sign_operator_command(command, &[&Provider]).unwrap();
        verify_operator_command(
            signed,
            &OperatorCommandPolicy::default(),
            OperatorCommandExpectation {
                manifest_digest: sha256(b"manifest"),
                machine_id: "machine",
                session_digest: sha256(b"session"),
                printer_job_id: "job",
                now_unix_ms: 501_000,
                trust_snapshot: &snapshot,
            },
            &Provider,
        )
        .unwrap()
    }

    #[test]
    fn pause_then_resume_is_allowed() {
        let mut tracker = OperatorCommandTracker::default();
        tracker
            .apply(&verified(1, OperatorCommandKind::Pause))
            .unwrap();
        let result = tracker
            .apply(&verified(2, OperatorCommandKind::Resume))
            .unwrap();
        assert_eq!(result.execution_state, OperatorExecutionState::Running);
    }

    #[test]
    fn terminal_commands_cannot_be_cleared() {
        let mut tracker = OperatorCommandTracker::default();
        tracker
            .apply(&verified(1, OperatorCommandKind::Cancel))
            .unwrap();
        assert!(matches!(
            tracker.apply(&verified(2, OperatorCommandKind::Resume)),
            Err(OperatorCommandTrackingError::TerminalState { .. })
        ));
    }
}
