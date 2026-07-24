// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Durable, hash-chained gateway authority snapshots.

use crate::audit::{AuditJournal, digest_audit_journal};
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::gateway_consensus_tracker::{GatewayConsensusTracker, GatewayConsensusTrackingError};
use crate::incident_ledger::{IncidentLedger, IncidentLedgerError};
use crate::operator_command_tracker::{OperatorCommandTracker, OperatorCommandTrackingError};
use crate::session::{MachineSessionTracker, SessionTrackingError};
use crate::submission_ledger::{SubmissionLedger, SubmissionLedgerError};
use crate::telemetry_tracker::{MachineTelemetryTracker, TelemetryTrackingError};
use crate::trust::{TrustSnapshot, TrustSnapshotError, digest_trust_snapshot};
use serde::{Deserialize, Serialize};

pub const GATEWAY_STATE_SCHEMA: &str = "symthaea.fabrication.gateway-state.v3";
pub const GATEWAY_STATE_ENVELOPE_SCHEMA: &str = "symthaea.fabrication.gateway-state-envelope.v1";
pub const MAX_GATEWAY_STATE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FabricationGatewayState {
    pub schema_version: String,
    pub generation: u64,
    pub committed_at_unix_ms: u64,
    pub previous_state_digest: Option<Sha256Digest>,
    pub trust_snapshot: TrustSnapshot,
    pub audit_journal: AuditJournal,
    pub session_tracker: MachineSessionTracker,
    pub telemetry_tracker: MachineTelemetryTracker,
    pub submission_ledger: SubmissionLedger,
    pub operator_command_tracker: OperatorCommandTracker,
    pub gateway_consensus_tracker: GatewayConsensusTracker,
    pub incident_ledger: IncidentLedger,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayStateEnvelope {
    pub schema_version: String,
    pub state: FabricationGatewayState,
    pub state_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatewayStateError {
    UnsupportedSchema,
    UnsupportedEnvelopeSchema,
    GenerationZero,
    InvalidPreviousDigest,
    GenerationOverflow,
    CommitTimeRegressed { previous: u64, current: u64 },
    TrustSnapshot(TrustSnapshotError),
    TrustSnapshotNotFreshAtCommit,
    AuditJournal(String),
    SessionTracker(SessionTrackingError),
    TelemetryTracker(TelemetryTrackingError),
    SubmissionLedger(SubmissionLedgerError),
    OperatorCommandTracker(OperatorCommandTrackingError),
    GatewayConsensusTracker(GatewayConsensusTrackingError),
    IncidentLedger(IncidentLedgerError),
    StateTooLarge { actual: usize, maximum: usize },
    DigestMismatch,
    ChainDigestMismatch,
    EvidenceRollback(&'static str),
    Encoding(String),
}

impl FabricationGatewayState {
    #[allow(clippy::too_many_arguments)]
    pub fn genesis(
        committed_at_unix_ms: u64,
        trust_snapshot: TrustSnapshot,
        audit_journal: AuditJournal,
        session_tracker: MachineSessionTracker,
        telemetry_tracker: MachineTelemetryTracker,
        submission_ledger: SubmissionLedger,
        operator_command_tracker: OperatorCommandTracker,
        gateway_consensus_tracker: GatewayConsensusTracker,
        incident_ledger: IncidentLedger,
    ) -> Result<Self, GatewayStateError> {
        let state = Self {
            schema_version: GATEWAY_STATE_SCHEMA.into(),
            generation: 1,
            committed_at_unix_ms,
            previous_state_digest: None,
            trust_snapshot,
            audit_journal,
            session_tracker,
            telemetry_tracker,
            submission_ledger,
            operator_command_tracker,
            gateway_consensus_tracker,
            incident_ledger,
        };
        state.validate()?;
        Ok(state)
    }

    /// Build the next snapshot from explicit updated components and bind it to
    /// the exact digest of the preceding snapshot.
    #[allow(clippy::too_many_arguments)]
    pub fn successor(
        previous: &Self,
        committed_at_unix_ms: u64,
        trust_snapshot: TrustSnapshot,
        audit_journal: AuditJournal,
        session_tracker: MachineSessionTracker,
        telemetry_tracker: MachineTelemetryTracker,
        submission_ledger: SubmissionLedger,
        operator_command_tracker: OperatorCommandTracker,
        gateway_consensus_tracker: GatewayConsensusTracker,
        incident_ledger: IncidentLedger,
    ) -> Result<Self, GatewayStateError> {
        previous.validate()?;
        if committed_at_unix_ms < previous.committed_at_unix_ms {
            return Err(GatewayStateError::CommitTimeRegressed {
                previous: previous.committed_at_unix_ms,
                current: committed_at_unix_ms,
            });
        }
        let generation = previous
            .generation
            .checked_add(1)
            .ok_or(GatewayStateError::GenerationOverflow)?;
        let state = Self {
            schema_version: GATEWAY_STATE_SCHEMA.into(),
            generation,
            committed_at_unix_ms,
            previous_state_digest: Some(previous.digest()?),
            trust_snapshot,
            audit_journal,
            session_tracker,
            telemetry_tracker,
            submission_ledger,
            operator_command_tracker,
            gateway_consensus_tracker,
            incident_ledger,
        };
        verify_gateway_state_successor(previous, &state)?;
        Ok(state)
    }

    pub fn validate(&self) -> Result<(), GatewayStateError> {
        if self.schema_version != GATEWAY_STATE_SCHEMA {
            return Err(GatewayStateError::UnsupportedSchema);
        }
        if self.generation == 0 {
            return Err(GatewayStateError::GenerationZero);
        }
        if (self.generation == 1) != self.previous_state_digest.is_none() {
            return Err(GatewayStateError::InvalidPreviousDigest);
        }
        self.trust_snapshot
            .validate()
            .map_err(GatewayStateError::TrustSnapshot)?;
        if !self
            .trust_snapshot
            .is_fresh_at(self.committed_at_unix_ms / 1_000)
        {
            return Err(GatewayStateError::TrustSnapshotNotFreshAtCommit);
        }
        let audit_report = self.audit_journal.verify();
        if !audit_report.intact() {
            return Err(GatewayStateError::AuditJournal(format!(
                "{:?}",
                audit_report.violations
            )));
        }
        self.session_tracker
            .validate()
            .map_err(GatewayStateError::SessionTracker)?;
        self.telemetry_tracker
            .validate()
            .map_err(GatewayStateError::TelemetryTracker)?;
        let submission_report = self.submission_ledger.verify();
        if !submission_report.intact() {
            return Err(GatewayStateError::SubmissionLedger(
                SubmissionLedgerError::VerificationFailed(submission_report.violations),
            ));
        }
        self.operator_command_tracker
            .validate()
            .map_err(GatewayStateError::OperatorCommandTracker)?;
        self.gateway_consensus_tracker
            .validate()
            .map_err(GatewayStateError::GatewayConsensusTracker)?;
        let incident_report = self.incident_ledger.verify();
        if !incident_report.intact() {
            return Err(GatewayStateError::IncidentLedger(
                IncidentLedgerError::VerificationFailed(incident_report.violations),
            ));
        }
        let encoded = serde_json::to_vec(self)
            .map_err(|error| GatewayStateError::Encoding(error.to_string()))?;
        if encoded.len() > MAX_GATEWAY_STATE_BYTES {
            return Err(GatewayStateError::StateTooLarge {
                actual: encoded.len(),
                maximum: MAX_GATEWAY_STATE_BYTES,
            });
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Sha256Digest, GatewayStateError> {
        self.validate_without_size_recursion()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| GatewayStateError::Encoding(error.to_string()))?;
        if bytes.len() > MAX_GATEWAY_STATE_BYTES {
            return Err(GatewayStateError::StateTooLarge {
                actual: bytes.len(),
                maximum: MAX_GATEWAY_STATE_BYTES,
            });
        }
        let mut hasher = Sha256::new();
        hasher.update(b"symthaea.fabrication.gateway-state-digest.v1\0");
        hasher.update(&bytes);
        Ok(hasher.finalize())
    }

    pub fn evidence_digests(&self) -> Result<GatewayEvidenceDigests, GatewayStateError> {
        Ok(GatewayEvidenceDigests {
            trust_snapshot: digest_trust_snapshot(&self.trust_snapshot)
                .map_err(GatewayStateError::TrustSnapshot)?,
            audit_journal: digest_audit_journal(&self.audit_journal)
                .map_err(|error| GatewayStateError::AuditJournal(format!("{error:?}")))?,
            audit_head: self.audit_journal.head(),
            session_tracker: self
                .session_tracker
                .digest()
                .map_err(GatewayStateError::SessionTracker)?,
            telemetry_tracker: self
                .telemetry_tracker
                .digest()
                .map_err(GatewayStateError::TelemetryTracker)?,
            submission_ledger: self
                .submission_ledger
                .digest()
                .map_err(GatewayStateError::SubmissionLedger)?,
            submission_head: self.submission_ledger.head(),
            operator_command_tracker: self
                .operator_command_tracker
                .digest()
                .map_err(GatewayStateError::OperatorCommandTracker)?,
            gateway_consensus_tracker: self
                .gateway_consensus_tracker
                .digest()
                .map_err(GatewayStateError::GatewayConsensusTracker)?,
            incident_ledger: self
                .incident_ledger
                .digest()
                .map_err(GatewayStateError::IncidentLedger)?,
            incident_head: self.incident_ledger.head(),
        })
    }

    fn validate_without_size_recursion(&self) -> Result<(), GatewayStateError> {
        if self.schema_version != GATEWAY_STATE_SCHEMA {
            return Err(GatewayStateError::UnsupportedSchema);
        }
        if self.generation == 0 {
            return Err(GatewayStateError::GenerationZero);
        }
        if (self.generation == 1) != self.previous_state_digest.is_none() {
            return Err(GatewayStateError::InvalidPreviousDigest);
        }
        self.trust_snapshot
            .validate()
            .map_err(GatewayStateError::TrustSnapshot)?;
        if !self
            .trust_snapshot
            .is_fresh_at(self.committed_at_unix_ms / 1_000)
        {
            return Err(GatewayStateError::TrustSnapshotNotFreshAtCommit);
        }
        let audit_report = self.audit_journal.verify();
        if !audit_report.intact() {
            return Err(GatewayStateError::AuditJournal(format!(
                "{:?}",
                audit_report.violations
            )));
        }
        self.session_tracker
            .validate()
            .map_err(GatewayStateError::SessionTracker)?;
        self.telemetry_tracker
            .validate()
            .map_err(GatewayStateError::TelemetryTracker)?;
        let submission_report = self.submission_ledger.verify();
        if !submission_report.intact() {
            return Err(GatewayStateError::SubmissionLedger(
                SubmissionLedgerError::VerificationFailed(submission_report.violations),
            ));
        }
        self.operator_command_tracker
            .validate()
            .map_err(GatewayStateError::OperatorCommandTracker)?;
        self.gateway_consensus_tracker
            .validate()
            .map_err(GatewayStateError::GatewayConsensusTracker)?;
        let incident_report = self.incident_ledger.verify();
        if !incident_report.intact() {
            return Err(GatewayStateError::IncidentLedger(
                IncidentLedgerError::VerificationFailed(incident_report.violations),
            ));
        }
        Ok(())
    }
}

impl GatewayStateEnvelope {
    pub fn seal(state: FabricationGatewayState) -> Result<Self, GatewayStateError> {
        let state_digest = state.digest()?;
        Ok(Self {
            schema_version: GATEWAY_STATE_ENVELOPE_SCHEMA.into(),
            state,
            state_digest,
        })
    }

    pub fn open(self) -> Result<FabricationGatewayState, GatewayStateError> {
        if self.schema_version != GATEWAY_STATE_ENVELOPE_SCHEMA {
            return Err(GatewayStateError::UnsupportedEnvelopeSchema);
        }
        if self.state.digest()? != self.state_digest {
            return Err(GatewayStateError::DigestMismatch);
        }
        Ok(self.state)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, GatewayStateError> {
        if self.schema_version != GATEWAY_STATE_ENVELOPE_SCHEMA {
            return Err(GatewayStateError::UnsupportedEnvelopeSchema);
        }
        if self.state.digest()? != self.state_digest {
            return Err(GatewayStateError::DigestMismatch);
        }
        let bytes = serde_json::to_vec(self)
            .map_err(|error| GatewayStateError::Encoding(error.to_string()))?;
        if bytes.len() > MAX_GATEWAY_STATE_BYTES {
            return Err(GatewayStateError::StateTooLarge {
                actual: bytes.len(),
                maximum: MAX_GATEWAY_STATE_BYTES,
            });
        }
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, GatewayStateError> {
        if bytes.len() > MAX_GATEWAY_STATE_BYTES {
            return Err(GatewayStateError::StateTooLarge {
                actual: bytes.len(),
                maximum: MAX_GATEWAY_STATE_BYTES,
            });
        }
        let envelope: Self = serde_json::from_slice(bytes)
            .map_err(|error| GatewayStateError::Encoding(error.to_string()))?;
        if envelope.schema_version != GATEWAY_STATE_ENVELOPE_SCHEMA {
            return Err(GatewayStateError::UnsupportedEnvelopeSchema);
        }
        if envelope.state.digest()? != envelope.state_digest {
            return Err(GatewayStateError::DigestMismatch);
        }
        Ok(envelope)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatewayEvidenceDigests {
    pub trust_snapshot: Sha256Digest,
    pub audit_journal: Sha256Digest,
    pub audit_head: Option<Sha256Digest>,
    pub session_tracker: Sha256Digest,
    pub telemetry_tracker: Sha256Digest,
    pub submission_ledger: Sha256Digest,
    pub submission_head: Option<Sha256Digest>,
    pub operator_command_tracker: Sha256Digest,
    pub gateway_consensus_tracker: Sha256Digest,
    pub incident_ledger: Sha256Digest,
    pub incident_head: Option<Sha256Digest>,
}

pub fn verify_gateway_state_successor(
    previous: &FabricationGatewayState,
    current: &FabricationGatewayState,
) -> Result<(), GatewayStateError> {
    previous.validate()?;
    current.validate()?;
    if current.generation != previous.generation.saturating_add(1) {
        return Err(GatewayStateError::ChainDigestMismatch);
    }
    if current.committed_at_unix_ms < previous.committed_at_unix_ms {
        return Err(GatewayStateError::CommitTimeRegressed {
            previous: previous.committed_at_unix_ms,
            current: current.committed_at_unix_ms,
        });
    }
    if current.previous_state_digest != Some(previous.digest()?) {
        return Err(GatewayStateError::ChainDigestMismatch);
    }
    let previous_trust_digest = digest_trust_snapshot(&previous.trust_snapshot)
        .map_err(GatewayStateError::TrustSnapshot)?;
    let current_trust_digest =
        digest_trust_snapshot(&current.trust_snapshot).map_err(GatewayStateError::TrustSnapshot)?;
    if current.trust_snapshot.sequence < previous.trust_snapshot.sequence {
        return Err(GatewayStateError::EvidenceRollback(
            "trust snapshot sequence regressed",
        ));
    }
    if current.trust_snapshot.sequence == previous.trust_snapshot.sequence
        && current_trust_digest != previous_trust_digest
    {
        return Err(GatewayStateError::EvidenceRollback(
            "same trust snapshot sequence was substituted",
        ));
    }
    if current.audit_journal.events.len() < previous.audit_journal.events.len()
        || current.audit_journal.events[..previous.audit_journal.events.len()]
            != previous.audit_journal.events
    {
        return Err(GatewayStateError::EvidenceRollback(
            "audit journal prefix was removed or changed",
        ));
    }
    current
        .session_tracker
        .verify_successor_of(&previous.session_tracker)
        .map_err(GatewayStateError::SessionTracker)?;
    current
        .telemetry_tracker
        .verify_successor_of(&previous.telemetry_tracker)
        .map_err(GatewayStateError::TelemetryTracker)?;
    current
        .submission_ledger
        .verify_successor_of(&previous.submission_ledger)
        .map_err(GatewayStateError::SubmissionLedger)?;
    current
        .operator_command_tracker
        .verify_successor_of(&previous.operator_command_tracker)
        .map_err(GatewayStateError::OperatorCommandTracker)?;
    current
        .gateway_consensus_tracker
        .verify_successor_of(&previous.gateway_consensus_tracker)
        .map_err(GatewayStateError::GatewayConsensusTracker)?;
    current
        .incident_ledger
        .verify_successor_of(&previous.incident_ledger)
        .map_err(GatewayStateError::IncidentLedger)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attestation::SignatureAlgorithm;
    use crate::gateway_consensus_tracker::GatewayConsensusTracker;
    use crate::incident_ledger::IncidentLedger;
    use crate::operator_command_tracker::OperatorCommandTracker;
    use crate::trust::{KeyLifecycleStatus, KeyTrustRecord, KeyUsage};
    use std::collections::BTreeSet;

    fn trust() -> TrustSnapshot {
        TrustSnapshot::new(
            1,
            100,
            1_000,
            vec![KeyTrustRecord {
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "root".into(),
                not_before_unix_s: 100,
                not_after_unix_s: None,
                status: KeyLifecycleStatus::Active,
                usages: BTreeSet::from([KeyUsage::FabricationManifest]),
            }],
        )
        .unwrap()
    }

    fn genesis() -> FabricationGatewayState {
        FabricationGatewayState::genesis(
            500_000,
            trust(),
            AuditJournal::default(),
            MachineSessionTracker::default(),
            MachineTelemetryTracker::default(),
            SubmissionLedger::default(),
            OperatorCommandTracker::default(),
            GatewayConsensusTracker::default(),
            IncidentLedger::default(),
        )
        .unwrap()
    }

    #[test]
    fn envelope_detects_state_tampering() {
        let mut envelope = GatewayStateEnvelope::seal(genesis()).unwrap();
        envelope.state.committed_at_unix_ms += 1;
        assert!(matches!(
            envelope.open(),
            Err(GatewayStateError::DigestMismatch)
        ));
    }

    #[test]
    fn successor_binds_exact_previous_digest() {
        let first = genesis();
        let second = FabricationGatewayState::successor(
            &first,
            500_001,
            first.trust_snapshot.clone(),
            first.audit_journal.clone(),
            first.session_tracker.clone(),
            first.telemetry_tracker.clone(),
            first.submission_ledger.clone(),
            first.operator_command_tracker.clone(),
            first.gateway_consensus_tracker.clone(),
            first.incident_ledger.clone(),
        )
        .unwrap();
        verify_gateway_state_successor(&first, &second).unwrap();
        let mut substituted = second.clone();
        substituted.previous_state_digest = Some(Sha256Digest([7; 32]));
        assert!(verify_gateway_state_successor(&first, &substituted).is_err());
    }

    #[test]
    fn serialized_envelope_round_trips() {
        let envelope = GatewayStateEnvelope::seal(genesis()).unwrap();
        let bytes = envelope.to_bytes().unwrap();
        let decoded = GatewayStateEnvelope::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, envelope);
    }
}
