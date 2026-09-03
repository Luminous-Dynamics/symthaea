// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded integrity chain for operational authority events.
//!
//! Production deployments should provide a cryptographic implementation of
//! [`AuditDigestProvider`]. The deterministic provider included here exists for
//! reproducible tests and continuity checking; it is explicitly not a
//! cryptographic authenticity mechanism.

use crate::update_control::ArtifactDigest;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditEvent {
    OperatorCommand {
        operator_id: u64,
        proposal_id: u64,
        command_code: u16,
        accepted: bool,
    },
    OperatorConstraint {
        constraint_code: u16,
    },
    UpdateTransition {
        release_id: u64,
        state_code: u16,
    },
    WatchdogTransition {
        from_code: u16,
        to_code: u16,
    },
    CheckpointRestored {
        generation: u64,
    },
    ActuatorServiceAuthorization {
        operator_id: u64,
        service_proposal_id: u64,
        actuator_code: u16,
        accepted: bool,
    },
    ActuatorServiceTransition {
        service_proposal_id: u64,
        actuator_code: u16,
        state_code: u16,
    },
}

impl AuditEvent {
    fn canonical_words(self) -> [u64; 5] {
        match self {
            Self::OperatorCommand {
                operator_id,
                proposal_id,
                command_code,
                accepted,
            } => [
                1,
                operator_id,
                proposal_id,
                command_code as u64,
                accepted as u64,
            ],
            Self::OperatorConstraint { constraint_code } => [2, constraint_code as u64, 0, 0, 0],
            Self::UpdateTransition {
                release_id,
                state_code,
            } => [3, release_id, state_code as u64, 0, 0],
            Self::WatchdogTransition { from_code, to_code } => {
                [4, from_code as u64, to_code as u64, 0, 0]
            }
            Self::CheckpointRestored { generation } => [5, generation, 0, 0, 0],
            Self::ActuatorServiceAuthorization {
                operator_id,
                service_proposal_id,
                actuator_code,
                accepted,
            } => [
                6,
                operator_id,
                service_proposal_id,
                actuator_code as u64,
                accepted as u64,
            ],
            Self::ActuatorServiceTransition {
                service_proposal_id,
                actuator_code,
                state_code,
            } => [
                7,
                service_proposal_id,
                actuator_code as u64,
                state_code as u64,
                0,
            ],
        }
    }
}

pub trait AuditDigestProvider {
    fn digest(&self, previous: ArtifactDigest, sequence: u64, event: AuditEvent) -> ArtifactDigest;
}

/// Deterministic continuity digest for tests and offline evidence comparison.
/// This is not collision resistant and must not be used as a signature or
/// adversarial tamper-proofing mechanism.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeterministicAuditDigest;

impl AuditDigestProvider for DeterministicAuditDigest {
    fn digest(&self, previous: ArtifactDigest, sequence: u64, event: AuditEvent) -> ArtifactDigest {
        let mut lanes = [
            0x243f_6a88_85a3_08d3u64,
            0x1319_8a2e_0370_7344u64,
            0xa409_3822_299f_31d0u64,
            0x082e_fa98_ec4e_6c89u64,
        ];
        for (index, chunk) in previous.0.chunks_exact(8).enumerate() {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(chunk);
            lanes[index] ^= u64::from_le_bytes(bytes);
        }
        let mut words = [0u64; 7];
        words[0] = sequence;
        words[1..6].copy_from_slice(&event.canonical_words());
        words[6] = sequence.rotate_left(29) ^ 0x9e37_79b9_7f4a_7c15;
        for (round, word) in words.into_iter().enumerate() {
            for (lane_index, lane) in lanes.iter_mut().enumerate() {
                let rotation = ((round * 11 + lane_index * 17) % 63 + 1) as u32;
                *lane ^= word
                    .wrapping_add((lane_index as u64 + 1) * 0x9e37_79b9)
                    .rotate_left(rotation);
                *lane = lane.wrapping_mul(0xbf58_476d_1ce4_e5b9).rotate_left(23) ^ (*lane >> 31);
            }
        }
        let mut output = [0u8; 32];
        for (index, lane) in lanes.into_iter().enumerate() {
            output[index * 8..(index + 1) * 8].copy_from_slice(&lane.to_le_bytes());
        }
        ArtifactDigest(output)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditRecord {
    pub sequence: u64,
    pub previous: ArtifactDigest,
    pub digest: ArtifactDigest,
    pub event: AuditEvent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditChainError {
    SequenceGap,
    PreviousDigestMismatch,
    DigestMismatch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLedger {
    capacity: usize,
    records: VecDeque<AuditRecord>,
    chain_head: ArtifactDigest,
    next_sequence: u64,
    dropped_records: u64,
}

impl AuditLedger {
    pub fn new(capacity: usize, genesis: ArtifactDigest) -> Self {
        Self {
            capacity: capacity.max(1),
            records: VecDeque::with_capacity(capacity.max(1)),
            chain_head: genesis,
            next_sequence: 1,
            dropped_records: 0,
        }
    }

    pub fn append(
        &mut self,
        provider: &impl AuditDigestProvider,
        event: AuditEvent,
    ) -> AuditRecord {
        let record = AuditRecord {
            sequence: self.next_sequence,
            previous: self.chain_head,
            digest: provider.digest(self.chain_head, self.next_sequence, event),
            event,
        };
        self.next_sequence = self.next_sequence.saturating_add(1);
        self.chain_head = record.digest;
        if self.records.len() == self.capacity {
            self.records.pop_front();
            self.dropped_records = self.dropped_records.saturating_add(1);
        }
        self.records.push_back(record);
        record
    }

    pub fn records(&self) -> Vec<AuditRecord> {
        self.records.iter().copied().collect()
    }

    pub fn chain_head(&self) -> ArtifactDigest {
        self.chain_head
    }

    pub fn dropped_records(&self) -> u64 {
        self.dropped_records
    }

    pub fn verify_records(
        provider: &impl AuditDigestProvider,
        records: &[AuditRecord],
        chain_head: ArtifactDigest,
    ) -> Result<(), AuditChainError> {
        let Some(first) = records.first().copied() else {
            return Ok(());
        };
        let mut expected_sequence = first.sequence;
        let mut previous = first.previous;
        for record in records {
            if record.sequence != expected_sequence {
                return Err(AuditChainError::SequenceGap);
            }
            if record.previous != previous {
                return Err(AuditChainError::PreviousDigestMismatch);
            }
            if provider.digest(record.previous, record.sequence, record.event) != record.digest {
                return Err(AuditChainError::DigestMismatch);
            }
            expected_sequence = expected_sequence.saturating_add(1);
            previous = record.digest;
        }
        if previous != chain_head {
            return Err(AuditChainError::DigestMismatch);
        }
        Ok(())
    }

    pub fn verify(&self, provider: &impl AuditDigestProvider) -> Result<(), AuditChainError> {
        let Some(first) = self.records.front().copied() else {
            return Ok(());
        };
        let mut expected_sequence = first.sequence;
        let mut previous = first.previous;
        for record in &self.records {
            if record.sequence != expected_sequence {
                return Err(AuditChainError::SequenceGap);
            }
            if record.previous != previous {
                return Err(AuditChainError::PreviousDigestMismatch);
            }
            if provider.digest(record.previous, record.sequence, record.event) != record.digest {
                return Err(AuditChainError::DigestMismatch);
            }
            expected_sequence = expected_sequence.saturating_add(1);
            previous = record.digest;
        }
        if previous != self.chain_head {
            return Err(AuditChainError::DigestMismatch);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn genesis() -> ArtifactDigest {
        ArtifactDigest([1; 32])
    }

    #[test]
    fn bounded_chain_verifies_after_old_records_are_dropped() {
        let provider = DeterministicAuditDigest;
        let mut ledger = AuditLedger::new(2, genesis());
        for operator_id in 1..=4 {
            ledger.append(
                &provider,
                AuditEvent::OperatorCommand {
                    operator_id,
                    proposal_id: operator_id,
                    command_code: 1,
                    accepted: true,
                },
            );
        }
        assert_eq!(ledger.records().len(), 2);
        assert_eq!(ledger.dropped_records(), 2);
        assert_eq!(ledger.verify(&provider), Ok(()));
    }

    #[test]
    fn modified_event_breaks_chain_verification() {
        let provider = DeterministicAuditDigest;
        let mut ledger = AuditLedger::new(4, genesis());
        ledger.append(
            &provider,
            AuditEvent::OperatorConstraint { constraint_code: 2 },
        );
        let mut records = ledger.records();
        records[0].event = AuditEvent::OperatorConstraint { constraint_code: 9 };
        let mut forged = AuditLedger::new(4, genesis());
        forged.records = records.into_iter().collect();
        forged.chain_head = ledger.chain_head();
        forged.next_sequence = 2;
        assert_eq!(
            forged.verify(&provider),
            Err(AuditChainError::DigestMismatch)
        );
    }

    #[test]
    fn actuator_service_events_are_distinct_in_canonical_chain() {
        let provider = DeterministicAuditDigest;
        let mut authorization = AuditLedger::new(4, genesis());
        authorization.append(
            &provider,
            AuditEvent::ActuatorServiceAuthorization {
                operator_id: 7,
                service_proposal_id: 9,
                actuator_code: 2,
                accepted: true,
            },
        );
        let mut transition = AuditLedger::new(4, genesis());
        transition.append(
            &provider,
            AuditEvent::ActuatorServiceTransition {
                service_proposal_id: 9,
                actuator_code: 2,
                state_code: 1,
            },
        );
        assert_ne!(authorization.chain_head(), transition.chain_head());
    }
}
