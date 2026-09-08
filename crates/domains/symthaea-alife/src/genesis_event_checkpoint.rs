// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compact continuation evidence for the canonical Genesis behavioral event stream.
//!
//! [`crate::GenesisEvent`] is intentionally drainable because one row is emitted for every
//! organism entering every [`crate::Population::step_social`] tick. Exact execution resume must
//! therefore preserve not only future population state but also where the already-persisted
//! behavioral stream ended.
//!
//! This v1 contract is scoped to the canonical population transition profile at **stable social
//! step boundaries**: no caller directly mutates `Population::organisms`, and chunks are drained
//! only between complete `step_social` calls. Under that profile each non-empty social tick emits
//! the complete pre-transition living set, so observed ticks begin at zero and remain contiguous
//! until the population first enters terminal silence.
//!
//! The checkpoint is compact rather than lossless: archived [`crate::GenesisEvent`] chunks remain
//! the scientific data. A cumulative SHA-256 chain binds the exact ordered event rows that have
//! been accepted so far. The chain is independent of *chunk* boundaries because it advances once
//! per event, but it intentionally preserves event order within a tick even though the current
//! observatory aggregation is order-insensitive there.
//!
//! Persisted checkpoint bytes are not self-authenticating. After deserialization, scientific
//! authority is recreated only by revalidating against the archived complete prefix. A future
//! signed/runtime-issued boundary receipt can authenticate the compact checkpoint without replaying
//! the archive. `next_social_tick` has the same narrow limitation as lifecycle continuation epoch:
//! event rows can prove it is not behind the last observation, but cannot independently prove how
//! many transition-free silent ticks occurred afterward.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Action, AgentId, GenesisEvent, ObservatoryError, analyze_genesis_events};

const CHAIN_ROOT_DOMAIN_V1: &[u8] = b"symthaea.alife.genesis-event-chain.root.v1\0";
const CHAIN_EVENT_DOMAIN_V1: &[u8] = b"symthaea.alife.genesis-event-chain.event.v1\0";

/// Compact persisted continuation boundary for canonical Genesis behavioral evidence.
///
/// Raw fields stay private. Deserialization alone does not authorize stream continuation; callers
/// must recreate a [`ValidatedGenesisEventCheckpointV1`] from the archived prefix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenesisEventCheckpointV1 {
    next_social_tick: u64,
    total_events: u64,
    last_observed_tick: Option<u64>,
    terminal_silence: bool,
    chain_sha256: [u8; 32],
}

/// Non-serializable capability proving one behavioral-event continuation boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedGenesisEventCheckpointV1 {
    checkpoint: GenesisEventCheckpointV1,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GenesisEventCheckpointErrorV1 {
    Observatory(ObservatoryError),
    ReservedIdentity {
        tick: u64,
        field: &'static str,
    },
    InvalidActionIndex {
        tick: u64,
        agent_id: AgentId,
        action: usize,
    },
    ActionIndexTooWide {
        tick: u64,
        agent_id: AgentId,
        action: usize,
    },
    TransferWithoutTransferAction {
        tick: u64,
        agent_id: AgentId,
        action: usize,
        amount: f64,
    },
    MissingReciprocalPartner {
        tick: u64,
        agent_id: AgentId,
        partner_id: AgentId,
    },
    PartnerMismatch {
        tick: u64,
        agent_id: AgentId,
        partner_id: AgentId,
        observed_partner: Option<AgentId>,
    },
    BoundaryMovedBackward {
        previous_next_social_tick: u64,
        resulting_next_social_tick: u64,
    },
    ChunkDoesNotStartAtBoundary {
        expected_tick: u64,
        observed_tick: u64,
    },
    InternalTickGap {
        expected_tick: u64,
        observed_tick: u64,
    },
    EventAtOrAfterResultingBoundary {
        event_tick: u64,
        resulting_next_social_tick: u64,
    },
    TickOverflow {
        tick: u64,
    },
    EventAfterTerminalSilence {
        boundary_next_social_tick: u64,
    },
    EventCountOverflow,
    CheckpointFieldMismatch {
        field: &'static str,
    },
}

impl From<ObservatoryError> for GenesisEventCheckpointErrorV1 {
    fn from(value: ObservatoryError) -> Self {
        Self::Observatory(value)
    }
}

impl GenesisEventCheckpointV1 {
    /// Validate a complete canonical behavioral prefix beginning at social tick zero.
    pub fn from_complete_prefix(
        events: &[GenesisEvent],
        next_social_tick: u64,
    ) -> Result<ValidatedGenesisEventCheckpointV1, GenesisEventCheckpointErrorV1> {
        ValidatedGenesisEventCheckpointV1::genesis()
            .validate_chunk(events, next_social_tick)
    }

    /// Recreate authority for a deserialized compact checkpoint by replaying the archived complete
    /// prefix and requiring every derived compact field to match.
    ///
    /// `next_social_tick` remains runtime-boundary metadata rather than an event-derived fact. This
    /// method proves that the declared boundary is *consistent* with the supplied prefix and that
    /// all other compact fields are exactly implied by it; it does not authenticate the declared
    /// number of trailing silent ticks against malicious replacement.
    pub fn validate_against_complete_prefix(
        self,
        events: &[GenesisEvent],
    ) -> Result<ValidatedGenesisEventCheckpointV1, GenesisEventCheckpointErrorV1> {
        let expected = Self::from_complete_prefix(events, self.next_social_tick)?;
        let expected_raw = expected.as_checkpoint();

        if self.total_events != expected_raw.total_events {
            return Err(GenesisEventCheckpointErrorV1::CheckpointFieldMismatch {
                field: "total_events",
            });
        }
        if self.last_observed_tick != expected_raw.last_observed_tick {
            return Err(GenesisEventCheckpointErrorV1::CheckpointFieldMismatch {
                field: "last_observed_tick",
            });
        }
        if self.terminal_silence != expected_raw.terminal_silence {
            return Err(GenesisEventCheckpointErrorV1::CheckpointFieldMismatch {
                field: "terminal_silence",
            });
        }
        if self.chain_sha256 != expected_raw.chain_sha256 {
            return Err(GenesisEventCheckpointErrorV1::CheckpointFieldMismatch {
                field: "chain_sha256",
            });
        }

        Ok(ValidatedGenesisEventCheckpointV1 { checkpoint: self })
    }
}

impl ValidatedGenesisEventCheckpointV1 {
    fn genesis() -> Self {
        Self {
            checkpoint: GenesisEventCheckpointV1 {
                next_social_tick: 0,
                total_events: 0,
                last_observed_tick: None,
                terminal_silence: false,
                chain_sha256: chain_root_v1(),
            },
        }
    }

    /// Opaque compact checkpoint to persist. Revalidate it against the archived prefix after load.
    pub fn as_checkpoint(&self) -> &GenesisEventCheckpointV1 {
        &self.checkpoint
    }

    pub fn into_checkpoint(self) -> GenesisEventCheckpointV1 {
        self.checkpoint
    }

    pub fn next_social_tick(&self) -> u64 {
        self.checkpoint.next_social_tick
    }

    pub fn total_events(&self) -> u64 {
        self.checkpoint.total_events
    }

    pub fn last_observed_tick(&self) -> Option<u64> {
        self.checkpoint.last_observed_tick
    }

    /// True once at least one complete social tick emitted zero events. Under the canonical
    /// population profile that means the population was empty entering that tick, and without
    /// external raw membership mutation later behavioral events are impossible.
    pub fn terminal_silence(&self) -> bool {
        self.checkpoint.terminal_silence
    }

    pub fn chain_sha256(&self) -> [u8; 32] {
        self.checkpoint.chain_sha256
    }

    /// Validate one stable-boundary drained chunk and return the resulting continuation authority.
    ///
    /// Chunks must begin exactly at this checkpoint's `next_social_tick` if they contain events.
    /// Observed ticks inside the chunk must then be contiguous. `resulting_next_social_tick` may
    /// extend beyond the final observed tick only to represent a trailing run of silent complete
    /// ticks; once that happens, canonical population execution can never emit another event.
    pub fn validate_chunk(
        &self,
        events: &[GenesisEvent],
        resulting_next_social_tick: u64,
    ) -> Result<Self, GenesisEventCheckpointErrorV1> {
        if resulting_next_social_tick < self.checkpoint.next_social_tick {
            return Err(GenesisEventCheckpointErrorV1::BoundaryMovedBackward {
                previous_next_social_tick: self.checkpoint.next_social_tick,
                resulting_next_social_tick,
            });
        }
        if self.checkpoint.terminal_silence && !events.is_empty() {
            return Err(GenesisEventCheckpointErrorV1::EventAfterTerminalSilence {
                boundary_next_social_tick: self.checkpoint.next_social_tick,
            });
        }

        validate_population_event_rows(events)?;
        let (chunk_last_tick, chunk_terminal_silence) = validate_tick_coverage(
            events,
            self.checkpoint.next_social_tick,
            resulting_next_social_tick,
        )?;

        let event_count = u64::try_from(events.len())
            .map_err(|_| GenesisEventCheckpointErrorV1::EventCountOverflow)?;
        let total_events = self
            .checkpoint
            .total_events
            .checked_add(event_count)
            .ok_or(GenesisEventCheckpointErrorV1::EventCountOverflow)?;

        let mut chain_sha256 = self.checkpoint.chain_sha256;
        for event in events {
            chain_sha256 = hash_event_v1(chain_sha256, event)?;
        }

        Ok(Self {
            checkpoint: GenesisEventCheckpointV1 {
                next_social_tick: resulting_next_social_tick,
                total_events,
                last_observed_tick: chunk_last_tick.or(self.checkpoint.last_observed_tick),
                terminal_silence: self.checkpoint.terminal_silence || chunk_terminal_silence,
                chain_sha256,
            },
        })
    }
}

fn validate_population_event_rows(
    events: &[GenesisEvent],
) -> Result<(), GenesisEventCheckpointErrorV1> {
    analyze_genesis_events(events)?;

    let mut partner_by_agent_tick = BTreeMap::<(u64, AgentId), Option<AgentId>>::new();

    for event in events {
        if event.agent_id == AgentId::UNALLOCATED {
            return Err(GenesisEventCheckpointErrorV1::ReservedIdentity {
                tick: event.tick,
                field: "agent_id",
            });
        }
        if event.lineage_id == AgentId::UNALLOCATED {
            return Err(GenesisEventCheckpointErrorV1::ReservedIdentity {
                tick: event.tick,
                field: "lineage_id",
            });
        }
        if event.partner_id == Some(AgentId::UNALLOCATED) {
            return Err(GenesisEventCheckpointErrorV1::ReservedIdentity {
                tick: event.tick,
                field: "partner_id",
            });
        }
        if event.action >= Action::SOCIAL_COUNT {
            return Err(GenesisEventCheckpointErrorV1::InvalidActionIndex {
                tick: event.tick,
                agent_id: event.agent_id,
                action: event.action,
            });
        }
        if event.transfer_amount > 0.0 && event.action != Action::Transfer.index() {
            return Err(
                GenesisEventCheckpointErrorV1::TransferWithoutTransferAction {
                    tick: event.tick,
                    agent_id: event.agent_id,
                    action: event.action,
                    amount: event.transfer_amount,
                },
            );
        }

        partner_by_agent_tick.insert((event.tick, event.agent_id), event.partner_id);
    }

    for event in events {
        let Some(partner_id) = event.partner_id else {
            continue;
        };
        let observed = partner_by_agent_tick
            .get(&(event.tick, partner_id))
            .copied()
            .ok_or(
                GenesisEventCheckpointErrorV1::MissingReciprocalPartner {
                    tick: event.tick,
                    agent_id: event.agent_id,
                    partner_id,
                },
            )?;
        if observed != Some(event.agent_id) {
            return Err(GenesisEventCheckpointErrorV1::PartnerMismatch {
                tick: event.tick,
                agent_id: event.agent_id,
                partner_id,
                observed_partner: observed,
            });
        }
    }

    Ok(())
}

fn validate_tick_coverage(
    events: &[GenesisEvent],
    expected_start_tick: u64,
    resulting_next_social_tick: u64,
) -> Result<(Option<u64>, bool), GenesisEventCheckpointErrorV1> {
    if resulting_next_social_tick < expected_start_tick {
        return Err(GenesisEventCheckpointErrorV1::BoundaryMovedBackward {
            previous_next_social_tick: expected_start_tick,
            resulting_next_social_tick,
        });
    }

    let Some(first) = events.first() else {
        return Ok((None, resulting_next_social_tick > expected_start_tick));
    };

    if first.tick != expected_start_tick {
        return Err(
            GenesisEventCheckpointErrorV1::ChunkDoesNotStartAtBoundary {
                expected_tick: expected_start_tick,
                observed_tick: first.tick,
            },
        );
    }

    let mut last_tick = first.tick;
    for event in events.iter().skip(1) {
        if event.tick == last_tick {
            continue;
        }
        let expected = last_tick
            .checked_add(1)
            .ok_or(GenesisEventCheckpointErrorV1::TickOverflow { tick: last_tick })?;
        if event.tick != expected {
            return Err(GenesisEventCheckpointErrorV1::InternalTickGap {
                expected_tick: expected,
                observed_tick: event.tick,
            });
        }
        last_tick = event.tick;
    }

    let minimum_resulting_boundary = last_tick
        .checked_add(1)
        .ok_or(GenesisEventCheckpointErrorV1::TickOverflow { tick: last_tick })?;
    if resulting_next_social_tick < minimum_resulting_boundary {
        return Err(
            GenesisEventCheckpointErrorV1::EventAtOrAfterResultingBoundary {
                event_tick: last_tick,
                resulting_next_social_tick,
            },
        );
    }

    Ok((
        Some(last_tick),
        resulting_next_social_tick > minimum_resulting_boundary,
    ))
}

fn chain_root_v1() -> [u8; 32] {
    Sha256::digest(CHAIN_ROOT_DOMAIN_V1).into()
}

fn hash_event_v1(
    previous: [u8; 32],
    event: &GenesisEvent,
) -> Result<[u8; 32], GenesisEventCheckpointErrorV1> {
    let action = u64::try_from(event.action).map_err(|_| {
        GenesisEventCheckpointErrorV1::ActionIndexTooWide {
            tick: event.tick,
            agent_id: event.agent_id,
            action: event.action,
        }
    })?;

    let mut hasher = Sha256::new();
    hasher.update(CHAIN_EVENT_DOMAIN_V1);
    hasher.update(previous);
    hasher.update(event.tick.to_le_bytes());
    hasher.update(event.agent_id.raw().to_le_bytes());
    match event.partner_id {
        Some(partner_id) => {
            hasher.update([1]);
            hasher.update(partner_id.raw().to_le_bytes());
        }
        None => {
            hasher.update([0]);
            hasher.update(0u64.to_le_bytes());
        }
    }
    hasher.update(action.to_le_bytes());
    hasher.update(event.resource_before.to_bits().to_le_bytes());
    hasher.update(event.resource_after.to_bits().to_le_bytes());
    hasher.update(event.transfer_amount.to_bits().to_le_bytes());
    hasher.update(event.generation.to_le_bytes());
    hasher.update(event.lineage_id.raw().to_le_bytes());
    Ok(hasher.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AgentIdAllocator;

    fn ids() -> (AgentId, AgentId) {
        let mut allocator = AgentIdAllocator::new();
        (allocator.allocate(), allocator.allocate())
    }

    fn paired_tick(tick: u64, a: AgentId, b: AgentId) -> Vec<GenesisEvent> {
        vec![
            GenesisEvent {
                tick,
                agent_id: a,
                partner_id: Some(b),
                action: Action::Rest.index(),
                resource_before: 0.6,
                resource_after: 0.59,
                transfer_amount: 0.0,
                generation: 0,
                lineage_id: a,
            },
            GenesisEvent {
                tick,
                agent_id: b,
                partner_id: Some(a),
                action: Action::Rest.index(),
                resource_before: 0.7,
                resource_after: 0.69,
                transfer_amount: 0.0,
                generation: 0,
                lineage_id: b,
            },
        ]
    }

    #[test]
    fn complete_prefix_and_tick_aligned_chunking_produce_identical_checkpoint() {
        let (a, b) = ids();
        let tick0 = paired_tick(0, a, b);
        let tick1 = paired_tick(1, a, b);
        let complete: Vec<_> = tick0.iter().chain(&tick1).copied().collect();

        let one_shot = GenesisEventCheckpointV1::from_complete_prefix(&complete, 2)
            .expect("valid complete behavioral prefix");
        let first = GenesisEventCheckpointV1::from_complete_prefix(&tick0, 1)
            .expect("valid first chunk");
        let chunked = first
            .validate_chunk(&tick1, 2)
            .expect("valid second chunk");

        assert_eq!(chunked.as_checkpoint(), one_shot.as_checkpoint());
        assert_eq!(chunked.total_events(), 4);
        assert_eq!(chunked.last_observed_tick(), Some(1));
        assert!(!chunked.terminal_silence());
    }

    #[test]
    fn serialized_checkpoint_requires_archived_prefix_revalidation() {
        let (a, b) = ids();
        let events = paired_tick(0, a, b);
        let validated = GenesisEventCheckpointV1::from_complete_prefix(&events, 1)
            .expect("valid prefix");
        let json = serde_json::to_string(validated.as_checkpoint()).expect("serialize checkpoint");
        let decoded: GenesisEventCheckpointV1 =
            serde_json::from_str(&json).expect("deserialize checkpoint");
        let revalidated = decoded
            .validate_against_complete_prefix(&events)
            .expect("archive-backed revalidation");
        assert_eq!(revalidated.as_checkpoint(), validated.as_checkpoint());
    }

    #[test]
    fn tampered_chain_cannot_revalidate_against_the_archive() {
        let (a, b) = ids();
        let events = paired_tick(0, a, b);
        let validated = GenesisEventCheckpointV1::from_complete_prefix(&events, 1)
            .expect("valid prefix");
        let mut raw = validated.into_checkpoint();
        raw.chain_sha256[0] ^= 0xff;
        assert_eq!(
            raw.validate_against_complete_prefix(&events),
            Err(GenesisEventCheckpointErrorV1::CheckpointFieldMismatch {
                field: "chain_sha256",
            })
        );
    }

    #[test]
    fn missing_reciprocal_partner_fails_closed() {
        let (a, b) = ids();
        let events = vec![GenesisEvent {
            tick: 0,
            agent_id: a,
            partner_id: Some(b),
            action: Action::Rest.index(),
            resource_before: 0.5,
            resource_after: 0.49,
            transfer_amount: 0.0,
            generation: 0,
            lineage_id: a,
        }];
        assert_eq!(
            GenesisEventCheckpointV1::from_complete_prefix(&events, 1),
            Err(GenesisEventCheckpointErrorV1::MissingReciprocalPartner {
                tick: 0,
                agent_id: a,
                partner_id: b,
            })
        );
    }

    #[test]
    fn positive_transfer_requires_the_transfer_action() {
        let (a, b) = ids();
        let mut events = paired_tick(0, a, b);
        events[0].transfer_amount = 0.1;
        assert_eq!(
            GenesisEventCheckpointV1::from_complete_prefix(&events, 1),
            Err(
                GenesisEventCheckpointErrorV1::TransferWithoutTransferAction {
                    tick: 0,
                    agent_id: a,
                    action: Action::Rest.index(),
                    amount: 0.1,
                }
            )
        );
    }

    #[test]
    fn action_index_outside_live_social_space_fails_closed() {
        let (a, b) = ids();
        let mut events = paired_tick(0, a, b);
        events[0].action = Action::SOCIAL_COUNT;
        assert_eq!(
            GenesisEventCheckpointV1::from_complete_prefix(&events, 1),
            Err(GenesisEventCheckpointErrorV1::InvalidActionIndex {
                tick: 0,
                agent_id: a,
                action: Action::SOCIAL_COUNT,
            })
        );
    }

    #[test]
    fn complete_prefix_rejects_missing_internal_social_tick() {
        let (a, b) = ids();
        let tick0 = paired_tick(0, a, b);
        let tick2 = paired_tick(2, a, b);
        let events: Vec<_> = tick0.iter().chain(&tick2).copied().collect();
        assert_eq!(
            GenesisEventCheckpointV1::from_complete_prefix(&events, 3),
            Err(GenesisEventCheckpointErrorV1::InternalTickGap {
                expected_tick: 1,
                observed_tick: 2,
            })
        );
    }

    #[test]
    fn trailing_silent_tick_is_terminal_for_canonical_population_execution() {
        let (a, b) = ids();
        let events = paired_tick(0, a, b);
        let silent = GenesisEventCheckpointV1::from_complete_prefix(&events, 2)
            .expect("one observed tick followed by one complete silent tick");
        assert!(silent.terminal_silence());
        assert_eq!(silent.next_social_tick(), 2);

        let later = paired_tick(2, a, b);
        assert_eq!(
            silent.validate_chunk(&later, 3),
            Err(GenesisEventCheckpointErrorV1::EventAfterTerminalSilence {
                boundary_next_social_tick: 2,
            })
        );
    }

    #[test]
    fn silent_tick_count_is_consistency_metadata_not_event_derived_fact() {
        let (a, b) = ids();
        let events = paired_tick(0, a, b);
        let validated = GenesisEventCheckpointV1::from_complete_prefix(&events, 1)
            .expect("valid event prefix");
        let mut declared = validated.into_checkpoint();

        // The rows establish tick 0 but cannot prove whether the runtime stopped at boundary 1 or
        // executed four additional empty social ticks. Marking the resulting trailing-silence state
        // consistently is therefore semantically valid but not origin-authenticated.
        declared.next_social_tick = 5;
        declared.terminal_silence = true;
        let revalidated = declared
            .validate_against_complete_prefix(&events)
            .expect("declared silent-tail boundary is event-consistent");
        assert_eq!(revalidated.next_social_tick(), 5);
        assert!(revalidated.terminal_silence());
    }

    #[test]
    fn reserved_population_identity_is_rejected() {
        let (a, _) = ids();
        let events = vec![GenesisEvent {
            tick: 0,
            agent_id: a,
            partner_id: None,
            action: Action::Rest.index(),
            resource_before: 0.5,
            resource_after: 0.49,
            transfer_amount: 0.0,
            generation: 0,
            lineage_id: AgentId::UNALLOCATED,
        }];
        assert_eq!(
            GenesisEventCheckpointV1::from_complete_prefix(&events, 1),
            Err(GenesisEventCheckpointErrorV1::ReservedIdentity {
                tick: 0,
                field: "lineage_id",
            })
        );
    }
}
