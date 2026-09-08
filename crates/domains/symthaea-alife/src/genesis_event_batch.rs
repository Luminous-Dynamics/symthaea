// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Complete per-social-step evidence batches for the flat Genesis behavioral event stream.
//!
//! A raw `Vec<GenesisEvent>` cannot represent a social tick in which the population was empty,
//! and independent drained vectors do not prove that no tick interval was omitted between them.
//! This module adds a reference evidence shape that represents every `step_social` call explicitly,
//! including zero-event ticks, while preserving the existing `GenesisEvent` rows unchanged.
//!
//! The contract is analysis/evidence-only in this tranche. `Population` does not emit these batches
//! yet. Future production wiring should construct one batch atomically from the same pre-step
//! population and event rows already owned by `step_social`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{Action, AgentId, GenesisEvent, ObservatoryError, analyze_genesis_events};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenesisTickBatchV1 {
    tick: u64,
    population_before: u64,
    events: Vec<GenesisEvent>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedGenesisTickBatchV1 {
    batch: GenesisTickBatchV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenesisTickChunkSummaryV1 {
    pub start_tick: u64,
    pub next_tick: u64,
    pub batch_count: u64,
    pub event_count: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GenesisTickBatchErrorV1 {
    Observatory(ObservatoryError),
    EventTickMismatch {
        batch_tick: u64,
        event_tick: u64,
        agent_id: AgentId,
    },
    EventCountOverflow,
    PopulationCountMismatch {
        population_before: u64,
        observed_events: u64,
    },
    ReservedAgentIdentity,
    ReservedPartnerIdentity {
        agent_id: AgentId,
    },
    InvalidAction {
        agent_id: AgentId,
        action: usize,
    },
    PartnerMissing {
        agent_id: AgentId,
        partner_id: AgentId,
    },
    PartnerNotReciprocal {
        agent_id: AgentId,
        partner_id: AgentId,
        observed_partner: Option<AgentId>,
    },
    NonContiguousTick {
        expected: u64,
        observed: u64,
    },
    TickOverflow {
        tick: u64,
    },
    BatchCountOverflow,
    ChunkEventCountOverflow,
}

impl GenesisTickBatchV1 {
    pub fn new(tick: u64, population_before: u64, events: Vec<GenesisEvent>) -> Self {
        Self {
            tick,
            population_before,
            events,
        }
    }

    pub fn validate(self) -> Result<ValidatedGenesisTickBatchV1, GenesisTickBatchErrorV1> {
        validate_batch(&self)?;
        Ok(ValidatedGenesisTickBatchV1 { batch: self })
    }
}

impl ValidatedGenesisTickBatchV1 {
    pub fn tick(&self) -> u64 {
        self.batch.tick
    }

    pub fn population_before(&self) -> u64 {
        self.batch.population_before
    }

    pub fn events(&self) -> &[GenesisEvent] {
        &self.batch.events
    }

    pub fn as_batch(&self) -> &GenesisTickBatchV1 {
        &self.batch
    }

    pub fn into_batch(self) -> GenesisTickBatchV1 {
        self.batch
    }
}

/// Validate a drained sequence of complete social-step batches beginning at `expected_start_tick`.
///
/// Empty batches are first-class evidence, so an empty population can still advance the social
/// clock without creating an apparent gap. The returned `next_tick` is the exact semantic boundary
/// after the chunk. This function does not authenticate the supplied start boundary; future
/// execution capsules must bind it to validated runtime/evidence state.
pub fn validate_genesis_tick_chunk(
    expected_start_tick: u64,
    batches: &[GenesisTickBatchV1],
) -> Result<GenesisTickChunkSummaryV1, GenesisTickBatchErrorV1> {
    let mut expected = expected_start_tick;
    let mut batch_count = 0u64;
    let mut event_count = 0u64;

    for batch in batches {
        if batch.tick != expected {
            return Err(GenesisTickBatchErrorV1::NonContiguousTick {
                expected,
                observed: batch.tick,
            });
        }
        validate_batch(batch)?;
        expected = expected
            .checked_add(1)
            .ok_or(GenesisTickBatchErrorV1::TickOverflow { tick: batch.tick })?;
        batch_count = batch_count
            .checked_add(1)
            .ok_or(GenesisTickBatchErrorV1::BatchCountOverflow)?;
        let batch_events = u64::try_from(batch.events.len())
            .map_err(|_| GenesisTickBatchErrorV1::EventCountOverflow)?;
        event_count = event_count
            .checked_add(batch_events)
            .ok_or(GenesisTickBatchErrorV1::ChunkEventCountOverflow)?;
    }

    Ok(GenesisTickChunkSummaryV1 {
        start_tick: expected_start_tick,
        next_tick: expected,
        batch_count,
        event_count,
    })
}

fn validate_batch(batch: &GenesisTickBatchV1) -> Result<(), GenesisTickBatchErrorV1> {
    let observed_events = u64::try_from(batch.events.len())
        .map_err(|_| GenesisTickBatchErrorV1::EventCountOverflow)?;
    if observed_events != batch.population_before {
        return Err(GenesisTickBatchErrorV1::PopulationCountMismatch {
            population_before: batch.population_before,
            observed_events,
        });
    }

    for event in &batch.events {
        if event.tick != batch.tick {
            return Err(GenesisTickBatchErrorV1::EventTickMismatch {
                batch_tick: batch.tick,
                event_tick: event.tick,
                agent_id: event.agent_id,
            });
        }
        if event.agent_id == AgentId::UNALLOCATED {
            return Err(GenesisTickBatchErrorV1::ReservedAgentIdentity);
        }
        if event.partner_id == Some(AgentId::UNALLOCATED) {
            return Err(GenesisTickBatchErrorV1::ReservedPartnerIdentity {
                agent_id: event.agent_id,
            });
        }
        if event.action >= Action::SOCIAL_COUNT {
            return Err(GenesisTickBatchErrorV1::InvalidAction {
                agent_id: event.agent_id,
                action: event.action,
            });
        }
    }

    analyze_genesis_events(&batch.events).map_err(GenesisTickBatchErrorV1::Observatory)?;

    let partners = batch
        .events
        .iter()
        .map(|event| (event.agent_id, event.partner_id))
        .collect::<BTreeMap<_, _>>();
    for event in &batch.events {
        let Some(partner_id) = event.partner_id else {
            continue;
        };
        let Some(observed_partner) = partners.get(&partner_id).copied() else {
            return Err(GenesisTickBatchErrorV1::PartnerMissing {
                agent_id: event.agent_id,
                partner_id,
            });
        };
        if observed_partner != Some(event.agent_id) {
            return Err(GenesisTickBatchErrorV1::PartnerNotReciprocal {
                agent_id: event.agent_id,
                partner_id,
                observed_partner,
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AgentIdAllocator;

    fn event(
        tick: u64,
        agent_id: AgentId,
        partner_id: Option<AgentId>,
        action: usize,
    ) -> GenesisEvent {
        GenesisEvent {
            tick,
            agent_id,
            partner_id,
            action,
            resource_before: 0.5,
            resource_after: 0.5,
            transfer_amount: 0.0,
            generation: 0,
            lineage_id: agent_id,
        }
    }

    #[test]
    fn empty_population_tick_is_first_class_and_advances_chunk_boundary() {
        let batches = vec![GenesisTickBatchV1::new(7, 0, Vec::new())];
        let summary = validate_genesis_tick_chunk(7, &batches).expect("valid empty tick");
        assert_eq!(summary.start_tick, 7);
        assert_eq!(summary.next_tick, 8);
        assert_eq!(summary.batch_count, 1);
        assert_eq!(summary.event_count, 0);
    }

    #[test]
    fn reciprocal_pair_is_valid_complete_tick() {
        let mut ids = AgentIdAllocator::new();
        let a = ids.allocate();
        let b = ids.allocate();
        let batch = GenesisTickBatchV1::new(
            3,
            2,
            vec![event(3, a, Some(b), 0), event(3, b, Some(a), 1)],
        );
        let validated = batch.validate().expect("reciprocal pair");
        assert_eq!(validated.tick(), 3);
        assert_eq!(validated.population_before(), 2);
    }

    #[test]
    fn population_count_must_equal_complete_tick_event_count() {
        let mut ids = AgentIdAllocator::new();
        let a = ids.allocate();
        let err = GenesisTickBatchV1::new(0, 2, vec![event(0, a, None, 0)])
            .validate()
            .expect_err("incomplete tick must fail");
        assert_eq!(
            err,
            GenesisTickBatchErrorV1::PopulationCountMismatch {
                population_before: 2,
                observed_events: 1,
            }
        );
    }

    #[test]
    fn partner_relation_must_be_reciprocal_within_the_same_tick() {
        let mut ids = AgentIdAllocator::new();
        let a = ids.allocate();
        let b = ids.allocate();
        let err = GenesisTickBatchV1::new(
            0,
            2,
            vec![event(0, a, Some(b), 0), event(0, b, None, 1)],
        )
        .validate()
        .expect_err("one-way partner relation must fail");
        assert_eq!(
            err,
            GenesisTickBatchErrorV1::PartnerNotReciprocal {
                agent_id: a,
                partner_id: b,
                observed_partner: None,
            }
        );
    }

    #[test]
    fn invalid_action_is_rejected_before_batch_can_be_authoritative() {
        let mut ids = AgentIdAllocator::new();
        let a = ids.allocate();
        let err = GenesisTickBatchV1::new(0, 1, vec![event(0, a, None, 99)])
            .validate()
            .expect_err("invalid action");
        assert_eq!(
            err,
            GenesisTickBatchErrorV1::InvalidAction {
                agent_id: a,
                action: 99,
            }
        );
    }

    #[test]
    fn drained_chunk_ticks_must_be_exactly_contiguous() {
        let batches = vec![
            GenesisTickBatchV1::new(4, 0, Vec::new()),
            GenesisTickBatchV1::new(6, 0, Vec::new()),
        ];
        assert_eq!(
            validate_genesis_tick_chunk(4, &batches),
            Err(GenesisTickBatchErrorV1::NonContiguousTick {
                expected: 5,
                observed: 6,
            })
        );
    }

    #[test]
    fn terminal_tick_cannot_create_an_unrepresentable_continuation_boundary() {
        let batches = vec![GenesisTickBatchV1::new(u64::MAX, 0, Vec::new())];
        assert_eq!(
            validate_genesis_tick_chunk(u64::MAX, &batches),
            Err(GenesisTickBatchErrorV1::TickOverflow { tick: u64::MAX })
        );
    }
}
