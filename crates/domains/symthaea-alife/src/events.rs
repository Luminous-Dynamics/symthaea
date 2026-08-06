// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Event-stream logging for Genesis v0, per `ALIFE_MULTIAGENT_GENESIS_PLAN_2026-07-25.md`'s
//! "Logging / instrumentation" section.
//!
//! The canonical artifact for a Genesis run is this event stream, not any single derived view --
//! every downstream analysis (encounter graph, transfer graph, energy histories, lineage trees,
//! relationship trajectories) should be reconstructable from it after the fact. One [`GenesisEvent`]
//! is emitted per organism per tick from [`crate::Population::step_social`].

use serde::{Deserialize, Serialize};

use crate::agent_id::AgentId;

/// One organism's outcome on one tick of a Genesis-enabled `step_social` call.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GenesisEvent {
    pub tick: u64,
    pub agent_id: AgentId,
    /// `None` when this organism went unpaired this tick (e.g. an odd-sized population, or a
    /// `PairingMode::FixedPartners` partner that has since died).
    pub partner_id: Option<AgentId>,
    /// `Action::index()` of the action actually executed this tick.
    pub action: usize,
    /// This organism's own energy immediately before this tick's action/consequence.
    pub resource_before: f64,
    /// This organism's own energy immediately after this tick's action/consequence (including
    /// any `Transfer` credit received this tick, applied in the same tick's Phase 3 -- see
    /// `Population::step_social`).
    pub resource_after: f64,
    /// The amount *this organism itself* transferred out this tick (`0.0` unless `action` was
    /// `Transfer` and a partner was present) -- what was credited to `partner_id`, not what this
    /// organism may have received (that's visible as `resource_after - resource_before` net of
    /// its own metabolic costs, or simply by reading the partner's own event this same tick).
    pub transfer_amount: f64,
    pub generation: u32,
    pub lineage_id: AgentId,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_id::AgentIdAllocator;

    #[test]
    fn genesis_event_is_plain_data_and_round_trips_through_json() {
        let mut alloc = AgentIdAllocator::new();
        let event = GenesisEvent {
            tick: 42,
            agent_id: alloc.allocate(),
            partner_id: Some(alloc.allocate()),
            action: 2,
            resource_before: 0.5,
            resource_after: 0.45,
            transfer_amount: 0.05,
            generation: 3,
            lineage_id: AgentId::UNALLOCATED,
        };
        let json = serde_json::to_string(&event).expect("serialize");
        let round_tripped: GenesisEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(event, round_tripped);
    }
}
