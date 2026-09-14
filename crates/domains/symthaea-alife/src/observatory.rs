// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic evolutionary observatory derived from the canonical Genesis event stream.
//!
//! The observatory is deliberately analysis-only. It does not alter selection, metabolism,
//! pairing, learning, reproduction, or death. Its job is to turn [`crate::GenesisEvent`] records
//! into deterministic, inspectable summaries while enforcing a small set of event-stream
//! invariants.
//!
//! ## Evidence boundary
//!
//! The current Genesis event stream is sufficient to establish observed trajectories,
//! interaction counts, transfer flow, lineage membership, and maximum observed generation. It is
//! **not** sufficient to establish exact birth/death times, immediate parentage, genome mutation
//! history, or causal necessity of any behavior. Those require additional first-class evidence
//! rather than inference from missing observations.

use std::collections::{BTreeMap, BTreeSet};

use crate::{AgentId, GenesisEvent};

/// One agent's deterministic summary across the supplied Genesis events.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentTrajectorySummary {
    pub agent_id: AgentId,
    pub lineage_id: AgentId,
    pub generation: u32,
    pub first_tick: u64,
    pub last_tick: u64,
    pub observations: u64,
    pub energy_start: f64,
    pub energy_end: f64,
    pub min_energy: f64,
    pub max_energy: f64,
    pub encounter_observations: u64,
    pub total_transfer_given: f64,
}

/// Directed transfer edge aggregated across the supplied event stream.
#[derive(Debug, Clone, PartialEq)]
pub struct TransferEdgeSummary {
    pub from: AgentId,
    pub to: AgentId,
    pub transfer_events: u64,
    pub total_amount: f64,
}

/// Aggregate facts for one observed lineage.
#[derive(Debug, Clone, PartialEq)]
pub struct LineageSummary {
    pub lineage_id: AgentId,
    pub members_observed: usize,
    pub max_generation: u32,
    pub first_tick: u64,
    pub last_tick: u64,
    pub total_transfer_given: f64,
}

/// Deterministic report reconstructed from a complete or partial Genesis event stream.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservatoryReport {
    pub event_count: usize,
    pub first_tick: Option<u64>,
    pub last_tick: Option<u64>,
    /// Number of distinct agent records observed on each tick. This is the population that
    /// entered that Genesis tick, not a claim about the post-birth/death population afterward.
    pub observed_agents_by_tick: BTreeMap<u64, usize>,
    pub agents: BTreeMap<AgentId, AgentTrajectorySummary>,
    pub lineages: BTreeMap<AgentId, LineageSummary>,
    pub transfers: BTreeMap<(AgentId, AgentId), TransferEdgeSummary>,
    pub max_generation: Option<u32>,
}

/// Structural problems that make a supplied event stream unsafe to summarize as one history.
#[derive(Debug, Clone, PartialEq)]
pub enum ObservatoryError {
    NonMonotonicTick {
        previous_tick: u64,
        next_tick: u64,
    },
    DuplicateAgentTick {
        tick: u64,
        agent_id: AgentId,
    },
    IdentityDrift {
        agent_id: AgentId,
        expected_lineage_id: AgentId,
        observed_lineage_id: AgentId,
        expected_generation: u32,
        observed_generation: u32,
    },
    SelfPartner {
        tick: u64,
        agent_id: AgentId,
    },
    TransferWithoutPartner {
        tick: u64,
        agent_id: AgentId,
        amount: f64,
    },
    InvalidNumericValue {
        tick: u64,
        agent_id: AgentId,
        field: &'static str,
        value: f64,
    },
}

/// Analyze Genesis events without changing or interpreting the underlying ALife dynamics.
///
/// Events must be non-decreasing by tick. Order *within* a tick is intentionally irrelevant:
/// all externally visible maps are ordered and the aggregation operators are commutative for the
/// quantities used here.
pub fn analyze_genesis_events(events: &[GenesisEvent]) -> Result<ObservatoryReport, ObservatoryError> {
    let mut agents = BTreeMap::<AgentId, AgentTrajectorySummary>::new();
    let mut transfers = BTreeMap::<(AgentId, AgentId), TransferEdgeSummary>::new();
    let mut observed_agents_by_tick = BTreeMap::<u64, usize>::new();
    let mut seen_agent_tick = BTreeSet::<(u64, AgentId)>::new();
    let mut previous_tick = None;
    let mut max_generation = None::<u32>;

    for event in events {
        if let Some(previous_tick) = previous_tick {
            if event.tick < previous_tick {
                return Err(ObservatoryError::NonMonotonicTick {
                    previous_tick,
                    next_tick: event.tick,
                });
            }
        }
        previous_tick = Some(event.tick);

        if !seen_agent_tick.insert((event.tick, event.agent_id)) {
            return Err(ObservatoryError::DuplicateAgentTick {
                tick: event.tick,
                agent_id: event.agent_id,
            });
        }

        if event.partner_id == Some(event.agent_id) {
            return Err(ObservatoryError::SelfPartner {
                tick: event.tick,
                agent_id: event.agent_id,
            });
        }

        validate_numeric(event, "resource_before", event.resource_before)?;
        validate_numeric(event, "resource_after", event.resource_after)?;
        validate_numeric(event, "transfer_amount", event.transfer_amount)?;
        if event.transfer_amount < 0.0 {
            return Err(ObservatoryError::InvalidNumericValue {
                tick: event.tick,
                agent_id: event.agent_id,
                field: "transfer_amount",
                value: event.transfer_amount,
            });
        }
        if event.transfer_amount > 0.0 && event.partner_id.is_none() {
            return Err(ObservatoryError::TransferWithoutPartner {
                tick: event.tick,
                agent_id: event.agent_id,
                amount: event.transfer_amount,
            });
        }

        *observed_agents_by_tick.entry(event.tick).or_default() += 1;
        max_generation = Some(max_generation.map_or(event.generation, |v| v.max(event.generation)));

        match agents.get_mut(&event.agent_id) {
            Some(summary) => {
                if summary.lineage_id != event.lineage_id || summary.generation != event.generation {
                    return Err(ObservatoryError::IdentityDrift {
                        agent_id: event.agent_id,
                        expected_lineage_id: summary.lineage_id,
                        observed_lineage_id: event.lineage_id,
                        expected_generation: summary.generation,
                        observed_generation: event.generation,
                    });
                }
                summary.last_tick = event.tick;
                summary.observations += 1;
                summary.energy_end = event.resource_after;
                summary.min_energy = summary
                    .min_energy
                    .min(event.resource_before)
                    .min(event.resource_after);
                summary.max_energy = summary
                    .max_energy
                    .max(event.resource_before)
                    .max(event.resource_after);
                summary.encounter_observations += u64::from(event.partner_id.is_some());
                summary.total_transfer_given += event.transfer_amount;
            }
            None => {
                agents.insert(
                    event.agent_id,
                    AgentTrajectorySummary {
                        agent_id: event.agent_id,
                        lineage_id: event.lineage_id,
                        generation: event.generation,
                        first_tick: event.tick,
                        last_tick: event.tick,
                        observations: 1,
                        energy_start: event.resource_before,
                        energy_end: event.resource_after,
                        min_energy: event.resource_before.min(event.resource_after),
                        max_energy: event.resource_before.max(event.resource_after),
                        encounter_observations: u64::from(event.partner_id.is_some()),
                        total_transfer_given: event.transfer_amount,
                    },
                );
            }
        }

        if event.transfer_amount > 0.0 {
            let partner = event.partner_id.expect("positive transfer validated to require partner");
            let edge = transfers
                .entry((event.agent_id, partner))
                .or_insert(TransferEdgeSummary {
                    from: event.agent_id,
                    to: partner,
                    transfer_events: 0,
                    total_amount: 0.0,
                });
            edge.transfer_events += 1;
            edge.total_amount += event.transfer_amount;
        }
    }

    let mut lineages = BTreeMap::<AgentId, LineageSummary>::new();
    for summary in agents.values() {
        let lineage = lineages.entry(summary.lineage_id).or_insert(LineageSummary {
            lineage_id: summary.lineage_id,
            members_observed: 0,
            max_generation: summary.generation,
            first_tick: summary.first_tick,
            last_tick: summary.last_tick,
            total_transfer_given: 0.0,
        });
        lineage.members_observed += 1;
        lineage.max_generation = lineage.max_generation.max(summary.generation);
        lineage.first_tick = lineage.first_tick.min(summary.first_tick);
        lineage.last_tick = lineage.last_tick.max(summary.last_tick);
        lineage.total_transfer_given += summary.total_transfer_given;
    }

    Ok(ObservatoryReport {
        event_count: events.len(),
        first_tick: events.first().map(|event| event.tick),
        last_tick: events.last().map(|event| event.tick),
        observed_agents_by_tick,
        agents,
        lineages,
        transfers,
        max_generation,
    })
}

fn validate_numeric(
    event: &GenesisEvent,
    field: &'static str,
    value: f64,
) -> Result<(), ObservatoryError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ObservatoryError::InvalidNumericValue {
            tick: event.tick,
            agent_id: event.agent_id,
            field,
            value,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AgentIdAllocator;

    fn event(
        tick: u64,
        agent_id: AgentId,
        partner_id: Option<AgentId>,
        transfer_amount: f64,
        generation: u32,
        lineage_id: AgentId,
    ) -> GenesisEvent {
        GenesisEvent {
            tick,
            agent_id,
            partner_id,
            action: 1,
            resource_before: 0.6,
            resource_after: 0.55,
            transfer_amount,
            generation,
            lineage_id,
        }
    }

    #[test]
    fn aggregates_agents_lineages_and_directed_transfers() {
        let mut ids = AgentIdAllocator::new();
        let founder = ids.allocate();
        let child = ids.allocate();
        let peer = ids.allocate();
        let events = vec![
            event(0, founder, Some(peer), 0.10, 0, founder),
            event(0, peer, Some(founder), 0.00, 0, peer),
            event(1, founder, Some(peer), 0.05, 0, founder),
            event(1, child, Some(peer), 0.00, 1, founder),
            event(1, peer, Some(founder), 0.00, 0, peer),
        ];

        let report = analyze_genesis_events(&events).expect("valid stream");
        assert_eq!(report.event_count, 5);
        assert_eq!(report.observed_agents_by_tick.get(&0), Some(&2));
        assert_eq!(report.observed_agents_by_tick.get(&1), Some(&3));
        assert_eq!(report.max_generation, Some(1));

        let founder_summary = report.agents.get(&founder).expect("founder summary");
        assert_eq!(founder_summary.observations, 2);
        assert!((founder_summary.total_transfer_given - 0.15).abs() < 1e-12);

        let lineage = report.lineages.get(&founder).expect("founder lineage");
        assert_eq!(lineage.members_observed, 2);
        assert_eq!(lineage.max_generation, 1);

        let edge = report
            .transfers
            .get(&(founder, peer))
            .expect("directed transfer edge");
        assert_eq!(edge.transfer_events, 2);
        assert!((edge.total_amount - 0.15).abs() < 1e-12);
    }

    #[test]
    fn order_within_a_tick_does_not_change_the_report() {
        let mut ids = AgentIdAllocator::new();
        let a = ids.allocate();
        let b = ids.allocate();
        let ab = vec![
            event(0, a, Some(b), 0.1, 0, a),
            event(0, b, Some(a), 0.0, 0, b),
        ];
        let ba = vec![ab[1], ab[0]];
        assert_eq!(
            analyze_genesis_events(&ab).expect("ab"),
            analyze_genesis_events(&ba).expect("ba")
        );
    }

    #[test]
    fn rejects_duplicate_agent_records_on_one_tick() {
        let mut ids = AgentIdAllocator::new();
        let a = ids.allocate();
        let events = vec![event(0, a, None, 0.0, 0, a), event(0, a, None, 0.0, 0, a)];
        assert!(matches!(
            analyze_genesis_events(&events),
            Err(ObservatoryError::DuplicateAgentTick { tick: 0, agent_id }) if agent_id == a
        ));
    }

    #[test]
    fn rejects_identity_drift_for_a_persistent_agent_id() {
        let mut ids = AgentIdAllocator::new();
        let a = ids.allocate();
        let other_lineage = ids.allocate();
        let events = vec![
            event(0, a, None, 0.0, 0, a),
            event(1, a, None, 0.0, 1, other_lineage),
        ];
        assert!(matches!(
            analyze_genesis_events(&events),
            Err(ObservatoryError::IdentityDrift { agent_id, .. }) if agent_id == a
        ));
    }

    #[test]
    fn rejects_positive_transfer_without_a_partner() {
        let mut ids = AgentIdAllocator::new();
        let a = ids.allocate();
        let events = vec![event(0, a, None, 0.1, 0, a)];
        assert!(matches!(
            analyze_genesis_events(&events),
            Err(ObservatoryError::TransferWithoutPartner { agent_id, .. }) if agent_id == a
        ));
    }
}
