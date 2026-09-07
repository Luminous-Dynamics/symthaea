// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Materialize admitted temporal allocations into committed flow snapshots.
//!
//! This crate closes the semantic loop between the new temporal planning stack and
//! `symthaea-resource-model::ResourceGraph` from #689.
//!
//! At one exact instant it selects the half-open set of active admitted
//! allocations, reconstructs only the endpoint ports required by those transfers,
//! and inserts their exact sent quantities/losses into a fresh `ResourceGraph`.
//!
//! `ResourceGraph::connect` then independently re-enforces static endpoint and
//! aggregate shared-port capacity. Materialization is therefore not a blind cast
//! from one positive planning type to another.
//!
//! A snapshot is descriptive planning state only. It is not a Mycelix lease,
//! physical observation, accounting receipt, operating authority, or HAL command.

#![deny(unsafe_code)]

use chrono::{DateTime, Utc};
use std::collections::BTreeMap;
use symthaea_resource_allocation::{AdmittedAllocation, AllocationBook};
use symthaea_resource_model::{ResourceEdge, ResourceGraph, ResourcePort};
use symthaea_resource_topology::ResourceTopology;
use thiserror::Error;

/// One committed-flow view materialized at an exact instant.
#[derive(Debug, Clone, PartialEq)]
pub struct CommittedFlowSnapshot {
    observed_at: DateTime<Utc>,
    graph: ResourceGraph,
    active_allocation_ids: Vec<String>,
}

impl CommittedFlowSnapshot {
    pub fn observed_at(&self) -> DateTime<Utc> {
        self.observed_at
    }

    pub fn graph(&self) -> &ResourceGraph {
        &self.graph
    }

    pub fn active_allocation_ids(&self) -> &[String] {
        &self.active_allocation_ids
    }
}

/// Materialize the exact half-open active allocation set at `observed_at`.
pub fn materialize_flow_snapshot(
    topology: &ResourceTopology,
    allocations: &AllocationBook,
    observed_at: DateTime<Utc>,
) -> Result<CommittedFlowSnapshot, FlowSnapshotError> {
    let active: Vec<&AdmittedAllocation> = allocations
        .allocations()
        .filter(|admitted| {
            let allocation = admitted.allocation();
            allocation.valid_from <= observed_at && observed_at < allocation.valid_until
        })
        .collect();

    let mut node_ports: BTreeMap<String, BTreeMap<String, ResourcePort>> = BTreeMap::new();
    for admitted in &active {
        let allocation = admitted.allocation();
        let link = topology
            .link(&allocation.link_id)
            .ok_or_else(|| FlowSnapshotError::UnknownLink(allocation.link_id.clone()))?;
        retain_port(
            topology,
            &mut node_ports,
            &link.from_node,
            &link.from_port,
        )?;
        retain_port(
            topology,
            &mut node_ports,
            &link.to_node,
            &link.to_port,
        )?;
    }

    let mut graph = ResourceGraph::default();
    for (node_id, ports) in node_ports {
        graph
            .add_node(node_id.clone(), ports.into_values())
            .map_err(|error| FlowSnapshotError::GraphConstruction {
                allocation_id: None,
                reason: error.to_string(),
            })?;
    }

    let mut active_allocation_ids = Vec::with_capacity(active.len());
    for admitted in active {
        let allocation = admitted.allocation();
        let link = topology
            .link(&allocation.link_id)
            .ok_or_else(|| FlowSnapshotError::UnknownLink(allocation.link_id.clone()))?;
        graph
            .connect(ResourceEdge {
                id: allocation.id.clone(),
                from_node: link.from_node.clone(),
                from_port: link.from_port.clone(),
                to_node: link.to_node.clone(),
                to_port: link.to_port.clone(),
                amount: allocation.sent,
                loss_fraction: link.loss_fraction,
            })
            .map_err(|error| FlowSnapshotError::GraphConstruction {
                allocation_id: Some(allocation.id.clone()),
                reason: error.to_string(),
            })?;
        active_allocation_ids.push(allocation.id.clone());
    }
    active_allocation_ids.sort();

    Ok(CommittedFlowSnapshot {
        observed_at,
        graph,
        active_allocation_ids,
    })
}

fn retain_port(
    topology: &ResourceTopology,
    nodes: &mut BTreeMap<String, BTreeMap<String, ResourcePort>>,
    node_id: &str,
    port_id: &str,
) -> Result<(), FlowSnapshotError> {
    let port = topology
        .port(node_id, port_id)
        .ok_or_else(|| FlowSnapshotError::UnknownPort {
            node_id: node_id.to_owned(),
            port_id: port_id.to_owned(),
        })?
        .clone();

    let ports = nodes.entry(node_id.to_owned()).or_default();
    if let Some(existing) = ports.get(port_id) {
        if existing != &port {
            return Err(FlowSnapshotError::EndpointInconsistency {
                node_id: node_id.to_owned(),
                port_id: port_id.to_owned(),
            });
        }
        return Ok(());
    }
    ports.insert(port_id.to_owned(), port);
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum FlowSnapshotError {
    #[error("unknown topology link {0}")]
    UnknownLink(String),
    #[error("unknown topology port {node_id}/{port_id}")]
    UnknownPort { node_id: String, port_id: String },
    #[error("topology returned inconsistent definitions for port {node_id}/{port_id}")]
    EndpointInconsistency { node_id: String, port_id: String },
    #[error("committed graph construction failed for {allocation_id:?}: {reason}")]
    GraphConstruction {
        allocation_id: Option<String>,
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone};
    use symthaea_resource_allocation::PlannedAllocation;
    use symthaea_resource_capacity::{
        CapacitySchedule, CapacitySemantics, CapacitySubject, CapacityWindow,
    };
    use symthaea_resource_model::{
        PortDirection, ResourceAmount, ResourceKind, ResourceUnit,
    };
    use symthaea_resource_topology::ResourceLink;

    fn t0() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 7, 12, 0, 0).unwrap()
    }

    fn power(value: f64) -> ResourceAmount {
        ResourceAmount::new(ResourceKind::Electricity, ResourceUnit::Watt, value).unwrap()
    }

    fn port(id: &str, direction: PortDirection, capacity: f64) -> ResourcePort {
        ResourcePort {
            id: id.into(),
            direction,
            capacity: power(capacity),
        }
    }

    fn topology(loss_fraction: f64) -> ResourceTopology {
        let mut topology = ResourceTopology::default();
        topology
            .add_node("source", [port("out", PortDirection::Output, 100.0)])
            .unwrap();
        topology
            .add_node("sink", [port("in", PortDirection::Input, 100.0)])
            .unwrap();
        topology
            .add_link(ResourceLink {
                id: "line".into(),
                from_node: "source".into(),
                from_port: "out".into(),
                to_node: "sink".into(),
                to_port: "in".into(),
                capacity: power(100.0),
                loss_fraction,
            })
            .unwrap();
        topology
    }

    fn capacity_schedule(topology: &ResourceTopology) -> CapacitySchedule {
        let mut schedule = CapacitySchedule::default();
        let entries = [
            (
                "source-cap",
                CapacitySubject::Port {
                    node_id: "source".into(),
                    port_id: "out".into(),
                },
            ),
            (
                "line-cap",
                CapacitySubject::Link {
                    link_id: "line".into(),
                },
            ),
            (
                "sink-cap",
                CapacitySubject::Port {
                    node_id: "sink".into(),
                    port_id: "in".into(),
                },
            ),
        ];
        for (id, subject) in entries {
            schedule
                .add_window(
                    topology,
                    CapacityWindow {
                        id: id.into(),
                        subject,
                        valid_from: t0(),
                        valid_until: t0() + Duration::hours(1),
                        capacity: power(100.0),
                        semantics: CapacitySemantics::Concurrent,
                    },
                )
                .unwrap();
        }
        schedule
    }

    fn allocation(id: &str, start: i64, end: i64, value: f64) -> PlannedAllocation {
        PlannedAllocation {
            id: id.into(),
            link_id: "line".into(),
            valid_from: t0() + Duration::minutes(start),
            valid_until: t0() + Duration::minutes(end),
            sent: power(value),
        }
    }

    #[test]
    fn snapshot_selects_exact_half_open_active_set() {
        let topology = topology(0.0);
        let capacities = capacity_schedule(&topology);
        let mut book = AllocationBook::default();
        book.admit(&topology, &capacities, allocation("a", 0, 30, 60.0))
            .unwrap();
        book.admit(&topology, &capacities, allocation("b", 30, 60, 40.0))
            .unwrap();

        let first = materialize_flow_snapshot(
            &topology,
            &book,
            t0() + Duration::minutes(15),
        )
        .unwrap();
        assert_eq!(first.active_allocation_ids(), &["a".to_string()]);
        assert_eq!(first.graph().edges().len(), 1);
        assert_eq!(first.graph().port_utilization("source", "out").unwrap(), 60.0);

        let boundary = materialize_flow_snapshot(
            &topology,
            &book,
            t0() + Duration::minutes(30),
        )
        .unwrap();
        assert_eq!(boundary.active_allocation_ids(), &["b".to_string()]);
        assert_eq!(boundary.graph().port_utilization("source", "out").unwrap(), 40.0);
    }

    #[test]
    fn snapshot_preserves_sent_delivered_loss_semantics() {
        let topology = topology(0.2);
        let capacities = capacity_schedule(&topology);
        let mut book = AllocationBook::default();
        book.admit(&topology, &capacities, allocation("lossy", 0, 60, 100.0))
            .unwrap();

        let snapshot = materialize_flow_snapshot(
            &topology,
            &book,
            t0() + Duration::minutes(10),
        )
        .unwrap();
        let edge = &snapshot.graph().edges()[0];
        assert_eq!(edge.amount.value, 100.0);
        assert_eq!(edge.delivered().unwrap().value, 80.0);
        assert_eq!(snapshot.graph().port_utilization("source", "out").unwrap(), 100.0);
        assert_eq!(snapshot.graph().port_utilization("sink", "in").unwrap(), 80.0);
    }

    #[test]
    fn snapshot_outside_all_allocations_is_empty() {
        let topology = topology(0.0);
        let capacities = capacity_schedule(&topology);
        let mut book = AllocationBook::default();
        book.admit(&topology, &capacities, allocation("a", 0, 30, 60.0))
            .unwrap();

        let snapshot = materialize_flow_snapshot(
            &topology,
            &book,
            t0() + Duration::minutes(45),
        )
        .unwrap();
        assert!(snapshot.active_allocation_ids().is_empty());
        assert!(snapshot.graph().edges().is_empty());
    }

    #[test]
    fn simultaneous_admitted_allocations_reproduce_aggregate_port_utilization() {
        let topology = topology(0.0);
        let capacities = capacity_schedule(&topology);
        let mut book = AllocationBook::default();
        book.admit(&topology, &capacities, allocation("a", 0, 60, 60.0))
            .unwrap();
        book.admit(&topology, &capacities, allocation("b", 0, 60, 40.0))
            .unwrap();

        let snapshot = materialize_flow_snapshot(
            &topology,
            &book,
            t0() + Duration::minutes(10),
        )
        .unwrap();
        assert_eq!(snapshot.graph().edges().len(), 2);
        assert_eq!(snapshot.graph().port_utilization("source", "out").unwrap(), 100.0);
        assert_eq!(snapshot.graph().port_utilization("sink", "in").unwrap(), 100.0);
        assert_eq!(
            snapshot.active_allocation_ids(),
            &["a".to_string(), "b".to_string()]
        );
    }
}
