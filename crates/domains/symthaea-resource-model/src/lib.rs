// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed resource-flow algebra for multiscale civic and engineered systems.
//!
//! This crate is intentionally dependency-light and topology-agnostic. It models
//! externally visible resources, ports, transfers, and conservation accounting so
//! higher layers can compose racks, buildings, microgrids, habitats, communities,
//! and federations without hard-coding one facility shape into the foundation.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

const PORT_EPSILON: f64 = 1e-9;

/// A conserved or capacity-bearing resource exchanged between system boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ResourceKind {
    Electricity,
    ThermalEnergy,
    CoolingCapacity,
    Compute,
    Storage,
    NetworkBandwidth,
    Water,
    Material,
}

/// Canonical unit vocabulary used at serialized resource boundaries.
///
/// Domain adapters may use richer internal unit systems, but must normalize to one
/// of these units before constructing a resource transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ResourceUnit {
    Joule,
    Watt,
    CpuSecond,
    GpuSecond,
    Byte,
    BitPerSecond,
    CubicMeter,
    CubicMeterPerSecond,
    Kilogram,
}

impl ResourceUnit {
    /// Whether this unit is meaningful for the supplied resource kind.
    pub fn supports(self, kind: ResourceKind) -> bool {
        match kind {
            ResourceKind::Electricity => matches!(self, Self::Joule | Self::Watt),
            ResourceKind::ThermalEnergy => matches!(self, Self::Joule | Self::Watt),
            ResourceKind::CoolingCapacity => matches!(self, Self::Watt),
            ResourceKind::Compute => matches!(self, Self::CpuSecond | Self::GpuSecond),
            ResourceKind::Storage => matches!(self, Self::Byte),
            ResourceKind::NetworkBandwidth => matches!(self, Self::BitPerSecond),
            ResourceKind::Water => {
                matches!(self, Self::CubicMeter | Self::CubicMeterPerSecond)
            }
            ResourceKind::Material => matches!(self, Self::Kilogram),
        }
    }
}

/// Stable key identifying one resource dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ResourceKey {
    pub kind: ResourceKind,
    pub unit: ResourceUnit,
}

impl ResourceKey {
    pub fn new(kind: ResourceKind, unit: ResourceUnit) -> Result<Self, ResourceError> {
        if !unit.supports(kind) {
            return Err(ResourceError::IncompatibleUnit { kind, unit });
        }
        Ok(Self { kind, unit })
    }
}

/// Non-negative amount or rate in a canonical resource dimension.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResourceAmount {
    pub key: ResourceKey,
    pub value: f64,
}

impl ResourceAmount {
    pub fn new(kind: ResourceKind, unit: ResourceUnit, value: f64) -> Result<Self, ResourceError> {
        let key = ResourceKey::new(kind, unit)?;
        if !value.is_finite() || value < 0.0 {
            return Err(ResourceError::InvalidNonNegativeValue(value));
        }
        Ok(Self { key, value })
    }

    pub fn scaled(self, factor: f64) -> Result<Self, ResourceError> {
        if !factor.is_finite() || factor < 0.0 {
            return Err(ResourceError::InvalidNonNegativeValue(factor));
        }
        Self::new(self.key.kind, self.key.unit, self.value * factor)
    }
}

/// Direction of a port relative to the node that owns it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PortDirection {
    Input,
    Output,
    Bidirectional,
}

/// Externally visible resource boundary on a node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourcePort {
    pub id: String,
    pub direction: PortDirection,
    pub capacity: ResourceAmount,
}

impl ResourcePort {
    pub fn accepts(&self, amount: ResourceAmount) -> bool {
        matches!(self.direction, PortDirection::Input | PortDirection::Bidirectional)
            && self.capacity.key == amount.key
            && amount.value <= self.capacity.value + PORT_EPSILON
    }

    pub fn provides(&self, amount: ResourceAmount) -> bool {
        matches!(self.direction, PortDirection::Output | PortDirection::Bidirectional)
            && self.capacity.key == amount.key
            && amount.value <= self.capacity.value + PORT_EPSILON
    }
}

/// Maximum externally visible capabilities of a resource node.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResourceEnvelope {
    capacities: BTreeMap<ResourceKey, f64>,
}

impl ResourceEnvelope {
    pub fn insert(&mut self, amount: ResourceAmount) {
        self.capacities
            .entry(amount.key)
            .and_modify(|value| *value += amount.value)
            .or_insert(amount.value);
    }

    pub fn capacity(&self, key: ResourceKey) -> f64 {
        self.capacities.get(&key).copied().unwrap_or(0.0)
    }

    pub fn iter(&self) -> impl Iterator<Item = (ResourceKey, f64)> + '_ {
        self.capacities.iter().map(|(key, value)| (*key, *value))
    }
}

/// Accounting for one resource dimension over a defined observation window.
///
/// `storage_delta` is signed: positive means net storage increase, negative means
/// storage release. All other terms must be finite and non-negative.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResourceBalance {
    pub key: ResourceKey,
    pub produced: f64,
    pub consumed: f64,
    pub imported: f64,
    pub exported: f64,
    pub storage_delta: f64,
    pub modeled_losses: f64,
}

impl ResourceBalance {
    pub fn validate(&self) -> Result<(), ResourceError> {
        for value in [
            self.produced,
            self.consumed,
            self.imported,
            self.exported,
            self.modeled_losses,
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(ResourceError::InvalidNonNegativeValue(value));
            }
        }
        if !self.storage_delta.is_finite() {
            return Err(ResourceError::InvalidSignedValue(self.storage_delta));
        }
        Ok(())
    }

    /// Positive residual means unaccounted inflow; negative means unaccounted outflow.
    pub fn residual(&self) -> Result<f64, ResourceError> {
        self.validate()?;
        Ok(self.produced + self.imported
            - self.consumed
            - self.exported
            - self.storage_delta
            - self.modeled_losses)
    }

    pub fn conserved_within(&self, tolerance: f64) -> Result<bool, ResourceError> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(ResourceError::InvalidNonNegativeValue(tolerance));
        }
        Ok(self.residual()?.abs() <= tolerance)
    }
}

/// Directed resource transfer between two named node ports.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceEdge {
    pub id: String,
    pub from_node: String,
    pub from_port: String,
    pub to_node: String,
    pub to_port: String,
    pub amount: ResourceAmount,
    /// Explicit fractional transfer loss in `[0, 1]`.
    pub loss_fraction: f64,
}

impl ResourceEdge {
    pub fn delivered(&self) -> Result<ResourceAmount, ResourceError> {
        if !self.loss_fraction.is_finite() || !(0.0..=1.0).contains(&self.loss_fraction) {
            return Err(ResourceError::InvalidLossFraction(self.loss_fraction));
        }
        self.amount.scaled(1.0 - self.loss_fraction)
    }

    pub fn lost(&self) -> Result<ResourceAmount, ResourceError> {
        if !self.loss_fraction.is_finite() || !(0.0..=1.0).contains(&self.loss_fraction) {
            return Err(ResourceError::InvalidLossFraction(self.loss_fraction));
        }
        self.amount.scaled(self.loss_fraction)
    }
}

/// Minimal topology container that validates transfers against port contracts.
///
/// Port capacity is shared across every edge attached to that port. For a
/// bidirectional port, simultaneous inbound and outbound utilization consume the
/// same physical capacity budget.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResourceGraph {
    nodes: BTreeMap<String, BTreeMap<String, ResourcePort>>,
    edges: Vec<ResourceEdge>,
}

impl ResourceGraph {
    pub fn add_node(
        &mut self,
        node_id: impl Into<String>,
        ports: impl IntoIterator<Item = ResourcePort>,
    ) -> Result<(), ResourceError> {
        let node_id = node_id.into();
        if self.nodes.contains_key(&node_id) {
            return Err(ResourceError::DuplicateNode(node_id));
        }
        let mut port_map = BTreeMap::new();
        for port in ports {
            if port_map.insert(port.id.clone(), port).is_some() {
                return Err(ResourceError::DuplicatePort(node_id));
            }
        }
        self.nodes.insert(node_id, port_map);
        Ok(())
    }

    pub fn connect(&mut self, edge: ResourceEdge) -> Result<(), ResourceError> {
        if self.edges.iter().any(|existing| existing.id == edge.id) {
            return Err(ResourceError::DuplicateEdge(edge.id));
        }
        if edge.from_node == edge.to_node && edge.from_port == edge.to_port {
            return Err(ResourceError::SelfLoopPort {
                node: edge.from_node,
                port: edge.from_port,
            });
        }

        let source = self
            .nodes
            .get(&edge.from_node)
            .ok_or_else(|| ResourceError::UnknownNode(edge.from_node.clone()))?
            .get(&edge.from_port)
            .ok_or_else(|| ResourceError::UnknownPort {
                node: edge.from_node.clone(),
                port: edge.from_port.clone(),
            })?;
        let destination = self
            .nodes
            .get(&edge.to_node)
            .ok_or_else(|| ResourceError::UnknownNode(edge.to_node.clone()))?
            .get(&edge.to_port)
            .ok_or_else(|| ResourceError::UnknownPort {
                node: edge.to_node.clone(),
                port: edge.to_port.clone(),
            })?;

        let delivered = edge.delivered()?;
        if !source.provides(edge.amount) {
            return Err(ResourceError::SourceContractViolation(edge.id));
        }
        if !destination.accepts(delivered) {
            return Err(ResourceError::DestinationContractViolation(edge.id));
        }

        let source_capacity = source.capacity.value;
        let destination_capacity = destination.capacity.value;
        let source_used = self.port_utilization(&edge.from_node, &edge.from_port)?;
        let destination_used = self.port_utilization(&edge.to_node, &edge.to_port)?;
        let attempted_source = source_used + edge.amount.value;
        if attempted_source > source_capacity + PORT_EPSILON {
            return Err(ResourceError::AggregatePortCapacityExceeded {
                node: edge.from_node.clone(),
                port: edge.from_port.clone(),
                capacity: source_capacity,
                attempted: attempted_source,
            });
        }
        let attempted_destination = destination_used + delivered.value;
        if attempted_destination > destination_capacity + PORT_EPSILON {
            return Err(ResourceError::AggregatePortCapacityExceeded {
                node: edge.to_node.clone(),
                port: edge.to_port.clone(),
                capacity: destination_capacity,
                attempted: attempted_destination,
            });
        }

        self.edges.push(edge);
        Ok(())
    }

    /// Sum physical utilization already committed to a port.
    ///
    /// Outbound edges consume sent quantity; inbound edges consume delivered
    /// quantity after explicit transfer loss.
    pub fn port_utilization(&self, node: &str, port: &str) -> Result<f64, ResourceError> {
        self.nodes
            .get(node)
            .ok_or_else(|| ResourceError::UnknownNode(node.to_owned()))?
            .get(port)
            .ok_or_else(|| ResourceError::UnknownPort {
                node: node.to_owned(),
                port: port.to_owned(),
            })?;

        let mut used = 0.0;
        for edge in &self.edges {
            if edge.from_node == node && edge.from_port == port {
                used += edge.amount.value;
            }
            if edge.to_node == node && edge.to_port == port {
                used += edge.delivered()?.value;
            }
        }
        Ok(used)
    }

    pub fn edges(&self) -> &[ResourceEdge] {
        &self.edges
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum ResourceError {
    #[error("unit {unit:?} is incompatible with resource kind {kind:?}")]
    IncompatibleUnit {
        kind: ResourceKind,
        unit: ResourceUnit,
    },
    #[error("expected a finite non-negative value, got {0}")]
    InvalidNonNegativeValue(f64),
    #[error("expected a finite signed value, got {0}")]
    InvalidSignedValue(f64),
    #[error("loss fraction must be finite and within [0, 1], got {0}")]
    InvalidLossFraction(f64),
    #[error("duplicate node {0}")]
    DuplicateNode(String),
    #[error("duplicate port on node {0}")]
    DuplicatePort(String),
    #[error("duplicate resource edge {0}")]
    DuplicateEdge(String),
    #[error("unknown node {0}")]
    UnknownNode(String),
    #[error("unknown port {node}/{port}")]
    UnknownPort { node: String, port: String },
    #[error("resource edge cannot loop from a port back to itself: {node}/{port}")]
    SelfLoopPort { node: String, port: String },
    #[error("source port contract rejected edge {0}")]
    SourceContractViolation(String),
    #[error("destination port contract rejected edge {0}")]
    DestinationContractViolation(String),
    #[error(
        "aggregate utilization {attempted} exceeds capacity {capacity} on port {node}/{port}"
    )]
    AggregatePortCapacityExceeded {
        node: String,
        port: String,
        capacity: f64,
        attempted: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn electricity(value: f64) -> ResourceAmount {
        ResourceAmount::new(ResourceKind::Electricity, ResourceUnit::Watt, value).unwrap()
    }

    fn port(id: &str, direction: PortDirection, capacity_w: f64) -> ResourcePort {
        ResourcePort {
            id: id.into(),
            direction,
            capacity: electricity(capacity_w),
        }
    }

    fn edge(id: &str, from: (&str, &str), to: (&str, &str), watts: f64) -> ResourceEdge {
        ResourceEdge {
            id: id.into(),
            from_node: from.0.into(),
            from_port: from.1.into(),
            to_node: to.0.into(),
            to_port: to.1.into(),
            amount: electricity(watts),
            loss_fraction: 0.0,
        }
    }

    #[test]
    fn incompatible_unit_is_rejected() {
        let result = ResourceAmount::new(ResourceKind::Water, ResourceUnit::GpuSecond, 1.0);
        assert!(matches!(result, Err(ResourceError::IncompatibleUnit { .. })));
    }

    #[test]
    fn ports_enforce_direction_and_capacity() {
        let input = port("in", PortDirection::Input, 100.0);
        let output = port("out", PortDirection::Output, 100.0);
        assert!(input.accepts(electricity(50.0)));
        assert!(!input.provides(electricity(50.0)));
        assert!(output.provides(electricity(50.0)));
        assert!(!output.accepts(electricity(50.0)));
        assert!(!input.accepts(electricity(101.0)));
    }

    #[test]
    fn envelope_keeps_distinct_resource_dimensions() {
        let mut envelope = ResourceEnvelope::default();
        envelope.insert(electricity(80.0));
        envelope.insert(electricity(20.0));
        let gpu = ResourceAmount::new(ResourceKind::Compute, ResourceUnit::GpuSecond, 4.0).unwrap();
        envelope.insert(gpu);
        assert_eq!(envelope.capacity(electricity(0.0).key), 100.0);
        assert_eq!(envelope.capacity(gpu.key), 4.0);
    }

    #[test]
    fn transfer_losses_are_explicit() {
        let edge = ResourceEdge {
            id: "feeder".into(),
            from_node: "generation".into(),
            from_port: "out".into(),
            to_node: "load".into(),
            to_port: "in".into(),
            amount: electricity(100.0),
            loss_fraction: 0.05,
        };
        assert_eq!(edge.delivered().unwrap().value, 95.0);
        assert_eq!(edge.lost().unwrap().value, 5.0);
    }

    #[test]
    fn balance_exposes_conservation_residual() {
        let balance = ResourceBalance {
            key: electricity(0.0).key,
            produced: 80.0,
            consumed: 70.0,
            imported: 30.0,
            exported: 20.0,
            storage_delta: 15.0,
            modeled_losses: 5.0,
        };
        assert!(balance.conserved_within(1e-9).unwrap());
        assert_eq!(balance.residual().unwrap(), 0.0);
    }

    #[test]
    fn storage_release_uses_signed_delta() {
        let balance = ResourceBalance {
            key: electricity(0.0).key,
            produced: 0.0,
            consumed: 10.0,
            imported: 0.0,
            exported: 0.0,
            storage_delta: -10.0,
            modeled_losses: 0.0,
        };
        assert!(balance.conserved_within(1e-9).unwrap());
    }

    #[test]
    fn graph_validates_resource_contracts() {
        let mut graph = ResourceGraph::default();
        graph
            .add_node("generation", [port("out", PortDirection::Output, 120.0)])
            .unwrap();
        graph
            .add_node("load", [port("in", PortDirection::Input, 120.0)])
            .unwrap();
        graph
            .connect(ResourceEdge {
                id: "feeder".into(),
                from_node: "generation".into(),
                from_port: "out".into(),
                to_node: "load".into(),
                to_port: "in".into(),
                amount: electricity(80.0),
                loss_fraction: 0.02,
            })
            .unwrap();
        assert_eq!(graph.edges().len(), 1);
        assert_eq!(graph.port_utilization("generation", "out").unwrap(), 80.0);
        assert_eq!(graph.port_utilization("load", "in").unwrap(), 78.4);
    }

    #[test]
    fn graph_rejects_single_over_capacity_transfer() {
        let mut graph = ResourceGraph::default();
        graph
            .add_node("generation", [port("out", PortDirection::Output, 50.0)])
            .unwrap();
        graph
            .add_node("load", [port("in", PortDirection::Input, 100.0)])
            .unwrap();
        let result = graph.connect(edge(
            "overload",
            ("generation", "out"),
            ("load", "in"),
            75.0,
        ));
        assert!(matches!(result, Err(ResourceError::SourceContractViolation(_))));
    }

    #[test]
    fn multiple_edges_share_source_port_capacity() {
        let mut graph = ResourceGraph::default();
        graph
            .add_node("source", [port("out", PortDirection::Output, 100.0)])
            .unwrap();
        graph
            .add_node("a", [port("in", PortDirection::Input, 100.0)])
            .unwrap();
        graph
            .add_node("b", [port("in", PortDirection::Input, 100.0)])
            .unwrap();
        graph
            .connect(edge("a", ("source", "out"), ("a", "in"), 60.0))
            .unwrap();
        let result = graph.connect(edge("b", ("source", "out"), ("b", "in"), 50.0));
        assert!(matches!(
            result,
            Err(ResourceError::AggregatePortCapacityExceeded {
                node,
                attempted,
                ..
            }) if node == "source" && (attempted - 110.0).abs() < 1e-9
        ));
    }

    #[test]
    fn multiple_edges_share_destination_port_capacity_after_losses() {
        let mut graph = ResourceGraph::default();
        graph
            .add_node("a", [port("out", PortDirection::Output, 100.0)])
            .unwrap();
        graph
            .add_node("b", [port("out", PortDirection::Output, 100.0)])
            .unwrap();
        graph
            .add_node("sink", [port("in", PortDirection::Input, 100.0)])
            .unwrap();
        let mut first = edge("a", ("a", "out"), ("sink", "in"), 60.0);
        first.loss_fraction = 0.1; // 54 W delivered
        graph.connect(first).unwrap();
        let result = graph.connect(edge("b", ("b", "out"), ("sink", "in"), 50.0));
        assert!(matches!(
            result,
            Err(ResourceError::AggregatePortCapacityExceeded { node, port, .. })
                if node == "sink" && port == "in"
        ));
    }

    #[test]
    fn bidirectional_port_shares_one_capacity_budget() {
        let mut graph = ResourceGraph::default();
        graph
            .add_node("bus", [port("ac", PortDirection::Bidirectional, 100.0)])
            .unwrap();
        graph
            .add_node("load", [port("in", PortDirection::Input, 100.0)])
            .unwrap();
        graph
            .add_node("source", [port("out", PortDirection::Output, 100.0)])
            .unwrap();
        graph
            .connect(edge("out", ("bus", "ac"), ("load", "in"), 70.0))
            .unwrap();
        let result = graph.connect(edge("in", ("source", "out"), ("bus", "ac"), 40.0));
        assert!(matches!(
            result,
            Err(ResourceError::AggregatePortCapacityExceeded { node, port, .. })
                if node == "bus" && port == "ac"
        ));
    }

    #[test]
    fn duplicate_edge_id_is_rejected() {
        let mut graph = ResourceGraph::default();
        graph
            .add_node("source", [port("out", PortDirection::Output, 100.0)])
            .unwrap();
        graph
            .add_node("sink", [port("in", PortDirection::Input, 100.0)])
            .unwrap();
        graph
            .connect(edge("same", ("source", "out"), ("sink", "in"), 20.0))
            .unwrap();
        assert!(matches!(
            graph.connect(edge("same", ("source", "out"), ("sink", "in"), 20.0)),
            Err(ResourceError::DuplicateEdge(id)) if id == "same"
        ));
    }
}
