// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed resource algebra for civic and engineered systems.
//!
//! The goal of this module is deliberately small: represent resource ports,
//! transfers, envelopes, balances, and graph topology without embedding any
//! particular facility topology or optimizer. Higher-level infrastructure can
//! then compose these primitives recursively.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Resource categories that can be exchanged across engineered/civic boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ResourceKind {
    Electricity,
    Compute,
    ThermalEnergy,
    CoolingCapacity,
    Water,
    NetworkBandwidth,
    StorageCapacity,
    Material,
}

/// Direction of a resource port relative to its owning node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PortDirection {
    Input,
    Output,
    Bidirectional,
}

/// A scalar resource quantity with an explicit engineering unit label.
///
/// Values are intentionally kept as `f64` here because this type is also used at
/// serialization boundaries. Domain adapters should normalize to canonical units
/// before constructing values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceQuantity {
    pub kind: ResourceKind,
    pub value: f64,
    pub unit: String,
}

impl ResourceQuantity {
    pub fn new(kind: ResourceKind, value: f64, unit: impl Into<String>) -> Self {
        Self {
            kind,
            value,
            unit: unit.into(),
        }
    }

    pub fn is_finite_nonnegative(&self) -> bool {
        self.value.is_finite() && self.value >= 0.0
    }

    pub fn compatible_with(&self, other: &Self) -> bool {
        self.kind == other.kind && self.unit == other.unit
    }
}

/// Named resource boundary on a node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourcePort {
    pub id: String,
    pub kind: ResourceKind,
    pub direction: PortDirection,
    pub capacity: ResourceQuantity,
}

impl ResourcePort {
    pub fn accepts(&self, quantity: &ResourceQuantity) -> bool {
        matches!(self.direction, PortDirection::Input | PortDirection::Bidirectional)
            && self.capacity.compatible_with(quantity)
            && quantity.is_finite_nonnegative()
            && quantity.value <= self.capacity.value
    }

    pub fn provides(&self, quantity: &ResourceQuantity) -> bool {
        matches!(self.direction, PortDirection::Output | PortDirection::Bidirectional)
            && self.capacity.compatible_with(quantity)
            && quantity.is_finite_nonnegative()
            && quantity.value <= self.capacity.value
    }
}

/// Maximum externally visible capabilities of a resource node.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResourceEnvelope {
    pub capacities: BTreeMap<ResourceKind, ResourceQuantity>,
}

impl ResourceEnvelope {
    pub fn insert(&mut self, quantity: ResourceQuantity) -> Option<ResourceQuantity> {
        self.capacities.insert(quantity.kind, quantity)
    }

    pub fn get(&self, kind: ResourceKind) -> Option<&ResourceQuantity> {
        self.capacities.get(&kind)
    }
}

/// Time-windowed resource accounting for one node boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceBalance {
    pub kind: ResourceKind,
    pub unit: String,
    pub produced: f64,
    pub consumed: f64,
    pub imported: f64,
    pub exported: f64,
    pub storage_delta: f64,
    pub modeled_losses: f64,
}

impl ResourceBalance {
    /// Conservation residual. A well-formed closed accounting window should be
    /// approximately zero after explicit storage changes and modeled losses.
    pub fn residual(&self) -> f64 {
        self.produced + self.imported
            - self.consumed
            - self.exported
            - self.storage_delta
            - self.modeled_losses
    }

    pub fn is_conserved_within(&self, tolerance: f64) -> bool {
        tolerance.is_finite() && tolerance >= 0.0 && self.residual().abs() <= tolerance
    }
}

/// Directed resource transfer between two named ports.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceEdge {
    pub id: String,
    pub from_node: String,
    pub from_port: String,
    pub to_node: String,
    pub to_port: String,
    pub quantity: ResourceQuantity,
    /// Explicit fractional transfer loss in `[0, 1]`.
    pub loss_fraction: f64,
}

impl ResourceEdge {
    pub fn delivered_quantity(&self) -> Option<ResourceQuantity> {
        if !self.quantity.is_finite_nonnegative()
            || !self.loss_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.loss_fraction)
        {
            return None;
        }
        Some(ResourceQuantity::new(
            self.quantity.kind,
            self.quantity.value * (1.0 - self.loss_fraction),
            self.quantity.unit.clone(),
        ))
    }
}

/// Minimal graph container for resource topology.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResourceGraph {
    pub nodes: BTreeMap<String, Vec<ResourcePort>>,
    pub edges: Vec<ResourceEdge>,
}

impl ResourceGraph {
    pub fn add_node(&mut self, id: impl Into<String>, ports: Vec<ResourcePort>) -> bool {
        self.nodes.insert(id.into(), ports).is_none()
    }

    pub fn add_edge(&mut self, edge: ResourceEdge) -> Result<(), &'static str> {
        let from_ports = self.nodes.get(&edge.from_node).ok_or("unknown source node")?;
        let to_ports = self.nodes.get(&edge.to_node).ok_or("unknown destination node")?;
        let from = from_ports
            .iter()
            .find(|port| port.id == edge.from_port)
            .ok_or("unknown source port")?;
        let to = to_ports
            .iter()
            .find(|port| port.id == edge.to_port)
            .ok_or("unknown destination port")?;

        if !from.provides(&edge.quantity) {
            return Err("source port cannot provide quantity");
        }
        if !to.accepts(&edge.quantity) {
            return Err("destination port cannot accept quantity");
        }
        if edge.delivered_quantity().is_none() {
            return Err("invalid loss fraction or quantity");
        }
        self.edges.push(edge);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn port(id: &str, direction: PortDirection, capacity_kw: f64) -> ResourcePort {
        ResourcePort {
            id: id.into(),
            kind: ResourceKind::Electricity,
            direction,
            capacity: ResourceQuantity::new(ResourceKind::Electricity, capacity_kw, "kW"),
        }
    }

    #[test]
    fn rejects_direction_mismatch_and_over_capacity() {
        let input = port("in", PortDirection::Input, 100.0);
        let output = port("out", PortDirection::Output, 100.0);
        let fifty = ResourceQuantity::new(ResourceKind::Electricity, 50.0, "kW");
        let too_much = ResourceQuantity::new(ResourceKind::Electricity, 150.0, "kW");

        assert!(input.accepts(&fifty));
        assert!(!input.provides(&fifty));
        assert!(output.provides(&fifty));
        assert!(!output.accepts(&fifty));
        assert!(!input.accepts(&too_much));
    }

    #[test]
    fn rejects_unit_mismatch() {
        let input = port("in", PortDirection::Input, 100.0);
        let joules = ResourceQuantity::new(ResourceKind::Electricity, 10.0, "J");
        assert!(!input.accepts(&joules));
    }

    #[test]
    fn accounts_for_explicit_losses() {
        let edge = ResourceEdge {
            id: "line".into(),
            from_node: "a".into(),
            from_port: "out".into(),
            to_node: "b".into(),
            to_port: "in".into(),
            quantity: ResourceQuantity::new(ResourceKind::Electricity, 100.0, "kW"),
            loss_fraction: 0.05,
        };
        assert_eq!(edge.delivered_quantity().unwrap().value, 95.0);
    }

    #[test]
    fn resource_balance_exposes_conservation_residual() {
        let balance = ResourceBalance {
            kind: ResourceKind::Electricity,
            unit: "kWh".into(),
            produced: 80.0,
            consumed: 70.0,
            imported: 30.0,
            exported: 20.0,
            storage_delta: 15.0,
            modeled_losses: 5.0,
        };
        assert!(balance.is_conserved_within(1e-9));
    }

    #[test]
    fn graph_validates_port_contracts() {
        let mut graph = ResourceGraph::default();
        graph.add_node("generator", vec![port("out", PortDirection::Output, 120.0)]);
        graph.add_node("load", vec![port("in", PortDirection::Input, 120.0)]);

        let edge = ResourceEdge {
            id: "feeder".into(),
            from_node: "generator".into(),
            from_port: "out".into(),
            to_node: "load".into(),
            to_port: "in".into(),
            quantity: ResourceQuantity::new(ResourceKind::Electricity, 80.0, "kW"),
            loss_fraction: 0.02,
        };
        assert!(graph.add_edge(edge).is_ok());
    }
}
