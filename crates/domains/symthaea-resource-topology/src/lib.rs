// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Potential resource-connectivity topology for multiscale infrastructure.
//!
//! This crate answers a deliberately narrower question than
//! `symthaea-resource-model::ResourceGraph`:
//!
//! - `ResourceTopology` describes where a resource *could* flow and the maximum
//!   physical capacity of each route.
//! - `ResourceGraph` describes resource flow that is already committed in one
//!   current/static flow snapshot.
//!
//! A topology link therefore never consumes port capacity merely by existing.
//! Multiple potential links may share one physical port even when their summed
//! link capacities exceed the port's simultaneous capacity. A later allocation or
//! scheduling layer must decide which links are active in a time window and must
//! enforce shared-port concurrency there.
//!
//! This crate contains no scheduler, reservation, lease, optimizer, authority, or
//! actuator path.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use symthaea_resource_model::{ResourceAmount, ResourcePort};
use thiserror::Error;

/// One directed physical route through which a typed resource may flow.
///
/// `capacity` is potential sent capacity, not a reservation and not evidence that
/// any resource is currently flowing. Destination capacity is checked against the
/// delivered amount after `loss_fraction`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceLink {
    pub id: String,
    pub from_node: String,
    pub from_port: String,
    pub to_node: String,
    pub to_port: String,
    pub capacity: ResourceAmount,
    /// Fraction of sent quantity lost across the route, in `[0, 1]`.
    pub loss_fraction: f64,
}

impl ResourceLink {
    /// Maximum quantity that can arrive at the destination boundary.
    pub fn delivered_capacity(&self) -> Result<ResourceAmount, TopologyError> {
        validate_loss(self.loss_fraction)?;
        self.capacity
            .scaled(1.0 - self.loss_fraction)
            .map_err(|error| TopologyError::InvalidCapacity(error.to_string()))
    }
}

/// Static physical connectivity without allocation state.
///
/// Nodes own reusable `ResourcePort` contracts from `symthaea-resource-model` so
/// topology, committed flow, and later allocation layers share the same typed
/// resource vocabulary. Links are potential routes only: adding a link never
/// changes a port-utilization counter and never creates execution authority.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResourceTopology {
    nodes: BTreeMap<String, BTreeMap<String, ResourcePort>>,
    links: BTreeMap<String, ResourceLink>,
}

impl ResourceTopology {
    pub fn add_node(
        &mut self,
        node_id: impl Into<String>,
        ports: impl IntoIterator<Item = ResourcePort>,
    ) -> Result<(), TopologyError> {
        let node_id = node_id.into();
        if node_id.trim().is_empty() {
            return Err(TopologyError::EmptyNodeId);
        }
        if self.nodes.contains_key(&node_id) {
            return Err(TopologyError::DuplicateNode(node_id));
        }

        let mut port_map = BTreeMap::new();
        for port in ports {
            if port.id.trim().is_empty() {
                return Err(TopologyError::EmptyPortId(node_id));
            }
            if port_map.insert(port.id.clone(), port).is_some() {
                return Err(TopologyError::DuplicatePort(node_id));
            }
        }
        self.nodes.insert(node_id, port_map);
        Ok(())
    }

    /// Add one potential route without reserving or consuming endpoint capacity.
    pub fn add_link(&mut self, link: ResourceLink) -> Result<(), TopologyError> {
        if link.id.trim().is_empty() {
            return Err(TopologyError::EmptyLinkId);
        }
        if self.links.contains_key(&link.id) {
            return Err(TopologyError::DuplicateLink(link.id));
        }
        if !link.capacity.value.is_finite() || link.capacity.value <= 0.0 {
            return Err(TopologyError::InvalidLinkCapacity(link.capacity.value));
        }
        validate_loss(link.loss_fraction)?;
        if link.from_node == link.to_node && link.from_port == link.to_port {
            return Err(TopologyError::SelfLoopPort {
                node: link.from_node,
                port: link.from_port,
            });
        }

        let source = self
            .port(&link.from_node, &link.from_port)
            .ok_or_else(|| unknown_endpoint(&link.from_node, &link.from_port))?;
        let destination = self
            .port(&link.to_node, &link.to_port)
            .ok_or_else(|| unknown_endpoint(&link.to_node, &link.to_port))?;

        if !source.provides(link.capacity) {
            return Err(TopologyError::SourceContractViolation(link.id));
        }
        let delivered = link.delivered_capacity()?;
        if !destination.accepts(delivered) {
            return Err(TopologyError::DestinationContractViolation(link.id));
        }

        self.links.insert(link.id.clone(), link);
        Ok(())
    }

    pub fn port(&self, node: &str, port: &str) -> Option<&ResourcePort> {
        self.nodes.get(node).and_then(|ports| ports.get(port))
    }

    pub fn link(&self, id: &str) -> Option<&ResourceLink> {
        self.links.get(id)
    }

    pub fn links(&self) -> impl Iterator<Item = &ResourceLink> {
        self.links.values()
    }

    pub fn links_from<'a>(
        &'a self,
        node: &'a str,
        port: &'a str,
    ) -> impl Iterator<Item = &'a ResourceLink> + 'a {
        self.links
            .values()
            .filter(move |link| link.from_node == node && link.from_port == port)
    }

    pub fn links_to<'a>(
        &'a self,
        node: &'a str,
        port: &'a str,
    ) -> impl Iterator<Item = &'a ResourceLink> + 'a {
        self.links
            .values()
            .filter(move |link| link.to_node == node && link.to_port == port)
    }
}

fn validate_loss(loss_fraction: f64) -> Result<(), TopologyError> {
    if !loss_fraction.is_finite() || !(0.0..=1.0).contains(&loss_fraction) {
        return Err(TopologyError::InvalidLossFraction(loss_fraction));
    }
    Ok(())
}

fn unknown_endpoint(node: &str, port: &str) -> TopologyError {
    TopologyError::UnknownPort {
        node: node.to_owned(),
        port: port.to_owned(),
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum TopologyError {
    #[error("resource topology node id must not be empty")]
    EmptyNodeId,
    #[error("resource topology port id must not be empty on node {0}")]
    EmptyPortId(String),
    #[error("resource topology link id must not be empty")]
    EmptyLinkId,
    #[error("duplicate topology node {0}")]
    DuplicateNode(String),
    #[error("duplicate topology port on node {0}")]
    DuplicatePort(String),
    #[error("duplicate topology link {0}")]
    DuplicateLink(String),
    #[error("unknown topology port {node}/{port}")]
    UnknownPort { node: String, port: String },
    #[error("topology link cannot loop from a port back to itself: {node}/{port}")]
    SelfLoopPort { node: String, port: String },
    #[error("topology link capacity must be finite and positive, got {0}")]
    InvalidLinkCapacity(f64),
    #[error("invalid resource capacity: {0}")]
    InvalidCapacity(String),
    #[error("loss fraction must be finite and within [0, 1], got {0}")]
    InvalidLossFraction(f64),
    #[error("source port contract rejected topology link {0}")]
    SourceContractViolation(String),
    #[error("destination port contract rejected topology link {0}")]
    DestinationContractViolation(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_resource_model::{
        PortDirection, ResourceKind, ResourceUnit,
    };

    fn electricity(value: f64) -> ResourceAmount {
        ResourceAmount::new(ResourceKind::Electricity, ResourceUnit::Watt, value).unwrap()
    }

    fn compute(value: f64) -> ResourceAmount {
        ResourceAmount::new(ResourceKind::Compute, ResourceUnit::GpuSecond, value).unwrap()
    }

    fn port(id: &str, direction: PortDirection, capacity: ResourceAmount) -> ResourcePort {
        ResourcePort {
            id: id.into(),
            direction,
            capacity,
        }
    }

    fn link(
        id: &str,
        from: (&str, &str),
        to: (&str, &str),
        watts: f64,
    ) -> ResourceLink {
        ResourceLink {
            id: id.into(),
            from_node: from.0.into(),
            from_port: from.1.into(),
            to_node: to.0.into(),
            to_port: to.1.into(),
            capacity: electricity(watts),
            loss_fraction: 0.0,
        }
    }

    #[test]
    fn potential_links_do_not_consume_shared_port_capacity() {
        let mut topology = ResourceTopology::default();
        topology
            .add_node(
                "source",
                [port("out", PortDirection::Output, electricity(100.0))],
            )
            .unwrap();
        topology
            .add_node(
                "a",
                [port("in", PortDirection::Input, electricity(100.0))],
            )
            .unwrap();
        topology
            .add_node(
                "b",
                [port("in", PortDirection::Input, electricity(100.0))],
            )
            .unwrap();

        topology
            .add_link(link("to-a", ("source", "out"), ("a", "in"), 80.0))
            .unwrap();
        topology
            .add_link(link("to-b", ("source", "out"), ("b", "in"), 80.0))
            .unwrap();

        assert_eq!(topology.links_from("source", "out").count(), 2);
        assert_eq!(topology.links().count(), 2);
    }

    #[test]
    fn one_link_cannot_exceed_source_port_contract() {
        let mut topology = ResourceTopology::default();
        topology
            .add_node(
                "source",
                [port("out", PortDirection::Output, electricity(50.0))],
            )
            .unwrap();
        topology
            .add_node(
                "sink",
                [port("in", PortDirection::Input, electricity(100.0))],
            )
            .unwrap();

        assert!(matches!(
            topology.add_link(link("too-large", ("source", "out"), ("sink", "in"), 75.0)),
            Err(TopologyError::SourceContractViolation(id)) if id == "too-large"
        ));
    }

    #[test]
    fn destination_contract_uses_delivered_capacity_after_loss() {
        let mut topology = ResourceTopology::default();
        topology
            .add_node(
                "source",
                [port("out", PortDirection::Output, electricity(100.0))],
            )
            .unwrap();
        topology
            .add_node(
                "sink",
                [port("in", PortDirection::Input, electricity(80.0))],
            )
            .unwrap();

        let mut lossy = link("lossy", ("source", "out"), ("sink", "in"), 100.0);
        lossy.loss_fraction = 0.2;
        topology.add_link(lossy).unwrap();
        assert_eq!(topology.link("lossy").unwrap().delivered_capacity().unwrap().value, 80.0);
    }

    #[test]
    fn incompatible_resource_dimensions_are_rejected() {
        let mut topology = ResourceTopology::default();
        topology
            .add_node(
                "gpu",
                [port("out", PortDirection::Output, compute(10.0))],
            )
            .unwrap();
        topology
            .add_node(
                "sink",
                [port("in", PortDirection::Input, electricity(100.0))],
            )
            .unwrap();

        let bad = ResourceLink {
            id: "wrong-kind".into(),
            from_node: "gpu".into(),
            from_port: "out".into(),
            to_node: "sink".into(),
            to_port: "in".into(),
            capacity: compute(5.0),
            loss_fraction: 0.0,
        };
        assert!(matches!(
            topology.add_link(bad),
            Err(TopologyError::DestinationContractViolation(id)) if id == "wrong-kind"
        ));
    }

    #[test]
    fn duplicate_link_identity_is_rejected() {
        let mut topology = ResourceTopology::default();
        topology
            .add_node(
                "source",
                [port("out", PortDirection::Output, electricity(100.0))],
            )
            .unwrap();
        topology
            .add_node(
                "sink",
                [port("in", PortDirection::Input, electricity(100.0))],
            )
            .unwrap();
        topology
            .add_link(link("same", ("source", "out"), ("sink", "in"), 50.0))
            .unwrap();

        assert!(matches!(
            topology.add_link(link("same", ("source", "out"), ("sink", "in"), 50.0)),
            Err(TopologyError::DuplicateLink(id)) if id == "same"
        ));
    }

    #[test]
    fn zero_capacity_link_is_rejected() {
        let mut topology = ResourceTopology::default();
        topology
            .add_node(
                "source",
                [port("out", PortDirection::Output, electricity(100.0))],
            )
            .unwrap();
        topology
            .add_node(
                "sink",
                [port("in", PortDirection::Input, electricity(100.0))],
            )
            .unwrap();
        assert!(matches!(
            topology.add_link(link("zero", ("source", "out"), ("sink", "in"), 0.0)),
            Err(TopologyError::InvalidLinkCapacity(value)) if value == 0.0
        ));
    }

    #[test]
    fn exact_same_port_self_loop_is_rejected() {
        let mut topology = ResourceTopology::default();
        topology
            .add_node(
                "bus",
                [port(
                    "ac",
                    PortDirection::Bidirectional,
                    electricity(100.0),
                )],
            )
            .unwrap();
        assert!(matches!(
            topology.add_link(link("loop", ("bus", "ac"), ("bus", "ac"), 10.0)),
            Err(TopologyError::SelfLoopPort { node, port }) if node == "bus" && port == "ac"
        ));
    }

    #[test]
    fn invalid_loss_fraction_is_rejected() {
        let mut topology = ResourceTopology::default();
        topology
            .add_node(
                "source",
                [port("out", PortDirection::Output, electricity(100.0))],
            )
            .unwrap();
        topology
            .add_node(
                "sink",
                [port("in", PortDirection::Input, electricity(100.0))],
            )
            .unwrap();
        let mut bad = link("bad-loss", ("source", "out"), ("sink", "in"), 10.0);
        bad.loss_fraction = 1.1;
        assert!(matches!(
            topology.add_link(bad),
            Err(TopologyError::InvalidLossFraction(value)) if value == 1.1
        ));
    }
}
