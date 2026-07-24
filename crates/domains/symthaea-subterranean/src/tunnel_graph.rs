// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded, deterministic tunnel topology and route planning.
//!
//! Depth alone is not a route. A long-duration underground mission needs an
//! explicit graph of junctions, refuges, relay bays, workfaces, and the
//! traversability of the passages between them. This module intentionally uses
//! a small fixed-capacity graph and an O(V^2) shortest-path search: the bounds
//! are inspectable, allocation is bounded, and equal-cost ties are stable.

use serde::{Deserialize, Serialize};

pub const MAX_TUNNEL_NODES: usize = 128;
pub const MAX_TUNNEL_EDGES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TunnelNodeId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TunnelNodeKind {
    Surface,
    Junction,
    Workface,
    RelayBay,
    Refuge,
    ServiceBay,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TunnelNode {
    pub id: TunnelNodeId,
    pub kind: TunnelNodeKind,
    pub depth_m: f64,
    pub survey_confidence: f64,
}

impl TunnelNode {
    pub fn validate(self) -> Result<Self, TunnelGraphError> {
        if !self.depth_m.is_finite() || self.depth_m < 0.0 || self.depth_m > 200.0 {
            return Err(TunnelGraphError::InvalidNode);
        }
        if !self.survey_confidence.is_finite() || !(0.0..=1.0).contains(&self.survey_confidence) {
            return Err(TunnelGraphError::InvalidNode);
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TunnelEdge {
    pub from: TunnelNodeId,
    pub to: TunnelNodeId,
    pub length_m: f64,
    pub energy_per_m: f64,
    pub obstruction_risk: f64,
    pub water_risk: f64,
    pub roof_risk: f64,
    pub confidence: f64,
    pub traversable: bool,
    pub bidirectional: bool,
    pub revision: u64,
}

impl TunnelEdge {
    pub fn validate(self) -> Result<Self, TunnelGraphError> {
        let bounded = [
            self.obstruction_risk,
            self.water_risk,
            self.roof_risk,
            self.confidence,
        ]
        .into_iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(&value));
        if self.from == self.to
            || !self.length_m.is_finite()
            || self.length_m <= 0.0
            || !self.energy_per_m.is_finite()
            || self.energy_per_m < 0.0
            || !bounded
        {
            return Err(TunnelGraphError::InvalidEdge);
        }
        Ok(self)
    }

    pub fn risk(self) -> f64 {
        self.obstruction_risk
            .max(self.water_risk)
            .max(self.roof_risk)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouteCostPolicy {
    pub distance_weight: f64,
    pub energy_weight: f64,
    pub risk_weight: f64,
    pub uncertainty_weight: f64,
}

impl RouteCostPolicy {
    pub const fn conservative() -> Self {
        Self {
            distance_weight: 1.0,
            energy_weight: 40.0,
            risk_weight: 120.0,
            uncertainty_weight: 80.0,
        }
    }

    pub fn validate(self) -> Result<Self, TunnelGraphError> {
        let valid = [
            self.distance_weight,
            self.energy_weight,
            self.risk_weight,
            self.uncertainty_weight,
        ]
        .into_iter()
        .all(|value| value.is_finite() && value >= 0.0);
        if !valid {
            return Err(TunnelGraphError::InvalidRoutePolicy);
        }
        Ok(self)
    }

    fn edge_cost(self, edge: TunnelEdge) -> f64 {
        edge.length_m * self.distance_weight
            + edge.length_m * edge.energy_per_m * self.energy_weight
            + edge.risk() * self.risk_weight
            + (1.0 - edge.confidence) * self.uncertainty_weight
    }
}

impl Default for RouteCostPolicy {
    fn default() -> Self {
        Self::conservative()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TunnelRoute {
    pub nodes: Vec<TunnelNodeId>,
    pub distance_m: f64,
    pub estimated_energy: f64,
    pub maximum_risk: f64,
    pub minimum_confidence: f64,
    pub total_cost: f64,
}

impl TunnelRoute {
    pub fn feasible(&self) -> bool {
        self.nodes.len() >= 2
            && self.distance_m.is_finite()
            && self.estimated_energy.is_finite()
            && self.total_cost.is_finite()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TunnelGraphError {
    InvalidNode,
    InvalidEdge,
    DuplicateNode,
    MissingEndpoint,
    NodeCapacity,
    EdgeCapacity,
    StaleEdgeRevision,
    RouteNotFound,
    InvalidRoutePolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundedTunnelGraph {
    nodes: Vec<TunnelNode>,
    edges: Vec<TunnelEdge>,
}

impl BoundedTunnelGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(MAX_TUNNEL_NODES),
            edges: Vec::with_capacity(MAX_TUNNEL_EDGES),
        }
    }

    pub fn validate(&self) -> Result<(), TunnelGraphError> {
        if self.nodes.len() > MAX_TUNNEL_NODES {
            return Err(TunnelGraphError::NodeCapacity);
        }
        if self.edges.len() > MAX_TUNNEL_EDGES {
            return Err(TunnelGraphError::EdgeCapacity);
        }
        for (index, node) in self.nodes.iter().copied().enumerate() {
            node.validate()?;
            if self.nodes[..index]
                .iter()
                .any(|candidate| candidate.id == node.id)
            {
                return Err(TunnelGraphError::DuplicateNode);
            }
        }
        for edge in self.edges.iter().copied() {
            edge.validate()?;
            if self.node(edge.from).is_none() || self.node(edge.to).is_none() {
                return Err(TunnelGraphError::MissingEndpoint);
            }
        }
        Ok(())
    }

    pub fn nodes(&self) -> &[TunnelNode] {
        &self.nodes
    }

    pub fn edges(&self) -> &[TunnelEdge] {
        &self.edges
    }

    pub fn node(&self, id: TunnelNodeId) -> Option<TunnelNode> {
        self.nodes.iter().copied().find(|node| node.id == id)
    }

    pub fn add_node(&mut self, node: TunnelNode) -> Result<(), TunnelGraphError> {
        let node = node.validate()?;
        if self.node(node.id).is_some() {
            return Err(TunnelGraphError::DuplicateNode);
        }
        if self.nodes.len() >= MAX_TUNNEL_NODES {
            return Err(TunnelGraphError::NodeCapacity);
        }
        self.nodes.push(node);
        self.nodes.sort_by_key(|candidate| candidate.id.0);
        Ok(())
    }

    pub fn upsert_edge(&mut self, edge: TunnelEdge) -> Result<(), TunnelGraphError> {
        let edge = edge.validate()?;
        if self.node(edge.from).is_none() || self.node(edge.to).is_none() {
            return Err(TunnelGraphError::MissingEndpoint);
        }
        if let Some(existing) = self
            .edges
            .iter_mut()
            .find(|candidate| candidate.from == edge.from && candidate.to == edge.to)
        {
            if edge.revision <= existing.revision {
                return Err(TunnelGraphError::StaleEdgeRevision);
            }
            *existing = edge;
            return Ok(());
        }
        if self.edges.len() >= MAX_TUNNEL_EDGES {
            return Err(TunnelGraphError::EdgeCapacity);
        }
        self.edges.push(edge);
        self.edges
            .sort_by_key(|candidate| (candidate.from.0, candidate.to.0));
        Ok(())
    }

    fn node_index(&self, id: TunnelNodeId) -> Option<usize> {
        self.nodes.iter().position(|node| node.id == id)
    }

    fn connecting_edge(&self, from: TunnelNodeId, to: TunnelNodeId) -> Option<TunnelEdge> {
        self.edges.iter().copied().find(|edge| {
            edge.traversable
                && ((edge.from == from && edge.to == to)
                    || (edge.bidirectional && edge.from == to && edge.to == from))
        })
    }

    pub fn route(
        &self,
        start: TunnelNodeId,
        goal: TunnelNodeId,
        policy: RouteCostPolicy,
    ) -> Result<TunnelRoute, TunnelGraphError> {
        let start_index = self
            .node_index(start)
            .ok_or(TunnelGraphError::MissingEndpoint)?;
        let goal_index = self
            .node_index(goal)
            .ok_or(TunnelGraphError::MissingEndpoint)?;
        if start == goal {
            return Ok(TunnelRoute {
                nodes: vec![start],
                distance_m: 0.0,
                estimated_energy: 0.0,
                maximum_risk: 0.0,
                minimum_confidence: 1.0,
                total_cost: 0.0,
            });
        }

        let mut distance = [f64::INFINITY; MAX_TUNNEL_NODES];
        let mut previous = [None; MAX_TUNNEL_NODES];
        let mut visited = [false; MAX_TUNNEL_NODES];
        distance[start_index] = 0.0;

        for _ in 0..self.nodes.len() {
            let mut current = None;
            let mut current_cost = f64::INFINITY;
            for index in 0..self.nodes.len() {
                let candidate = distance[index];
                if !visited[index]
                    && (candidate < current_cost
                        || (candidate == current_cost && current.is_none_or(|value| index < value)))
                {
                    current = Some(index);
                    current_cost = candidate;
                }
            }
            let Some(current_index) = current else {
                break;
            };
            if !current_cost.is_finite() {
                break;
            }
            if current_index == goal_index {
                break;
            }
            visited[current_index] = true;
            let current_id = self.nodes[current_index].id;
            for neighbor_index in 0..self.nodes.len() {
                if visited[neighbor_index] || neighbor_index == current_index {
                    continue;
                }
                let neighbor_id = self.nodes[neighbor_index].id;
                let Some(edge) = self.connecting_edge(current_id, neighbor_id) else {
                    continue;
                };
                let candidate_cost = current_cost + policy.edge_cost(edge);
                if candidate_cost < distance[neighbor_index] {
                    distance[neighbor_index] = candidate_cost;
                    previous[neighbor_index] = Some(current_index);
                }
            }
        }

        if !distance[goal_index].is_finite() {
            return Err(TunnelGraphError::RouteNotFound);
        }
        let mut reversed = Vec::with_capacity(self.nodes.len());
        let mut cursor = goal_index;
        reversed.push(self.nodes[cursor].id);
        while cursor != start_index {
            let Some(parent) = previous[cursor] else {
                return Err(TunnelGraphError::RouteNotFound);
            };
            cursor = parent;
            reversed.push(self.nodes[cursor].id);
        }
        reversed.reverse();

        let mut route = TunnelRoute {
            nodes: reversed,
            distance_m: 0.0,
            estimated_energy: 0.0,
            maximum_risk: 0.0,
            minimum_confidence: 1.0,
            total_cost: distance[goal_index],
        };
        for pair in route.nodes.windows(2) {
            let Some(edge) = self.connecting_edge(pair[0], pair[1]) else {
                return Err(TunnelGraphError::RouteNotFound);
            };
            route.distance_m += edge.length_m;
            route.estimated_energy += edge.length_m * edge.energy_per_m;
            route.maximum_risk = route.maximum_risk.max(edge.risk());
            route.minimum_confidence = route.minimum_confidence.min(edge.confidence);
        }
        Ok(route)
    }
}

impl Default for BoundedTunnelGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: u32, kind: TunnelNodeKind) -> TunnelNode {
        TunnelNode {
            id: TunnelNodeId(id),
            kind,
            depth_m: id as f64 * 10.0,
            survey_confidence: 0.9,
        }
    }

    fn edge(from: u32, to: u32, length: f64, risk: f64) -> TunnelEdge {
        TunnelEdge {
            from: TunnelNodeId(from),
            to: TunnelNodeId(to),
            length_m: length,
            energy_per_m: 0.001,
            obstruction_risk: risk,
            water_risk: 0.0,
            roof_risk: 0.0,
            confidence: 0.9,
            traversable: true,
            bidirectional: true,
            revision: 1,
        }
    }

    #[test]
    fn conservative_routing_prefers_longer_safe_passage() {
        let mut graph = BoundedTunnelGraph::new();
        for candidate in [
            node(0, TunnelNodeKind::Surface),
            node(1, TunnelNodeKind::Junction),
            node(2, TunnelNodeKind::Junction),
            node(3, TunnelNodeKind::Workface),
        ] {
            graph.add_node(candidate).expect("bounded fixture");
        }
        graph.upsert_edge(edge(0, 1, 10.0, 0.9)).expect("edge");
        graph.upsert_edge(edge(1, 3, 10.0, 0.9)).expect("edge");
        graph.upsert_edge(edge(0, 2, 18.0, 0.05)).expect("edge");
        graph.upsert_edge(edge(2, 3, 18.0, 0.05)).expect("edge");
        let route = graph
            .route(TunnelNodeId(0), TunnelNodeId(3), RouteCostPolicy::default())
            .expect("safe route");
        assert_eq!(
            route.nodes,
            vec![TunnelNodeId(0), TunnelNodeId(2), TunnelNodeId(3)]
        );
        assert!(route.maximum_risk < 0.1);
    }

    #[test]
    fn blocked_edge_is_never_routed() {
        let mut graph = BoundedTunnelGraph::new();
        graph
            .add_node(node(0, TunnelNodeKind::Surface))
            .expect("node");
        graph
            .add_node(node(1, TunnelNodeKind::Workface))
            .expect("node");
        let mut blocked = edge(0, 1, 2.0, 0.0);
        blocked.traversable = false;
        graph.upsert_edge(blocked).expect("edge");
        assert_eq!(
            graph.route(TunnelNodeId(0), TunnelNodeId(1), RouteCostPolicy::default()),
            Err(TunnelGraphError::RouteNotFound)
        );
    }

    #[test]
    fn stale_edge_revision_cannot_reopen_a_passage() {
        let mut graph = BoundedTunnelGraph::new();
        graph
            .add_node(node(0, TunnelNodeKind::Surface))
            .expect("node");
        graph
            .add_node(node(1, TunnelNodeKind::Workface))
            .expect("node");
        let mut current = edge(0, 1, 2.0, 0.0);
        current.revision = 2;
        current.traversable = false;
        graph.upsert_edge(current).expect("edge");
        let stale = edge(0, 1, 2.0, 0.0);
        assert_eq!(
            graph.upsert_edge(stale),
            Err(TunnelGraphError::StaleEdgeRevision)
        );
    }
}
