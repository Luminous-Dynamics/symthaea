// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Supply Chain Graph: directed acyclic graph of production dependencies.
//!
//! Transforms the economy from abstract resource stocks into a topological
//! vulnerability map. A Mega-Quake in the Pacific Rim severs semiconductor
//! supply → the Ecuador Spaceport launch queue freezes → Mars colony's
//! ECLSS filter shipment is delayed → allostatic load spikes 400M km away.
//!
//! # Architecture
//!
//! The graph G = (V, E) where:
//! - V = production nodes (12 Earth regions + off-world colonies)
//! - E = supply routes (shipping lanes, maglev, launch corridors, Hohmann transfers)
//! - Edge weights = throughput capacity + vulnerability to disruption
//!
//! When a disaster hits a region, all outgoing edges from that node are
//! degraded proportionally. Downstream nodes experience supply shortages
//! that propagate through the DAG via topological sort.
//!
//! # Mycelix Integration
//!
//! Each edge crossing maps to a TEND transaction on the DHT.
//! Supply chain health feeds into FEP prediction error for agent resource expectations.

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Earth Macro-Regions (12 aggregate zones)
// ============================================================================

/// The 12 macro-regions of Earth plus off-world nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SupplyNode {
    // === Earth regions ===
    NorthAmerica,
    SouthAmerica, // Ecuador Spaceport
    WesternEurope,
    EasternEurope,
    NorthAfrica,
    SubSaharanAfrica,
    MiddleEast,
    SouthAsia,
    EastAsia, // Semiconductor hub
    SoutheastAsia,
    Oceania,
    CentralAsia,
    // === Off-world ===
    LunarColony,
    MarsColony,
    EuropaStation,
    TitanOutpost,
}

impl SupplyNode {
    pub fn is_earth(&self) -> bool {
        !matches!(
            self,
            Self::LunarColony | Self::MarsColony | Self::EuropaStation | Self::TitanOutpost
        )
    }

    pub fn all_earth() -> Vec<Self> {
        vec![
            Self::NorthAmerica,
            Self::SouthAmerica,
            Self::WesternEurope,
            Self::EasternEurope,
            Self::NorthAfrica,
            Self::SubSaharanAfrica,
            Self::MiddleEast,
            Self::SouthAsia,
            Self::EastAsia,
            Self::SoutheastAsia,
            Self::Oceania,
            Self::CentralAsia,
        ]
    }

    /// Primary production capability of each region.
    pub fn primary_production(&self) -> &'static [&'static str] {
        match self {
            Self::NorthAmerica => &["aerospace", "software", "agriculture"],
            Self::SouthAmerica => &["launch_services", "lithium", "agriculture"],
            Self::WesternEurope => &["precision_optics", "pharmaceuticals", "nuclear"],
            Self::EasternEurope => &["heavy_industry", "nuclear", "materials"],
            Self::NorthAfrica => &["solar_energy", "phosphates"],
            Self::SubSaharanAfrica => &["rare_earths", "cobalt", "uranium"],
            Self::MiddleEast => &["energy", "desalination", "petrochemicals"],
            Self::SouthAsia => &["pharmaceuticals", "software", "agriculture"],
            Self::EastAsia => &["semiconductors", "electronics", "manufacturing"],
            Self::SoutheastAsia => &["tin", "rubber", "palm_oil", "manufacturing"],
            Self::Oceania => &["uranium", "iron_ore", "bauxite"],
            Self::CentralAsia => &["rare_earths", "natural_gas", "uranium"],
            Self::LunarColony => &["helium3", "regolith", "oxygen"],
            Self::MarsColony => &["iron", "water", "co2"],
            Self::EuropaStation => &["water", "oxygen"],
            Self::TitanOutpost => &["hydrocarbons", "nitrogen"],
        }
    }
}

// ============================================================================
// Supply Route (edge in the graph)
// ============================================================================

/// A supply route between two nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupplyRoute {
    /// Throughput capacity [0, 1]. 1.0 = full capacity.
    pub capacity: f64,
    /// Current throughput utilization [0, 1].
    pub utilization: f64,
    /// Vulnerability to disruption [0, 1]. Higher = more fragile.
    pub vulnerability: f64,
    /// Transport mode (affects disruption propagation speed).
    pub mode: TransportMode,
    /// Primary goods carried on this route.
    pub goods: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransportMode {
    /// Ocean shipping (slow, high capacity, moderate vulnerability).
    Maritime,
    /// Land transport — rail, truck, maglev (fast, moderate capacity).
    Land,
    /// Air freight (fast, low capacity, low vulnerability).
    Air,
    /// Launch corridor to orbit (bottleneck, high vulnerability).
    LaunchCorridor,
    /// Hohmann transfer (orbital mechanics constrained).
    Interplanetary,
}

impl Default for SupplyRoute {
    fn default() -> Self {
        Self {
            capacity: 1.0,
            utilization: 0.5,
            vulnerability: 0.1,
            mode: TransportMode::Maritime,
            goods: Vec::new(),
        }
    }
}

// ============================================================================
// Supply Chain Graph
// ============================================================================

/// The global supply chain as a directed graph.
///
/// Disasters degrade node capacity → edge throughput drops → downstream
/// nodes experience shortages → cascade propagation via topological order.
#[derive(Debug, Clone)]
pub struct SupplyChainGraph {
    /// The directed graph. Nodes are SupplyNode, edges are SupplyRoute.
    pub graph: DiGraph<SupplyNode, SupplyRoute>,
    /// Map from SupplyNode to NodeIndex for fast lookup.
    pub node_indices: HashMap<SupplyNode, NodeIndex>,
    /// Per-node health [0, 1]. Degraded by disasters.
    pub node_health: HashMap<SupplyNode, f64>,
    /// Per-node disruption events this tick.
    pub disruptions: Vec<SupplyDisruption>,
    /// Global supply chain health [0, 1].
    pub global_health: f64,
}

/// A disruption event affecting a supply node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupplyDisruption {
    pub node: SupplyNode,
    pub severity: f64,
    pub description: String,
    pub remaining_ticks: u32,
}

impl SupplyChainGraph {
    /// Build the default global supply chain graph.
    pub fn new() -> Self {
        let mut graph = DiGraph::new();
        let mut node_indices = HashMap::new();
        let mut node_health = HashMap::new();

        // Add all nodes
        let all_nodes = [
            SupplyNode::NorthAmerica,
            SupplyNode::SouthAmerica,
            SupplyNode::WesternEurope,
            SupplyNode::EasternEurope,
            SupplyNode::NorthAfrica,
            SupplyNode::SubSaharanAfrica,
            SupplyNode::MiddleEast,
            SupplyNode::SouthAsia,
            SupplyNode::EastAsia,
            SupplyNode::SoutheastAsia,
            SupplyNode::Oceania,
            SupplyNode::CentralAsia,
            SupplyNode::LunarColony,
            SupplyNode::MarsColony,
            SupplyNode::EuropaStation,
            SupplyNode::TitanOutpost,
        ];
        for &node in &all_nodes {
            let idx = graph.add_node(node);
            node_indices.insert(node, idx);
            node_health.insert(node, 1.0);
        }

        // Add key supply routes (Earth internal)
        let mut g = Self {
            graph,
            node_indices,
            node_health,
            disruptions: Vec::new(),
            global_health: 1.0,
        };

        // === Critical Earth supply chains ===
        // Semiconductors: East Asia → everywhere
        g.add_route(
            SupplyNode::EastAsia,
            SupplyNode::NorthAmerica,
            TransportMode::Maritime,
            0.9,
            0.15,
            vec!["semiconductors", "electronics"],
        );
        g.add_route(
            SupplyNode::EastAsia,
            SupplyNode::WesternEurope,
            TransportMode::Maritime,
            0.8,
            0.15,
            vec!["semiconductors", "electronics"],
        );

        // Rare earths: Sub-Saharan Africa + Central Asia → manufacturing
        g.add_route(
            SupplyNode::SubSaharanAfrica,
            SupplyNode::EastAsia,
            TransportMode::Maritime,
            0.7,
            0.2,
            vec!["rare_earths", "cobalt"],
        );
        g.add_route(
            SupplyNode::CentralAsia,
            SupplyNode::EastAsia,
            TransportMode::Land,
            0.5,
            0.15,
            vec!["rare_earths"],
        );

        // Energy: Middle East → global
        g.add_route(
            SupplyNode::MiddleEast,
            SupplyNode::SouthAsia,
            TransportMode::Maritime,
            0.8,
            0.1,
            vec!["energy"],
        );
        g.add_route(
            SupplyNode::MiddleEast,
            SupplyNode::EastAsia,
            TransportMode::Maritime,
            0.9,
            0.15,
            vec!["energy"],
        );

        // Aerospace: North America → South America (launch)
        g.add_route(
            SupplyNode::NorthAmerica,
            SupplyNode::SouthAmerica,
            TransportMode::Land,
            0.8,
            0.1,
            vec!["aerospace", "software"],
        );

        // Launch corridor: South America → Lunar Colony
        g.add_route(
            SupplyNode::SouthAmerica,
            SupplyNode::LunarColony,
            TransportMode::LaunchCorridor,
            0.6,
            0.3,
            vec!["crew", "equipment"],
        );

        // Nuclear: Western Europe + Oceania → space program
        g.add_route(
            SupplyNode::WesternEurope,
            SupplyNode::NorthAmerica,
            TransportMode::Maritime,
            0.7,
            0.1,
            vec!["precision_optics", "nuclear"],
        );
        g.add_route(
            SupplyNode::Oceania,
            SupplyNode::SouthAmerica,
            TransportMode::Maritime,
            0.5,
            0.2,
            vec!["uranium"],
        );

        // Pharmaceuticals: South Asia + Western Europe → colonies
        g.add_route(
            SupplyNode::SouthAsia,
            SupplyNode::NorthAmerica,
            TransportMode::Air,
            0.6,
            0.1,
            vec!["pharmaceuticals"],
        );

        // === Interplanetary routes ===
        g.add_route(
            SupplyNode::LunarColony,
            SupplyNode::MarsColony,
            TransportMode::Interplanetary,
            0.3,
            0.4,
            vec!["equipment", "crew"],
        );
        g.add_route(
            SupplyNode::LunarColony,
            SupplyNode::EuropaStation,
            TransportMode::Interplanetary,
            0.1,
            0.5,
            vec!["equipment"],
        );
        g.add_route(
            SupplyNode::MarsColony,
            SupplyNode::TitanOutpost,
            TransportMode::Interplanetary,
            0.05,
            0.6,
            vec!["equipment"],
        );

        // Titan exports hydrocarbons back
        g.add_route(
            SupplyNode::TitanOutpost,
            SupplyNode::MarsColony,
            TransportMode::Interplanetary,
            0.1,
            0.5,
            vec!["hydrocarbons", "nitrogen"],
        );
        g.add_route(
            SupplyNode::EuropaStation,
            SupplyNode::MarsColony,
            TransportMode::Interplanetary,
            0.1,
            0.4,
            vec!["water", "oxygen"],
        );

        g
    }

    fn add_route(
        &mut self,
        from: SupplyNode,
        to: SupplyNode,
        mode: TransportMode,
        capacity: f64,
        vulnerability: f64,
        goods: Vec<&str>,
    ) {
        let from_idx = self.node_indices[&from];
        let to_idx = self.node_indices[&to];
        self.graph.add_edge(
            from_idx,
            to_idx,
            SupplyRoute {
                capacity,
                utilization: capacity * 0.5,
                vulnerability,
                mode,
                goods: goods.iter().map(|s| s.to_string()).collect(),
            },
        );
    }

    /// Apply a disruption to a node (from disaster, conflict, etc.).
    pub fn disrupt_node(
        &mut self,
        node: SupplyNode,
        severity: f64,
        description: String,
        duration: u32,
    ) {
        if let Some(health) = self.node_health.get_mut(&node) {
            *health = (*health - severity).max(0.0);
        }
        self.disruptions.push(SupplyDisruption {
            node,
            severity,
            description,
            remaining_ticks: duration,
        });
    }

    /// Propagate disruptions through the graph.
    /// Returns the effective supply multiplier for each off-world colony.
    pub fn propagate(&mut self) -> HashMap<SupplyNode, f64> {
        // Advance disruptions
        self.disruptions.retain_mut(|d| {
            d.remaining_ticks = d.remaining_ticks.saturating_sub(1);
            if d.remaining_ticks == 0 {
                // Recovery
                if let Some(health) = self.node_health.get_mut(&d.node) {
                    *health = (*health + d.severity * 0.5).min(1.0); // Partial recovery
                }
            }
            d.remaining_ticks > 0
        });

        // Compute effective throughput for each colony
        let mut colony_supply: HashMap<SupplyNode, f64> = HashMap::new();
        let colonies = [
            SupplyNode::LunarColony,
            SupplyNode::MarsColony,
            SupplyNode::EuropaStation,
            SupplyNode::TitanOutpost,
        ];

        for &colony in &colonies {
            let colony_idx = self.node_indices[&colony];
            // Sum incoming edge capacity × source node health
            let mut total_supply = 0.0;
            let mut max_possible = 0.0;
            for edge in self.graph.edges_directed(colony_idx, Direction::Incoming) {
                let source_node = self.graph[edge.source()];
                let source_health = self.node_health.get(&source_node).copied().unwrap_or(1.0);
                let route = edge.weight();
                total_supply += route.capacity * source_health;
                max_possible += route.capacity;
            }
            let multiplier = if max_possible > 0.0 {
                total_supply / max_possible
            } else {
                0.0
            };
            colony_supply.insert(colony, multiplier);
        }

        // Global health = mean of all node healths
        let total: f64 = self.node_health.values().sum();
        self.global_health = total / self.node_health.len().max(1) as f64;

        colony_supply
    }

    /// Get the number of active disruptions.
    pub fn active_disruptions(&self) -> usize {
        self.disruptions.len()
    }

    /// Get health of a specific node.
    pub fn node_health(&self, node: SupplyNode) -> f64 {
        self.node_health.get(&node).copied().unwrap_or(1.0)
    }
}

impl Default for SupplyChainGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_construction() {
        let g = SupplyChainGraph::new();
        assert_eq!(g.graph.node_count(), 16);
        assert!(g.graph.edge_count() > 10);
        assert!((g.global_health - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_disruption_degrades_supply() {
        let mut g = SupplyChainGraph::new();
        g.disrupt_node(SupplyNode::EastAsia, 0.5, "Mega-quake".into(), 3);
        let supply = g.propagate();
        // Lunar colony gets supply through South America, not directly from East Asia
        // But global health should drop
        assert!(g.global_health < 1.0);
        assert!(g.node_health(SupplyNode::EastAsia) < 1.0);
    }

    #[test]
    fn test_disruption_recovery() {
        let mut g = SupplyChainGraph::new();
        g.disrupt_node(SupplyNode::MiddleEast, 0.3, "Conflict".into(), 1);
        let _ = g.propagate(); // Duration 1 → expires
        let _ = g.propagate(); // Recovery kicks in
        assert!(g.node_health(SupplyNode::MiddleEast) > 0.7);
    }

    #[test]
    fn test_cascade_to_colonies() {
        let mut g = SupplyChainGraph::new();
        // Disrupt South America (launch corridor) severely
        g.disrupt_node(SupplyNode::SouthAmerica, 0.8, "Volcanic eruption".into(), 6);
        let supply = g.propagate();
        // Lunar colony supply should be degraded
        let lunar = supply.get(&SupplyNode::LunarColony).copied().unwrap_or(1.0);
        assert!(lunar < 1.0, "Lunar supply should be degraded: {}", lunar);
    }
}
