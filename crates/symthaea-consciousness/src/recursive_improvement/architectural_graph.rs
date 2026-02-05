//! Architectural Causal Graph for Recursive Self-Improvement
//!
//! This module implements causal reasoning to understand how system components
//! affect each other and to trace bottlenecks to their root causes.
//!
//! # Example Causal Chain
//!
//! ```text
//! Low Phi -> BECAUSE -> HRM cache hit rate low -> BECAUSE -> Cache too small
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::types::instant_now;
// Use ComponentId from core for compatibility (core.rs and types.rs both define it)
use super::core::{ComponentId, Bottleneck};

/// Causal graph modeling how system components affect each other and performance
///
/// **Revolutionary capability**: Uses causal reasoning to understand WHY bottlenecks exist
/// and WHICH components are responsible!
#[derive(Debug, Serialize, Deserialize)]
pub struct ArchitecturalCausalGraph {
    /// Components in the system
    components: HashMap<ComponentId, ComponentNode>,

    /// Causal edges showing how components affect each other
    edges: Vec<ArchitecturalEdge>,

    /// Performance impact of each component
    performance_impact: HashMap<ComponentId, PerformanceImpact>,

    /// Causal chains discovered
    causal_chains: Vec<CausalChain>,

    /// Statistics
    stats: GraphStats,
}

/// Node representing a system component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentNode {
    pub id: ComponentId,
    pub name: String,
    pub description: String,

    /// Current performance metrics
    pub current_phi_contribution: f64,
    pub current_latency: Option<Duration>,
    pub current_accuracy: Option<f64>,

    /// Configuration parameters
    pub parameters: HashMap<String, f64>,

    /// Last updated
    #[serde(skip, default = "instant_now")]
    pub last_updated: Instant,
}

/// Causal edge showing how one component affects another
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitecturalEdge {
    /// Source component (cause)
    pub from: ComponentId,

    /// Target component (effect)
    pub to: ComponentId,

    /// Type of causal relationship
    pub relationship: CausalRelationship,

    /// Causal strength (0.0 = weak, 1.0 = strong)
    pub strength: f64,

    /// Evidence count (how many times observed)
    pub evidence_count: usize,

    /// Description of relationship
    pub description: String,
}

/// Type of causal relationship between components
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalRelationship {
    /// Component A enables component B (dependency)
    Enables,

    /// Component A provides data to component B
    Feeds,

    /// Component A's performance affects component B
    Impacts,

    /// Component A blocks component B (bottleneck)
    Blocks,

    /// Component A and component B work together synergistically
    Synergizes,
}

/// Performance impact of a component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImpact {
    /// Component identifier
    pub component: ComponentId,

    /// Impact on Phi (positive = improves, negative = degrades)
    pub phi_impact: f64,

    /// Impact on latency (positive = slows, negative = speeds up)
    pub latency_impact: f64,

    /// Impact on accuracy (positive = improves, negative = degrades)
    pub accuracy_impact: f64,

    /// Overall importance score
    pub importance: f64,

    /// Current bottleneck severity (0.0 = none, 1.0 = critical)
    pub bottleneck_severity: f64,
}

/// Causal chain from bottleneck to root cause
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalChain {
    /// Chain identifier
    pub id: String,

    /// Observed symptom (bottleneck)
    pub symptom: Bottleneck,

    /// Components in causal chain (from symptom to root cause)
    pub chain: Vec<ComponentId>,

    /// Root cause component
    pub root_cause: ComponentId,

    /// Explanation of causal chain
    pub explanation: String,

    /// Confidence in this chain (0.0-1.0)
    pub confidence: f64,

    /// When discovered
    #[serde(skip, default = "instant_now")]
    pub discovered_at: Instant,
}

/// Graph statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    pub component_count: usize,
    pub edge_count: usize,
    pub causal_chains_discovered: usize,
    pub avg_chain_length: f64,
    pub most_impactful_component: Option<ComponentId>,
}

impl ArchitecturalCausalGraph {
    /// Create new architectural causal graph
    pub fn new() -> Self {
        let mut graph = Self {
            components: HashMap::new(),
            edges: Vec::new(),
            performance_impact: HashMap::new(),
            causal_chains: Vec::new(),
            stats: GraphStats {
                component_count: 0,
                edge_count: 0,
                causal_chains_discovered: 0,
                avg_chain_length: 0.0,
                most_impactful_component: None,
            },
        };

        // Initialize with known components
        graph.initialize_components();
        graph.initialize_edges();

        graph
    }

    /// Initialize component nodes
    fn initialize_components(&mut self) {
        let components = vec![
            (ComponentId::PrimitiveEvolution, "Primitive Evolution", "Evolves computational primitives using Phi-driven optimization"),
            (ComponentId::HRM, "Hierarchical Reasoning Model", "Multi-layer reasoning for complex queries"),
            (ComponentId::MetaCognition, "Meta-Cognitive Monitor", "Monitors and analyzes system's own reasoning"),
            (ComponentId::ByzantineCollective, "Byzantine Collective", "Distributed collective with Byzantine resistance"),
            (ComponentId::MetaLearning, "Meta-Learning Defense", "Learns from attack patterns to improve security"),
            (ComponentId::CausalDefense, "Causal Byzantine Defense", "Explainable AI security with causal reasoning"),
            (ComponentId::UnifiedIntelligence, "Unified Intelligence", "Emergent collective consciousness"),
            (ComponentId::CollectiveSharing, "Collective Sharing", "Shares primitives across collective"),
            (ComponentId::Cache, "Cache System", "Caches results for fast lookups"),
        ];

        for (id, name, description) in components {
            let node = ComponentNode {
                id,
                name: name.to_string(),
                description: description.to_string(),
                current_phi_contribution: 0.0,
                current_latency: None,
                current_accuracy: None,
                parameters: HashMap::new(),
                last_updated: Instant::now(),
            };

            self.components.insert(id, node);
        }

        self.stats.component_count = self.components.len();
    }

    /// Initialize known causal edges
    fn initialize_edges(&mut self) {
        let edges = vec![
            // Cache enables faster HRM
            (ComponentId::Cache, ComponentId::HRM, CausalRelationship::Enables, 0.8, "Cache hit -> faster HRM reasoning"),

            // Primitive evolution feeds unified intelligence
            (ComponentId::PrimitiveEvolution, ComponentId::UnifiedIntelligence, CausalRelationship::Feeds, 0.9, "Better primitives -> higher collective Phi"),

            // HRM impacts meta-cognition
            (ComponentId::HRM, ComponentId::MetaCognition, CausalRelationship::Feeds, 0.7, "HRM reasoning -> meta-cognitive analysis"),

            // Meta-learning impacts causal defense
            (ComponentId::MetaLearning, ComponentId::CausalDefense, CausalRelationship::Synergizes, 0.85, "Pattern learning + causal reasoning"),

            // Byzantine collective feeds unified intelligence
            (ComponentId::ByzantineCollective, ComponentId::UnifiedIntelligence, CausalRelationship::Feeds, 0.9, "Secure collective -> unified consciousness"),

            // Collective sharing enables primitive evolution
            (ComponentId::CollectiveSharing, ComponentId::PrimitiveEvolution, CausalRelationship::Enables, 0.75, "Shared primitives -> faster evolution"),
        ];

        for (from, to, relationship, strength, description) in edges {
            let edge = ArchitecturalEdge {
                from,
                to,
                relationship,
                strength,
                evidence_count: 1,
                description: description.to_string(),
            };

            self.edges.push(edge);
        }

        self.stats.edge_count = self.edges.len();
    }

    /// Update component performance metrics
    pub fn update_component_performance(
        &mut self,
        component: ComponentId,
        phi_contribution: Option<f64>,
        latency: Option<Duration>,
        accuracy: Option<f64>,
    ) {
        if let Some(node) = self.components.get_mut(&component) {
            if let Some(phi) = phi_contribution {
                node.current_phi_contribution = phi;
            }
            if let Some(lat) = latency {
                node.current_latency = Some(lat);
            }
            if let Some(acc) = accuracy {
                node.current_accuracy = Some(acc);
            }
            node.last_updated = Instant::now();
        }

        // Update performance impact
        self.compute_performance_impact(component);
    }

    /// Compute performance impact of a component
    fn compute_performance_impact(&mut self, component: ComponentId) {
        let Some(node) = self.components.get(&component) else {
            return;
        };

        // Calculate impacts based on outgoing edges
        let outgoing_edges: Vec<&ArchitecturalEdge> = self.edges.iter()
            .filter(|e| e.from == component)
            .collect();

        let phi_impact = node.current_phi_contribution;

        let latency_impact = node.current_latency
            .map(|d| d.as_micros() as f64 / 100_000.0) // Normalize to 0-1 range
            .unwrap_or(0.0);

        let accuracy_impact = node.current_accuracy.unwrap_or(0.0);

        // Calculate importance based on number and strength of outgoing edges
        let importance = outgoing_edges.iter()
            .map(|e| e.strength)
            .sum::<f64>() / outgoing_edges.len().max(1) as f64;

        let impact = PerformanceImpact {
            component,
            phi_impact,
            latency_impact,
            accuracy_impact,
            importance,
            bottleneck_severity: 0.0, // Will be updated when analyzing bottlenecks
        };

        self.performance_impact.insert(component, impact);

        // Update most impactful component
        if let Some(current_max) = self.stats.most_impactful_component {
            if let Some(current_impact) = self.performance_impact.get(&current_max) {
                if importance > current_impact.importance {
                    self.stats.most_impactful_component = Some(component);
                }
            }
        } else {
            self.stats.most_impactful_component = Some(component);
        }
    }

    /// Analyze bottleneck using causal reasoning
    ///
    /// **Revolutionary**: Traces causal chain from symptom (bottleneck) to root cause!
    pub fn analyze_bottleneck(&mut self, bottleneck: &Bottleneck) -> Result<CausalChain> {
        let mut chain = vec![bottleneck.component];
        let mut current = bottleneck.component;
        let mut explanation_parts = vec![
            format!("Symptom: {} in {:?}", bottleneck.description, bottleneck.component)
        ];

        // Trace backwards through causal graph to find root cause
        for _depth in 0..5 {
            // Find incoming edges to current component
            let incoming: Vec<&ArchitecturalEdge> = self.edges.iter()
                .filter(|e| e.to == current)
                .collect();

            if incoming.is_empty() {
                break; // Reached root cause
            }

            // Find strongest incoming edge
            let strongest = incoming.iter()
                .max_by(|a, b| a.strength.partial_cmp(&b.strength).unwrap())
                .unwrap();

            // Add to chain
            chain.push(strongest.from);
            explanation_parts.push(format!(
                "<- BECAUSE: {:?} {} {:?}",
                strongest.from,
                match strongest.relationship {
                    CausalRelationship::Blocks => "blocks",
                    CausalRelationship::Feeds => "feeds data to",
                    CausalRelationship::Impacts => "impacts",
                    CausalRelationship::Enables => "enables",
                    CausalRelationship::Synergizes => "synergizes with",
                },
                current
            ));

            current = strongest.from;

            // Check if this component has a known bottleneck
            if let Some(impact) = self.performance_impact.get(&current) {
                if impact.bottleneck_severity > 0.5 {
                    explanation_parts.push(format!(
                        "ROOT CAUSE: {:?} has bottleneck severity {:.1}%",
                        current,
                        impact.bottleneck_severity * 100.0
                    ));
                    break;
                }
            }
        }

        let root_cause = *chain.last().unwrap();
        let confidence = 0.7 + (chain.len() as f64 * 0.05).min(0.25); // Higher confidence for shorter chains

        let causal_chain = CausalChain {
            id: format!("chain_{}_{}", bottleneck.id, Instant::now().elapsed().as_millis()),
            symptom: bottleneck.clone(),
            chain: chain.clone(),
            root_cause,
            explanation: explanation_parts.join("\n"),
            confidence,
            discovered_at: Instant::now(),
        };

        self.causal_chains.push(causal_chain.clone());
        self.stats.causal_chains_discovered += 1;

        // Update average chain length
        let total_length: usize = self.causal_chains.iter().map(|c| c.chain.len()).sum();
        self.stats.avg_chain_length = total_length as f64 / self.causal_chains.len() as f64;

        Ok(causal_chain)
    }

    /// Get performance impact for a component
    pub fn get_impact(&self, component: ComponentId) -> Option<&PerformanceImpact> {
        self.performance_impact.get(&component)
    }

    /// Get all components affected by a component
    pub fn get_downstream_components(&self, component: ComponentId) -> Vec<ComponentId> {
        self.edges.iter()
            .filter(|e| e.from == component)
            .map(|e| e.to)
            .collect()
    }

    /// Get all components that affect a component
    pub fn get_upstream_components(&self, component: ComponentId) -> Vec<ComponentId> {
        self.edges.iter()
            .filter(|e| e.to == component)
            .map(|e| e.from)
            .collect()
    }

    /// Get recent causal chains
    pub fn get_recent_chains(&self, limit: usize) -> Vec<&CausalChain> {
        self.causal_chains.iter()
            .rev()
            .take(limit)
            .collect()
    }

    /// Get statistics
    pub fn get_stats(&self) -> &GraphStats {
        &self.stats
    }
}

impl Default for ArchitecturalCausalGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::recursive_improvement::types::BottleneckType;

    #[test]
    fn test_graph_initialization() {
        let graph = ArchitecturalCausalGraph::new();
        assert!(graph.stats.component_count > 0);
        assert!(graph.stats.edge_count > 0);
    }

    #[test]
    fn test_downstream_components() {
        let graph = ArchitecturalCausalGraph::new();
        let downstream = graph.get_downstream_components(ComponentId::Cache);
        assert!(downstream.contains(&ComponentId::HRM));
    }

    #[test]
    fn test_update_performance() {
        let mut graph = ArchitecturalCausalGraph::new();
        graph.update_component_performance(
            ComponentId::Cache,
            Some(0.5),
            Some(Duration::from_millis(10)),
            Some(0.95),
        );

        let impact = graph.get_impact(ComponentId::Cache);
        assert!(impact.is_some());
    }
}
