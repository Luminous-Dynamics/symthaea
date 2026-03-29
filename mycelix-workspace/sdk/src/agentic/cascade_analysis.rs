// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Cascade Analysis Engine
//!
//! Model trust propagation and systemic risk in agent networks.
//!
//! ## Features
//!
//! - **Propagation Modeling**: How trust changes ripple through networks
//! - **Systemic Risk Detection**: Identify fragile network topologies
//! - **Contagion Simulation**: Model cascading failures
//! - **Stability Analysis**: Network resilience metrics
//!
//! ## Use Cases
//!
//! - Predict impact of a major agent failing
//! - Identify critical agents whose failure would cascade
//! - Design resilient network topologies
//! - Early warning for systemic instability

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

// ============================================================================
// Configuration
// ============================================================================

/// Cascade analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeConfig {
    /// Trust propagation factor (how much trust change propagates)
    pub propagation_factor: f64,
    /// Decay per hop in propagation
    pub hop_decay: f64,
    /// Maximum propagation depth
    pub max_depth: u32,
    /// Threshold for cascade trigger
    pub cascade_threshold: f64,
    /// Minimum edge weight to consider
    pub min_edge_weight: f64,
    /// Simulation iterations
    pub simulation_iterations: u32,
    /// Recovery rate per tick
    pub recovery_rate: f64,
}

impl Default for CascadeConfig {
    fn default() -> Self {
        Self {
            propagation_factor: 0.3,
            hop_decay: 0.5,
            max_depth: 5,
            cascade_threshold: 0.3,
            min_edge_weight: 0.1,
            simulation_iterations: 100,
            recovery_rate: 0.05,
        }
    }
}

// ============================================================================
// Network Model
// ============================================================================

/// Agent in the trust network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkAgent {
    /// Agent ID
    pub id: String,
    /// Current trust score
    pub trust: f64,
    /// Baseline trust (stable state)
    pub baseline_trust: f64,
    /// Is this agent currently stressed?
    pub stressed: bool,
    /// Has this agent failed?
    pub failed: bool,
    /// Recovery progress (0.0-1.0)
    pub recovery: f64,
    /// Agent importance (centrality measure)
    pub importance: f64,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl NetworkAgent {
    /// Create a new network agent with the given ID and initial trust score.
    pub fn new(id: String, trust: f64) -> Self {
        Self {
            id,
            trust,
            baseline_trust: trust,
            stressed: false,
            failed: false,
            recovery: 1.0,
            importance: 0.0,
            metadata: HashMap::new(),
        }
    }

    /// Check if agent is healthy
    pub fn is_healthy(&self) -> bool {
        !self.failed && !self.stressed
    }

    /// Apply stress to agent
    pub fn apply_stress(&mut self, amount: f64, threshold: f64) {
        self.trust = (self.trust - amount).max(0.0);
        if self.trust < threshold {
            self.failed = true;
            self.trust = 0.0;
        } else if self.trust < self.baseline_trust * 0.7 {
            self.stressed = true;
        }
    }

    /// Recover agent
    pub fn recover(&mut self, rate: f64) {
        if self.failed {
            self.recovery += rate;
            if self.recovery >= 1.0 {
                self.failed = false;
                self.trust = self.baseline_trust * 0.5;
                self.recovery = 1.0;
            }
        } else if self.stressed {
            self.trust = (self.trust + rate * self.baseline_trust).min(self.baseline_trust);
            if self.trust >= self.baseline_trust * 0.9 {
                self.stressed = false;
            }
        }
    }
}

/// Edge in the trust network (directed dependency)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkEdge {
    /// Source agent
    pub from: String,
    /// Target agent
    pub to: String,
    /// Edge weight (dependency strength)
    pub weight: f64,
    /// Edge type
    pub edge_type: EdgeType,
}

/// Types of network edges
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeType {
    /// Trust attestation
    Attestation,
    /// Direct interaction
    Interaction,
    /// Delegation relationship
    Delegation,
    /// Collateral backing
    Collateral,
    /// Generic dependency
    Dependency,
}

/// Trust network graph
#[derive(Debug, Clone)]
pub struct TrustNetwork {
    /// Agents in the network
    agents: HashMap<String, NetworkAgent>,
    /// Outgoing edges (from -> [to])
    outgoing: HashMap<String, Vec<NetworkEdge>>,
    /// Incoming edges (to -> [from])
    incoming: HashMap<String, Vec<NetworkEdge>>,
}

impl TrustNetwork {
    /// Create empty network
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            outgoing: HashMap::new(),
            incoming: HashMap::new(),
        }
    }

    /// Add agent to network
    pub fn add_agent(&mut self, agent: NetworkAgent) {
        let id = agent.id.clone();
        self.agents.insert(id.clone(), agent);
        self.outgoing.entry(id.clone()).or_default();
        self.incoming.entry(id).or_default();
    }

    /// Add edge to network
    pub fn add_edge(&mut self, edge: NetworkEdge) {
        self.outgoing
            .entry(edge.from.clone())
            .or_default()
            .push(edge.clone());
        self.incoming.entry(edge.to.clone()).or_default().push(edge);
    }

    /// Get agent by ID
    pub fn get_agent(&self, id: &str) -> Option<&NetworkAgent> {
        self.agents.get(id)
    }

    /// Get mutable agent by ID
    pub fn get_agent_mut(&mut self, id: &str) -> Option<&mut NetworkAgent> {
        self.agents.get_mut(id)
    }

    /// Get outgoing edges for agent
    pub fn outgoing_edges(&self, id: &str) -> impl Iterator<Item = &NetworkEdge> {
        self.outgoing.get(id).into_iter().flatten()
    }

    /// Get incoming edges for agent
    pub fn incoming_edges(&self, id: &str) -> impl Iterator<Item = &NetworkEdge> {
        self.incoming.get(id).into_iter().flatten()
    }

    /// Get all agents
    pub fn agents(&self) -> impl Iterator<Item = &NetworkAgent> {
        self.agents.values()
    }

    /// Get all agent IDs
    pub fn agent_ids(&self) -> impl Iterator<Item = &String> {
        self.agents.keys()
    }

    /// Number of agents
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    /// Number of edges
    pub fn edge_count(&self) -> usize {
        self.outgoing.values().map(|v| v.len()).sum()
    }

    /// Calculate degree centrality for all agents
    pub fn calculate_centrality(&mut self) {
        let n = self.agents.len() as f64;
        if n <= 1.0 {
            return;
        }

        for agent_id in self.agents.keys().cloned().collect::<Vec<_>>() {
            let in_degree = self.incoming.get(&agent_id).map(|v| v.len()).unwrap_or(0) as f64;
            let out_degree = self.outgoing.get(&agent_id).map(|v| v.len()).unwrap_or(0) as f64;

            // Normalized degree centrality
            let centrality = (in_degree + out_degree) / (2.0 * (n - 1.0));

            if let Some(agent) = self.agents.get_mut(&agent_id) {
                agent.importance = centrality;
            }
        }
    }

    /// Reset all agents to baseline
    pub fn reset(&mut self) {
        for agent in self.agents.values_mut() {
            agent.trust = agent.baseline_trust;
            agent.stressed = false;
            agent.failed = false;
            agent.recovery = 1.0;
        }
    }

    /// Clone network state
    pub fn snapshot(&self) -> NetworkSnapshot {
        NetworkSnapshot {
            agent_states: self
                .agents
                .iter()
                .map(|(id, a)| {
                    (
                        id.clone(),
                        AgentState {
                            trust: a.trust,
                            stressed: a.stressed,
                            failed: a.failed,
                        },
                    )
                })
                .collect(),
        }
    }

    /// Restore network state from snapshot
    pub fn restore(&mut self, snapshot: &NetworkSnapshot) {
        for (id, state) in &snapshot.agent_states {
            if let Some(agent) = self.agents.get_mut(id) {
                agent.trust = state.trust;
                agent.stressed = state.stressed;
                agent.failed = state.failed;
            }
        }
    }
}

impl Default for TrustNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// Network state snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSnapshot {
    /// Map of agent ID to captured state.
    pub agent_states: HashMap<String, AgentState>,
}

/// Agent state for snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    /// Current trust score.
    pub trust: f64,
    /// Whether the agent is stressed.
    pub stressed: bool,
    /// Whether the agent has failed.
    pub failed: bool,
}

// ============================================================================
// Cascade Engine
// ============================================================================

/// Cascade analysis engine
#[derive(Debug)]
pub struct CascadeEngine {
    config: CascadeConfig,
    network: TrustNetwork,
    history: Vec<CascadeEvent>,
}

/// Cascade event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeEvent {
    /// Event ID
    pub id: String,
    /// Event type
    pub event_type: CascadeEventType,
    /// Affected agent
    pub agent_id: String,
    /// Trust change
    pub trust_delta: f64,
    /// Depth in cascade
    pub depth: u32,
    /// Trigger agent (if cascaded)
    pub triggered_by: Option<String>,
    /// Timestamp
    pub timestamp: u64,
}

/// Types of cascade events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CascadeEventType {
    /// Initial shock
    Shock,
    /// Propagated stress
    Propagation,
    /// Agent failure
    Failure,
    /// Recovery
    Recovery,
}

impl CascadeEngine {
    /// Create new cascade engine
    pub fn new(config: CascadeConfig) -> Self {
        Self {
            config,
            network: TrustNetwork::new(),
            history: Vec::new(),
        }
    }

    /// Get network reference
    pub fn network(&self) -> &TrustNetwork {
        &self.network
    }

    /// Get mutable network reference
    pub fn network_mut(&mut self) -> &mut TrustNetwork {
        &mut self.network
    }

    /// Apply shock to an agent and simulate cascade
    pub fn apply_shock(
        &mut self,
        agent_id: &str,
        shock_magnitude: f64,
        timestamp: u64,
    ) -> CascadeResult {
        // Record initial shock
        self.history.push(CascadeEvent {
            id: format!("shock-{}-{}", agent_id, timestamp),
            event_type: CascadeEventType::Shock,
            agent_id: agent_id.to_string(),
            trust_delta: -shock_magnitude,
            depth: 0,
            triggered_by: None,
            timestamp,
        });

        // Apply shock to initial agent
        if let Some(agent) = self.network.get_agent_mut(agent_id) {
            agent.apply_stress(shock_magnitude, self.config.cascade_threshold);
        }

        // Propagate cascade
        let mut affected = HashSet::new();
        affected.insert(agent_id.to_string());

        let mut queue: VecDeque<(String, u32, f64)> = VecDeque::new();
        queue.push_back((agent_id.to_string(), 0, shock_magnitude));

        while let Some((current_id, depth, magnitude)) = queue.pop_front() {
            if depth >= self.config.max_depth {
                continue;
            }

            // Propagate to dependents
            let outgoing: Vec<_> = self
                .network
                .outgoing_edges(&current_id)
                .filter(|e| e.weight >= self.config.min_edge_weight)
                .cloned()
                .collect();

            for edge in outgoing {
                if affected.contains(&edge.to) {
                    continue;
                }

                let propagated_magnitude = magnitude
                    * self.config.propagation_factor
                    * edge.weight
                    * self.config.hop_decay.powi(depth as i32);

                if propagated_magnitude < 0.01 {
                    continue;
                }

                // Apply stress to dependent
                let failed = if let Some(agent) = self.network.get_agent_mut(&edge.to) {
                    agent.apply_stress(propagated_magnitude, self.config.cascade_threshold);
                    agent.failed
                } else {
                    false
                };

                affected.insert(edge.to.clone());

                self.history.push(CascadeEvent {
                    id: format!("prop-{}-{}", edge.to, timestamp),
                    event_type: if failed {
                        CascadeEventType::Failure
                    } else {
                        CascadeEventType::Propagation
                    },
                    agent_id: edge.to.clone(),
                    trust_delta: -propagated_magnitude,
                    depth: depth + 1,
                    triggered_by: Some(current_id.clone()),
                    timestamp,
                });

                queue.push_back((edge.to, depth + 1, propagated_magnitude));
            }
        }

        // Calculate result
        let failed_count = self.network.agents().filter(|a| a.failed).count();

        let stressed_count = self.network.agents().filter(|a| a.stressed).count();

        let total_trust_loss: f64 = self
            .network
            .agents()
            .map(|a| a.baseline_trust - a.trust)
            .sum();

        CascadeResult {
            initial_agent: agent_id.to_string(),
            shock_magnitude,
            agents_affected: affected.len(),
            agents_failed: failed_count,
            agents_stressed: stressed_count,
            total_trust_loss,
            max_depth_reached: self
                .history
                .iter()
                .filter(|e| e.timestamp == timestamp)
                .map(|e| e.depth)
                .max()
                .unwrap_or(0),
            events: self
                .history
                .iter()
                .filter(|e| e.timestamp == timestamp)
                .cloned()
                .collect(),
        }
    }

    /// Simulate recovery over time
    pub fn simulate_recovery(&mut self, ticks: u32) -> RecoveryResult {
        let mut tick_snapshots = Vec::new();

        for tick in 0..ticks {
            for agent in self.network.agents.values_mut() {
                agent.recover(self.config.recovery_rate);
            }

            tick_snapshots.push(TickSnapshot {
                tick,
                failed_count: self.network.agents().filter(|a| a.failed).count(),
                stressed_count: self.network.agents().filter(|a| a.stressed).count(),
                average_trust: self.network.agents().map(|a| a.trust).sum::<f64>()
                    / self.network.agent_count() as f64,
            });

            // Check if fully recovered
            if self.network.agents().all(|a| a.is_healthy()) {
                break;
            }
        }

        RecoveryResult {
            ticks_to_recovery: tick_snapshots.len() as u32,
            fully_recovered: self.network.agents().all(|a| a.is_healthy()),
            snapshots: tick_snapshots,
        }
    }

    /// Identify critical agents (whose failure would cause most damage)
    pub fn identify_critical_agents(&mut self, top_n: usize) -> Vec<CriticalAgent> {
        // Save current state
        let snapshot = self.network.snapshot();

        let mut results = Vec::new();
        let agent_ids: Vec<_> = self.network.agent_ids().cloned().collect();

        for agent_id in &agent_ids {
            // Reset network
            self.network.restore(&snapshot);

            // Simulate this agent failing
            let result = self.apply_shock(agent_id, 1.0, 0);

            results.push(CriticalAgent {
                agent_id: agent_id.clone(),
                failure_impact: result.total_trust_loss,
                cascade_reach: result.agents_affected,
                failures_caused: result.agents_failed,
            });

            // Reset network
            self.network.restore(&snapshot);
        }

        // Sort by impact (unwrap is safe for f64 that aren't NaN)
        #[allow(clippy::unwrap_used)]
        results.sort_by(|a, b| b.failure_impact.partial_cmp(&a.failure_impact).unwrap());
        results.truncate(top_n);

        // Restore original state
        self.network.restore(&snapshot);

        results
    }

    /// Calculate network resilience score
    pub fn resilience_score(&mut self) -> ResilienceScore {
        let snapshot = self.network.snapshot();

        // Test various shock scenarios
        let mut shock_results = Vec::new();
        let agent_ids: Vec<_> = self.network.agent_ids().cloned().collect();

        for agent_id in agent_ids.iter().take(10.min(agent_ids.len())) {
            self.network.restore(&snapshot);
            let result = self.apply_shock(agent_id, 0.5, 0);
            shock_results.push(result);
            self.network.restore(&snapshot);
        }

        // Calculate metrics
        let avg_cascade_size = shock_results
            .iter()
            .map(|r| r.agents_affected as f64)
            .sum::<f64>()
            / shock_results.len() as f64;

        let avg_failures = shock_results
            .iter()
            .map(|r| r.agents_failed as f64)
            .sum::<f64>()
            / shock_results.len() as f64;

        let max_cascade = shock_results
            .iter()
            .map(|r| r.agents_affected)
            .max()
            .unwrap_or(0);

        // Calculate centrality distribution
        let total_agents = self.network.agent_count() as f64;
        let high_importance = self.network.agents().filter(|a| a.importance > 0.5).count() as f64;

        let concentration = high_importance / total_agents;

        // Resilience = 1 - (normalized risk factors)
        let cascade_risk = avg_cascade_size / total_agents;
        let failure_risk = avg_failures / total_agents;
        let concentration_risk = concentration;

        let resilience = 1.0 - ((cascade_risk + failure_risk + concentration_risk) / 3.0).min(1.0);

        ResilienceScore {
            overall: resilience,
            cascade_resistance: 1.0 - cascade_risk,
            failure_resistance: 1.0 - failure_risk,
            concentration_risk,
            average_cascade_size: avg_cascade_size,
            maximum_cascade_size: max_cascade,
        }
    }

    /// Find contagion paths between agents
    pub fn find_contagion_paths(&self, from: &str, to: &str, max_depth: u32) -> Vec<ContagionPath> {
        let mut paths = Vec::new();
        let mut visited = HashSet::new();
        let mut current_path = Vec::new();

        self.dfs_paths(
            from,
            to,
            max_depth,
            &mut visited,
            &mut current_path,
            &mut paths,
        );

        paths
    }

    fn dfs_paths(
        &self,
        current: &str,
        target: &str,
        remaining_depth: u32,
        visited: &mut HashSet<String>,
        current_path: &mut Vec<String>,
        paths: &mut Vec<ContagionPath>,
    ) {
        if remaining_depth == 0 {
            return;
        }

        visited.insert(current.to_string());
        current_path.push(current.to_string());

        if current == target && current_path.len() > 1 {
            paths.push(ContagionPath {
                nodes: current_path.clone(),
                total_weight: self.calculate_path_weight(current_path),
            });
        } else {
            for edge in self.network.outgoing_edges(current) {
                if !visited.contains(&edge.to) && edge.weight >= self.config.min_edge_weight {
                    self.dfs_paths(
                        &edge.to,
                        target,
                        remaining_depth - 1,
                        visited,
                        current_path,
                        paths,
                    );
                }
            }
        }

        current_path.pop();
        visited.remove(current);
    }

    fn calculate_path_weight(&self, path: &[String]) -> f64 {
        let mut weight = 1.0;

        for window in path.windows(2) {
            if let Some(edge) = self
                .network
                .outgoing_edges(&window[0])
                .find(|e| e.to == window[1])
            {
                weight *= edge.weight * self.config.hop_decay;
            }
        }

        weight
    }

    /// Analyze network topology for systemic risk
    pub fn topology_analysis(&self) -> TopologyAnalysis {
        let n = self.network.agent_count();
        let m = self.network.edge_count();

        // Calculate various topology metrics
        let density = if n > 1 {
            m as f64 / (n * (n - 1)) as f64
        } else {
            0.0
        };

        // Find strongly connected components (simplified)
        let avg_degree = if n > 0 {
            (2 * m) as f64 / n as f64
        } else {
            0.0
        };

        // Calculate clustering coefficient (simplified)
        let mut total_clustering = 0.0;
        for agent_id in self.network.agent_ids() {
            let neighbors: HashSet<_> = self
                .network
                .outgoing_edges(agent_id)
                .map(|e| e.to.clone())
                .chain(
                    self.network
                        .incoming_edges(agent_id)
                        .map(|e| e.from.clone()),
                )
                .collect();

            if neighbors.len() < 2 {
                continue;
            }

            let mut triangles = 0;
            let neighbor_vec: Vec<_> = neighbors.iter().collect();

            for i in 0..neighbor_vec.len() {
                for j in (i + 1)..neighbor_vec.len() {
                    // Check if neighbors are connected
                    if self
                        .network
                        .outgoing_edges(neighbor_vec[i])
                        .any(|e| &e.to == neighbor_vec[j])
                    {
                        triangles += 1;
                    }
                }
            }

            let possible = neighbors.len() * (neighbors.len() - 1) / 2;
            if possible > 0 {
                total_clustering += triangles as f64 / possible as f64;
            }
        }

        let clustering_coefficient = if n > 0 {
            total_clustering / n as f64
        } else {
            0.0
        };

        TopologyAnalysis {
            node_count: n,
            edge_count: m,
            density,
            average_degree: avg_degree,
            clustering_coefficient,
            risk_assessment: if density > 0.5 && clustering_coefficient > 0.3 {
                TopologyRisk::High
            } else if density > 0.2 || clustering_coefficient > 0.2 {
                TopologyRisk::Medium
            } else {
                TopologyRisk::Low
            },
        }
    }

    /// Get cascade history
    pub fn history(&self) -> &[CascadeEvent] {
        &self.history
    }

    /// Clear history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
}

/// Cascade simulation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeResult {
    /// Agent that received the initial shock.
    pub initial_agent: String,
    /// Magnitude of the initial shock.
    pub shock_magnitude: f64,
    /// Total number of agents affected by the cascade.
    pub agents_affected: usize,
    /// Number of agents that failed.
    pub agents_failed: usize,
    /// Number of agents under stress.
    pub agents_stressed: usize,
    /// Total trust lost across all agents.
    pub total_trust_loss: f64,
    /// Maximum propagation depth reached.
    pub max_depth_reached: u32,
    /// Events generated during the cascade.
    pub events: Vec<CascadeEvent>,
}

/// Recovery simulation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryResult {
    /// Number of ticks until recovery (or simulation end).
    pub ticks_to_recovery: u32,
    /// Whether the network fully recovered.
    pub fully_recovered: bool,
    /// Per-tick snapshots of network state.
    pub snapshots: Vec<TickSnapshot>,
}

/// Snapshot at a simulation tick
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickSnapshot {
    /// Tick number.
    pub tick: u32,
    /// Number of failed agents at this tick.
    pub failed_count: usize,
    /// Number of stressed agents at this tick.
    pub stressed_count: usize,
    /// Average trust across all agents.
    pub average_trust: f64,
}

/// Critical agent analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalAgent {
    /// Agent identifier.
    pub agent_id: String,
    /// Total trust loss if this agent fails.
    pub failure_impact: f64,
    /// Number of agents affected by failure cascade.
    pub cascade_reach: usize,
    /// Number of agent failures caused.
    pub failures_caused: usize,
}

/// Network resilience score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResilienceScore {
    /// Overall resilience score (0.0-1.0).
    pub overall: f64,
    /// Resistance to cascade propagation (0.0-1.0).
    pub cascade_resistance: f64,
    /// Resistance to agent failures (0.0-1.0).
    pub failure_resistance: f64,
    /// Risk from trust concentration in few agents (0.0-1.0).
    pub concentration_risk: f64,
    /// Average number of agents affected per shock.
    pub average_cascade_size: f64,
    /// Maximum cascade size observed.
    pub maximum_cascade_size: usize,
}

/// Contagion path between agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContagionPath {
    /// Ordered list of agent IDs forming the path.
    pub nodes: Vec<String>,
    /// Cumulative propagation weight along the path.
    pub total_weight: f64,
}

/// Topology analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyAnalysis {
    /// Number of nodes in the network.
    pub node_count: usize,
    /// Number of edges in the network.
    pub edge_count: usize,
    /// Graph density (0.0-1.0).
    pub density: f64,
    /// Average node degree.
    pub average_degree: f64,
    /// Clustering coefficient (0.0-1.0).
    pub clustering_coefficient: f64,
    /// Overall topology risk assessment.
    pub risk_assessment: TopologyRisk,
}

/// Topology risk level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyRisk {
    /// Low systemic risk.
    Low,
    /// Medium systemic risk.
    Medium,
    /// High systemic risk.
    High,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_network() -> TrustNetwork {
        let mut network = TrustNetwork::new();

        // Create agents
        for i in 0..5 {
            network.add_agent(NetworkAgent::new(format!("agent-{}", i), 0.8));
        }

        // Create edges (linear chain with some cross-links)
        network.add_edge(NetworkEdge {
            from: "agent-0".to_string(),
            to: "agent-1".to_string(),
            weight: 0.7,
            edge_type: EdgeType::Attestation,
        });
        network.add_edge(NetworkEdge {
            from: "agent-1".to_string(),
            to: "agent-2".to_string(),
            weight: 0.6,
            edge_type: EdgeType::Attestation,
        });
        network.add_edge(NetworkEdge {
            from: "agent-2".to_string(),
            to: "agent-3".to_string(),
            weight: 0.5,
            edge_type: EdgeType::Attestation,
        });
        network.add_edge(NetworkEdge {
            from: "agent-3".to_string(),
            to: "agent-4".to_string(),
            weight: 0.4,
            edge_type: EdgeType::Attestation,
        });
        // Cross-link
        network.add_edge(NetworkEdge {
            from: "agent-0".to_string(),
            to: "agent-3".to_string(),
            weight: 0.3,
            edge_type: EdgeType::Interaction,
        });

        network
    }

    #[test]
    fn test_network_creation() {
        let network = create_test_network();
        assert_eq!(network.agent_count(), 5);
        assert_eq!(network.edge_count(), 5);
    }

    #[test]
    fn test_cascade_propagation() {
        let mut engine = CascadeEngine::new(CascadeConfig::default());
        *engine.network_mut() = create_test_network();

        let result = engine.apply_shock("agent-0", 0.5, 1000);

        assert_eq!(result.initial_agent, "agent-0");
        assert!(result.agents_affected > 1, "Cascade should spread");
        assert!(result.total_trust_loss > 0.0, "Should cause trust loss");
    }

    #[test]
    fn test_critical_agent_identification() {
        let mut engine = CascadeEngine::new(CascadeConfig::default());
        *engine.network_mut() = create_test_network();
        engine.network_mut().calculate_centrality();

        let critical = engine.identify_critical_agents(3);

        assert!(!critical.is_empty());
        // First agent should have highest impact (it's at the start of chain)
        assert!(critical[0].failure_impact >= critical.last().unwrap().failure_impact);
    }

    #[test]
    fn test_resilience_score() {
        let mut engine = CascadeEngine::new(CascadeConfig::default());
        *engine.network_mut() = create_test_network();
        engine.network_mut().calculate_centrality();

        let score = engine.resilience_score();

        assert!(score.overall >= 0.0 && score.overall <= 1.0);
        assert!(score.cascade_resistance >= 0.0 && score.cascade_resistance <= 1.0);
    }

    #[test]
    fn test_recovery_simulation() {
        let mut engine = CascadeEngine::new(CascadeConfig {
            recovery_rate: 0.2,
            ..Default::default()
        });
        *engine.network_mut() = create_test_network();

        // Apply shock
        engine.apply_shock("agent-0", 0.5, 1000);

        // Simulate recovery
        let result = engine.simulate_recovery(50);

        assert!(result.snapshots.len() > 0);
        // Should recover or at least improve
        if result.snapshots.len() > 1 {
            let first = &result.snapshots[0];
            let last = result.snapshots.last().unwrap();
            assert!(last.average_trust >= first.average_trust);
        }
    }

    #[test]
    fn test_contagion_paths() {
        let engine = CascadeEngine::new(CascadeConfig::default());
        let mut network = create_test_network();
        let engine_with_network = CascadeEngine {
            config: engine.config,
            network,
            history: Vec::new(),
        };

        let paths = engine_with_network.find_contagion_paths("agent-0", "agent-4", 5);

        assert!(!paths.is_empty(), "Should find at least one path");

        // There should be both direct and indirect paths
        let path_lengths: Vec<_> = paths.iter().map(|p| p.nodes.len()).collect();
        assert!(
            path_lengths.iter().any(|&l| l > 2),
            "Should have paths longer than 2"
        );
    }

    #[test]
    fn test_topology_analysis() {
        let mut engine = CascadeEngine::new(CascadeConfig::default());
        *engine.network_mut() = create_test_network();

        let analysis = engine.topology_analysis();

        assert_eq!(analysis.node_count, 5);
        assert_eq!(analysis.edge_count, 5);
        assert!(analysis.density > 0.0);
        assert!(analysis.average_degree > 0.0);
    }

    #[test]
    fn test_agent_stress_and_failure() {
        let mut agent = NetworkAgent::new("test".to_string(), 0.8);

        // Apply mild stress
        agent.apply_stress(0.2, 0.3);
        assert!(!agent.failed);
        assert!((agent.trust - 0.6).abs() < 0.01);

        // Apply severe stress
        agent.apply_stress(0.5, 0.3);
        assert!(agent.failed);
        assert!((agent.trust - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_network_snapshot_restore() {
        let mut network = create_test_network();

        // Modify state
        if let Some(agent) = network.get_agent_mut("agent-0") {
            agent.trust = 0.3;
            agent.stressed = true;
        }

        // Take snapshot
        let snapshot = network.snapshot();

        // Modify more
        if let Some(agent) = network.get_agent_mut("agent-0") {
            agent.trust = 0.1;
            agent.failed = true;
        }

        // Restore
        network.restore(&snapshot);

        let agent = network.get_agent("agent-0").unwrap();
        assert!((agent.trust - 0.3).abs() < 0.01);
        assert!(agent.stressed);
        assert!(!agent.failed);
    }
}
