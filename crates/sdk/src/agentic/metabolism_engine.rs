// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Metabolism Engine - SCEI Component
//!
//! The Metabolism Engine tracks resource flows through the agent ecosystem,
//! managing energy/KREDIT consumption, production, and transformation. It provides:
//!
//! - **Resource Flow Tracking**: Monitor KREDIT, attention, and compute flows
//! - **Metabolic Rates**: Track consumption/production rates per agent
//! - **Energy Balance**: Ensure sustainable resource utilization
//! - **Flow Optimization**: Suggest resource reallocation for efficiency
//! - **Metabolic Health**: Detect resource starvation or hoarding
//!
//! # Integration with MATL
//!
//! Metabolic health feeds into K-Vector updates:
//! - Sustainable resource use boosts k_s (sustainability)
//! - Resource hoarding penalties affect k_h (harmonic)
//! - Starvation triggers k_a (activity) decay

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "ts-export")]
use ts_rs::TS;

// ============================================================================
// Core Types
// ============================================================================

/// Types of resources in the ecosystem
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum ResourceType {
    /// KREDIT - primary economic resource
    Kredit,
    /// Attention - cognitive processing allocation
    Attention,
    /// Compute - computational resources
    Compute,
    /// Storage - data storage capacity
    Storage,
    /// Bandwidth - network communication
    Bandwidth,
    /// Custom resource type
    Custom(String),
}

/// Direction of resource flow
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum FlowDirection {
    /// Resource flowing into agent
    Inflow,
    /// Resource flowing out of agent
    Outflow,
    /// Internal transformation (resource type change)
    Transform,
}

/// A single resource flow event
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct ResourceFlow {
    /// Resource type
    pub resource_type: ResourceType,
    /// Flow direction
    pub direction: FlowDirection,
    /// Amount of resource
    pub amount: f64,
    /// Source agent (None for system)
    pub source: Option<String>,
    /// Destination agent (None for system)
    pub destination: Option<String>,
    /// Timestamp of flow
    pub timestamp: u64,
    /// Associated action/operation
    pub operation: Option<String>,
}

/// Current balance of resources for an agent
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct ResourceBalance {
    /// Current KREDIT balance
    pub kredit: f64,
    /// Current attention allocation
    pub attention: f64,
    /// Current compute allocation
    pub compute: f64,
    /// Current storage usage
    pub storage: f64,
    /// Current bandwidth usage
    pub bandwidth: f64,
    /// Custom resource balances
    pub custom: HashMap<String, f64>,
}

impl ResourceBalance {
    /// Get the current balance for a resource type
    pub fn get(&self, resource_type: &ResourceType) -> f64 {
        match resource_type {
            ResourceType::Kredit => self.kredit,
            ResourceType::Attention => self.attention,
            ResourceType::Compute => self.compute,
            ResourceType::Storage => self.storage,
            ResourceType::Bandwidth => self.bandwidth,
            ResourceType::Custom(name) => *self.custom.get(name).unwrap_or(&0.0),
        }
    }

    /// Set the balance for a resource type
    pub fn set(&mut self, resource_type: &ResourceType, value: f64) {
        match resource_type {
            ResourceType::Kredit => self.kredit = value,
            ResourceType::Attention => self.attention = value,
            ResourceType::Compute => self.compute = value,
            ResourceType::Storage => self.storage = value,
            ResourceType::Bandwidth => self.bandwidth = value,
            ResourceType::Custom(name) => {
                self.custom.insert(name.clone(), value);
            }
        }
    }

    /// Add (or subtract if negative) an amount to a resource balance
    pub fn add(&mut self, resource_type: &ResourceType, amount: f64) {
        let current = self.get(resource_type);
        self.set(resource_type, current + amount);
    }
}

/// Rate of resource consumption/production
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct MetabolicRate {
    /// Average inflow rate (units per hour)
    pub inflow_rate: f64,
    /// Average outflow rate (units per hour)
    pub outflow_rate: f64,
    /// Net rate (positive = accumulating, negative = depleting)
    pub net_rate: f64,
    /// Variance in rate (stability measure)
    pub variance: f64,
    /// Samples used to calculate rate
    pub sample_count: u64,
}

impl MetabolicRate {
    /// Update the metabolic rate with a new resource flow observation
    pub fn update(&mut self, flow: &ResourceFlow, window_hours: f64) {
        let rate = flow.amount / window_hours;

        match flow.direction {
            FlowDirection::Inflow => {
                self.inflow_rate = (self.inflow_rate * self.sample_count as f64 + rate)
                    / (self.sample_count + 1) as f64;
            }
            FlowDirection::Outflow => {
                self.outflow_rate = (self.outflow_rate * self.sample_count as f64 + rate)
                    / (self.sample_count + 1) as f64;
            }
            FlowDirection::Transform => {
                // Transforms don't affect net rate
            }
        }

        self.net_rate = self.inflow_rate - self.outflow_rate;
        self.sample_count += 1;
    }
}

/// Metabolic process representing resource transformation
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct MetabolicProcess {
    /// Process identifier
    pub process_id: String,
    /// Input resources required
    pub inputs: Vec<(ResourceType, f64)>,
    /// Output resources produced
    pub outputs: Vec<(ResourceType, f64)>,
    /// Efficiency factor (0.0 - 1.0)
    pub efficiency: f64,
    /// Whether process is currently active
    pub active: bool,
    /// Last execution timestamp
    pub last_execution: u64,
}

impl MetabolicProcess {
    /// Create a new metabolic process
    pub fn new(process_id: impl Into<String>) -> Self {
        Self {
            process_id: process_id.into(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            efficiency: 1.0,
            active: true,
            last_execution: 0,
        }
    }

    /// Add an input resource requirement to this process (builder pattern)
    pub fn with_input(mut self, resource: ResourceType, amount: f64) -> Self {
        self.inputs.push((resource, amount));
        self
    }

    /// Add an output resource produced by this process (builder pattern)
    pub fn with_output(mut self, resource: ResourceType, amount: f64) -> Self {
        self.outputs.push((resource, amount));
        self
    }

    /// Set process efficiency factor (builder pattern)
    pub fn with_efficiency(mut self, efficiency: f64) -> Self {
        self.efficiency = efficiency.clamp(0.0, 1.0);
        self
    }

    /// Calculate net resource change from this process
    pub fn net_change(&self, resource: &ResourceType) -> f64 {
        let input: f64 = self
            .inputs
            .iter()
            .filter(|(r, _)| r == resource)
            .map(|(_, a)| a)
            .sum();

        let output: f64 = self
            .outputs
            .iter()
            .filter(|(r, _)| r == resource)
            .map(|(_, a)| a * self.efficiency)
            .sum();

        output - input
    }
}

// ============================================================================
// Agent Metabolism State
// ============================================================================

/// Complete metabolic state for an agent
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct MetabolismState {
    /// Agent identifier
    pub agent_id: String,
    /// Current resource balances
    pub balances: ResourceBalance,
    /// Metabolic rates by resource type
    pub rates: HashMap<String, MetabolicRate>,
    /// Active metabolic processes
    pub processes: Vec<MetabolicProcess>,
    /// Recent flow history (last N flows)
    pub recent_flows: Vec<ResourceFlow>,
    /// Metabolic health score (0.0 - 1.0)
    pub health_score: f64,
    /// Health status
    pub health_status: MetabolicHealthStatus,
    /// Last update timestamp
    pub last_update: u64,
}

/// Metabolic health status
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum MetabolicHealthStatus {
    /// Healthy sustainable metabolism
    Healthy,
    /// Slight imbalance, monitor
    Warning,
    /// Resource starvation detected
    Starving,
    /// Resource hoarding detected
    Hoarding,
    /// Critical metabolic failure
    Critical,
}

impl MetabolismState {
    /// Create a new metabolism state for an agent
    pub fn new(agent_id: String) -> Self {
        Self {
            agent_id,
            balances: ResourceBalance::default(),
            rates: HashMap::new(),
            processes: Vec::new(),
            recent_flows: Vec::new(),
            health_score: 1.0,
            health_status: MetabolicHealthStatus::Healthy,
            last_update: 0,
        }
    }

    /// Record a resource flow
    pub fn record_flow(&mut self, flow: ResourceFlow) {
        // Update balance
        match flow.direction {
            FlowDirection::Inflow => {
                self.balances.add(&flow.resource_type, flow.amount);
            }
            FlowDirection::Outflow => {
                self.balances.add(&flow.resource_type, -flow.amount);
            }
            FlowDirection::Transform => {
                // Handled by process
            }
        }

        // Update rate
        let rate_key = format!("{:?}", flow.resource_type);
        let rate = self.rates.entry(rate_key).or_default();
        rate.update(&flow, 1.0); // Assume 1 hour window

        // Store in history
        self.recent_flows.push(flow);
        if self.recent_flows.len() > 1000 {
            self.recent_flows.remove(0);
        }

        self.last_update = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.update_health();
    }

    fn update_health(&mut self) {
        // Calculate health based on multiple factors
        let mut health: f64 = 1.0;

        // Check KREDIT balance
        if self.balances.kredit < 0.0 {
            health -= 0.3; // Negative KREDIT is bad
        } else if self.balances.kredit < 100.0 {
            health -= 0.1; // Low KREDIT is concerning
        }

        // Check for hoarding (very high balance with low outflow)
        let kredit_rate = self.rates.get("Kredit");
        if let Some(rate) = kredit_rate {
            if self.balances.kredit > 10000.0 && rate.outflow_rate < 10.0 {
                health -= 0.2; // Hoarding penalty
            }
            if rate.net_rate < -100.0 {
                health -= 0.2; // Rapid depletion
            }
        }

        // Check attention allocation
        if self.balances.attention < 0.1 {
            health -= 0.15; // Attention starvation
        }

        self.health_score = health.clamp(0.0, 1.0);

        // Determine status - check extreme conditions first
        self.health_status = if self.balances.kredit < 0.0 {
            MetabolicHealthStatus::Starving
        } else if self.balances.kredit > 10000.0
            && self
                .rates
                .get("Kredit")
                .map(|r| r.outflow_rate < 10.0)
                .unwrap_or(false)
        {
            MetabolicHealthStatus::Hoarding
        } else if health >= 0.8 {
            MetabolicHealthStatus::Healthy
        } else if health >= 0.6 {
            MetabolicHealthStatus::Warning
        } else {
            MetabolicHealthStatus::Critical
        };
    }

    /// Get K-Vector adjustments based on metabolic health
    pub fn kvector_adjustment(&self) -> KVectorMetabolicAdjustment {
        let k_s_delta = match self.health_status {
            MetabolicHealthStatus::Healthy => 0.03,
            MetabolicHealthStatus::Warning => 0.0,
            MetabolicHealthStatus::Starving => -0.05,
            MetabolicHealthStatus::Hoarding => -0.08,
            MetabolicHealthStatus::Critical => -0.10,
        };

        let k_h_delta = if self.health_status == MetabolicHealthStatus::Hoarding {
            -0.05 // Harmonic penalty for hoarding
        } else if self.health_score > 0.9 {
            0.02 // Harmonic boost for healthy metabolism
        } else {
            0.0
        };

        let k_a_delta = if self.health_status == MetabolicHealthStatus::Starving {
            -0.05 // Activity penalty when starving
        } else if self.health_status == MetabolicHealthStatus::Healthy {
            0.01
        } else {
            0.0
        };

        KVectorMetabolicAdjustment {
            k_s_delta,
            k_h_delta,
            k_a_delta,
            health_score: self.health_score,
            health_status: self.health_status.clone(),
        }
    }
}

/// K-Vector adjustments from metabolism
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct KVectorMetabolicAdjustment {
    /// Sustainability dimension delta
    pub k_s_delta: f32,
    /// Historical dimension delta
    pub k_h_delta: f32,
    /// Activity dimension delta
    pub k_a_delta: f32,
    /// Overall metabolic health score
    pub health_score: f64,
    /// Current health status
    pub health_status: MetabolicHealthStatus,
}

// ============================================================================
// Metabolism Engine
// ============================================================================

/// Configuration for the metabolism engine
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetabolismEngineConfig {
    /// Maximum flow history per agent
    pub max_flow_history: usize,
    /// Health check interval (seconds)
    pub health_check_interval: u64,
    /// Hoarding threshold (KREDIT balance)
    pub hoarding_threshold: f64,
    /// Starvation threshold (KREDIT balance)
    pub starvation_threshold: f64,
    /// Rate calculation window (hours)
    pub rate_window_hours: f64,
}

impl Default for MetabolismEngineConfig {
    fn default() -> Self {
        Self {
            max_flow_history: 1000,
            health_check_interval: 300, // 5 minutes
            hoarding_threshold: 10000.0,
            starvation_threshold: 0.0,
            rate_window_hours: 1.0,
        }
    }
}

/// The Metabolism Engine
pub struct MetabolismEngine {
    config: MetabolismEngineConfig,
    agents: HashMap<String, MetabolismState>,
    global_flows: Vec<ResourceFlow>,
    total_kredit_circulating: f64,
}

impl MetabolismEngine {
    /// Create a new metabolism engine with the given configuration
    pub fn new(config: MetabolismEngineConfig) -> Self {
        Self {
            config,
            agents: HashMap::new(),
            global_flows: Vec::new(),
            total_kredit_circulating: 0.0,
        }
    }

    /// Record a resource flow
    pub fn record_flow(&mut self, flow: ResourceFlow) {
        // Update source agent
        if let Some(ref source) = flow.source {
            let state = self
                .agents
                .entry(source.clone())
                .or_insert_with(|| MetabolismState::new(source.clone()));

            let outflow = ResourceFlow {
                direction: FlowDirection::Outflow,
                ..flow.clone()
            };
            state.record_flow(outflow);
        }

        // Update destination agent
        if let Some(ref dest) = flow.destination {
            let state = self
                .agents
                .entry(dest.clone())
                .or_insert_with(|| MetabolismState::new(dest.clone()));

            let inflow = ResourceFlow {
                direction: FlowDirection::Inflow,
                ..flow.clone()
            };
            state.record_flow(inflow);
        }

        // Track global flows
        self.global_flows.push(flow.clone());
        if self.global_flows.len() > 10000 {
            self.global_flows.remove(0);
        }

        // Update circulating supply for KREDIT
        if flow.resource_type == ResourceType::Kredit {
            if flow.source.is_none() {
                // Minted from system
                self.total_kredit_circulating += flow.amount;
            } else if flow.destination.is_none() {
                // Burned to system
                self.total_kredit_circulating -= flow.amount;
            }
        }
    }

    /// Get agent's metabolism state
    pub fn get_state(&self, agent_id: &str) -> Option<&MetabolismState> {
        self.agents.get(agent_id)
    }

    /// Get agents with health issues
    pub fn unhealthy_agents(&self) -> Vec<(&str, &MetabolismState)> {
        self.agents
            .iter()
            .filter(|(_, state)| state.health_status != MetabolicHealthStatus::Healthy)
            .map(|(id, state)| (id.as_str(), state))
            .collect()
    }

    /// Get K-Vector adjustment for an agent
    pub fn get_kvector_adjustment(&self, agent_id: &str) -> Option<KVectorMetabolicAdjustment> {
        self.agents.get(agent_id).map(|s| s.kvector_adjustment())
    }

    /// Get global metabolism statistics
    pub fn stats(&self) -> MetabolismStats {
        let total_agents = self.agents.len();
        let healthy_agents = self
            .agents
            .values()
            .filter(|s| s.health_status == MetabolicHealthStatus::Healthy)
            .count();
        let starving_agents = self
            .agents
            .values()
            .filter(|s| s.health_status == MetabolicHealthStatus::Starving)
            .count();
        let hoarding_agents = self
            .agents
            .values()
            .filter(|s| s.health_status == MetabolicHealthStatus::Hoarding)
            .count();

        let avg_health = if total_agents > 0 {
            self.agents.values().map(|s| s.health_score).sum::<f64>() / total_agents as f64
        } else {
            1.0
        };

        MetabolismStats {
            total_agents,
            healthy_agents,
            starving_agents,
            hoarding_agents,
            average_health_score: avg_health,
            total_kredit_circulating: self.total_kredit_circulating,
            total_flows_recorded: self.global_flows.len(),
        }
    }

    /// Suggest resource reallocation for system health
    pub fn suggest_reallocation(&self) -> Vec<ReallocationSuggestion> {
        let mut suggestions = Vec::new();

        // Find hoarding agents
        let hoarders: Vec<_> = self
            .agents
            .iter()
            .filter(|(_, s)| s.health_status == MetabolicHealthStatus::Hoarding)
            .collect();

        // Find starving agents
        let starving: Vec<_> = self
            .agents
            .iter()
            .filter(|(_, s)| s.health_status == MetabolicHealthStatus::Starving)
            .collect();

        // Suggest transfers from hoarders to starving
        for (hoarder_id, hoarder_state) in &hoarders {
            for (starving_id, _) in &starving {
                let excess = hoarder_state.balances.kredit - self.config.hoarding_threshold;
                if excess > 100.0 {
                    suggestions.push(ReallocationSuggestion {
                        from_agent: hoarder_id.to_string(),
                        to_agent: starving_id.to_string(),
                        resource: ResourceType::Kredit,
                        amount: (excess * 0.1).min(1000.0), // Suggest 10% of excess, max 1000
                        reason: "Redistribute from hoarding to starving agent".to_string(),
                    });
                }
            }
        }

        suggestions
    }
}

/// Suggestion for resource reallocation
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct ReallocationSuggestion {
    /// Agent to take resources from
    pub from_agent: String,
    /// Agent to give resources to
    pub to_agent: String,
    /// Type of resource to reallocate
    pub resource: ResourceType,
    /// Amount to reallocate
    pub amount: f64,
    /// Reason for the reallocation
    pub reason: String,
}

/// Global metabolism statistics
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct MetabolismStats {
    /// Total number of agents tracked
    pub total_agents: usize,
    /// Agents in healthy metabolic state
    pub healthy_agents: usize,
    /// Agents experiencing resource starvation
    pub starving_agents: usize,
    /// Agents hoarding resources
    pub hoarding_agents: usize,
    /// Average health score across all agents
    pub average_health_score: f64,
    /// Total KREDIT in circulation
    pub total_kredit_circulating: f64,
    /// Total resource flows recorded
    pub total_flows_recorded: usize,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_balance() {
        let mut balance = ResourceBalance::default();
        assert_eq!(balance.get(&ResourceType::Kredit), 0.0);

        balance.add(&ResourceType::Kredit, 100.0);
        assert_eq!(balance.get(&ResourceType::Kredit), 100.0);

        balance.add(&ResourceType::Kredit, -30.0);
        assert_eq!(balance.get(&ResourceType::Kredit), 70.0);
    }

    #[test]
    fn test_metabolic_process() {
        let process = MetabolicProcess::new("compute_task")
            .with_input(ResourceType::Kredit, 10.0)
            .with_input(ResourceType::Compute, 5.0)
            .with_output(ResourceType::Attention, 8.0)
            .with_efficiency(0.9);

        assert_eq!(process.net_change(&ResourceType::Kredit), -10.0);
        assert!((process.net_change(&ResourceType::Attention) - 7.2).abs() < 0.01);
    }

    #[test]
    fn test_metabolism_state_health() {
        let mut state = MetabolismState::new("agent-1".to_string());

        // Start with healthy state
        state.balances.kredit = 1000.0;
        state.balances.attention = 0.5;
        state.update_health();
        assert_eq!(state.health_status, MetabolicHealthStatus::Healthy);
        assert!(state.health_score >= 0.8);

        // Become starving (negative kredit)
        state.balances.kredit = -100.0;
        state.update_health();
        assert_eq!(state.health_status, MetabolicHealthStatus::Starving);
        assert!(state.health_score <= 0.7); // -0.3 penalty brings it to 0.7
    }

    #[test]
    fn test_metabolism_engine_flows() {
        let mut engine = MetabolismEngine::new(MetabolismEngineConfig::default());

        // Record a KREDIT transfer
        engine.record_flow(ResourceFlow {
            resource_type: ResourceType::Kredit,
            direction: FlowDirection::Inflow,
            amount: 500.0,
            source: None, // System mint
            destination: Some("agent-1".to_string()),
            timestamp: 1000,
            operation: Some("initial_allocation".to_string()),
        });

        let state = engine.get_state("agent-1").unwrap();
        assert_eq!(state.balances.kredit, 500.0);
        assert_eq!(engine.total_kredit_circulating, 500.0);
    }

    #[test]
    fn test_metabolism_engine_stats() {
        let mut engine = MetabolismEngine::new(MetabolismEngineConfig::default());

        // Create some agents with different states
        engine.record_flow(ResourceFlow {
            resource_type: ResourceType::Kredit,
            direction: FlowDirection::Inflow,
            amount: 1000.0,
            source: None,
            destination: Some("healthy-agent".to_string()),
            timestamp: 1000,
            operation: None,
        });

        engine.record_flow(ResourceFlow {
            resource_type: ResourceType::Kredit,
            direction: FlowDirection::Outflow,
            amount: 1100.0,
            source: Some("starving-agent".to_string()),
            destination: None,
            timestamp: 1000,
            operation: None,
        });

        let stats = engine.stats();
        assert_eq!(stats.total_agents, 2);
        assert!(stats.starving_agents >= 1);
    }

    #[test]
    fn test_kvector_adjustment() {
        let mut state = MetabolismState::new("test".to_string());
        state.balances.kredit = 15000.0; // Hoarding level

        // Need to set up rate data for hoarding detection
        let mut rate = MetabolicRate::default();
        rate.outflow_rate = 5.0; // Low outflow (hoarding behavior)
        rate.inflow_rate = 100.0;
        rate.sample_count = 10;
        state.rates.insert("Kredit".to_string(), rate);

        state.update_health();

        let adj = state.kvector_adjustment();
        assert!(adj.k_s_delta < 0.0); // Sustainability penalty
        assert!(adj.k_h_delta < 0.0); // Harmonic penalty for hoarding
    }

    #[test]
    fn test_reallocation_suggestions() {
        let mut engine = MetabolismEngine::new(MetabolismEngineConfig::default());

        // Create a hoarding agent with many inflows and few outflows
        for _ in 0..10 {
            engine.record_flow(ResourceFlow {
                resource_type: ResourceType::Kredit,
                direction: FlowDirection::Inflow,
                amount: 2000.0,
                source: None,
                destination: Some("hoarder".to_string()),
                timestamp: 1000,
                operation: None,
            });
        }

        // Create a starving agent with negative balance
        engine.record_flow(ResourceFlow {
            resource_type: ResourceType::Kredit,
            direction: FlowDirection::Outflow,
            amount: 1000.0,
            source: Some("starving".to_string()),
            destination: None,
            timestamp: 1000,
            operation: None,
        });

        // Manually set up health status since the detection requires rate data
        if let Some(state) = engine.agents.get_mut("hoarder") {
            state.health_status = MetabolicHealthStatus::Hoarding;
        }
        if let Some(state) = engine.agents.get_mut("starving") {
            state.health_status = MetabolicHealthStatus::Starving;
        }

        let suggestions = engine.suggest_reallocation();
        assert!(!suggestions.is_empty());
        assert_eq!(suggestions[0].from_agent, "hoarder");
        assert_eq!(suggestions[0].to_agent, "starving");
    }
}
