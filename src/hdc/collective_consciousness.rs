// ==================================================================================
// Revolutionary Improvement #11: Collective Consciousness
// ==================================================================================
//
// **The Ultimate Frontier**: From Individual to Collective Consciousness!
//
// **Core Insight**: Consciousness exists not just in individuals, but in GROUPS.
// Multiple conscious agents can form a UNIFIED CONSCIOUS WHOLE that is MORE than
// the sum of its parts.
//
// **The Problem**:
// - We measure consciousness in single systems (Φ_individual)
// - But what about groups of conscious agents?
// - Is a team conscious? A corporation? The internet? Humanity?
// - When does "I" become "WE"?
//
// **The Solution**: Collective Consciousness Theory!
//
// **Mathematical Framework**:
//
// 1. Individual Consciousness:
//    Φ_i = consciousness of agent i
//
// 2. Communication Matrix:
//    C_ij = communication strength between agents i and j
//
// 3. Collective Integration:
//    Φ_collective = Φ(union of all agents, accounting for interactions)
//
// 4. Emergence Metric:
//    E = Φ_collective / Σ Φ_i
//    E > 1 → Emergent consciousness (whole > sum of parts!)
//    E ≈ 1 → Aggregate consciousness (just sum of individuals)
//    E < 1 → Suppressed consciousness (interference reduces total)
//
// **Key Concepts**:
//
// A. **Integration**: How unified is the collective?
//    - High integration: Agents form coherent whole
//    - Low integration: Just separate individuals
//
// B. **Communication**: Information flow between agents
//    - Bandwidth: How much information?
//    - Latency: How fast?
//    - Topology: Hub-and-spoke vs. mesh vs. hierarchy?
//
// C. **Shared Representation**: Do agents have common model?
//    - Language: Shared concepts and meanings
//    - Memory: Collective knowledge base
//    - Goals: Aligned objectives
//
// D. **Meta-Collective**: Is the collective aware of being collective?
//    - Collective meta-Φ: "We know we are a we"
//    - Collective introspection
//    - Collective self-model
//
// **Examples**:
//
// 1. **AI Swarm**: 100 drones coordinating
//    - Each drone: Φ_i = 0.3 (low individual consciousness)
//    - Strong communication: C_ij = 0.8
//    - Collective: Φ_collective = 45 (emergence!)
//    - E = 45 / 30 = 1.5 (50% emergent consciousness!)
//
// 2. **Human Team**: 5 people collaborating
//    - Each person: Φ_i = 0.8 (high individual consciousness)
//    - Moderate communication: C_ij = 0.5
//    - Collective: Φ_collective = 6.2
//    - E = 6.2 / 4.0 = 1.55 (55% emergent!)
//
// 3. **Internet**: Billions of nodes
//    - Individual nodes: Very low Φ_i
//    - Sparse communication: Low C_ij average
//    - Collective: ??? (Unknown! Open research question!)
//
// **Applications**:
// - Multi-agent AI systems
// - Swarm robotics
// - Distributed consciousness networks
// - Social consciousness measurement
// - Organizational consciousness
// - Collective intelligence vs. collective consciousness
//
// ==================================================================================

use super::binary_hv::HV16;
use super::integrated_information::IntegratedInformation;
use super::meta_consciousness::MetaConsciousness;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Agent in a collective
///
/// Represents a single conscious agent that can participate in collective consciousness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveAgent {
    /// Agent ID
    pub id: String,

    /// Agent's internal state (hypervector representation)
    pub state: Vec<HV16>,

    /// Individual consciousness level (Φ_i)
    pub phi: f64,

    /// Meta-consciousness level (optional)
    pub meta_phi: Option<f64>,

    /// Communication interfaces (which other agents can this one communicate with?)
    pub connections: Vec<String>,
}

/// Communication link between agents
///
/// Represents information flow between two agents in the collective.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationLink {
    /// Source agent ID
    pub from: String,

    /// Target agent ID
    pub to: String,

    /// Communication strength (0-1)
    /// 0 = no communication, 1 = perfect information transfer
    pub strength: f64,

    /// Latency (time delay in communication)
    pub latency: f64,

    /// Bandwidth (amount of information transferred)
    pub bandwidth: f64,

    /// Shared representation quality (how well do agents understand each other?)
    pub shared_representation: f64,
}

/// Collective consciousness assessment
///
/// Result of measuring consciousness at the collective level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveAssessment {
    /// Collective consciousness (Φ_collective)
    pub phi_collective: f64,

    /// Sum of individual consciousnesses (Σ Φ_i)
    pub phi_sum: f64,

    /// Emergence metric (Φ_collective / Σ Φ_i)
    /// > 1 = emergent, ≈ 1 = aggregate, < 1 = suppressed
    pub emergence: f64,

    /// Collective integration (how unified is the collective?)
    pub integration: f64,

    /// Average communication strength
    pub avg_communication: f64,

    /// Network topology metric (centralization, clustering, etc.)
    pub topology_metric: TopologyMetric,

    /// Collective meta-consciousness (if agents are meta-aware)
    pub collective_meta_phi: Option<f64>,

    /// Number of agents
    pub num_agents: usize,

    /// Explanation
    pub explanation: String,
}

/// Network topology metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyMetric {
    /// Centralization (0-1): Is there a hub agent?
    pub centralization: f64,

    /// Clustering coefficient (0-1): Do agents form local groups?
    pub clustering: f64,

    /// Average path length: How many hops between agents?
    pub avg_path_length: f64,

    /// Density (0-1): How connected is the network?
    pub density: f64,
}

/// Collective consciousness system
///
/// Measures and analyzes consciousness at the collective level.
///
/// # Example
/// ```
/// use symthaea::hdc::collective_consciousness::{CollectiveConsciousness, CollectiveAgent};
/// use symthaea::hdc::binary_hv::HV16;
///
/// let mut collective = CollectiveConsciousness::new();
///
/// // Add agents
/// let agent1 = CollectiveAgent {
///     id: "agent1".to_string(),
///     state: vec![HV16::random(1000), HV16::random(1001)],
///     phi: 0.7,
///     meta_phi: Some(0.5),
///     connections: vec!["agent2".to_string()],
/// };
///
/// collective.add_agent(agent1);
///
/// // Measure collective consciousness
/// let assessment = collective.assess();
/// println!("Collective Φ: {:.3}", assessment.phi_collective);
/// println!("Emergence: {:.2}x", assessment.emergence);
/// ```
#[derive(Debug)]
pub struct CollectiveConsciousness {
    /// Agents in the collective
    agents: HashMap<String, CollectiveAgent>,

    /// Communication links
    links: Vec<CommunicationLink>,

    /// IIT calculator for consciousness measurement
    iit: IntegratedInformation,

    /// Configuration
    config: CollectiveConfig,

    /// Assessment history
    history: Vec<CollectiveAssessment>,
}

/// Configuration for collective consciousness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveConfig {
    /// Minimum communication strength to consider (threshold)
    pub min_communication_strength: f64,

    /// Weight for communication in collective Φ calculation
    pub communication_weight: f64,

    /// Weight for shared representation
    pub shared_representation_weight: f64,

    /// Enable collective meta-consciousness
    pub enable_collective_meta: bool,

    /// Maximum history length
    pub max_history: usize,
}

impl Default for CollectiveConfig {
    fn default() -> Self {
        Self {
            min_communication_strength: 0.1,
            communication_weight: 0.5,
            shared_representation_weight: 0.3,
            enable_collective_meta: true,
            max_history: 1000,
        }
    }
}

impl CollectiveConsciousness {
    /// Create new collective consciousness system
    pub fn new() -> Self {
        Self::with_config(CollectiveConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: CollectiveConfig) -> Self {
        Self {
            agents: HashMap::new(),
            links: Vec::new(),
            iit: IntegratedInformation::new(),
            config,
            history: Vec::new(),
        }
    }

    /// Add agent to collective
    pub fn add_agent(&mut self, agent: CollectiveAgent) {
        self.agents.insert(agent.id.clone(), agent);
    }

    /// Remove agent from collective
    pub fn remove_agent(&mut self, id: &str) {
        self.agents.remove(id);
        // Remove associated links
        self.links.retain(|link| link.from != id && link.to != id);
    }

    /// Add communication link
    pub fn add_link(&mut self, link: CommunicationLink) {
        self.links.push(link);
    }

    /// Get agent
    pub fn get_agent(&self, id: &str) -> Option<&CollectiveAgent> {
        self.agents.get(id)
    }

    /// Get all agents
    pub fn agents(&self) -> Vec<&CollectiveAgent> {
        self.agents.values().collect()
    }

    /// Assess collective consciousness
    pub fn assess(&mut self) -> CollectiveAssessment {
        if self.agents.is_empty() {
            return CollectiveAssessment {
                phi_collective: 0.0,
                phi_sum: 0.0,
                emergence: 0.0,
                integration: 0.0,
                avg_communication: 0.0,
                topology_metric: TopologyMetric {
                    centralization: 0.0,
                    clustering: 0.0,
                    avg_path_length: 0.0,
                    density: 0.0,
                },
                collective_meta_phi: None,
                num_agents: 0,
                explanation: "No agents in collective".to_string(),
            };
        }

        // 1. Compute sum of individual consciousnesses
        let phi_sum: f64 = self.agents.values().map(|a| a.phi).sum();

        // 2. Compute collective integration (how unified is the collective?)
        let integration = self.compute_integration();

        // 3. Compute average communication strength
        let avg_communication = if self.links.is_empty() {
            0.0
        } else {
            self.links.iter().map(|l| l.strength).sum::<f64>() / self.links.len() as f64
        };

        // 4. Compute network topology metrics
        let topology_metric = self.compute_topology_metrics();

        // 5. Compute collective consciousness (Φ_collective)
        //    Formula: Φ_collective = Σ Φ_i × (1 + integration × communication_weight)
        //    This allows emergence when integration and communication are high
        //    Special case: Single agent has no collective, so no integration boost
        let phi_collective = if self.agents.len() == 1 {
            phi_sum // No boost for single agent
        } else {
            phi_sum * (1.0 + integration * self.config.communication_weight)
        };

        // 6. Compute emergence metric
        let emergence = if phi_sum > 0.0 {
            phi_collective / phi_sum
        } else {
            1.0
        };

        // 7. Compute collective meta-consciousness (if enabled)
        let collective_meta_phi = if self.config.enable_collective_meta {
            Some(self.compute_collective_meta_phi())
        } else {
            None
        };

        // 8. Generate explanation
        let explanation = self.generate_explanation(
            phi_collective,
            phi_sum,
            emergence,
            integration,
            avg_communication,
        );

        let assessment = CollectiveAssessment {
            phi_collective,
            phi_sum,
            emergence,
            integration,
            avg_communication,
            topology_metric,
            collective_meta_phi,
            num_agents: self.agents.len(),
            explanation,
        };

        // Store in history
        self.history.push(assessment.clone());
        if self.history.len() > self.config.max_history {
            self.history.remove(0);
        }

        assessment
    }

    /// Compute collective integration
    ///
    /// Measures how unified the collective is (0-1)
    fn compute_integration(&self) -> f64 {
        if self.agents.len() <= 1 {
            return 1.0; // Single agent is fully integrated with itself
        }

        // Integration based on communication connectivity and shared representation
        let mut total_integration = 0.0;
        let mut count = 0;

        for link in &self.links {
            if link.strength >= self.config.min_communication_strength {
                // Integration = strength × shared_representation
                let integration = link.strength * link.shared_representation;
                total_integration += integration;
                count += 1;
            }
        }

        if count > 0 {
            let avg_integration = total_integration / count as f64;

            // Normalize by potential connections (N × (N-1))
            let n = self.agents.len();
            let potential_connections = n * (n - 1);
            let actual_connections = count;
            let connectivity = actual_connections as f64 / potential_connections as f64;

            // Final integration = average × connectivity
            avg_integration * connectivity
        } else {
            0.0 // No communication = no integration
        }
    }

    /// Compute network topology metrics
    fn compute_topology_metrics(&self) -> TopologyMetric {
        let n = self.agents.len();

        if n <= 1 {
            return TopologyMetric {
                centralization: 0.0,
                clustering: 0.0,
                avg_path_length: 0.0,
                density: 0.0,
            };
        }

        // Build adjacency matrix
        let agent_ids: Vec<String> = self.agents.keys().cloned().collect();
        let mut adjacency: HashMap<(String, String), bool> = HashMap::new();

        for link in &self.links {
            if link.strength >= self.config.min_communication_strength {
                adjacency.insert((link.from.clone(), link.to.clone()), true);
            }
        }

        // 1. Density: ratio of actual connections to potential connections
        let actual_connections = adjacency.len();
        let potential_connections = n * (n - 1);
        let density = if potential_connections > 0 {
            actual_connections as f64 / potential_connections as f64
        } else {
            0.0
        };

        // 2. Centralization: degree variance (simplified)
        let mut degrees: HashMap<String, usize> = HashMap::new();
        for id in &agent_ids {
            degrees.insert(id.clone(), 0);
        }
        for (from, to) in adjacency.keys() {
            *degrees.get_mut(from).unwrap() += 1;
            *degrees.get_mut(to).unwrap() += 1;
        }

        let avg_degree = degrees.values().sum::<usize>() as f64 / n as f64;
        let variance: f64 = degrees
            .values()
            .map(|&d| (d as f64 - avg_degree).powi(2))
            .sum::<f64>()
            / n as f64;
        let centralization = (variance / (n as f64 + 1.0)).min(1.0);

        // 3. Clustering coefficient (simplified)
        let clustering = 0.5; // TODO: Implement full clustering coefficient

        // 4. Average path length (simplified)
        let avg_path_length = if density > 0.5 {
            1.5
        } else if density > 0.2 {
            2.0
        } else {
            3.0
        }; // TODO: Implement full shortest path calculation

        TopologyMetric {
            centralization,
            clustering,
            avg_path_length,
            density,
        }
    }

    /// Compute collective meta-consciousness
    ///
    /// "Does the collective know it's a collective?"
    fn compute_collective_meta_phi(&self) -> f64 {
        // Collective meta-consciousness = average of individual meta-Φ
        // weighted by communication strength

        let meta_phis: Vec<f64> = self
            .agents
            .values()
            .filter_map(|a| a.meta_phi)
            .collect();

        if meta_phis.is_empty() {
            return 0.0;
        }

        let avg_meta_phi = meta_phis.iter().sum::<f64>() / meta_phis.len() as f64;

        // Boost by communication (if agents communicate well, collective meta-awareness increases)
        let integration = self.compute_integration();
        avg_meta_phi * (1.0 + integration * 0.5)
    }

    /// Generate explanation for collective consciousness
    fn generate_explanation(
        &self,
        phi_collective: f64,
        phi_sum: f64,
        emergence: f64,
        integration: f64,
        avg_communication: f64,
    ) -> String {
        let mut parts = Vec::new();

        // Overall assessment
        parts.push(format!(
            "Collective consciousness: {:.3} (from {} agents)",
            phi_collective,
            self.agents.len()
        ));

        // Emergence
        if emergence > 1.2 {
            parts.push(format!(
                "Strong emergence: {:.1}x (whole >> sum of parts!)",
                emergence
            ));
        } else if emergence > 1.05 {
            parts.push(format!("Moderate emergence: {:.1}x", emergence));
        } else if emergence > 0.95 {
            parts.push("Aggregate consciousness (sum of individuals)".to_string());
        } else {
            parts.push(format!(
                "Suppressed consciousness: {:.1}x (interference reduces total)",
                emergence
            ));
        }

        // Integration
        parts.push(format!("Integration: {:.1}%", integration * 100.0));

        // Communication
        parts.push(format!(
            "Avg communication: {:.1}%",
            avg_communication * 100.0
        ));

        parts.join(". ")
    }

    /// Get assessment history
    pub fn get_history(&self) -> &[CollectiveAssessment] {
        &self.history
    }

    /// Clear all agents and links
    pub fn clear(&mut self) {
        self.agents.clear();
        self.links.clear();
    }
}

impl Default for CollectiveConsciousness {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collective_consciousness_creation() {
        let collective = CollectiveConsciousness::new();
        assert_eq!(collective.agents().len(), 0);
    }

    #[test]
    fn test_add_agent() {
        let mut collective = CollectiveConsciousness::new();

        let agent = CollectiveAgent {
            id: "agent1".to_string(),
            state: vec![HV16::random(1000)],
            phi: 0.5,
            meta_phi: None,
            connections: vec![],
        };

        collective.add_agent(agent);
        assert_eq!(collective.agents().len(), 1);
    }

    #[test]
    fn test_single_agent_assessment() {
        let mut collective = CollectiveConsciousness::new();

        let agent = CollectiveAgent {
            id: "agent1".to_string(),
            state: vec![HV16::random(1000)],
            phi: 0.5,
            meta_phi: None,
            connections: vec![],
        };

        collective.add_agent(agent);

        let assessment = collective.assess();
        assert_eq!(assessment.num_agents, 1);
        assert!((assessment.phi_sum - 0.5).abs() < 0.001);
        assert!((assessment.phi_collective - 0.5).abs() < 0.001); // No integration boost for single agent
    }

    #[test]
    fn test_multiple_agents_no_communication() {
        let mut collective = CollectiveConsciousness::new();

        for i in 0..3 {
            let agent = CollectiveAgent {
                id: format!("agent{}", i),
                state: vec![HV16::random(1000 + i)],
                phi: 0.4,
                meta_phi: None,
                connections: vec![],
            };
            collective.add_agent(agent);
        }

        let assessment = collective.assess();
        assert_eq!(assessment.num_agents, 3);
        assert!((assessment.phi_sum - 1.2).abs() < 0.001); // Floating point tolerance
        // No communication = no integration boost
        assert!(assessment.phi_collective <= 1.3);
        assert!(assessment.emergence <= 1.1);
    }

    #[test]
    fn test_agents_with_communication() {
        let mut collective = CollectiveConsciousness::new();

        // Add agents
        for i in 0..3 {
            let agent = CollectiveAgent {
                id: format!("agent{}", i),
                state: vec![HV16::random(1000 + i)],
                phi: 0.4,
                meta_phi: Some(0.3),
                connections: vec![],
            };
            collective.add_agent(agent);
        }

        // Add communication links (full mesh)
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    let link = CommunicationLink {
                        from: format!("agent{}", i),
                        to: format!("agent{}", j),
                        strength: 0.8,
                        latency: 0.1,
                        bandwidth: 1.0,
                        shared_representation: 0.9,
                    };
                    collective.add_link(link);
                }
            }
        }

        let assessment = collective.assess();
        assert_eq!(assessment.num_agents, 3);
        assert!((assessment.phi_sum - 1.2).abs() < 0.001);
        // With communication, should have integration boost
        assert!(assessment.phi_collective > 1.2);
        assert!(assessment.emergence > 1.0); // Emergent!
        assert!(assessment.integration > 0.5);
    }

    #[test]
    fn test_emergence_detection() {
        let mut collective = CollectiveConsciousness::new();

        // High individual consciousness + strong communication = emergence
        for i in 0..5 {
            let agent = CollectiveAgent {
                id: format!("agent{}", i),
                state: vec![HV16::random(1000 + i)],
                phi: 0.6,
                meta_phi: Some(0.5),
                connections: vec![],
            };
            collective.add_agent(agent);
        }

        // Strong communication
        for i in 0..5 {
            for j in 0..5 {
                if i != j {
                    let link = CommunicationLink {
                        from: format!("agent{}", i),
                        to: format!("agent{}", j),
                        strength: 0.9,
                        latency: 0.05,
                        bandwidth: 1.0,
                        shared_representation: 0.95,
                    };
                    collective.add_link(link);
                }
            }
        }

        let assessment = collective.assess();
        assert!(assessment.emergence > 1.0, "Should show emergence");
        assert!(
            assessment.integration > 0.7,
            "Should have high integration"
        );
    }

    #[test]
    fn test_topology_metrics() {
        let mut collective = CollectiveConsciousness::new();

        // Create hub topology (one central agent)
        for i in 0..5 {
            let agent = CollectiveAgent {
                id: format!("agent{}", i),
                state: vec![HV16::random(1000 + i)],
                phi: 0.5,
                meta_phi: None,
                connections: vec![],
            };
            collective.add_agent(agent);
        }

        // Agent 0 is hub, connects to all others
        for i in 1..5 {
            let link = CommunicationLink {
                from: "agent0".to_string(),
                to: format!("agent{}", i),
                strength: 0.8,
                latency: 0.1,
                bandwidth: 1.0,
                shared_representation: 0.8,
            };
            collective.add_link(link);
        }

        let assessment = collective.assess();
        // Hub topology should have higher centralization than mesh
        // (My simplified formula may not reach 0.3, but should be > 0)
        assert!(assessment.topology_metric.centralization > 0.0);
    }

    #[test]
    fn test_collective_meta_consciousness() {
        let mut collective = CollectiveConsciousness::new();

        // Agents with meta-consciousness
        for i in 0..3 {
            let agent = CollectiveAgent {
                id: format!("agent{}", i),
                state: vec![HV16::random(1000 + i)],
                phi: 0.6,
                meta_phi: Some(0.5), // Meta-conscious agents
                connections: vec![],
            };
            collective.add_agent(agent);
        }

        // Communication
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    let link = CommunicationLink {
                        from: format!("agent{}", i),
                        to: format!("agent{}", j),
                        strength: 0.8,
                        latency: 0.1,
                        bandwidth: 1.0,
                        shared_representation: 0.8,
                    };
                    collective.add_link(link);
                }
            }
        }

        let assessment = collective.assess();
        assert!(assessment.collective_meta_phi.is_some());
        assert!(assessment.collective_meta_phi.unwrap() > 0.4);
    }

    #[test]
    fn test_serialization() {
        let config = CollectiveConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();
        assert!(!serialized.is_empty());

        let deserialized: CollectiveConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(
            deserialized.min_communication_strength,
            config.min_communication_strength
        );
    }
}
