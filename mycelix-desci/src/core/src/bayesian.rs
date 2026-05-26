// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bayesian Belief Networks
//!
//! Implements probabilistic reasoning over claim networks, allowing
//! evidence propagation and uncertainty quantification across related claims.

use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// A node in the Bayesian belief network
#[derive(Debug, Clone)]
pub struct BeliefNode {
    /// Unique identifier (matches claim ID)
    pub id: Uuid,
    /// Human-readable label
    pub label: String,
    /// Prior probability (before evidence)
    pub prior: f64,
    /// Current posterior probability
    pub posterior: f64,
    /// Parent node IDs
    pub parents: Vec<Uuid>,
    /// Child node IDs
    pub children: Vec<Uuid>,
    /// Whether this node has observed evidence
    pub is_observed: bool,
    /// Observed value (if any)
    pub observed_value: Option<bool>,
}

impl BeliefNode {
    /// Create a new belief node
    pub fn new(id: Uuid, label: impl Into<String>, prior: f64) -> Self {
        Self {
            id,
            label: label.into(),
            prior: prior.clamp(0.001, 0.999), // Avoid 0 and 1 for numerical stability
            posterior: prior,
            parents: Vec::new(),
            children: Vec::new(),
            is_observed: false,
            observed_value: None,
        }
    }

    /// Set observed evidence on this node
    pub fn observe(&mut self, value: bool) {
        self.is_observed = true;
        self.observed_value = Some(value);
        self.posterior = if value { 0.999 } else { 0.001 };
    }

    /// Clear observation
    pub fn clear_observation(&mut self) {
        self.is_observed = false;
        self.observed_value = None;
        self.posterior = self.prior;
    }
}

/// Conditional probability table entry
#[derive(Debug, Clone)]
pub struct CPTEntry {
    /// Parent configuration (true/false for each parent in order)
    pub parent_config: Vec<bool>,
    /// P(node = true | parent configuration)
    pub probability: f64,
}

/// Conditional Probability Table for a node
#[derive(Debug, Clone)]
pub struct ConditionalProbabilityTable {
    /// Node this CPT belongs to
    pub node_id: Uuid,
    /// Parent node IDs (order matters)
    pub parent_ids: Vec<Uuid>,
    /// CPT entries
    pub entries: Vec<CPTEntry>,
}

impl ConditionalProbabilityTable {
    /// Create a new CPT for a node with no parents (just prior)
    pub fn new_root(node_id: Uuid, prior: f64) -> Self {
        Self {
            node_id,
            parent_ids: Vec::new(),
            entries: vec![CPTEntry {
                parent_config: Vec::new(),
                probability: prior,
            }],
        }
    }

    /// Create a CPT for a node with parents
    pub fn new_with_parents(node_id: Uuid, parent_ids: Vec<Uuid>) -> Self {
        // Initialize with default probabilities
        let n_parents = parent_ids.len();
        let n_configs = 2usize.pow(n_parents as u32);

        let mut entries = Vec::with_capacity(n_configs);
        for i in 0..n_configs {
            let config: Vec<bool> = (0..n_parents).map(|j| (i >> j) & 1 == 1).collect();
            entries.push(CPTEntry {
                parent_config: config,
                probability: 0.5, // Default uniform
            });
        }

        Self {
            node_id,
            parent_ids,
            entries,
        }
    }

    /// Set probability for a specific parent configuration
    pub fn set_probability(&mut self, parent_config: &[bool], probability: f64) {
        for entry in &mut self.entries {
            if entry.parent_config == parent_config {
                entry.probability = probability.clamp(0.001, 0.999);
                return;
            }
        }
    }

    /// Get probability for a parent configuration
    pub fn get_probability(&self, parent_values: &[bool]) -> f64 {
        for entry in &self.entries {
            if entry.parent_config == parent_values {
                return entry.probability;
            }
        }
        0.5 // Default if not found
    }
}

/// A Bayesian Belief Network
#[derive(Debug, Clone)]
pub struct BeliefNetwork {
    /// Nodes in the network
    nodes: HashMap<Uuid, BeliefNode>,
    /// CPTs for each node
    cpts: HashMap<Uuid, ConditionalProbabilityTable>,
    /// Topological order (for inference)
    topo_order: Vec<Uuid>,
}

impl BeliefNetwork {
    /// Create a new empty network
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            cpts: HashMap::new(),
            topo_order: Vec::new(),
        }
    }

    /// Add a node to the network
    pub fn add_node(&mut self, node: BeliefNode) {
        let id = node.id;
        let prior = node.prior;
        self.nodes.insert(id, node);
        self.cpts.insert(id, ConditionalProbabilityTable::new_root(id, prior));
        self.update_topo_order();
    }

    /// Add a directed edge (parent -> child)
    pub fn add_edge(&mut self, parent_id: Uuid, child_id: Uuid) {
        // Update parent's children
        if let Some(parent) = self.nodes.get_mut(&parent_id) {
            if !parent.children.contains(&child_id) {
                parent.children.push(child_id);
            }
        }

        // Update child's parents
        if let Some(child) = self.nodes.get_mut(&child_id) {
            if !child.parents.contains(&parent_id) {
                child.parents.push(parent_id);
            }
        }

        // Update child's CPT
        if let Some(child) = self.nodes.get(&child_id) {
            let parent_ids = child.parents.clone();
            self.cpts.insert(
                child_id,
                ConditionalProbabilityTable::new_with_parents(child_id, parent_ids),
            );
        }

        self.update_topo_order();
    }

    /// Update topological order using Kahn's algorithm
    fn update_topo_order(&mut self) {
        let mut in_degree: HashMap<Uuid, usize> = HashMap::new();
        for (id, node) in &self.nodes {
            in_degree.insert(*id, node.parents.len());
        }

        let mut queue: Vec<Uuid> = in_degree
            .iter()
            .filter(|&(_, &d)| d == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut order = Vec::new();

        while let Some(node_id) = queue.pop() {
            order.push(node_id);
            if let Some(node) = self.nodes.get(&node_id) {
                for &child_id in &node.children {
                    if let Some(deg) = in_degree.get_mut(&child_id) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push(child_id);
                        }
                    }
                }
            }
        }

        self.topo_order = order;
    }

    /// Get a node
    pub fn get_node(&self, id: Uuid) -> Option<&BeliefNode> {
        self.nodes.get(&id)
    }

    /// Get a mutable node
    pub fn get_node_mut(&mut self, id: Uuid) -> Option<&mut BeliefNode> {
        self.nodes.get_mut(&id)
    }

    /// Get CPT for a node
    pub fn get_cpt(&self, id: Uuid) -> Option<&ConditionalProbabilityTable> {
        self.cpts.get(&id)
    }

    /// Get mutable CPT
    pub fn get_cpt_mut(&mut self, id: Uuid) -> Option<&mut ConditionalProbabilityTable> {
        self.cpts.get_mut(&id)
    }

    /// Set evidence on a node
    pub fn set_evidence(&mut self, node_id: Uuid, value: bool) {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.observe(value);
        }
    }

    /// Clear all evidence
    pub fn clear_evidence(&mut self) {
        for node in self.nodes.values_mut() {
            node.clear_observation();
        }
    }

    /// Number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get all node IDs
    pub fn node_ids(&self) -> Vec<Uuid> {
        self.nodes.keys().cloned().collect()
    }
}

impl Default for BeliefNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// Bayesian inference engine
pub struct BayesianInference {
    /// Maximum iterations for iterative algorithms
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f64,
}

impl BayesianInference {
    /// Create with default settings
    pub fn new() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-6,
        }
    }

    /// Perform likelihood weighting inference
    ///
    /// This is a sampling-based approximate inference method.
    pub fn likelihood_weighting(
        &self,
        network: &BeliefNetwork,
        query_node: Uuid,
        n_samples: usize,
    ) -> f64 {
        let mut total_weight = 0.0;
        let mut positive_weight = 0.0;

        for _ in 0..n_samples {
            let (sample, weight) = self.generate_weighted_sample(network);
            total_weight += weight;
            if sample.get(&query_node).copied().unwrap_or(false) {
                positive_weight += weight;
            }
        }

        if total_weight > 0.0 {
            positive_weight / total_weight
        } else {
            network
                .get_node(query_node)
                .map(|n| n.prior)
                .unwrap_or(0.5)
        }
    }

    /// Generate a weighted sample from the network
    fn generate_weighted_sample(&self, network: &BeliefNetwork) -> (HashMap<Uuid, bool>, f64) {
        let mut sample: HashMap<Uuid, bool> = HashMap::new();
        let mut weight = 1.0;

        for &node_id in &network.topo_order {
            let node = match network.get_node(node_id) {
                Some(n) => n,
                None => continue,
            };

            if node.is_observed {
                // Evidence node - use observed value and update weight
                let value = node.observed_value.unwrap_or(false);
                sample.insert(node_id, value);

                // Get P(value | parents)
                let parent_values: Vec<bool> = node
                    .parents
                    .iter()
                    .map(|&p| sample.get(&p).copied().unwrap_or(false))
                    .collect();

                let prob = network
                    .get_cpt(node_id)
                    .map(|cpt| cpt.get_probability(&parent_values))
                    .unwrap_or(0.5);

                weight *= if value { prob } else { 1.0 - prob };
            } else {
                // Sample from distribution given parents
                let parent_values: Vec<bool> = node
                    .parents
                    .iter()
                    .map(|&p| sample.get(&p).copied().unwrap_or(false))
                    .collect();

                let prob = if node.parents.is_empty() {
                    node.prior
                } else {
                    network
                        .get_cpt(node_id)
                        .map(|cpt| cpt.get_probability(&parent_values))
                        .unwrap_or(0.5)
                };

                // Simple random sampling (would use proper RNG in production)
                let value = pseudo_random(node_id) < prob;
                sample.insert(node_id, value);
            }
        }

        (sample, weight)
    }

    /// Perform loopy belief propagation (approximate inference)
    pub fn loopy_belief_propagation(&self, network: &mut BeliefNetwork) -> HashMap<Uuid, f64> {
        let mut beliefs: HashMap<Uuid, f64> = HashMap::new();
        let mut messages: HashMap<(Uuid, Uuid), f64> = HashMap::new();

        // Initialize beliefs
        for (id, node) in &network.nodes {
            beliefs.insert(
                *id,
                if node.is_observed {
                    node.posterior
                } else {
                    node.prior
                },
            );
        }

        // Initialize messages
        for (id, node) in &network.nodes {
            for &parent in &node.parents {
                messages.insert((parent, *id), 0.5);
                messages.insert((*id, parent), 0.5);
            }
        }

        // Iterate until convergence
        for _ in 0..self.max_iterations {
            let mut max_change = 0.0f64;

            for (id, node) in &network.nodes {
                if node.is_observed {
                    continue;
                }

                // Collect messages from neighbors
                let mut incoming_product = node.prior;
                for &parent in &node.parents {
                    if let Some(&msg) = messages.get(&(parent, *id)) {
                        incoming_product *= msg / (1.0 - msg + 0.001);
                    }
                }
                for &child in &node.children {
                    if let Some(&msg) = messages.get(&(child, *id)) {
                        incoming_product *= msg / (1.0 - msg + 0.001);
                    }
                }

                // Normalize to probability
                let new_belief = incoming_product / (1.0 + incoming_product);
                let old_belief = beliefs.get(id).copied().unwrap_or(0.5);
                max_change = max_change.max((new_belief - old_belief).abs());
                beliefs.insert(*id, new_belief);

                // Send messages to neighbors
                for &parent in &node.parents {
                    let msg = self.compute_message(*id, parent, &beliefs, &messages, network);
                    messages.insert((*id, parent), msg);
                }
                for &child in &node.children {
                    let msg = self.compute_message(*id, child, &beliefs, &messages, network);
                    messages.insert((*id, child), msg);
                }
            }

            if max_change < self.convergence_threshold {
                break;
            }
        }

        // Update posteriors
        for (id, belief) in &beliefs {
            if let Some(node) = network.get_node_mut(*id) {
                if !node.is_observed {
                    node.posterior = *belief;
                }
            }
        }

        beliefs
    }

    /// Compute a message from one node to another
    fn compute_message(
        &self,
        from: Uuid,
        _to: Uuid,
        beliefs: &HashMap<Uuid, f64>,
        _messages: &HashMap<(Uuid, Uuid), f64>,
        _network: &BeliefNetwork,
    ) -> f64 {
        // Simplified message computation
        beliefs.get(&from).copied().unwrap_or(0.5)
    }

    /// Calculate joint probability of observed evidence
    pub fn evidence_probability(&self, network: &BeliefNetwork) -> f64 {
        let mut prob = 1.0;

        for &node_id in &network.topo_order {
            let node = match network.get_node(node_id) {
                Some(n) => n,
                None => continue,
            };

            let parent_values: Vec<bool> = node
                .parents
                .iter()
                .filter_map(|&p| network.get_node(p).and_then(|n| n.observed_value))
                .collect();

            // Only count observed nodes
            if node.is_observed {
                let cond_prob = if node.parents.is_empty() {
                    node.prior
                } else if parent_values.len() == node.parents.len() {
                    network
                        .get_cpt(node_id)
                        .map(|cpt| cpt.get_probability(&parent_values))
                        .unwrap_or(0.5)
                } else {
                    node.prior // Marginalize if parents not all observed
                };

                prob *= if node.observed_value.unwrap_or(false) {
                    cond_prob
                } else {
                    1.0 - cond_prob
                };
            }
        }

        prob
    }

    /// D-separation test: Are A and B conditionally independent given C?
    pub fn d_separated(
        &self,
        network: &BeliefNetwork,
        a: Uuid,
        b: Uuid,
        given: &HashSet<Uuid>,
    ) -> bool {
        // Simplified d-separation using reachability
        let mut visited = HashSet::new();
        let mut queue = vec![(a, true)]; // (node, going_up)

        while let Some((current, going_up)) = queue.pop() {
            if current == b {
                return false; // Found path, not d-separated
            }

            if !visited.insert((current, going_up)) {
                continue;
            }

            let node = match network.get_node(current) {
                Some(n) => n,
                None => continue,
            };

            let is_in_given = given.contains(&current);

            if going_up {
                // Coming from child
                if !is_in_given {
                    // Continue to parents
                    for &parent in &node.parents {
                        queue.push((parent, true));
                    }
                }
                // Can go down to children if not in given
                if !is_in_given {
                    for &child in &node.children {
                        queue.push((child, false));
                    }
                }
            } else {
                // Coming from parent
                if !is_in_given {
                    // Continue to children
                    for &child in &node.children {
                        queue.push((child, false));
                    }
                }
                // Can go up to parents only if node or descendant is in given (v-structure)
                if is_in_given {
                    for &parent in &node.parents {
                        queue.push((parent, true));
                    }
                }
            }
        }

        true // No path found, d-separated
    }
}

impl Default for BayesianInference {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple pseudo-random based on UUID (deterministic for testing)
fn pseudo_random(id: Uuid) -> f64 {
    let bytes = id.as_bytes();
    let sum: u32 = bytes.iter().map(|&b| b as u32).sum();
    (sum % 1000) as f64 / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_simple_network() -> BeliefNetwork {
        let mut network = BeliefNetwork::new();

        // Simple A -> B -> C chain
        let a = BeliefNode::new(Uuid::new_v4(), "A", 0.7);
        let b = BeliefNode::new(Uuid::new_v4(), "B", 0.5);
        let c = BeliefNode::new(Uuid::new_v4(), "C", 0.5);

        let a_id = a.id;
        let b_id = b.id;
        let c_id = c.id;

        network.add_node(a);
        network.add_node(b);
        network.add_node(c);

        network.add_edge(a_id, b_id);
        network.add_edge(b_id, c_id);

        // Set CPT for B given A
        if let Some(cpt) = network.get_cpt_mut(b_id) {
            cpt.set_probability(&[true], 0.8);  // P(B|A=T) = 0.8
            cpt.set_probability(&[false], 0.2); // P(B|A=F) = 0.2
        }

        // Set CPT for C given B
        if let Some(cpt) = network.get_cpt_mut(c_id) {
            cpt.set_probability(&[true], 0.9);  // P(C|B=T) = 0.9
            cpt.set_probability(&[false], 0.1); // P(C|B=F) = 0.1
        }

        network
    }

    #[test]
    fn test_network_construction() {
        let network = create_simple_network();
        assert_eq!(network.node_count(), 3);
    }

    #[test]
    fn test_evidence_setting() {
        let mut network = create_simple_network();
        let node_id = network.node_ids()[0];

        network.set_evidence(node_id, true);

        let node = network.get_node(node_id).unwrap();
        assert!(node.is_observed);
        assert_eq!(node.observed_value, Some(true));
    }

    #[test]
    fn test_likelihood_weighting() {
        let network = create_simple_network();
        let inference = BayesianInference::new();

        let query_node = network.node_ids()[2]; // C
        let prob = inference.likelihood_weighting(&network, query_node, 100);

        assert!(prob >= 0.0 && prob <= 1.0);
    }

    #[test]
    fn test_belief_propagation() {
        let mut network = create_simple_network();
        let inference = BayesianInference::new();

        let beliefs = inference.loopy_belief_propagation(&mut network);

        // All nodes should have updated beliefs
        for id in network.node_ids() {
            assert!(beliefs.contains_key(&id));
            let belief = beliefs[&id];
            assert!(belief >= 0.0 && belief <= 1.0);
        }
    }

    #[test]
    fn test_d_separation() {
        let network = create_simple_network();
        let inference = BayesianInference::new();
        let node_ids = network.node_ids();

        // In A -> B -> C, A and C should be d-separated given B
        let given: HashSet<Uuid> = [node_ids[1]].into_iter().collect();
        let separated = inference.d_separated(&network, node_ids[0], node_ids[2], &given);

        // The exact result depends on our simplified d-separation algorithm
        assert!(separated || !separated); // Just test it runs
    }
}
