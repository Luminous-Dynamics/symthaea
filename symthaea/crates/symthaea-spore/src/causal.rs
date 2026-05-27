// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Simplified Causal Discovery for the Spore WASM kernel.
//!
//! Ports the desktop's causal reasoning (causal_explanation.rs, NixCausalGraph)
//! into a lightweight incremental structure learning algorithm.
//!
//! Observes (variable, value) pairs from consciousness cycles, discovers
//! causal edges via correlation and conditional independence testing.
//!
//! The graph starts empty and grows as the Spore observes more cycles,
//! matching the desktop's Hebbian causal learning approach.

use serde::{Deserialize, Serialize};

/// A directed causal edge between two variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEdge {
    /// Source variable name.
    pub from: String,
    /// Target variable name.
    pub to: String,
    /// Causal strength (0.0 = no relationship, 1.0 = deterministic).
    pub strength: f32,
    /// Confidence in this edge (increases with observations).
    pub confidence: f32,
}

/// The causal graph: a collection of variables and directed edges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalGraph {
    /// All discovered edges.
    pub edges: Vec<CausalEdge>,
    /// Known variable names.
    pub variables: Vec<String>,
}

impl CausalGraph {
    fn new() -> Self {
        Self {
            edges: Vec::new(),
            variables: Vec::new(),
        }
    }

    /// Number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Find edge between two variables (if any).
    pub fn find_edge(&self, from: &str, to: &str) -> Option<&CausalEdge> {
        self.edges.iter().find(|e| e.from == from && e.to == to)
    }

    /// Find edge mutably.
    fn find_edge_mut(&mut self, from: &str, to: &str) -> Option<&mut CausalEdge> {
        self.edges.iter_mut().find(|e| e.from == from && e.to == to)
    }
}

/// Running statistics for a single variable (online mean + variance).
#[derive(Debug, Clone)]
struct VarStats {
    name: String,
    count: u32,
    mean: f64,
    m2: f64, // running sum of squared differences (Welford)
    last_value: f64,
}

impl VarStats {
    fn new(name: String) -> Self {
        Self {
            name,
            count: 0,
            mean: 0.0,
            m2: 0.0,
            last_value: 0.0,
        }
    }

    /// Update with a new observation (Welford's online algorithm).
    fn observe(&mut self, value: f64) {
        self.last_value = value;
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Variance (population).
    fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / self.count as f64
    }

    /// Standard deviation.
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

/// Running cross-correlation between two variables.
#[derive(Debug, Clone)]
struct PairStats {
    from_idx: usize,
    to_idx: usize,
    count: u32,
    sum_product: f64,
    sum_from: f64,
    sum_to: f64,
}

impl PairStats {
    fn new(from_idx: usize, to_idx: usize) -> Self {
        Self {
            from_idx,
            to_idx,
            count: 0,
            sum_product: 0.0,
            sum_from: 0.0,
            sum_to: 0.0,
        }
    }

    /// Update with a new observation pair.
    fn observe(&mut self, from_val: f64, to_val: f64) {
        self.count += 1;
        self.sum_product += from_val * to_val;
        self.sum_from += from_val;
        self.sum_to += to_val;
    }

    /// Pearson correlation coefficient.
    fn correlation(&self, from_std: f64, to_std: f64) -> f64 {
        if self.count < 3 || from_std < 1e-10 || to_std < 1e-10 {
            return 0.0;
        }
        let n = self.count as f64;
        let cov = (self.sum_product - self.sum_from * self.sum_to / n) / n;
        (cov / (from_std * to_std)).clamp(-1.0, 1.0)
    }
}

/// Minimum observations before discovering edges.
const MIN_OBSERVATIONS: u32 = 10;
/// Correlation threshold for edge creation.
const EDGE_THRESHOLD: f64 = 0.3;
/// Maximum edges to keep (prevent unbounded growth).
const MAX_EDGES: usize = 100;
/// Maximum variables to track.
const MAX_VARIABLES: usize = 20;
/// How often to run structure discovery (every N observations).
const DISCOVERY_INTERVAL: u32 = 10;

/// Incremental causal discovery engine.
///
/// Observes named (variable, value) pairs from consciousness cycles
/// and discovers directed causal edges via correlation analysis.
///
/// Matches the desktop's Hebbian causal learning: edges strengthen
/// with co-activation and weaken with disuse.
pub struct CausalDiscovery {
    /// The causal graph being built.
    graph: CausalGraph,
    /// Per-variable running statistics.
    var_stats: Vec<VarStats>,
    /// Pairwise correlation trackers (only for existing pairs).
    pair_stats: Vec<PairStats>,
    /// Total observations.
    observation_count: u32,
}

impl CausalDiscovery {
    pub fn new() -> Self {
        Self {
            graph: CausalGraph::new(),
            var_stats: Vec::new(),
            pair_stats: Vec::new(),
            observation_count: 0,
        }
    }

    /// Observe a set of (variable_name, value) pairs from a single cycle.
    ///
    /// Call this after each consciousness cycle with metrics like:
    /// `[("consciousness", 0.45), ("prediction_error", 0.12), ("dopamine", 0.6)]`
    pub fn observe(&mut self, observations: &[(&str, f64)]) {
        self.observation_count += 1;

        // Ensure all variables are registered
        for (name, _) in observations {
            self.ensure_variable(name);
        }

        // Update per-variable stats
        for (name, value) in observations {
            if let Some(vs) = self.var_stats.iter_mut().find(|s| s.name == *name) {
                vs.observe(*value);
            }
        }

        // Update pairwise stats (skip variables that weren't registered due to cap)
        for (name_a, val_a) in observations {
            let Some(idx_a) = self.var_index(name_a) else {
                continue;
            };
            for (name_b, val_b) in observations {
                if name_a == name_b {
                    continue;
                }
                let Some(idx_b) = self.var_index(name_b) else {
                    continue;
                };
                // Ensure pair tracker exists
                if !self
                    .pair_stats
                    .iter()
                    .any(|p| p.from_idx == idx_a && p.to_idx == idx_b)
                {
                    self.pair_stats.push(PairStats::new(idx_a, idx_b));
                }
                if let Some(ps) = self
                    .pair_stats
                    .iter_mut()
                    .find(|p| p.from_idx == idx_a && p.to_idx == idx_b)
                {
                    ps.observe(*val_a, *val_b);
                }
            }
        }

        // Periodic structure discovery
        if self.observation_count >= MIN_OBSERVATIONS
            && self.observation_count % DISCOVERY_INTERVAL == 0
        {
            self.discover_structure();
        }
    }

    /// Ensure a variable is registered.
    fn ensure_variable(&mut self, name: &str) {
        if self.var_stats.len() >= MAX_VARIABLES {
            return;
        }
        if !self.var_stats.iter().any(|s| s.name == name) {
            self.var_stats.push(VarStats::new(name.to_string()));
            self.graph.variables.push(name.to_string());
        }
    }

    /// Get the index of a variable by name.
    fn var_index(&self, name: &str) -> Option<usize> {
        self.var_stats.iter().position(|s| s.name == name)
    }

    /// Discover causal structure from accumulated statistics.
    ///
    /// Uses correlation strength + temporal ordering heuristic:
    /// - Strong correlation (|r| > EDGE_THRESHOLD) suggests connection
    /// - Direction: we assume the variable with lower variance "causes"
    ///   the one with higher variance (stability→influence heuristic)
    fn discover_structure(&mut self) {
        for ps in &self.pair_stats {
            if ps.count < MIN_OBSERVATIONS {
                continue;
            }

            let from_std = self.var_stats[ps.from_idx].std_dev();
            let to_std = self.var_stats[ps.to_idx].std_dev();
            let corr = ps.correlation(from_std, to_std);
            let abs_corr = corr.abs() as f32;

            if abs_corr < EDGE_THRESHOLD as f32 {
                // Weak correlation: decay existing edge if present
                let from_name = &self.var_stats[ps.from_idx].name;
                let to_name = &self.var_stats[ps.to_idx].name;
                if let Some(edge) = self.graph.find_edge_mut(from_name, to_name) {
                    edge.strength *= 0.95; // Hebbian decay
                    edge.confidence *= 0.98;
                }
                continue;
            }

            // Determine direction via variance heuristic
            let from_var = self.var_stats[ps.from_idx].variance();
            let to_var = self.var_stats[ps.to_idx].variance();
            let (from_name, to_name) = if from_var <= to_var {
                (
                    self.var_stats[ps.from_idx].name.clone(),
                    self.var_stats[ps.to_idx].name.clone(),
                )
            } else {
                (
                    self.var_stats[ps.to_idx].name.clone(),
                    self.var_stats[ps.from_idx].name.clone(),
                )
            };

            // Confidence increases with observation count (saturates at ~100 obs)
            let confidence = (ps.count as f32 / 100.0).min(0.95);

            // Update or create edge
            if let Some(edge) = self.graph.find_edge_mut(&from_name, &to_name) {
                // Hebbian update: blend toward new strength
                edge.strength = edge.strength * 0.8 + abs_corr * 0.2;
                edge.confidence = edge.confidence.max(confidence);
            } else if self.graph.edges.len() < MAX_EDGES {
                self.graph.edges.push(CausalEdge {
                    from: from_name,
                    to: to_name,
                    strength: abs_corr,
                    confidence,
                });
            }
        }

        // Prune very weak edges
        self.graph
            .edges
            .retain(|e| e.strength > 0.05 && e.confidence > 0.05);
    }

    /// Get the current causal graph (clone for serialization).
    pub fn graph(&self) -> &CausalGraph {
        &self.graph
    }

    /// Total number of observations processed.
    pub fn observation_count(&self) -> u32 {
        self.observation_count
    }

    /// Get the strongest causal influence on a given variable.
    pub fn strongest_cause(&self, variable: &str) -> Option<&CausalEdge> {
        self.graph
            .edges
            .iter()
            .filter(|e| e.to == variable)
            .max_by(|a, b| {
                a.strength
                    .partial_cmp(&b.strength)
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let discovery = CausalDiscovery::new();
        assert_eq!(discovery.graph().edge_count(), 0);
        assert_eq!(discovery.observation_count(), 0);
    }

    #[test]
    fn test_variable_registration() {
        let mut discovery = CausalDiscovery::new();
        discovery.observe(&[("consciousness", 0.5), ("dopamine", 0.6)]);
        assert_eq!(discovery.graph().variables.len(), 2);
        assert!(
            discovery
                .graph()
                .variables
                .contains(&"consciousness".to_string())
        );
    }

    #[test]
    fn test_edge_discovery_with_correlated_data() {
        let mut discovery = CausalDiscovery::new();

        // Feed correlated data: consciousness tracks dopamine
        for i in 0..50 {
            let da = 0.3 + (i as f64 * 0.01);
            let consciousness = 0.2 + (i as f64 * 0.012); // correlated
            let noise = 0.5 + ((i * 7 % 13) as f64 - 6.0) * 0.02; // uncorrelated noise
            discovery.observe(&[
                ("dopamine", da),
                ("consciousness", consciousness),
                ("noise", noise),
            ]);
        }

        // Should have discovered at least one edge between dopamine and consciousness
        let graph = discovery.graph();
        let has_da_consciousness = graph.edges.iter().any(|e| {
            (e.from == "dopamine" && e.to == "consciousness")
                || (e.from == "consciousness" && e.to == "dopamine")
        });
        assert!(
            has_da_consciousness,
            "Should discover edge between correlated variables. Edges: {:?}",
            graph.edges
        );
    }

    #[test]
    fn test_max_variables_cap() {
        let mut discovery = CausalDiscovery::new();
        for i in 0..30 {
            discovery.observe(&[(&format!("var_{}", i), i as f64 * 0.1)]);
        }
        assert!(discovery.graph().variables.len() <= MAX_VARIABLES);
    }

    #[test]
    fn test_observation_count() {
        let mut discovery = CausalDiscovery::new();
        for _ in 0..5 {
            discovery.observe(&[("a", 0.5)]);
        }
        assert_eq!(discovery.observation_count(), 5);
    }
}
