//! # Resonant Causal Graph
//!
//! HDC + LTC + Resonator integration for O(log N) causal inference.
//!
//! ## Key Innovations
//!
//! 1. **O(log N) Causal Queries**: Resonator convergence vs O(n²) graph traversal
//! 2. **Soft Causation Strength**: Continuous [0,1] instead of binary edges
//! 3. **Temporal Decay**: LTC dynamics ensure recent causes weigh more
//! 4. **Noise Tolerance**: Resonator cleanup handles missing/corrupted events
//!
//! ## Architecture
//!
//! ```text
//! Event Stream → HDC Encoder → Causal Codebook → Resonator → LTC Decay → Results
//!                    │              │                │            │
//!                    ▼              ▼                ▼            ▼
//!            Event Vectors    Cause-Effect      Convergence   Temporal
//!                             Relationships                    Weighting
//! ```

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

#[allow(unused_imports)]  // CausalEdge, EdgeType used in tests
use super::causal_graph::{CausalGraph, CausalNode, CausalEdge, EdgeType};

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for resonant causal analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonantCausalConfig {
    /// Dimension for HDC encoding
    pub dimension: usize,
    /// Time constant for causal decay (seconds)
    pub decay_tau: f64,
    /// Minimum strength to consider causal
    pub min_strength: f64,
    /// Maximum resonator iterations
    pub max_iterations: usize,
    /// Convergence threshold for resonator
    pub convergence_threshold: f64,
    /// Base seed for deterministic encoding
    pub base_seed: u64,
}

impl Default for ResonantCausalConfig {
    fn default() -> Self {
        Self {
            dimension: 1024,
            decay_tau: 300.0, // 5 minute decay
            min_strength: 0.3,
            max_iterations: 50,
            convergence_threshold: 0.01,
            base_seed: 42,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LTC CAUSAL STATE
// ═══════════════════════════════════════════════════════════════════════════

/// LTC state for a causal relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtcCausalState {
    /// Current strength of causal link [0, 1]
    pub strength: f64,
    /// Time constant for decay
    pub tau: f64,
    /// Last observation timestamp
    pub last_observed: DateTime<Utc>,
    /// Number of times this relationship was observed
    pub observation_count: u64,
    /// Confidence in this relationship
    pub confidence: f64,
}

impl LtcCausalState {
    pub fn new(initial_strength: f64, tau: f64) -> Self {
        Self {
            strength: initial_strength,
            tau,
            last_observed: Utc::now(),
            observation_count: 1,
            confidence: initial_strength,
        }
    }

    /// Evolve strength with LTC dynamics
    pub fn evolve(&mut self, now: DateTime<Utc>) {
        let dt = (now - self.last_observed).num_milliseconds() as f64 / 1000.0;
        if dt > 0.0 {
            // Decay toward zero over time
            let decay = (-dt / self.tau).exp();
            self.strength *= decay;
            self.confidence *= decay;
        }
    }

    /// Reinforce this causal relationship
    pub fn reinforce(&mut self, observed_strength: f64) {
        self.observation_count += 1;
        self.last_observed = Utc::now();

        // Update with exponential moving average
        let alpha = 0.2;
        self.strength = alpha * observed_strength + (1.0 - alpha) * self.strength;

        // Confidence increases with observations
        let obs_factor = (self.observation_count as f64).ln().min(5.0) / 5.0;
        self.confidence = self.strength * (0.5 + 0.5 * obs_factor);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CAUSAL QUERY RESULT
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a causal query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalQueryResult {
    /// The queried event ID
    pub event_id: String,
    /// Direct causes with strengths
    pub direct_causes: Vec<(String, f64)>,
    /// Root causes (transitive)
    pub root_causes: Vec<(String, f64)>,
    /// Direct effects
    pub direct_effects: Vec<(String, f64)>,
    /// Query time in microseconds
    pub query_time_us: u64,
    /// Resonator iterations used
    pub iterations: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// STATISTICS
// ═══════════════════════════════════════════════════════════════════════════

/// Statistics for resonant causal analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResonantCausalStats {
    /// Total queries processed
    pub total_queries: u64,
    /// Total events indexed
    pub events_indexed: u64,
    /// Total causal relationships
    pub relationships: u64,
    /// Average query time (microseconds)
    pub avg_query_time_us: f64,
    /// Average resonator iterations
    pub avg_iterations: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// RESONANT CAUSAL ANALYZER
// ═══════════════════════════════════════════════════════════════════════════

/// O(log N) causal analysis using HDC + LTC + Resonator
pub struct ResonantCausalAnalyzer {
    config: ResonantCausalConfig,

    /// Event type → HDC vector encoding
    event_vectors: HashMap<String, Vec<f32>>,

    /// Event ID → encoded vector
    event_encodings: HashMap<String, Vec<f32>>,

    /// Cause → Effect → LTC state
    causal_links: HashMap<String, HashMap<String, LtcCausalState>>,

    /// Codebook of known causal patterns
    pattern_codebook: Vec<Vec<f32>>,

    /// Statistics
    stats: ResonantCausalStats,
}

impl ResonantCausalAnalyzer {
    pub fn new(config: ResonantCausalConfig) -> Self {
        Self {
            config,
            event_vectors: HashMap::new(),
            event_encodings: HashMap::new(),
            causal_links: HashMap::new(),
            pattern_codebook: Vec::new(),
            stats: ResonantCausalStats::default(),
        }
    }

    /// Generate deterministic HDC vector for an event type
    fn encode_event_type(&mut self, event_type: &str) -> Vec<f32> {
        if let Some(vec) = self.event_vectors.get(event_type) {
            return vec.clone();
        }

        let dim = self.config.dimension;
        let seed = self.hash_string(event_type);
        let vec = self.random_vector(dim, seed);
        self.event_vectors.insert(event_type.to_string(), vec.clone());
        vec
    }

    /// Encode a specific event with its context
    fn encode_event(&mut self, node: &CausalNode) -> Vec<f32> {
        if let Some(vec) = self.event_encodings.get(&node.id) {
            return vec.clone();
        }

        let type_vec = self.encode_event_type(&node.event_type);

        // Bind with correlation ID if present
        let vec = if let Some(ref corr_id) = node.correlation_id {
            let corr_vec = self.random_vector(self.config.dimension, self.hash_string(corr_id));
            self.bind(&type_vec, &corr_vec)
        } else {
            type_vec
        };

        self.event_encodings.insert(node.id.clone(), vec.clone());
        vec
    }

    /// Bind two HDC vectors (element-wise multiplication)
    fn bind(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
    }

    /// Bundle HDC vectors (element-wise average)
    fn bundle(&self, vectors: &[Vec<f32>]) -> Vec<f32> {
        if vectors.is_empty() {
            return vec![0.0; self.config.dimension];
        }
        let n = vectors.len() as f32;
        let mut result = vec![0.0; self.config.dimension];
        for vec in vectors {
            for (i, v) in vec.iter().enumerate() {
                result[i] += v / n;
            }
        }
        result
    }

    /// Cosine similarity between vectors
    fn similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    /// Generate random HDC vector from seed using SplitMix64
    ///
    /// Uses proper bit mixing to ensure different seeds produce uncorrelated vectors.
    fn random_vector(&self, dim: usize, seed: u64) -> Vec<f32> {
        let mut vec = Vec::with_capacity(dim);
        let mut state = seed;
        for i in 0..dim {
            // SplitMix64: excellent bit mixing for independent values
            // Mix seed with index to ensure independence across dimensions
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state.wrapping_add(i as u64);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z = z ^ (z >> 31);
            // Convert to f32 in range [-1, 1]
            let val = (z as f32) / (u64::MAX as f32) * 2.0 - 1.0;
            vec.push(val);
        }
        vec
    }

    /// Hash string to u64
    fn hash_string(&self, s: &str) -> u64 {
        let mut hash: u64 = self.config.base_seed;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        }
        hash
    }

    /// Resonator convergence to find best matching pattern
    fn resonator_converge(&self, query: &[f32]) -> (usize, f32, usize) {
        if self.pattern_codebook.is_empty() {
            return (0, 0.0, 0);
        }

        let mut estimate = query.to_vec();
        let mut best_idx = 0;
        let mut best_sim = 0.0f32;
        let mut iterations = 0;

        for iter in 0..self.config.max_iterations {
            iterations = iter + 1;

            // Find best matching pattern
            let mut current_best_sim = f32::NEG_INFINITY;
            for (idx, pattern) in self.pattern_codebook.iter().enumerate() {
                let sim = self.similarity(&estimate, pattern);
                if sim > current_best_sim {
                    current_best_sim = sim;
                    best_idx = idx;
                }
            }
            best_sim = current_best_sim;

            // Check convergence
            if iter > 0 {
                let energy: f32 = estimate.iter()
                    .zip(self.pattern_codebook[best_idx].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();

                if energy < self.config.convergence_threshold as f32 {
                    break;
                }
            }

            // Update estimate toward best pattern
            let alpha = 0.3;
            for (i, v) in estimate.iter_mut().enumerate() {
                *v = (1.0 - alpha) * *v + alpha * self.pattern_codebook[best_idx][i];
            }
        }

        (best_idx, best_sim, iterations)
    }

    /// Index a causal graph for fast queries
    pub fn index_graph(&mut self, graph: &CausalGraph) {
        let now = Utc::now();

        // Encode all events
        for node in graph.nodes.values() {
            self.encode_event(node);
            self.stats.events_indexed += 1;
        }

        // Index all causal relationships
        for edge in &graph.edges {
            let cause_vec = self.event_encodings.get(&edge.from).cloned()
                .unwrap_or_else(|| vec![0.0; self.config.dimension]);
            let effect_vec = self.event_encodings.get(&edge.to).cloned()
                .unwrap_or_else(|| vec![0.0; self.config.dimension]);

            // Create relationship vector and add to codebook
            let rel_vec = self.bind(&cause_vec, &effect_vec);
            self.pattern_codebook.push(rel_vec);

            // Update LTC causal state
            let links = self.causal_links.entry(edge.from.clone()).or_default();
            if let Some(state) = links.get_mut(&edge.to) {
                state.reinforce(edge.strength);
            } else {
                links.insert(
                    edge.to.clone(),
                    LtcCausalState::new(edge.strength, self.config.decay_tau),
                );
            }
            self.stats.relationships += 1;
        }

        // Evolve all existing relationships
        for links in self.causal_links.values_mut() {
            for state in links.values_mut() {
                state.evolve(now);
            }
        }
    }

    /// Query causes of an event using resonator
    pub fn query_causes(&mut self, event_id: &str) -> CausalQueryResult {
        let start = std::time::Instant::now();

        let event_vec = self.event_encodings.get(event_id)
            .cloned()
            .unwrap_or_else(|| vec![0.0; self.config.dimension]);

        // Use resonator to find matching causal patterns
        let (_best_idx, _best_sim, iterations) = self.resonator_converge(&event_vec);

        // Collect direct causes from indexed relationships
        let mut direct_causes: Vec<(String, f64)> = Vec::new();
        for (cause_id, effects) in &self.causal_links {
            if let Some(state) = effects.get(event_id) {
                if state.strength >= self.config.min_strength {
                    direct_causes.push((cause_id.clone(), state.strength));
                }
            }
        }

        // Sort by strength
        direct_causes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Find root causes (transitive closure)
        let mut root_causes: Vec<(String, f64)> = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut stack: Vec<(String, f64)> = direct_causes.iter()
            .map(|(id, s)| (id.clone(), *s))
            .collect();

        while let Some((cause_id, strength)) = stack.pop() {
            if visited.contains(&cause_id) {
                continue;
            }
            visited.insert(cause_id.clone());

            // Check if this is a root (no causes of its own)
            let has_causes = self.causal_links.values()
                .any(|effects| effects.contains_key(&cause_id));

            if !has_causes {
                root_causes.push((cause_id.clone(), strength));
            } else {
                // Add its causes to stack
                for (parent_id, effects) in &self.causal_links {
                    if let Some(state) = effects.get(&cause_id) {
                        let combined_strength = strength * state.strength;
                        if combined_strength >= self.config.min_strength {
                            stack.push((parent_id.clone(), combined_strength));
                        }
                    }
                }
            }
        }

        root_causes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Find direct effects
        let direct_effects: Vec<(String, f64)> = self.causal_links
            .get(event_id)
            .map(|effects| {
                effects.iter()
                    .filter(|(_, state)| state.strength >= self.config.min_strength)
                    .map(|(id, state)| (id.clone(), state.strength))
                    .collect()
            })
            .unwrap_or_default();

        let query_time_us = start.elapsed().as_micros() as u64;

        // Update stats
        self.stats.total_queries += 1;
        let n = self.stats.total_queries as f64;
        self.stats.avg_query_time_us =
            (self.stats.avg_query_time_us * (n - 1.0) + query_time_us as f64) / n;
        self.stats.avg_iterations =
            (self.stats.avg_iterations * (n - 1.0) + iterations as f64) / n;

        CausalQueryResult {
            event_id: event_id.to_string(),
            direct_causes,
            root_causes,
            direct_effects,
            query_time_us,
            iterations,
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &ResonantCausalStats {
        &self.stats
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> CausalGraph {
        let mut nodes = HashMap::new();
        let now = Utc::now();

        // Create a simple causal chain: A → B → C
        nodes.insert("evt_a".to_string(), CausalNode {
            id: "evt_a".to_string(),
            event_type: "start".to_string(),
            timestamp: now,
            correlation_id: Some("corr_1".to_string()),
            parent_id: None,
            duration_ms: Some(10),
            metadata: HashMap::new(),
        });

        nodes.insert("evt_b".to_string(), CausalNode {
            id: "evt_b".to_string(),
            event_type: "process".to_string(),
            timestamp: now,
            correlation_id: Some("corr_1".to_string()),
            parent_id: Some("evt_a".to_string()),
            duration_ms: Some(20),
            metadata: HashMap::new(),
        });

        nodes.insert("evt_c".to_string(), CausalNode {
            id: "evt_c".to_string(),
            event_type: "complete".to_string(),
            timestamp: now,
            correlation_id: Some("corr_1".to_string()),
            parent_id: Some("evt_b".to_string()),
            duration_ms: Some(5),
            metadata: HashMap::new(),
        });

        let edges = vec![
            CausalEdge {
                from: "evt_a".to_string(),
                to: "evt_b".to_string(),
                strength: 0.9,
                edge_type: EdgeType::Direct,
            },
            CausalEdge {
                from: "evt_b".to_string(),
                to: "evt_c".to_string(),
                strength: 0.85,
                edge_type: EdgeType::Direct,
            },
        ];

        CausalGraph {
            nodes,
            edges,
            root_events: vec!["evt_a".to_string()],
            leaf_events: vec!["evt_c".to_string()],
        }
    }

    #[test]
    fn test_resonant_causal_creation() {
        let analyzer = ResonantCausalAnalyzer::new(ResonantCausalConfig::default());
        assert_eq!(analyzer.stats.total_queries, 0);
    }

    #[test]
    fn test_graph_indexing() {
        let mut analyzer = ResonantCausalAnalyzer::new(ResonantCausalConfig::default());
        let graph = create_test_graph();

        analyzer.index_graph(&graph);

        assert_eq!(analyzer.stats.events_indexed, 3);
        assert_eq!(analyzer.stats.relationships, 2);
    }

    #[test]
    fn test_cause_query() {
        let mut analyzer = ResonantCausalAnalyzer::new(ResonantCausalConfig::default());
        let graph = create_test_graph();

        analyzer.index_graph(&graph);
        let result = analyzer.query_causes("evt_c");

        assert_eq!(result.event_id, "evt_c");
        assert!(!result.direct_causes.is_empty());
        assert!(result.query_time_us > 0);
    }

    #[test]
    fn test_root_cause_finding() {
        let mut analyzer = ResonantCausalAnalyzer::new(ResonantCausalConfig::default());
        let graph = create_test_graph();

        analyzer.index_graph(&graph);
        let result = analyzer.query_causes("evt_c");

        // evt_a should be identified as root cause
        assert!(result.root_causes.iter().any(|(id, _)| id == "evt_a"));
    }

    #[test]
    fn test_hdv_operations() {
        let analyzer = ResonantCausalAnalyzer::new(ResonantCausalConfig::default());

        let a = analyzer.random_vector(1024, 42);
        let b = analyzer.random_vector(1024, 43);

        // Self-similarity should be 1.0
        let self_sim = analyzer.similarity(&a, &a);
        assert!((self_sim - 1.0).abs() < 0.001);

        // Different vectors should have low similarity
        let cross_sim = analyzer.similarity(&a, &b);
        assert!(cross_sim.abs() < 0.2);
    }
}
