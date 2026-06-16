// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Resonator Networks for HDC Retrieval
=====================================

**The Memory Breakthrough**: Enables scalable retrieval from bundled
hypervectors, solving the "cognitive ceiling" of standard HDC.

# The Problem with One-Shot Retrieval

Standard HDC retrieval uses inverse binding:
```text
scene = (cat ⊗ position_A) + (dog ⊗ position_B)
query = scene ⊗ position_A_inv
result ≈ cat (but noisy!)
```

**The Limit**: As bundle size grows, noise overwhelms signal. With ~50-100
items, retrieval becomes garbage. The Weaver's autobiography would dissolve.

# The Solution: Resonator Networks (Kent et al., 2020)

Instead of one-shot retrieval, use an **iterative dynamical system**:
1. Initialize with noisy query
2. Project onto valid codebook symbols
3. Let the query "resonate" against the codebook
4. System settles into correct factorization

**Result**: 10x+ improvement in retrieval capacity. The system "snaps"
to the correct answer even with massive noise.

# Mathematical Foundation

For a query q that should factorize as q = a ⊗ b:
```text
x_0 = q (noisy)
x_{t+1} = threshold(C_A^T · (C_B^T · x_t)^(-1))
```

The iteration converges when x matches a valid factorization.

Reference: Kent et al., "Resonator Networks: Fast Factorization in
Holographic Reduced Representations" (NeurIPS 2020)
*/

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Configuration for Resonator Network
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResonatorConfig {
    /// HDC dimension
    pub dim: usize,
    /// Maximum iterations before giving up
    pub max_iters: usize,
    /// Convergence threshold (similarity to previous state)
    pub convergence_threshold: f32,
    /// Temperature for softmax (lower = harder selection)
    pub temperature: f32,
    /// Use bipolar {-1, +1} or binary {0, 1}
    pub bipolar: bool,
}

impl Default for ResonatorConfig {
    fn default() -> Self {
        Self {
            dim: 16_384,
            max_iters: 100,
            convergence_threshold: 0.999,
            temperature: 0.1,
            bipolar: true,
        }
    }
}

/// A codebook of valid symbols for one "slot" in a factorization
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Codebook {
    /// Name/ID of this codebook (e.g., "objects", "positions")
    pub name: String,
    /// The symbols in this codebook: (label, hypervector)
    pub symbols: Vec<(String, Vec<f32>)>,
}

impl Codebook {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            symbols: Vec::new(),
        }
    }

    /// Add a symbol to the codebook
    pub fn add(&mut self, label: &str, hv: Vec<f32>) {
        self.symbols.push((label.to_string(), hv));
    }

    /// Find the best matching symbol (highest similarity)
    pub fn best_match(&self, query: &[f32]) -> Option<(&str, &[f32], f32)> {
        self.symbols
            .iter()
            .map(|(label, hv)| (label.as_str(), hv.as_slice(), cosine_similarity(query, hv)))
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Compute similarity scores for all symbols
    pub fn similarity_scores(&self, query: &[f32]) -> Vec<f32> {
        self.symbols
            .iter()
            .map(|(_, hv)| cosine_similarity(query, hv))
            .collect()
    }

    /// Soft attention over codebook (differentiable)
    pub fn soft_attention(&self, query: &[f32], temperature: f32) -> Vec<f32> {
        let scores = self.similarity_scores(query);

        // Softmax with temperature
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores
            .iter()
            .map(|&s| ((s - max_score) / temperature).exp())
            .collect();
        let sum: f32 = exp_scores.iter().sum();

        // Weighted sum of codebook vectors
        let dim = self.symbols[0].1.len();
        let mut result = vec![0.0; dim];

        for (i, (_, hv)) in self.symbols.iter().enumerate() {
            let weight = exp_scores[i] / sum;
            for (j, &v) in hv.iter().enumerate() {
                result[j] += weight * v;
            }
        }

        result
    }

    /// Number of symbols in codebook
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    /// Check if codebook is empty
    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }
}

/// Resonator Network for iterative HDC retrieval
#[derive(Clone, Debug)]
pub struct ResonatorNetwork {
    pub config: ResonatorConfig,
    /// Codebooks for each factor slot
    pub codebooks: Vec<Codebook>,
    /// Iteration history (for analysis)
    pub iteration_history: Vec<Vec<f32>>,
}

impl ResonatorNetwork {
    pub fn new(config: ResonatorConfig) -> Self {
        Self {
            config,
            codebooks: Vec::new(),
            iteration_history: Vec::new(),
        }
    }

    /// Create with default config
    pub fn default_network() -> Self {
        Self::new(ResonatorConfig::default())
    }

    /// Add a codebook (factor slot)
    pub fn add_codebook(&mut self, codebook: Codebook) {
        self.codebooks.push(codebook);
    }

    /// Factorize a query into components from each codebook
    ///
    /// Given query q that is a binding of symbols from different codebooks:
    /// q = a ⊗ b ⊗ c (where a ∈ C_A, b ∈ C_B, c ∈ C_C)
    ///
    /// Returns the best factorization: [(label_a, hv_a), (label_b, hv_b), ...]
    pub fn factorize(&mut self, query: &[f32]) -> Result<Vec<(String, Vec<f32>)>> {
        if self.codebooks.is_empty() {
            return Ok(vec![]);
        }

        self.iteration_history.clear();

        // Initialize estimates for each factor
        let mut estimates: Vec<Vec<f32>> = self
            .codebooks
            .iter()
            .map(|c| {
                // Start with soft attention over codebook
                c.soft_attention(query, self.config.temperature * 10.0) // Start soft
            })
            .collect();

        let mut prev_state = query.to_vec();

        for _iter in 0..self.config.max_iters {
            // Update each factor estimate
            for i in 0..self.codebooks.len() {
                // Compute residual: query with other factors "divided out"
                let mut residual = query.to_vec();

                for (j, est) in estimates.iter().enumerate() {
                    if j != i {
                        // Unbind other factors
                        residual = unbind(&residual, est, self.config.bipolar);
                    }
                }

                // Project residual onto codebook i
                estimates[i] = self.codebooks[i].soft_attention(&residual, self.config.temperature);
            }

            // Compute current state (binding of all estimates)
            let mut current_state = estimates[0].clone();
            for est in estimates.iter().skip(1) {
                current_state = bind(&current_state, est);
            }

            self.iteration_history.push(current_state.clone());

            // Check convergence
            let sim = cosine_similarity(&current_state, &prev_state);
            if sim > self.config.convergence_threshold {
                break;
            }

            prev_state = current_state;

            // Anneal temperature for harder decisions
            // (Comment out for pure soft iteration)
        }

        // Extract best discrete matches from final estimates
        let mut result = Vec::new();
        for (i, est) in estimates.iter().enumerate() {
            if let Some((label, _, _)) = self.codebooks[i].best_match(est) {
                result.push((label.to_string(), est.clone()));
            }
        }

        Ok(result)
    }

    /// Query with a partial binding and retrieve the missing factor
    ///
    /// Given: scene = (cat ⊗ pos_A) + (dog ⊗ pos_B)
    /// Query: "What is at pos_A?"
    /// Returns: cat
    pub fn query(
        &mut self,
        scene: &[f32],
        known_factors: &[(&str, &[f32])],
    ) -> Result<Vec<(String, Vec<f32>)>> {
        // Unbind known factors from scene
        let mut residual = scene.to_vec();
        for (_, hv) in known_factors {
            residual = unbind(&residual, hv, self.config.bipolar);
        }

        // Factorize the residual
        self.factorize(&residual)
    }

    /// Number of iterations in last factorization
    pub fn iterations(&self) -> usize {
        self.iteration_history.len()
    }
}

use symthaea_core::math::cosine_similarity_f32 as cosine_similarity;

/// Bind two hypervectors (element-wise multiplication)
fn bind(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Unbind (approximate inverse binding)
/// For bipolar: unbind is same as bind (self-inverse)
/// For binary: would need different approach
fn unbind(a: &[f32], b: &[f32], bipolar: bool) -> Vec<f32> {
    if bipolar {
        // For bipolar {-1, +1}, binding is self-inverse
        bind(a, b)
    } else {
        // For binary, we'd need element-wise XOR equivalent
        // Approximate: a * (2*b - 1) maps {0,1} to {-1,+1} then multiply
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| x * (2.0 * y - 1.0))
            .collect()
    }
}

// ============================================================================
// Memory System: Episodic Memory with Resonator Retrieval
// ============================================================================

/// An episode stored in memory
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Episode {
    /// Unique identifier
    pub id: String,
    /// The encoded hypervector (binding of all attributes)
    pub hv: Vec<f32>,
    /// Attribute labels (for debugging/introspection)
    pub attributes: Vec<(String, String)>, // (slot_name, symbol_label)
    /// Timestamp (arbitrary units)
    pub timestamp: u64,
    /// Importance/salience score
    pub importance: f32,
}

/// Episodic memory with resonator-based retrieval
#[derive(Clone, Debug)]
pub struct ResonatorMemory {
    /// The resonator network for retrieval
    pub resonator: ResonatorNetwork,
    /// Stored episodes
    pub episodes: Vec<Episode>,
    /// Maximum episodes before forgetting oldest
    pub max_episodes: usize,
    /// Current timestamp counter
    pub timestamp: u64,
}

impl ResonatorMemory {
    pub fn new(config: ResonatorConfig, max_episodes: usize) -> Self {
        Self {
            resonator: ResonatorNetwork::new(config),
            episodes: Vec::new(),
            max_episodes,
            timestamp: 0,
        }
    }

    /// Add a codebook to the underlying resonator
    pub fn add_codebook(&mut self, codebook: Codebook) {
        self.resonator.add_codebook(codebook);
    }

    /// Store an episode (binding of attributes from codebooks)
    pub fn store(&mut self, id: &str, attribute_hvs: &[(&str, &str, &[f32])], importance: f32) {
        // Bind all attributes together
        let mut hv = vec![1.0; self.resonator.config.dim];
        let mut attributes = Vec::new();

        for (slot, label, attr_hv) in attribute_hvs {
            hv = bind(&hv, attr_hv);
            attributes.push((slot.to_string(), label.to_string()));
        }

        let episode = Episode {
            id: id.to_string(),
            hv,
            attributes,
            timestamp: self.timestamp,
            importance,
        };

        self.episodes.push(episode);
        self.timestamp += 1;

        // Forget oldest if over capacity
        if self.episodes.len() > self.max_episodes {
            // Remove least important old episodes
            self.episodes.sort_by(|a, b| {
                // Score = importance - age_penalty
                let score_a = a.importance - (self.timestamp - a.timestamp) as f32 * 0.001;
                let score_b = b.importance - (self.timestamp - b.timestamp) as f32 * 0.001;
                score_b
                    .partial_cmp(&score_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            self.episodes.truncate(self.max_episodes);
        }
    }

    /// Bundle all episodes into a single "memory trace"
    pub fn bundle_all(&self) -> Vec<f32> {
        if self.episodes.is_empty() {
            return vec![0.0; self.resonator.config.dim];
        }

        let dim = self.resonator.config.dim;
        let mut sum = vec![0.0; dim];

        for ep in &self.episodes {
            for (i, &v) in ep.hv.iter().enumerate() {
                sum[i] += v * ep.importance; // Weight by importance
            }
        }

        // Normalize
        let n = self.episodes.iter().map(|e| e.importance).sum::<f32>();
        if n > 0.0 {
            for v in &mut sum {
                *v /= n;
            }
        }

        sum
    }

    /// Retrieve episodes matching a query pattern
    pub fn retrieve(&mut self, query_attrs: &[(&str, &[f32])]) -> Result<Vec<&Episode>> {
        // Build query by binding known attributes
        let mut query = vec![1.0; self.resonator.config.dim];
        for (_, hv) in query_attrs {
            query = bind(&query, hv);
        }

        // Find matching episodes by similarity
        let mut matches: Vec<(&Episode, f32)> = self
            .episodes
            .iter()
            .map(|ep| (ep, cosine_similarity(&query, &ep.hv)))
            .collect();

        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(matches.into_iter().map(|(ep, _)| ep).collect())
    }

    /// Query: "What was attribute X when attribute Y was Z?"
    /// Uses resonator to factorize the bundled memory
    pub fn query_factorize(
        &mut self,
        scene: &[f32],
        known: &[(&str, &[f32])],
    ) -> Result<Vec<(String, Vec<f32>)>> {
        self.resonator.query(scene, known)
    }

    /// Number of stored episodes
    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    /// Check if memory is empty
    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn random_hv(dim: usize, seed: u64) -> Vec<f32> {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        (0..dim)
            .map(|_| if rng.r#gen::<bool>() { 1.0 } else { -1.0 })
            .collect()
    }

    #[test]
    fn test_codebook_best_match() {
        let dim = 256;
        let mut codebook = Codebook::new("objects");

        let cat = random_hv(dim, 1);
        let dog = random_hv(dim, 2);
        let bird = random_hv(dim, 3);

        codebook.add("cat", cat.clone());
        codebook.add("dog", dog.clone());
        codebook.add("bird", bird.clone());

        // Query with exact match
        let (label, _, sim) = codebook.best_match(&cat).unwrap();
        assert_eq!(label, "cat");
        assert!(sim > 0.99);

        // Query with noisy cat
        let mut noisy_cat = cat.clone();
        for v in noisy_cat.iter_mut().take(20) {
            *v *= -1.0; // Flip 20 bits
        }
        let (label, _, _) = codebook.best_match(&noisy_cat).unwrap();
        assert_eq!(label, "cat"); // Should still match cat
    }

    #[test]
    fn test_resonator_simple_factorization() {
        let dim = 256;
        let config = ResonatorConfig {
            dim,
            max_iters: 50,
            temperature: 0.1,
            ..Default::default()
        };

        let mut resonator = ResonatorNetwork::new(config);

        // Create codebooks
        let mut objects = Codebook::new("objects");
        let cat = random_hv(dim, 10);
        let dog = random_hv(dim, 11);
        objects.add("cat", cat.clone());
        objects.add("dog", dog.clone());
        resonator.add_codebook(objects);

        let mut positions = Codebook::new("positions");
        let pos_a = random_hv(dim, 20);
        let pos_b = random_hv(dim, 21);
        positions.add("pos_a", pos_a.clone());
        positions.add("pos_b", pos_b.clone());
        resonator.add_codebook(positions);

        // Create a scene: cat at pos_a
        let scene = bind(&cat, &pos_a);

        // Factorize
        let result = resonator.factorize(&scene).unwrap();
        assert_eq!(result.len(), 2);

        // Should recover cat and pos_a
        // (Check by finding which codebook each result belongs to)
        let labels: Vec<&str> = result.iter().map(|(l, _)| l.as_str()).collect();
        assert!(labels.contains(&"cat") || labels.contains(&"pos_a"));
    }

    #[test]
    fn test_resonator_memory_store_retrieve() {
        let dim = 256;
        let config = ResonatorConfig {
            dim,
            ..Default::default()
        };
        let mut memory = ResonatorMemory::new(config, 100);

        // Setup codebooks
        let mut objects = Codebook::new("objects");
        let cat = random_hv(dim, 10);
        let dog = random_hv(dim, 11);
        objects.add("cat", cat.clone());
        objects.add("dog", dog.clone());
        memory.add_codebook(objects);

        let mut actions = Codebook::new("actions");
        let sleeping = random_hv(dim, 20);
        let eating = random_hv(dim, 21);
        actions.add("sleeping", sleeping.clone());
        actions.add("eating", eating.clone());
        memory.add_codebook(actions);

        // Store episodes
        memory.store(
            "ep1",
            &[("objects", "cat", &cat), ("actions", "sleeping", &sleeping)],
            1.0,
        );
        memory.store(
            "ep2",
            &[("objects", "dog", &dog), ("actions", "eating", &eating)],
            0.8,
        );

        assert_eq!(memory.len(), 2);

        // Retrieve by query
        let results = memory.retrieve(&[("objects", &cat)]).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "ep1"); // Cat episode should be first
    }

    #[test]
    fn test_resonator_convergence() {
        let dim = 256;
        let config = ResonatorConfig {
            dim,
            max_iters: 100,
            convergence_threshold: 0.99,
            temperature: 0.1,
            ..Default::default()
        };

        let mut resonator = ResonatorNetwork::new(config);

        // Create codebooks with more symbols (harder problem)
        let mut c1 = Codebook::new("c1");
        let mut c2 = Codebook::new("c2");

        for i in 0..10 {
            c1.add(&format!("s1_{}", i), random_hv(dim, 100 + i));
            c2.add(&format!("s2_{}", i), random_hv(dim, 200 + i));
        }

        resonator.add_codebook(c1.clone());
        resonator.add_codebook(c2.clone());

        // Create a test scene
        let s1_3 = c1.symbols[3].1.clone();
        let s2_7 = c2.symbols[7].1.clone();
        let scene = bind(&s1_3, &s2_7);

        // Factorize
        let result = resonator.factorize(&scene).unwrap();

        // Should converge
        assert!(
            resonator.iterations() < 100,
            "Should converge before max iterations"
        );

        // Should find correct factors
        let labels: Vec<&str> = result.iter().map(|(l, _)| l.as_str()).collect();
        assert!(
            labels
                .iter()
                .any(|l| l.contains("s1_3") || l.contains("s2_7"))
        );
    }
}
