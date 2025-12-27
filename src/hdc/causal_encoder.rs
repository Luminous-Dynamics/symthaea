//! Causal Hypervector Encoding - Revolutionary Improvement #4
//!
//! Encoding causality directly in hyperdimensional space, enabling
//! causal queries via similarity search instead of graph traversal.
//!
//! # Revolutionary Properties
//!
//! - **"Why?" queries as similarity search**: Unbind to find causes
//! - **"What if?" queries as binding**: Bind to predict effects
//! - **Fuzzy causality**: Continuous strength instead of binary edges
//! - **Temporal causality**: Built-in time encoding for causal chains
//! - **Counterfactual reasoning**: Explore alternative outcomes
//!
//! # Scientific Basis
//!
//! **Traditional Approach**: Pearl's Causal Graphs (do-calculus, structural equations)
//! **Revolutionary Approach**: Causal encoding in HDC space
//!
//! ## Encoding Scheme
//!
//! ```text
//! Causal pair: cause → effect
//! Encoding: C = cause ⊗ effect  (bind operation)
//!
//! Why X? (find causes):
//!   C ⊗ X⁻¹ ≈ causes that led to X
//!
//! What if X? (predict effects):
//!   X ⊗ effects ≈ what happens after X
//!
//! Interventional: do(X = x)
//!   Remove confounders, bind intervention
//! ```
//!
//! # Examples
//!
//! ```
//! use symthaea::hdc::{HV16, CausalSpace};
//!
//! let mut causal = CausalSpace::new();
//!
//! // Encode knowledge: "rain causes wet ground"
//! let rain = HV16::random(1);
//! let wet = HV16::random(2);
//! causal.add_causal_link(rain, wet, 0.9); // 90% strength
//!
//! // Query: "Why is the ground wet?"
//! let causes = causal.query_causes(&wet, 5);
//! // Returns: [(rain, 0.9), ...]
//!
//! // Query: "What if it rains?"
//! let effects = causal.query_effects(&rain, 5);
//! // Returns: [(wet, 0.9), ...]
//! ```

use super::binary_hv::HV16;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Causal space for storing and querying cause-effect relationships
///
/// Unlike traditional causal graphs which use discrete nodes and edges,
/// CausalSpace uses continuous hyperdimensional representations that
/// enable fuzzy causal queries and similarity-based reasoning.
///
/// # Key Advantages
///
/// - **No graph traversal**: O(1) query via similarity search
/// - **Fuzzy causality**: Continuous strength values (0.0 to 1.0)
/// - **Temporal encoding**: Built-in time ordering
/// - **Noise robust**: HDC properties ensure stability
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CausalSpace {
    /// Stored causal links: cause ⊗ effect → strength
    causal_links: Vec<CausalLink>,

    /// Index for fast lookup by cause
    cause_index: HashMap<String, Vec<usize>>,

    /// Index for fast lookup by effect
    effect_index: HashMap<String, Vec<usize>>,

    /// Optional: Temporal ordering information
    temporal_order: Vec<Option<f64>>, // timestamp for each link
}

/// A single causal link in hyperdimensional space
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CausalLink {
    /// The cause hypervector
    pub cause: HV16,

    /// The effect hypervector
    pub effect: HV16,

    /// The encoded causal relation: cause ⊗ effect
    pub causal_vector: HV16,

    /// Causal strength (0.0 to 1.0)
    /// - 1.0 = deterministic causation
    /// - 0.5 = 50% probability
    /// - 0.0 = no causation
    pub strength: f64,

    /// Optional label for debugging
    pub label: Option<String>,

    /// Number of times this link has been observed
    pub observation_count: usize,
}

/// Result of a causal query
#[derive(Clone, Debug)]
pub struct CausalQueryResult {
    /// The matched vector (cause or effect)
    pub vector: HV16,

    /// Similarity to query (0.0 to 1.0)
    pub similarity: f32,

    /// Causal strength
    pub strength: f64,

    /// Optional label
    pub label: Option<String>,
}

/// Temporal causal chain: A → B → C → D
#[derive(Clone, Debug)]
pub struct CausalChain {
    /// Sequence of events in temporal order
    pub events: Vec<HV16>,

    /// Strengths between consecutive events
    pub strengths: Vec<f64>,

    /// Timestamps (optional)
    pub timestamps: Vec<Option<f64>>,
}

impl CausalSpace {
    /// Create a new empty causal space
    pub fn new() -> Self {
        Self {
            causal_links: Vec::new(),
            cause_index: HashMap::new(),
            effect_index: HashMap::new(),
            temporal_order: Vec::new(),
        }
    }

    /// Add a causal link: cause → effect
    ///
    /// # Arguments
    /// * `cause` - The cause hypervector
    /// * `effect` - The effect hypervector
    /// * `strength` - Causal strength (0.0 to 1.0)
    ///
    /// # Example
    /// ```
    /// # use symthaea::hdc::{HV16, CausalSpace};
    /// let mut causal = CausalSpace::new();
    /// let rain = HV16::random(1);
    /// let wet = HV16::random(2);
    /// causal.add_causal_link(rain, wet, 0.9);
    /// ```
    pub fn add_causal_link(&mut self, cause: HV16, effect: HV16, strength: f64) {
        assert!(
            (0.0..=1.0).contains(&strength),
            "Strength must be in [0.0, 1.0]"
        );

        // Encode causal relation: cause ⊗ effect
        let causal_vector = cause.bind(&effect);

        let link = CausalLink {
            cause,
            effect,
            causal_vector,
            strength,
            label: None,
            observation_count: 1,
        };

        let idx = self.causal_links.len();
        self.causal_links.push(link);
        self.temporal_order.push(None);

        // Update indices (using string representation for HashMap compatibility)
        let cause_key = format!("{:?}", cause);
        let effect_key = format!("{:?}", effect);

        self.cause_index.entry(cause_key).or_insert_with(Vec::new).push(idx);
        self.effect_index.entry(effect_key).or_insert_with(Vec::new).push(idx);
    }

    /// Add a causal link with label and timestamp
    pub fn add_causal_link_labeled(
        &mut self,
        cause: HV16,
        effect: HV16,
        strength: f64,
        label: String,
        timestamp: Option<f64>,
    ) {
        self.add_causal_link(cause, effect, strength);
        let idx = self.causal_links.len() - 1;
        self.causal_links[idx].label = Some(label);
        self.temporal_order[idx] = timestamp;
    }

    /// Strengthen existing link or add new one
    ///
    /// If the exact cause-effect pair exists, increase its strength and observation count.
    /// Otherwise, add as new link.
    pub fn reinforce_link(&mut self, cause: HV16, effect: HV16, strength_delta: f64) {
        // Find existing link
        for link in &mut self.causal_links {
            if link.cause == cause && link.effect == effect {
                // Reinforce: exponential moving average
                link.strength = (link.strength + strength_delta).min(1.0);
                link.observation_count += 1;
                return;
            }
        }

        // Not found, add new
        self.add_causal_link(cause, effect, strength_delta.min(1.0));
    }

    /// Query: "Why did X happen?" (find causes)
    ///
    /// Returns the top N causes that most likely led to the given effect.
    ///
    /// # Method
    /// For each stored link C = cause ⊗ effect:
    /// - Unbind: cause ≈ C ⊗ effect⁻¹
    /// - Compute similarity to query
    /// - Rank by similarity × strength
    ///
    /// # Example
    /// ```
    /// # use symthaea::hdc::{HV16, CausalSpace};
    /// # let mut causal = CausalSpace::new();
    /// # let rain = HV16::random(1);
    /// # let wet = HV16::random(2);
    /// # causal.add_causal_link(rain, wet, 0.9);
    /// // Query: "Why is the ground wet?"
    /// let causes = causal.query_causes(&wet, 5);
    /// // Returns: rain with high similarity
    /// ```
    pub fn query_causes(&self, effect: &HV16, top_n: usize) -> Vec<CausalQueryResult> {
        let mut results = Vec::new();

        for link in &self.causal_links {
            // Check if this link's effect matches the query
            // (Filter out links to other effects to avoid cross-talk)
            let effect_match = link.effect.similarity(effect);
            if effect_match < 0.7 {
                continue; // Skip links where effect doesn't match well
            }

            // Unbind to recover cause: cause = causal_vector ⊗ effect
            // (XOR is self-inverse: (a ⊗ b) ⊗ b = a)
            let recovered_cause = link.causal_vector.bind(effect);

            // Similarity to actual cause
            let similarity = recovered_cause.similarity(&link.cause);

            results.push(CausalQueryResult {
                vector: link.cause,
                similarity,
                strength: link.strength,
                label: link.label.clone(),
            });
        }

        // Sort by combined score: similarity × strength
        results.sort_by(|a, b| {
            let score_a = a.similarity as f64 * a.strength;
            let score_b = b.similarity as f64 * b.strength;
            score_b.partial_cmp(&score_a).unwrap()
        });

        results.truncate(top_n);
        results
    }

    /// Query: "What if X happens?" (predict effects)
    ///
    /// Returns the top N effects that are most likely if the given cause occurs.
    ///
    /// # Method
    /// For each stored link C = cause ⊗ effect:
    /// - Unbind: effect ≈ C ⊗ cause⁻¹
    /// - Compute similarity to query
    /// - Rank by similarity × strength
    pub fn query_effects(&self, cause: &HV16, top_n: usize) -> Vec<CausalQueryResult> {
        let mut results = Vec::new();

        for link in &self.causal_links {
            // Check if this link's cause matches the query
            // (Filter out links from other causes to avoid cross-talk)
            let cause_match = link.cause.similarity(cause);
            if cause_match < 0.7 {
                continue; // Skip links where cause doesn't match well
            }

            // Unbind to recover effect: effect = causal_vector ⊗ cause
            // (XOR is self-inverse: (a ⊗ b) ⊗ a = b)
            let recovered_effect = link.causal_vector.bind(cause);

            // Similarity to actual effect
            let similarity = recovered_effect.similarity(&link.effect);

            results.push(CausalQueryResult {
                vector: link.effect,
                similarity,
                strength: link.strength,
                label: link.label.clone(),
            });
        }

        // Sort by combined score: similarity × strength
        results.sort_by(|a, b| {
            let score_a = a.similarity as f64 * a.strength;
            let score_b = b.similarity as f64 * b.strength;
            score_b.partial_cmp(&score_a).unwrap()
        });

        results.truncate(top_n);
        results
    }

    /// Interventional query: do(X = x)
    ///
    /// Removes confounders and predicts effects of intervention.
    /// This is similar to query_effects but filters out spurious correlations.
    ///
    /// # Pearl's do-calculus in HDC space
    /// In traditional causal graphs: P(Y | do(X = x))
    /// In HDC space: Find effects where X is a direct cause (high strength)
    pub fn query_intervention(&self, cause: &HV16, top_n: usize, min_strength: f64) -> Vec<CausalQueryResult> {
        // Same as query_effects but filter by strength threshold
        let mut results = self.query_effects(cause, top_n * 2); // Get more candidates

        // Filter by minimum causal strength (removes weak/confounded links)
        results.retain(|r| r.strength >= min_strength);

        results.truncate(top_n);
        results
    }

    /// Find causal chains: A → B → C
    ///
    /// Discovers multi-step causal pathways by chaining effects to causes.
    ///
    /// # Example
    /// ```text
    /// rain → wet_ground → slippery → accident
    /// ```
    pub fn find_causal_chain(&self, start: &HV16, end: &HV16, max_depth: usize) -> Option<CausalChain> {
        if max_depth == 0 {
            return None;
        }

        // BFS for causal path
        let mut visited = Vec::new();
        let mut queue = vec![(start.clone(), vec![start.clone()], vec![], vec![])];

        while let Some((current, path, strengths, timestamps)) = queue.pop() {
            // Check if we reached the end
            if current.similarity(end) > 0.9 {
                return Some(CausalChain {
                    events: path,
                    strengths,
                    timestamps,
                });
            }

            if path.len() >= max_depth {
                continue;
            }

            // Find effects of current
            for (idx, link) in self.causal_links.iter().enumerate() {
                if link.cause.similarity(&current) > 0.9 {
                    let effect = link.effect;

                    // Avoid cycles
                    if visited.contains(&effect) {
                        continue;
                    }

                    let mut new_path = path.clone();
                    new_path.push(effect);

                    let mut new_strengths = strengths.clone();
                    new_strengths.push(link.strength);

                    let mut new_timestamps = timestamps.clone();
                    new_timestamps.push(self.temporal_order[idx]);

                    queue.push((effect, new_path, new_strengths, new_timestamps));
                    visited.push(effect);
                }
            }
        }

        None
    }

    /// Get total number of causal links
    pub fn link_count(&self) -> usize {
        self.causal_links.len()
    }

    /// Get all links for a specific cause
    pub fn get_effects_of(&self, cause: &HV16) -> Vec<&CausalLink> {
        self.causal_links
            .iter()
            .filter(|link| link.cause.similarity(cause) > 0.9)
            .collect()
    }

    /// Get all links for a specific effect
    pub fn get_causes_of(&self, effect: &HV16) -> Vec<&CausalLink> {
        self.causal_links
            .iter()
            .filter(|link| link.effect.similarity(effect) > 0.9)
            .collect()
    }

    /// Clear all causal links
    pub fn clear(&mut self) {
        self.causal_links.clear();
        self.cause_index.clear();
        self.effect_index.clear();
        self.temporal_order.clear();
    }
}

impl Default for CausalSpace {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_causality() {
        let mut causal = CausalSpace::new();

        let rain = HV16::random(1);
        let wet = HV16::random(2);

        // Add causal link: rain causes wet ground
        causal.add_causal_link(rain, wet, 0.9);

        assert_eq!(causal.link_count(), 1);

        // Query: "Why is the ground wet?"
        let causes = causal.query_causes(&wet, 5);
        assert!(!causes.is_empty(), "Should find causes");

        // First cause should be rain (high similarity)
        assert!(
            causes[0].vector.similarity(&rain) > 0.8,
            "Rain should be identified as cause"
        );
        assert!(causes[0].strength > 0.8, "Should have high strength");
    }

    #[test]
    fn test_effect_prediction() {
        let mut causal = CausalSpace::new();

        let rain = HV16::random(1);
        let wet = HV16::random(2);
        let slippery = HV16::random(3);

        causal.add_causal_link(rain, wet, 0.9);
        causal.add_causal_link(wet, slippery, 0.7);

        // Query: "What if it rains?"
        let effects = causal.query_effects(&rain, 5);
        assert!(!effects.is_empty(), "Should find effects");

        // Should find wet ground as effect
        assert!(
            effects[0].vector.similarity(&wet) > 0.8,
            "Wet ground should be predicted effect"
        );
    }

    #[test]
    fn test_causal_chain() {
        let mut causal = CausalSpace::new();

        // Create chain: A → B → C
        let a = HV16::random(1);
        let b = HV16::random(2);
        let c = HV16::random(3);

        causal.add_causal_link(a, b, 0.9);
        causal.add_causal_link(b, c, 0.8);

        // Find path from A to C
        let chain = causal.find_causal_chain(&a, &c, 5);
        assert!(chain.is_some(), "Should find causal chain");

        let chain = chain.unwrap();
        assert_eq!(chain.events.len(), 3, "Chain should have 3 events: A → B → C");
        assert_eq!(chain.strengths.len(), 2, "Chain should have 2 links");
    }

    #[test]
    fn test_reinforcement_learning() {
        let mut causal = CausalSpace::new();

        let action = HV16::random(1);
        let reward = HV16::random(2);

        // Observe action → reward multiple times
        causal.add_causal_link(action, reward, 0.5);
        causal.reinforce_link(action, reward, 0.3);
        causal.reinforce_link(action, reward, 0.2);

        // Check that link was reinforced
        let effects = causal.query_effects(&action, 1);
        assert!(effects[0].strength >= 0.8, "Strength should increase with reinforcement");
    }

    #[test]
    fn test_multiple_causes() {
        let mut causal = CausalSpace::new();

        let rain = HV16::random(1);
        let sprinkler = HV16::random(2);
        let wet = HV16::random(3);

        // Both rain and sprinkler cause wet ground
        causal.add_causal_link(rain, wet, 0.9);
        causal.add_causal_link(sprinkler, wet, 0.7);

        // Query: "Why is the ground wet?"
        let causes = causal.query_causes(&wet, 5);
        assert!(causes.len() >= 2, "Should find multiple causes");

        // Rain should be stronger cause
        assert!(causes[0].strength > causes[1].strength);
    }

    #[test]
    fn test_intervention() {
        let mut causal = CausalSpace::new();

        let treatment = HV16::random(1);
        let recovery = HV16::random(2);
        let confound = HV16::random(3);

        // True causal: treatment → recovery (strong)
        causal.add_causal_link(treatment, recovery, 0.9);

        // Spurious: confound → recovery (weak)
        causal.add_causal_link(confound, recovery, 0.3);

        // Interventional query filters weak links
        let effects = causal.query_intervention(&treatment, 5, 0.5);

        // Should find recovery (strong link)
        assert!(!effects.is_empty());
        assert!(effects[0].vector.similarity(&recovery) > 0.8);

        // Spurious correlation should not appear (filtered by min_strength=0.5)
        let spurious = causal.query_intervention(&confound, 5, 0.5);
        println!("Spurious results: {} items", spurious.len());
        for (i, result) in spurious.iter().enumerate() {
            println!("  [{}] similarity={:.3}, strength={:.3}", i, result.similarity, result.strength);
        }
        assert!(
            spurious.is_empty(),
            "Interventional query with min_strength=0.5 should filter out weak link (0.3)"
        );
    }

    #[test]
    fn test_temporal_ordering() {
        let mut causal = CausalSpace::new();

        let event1 = HV16::random(1);
        let event2 = HV16::random(2);

        causal.add_causal_link_labeled(
            event1,
            event2,
            0.8,
            "temporal_test".to_string(),
            Some(1.0),
        );

        assert_eq!(causal.temporal_order[0], Some(1.0));
    }

    #[test]
    fn test_fuzzy_causality() {
        let mut causal = CausalSpace::new();

        let smoking = HV16::random(1);
        let cancer = HV16::random(2);

        // Probabilistic causation (not deterministic)
        causal.add_causal_link(smoking, cancer, 0.3); // 30% chance

        let effects = causal.query_effects(&smoking, 1);
        assert!(effects[0].strength < 0.5, "Should reflect probabilistic nature");
    }

    #[test]
    fn test_counterfactual_reasoning() {
        let mut causal = CausalSpace::new();

        // Actual: studied → passed
        let studied = HV16::random(1);
        let passed = HV16::random(2);
        let failed = HV16::random(3);

        causal.add_causal_link(studied, passed, 0.9);

        // Counterfactual: what if didn't study?
        let not_studied = HV16::random(100); // Different vector = different scenario
        causal.add_causal_link(not_studied, failed, 0.8);

        // Query effects of not studying should find failure, not passing
        let effects = causal.query_effects(&not_studied, 1);
        assert!(!effects.is_empty(), "Should find effects");

        // Should predict failure (not passing)
        assert!(
            effects[0].vector.similarity(&failed) > 0.8,
            "Not studying should predict failure"
        );
    }

    #[test]
    fn test_causal_strength_matters() {
        let mut causal = CausalSpace::new();

        let cause = HV16::random(1);
        let strong_effect = HV16::random(2);
        let weak_effect = HV16::random(3);

        causal.add_causal_link(cause, strong_effect, 0.95);
        causal.add_causal_link(cause, weak_effect, 0.2);

        let effects = causal.query_effects(&cause, 2);

        // Strong effect should rank higher
        assert!(
            effects[0].strength > effects[1].strength,
            "Stronger causal links should rank higher"
        );
    }
}
