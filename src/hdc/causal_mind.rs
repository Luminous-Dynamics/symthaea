//! # Causal Mind: The Revolutionary Core of Symthaea
//!
//! This module unifies HDC, causal reasoning, and language understanding into
//! a single coherent architecture that makes Symthaea paradigm-shifting.
//!
//! ## The Core Insight
//!
//! Traditional AI (including LLMs) conflates correlation with causation.
//! Symthaea's revolutionary approach: **make causality native to the representation**.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                         CAUSAL MIND                                  │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
//! │   │   Language   │───▶│   Causal     │───▶│   Causal     │         │
//! │   │   Parser     │    │   Encoder    │    │   Reasoner   │         │
//! │   └──────────────┘    └──────────────┘    └──────────────┘         │
//! │          │                   │                   │                  │
//! │          ▼                   ▼                   ▼                  │
//! │   ┌──────────────────────────────────────────────────────┐         │
//! │   │              Unified Causal Hypervector Space        │         │
//! │   │                                                      │         │
//! │   │   concept = meaning ⊗ ROLE ⊗ causal_structure        │         │
//! │   └──────────────────────────────────────────────────────┘         │
//! │                              │                                      │
//! │          ┌───────────────────┼───────────────────┐                 │
//! │          ▼                   ▼                   ▼                  │
//! │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
//! │   │   Causal     │    │     Φ        │    │   Active     │         │
//! │   │   Discovery  │    │   Guide      │    │   Inference  │         │
//! │   │   (Learned)  │    │              │    │              │         │
//! │   └──────────────┘    └──────────────┘    └──────────────┘         │
//! │                                                                     │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Innovations
//!
//! ### 1. Causal Structure as First-Class Citizen
//! Every concept carries its causal relationships embedded in the vector.
//!
//! ### 2. Learned Causal Discovery
//! Instead of hand-crafted heuristics (RECI, ANM, IGCI), we learn.
//!
//! ### 3. Phi-Guided Learning
//! Use integrated information (Φ) to guide representation learning.
//!
//! ### 4. Language ↔ Causality Bridge
//! Natural language → causal structure → reasoning → natural language

use super::binary_hv::HV16;
use super::causal_encoder::CausalSpace;
use std::collections::HashMap;

// =============================================================================
// CAUSAL ROLE MARKERS
// =============================================================================

/// Unique hypervectors for causal roles (generated once, reused)
#[derive(Clone, Debug)]
pub struct CausalRoleMarkers {
    /// X CAUSES Y
    pub causes: HV16,
    /// X IS_CAUSED_BY Y
    pub caused_by: HV16,
    /// X ENABLES Y (necessary but not sufficient)
    pub enables: HV16,
    /// X PREVENTS Y
    pub prevents: HV16,
    /// X CORRELATES_WITH Y (association, not causation)
    pub correlates: HV16,
    /// X BEFORE Y (temporal)
    pub before: HV16,
    /// X AFTER Y (temporal)
    pub after: HV16,
    /// X INTERVENE (do(X))
    pub intervene: HV16,
    /// Causal strength markers
    pub strength_high: HV16,    // > 0.7
    pub strength_medium: HV16,  // 0.3-0.7
    pub strength_low: HV16,     // < 0.3
}

impl CausalRoleMarkers {
    pub fn new() -> Self {
        Self {
            causes: HV16::random(1001),
            caused_by: HV16::random(1002),
            enables: HV16::random(1003),
            prevents: HV16::random(1004),
            correlates: HV16::random(1005),
            before: HV16::random(1006),
            after: HV16::random(1007),
            intervene: HV16::random(1008),
            strength_high: HV16::random(1009),
            strength_medium: HV16::random(1010),
            strength_low: HV16::random(1011),
        }
    }

    /// Get strength marker for a given value
    pub fn strength_marker(&self, strength: f64) -> &HV16 {
        if strength > 0.7 {
            &self.strength_high
        } else if strength > 0.3 {
            &self.strength_medium
        } else {
            &self.strength_low
        }
    }
}

impl Default for CausalRoleMarkers {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// CAUSAL CONCEPT: Concept with embedded causal structure
// =============================================================================

/// A concept with its full causal context embedded
#[derive(Clone, Debug)]
pub struct CausalConcept {
    /// The base semantic hypervector
    pub semantic: HV16,

    /// Human-readable name
    pub name: String,

    /// Full causal hypervector (semantic + causal structure)
    pub causal_hv: HV16,

    /// Direct causes (what causes this concept)
    pub causes: Vec<(String, f64)>,

    /// Direct effects (what this concept causes)
    pub effects: Vec<(String, f64)>,

    /// Confidence in causal structure (based on evidence)
    pub confidence: f64,
}

impl CausalConcept {
    /// Create a new causal concept from semantic representation
    pub fn new(semantic: HV16, name: String) -> Self {
        Self {
            semantic: semantic.clone(),
            name,
            causal_hv: semantic,
            causes: Vec::new(),
            effects: Vec::new(),
            confidence: 0.5,
        }
    }

    /// Add a cause and update the causal hypervector
    pub fn add_cause(&mut self, cause_name: String, cause_hv: &HV16, strength: f64, markers: &CausalRoleMarkers) {
        self.causes.push((cause_name, strength));
        let strength_marker = markers.strength_marker(strength);
        let cause_signature = cause_hv.bind(&markers.caused_by).bind(strength_marker);
        self.causal_hv = HV16::bundle(&[self.causal_hv.clone(), cause_signature]);
    }

    /// Add an effect and update the causal hypervector
    pub fn add_effect(&mut self, effect_name: String, effect_hv: &HV16, strength: f64, markers: &CausalRoleMarkers) {
        self.effects.push((effect_name, strength));
        let strength_marker = markers.strength_marker(strength);
        let effect_signature = effect_hv.bind(&markers.causes).bind(strength_marker);
        self.causal_hv = HV16::bundle(&[self.causal_hv.clone(), effect_signature]);
    }
}

// =============================================================================
// CAUSAL MIND: The unified reasoning system
// =============================================================================

/// The Causal Mind: Unified causal reasoning in hyperdimensional space
pub struct CausalMind {
    /// Causal role markers
    markers: CausalRoleMarkers,

    /// All known concepts with their causal structure
    concepts: HashMap<String, CausalConcept>,

    /// The underlying causal space for queries
    causal_space: CausalSpace,

    /// Learned causal discovery model
    causal_discovery: LearnedCausalDiscovery,

    /// Current Φ (integrated information)
    phi: f64,
}

impl CausalMind {
    /// Create a new CausalMind
    pub fn new() -> Self {
        Self {
            markers: CausalRoleMarkers::new(),
            concepts: HashMap::new(),
            causal_space: CausalSpace::new(),
            causal_discovery: LearnedCausalDiscovery::new(),
            phi: 0.0,
        }
    }

    /// Get or create a concept hypervector
    fn get_or_create_concept(&mut self, name: &str) -> HV16 {
        if let Some(concept) = self.concepts.get(name) {
            return concept.semantic.clone();
        }

        // Create new concept with deterministic seed from name
        let seed = name.bytes().fold(42u64, |acc, b| acc.wrapping_add(b as u64).wrapping_mul(31));
        let semantic = HV16::random(seed);
        let concept = CausalConcept::new(semantic.clone(), name.to_string());
        self.concepts.insert(name.to_string(), concept);
        semantic
    }

    /// Add a causal relation: cause → effect
    pub fn add_causal_link(&mut self, cause: &str, effect: &str, strength: f64) {
        let cause_hv = self.get_or_create_concept(cause);
        let effect_hv = self.get_or_create_concept(effect);

        // Add to causal space
        self.causal_space.add_causal_link(cause_hv.clone(), effect_hv.clone(), strength);

        // Update concepts' causal structure
        if let Some(cause_concept) = self.concepts.get_mut(cause) {
            cause_concept.add_effect(effect.to_string(), &effect_hv, strength, &self.markers);
        }
        if let Some(effect_concept) = self.concepts.get_mut(effect) {
            effect_concept.add_cause(cause.to_string(), &cause_hv, strength, &self.markers);
        }

        // Update Φ
        self.update_phi();
    }

    /// Learn causal structure from text (simple pattern matching)
    pub fn learn_from_text(&mut self, text: &str) {
        // Simple causal pattern extraction
        let causal_patterns = [
            ("causes", 0.8),
            ("leads to", 0.7),
            ("results in", 0.7),
            ("produces", 0.6),
            ("triggers", 0.7),
            ("induces", 0.6),
        ];

        let text_lower = text.to_lowercase();

        for (pattern, strength) in causal_patterns {
            if let Some(pos) = text_lower.find(pattern) {
                // Extract words before and after the pattern
                let before = &text[..pos].trim();
                let after_start = pos + pattern.len();
                let after = &text[after_start..].trim();

                // Get last word of 'before' as cause, first word of 'after' as effect
                if let Some(cause) = before.split_whitespace().last() {
                    if let Some(effect) = after.split_whitespace().next() {
                        // Clean up punctuation
                        let cause = cause.trim_matches(|c: char| !c.is_alphanumeric());
                        let effect = effect.trim_matches(|c: char| !c.is_alphanumeric());

                        if !cause.is_empty() && !effect.is_empty() {
                            self.add_causal_link(cause, effect, strength);
                        }
                    }
                }
            }
        }
    }

    /// Discover causal direction from observational data
    pub fn discover_causality(&self, x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
        self.causal_discovery.discover(x, y)
    }

    /// Train the causal discovery model on a single example
    pub fn train_discovery(&mut self, x: &[f64], y: &[f64], true_direction: CausalDirection) {
        self.causal_discovery.train(x, y, true_direction);
    }

    /// Query: "Why did X happen?" (find causes)
    pub fn query_why(&self, concept: &str) -> Vec<CausalExplanation> {
        let concept_hv = match self.concepts.get(concept) {
            Some(c) => &c.semantic,
            None => return Vec::new(),
        };

        let causes = self.causal_space.query_causes(concept_hv, 5);

        causes.iter().map(|r| {
            CausalExplanation {
                explanation: format!("{} was caused by factor with strength {:.2}", concept, r.strength),
                strength: r.strength,
                confidence: r.similarity as f64,
            }
        }).collect()
    }

    /// Query: "What if X happens?" (predict effects)
    pub fn query_what_if(&self, concept: &str) -> Vec<CausalPrediction> {
        let concept_hv = match self.concepts.get(concept) {
            Some(c) => &c.semantic,
            None => return Vec::new(),
        };

        let effects = self.causal_space.query_effects(concept_hv, 5);

        effects.iter().map(|r| {
            CausalPrediction {
                prediction: format!("If {} occurs, effect with strength {:.2}", concept, r.strength),
                probability: r.strength,
                confidence: r.similarity as f64,
            }
        }).collect()
    }

    /// Interventional query: do(X) - what happens if we force X?
    pub fn query_intervention(&self, concept: &str, min_strength: f64) -> Vec<CausalPrediction> {
        let concept_hv = match self.concepts.get(concept) {
            Some(c) => &c.semantic,
            None => return Vec::new(),
        };

        let effects = self.causal_space.query_intervention(concept_hv, 5, min_strength);

        effects.iter().map(|r| {
            CausalPrediction {
                prediction: format!("Intervening on {} causes effect with strength {:.2}", concept, r.strength),
                probability: r.strength,
                confidence: r.similarity as f64,
            }
        }).collect()
    }

    /// Update Phi (integrated information)
    fn update_phi(&mut self) {
        let n_concepts = self.concepts.len() as f64;
        let n_links = self.causal_space.link_count() as f64;

        if n_concepts > 0.0 {
            let connectivity = n_links / (n_concepts * n_concepts).max(1.0);
            let avg_confidence: f64 = self.concepts.values()
                .map(|c| c.confidence)
                .sum::<f64>() / n_concepts.max(1.0);

            self.phi = connectivity * avg_confidence;
        }
    }

    /// Get current Phi value
    pub fn phi(&self) -> f64 {
        self.phi
    }

    /// Get number of concepts
    pub fn concept_count(&self) -> usize {
        self.concepts.len()
    }

    /// Get number of causal links
    pub fn link_count(&self) -> usize {
        self.causal_space.link_count()
    }

    /// Get the learned causal discovery model for direct access
    pub fn causal_discovery(&self) -> &LearnedCausalDiscovery {
        &self.causal_discovery
    }

    /// Get mutable access to the learned causal discovery model
    pub fn causal_discovery_mut(&mut self) -> &mut LearnedCausalDiscovery {
        &mut self.causal_discovery
    }
}

impl Default for CausalMind {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// LEARNED CAUSAL DISCOVERY
// =============================================================================

/// Learned causal discovery model
///
/// Instead of hand-crafted heuristics (RECI, ANM, IGCI), this learns
/// to discover causal direction from data.
#[derive(Clone, Debug)]
pub struct LearnedCausalDiscovery {
    /// Learned weights
    weights: CausalDiscoveryWeights,
    /// Training history
    pub training_examples: usize,
    /// Running accuracy
    correct_predictions: usize,
}

/// Weights for combining causal discovery features
#[derive(Clone, Debug)]
pub struct CausalDiscoveryWeights {
    /// Weight for RECI-like features
    pub reci_weight: f64,
    /// Weight for IGCI-like features
    pub igci_weight: f64,
    /// Weight for ANM-like features
    pub anm_weight: f64,
    /// Weight for higher-order statistics
    pub higher_order_weight: f64,
    /// Bias term
    pub bias: f64,
}

impl LearnedCausalDiscovery {
    pub fn new() -> Self {
        Self {
            weights: CausalDiscoveryWeights {
                reci_weight: 0.4,
                igci_weight: 0.1,
                anm_weight: 0.3,
                higher_order_weight: 0.2,
                bias: 0.0,
            },
            training_examples: 0,
            correct_predictions: 0,
        }
    }

    /// Get current accuracy
    pub fn accuracy(&self) -> f64 {
        if self.training_examples == 0 {
            0.5
        } else {
            self.correct_predictions as f64 / self.training_examples as f64
        }
    }

    /// Discover causal direction using learned model
    pub fn discover(&self, x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
        let features = self.extract_features(x, y);

        let score = self.weights.reci_weight * features.reci_score
            + self.weights.igci_weight * features.igci_score
            + self.weights.anm_weight * features.anm_score
            + self.weights.higher_order_weight * features.higher_order_score
            + self.weights.bias;

        let p_forward = 1.0 / (1.0 + (-score).exp());
        let p_backward = 1.0 - p_forward;
        let confidence = (p_forward - 0.5).abs() * 2.0;

        CausalDiscoveryResult {
            direction: if p_forward > 0.5 {
                CausalDirection::Forward
            } else {
                CausalDirection::Backward
            },
            p_forward,
            p_backward,
            confidence,
            features,
        }
    }

    /// Extract features for causal discovery
    pub fn extract_features(&self, x: &[f64], y: &[f64]) -> CausalFeatures {
        let error_xy = self.regression_error(x, y);
        let error_yx = self.regression_error(y, x);
        let reci_score = (error_yx - error_xy) / (error_xy + error_yx + 1e-10);

        let igci_score = self.igci_score(x, y);
        let anm_score = self.anm_score(x, y);
        let higher_order_score = self.higher_order_score(x, y);

        CausalFeatures {
            reci_score,
            igci_score,
            anm_score,
            higher_order_score,
        }
    }

    fn regression_error(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        if n < 2.0 { return 0.0; }

        let mean_x: f64 = x.iter().sum::<f64>() / n;
        let mean_y: f64 = y.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
        }

        let slope = if var_x > 1e-10 { cov / var_x } else { 0.0 };
        let intercept = mean_y - slope * mean_x;

        let mut mse = 0.0;
        for i in 0..x.len() {
            let pred = slope * x[i] + intercept;
            let err = y[i] - pred;
            mse += err * err;
        }

        (mse / n).sqrt()
    }

    fn igci_score(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
        if n < 10 { return 0.0; }

        let mut pairs: Vec<(f64, f64)> = x.iter().zip(y.iter()).map(|(&a, &b)| (a, b)).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut slopes = Vec::with_capacity(n - 1);
        for i in 1..pairs.len() {
            let dx = pairs[i].0 - pairs[i-1].0;
            let dy = pairs[i].1 - pairs[i-1].1;
            if dx.abs() > 1e-10 {
                slopes.push((dy / dx).abs().ln().max(-10.0).min(10.0));
            }
        }

        if slopes.is_empty() { return 0.0; }
        let mean_slope: f64 = slopes.iter().sum::<f64>() / slopes.len() as f64;
        mean_slope.tanh()
    }

    fn anm_score(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        if n < 2.0 { return 0.0; }

        let mean_x: f64 = x.iter().sum::<f64>() / n;
        let mean_y: f64 = y.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
        }

        let slope = if var_x > 1e-10 { cov / var_x } else { 0.0 };
        let intercept = mean_y - slope * mean_x;

        let residuals: Vec<f64> = x.iter().zip(y.iter())
            .map(|(&xi, &yi)| yi - (slope * xi + intercept))
            .collect();

        let mean_r: f64 = residuals.iter().sum::<f64>() / n;
        let mut corr_num = 0.0;
        let mut var_r = 0.0;
        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dr = residuals[i] - mean_r;
            corr_num += dx * dr;
            var_r += dr * dr;
        }

        let corr = if var_x > 1e-10 && var_r > 1e-10 {
            corr_num / (var_x.sqrt() * var_r.sqrt())
        } else {
            0.0
        };

        -corr.abs()
    }

    fn higher_order_score(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        if n < 2.0 { return 0.0; }

        let mean_x: f64 = x.iter().sum::<f64>() / n;
        let mean_y: f64 = y.iter().sum::<f64>() / n;

        let var_x: f64 = x.iter().map(|&v| (v - mean_x).powi(2)).sum::<f64>() / n;
        let var_y: f64 = y.iter().map(|&v| (v - mean_y).powi(2)).sum::<f64>() / n;

        let std_x = var_x.sqrt().max(1e-10);
        let std_y = var_y.sqrt().max(1e-10);

        let skew_x: f64 = x.iter().map(|&v| ((v - mean_x) / std_x).powi(3)).sum::<f64>() / n;
        let skew_y: f64 = y.iter().map(|&v| ((v - mean_y) / std_y).powi(3)).sum::<f64>() / n;

        (skew_x.abs() - skew_y.abs()).tanh()
    }

    /// Train on a single example (online learning)
    pub fn train(&mut self, x: &[f64], y: &[f64], true_direction: CausalDirection) {
        let result = self.discover(x, y);

        // Track accuracy
        if result.direction == true_direction {
            self.correct_predictions += 1;
        }

        let target = match true_direction {
            CausalDirection::Forward => 1.0,
            CausalDirection::Backward => -1.0,
        };
        let prediction = if result.p_forward > 0.5 { 1.0 } else { -1.0 };
        let error = target - prediction;

        let learning_rate = 0.01 / (1.0 + self.training_examples as f64 * 0.001); // Decaying LR
        let features = result.features;

        self.weights.reci_weight += learning_rate * error * features.reci_score;
        self.weights.igci_weight += learning_rate * error * features.igci_score;
        self.weights.anm_weight += learning_rate * error * features.anm_score;
        self.weights.higher_order_weight += learning_rate * error * features.higher_order_score;
        self.weights.bias += learning_rate * error;

        self.training_examples += 1;
    }

    /// Get current weights
    pub fn weights(&self) -> &CausalDiscoveryWeights {
        &self.weights
    }
}

impl Default for LearnedCausalDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// SUPPORTING TYPES
// =============================================================================

/// Type of causal relation
#[derive(Clone, Debug, PartialEq)]
pub enum CausalRelationType {
    Causes,
    Prevents,
    Enables,
}

/// Result of causal discovery
#[derive(Clone, Debug)]
pub struct CausalDiscoveryResult {
    pub direction: CausalDirection,
    pub p_forward: f64,
    pub p_backward: f64,
    pub confidence: f64,
    pub features: CausalFeatures,
}

/// Features used for causal discovery
#[derive(Clone, Debug)]
pub struct CausalFeatures {
    pub reci_score: f64,
    pub igci_score: f64,
    pub anm_score: f64,
    pub higher_order_score: f64,
}

/// Causal direction
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CausalDirection {
    Forward,  // X → Y
    Backward, // Y → X
}

/// Explanation for a causal query
#[derive(Clone, Debug)]
pub struct CausalExplanation {
    pub explanation: String,
    pub strength: f64,
    pub confidence: f64,
}

/// Prediction from a causal query
#[derive(Clone, Debug)]
pub struct CausalPrediction {
    pub prediction: String,
    pub probability: f64,
    pub confidence: f64,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_mind_creation() {
        let mind = CausalMind::new();
        assert_eq!(mind.phi(), 0.0);
        assert_eq!(mind.concept_count(), 0);
    }

    #[test]
    fn test_add_causal_link() {
        let mut mind = CausalMind::new();
        mind.add_causal_link("rain", "wet_ground", 0.9);

        assert_eq!(mind.concept_count(), 2);
        assert_eq!(mind.link_count(), 1);
    }

    #[test]
    fn test_learn_from_text() {
        let mut mind = CausalMind::new();
        mind.learn_from_text("Smoking causes cancer");

        assert!(mind.concept_count() >= 2);
        assert!(mind.link_count() >= 1);
    }

    #[test]
    fn test_learned_causal_discovery() {
        let discovery = LearnedCausalDiscovery::new();

        // Create synthetic data: Y = 2X + noise
        let x: Vec<f64> = (0..100).map(|i| i as f64 / 10.0).collect();
        let y: Vec<f64> = x.iter().enumerate().map(|(i, &xi)| {
            2.0 * xi + (i as f64 * 0.1).sin() * 0.1
        }).collect();

        let result = discovery.discover(&x, &y);
        println!("P(X→Y): {:.3}, Confidence: {:.3}", result.p_forward, result.confidence);
    }

    #[test]
    fn test_training() {
        let mut discovery = LearnedCausalDiscovery::new();

        // Generate training data
        for seed in 0..100 {
            let x: Vec<f64> = (0..50).map(|i| (i as f64 + seed as f64) / 10.0).collect();
            let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 0.1).collect();

            discovery.train(&x, &y, CausalDirection::Forward);
        }

        println!("Training accuracy: {:.1}%", discovery.accuracy() * 100.0);
        assert!(discovery.training_examples == 100);
    }
}
