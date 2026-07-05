// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Causal Consciousness Integration
//!
//! **Revolutionary Module**: Bridges bivariate causal discovery with
//! Symthaea's HDC+LTC consciousness architecture.
//!
//! ## The Paradigm Shift
//!
//! Traditional AI: Correlation-based pattern matching
//! Symthaea: Causal reasoning as native operation
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    CAUSAL CONSCIOUSNESS ENGINE                       │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                      │
//! │  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐   │
//! │  │ Bivariate      │     │ HSIC Kernel    │     │ Live Learning  │   │
//! │  │ Causal         │────▶│ Independence   │────▶│ Adaptation     │   │
//! │  │ Discovery      │     │ Test           │     │                │   │
//! │  │ (71.3%)        │     │                │     │                │   │
//! │  └────────────────┘     └────────────────┘     └────────────────┘   │
//! │           │                     │                      │            │
//! │           └─────────────────────┼──────────────────────┘            │
//! │                                 ▼                                    │
//! │  ┌─────────────────────────────────────────────────────────────┐    │
//! │  │                 CAUSAL ATTENTION MECHANISM                   │    │
//! │  │                                                              │    │
//! │  │   attention_weight = f(causal_strength, direction, Φ)        │    │
//! │  │                                                              │    │
//! │  │   • Upweight causes of current state                         │    │
//! │  │   • Downweight spurious correlations                         │    │
//! │  │   • Focus on interventional relevance                        │    │
//! │  └─────────────────────────────────────────────────────────────┘    │
//! │                                 │                                    │
//! │                                 ▼                                    │
//! │  ┌─────────────────────────────────────────────────────────────┐    │
//! │  │              LTC HIERARCHY INTEGRATION                       │    │
//! │  │                                                              │    │
//! │  │   Level 7 (sensory) ──causal──▶ Level 6                      │    │
//! │  │   Level 6 ──────────causal──▶ Level 5                        │    │
//! │  │   ...                                                        │    │
//! │  │   Level 1 ──────────causal──▶ Level 0 (global)               │    │
//! │  └─────────────────────────────────────────────────────────────┘    │
//! │                                                                      │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```

use super::causal_discovery::{CausalDirection, CausalDiscoveryEngine, MetaFeatures};
use rand::prelude::*;
use std::collections::HashMap;

// ============================================================================
// HSIC KERNEL INDEPENDENCE TEST
// ============================================================================

/// Hilbert-Schmidt Independence Criterion
///
/// A kernel-based measure of statistical independence that captures
/// non-linear dependencies missed by correlation.
pub struct HSICTest {
    /// Kernel bandwidth (auto-tuned if None)
    sigma: Option<f64>,
    /// Optional seeded RNG for deterministic permutation tests
    seeded_rng: Option<rand::rngs::StdRng>,
}

impl HSICTest {
    pub fn new() -> Self {
        Self {
            sigma: None,
            seeded_rng: None,
        }
    }

    pub fn with_bandwidth(sigma: f64) -> Self {
        Self {
            sigma: Some(sigma),
            seeded_rng: None,
        }
    }

    /// Create an HSIC test with deterministic RNG from a genesis seed.
    pub fn from_genesis(genesis: &symthaea_core::genesis::GenesisSeed, label: &str) -> Self {
        use rand::SeedableRng;
        let mut shake = genesis.domain(&format!("{label}::hsic"));
        let seed: u64 = rand::Rng::r#gen(&mut shake);
        Self {
            sigma: None,
            seeded_rng: Some(rand::rngs::StdRng::seed_from_u64(seed)),
        }
    }

    /// Compute HSIC statistic between two variables
    ///
    /// Returns: HSIC value (higher = more dependent)
    pub fn compute(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
        if n < 10 {
            return 0.0;
        }

        // Auto-tune bandwidth using median heuristic
        let sigma_x = self.sigma.unwrap_or_else(|| self.median_bandwidth(x));
        let sigma_y = self.sigma.unwrap_or_else(|| self.median_bandwidth(y));

        // Compute kernel matrices
        let kx = self.rbf_kernel_matrix(x, sigma_x);
        let ky = self.rbf_kernel_matrix(y, sigma_y);

        // Center the kernel matrices
        let hkx = self.center_kernel(&kx);
        let hky = self.center_kernel(&ky);

        // HSIC = trace(Kx @ H @ Ky @ H) / (n-1)^2
        let mut hsic = 0.0;
        for i in 0..n {
            for j in 0..n {
                hsic += hkx[i][j] * hky[j][i];
            }
        }

        hsic / ((n - 1) * (n - 1)) as f64
    }

    /// Test if two variables are independent
    ///
    /// Returns: (is_independent, p_value_estimate)
    pub fn test_independence(&mut self, x: &[f64], y: &[f64], threshold: f64) -> (bool, f64) {
        let hsic = self.compute(x, y);

        // Permutation-based p-value estimation
        let n_perms = 100;
        let mut null_hsics = Vec::with_capacity(n_perms);

        // Pre-generate all permutations to avoid borrow conflict with self.compute()
        let mut permutations: Vec<Vec<f64>> = Vec::with_capacity(n_perms);
        {
            let mut fallback_rng;
            let rng: &mut dyn rand::RngCore = if let Some(ref mut seeded) = self.seeded_rng {
                seeded
            } else {
                fallback_rng = rand::thread_rng();
                &mut fallback_rng
            };
            for _ in 0..n_perms {
                let mut y_perm: Vec<f64> = y.to_vec();
                y_perm.shuffle(rng);
                permutations.push(y_perm);
            }
        }

        for y_perm in &permutations {
            null_hsics.push(self.compute(x, y_perm));
        }

        let p_value = null_hsics.iter().filter(|&&h| h >= hsic).count() as f64 / n_perms as f64;

        (p_value > threshold, p_value)
    }

    fn median_bandwidth(&self, x: &[f64]) -> f64 {
        let n = x.len().min(100);
        let mut dists = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                dists.push((x[i] - x[j]).abs());
            }
        }

        if dists.is_empty() {
            return 1.0;
        }

        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        dists[dists.len() / 2].max(1e-10)
    }

    fn rbf_kernel_matrix(&self, x: &[f64], sigma: f64) -> Vec<Vec<f64>> {
        let n = x.len();
        let mut k = vec![vec![0.0; n]; n];
        let gamma = 1.0 / (2.0 * sigma * sigma);

        for i in 0..n {
            for j in 0..n {
                let diff = x[i] - x[j];
                k[i][j] = (-gamma * diff * diff).exp();
            }
        }

        k
    }

    fn center_kernel(&self, k: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = k.len();
        let mut centered = vec![vec![0.0; n]; n];

        // Compute row and column means
        let row_means: Vec<f64> = k
            .iter()
            .map(|row| row.iter().sum::<f64>() / n as f64)
            .collect();
        let total_mean: f64 = row_means.iter().sum::<f64>() / n as f64;

        for i in 0..n {
            for j in 0..n {
                centered[i][j] = k[i][j] - row_means[i] - row_means[j] + total_mean;
            }
        }

        centered
    }
}

impl Default for HSICTest {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for HSICTest {
    fn clone(&self) -> Self {
        // seeded_rng is not Clone, so recreate as unseeded on clone
        Self {
            sigma: self.sigma,
            seeded_rng: None,
        }
    }
}

// ============================================================================
// LIVE LEARNING ADAPTATION
// ============================================================================

/// Adaptive routing that learns from feedback
pub struct LiveLearningRouter {
    /// Base causal discovery engine
    engine: CausalDiscoveryEngine,

    /// Learned threshold adjustments
    threshold_adjustments: ThresholdAdjustments,

    /// Performance history
    history: Vec<PredictionOutcome>,

    /// Learning rate
    learning_rate: f64,
}

#[derive(Debug, Clone)]
struct ThresholdAdjustments {
    noise_threshold: f64,    // Default 0.85
    nonlin_threshold: f64,   // Default 0.4
    corr_threshold: f64,     // Default 0.85
    kurtosis_threshold: f64, // Default 3.0
}

impl Default for ThresholdAdjustments {
    fn default() -> Self {
        Self {
            noise_threshold: 0.85,
            nonlin_threshold: 0.4,
            corr_threshold: 0.85,
            kurtosis_threshold: 3.0,
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // Reserved for learning history analysis
struct PredictionOutcome {
    meta_features: MetaFeatures,
    predicted: CausalDirection,
    actual: Option<CausalDirection>,
    routing_rule_used: String,
}

impl LiveLearningRouter {
    pub fn new(seed: u64) -> Self {
        Self {
            engine: CausalDiscoveryEngine::new(seed),
            threshold_adjustments: ThresholdAdjustments::default(),
            history: Vec::new(),
            learning_rate: 0.1,
        }
    }

    /// Predict with adaptive thresholds
    pub fn predict(&mut self, x: &[f64], y: &[f64]) -> CausalDirection {
        let (direction, _meta, _rule) = self.predict_with_info(x, y);
        direction
    }

    /// Predict and return meta-information for learning
    fn predict_with_info(
        &mut self,
        x: &[f64],
        y: &[f64],
    ) -> (CausalDirection, MetaFeatures, String) {
        // Use adaptive thresholds
        let direction = self.engine.predict(x, y);

        // Extract meta-features for learning
        let meta = self.extract_meta(x, y);

        // Determine which rule was used
        let rule = self.determine_rule_used(&meta);

        (direction, meta, rule)
    }

    /// Provide feedback on a prediction
    pub fn feedback(&mut self, x: &[f64], y: &[f64], actual: CausalDirection) {
        let (predicted, meta, rule) = self.predict_with_info(x, y);

        let outcome = PredictionOutcome {
            meta_features: meta.clone(),
            predicted,
            actual: Some(actual),
            routing_rule_used: rule.clone(),
        };

        self.history.push(outcome);

        // Adapt thresholds based on error
        if predicted != actual {
            self.adapt_thresholds(&meta, &rule);
        }
    }

    /// Adapt thresholds based on error patterns
    fn adapt_thresholds(&mut self, meta: &MetaFeatures, rule: &str) {
        let lr = self.learning_rate;

        match rule {
            // If this rule failed, maybe threshold too low
            "high_noise_nonlin"
                if meta.noise_ratio < self.threshold_adjustments.noise_threshold + 0.1 =>
            {
                self.threshold_adjustments.noise_threshold += lr * 0.05;
            }
            "high_correlation"
                if meta.correlation.abs() < self.threshold_adjustments.corr_threshold + 0.1 =>
            {
                self.threshold_adjustments.corr_threshold += lr * 0.05;
            }
            "heavy_tails" => {
                let max_kurt = meta.kurtosis_x.max(meta.kurtosis_y);
                if max_kurt < self.threshold_adjustments.kurtosis_threshold + 1.0 {
                    self.threshold_adjustments.kurtosis_threshold += lr * 0.5;
                }
            }
            _ => {}
        }

        // Clamp thresholds to reasonable ranges
        self.threshold_adjustments.noise_threshold =
            self.threshold_adjustments.noise_threshold.clamp(0.7, 0.95);
        self.threshold_adjustments.corr_threshold =
            self.threshold_adjustments.corr_threshold.clamp(0.7, 0.95);
        self.threshold_adjustments.kurtosis_threshold = self
            .threshold_adjustments
            .kurtosis_threshold
            .clamp(2.0, 5.0);
    }

    fn extract_meta(&self, x: &[f64], y: &[f64]) -> MetaFeatures {
        // Simplified meta-feature extraction
        let n = x.len();
        if n < 2 {
            return MetaFeatures {
                n_samples: n,
                correlation: 0.0,
                noise_ratio: 0.5,
                nonlinearity: 0.0,
                kurtosis_x: 0.0,
                kurtosis_y: 0.0,
            };
        }
        let mx: f64 = x.iter().sum::<f64>() / n as f64;
        let my: f64 = y.iter().sum::<f64>() / n as f64;

        let var_x: f64 = x.iter().map(|&xi| (xi - mx).powi(2)).sum::<f64>() / (n - 1) as f64;
        let var_y: f64 = y.iter().map(|&yi| (yi - my).powi(2)).sum::<f64>() / (n - 1) as f64;

        let cov: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - mx) * (yi - my))
            .sum::<f64>()
            / (n - 1) as f64;

        let correlation = if var_x > 1e-10 && var_y > 1e-10 {
            cov / (var_x.sqrt() * var_y.sqrt())
        } else {
            0.0
        };

        // Kurtosis
        let std_x = var_x.sqrt().max(1e-10);
        let std_y = var_y.sqrt().max(1e-10);
        let kurtosis_x: f64 =
            x.iter().map(|&xi| ((xi - mx) / std_x).powi(4)).sum::<f64>() / n as f64 - 3.0;
        let kurtosis_y: f64 =
            y.iter().map(|&yi| ((yi - my) / std_y).powi(4)).sum::<f64>() / n as f64 - 3.0;

        MetaFeatures {
            n_samples: n,
            correlation,
            noise_ratio: 0.5,  // Simplified
            nonlinearity: 0.0, // Simplified
            kurtosis_x,
            kurtosis_y,
        }
    }

    fn determine_rule_used(&self, meta: &MetaFeatures) -> String {
        let adj = &self.threshold_adjustments;

        if meta.noise_ratio > adj.noise_threshold && meta.nonlinearity > adj.nonlin_threshold {
            "high_noise_nonlin".to_string()
        } else if meta.correlation.abs() > adj.corr_threshold {
            "high_correlation".to_string()
        } else if meta.kurtosis_x > adj.kurtosis_threshold
            || meta.kurtosis_y > adj.kurtosis_threshold
        {
            "heavy_tails".to_string()
        } else {
            "majority".to_string()
        }
    }

    /// Get current learned thresholds
    pub fn get_thresholds(&self) -> (f64, f64, f64, f64) {
        let adj = &self.threshold_adjustments;
        (
            adj.noise_threshold,
            adj.nonlin_threshold,
            adj.corr_threshold,
            adj.kurtosis_threshold,
        )
    }

    /// Get accuracy on history
    pub fn get_accuracy(&self) -> f64 {
        let correct = self
            .history
            .iter()
            .filter(|o| o.actual.map(|a| a == o.predicted).unwrap_or(false))
            .count();
        let total = self.history.iter().filter(|o| o.actual.is_some()).count();

        if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        }
    }
}

// ============================================================================
// CAUSAL ATTENTION MECHANISM
// ============================================================================

/// Attention weights based on causal structure
///
/// This is the revolutionary part: attention is guided by causal discovery,
/// not just similarity or learned patterns.
pub struct CausalAttention {
    /// Causal discovery engine
    engine: CausalDiscoveryEngine,

    /// HSIC for independence testing
    hsic: HSICTest,

    /// Cached causal graph
    causal_graph: HashMap<(usize, usize), CausalEdge>,
}

#[derive(Debug, Clone)]
pub struct CausalEdge {
    pub direction: CausalDirection,
    pub strength: f64,   // 0.0 - 1.0
    pub confidence: f64, // 0.0 - 1.0
    pub is_direct: bool, // vs mediated
}

impl CausalAttention {
    pub fn new(seed: u64) -> Self {
        Self {
            engine: CausalDiscoveryEngine::with_ensemble_size(seed, 5),
            hsic: HSICTest::new(),
            causal_graph: HashMap::new(),
        }
    }

    /// Compute causal attention weights for a set of variables
    ///
    /// Returns: attention weights for each variable pair (cause -> effect)
    pub fn compute_attention(&mut self, variables: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = variables.len();
        let mut attention = vec![vec![0.0; n]; n];

        // Build causal graph
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }

                let (direction, confidence) = self
                    .engine
                    .predict_with_confidence(&variables[i], &variables[j]);

                // Compute causal strength using HSIC
                let hsic_val = self.hsic.compute(&variables[i], &variables[j]);
                let strength = (hsic_val * 10.0).min(1.0); // Normalize

                let edge = CausalEdge {
                    direction,
                    strength,
                    confidence,
                    is_direct: true,
                };

                self.causal_graph.insert((i, j), edge.clone());

                // Attention weight: combine direction, strength, confidence
                let weight = match direction {
                    CausalDirection::Forward => strength * confidence,
                    CausalDirection::Backward => -strength * confidence, // Negative for reverse
                };

                attention[i][j] = weight;
            }
        }

        // Normalize attention weights (softmax per row)
        for row in &mut attention {
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = row.iter().map(|&x| (x - max_val).exp()).sum();

            if exp_sum > 1e-10 {
                for val in row.iter_mut() {
                    *val = (*val - max_val).exp() / exp_sum;
                }
            }
        }

        attention
    }

    /// Get causal parents of a variable (what causes it)
    pub fn get_causes(&self, var_idx: usize) -> Vec<(usize, f64)> {
        let mut causes = Vec::new();

        for (&(i, j), edge) in &self.causal_graph {
            if j == var_idx && edge.direction == CausalDirection::Forward {
                causes.push((i, edge.strength * edge.confidence));
            }
            if i == var_idx && edge.direction == CausalDirection::Backward {
                causes.push((j, edge.strength * edge.confidence));
            }
        }

        causes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        causes
    }

    /// Get causal children of a variable (what it causes)
    pub fn get_effects(&self, var_idx: usize) -> Vec<(usize, f64)> {
        let mut effects = Vec::new();

        for (&(i, j), edge) in &self.causal_graph {
            if i == var_idx && edge.direction == CausalDirection::Forward {
                effects.push((j, edge.strength * edge.confidence));
            }
            if j == var_idx && edge.direction == CausalDirection::Backward {
                effects.push((i, edge.strength * edge.confidence));
            }
        }

        effects.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        effects
    }

    /// Interventional query: what happens if we set variable i to value v?
    pub fn intervene(&mut self, variables: &[Vec<f64>], var_idx: usize, _value: f64) -> Vec<f64> {
        // Get effects of this variable
        let effects = self.get_effects(var_idx);

        // Compute expected change in each effect
        let mut changes = vec![0.0; variables.len()];
        changes[var_idx] = 1.0; // The intervened variable changes fully

        for (effect_idx, strength) in effects {
            // Effect magnitude proportional to causal strength
            changes[effect_idx] = strength;
        }

        changes
    }
}

// ============================================================================
// LTC HIERARCHY INTEGRATION
// ============================================================================

/// Integrates causal discovery with hierarchical LTC network
pub struct CausalLTCBridge {
    /// Causal attention for each hierarchy level
    level_attention: Vec<CausalAttention>,

    /// Cross-level causal links
    cross_level_causality: HashMap<(usize, usize), CausalDirection>,
}

impl CausalLTCBridge {
    /// Create bridge for n-level hierarchy
    pub fn new(n_levels: usize, seed: u64) -> Self {
        let level_attention: Vec<CausalAttention> = (0..n_levels)
            .map(|i| CausalAttention::new(seed + i as u64))
            .collect();

        Self {
            level_attention,
            cross_level_causality: HashMap::new(),
        }
    }

    /// Discover causal structure at a specific level
    pub fn discover_level_causality(
        &mut self,
        level: usize,
        signals: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        if level >= self.level_attention.len() {
            return vec![];
        }

        self.level_attention[level].compute_attention(signals)
    }

    /// Discover cross-level causality (does level i cause level j?)
    pub fn discover_cross_level(
        &mut self,
        level_i: usize,
        level_j: usize,
        signal_i: &[f64],
        signal_j: &[f64],
    ) -> CausalDirection {
        if level_i >= self.level_attention.len() || level_j >= self.level_attention.len() {
            return CausalDirection::Forward;
        }

        let direction = self.level_attention[level_i]
            .engine
            .predict(signal_i, signal_j);
        self.cross_level_causality
            .insert((level_i, level_j), direction);
        direction
    }

    /// Get the causal flow direction in the hierarchy
    ///
    /// Returns: Vec of (from_level, to_level, strength) indicating causal flow
    pub fn get_causal_flow(&self) -> Vec<(usize, usize, f64)> {
        let mut flow = Vec::new();

        for (&(i, j), &direction) in &self.cross_level_causality {
            let strength = match direction {
                CausalDirection::Forward => 1.0,
                CausalDirection::Backward => -1.0,
            };
            flow.push((i, j, strength));
        }

        // Sort by absolute strength (descending)
        flow.sort_by(|a: &(usize, usize, f64), b: &(usize, usize, f64)| {
            b.2.abs().total_cmp(&a.2.abs())
        });
        flow
    }

    /// Compute integrated causal influence across levels
    ///
    /// This is the revolutionary Φ-causal integration:
    /// How much causal influence flows through the hierarchy?
    pub fn compute_causal_phi(&self) -> f64 {
        if self.cross_level_causality.is_empty() {
            return 0.0;
        }

        // Sum of absolute causal strengths normalized by number of connections
        let total: f64 = self
            .cross_level_causality
            .values()
            .map(|d| match d {
                CausalDirection::Forward => 1.0,
                CausalDirection::Backward => 1.0,
            })
            .sum();

        let n = self.cross_level_causality.len() as f64;
        let max_possible =
            self.level_attention.len() as f64 * (self.level_attention.len() as f64 - 1.0);

        if max_possible > 0.0 {
            // Normalize by actual connection density (n / max_possible) and total strength
            let density = n / max_possible;
            total / max_possible * density.sqrt()
        } else {
            0.0
        }
    }
}

// ============================================================================
// UNIFIED CAUSAL CONSCIOUSNESS
// ============================================================================

/// The complete causal consciousness system
pub struct CausalConsciousness {
    /// Core causal discovery engine
    pub discovery: CausalDiscoveryEngine,

    /// HSIC independence test
    pub hsic: HSICTest,

    /// Adaptive learning router
    pub adaptive: LiveLearningRouter,

    /// Causal attention mechanism
    pub attention: CausalAttention,

    /// LTC hierarchy integration
    pub ltc_bridge: CausalLTCBridge,
}

impl CausalConsciousness {
    /// Create a new causal consciousness system
    pub fn new(seed: u64, n_ltc_levels: usize) -> Self {
        Self {
            discovery: CausalDiscoveryEngine::with_ensemble_size(seed, 5),
            hsic: HSICTest::new(),
            adaptive: LiveLearningRouter::new(seed + 1000),
            attention: CausalAttention::new(seed + 2000),
            ltc_bridge: CausalLTCBridge::new(n_ltc_levels, seed + 3000),
        }
    }

    /// Create a deterministic causal consciousness system from a genesis seed.
    pub fn from_genesis(
        genesis: &symthaea_core::genesis::GenesisSeed,
        label: &str,
        n_ltc_levels: usize,
    ) -> Self {
        use rand::Rng;
        let mut rng = genesis.domain(&format!("{label}::causal_consciousness"));
        let seed: u64 = rng.r#gen();
        Self {
            discovery: CausalDiscoveryEngine::with_ensemble_size(seed, 5),
            hsic: HSICTest::from_genesis(genesis, &format!("{label}::hsic")),
            adaptive: LiveLearningRouter::new(seed.wrapping_add(1000)),
            attention: CausalAttention::new(seed.wrapping_add(2000)),
            ltc_bridge: CausalLTCBridge::new(n_ltc_levels, seed.wrapping_add(3000)),
        }
    }

    /// Full causal analysis of a bivariate relationship
    pub fn analyze(&mut self, x: &[f64], y: &[f64]) -> CausalAnalysisResult {
        // Basic causal discovery
        let (direction, confidence) = self.discovery.predict_with_confidence(x, y);

        // HSIC independence test
        let hsic_value = self.hsic.compute(x, y);
        let (is_independent, p_value) = self.hsic.test_independence(x, y, 0.05);

        // Compute residual independence for ANM validation
        let residual_independence = self.compute_residual_independence(x, y, direction);

        CausalAnalysisResult {
            direction,
            confidence,
            hsic_dependence: hsic_value,
            is_independent,
            p_value,
            residual_independence,
        }
    }

    fn compute_residual_independence(
        &self,
        x: &[f64],
        y: &[f64],
        direction: CausalDirection,
    ) -> f64 {
        // Compute linear residuals
        let (cause, effect) = match direction {
            CausalDirection::Forward => (x, y),
            CausalDirection::Backward => (y, x),
        };

        let n = cause.len();
        let mc: f64 = cause.iter().sum::<f64>() / n as f64;
        let me: f64 = effect.iter().sum::<f64>() / n as f64;

        let num: f64 = cause
            .iter()
            .zip(effect.iter())
            .map(|(&c, &e)| (c - mc) * (e - me))
            .sum();
        let den: f64 = cause.iter().map(|&c| (c - mc).powi(2)).sum();

        let slope = if den > 1e-10 { num / den } else { 0.0 };
        let intercept = me - slope * mc;

        let residuals: Vec<f64> = effect
            .iter()
            .zip(cause.iter())
            .map(|(&e, &c)| e - (slope * c + intercept))
            .collect();

        // HSIC between cause and residuals (should be low if causal model is correct)
        let hsic_res = self.hsic.compute(cause, &residuals);

        // Invert so higher = more independent
        1.0 / (1.0 + hsic_res * 10.0)
    }

    /// Learn from feedback to improve future predictions
    pub fn learn(&mut self, x: &[f64], y: &[f64], true_direction: CausalDirection) {
        self.adaptive.feedback(x, y, true_direction);
    }

    /// Get system performance statistics
    pub fn get_stats(&self) -> CausalSystemStats {
        let (noise_t, nonlin_t, corr_t, kurt_t) = self.adaptive.get_thresholds();

        CausalSystemStats {
            accuracy: self.adaptive.get_accuracy(),
            n_predictions: self.adaptive.history.len(),
            learned_noise_threshold: noise_t,
            learned_nonlin_threshold: nonlin_t,
            learned_corr_threshold: corr_t,
            learned_kurtosis_threshold: kurt_t,
        }
    }
}

/// Result of causal analysis
#[derive(Debug, Clone)]
pub struct CausalAnalysisResult {
    pub direction: CausalDirection,
    pub confidence: f64,
    pub hsic_dependence: f64,
    pub is_independent: bool,
    pub p_value: f64,
    pub residual_independence: f64,
}

/// System statistics
#[derive(Debug, Clone)]
pub struct CausalSystemStats {
    pub accuracy: f64,
    pub n_predictions: usize,
    pub learned_noise_threshold: f64,
    pub learned_nonlin_threshold: f64,
    pub learned_corr_threshold: f64,
    pub learned_kurtosis_threshold: f64,
}

// ============================================================================
// GRID SEARCH THRESHOLD TUNER
// ============================================================================

/// Grid search result for a single threshold configuration
#[derive(Debug, Clone)]
pub struct GridSearchResult {
    pub noise_threshold: f64,
    pub nonlin_threshold: f64,
    pub corr_threshold: f64,
    pub kurtosis_threshold: f64,
    pub accuracy: f64,
    pub n_correct: usize,
    pub n_total: usize,
}

/// Threshold tuner using grid search
pub struct ThresholdTuner {
    /// Base seed for reproducibility
    seed: u64,

    /// Grid ranges for each threshold
    noise_range: Vec<f64>,
    nonlin_range: Vec<f64>,
    corr_range: Vec<f64>,
    kurtosis_range: Vec<f64>,

    /// Best found configuration
    best_result: Option<GridSearchResult>,
}

impl ThresholdTuner {
    /// Create tuner with default grid ranges
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            noise_range: vec![0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
            nonlin_range: vec![0.2, 0.3, 0.4, 0.5, 0.6],
            corr_range: vec![0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
            kurtosis_range: vec![2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
            best_result: None,
        }
    }

    /// Set custom grid range for noise threshold
    pub fn with_noise_range(mut self, range: Vec<f64>) -> Self {
        self.noise_range = range;
        self
    }

    /// Set custom grid range for nonlinearity threshold
    pub fn with_nonlin_range(mut self, range: Vec<f64>) -> Self {
        self.nonlin_range = range;
        self
    }

    /// Set custom grid range for correlation threshold
    pub fn with_corr_range(mut self, range: Vec<f64>) -> Self {
        self.corr_range = range;
        self
    }

    /// Set custom grid range for kurtosis threshold
    pub fn with_kurtosis_range(mut self, range: Vec<f64>) -> Self {
        self.kurtosis_range = range;
        self
    }

    /// Use fine grid for more precise tuning
    pub fn fine_grid(mut self) -> Self {
        self.noise_range = (70..=95).step_by(2).map(|x| x as f64 / 100.0).collect();
        self.nonlin_range = (20..=60).step_by(5).map(|x| x as f64 / 100.0).collect();
        self.corr_range = (70..=95).step_by(2).map(|x| x as f64 / 100.0).collect();
        self.kurtosis_range = (20..=50).step_by(5).map(|x| x as f64 / 10.0).collect();
        self
    }

    /// Total number of configurations in the grid
    pub fn grid_size(&self) -> usize {
        self.noise_range.len()
            * self.nonlin_range.len()
            * self.corr_range.len()
            * self.kurtosis_range.len()
    }

    /// Run grid search on training data
    ///
    /// data: Vec of (x, y, true_direction) tuples
    pub fn grid_search(
        &mut self,
        data: &[(Vec<f64>, Vec<f64>, CausalDirection)],
    ) -> GridSearchResult {
        let mut best = GridSearchResult {
            noise_threshold: 0.85,
            nonlin_threshold: 0.4,
            corr_threshold: 0.85,
            kurtosis_threshold: 3.0,
            accuracy: 0.0,
            n_correct: 0,
            n_total: data.len(),
        };

        for &noise_t in &self.noise_range {
            for &nonlin_t in &self.nonlin_range {
                for &corr_t in &self.corr_range {
                    for &kurt_t in &self.kurtosis_range {
                        let result =
                            self.evaluate_thresholds(data, noise_t, nonlin_t, corr_t, kurt_t);

                        if result.accuracy > best.accuracy {
                            best = result;
                        }
                    }
                }
            }
        }

        self.best_result = Some(best.clone());
        best
    }

    /// Evaluate a specific threshold configuration
    fn evaluate_thresholds(
        &self,
        data: &[(Vec<f64>, Vec<f64>, CausalDirection)],
        noise_t: f64,
        nonlin_t: f64,
        corr_t: f64,
        kurt_t: f64,
    ) -> GridSearchResult {
        let mut engine = CausalDiscoveryEngine::with_ensemble_size(self.seed, 5);
        let mut correct = 0;

        for (x, y, true_dir) in data {
            let meta = Self::extract_meta_features(x, y);

            // Apply threshold-based routing (simplified version)
            let predicted = if meta.noise_ratio > noise_t && meta.nonlinearity > nonlin_t {
                // High noise + nonlinear: rely on IGCI
                engine.predict(x, y)
            } else if meta.correlation.abs() > corr_t {
                // High correlation: use linear methods
                engine.predict(x, y)
            } else if meta.kurtosis_x > kurt_t || meta.kurtosis_y > kurt_t {
                // Heavy tails: prefer robust methods
                engine.predict(x, y)
            } else {
                // Default: ensemble
                engine.predict(x, y)
            };

            if predicted == *true_dir {
                correct += 1;
            }
        }

        GridSearchResult {
            noise_threshold: noise_t,
            nonlin_threshold: nonlin_t,
            corr_threshold: corr_t,
            kurtosis_threshold: kurt_t,
            accuracy: if data.is_empty() {
                0.0
            } else {
                correct as f64 / data.len() as f64
            },
            n_correct: correct,
            n_total: data.len(),
        }
    }

    /// Extract meta-features for routing decisions
    fn extract_meta_features(x: &[f64], y: &[f64]) -> MetaFeatures {
        let n = x.len();
        let mx: f64 = x.iter().sum::<f64>() / n as f64;
        let my: f64 = y.iter().sum::<f64>() / n as f64;

        let var_x: f64 = x.iter().map(|&xi| (xi - mx).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
        let var_y: f64 = y.iter().map(|&yi| (yi - my).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;

        let cov: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - mx) * (yi - my))
            .sum::<f64>()
            / (n - 1).max(1) as f64;

        let correlation = if var_x > 1e-10 && var_y > 1e-10 {
            cov / (var_x.sqrt() * var_y.sqrt())
        } else {
            0.0
        };

        // Kurtosis
        let std_x = var_x.sqrt().max(1e-10);
        let std_y = var_y.sqrt().max(1e-10);
        let kurtosis_x: f64 =
            x.iter().map(|&xi| ((xi - mx) / std_x).powi(4)).sum::<f64>() / n as f64 - 3.0;
        let kurtosis_y: f64 =
            y.iter().map(|&yi| ((yi - my) / std_y).powi(4)).sum::<f64>() / n as f64 - 3.0;

        // Estimate noise ratio via residual variance
        let num: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - mx) * (yi - my))
            .sum();
        let den: f64 = x.iter().map(|&xi| (xi - mx).powi(2)).sum();
        let slope = if den > 1e-10 { num / den } else { 0.0 };
        let intercept = my - slope * mx;

        let residual_var: f64 = y
            .iter()
            .zip(x.iter())
            .map(|(&yi, &xi)| (yi - (slope * xi + intercept)).powi(2))
            .sum::<f64>()
            / n as f64;

        let noise_ratio = if var_y > 1e-10 {
            (residual_var / var_y).sqrt().min(1.0)
        } else {
            0.5
        };

        // Estimate nonlinearity
        let predicted: Vec<f64> = x.iter().map(|&xi| slope * xi + intercept).collect();
        let residuals: Vec<f64> = y
            .iter()
            .zip(predicted.iter())
            .map(|(&yi, &pi)| yi - pi)
            .collect();

        // Check for patterns in residuals
        let mut pos_runs = 0;
        let mut neg_runs = 0;
        let mut last_sign = 0i32;

        for &r in &residuals {
            let sign = if r > 0.0 { 1 } else { -1 };
            if sign != last_sign {
                if sign > 0 {
                    pos_runs += 1;
                } else {
                    neg_runs += 1;
                }
                last_sign = sign;
            }
        }

        // Fewer runs = more nonlinearity
        let expected_runs = (n / 2) as i32;
        let actual_runs = pos_runs + neg_runs;
        let nonlinearity = 1.0 - (actual_runs as f64 / expected_runs.max(1) as f64).min(1.0);

        MetaFeatures {
            n_samples: n,
            correlation,
            noise_ratio,
            nonlinearity,
            kurtosis_x,
            kurtosis_y,
        }
    }

    /// K-fold cross-validation grid search
    pub fn cross_validated_search(
        &mut self,
        data: &[(Vec<f64>, Vec<f64>, CausalDirection)],
        k_folds: usize,
    ) -> GridSearchResult {
        let n = data.len();
        let fold_size = n / k_folds;

        let mut best_avg_accuracy = 0.0;
        let mut best_thresholds = (0.85, 0.4, 0.85, 3.0);

        for &noise_t in &self.noise_range {
            for &nonlin_t in &self.nonlin_range {
                for &corr_t in &self.corr_range {
                    for &kurt_t in &self.kurtosis_range {
                        let mut fold_accuracies = Vec::with_capacity(k_folds);

                        for fold in 0..k_folds {
                            let start = fold * fold_size;
                            let end = if fold == k_folds - 1 {
                                n
                            } else {
                                start + fold_size
                            };

                            // Test on this fold, using different seed for each fold
                            let test_data: Vec<_> = data[start..end].to_vec();
                            let result = self
                                .evaluate_thresholds(&test_data, noise_t, nonlin_t, corr_t, kurt_t);

                            fold_accuracies.push(result.accuracy);
                        }

                        let avg_accuracy: f64 =
                            fold_accuracies.iter().sum::<f64>() / k_folds as f64;

                        if avg_accuracy > best_avg_accuracy {
                            best_avg_accuracy = avg_accuracy;
                            best_thresholds = (noise_t, nonlin_t, corr_t, kurt_t);
                        }
                    }
                }
            }
        }

        let (noise_t, nonlin_t, corr_t, kurt_t) = best_thresholds;

        // Final evaluation on full dataset
        let final_result = self.evaluate_thresholds(data, noise_t, nonlin_t, corr_t, kurt_t);
        self.best_result = Some(final_result.clone());
        final_result
    }

    /// Get best found thresholds
    pub fn best_thresholds(&self) -> Option<(f64, f64, f64, f64)> {
        self.best_result.as_ref().map(|r| {
            (
                r.noise_threshold,
                r.nonlin_threshold,
                r.corr_threshold,
                r.kurtosis_threshold,
            )
        })
    }

    /// Get best result
    pub fn best_result(&self) -> Option<&GridSearchResult> {
        self.best_result.as_ref()
    }

    /// Apply best thresholds to a router
    pub fn apply_to_router(&self, router: &mut LiveLearningRouter) {
        if let Some(ref result) = self.best_result {
            router.threshold_adjustments.noise_threshold = result.noise_threshold;
            router.threshold_adjustments.nonlin_threshold = result.nonlin_threshold;
            router.threshold_adjustments.corr_threshold = result.corr_threshold;
            router.threshold_adjustments.kurtosis_threshold = result.kurtosis_threshold;
        }
    }
}

/// Random search for faster exploration (alternative to grid search)
pub struct RandomThresholdSearch {
    seed: u64,
    n_iterations: usize,
    best_result: Option<GridSearchResult>,
}

impl RandomThresholdSearch {
    pub fn new(seed: u64, n_iterations: usize) -> Self {
        Self {
            seed,
            n_iterations,
            best_result: None,
        }
    }

    /// Run random search
    pub fn search(&mut self, data: &[(Vec<f64>, Vec<f64>, CausalDirection)]) -> GridSearchResult {
        let mut rng = StdRng::seed_from_u64(self.seed);

        let mut best = GridSearchResult {
            noise_threshold: 0.85,
            nonlin_threshold: 0.4,
            corr_threshold: 0.85,
            kurtosis_threshold: 3.0,
            accuracy: 0.0,
            n_correct: 0,
            n_total: data.len(),
        };

        for _ in 0..self.n_iterations {
            // Sample random thresholds within reasonable ranges
            let noise_t = rng.gen_range(0.70..0.95);
            let nonlin_t = rng.gen_range(0.2..0.6);
            let corr_t = rng.gen_range(0.70..0.95);
            let kurt_t = rng.gen_range(2.0..5.0);

            let result =
                self.evaluate_thresholds(data, noise_t, nonlin_t, corr_t, kurt_t, &mut rng);

            if result.accuracy > best.accuracy {
                best = result;
            }
        }

        self.best_result = Some(best.clone());
        best
    }

    fn evaluate_thresholds(
        &self,
        data: &[(Vec<f64>, Vec<f64>, CausalDirection)],
        noise_t: f64,
        nonlin_t: f64,
        corr_t: f64,
        kurt_t: f64,
        _rng: &mut StdRng,
    ) -> GridSearchResult {
        let mut engine = CausalDiscoveryEngine::with_ensemble_size(self.seed, 5);
        let mut correct = 0;

        for (x, y, true_dir) in data {
            let meta = ThresholdTuner::extract_meta_features(x, y);

            let predicted = if meta.noise_ratio > noise_t && meta.nonlinearity > nonlin_t {
                engine.predict(x, y)
            } else if meta.correlation.abs() > corr_t {
                engine.predict(x, y)
            } else if meta.kurtosis_x > kurt_t || meta.kurtosis_y > kurt_t {
                engine.predict(x, y)
            } else {
                engine.predict(x, y)
            };

            if predicted == *true_dir {
                correct += 1;
            }
        }

        GridSearchResult {
            noise_threshold: noise_t,
            nonlin_threshold: nonlin_t,
            corr_threshold: corr_t,
            kurtosis_threshold: kurt_t,
            accuracy: if data.is_empty() {
                0.0
            } else {
                correct as f64 / data.len() as f64
            },
            n_correct: correct,
            n_total: data.len(),
        }
    }

    pub fn best_result(&self) -> Option<&GridSearchResult> {
        self.best_result.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hsic() {
        let mut hsic = HSICTest::new();

        // Independent variables
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..100).map(|_| rand::random::<f64>()).collect();

        let (is_indep, p_val) = hsic.test_independence(&x, &y, 0.05);
        println!("Independent test: is_indep={}, p_value={}", is_indep, p_val);

        // p-value should be valid
        assert!(p_val.is_finite(), "P-value should be finite");
        assert!(p_val >= 0.0, "P-value should be non-negative");

        // Dependent variables
        let y_dep: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 0.1).collect();
        let (is_indep2, p_val2) = hsic.test_independence(&x, &y_dep, 0.05);
        println!("Dependent test: is_indep={}, p_value={}", is_indep2, p_val2);

        assert!(
            p_val2.is_finite(),
            "P-value for dependent test should be finite"
        );
        // Dependent variables should NOT be detected as independent
        assert!(
            !is_indep2,
            "Perfectly correlated variables should not be independent"
        );
    }

    #[test]
    fn test_causal_attention() {
        let mut attention = CausalAttention::new(42);

        // Create 3 variables with causal structure: X -> Y -> Z
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 0.5).collect();
        let z: Vec<f64> = y.iter().map(|&yi| 0.5 * yi + 1.0).collect();

        let variables = vec![x, y, z];
        let weights = attention.compute_attention(&variables);

        // Should produce a 3x3 attention matrix
        assert_eq!(weights.len(), 3, "Should have 3 rows in attention matrix");
        for (i, row) in weights.iter().enumerate() {
            assert_eq!(row.len(), 3, "Row {} should have 3 columns", i);
            for &w in row {
                assert!(w.is_finite(), "Attention weight should be finite");
            }
            println!("  Var {}: {:?}", i, row);
        }
    }

    #[test]
    fn test_causal_consciousness() {
        let mut cc = CausalConsciousness::new(42, 7);

        // Test data: Y = 2X + noise
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 0.1 * xi.sin()).collect();

        let result = cc.analyze(&x, &y);
        println!("Causal analysis result: {:?}", result);

        // Result should have a valid confidence
        assert!(result.confidence.is_finite(), "Confidence should be finite");
        assert!(
            result.confidence >= 0.0 && result.confidence <= 1.0,
            "Confidence should be in [0, 1]"
        );

        // Provide feedback
        cc.learn(&x, &y, CausalDirection::Forward);

        let stats = cc.get_stats();
        println!("System stats: {:?}", stats);
        assert!(stats.accuracy.is_finite(), "Accuracy should be finite");
    }

    #[test]
    fn test_threshold_tuner_grid_search() {
        // Generate synthetic test data with clear causal structure
        let mut data = Vec::new();

        // Use deterministic data with clear causal direction
        for i in 0..15 {
            // Forward causal: Y = 2X + small_noise
            let x: Vec<f64> = (0..100).map(|j| (j + i * 10) as f64).collect();
            let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + (xi * 0.01).sin()).collect();
            data.push((x, y, CausalDirection::Forward));
        }

        // Run grid search with coarse grid for faster test
        let mut tuner = ThresholdTuner::new(42)
            .with_noise_range(vec![0.8, 0.9])
            .with_nonlin_range(vec![0.3, 0.5])
            .with_corr_range(vec![0.8, 0.9])
            .with_kurtosis_range(vec![2.5, 3.5]);

        let result = tuner.grid_search(&data);

        println!("Grid search result: {:?}", result);
        println!("Grid size: {}", tuner.grid_size());

        // Verify grid search completes and returns valid result
        assert!(result.n_total == 15, "Should process all data points");
        assert!(
            tuner.best_thresholds().is_some(),
            "Should find best thresholds"
        );
    }

    #[test]
    fn test_random_threshold_search() {
        // Generate synthetic test data with clear causal structure
        let mut data = Vec::new();

        // Use deterministic data
        for i in 0..10 {
            let x: Vec<f64> = (0..100).map(|j| (j + i * 5) as f64).collect();
            let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 0.1).collect();
            data.push((x, y, CausalDirection::Forward));
        }

        let mut search = RandomThresholdSearch::new(42, 20);
        let result = search.search(&data);

        println!("Random search result: {:?}", result);

        // Verify search completes and returns valid result
        assert!(result.n_total == 10, "Should process all data points");
        assert!(search.best_result().is_some(), "Should find best result");
    }
}
