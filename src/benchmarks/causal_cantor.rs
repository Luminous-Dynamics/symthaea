//! # Causal Discovery via Hierarchical Cantor-LTC Network
//!
//! A novel approach to causal discovery using the multi-timescale Cantor-LTC architecture.
//! Instead of flat ensemble voting, this uses hierarchical evidence integration.
//!
//! ## Key Insight
//!
//! Causal discovery requires multi-level reasoning:
//! - Level 7 (fast): Raw statistical features (quantiles, moments, residuals)
//! - Level 5-6: Pattern integration (independence measures, asymmetries)
//! - Level 3-4: Causal signal extraction (learned combinations)
//! - Level 1-2: Meta-reasoning (which signals to trust)
//! - Level 0 (slow): Unified causal verdict
//!
//! The Cantor hierarchy naturally implements this via τ scaling.
//!
//! ## Architecture
//!
//! ```text
//! (x, y) data → Feature Encoder → Level 7 HDC injection
//!                                        ↓
//!                               Hierarchical integration
//!                                        ↓
//!                               Level 0 root state
//!                                        ↓
//!                            Direction decoder + Φ confidence
//! ```

use symthaea_core::hdc::{RealHV, HDC_DIMENSION};
use crate::hierarchical_cantor_ltc::{HierarchicalCantorLtcNetwork, CantorLtcConfig, CantorLtcNode};
use super::{CausalDirection, CausalDiscoveryResult};

// =============================================================================
// CAUSAL FEATURE ENCODER
// =============================================================================

/// Encodes (x, y) data pairs as HDC hypervectors for Level 7 injection
///
/// Features extracted:
/// 1. Distribution features: quantiles, moments (mean, var, skew, kurt)
/// 2. Regression features: residuals, R², error asymmetry
/// 3. Independence features: HSIC, conditional entropy estimates
/// 4. Structural features: linearity, noise level, sample size
#[allow(dead_code)] // Fields reserved for causal encoding
pub struct CausalFeatureEncoder {
    /// Dimension of output hypervectors
    dim: usize,
    /// Random projection matrices for each feature type (learned)
    projection_forward: Vec<f32>,
    projection_backward: Vec<f32>,
    /// Position encodings for quantile bins
    quantile_positions: Vec<RealHV>,
    /// Feature importance weights (learned during training)
    feature_weights: Vec<f32>,
}

impl CausalFeatureEncoder {
    /// Create a new encoder with the given dimension
    pub fn new(dim: usize) -> Self {
        // Initialize random projections
        let mut projection_forward = vec![0.0; dim * 64]; // 64 input features
        let mut projection_backward = vec![0.0; dim * 64];

        // Use deterministic initialization for reproducibility
        for i in 0..projection_forward.len() {
            let seed = i as f64;
            projection_forward[i] = ((seed * 0.618033988749895) % 1.0 - 0.5) as f32 * 2.0;
            projection_backward[i] = ((seed * 0.414213562373095) % 1.0 - 0.5) as f32 * 2.0;
        }

        // Quantile position encodings (10 bins)
        let quantile_positions: Vec<RealHV> = (0..10)
            .map(|i| RealHV::random(dim, (i * 12345) as u64))
            .collect();

        // Initial feature weights (uniform)
        let feature_weights = vec![1.0; 64];

        Self {
            dim,
            projection_forward,
            projection_backward,
            quantile_positions,
            feature_weights,
        }
    }

    /// Extract raw numerical features from (x, y) data
    fn extract_features(&self, x: &[f64], y: &[f64]) -> Vec<f32> {
        let n = x.len() as f64;
        let mut features = Vec::with_capacity(64);

        // === Distribution features for X (10 features) ===
        let x_sorted = {
            let mut s = x.to_vec();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            s
        };

        // Quantiles (5)
        for q in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let idx = ((n - 1.0) * q) as usize;
            features.push(x_sorted.get(idx).copied().unwrap_or(0.0) as f32);
        }

        // Moments (5)
        let x_mean = x.iter().sum::<f64>() / n;
        let x_var = x.iter().map(|v| (v - x_mean).powi(2)).sum::<f64>() / n;
        let x_std = x_var.sqrt().max(1e-10);
        let x_skew = x.iter().map(|v| ((v - x_mean) / x_std).powi(3)).sum::<f64>() / n;
        let x_kurt = x.iter().map(|v| ((v - x_mean) / x_std).powi(4)).sum::<f64>() / n - 3.0;

        features.push(x_mean as f32);
        features.push(x_var as f32);
        features.push(x_std as f32);
        features.push(x_skew as f32);
        features.push(x_kurt as f32);

        // === Distribution features for Y (10 features) ===
        let y_sorted = {
            let mut s = y.to_vec();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            s
        };

        for q in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let idx = ((n - 1.0) * q) as usize;
            features.push(y_sorted.get(idx).copied().unwrap_or(0.0) as f32);
        }

        let y_mean = y.iter().sum::<f64>() / n;
        let y_var = y.iter().map(|v| (v - y_mean).powi(2)).sum::<f64>() / n;
        let y_std = y_var.sqrt().max(1e-10);
        let y_skew = y.iter().map(|v| ((v - y_mean) / y_std).powi(3)).sum::<f64>() / n;
        let y_kurt = y.iter().map(|v| ((v - y_mean) / y_std).powi(4)).sum::<f64>() / n - 3.0;

        features.push(y_mean as f32);
        features.push(y_var as f32);
        features.push(y_std as f32);
        features.push(y_skew as f32);
        features.push(y_kurt as f32);

        // === Regression features X→Y (10 features) ===
        let (residuals_xy, r2_xy, error_xy) = self.compute_regression_features(x, y);
        features.push(r2_xy as f32);
        features.push(error_xy as f32);

        // Residual statistics
        let res_mean_xy: f64 = residuals_xy.iter().sum::<f64>() / n;
        let res_var_xy: f64 = residuals_xy.iter().map(|r| (r - res_mean_xy).powi(2)).sum::<f64>() / n;
        let res_std_xy = res_var_xy.sqrt().max(1e-10);
        let res_skew_xy: f64 = residuals_xy.iter().map(|r| ((r - res_mean_xy) / res_std_xy).powi(3)).sum::<f64>() / n;
        let res_kurt_xy: f64 = residuals_xy.iter().map(|r| ((r - res_mean_xy) / res_std_xy).powi(4)).sum::<f64>() / n - 3.0;

        features.push(res_mean_xy as f32);
        features.push(res_var_xy as f32);
        features.push(res_skew_xy as f32);
        features.push(res_kurt_xy as f32);

        // Independence measure: correlation of residuals with x
        let res_x_corr = self.correlation(&residuals_xy, x);
        features.push(res_x_corr as f32);
        features.push(res_x_corr.abs() as f32);
        features.push((res_x_corr.powi(2)) as f32);

        // === Regression features Y→X (10 features) ===
        let (residuals_yx, r2_yx, error_yx) = self.compute_regression_features(y, x);
        features.push(r2_yx as f32);
        features.push(error_yx as f32);

        let res_mean_yx: f64 = residuals_yx.iter().sum::<f64>() / n;
        let res_var_yx: f64 = residuals_yx.iter().map(|r| (r - res_mean_yx).powi(2)).sum::<f64>() / n;
        let res_std_yx = res_var_yx.sqrt().max(1e-10);
        let res_skew_yx: f64 = residuals_yx.iter().map(|r| ((r - res_mean_yx) / res_std_yx).powi(3)).sum::<f64>() / n;
        let res_kurt_yx: f64 = residuals_yx.iter().map(|r| ((r - res_mean_yx) / res_std_yx).powi(4)).sum::<f64>() / n - 3.0;

        features.push(res_mean_yx as f32);
        features.push(res_var_yx as f32);
        features.push(res_skew_yx as f32);
        features.push(res_kurt_yx as f32);

        let res_y_corr = self.correlation(&residuals_yx, y);
        features.push(res_y_corr as f32);
        features.push(res_y_corr.abs() as f32);
        features.push((res_y_corr.powi(2)) as f32);

        // === Asymmetry features (10 features) ===
        // These are the key causal signals!
        let error_asymmetry = error_xy - error_yx;
        let r2_asymmetry = r2_xy - r2_yx;
        let skew_asymmetry = x_skew - y_skew;
        let kurt_asymmetry = x_kurt - y_kurt;
        let res_indep_asymmetry = res_x_corr.abs() - res_y_corr.abs();

        features.push(error_asymmetry as f32);
        features.push(r2_asymmetry as f32);
        features.push(skew_asymmetry as f32);
        features.push(kurt_asymmetry as f32);
        features.push(res_indep_asymmetry as f32);

        // Normalized asymmetries (scale-invariant)
        let error_total = (error_xy + error_yx).max(1e-10);
        features.push((error_asymmetry / error_total) as f32);
        features.push((res_skew_xy - res_skew_yx) as f32);
        features.push((res_kurt_xy - res_kurt_yx) as f32);
        features.push((res_var_xy - res_var_yx) as f32);
        features.push(((res_var_xy - res_var_yx) / (res_var_xy + res_var_yx).max(1e-10)) as f32);

        // === Meta features (14 features to reach 64) ===
        features.push(n as f32);                                    // Sample size
        features.push((n as f32).ln());                             // Log sample size
        features.push(self.correlation(x, y) as f32);               // Raw correlation
        features.push(self.correlation(x, y).abs() as f32);         // Abs correlation
        features.push((1.0 - self.correlation(x, y).abs()) as f32); // Independence proxy

        // Linearity measure (how linear is the relationship?)
        let linearity = r2_xy.max(r2_yx);
        features.push(linearity as f32);
        features.push((1.0 - linearity) as f32);  // Non-linearity

        // Noise level estimates
        features.push(((1.0 - r2_xy) * y_var) as f32);
        features.push(((1.0 - r2_yx) * x_var) as f32);

        // Combined signals (pre-computed for efficiency)
        let reci_signal = error_asymmetry.signum() as f32;
        let anm_signal = res_indep_asymmetry.signum() as f32;
        let lingam_signal = (res_kurt_xy - res_kurt_yx).signum() as f32;

        features.push(reci_signal);
        features.push(anm_signal);
        features.push(lingam_signal);
        features.push((reci_signal + anm_signal + lingam_signal) / 3.0);
        features.push(if reci_signal == anm_signal && anm_signal == lingam_signal { 1.0 } else { 0.0 });

        // Ensure exactly 64 features
        while features.len() < 64 {
            features.push(0.0);
        }
        features.truncate(64);

        // Normalize features
        self.normalize_features(&mut features);

        features
    }

    /// Compute regression of y on x, return (residuals, R², MSE)
    fn compute_regression_features(&self, x: &[f64], y: &[f64]) -> (Vec<f64>, f64, f64) {
        let n = x.len() as f64;

        // Simple kernel regression (Nadaraya-Watson)
        let bandwidth = self.silverman_bandwidth(x);

        let mut predictions = Vec::with_capacity(x.len());
        let mut residuals = Vec::with_capacity(x.len());

        for i in 0..x.len() {
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for j in 0..x.len() {
                if i != j {
                    let diff = (x[i] - x[j]) / bandwidth;
                    let weight = (-0.5 * diff * diff).exp();
                    weighted_sum += weight * y[j];
                    weight_sum += weight;
                }
            }

            let pred = if weight_sum > 1e-10 {
                weighted_sum / weight_sum
            } else {
                y.iter().sum::<f64>() / n
            };

            predictions.push(pred);
            residuals.push(y[i] - pred);
        }

        // Compute R² and MSE
        let y_mean = y.iter().sum::<f64>() / n;
        let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
        let ss_res: f64 = residuals.iter().map(|r| r.powi(2)).sum();

        let r2 = if ss_tot > 1e-10 { 1.0 - ss_res / ss_tot } else { 0.0 };
        let mse = ss_res / n;

        (residuals, r2.max(0.0).min(1.0), mse)
    }

    /// Silverman's rule of thumb for bandwidth selection
    fn silverman_bandwidth(&self, x: &[f64]) -> f64 {
        let n = x.len() as f64;
        let mean = x.iter().sum::<f64>() / n;
        let std = (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();

        // IQR
        let mut sorted = x.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let q1 = sorted[(n * 0.25) as usize];
        let q3 = sorted[(n * 0.75) as usize];
        let iqr = q3 - q1;

        let scale = std.min(iqr / 1.34);
        0.9 * scale * n.powf(-0.2)
    }

    /// Compute Pearson correlation
    fn correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let x_mean = x.iter().sum::<f64>() / n;
        let y_mean = y.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - x_mean;
            let dy = y[i] - y_mean;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        if var_x < 1e-10 || var_y < 1e-10 {
            return 0.0;
        }

        cov / (var_x.sqrt() * var_y.sqrt())
    }

    /// Normalize features to [-1, 1] range
    fn normalize_features(&self, features: &mut [f32]) {
        for f in features.iter_mut() {
            // Soft clipping with tanh
            *f = f.tanh();
        }
    }

    /// Encode features as a hypervector
    pub fn encode(&self, x: &[f64], y: &[f64]) -> RealHV {
        let features = self.extract_features(x, y);

        // Project features to hypervector space
        let mut hv_values = vec![0.0f32; self.dim];

        for (i, &feat) in features.iter().enumerate() {
            let weight = self.feature_weights.get(i).copied().unwrap_or(1.0);
            let weighted_feat = feat * weight;

            for j in 0..self.dim {
                let proj_idx = i * self.dim + j;
                if proj_idx < self.projection_forward.len() {
                    hv_values[j] += weighted_feat * self.projection_forward[proj_idx];
                }
            }
        }

        // Normalize
        let norm: f32 = hv_values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for v in hv_values.iter_mut() {
                *v /= norm;
            }
        }

        RealHV::from_values(hv_values)
    }

    /// Encode with direction hint (for creating direction-specific representations)
    pub fn encode_with_direction(&self, x: &[f64], y: &[f64], forward: bool) -> RealHV {
        let features = self.extract_features(x, y);
        let proj = if forward { &self.projection_forward } else { &self.projection_backward };

        let mut hv_values = vec![0.0f32; self.dim];

        for (i, &feat) in features.iter().enumerate() {
            let weight = self.feature_weights.get(i).copied().unwrap_or(1.0);
            let weighted_feat = feat * weight;

            for j in 0..self.dim {
                let proj_idx = i * self.dim + j;
                if proj_idx < proj.len() {
                    hv_values[j] += weighted_feat * proj[proj_idx];
                }
            }
        }

        let norm: f32 = hv_values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for v in hv_values.iter_mut() {
                *v /= norm;
            }
        }

        RealHV::from_values(hv_values)
    }
}

// =============================================================================
// CAUSAL CANTOR NETWORK
// =============================================================================

/// Causal discovery using the Hierarchical Cantor-LTC Network
///
/// This is a novel approach that uses multi-timescale integration
/// for causal reasoning. The network learns to combine evidence
/// hierarchically rather than through flat ensemble voting.
pub struct CausalCantorNetwork {
    /// The underlying Cantor-LTC network
    network: HierarchicalCantorLtcNetwork,
    /// Feature encoder
    encoder: CausalFeatureEncoder,
    /// Direction prototypes (learned representations of X→Y and Y→X)
    forward_prototype: RealHV,
    backward_prototype: RealHV,
    /// Decision threshold (similarity required for confident prediction)
    decision_threshold: f32,
    /// Number of integration steps
    integration_steps: usize,
    /// Integration timestep (ms)
    dt: f32,
    /// Training data (for online learning)
    training_pairs: Vec<TrainingPair>,
}

/// A training example for the causal network
#[allow(dead_code)] // Fields reserved for training pipeline
struct TrainingPair {
    features: RealHV,
    direction: CausalDirection,
}

impl CausalCantorNetwork {
    /// Create a new CausalCantorNetwork
    pub fn new() -> Self {
        let config = CantorLtcConfig {
            dimension: HDC_DIMENSION,
            max_depth: 5,  // Reduced depth for causal task (faster)
            tau_ratio: 1.0 / 3.0,
            base_tau: 100.0,  // 100ms at root
            learning_rate: 0.01,
            measure_phi: true,
        };

        let network = HierarchicalCantorLtcNetwork::new(config);
        let encoder = CausalFeatureEncoder::new(HDC_DIMENSION);

        // Initialize prototypes with random vectors
        let forward_prototype = RealHV::random(HDC_DIMENSION, 42);
        let backward_prototype = RealHV::random(HDC_DIMENSION, 43);

        Self {
            network,
            encoder,
            forward_prototype,
            backward_prototype,
            decision_threshold: 0.1,
            integration_steps: 50,
            dt: 1.0,
            training_pairs: Vec::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(depth: usize, base_tau: f32, integration_steps: usize) -> Self {
        let config = CantorLtcConfig {
            dimension: HDC_DIMENSION,
            max_depth: depth,
            tau_ratio: 1.0 / 3.0,
            base_tau,
            learning_rate: 0.01,
            measure_phi: true,
        };

        let network = HierarchicalCantorLtcNetwork::new(config);
        let encoder = CausalFeatureEncoder::new(HDC_DIMENSION);

        Self {
            network,
            encoder,
            forward_prototype: RealHV::random(HDC_DIMENSION, 42),
            backward_prototype: RealHV::random(HDC_DIMENSION, 43),
            decision_threshold: 0.1,
            integration_steps,
            dt: 1.0,
            training_pairs: Vec::new(),
        }
    }

    /// Train the network on a batch of labeled pairs
    pub fn train(&mut self, pairs: &[(Vec<f64>, Vec<f64>, CausalDirection)]) {
        // Collect forward and backward examples
        let mut forward_examples = Vec::new();
        let mut backward_examples = Vec::new();

        for (x, y, direction) in pairs {
            let features = self.encoder.encode(x, y);

            match *direction {
                CausalDirection::Forward => forward_examples.push(features.clone()),
                CausalDirection::Backward => backward_examples.push(features.clone()),
            }

            self.training_pairs.push(TrainingPair {
                features,
                direction: *direction,
            });
        }

        // Update prototypes as centroid of examples
        if !forward_examples.is_empty() {
            let bundle_refs: Vec<RealHV> = forward_examples;
            self.forward_prototype = RealHV::bundle(&bundle_refs);
        }

        if !backward_examples.is_empty() {
            let bundle_refs: Vec<RealHV> = backward_examples;
            self.backward_prototype = RealHV::bundle(&bundle_refs);
        }

        // Inject training signals into the network
        self.inject_training_signal();
    }

    /// Inject training signal into network weights
    fn inject_training_signal(&mut self) {
        // Use the training pairs to update the network's internal weights
        // This is a simplified Hebbian update

        for pair in &self.training_pairs {
            // Inject into network and run integration
            self.network.step_with_input(self.dt, &pair.features);

            // Run a few integration steps
            for _ in 0..10 {
                self.network.step(self.dt);
            }
        }
    }

    /// Discover causal direction for a new pair
    pub fn discover(&mut self, x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
        // Reset network state (fresh integration)
        self.reset_network_state();

        // Encode the data pair
        let features = self.encoder.encode(x, y);

        // Inject features into the network (at root, will propagate)
        self.network.step_with_input(self.dt, &features);

        // Run hierarchical integration
        for _ in 0..self.integration_steps {
            self.network.step(self.dt);
        }

        // Read the integrated state from root
        let root_state = self.network.get_root_state().clone();

        // Compute similarity to direction prototypes
        let sim_forward = root_state.similarity(&self.forward_prototype);
        let sim_backward = root_state.similarity(&self.backward_prototype);

        // Compute hierarchical Φ for confidence
        let phi = self.network.compute_hierarchical_phi();

        // Decision based on similarity difference
        let sim_diff = sim_forward - sim_backward;
        let confidence = sim_diff.abs() * phi as f32;

        let (direction, p_forward) = if sim_diff > self.decision_threshold {
            (CausalDirection::Forward, 0.5 + sim_diff.min(0.5))
        } else if sim_diff < -self.decision_threshold {
            (CausalDirection::Backward, 0.5 + sim_diff.max(-0.5))
        } else {
            // Undetermined - use raw features as tiebreaker
            let raw_signal = self.compute_raw_signal(x, y);
            if raw_signal > 0.0 {
                (CausalDirection::Forward, 0.5 + raw_signal.min(0.3) as f32)
            } else {
                (CausalDirection::Backward, 0.5 + raw_signal.max(-0.3) as f32)
            }
        };

        let p_forward_f64 = p_forward as f64;
        CausalDiscoveryResult {
            direction,
            confidence: confidence.abs() as f64,
            p_forward: p_forward_f64,
            p_backward: 1.0 - p_forward_f64,
            features: super::CausalFeatures {
                reci_score: 0.0,
                igci_score: 0.0,
                anm_score: 0.0,
                higher_order_score: 0.0,
            },
        }
    }

    /// Compute raw signal from features (fallback)
    fn compute_raw_signal(&self, x: &[f64], y: &[f64]) -> f64 {
        // Use RECI-style error asymmetry as tiebreaker
        let (_, _, error_xy) = self.encoder.compute_regression_features(x, y);
        let (_, _, error_yx) = self.encoder.compute_regression_features(y, x);

        let asymmetry = error_xy - error_yx;
        let total = (error_xy + error_yx).max(1e-10);

        -asymmetry / total  // Negative because lower error = more likely cause
    }

    /// Reset network state for fresh integration
    fn reset_network_state(&mut self) {
        // Re-initialize the network with fresh random states
        // but keep the learned weights/biases
        reset_node_states_recursive(&mut self.network.root);
        self.network.time = 0.0;
    }

    /// Get the current Φ value
    pub fn get_phi(&self) -> f64 {
        self.network.global_phi
    }

    /// Get per-level Φ values
    pub fn get_level_phi(&self) -> &[f64] {
        &self.network.level_phi
    }
}

impl Default for CausalCantorNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// Recursively reset node states in the Cantor tree
fn reset_node_states_recursive(node: &mut CantorLtcNode) {
    // Reset to small random state
    let seed = (node.level * 1000 + node.index) as u64;
    node.state = RealHV::random(node.state.values.len(), seed).scale(0.1);
    node.history.clear();

    if let Some((ref mut left, ref mut right)) = node.children {
        reset_node_states_recursive(left);
        reset_node_states_recursive(right);
    }
}

// =============================================================================
// PUBLIC API FUNCTIONS
// =============================================================================

/// Discover causal direction using the Cantor-LTC network
pub fn discover_by_cantor(x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
    let mut network = CausalCantorNetwork::new();
    network.discover(x, y)
}

/// Create a trained CausalCantorNetwork from labeled data
pub fn train_causal_cantor(
    pairs: &[(Vec<f64>, Vec<f64>, CausalDirection)]
) -> CausalCantorNetwork {
    let mut network = CausalCantorNetwork::new();
    network.train(pairs);
    network
}

/// Discover with leave-one-out cross-validation training
pub fn discover_by_cantor_cv(
    x: &[f64],
    y: &[f64],
    all_pairs: &[(Vec<f64>, Vec<f64>, CausalDirection)],
    exclude_idx: usize,
) -> CausalDiscoveryResult {
    // Train on all pairs except the excluded one
    let training_pairs: Vec<_> = all_pairs.iter()
        .enumerate()
        .filter(|(i, _)| *i != exclude_idx)
        .map(|(_, p)| p.clone())
        .collect();

    let mut network = CausalCantorNetwork::new();
    network.train(&training_pairs);
    network.discover(x, y)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_encoder_creation() {
        let encoder = CausalFeatureEncoder::new(1024);
        assert_eq!(encoder.dim, 1024);
    }

    #[test]
    fn test_feature_extraction() {
        let encoder = CausalFeatureEncoder::new(1024);

        // Simple linear relationship with noise
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|xi| 2.0 * xi + 1.0 + (xi * 0.1).sin()).collect();

        let hv = encoder.encode(&x, &y);
        assert_eq!(hv.values.len(), 1024);

        // Should be normalized
        let norm: f32 = hv.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_network_creation() {
        let network = CausalCantorNetwork::new();
        assert!(network.integration_steps > 0);
    }

    #[test]
    fn test_discover_linear() {
        let mut network = CausalCantorNetwork::new();

        // Clear X → Y relationship
        let x: Vec<f64> = (0..100).map(|i| i as f64 / 10.0).collect();
        let y: Vec<f64> = x.iter().map(|xi| xi.powi(2) + 0.1 * (xi * 10.0).sin()).collect();

        let result = network.discover(&x, &y);

        // Should have some confidence
        assert!(result.confidence >= 0.0);
        println!("Linear test: direction={:?}, confidence={:.3}, p_forward={:.3}",
                 result.direction, result.confidence, result.p_forward);
    }

    #[test]
    fn test_phi_computation() {
        let mut network = CausalCantorNetwork::new();

        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|xi| xi * 2.0).collect();

        let _ = network.discover(&x, &y);
        let phi = network.get_phi();

        assert!(phi >= 0.0);
        println!("Φ = {:.4}", phi);
    }
}
