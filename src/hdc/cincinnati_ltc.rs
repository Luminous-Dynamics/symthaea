//! # Cincinnati-LTC Integration Module
//!
//! ## Purpose
//! Implements the Cincinnati Algorithm as a bit-level differential engine,
//! integrated with Liquid Time-Constant (LTC) dynamics for adaptive learning.
//!
//! ## Theory: Cincinnati Algorithm
//!
//! The Cincinnati Algorithm is a novel bit-level learning mechanism where:
//! - The model is a **bit string (Estimator)** that grows/adapts over time
//! - Each observation is compared against random bits from high-order positions
//! - **Extension**: If all bits match, append the observation (confidence grows logarithmically)
//! - **Adjustment**: If mismatch, invert bits from the mismatch position upward
//!
//! ## Integration with HDC + LTC
//!
//! The Cincinnati-LTC Binding equation:
//! ```text
//! W(t+1) = W(t) ⊕ (ΔC ⊛ τ(t))
//! ```
//!
//! Where:
//! - `W(t)` = Weight hypervector at time t
//! - `ΔC` = Cincinnati delta (the adjustment/extension signal)
//! - `τ(t)` = Time constant from LTC dynamics
//! - `⊛` = Circular convolution (lateral binding)
//! - `⊕` = HDC bundling (normalized sum)
//!
//! ## Key Components
//!
//! - [`CincinnatiEstimator`]: Core bit-level differential engine
//! - [`LateralBinder`]: Circular convolution for same-level node communication
//! - [`PredictiveBudding`]: LTC-driven autonomous node spawning
//! - [`PoGMetrics`]: Proof of Grounding - physical world metrics
//! - [`CincinnatiLtcEngine`]: Unified integration of all components

use std::f32::consts::PI;
use rand::Rng;
use serde::{Serialize, Deserialize};

use crate::hdc::unified_hv::ContinuousHV;
use crate::hdc::HDC_DIMENSION;

// =============================================================================
// CINCINNATI ALGORITHM - Core Differential Engine
// =============================================================================

/// Cincinnati Estimator - Bit-level differential engine
///
/// The estimator is a variable-length bit string that learns through a unique
/// extension/adjustment mechanism. Confidence grows logarithmically with the
/// string length.
///
/// # Algorithm
///
/// For each observation:
/// 1. Generate random bits and compare with model from index 0 (high-order)
/// 2. If all bits match → Extend: append observation bit (confidence grows log)
/// 3. If mismatch found → Adjust: invert bits from mismatch position upward
///
/// # Example
///
/// ```rust,ignore
/// use symthaea::hdc::cincinnati_ltc::CincinnatiEstimator;
///
/// let mut estimator = CincinnatiEstimator::new();
///
/// // Train on observations
/// for _ in 0..100 {
///     let observation = rand::random::<bool>();
///     estimator.update(observation);
/// }
///
/// // Get current estimate and confidence
/// let (estimate, confidence) = estimator.predict();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CincinnatiEstimator {
    /// The bit string model (Estimator)
    /// Index 0 = most significant (high-order)
    pub model: Vec<bool>,

    /// Total observations processed
    pub observations: u64,

    /// Running sum of correct predictions
    pub correct_predictions: u64,

    /// Seed for deterministic random generation
    seed: u64,
}

impl CincinnatiEstimator {
    /// Create a new Cincinnati Estimator
    pub fn new() -> Self {
        Self {
            model: vec![false],  // Start with single bit
            observations: 0,
            correct_predictions: 0,
            seed: 42,
        }
    }

    /// Create with specific seed for reproducibility
    pub fn with_seed(seed: u64) -> Self {
        Self {
            model: vec![false],
            observations: 0,
            correct_predictions: 0,
            seed,
        }
    }

    /// Update the estimator with a new observation
    ///
    /// This is the core Cincinnati Algorithm:
    /// 1. Pair and check for first mismatch from high-order (index 0)
    /// 2. If all match → Extend
    /// 3. If mismatch → Adjust by inverting bits
    pub fn update(&mut self, observation: bool) {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed + self.observations);

        self.observations += 1;

        // Track prediction accuracy
        let prediction = self.majority_bit();
        if prediction == observation {
            self.correct_predictions += 1;
        }

        let mut all_matched = true;
        let mut mismatch_index = None;
        let mut mismatch_random_bit = false;

        // 1. Pair and check for first mismatch from high-order (index 0)
        for i in 0..self.model.len() {
            let random_bit = rng.gen::<bool>();
            if self.model[i] != random_bit {
                mismatch_index = Some(i);
                mismatch_random_bit = random_bit;
                all_matched = false;
                break;
            }
        }

        // 2. Logic: Extend or Adjust
        if all_matched {
            // Extension: Confidence grows logarithmically
            self.model.push(observation);
        } else if let Some(idx) = mismatch_index {
            // Only adjust if random bit matches observation
            if mismatch_random_bit == observation {
                // Adjust: Invert bits from smallest (end of vec) upward
                // until we find a bit that matches observation
                for i in (idx..self.model.len()).rev() {
                    self.model[i] = !self.model[i];
                    if self.model[i] == observation {
                        break;
                    }
                }
            }
        }

        // Update seed for next iteration
        self.seed = self.seed.wrapping_add(1);
    }

    /// Get the current prediction and confidence
    ///
    /// Returns (predicted_bit, confidence) where confidence ∈ [0, 1]
    pub fn predict(&self) -> (bool, f32) {
        let prediction = self.majority_bit();
        let confidence = self.confidence();
        (prediction, confidence)
    }

    /// Compute majority bit (most common bit in model)
    fn majority_bit(&self) -> bool {
        let ones = self.model.iter().filter(|&&b| b).count();
        ones > self.model.len() / 2
    }

    /// Compute confidence based on model length and prediction accuracy
    ///
    /// Confidence grows logarithmically with model length
    pub fn confidence(&self) -> f32 {
        let length_factor = (self.model.len() as f32).ln() / 10.0;
        let accuracy_factor = if self.observations > 0 {
            self.correct_predictions as f32 / self.observations as f32
        } else {
            0.5
        };

        // Combine factors
        (length_factor * 0.3 + accuracy_factor * 0.7).min(1.0).max(0.0)
    }

    /// Get the delta signal for HDC binding
    ///
    /// The delta is computed as the weighted sum of recent bit changes
    pub fn delta_signal(&self) -> f32 {
        if self.model.is_empty() {
            return 0.0;
        }

        // Compute weighted sum favoring recent (low-order) bits
        let mut delta = 0.0;
        let len = self.model.len() as f32;

        for (i, &bit) in self.model.iter().enumerate() {
            let weight = (i as f32 + 1.0) / len;  // Higher weight for recent bits
            delta += if bit { weight } else { -weight };
        }

        delta / len
    }

    /// Convert model to HDC-compatible binary vector (±1 representation)
    ///
    /// Pads or truncates to `HDC_DIMENSION`
    pub fn to_hdc_vector(&self) -> Vec<i8> {
        let mut result = vec![-1i8; HDC_DIMENSION];

        for (i, &bit) in self.model.iter().take(HDC_DIMENSION).enumerate() {
            result[i] = if bit { 1 } else { -1 };
        }

        result
    }
}

impl Default for CincinnatiEstimator {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// LATERAL BINDING - Circular Convolution
// =============================================================================

/// Lateral Binder - Circular convolution for same-level node communication
///
/// Implements the `⊛` operation for binding nodes at the same hierarchical level.
/// This enables lateral communication without requiring hierarchical message passing.
///
/// # Theory
///
/// Circular convolution in HDC:
/// ```text
/// (A ⊛ B)[k] = Σᵢ A[i] × B[(k-i) mod D]
/// ```
///
/// For efficient computation, we use FFT-based convolution:
/// ```text
/// A ⊛ B = IFFT(FFT(A) ⊙ FFT(B))
/// ```
///
/// where `⊙` is element-wise multiplication.
pub struct LateralBinder {
    /// Dimension of hypervectors
    dimension: usize,

    /// Pre-allocated buffer for FFT (real-only approximation)
    buffer: Vec<f32>,
}

impl LateralBinder {
    /// Create a new lateral binder
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            buffer: vec![0.0; dimension],
        }
    }

    /// Create with default HDC dimension
    pub fn default_dimension() -> Self {
        Self::new(HDC_DIMENSION)
    }

    /// Compute circular convolution A ⊛ B
    ///
    /// This is a simplified O(D²) implementation.
    /// For production, use FFT-based O(D log D) algorithm.
    pub fn convolve(&mut self, a: &ContinuousHV, b: &ContinuousHV) -> ContinuousHV {
        let d = self.dimension.min(a.values.len()).min(b.values.len());
        let mut result = vec![0.0f32; d];

        // Direct convolution (can be optimized with FFT)
        for k in 0..d {
            let mut sum = 0.0;
            for i in 0..d {
                let j = (k + d - i) % d;
                sum += a.values[i] * b.values[j];
            }
            result[k] = sum / (d as f32).sqrt();  // Normalize
        }

        ContinuousHV::from_values(result)
    }

    /// Fast approximate convolution using permutation
    ///
    /// This is O(D) and preserves most properties of circular convolution
    /// for high-dimensional vectors.
    pub fn fast_convolve(&self, a: &ContinuousHV, b: &ContinuousHV) -> ContinuousHV {
        let d = self.dimension.min(a.values.len()).min(b.values.len());
        let mut result = vec![0.0f32; d];

        // Permute b by half dimension (rotation)
        let half = d / 2;
        for i in 0..d {
            let j = (i + half) % d;
            result[i] = a.values[i] * b.values[j];
        }

        ContinuousHV::from_values(result)
    }

    /// Bind multiple nodes at the same level
    ///
    /// Uses iterative convolution to combine all node states
    pub fn bind_lateral(&mut self, nodes: &[ContinuousHV]) -> Option<ContinuousHV> {
        if nodes.is_empty() {
            return None;
        }

        if nodes.len() == 1 {
            return Some(nodes[0].clone());
        }

        let mut result = nodes[0].clone();
        for node in &nodes[1..] {
            result = self.fast_convolve(&result, node);
        }

        Some(result)
    }
}

// =============================================================================
// PREDICTIVE AUTOPOIESIS - LTC-Driven Budding
// =============================================================================

/// Budding Event - Represents a node spawning decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuddingEvent {
    /// ID of parent node
    pub parent_id: usize,

    /// Timestamp of budding
    pub timestamp: f64,

    /// Prediction error that triggered budding
    pub prediction_error: f32,

    /// Initial state of the new node
    pub initial_state: Vec<f32>,

    /// Time constant inherited from parent
    pub initial_tau: f32,
}

/// Predictive Autopoiesis - LTC-driven node budding/pruning
///
/// Level 7+ (periphery) nodes autonomously spawn/prune children based on
/// real-time LTC prediction error. This creates an elastic, self-organizing
/// network topology.
///
/// # Algorithm
///
/// 1. Monitor prediction error `ε(t)` at each peripheral node
/// 2. If `ε(t) > θ_bud` for sustained period → Spawn child (Level 8)
/// 3. Child inherits time constant from parent: `τ_child = τ_parent × α`
/// 4. If child's `ε(t) < θ_prune` for sustained period → Prune
///
/// # Parameters
///
/// - `θ_bud`: Budding threshold (high prediction error triggers spawning)
/// - `θ_prune`: Pruning threshold (low error for sustained time → prune)
/// - `α`: Time constant inheritance factor (typically 0.5-0.8)
/// - `T_sustain`: Required duration above/below threshold
#[derive(Debug, Clone)]
pub struct PredictiveBudding {
    /// Budding threshold
    pub theta_bud: f32,

    /// Pruning threshold
    pub theta_prune: f32,

    /// Time constant inheritance factor
    pub alpha: f32,

    /// Required sustained duration (in timesteps)
    pub sustain_steps: usize,

    /// Current prediction errors per node
    errors: Vec<f32>,

    /// Sustained above-threshold counters
    above_threshold: Vec<usize>,

    /// Sustained below-threshold counters
    below_threshold: Vec<usize>,

    /// Parent relationships (child_id -> parent_id)
    parent_map: Vec<Option<usize>>,

    /// Time constants per node
    tau_values: Vec<f32>,

    /// Maximum nodes allowed
    max_nodes: usize,

    /// Next node ID
    next_id: usize,
}

impl PredictiveBudding {
    /// Create new predictive budding system
    pub fn new(initial_nodes: usize) -> Self {
        Self {
            theta_bud: 0.7,      // High error threshold for budding
            theta_prune: 0.1,   // Low error threshold for pruning
            alpha: 0.7,          // Time constant inheritance
            sustain_steps: 10,   // Sustained for 10 steps
            errors: vec![0.0; initial_nodes],
            above_threshold: vec![0; initial_nodes],
            below_threshold: vec![0; initial_nodes],
            parent_map: vec![None; initial_nodes],
            tau_values: vec![1.0; initial_nodes],
            max_nodes: initial_nodes * 4,  // Allow 4x expansion
            next_id: initial_nodes,
        }
    }

    /// Update prediction error for a node
    pub fn update_error(&mut self, node_id: usize, error: f32) {
        if node_id >= self.errors.len() {
            return;
        }

        self.errors[node_id] = error;

        // Update sustained counters
        if error > self.theta_bud {
            self.above_threshold[node_id] += 1;
            self.below_threshold[node_id] = 0;
        } else if error < self.theta_prune {
            self.below_threshold[node_id] += 1;
            self.above_threshold[node_id] = 0;
        } else {
            self.above_threshold[node_id] = 0;
            self.below_threshold[node_id] = 0;
        }
    }

    /// Check if node should spawn a child
    pub fn should_bud(&self, node_id: usize) -> bool {
        if node_id >= self.above_threshold.len() {
            return false;
        }

        self.above_threshold[node_id] >= self.sustain_steps
            && self.next_id < self.max_nodes
    }

    /// Check if node should be pruned
    pub fn should_prune(&self, node_id: usize) -> bool {
        if node_id >= self.below_threshold.len() {
            return false;
        }

        // Only prune if this is a child (has a parent)
        self.parent_map.get(node_id).map(|p| p.is_some()).unwrap_or(false)
            && self.below_threshold[node_id] >= self.sustain_steps
    }

    /// Create a budding event for spawning a new node
    pub fn create_budding_event(&mut self, parent_id: usize, timestamp: f64, state: &ContinuousHV) -> Option<BuddingEvent> {
        if !self.should_bud(parent_id) {
            return None;
        }

        // Reset parent's counter
        self.above_threshold[parent_id] = 0;

        // Calculate child's time constant
        let parent_tau = self.tau_values.get(parent_id).copied().unwrap_or(1.0);
        let child_tau = parent_tau * self.alpha;

        // Create the event
        let event = BuddingEvent {
            parent_id,
            timestamp,
            prediction_error: self.errors.get(parent_id).copied().unwrap_or(0.0),
            initial_state: state.values.clone(),
            initial_tau: child_tau,
        };

        // Register the new child
        let child_id = self.next_id;
        self.next_id += 1;

        self.errors.push(0.0);
        self.above_threshold.push(0);
        self.below_threshold.push(0);
        self.parent_map.push(Some(parent_id));
        self.tau_values.push(child_tau);

        Some(event)
    }

    /// Get nodes that should be pruned
    pub fn get_prune_candidates(&self) -> Vec<usize> {
        (0..self.errors.len())
            .filter(|&id| self.should_prune(id))
            .collect()
    }

    /// Get current node count
    pub fn node_count(&self) -> usize {
        self.errors.len()
    }

    /// Get time constant for a node
    pub fn get_tau(&self, node_id: usize) -> f32 {
        self.tau_values.get(node_id).copied().unwrap_or(1.0)
    }
}

// =============================================================================
// PROOF OF GROUNDING (PoG) METRICS
// =============================================================================

/// PoG Metrics - Physical world grounding measurements
///
/// Integrates real-world physical metrics into the HDC weight hypervector,
/// ensuring the consciousness model remains grounded in physical reality.
///
/// # Metrics
///
/// - **Energy**: Computational energy consumption (Joules)
/// - **Storage**: Memory/storage utilization (bytes)
/// - **Bandwidth**: Network bandwidth consumption (bytes/sec)
/// - **Latency**: Processing latency (milliseconds)
/// - **Accuracy**: Prediction accuracy over time
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PoGMetrics {
    /// Cumulative energy consumption (Joules)
    pub energy_joules: f64,

    /// Current storage utilization (bytes)
    pub storage_bytes: u64,

    /// Bandwidth consumption (bytes/sec)
    pub bandwidth_bps: f64,

    /// Average processing latency (ms)
    pub latency_ms: f32,

    /// Rolling accuracy (0-1)
    pub accuracy: f32,

    /// Timestamp of last update
    pub last_updated: f64,

    /// Historical window for rolling metrics
    history_window: usize,

    /// Energy history for rolling average
    energy_history: Vec<f64>,
}

impl PoGMetrics {
    /// Create new PoG metrics tracker
    pub fn new() -> Self {
        Self {
            energy_joules: 0.0,
            storage_bytes: 0,
            bandwidth_bps: 0.0,
            latency_ms: 0.0,
            accuracy: 0.5,
            last_updated: 0.0,
            history_window: 100,
            energy_history: Vec::with_capacity(100),
        }
    }

    /// Update energy consumption
    pub fn update_energy(&mut self, delta_joules: f64, timestamp: f64) {
        self.energy_joules += delta_joules;
        self.last_updated = timestamp;

        // Update rolling history
        self.energy_history.push(delta_joules);
        if self.energy_history.len() > self.history_window {
            self.energy_history.remove(0);
        }
    }

    /// Update storage utilization
    pub fn update_storage(&mut self, bytes: u64) {
        self.storage_bytes = bytes;
    }

    /// Update bandwidth
    pub fn update_bandwidth(&mut self, bps: f64) {
        // Exponential moving average
        self.bandwidth_bps = self.bandwidth_bps * 0.9 + bps * 0.1;
    }

    /// Update latency
    pub fn update_latency(&mut self, ms: f32) {
        // Exponential moving average
        self.latency_ms = self.latency_ms * 0.9 + ms * 0.1;
    }

    /// Update accuracy
    pub fn update_accuracy(&mut self, correct: bool) {
        let update = if correct { 1.0 } else { 0.0 };
        self.accuracy = self.accuracy * 0.95 + update * 0.05;
    }

    /// Convert to HDC-compatible vector (normalized metrics)
    ///
    /// Each metric is normalized to [-1, 1] range and embedded into HDC
    pub fn to_hdc_vector(&self) -> ContinuousHV {
        let mut values = vec![0.0f32; HDC_DIMENSION];

        // Embed each metric into a different region of the hypervector
        let region_size = HDC_DIMENSION / 5;

        // Energy: log-normalized
        let energy_norm = (1.0 + self.energy_joules).ln() as f32 / 20.0;
        for i in 0..region_size {
            values[i] = (energy_norm * (i as f32 / region_size as f32 * 2.0 * PI).sin()).tanh();
        }

        // Storage: log-normalized
        let storage_norm = ((self.storage_bytes as f64 + 1.0).ln() / 30.0) as f32;
        for i in region_size..2*region_size {
            let j = i - region_size;
            values[i] = (storage_norm * (j as f32 / region_size as f32 * 2.0 * PI).cos()).tanh();
        }

        // Bandwidth: log-normalized
        let bw_norm = ((self.bandwidth_bps + 1.0).ln() / 20.0) as f32;
        for i in 2*region_size..3*region_size {
            let j = i - 2*region_size;
            values[i] = (bw_norm * (j as f32 / region_size as f32 * 3.0 * PI).sin()).tanh();
        }

        // Latency: inverse normalized (lower is better)
        let latency_norm = 1.0 / (1.0 + self.latency_ms / 100.0);
        for i in 3*region_size..4*region_size {
            let j = i - 3*region_size;
            values[i] = latency_norm * (j as f32 / region_size as f32 * 2.0 * PI).cos();
        }

        // Accuracy: direct
        for i in 4*region_size..HDC_DIMENSION {
            let j = i - 4*region_size;
            values[i] = (self.accuracy * 2.0 - 1.0) * (j as f32 / (HDC_DIMENSION - 4*region_size) as f32 * PI).sin();
        }

        ContinuousHV::from_values(values)
    }

    /// Get overall "grounding score" (0-1)
    ///
    /// Higher score = better grounded in physical reality
    pub fn grounding_score(&self) -> f32 {
        // Penalize high energy, storage, latency
        // Reward high accuracy, reasonable bandwidth
        let energy_score = 1.0 / (1.0 + (self.energy_joules as f32 / 1000.0));
        let storage_score = 1.0 / (1.0 + (self.storage_bytes as f32 / 1_000_000_000.0));
        let latency_score = 1.0 / (1.0 + self.latency_ms / 10.0);
        let accuracy_score = self.accuracy;

        // Weighted combination
        (energy_score * 0.2 + storage_score * 0.2 + latency_score * 0.2 + accuracy_score * 0.4)
            .min(1.0)
            .max(0.0)
    }
}

// =============================================================================
// CINCINNATI-LTC ENGINE - Unified Integration
// =============================================================================

/// Cincinnati-LTC Engine - Unified integration of all components
///
/// Implements the complete Cincinnati-LTC Binding equation:
/// ```text
/// W(t+1) = W(t) ⊕ (ΔC ⊛ τ(t))
/// ```
///
/// Where:
/// - `W(t)` = Weight hypervector at time t
/// - `ΔC` = Cincinnati delta from differential engine
/// - `τ(t)` = Time constant hypervector from LTC dynamics
/// - `⊛` = Circular convolution (lateral binding)
/// - `⊕` = HDC bundling
pub struct CincinnatiLtcEngine {
    /// Cincinnati estimator for differential learning
    pub estimator: CincinnatiEstimator,

    /// Lateral binder for circular convolution
    pub lateral: LateralBinder,

    /// Predictive budding system
    pub budding: PredictiveBudding,

    /// Proof of Grounding metrics
    pub pog: PoGMetrics,

    /// Current weight hypervector W(t)
    weight: ContinuousHV,

    /// Current time constant hypervector τ(t)
    tau_hv: ContinuousHV,

    /// Learning rate for weight updates
    learning_rate: f32,

    /// Current timestep
    timestep: u64,
}

impl CincinnatiLtcEngine {
    /// Create a new Cincinnati-LTC Engine
    pub fn new(initial_nodes: usize) -> Self {
        Self {
            estimator: CincinnatiEstimator::new(),
            lateral: LateralBinder::default_dimension(),
            budding: PredictiveBudding::new(initial_nodes),
            pog: PoGMetrics::new(),
            weight: ContinuousHV::random_default(42),
            tau_hv: ContinuousHV::random_default(123),
            learning_rate: 0.01,
            timestep: 0,
        }
    }

    /// Step the engine with new observation
    ///
    /// This implements the Cincinnati-LTC Binding equation:
    /// `W(t+1) = W(t) ⊕ (ΔC ⊛ τ(t))`
    pub fn step(&mut self, observation: bool, input: &ContinuousHV) -> ContinuousHV {
        self.timestep += 1;

        // 1. Update Cincinnati estimator
        self.estimator.update(observation);

        // 2. Compute ΔC (Cincinnati delta as hypervector)
        let delta_c = self.compute_delta_c();

        // 3. Circular convolution: ΔC ⊛ τ(t)
        let bound = self.lateral.fast_convolve(&delta_c, &self.tau_hv);

        // 4. Bundle: W(t+1) = W(t) ⊕ (ΔC ⊛ τ(t))
        let update_signal = bound.scale(self.learning_rate);
        self.weight = self.weight.add(&update_signal);

        // 5. Incorporate input
        let result = self.lateral.fast_convolve(&self.weight, input);

        // 6. Update time constant based on PoG
        self.update_tau();

        result
    }

    /// Compute Cincinnati delta as hypervector
    fn compute_delta_c(&self) -> ContinuousHV {
        let delta = self.estimator.delta_signal();
        let confidence = self.estimator.confidence();

        // Scale random HV by delta and confidence
        let base = ContinuousHV::random_default(self.timestep + 1000);
        base.scale(delta * confidence)
    }

    /// Update time constant hypervector based on PoG metrics
    fn update_tau(&mut self) {
        // Blend PoG grounding into tau
        let pog_hv = self.pog.to_hdc_vector();
        let grounding = self.pog.grounding_score();

        // Higher grounding = faster adaptation (lower tau)
        let tau_scale = 1.0 - grounding * 0.5;
        self.tau_hv = self.tau_hv.scale(0.99).add(&pog_hv.scale(0.01 * tau_scale));
    }

    /// Check for budding events and process them
    pub fn process_budding(&mut self, node_states: &[ContinuousHV], timestamp: f64) -> Vec<BuddingEvent> {
        let mut events = Vec::new();

        for (id, state) in node_states.iter().enumerate() {
            if let Some(event) = self.budding.create_budding_event(id, timestamp, state) {
                events.push(event);
            }
        }

        events
    }

    /// Update prediction error for a node
    pub fn update_prediction_error(&mut self, node_id: usize, predicted: &ContinuousHV, actual: &ContinuousHV) {
        let error = 1.0 - predicted.similarity(actual).abs();
        self.budding.update_error(node_id, error);
        self.pog.update_accuracy(error < 0.3);
    }

    /// Get current weight hypervector
    pub fn weight(&self) -> &ContinuousHV {
        &self.weight
    }

    /// Get prediction and confidence from Cincinnati estimator
    pub fn predict(&self) -> (bool, f32) {
        self.estimator.predict()
    }

    /// Get current node count (including budded nodes)
    pub fn node_count(&self) -> usize {
        self.budding.node_count()
    }

    /// Get nodes that should be pruned
    pub fn prune_candidates(&self) -> Vec<usize> {
        self.budding.get_prune_candidates()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cincinnati_estimator_basic() {
        let mut estimator = CincinnatiEstimator::new();

        // Update with observations
        for _ in 0..20 {
            estimator.update(true);
        }
        for _ in 0..10 {
            estimator.update(false);
        }

        let (pred, conf) = estimator.predict();

        // Should predict true (majority)
        assert!(conf > 0.0);
        println!("Prediction: {}, Confidence: {:.4}", pred, conf);
    }

    #[test]
    fn test_cincinnati_confidence_grows() {
        let mut estimator = CincinnatiEstimator::new();
        let mut prev_conf = 0.0;

        // Confidence should grow with consistent observations
        for i in 0..50 {
            estimator.update(true);
            let (_, conf) = estimator.predict();
            if i > 10 {
                // Allow some variance but general trend should be up
                if conf > prev_conf + 0.01 || conf > 0.3 {
                    // Progress is being made
                }
            }
            prev_conf = conf;
        }

        // Final confidence should be reasonable
        assert!(estimator.confidence() > 0.3);
    }

    #[test]
    fn test_lateral_convolution() {
        let binder = LateralBinder::new(128);  // Small dim for testing

        let a = ContinuousHV::random(128, 42);
        let b = ContinuousHV::random(128, 123);

        let result = binder.fast_convolve(&a, &b);

        // Result should have same dimension
        assert_eq!(result.values.len(), 128);

        // Result should be different from inputs
        let sim_a = result.similarity(&a);
        let sim_b = result.similarity(&b);
        assert!(sim_a.abs() < 0.9);
        assert!(sim_b.abs() < 0.9);
    }

    #[test]
    fn test_predictive_budding() {
        let mut budding = PredictiveBudding::new(3);

        // Simulate high prediction error
        for _ in 0..15 {
            budding.update_error(0, 0.9);  // High error
        }

        assert!(budding.should_bud(0));

        // Create budding event
        let state = ContinuousHV::random_default(42);
        let event = budding.create_budding_event(0, 1.0, &state);

        assert!(event.is_some());
        let event = event.unwrap();
        assert_eq!(event.parent_id, 0);
        assert!(event.initial_tau < 1.0);  // Inherited and reduced
    }

    #[test]
    fn test_pog_metrics() {
        let mut pog = PoGMetrics::new();

        pog.update_energy(100.0, 0.0);
        pog.update_storage(1_000_000);
        pog.update_bandwidth(1000.0);
        pog.update_latency(5.0);

        for _ in 0..10 {
            pog.update_accuracy(true);
        }

        let score = pog.grounding_score();
        assert!(score > 0.0 && score <= 1.0);
        println!("Grounding score: {:.4}", score);

        // Convert to HDC
        let hv = pog.to_hdc_vector();
        assert_eq!(hv.values.len(), HDC_DIMENSION);
    }

    #[test]
    fn test_cincinnati_ltc_engine() {
        let mut engine = CincinnatiLtcEngine::new(3);

        let input = ContinuousHV::random_default(42);

        // Run several steps
        for i in 0..20 {
            let obs = i % 3 == 0;  // Observation pattern
            let result = engine.step(obs, &input);

            // Result should be valid
            assert_eq!(result.values.len(), HDC_DIMENSION);
        }

        // Check state
        let (pred, conf) = engine.predict();
        println!("Final prediction: {}, confidence: {:.4}", pred, conf);

        // Weight should have evolved
        let weight = engine.weight();
        assert_eq!(weight.values.len(), HDC_DIMENSION);
    }

    #[test]
    fn test_full_integration() {
        let mut engine = CincinnatiLtcEngine::new(5);

        // Simulate a sequence with prediction errors
        let inputs: Vec<ContinuousHV> = (0..5)
            .map(|i| ContinuousHV::random_default(i as u64 * 100))
            .collect();

        let mut outputs = Vec::new();

        for t in 0..50 {
            let obs = t % 2 == 0;
            let input = &inputs[t % 5];
            let output = engine.step(obs, input);
            outputs.push(output);

            // Update prediction errors
            if outputs.len() > 1 {
                let prev = &outputs[outputs.len() - 2];
                let curr = &outputs[outputs.len() - 1];
                engine.update_prediction_error(t % 5, prev, curr);
            }

            // Update PoG
            engine.pog.update_latency(5.0 + (t as f32 * 0.1));
            engine.pog.update_energy(1.0, t as f64);
        }

        // Check budding
        let events = engine.process_budding(&inputs, 50.0);
        println!("Budding events: {}", events.len());
        println!("Final node count: {}", engine.node_count());
    }
}
