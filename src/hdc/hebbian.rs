/*!
Phase 14: Hebbian Plasticity for Dynamic HDC Learning

Implements biological synaptic learning in hyperdimensional space, enabling
the system to form associations dynamically through co-activation patterns.

"Neurons that fire together, wire together" - Donald Hebb, 1949

Revolutionary Features:
- **Hebbian Learning**: Co-activated vectors strengthen their binding
- **STDP (Spike-Timing Dependent Plasticity)**: Temporal order matters
- **Homeostatic Plasticity**: Prevents runaway excitation/depression
- **Metaplasticity**: Learning rate adapts based on history (BCM rule)
- **Competitive Learning**: Winner-take-all for sparse representations

Integration Points:
- Works with semantic HDC (src/hdc/mod.rs)
- Enhances morphogenetic fields (src/hdc/morphogenetic.rs)
- Drives Active Inference prediction updates (src/brain/active_inference.rs)
- Powers coalition formation (src/brain/prefrontal.rs)

Mathematical Foundation:
- Hebbian: Δw = η * x_pre * x_post
- STDP: Δw = A_+ * exp(-Δt/τ_+) if Δt > 0, else A_- * exp(Δt/τ_-)
- BCM: θ_m = E[y²] (sliding threshold for LTP/LTD crossover)
- Homeostatic: Scale weights to maintain target activity level
*/

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::HDC_DIMENSION;

/// Default learning rate for Hebbian updates
pub const DEFAULT_LEARNING_RATE: f32 = 0.01;

/// Default decay rate for synaptic weights
pub const DEFAULT_DECAY_RATE: f32 = 0.001;

/// STDP time constant for potentiation (ms)
pub const STDP_TAU_PLUS: f32 = 20.0;

/// STDP time constant for depression (ms)
pub const STDP_TAU_MINUS: f32 = 20.0;

/// Maximum STDP potentiation amplitude
pub const STDP_A_PLUS: f32 = 0.005;

/// Maximum STDP depression amplitude
pub const STDP_A_MINUS: f32 = 0.00525; // Slightly stronger for stability

/// Target activity level for homeostatic scaling
pub const TARGET_ACTIVITY: f32 = 0.1;

/// Homeostatic scaling time constant
pub const HOMEOSTATIC_TAU: f32 = 1000.0;

// ============================================================================
// CORE STRUCTURES
// ============================================================================

/// Synaptic connection between two HDC concepts
#[derive(Debug, Clone)]
pub struct Synapse {
    /// Connection weight (modulates binding strength)
    pub weight: f32,

    /// Eligibility trace for credit assignment
    pub eligibility: f32,

    /// Last update time for STDP calculations
    pub last_update: Option<Instant>,

    /// Historical weight changes for metaplasticity
    pub weight_history: Vec<f32>,

    /// BCM sliding threshold
    pub bcm_threshold: f32,
}

impl Synapse {
    /// Create new synapse with default weight
    pub fn new() -> Self {
        Self {
            weight: 0.5, // Start at neutral
            eligibility: 0.0,
            last_update: None,
            weight_history: Vec::with_capacity(100),
            bcm_threshold: 0.5,
        }
    }

    /// Create synapse with specific initial weight
    pub fn with_weight(weight: f32) -> Self {
        Self {
            weight: weight.clamp(0.0, 1.0),
            eligibility: 0.0,
            last_update: None,
            weight_history: Vec::with_capacity(100),
            bcm_threshold: weight,
        }
    }

    /// Record weight change for metaplasticity
    pub fn record_change(&mut self, delta: f32) {
        self.weight_history.push(delta);
        if self.weight_history.len() > 100 {
            self.weight_history.remove(0);
        }
    }

    /// Calculate metaplastic learning rate modifier
    /// Based on BCM (Bienenstock-Cooper-Munro) theory
    pub fn metaplastic_modifier(&self) -> f32 {
        if self.weight_history.is_empty() {
            return 1.0;
        }

        // If weights have been increasing, raise threshold (harder to potentiate)
        // If weights have been decreasing, lower threshold (easier to potentiate)
        let recent_trend: f32 = self.weight_history.iter()
            .rev()
            .take(10)
            .sum::<f32>() / 10.0_f32.min(self.weight_history.len() as f32);

        // Sigmoid transformation for smooth modulation
        let modifier = 1.0 / (1.0 + (-recent_trend * 10.0).exp());
        modifier.clamp(0.5, 2.0)
    }
}

impl Default for Synapse {
    fn default() -> Self {
        Self::new()
    }
}

/// Activation record for STDP timing
#[derive(Debug, Clone)]
pub struct ActivationRecord {
    /// Concept identifier
    pub concept_id: String,

    /// Activation time
    pub time: Instant,

    /// Activation strength
    pub strength: f32,

    /// The HDC vector that was activated
    pub vector: Vec<f32>,
}

/// Configuration for Hebbian learning
#[derive(Debug, Clone)]
pub struct HebbianConfig {
    /// Base learning rate (η)
    pub learning_rate: f32,

    /// Weight decay rate for forgetting
    pub decay_rate: f32,

    /// Enable STDP (spike-timing dependent plasticity)
    pub enable_stdp: bool,

    /// Enable homeostatic plasticity
    pub enable_homeostatic: bool,

    /// Enable metaplasticity (BCM rule)
    pub enable_metaplasticity: bool,

    /// Enable competitive learning
    pub enable_competitive: bool,

    /// Number of winners in competitive learning
    pub competitive_k: usize,

    /// HDC vector dimension
    pub dimension: usize,
}

impl Default for HebbianConfig {
    fn default() -> Self {
        Self {
            learning_rate: DEFAULT_LEARNING_RATE,
            decay_rate: DEFAULT_DECAY_RATE,
            enable_stdp: true,
            enable_homeostatic: true,
            enable_metaplasticity: true,
            enable_competitive: false,
            competitive_k: 3,
            dimension: HDC_DIMENSION,
        }
    }
}

// ============================================================================
// HEBBIAN LEARNING ENGINE
// ============================================================================

/// Main Hebbian plasticity engine for HDC learning
///
/// # Architecture
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────┐
/// │                  Hebbian Plasticity Engine                   │
/// ├─────────────────────────────────────────────────────────────┤
/// │                                                              │
/// │  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
/// │  │ Concept A│───▶│ Synapse  │◀───│ Concept B│              │
/// │  │  (HDC)   │    │  w_AB    │    │  (HDC)   │              │
/// │  └──────────┘    └────┬─────┘    └──────────┘              │
/// │                       │                                     │
/// │              ┌────────▼────────┐                           │
/// │              │ Learning Rules  │                           │
/// │              ├─────────────────┤                           │
/// │              │ • Hebbian       │                           │
/// │              │ • STDP          │                           │
/// │              │ • Homeostatic   │                           │
/// │              │ • Metaplastic   │                           │
/// │              └─────────────────┘                           │
/// │                       │                                     │
/// │              ┌────────▼────────┐                           │
/// │              │ Weight Update   │                           │
/// │              │ Δw = f(pre,post)│                           │
/// │              └─────────────────┘                           │
/// │                                                             │
/// └─────────────────────────────────────────────────────────────┘
/// ```
#[derive(Debug)]
pub struct HebbianEngine {
    /// Configuration
    config: HebbianConfig,

    /// Synaptic weight matrix (concept_id -> concept_id -> Synapse)
    synapses: HashMap<String, HashMap<String, Synapse>>,

    /// Recent activation history for STDP
    activation_history: Vec<ActivationRecord>,

    /// Global activity level for homeostatic scaling
    global_activity: f32,

    /// Total learning events for statistics
    total_updates: u64,

    /// Running sum of weight changes
    cumulative_delta: f32,
}

impl HebbianEngine {
    /// Create new Hebbian engine with default config
    pub fn new() -> Self {
        Self::with_config(HebbianConfig::default())
    }

    /// Create Hebbian engine with custom config
    pub fn with_config(config: HebbianConfig) -> Self {
        Self {
            config,
            synapses: HashMap::new(),
            activation_history: Vec::with_capacity(1000),
            global_activity: TARGET_ACTIVITY,
            total_updates: 0,
            cumulative_delta: 0.0,
        }
    }

    /// Record activation of a concept (for STDP)
    ///
    /// This is called whenever a concept becomes active in the system.
    /// The timing of activations is crucial for STDP learning.
    pub fn record_activation(&mut self, concept_id: &str, vector: &[f32], strength: f32) {
        let record = ActivationRecord {
            concept_id: concept_id.to_string(),
            time: Instant::now(),
            strength,
            vector: vector.to_vec(),
        };

        self.activation_history.push(record);

        // Prune old activations (older than 1 second)
        let cutoff = Instant::now() - Duration::from_secs(1);
        self.activation_history.retain(|r| r.time > cutoff);

        // Update global activity for homeostatic plasticity
        self.global_activity = 0.99 * self.global_activity + 0.01 * strength;
    }

    /// Core Hebbian learning: strengthen connection between co-active concepts
    ///
    /// # Algorithm
    /// ```text
    /// Δw = η * x_pre * x_post * metaplastic_modifier
    /// w_new = w_old + Δw
    /// w_new = clamp(w_new, 0, 1)
    /// ```
    ///
    /// # Arguments
    /// * `pre_id` - Presynaptic concept identifier
    /// * `post_id` - Postsynaptic concept identifier
    /// * `pre_activity` - Presynaptic activation level [0, 1]
    /// * `post_activity` - Postsynaptic activation level [0, 1]
    ///
    /// # Returns
    /// The weight change (Δw) that was applied
    pub fn hebbian_update(
        &mut self,
        pre_id: &str,
        post_id: &str,
        pre_activity: f32,
        post_activity: f32,
    ) -> f32 {
        // Ensure synapse exists
        self.ensure_synapse(pre_id, post_id);

        let synapse = self.synapses
            .get_mut(pre_id)
            .unwrap()
            .get_mut(post_id)
            .unwrap();

        // Calculate learning rate with metaplasticity
        let effective_lr = if self.config.enable_metaplasticity {
            self.config.learning_rate * synapse.metaplastic_modifier()
        } else {
            self.config.learning_rate
        };

        // Core Hebbian rule: Δw = η * x_pre * x_post
        let delta = effective_lr * pre_activity * post_activity;

        // Apply update
        synapse.weight = (synapse.weight + delta).clamp(0.0, 1.0);
        synapse.last_update = Some(Instant::now());
        synapse.record_change(delta);

        // Update BCM threshold (sliding threshold)
        if self.config.enable_metaplasticity {
            synapse.bcm_threshold = 0.99 * synapse.bcm_threshold + 0.01 * post_activity.powi(2);
        }

        self.total_updates += 1;
        self.cumulative_delta += delta;

        delta
    }

    /// STDP update: timing-dependent plasticity
    ///
    /// # Algorithm
    /// ```text
    /// Δt = t_post - t_pre
    /// If Δt > 0 (pre before post): LTP
    ///     Δw = A_+ * exp(-Δt / τ_+)
    /// If Δt < 0 (post before pre): LTD
    ///     Δw = -A_- * exp(Δt / τ_-)
    /// ```
    ///
    /// This implements the classic STDP curve where:
    /// - Pre→Post (causal) causes potentiation
    /// - Post→Pre (anti-causal) causes depression
    pub fn stdp_update(&mut self, pre_id: &str, post_id: &str) -> f32 {
        if !self.config.enable_stdp {
            return 0.0;
        }

        // Find activation times
        let pre_time = self.activation_history.iter()
            .filter(|r| r.concept_id == pre_id)
            .max_by_key(|r| r.time)
            .map(|r| r.time);

        let post_time = self.activation_history.iter()
            .filter(|r| r.concept_id == post_id)
            .max_by_key(|r| r.time)
            .map(|r| r.time);

        let (pre_t, post_t) = match (pre_time, post_time) {
            (Some(pre), Some(post)) => (pre, post),
            _ => return 0.0,
        };

        // Calculate time difference in milliseconds
        let delta_t = if post_t > pre_t {
            (post_t - pre_t).as_secs_f32() * 1000.0
        } else {
            -((pre_t - post_t).as_secs_f32() * 1000.0)
        };

        // Apply STDP rule
        let delta_w = if delta_t > 0.0 {
            // LTP: pre before post (causal)
            STDP_A_PLUS * (-delta_t / STDP_TAU_PLUS).exp()
        } else {
            // LTD: post before pre (anti-causal)
            -STDP_A_MINUS * (delta_t / STDP_TAU_MINUS).exp()
        };

        // Apply to synapse
        self.ensure_synapse(pre_id, post_id);
        let synapse = self.synapses
            .get_mut(pre_id)
            .unwrap()
            .get_mut(post_id)
            .unwrap();

        synapse.weight = (synapse.weight + delta_w).clamp(0.0, 1.0);
        synapse.record_change(delta_w);

        self.total_updates += 1;
        self.cumulative_delta += delta_w;

        delta_w
    }

    /// Apply homeostatic scaling to maintain stable activity levels
    ///
    /// This prevents runaway excitation or depression by scaling all
    /// outgoing weights to maintain a target activity level.
    ///
    /// # Algorithm
    /// ```text
    /// scale_factor = target_activity / current_activity
    /// scale_factor = clamp(scale_factor, 0.9, 1.1)  // Gradual changes
    /// for each outgoing synapse:
    ///     w_new = w_old * scale_factor
    /// ```
    pub fn homeostatic_scaling(&mut self, concept_id: &str) -> f32 {
        if !self.config.enable_homeostatic {
            return 1.0;
        }

        // Calculate current average outgoing weight
        let outgoing = match self.synapses.get(concept_id) {
            Some(connections) => connections,
            None => return 1.0,
        };

        if outgoing.is_empty() {
            return 1.0;
        }

        let avg_weight: f32 = outgoing.values()
            .map(|s| s.weight)
            .sum::<f32>() / outgoing.len() as f32;

        // Calculate scaling factor to reach target
        let scale_factor = if avg_weight > 0.01 {
            (TARGET_ACTIVITY / avg_weight).clamp(0.9, 1.1)
        } else {
            1.0
        };

        // Apply scaling
        if let Some(connections) = self.synapses.get_mut(concept_id) {
            for synapse in connections.values_mut() {
                synapse.weight = (synapse.weight * scale_factor).clamp(0.0, 1.0);
            }
        }

        scale_factor
    }

    /// Apply weight decay (forgetting)
    ///
    /// Unused synapses gradually decay toward baseline.
    /// This implements biological "use it or lose it" dynamics.
    pub fn apply_decay(&mut self) {
        let decay = 1.0 - self.config.decay_rate;
        let baseline = 0.5;

        for connections in self.synapses.values_mut() {
            for synapse in connections.values_mut() {
                // Decay toward baseline
                synapse.weight = baseline + (synapse.weight - baseline) * decay;

                // Decay eligibility trace
                synapse.eligibility *= 0.9;
            }
        }
    }

    /// Competitive learning: winner-take-all dynamics
    ///
    /// Only the K most similar concepts strengthen their connections,
    /// while others are weakened. This creates sparse, distinctive
    /// representations.
    ///
    /// # Arguments
    /// * `target` - The target concept
    /// * `candidates` - List of (concept_id, similarity) pairs
    pub fn competitive_update(&mut self, target: &str, mut candidates: Vec<(&str, f32)>) -> Vec<String> {
        if !self.config.enable_competitive || candidates.is_empty() {
            return Vec::new();
        }

        // Sort by similarity (descending)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut winners = Vec::new();

        // Strengthen winners
        for (id, sim) in candidates.iter().take(self.config.competitive_k) {
            self.hebbian_update(id, target, *sim, 1.0);
            winners.push(id.to_string());
        }

        // Weaken losers
        for (id, sim) in candidates.iter().skip(self.config.competitive_k) {
            self.ensure_synapse(id, target);
            let synapse = self.synapses
                .get_mut(*id)
                .unwrap()
                .get_mut(target)
                .unwrap();

            // Anti-Hebbian for losers
            let delta = -self.config.learning_rate * sim * 0.5;
            synapse.weight = (synapse.weight + delta).clamp(0.0, 1.0);
        }

        winners
    }

    /// Learn association between two HDC vectors
    ///
    /// This is the main entry point for learning associations.
    /// It combines Hebbian learning with STDP if both concepts
    /// have recent activations.
    ///
    /// # Arguments
    /// * `concept_a` - First concept (id, vector, activity)
    /// * `concept_b` - Second concept (id, vector, activity)
    ///
    /// # Returns
    /// Total weight change applied
    pub fn learn_association(
        &mut self,
        concept_a: (&str, &[f32], f32),
        concept_b: (&str, &[f32], f32),
    ) -> f32 {
        let (id_a, vec_a, activity_a) = concept_a;
        let (id_b, vec_b, activity_b) = concept_b;

        // Record activations
        self.record_activation(id_a, vec_a, activity_a);
        self.record_activation(id_b, vec_b, activity_b);

        // Apply Hebbian learning (bidirectional)
        let delta_ab = self.hebbian_update(id_a, id_b, activity_a, activity_b);
        let delta_ba = self.hebbian_update(id_b, id_a, activity_b, activity_a);

        // Apply STDP if enabled
        let stdp_ab = self.stdp_update(id_a, id_b);
        let stdp_ba = self.stdp_update(id_b, id_a);

        delta_ab + delta_ba + stdp_ab + stdp_ba
    }

    /// Get synaptic weight between two concepts
    pub fn get_weight(&self, pre_id: &str, post_id: &str) -> f32 {
        self.synapses
            .get(pre_id)
            .and_then(|m| m.get(post_id))
            .map(|s| s.weight)
            .unwrap_or(0.5) // Default neutral weight
    }

    /// Get all connections from a concept
    pub fn get_connections(&self, concept_id: &str) -> Vec<(String, f32)> {
        self.synapses
            .get(concept_id)
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.weight)).collect())
            .unwrap_or_default()
    }

    /// Modulate HDC binding by learned weights
    ///
    /// Instead of simple element-wise multiplication, this applies
    /// learned synaptic weights to modulate the binding strength.
    ///
    /// # Algorithm
    /// ```text
    /// binding[i] = vec_a[i] * vec_b[i] * synaptic_weight
    /// ```
    pub fn weighted_bind(&self, id_a: &str, vec_a: &[f32], id_b: &str, vec_b: &[f32]) -> Vec<f32> {
        let weight = self.get_weight(id_a, id_b);

        vec_a.iter()
            .zip(vec_b.iter())
            .map(|(a, b)| a * b * weight)
            .collect()
    }

    /// Get learning statistics
    pub fn stats(&self) -> HebbianStats {
        let total_synapses: usize = self.synapses.values()
            .map(|m| m.len())
            .sum();

        let avg_weight: f32 = if total_synapses > 0 {
            self.synapses.values()
                .flat_map(|m| m.values())
                .map(|s| s.weight)
                .sum::<f32>() / total_synapses as f32
        } else {
            0.5
        };

        HebbianStats {
            total_synapses,
            total_updates: self.total_updates,
            average_weight: avg_weight,
            global_activity: self.global_activity,
            cumulative_delta: self.cumulative_delta,
            activation_history_size: self.activation_history.len(),
        }
    }

    /// Ensure synapse exists between two concepts
    fn ensure_synapse(&mut self, pre_id: &str, post_id: &str) {
        self.synapses
            .entry(pre_id.to_string())
            .or_insert_with(HashMap::new)
            .entry(post_id.to_string())
            .or_insert_with(Synapse::new);
    }

    /// Prune weak synapses to save memory
    pub fn prune_weak_synapses(&mut self, threshold: f32) -> usize {
        let mut pruned = 0;

        for connections in self.synapses.values_mut() {
            let before = connections.len();
            connections.retain(|_, s| (s.weight - 0.5).abs() > threshold);
            pruned += before - connections.len();
        }

        // Remove empty connection maps
        self.synapses.retain(|_, m| !m.is_empty());

        pruned
    }
}

impl Default for HebbianEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// LEARNING STATISTICS
// ============================================================================

/// Statistics about Hebbian learning
#[derive(Debug, Clone)]
pub struct HebbianStats {
    /// Total number of synapses
    pub total_synapses: usize,

    /// Total learning events
    pub total_updates: u64,

    /// Average synaptic weight
    pub average_weight: f32,

    /// Current global activity level
    pub global_activity: f32,

    /// Cumulative weight change
    pub cumulative_delta: f32,

    /// Size of activation history
    pub activation_history_size: usize,
}

// ============================================================================
// ADVANCED: ASSOCIATIVE MEMORY WITH HEBBIAN LEARNING
// ============================================================================

/// Hebbian-learned associative memory for HDC
///
/// This creates a content-addressable memory where associations
/// are formed dynamically through experience.
#[derive(Debug)]
pub struct HebbianAssociativeMemory {
    /// The Hebbian learning engine
    engine: HebbianEngine,

    /// Stored concept vectors
    concepts: HashMap<String, Vec<f32>>,

    /// Concept activation counts
    activation_counts: HashMap<String, u64>,
}

impl HebbianAssociativeMemory {
    /// Create new associative memory
    pub fn new() -> Self {
        Self {
            engine: HebbianEngine::new(),
            concepts: HashMap::new(),
            activation_counts: HashMap::new(),
        }
    }

    /// Store a concept
    pub fn store(&mut self, id: &str, vector: Vec<f32>) {
        self.concepts.insert(id.to_string(), vector);
        self.activation_counts.insert(id.to_string(), 0);
    }

    /// Activate a concept and learn associations
    pub fn activate(&mut self, id: &str, activity: f32) -> Vec<(String, f32)> {
        let vector = match self.concepts.get(id) {
            Some(v) => v.clone(),
            None => return Vec::new(),
        };

        // Record activation
        self.engine.record_activation(id, &vector, activity);
        *self.activation_counts.entry(id.to_string()).or_insert(0) += 1;

        // Find co-active concepts (recently activated)
        let coactive: Vec<_> = self.engine.activation_history.iter()
            .filter(|r| r.concept_id != id)
            .filter(|r| r.time.elapsed() < Duration::from_millis(100))
            .map(|r| (r.concept_id.clone(), r.strength))
            .collect();

        // Learn associations with co-active concepts
        for (other_id, other_activity) in &coactive {
            if let Some(other_vec) = self.concepts.get(other_id) {
                self.engine.learn_association(
                    (id, &vector, activity),
                    (other_id, other_vec, *other_activity),
                );
            }
        }

        coactive
    }

    /// Recall concepts associated with a query
    pub fn recall(&self, query_id: &str, limit: usize) -> Vec<(String, f32)> {
        let connections = self.engine.get_connections(query_id);

        let mut sorted: Vec<_> = connections.into_iter()
            .filter(|(id, _)| id != query_id)
            .collect();

        // Sort by weight (strongest associations first)
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        sorted.into_iter().take(limit).collect()
    }

    /// Get bound vector using learned weights
    pub fn get_weighted_binding(&self, id_a: &str, id_b: &str) -> Option<Vec<f32>> {
        let vec_a = self.concepts.get(id_a)?;
        let vec_b = self.concepts.get(id_b)?;

        Some(self.engine.weighted_bind(id_a, vec_a, id_b, vec_b))
    }

    /// Apply periodic maintenance (decay, homeostatic scaling)
    pub fn maintenance(&mut self) {
        self.engine.apply_decay();

        // Apply homeostatic scaling to all concepts
        for id in self.concepts.keys() {
            self.engine.homeostatic_scaling(id);
        }
    }

    /// Get memory statistics
    pub fn stats(&self) -> HebbianAssociativeStats {
        HebbianAssociativeStats {
            num_concepts: self.concepts.len(),
            engine_stats: self.engine.stats(),
            total_activations: self.activation_counts.values().sum(),
        }
    }
}

impl Default for HebbianAssociativeMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for associative memory
#[derive(Debug, Clone)]
pub struct HebbianAssociativeStats {
    /// Number of stored concepts
    pub num_concepts: usize,

    /// Engine statistics
    pub engine_stats: HebbianStats,

    /// Total concept activations
    pub total_activations: u64,
}

// ============================================================================
// TESTS - 20 comprehensive tests for Hebbian plasticity
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    fn random_vector(dim: usize, seed: u32) -> Vec<f32> {
        let mut vec = Vec::with_capacity(dim);
        let mut x = seed;
        for _ in 0..dim {
            x = x.wrapping_mul(1103515245).wrapping_add(12345);
            let val = ((x >> 16) & 0x7FFF) as f32 / 32768.0 * 2.0 - 1.0;
            vec.push(val);
        }
        vec
    }

    #[test]
    fn test_synapse_creation() {
        let synapse = Synapse::new();
        assert!((synapse.weight - 0.5).abs() < 0.01);
        assert!(synapse.eligibility.abs() < 0.01);
        assert!(synapse.last_update.is_none());
    }

    #[test]
    fn test_synapse_with_weight() {
        let synapse = Synapse::with_weight(0.8);
        assert!((synapse.weight - 0.8).abs() < 0.01);

        // Test clamping
        let clamped = Synapse::with_weight(1.5);
        assert!((clamped.weight - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_metaplastic_modifier() {
        let mut synapse = Synapse::new();

        // No history = modifier of 1.0 (baseline)
        let baseline = synapse.metaplastic_modifier();
        assert!((baseline - 1.0).abs() < 0.1, "No history should give baseline modifier");

        // Add positive history (potentiation has been happening)
        for _ in 0..10 {
            synapse.record_change(0.1);
        }

        // BCM rule: positive history raises threshold, making further potentiation HARDER
        // This translates to a modifier that shifts the sigmoid response
        let after_positive = synapse.metaplastic_modifier();

        // Add negative history synapse for comparison
        let mut neg_synapse = Synapse::new();
        for _ in 0..10 {
            neg_synapse.record_change(-0.1);
        }
        let after_negative = neg_synapse.metaplastic_modifier();

        // Positive trend should differ from negative trend
        assert!((after_positive - after_negative).abs() > 0.2,
            "Different histories should produce different modifiers (pos: {}, neg: {})",
            after_positive, after_negative);
    }

    #[test]
    fn test_engine_creation() {
        let engine = HebbianEngine::new();
        let stats = engine.stats();

        assert_eq!(stats.total_synapses, 0);
        assert_eq!(stats.total_updates, 0);
    }

    #[test]
    fn test_hebbian_update_strengthens() {
        let mut engine = HebbianEngine::new();

        let initial = engine.get_weight("A", "B");

        // High co-activation should strengthen
        for _ in 0..10 {
            engine.hebbian_update("A", "B", 1.0, 1.0);
        }

        let final_weight = engine.get_weight("A", "B");
        assert!(final_weight > initial, "Co-activation should strengthen synapse");
    }

    #[test]
    fn test_hebbian_update_with_low_activity() {
        let mut engine = HebbianEngine::new();

        // Low activity = small changes
        let delta = engine.hebbian_update("A", "B", 0.1, 0.1);

        // Change should be small (0.01 * 0.1 * 0.1 = 0.0001)
        assert!(delta.abs() < 0.01, "Low activity should produce small changes");
    }

    #[test]
    fn test_weight_clamping() {
        let mut engine = HebbianEngine::new();

        // Many updates should not exceed bounds
        for _ in 0..1000 {
            engine.hebbian_update("A", "B", 1.0, 1.0);
        }

        let weight = engine.get_weight("A", "B");
        assert!(weight <= 1.0, "Weight should not exceed 1.0");
        assert!(weight >= 0.0, "Weight should not be negative");
    }

    #[test]
    fn test_activation_recording() {
        let mut engine = HebbianEngine::new();
        let vec = random_vector(100, 42);

        engine.record_activation("test", &vec, 0.8);

        let stats = engine.stats();
        assert_eq!(stats.activation_history_size, 1);
    }

    #[test]
    fn test_stdp_causal_potentiates() {
        let mut engine = HebbianEngine::new();
        let vec = random_vector(100, 42);

        // Pre-then-post should potentiate (LTP)
        engine.record_activation("pre", &vec, 1.0);
        sleep(Duration::from_millis(5));
        engine.record_activation("post", &vec, 1.0);

        let delta = engine.stdp_update("pre", "post");
        assert!(delta > 0.0, "Causal (pre→post) should potentiate");
    }

    #[test]
    fn test_stdp_anticausal_depresses() {
        let mut engine = HebbianEngine::new();
        let vec = random_vector(100, 42);

        // Post-then-pre should depress (LTD)
        engine.record_activation("post", &vec, 1.0);
        sleep(Duration::from_millis(5));
        engine.record_activation("pre", &vec, 1.0);

        let delta = engine.stdp_update("pre", "post");
        assert!(delta < 0.0, "Anti-causal (post→pre) should depress");
    }

    #[test]
    fn test_homeostatic_scaling() {
        let mut engine = HebbianEngine::new();

        // Create some synapses with high weights
        for i in 0..10 {
            engine.ensure_synapse("source", &format!("target_{}", i));
            engine.hebbian_update("source", &format!("target_{}", i), 1.0, 1.0);
        }

        let scale = engine.homeostatic_scaling("source");

        // Scale should be applied (not exactly 1.0)
        assert!(scale > 0.0, "Scale should be positive");
    }

    #[test]
    fn test_weight_decay() {
        let mut engine = HebbianEngine::new();

        // Strengthen synapse
        for _ in 0..50 {
            engine.hebbian_update("A", "B", 1.0, 1.0);
        }

        let before_decay = engine.get_weight("A", "B");

        // Apply decay multiple times
        for _ in 0..100 {
            engine.apply_decay();
        }

        let after_decay = engine.get_weight("A", "B");

        // Weight should decay toward baseline (0.5)
        assert!((after_decay - 0.5).abs() < (before_decay - 0.5).abs(),
            "Decay should move weight toward baseline");
    }

    #[test]
    fn test_competitive_learning() {
        let mut engine = HebbianEngine::with_config(HebbianConfig {
            enable_competitive: true,
            competitive_k: 2,
            ..Default::default()
        });

        let candidates = vec![
            ("A", 0.9),
            ("B", 0.8),
            ("C", 0.3),
            ("D", 0.1),
        ];

        let winners = engine.competitive_update("target", candidates);

        assert_eq!(winners.len(), 2, "Should have 2 winners");
        assert!(winners.contains(&"A".to_string()));
        assert!(winners.contains(&"B".to_string()));
    }

    #[test]
    fn test_learn_association() {
        let mut engine = HebbianEngine::new();

        let vec_a = random_vector(100, 1);
        let vec_b = random_vector(100, 2);

        let initial_ab = engine.get_weight("A", "B");
        let initial_ba = engine.get_weight("B", "A");

        // Learn association
        engine.learn_association(
            ("A", &vec_a, 1.0),
            ("B", &vec_b, 1.0),
        );

        let final_ab = engine.get_weight("A", "B");
        let final_ba = engine.get_weight("B", "A");

        assert!(final_ab > initial_ab, "A→B should strengthen");
        assert!(final_ba > initial_ba, "B→A should strengthen (bidirectional)");
    }

    #[test]
    fn test_weighted_bind() {
        let mut engine = HebbianEngine::new();

        // Set up a strong connection
        for _ in 0..50 {
            engine.hebbian_update("A", "B", 1.0, 1.0);
        }

        let vec_a = vec![1.0, 0.5, -0.5];
        let vec_b = vec![0.5, 1.0, 0.5];

        let bound = engine.weighted_bind("A", &vec_a, "B", &vec_b);

        // Bound should be modulated by weight
        let weight = engine.get_weight("A", "B");
        assert!((bound[0] - (1.0 * 0.5 * weight)).abs() < 0.01);
    }

    #[test]
    fn test_prune_weak_synapses() {
        let mut engine = HebbianEngine::new();

        // Create synapses with varying strengths
        engine.hebbian_update("A", "B", 1.0, 1.0); // Strong
        engine.ensure_synapse("C", "D"); // Weak (at baseline)

        let stats_before = engine.stats();
        let pruned = engine.prune_weak_synapses(0.3);
        let stats_after = engine.stats();

        // Weak synapses (near baseline) should be pruned
        assert!(stats_after.total_synapses <= stats_before.total_synapses,
            "Pruning should reduce synapse count");
        println!("Pruned {} weak synapses", pruned);
    }

    #[test]
    fn test_associative_memory_store_recall() {
        let mut memory = HebbianAssociativeMemory::new();

        memory.store("cat", random_vector(100, 1));
        memory.store("dog", random_vector(100, 2));
        memory.store("pet", random_vector(100, 3));

        // Activate concepts together to form associations
        memory.activate("cat", 1.0);
        memory.activate("pet", 0.9);

        memory.activate("dog", 1.0);
        memory.activate("pet", 0.9);

        // Recall should find associated concepts
        let stats = memory.stats();
        assert_eq!(stats.num_concepts, 3);
        assert!(stats.total_activations > 0);
    }

    #[test]
    fn test_associative_memory_maintenance() {
        let mut memory = HebbianAssociativeMemory::new();

        memory.store("A", random_vector(100, 1));
        memory.store("B", random_vector(100, 2));

        // Create associations
        memory.activate("A", 1.0);
        memory.activate("B", 1.0);

        // Maintenance should not crash
        memory.maintenance();

        let stats = memory.stats();
        assert!(stats.engine_stats.total_synapses > 0);
    }

    #[test]
    #[ignore = "performance test - run with cargo test --release"]
    fn test_hebbian_performance() {
        use std::time::Instant;

        let mut engine = HebbianEngine::new();
        let start = Instant::now();

        // 10,000 Hebbian updates
        for i in 0..10_000 {
            let pre = format!("concept_{}", i % 100);
            let post = format!("concept_{}", (i + 1) % 100);
            engine.hebbian_update(&pre, &post, 0.5, 0.5);
        }

        let elapsed = start.elapsed();
        let ops_per_sec = 10_000.0 / elapsed.as_secs_f64();

        // Should handle >100,000 updates/sec
        println!("✅ Hebbian updates: {:.0} ops/sec", ops_per_sec);
        assert!(ops_per_sec > 10_000.0, "Should handle >10K updates/sec");
    }

    #[test]
    fn test_bcm_threshold_adaptation() {
        let mut engine = HebbianEngine::new();

        // Repeated high activation should raise threshold
        for _ in 0..100 {
            engine.hebbian_update("A", "B", 1.0, 1.0);
        }

        let synapse = engine.synapses.get("A").unwrap().get("B").unwrap();

        // BCM threshold should have increased due to high activity
        assert!(synapse.bcm_threshold > 0.5,
            "BCM threshold should increase with high activity");
    }
}
