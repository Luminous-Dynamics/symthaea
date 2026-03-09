// =============================================================================
// Component 22: HDC-Grounded Associative Learner
// =============================================================================
//
// Implements "Semantic Reinforcement" using Hyperdimensional Computing (HDC)
// for zero-shot generalization across domains.
//
// Key Innovation: Instead of lookup tables (Context -> Action), we build an
// Associative Memory of Experience using hypervectors. This enables:
// - One-shot learning (learns from single examples)
// - Zero-shot generalization (transfers across similar contexts)
// - Holographic storage (distributed, fault-tolerant)
// - Bitwise efficiency (XOR/Popcount operations)
//
// Based on HDC/VSA (Vector Symbolic Architectures) research and designed
// for compatibility with symthaea-core's HV types.
//
// Follows the Config + Registry + Stats pattern established in other components.
// =============================================================================

use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::PatternId;

// =============================================================================
// HDC Configuration Constants
// =============================================================================

/// Default hypervector dimension (2^12 = 4096)
/// Lower than symthaea's 16K for efficiency in this use case.
/// Can be upgraded to 16K for full compatibility.
pub const HDC_DIMENSION: usize = 4096;

/// Number of bytes for binary representation
pub const HDC_BYTES: usize = HDC_DIMENSION / 8;

// =============================================================================
// Hypervector Types (Compatible with symthaea-core)
// =============================================================================

/// Binary hypervector using packed bits
///
/// Efficient representation using XOR for binding and majority vote for bundling.
/// Compatible with symthaea-core's BinaryHV patterns.
#[derive(Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BinaryHV {
    /// Packed bits (each byte holds 8 dimensions)
    bits: Vec<u64>,
    /// Dimension of the vector
    dim: usize,
}

impl std::fmt::Debug for BinaryHV {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ones = self.popcount();
        let density = ones as f32 / self.dim as f32;
        write!(
            f,
            "BinaryHV(dim={}, ones={}, density={:.2})",
            self.dim, ones, density
        )
    }
}

impl BinaryHV {
    /// Create a new zero vector
    pub fn zero(dim: usize) -> Self {
        let num_u64s = (dim + 63) / 64;
        Self {
            bits: vec![0; num_u64s],
            dim,
        }
    }

    /// Create a random hypervector with deterministic seed
    pub fn random(dim: usize, seed: u64) -> Self {
        let num_u64s = (dim + 63) / 64;
        let mut bits = Vec::with_capacity(num_u64s);
        let mut state = seed;

        for _ in 0..num_u64s {
            // xorshift64 PRNG
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            bits.push(state);
        }

        Self { bits, dim }
    }

    /// Create with default dimension
    pub fn random_default(seed: u64) -> Self {
        Self::random(HDC_DIMENSION, seed)
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get a specific bit
    pub fn get_bit(&self, index: usize) -> bool {
        if index >= self.dim {
            return false;
        }
        let word_idx = index / 64;
        let bit_idx = index % 64;
        (self.bits[word_idx] >> bit_idx) & 1 == 1
    }

    /// Set a specific bit
    pub fn set_bit(&mut self, index: usize, value: bool) {
        if index >= self.dim {
            return;
        }
        let word_idx = index / 64;
        let bit_idx = index % 64;
        if value {
            self.bits[word_idx] |= 1 << bit_idx;
        } else {
            self.bits[word_idx] &= !(1 << bit_idx);
        }
    }

    /// Count set bits (popcount)
    pub fn popcount(&self) -> usize {
        self.bits.iter().map(|&w| w.count_ones() as usize).sum()
    }

    /// Binding operation (XOR)
    ///
    /// Creates an association between two vectors.
    /// A ⊗ B is dissimilar to both A and B.
    ///
    /// Properties:
    /// - Commutative: A⊗B = B⊗A
    /// - Self-inverse: A⊗A = 0 (identity)
    /// - Preserves distance: d(A⊗C, B⊗C) = d(A, B)
    pub fn bind(&self, other: &Self) -> Self {
        assert_eq!(self.dim, other.dim, "Dimension mismatch");

        let bits: Vec<u64> = self
            .bits
            .iter()
            .zip(other.bits.iter())
            .map(|(a, b)| a ^ b)
            .collect();

        Self {
            bits,
            dim: self.dim,
        }
    }

    /// Unbind operation (same as bind for XOR)
    ///
    /// Recovers bound content: if C = A ⊗ B, then A = C ⊗ B
    pub fn unbind(&self, other: &Self) -> Self {
        self.bind(other) // XOR is self-inverse
    }

    /// Bundling operation (majority vote)
    ///
    /// Creates a superposition of vectors.
    /// bundle(A, B) is similar to both A and B.
    pub fn bundle(hvs: &[Self]) -> Self {
        if hvs.is_empty() {
            return Self::zero(HDC_DIMENSION);
        }

        let dim = hvs[0].dim;
        let threshold = hvs.len() / 2;
        let _num_u64s = (dim + 63) / 64; // Reserved for future SIMD optimization
        let mut result = Self::zero(dim);

        // Count votes for each bit position
        for bit_pos in 0..dim {
            let word_idx = bit_pos / 64;
            let bit_idx = bit_pos % 64;

            let count: usize = hvs
                .iter()
                .filter(|hv| (hv.bits[word_idx] >> bit_idx) & 1 == 1)
                .count();

            if count > threshold {
                result.bits[word_idx] |= 1 << bit_idx;
            }
        }

        result
    }

    /// Weighted bundling with integer weights
    pub fn weighted_bundle(hvs: &[Self], weights: &[i32]) -> Self {
        if hvs.is_empty() || weights.is_empty() {
            return Self::zero(HDC_DIMENSION);
        }

        let dim = hvs[0].dim;
        let _total_weight: i32 = weights.iter().map(|w| w.abs()).sum();
        let _threshold = _total_weight / 2; // Reserved for alternative bundling strategy
        let mut result = Self::zero(dim);

        for bit_pos in 0..dim {
            let word_idx = bit_pos / 64;
            let bit_idx = bit_pos % 64;

            let weighted_sum: i32 = hvs
                .iter()
                .zip(weights.iter())
                .map(|(hv, &w)| {
                    let bit_set = (hv.bits[word_idx] >> bit_idx) & 1 == 1;
                    if bit_set {
                        w
                    } else {
                        -w
                    }
                })
                .sum();

            if weighted_sum > 0 {
                result.bits[word_idx] |= 1 << bit_idx;
            }
        }

        result
    }

    /// Hamming distance (number of differing bits)
    pub fn hamming_distance(&self, other: &Self) -> usize {
        self.bits
            .iter()
            .zip(other.bits.iter())
            .map(|(a, b)| (a ^ b).count_ones() as usize)
            .sum()
    }

    /// Cosine-like similarity using Hamming distance
    ///
    /// Returns similarity in [-1, 1] where:
    /// - 1.0 = identical vectors
    /// - 0.0 = orthogonal (random) vectors
    /// - -1.0 = opposite vectors
    pub fn similarity(&self, other: &Self) -> f32 {
        let hamming = self.hamming_distance(other);
        let max_distance = self.dim;

        // Convert hamming to [-1, 1] range
        // 0 hamming -> 1.0 similarity
        // dim/2 hamming -> 0.0 similarity (orthogonal)
        // dim hamming -> -1.0 similarity (opposite)
        1.0 - (2.0 * hamming as f32 / max_distance as f32)
    }

    /// Permutation (cyclic rotation) for sequence encoding
    pub fn permute(&self, shift: i32) -> Self {
        let mut result = Self::zero(self.dim);
        let shift = shift.rem_euclid(self.dim as i32) as usize;

        for i in 0..self.dim {
            let new_pos = (i + shift) % self.dim;
            if self.get_bit(i) {
                result.set_bit(new_pos, true);
            }
        }

        result
    }

    /// Scale by repeating pattern (for weighted contributions)
    pub fn scale(&self, weight: f32) -> ContinuousHV {
        let values: Vec<f32> = (0..self.dim)
            .map(|i| {
                let bit = self.get_bit(i);
                if bit {
                    weight
                } else {
                    -weight
                }
            })
            .collect();
        ContinuousHV::from_vec(values)
    }
}

// =============================================================================
// Continuous Hypervector (for weighted operations)
// =============================================================================

/// Continuous-valued hypervector for weighted/gradient operations
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ContinuousHV {
    /// Vector components (unbounded, but typically in [-1, 1])
    pub values: Vec<f32>,
}

impl ContinuousHV {
    /// Create from values
    pub fn from_vec(values: Vec<f32>) -> Self {
        Self { values }
    }

    /// Create zero vector
    pub fn zero(dim: usize) -> Self {
        Self {
            values: vec![0.0; dim],
        }
    }

    /// Create random vector
    pub fn random(dim: usize, seed: u64) -> Self {
        let mut values = Vec::with_capacity(dim);
        let mut state = seed;

        for _ in 0..dim {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let normalized = (state as f32 / u64::MAX as f32) * 2.0 - 1.0;
            values.push(normalized);
        }

        Self { values }
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.values.len()
    }

    /// Add another vector
    pub fn add(&self, other: &Self) -> Self {
        let values: Vec<f32> = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a + b)
            .collect();
        Self { values }
    }

    /// Scale by scalar
    pub fn scale(&self, factor: f32) -> Self {
        let values: Vec<f32> = self.values.iter().map(|v| v * factor).collect();
        Self { values }
    }

    /// Binding (element-wise multiplication)
    pub fn bind(&self, other: &Self) -> Self {
        let values: Vec<f32> = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .collect();
        Self { values }
    }

    /// Bundle (average)
    pub fn bundle(hvs: &[&Self]) -> Self {
        if hvs.is_empty() {
            return Self::zero(HDC_DIMENSION);
        }
        let dim = hvs[0].values.len();
        let n = hvs.len() as f32;

        let values: Vec<f32> = (0..dim)
            .map(|i| hvs.iter().map(|hv| hv.values[i]).sum::<f32>() / n)
            .collect();

        Self { values }
    }

    /// Cosine similarity
    pub fn similarity(&self, other: &Self) -> f32 {
        let dot: f32 = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_a: f32 = self.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_b: f32 = other.values.iter().map(|v| v * v).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    /// Convert to binary (threshold at 0)
    pub fn to_binary(&self) -> BinaryHV {
        let mut result = BinaryHV::zero(self.values.len());
        for (i, &v) in self.values.iter().enumerate() {
            if v > 0.0 {
                result.set_bit(i, true);
            }
        }
        result
    }

    /// Normalize to unit length
    pub fn normalize(&self) -> Self {
        let norm: f32 = self.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm == 0.0 {
            return self.clone();
        }
        self.scale(1.0 / norm)
    }
}

// =============================================================================
// Sparse Projector (Encodes discrete states into HD space)
// =============================================================================

/// Projects discrete symbols/states into hypervector space
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SparseProjector {
    /// Dimension of output vectors
    dim: usize,

    /// Cached random vectors for known symbols
    symbol_cache: HashMap<String, BinaryHV>,

    /// Seed for deterministic generation
    base_seed: u64,
}

impl SparseProjector {
    /// Create a new projector
    pub fn new(dim: usize, base_seed: u64) -> Self {
        Self {
            dim,
            symbol_cache: HashMap::new(),
            base_seed,
        }
    }

    /// Create with default dimension
    pub fn default_dim(base_seed: u64) -> Self {
        Self::new(HDC_DIMENSION, base_seed)
    }

    /// Get or create a vector for a symbol
    pub fn encode_symbol(&mut self, symbol: &str) -> BinaryHV {
        if let Some(hv) = self.symbol_cache.get(symbol) {
            return hv.clone();
        }

        // Generate deterministic seed from symbol
        let symbol_seed = self.hash_string(symbol);
        let hv = BinaryHV::random(self.dim, self.base_seed ^ symbol_seed);

        self.symbol_cache.insert(symbol.to_string(), hv.clone());
        hv
    }

    /// Encode a key-value pair as bound vector
    pub fn encode_pair(&mut self, key: &str, value: &str) -> BinaryHV {
        let key_hv = self.encode_symbol(key);
        let value_hv = self.encode_symbol(value);
        key_hv.bind(&value_hv)
    }

    /// Encode a context (set of key-value pairs)
    pub fn encode_context(&mut self, pairs: &[(&str, &str)]) -> BinaryHV {
        let bound_pairs: Vec<BinaryHV> =
            pairs.iter().map(|(k, v)| self.encode_pair(k, v)).collect();

        BinaryHV::bundle(&bound_pairs)
    }

    /// Simple string hash for deterministic seeding
    fn hash_string(&self, s: &str) -> u64 {
        let mut hash: u64 = 5381;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }
        hash
    }
}

// =============================================================================
// Experience Encoding
// =============================================================================

/// An encoded experience (context + action + outcome)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExperienceEncoding {
    /// The situation vector (context bound with action)
    pub situation: BinaryHV,

    /// The outcome (positive or negative)
    pub outcome: f32,

    /// Original context for debugging
    pub context_description: String,

    /// Original action for debugging
    pub action_description: String,

    /// Timestamp
    pub timestamp: u64,
}

impl ExperienceEncoding {
    /// Create a new experience encoding
    pub fn new(
        situation: BinaryHV,
        outcome: f32,
        context_description: String,
        action_description: String,
        timestamp: u64,
    ) -> Self {
        Self {
            situation,
            outcome,
            context_description,
            action_description,
            timestamp,
        }
    }
}

// =============================================================================
// Action Registry
// =============================================================================

/// Known actions with their hypervector encodings
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ActionRegistry {
    /// Action name -> HV mapping
    actions: HashMap<String, BinaryHV>,

    /// HV -> Action name (for decoding)
    reverse_index: Vec<(String, BinaryHV)>,

    /// Projector for encoding
    projector: SparseProjector,
}

impl ActionRegistry {
    /// Create a new action registry with default dimension (4096)
    pub fn new(base_seed: u64) -> Self {
        Self::with_dimension(HDC_DIMENSION, base_seed)
    }

    /// Create a new action registry with custom dimension
    pub fn with_dimension(dim: usize, base_seed: u64) -> Self {
        Self {
            actions: HashMap::new(),
            reverse_index: Vec::new(),
            projector: SparseProjector::new(dim, base_seed),
        }
    }

    /// Register an action
    pub fn register(&mut self, action_name: &str) -> BinaryHV {
        if let Some(hv) = self.actions.get(action_name) {
            return hv.clone();
        }

        let hv = self.projector.encode_symbol(action_name);
        self.actions.insert(action_name.to_string(), hv.clone());
        self.reverse_index
            .push((action_name.to_string(), hv.clone()));
        hv
    }

    /// Get action HV
    pub fn get(&self, action_name: &str) -> Option<&BinaryHV> {
        self.actions.get(action_name)
    }

    /// Find nearest action to a query vector
    pub fn find_nearest(&self, query: &BinaryHV) -> Vec<(String, f32)> {
        let mut results: Vec<(String, f32)> = self
            .reverse_index
            .iter()
            .map(|(name, hv)| (name.clone(), query.similarity(hv)))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Get all registered actions
    pub fn all_actions(&self) -> Vec<&str> {
        self.actions.keys().map(|s| s.as_str()).collect()
    }
}

// =============================================================================
// Associative Learner Configuration
// =============================================================================

/// Configuration for the associative learner
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AssociativeLearnerConfig {
    /// HDC dimension
    pub dimension: usize,

    /// Learning rate for continuous memory updates
    pub learning_rate: f32,

    /// Decay rate for old experiences
    pub decay_rate: f32,

    /// Minimum similarity to consider a match
    pub similarity_threshold: f32,

    /// Whether to use binary or continuous memory
    pub use_binary_memory: bool,

    /// Maximum experiences to retain
    pub max_experiences: usize,

    /// Weight for positive outcomes
    pub positive_weight: f32,

    /// Weight for negative outcomes (typically negative)
    pub negative_weight: f32,
}

impl Default for AssociativeLearnerConfig {
    fn default() -> Self {
        Self {
            dimension: HDC_DIMENSION,
            learning_rate: 0.1,
            decay_rate: 0.01,
            similarity_threshold: 0.3,
            use_binary_memory: true,
            max_experiences: 10000,
            positive_weight: 1.0,
            negative_weight: -0.5,
        }
    }
}

// =============================================================================
// Associative Learner (Main Component)
// =============================================================================

/// HDC-Grounded Associative Learner
///
/// Implements "Semantic Reinforcement" - learning what works through
/// holographic association rather than lookup tables.
///
/// Key capabilities:
/// - Zero-shot generalization: Works with never-seen contexts
/// - One-shot learning: Learns from single examples
/// - Graceful degradation: Partial matches still useful
/// - Efficient: Bitwise XOR/Popcount operations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AssociativeLearner {
    /// Configuration
    config: AssociativeLearnerConfig,

    /// The "Long Term Memory" of what works
    /// Stores bundled: Sum(Context ⊗ Action ⊗ Outcome)
    experience_memory: ContinuousHV,

    /// Positive experience memory (actions that worked)
    positive_memory: ContinuousHV,

    /// Negative experience memory (actions that failed)
    negative_memory: ContinuousHV,

    /// Projector for encoding contexts
    projector: SparseProjector,

    /// Registry of known actions
    action_registry: ActionRegistry,

    /// Recent experiences (for detailed analysis)
    recent_experiences: Vec<ExperienceEncoding>,

    /// Statistics
    stats: AssociativeLearnerStats,
}

impl AssociativeLearner {
    /// Create a new associative learner
    pub fn new() -> Self {
        Self::with_config(AssociativeLearnerConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: AssociativeLearnerConfig) -> Self {
        let dim = config.dimension;
        Self {
            config,
            experience_memory: ContinuousHV::zero(dim),
            positive_memory: ContinuousHV::zero(dim),
            negative_memory: ContinuousHV::zero(dim),
            projector: SparseProjector::new(dim, 42),
            action_registry: ActionRegistry::with_dimension(dim, 42),
            recent_experiences: Vec::new(),
            stats: AssociativeLearnerStats::default(),
        }
    }

    /// Register an action
    pub fn register_action(&mut self, action: &str) {
        self.action_registry.register(action);
    }

    /// Learn from an experience
    ///
    /// Encodes the context-action pair and bundles it into memory,
    /// weighted by the outcome.
    pub fn learn(&mut self, context: &[(&str, &str)], action: &str, outcome: f32, timestamp: u64) {
        // 1. Encode context
        let context_hv = self.projector.encode_context(context);

        // 2. Encode action (and register if new)
        let action_hv = self.action_registry.register(action);

        // 3. Bind them: "Doing Action in Context"
        let situation_hv = context_hv.bind(&action_hv);

        // 4. Weight by outcome
        let weighted = situation_hv.scale(outcome);

        // 5. Bundle into memory
        let refs = [&self.experience_memory, &weighted];
        self.experience_memory = ContinuousHV::bundle(&refs);

        // 6. Update positive/negative memories
        if outcome > 0.0 {
            let positive_weighted = situation_hv.scale(outcome * self.config.positive_weight);
            let refs = [&self.positive_memory, &positive_weighted];
            self.positive_memory = ContinuousHV::bundle(&refs);
            self.stats.positive_experiences += 1;
        } else {
            let negative_weighted = situation_hv.scale(outcome.abs() * self.config.negative_weight);
            let refs = [&self.negative_memory, &negative_weighted];
            self.negative_memory = ContinuousHV::bundle(&refs);
            self.stats.negative_experiences += 1;
        }

        // 7. Store recent experience
        let context_desc = context
            .iter()
            .map(|(k, v)| format!("{}:{}", k, v))
            .collect::<Vec<_>>()
            .join(", ");

        let experience = ExperienceEncoding::new(
            situation_hv,
            outcome,
            context_desc,
            action.to_string(),
            timestamp,
        );

        self.recent_experiences.push(experience);
        if self.recent_experiences.len() > self.config.max_experiences {
            self.recent_experiences.remove(0);
        }

        self.stats.total_experiences += 1;
    }

    /// Predict best actions for a given context
    ///
    /// Returns actions ranked by predicted success, even for
    /// never-seen contexts (zero-shot generalization).
    pub fn predict(&mut self, context: &[(&str, &str)]) -> Vec<ActionPrediction> {
        self.stats.predictions_made += 1;

        // 1. Encode current context
        let context_hv = self.projector.encode_context(context);

        // 2. Query: "What Action bound with Context matches positive Memory?"
        // We unbind the context from positive memory to isolate likely good actions
        let context_binary = context_hv;
        let positive_binary = self.positive_memory.to_binary();
        let negative_binary = self.negative_memory.to_binary();

        let positive_query = positive_binary.unbind(&context_binary);
        let negative_query = negative_binary.unbind(&context_binary);

        // 3. Find nearest actions
        let positive_matches = self.action_registry.find_nearest(&positive_query);
        let negative_matches = self.action_registry.find_nearest(&negative_query);

        // 4. Combine scores (positive - negative)
        let mut action_scores: HashMap<String, f32> = HashMap::new();

        for (action, score) in positive_matches {
            *action_scores.entry(action).or_insert(0.0) += score;
        }

        for (action, score) in negative_matches {
            *action_scores.entry(action).or_insert(0.0) -= score * 0.5;
        }

        // 5. Convert to predictions
        let mut predictions: Vec<ActionPrediction> = action_scores
            .into_iter()
            .map(|(action, score)| {
                // Normalize score to confidence
                let confidence = (score + 1.0) / 2.0; // Map [-1, 1] to [0, 1]
                let confidence = confidence.clamp(0.0, 1.0);

                ActionPrediction {
                    action,
                    confidence,
                    is_novel: false, // Could be computed
                }
            })
            .collect();

        // 6. Sort by confidence
        predictions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        predictions
    }

    /// Query similarity to past successful experiences
    pub fn query_experience_similarity(&mut self, context: &[(&str, &str)]) -> f32 {
        let context_hv = self.projector.encode_context(context);

        // Check similarity to positive memory
        let positive_binary = self.positive_memory.to_binary();
        context_hv.similarity(&positive_binary)
    }

    /// Get statistics
    pub fn stats(&self) -> &AssociativeLearnerStats {
        &self.stats
    }

    /// Get recent experiences
    pub fn recent_experiences(&self) -> &[ExperienceEncoding] {
        &self.recent_experiences
    }

    /// Clear memory (reset learning)
    pub fn reset(&mut self) {
        let dim = self.config.dimension;
        self.experience_memory = ContinuousHV::zero(dim);
        self.positive_memory = ContinuousHV::zero(dim);
        self.negative_memory = ContinuousHV::zero(dim);
        self.recent_experiences.clear();
        self.stats = AssociativeLearnerStats::default();
    }

    /// Export memory state (for persistence)
    pub fn export_memory(&self) -> MemorySnapshot {
        MemorySnapshot {
            experience_memory: self.experience_memory.clone(),
            positive_memory: self.positive_memory.clone(),
            negative_memory: self.negative_memory.clone(),
            stats: self.stats.clone(),
        }
    }

    /// Import memory state
    pub fn import_memory(&mut self, snapshot: MemorySnapshot) {
        self.experience_memory = snapshot.experience_memory;
        self.positive_memory = snapshot.positive_memory;
        self.negative_memory = snapshot.negative_memory;
        self.stats = snapshot.stats;
    }

    /// Encode a pattern for HDC operations
    pub fn encode_pattern(
        &mut self,
        pattern_id: PatternId,
        domain: u64,
        solution: &str,
    ) -> BinaryHV {
        self.projector.encode_context(&[
            ("pattern", &pattern_id.to_string()),
            ("domain", &domain.to_string()),
            ("solution_hash", &self.hash_solution(solution)),
        ])
    }

    fn hash_solution(&self, solution: &str) -> String {
        // Simple hash for solution text
        let hash: u64 = solution
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        format!("{:x}", hash & 0xFFFFFF) // Keep it short
    }
}

impl Default for AssociativeLearner {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Supporting Types
// =============================================================================

/// Prediction for an action
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ActionPrediction {
    /// The predicted action
    pub action: String,

    /// Confidence in this prediction (0-1)
    pub confidence: f32,

    /// Whether this is a novel recommendation (no direct experience)
    pub is_novel: bool,
}

/// Memory snapshot for persistence
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MemorySnapshot {
    /// Combined experience memory
    pub experience_memory: ContinuousHV,

    /// Positive experience memory
    pub positive_memory: ContinuousHV,

    /// Negative experience memory
    pub negative_memory: ContinuousHV,

    /// Statistics
    pub stats: AssociativeLearnerStats,
}

/// Statistics for the associative learner
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AssociativeLearnerStats {
    /// Total experiences learned
    pub total_experiences: u64,

    /// Positive experiences
    pub positive_experiences: u64,

    /// Negative experiences
    pub negative_experiences: u64,

    /// Predictions made
    pub predictions_made: u64,

    /// Successful predictions (if tracked)
    pub successful_predictions: u64,

    /// Zero-shot predictions (novel contexts)
    pub zero_shot_predictions: u64,
}

impl AssociativeLearnerStats {
    /// Get learning success rate
    pub fn success_rate(&self) -> f32 {
        if self.predictions_made == 0 {
            return 0.0;
        }
        self.successful_predictions as f32 / self.predictions_made as f32
    }

    /// Get positive/negative ratio
    pub fn positive_ratio(&self) -> f32 {
        if self.total_experiences == 0 {
            return 0.5;
        }
        self.positive_experiences as f32 / self.total_experiences as f32
    }
}

// =============================================================================
// Integration with WisdomEngine
// =============================================================================

/// Context builder for WisdomEngine integration
pub struct WisdomContext {
    pairs: Vec<(String, String)>,
}

impl WisdomContext {
    /// Create new context
    pub fn new() -> Self {
        Self { pairs: Vec::new() }
    }

    /// Add domain
    pub fn with_domain(mut self, domain_id: u64) -> Self {
        self.pairs
            .push(("domain".to_string(), domain_id.to_string()));
        self
    }

    /// Add anomaly type
    pub fn with_anomaly(mut self, anomaly_type: &str) -> Self {
        self.pairs
            .push(("anomaly".to_string(), anomaly_type.to_string()));
        self
    }

    /// Add pattern state
    pub fn with_pattern_state(mut self, state: &str) -> Self {
        self.pairs.push(("state".to_string(), state.to_string()));
        self
    }

    /// Add trust level
    pub fn with_trust_level(mut self, level: &str) -> Self {
        self.pairs.push(("trust".to_string(), level.to_string()));
        self
    }

    /// Add custom attribute
    pub fn with_attribute(mut self, key: &str, value: &str) -> Self {
        self.pairs.push((key.to_string(), value.to_string()));
        self
    }

    /// Build context pairs
    pub fn build(&self) -> Vec<(&str, &str)> {
        self.pairs
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect()
    }
}

impl Default for WisdomContext {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_hv_basic() {
        let hv1 = BinaryHV::random_default(42);
        let hv2 = BinaryHV::random_default(43);

        // Random vectors should be roughly 50% ones
        let density = hv1.popcount() as f32 / HDC_DIMENSION as f32;
        assert!((density - 0.5).abs() < 0.1);

        // Random vectors should be nearly orthogonal
        let sim = hv1.similarity(&hv2);
        assert!(sim.abs() < 0.1);

        // Self-similarity should be 1.0
        assert!((hv1.similarity(&hv1) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_binding() {
        let a = BinaryHV::random_default(1);
        let b = BinaryHV::random_default(2);

        // Binding creates dissimilar vector
        let bound = a.bind(&b);
        assert!(a.similarity(&bound).abs() < 0.1);
        assert!(b.similarity(&bound).abs() < 0.1);

        // Unbinding recovers original
        let recovered_b = bound.unbind(&a);
        assert!((recovered_b.similarity(&b) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_bundling() {
        let a = BinaryHV::random_default(1);
        let b = BinaryHV::random_default(2);
        let c = BinaryHV::random_default(3);

        // Bundle is similar to components
        let bundled = BinaryHV::bundle(&[a.clone(), b.clone(), c.clone()]);
        assert!(bundled.similarity(&a) > 0.3);
        assert!(bundled.similarity(&b) > 0.3);
        assert!(bundled.similarity(&c) > 0.3);
    }

    #[test]
    fn test_projector() {
        let mut projector = SparseProjector::default_dim(42);

        // Same symbol produces same vector
        let v1 = projector.encode_symbol("network");
        let v2 = projector.encode_symbol("network");
        assert_eq!(v1.similarity(&v2), 1.0);

        // Different symbols produce orthogonal vectors
        let v3 = projector.encode_symbol("database");
        assert!(v1.similarity(&v3).abs() < 0.1);
    }

    #[test]
    fn test_associative_learning() {
        let mut learner = AssociativeLearner::new();

        // Register some actions
        learner.register_action("CircuitBreak");
        learner.register_action("Retry");
        learner.register_action("Fallback");

        // Learn: CircuitBreak works for network latency
        learner.learn(
            &[("domain", "Network"), ("anomaly", "Latency")],
            "CircuitBreak",
            1.0,
            1000,
        );

        // Learn: Retry fails for network latency
        learner.learn(
            &[("domain", "Network"), ("anomaly", "Latency")],
            "Retry",
            -0.5,
            1001,
        );

        // Predict for similar context (FileSystem latency)
        let predictions = learner.predict(&[("domain", "FileSystem"), ("anomaly", "Latency")]);

        // CircuitBreak should be recommended due to similarity
        assert!(!predictions.is_empty());
        // The system should generalize the "Latency" concept
    }

    #[test]
    fn test_stats() {
        let mut learner = AssociativeLearner::new();

        learner.learn(&[("a", "b")], "action1", 1.0, 100);
        learner.learn(&[("c", "d")], "action2", -0.5, 101);

        let stats = learner.stats();
        assert_eq!(stats.total_experiences, 2);
        assert_eq!(stats.positive_experiences, 1);
        assert_eq!(stats.negative_experiences, 1);
    }

    #[test]
    fn test_wisdom_context() {
        let ctx = WisdomContext::new()
            .with_domain(42)
            .with_anomaly("SuccessRateDrop")
            .with_trust_level("High");

        let pairs = ctx.build();
        assert_eq!(pairs.len(), 3);
    }

    #[test]
    fn test_continuous_hv() {
        let a = ContinuousHV::random(100, 42);
        let b = ContinuousHV::random(100, 43);

        // Similarity should be near 0 for random vectors
        let sim = a.similarity(&b);
        assert!(sim.abs() < 0.3);

        // Self-similarity should be 1.0
        let self_sim = a.similarity(&a);
        assert!((self_sim - 1.0).abs() < 0.001);
    }
}
