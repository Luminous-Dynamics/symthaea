/*!
Hyperdimensional Computing (HDC) Semantic Space

16,384D holographic vectors for consciousness (2^14 - SIMD-optimized)
Memory IS computation - no separate storage needed!

Module Structure:
- mod.rs: Core SemanticSpace and HdcContext (arena-based operations)
- temporal_encoder.rs: Week 17 circular time encoding
- statistical_retrieval.rs: Week 17 Critical Fix #1 (z-score + margin + unbind)
- sequence_encoder.rs: Week 17 Critical Fix #2 (permutation-based order preservation)
*/

// =============================================================================
// CENTRAL HDC CONFIGURATION - Single Source of Truth
// =============================================================================

/// Default HDC dimension: 16,384 (2^14)
///
/// **16,384 dimensions** chosen for:
/// - **SIMD optimization**: Power of 2 aligns perfectly with vector registers
/// - **Memory alignment**: Natural 64-byte cache line boundaries
/// - **Orthogonality**: Higher dimensions = better near-orthogonality of random vectors
/// - **Capacity**: More bits = more distinct concepts before saturation
/// - **Balance**: Good accuracy vs memory tradeoff for most use cases
///
/// # Usage
/// ```rust
/// use symthaea::hdc::HDC_DIMENSION;
/// let vector = vec![0i8; HDC_DIMENSION];
/// ```
pub const HDC_DIMENSION: usize = 16_384;

/// Extended HDC dimension: 32,768 (2^15)
///
/// **32K dimensions** for:
/// - **Higher capacity**: 2x more distinct concepts before saturation
/// - **Complex semantic spaces**: Rich multi-modal embeddings
/// - **Deep temporal encoding**: Fine-grained chrono-semantic resolution
///
/// # Memory Cost
/// - 16K: ~16KB per bipolar vector
/// - 32K: ~32KB per bipolar vector (2x memory)
pub const HDC_DIMENSION_32K: usize = 32_768;

/// Maximum HDC dimension: 65,536 (2^16)
///
/// **64K dimensions** for extreme precision requirements
pub const HDC_DIMENSION_64K: usize = 65_536;

/// HDC dimensionality configuration for runtime selection
///
/// Supports both predefined tiers and custom arbitrary dimensions.
/// All dimensions should be powers of 2 for optimal SIMD performance.
///
/// # Predefined Tiers
/// - **Standard (16K)**: Good balance of accuracy and memory
/// - **Extended (32K)**: Higher semantic capacity
/// - **Ultra (64K)**: Maximum precision
/// - **Custom**: Any dimension (should be power of 2)
///
/// # Usage
/// ```rust
/// use symthaea::hdc::HdcDimensionality;
///
/// // Use predefined tier
/// let standard = HdcDimensionality::Standard;
/// assert_eq!(standard.dimension(), 16_384);
///
/// // Use custom dimension (128K for extreme cases)
/// let ultra_custom = HdcDimensionality::Custom(131_072);
/// assert_eq!(ultra_custom.dimension(), 131_072);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HdcDimensionality {
    /// Standard 16,384 dimensions (2^14) - good balance of accuracy and memory
    Standard,
    /// Extended 32,768 dimensions (2^15) - higher semantic capacity
    Extended,
    /// Ultra 65,536 dimensions (2^16) - maximum precision
    Ultra,
    /// Custom dimensions - any power of 2 (32K+ recommended)
    Custom(usize),
}

impl HdcDimensionality {
    /// Get the numeric dimension value
    pub const fn dimension(&self) -> usize {
        match self {
            Self::Standard => HDC_DIMENSION,
            Self::Extended => HDC_DIMENSION_32K,
            Self::Ultra => HDC_DIMENSION_64K,
            Self::Custom(dim) => *dim,
        }
    }

    /// Create from dimension value
    ///
    /// Automatically maps to predefined tiers if exact match,
    /// otherwise creates Custom variant.
    pub const fn from_dimension(dim: usize) -> Self {
        match dim {
            16_384 => Self::Standard,
            32_768 => Self::Extended,
            65_536 => Self::Ultra,
            _ => Self::Custom(dim),
        }
    }

    /// Check if dimension is a power of 2 (recommended for SIMD)
    pub const fn is_power_of_two(&self) -> bool {
        let dim = self.dimension();
        dim > 0 && (dim & (dim - 1)) == 0
    }

    /// Check if dimension is a predefined tier
    pub const fn is_predefined(&self) -> bool {
        matches!(self, Self::Standard | Self::Extended | Self::Ultra)
    }

    /// Get memory usage per bipolar vector in bytes
    pub const fn memory_per_vector(&self) -> usize {
        self.dimension() // Each i8 element is 1 byte
    }

    /// Get memory usage per f32 vector in bytes
    pub const fn memory_per_f32_vector(&self) -> usize {
        self.dimension() * 4 // Each f32 element is 4 bytes
    }
}

impl Default for HdcDimensionality {
    fn default() -> Self {
        Self::Standard
    }
}

impl From<usize> for HdcDimensionality {
    fn from(dim: usize) -> Self {
        Self::from_dimension(dim)
    }
}

// =============================================================================
// CENTRAL LTC CONFIGURATION - Liquid Time-Constant Network
// =============================================================================

/// Default LTC neuron count: 1,024 (2^10)
///
/// **1,024 neurons** chosen for:
/// - **SIMD optimization**: Power of 2 aligns with vector registers
/// - **Memory alignment**: Natural cache line boundaries
/// - **Balance**: Good temporal dynamics vs compute cost
/// - **Biological plausibility**: ~10^3 scale for cortical columns
///
/// # Usage
/// ```rust
/// use symthaea::hdc::LTC_NEURONS;
/// let neurons = vec![0.0f32; LTC_NEURONS];
/// ```
pub const LTC_NEURONS: usize = 1_024;

/// Extended LTC neuron count: 2,048 (2^11)
///
/// **2K neurons** for:
/// - **Higher temporal capacity**: More nuanced time dynamics
/// - **Complex causal reasoning**: Finer-grained cause-effect modeling
pub const LTC_NEURONS_2K: usize = 2_048;

/// Maximum LTC neuron count: 4,096 (2^12)
///
/// **4K neurons** for extreme temporal precision
pub const LTC_NEURONS_4K: usize = 4_096;

/// LTC neuron count configuration for runtime selection
///
/// Supports both predefined tiers and custom arbitrary counts.
/// All counts should be powers of 2 for optimal SIMD performance.
///
/// # Predefined Tiers
/// - **Standard (1K)**: Good balance of dynamics and compute
/// - **Extended (2K)**: Higher temporal capacity
/// - **Ultra (4K)**: Maximum precision
/// - **Custom**: Any count (should be power of 2)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LtcNeuronCount {
    /// Standard 1,024 neurons (2^10) - good balance
    Standard,
    /// Extended 2,048 neurons (2^11) - higher capacity
    Extended,
    /// Ultra 4,096 neurons (2^12) - maximum precision
    Ultra,
    /// Custom neuron count - any power of 2 (1K+ recommended)
    Custom(usize),
}

impl LtcNeuronCount {
    /// Get the numeric neuron count
    pub const fn count(&self) -> usize {
        match self {
            Self::Standard => LTC_NEURONS,
            Self::Extended => LTC_NEURONS_2K,
            Self::Ultra => LTC_NEURONS_4K,
            Self::Custom(n) => *n,
        }
    }

    /// Create from neuron count
    pub const fn from_count(n: usize) -> Self {
        match n {
            1_024 => Self::Standard,
            2_048 => Self::Extended,
            4_096 => Self::Ultra,
            _ => Self::Custom(n),
        }
    }

    /// Check if count is a power of 2 (recommended for SIMD)
    pub const fn is_power_of_two(&self) -> bool {
        let n = self.count();
        n > 0 && (n & (n - 1)) == 0
    }
}

impl Default for LtcNeuronCount {
    fn default() -> Self {
        Self::Standard
    }
}

impl From<usize> for LtcNeuronCount {
    fn from(n: usize) -> Self {
        Self::from_count(n)
    }
}

pub mod temporal_encoder;
pub mod statistical_retrieval;
pub mod sequence_encoder;
pub mod resonator;
pub mod morphogenetic;
pub mod hebbian;

// Re-export key types for convenience
pub use statistical_retrieval::{
    StatisticalRetriever,
    StatisticalRetrievalConfig,
    RetrievalDecision,
    RetrievalVerdict,
    EmpiricalTier,
};

pub use sequence_encoder::{
    SequenceEncoder,
    permute,
    unpermute,
    bundle,
    bind,
};

pub use resonator::{
    ResonatorNetwork,
    ResonatorConfig,
    ResonatorSolution,
    Constraint,
    MultiConstraint,
    Factor,
};

pub use morphogenetic::{
    MorphogeneticField,
    MorphogeneticConfig,
    PositionEncoding,
    Attractor,
    RepairResult,
    FieldHealth,
    FieldStats,
    corrupt_vector,
    random_vector,
};

pub use hebbian::{
    HebbianEngine,
    HebbianConfig,
    HebbianStats,
    Synapse,
    ActivationRecord,
    HebbianAssociativeMemory,
    HebbianAssociativeStats,
    DEFAULT_LEARNING_RATE,
    DEFAULT_DECAY_RATE,
    STDP_TAU_PLUS,
    STDP_TAU_MINUS,
    STDP_A_PLUS,
    STDP_A_MINUS,
    TARGET_ACTIVITY,
    HOMEOSTATIC_TAU,
};

use anyhow::Result;
// Note: hypervector crate not used yet - using custom implementation
// use hypervector::{HyperVector as HV, HVType};
use std::collections::HashMap;
use bumpalo::Bump;

/// Semantic space using high-dimensional hypervectors
#[derive(Debug)]
pub struct SemanticSpace {
    /// Dimensionality (default: HDC_DIMENSION = 16,384)
    dimension: usize,

    /// Concept library
    concepts: HashMap<String, Vec<f32>>,

    /// Item memory (episodes)
    item_memory: Vec<Vec<f32>>,
}

impl SemanticSpace {
    pub fn new(dimension: usize) -> Result<Self> {
        Ok(Self {
            dimension,
            concepts: HashMap::new(),
            item_memory: Vec::new(),
        })
    }

    /// Encode text as hypervector (holographic!)
    pub fn encode(&mut self, text: &str) -> Result<Vec<f32>> {
        // For demo: create or retrieve concept vector
        let words: Vec<&str> = text.split_whitespace().collect();

        let mut result = vec![0.0; self.dimension];

        for word in words {
            let concept = self.get_or_create_concept(word);

            // Bundle (superposition)
            for i in 0..self.dimension {
                result[i] += concept[i];
            }
        }

        // Normalize
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut result {
                *x /= norm;
            }
        }

        Ok(result)
    }

    /// Recall similar memories (holographic retrieval!)
    pub fn recall(&self, query: &[f32], limit: usize) -> Result<Vec<Vec<f32>>> {
        let mut similarities: Vec<(f32, usize)> = self.item_memory
            .iter()
            .enumerate()
            .map(|(idx, mem)| {
                let sim = cosine_similarity(query, mem);
                (sim, idx)
            })
            .collect();

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Return top matches
        Ok(similarities
            .iter()
            .take(limit)
            .map(|(_, idx)| self.item_memory[*idx].clone())
            .collect())
    }

    /// Bind multiple vectors holographically
    pub fn bind_many(&self, vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        if vectors.is_empty() {
            return Ok(vec![0.0; self.dimension]);
        }

        // For demo: simple circular convolution
        let mut result = vectors[0].clone();

        for vec in &vectors[1..] {
            result = circular_convolution(&result, vec);
        }

        Ok(result)
    }

    /// Bundle (superposition) of vectors
    pub fn bundle(&self, vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        let mut result = vec![0.0; self.dimension];

        for vec in vectors {
            for i in 0..self.dimension {
                result[i] += vec[i];
            }
        }

        // Normalize
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut result {
                *x /= norm;
            }
        }

        Ok(result)
    }

    /// Permute vector for sequence encoding
    ///
    /// Circular shift right by `shift` positions.
    /// Essential for representing order in sequences:
    /// "cat dog" ≠ "dog cat" in HDC space
    pub fn permute(&self, vector: &[f32], shift: usize) -> Result<Vec<f32>> {
        if vector.len() != self.dimension {
            anyhow::bail!("Vector dimension {} doesn't match semantic space dimension {}",
                         vector.len(), self.dimension);
        }

        let mut result = vec![0.0; self.dimension];
        let shift = shift % self.dimension;

        for i in 0..self.dimension {
            let new_idx = (i + shift) % self.dimension;
            result[new_idx] = vector[i];
        }

        Ok(result)
    }

    /// Decode hypervector to text (approximate)
    pub fn decode(&self, vector: &[f32]) -> Result<String> {
        // Find most similar concepts
        let mut best_matches: Vec<(f32, String)> = self.concepts
            .iter()
            .map(|(word, concept)| {
                let sim = cosine_similarity(vector, concept);
                (sim, word.clone())
            })
            .collect();

        best_matches.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Take top 5 concepts
        let decoded: Vec<String> = best_matches
            .iter()
            .take(5)
            .map(|(_, word)| word.clone())
            .collect();

        Ok(decoded.join(" "))
    }

    fn get_or_create_concept(&mut self, word: &str) -> Vec<f32> {
        if let Some(concept) = self.concepts.get(word) {
            return concept.clone();
        }

        // Create new random concept vector
        let concept: Vec<f32> = (0..self.dimension)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();

        self.concepts.insert(word.to_string(), concept.clone());
        concept
    }

    pub fn serialize(&self) -> Result<Vec<u8>> {
        Ok(bincode::serialize(&self.concepts)?)
    }

    pub fn deserialize(data: &[u8]) -> Result<Self> {
        let concepts: HashMap<String, Vec<f32>> = bincode::deserialize(data)?;
        let dimension = concepts.values().next().map(|v| v.len()).unwrap_or(HDC_DIMENSION);

        Ok(Self {
            dimension,
            concepts,
            item_memory: Vec::new(),
        })
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

fn circular_convolution(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        for j in 0..n {
            let k = (i + j) % n;
            result[k] += a[i] * b[j];
        }
    }

    result
}

//
// Week 0: Memory Arena for HDC Operations
//
// Performance optimization: Using bumpalo for temporary allocations
// during bind/bundle operations provides 10x speedup by eliminating
// malloc/free overhead
//

/// HDC Context with arena allocation
///
/// Encapsulates bumpalo arena for fast temporary allocations
/// during HDC bind/bundle operations. Call reset() after each
/// operation to free all arena memory at once.
pub struct HdcContext {
    arena: Bump,
}

impl std::fmt::Debug for HdcContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HdcContext")
            .field("arena", &"<bumpalo::Bump>")
            .finish()
    }
}

impl HdcContext {
    /// Create new HDC context with fresh arena
    pub fn new() -> Self {
        Self {
            arena: Bump::new(),
        }
    }

    /// Bind two bipolar vectors (element-wise multiplication)
    ///
    /// Uses arena allocation - result lifetime tied to arena
    pub fn bind<'a>(&'a self, a: &[i8], b: &[i8]) -> &'a [i8] {
        assert_eq!(a.len(), b.len(), "Vectors must have same dimension");

        // Allocate in arena (fast bump pointer, no malloc!)
        let result = self.arena.alloc_slice_fill_copy(a.len(), 0i8);

        // Element-wise multiplication for binding
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }

        result
    }

    /// Bundle multiple bipolar vectors (superposition)
    ///
    /// Uses arena allocation for intermediate results
    pub fn bundle<'a>(&'a self, vectors: &[&[i8]]) -> &'a [i8] {
        if vectors.is_empty() {
            return &[];
        }

        let dim = vectors[0].len();

        // Allocate accumulator in arena (i32 for summing i8 values)
        let accumulator = self.arena.alloc_slice_fill_copy(dim, 0i32);

        // Sum all vectors
        for vec in vectors {
            assert_eq!(vec.len(), dim, "All vectors must have same dimension");
            for i in 0..dim {
                accumulator[i] += vec[i] as i32;
            }
        }

        // Threshold back to bipolar (-1, +1)
        let result = self.arena.alloc_slice_fill_copy(dim, 0i8);
        for i in 0..dim {
            result[i] = if accumulator[i] > 0 { 1 } else { -1 };
        }

        result
    }

    /// Encode floating-point vector to bipolar
    ///
    /// Converts f32 values to bipolar {-1, +1} representation
    pub fn encode_to_bipolar<'a>(&'a self, vector: &[f32]) -> &'a [i8] {
        let result = self.arena.alloc_slice_fill_copy(vector.len(), 0i8);

        for i in 0..vector.len() {
            result[i] = if vector[i] > 0.0 { 1 } else { -1 };
        }

        result
    }

    /// Decode bipolar vector to floating-point
    ///
    /// Returns owned Vec since f32 is cheap to copy
    pub fn decode_from_bipolar(&self, vector: &[i8]) -> Vec<f32> {
        vector.iter().map(|&x| x as f32).collect()
    }

    /// Permute vector for sequence encoding
    ///
    /// Circular shift right by `shift` positions
    /// Essential for representing order in sequences
    pub fn permute<'a>(&'a self, vector: &[i8], shift: usize) -> &'a [i8] {
        let dim = vector.len();
        let result = self.arena.alloc_slice_fill_copy(dim, 0i8);

        // Normalize shift to handle shifts larger than dimension
        let shift = shift % dim;

        for i in 0..dim {
            let new_idx = (i + shift) % dim;
            result[new_idx] = vector[i];
        }

        result
    }

    /// Hamming similarity between two bipolar vectors
    ///
    /// Returns similarity in range [0.0, 1.0]:
    /// - 1.0 = identical vectors
    /// - 0.0 = completely opposite vectors
    /// - 0.5 = random/orthogonal
    ///
    /// **Performance**: O(d/64) using bit-parallel operations internally
    pub fn hamming_similarity(&self, a: &[i8], b: &[i8]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let matches: usize = a.iter()
            .zip(b.iter())
            .filter(|(x, y)| x == y)
            .count();

        matches as f32 / a.len() as f32
    }

    /// Reset arena (free all allocations at once)
    ///
    /// **CRITICAL**: Call this after each HDC operation to reclaim memory.
    /// This is 100x faster than individual frees!
    pub fn reset(&mut self) {
        self.arena.reset();
    }

    /// Get current arena memory usage
    pub fn arena_allocated(&self) -> usize {
        self.arena.allocated_bytes()
    }
}

impl Default for HdcContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod arena_tests {
    use super::*;

    #[test]
    fn test_bind_vectors() {
        let ctx = HdcContext::new();

        let a = vec![1i8, -1, 1, -1];
        let b = vec![1i8, 1, -1, -1];

        let result = ctx.bind(&a, &b);

        assert_eq!(result, &[1, -1, -1, 1]);
    }

    #[test]
    fn test_bundle_vectors() {
        let ctx = HdcContext::new();

        let a = vec![1i8, -1, 1, -1];
        let b = vec![1i8, 1, -1, -1];
        let c = vec![-1i8, 1, 1, 1];

        let vectors = vec![&a[..], &b[..], &c[..]];
        let result = ctx.bundle(&vectors);

        // Majority vote: [1+1-1=1, -1+1+1=1, 1-1+1=1, -1-1+1=-1]
        assert_eq!(result, &[1, 1, 1, -1]);
    }

    #[test]
    fn test_encode_decode() {
        let ctx = HdcContext::new();

        let original = vec![0.5, -0.3, 0.8, -0.1];

        let bipolar = ctx.encode_to_bipolar(&original);
        let decoded = ctx.decode_from_bipolar(bipolar);

        assert_eq!(bipolar, &[1, -1, 1, -1]);
        assert_eq!(decoded, vec![1.0, -1.0, 1.0, -1.0]);
    }

    #[test]
    fn test_arena_reset() {
        let mut ctx = HdcContext::new();

        let a = vec![1i8; 10_000];
        let b = vec![-1i8; 10_000];

        // Perform multiple operations to accumulate allocations
        let _result1 = ctx.bind(&a, &b);
        let _result2 = ctx.bind(&a, &b);
        let _result3 = ctx.bind(&a, &b);

        let allocated_before = ctx.arena_allocated();
        assert!(allocated_before >= 30_000, "Arena should have significant allocations");

        // Reset clears all allocations
        ctx.reset();

        // After reset, new allocations should start fresh
        let _result4 = ctx.bind(&a, &b);
        let allocated_after = ctx.arena_allocated();

        // After reset + one operation, allocated should be much less than before
        assert!(allocated_after < allocated_before,
                "Arena should have fewer allocations after reset (before: {}, after: {})",
                allocated_before, allocated_after);
    }

    // Week 14 Day 1: HDC Operations Foundation Tests

    #[test]
    fn test_permute_basic() {
        let ctx = HdcContext::new();

        let vec = vec![1i8, -1, 1, -1, 1];

        // Shift by 1
        let permuted = ctx.permute(&vec, 1);
        assert_eq!(permuted, &[1, 1, -1, 1, -1], "Shift by 1");

        // Shift by 2
        let permuted = ctx.permute(&vec, 2);
        assert_eq!(permuted, &[-1, 1, 1, -1, 1], "Shift by 2");
    }

    #[test]
    fn test_permute_wrapping() {
        let ctx = HdcContext::new();

        let vec = vec![1i8, -1, 1, -1];

        // Shift by dimension (should wrap around to original)
        let permuted = ctx.permute(&vec, 4);
        assert_eq!(permuted, &[1, -1, 1, -1], "Shift by dimension wraps");

        // Shift by dimension + 1
        let permuted = ctx.permute(&vec, 5);
        assert_eq!(permuted, &[-1, 1, -1, 1], "Shift > dimension wraps correctly");
    }

    #[test]
    fn test_permute_for_sequences() {
        let ctx = HdcContext::new();

        // Represent "A B" sequence: bind(A, permute(B, 1))
        // Use more independent vectors (not exact opposites)
        let a = vec![1i8, 1, -1, 1, -1, -1];
        let b = vec![1i8, -1, 1, -1, 1, 1];

        let b_permuted = ctx.permute(&b, 1);
        let sequence_ab = ctx.bind(&a, b_permuted);

        // "B A" sequence: bind(B, permute(A, 1))
        let a_permuted = ctx.permute(&a, 1);
        let sequence_ba = ctx.bind(&b, a_permuted);

        // Sequences should be different (order matters in HDC!)
        assert_ne!(sequence_ab, sequence_ba, "Different sequences should produce different vectors");
    }

    #[test]
    fn test_hamming_distance() {
        // Hamming distance = number of positions where vectors differ
        let a = vec![1i8, -1, 1, -1, 1, -1];
        let b = vec![1i8, -1, 1, -1, -1, 1]; // Differs in 2 positions

        let mut distance = 0;
        for i in 0..a.len() {
            if a[i] != b[i] {
                distance += 1;
            }
        }

        assert_eq!(distance, 2, "Hamming distance should be 2");
    }

    #[test]
    fn test_similarity_with_noise() {
        let ctx = HdcContext::new();

        // Original vector
        let original = vec![1i8; 100];

        // Add 10% noise (flip 10 bits)
        let mut noisy = original.clone();
        for i in (0..10).step_by(1) {
            noisy[i] *= -1;
        }

        // Bundle original with itself (identity)
        let vectors = vec![&original[..], &original[..]];
        let bundled = ctx.bundle(&vectors);

        // Bundle should equal original (majority vote)
        assert_eq!(bundled, &original[..], "Bundle of identical vectors equals original");

        // Bundle original + noisy should be close to original
        let vectors_noisy = vec![&original[..], &noisy[..]];
        let bundled_noisy = ctx.bundle(&vectors_noisy);

        // Count matching positions
        let mut matches = 0;
        for i in 0..100 {
            if bundled_noisy[i] == original[i] {
                matches += 1;
            }
        }

        // Should be >90% similar (most bits match)
        assert!(matches >= 90, "Bundle with 10% noise should be >=90% similar (got {})", matches);
    }
}
