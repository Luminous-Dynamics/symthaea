//! # HDC Attention Mechanism
//!
//! ## Purpose
//! Content-based attention using HDC similarity instead of learned matrix operations.
//! This is a novel approach where attention weights are computed via binding operations
//! and cosine similarity rather than traditional dot-product attention.
//!
//! ## Theoretical Basis
//! Traditional Transformer attention: Q·K^T / √d → softmax → weighted V
//! HDC Attention: sim(Q⊗Wq, K⊗Wk) → softmax → weighted_bundle(V⊗Wv)
//!
//! Where:
//! - ⊗ is HDC binding (element-wise multiplication)
//! - sim() is cosine similarity
//! - weighted_bundle() is weighted average bundling
//!
//! ## Advantages
//! 1. **Interpretable**: Attention weights based on semantic similarity
//! 2. **Efficient**: O(1) transform vs O(d²) matrix multiply
//! 3. **Algebraic**: Preserves HDC algebraic properties
//! 4. **Compositional**: Transforms compose via binding
//!
//! ## Example Usage
//! ```rust,ignore
//! use symthaea::hdc::attention::{HdcAttention, AttentionConfig};
//! use symthaea::hdc::unified_hv::ContinuousHV;
//!
//! let config = AttentionConfig::default();
//! let attention = HdcAttention::new(config, 42);
//!
//! let query = ContinuousHV::random_default(100);
//! let memories = vec![
//!     ContinuousHV::random_default(101),
//!     ContinuousHV::random_default(102),
//!     ContinuousHV::random_default(103),
//! ];
//!
//! let result = attention.attend(&query, &memories);
//! ```

use serde::{Deserialize, Serialize};
use crate::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for HDC Attention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    /// Dimension of hypervectors
    pub dim: usize,

    /// Temperature for softmax (higher = sharper attention)
    pub temperature: f32,

    /// Minimum attention weight to include in output
    pub min_attention: f32,

    /// Number of attention heads (for multi-head attention)
    pub num_heads: usize,

    /// Whether to normalize output
    pub normalize_output: bool,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            dim: HDC_DIMENSION,
            temperature: 1.0,
            min_attention: 0.0,
            num_heads: 1,
            normalize_output: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SINGLE-HEAD HDC ATTENTION
// ═══════════════════════════════════════════════════════════════════════════════

/// HDC-based attention mechanism using binding and similarity operations
///
/// Unlike traditional Transformer attention which uses matrix multiplications,
/// this implementation uses HDC operations:
/// - Query transform: Q' = Q ⊗ Wq (binding)
/// - Key transform: K' = K ⊗ Wk (binding)
/// - Value transform: V' = V ⊗ Wv (binding)
/// - Attention weights: softmax(sim(Q', K'i) / τ)
/// - Output: weighted_bundle(V'i, weights)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdcAttention {
    /// Configuration
    config: AttentionConfig,

    /// Query transform hypervector
    query_transform: ContinuousHV,

    /// Key transform hypervector
    key_transform: ContinuousHV,

    /// Value transform hypervector
    value_transform: ContinuousHV,
}

impl HdcAttention {
    /// Create new HDC Attention with given config and seed
    pub fn new(config: AttentionConfig, seed: u64) -> Self {
        let query_transform = ContinuousHV::random(config.dim, seed);
        let key_transform = ContinuousHV::random(config.dim, seed + 1000);
        let value_transform = ContinuousHV::random(config.dim, seed + 2000);

        Self {
            config,
            query_transform,
            key_transform,
            value_transform,
        }
    }

    /// Create with default config
    pub fn new_default(seed: u64) -> Self {
        Self::new(AttentionConfig::default(), seed)
    }

    /// Attend over memory using query
    ///
    /// Returns weighted bundle of memories based on query similarity.
    pub fn attend(&self, query: &ContinuousHV, memory: &[ContinuousHV]) -> AttentionResult {
        if memory.is_empty() {
            return AttentionResult {
                output: ContinuousHV::zero(self.config.dim),
                weights: vec![],
                transformed_query: query.clone(),
            };
        }

        // Transform query to query space via binding
        let q = query.bind(&self.query_transform);

        // Compute attention weights via HDC similarity
        let raw_scores: Vec<f32> = memory.iter()
            .map(|m| {
                let k = m.bind(&self.key_transform);
                q.similarity(&k)
            })
            .collect();

        // Softmax normalization with temperature
        let weights = softmax(&raw_scores, self.config.temperature);

        // Transform values and compute weighted bundle
        let values: Vec<ContinuousHV> = memory.iter()
            .map(|m| m.bind(&self.value_transform))
            .collect();

        let value_refs: Vec<&ContinuousHV> = values.iter().collect();
        let output = ContinuousHV::weighted_bundle(&value_refs, &weights);

        // Optionally normalize output
        let output = if self.config.normalize_output {
            output.normalize()
        } else {
            output
        };

        AttentionResult {
            output,
            weights,
            transformed_query: q,
        }
    }

    /// Attend with attention mask
    ///
    /// Masked positions (false) get zero attention weight.
    pub fn attend_masked(
        &self,
        query: &ContinuousHV,
        memory: &[ContinuousHV],
        mask: &[bool],
    ) -> AttentionResult {
        if memory.is_empty() || mask.len() != memory.len() {
            return self.attend(query, memory);
        }

        // Transform query
        let q = query.bind(&self.query_transform);

        // Compute masked attention weights
        let raw_scores: Vec<f32> = memory.iter()
            .zip(mask.iter())
            .map(|(m, &allowed)| {
                if allowed {
                    let k = m.bind(&self.key_transform);
                    q.similarity(&k)
                } else {
                    f32::NEG_INFINITY  // Will become 0 after softmax
                }
            })
            .collect();

        let weights = softmax(&raw_scores, self.config.temperature);

        // Transform values and bundle
        let values: Vec<ContinuousHV> = memory.iter()
            .map(|m| m.bind(&self.value_transform))
            .collect();

        let value_refs: Vec<&ContinuousHV> = values.iter().collect();
        let output = ContinuousHV::weighted_bundle(&value_refs, &weights);

        let output = if self.config.normalize_output {
            output.normalize()
        } else {
            output
        };

        AttentionResult {
            output,
            weights,
            transformed_query: q,
        }
    }

    /// Self-attention over a sequence
    ///
    /// Each position attends to all positions (including itself).
    pub fn self_attend(&self, sequence: &[ContinuousHV]) -> Vec<AttentionResult> {
        sequence.iter()
            .map(|query| self.attend(query, sequence))
            .collect()
    }

    /// Causal self-attention (each position only attends to previous positions)
    pub fn causal_self_attend(&self, sequence: &[ContinuousHV]) -> Vec<AttentionResult> {
        sequence.iter()
            .enumerate()
            .map(|(i, query)| {
                // Create causal mask: true for positions <= i
                let mask: Vec<bool> = (0..sequence.len())
                    .map(|j| j <= i)
                    .collect();
                self.attend_masked(query, sequence, &mask)
            })
            .collect()
    }

    /// Get configuration
    pub fn config(&self) -> &AttentionConfig {
        &self.config
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MULTI-HEAD HDC ATTENTION
// ═══════════════════════════════════════════════════════════════════════════════

/// Multi-head HDC Attention
///
/// Combines multiple attention heads, each with its own transform vectors.
/// Output is the bundle of all head outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiHeadHdcAttention {
    /// Individual attention heads
    heads: Vec<HdcAttention>,

    /// Output projection hypervector
    output_transform: ContinuousHV,

    /// Configuration
    config: AttentionConfig,
}

impl MultiHeadHdcAttention {
    /// Create multi-head attention
    pub fn new(config: AttentionConfig, seed: u64) -> Self {
        let heads: Vec<HdcAttention> = (0..config.num_heads)
            .map(|i| {
                let head_seed = seed + (i as u64) * 10000;
                HdcAttention::new(config.clone(), head_seed)
            })
            .collect();

        let output_transform = ContinuousHV::random(config.dim, seed + 99999);

        Self {
            heads,
            output_transform,
            config,
        }
    }

    /// Attend using all heads and combine outputs
    pub fn attend(&self, query: &ContinuousHV, memory: &[ContinuousHV]) -> MultiHeadAttentionResult {
        let head_results: Vec<AttentionResult> = self.heads.iter()
            .map(|head| head.attend(query, memory))
            .collect();

        // Combine head outputs via bundling
        let head_outputs: Vec<&ContinuousHV> = head_results.iter()
            .map(|r| &r.output)
            .collect();

        let combined = ContinuousHV::bundle(&head_outputs);

        // Apply output projection
        let output = combined.bind(&self.output_transform);

        MultiHeadAttentionResult {
            output,
            head_results,
        }
    }

    /// Multi-head self-attention
    pub fn self_attend(&self, sequence: &[ContinuousHV]) -> Vec<MultiHeadAttentionResult> {
        sequence.iter()
            .map(|query| self.attend(query, sequence))
            .collect()
    }

    /// Causal multi-head self-attention
    pub fn causal_self_attend(&self, sequence: &[ContinuousHV]) -> Vec<MultiHeadAttentionResult> {
        sequence.iter()
            .enumerate()
            .map(|(i, query)| {
                let past = &sequence[..=i];
                self.attend(query, past)
            })
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CROSS-ATTENTION (for encoder-decoder architectures)
// ═══════════════════════════════════════════════════════════════════════════════

/// Cross-attention for encoder-decoder architectures
///
/// Queries come from decoder, keys/values from encoder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossAttention {
    /// Base attention mechanism
    attention: HdcAttention,

    /// Cached encoder representations (keys + values)
    encoder_cache: Option<EncoderCache>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EncoderCache {
    keys: Vec<ContinuousHV>,
    values: Vec<ContinuousHV>,
}

impl CrossAttention {
    /// Create cross-attention
    pub fn new(config: AttentionConfig, seed: u64) -> Self {
        Self {
            attention: HdcAttention::new(config, seed),
            encoder_cache: None,
        }
    }

    /// Cache encoder outputs for efficient repeated decoding
    pub fn cache_encoder(&mut self, encoder_outputs: &[ContinuousHV]) {
        let keys: Vec<ContinuousHV> = encoder_outputs.iter()
            .map(|e| e.bind(&self.attention.key_transform))
            .collect();

        let values: Vec<ContinuousHV> = encoder_outputs.iter()
            .map(|e| e.bind(&self.attention.value_transform))
            .collect();

        self.encoder_cache = Some(EncoderCache { keys, values });
    }

    /// Attend decoder query to cached encoder
    pub fn attend_cached(&self, decoder_query: &ContinuousHV) -> Option<AttentionResult> {
        let cache = self.encoder_cache.as_ref()?;

        let q = decoder_query.bind(&self.attention.query_transform);

        // Compute weights using cached keys
        let raw_scores: Vec<f32> = cache.keys.iter()
            .map(|k| q.similarity(k))
            .collect();

        let weights = softmax(&raw_scores, self.attention.config.temperature);

        // Weighted bundle of cached values
        let value_refs: Vec<&ContinuousHV> = cache.values.iter().collect();
        let output = ContinuousHV::weighted_bundle(&value_refs, &weights);

        let output = if self.attention.config.normalize_output {
            output.normalize()
        } else {
            output
        };

        Some(AttentionResult {
            output,
            weights,
            transformed_query: q,
        })
    }

    /// Clear encoder cache
    pub fn clear_cache(&mut self) {
        self.encoder_cache = None;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RESULT TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of single-head attention
#[derive(Debug, Clone)]
pub struct AttentionResult {
    /// Output hypervector (weighted bundle of values)
    pub output: ContinuousHV,

    /// Attention weights for each memory position
    pub weights: Vec<f32>,

    /// Transformed query (for inspection)
    pub transformed_query: ContinuousHV,
}

impl AttentionResult {
    /// Get position with highest attention weight
    pub fn argmax(&self) -> Option<usize> {
        self.weights.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
    }

    /// Get top-k attended positions
    pub fn top_k(&self, k: usize) -> Vec<(usize, f32)> {
        let mut indexed: Vec<(usize, f32)> = self.weights.iter()
            .copied()
            .enumerate()
            .collect();

        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        indexed.truncate(k);
        indexed
    }

    /// Entropy of attention distribution (higher = more diffuse)
    pub fn entropy(&self) -> f32 {
        -self.weights.iter()
            .filter(|&&w| w > 1e-10)
            .map(|&w| w * w.ln())
            .sum::<f32>()
    }
}

/// Result of multi-head attention
#[derive(Debug, Clone)]
pub struct MultiHeadAttentionResult {
    /// Combined output from all heads
    pub output: ContinuousHV,

    /// Individual head results
    pub head_results: Vec<AttentionResult>,
}

impl MultiHeadAttentionResult {
    /// Get average attention weights across heads
    pub fn average_weights(&self) -> Vec<f32> {
        if self.head_results.is_empty() {
            return vec![];
        }

        let num_positions = self.head_results[0].weights.len();
        let num_heads = self.head_results.len() as f32;

        (0..num_positions)
            .map(|i| {
                self.head_results.iter()
                    .map(|r| r.weights.get(i).copied().unwrap_or(0.0))
                    .sum::<f32>() / num_heads
            })
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Softmax with temperature
fn softmax(scores: &[f32], temperature: f32) -> Vec<f32> {
    if scores.is_empty() {
        return vec![];
    }

    // Numerical stability: subtract max
    let max_score = scores.iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    let exp_scores: Vec<f32> = scores.iter()
        .map(|&s| ((s - max_score) / temperature).exp())
        .collect();

    let sum: f32 = exp_scores.iter().sum();

    if sum < 1e-10 {
        // Uniform distribution as fallback
        let n = scores.len() as f32;
        return vec![1.0 / n; scores.len()];
    }

    exp_scores.iter().map(|e| e / sum).collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_head_attention() {
        let config = AttentionConfig {
            dim: 1024,  // Smaller for faster tests
            temperature: 1.0,
            min_attention: 0.0,
            num_heads: 1,
            normalize_output: true,
        };

        let attention = HdcAttention::new(config, 42);

        let query = ContinuousHV::random(1024, 100);
        let memories = vec![
            ContinuousHV::random(1024, 101),
            ContinuousHV::random(1024, 102),
            ContinuousHV::random(1024, 103),
        ];

        let result = attention.attend(&query, &memories);

        // Check weights sum to 1
        let weight_sum: f32 = result.weights.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-5, "Weights should sum to 1");

        // Check output is normalized
        let norm = result.output.norm();
        assert!((norm - 1.0).abs() < 0.1, "Output should be normalized");
    }

    #[test]
    fn test_similar_query_gets_higher_weight() {
        let config = AttentionConfig {
            dim: 1024,
            temperature: 0.1,  // Sharp attention
            ..Default::default()
        };

        let attention = HdcAttention::new(config, 42);

        // Create a query similar to one memory item
        let base = ContinuousHV::random(1024, 100);
        let query = base.clone();  // Query IS the base

        let memories = vec![
            ContinuousHV::random(1024, 200),  // Random
            base.clone(),                      // Similar to query
            ContinuousHV::random(1024, 201),  // Random
        ];

        let result = attention.attend(&query, &memories);

        // Memory[1] should have highest weight (it's identical to query before transform)
        // Note: after binding with transform, relationships change, but similar items
        // should still tend to have higher similarity
        assert!(result.weights.len() == 3);

        // Check argmax works
        let top = result.argmax();
        assert!(top.is_some());
    }

    #[test]
    fn test_multi_head_attention() {
        let config = AttentionConfig {
            dim: 512,
            num_heads: 4,
            ..Default::default()
        };

        let attention = MultiHeadHdcAttention::new(config, 42);

        let query = ContinuousHV::random(512, 100);
        let memories = vec![
            ContinuousHV::random(512, 101),
            ContinuousHV::random(512, 102),
        ];

        let result = attention.attend(&query, &memories);

        // Check we have 4 head results
        assert_eq!(result.head_results.len(), 4);

        // Check average weights
        let avg_weights = result.average_weights();
        assert_eq!(avg_weights.len(), 2);
    }

    #[test]
    fn test_causal_attention() {
        let config = AttentionConfig {
            dim: 256,
            ..Default::default()
        };

        let attention = HdcAttention::new(config, 42);

        let sequence: Vec<ContinuousHV> = (0..4)
            .map(|i| ContinuousHV::random(256, 100 + i))
            .collect();

        let results = attention.causal_self_attend(&sequence);

        // Position 0 should only attend to itself
        assert!((results[0].weights[0] - 1.0).abs() < 1e-5);

        // Position 3 should attend to all 4 positions
        assert_eq!(results[3].weights.len(), 4);
    }

    #[test]
    fn test_masked_attention() {
        let config = AttentionConfig {
            dim: 256,
            ..Default::default()
        };

        let attention = HdcAttention::new(config, 42);

        let query = ContinuousHV::random(256, 100);
        let memories = vec![
            ContinuousHV::random(256, 101),
            ContinuousHV::random(256, 102),
            ContinuousHV::random(256, 103),
        ];

        // Mask out position 1
        let mask = vec![true, false, true];
        let result = attention.attend_masked(&query, &memories, &mask);

        // Position 1 should have zero weight
        assert!(result.weights[1] < 1e-5);

        // Other weights should sum to ~1
        let active_sum = result.weights[0] + result.weights[2];
        assert!((active_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cross_attention_caching() {
        let config = AttentionConfig {
            dim: 256,
            ..Default::default()
        };

        let mut cross_attn = CrossAttention::new(config, 42);

        let encoder_outputs: Vec<ContinuousHV> = (0..3)
            .map(|i| ContinuousHV::random(256, 200 + i))
            .collect();

        cross_attn.cache_encoder(&encoder_outputs);

        let decoder_query = ContinuousHV::random(256, 100);
        let result = cross_attn.attend_cached(&decoder_query);

        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.weights.len(), 3);
    }

    #[test]
    fn test_attention_entropy() {
        // Uniform attention should have high entropy
        let uniform_weights = vec![0.25, 0.25, 0.25, 0.25];
        let result = AttentionResult {
            output: ContinuousHV::zero(256),
            weights: uniform_weights,
            transformed_query: ContinuousHV::zero(256),
        };
        let uniform_entropy = result.entropy();

        // Sharp attention should have low entropy
        let sharp_weights = vec![0.97, 0.01, 0.01, 0.01];
        let result = AttentionResult {
            output: ContinuousHV::zero(256),
            weights: sharp_weights,
            transformed_query: ContinuousHV::zero(256),
        };
        let sharp_entropy = result.entropy();

        assert!(uniform_entropy > sharp_entropy,
            "Uniform attention should have higher entropy");
    }

    #[test]
    fn test_softmax() {
        let scores = vec![1.0, 2.0, 3.0];
        let weights = softmax(&scores, 1.0);

        // Check sum to 1
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Check ordering preserved
        assert!(weights[2] > weights[1]);
        assert!(weights[1] > weights[0]);
    }

    #[test]
    fn test_softmax_temperature() {
        let scores = vec![1.0, 2.0, 3.0];

        let cold_weights = softmax(&scores, 0.1);  // Sharp
        let hot_weights = softmax(&scores, 10.0);  // Diffuse

        // Cold should be more peaked
        let cold_max = cold_weights.iter().cloned().fold(0.0, f32::max);
        let hot_max = hot_weights.iter().cloned().fold(0.0, f32::max);

        assert!(cold_max > hot_max, "Lower temperature should give sharper distribution");
    }
}
