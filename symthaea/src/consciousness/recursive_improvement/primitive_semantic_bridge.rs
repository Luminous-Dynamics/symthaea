//! # Primitive Semantic Bridge: Grounded Concept Crystallization
//!
//! This module bridges semantic embeddings to the primitive system, enabling
//! concepts to crystallize with their primitive "birth certificate".
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    PRIMITIVE SEMANTIC BRIDGE                            │
//! │                                                                          │
//! │  Text Input                                                              │
//! │      │                                                                   │
//! │      ▼                                                                   │
//! │  ┌──────────────────┐                                                    │
//! │  │ Embedding (1024D)│ ← From SemanticBridge (Qwen3/BGE)                 │
//! │  └────────┬─────────┘                                                    │
//! │           │                                                              │
//! │           ▼                                                              │
//! │  ┌──────────────────────────────┐                                        │
//! │  │ Primitive Decomposition      │ ← Match against 186+ primitives       │
//! │  │                              │                                        │
//! │  │  Returns: [(CAUSE, 0.72),    │                                        │
//! │  │           (REMEMBER, 0.68),  │                                        │
//! │  │           (ATTEND, 0.61)]    │                                        │
//! │  └────────┬─────────────────────┘                                        │
//! │           │                                                              │
//! │           ▼                                                              │
//! │  ┌──────────────────────────────┐                                        │
//! │  │ Build Concept HV from        │ ← HV16::bundle(primitive_encodings)   │
//! │  │ Primitive Bundle             │                                        │
//! │  └────────┬─────────────────────┘                                        │
//! │           │                                                              │
//! │           ▼                                                              │
//! │  ┌──────────────────────────────┐                                        │
//! │  │ CrystalizedConcept {         │                                        │
//! │  │   hypervector: bundle(...),  │ ← Built FROM primitives               │
//! │  │   primitive_birth: [...],    │ ← Birth certificate stored            │
//! │  │ }                            │                                        │
//! │  └──────────────────────────────┘                                        │
//! │                                                                          │
//! │  RESULT: Concepts ARE their primitive composition!                       │
//! │          Decomposition = just read primitive_birth                       │
//! │          Naming = use primitive structure                                │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Innovation
//!
//! Instead of:
//! - Concept HV = blake3_hash(attractor_signature)  // Semantic-blind
//!
//! We do:
//! - Concept HV = bundle(active_primitive_encodings)  // Semantic-grounded
//!
//! This means decomposition becomes TRIVIAL - just read the birth certificate!

use symthaea_core::hdc::primitive_system::{PrimitiveSystem, PrimitiveTier, Primitive};
use symthaea_core::hdc::binary_hv::HV16;
use crate::hdc::semantic_primitive_encoder::SemanticPrimitiveEncoder;
use crate::embeddings::HdcBridge;
#[cfg(feature = "embeddings")]
use crate::embeddings::qwen3::{Qwen3Embedder, Qwen3Config, QWEN3_FULL_DIMENSION};
use super::{
    SemanticBridge, SemanticBridgeConfig, SemanticInput, ActionContext,
    ConsciousnessWorldModel, WorldModelConfig,
    ConsciousnessTransition, ConsciousnessAction, LatentConsciousnessState,
};
use crate::dynamics::CrystalizedConcept;
use std::collections::VecDeque;

/// Configuration for the primitive semantic bridge
#[derive(Debug, Clone)]
pub struct PrimitiveSemanticBridgeConfig {
    /// Minimum similarity to count as primitive activation (above random ~0.50)
    pub activation_threshold: f32,

    /// Maximum number of primitives to track per input
    pub max_active_primitives: usize,

    /// Whether to build concept HV from primitives (true) or hash (false)
    pub use_primitive_hypervectors: bool,

    /// Use aligned encoding (SAME projection for primitives and embeddings)
    /// This dramatically improves activation strength from ~51% to 60-80%
    pub use_aligned_encoding: bool,

    /// Use real ML embeddings when available (requires embeddings feature)
    /// When true, uses Qwen3/MiniLM instead of simulated embeddings
    pub use_real_embeddings: bool,

    /// Path to embedding model (default: Qwen3)
    pub embedding_model_path: Option<String>,

    /// Underlying semantic bridge config
    pub semantic_config: SemanticBridgeConfig,
}

impl Default for PrimitiveSemanticBridgeConfig {
    fn default() -> Self {
        Self {
            activation_threshold: 0.52,  // Just above random (0.50)
            max_active_primitives: 20,
            use_primitive_hypervectors: true,
            use_aligned_encoding: true,  // Use aligned encoding by default
            use_real_embeddings: true,   // Use real ML embeddings when available
            embedding_model_path: Some("models/qwen3-embedding-0.6b-onnx".to_string()),
            semantic_config: SemanticBridgeConfig::default(),
        }
    }
}

/// Statistics for the primitive semantic bridge
#[derive(Debug, Clone, Default)]
pub struct PrimitiveSemanticBridgeStats {
    /// Total inputs processed
    pub inputs_processed: u64,

    /// Total primitive activations detected
    pub primitive_activations: u64,

    /// Average primitives per input
    pub avg_primitives_per_input: f32,

    /// Concepts crystallized with primitive birth
    pub concepts_with_primitives: u64,

    /// Peak primitive activation seen
    pub peak_primitive_activation: f32,
}

/// Active primitive with activation details
#[derive(Debug, Clone)]
pub struct ActivePrimitive {
    /// Primitive name
    pub name: String,

    /// Primitive tier
    pub tier: PrimitiveTier,

    /// Activation strength (similarity)
    pub activation: f32,

    /// The primitive's encoding
    pub encoding: HV16,
}

/// Result of primitive decomposition
#[derive(Debug, Clone)]
pub struct PrimitiveDecomposition {
    /// Active primitives, sorted by activation (descending)
    pub primitives: Vec<ActivePrimitive>,

    /// Combined hypervector (bundle of active primitives)
    pub combined_hv: Option<HV16>,

    /// Dominant tier
    pub dominant_tier: Option<PrimitiveTier>,

    /// Tier distribution
    pub tier_counts: Vec<(PrimitiveTier, usize)>,
}

/// The Primitive Semantic Bridge
///
/// Connects semantic embeddings to primitives, enabling grounded crystallization.
pub struct PrimitiveSemanticBridge {
    /// Configuration
    config: PrimitiveSemanticBridgeConfig,

    /// Statistics
    stats: PrimitiveSemanticBridgeStats,

    /// Current primitive decomposition (updated on each process)
    current_decomposition: Option<PrimitiveDecomposition>,

    /// Recent decomposition history
    decomposition_history: VecDeque<PrimitiveDecomposition>,

    /// HDC bridge for projections
    hdc_bridge: HdcBridge,

    /// Real ML embedder (when embeddings feature is enabled)
    #[cfg(feature = "embeddings")]
    embedder: Option<Qwen3Embedder>,

    /// Embedding dimension (1024 for Qwen3, 384 for MiniLM)
    embed_dim: usize,
}

impl Default for PrimitiveSemanticBridge {
    fn default() -> Self {
        Self::new(PrimitiveSemanticBridgeConfig::default())
    }
}

impl PrimitiveSemanticBridge {
    /// Create a new primitive semantic bridge
    pub fn new(config: PrimitiveSemanticBridgeConfig) -> Self {
        // Try to initialize real embedder if enabled
        #[cfg(feature = "embeddings")]
        let (embedder, embed_dim) = if config.use_real_embeddings {
            let model_path = config.embedding_model_path.clone()
                .unwrap_or_else(|| "models/qwen3-embedding-0.6b-onnx".to_string());

            match Qwen3Embedder::load(&model_path) {
                Ok(emb) => {
                    let dim = emb.dimension();
                    tracing::info!(
                        "PrimitiveSemanticBridge: Loaded real embedder from {} ({}D)",
                        model_path, dim
                    );
                    (Some(emb), dim)
                }
                Err(e) => {
                    tracing::warn!(
                        "PrimitiveSemanticBridge: Failed to load embedder from {}: {}. Using simulated.",
                        model_path, e
                    );
                    (None, QWEN3_FULL_DIMENSION)
                }
            }
        } else {
            (None, QWEN3_FULL_DIMENSION)
        };

        #[cfg(not(feature = "embeddings"))]
        let embed_dim = 1024;

        Self {
            config,
            stats: PrimitiveSemanticBridgeStats::default(),
            current_decomposition: None,
            decomposition_history: VecDeque::with_capacity(50),
            hdc_bridge: HdcBridge::default(),
            #[cfg(feature = "embeddings")]
            embedder,
            embed_dim,
        }
    }

    /// Decompose an embedding into primitive activations
    ///
    /// This is the core operation: find which primitives are "active" in this embedding.
    ///
    /// With `use_aligned_encoding = true`, uses the SemanticPrimitiveEncoder which
    /// projects both primitives AND embeddings through the SAME HdcBridge. This
    /// produces meaningful similarities (60-80%) instead of random (~50%).
    pub fn decompose_embedding(&mut self, embedding: &[f32]) -> PrimitiveDecomposition {
        let mut active: Vec<ActivePrimitive> = Vec::new();

        if self.config.use_aligned_encoding {
            // ALIGNED MODE: Use SemanticPrimitiveEncoder for meaningful similarities
            let encoder = SemanticPrimitiveEncoder::global();

            // Project embedding using the SAME bridge as primitives
            let embedding_hv = encoder.project_embedding(embedding);

            // Compare against aligned primitive HV16s
            for prim in encoder.all_primitives() {
                let similarity = embedding_hv.similarity(&prim.hv16);

                if similarity > self.config.activation_threshold {
                    active.push(ActivePrimitive {
                        name: prim.name.clone(),
                        tier: prim.tier,
                        activation: similarity,
                        encoding: prim.hv16.clone(),
                    });

                    if similarity > self.stats.peak_primitive_activation {
                        self.stats.peak_primitive_activation = similarity;
                    }
                }
            }
        } else {
            // LEGACY MODE: Original hash-based encoding (produces ~51% similarities)
            let primitive_system = PrimitiveSystem::global();
            let embedding_hv = self.embedding_to_hv16(embedding);

            for primitive in primitive_system.all_primitives() {
                let similarity = embedding_hv.similarity(&primitive.encoding);

                if similarity > self.config.activation_threshold {
                    active.push(ActivePrimitive {
                        name: primitive.name.clone(),
                        tier: primitive.tier,
                        activation: similarity,
                        encoding: primitive.encoding.clone(),
                    });

                    if similarity > self.stats.peak_primitive_activation {
                        self.stats.peak_primitive_activation = similarity;
                    }
                }
            }
        }

        // Sort by activation (descending)
        active.sort_by(|a, b| b.activation.partial_cmp(&a.activation).unwrap_or(std::cmp::Ordering::Equal));

        // Truncate to max
        active.truncate(self.config.max_active_primitives);

        // Compute combined HV if we have primitives
        let combined_hv = if !active.is_empty() && self.config.use_primitive_hypervectors {
            let encodings: Vec<HV16> = active.iter()
                .map(|p| p.encoding.clone())
                .collect();
            Some(HV16::bundle(&encodings))
        } else {
            None
        };

        // Compute tier distribution
        let mut tier_map: std::collections::HashMap<PrimitiveTier, usize> = std::collections::HashMap::new();
        for p in &active {
            *tier_map.entry(p.tier).or_insert(0) += 1;
        }
        let mut tier_counts: Vec<_> = tier_map.into_iter().collect();
        tier_counts.sort_by(|a, b| b.1.cmp(&a.1));

        let dominant_tier = tier_counts.first().map(|(t, _)| *t);

        // Update stats
        self.stats.primitive_activations += active.len() as u64;

        PrimitiveDecomposition {
            primitives: active,
            combined_hv,
            dominant_tier,
            tier_counts,
        }
    }

    /// Convert embedding to HV16
    ///
    /// Projects a dense embedding into binary hypervector space.
    fn embedding_to_hv16(&self, embedding: &[f32]) -> HV16 {
        // Use deterministic projection based on embedding content
        // Hash the embedding to get a seed, then generate HV16
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        for &v in embedding {
            v.to_bits().hash(&mut hasher);
        }

        let seed = hasher.finish();

        // Generate base HV from seed
        let mut hv = HV16::random(seed);

        // Modulate by embedding features
        // Use first few dimensions to flip specific bits
        let dim = embedding.len().min(100);
        for i in 0..dim {
            if embedding[i] > 0.0 {
                // XOR with a feature-specific pattern
                let feature_hv = HV16::random(seed.wrapping_add(i as u64));
                hv = HV16::bind(&hv, &feature_hv);
            }
        }

        hv
    }

    /// Process a semantic input and return primitive-enriched result
    pub fn process(&mut self, text: &str, context: ActionContext) -> PrimitiveProcessingResult {
        self.stats.inputs_processed += 1;

        // Try real embeddings first, fall back to simulated
        let embedding = self.get_embedding(text, &context);

        // Decompose into primitives
        let decomposition = self.decompose_embedding(&embedding);

        // Update rolling average
        let n = self.stats.inputs_processed as f32;
        let current_count = decomposition.primitives.len() as f32;
        self.stats.avg_primitives_per_input =
            (self.stats.avg_primitives_per_input * (n - 1.0) + current_count) / n;

        // Store decomposition
        self.current_decomposition = Some(decomposition.clone());
        self.decomposition_history.push_back(decomposition.clone());
        if self.decomposition_history.len() > 50 {
            self.decomposition_history.pop_front();
        }

        // Extract primitive birth certificate
        let primitive_birth: Vec<(String, f32)> = decomposition.primitives.iter()
            .map(|p| (p.name.clone(), p.activation))
            .collect();

        PrimitiveProcessingResult {
            decomposition,
            primitive_birth,
            embedding,
        }
    }

    /// Get embedding for text - uses real ML embedder if available, otherwise simulated
    fn get_embedding(&mut self, text: &str, context: &ActionContext) -> Vec<f32> {
        #[cfg(feature = "embeddings")]
        {
            if let Some(ref mut embedder) = self.embedder {
                match embedder.embed(text) {
                    Ok(result) => {
                        // Pad or truncate to expected dimension if needed
                        let mut embedding = result.embedding;
                        if embedding.len() < self.embed_dim {
                            embedding.resize(self.embed_dim, 0.0);
                        } else if embedding.len() > self.embed_dim {
                            embedding.truncate(self.embed_dim);
                        }
                        return embedding;
                    }
                    Err(e) => {
                        tracing::debug!("Real embedding failed, using simulated: {}", e);
                    }
                }
            }
        }

        // Fallback to simulated embedding
        self.simulate_embedding(text, context)
    }

    /// Simulate embedding for text
    ///
    /// With aligned encoding: Uses content-addressable semantic embedding that
    /// matches the primitive embedding scheme, enabling meaningful similarity.
    ///
    /// Without aligned encoding: Uses deterministic hash-based embedding (legacy).
    fn simulate_embedding(&self, text: &str, context: &ActionContext) -> Vec<f32> {
        let dim = self.embed_dim; // Use configured dimension

        if self.config.use_aligned_encoding {
            // ALIGNED MODE: Generate semantic embedding from text content
            // This uses the same algorithm as SemanticPrimitiveEncoder
            Self::generate_semantic_embedding(text, context, dim)
        } else {
            // LEGACY MODE: Hash-based pseudo-random embedding
            let mut embedding = vec![0.0f32; dim];

            let mut hash: u64 = 0x5174_1AEA;
            for byte in text.bytes() {
                hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
            }

            let context_offset = match context {
                ActionContext::Query => 1000,
                ActionContext::Response => 2000,
                ActionContext::Error => 3000,
                ActionContext::Thought => 4000,
                ActionContext::Memory => 5000,
                ActionContext::Goal => 6000,
            };
            hash = hash.wrapping_add(context_offset);

            let mut state = hash;
            for i in 0..dim {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let val = ((state >> 33) as f32) / (u32::MAX as f32);
                embedding[i] = (val - 0.5) * 2.0;
            }

            let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 {
                for x in embedding.iter_mut() {
                    *x /= norm;
                }
            }

            embedding
        }
    }

    /// Generate semantic embedding from text content (aligned mode)
    ///
    /// Creates embeddings based on CONTENT, not just hash. Words activate
    /// specific dimensions, enabling semantic similarity.
    fn generate_semantic_embedding(text: &str, context: &ActionContext, dim: usize) -> Vec<f32> {
        let mut embedding = vec![0.0f32; dim];

        // 1. Content-based word contributions
        let lower_text = text.to_lowercase();
        let words: Vec<&str> = lower_text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 2)
            .collect();

        for (i, word) in words.iter().enumerate() {
            let word_hash = Self::content_hash(word);
            let start_dim = (word_hash as usize % (dim / 4)) * 4;

            for j in 0..4 {
                let idx = (start_dim + j) % dim;
                let sign = if (word_hash >> j) & 1 == 0 { 1.0 } else { -1.0 };
                embedding[idx] += sign * (1.0 / (i + 1) as f32);
            }
        }

        // 2. Context-specific signature
        let context_base = match context {
            ActionContext::Query => 0,
            ActionContext::Response => 128,
            ActionContext::Error => 256,
            ActionContext::Thought => 384,
            ActionContext::Memory => 512,
            ActionContext::Goal => 640,
        };

        for i in 0..64 {
            let idx = (context_base + i) % dim;
            embedding[idx] += 0.3;
        }

        // 3. Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            for x in embedding.iter_mut() {
                *x /= norm;
            }
        }

        embedding
    }

    /// Content-based hash for semantic embedding
    fn content_hash(text: &str) -> u64 {
        let mut hash: u64 = 0x5974_1AEA;
        for byte in text.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        }
        hash
    }

    /// Get current primitive decomposition
    pub fn current_decomposition(&self) -> Option<&PrimitiveDecomposition> {
        self.current_decomposition.as_ref()
    }

    /// Get decomposition history
    pub fn decomposition_history(&self) -> &VecDeque<PrimitiveDecomposition> {
        &self.decomposition_history
    }

    /// Get statistics
    pub fn stats(&self) -> &PrimitiveSemanticBridgeStats {
        &self.stats
    }

    /// Generate a primitive-based name suggestion
    pub fn suggest_name(&self, primitive_birth: &[(String, f32)]) -> String {
        if primitive_birth.is_empty() {
            return "UnknownConcept".to_string();
        }

        // Use top 2-3 primitives to form a name
        let top: Vec<&str> = primitive_birth.iter()
            .take(3)
            .map(|(name, _)| name.as_str())
            .collect();

        match top.len() {
            1 => format!("{}Pattern", top[0]),
            2 => format!("{}_{}_Pattern", top[0], top[1]),
            _ => format!("{}_{}_{}Pattern", top[0], top[1], top[2]),
        }
    }

    /// Get a human-readable description of a concept's primitive composition
    pub fn describe_primitives(&self, primitive_birth: &[(String, f32)]) -> String {
        if primitive_birth.is_empty() {
            return "No primitive composition recorded.".to_string();
        }

        let descriptions: Vec<String> = primitive_birth.iter()
            .take(5)
            .map(|(name, strength)| format!("{} ({:.0}%)", name, strength * 100.0))
            .collect();

        format!("Composed of: {}", descriptions.join(" + "))
    }
}

/// Result of primitive-enriched processing
#[derive(Debug, Clone)]
pub struct PrimitiveProcessingResult {
    /// Full decomposition details
    pub decomposition: PrimitiveDecomposition,

    /// Birth certificate for crystallization
    pub primitive_birth: Vec<(String, f32)>,

    /// The embedding used
    pub embedding: Vec<f32>,
}

/// Helper function to get tier name
pub fn tier_name(tier: PrimitiveTier) -> &'static str {
    match tier {
        PrimitiveTier::NSM => "NSM (Semantic)",
        PrimitiveTier::Mathematical => "Mathematical",
        PrimitiveTier::Physical => "Physical",
        PrimitiveTier::Geometric => "Geometric",
        PrimitiveTier::Strategic => "Strategic",
        PrimitiveTier::MetaCognitive => "MetaCognitive",
        PrimitiveTier::Temporal => "Temporal",
        PrimitiveTier::Compositional => "Compositional",
        PrimitiveTier::Consciousness => "Consciousness",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let bridge = PrimitiveSemanticBridge::default();
        assert_eq!(bridge.stats.inputs_processed, 0);
    }

    #[test]
    fn test_process_query() {
        let mut bridge = PrimitiveSemanticBridge::default();

        let result = bridge.process("How do I configure nginx?", ActionContext::Query);

        // Should have some primitives (even if weak)
        // With our threshold of 0.52, we might not get many
        assert!(result.embedding.len() == 1024);
    }

    #[test]
    fn test_decomposition_stored() {
        let mut bridge = PrimitiveSemanticBridge::default();

        bridge.process("Test input", ActionContext::Query);

        assert!(bridge.current_decomposition().is_some());
        assert_eq!(bridge.decomposition_history().len(), 1);
    }
}
