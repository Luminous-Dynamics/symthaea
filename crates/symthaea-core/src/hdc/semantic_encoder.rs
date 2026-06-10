// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Universal Semantic Encoder for HDC
//!
//! Converts text to semantically-meaningful hypervectors using a hybrid approach:
//! 1. Check embedding cache (instant, <1ms)
//! 2. ONNX model if `embeddings` feature enabled (accurate, ~50ms)
//! 3. Fall back to character n-grams (fast, no semantics)
//!
//! ## Architecture
//!
//! ```text
//! Text → Sentence Embedding (384-dim) → Random Projection → ContinuousHV (16,384-dim)
//! ```
//!
//! ## Why Random Projection Works
//!
//! The Johnson-Lindenstrauss lemma guarantees that random projection preserves
//! pairwise distances with high probability. This means semantic similarity
//! in the original embedding space is preserved in HDC space.
//!
//! ## Features
//!
//! - `embeddings`: Enable ONNX-based sentence transformer (all-MiniLM-L6-v2)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::hdc::semantic_encoder::{SemanticEncoder, create_encoder, EncoderType};
//!
//! // Create best available encoder
//! let encoder = create_encoder(EncoderType::CachedSemantic);
//! let hv = encoder.encode("I helped an old lady cross the street");
//! ```

use super::HDC_DIMENSION;
use super::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::RwLock;

// ============================================================================
// Trait Definition
// ============================================================================

/// Universal semantic encoding trait
pub trait SemanticEncoder: Send + Sync {
    /// Encode text to semantically-meaningful hypervector
    fn encode(&self, text: &str) -> ContinuousHV;

    /// Batch encoding (more efficient for multiple texts)
    fn encode_batch(&self, texts: &[&str]) -> Vec<ContinuousHV> {
        texts.iter().map(|t| self.encode(t)).collect()
    }

    /// Compute semantic similarity between two texts
    fn similarity(&self, a: &str, b: &str) -> f64 {
        let hv_a = self.encode(a);
        let hv_b = self.encode(b);
        hv_a.similarity(&hv_b) as f64
    }

    /// Get encoder name for logging
    fn name(&self) -> &'static str;
}

// ============================================================================
// Random Projection Matrix
// ============================================================================

/// Generates a deterministic random projection matrix for dimensionality expansion
/// Uses the Johnson-Lindenstrauss lemma to preserve pairwise distances
pub struct RandomProjection {
    /// Projection matrix: input_dim × output_dim
    matrix: Vec<Vec<f32>>,
    input_dim: usize,
    output_dim: usize,
}

impl RandomProjection {
    /// Create a new random projection matrix
    ///
    /// Uses a fixed seed for reproducibility across runs
    pub fn new(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut matrix = vec![vec![0.0f32; output_dim]; input_dim];

        // Generate pseudo-random values using hash function
        // Each element is ±1/√d (sparse random projection)
        let scale = 1.0 / (input_dim as f32).sqrt();

        for i in 0..input_dim {
            for j in 0..output_dim {
                let mut hasher = DefaultHasher::new();
                (seed, i, j).hash(&mut hasher);
                let hash = hasher.finish();

                // Use hash to determine sign
                matrix[i][j] = if hash.is_multiple_of(2) {
                    scale
                } else {
                    -scale
                };
            }
        }

        Self {
            matrix,
            input_dim,
            output_dim,
        }
    }

    /// Project a vector from input_dim to output_dim.
    ///
    /// Returns `Err` if `input.len()` does not match the configured `input_dim`.
    pub fn project(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        if input.len() != self.input_dim {
            return Err(format!(
                "RandomProjection input dimension mismatch: expected {}, got {}",
                self.input_dim,
                input.len()
            ));
        }

        let mut output = vec![0.0f32; self.output_dim];

        for (i, &x) in input.iter().enumerate() {
            for j in 0..self.output_dim {
                output[j] += x * self.matrix[i][j];
            }
        }

        Ok(output)
    }
}

// ============================================================================
// Character N-gram Encoder (Baseline)
// ============================================================================

/// Simple character n-gram encoder (no semantic understanding)
///
/// This is the baseline encoder that doesn't require any external dependencies.
/// It's fast but doesn't capture semantic meaning.
pub struct CharNgramEncoder {
    dim: usize,
    seed: u64,
}

impl CharNgramEncoder {
    pub fn new(seed: u64) -> Self {
        Self {
            dim: HDC_DIMENSION,
            seed,
        }
    }

    fn char_to_hv(&self, c: char) -> ContinuousHV {
        ContinuousHV::random(self.dim, self.seed + c as u64)
    }
}

impl SemanticEncoder for CharNgramEncoder {
    fn encode(&self, text: &str) -> ContinuousHV {
        let chars: Vec<char> = text.to_lowercase().chars().collect();
        let mut result = ContinuousHV::zero(self.dim);

        // Bind trigrams
        for window in chars.windows(3) {
            let v1 = self.char_to_hv(window[0]);
            let v2 = self.char_to_hv(window[1]);
            let v3 = self.char_to_hv(window[2]);

            let bound = v1.bind(&v2).bind(&v3);
            result = result.add(&bound);
        }

        result.normalize()
    }

    fn name(&self) -> &'static str {
        "CharNgram"
    }
}

// ============================================================================
// Cached Embedding Encoder
// ============================================================================

/// Pre-computed embedding format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingEntry {
    pub text_hash: u64,
    pub embedding: Vec<f32>,
}

/// Cached semantic encoder using pre-computed embeddings
///
/// Embeddings are computed offline (e.g., using Python sentence-transformers)
/// and loaded at runtime for fast lookup.
pub struct CachedSemanticEncoder {
    /// Cache: text_hash -> embedding (384-dim typical)
    cache: RwLock<HashMap<u64, Vec<f32>>>,
    /// Random projection matrix for dimensionality expansion
    projection: RandomProjection,
    /// Fallback encoder for cache misses
    fallback: CharNgramEncoder,
    /// Embedding dimension (before projection)
    embedding_dim: usize,
}

impl CachedSemanticEncoder {
    /// Create a new cached encoder
    ///
    /// If cache_path exists, loads pre-computed embeddings.
    /// Otherwise, starts with empty cache (falls back to CharNgram).
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            projection: RandomProjection::new(embedding_dim, HDC_DIMENSION, 42),
            fallback: CharNgramEncoder::new(42),
            embedding_dim,
        }
    }

    /// Load embeddings from a JSON file
    pub fn load_from_json<P: AsRef<Path>>(&self, path: P) -> Result<usize, String> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| format!("Failed to read embeddings: {e}"))?;

        let entries: Vec<EmbeddingEntry> = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse embeddings: {e}"))?;

        let mut cache = self.cache.write().unwrap_or_else(|e| e.into_inner());
        let count = entries.len();

        for entry in entries {
            cache.insert(entry.text_hash, entry.embedding);
        }

        Ok(count)
    }

    /// Add a single embedding to the cache
    pub fn add_embedding(&self, text: &str, embedding: Vec<f32>) {
        let hash = Self::hash_text(text);
        let mut cache = self.cache.write().unwrap_or_else(|e| e.into_inner());
        cache.insert(hash, embedding);
    }

    /// Hash text for cache lookup
    fn hash_text(text: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let normalized = text.to_lowercase().trim().to_string();
        let mut hasher = DefaultHasher::new();
        normalized.hash(&mut hasher);
        hasher.finish()
    }

    /// Check if text is in cache
    pub fn is_cached(&self, text: &str) -> bool {
        let hash = Self::hash_text(text);
        self.cache
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(&hash)
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.read().unwrap_or_else(|e| e.into_inner());
        (cache.len(), self.embedding_dim)
    }
}

impl SemanticEncoder for CachedSemanticEncoder {
    fn encode(&self, text: &str) -> ContinuousHV {
        let hash = Self::hash_text(text);

        // Try cache first
        if let Some(embedding) = self
            .cache
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(&hash)
        {
            // Project from embedding space to HDC space
            match self.projection.project(embedding) {
                Ok(projected) => return ContinuousHV::from_values(projected).normalize(),
                Err(_) => return self.fallback.encode(text),
            }
        }

        // Fall back to character n-grams
        self.fallback.encode(text)
    }

    fn name(&self) -> &'static str {
        "CachedSemantic"
    }
}

// ============================================================================
// Word Embedding Aggregation Encoder
// ============================================================================

/// Word embedding based encoder using pre-loaded word vectors
///
/// Aggregates word embeddings using weighted average (TF-IDF style weighting optional)
pub struct WordEmbeddingEncoder {
    /// Word -> embedding mapping
    word_vectors: HashMap<String, Vec<f32>>,
    /// Random projection for dimensionality expansion
    projection: RandomProjection,
    /// Fallback for OOV words
    fallback: CharNgramEncoder,
    /// Word embedding dimension
    word_dim: usize,
}

impl WordEmbeddingEncoder {
    /// Create a new word embedding encoder
    pub fn new(word_dim: usize) -> Self {
        Self {
            word_vectors: HashMap::new(),
            projection: RandomProjection::new(word_dim, HDC_DIMENSION, 42),
            fallback: CharNgramEncoder::new(42),
            word_dim,
        }
    }

    /// Load word vectors from a text file (word2vec/GloVe format)
    ///
    /// Format: word dim1 dim2 dim3 ...
    pub fn load_word_vectors<P: AsRef<Path>>(
        &mut self,
        path: P,
        max_words: usize,
    ) -> Result<usize, String> {
        use std::io::{BufRead, BufReader};

        let file = std::fs::File::open(path.as_ref())
            .map_err(|e| format!("Failed to open word vectors: {e}"))?;

        let reader = BufReader::new(file);
        let mut count = 0;

        for line in reader.lines().take(max_words) {
            let line = line.map_err(|e| format!("Read error: {e}"))?;
            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts.len() < self.word_dim + 1 {
                continue;
            }

            let word = parts[0].to_lowercase();
            let embedding: Vec<f32> = parts[1..]
                .iter()
                .take(self.word_dim)
                .filter_map(|s| s.parse().ok())
                .collect();

            if embedding.len() == self.word_dim {
                self.word_vectors.insert(word, embedding);
                count += 1;
            }
        }

        Ok(count)
    }

    /// Get embedding for a word (or zero vector if OOV)
    fn get_word_embedding(&self, word: &str) -> Option<&Vec<f32>> {
        self.word_vectors.get(&word.to_lowercase())
    }

    /// Aggregate word embeddings using mean pooling
    fn aggregate_embeddings(&self, embeddings: &[&Vec<f32>]) -> Vec<f32> {
        if embeddings.is_empty() {
            return vec![0.0; self.word_dim];
        }

        let mut result = vec![0.0f32; self.word_dim];
        for emb in embeddings {
            for (i, &v) in emb.iter().enumerate() {
                result[i] += v;
            }
        }

        let n = embeddings.len() as f32;
        for v in &mut result {
            *v /= n;
        }

        result
    }
}

impl SemanticEncoder for WordEmbeddingEncoder {
    fn encode(&self, text: &str) -> ContinuousHV {
        // Tokenize (simple whitespace + punctuation removal)
        let lowercase = text.to_lowercase();
        let words: Vec<&str> = lowercase
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .collect();

        // Get embeddings for known words
        let embeddings: Vec<&Vec<f32>> = words
            .iter()
            .filter_map(|w| self.get_word_embedding(w))
            .collect();

        if embeddings.is_empty() {
            // Fall back to character n-grams if no words found
            return self.fallback.encode(text);
        }

        // Aggregate and project
        let aggregated = self.aggregate_embeddings(&embeddings);
        match self.projection.project(&aggregated) {
            Ok(projected) => ContinuousHV::from_values(projected).normalize(),
            Err(_) => self.fallback.encode(text),
        }
    }

    fn name(&self) -> &'static str {
        "WordEmbedding"
    }
}

// ============================================================================
// Moral Semantic Encoder (Pure Rust - No External Dependencies)
// ============================================================================

/// Moral semantic encoder optimized for ethical text analysis
///
/// Uses a pre-defined vocabulary of morally-relevant terms organized into
/// semantic categories. Encodes text by combining category-specific base vectors
/// with position encoding and action/entity relationships.
///
/// This provides significantly better ethical text understanding than CharNgram
/// without requiring any external dependencies or model downloads.
pub struct MoralSemanticEncoder {
    /// Category -> base hypervector (each category has a unique random HV)
    category_bases: HashMap<&'static str, ContinuousHV>,
    /// Word -> (category, valence) mapping
    moral_vocabulary: HashMap<&'static str, (&'static str, f32)>,
    /// Position encoding for word order
    position_bases: Vec<ContinuousHV>,
    /// Dimension
    dim: usize,
    seed: u64,
}

impl MoralSemanticEncoder {
    pub fn new(seed: u64) -> Self {
        let dim = HDC_DIMENSION;

        // Define moral categories
        let categories = [
            "action_positive",   // help, save, protect, give
            "action_negative",   // harm, steal, lie, cheat
            "entity_vulnerable", // child, elderly, disabled, sick
            "entity_authority",  // parent, teacher, judge, doctor
            "entity_neutral",    // friend, stranger, colleague
            "consequence_good",  // benefit, improve, heal
            "consequence_bad",   // hurt, damage, destroy
            "context_duty",      // promise, obligation, responsibility
            "context_fairness",  // equal, fair, just, deserve
            "context_care",      // love, compassion, empathy
            "modifier_intent",   // deliberately, accidentally, knowingly
            "modifier_degree",   // slightly, severely, completely
        ];

        // Create base vectors for each category
        let mut category_bases = HashMap::new();
        for (i, cat) in categories.iter().enumerate() {
            category_bases.insert(*cat, ContinuousHV::random(dim, seed + i as u64 * 1000));
        }

        // Build moral vocabulary with categories and valence (-1 to +1)
        let mut moral_vocabulary = HashMap::new();

        // Positive actions
        for (word, valence) in [
            ("help", 0.8),
            ("helped", 0.8),
            ("helping", 0.8),
            ("save", 0.9),
            ("saved", 0.9),
            ("saving", 0.9),
            ("protect", 0.8),
            ("protected", 0.8),
            ("protecting", 0.8),
            ("give", 0.6),
            ("gave", 0.6),
            ("giving", 0.6),
            ("share", 0.6),
            ("shared", 0.6),
            ("sharing", 0.6),
            ("donate", 0.7),
            ("donated", 0.7),
            ("donating", 0.7),
            ("support", 0.6),
            ("supported", 0.6),
            ("supporting", 0.6),
            ("assist", 0.6),
            ("assisted", 0.6),
            ("assisting", 0.6),
            ("care", 0.7),
            ("cared", 0.7),
            ("caring", 0.7),
            ("comfort", 0.6),
            ("comforted", 0.6),
            ("comforting", 0.6),
        ] {
            moral_vocabulary.insert(word, ("action_positive", valence));
        }

        // Negative actions
        for (word, valence) in [
            ("harm", -0.9),
            ("harmed", -0.9),
            ("harming", -0.9),
            ("steal", -0.8),
            ("stole", -0.8),
            ("stolen", -0.8),
            ("stealing", -0.8),
            ("lie", -0.7),
            ("lied", -0.7),
            ("lying", -0.7),
            ("cheat", -0.8),
            ("cheated", -0.8),
            ("cheating", -0.8),
            ("hurt", -0.8),
            ("hurting", -0.8),
            ("kill", -1.0),
            ("killed", -1.0),
            ("killing", -1.0),
            ("attack", -0.9),
            ("attacked", -0.9),
            ("attacking", -0.9),
            ("betray", -0.9),
            ("betrayed", -0.9),
            ("betraying", -0.9),
            ("deceive", -0.8),
            ("deceived", -0.8),
            ("deceiving", -0.8),
            ("abuse", -0.9),
            ("abused", -0.9),
            ("abusing", -0.9),
            ("neglect", -0.7),
            ("neglected", -0.7),
            ("neglecting", -0.7),
            ("ignore", -0.5),
            ("ignored", -0.5),
            ("ignoring", -0.5),
            ("break", -0.6),
            ("broke", -0.6),
            ("broken", -0.6),
            ("breaking", -0.6),
        ] {
            moral_vocabulary.insert(word, ("action_negative", valence));
        }

        // Vulnerable entities
        for word in [
            "child", "children", "baby", "babies", "infant", "elderly", "disabled", "sick",
            "patient", "victim", "homeless", "poor", "orphan", "widow", "refugee",
        ] {
            moral_vocabulary.insert(word, ("entity_vulnerable", 0.8));
        }

        // Authority entities
        for word in [
            "parent",
            "parents",
            "father",
            "mother",
            "teacher",
            "judge",
            "doctor",
            "nurse",
            "police",
            "officer",
            "boss",
            "employer",
            "leader",
            "authority",
            "guardian",
        ] {
            moral_vocabulary.insert(word, ("entity_authority", 0.6));
        }

        // Neutral entities
        for word in [
            "friend",
            "friends",
            "stranger",
            "strangers",
            "colleague",
            "neighbor",
            "person",
            "people",
            "someone",
            "anyone",
            "everyone",
            "man",
            "woman",
            "individual",
        ] {
            moral_vocabulary.insert(word, ("entity_neutral", 0.3));
        }

        // Good consequences
        for word in [
            "benefit",
            "benefited",
            "improve",
            "improved",
            "heal",
            "healed",
            "save",
            "prosper",
            "flourish",
            "thrive",
            "succeed",
            "happy",
            "happiness",
            "joy",
            "relief",
            "safe",
            "safety",
            "better",
        ] {
            moral_vocabulary.insert(word, ("consequence_good", 0.7));
        }

        // Bad consequences
        for word in [
            "damage",
            "damaged",
            "destroy",
            "destroyed",
            "suffer",
            "suffered",
            "pain",
            "loss",
            "death",
            "injury",
            "injured",
            "trauma",
            "distress",
            "grief",
            "sad",
            "sadness",
            "worse",
            "worst",
        ] {
            moral_vocabulary.insert(word, ("consequence_bad", -0.7));
        }

        // Duty context
        for word in [
            "promise",
            "promised",
            "obligation",
            "duty",
            "responsibility",
            "contract",
            "agreement",
            "vow",
            "oath",
            "commitment",
            "trust",
            "loyal",
            "loyalty",
            "faithful",
        ] {
            moral_vocabulary.insert(word, ("context_duty", 0.5));
        }

        // Fairness context
        for word in [
            "equal",
            "equality",
            "fair",
            "fairness",
            "just",
            "justice",
            "deserve",
            "deserved",
            "right",
            "rights",
            "honest",
            "honesty",
            "impartial",
            "balanced",
            "equitable",
        ] {
            moral_vocabulary.insert(word, ("context_fairness", 0.6));
        }

        // Care context
        for word in [
            "love",
            "loved",
            "compassion",
            "compassionate",
            "empathy",
            "kindness",
            "kind",
            "gentle",
            "tender",
            "warm",
            "affection",
            "sympathy",
            "concern",
            "worried",
        ] {
            moral_vocabulary.insert(word, ("context_care", 0.6));
        }

        // Intent modifiers
        for (word, valence) in [
            ("deliberately", 0.9),
            ("intentionally", 0.9),
            ("purposely", 0.9),
            ("accidentally", 0.2),
            ("unintentionally", 0.2),
            ("mistakenly", 0.2),
            ("knowingly", 0.8),
            ("unknowingly", 0.2),
            ("willfully", 0.9),
        ] {
            moral_vocabulary.insert(word, ("modifier_intent", valence));
        }

        // Degree modifiers
        for (word, valence) in [
            ("slightly", 0.2),
            ("somewhat", 0.3),
            ("moderately", 0.5),
            ("significantly", 0.7),
            ("severely", 0.9),
            ("completely", 1.0),
            ("totally", 1.0),
            ("barely", 0.1),
            ("extremely", 0.95),
        ] {
            moral_vocabulary.insert(word, ("modifier_degree", valence));
        }

        // Create position encoding vectors (up to 50 positions)
        let position_bases: Vec<ContinuousHV> = (0..50)
            .map(|i| ContinuousHV::random(dim, seed + 10000 + i as u64 * 100))
            .collect();

        Self {
            category_bases,
            moral_vocabulary,
            position_bases,
            dim,
            seed,
        }
    }

    /// Tokenize text into words
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '\'')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }

    /// Get the base vector for a word's category
    fn word_to_category_hv(&self, word: &str) -> Option<(&'static str, f32, ContinuousHV)> {
        self.moral_vocabulary.get(word).map(|(cat, valence)| {
            let base = self
                .category_bases
                .get(cat)
                .expect("category base must exist for known moral vocabulary category")
                .clone();
            (*cat, *valence, base)
        })
    }
}

impl SemanticEncoder for MoralSemanticEncoder {
    fn encode(&self, text: &str) -> ContinuousHV {
        let words = self.tokenize(text);

        if words.is_empty() {
            return ContinuousHV::zero(self.dim);
        }

        let mut components = Vec::new();
        let mut category_counts: HashMap<&str, f32> = HashMap::new();
        let mut total_valence = 0.0f32;
        let mut valence_count = 0;

        // Process each word
        for (pos, word) in words.iter().enumerate() {
            // Check if word is in moral vocabulary
            if let Some((category, valence, base_hv)) = self.word_to_category_hv(word) {
                // Position-encoded category vector: pos_hv ⊗ category_base_hv
                let pos_hv = &self.position_bases[pos.min(49)];
                let positioned = pos_hv.bind(&base_hv);

                // Scale by absolute valence (importance)
                let scaled = positioned.scale(valence.abs());
                components.push(scaled);

                // Track category distribution
                *category_counts.entry(category).or_insert(0.0) += 1.0;
                total_valence += valence;
                valence_count += 1;
            } else {
                // For unknown words, use character-hash based encoding
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                use std::hash::{Hash, Hasher};
                word.hash(&mut hasher);
                let word_hv = ContinuousHV::random(self.dim, hasher.finish());

                // Position-encode unknown words too
                let pos_hv = &self.position_bases[pos.min(49)];
                components.push(pos_hv.bind(&word_hv).scale(0.1)); // Lower weight
            }
        }

        if components.is_empty() {
            return ContinuousHV::zero(self.dim);
        }

        // Bundle all components
        let bundled = ContinuousHV::bundle_owned(&components);

        // Add category distribution as a "signature" vector
        // This captures the overall moral profile of the text
        let mut signature_components = Vec::new();
        for (cat, count) in &category_counts {
            if let Some(cat_base) = self.category_bases.get(cat) {
                let weight = count / words.len() as f32;
                signature_components.push(cat_base.scale(weight * 0.5));
            }
        }

        // Add valence indicator (overall positive/negative)
        let avg_valence = if valence_count > 0 {
            total_valence / valence_count as f32
        } else {
            0.0
        };
        let valence_indicator =
            ContinuousHV::random(self.dim, self.seed + 99999).scale(avg_valence * 0.3);
        signature_components.push(valence_indicator);

        // Combine bundled content with signature
        let signature = if !signature_components.is_empty() {
            ContinuousHV::bundle_owned(&signature_components)
        } else {
            ContinuousHV::zero(self.dim)
        };

        // Final encoding: content + signature
        bundled.add(&signature).normalize()
    }

    fn name(&self) -> &'static str {
        "MoralSemantic"
    }
}

// ============================================================================
// ONNX Encoder (requires `embeddings` feature)
// ============================================================================

#[cfg(feature = "embeddings")]
pub mod onnx {
    use super::*;
    use ort::session::Session;
    use std::sync::Mutex;

    /// ONNX-based sentence transformer encoder
    ///
    /// Uses all-MiniLM-L6-v2 model for high-quality semantic embeddings.
    /// Requires the `embeddings` feature to be enabled.
    pub struct OnnxSemanticEncoder {
        /// Session wrapped in Mutex for interior mutability (ORT 2.0 requires &mut)
        session: Mutex<Session>,
        tokenizer: tokenizers::Tokenizer,
        projection: RandomProjection,
        embedding_dim: usize,
    }

    impl OnnxSemanticEncoder {
        /// Create a new ONNX encoder
        ///
        /// Downloads the model from HuggingFace Hub if not cached locally.
        pub fn new() -> Result<Self, String> {
            // Model: sentence-transformers/all-MiniLM-L6-v2
            let embedding_dim = 384;

            // Initialize ONNX Runtime (ORT 2.0: commit() returns bool)
            ort::init().with_name("symthaea").commit();

            // Download model from HuggingFace Hub
            let api = hf_hub::api::sync::Api::new()
                .map_err(|e| format!("Failed to init HF Hub: {}", e))?;

            let repo = api.model("sentence-transformers/all-MiniLM-L6-v2".to_string());

            // The ONNX export lives under `onnx/model.onnx` in the
            // current HuggingFace layout (the bare `model.onnx` path
            // returns 404). Verified 2026-04-13 via curl HEAD on the
            // resolve/main URL.
            let model_path = repo
                .get("onnx/model.onnx")
                .map_err(|e| format!("Failed to download model: {}", e))?;

            let tokenizer_path = repo
                .get("tokenizer.json")
                .map_err(|e| format!("Failed to download tokenizer: {}", e))?;

            // Load tokenizer
            let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

            // Load ONNX session
            let session = Session::builder()
                .map_err(|e| format!("Failed to create session builder: {}", e))?
                .with_intra_threads(4)
                .map_err(|e| format!("Failed to set threads: {}", e))?
                .commit_from_file(&model_path)
                .map_err(|e| format!("Failed to load model: {}", e))?;

            Ok(Self {
                session: Mutex::new(session),
                tokenizer,
                projection: RandomProjection::new(embedding_dim, HDC_DIMENSION, 42),
                embedding_dim,
            })
        }

        /// Encode a single text to embedding
        fn encode_to_embedding(&self, text: &str) -> Result<Vec<f32>, String> {
            // Tokenize
            let encoding = self
                .tokenizer
                .encode(text, true)
                .map_err(|e| format!("Tokenization failed: {}", e))?;

            // Lock the session for inference (ORT 2.0 requires &mut)
            let mut session = self
                .session
                .lock()
                .map_err(|e| format!("Failed to lock session: {}", e))?;

            let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
            let attention_mask: Vec<i64> = encoding
                .get_attention_mask()
                .iter()
                .map(|&x| x as i64)
                .collect();
            let token_type_ids: Vec<i64> =
                encoding.get_type_ids().iter().map(|&x| x as i64).collect();

            let seq_len = input_ids.len();

            // Create input tensors using ort 2.0 API with (shape, data) tuple format
            use ort::value::Tensor;

            // ort 2.0: Use shape tuple + Vec for tensor creation
            let input_ids_tensor = Tensor::from_array(([1usize, seq_len], input_ids))
                .map_err(|e| format!("Failed to create input_ids tensor: {}", e))?;
            let attention_mask_tensor = Tensor::from_array(([1usize, seq_len], attention_mask))
                .map_err(|e| format!("Failed to create attention_mask tensor: {}", e))?;
            let token_type_ids_tensor = Tensor::from_array(([1usize, seq_len], token_type_ids))
                .map_err(|e| format!("Failed to create token_type_ids tensor: {}", e))?;

            // Run inference using the locked session
            let outputs = session
                .run(ort::inputs![
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attention_mask_tensor,
                    "token_type_ids" => token_type_ids_tensor,
                ])
                .map_err(|e| format!("Inference failed: {}", e))?;

            // Extract sentence embedding (mean pooling over tokens)
            // ort 2.0 API: try_extract_tensor returns (&Shape, &[T]) tuple
            let extracted = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Failed to extract output: {}", e))?;

            // Shape is [1, seq_len, embedding_dim]
            let shape: Vec<usize> = extracted.0.iter().map(|&d| d as usize).collect();
            let hidden_dim = if shape.len() >= 3 {
                shape[2]
            } else {
                self.embedding_dim
            };

            // Mean pooling: average over sequence dimension (dim 1)
            let mut embedding = vec![0.0f32; self.embedding_dim];
            let data: Vec<f32> = extracted.1.to_vec();

            for token_idx in 0..seq_len {
                for dim_idx in 0..self.embedding_dim.min(hidden_dim) {
                    // Index into flattened [1, seq_len, hidden_dim] tensor
                    let idx = token_idx * hidden_dim + dim_idx;
                    embedding[dim_idx] += data[idx];
                }
            }

            // Average and normalize
            for v in &mut embedding {
                *v /= seq_len as f32;
            }

            // L2 normalize
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                for v in &mut embedding {
                    *v /= norm;
                }
            }

            Ok(embedding)
        }
    }

    impl SemanticEncoder for OnnxSemanticEncoder {
        fn encode(&self, text: &str) -> ContinuousHV {
            match self.encode_to_embedding(text) {
                Ok(embedding) => match self.projection.project(&embedding) {
                    Ok(projected) => ContinuousHV::from_values(projected).normalize(),
                    Err(_) => CharNgramEncoder::new(42).encode(text),
                },
                Err(_e) => {
                    // Fall back to char n-gram on error
                    CharNgramEncoder::new(42).encode(text)
                }
            }
        }

        fn name(&self) -> &'static str {
            "OnnxSemantic"
        }
    }
}

// ============================================================================
// Encoder Factory
// ============================================================================

/// Available encoder types
#[derive(Debug, Clone, Copy)]
pub enum EncoderType {
    /// Character n-grams (fast, no semantics)
    CharNgram,
    /// Cached sentence embeddings (fast, semantic)
    CachedSemantic,
    /// Word embedding aggregation (medium, semantic)
    WordEmbedding,
    /// Moral semantic encoder (fast, ethical semantics, pure Rust)
    MoralSemantic,
    /// ONNX sentence transformer (accurate, requires `embeddings` feature)
    #[cfg(feature = "embeddings")]
    OnnxSemantic,
}

/// Create an encoder of the specified type
pub fn create_encoder(encoder_type: EncoderType) -> Box<dyn SemanticEncoder> {
    match encoder_type {
        EncoderType::CharNgram => Box::new(CharNgramEncoder::new(42)),
        EncoderType::CachedSemantic => Box::new(CachedSemanticEncoder::new(384)),
        EncoderType::WordEmbedding => Box::new(WordEmbeddingEncoder::new(300)),
        EncoderType::MoralSemantic => Box::new(MoralSemanticEncoder::new(42)),
        #[cfg(feature = "embeddings")]
        EncoderType::OnnxSemantic => match onnx::OnnxSemanticEncoder::new() {
            Ok(encoder) => Box::new(encoder),
            Err(e) => {
                eprintln!(
                    "Warning: Failed to create ONNX encoder ({}), falling back to MoralSemantic",
                    e
                );
                Box::new(MoralSemanticEncoder::new(42))
            }
        },
    }
}

/// Create the best available encoder
///
/// Tries ONNX first (if feature enabled), then MoralSemantic for ethics,
/// finally CharNgram as baseline.
pub fn create_best_encoder() -> Box<dyn SemanticEncoder> {
    #[cfg(feature = "embeddings")]
    {
        match onnx::OnnxSemanticEncoder::new() {
            Ok(encoder) => return Box::new(encoder),
            Err(e) => {
                eprintln!(
                    "Warning: ONNX encoder unavailable ({}), using MoralSemantic",
                    e
                );
            }
        }
    }

    // MoralSemantic is the best pure-Rust option for ethical text
    Box::new(MoralSemanticEncoder::new(42))
}

/// Create the best encoder specifically for ethical text analysis
///
/// Always uses MoralSemanticEncoder unless ONNX is available.
pub fn create_ethics_encoder() -> Box<dyn SemanticEncoder> {
    #[cfg(feature = "embeddings")]
    {
        if let Ok(encoder) = onnx::OnnxSemanticEncoder::new() {
            return Box::new(encoder);
        }
    }

    Box::new(MoralSemanticEncoder::new(42))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_char_ngram_encoder() {
        let encoder = CharNgramEncoder::new(42);

        let hv1 = encoder.encode("hello world");
        let hv2 = encoder.encode("hello world");
        let hv3 = encoder.encode("goodbye universe");

        // Same text should give same encoding
        assert!((hv1.similarity(&hv2) - 1.0).abs() < 0.001);

        // Different text should give different encoding
        assert!(hv1.similarity(&hv3) < 0.9);
    }

    #[test]
    fn test_random_projection() {
        let proj = RandomProjection::new(10, 100, 42);

        let v1 = vec![1.0f32; 10];
        let v2 = vec![0.5f32; 10];

        let p1 = proj.project(&v1).unwrap();
        let p2 = proj.project(&v2).unwrap();

        assert_eq!(p1.len(), 100);
        assert_eq!(p2.len(), 100);

        // Projections should be different
        assert!(p1 != p2);
    }

    #[test]
    fn test_cached_encoder() {
        let encoder = CachedSemanticEncoder::new(384);

        // Add a test embedding
        let test_embedding = vec![0.1f32; 384];
        encoder.add_embedding("test sentence", test_embedding.clone());

        // Check it's cached
        assert!(encoder.is_cached("test sentence"));
        assert!(encoder.is_cached("TEST SENTENCE")); // Case insensitive

        // Encode should use cached value
        let hv = encoder.encode("test sentence");
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_semantic_similarity_concept() {
        let encoder = CharNgramEncoder::new(42);

        // Similar sentences should have higher similarity than unrelated ones
        let s1 = "The cat sat on the mat";
        let s2 = "The cat sat on the rug";
        let s3 = "Quantum physics is complex";

        let sim_12 = encoder.similarity(s1, s2);
        let sim_13 = encoder.similarity(s1, s3);

        // s1 and s2 share more n-grams than s1 and s3
        assert!(
            sim_12 > sim_13,
            "Similar sentences should have higher n-gram overlap"
        );
    }

    #[test]
    fn test_moral_semantic_encoder_basic() {
        let encoder = MoralSemanticEncoder::new(42);

        let hv1 = encoder.encode("I helped an old lady cross the street");
        let hv2 = encoder.encode("I helped an old lady cross the street");
        let hv3 = encoder.encode("I stole money from my friend");

        // Same text should give same encoding
        assert!((hv1.similarity(&hv2) - 1.0).abs() < 0.001);

        // Different moral valence should give different encoding
        assert!(hv1.similarity(&hv3) < 0.9);
    }

    #[test]
    fn test_moral_semantic_captures_valence() {
        let encoder = MoralSemanticEncoder::new(42);

        // Positive ethical scenarios
        let positive1 = encoder.encode("I helped my friend when they were sick");
        let positive2 = encoder.encode("She donated money to help the homeless");

        // Negative ethical scenarios
        let negative1 = encoder.encode("He stole from the elderly victim");
        let negative2 = encoder.encode("They lied and betrayed their friend");

        // Same-valence scenarios should be more similar than opposite-valence
        let pos_sim = positive1.similarity(&positive2);
        let neg_sim = negative1.similarity(&negative2);
        let cross_sim1 = positive1.similarity(&negative1);
        let cross_sim2 = positive2.similarity(&negative2);

        // Positive scenarios should be more similar to each other
        assert!(
            pos_sim > cross_sim1,
            "Same-valence scenarios should cluster: pos_sim={:.3}, cross={:.3}",
            pos_sim,
            cross_sim1
        );
        // Negative scenarios should be more similar to each other
        assert!(
            neg_sim > cross_sim2,
            "Same-valence scenarios should cluster: neg_sim={:.3}, cross={:.3}",
            neg_sim,
            cross_sim2
        );
    }

    #[test]
    fn test_moral_semantic_entity_sensitivity() {
        let encoder = MoralSemanticEncoder::new(42);

        // Same action, different entities
        let with_child = encoder.encode("I helped a child cross the street");
        let with_stranger = encoder.encode("I helped a stranger cross the street");
        let with_friend = encoder.encode("I helped my friend cross the street");

        // All should be positive (helping), but entity type affects encoding
        // Child (vulnerable) should have distinct encoding from neutral entities
        let child_stranger_sim = with_child.similarity(&with_stranger);
        let friend_stranger_sim = with_friend.similarity(&with_stranger);

        // These should be reasonably similar (same action) but not identical
        assert!(
            child_stranger_sim > 0.5,
            "Same action should have baseline similarity"
        );
        assert!(
            child_stranger_sim < 0.99,
            "Different entities should differ"
        );
    }

    #[test]
    fn test_create_ethics_encoder() {
        let encoder = create_ethics_encoder();
        assert_eq!(encoder.name(), "MoralSemantic");

        let hv = encoder.encode("I helped someone in need");
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }
}
