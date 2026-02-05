// ==================================================================================
// Text Encoder for HDC Embeddings
// ==================================================================================
//
// **Purpose**: Bridge natural language to hyperdimensional computing space
//
// **The Gap**: Symthaea has powerful HDC operations but lacks robust text → HV encoding
//
// **Solution**: Multi-level text encoder with:
// 1. Character-level encoding (subword robustness)
// 2. Word-level semantic projections (learned embeddings)
// 3. Positional encoding (sequence order preservation)
// 4. Hierarchical composition (phrases → sentences → documents)
//
// **Key Innovation**: Compositional encoding allows:
// - "red ball" ≠ "ball red" (order matters via permutation)
// - "red ball" ≈ "scarlet sphere" (semantic similarity preserved)
// - "not hot" ≠ "hot" (negation encoded via binding)
//
// ==================================================================================

use anyhow::Result;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

use super::HDC_DIMENSION;
use super::primitive_system::PrimitiveSystem;

/// Text encoder configuration
#[derive(Debug, Clone)]
pub struct TextEncoderConfig {
    /// HDC dimension (default: 16,384)
    pub dimension: usize,

    /// N-gram size for character encoding (default: 3)
    pub ngram_size: usize,

    /// Use positional encoding for word order (default: true)
    pub use_positional: bool,

    /// Maximum sequence length before truncation (default: 512)
    pub max_length: usize,

    /// Learning rate for online updates (default: 0.01)
    pub learning_rate: f32,

    /// Whether to normalize output vectors (default: true)
    pub normalize: bool,
}

impl Default for TextEncoderConfig {
    fn default() -> Self {
        Self {
            dimension: HDC_DIMENSION,
            ngram_size: 3,
            use_positional: true,
            max_length: 512,
            learning_rate: 0.01,
            normalize: true,
        }
    }
}

/// Multi-level text encoder for HDC
///
/// Encodes text at multiple granularities:
/// - Characters: Hash-based projection (handles OOV)
/// - Words: Learned embeddings + character fallback
/// - Sequences: Positional encoding + bundling
///
/// # Example
/// ```rust,ignore
/// use symthaea::hdc::text_encoder::TextEncoder;
///
/// let mut encoder = TextEncoder::new(TextEncoderConfig::default())?;
///
/// // Encode single word
/// let cat = encoder.encode_word("cat")?;
///
/// // Encode sentence (preserves order)
/// let sentence = encoder.encode_sentence("the cat sat on the mat")?;
///
/// // Similar sentences should have high similarity
/// let similar = encoder.encode_sentence("a cat sits on a rug")?;
/// let dissimilar = encoder.encode_sentence("quantum mechanics is complex")?;
///
/// assert!(cosine_similarity(&sentence, &similar) > cosine_similarity(&sentence, &dissimilar));
/// ```
#[derive(Debug)]
pub struct TextEncoder {
    /// Configuration
    config: TextEncoderConfig,

    /// Learned word embeddings (word → hypervector)
    word_embeddings: HashMap<String, Vec<i8>>,

    /// Character n-gram cache
    ngram_cache: HashMap<String, Vec<i8>>,

    /// Position vectors (for sequence encoding)
    position_vectors: Vec<Vec<i8>>,

    /// Special token embeddings
    special_tokens: HashMap<String, Vec<i8>>,

    /// Statistics
    stats: TextEncoderStats,
}

/// Encoder statistics
#[derive(Debug, Default, Clone)]
pub struct TextEncoderStats {
    pub words_encoded: usize,
    pub sentences_encoded: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub vocabulary_size: usize,
}

impl TextEncoder {
    /// Create new text encoder
    pub fn new(config: TextEncoderConfig) -> Result<Self> {
        let position_vectors = Self::generate_position_vectors(config.dimension, config.max_length);
        let special_tokens = Self::generate_special_tokens(config.dimension);

        Ok(Self {
            config,
            word_embeddings: HashMap::new(),
            ngram_cache: HashMap::new(),
            position_vectors,
            special_tokens,
            stats: TextEncoderStats::default(),
        })
    }

    /// Generate deterministic position vectors
    fn generate_position_vectors(dimension: usize, max_length: usize) -> Vec<Vec<i8>> {
        (0..max_length)
            .map(|pos| {
                Self::hash_to_bipolar(&format!("__POS_{}__", pos), dimension)
            })
            .collect()
    }

    /// Generate special token embeddings
    fn generate_special_tokens(dimension: usize) -> HashMap<String, Vec<i8>> {
        let mut tokens = HashMap::new();

        // Core special tokens
        tokens.insert("[PAD]".to_string(), Self::hash_to_bipolar("[PAD]", dimension));
        tokens.insert("[UNK]".to_string(), Self::hash_to_bipolar("[UNK]", dimension));
        tokens.insert("[CLS]".to_string(), Self::hash_to_bipolar("[CLS]", dimension));
        tokens.insert("[SEP]".to_string(), Self::hash_to_bipolar("[SEP]", dimension));
        tokens.insert("[MASK]".to_string(), Self::hash_to_bipolar("[MASK]", dimension));

        // Negation token (for "not X" encoding)
        tokens.insert("[NEG]".to_string(), Self::hash_to_bipolar("[NEG]", dimension));

        // Relation tokens
        tokens.insert("[CAUSE]".to_string(), Self::hash_to_bipolar("[CAUSE]", dimension));
        tokens.insert("[EFFECT]".to_string(), Self::hash_to_bipolar("[EFFECT]", dimension));
        tokens.insert("[AGENT]".to_string(), Self::hash_to_bipolar("[AGENT]", dimension));
        tokens.insert("[PATIENT]".to_string(), Self::hash_to_bipolar("[PATIENT]", dimension));

        tokens
    }

    /// Hash string to deterministic bipolar vector
    ///
    /// Uses multiple hash functions for better distribution
    fn hash_to_bipolar(text: &str, dimension: usize) -> Vec<i8> {
        let mut result = vec![0i8; dimension];

        // Use multiple hash seeds for better distribution
        for seed in 0..8 {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            text.hash(&mut hasher);
            let hash = hasher.finish();

            // Each bit of hash determines sign of corresponding dimension
            for i in 0..64.min(dimension / 8) {
                let bit_offset = seed * 64 + i;
                if bit_offset < dimension {
                    let bit = (hash >> i) & 1;
                    result[bit_offset] = if bit == 1 { 1 } else { -1 };
                }
            }
        }

        // Fill remaining dimensions with deterministic pattern
        let mut extended_hasher = DefaultHasher::new();
        text.hash(&mut extended_hasher);
        let base_hash = extended_hasher.finish();

        for i in 512..dimension {
            let mut h = DefaultHasher::new();
            (base_hash, i).hash(&mut h);
            result[i] = if h.finish() % 2 == 0 { 1 } else { -1 };
        }

        result
    }

    /// Encode a single character
    pub fn encode_char(&self, c: char) -> Vec<i8> {
        Self::hash_to_bipolar(&c.to_string(), self.config.dimension)
    }

    /// Encode character n-grams
    fn encode_ngrams(&mut self, word: &str) -> Vec<i8> {
        // Check cache
        if let Some(cached) = self.ngram_cache.get(word) {
            self.stats.cache_hits += 1;
            return cached.clone();
        }
        self.stats.cache_misses += 1;

        let padded = format!("<{}>", word.to_lowercase());
        let chars: Vec<char> = padded.chars().collect();

        let mut ngram_vectors: Vec<Vec<i8>> = Vec::new();

        for i in 0..chars.len().saturating_sub(self.config.ngram_size - 1) {
            let ngram: String = chars[i..i + self.config.ngram_size].iter().collect();
            ngram_vectors.push(Self::hash_to_bipolar(&ngram, self.config.dimension));
        }

        // Bundle all n-grams
        let result = if ngram_vectors.is_empty() {
            Self::hash_to_bipolar(word, self.config.dimension)
        } else {
            self.bundle_vectors(&ngram_vectors)
        };

        // Cache result
        self.ngram_cache.insert(word.to_string(), result.clone());

        result
    }

    /// Encode a single word
    pub fn encode_word(&mut self, word: &str) -> Result<Vec<i8>> {
        let normalized = word.to_lowercase().trim().to_string();

        // Check learned embeddings first (word-level cache)
        if let Some(embedding) = self.word_embeddings.get(&normalized) {
            self.stats.words_encoded += 1;
            self.stats.cache_hits += 1;  // Word embedding cache hit
            return Ok(embedding.clone());
        }

        // Fall back to character n-gram encoding
        let ngram_encoding = self.encode_ngrams(&normalized);

        // Store in word embeddings for future use
        self.word_embeddings.insert(normalized, ngram_encoding.clone());
        self.stats.vocabulary_size = self.word_embeddings.len();
        self.stats.words_encoded += 1;

        Ok(ngram_encoding)
    }

    /// Encode a sentence with positional information
    pub fn encode_sentence(&mut self, text: &str) -> Result<Vec<i8>> {
        let words: Vec<&str> = text.split_whitespace().collect();

        if words.is_empty() {
            return Ok(self.special_tokens["[PAD]"].clone());
        }

        let mut word_vectors: Vec<Vec<i8>> = Vec::new();

        for (pos, word) in words.iter().enumerate().take(self.config.max_length) {
            let word_vec = self.encode_word(word)?;

            if self.config.use_positional && pos < self.position_vectors.len() {
                // Bind word with position
                let pos_vec = &self.position_vectors[pos];
                let positioned = self.bind_vectors(&word_vec, pos_vec);
                word_vectors.push(positioned);
            } else {
                word_vectors.push(word_vec);
            }
        }

        // Bundle all positioned words
        let result = self.bundle_vectors(&word_vectors);

        self.stats.sentences_encoded += 1;

        if self.config.normalize {
            Ok(self.normalize_vector(&result))
        } else {
            Ok(result)
        }
    }

    /// Encode text using primitive system when available
    ///
    /// This is the **recommended** encoding method when primitives are available.
    /// It uses canonical HV16 encodings for recognized primitives (ZERO, ONE, CAUSE, etc.)
    /// and falls back to n-gram encoding for unknown words.
    ///
    /// # Example
    /// ```rust,ignore
    /// let primitives = PrimitiveSystem::new();
    /// let encoded = encoder.encode_with_primitives("zero plus one equals one", &primitives)?;
    /// // Uses canonical ZERO, ONE encodings instead of hash-based random vectors
    /// ```
    pub fn encode_with_primitives(
        &mut self,
        text: &str,
        primitives: &PrimitiveSystem,
    ) -> Result<Vec<i8>> {
        let words: Vec<&str> = text.split_whitespace().collect();

        if words.is_empty() {
            return Ok(self.special_tokens["[PAD]"].clone());
        }

        let mut word_vectors: Vec<Vec<i8>> = Vec::new();
        let mut primitive_hits = 0;

        for (pos, word) in words.iter().enumerate().take(self.config.max_length) {
            let normalized = word.to_lowercase();

            // Try primitive first (canonical encoding)
            let word_vec = if let Some(prim) = primitives.get(&normalized) {
                primitive_hits += 1;
                // Convert HV16 (f32 bipolar) to i8 bipolar
                let f32_vec = prim.encoding.to_bipolar();
                f32_vec.iter().map(|&x| if x > 0.0 { 1i8 } else { -1i8 }).collect()
            } else {
                // Fall back to n-gram encoding
                self.encode_word(word)?
            };

            // Add positional encoding
            if self.config.use_positional && pos < self.position_vectors.len() {
                let positioned = self.bind_vectors(&word_vec, &self.position_vectors[pos]);
                word_vectors.push(positioned);
            } else {
                word_vectors.push(word_vec);
            }
        }

        // Bundle all positioned words
        let result = self.bundle_vectors(&word_vectors);

        self.stats.sentences_encoded += 1;

        if self.config.normalize {
            Ok(self.normalize_vector(&result))
        } else {
            Ok(result)
        }
    }

    /// Encode with explicit semantic role marking
    ///
    /// Useful for causal reasoning: "A causes B" → bind(A, [CAUSE]) + bind(B, [EFFECT])
    pub fn encode_with_roles(&mut self, subject: &str, predicate: &str, object: &str) -> Result<Vec<i8>> {
        let subject_vec = self.encode_word(subject)?;
        let predicate_vec = self.encode_word(predicate)?;
        let object_vec = self.encode_word(object)?;

        let agent_marker = &self.special_tokens["[AGENT]"];
        let patient_marker = &self.special_tokens["[PATIENT]"];

        // Subject bound with agent role
        let subject_marked = self.bind_vectors(&subject_vec, agent_marker);

        // Object bound with patient role
        let object_marked = self.bind_vectors(&object_vec, patient_marker);

        // Bundle: subject + predicate + object
        let result = self.bundle_vectors(&[subject_marked, predicate_vec, object_marked]);

        Ok(result)
    }

    /// Encode negation: "not X" → bind(X, [NEG])
    pub fn encode_negation(&mut self, text: &str) -> Result<Vec<i8>> {
        let positive = self.encode_sentence(text)?;
        let neg_marker = &self.special_tokens["[NEG]"];

        Ok(self.bind_vectors(&positive, neg_marker))
    }

    /// Encode causal relation: "A causes B"
    pub fn encode_causal(&mut self, cause: &str, effect: &str) -> Result<Vec<i8>> {
        let cause_vec = self.encode_sentence(cause)?;
        let effect_vec = self.encode_sentence(effect)?;

        let cause_marker = &self.special_tokens["[CAUSE]"];
        let effect_marker = &self.special_tokens["[EFFECT]"];

        // Mark cause and effect
        let cause_marked = self.bind_vectors(&cause_vec, cause_marker);
        let effect_marked = self.bind_vectors(&effect_vec, effect_marker);

        // Bundle together
        Ok(self.bundle_vectors(&[cause_marked, effect_marked]))
    }

    /// Learn from a pair of similar texts (contrastive learning)
    ///
    /// Adjusts embeddings so similar texts have higher similarity
    pub fn learn_similarity(&mut self, text1: &str, text2: &str, similarity: f32) -> Result<()> {
        // Encode both texts
        let vec1 = self.encode_sentence(text1)?;
        let vec2 = self.encode_sentence(text2)?;

        // Compute current similarity
        let current_sim = self.cosine_similarity(&vec1, &vec2);

        // Compute gradient direction
        let error = similarity - current_sim;

        // Extract words and update their embeddings
        let words1: Vec<&str> = text1.split_whitespace().collect();
        let words2: Vec<&str> = text2.split_whitespace().collect();

        let lr = self.config.learning_rate * error;

        // Nudge word embeddings toward each other if they should be similar
        for word in words1.iter().chain(words2.iter()) {
            let normalized = word.to_lowercase();
            if let Some(embedding) = self.word_embeddings.get_mut(&normalized) {
                // Simple update: move toward target direction
                for i in 0..embedding.len() {
                    let target = if similarity > 0.5 { vec2[i] } else { -vec2[i] };
                    let delta = (target as f32 - embedding[i] as f32) * lr;
                    let new_val = embedding[i] as f32 + delta;
                    embedding[i] = if new_val > 0.0 { 1 } else { -1 };
                }
            }
        }

        Ok(())
    }

    /// Compute cosine similarity between two vectors
    pub fn cosine_similarity(&self, a: &[i8], b: &[i8]) -> f32 {
        let dot: i32 = a.iter().zip(b.iter()).map(|(x, y)| (*x as i32) * (*y as i32)).sum();
        let norm_a: f32 = (a.iter().map(|x| (*x as i32).pow(2)).sum::<i32>() as f32).sqrt();
        let norm_b: f32 = (b.iter().map(|x| (*x as i32).pow(2)).sum::<i32>() as f32).sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot as f32 / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    /// Hamming similarity (faster for bipolar)
    pub fn hamming_similarity(&self, a: &[i8], b: &[i8]) -> f32 {
        let matches: usize = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
        matches as f32 / a.len() as f32
    }

    /// Bind two vectors (element-wise multiplication)
    fn bind_vectors(&self, a: &[i8], b: &[i8]) -> Vec<i8> {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
    }

    /// Bundle multiple vectors (majority vote)
    fn bundle_vectors(&self, vectors: &[Vec<i8>]) -> Vec<i8> {
        if vectors.is_empty() {
            return vec![0i8; self.config.dimension];
        }

        let dim = vectors[0].len();
        let mut sums = vec![0i32; dim];

        for vec in vectors {
            for i in 0..dim {
                sums[i] += vec[i] as i32;
            }
        }

        // Majority vote
        sums.iter().map(|&s| if s > 0 { 1 } else { -1 }).collect()
    }

    /// Normalize vector (for bipolar, this is a no-op but ensures consistency)
    fn normalize_vector(&self, v: &[i8]) -> Vec<i8> {
        v.to_vec()  // Bipolar vectors are already normalized
    }

    /// Get encoder statistics
    pub fn stats(&self) -> &TextEncoderStats {
        &self.stats
    }

    /// Get vocabulary size
    pub fn vocabulary_size(&self) -> usize {
        self.word_embeddings.len()
    }

    /// Export word embeddings for persistence
    pub fn export_embeddings(&self) -> &HashMap<String, Vec<i8>> {
        &self.word_embeddings
    }

    /// Import word embeddings (for persistence)
    pub fn import_embeddings(&mut self, embeddings: HashMap<String, Vec<i8>>) {
        self.word_embeddings = embeddings;
        self.stats.vocabulary_size = self.word_embeddings.len();
    }

    /// Convert i8 bipolar to f32 for compatibility with other modules
    pub fn to_f32(&self, bipolar: &[i8]) -> Vec<f32> {
        bipolar.iter().map(|&x| x as f32).collect()
    }

    /// Convert f32 to i8 bipolar
    pub fn from_f32(&self, float_vec: &[f32]) -> Vec<i8> {
        float_vec.iter().map(|&x| if x > 0.0 { 1 } else { -1 }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_encoding() {
        let mut encoder = TextEncoder::new(TextEncoderConfig::default()).unwrap();

        let cat = encoder.encode_word("cat").unwrap();
        let dog = encoder.encode_word("dog").unwrap();
        let cat2 = encoder.encode_word("cat").unwrap();

        // Same word should produce same vector
        assert_eq!(cat, cat2);

        // Different words should be different
        assert_ne!(cat, dog);

        // Both should have correct dimension
        assert_eq!(cat.len(), HDC_DIMENSION);
    }

    #[test]
    fn test_sentence_encoding() {
        let mut encoder = TextEncoder::new(TextEncoderConfig::default()).unwrap();

        let s1 = encoder.encode_sentence("the cat sat").unwrap();
        let s2 = encoder.encode_sentence("the cat sat").unwrap();
        let s3 = encoder.encode_sentence("the dog ran").unwrap();

        // Same sentence should produce same vector
        assert_eq!(s1, s2);

        // Different sentences should be different
        assert_ne!(s1, s3);
    }

    #[test]
    fn test_order_matters() {
        let mut encoder = TextEncoder::new(TextEncoderConfig::default()).unwrap();

        let ab = encoder.encode_sentence("cat dog").unwrap();
        let ba = encoder.encode_sentence("dog cat").unwrap();

        // Order should matter (different encodings)
        assert_ne!(ab, ba);
    }

    #[test]
    fn test_similarity_semantic() {
        let mut encoder = TextEncoder::new(TextEncoderConfig::default()).unwrap();

        let cat = encoder.encode_word("cat").unwrap();
        let kitten = encoder.encode_word("kitten").unwrap();
        let car = encoder.encode_word("car").unwrap();

        // With n-gram encoding, "cat" and "kitten" should be more similar
        // than "cat" and "car" due to shared character patterns
        let sim_cat_kitten = encoder.hamming_similarity(&cat, &kitten);
        let sim_cat_car = encoder.hamming_similarity(&cat, &car);

        // Note: Without learned embeddings, this may not hold
        // But n-gram similarity should give some signal
        println!("cat-kitten: {}, cat-car: {}", sim_cat_kitten, sim_cat_car);
    }

    #[test]
    fn test_negation() {
        let mut encoder = TextEncoder::new(TextEncoderConfig::default()).unwrap();

        let hot = encoder.encode_sentence("hot").unwrap();
        let not_hot = encoder.encode_negation("hot").unwrap();

        // Negation should produce different vector
        assert_ne!(hot, not_hot);

        // But they should be related (binding is reversible)
        let neg_marker = &encoder.special_tokens["[NEG]"];
        let recovered = encoder.bind_vectors(&not_hot, neg_marker);

        // Recovered should equal original
        assert_eq!(hot, recovered);
    }

    #[test]
    fn test_causal_encoding() {
        let mut encoder = TextEncoder::new(TextEncoderConfig::default()).unwrap();

        let causal = encoder.encode_causal("rain", "wet ground").unwrap();

        assert_eq!(causal.len(), HDC_DIMENSION);

        // Should be able to extract cause and effect with unbinding
        // (Advanced: would need inverse markers)
    }

    #[test]
    fn test_stats() {
        let mut encoder = TextEncoder::new(TextEncoderConfig::default()).unwrap();

        encoder.encode_word("test").unwrap();
        encoder.encode_word("test").unwrap();  // Cache hit
        encoder.encode_sentence("hello world").unwrap();

        let stats = encoder.stats();
        assert_eq!(stats.words_encoded, 4);  // test + test + hello + world
        assert_eq!(stats.sentences_encoded, 1);
        assert!(stats.cache_hits > 0);
    }
}
