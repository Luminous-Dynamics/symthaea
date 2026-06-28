// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
HDC-Native Similarity Search

Consciousness-native O(1) similarity using XOR + popcount for bipolar vectors.
This replaces generic HNSW/cosine approaches with HDC-specific operations
that leverage the unique properties of hyperdimensional computing.

## Performance Characteristics

- **Similarity**: O(D/64) via SIMD XOR + popcount (effectively O(1) for fixed D)
- **Bundled Query**: O(D) superposition + threshold
- **Memory**: 1 bit per dimension (16KB for 128K dimensions)

## Key Insight

In HDC, vectors are quasi-orthogonal by design. This means:
1. Random vectors have ~50% Hamming similarity
2. Similar concepts have >70% Hamming similarity
3. XOR of similar vectors has low popcount (few 1s)

This allows direct bitwise operations instead of floating-point math.
*/

use std::collections::HashMap;
use std::sync::RwLock;

/// Packed bipolar vector using u64 for SIMD-friendly popcount
///
/// Stores bipolar {-1, +1} as binary {0, 1} in packed u64 words.
/// Each u64 holds 64 dimensions, enabling hardware popcount.
#[derive(Clone, Debug, Default)]
pub struct PackedBipolar {
    /// Packed bits: 0 = -1, 1 = +1
    words: Vec<u64>,
    /// Original dimension count
    dimension: usize,
}

impl PackedBipolar {
    /// Create from bipolar i8 vector
    pub fn from_bipolar(bipolar: &[i8]) -> Self {
        let dimension = bipolar.len();
        let num_words = dimension.div_ceil(64);
        let mut words = vec![0u64; num_words];

        for (i, &val) in bipolar.iter().enumerate() {
            if val > 0 {
                let word_idx = i / 64;
                let bit_idx = i % 64;
                words[word_idx] |= 1u64 << bit_idx;
            }
        }

        Self { words, dimension }
    }

    /// Create from continuous values (thresholding at 0)
    ///
    /// Values > 0 become +1 (bit=1), values <= 0 become -1 (bit=0)
    pub fn from_continuous(values: &[f32]) -> Self {
        let dimension = values.len();
        let num_words = dimension.div_ceil(64);
        let mut words = vec![0u64; num_words];

        for (i, &val) in values.iter().enumerate() {
            if val > 0.0 {
                let word_idx = i / 64;
                let bit_idx = i % 64;
                words[word_idx] |= 1u64 << bit_idx;
            }
        }

        Self { words, dimension }
    }

    /// Create from boolean vector
    pub fn from_binary(binary: &[bool]) -> Self {
        let dimension = binary.len();
        let num_words = dimension.div_ceil(64);
        let mut words = vec![0u64; num_words];

        for (i, &val) in binary.iter().enumerate() {
            if val {
                let word_idx = i / 64;
                let bit_idx = i % 64;
                words[word_idx] |= 1u64 << bit_idx;
            }
        }

        Self { words, dimension }
    }

    /// Convert back to bipolar i8 vector
    pub fn to_bipolar(&self) -> Vec<i8> {
        let mut result = vec![-1i8; self.dimension];

        for (word_idx, &word) in self.words.iter().enumerate() {
            for bit_idx in 0..64 {
                let i = word_idx * 64 + bit_idx;
                if i >= self.dimension {
                    break;
                }
                if (word >> bit_idx) & 1 == 1 {
                    result[i] = 1;
                }
            }
        }

        result
    }

    /// XOR similarity: 1.0 - (hamming_distance / dimension)
    ///
    /// This is the core O(1) operation for HDC similarity.
    /// Uses hardware popcount for counting differing bits.
    #[inline]
    pub fn xor_similarity(&self, other: &PackedBipolar) -> f32 {
        assert_eq!(
            self.dimension, other.dimension,
            "xor_similarity: dimension mismatch {} vs {}",
            self.dimension, other.dimension
        );

        let mut diff_bits: u32 = 0;

        // SIMD-friendly: XOR words then popcount
        for (a, b) in self.words.iter().zip(other.words.iter()) {
            diff_bits += (a ^ b).count_ones();
        }

        // Similarity = 1 - (differing / total)
        1.0 - (diff_bits as f32 / self.dimension as f32)
    }

    /// Hamming distance (number of differing bits)
    #[inline]
    pub fn hamming_distance(&self, other: &PackedBipolar) -> u32 {
        self.words
            .iter()
            .zip(other.words.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }

    /// Bundle (superposition) of multiple packed vectors
    ///
    /// Uses majority voting: for each bit position, count 1s vs 0s
    pub fn bundle(vectors: &[&PackedBipolar]) -> Option<PackedBipolar> {
        if vectors.is_empty() {
            return None;
        }

        let dimension = vectors[0].dimension;
        let num_words = vectors[0].words.len();
        let threshold = vectors.len() / 2;

        let mut result_words = vec![0u64; num_words];

        // For each bit position, count votes
        for bit_idx in 0..dimension {
            let word_idx = bit_idx / 64;
            let bit_pos = bit_idx % 64;

            let ones: usize = vectors
                .iter()
                .filter(|v| (v.words[word_idx] >> bit_pos) & 1 == 1)
                .count();

            if ones > threshold {
                result_words[word_idx] |= 1u64 << bit_pos;
            }
        }

        Some(PackedBipolar {
            words: result_words,
            dimension,
        })
    }

    /// Bind (element-wise XOR) of two packed vectors
    ///
    /// In binary representation, XOR is equivalent to multiplication
    /// of bipolar vectors: (+1)*(-1) = -1 = 0, (+1)*(+1) = +1 = 1
    #[inline]
    pub fn bind(&self, other: &PackedBipolar) -> PackedBipolar {
        assert_eq!(
            self.dimension, other.dimension,
            "bind: dimension mismatch {} vs {}",
            self.dimension, other.dimension
        );

        let words: Vec<u64> = self
            .words
            .iter()
            .zip(other.words.iter())
            .map(|(a, b)| a ^ b)
            .collect();

        PackedBipolar {
            words,
            dimension: self.dimension,
        }
    }

    /// Permute (circular shift) for sequence encoding
    pub fn permute(&self, shift: usize) -> PackedBipolar {
        let bipolar = self.to_bipolar();
        let mut shifted = vec![-1i8; self.dimension];

        for i in 0..self.dimension {
            let new_idx = (i + shift) % self.dimension;
            shifted[new_idx] = bipolar[i];
        }

        PackedBipolar::from_bipolar(&shifted)
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.words.len() * 8
    }
}

/// Native HDC similarity index
///
/// Provides O(1) similarity search using XOR + popcount.
/// Unlike HNSW which approximates nearest neighbors,
/// this performs exact similarity computation efficiently.
pub struct NativeSimilarityIndex {
    /// Stored patterns with their IDs
    patterns: RwLock<HashMap<String, PackedBipolar>>,
    /// Dimension of all vectors
    dimension: usize,
    /// Similarity threshold for matching
    threshold: f32,
}

impl NativeSimilarityIndex {
    /// Create new index with specified dimension
    pub fn new(dimension: usize) -> Self {
        Self {
            patterns: RwLock::new(HashMap::new()),
            dimension,
            threshold: 0.7, // Default: 70% similarity for match
        }
    }

    /// Set similarity threshold
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Store a pattern with ID
    pub fn store(&self, id: &str, pattern: &[i8]) -> Result<(), String> {
        if pattern.len() != self.dimension {
            return Err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimension,
                pattern.len()
            ));
        }

        let packed = PackedBipolar::from_bipolar(pattern);
        let mut patterns = self.patterns.write().expect("patterns RwLock poisoned");
        patterns.insert(id.to_string(), packed);

        Ok(())
    }

    /// Store packed pattern directly
    pub fn store_packed(&self, id: &str, pattern: PackedBipolar) -> Result<(), String> {
        if pattern.dimension() != self.dimension {
            return Err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimension,
                pattern.dimension()
            ));
        }

        let mut patterns = self.patterns.write().expect("patterns RwLock poisoned");
        patterns.insert(id.to_string(), pattern);

        Ok(())
    }

    /// Query for similar patterns (O(n) scan with O(1) similarity per item)
    ///
    /// Returns (id, similarity) pairs above threshold, sorted by similarity
    pub fn query(&self, pattern: &[i8], limit: usize) -> Vec<(String, f32)> {
        let query_packed = PackedBipolar::from_bipolar(pattern);
        self.query_packed(&query_packed, limit)
    }

    /// Query with packed pattern
    pub fn query_packed(&self, query: &PackedBipolar, limit: usize) -> Vec<(String, f32)> {
        let patterns = self.patterns.read().expect("patterns RwLock poisoned");

        let mut matches: Vec<(String, f32)> = patterns
            .iter()
            .map(|(id, stored)| {
                let sim = query.xor_similarity(stored);
                (id.clone(), sim)
            })
            .filter(|(_, sim)| *sim >= self.threshold)
            .collect();

        // Sort by similarity descending
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(limit);

        matches
    }

    /// Find exact match (similarity > 0.99)
    pub fn find_exact(&self, pattern: &[i8]) -> Option<String> {
        let query_packed = PackedBipolar::from_bipolar(pattern);
        let patterns = self.patterns.read().expect("patterns RwLock poisoned");

        for (id, stored) in patterns.iter() {
            if query_packed.xor_similarity(stored) > 0.99 {
                return Some(id.clone());
            }
        }

        None
    }

    /// Retrieve pattern by ID
    pub fn get(&self, id: &str) -> Option<Vec<i8>> {
        let patterns = self.patterns.read().expect("patterns RwLock poisoned");
        patterns.get(id).map(|p| p.to_bipolar())
    }

    /// Get packed pattern by ID
    pub fn get_packed(&self, id: &str) -> Option<PackedBipolar> {
        let patterns = self.patterns.read().expect("patterns RwLock poisoned");
        patterns.get(id).cloned()
    }

    /// Remove pattern by ID
    pub fn remove(&self, id: &str) -> bool {
        let mut patterns = self.patterns.write().expect("patterns RwLock poisoned");
        patterns.remove(id).is_some()
    }

    /// Number of stored patterns
    pub fn len(&self) -> usize {
        self.patterns
            .read()
            .expect("patterns RwLock poisoned")
            .len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        let patterns = self.patterns.read().expect("patterns RwLock poisoned");
        patterns.values().map(|p| p.memory_bytes()).sum::<usize>()
            + patterns.keys().map(|k| k.len()).sum::<usize>()
    }
}

/// Bundled query for multi-concept search
///
/// Superimposes multiple query patterns and searches for
/// patterns similar to the bundle. Useful for "OR" queries.
pub struct BundledQuery {
    patterns: Vec<PackedBipolar>,
    dimension: usize,
}

impl BundledQuery {
    pub fn new(dimension: usize) -> Self {
        Self {
            patterns: Vec::new(),
            dimension,
        }
    }

    /// Add a pattern to the bundle
    pub fn add(&mut self, pattern: &[i8]) -> Result<(), String> {
        if pattern.len() != self.dimension {
            return Err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimension,
                pattern.len()
            ));
        }

        self.patterns.push(PackedBipolar::from_bipolar(pattern));
        Ok(())
    }

    /// Create bundled query pattern
    pub fn bundle(&self) -> Option<PackedBipolar> {
        let refs: Vec<&PackedBipolar> = self.patterns.iter().collect();
        PackedBipolar::bundle(&refs)
    }

    /// Execute query against index
    pub fn execute(&self, index: &NativeSimilarityIndex, limit: usize) -> Vec<(String, f32)> {
        match self.bundle() {
            Some(bundled) => index.query_packed(&bundled, limit),
            None => Vec::new(),
        }
    }
}

/// Sequence query using permutation binding
///
/// Encodes ordered sequences like "A then B then C" using
/// HDC's permutation binding mechanism.
pub struct SequenceQuery {
    elements: Vec<PackedBipolar>,
    dimension: usize,
}

impl SequenceQuery {
    pub fn new(dimension: usize) -> Self {
        Self {
            elements: Vec::new(),
            dimension,
        }
    }

    /// Add next element in sequence
    pub fn push(&mut self, pattern: &[i8]) -> Result<(), String> {
        if pattern.len() != self.dimension {
            return Err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimension,
                pattern.len()
            ));
        }

        self.elements.push(PackedBipolar::from_bipolar(pattern));
        Ok(())
    }

    /// Build sequence encoding: Σ_i permute(element_i, i)
    pub fn encode(&self) -> Option<PackedBipolar> {
        if self.elements.is_empty() {
            return None;
        }

        let permuted: Vec<PackedBipolar> = self
            .elements
            .iter()
            .enumerate()
            .map(|(i, elem)| elem.permute(i))
            .collect();

        let refs: Vec<&PackedBipolar> = permuted.iter().collect();
        PackedBipolar::bundle(&refs)
    }

    /// Execute sequence query against index
    pub fn execute(&self, index: &NativeSimilarityIndex, limit: usize) -> Vec<(String, f32)> {
        match self.encode() {
            Some(encoded) => index.query_packed(&encoded, limit),
            None => Vec::new(),
        }
    }
}

/// Statistics for monitoring index performance
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub pattern_count: usize,
    pub dimension: usize,
    pub memory_bytes: usize,
    pub bits_per_dimension: f32,
    pub avg_query_ops: u64, // XOR + popcount ops per query
}

impl NativeSimilarityIndex {
    /// Get index statistics
    pub fn stats(&self) -> IndexStats {
        let pattern_count = self.len();
        let memory = self.memory_bytes();

        IndexStats {
            pattern_count,
            dimension: self.dimension,
            memory_bytes: memory,
            bits_per_dimension: if self.dimension > 0 {
                (memory * 8) as f32 / (pattern_count.max(1) * self.dimension) as f32
            } else {
                0.0
            },
            avg_query_ops: (self.dimension as u64 / 64) * pattern_count as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_bipolar_roundtrip() {
        let original = vec![1i8, -1, 1, 1, -1, -1, 1, -1];
        let packed = PackedBipolar::from_bipolar(&original);
        let recovered = packed.to_bipolar();

        assert_eq!(original, recovered);
    }

    #[test]
    fn test_xor_similarity_identical() {
        let vec = vec![1i8; 64];
        let packed = PackedBipolar::from_bipolar(&vec);

        let sim = packed.xor_similarity(&packed);
        assert!(
            (sim - 1.0).abs() < 0.001,
            "Identical vectors should have similarity 1.0"
        );
    }

    #[test]
    fn test_xor_similarity_opposite() {
        let vec_a = vec![1i8; 64];
        let vec_b = vec![-1i8; 64];

        let packed_a = PackedBipolar::from_bipolar(&vec_a);
        let packed_b = PackedBipolar::from_bipolar(&vec_b);

        let sim = packed_a.xor_similarity(&packed_b);
        assert!(
            (sim - 0.0).abs() < 0.001,
            "Opposite vectors should have similarity 0.0"
        );
    }

    #[test]
    fn test_xor_similarity_half() {
        // Create vectors that differ in exactly half the positions
        let vec_a = vec![1i8; 64];
        let mut vec_b = vec![1i8; 64];
        for i in 0..32 {
            vec_b[i] = -1;
        }

        let packed_a = PackedBipolar::from_bipolar(&vec_a);
        let packed_b = PackedBipolar::from_bipolar(&vec_b);

        let sim = packed_a.xor_similarity(&packed_b);
        assert!(
            (sim - 0.5).abs() < 0.001,
            "Half-different vectors should have similarity 0.5, got {}",
            sim
        );
    }

    #[test]
    fn test_bind_xor() {
        let vec_a = vec![1i8, -1, 1, -1];
        let vec_b = vec![1i8, 1, -1, -1];

        let packed_a = PackedBipolar::from_bipolar(&vec_a);
        let packed_b = PackedBipolar::from_bipolar(&vec_b);

        let bound = packed_a.bind(&packed_b);
        let result = bound.to_bipolar();

        // XOR: (1,1)->1, (-1,1)->-1, (1,-1)->-1, (-1,-1)->1
        // But in our binary encoding: 1->1, -1->0
        // So: (1^1)=0->-1, (0^1)=1->1, (1^0)=1->1, (0^0)=0->-1
        assert_eq!(
            result,
            vec![-1, 1, 1, -1],
            "Bind should XOR binary representations"
        );
    }

    #[test]
    fn test_bundle_majority() {
        let vec_a = PackedBipolar::from_bipolar(&[1, 1, 1, -1]);
        let vec_b = PackedBipolar::from_bipolar(&[1, -1, 1, -1]);
        let vec_c = PackedBipolar::from_bipolar(&[-1, 1, 1, 1]);

        let bundled = PackedBipolar::bundle(&[&vec_a, &vec_b, &vec_c]).unwrap();
        let result = bundled.to_bipolar();

        // Majority: (1,1,-1)->1, (1,-1,1)->1, (1,1,1)->1, (-1,-1,1)->-1
        assert_eq!(result, vec![1, 1, 1, -1]);
    }

    #[test]
    fn test_native_index_store_query() {
        let index = NativeSimilarityIndex::new(64).with_threshold(0.6);

        let pattern_a = vec![1i8; 64];
        let pattern_b = vec![-1i8; 64];

        index.store("pattern_a", &pattern_a).unwrap();
        index.store("pattern_b", &pattern_b).unwrap();

        // Query with pattern_a should match pattern_a
        let results = index.query(&pattern_a, 10);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "pattern_a");
        assert!((results[0].1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_permute_sequence() {
        let vec = vec![1i8, -1, 1, -1, 1, -1, 1, -1];
        let packed = PackedBipolar::from_bipolar(&vec);

        let permuted = packed.permute(2);
        let result = permuted.to_bipolar();

        // Circular shift right by 2
        assert_eq!(
            result,
            vec![1, -1, 1, -1, 1, -1, 1, -1],
            "Permute should circular shift"
        );
    }

    #[test]
    fn test_sequence_query() {
        let dim = 64;
        let index = NativeSimilarityIndex::new(dim).with_threshold(0.5);

        // Store a known sequence encoding
        let elem_a = vec![1i8; dim];
        let elem_b: Vec<i8> = (0..dim).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();

        let mut seq = SequenceQuery::new(dim);
        seq.push(&elem_a).unwrap();
        seq.push(&elem_b).unwrap();

        let encoded = seq.encode().unwrap();
        index.store_packed("seq_ab", encoded.clone()).unwrap();

        // Query should find the stored sequence
        let results = seq.execute(&index, 5);
        assert!(!results.is_empty());
        assert!(
            results[0].1 > 0.8,
            "Should have high similarity to exact match"
        );
    }

    #[test]
    fn test_memory_efficiency() {
        let dim = 16384; // Standard HDC dimension
        let vec = vec![1i8; dim];
        let packed = PackedBipolar::from_bipolar(&vec);

        // Should use dim/8 bytes (2KB for 16K dimensions)
        let expected_bytes = (dim + 63) / 64 * 8;
        assert_eq!(packed.memory_bytes(), expected_bytes);
        assert_eq!(expected_bytes, 2048, "16K dimensions should use 2KB");
    }

    #[test]
    fn test_text_bridge_to_bipolar_to_memory() {
        use crate::hdc::BinaryHV;
        use crate::hdc::semantic_bridge::SemanticBridge;
        use crate::hdc::unified_hv::ContinuousHV;

        // 1. Text concept -> SemanticBridge -> BinaryHV
        let bridge = SemanticBridge::with_vocabulary(&["sensory", "cortex", "active", "inference"]);
        let concept_binary = bridge.encode_phrase("sensory cortex active inference");
        assert_ne!(concept_binary, BinaryHV::zero());

        // 2. Convert BinaryHV -> ContinuousHV (representing NeuralBridgeV2 ContinuousHV projection)
        let mut continuous_values = vec![0.0f32; BinaryHV::DIM];
        for i in 0..BinaryHV::DIM {
            let bit = (concept_binary.0[i / 8] >> (i % 8)) & 1;
            continuous_values[i] = if bit == 1 { 1.0f32 } else { -1.0f32 };
        }
        let continuous_hv = ContinuousHV::from_vec(continuous_values);

        // 3. Convert ContinuousHV -> PackedBipolar (standardized boundary)
        let packed = PackedBipolar::from_continuous(&continuous_hv.values);
        assert_eq!(packed.dimension, BinaryHV::DIM);

        // 4. Store in NativeSimilarityIndex memory query
        let index = NativeSimilarityIndex::new(BinaryHV::DIM).with_threshold(0.5);
        index
            .store_packed("cortex_activation", packed.clone())
            .unwrap();

        // 5. Retrieve via query
        let results = index.query_packed(&packed, 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "cortex_activation");
        assert!((results[0].1 - 1.0).abs() < 1e-4);
    }
}
