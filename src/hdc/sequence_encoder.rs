/*!
Week 17 Critical Fix #2: Permutation-Based Ordered Binding

XOR/multiplication binding is COMMUTATIVE - order disappears!
- bind(A, B) == bind(B, A)
- "cat sat mat" == "mat sat cat" with naive binding

Solution: Use permutation to mark position BEFORE binding:
- seq = bundle([permute(0, hv(t0)), permute(1, hv(t1)), ...])
- This preserves order: "cat sat mat" ≠ "mat sat cat"

Key Algorithm:
```text
fn encode_sequence(tokens: &[T]) -> HV {
    let components: Vec<HV> = tokens.iter()
        .enumerate()
        .map(|(pos, token)| permute(encode(token), pos))
        .collect();
    bundle(components)
}

fn unbind_at_position(sequence: &HV, position: usize) -> HV {
    unpermute(sequence, position)  // Inverse permutation
}
```

Integration Points:
- Used by Hippocampus for sequential event encoding
- Used by Prefrontal for temporal reasoning
- Used by temporal_encoder for time-position binding
*/

use anyhow::Result;

/// Permute a bipolar vector by circular shift
///
/// This is the core operation for preserving order in HDC sequences.
/// permute(hv, k) shifts all elements right by k positions (circular).
///
/// # Properties
/// - permute(hv, 0) == hv (identity)
/// - unpermute(permute(hv, k), k) == hv (inverse)
/// - permute(hv, k1 + k2) == permute(permute(hv, k1), k2) (composable)
pub fn permute(hv: &[i8], k: usize) -> Vec<i8> {
    if hv.is_empty() || k == 0 {
        return hv.to_vec();
    }

    let n = hv.len();
    let k = k % n;  // Normalize shift

    let mut result = vec![0i8; n];
    for i in 0..n {
        let new_idx = (i + k) % n;
        result[new_idx] = hv[i];
    }
    result
}

/// Inverse permutation - undo a permute operation
///
/// unpermute(hv, k) shifts left by k positions (circular).
/// Equivalent to permute(hv, n - k) where n = dimension.
pub fn unpermute(hv: &[i8], k: usize) -> Vec<i8> {
    if hv.is_empty() || k == 0 {
        return hv.to_vec();
    }

    let n = hv.len();
    let k = k % n;
    let reverse_shift = n - k;

    permute(hv, reverse_shift)
}

/// Bundle multiple bipolar vectors (majority vote)
///
/// The result is a single vector where each dimension is the
/// sign of the sum of corresponding dimensions across inputs.
pub fn bundle(vectors: &[Vec<i8>]) -> Vec<i8> {
    if vectors.is_empty() {
        return Vec::new();
    }

    let dim = vectors[0].len();
    let mut sum = vec![0i32; dim];

    for vec in vectors {
        assert_eq!(vec.len(), dim, "All vectors must have same dimension");
        for i in 0..dim {
            sum[i] += vec[i] as i32;
        }
    }

    // Threshold to bipolar
    sum.into_iter()
        .map(|s| if s >= 0 { 1i8 } else { -1i8 })
        .collect()
}

/// Bind two bipolar vectors (element-wise multiplication)
///
/// # Properties
/// - bind(A, B) == bind(B, A) (COMMUTATIVE - why we need permute!)
/// - bind(A, bind(A, B)) == B (self-inverse for unbinding)
/// - bind(A, A) == all 1s
pub fn bind(a: &[i8], b: &[i8]) -> Vec<i8> {
    assert_eq!(a.len(), b.len(), "Vectors must have same dimension");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Sequence encoder using permutation-based position encoding
pub struct SequenceEncoder {
    /// Dimensionality of hypervectors
    dimensions: usize,

    /// Base vectors for atoms (if using character/byte encoding)
    atom_vectors: Option<Vec<Vec<i8>>>,
}

impl SequenceEncoder {
    /// Create new sequence encoder
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            atom_vectors: None,
        }
    }

    /// Initialize with random atom vectors for 256 bytes
    pub fn with_byte_atoms(dimensions: usize) -> Self {
        let atom_vectors: Vec<Vec<i8>> = (0..256)
            .map(|_| {
                (0..dimensions)
                    .map(|_| if rand::random::<bool>() { 1 } else { -1 })
                    .collect()
            })
            .collect();

        Self {
            dimensions,
            atom_vectors: Some(atom_vectors),
        }
    }

    /// Encode a sequence of bipolar vectors with position preservation
    ///
    /// Each vector is permuted by its position before bundling:
    /// result = bundle([permute(v0, 0), permute(v1, 1), ...])
    ///
    /// # Why This Works
    /// - permute(v, k) is unique for each position k
    /// - bundling combines them into single vector
    /// - unpermute(result, k) approximately recovers v_k
    pub fn encode_vectors(&self, vectors: &[Vec<i8>]) -> Vec<i8> {
        if vectors.is_empty() {
            return vec![0i8; self.dimensions];
        }

        let permuted: Vec<Vec<i8>> = vectors.iter()
            .enumerate()
            .map(|(pos, vec)| permute(vec, pos))
            .collect();

        bundle(&permuted)
    }

    /// Encode a byte sequence (e.g., UTF-8 string)
    ///
    /// Requires initialization with with_byte_atoms()
    pub fn encode_bytes(&self, bytes: &[u8]) -> Result<Vec<i8>> {
        let atoms = self.atom_vectors.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Byte encoding requires atom vectors - use with_byte_atoms()"))?;

        let vectors: Vec<Vec<i8>> = bytes.iter()
            .map(|&b| atoms[b as usize].clone())
            .collect();

        Ok(self.encode_vectors(&vectors))
    }

    /// Encode a string as sequence (each character as position)
    pub fn encode_string(&self, s: &str) -> Result<Vec<i8>> {
        self.encode_bytes(s.as_bytes())
    }

    /// Probe what's at a specific position in an encoded sequence
    ///
    /// Returns the "residual" at position k by unpermuting.
    /// Compare with known item vectors to identify what was stored there.
    pub fn probe_position(&self, sequence: &[i8], position: usize) -> Vec<i8> {
        unpermute(sequence, position)
    }

    /// Encode role-filler bindings (semantic frame encoding)
    ///
    /// Given a set of (role, filler) pairs, creates a holographic frame:
    /// frame = bundle([bind(role1, filler1), bind(role2, filler2), ...])
    ///
    /// # Example
    /// ```ignore
    /// let agent = encoder.random_hv();
    /// let action = encoder.random_hv();
    /// let patient = encoder.random_hv();
    ///
    /// let cat = encoder.random_hv();
    /// let chased = encoder.random_hv();
    /// let mouse = encoder.random_hv();
    ///
    /// // "The cat chased the mouse"
    /// let frame = encoder.encode_role_fillers(&[
    ///     (&agent, &cat),      // agent = cat
    ///     (&action, &chased),  // action = chased
    ///     (&patient, &mouse),  // patient = mouse
    /// ]);
    ///
    /// // Query: "What is the agent?"
    /// let query_agent = encoder.unbind(&frame, &agent);
    /// // query_agent ≈ cat
    /// ```
    pub fn encode_role_fillers(&self, bindings: &[(&[i8], &[i8])]) -> Vec<i8> {
        let bound: Vec<Vec<i8>> = bindings.iter()
            .map(|(role, filler)| bind(role, filler))
            .collect();

        bundle(&bound)
    }

    /// Unbind a query from a frame to retrieve filler
    ///
    /// Given frame = bundle([bind(r1, f1), ...]) and query r1,
    /// returns f1 (approximately).
    pub fn unbind(&self, frame: &[i8], query: &[i8]) -> Vec<i8> {
        bind(frame, query)  // Binding is self-inverse!
    }

    /// Generate random hypervector
    pub fn random_hv(&self) -> Vec<i8> {
        (0..self.dimensions)
            .map(|_| if rand::random::<bool>() { 1 } else { -1 })
            .collect()
    }

    /// Hamming similarity between two bipolar vectors
    pub fn similarity(&self, a: &[i8], b: &[i8]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let matches: usize = a.iter()
            .zip(b.iter())
            .filter(|(x, y)| x == y)
            .count();

        matches as f32 / a.len() as f32
    }

    /// Get dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}

impl Default for SequenceEncoder {
    fn default() -> Self {
        Self::new(2048)
    }
}

// ============================================================================
// TESTS - Comprehensive validation of permutation-based sequence encoding
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: Generate random bipolar vector
    fn random_bipolar(dim: usize) -> Vec<i8> {
        (0..dim)
            .map(|_| if rand::random::<bool>() { 1 } else { -1 })
            .collect()
    }

    /// Helper: Hamming similarity
    fn similarity(a: &[i8], b: &[i8]) -> f32 {
        let matches: usize = a.iter()
            .zip(b.iter())
            .filter(|(x, y)| x == y)
            .count();
        matches as f32 / a.len() as f32
    }

    #[test]
    fn test_permute_identity() {
        let hv = random_bipolar(2048);
        let permuted = permute(&hv, 0);
        assert_eq!(hv, permuted, "Permute by 0 should be identity");
    }

    #[test]
    fn test_permute_unpermute_inverse() {
        let hv = random_bipolar(2048);

        for k in [1, 5, 100, 1000, 2047] {
            let permuted = permute(&hv, k);
            let restored = unpermute(&permuted, k);
            assert_eq!(hv, restored, "Unpermute should be inverse of permute for k={}", k);
        }
    }

    #[test]
    fn test_permute_creates_orthogonal() {
        let hv = random_bipolar(2048);

        // Different permutations should be nearly orthogonal
        let p0 = permute(&hv, 0);
        let p1 = permute(&hv, 1);
        let p100 = permute(&hv, 100);

        let sim_0_1 = similarity(&p0, &p1);
        let sim_0_100 = similarity(&p0, &p100);

        assert!(sim_0_1 < 0.55, "permute(hv, 0) vs permute(hv, 1) should be ~orthogonal, got {}", sim_0_1);
        assert!(sim_0_100 < 0.55, "permute(hv, 0) vs permute(hv, 100) should be ~orthogonal, got {}", sim_0_100);
    }

    #[test]
    fn test_bind_commutative() {
        let a = random_bipolar(2048);
        let b = random_bipolar(2048);

        let ab = bind(&a, &b);
        let ba = bind(&b, &a);

        assert_eq!(ab, ba, "Binding should be commutative (this is WHY we need permute!)");
    }

    #[test]
    fn test_bind_self_inverse() {
        let a = random_bipolar(2048);
        let b = random_bipolar(2048);

        let bound = bind(&a, &b);
        let recovered_b = bind(&bound, &a);

        assert_eq!(b, recovered_b, "Binding with same vector should recover original");
    }

    #[test]
    fn test_sequence_order_matters() {
        let encoder = SequenceEncoder::new(2048);

        // Create three distinct vectors
        let cat = encoder.random_hv();
        let sat = encoder.random_hv();
        let mat = encoder.random_hv();

        // Encode "cat sat mat" vs "mat sat cat"
        let seq1 = encoder.encode_vectors(&[cat.clone(), sat.clone(), mat.clone()]);
        let seq2 = encoder.encode_vectors(&[mat.clone(), sat.clone(), cat.clone()]);

        let sim = encoder.similarity(&seq1, &seq2);

        // These should NOT be identical (order preserved!)
        assert!(sim < 0.9, "Different orderings should produce different sequences, got similarity {}", sim);

        // But same ordering should be identical
        let seq1_again = encoder.encode_vectors(&[cat.clone(), sat.clone(), mat.clone()]);
        assert_eq!(seq1, seq1_again, "Same ordering should produce same sequence");
    }

    #[test]
    fn test_probe_position_recovery() {
        let encoder = SequenceEncoder::new(2048);

        let a = encoder.random_hv();
        let b = encoder.random_hv();
        let c = encoder.random_hv();

        let sequence = encoder.encode_vectors(&[a.clone(), b.clone(), c.clone()]);

        // Probe each position
        let probe_0 = encoder.probe_position(&sequence, 0);
        let probe_1 = encoder.probe_position(&sequence, 1);
        let probe_2 = encoder.probe_position(&sequence, 2);

        // Should be most similar to original at that position
        let sim_0_a = encoder.similarity(&probe_0, &a);
        let sim_0_b = encoder.similarity(&probe_0, &b);
        let sim_1_b = encoder.similarity(&probe_1, &b);
        let sim_2_c = encoder.similarity(&probe_2, &c);

        assert!(sim_0_a > sim_0_b, "Position 0 probe should be more similar to a than b");
        assert!(sim_1_b > 0.6, "Position 1 probe should recover b, got {}", sim_1_b);
        assert!(sim_2_c > 0.6, "Position 2 probe should recover c, got {}", sim_2_c);
    }

    #[test]
    fn test_role_filler_encoding() {
        let encoder = SequenceEncoder::new(2048);

        // Define roles
        let agent = encoder.random_hv();
        let action = encoder.random_hv();
        let patient = encoder.random_hv();

        // Define fillers
        let cat = encoder.random_hv();
        let chased = encoder.random_hv();
        let mouse = encoder.random_hv();

        // Encode "cat chased mouse"
        let frame = encoder.encode_role_fillers(&[
            (&agent, &cat),
            (&action, &chased),
            (&patient, &mouse),
        ]);

        // Query: "What is the agent?"
        let query_result = encoder.unbind(&frame, &agent);

        // Should be most similar to cat
        let sim_cat = encoder.similarity(&query_result, &cat);
        let sim_mouse = encoder.similarity(&query_result, &mouse);
        let sim_chased = encoder.similarity(&query_result, &chased);

        assert!(sim_cat > sim_mouse, "Agent query should return cat, not mouse");
        assert!(sim_cat > sim_chased, "Agent query should return cat, not chased");
        assert!(sim_cat > 0.6, "Agent query should strongly match cat, got {}", sim_cat);
    }

    #[test]
    fn test_string_encoding() {
        let encoder = SequenceEncoder::with_byte_atoms(2048);

        let hello = encoder.encode_string("hello").unwrap();
        let hello2 = encoder.encode_string("hello").unwrap();
        let world = encoder.encode_string("world").unwrap();
        let olleh = encoder.encode_string("olleh").unwrap();  // Reversed

        // Same string = identical encoding
        assert_eq!(hello, hello2, "Same string should produce same encoding");

        // Different strings = different encodings
        let sim_hello_world = encoder.similarity(&hello, &world);
        assert!(sim_hello_world < 0.7, "Different strings should be dissimilar, got {}", sim_hello_world);

        // Reversed string should NOT be the same (order matters!)
        let sim_hello_olleh = encoder.similarity(&hello, &olleh);
        assert!(sim_hello_olleh < 0.7, "Reversed string should be different, got {}", sim_hello_olleh);
    }

    #[test]
    fn test_bundle_preserves_all() {
        let encoder = SequenceEncoder::new(2048);

        let a = encoder.random_hv();
        let b = encoder.random_hv();
        let c = encoder.random_hv();

        let bundled = bundle(&[a.clone(), b.clone(), c.clone()]);

        // Bundled should be somewhat similar to all components
        let sim_a = encoder.similarity(&bundled, &a);
        let sim_b = encoder.similarity(&bundled, &b);
        let sim_c = encoder.similarity(&bundled, &c);

        assert!(sim_a > 0.55 && sim_a < 0.85, "Bundle should be moderately similar to a, got {}", sim_a);
        assert!(sim_b > 0.55 && sim_b < 0.85, "Bundle should be moderately similar to b, got {}", sim_b);
        assert!(sim_c > 0.55 && sim_c < 0.85, "Bundle should be moderately similar to c, got {}", sim_c);
    }

    #[test]
    fn test_permute_wrapping() {
        let hv = vec![1i8, -1, 1, -1, 1];

        // Permute by dimension should wrap to identity
        let permuted = permute(&hv, 5);
        assert_eq!(hv, permuted, "Permute by dimension should be identity");

        // Permute by dimension+1 should be same as permute by 1
        let p1 = permute(&hv, 1);
        let p6 = permute(&hv, 6);
        assert_eq!(p1, p6, "Permute should wrap around");
    }

    #[test]
    fn test_sequence_with_statistical_retriever() {
        // Integration test with statistical retrieval
        use crate::hdc::statistical_retrieval::StatisticalRetriever;

        let encoder = SequenceEncoder::new(2048);
        let retriever = StatisticalRetriever::new(2048);

        let a = encoder.random_hv();
        let b = encoder.random_hv();

        // Encode sequence
        let seq = encoder.encode_vectors(&[a.clone(), b.clone()]);

        // Probe position 0
        let probe_0 = encoder.probe_position(&seq, 0);

        // Use statistical retrieval to verify match
        let decision = retriever.decide_simple(&probe_0, &a);

        // Should be a confident match
        assert!(decision.z_score > 2.0,
                "Position probe should have significant z-score for original item, got {}", decision.z_score);
    }

    #[test]
    fn test_long_sequence_degradation() {
        let encoder = SequenceEncoder::new(2048);

        // Create long sequence
        let items: Vec<Vec<i8>> = (0..50)
            .map(|_| encoder.random_hv())
            .collect();

        let sequence = encoder.encode_vectors(&items);

        // Probe middle and end positions
        let probe_0 = encoder.probe_position(&sequence, 0);
        let probe_25 = encoder.probe_position(&sequence, 25);
        let probe_49 = encoder.probe_position(&sequence, 49);

        let sim_0 = encoder.similarity(&probe_0, &items[0]);
        let sim_25 = encoder.similarity(&probe_25, &items[25]);
        let sim_49 = encoder.similarity(&probe_49, &items[49]);

        // All should still be recoverable (above random 0.5)
        assert!(sim_0 > 0.52, "Position 0 in long sequence should be recoverable, got {}", sim_0);
        assert!(sim_25 > 0.52, "Position 25 in long sequence should be recoverable, got {}", sim_25);
        assert!(sim_49 > 0.52, "Position 49 in long sequence should be recoverable, got {}", sim_49);

        println!("Long sequence (50 items) recovery: pos0={:.3}, pos25={:.3}, pos49={:.3}",
                 sim_0, sim_25, sim_49);
    }
}
