//! Real-Valued Hypervectors for Continuous Relationships
//!
//! This module implements real-valued hyperdimensional vectors as an alternative
//! to binary hypervectors for encoding continuous relationships and topology.
//!
//! # Why Real-Valued?
//!
//! Binary hypervectors (HV16) have a fundamental limitation: operations like
//! XOR (bind) and majority vote (bundle) tend to create uniform similarity
//! around 0.5, making it difficult to encode fine-grained distinctions needed
//! for topology and Φ measurement.
//!
//! Real-valued hypervectors solve this by:
//! - Element-wise multiplication (bind) preserves magnitude
//! - Averaging (bundle) preserves structure
//! - Cosine similarity captures gradients
//!
//! # Mathematical Properties
//!
//! For vectors A, B with small noise ε:
//! - `bind(A, 1+ε) ≈ A` (magnitude preservation)
//! - `similarity(A, A+ε) > 0.9` (gradient preservation)
//! - `bundle([A, B, C])` preserves contribution of each
//!
//! This enables encoding of continuous relationships like graph topology
//! where edge weights and connectivity patterns matter.

use serde::{Deserialize, Serialize};

/// Real-valued hypervector (2048 dimensions)
///
/// Each dimension is a floating-point value in the range [-1, 1].
/// This allows for fine-grained similarity distinctions needed for
/// encoding continuous relationships like graph topology.
///
/// # Examples
///
/// ```
/// use symthaea::hdc::real_hv::RealHV;
///
/// let a = RealHV::random(2048, 42);
/// let b = RealHV::random(2048, 43);
///
/// // Binding (element-wise multiplication)
/// let c = a.bind(&b);
///
/// // Similarity (cosine similarity)
/// let sim = a.similarity(&b);  // ~0.0 for random vectors
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RealHV {
    pub values: Vec<f32>,
}

impl RealHV {
    /// Default dimension for hypervectors
    pub const DEFAULT_DIM: usize = 2048;

    /// Create a zero vector (all dimensions = 0)
    pub fn zero(dim: usize) -> Self {
        Self {
            values: vec![0.0; dim],
        }
    }

    /// Create a ones vector (all dimensions = 1)
    pub fn ones(dim: usize) -> Self {
        Self {
            values: vec![1.0; dim],
        }
    }

    /// Create a random hypervector with values in [-1, 1]
    ///
    /// Uses deterministic random generation based on seed for reproducibility.
    ///
    /// # Example
    /// ```
    /// # use symthaea::hdc::real_hv::RealHV;
    /// let v1 = RealHV::random(2048, 42);
    /// let v2 = RealHV::random(2048, 42);
    /// assert_eq!(v1.values, v2.values);  // Same seed = same vector
    /// ```
    pub fn random(dim: usize, seed: u64) -> Self {
        use blake3::Hasher;

        let mut hasher = Hasher::new();
        hasher.update(&seed.to_le_bytes());

        // Generate bytes and convert to f32 values
        let mut bytes = vec![0u8; dim * 4];  // 4 bytes per f32
        let mut xof = hasher.finalize_xof();
        xof.fill(&mut bytes);

        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| {
                // Convert 4 bytes to u32, then to normalized f32 in [-1, 1]
                let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let normalized = (bits as f64) / (u32::MAX as f64);  // 0 to 1
                ((normalized * 2.0) - 1.0) as f32  // -1 to 1
            })
            .collect();

        Self { values }
    }

    /// Create basis vector for a specific index
    ///
    /// Basis vectors are unique, deterministic vectors for each index.
    /// Used in graph encoding to represent nodes uniquely.
    pub fn basis(index: usize, dim: usize) -> Self {
        Self::random(dim, 2000000 + index as u64)
    }

    /// Bind two vectors via element-wise multiplication
    ///
    /// Binding creates associations: "cat" ⊗ "orange" = "orange cat"
    ///
    /// Properties:
    /// - Commutative: A ⊗ B = B ⊗ A
    /// - Associative: (A ⊗ B) ⊗ C = A ⊗ (B ⊗ C)
    /// - Approximate inverse: A ⊗ B ⊗ (1/B) ≈ A
    /// - Magnitude preservation: ||A ⊗ (1+ε)|| ≈ ||A||
    ///
    /// # Example
    /// ```
    /// # use symthaea::hdc::real_hv::RealHV;
    /// let cat = RealHV::random(2048, 1);
    /// let orange = RealHV::random(2048, 2);
    /// let orange_cat = cat.bind(&orange);
    ///
    /// // Unbinding approximately recovers original
    /// let recovered = orange_cat.bind(&orange.inverse());
    /// assert!(recovered.similarity(&cat) > 0.7);
    /// ```
    pub fn bind(&self, other: &Self) -> Self {
        assert_eq!(self.values.len(), other.values.len(),
                   "Cannot bind vectors of different dimensions");

        let values: Vec<f32> = self.values
            .iter()
            .zip(&other.values)
            .map(|(a, b)| a * b)
            .collect();

        Self { values }
    }

    /// Element-wise addition of two vectors
    ///
    /// Addition creates linear combinations: A + B
    ///
    /// Properties:
    /// - Commutative: A + B = B + A
    /// - Associative: (A + B) + C = A + (B + C)
    /// - Used for noise application: A ⊗ (1 + ε) where ε is small
    ///
    /// # Example
    /// ```
    /// # use symthaea::hdc::real_hv::RealHV;
    /// let a = RealHV::random(2048, 1);
    /// let b = RealHV::random(2048, 2);
    /// let sum = a.add(&b);
    /// ```
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.values.len(), other.values.len(),
                   "Cannot add vectors of different dimensions");

        let values: Vec<f32> = self.values
            .iter()
            .zip(&other.values)
            .map(|(a, b)| a + b)
            .collect();

        Self { values }
    }

    /// Bundle multiple vectors via averaging
    ///
    /// Bundling creates prototypes: bundle([cat1, cat2, cat3]) = "cat prototype"
    ///
    /// Properties:
    /// - Commutative: bundle({A, B}) = bundle({B, A})
    /// - Idempotent-ish: bundle({A, A, A}) = A
    /// - Linear: bundle({A, B, C}) = (A + B + C) / 3
    ///
    /// # Example
    /// ```
    /// # use symthaea::hdc::real_hv::RealHV;
    /// let cat1 = RealHV::random(2048, 1);
    /// let cat2 = RealHV::random(2048, 2);
    /// let cat3 = RealHV::random(2048, 3);
    ///
    /// let cat_prototype = RealHV::bundle(&[cat1.clone(), cat2.clone(), cat3.clone()]);
    ///
    /// // Prototype is similar to all inputs
    /// assert!(cat_prototype.similarity(&cat1) > 0.5);
    /// assert!(cat_prototype.similarity(&cat2) > 0.5);
    /// assert!(cat_prototype.similarity(&cat3) > 0.5);
    /// ```
    pub fn bundle(vectors: &[Self]) -> Self {
        if vectors.is_empty() {
            return Self::zero(Self::DEFAULT_DIM);
        }

        let dim = vectors[0].values.len();
        let n = vectors.len() as f32;

        let mut sum = vec![0.0f32; dim];

        for vector in vectors {
            assert_eq!(vector.values.len(), dim,
                       "All vectors must have same dimension");
            for (i, &val) in vector.values.iter().enumerate() {
                sum[i] += val;
            }
        }

        let values: Vec<f32> = sum.iter().map(|&s| s / n).collect();

        Self { values }
    }

    /// Compute cosine similarity with another vector
    ///
    /// Returns a value in [-1, 1] where:
    /// - 1.0 = identical vectors
    /// - 0.0 = orthogonal vectors
    /// - -1.0 = opposite vectors
    ///
    /// For random vectors, expected similarity ≈ 0.0 (not 0.5 like binary!)
    ///
    /// # Example
    /// ```
    /// # use symthaea::hdc::real_hv::RealHV;
    /// let a = RealHV::random(2048, 42);
    /// let b = RealHV::random(2048, 43);
    ///
    /// let sim = a.similarity(&b);
    /// assert!(sim.abs() < 0.2);  // Random vectors ≈ 0, not 0.5!
    /// ```
    pub fn similarity(&self, other: &Self) -> f32 {
        assert_eq!(self.values.len(), other.values.len(),
                   "Cannot compute similarity of vectors with different dimensions");

        let dot: f32 = self.values
            .iter()
            .zip(&other.values)
            .map(|(a, b)| a * b)
            .sum();

        let norm_self: f32 = self.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_other: f32 = other.values.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_self == 0.0 || norm_other == 0.0 {
            return 0.0;
        }

        dot / (norm_self * norm_other)
    }

    /// Compute the inverse for approximate unbinding
    ///
    /// For element-wise multiplication binding, inverse is element-wise reciprocal.
    /// Note: This is approximate and may amplify small values.
    pub fn inverse(&self) -> Self {
        let values: Vec<f32> = self.values
            .iter()
            .map(|&x| {
                if x.abs() < 1e-6 {
                    0.0  // Avoid division by very small numbers
                } else {
                    1.0 / x
                }
            })
            .collect();

        Self { values }
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: f32) -> Self {
        let values: Vec<f32> = self.values.iter().map(|&x| x * scalar).collect();
        Self { values }
    }

    /// Normalize to unit length (L2 norm = 1)
    pub fn normalize(&self) -> Self {
        let norm: f32 = self.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            return self.clone();
        }
        self.scale(1.0 / norm)
    }

    /// Get dimension of the vector
    pub fn dim(&self) -> usize {
        self.values.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_hv_bind_preserves_similarity_gradient() {
        println!("\n🔬 TESTING REAL-VALUED HYPERVECTORS: The Critical Test!");
        println!("{}", "=".repeat(80));
        println!("\n📖 HYPOTHESIS: Real-valued HDVs preserve similarity gradients");
        println!("   Unlike binary HDVs (which gave uniform ~0.5 similarity),");
        println!("   real-valued HDVs should maintain fine-grained distinctions.");
        println!();

        let dim = 2048;
        let a = RealHV::random(dim, 42);

        // Create vectors with different amounts of noise
        let noise_0_1 = RealHV::random(dim, 100).scale(0.1);  // 10% noise
        let noise_0_3 = RealHV::random(dim, 101).scale(0.3);  // 30% noise
        let noise_0_5 = RealHV::random(dim, 102).scale(0.5);  // 50% noise
        let noise_1_0 = RealHV::random(dim, 103);             // 100% noise (full random)

        // Bind with (1 + noise): a * (1 + ε) ≈ a for small ε
        let ones = RealHV::ones(dim);
        let a_with_0_1 = a.bind(&ones.add(&noise_0_1));
        let a_with_0_3 = a.bind(&ones.add(&noise_0_3));
        let a_with_0_5 = a.bind(&ones.add(&noise_0_5));
        let a_with_1_0 = a.bind(&noise_1_0);

        println!("✅ Created test vectors with varying noise levels");
        println!();

        // Measure similarities
        let sim_0_1 = a.similarity(&a_with_0_1);
        let sim_0_3 = a.similarity(&a_with_0_3);
        let sim_0_5 = a.similarity(&a_with_0_5);
        let sim_1_0 = a.similarity(&a_with_1_0);

        println!("📊 Similarity Measurements:");
        println!("   A ↔ A*noise_0.1: {:.6}", sim_0_1);
        println!("   A ↔ A*noise_0.3: {:.6}", sim_0_3);
        println!("   A ↔ A*noise_0.5: {:.6}", sim_0_5);
        println!("   A ↔ A*noise_1.0: {:.6}", sim_1_0);
        println!();

        // Check for gradient
        let has_gradient = sim_0_1 > sim_0_3 && sim_0_3 > sim_0_5 && sim_0_5 > sim_1_0;

        println!("📈 Gradient Analysis:");
        println!("   Decreasing similarity with noise? {}", if has_gradient { "✅ YES" } else { "❌ NO" });
        println!();

        // Compare to binary HDV behavior (which would give uniform ~0.5)
        if sim_0_1 > 0.7 && sim_1_0.abs() < 0.2 {
            println!("🎉 SUCCESS: Real-valued HDVs create heterogeneous similarity!");
            println!("   - Small noise: {:.2} (vs binary ~0.5)", sim_0_1);
            println!("   - Large noise: {:.2} (vs binary ~0.5)", sim_1_0);
            println!("   - CLEAR GRADIENT preserved! ✅");
            println!();
            println!("💡 IMPLICATION: Real-valued HDVs CAN encode fine-grained topology!");
            println!("   This is exactly what we need for Φ measurement!");
        } else {
            println!("⚠️  WARNING: Similarity pattern unexpected");
            println!("   May need adjustment to binding/noise method");
        }

        println!();
        println!("{}", "=".repeat(80));

        // Assert the key property: gradient preservation
        assert!(sim_0_1 > 0.7,
                "Small noise should preserve high similarity, got {:.4}", sim_0_1);
        assert!(sim_1_0.abs() < 0.3,
                "Large noise should create low similarity, got {:.4}", sim_1_0);
        assert!(has_gradient,
                "Similarity should decrease with increasing noise");
    }

    #[test]
    fn test_real_hv_random_vectors_near_orthogonal() {
        // Random real-valued vectors should be approximately orthogonal (sim ≈ 0)
        // NOT similar at 0.5 like binary vectors!
        let a = RealHV::random(2048, 1);
        let b = RealHV::random(2048, 2);

        let sim = a.similarity(&b);

        println!("\n📊 Random Vector Similarity:");
        println!("   A ↔ B: {:.6}", sim);
        println!("   Expected: ≈ 0.0 (NOT 0.5 like binary!)");

        assert!(sim.abs() < 0.15,
                "Random real-valued vectors should be nearly orthogonal, got {:.4}", sim);
    }

    #[test]
    fn test_real_hv_bundle_preserves_components() {
        let a = RealHV::random(2048, 1);
        let b = RealHV::random(2048, 2);
        let c = RealHV::random(2048, 3);

        let bundled = RealHV::bundle(&[a.clone(), b.clone(), c.clone()]);

        let sim_a = bundled.similarity(&a);
        let sim_b = bundled.similarity(&b);
        let sim_c = bundled.similarity(&c);

        println!("\n📊 Bundle Similarity:");
        println!("   Bundle ↔ A: {:.6}", sim_a);
        println!("   Bundle ↔ B: {:.6}", sim_b);
        println!("   Bundle ↔ C: {:.6}", sim_c);

        // All should be moderately similar (>0.4) since they're part of the bundle
        assert!(sim_a > 0.4, "Bundle should be similar to component A");
        assert!(sim_b > 0.4, "Bundle should be similar to component B");
        assert!(sim_c > 0.4, "Bundle should be similar to component C");
    }
}
