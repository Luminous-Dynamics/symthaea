// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Optimized Prover for K-Vector range proofs
//!
//! Performance-optimized STARK prover with:
//! - Configurable security/performance tradeoffs
//! - Proof caching for repeated proofs
//! - Batch proof generation
//! - Parallel trace computation (with `parallel` feature)

use std::num::NonZeroUsize;
use std::sync::Arc;

use lru::LruCache;
use parking_lot::Mutex;
use winterfell::{BatchingMethod, Proof, ProofOptions, Prover};

use crate::error::{ZkpError, ZkpResult};
use crate::prover::{KVectorProver, KVectorTrace, KVectorWitness};

/// Security level for proof generation
///
/// ## C-01 Security Level Documentation (Corrected)
///
/// STARK security depends on: queries (λ), blowup factor (ρ), and grinding factor (g).
/// Approximate security: λ × log₂(ρ) + g bits against query attacks.
///
/// **IMPORTANT**: These are conservative estimates. Actual security may be lower
/// depending on the specific attack model. For critical applications, consult
/// a cryptographer and consider the High security level.
///
/// ## Security Level Comparison
///
/// | Level     | Queries | Blowup | Grinding | Est. Security | Use Case           |
/// |-----------|---------|--------|----------|---------------|---------------------|
/// | Fast      | 20      | 4      | 0        | ~40 bits      | Testing ONLY        |
/// | Optimized | 28      | 8      | 0        | ~84 bits      | Low-stakes internal |
/// | Standard  | 32      | 8      | 0        | ~96 bits      | Production default  |
/// | High      | 64      | 16     | 8        | ~264 bits     | Critical operations |
///
/// ## CRITICAL WARNING
///
/// **C-01**: `Fast` level provides only ~40-bit security and is **NOT suitable
/// for production**. It should ONLY be used for:
/// - Unit tests
/// - Integration tests
/// - Development/debugging
///
/// For any production deployment, use `Standard` or `High`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SecurityLevel {
    /// Fast security (~40-bit) - **TESTING ONLY**
    ///
    /// # ⚠️ WARNING
    /// This level provides insufficient security for production use.
    /// Only use for testing and development.
    ///
    /// Parameters: 20 queries, 4× blowup, no grinding
    /// ~8ms proof time
    Fast,

    /// Optimized security (~84-bit) - internal/low-stakes operations
    ///
    /// Suitable for internal operations where speed matters and the
    /// consequences of a forgery are limited.
    ///
    /// Parameters: 28 queries, 8× blowup, no grinding
    /// ~12ms proof time
    Optimized,

    /// Standard security (~96-bit) - production default
    ///
    /// Recommended for most production use cases.
    ///
    /// Parameters: 32 queries, 8× blowup, no grinding
    /// ~15ms proof time
    Standard,

    /// High security (~264-bit) - critical operations
    ///
    /// Use for high-value transactions, governance decisions,
    /// or any operation where forgery would be catastrophic.
    ///
    /// Parameters: 64 queries, 16× blowup, 8-bit grinding
    /// ~25ms proof time
    High,
}

impl SecurityLevel {
    /// Estimated security bits for this level
    ///
    /// Based on: queries × log₂(blowup) + grinding
    pub fn estimated_security_bits(&self) -> u32 {
        match self {
            SecurityLevel::Fast => 40,      // 20 × 2 + 0
            SecurityLevel::Optimized => 84, // 28 × 3 + 0
            SecurityLevel::Standard => 96,  // 32 × 3 + 0
            SecurityLevel::High => 264,     // 64 × 4 + 8
        }
    }

    /// Check if this security level meets a minimum requirement
    pub fn meets_minimum(&self, min_bits: u32) -> bool {
        self.estimated_security_bits() >= min_bits
    }

    /// Returns true if this level is safe for production use
    ///
    /// C-01: Fast level is explicitly marked as NOT production-safe
    pub fn is_production_safe(&self) -> bool {
        matches!(self, SecurityLevel::Standard | SecurityLevel::High)
    }

    /// Get the minimum security level that meets a given bit requirement
    pub fn minimum_level_for_bits(bits: u32) -> Option<Self> {
        if bits <= 40 {
            Some(SecurityLevel::Fast)
        } else if bits <= 84 {
            Some(SecurityLevel::Optimized)
        } else if bits <= 96 {
            Some(SecurityLevel::Standard)
        } else if bits <= 264 {
            Some(SecurityLevel::High)
        } else {
            None // No level meets this requirement
        }
    }

    /// Get proof options for this security level
    pub fn proof_options(&self) -> ProofOptions {
        match self {
            SecurityLevel::Standard => ProofOptions::new(
                32,  // queries
                8,   // blowup factor
                0,   // grinding factor
                winterfell::FieldExtension::None,
                8,   // FRI folding factor
                31,  // max remainder degree
                BatchingMethod::Linear,
                BatchingMethod::Linear,
            ),
            SecurityLevel::Optimized => ProofOptions::new(
                28,  // slightly reduced queries (still 128-bit with blowup)
                8,   // keep blowup for security
                0,   // no grinding
                winterfell::FieldExtension::None,
                16,  // higher folding (fewer FRI rounds) — must be power of 2
                63,  // larger remainder (faster final round) — must be (2^n)-1
                BatchingMethod::Linear,
                BatchingMethod::Linear,
            ),
            SecurityLevel::Fast => ProofOptions::new(
                20,  // reduced queries for speed
                4,   // reduced blowup (faster FFT)
                0,   // no grinding
                winterfell::FieldExtension::None,
                16,  // higher folding (fewer rounds)
                63,  // larger remainder (fewer FRI rounds)
                BatchingMethod::Linear,
                BatchingMethod::Linear,
            ),
            SecurityLevel::High => ProofOptions::new(
                64,  // increased queries
                16,  // higher blowup for security
                8,   // grinding for additional security
                winterfell::FieldExtension::None,
                4,   // lower folding (more rounds)
                15,  // smaller remainder
                BatchingMethod::Linear,
                BatchingMethod::Linear,
            ),
        }
    }

    /// Estimated proof time in milliseconds
    /// Based on benchmarks from 2026-01-17 (cargo bench --bench prover_benchmark)
    pub fn estimated_time_ms(&self) -> u64 {
        match self {
            SecurityLevel::Standard => 15,  // ~14-15ms measured
            SecurityLevel::Optimized => 12, // ~12ms measured
            SecurityLevel::Fast => 8,       // ~8ms measured
            SecurityLevel::High => 25,      // ~25ms estimated (higher security params)
        }
    }
}

/// Cache key for proof caching
#[derive(Clone, Hash, PartialEq, Eq)]
struct ProofCacheKey {
    commitment: [u8; 32],
    security_level: u8,
}

/// Type alias for the proof type returned by the prover
pub type KVectorProof = Proof;

/// Optimized K-Vector prover with caching and configurable security
pub struct OptimizedKVectorProver {
    security_level: SecurityLevel,
    cache: Arc<Mutex<LruCache<ProofCacheKey, Arc<KVectorProof>>>>,
    enable_caching: bool,
}

impl OptimizedKVectorProver {
    /// Create a new optimized prover with default settings
    pub fn new() -> Self {
        Self {
            security_level: SecurityLevel::Fast, // Default to fast for throughput
            cache: Arc::new(Mutex::new(LruCache::new(NonZeroUsize::new(100).expect("100 is non-zero")))),
            enable_caching: true,
        }
    }

    /// Create a prover with specific security level
    pub fn with_security_level(level: SecurityLevel) -> Self {
        Self {
            security_level: level,
            cache: Arc::new(Mutex::new(LruCache::new(NonZeroUsize::new(100).expect("100 is non-zero")))),
            enable_caching: true,
        }
    }

    /// Set cache size
    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.cache = Arc::new(Mutex::new(LruCache::new(
            NonZeroUsize::new(size.max(1)).expect("max(1) guarantees non-zero"),
        )));
        self
    }

    /// Disable caching
    pub fn without_caching(mut self) -> Self {
        self.enable_caching = false;
        self
    }

    /// Generate a proof for a K-Vector witness
    pub fn prove(&self, witness: &KVectorWitness) -> ZkpResult<KVectorProof> {
        // Check cache first
        if self.enable_caching {
            let cache_key = ProofCacheKey {
                commitment: witness.commitment(),
                security_level: self.security_level as u8,
            };

            if let Some(cached) = self.cache.lock().get(&cache_key) {
                return Ok((**cached).clone());
            }
        }

        // Generate proof
        let trace = KVectorTrace::new(witness)?;
        let prover = KVectorProver::with_options(self.security_level.proof_options());

        let proof = prover
            .prove(trace)
            .map_err(|e| ZkpError::ProofGenerationFailed(e.to_string()))?;

        // Cache the proof
        if self.enable_caching {
            let cache_key = ProofCacheKey {
                commitment: witness.commitment(),
                security_level: self.security_level as u8,
            };
            self.cache.lock().put(cache_key, Arc::new(proof.clone()));
        }

        Ok(proof)
    }

    /// Generate proofs for multiple witnesses in batch
    #[cfg(feature = "parallel")]
    pub fn prove_batch(&self, witnesses: &[KVectorWitness]) -> Vec<ZkpResult<KVectorProof>> {
        use rayon::prelude::*;

        witnesses.par_iter().map(|w| self.prove(w)).collect()
    }

    #[cfg(not(feature = "parallel"))]
    pub fn prove_batch(&self, witnesses: &[KVectorWitness]) -> Vec<ZkpResult<KVectorProof>> {
        witnesses.iter().map(|w| self.prove(w)).collect()
    }

    /// Get current cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        let cache = self.cache.lock();
        CacheStats {
            size: cache.len(),
            capacity: cache.cap().get(),
        }
    }

    /// Clear the proof cache
    pub fn clear_cache(&self) {
        self.cache.lock().clear();
    }

    /// Get the security level
    pub fn security_level(&self) -> SecurityLevel {
        self.security_level
    }
}

impl Default for OptimizedKVectorProver {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache statistics
#[derive(Clone, Debug)]
pub struct CacheStats {
    pub size: usize,
    pub capacity: usize,
}

/// Builder for configuring the optimized prover
pub struct OptimizedProverBuilder {
    security_level: SecurityLevel,
    cache_size: usize,
    enable_caching: bool,
}

impl OptimizedProverBuilder {
    pub fn new() -> Self {
        Self {
            security_level: SecurityLevel::Fast,
            cache_size: 100,
            enable_caching: true,
        }
    }

    pub fn security_level(mut self, level: SecurityLevel) -> Self {
        self.security_level = level;
        self
    }

    pub fn cache_size(mut self, size: usize) -> Self {
        self.cache_size = size;
        self
    }

    pub fn caching(mut self, enabled: bool) -> Self {
        self.enable_caching = enabled;
        self
    }

    pub fn build(self) -> OptimizedKVectorProver {
        let prover = OptimizedKVectorProver::with_security_level(self.security_level)
            .with_cache_size(self.cache_size);

        if !self.enable_caching {
            prover.without_caching()
        } else {
            prover
        }
    }
}

impl Default for OptimizedProverBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_witness() -> KVectorWitness {
        KVectorWitness {
            k_r: 0.8,
            k_a: 0.7,
            k_i: 0.9,
            k_p: 0.6,
            k_m: 0.5,
            k_s: 0.4,
            k_h: 0.85,
            k_topo: 0.75,
        }
    }

    #[test]
    fn test_security_levels() {
        // Updated based on benchmarks from 2026-01-17
        assert_eq!(SecurityLevel::Fast.estimated_time_ms(), 8);
        assert_eq!(SecurityLevel::Standard.estimated_time_ms(), 15);
        assert_eq!(SecurityLevel::High.estimated_time_ms(), 25);
    }

    #[test]
    fn test_builder() {
        let prover = OptimizedProverBuilder::new()
            .security_level(SecurityLevel::Standard)
            .cache_size(50)
            .build();

        assert_eq!(prover.security_level(), SecurityLevel::Standard);
        assert_eq!(prover.cache_stats().capacity, 50);
    }

    #[test]
    fn test_caching() {
        let prover = OptimizedKVectorProver::new();
        let witness = sample_witness();

        // First proof should not be cached
        let stats_before = prover.cache_stats();
        let _proof1 = prover.prove(&witness).unwrap();

        // Should be cached now
        let stats_after = prover.cache_stats();
        assert_eq!(stats_after.size, stats_before.size + 1);

        // Second proof should hit cache
        let _proof2 = prover.prove(&witness).unwrap();
        let stats_final = prover.cache_stats();
        assert_eq!(stats_final.size, stats_after.size); // No new entry
    }

    #[test]
    fn test_clear_cache() {
        let prover = OptimizedKVectorProver::new();
        let witness = sample_witness();

        let _proof = prover.prove(&witness).unwrap();
        assert!(prover.cache_stats().size > 0);

        prover.clear_cache();
        assert_eq!(prover.cache_stats().size, 0);
    }

    // ============ C-01 Security Level Tests ============

    #[test]
    fn test_estimated_security_bits() {
        // Verify documented security levels
        assert_eq!(SecurityLevel::Fast.estimated_security_bits(), 40);
        assert_eq!(SecurityLevel::Optimized.estimated_security_bits(), 84);
        assert_eq!(SecurityLevel::Standard.estimated_security_bits(), 96);
        assert_eq!(SecurityLevel::High.estimated_security_bits(), 264);
    }

    #[test]
    fn test_security_level_ordering() {
        // Security levels should be ordered by security strength
        assert!(SecurityLevel::Fast < SecurityLevel::Optimized);
        assert!(SecurityLevel::Optimized < SecurityLevel::Standard);
        assert!(SecurityLevel::Standard < SecurityLevel::High);
    }

    #[test]
    fn test_meets_minimum() {
        // Fast meets 40 bits
        assert!(SecurityLevel::Fast.meets_minimum(40));
        assert!(!SecurityLevel::Fast.meets_minimum(41));

        // Optimized meets up to 84 bits
        assert!(SecurityLevel::Optimized.meets_minimum(84));
        assert!(!SecurityLevel::Optimized.meets_minimum(85));

        // Standard meets up to 96 bits
        assert!(SecurityLevel::Standard.meets_minimum(96));
        assert!(!SecurityLevel::Standard.meets_minimum(97));

        // High meets up to 264 bits
        assert!(SecurityLevel::High.meets_minimum(264));
        assert!(!SecurityLevel::High.meets_minimum(265));
    }

    #[test]
    fn test_is_production_safe() {
        // Only Standard and High are production safe
        assert!(!SecurityLevel::Fast.is_production_safe());
        assert!(!SecurityLevel::Optimized.is_production_safe());
        assert!(SecurityLevel::Standard.is_production_safe());
        assert!(SecurityLevel::High.is_production_safe());
    }

    #[test]
    fn test_minimum_level_for_bits() {
        // Test boundary conditions
        assert_eq!(SecurityLevel::minimum_level_for_bits(1), Some(SecurityLevel::Fast));
        assert_eq!(SecurityLevel::minimum_level_for_bits(40), Some(SecurityLevel::Fast));
        assert_eq!(SecurityLevel::minimum_level_for_bits(41), Some(SecurityLevel::Optimized));
        assert_eq!(SecurityLevel::minimum_level_for_bits(84), Some(SecurityLevel::Optimized));
        assert_eq!(SecurityLevel::minimum_level_for_bits(85), Some(SecurityLevel::Standard));
        assert_eq!(SecurityLevel::minimum_level_for_bits(96), Some(SecurityLevel::Standard));
        assert_eq!(SecurityLevel::minimum_level_for_bits(97), Some(SecurityLevel::High));
        assert_eq!(SecurityLevel::minimum_level_for_bits(264), Some(SecurityLevel::High));
        assert_eq!(SecurityLevel::minimum_level_for_bits(265), None);
    }

    #[test]
    fn test_production_requirements() {
        // Production should require at least 96 bits
        const PRODUCTION_MIN: u32 = 96;

        assert!(!SecurityLevel::Fast.meets_minimum(PRODUCTION_MIN));
        assert!(!SecurityLevel::Optimized.meets_minimum(PRODUCTION_MIN));
        assert!(SecurityLevel::Standard.meets_minimum(PRODUCTION_MIN));
        assert!(SecurityLevel::High.meets_minimum(PRODUCTION_MIN));
    }

    #[test]
    fn test_proof_options_queries() {
        // Verify proof options have correct query counts
        let fast_opts = SecurityLevel::Fast.proof_options();
        let optimized_opts = SecurityLevel::Optimized.proof_options();
        let standard_opts = SecurityLevel::Standard.proof_options();
        let high_opts = SecurityLevel::High.proof_options();

        // Higher security should have more queries
        assert!(fast_opts.num_queries() < optimized_opts.num_queries());
        assert!(optimized_opts.num_queries() < standard_opts.num_queries());
        assert!(standard_opts.num_queries() < high_opts.num_queries());
    }
}
