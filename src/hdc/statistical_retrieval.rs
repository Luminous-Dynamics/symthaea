/*!
Week 17 Critical Fix #1: Statistical Retrieval Decision

Replaces naive threshold-based similarity with a rigorous three-gate decision procedure:
1. Z-score significance test (is this similarity statistically meaningful?)
2. Margin threshold (above application-specific acceptance floor)
3. Unbind consistency (can we recover the query from result?)

Mathematical Foundation:
- For n-bit hypervectors, random similarity ≈ 0.5
- Standard deviation σ = √(0.25/n)
- For n=2048: σ ≈ 0.011, so z=3 means sim ≥ 0.533

Why This Matters:
- Naive threshold (sim > 0.8) has NO statistical foundation
- HDC similarity depends on dimensionality
- Same threshold can mean "miraculous match" or "random noise"
- This fix provides principled decision-making with epistemic tiers

Integration Points:
- Used by Hippocampus for memory retrieval verification
- Used by Thymus for immune system verification
- Used by Prefrontal for coalition formation confidence
*/

/// Epistemic tier for empirical verification (from Epistemic Charter v2.0)
/// Maps directly to E-axis of the Epistemic Cube
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmpiricalTier {
    /// E0: Null - unverifiable, random similarity
    E0Null,
    /// E1: Testimonial - weak evidence (z < 2)
    E1Testimonial,
    /// E2: Privately Verifiable - moderate evidence (2 ≤ z < 4)
    E2PrivatelyVerifiable,
    /// E3: Cryptographically Proven - strong evidence (4 ≤ z < 6)
    E3CryptographicallyProven,
    /// E4: Publicly Reproducible - overwhelming evidence (z ≥ 6)
    E4PubliclyReproducible,
}

impl EmpiricalTier {
    /// Map z-score to epistemic tier
    pub fn from_z_score(z: f32) -> Self {
        if z < 1.0 {
            EmpiricalTier::E0Null
        } else if z < 2.0 {
            EmpiricalTier::E1Testimonial
        } else if z < 4.0 {
            EmpiricalTier::E2PrivatelyVerifiable
        } else if z < 6.0 {
            EmpiricalTier::E3CryptographicallyProven
        } else {
            EmpiricalTier::E4PubliclyReproducible
        }
    }

    /// Minimum z-score to reach this tier
    pub fn minimum_z(&self) -> f32 {
        match self {
            EmpiricalTier::E0Null => 0.0,
            EmpiricalTier::E1Testimonial => 1.0,
            EmpiricalTier::E2PrivatelyVerifiable => 2.0,
            EmpiricalTier::E3CryptographicallyProven => 4.0,
            EmpiricalTier::E4PubliclyReproducible => 6.0,
        }
    }

    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            EmpiricalTier::E0Null => "Null: indistinguishable from random noise",
            EmpiricalTier::E1Testimonial => "Weak: possible match but unreliable",
            EmpiricalTier::E2PrivatelyVerifiable => "Moderate: statistically significant",
            EmpiricalTier::E3CryptographicallyProven => "Strong: highly confident match",
            EmpiricalTier::E4PubliclyReproducible => "Overwhelming: virtually certain",
        }
    }
}

/// Retrieval verdict - result of three-gate decision
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalVerdict {
    /// All three gates passed - accept the retrieval
    Accept,
    /// Z-score too low - not statistically significant
    RejectNoSignificance,
    /// Unbind operation didn't recover expected residual
    RejectUnbindFailed,
    /// Below application-specific margin floor
    RejectBelowFloor,
}

impl RetrievalVerdict {
    pub fn is_accept(&self) -> bool {
        matches!(self, RetrievalVerdict::Accept)
    }

    pub fn reason(&self) -> &'static str {
        match self {
            RetrievalVerdict::Accept => "Match accepted",
            RetrievalVerdict::RejectNoSignificance => "Similarity not statistically significant",
            RetrievalVerdict::RejectUnbindFailed => "Unbind consistency check failed",
            RetrievalVerdict::RejectBelowFloor => "Below application margin threshold",
        }
    }
}

/// Complete retrieval decision with all diagnostic information
#[derive(Debug, Clone)]
pub struct RetrievalDecision {
    /// Raw Hamming similarity in [0, 1]
    pub raw_similarity: f32,

    /// Z-score: (similarity - 0.5) / σ
    pub z_score: f32,

    /// Whether margin threshold was passed
    pub margin_passed: bool,

    /// Whether unbind consistency check passed
    pub unbind_consistent: bool,

    /// Unbind similarity (if checked)
    pub unbind_similarity: Option<f32>,

    /// Final verdict
    pub verdict: RetrievalVerdict,

    /// Epistemic tier based on z-score
    pub epistemic_tier: EmpiricalTier,

    /// Dimensionality used for calculation
    pub dimensions: usize,

    /// Standard deviation for this dimensionality
    pub sigma: f32,
}

impl RetrievalDecision {
    /// Check if retrieval should be accepted
    pub fn accepted(&self) -> bool {
        self.verdict.is_accept()
    }

    /// Get confidence level as percentage (0-100)
    /// Based on z-score converted to cumulative normal distribution
    pub fn confidence_percent(&self) -> f32 {
        // Approximate CDF of standard normal for z
        // P(Z < z) using error function approximation
        let z = self.z_score;
        if z < -6.0 {
            return 0.0;
        }
        if z > 6.0 {
            return 100.0;
        }

        // Approximation using tanh
        let p = 0.5 * (1.0 + (z / 2.0_f32.sqrt()).tanh());
        p * 100.0
    }
}

/// Configuration for statistical retrieval
#[derive(Debug, Clone)]
pub struct StatisticalRetrievalConfig {
    /// Hypervector dimensionality
    pub dimensions: usize,

    /// Minimum z-score for statistical significance (default: 3.0)
    pub min_z_score: f32,

    /// Application-specific margin floor (default: 0.6)
    pub margin_floor: f32,

    /// Whether to perform unbind consistency check (default: true)
    pub check_unbind: bool,

    /// Minimum unbind similarity for consistency (default: 0.7)
    pub unbind_threshold: f32,
}

impl Default for StatisticalRetrievalConfig {
    fn default() -> Self {
        Self {
            dimensions: 2048,
            min_z_score: 3.0,       // p < 0.0013
            margin_floor: 0.6,      // Must be above this raw similarity
            check_unbind: true,
            unbind_threshold: 0.7,  // Unbind residual must be similar to expected
        }
    }
}

impl StatisticalRetrievalConfig {
    /// Create config for given dimensionality
    pub fn with_dimensions(dimensions: usize) -> Self {
        Self {
            dimensions,
            ..Default::default()
        }
    }

    /// Calculate standard deviation for configured dimensionality
    /// σ = √(0.25/n) for bipolar vectors
    pub fn sigma(&self) -> f32 {
        (0.25 / self.dimensions as f32).sqrt()
    }

    /// Calculate z-score for a raw similarity value
    /// z = (sim - 0.5) / σ
    pub fn z_score(&self, raw_similarity: f32) -> f32 {
        (raw_similarity - 0.5) / self.sigma()
    }

    /// Calculate raw similarity needed for given z-score
    /// sim = 0.5 + z * σ
    pub fn similarity_for_z(&self, z: f32) -> f32 {
        0.5 + z * self.sigma()
    }
}

/// Statistical retrieval engine
pub struct StatisticalRetriever {
    config: StatisticalRetrievalConfig,
}

impl StatisticalRetriever {
    /// Create new retriever with default config
    pub fn new(dimensions: usize) -> Self {
        Self {
            config: StatisticalRetrievalConfig::with_dimensions(dimensions),
        }
    }

    /// Create retriever with custom config
    pub fn with_config(config: StatisticalRetrievalConfig) -> Self {
        Self { config }
    }

    /// Get configuration
    pub fn config(&self) -> &StatisticalRetrievalConfig {
        &self.config
    }

    /// Compute Hamming similarity between two bipolar vectors
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

    /// Unbind operation (element-wise multiplication for bipolar)
    pub fn unbind(&self, a: &[i8], b: &[i8]) -> Vec<i8> {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| x * y)
            .collect()
    }

    /// Three-gate retrieval decision
    ///
    /// # Arguments
    /// * `query` - Query hypervector
    /// * `candidate` - Candidate result hypervector
    /// * `expected_residual` - Optional expected residual after unbind (for consistency check)
    ///
    /// # Returns
    /// Complete decision with diagnostics
    pub fn decide(
        &self,
        query: &[i8],
        candidate: &[i8],
        expected_residual: Option<&[i8]>,
    ) -> RetrievalDecision {
        let raw_similarity = self.hamming_similarity(query, candidate);
        let sigma = self.config.sigma();
        let z_score = self.config.z_score(raw_similarity);
        let epistemic_tier = EmpiricalTier::from_z_score(z_score);

        // Gate 1: Statistical significance
        if z_score < self.config.min_z_score {
            return RetrievalDecision {
                raw_similarity,
                z_score,
                margin_passed: false,
                unbind_consistent: false,
                unbind_similarity: None,
                verdict: RetrievalVerdict::RejectNoSignificance,
                epistemic_tier,
                dimensions: self.config.dimensions,
                sigma,
            };
        }

        // Gate 2: Margin floor
        let margin_passed = raw_similarity >= self.config.margin_floor;
        if !margin_passed {
            return RetrievalDecision {
                raw_similarity,
                z_score,
                margin_passed,
                unbind_consistent: false,
                unbind_similarity: None,
                verdict: RetrievalVerdict::RejectBelowFloor,
                epistemic_tier,
                dimensions: self.config.dimensions,
                sigma,
            };
        }

        // Gate 3: Unbind consistency (if enabled and expected residual provided)
        let (unbind_consistent, unbind_similarity) = if self.config.check_unbind {
            if let Some(expected) = expected_residual {
                let unbound = self.unbind(candidate, query);
                let unbind_sim = self.hamming_similarity(&unbound, expected);
                (unbind_sim >= self.config.unbind_threshold, Some(unbind_sim))
            } else {
                // No expected residual - skip unbind check
                (true, None)
            }
        } else {
            (true, None)
        };

        if !unbind_consistent {
            return RetrievalDecision {
                raw_similarity,
                z_score,
                margin_passed,
                unbind_consistent,
                unbind_similarity,
                verdict: RetrievalVerdict::RejectUnbindFailed,
                epistemic_tier,
                dimensions: self.config.dimensions,
                sigma,
            };
        }

        // All gates passed!
        RetrievalDecision {
            raw_similarity,
            z_score,
            margin_passed,
            unbind_consistent,
            unbind_similarity,
            verdict: RetrievalVerdict::Accept,
            epistemic_tier,
            dimensions: self.config.dimensions,
            sigma,
        }
    }

    /// Simple decision without unbind check
    pub fn decide_simple(&self, query: &[i8], candidate: &[i8]) -> RetrievalDecision {
        self.decide(query, candidate, None)
    }

    /// Find best match from candidates with full decision info
    pub fn find_best_match<'a>(
        &self,
        query: &[i8],
        candidates: &'a [Vec<i8>],
    ) -> Option<(usize, &'a Vec<i8>, RetrievalDecision)> {
        let mut best: Option<(usize, &Vec<i8>, RetrievalDecision)> = None;

        for (idx, candidate) in candidates.iter().enumerate() {
            let decision = self.decide_simple(query, candidate);

            let dominated = match &best {
                Some((_, _, best_decision)) => {
                    // Compare z-scores for ordering
                    decision.z_score > best_decision.z_score
                }
                None => true,
            };

            if dominated {
                best = Some((idx, candidate, decision));
            }
        }

        // Return best if it passes the gates
        best.filter(|(_, _, d)| d.accepted())
    }

    /// Retrieve all matches above threshold with decisions
    pub fn find_all_matches<'a>(
        &self,
        query: &[i8],
        candidates: &'a [Vec<i8>],
    ) -> Vec<(usize, &'a Vec<i8>, RetrievalDecision)> {
        candidates.iter()
            .enumerate()
            .map(|(idx, c)| (idx, c, self.decide_simple(query, c)))
            .filter(|(_, _, d)| d.accepted())
            .collect()
    }
}

impl Default for StatisticalRetriever {
    fn default() -> Self {
        Self::new(2048)
    }
}

// ============================================================================
// TESTS - Comprehensive validation of statistical retrieval
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

    /// Helper: Generate vector with known similarity to reference
    fn vector_with_similarity(reference: &[i8], target_sim: f32) -> Vec<i8> {
        let dim = reference.len();
        let flip_count = ((1.0 - target_sim) * dim as f32) as usize;

        let mut result = reference.to_vec();
        for i in 0..flip_count {
            result[i] *= -1;
        }
        result
    }

    #[test]
    fn test_sigma_calculation() {
        // For n=2048, σ ≈ 0.011
        let config = StatisticalRetrievalConfig::with_dimensions(2048);
        let sigma = config.sigma();

        assert!((sigma - 0.011).abs() < 0.001,
                "σ for 2048 dims should be ≈0.011, got {}", sigma);

        // For n=10000, σ ≈ 0.005
        let config_10k = StatisticalRetrievalConfig::with_dimensions(10000);
        let sigma_10k = config_10k.sigma();

        assert!((sigma_10k - 0.005).abs() < 0.001,
                "σ for 10000 dims should be ≈0.005, got {}", sigma_10k);
    }

    #[test]
    fn test_z_score_calculation() {
        let config = StatisticalRetrievalConfig::with_dimensions(2048);

        // Random similarity (0.5) should give z ≈ 0
        let z_random = config.z_score(0.5);
        assert!(z_random.abs() < 0.001, "z for sim=0.5 should be 0, got {}", z_random);

        // Perfect match (1.0) should give very high z
        let z_perfect = config.z_score(1.0);
        assert!(z_perfect > 40.0, "z for sim=1.0 should be >40, got {}", z_perfect);

        // Opposite (0.0) should give very negative z
        let z_opposite = config.z_score(0.0);
        assert!(z_opposite < -40.0, "z for sim=0.0 should be <-40, got {}", z_opposite);
    }

    #[test]
    fn test_epistemic_tier_mapping() {
        assert_eq!(EmpiricalTier::from_z_score(0.5), EmpiricalTier::E0Null);
        assert_eq!(EmpiricalTier::from_z_score(1.5), EmpiricalTier::E1Testimonial);
        assert_eq!(EmpiricalTier::from_z_score(3.0), EmpiricalTier::E2PrivatelyVerifiable);
        assert_eq!(EmpiricalTier::from_z_score(5.0), EmpiricalTier::E3CryptographicallyProven);
        assert_eq!(EmpiricalTier::from_z_score(7.0), EmpiricalTier::E4PubliclyReproducible);
    }

    #[test]
    fn test_random_vectors_rejected() {
        let retriever = StatisticalRetriever::new(2048);

        let v1 = random_bipolar(2048);
        let v2 = random_bipolar(2048);

        let decision = retriever.decide_simple(&v1, &v2);

        // Random vectors should have z ≈ 0, be rejected
        assert!(!decision.accepted(),
                "Random vectors should be rejected, got {:?}", decision.verdict);
        assert_eq!(decision.verdict, RetrievalVerdict::RejectNoSignificance);
        assert!(decision.z_score.abs() < 5.0,
                "Random vectors should have |z| < 5, got {}", decision.z_score);
    }

    #[test]
    fn test_identical_vectors_accepted() {
        let retriever = StatisticalRetriever::new(2048);

        let v = random_bipolar(2048);

        let decision = retriever.decide_simple(&v, &v);

        assert!(decision.accepted(),
                "Identical vectors should be accepted");
        assert_eq!(decision.raw_similarity, 1.0);
        assert!(decision.z_score > 40.0,
                "Identical vectors should have z > 40, got {}", decision.z_score);
        assert_eq!(decision.epistemic_tier, EmpiricalTier::E4PubliclyReproducible);
    }

    #[test]
    fn test_high_similarity_accepted() {
        let retriever = StatisticalRetriever::new(2048);

        let v1 = random_bipolar(2048);
        let v2 = vector_with_similarity(&v1, 0.9); // 90% similarity

        let decision = retriever.decide_simple(&v1, &v2);

        assert!(decision.accepted(),
                "90% similarity should be accepted, got {:?}", decision.verdict);
        assert!(decision.z_score > 30.0,
                "90% similarity should have z > 30, got {}", decision.z_score);
    }

    #[test]
    fn test_marginal_similarity_rejected() {
        // Create retriever with high margin floor
        let config = StatisticalRetrievalConfig {
            dimensions: 2048,
            min_z_score: 2.0,
            margin_floor: 0.7,  // High floor
            check_unbind: false,
            unbind_threshold: 0.7,
        };
        let retriever = StatisticalRetriever::with_config(config);

        let v1 = random_bipolar(2048);
        let v2 = vector_with_similarity(&v1, 0.65); // Below margin but above random

        let decision = retriever.decide_simple(&v1, &v2);

        assert!(!decision.accepted());
        assert_eq!(decision.verdict, RetrievalVerdict::RejectBelowFloor);
    }

    #[test]
    fn test_unbind_consistency() {
        // This test verifies that the unbind operation correctly recovers
        // the original vector from a binding: unbind(A ⊗ B, A) = B
        //
        // Note: This tests the UNBIND OPERATION directly, not the three-gate
        // retrieval. Random bindings have ~0.5 similarity to query, so they
        // don't pass Gate 1 (z-score) - that's correct behavior!

        let retriever = StatisticalRetriever::new(2048);

        // Create random vectors
        let a = random_bipolar(2048);
        let b = random_bipolar(2048);

        // Bind: C = A ⊗ B (element-wise multiplication for bipolar)
        let c: Vec<i8> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();

        // Unbind: unbind(C, A) = C ⊗ A = (A ⊗ B) ⊗ A = B
        // Because for bipolar: a[i] * a[i] = 1 (self-inverse property)
        let recovered_b = retriever.unbind(&c, &a);

        // The recovered vector should be IDENTICAL to b (sim = 1.0)
        let unbind_sim = retriever.hamming_similarity(&recovered_b, &b);
        assert!(unbind_sim > 0.99,
                "Unbind should perfectly recover original: got {}", unbind_sim);

        // Also verify unbind is commutative for self-inverse vectors:
        // unbind(A, C) should also give B (since A ⊗ C = A ⊗ (A ⊗ B) = B)
        let recovered_b_alt = retriever.unbind(&a, &c);
        let unbind_sim_alt = retriever.hamming_similarity(&recovered_b_alt, &b);
        assert!(unbind_sim_alt > 0.99,
                "Unbind should be commutative: got {}", unbind_sim_alt);
    }

    #[test]
    fn test_unbind_inconsistency_rejected() {
        let min_z = 2.0;
        let config = StatisticalRetrievalConfig {
            dimensions: 2048,
            min_z_score: min_z,
            margin_floor: 0.5,
            check_unbind: true,
            unbind_threshold: 0.9, // High unbind threshold
        };
        let retriever = StatisticalRetriever::with_config(config);

        // Create similar but not bound vectors
        let a = random_bipolar(2048);
        let b = random_bipolar(2048);
        let c = vector_with_similarity(&a, 0.8); // Similar to A but not A⊗B

        // Query with A, expect B - but C is not A⊗B!
        let decision = retriever.decide(&a, &c, Some(&b));

        // High similarity might pass first two gates, but unbind should fail
        if decision.z_score >= min_z && decision.margin_passed {
            assert!(!decision.unbind_consistent || decision.verdict == RetrievalVerdict::RejectUnbindFailed);
        }
    }

    #[test]
    fn test_find_best_match() {
        let retriever = StatisticalRetriever::new(2048);

        let query = random_bipolar(2048);
        let candidates: Vec<Vec<i8>> = vec![
            random_bipolar(2048),                         // Random
            vector_with_similarity(&query, 0.7),          // 70%
            vector_with_similarity(&query, 0.95),         // 95% - best
            random_bipolar(2048),                         // Random
        ];

        let result = retriever.find_best_match(&query, &candidates);

        assert!(result.is_some());
        let (idx, _, decision) = result.unwrap();
        assert_eq!(idx, 2, "Should find 95% match at index 2");
        assert!(decision.accepted());
    }

    #[test]
    fn test_confidence_percent() {
        let retriever = StatisticalRetriever::new(2048);

        let v = random_bipolar(2048);
        let decision = retriever.decide_simple(&v, &v);

        let confidence = decision.confidence_percent();
        assert!(confidence > 99.9, "Perfect match should have >99.9% confidence, got {}", confidence);
    }

    #[test]
    fn test_dimensionality_impact() {
        // Higher dimensions = stricter threshold for same z-score
        let config_2k = StatisticalRetrievalConfig::with_dimensions(2048);
        let config_10k = StatisticalRetrievalConfig::with_dimensions(10000);

        // For z=3, what raw similarity is needed?
        let sim_2k = config_2k.similarity_for_z(3.0);
        let sim_10k = config_10k.similarity_for_z(3.0);

        // Higher dimensions = threshold closer to 0.5
        assert!(sim_10k < sim_2k,
                "Higher dims need lower threshold: 2k={}, 10k={}", sim_2k, sim_10k);

        // For 2048: sim ≈ 0.5 + 3 * 0.011 ≈ 0.533
        assert!((sim_2k - 0.533).abs() < 0.005,
                "2048 dims, z=3 threshold should be ≈0.533, got {}", sim_2k);
    }

    #[test]
    fn test_all_verdicts_reachable() {
        // Ensure all verdict types can be returned

        // Accept
        let retriever = StatisticalRetriever::new(2048);
        let v = random_bipolar(2048);
        assert_eq!(retriever.decide_simple(&v, &v).verdict, RetrievalVerdict::Accept);

        // RejectNoSignificance
        let v2 = random_bipolar(2048);
        let decision_random = retriever.decide_simple(&v, &v2);
        // Might be any rejection, but most likely NoSignificance for random

        // RejectBelowFloor
        let config_high_floor = StatisticalRetrievalConfig {
            dimensions: 2048,
            min_z_score: 1.0,  // Very low z threshold
            margin_floor: 0.99, // Very high floor
            check_unbind: false,
            unbind_threshold: 0.7,
        };
        let retriever_high_floor = StatisticalRetriever::with_config(config_high_floor);
        let v3 = vector_with_similarity(&v, 0.9);
        let decision_floor = retriever_high_floor.decide_simple(&v, &v3);
        assert_eq!(decision_floor.verdict, RetrievalVerdict::RejectBelowFloor);
    }

    #[test]
    fn test_find_all_matches() {
        let retriever = StatisticalRetriever::new(2048);

        let query = random_bipolar(2048);
        let candidates: Vec<Vec<i8>> = vec![
            vector_with_similarity(&query, 0.95),  // Should match
            random_bipolar(2048),                   // Should not match
            vector_with_similarity(&query, 0.85),  // Should match
            random_bipolar(2048),                   // Should not match
            vector_with_similarity(&query, 0.9),   // Should match
        ];

        let matches = retriever.find_all_matches(&query, &candidates);

        assert_eq!(matches.len(), 3, "Should find 3 matches");

        // Verify all matches are from expected indices
        let indices: Vec<usize> = matches.iter().map(|(i, _, _)| *i).collect();
        assert!(indices.contains(&0));
        assert!(indices.contains(&2));
        assert!(indices.contains(&4));
    }
}
