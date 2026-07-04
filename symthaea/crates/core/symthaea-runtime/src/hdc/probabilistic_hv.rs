// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Probabilistic Hyperdimensional Computing (Probabilistic HDC)
//!
//! Extends classical HDC with explicit uncertainty representation, enabling
//! Bayesian reasoning in hyperdimensional space. Where standard hypervectors
//! encode beliefs as deterministic points, probabilistic hypervectors (PHVs)
//! encode beliefs as *distributions*, capturing both what we think (mean)
//! and how sure we are (variance).
//!
//! # Connection to Consciousness and Active Inference
//!
//! Uncertainty is fundamental to perception and cognition. The brain does not
//! encode the world as a fixed snapshot; it maintains probabilistic beliefs
//! that are continuously updated via sensory evidence (Bayesian brain hypothesis).
//! Active Inference formalises this: an agent minimises free energy by updating
//! its generative model --- precisely the Bayesian update operation provided here.
//!
//! Probabilistic HDC gives Symthaea's Holographic Liquid Brain a native
//! substrate for:
//!
//! - **Epistemic uncertainty**: "I'm 90% sure dimension i is +1"
//! - **Bayesian belief updating**: posterior = f(prior, evidence)
//! - **Confidence-weighted reasoning**: certain dimensions dominate similarity
//! - **Information-theoretic introspection**: how much does this vector *know*?
//!
//! # Mathematical Foundations
//!
//! A Probabilistic Hypervector (PHV) represents each dimension as an
//! independent Gaussian: PHV = (mu, sigma^2) where mu in R^d and sigma^2 in R^d.
//!
//! All algebraic operations (bind, bundle, permute) are lifted to operate
//! on the distribution parameters, propagating uncertainty through computation
//! using standard rules for products and sums of independent normals.
//!
//! # Examples
//!
//! ```rust,ignore
//! use symthaea::hdc::probabilistic_hdc::{ProbabilisticHV, PHVConfig};
//!
//! let config = PHVConfig::default();
//! let mut belief = ProbabilisticHV::random(16_384, 42);
//!
//! // Observe evidence and update
//! let evidence: Vec<f64> = vec![1.0; 16_384];
//! belief.bayesian_update(&evidence, 0.1);
//!
//! // Uncertainty has decreased
//! assert!(belief.confidence() > 0.5);
//! ```

use serde::{Deserialize, Serialize};

use super::unified_hv::{BinaryHV, ContinuousHV};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for probabilistic hypervector operations.
///
/// Controls the default, minimum, and maximum variance values to prevent
/// numerical instability (zero variance causes division-by-zero; unbounded
/// variance loses all information).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PHVConfig {
    /// Default per-dimension variance for newly created PHVs.
    pub default_variance: f64,
    /// Floor on variance to prevent division-by-zero in KL and similarity.
    pub min_variance: f64,
    /// Ceiling on variance to keep information content bounded.
    pub max_variance: f64,
}

impl Default for PHVConfig {
    fn default() -> Self {
        Self {
            default_variance: 1.0,
            min_variance: 1e-12,
            max_variance: 1e6,
        }
    }
}

// ============================================================================
// Similarity result
// ============================================================================

/// Result of a confidence-weighted similarity computation between two PHVs.
///
/// In addition to the expected similarity (a scalar), this captures how
/// *reliable* that estimate is, reflecting the uncertainty in both operands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PHVSimilarity {
    /// Expected cosine-like similarity weighted by inverse variance.
    pub expected_similarity: f64,
    /// Variance of the similarity estimate (higher = less reliable).
    pub similarity_variance: f64,
    /// Overall confidence in [0, 1]: 1 means both vectors are maximally certain.
    pub confidence: f64,
}

// ============================================================================
// Probabilistic Hypervector
// ============================================================================

/// A probabilistic hypervector: (mean, variance) per dimension.
///
/// This is a diagonal-Gaussian representation of a belief over
/// hyperdimensional space. Each dimension i is modelled as an independent
/// Gaussian N(mean[i], variance[i]).
///
/// # Design Choices
///
/// - **Diagonal covariance**: Full covariance would be O(d^2) = O(268M)
///   parameters at d=16,384, which is impractical. The diagonal assumption
///   is standard in variational inference (mean-field approximation).
/// - **f64 precision**: Variance propagation accumulates products; f32
///   overflows/underflows too easily.
/// - **No Cargo.toml dependencies**: All math is implemented inline using
///   standard library operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticHV {
    /// Mean vector mu in R^d.
    pub mean: Vec<f64>,
    /// Per-dimension variance sigma^2 in R^d (always > 0).
    pub variance: Vec<f64>,
    /// Dimensionality of the hypervector.
    pub dim: usize,
}

impl ProbabilisticHV {
    // ========================================================================
    // Constructors
    // ========================================================================

    /// Create a new PHV with zero mean and uniform default variance.
    ///
    /// This represents maximal epistemic uncertainty: we know nothing,
    /// so every dimension is centred at zero with unit variance.
    pub fn new(dim: usize) -> Self {
        let config = PHVConfig::default();
        Self {
            mean: vec![0.0; dim],
            variance: vec![config.default_variance; dim],
            dim,
        }
    }

    /// Create a new PHV with zero mean and the specified uniform variance.
    pub fn with_variance(dim: usize, variance: f64) -> Self {
        let config = PHVConfig::default();
        let clamped = variance.max(config.min_variance).min(config.max_variance);
        Self {
            mean: vec![0.0; dim],
            variance: vec![clamped; dim],
            dim,
        }
    }

    /// Convert a binary hypervector (BinaryHV) to a probabilistic one.
    ///
    /// Each bit maps to mean +1.0 (bit=1) or -1.0 (bit=0), with zero
    /// variance: a binary vector is a maximally certain belief.
    pub fn from_hv16(hv: &BinaryHV) -> Self {
        let dim = BinaryHV::DIM; // 16,384
        let mut mean = Vec::with_capacity(dim);
        for i in 0..dim {
            let bit = hv.get_bit(i);
            mean.push(if bit == 1 { 1.0 } else { -1.0 });
        }
        Self {
            mean,
            variance: vec![0.0; dim],
            dim,
        }
    }

    /// Convert a real-valued hypervector to a probabilistic one.
    ///
    /// The mean is taken directly from the ContinuousHV values; variance is zero
    /// (the observation is treated as perfectly certain).
    pub fn from_real_hv(hv: &ContinuousHV) -> Self {
        let dim = hv.values.len();
        let mean: Vec<f64> = hv.values.iter().map(|&v| v as f64).collect();
        Self {
            mean,
            variance: vec![0.0; dim],
            dim,
        }
    }

    /// Create a random PHV with deterministic mean and default variance.
    ///
    /// Uses a simple xorshift64 PRNG seeded from `seed` to generate
    /// mean values in [-1, 1]. Variance is set to the default (1.0).
    pub fn random(dim: usize, seed: u64) -> Self {
        let config = PHVConfig::default();
        let mut state = seed;
        // Avoid zero state which would make xorshift degenerate
        if state == 0 {
            state = 0xDEAD_BEEF_CAFE_BABE;
        }
        let mut mean = Vec::with_capacity(dim);
        for _ in 0..dim {
            // xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Map to [-1, 1]
            let normalised = (state as f64) / (u64::MAX as f64); // [0, 1)
            mean.push(normalised * 2.0 - 1.0);
        }
        Self {
            mean,
            variance: vec![config.default_variance; dim],
            dim,
        }
    }

    // ========================================================================
    // Core HDC Operations (lifted to distributions)
    // ========================================================================

    /// Probabilistic binding (element-wise product of Gaussians).
    ///
    /// For independent PHVs A = (mu_a, s2_a) and B = (mu_b, s2_b), the
    /// product Z_i = A_i * B_i has:
    ///
    ///   E[Z_i]   = mu_a_i * mu_b_i
    ///   Var[Z_i] = mu_a_i^2 * s2_b_i + mu_b_i^2 * s2_a_i + s2_a_i * s2_b_i
    ///
    /// This follows from E[X*Y] = E[X]*E[Y] and Var[X*Y] for independent X, Y.
    ///
    /// # Panics
    ///
    /// Panics if `self` and `other` have different dimensions.
    pub fn bind(&self, other: &Self) -> Self {
        assert_eq!(
            self.dim, other.dim,
            "Cannot bind PHVs of different dimensions ({} vs {})",
            self.dim, other.dim
        );
        let config = PHVConfig::default();
        let mut result_mean = Vec::with_capacity(self.dim);
        let mut result_var = Vec::with_capacity(self.dim);

        for i in 0..self.dim {
            let ma = self.mean[i];
            let mb = other.mean[i];
            let va = self.variance[i];
            let vb = other.variance[i];

            result_mean.push(ma * mb);
            // Var(X*Y) = E[X]^2 Var(Y) + E[Y]^2 Var(X) + Var(X) Var(Y)
            let v = ma * ma * vb + mb * mb * va + va * vb;
            result_var.push(v.max(config.min_variance));
        }

        Self {
            mean: result_mean,
            variance: result_var,
            dim: self.dim,
        }
    }

    /// Probabilistic bundling (weighted average with uncertainty propagation).
    ///
    /// Given PHVs A_1, ..., A_n, the bundle is:
    ///
    ///   mu_bundle   = (1/n) * sum(mu_i)
    ///   var_bundle  = (1/n^2) * sum(s2_i) + var(mu_i) / n
    ///
    /// The first term propagates individual uncertainties through averaging;
    /// the second captures *disagreement* between the input vectors. Even if
    /// each input is perfectly certain, conflicting means produce uncertainty
    /// in the bundle --- a form of aleatoric uncertainty from mixing.
    ///
    /// # Panics
    ///
    /// Panics if `hvs` is empty or if the PHVs have different dimensions.
    pub fn bundle(hvs: &[&Self]) -> Self {
        assert!(!hvs.is_empty(), "Cannot bundle empty set of PHVs");
        let dim = hvs[0].dim;
        for hv in hvs.iter().skip(1) {
            assert_eq!(
                hv.dim, dim,
                "All PHVs in a bundle must have the same dimension"
            );
        }

        let config = PHVConfig::default();
        let n = hvs.len() as f64;
        let n2 = n * n;
        let mut result_mean = vec![0.0; dim];
        let mut result_var = vec![0.0; dim];

        for i in 0..dim {
            // Sum of means
            let sum_mu: f64 = hvs.iter().map(|hv| hv.mean[i]).sum();
            let mean_of_means = sum_mu / n;

            // Sum of variances (for the averaging noise term)
            let sum_var: f64 = hvs.iter().map(|hv| hv.variance[i]).sum();

            // Variance of means (disagreement)
            let var_of_means: f64 = hvs
                .iter()
                .map(|hv| {
                    let diff = hv.mean[i] - mean_of_means;
                    diff * diff
                })
                .sum::<f64>()
                / n;

            result_mean[i] = mean_of_means;
            let v = sum_var / n2 + var_of_means / n;
            result_var[i] = v.max(config.min_variance);
        }

        Self {
            mean: result_mean,
            variance: result_var,
            dim,
        }
    }

    /// Probabilistic permutation (circular shift).
    ///
    /// Rotates both the mean and variance vectors by `shift` positions.
    /// This preserves the full uncertainty structure while encoding
    /// sequential/positional information, just as deterministic permute does
    /// in standard HDC.
    pub fn permute(&self, shift: usize) -> Self {
        if self.dim == 0 {
            return self.clone();
        }
        let effective_shift = shift % self.dim;
        let mut new_mean = vec![0.0; self.dim];
        let mut new_var = vec![0.0; self.dim];

        for i in 0..self.dim {
            let target = (i + effective_shift) % self.dim;
            new_mean[target] = self.mean[i];
            new_var[target] = self.variance[i];
        }

        Self {
            mean: new_mean,
            variance: new_var,
            dim: self.dim,
        }
    }

    // ========================================================================
    // Similarity and Divergence
    // ========================================================================

    /// Confidence-weighted similarity between two PHVs.
    ///
    /// Instead of raw cosine similarity, each dimension's contribution is
    /// weighted by the inverse of the geometric mean of the two variances:
    ///
    ///   sim = (1/d) * sum_i [ mu_a_i * mu_b_i / (sigma_a_i * sigma_b_i) ]
    ///
    /// Dimensions where *both* vectors are certain dominate the result;
    /// uncertain dimensions contribute less. This is the natural metric
    /// for probabilistic beliefs --- the brain should trust its confident
    /// percepts more than its uncertain ones.
    ///
    /// Returns a [`PHVSimilarity`] containing the expected similarity,
    /// its variance, and an overall confidence score.
    pub fn similarity(&self, other: &Self) -> PHVSimilarity {
        assert_eq!(
            self.dim, other.dim,
            "Cannot compute similarity of PHVs with different dimensions"
        );

        let config = PHVConfig::default();
        let d = self.dim as f64;
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut var_sum = 0.0;

        for i in 0..self.dim {
            let va = self.variance[i].max(config.min_variance);
            let vb = other.variance[i].max(config.min_variance);
            let sigma_a = va.sqrt();
            let sigma_b = vb.sqrt();
            let weight = 1.0 / (sigma_a * sigma_b);

            weighted_sum += self.mean[i] * other.mean[i] * weight;
            weight_sum += weight;

            // Variance of the product mu_a * mu_b given uncertainties
            let product_var = self.mean[i].powi(2) * vb + other.mean[i].powi(2) * va + va * vb;
            var_sum += product_var * weight * weight;
        }

        let expected_sim = if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.0
        };

        let sim_var = if weight_sum > 0.0 {
            var_sum / (weight_sum * weight_sum)
        } else {
            f64::INFINITY
        };

        // Confidence: average precision (1/variance), normalised to [0, 1]
        let avg_precision_self: f64 = self
            .variance
            .iter()
            .map(|&v| 1.0 / v.max(config.min_variance))
            .sum::<f64>()
            / d;
        let avg_precision_other: f64 = other
            .variance
            .iter()
            .map(|&v| 1.0 / v.max(config.min_variance))
            .sum::<f64>()
            / d;
        // Geometric mean of normalised precisions; at default_variance=1.0,
        // precision=1.0 maps to confidence ~0.5 via sigmoid-like scaling.
        let combined = (avg_precision_self * avg_precision_other).sqrt();
        let confidence = combined / (1.0 + combined); // sigmoid-ish in [0, 1)

        PHVSimilarity {
            expected_similarity: expected_sim,
            similarity_variance: sim_var,
            confidence,
        }
    }

    /// KL divergence D_KL(self || other) assuming diagonal Gaussian.
    ///
    ///   D_KL(P||Q) = (1/2) sum_i [ log(s2_q_i / s2_p_i)
    ///                               + (s2_p_i + (mu_p_i - mu_q_i)^2) / s2_q_i
    ///                               - 1 ]
    ///
    /// Returns 0.0 for identical distributions. Always non-negative.
    ///
    /// Note: D_KL is asymmetric. For a symmetric measure, compute
    /// `(a.kl_divergence(b) + b.kl_divergence(a)) / 2`.
    ///
    /// # Panics
    ///
    /// Panics if dimensions differ.
    pub fn kl_divergence(&self, other: &Self) -> f64 {
        assert_eq!(
            self.dim, other.dim,
            "Cannot compute KL divergence of PHVs with different dimensions"
        );

        let config = PHVConfig::default();
        let mut kl = 0.0;

        for i in 0..self.dim {
            let sp = self.variance[i].max(config.min_variance);
            let sq = other.variance[i].max(config.min_variance);
            let diff = self.mean[i] - other.mean[i];

            kl += (sq / sp).ln() + (sp + diff * diff) / sq - 1.0;
        }

        // The formula includes a factor of 1/2
        (kl * 0.5).max(0.0)
    }

    // ========================================================================
    // Bayesian Update (Active Inference core operation)
    // ========================================================================

    /// Bayesian update: incorporate an observation to refine the belief.
    ///
    /// Given a prior PHV (mu_prior, s2_prior) and an observation vector x
    /// with assumed observation variance s2_obs, the posterior is:
    ///
    ///   mu_post_i  = (s2_obs * mu_prior_i + s2_prior_i * x_i) / (s2_prior_i + s2_obs)
    ///   s2_post_i  = (s2_prior_i * s2_obs) / (s2_prior_i + s2_obs)
    ///
    /// This is the standard Gaussian conjugate update. Variance always
    /// decreases: evidence makes us more certain. This is the mathematical
    /// core of Active Inference's perceptual updating --- the agent refines
    /// its generative model upon receiving sensory data.
    ///
    /// # Arguments
    ///
    /// * `observation` - Observed values, one per dimension.
    /// * `obs_variance` - Assumed variance of the observation noise
    ///   (scalar, applied uniformly). Smaller values mean we trust the
    ///   observation more.
    ///
    /// # Panics
    ///
    /// Panics if `observation.len() != self.dim`.
    pub fn bayesian_update(&mut self, observation: &[f64], obs_variance: f64) {
        assert_eq!(
            observation.len(),
            self.dim,
            "Observation dimension ({}) must match PHV dimension ({})",
            observation.len(),
            self.dim
        );

        let config = PHVConfig::default();
        let obs_var = obs_variance.max(config.min_variance);

        for ((m, v), &obs) in self
            .mean
            .iter_mut()
            .zip(self.variance.iter_mut())
            .zip(observation.iter())
        {
            let sp = v.max(config.min_variance);
            let denom = sp + obs_var;

            // Posterior mean: weighted combination of prior and observation
            *m = (obs_var * *m + sp * obs) / denom;
            // Posterior variance: harmonic-mean-like reduction
            *v = (sp * obs_var) / denom;
        }
    }

    // ========================================================================
    // Information-Theoretic Measures
    // ========================================================================

    /// Differential entropy of the PHV (in nats).
    ///
    /// For a d-dimensional diagonal Gaussian:
    ///
    ///   H = (d/2) * log(2*pi*e) + (1/2) * sum_i log(s2_i)
    ///
    /// Higher entropy means more uncertainty. A PHV with zero variance
    /// (deterministic) has entropy -infinity; a PHV with unit variance
    /// has entropy (d/2) * log(2*pi*e).
    pub fn entropy(&self) -> f64 {
        let config = PHVConfig::default();
        let d = self.dim as f64;
        let base = d * 0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E).ln();
        let log_var_sum: f64 = self
            .variance
            .iter()
            .map(|&v| v.max(config.min_variance).ln())
            .sum();
        base + 0.5 * log_var_sum
    }

    /// Information content (negative entropy): how much the PHV *knows*.
    ///
    ///   I(A) = -H(A) = -(d/2) * log(2*pi*e) - (1/2) * sum_i log(s2_i)
    ///
    /// Low variance = high information content. A perfectly certain vector
    /// (s2 -> 0) has information content -> +infinity. In practice, clamped
    /// by `min_variance`.
    ///
    /// This connects to IIT's notion of integrated information: a system
    /// that "knows" more (lower uncertainty) has higher information content.
    pub fn information_content(&self) -> f64 {
        -self.entropy()
    }

    /// Average confidence (mean precision) across all dimensions.
    ///
    /// Precision is the inverse of variance. Returns the mean precision,
    /// which is a scalar summary of "how certain is this PHV overall?"
    ///
    /// Returns a value in (0, +inf). Higher means more certain.
    pub fn confidence(&self) -> f64 {
        let config = PHVConfig::default();
        let total: f64 = self
            .variance
            .iter()
            .map(|&v| 1.0 / v.max(config.min_variance))
            .sum();
        total / self.dim as f64
    }

    // ========================================================================
    // Conversion
    // ========================================================================

    /// Collapse the PHV to a deterministic ContinuousHV by taking the mean.
    ///
    /// This discards all uncertainty information. It is the "most likely"
    /// hypervector under the diagonal Gaussian model.
    pub fn to_real_hv(&self) -> ContinuousHV {
        let values: Vec<f32> = self.mean.iter().map(|&v| v as f32).collect();
        ContinuousHV::from_values(values)
    }

    /// Return the dimension of this PHV.
    pub fn dimension(&self) -> usize {
        self.dim
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_DIM: usize = 256;

    #[test]
    fn test_bind_self_twice_recovers_identity() {
        // Binding A with itself in standard HDC gives identity (XOR: A^A = 0).
        // In probabilistic HDC with zero variance, bind(A, A) should give
        // mean[i] = mean[i]^2 which for +/- 1 values gives +1 everywhere.
        // With nonzero variance, the mean should be close to the squared original.
        let a = ProbabilisticHV::random(TEST_DIM, 42);
        let bound = a.bind(&a);

        // mean of bind(A,A) = mu_a^2 per dimension
        for i in 0..TEST_DIM {
            let expected = a.mean[i] * a.mean[i];
            assert!(
                (bound.mean[i] - expected).abs() < 1e-10,
                "Bind self: dim {} expected {} got {}",
                i,
                expected,
                bound.mean[i]
            );
        }
    }

    #[test]
    fn test_bundle_reduces_variance() {
        // Central limit theorem: bundling n independent PHVs with variance v
        // should yield variance approximately v/n (plus disagreement term).
        // With identical copies, disagreement = 0, so var -> v / n^2 * n = v / n.
        let a = ProbabilisticHV::with_variance(TEST_DIM, 1.0);
        // Bundle 10 copies of the same PHV
        let refs: Vec<&ProbabilisticHV> = (0..10).map(|_| &a).collect();
        let bundled = ProbabilisticHV::bundle(&refs);

        let original_var = 1.0;
        let n = 10.0;
        // Expected variance = sum(s2_i) / n^2 + var(mu_i)/n
        // Since all mu_i are identical (0.0), var(mu_i) = 0
        // So expected = n * original_var / n^2 = original_var / n = 0.1
        let expected_var = original_var / n;

        for i in 0..TEST_DIM {
            assert!(
                (bundled.variance[i] - expected_var).abs() < 1e-10,
                "Bundle variance: dim {} expected {} got {}",
                i,
                expected_var,
                bundled.variance[i]
            );
        }
    }

    #[test]
    fn test_kl_divergence_identical_is_zero() {
        let a = ProbabilisticHV::random(TEST_DIM, 99);
        let kl = a.kl_divergence(&a);
        assert!(
            kl.abs() < 1e-10,
            "KL divergence of identical PHVs should be 0, got {}",
            kl
        );
    }

    #[test]
    fn test_kl_divergence_non_negative() {
        let a = ProbabilisticHV::random(TEST_DIM, 1);
        let b = ProbabilisticHV::random(TEST_DIM, 2);
        let kl_ab = a.kl_divergence(&b);
        let kl_ba = b.kl_divergence(&a);
        assert!(
            kl_ab >= 0.0,
            "KL(a||b) should be non-negative, got {}",
            kl_ab
        );
        assert!(
            kl_ba >= 0.0,
            "KL(b||a) should be non-negative, got {}",
            kl_ba
        );
    }

    #[test]
    fn test_bayesian_update_reduces_variance() {
        let mut phv = ProbabilisticHV::with_variance(TEST_DIM, 2.0);
        let initial_var: f64 = phv.variance.iter().sum::<f64>() / TEST_DIM as f64;

        // Observe with moderate noise
        let obs = vec![0.5; TEST_DIM];
        phv.bayesian_update(&obs, 1.0);

        let updated_var: f64 = phv.variance.iter().sum::<f64>() / TEST_DIM as f64;
        assert!(
            updated_var < initial_var,
            "Bayesian update should reduce variance: {} -> {}",
            initial_var,
            updated_var
        );

        // Exact posterior variance: (2.0 * 1.0) / (2.0 + 1.0) = 2/3
        let expected_var = (2.0 * 1.0) / (2.0 + 1.0);
        assert!(
            (updated_var - expected_var).abs() < 1e-10,
            "Posterior variance should be {}, got {}",
            expected_var,
            updated_var
        );
    }

    #[test]
    fn test_bayesian_update_shifts_mean_toward_observation() {
        let mut phv = ProbabilisticHV::with_variance(TEST_DIM, 1.0);
        // Initial mean is 0.0 everywhere
        let obs = vec![1.0; TEST_DIM];
        phv.bayesian_update(&obs, 1.0);

        // Posterior mean: (1.0 * 0.0 + 1.0 * 1.0) / (1.0 + 1.0) = 0.5
        for i in 0..TEST_DIM {
            assert!(
                (phv.mean[i] - 0.5).abs() < 1e-10,
                "Mean should be 0.5 after update, got {} at dim {}",
                phv.mean[i],
                i
            );
        }
    }

    #[test]
    fn test_high_confidence_means_high_information_content() {
        let certain = ProbabilisticHV::with_variance(TEST_DIM, 0.01);
        let uncertain = ProbabilisticHV::with_variance(TEST_DIM, 10.0);

        assert!(
            certain.information_content() > uncertain.information_content(),
            "Certain PHV should have higher information content: {} vs {}",
            certain.information_content(),
            uncertain.information_content()
        );

        assert!(
            certain.confidence() > uncertain.confidence(),
            "Certain PHV should have higher confidence: {} vs {}",
            certain.confidence(),
            uncertain.confidence()
        );
    }

    #[test]
    fn test_phv_from_hv16_has_zero_variance() {
        let hv = BinaryHV::random(42);
        let phv = ProbabilisticHV::from_hv16(&hv);

        assert_eq!(phv.dim, BinaryHV::DIM);
        for i in 0..phv.dim {
            assert_eq!(
                phv.variance[i], 0.0,
                "PHV from BinaryHV should have zero variance at dim {}",
                i
            );
            // Mean should be +1 or -1
            assert!(
                (phv.mean[i] - 1.0).abs() < 1e-10 || (phv.mean[i] + 1.0).abs() < 1e-10,
                "PHV from BinaryHV should have mean +/-1, got {} at dim {}",
                phv.mean[i],
                i
            );
        }
    }

    #[test]
    fn test_phv_from_real_hv_has_zero_variance() {
        let hv = ContinuousHV::random(TEST_DIM, 77);
        let phv = ProbabilisticHV::from_real_hv(&hv);

        assert_eq!(phv.dim, TEST_DIM);
        for i in 0..phv.dim {
            assert_eq!(
                phv.variance[i], 0.0,
                "PHV from ContinuousHV should have zero variance at dim {}",
                i
            );
            assert!(
                (phv.mean[i] - hv.values[i] as f64).abs() < 1e-5,
                "PHV mean should match ContinuousHV value at dim {}",
                i
            );
        }
    }

    #[test]
    fn test_permute_preserves_content() {
        let a = ProbabilisticHV::random(TEST_DIM, 55);
        let shifted = a.permute(17);

        // Check the values rotated correctly
        for i in 0..TEST_DIM {
            let target = (i + 17) % TEST_DIM;
            assert!(
                (shifted.mean[target] - a.mean[i]).abs() < 1e-12,
                "Permute: mean mismatch at {} -> {}",
                i,
                target
            );
            assert!(
                (shifted.variance[target] - a.variance[i]).abs() < 1e-12,
                "Permute: variance mismatch at {} -> {}",
                i,
                target
            );
        }
    }

    #[test]
    fn test_permute_full_cycle() {
        let a = ProbabilisticHV::random(TEST_DIM, 33);
        let cycled = a.permute(TEST_DIM);

        // Full cycle should give back the original
        for i in 0..TEST_DIM {
            assert!(
                (cycled.mean[i] - a.mean[i]).abs() < 1e-12,
                "Full cycle permute should be identity"
            );
        }
    }

    #[test]
    fn test_to_real_hv_roundtrip() {
        let original = ContinuousHV::random(TEST_DIM, 88);
        let phv = ProbabilisticHV::from_real_hv(&original);
        let recovered = phv.to_real_hv();

        for i in 0..TEST_DIM {
            assert!(
                (original.values[i] - recovered.values[i]).abs() < 1e-5,
                "ContinuousHV roundtrip failed at dim {}: {} vs {}",
                i,
                original.values[i],
                recovered.values[i]
            );
        }
    }

    #[test]
    fn test_entropy_increases_with_variance() {
        let low_var = ProbabilisticHV::with_variance(TEST_DIM, 0.1);
        let high_var = ProbabilisticHV::with_variance(TEST_DIM, 10.0);

        assert!(
            high_var.entropy() > low_var.entropy(),
            "Higher variance should give higher entropy: {} vs {}",
            high_var.entropy(),
            low_var.entropy()
        );
    }

    #[test]
    fn test_similarity_confidence_scales_with_certainty() {
        // Two certain PHVs should give high confidence
        let a = ProbabilisticHV::with_variance(TEST_DIM, 0.001);
        let b = ProbabilisticHV::with_variance(TEST_DIM, 0.001);
        let sim_certain = a.similarity(&b);

        // Two uncertain PHVs should give low confidence
        let c = ProbabilisticHV::with_variance(TEST_DIM, 100.0);
        let d = ProbabilisticHV::with_variance(TEST_DIM, 100.0);
        let sim_uncertain = c.similarity(&d);

        assert!(
            sim_certain.confidence > sim_uncertain.confidence,
            "Certain PHVs should yield higher similarity confidence: {} vs {}",
            sim_certain.confidence,
            sim_uncertain.confidence
        );
    }

    #[test]
    fn test_bundle_of_one_is_identity() {
        let a = ProbabilisticHV::random(TEST_DIM, 42);
        let bundled = ProbabilisticHV::bundle(&[&a]);

        for i in 0..TEST_DIM {
            assert!(
                (bundled.mean[i] - a.mean[i]).abs() < 1e-10,
                "Bundle of one should preserve mean"
            );
            // Variance: sum_var / 1 + var_of_means/1
            // var_of_means = 0 (only one vector), sum_var = a.variance
            // So result = a.variance / 1 + 0 = a.variance
            assert!(
                (bundled.variance[i] - a.variance[i]).abs() < 1e-10,
                "Bundle of one should preserve variance"
            );
        }
    }

    #[test]
    fn test_repeated_bayesian_updates_converge() {
        // Many observations should drive variance toward zero
        let mut phv = ProbabilisticHV::with_variance(TEST_DIM, 10.0);
        let obs = vec![0.7; TEST_DIM];

        for _ in 0..100 {
            phv.bayesian_update(&obs, 1.0);
        }

        // After many updates, variance should be very small
        let avg_var: f64 = phv.variance.iter().sum::<f64>() / TEST_DIM as f64;
        assert!(
            avg_var < 0.01,
            "After 100 Bayesian updates, variance should be tiny, got {}",
            avg_var
        );

        // Mean should have converged close to the observation
        for i in 0..TEST_DIM {
            assert!(
                (phv.mean[i] - 0.7).abs() < 0.01,
                "After 100 updates, mean should be ~0.7, got {} at dim {}",
                phv.mean[i],
                i
            );
        }
    }
}
