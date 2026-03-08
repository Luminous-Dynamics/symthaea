//! K-Vector - Multi-dimensional Trust Scoring
//!
//! The K-Vector provides a comprehensive trust profile for agents and participants
//! in the Mycelix network. Each dimension captures a different aspect of trustworthiness.
//!
//! ## Dimensions (10 total)
//!
//! - `k_r` (Reputation): Historical track record and peer evaluations
//! - `k_a` (Activity): Recent participation and contribution frequency
//! - `k_i` (Integrity): Adherence to protocol rules, no detected violations
//! - `k_p` (Performance): Quality of contributions (gradients, votes, etc.)
//! - `k_m` (Membership): Duration of active participation in network
//! - `k_s` (Stake): Economic commitment (MYC, ETH, USDC staking)
//! - `k_h` (Historical): Long-term consistency and reliability
//! - `k_topo` (Topology): Network centrality and connectivity contribution
//! - `k_v` (Verification): Identity verification and credential validity
//! - `k_coherence` (Coherence): Output consistency / coherence metric
//!
//! Trust Score Formula (weighted average):
//!   T = 0.25×k_r + 0.11×k_a + 0.20×k_i + 0.11×k_p + 0.05×k_m + 0.07×k_s + 0.05×k_h + 0.04×k_topo + 0.12×k_v
//!
//! Note: k_coherence (Coherence) is tracked but has 0% weight in trust calculation.
//! Phi coherence is used for gating high-stakes operations, not trust scoring.
//!
//! ## Performance Optimizations
//!
//! - SIMD-friendly array layout for vectorized operations
//! - Cached trust score computation with invalidation
//! - Pre-computed weight arrays for batch processing

// K-Vector dimensions documented via struct-level and field-level docs

use serde::{Deserialize, Serialize};
use std::cell::Cell;

#[cfg(feature = "ts-export")]
use ts_rs::TS;

/// K-Vector: 10-dimensional trust profile
///
/// All components are normalized to [0.0, 1.0] range.
/// Higher values indicate greater trustworthiness in that dimension.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/matl/"))]
pub struct KVector {
    /// k_r: Reputation - Historical track record and peer evaluations (0.0-1.0)
    /// Weight: 0.24 (24%)
    pub k_r: f32,

    /// k_a: Activity - Recent participation and contribution frequency (0.0-1.0)
    /// Weight: 0.11 (11%)
    pub k_a: f32,

    /// k_i: Integrity - Adherence to protocol rules, no detected violations (0.0-1.0)
    /// Weight: 0.19 (19%)
    pub k_i: f32,

    /// k_p: Performance - Quality of contributions (gradients, votes, etc.) (0.0-1.0)
    /// Weight: 0.11 (11%)
    pub k_p: f32,

    /// k_m: Membership - Duration of active participation in network (0.0-1.0)
    /// Weight: 0.05 (5%)
    pub k_m: f32,

    /// k_s: Stake - Economic commitment (MYC, ETH, USDC staking) (0.0-1.0)
    /// Weight: 0.07 (7%)
    pub k_s: f32,

    /// k_h: Historical - Long-term consistency and reliability (0.0-1.0)
    /// Weight: 0.05 (5%)
    pub k_h: f32,

    /// k_topo: Topology - Network centrality and connectivity contribution (0.0-1.0)
    /// Weight: 0.04 (4%)
    pub k_topo: f32,

    /// k_v: Verification - Identity verification and credential validity (0.0-1.0)
    /// Weight: 0.09 (9%)
    ///
    /// Measures the strength of identity verification:
    /// - 0.0: Unverified anonymous identity
    /// - 0.3: Self-asserted identity (no external verification)
    /// - 0.5: Basic credential verification (email, phone)
    /// - 0.7: Strong credential verification (government ID, WebAuthn)
    /// - 0.9: Multi-factor verified with unexpired credentials
    /// - 1.0: Fully verified with ZK proofs and active attestations
    pub k_v: f32,

    /// k_coherence: Coherence - Output consistency metric (0.0-1.0)
    /// Weight: 0.00 (0%) - TRACKED FOR GATING ONLY, NOT USED IN TRUST SCORE
    ///
    /// Measures the agent's output coherence over time:
    /// - 0.0-0.1: Critical - agent should be suspended
    /// - 0.1-0.3: Degraded - restricted operations only
    /// - 0.3-0.5: Unstable - monitoring required
    /// - 0.5-0.7: Stable - normal operations allowed
    /// - 0.7-1.0: Coherent - high-stakes operations allowed
    ///
    /// This value is used by ZK operation gating (see coherence_bridge.rs) to prevent
    /// incoherent agents from performing critical operations. It does NOT affect
    /// the trust score calculation.
    pub k_coherence: f32,
}

/// Default weights for K-Vector trust score calculation
///
/// Note: k_coherence weight is 0% - coherence is tracked for gating but doesn't affect trust score.
/// The 5% that was on k_coherence has been redistributed to k_r (+1%), k_i (+1%), and k_v (+3%).
pub const KVECTOR_WEIGHTS: KVectorWeights = KVectorWeights {
    w_r: 0.25,    // Reputation (was 0.24)
    w_a: 0.11,    // Activity
    w_i: 0.20,    // Integrity (was 0.19)
    w_p: 0.11,    // Performance
    w_m: 0.05,    // Membership
    w_s: 0.07,    // Stake
    w_h: 0.05,    // Historical
    w_topo: 0.04, // Topology
    w_v: 0.12,    // Verification (was 0.09)
    w_phi: 0.00,  // Coherence - tracked but not weighted (used for gating only)
};

/// Pre-computed weight array for SIMD-friendly operations
/// Order: [k_r, k_a, k_i, k_p, k_m, k_s, k_h, k_topo, k_v, k_coherence]
/// Note: k_coherence (index 9) has 0 weight - tracked for gating, not scoring
pub const KVECTOR_WEIGHTS_ARRAY: [f32; 10] =
    [0.25, 0.11, 0.20, 0.11, 0.05, 0.07, 0.05, 0.04, 0.12, 0.00];

/// Configurable weights for K-Vector components
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/matl/"))]
pub struct KVectorWeights {
    /// Weight for reputation dimension (k_r)
    pub w_r: f32,
    /// Weight for activity dimension (k_a)
    pub w_a: f32,
    /// Weight for integrity dimension (k_i)
    pub w_i: f32,
    /// Weight for performance dimension (k_p)
    pub w_p: f32,
    /// Weight for membership dimension (k_m)
    pub w_m: f32,
    /// Weight for stake dimension (k_s)
    pub w_s: f32,
    /// Weight for historical dimension (k_h)
    pub w_h: f32,
    /// Weight for topology dimension (k_topo)
    pub w_topo: f32,
    /// Weight for verification dimension (k_v)
    pub w_v: f32,
    /// Weight for coherence dimension (k_coherence)
    pub w_phi: f32,
}

impl KVectorWeights {
    /// Verify weights sum to 1.0 (within floating point tolerance)
    pub fn is_valid(&self) -> bool {
        let sum = self.w_r
            + self.w_a
            + self.w_i
            + self.w_p
            + self.w_m
            + self.w_s
            + self.w_h
            + self.w_topo
            + self.w_v
            + self.w_phi;
        (sum - 1.0).abs() < 0.001
    }

    /// Normalize weights to sum to 1.0
    pub fn normalize(&mut self) {
        let sum = self.w_r
            + self.w_a
            + self.w_i
            + self.w_p
            + self.w_m
            + self.w_s
            + self.w_h
            + self.w_topo
            + self.w_v
            + self.w_phi;
        if sum > 0.0 {
            self.w_r /= sum;
            self.w_a /= sum;
            self.w_i /= sum;
            self.w_p /= sum;
            self.w_m /= sum;
            self.w_s /= sum;
            self.w_h /= sum;
            self.w_topo /= sum;
            self.w_v /= sum;
            self.w_phi /= sum;
        }
    }
}

impl KVector {
    /// Create a new K-Vector with all components specified
    ///
    /// For constructing from arrays, use `from_array()` instead.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        k_r: f32,
        k_a: f32,
        k_i: f32,
        k_p: f32,
        k_m: f32,
        k_s: f32,
        k_h: f32,
        k_topo: f32,
        k_v: f32,
        k_coherence: f32,
    ) -> Self {
        Self {
            k_r: k_r.clamp(0.0, 1.0),
            k_a: k_a.clamp(0.0, 1.0),
            k_i: k_i.clamp(0.0, 1.0),
            k_p: k_p.clamp(0.0, 1.0),
            k_m: k_m.clamp(0.0, 1.0),
            k_s: k_s.clamp(0.0, 1.0),
            k_h: k_h.clamp(0.0, 1.0),
            k_topo: k_topo.clamp(0.0, 1.0),
            k_v: k_v.clamp(0.0, 1.0),
            k_coherence: k_coherence.clamp(0.0, 1.0),
        }
    }

    /// Create a new K-Vector without k_coherence (legacy 9-dimensional compatibility)
    ///
    /// The k_coherence dimension defaults to 0.5 (neutral coherence).
    #[allow(clippy::too_many_arguments)]
    pub fn new_9d(
        k_r: f32,
        k_a: f32,
        k_i: f32,
        k_p: f32,
        k_m: f32,
        k_s: f32,
        k_h: f32,
        k_topo: f32,
        k_v: f32,
    ) -> Self {
        Self::new(k_r, k_a, k_i, k_p, k_m, k_s, k_h, k_topo, k_v, 0.5)
    }

    /// Convert to array form for SIMD-friendly batch operations
    /// Order: [k_r, k_a, k_i, k_p, k_m, k_s, k_h, k_topo, k_v, k_coherence]
    #[inline]
    pub fn to_array(&self) -> [f32; 10] {
        [
            self.k_r,
            self.k_a,
            self.k_i,
            self.k_p,
            self.k_m,
            self.k_s,
            self.k_h,
            self.k_topo,
            self.k_v,
            self.k_coherence,
        ]
    }

    /// Create from array (inverse of to_array)
    #[inline]
    pub fn from_array(arr: [f32; 10]) -> Self {
        Self::new(
            arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8], arr[9],
        )
    }

    /// Optimized trust score using array dot product
    /// This is more SIMD-friendly than the individual field access version
    #[inline]
    pub fn trust_score_fast(&self) -> f32 {
        let vals = self.to_array();
        // Manual unroll for better optimization
        vals[0] * KVECTOR_WEIGHTS_ARRAY[0]
            + vals[1] * KVECTOR_WEIGHTS_ARRAY[1]
            + vals[2] * KVECTOR_WEIGHTS_ARRAY[2]
            + vals[3] * KVECTOR_WEIGHTS_ARRAY[3]
            + vals[4] * KVECTOR_WEIGHTS_ARRAY[4]
            + vals[5] * KVECTOR_WEIGHTS_ARRAY[5]
            + vals[6] * KVECTOR_WEIGHTS_ARRAY[6]
            + vals[7] * KVECTOR_WEIGHTS_ARRAY[7]
            + vals[8] * KVECTOR_WEIGHTS_ARRAY[8]
            + vals[9] * KVECTOR_WEIGHTS_ARRAY[9]
    }

    /// Create a new K-Vector without verification or phi dimensions (legacy compatibility)
    ///
    /// The k_v defaults to 0.0 (unverified), k_coherence to 0.5 (neutral).
    #[allow(clippy::too_many_arguments)]
    pub fn new_legacy(
        k_r: f32,
        k_a: f32,
        k_i: f32,
        k_p: f32,
        k_m: f32,
        k_s: f32,
        k_h: f32,
        k_topo: f32,
    ) -> Self {
        Self::new(k_r, k_a, k_i, k_p, k_m, k_s, k_h, k_topo, 0.0, 0.5)
    }

    /// Create a zero K-Vector (new/untrusted participant)
    pub fn zero() -> Self {
        Self {
            k_r: 0.0,
            k_a: 0.0,
            k_i: 0.0,
            k_p: 0.0,
            k_m: 0.0,
            k_s: 0.0,
            k_h: 0.0,
            k_topo: 0.0,
            k_v: 0.0,
            k_coherence: 0.0,
        }
    }

    /// Create a default K-Vector for new participants with baseline trust
    pub fn new_participant() -> Self {
        Self {
            k_r: 0.5,         // Neutral reputation
            k_a: 0.0,         // No activity yet
            k_i: 1.0,         // Assume integrity until proven otherwise
            k_p: 0.5,         // Unknown performance
            k_m: 0.0,         // Just joined
            k_s: 0.0,         // No stake yet
            k_h: 0.5,         // No history
            k_topo: 0.0,      // Not connected yet
            k_v: 0.0,         // Not verified yet
            k_coherence: 0.5, // Neutral coherence (not yet measured)
        }
    }

    /// Create a K-Vector for a verified participant
    ///
    /// Similar to new_participant but with verification level set.
    pub fn new_verified_participant(verification_level: f32) -> Self {
        Self {
            k_r: 0.5,
            k_a: 0.0,
            k_i: 1.0,
            k_p: 0.5,
            k_m: 0.0,
            k_s: 0.0,
            k_h: 0.5,
            k_topo: 0.0,
            k_v: verification_level.clamp(0.0, 1.0),
            k_coherence: 0.5,
        }
    }

    /// Calculate weighted trust score using default weights
    ///
    /// Formula: T = 0.25×k_r + 0.15×k_a + 0.20×k_i + 0.15×k_p + 0.05×k_m + 0.10×k_s + 0.05×k_h + 0.05×k_topo
    #[inline]
    pub fn trust_score(&self) -> f32 {
        // Use the optimized fast path with pre-computed weights array
        self.trust_score_fast()
    }

    /// Calculate weighted trust score with custom weights
    #[inline]
    pub fn trust_score_with_weights(&self, weights: &KVectorWeights) -> f32 {
        weights.w_r * self.k_r
            + weights.w_a * self.k_a
            + weights.w_i * self.k_i
            + weights.w_p * self.k_p
            + weights.w_m * self.k_m
            + weights.w_s * self.k_s
            + weights.w_h * self.k_h
            + weights.w_topo * self.k_topo
            + weights.w_v * self.k_v
            + weights.w_phi * self.k_coherence
    }

    /// Batch compute trust scores for multiple K-Vectors
    ///
    /// This is more efficient than computing individually due to:
    /// - Better cache locality
    /// - Reduced function call overhead
    /// - Potential for SIMD auto-vectorization
    #[inline]
    pub fn batch_trust_scores(vectors: &[KVector]) -> Vec<f32> {
        vectors.iter().map(|v| v.trust_score_fast()).collect()
    }

    /// Batch compute trust scores with custom weights
    pub fn batch_trust_scores_weighted(vectors: &[KVector], weights: &KVectorWeights) -> Vec<f32> {
        vectors
            .iter()
            .map(|v| v.trust_score_with_weights(weights))
            .collect()
    }

    /// Calculate reputation-squared weighted score (for RB-BFT voting)
    ///
    /// Used in Reputation-Based Byzantine Fault Tolerance where votes
    /// are weighted by reputation squared to amplify trusted voices.
    pub fn rbbft_voting_weight(&self) -> f32 {
        self.k_r.powi(2)
    }

    /// Check if K-Vector meets minimum thresholds for a governance tier
    pub fn meets_governance_threshold(&self, tier: GovernanceTier) -> bool {
        let score = self.trust_score();
        match tier {
            GovernanceTier::Observer => true,
            GovernanceTier::Basic => score >= 0.3,
            GovernanceTier::Major => score >= 0.4,
            GovernanceTier::Constitutional => score >= 0.6,
        }
    }

    /// Get the weakest dimension (for targeted improvement)
    pub fn weakest_dimension(&self) -> (KVectorDimension, f32) {
        let dims = [
            (KVectorDimension::Reputation, self.k_r),
            (KVectorDimension::Activity, self.k_a),
            (KVectorDimension::Integrity, self.k_i),
            (KVectorDimension::Performance, self.k_p),
            (KVectorDimension::Membership, self.k_m),
            (KVectorDimension::Stake, self.k_s),
            (KVectorDimension::Historical, self.k_h),
            (KVectorDimension::Topology, self.k_topo),
            (KVectorDimension::Verification, self.k_v),
            (KVectorDimension::Coherence, self.k_coherence),
        ];
        dims.into_iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((KVectorDimension::Reputation, 0.0))
    }

    /// Get the strongest dimension
    pub fn strongest_dimension(&self) -> (KVectorDimension, f32) {
        let dims = [
            (KVectorDimension::Reputation, self.k_r),
            (KVectorDimension::Activity, self.k_a),
            (KVectorDimension::Integrity, self.k_i),
            (KVectorDimension::Performance, self.k_p),
            (KVectorDimension::Membership, self.k_m),
            (KVectorDimension::Stake, self.k_s),
            (KVectorDimension::Historical, self.k_h),
            (KVectorDimension::Topology, self.k_topo),
            (KVectorDimension::Verification, self.k_v),
            (KVectorDimension::Coherence, self.k_coherence),
        ];
        dims.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((KVectorDimension::Reputation, 0.0))
    }

    /// Get the verification level
    pub fn verification_level(&self) -> f32 {
        self.k_v
    }

    /// Check if this K-Vector represents a verified identity
    ///
    /// Returns true if k_v >= 0.5 (basic credential verification or better)
    pub fn is_verified(&self) -> bool {
        self.k_v >= 0.5
    }

    /// Check if this K-Vector represents a strongly verified identity
    ///
    /// Returns true if k_v >= 0.7 (government ID, WebAuthn, etc.)
    pub fn is_strongly_verified(&self) -> bool {
        self.k_v >= 0.7
    }

    /// Decay K-Vector over time (for inactive participants)
    ///
    /// Activity decays fastest, reputation/historical decay slowly
    pub fn apply_decay(&mut self, decay_factor: f32) {
        let df = decay_factor.clamp(0.0, 1.0);
        self.k_a *= df; // Activity decays fully
        self.k_r *= 0.5 + 0.5 * df; // Reputation decays slowly
        self.k_h *= 0.5 + 0.5 * df; // Historical decays slowly
        self.k_topo *= df; // Topology decays fully
                           // Integrity, Performance, Membership, Stake don't decay (they're state-based)
    }

    /// Merge with another K-Vector (e.g., from different data sources)
    pub fn merge(&self, other: &KVector, self_weight: f32) -> KVector {
        let ow = 1.0 - self_weight;
        KVector::new(
            self.k_r * self_weight + other.k_r * ow,
            self.k_a * self_weight + other.k_a * ow,
            self.k_i * self_weight + other.k_i * ow,
            self.k_p * self_weight + other.k_p * ow,
            self.k_m * self_weight + other.k_m * ow,
            self.k_s * self_weight + other.k_s * ow,
            self.k_h * self_weight + other.k_h * ow,
            self.k_topo * self_weight + other.k_topo * ow,
            self.k_v * self_weight + other.k_v * ow,
            self.k_coherence * self_weight + other.k_coherence * ow,
        )
    }

    /// Update the verification dimension based on credential status
    ///
    /// This method is used by the Byzantine↔Identity bridge to update
    /// the verification score based on credential verification results.
    pub fn with_verification(mut self, k_v: f32) -> Self {
        self.k_v = k_v.clamp(0.0, 1.0);
        self
    }

    /// Update the coherence dimension based on Phi measurement
    ///
    /// This method is used by the coherence_integration module to update
    /// the coherence score based on integrated information (Φ) measurement.
    pub fn with_coherence(mut self, k_coherence: f32) -> Self {
        self.k_coherence = k_coherence.clamp(0.0, 1.0);
        self
    }

    /// Get the coherence level
    pub fn coherence_level(&self) -> f32 {
        self.k_coherence
    }

    /// Check if this agent has high coherence (Φ >= 0.7)
    pub fn is_highly_coherent(&self) -> bool {
        self.k_coherence >= 0.7
    }

    /// Calculate verification score from credential status
    ///
    /// Maps credential verification status to k_v value:
    /// - `has_valid_credential`: +0.3 if true
    /// - `credential_not_expired`: +0.2 if true
    /// - `signature_verified`: +0.2 if true
    /// - `has_attestations`: +0.15 if has any attestations
    /// - `has_zk_proof`: +0.15 if ZK verification available
    pub fn calculate_verification_score(
        has_valid_credential: bool,
        credential_not_expired: bool,
        signature_verified: bool,
        attestation_count: u32,
        has_zk_proof: bool,
    ) -> f32 {
        let mut score = 0.0;
        if has_valid_credential {
            score += 0.3;
        }
        if credential_not_expired {
            score += 0.2;
        }
        if signature_verified {
            score += 0.2;
        }
        if attestation_count > 0 {
            // Diminishing returns for attestations
            score += 0.15 * (1.0 - 1.0 / (1.0 + attestation_count as f32 * 0.5));
        }
        if has_zk_proof {
            score += 0.15;
        }
        score.clamp(0.0, 1.0)
    }
}

impl Default for KVector {
    fn default() -> Self {
        Self::new_participant()
    }
}

/// K-Vector dimension identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/matl/"))]
pub enum KVectorDimension {
    /// k_r: Historical track record and peer evaluations
    Reputation,
    /// k_a: Recent participation and contribution frequency
    Activity,
    /// k_i: Adherence to protocol rules
    Integrity,
    /// k_p: Quality of contributions
    Performance,
    /// k_m: Duration of active participation
    Membership,
    /// k_s: Economic commitment (staking)
    Stake,
    /// k_h: Long-term consistency and reliability
    Historical,
    /// k_topo: Network centrality and connectivity
    Topology,
    /// k_v: Identity verification and credential validity
    Verification,
    /// k_coherence: Integrated information / consciousness metric
    Coherence,
}

impl KVectorDimension {
    /// Get the short code (e.g., "k_r", "k_a")
    pub fn code(&self) -> &'static str {
        match self {
            Self::Reputation => "k_r",
            Self::Activity => "k_a",
            Self::Integrity => "k_i",
            Self::Performance => "k_p",
            Self::Membership => "k_m",
            Self::Stake => "k_s",
            Self::Historical => "k_h",
            Self::Topology => "k_topo",
            Self::Verification => "k_v",
            Self::Coherence => "k_coherence",
        }
    }

    /// Get the human-readable dimension name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Reputation => "Reputation",
            Self::Activity => "Activity",
            Self::Integrity => "Integrity",
            Self::Performance => "Performance",
            Self::Membership => "Membership",
            Self::Stake => "Stake",
            Self::Historical => "Historical",
            Self::Topology => "Topology",
            Self::Verification => "Verification",
            Self::Coherence => "Coherence",
        }
    }

    /// Get a description of what this dimension measures
    pub fn description(&self) -> &'static str {
        match self {
            Self::Reputation => "Historical track record and peer evaluations",
            Self::Activity => "Recent participation and contribution frequency",
            Self::Integrity => "Adherence to protocol rules, no detected violations",
            Self::Performance => "Quality of contributions (gradients, votes, etc.)",
            Self::Membership => "Duration of active participation in network",
            Self::Stake => "Economic commitment (MYC, ETH, USDC staking)",
            Self::Historical => "Long-term consistency and reliability",
            Self::Topology => "Network centrality and connectivity contribution",
            Self::Verification => "Identity verification and credential validity",
            Self::Coherence => "Output consistency and behavioral coherence",
        }
    }

    /// Get the default weight for this dimension in trust score calculation
    pub fn default_weight(&self) -> f32 {
        match self {
            Self::Reputation => KVECTOR_WEIGHTS.w_r,
            Self::Activity => KVECTOR_WEIGHTS.w_a,
            Self::Integrity => KVECTOR_WEIGHTS.w_i,
            Self::Performance => KVECTOR_WEIGHTS.w_p,
            Self::Membership => KVECTOR_WEIGHTS.w_m,
            Self::Stake => KVECTOR_WEIGHTS.w_s,
            Self::Historical => KVECTOR_WEIGHTS.w_h,
            Self::Topology => KVECTOR_WEIGHTS.w_topo,
            Self::Verification => KVECTOR_WEIGHTS.w_v,
            Self::Coherence => KVECTOR_WEIGHTS.w_phi,
        }
    }

    /// Get all K-Vector dimensions
    pub fn all() -> &'static [KVectorDimension] {
        &[
            Self::Reputation,
            Self::Activity,
            Self::Integrity,
            Self::Performance,
            Self::Membership,
            Self::Stake,
            Self::Historical,
            Self::Topology,
            Self::Verification,
            Self::Coherence,
        ]
    }
}

/// Governance tiers based on trust score (Φ thresholds)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/matl/"))]
pub enum GovernanceTier {
    /// Observer: Can view but not participate in governance
    Observer = 0,
    /// Basic: Can participate in basic proposals (Φ >= 0.3)
    Basic = 1,
    /// Major: Can participate in major decisions (Φ >= 0.4)
    Major = 2,
    /// Constitutional: Can propose constitutional changes (Φ >= 0.6)
    Constitutional = 3,
}

impl GovernanceTier {
    /// Minimum trust score required for this tier
    pub fn min_trust_score(&self) -> f32 {
        match self {
            Self::Observer => 0.0,
            Self::Basic => 0.3,
            Self::Major => 0.4,
            Self::Constitutional => 0.6,
        }
    }

    /// Get the tier for a given trust score
    pub fn from_trust_score(score: f32) -> Self {
        if score >= 0.6 {
            Self::Constitutional
        } else if score >= 0.4 {
            Self::Major
        } else if score >= 0.3 {
            Self::Basic
        } else {
            Self::Observer
        }
    }
}

/// A K-Vector with cached trust score for repeated access patterns
///
/// Use this when you need to access the trust score multiple times
/// without modifying the underlying K-Vector. The cache is invalidated
/// automatically when the inner K-Vector is accessed mutably.
///
/// # Performance
///
/// - First trust_score() call: O(9) multiplications + additions
/// - Subsequent calls: O(1) cache lookup
/// - Memory overhead: 8 bytes (`Option<f32>` + padding)
#[derive(Debug, Clone)]
pub struct CachedKVector {
    inner: KVector,
    cached_score: Cell<Option<f32>>,
}

impl CachedKVector {
    /// Create a new cached K-Vector
    pub fn new(kv: KVector) -> Self {
        Self {
            inner: kv,
            cached_score: Cell::new(None),
        }
    }

    /// Get the trust score (cached after first computation)
    #[inline]
    pub fn trust_score(&self) -> f32 {
        if let Some(score) = self.cached_score.get() {
            return score;
        }
        let score = self.inner.trust_score();
        self.cached_score.set(Some(score));
        score
    }

    /// Get the inner K-Vector (invalidates cache on mutable access)
    pub fn inner(&self) -> &KVector {
        &self.inner
    }

    /// Get mutable access to inner K-Vector (invalidates cache)
    pub fn inner_mut(&mut self) -> &mut KVector {
        self.cached_score.set(None);
        &mut self.inner
    }

    /// Invalidate the cache (call after external modifications)
    pub fn invalidate_cache(&self) {
        self.cached_score.set(None);
    }

    /// Check if cache is populated
    pub fn is_cached(&self) -> bool {
        self.cached_score.get().is_some()
    }
}

impl From<KVector> for CachedKVector {
    fn from(kv: KVector) -> Self {
        Self::new(kv)
    }
}

/// Batch operations for K-Vectors optimized for large collections
pub struct KVectorBatch;

impl KVectorBatch {
    /// Compute trust scores for a batch of K-Vectors
    /// Pre-allocates result vector for efficiency
    pub fn compute_trust_scores(vectors: &[KVector]) -> Vec<f32> {
        let mut results = Vec::with_capacity(vectors.len());
        for v in vectors {
            results.push(v.trust_score_fast());
        }
        results
    }

    /// Filter K-Vectors by trust threshold
    /// Returns indices of vectors meeting the threshold
    pub fn filter_by_threshold(vectors: &[KVector], threshold: f32) -> Vec<usize> {
        vectors
            .iter()
            .enumerate()
            .filter_map(|(i, v)| {
                if v.trust_score_fast() >= threshold {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Compute weighted average of multiple K-Vectors
    /// Weights are normalized automatically
    pub fn weighted_average(vectors: &[KVector], weights: &[f32]) -> Option<KVector> {
        if vectors.is_empty() || vectors.len() != weights.len() {
            return None;
        }

        let total_weight: f32 = weights.iter().sum();
        if total_weight <= 0.0 {
            return None;
        }

        // Accumulate weighted sums
        let mut sums = [0.0f32; 10];
        for (v, w) in vectors.iter().zip(weights.iter()) {
            let arr = v.to_array();
            let normalized_w = w / total_weight;
            for (i, val) in arr.iter().enumerate() {
                sums[i] += val * normalized_w;
            }
        }

        Some(KVector::from_array(sums))
    }

    /// Find the K-Vector with highest trust score
    pub fn max_trust(vectors: &[KVector]) -> Option<(usize, f32)> {
        vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.trust_score_fast()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Find the K-Vector with lowest trust score
    pub fn min_trust(vectors: &[KVector]) -> Option<(usize, f32)> {
        vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.trust_score_fast()))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trust_score_calculation() {
        // With 10 dimensions:
        // 0.24*0.8 + 0.11*0.6 + 0.19*0.9 + 0.11*0.7 + 0.05*0.3 + 0.07*0.5 + 0.05*0.6 + 0.04*0.4 + 0.09*0.8 + 0.05*0.7
        let kv = KVector::new(0.8, 0.6, 0.9, 0.7, 0.3, 0.5, 0.6, 0.4, 0.8, 0.7);
        let score = kv.trust_score();
        // Score should be around 0.68 with new weights
        assert!(score > 0.6 && score < 0.8);
    }

    #[test]
    fn test_weights_sum_to_one() {
        assert!(KVECTOR_WEIGHTS.is_valid());
    }

    #[test]
    fn test_governance_tiers() {
        let kv = KVector::new(0.8, 0.6, 0.9, 0.7, 0.3, 0.5, 0.6, 0.4, 0.8, 0.7);
        assert!(kv.meets_governance_threshold(GovernanceTier::Constitutional));

        let low_kv = KVector::new(0.2, 0.1, 0.3, 0.2, 0.1, 0.0, 0.1, 0.0, 0.0, 0.1);
        assert!(!low_kv.meets_governance_threshold(GovernanceTier::Basic));
    }

    #[test]
    fn test_weakest_dimension() {
        // k_m = 0.1 is the weakest (Membership)
        let kv = KVector::new(0.8, 0.6, 0.9, 0.7, 0.1, 0.5, 0.6, 0.4, 0.2, 0.5);
        let (dim, val) = kv.weakest_dimension();
        assert_eq!(dim, KVectorDimension::Membership);
        assert!((val - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_rbbft_voting_weight() {
        let kv = KVector::new(0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!((kv.rbbft_voting_weight() - 0.81).abs() < 0.001);
    }

    #[test]
    fn test_decay() {
        let mut kv = KVector::new(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8);
        kv.apply_decay(0.5);
        assert!((kv.k_a - 0.4).abs() < 0.001); // Activity decays fully
        assert!((kv.k_r - 0.6).abs() < 0.001); // Reputation decays slowly
        assert!((kv.k_i - 0.8).abs() < 0.001); // Integrity doesn't decay
        assert!((kv.k_v - 0.8).abs() < 0.001); // Verification doesn't decay (it's credential-based)
        assert!((kv.k_coherence - 0.8).abs() < 0.001); // Coherence doesn't decay (it's state-based)
    }

    #[test]
    fn test_verification_dimension() {
        let unverified = KVector::new_participant();
        assert_eq!(unverified.k_v, 0.0);
        assert!(!unverified.is_verified());
        assert!(!unverified.is_strongly_verified());

        let verified = KVector::new_verified_participant(0.7);
        assert_eq!(verified.k_v, 0.7);
        assert!(verified.is_verified());
        assert!(verified.is_strongly_verified());

        let basic_verified = KVector::new_verified_participant(0.5);
        assert!(basic_verified.is_verified());
        assert!(!basic_verified.is_strongly_verified());
    }

    #[test]
    fn test_coherence_dimension() {
        let neutral = KVector::new_participant();
        assert_eq!(neutral.k_coherence, 0.5); // Neutral coherence for new participants
        assert!(!neutral.is_highly_coherent());

        let coherent = KVector::new_participant().with_coherence(0.8);
        assert!((coherent.k_coherence - 0.8).abs() < 0.001);
        assert!(coherent.is_highly_coherent());
    }

    #[test]
    fn test_calculate_verification_score() {
        // Full verification
        let score = KVector::calculate_verification_score(true, true, true, 5, true);
        assert!(score > 0.9);

        // No verification
        let score = KVector::calculate_verification_score(false, false, false, 0, false);
        assert!((score - 0.0).abs() < 0.001);

        // Partial verification (valid credential, not expired, no signature)
        let score = KVector::calculate_verification_score(true, true, false, 0, false);
        assert!((score - 0.5).abs() < 0.01);

        // Attestations provide diminishing returns
        let score1 = KVector::calculate_verification_score(false, false, false, 1, false);
        let score10 = KVector::calculate_verification_score(false, false, false, 10, false);
        assert!(score10 > score1);
        assert!(score10 < 0.15); // Capped by attestation contribution
    }

    #[test]
    fn test_with_verification() {
        let kv = KVector::new_participant().with_verification(0.75);
        assert!((kv.k_v - 0.75).abs() < 0.001);
        assert!(kv.is_strongly_verified());
    }

    #[test]
    fn test_legacy_constructor() {
        let kv = KVector::new_legacy(0.8, 0.6, 0.9, 0.7, 0.3, 0.5, 0.6, 0.4);
        assert_eq!(kv.k_v, 0.0); // Default to unverified
        assert_eq!(kv.k_coherence, 0.5); // Default to neutral coherence
    }

    #[test]
    fn test_9d_constructor() {
        let kv = KVector::new_9d(0.8, 0.6, 0.9, 0.7, 0.3, 0.5, 0.6, 0.4, 0.8);
        assert_eq!(kv.k_v, 0.8);
        assert_eq!(kv.k_coherence, 0.5); // Default to neutral coherence
    }

    #[test]
    fn test_dimension_all() {
        let dims = KVectorDimension::all();
        assert_eq!(dims.len(), 10);
        assert!(dims.contains(&KVectorDimension::Verification));
        assert!(dims.contains(&KVectorDimension::Coherence));
    }

    // =========================================================================
    // Performance optimization tests
    // =========================================================================

    #[test]
    fn test_to_array_from_array() {
        let kv = KVector::new(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.15);
        let arr = kv.to_array();
        let kv2 = KVector::from_array(arr);
        assert_eq!(kv, kv2);
    }

    #[test]
    fn test_trust_score_fast_matches_regular() {
        let kv = KVector::new(0.8, 0.6, 0.9, 0.7, 0.3, 0.5, 0.6, 0.4, 0.8, 0.7);
        let regular = kv.trust_score_with_weights(&KVECTOR_WEIGHTS);
        let fast = kv.trust_score_fast();
        assert!((regular - fast).abs() < 0.001);
    }

    #[test]
    fn test_batch_trust_scores() {
        let vectors = vec![
            KVector::new(0.8, 0.6, 0.9, 0.7, 0.3, 0.5, 0.6, 0.4, 0.8, 0.7),
            KVector::new(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
            KVector::new(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        ];
        let scores = KVector::batch_trust_scores(&vectors);
        assert_eq!(scores.len(), 3);
        assert!((scores[0] - vectors[0].trust_score()).abs() < 0.001);
        assert!((scores[1] - vectors[1].trust_score()).abs() < 0.001);
        assert!((scores[2] - vectors[2].trust_score()).abs() < 0.001);
    }

    #[test]
    fn test_cached_kvector() {
        let kv = KVector::new(0.8, 0.6, 0.9, 0.7, 0.3, 0.5, 0.6, 0.4, 0.8, 0.7);
        let cached = CachedKVector::new(kv);

        // Not cached yet
        assert!(!cached.is_cached());

        // First access computes and caches
        let score1 = cached.trust_score();
        assert!(cached.is_cached());

        // Second access uses cache
        let score2 = cached.trust_score();
        assert!((score1 - score2).abs() < 0.001);
    }

    #[test]
    fn test_cached_kvector_invalidation() {
        let kv = KVector::new(0.8, 0.6, 0.9, 0.7, 0.3, 0.5, 0.6, 0.4, 0.8, 0.7);
        let mut cached = CachedKVector::new(kv);

        let _ = cached.trust_score(); // Populate cache
        assert!(cached.is_cached());

        // Mutable access should invalidate
        let _ = cached.inner_mut();
        assert!(!cached.is_cached());
    }

    #[test]
    fn test_kvector_batch_operations() {
        let vectors = vec![
            KVector::new(0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9),
            KVector::new(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
            KVector::new(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        ];

        // Test filter
        let high_trust = KVectorBatch::filter_by_threshold(&vectors, 0.5);
        assert_eq!(high_trust.len(), 2); // First two meet threshold

        // Test max
        let (max_idx, max_score) = KVectorBatch::max_trust(&vectors).unwrap();
        assert_eq!(max_idx, 0);
        assert!(max_score > 0.8);

        // Test min
        let (min_idx, min_score) = KVectorBatch::min_trust(&vectors).unwrap();
        assert_eq!(min_idx, 2);
        assert!(min_score < 0.2);
    }

    #[test]
    fn test_kvector_weighted_average() {
        let vectors = vec![
            KVector::new(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            KVector::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ];
        let weights = vec![1.0, 1.0];

        let avg = KVectorBatch::weighted_average(&vectors, &weights).unwrap();
        assert!((avg.k_r - 0.5).abs() < 0.001);
        assert!((avg.k_v - 0.5).abs() < 0.001);
    }
}
