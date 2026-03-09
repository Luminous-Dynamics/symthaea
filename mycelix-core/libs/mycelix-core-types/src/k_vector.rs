//! K-Vector: 8-Dimensional Trust Representation
//!
//! The K-Vector captures trust across 8 orthogonal dimensions, enabling
//! nuanced reputation assessment in the Mycelix network.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// 8-dimensional trust vector
///
/// Each dimension captures a different aspect of an agent's trustworthiness:
/// - `k_r`: Reputation from historical behavior
/// - `k_a`: Activity level and engagement
/// - `k_i`: Integrity (consistency between claims and actions)
/// - `k_p`: Performance in assigned tasks
/// - `k_m`: Membership duration
/// - `k_s`: Stake weight (economic commitment)
/// - `k_h`: Historical consistency over time
/// - `k_topo`: Network topology contribution (connectivity)
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KVector {
    /// Reputation score (0.0-1.0)
    /// Weight: 0.25
    pub k_r: f32,

    /// Activity score (0.0-1.0)
    /// Weight: 0.15
    pub k_a: f32,

    /// Integrity score (0.0-1.0)
    /// Weight: 0.20
    pub k_i: f32,

    /// Performance score (0.0-1.0)
    /// Weight: 0.15
    pub k_p: f32,

    /// Membership duration score (0.0-1.0)
    /// Weight: 0.05
    pub k_m: f32,

    /// Stake weight score (0.0-1.0)
    /// Weight: 0.10
    pub k_s: f32,

    /// Historical consistency score (0.0-1.0)
    /// Weight: 0.05
    pub k_h: f32,

    /// Network topology contribution (0.0-1.0)
    /// Weight: 0.05
    pub k_topo: f32,
}

/// Weights for each K-Vector dimension
pub const K_VECTOR_WEIGHTS: [f32; 8] = [
    0.25, // k_r - Reputation
    0.15, // k_a - Activity
    0.20, // k_i - Integrity
    0.15, // k_p - Performance
    0.05, // k_m - Membership
    0.10, // k_s - Stake
    0.05, // k_h - History
    0.05, // k_topo - Topology
];

impl KVector {
    /// Create a new K-Vector with all dimensions set to 0.5 (neutral)
    pub fn neutral() -> Self {
        Self {
            k_r: 0.5,
            k_a: 0.5,
            k_i: 0.5,
            k_p: 0.5,
            k_m: 0.5,
            k_s: 0.5,
            k_h: 0.5,
            k_topo: 0.5,
        }
    }

    /// Create a new K-Vector with all dimensions set to 0.0
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
        }
    }

    /// Create a new K-Vector from an array of 8 values
    pub fn from_array(values: [f32; 8]) -> Self {
        Self {
            k_r: values[0],
            k_a: values[1],
            k_i: values[2],
            k_p: values[3],
            k_m: values[4],
            k_s: values[5],
            k_h: values[6],
            k_topo: values[7],
        }
    }

    /// Convert to array representation
    pub fn to_array(&self) -> [f32; 8] {
        [
            self.k_r,
            self.k_a,
            self.k_i,
            self.k_p,
            self.k_m,
            self.k_s,
            self.k_h,
            self.k_topo,
        ]
    }

    /// Calculate the weighted trust score (scalar projection)
    ///
    /// Formula: T = 0.25×k_r + 0.15×k_a + 0.20×k_i + 0.15×k_p +
    ///              0.05×k_m + 0.10×k_s + 0.05×k_h + 0.05×k_topo
    pub fn trust_score(&self) -> f32 {
        K_VECTOR_WEIGHTS[0] * self.k_r
            + K_VECTOR_WEIGHTS[1] * self.k_a
            + K_VECTOR_WEIGHTS[2] * self.k_i
            + K_VECTOR_WEIGHTS[3] * self.k_p
            + K_VECTOR_WEIGHTS[4] * self.k_m
            + K_VECTOR_WEIGHTS[5] * self.k_s
            + K_VECTOR_WEIGHTS[6] * self.k_h
            + K_VECTOR_WEIGHTS[7] * self.k_topo
    }

    /// Calculate the L2 norm (magnitude) of the K-Vector
    pub fn magnitude(&self) -> f32 {
        let sum_sq = self.k_r.powi(2)
            + self.k_a.powi(2)
            + self.k_i.powi(2)
            + self.k_p.powi(2)
            + self.k_m.powi(2)
            + self.k_s.powi(2)
            + self.k_h.powi(2)
            + self.k_topo.powi(2);
        sum_sq.sqrt()
    }

    /// Calculate cosine similarity with another K-Vector
    pub fn cosine_similarity(&self, other: &KVector) -> f32 {
        let dot = self.dot(other);
        let mag_self = self.magnitude();
        let mag_other = other.magnitude();

        if mag_self == 0.0 || mag_other == 0.0 {
            return 0.0;
        }

        dot / (mag_self * mag_other)
    }

    /// Calculate dot product with another K-Vector
    pub fn dot(&self, other: &KVector) -> f32 {
        self.k_r * other.k_r
            + self.k_a * other.k_a
            + self.k_i * other.k_i
            + self.k_p * other.k_p
            + self.k_m * other.k_m
            + self.k_s * other.k_s
            + self.k_h * other.k_h
            + self.k_topo * other.k_topo
    }

    /// Validate that all components are in the valid range [0.0, 1.0]
    pub fn is_valid(&self) -> bool {
        let in_range = |v: f32| (0.0..=1.0).contains(&v) && v.is_finite();

        in_range(self.k_r)
            && in_range(self.k_a)
            && in_range(self.k_i)
            && in_range(self.k_p)
            && in_range(self.k_m)
            && in_range(self.k_s)
            && in_range(self.k_h)
            && in_range(self.k_topo)
    }

    /// Clamp all values to [0.0, 1.0] range
    pub fn clamp(&self) -> Self {
        Self {
            k_r: self.k_r.clamp(0.0, 1.0),
            k_a: self.k_a.clamp(0.0, 1.0),
            k_i: self.k_i.clamp(0.0, 1.0),
            k_p: self.k_p.clamp(0.0, 1.0),
            k_m: self.k_m.clamp(0.0, 1.0),
            k_s: self.k_s.clamp(0.0, 1.0),
            k_h: self.k_h.clamp(0.0, 1.0),
            k_topo: self.k_topo.clamp(0.0, 1.0),
        }
    }

    /// Apply exponential decay to all components
    ///
    /// Used for temporal decay of trust scores.
    /// decay_rate should be in (0, 1), where lower values mean faster decay.
    pub fn decay(&self, decay_rate: f32) -> Self {
        Self {
            k_r: self.k_r * decay_rate,
            k_a: self.k_a * decay_rate,
            k_i: self.k_i * decay_rate,
            k_p: self.k_p * decay_rate,
            k_m: self.k_m * decay_rate,
            k_s: self.k_s * decay_rate,
            k_h: self.k_h * decay_rate,
            k_topo: self.k_topo * decay_rate,
        }
    }

    /// Merge with another K-Vector using exponential moving average
    ///
    /// alpha controls the weight of the new value (0.0 = keep old, 1.0 = use new)
    pub fn ema_merge(&self, other: &KVector, alpha: f32) -> Self {
        let blend = |old: f32, new: f32| old * (1.0 - alpha) + new * alpha;

        Self {
            k_r: blend(self.k_r, other.k_r),
            k_a: blend(self.k_a, other.k_a),
            k_i: blend(self.k_i, other.k_i),
            k_p: blend(self.k_p, other.k_p),
            k_m: blend(self.k_m, other.k_m),
            k_s: blend(self.k_s, other.k_s),
            k_h: blend(self.k_h, other.k_h),
            k_topo: blend(self.k_topo, other.k_topo),
        }
    }
}

impl Default for KVector {
    fn default() -> Self {
        Self::neutral()
    }
}

/// Builder for constructing K-Vectors incrementally
#[derive(Debug, Default)]
pub struct KVectorBuilder {
    k_r: Option<f32>,
    k_a: Option<f32>,
    k_i: Option<f32>,
    k_p: Option<f32>,
    k_m: Option<f32>,
    k_s: Option<f32>,
    k_h: Option<f32>,
    k_topo: Option<f32>,
}

impl KVectorBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reputation(mut self, value: f32) -> Self {
        self.k_r = Some(value);
        self
    }

    pub fn activity(mut self, value: f32) -> Self {
        self.k_a = Some(value);
        self
    }

    pub fn integrity(mut self, value: f32) -> Self {
        self.k_i = Some(value);
        self
    }

    pub fn performance(mut self, value: f32) -> Self {
        self.k_p = Some(value);
        self
    }

    pub fn membership(mut self, value: f32) -> Self {
        self.k_m = Some(value);
        self
    }

    pub fn stake(mut self, value: f32) -> Self {
        self.k_s = Some(value);
        self
    }

    pub fn history(mut self, value: f32) -> Self {
        self.k_h = Some(value);
        self
    }

    pub fn topology(mut self, value: f32) -> Self {
        self.k_topo = Some(value);
        self
    }

    /// Build the K-Vector, using 0.5 (neutral) for unset values
    pub fn build(self) -> KVector {
        KVector {
            k_r: self.k_r.unwrap_or(0.5),
            k_a: self.k_a.unwrap_or(0.5),
            k_i: self.k_i.unwrap_or(0.5),
            k_p: self.k_p.unwrap_or(0.5),
            k_m: self.k_m.unwrap_or(0.5),
            k_s: self.k_s.unwrap_or(0.5),
            k_h: self.k_h.unwrap_or(0.5),
            k_topo: self.k_topo.unwrap_or(0.5),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neutral_trust_score() {
        let k = KVector::neutral();
        assert!((k.trust_score() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_zero_trust_score() {
        let k = KVector::zero();
        assert!((k.trust_score() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_perfect_trust_score() {
        let k = KVector {
            k_r: 1.0,
            k_a: 1.0,
            k_i: 1.0,
            k_p: 1.0,
            k_m: 1.0,
            k_s: 1.0,
            k_h: 1.0,
            k_topo: 1.0,
        };
        assert!((k.trust_score() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_validity() {
        let valid = KVector::neutral();
        assert!(valid.is_valid());

        let invalid = KVector {
            k_r: 1.5,
            ..KVector::neutral()
        };
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_builder() {
        let k = KVectorBuilder::new().reputation(0.8).integrity(0.9).build();

        assert!((k.k_r - 0.8).abs() < 0.001);
        assert!((k.k_i - 0.9).abs() < 0.001);
        assert!((k.k_a - 0.5).abs() < 0.001); // Default
    }

    #[test]
    fn test_cosine_similarity() {
        let a = KVector::neutral();
        let b = KVector::neutral();
        assert!((a.cosine_similarity(&b) - 1.0).abs() < 0.001);

        let c = KVector::zero();
        assert!((a.cosine_similarity(&c) - 0.0).abs() < 0.001);
    }
}
