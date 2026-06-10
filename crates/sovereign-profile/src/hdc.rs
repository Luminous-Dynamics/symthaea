// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! HDC (Hyperdimensional Computing) encoding for the 8D Sovereign Profile.
//!
//! Encodes the 8 civic dimensions as a 16,384-bit `BinaryHV` — a holographic
//! vector where each dimension contributes proportionally to its weight.
//! Civic tier derivation uses `popcount(V_profile AND V_mask) >= k` — a single
//! O(1) operation that respects multiple pathways to citizenship.
//!
//! ## Privacy
//!
//! The BinaryHV never leaves the device. ZKP circuits prove
//! `popcount(V AND M) >= k` without revealing V. The network knows you pass
//! the threshold but not which dimensions are strong.
//!
//! ## Encoding Strategy
//!
//! Each dimension [0.0, 1.0] is threshold-encoded into 10 sub-bands:
//! a value of 0.7 activates 7 of 10 sub-band vectors. These are bound to
//! a dimension-specific basis vector and weighted-bundled with community
//! `DimensionWeights` to produce the final profile HV.

use crate::weights::DimensionWeights;
use crate::{CivicTier, SovereignDimension, SovereignProfile};

// ============================================================================
// Constants
// ============================================================================

/// Number of bits in each hypervector.
pub const HDC_DIMENSION: usize = 16_384;

/// Number of bytes in each hypervector.
pub const HDC_BYTES: usize = 2048;

/// Number of sub-bands per dimension for threshold encoding.
/// A dimension value of 0.7 activates 7 of 10 sub-bands.
const SUB_BANDS: usize = 10;

/// Seed base for dimension basis vectors. Dimension i uses seed BASE + i.
const DIMENSION_SEED_BASE: u64 = 0x5356_5249_474E_0000; // "SVRIGN\0\0"

/// Seed base for sub-band vectors within a dimension.
/// Sub-band j of dimension i uses seed SUB_BAND_BASE + i * 100 + j.
const SUB_BAND_SEED_BASE: u64 = 0x5342_414E_4400_0000; // "SBAND\0\0\0"

/// Seed for tier threshold masks. Tier t uses TIER_SEED_BASE + t.
const TIER_SEED_BASE: u64 = 0x5449_4552_0000_0000; // "TIER\0\0\0\0"

// ============================================================================
// BinaryHV — adapted from symthaea-hdc-crypto (BLAKE3 XOF, 16,384-bit)
// ============================================================================

/// 16,384-bit binary hypervector (2KB).
///
/// Bipolar encoding: bit=1 → +1, bit=0 → -1.
/// 32-byte aligned for SIMD compatibility.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(align(32))]
pub struct BinaryHV(
    #[cfg_attr(feature = "serde", serde(with = "serde_bytes_2048"))] pub [u8; HDC_BYTES],
);

impl BinaryHV {
    pub const DIM: usize = HDC_DIMENSION;
    pub const BYTES: usize = HDC_BYTES;

    /// All-zero vector.
    pub const fn zero() -> Self {
        Self([0u8; HDC_BYTES])
    }

    /// Deterministic random vector from a seed (BLAKE3 XOF).
    pub fn random(seed: u64) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&seed.to_le_bytes());
        let mut bytes = [0u8; HDC_BYTES];
        let mut reader = hasher.finalize_xof();
        reader.fill(&mut bytes);
        Self(bytes)
    }

    /// XOR binding.
    #[inline]
    pub fn bind(&self, other: &Self) -> Self {
        let mut result = [0u8; HDC_BYTES];
        for i in 0..HDC_BYTES {
            result[i] = self.0[i] ^ other.0[i];
        }
        Self(result)
    }

    /// AND intersection.
    #[inline]
    pub fn intersection(&self, other: &Self) -> Self {
        let mut result = [0u8; HDC_BYTES];
        for i in 0..HDC_BYTES {
            result[i] = self.0[i] & other.0[i];
        }
        Self(result)
    }

    /// Majority-vote bundle.
    pub fn bundle(vectors: &[Self]) -> Self {
        if vectors.is_empty() {
            return Self::zero();
        }
        let threshold = vectors.len() as i32 / 2;
        let mut result = [0u8; HDC_BYTES];
        for byte_idx in 0..HDC_BYTES {
            let mut byte_result = 0u8;
            for bit in 0..8u8 {
                let mut count = 0i32;
                let mask = 1u8 << bit;
                for v in vectors {
                    if v.0[byte_idx] & mask != 0 {
                        count += 1;
                    }
                }
                if count > threshold {
                    byte_result |= mask;
                }
            }
            result[byte_idx] = byte_result;
        }
        Self(result)
    }

    /// Weighted majority-vote bundle.
    ///
    /// Each vector is weighted by `weights[i]`. A bit is set if the
    /// sum of weights for vectors with that bit set exceeds half the
    /// total weight.
    pub fn weighted_bundle(vectors: &[Self], weights: &[f64]) -> Self {
        assert_eq!(vectors.len(), weights.len());
        if vectors.is_empty() {
            return Self::zero();
        }
        let total_weight: f64 = weights.iter().sum();
        let threshold = total_weight / 2.0;
        let mut result = [0u8; HDC_BYTES];
        for byte_idx in 0..HDC_BYTES {
            let mut byte_result = 0u8;
            for bit in 0..8u8 {
                let mask = 1u8 << bit;
                let mut weighted_sum = 0.0_f64;
                for (i, v) in vectors.iter().enumerate() {
                    if v.0[byte_idx] & mask != 0 {
                        weighted_sum += weights[i];
                    }
                }
                if weighted_sum > threshold {
                    byte_result |= mask;
                }
            }
            result[byte_idx] = byte_result;
        }
        Self(result)
    }

    /// Hamming similarity [0.0, 1.0].
    #[inline]
    pub fn similarity(&self, other: &Self) -> f32 {
        let matching: u32 = self
            .0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| (!(a ^ b)).count_ones())
            .sum();
        matching as f32 / Self::DIM as f32
    }

    /// Population count (number of 1-bits).
    #[inline]
    pub fn popcount(&self) -> u32 {
        self.0.iter().map(|byte| byte.count_ones()).sum()
    }

    /// Density: fraction of 1-bits [0.0, 1.0].
    #[inline]
    pub fn density(&self) -> f32 {
        self.popcount() as f32 / Self::DIM as f32
    }
}

impl std::fmt::Debug for BinaryHV {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BinaryHV(density={:.3}, pop={})",
            self.density(),
            self.popcount()
        )
    }
}

impl Default for BinaryHV {
    fn default() -> Self {
        Self::zero()
    }
}

// ============================================================================
// Profile encoding
// ============================================================================

/// Encode a SovereignProfile as a 16,384-bit BinaryHV.
///
/// ## Encoding strategy: Proportional Bit Activation
///
/// Each of the 16,384 bits is "owned" by one of the 8 dimensions.
/// Bits are allocated proportionally to the dimension weights.
/// For each dimension's bit region, the fraction of bits set to 1
/// equals the dimension's value (0.0 → no bits, 1.0 → all bits).
///
/// This guarantees:
/// - Monotonicity: higher values → more bits → higher popcount
/// - Determinism: same profile + weights → same HV
/// - Weight respect: dimensions with higher weight own more bits
/// - Popcount ∝ combined_score (linear relationship)
pub fn encode_profile(profile: &SovereignProfile, weights: &DimensionWeights) -> BinaryHV {
    let dims = profile.as_array();
    let mut result = [0u8; HDC_BYTES];

    // Allocate bits to dimensions proportional to weights
    let total_bits = HDC_DIMENSION;
    let mut bit_offset = 0;

    for (i, (&value, &weight)) in dims.iter().zip(weights.weights.iter()).enumerate() {
        let value = if value.is_finite() {
            value.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let dim_bits = (weight * total_bits as f64).round() as usize;
        let active_bits = (value * dim_bits as f64).round() as usize;

        // Use a deterministic pattern: the first `active_bits` of this
        // dimension's region are set. We permute using the dimension seed
        // so the bit pattern is not contiguous (spread across the HV).
        let basis = BinaryHV::random(DIMENSION_SEED_BASE + i as u64);

        // Set bits where the basis vector has 1s, up to `active_bits` count
        let mut set_count = 0;
        for bit_pos in 0..dim_bits {
            if set_count >= active_bits {
                break;
            }
            let global_bit = bit_offset + bit_pos;
            if global_bit >= total_bits {
                break;
            }
            // Use basis vector to determine which bits in this region to activate
            let basis_byte = global_bit / 8;
            let basis_bit = global_bit % 8;
            if basis.0[basis_byte] & (1u8 << basis_bit) != 0 {
                let byte_idx = global_bit / 8;
                let bit_idx = global_bit % 8;
                result[byte_idx] |= 1u8 << bit_idx;
                set_count += 1;
            }
        }

        // If we didn't get enough bits from basis-matching, fill remaining
        // by setting bits where basis is 0
        if set_count < active_bits {
            for bit_pos in 0..dim_bits {
                if set_count >= active_bits {
                    break;
                }
                let global_bit = bit_offset + bit_pos;
                if global_bit >= total_bits {
                    break;
                }
                let basis_byte = global_bit / 8;
                let basis_bit = global_bit % 8;
                if basis.0[basis_byte] & (1u8 << basis_bit) == 0 {
                    let byte_idx = global_bit / 8;
                    let bit_idx = global_bit % 8;
                    if result[byte_idx] & (1u8 << bit_idx) == 0 {
                        result[byte_idx] |= 1u8 << bit_idx;
                        set_count += 1;
                    }
                }
            }
        }

        bit_offset += dim_bits;
    }

    BinaryHV(result)
}

// ============================================================================
// Tier threshold masks
// ============================================================================

/// Popcount-based tier thresholds.
///
/// Instead of comparing against per-tier masks, we use a simpler and more
/// robust approach: the profile's own popcount (bit density) correlates
/// with its combined score. Higher-valued profiles activate more sub-bands
/// → more bits set → higher popcount.
///
/// Thresholds are calibrated by encoding reference profiles at each tier
/// boundary and recording their popcount.
pub struct TierThresholds {
    /// Minimum popcount for each tier [Observer, Participant, Citizen, Steward, Guardian].
    pub min_popcount: [u32; 5],
}

impl TierThresholds {
    /// Calibrate popcount thresholds by encoding reference profiles.
    ///
    /// For each tier, encodes a uniform profile at the tier's minimum score
    /// and uses its popcount as the threshold.
    pub fn calibrate(weights: &DimensionWeights) -> Self {
        let tier_scores = [0.0, 0.3, 0.4, 0.6, 0.8];
        let mut min_popcount = [0u32; 5];

        for (idx, &score) in tier_scores.iter().enumerate() {
            if score <= 0.0 {
                min_popcount[idx] = 0;
            } else {
                let ref_profile = SovereignProfile::from_array([score; 8]);
                let hv = encode_profile(&ref_profile, weights);
                min_popcount[idx] = hv.popcount();
            }
        }

        Self { min_popcount }
    }

    /// Get the minimum popcount for a specific tier.
    pub fn threshold_for(&self, tier: CivicTier) -> u32 {
        match tier {
            CivicTier::Observer => self.min_popcount[0],
            CivicTier::Participant => self.min_popcount[1],
            CivicTier::Citizen => self.min_popcount[2],
            CivicTier::Steward => self.min_popcount[3],
            CivicTier::Guardian => self.min_popcount[4],
        }
    }
}

/// Derive civic tier from a profile's popcount.
///
/// Returns the highest tier whose popcount threshold is met by the
/// profile HV's popcount. This is O(1) — a single popcount operation
/// compared against 4 thresholds.
pub fn tier_from_popcount(profile_hv: &BinaryHV, thresholds: &TierThresholds) -> CivicTier {
    let pop = profile_hv.popcount();

    if pop >= thresholds.min_popcount[4] {
        CivicTier::Guardian
    } else if pop >= thresholds.min_popcount[3] {
        CivicTier::Steward
    } else if pop >= thresholds.min_popcount[2] {
        CivicTier::Citizen
    } else if pop >= thresholds.min_popcount[1] {
        CivicTier::Participant
    } else {
        CivicTier::Observer
    }
}

// ============================================================================
// Serde support for [u8; 2048]
// ============================================================================

#[cfg(feature = "serde")]
mod serde_bytes_2048 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(data: &[u8; 2048], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        data[..].serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 2048], D::Error>
    where
        D: Deserializer<'de>,
    {
        let slice: Vec<u8> = Deserialize::deserialize(deserializer)?;
        if slice.len() != 2048 {
            return Err(serde::de::Error::custom(format!(
                "Expected 2048 bytes, got {}",
                slice.len()
            )));
        }
        let mut array = [0u8; 2048];
        array.copy_from_slice(&slice);
        Ok(array)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_hv_random_deterministic() {
        let a = BinaryHV::random(42);
        let b = BinaryHV::random(42);
        assert_eq!(a, b);
    }

    #[test]
    fn binary_hv_random_balanced_density() {
        let v = BinaryHV::random(42);
        let d = v.density();
        assert!(
            d > 0.45 && d < 0.55,
            "Random density should be ~0.5, got {d}"
        );
    }

    #[test]
    fn binary_hv_similarity_self_is_one() {
        let a = BinaryHV::random(42);
        assert_eq!(a.similarity(&a), 1.0);
    }

    #[test]
    fn binary_hv_similarity_random_is_half() {
        let a = BinaryHV::random(1);
        let b = BinaryHV::random(2);
        let sim = a.similarity(&b);
        assert!(
            (sim - 0.5).abs() < 0.05,
            "Random similarity should be ~0.5, got {sim}"
        );
    }

    #[test]
    fn binary_hv_intersection_popcount() {
        let a = BinaryHV::random(1);
        let b = BinaryHV::random(2);
        let inter = a.intersection(&b);
        // ~25% of bits should be 1 (both random ~50%)
        let density = inter.density();
        assert!(
            density > 0.2 && density < 0.3,
            "AND density should be ~0.25, got {density}"
        );
    }

    #[test]
    fn binary_hv_weighted_bundle_respects_weights() {
        let a = BinaryHV::random(1);
        let b = BinaryHV::random(2);
        // Weight a at 0.9, b at 0.1 — result should be very similar to a
        let result = BinaryHV::weighted_bundle(&[a, b], &[0.9, 0.1]);
        assert!(
            result.similarity(&a) > 0.8,
            "Heavily weighted vector should dominate"
        );
        assert!(
            result.similarity(&b) < 0.6,
            "Lightly weighted vector should not dominate"
        );
    }

    // --- Profile encoding ---

    #[test]
    fn encode_zero_profile_is_zero_ish() {
        let profile = SovereignProfile::zero();
        let weights = DimensionWeights::equal();
        let hv = encode_profile(&profile, &weights);
        // All dimensions at 0 → zero sub-bands → zero HV
        assert_eq!(hv.popcount(), 0);
    }

    #[test]
    fn encode_full_profile_has_nonzero_density() {
        let profile = SovereignProfile::from_array([1.0; 8]);
        let weights = DimensionWeights::equal();
        let hv = encode_profile(&profile, &weights);
        assert!(
            hv.popcount() > 0,
            "Full profile should have nonzero popcount, got {}",
            hv.popcount()
        );
        // Weighted bundle of 8 equally-weighted dimensions → density ~0.5
        assert!(
            hv.density() > 0.1,
            "Full profile density should be > 0.1, got {}",
            hv.density()
        );
    }

    #[test]
    fn encode_is_deterministic() {
        let profile = SovereignProfile::from_array([0.5; 8]);
        let weights = DimensionWeights::governance();
        let hv1 = encode_profile(&profile, &weights);
        let hv2 = encode_profile(&profile, &weights);
        assert_eq!(hv1, hv2);
    }

    #[test]
    fn similar_profiles_have_high_similarity() {
        let weights = DimensionWeights::equal();
        let p1 = SovereignProfile::from_array([0.7; 8]);
        let p2 = SovereignProfile::from_array([0.8; 8]);
        let hv1 = encode_profile(&p1, &weights);
        let hv2 = encode_profile(&p2, &weights);
        let sim = hv1.similarity(&hv2);
        assert!(
            sim > 0.6,
            "Similar profiles should have high similarity, got {sim}"
        );
    }

    #[test]
    fn different_profiles_have_lower_similarity() {
        let weights = DimensionWeights::equal();
        let p1 = SovereignProfile::from_array([0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1]);
        let p2 = SovereignProfile::from_array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9]);
        let hv1 = encode_profile(&p1, &weights);
        let hv2 = encode_profile(&p2, &weights);
        let sim = hv1.similarity(&hv2);
        assert!(
            sim < 0.7,
            "Different profiles should have lower similarity, got {sim}"
        );
    }

    // --- Tier from popcount ---

    #[test]
    fn tier_thresholds_are_deterministic() {
        let weights = DimensionWeights::governance();
        let t1 = TierThresholds::calibrate(&weights);
        let t2 = TierThresholds::calibrate(&weights);
        assert_eq!(t1.min_popcount, t2.min_popcount);
    }

    #[test]
    fn full_profile_achieves_guardian_tier() {
        let weights = DimensionWeights::governance();
        let thresholds = TierThresholds::calibrate(&weights);
        let profile = SovereignProfile::from_array([1.0; 8]);
        let hv = encode_profile(&profile, &weights);
        let tier = tier_from_popcount(&hv, &thresholds);
        assert_eq!(
            tier,
            CivicTier::Guardian,
            "Full profile should be Guardian, popcount={}",
            hv.popcount()
        );
    }

    #[test]
    fn zero_profile_is_observer() {
        let weights = DimensionWeights::governance();
        let thresholds = TierThresholds::calibrate(&weights);
        let profile = SovereignProfile::zero();
        let hv = encode_profile(&profile, &weights);
        let tier = tier_from_popcount(&hv, &thresholds);
        assert_eq!(tier, CivicTier::Observer, "Zero profile should be Observer");
    }

    #[test]
    fn popcount_tier_is_monotonic() {
        let weights = DimensionWeights::governance();
        let thresholds = TierThresholds::calibrate(&weights);
        let levels = [0.0, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0];

        let mut prev_tier = CivicTier::Observer;
        for &level in &levels {
            let profile = SovereignProfile::from_array([level; 8]);
            let hv = encode_profile(&profile, &weights);
            let tier = tier_from_popcount(&hv, &thresholds);
            assert!(tier >= prev_tier,
                "Tier should be monotonically increasing: level={level}, pop={}, got {tier:?} < {prev_tier:?}",
                hv.popcount());
            prev_tier = tier;
        }
    }

    #[test]
    fn tier_thresholds_are_ordered() {
        let weights = DimensionWeights::governance();
        let thresholds = TierThresholds::calibrate(&weights);
        for i in 1..5 {
            assert!(
                thresholds.min_popcount[i] >= thresholds.min_popcount[i - 1],
                "Tier thresholds should be non-decreasing: [{}]={} < [{}]={}",
                i - 1,
                thresholds.min_popcount[i - 1],
                i,
                thresholds.min_popcount[i]
            );
        }
    }

    #[test]
    fn popcount_increases_with_dimension_values() {
        let weights = DimensionWeights::equal();
        let low = encode_profile(&SovereignProfile::from_array([0.2; 8]), &weights);
        let mid = encode_profile(&SovereignProfile::from_array([0.5; 8]), &weights);
        let high = encode_profile(&SovereignProfile::from_array([0.9; 8]), &weights);
        assert!(
            mid.popcount() > low.popcount(),
            "Mid popcount {} should exceed low {}",
            mid.popcount(),
            low.popcount()
        );
        assert!(
            high.popcount() > mid.popcount(),
            "High popcount {} should exceed mid {}",
            high.popcount(),
            mid.popcount()
        );
    }
}
