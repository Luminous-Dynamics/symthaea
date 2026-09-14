// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # Continuous Hypervectors and Unitary Roles
//!
//! `ContinuousHV` stores arbitrary continuous-valued distributed state.
//! `UnitaryRole` represents the narrower real Hadamard-unitary role algebra used
//! when an HLS experiment requires reversible, norm-preserving association.
//!
//! Generic `ContinuousHV::bind()` is retained for compatibility. Multiplying two
//! arbitrary continuous hypervectors is not generally unitary and therefore must
//! not be assumed to preserve norm or support exact unbinding.

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

/// Standard HDC dimension (2^14 = 16,384).
pub const HDC_DIMENSION: usize = 16_384;
const RNG_SEED_XOR: u64 = 0x9E37_79B9_7F4A_7C15;
const RNG_ZERO_ESCAPE: u64 = 0xD1B5_4A32_D192_ED03;
const ROLE_SEED_DOMAIN: u64 = 0xA076_1D64_78BD_642F;
const SPLITMIX_MUL_1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_MUL_2: u64 = 0x94D0_49BB_1331_11EB;

#[inline]
fn seeded_xorshift_state(seed: u64) -> u64 {
    let state = seed ^ RNG_SEED_XOR;
    if state == 0 { RNG_ZERO_ESCAPE } else { state }
}

/// Nonlinear deterministic seed finalizer for unitary-role codewords.
///
/// Xorshift expansion is linear over GF(2). Feeding structured integer seeds
/// directly into it therefore leaks seed-XOR relations into bipolar role
/// multiplication: pairs with the same seed XOR can produce identical composed
/// roles. This SplitMix64-style finalizer breaks that algebraic relation before
/// xorshift expands the state. Continuous-valued random vectors intentionally
/// keep their historical generator unchanged.
#[inline]
fn mixed_role_seed(seed: u64) -> u64 {
    let mut value = (seed ^ ROLE_SEED_DOMAIN).wrapping_add(RNG_SEED_XOR);
    value = (value ^ (value >> 30)).wrapping_mul(SPLITMIX_MUL_1);
    value = (value ^ (value >> 27)).wrapping_mul(SPLITMIX_MUL_2);
    value ^ (value >> 31)
}

#[inline]
fn seeded_role_xorshift_state(seed: u64) -> u64 {
    // Unlike the legacy continuous-vector generator, the role path must accept
    // the full u64 state domain without remapping zero to a second valid state.
    // `mixed_role_seed` is bijective, so preserving its zero output avoids a
    // deterministic two-seed alias before role expansion begins.
    mixed_role_seed(seed)
}

/// A real Hadamard-unitary HDC role vector.
///
/// Every component is exactly `-1.0` or `+1.0`. Consequently, applying a role
/// to a continuous hypervector is an orthogonal diagonal transformation:
///
/// ```text
/// B_r(x) = r ⊙ x
/// ||B_r(x)||₂ = ||x||₂
/// <B_r(x), B_r(y)> = <x, y>
/// B_r(B_r(x)) = x
/// ```
///
/// This type exists so reversible role binding is an explicit invariant rather
/// than an accidental property of a particular random vector initialization.
#[derive(Debug, Clone, PartialEq)]
pub struct UnitaryRole {
    values: Vec<f32>,
}

impl UnitaryRole {
    /// Deterministically generate a bipolar role from `seed`.
    ///
    /// The seed is nonlinearly pre-mixed before expansion so simple arithmetic
    /// relations among caller seeds do not become exact algebraic relations among
    /// composed bipolar roles. Each component first advances the full 64-bit
    /// state by an odd Weyl increment and then applies the bijective xorshift
    /// transition. This makes zero a valid initial role state instead of an
    /// absorbing or remapped special case.
    pub fn new(dim: usize, seed: u64) -> Self {
        let mut values = Vec::with_capacity(dim);
        let mut state = seeded_role_xorshift_state(seed);

        for _ in 0..dim {
            // Addition by an odd Weyl increment and xorshift are both bijections
            // on u64. Distinct internal role states therefore remain distinct at
            // every transition, including when the mixed initial state is zero.
            state = state.wrapping_add(RNG_SEED_XOR);
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            values.push(if state & (1u64 << 63) == 0 { -1.0 } else { 1.0 });
        }

        Self { values }
    }

    /// Construct a role from explicit values, rejecting any non-unitary entry.
    pub fn try_from_values(values: Vec<f32>) -> Result<Self, String> {
        if let Some((index, value)) = values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| *value != -1.0 && *value != 1.0)
        {
            return Err(format!(
                "unitary role component at index {index} must be -1 or +1, got {value}"
            ));
        }
        Ok(Self { values })
    }

    /// Number of role components.
    #[inline]
    pub fn dim(&self) -> usize {
        self.values.len()
    }

    /// Read-only role components.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// Apply this role to a continuous value/state hypervector.
    #[inline]
    pub fn bind(&self, value: &ContinuousHV) -> ContinuousHV {
        assert_eq!(self.dim(), value.dim(), "Dimension mismatch");
        ContinuousHV::from_values(
            self.values
                .iter()
                .zip(value.values.iter())
                .map(|(role, value)| role * value)
                .collect(),
        )
    }

    /// Recover a value bound by this same role.
    ///
    /// Real bipolar Hadamard roles are self-inverse, so unbinding is identical
    /// to binding and is exact apart from ordinary IEEE signed-zero behavior.
    #[inline]
    pub fn unbind(&self, bound: &ContinuousHV) -> ContinuousHV {
        self.bind(bound)
    }

    /// Compose two unitary roles. Closure is exact because ±1 × ±1 = ±1.
    #[inline]
    pub fn compose(&self, other: &Self) -> Self {
        assert_eq!(self.dim(), other.dim(), "Dimension mismatch");
        Self {
            values: self
                .values
                .iter()
                .zip(other.values.iter())
                .map(|(a, b)| a * b)
                .collect(),
        }
    }

    /// Cyclically permute a role while preserving unitarity.
    pub fn permute(&self, positions: usize) -> Self {
        let dim = self.dim();
        if dim == 0 {
            return self.clone();
        }
        let shift = positions % dim;
        if shift == 0 {
            return self.clone();
        }
        let mut values = vec![1.0; dim];
        for i in 0..dim {
            values[(i + shift) % dim] = self.values[i];
        }
        Self { values }
    }
}

impl Serialize for UnitaryRole {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.values.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for UnitaryRole {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let values = Vec::<f32>::deserialize(deserializer)?;
        Self::try_from_values(values).map_err(de::Error::custom)
    }
}

/// A continuous-valued hypervector using f32 components.
///
/// Each component typically ranges in [-1, 1], though operations may produce
/// values outside this range before normalization.
///
/// # Memory
///
/// 64 KB per vector at the default 16,384 dimensions (16,384 × 4 bytes).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContinuousHV {
    /// Vector components.
    pub values: Vec<f32>,
}

impl ContinuousHV {
    /// Create a zero-initialized hypervector of the given dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            values: vec![0.0; dim],
        }
    }

    /// Create a deterministic random hypervector with values in [-1, 1].
    pub fn new_random(dim: usize, seed: u64) -> Self {
        let mut values = Vec::with_capacity(dim);
        let mut state = seeded_xorshift_state(seed);

        for _ in 0..dim {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let normalized = ((state >> 40) as f32) * (2.0 / (1u64 << 24) as f32) - 1.0;
            values.push(normalized);
        }

        Self { values }
    }

    /// Create from an existing vector of values.
    pub fn from_values(values: Vec<f32>) -> Self {
        Self { values }
    }

    /// Create from a slice.
    pub fn from_slice(slice: &[f32]) -> Self {
        Self {
            values: slice.to_vec(),
        }
    }

    /// Get the dimension (number of components).
    #[inline]
    pub fn dim(&self) -> usize {
        self.values.len()
    }

    /// Generic elementwise continuous binding.
    ///
    /// This operation is useful as a learned/modulatory Hadamard product, but it
    /// is **not generally unitary** when both operands have arbitrary continuous
    /// magnitudes. Use [`UnitaryRole::bind`] for reversible HLS role binding.
    #[inline]
    pub fn bind(&self, other: &Self) -> Self {
        assert_eq!(self.values.len(), other.values.len(), "Dimension mismatch");
        let values = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .collect();
        Self { values }
    }

    /// Bundling operation (elementwise average).
    ///
    /// Creates a superposition similar to all inputs. Returns a zero vector of
    /// `HDC_DIMENSION` if the slice is empty.
    #[inline]
    pub fn bundle(hvs: &[&Self]) -> Self {
        if hvs.is_empty() {
            return Self::new(HDC_DIMENSION);
        }
        let dim = hvs[0].values.len();
        assert!(hvs.iter().all(|hv| hv.dim() == dim), "Dimension mismatch");
        let inv_n = 1.0 / hvs.len() as f32;
        let mut values = vec![0.0f32; dim];
        for hv in hvs {
            for (acc, &value) in values.iter_mut().zip(hv.values.iter()) {
                *acc += value;
            }
        }
        for value in &mut values {
            *value *= inv_n;
        }
        Self { values }
    }

    /// Cosine similarity in [-1, 1].
    #[inline]
    pub fn similarity(&self, other: &Self) -> f32 {
        assert_eq!(self.values.len(), other.values.len(), "Dimension mismatch");
        let mut dot = 0.0f32;
        let mut norm_a_sq = 0.0f32;
        let mut norm_b_sq = 0.0f32;

        for i in 0..self.values.len() {
            let a = self.values[i];
            let b = other.values[i];
            dot += a * b;
            norm_a_sq += a * a;
            norm_b_sq += b * b;
        }

        let denom = (norm_a_sq * norm_b_sq).sqrt();
        if denom < 1e-10 {
            return 0.0;
        }
        (dot / denom).clamp(-1.0, 1.0)
    }

    /// Cyclic right shift by `positions` elements.
    pub fn permute(&self, positions: usize) -> Self {
        let dim = self.values.len();
        if dim == 0 {
            return self.clone();
        }
        let shift = positions % dim;
        if shift == 0 {
            return self.clone();
        }
        let mut values = vec![0.0f32; dim];
        for i in 0..dim {
            values[(i + shift) % dim] = self.values[i];
        }
        Self { values }
    }

    /// L2 norm.
    #[inline]
    pub fn norm(&self) -> f32 {
        self.values.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize to unit length. Returns a clone if the norm is near zero.
    #[inline]
    pub fn normalize(&self) -> Self {
        let norm = self.norm();
        if norm < 1e-10 {
            return self.clone();
        }
        Self {
            values: self.values.iter().map(|x| x / norm).collect(),
        }
    }

    /// Scale by a constant factor.
    #[inline]
    pub fn scale(&self, factor: f32) -> Self {
        Self {
            values: self.values.iter().map(|x| x * factor).collect(),
        }
    }

    /// In-place scale.
    #[inline]
    pub fn scale_in_place(&mut self, factor: f32) {
        for value in &mut self.values {
            *value *= factor;
        }
    }

    /// Add `other * scale` to self in-place.
    #[inline]
    pub fn add_scaled(&mut self, other: &Self, scale: f32) {
        debug_assert_eq!(self.values.len(), other.values.len());
        for (value, &other_value) in self.values.iter_mut().zip(other.values.iter()) {
            *value += other_value * scale;
        }
    }

    /// In-place linear interpolation: `self = (1 - alpha) * self + alpha * target`.
    #[inline]
    pub fn lerp_in_place(&mut self, target: &Self, alpha: f32) {
        debug_assert_eq!(self.values.len(), target.values.len());
        let one_minus = 1.0 - alpha;
        for (value, &target_value) in self.values.iter_mut().zip(target.values.iter()) {
            *value = one_minus * *value + alpha * target_value;
        }
    }

    /// Elementwise addition.
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.values.len(), other.values.len(), "Dimension mismatch");
        Self {
            values: self
                .values
                .iter()
                .zip(other.values.iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    /// Elementwise subtraction.
    #[inline]
    pub fn subtract(&self, other: &Self) -> Self {
        assert_eq!(self.values.len(), other.values.len(), "Dimension mismatch");
        Self {
            values: self
                .values
                .iter()
                .zip(other.values.iter())
                .map(|(a, b)| a - b)
                .collect(),
        }
    }

    /// Raw dot product.
    #[inline]
    pub fn dot(&self, other: &Self) -> f32 {
        assert_eq!(self.values.len(), other.values.len(), "Dimension mismatch");
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Generate approximately orthogonal unit hypervectors via modified
    /// Gram-Schmidt. These are continuous basis vectors, not unitary roles.
    pub fn orthogonal_set(dim: usize, count: usize, seed: u64) -> Vec<Self> {
        if count == 0 {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let mut vector = Self::new_random(dim, seed.wrapping_add(i as u64 * 7919));
            for previous in &result {
                let projection = vector.dot(previous);
                for (value, &basis) in vector.values.iter_mut().zip(previous.values.iter()) {
                    *value -= projection * basis;
                }
            }
            let norm = vector.norm();
            if norm > 1e-10 {
                for value in &mut vector.values {
                    *value /= norm;
                }
            }
            result.push(vector);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuous_random_is_deterministic() {
        assert_eq!(
            ContinuousHV::new_random(1024, 42),
            ContinuousHV::new_random(1024, 42)
        );
    }

    #[test]
    fn role_stream_accepts_full_seed_domain_and_vector_escape_remains_deterministic() {
        let mixed_zero_seed = 0xC1BE_9B22_F808_E7C4;
        assert_eq!(mixed_role_seed(mixed_zero_seed), 0);
        let role = UnitaryRole::new(256, mixed_zero_seed);
        assert!(role.as_slice().iter().any(|value| *value == -1.0));
        assert!(role.as_slice().iter().any(|value| *value == 1.0));

        let pathological_vector_seed = RNG_SEED_XOR;
        let vector = ContinuousHV::new_random(256, pathological_vector_seed);
        assert!(vector.values.iter().any(|value| *value != -1.0));
        assert_eq!(
            vector,
            ContinuousHV::new_random(256, pathological_vector_seed)
        );
    }

    #[test]
    fn generic_continuous_bind_remains_available() {
        let a = ContinuousHV::new_random(1024, 1);
        let b = ContinuousHV::new_random(1024, 2);
        let bound = a.bind(&b);
        assert!(bound.similarity(&a).abs() < 0.15);
        assert!(bound.similarity(&b).abs() < 0.15);
    }

    #[test]
    fn bundle_is_similar_to_members() {
        let a = ContinuousHV::new_random(1024, 10);
        let b = ContinuousHV::new_random(1024, 20);
        let bundled = ContinuousHV::bundle(&[&a, &b]);
        assert!(bundled.similarity(&a) > 0.4);
        assert!(bundled.similarity(&b) > 0.4);
    }

    #[test]
    fn permutation_roundtrip_at_full_dimension() {
        let hv = ContinuousHV::new_random(128, 5);
        assert_eq!(hv, hv.permute(128));
    }

    #[test]
    fn normalization_produces_unit_norm() {
        let normed = ContinuousHV::new_random(256, 7).normalize();
        assert!((normed.norm() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn orthogonal_set_is_low_interference() {
        let set = ContinuousHV::orthogonal_set(1024, 5, 42);
        assert_eq!(set.len(), 5);
        for vector in &set {
            assert!((vector.norm() - 1.0).abs() < 1e-4);
        }
        for i in 0..set.len() {
            for j in (i + 1)..set.len() {
                assert!(set[i].dot(&set[j]).abs() < 0.05);
            }
        }
    }

    #[test]
    fn unitary_role_components_are_exactly_bipolar() {
        let role = UnitaryRole::new(4096, 42);
        assert!(role.as_slice().iter().all(|value| *value == -1.0 || *value == 1.0));
    }

    #[test]
    fn unitary_binding_preserves_norm() {
        let role = UnitaryRole::new(4096, 42);
        let value = ContinuousHV::new_random(4096, 99);
        let bound = role.bind(&value);
        assert_eq!(bound.norm(), value.norm());
    }

    #[test]
    fn unitary_binding_roundtrip_is_exact() {
        let role = UnitaryRole::new(4096, 42);
        let value = ContinuousHV::new_random(4096, 99);
        let recovered = role.unbind(&role.bind(&value));
        assert_eq!(recovered, value);
    }

    #[test]
    fn same_role_preserves_dot_product_and_similarity() {
        let role = UnitaryRole::new(4096, 42);
        let a = ContinuousHV::new_random(4096, 100);
        let b = ContinuousHV::new_random(4096, 101);
        let bound_a = role.bind(&a);
        let bound_b = role.bind(&b);
        assert_eq!(bound_a.dot(&bound_b), a.dot(&b));
        assert!((bound_a.similarity(&bound_b) - a.similarity(&b)).abs() < 1e-7);
    }

    #[test]
    fn unitary_role_seed_mixing_breaks_equal_xor_composition_collision() {
        let dim = 4096;
        assert_eq!(30_u64 ^ 40_u64, 31_u64 ^ 41_u64);
        let left = UnitaryRole::new(dim, 30).compose(&UnitaryRole::new(dim, 40));
        let right = UnitaryRole::new(dim, 31).compose(&UnitaryRole::new(dim, 41));
        assert_ne!(left, right);
        let normalized_dot = left
            .as_slice()
            .iter()
            .zip(right.as_slice())
            .map(|(&a, &b)| (a * b) as f64)
            .sum::<f64>()
            / dim as f64;
        assert!(
            normalized_dot.abs() < 0.1,
            "equal seed-XOR pairs remained strongly coupled: {normalized_dot}"
        );
    }

    #[test]
    fn unitary_role_stream_breaks_previous_zero_escape_alias() {
        let dim = 4096;
        let mixed_zero_seed = 0xC1BE_9B22_F808_E7C4;
        let mixed_escape_seed = 0xD8C6_4A40_75AB_3494;
        assert_eq!(mixed_role_seed(mixed_zero_seed), 0);
        assert_eq!(mixed_role_seed(mixed_escape_seed), RNG_ZERO_ESCAPE);

        let zero_role = UnitaryRole::new(dim, mixed_zero_seed);
        let escape_role = UnitaryRole::new(dim, mixed_escape_seed);
        assert_ne!(zero_role, escape_role);

        let normalized_dot = zero_role
            .as_slice()
            .iter()
            .zip(escape_role.as_slice())
            .map(|(&a, &b)| (a * b) as f64)
            .sum::<f64>()
            / dim as f64;
        assert!(
            normalized_dot.abs() < 0.1,
            "previous zero/escape alias remained strongly coupled: {normalized_dot}"
        );
    }

    #[test]
    fn structured_unitary_role_codebooks_have_no_exact_composition_collisions() {
        let dim = 4096;
        let keys = (0..8)
            .map(|index| UnitaryRole::new(dim, 200 + index))
            .collect::<Vec<_>>();
        let values = (0..4)
            .map(|index| UnitaryRole::new(dim, 400 + index))
            .collect::<Vec<_>>();
        let mut compositions = Vec::with_capacity(keys.len() * values.len());
        for key in &keys {
            for value in &values {
                compositions.push(key.compose(value));
            }
        }
        for left in 0..compositions.len() {
            for right in (left + 1)..compositions.len() {
                assert_ne!(
                    compositions[left], compositions[right],
                    "structured codebook produced exact composed-role collision at {left} and {right}"
                );
            }
        }
    }

    #[test]
    fn unitary_roles_are_closed_under_composition_and_permutation() {
        let a = UnitaryRole::new(1024, 1);
        let b = UnitaryRole::new(1024, 2);
        let composed = a.compose(&b);
        let permuted = composed.permute(17);
        assert!(permuted.as_slice().iter().all(|value| *value == -1.0 || *value == 1.0));
    }

    #[test]
    fn unitary_role_serialization_preserves_invariant() {
        let role = UnitaryRole::new(64, 42);
        let json = serde_json::to_string(&role).unwrap();
        let restored: UnitaryRole = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, role);
        assert!(serde_json::from_str::<UnitaryRole>("[1.0,0.5,-1.0]").is_err());
    }

    #[test]
    fn continuous_serde_roundtrip() {
        let hv = ContinuousHV::new_random(64, 123);
        let json = serde_json::to_string(&hv).unwrap();
        let restored: ContinuousHV = serde_json::from_str(&json).unwrap();
        assert_eq!(hv, restored);
    }
}
