// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic Genesis-derived encoders for ordered continuous values.
//!
//! This module owns only the reusable HDC mechanism for converting an already
//! validated normalized scalar in `[0, 1]` into a similarity-preserving
//! [`ContinuousHV`]. Domain semantics such as physical units, normalization
//! ranges, saturation policy, uncertainty, and sensor identity remain with the
//! caller.
//!
//! The linear level codebook intentionally does **not** reproduce the historical
//! cumulative-bundle heuristic used by several domain encoders. Under cosine
//! similarity that heuristic compresses resolution toward the high end of the
//! range. Instead, this module uses the established HDC linear-mapping pattern:
//! start from one deterministic bipolar basis vector and progressively flip a
//! non-reused deterministic subset of components. Equal level separations then
//! produce approximately equal cosine separations throughout the range.

use crate::{genesis::GenesisSeed, hdc::ContinuousHV};
use rand::RngCore;
use std::{error::Error, fmt};

/// Minimum supported number of ordered quantization levels.
pub const MIN_LEVEL_COUNT: usize = 2;
/// Maximum supported number of ordered quantization levels.
pub const MAX_LEVEL_COUNT: usize = 256;
/// Maximum supported hypervector dimension for this codebook.
///
/// The bound keeps construction memory and permutation work explicit. It is
/// above Symthaea's current 64K HDC tier while still allowing research-scale
/// custom dimensions.
pub const MAX_LEVEL_DIMENSION: usize = 1_048_576;

const MAX_NAMESPACE_LEN: usize = 160;

/// Construction or input error for [`LinearLevelCodebook`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelCodebookError {
    /// The level count is outside the supported inclusive range.
    InvalidLevelCount {
        /// Requested level count.
        requested: usize,
    },
    /// Hypervector dimension must be greater than zero.
    ZeroDimension,
    /// Hypervector dimension exceeds [`MAX_LEVEL_DIMENSION`].
    DimensionTooLarge {
        /// Requested dimension.
        requested: usize,
    },
    /// The dimension cannot provide at least one new flipped component per
    /// adjacent level while keeping the endpoints approximately orthogonal.
    InsufficientDimension,
    /// The Genesis namespace is empty, too long, or contains unsupported bytes.
    InvalidNamespace,
    /// The supplied normalized value is NaN or infinite.
    NonFiniteInput,
    /// The supplied value is finite but outside `[0, 1]`.
    NormalizedInputOutOfRange,
}

impl fmt::Display for LevelCodebookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLevelCount { requested } => write!(
                f,
                "level count {requested} is outside {MIN_LEVEL_COUNT}..={MAX_LEVEL_COUNT}"
            ),
            Self::ZeroDimension => write!(f, "HDC dimension must be greater than zero"),
            Self::DimensionTooLarge { requested } => write!(
                f,
                "HDC dimension {requested} exceeds maximum {MAX_LEVEL_DIMENSION}"
            ),
            Self::InsufficientDimension => write!(
                f,
                "HDC dimension is too small to separate all requested levels"
            ),
            Self::InvalidNamespace => write!(f, "Genesis namespace is invalid"),
            Self::NonFiniteInput => write!(f, "normalized input is NaN or infinite"),
            Self::NormalizedInputOutOfRange => {
                write!(f, "normalized input must lie in the inclusive range [0, 1]")
            }
        }
    }
}

impl Error for LevelCodebookError {}

/// Result of encoding one normalized ordered value.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedLinearLevel {
    /// Similarity-preserving HDC representation.
    pub representation: ContinuousHV,
    /// Quantized level selected under the codebook's frozen rule.
    pub selected_level: usize,
    /// Number of basis components flipped relative to level zero.
    pub flipped_components: usize,
}

/// Genesis-derived linear level codebook for normalized ordered values.
///
/// The construction follows the standard HDC level-mapping idea: level zero is
/// a deterministic bipolar basis vector, and later levels progressively flip
/// non-reused components according to one deterministic permutation. The last
/// level flips half the dimensions, making the endpoints approximately
/// orthogonal under cosine similarity.
///
/// For levels `a` and `b`, expected similarity is governed by the number of
/// components flipped between them, not by their absolute position on the line.
/// This avoids the high-end resolution compression of cumulative `0..=k`
/// bundling.
///
/// Quantization uses nearest-grid mapping with an explicit positive tie rule:
///
/// `k = floor(x * (level_count - 1) + 0.5)`.
///
/// The codebook stores O(D) state: one unit-norm bipolar basis HV and one
/// permutation of component indices. It does not store one full HV per level.
#[derive(Debug, Clone)]
pub struct LinearLevelCodebook {
    namespace: String,
    dimension: usize,
    level_count: usize,
    base: ContinuousHV,
    flip_order: Vec<usize>,
}

impl LinearLevelCodebook {
    /// Construct a deterministic linear codebook from a Genesis seed.
    ///
    /// Public encoding semantics use two domain labels:
    ///
    /// - `"<namespace>::linear-level::base-v1"` for the bipolar basis signs;
    /// - `"<namespace>::linear-level::flip-order-v1"` for the Fisher-Yates
    ///   permutation stream.
    ///
    /// Construction fails before large allocation if the namespace, level
    /// count, or dimension is invalid.
    pub fn from_genesis(
        genesis: &GenesisSeed,
        namespace: &str,
        level_count: usize,
        dimension: usize,
    ) -> Result<Self, LevelCodebookError> {
        validate_namespace(namespace)?;

        if !(MIN_LEVEL_COUNT..=MAX_LEVEL_COUNT).contains(&level_count) {
            return Err(LevelCodebookError::InvalidLevelCount {
                requested: level_count,
            });
        }
        if dimension == 0 {
            return Err(LevelCodebookError::ZeroDimension);
        }
        if dimension > MAX_LEVEL_DIMENSION {
            return Err(LevelCodebookError::DimensionTooLarge {
                requested: dimension,
            });
        }

        let transition_count = level_count - 1;
        if dimension / 2 < transition_count {
            return Err(LevelCodebookError::InsufficientDimension);
        }

        let base_label = format!("{namespace}::linear-level::base-v1");
        let raw_base = ContinuousHV::from_genesis(genesis, &base_label, dimension);
        let component_magnitude = 1.0f32 / (dimension as f32).sqrt();
        let base = ContinuousHV::from_values(
            raw_base
                .values
                .into_iter()
                .map(|value| {
                    if value < 0.0 {
                        -component_magnitude
                    } else {
                        component_magnitude
                    }
                })
                .collect(),
        );

        let mut flip_order: Vec<usize> = (0..dimension).collect();
        let permutation_label = format!("{namespace}::linear-level::flip-order-v1");
        let mut rng = genesis.domain(&permutation_label);
        fisher_yates(&mut flip_order, &mut rng);

        Ok(Self {
            namespace: namespace.to_owned(),
            dimension,
            level_count,
            base,
            flip_order,
        })
    }

    /// Stable Genesis namespace used to derive this codebook.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Hypervector dimension produced by this codebook.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Number of ordered quantization levels.
    pub fn level_count(&self) -> usize {
        self.level_count
    }

    /// Return the level selected by the frozen nearest-grid quantization rule.
    ///
    /// This function never clamps invalid caller input. A domain that wants
    /// saturation must make that decision explicitly before calling the shared
    /// mechanism and retain its own saturation status.
    pub fn selected_level(&self, normalized: f32) -> Result<usize, LevelCodebookError> {
        validate_normalized(normalized)?;

        let scaled = normalized * (self.level_count - 1) as f32;
        Ok(((scaled + 0.5).floor() as usize).min(self.level_count - 1))
    }

    /// Number of non-reused basis components flipped at one discrete level.
    pub fn flipped_components_for_level(&self, level: usize) -> usize {
        debug_assert!(level < self.level_count);
        let total_flips = self.dimension / 2;
        level * total_flips / (self.level_count - 1)
    }

    /// Encode one already-normalized ordered scalar.
    ///
    /// The hot path allocates one output HV by cloning the unit-norm basis and
    /// then flips only the prefix of the deterministic permutation required by
    /// the selected level. No per-level codebook HV allocation occurs.
    pub fn encode_normalized(
        &self,
        normalized: f32,
    ) -> Result<EncodedLinearLevel, LevelCodebookError> {
        let selected_level = self.selected_level(normalized)?;
        let flipped_components = self.flipped_components_for_level(selected_level);
        let mut representation = self.base.clone();

        for &index in &self.flip_order[..flipped_components] {
            representation.values[index] = -representation.values[index];
        }

        Ok(EncodedLinearLevel {
            representation,
            selected_level,
            flipped_components,
        })
    }
}

fn validate_namespace(namespace: &str) -> Result<(), LevelCodebookError> {
    let allowed = namespace.bytes().all(|byte| {
        byte.is_ascii_alphanumeric()
            || matches!(byte, b'-' | b'_' | b'.' | b':' | b'/' | b'@')
    });

    if namespace.is_empty() || namespace.len() > MAX_NAMESPACE_LEN || !allowed {
        Err(LevelCodebookError::InvalidNamespace)
    } else {
        Ok(())
    }
}

fn validate_normalized(value: f32) -> Result<(), LevelCodebookError> {
    if !value.is_finite() {
        return Err(LevelCodebookError::NonFiniteInput);
    }
    if !(0.0..=1.0).contains(&value) {
        return Err(LevelCodebookError::NormalizedInputOutOfRange);
    }
    Ok(())
}

/// Language-neutral Fisher-Yates permutation driven directly by a Genesis
/// SHAKE stream. `sample_below` uses rejection sampling rather than relying on
/// `rand`'s distribution/shuffle implementation, so the permutation algorithm
/// is independently reproducible across implementations.
fn fisher_yates<R: RngCore>(values: &mut [usize], rng: &mut R) {
    for i in (1..values.len()).rev() {
        let j = sample_below(rng, i + 1);
        values.swap(i, j);
    }
}

fn sample_below<R: RngCore>(rng: &mut R, upper: usize) -> usize {
    debug_assert!(upper > 0);
    let range = 1u128 << 64;
    let upper = upper as u128;
    let limit = range - (range % upper);

    loop {
        let value = rng.next_u64() as u128;
        if value < limit {
            return (value % upper) as usize;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn codebook(namespace: &str, levels: usize, dimension: usize) -> LinearLevelCodebook {
        LinearLevelCodebook::from_genesis(
            &GenesisSeed::from_phrase("hdc-enc-001-test-genesis"),
            namespace,
            levels,
            dimension,
        )
        .unwrap()
    }

    fn cosine(a: &ContinuousHV, b: &ContinuousHV) -> f32 {
        assert_eq!(a.dim(), b.dim());
        let mut dot = 0.0f32;
        let mut aa = 0.0f32;
        let mut bb = 0.0f32;
        for (&x, &y) in a.values.iter().zip(&b.values) {
            dot += x * y;
            aa += x * x;
            bb += y * y;
        }
        dot / (aa * bb).sqrt()
    }

    #[test]
    fn construction_is_deterministic() {
        let genesis = GenesisSeed::from_phrase("deterministic-codebook");
        let a = LinearLevelCodebook::from_genesis(&genesis, "test::level", 32, 1024).unwrap();
        let b = LinearLevelCodebook::from_genesis(&genesis, "test::level", 32, 1024).unwrap();

        let encoded_a = a.encode_normalized(0.625).unwrap();
        let encoded_b = b.encode_normalized(0.625).unwrap();
        assert_eq!(encoded_a, encoded_b);
    }

    #[test]
    fn distinct_namespaces_do_not_reuse_the_same_codebook() {
        let a = codebook("field::pressure", 16, 1024)
            .encode_normalized(0.5)
            .unwrap();
        let b = codebook("field::temperature", 16, 1024)
            .encode_normalized(0.5)
            .unwrap();

        assert_ne!(a.representation.values, b.representation.values);
    }

    #[test]
    fn nearest_grid_quantization_is_explicit_at_boundaries() {
        let codebook = codebook("quantization", 5, 128);

        for (value, expected) in [
            (0.0, 0),
            (0.124_999, 0),
            (0.125, 1),
            (0.374_999, 1),
            (0.375, 2),
            (0.624_999, 2),
            (0.625, 3),
            (0.874_999, 3),
            (0.875, 4),
            (1.0, 4),
        ] {
            assert_eq!(codebook.selected_level(value).unwrap(), expected);
        }
    }

    #[test]
    fn same_bin_has_exactly_the_same_representation() {
        let codebook = codebook("same-bin", 17, 512);
        let a = codebook.encode_normalized(0.501).unwrap();
        let b = codebook.encode_normalized(0.529).unwrap();

        assert_eq!(a.selected_level, b.selected_level);
        assert_eq!(a.representation, b.representation);
    }

    #[test]
    fn adjacent_similarity_is_position_independent_with_even_partition() {
        let codebook = codebook("uniform-adjacency", 9, 4096);
        let mut similarities = Vec::new();
        let mut previous = codebook.encode_normalized(0.0).unwrap();

        for level in 1..codebook.level_count() {
            let value = level as f32 / (codebook.level_count() - 1) as f32;
            let current = codebook.encode_normalized(value).unwrap();
            similarities.push(cosine(
                &previous.representation,
                &current.representation,
            ));
            previous = current;
        }

        let min = similarities.iter().copied().fold(f32::INFINITY, f32::min);
        let max = similarities
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(max - min < 1.0e-5, "adjacent similarities: {similarities:?}");
    }

    #[test]
    fn nearby_levels_are_more_similar_than_distant_levels() {
        let codebook = codebook("continuity", 9, 4096);
        let anchor = codebook.encode_normalized(0.50).unwrap();
        let near = codebook.encode_normalized(0.625).unwrap();
        let far = codebook.encode_normalized(1.0).unwrap();

        let near_similarity = cosine(&anchor.representation, &near.representation);
        let far_similarity = cosine(&anchor.representation, &far.representation);
        assert!(
            near_similarity > far_similarity,
            "near={near_similarity}, far={far_similarity}"
        );
    }

    #[test]
    fn range_endpoints_are_approximately_orthogonal() {
        let codebook = codebook("endpoints", 17, 4096);
        let low = codebook.encode_normalized(0.0).unwrap();
        let high = codebook.encode_normalized(1.0).unwrap();

        assert!(cosine(&low.representation, &high.representation).abs() < 1.0e-5);
    }

    #[test]
    fn output_dimension_level_and_flip_count_are_observable() {
        let codebook = codebook("metadata", 9, 4096);
        let encoded = codebook.encode_normalized(0.75).unwrap();

        assert_eq!(codebook.namespace(), "metadata");
        assert_eq!(codebook.dimension(), 4096);
        assert_eq!(codebook.level_count(), 9);
        assert_eq!(encoded.representation.dim(), 4096);
        assert_eq!(encoded.selected_level, 6);
        assert_eq!(encoded.flipped_components, 1536);
    }

    #[test]
    fn every_output_component_has_fixed_unit_norm_magnitude() {
        let codebook = codebook("unit-norm", 9, 1024);
        let encoded = codebook.encode_normalized(0.75).unwrap();
        let expected = 1.0f32 / (1024.0f32).sqrt();

        assert!(encoded
            .representation
            .values
            .iter()
            .all(|value| value.abs() == expected));
        let norm_sq: f32 = encoded
            .representation
            .values
            .iter()
            .map(|value| value * value)
            .sum();
        assert!((norm_sq - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn invalid_construction_fails_before_large_allocation() {
        let genesis = GenesisSeed::from_phrase("invalid-construction");

        assert_eq!(
            LinearLevelCodebook::from_genesis(&genesis, "valid", 1, 128).unwrap_err(),
            LevelCodebookError::InvalidLevelCount { requested: 1 }
        );
        assert_eq!(
            LinearLevelCodebook::from_genesis(
                &genesis,
                "valid",
                MAX_LEVEL_COUNT + 1,
                1024,
            )
            .unwrap_err(),
            LevelCodebookError::InvalidLevelCount {
                requested: MAX_LEVEL_COUNT + 1
            }
        );
        assert_eq!(
            LinearLevelCodebook::from_genesis(&genesis, "valid", 2, 0).unwrap_err(),
            LevelCodebookError::ZeroDimension
        );
        assert_eq!(
            LinearLevelCodebook::from_genesis(&genesis, "valid", 2, MAX_LEVEL_DIMENSION + 1)
                .unwrap_err(),
            LevelCodebookError::DimensionTooLarge {
                requested: MAX_LEVEL_DIMENSION + 1
            }
        );
        assert_eq!(
            LinearLevelCodebook::from_genesis(&genesis, "valid", 65, 64).unwrap_err(),
            LevelCodebookError::InsufficientDimension
        );
        assert_eq!(
            LinearLevelCodebook::from_genesis(&genesis, "", 2, 128).unwrap_err(),
            LevelCodebookError::InvalidNamespace
        );
        assert_eq!(
            LinearLevelCodebook::from_genesis(&genesis, "contains space", 2, 128).unwrap_err(),
            LevelCodebookError::InvalidNamespace
        );
    }

    #[test]
    fn invalid_inputs_are_never_silently_clamped() {
        let codebook = codebook("strict-input", 16, 256);

        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                codebook.encode_normalized(value).unwrap_err(),
                LevelCodebookError::NonFiniteInput
            );
        }
        for value in [-0.001, 1.001] {
            assert_eq!(
                codebook.encode_normalized(value).unwrap_err(),
                LevelCodebookError::NormalizedInputOutOfRange
            );
        }
    }

    #[test]
    fn signed_zero_maps_to_the_same_low_endpoint() {
        let codebook = codebook("signed-zero", 16, 256);
        let positive = codebook.encode_normalized(0.0).unwrap();
        let negative = codebook.encode_normalized(-0.0).unwrap();

        assert_eq!(positive, negative);
        assert_eq!(positive.selected_level, 0);
        assert_eq!(positive.flipped_components, 0);
    }
}
