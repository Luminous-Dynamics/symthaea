//! Fixed-codebook scalar-level encoder for packed binary HDC.
//!
//! The caller supplies a pinned value-level codebook, pre-quantized feature levels,
//! and fixed position rotations. Quantization policy and codebook generation/provenance
//! are intentionally separate so input preprocessing remains explicit and reproducible.

use crate::packed_hdc::PackedHv;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarEncoderError {
    ZeroCapacity,
    FeatureLevelOutOfRange,
    EmptyBundle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarLevelEncoder<
    const LEVELS: usize,
    const FEATURES: usize,
    const WORDS: usize,
> {
    level_codebook: [PackedHv<WORDS>; LEVELS],
    position_rotations: [usize; FEATURES],
}

impl<const LEVELS: usize, const FEATURES: usize, const WORDS: usize>
    ScalarLevelEncoder<LEVELS, FEATURES, WORDS>
{
    pub const fn new(
        level_codebook: [PackedHv<WORDS>; LEVELS],
        position_rotations: [usize; FEATURES],
    ) -> Result<Self, ScalarEncoderError> {
        if LEVELS == 0 || FEATURES == 0 || WORDS == 0 {
            return Err(ScalarEncoderError::ZeroCapacity);
        }
        Ok(Self { level_codebook, position_rotations })
    }

    /// Encode a fixed set of discretized feature levels.
    ///
    /// Each feature selects a value-level vector, applies its fixed cyclic position
    /// permutation, and participates in a bitwise majority bundle. Even-sized bundles
    /// resolve exact bit ties to zero.
    pub fn encode(
        &self,
        feature_levels: &[usize; FEATURES],
    ) -> Result<PackedHv<WORDS>, ScalarEncoderError> {
        if FEATURES == 0 {
            return Err(ScalarEncoderError::ZeroCapacity);
        }
        let mut feature_vectors = [PackedHv::<WORDS>::zero(); FEATURES];
        let mut feature_index = 0usize;
        while feature_index < FEATURES {
            let level = feature_levels[feature_index];
            if level >= LEVELS {
                return Err(ScalarEncoderError::FeatureLevelOutOfRange);
            }
            feature_vectors[feature_index] = self.level_codebook[level]
                .permute_left(self.position_rotations[feature_index]);
            feature_index += 1;
        }
        PackedHv::majority_bundle(&feature_vectors).ok_or(ScalarEncoderError::EmptyBundle)
    }

    /// Static payload for the pinned value-level vectors and position rotations.
    pub const fn payload_bytes() -> Option<usize> {
        let vector_bytes = match WORDS.checked_mul(core::mem::size_of::<u64>()) {
            Some(value) => value,
            None => return None,
        };
        let codebook_bytes = match LEVELS.checked_mul(vector_bytes) {
            Some(value) => value,
            None => return None,
        };
        let rotation_bytes = match FEATURES.checked_mul(core::mem::size_of::<usize>()) {
            Some(value) => value,
            None => return None,
        };
        codebook_bytes.checked_add(rotation_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_binds_value_level_to_feature_position_and_bundles() {
        let codebook = [
            PackedHv::<1>::from_words([0b0001]),
            PackedHv::<1>::from_words([0b0010]),
        ];
        let encoder = ScalarLevelEncoder::<2, 3, 1>::new(codebook, [0, 1, 2]).unwrap();
        let encoded = encoder.encode(&[0, 1, 0]).unwrap();
        assert_eq!(encoded.words(), &[0b0100]);
    }

    #[test]
    fn encoder_rejects_out_of_range_feature_level() {
        let encoder = ScalarLevelEncoder::<2, 2, 1>::new(
            [PackedHv::zero(), PackedHv::from_words([u64::MAX])],
            [0, 1],
        ).unwrap();
        assert_eq!(encoder.encode(&[0, 2]), Err(ScalarEncoderError::FeatureLevelOutOfRange));
    }

    #[test]
    fn encoder_zero_dimensions_are_rejected() {
        assert_eq!(
            ScalarLevelEncoder::<0, 2, 1>::new([], [0, 1]),
            Err(ScalarEncoderError::ZeroCapacity)
        );
        assert_eq!(
            ScalarLevelEncoder::<2, 0, 1>::new([PackedHv::zero(); 2], []),
            Err(ScalarEncoderError::ZeroCapacity)
        );
        assert_eq!(
            ScalarLevelEncoder::<2, 2, 0>::new([PackedHv::zero(); 2], [0, 1]),
            Err(ScalarEncoderError::ZeroCapacity)
        );
    }

    #[test]
    fn codebook_and_position_payload_are_exact() {
        // Two 64-bit value vectors plus three usize position rotations.
        assert_eq!(
            ScalarLevelEncoder::<2, 3, 1>::payload_bytes(),
            Some(16 + 3 * core::mem::size_of::<usize>())
        );
    }
}
