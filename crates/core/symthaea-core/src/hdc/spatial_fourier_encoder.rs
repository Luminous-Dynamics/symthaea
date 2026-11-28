// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Metric-local 2D coordinate features with an explicit translation action.
//!
//! This is an additive research primitive. It does not replace GridEncoder or
//! PatchHdcEncoder and does not implement a map, SLAM, or a rotation-equivariant
//! representation.
//!
//! For each sampled frequency omega_i ~ Normal(0, ell^-2 I), a coordinate
//! x=(x,y) is encoded as (cos(omega_i dot x), sin(omega_i dot x)). The mean
//! pairwise inner product estimates the Gaussian RBF kernel
//! exp(-||x-y||^2 / (2 ell^2)).
//!
//! Translation is applied explicitly as a 2D rotation of each quadrature pair.
//! This is deliberately NOT implemented with ContinuousHV::bind: that
//! operation is a Hadamard product, not complex multiplication, and is not
//! self-inverse for continuous random components.
//!
//! Frequency generation uses SplitMix64 + Box-Muller so identical
//! (dimension, length scale, seed) inputs yield identical features without
//! depending on a process-global RNG. This generator is deterministic, not
//! cryptographic.

use std::fmt;

use super::ContinuousHV;

/// Configuration or input error for SpatialFourierEncoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialFourierError {
    /// Dimensions must be even and at least eight.
    InvalidDimension,
    /// The RBF length scale must be finite and strictly positive.
    InvalidLengthScale,
    /// A coordinate or translation component was NaN or infinite.
    NonFiniteCoordinate,
    /// A finite input overflowed while forming a phase.
    NonFinitePhase,
    /// A code was produced by a different encoder configuration.
    ConfigurationMismatch,
    /// A code has an unexpected internal vector length.
    InvalidCodeDimension,
}

impl fmt::Display for SpatialFourierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension => write!(f, "dimension must be even and at least 8"),
            Self::InvalidLengthScale => {
                write!(f, "length scale must be finite and strictly positive")
            }
            Self::NonFiniteCoordinate => {
                write!(f, "coordinates and translations must be finite")
            }
            Self::NonFinitePhase => {
                write!(f, "coordinate produced a non-finite Fourier phase")
            }
            Self::ConfigurationMismatch => {
                write!(f, "spatial Fourier codes use different encoder configurations")
            }
            Self::InvalidCodeDimension => {
                write!(f, "spatial Fourier code has an invalid vector dimension")
            }
        }
    }
}

impl std::error::Error for SpatialFourierError {}

/// Exact identity of the encoder configuration attached to each code.
///
/// The length scale is compared by bit pattern, not approximate float
/// equality. The algorithm version makes future sampling changes explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EncoderIdentity {
    dimension: usize,
    length_scale_bits: u64,
    seed: u64,
    algorithm_version: u32,
}

/// A metric-local coordinate code.
///
/// The underlying vector can be inspected by existing HDC/feature consumers,
/// but its quadrature-pair layout is meaningful only with its originating
/// encoder configuration. Use the encoder's similarity and translate methods
/// so incompatible codes are rejected.
#[derive(Debug, Clone)]
pub struct SpatialFourierCode {
    vector: ContinuousHV,
    identity: EncoderIdentity,
}

impl SpatialFourierCode {
    /// Borrow the encoded quadrature values as a ContinuousHV.
    ///
    /// Do not use Hadamard binding as a substitute for the phase-rotation
    /// action. A consumer composing this code with semantic features must
    /// define and test a compatible binding algebra separately.
    pub fn as_hv(&self) -> &ContinuousHV {
        &self.vector
    }

    /// Number of scalar features in this code.
    pub fn dimension(&self) -> usize {
        self.vector.dim()
    }
}

/// Random Fourier feature encoder for metric-local 2D coordinates.
///
/// Dimension is the number of scalar components and must be even; half as
/// many independent 2D frequency vectors are sampled. Length scale controls
/// the spatial scale over which similarity decays.
#[derive(Debug, Clone)]
pub struct SpatialFourierEncoder {
    identity: EncoderIdentity,
    frequencies: Vec<[f64; 2]>,
}

impl SpatialFourierEncoder {
    const ALGORITHM_VERSION: u32 = 1;

    /// Construct a deterministic encoder.
    ///
    /// Returns InvalidDimension when dimension is odd or less than eight, and
    /// InvalidLengthScale when the scale is not finite and positive.
    pub fn new(
        dimension: usize,
        length_scale: f64,
        seed: u64,
    ) -> Result<Self, SpatialFourierError> {
        if dimension < 8 || dimension % 2 != 0 {
            return Err(SpatialFourierError::InvalidDimension);
        }
        if !length_scale.is_finite() || length_scale <= 0.0 {
            return Err(SpatialFourierError::InvalidLengthScale);
        }

        let identity = EncoderIdentity {
            dimension,
            length_scale_bits: length_scale.to_bits(),
            seed,
            algorithm_version: Self::ALGORITHM_VERSION,
        };

        let mut rng_state = seed;
        let frequencies: Vec<[f64; 2]> = (0..dimension / 2)
            .map(|_| {
                let (normal_x, normal_y) = standard_normal_pair(&mut rng_state);
                [normal_x / length_scale, normal_y / length_scale]
            })
            .collect();
        if frequencies
            .iter()
            .flatten()
            .any(|frequency| !frequency.is_finite())
        {
            return Err(SpatialFourierError::InvalidLengthScale);
        }

        Ok(Self {
            identity,
            frequencies,
        })
    }

    /// Scalar feature dimension.
    pub fn dimension(&self) -> usize {
        self.identity.dimension
    }

    /// RBF length scale used by this encoder.
    pub fn length_scale(&self) -> f64 {
        f64::from_bits(self.identity.length_scale_bits)
    }

    /// Seed used for deterministic frequency generation.
    pub fn seed(&self) -> u64 {
        self.identity.seed
    }

    /// Encode a finite 2D coordinate as cosine/sine quadrature pairs.
    pub fn encode(
        &self,
        x: f64,
        y: f64,
    ) -> Result<SpatialFourierCode, SpatialFourierError> {
        if !x.is_finite() || !y.is_finite() {
            return Err(SpatialFourierError::NonFiniteCoordinate);
        }

        let mut values = Vec::with_capacity(self.dimension());
        for [omega_x, omega_y] in &self.frequencies {
            let phase = omega_x * x + omega_y * y;
            if !phase.is_finite() {
                return Err(SpatialFourierError::NonFinitePhase);
            }
            values.push(phase.cos() as f32);
            values.push(phase.sin() as f32);
        }

        if values.iter().any(|value| !value.is_finite()) {
            return Err(SpatialFourierError::NonFinitePhase);
        }

        Ok(SpatialFourierCode {
            vector: ContinuousHV::from_values(values),
            identity: self.identity,
        })
    }

    /// Apply an exact translation action to an existing coordinate code.
    ///
    /// For each frequency omega, this rotates the pair by omega dot (dx,dy).
    /// The result matches encoding (x+dx,y+dy) up to floating-point rounding,
    /// without needing the original x/y values.
    pub fn translate(
        &self,
        code: &SpatialFourierCode,
        dx: f64,
        dy: f64,
    ) -> Result<SpatialFourierCode, SpatialFourierError> {
        self.validate_code(code)?;
        if !dx.is_finite() || !dy.is_finite() {
            return Err(SpatialFourierError::NonFiniteCoordinate);
        }

        let input = code.vector.as_slice();
        let mut output = Vec::with_capacity(self.dimension());

        for (index, [omega_x, omega_y]) in self.frequencies.iter().enumerate() {
            let phase = omega_x * dx + omega_y * dy;
            if !phase.is_finite() {
                return Err(SpatialFourierError::NonFinitePhase);
            }

            let (sin_phase, cos_phase) = phase.sin_cos();
            let cosine = input[index * 2] as f64;
            let sine = input[index * 2 + 1] as f64;

            // Complex multiplication by exp(i * phase), expressed as an
            // explicit real 2x2 rotation. No ContinuousHV::bind is involved.
            let translated_cosine = cosine * cos_phase - sine * sin_phase;
            let translated_sine = sine * cos_phase + cosine * sin_phase;

            output.push(translated_cosine as f32);
            output.push(translated_sine as f32);
        }

        if output.iter().any(|value| !value.is_finite()) {
            return Err(SpatialFourierError::NonFinitePhase);
        }

        Ok(SpatialFourierCode {
            vector: ContinuousHV::from_values(output),
            identity: self.identity,
        })
    }

    /// Cosine similarity between codes from this exact encoder configuration.
    ///
    /// Computes the full local dot product directly rather than calling
    /// ContinuousHV::similarity, whose process-wide stride can subsample
    /// components.
    pub fn similarity(
        &self,
        left: &SpatialFourierCode,
        right: &SpatialFourierCode,
    ) -> Result<f32, SpatialFourierError> {
        self.validate_code(left)?;
        self.validate_code(right)?;

        let a = left.vector.as_slice();
        let b = right.vector.as_slice();
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;

        for (&a_value, &b_value) in a.iter().zip(b.iter()) {
            let a_value = a_value as f64;
            let b_value = b_value as f64;
            dot += a_value * b_value;
            norm_a += a_value * a_value;
            norm_b += b_value * b_value;
        }

        if norm_a <= f64::EPSILON || norm_b <= f64::EPSILON {
            return Err(SpatialFourierError::InvalidCodeDimension);
        }

        Ok((dot / (norm_a.sqrt() * norm_b.sqrt())).clamp(-1.0, 1.0) as f32)
    }

    fn validate_code(
        &self,
        code: &SpatialFourierCode,
    ) -> Result<(), SpatialFourierError> {
        if code.identity != self.identity {
            return Err(SpatialFourierError::ConfigurationMismatch);
        }
        if code.vector.dim() != self.dimension() {
            return Err(SpatialFourierError::InvalidCodeDimension);
        }
        Ok(())
    }
}

/// Deterministic SplitMix64 output. This is a reproducibility PRNG, not a
/// cryptographic random-number generator.
fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Uniform value strictly inside (0,1), using 53 random mantissa bits.
fn uniform_open01(state: &mut u64) -> f64 {
    const TWO_POW_53: f64 = 9_007_199_254_740_992.0;
    let mantissa = splitmix64_next(state) >> 11;
    (mantissa as f64 + 0.5) / TWO_POW_53
}

/// One Box-Muller pair of independent standard normal values.
fn standard_normal_pair(state: &mut u64) -> (f64, f64) {
    let radius = (-2.0 * uniform_open01(state).ln()).sqrt();
    let angle = std::f64::consts::TAU * uniform_open01(state);
    (radius * angle.cos(), radius * angle.sin())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoder(dimension: usize, length_scale: f64, seed: u64) -> SpatialFourierEncoder {
        SpatialFourierEncoder::new(dimension, length_scale, seed).expect("valid encoder config")
    }

    #[test]
    fn same_seed_and_coordinate_are_bit_deterministic() {
        let first = encoder(512, 1.25, 17);
        let second = encoder(512, 1.25, 17);
        let a = first.encode(0.25, -3.5).expect("finite point");
        let b = second.encode(0.25, -3.5).expect("finite point");
        assert_eq!(a.as_hv().as_slice(), b.as_hv().as_slice());
    }

    #[test]
    fn identical_coordinates_have_unit_similarity() {
        let enc = encoder(256, 2.0, 91);
        let a = enc.encode(1.25, -0.75).expect("finite point");
        let b = enc.encode(1.25, -0.75).expect("finite point");
        let similarity = enc.similarity(&a, &b).expect("compatible codes");
        assert!((similarity - 1.0).abs() < 1e-7, "similarity={similarity}");
    }

    #[test]
    fn nearby_points_are_more_similar_than_distant_points() {
        let enc = encoder(2048, 1.0, 2026);
        let origin = enc.encode(0.0, 0.0).expect("finite origin");
        let nearby = enc.encode(0.1, 0.0).expect("finite nearby point");
        let distant = enc.encode(2.0, 0.0).expect("finite distant point");

        let near_similarity = enc.similarity(&origin, &nearby).expect("compatible");
        let far_similarity = enc.similarity(&origin, &distant).expect("compatible");
        assert!(
            near_similarity > far_similarity + 0.4,
            "near={near_similarity}, far={far_similarity}"
        );
    }

    #[test]
    fn finite_features_approximate_the_declared_rbf_kernel() {
        let length_scale = 1.0;
        let enc = encoder(4096, length_scale, 71);
        let origin = enc.encode(0.0, 0.0).expect("finite origin");
        let point = enc.encode(0.7, -0.2).expect("finite point");
        let observed = enc.similarity(&origin, &point).expect("compatible");
        let distance_squared: f64 = 0.7_f64.powi(2) + (-0.2_f64).powi(2);
        let expected = (-distance_squared / (2.0 * length_scale.powi(2))).exp();
        assert!(
            (observed as f64 - expected).abs() < 0.04,
            "observed={observed}, expected={expected}"
        );
    }

    #[test]
    fn equal_radius_directions_are_isotropic_in_a_seed_sweep() {
        let mut x_total = 0.0f64;
        let mut y_total = 0.0f64;
        let seeds = 64u64;
        for seed in 0..seeds {
            let enc = encoder(1024, 1.0, seed);
            let origin = enc.encode(0.0, 0.0).expect("finite origin");
            let along_x = enc.encode(1.0, 0.0).expect("finite point");
            let along_y = enc.encode(0.0, 1.0).expect("finite point");
            x_total += enc.similarity(&origin, &along_x).expect("compatible") as f64;
            y_total += enc.similarity(&origin, &along_y).expect("compatible") as f64;
        }
        let x_mean = x_total / seeds as f64;
        let y_mean = y_total / seeds as f64;
        assert!(
            (x_mean - y_mean).abs() < 0.03,
            "anisotropy across seed sweep: x={x_mean}, y={y_mean}"
        );
    }

    #[test]
    fn translation_matches_reencoding_the_translated_coordinate() {
        let enc = encoder(1024, 1.5, 44);
        let original = enc.encode(0.4, -0.2).expect("finite point");
        let translated = enc.translate(&original, 0.3, 0.7).expect("finite shift");
        let direct = enc.encode(0.7, 0.5).expect("finite translated point");
        let similarity = enc.similarity(&translated, &direct).expect("compatible");
        assert!(
            similarity > 0.99999,
            "translation/re-encode similarity={similarity}"
        );
    }

    #[test]
    fn translation_composes_and_has_an_inverse() {
        let enc = encoder(1024, 0.9, 123);
        let origin = enc.encode(-0.8, 1.1).expect("finite point");
        let first = enc.translate(&origin, 0.35, -0.6).expect("first shift");
        let composed = enc.translate(&first, -0.2, 0.15).expect("second shift");
        let direct = enc.translate(&origin, 0.15, -0.45).expect("combined shift");
        let inverse = enc.translate(&composed, -0.15, 0.45).expect("inverse shift");

        assert!(enc.similarity(&composed, &direct).expect("compatible") > 0.99999);
        assert!(enc.similarity(&inverse, &origin).expect("compatible") > 0.99999);
    }

    #[test]
    fn phase_sign_and_axis_swap_negative_controls_are_detectable() {
        let enc = encoder(2048, 1.0, 5501);
        let origin = enc.encode(0.0, 0.0).expect("finite origin");
        let correct = enc.translate(&origin, 0.8, 0.3).expect("correct shift");
        let wrong_sign = enc.translate(&origin, -0.8, -0.3).expect("wrong sign control");
        let swapped_axes = enc.translate(&origin, 0.3, 0.8).expect("axis swap control");

        let sign_similarity = enc.similarity(&correct, &wrong_sign).expect("compatible");
        let axis_similarity = enc.similarity(&correct, &swapped_axes).expect("compatible");
        assert!(sign_similarity < 0.6, "sign control too similar: {sign_similarity}");
        assert!(axis_similarity < 0.95, "axis control too similar: {axis_similarity}");
    }

    #[test]
    fn incompatible_encoder_configurations_are_rejected() {
        let enc_a = encoder(256, 1.0, 1);
        let enc_b = encoder(256, 1.0, 2);
        let enc_c = encoder(256, 2.0, 1);
        let code_a = enc_a.encode(0.0, 0.0).expect("finite point");
        let code_b = enc_b.encode(0.0, 0.0).expect("finite point");
        let code_c = enc_c.encode(0.0, 0.0).expect("finite point");

        assert_eq!(
            enc_a.translate(&code_b, 1.0, 0.0).unwrap_err(),
            SpatialFourierError::ConfigurationMismatch
        );
        assert_eq!(
            enc_a.similarity(&code_a, &code_c).unwrap_err(),
            SpatialFourierError::ConfigurationMismatch
        );
    }

    #[test]
    fn invalid_dimensions_and_scales_fail_closed() {
        assert_eq!(
            SpatialFourierEncoder::new(7, 1.0, 0).unwrap_err(),
            SpatialFourierError::InvalidDimension
        );
        assert_eq!(
            SpatialFourierEncoder::new(9, 1.0, 0).unwrap_err(),
            SpatialFourierError::InvalidDimension
        );
        assert_eq!(
            SpatialFourierEncoder::new(128, 0.0, 0).unwrap_err(),
            SpatialFourierError::InvalidLengthScale
        );
        assert_eq!(
            SpatialFourierEncoder::new(128, f64::NAN, 0).unwrap_err(),
            SpatialFourierError::InvalidLengthScale
        );
    }

    #[test]
    fn non_finite_coordinates_and_translations_fail_closed() {
        let enc = encoder(128, 1.0, 7);
        assert_eq!(
            enc.encode(f64::NAN, 0.0).unwrap_err(),
            SpatialFourierError::NonFiniteCoordinate
        );
        let code = enc.encode(0.0, 0.0).expect("finite point");
        assert_eq!(
            enc.translate(&code, f64::INFINITY, 0.0).unwrap_err(),
            SpatialFourierError::NonFiniteCoordinate
        );
        assert_eq!(
            enc.encode(f64::MAX, f64::MAX).unwrap_err(),
            SpatialFourierError::NonFinitePhase
        );
    }
}
