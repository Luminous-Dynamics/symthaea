// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Continuous metric HDC encodings for 2D and 3D spatial coordinates.
//!
//! Unlike categorical row/column basis vectors, this encoder preserves local
//! metric structure: nearby coordinates produce similar hypervectors and
//! similarity changes smoothly with displacement over the configured frequency
//! scales.
//!
//! The construction mirrors Symthaea's smooth temporal phase encoder but uses
//! deterministic random Fourier directions. Each sin/cos pair shares a spatial
//! frequency vector `omega`, so the full pairwise dot product between encoded
//! points is an average of `cos(omega · delta)`. This gives a translation-invariant
//! spatial kernel.
//!
//! `ContinuousHV::similarity()` is intentionally **not** used as that kernel:
//! generic cognitive similarity may use the global adaptive `STRIDE`, which can
//! sample one component of a sin/cos pair without the other and destroy the
//! Fourier identity above. `SpatialHV` therefore retains its spatial domain and
//! basis metadata and provides a full-resolution similarity independent of the
//! cognitive throttle. Converting to a generic `ContinuousHV` is explicit and is
//! intended for binding/bundling, not for authoritative spatial comparison.
//!
//! 2D and 3D use **separate domain-separated direction banks**. 2D directions are
//! sampled uniformly on the unit circle; 3D directions are sampled uniformly on
//! the unit sphere. Both therefore preserve the exact configured log-spaced
//! frequency magnitude rather than obtaining 2D frequencies by projecting a 3D
//! direction and silently shrinking its XY magnitude.
//!
//! Metric coordinates and frequency banks remain `f64` through phase generation.
//! Only the final bounded sin/cos features are narrowed to `f32` for storage in
//! `ContinuousHV`. This avoids discarding metric precision before semantic
//! encoding and keeps the encoder compatible with the spatial-world metric types.
//!
//! Fourier features are periodic and therefore are not a globally injective
//! coordinate system. This encoder is intended for semantic locality and
//! relational reasoning; authoritative metric geometry must remain in an explicit
//! metric representation.
//!
//! This module is deliberately separate from the legacy categorical grid bases.
//! Introducing it therefore cannot silently change existing visual or ARC-style
//! encodings; consumers must opt in explicitly in a later qualified change.

use crate::hdc::{ContinuousHV, HDC_DIMENSION};

const SEED_DOMAIN_2D: u64 = 0x3244_5f46_5245_5155; // "2D_FREQU"
const SEED_DOMAIN_3D: u64 = 0x3344_5f46_5245_5155; // "3D_FREQU"
const FREQUENCY_SPAN: f64 = 256.0;

/// Configuration for continuous spatial encoding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialEncoderConfig {
    /// Requested hypervector dimensionality.
    ///
    /// The encoder emits sin/cos pairs, so an odd request is rounded down to the
    /// nearest even dimension. Values below four are rejected. The encoder stores
    /// the normalized emitted dimension in its returned configuration.
    pub dimensions: usize,
    /// Largest spatial frequency in radians per metric/world unit.
    ///
    /// Lower values make locality broader; higher values increase spatial
    /// discrimination. Must be finite, strictly positive, and large enough that
    /// `max_frequency / 256` remains representable as a positive `f64`.
    pub max_frequency: f64,
    /// Deterministic seed for Fourier direction generation.
    pub seed: u64,
}

impl SpatialEncoderConfig {
    /// Validate dimensionality and frequency bounds before allocation.
    pub fn validate(&self) -> Result<(), String> {
        if self.dimensions < 4 {
            return Err(format!(
                "spatial encoder dimensions must be >= 4, got {}",
                self.dimensions
            ));
        }
        if !self.max_frequency.is_finite() || self.max_frequency <= 0.0 {
            return Err(format!(
                "spatial encoder max_frequency must be finite and > 0, got {}",
                self.max_frequency
            ));
        }
        if self.max_frequency / FREQUENCY_SPAN == 0.0 {
            return Err(format!(
                "spatial encoder max_frequency is too small for the {}x frequency span: {}",
                FREQUENCY_SPAN, self.max_frequency
            ));
        }
        Ok(())
    }
}

impl Default for SpatialEncoderConfig {
    fn default() -> Self {
        Self {
            dimensions: HDC_DIMENSION,
            // Conservative default: centimetre-scale deltas remain highly
            // similar when one world unit is interpreted as roughly one metre,
            // while O(1) displacements are substantially more discriminable.
            // Consumers must still choose units/frequency explicitly for their
            // metric domain rather than treating this as a universal calibration.
            max_frequency: 16.0,
            seed: 0x5350_4154_4941_4c31,
        }
    }
}

/// Spatial domain carried by a protected spatial hypervector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialEncodingDomain {
    /// Two-dimensional metric encoding using the 2D Fourier bank.
    TwoD,
    /// Three-dimensional metric encoding using the 3D Fourier bank.
    ThreeD,
}

/// Spatial HDC value that retains the exact domain and Fourier-basis contract.
///
/// This wrapper prevents two semantic mistakes that a bare `ContinuousHV` cannot
/// detect: comparing 2D features with 3D features, and comparing features built
/// from different seeds/frequency schedules/dimensions. Its similarity is always
/// full-resolution and does not consult the global cognitive `STRIDE`.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialHV {
    domain: SpatialEncodingDomain,
    basis: SpatialEncoderConfig,
    hv: ContinuousHV,
}

impl SpatialHV {
    /// Spatial domain used to construct this feature vector.
    pub const fn domain(&self) -> SpatialEncodingDomain {
        self.domain
    }

    /// Normalized encoder configuration identifying the Fourier basis.
    pub const fn basis(&self) -> SpatialEncoderConfig {
        self.basis
    }

    /// Emitted hypervector dimension.
    pub fn dimension(&self) -> usize {
        self.hv.dim()
    }

    /// Raw feature components for diagnostics or explicit low-level integration.
    pub fn as_slice(&self) -> &[f32] {
        self.hv.as_slice()
    }

    /// Full-resolution similarity between values from the exact same spatial basis.
    ///
    /// This method deliberately bypasses `ContinuousHV::similarity()`, whose
    /// adaptive global stride is appropriate for cognitive throttling but not for
    /// preserving paired Fourier spatial semantics.
    pub fn similarity(&self, other: &Self) -> Result<f32, String> {
        if self.domain != other.domain {
            return Err(format!(
                "spatial HDC domain mismatch: {:?} vs {:?}",
                self.domain, other.domain
            ));
        }
        if self.basis != other.basis {
            return Err("spatial HDC Fourier basis mismatch".to_string());
        }
        full_cosine_similarity(self.hv.as_slice(), other.hv.as_slice())
    }

    /// Borrow the underlying generic HDC value for explicit binding/bundling use.
    ///
    /// The returned `ContinuousHV` no longer protects spatial-domain/basis
    /// semantics. In particular, its generic `similarity()` is **not** the
    /// qualified spatial kernel while adaptive cognitive stride is enabled.
    pub const fn as_continuous(&self) -> &ContinuousHV {
        &self.hv
    }

    /// Explicitly discard spatial-domain/basis safeguards and return generic HDC.
    ///
    /// Use this only when deliberately entering ordinary HDC algebra such as
    /// binding/bundling. Spatial comparisons should remain on `SpatialHV` or use
    /// `SpatialEncoder::similarity_2d` / `similarity_3d`.
    pub fn into_continuous(self) -> ContinuousHV {
        self.hv
    }
}

/// Smooth, translation-consistent HDC encoder for metric 2D/3D coordinates.
#[derive(Debug, Clone)]
pub struct SpatialEncoder {
    config: SpatialEncoderConfig,
    /// Dimension-correct `(wx, wy)` bank for 2D encoding.
    frequencies_2d: Vec<[f64; 2]>,
    /// Dimension-correct `(wx, wy, wz)` bank for 3D encoding.
    frequencies_3d: Vec<[f64; 3]>,
}

impl SpatialEncoder {
    /// Build deterministic dimension-specific Fourier feature banks from `config`.
    ///
    /// Frequency magnitudes are log-spaced from exactly `max_frequency / 256` to
    /// `max_frequency`. 2D/3D direction streams are domain-separated so neither
    /// representation is a projection or accidental prefix of the other.
    pub fn new(config: SpatialEncoderConfig) -> Result<Self, String> {
        config.validate()?;
        let pair_count = config.dimensions / 2;
        let config = SpatialEncoderConfig {
            dimensions: pair_count * 2,
            ..config
        };
        let mut rng_2d = SplitMix64::new(config.seed ^ SEED_DOMAIN_2D);
        let mut rng_3d = SplitMix64::new(config.seed ^ SEED_DOMAIN_3D);
        let mut frequencies_2d = Vec::with_capacity(pair_count);
        let mut frequencies_3d = Vec::with_capacity(pair_count);

        for pair in 0..pair_count {
            let magnitude = frequency_magnitude(config.max_frequency, pair, pair_count);

            // Uniform 2D direction on the unit circle. The XY norm remains
            // `magnitude` up to ordinary f64 trigonometric roundoff.
            let theta_2d = std::f64::consts::TAU * rng_2d.next_unit_f64();
            frequencies_2d.push([
                magnitude * theta_2d.cos(),
                magnitude * theta_2d.sin(),
            ]);

            // Uniform 3D direction on the unit sphere using an independent,
            // domain-separated deterministic stream.
            let z = 2.0 * rng_3d.next_unit_f64() - 1.0;
            let theta_3d = std::f64::consts::TAU * rng_3d.next_unit_f64();
            let radial = (1.0 - z * z).max(0.0).sqrt();
            frequencies_3d.push([
                magnitude * radial * theta_3d.cos(),
                magnitude * radial * theta_3d.sin(),
                magnitude * z,
            ]);
        }

        Ok(Self {
            config,
            frequencies_2d,
            frequencies_3d,
        })
    }

    /// Actual emitted hypervector dimension after enforcing sin/cos pairing.
    pub fn dimension(&self) -> usize {
        debug_assert_eq!(self.frequencies_2d.len(), self.frequencies_3d.len());
        debug_assert_eq!(self.config.dimensions, self.frequencies_2d.len() * 2);
        self.config.dimensions
    }

    /// Normalized configuration used to construct this encoder.
    pub const fn config(&self) -> SpatialEncoderConfig {
        self.config
    }

    /// Encode a finite `(x, y)` metric coordinate without narrowing it to `f32` first.
    pub fn encode_2d(&self, x: f64, y: f64) -> Result<SpatialHV, String> {
        Self::validate_point(&[x, y])?;
        let mut values = Vec::with_capacity(self.dimension());
        for [wx, wy] in &self.frequencies_2d {
            let phase = phase_2d([*wx, *wy], [x, y]);
            values.push(phase.sin() as f32);
            values.push(phase.cos() as f32);
        }
        Ok(SpatialHV {
            domain: SpatialEncodingDomain::TwoD,
            basis: self.config,
            hv: ContinuousHV::from_vec(values).normalize(),
        })
    }

    /// Encode a finite `(x, y, z)` metric coordinate without narrowing it to `f32` first.
    pub fn encode_3d(&self, x: f64, y: f64, z: f64) -> Result<SpatialHV, String> {
        Self::validate_point(&[x, y, z])?;
        let mut values = Vec::with_capacity(self.dimension());
        for [wx, wy, wz] in &self.frequencies_3d {
            let phase = phase_3d([*wx, *wy, *wz], [x, y, z]);
            values.push(phase.sin() as f32);
            values.push(phase.cos() as f32);
        }
        Ok(SpatialHV {
            domain: SpatialEncodingDomain::ThreeD,
            basis: self.config,
            hv: ContinuousHV::from_vec(values).normalize(),
        })
    }

    /// Translation-invariant Fourier-kernel similarity between finite 2D positions.
    ///
    /// This computes the paired spatial kernel directly in `f64`; it does not
    /// construct generic HDC values or consult the global cognitive stride.
    pub fn similarity_2d(&self, a: [f64; 2], b: [f64; 2]) -> Result<f32, String> {
        Self::validate_point(&a)?;
        Self::validate_point(&b)?;
        let sum = self
            .frequencies_2d
            .iter()
            .map(|frequency| {
                let phase_a = phase_2d(*frequency, a);
                let phase_b = phase_2d(*frequency, b);
                (phase_a - phase_b).cos()
            })
            .sum::<f64>();
        Ok((sum / self.frequencies_2d.len() as f64).clamp(-1.0, 1.0) as f32)
    }

    /// Translation-invariant Fourier-kernel similarity between finite 3D positions.
    ///
    /// This computes the paired spatial kernel directly in `f64`; it does not
    /// construct generic HDC values or consult the global cognitive stride.
    pub fn similarity_3d(&self, a: [f64; 3], b: [f64; 3]) -> Result<f32, String> {
        Self::validate_point(&a)?;
        Self::validate_point(&b)?;
        let sum = self
            .frequencies_3d
            .iter()
            .map(|frequency| {
                let phase_a = phase_3d(*frequency, a);
                let phase_b = phase_3d(*frequency, b);
                (phase_a - phase_b).cos()
            })
            .sum::<f64>();
        Ok((sum / self.frequencies_3d.len() as f64).clamp(-1.0, 1.0) as f32)
    }

    fn validate_point(values: &[f64]) -> Result<(), String> {
        if let Some((index, value)) = values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(format!(
                "spatial coordinate component {index} must be finite, got {value}"
            ));
        }
        Ok(())
    }
}

fn phase_2d(frequency: [f64; 2], point: [f64; 2]) -> f64 {
    (phase_component(frequency[0], point[0]) + phase_component(frequency[1], point[1]))
        .rem_euclid(std::f64::consts::TAU)
}

fn phase_3d(frequency: [f64; 3], point: [f64; 3]) -> f64 {
    (phase_component(frequency[0], point[0])
        + phase_component(frequency[1], point[1])
        + phase_component(frequency[2], point[2]))
    .rem_euclid(std::f64::consts::TAU)
}

/// Return one frequency-coordinate product modulo tau without overflowing.
///
/// Ordinary values take the direct product path. If `omega * coordinate` would
/// overflow, the coordinate is reduced by that frequency's spatial period before
/// multiplication. This preserves the Fourier phase while keeping every
/// intermediate finite.
fn phase_component(omega: f64, coordinate: f64) -> f64 {
    if omega == 0.0 || coordinate == 0.0 {
        return 0.0;
    }

    let abs_omega = omega.abs();
    if coordinate.abs() <= f64::MAX / abs_omega {
        return (omega * coordinate).rem_euclid(std::f64::consts::TAU);
    }

    let period = std::f64::consts::TAU / abs_omega;
    debug_assert!(period.is_finite() && period > 0.0);
    let reduced_coordinate = coordinate.rem_euclid(period);
    (omega * reduced_coordinate).rem_euclid(std::f64::consts::TAU)
}

fn full_cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, String> {
    if a.len() != b.len() {
        return Err(format!(
            "spatial HDC dimension mismatch: {} vs {}",
            a.len(),
            b.len()
        ));
    }

    let mut dot = 0.0_f64;
    let mut norm_a_sq = 0.0_f64;
    let mut norm_b_sq = 0.0_f64;
    for (&left, &right) in a.iter().zip(b.iter()) {
        let left = f64::from(left);
        let right = f64::from(right);
        dot += left * right;
        norm_a_sq += left * left;
        norm_b_sq += right * right;
    }

    let denom = (norm_a_sq * norm_b_sq).sqrt();
    if denom < 1e-20 {
        return Ok(0.0);
    }
    Ok((dot / denom).clamp(-1.0, 1.0) as f32)
}

fn frequency_magnitude(max_frequency: f64, pair: usize, pair_count: usize) -> f64 {
    // Log-spaced magnitudes provide simultaneous broad locality and fine
    // discrimination. The first pair is max/256 and the final pair is max.
    let t = if pair_count <= 1 {
        1.0
    } else {
        pair as f64 / (pair_count - 1) as f64
    };
    let min_frequency = max_frequency / FREQUENCY_SPAN;
    min_frequency * FREQUENCY_SPAN.powf(t)
}

#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_unit_f64(&mut self) -> f64 {
        // Use the 53 high bits so the integer-to-f64 conversion is exact and
        // every result is in [0, 1). This is deterministic without process-global RNG.
        const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * SCALE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoder() -> SpatialEncoder {
        SpatialEncoder::new(SpatialEncoderConfig {
            dimensions: 4096,
            max_frequency: 12.0,
            seed: 42,
        })
        .unwrap()
    }

    #[test]
    fn deterministic_for_same_seed_and_coordinate() {
        let a = encoder();
        let b = encoder();
        let a2 = a.encode_2d(1.25, -0.5).unwrap();
        let b2 = b.encode_2d(1.25, -0.5).unwrap();
        let a3 = a.encode_3d(1.25, -0.5, 7.0).unwrap();
        let b3 = b.encode_3d(1.25, -0.5, 7.0).unwrap();
        assert!(a2.similarity(&b2).unwrap() > 0.99999);
        assert!(a3.similarity(&b3).unwrap() > 0.99999);
    }

    #[test]
    fn protected_spatial_hv_rejects_cross_domain_and_cross_basis_similarity() {
        let enc = encoder();
        let two_d = enc.encode_2d(1.0, 2.0).unwrap();
        let three_d = enc.encode_3d(1.0, 2.0, 0.0).unwrap();
        assert!(two_d.similarity(&three_d).is_err());

        let other_basis = SpatialEncoder::new(SpatialEncoderConfig {
            seed: enc.config().seed + 1,
            ..enc.config()
        })
        .unwrap()
        .encode_2d(1.0, 2.0)
        .unwrap();
        assert!(two_d.similarity(&other_basis).is_err());
    }

    #[test]
    fn protected_spatial_hv_similarity_matches_direct_kernel() {
        let enc = encoder();
        let a_position = [0.25, -0.75];
        let b_position = [0.75, -0.25];
        let direct = enc.similarity_2d(a_position, b_position).unwrap();
        let a = enc.encode_2d(a_position[0], a_position[1]).unwrap();
        let b = enc.encode_2d(b_position[0], b_position[1]).unwrap();
        let encoded = a.similarity(&b).unwrap();
        assert!((direct - encoded).abs() < 1e-5, "direct={direct}, encoded={encoded}");
    }

    #[test]
    fn explicit_generic_hdc_downgrade_preserves_dimension() {
        let enc = encoder();
        let spatial = enc.encode_3d(1.0, 2.0, 3.0).unwrap();
        assert_eq!(spatial.domain(), SpatialEncodingDomain::ThreeD);
        assert_eq!(spatial.basis(), enc.config());
        let generic = spatial.into_continuous();
        assert_eq!(generic.dim(), enc.dimension());
    }

    #[test]
    fn dimension_specific_banks_preserve_frequency_magnitude_schedule() {
        let enc = encoder();
        let pair_count = enc.frequencies_2d.len();
        for index in [0, pair_count / 2, pair_count - 1] {
            let expected = frequency_magnitude(enc.config.max_frequency, index, pair_count);
            let [wx2, wy2] = enc.frequencies_2d[index];
            let actual_2d = wx2.hypot(wy2);
            let [wx3, wy3, wz3] = enc.frequencies_3d[index];
            let actual_3d = wx3.hypot(wy3).hypot(wz3);
            let tolerance = expected.max(1.0) * 2e-12;
            assert!(
                (actual_2d - expected).abs() < tolerance,
                "2d index={index}, expected={expected}, actual={actual_2d}"
            );
            assert!(
                (actual_3d - expected).abs() < tolerance,
                "3d index={index}, expected={expected}, actual={actual_3d}"
            );
        }
    }

    #[test]
    fn frequency_schedule_never_inverts_below_old_clamp() {
        let enc = SpatialEncoder::new(SpatialEncoderConfig {
            dimensions: 64,
            max_frequency: 1e-8,
            seed: 7,
        })
        .unwrap();
        let first = enc.frequencies_2d.first().unwrap();
        let last = enc.frequencies_2d.last().unwrap();
        let first_norm = first[0].hypot(first[1]);
        let last_norm = last[0].hypot(last[1]);
        assert!((first_norm - 1e-8 / FREQUENCY_SPAN).abs() < 1e-20);
        assert!((last_norm - 1e-8).abs() < 1e-20);
        assert!(first_norm < last_norm);
    }

    #[test]
    fn identical_position_has_unit_similarity() {
        let enc = encoder();
        let sim = enc
            .similarity_3d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
            .unwrap();
        assert!((sim - 1.0).abs() < 1e-5, "got {sim}");
    }

    #[test]
    fn nearby_positions_are_more_similar_than_distant_positions_2d() {
        let enc = encoder();
        let origin = [0.0, 0.0];
        let near = enc.similarity_2d(origin, [0.01, 0.0]).unwrap();
        let medium = enc.similarity_2d(origin, [0.25, 0.0]).unwrap();
        let far = enc.similarity_2d(origin, [2.0, 0.0]).unwrap();
        assert!(near > medium, "near={near}, medium={medium}");
        assert!(medium > far, "medium={medium}, far={far}");
        assert!(near > 0.99, "near locality margin regressed: {near}");
        assert!(far < 0.75, "far discrimination margin regressed: {far}");
    }

    #[test]
    fn two_dimensional_kernel_is_approximately_isotropic() {
        let enc = encoder();
        let radius = 0.5_f64;
        let diagonal = radius / 2.0_f64.sqrt();
        let x = enc.similarity_2d([0.0, 0.0], [radius, 0.0]).unwrap();
        let y = enc.similarity_2d([0.0, 0.0], [0.0, radius]).unwrap();
        let d = enc
            .similarity_2d([0.0, 0.0], [diagonal, diagonal])
            .unwrap();
        let max = x.max(y).max(d);
        let min = x.min(y).min(d);
        assert!(max - min < 0.04, "x={x}, y={y}, diagonal={d}");
    }

    #[test]
    fn nearby_positions_are_more_similar_than_distant_positions_3d() {
        let enc = encoder();
        let origin = [0.0, 0.0, 0.0];
        let near = enc
            .similarity_3d(origin, [0.01, 0.01, 0.01])
            .unwrap();
        let far = enc.similarity_3d(origin, [2.0, 2.0, 2.0]).unwrap();
        assert!(near > 0.99, "near={near}");
        assert!(far < 0.75, "far={far}");
        assert!(near > far, "near={near}, far={far}");
    }

    #[test]
    fn pairwise_similarity_is_translation_consistent_2d() {
        let enc = encoder();
        let a = [0.25, -0.75];
        let b = [0.75, -0.25];
        let shift = [100.0, -37.0];
        let lhs = enc.similarity_2d(a, b).unwrap();
        let rhs = enc
            .similarity_2d(
                [a[0] + shift[0], a[1] + shift[1]],
                [b[0] + shift[0], b[1] + shift[1]],
            )
            .unwrap();
        assert!((lhs - rhs).abs() < 2e-4, "lhs={lhs}, rhs={rhs}");
    }

    #[test]
    fn pairwise_similarity_is_translation_consistent_3d() {
        let enc = encoder();
        let a = [0.25, -0.75, 1.0];
        let b = [0.75, -0.25, 1.5];
        let shift = [100.0, -37.0, 9.0];
        let lhs = enc.similarity_3d(a, b).unwrap();
        let rhs = enc
            .similarity_3d(
                [a[0] + shift[0], a[1] + shift[1], a[2] + shift[2]],
                [b[0] + shift[0], b[1] + shift[1], b[2] + shift[2]],
            )
            .unwrap();
        assert!((lhs - rhs).abs() < 2e-4, "lhs={lhs}, rhs={rhs}");
    }

    #[test]
    fn f64_metric_precision_survives_beyond_f32_resolution() {
        let enc = encoder();
        let base = 16_777_216.0_f64; // 2^24; f32 cannot retain a +0.25 delta here.
        let local = enc.similarity_2d([0.0, 0.0], [0.25, 0.0]).unwrap();
        let shifted = enc
            .similarity_2d([base, -base], [base + 0.25, -base])
            .unwrap();
        assert!((local - shifted).abs() < 2e-4, "local={local}, shifted={shifted}");
        assert!(shifted < 0.999, "sub-f32-resolution displacement was erased: {shifted}");
    }

    #[test]
    fn requested_odd_dimension_is_normalized_to_emitted_even_basis() {
        let enc = SpatialEncoder::new(SpatialEncoderConfig {
            dimensions: 257,
            max_frequency: 8.0,
            seed: 7,
        })
        .unwrap();
        assert_eq!(enc.dimension(), 256);
        assert_eq!(enc.config().dimensions, 256);
    }

    #[test]
    fn rejects_invalid_configuration() {
        assert!(SpatialEncoder::new(SpatialEncoderConfig {
            dimensions: 2,
            max_frequency: 1.0,
            seed: 0,
        })
        .is_err());
        assert!(SpatialEncoder::new(SpatialEncoderConfig {
            dimensions: 256,
            max_frequency: 0.0,
            seed: 0,
        })
        .is_err());
        assert!(SpatialEncoder::new(SpatialEncoderConfig {
            dimensions: 256,
            max_frequency: f64::from_bits(1),
            seed: 0,
        })
        .is_err());
    }

    #[test]
    fn rejects_non_finite_coordinates() {
        let enc = encoder();
        assert!(enc.encode_2d(f64::NAN, 0.0).is_err());
        assert!(enc.encode_3d(0.0, f64::INFINITY, 0.0).is_err());
        assert!(enc
            .similarity_3d([0.0, 0.0, 0.0], [0.0, 0.0, f64::NEG_INFINITY])
            .is_err());
    }

    #[test]
    fn extreme_finite_coordinates_remain_finite() {
        let enc = encoder();
        let hv2 = enc.encode_2d(f64::MAX, -f64::MAX).unwrap();
        let hv3 = enc
            .encode_3d(f64::MAX, -f64::MAX, f64::MAX)
            .unwrap();
        assert!(hv2.as_slice().iter().all(|value| value.is_finite()));
        assert!(hv3.as_slice().iter().all(|value| value.is_finite()));
    }
}
