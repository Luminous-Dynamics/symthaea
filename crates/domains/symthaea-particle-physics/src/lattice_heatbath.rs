// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! SU(2) heat-bath sampling primitives for Cabibbo-Marinari lattice updates.
//!
//! The nonzero-coupling scalar draw uses the Kennedy-Pendleton rejection
//! construction. Its target distribution is qualified independently by the
//! LQCD-016E generic-rejection/quadrature oracle:
//!
//! `p(a0) da0 ∝ sqrt(1-a0^2) exp(alpha a0) da0`, `a0 ∈ [-1,1]`.
//!
//! This module stops at an SU(2) quaternion sample. It does not compute an SU(3)
//! local force, update a gauge link, or claim ensemble equilibrium.

use crate::lattice_sweep::Uniform01Source;
use std::f64::consts::TAU;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Su2HeatbathMethod {
    /// Kennedy-Pendleton rejection sampler for `alpha > 0`.
    KennedyPendleton,
    /// Exact generic rejection for the Haar scalar limit `alpha == 0`.
    HaarScalarRejection,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Su2HeatbathSample {
    /// Unit quaternion `(a0,a1,a2,a3)` representing an SU(2) matrix.
    pub quaternion: [f64; 4],
    /// Number of scalar rejection proposals used before acceptance.
    pub scalar_attempts: usize,
    pub method: Su2HeatbathMethod,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Su2HeatbathError {
    InvalidAlpha(f64),
    InvalidMaxAttempts(usize),
    InvalidUniformDraw(f64),
    RejectionLimitExceeded { alpha: f64, max_attempts: usize },
    NonFiniteSample([f64; 4]),
}

fn open01(source: &mut impl Uniform01Source) -> Result<f64, Su2HeatbathError> {
    let u = source.next_uniform();
    if !u.is_finite() || u <= 0.0 || u >= 1.0 {
        return Err(Su2HeatbathError::InvalidUniformDraw(u));
    }
    Ok(u)
}

fn draw_direction_s2(
    source: &mut impl Uniform01Source,
) -> Result<[f64; 3], Su2HeatbathError> {
    let z = 2.0 * open01(source)? - 1.0;
    let phi = TAU * open01(source)?;
    let radial = (1.0 - z * z).max(0.0).sqrt();
    Ok([radial * phi.cos(), radial * phi.sin(), z])
}

fn draw_haar_scalar(
    source: &mut impl Uniform01Source,
    max_attempts: usize,
) -> Result<(f64, usize), Su2HeatbathError> {
    for attempt in 1..=max_attempts {
        let a0 = 2.0 * open01(source)? - 1.0;
        let acceptance = (1.0 - a0 * a0).max(0.0).sqrt();
        if open01(source)? < acceptance {
            return Ok((a0, attempt));
        }
    }
    Err(Su2HeatbathError::RejectionLimitExceeded {
        alpha: 0.0,
        max_attempts,
    })
}

/// Draw the scalar quaternion component with the Kennedy-Pendleton construction.
///
/// For each attempt, with independent U(0,1) draws `r0..r3`:
///
/// - `x1 = -ln(r1) / alpha`
/// - `x2 = -ln(r2) / alpha`
/// - `d = x2 + x1 cos^2(2π r3)`
/// - accept when `r0^2 < 1 - d/2`
/// - return `a0 = 1 - d`
///
/// This produces the independently qualified scalar heat-bath density for
/// `alpha > 0`.
pub fn draw_kennedy_pendleton_scalar(
    source: &mut impl Uniform01Source,
    alpha: f64,
    max_attempts: usize,
) -> Result<(f64, usize), Su2HeatbathError> {
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(Su2HeatbathError::InvalidAlpha(alpha));
    }
    if max_attempts == 0 {
        return Err(Su2HeatbathError::InvalidMaxAttempts(0));
    }

    for attempt in 1..=max_attempts {
        let r0 = open01(source)?;
        let r1 = open01(source)?;
        let r2 = open01(source)?;
        let r3 = open01(source)?;

        let x1 = -r1.ln() / alpha;
        let x2 = -r2.ln() / alpha;
        let cosine = (TAU * r3).cos();
        let d = x2 + x1 * cosine * cosine;
        let threshold = 1.0 - 0.5 * d;

        if threshold > 0.0 && r0 * r0 < threshold {
            let a0 = 1.0 - d;
            if a0.is_finite() && (-1.0..=1.0).contains(&a0) {
                return Ok((a0, attempt));
            }
        }
    }

    Err(Su2HeatbathError::RejectionLimitExceeded {
        alpha,
        max_attempts,
    })
}

/// Draw one unit SU(2) quaternion from the heat-bath conditional.
///
/// `alpha == 0` uses an exact generic rejection construction for the Haar scalar
/// marginal; `alpha > 0` uses Kennedy-Pendleton. Negative alpha is rejected: a
/// caller that obtains a signed local-force normalization must absorb the sign
/// into its force orientation before invoking this distribution sampler.
pub fn draw_su2_heatbath_quaternion(
    source: &mut impl Uniform01Source,
    alpha: f64,
    max_attempts: usize,
) -> Result<Su2HeatbathSample, Su2HeatbathError> {
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(Su2HeatbathError::InvalidAlpha(alpha));
    }
    if max_attempts == 0 {
        return Err(Su2HeatbathError::InvalidMaxAttempts(0));
    }

    let (a0, scalar_attempts, method) = if alpha == 0.0 {
        let (a0, attempts) = draw_haar_scalar(source, max_attempts)?;
        (a0, attempts, Su2HeatbathMethod::HaarScalarRejection)
    } else {
        let (a0, attempts) = draw_kennedy_pendleton_scalar(source, alpha, max_attempts)?;
        (a0, attempts, Su2HeatbathMethod::KennedyPendleton)
    };

    let direction = draw_direction_s2(source)?;
    let radius = (1.0 - a0 * a0).max(0.0).sqrt();
    let quaternion = [
        a0,
        radius * direction[0],
        radius * direction[1],
        radius * direction[2],
    ];
    if quaternion.iter().any(|x| !x.is_finite()) {
        return Err(Su2HeatbathError::NonFiniteSample(quaternion));
    }

    Ok(Su2HeatbathSample {
        quaternion,
        scalar_attempts,
        method,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_rng::{
        LatticeChaCha8Stream, LatticeStreamCoordinates, LatticeStreamDomain,
    };

    fn source(seed_byte: u8, replica: u16) -> LatticeChaCha8Stream {
        LatticeChaCha8Stream::new(
            [seed_byte; 32],
            LatticeStreamCoordinates {
                domain: LatticeStreamDomain::Qualification,
                ensemble_slot: 0x160f,
                replica,
                rank: 0,
            },
        )
        .unwrap()
    }

    fn moment_fixture(alpha: f64, expected_mean: f64, expected_second: f64, replica: u16) {
        const N: usize = 50_000;
        let mut rng = source(0x6b, replica);
        let mut sum = 0.0;
        let mut sum2 = 0.0;
        let mut sum4 = 0.0;
        let mut attempts = 0usize;
        let mut max_norm_error: f64 = 0.0;
        for _ in 0..N {
            let sample = draw_su2_heatbath_quaternion(&mut rng, alpha, 256).unwrap();
            let a0 = sample.quaternion[0];
            sum += a0;
            sum2 += a0 * a0;
            sum4 += a0.powi(4);
            attempts += sample.scalar_attempts;
            let norm_sq = sample.quaternion.iter().map(|x| x * x).sum::<f64>();
            max_norm_error = max_norm_error.max((norm_sq - 1.0).abs());
        }
        let mean = sum / N as f64;
        let second = sum2 / N as f64;
        let mean_variance = (second - mean * mean).max(0.0);
        let mean_se = (mean_variance / N as f64).sqrt();
        let second_variance = (sum4 / N as f64 - second * second).max(0.0);
        let second_se = (second_variance / N as f64).sqrt();

        assert!((mean - expected_mean).abs() < 5.0 * mean_se + 1.0e-12);
        assert!((second - expected_second).abs() < 5.0 * second_se + 1.0e-12);
        assert!(max_norm_error < 1.0e-12);
        assert!(attempts >= N);
    }

    #[test]
    fn kennedy_pendleton_matches_independent_density_oracle_at_alpha_1p5() {
        moment_fixture(
            1.5,
            0.344_144_010_564_440_55,
            0.311_711_990_151_271_66,
            1,
        );
    }

    #[test]
    fn kennedy_pendleton_matches_independent_density_oracle_at_alpha_5() {
        moment_fixture(
            5.0,
            0.719_340_588_779_950_2,
            0.568_395_662_588_997_8,
            2,
        );
    }

    #[test]
    fn haar_limit_matches_known_first_two_scalar_moments() {
        moment_fixture(0.0, 0.0, 0.25, 3);
    }

    #[test]
    fn method_and_failure_boundaries_are_explicit() {
        let mut rng = source(0x42, 4);
        let kp = draw_su2_heatbath_quaternion(&mut rng, 2.0, 256).unwrap();
        assert_eq!(kp.method, Su2HeatbathMethod::KennedyPendleton);

        let mut rng = source(0x43, 5);
        let haar = draw_su2_heatbath_quaternion(&mut rng, 0.0, 256).unwrap();
        assert_eq!(haar.method, Su2HeatbathMethod::HaarScalarRejection);

        assert!(matches!(
            draw_kennedy_pendleton_scalar(&mut rng, 0.0, 256),
            Err(Su2HeatbathError::InvalidAlpha(0.0))
        ));
        assert!(matches!(
            draw_su2_heatbath_quaternion(&mut rng, -1.0, 256),
            Err(Su2HeatbathError::InvalidAlpha(-1.0))
        ));
        assert!(matches!(
            draw_su2_heatbath_quaternion(&mut rng, 1.0, 0),
            Err(Su2HeatbathError::InvalidMaxAttempts(0))
        ));
    }
}
