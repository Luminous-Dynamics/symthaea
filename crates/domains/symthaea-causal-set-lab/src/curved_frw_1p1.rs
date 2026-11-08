// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fixed-cardinality causal-set sprinkling for a conformal 1+1D FRW diamond.
//!
//! Metric and conventions (c = 1):
//!
//!     ds^2 = a(eta)^2 (-d eta^2 + dx^2)
//!     a(eta) = exp(kappa * eta^2 / 2), kappa >= 0
//!
//! The bounded diamond is |eta| <= T/2 and
//! |x| <= T/2 - |eta|. Its volume density is sqrt(|g|) = a^2.
//! Causal order is conformally invariant and is represented by the same null
//! coordinates as the flat 1+1D causal-set crate:
//!
//!     u = eta + x + T/2, v = eta - x + T/2
//!
//! Both coordinates lie in [0,T]. Curvature and proper time are not encoded
//! by the order alone; the curved volume measure is sampled explicitly.
//!
//! This module is kinematics and sampling only. It does not implement a
//! causal-set d'Alembertian, propagator, quantum measure or continuum-limit
//! proof. The curvature sign convention gives R = 2 a'' / a^3.

use crate::{CausalSet2D, CausalSetError, Event2D};
use std::error::Error;
use std::fmt;

/// Number of Simpson subinterval pairs used to tabulate the eta marginal CDF.
const CDF_INTERVALS: usize = 2048;
const PROPER_TIME_INTERVALS: usize = 1024;
const CDF_BISECTION_STEPS: usize = 48;

/// Invalid metric parameters or numerical failure in curved sprinkling.
#[derive(Clone, Debug, PartialEq)]
pub enum CurvedFrwError {
    InvalidProperTime,
    InvalidCurvature,
    ExponentialOverflow,
    NumericalFailure,
    CausalSet(CausalSetError),
}

impl fmt::Display for CurvedFrwError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProperTime => write!(f, "proper-time coordinate extent must be finite and positive"),
            Self::InvalidCurvature => write!(f, "kappa must be finite and non-negative"),
            Self::ExponentialOverflow => write!(f, "curvature and region size overflow the volume density"),
            Self::NumericalFailure => write!(f, "curved-volume quadrature or inverse CDF failed"),
            Self::CausalSet(error) => write!(f, "causal-set construction failed: {error}"),
        }
    }
}

impl Error for CurvedFrwError {}

impl From<CausalSetError> for CurvedFrwError {
    fn from(value: CausalSetError) -> Self {
        Self::CausalSet(value)
    }
}

/// A conformal 1+1D FRW metric restricted to one symmetric Alexandrov diamond.
#[derive(Clone, Debug)]
pub struct ConformalFrw1p1 {
    kappa: f64,
    proper_time_coordinate_extent: f64,
    total_volume: f64,
    /// Cumulative integral of (T - 2r) exp(kappa r^2) on a uniform r grid.
    radial_cdf: Vec<f64>,
}

impl ConformalFrw1p1 {
    /// Construct the metric on a causal diamond of conformal-time height T.
    ///
    /// The total volume is
    ///
    ///     V = 2 integral_0^(T/2) (T - 2r) exp(kappa r^2) dr.
    ///
    /// The fixed-cardinality sampler uses the corresponding exact conditional
    /// density tabulated by composite Simpson quadrature.
    pub fn new(kappa: f64, proper_time_coordinate_extent: f64) -> Result<Self, CurvedFrwError> {
        if !proper_time_coordinate_extent.is_finite() || proper_time_coordinate_extent <= 0.0 {
            return Err(CurvedFrwError::InvalidProperTime);
        }
        if !kappa.is_finite() || kappa < 0.0 {
            return Err(CurvedFrwError::InvalidCurvature);
        }
        let half = proper_time_coordinate_extent * 0.5;
        let maximum_exponent = kappa * half * half;
        if !maximum_exponent.is_finite() || maximum_exponent > 600.0 {
            return Err(CurvedFrwError::ExponentialOverflow);
        }

        let step = half / CDF_INTERVALS as f64;
        let mut radial_cdf = Vec::with_capacity(CDF_INTERVALS + 1);
        radial_cdf.push(0.0);
        let mut cumulative = 0.0;
        for index in 0..CDF_INTERVALS {
            let left = index as f64 * step;
            let right = (index + 1) as f64 * step;
            let middle = 0.5 * (left + right);
            let segment = step / 6.0
                * (radial_density(left, proper_time_coordinate_extent, kappa)
                    + 4.0 * radial_density(middle, proper_time_coordinate_extent, kappa)
                    + radial_density(right, proper_time_coordinate_extent, kappa));
            cumulative += segment;
            if !cumulative.is_finite() {
                return Err(CurvedFrwError::NumericalFailure);
            }
            radial_cdf.push(cumulative);
        }

        let total_volume = 2.0 * cumulative;
        if !total_volume.is_finite() || total_volume <= 0.0 {
            return Err(CurvedFrwError::NumericalFailure);
        }

        Ok(Self {
            kappa,
            proper_time_coordinate_extent,
            total_volume,
            radial_cdf,
        })
    }

    pub fn kappa(&self) -> f64 {
        self.kappa
    }

    pub fn proper_time_coordinate_extent(&self) -> f64 {
        self.proper_time_coordinate_extent
    }

    pub fn total_volume(&self) -> f64 {
        self.total_volume
    }

    pub fn scale_factor(&self, eta: f64) -> Option<f64> {
        if !eta.is_finite() || eta.abs() > 0.5 * self.proper_time_coordinate_extent {
            return None;
        }
        let value = (0.5 * self.kappa * eta * eta).exp();
        value.is_finite().then_some(value)
    }

    /// Lorentzian volume density sqrt(|g|) = a(eta)^2.
    pub fn volume_density(&self, eta: f64) -> Option<f64> {
        if !eta.is_finite() || eta.abs() > 0.5 * self.proper_time_coordinate_extent {
            return None;
        }
        let value = (self.kappa * eta * eta).exp();
        value.is_finite().then_some(value)
    }

    /// Ricci scalar for the convention R = 2 a'' / a^3.
    pub fn ricci_scalar(&self, eta: f64) -> Option<f64> {
        let scale = self.scale_factor(eta)?;
        let value = 2.0 * self.kappa / scale.powi(2);
        value.is_finite().then_some(value)
    }

    /// Comoving proper-time duration between two conformal times.
    ///
    /// Composite Simpson integration is used as an independent numerical
    /// reference. The result is non-negative and requires eta_start <= eta_end.
    pub fn comoving_proper_time(&self, eta_start: f64, eta_end: f64) -> Option<f64> {
        let half = 0.5 * self.proper_time_coordinate_extent;
        if !eta_start.is_finite()
            || !eta_end.is_finite()
            || eta_start < -half
            || eta_end > half
            || eta_end < eta_start
        {
            return None;
        }
        if eta_start == eta_end {
            return Some(0.0);
        }

        let intervals = PROPER_TIME_INTERVALS;
        let step = (eta_end - eta_start) / intervals as f64;
        let mut sum = self.scale_factor(eta_start)? + self.scale_factor(eta_end)?;
        for index in 1..intervals {
            let eta = eta_start + index as f64 * step;
            let weight = if index % 2 == 0 { 2.0 } else { 4.0 };
            sum += weight * self.scale_factor(eta)?;
        }
        let result = sum * step / 3.0;
        result.is_finite().then_some(result)
    }

    /// Expected eta^2 under the normalized curved-volume sprinkling density.
    pub fn expected_eta_squared(&self) -> f64 {
        let half = 0.5 * self.proper_time_coordinate_extent;
        let step = half / CDF_INTERVALS as f64;
        let mut integral = 0.0;
        for index in 0..CDF_INTERVALS {
            let left = index as f64 * step;
            let right = (index + 1) as f64 * step;
            let middle = 0.5 * (left + right);
            let f = |radius: f64| {
                radius * radius
                    * radial_density(radius, self.proper_time_coordinate_extent, self.kappa)
            };
            integral += step / 6.0 * (f(left) + 4.0 * f(middle) + f(right));
        }
        2.0 * integral / self.total_volume
    }

    /// Deterministic fixed-cardinality Poisson-conditioned sprinkling.
    ///
    /// Each event is independently sampled from the normalized Lorentzian
    /// volume measure. The transformation to (u,v) preserves the causal order
    /// exactly because the metric is conformal to Minkowski spacetime.
    pub fn sprinkle_fixed_cardinality(
        &self,
        seed: u64,
        event_count: usize,
    ) -> Result<CausalSet2D, CurvedFrwError> {
        let mut rng = SplitMix64::new(seed);
        let half = 0.5 * self.proper_time_coordinate_extent;
        let mut events = Vec::with_capacity(event_count);

        for id in 0..event_count {
            let radius = self.sample_abs_eta(&mut rng)?;
            let eta = if rng.next_u64() & 1 == 0 {
                -radius
            } else {
                radius
            };
            let spatial_half_width = (half - radius).max(0.0);
            let x = (2.0 * rng.unit_f64() - 1.0) * spatial_half_width;
            let mut u = eta + x + half;
            let mut v = eta - x + half;

            // Clamp only floating-point roundoff at the null-coordinate bounds.
            u = u.clamp(0.0, self.proper_time_coordinate_extent);
            v = v.clamp(0.0, self.proper_time_coordinate_extent);
            events.push(Event2D { id, u, v });
        }

        Ok(CausalSet2D::new(
            events,
            self.proper_time_coordinate_extent,
            self.proper_time_coordinate_extent,
        )?)
    }

    fn sample_abs_eta(&self, rng: &mut SplitMix64) -> Result<f64, CurvedFrwError> {
        let target = rng.unit_f64() * self.radial_cdf[CDF_INTERVALS];
        let upper = self
            .radial_cdf
            .partition_point(|value| *value < target)
            .clamp(1, CDF_INTERVALS);
        let lower = upper - 1;
        let half = 0.5 * self.proper_time_coordinate_extent;
        let step = half / CDF_INTERVALS as f64;
        let left = lower as f64 * step;
        let right = upper as f64 * step;
        let base = self.radial_cdf[lower];
        let local_target = target - base;
        let mut low = left;
        let mut high = right;

        for _ in 0..CDF_BISECTION_STEPS {
            let middle = 0.5 * (low + high);
            let partial = simpson_integral(
                left,
                middle,
                |radius| radial_density(radius, self.proper_time_coordinate_extent, self.kappa),
            );
            if partial < local_target {
                low = middle;
            } else {
                high = middle;
            }
        }

        let sample = 0.5 * (low + high);
        if sample.is_finite() {
            Ok(sample)
        } else {
            Err(CurvedFrwError::NumericalFailure)
        }
    }
}

fn radial_density(radius: f64, extent: f64, kappa: f64) -> f64 {
    (extent - 2.0 * radius) * (kappa * radius * radius).exp()
}

fn simpson_integral<F>(left: f64, right: f64, function: F) -> f64
where
    F: Fn(f64) -> f64,
{
    if right <= left {
        return 0.0;
    }
    let middle = 0.5 * (left + right);
    (right - left) / 6.0 * (function(left) + 4.0 * function(middle) + function(right))
}

#[derive(Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn unit_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_limit_recovers_the_minkowski_diamond_volume() {
        let metric = ConformalFrw1p1::new(0.0, 2.0).unwrap();
        assert!((metric.total_volume() - 2.0).abs() < 1.0e-12);
        assert!((metric.ricci_scalar(0.0).unwrap()).abs() < 1.0e-14);
        assert!((metric.expected_eta_squared() - 1.0 / 6.0).abs() < 1.0e-3);
    }

    #[test]
    fn analytic_scale_factor_and_curvature_are_recovered() {
        let metric = ConformalFrw1p1::new(0.8, 2.0).unwrap();
        for eta in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            let expected_a = (0.5_f64 * 0.8 * eta * eta).exp();
            let expected_r = 1.6 * (-0.8 * eta * eta).exp();
            assert!((metric.scale_factor(eta).unwrap() - expected_a).abs() < 1.0e-12);
            assert!((metric.ricci_scalar(eta).unwrap() - expected_r).abs() < 1.0e-12);
        }
    }

    #[test]
    fn fixed_seed_curved_sprinkling_is_reproducible_and_bounded() {
        let metric = ConformalFrw1p1::new(0.6, 2.0).unwrap();
        let left = metric.sprinkle_fixed_cardinality(0xC0FFEE, 512).unwrap();
        let right = metric.sprinkle_fixed_cardinality(0xC0FFEE, 512).unwrap();
        assert_eq!(left.events(), right.events());
        assert_eq!(left.len(), 512);
        assert!(left.events().iter().all(|event| {
            (0.0..=2.0).contains(&event.u) && (0.0..=2.0).contains(&event.v)
        }));
    }

    #[test]
    fn conformal_curvature_changes_volume_but_not_causal_order_rule() {
        let flat = ConformalFrw1p1::new(0.0, 2.0).unwrap();
        let curved = ConformalFrw1p1::new(1.0, 2.0).unwrap();
        assert!(curved.total_volume() > flat.total_volume());

        let causet = curved.sprinkle_fixed_cardinality(1234, 128).unwrap();
        for first in causet.events() {
            for second in causet.events() {
                let eta_first = first.coordinate_time() - 1.0;
                let x_first = first.coordinate_space();
                let eta_second = second.coordinate_time() - 1.0;
                let x_second = second.coordinate_space();
                let expected = eta_second > eta_first
                    && eta_second - eta_first >= (x_second - x_first).abs();
                assert_eq!(first.causally_precedes(*second), expected);
            }
        }
    }

    #[test]
    fn proper_time_integral_has_flat_limit_and_is_monotone() {
        let flat = ConformalFrw1p1::new(0.0, 2.0).unwrap();
        let curved = ConformalFrw1p1::new(0.8, 2.0).unwrap();
        let flat_duration = flat.comoving_proper_time(-1.0, 1.0).unwrap();
        let curved_duration = curved.comoving_proper_time(-1.0, 1.0).unwrap();
        assert!((flat_duration - 2.0).abs() < 1.0e-12);
        assert!(curved_duration > flat_duration);
        assert!(curved.comoving_proper_time(0.5, -0.5).is_none());
    }

    #[test]
    fn invalid_metric_parameters_fail_closed() {
        assert_eq!(
            ConformalFrw1p1::new(-0.1, 1.0).unwrap_err(),
            CurvedFrwError::InvalidCurvature
        );
        assert_eq!(
            ConformalFrw1p1::new(0.1, 0.0).unwrap_err(),
            CurvedFrwError::InvalidProperTime
        );
        assert_eq!(
            ConformalFrw1p1::new(1000.0, 2.0).unwrap_err(),
            CurvedFrwError::ExponentialOverflow
        );
    }
}
