// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Analytical reference profiles for finite-volume causal-set link sampling.
//!
//! Spatial dimension d is 1, 2, or 3; spacetime dimension is D=d+1.
//! This module distinguishes:
//! - the sigma-finite Lorentz-invariant link-intensity profile per unit
//!   hyperbolic volume;
//! - the normalized distribution induced by a finite constant-time cone;
//! - the normalized distribution induced by an event-centred product window.
//!
//! These are different estimands. A finite rapidity cutoff is a preferred-frame
//! numerical regularization, not an intrinsic Lorentz-invariant truncation.
//! This module contains analytic probability profiles, not a causal-set
//! sprinkling or exact link-empty-interval simulator.

use std::error::Error;
use std::f64::consts::PI;
use std::fmt;

/// Number of spatial dimensions supported by the link-profile reference.
pub const MIN_LINK_SPATIAL_DIMENSION: usize = 1;
pub const MAX_LINK_SPATIAL_DIMENSION: usize = 3;

/// The analytic quadrature is truncated only where the exponential tail is
/// below exp(-16^2) even in the slowest-decaying supported dimension.
const MAX_DIMENSIONLESS_TIME_FOR_QUADRATURE: f64 = 16.0;
const SIMPSON_PANELS: usize = 16_384;

/// A conservative finite numerical proxy for the paper's asymptotic condition
/// "much greater than one". This is a declared benchmark convention, not a
/// universal physical constant or a value specified by the paper.
pub const MIN_ROOT_ADMISSIBILITY_SCALE: f64 = 100.0;

/// Invalid dimension, scale or numerical regime in the link-sampling model.
#[derive(Clone, Debug, PartialEq)]
pub enum LinkSamplingBiasError {
    UnsupportedSpatialDimension { dimension: usize },
    InvalidDensity,
    InvalidTimeExtent,
    InvalidDimensionlessTime,
    InvalidRapidity,
    InvalidWindowFraction,
    AsymptoticRegimeNotReached,
    NumericalOverflow,
}

impl fmt::Display for LinkSamplingBiasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSpatialDimension { dimension } => {
                write!(f, "spatial dimension must be in 1..=3, got {dimension}")
            }
            Self::InvalidDensity => write!(f, "sprinkling density must be finite and positive"),
            Self::InvalidTimeExtent => write!(f, "time extent must be finite and positive"),
            Self::InvalidDimensionlessTime => {
                write!(
                    f,
                    "dimensionless proper time must be finite and non-negative"
                )
            }
            Self::InvalidRapidity => write!(f, "rapidity must be finite and non-negative"),
            Self::InvalidWindowFraction => {
                write!(
                    f,
                    "window fractions mu and eta must both lie strictly between zero and one"
                )
            }
            Self::AsymptoticRegimeNotReached => write!(
                f,
                "rho * T^(d+1) must exceed one for the double-scaling rapidity cutoff"
            ),
            Self::NumericalOverflow => write!(f, "link-sampling calculation overflowed"),
        }
    }
}

impl Error for LinkSamplingBiasError {}

/// Alexandrov-interval volume coefficient v_d for spatial dimension d.
///
/// The interval volume between timelike-separated endpoints with proper-time
/// separation tau is V = v_d * tau^(d+1). The exact supported values are:
/// d=1: 1/2, d=2: pi/12, d=3: pi/24.
pub fn alexandrov_volume_coefficient(
    spatial_dimension: usize,
) -> Result<f64, LinkSamplingBiasError> {
    match spatial_dimension {
        1 => Ok(0.5),
        2 => Ok(PI / 12.0),
        3 => Ok(PI / 24.0),
        dimension => Err(LinkSamplingBiasError::UnsupportedSpatialDimension { dimension }),
    }
}

/// Link intensity per unit hyperbolic volume as a density in dimensionless
/// proper time tau_hat = tau * (rho * v_d)^(1/(d+1)).
///
/// This is not a probability distribution over the complete infinite set of
/// links: that set has infinite hyperbolic volume and no uniform probability
/// measure. The returned profile integrates to one as a tau_hat profile per
/// unit hyperbolic volume.
pub fn link_intensity_profile(
    spatial_dimension: usize,
    dimensionless_proper_time: f64,
) -> Result<f64, LinkSamplingBiasError> {
    validate_dimension(spatial_dimension)?;
    validate_dimensionless_time(dimensionless_proper_time)?;
    if dimensionless_proper_time == 0.0 {
        return Ok(0.0);
    }
    let exponent = dimensionless_proper_time.powi((spatial_dimension + 1) as i32);
    let density = (spatial_dimension + 1) as f64
        * dimensionless_proper_time.powi(spatial_dimension as i32)
        * (-exponent).exp();
    if density.is_finite() {
        Ok(density)
    } else {
        Err(LinkSamplingBiasError::NumericalOverflow)
    }
}

/// Hyperbolic d-ball volume V_H^d(chi), including the area of S^(d-1).
///
/// d=1: 2 chi
/// d=2: 2 pi (cosh(chi)-1)
/// d=3: 2 pi (sinh(chi) cosh(chi)-chi)
///
/// The volume can overflow for large rapidity; use
/// log_hyperbolic_ball_volume when a log-domain value is needed.
pub fn hyperbolic_ball_volume(
    spatial_dimension: usize,
    rapidity: f64,
) -> Result<f64, LinkSamplingBiasError> {
    let log_volume = log_hyperbolic_ball_volume(spatial_dimension, rapidity)?;
    let volume = log_volume.exp();
    if volume.is_finite() {
        Ok(volume)
    } else {
        Err(LinkSamplingBiasError::NumericalOverflow)
    }
}

/// Logarithm of the hyperbolic d-ball volume, stable for large rapidities.
pub fn log_hyperbolic_ball_volume(
    spatial_dimension: usize,
    rapidity: f64,
) -> Result<f64, LinkSamplingBiasError> {
    validate_dimension(spatial_dimension)?;
    if !rapidity.is_finite() || rapidity < 0.0 {
        return Err(LinkSamplingBiasError::InvalidRapidity);
    }
    if rapidity == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }

    let log_volume = match spatial_dimension {
        1 => (2.0 * rapidity).ln(),
        2 if rapidity < 1.0e-4 => {
            // V_2 = pi chi^2 * (1 + chi^2/12 + O(chi^4)).
            PI.ln() + 2.0 * rapidity.ln() + (rapidity * rapidity / 12.0).ln_1p()
        }
        2 if rapidity < 40.0 => (2.0 * PI * (rapidity.cosh() - 1.0)).ln(),
        2 => {
            // cosh(chi)-1 = exp(chi)/2 * (1-exp(-chi))^2.
            PI.ln() + rapidity + 2.0 * (-(-rapidity).exp_m1()).ln()
        }
        3 if rapidity < 1.0e-3 => {
            // sinh(chi)cosh(chi)-chi =
            // 2/3 chi^3 + 2/15 chi^5 + 4/315 chi^7 + O(chi^9).
            let chi2 = rapidity * rapidity;
            let series =
                (2.0 / 3.0) * chi2 * rapidity * (1.0 + chi2 / 5.0 + 2.0 * chi2 * chi2 / 105.0);
            (2.0 * PI * series).ln()
        }
        3 if rapidity < 20.0 => (2.0 * PI * (rapidity.sinh() * rapidity.cosh() - rapidity)).ln(),
        3 => {
            // sinh(chi)cosh(chi)-chi =
            // exp(2 chi)/4 * (1-exp(-4chi)-4chi exp(-2chi)).
            let correction =
                1.0 - (-4.0 * rapidity).exp() - 4.0 * rapidity * (-2.0 * rapidity).exp();
            (PI / 2.0).ln() + 2.0 * rapidity + correction.ln()
        }
        _ => unreachable!("dimension validated above"),
    };

    if log_volume.is_finite() || log_volume == f64::NEG_INFINITY {
        Ok(log_volume)
    } else {
        Err(LinkSamplingBiasError::NumericalOverflow)
    }
}

/// Finite-cone proper-time distribution from a root event.
///
/// The finite cone is the future cone truncated by a constant coordinate-time
/// plane t<T. In dimensionless proper time x=tau_hat, its unnormalized density
/// is x^d exp(-x^(d+1)) V_H^d(acosh(T_hat/x)), for 0<x<T_hat, where
/// T_hat=T*(rho*v_d)^(1/(d+1)).
///
/// The normalization and mean are evaluated by deterministic log-domain
/// Simpson quadrature. The quadrature omits only the tail x>16, whose
/// exponential factor is less than exp(-256) for all supported dimensions.
#[derive(Clone, Debug, PartialEq)]
pub struct FiniteConeLinkProfile {
    spatial_dimension: usize,
    dimensionless_time_extent: f64,
    log_normalization: f64,
    mean_dimensionless_proper_time: f64,
}

impl FiniteConeLinkProfile {
    pub fn new(
        spatial_dimension: usize,
        dimensionless_time_extent: f64,
    ) -> Result<Self, LinkSamplingBiasError> {
        validate_dimension(spatial_dimension)?;
        if !dimensionless_time_extent.is_finite() || dimensionless_time_extent <= 0.0 {
            return Err(LinkSamplingBiasError::InvalidTimeExtent);
        }

        let upper = dimensionless_time_extent.min(MAX_DIMENSIONLESS_TIME_FOR_QUADRATURE);
        let log_unnormalized =
            |x: f64| finite_cone_log_weight(spatial_dimension, dimensionless_time_extent, x);
        let log_normalization =
            log_simpson_integral_fallible(upper, SIMPSON_PANELS, &log_unnormalized)?;
        let log_first_moment = log_simpson_integral_fallible(upper, SIMPSON_PANELS, &|x| {
            if x <= 0.0 {
                Ok(f64::NEG_INFINITY)
            } else {
                Ok(log_unnormalized(x)? + x.ln())
            }
        })?;
        let mean = (log_first_moment - log_normalization).exp();
        if !mean.is_finite() {
            return Err(LinkSamplingBiasError::NumericalOverflow);
        }

        Ok(Self {
            spatial_dimension,
            dimensionless_time_extent,
            log_normalization,
            mean_dimensionless_proper_time: mean,
        })
    }

    pub fn spatial_dimension(&self) -> usize {
        self.spatial_dimension
    }

    pub fn dimensionless_time_extent(&self) -> f64 {
        self.dimensionless_time_extent
    }

    pub fn mean_dimensionless_proper_time(&self) -> f64 {
        self.mean_dimensionless_proper_time
    }

    pub fn density(&self, dimensionless_proper_time: f64) -> Result<f64, LinkSamplingBiasError> {
        validate_dimensionless_time(dimensionless_proper_time)?;
        if dimensionless_proper_time == 0.0
            || dimensionless_proper_time >= self.dimensionless_time_extent
        {
            return Ok(0.0);
        }
        let log_density = finite_cone_log_weight(
            self.spatial_dimension,
            self.dimensionless_time_extent,
            dimensionless_proper_time,
        )? - self.log_normalization;
        let density = log_density.exp();
        if density.is_finite() {
            Ok(density)
        } else {
            Err(LinkSamplingBiasError::NumericalOverflow)
        }
    }
}

/// Event-centred product-window truncation of the link-intensity profile.
///
/// The cutoffs in rapidity and proper time are independent. The finite rapidity
/// window introduces a declared preferred timelike frame; it is a numerical
/// sampling window, not an intrinsic Lorentz-invariant truncation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProductWindowLinkProfile {
    spatial_dimension: usize,
    dimensionless_proper_time_cutoff: f64,
}

impl ProductWindowLinkProfile {
    pub fn new(
        spatial_dimension: usize,
        dimensionless_proper_time_cutoff: f64,
    ) -> Result<Self, LinkSamplingBiasError> {
        validate_dimension(spatial_dimension)?;
        if !dimensionless_proper_time_cutoff.is_finite() || dimensionless_proper_time_cutoff <= 0.0
        {
            return Err(LinkSamplingBiasError::InvalidDimensionlessTime);
        }
        Ok(Self {
            spatial_dimension,
            dimensionless_proper_time_cutoff,
        })
    }

    pub fn dimensionless_proper_time_cutoff(&self) -> f64 {
        self.dimensionless_proper_time_cutoff
    }

    /// Normalized density in dimensionless proper time, truncated at x_c.
    pub fn density(&self, dimensionless_proper_time: f64) -> Result<f64, LinkSamplingBiasError> {
        validate_dimensionless_time(dimensionless_proper_time)?;
        let cutoff = self.dimensionless_proper_time_cutoff;
        if dimensionless_proper_time == 0.0 || dimensionless_proper_time >= cutoff {
            return Ok(0.0);
        }

        let exponent = cutoff.powi((self.spatial_dimension + 1) as i32);
        let normalization = if exponent > 745.0 {
            1.0
        } else {
            -(-exponent).exp_m1()
        };
        let density = link_intensity_profile(self.spatial_dimension, dimensionless_proper_time)?
            / normalization;
        if density.is_finite() {
            Ok(density)
        } else {
            Err(LinkSamplingBiasError::NumericalOverflow)
        }
    }

    pub fn mean_dimensionless_proper_time(&self) -> Result<f64, LinkSamplingBiasError> {
        let cutoff = self
            .dimensionless_proper_time_cutoff
            .min(MAX_DIMENSIONLESS_TIME_FOR_QUADRATURE);
        let d = self.spatial_dimension;
        let log_first_moment = log_simpson_integral(cutoff, SIMPSON_PANELS, &|x| {
            if x <= 0.0 {
                f64::NEG_INFINITY
            } else {
                ((d + 1) as f64).ln() + (d + 1) as f64 * x.ln() - x.powi((d + 1) as i32)
            }
        })?;
        let tail_exponent = self.dimensionless_proper_time_cutoff.powi((d + 1) as i32);
        let log_normalization = if tail_exponent > 745.0 {
            0.0
        } else {
            (-(-tail_exponent).exp_m1()).ln()
        };
        let mean = (log_first_moment - log_normalization).exp();
        if mean.is_finite() {
            Ok(mean)
        } else {
            Err(LinkSamplingBiasError::NumericalOverflow)
        }
    }
}

/// Result of the paper's controlled double-scaling product-window prescription.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoubleScalingProductWindow {
    pub spatial_dimension: usize,
    pub density: f64,
    pub ambient_time_extent: f64,
    pub remaining_future_time: f64,
    pub mu: f64,
    pub eta: f64,
    pub rapidity_cutoff: f64,
    pub proper_time_cutoff: f64,
    pub dimensionless_proper_time_cutoff: f64,
    /// Log of the dimensionless root-admissibility scale:
    /// (DeltaT/T)^(d+1) * (rho*T^(d+1))^(1-mu).
    pub log_root_admissibility_scale: f64,
    /// True when the declared numerical threshold (100) is reached.
    pub asymptotically_admissible: bool,
}

/// Compute the paper's double-scaling cutoffs for a root with remaining
/// coordinate time DeltaT_x in an ambient future cone of extent T.
///
/// rho has units length^(-(d+1)); T and DeltaT_x have units of length.
/// The formula is asymptotic and therefore requires rho*T^(d+1)>1.
pub fn double_scaling_product_window(
    spatial_dimension: usize,
    density: f64,
    ambient_time_extent: f64,
    remaining_future_time: f64,
    mu: f64,
    eta: f64,
) -> Result<DoubleScalingProductWindow, LinkSamplingBiasError> {
    validate_dimension(spatial_dimension)?;
    if !density.is_finite() || density <= 0.0 {
        return Err(LinkSamplingBiasError::InvalidDensity);
    }
    if !ambient_time_extent.is_finite()
        || ambient_time_extent <= 0.0
        || !remaining_future_time.is_finite()
        || remaining_future_time <= 0.0
        || remaining_future_time > ambient_time_extent
    {
        return Err(LinkSamplingBiasError::InvalidTimeExtent);
    }
    if !mu.is_finite() || !eta.is_finite() || mu <= 0.0 || mu >= 1.0 || eta <= 0.0 || eta >= 1.0 {
        return Err(LinkSamplingBiasError::InvalidWindowFraction);
    }

    let dimension = (spatial_dimension + 1) as f64;
    let log_density_volume = density.ln() + dimension * ambient_time_extent.ln();
    if !log_density_volume.is_finite() || log_density_volume <= 0.0 {
        return Err(LinkSamplingBiasError::AsymptoticRegimeNotReached);
    }

    let rapidity_cutoff = mu / dimension * log_density_volume;
    let cosh_rapidity = rapidity_cutoff.cosh();
    if !cosh_rapidity.is_finite() || cosh_rapidity <= 0.0 {
        return Err(LinkSamplingBiasError::NumericalOverflow);
    }
    let proper_time_cutoff = eta * remaining_future_time / cosh_rapidity;
    let volume_coefficient = alexandrov_volume_coefficient(spatial_dimension)?;
    let dimensionless_proper_time_cutoff =
        proper_time_cutoff * (density * volume_coefficient).powf(1.0 / dimension);
    let log_root_admissibility_scale = dimension
        * (remaining_future_time / ambient_time_extent).ln()
        + (1.0 - mu) * log_density_volume;

    if !proper_time_cutoff.is_finite()
        || !dimensionless_proper_time_cutoff.is_finite()
        || !log_root_admissibility_scale.is_finite()
    {
        return Err(LinkSamplingBiasError::NumericalOverflow);
    }

    Ok(DoubleScalingProductWindow {
        spatial_dimension,
        density,
        ambient_time_extent,
        remaining_future_time,
        mu,
        eta,
        rapidity_cutoff,
        proper_time_cutoff,
        dimensionless_proper_time_cutoff,
        log_root_admissibility_scale,
        asymptotically_admissible: log_root_admissibility_scale
            >= MIN_ROOT_ADMISSIBILITY_SCALE.ln(),
    })
}

fn finite_cone_log_weight(
    spatial_dimension: usize,
    dimensionless_time_extent: f64,
    dimensionless_proper_time: f64,
) -> Result<f64, LinkSamplingBiasError> {
    if dimensionless_proper_time <= 0.0 || dimensionless_proper_time >= dimensionless_time_extent {
        return Ok(f64::NEG_INFINITY);
    }
    let rapidity = (dimensionless_time_extent / dimensionless_proper_time).acosh();
    let log_volume = log_hyperbolic_ball_volume(spatial_dimension, rapidity)?;
    let exponent = dimensionless_proper_time.powi((spatial_dimension + 1) as i32);
    Ok(spatial_dimension as f64 * dimensionless_proper_time.ln() - exponent + log_volume)
}

fn log_simpson_integral<F>(
    upper: f64,
    panels: usize,
    log_function: &F,
) -> Result<f64, LinkSamplingBiasError>
where
    F: Fn(f64) -> f64,
{
    if !upper.is_finite() || upper <= 0.0 || panels < 2 || panels % 2 != 0 {
        return Err(LinkSamplingBiasError::InvalidDimensionlessTime);
    }

    let step = upper / panels as f64;
    let first = log_function(0.0);
    let last = log_function(upper);
    if first.is_nan() || first == f64::INFINITY || last.is_nan() || last == f64::INFINITY {
        return Err(LinkSamplingBiasError::NumericalOverflow);
    }
    let mut maximum_log = first.max(last);

    for index in 1..panels {
        let value = log_function(index as f64 * step);
        if value.is_nan() || value == f64::INFINITY {
            return Err(LinkSamplingBiasError::NumericalOverflow);
        }
        maximum_log = maximum_log.max(value);
    }
    if !maximum_log.is_finite() {
        return Err(LinkSamplingBiasError::NumericalOverflow);
    }

    let mut weighted_sum = (first - maximum_log).exp() + (last - maximum_log).exp();
    for index in 1..panels {
        let x = index as f64 * step;
        let log_value = log_function(x);
        let coefficient = if index % 2 == 0 { 2.0 } else { 4.0 };
        weighted_sum += coefficient * (log_value - maximum_log).exp();
    }

    let scaled_integral = (step / 3.0) * weighted_sum;
    let log_integral = maximum_log + scaled_integral.ln();
    if log_integral.is_finite() {
        Ok(log_integral)
    } else {
        Err(LinkSamplingBiasError::NumericalOverflow)
    }
}

fn log_simpson_integral_fallible<F>(
    upper: f64,
    panels: usize,
    log_function: &F,
) -> Result<f64, LinkSamplingBiasError>
where
    F: Fn(f64) -> Result<f64, LinkSamplingBiasError>,
{
    if !upper.is_finite() || upper <= 0.0 || panels < 2 || panels % 2 != 0 {
        return Err(LinkSamplingBiasError::InvalidDimensionlessTime);
    }

    let step = upper / panels as f64;
    let first = log_function(0.0)?;
    let last = log_function(upper)?;
    if first.is_nan() || first == f64::INFINITY || last.is_nan() || last == f64::INFINITY {
        return Err(LinkSamplingBiasError::NumericalOverflow);
    }
    let mut maximum_log = first.max(last);

    for index in 1..panels {
        let value = log_function(index as f64 * step)?;
        if value.is_nan() || value == f64::INFINITY {
            return Err(LinkSamplingBiasError::NumericalOverflow);
        }
        maximum_log = maximum_log.max(value);
    }
    if !maximum_log.is_finite() {
        return Err(LinkSamplingBiasError::NumericalOverflow);
    }

    let mut weighted_sum = (first - maximum_log).exp() + (last - maximum_log).exp();
    for index in 1..panels {
        let value = log_function(index as f64 * step)?;
        let coefficient = if index % 2 == 0 { 2.0 } else { 4.0 };
        weighted_sum += coefficient * (value - maximum_log).exp();
    }

    let scaled_integral = (step / 3.0) * weighted_sum;
    let log_integral = maximum_log + scaled_integral.ln();
    if log_integral.is_finite() {
        Ok(log_integral)
    } else {
        Err(LinkSamplingBiasError::NumericalOverflow)
    }
}

fn validate_dimension(dimension: usize) -> Result<(), LinkSamplingBiasError> {
    if (MIN_LINK_SPATIAL_DIMENSION..=MAX_LINK_SPATIAL_DIMENSION).contains(&dimension) {
        Ok(())
    } else {
        Err(LinkSamplingBiasError::UnsupportedSpatialDimension { dimension })
    }
}

fn validate_dimensionless_time(value: f64) -> Result<(), LinkSamplingBiasError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(LinkSamplingBiasError::InvalidDimensionlessTime)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alexandrov_volume_coefficients_match_closed_forms() {
        assert!((alexandrov_volume_coefficient(1).unwrap() - 0.5).abs() < 1.0e-15);
        assert!((alexandrov_volume_coefficient(2).unwrap() - PI / 12.0).abs() < 1.0e-15);
        assert!((alexandrov_volume_coefficient(3).unwrap() - PI / 24.0).abs() < 1.0e-15);
    }

    #[test]
    fn hyperbolic_ball_volumes_match_closed_forms() {
        let chi: f64 = 1.2;
        assert!((hyperbolic_ball_volume(1, chi).unwrap() - 2.0 * chi).abs() < 1.0e-14);
        assert!(
            (hyperbolic_ball_volume(2, chi).unwrap() - 2.0 * PI * (chi.cosh() - 1.0)).abs()
                < 1.0e-13
        );
        assert!(
            (hyperbolic_ball_volume(3, chi).unwrap() - 2.0 * PI * (chi.sinh() * chi.cosh() - chi))
                .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn link_intensity_profile_is_normalized_in_all_supported_dimensions() {
        for dimension in 1..=3 {
            let integral = log_simpson_integral(
                MAX_DIMENSIONLESS_TIME_FOR_QUADRATURE,
                SIMPSON_PANELS,
                &|x| {
                    ((dimension + 1) as f64).ln() + dimension as f64 * x.ln()
                        - x.powi((dimension + 1) as i32)
                },
            )
            .unwrap()
            .exp();
            assert!(
                (integral - 1.0).abs() < 1.0e-10,
                "d={dimension}: {integral}"
            );
        }
    }

    #[test]
    fn product_window_profile_recovers_the_link_intensity_mean() {
        let expected = [
            0.886_226_925_452_758,
            0.892_979_511_569_249,
            0.906_402_477_055_477,
        ];
        for dimension in 1..=3 {
            let profile = ProductWindowLinkProfile::new(dimension, 10.0).unwrap();
            let mean = profile.mean_dimensionless_proper_time().unwrap();
            assert!(
                (mean - expected[dimension - 1]).abs() < 2.0e-7,
                "d={dimension}: {mean} != {}",
                expected[dimension - 1]
            );
        }
    }

    #[test]
    fn finite_cone_profile_matches_an_independent_quadrature_fixture() {
        let expected_means = [0.844_486_322_56, 0.734_866_360_21, 0.688_128_286_46];
        let dimensionless_extents = [
            (128_000.0 * 0.5_f64).sqrt(),
            (128_000.0 * (PI / 12.0)).powf(1.0 / 3.0),
            (128_000.0 * (PI / 24.0)).powf(0.25),
        ];

        for dimension in 1..=3 {
            let profile =
                FiniteConeLinkProfile::new(dimension, dimensionless_extents[dimension - 1])
                    .unwrap();
            let mean = profile.mean_dimensionless_proper_time();
            assert!(
                (mean - expected_means[dimension - 1]).abs() < 2.0e-8,
                "d={dimension}: {mean} != {}",
                expected_means[dimension - 1]
            );
            // A normalized density integrates to one; use a coarser independent
            // trapezoid to avoid testing Simpson with itself.
            let upper = dimensionless_extents[dimension - 1].min(16.0);
            let step = upper / 20_000.0;
            let mut integral = 0.5 * profile.density(0.0).unwrap();
            for index in 1..20_000 {
                let x = index as f64 * step;
                integral += profile.density(x).unwrap();
            }
            integral += 0.5 * profile.density(upper).unwrap();
            integral *= step;
            assert!((integral - 1.0).abs() < 2.0e-5, "d={dimension}: {integral}");
        }
    }

    #[test]
    fn finite_cone_and_product_window_are_distinct_estimands() {
        let dimension = 3;
        let density = 128_000.0;
        let extent = 1.0;
        let coefficient = alexandrov_volume_coefficient(dimension).unwrap();
        let t_hat = extent * (density * coefficient).powf(1.0 / 4.0);
        let finite = FiniteConeLinkProfile::new(dimension, t_hat).unwrap();
        let product = ProductWindowLinkProfile::new(dimension, 10.0).unwrap();

        assert!(
            product.mean_dimensionless_proper_time().unwrap()
                > finite.mean_dimensionless_proper_time()
        );
    }

    #[test]
    fn double_scaling_cutoffs_match_the_declared_formula() {
        let window = double_scaling_product_window(3, 128_000.0, 1.0, 1.0, 0.6, 0.9).unwrap();
        let expected_rapidity = 0.6 / 4.0 * (128_000.0_f64).ln();
        assert!((window.rapidity_cutoff - expected_rapidity).abs() < 1.0e-14);
        assert!(window.proper_time_cutoff > 0.0);
        assert!(window.dimensionless_proper_time_cutoff > 1.0);
        assert!(window.asymptotically_admissible);
    }

    #[test]
    fn boundary_near_roots_fail_the_asymptotic_admissibility_gate() {
        let interior = double_scaling_product_window(3, 128_000.0, 1.0, 1.0, 0.6, 0.9).unwrap();
        let boundary_near =
            double_scaling_product_window(3, 128_000.0, 1.0, 1.0e-4, 0.6, 0.9).unwrap();
        assert!(interior.asymptotically_admissible);
        assert!(!boundary_near.asymptotically_admissible);
    }

    #[test]
    fn invalid_dimensions_scales_and_asymptotic_inputs_fail_closed() {
        assert!(matches!(
            alexandrov_volume_coefficient(4),
            Err(LinkSamplingBiasError::UnsupportedSpatialDimension { .. })
        ));
        assert!(matches!(
            link_intensity_profile(2, f64::NAN),
            Err(LinkSamplingBiasError::InvalidDimensionlessTime)
        ));
        assert!(matches!(
            FiniteConeLinkProfile::new(1, 0.0),
            Err(LinkSamplingBiasError::InvalidTimeExtent)
        ));
        assert!(matches!(
            double_scaling_product_window(2, 0.0, 1.0, 1.0, 0.6, 0.9),
            Err(LinkSamplingBiasError::InvalidDensity)
        ));
        assert!(matches!(
            double_scaling_product_window(2, 100.0, 1.0, 1.0, 0.0, 0.9),
            Err(LinkSamplingBiasError::InvalidWindowFraction)
        ));
        assert!(matches!(
            double_scaling_product_window(2, 0.1, 1.0, 1.0, 0.6, 0.9),
            Err(LinkSamplingBiasError::AsymptoticRegimeNotReached)
        ));
    }
}
