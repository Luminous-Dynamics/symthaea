// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Rectangular Wilson-loop observables on explicit SU(3) gauge fields.
//!
//! Unlike the phenomenological area-law helper in `lattice_qcd`, this module
//! transports the actual lattice links around closed rectangles and measures
//! the normalized fundamental trace. The default contract is deliberately
//! non-winding: each rectangle side must be strictly shorter than the periodic
//! extent in that direction so a static-potential observable cannot silently
//! become a Polyakov/winding loop.

use crate::lattice_gauge::{
    LatticeGaugeError, Site4, WilsonGaugeField, su3_dagger, su3_identity, su3_mul, su3_trace,
};
use crate::symmetry_groups::Complex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RectangularWilsonLoopSpec {
    pub mu: usize,
    pub nu: usize,
    pub mu_length: usize,
    pub nu_length: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WilsonObservableError {
    Gauge(LatticeGaugeError),
    SameDirection(usize),
    InvalidLength { direction: usize, length: usize },
    WindingRectangle {
        direction: usize,
        length: usize,
        extent: usize,
    },
    InvalidTemporalSpacing(f64),
    InvalidLoopExpectation(f64),
}

impl From<LatticeGaugeError> for WilsonObservableError {
    fn from(value: LatticeGaugeError) -> Self {
        Self::Gauge(value)
    }
}

impl RectangularWilsonLoopSpec {
    pub fn validate(&self, field: &WilsonGaugeField) -> Result<(), WilsonObservableError> {
        if self.mu >= 4 {
            return Err(LatticeGaugeError::InvalidDirection(self.mu).into());
        }
        if self.nu >= 4 {
            return Err(LatticeGaugeError::InvalidDirection(self.nu).into());
        }
        if self.mu == self.nu {
            return Err(WilsonObservableError::SameDirection(self.mu));
        }
        if self.mu_length == 0 {
            return Err(WilsonObservableError::InvalidLength {
                direction: self.mu,
                length: 0,
            });
        }
        if self.nu_length == 0 {
            return Err(WilsonObservableError::InvalidLength {
                direction: self.nu,
                length: 0,
            });
        }
        let dims = field.dims();
        for (direction, length) in [(self.mu, self.mu_length), (self.nu, self.nu_length)] {
            if length >= dims[direction] {
                return Err(WilsonObservableError::WindingRectangle {
                    direction,
                    length,
                    extent: dims[direction],
                });
            }
        }
        Ok(())
    }
}

/// Normalized fundamental Wilson loop `Tr W / 3` for one contractible rectangle.
pub fn rectangular_wilson_loop(
    field: &WilsonGaugeField,
    start: Site4,
    spec: RectangularWilsonLoopSpec,
) -> Result<Complex, WilsonObservableError> {
    spec.validate(field)?;
    // Validate the starting site even before the first path step.
    field.link(start, spec.mu)?;

    let mut product = su3_identity();
    let mut site = start;

    for _ in 0..spec.mu_length {
        product = su3_mul(&product, field.link(site, spec.mu)?);
        site = field.shift(site, spec.mu, 1)?;
    }
    for _ in 0..spec.nu_length {
        product = su3_mul(&product, field.link(site, spec.nu)?);
        site = field.shift(site, spec.nu, 1)?;
    }
    for _ in 0..spec.mu_length {
        site = field.shift(site, spec.mu, -1)?;
        product = su3_mul(&product, &su3_dagger(field.link(site, spec.mu)?));
    }
    for _ in 0..spec.nu_length {
        site = field.shift(site, spec.nu, -1)?;
        product = su3_mul(&product, &su3_dagger(field.link(site, spec.nu)?));
    }

    debug_assert_eq!(site, start);
    let trace = su3_trace(&product);
    Ok(Complex::new(trace.re / 3.0, trace.im / 3.0))
}

/// Average one rectangle over every lattice origin in its declared plane.
pub fn average_rectangular_wilson_loop(
    field: &WilsonGaugeField,
    spec: RectangularWilsonLoopSpec,
) -> Result<Complex, WilsonObservableError> {
    spec.validate(field)?;
    let dims = field.dims();
    let mut re = 0.0;
    let mut im = 0.0;
    let mut count = 0usize;
    for x in 0..dims[0] {
        for y in 0..dims[1] {
            for z in 0..dims[2] {
                for t in 0..dims[3] {
                    let value = rectangular_wilson_loop(field, [x, y, z, t], spec)?;
                    re += value.re;
                    im += value.im;
                    count += 1;
                }
            }
        }
    }
    Ok(Complex::new(re / count as f64, im / count as f64))
}

/// Average a spatial-temporal rectangle over all three spatial orientations and
/// all lattice origins. Temporal direction is fixed to axis 3.
pub fn average_spatial_temporal_wilson_loop(
    field: &WilsonGaugeField,
    spatial_length: usize,
    temporal_length: usize,
) -> Result<Complex, WilsonObservableError> {
    let mut re = 0.0;
    let mut im = 0.0;
    for spatial_direction in 0..3 {
        let value = average_rectangular_wilson_loop(
            field,
            RectangularWilsonLoopSpec {
                mu: spatial_direction,
                nu: 3,
                mu_length: spatial_length,
                nu_length: temporal_length,
            },
        )?;
        re += value.re;
        im += value.im;
    }
    Ok(Complex::new(re / 3.0, im / 3.0))
}

/// Effective static potential from two positive ensemble-averaged Wilson loops:
/// `V_eff(R,T) = log[W(R,T) / W(R,T+1)] / a_t`.
///
/// This is estimator algebra only. It does not identify a plateau or perform a
/// correlated fit.
pub fn effective_static_potential(
    w_t: f64,
    w_t_plus_one: f64,
    temporal_spacing: f64,
) -> Result<f64, WilsonObservableError> {
    if !temporal_spacing.is_finite() || temporal_spacing <= 0.0 {
        return Err(WilsonObservableError::InvalidTemporalSpacing(
            temporal_spacing,
        ));
    }
    for value in [w_t, w_t_plus_one] {
        if !value.is_finite() || value <= 0.0 {
            return Err(WilsonObservableError::InvalidLoopExpectation(value));
        }
    }
    Ok((w_t / w_t_plus_one).ln() / temporal_spacing)
}

/// Creutz ratio
/// `chi(R,T) = -log[W(R,T) W(R-1,T-1) / (W(R,T-1) W(R-1,T))]`.
///
/// For an exact area law `W(R,T)=exp(-sigma R T)` this returns `sigma` in
/// lattice units. Physical string-tension extraction requires ensemble,
/// finite-volume, scale-setting and continuum evidence outside this function.
pub fn creutz_ratio(
    w_r_t: f64,
    w_r_minus_one_t_minus_one: f64,
    w_r_t_minus_one: f64,
    w_r_minus_one_t: f64,
) -> Result<f64, WilsonObservableError> {
    for value in [
        w_r_t,
        w_r_minus_one_t_minus_one,
        w_r_t_minus_one,
        w_r_minus_one_t,
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(WilsonObservableError::InvalidLoopExpectation(value));
        }
    }
    Ok(-((w_r_t * w_r_minus_one_t_minus_one)
        / (w_r_t_minus_one * w_r_minus_one_t))
        .ln())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_gauge::su3_diagonal;

    fn spec(r: usize, t: usize) -> RectangularWilsonLoopSpec {
        RectangularWilsonLoopSpec {
            mu: 0,
            nu: 1,
            mu_length: r,
            nu_length: t,
        }
    }

    #[test]
    fn identity_rectangles_match_independent_oracle() {
        let field = WilsonGaugeField::identity([3, 3, 2, 2]).unwrap();
        for shape in [(1, 1), (2, 1), (1, 2), (2, 2)] {
            let value = average_rectangular_wilson_loop(&field, spec(shape.0, shape.1)).unwrap();
            assert!((value.re - 1.0).abs() < 1.0e-14);
            assert!(value.im.abs() < 1.0e-14);
        }
    }

    #[test]
    fn localized_flux_matches_independent_oracle() {
        let mut field = WilsonGaugeField::identity([3, 3, 2, 2]).unwrap();
        field
            .set_link([0, 0, 0, 0], 0, su3_diagonal(0.3, -0.1))
            .unwrap();

        let local = rectangular_wilson_loop(&field, [0, 0, 0, 0], spec(1, 1)).unwrap();
        assert!((local.re - 0.976_802_410_748_291_2).abs() < 1.0e-14);
        assert!((local.im + 0.000_994_180_260_183_265_7).abs() < 1.0e-14);

        let expected = [
            ((1, 1), 0.998_711_245_041_571_7),
            ((2, 1), 0.997_422_490_083_143_5),
            ((1, 2), 0.998_711_245_041_571_7),
            ((2, 2), 0.997_422_490_083_143_5),
        ];
        for ((r, t), target) in expected {
            let value = average_rectangular_wilson_loop(&field, spec(r, t)).unwrap();
            assert!((value.re - target).abs() < 1.0e-14);
        }
    }

    #[test]
    fn area_law_creutz_ratio_recovers_injected_sigma() {
        let sigma = 0.23;
        let area = |r: f64, t: f64| (-sigma * r * t).exp();
        let chi = creutz_ratio(
            area(2.0, 2.0),
            area(1.0, 1.0),
            area(2.0, 1.0),
            area(1.0, 2.0),
        )
        .unwrap();
        assert!((chi - sigma).abs() < 1.0e-14);
    }

    #[test]
    fn exponential_time_dependence_recovers_injected_potential() {
        let potential = 0.41;
        let w3 = (-potential * 3.0f64).exp();
        let w4 = (-potential * 4.0f64).exp();
        let estimate = effective_static_potential(w3, w4, 1.0).unwrap();
        assert!((estimate - potential).abs() < 1.0e-14);
    }

    #[test]
    fn contractible_api_rejects_winding_rectangles() {
        let field = WilsonGaugeField::identity([3, 3, 2, 2]).unwrap();
        assert!(matches!(
            average_rectangular_wilson_loop(&field, spec(3, 1)),
            Err(WilsonObservableError::WindingRectangle {
                direction: 0,
                length: 3,
                extent: 3,
            })
        ));
    }

    #[test]
    fn estimator_algebra_fails_closed_on_nonpositive_inputs() {
        assert!(matches!(
            effective_static_potential(1.0, 0.0, 1.0),
            Err(WilsonObservableError::InvalidLoopExpectation(0.0))
        ));
        assert!(matches!(
            creutz_ratio(1.0, 1.0, -1.0, 1.0),
            Err(WilsonObservableError::InvalidLoopExpectation(-1.0))
        ));
    }
}
