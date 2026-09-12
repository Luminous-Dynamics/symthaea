// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Exact small-lattice SU(3) gauge primitives.
//!
//! This module is the production-side counterpart to the independent
//! `scripts/lqcd_wilson_su3_oracle.py` reference implementation. It deliberately
//! stops at exact algebraic lattice semantics: periodic indexing, SU(3) link
//! validation, plaquettes, Wilson action, Polyakov loops, and local gauge
//! transformations.
//!
//! It is **not** an ensemble generator and carries no authority for continuum
//! physics, glueball masses, string tension, or other physical predictions.

use crate::symmetry_groups::Complex;

pub type Site4 = [usize; 4];
pub type Su3Matrix = [[Complex; 3]; 3];

#[derive(Debug, Clone, PartialEq)]
pub enum LatticeGaugeError {
    InvalidExtent([usize; 4]),
    InvalidDirection(usize),
    SamePlaquetteDirection(usize),
    SiteOutOfBounds(Site4),
    InvalidBeta(f64),
    InvalidGaugeTransformLength { expected: usize, actual: usize },
    InvalidSu3Link {
        unitarity_error: f64,
        determinant_error: f64,
    },
}

#[inline]
fn c_add(a: Complex, b: Complex) -> Complex {
    Complex::new(a.re + b.re, a.im + b.im)
}

#[inline]
fn c_sub(a: Complex, b: Complex) -> Complex {
    Complex::new(a.re - b.re, a.im - b.im)
}

#[inline]
fn c_mul(a: Complex, b: Complex) -> Complex {
    Complex::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

#[inline]
fn c_conj(a: Complex) -> Complex {
    Complex::new(a.re, -a.im)
}

#[inline]
fn c_abs(a: Complex) -> f64 {
    a.norm_sq().sqrt()
}

pub fn su3_identity() -> Su3Matrix {
    [
        [Complex::ONE, Complex::ZERO, Complex::ZERO],
        [Complex::ZERO, Complex::ONE, Complex::ZERO],
        [Complex::ZERO, Complex::ZERO, Complex::ONE],
    ]
}

/// Deterministic diagonal SU(3) element
/// `diag(e^{i theta}, e^{i phi}, e^{-i(theta+phi)})`.
pub fn su3_diagonal(theta: f64, phi: f64) -> Su3Matrix {
    let phases = [theta, phi, -(theta + phi)];
    let mut out = [[Complex::ZERO; 3]; 3];
    for i in 0..3 {
        out[i][i] = Complex::new(phases[i].cos(), phases[i].sin());
    }
    out
}

pub fn su3_mul(a: &Su3Matrix, b: &Su3Matrix) -> Su3Matrix {
    let mut out = [[Complex::ZERO; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = Complex::ZERO;
            for k in 0..3 {
                sum = c_add(sum, c_mul(a[i][k], b[k][j]));
            }
            out[i][j] = sum;
        }
    }
    out
}

pub fn su3_dagger(a: &Su3Matrix) -> Su3Matrix {
    let mut out = [[Complex::ZERO; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = c_conj(a[j][i]);
        }
    }
    out
}

pub fn su3_trace(a: &Su3Matrix) -> Complex {
    c_add(c_add(a[0][0], a[1][1]), a[2][2])
}

pub fn su3_determinant(a: &Su3Matrix) -> Complex {
    let minor_00 = c_sub(c_mul(a[1][1], a[2][2]), c_mul(a[1][2], a[2][1]));
    let minor_01 = c_sub(c_mul(a[1][0], a[2][2]), c_mul(a[1][2], a[2][0]));
    let minor_02 = c_sub(c_mul(a[1][0], a[2][1]), c_mul(a[1][1], a[2][0]));
    c_add(
        c_sub(c_mul(a[0][0], minor_00), c_mul(a[0][1], minor_01)),
        c_mul(a[0][2], minor_02),
    )
}

pub fn su3_unitarity_error(a: &Su3Matrix) -> f64 {
    let product = su3_mul(a, &su3_dagger(a));
    let identity = su3_identity();
    let mut max_err: f64 = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            max_err = max_err.max(c_abs(c_sub(product[i][j], identity[i][j])));
        }
    }
    max_err
}

pub fn su3_determinant_error(a: &Su3Matrix) -> f64 {
    c_abs(c_sub(su3_determinant(a), Complex::ONE))
}

pub fn validate_su3(a: &Su3Matrix, tolerance: f64) -> Result<(), LatticeGaugeError> {
    let unitarity_error = su3_unitarity_error(a);
    let determinant_error = su3_determinant_error(a);
    if !unitarity_error.is_finite()
        || !determinant_error.is_finite()
        || unitarity_error > tolerance
        || determinant_error > tolerance
    {
        return Err(LatticeGaugeError::InvalidSu3Link {
            unitarity_error,
            determinant_error,
        });
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct WilsonGaugeField {
    dims: [usize; 4],
    links: Vec<Su3Matrix>,
}

impl WilsonGaugeField {
    pub fn identity(dims: [usize; 4]) -> Result<Self, LatticeGaugeError> {
        if dims.iter().any(|&n| n == 0) {
            return Err(LatticeGaugeError::InvalidExtent(dims));
        }
        let site_count = dims
            .iter()
            .try_fold(1usize, |acc, &n| acc.checked_mul(n))
            .ok_or(LatticeGaugeError::InvalidExtent(dims))?;
        let link_count = site_count
            .checked_mul(4)
            .ok_or(LatticeGaugeError::InvalidExtent(dims))?;
        Ok(Self {
            dims,
            links: vec![su3_identity(); link_count],
        })
    }

    pub fn dims(&self) -> [usize; 4] {
        self.dims
    }

    pub fn site_count(&self) -> usize {
        self.dims.iter().product()
    }

    fn site_index(&self, site: Site4) -> Result<usize, LatticeGaugeError> {
        if site.iter().zip(self.dims).any(|(&x, n)| x >= n) {
            return Err(LatticeGaugeError::SiteOutOfBounds(site));
        }
        Ok((((site[0] * self.dims[1] + site[1]) * self.dims[2] + site[2])
            * self.dims[3])
            + site[3])
    }

    fn site_from_index(&self, mut index: usize) -> Site4 {
        let t = index % self.dims[3];
        index /= self.dims[3];
        let z = index % self.dims[2];
        index /= self.dims[2];
        let y = index % self.dims[1];
        index /= self.dims[1];
        let x = index;
        [x, y, z, t]
    }

    fn link_index(&self, site: Site4, mu: usize) -> Result<usize, LatticeGaugeError> {
        if mu >= 4 {
            return Err(LatticeGaugeError::InvalidDirection(mu));
        }
        Ok(self.site_index(site)? * 4 + mu)
    }

    pub fn shift(&self, site: Site4, mu: usize, step: isize) -> Result<Site4, LatticeGaugeError> {
        if mu >= 4 {
            return Err(LatticeGaugeError::InvalidDirection(mu));
        }
        self.site_index(site)?;
        let extent = self.dims[mu] as isize;
        let shifted = (site[mu] as isize + step).rem_euclid(extent) as usize;
        let mut out = site;
        out[mu] = shifted;
        Ok(out)
    }

    pub fn link(&self, site: Site4, mu: usize) -> Result<&Su3Matrix, LatticeGaugeError> {
        let index = self.link_index(site, mu)?;
        Ok(&self.links[index])
    }

    pub fn set_link(
        &mut self,
        site: Site4,
        mu: usize,
        value: Su3Matrix,
    ) -> Result<(), LatticeGaugeError> {
        validate_su3(&value, 1e-12)?;
        let index = self.link_index(site, mu)?;
        self.links[index] = value;
        Ok(())
    }

    pub fn plaquette(
        &self,
        site: Site4,
        mu: usize,
        nu: usize,
    ) -> Result<Su3Matrix, LatticeGaugeError> {
        if mu >= 4 {
            return Err(LatticeGaugeError::InvalidDirection(mu));
        }
        if nu >= 4 {
            return Err(LatticeGaugeError::InvalidDirection(nu));
        }
        if mu == nu {
            return Err(LatticeGaugeError::SamePlaquetteDirection(mu));
        }

        let x_plus_mu = self.shift(site, mu, 1)?;
        let x_plus_nu = self.shift(site, nu, 1)?;
        let u_mu_x = *self.link(site, mu)?;
        let u_nu_x_plus_mu = *self.link(x_plus_mu, nu)?;
        let u_mu_x_plus_nu = *self.link(x_plus_nu, mu)?;
        let u_nu_x = *self.link(site, nu)?;

        Ok(su3_mul(
            &su3_mul(
                &su3_mul(&u_mu_x, &u_nu_x_plus_mu),
                &su3_dagger(&u_mu_x_plus_nu),
            ),
            &su3_dagger(&u_nu_x),
        ))
    }

    pub fn average_plaquette(&self) -> Result<f64, LatticeGaugeError> {
        let mut sum = 0.0;
        let mut count = 0usize;
        for index in 0..self.site_count() {
            let site = self.site_from_index(index);
            for mu in 0..4 {
                for nu in (mu + 1)..4 {
                    sum += su3_trace(&self.plaquette(site, mu, nu)?).re / 3.0;
                    count += 1;
                }
            }
        }
        Ok(sum / count as f64)
    }

    pub fn wilson_action(&self, beta: f64) -> Result<f64, LatticeGaugeError> {
        if !beta.is_finite() || beta < 0.0 {
            return Err(LatticeGaugeError::InvalidBeta(beta));
        }
        let mut action = 0.0;
        for index in 0..self.site_count() {
            let site = self.site_from_index(index);
            for mu in 0..4 {
                for nu in (mu + 1)..4 {
                    let trace = su3_trace(&self.plaquette(site, mu, nu)?).re / 3.0;
                    action += beta * (1.0 - trace);
                }
            }
        }
        Ok(action)
    }

    pub fn polyakov_loop(&self, spatial: [usize; 3]) -> Result<Complex, LatticeGaugeError> {
        if spatial[0] >= self.dims[0]
            || spatial[1] >= self.dims[1]
            || spatial[2] >= self.dims[2]
        {
            return Err(LatticeGaugeError::SiteOutOfBounds([
                spatial[0],
                spatial[1],
                spatial[2],
                0,
            ]));
        }
        let mut product = su3_identity();
        for t in 0..self.dims[3] {
            product = su3_mul(&product, self.link([spatial[0], spatial[1], spatial[2], t], 3)?);
        }
        let trace = su3_trace(&product);
        Ok(Complex::new(trace.re / 3.0, trace.im / 3.0))
    }

    pub fn gauge_transform(
        &self,
        local_gauge: &[Su3Matrix],
    ) -> Result<Self, LatticeGaugeError> {
        if local_gauge.len() != self.site_count() {
            return Err(LatticeGaugeError::InvalidGaugeTransformLength {
                expected: self.site_count(),
                actual: local_gauge.len(),
            });
        }
        for g in local_gauge {
            validate_su3(g, 1e-12)?;
        }

        let mut out = Self::identity(self.dims)?;
        for index in 0..self.site_count() {
            let site = self.site_from_index(index);
            for mu in 0..4 {
                let x_plus_mu = self.shift(site, mu, 1)?;
                let next_index = self.site_index(x_plus_mu)?;
                let transformed = su3_mul(
                    &su3_mul(&local_gauge[index], self.link(site, mu)?),
                    &su3_dagger(&local_gauge[next_index]),
                );
                out.links[index * 4 + mu] = transformed;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_field_matches_independent_oracle() {
        let field = WilsonGaugeField::identity([2, 2, 1, 1]).unwrap();
        assert!(field.wilson_action(6.0).unwrap().abs() < 1e-14);
        assert!((field.average_plaquette().unwrap() - 1.0).abs() < 1e-14);
        let polyakov = field.polyakov_loop([0, 0, 0]).unwrap();
        assert!((polyakov.re - 1.0).abs() < 1e-14);
        assert!(polyakov.im.abs() < 1e-14);
    }

    #[test]
    fn localized_flux_matches_independent_oracle_fixture() {
        let mut field = WilsonGaugeField::identity([2, 2, 1, 1]).unwrap();
        field
            .set_link([0, 0, 0, 0], 0, su3_diagonal(0.3, -0.1))
            .unwrap();

        let action = field.wilson_action(6.0).unwrap();
        let plaquette = field.average_plaquette().unwrap();
        assert!((action - 0.278_371_071_020_505_4).abs() < 1e-12);
        assert!((plaquette - 0.998_066_867_562_357_6).abs() < 1e-12);
    }

    #[test]
    fn local_gauge_transform_preserves_oracle_observables() {
        let mut field = WilsonGaugeField::identity([2, 2, 1, 1]).unwrap();
        field
            .set_link([0, 0, 0, 0], 0, su3_diagonal(0.3, -0.1))
            .unwrap();

        let mut gauges = Vec::new();
        for index in 0..field.site_count() {
            let site = field.site_from_index(index);
            gauges.push(su3_diagonal(
                0.07 * site.iter().sum::<usize>() as f64,
                -0.03 * (1 + site[0]) as f64,
            ));
        }
        let transformed = field.gauge_transform(&gauges).unwrap();

        let original_action = field.wilson_action(6.0).unwrap();
        let transformed_action = transformed.wilson_action(6.0).unwrap();
        let original_plaquette = field.average_plaquette().unwrap();
        let transformed_plaquette = transformed.average_plaquette().unwrap();
        assert!((original_action - transformed_action).abs() < 1e-12);
        assert!((original_plaquette - transformed_plaquette).abs() < 1e-12);
    }

    #[test]
    fn periodic_boundary_closes() {
        let field = WilsonGaugeField::identity([2, 3, 4, 5]).unwrap();
        assert_eq!(
            field.shift([1, 2, 3, 4], 0, 1).unwrap(),
            [0, 2, 3, 4]
        );
        assert_eq!(
            field.shift([1, 2, 3, 4], 3, 1).unwrap(),
            [1, 2, 3, 0]
        );
    }

    #[test]
    fn invalid_link_fails_closed() {
        let mut field = WilsonGaugeField::identity([1, 1, 1, 1]).unwrap();
        let mut invalid = su3_identity();
        invalid[0][0] = Complex::new(2.0, 0.0);
        assert!(matches!(
            field.set_link([0, 0, 0, 0], 0, invalid),
            Err(LatticeGaugeError::InvalidSu3Link { .. })
        ));
    }
}
