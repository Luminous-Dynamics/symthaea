// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared local SU(2)-subgroup force semantics for pure-SU(3) Wilson updates.
//!
//! Heat-bath and overrelaxation both depend on the same local affine trace
//! functional
//!
//! `T(a) = constant + q . a`,
//!
//! where `a` is a unit SU(2) quaternion embedded in one Cabibbo-Marinari
//! subgroup. This module owns that common calculation so stochastic and
//! microcanonical kernels cannot silently drift into different staple or sign
//! conventions.
//!
//! Two backends are provided:
//! - `FiniteProbe`: slow, orientation-agnostic semantic reference;
//! - `Staple`: direct six-staple projection, with automatic reference fallback
//!   on degenerate length-1 extents where canonical plaquette de-duplication
//!   matters.

use crate::lattice_gauge::{
    LatticeGaugeError, Site4, Su3Matrix, WilsonGaugeField, su3_dagger, su3_identity, su3_mul,
    su3_trace, validate_su3,
};
use crate::lattice_metropolis::Su2Subgroup;
use crate::symmetry_groups::Complex;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineSubgroupForce {
    /// Spectator-color contribution independent of the subgroup quaternion.
    pub constant: f64,
    /// Quaternion coefficients in `T(a) = constant + q . a`.
    pub quaternion: [f64; 4],
}

impl AffineSubgroupForce {
    pub fn norm(self) -> f64 {
        self.quaternion.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubgroupForceBackend {
    FiniteProbe,
    Staple,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubgroupForceEvaluation {
    pub force: AffineSubgroupForce,
    pub requested_backend: SubgroupForceBackend,
    pub used_reference_fallback: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SubgroupForceError {
    Gauge(LatticeGaugeError),
    NonUnitQuaternion([f64; 4]),
    NonFiniteForce([f64; 4]),
}

impl From<LatticeGaugeError> for SubgroupForceError {
    fn from(value: LatticeGaugeError) -> Self {
        Self::Gauge(value)
    }
}

#[inline]
fn c_add(a: Complex, b: Complex) -> Complex {
    Complex::new(a.re + b.re, a.im + b.im)
}

fn zero_matrix() -> Su3Matrix {
    [[Complex::ZERO; 3]; 3]
}

fn matrix_add_assign(dst: &mut Su3Matrix, src: &Su3Matrix) {
    for i in 0..3 {
        for j in 0..3 {
            dst[i][j] = c_add(dst[i][j], src[i][j]);
        }
    }
}

/// Embed a unit quaternion `(a0,a1,a2,a3)` as
/// `a0 I + i (a1 sigma1 + a2 sigma2 + a3 sigma3)` in one SU(2) subgroup.
pub fn embedded_subgroup_quaternion(
    subgroup: Su2Subgroup,
    quaternion: [f64; 4],
) -> Result<Su3Matrix, SubgroupForceError> {
    let norm_sq = quaternion.iter().map(|x| x * x).sum::<f64>();
    if !norm_sq.is_finite() || (norm_sq - 1.0).abs() > 2.0e-12 {
        return Err(SubgroupForceError::NonUnitQuaternion(quaternion));
    }
    let [a0, a1, a2, a3] = quaternion;
    let mut out = su3_identity();
    let (i, j) = subgroup.indices();
    out[i][i] = Complex::new(a0, a3);
    out[i][j] = Complex::new(a2, a1);
    out[j][i] = Complex::new(-a2, a1);
    out[j][j] = Complex::new(a0, -a3);
    validate_su3(&out, 1.0e-12)?;
    Ok(out)
}

/// Recover the subgroup quaternion from a matrix known to be an embedded SU(2)
/// element using the same convention as `embedded_subgroup_quaternion`.
pub fn quaternion_from_embedded_subgroup(
    subgroup: Su2Subgroup,
    matrix: &Su3Matrix,
) -> [f64; 4] {
    let (i, j) = subgroup.indices();
    [
        0.5 * (matrix[i][i].re + matrix[j][j].re),
        0.5 * (matrix[i][j].im + matrix[j][i].im),
        0.5 * (matrix[i][j].re - matrix[j][i].re),
        0.5 * (matrix[i][i].im - matrix[j][j].im),
    ]
}

fn touching_plaquettes(
    field: &WilsonGaugeField,
    site: Site4,
    mu: usize,
) -> Result<BTreeSet<(Site4, usize, usize)>, SubgroupForceError> {
    if mu >= 4 {
        return Err(LatticeGaugeError::InvalidDirection(mu).into());
    }
    let mut plaquettes = BTreeSet::new();
    for nu in 0..4 {
        if nu == mu {
            continue;
        }
        let (a, b) = if mu < nu { (mu, nu) } else { (nu, mu) };
        plaquettes.insert((site, a, b));
        plaquettes.insert((field.shift(site, nu, -1)?, a, b));
    }
    Ok(plaquettes)
}

/// Sum `Re Tr(P)` over the unique canonical plaquettes touching one link.
pub fn touching_trace_sum(
    field: &WilsonGaugeField,
    site: Site4,
    mu: usize,
) -> Result<f64, SubgroupForceError> {
    let mut sum = 0.0;
    for (base, a, b) in touching_plaquettes(field, site, mu)? {
        sum += su3_trace(&field.plaquette(base, a, b)?).re;
    }
    Ok(sum)
}

fn evaluated_trace_sum(
    field: &WilsonGaugeField,
    site: Site4,
    mu: usize,
    subgroup: Su2Subgroup,
    quaternion: [f64; 4],
) -> Result<f64, SubgroupForceError> {
    let mut candidate = field.clone();
    let original = *candidate.link(site, mu)?;
    let rotation = embedded_subgroup_quaternion(subgroup, quaternion)?;
    candidate.set_link(site, mu, su3_mul(&rotation, &original))?;
    touching_trace_sum(&candidate, site, mu)
}

/// Slow semantic reference: reconstruct the affine force using five unit probes.
pub fn probe_affine_subgroup_force(
    field: &WilsonGaugeField,
    site: Site4,
    mu: usize,
    subgroup: Su2Subgroup,
) -> Result<AffineSubgroupForce, SubgroupForceError> {
    let plus = evaluated_trace_sum(field, site, mu, subgroup, [1.0, 0.0, 0.0, 0.0])?;
    let minus = evaluated_trace_sum(field, site, mu, subgroup, [-1.0, 0.0, 0.0, 0.0])?;
    let constant = 0.5 * (plus + minus);
    let quaternion = [
        0.5 * (plus - minus),
        evaluated_trace_sum(field, site, mu, subgroup, [0.0, 1.0, 0.0, 0.0])? - constant,
        evaluated_trace_sum(field, site, mu, subgroup, [0.0, 0.0, 1.0, 0.0])? - constant,
        evaluated_trace_sum(field, site, mu, subgroup, [0.0, 0.0, 0.0, 1.0])? - constant,
    ];
    if quaternion.iter().any(|x| !x.is_finite()) {
        return Err(SubgroupForceError::NonFiniteForce(quaternion));
    }
    Ok(AffineSubgroupForce { constant, quaternion })
}

/// Standard six-staple matrix `H`, oriented so the local trace contribution is
/// `Re Tr[U_mu(x) H]`. Requires every lattice extent to be at least two.
pub fn link_staple_non_degenerate(
    field: &WilsonGaugeField,
    site: Site4,
    mu: usize,
) -> Result<Su3Matrix, SubgroupForceError> {
    if mu >= 4 {
        return Err(LatticeGaugeError::InvalidDirection(mu).into());
    }
    let dims = field.dims();
    if dims.iter().any(|&n| n < 2) {
        return Err(LatticeGaugeError::InvalidExtent(dims).into());
    }

    let mut staple = zero_matrix();
    let x_plus_mu = field.shift(site, mu, 1)?;
    for nu in 0..4 {
        if nu == mu {
            continue;
        }

        let x_plus_nu = field.shift(site, nu, 1)?;
        let forward = su3_mul(
            &su3_mul(
                field.link(x_plus_mu, nu)?,
                &su3_dagger(field.link(x_plus_nu, mu)?),
            ),
            &su3_dagger(field.link(site, nu)?),
        );
        matrix_add_assign(&mut staple, &forward);

        let x_minus_nu = field.shift(site, nu, -1)?;
        let x_minus_nu_plus_mu = field.shift(x_minus_nu, mu, 1)?;
        let backward = su3_mul(
            &su3_mul(
                &su3_dagger(field.link(x_minus_nu_plus_mu, nu)?),
                &su3_dagger(field.link(x_minus_nu, mu)?),
            ),
            field.link(x_minus_nu, nu)?,
        );
        matrix_add_assign(&mut staple, &backward);
    }
    Ok(staple)
}

fn spectator_index(subgroup: Su2Subgroup) -> usize {
    match subgroup {
        Su2Subgroup::Pair01 => 2,
        Su2Subgroup::Pair02 => 1,
        Su2Subgroup::Pair12 => 0,
    }
}

/// Direct quaternion projection of `X = U H` for non-degenerate lattices.
pub fn staple_affine_subgroup_force(
    field: &WilsonGaugeField,
    site: Site4,
    mu: usize,
    subgroup: Su2Subgroup,
) -> Result<AffineSubgroupForce, SubgroupForceError> {
    let staple = link_staple_non_degenerate(field, site, mu)?;
    let x = su3_mul(field.link(site, mu)?, &staple);
    let (i, j) = subgroup.indices();
    let k = spectator_index(subgroup);
    let quaternion = [
        x[i][i].re + x[j][j].re,
        -x[j][i].im - x[i][j].im,
        x[j][i].re - x[i][j].re,
        -x[i][i].im + x[j][j].im,
    ];
    if quaternion.iter().any(|value| !value.is_finite()) {
        return Err(SubgroupForceError::NonFiniteForce(quaternion));
    }
    Ok(AffineSubgroupForce {
        constant: x[k][k].re,
        quaternion,
    })
}

/// Evaluate the requested backend while preserving canonical plaquette semantics.
pub fn evaluate_subgroup_force(
    field: &WilsonGaugeField,
    site: Site4,
    mu: usize,
    subgroup: Su2Subgroup,
    backend: SubgroupForceBackend,
) -> Result<SubgroupForceEvaluation, SubgroupForceError> {
    match backend {
        SubgroupForceBackend::FiniteProbe => Ok(SubgroupForceEvaluation {
            force: probe_affine_subgroup_force(field, site, mu, subgroup)?,
            requested_backend: backend,
            used_reference_fallback: false,
        }),
        SubgroupForceBackend::Staple if field.dims().iter().any(|&n| n < 2) => {
            Ok(SubgroupForceEvaluation {
                force: probe_affine_subgroup_force(field, site, mu, subgroup)?,
                requested_backend: backend,
                used_reference_fallback: true,
            })
        }
        SubgroupForceBackend::Staple => Ok(SubgroupForceEvaluation {
            force: staple_affine_subgroup_force(field, site, mu, subgroup)?,
            requested_backend: backend,
            used_reference_fallback: false,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_metropolis::{Su2SubgroupProposal, embedded_su2_rotation};

    fn install_rotation(
        field: &mut WilsonGaugeField,
        site: Site4,
        mu: usize,
        subgroup: Su2Subgroup,
        axis: [f64; 3],
        angle: f64,
    ) {
        let original = *field.link(site, mu).unwrap();
        let rotation = embedded_su2_rotation(Su2SubgroupProposal { subgroup, axis, angle }).unwrap();
        field.set_link(site, mu, su3_mul(&rotation, &original)).unwrap();
    }

    fn fixture() -> WilsonGaugeField {
        let mut field = WilsonGaugeField::identity([2, 2, 2, 2]).unwrap();
        install_rotation(&mut field, [0, 0, 0, 0], 0, Su2Subgroup::Pair01, [1.0, 2.0, 3.0], 0.31);
        install_rotation(&mut field, [1, 0, 1, 0], 2, Su2Subgroup::Pair02, [2.0, -1.0, 1.0], -0.27);
        install_rotation(&mut field, [0, 1, 0, 1], 3, Su2Subgroup::Pair12, [1.0, 1.0, -2.0], 0.22);
        install_rotation(&mut field, [1, 1, 1, 1], 1, Su2Subgroup::Pair01, [-2.0, 1.0, 1.0], 0.19);
        field
    }

    #[test]
    fn finite_probe_remains_pinned_to_independent_oracle() {
        let field = fixture();
        let force = probe_affine_subgroup_force(&field, [0, 0, 0, 0], 0, Su2Subgroup::Pair01).unwrap();
        assert!((force.constant - 5.963_770_896_365_891).abs() < 1.0e-12);
        let expected = [
            11.366_866_478_675_853,
            -0.957_654_316_646_473_3,
            -1.959_698_913_884_743_4,
            -3.029_933_415_654_614_5,
        ];
        for (got, want) in force.quaternion.into_iter().zip(expected) {
            assert!((got - want).abs() < 1.0e-12);
        }
    }

    #[test]
    fn direct_staple_matches_reference_for_every_subgroup() {
        let field = fixture();
        for subgroup in Su2Subgroup::ALL {
            let reference = probe_affine_subgroup_force(&field, [0, 0, 0, 0], 0, subgroup).unwrap();
            let optimized = staple_affine_subgroup_force(&field, [0, 0, 0, 0], 0, subgroup).unwrap();
            assert!((reference.constant - optimized.constant).abs() < 2.0e-12);
            for i in 0..4 {
                assert!((reference.quaternion[i] - optimized.quaternion[i]).abs() < 2.0e-12);
            }
        }
    }

    #[test]
    fn degenerate_extent_staple_request_falls_back_to_reference() {
        let field = WilsonGaugeField::identity([2, 2, 1, 2]).unwrap();
        let evaluation = evaluate_subgroup_force(
            &field,
            [0, 0, 0, 0],
            0,
            Su2Subgroup::Pair01,
            SubgroupForceBackend::Staple,
        )
        .unwrap();
        assert!(evaluation.used_reference_fallback);
        assert_eq!(evaluation.requested_backend, SubgroupForceBackend::Staple);
    }

    #[test]
    fn embedding_round_trips_quaternion_convention() {
        let q = [0.8, 0.2, -0.4, (0.16f64).sqrt()];
        let norm = q.iter().map(|x| x * x).sum::<f64>().sqrt();
        let q = q.map(|x| x / norm);
        for subgroup in Su2Subgroup::ALL {
            let matrix = embedded_subgroup_quaternion(subgroup, q).unwrap();
            let recovered = quaternion_from_embedded_subgroup(subgroup, &matrix);
            for i in 0..4 {
                assert!((recovered[i] - q[i]).abs() < 1.0e-12);
            }
        }
    }
}
