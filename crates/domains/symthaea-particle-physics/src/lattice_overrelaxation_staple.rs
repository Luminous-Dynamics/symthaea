// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Optimized staple-based microcanonical overrelaxation for pure SU(3).
//!
//! This module is required to remain numerically equivalent to the slower
//! orientation-agnostic finite-probe reference in `lattice_overrelaxation`.
//! On degenerate extents (length 1), where forward/backward staple terms can
//! refer to the same canonical plaquette, it deliberately falls back to the
//! reference force rather than double-counting.

use crate::lattice_gauge::{
    LatticeGaugeError, Site4, Su3Matrix, WilsonGaugeField, su3_dagger, su3_identity, su3_mul,
    validate_su3,
};
use crate::lattice_metropolis::Su2Subgroup;
use crate::lattice_overrelaxation::{
    AffineSubgroupForce, LatticeOverrelaxationError, OverrelaxationStepResult,
    equal_action_reflection, probe_affine_subgroup_force, touching_trace_sum,
};
use crate::symmetry_groups::Complex;

const MICROCANONICAL_TOLERANCE: f64 = 1.0e-10;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OverrelaxationSweepStats {
    pub subgroup_updates: usize,
    pub max_abs_touching_trace_delta: f64,
    /// Number of subgroup force evaluations that used the finite-probe fallback
    /// because at least one lattice extent was one site long.
    pub reference_fallback_updates: usize,
}

fn c_add(a: Complex, b: Complex) -> Complex {
    Complex::new(a.re + b.re, a.im + b.im)
}

fn matrix_add_assign(dst: &mut Su3Matrix, src: &Su3Matrix) {
    for i in 0..3 {
        for j in 0..3 {
            dst[i][j] = c_add(dst[i][j], src[i][j]);
        }
    }
}

fn zero_matrix() -> Su3Matrix {
    [[Complex::ZERO; 3]; 3]
}

fn embedded_quaternion(
    subgroup: Su2Subgroup,
    quaternion: [f64; 4],
) -> Result<Su3Matrix, LatticeOverrelaxationError> {
    let norm_sq = quaternion.iter().map(|x| x * x).sum::<f64>();
    if !norm_sq.is_finite() || (norm_sq - 1.0).abs() > 2.0e-12 {
        return Err(LatticeOverrelaxationError::NonUnitQuaternion(quaternion));
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

/// Standard six-staple matrix H for a link U_mu(x), oriented so that
/// the local plaquette trace sum is Re Tr[U_mu(x) H].
///
/// This formula assumes all extents are at least two. Callers should use
/// `analytic_affine_subgroup_force`, which falls back to the finite-probe
/// reference on degenerate extents.
pub fn link_staple_non_degenerate(
    field: &WilsonGaugeField,
    site: Site4,
    mu: usize,
) -> Result<Su3Matrix, LatticeOverrelaxationError> {
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

        // Forward staple:
        // U_nu(x+mu) U_mu^dagger(x+nu) U_nu^dagger(x)
        let x_plus_nu = field.shift(site, nu, 1)?;
        let forward = su3_mul(
            &su3_mul(
                field.link(x_plus_mu, nu)?,
                &su3_dagger(field.link(x_plus_nu, mu)?),
            ),
            &su3_dagger(field.link(site, nu)?),
        );
        matrix_add_assign(&mut staple, &forward);

        // Backward staple:
        // U_nu^dagger(x-nu+mu) U_mu^dagger(x-nu) U_nu(x-nu)
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

/// Analytic quaternion projection of X = U H for the requested SU(2) subgroup.
///
/// For a subgroup quaternion a=(a0,a1,a2,a3), the local trace is
/// `T(a) = constant + q . a`. On non-degenerate lattices q is obtained directly
/// from the 2x2 block of X. Degenerate lattices fall back to the independently
/// qualified finite-probe construction to preserve canonical-plaquette semantics.
pub fn analytic_affine_subgroup_force(
    field: &WilsonGaugeField,
    site: Site4,
    mu: usize,
    subgroup: Su2Subgroup,
) -> Result<(AffineSubgroupForce, bool), LatticeOverrelaxationError> {
    if field.dims().iter().any(|&n| n < 2) {
        return Ok((probe_affine_subgroup_force(field, site, mu, subgroup)?, true));
    }

    let staple = link_staple_non_degenerate(field, site, mu)?;
    let x = su3_mul(field.link(site, mu)?, &staple);
    let (i, j) = subgroup.indices();
    let k = spectator_index(subgroup);

    let force = AffineSubgroupForce {
        constant: x[k][k].re,
        quaternion: [
            x[i][i].re + x[j][j].re,
            -x[j][i].im - x[i][j].im,
            x[j][i].re - x[i][j].re,
            -x[i][i].im + x[j][j].im,
        ],
    };
    Ok((force, false))
}

pub fn overrelax_subgroup_staple(
    field: &mut WilsonGaugeField,
    site: Site4,
    mu: usize,
    subgroup: Su2Subgroup,
) -> Result<(OverrelaxationStepResult, bool), LatticeOverrelaxationError> {
    let before = touching_trace_sum(field, site, mu)?;
    let original = *field.link(site, mu)?;
    let (force, used_reference_fallback) =
        analytic_affine_subgroup_force(field, site, mu, subgroup)?;
    let reflection = equal_action_reflection(force.quaternion)?;
    let rotation = embedded_quaternion(subgroup, reflection)?;
    field.set_link(site, mu, su3_mul(&rotation, &original))?;
    let after = touching_trace_sum(field, site, mu)?;
    let drift = after - before;
    if !drift.is_finite() || drift.abs() > MICROCANONICAL_TOLERANCE {
        field.set_link(site, mu, original)?;
        return Err(LatticeOverrelaxationError::MicrocanonicalDrift { before, after });
    }
    Ok((
        OverrelaxationStepResult {
            subgroup,
            force,
            reflection,
            touching_trace_delta: drift,
        },
        used_reference_fallback,
    ))
}

/// One deterministic lexicographic microcanonical sweep over every link and all
/// three Cabibbo-Marinari SU(2) subgroups.
pub fn overrelax_sweep_staple(
    field: &mut WilsonGaugeField,
) -> Result<OverrelaxationSweepStats, LatticeOverrelaxationError> {
    let dims = field.dims();
    let mut subgroup_updates = 0usize;
    let mut max_abs_touching_trace_delta: f64 = 0.0;
    let mut reference_fallback_updates = 0usize;

    for x in 0..dims[0] {
        for y in 0..dims[1] {
            for z in 0..dims[2] {
                for t in 0..dims[3] {
                    let site = [x, y, z, t];
                    for mu in 0..4 {
                        for subgroup in Su2Subgroup::ALL {
                            let (result, fallback) =
                                overrelax_subgroup_staple(field, site, mu, subgroup)?;
                            subgroup_updates += 1;
                            reference_fallback_updates += usize::from(fallback);
                            max_abs_touching_trace_delta = max_abs_touching_trace_delta
                                .max(result.touching_trace_delta.abs());
                        }
                    }
                }
            }
        }
    }

    Ok(OverrelaxationSweepStats {
        subgroup_updates,
        max_abs_touching_trace_delta,
        reference_fallback_updates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_metropolis::{
        Su2SubgroupProposal, embedded_su2_rotation,
    };
    use crate::lattice_overrelaxation::{
        overrelax_subgroup_reference, probe_affine_subgroup_force,
    };

    fn matrix_max_error(a: &Su3Matrix, b: &Su3Matrix) -> f64 {
        let mut max_error: f64 = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                let dr = a[i][j].re - b[i][j].re;
                let di = a[i][j].im - b[i][j].im;
                max_error = max_error.max((dr * dr + di * di).sqrt());
            }
        }
        max_error
    }

    fn install_rotation(
        field: &mut WilsonGaugeField,
        site: Site4,
        mu: usize,
        subgroup: Su2Subgroup,
        axis: [f64; 3],
        angle: f64,
    ) {
        let original = *field.link(site, mu).unwrap();
        let rotation = embedded_su2_rotation(Su2SubgroupProposal {
            subgroup,
            axis,
            angle,
        })
        .unwrap();
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
    fn staple_projection_matches_finite_probe_reference() {
        let field = fixture();
        for subgroup in Su2Subgroup::ALL {
            let reference =
                probe_affine_subgroup_force(&field, [0, 0, 0, 0], 0, subgroup).unwrap();
            let (analytic, fallback) =
                analytic_affine_subgroup_force(&field, [0, 0, 0, 0], 0, subgroup).unwrap();
            assert!(!fallback);
            assert!((analytic.constant - reference.constant).abs() < 2.0e-12);
            for i in 0..4 {
                assert!((analytic.quaternion[i] - reference.quaternion[i]).abs() < 2.0e-12);
            }
        }
    }

    #[test]
    fn optimized_step_matches_reference_resulting_link() {
        let field = fixture();
        for subgroup in Su2Subgroup::ALL {
            let mut reference_field = field.clone();
            let mut optimized_field = field.clone();
            overrelax_subgroup_reference(
                &mut reference_field,
                [0, 0, 0, 0],
                0,
                subgroup,
            )
            .unwrap();
            let (_, fallback) = overrelax_subgroup_staple(
                &mut optimized_field,
                [0, 0, 0, 0],
                0,
                subgroup,
            )
            .unwrap();
            assert!(!fallback);
            assert!(matrix_max_error(
                reference_field.link([0, 0, 0, 0], 0).unwrap(),
                optimized_field.link([0, 0, 0, 0], 0).unwrap(),
            ) < 2.0e-12);
        }
    }

    #[test]
    fn complete_non_degenerate_sweep_is_microcanonical() {
        let mut field = fixture();
        let before = field.wilson_action(5.7).unwrap();
        let stats = overrelax_sweep_staple(&mut field).unwrap();
        let after = field.wilson_action(5.7).unwrap();
        assert_eq!(stats.subgroup_updates, 2 * 2 * 2 * 2 * 4 * 3);
        assert_eq!(stats.reference_fallback_updates, 0);
        assert!(stats.max_abs_touching_trace_delta < 1.0e-10);
        assert!((after - before).abs() < 2.0e-9);
    }

    #[test]
    fn degenerate_extent_falls_back_instead_of_double_counting() {
        let mut field = WilsonGaugeField::identity([2, 2, 1, 2]).unwrap();
        install_rotation(&mut field, [0, 0, 0, 0], 0, Su2Subgroup::Pair01, [1.0, 2.0, 3.0], 0.31);
        install_rotation(&mut field, [1, 1, 0, 1], 3, Su2Subgroup::Pair12, [1.0, -1.0, 2.0], -0.23);
        let mut reference = field.clone();
        overrelax_subgroup_reference(
            &mut reference,
            [0, 0, 0, 0],
            0,
            Su2Subgroup::Pair02,
        )
        .unwrap();
        let (_, fallback) = overrelax_subgroup_staple(
            &mut field,
            [0, 0, 0, 0],
            0,
            Su2Subgroup::Pair02,
        )
        .unwrap();
        assert!(fallback);
        assert!(matrix_max_error(
            reference.link([0, 0, 0, 0], 0).unwrap(),
            field.link([0, 0, 0, 0], 0).unwrap(),
        ) < 2.0e-12);
    }
}
