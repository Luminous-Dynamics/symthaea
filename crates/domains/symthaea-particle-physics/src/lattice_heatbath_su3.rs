// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reference Cabibbo-Marinari SU(3) subgroup heat-bath update.
//!
//! This implementation intentionally reconstructs the affine local subgroup
//! force by five finite probes instead of relying on optimized staple algebra.
//! It therefore serves as a semantic reference for a later direct-staple
//! implementation.

use crate::lattice_gauge::{
    LatticeGaugeError, Site4, Su3Matrix, WilsonGaugeField, su3_identity, su3_mul, su3_trace,
    validate_su3,
};
use crate::lattice_heatbath::{
    Su2HeatbathError, Su2HeatbathSample, draw_su2_heatbath_quaternion,
};
use crate::lattice_metropolis::Su2Subgroup;
use crate::lattice_sweep::Uniform01Source;
use crate::symmetry_groups::Complex;
use std::collections::BTreeSet;

const FORCE_EPSILON: f64 = 1.0e-28;
const ORIENTATION_TOLERANCE: f64 = 2.0e-12;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Su3SubgroupHeatbathForce {
    pub constant: f64,
    pub quaternion: [f64; 4],
    pub norm: f64,
    /// Wilson conditional coupling `alpha = (beta / 3) * norm`.
    pub alpha: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Su3SubgroupHeatbathDraw {
    pub subgroup: Su2Subgroup,
    pub canonical: Su2HeatbathSample,
    /// Quaternion after orienting the canonical heat-bath sample into the local
    /// force direction.
    pub oriented_quaternion: [f64; 4],
    pub rotation: Su3Matrix,
    pub force: Su3SubgroupHeatbathForce,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Su3SubgroupHeatbathStepResult {
    pub draw: Su3SubgroupHeatbathDraw,
    pub touching_trace_before: f64,
    pub touching_trace_after: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Su3SubgroupHeatbathError {
    Gauge(LatticeGaugeError),
    Heatbath(Su2HeatbathError),
    InvalidBeta(f64),
    NonFiniteForce([f64; 4]),
    OrientationMismatch { projected: f64, expected: f64 },
}

impl From<LatticeGaugeError> for Su3SubgroupHeatbathError {
    fn from(value: LatticeGaugeError) -> Self {
        Self::Gauge(value)
    }
}

impl From<Su2HeatbathError> for Su3SubgroupHeatbathError {
    fn from(value: Su2HeatbathError) -> Self {
        Self::Heatbath(value)
    }
}

fn embedded_quaternion(
    subgroup: Su2Subgroup,
    quaternion: [f64; 4],
) -> Result<Su3Matrix, Su3SubgroupHeatbathError> {
    let norm_sq = quaternion.iter().map(|x| x * x).sum::<f64>();
    if !norm_sq.is_finite() || (norm_sq - 1.0).abs() > 2.0e-12 {
        return Err(Su3SubgroupHeatbathError::NonFiniteForce(quaternion));
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

fn quaternion_from_embedded(
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
) -> Result<BTreeSet<(Site4, usize, usize)>, Su3SubgroupHeatbathError> {
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

pub fn heatbath_touching_trace_sum(
    field: &WilsonGaugeField,
    site: Site4,
    mu: usize,
) -> Result<f64, Su3SubgroupHeatbathError> {
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
) -> Result<f64, Su3SubgroupHeatbathError> {
    let mut candidate = field.clone();
    let original = *candidate.link(site, mu)?;
    let rotation = embedded_quaternion(subgroup, quaternion)?;
    candidate.set_link(site, mu, su3_mul(&rotation, &original))?;
    heatbath_touching_trace_sum(&candidate, site, mu)
}

/// Reconstruct `T(a) = constant + q.a` by five unit-quaternion probes.
pub fn probe_su3_subgroup_heatbath_force(
    field: &WilsonGaugeField,
    site: Site4,
    mu: usize,
    subgroup: Su2Subgroup,
    beta: f64,
) -> Result<Su3SubgroupHeatbathForce, Su3SubgroupHeatbathError> {
    if !beta.is_finite() || beta < 0.0 {
        return Err(Su3SubgroupHeatbathError::InvalidBeta(beta));
    }
    let plus = evaluated_trace_sum(field, site, mu, subgroup, [1.0, 0.0, 0.0, 0.0])?;
    let minus = evaluated_trace_sum(field, site, mu, subgroup, [-1.0, 0.0, 0.0, 0.0])?;
    let constant = 0.5 * (plus + minus);
    let q0 = 0.5 * (plus - minus);
    let q1 = evaluated_trace_sum(field, site, mu, subgroup, [0.0, 1.0, 0.0, 0.0])?
        - constant;
    let q2 = evaluated_trace_sum(field, site, mu, subgroup, [0.0, 0.0, 1.0, 0.0])?
        - constant;
    let q3 = evaluated_trace_sum(field, site, mu, subgroup, [0.0, 0.0, 0.0, 1.0])?
        - constant;
    let quaternion = [q0, q1, q2, q3];
    if quaternion.iter().any(|x| !x.is_finite()) {
        return Err(Su3SubgroupHeatbathError::NonFiniteForce(quaternion));
    }
    let norm = quaternion.iter().map(|x| x * x).sum::<f64>().sqrt();
    let alpha = (beta / 3.0) * norm;
    Ok(Su3SubgroupHeatbathForce {
        constant,
        quaternion,
        norm,
        alpha,
    })
}

fn dot4(a: [f64; 4], b: [f64; 4]) -> f64 {
    a.into_iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Orient a canonical heat-bath quaternion into the local force direction.
///
/// If `qhat = q/|q|`, the oriented rotation is `R(qhat) R(b)`, which satisfies
/// `q . a = |q| b0`. At zero force the local action is subgroup-independent and
/// the canonical Haar sample is used directly.
pub fn orient_heatbath_quaternion_to_force(
    subgroup: Su2Subgroup,
    force: Su3SubgroupHeatbathForce,
    canonical: Su2HeatbathSample,
) -> Result<([f64; 4], Su3Matrix), Su3SubgroupHeatbathError> {
    let canonical_matrix = embedded_quaternion(subgroup, canonical.quaternion)?;
    let rotation = if force.norm <= FORCE_EPSILON {
        canonical_matrix
    } else {
        let qhat = force.quaternion.map(|x| x / force.norm);
        let force_matrix = embedded_quaternion(subgroup, qhat)?;
        su3_mul(&force_matrix, &canonical_matrix)
    };
    validate_su3(&rotation, 1.0e-12)?;
    let oriented = quaternion_from_embedded(subgroup, &rotation);

    if force.norm > FORCE_EPSILON {
        let projected = dot4(force.quaternion, oriented) / force.norm;
        let expected = canonical.quaternion[0];
        if !projected.is_finite() || (projected - expected).abs() > ORIENTATION_TOLERANCE {
            return Err(Su3SubgroupHeatbathError::OrientationMismatch {
                projected,
                expected,
            });
        }
    }
    Ok((oriented, rotation))
}

/// Draw one subgroup rotation from the exact local Wilson conditional.
pub fn draw_su3_subgroup_heatbath_rotation(
    field: &WilsonGaugeField,
    site: Site4,
    mu: usize,
    subgroup: Su2Subgroup,
    beta: f64,
    source: &mut impl Uniform01Source,
    max_attempts: usize,
) -> Result<Su3SubgroupHeatbathDraw, Su3SubgroupHeatbathError> {
    let force = probe_su3_subgroup_heatbath_force(field, site, mu, subgroup, beta)?;
    let canonical = draw_su2_heatbath_quaternion(source, force.alpha, max_attempts)?;
    let (oriented_quaternion, rotation) =
        orient_heatbath_quaternion_to_force(subgroup, force, canonical)?;
    Ok(Su3SubgroupHeatbathDraw {
        subgroup,
        canonical,
        oriented_quaternion,
        rotation,
        force,
    })
}

/// Replace one link along one Cabibbo-Marinari subgroup conditional.
pub fn heatbath_subgroup_step_reference(
    field: &mut WilsonGaugeField,
    site: Site4,
    mu: usize,
    subgroup: Su2Subgroup,
    beta: f64,
    source: &mut impl Uniform01Source,
    max_attempts: usize,
) -> Result<Su3SubgroupHeatbathStepResult, Su3SubgroupHeatbathError> {
    let touching_trace_before = heatbath_touching_trace_sum(field, site, mu)?;
    let original = *field.link(site, mu)?;
    let draw = draw_su3_subgroup_heatbath_rotation(
        field,
        site,
        mu,
        subgroup,
        beta,
        source,
        max_attempts,
    )?;
    field.set_link(site, mu, su3_mul(&draw.rotation, &original))?;
    let touching_trace_after = heatbath_touching_trace_sum(field, site, mu)?;
    Ok(Su3SubgroupHeatbathStepResult {
        draw,
        touching_trace_before,
        touching_trace_after,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_metropolis::{Su2SubgroupProposal, embedded_su2_rotation};
    use crate::lattice_rng::{
        LatticeChaCha8Stream, LatticeStreamCoordinates, LatticeStreamDomain,
    };

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

    fn source(replica: u16) -> LatticeChaCha8Stream {
        LatticeChaCha8Stream::new(
            [0x7d; 32],
            LatticeStreamCoordinates {
                domain: LatticeStreamDomain::Qualification,
                ensemble_slot: 0x1607,
                replica,
                rank: 0,
            },
        )
        .unwrap()
    }

    #[test]
    fn finite_probe_force_matches_independent_overrelaxation_oracle_fixture() {
        let field = fixture();
        let force = probe_su3_subgroup_heatbath_force(
            &field,
            [0, 0, 0, 0],
            0,
            Su2Subgroup::Pair01,
            5.7,
        )
        .unwrap();
        assert!((force.constant - 5.963_770_896_365_891).abs() < 1.0e-12);
        let expected = [
            11.366_866_478_675_853,
            -0.957_654_316_646_473_3,
            -1.959_698_913_884_743_4,
            -3.029_933_415_654_614_5,
        ];
        for i in 0..4 {
            assert!((force.quaternion[i] - expected[i]).abs() < 1.0e-12);
        }
        assert!((force.alpha - (5.7 / 3.0) * force.norm).abs() < 1.0e-14);
    }

    #[test]
    fn force_orientation_maps_canonical_scalar_to_local_projection() {
        let field = fixture();
        let raw_force = probe_su3_subgroup_heatbath_force(
            &field,
            [0, 0, 0, 0],
            0,
            Su2Subgroup::Pair01,
            1.5,
        )
        .unwrap();
        // Rescale beta so this fixed local force has the independently qualified
        // alpha=1.5 target from LQCD-016E.
        let beta = 3.0 * 1.5 / raw_force.norm;
        let force = probe_su3_subgroup_heatbath_force(
            &field,
            [0, 0, 0, 0],
            0,
            Su2Subgroup::Pair01,
            beta,
        )
        .unwrap();
        assert!((force.alpha - 1.5).abs() < 1.0e-12);

        let mut rng = source(1);
        let canonical = draw_su2_heatbath_quaternion(&mut rng, force.alpha, 256).unwrap();
        let (oriented, rotation) = orient_heatbath_quaternion_to_force(
            Su2Subgroup::Pair01,
            force,
            canonical,
        )
        .unwrap();
        validate_su3(&rotation, 1.0e-12).unwrap();
        let projected = dot4(force.quaternion, oriented) / force.norm;
        assert!((projected - canonical.quaternion[0]).abs() < 2.0e-12);
    }

    #[test]
    fn projected_conditional_moments_match_independent_alpha_1p5_target() {
        const N: usize = 30_000;
        let field = fixture();
        let base_force = probe_su3_subgroup_heatbath_force(
            &field,
            [0, 0, 0, 0],
            0,
            Su2Subgroup::Pair01,
            1.0,
        )
        .unwrap();
        let beta = 3.0 * 1.5 / base_force.norm;
        let force = probe_su3_subgroup_heatbath_force(
            &field,
            [0, 0, 0, 0],
            0,
            Su2Subgroup::Pair01,
            beta,
        )
        .unwrap();
        let mut rng = source(2);
        let mut sum = 0.0;
        let mut sum2 = 0.0;
        let mut sum4 = 0.0;
        for _ in 0..N {
            let canonical = draw_su2_heatbath_quaternion(&mut rng, force.alpha, 256).unwrap();
            let (oriented, _) = orient_heatbath_quaternion_to_force(
                Su2Subgroup::Pair01,
                force,
                canonical,
            )
            .unwrap();
            let projected = dot4(force.quaternion, oriented) / force.norm;
            sum += projected;
            sum2 += projected * projected;
            sum4 += projected.powi(4);
        }
        let mean = sum / N as f64;
        let second = sum2 / N as f64;
        let mean_se = ((second - mean * mean).max(0.0) / N as f64).sqrt();
        let second_se = ((sum4 / N as f64 - second * second).max(0.0) / N as f64).sqrt();
        assert!((mean - 0.344_144_010_564_440_55).abs() < 5.0 * mean_se + 1.0e-12);
        assert!((second - 0.311_711_990_151_271_66).abs() < 5.0 * second_se + 1.0e-12);
    }

    #[test]
    fn reference_step_changes_link_but_preserves_su3_membership() {
        let mut field = fixture();
        let before = *field.link([0, 0, 0, 0], 0).unwrap();
        let mut rng = source(3);
        let result = heatbath_subgroup_step_reference(
            &mut field,
            [0, 0, 0, 0],
            0,
            Su2Subgroup::Pair01,
            5.7,
            &mut rng,
            256,
        )
        .unwrap();
        let after = *field.link([0, 0, 0, 0], 0).unwrap();
        validate_su3(&after, 1.0e-12).unwrap();
        assert_ne!(before, after);
        assert!(result.touching_trace_before.is_finite());
        assert!(result.touching_trace_after.is_finite());
        assert!(result.draw.force.alpha > 0.0);
    }
}
