// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cabibbo-Marinari SU(3) subgroup heat-bath updates.
//!
//! The local Wilson force is supplied by `lattice_subgroup_force`, so the
//! stochastic heat-bath path and deterministic overrelaxation path share one
//! sign/staple/quaternion convention. Callers may request either the slow
//! finite-probe semantic reference or the direct-staple backend.

use crate::lattice_gauge::{LatticeGaugeError, Site4, Su3Matrix, WilsonGaugeField, su3_mul, validate_su3};
use crate::lattice_heatbath::{Su2HeatbathError, Su2HeatbathSample, draw_su2_heatbath_quaternion};
use crate::lattice_metropolis::Su2Subgroup;
use crate::lattice_subgroup_force::{
    SubgroupForceBackend, SubgroupForceError, embedded_subgroup_quaternion,
    evaluate_subgroup_force, quaternion_from_embedded_subgroup, touching_trace_sum,
};
use crate::lattice_sweep::Uniform01Source;

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
    pub oriented_quaternion: [f64; 4],
    pub rotation: Su3Matrix,
    pub force: Su3SubgroupHeatbathForce,
    pub force_backend: SubgroupForceBackend,
    pub used_reference_fallback: bool,
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
    Force(SubgroupForceError),
    InvalidBeta(f64),
    OrientationMismatch { projected: f64, expected: f64 },
}

impl From<LatticeGaugeError> for Su3SubgroupHeatbathError {
    fn from(value: LatticeGaugeError) -> Self { Self::Gauge(value) }
}
impl From<Su2HeatbathError> for Su3SubgroupHeatbathError {
    fn from(value: Su2HeatbathError) -> Self { Self::Heatbath(value) }
}
impl From<SubgroupForceError> for Su3SubgroupHeatbathError {
    fn from(value: SubgroupForceError) -> Self { Self::Force(value) }
}

fn dot4(a: [f64; 4], b: [f64; 4]) -> f64 {
    a.into_iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn evaluate_su3_subgroup_heatbath_force(
    field: &WilsonGaugeField,
    site: Site4,
    mu: usize,
    subgroup: Su2Subgroup,
    beta: f64,
    backend: SubgroupForceBackend,
) -> Result<(Su3SubgroupHeatbathForce, bool), Su3SubgroupHeatbathError> {
    if !beta.is_finite() || beta < 0.0 {
        return Err(Su3SubgroupHeatbathError::InvalidBeta(beta));
    }
    let evaluation = evaluate_subgroup_force(field, site, mu, subgroup, backend)?;
    let norm = evaluation.force.norm();
    let alpha = (beta / 3.0) * norm;
    Ok((
        Su3SubgroupHeatbathForce {
            constant: evaluation.force.constant,
            quaternion: evaluation.force.quaternion,
            norm,
            alpha,
        },
        evaluation.used_reference_fallback,
    ))
}

/// Compatibility/reference wrapper retaining the original finite-probe API.
pub fn probe_su3_subgroup_heatbath_force(
    field: &WilsonGaugeField,
    site: Site4,
    mu: usize,
    subgroup: Su2Subgroup,
    beta: f64,
) -> Result<Su3SubgroupHeatbathForce, Su3SubgroupHeatbathError> {
    Ok(evaluate_su3_subgroup_heatbath_force(
        field,
        site,
        mu,
        subgroup,
        beta,
        SubgroupForceBackend::FiniteProbe,
    )?.0)
}

/// Orient a canonical heat-bath quaternion into the local force direction.
///
/// If `qhat = q/|q|`, the oriented rotation is `R(qhat) R(b)`, so
/// `q . a = |q| b0`. At zero force, the local action is subgroup-independent
/// and the canonical Haar sample is used directly.
pub fn orient_heatbath_quaternion_to_force(
    subgroup: Su2Subgroup,
    force: Su3SubgroupHeatbathForce,
    canonical: Su2HeatbathSample,
) -> Result<([f64; 4], Su3Matrix), Su3SubgroupHeatbathError> {
    let canonical_matrix = embedded_subgroup_quaternion(subgroup, canonical.quaternion)?;
    let rotation = if force.norm <= FORCE_EPSILON {
        canonical_matrix
    } else {
        let qhat = force.quaternion.map(|x| x / force.norm);
        let force_matrix = embedded_subgroup_quaternion(subgroup, qhat)?;
        su3_mul(&force_matrix, &canonical_matrix)
    };
    validate_su3(&rotation, 1.0e-12)?;
    let oriented = quaternion_from_embedded_subgroup(subgroup, &rotation);
    if force.norm > FORCE_EPSILON {
        let projected = dot4(force.quaternion, oriented) / force.norm;
        let expected = canonical.quaternion[0];
        if !projected.is_finite() || (projected - expected).abs() > ORIENTATION_TOLERANCE {
            return Err(Su3SubgroupHeatbathError::OrientationMismatch { projected, expected });
        }
    }
    Ok((oriented, rotation))
}

pub fn draw_su3_subgroup_heatbath_rotation_with_backend(
    field: &WilsonGaugeField,
    site: Site4,
    mu: usize,
    subgroup: Su2Subgroup,
    beta: f64,
    backend: SubgroupForceBackend,
    source: &mut impl Uniform01Source,
    max_attempts: usize,
) -> Result<Su3SubgroupHeatbathDraw, Su3SubgroupHeatbathError> {
    let (force, used_reference_fallback) =
        evaluate_su3_subgroup_heatbath_force(field, site, mu, subgroup, beta, backend)?;
    let canonical = draw_su2_heatbath_quaternion(source, force.alpha, max_attempts)?;
    let (oriented_quaternion, rotation) =
        orient_heatbath_quaternion_to_force(subgroup, force, canonical)?;
    Ok(Su3SubgroupHeatbathDraw {
        subgroup,
        canonical,
        oriented_quaternion,
        rotation,
        force,
        force_backend: backend,
        used_reference_fallback,
    })
}

/// Compatibility/reference wrapper using the finite-probe force backend.
pub fn draw_su3_subgroup_heatbath_rotation(
    field: &WilsonGaugeField,
    site: Site4,
    mu: usize,
    subgroup: Su2Subgroup,
    beta: f64,
    source: &mut impl Uniform01Source,
    max_attempts: usize,
) -> Result<Su3SubgroupHeatbathDraw, Su3SubgroupHeatbathError> {
    draw_su3_subgroup_heatbath_rotation_with_backend(
        field,
        site,
        mu,
        subgroup,
        beta,
        SubgroupForceBackend::FiniteProbe,
        source,
        max_attempts,
    )
}

pub fn heatbath_subgroup_step_with_backend(
    field: &mut WilsonGaugeField,
    site: Site4,
    mu: usize,
    subgroup: Su2Subgroup,
    beta: f64,
    backend: SubgroupForceBackend,
    source: &mut impl Uniform01Source,
    max_attempts: usize,
) -> Result<Su3SubgroupHeatbathStepResult, Su3SubgroupHeatbathError> {
    let touching_trace_before = touching_trace_sum(field, site, mu)?;
    let original = *field.link(site, mu)?;
    let draw = draw_su3_subgroup_heatbath_rotation_with_backend(
        field, site, mu, subgroup, beta, backend, source, max_attempts,
    )?;
    field.set_link(site, mu, su3_mul(&draw.rotation, &original))?;
    let touching_trace_after = touching_trace_sum(field, site, mu)?;
    Ok(Su3SubgroupHeatbathStepResult {
        draw,
        touching_trace_before,
        touching_trace_after,
    })
}

/// Compatibility/reference wrapper using the finite-probe force backend.
pub fn heatbath_subgroup_step_reference(
    field: &mut WilsonGaugeField,
    site: Site4,
    mu: usize,
    subgroup: Su2Subgroup,
    beta: f64,
    source: &mut impl Uniform01Source,
    max_attempts: usize,
) -> Result<Su3SubgroupHeatbathStepResult, Su3SubgroupHeatbathError> {
    heatbath_subgroup_step_with_backend(
        field,
        site,
        mu,
        subgroup,
        beta,
        SubgroupForceBackend::FiniteProbe,
        source,
        max_attempts,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_metropolis::{Su2SubgroupProposal, embedded_su2_rotation};
    use crate::lattice_rng::{LatticeChaCha8Stream, LatticeStreamCoordinates, LatticeStreamDomain};

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

    fn source(replica: u16) -> LatticeChaCha8Stream {
        LatticeChaCha8Stream::new(
            [0x7d; 32],
            LatticeStreamCoordinates {
                domain: LatticeStreamDomain::Qualification,
                ensemble_slot: 0x1607,
                replica,
                rank: 0,
            },
        ).unwrap()
    }

    #[test]
    fn reference_force_remains_pinned_to_independent_oracle() {
        let field = fixture();
        let force = probe_su3_subgroup_heatbath_force(
            &field, [0, 0, 0, 0], 0, Su2Subgroup::Pair01, 5.7,
        ).unwrap();
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
    }

    #[test]
    fn staple_and_probe_backends_produce_same_force() {
        let field = fixture();
        for subgroup in Su2Subgroup::ALL {
            let (probe, probe_fallback) = evaluate_su3_subgroup_heatbath_force(
                &field, [0,0,0,0], 0, subgroup, 5.7, SubgroupForceBackend::FiniteProbe,
            ).unwrap();
            let (staple, staple_fallback) = evaluate_su3_subgroup_heatbath_force(
                &field, [0,0,0,0], 0, subgroup, 5.7, SubgroupForceBackend::Staple,
            ).unwrap();
            assert!(!probe_fallback && !staple_fallback);
            assert!((probe.constant - staple.constant).abs() < 2.0e-12);
            assert!((probe.alpha - staple.alpha).abs() < 2.0e-12);
            for i in 0..4 {
                assert!((probe.quaternion[i] - staple.quaternion[i]).abs() < 2.0e-12);
            }
        }
    }

    #[test]
    fn force_orientation_maps_canonical_scalar_to_local_projection() {
        let field = fixture();
        let raw = probe_su3_subgroup_heatbath_force(
            &field, [0,0,0,0], 0, Su2Subgroup::Pair01, 1.0,
        ).unwrap();
        let beta = 3.0 * 1.5 / raw.norm;
        let force = probe_su3_subgroup_heatbath_force(
            &field, [0,0,0,0], 0, Su2Subgroup::Pair01, beta,
        ).unwrap();
        let mut rng = source(1);
        let canonical = draw_su2_heatbath_quaternion(&mut rng, force.alpha, 256).unwrap();
        let (oriented, rotation) =
            orient_heatbath_quaternion_to_force(Su2Subgroup::Pair01, force, canonical).unwrap();
        validate_su3(&rotation, 1.0e-12).unwrap();
        let projected = dot4(force.quaternion, oriented) / force.norm;
        assert!((projected - canonical.quaternion[0]).abs() < 2.0e-12);
    }

    #[test]
    fn optimized_and_reference_steps_match_with_identical_random_stream() {
        let mut reference = fixture();
        let mut optimized = fixture();
        let mut reference_rng = source(2);
        let mut optimized_rng = source(2);
        let a = heatbath_subgroup_step_with_backend(
            &mut reference, [0,0,0,0], 0, Su2Subgroup::Pair01, 5.7,
            SubgroupForceBackend::FiniteProbe, &mut reference_rng, 256,
        ).unwrap();
        let b = heatbath_subgroup_step_with_backend(
            &mut optimized, [0,0,0,0], 0, Su2Subgroup::Pair01, 5.7,
            SubgroupForceBackend::Staple, &mut optimized_rng, 256,
        ).unwrap();
        assert!(!b.draw.used_reference_fallback);
        for i in 0..3 {
            for j in 0..3 {
                let x = reference.link([0,0,0,0], 0).unwrap()[i][j];
                let y = optimized.link([0,0,0,0], 0).unwrap()[i][j];
                assert!((x.re-y.re).abs() < 2.0e-12);
                assert!((x.im-y.im).abs() < 2.0e-12);
            }
        }
        assert!((a.draw.force.alpha - b.draw.force.alpha).abs() < 2.0e-12);
    }

    #[test]
    fn degenerate_extent_reports_staple_fallback() {
        let field = WilsonGaugeField::identity([2,2,1,2]).unwrap();
        let (_, fallback) = evaluate_su3_subgroup_heatbath_force(
            &field, [0,0,0,0], 0, Su2Subgroup::Pair01, 5.7,
            SubgroupForceBackend::Staple,
        ).unwrap();
        assert!(fallback);
    }
}
