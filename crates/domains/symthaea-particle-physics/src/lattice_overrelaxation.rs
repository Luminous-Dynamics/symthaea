// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reference microcanonical overrelaxation for pure-SU(3) Wilson gauge fields.
//!
//! This implementation intentionally mirrors the independent finite-probe
//! LQCD-016B oracle instead of using an optimized staple projection. It is
//! therefore slow, but orientation-agnostic and suitable as a parity reference
//! for a later optimized kernel.
//!
//! Overrelaxation is deterministic and microcanonical. In the current Symthaea
//! lattice program it is never treated as a replacement for an independently
//! qualified stochastic/ergodic transition kernel.

use crate::lattice_gauge::{
    LatticeGaugeError, Site4, Su3Matrix, WilsonGaugeField, su3_identity, su3_mul, su3_trace,
    validate_su3,
};
use crate::lattice_metropolis::Su2Subgroup;
use crate::symmetry_groups::Complex;
use std::collections::BTreeSet;

const MICROCANONICAL_TOLERANCE: f64 = 1.0e-10;
const FORCE_EPSILON: f64 = 1.0e-28;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineSubgroupForce {
    /// Spectator-color contribution independent of the SU(2) quaternion.
    pub constant: f64,
    /// Coefficients q in T(a) = constant + q . a.
    pub quaternion: [f64; 4],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OverrelaxationStepResult {
    pub subgroup: Su2Subgroup,
    pub force: AffineSubgroupForce,
    pub reflection: [f64; 4],
    /// Difference in the sum of Re Tr(P) for the unique plaquettes touching the link.
    pub touching_trace_delta: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LatticeOverrelaxationError {
    Gauge(LatticeGaugeError),
    NonUnitQuaternion([f64; 4]),
    DegenerateForce([f64; 4]),
    MicrocanonicalDrift { before: f64, after: f64 },
}

impl From<LatticeGaugeError> for LatticeOverrelaxationError {
    fn from(value: LatticeGaugeError) -> Self {
        Self::Gauge(value)
    }
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

fn touching_plaquettes(
    field: &WilsonGaugeField,
    site: Site4,
    mu: usize,
) -> Result<BTreeSet<(Site4, usize, usize)>, LatticeOverrelaxationError> {
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

pub fn touching_trace_sum(
    field: &WilsonGaugeField,
    site: Site4,
    mu: usize,
) -> Result<f64, LatticeOverrelaxationError> {
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
) -> Result<f64, LatticeOverrelaxationError> {
    let mut candidate = field.clone();
    let original = *candidate.link(site, mu)?;
    let rotation = embedded_quaternion(subgroup, quaternion)?;
    candidate.set_link(site, mu, su3_mul(&rotation, &original))?;
    touching_trace_sum(&candidate, site, mu)
}

/// Reconstruct the affine quaternion force without assuming a staple orientation.
///
/// Five unit-quaternion probes determine
/// `T(a) = constant + q0*a0 + q1*a1 + q2*a2 + q3*a3`.
pub fn probe_affine_subgroup_force(
    field: &WilsonGaugeField,
    site: Site4,
    mu: usize,
    subgroup: Su2Subgroup,
) -> Result<AffineSubgroupForce, LatticeOverrelaxationError> {
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
    Ok(AffineSubgroupForce {
        constant,
        quaternion: [q0, q1, q2, q3],
    })
}

pub fn equal_action_reflection(
    force: [f64; 4],
) -> Result<[f64; 4], LatticeOverrelaxationError> {
    let norm_sq = force.iter().map(|x| x * x).sum::<f64>();
    if !norm_sq.is_finite() || norm_sq <= FORCE_EPSILON {
        return Err(LatticeOverrelaxationError::DegenerateForce(force));
    }
    let mut reflection = force.map(|x| 2.0 * force[0] * x / norm_sq);
    reflection[0] -= 1.0;
    let reflection_norm_sq = reflection.iter().map(|x| x * x).sum::<f64>();
    if !reflection_norm_sq.is_finite() || (reflection_norm_sq - 1.0).abs() > 2.0e-12 {
        return Err(LatticeOverrelaxationError::NonUnitQuaternion(reflection));
    }
    Ok(reflection)
}

/// Apply one deterministic microcanonical subgroup reflection.
///
/// The reference kernel fails closed and restores the original link if the
/// touched-plaquette trace changes beyond floating-point qualification tolerance.
pub fn overrelax_subgroup_reference(
    field: &mut WilsonGaugeField,
    site: Site4,
    mu: usize,
    subgroup: Su2Subgroup,
) -> Result<OverrelaxationStepResult, LatticeOverrelaxationError> {
    let before = touching_trace_sum(field, site, mu)?;
    let original = *field.link(site, mu)?;
    let force = probe_affine_subgroup_force(field, site, mu, subgroup)?;
    let reflection = equal_action_reflection(force.quaternion)?;
    let rotation = embedded_quaternion(subgroup, reflection)?;
    field.set_link(site, mu, su3_mul(&rotation, &original))?;
    let after = touching_trace_sum(field, site, mu)?;
    let drift = after - before;
    if !drift.is_finite() || drift.abs() > MICROCANONICAL_TOLERANCE {
        field.set_link(site, mu, original)?;
        return Err(LatticeOverrelaxationError::MicrocanonicalDrift { before, after });
    }
    Ok(OverrelaxationStepResult {
        subgroup,
        force,
        reflection,
        touching_trace_delta: drift,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_metropolis::{
        Su2SubgroupProposal, embedded_su2_rotation,
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

    fn fixture() -> WilsonGaugeField {
        let mut field = WilsonGaugeField::identity([2, 2, 2, 2]).unwrap();
        let fixtures = [
            ([0, 0, 0, 0], 0, Su2Subgroup::Pair01, [1.0, 2.0, 3.0], 0.31),
            ([1, 0, 1, 0], 2, Su2Subgroup::Pair02, [2.0, -1.0, 1.0], -0.27),
            ([0, 1, 0, 1], 3, Su2Subgroup::Pair12, [1.0, 1.0, -2.0], 0.22),
            ([1, 1, 1, 1], 1, Su2Subgroup::Pair01, [-2.0, 1.0, 1.0], 0.19),
        ];
        for (site, mu, subgroup, axis, angle) in fixtures {
            let original = *field.link(site, mu).unwrap();
            let rotation = embedded_su2_rotation(Su2SubgroupProposal {
                subgroup,
                axis,
                angle,
            })
            .unwrap();
            field.set_link(site, mu, su3_mul(&rotation, &original)).unwrap();
        }
        field
    }

    #[test]
    fn reference_force_matches_independent_oracle() {
        let field = fixture();
        let cases = [
            (
                Su2Subgroup::Pair01,
                5.963_770_896_365_891,
                [11.366_866_478_675_853, -0.957_654_316_646_473_3, -1.959_698_913_884_743_4, -3.029_933_415_654_614_5],
                [0.805_258_339_905_583_7, -0.152_092_350_615_349_3, -0.311_234_658_613_362_0, -0.481_206_722_910_876_8],
            ),
            (
                Su2Subgroup::Pair02,
                5.714_001_419_314_281,
                [11.616_635_955_727_462, -0.398_555_395_737_37, 0.265_863_118_756_380_25, -1.671_279_768_120_061_1],
                [0.956_182_808_400_720_5, -0.067_114_715_164_366_68, 0.044_770_006_074_143_33, -0.281_435_069_746_891_5],
            ),
            (
                Su2Subgroup::Pair12,
                5.652_865_059_361_572,
                [11.677_772_315_680_171, 0.044_390_280_591_795_95, -8.881_784_197_001_252e-16, 1.358_653_647_534_552_5],
                [0.973_260_985_769_380_6, 0.007_500_883_428_043_168, -1.500_806_640_700_871_8e-16, 0.229_579_594_753_131_86],
            ),
        ];
        for (subgroup, expected_constant, expected_force, expected_reflection) in cases {
            let force = probe_affine_subgroup_force(&field, [0, 0, 0, 0], 0, subgroup).unwrap();
            assert!((force.constant - expected_constant).abs() < 1.0e-12);
            for i in 0..4 {
                assert!((force.quaternion[i] - expected_force[i]).abs() < 1.0e-12);
            }
            let reflection = equal_action_reflection(force.quaternion).unwrap();
            for i in 0..4 {
                assert!((reflection[i] - expected_reflection[i]).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn all_subgroups_preserve_full_wilson_action() {
        let field = fixture();
        let before = field.wilson_action(5.7).unwrap();
        for subgroup in Su2Subgroup::ALL {
            let mut candidate = field.clone();
            let result = overrelax_subgroup_reference(
                &mut candidate,
                [0, 0, 0, 0],
                0,
                subgroup,
            )
            .unwrap();
            assert!(result.touching_trace_delta.abs() < 1.0e-12);
            assert!((candidate.wilson_action(5.7).unwrap() - before).abs() < 2.0e-12);
        }
    }

    #[test]
    fn subgroup_reflection_is_an_involution_on_oracle_fixture() {
        let field = fixture();
        for subgroup in Su2Subgroup::ALL {
            let mut candidate = field.clone();
            let original = *candidate.link([0, 0, 0, 0], 0).unwrap();
            overrelax_subgroup_reference(&mut candidate, [0, 0, 0, 0], 0, subgroup).unwrap();
            overrelax_subgroup_reference(&mut candidate, [0, 0, 0, 0], 0, subgroup).unwrap();
            let restored = *candidate.link([0, 0, 0, 0], 0).unwrap();
            assert!(matrix_max_error(&original, &restored) < 2.0e-12);
        }
    }

    #[test]
    fn identity_fixture_is_a_valid_noop_reflection() {
        let mut field = WilsonGaugeField::identity([2, 2, 2, 2]).unwrap();
        let before = *field.link([0, 0, 0, 0], 0).unwrap();
        let result = overrelax_subgroup_reference(
            &mut field,
            [0, 0, 0, 0],
            0,
            Su2Subgroup::Pair01,
        )
        .unwrap();
        let after = *field.link([0, 0, 0, 0], 0).unwrap();
        assert!(matrix_max_error(&before, &after) < 1.0e-12);
        assert!(result.touching_trace_delta.abs() < 1.0e-12);
    }
}
