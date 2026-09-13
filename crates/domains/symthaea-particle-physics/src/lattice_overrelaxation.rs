// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Microcanonical Cabibbo-Marinari overrelaxation using the shared subgroup force.
//!
//! This module contains only the equal-action reflection and sweep semantics.
//! Local Wilson force/staple/quaternion conventions live in
//! `lattice_subgroup_force` and are therefore shared with the heat-bath path.

use crate::lattice_gauge::{Site4, WilsonGaugeField, su3_mul};
use crate::lattice_metropolis::Su2Subgroup;
use crate::lattice_subgroup_force::{
    AffineSubgroupForce, SubgroupForceBackend, SubgroupForceError,
    embedded_subgroup_quaternion, evaluate_subgroup_force, touching_trace_sum,
};

const MICROCANONICAL_TOLERANCE: f64 = 1.0e-10;
const FORCE_EPSILON: f64 = 1.0e-28;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OverrelaxationStepResult {
    pub subgroup: Su2Subgroup,
    pub force: AffineSubgroupForce,
    pub reflection: [f64; 4],
    pub touching_trace_delta: f64,
    pub force_backend: SubgroupForceBackend,
    pub used_reference_fallback: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OverrelaxationSweepStats {
    pub subgroup_updates: usize,
    pub max_abs_touching_trace_delta: f64,
    pub reference_fallback_updates: usize,
    pub force_backend: SubgroupForceBackend,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LatticeOverrelaxationError {
    Force(SubgroupForceError),
    DegenerateForce([f64; 4]),
    NonUnitReflection([f64; 4]),
    MicrocanonicalDrift { before: f64, after: f64 },
}

impl From<SubgroupForceError> for LatticeOverrelaxationError {
    fn from(value: SubgroupForceError) -> Self { Self::Force(value) }
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
        return Err(LatticeOverrelaxationError::NonUnitReflection(reflection));
    }
    Ok(reflection)
}

pub fn overrelax_subgroup_with_backend(
    field: &mut WilsonGaugeField,
    site: Site4,
    mu: usize,
    subgroup: Su2Subgroup,
    backend: SubgroupForceBackend,
) -> Result<OverrelaxationStepResult, LatticeOverrelaxationError> {
    let before = touching_trace_sum(field, site, mu)?;
    let original = *field.link(site, mu).map_err(SubgroupForceError::Gauge)?;
    let evaluation = evaluate_subgroup_force(field, site, mu, subgroup, backend)?;
    let reflection = equal_action_reflection(evaluation.force.quaternion)?;
    let rotation = embedded_subgroup_quaternion(subgroup, reflection)?;
    field
        .set_link(site, mu, su3_mul(&rotation, &original))
        .map_err(SubgroupForceError::Gauge)?;
    let after = touching_trace_sum(field, site, mu)?;
    let drift = after - before;
    if !drift.is_finite() || drift.abs() > MICROCANONICAL_TOLERANCE {
        field
            .set_link(site, mu, original)
            .map_err(SubgroupForceError::Gauge)?;
        return Err(LatticeOverrelaxationError::MicrocanonicalDrift { before, after });
    }
    Ok(OverrelaxationStepResult {
        subgroup,
        force: evaluation.force,
        reflection,
        touching_trace_delta: drift,
        force_backend: backend,
        used_reference_fallback: evaluation.used_reference_fallback,
    })
}

pub fn overrelax_subgroup_reference(
    field: &mut WilsonGaugeField,
    site: Site4,
    mu: usize,
    subgroup: Su2Subgroup,
) -> Result<OverrelaxationStepResult, LatticeOverrelaxationError> {
    overrelax_subgroup_with_backend(field, site, mu, subgroup, SubgroupForceBackend::FiniteProbe)
}

pub fn overrelax_sweep_with_backend(
    field: &mut WilsonGaugeField,
    backend: SubgroupForceBackend,
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
                            let result = overrelax_subgroup_with_backend(
                                field, site, mu, subgroup, backend,
                            )?;
                            subgroup_updates += 1;
                            reference_fallback_updates += usize::from(result.used_reference_fallback);
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
        force_backend: backend,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_gauge::{Site4, Su3Matrix};
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
        let mut field = WilsonGaugeField::identity([2,2,2,2]).unwrap();
        install_rotation(&mut field, [0,0,0,0], 0, Su2Subgroup::Pair01, [1.0,2.0,3.0], 0.31);
        install_rotation(&mut field, [1,0,1,0], 2, Su2Subgroup::Pair02, [2.0,-1.0,1.0], -0.27);
        install_rotation(&mut field, [0,1,0,1], 3, Su2Subgroup::Pair12, [1.0,1.0,-2.0], 0.22);
        install_rotation(&mut field, [1,1,1,1], 1, Su2Subgroup::Pair01, [-2.0,1.0,1.0], 0.19);
        field
    }

    fn matrix_max_error(a: &Su3Matrix, b: &Su3Matrix) -> f64 {
        let mut error: f64 = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                let dr = a[i][j].re - b[i][j].re;
                let di = a[i][j].im - b[i][j].im;
                error = error.max((dr*dr + di*di).sqrt());
            }
        }
        error
    }

    #[test]
    fn reflection_remains_pinned_to_independent_oracle() {
        let force = [
            11.366_866_478_675_853,
            -0.957_654_316_646_473_3,
            -1.959_698_913_884_743_4,
            -3.029_933_415_654_614_5,
        ];
        let expected = [
            0.805_258_339_905_583_7,
            -0.152_092_350_615_349_3,
            -0.311_234_658_613_362_0,
            -0.481_206_722_910_876_8,
        ];
        let reflection = equal_action_reflection(force).unwrap();
        for i in 0..4 {
            assert!((reflection[i] - expected[i]).abs() < 1.0e-12);
        }
    }

    #[test]
    fn staple_and_reference_steps_produce_same_link() {
        let field = fixture();
        for subgroup in Su2Subgroup::ALL {
            let mut reference = field.clone();
            let mut optimized = field.clone();
            overrelax_subgroup_with_backend(
                &mut reference, [0,0,0,0], 0, subgroup, SubgroupForceBackend::FiniteProbe,
            ).unwrap();
            let result = overrelax_subgroup_with_backend(
                &mut optimized, [0,0,0,0], 0, subgroup, SubgroupForceBackend::Staple,
            ).unwrap();
            assert!(!result.used_reference_fallback);
            assert!(matrix_max_error(
                reference.link([0,0,0,0], 0).unwrap(),
                optimized.link([0,0,0,0], 0).unwrap(),
            ) < 2.0e-12);
        }
    }

    #[test]
    fn complete_staple_sweep_is_microcanonical() {
        let mut field = fixture();
        let before = field.wilson_action(5.7).unwrap();
        let stats = overrelax_sweep_with_backend(&mut field, SubgroupForceBackend::Staple).unwrap();
        let after = field.wilson_action(5.7).unwrap();
        assert_eq!(stats.subgroup_updates, 2*2*2*2*4*3);
        assert_eq!(stats.reference_fallback_updates, 0);
        assert!(stats.max_abs_touching_trace_delta < 1.0e-10);
        assert!((after-before).abs() < 2.0e-9);
    }

    #[test]
    fn degenerate_extent_uses_reference_fallback() {
        let mut field = WilsonGaugeField::identity([2,2,1,2]).unwrap();
        let result = overrelax_subgroup_with_backend(
            &mut field, [0,0,0,0], 0, Su2Subgroup::Pair01, SubgroupForceBackend::Staple,
        ).unwrap();
        assert!(result.used_reference_fallback);
    }
}
