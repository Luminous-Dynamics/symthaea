// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic sweep orchestration for pure-SU(3) Metropolis updates.
//!
//! Randomness is injected through `Uniform01Source`. This module defines the
//! proposal transform and deterministic sweep schedule, but does not certify the
//! statistical quality of any concrete RNG implementation.

use crate::lattice_gauge::WilsonGaugeField;
use crate::lattice_metropolis::{
    LatticeMetropolisError, Su2Subgroup, Su2SubgroupProposal, metropolis_subgroup_step,
};
use std::f64::consts::TAU;

pub trait Uniform01Source {
    /// Return the next nominal U(0,1) variate. Values are validated by callers.
    fn next_uniform(&mut self) -> f64;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymmetricProposalConfig {
    /// Maximum absolute subgroup rotation angle, in radians.
    pub max_angle: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SweepStats {
    pub attempted: usize,
    pub accepted: usize,
    /// Sum of action differences only for accepted transitions.
    pub accepted_delta_action_sum: f64,
    pub mean_acceptance_probability: f64,
}

impl SweepStats {
    pub fn acceptance_rate(self) -> f64 {
        if self.attempted == 0 {
            0.0
        } else {
            self.accepted as f64 / self.attempted as f64
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LatticeSweepError {
    Update(LatticeMetropolisError),
    InvalidMaxAngle(f64),
    InvalidProposalUniform(f64),
    InvalidAcceptanceUniform(f64),
}

impl From<LatticeMetropolisError> for LatticeSweepError {
    fn from(value: LatticeMetropolisError) -> Self {
        Self::Update(value)
    }
}

fn proposal_uniform(source: &mut impl Uniform01Source) -> Result<f64, LatticeSweepError> {
    let u = source.next_uniform();
    // Open interval ensures the inverse map u -> 1-u also remains valid.
    if !u.is_finite() || u <= 0.0 || u >= 1.0 {
        return Err(LatticeSweepError::InvalidProposalUniform(u));
    }
    Ok(u)
}

fn acceptance_uniform(source: &mut impl Uniform01Source) -> Result<f64, LatticeSweepError> {
    let u = source.next_uniform();
    if !u.is_finite() || !(0.0..1.0).contains(&u) {
        return Err(LatticeSweepError::InvalidAcceptanceUniform(u));
    }
    Ok(u)
}

/// Draw a symmetric embedded-SU(2) proposal from ideal independent U(0,1)
/// variates.
///
/// - the axis is uniform on S^2 through `(z, phi)`;
/// - the angle is uniform on `(-max_angle, max_angle)`;
/// - angle inversion corresponds to `u -> 1-u`, preserving proposal density.
///
/// This is a transform theorem only. It does not prove that a supplied source
/// actually produces independent uniform variates.
pub fn draw_symmetric_subgroup_proposal(
    source: &mut impl Uniform01Source,
    subgroup: Su2Subgroup,
    config: SymmetricProposalConfig,
) -> Result<Su2SubgroupProposal, LatticeSweepError> {
    if !config.max_angle.is_finite() || config.max_angle <= 0.0 || config.max_angle > std::f64::consts::PI {
        return Err(LatticeSweepError::InvalidMaxAngle(config.max_angle));
    }

    let uz = proposal_uniform(source)?;
    let uphi = proposal_uniform(source)?;
    let uangle = proposal_uniform(source)?;

    let z = 2.0 * uz - 1.0;
    let phi = TAU * uphi;
    let radial = (1.0 - z * z).max(0.0).sqrt();
    let axis = [radial * phi.cos(), radial * phi.sin(), z];
    let angle = config.max_angle * (2.0 * uangle - 1.0);

    Ok(Su2SubgroupProposal {
        subgroup,
        axis,
        angle,
    })
}

/// Perform one lexicographic sweep over every site, direction, and the three
/// canonical SU(2) subgroups.
///
/// Each local transition is delegated to `metropolis_subgroup_step`. Under an
/// ideal independent uniform source, the proposal transform is symmetric. The
/// deterministic composition of target-invariant local kernels remains target
/// invariant, but this function does not establish ergodicity or equilibration.
pub fn metropolis_sweep(
    field: &mut WilsonGaugeField,
    beta: f64,
    config: SymmetricProposalConfig,
    source: &mut impl Uniform01Source,
) -> Result<SweepStats, LatticeSweepError> {
    if !config.max_angle.is_finite() || config.max_angle <= 0.0 || config.max_angle > std::f64::consts::PI {
        return Err(LatticeSweepError::InvalidMaxAngle(config.max_angle));
    }

    let dims = field.dims();
    let mut attempted = 0usize;
    let mut accepted = 0usize;
    let mut accepted_delta_action_sum = 0.0;
    let mut acceptance_probability_sum = 0.0;

    for x in 0..dims[0] {
        for y in 0..dims[1] {
            for z in 0..dims[2] {
                for t in 0..dims[3] {
                    let site = [x, y, z, t];
                    for mu in 0..4 {
                        for subgroup in Su2Subgroup::ALL {
                            let proposal = draw_symmetric_subgroup_proposal(source, subgroup, config)?;
                            let draw = acceptance_uniform(source)?;
                            let result = metropolis_subgroup_step(
                                field, site, mu, beta, proposal, draw,
                            )?;
                            attempted += 1;
                            acceptance_probability_sum += result.acceptance_probability;
                            if result.accepted {
                                accepted += 1;
                                accepted_delta_action_sum += result.delta_action;
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(SweepStats {
        attempted,
        accepted,
        accepted_delta_action_sum,
        mean_acceptance_probability: if attempted == 0 {
            0.0
        } else {
            acceptance_probability_sum / attempted as f64
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_metropolis::embedded_su2_rotation;
    use crate::lattice_gauge::{Su3Matrix, su3_dagger};

    struct ScriptedSource {
        values: Vec<f64>,
        cursor: usize,
    }

    impl ScriptedSource {
        fn new(values: Vec<f64>) -> Self {
            Self { values, cursor: 0 }
        }
    }

    impl Uniform01Source for ScriptedSource {
        fn next_uniform(&mut self) -> f64 {
            let value = self.values[self.cursor];
            self.cursor += 1;
            value
        }
    }

    #[derive(Clone)]
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
    }

    impl Uniform01Source for SplitMix64 {
        fn next_uniform(&mut self) -> f64 {
            // Strictly inside (0,1), suitable for deterministic qualification.
            ((self.next_u64() >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
        }
    }

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

    #[test]
    fn proposal_transform_has_explicit_inverse_measure_map() {
        let config = SymmetricProposalConfig { max_angle: 0.4 };
        let mut forward_source = ScriptedSource::new(vec![0.25, 0.33, 0.8]);
        let mut reverse_source = ScriptedSource::new(vec![0.25, 0.33, 0.2]);
        let forward = draw_symmetric_subgroup_proposal(
            &mut forward_source,
            Su2Subgroup::Pair02,
            config,
        )
        .unwrap();
        let reverse = draw_symmetric_subgroup_proposal(
            &mut reverse_source,
            Su2Subgroup::Pair02,
            config,
        )
        .unwrap();
        assert!((forward.angle + reverse.angle).abs() < 1e-15);
        assert_eq!(forward.axis, reverse.axis);
        let f = embedded_su2_rotation(forward).unwrap();
        let r = embedded_su2_rotation(reverse).unwrap();
        assert!(matrix_max_error(&r, &su3_dagger(&f)) < 1e-12);
    }

    #[test]
    fn full_sweep_is_deterministically_replayable() {
        let mut a = WilsonGaugeField::identity([2, 2, 1, 1]).unwrap();
        let mut b = WilsonGaugeField::identity([2, 2, 1, 1]).unwrap();
        let mut rng_a = SplitMix64::new(0x5EED_0123_4567_89AB);
        let mut rng_b = rng_a.clone();
        let config = SymmetricProposalConfig { max_angle: 0.18 };
        let stats_a = metropolis_sweep(&mut a, 6.0, config, &mut rng_a).unwrap();
        let stats_b = metropolis_sweep(&mut b, 6.0, config, &mut rng_b).unwrap();
        assert_eq!(stats_a, stats_b);
        assert_eq!(stats_a.attempted, 2 * 2 * 1 * 1 * 4 * 3);
        assert!((a.wilson_action(6.0).unwrap() - b.wilson_action(6.0).unwrap()).abs() < 1e-14);
        assert!((a.average_plaquette().unwrap() - b.average_plaquette().unwrap()).abs() < 1e-14);
        assert!((0.0..=1.0).contains(&stats_a.acceptance_rate()));
        assert!((0.0..=1.0).contains(&stats_a.mean_acceptance_probability));
    }

    #[test]
    fn invalid_source_and_configuration_fail_closed() {
        let mut source = ScriptedSource::new(vec![1.0]);
        assert!(matches!(
            draw_symmetric_subgroup_proposal(
                &mut source,
                Su2Subgroup::Pair01,
                SymmetricProposalConfig { max_angle: 0.2 },
            ),
            Err(LatticeSweepError::InvalidProposalUniform(1.0))
        ));

        let mut field = WilsonGaugeField::identity([2, 2, 1, 1]).unwrap();
        let mut rng = SplitMix64::new(1);
        assert!(matches!(
            metropolis_sweep(
                &mut field,
                6.0,
                SymmetricProposalConfig { max_angle: 0.0 },
                &mut rng,
            ),
            Err(LatticeSweepError::InvalidMaxAngle(0.0))
        ));
    }
}
