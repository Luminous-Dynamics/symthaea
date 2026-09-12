// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Composed pure-SU(3) Cabibbo-Marinari heat-bath + overrelaxation cycle.
//!
//! A cycle is exactly one stochastic heat-bath pass through every link and all
//! three embedded SU(2) subgroups, followed by a caller-declared number of
//! deterministic microcanonical overrelaxation sweeps.
//!
//! The RNG source is accepted only by the heat-bath pass. Overrelaxation has no
//! RNG argument, making accidental random-stream consumption by the
//! microcanonical phase structurally impossible through this API.

use crate::lattice_gauge::WilsonGaugeField;
use crate::lattice_heatbath_su3::{
    Su3SubgroupHeatbathError, heatbath_subgroup_step_with_backend,
};
use crate::lattice_metropolis::Su2Subgroup;
use crate::lattice_overrelaxation::{
    LatticeOverrelaxationError, overrelax_sweep_with_backend,
};
use crate::lattice_subgroup_force::SubgroupForceBackend;
use crate::lattice_sweep::Uniform01Source;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeatbathOverrelaxationSchedule {
    pub force_backend: SubgroupForceBackend,
    /// Number of complete microcanonical sweeps after each stochastic heat-bath pass.
    pub overrelaxation_sweeps: usize,
    /// Per-subgroup scalar-rejection safety bound for the SU(2) heat-bath draw.
    pub max_heatbath_attempts: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeatbathSweepStats {
    pub subgroup_updates: usize,
    pub scalar_rejection_attempts: usize,
    pub reference_fallback_updates: usize,
    pub max_alpha: f64,
    pub mean_alpha: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeatbathOverrelaxationStats {
    pub heatbath: HeatbathSweepStats,
    pub overrelaxation_sweeps: usize,
    pub overrelaxation_subgroup_updates: usize,
    pub overrelaxation_reference_fallback_updates: usize,
    pub max_abs_overrelaxation_trace_drift: f64,
    pub force_backend: SubgroupForceBackend,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HeatbathOverrelaxationError {
    InvalidMaxHeatbathAttempts(usize),
    Heatbath(Su3SubgroupHeatbathError),
    Overrelaxation(LatticeOverrelaxationError),
}

impl From<Su3SubgroupHeatbathError> for HeatbathOverrelaxationError {
    fn from(value: Su3SubgroupHeatbathError) -> Self { Self::Heatbath(value) }
}
impl From<LatticeOverrelaxationError> for HeatbathOverrelaxationError {
    fn from(value: LatticeOverrelaxationError) -> Self { Self::Overrelaxation(value) }
}

impl HeatbathOverrelaxationSchedule {
    pub fn validate(self) -> Result<(), HeatbathOverrelaxationError> {
        if self.max_heatbath_attempts == 0 {
            return Err(HeatbathOverrelaxationError::InvalidMaxHeatbathAttempts(0));
        }
        Ok(())
    }

    pub fn identity(self) -> String {
        format!(
            "cabibbo_marinari_heatbath+{}or:{:?}",
            self.overrelaxation_sweeps,
            self.force_backend,
        )
        .to_ascii_lowercase()
    }
}

/// One complete stochastic heat-bath pass over all links/subgroups.
pub fn heatbath_sweep_with_backend(
    field: &mut WilsonGaugeField,
    beta: f64,
    backend: SubgroupForceBackend,
    source: &mut impl Uniform01Source,
    max_attempts: usize,
) -> Result<HeatbathSweepStats, HeatbathOverrelaxationError> {
    if max_attempts == 0 {
        return Err(HeatbathOverrelaxationError::InvalidMaxHeatbathAttempts(0));
    }

    let dims = field.dims();
    let mut subgroup_updates = 0usize;
    let mut scalar_rejection_attempts = 0usize;
    let mut reference_fallback_updates = 0usize;
    let mut alpha_sum = 0.0;
    let mut max_alpha: f64 = 0.0;

    for x in 0..dims[0] {
        for y in 0..dims[1] {
            for z in 0..dims[2] {
                for t in 0..dims[3] {
                    let site = [x, y, z, t];
                    for mu in 0..4 {
                        for subgroup in Su2Subgroup::ALL {
                            let result = heatbath_subgroup_step_with_backend(
                                field,
                                site,
                                mu,
                                subgroup,
                                beta,
                                backend,
                                source,
                                max_attempts,
                            )?;
                            subgroup_updates += 1;
                            scalar_rejection_attempts += result.draw.canonical.scalar_attempts;
                            reference_fallback_updates +=
                                usize::from(result.draw.used_reference_fallback);
                            alpha_sum += result.draw.force.alpha;
                            max_alpha = max_alpha.max(result.draw.force.alpha);
                        }
                    }
                }
            }
        }
    }

    Ok(HeatbathSweepStats {
        subgroup_updates,
        scalar_rejection_attempts,
        reference_fallback_updates,
        max_alpha,
        mean_alpha: alpha_sum / subgroup_updates as f64,
    })
}

/// Execute one stochastic heat-bath pass followed by the declared number of
/// deterministic microcanonical sweeps.
pub fn heatbath_overrelaxation_cycle(
    field: &mut WilsonGaugeField,
    beta: f64,
    schedule: HeatbathOverrelaxationSchedule,
    source: &mut impl Uniform01Source,
) -> Result<HeatbathOverrelaxationStats, HeatbathOverrelaxationError> {
    schedule.validate()?;
    let heatbath = heatbath_sweep_with_backend(
        field,
        beta,
        schedule.force_backend,
        source,
        schedule.max_heatbath_attempts,
    )?;

    let mut overrelaxation_subgroup_updates = 0usize;
    let mut overrelaxation_reference_fallback_updates = 0usize;
    let mut max_abs_overrelaxation_trace_drift: f64 = 0.0;
    for _ in 0..schedule.overrelaxation_sweeps {
        let stats = overrelax_sweep_with_backend(field, schedule.force_backend)?;
        overrelaxation_subgroup_updates += stats.subgroup_updates;
        overrelaxation_reference_fallback_updates += stats.reference_fallback_updates;
        max_abs_overrelaxation_trace_drift = max_abs_overrelaxation_trace_drift
            .max(stats.max_abs_touching_trace_delta);
    }

    Ok(HeatbathOverrelaxationStats {
        heatbath,
        overrelaxation_sweeps: schedule.overrelaxation_sweeps,
        overrelaxation_subgroup_updates,
        overrelaxation_reference_fallback_updates,
        max_abs_overrelaxation_trace_drift,
        force_backend: schedule.force_backend,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_rng::{LatticeChaCha8Stream, LatticeStreamCoordinates, LatticeStreamDomain};

    #[derive(Clone)]
    struct CountingSource<S> {
        inner: S,
        draws: usize,
    }

    impl<S: Uniform01Source> Uniform01Source for CountingSource<S> {
        fn next_uniform(&mut self) -> f64 {
            self.draws += 1;
            self.inner.next_uniform()
        }
    }

    fn source(replica: u16) -> CountingSource<LatticeChaCha8Stream> {
        CountingSource {
            inner: LatticeChaCha8Stream::new(
                [0x9d; 32],
                LatticeStreamCoordinates {
                    domain: LatticeStreamDomain::Qualification,
                    ensemble_slot: 0x1618,
                    replica,
                    rank: 0,
                },
            ).unwrap(),
            draws: 0,
        }
    }

    fn schedule(overrelaxation_sweeps: usize) -> HeatbathOverrelaxationSchedule {
        HeatbathOverrelaxationSchedule {
            force_backend: SubgroupForceBackend::Staple,
            overrelaxation_sweeps,
            max_heatbath_attempts: 256,
        }
    }

    #[test]
    fn one_heatbath_pass_visits_every_subgroup_once() {
        let mut field = WilsonGaugeField::identity([2,2,2,2]).unwrap();
        let mut rng = source(1);
        let stats = heatbath_sweep_with_backend(
            &mut field,
            5.7,
            SubgroupForceBackend::Staple,
            &mut rng,
            256,
        ).unwrap();
        assert_eq!(stats.subgroup_updates, 2*2*2*2*4*3);
        assert_eq!(stats.reference_fallback_updates, 0);
        assert!(stats.scalar_rejection_attempts >= stats.subgroup_updates);
        assert!(stats.mean_alpha.is_finite() && stats.mean_alpha >= 0.0);
        assert!(stats.max_alpha >= stats.mean_alpha);
    }

    #[test]
    fn overrelaxation_count_does_not_change_rng_consumption_in_one_cycle() {
        let mut no_or_field = WilsonGaugeField::identity([2,2,2,2]).unwrap();
        let mut with_or_field = WilsonGaugeField::identity([2,2,2,2]).unwrap();
        let mut no_or_rng = source(2);
        let mut with_or_rng = source(2);

        heatbath_overrelaxation_cycle(&mut no_or_field, 5.7, schedule(0), &mut no_or_rng).unwrap();
        heatbath_overrelaxation_cycle(&mut with_or_field, 5.7, schedule(3), &mut with_or_rng).unwrap();
        assert_eq!(no_or_rng.draws, with_or_rng.draws);
    }

    #[test]
    fn schedule_reports_exact_microcanonical_work() {
        let mut field = WilsonGaugeField::identity([2,2,2,2]).unwrap();
        let mut rng = source(3);
        let stats = heatbath_overrelaxation_cycle(&mut field, 5.7, schedule(2), &mut rng).unwrap();
        let updates_per_sweep = 2*2*2*2*4*3;
        assert_eq!(stats.heatbath.subgroup_updates, updates_per_sweep);
        assert_eq!(stats.overrelaxation_sweeps, 2);
        assert_eq!(stats.overrelaxation_subgroup_updates, 2 * updates_per_sweep);
        assert_eq!(stats.overrelaxation_reference_fallback_updates, 0);
        assert!(stats.max_abs_overrelaxation_trace_drift < 1.0e-10);
    }

    #[test]
    fn cycle_is_deterministically_replayable() {
        let mut a = WilsonGaugeField::identity([2,2,2,2]).unwrap();
        let mut b = WilsonGaugeField::identity([2,2,2,2]).unwrap();
        let mut ra = source(4);
        let mut rb = source(4);
        let sa = heatbath_overrelaxation_cycle(&mut a, 5.7, schedule(1), &mut ra).unwrap();
        let sb = heatbath_overrelaxation_cycle(&mut b, 5.7, schedule(1), &mut rb).unwrap();
        assert_eq!(sa, sb);
        assert_eq!(ra.draws, rb.draws);
        assert!((a.average_plaquette().unwrap() - b.average_plaquette().unwrap()).abs() < 1.0e-14);
        assert!((a.wilson_action(5.7).unwrap() - b.wilson_action(5.7).unwrap()).abs() < 1.0e-12);
    }

    #[test]
    fn degenerate_lattice_reports_fallbacks_for_both_phases() {
        let mut field = WilsonGaugeField::identity([2,2,1,2]).unwrap();
        let mut rng = source(5);
        let stats = heatbath_overrelaxation_cycle(&mut field, 5.7, schedule(1), &mut rng).unwrap();
        assert_eq!(stats.heatbath.reference_fallback_updates, stats.heatbath.subgroup_updates);
        assert_eq!(
            stats.overrelaxation_reference_fallback_updates,
            stats.overrelaxation_subgroup_updates,
        );
    }
}
