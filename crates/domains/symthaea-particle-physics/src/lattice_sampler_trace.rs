// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sampler-neutral qualification traces for pure-SU(3) ensemble experiments.
//!
//! This module gives random-walk Metropolis and heat-bath+overrelaxation the
//! same burn-in / measurement schedule and observable schema while preserving
//! algorithm-specific work counters. It deliberately does not infer
//! equilibration, convergence, ESS adequacy, or sampler superiority.

use crate::lattice_gauge::{LatticeGaugeError, WilsonGaugeField};
use crate::lattice_hb_or_sweep::{
    HeatbathOverrelaxationError, HeatbathOverrelaxationSchedule,
    heatbath_overrelaxation_cycle,
};
use crate::lattice_sweep::{
    LatticeSweepError, SymmetricProposalConfig, Uniform01Source, metropolis_sweep,
};
use crate::symmetry_groups::Complex;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QualificationSampler {
    RandomWalkMetropolis { proposal_max_angle: f64 },
    HeatbathOverrelaxation { schedule: HeatbathOverrelaxationSchedule },
}

impl QualificationSampler {
    /// Version-stable evidence identity for the transition kernel.
    pub fn stable_id(self) -> String {
        match self {
            Self::RandomWalkMetropolis { proposal_max_angle } => {
                format!("cm_metropolis_v1:max_angle={proposal_max_angle:.17e}")
            }
            Self::HeatbathOverrelaxation { schedule } => schedule.identity(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SamplerTracePlan {
    pub dims: [usize; 4],
    pub beta: f64,
    /// Complete transition cycles discarded before any retained measurement.
    pub burn_in_cycles: u64,
    /// Complete transition cycles between retained measurements.
    pub measurement_stride: u64,
    pub measurements: usize,
    pub sampler: QualificationSampler,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerTraceSample {
    pub cycle: u64,
    pub wilson_action: f64,
    pub average_plaquette: f64,
    pub spatial_mean_polyakov: Complex,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SamplerWorkCounters {
    pub transition_cycles: u64,
    pub stochastic_subgroup_updates: u64,
    pub metropolis_accepted_updates: u64,
    pub heatbath_scalar_rejection_attempts: u64,
    pub overrelaxation_subgroup_updates: u64,
    pub reference_fallback_updates: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SamplerQualificationTrace {
    pub sampler_id: String,
    pub samples: Vec<SamplerTraceSample>,
    pub work: SamplerWorkCounters,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SamplerTraceError {
    Gauge(LatticeGaugeError),
    Metropolis(LatticeSweepError),
    HeatbathOverrelaxation(HeatbathOverrelaxationError),
    InvalidExtent([usize; 4]),
    FieldExtentMismatch { expected: [usize; 4], actual: [usize; 4] },
    InvalidBeta(f64),
    InvalidMeasurementStride(u64),
    InvalidMeasurementCount(usize),
    MeasurementScheduleOverflow,
}

impl From<LatticeGaugeError> for SamplerTraceError {
    fn from(value: LatticeGaugeError) -> Self { Self::Gauge(value) }
}
impl From<LatticeSweepError> for SamplerTraceError {
    fn from(value: LatticeSweepError) -> Self { Self::Metropolis(value) }
}
impl From<HeatbathOverrelaxationError> for SamplerTraceError {
    fn from(value: HeatbathOverrelaxationError) -> Self { Self::HeatbathOverrelaxation(value) }
}

impl SamplerTracePlan {
    pub fn measurement_schedule(&self) -> Result<Vec<u64>, SamplerTraceError> {
        if self.dims.iter().any(|&n| n == 0) {
            return Err(SamplerTraceError::InvalidExtent(self.dims));
        }
        if !self.beta.is_finite() || self.beta <= 0.0 {
            return Err(SamplerTraceError::InvalidBeta(self.beta));
        }
        if self.measurement_stride == 0 {
            return Err(SamplerTraceError::InvalidMeasurementStride(0));
        }
        if self.measurements == 0 {
            return Err(SamplerTraceError::InvalidMeasurementCount(0));
        }
        let mut schedule = Vec::with_capacity(self.measurements);
        for i in 1..=self.measurements {
            let offset = self
                .measurement_stride
                .checked_mul(i as u64)
                .ok_or(SamplerTraceError::MeasurementScheduleOverflow)?;
            schedule.push(
                self.burn_in_cycles
                    .checked_add(offset)
                    .ok_or(SamplerTraceError::MeasurementScheduleOverflow)?,
            );
        }
        Ok(schedule)
    }
}

fn spatial_mean_polyakov(field: &WilsonGaugeField) -> Result<Complex, LatticeGaugeError> {
    let dims = field.dims();
    let mut re = 0.0;
    let mut im = 0.0;
    let mut count = 0usize;
    for x in 0..dims[0] {
        for y in 0..dims[1] {
            for z in 0..dims[2] {
                let p = field.polyakov_loop([x, y, z])?;
                re += p.re;
                im += p.im;
                count += 1;
            }
        }
    }
    Ok(Complex::new(re / count as f64, im / count as f64))
}

fn execute_cycle(
    field: &mut WilsonGaugeField,
    beta: f64,
    sampler: QualificationSampler,
    source: &mut impl Uniform01Source,
    work: &mut SamplerWorkCounters,
) -> Result<(), SamplerTraceError> {
    match sampler {
        QualificationSampler::RandomWalkMetropolis { proposal_max_angle } => {
            let stats = metropolis_sweep(
                field,
                beta,
                SymmetricProposalConfig { max_angle: proposal_max_angle },
                source,
            )?;
            work.stochastic_subgroup_updates += stats.attempted as u64;
            work.metropolis_accepted_updates += stats.accepted as u64;
        }
        QualificationSampler::HeatbathOverrelaxation { schedule } => {
            let stats = heatbath_overrelaxation_cycle(field, beta, schedule, source)?;
            work.stochastic_subgroup_updates += stats.heatbath.subgroup_updates as u64;
            work.heatbath_scalar_rejection_attempts +=
                stats.heatbath.scalar_rejection_attempts as u64;
            work.overrelaxation_subgroup_updates +=
                stats.overrelaxation_subgroup_updates as u64;
            work.reference_fallback_updates +=
                (stats.heatbath.reference_fallback_updates
                    + stats.overrelaxation_reference_fallback_updates) as u64;
        }
    }
    work.transition_cycles += 1;
    Ok(())
}

pub fn run_sampler_qualification_trace(
    field: &mut WilsonGaugeField,
    plan: &SamplerTracePlan,
    source: &mut impl Uniform01Source,
) -> Result<SamplerQualificationTrace, SamplerTraceError> {
    let schedule = plan.measurement_schedule()?;
    if field.dims() != plan.dims {
        return Err(SamplerTraceError::FieldExtentMismatch {
            expected: plan.dims,
            actual: field.dims(),
        });
    }

    let final_cycle = *schedule.last().expect("validated non-empty schedule");
    let mut cursor = 0usize;
    let mut samples = Vec::with_capacity(schedule.len());
    let mut work = SamplerWorkCounters::default();

    for cycle in 1..=final_cycle {
        execute_cycle(field, plan.beta, plan.sampler, source, &mut work)?;
        if cursor < schedule.len() && cycle == schedule[cursor] {
            samples.push(SamplerTraceSample {
                cycle,
                wilson_action: field.wilson_action(plan.beta)?,
                average_plaquette: field.average_plaquette()?,
                spatial_mean_polyakov: spatial_mean_polyakov(field)?,
            });
            cursor += 1;
        }
    }

    Ok(SamplerQualificationTrace {
        sampler_id: plan.sampler.stable_id(),
        samples,
        work,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_rng::{LatticeChaCha8Stream, LatticeStreamCoordinates, LatticeStreamDomain};
    use crate::lattice_subgroup_force::SubgroupForceBackend;

    fn source(replica: u16) -> LatticeChaCha8Stream {
        LatticeChaCha8Stream::new(
            [0xac; 32],
            LatticeStreamCoordinates {
                domain: LatticeStreamDomain::Qualification,
                ensemble_slot: 0x1619,
                replica,
                rank: 0,
            },
        ).unwrap()
    }

    fn plan(sampler: QualificationSampler) -> SamplerTracePlan {
        SamplerTracePlan {
            dims: [2,2,2,2],
            beta: 5.7,
            burn_in_cycles: 1,
            measurement_stride: 1,
            measurements: 2,
            sampler,
        }
    }

    #[test]
    fn sampler_ids_are_explicitly_versioned() {
        assert_eq!(
            QualificationSampler::RandomWalkMetropolis { proposal_max_angle: 0.5 }.stable_id(),
            "cm_metropolis_v1:max_angle=5.00000000000000000e-1"
        );
        assert_eq!(
            QualificationSampler::HeatbathOverrelaxation {
                schedule: HeatbathOverrelaxationSchedule {
                    force_backend: SubgroupForceBackend::Staple,
                    overrelaxation_sweeps: 2,
                    max_heatbath_attempts: 256,
                },
            }.stable_id(),
            "cm_heatbath_or_v1:force=staple:or_sweeps=2:max_attempts=256"
        );
    }

    #[test]
    fn schedule_never_samples_at_burnin_boundary() {
        let p = plan(QualificationSampler::RandomWalkMetropolis { proposal_max_angle: 0.5 });
        assert_eq!(p.measurement_schedule().unwrap(), vec![2,3]);
    }

    #[test]
    fn both_samplers_emit_the_same_trace_shape() {
        let metropolis = QualificationSampler::RandomWalkMetropolis { proposal_max_angle: 0.5 };
        let hbor = QualificationSampler::HeatbathOverrelaxation {
            schedule: HeatbathOverrelaxationSchedule {
                force_backend: SubgroupForceBackend::Staple,
                overrelaxation_sweeps: 1,
                max_heatbath_attempts: 256,
            },
        };

        let mut field_a = WilsonGaugeField::identity([2,2,2,2]).unwrap();
        let mut field_b = WilsonGaugeField::identity([2,2,2,2]).unwrap();
        let mut rng_a = source(1);
        let mut rng_b = source(2);
        let a = run_sampler_qualification_trace(&mut field_a, &plan(metropolis), &mut rng_a).unwrap();
        let b = run_sampler_qualification_trace(&mut field_b, &plan(hbor), &mut rng_b).unwrap();
        assert_eq!(a.samples.len(), 2);
        assert_eq!(b.samples.len(), 2);
        assert_eq!(a.samples.iter().map(|s| s.cycle).collect::<Vec<_>>(), vec![2,3]);
        assert_eq!(b.samples.iter().map(|s| s.cycle).collect::<Vec<_>>(), vec![2,3]);
        assert!(a.work.stochastic_subgroup_updates > 0);
        assert!(a.work.metropolis_accepted_updates <= a.work.stochastic_subgroup_updates);
        assert_eq!(a.work.heatbath_scalar_rejection_attempts, 0);
        assert!(b.work.stochastic_subgroup_updates > 0);
        assert!(b.work.heatbath_scalar_rejection_attempts >= b.work.stochastic_subgroup_updates);
        assert!(b.work.overrelaxation_subgroup_updates > 0);
    }

    #[test]
    fn deterministic_replay_is_exact_within_each_sampler() {
        for sampler in [
            QualificationSampler::RandomWalkMetropolis { proposal_max_angle: 0.5 },
            QualificationSampler::HeatbathOverrelaxation {
                schedule: HeatbathOverrelaxationSchedule {
                    force_backend: SubgroupForceBackend::Staple,
                    overrelaxation_sweeps: 1,
                    max_heatbath_attempts: 256,
                },
            },
        ] {
            let mut a = WilsonGaugeField::identity([2,2,2,2]).unwrap();
            let mut b = WilsonGaugeField::identity([2,2,2,2]).unwrap();
            let mut ra = source(7);
            let mut rb = source(7);
            let ta = run_sampler_qualification_trace(&mut a, &plan(sampler), &mut ra).unwrap();
            let tb = run_sampler_qualification_trace(&mut b, &plan(sampler), &mut rb).unwrap();
            assert_eq!(ta, tb);
        }
    }

    #[test]
    fn trace_contains_no_convergence_or_winner_field() {
        let mut field = WilsonGaugeField::identity([2,2,2,2]).unwrap();
        let mut rng = source(9);
        let trace = run_sampler_qualification_trace(
            &mut field,
            &plan(QualificationSampler::RandomWalkMetropolis { proposal_max_angle: 0.5 }),
            &mut rng,
        ).unwrap();
        assert_eq!(trace.samples.len(), 2);
        assert_eq!(trace.work.transition_cycles, 3);
    }
}
