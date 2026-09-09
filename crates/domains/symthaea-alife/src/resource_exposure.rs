// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! First-class evidence for the external resource scalar actually supplied to a Genesis social tick.
//!
//! Existing [`crate::GenesisEvent`] rows record organism energy before/after an action, not the
//! external environment signal consumed by `Population::step_social`. That distinction matters for
//! causal rescue experiments: storing a perturbation schedule does not prove the natural population
//! actually received the scheduled treatment.
//!
//! [`GenesisResourceExposureRunnerV1`] therefore wraps the qualified social runner and records the
//! exact finite scalar returned by the caller's resource function on each successful social step.
//! A separate session validator reconstructs Earth forcing from the fork capsule, reapplies the
//! frozen perturbation schedule, and requires bit-exact equality to the recorded consumed values and
//! exact tick/population alignment with the natural behavioral batches.
//!
//! Persisted exposure rows remain semantic evidence, not authenticated origin evidence. The runner
//! establishes an opt-in qualified emission profile; it does not prove every raw Population caller
//! used this path.

use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

use serde::{Deserialize, Serialize};

use crate::{
    EarthForcedEnvironment, EncounterScheduler, GenesisSocialRunnerErrorV1, GenesisSocialRunnerV1,
    GenesisTickBatchErrorV1, GenesisTickBatchV1, LifecycleEventV1, PerturbationError, Population,
    PopulationConfig, StepSummary, ValidatedCausalRescueReplaySessionV1,
};

/// Exact external resource consumed by one qualified social tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenesisResourceExposureV1 {
    tick: u64,
    population_before: u64,
    resource_bits: u64,
}

impl GenesisResourceExposureV1 {
    pub fn tick(self) -> u64 {
        self.tick
    }

    pub fn population_before(self) -> u64 {
        self.population_before
    }

    pub fn resource(self) -> f64 {
        f64::from_bits(self.resource_bits)
    }

    pub fn resource_bits(self) -> u64 {
        self.resource_bits
    }
}

/// Non-serializable capability proving one exposure interval matches the session's natural
/// behavior interval and the declared Earth-forcing perturbation schedule.
#[derive(Debug, Clone)]
pub struct ValidatedGenesisTreatmentExposureV1 {
    exposures: Vec<GenesisResourceExposureV1>,
    start_tick: u64,
    next_tick: u64,
}

impl ValidatedGenesisTreatmentExposureV1 {
    pub fn exposures(&self) -> &[GenesisResourceExposureV1] {
        &self.exposures
    }

    pub fn start_tick(&self) -> u64 {
        self.start_tick
    }

    pub fn next_tick(&self) -> u64 {
        self.next_tick
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GenesisResourceExposureErrorV1 {
    AlreadyUnqualified,
    Runner(GenesisSocialRunnerErrorV1),
    PopulationCountOverflow,
    MissingResourceCallback { tick: u64 },
    MultipleResourceCallbacks { tick: u64, calls: u64 },
    ResourcePopulationMismatch { tick: u64, expected: u64, observed: u64 },
    ExposureCountMismatch { expected: usize, observed: usize },
    TickMismatch { expected: u64, observed: u64 },
    Behavior(GenesisTickBatchErrorV1),
    BehaviorPopulationMismatch { tick: u64, expected: u64, observed: u64 },
    Perturbation(PerturbationError),
    ResourceMismatch { tick: u64, expected_bits: u64, observed_bits: u64 },
    EnvironmentEndpointMismatch,
}

impl From<GenesisSocialRunnerErrorV1> for GenesisResourceExposureErrorV1 {
    fn from(value: GenesisSocialRunnerErrorV1) -> Self {
        Self::Runner(value)
    }
}

/// Qualified social execution that records the exact external resource returned to Population.
pub struct GenesisResourceExposureRunnerV1 {
    runner: GenesisSocialRunnerV1,
    completed_exposures: Vec<GenesisResourceExposureV1>,
    qualified: bool,
}

impl GenesisResourceExposureRunnerV1 {
    pub fn new(cfg: PopulationConfig, initial_count: usize, seed_base: u64) -> Self {
        Self {
            runner: GenesisSocialRunnerV1::new(cfg, initial_count, seed_base),
            completed_exposures: Vec::new(),
            qualified: true,
        }
    }

    /// Execute one qualified social step while capturing exactly the scalar Population consumed.
    ///
    /// The current production `Population::step_social` invokes `resource_for(n)` exactly once. This
    /// wrapper nevertheless measures the callback count at runtime and fails closed if that contract
    /// changes. A non-finite resource panics before it can be returned to Population; the inner
    /// runner clears its qualification bit and this wrapper clears its own before resuming the panic.
    pub fn step_social(
        &mut self,
        mut resource_for: impl FnMut(usize) -> f64,
        scheduler: &mut EncounterScheduler,
    ) -> Result<StepSummary, GenesisResourceExposureErrorV1> {
        if !self.is_qualified() {
            return Err(GenesisResourceExposureErrorV1::AlreadyUnqualified);
        }

        let tick = self.runner.tick_cursor_snapshot().next_tick();
        let expected_population = u64::try_from(self.runner.len())
            .map_err(|_| GenesisResourceExposureErrorV1::PopulationCountOverflow)?;
        let mut calls = 0u64;
        let mut observed_population = None;
        let mut observed_resource_bits = None;

        let step = catch_unwind(AssertUnwindSafe(|| {
            self.runner.step_social(
                |population_before| {
                    calls = calls.checked_add(1).expect("resource callback count overflow");
                    let resource = resource_for(population_before);
                    if !resource.is_finite() {
                        panic!("qualified Genesis resource exposure must be finite");
                    }
                    if observed_population.is_none() {
                        observed_population = u64::try_from(population_before).ok();
                        observed_resource_bits = Some(resource.to_bits());
                    }
                    resource
                },
                scheduler,
            )
        }));
        let summary = match step {
            Ok(result) => result.map_err(GenesisResourceExposureErrorV1::Runner)?,
            Err(payload) => {
                self.qualified = false;
                resume_unwind(payload);
            }
        };

        if calls != 1 {
            self.qualified = false;
            return if calls == 0 {
                Err(GenesisResourceExposureErrorV1::MissingResourceCallback { tick })
            } else {
                Err(GenesisResourceExposureErrorV1::MultipleResourceCallbacks { tick, calls })
            };
        }
        let observed_population = match observed_population {
            Some(value) => value,
            None => {
                self.qualified = false;
                return Err(GenesisResourceExposureErrorV1::MissingResourceCallback { tick });
            }
        };
        if observed_population != expected_population {
            self.qualified = false;
            return Err(GenesisResourceExposureErrorV1::ResourcePopulationMismatch {
                tick,
                expected: expected_population,
                observed: observed_population,
            });
        }
        let resource_bits = match observed_resource_bits {
            Some(bits) => bits,
            None => {
                self.qualified = false;
                return Err(GenesisResourceExposureErrorV1::MissingResourceCallback { tick });
            }
        };

        self.completed_exposures.push(GenesisResourceExposureV1 {
            tick,
            population_before: observed_population,
            resource_bits,
        });
        Ok(summary)
    }

    pub fn cull_weakest(&mut self, n: usize) -> Result<usize, GenesisResourceExposureErrorV1> {
        if !self.is_qualified() {
            return Err(GenesisResourceExposureErrorV1::AlreadyUnqualified);
        }
        self.runner.cull_weakest(n).map_err(Into::into)
    }

    pub fn population(&self) -> &Population {
        self.runner.population()
    }

    pub fn is_qualified(&self) -> bool {
        self.qualified && self.runner.is_qualified()
    }

    pub fn completed_exposures(&self) -> &[GenesisResourceExposureV1] {
        &self.completed_exposures
    }

    pub fn completed_batches(&self) -> &[GenesisTickBatchV1] {
        self.runner.completed_batches()
    }

    pub fn drain_completed_exposures(&mut self) -> Vec<GenesisResourceExposureV1> {
        std::mem::take(&mut self.completed_exposures)
    }

    pub fn drain_completed_batches(&mut self) -> Vec<GenesisTickBatchV1> {
        self.runner.drain_completed_batches()
    }

    pub fn lifecycle_events(&self) -> &[LifecycleEventV1] {
        self.runner.lifecycle_events()
    }

    pub fn drain_lifecycle_events(&mut self) -> Vec<LifecycleEventV1> {
        self.runner.drain_lifecycle_events()
    }
}

/// Validate that persisted exposure rows match the treatment protocol bound into a validated causal
/// replay session. This proves schedule/environment/resource semantic consistency. Authentic origin
/// of the persisted rows remains a separate evidence claim.
pub fn validate_resource_exposure_for_session_v1(
    session: &ValidatedCausalRescueReplaySessionV1,
    exposures: &[GenesisResourceExposureV1],
) -> Result<ValidatedGenesisTreatmentExposureV1, GenesisResourceExposureErrorV1> {
    let start_tick = session.fork().evidence().behavior_next_tick();
    let next_tick = session.natural_end().evidence().behavior_next_tick();
    let batches = session.natural_end().evidence().behavior_batches();
    if exposures.len() != batches.len() {
        return Err(GenesisResourceExposureErrorV1::ExposureCountMismatch {
            expected: batches.len(),
            observed: exposures.len(),
        });
    }

    let mut environment = EarthForcedEnvironment::from_validated_snapshot_v1(
        session.fork().environment(),
    );
    let schedule = session.perturbation_schedule();

    for (index, (exposure, raw_batch)) in exposures.iter().copied().zip(batches).enumerate() {
        let expected_tick = start_tick
            .checked_add(index as u64)
            .expect("validated behavior interval length must fit u64");
        if exposure.tick != expected_tick {
            return Err(GenesisResourceExposureErrorV1::TickMismatch {
                expected: expected_tick,
                observed: exposure.tick,
            });
        }
        let batch = raw_batch
            .clone()
            .validate()
            .map_err(GenesisResourceExposureErrorV1::Behavior)?;
        if exposure.population_before != batch.population_before() {
            return Err(GenesisResourceExposureErrorV1::BehaviorPopulationMismatch {
                tick: exposure.tick,
                expected: batch.population_before(),
                observed: exposure.population_before,
            });
        }

        let baseline = environment.step();
        let expected_resource = schedule
            .apply_resource(exposure.tick, baseline)
            .map_err(GenesisResourceExposureErrorV1::Perturbation)?;
        if exposure.resource_bits != expected_resource.to_bits() {
            return Err(GenesisResourceExposureErrorV1::ResourceMismatch {
                tick: exposure.tick,
                expected_bits: expected_resource.to_bits(),
                observed_bits: exposure.resource_bits,
            });
        }
    }

    let expected_environment = EarthForcedEnvironment::from_validated_snapshot_v1(
        session.natural_end().environment(),
    )
    .snapshot_v1();
    if environment.snapshot_v1() != expected_environment {
        return Err(GenesisResourceExposureErrorV1::EnvironmentEndpointMismatch);
    }

    Ok(ValidatedGenesisTreatmentExposureV1 {
        exposures: exposures.to_vec(),
        start_tick,
        next_tick,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OrganismConfig, PairingMode};

    fn quiet_cfg() -> PopulationConfig {
        PopulationConfig {
            death_energy_threshold: -1.0,
            reproduction_energy_threshold: 2.0,
            reproduction_energy_cost: 0.4,
            organism_cfg: OrganismConfig::default(),
            ..Default::default()
        }
    }

    #[test]
    fn qualified_runner_records_exact_scalar_returned_to_population() {
        let mut runner = GenesisResourceExposureRunnerV1::new(quiet_cfg(), 3, 17);
        let mut scheduler = EncounterScheduler::new(PairingMode::Random, 99);
        let mut callback_calls = 0usize;
        runner
            .step_social(
                |population| {
                    callback_calls += 1;
                    0.25 + population as f64 * 0.03125
                },
                &mut scheduler,
            )
            .expect("qualified resource exposure step");

        assert_eq!(callback_calls, 1);
        assert_eq!(runner.completed_exposures().len(), 1);
        let exposure = runner.completed_exposures()[0];
        assert_eq!(exposure.tick(), 0);
        assert_eq!(exposure.population_before(), 3);
        assert_eq!(exposure.resource().to_bits(), (0.34375f64).to_bits());
        let batch = runner.completed_batches()[0].clone().validate().unwrap();
        assert_eq!(batch.tick(), exposure.tick());
        assert_eq!(batch.population_before(), exposure.population_before());
    }

    #[test]
    fn drain_keeps_resource_and_behavior_chunks_available_as_separate_explicit_evidence() {
        let mut runner = GenesisResourceExposureRunnerV1::new(quiet_cfg(), 2, 23);
        let mut scheduler = EncounterScheduler::new(PairingMode::Random, 101);
        for tick in 0..3 {
            runner
                .step_social(|_| 0.4 + tick as f64 * 0.05, &mut scheduler)
                .unwrap();
        }
        let exposures = runner.drain_completed_exposures();
        let batches = runner.drain_completed_batches();
        assert_eq!(exposures.len(), 3);
        assert_eq!(batches.len(), 3);
        for (exposure, batch) in exposures.iter().zip(batches) {
            let batch = batch.validate().unwrap();
            assert_eq!(exposure.tick(), batch.tick());
            assert_eq!(exposure.population_before(), batch.population_before());
        }
    }
}
