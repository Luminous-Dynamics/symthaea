// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Qualified social-step execution wrapper for complete Genesis behavioral evidence.
//!
//! This wrapper is intentionally narrower than `Population`: it owns a fresh population and allows
//! social stepping only through a path that preflights monotonic Genesis tick authority and turns
//! the legacy flat `GenesisEvent` rows into one validated complete [`GenesisTickBatchV1`] per
//! `step_social` call. Callers receive read-only access to the underlying population, so they cannot
//! bypass this wrapper's social-step evidence boundary through the wrapper API itself.
//!
//! The raw `Population` type remains publicly mutable elsewhere in the crate/workspace; issue #910
//! tracks that broader authority migration. Therefore this wrapper establishes a qualified execution
//! profile for callers that opt into it, not a proof that every existing `Population` usage is
//! evidence-safe.

use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

use crate::{
    EncounterScheduler, GenesisTickBatchErrorV1, GenesisTickBatchV1, GenesisTickCursorErrorV1,
    GenesisTickCursorSnapshotV1, GenesisTickCursorV1, LifecycleEventV1, Population,
    PopulationConfig, StepSummary,
};

#[derive(Debug, Clone, PartialEq)]
pub enum GenesisSocialRunnerErrorV1 {
    AlreadyUnqualified,
    Tick(GenesisTickCursorErrorV1),
    PopulationCountOverflow,
    Batch(GenesisTickBatchErrorV1),
}

/// Social-only qualified execution profile for Genesis behavioral evidence.
pub struct GenesisSocialRunnerV1 {
    population: Population,
    tick_cursor: GenesisTickCursorV1,
    completed_batches: Vec<GenesisTickBatchV1>,
    qualified: bool,
}

impl GenesisSocialRunnerV1 {
    /// Construct a fresh population and a fresh Genesis tick authority at tick zero.
    pub fn new(cfg: PopulationConfig, initial_count: usize, seed_base: u64) -> Self {
        Self {
            population: Population::new(cfg, initial_count, seed_base),
            tick_cursor: GenesisTickCursorV1::new(),
            completed_batches: Vec::new(),
            qualified: true,
        }
    }

    /// Execute exactly one social step and append one complete validated tick batch.
    ///
    /// Tick exhaustion is checked before `resource_for` can run because cursor preflight happens
    /// before delegating to `Population::step_social`. After the population step returns, the
    /// wrapper drains every legacy flat row emitted by that call, validates the complete batch, and
    /// only then commits the wrapper tick reservation.
    ///
    /// If a post-step evidence invariant fails, the wrapper marks itself permanently unqualified:
    /// the population has already changed, so silently retrying the same tick would fabricate a
    /// history. The error is returned and all later qualified steps fail closed.
    ///
    /// The delegated call is also wrapped in `catch_unwind` solely so a caller that catches a
    /// panic cannot continue using this wrapper as though its qualification survived a potentially
    /// partial population/environment transition. The original panic is immediately resumed after
    /// the qualification bit is cleared.
    pub fn step_social(
        &mut self,
        resource_for: impl FnMut(usize) -> f64,
        scheduler: &mut EncounterScheduler,
    ) -> Result<StepSummary, GenesisSocialRunnerErrorV1> {
        if !self.qualified {
            return Err(GenesisSocialRunnerErrorV1::AlreadyUnqualified);
        }

        let reservation = self
            .tick_cursor
            .preflight()
            .map_err(GenesisSocialRunnerErrorV1::Tick)?;
        let tick = reservation.tick();
        let population_before = u64::try_from(self.population.len())
            .map_err(|_| GenesisSocialRunnerErrorV1::PopulationCountOverflow)?;

        let step_result = catch_unwind(AssertUnwindSafe(|| {
            self.population.step_social(resource_for, scheduler)
        }));
        let summary = match step_result {
            Ok(summary) => summary,
            Err(payload) => {
                self.qualified = false;
                resume_unwind(payload);
            }
        };

        let events = self.population.drain_event_log();
        let batch = GenesisTickBatchV1::new(tick, population_before, events);
        let validated = match batch.validate() {
            Ok(validated) => validated,
            Err(error) => {
                self.qualified = false;
                return Err(GenesisSocialRunnerErrorV1::Batch(error));
            }
        };

        if let Err(error) = self.tick_cursor.commit(reservation) {
            self.qualified = false;
            return Err(GenesisSocialRunnerErrorV1::Tick(error));
        }
        self.completed_batches.push(validated.into_batch());
        Ok(summary)
    }

    /// Authoritative ecological cull that keeps `Population` lifecycle accounting intact.
    ///
    /// This does not advance the social tick cursor. The resulting lifecycle transition remains in
    /// the population recorder and may be drained independently through [`Self::drain_lifecycle_events`].
    pub fn cull_weakest(&mut self, n: usize) -> Result<usize, GenesisSocialRunnerErrorV1> {
        if !self.qualified {
            return Err(GenesisSocialRunnerErrorV1::AlreadyUnqualified);
        }
        Ok(self.population.cull_weakest(n))
    }

    /// Read-only access to the owned population.
    pub fn population(&self) -> &Population {
        &self.population
    }

    pub fn len(&self) -> usize {
        self.population.len()
    }

    pub fn is_empty(&self) -> bool {
        self.population.is_empty()
    }

    pub fn is_qualified(&self) -> bool {
        self.qualified
    }

    pub fn tick_cursor_snapshot(&self) -> GenesisTickCursorSnapshotV1 {
        self.tick_cursor.snapshot()
    }

    pub fn completed_batches(&self) -> &[GenesisTickBatchV1] {
        &self.completed_batches
    }

    pub fn lifecycle_epoch(&self) -> u64 {
        self.population.lifecycle_epoch()
    }

    pub fn lifecycle_events(&self) -> &[LifecycleEventV1] {
        self.population.lifecycle_events()
    }

    /// Drain complete social-step evidence without resetting monotonic tick authority.
    pub fn drain_completed_batches(&mut self) -> Vec<GenesisTickBatchV1> {
        std::mem::take(&mut self.completed_batches)
    }

    /// Drain lifecycle evidence through the population's authoritative recorder boundary.
    pub fn drain_lifecycle_events(&mut self) -> Vec<LifecycleEventV1> {
        self.population.drain_lifecycle_events()
    }

    /// Consume the qualified wrapper and return the raw population.
    ///
    /// This intentionally ends the wrapper's qualification boundary; no behavioral continuation
    /// claim is returned with the population.
    pub fn into_population(self) -> Population {
        self.population
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OrganismConfig, PairingMode, PopulationConfig, validate_genesis_tick_chunk};

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
    fn one_social_step_produces_one_complete_validated_batch() {
        let mut runner = GenesisSocialRunnerV1::new(quiet_cfg(), 4, 17);
        let mut scheduler = EncounterScheduler::new(PairingMode::Random, 99);
        runner
            .step_social(|_| 0.5, &mut scheduler)
            .expect("qualified social step");

        assert!(runner.is_qualified());
        assert_eq!(runner.tick_cursor_snapshot().next_tick(), 1);
        assert_eq!(runner.completed_batches().len(), 1);
        let batch = &runner.completed_batches()[0];
        let validated = batch.clone().validate().expect("stored batch remains valid");
        assert_eq!(validated.tick(), 0);
        assert_eq!(validated.population_before(), 4);
        assert_eq!(validated.events().len(), 4);
    }

    #[test]
    fn empty_population_social_step_is_not_silently_lost() {
        let mut runner = GenesisSocialRunnerV1::new(quiet_cfg(), 0, 17);
        let mut scheduler = EncounterScheduler::new(PairingMode::Random, 99);
        let summary = runner
            .step_social(|_| 0.5, &mut scheduler)
            .expect("empty social tick remains valid");
        assert_eq!(summary.population, 0);
        assert_eq!(runner.completed_batches().len(), 1);
        let validated = runner.completed_batches()[0]
            .clone()
            .validate()
            .expect("empty batch");
        assert_eq!(validated.population_before(), 0);
        assert!(validated.events().is_empty());
        assert_eq!(runner.tick_cursor_snapshot().next_tick(), 1);
    }

    #[test]
    fn drained_batches_rejoin_as_an_exact_contiguous_tick_chunk() {
        let mut runner = GenesisSocialRunnerV1::new(quiet_cfg(), 3, 17);
        let mut scheduler = EncounterScheduler::new(PairingMode::Random, 99);
        runner
            .step_social(|_| 0.5, &mut scheduler)
            .expect("tick zero");
        runner
            .step_social(|_| 0.5, &mut scheduler)
            .expect("tick one");

        let chunk = runner.drain_completed_batches();
        let report = validate_genesis_tick_chunk(0, &chunk).expect("contiguous chunk");
        assert_eq!(report.next_tick, 2);
        assert_eq!(report.batch_count, 2);
        assert_eq!(runner.tick_cursor_snapshot().next_tick(), 2);
        assert!(runner.completed_batches().is_empty());
    }

    #[test]
    fn wrapper_drains_legacy_flat_rows_after_each_qualified_step() {
        let mut runner = GenesisSocialRunnerV1::new(quiet_cfg(), 2, 17);
        let mut scheduler = EncounterScheduler::new(PairingMode::Random, 99);
        runner
            .step_social(|_| 0.5, &mut scheduler)
            .expect("qualified social step");
        assert!(
            runner.population.drain_event_log().is_empty(),
            "legacy flat rows must be consumed into the complete batch by the wrapper"
        );
    }

    #[test]
    fn ecological_cull_preserves_lifecycle_authority_without_advancing_social_tick() {
        let mut runner = GenesisSocialRunnerV1::new(quiet_cfg(), 3, 17);
        let tick_before = runner.tick_cursor_snapshot();
        let lifecycle_before = runner.lifecycle_events().len();
        assert_eq!(runner.cull_weakest(1), Ok(1));
        assert_eq!(runner.tick_cursor_snapshot(), tick_before);
        assert_eq!(runner.lifecycle_events().len(), lifecycle_before + 1);
    }

    #[test]
    fn caught_inner_panic_permanently_invalidates_the_runner() {
        let mut runner = GenesisSocialRunnerV1::new(quiet_cfg(), 2, 17);
        let mut scheduler = EncounterScheduler::new(PairingMode::Random, 99);
        let before = runner.tick_cursor_snapshot();

        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _ = runner.step_social(
                |_| -> f64 { panic!("synthetic resource failure") },
                &mut scheduler,
            );
        }));
        assert!(panic.is_err());
        assert!(!runner.is_qualified());
        assert_eq!(runner.tick_cursor_snapshot(), before);
        assert_eq!(
            runner.step_social(|_| 0.5, &mut scheduler),
            Err(GenesisSocialRunnerErrorV1::AlreadyUnqualified)
        );
    }
}

mod resume;
