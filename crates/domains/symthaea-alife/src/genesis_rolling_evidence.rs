// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Atomic rolling evidence checkpoints for the qualified Genesis social execution profile.
//!
//! [`crate::GenesisSocialRunnerV1`] intentionally exposes low-level drains because it predates the
//! cross-evidence checkpoint contract. This stricter wrapper owns that runner privately and never
//! exposes those drains. Callers may execute social steps, perform the authoritative ecological
//! cull, inspect the population read-only, and atomically checkpoint behavior + lifecycle evidence.
//!
//! A successful checkpoint advances two validated continuation boundaries together:
//!
//! - the next complete behavioral tick;
//! - the lossless validated lifecycle prefix.
//!
//! If any post-drain lifecycle or cross-evidence validation fails, this wrapper permanently marks
//! itself unqualified. Retrying would be unsafe because the underlying evidence buffers have already
//! been consumed.

use crate::{
    EncounterScheduler, GenesisEvidenceCheckpointErrorV1, GenesisEvidenceCheckpointV1,
    GenesisSocialRunnerErrorV1, GenesisSocialRunnerV1, LifecycleCheckpointErrorV1,
    LifecycleCheckpointV1, Population, PopulationConfig, StepSummary,
    ValidatedGenesisEvidenceCheckpointV1, ValidatedLifecycleCheckpointV1,
};

#[derive(Debug, Clone, PartialEq)]
pub enum GenesisRollingEvidenceErrorV1 {
    AlreadyUnqualified,
    Runner(GenesisSocialRunnerErrorV1),
    Lifecycle(LifecycleCheckpointErrorV1),
    Evidence(GenesisEvidenceCheckpointErrorV1),
}

impl From<GenesisSocialRunnerErrorV1> for GenesisRollingEvidenceErrorV1 {
    fn from(value: GenesisSocialRunnerErrorV1) -> Self {
        Self::Runner(value)
    }
}

/// One persisted raw checkpoint paired with the non-serializable capability that validated it.
#[derive(Debug, Clone)]
pub struct GenesisRollingEvidenceCheckpointV1 {
    persisted: GenesisEvidenceCheckpointV1,
    validated: ValidatedGenesisEvidenceCheckpointV1,
}

impl GenesisRollingEvidenceCheckpointV1 {
    /// Serializable raw checkpoint. It must be revalidated after deserialization before use.
    pub fn persisted(&self) -> &GenesisEvidenceCheckpointV1 {
        &self.persisted
    }

    /// In-process authority produced by the same atomic checkpoint operation.
    pub fn validated(&self) -> &ValidatedGenesisEvidenceCheckpointV1 {
        &self.validated
    }

    pub fn into_persisted(self) -> GenesisEvidenceCheckpointV1 {
        self.persisted
    }
}

/// Qualified social execution plus one rolling cross-evidence continuation boundary.
pub struct GenesisRollingEvidenceRunnerV1 {
    runner: GenesisSocialRunnerV1,
    lifecycle: Option<ValidatedLifecycleCheckpointV1>,
    behavior_next_tick: u64,
    qualified: bool,
}

impl GenesisRollingEvidenceRunnerV1 {
    pub fn new(cfg: PopulationConfig, initial_count: usize, seed_base: u64) -> Self {
        Self {
            runner: GenesisSocialRunnerV1::new(cfg, initial_count, seed_base),
            lifecycle: None,
            behavior_next_tick: 0,
            qualified: true,
        }
    }

    pub fn step_social(
        &mut self,
        resource_for: impl FnMut(usize) -> f64,
        scheduler: &mut EncounterScheduler,
    ) -> Result<StepSummary, GenesisRollingEvidenceErrorV1> {
        if !self.is_qualified() {
            return Err(GenesisRollingEvidenceErrorV1::AlreadyUnqualified);
        }
        self.runner
            .step_social(resource_for, scheduler)
            .map_err(GenesisRollingEvidenceErrorV1::Runner)
    }

    pub fn cull_weakest(&mut self, n: usize) -> Result<usize, GenesisRollingEvidenceErrorV1> {
        if !self.is_qualified() {
            return Err(GenesisRollingEvidenceErrorV1::AlreadyUnqualified);
        }
        self.runner
            .cull_weakest(n)
            .map_err(GenesisRollingEvidenceErrorV1::Runner)
    }

    pub fn population(&self) -> &Population {
        self.runner.population()
    }

    pub fn is_qualified(&self) -> bool {
        self.qualified && self.runner.is_qualified()
    }

    pub fn behavior_next_tick(&self) -> u64 {
        self.behavior_next_tick
    }

    pub fn lifecycle_checkpoint(&self) -> Option<&ValidatedLifecycleCheckpointV1> {
        self.lifecycle.as_ref()
    }

    /// Atomically consume all evidence accumulated since the previous successful checkpoint.
    ///
    /// The first call normalizes the complete lifecycle prefix beginning with founders. Later calls
    /// validate only the newly drained lifecycle suffix against the retained validated prefix. The
    /// resulting complete lifecycle checkpoint is then cross-validated with the complete behavioral
    /// tick chunk before either continuation boundary is advanced.
    pub fn checkpoint_evidence(
        &mut self,
    ) -> Result<GenesisRollingEvidenceCheckpointV1, GenesisRollingEvidenceErrorV1> {
        if !self.is_qualified() {
            return Err(GenesisRollingEvidenceErrorV1::AlreadyUnqualified);
        }

        let behavior_start_tick = self.behavior_next_tick;
        let behavior_next_tick = self.runner.tick_cursor_snapshot().next_tick();
        let behavior_batches = self.runner.drain_completed_batches();
        let lifecycle_events = self.runner.drain_lifecycle_events();
        let resulting_epoch = self.runner.lifecycle_epoch();

        let lifecycle = match self.lifecycle.take() {
            Some(previous) => previous.validate_chunk(&lifecycle_events, resulting_epoch),
            None => LifecycleCheckpointV1::from_complete_prefix(&lifecycle_events, resulting_epoch),
        };
        let lifecycle = match lifecycle {
            Ok(lifecycle) => lifecycle,
            Err(error) => {
                self.qualified = false;
                return Err(GenesisRollingEvidenceErrorV1::Lifecycle(error));
            }
        };

        let raw = GenesisEvidenceCheckpointV1::new(
            behavior_start_tick,
            behavior_next_tick,
            behavior_batches,
            lifecycle.clone().into_checkpoint(),
        );
        let persisted = raw.clone();
        let validated = match raw.validate_after(behavior_start_tick) {
            Ok(validated) => validated,
            Err(error) => {
                self.qualified = false;
                return Err(GenesisRollingEvidenceErrorV1::Evidence(error));
            }
        };

        self.behavior_next_tick = validated.behavior_next_tick();
        self.lifecycle = Some(validated.lifecycle().clone());

        Ok(GenesisRollingEvidenceCheckpointV1 {
            persisted,
            validated,
        })
    }

    /// Consume the qualified profile and return the raw population, ending this evidence boundary.
    pub fn into_population(self) -> Population {
        self.runner.into_population()
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

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
    fn rolling_checkpoints_advance_behavior_and_lifecycle_together() {
        let mut runner = GenesisRollingEvidenceRunnerV1::new(quiet_cfg(), 3, 17);
        let mut scheduler = EncounterScheduler::new(PairingMode::Random, 99);

        runner.step_social(|_| 0.5, &mut scheduler).expect("tick 0");
        let first = runner.checkpoint_evidence().expect("first checkpoint");
        assert_eq!(first.validated().behavior_start_tick(), 0);
        assert_eq!(first.validated().behavior_next_tick(), 1);
        assert_eq!(first.validated().lifecycle().continuation_epoch(), 1);
        assert_eq!(runner.behavior_next_tick(), 1);

        runner.step_social(|_| 0.5, &mut scheduler).expect("tick 1");
        let second = runner.checkpoint_evidence().expect("second checkpoint");
        assert_eq!(second.validated().behavior_start_tick(), 1);
        assert_eq!(second.validated().behavior_next_tick(), 2);
        assert_eq!(second.validated().lifecycle().continuation_epoch(), 2);
        assert_eq!(runner.behavior_next_tick(), 2);
    }

    #[test]
    fn persisted_rolling_checkpoint_requires_revalidation_after_json_round_trip() {
        let mut runner = GenesisRollingEvidenceRunnerV1::new(quiet_cfg(), 2, 23);
        let mut scheduler = EncounterScheduler::new(PairingMode::Random, 101);
        runner.step_social(|_| 0.5, &mut scheduler).expect("tick 0");
        let checkpoint = runner.checkpoint_evidence().expect("checkpoint");

        let json = serde_json::to_string(checkpoint.persisted()).expect("serialize");
        let raw: GenesisEvidenceCheckpointV1 = serde_json::from_str(&json).expect("deserialize");
        let revalidated = raw.validate_after(0).expect("revalidate persisted checkpoint");
        assert_eq!(revalidated.behavior_next_tick(), 1);
        assert_eq!(revalidated.lifecycle().continuation_epoch(), 1);
    }

    #[test]
    fn cull_only_checkpoint_updates_lifecycle_without_advancing_behavior_clock() {
        let mut runner = GenesisRollingEvidenceRunnerV1::new(quiet_cfg(), 3, 31);
        let mut scheduler = EncounterScheduler::new(PairingMode::Random, 103);
        runner.step_social(|_| 0.5, &mut scheduler).expect("tick 0");
        runner.checkpoint_evidence().expect("baseline checkpoint");

        runner.cull_weakest(1).expect("authoritative cull");
        let cull_checkpoint = runner.checkpoint_evidence().expect("cull checkpoint");
        assert_eq!(cull_checkpoint.validated().behavior_start_tick(), 1);
        assert_eq!(cull_checkpoint.validated().behavior_next_tick(), 1);
        assert_eq!(cull_checkpoint.validated().lifecycle().continuation_epoch(), 1);
        assert_eq!(cull_checkpoint.validated().lifecycle().death_count(), 1);

        runner.step_social(|_| 0.5, &mut scheduler).expect("tick 1");
        let next = runner.checkpoint_evidence().expect("post-cull social checkpoint");
        assert_eq!(next.validated().behavior_start_tick(), 1);
        assert_eq!(next.validated().behavior_next_tick(), 2);
        assert_eq!(next.validated().behavior_batches()[0].clone().validate().unwrap().population_before(), 2);
    }

    #[test]
    fn caught_inner_panic_prevents_later_atomic_checkpoint_claims() {
        let mut runner = GenesisRollingEvidenceRunnerV1::new(quiet_cfg(), 2, 41);
        let mut scheduler = EncounterScheduler::new(PairingMode::Random, 107);
        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _ = runner.step_social(
                |_| -> f64 { panic!("synthetic environment failure") },
                &mut scheduler,
            );
        }));
        assert!(panic.is_err());
        assert!(!runner.is_qualified());
        assert!(matches!(
            runner.checkpoint_evidence(),
            Err(GenesisRollingEvidenceErrorV1::AlreadyUnqualified)
        ));
    }
}

mod resume;
