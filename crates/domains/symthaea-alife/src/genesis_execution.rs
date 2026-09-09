// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact-resume execution capsule for the Earth-forced Genesis social profile.
//!
//! This module composes four previously independent causal authorities at one quiescent evidence
//! boundary: Population, encounter scheduler, Earth forcing, and Genesis behavior+lifecycle
//! evidence. The capsule is deliberately profile-specific. It does not claim to serialize arbitrary
//! closures, arbitrary environments, or every future Genesis execution mode.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::earth_forcing::{
    EarthForcedEnvironmentSnapshotErrorV1, EarthForcedEnvironmentSnapshotV1,
    ValidatedEarthForcedEnvironmentSnapshotV1,
};
use crate::population::PopulationLiveSnapshotErrorV1;
use crate::{
    AgentId, EncounterScheduler, EncounterSchedulerSnapshotErrorV1, EncounterSchedulerSnapshotV1,
    GenesisEvidenceCheckpointErrorV1, GenesisEvidenceCheckpointV1,
    GenesisRollingEvidenceErrorV1, GenesisRollingEvidenceRunnerV1, PairingMode, PopulationConfig,
    PopulationSnapshotErrorV1, PopulationSnapshotV1, StepSummary,
    ValidatedEncounterSchedulerSnapshotV1, ValidatedGenesisEvidenceCheckpointV1,
    ValidatedPopulationSnapshotV1,
};
use crate::EarthForcedEnvironment;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GenesisExecutionProtocolV1 {
    EarthForcedSocialV1,
}

/// Serializable complete causal state for the Earth-forced Genesis social execution profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenesisEarthExecutionCapsuleV1 {
    protocol: GenesisExecutionProtocolV1,
    population: PopulationSnapshotV1,
    scheduler: EncounterSchedulerSnapshotV1,
    environment: EarthForcedEnvironmentSnapshotV1,
    evidence: GenesisEvidenceCheckpointV1,
}

/// Non-serializable execution authority produced only after all component and cross-authority checks.
#[derive(Debug, Clone)]
pub struct ValidatedGenesisEarthExecutionCapsuleV1 {
    protocol: GenesisExecutionProtocolV1,
    population: ValidatedPopulationSnapshotV1,
    scheduler: ValidatedEncounterSchedulerSnapshotV1,
    environment: ValidatedEarthForcedEnvironmentSnapshotV1,
    evidence: ValidatedGenesisEvidenceCheckpointV1,
}

#[derive(Debug)]
pub enum GenesisEarthExecutionCapsuleErrorV1 {
    Evidence(GenesisEvidenceCheckpointErrorV1),
    Population(PopulationSnapshotErrorV1),
    Scheduler(EncounterSchedulerSnapshotErrorV1),
    Environment(EarthForcedEnvironmentSnapshotErrorV1),
    EnvironmentClockMismatch { expected: u64, observed: u64 },
    UnknownSchedulerAgent { agent_id: AgentId },
    UnknownSchedulerPartner { partner_id: AgentId },
    LiveSchedulerAsymmetry {
        agent_id: AgentId,
        partner_id: AgentId,
    },
}

#[derive(Debug)]
pub enum GenesisEarthExecutionErrorV1 {
    AlreadyUnqualified,
    Rolling(GenesisRollingEvidenceErrorV1),
    PopulationLive(PopulationLiveSnapshotErrorV1),
    Capsule(GenesisEarthExecutionCapsuleErrorV1),
    EnvironmentMustStartAtTickZero { observed: u64 },
}

/// Persisted+validated result of one atomic execution checkpoint.
#[derive(Debug, Clone)]
pub struct GenesisEarthExecutionCheckpointV1 {
    persisted: GenesisEarthExecutionCapsuleV1,
    validated: ValidatedGenesisEarthExecutionCapsuleV1,
}

impl GenesisEarthExecutionCheckpointV1 {
    pub fn persisted(&self) -> &GenesisEarthExecutionCapsuleV1 {
        &self.persisted
    }

    pub fn validated(&self) -> &ValidatedGenesisEarthExecutionCapsuleV1 {
        &self.validated
    }

    pub fn into_persisted(self) -> GenesisEarthExecutionCapsuleV1 {
        self.persisted
    }
}

impl GenesisEarthExecutionCapsuleV1 {
    /// Validate a persisted capsule as the continuation after `previous_behavior_next_tick`.
    pub fn validate_after(
        self,
        previous_behavior_next_tick: u64,
    ) -> Result<ValidatedGenesisEarthExecutionCapsuleV1, GenesisEarthExecutionCapsuleErrorV1> {
        let evidence = self
            .evidence
            .validate_after(previous_behavior_next_tick)
            .map_err(GenesisEarthExecutionCapsuleErrorV1::Evidence)?;
        let population = self
            .population
            .validate_for_genesis_social(&evidence)
            .map_err(GenesisEarthExecutionCapsuleErrorV1::Population)?;
        let scheduler = self
            .scheduler
            .validate()
            .map_err(GenesisEarthExecutionCapsuleErrorV1::Scheduler)?;
        let environment = self
            .environment
            .validate()
            .map_err(GenesisEarthExecutionCapsuleErrorV1::Environment)?;

        if environment.tick() != evidence.behavior_next_tick() {
            return Err(GenesisEarthExecutionCapsuleErrorV1::EnvironmentClockMismatch {
                expected: evidence.behavior_next_tick(),
                observed: environment.tick(),
            });
        }

        validate_scheduler_against_lifecycle(&scheduler, &evidence)?;

        Ok(ValidatedGenesisEarthExecutionCapsuleV1 {
            protocol: self.protocol,
            population,
            scheduler,
            environment,
            evidence,
        })
    }
}

impl ValidatedGenesisEarthExecutionCapsuleV1 {
    pub fn protocol(&self) -> GenesisExecutionProtocolV1 {
        self.protocol
    }

    pub fn population(&self) -> &ValidatedPopulationSnapshotV1 {
        &self.population
    }

    pub fn scheduler(&self) -> &ValidatedEncounterSchedulerSnapshotV1 {
        &self.scheduler
    }

    pub fn environment(&self) -> &ValidatedEarthForcedEnvironmentSnapshotV1 {
        &self.environment
    }

    pub fn evidence(&self) -> &ValidatedGenesisEvidenceCheckpointV1 {
        &self.evidence
    }
}

fn validate_scheduler_against_lifecycle(
    scheduler: &ValidatedEncounterSchedulerSnapshotV1,
    evidence: &ValidatedGenesisEvidenceCheckpointV1,
) -> Result<(), GenesisEarthExecutionCapsuleErrorV1> {
    let lifecycle = evidence.lifecycle();
    let records = lifecycle.ledger().records();
    let alive = lifecycle.alive_ids();
    let fixed = scheduler.fixed_partners();

    let mut map = BTreeMap::new();
    for entry in fixed {
        if !records.contains_key(&entry.agent_id()) {
            return Err(GenesisEarthExecutionCapsuleErrorV1::UnknownSchedulerAgent {
                agent_id: entry.agent_id(),
            });
        }
        if !records.contains_key(&entry.partner_id()) {
            return Err(GenesisEarthExecutionCapsuleErrorV1::UnknownSchedulerPartner {
                partner_id: entry.partner_id(),
            });
        }
        map.insert(entry.agent_id(), entry.partner_id());
    }

    // Stale mappings involving dead agents are legitimate history. But if both endpoints are
    // currently alive, the live scheduler relation must still be mutual at a stable boundary.
    for (&agent_id, &partner_id) in &map {
        if alive.contains(&agent_id)
            && alive.contains(&partner_id)
            && map.get(&partner_id).copied() != Some(agent_id)
        {
            return Err(GenesisEarthExecutionCapsuleErrorV1::LiveSchedulerAsymmetry {
                agent_id,
                partner_id,
            });
        }
    }
    Ok(())
}

/// Closed Earth-forced execution profile: no external scheduler/environment mutation surface.
pub struct GenesisEarthExecutionV1 {
    runner: GenesisRollingEvidenceRunnerV1,
    scheduler: EncounterScheduler,
    environment: EarthForcedEnvironment,
    qualified: bool,
}

impl GenesisEarthExecutionV1 {
    pub fn new(
        cfg: PopulationConfig,
        initial_count: usize,
        seed_base: u64,
        pairing_mode: PairingMode,
        scheduler_seed: u64,
        environment: EarthForcedEnvironment,
    ) -> Result<Self, GenesisEarthExecutionErrorV1> {
        let environment_state = environment
            .snapshot_v1()
            .validate()
            .map_err(|error| {
                GenesisEarthExecutionErrorV1::Capsule(
                    GenesisEarthExecutionCapsuleErrorV1::Environment(error),
                )
            })?;
        if environment_state.tick() != 0 {
            return Err(GenesisEarthExecutionErrorV1::EnvironmentMustStartAtTickZero {
                observed: environment_state.tick(),
            });
        }

        Ok(Self {
            runner: GenesisRollingEvidenceRunnerV1::new(cfg, initial_count, seed_base),
            scheduler: EncounterScheduler::new(pairing_mode, scheduler_seed),
            environment,
            qualified: true,
        })
    }

    pub fn from_validated_capsule(
        capsule: &ValidatedGenesisEarthExecutionCapsuleV1,
    ) -> Result<Self, GenesisEarthExecutionErrorV1> {
        let runner = GenesisRollingEvidenceRunnerV1::from_validated_resume(
            capsule.population(),
            capsule.evidence(),
        )
        .map_err(GenesisEarthExecutionErrorV1::PopulationLive)?;
        Ok(Self {
            runner,
            scheduler: EncounterScheduler::from_validated_snapshot_v1(capsule.scheduler()),
            environment: EarthForcedEnvironment::from_validated_snapshot_v1(capsule.environment()),
            qualified: true,
        })
    }

    pub fn is_qualified(&self) -> bool {
        self.qualified && self.runner.is_qualified()
    }

    /// Execute one social tick with exactly one Earth-forcing step as the resource input.
    pub fn step_social(&mut self) -> Result<StepSummary, GenesisEarthExecutionErrorV1> {
        if !self.is_qualified() {
            return Err(GenesisEarthExecutionErrorV1::AlreadyUnqualified);
        }
        let runner = &mut self.runner;
        let scheduler = &mut self.scheduler;
        let environment = &mut self.environment;
        runner
            .step_social(|_| environment.step(), scheduler)
            .map_err(GenesisEarthExecutionErrorV1::Rolling)
    }

    pub fn cull_weakest(&mut self, n: usize) -> Result<usize, GenesisEarthExecutionErrorV1> {
        if !self.is_qualified() {
            return Err(GenesisEarthExecutionErrorV1::AlreadyUnqualified);
        }
        self.runner
            .cull_weakest(n)
            .map_err(GenesisEarthExecutionErrorV1::Rolling)
    }

    /// Atomically checkpoint evidence first, then bind Population, scheduler, and environment state
    /// to that exact validated boundary. Any post-drain failure permanently invalidates this wrapper.
    pub fn checkpoint_execution(
        &mut self,
    ) -> Result<GenesisEarthExecutionCheckpointV1, GenesisEarthExecutionErrorV1> {
        if !self.is_qualified() {
            return Err(GenesisEarthExecutionErrorV1::AlreadyUnqualified);
        }

        let evidence_checkpoint = self
            .runner
            .checkpoint_evidence()
            .map_err(GenesisEarthExecutionErrorV1::Rolling)?;
        let previous_behavior_next_tick = evidence_checkpoint.validated().behavior_start_tick();
        let population = match self
            .runner
            .population()
            .snapshot_v1_at_evidence_boundary(evidence_checkpoint.validated())
        {
            Ok(snapshot) => snapshot,
            Err(error) => {
                self.qualified = false;
                return Err(GenesisEarthExecutionErrorV1::PopulationLive(error));
            }
        };

        let raw = GenesisEarthExecutionCapsuleV1 {
            protocol: GenesisExecutionProtocolV1::EarthForcedSocialV1,
            population,
            scheduler: self.scheduler.snapshot_v1(),
            environment: self.environment.snapshot_v1(),
            evidence: evidence_checkpoint.into_persisted(),
        };
        let persisted = raw.clone();
        let validated = match raw.validate_after(previous_behavior_next_tick) {
            Ok(validated) => validated,
            Err(error) => {
                self.qualified = false;
                return Err(GenesisEarthExecutionErrorV1::Capsule(error));
            }
        };

        Ok(GenesisEarthExecutionCheckpointV1 {
            persisted,
            validated,
        })
    }

    pub fn population(&self) -> &crate::Population {
        self.runner.population()
    }

    pub fn scheduler_snapshot(&self) -> EncounterSchedulerSnapshotV1 {
        self.scheduler.snapshot_v1()
    }

    pub fn environment_snapshot(&self) -> EarthForcedEnvironmentSnapshotV1 {
        self.environment.snapshot_v1()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OrganismConfig;

    fn cfg() -> PopulationConfig {
        PopulationConfig {
            death_energy_threshold: -1.0,
            reproduction_energy_threshold: 2.0,
            reproduction_energy_cost: 0.4,
            organism_cfg: OrganismConfig {
                social_enabled: true,
                transfer_quantum: 0.03,
                action_temperature: 0.67,
                perceptual_grain: Some(0.1),
                ..OrganismConfig::default()
            },
            mutation_rate: 0.0,
            mutation_std: 0.0,
            ..Default::default()
        }
    }

    fn new_execution(mode: PairingMode) -> GenesisEarthExecutionV1 {
        GenesisEarthExecutionV1::new(
            cfg(),
            6,
            0xA11F_9001,
            mode,
            0x5CED_9002,
            EarthForcedEnvironment::earth_like(79.0).with_secular_drift(-0.015625),
        )
        .expect("fresh execution")
    }

    fn prefix(exec: &mut GenesisEarthExecutionV1, mode: PairingMode) {
        for _ in 0..4 {
            exec.step_social().expect("prefix tick");
        }
        if mode == PairingMode::FixedPartners {
            exec.cull_weakest(1).expect("create stale fixed-partner history");
        }
        for _ in 0..7 {
            exec.step_social().expect("prefix tick");
        }
    }

    fn suffix(exec: &mut GenesisEarthExecutionV1) {
        for _ in 0..23 {
            exec.step_social().expect("suffix tick");
        }
    }

    fn assert_exact_split_run(mode: PairingMode) {
        let mut uninterrupted = new_execution(mode);
        let mut split = new_execution(mode);

        prefix(&mut uninterrupted, mode);
        prefix(&mut split, mode);
        let first_uninterrupted = uninterrupted
            .checkpoint_execution()
            .expect("baseline prefix checkpoint");
        let first_split = split
            .checkpoint_execution()
            .expect("split prefix checkpoint");
        assert_eq!(
            serde_json::to_string(first_uninterrupted.persisted()).unwrap(),
            serde_json::to_string(first_split.persisted()).unwrap(),
            "deterministic prefix capsules differ before restore"
        );

        let encoded = serde_json::to_string(first_split.persisted()).expect("serialize capsule");
        let loaded: GenesisEarthExecutionCapsuleV1 =
            serde_json::from_str(&encoded).expect("deserialize capsule");
        let validated = loaded.validate_after(0).expect("validate loaded capsule");
        let mut restored =
            GenesisEarthExecutionV1::from_validated_capsule(&validated).expect("restore execution");

        suffix(&mut uninterrupted);
        suffix(&mut restored);
        let final_uninterrupted = uninterrupted
            .checkpoint_execution()
            .expect("baseline final checkpoint");
        let final_restored = restored
            .checkpoint_execution()
            .expect("restored final checkpoint");

        assert_eq!(
            serde_json::to_string(final_uninterrupted.persisted()).unwrap(),
            serde_json::to_string(final_restored.persisted()).unwrap(),
            "uninterrupted and restored executions diverged"
        );
    }

    #[test]
    fn random_scheduler_full_execution_is_exact_across_serialized_resume() {
        assert_exact_split_run(PairingMode::Random);
    }

    #[test]
    fn fixed_partner_full_execution_with_stale_history_is_exact_across_serialized_resume() {
        assert_exact_split_run(PairingMode::FixedPartners);
    }

    #[test]
    fn environment_clock_is_bound_to_behavior_clock() {
        let mut execution = new_execution(PairingMode::Random);
        for _ in 0..5 {
            execution.step_social().unwrap();
        }
        let checkpoint = execution.checkpoint_execution().unwrap();
        let mut value = serde_json::to_value(checkpoint.persisted()).unwrap();
        value["environment"]["tick"] = serde_json::json!(6u64);
        let tampered: GenesisEarthExecutionCapsuleV1 = serde_json::from_value(value).unwrap();
        assert!(matches!(
            tampered.validate_after(0),
            Err(GenesisEarthExecutionCapsuleErrorV1::EnvironmentClockMismatch {
                expected: 5,
                observed: 6
            })
        ));
    }

    #[test]
    fn scheduler_cannot_reference_identity_absent_from_lifecycle_history() {
        let mut execution = new_execution(PairingMode::FixedPartners);
        execution.step_social().unwrap();
        let checkpoint = execution.checkpoint_execution().unwrap();
        let mut value = serde_json::to_value(checkpoint.persisted()).unwrap();
        let entries = value["scheduler"]["fixed_partners"]
            .as_array_mut()
            .expect("fixed partner entries");
        assert!(!entries.is_empty());
        entries[0]["partner_id"] = serde_json::json!(9_999_999u64);
        let tampered: GenesisEarthExecutionCapsuleV1 = serde_json::from_value(value).unwrap();
        assert!(matches!(
            tampered.validate_after(0),
            Err(GenesisEarthExecutionCapsuleErrorV1::UnknownSchedulerPartner { .. })
        ));
    }
}
