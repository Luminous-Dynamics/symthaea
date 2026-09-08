// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-authority persistence contract for one population at a stable Genesis evidence boundary.
//!
//! Population vector order is causal state. `Population::step` scans organisms in live order,
//! `step_social` passes that order to encounter scheduling, and `cull_weakest` itself reorders the
//! vector before removing agents. This snapshot therefore preserves organism order exactly and
//! deliberately does **not** sort organisms into a canonical ID order.
//!
//! Validation is scoped to the qualified Genesis social evidence profile. A raw snapshot becomes a
//! [`ValidatedPopulationSnapshotV1`] only when it agrees with a previously validated
//! [`crate::ValidatedGenesisEvidenceCheckpointV1`]: lifecycle history authorizes identity/seed
//! cursors and birth/death counters, while the behavioral continuation boundary authorizes the
//! population social clock. This establishes semantic consistency, not cryptographic provenance of
//! the persisted bytes; a future execution capsule must bind snapshot origin/protocol/environment.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{
    AgentId, AgentIdAllocator, AgentIdAllocatorRestoreErrorV1, AgentIdAllocatorSnapshotV1,
    EvolutionRngRestoreErrorV1, EvolutionRngSnapshotV1, EvolutionRngStreamsV1, Genome,
    OrganismSeedAllocatorErrorV1, OrganismSeedAllocatorSnapshotV1, OrganismSeedAllocatorV1,
    OrganismSnapshotErrorV1, OrganismSnapshotV1, PopulationConfigSnapshotErrorV1,
    PopulationConfigSnapshotV1, PopulationOrganismTemplateSnapshotV1,
    ValidatedGenesisEvidenceCheckpointV1, ValidatedOrganismSnapshotV1,
    ValidatedPopulationConfigSnapshotV1,
};

/// Raw serializable population-owned causal state.
///
/// `organisms` is the exact live `Vec` order. Scheduler and environment state are intentionally
/// absent; they belong to the higher execution capsule rather than to `Population` itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationSnapshotV1 {
    config: PopulationConfigSnapshotV1,
    organisms: Vec<OrganismSnapshotV1>,
    organism_seed_allocator: OrganismSeedAllocatorSnapshotV1,
    evolution_rng: EvolutionRngSnapshotV1,
    id_allocator: AgentIdAllocatorSnapshotV1,
    total_births: u64,
    total_deaths: u64,
    current_tick: u64,
}

/// Non-serializable restore capability produced only after cross-checking the population snapshot
/// against a qualified behavior+lifecycle evidence boundary.
#[derive(Debug, Clone)]
pub struct ValidatedPopulationSnapshotV1 {
    config: ValidatedPopulationConfigSnapshotV1,
    organisms: Vec<ValidatedOrganismSnapshotV1>,
    organism_seed_allocator: OrganismSeedAllocatorV1,
    evolution_rng: EvolutionRngStreamsV1,
    id_allocator: AgentIdAllocator,
    total_births: u64,
    total_deaths: u64,
    current_tick: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PopulationSnapshotErrorV1 {
    Config(PopulationConfigSnapshotErrorV1),
    Organism {
        index: usize,
        error: OrganismSnapshotErrorV1,
    },
    OrganismSeedAllocator(OrganismSeedAllocatorErrorV1),
    EvolutionRng(EvolutionRngRestoreErrorV1),
    IdAllocator(AgentIdAllocatorRestoreErrorV1),
    DuplicateLiveAgentId {
        agent_id: AgentId,
    },
    LiveSetMismatch {
        expected: BTreeSet<AgentId>,
        observed: BTreeSet<AgentId>,
    },
    MissingLifecycleRecord {
        agent_id: AgentId,
    },
    LifecycleIdentityMismatch {
        agent_id: AgentId,
    },
    OrganismPolicyMismatch {
        agent_id: AgentId,
    },
    BehaviorClockMismatch {
        expected: u64,
        observed: u64,
    },
    CountOverflow {
        field: &'static str,
    },
    BirthCountMismatch {
        expected: u64,
        observed: u64,
    },
    DeathCountMismatch {
        expected: u64,
        observed: u64,
    },
    PopulationAccountingMismatch {
        expected: u64,
        observed: u64,
    },
}

impl PopulationSnapshotV1 {
    /// Crate-owned construction seam for the future live `Population` wiring tranche.
    ///
    /// Keeping fields private prevents callers from treating a bag of separately serialized
    /// cursors as restore authority. Validation still remains mandatory after construction/load.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        config: PopulationConfigSnapshotV1,
        organisms: Vec<OrganismSnapshotV1>,
        organism_seed_allocator: OrganismSeedAllocatorSnapshotV1,
        evolution_rng: EvolutionRngSnapshotV1,
        id_allocator: AgentIdAllocatorSnapshotV1,
        total_births: u64,
        total_deaths: u64,
        current_tick: u64,
    ) -> Self {
        Self {
            config,
            organisms,
            organism_seed_allocator,
            evolution_rng,
            id_allocator,
            total_births,
            total_deaths,
            current_tick,
        }
    }

    /// Validate this population against one already-qualified Genesis behavior+lifecycle boundary.
    ///
    /// The raw value is borrowed rather than consumed so storage code may keep the persisted bytes;
    /// all nested raw snapshots are cloned before their consuming validators run.
    pub fn validate_for_genesis_social(
        &self,
        evidence: &ValidatedGenesisEvidenceCheckpointV1,
    ) -> Result<ValidatedPopulationSnapshotV1, PopulationSnapshotErrorV1> {
        let config = self
            .config
            .validate()
            .map_err(PopulationSnapshotErrorV1::Config)?;
        let lifecycle = evidence.lifecycle();

        if self.current_tick != evidence.behavior_next_tick() {
            return Err(PopulationSnapshotErrorV1::BehaviorClockMismatch {
                expected: evidence.behavior_next_tick(),
                observed: self.current_tick,
            });
        }

        let organism_seed_allocator = OrganismSeedAllocatorV1::from_snapshot_for_lifecycle(
            self.organism_seed_allocator,
            lifecycle.ledger(),
        )
        .map_err(PopulationSnapshotErrorV1::OrganismSeedAllocator)?;
        let id_allocator = AgentIdAllocator::from_snapshot_for_lifecycle(
            self.id_allocator,
            lifecycle.ledger(),
        )
        .map_err(PopulationSnapshotErrorV1::IdAllocator)?;
        let evolution_rng = EvolutionRngStreamsV1::from_snapshot(self.evolution_rng)
            .map_err(PopulationSnapshotErrorV1::EvolutionRng)?;

        let expected_births = u64::try_from(lifecycle.birth_count())
            .map_err(|_| PopulationSnapshotErrorV1::CountOverflow { field: "birth_count" })?;
        if self.total_births != expected_births {
            return Err(PopulationSnapshotErrorV1::BirthCountMismatch {
                expected: expected_births,
                observed: self.total_births,
            });
        }
        let expected_deaths = u64::try_from(lifecycle.death_count())
            .map_err(|_| PopulationSnapshotErrorV1::CountOverflow { field: "death_count" })?;
        if self.total_deaths != expected_deaths {
            return Err(PopulationSnapshotErrorV1::DeathCountMismatch {
                expected: expected_deaths,
                observed: self.total_deaths,
            });
        }
        let founders = u64::try_from(lifecycle.founder_count())
            .map_err(|_| PopulationSnapshotErrorV1::CountOverflow { field: "founder_count" })?;
        let expected_population = founders
            .checked_add(expected_births)
            .and_then(|value| value.checked_sub(expected_deaths))
            .ok_or(PopulationSnapshotErrorV1::CountOverflow {
                field: "population_accounting",
            })?;
        let observed_population = u64::try_from(self.organisms.len()).map_err(|_| {
            PopulationSnapshotErrorV1::CountOverflow {
                field: "organisms.len",
            }
        })?;
        if observed_population != expected_population {
            return Err(PopulationSnapshotErrorV1::PopulationAccountingMismatch {
                expected: expected_population,
                observed: observed_population,
            });
        }

        let mut organisms = Vec::with_capacity(self.organisms.len());
        let mut observed_live = BTreeSet::new();
        for (index, raw) in self.organisms.iter().enumerate() {
            let validated = raw.clone().validate().map_err(|error| {
                PopulationSnapshotErrorV1::Organism { index, error }
            })?;
            if !observed_live.insert(validated.id()) {
                return Err(PopulationSnapshotErrorV1::DuplicateLiveAgentId {
                    agent_id: validated.id(),
                });
            }
            organisms.push(validated);
        }

        let expected_live = lifecycle.alive_ids().clone();
        if observed_live != expected_live {
            return Err(PopulationSnapshotErrorV1::LiveSetMismatch {
                expected: expected_live,
                observed: observed_live,
            });
        }

        for organism in &organisms {
            let record = lifecycle
                .ledger()
                .records()
                .get(&organism.id())
                .copied()
                .ok_or(PopulationSnapshotErrorV1::MissingLifecycleRecord {
                    agent_id: organism.id(),
                })?;
            if record.lineage_id != organism.lineage_id()
                || record.generation != organism.generation()
            {
                return Err(PopulationSnapshotErrorV1::LifecycleIdentityMismatch {
                    agent_id: organism.id(),
                });
            }

            // Reconstruct the exact config a living organism should carry from two independent
            // authorities: lifecycle owns the heritable genome; PopulationConfig owns the newborn
            // non-heritable template. This catches both genome drift and per-organism policy drift.
            let expected_config = record.genome.to_genome().apply_to(config.config().organism_cfg);
            let expected_snapshot =
                PopulationOrganismTemplateSnapshotV1::from_config(expected_config);
            let observed_snapshot =
                PopulationOrganismTemplateSnapshotV1::from_config(organism.config());
            if expected_snapshot != observed_snapshot {
                return Err(PopulationSnapshotErrorV1::OrganismPolicyMismatch {
                    agent_id: organism.id(),
                });
            }

            // The explicit genome comparison documents the narrower hereditary theorem even though
            // the full config comparison above also contains these fields.
            if record.genome.to_genome() != Genome::from_config(&organism.config()) {
                return Err(PopulationSnapshotErrorV1::OrganismPolicyMismatch {
                    agent_id: organism.id(),
                });
            }
        }

        Ok(ValidatedPopulationSnapshotV1 {
            config,
            organisms,
            organism_seed_allocator,
            evolution_rng,
            id_allocator,
            total_births: self.total_births,
            total_deaths: self.total_deaths,
            current_tick: self.current_tick,
        })
    }
}

impl ValidatedPopulationSnapshotV1 {
    pub fn config(&self) -> &ValidatedPopulationConfigSnapshotV1 {
        &self.config
    }

    /// Exact live population order; validation never canonicalizes this sequence.
    pub fn organisms(&self) -> &[ValidatedOrganismSnapshotV1] {
        &self.organisms
    }

    pub fn organism_seed_allocator_snapshot(&self) -> OrganismSeedAllocatorSnapshotV1 {
        self.organism_seed_allocator.snapshot()
    }

    pub fn evolution_rng_snapshot(&self) -> EvolutionRngSnapshotV1 {
        self.evolution_rng.snapshot()
    }

    pub fn id_allocator_snapshot(&self) -> AgentIdAllocatorSnapshotV1 {
        self.id_allocator.snapshot()
    }

    pub fn total_births(&self) -> u64 {
        self.total_births
    }

    pub fn total_deaths(&self) -> u64 {
        self.total_deaths
    }

    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GenomeEvidenceV1, GenesisEvidenceCheckpointV1, LifecycleCheckpointV1, LifecycleEventV1,
        LifecycleTransitionV1, OrganismConfig,
    };
    use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig};

    fn raw_organism_snapshot(id: AgentId, config: OrganismConfig) -> OrganismSnapshotV1 {
        let agent = ActiveInferenceAgent::new(ActiveInferenceAgentConfig {
            state_dim: 2,
            obs_dim: 2,
            num_actions: 2,
            action_temperature: config.action_temperature,
            ..Default::default()
        });
        let boundary = serde_json::json!({
            "partition": { "internal_dim": 2, "sensory_dim": 1, "active_dim": 1 },
            "permeability": { "sensory": 0.5, "active": 0.5, "effective": 0.5 },
            "permeability_ema": { "sensory": 0.5, "active": 0.5, "effective": 0.5 },
            "alpha": 0.1,
            "history": vec![0.5_f64; symthaea_fep::MARKOV_BOUNDARY_HISTORY_CAP_V1],
            "history_idx": 0,
            "history_count": 0
        });
        serde_json::from_value(serde_json::json!({
            "id": id,
            "agent": agent.snapshot_v1(),
            "boundary": boundary,
            "energy_bits": 0.8_f64.to_bits(),
            "config": crate::OrganismConfigSnapshotV1::from_config(config),
            "last_resource_observed_bits": 0.5_f64.to_bits(),
            "ledger": [],
            "lineage_id": id,
            "generation": 0
        }))
        .expect("construct valid raw organism fixture through public serde contract")
    }

    fn founder_fixture(
        count: usize,
        seed_base: u64,
    ) -> (
        crate::PopulationConfig,
        Vec<AgentId>,
        ValidatedGenesisEvidenceCheckpointV1,
        OrganismSeedAllocatorSnapshotV1,
        AgentIdAllocatorSnapshotV1,
        EvolutionRngSnapshotV1,
    ) {
        let config = crate::PopulationConfig {
            organism_cfg: OrganismConfig::default(),
            ..Default::default()
        };
        let mut ids = AgentIdAllocator::new();
        let mut seeds = OrganismSeedAllocatorV1::new(seed_base);
        let mut events = Vec::new();
        let mut allocated = Vec::new();
        for sequence in 0..count {
            let id = ids.allocate();
            seeds.allocate().expect("fixture seed");
            allocated.push(id);
            events.push(LifecycleEventV1 {
                sequence: sequence as u64,
                tick: 0,
                transition: LifecycleTransitionV1::Founder {
                    agent_id: id,
                    lineage_id: id,
                    generation: 0,
                    genome: GenomeEvidenceV1::from_genome(Genome::from_config(&config.organism_cfg)),
                    initial_energy_bits: 0.8_f64.to_bits(),
                },
            });
        }
        let lifecycle = LifecycleCheckpointV1::from_complete_prefix(&events, 0)
            .expect("fixture lifecycle")
            .into_checkpoint();
        let evidence = GenesisEvidenceCheckpointV1::new(0, 0, Vec::new(), lifecycle)
            .validate_after(0)
            .expect("fixture cross evidence");
        (
            config,
            allocated,
            evidence,
            seeds.snapshot(),
            ids.snapshot(),
            EvolutionRngStreamsV1::new(seed_base).snapshot(),
        )
    }

    fn snapshot_with_order(
        config: crate::PopulationConfig,
        ids: &[AgentId],
        seed_snapshot: OrganismSeedAllocatorSnapshotV1,
        id_snapshot: AgentIdAllocatorSnapshotV1,
        rng_snapshot: EvolutionRngSnapshotV1,
    ) -> PopulationSnapshotV1 {
        let organisms = ids
            .iter()
            .copied()
            .map(|id| raw_organism_snapshot(id, config.organism_cfg))
            .collect();
        PopulationSnapshotV1::new(
            PopulationConfigSnapshotV1::from_config(config),
            organisms,
            seed_snapshot,
            rng_snapshot,
            id_snapshot,
            0,
            0,
            0,
        )
    }

    #[test]
    fn validation_preserves_causal_population_order_instead_of_sorting_ids() {
        let (config, ids, evidence, seeds, id_allocator, rng) = founder_fixture(3, 17);
        let reversed = [ids[2], ids[1], ids[0]];
        let snapshot = snapshot_with_order(config, &reversed, seeds, id_allocator, rng);
        let validated = snapshot
            .validate_for_genesis_social(&evidence)
            .expect("reordered live vector is valid causal state");
        let observed = validated
            .organisms()
            .iter()
            .map(ValidatedOrganismSnapshotV1::id)
            .collect::<Vec<_>>();
        assert_eq!(observed, reversed);
    }

    #[test]
    fn lifecycle_authorizes_both_allocator_cursors() {
        let (config, ids, evidence, seeds, _, rng) = founder_fixture(2, 23);
        let forged_id: AgentIdAllocatorSnapshotV1 =
            serde_json::from_str("{\"next\":1}").expect("forged raw id cursor");
        let snapshot = snapshot_with_order(config, &ids, seeds, forged_id, rng);
        assert!(matches!(
            snapshot.validate_for_genesis_social(&evidence),
            Err(PopulationSnapshotErrorV1::IdAllocator(
                AgentIdAllocatorRestoreErrorV1::NextIdentityMismatch { .. }
            ))
        ));
    }

    #[test]
    fn population_counters_are_bound_to_lifecycle_not_self_reported() {
        let (config, ids, evidence, seeds, id_allocator, rng) = founder_fixture(1, 29);
        let mut snapshot = snapshot_with_order(config, &ids, seeds, id_allocator, rng);
        snapshot.total_births = 1;
        assert_eq!(
            snapshot.validate_for_genesis_social(&evidence).unwrap_err(),
            PopulationSnapshotErrorV1::BirthCountMismatch {
                expected: 0,
                observed: 1,
            }
        );
    }

    #[test]
    fn population_social_clock_must_match_validated_behavior_boundary() {
        let (config, ids, evidence, seeds, id_allocator, rng) = founder_fixture(1, 31);
        let mut snapshot = snapshot_with_order(config, &ids, seeds, id_allocator, rng);
        snapshot.current_tick = 1;
        assert_eq!(
            snapshot.validate_for_genesis_social(&evidence).unwrap_err(),
            PopulationSnapshotErrorV1::BehaviorClockMismatch {
                expected: 0,
                observed: 1,
            }
        );
    }

    #[test]
    fn lifecycle_genome_plus_population_template_bind_full_live_organism_policy() {
        let (config, ids, evidence, seeds, id_allocator, rng) = founder_fixture(1, 37);
        let mut changed = config.organism_cfg;
        changed.metabolic_cost += 0.001; // non-heritable drift must still be caught.
        let snapshot = PopulationSnapshotV1::new(
            PopulationConfigSnapshotV1::from_config(config),
            vec![raw_organism_snapshot(ids[0], changed)],
            seeds,
            rng,
            id_allocator,
            0,
            0,
            0,
        );
        assert_eq!(
            snapshot.validate_for_genesis_social(&evidence).unwrap_err(),
            PopulationSnapshotErrorV1::OrganismPolicyMismatch { agent_id: ids[0] }
        );
    }
}
