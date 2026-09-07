// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! First-class lifecycle evidence contract for Genesis ALife.
//!
//! The ordinary [`crate::GenesisEvent`] stream is behavioral evidence: it records what an
//! organism did on a social tick. It does not, by itself, establish exact founder creation,
//! immediate parentage, genome-source ancestry, birth/death timing, or extinction. This module
//! defines those lifecycle facts separately so later simulation wiring can emit them at the
//! authoritative transition instead of reconstructing them heuristically from missing behavior.
//!
//! No simulation path emits these events yet. This tranche freezes the data contract and a
//! fail-closed validator before hot reproduction/death code is modified.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{AgentId, Genome};

/// Exact bit-preserving snapshot of the currently heritable [`Genome`] surface.
///
/// Storing IEEE-754 bits rather than a text/debug representation makes the evidence identity
/// independent of formatting and preserves values such as signed zero exactly. Validation still
/// rejects non-finite values before the snapshot can participate in a qualified lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenomeEvidenceV1 {
    pub set_point_bits: u64,
    pub forage_efficiency_bits: u64,
    pub action_temperature_bits: u64,
    pub perceptual_grain_bits: Option<u64>,
}

impl GenomeEvidenceV1 {
    pub fn from_genome(genome: Genome) -> Self {
        Self {
            set_point_bits: genome.set_point.to_bits(),
            forage_efficiency_bits: genome.forage_efficiency.to_bits(),
            action_temperature_bits: genome.action_temperature.to_bits(),
            perceptual_grain_bits: genome.perceptual_grain.map(f64::to_bits),
        }
    }

    pub fn to_genome(self) -> Genome {
        Genome {
            set_point: f64::from_bits(self.set_point_bits),
            forage_efficiency: f64::from_bits(self.forage_efficiency_bits),
            action_temperature: f64::from_bits(self.action_temperature_bits),
            perceptual_grain: self.perceptual_grain_bits.map(f64::from_bits),
        }
    }

    fn validate(self) -> Result<(), LifecycleError> {
        validate_bits("set_point", self.set_point_bits)?;
        validate_bits("forage_efficiency", self.forage_efficiency_bits)?;
        validate_bits("action_temperature", self.action_temperature_bits)?;
        if let Some(bits) = self.perceptual_grain_bits {
            validate_bits("perceptual_grain", bits)?;
        }
        Ok(())
    }
}

impl From<Genome> for GenomeEvidenceV1 {
    fn from(value: Genome) -> Self {
        Self::from_genome(value)
    }
}

/// Death causes already represented by authoritative `Population` transitions.
///
/// New causes should be added only when a real owner transition exists. A generic free-form
/// "other" variant is intentionally absent because it would erase causal precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LifecycleDeathCauseV1 {
    PopulationEnergyThreshold,
    CullWeakest,
}

/// One authoritative lifecycle transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LifecycleTransitionV1 {
    /// Initial population member. Founders are their own lineage root at generation zero.
    Founder {
        agent_id: AgentId,
        lineage_id: AgentId,
        generation: u32,
        genome: GenomeEvidenceV1,
        initial_energy_bits: u64,
    },
    /// One reproduction event.
    ///
    /// `reproductive_parent_id` is the organism that physically reproduced and therefore owns
    /// lineage/generation ancestry. `genome_source_id` is the organism whose genome was copied
    /// before mutation. They differ under `InheritanceMode::RandomPeer`, so collapsing them would
    /// make the control condition scientifically ambiguous.
    Birth {
        reproductive_parent_id: AgentId,
        genome_source_id: AgentId,
        offspring_id: AgentId,
        lineage_id: AgentId,
        generation: u32,
        reproductive_parent_genome: GenomeEvidenceV1,
        genome_source_genome: GenomeEvidenceV1,
        offspring_genome: GenomeEvidenceV1,
        initial_energy_bits: u64,
    },
    /// Exact removal of a living organism by an authoritative population transition.
    Death {
        agent_id: AgentId,
        cause: LifecycleDeathCauseV1,
        energy_bits: u64,
    },
}

/// Append-only lifecycle record. `sequence` is global within one population lifecycle and must
/// begin at zero and increase by exactly one; `tick` must never decrease.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleEventV1 {
    pub sequence: u64,
    pub tick: u64,
    pub transition: LifecycleTransitionV1,
}

/// Validated lifecycle state for one agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgentLifecycleRecordV1 {
    pub agent_id: AgentId,
    pub lineage_id: AgentId,
    pub generation: u32,
    pub born_tick: u64,
    pub reproductive_parent_id: Option<AgentId>,
    pub genome_source_id: Option<AgentId>,
    pub genome: GenomeEvidenceV1,
    pub initial_energy_bits: u64,
    pub died_tick: Option<u64>,
    pub death_cause: Option<LifecycleDeathCauseV1>,
}

impl AgentLifecycleRecordV1 {
    pub fn is_alive(self) -> bool {
        self.died_tick.is_none()
    }
}

/// Exact point at which the validated living set became empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtinctionEvidenceV1 {
    pub tick: u64,
    pub sequence: u64,
}

/// Deterministically reconstructed lifecycle and two distinct ancestry DAGs.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LifecycleLedgerV1 {
    records: BTreeMap<AgentId, AgentLifecycleRecordV1>,
    alive: BTreeSet<AgentId>,
    reproductive_edges: BTreeSet<(AgentId, AgentId)>,
    genetic_edges: BTreeSet<(AgentId, AgentId)>,
    founder_count: usize,
    birth_count: usize,
    death_count: usize,
    extinction: Option<ExtinctionEvidenceV1>,
}

impl LifecycleLedgerV1 {
    pub fn records(&self) -> &BTreeMap<AgentId, AgentLifecycleRecordV1> {
        &self.records
    }

    pub fn alive(&self) -> &BTreeSet<AgentId> {
        &self.alive
    }

    pub fn reproductive_edges(&self) -> &BTreeSet<(AgentId, AgentId)> {
        &self.reproductive_edges
    }

    pub fn genetic_edges(&self) -> &BTreeSet<(AgentId, AgentId)> {
        &self.genetic_edges
    }

    pub fn founder_count(&self) -> usize {
        self.founder_count
    }

    pub fn birth_count(&self) -> usize {
        self.birth_count
    }

    pub fn death_count(&self) -> usize {
        self.death_count
    }

    pub fn extinction(&self) -> Option<ExtinctionEvidenceV1> {
        self.extinction
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LifecycleError {
    NonContiguousSequence { expected: u64, observed: u64 },
    NonMonotonicTick { previous_tick: u64, next_tick: u64 },
    FounderAfterLifecycleStarted { agent_id: AgentId },
    FounderTickNotZero { agent_id: AgentId, tick: u64 },
    DuplicateAgentId { agent_id: AgentId },
    FounderLineageMismatch { agent_id: AgentId, lineage_id: AgentId },
    FounderGenerationNotZero { agent_id: AgentId, generation: u32 },
    UnknownReproductiveParent { parent_id: AgentId },
    DeadReproductiveParent { parent_id: AgentId },
    UnknownGenomeSource { genome_source_id: AgentId },
    DeadGenomeSource { genome_source_id: AgentId },
    ReproductiveParentGenomeMismatch { parent_id: AgentId },
    GenomeSourceMismatch { genome_source_id: AgentId },
    BirthLineageMismatch {
        parent_id: AgentId,
        expected_lineage_id: AgentId,
        observed_lineage_id: AgentId,
    },
    BirthGenerationMismatch {
        parent_id: AgentId,
        expected_generation: u32,
        observed_generation: u32,
    },
    GenerationOverflow { parent_id: AgentId },
    UnknownDeathAgent { agent_id: AgentId },
    DuplicateDeath { agent_id: AgentId },
    InvalidNumericBits { field: &'static str, bits: u64 },
}

/// Validate and reconstruct a complete lifecycle stream.
///
/// This function treats the stream as complete from population genesis: sequence zero must be
/// present, founder events must occur before any birth/death, and every later sequence must be
/// contiguous. That is what permits an empty validated living set to count as exact extinction
/// rather than merely absence of observations.
pub fn analyze_lifecycle_events(
    events: &[LifecycleEventV1],
) -> Result<LifecycleLedgerV1, LifecycleError> {
    let mut ledger = LifecycleLedgerV1::default();
    let mut expected_sequence = 0u64;
    let mut previous_tick = None;
    let mut founder_phase_open = true;

    for event in events {
        if event.sequence != expected_sequence {
            return Err(LifecycleError::NonContiguousSequence {
                expected: expected_sequence,
                observed: event.sequence,
            });
        }
        expected_sequence = expected_sequence
            .checked_add(1)
            .unwrap_or(u64::MAX);

        if let Some(previous_tick) = previous_tick {
            if event.tick < previous_tick {
                return Err(LifecycleError::NonMonotonicTick {
                    previous_tick,
                    next_tick: event.tick,
                });
            }
        }
        previous_tick = Some(event.tick);

        match event.transition {
            LifecycleTransitionV1::Founder {
                agent_id,
                lineage_id,
                generation,
                genome,
                initial_energy_bits,
            } => {
                if !founder_phase_open {
                    return Err(LifecycleError::FounderAfterLifecycleStarted { agent_id });
                }
                if event.tick != 0 {
                    return Err(LifecycleError::FounderTickNotZero {
                        agent_id,
                        tick: event.tick,
                    });
                }
                if ledger.records.contains_key(&agent_id) {
                    return Err(LifecycleError::DuplicateAgentId { agent_id });
                }
                if lineage_id != agent_id {
                    return Err(LifecycleError::FounderLineageMismatch {
                        agent_id,
                        lineage_id,
                    });
                }
                if generation != 0 {
                    return Err(LifecycleError::FounderGenerationNotZero {
                        agent_id,
                        generation,
                    });
                }
                genome.validate()?;
                validate_bits("initial_energy", initial_energy_bits)?;
                ledger.records.insert(
                    agent_id,
                    AgentLifecycleRecordV1 {
                        agent_id,
                        lineage_id,
                        generation,
                        born_tick: event.tick,
                        reproductive_parent_id: None,
                        genome_source_id: None,
                        genome,
                        initial_energy_bits,
                        died_tick: None,
                        death_cause: None,
                    },
                );
                ledger.alive.insert(agent_id);
                ledger.founder_count += 1;
            }
            LifecycleTransitionV1::Birth {
                reproductive_parent_id,
                genome_source_id,
                offspring_id,
                lineage_id,
                generation,
                reproductive_parent_genome,
                genome_source_genome,
                offspring_genome,
                initial_energy_bits,
            } => {
                founder_phase_open = false;
                if ledger.records.contains_key(&offspring_id) {
                    return Err(LifecycleError::DuplicateAgentId {
                        agent_id: offspring_id,
                    });
                }

                let parent = ledger
                    .records
                    .get(&reproductive_parent_id)
                    .copied()
                    .ok_or(LifecycleError::UnknownReproductiveParent {
                        parent_id: reproductive_parent_id,
                    })?;
                if !parent.is_alive() {
                    return Err(LifecycleError::DeadReproductiveParent {
                        parent_id: reproductive_parent_id,
                    });
                }
                if parent.genome != reproductive_parent_genome {
                    return Err(LifecycleError::ReproductiveParentGenomeMismatch {
                        parent_id: reproductive_parent_id,
                    });
                }

                let genome_source = ledger
                    .records
                    .get(&genome_source_id)
                    .copied()
                    .ok_or(LifecycleError::UnknownGenomeSource { genome_source_id })?;
                if !genome_source.is_alive() {
                    return Err(LifecycleError::DeadGenomeSource { genome_source_id });
                }
                if genome_source.genome != genome_source_genome {
                    return Err(LifecycleError::GenomeSourceMismatch { genome_source_id });
                }

                let expected_generation = parent
                    .generation
                    .checked_add(1)
                    .ok_or(LifecycleError::GenerationOverflow {
                        parent_id: reproductive_parent_id,
                    })?;
                if lineage_id != parent.lineage_id {
                    return Err(LifecycleError::BirthLineageMismatch {
                        parent_id: reproductive_parent_id,
                        expected_lineage_id: parent.lineage_id,
                        observed_lineage_id: lineage_id,
                    });
                }
                if generation != expected_generation {
                    return Err(LifecycleError::BirthGenerationMismatch {
                        parent_id: reproductive_parent_id,
                        expected_generation,
                        observed_generation: generation,
                    });
                }

                reproductive_parent_genome.validate()?;
                genome_source_genome.validate()?;
                offspring_genome.validate()?;
                validate_bits("initial_energy", initial_energy_bits)?;

                ledger.records.insert(
                    offspring_id,
                    AgentLifecycleRecordV1 {
                        agent_id: offspring_id,
                        lineage_id,
                        generation,
                        born_tick: event.tick,
                        reproductive_parent_id: Some(reproductive_parent_id),
                        genome_source_id: Some(genome_source_id),
                        genome: offspring_genome,
                        initial_energy_bits,
                        died_tick: None,
                        death_cause: None,
                    },
                );
                ledger.alive.insert(offspring_id);
                ledger
                    .reproductive_edges
                    .insert((reproductive_parent_id, offspring_id));
                ledger.genetic_edges.insert((genome_source_id, offspring_id));
                ledger.birth_count += 1;
            }
            LifecycleTransitionV1::Death {
                agent_id,
                cause,
                energy_bits,
            } => {
                founder_phase_open = false;
                validate_bits("death_energy", energy_bits)?;
                let record = ledger
                    .records
                    .get_mut(&agent_id)
                    .ok_or(LifecycleError::UnknownDeathAgent { agent_id })?;
                if !record.is_alive() {
                    return Err(LifecycleError::DuplicateDeath { agent_id });
                }
                record.died_tick = Some(event.tick);
                record.death_cause = Some(cause);
                ledger.alive.remove(&agent_id);
                ledger.death_count += 1;
                if ledger.alive.is_empty() && ledger.founder_count > 0 && ledger.extinction.is_none() {
                    ledger.extinction = Some(ExtinctionEvidenceV1 {
                        tick: event.tick,
                        sequence: event.sequence,
                    });
                }
            }
        }
    }

    Ok(ledger)
}

fn validate_bits(field: &'static str, bits: u64) -> Result<(), LifecycleError> {
    if f64::from_bits(bits).is_finite() {
        Ok(())
    } else {
        Err(LifecycleError::InvalidNumericBits { field, bits })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AgentIdAllocator, OrganismConfig};

    fn genome() -> GenomeEvidenceV1 {
        GenomeEvidenceV1::from_genome(Genome::from_config(&OrganismConfig::default()))
    }

    fn founder(sequence: u64, id: AgentId) -> LifecycleEventV1 {
        LifecycleEventV1 {
            sequence,
            tick: 0,
            transition: LifecycleTransitionV1::Founder {
                agent_id: id,
                lineage_id: id,
                generation: 0,
                genome: genome(),
                initial_energy_bits: 0.5f64.to_bits(),
            },
        }
    }

    #[test]
    fn reconstructs_distinct_reproductive_and_genetic_ancestry() {
        let mut ids = AgentIdAllocator::new();
        let parent = ids.allocate();
        let donor = ids.allocate();
        let child = ids.allocate();
        let child_genome = GenomeEvidenceV1::from_genome(Genome {
            forage_efficiency: 0.25,
            ..genome().to_genome()
        });
        let events = vec![
            founder(0, parent),
            founder(1, donor),
            LifecycleEventV1 {
                sequence: 2,
                tick: 4,
                transition: LifecycleTransitionV1::Birth {
                    reproductive_parent_id: parent,
                    genome_source_id: donor,
                    offspring_id: child,
                    lineage_id: parent,
                    generation: 1,
                    reproductive_parent_genome: genome(),
                    genome_source_genome: genome(),
                    offspring_genome: child_genome,
                    initial_energy_bits: 0.4f64.to_bits(),
                },
            },
        ];

        let ledger = analyze_lifecycle_events(&events).expect("valid lifecycle");
        assert!(ledger.reproductive_edges().contains(&(parent, child)));
        assert!(ledger.genetic_edges().contains(&(donor, child)));
        let child_record = ledger.records().get(&child).expect("child record");
        assert_eq!(child_record.lineage_id, parent);
        assert_eq!(child_record.genome_source_id, Some(donor));
        assert_eq!(child_record.genome, child_genome);
    }

    #[test]
    fn exact_extinction_requires_the_last_validated_death() {
        let mut ids = AgentIdAllocator::new();
        let a = ids.allocate();
        let b = ids.allocate();
        let events = vec![
            founder(0, a),
            founder(1, b),
            LifecycleEventV1 {
                sequence: 2,
                tick: 10,
                transition: LifecycleTransitionV1::Death {
                    agent_id: a,
                    cause: LifecycleDeathCauseV1::PopulationEnergyThreshold,
                    energy_bits: 0.0f64.to_bits(),
                },
            },
            LifecycleEventV1 {
                sequence: 3,
                tick: 12,
                transition: LifecycleTransitionV1::Death {
                    agent_id: b,
                    cause: LifecycleDeathCauseV1::PopulationEnergyThreshold,
                    energy_bits: 0.0f64.to_bits(),
                },
            },
        ];
        let ledger = analyze_lifecycle_events(&events).expect("valid lifecycle");
        assert!(ledger.alive().is_empty());
        assert_eq!(
            ledger.extinction(),
            Some(ExtinctionEvidenceV1 {
                tick: 12,
                sequence: 3,
            })
        );
    }

    #[test]
    fn rejects_missing_sequence_instead_of_treating_partial_history_as_complete() {
        let mut ids = AgentIdAllocator::new();
        let a = ids.allocate();
        let events = vec![founder(1, a)];
        assert_eq!(
            analyze_lifecycle_events(&events),
            Err(LifecycleError::NonContiguousSequence {
                expected: 0,
                observed: 1,
            })
        );
    }

    #[test]
    fn rejects_birth_with_wrong_lineage_or_generation() {
        let mut ids = AgentIdAllocator::new();
        let parent = ids.allocate();
        let wrong_lineage = ids.allocate();
        let child = ids.allocate();
        let events = vec![
            founder(0, parent),
            founder(1, wrong_lineage),
            LifecycleEventV1 {
                sequence: 2,
                tick: 1,
                transition: LifecycleTransitionV1::Birth {
                    reproductive_parent_id: parent,
                    genome_source_id: parent,
                    offspring_id: child,
                    lineage_id: wrong_lineage,
                    generation: 7,
                    reproductive_parent_genome: genome(),
                    genome_source_genome: genome(),
                    offspring_genome: genome(),
                    initial_energy_bits: 0.4f64.to_bits(),
                },
            },
        ];
        assert!(matches!(
            analyze_lifecycle_events(&events),
            Err(LifecycleError::BirthLineageMismatch { parent_id, .. }) if parent_id == parent
        ));
    }

    #[test]
    fn rejects_birth_from_a_dead_parent() {
        let mut ids = AgentIdAllocator::new();
        let parent = ids.allocate();
        let child = ids.allocate();
        let events = vec![
            founder(0, parent),
            LifecycleEventV1 {
                sequence: 1,
                tick: 1,
                transition: LifecycleTransitionV1::Death {
                    agent_id: parent,
                    cause: LifecycleDeathCauseV1::CullWeakest,
                    energy_bits: 0.2f64.to_bits(),
                },
            },
            LifecycleEventV1 {
                sequence: 2,
                tick: 2,
                transition: LifecycleTransitionV1::Birth {
                    reproductive_parent_id: parent,
                    genome_source_id: parent,
                    offspring_id: child,
                    lineage_id: parent,
                    generation: 1,
                    reproductive_parent_genome: genome(),
                    genome_source_genome: genome(),
                    offspring_genome: genome(),
                    initial_energy_bits: 0.4f64.to_bits(),
                },
            },
        ];
        assert_eq!(
            analyze_lifecycle_events(&events),
            Err(LifecycleError::DeadReproductiveParent { parent_id: parent })
        );
    }

    #[test]
    fn lifecycle_event_round_trips_through_json_without_float_text_identity() {
        let mut ids = AgentIdAllocator::new();
        let event = founder(0, ids.allocate());
        let encoded = serde_json::to_string(&event).expect("serialize lifecycle event");
        let decoded: LifecycleEventV1 =
            serde_json::from_str(&encoded).expect("deserialize lifecycle event");
        assert_eq!(decoded, event);
    }
}
