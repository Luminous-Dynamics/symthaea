// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lossless continuation checkpoints for chunked first-class lifecycle evidence.
//!
//! [`crate::analyze_lifecycle_events`] deliberately accepts only a complete lifecycle beginning at
//! sequence zero. Long-running populations, however, need to persist and drain event chunks. A
//! checkpoint therefore normalizes an already validated prefix into one record per organism while
//! preserving every field required to reconstruct the exact typed founder/birth/death stream.
//!
//! Persisted [`LifecycleCheckpointV1`] values are intentionally *not* authority-bearing merely
//! because they deserialize. Scientific queries and suffix continuation are available only through
//! [`ValidatedLifecycleCheckpointV1`], a non-serializable capability produced by reconstructing the
//! complete prefix and re-running the canonical lifecycle validator.
//!
//! This v1 prioritizes semantic identity over asymptotic validation cost. A future optimized
//! continuation validator must prove equivalence to this reference behavior before replacing it.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    AgentId, ExtinctionEvidenceV1, GenomeEvidenceV1, LifecycleDeathCauseV1, LifecycleError,
    LifecycleEventV1, LifecycleLedgerV1, LifecycleTransitionV1, analyze_lifecycle_events,
};

/// Lossless normalized lifecycle state for one previously observed organism.
///
/// This type stays private: callers query canonical validated ledger/event views rather than raw
/// normalized persistence records. That prevents a separately deserialized agent-shaped value from
/// being confused with qualified lifecycle evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct LifecycleCheckpointAgentV1 {
    lineage_id: AgentId,
    generation: u32,
    born_tick: u64,
    born_sequence: u64,
    reproductive_parent_id: Option<AgentId>,
    genome_source_id: Option<AgentId>,
    reproductive_parent_genome: Option<GenomeEvidenceV1>,
    genome_source_genome: Option<GenomeEvidenceV1>,
    genome: GenomeEvidenceV1,
    initial_energy_bits: u64,
    died_tick: Option<u64>,
    death_sequence: Option<u64>,
    death_cause: Option<LifecycleDeathCauseV1>,
    death_energy_bits: Option<u64>,
}

/// Persistable, lossless normalized lifecycle prefix.
///
/// All raw fields are private. Deserialization yields only this opaque persistence object; callers
/// must consume it with [`Self::validate`] before reading scientific state or extending the stream.
///
/// `continuation_epoch` deserves a precise claim boundary: canonical lifecycle events prove that it
/// is not *earlier* than the final observed event, but events cannot independently prove how many
/// transition-free population steps occurred afterward. A checkpoint created directly from a
/// recorder can therefore preserve that boundary exactly in-process, while a persisted copy needs
/// a future recorder-issued boundary receipt / authenticated evidence chain to establish the epoch
/// itself against malicious replacement. This module claims semantic consistency, not origin
/// authentication.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleCheckpointV1 {
    next_sequence: u64,
    continuation_epoch: u64,
    agents: BTreeMap<AgentId, LifecycleCheckpointAgentV1>,
}

/// Authority-bearing capability obtained only after canonical checkpoint validation.
///
/// This wrapper deliberately does not implement `Serialize` or `Deserialize`. Persist the opaque
/// [`LifecycleCheckpointV1`] via [`Self::as_checkpoint`] / [`Self::into_checkpoint`], then validate
/// it again after loading before using any scientific query or continuation operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedLifecycleCheckpointV1 {
    checkpoint: LifecycleCheckpointV1,
    ledger: LifecycleLedgerV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LifecycleCheckpointErrorV1 {
    Prefix(LifecycleError),
    Integrity(LifecycleError),
    PrefixEpochBeforeLastEvent {
        continuation_epoch: u64,
        last_event_tick: u64,
    },
    EpochMovedBackward {
        checkpoint_epoch: u64,
        resulting_epoch: u64,
    },
    EventBeforeCheckpointEpoch {
        checkpoint_epoch: u64,
        event_tick: u64,
    },
    EventAfterResultingEpoch {
        event_tick: u64,
        resulting_epoch: u64,
    },
    SequenceOverflow {
        sequence: u64,
    },
    NextSequenceMismatch {
        expected: u64,
        observed: u64,
    },
    DuplicateNormalizedAgent {
        agent_id: AgentId,
    },
    MissingNormalizedAgent {
        agent_id: AgentId,
    },
    IncompleteBirthRecord {
        agent_id: AgentId,
    },
    IncompleteDeathRecord {
        agent_id: AgentId,
    },
}

impl LifecycleCheckpointV1 {
    /// Normalize a complete, canonically valid lifecycle prefix and return a validated capability.
    ///
    /// `continuation_epoch` should be read from the recorder that emitted `events`. It may exceed
    /// the final event tick when population steps produced no lifecycle transition. The lifecycle
    /// prefix can verify only the lower-bound consistency of that metadata; see the type-level
    /// claim boundary above.
    pub fn from_complete_prefix(
        events: &[LifecycleEventV1],
        continuation_epoch: u64,
    ) -> Result<ValidatedLifecycleCheckpointV1, LifecycleCheckpointErrorV1> {
        analyze_lifecycle_events(events).map_err(LifecycleCheckpointErrorV1::Prefix)?;
        validate_boundary_epoch(events, continuation_epoch)?;

        let next_sequence = next_sequence_after(events)?;
        let mut agents = BTreeMap::<AgentId, LifecycleCheckpointAgentV1>::new();

        for event in events {
            match event.transition {
                LifecycleTransitionV1::Founder {
                    agent_id,
                    lineage_id,
                    generation,
                    genome,
                    initial_energy_bits,
                } => {
                    let previous = agents.insert(
                        agent_id,
                        LifecycleCheckpointAgentV1 {
                            lineage_id,
                            generation,
                            born_tick: event.tick,
                            born_sequence: event.sequence,
                            reproductive_parent_id: None,
                            genome_source_id: None,
                            reproductive_parent_genome: None,
                            genome_source_genome: None,
                            genome,
                            initial_energy_bits,
                            died_tick: None,
                            death_sequence: None,
                            death_cause: None,
                            death_energy_bits: None,
                        },
                    );
                    if previous.is_some() {
                        return Err(LifecycleCheckpointErrorV1::DuplicateNormalizedAgent {
                            agent_id,
                        });
                    }
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
                    let previous = agents.insert(
                        offspring_id,
                        LifecycleCheckpointAgentV1 {
                            lineage_id,
                            generation,
                            born_tick: event.tick,
                            born_sequence: event.sequence,
                            reproductive_parent_id: Some(reproductive_parent_id),
                            genome_source_id: Some(genome_source_id),
                            reproductive_parent_genome: Some(reproductive_parent_genome),
                            genome_source_genome: Some(genome_source_genome),
                            genome: offspring_genome,
                            initial_energy_bits,
                            died_tick: None,
                            death_sequence: None,
                            death_cause: None,
                            death_energy_bits: None,
                        },
                    );
                    if previous.is_some() {
                        return Err(LifecycleCheckpointErrorV1::DuplicateNormalizedAgent {
                            agent_id: offspring_id,
                        });
                    }
                }
                LifecycleTransitionV1::Death {
                    agent_id,
                    cause,
                    energy_bits,
                } => {
                    let record = agents.get_mut(&agent_id).ok_or(
                        LifecycleCheckpointErrorV1::MissingNormalizedAgent { agent_id },
                    )?;
                    record.died_tick = Some(event.tick);
                    record.death_sequence = Some(event.sequence);
                    record.death_cause = Some(cause);
                    record.death_energy_bits = Some(energy_bits);
                }
            }
        }

        LifecycleCheckpointV1 {
            next_sequence,
            continuation_epoch,
            agents,
        }
        .validate()
    }

    /// Consume an opaque persisted checkpoint and produce an authority-bearing capability only if
    /// its reconstructed prefix satisfies the complete canonical lifecycle validator.
    pub fn validate(
        self,
    ) -> Result<ValidatedLifecycleCheckpointV1, LifecycleCheckpointErrorV1> {
        let events = self.reconstruct_complete_prefix_unvalidated()?;
        let ledger = analyze_lifecycle_events(&events).map_err(LifecycleCheckpointErrorV1::Integrity)?;
        validate_boundary_epoch(&events, self.continuation_epoch)?;
        let expected = next_sequence_after(&events)?;
        if self.next_sequence != expected {
            return Err(LifecycleCheckpointErrorV1::NextSequenceMismatch {
                expected,
                observed: self.next_sequence,
            });
        }
        Ok(ValidatedLifecycleCheckpointV1 {
            checkpoint: self,
            ledger,
        })
    }

    /// Integrity probe for storage/ingestion code that does not yet need the validated capability.
    /// Scientific queries still remain unavailable on this raw type.
    pub fn validate_integrity(&self) -> Result<(), LifecycleCheckpointErrorV1> {
        self.clone().validate().map(|_| ())
    }

    fn reconstruct_complete_prefix_unvalidated(
        &self,
    ) -> Result<Vec<LifecycleEventV1>, LifecycleCheckpointErrorV1> {
        let mut events = Vec::with_capacity(
            self.agents.len()
                + self
                    .agents
                    .values()
                    .filter(|record| record.death_sequence.is_some())
                    .count(),
        );

        for (&agent_id, record) in &self.agents {
            let birth_transition = match (
                record.reproductive_parent_id,
                record.genome_source_id,
                record.reproductive_parent_genome,
                record.genome_source_genome,
            ) {
                (None, None, None, None) => LifecycleTransitionV1::Founder {
                    agent_id,
                    lineage_id: record.lineage_id,
                    generation: record.generation,
                    genome: record.genome,
                    initial_energy_bits: record.initial_energy_bits,
                },
                (
                    Some(reproductive_parent_id),
                    Some(genome_source_id),
                    Some(reproductive_parent_genome),
                    Some(genome_source_genome),
                ) => LifecycleTransitionV1::Birth {
                    reproductive_parent_id,
                    genome_source_id,
                    offspring_id: agent_id,
                    lineage_id: record.lineage_id,
                    generation: record.generation,
                    reproductive_parent_genome,
                    genome_source_genome,
                    offspring_genome: record.genome,
                    initial_energy_bits: record.initial_energy_bits,
                },
                _ => {
                    return Err(LifecycleCheckpointErrorV1::IncompleteBirthRecord {
                        agent_id,
                    });
                }
            };
            events.push(LifecycleEventV1 {
                sequence: record.born_sequence,
                tick: record.born_tick,
                transition: birth_transition,
            });

            match (
                record.died_tick,
                record.death_sequence,
                record.death_cause,
                record.death_energy_bits,
            ) {
                (None, None, None, None) => {}
                (Some(tick), Some(sequence), Some(cause), Some(energy_bits)) => {
                    events.push(LifecycleEventV1 {
                        sequence,
                        tick,
                        transition: LifecycleTransitionV1::Death {
                            agent_id,
                            cause,
                            energy_bits,
                        },
                    });
                }
                _ => {
                    return Err(LifecycleCheckpointErrorV1::IncompleteDeathRecord {
                        agent_id,
                    });
                }
            }
        }

        events.sort_by_key(|event| event.sequence);
        Ok(events)
    }
}

impl ValidatedLifecycleCheckpointV1 {
    /// Opaque serializable checkpoint to persist. Revalidate it after loading before use.
    pub fn as_checkpoint(&self) -> &LifecycleCheckpointV1 {
        &self.checkpoint
    }

    /// Consume the validated capability and return its opaque persistence representation.
    pub fn into_checkpoint(self) -> LifecycleCheckpointV1 {
        self.checkpoint
    }

    /// Exact typed lifecycle prefix represented by the validated checkpoint.
    pub fn reconstruct_complete_prefix(
        &self,
    ) -> Result<Vec<LifecycleEventV1>, LifecycleCheckpointErrorV1> {
        self.checkpoint.reconstruct_complete_prefix_unvalidated()
    }

    /// Validate a drained suffix and return the validated capability for the resulting boundary.
    ///
    /// `resulting_epoch` is continuation metadata supplied by the recorder/runtime boundary. The
    /// complete lifecycle stream proves event ordering and that no event exceeds this boundary; it
    /// does not independently prove silent transition-free step count.
    pub fn validate_chunk(
        &self,
        chunk: &[LifecycleEventV1],
        resulting_epoch: u64,
    ) -> Result<Self, LifecycleCheckpointErrorV1> {
        let checkpoint_epoch = self.checkpoint.continuation_epoch;
        if resulting_epoch < checkpoint_epoch {
            return Err(LifecycleCheckpointErrorV1::EpochMovedBackward {
                checkpoint_epoch,
                resulting_epoch,
            });
        }

        let mut previous_tick = checkpoint_epoch;
        for event in chunk {
            if event.tick < checkpoint_epoch {
                return Err(LifecycleCheckpointErrorV1::EventBeforeCheckpointEpoch {
                    checkpoint_epoch,
                    event_tick: event.tick,
                });
            }
            if event.tick < previous_tick {
                return Err(LifecycleCheckpointErrorV1::Integrity(
                    LifecycleError::NonMonotonicTick {
                        previous_tick,
                        next_tick: event.tick,
                    },
                ));
            }
            if event.tick > resulting_epoch {
                return Err(LifecycleCheckpointErrorV1::EventAfterResultingEpoch {
                    event_tick: event.tick,
                    resulting_epoch,
                });
            }
            previous_tick = event.tick;
        }

        let mut complete = self.reconstruct_complete_prefix()?;
        complete.extend_from_slice(chunk);
        LifecycleCheckpointV1::from_complete_prefix(&complete, resulting_epoch)
    }

    /// Canonically validated lifecycle state available to downstream evolutionary-history and
    /// rescue-attribution analysis.
    pub fn ledger(&self) -> &LifecycleLedgerV1 {
        &self.ledger
    }

    pub fn next_sequence(&self) -> u64 {
        self.checkpoint.next_sequence
    }

    /// Declared continuation boundary preserved from the recorder/runtime. This value is checked
    /// for consistency with the validated prefix but is not independently derivable from silent
    /// transition-free steps in that prefix.
    pub fn continuation_epoch(&self) -> u64 {
        self.checkpoint.continuation_epoch
    }

    pub fn founder_count(&self) -> usize {
        self.ledger.founder_count()
    }

    pub fn birth_count(&self) -> usize {
        self.ledger.birth_count()
    }

    pub fn death_count(&self) -> usize {
        self.ledger.death_count()
    }

    pub fn alive_ids(&self) -> &BTreeSet<AgentId> {
        self.ledger.alive()
    }

    pub fn extinction(&self) -> Option<ExtinctionEvidenceV1> {
        self.ledger.extinction()
    }
}

fn validate_boundary_epoch(
    events: &[LifecycleEventV1],
    continuation_epoch: u64,
) -> Result<(), LifecycleCheckpointErrorV1> {
    if let Some(last) = events.last() {
        if continuation_epoch < last.tick {
            return Err(LifecycleCheckpointErrorV1::PrefixEpochBeforeLastEvent {
                continuation_epoch,
                last_event_tick: last.tick,
            });
        }
    }
    Ok(())
}

fn next_sequence_after(
    events: &[LifecycleEventV1],
) -> Result<u64, LifecycleCheckpointErrorV1> {
    match events.last() {
        Some(last) => last
            .sequence
            .checked_add(1)
            .ok_or(LifecycleCheckpointErrorV1::SequenceOverflow {
                sequence: last.sequence,
            }),
        None => Ok(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        InheritanceMode, OrganismConfig, Population, PopulationConfig,
        analyze_evolutionary_history, analyze_lifecycle_events,
    };

    fn quiet_cfg() -> PopulationConfig {
        PopulationConfig {
            death_energy_threshold: -1.0,
            reproduction_energy_threshold: 2.0,
            reproduction_energy_cost: 0.4,
            organism_cfg: OrganismConfig::default(),
            ..Default::default()
        }
    }

    fn birth_cfg() -> PopulationConfig {
        PopulationConfig {
            death_energy_threshold: -1.0,
            reproduction_energy_threshold: 0.0,
            reproduction_energy_cost: 0.4,
            mutation_rate: 1.0,
            mutation_std: 0.05,
            inheritance: InheritanceMode::RandomPeer,
            organism_cfg: OrganismConfig::default(),
        }
    }

    #[test]
    fn normalization_reconstructs_the_exact_typed_prefix() {
        let mut pop = Population::new(birth_cfg(), 3, 17);
        pop.step(|_| 1.0);
        pop.cull_weakest(1);
        let prefix = pop.drain_lifecycle_events();
        let checkpoint =
            LifecycleCheckpointV1::from_complete_prefix(&prefix, pop.lifecycle_epoch())
                .expect("valid prefix");
        assert_eq!(
            checkpoint.reconstruct_complete_prefix().expect("reconstruct"),
            prefix
        );
    }

    #[test]
    fn validates_a_drained_suffix_equivalently_to_the_complete_stream() {
        let mut pop = Population::new(birth_cfg(), 3, 23);
        let prefix = pop.drain_lifecycle_events();
        let checkpoint =
            LifecycleCheckpointV1::from_complete_prefix(&prefix, pop.lifecycle_epoch())
                .expect("valid prefix");

        pop.step(|_| 1.0);
        let chunk = pop.drain_lifecycle_events();
        let continued = checkpoint
            .validate_chunk(&chunk, pop.lifecycle_epoch())
            .expect("valid continuation");

        let mut complete = prefix;
        complete.extend_from_slice(&chunk);
        let full = analyze_lifecycle_events(&complete).expect("full validation");

        assert_eq!(continued.ledger().records(), full.records());
        assert_eq!(continued.ledger().alive(), full.alive());
        assert_eq!(continued.ledger().reproductive_edges(), full.reproductive_edges());
        assert_eq!(continued.ledger().genetic_edges(), full.genetic_edges());
        assert_eq!(continued.extinction(), full.extinction());
    }

    #[test]
    fn evolutionary_history_survives_checkpoint_compaction() {
        let mut pop = Population::new(birth_cfg(), 3, 29);
        pop.step(|_| 1.0);
        pop.cull_weakest(2);
        let complete = pop.drain_lifecycle_events();
        let full_ledger = analyze_lifecycle_events(&complete).expect("complete ledger");
        let full_history = analyze_evolutionary_history(&full_ledger).expect("full history");

        let checkpoint =
            LifecycleCheckpointV1::from_complete_prefix(&complete, pop.lifecycle_epoch())
                .expect("checkpoint");
        let compacted_history =
            analyze_evolutionary_history(checkpoint.ledger()).expect("compacted history");

        assert_eq!(compacted_history, full_history);
    }

    #[test]
    fn empty_chunks_can_advance_continuation_epoch_without_inventing_events() {
        let mut pop = Population::new(quiet_cfg(), 2, 31);
        let prefix = pop.drain_lifecycle_events();
        let checkpoint =
            LifecycleCheckpointV1::from_complete_prefix(&prefix, pop.lifecycle_epoch())
                .expect("valid prefix");

        pop.step(|_| 1.0);
        assert!(pop.lifecycle_events().is_empty());
        let advanced = checkpoint
            .validate_chunk(&[], pop.lifecycle_epoch())
            .expect("empty transition interval");
        assert_eq!(advanced.next_sequence(), checkpoint.next_sequence());
        assert_eq!(advanced.continuation_epoch(), 1);
        assert_eq!(advanced.reconstruct_complete_prefix().unwrap(), prefix);
    }

    #[test]
    fn stale_chunk_tick_is_rejected_after_an_empty_epoch_advance() {
        let mut pop = Population::new(quiet_cfg(), 2, 41);
        let prefix = pop.drain_lifecycle_events();
        let checkpoint =
            LifecycleCheckpointV1::from_complete_prefix(&prefix, pop.lifecycle_epoch())
                .expect("prefix");
        let advanced = checkpoint.validate_chunk(&[], 1).expect("advance epoch");

        pop.cull_weakest(1);
        let stale = pop.drain_lifecycle_events();
        assert_eq!(stale[0].tick, 0);
        assert_eq!(
            advanced.validate_chunk(&stale, 1),
            Err(LifecycleCheckpointErrorV1::EventBeforeCheckpointEpoch {
                checkpoint_epoch: 1,
                event_tick: 0,
            })
        );
    }

    #[test]
    fn deserialized_checkpoint_requires_validation_capability() {
        let mut pop = Population::new(birth_cfg(), 3, 53);
        pop.step(|_| 1.0);
        let complete = pop.drain_lifecycle_events();
        let validated =
            LifecycleCheckpointV1::from_complete_prefix(&complete, pop.lifecycle_epoch())
                .expect("checkpoint");

        let encoded = serde_json::to_string(validated.as_checkpoint()).expect("serialize");
        let raw: LifecycleCheckpointV1 = serde_json::from_str(&encoded).expect("deserialize");
        let revalidated = raw.validate().expect("canonical validation capability");
        assert_eq!(revalidated.ledger().records(), validated.ledger().records());
        assert_eq!(revalidated.continuation_epoch(), validated.continuation_epoch());
    }

    #[test]
    fn tampered_serialized_checkpoint_fails_before_capability_creation() {
        let mut pop = Population::new(birth_cfg(), 3, 59);
        pop.step(|_| 1.0);
        let complete = pop.drain_lifecycle_events();
        let validated =
            LifecycleCheckpointV1::from_complete_prefix(&complete, pop.lifecycle_epoch())
                .expect("checkpoint");

        let mut encoded = serde_json::to_value(validated.as_checkpoint()).expect("serialize");
        let agents = encoded
            .get_mut("agents")
            .and_then(serde_json::Value::as_object_mut)
            .expect("agents object");
        let (_, child) = agents
            .iter_mut()
            .find(|(_, value)| !value["reproductive_parent_id"].is_null())
            .expect("birth record");
        let bits = child["genome_source_genome"]["set_point_bits"]
            .as_u64()
            .expect("bits");
        child["genome_source_genome"]["set_point_bits"] = serde_json::json!(bits ^ 1);

        let forged: LifecycleCheckpointV1 =
            serde_json::from_value(encoded).expect("deserialize forged checkpoint");
        assert!(matches!(
            forged.validate(),
            Err(LifecycleCheckpointErrorV1::Integrity(
                LifecycleError::GenomeSourceMismatch { .. }
            ))
        ));
    }

    #[test]
    fn counts_liveness_and_extinction_exist_only_on_validated_capability() {
        let mut cfg = quiet_cfg();
        cfg.death_energy_threshold = 2.0;
        let mut pop = Population::new(cfg, 2, 71);
        pop.step(|_| 0.0);
        let complete = pop.drain_lifecycle_events();
        let checkpoint =
            LifecycleCheckpointV1::from_complete_prefix(&complete, pop.lifecycle_epoch())
                .expect("checkpoint");

        assert_eq!(checkpoint.founder_count(), 2);
        assert_eq!(checkpoint.birth_count(), 0);
        assert_eq!(checkpoint.death_count(), 2);
        assert!(checkpoint.alive_ids().is_empty());
        assert_eq!(checkpoint.extinction(), checkpoint.ledger().extinction());
    }

    #[test]
    fn silent_epoch_metadata_is_consistency_checked_not_event_derived() {
        let mut pop = Population::new(quiet_cfg(), 1, 83);
        let prefix = pop.drain_lifecycle_events();
        pop.step(|_| 1.0);
        pop.step(|_| 1.0);
        assert!(pop.lifecycle_events().is_empty());

        let checkpoint =
            LifecycleCheckpointV1::from_complete_prefix(&prefix, pop.lifecycle_epoch())
                .expect("silent continuation metadata");
        assert_eq!(checkpoint.continuation_epoch(), 2);
        assert_eq!(checkpoint.reconstruct_complete_prefix().unwrap(), prefix);
    }
}
