// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pure matched-randomness provider for future Genesis causal replays.
//!
//! A causal replay should reuse the *stochastic input* observed on the natural trajectory when the
//! same semantic event exists in both worlds. It must not force the same action, partner outcome,
//! mutation phenotype, or survival result, because those downstream differences may be the causal
//! effect under study. When an intervention creates a genuinely novel stochastic event, this layer
//! falls back to [`crate::CounterfactualRandomFieldV1`] so unrelated event-count changes cannot
//! phase-shift later shared randomness.
//!
//! The legacy scheduler needs a special rule because its natural randomness is represented by an
//! observed Fisher-Yates order, not independent per-agent variates. Shared identities keep their
//! exact historical relative order. Novel identities receive stateless fallback priorities and are
//! interleaved against evenly-spaced priority anchors derived from the historical order. With no
//! novel identities, the observed order is reproduced exactly.
//!
//! This module is still contract/runtime-helper only. Production FEP, scheduler, mutation, and
//! Population execution paths do not call it yet.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    AgentId, CounterfactualDrawKeyV1, CounterfactualRandomFieldV1, GenomeTraitV1,
    ObservedSchedulerOrderKindV1, ObservedStochasticKeyV1, ValidatedObservedStochasticTapeV1,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchedRandomOriginV1 {
    ObservedTape,
    CounterfactualFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatchedUnitSampleV1 {
    bits: u64,
    origin: MatchedRandomOriginV1,
}

impl MatchedUnitSampleV1 {
    pub fn value(self) -> f64 {
        f64::from_bits(self.bits)
    }

    pub fn bits(self) -> u64 {
        self.bits
    }

    pub fn origin(self) -> MatchedRandomOriginV1 {
        self.origin
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatchedSeedV1 {
    value: u64,
    origin: MatchedRandomOriginV1,
}

impl MatchedSeedV1 {
    pub fn value(self) -> u64 {
        self.value
    }

    pub fn origin(self) -> MatchedRandomOriginV1 {
        self.origin
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatchedInheritanceSelectionV1 {
    pub source_index: usize,
    pub source_id: AgentId,
    pub origin: MatchedRandomOriginV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchedSchedulerOrderV1 {
    pub order: Vec<AgentId>,
    pub observed_shared_count: usize,
    pub novel_count: usize,
    pub observed_order_available: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchedStochasticReplayErrorV1 {
    EmptyCandidateSet,
    ReservedIdentity,
    DuplicateIdentity,
    FallbackMutationBirthOrdinalUnsupported { birth_ordinal: u64 },
}

/// Pure composition of one validated natural stochastic tape and one stateless fallback field.
pub struct MatchedStochasticReplayV1<'a> {
    tape: &'a ValidatedObservedStochasticTapeV1,
    fallback: CounterfactualRandomFieldV1,
}

impl<'a> MatchedStochasticReplayV1<'a> {
    pub fn new(
        tape: &'a ValidatedObservedStochasticTapeV1,
        fallback: CounterfactualRandomFieldV1,
    ) -> Self {
        Self { tape, fallback }
    }

    pub fn tape(&self) -> &'a ValidatedObservedStochasticTapeV1 {
        self.tape
    }

    pub fn fallback(&self) -> CounterfactualRandomFieldV1 {
        self.fallback
    }

    pub fn agent_action_sample(
        &self,
        tick: u64,
        agent_id: AgentId,
        draw_ordinal: u64,
    ) -> Result<MatchedUnitSampleV1, MatchedStochasticReplayErrorV1> {
        reject_reserved(&[agent_id])?;
        let observed_key =
            ObservedStochasticKeyV1::agent_action_sample(tick, agent_id, draw_ordinal);
        if let Some(value) = self.tape.unit_for(observed_key) {
            return Ok(observed_unit(value));
        }
        Ok(fallback_unit(
            self.fallback.draw_unit_f64(CounterfactualDrawKeyV1::agent_action_sample(
                tick,
                agent_id,
                draw_ordinal,
            )),
        ))
    }

    pub fn mutation_occurrence_sample(
        &self,
        tick: u64,
        reproductive_parent_id: AgentId,
        genome_source_id: AgentId,
        birth_ordinal: u64,
        trait_id: GenomeTraitV1,
    ) -> Result<MatchedUnitSampleV1, MatchedStochasticReplayErrorV1> {
        reject_reserved(&[reproductive_parent_id, genome_source_id])?;
        let observed_key = ObservedStochasticKeyV1::mutation_occurrence(
            tick,
            reproductive_parent_id,
            genome_source_id,
            birth_ordinal,
            trait_id,
        );
        if let Some(value) = self.tape.unit_for(observed_key) {
            return Ok(observed_unit(value));
        }
        require_current_v1_birth_ordinal(birth_ordinal)?;
        Ok(fallback_unit(self.fallback.draw_unit_f64(
            CounterfactualDrawKeyV1::mutation_occurrence(
                tick,
                reproductive_parent_id,
                genome_source_id,
                trait_id,
            ),
        )))
    }

    pub fn mutation_perturbation_sample(
        &self,
        tick: u64,
        reproductive_parent_id: AgentId,
        genome_source_id: AgentId,
        birth_ordinal: u64,
        trait_id: GenomeTraitV1,
    ) -> Result<MatchedUnitSampleV1, MatchedStochasticReplayErrorV1> {
        reject_reserved(&[reproductive_parent_id, genome_source_id])?;
        let observed_key = ObservedStochasticKeyV1::mutation_perturbation(
            tick,
            reproductive_parent_id,
            genome_source_id,
            birth_ordinal,
            trait_id,
        );
        if let Some(value) = self.tape.unit_for(observed_key) {
            return Ok(observed_unit(value));
        }
        require_current_v1_birth_ordinal(birth_ordinal)?;
        Ok(fallback_unit(self.fallback.draw_unit_f64(
            CounterfactualDrawKeyV1::mutation_perturbation(
                tick,
                reproductive_parent_id,
                genome_source_id,
                trait_id,
            ),
        )))
    }

    pub fn offspring_construction_seed(
        &self,
        tick: u64,
        reproductive_parent_id: AgentId,
        birth_ordinal: u64,
    ) -> Result<MatchedSeedV1, MatchedStochasticReplayErrorV1> {
        reject_reserved(&[reproductive_parent_id])?;
        let observed_key = ObservedStochasticKeyV1::offspring_construction_seed(
            tick,
            reproductive_parent_id,
            birth_ordinal,
        );
        if let Some(value) = self.tape.nonzero_u64_for(observed_key) {
            return Ok(MatchedSeedV1 {
                value,
                origin: MatchedRandomOriginV1::ObservedTape,
            });
        }
        Ok(MatchedSeedV1 {
            value: self
                .fallback
                .draw_nonzero_u64(CounterfactualDrawKeyV1::offspring_construction_seed(
                    tick,
                    reproductive_parent_id,
                    birth_ordinal,
                )),
            origin: MatchedRandomOriginV1::CounterfactualFallback,
        })
    }

    /// Select a RandomPeer source using the natural unit sample when it exists.
    ///
    /// Natural replay preserves the legacy population-order sampling rule exactly. A genuinely
    /// novel birth has no historical unit variate; in that case the stateless field assigns each
    /// candidate an independent semantic priority and the minimum wins. The selected identity is
    /// therefore independent of candidate iteration order under fallback.
    pub fn select_inheritance_source(
        &self,
        tick: u64,
        reproductive_parent_id: AgentId,
        birth_ordinal: u64,
        candidate_ids: &[AgentId],
    ) -> Result<MatchedInheritanceSelectionV1, MatchedStochasticReplayErrorV1> {
        validate_identity_slice(candidate_ids, false)?;
        reject_reserved(&[reproductive_parent_id])?;
        let observed_key = ObservedStochasticKeyV1::inheritance_source_sample(
            tick,
            reproductive_parent_id,
            birth_ordinal,
        );
        if let Some(value) = self.tape.unit_for(observed_key) {
            let index = ((value * candidate_ids.len() as f64) as usize).min(candidate_ids.len() - 1);
            return Ok(MatchedInheritanceSelectionV1 {
                source_index: index,
                source_id: candidate_ids[index],
                origin: MatchedRandomOriginV1::ObservedTape,
            });
        }

        let (source_index, source_id, _) = candidate_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(index, candidate_id)| {
                let priority = self.fallback.draw_u64(
                    CounterfactualDrawKeyV1::inheritance_candidate_priority(
                        tick,
                        reproductive_parent_id,
                        candidate_id,
                    ),
                );
                (index, candidate_id, priority)
            })
            .min_by_key(|(_, candidate_id, priority)| (*priority, candidate_id.raw()))
            .expect("non-empty candidate set validated above");

        Ok(MatchedInheritanceSelectionV1 {
            source_index,
            source_id,
            origin: MatchedRandomOriginV1::CounterfactualFallback,
        })
    }

    /// Produce one matched scheduler order for the current cohort.
    ///
    /// If natural order evidence exists, every shared identity keeps its historical relative
    /// position. Novel identities are deterministically interleaved by stateless fallback
    /// priority. If no natural order exists, the whole cohort is ordered by fallback priority.
    pub fn scheduler_order(
        &self,
        tick: u64,
        kind: ObservedSchedulerOrderKindV1,
        current_ids: &[AgentId],
    ) -> Result<MatchedSchedulerOrderV1, MatchedStochasticReplayErrorV1> {
        validate_identity_slice(current_ids, true)?;
        let observed = self.tape.scheduler_order(tick, kind);
        let ranks = observed
            .map(|entry| {
                entry
                    .shuffled_order
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(rank, id)| (id, rank))
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();
        let observed_len = observed.map(|entry| entry.shuffled_order.len()).unwrap_or(0);

        let mut decorated = current_ids
            .iter()
            .copied()
            .map(|id| {
                if let Some(&rank) = ranks.get(&id) {
                    (
                        id,
                        observed_rank_priority(rank, observed_len),
                        0u8,
                        true,
                    )
                } else {
                    (
                        id,
                        self.fallback.draw_u64(CounterfactualDrawKeyV1::scheduler_priority(
                            tick, id,
                        )),
                        1u8,
                        false,
                    )
                }
            })
            .collect::<Vec<_>>();
        decorated.sort_by_key(|(id, priority, class, _)| (*priority, *class, id.raw()));

        let observed_shared_count = decorated.iter().filter(|(_, _, _, shared)| *shared).count();
        let novel_count = decorated.len() - observed_shared_count;
        Ok(MatchedSchedulerOrderV1 {
            order: decorated.into_iter().map(|(id, _, _, _)| id).collect(),
            observed_shared_count,
            novel_count,
            observed_order_available: observed.is_some(),
        })
    }
}

fn observed_unit(value: f64) -> MatchedUnitSampleV1 {
    MatchedUnitSampleV1 {
        bits: value.to_bits(),
        origin: MatchedRandomOriginV1::ObservedTape,
    }
}

fn fallback_unit(value: f64) -> MatchedUnitSampleV1 {
    MatchedUnitSampleV1 {
        bits: value.to_bits(),
        origin: MatchedRandomOriginV1::CounterfactualFallback,
    }
}

fn reject_reserved(ids: &[AgentId]) -> Result<(), MatchedStochasticReplayErrorV1> {
    if ids.iter().any(|&id| id == AgentId::UNALLOCATED) {
        return Err(MatchedStochasticReplayErrorV1::ReservedIdentity);
    }
    Ok(())
}

fn validate_identity_slice(
    ids: &[AgentId],
    allow_empty: bool,
) -> Result<(), MatchedStochasticReplayErrorV1> {
    if ids.is_empty() && !allow_empty {
        return Err(MatchedStochasticReplayErrorV1::EmptyCandidateSet);
    }
    reject_reserved(ids)?;
    let unique = ids.iter().copied().collect::<BTreeSet<_>>();
    if unique.len() != ids.len() {
        return Err(MatchedStochasticReplayErrorV1::DuplicateIdentity);
    }
    Ok(())
}

fn require_current_v1_birth_ordinal(
    birth_ordinal: u64,
) -> Result<(), MatchedStochasticReplayErrorV1> {
    // CounterfactualRandomFieldV1 currently addresses one production birth per parent/tick. The
    // observed tape can represent future multi-birth semantics, but fallback must fail closed until
    // the stateless field gains an explicit birth ordinal for mutation keys.
    if birth_ordinal != 0 {
        return Err(
            MatchedStochasticReplayErrorV1::FallbackMutationBirthOrdinalUnsupported {
                birth_ordinal,
            },
        );
    }
    Ok(())
}

fn observed_rank_priority(rank: usize, len: usize) -> u64 {
    debug_assert!(len > 0);
    debug_assert!(rank < len);
    // Midpoint quantiles: (rank + 0.5) / len mapped over the full 64-bit priority interval. This
    // keeps every observed rank distinct and leaves room for novel stateless priorities to
    // interleave without changing the relative order of observed identities.
    let scale = 1u128 << 64;
    let numerator = (2u128 * rank as u128 + 1) * scale;
    let denominator = 2u128 * len as u128;
    (numerator / denominator) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AgentIdAllocator, ObservedSchedulerOrderV1, ObservedStochasticDrawV1,
        ObservedStochasticTapeSubjectV1, ObservedStochasticTapeV1, ObservedStochasticValueV1,
    };

    fn ids(n: usize) -> Vec<AgentId> {
        let mut allocator = AgentIdAllocator::new();
        (0..n).map(|_| allocator.allocate()).collect()
    }

    fn validated_tape(agents: &[AgentId]) -> crate::ValidatedObservedStochasticTapeV1 {
        let action = ObservedStochasticDrawV1 {
            key: ObservedStochasticKeyV1::agent_action_sample(11, agents[0], 0),
            value: ObservedStochasticValueV1::unit(0.3125),
        };
        let inheritance = ObservedStochasticDrawV1 {
            key: ObservedStochasticKeyV1::inheritance_source_sample(12, agents[0], 0),
            value: ObservedStochasticValueV1::unit(0.74),
        };
        let seed = ObservedStochasticDrawV1 {
            key: ObservedStochasticKeyV1::offspring_construction_seed(12, agents[0], 0),
            value: ObservedStochasticValueV1::nonzero_u64(4242),
        };
        let mut scalar_draws = vec![action, inheritance, seed];
        scalar_draws.sort_by_key(|draw| draw.key);
        ObservedStochasticTapeV1 {
            subject: ObservedStochasticTapeSubjectV1 {
                start_behavior_next_tick: 10,
                start_lifecycle_next_sequence: 4,
                end_behavior_next_tick: 20,
            },
            scalar_draws,
            scheduler_orders: vec![ObservedSchedulerOrderV1 {
                tick: 13,
                kind: ObservedSchedulerOrderKindV1::RandomPopulation,
                input_order: agents[..4].to_vec(),
                shuffled_order: vec![agents[2], agents[0], agents[3], agents[1]],
            }],
        }
        .validate()
        .unwrap()
    }

    #[test]
    fn observed_action_variate_wins_without_forcing_an_action_outcome() {
        let agents = ids(5);
        let tape = validated_tape(&agents);
        let replay = MatchedStochasticReplayV1::new(&tape, CounterfactualRandomFieldV1::new(99));
        let sample = replay.agent_action_sample(11, agents[0], 0).unwrap();
        assert_eq!(sample.value(), 0.3125);
        assert_eq!(sample.origin(), MatchedRandomOriginV1::ObservedTape);
    }

    #[test]
    fn missing_action_variate_uses_stateless_fallback_without_phase_shift() {
        let agents = ids(5);
        let tape = validated_tape(&agents);
        let replay = MatchedStochasticReplayV1::new(&tape, CounterfactualRandomFieldV1::new(99));
        let first = replay.agent_action_sample(15, agents[4], 0).unwrap();
        for ordinal in 0..100 {
            let _ = replay.agent_action_sample(16, agents[3], ordinal).unwrap();
        }
        let again = replay.agent_action_sample(15, agents[4], 0).unwrap();
        assert_eq!(first, again);
        assert_eq!(first.origin(), MatchedRandomOriginV1::CounterfactualFallback);
    }

    #[test]
    fn observed_inheritance_sample_replays_legacy_population_order_selection() {
        let agents = ids(5);
        let tape = validated_tape(&agents);
        let replay = MatchedStochasticReplayV1::new(&tape, CounterfactualRandomFieldV1::new(1));
        let selected = replay
            .select_inheritance_source(12, agents[0], 0, &agents)
            .unwrap();
        assert_eq!(selected.source_index, 3); // floor(0.74 * 5)
        assert_eq!(selected.source_id, agents[3]);
        assert_eq!(selected.origin, MatchedRandomOriginV1::ObservedTape);
    }

    #[test]
    fn fallback_inheritance_selects_same_identity_independent_of_candidate_iteration_order() {
        let agents = ids(5);
        let tape = validated_tape(&agents);
        let replay = MatchedStochasticReplayV1::new(&tape, CounterfactualRandomFieldV1::new(77));
        let a = replay
            .select_inheritance_source(17, agents[0], 0, &agents)
            .unwrap();
        let mut reversed = agents.clone();
        reversed.reverse();
        let b = replay
            .select_inheritance_source(17, agents[0], 0, &reversed)
            .unwrap();
        assert_eq!(a.source_id, b.source_id);
        assert_eq!(a.origin, MatchedRandomOriginV1::CounterfactualFallback);
        assert_eq!(b.origin, MatchedRandomOriginV1::CounterfactualFallback);
    }

    #[test]
    fn exact_scheduler_order_is_reproduced_when_the_historical_cohort_is_present() {
        let agents = ids(5);
        let tape = validated_tape(&agents);
        let replay = MatchedStochasticReplayV1::new(&tape, CounterfactualRandomFieldV1::new(55));
        let order = replay
            .scheduler_order(13, ObservedSchedulerOrderKindV1::RandomPopulation, &agents[..4])
            .unwrap();
        assert_eq!(order.order, vec![agents[2], agents[0], agents[3], agents[1]]);
        assert_eq!(order.observed_shared_count, 4);
        assert_eq!(order.novel_count, 0);
    }

    #[test]
    fn scheduler_subset_preserves_the_exact_relative_order_of_shared_identities() {
        let agents = ids(5);
        let tape = validated_tape(&agents);
        let replay = MatchedStochasticReplayV1::new(&tape, CounterfactualRandomFieldV1::new(55));
        let subset = [agents[2], agents[3], agents[1]];
        let order = replay
            .scheduler_order(13, ObservedSchedulerOrderKindV1::RandomPopulation, &subset)
            .unwrap();
        assert_eq!(order.order, vec![agents[2], agents[3], agents[1]]);
        assert_eq!(order.observed_shared_count, 3);
    }

    #[test]
    fn novel_scheduler_identity_is_deterministically_interleaved_without_reordering_shared_ids() {
        let agents = ids(5);
        let tape = validated_tape(&agents);
        let replay = MatchedStochasticReplayV1::new(&tape, CounterfactualRandomFieldV1::new(1234));
        let current = [agents[0], agents[1], agents[2], agents[3], agents[4]];
        let first = replay
            .scheduler_order(13, ObservedSchedulerOrderKindV1::RandomPopulation, &current)
            .unwrap();
        let mut permuted = current;
        permuted.reverse();
        let second = replay
            .scheduler_order(13, ObservedSchedulerOrderKindV1::RandomPopulation, &permuted)
            .unwrap();
        assert_eq!(first.order, second.order, "merge must not depend on input iteration order");
        let shared = first
            .order
            .iter()
            .copied()
            .filter(|id| *id != agents[4])
            .collect::<Vec<_>>();
        assert_eq!(shared, vec![agents[2], agents[0], agents[3], agents[1]]);
        assert_eq!(first.observed_shared_count, 4);
        assert_eq!(first.novel_count, 1);
    }

    #[test]
    fn observed_construction_seed_wins_and_missing_seed_uses_nonzero_fallback() {
        let agents = ids(5);
        let tape = validated_tape(&agents);
        let replay = MatchedStochasticReplayV1::new(&tape, CounterfactualRandomFieldV1::new(88));
        let observed = replay.offspring_construction_seed(12, agents[0], 0).unwrap();
        assert_eq!(observed.value(), 4242);
        assert_eq!(observed.origin(), MatchedRandomOriginV1::ObservedTape);
        let fallback = replay.offspring_construction_seed(18, agents[0], 2).unwrap();
        assert_ne!(fallback.value(), 0);
        assert_eq!(fallback.origin(), MatchedRandomOriginV1::CounterfactualFallback);
    }

    #[test]
    fn mutation_fallback_fails_closed_for_future_multi_birth_ordinal_until_field_address_expands() {
        let agents = ids(5);
        let tape = validated_tape(&agents);
        let replay = MatchedStochasticReplayV1::new(&tape, CounterfactualRandomFieldV1::new(88));
        assert_eq!(
            replay.mutation_occurrence_sample(
                18,
                agents[0],
                agents[1],
                1,
                GenomeTraitV1::SetPoint,
            ),
            Err(MatchedStochasticReplayErrorV1::FallbackMutationBirthOrdinalUnsupported {
                birth_ordinal: 1,
            })
        );
    }
}
