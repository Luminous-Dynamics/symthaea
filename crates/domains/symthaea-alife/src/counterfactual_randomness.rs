// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stateless pseudo-random field for matched counterfactual Genesis experiments.
//!
//! Exact snapshot/restore guarantees that one world continues identically after persistence. A
//! causal fork needs a stronger property: if an intervention changes the number or ordering of
//! later stochastic events, that difference must not phase-shift the pseudo-random draws assigned
//! to logically shared events. Sequential RNG streams cannot provide that guarantee by themselves.
//!
//! This module therefore maps an immutable experiment root seed plus a semantic draw key directly
//! to a deterministic pseudo-random value. There is no mutable cursor and draw order is irrelevant.
//! The field is deliberately **not cryptographic**; it uses the SplitMix64 finalizer as a small,
//! inspectable deterministic mixer. Statistical adequacy for any future scientific protocol must be
//! qualified separately from the structural no-phase-shift theorem established here.
//!
//! Live Population, scheduler, mutation, and FEP action paths do not consume this field yet. Those
//! adapters belong in later profile-specific experimental tranches after the exact-resume substrate
//! and this contract execute successfully.

use serde::{Deserialize, Serialize};

use crate::{AgentId, GenomeTraitV1};

const FIELD_VERSION_TAG_V1: u64 = 0x4346_524E_4456_3031; // "CFRNDV01"

/// Domain separation for the stochastic surfaces relevant to a matched Genesis counterfactual.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CounterfactualRandomDomainV1 {
    SchedulerPriority,
    InheritanceCandidatePriority,
    MutationOccurrence,
    MutationPerturbation,
    OffspringConstructionSeed,
    AgentActionSample,
}

impl CounterfactualRandomDomainV1 {
    fn tag(self) -> u64 {
        match self {
            Self::SchedulerPriority => 0x5343_4845_445F_5052,
            Self::InheritanceCandidatePriority => 0x494E_4845_525F_5052,
            Self::MutationOccurrence => 0x4D55_545F_4F43_4352,
            Self::MutationPerturbation => 0x4D55_545F_5045_5254,
            Self::OffspringConstructionSeed => 0x4F46_465F_5345_4544,
            Self::AgentActionSample => 0x4143_5449_4F4E_524E,
        }
    }
}

/// Complete semantic address for one pseudo-random draw.
///
/// `subject_id` is normally the primary agent/reproductive parent. `counterpart_id` is zero when
/// the draw has no second identity, otherwise it names the candidate/source/partner whose relation
/// is being randomized. `ordinal` distinguishes multiple draws of the same semantic kind, such as
/// mutation occurrence versus per-trait perturbation or future multiple births by one parent/tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CounterfactualDrawKeyV1 {
    pub domain: CounterfactualRandomDomainV1,
    pub tick: u64,
    pub subject_id: u64,
    pub counterpart_id: u64,
    pub ordinal: u64,
}

impl CounterfactualDrawKeyV1 {
    pub fn scheduler_priority(tick: u64, agent_id: AgentId) -> Self {
        Self {
            domain: CounterfactualRandomDomainV1::SchedulerPriority,
            tick,
            subject_id: agent_id.raw(),
            counterpart_id: 0,
            ordinal: 0,
        }
    }

    pub fn inheritance_candidate_priority(
        tick: u64,
        reproductive_parent_id: AgentId,
        candidate_id: AgentId,
    ) -> Self {
        Self {
            domain: CounterfactualRandomDomainV1::InheritanceCandidatePriority,
            tick,
            subject_id: reproductive_parent_id.raw(),
            counterpart_id: candidate_id.raw(),
            ordinal: 0,
        }
    }

    pub fn mutation_occurrence(
        tick: u64,
        reproductive_parent_id: AgentId,
        genome_source_id: AgentId,
        trait_id: GenomeTraitV1,
    ) -> Self {
        Self {
            domain: CounterfactualRandomDomainV1::MutationOccurrence,
            tick,
            subject_id: reproductive_parent_id.raw(),
            counterpart_id: genome_source_id.raw(),
            ordinal: genome_trait_tag(trait_id),
        }
    }

    pub fn mutation_perturbation(
        tick: u64,
        reproductive_parent_id: AgentId,
        genome_source_id: AgentId,
        trait_id: GenomeTraitV1,
    ) -> Self {
        Self {
            domain: CounterfactualRandomDomainV1::MutationPerturbation,
            tick,
            subject_id: reproductive_parent_id.raw(),
            counterpart_id: genome_source_id.raw(),
            ordinal: genome_trait_tag(trait_id),
        }
    }

    pub fn offspring_construction_seed(
        tick: u64,
        reproductive_parent_id: AgentId,
        birth_ordinal: u64,
    ) -> Self {
        Self {
            domain: CounterfactualRandomDomainV1::OffspringConstructionSeed,
            tick,
            subject_id: reproductive_parent_id.raw(),
            counterpart_id: 0,
            ordinal: birth_ordinal,
        }
    }

    pub fn agent_action_sample(tick: u64, agent_id: AgentId, draw_ordinal: u64) -> Self {
        Self {
            domain: CounterfactualRandomDomainV1::AgentActionSample,
            tick,
            subject_id: agent_id.raw(),
            counterpart_id: 0,
            ordinal: draw_ordinal,
        }
    }
}

/// Serializable root of one stateless matched-randomness field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CounterfactualRandomFieldV1 {
    root_seed: u64,
}

impl CounterfactualRandomFieldV1 {
    pub fn new(root_seed: u64) -> Self {
        Self { root_seed }
    }

    pub fn root_seed(&self) -> u64 {
        self.root_seed
    }

    /// Deterministically map one semantic key to a full-width pseudo-random word.
    ///
    /// This method has no mutable state: querying any other key before or after this key cannot
    /// change its result.
    pub fn draw_u64(&self, key: CounterfactualDrawKeyV1) -> u64 {
        let mut x = mix64(self.root_seed ^ FIELD_VERSION_TAG_V1);
        x = mix64(x ^ key.domain.tag());
        x = mix64(x ^ key.tick);
        x = mix64(x ^ key.subject_id);
        x = mix64(x ^ key.counterpart_id);
        mix64(x ^ key.ordinal)
    }

    /// Exact deterministic unit interval sample in `[0, 1)` using the high 53 bits.
    pub fn draw_unit_f64(&self, key: CounterfactualDrawKeyV1) -> f64 {
        const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
        ((self.draw_u64(key) >> 11) as f64) * SCALE
    }

    /// Deterministic nonzero seed for APIs whose all-zero RNG state is invalid.
    pub fn draw_nonzero_u64(&self, key: CounterfactualDrawKeyV1) -> u64 {
        let value = self.draw_u64(key);
        if value == 0 {
            // Fixed versioned fallback. This branch is deterministic and does not consume another
            // draw; it exists only because several legacy xorshift owners reject zero state.
            0x9E37_79B9_7F4A_7C15
        } else {
            value
        }
    }
}

fn genome_trait_tag(trait_id: GenomeTraitV1) -> u64 {
    match trait_id {
        GenomeTraitV1::SetPoint => 1,
        GenomeTraitV1::ForageEfficiency => 2,
        GenomeTraitV1::ActionTemperature => 3,
        GenomeTraitV1::PerceptualGrain => 4,
    }
}

/// SplitMix64 finalizer, used here only as a deterministic non-cryptographic mixing primitive.
fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AgentIdAllocator;

    fn ids(n: usize) -> Vec<AgentId> {
        let mut allocator = AgentIdAllocator::new();
        (0..n).map(|_| allocator.allocate()).collect()
    }

    #[test]
    fn same_semantic_key_is_bit_exact_independent_of_query_order() {
        let agents = ids(4);
        let field = CounterfactualRandomFieldV1::new(0xCA55_A11F);
        let target = CounterfactualDrawKeyV1::scheduler_priority(17, agents[2]);
        let expected = field.draw_u64(target);

        // Query unrelated domains/events in between. A sequential stream would advance; this field
        // cannot because the result is a pure function of `(root_seed, key)`.
        for tick in 0..100u64 {
            let _ = field.draw_u64(CounterfactualDrawKeyV1::agent_action_sample(
                tick,
                agents[(tick as usize) % agents.len()],
                tick % 3,
            ));
        }
        assert_eq!(field.draw_u64(target), expected);
    }

    #[test]
    fn insertion_or_removal_of_other_agents_does_not_shift_shared_scheduler_priorities() {
        let agents = ids(6);
        let field = CounterfactualRandomFieldV1::new(77);
        let tick = 9;
        let small = &agents[..4];
        let large = &agents[..6];

        let small_priorities = small
            .iter()
            .map(|&id| (id, field.draw_u64(CounterfactualDrawKeyV1::scheduler_priority(tick, id))))
            .collect::<Vec<_>>();
        let large_priorities = large
            .iter()
            .map(|&id| (id, field.draw_u64(CounterfactualDrawKeyV1::scheduler_priority(tick, id))))
            .collect::<Vec<_>>();

        for (id, priority) in small_priorities {
            assert_eq!(
                large_priorities.iter().find(|(candidate, _)| *candidate == id).unwrap().1,
                priority,
                "shared agent random priority shifted when unrelated agents were inserted"
            );
        }
    }

    #[test]
    fn unrelated_birth_queries_do_not_phase_shift_a_shared_mutation_draw() {
        let agents = ids(5);
        let field = CounterfactualRandomFieldV1::new(0x5EED_1234);
        let shared = CounterfactualDrawKeyV1::mutation_perturbation(
            12,
            agents[1],
            agents[3],
            GenomeTraitV1::ForageEfficiency,
        );
        let expected = field.draw_u64(shared);

        for ordinal in 0..32u64 {
            let _ = field.draw_u64(CounterfactualDrawKeyV1::offspring_construction_seed(
                12,
                agents[4],
                ordinal,
            ));
        }
        assert_eq!(field.draw_u64(shared), expected);
    }

    #[test]
    fn stochastic_domains_are_separated_for_the_same_subject_coordinates() {
        let agents = ids(2);
        let field = CounterfactualRandomFieldV1::new(42);
        let occurrence = CounterfactualDrawKeyV1::mutation_occurrence(
            5,
            agents[0],
            agents[1],
            GenomeTraitV1::SetPoint,
        );
        let perturbation = CounterfactualDrawKeyV1::mutation_perturbation(
            5,
            agents[0],
            agents[1],
            GenomeTraitV1::SetPoint,
        );
        assert_ne!(field.draw_u64(occurrence), field.draw_u64(perturbation));
    }

    #[test]
    fn every_genome_v1_trait_gets_a_distinct_mutation_address() {
        let agents = ids(2);
        let field = CounterfactualRandomFieldV1::new(991);
        let traits = [
            GenomeTraitV1::SetPoint,
            GenomeTraitV1::ForageEfficiency,
            GenomeTraitV1::ActionTemperature,
            GenomeTraitV1::PerceptualGrain,
        ];
        let mut keys = traits
            .into_iter()
            .map(|trait_id| {
                field.draw_u64(CounterfactualDrawKeyV1::mutation_occurrence(
                    3,
                    agents[0],
                    agents[1],
                    trait_id,
                ))
            })
            .collect::<Vec<_>>();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), 4);
    }

    #[test]
    fn unit_samples_are_finite_and_strictly_below_one() {
        let agents = ids(1);
        let field = CounterfactualRandomFieldV1::new(0);
        for ordinal in 0..10_000u64 {
            let value = field.draw_unit_f64(CounterfactualDrawKeyV1::agent_action_sample(
                ordinal,
                agents[0],
                ordinal,
            ));
            assert!(value.is_finite());
            assert!((0.0..1.0).contains(&value));
        }
    }

    #[test]
    fn persistence_round_trip_preserves_the_random_field_exactly() {
        let agents = ids(2);
        let field = CounterfactualRandomFieldV1::new(0xDEAD_BEEF_CAFE_BABE);
        let encoded = serde_json::to_string(&field).expect("serialize random field");
        let restored: CounterfactualRandomFieldV1 =
            serde_json::from_str(&encoded).expect("deserialize random field");
        assert_eq!(restored, field);
        let key = CounterfactualDrawKeyV1::inheritance_candidate_priority(
            101,
            agents[0],
            agents[1],
        );
        assert_eq!(restored.draw_u64(key), field.draw_u64(key));
    }
}
