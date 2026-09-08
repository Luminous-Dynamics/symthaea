// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic RNG-stream contract for evolutionary controls.
//!
//! `Population` historically used one xorshift64 state for two logically distinct concerns:
//!
//! 1. choosing a genome source under `InheritanceMode::RandomPeer`;
//! 2. mutating the chosen genome.
//!
//! The production population now owns [`EvolutionRngStreamsV1`], keeping those concerns on
//! independent deterministic streams so source-selection work cannot shift later mutation draws.
//! This module also defines the exact persistence/restore boundary needed by future execution
//! checkpoints: snapshots may deserialize from untrusted storage, but zero xorshift states are
//! rejected before they can become live RNG authority.

use serde::{Deserialize, Serialize};

use crate::Genome;

/// The historical population mutation/source RNG started from this offset.
///
/// Keeping this exact value for the mutation stream preserves the pre-split `FromParent`
/// mutation sequence, because that mode never consumed RandomPeer source-selection draws.
pub const LEGACY_MUTATION_SEED_OFFSET_V1: u64 = 1_000_011;

/// Independent deterministic seed domain for RandomPeer genome-source selection.
pub const INHERITANCE_SOURCE_SEED_OFFSET_V1: u64 = 2_000_033;

/// Persistable exact states for both deterministic evolutionary RNG streams.
///
/// Fields remain public for existing measurement/tests, but a deserialized snapshot is not itself
/// executable authority. Use [`EvolutionRngStreamsV1::from_snapshot`] to validate and restore it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvolutionRngSnapshotV1 {
    pub mutation_state: u64,
    pub inheritance_source_state: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvolutionRngRestoreErrorV1 {
    ZeroMutationState,
    ZeroInheritanceSourceState,
}

/// Two independent xorshift64 states for evolutionary randomness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvolutionRngStreamsV1 {
    mutation_state: u64,
    inheritance_source_state: u64,
}

impl EvolutionRngStreamsV1 {
    pub fn new(seed_base: u64) -> Self {
        Self {
            mutation_state: seed_base
                .wrapping_add(LEGACY_MUTATION_SEED_OFFSET_V1)
                .max(1),
            inheritance_source_state: seed_base
                .wrapping_add(INHERITANCE_SOURCE_SEED_OFFSET_V1)
                .max(1),
        }
    }

    /// Restore the exact deterministic evolutionary RNG state from persisted evidence.
    ///
    /// Xorshift64's all-zero state is absorbing: once entered, every future draw remains zero.
    /// Fresh streams can never start at zero, and nonzero xorshift states never evolve into zero,
    /// so a zero value in persisted state is invalid/corrupt rather than a legitimate checkpoint.
    pub fn from_snapshot(
        snapshot: EvolutionRngSnapshotV1,
    ) -> Result<Self, EvolutionRngRestoreErrorV1> {
        if snapshot.mutation_state == 0 {
            return Err(EvolutionRngRestoreErrorV1::ZeroMutationState);
        }
        if snapshot.inheritance_source_state == 0 {
            return Err(EvolutionRngRestoreErrorV1::ZeroInheritanceSourceState);
        }
        Ok(Self {
            mutation_state: snapshot.mutation_state,
            inheritance_source_state: snapshot.inheritance_source_state,
        })
    }

    /// Mutable mutation state consumed by [`Genome::mutate`].
    ///
    /// The RNG algorithm remains owned by `Genome::mutate`; this contract only isolates its state
    /// from inheritance-source sampling.
    pub fn mutation_state_mut(&mut self) -> &mut u64 {
        &mut self.mutation_state
    }

    /// Draw only from the RandomPeer genome-source stream.
    pub fn next_inheritance_source_unit(&mut self) -> f64 {
        next_unit(&mut self.inheritance_source_state)
    }

    pub fn snapshot(&self) -> EvolutionRngSnapshotV1 {
        EvolutionRngSnapshotV1 {
            mutation_state: self.mutation_state,
            inheritance_source_state: self.inheritance_source_state,
        }
    }
}

fn next_unit(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OrganismConfig;

    #[test]
    fn mutation_stream_preserves_the_legacy_from_parent_seed_exactly() {
        let seed = 42u64;
        let streams = EvolutionRngStreamsV1::new(seed);
        assert_eq!(
            streams.snapshot().mutation_state,
            seed.wrapping_add(1_000_011).max(1)
        );
    }

    #[test]
    fn from_parent_mutation_sequence_matches_the_historical_raw_state() {
        let seed = 0x5eed_u64;
        let genome = Genome::from_config(&OrganismConfig::default());
        let mut legacy_state = seed
            .wrapping_add(LEGACY_MUTATION_SEED_OFFSET_V1)
            .max(1);
        let mut streams = EvolutionRngStreamsV1::new(seed);

        for birth in 0..64 {
            let legacy = genome.mutate(&mut legacy_state, 0.37, 0.05);
            let split = genome.mutate(streams.mutation_state_mut(), 0.37, 0.05);
            assert_eq!(
                legacy, split,
                "split mutation stream drifted from historical FromParent sequence at birth {birth}"
            );
            assert_eq!(
                legacy_state,
                streams.snapshot().mutation_state,
                "mutation RNG state drifted at birth {birth}"
            );
        }
    }

    #[test]
    fn inheritance_source_draws_do_not_change_mutation_state() {
        let mut streams = EvolutionRngStreamsV1::new(7);
        let before = streams.snapshot().mutation_state;
        for _ in 0..100 {
            let value = streams.next_inheritance_source_unit();
            assert!((0.0..=1.0).contains(&value));
        }
        assert_eq!(streams.snapshot().mutation_state, before);
    }

    #[test]
    fn source_sampling_cannot_shift_the_next_mutation_when_source_genome_is_equal() {
        let genome = Genome::from_config(&OrganismConfig::default());
        let mut selected = EvolutionRngStreamsV1::new(123);
        let mut random_peer = EvolutionRngStreamsV1::new(123);

        // Simulate arbitrary RandomPeer source-selection work before the same birth. The mutation
        // stream must remain untouched by those draws.
        for _ in 0..17 {
            random_peer.next_inheritance_source_unit();
        }

        let selected_mutant = genome.mutate(selected.mutation_state_mut(), 1.0, 0.05);
        let random_peer_mutant = genome.mutate(random_peer.mutation_state_mut(), 1.0, 0.05);

        assert_eq!(selected_mutant, random_peer_mutant);
        assert_eq!(
            selected.snapshot().mutation_state,
            random_peer.snapshot().mutation_state
        );
    }

    #[test]
    fn inheritance_source_sequence_is_deterministic_and_domain_separated() {
        let mut a = EvolutionRngStreamsV1::new(999);
        let mut b = EvolutionRngStreamsV1::new(999);
        assert_ne!(
            a.snapshot().mutation_state,
            a.snapshot().inheritance_source_state
        );
        for _ in 0..32 {
            assert_eq!(
                a.next_inheritance_source_unit().to_bits(),
                b.next_inheritance_source_unit().to_bits()
            );
        }
    }

    #[test]
    fn validated_restore_preserves_both_future_streams_exactly() {
        let genome = Genome::from_config(&OrganismConfig::default());
        let mut uninterrupted = EvolutionRngStreamsV1::new(0x51eed);

        for _ in 0..7 {
            uninterrupted.next_inheritance_source_unit();
        }
        for _ in 0..5 {
            let _ = genome.mutate(uninterrupted.mutation_state_mut(), 0.63, 0.07);
        }

        let persisted = uninterrupted.snapshot();
        let encoded = serde_json::to_string(&persisted).expect("serialize RNG snapshot");
        let decoded: EvolutionRngSnapshotV1 =
            serde_json::from_str(&encoded).expect("deserialize RNG snapshot");
        let mut restored =
            EvolutionRngStreamsV1::from_snapshot(decoded).expect("validated RNG restore");

        assert_eq!(restored.snapshot(), uninterrupted.snapshot());
        for _ in 0..32 {
            assert_eq!(
                restored.next_inheritance_source_unit().to_bits(),
                uninterrupted.next_inheritance_source_unit().to_bits()
            );
            assert_eq!(
                genome.mutate(restored.mutation_state_mut(), 0.41, 0.03),
                genome.mutate(uninterrupted.mutation_state_mut(), 0.41, 0.03)
            );
        }
        assert_eq!(restored.snapshot(), uninterrupted.snapshot());
    }

    #[test]
    fn zero_state_snapshots_fail_closed_before_becoming_live_streams() {
        assert_eq!(
            EvolutionRngStreamsV1::from_snapshot(EvolutionRngSnapshotV1 {
                mutation_state: 0,
                inheritance_source_state: 1,
            }),
            Err(EvolutionRngRestoreErrorV1::ZeroMutationState)
        );
        assert_eq!(
            EvolutionRngStreamsV1::from_snapshot(EvolutionRngSnapshotV1 {
                mutation_state: 1,
                inheritance_source_state: 0,
            }),
            Err(EvolutionRngRestoreErrorV1::ZeroInheritanceSourceState)
        );
    }
}
