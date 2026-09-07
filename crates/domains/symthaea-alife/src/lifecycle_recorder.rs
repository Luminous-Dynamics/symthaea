// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic append-only recorder for first-class ALife lifecycle evidence.
//!
//! [`crate::lifecycle`] defines what lifecycle evidence means and validates a complete stream.
//! This module owns the smaller production concern needed before wiring those events into
//! [`crate::Population`]: assign one contiguous sequence, own population-local lifecycle time, and
//! build lifecycle transitions directly from authoritative [`crate::Organism`] state rather than
//! retyping IDs/genomes at each call site.
//!
//! The recorder deliberately does not infer lifecycle facts from behavioral absence. It receives
//! founder/birth/death transitions only when an authoritative owner says they happened.
//!
//! ## Population-local epoch contract
//!
//! Callers do **not** supply event timestamps. The recorder owns `current_epoch`.
//!
//! A future production `Population` integration should use the clock as follows:
//!
//! 1. founders are emitted at epoch 0;
//! 2. any out-of-band cull before the next population step is emitted at the current epoch;
//! 3. births/deaths caused by that `step` / `step_social` are emitted at the same epoch;
//! 4. after all authoritative transitions for the step are recorded, call [`Self::advance_epoch`]
//!    exactly once.
//!
//! This deliberately allows multiple transitions to share one epoch. The globally contiguous
//! `sequence` field establishes their exact order. In particular, predator-prey currently culls
//! prey immediately before the prey population's own `step`; the cull and that step therefore
//! belong to the same population-local epoch rather than borrowing the outer simulator's clock.

use crate::{
    Genome, GenomeEvidenceV1, LifecycleDeathCauseV1, LifecycleEventV1, LifecycleTransitionV1,
    Organism,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleRecorderErrorV1 {
    SequenceOverflow,
    EpochOverflow,
    InvalidFounderIdentity,
    InvalidNumericEvidence,
}

/// Append-only lifecycle evidence recorder.
///
/// `drain()` clears buffered events but intentionally does **not** reset `next_sequence` or
/// `current_epoch`; callers may persist long simulations in chunks while retaining one globally
/// contiguous lifecycle stream. A drained chunk therefore is not, by itself, a complete input to
/// [`crate::analyze_lifecycle_events`] unless it starts at sequence zero.
#[derive(Debug, Default)]
pub struct LifecycleRecorderV1 {
    events: Vec<LifecycleEventV1>,
    next_sequence: u64,
    current_epoch: u64,
}

impl LifecycleRecorderV1 {
    /// Build a recorder and emit deterministic founder events in the supplied population order.
    ///
    /// Founders are read from the actual organism state at lifecycle epoch 0. The recorder refuses
    /// to manufacture a founder event for an organism whose lineage/generation no longer
    /// represents genesis.
    pub fn from_founders(founders: &[Organism]) -> Result<Self, LifecycleRecorderErrorV1> {
        let mut recorder = Self::default();
        for founder in founders {
            if founder.lineage_id != founder.id || founder.generation != 0 {
                return Err(LifecycleRecorderErrorV1::InvalidFounderIdentity);
            }
            validate_organism_snapshot(founder)?;
            recorder.push_transition(LifecycleTransitionV1::Founder {
                agent_id: founder.id,
                lineage_id: founder.lineage_id,
                generation: founder.generation,
                genome: genome_evidence(founder),
                initial_energy_bits: founder.energy.to_bits(),
            })?;
        }
        Ok(recorder)
    }

    /// Record one authoritative birth at the recorder's current population-local epoch using exact
    /// snapshots of the reproductive parent, genome source, and already-constructed offspring.
    ///
    /// Keeping `reproductive_parent` and `genome_source` as separate arguments preserves the
    /// `InheritanceMode::RandomPeer` control exactly.
    pub fn record_birth(
        &mut self,
        reproductive_parent: &Organism,
        genome_source: &Organism,
        offspring: &Organism,
    ) -> Result<(), LifecycleRecorderErrorV1> {
        validate_organism_snapshot(reproductive_parent)?;
        validate_organism_snapshot(genome_source)?;
        validate_organism_snapshot(offspring)?;
        self.push_transition(LifecycleTransitionV1::Birth {
            reproductive_parent_id: reproductive_parent.id,
            genome_source_id: genome_source.id,
            offspring_id: offspring.id,
            lineage_id: offspring.lineage_id,
            generation: offspring.generation,
            reproductive_parent_genome: genome_evidence(reproductive_parent),
            genome_source_genome: genome_evidence(genome_source),
            offspring_genome: genome_evidence(offspring),
            initial_energy_bits: offspring.energy.to_bits(),
        })
    }

    /// Record one authoritative removal at the recorder's current epoch before the organism leaves
    /// the population.
    pub fn record_death(
        &mut self,
        organism: &Organism,
        cause: LifecycleDeathCauseV1,
    ) -> Result<(), LifecycleRecorderErrorV1> {
        if !organism.energy.is_finite() {
            return Err(LifecycleRecorderErrorV1::InvalidNumericEvidence);
        }
        self.push_transition(LifecycleTransitionV1::Death {
            agent_id: organism.id,
            cause,
            energy_bits: organism.energy.to_bits(),
        })
    }

    /// Population-local epoch assigned to newly recorded lifecycle transitions.
    pub fn current_epoch(&self) -> u64 {
        self.current_epoch
    }

    /// Advance the population-local lifecycle epoch exactly once after an authoritative population
    /// step has finished recording all of its transitions.
    ///
    /// Out-of-band culls that happen before the next step intentionally use the current epoch. This
    /// method changes no sequence or buffered-event state on overflow.
    pub fn advance_epoch(&mut self) -> Result<u64, LifecycleRecorderErrorV1> {
        let next = self
            .current_epoch
            .checked_add(1)
            .ok_or(LifecycleRecorderErrorV1::EpochOverflow)?;
        self.current_epoch = next;
        Ok(next)
    }

    /// Buffered events that have not yet been drained.
    pub fn events(&self) -> &[LifecycleEventV1] {
        &self.events
    }

    /// Next global lifecycle sequence number, including events already drained by the caller.
    pub fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    /// Drain buffered events without resetting global sequence or epoch state.
    pub fn drain(&mut self) -> Vec<LifecycleEventV1> {
        std::mem::take(&mut self.events)
    }

    fn push_transition(
        &mut self,
        transition: LifecycleTransitionV1,
    ) -> Result<(), LifecycleRecorderErrorV1> {
        let sequence = self.next_sequence;
        let next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(LifecycleRecorderErrorV1::SequenceOverflow)?;
        self.events.push(LifecycleEventV1 {
            sequence,
            tick: self.current_epoch,
            transition,
        });
        self.next_sequence = next_sequence;
        Ok(())
    }
}

fn genome_evidence(organism: &Organism) -> GenomeEvidenceV1 {
    GenomeEvidenceV1::from_genome(Genome::from_config(&organism.cfg))
}

fn validate_organism_snapshot(organism: &Organism) -> Result<(), LifecycleRecorderErrorV1> {
    let genome = Genome::from_config(&organism.cfg);
    let finite_genome = genome.set_point.is_finite()
        && genome.forage_efficiency.is_finite()
        && genome.action_temperature.is_finite()
        && genome.perceptual_grain.is_none_or(f64::is_finite);
    if !finite_genome || !organism.energy.is_finite() {
        return Err(LifecycleRecorderErrorV1::InvalidNumericEvidence);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AgentIdAllocator, OrganismConfig, analyze_lifecycle_events};

    fn founder(ids: &mut AgentIdAllocator, seed: u64) -> Organism {
        let id = ids.allocate();
        Organism::new(OrganismConfig::default(), seed).with_id(id)
    }

    #[test]
    fn founder_recording_is_immediately_valid_lifecycle_evidence() {
        let mut ids = AgentIdAllocator::new();
        let founders = vec![founder(&mut ids, 1), founder(&mut ids, 2)];
        let recorder = LifecycleRecorderV1::from_founders(&founders).expect("founders");
        assert_eq!(recorder.current_epoch(), 0);
        assert_eq!(recorder.events()[0].sequence, 0);
        assert_eq!(recorder.events()[0].tick, 0);
        assert_eq!(recorder.events()[1].sequence, 1);
        assert_eq!(recorder.events()[1].tick, 0);
        let ledger = analyze_lifecycle_events(recorder.events()).expect("valid lifecycle");
        assert_eq!(ledger.founder_count(), 2);
        assert_eq!(ledger.alive().len(), 2);
    }

    #[test]
    fn birth_keeps_reproductive_and_genetic_ancestry_distinct() {
        let mut ids = AgentIdAllocator::new();
        let parent = founder(&mut ids, 1);
        let mut donor = founder(&mut ids, 2);
        donor.cfg.forage_efficiency = 0.31;
        let mut child = Organism::new(donor.cfg, 3)
            .with_id(ids.allocate())
            .with_lineage(parent.lineage_id, parent.generation + 1);
        child.energy = 0.4;

        let founders = [parent, donor];
        let parent = &founders[0];
        let donor = &founders[1];
        let mut recorder = LifecycleRecorderV1::from_founders(&founders).expect("founders");
        recorder
            .record_birth(parent, donor, &child)
            .expect("birth");
        let ledger = analyze_lifecycle_events(recorder.events()).expect("valid lifecycle");
        assert!(ledger.reproductive_edges().contains(&(parent.id, child.id)));
        assert!(ledger.genetic_edges().contains(&(donor.id, child.id)));
    }

    #[test]
    fn drain_preserves_global_sequence_and_epoch() {
        let mut ids = AgentIdAllocator::new();
        let founder = founder(&mut ids, 1);
        let founders = [founder];
        let founder = &founders[0];
        let mut recorder = LifecycleRecorderV1::from_founders(&founders).expect("founder");
        recorder.advance_epoch().expect("epoch 1");
        let drained = recorder.drain();
        assert_eq!(drained[0].sequence, 0);
        assert_eq!(recorder.next_sequence(), 1);
        assert_eq!(recorder.current_epoch(), 1);
        recorder
            .record_death(founder, LifecycleDeathCauseV1::CullWeakest)
            .expect("death");
        assert_eq!(recorder.events()[0].sequence, 1);
        assert_eq!(recorder.events()[0].tick, 1);
    }

    #[test]
    fn multiple_transitions_share_an_epoch_and_sequence_orders_them() {
        let mut ids = AgentIdAllocator::new();
        let a = founder(&mut ids, 1);
        let b = founder(&mut ids, 2);
        let founders = [a, b];
        let mut recorder = LifecycleRecorderV1::from_founders(&founders).expect("founders");
        let founder_count = recorder.events().len();

        recorder
            .record_death(&founders[0], LifecycleDeathCauseV1::CullWeakest)
            .expect("first removal");
        recorder
            .record_death(&founders[1], LifecycleDeathCauseV1::EnergyThreshold)
            .expect("second removal");

        let first = &recorder.events()[founder_count];
        let second = &recorder.events()[founder_count + 1];
        assert_eq!(first.tick, 0);
        assert_eq!(second.tick, 0);
        assert!(first.sequence < second.sequence);
    }

    #[test]
    fn later_population_step_epoch_is_explicit_not_caller_supplied() {
        let mut ids = AgentIdAllocator::new();
        let founder = founder(&mut ids, 1);
        let founders = [founder];
        let founder = &founders[0];
        let mut recorder = LifecycleRecorderV1::from_founders(&founders).expect("founder");

        recorder
            .record_death(founder, LifecycleDeathCauseV1::CullWeakest)
            .expect("epoch-zero cull");
        recorder.advance_epoch().expect("finish first population epoch");
        assert_eq!(recorder.current_epoch(), 1);

        // The next event cannot accidentally be stamped with a stale/external tick because no
        // timestamp enters the record API at all.
        recorder
            .record_death(founder, LifecycleDeathCauseV1::EnergyThreshold)
            .expect("epoch-one death");
        assert_eq!(recorder.events().last().expect("event").tick, 1);
    }

    #[test]
    fn epoch_overflow_fails_without_changing_clock() {
        let mut recorder = LifecycleRecorderV1 {
            current_epoch: u64::MAX,
            ..LifecycleRecorderV1::default()
        };
        assert_eq!(
            recorder.advance_epoch(),
            Err(LifecycleRecorderErrorV1::EpochOverflow)
        );
        assert_eq!(recorder.current_epoch(), u64::MAX);
        assert_eq!(recorder.next_sequence(), 0);
        assert!(recorder.events().is_empty());
    }
}
