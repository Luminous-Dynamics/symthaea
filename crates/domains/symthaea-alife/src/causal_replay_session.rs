// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bound protocol for one matched Genesis mutation-rescue counterfactual.
//!
//! This module does not execute an intervention. It composes already-separated authorities into
//! one fail-closed experimental subject: an exact pre-birth Earth-forced execution capsule, an
//! observed natural endpoint, the observed stochastic interval between them, stateless fallback
//! randomness for branch-novel events, one exact mutation-reversion request, and a bit-preserving
//! perturbation/outcome protocol.
//!
//! The session validates semantic continuity only. It does not authenticate the origin of persisted
//! bytes, prove stochastic-tape completeness, or establish a causal rescue result. Live replay and
//! sham-vs-revert execution remain separate qualification tranches.

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use crate::{
    AgentId, BirthMutationInterventionKindV1, BirthMutationInterventionV1,
    CounterfactualRandomFieldV1, GenesisEarthExecutionCapsuleErrorV1,
    GenesisEarthExecutionCapsuleV1, GenesisExecutionProtocolV1, GenomeEvidenceV1, GenomeTraitV1,
    LifecycleCheckpointErrorV1, MatchedStochasticReplayV1, ObservedStochasticTapeErrorV1,
    ObservedStochasticTapeV1, PerturbationError, PerturbationSchedule, ResourcePerturbation,
    ValidatedGenesisEarthExecutionCapsuleV1, ValidatedObservedStochasticTapeV1,
};

/// Bit-preserving persistence for one deterministic resource perturbation window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourcePerturbationSnapshotV1 {
    start_tick: u64,
    end_tick_exclusive: u64,
    multiplier_bits: u64,
    delta_bits: u64,
}

impl ResourcePerturbationSnapshotV1 {
    pub fn from_perturbation(window: ResourcePerturbation) -> Self {
        Self {
            start_tick: window.start_tick(),
            end_tick_exclusive: window.end_tick_exclusive(),
            multiplier_bits: window.multiplier().to_bits(),
            delta_bits: window.delta().to_bits(),
        }
    }

    pub fn start_tick(self) -> u64 {
        self.start_tick
    }

    pub fn end_tick_exclusive(self) -> u64 {
        self.end_tick_exclusive
    }

    pub fn multiplier(self) -> f64 {
        f64::from_bits(self.multiplier_bits)
    }

    pub fn delta(self) -> f64 {
        f64::from_bits(self.delta_bits)
    }
}

/// Frozen primary outcome definition for a rescue replay.
///
/// `evaluation_end_tick` is inclusive, matching `analyze_recovery_through`. The baseline is the
/// exact half-open interval `[focal.start - baseline_lookback, focal.start)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CausalRescueOutcomeProtocolV1 {
    focal_perturbation_index: u64,
    baseline_lookback_ticks: u64,
    recovery_fraction_bits: u64,
    evaluation_end_tick: u64,
}

impl CausalRescueOutcomeProtocolV1 {
    pub fn new(
        focal_perturbation_index: u64,
        baseline_lookback_ticks: u64,
        recovery_fraction: f64,
        evaluation_end_tick: u64,
    ) -> Self {
        Self {
            focal_perturbation_index,
            baseline_lookback_ticks,
            recovery_fraction_bits: recovery_fraction.to_bits(),
            evaluation_end_tick,
        }
    }

    pub fn focal_perturbation_index(self) -> u64 {
        self.focal_perturbation_index
    }

    pub fn baseline_lookback_ticks(self) -> u64 {
        self.baseline_lookback_ticks
    }

    pub fn recovery_fraction(self) -> f64 {
        f64::from_bits(self.recovery_fraction_bits)
    }

    pub fn evaluation_end_tick(self) -> u64 {
        self.evaluation_end_tick
    }
}

/// Persisted experimental specification. Deserialization is not experimental authority; call
/// [`Self::validate`] before constructing any replay helper or interpreting the session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalRescueReplaySessionV1 {
    fork_previous_behavior_next_tick: u64,
    fork_capsule: GenesisEarthExecutionCapsuleV1,
    natural_end_capsule: GenesisEarthExecutionCapsuleV1,
    observed_stochastic_tape: ObservedStochasticTapeV1,
    fallback_random_field: CounterfactualRandomFieldV1,
    intervention: BirthMutationInterventionV1,
    perturbations: Vec<ResourcePerturbationSnapshotV1>,
    outcome: CausalRescueOutcomeProtocolV1,
}

impl CausalRescueReplaySessionV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        fork_previous_behavior_next_tick: u64,
        fork_capsule: GenesisEarthExecutionCapsuleV1,
        natural_end_capsule: GenesisEarthExecutionCapsuleV1,
        observed_stochastic_tape: ObservedStochasticTapeV1,
        fallback_random_field: CounterfactualRandomFieldV1,
        intervention: BirthMutationInterventionV1,
        perturbations: Vec<ResourcePerturbationSnapshotV1>,
        outcome: CausalRescueOutcomeProtocolV1,
    ) -> Self {
        Self {
            fork_previous_behavior_next_tick,
            fork_capsule,
            natural_end_capsule,
            observed_stochastic_tape,
            fallback_random_field,
            intervention,
            perturbations,
            outcome,
        }
    }

    pub fn validate(
        self,
    ) -> Result<ValidatedCausalRescueReplaySessionV1, CausalRescueReplaySessionErrorV1> {
        let fork = self
            .fork_capsule
            .validate_after(self.fork_previous_behavior_next_tick)
            .map_err(CausalRescueReplaySessionErrorV1::ForkCapsule)?;
        let fork_tick = fork.evidence().behavior_next_tick();

        let natural_end = self
            .natural_end_capsule
            .validate_after(fork_tick)
            .map_err(CausalRescueReplaySessionErrorV1::NaturalEndCapsule)?;
        let natural_end_tick = natural_end.evidence().behavior_next_tick();
        if natural_end_tick <= fork_tick {
            return Err(CausalRescueReplaySessionErrorV1::EmptyReplayInterval {
                fork_tick,
                natural_end_tick,
            });
        }
        if fork.protocol() != natural_end.protocol() {
            return Err(CausalRescueReplaySessionErrorV1::ProtocolMismatch {
                fork: fork.protocol(),
                natural_end: natural_end.protocol(),
            });
        }

        validate_lifecycle_extension(&fork, &natural_end)?;

        let tape = self
            .observed_stochastic_tape
            .validate()
            .map_err(CausalRescueReplaySessionErrorV1::ObservedTape)?;
        let tape_subject = tape.subject();
        if tape_subject.start_behavior_next_tick != fork_tick {
            return Err(CausalRescueReplaySessionErrorV1::TapeStartBehaviorMismatch {
                expected: fork_tick,
                observed: tape_subject.start_behavior_next_tick,
            });
        }
        let fork_lifecycle_sequence = fork.evidence().lifecycle().next_sequence();
        if tape_subject.start_lifecycle_next_sequence != fork_lifecycle_sequence {
            return Err(CausalRescueReplaySessionErrorV1::TapeStartLifecycleMismatch {
                expected: fork_lifecycle_sequence,
                observed: tape_subject.start_lifecycle_next_sequence,
            });
        }
        if tape_subject.end_behavior_next_tick != natural_end_tick {
            return Err(CausalRescueReplaySessionErrorV1::TapeEndBehaviorMismatch {
                expected: natural_end_tick,
                observed: tape_subject.end_behavior_next_tick,
            });
        }

        validate_intervention_shape(&self.intervention, fork_tick)?;
        let subject = self.intervention.subject();
        let fork_alive = fork.evidence().lifecycle().alive_ids();
        if !fork_alive.contains(&subject.reproductive_parent_id()) {
            return Err(CausalRescueReplaySessionErrorV1::InterventionParentNotAliveAtFork {
                parent_id: subject.reproductive_parent_id(),
            });
        }
        if !fork_alive.contains(&subject.genome_source_id()) {
            return Err(CausalRescueReplaySessionErrorV1::InterventionSourceNotAliveAtFork {
                source_id: subject.genome_source_id(),
            });
        }

        let natural_offspring_id = bind_intervention_to_observed_birth(
            &fork,
            &natural_end,
            &self.intervention,
        )?;

        let perturbations = validate_perturbations(&self.perturbations)?;
        validate_outcome_protocol(
            self.outcome,
            &perturbations,
            subject.tick(),
            natural_end_tick,
        )?;

        Ok(ValidatedCausalRescueReplaySessionV1 {
            fork,
            natural_end,
            tape,
            fallback_random_field: self.fallback_random_field,
            intervention: self.intervention,
            perturbations,
            outcome: self.outcome,
            natural_offspring_id,
        })
    }
}

/// Non-serializable authority-bearing session produced only after all component and cross-domain
/// checks succeed.
#[derive(Debug, Clone)]
pub struct ValidatedCausalRescueReplaySessionV1 {
    fork: ValidatedGenesisEarthExecutionCapsuleV1,
    natural_end: ValidatedGenesisEarthExecutionCapsuleV1,
    tape: ValidatedObservedStochasticTapeV1,
    fallback_random_field: CounterfactualRandomFieldV1,
    intervention: BirthMutationInterventionV1,
    perturbations: Vec<ResourcePerturbation>,
    outcome: CausalRescueOutcomeProtocolV1,
    natural_offspring_id: AgentId,
}

impl ValidatedCausalRescueReplaySessionV1 {
    pub fn fork(&self) -> &ValidatedGenesisEarthExecutionCapsuleV1 {
        &self.fork
    }

    pub fn natural_end(&self) -> &ValidatedGenesisEarthExecutionCapsuleV1 {
        &self.natural_end
    }

    pub fn observed_tape(&self) -> &ValidatedObservedStochasticTapeV1 {
        &self.tape
    }

    pub fn replay_source(&self) -> MatchedStochasticReplayV1<'_> {
        MatchedStochasticReplayV1::new(&self.tape, self.fallback_random_field)
    }

    pub fn intervention(&self) -> &BirthMutationInterventionV1 {
        &self.intervention
    }

    pub fn perturbations(&self) -> &[ResourcePerturbation] {
        &self.perturbations
    }

    pub fn perturbation_schedule(&self) -> PerturbationSchedule {
        PerturbationSchedule::new(self.perturbations.clone())
    }

    pub fn outcome(&self) -> CausalRescueOutcomeProtocolV1 {
        self.outcome
    }

    pub fn natural_offspring_id(&self) -> AgentId {
        self.natural_offspring_id
    }
}

#[derive(Debug)]
pub enum CausalRescueReplaySessionErrorV1 {
    ForkCapsule(GenesisEarthExecutionCapsuleErrorV1),
    NaturalEndCapsule(GenesisEarthExecutionCapsuleErrorV1),
    ForkLifecycle(LifecycleCheckpointErrorV1),
    NaturalEndLifecycle(LifecycleCheckpointErrorV1),
    LifecyclePrefixMismatch,
    EmptyReplayInterval {
        fork_tick: u64,
        natural_end_tick: u64,
    },
    ProtocolMismatch {
        fork: GenesisExecutionProtocolV1,
        natural_end: GenesisExecutionProtocolV1,
    },
    ObservedTape(ObservedStochasticTapeErrorV1),
    TapeStartBehaviorMismatch {
        expected: u64,
        observed: u64,
    },
    TapeStartLifecycleMismatch {
        expected: u64,
        observed: u64,
    },
    TapeEndBehaviorMismatch {
        expected: u64,
        observed: u64,
    },
    InterventionMustRevertMutation,
    InterventionEmptyTraitSet,
    InterventionTraitsNotStrictlySorted,
    InterventionTickMismatch {
        expected: u64,
        observed: u64,
    },
    InterventionParentNotAliveAtFork {
        parent_id: AgentId,
    },
    InterventionSourceNotAliveAtFork {
        source_id: AgentId,
    },
    ObservedNaturalBirthMissing,
    ObservedNaturalBirthAmbiguous {
        matches: usize,
    },
    ObservedNaturalBirthAlreadyPresentAtFork {
        offspring_id: AgentId,
    },
    InterventionTraitNotObservedMutated {
        trait_id: GenomeTraitV1,
    },
    EmptyPerturbationSchedule,
    PerturbationWindowInvalid {
        index: usize,
        start_tick: u64,
        end_tick_exclusive: u64,
    },
    PerturbationDomain {
        index: usize,
        error: PerturbationError,
    },
    PerturbationOrderNotCanonical {
        index: usize,
    },
    FocalPerturbationIndexOutOfBounds {
        index: u64,
        len: usize,
    },
    ZeroBaselineLookback,
    BaselineWindowUnderflow {
        focal_start_tick: u64,
        baseline_lookback_ticks: u64,
    },
    InvalidRecoveryFraction {
        bits: u64,
    },
    EvaluationEndsBeforeFocalPerturbation {
        evaluation_end_tick: u64,
        focal_end_tick_exclusive: u64,
    },
    EvaluationEndsBeforeIntervention {
        evaluation_end_tick: u64,
        intervention_tick: u64,
    },
    EvaluationOutsideNaturalEndpoint {
        evaluation_end_tick: u64,
        natural_end_behavior_next_tick: u64,
    },
}

fn validate_lifecycle_extension(
    fork: &ValidatedGenesisEarthExecutionCapsuleV1,
    natural_end: &ValidatedGenesisEarthExecutionCapsuleV1,
) -> Result<(), CausalRescueReplaySessionErrorV1> {
    let fork_prefix = fork
        .evidence()
        .lifecycle()
        .reconstruct_complete_prefix()
        .map_err(CausalRescueReplaySessionErrorV1::ForkLifecycle)?;
    let natural_prefix = natural_end
        .evidence()
        .lifecycle()
        .reconstruct_complete_prefix()
        .map_err(CausalRescueReplaySessionErrorV1::NaturalEndLifecycle)?;
    if natural_prefix.len() < fork_prefix.len()
        || natural_prefix[..fork_prefix.len()] != fork_prefix[..]
    {
        return Err(CausalRescueReplaySessionErrorV1::LifecyclePrefixMismatch);
    }
    Ok(())
}

fn validate_intervention_shape(
    intervention: &BirthMutationInterventionV1,
    fork_tick: u64,
) -> Result<(), CausalRescueReplaySessionErrorV1> {
    let observed_tick = intervention.subject().tick();
    if observed_tick != fork_tick {
        return Err(CausalRescueReplaySessionErrorV1::InterventionTickMismatch {
            expected: fork_tick,
            observed: observed_tick,
        });
    }

    let traits = match intervention.kind() {
        BirthMutationInterventionKindV1::Sham => {
            return Err(CausalRescueReplaySessionErrorV1::InterventionMustRevertMutation);
        }
        BirthMutationInterventionKindV1::RevertTraits { traits } => traits,
    };
    if traits.is_empty() {
        return Err(CausalRescueReplaySessionErrorV1::InterventionEmptyTraitSet);
    }
    if traits.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(CausalRescueReplaySessionErrorV1::InterventionTraitsNotStrictlySorted);
    }
    Ok(())
}

fn bind_intervention_to_observed_birth(
    fork: &ValidatedGenesisEarthExecutionCapsuleV1,
    natural_end: &ValidatedGenesisEarthExecutionCapsuleV1,
    intervention: &BirthMutationInterventionV1,
) -> Result<AgentId, CausalRescueReplaySessionErrorV1> {
    let subject = intervention.subject();
    let natural_genome = subject.natural_offspring_genome();
    let records = natural_end.evidence().lifecycle().ledger().records();
    let matching = records
        .values()
        .filter(|record| {
            record.born_tick == subject.tick()
                && record.reproductive_parent_id == Some(subject.reproductive_parent_id())
                && record.genome_source_id == Some(subject.genome_source_id())
                && record.genome == natural_genome
        })
        .map(|record| record.agent_id)
        .collect::<Vec<_>>();

    let offspring_id = match matching.as_slice() {
        [] => return Err(CausalRescueReplaySessionErrorV1::ObservedNaturalBirthMissing),
        [only] => *only,
        many => {
            return Err(CausalRescueReplaySessionErrorV1::ObservedNaturalBirthAmbiguous {
                matches: many.len(),
            });
        }
    };
    if fork
        .evidence()
        .lifecycle()
        .ledger()
        .records()
        .contains_key(&offspring_id)
    {
        return Err(
            CausalRescueReplaySessionErrorV1::ObservedNaturalBirthAlreadyPresentAtFork {
                offspring_id,
            },
        );
    }

    let source_genome = fork
        .evidence()
        .lifecycle()
        .ledger()
        .records()
        .get(&subject.genome_source_id())
        .expect("source liveness was validated before birth binding")
        .genome;
    if let BirthMutationInterventionKindV1::RevertTraits { traits } = intervention.kind() {
        for &trait_id in traits {
            if !trait_differs(trait_id, source_genome, natural_genome) {
                return Err(
                    CausalRescueReplaySessionErrorV1::InterventionTraitNotObservedMutated {
                        trait_id,
                    },
                );
            }
        }
    }

    Ok(offspring_id)
}

fn trait_differs(
    trait_id: GenomeTraitV1,
    source: GenomeEvidenceV1,
    offspring: GenomeEvidenceV1,
) -> bool {
    match trait_id {
        GenomeTraitV1::SetPoint => source.set_point_bits != offspring.set_point_bits,
        GenomeTraitV1::ForageEfficiency => {
            source.forage_efficiency_bits != offspring.forage_efficiency_bits
        }
        GenomeTraitV1::ActionTemperature => {
            source.action_temperature_bits != offspring.action_temperature_bits
        }
        GenomeTraitV1::PerceptualGrain => {
            source.perceptual_grain_bits != offspring.perceptual_grain_bits
        }
    }
}

fn validate_perturbations(
    snapshots: &[ResourcePerturbationSnapshotV1],
) -> Result<Vec<ResourcePerturbation>, CausalRescueReplaySessionErrorV1> {
    if snapshots.is_empty() {
        return Err(CausalRescueReplaySessionErrorV1::EmptyPerturbationSchedule);
    }

    let mut windows = Vec::with_capacity(snapshots.len());
    for (index, snapshot) in snapshots.iter().copied().enumerate() {
        if snapshot.end_tick_exclusive <= snapshot.start_tick {
            return Err(CausalRescueReplaySessionErrorV1::PerturbationWindowInvalid {
                index,
                start_tick: snapshot.start_tick,
                end_tick_exclusive: snapshot.end_tick_exclusive,
            });
        }
        let duration = snapshot.end_tick_exclusive - snapshot.start_tick;
        let window = ResourcePerturbation::new(
            snapshot.start_tick,
            duration,
            snapshot.multiplier(),
            snapshot.delta(),
        )
        .map_err(|error| CausalRescueReplaySessionErrorV1::PerturbationDomain {
            index,
            error,
        })?;
        windows.push(window);
    }

    for (index, pair) in windows.windows(2).enumerate() {
        if compare_perturbations(pair[0], pair[1]) == Ordering::Greater {
            return Err(CausalRescueReplaySessionErrorV1::PerturbationOrderNotCanonical {
                index: index + 1,
            });
        }
    }
    Ok(windows)
}

fn compare_perturbations(a: ResourcePerturbation, b: ResourcePerturbation) -> Ordering {
    a.start_tick()
        .cmp(&b.start_tick())
        .then_with(|| a.end_tick_exclusive().cmp(&b.end_tick_exclusive()))
        .then_with(|| a.multiplier().total_cmp(&b.multiplier()))
        .then_with(|| a.delta().total_cmp(&b.delta()))
}

fn validate_outcome_protocol(
    outcome: CausalRescueOutcomeProtocolV1,
    perturbations: &[ResourcePerturbation],
    intervention_tick: u64,
    natural_end_behavior_next_tick: u64,
) -> Result<(), CausalRescueReplaySessionErrorV1> {
    let focal_index = usize::try_from(outcome.focal_perturbation_index).map_err(|_| {
        CausalRescueReplaySessionErrorV1::FocalPerturbationIndexOutOfBounds {
            index: outcome.focal_perturbation_index,
            len: perturbations.len(),
        }
    })?;
    let focal = perturbations.get(focal_index).copied().ok_or(
        CausalRescueReplaySessionErrorV1::FocalPerturbationIndexOutOfBounds {
            index: outcome.focal_perturbation_index,
            len: perturbations.len(),
        },
    )?;

    if outcome.baseline_lookback_ticks == 0 {
        return Err(CausalRescueReplaySessionErrorV1::ZeroBaselineLookback);
    }
    if focal
        .start_tick()
        .checked_sub(outcome.baseline_lookback_ticks)
        .is_none()
    {
        return Err(CausalRescueReplaySessionErrorV1::BaselineWindowUnderflow {
            focal_start_tick: focal.start_tick(),
            baseline_lookback_ticks: outcome.baseline_lookback_ticks,
        });
    }

    let recovery_fraction = outcome.recovery_fraction();
    if !recovery_fraction.is_finite() || !(0.0 < recovery_fraction && recovery_fraction <= 1.0) {
        return Err(CausalRescueReplaySessionErrorV1::InvalidRecoveryFraction {
            bits: outcome.recovery_fraction_bits,
        });
    }
    if outcome.evaluation_end_tick < focal.end_tick_exclusive() {
        return Err(
            CausalRescueReplaySessionErrorV1::EvaluationEndsBeforeFocalPerturbation {
                evaluation_end_tick: outcome.evaluation_end_tick,
                focal_end_tick_exclusive: focal.end_tick_exclusive(),
            },
        );
    }
    if outcome.evaluation_end_tick < intervention_tick {
        return Err(CausalRescueReplaySessionErrorV1::EvaluationEndsBeforeIntervention {
            evaluation_end_tick: outcome.evaluation_end_tick,
            intervention_tick,
        });
    }
    if outcome.evaluation_end_tick >= natural_end_behavior_next_tick {
        return Err(CausalRescueReplaySessionErrorV1::EvaluationOutsideNaturalEndpoint {
            evaluation_end_tick: outcome.evaluation_end_tick,
            natural_end_behavior_next_tick,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BirthMutationInterventionV1, EarthForcedEnvironment, EvolutionBirthPlanV1,
        GenesisEarthExecutionV1, InheritanceMode, ObservedStochasticTapeSubjectV1,
        OrganismConfig, PairingMode, PopulationConfig,
    };

    fn reproductive_cfg() -> PopulationConfig {
        PopulationConfig {
            death_energy_threshold: -1.0,
            reproduction_energy_threshold: 0.0,
            reproduction_energy_cost: 0.4,
            organism_cfg: OrganismConfig {
                social_enabled: true,
                ..OrganismConfig::default()
            },
            mutation_rate: 1.0,
            mutation_std: 0.08,
            inheritance: InheritanceMode::FromParent,
        }
    }

    fn natural_session_fixture() -> CausalRescueReplaySessionV1 {
        let mut execution = GenesisEarthExecutionV1::new(
            reproductive_cfg(),
            2,
            0xCA55_1001,
            PairingMode::Random,
            0xCA55_1002,
            EarthForcedEnvironment::earth_like(79.0),
        )
        .expect("fresh execution");

        let fork_ids = execution
            .population()
            .organisms
            .iter()
            .map(|organism| organism.id)
            .collect::<Vec<_>>();
        let fork_checkpoint = execution
            .checkpoint_execution()
            .expect("quiescent fork checkpoint");
        for _ in 0..3 {
            execution.step_social().expect("natural social tick");
        }
        let natural_end = execution
            .checkpoint_execution()
            .expect("natural endpoint checkpoint");

        let birth = natural_end
            .validated()
            .evidence()
            .lifecycle()
            .ledger()
            .records()
            .values()
            .find(|record| record.born_tick == 0 && record.reproductive_parent_id.is_some())
            .copied()
            .expect("tick-zero natural birth");
        let parent_id = birth.reproductive_parent_id.expect("birth parent");
        let source_id = birth.genome_source_id.expect("birth source");
        let parent_index = fork_ids
            .iter()
            .position(|&id| id == parent_id)
            .expect("parent in fork population");
        let source_index = fork_ids
            .iter()
            .position(|&id| id == source_id)
            .expect("source in fork population");
        let fork_records = fork_checkpoint
            .validated()
            .evidence()
            .lifecycle()
            .ledger()
            .records();
        let plan = EvolutionBirthPlanV1 {
            reproductive_parent_index: parent_index,
            reproductive_parent_id: parent_id,
            reproductive_parent_genome: fork_records[&parent_id].genome.to_genome(),
            genome_source_index: source_index,
            genome_source_id: source_id,
            genome_source_genome: fork_records[&source_id].genome.to_genome(),
            offspring_genome: birth.genome.to_genome(),
        };
        let intervention = BirthMutationInterventionV1::revert_all_changed_for_plan(0, &plan)
            .expect("mutation-enabled natural birth must change at least one trait");

        let tape = ObservedStochasticTapeV1 {
            subject: ObservedStochasticTapeSubjectV1 {
                start_behavior_next_tick: 0,
                start_lifecycle_next_sequence: fork_checkpoint
                    .validated()
                    .evidence()
                    .lifecycle()
                    .next_sequence(),
                end_behavior_next_tick: 3,
            },
            scalar_draws: vec![],
            scheduler_orders: vec![],
        };
        let perturbation = ResourcePerturbation::new(1, 1, 0.5, 0.0).unwrap();

        CausalRescueReplaySessionV1::new(
            0,
            fork_checkpoint.persisted().clone(),
            natural_end.persisted().clone(),
            tape,
            CounterfactualRandomFieldV1::new(0xCA55_2001),
            intervention,
            vec![ResourcePerturbationSnapshotV1::from_perturbation(
                perturbation,
            )],
            CausalRescueOutcomeProtocolV1::new(0, 1, 0.9, 2),
        )
    }

    #[test]
    fn session_binds_real_natural_birth_and_exact_lifecycle_extension() {
        let validated = natural_session_fixture().validate().expect("valid causal session");
        assert_eq!(validated.fork().evidence().behavior_next_tick(), 0);
        assert_eq!(validated.natural_end().evidence().behavior_next_tick(), 3);
        assert_eq!(validated.intervention().subject().tick(), 0);
        assert!(
            validated
                .natural_end()
                .evidence()
                .lifecycle()
                .ledger()
                .records()
                .contains_key(&validated.natural_offspring_id())
        );
    }

    #[test]
    fn tape_must_begin_at_exact_fork_behavior_and_lifecycle_boundary() {
        let mut session = natural_session_fixture();
        session.observed_stochastic_tape.subject.start_lifecycle_next_sequence += 1;
        assert!(matches!(
            session.validate(),
            Err(CausalRescueReplaySessionErrorV1::TapeStartLifecycleMismatch { .. })
        ));
    }

    #[test]
    fn session_requires_a_real_reversion_not_a_sham() {
        let session = natural_session_fixture();
        let subject = session.intervention.subject().clone();
        let encoded = serde_json::json!({
            "subject": subject,
            "kind": "Sham"
        });
        let sham: BirthMutationInterventionV1 = serde_json::from_value(encoded).unwrap();
        let forged = CausalRescueReplaySessionV1 {
            intervention: sham,
            ..session
        };
        assert!(matches!(
            forged.validate(),
            Err(CausalRescueReplaySessionErrorV1::InterventionMustRevertMutation)
        ));
    }

    #[test]
    fn perturbation_schedule_is_bit_preserving_and_canonical() {
        let first = ResourcePerturbation::new(4, 2, 0.5, -0.0).unwrap();
        let second = ResourcePerturbation::new(8, 1, 1.0, 0.125).unwrap();
        let snapshots = [
            ResourcePerturbationSnapshotV1::from_perturbation(first),
            ResourcePerturbationSnapshotV1::from_perturbation(second),
        ];
        let decoded = validate_perturbations(&snapshots).expect("canonical schedule");
        assert_eq!(decoded[0].delta().to_bits(), (-0.0f64).to_bits());

        let reversed = [snapshots[1], snapshots[0]];
        assert!(matches!(
            validate_perturbations(&reversed),
            Err(CausalRescueReplaySessionErrorV1::PerturbationOrderNotCanonical { .. })
        ));
    }

    #[test]
    fn outcome_window_must_fit_inside_the_observed_natural_endpoint() {
        let perturbation = ResourcePerturbation::new(1, 1, 0.5, 0.0).unwrap();
        let outcome = CausalRescueOutcomeProtocolV1::new(0, 1, 0.9, 3);
        assert!(matches!(
            validate_outcome_protocol(outcome, &[perturbation], 0, 3),
            Err(CausalRescueReplaySessionErrorV1::EvaluationOutsideNaturalEndpoint {
                evaluation_end_tick: 3,
                natural_end_behavior_next_tick: 3,
            })
        ));
    }
}
