// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-alife
//!
//! Artificial life for Symthaea, per `ALIFE_PLAN_2026-07-08.md`.
//!
//! An [`Organism`] is a Markov blanket ([`symthaea_fep::markov_blanket`]) wrapping one
//! [`symthaea_fep::ActiveInferenceAgent`]: it perceives an exogenous environment signal it
//! does not control, selects actions to minimize expected free energy, and pays real
//! metabolic energy for those actions. Deliberately built directly on `symthaea-fep` —
//! no HDC hypervectors, no `EmbodimentBridge`, no consciousness-loop coupling.
//!
//! ## Phase 0 scope
//!
//! One organism, one exogenous resource signal, two actions (forage / rest), a real energy
//! budget. Two claims, both ground-truth tested in `tests/phase0_ground_truth.rs`:
//!
//! 1. An organism that actually calls `perceive()` tracks the resource signal better than
//!    one that never does (tests that perception is doing real work, not theater).
//! 2. An organism whose actions come from `select_action()` regulates its energy better than
//!    one whose actions are uniform-random (tests that action selection is doing real work).
//!
//! Neither claim is about consciousness or Φ — see the plan doc's Non-goals.

pub mod agent_id;
pub mod causal_replay_session;
pub mod coalition;
pub mod counterfactual_randomness;
pub mod earth_forcing;
pub mod encounter;
pub mod encounter_snapshot;
pub mod environment;
pub mod events;
pub mod evolvability;
pub mod evolvability_comparison;
pub mod evolution_birth;
pub mod evolution_intervention;
pub mod evolution_rng;
pub mod evolutionary_history;
pub mod exact_sign_test;
pub mod genesis_evidence_checkpoint;
pub mod genesis_event_batch;
pub mod genesis_execution;
pub mod genesis_rolling_evidence;
pub mod genesis_social_runner;
pub mod genesis_tick;
pub mod genome;
pub mod hierarchy;
pub mod inheritance_source;
pub mod ledger;
pub mod lifecycle;
pub mod lifecycle_checkpoint;
pub mod lifecycle_recorder;
pub mod ma001;
pub mod ma001l;
pub mod ma001r;
pub mod matched_stochastic_replay;
pub mod metabolism;
pub mod observatory;
pub mod observed_stochastic_tape;
pub mod organism;
pub mod organism_seed;
pub mod organism_snapshot;
pub mod perturbation;
pub mod population;
pub mod population_config_snapshot;
pub mod population_snapshot;
pub mod predator_prey;
pub mod repeat_shock;
pub mod repeat_shock_did;
pub mod repeat_shock_relative;
pub mod types;

pub use agent_id::{
    AgentId, AgentIdAllocator, AgentIdAllocatorRestoreErrorV1, AgentIdAllocatorSnapshotV1,
};
pub use causal_replay_session::{
    CausalRescueOutcomeProtocolV1, CausalRescueReplaySessionErrorV1,
    CausalRescueReplaySessionV1, ResourcePerturbationSnapshotV1,
    ValidatedCausalRescueReplaySessionV1,
};
pub use coalition::{Coalition, detect_coalitions, detect_paying_coalitions};
pub use counterfactual_randomness::{
    CounterfactualDrawKeyV1, CounterfactualRandomDomainV1, CounterfactualRandomFieldV1,
};
pub use earth_forcing::EarthForcedEnvironment;
pub use encounter::{EncounterScheduler, PairingMode};
pub use encounter_snapshot::{
    EncounterFixedPartnerEntrySnapshotV1, EncounterPairingModeSnapshotV1,
    EncounterSchedulerSnapshotErrorV1, EncounterSchedulerSnapshotV1,
    ValidatedEncounterSchedulerSnapshotV1,
};
pub use environment::Environment;
pub use events::GenesisEvent;
pub use evolvability::{
    EvolvabilityError, RecoveryMetrics, analyze_recovery, analyze_recovery_through,
};
pub use evolvability_comparison::{
    RecoveryComparison, RecoveryComparisonError, RecoveryLatencyComparison, compare_recovery,
};
pub use evolution_birth::{
    EvolutionBirthPlanErrorV1, EvolutionBirthPlanV1, prepare_evolution_birth_v1,
};
pub use evolution_intervention::{
    AppliedBirthMutationInterventionV1, BirthMutationInterventionErrorV1,
    BirthMutationInterventionKindV1, BirthMutationInterventionReceiptV1,
    BirthMutationInterventionV1, EvolutionBirthSubjectV1, GenomeTraitV1,
    changed_traits_for_plan_v1,
};
pub use evolution_rng::{
    EvolutionRngRestoreErrorV1, EvolutionRngSnapshotV1, EvolutionRngStreamsV1,
    INHERITANCE_SOURCE_SEED_OFFSET_V1, LEGACY_MUTATION_SEED_OFFSET_V1,
};
pub use evolutionary_history::{
    EvolutionaryHistoryError, EvolutionaryHistoryReportV1, GenomeDeltaV1, LineageHistoryV1,
    MutationAncestryV1, analyze_evolutionary_history,
};
pub use exact_sign_test::{
    ExactPositiveSignTestV1, ExactSignTestErrorV1, MAX_EXACT_SIGN_TEST_NON_TIES,
    exact_positive_sign_test,
};
pub use genesis_evidence_checkpoint::{
    GenesisEvidenceCheckpointErrorV1, GenesisEvidenceCheckpointV1,
    ValidatedGenesisEvidenceCheckpointV1,
};
pub use genesis_event_batch::{
    GenesisTickBatchErrorV1, GenesisTickBatchV1, GenesisTickChunkSummaryV1,
    ValidatedGenesisTickBatchV1, validate_genesis_tick_chunk,
};
pub use genesis_execution::{
    GenesisEarthExecutionCapsuleErrorV1, GenesisEarthExecutionCapsuleV1,
    GenesisEarthExecutionCheckpointV1, GenesisEarthExecutionErrorV1, GenesisEarthExecutionV1,
    GenesisExecutionProtocolV1, ValidatedGenesisEarthExecutionCapsuleV1,
};
pub use genesis_rolling_evidence::{
    GenesisRollingEvidenceCheckpointV1, GenesisRollingEvidenceErrorV1,
    GenesisRollingEvidenceRunnerV1,
};
pub use genesis_social_runner::{GenesisSocialRunnerErrorV1, GenesisSocialRunnerV1};
pub use genesis_tick::{
    GenesisTickCursorErrorV1, GenesisTickCursorSnapshotV1, GenesisTickCursorV1,
    GenesisTickReservationV1,
};
pub use genome::Genome;
pub use hierarchy::HierarchicalStack;
pub use inheritance_source::{
    GenomeSourceSelectionErrorV1, GenomeSourceSelectionV1, select_genome_source_v1,
};
pub use ledger::{InteractionRecord, compress_for_observation};
pub use lifecycle::{
    AgentLifecycleRecordV1, ExtinctionEvidenceV1, GenomeEvidenceV1, LifecycleDeathCauseV1,
    LifecycleError, LifecycleEventV1, LifecycleLedgerV1, LifecycleTransitionV1,
    analyze_lifecycle_events,
};
pub use lifecycle_checkpoint::{
    LifecycleCheckpointErrorV1, LifecycleCheckpointV1, ValidatedLifecycleCheckpointV1,
};
pub use lifecycle_recorder::{LifecycleRecorderErrorV1, LifecycleRecorderV1};
pub use matched_stochastic_replay::{
    MatchedInheritanceSelectionV1, MatchedRandomOriginV1, MatchedSchedulerOrderV1,
    MatchedSeedV1, MatchedStochasticReplayErrorV1, MatchedStochasticReplayV1,
    MatchedUnitSampleV1,
};
pub use metabolism::{
    K_ALIFE_BOLTZMANN, landauer_minimum, prigogine_dissipation_cost, shannon_entropy_bits,
};
pub use observatory::{
    AgentTrajectorySummary, LineageSummary, ObservatoryError, ObservatoryReport,
    TransferEdgeSummary, analyze_genesis_events,
};
pub use observed_stochastic_tape::{
    ObservedSchedulerOrderKindV1, ObservedSchedulerOrderV1, ObservedStochasticDomainV1,
    ObservedStochasticDrawV1, ObservedStochasticKeyV1, ObservedStochasticTapeErrorV1,
    ObservedStochasticTapeSubjectV1, ObservedStochasticTapeV1, ObservedStochasticValueV1,
    ValidatedObservedStochasticTapeV1,
};
pub use organism::{Action, Organism, OrganismConfig, OrganismTick, PendingSocialLearning};
pub use organism_seed::{
    OrganismSeedAllocatorErrorV1, OrganismSeedAllocatorSnapshotV1, OrganismSeedAllocatorV1,
};
pub use organism_snapshot::{
    OrganismConfigSnapshotV1, OrganismLedgerEntrySnapshotV1, OrganismSnapshotErrorV1,
    OrganismSnapshotV1, ValidatedOrganismSnapshotV1,
};
pub use perturbation::{PerturbationError, PerturbationSchedule, ResourcePerturbation};
pub use population::{
    InheritanceMode, Population, PopulationConfig, StepSummary, resolve_pair_transfer,
};
pub use population_config_snapshot::{
    InheritanceModeSnapshotV1, PopulationConfigSnapshotErrorV1, PopulationConfigSnapshotV1,
    PopulationOrganismTemplateSnapshotV1, ValidatedPopulationConfigSnapshotV1,
};
pub use population_snapshot::{
    PopulationSnapshotErrorV1, PopulationSnapshotV1, ValidatedPopulationSnapshotV1,
};
pub use predator_prey::{PredatorPreyConfig, PredatorPreySim, PredatorPreyStep};
pub use repeat_shock::{
    RepeatShockErrorV1, RepeatShockLatencyV1, RepeatShockTransferV1,
    RepeatShockTransferVerdictV1, compare_repeated_shocks,
};
pub use repeat_shock_did::{
    RepeatedShockDidErrorV1, RepeatedShockDidV1, RepeatedShockDidVerdictV1,
    compare_repeated_shock_did,
};
pub use repeat_shock_relative::{
    RepeatedShockRelativeErrorV1, RepeatedShockRelativeV1, RepeatedShockRelativeVerdictV1,
    compare_repeated_shock_relative,
};
pub use types::BoundaryModulators;
