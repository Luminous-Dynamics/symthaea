// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! `symthaea-forge`: a bounded mutate -> verify -> measure -> select search loop over one target
//! function's Rust AST.
//!
//! [`mutations`] applies local structural AST mutations via `syn`; [`fitness`] gates compilation
//! and tests before any configured benchmark; [`sandbox`] provides an exclusive, fail-closed
//! temporary staging lease because Cargo must see candidate source on disk; [`search`] restores the
//! canonical source before retaining a candidate in memory; and [`certificate`] creates a
//! content-addressed human-reviewable survivor record.
//!
//! [`candidate`] is the authority-free bridge into `symthaea-algorithms`: it requires the caller to
//! supply semantic registry context and a baseline implementation. Forge-local traces become a
//! semantic discovery ledger only when the exact `DiscoveryRun` and complete observation archive
//! are supplied externally. [`bundle`] proves structural persisted-file integrity,
//! [`bundle_verify`] proves the existing cross-file semantics of that bundle, and
//! [`bundle_proposal_verify`] additionally rehydrates generator-local proposal evidence and proves
//! exact trace/proposal coverage. [`trial_semantics`] validates event/schema/attempt/artifact/
//! transformation consistency, [`trials`] preserves exact concrete rewrite instances, [`learning`]
//! provides exact-instance audit summaries, [`context_learning`] constrains cross-run aggregation,
//! [`family_learning`] introduces generator-scoped operator families,
//! [`family_context_learning`] permits family aggregation only across distinct exact-context runs,
//! [`sequence_learning`] reconstructs accepted family history at generation boundaries,
//! [`sequence_stats`] aggregates bounded-history conditional outcomes only across exact-context
//! repeated-search cohorts, [`corpus_split`] freezes outcome-independent train/validation/holdout
//! membership, [`proposal_exposure`] defines the complete opportunity evidence required before
//! those observational outcomes may inform future policy learning, [`proposal_recording`] binds
//! either live or persisted raw proposal evidence through one semantic conversion path,
//! [`proposal_trace`] retains the live draw as generator-local content-addressed evidence,
//! [`proposal_coverage`] proves that raw proposal evidence exactly covers the search trace,
//! [`proposal_qualification`] composes those layers while also binding no-op-only runs to the exact
//! semantic baseline implementation, [`proposal_dataset`] freezes one exact row per
//! `(attempt, policy family)` before any estimator or learned policy exists, [`proposal_corpus`]
//! binds those tables to the frozen corpus split while exposing separate training/validation
//! datasets and an identity-only holdout seal, [`proposal_history`] attaches the validated accepted
//! state entering each generation to every proposal row, including no-op attempts,
//! [`proposal_endpoints`] freezes outcome observability so missing/censored labels cannot be
//! silently rewritten as failures, [`proposal_study`] freezes the feature/target/estimator/
//! validation contract before any model can be fitted, crate-private model machinery binds the
//! frozen model and generic receipt internals, [`proposal_support`] proves the role-isolated training
//! corpus satisfies every precommitted family-support threshold before a fit permit can exist,
//! [`proposal_validation_coverage`] separately proves the validation corpus has enough actually
//! observed labels before scoring, [`proposal_validation_blind`] exposes only label-blind
//! validation targets/predictions, the crate-private deterministic scorer joins frozen predictions
//! to labels only during scoring, and [`proposal_validation_chain`] is the only public
//! validation/holdout receipt path, requiring that exact blind row-reconstructable evidence.
//!
//! # Authority boundary
//!
//! - Candidate failure and experiment-apparatus failure are distinct states.
//! - Each Forge search-loop attempt has an explicit content-addressed occurrence identity.
//! - Concrete transformation identity and reusable transformation-family identity are separate.
//! - Sequence memory conditions siblings on the state entering their generation; a winner changes
//!   state only for later generations.
//! - Proposal history for every row is derived only from accepted steps in strictly earlier
//!   generations, so no sibling can observe the winner of its own generation as prior state.
//! - History order is explicit and content-addressed; first-order and higher-order statistics do not
//!   silently mix.
//! - Corpus role ranking depends only on a frozen salt and semantic `DiscoveryRun` identity, never
//!   on outcome-bearing batch/cohort identity.
//! - Proposal exposure records exact eligible-site counts and selected pair; raw family success is
//!   not eligible for policy learning without complete exposure qualification.
//! - Proposal telemetry and mutation behavior share one sampler; semantic binding never resamples
//!   or reapplies a mutation.
//! - Raw proposal archives record generator behavior faithfully even when a later semantic policy
//!   refuses to qualify that behavior for learning.
//! - Proposal-stage traces and raw proposal archives must provide one-to-one coverage both before
//!   return from search and after persistent bundle rehydration.
//! - No-op-only proposal histories still have to bind the exact declared baseline artifact.
//! - Proposal observation tables retain zero-opportunity and unselected families; absence is never
//!   silently converted into failure or success evidence.
//! - Unselected family outcomes remain counterfactual-unobserved; apparatus-interrupted downstream
//!   outcomes remain censored rather than being encoded as negative labels.
//! - Training and validation proposal tables are exposed through different role-checked types.
//! - Holdout is identity-only in v1; this crate provides no convenience API that exposes holdout
//!   observation rows before a future frozen-model evaluation boundary exists.
//! - Proposal-study feature schemas can name only pre-decision observables; post-decision leakage
//!   fields are intentionally absent from the v1 feature vocabulary.
//! - The primary validation metric, estimator implementation/configuration, training seed, support
//!   thresholds, endpoint, exposure unit, and corpus identities are frozen before fitting.
//! - Training support is counted only from observed labels; censored and counterfactual outcomes
//!   remain visible but cannot satisfy label-support thresholds or mint a fit permit.
//! - Validation coverage is independently precommitted and counted only from observed labels across
//!   distinct validation runs; an under-observed validation slice cannot mint a score permit.
//! - Public validation prediction generation covers every proposal row through label-blind targets;
//!   whether a row is observed/censored/counterfactual is joined only after predictions are frozen.
//! - Generic aggregate validation/holdout constructors remain crate-internal; the public receipt
//!   path requires the exact blind target set, blind prediction set, and deterministic Brier receipt.
//! - Deterministic validation v1 rejects report-only metrics until they also have deterministic,
//!   row-reconstructable scorers.
//! - Validation acceptance threshold/scorer identity is precommitted before a model artifact can be
//!   frozen; failed validation cannot mint even an identity-only holdout evaluation permit.
//! - Holdout permits still expose no holdout observations and grant no search or deployment authority.
//! - No staged mutation can be committed through [`sandbox`].
//! - Pre-existing `.forge-orig` state is never auto-restored.
//! - A benchmark process spawn failure aborts the experiment; an executed-but-invalid candidate
//!   benchmark is a candidate evaluation rejection.
//! - Persistent output must resolve outside the canonical workspace and is create-new only.
//! - Forge's local benchmark remains a search heuristic, not replicated superiority evidence.
//! - Learning tables, sequences, conditional statistics, corpus partitions, exposure receipts, and
//!   proposal datasets are descriptive research records only.
//! - No promotion, merge, activation, model fitting, inverse-propensity estimator, search-policy
//!   mutation, or runtime authority is provided here.

pub mod bundle;
pub mod bundle_proposal_verify;
pub mod bundle_verify;
pub mod candidate;
pub mod certificate;
pub mod context_learning;
pub mod corpus_split;
pub mod family_context_learning;
pub mod family_learning;
pub mod fitness;
pub mod learning;
pub mod mutations;
pub mod observations;
pub mod proposal_corpus;
pub mod proposal_coverage;
pub mod proposal_dataset;
pub mod proposal_endpoints;
pub mod proposal_exposure;
pub mod proposal_history;
mod proposal_model;
pub mod proposal_qualification;
pub mod proposal_recording;
pub mod proposal_study;
pub mod proposal_support;
pub mod proposal_trace;
pub mod proposal_validation_blind;
pub mod proposal_validation_chain;
pub mod proposal_validation_coverage;
mod proposal_validation_predictions;
pub mod sandbox;
pub mod search;
pub mod sequence_learning;
pub mod sequence_stats;
pub mod trace;
pub mod trial_semantics;
pub mod trials;

pub use bundle::{
    read_completed_manifest, BundleError, ForgeBundleManifest, ForgeBundleOutcome, ABORT_FILE,
    CANDIDATE_FILE, CERTIFICATE_FILE, MANIFEST_FILE, OBSERVATIONS_FILE, RAW_PROPOSALS_FILE,
    REPORT_FILE, TRACE_FILE,
};
pub use bundle_proposal_verify::{
    read_completed_proposal_bundle, BundleProposalSemanticError, VerifiedForgeProposalBundle,
};
pub use bundle_verify::{
    read_completed_bundle, BundleSemanticError, ForgeAbortRecord, ForgeAbortStats,
    VerifiedForgeBundle,
};
pub use candidate::{ledger_from_forge_trace, proposal_from_forge, ForgeProposalError};
pub use certificate::{CertificateError, ForgeCandidate, ForgeCertificate};
pub use context_learning::{
    ContextBoundForgeBatch, ContextualTransformationStats, ExactContextForgeCohort,
    ExactContextTransformationTable, ForgeContextLearningError,
};
pub use corpus_split::{
    ForgeCorpusAssignment, ForgeCorpusRole, ForgeCorpusSplitError, ForgeCorpusSplitSpec,
    ForgeSequenceCorpusSplit,
};
pub use family_context_learning::{
    ContextBoundForgeFamilyBatch, ContextualForgeFamilyStats, ExactContextForgeFamilyCohort,
    ExactContextForgeFamilyOutcomeTable, ForgeFamilyContextError,
};
pub use family_learning::{
    ForgeFamilyLearningError, ForgeFamilyOutcomeStats, ForgeFamilyOutcomeTable, ForgeFamilyTrial,
    ForgeFamilyTrialSet, ForgeTransformationFamilyId,
};
pub use fitness::{BenchmarkError, GateExecutionError};
pub use learning::{
    ForgeLearningError, ForgeTrialBatch, TransformationOutcomeStats, TransformationOutcomeTable,
};
pub use observations::ForgeObservationError;
pub use proposal_corpus::{
    ForgeProposalCorpusError, ForgeProposalCorpusManifest, ForgeProposalCorpusMember,
    ForgeProposalHoldoutMember, ForgeProposalHoldoutSeal, ForgeProposalTrainingSet,
    ForgeProposalValidationSet,
};
pub use proposal_coverage::{validate_forge_raw_proposal_coverage, ForgeProposalCoverageError};
pub use proposal_dataset::{
    ForgeProposalDatasetError, ForgeProposalObservationRow, ForgeProposalObservationTable,
    ForgeProposalRowDecision, ForgeProposalRowOutcome,
};
pub use proposal_endpoints::{
    ForgeProposalEndpoint, ForgeProposalEndpointError, ForgeProposalEndpointRecord,
    ForgeProposalEndpointTable, ForgeProposalEndpointValue,
};
pub use proposal_exposure::{
    ForgeFamilyOpportunity, ForgeProposalDecision, ForgeProposalExposure,
    ForgeProposalExposureArchive, ForgeProposalExposureError, ForgeProposalLearningQualification,
    ForgeProposalPolicy, ForgeProposalRule, ForgeProposalSelection,
};
pub use proposal_history::{
    ForgeConditionedProposalObservationRow, ForgeConditionedProposalObservationTable,
    ForgeProposalGenerationState, ForgeProposalHistoryError,
};
pub use proposal_model::{
    ForgeProposalFrozenModel, ForgeProposalMetricScore, ForgeProposalModelError,
    ForgeProposalValidationGateSpec,
};
pub use proposal_qualification::{
    ForgeProposalQualificationError, ForgeQualifiedProposalEvidence,
};
pub use proposal_recording::{
    exposure_from_raw_record, exposure_from_recorded_mutation, proposal_policy_for_mutator,
    ForgeProposalRecordingError,
};
pub use proposal_study::{
    ForgeProposalEstimatorSpec, ForgeProposalEvaluationSpec, ForgeProposalExposureUnit,
    ForgeProposalFeature, ForgeProposalFeatureSchema, ForgeProposalMetric,
    ForgeProposalMissingnessPolicy, ForgeProposalStudyError, ForgeProposalStudySpec,
    ForgeProposalSupportSpec,
};
pub use proposal_support::{
    ForgeProposalFamilySupport, ForgeProposalFitPermit, ForgeProposalSupportError,
    ForgeProposalSupportReceipt,
};
pub use proposal_trace::{
    ForgeRawMutationEffect, ForgeRawOpportunity, ForgeRawProposalArchive, ForgeRawProposalDecision,
    ForgeRawProposalError, ForgeRawProposalRecord, ForgeRawSelection,
};
pub use proposal_validation_blind::{
    ForgeProposalBlindDeterministicBrierReceipt, ForgeProposalBlindValidationError,
    ForgeProposalBlindValidationPrediction, ForgeProposalBlindValidationPredictionSet,
    ForgeProposalValidationTarget, ForgeProposalValidationTargetSet,
};
pub use proposal_validation_chain::{
    ForgeProposalDeterministicHoldoutPermit, ForgeProposalDeterministicValidationError,
    ForgeProposalDeterministicValidationReceipt,
};
pub use proposal_validation_coverage::{
    ForgeProposalValidationCoverageError, ForgeProposalValidationCoverageReceipt,
    ForgeProposalValidationCoverageSpec, ForgeProposalValidationFamilyCoverage,
    ForgeProposalValidationScorePermit,
};
pub use proposal_validation_predictions::{
    forge_deterministic_brier_configuration_id, forge_deterministic_brier_scorer_id,
    ForgeProposalValidationPredictionError,
};
pub use search::{
    run_search, run_search_recorded, ForgeConfig, SearchFailure, SearchOutcome, SearchRecord,
    SearchStats,
};
pub use sequence_learning::{
    ForgeAcceptedFamilyStep, ForgeConditionedFamilyTrial, ForgeFamilyHistory,
    ForgeSearchFamilySequence, ForgeSequenceLearningError,
};
pub use sequence_stats::{
    ContextBoundForgeSequenceBatch, ContextualFamilyTransitionStats,
    ExactContextConditionalFamilyTable, ExactContextForgeSequenceCohort,
    ForgeConditionalFamilyKey, ForgeHistoryOrder, ForgeSequenceStatsError,
};
pub use trace::{
    validate_forge_trace, validate_forge_trace_observations, ForgeAttemptId, ForgeTraceError,
    ForgeTraceEvent,
};
pub use trial_semantics::{
    validate_forge_trial_semantics, ForgeTrialSemanticError, BENCHMARK_FAILURE_SCHEMA,
    CANDIDATE_DECISION_SCHEMA, CANDIDATE_GENERATED_SCHEMA, CORRECTNESS_GATES_SCHEMA,
    NO_CANDIDATE_SCHEMA, SEARCH_ABORTED_SCHEMA, SEARCH_SUMMARY_SCHEMA, SELECTION_SCHEMA,
};
pub use trials::{
    extract_transformation_trials, ForgeTrialError, ForgeTrialOutcome, ForgeTrialSet,
    TransformationTrial,
};
