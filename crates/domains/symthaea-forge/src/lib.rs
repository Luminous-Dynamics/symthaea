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
//! are supplied externally. [`bundle`] proves structural persisted-file integrity, while
//! [`bundle_verify`] proves the cross-file semantics of that bundle. [`trial_semantics`] validates
//! event/schema/attempt/artifact/transformation consistency, [`trials`] preserves exact concrete
//! rewrite instances, [`learning`] provides exact-instance audit summaries, [`context_learning`]
//! constrains cross-run aggregation, [`family_learning`] introduces generator-scoped operator
//! families, [`family_context_learning`] permits family aggregation only across distinct exact-
//! context runs, [`sequence_learning`] reconstructs accepted family history at generation
//! boundaries, and [`sequence_stats`] aggregates bounded-history conditional outcomes only across
//! exact-context repeated-search cohorts.
//!
//! # Authority boundary
//!
//! - Candidate failure and experiment-apparatus failure are distinct states.
//! - Each Forge search-loop attempt has an explicit content-addressed occurrence identity.
//! - Concrete transformation identity and reusable transformation-family identity are separate.
//! - Sequence memory conditions siblings on the state entering their generation; a winner changes
//!   state only for later generations.
//! - History order is explicit and content-addressed; first-order and higher-order statistics do not
//!   silently mix.
//! - No staged mutation can be committed through [`sandbox`].
//! - Pre-existing `.forge-orig` state is never auto-restored.
//! - A benchmark process spawn failure aborts the experiment; an executed-but-invalid candidate
//!   benchmark is a candidate evaluation rejection.
//! - Persistent output must resolve outside the canonical workspace and is create-new only.
//! - Forge's local benchmark remains a search heuristic, not replicated superiority evidence.
//! - Learning tables, sequences, and conditional statistics are descriptive search memory only.
//! - No promotion, merge, activation, search-policy mutation, or runtime authority is provided here.

pub mod bundle;
pub mod bundle_verify;
pub mod candidate;
pub mod certificate;
pub mod context_learning;
pub mod family_context_learning;
pub mod family_learning;
pub mod fitness;
pub mod learning;
pub mod mutations;
pub mod observations;
pub mod sandbox;
pub mod search;
pub mod sequence_learning;
pub mod sequence_stats;
pub mod trace;
pub mod trial_semantics;
pub mod trials;

pub use bundle::{
    read_completed_manifest, BundleError, ForgeBundleManifest, ForgeBundleOutcome, ABORT_FILE,
    CANDIDATE_FILE, CERTIFICATE_FILE, MANIFEST_FILE, OBSERVATIONS_FILE, REPORT_FILE, TRACE_FILE,
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
