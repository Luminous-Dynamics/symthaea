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
//! the stronger event/schema/attempt/artifact/transformation relationships required before Forge
//! history is used for learning.
//!
//! # Authority boundary
//!
//! - Candidate failure and experiment-apparatus failure are distinct states.
//! - Each Forge search-loop attempt has an explicit content-addressed occurrence identity.
//! - No staged mutation can be committed through [`sandbox`].
//! - Pre-existing `.forge-orig` state is never auto-restored.
//! - A benchmark process spawn failure aborts the experiment; an executed-but-invalid candidate
//!   benchmark is a candidate evaluation rejection.
//! - Persistent output must resolve outside the canonical workspace and is create-new only.
//! - Forge's local benchmark remains a search heuristic, not replicated superiority evidence.
//! - No promotion, merge, activation, or runtime authority is provided here.

pub mod bundle;
pub mod bundle_verify;
pub mod candidate;
pub mod certificate;
pub mod fitness;
pub mod mutations;
pub mod observations;
pub mod sandbox;
pub mod search;
pub mod trace;
pub mod trial_semantics;

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
pub use fitness::{BenchmarkError, GateExecutionError};
pub use observations::ForgeObservationError;
pub use search::{
    run_search, run_search_recorded, ForgeConfig, SearchFailure, SearchOutcome, SearchRecord,
    SearchStats,
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
