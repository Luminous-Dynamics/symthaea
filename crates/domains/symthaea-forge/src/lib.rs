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
//! supply the semantic `ProblemSpec`/`AlgorithmRecord` context and a baseline implementation, then
//! converts the exact Forge survivor into a generic `CandidateProposal`. It also replays Forge's
//! generator-local [`trace`] into a semantic hash-chained discovery ledger only when the exact
//! `DiscoveryRun` is supplied externally. [`bundle`] makes persistent CLI output complete only when
//! a final content-addressed manifest validates the exact files on disk.
//!
//! # Authority boundary
//!
//! - No staged mutation can be committed through [`sandbox`]; the staging type has no `commit()`
//!   operation and explicit restoration errors abort search.
//! - Pre-existing `.forge-orig` state is never auto-restored because it might belong to another
//!   live Forge process. Search stops for explicit recovery.
//! - A configured benchmark failure is an evaluation failure, not correctness-only mode.
//! - Persistent CLI output must resolve outside the canonical workspace and existing evidence
//!   files are not overwritten.
//! - Forge's single-run benchmark score is a search heuristic, not replicated evidence of
//!   superiority and not a production-promotion decision.
//! - No general-purpose program synthesis or autonomous production activation is provided here.

pub mod bundle;
pub mod candidate;
pub mod certificate;
pub mod fitness;
pub mod mutations;
pub mod sandbox;
pub mod search;
pub mod trace;

pub use bundle::{
    read_completed_manifest, BundleError, ForgeBundleManifest, ForgeBundleOutcome, CANDIDATE_FILE,
    CERTIFICATE_FILE, MANIFEST_FILE, REPORT_FILE, TRACE_FILE,
};
pub use candidate::{ledger_from_forge_trace, proposal_from_forge, ForgeProposalError};
pub use certificate::{CertificateError, ForgeCandidate, ForgeCertificate};
pub use search::{run_search, ForgeConfig, SearchOutcome};
pub use trace::{validate_forge_trace, ForgeTraceError, ForgeTraceEvent};
