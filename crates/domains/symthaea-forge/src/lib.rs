// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! `symthaea-forge`: a bounded mutate -> verify -> measure -> select search loop over one target
//! function's Rust AST.
//!
//! [`mutations`] applies local structural AST mutations via `syn`; [`fitness`] gates compilation
//! and tests before any configured benchmark; [`sandbox`] provides an exclusive, fail-closed
//! temporary staging lease because Cargo must see candidate source on disk; [`search`] restores the
//! canonical source before retaining a candidate in memory; and [`certificate`] creates a
//! human-reviewable report of the search survivor.
//!
//! # Authority boundary
//!
//! - No staged mutation can be committed through [`sandbox`]; the staging type has no `commit()`
//!   operation and explicit restoration errors abort search.
//! - Pre-existing `.forge-orig` state is never auto-restored because it might belong to another
//!   live Forge process. Search stops for explicit recovery.
//! - A configured benchmark failure is an evaluation failure, not correctness-only mode.
//! - Persistent CLI output is certificate/report evidence only and must resolve outside the
//!   canonical workspace. Existing output files are not overwritten.
//! - Forge's single-run benchmark score is a search heuristic, not replicated evidence of
//!   superiority and not a production-promotion decision.
//! - No general-purpose program synthesis or autonomous production activation is provided here.
//!
//! The evidence-first `symthaea-algorithms` stack is the intended destination for typed candidate
//! identity, ordered transformation lineage, reproducible evaluation receipts, repeatability,
//! robust comparison, and any future separately reviewed promotion policy.

pub mod certificate;
pub mod fitness;
pub mod mutations;
pub mod sandbox;
pub mod search;

pub use certificate::ForgeCertificate;
pub use search::{ForgeConfig, SearchOutcome, run_search};
