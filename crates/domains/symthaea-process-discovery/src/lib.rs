// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # symthaea-process-discovery
//!
//! Generate -> verify -> select search over chemical reaction candidates,
//! with a pluggable [`policy::ScopePolicy`] guardrail. Phase 1 of
//! `symthaea/CHEMICAL_PROCESS_DISCOVERY_PLAN_2026-07-12.md`.
//!
//! Three `ScopePolicy` implementations are compared, not ranked in advance,
//! by `examples/policy_comparison.rs`:
//! - [`policy::AllowlistOnlyPolicy`] -- reactants and products both must be
//!   curated-library members. Can never produce a new molecule.
//! - [`policy::OpenWithHeuristicScreenPolicy`] -- open reactant space (in
//!   this phase, not yet exercised by the generator -- see `search.rs`),
//!   screened by generic structural hazard heuristics. Named for what it
//!   actually is: a heuristic screen, not a populated blocklist.
//! - [`policy::HybridAllowlistReactantsPolicy`] -- curated reactants, open
//!   products, same hazard screen as defense-in-depth.
//!
//! **No candidate is ever auto-applied.** `certificate::ProcessCertificate`
//! is the only output; nothing in this crate synthesizes, orders, or acts on
//! anything.
//!
//! Gate ordering (`oracle.rs`), cheapest first: [`validity`] (structural
//! sanity + element/charge conservation) -> [`policy::ScopePolicy`] -> the
//! materials-composition stability estimate, which is advisory telemetry
//! only, not a pass/fail gate (see `oracle.rs` for why).

pub mod aromaticity;
pub mod audit;
pub mod cache;
pub mod certificate;
pub mod corpus;
pub mod formula;
pub mod hazard_heuristics;
pub mod isomorphism;
pub mod metrics;
pub mod normalization;
pub mod oracle;
pub mod plugin;
pub mod policy;
pub mod pubchem;
pub mod rdkit;
pub mod search;
pub mod templates;
pub mod types;
pub mod validity;
