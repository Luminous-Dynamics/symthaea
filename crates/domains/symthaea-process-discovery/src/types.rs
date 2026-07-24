// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Core types for the generate -> verify -> select reaction search.

use symthaea_organic_chemistry::smiles::Molecule;

/// A candidate reaction: reactants transformed by a named template into products.
#[derive(Debug, Clone)]
pub struct ReactionCandidate {
    pub reactants: Vec<Molecule>,
    pub products: Vec<Molecule>,
    /// Name of the `ReactionTemplate` that produced this candidate.
    pub template: &'static str,
}

/// The result of a `ScopePolicy` check: allowed or not, always with a reason
/// (even "allowed" cases record why, so a certificate never has to guess).
#[derive(Debug, Clone)]
pub struct ScopeDecision {
    pub allowed: bool,
    pub reason: String,
}

impl ScopeDecision {
    pub fn allow(reason: impl Into<String>) -> Self {
        Self {
            allowed: true,
            reason: reason.into(),
        }
    }

    pub fn deny(reason: impl Into<String>) -> Self {
        Self {
            allowed: false,
            reason: reason.into(),
        }
    }
}

/// Configuration for one search run.
///
/// **No `policy` field here** -- an earlier version had one
/// (`ScopePolicyKind`), but `run_search()` also takes `&dyn ScopePolicy`
/// directly, so the enum field was never read; an external review caught
/// this as dead, misleading configuration. `ScopePolicy::name()` already
/// carries the identity a certificate needs, so the enum was removed
/// entirely rather than wired -- two sources of truth for the same fact is
/// exactly what caused the drift in the first place.
///
/// **No `seed` field either**: Phase 1's generator is a deterministic
/// enumeration, not randomized -- there is nothing to seed yet. Re-add this
/// when a real randomized generator exists and actually consumes it.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// SMILES of the seed reactants the search draws from.
    pub seed_reactants: Vec<String>,
    /// Caps total candidates attempted (enumeration order, not a literal
    /// generation count -- see `search.rs`'s module doc).
    pub candidate_cap: usize,
}
