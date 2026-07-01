// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-lean-bridge
//!
//! Emits externally-verifiable Lean 4 proof scripts from Symthaea's internal proof
//! representations. Two input paths:
//!
//! 1. `ProofResult` from `symthaea_core::hdc::logic_engine` — sequential
//!    `Vec<ProofStepLogic>` (natural deduction, resolution, DPLL).
//! 2. Z3 SMT-LIB2 witnesses emitted by `conjecture_engine::auto_prove_via_z3`.
//!
//! The emitted `.lean` files are designed to be checked by `lean4 --check`.
//! This crate does NOT invoke the Lean toolchain itself by default; see
//! `runner::check_with_lean4` for the optional subprocess wrapper used by
//! integration tests and the `prove_minif2f` / `prove_putnam` examples.
//!
//! ## Scope (Phase 1)
//!
//! - In scope: linear arithmetic, algebraic, combinatorial problems.
//! - Out of scope: Z3 `qfnra` / nlsat traces (nonlinear reals). These emit
//!   `sorry`-tagged Lean and are counted as failures, not false positives.
//!
//! ## Module layout
//!
//! - [`term`]: Lean 4 FOL term builder (Var, App, Lambda, Forall).
//! - [`tactic`]: `ProofStepLogic` → Lean tactic script.
//! - [`z3_to_lean`]: Z3 SMT-LIB2 → Lean proof script.
//! - [`runner`]: subprocess wrapper around `lean4 --check`.

pub mod bridge;
pub mod fol_ext_bridge; // Phase 2: FolFormulaExt → Lean proof script w/ Mathlib
pub mod minif2f_ingest; // Phase 4: miniF2F-v2 automated parser + translator
pub mod runner;
pub mod tactic;
pub mod term;
pub mod z3_to_lean;

/// Crate version — bumped when emitted Lean surface format changes.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_version_is_nonempty() {
        assert!(!CRATE_VERSION.is_empty());
    }
}
