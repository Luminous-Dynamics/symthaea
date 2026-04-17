// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Z3 SMT-LIB2 witness → Lean 4 proof script.
//!
//! Phase 1 scope:
//! - Linear arithmetic (`QF_LIA`, `QF_LRA`) → Lean `omega` / `linarith`.
//! - Simple congruence (`UF`) → Lean `rfl` / `simp`.
//! - Pure propositional (`QF_UF`) → Lean `decide` / `tauto`.
//!
//! Out of scope (Phase 2+):
//! - `QF_NRA` nonlinear real arithmetic (Z3's nlsat trace doesn't map to
//!   Lean's kernel). These emit [`tactic::LeanTactic::Sorry`] and are
//!   counted as failures in the external-verify gate.

use crate::tactic::{LeanProofScript, LeanTactic};

/// Classification of the SMT logic used by a Z3 witness. Drives tactic choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmtLogic {
    /// Linear integer arithmetic → Lean `omega`.
    QfLia,
    /// Linear real arithmetic → Lean `linarith`.
    QfLra,
    /// Uninterpreted functions → Lean `rfl` or `simp`.
    QfUf,
    /// Propositional → Lean `decide`.
    Prop,
    /// Nonlinear real arithmetic → `sorry` in Phase 1.
    QfNra,
    /// Anything we don't classify yet.
    Unknown,
}

impl SmtLogic {
    /// Suggest the Lean tactic to close the goal for this logic.
    pub fn suggested_tactic(self) -> LeanTactic {
        match self {
            SmtLogic::QfLia => LeanTactic::Raw("omega".into()),
            SmtLogic::QfLra => LeanTactic::Raw("linarith".into()),
            SmtLogic::QfUf => LeanTactic::Rfl,
            SmtLogic::Prop => LeanTactic::Raw("decide".into()),
            SmtLogic::QfNra => LeanTactic::Sorry,
            SmtLogic::Unknown => LeanTactic::Sorry,
        }
    }
}

/// Parse the `(set-logic ...)` header from an SMT-LIB2 script. Very
/// forgiving — missing or malformed logic falls through to `Unknown`.
pub fn parse_set_logic(smtlib: &str) -> SmtLogic {
    for line in smtlib.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("(set-logic") {
            let payload = rest.trim().trim_end_matches(')').trim();
            return match payload {
                "QF_LIA" | "LIA" => SmtLogic::QfLia,
                "QF_LRA" | "LRA" => SmtLogic::QfLra,
                "QF_UF" | "UF" => SmtLogic::QfUf,
                "QF_NRA" | "NRA" => SmtLogic::QfNra,
                "QF_BV" => SmtLogic::Prop,
                _ => SmtLogic::Unknown,
            };
        }
    }
    SmtLogic::Unknown
}

/// Build a Lean proof script from a Z3 SMT-LIB2 problem statement.
///
/// The generated script assumes the goal is already expressed in Lean
/// syntax by the caller (we don't re-parse SMT-LIB here — that's the job of
/// `symthaea-core::hdc::conjecture_engine::expr_to_smtlib2` in reverse).
pub fn lean_from_z3(theorem_name: &str, lean_statement: &str, smtlib: &str) -> LeanProofScript {
    let logic = parse_set_logic(smtlib);
    LeanProofScript {
        theorem_name: theorem_name.to_string(),
        statement: lean_statement.to_string(),
        tactics: vec![logic.suggested_tactic()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_lia() {
        let smt = "(set-logic QF_LIA)\n(declare-const x Int)";
        assert_eq!(parse_set_logic(smt), SmtLogic::QfLia);
    }

    #[test]
    fn parses_nra() {
        let smt = "(set-logic QF_NRA)\n";
        assert_eq!(parse_set_logic(smt), SmtLogic::QfNra);
    }

    #[test]
    fn unknown_logic_falls_through() {
        let smt = "(declare-const x Int)";
        assert_eq!(parse_set_logic(smt), SmtLogic::Unknown);
    }

    #[test]
    fn nra_becomes_sorry() {
        let script = lean_from_z3("t", "x * x ≥ 0", "(set-logic QF_NRA)\n");
        assert!(script.contains_sorry());
    }

    #[test]
    fn lia_becomes_omega() {
        let script = lean_from_z3("t", "0 ≤ n + 1", "(set-logic QF_LIA)\n");
        assert!(!script.contains_sorry());
        assert!(script.to_lean().contains("omega"));
    }
}
