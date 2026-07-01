// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! `ProofStepLogic` → Lean 4 tactic translation.
//!
//! `symthaea_core::hdc::logic_engine::ProofStepLogic` is a sequential
//! `Vec` of steps with `rule` and `justification` fields. Lean tactic mode
//! is itself sequential, so the mapping is structural. Week-3 milestone:
//! 1 miniF2F problem emits Lean that `lean4 --check` accepts.
//!
//! Phase 1 rules supported (from logic_engine.rs):
//! - Modus Ponens → `apply`, `exact`
//! - Modus Tollens → `apply`, contrapositive
//! - Hypothetical Syllogism → chained `apply`
//! - Disjunctive Syllogism → `cases`, `exact`
//! - Resolution (refutation) → `exact absurd`
//! - Axiom introduction → `intro h`
//!
//! Phase 2+ (not here): branching reconstruction for case-split proofs.

/// A single Lean 4 tactic line.
#[derive(Debug, Clone, PartialEq)]
pub enum LeanTactic {
    /// `intro h` — introduces a hypothesis.
    Intro(String),
    /// `apply lemma_name`.
    Apply(String),
    /// `exact term`.
    Exact(String),
    /// `cases h with ...` (simplified single-arg form).
    Cases(String),
    /// `rfl`.
    Rfl,
    /// `trivial`.
    Trivial,
    /// `sorry` — Phase 1 fallback for out-of-scope steps; emits a
    /// compile-but-unchecked Lean proof. These count as failures, not
    /// successes, in the external-verify gate.
    Sorry,
    /// Raw Lean tactic text (escape hatch for rules we haven't modeled yet).
    Raw(String),
}

impl LeanTactic {
    /// Render a single tactic as a line of Lean 4 source.
    pub fn to_lean(&self) -> String {
        match self {
            LeanTactic::Intro(h) => format!("intro {}", h),
            LeanTactic::Apply(name) => format!("apply {}", name),
            LeanTactic::Exact(term) => format!("exact {}", term),
            LeanTactic::Cases(h) => format!("cases {}", h),
            LeanTactic::Rfl => "rfl".to_string(),
            LeanTactic::Trivial => "trivial".to_string(),
            LeanTactic::Sorry => "sorry".to_string(),
            LeanTactic::Raw(s) => s.clone(),
        }
    }
}

/// A full Lean 4 proof script — a theorem statement plus a tactic block.
#[derive(Debug, Clone)]
pub struct LeanProofScript {
    /// `theorem name : statement :=` (statement rendered by caller).
    pub theorem_name: String,
    /// Statement in Lean 4 syntax (typically from `term::LeanTerm::to_lean`).
    pub statement: String,
    /// Tactic block.
    pub tactics: Vec<LeanTactic>,
}

impl LeanProofScript {
    /// Render the full `.lean` file body.
    pub fn to_lean(&self) -> String {
        let tactic_lines: Vec<String> = self
            .tactics
            .iter()
            .map(|t| format!("  {}", t.to_lean()))
            .collect();
        format!(
            "theorem {} : {} := by\n{}\n",
            self.theorem_name,
            self.statement,
            tactic_lines.join("\n")
        )
    }

    /// Returns `true` iff the script contains at least one `sorry` — used by
    /// the external-verify gate to distinguish "accepted by Lean" from
    /// "provably proved".
    pub fn contains_sorry(&self) -> bool {
        self.tactics.iter().any(|t| matches!(t, LeanTactic::Sorry))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_trivial_proof() {
        let script = LeanProofScript {
            theorem_name: "t_trivial".into(),
            statement: "True".into(),
            tactics: vec![LeanTactic::Trivial],
        };
        let expected = "theorem t_trivial : True := by\n  trivial\n";
        assert_eq!(script.to_lean(), expected);
    }

    #[test]
    fn sorry_detected() {
        let script = LeanProofScript {
            theorem_name: "t".into(),
            statement: "P".into(),
            tactics: vec![LeanTactic::Sorry],
        };
        assert!(script.contains_sorry());
    }

    #[test]
    fn intro_apply_exact_script() {
        let script = LeanProofScript {
            theorem_name: "t_mp".into(),
            statement: "(P → Q) → P → Q".into(),
            tactics: vec![
                LeanTactic::Intro("hpq".into()),
                LeanTactic::Intro("hp".into()),
                LeanTactic::Apply("hpq".into()),
                LeanTactic::Exact("hp".into()),
            ],
        };
        let rendered = script.to_lean();
        assert!(rendered.contains("intro hpq"));
        assert!(rendered.contains("intro hp"));
        assert!(rendered.contains("apply hpq"));
        assert!(rendered.contains("exact hp"));
        assert!(!script.contains_sorry());
    }
}
