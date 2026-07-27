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
    ///
    /// `Intro`/`Apply`/`Cases` hold bare identifiers and are sanitized as
    /// such; `Exact` holds a term expression (may be a compound proof term
    /// like `(False.elim h)`, not just a name) and is sanitized as
    /// statement-like syntax. `Raw` is intentionally unsanitized -- it is
    /// the crate's internal escape hatch for hand-assembled tactic text
    /// (e.g. the `first | ... | ...` cascade in `fol_ext_bridge`), which
    /// callers must build only from already-sanitized pieces.
    pub fn to_lean(&self) -> String {
        use crate::sanitize::{sanitize_ident, sanitize_statement};
        match self {
            LeanTactic::Intro(h) => format!("intro {}", sanitize_ident(h)),
            LeanTactic::Apply(name) => format!("apply {}", sanitize_ident(name)),
            LeanTactic::Exact(term) => format!("exact {}", sanitize_statement(term)),
            LeanTactic::Cases(h) => format!("cases {}", sanitize_ident(h)),
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
    ///
    /// `theorem_name`/`statement` are public fields constructible with
    /// arbitrary strings (see e.g. `z3_to_lean::lean_from_z3`, which passes
    /// through a caller-supplied Lean statement verbatim), so this is the
    /// choke point that guarantees neither can break out of the emitted
    /// `theorem ... := by` declaration into a new top-level Lean command.
    pub fn to_lean(&self) -> String {
        use crate::sanitize::{sanitize_ident, sanitize_statement};
        let tactic_lines: Vec<String> = self
            .tactics
            .iter()
            .map(|t| format!("  {}", t.to_lean()))
            .collect();
        format!(
            "theorem {} : {} := by\n{}\n",
            sanitize_ident(&self.theorem_name),
            sanitize_statement(&self.statement),
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

    /// Regression test for the Lean-source-injection finding: since
    /// `theorem_name`/`statement` are public fields constructible with any
    /// string (e.g. by `z3_to_lean::lean_from_z3`, which passes a
    /// caller-supplied Lean statement through verbatim), a crafted value
    /// containing a newline + top-level command must not survive
    /// `to_lean()` and break out of the `theorem ... := by` declaration.
    ///
    /// `theorem_name` goes through `sanitize_ident` (strips every unsafe
    /// character, including `#`) so `#eval` can't survive at all there.
    /// `statement` goes through `sanitize_statement`, which only neutralizes
    /// control characters and otherwise preserves printable content by
    /// design (statements are meant to hold real Lean syntax) -- so the
    /// right invariant for `statement` isn't "no `#eval` substring", it's
    /// "the injected newline can no longer start a new top-level line".
    #[test]
    fn malicious_theorem_name_and_statement_cannot_inject_lean_commands() {
        let malicious = LeanProofScript {
            theorem_name: "t\nend\n#eval IO.Process.run { cmd := \"sh\" }".into(),
            statement: "True\nend\n#eval 2".into(),
            tactics: vec![LeanTactic::Trivial],
        };
        let rendered = malicious.to_lean();

        // theorem_name: sanitize_ident strips '#' entirely, so no trace of
        // the injected command can survive anywhere in the identifier.
        assert!(
            !rendered.contains("#eval IO.Process.run"),
            "injected #eval must be neutralized in theorem_name, got: {}",
            rendered
        );
        assert!(rendered.starts_with("theorem "));
        assert!(rendered.contains(":= by"));

        // statement: content is preserved, but the newline that would let
        // "end"/"#eval 2" start a fresh top-level line must be gone --
        // compare against an equivalent script whose statement has no
        // embedded newline, and assert the line count is identical (the
        // malicious newline didn't add a line).
        let benign = LeanProofScript {
            theorem_name: "t".into(),
            statement: "True end #eval 2".into(),
            tactics: vec![LeanTactic::Trivial],
        };
        assert_eq!(
            malicious.to_lean().lines().count(),
            benign.to_lean().lines().count(),
            "a malicious statement must not add lines to the rendered file"
        );
        assert!(
            !rendered
                .lines()
                .any(|l| l.trim() == "end" || l.trim() == "#eval 2"),
            "injected content must not land on its own top-level line, got: {}",
            rendered
        );
    }
}
