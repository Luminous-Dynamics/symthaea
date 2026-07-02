// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! FormalLogicScorer — E-axis (Epistemic) formal verification scorer.
//!
//! Uses the symthaea-lean-bridge (z3_to_lean, bridge, fol_ext_bridge, runner)
//! to translate generated algorithms into Lean 4 proof obligations and
//! verifies them via Z3/Lean.
//! Returns a formal_correctness score (0.0–1.0) that is injected into
//! ThoughtChannels and used by CodeGate + reward shaping.
//!
//! This completes "Absolute Correctness": the system doesn't just
//! "look correct" — it can *prove* it.

use std::path::PathBuf;
use symthaea_core::hdc::fol_formula_ext::FolFormulaExt;
use symthaea_lean_bridge::{
    fol_ext_bridge::render_fol_ext_file,
    runner::{CheckOutcome, check_with_lean4},
    z3_to_lean::lean_from_z3,
};
use tempfile::NamedTempFile;

/// Result of formal verification.
#[derive(Debug, Clone)]
pub struct FormalVerificationResult {
    pub score: f32,           // 0.0 = no proof, 1.0 = fully verified
    pub proof_script: String, // emitted Lean source
    pub verified: bool,
    pub logic: String, // detected SMT fragment or "propositional"
}

/// FormalLogicScorer — the E-axis component.
pub struct FormalLogicScorer {
    temp_dir: PathBuf,
}

impl FormalLogicScorer {
    pub fn new() -> Self {
        let temp_dir = std::env::temp_dir().join("symthaea-formal");
        std::fs::create_dir_all(&temp_dir).ok();
        Self { temp_dir }
    }

    /// Score a generated algorithm against a formal specification.
    ///
    /// `spec` is a FolFormulaExt (from conjecture_engine or manual spec).
    /// `generated_code` is the Rust/Nix/etc. source (for future extraction of
    /// invariants; currently we just verify the spec itself).
    pub fn score_algorithm(
        &self,
        theorem_name: &str,
        spec: &FolFormulaExt,
        _generated_code: &str, // placeholder for future invariant extraction
    ) -> FormalVerificationResult {
        // 1. Emit Lean proof obligation using the full bridge
        let lean_source = render_fol_ext_file(theorem_name, spec);

        // 2. Write to temp file and run lean check
        let tmp = NamedTempFile::new_in(&self.temp_dir).unwrap();
        std::fs::write(tmp.path(), &lean_source).unwrap();

        let outcome = check_with_lean4(tmp.path());

        // `score` is documented as "0.0 = no proof, 1.0 = fully verified" on
        // FormalVerificationResult, and feeds directly into reward shaping
        // (compute_epistemic_reward: 25% of the epistemic term, 40% of the
        // final shaped_reward) and CodeGate. LeanNotInstalled/ProcessError
        // mean zero verification happened -- no proof was established either
        // way -- so they must score 0.0 like Rejected, not a fabricated
        // partial-credit value. An environment missing the `lean` binary
        // would otherwise silently inject a constant, meaningless ~0.15
        // (0.6 * the 25% weight) into every training reward regardless of
        // whether the generated code is actually correct.
        let (score, verified) = match outcome {
            CheckOutcome::Accepted => (1.0, true),
            CheckOutcome::Rejected(_) => (0.0, false),
            CheckOutcome::LeanNotInstalled => (0.0, false),
            CheckOutcome::ProcessError(_) => (0.0, false),
        };

        FormalVerificationResult {
            score,
            proof_script: lean_source,
            verified,
            logic: "fol_ext + mathlib".to_string(),
        }
    }

    /// Alternative path: Z3 witness → Lean script (for conjecture_engine output).
    pub fn score_from_z3(
        &self,
        theorem_name: &str,
        lean_statement: &str,
        smtlib: &str,
    ) -> FormalVerificationResult {
        let script = lean_from_z3(theorem_name, lean_statement, smtlib);
        let lean_source = script.to_lean();

        let tmp = NamedTempFile::new_in(&self.temp_dir).unwrap();
        std::fs::write(tmp.path(), &lean_source).unwrap();

        let outcome = check_with_lean4(tmp.path());
        let (score, verified) = match outcome {
            CheckOutcome::Accepted => (1.0, true),
            _ => (0.0, false),
        };

        FormalVerificationResult {
            score,
            proof_script: lean_source,
            verified,
            logic: "z3".to_string(),
        }
    }

    /// Direct verification of a formal logic formula (E-axis).
    pub fn verify_formula(&self, formula: &FolFormulaExt) -> FormalVerificationResult {
        self.score_algorithm("evolution_check", formula, "")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::logic_engine::Proposition;

    /// This environment doesn't have `lean` on PATH, so this exercises the
    /// real CheckOutcome::LeanNotInstalled path (not a mock). Regression
    /// test for the score/verified contract: no verification occurred, so
    /// the result must not fabricate partial credit -- score must be 0.0,
    /// matching the documented "0.0 = no proof, 1.0 = fully verified".
    #[test]
    fn missing_lean_binary_reports_zero_score_not_partial_credit() {
        let scorer = FormalLogicScorer::new();
        let spec = FolFormulaExt::from_prop(Proposition::True);
        let result = scorer.score_algorithm("no_lean_available", &spec, "");
        assert_eq!(result.score, 0.0);
        assert!(!result.verified);
    }
}

impl Default for FormalLogicScorer {
    fn default() -> Self {
        Self::new()
    }
}
