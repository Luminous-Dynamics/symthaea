// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Combined Epistemic Scorers for Code Intelligence

use crate::formal_logic_scorer::FormalLogicScorer;
use crate::moral_safety_scorer::compute_moral_safety;
use crate::narrative_maintainability_scorer::compute_narrative_maintainability;
use symthaea_core::hdc::fol_formula_ext::FolFormulaExt;

/// Simple idiomaticity heuristic for Rust.
pub fn compute_idiomaticity(code: &str) -> f32 {
    let modern_patterns = [
        "?",
        "thiserror",
        "anyhow",
        "tokio::",
        "async fn",
        "impl Iterator",
        "into_iter",
        "collect::<Vec<_>>",
        "map(|x|",
        "filter(|x|",
        "Result<",
        "Option<",
        "Cow<",
        "Arc<",
        "Rc<",
    ];
    let legacy_patterns = [
        ".unwrap()",
        ".expect(\"",
        "panic!(",
        "assert!(",
        "try!",
        "Box::new",
        "Vec::new()",
        "String::from",
        "format!(",
        "println!",
    ];

    let mut modern_score = 0.0f32;
    let mut legacy_score = 0.0f32;

    for p in modern_patterns {
        if code.contains(p) {
            modern_score += 0.15;
        }
    }
    for p in legacy_patterns {
        if code.contains(p) {
            legacy_score += 0.1;
        }
    }

    let base = 0.6 + modern_score - legacy_score;
    base.clamp(0.0, 1.0)
}

/// Main reward shaping function for compiler-grounded training.
/// Blends correctness with epistemic cube metrics and formal proofs.
pub fn compute_epistemic_reward(
    code: &str,
    correctness: f32,
    baseline: f32,
    spec: Option<&FolFormulaExt>,
) -> (f32, f32) {
    // (shaped_reward, epistemic_score)
    let moral = compute_moral_safety(code);
    let narrative = compute_narrative_maintainability(code);
    let idiomatic = compute_idiomaticity(code);

    // E-axis (Epistemic) Formal Verification
    let formal_score = if let Some(s) = spec {
        let formal_scorer = FormalLogicScorer::new();
        let result = formal_scorer.score_algorithm("generated_algorithm", s, code);
        result.score
    } else {
        0.5 // Default neutral if no spec provided
    };

    let epistemic = 0.30 * correctness
        + 0.25 * formal_score
        + 0.20 * moral
        + 0.15 * narrative
        + 0.10 * idiomatic;

    let shaped_reward = correctness * 0.60 + epistemic * 0.40;
    let advantage = shaped_reward - baseline;

    (advantage.clamp(-1.0, 1.0), epistemic)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_epistemic_reward() {
        let good_code = "/// documented\npub fn main() { let x = Some(1)?; }";
        let (reward, score) = compute_epistemic_reward(good_code, 1.0, 0.5, None);
        assert!(score > 0.6);
        assert!(reward > 0.0);
    }
}
