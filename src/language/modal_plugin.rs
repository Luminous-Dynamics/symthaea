// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Modal-logic domain plugin — validity of a modal formula in K/T/S4/S5.
//!
//! Answers questions of the form "is `[]p -> p` valid in T?" deterministically
//! by parsing the formula (`symthaea_modal::parse`) and running the real
//! Kripke-frame validity checker (`is_valid`). This exposes the *machinery* of
//! metaphysical modality (what follows necessarily from what) — not answers to
//! substantive metaphysical questions.

use crate::language::domain_plugin::{ComputedResult, DomainPlugin, Entity};
use crate::mind::structured_thought::{ETier, EpistemicCube, MTier, NTier};
use symthaea_modal::{System, is_valid, parse};

pub struct ModalDomainPlugin;

/// Leading prose words to strip before the formula proper.
const STOP: &[&str] = &[
    "is",
    "the",
    "formula",
    "does",
    "it",
    "true",
    "that",
    "prove",
    "a",
    "an",
    "if",
    "whether",
    "would",
    "be",
    "expression",
    "statement",
    "sentence",
    "axiom",
];

impl ModalDomainPlugin {
    fn has_cue(text: &str) -> bool {
        let t = text.to_lowercase();
        t.contains("valid in")
            && (t.contains("[]")
                || t.contains("<>")
                || t.contains('□')
                || t.contains('◇')
                || t.contains("box")
                || t.contains("necess")
                || t.contains("possib")
                || t.contains("diamond")
                || t.contains("modal"))
    }

    /// Drop leading filler words so the remainder begins at the formula.
    fn strip_prose(before: &str) -> String {
        let mut words: Vec<&str> = before.split_whitespace().collect();
        while let Some(&w) = words.first() {
            let lw: String = w
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();
            if STOP.contains(&lw.as_str()) {
                words.remove(0);
            } else {
                break;
            }
        }
        words.join(" ")
    }
}

impl DomainPlugin for ModalDomainPlugin {
    fn domain_name(&self) -> &str {
        "modal_logic"
    }

    fn extract_entities(&self, _text: &str) -> Vec<Entity> {
        Vec::new()
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        if Self::has_cue(topic) { 0.9 } else { 0.1 }
    }

    fn vocabulary(&self) -> Vec<String> {
        [
            "modal",
            "necessity",
            "possibility",
            "valid",
            "kripke",
            "axiom",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn compute(&self, input: &str, _entities: &[Entity]) -> Option<ComputedResult> {
        if !Self::has_cue(input) {
            return None;
        }
        let lower = input.to_lowercase();
        let idx = lower.find("valid in")?;
        let before = &input[..idx];
        let after = &input[idx + "valid in".len()..];

        // System: the first K/T/S4/S5 token after "valid in [system]".
        let sys_region = after.to_lowercase().replace("system", " ");
        let sys_tok = sys_region
            .split(|c: char| !c.is_alphanumeric())
            .find(|t| !t.is_empty())?;
        let system: System = sys_tok.parse().ok()?;

        // Formula: strip the leading prose, then parse.
        let formula_src = Self::strip_prose(before);
        let formula = parse(&formula_src).ok()?;

        let valid = is_valid(&formula, system);
        Some(ComputedResult {
            answer: if valid {
                format!(
                    "In modal system {sys_tok}, the formula `{}` is VALID \
                     (true in every world of every {sys_tok}-frame).",
                    formula_src.trim()
                )
            } else {
                format!(
                    "In modal system {sys_tok}, the formula `{}` is NOT valid \
                     (a countermodel exists over a {sys_tok}-frame).",
                    formula_src.trim()
                )
            },
            cube: EpistemicCube {
                e: ETier::E4,
                n: NTier::N3,
                m: MTier::M3,
                h: None,
            },
            psi: 0.0,
            proof_available: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_axiom_valid_in_t_not_k() {
        let p = ModalDomainPlugin;
        let in_t = p
            .compute("is the formula []p -> p valid in T?", &[])
            .unwrap();
        assert!(in_t.answer.contains("VALID"), "{}", in_t.answer);

        let in_k = p.compute("is []p -> p valid in K?", &[]).unwrap();
        assert!(in_k.answer.contains("NOT valid"), "{}", in_k.answer);
    }

    #[test]
    fn four_axiom_valid_in_s4() {
        let p = ModalDomainPlugin;
        let r = p
            .compute("is the axiom []p -> [][]p valid in S4?", &[])
            .unwrap();
        assert!(r.answer.contains("VALID"), "{}", r.answer);
    }

    #[test]
    fn unicode_form() {
        let p = ModalDomainPlugin;
        let r = p.compute("is □p → p valid in T?", &[]).unwrap();
        assert!(r.answer.contains("VALID"), "{}", r.answer);
    }

    #[test]
    fn no_cue_none() {
        let p = ModalDomainPlugin;
        // "valid in" without any modal marker must not fire.
        assert!(p.compute("is this contract valid in Texas?", &[]).is_none());
    }
}
