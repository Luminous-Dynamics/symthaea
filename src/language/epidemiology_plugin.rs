// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Epidemiology domain plugin — deterministic SIR epidemic metrics.
//!
//! Wires `symthaea-epidemiology` into the facade: given transmission (β) and
//! recovery (γ) rates in the query, computes R₀, the herd-immunity threshold,
//! and the final epidemic size, bypassing the LLM.

use crate::language::domain_plugin::{ComputedResult, DomainPlugin, Entity};
use crate::mind::structured_thought::{ETier, EpistemicCube, MTier, NTier};
use symthaea_epidemiology::Sir;

/// Domain plugin for SIR epidemic questions.
pub struct EpidemiologyDomainPlugin;

const CUES: &[&str] = &[
    "r0",
    "r naught",
    "reproduction number",
    "herd immunity",
    "basic reproduction",
    "epidemic",
];

impl EpidemiologyDomainPlugin {
    fn has_cue(text: &str) -> bool {
        let t = text.to_lowercase();
        CUES.iter().any(|c| t.contains(c))
    }

    /// Read the number following a labeled parameter, tolerating `beta=0.3`,
    /// `beta 0.3`, and the greek letters β/γ.
    fn labeled_value(toks: &[&str], labels: &[&str]) -> Option<f64> {
        for (i, t) in toks.iter().enumerate() {
            if labels.contains(t) {
                if let Some(next) = toks.get(i + 1) {
                    // Strip surrounding punctuation so "0.1?" / "0.3," resolve.
                    if let Ok(v) = next
                        .trim_matches(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
                        .parse::<f64>()
                    {
                        return Some(v);
                    }
                }
            }
        }
        None
    }

    fn parse_params(text: &str) -> (Option<f64>, Option<f64>) {
        // Normalise `=` to spaces so `beta=0.3` splits into `beta 0.3`.
        let lower = text.to_lowercase().replace('=', " ");
        let toks: Vec<&str> = lower
            .split(|c: char| c.is_whitespace() || c == ',')
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .collect();
        let beta = Self::labeled_value(&toks, &["beta", "β", "transmission"]);
        let gamma = Self::labeled_value(&toks, &["gamma", "γ", "recovery"]);
        (beta, gamma)
    }
}

impl DomainPlugin for EpidemiologyDomainPlugin {
    fn domain_name(&self) -> &str {
        "epidemiology"
    }

    fn extract_entities(&self, text: &str) -> Vec<Entity> {
        if !Self::has_cue(text) {
            return Vec::new();
        }
        let (beta, gamma) = Self::parse_params(text);
        let mut ents = Vec::new();
        if let Some(b) = beta {
            ents.push(Entity::new("beta", format!("{b}"), 0, 0));
        }
        if let Some(g) = gamma {
            ents.push(Entity::new("gamma", format!("{g}"), 0, 0));
        }
        ents
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        if Self::has_cue(topic) { 0.9 } else { 0.1 }
    }

    fn vocabulary(&self) -> Vec<String> {
        [
            "epidemic",
            "reproduction",
            "herd",
            "immunity",
            "beta",
            "gamma",
            "sir",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn compute(&self, input: &str, _entities: &[Entity]) -> Option<ComputedResult> {
        if !Self::has_cue(input) {
            return None;
        }
        let (beta, gamma) = Self::parse_params(input);
        let (beta, gamma) = (beta?, gamma?);
        if gamma <= 0.0 {
            return None;
        }
        let sir = Sir { beta, gamma };
        let r0 = sir.basic_reproduction_number();
        Some(ComputedResult {
            answer: format!(
                "For β={beta}, γ={gamma}: R₀ = {r0:.2}, herd-immunity threshold = {:.1}%, \
                 final epidemic size ≈ {:.1}%.",
                sir.herd_immunity_threshold() * 100.0,
                sir.final_size() * 100.0
            ),
            cube: EpistemicCube {
                e: ETier::E4,
                n: NTier::N3,
                m: MTier::M3,
                h: None,
            },
            psi: 0.0,
            proof_available: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_r0_and_threshold() {
        let p = EpidemiologyDomainPlugin;
        let r = p
            .compute("what is R0 for beta 0.3 and gamma 0.1?", &[])
            .unwrap();
        assert!(r.answer.contains("R₀ = 3.00"), "answer: {}", r.answer);
        assert!(r.answer.contains("66.7%")); // herd immunity 1 - 1/3
    }

    #[test]
    fn handles_equals_syntax() {
        let p = EpidemiologyDomainPlugin;
        let r = p
            .compute("reproduction number if beta=0.4, gamma=0.1", &[])
            .unwrap();
        assert!(r.answer.contains("R₀ = 4.00"));
    }

    #[test]
    fn no_params_no_computation() {
        let p = EpidemiologyDomainPlugin;
        assert!(p.compute("tell me about the epidemic", &[]).is_none());
    }
}
