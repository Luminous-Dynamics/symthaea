// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Economics domain plugin — deterministic net-present-value answers.
//!
//! Wires `symthaea-economics` into the facade so Symthaea computes NPV from a
//! discount rate and a cash-flow series parsed from the query, bypassing the LLM.
//! Mirrors [`super::chemistry_plugin`] / [`super::physiology_plugin`].

use crate::language::domain_plugin::{ComputedResult, DomainPlugin, Entity};
use crate::mind::structured_thought::{ETier, EpistemicCube, MTier, NTier};
use symthaea_economics::finance::npv;

/// Domain plugin for financial NPV queries.
pub struct EconomicsDomainPlugin;

const CUES: &[&str] = &[
    "npv",
    "net present value",
    "present value",
    "discounted cash",
];

impl EconomicsDomainPlugin {
    fn has_cue(text: &str) -> bool {
        let t = text.to_lowercase();
        CUES.iter().any(|c| t.contains(c))
    }

    /// Parse a discount rate (a `%`-tagged number → fraction) and the ordered
    /// cash-flow series (every other number, signs preserved).
    fn parse_inputs(text: &str) -> (Option<f64>, Vec<f64>) {
        let lower = text.to_lowercase();
        let toks: Vec<&str> = lower
            .split(|c: char| c.is_whitespace() || c == ',')
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .collect();
        let numeric = |s: &str| -> Option<f64> {
            s.trim_matches(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
                .parse::<f64>()
                .ok()
        };

        let mut rate = None;
        let mut rate_idx = None;
        for (i, t) in toks.iter().enumerate() {
            if t.contains('%') {
                if let Some(v) = numeric(t) {
                    rate = Some(v / 100.0);
                    rate_idx = Some(i);
                }
            } else if *t == "%" && i > 0 {
                if let Some(v) = numeric(toks[i - 1]) {
                    rate = Some(v / 100.0);
                    rate_idx = Some(i - 1);
                }
            }
        }

        let flows: Vec<f64> = toks
            .iter()
            .enumerate()
            .filter(|(i, t)| Some(*i) != rate_idx && !t.contains('%'))
            .filter_map(|(_, t)| numeric(t))
            .collect();
        (rate, flows)
    }
}

impl DomainPlugin for EconomicsDomainPlugin {
    fn domain_name(&self) -> &str {
        "economics"
    }

    fn extract_entities(&self, text: &str) -> Vec<Entity> {
        if !Self::has_cue(text) {
            return Vec::new();
        }
        let (rate, flows) = Self::parse_inputs(text);
        let mut ents = Vec::new();
        if let Some(r) = rate {
            ents.push(Entity::new("discount_rate", format!("{r}"), 0, 0));
        }
        for f in flows {
            ents.push(Entity::new("cash_flow", format!("{f}"), 0, 0));
        }
        ents
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        if Self::has_cue(topic) { 0.9 } else { 0.1 }
    }

    fn vocabulary(&self) -> Vec<String> {
        [
            "npv", "present", "value", "discount", "rate", "cash", "flow",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn compute(&self, input: &str, _entities: &[Entity]) -> Option<ComputedResult> {
        if !Self::has_cue(input) {
            return None;
        }
        let (rate, flows) = Self::parse_inputs(input);
        let rate = rate?;
        if flows.len() < 2 {
            return None;
        }
        let value = npv(rate, &flows);
        Some(ComputedResult {
            answer: format!("NPV at {:.1}% of {:?} is {value:.2}.", rate * 100.0, flows),
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
    fn computes_npv() {
        let p = EconomicsDomainPlugin;
        let r = p
            .compute("NPV at 10% of cash flows -1000 500 500 500", &[])
            .unwrap();
        assert!(r.answer.contains("243.4"), "answer: {}", r.answer);
    }

    #[test]
    fn requires_a_rate() {
        let p = EconomicsDomainPlugin;
        // No % → no rate → cannot compute.
        assert!(p.compute("NPV of -1000 500 500", &[]).is_none());
    }

    #[test]
    fn no_cue_no_computation() {
        let p = EconomicsDomainPlugin;
        assert!(p.compute("the price is 10% off", &[]).is_none());
    }
}
