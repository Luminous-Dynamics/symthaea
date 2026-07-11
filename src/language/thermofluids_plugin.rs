// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Thermofluids domain plugin — Carnot efficiency between two temperatures.

use crate::language::domain_plugin::{ComputedResult, DomainPlugin, Entity};
use crate::language::plugin_parse::values_for_unit;
use crate::mind::structured_thought::{ETier, EpistemicCube, MTier, NTier};
use symthaea_thermofluids::thermal::carnot_efficiency;

pub struct ThermofluidsDomainPlugin;

const CUES: &[&str] = &["carnot", "heat engine", "thermal efficiency"];

impl ThermofluidsDomainPlugin {
    fn has_cue(text: &str) -> bool {
        let t = text.to_lowercase();
        CUES.iter().any(|c| t.contains(c))
    }
}

impl DomainPlugin for ThermofluidsDomainPlugin {
    fn domain_name(&self) -> &str {
        "thermofluids"
    }

    fn extract_entities(&self, _text: &str) -> Vec<Entity> {
        Vec::new()
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        if Self::has_cue(topic) { 0.9 } else { 0.1 }
    }

    fn vocabulary(&self) -> Vec<String> {
        [
            "carnot",
            "heat",
            "engine",
            "efficiency",
            "temperature",
            "kelvin",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn compute(&self, input: &str, _entities: &[Entity]) -> Option<ComputedResult> {
        if !Self::has_cue(input) {
            return None;
        }
        let temps = values_for_unit(input, &["k", "kelvin"]);
        if temps.len() < 2 {
            return None;
        }
        let cold = temps.iter().cloned().fold(f64::INFINITY, f64::min);
        let hot = temps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if hot <= 0.0 {
            return None;
        }
        let eff = carnot_efficiency(cold, hot);
        Some(ComputedResult {
            answer: format!(
                "The Carnot efficiency between {cold} K (cold) and {hot} K (hot) is {:.1}%.",
                eff * 100.0
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
    fn carnot_between_two_temps() {
        let p = ThermofluidsDomainPlugin;
        let r = p
            .compute("carnot efficiency between 300 K and 600 K", &[])
            .unwrap();
        assert!(r.answer.contains("50.0%"), "{}", r.answer);
    }

    #[test]
    fn needs_two_temperatures() {
        let p = ThermofluidsDomainPlugin;
        assert!(p.compute("carnot efficiency at 600 K", &[]).is_none());
    }

    #[test]
    fn no_cue_none() {
        let p = ThermofluidsDomainPlugin;
        assert!(p.compute("it is 300 K and 600 K outside", &[]).is_none());
    }
}
