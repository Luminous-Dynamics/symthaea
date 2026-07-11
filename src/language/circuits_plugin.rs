// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Circuits domain plugin — Ohm's law (solve for the missing of V, I, R).

use crate::language::domain_plugin::{ComputedResult, DomainPlugin, Entity};
use crate::language::plugin_parse::value_for_unit;
use crate::mind::structured_thought::{ETier, EpistemicCube, MTier, NTier};
use symthaea_circuits::dc::{current, voltage};

pub struct CircuitsDomainPlugin;

const CUES: &[&str] = &["ohm", "voltage across", "current through", "resistor"];

fn result(answer: String) -> ComputedResult {
    ComputedResult {
        answer,
        cube: EpistemicCube {
            e: ETier::E4,
            n: NTier::N3,
            m: MTier::M3,
            h: None,
        },
        psi: 0.0,
        proof_available: false,
    }
}

impl CircuitsDomainPlugin {
    fn has_cue(text: &str) -> bool {
        let t = text.to_lowercase();
        CUES.iter().any(|c| t.contains(c))
    }
}

impl DomainPlugin for CircuitsDomainPlugin {
    fn domain_name(&self) -> &str {
        "circuits"
    }

    fn extract_entities(&self, _text: &str) -> Vec<Entity> {
        Vec::new()
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        if Self::has_cue(topic) { 0.9 } else { 0.1 }
    }

    fn vocabulary(&self) -> Vec<String> {
        ["ohm", "volt", "amp", "resistor", "current", "voltage"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn compute(&self, input: &str, _entities: &[Entity]) -> Option<ComputedResult> {
        if !Self::has_cue(input) {
            return None;
        }
        let v = value_for_unit(input, &["v", "volt", "volts"]);
        let i = value_for_unit(input, &["a", "amp", "amps", "ampere", "amperes"]);
        let r = value_for_unit(input, &["ohm", "ohms"]);
        match (v, i, r) {
            (Some(v), None, Some(r)) => Some(result(format!(
                "Ohm's law: the current is {:.4} A.",
                current(v, r)
            ))),
            (None, Some(i), Some(r)) => Some(result(format!(
                "Ohm's law: the voltage is {:.4} V.",
                voltage(i, r)
            ))),
            (Some(v), Some(i), None) if i != 0.0 => Some(result(format!(
                "Ohm's law: the resistance is {:.4} Ω.",
                v / i
            ))),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_for_voltage() {
        let p = CircuitsDomainPlugin;
        // 3 A through 4 ohm → 12 V.
        let r = p
            .compute("voltage across a 4 ohm resistor carrying 3 amps", &[])
            .unwrap();
        assert!(r.answer.contains("12.0"), "{}", r.answer);
    }

    #[test]
    fn solves_for_current() {
        let p = CircuitsDomainPlugin;
        // 12 V across 4 ohm → 3 A.
        let r = p
            .compute("current through a 4 ohm resistor at 12 volts", &[])
            .unwrap();
        assert!(r.answer.contains("3.0"), "{}", r.answer);
    }

    #[test]
    fn needs_two_of_three() {
        let p = CircuitsDomainPlugin;
        assert!(p.compute("a 4 ohm resistor", &[]).is_none());
    }
}
