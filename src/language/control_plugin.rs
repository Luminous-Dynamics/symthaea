// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Control-theory domain plugin — second-order step response (overshoot/settling).

use crate::language::domain_plugin::{ComputedResult, DomainPlugin, Entity};
use crate::language::plugin_parse::labeled;
use crate::mind::structured_thought::{ETier, EpistemicCube, MTier, NTier};
use symthaea_control_theory::SecondOrder;

pub struct ControlDomainPlugin;

const CUES: &[&str] = &[
    "overshoot",
    "settling time",
    "damping ratio",
    "second order",
    "second-order",
];

impl ControlDomainPlugin {
    fn has_cue(text: &str) -> bool {
        let t = text.to_lowercase();
        CUES.iter().any(|c| t.contains(c))
    }
}

impl DomainPlugin for ControlDomainPlugin {
    fn domain_name(&self) -> &str {
        "control"
    }

    fn extract_entities(&self, _text: &str) -> Vec<Entity> {
        Vec::new()
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        if Self::has_cue(topic) { 0.9 } else { 0.1 }
    }

    fn vocabulary(&self) -> Vec<String> {
        [
            "overshoot",
            "settling",
            "damping",
            "frequency",
            "control",
            "response",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn compute(&self, input: &str, _entities: &[Entity]) -> Option<ComputedResult> {
        if !Self::has_cue(input) {
            return None;
        }
        // The damping ratio follows "ratio"/"damping"/"zeta"; ωₙ follows
        // "frequency"/"omega".
        let zeta = labeled(input, &["ratio", "damping", "zeta"])?;
        let wn = labeled(input, &["frequency", "omega", "omegan", "wn"])?;
        if wn <= 0.0 || zeta < 0.0 {
            return None;
        }
        let sys = SecondOrder {
            natural_freq: wn,
            damping_ratio: zeta,
        };
        Some(ComputedResult {
            answer: format!(
                "A second-order system with ζ={zeta}, ωₙ={wn}: percent overshoot {:.1}%, \
                 2% settling time {:.2} s.",
                sys.percent_overshoot(),
                sys.settling_time()
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
    fn overshoot_and_settling() {
        let p = ControlDomainPlugin;
        // ζ=0.5, ωn=1 → overshoot ≈ 16.3%, settling ≈ 8 s.
        let r = p
            .compute(
                "overshoot for damping ratio 0.5 and natural frequency 1",
                &[],
            )
            .unwrap();
        assert!(r.answer.contains("16.3"), "{}", r.answer);
        assert!(r.answer.contains("8.00"));
    }

    #[test]
    fn needs_both_params() {
        let p = ControlDomainPlugin;
        assert!(p.compute("overshoot for damping ratio 0.5", &[]).is_none());
    }
}
