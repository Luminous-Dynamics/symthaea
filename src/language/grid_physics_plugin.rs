// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Electrical-power domain plugin — single-line radial feeder voltage-drop
//! check via `symthaea-grid-physics` (linearized DistFlow).
//!
//! Fills the last facade-routing gap in the engineering faculty: `EngineeringManager`
//! already exposes `evaluate_electrical`, but it takes a full `Feeder` topology
//! (a tree of buses), which doesn't fit the "few labeled scalars in one sentence"
//! shape every other discipline plugin uses. This plugin covers the realistic
//! single-sentence case: a two-bus feeder (substation root -> one load bus),
//! which is still a genuine exercise of the real DistFlow solver, not a toy
//! shortcut. Multi-bus topologies remain an `EngineeringManager`-level capability
//! (`evaluate_electrical` / `discharge_electrical_check`), reachable directly,
//! not through natural language.
//!
//! Envelope: linearized DistFlow, radial, single branch, balanced. No meshed
//! networks, multi-bus feeders, or unbalanced phases — see
//! `symthaea-grid-physics::feeder` module docs for the full scope statement.

use crate::language::domain_plugin::{ComputedResult, DomainPlugin, Entity};
use crate::language::plugin_parse::labeled;
use crate::mind::structured_thought::{ETier, EpistemicCube, MTier, NTier};
use symthaea_grid_physics::feeder::{Feeder, Line, Node};

pub struct GridPhysicsDomainPlugin;

const CUES: &[&str] = &[
    "feeder",
    "voltage drop",
    "distflow",
    "distribution feeder",
    "bus voltage",
    "per-unit voltage",
];

impl GridPhysicsDomainPlugin {
    fn has_cue(text: &str) -> bool {
        let t = text.to_lowercase();
        CUES.iter().any(|c| t.contains(c))
    }
}

impl DomainPlugin for GridPhysicsDomainPlugin {
    fn domain_name(&self) -> &str {
        "grid-physics"
    }

    fn extract_entities(&self, _text: &str) -> Vec<Entity> {
        Vec::new()
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        if Self::has_cue(topic) { 0.9 } else { 0.1 }
    }

    fn vocabulary(&self) -> Vec<String> {
        [
            "feeder",
            "voltage",
            "distribution",
            "resistance",
            "reactance",
            "load",
            "bus",
            "distflow",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn compute(&self, input: &str, _entities: &[Entity]) -> Option<ComputedResult> {
        if !Self::has_cue(input) {
            return None;
        }
        // Substation base voltage (V) and the one branch's electrical params.
        let base_voltage_v = labeled(input, &["voltage", "base"])?;
        let resistance_ohm = labeled(input, &["resistance", "ohms", "ohm"])?;
        let reactance_ohm = labeled(input, &["reactance"])?;
        let load_kw = labeled(input, &["load", "kw"])?;
        let load_kvar = labeled(input, &["kvar", "reactive"]).unwrap_or(0.0);
        // Acceptable per-unit voltage band; ANSI C84.1 Range A (0.95-1.05) is
        // the honest default absent an explicit band.
        let min_pu = labeled(input, &["min", "minimum"]).unwrap_or(0.95);
        let max_pu = labeled(input, &["max", "maximum"]).unwrap_or(1.05);
        if base_voltage_v <= 0.0 || resistance_ohm < 0.0 || reactance_ohm < 0.0 {
            return None;
        }
        let feeder = Feeder::new(
            base_voltage_v,
            vec![
                Node::root(),
                Node::load(
                    0,
                    Line {
                        resistance_ohm,
                        reactance_ohm,
                    },
                    load_kw,
                    load_kvar,
                ),
            ],
        )
        .ok()?;
        let solution = feeder.solve();
        let v_pu = solution.voltage_pu(1);
        let passes = v_pu >= min_pu && v_pu <= max_pu;
        Some(ComputedResult {
            answer: format!(
                "A single-branch {base_voltage_v} V feeder (R={resistance_ohm} Ω, \
                 X={reactance_ohm} Ω) serving {load_kw} kW / {load_kvar} kVAR: load-bus voltage \
                 {v_pu:.4} pu vs required [{min_pu:.3}, {max_pu:.3}] -> {}. Linearized DistFlow, \
                 radial, single branch, balanced — not a full nonlinear AC power-flow solve.",
                if passes { "PASS" } else { "FAIL" }
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
    fn healthy_feeder_stays_in_band() {
        let p = GridPhysicsDomainPlugin;
        // A short, lightly-loaded branch off a 12470 V feeder should barely sag.
        let r = p
            .compute(
                "feeder voltage 12470 resistance 2 reactance 1 load 50 kvar 10",
                &[],
            )
            .unwrap();
        assert!(r.answer.contains("PASS"), "{}", r.answer);
    }

    #[test]
    fn heavy_load_sags_out_of_band() {
        let p = GridPhysicsDomainPlugin;
        // A very heavy load on a low base voltage over a resistive branch
        // drives the load-bus voltage well below a tight 0.99-1.01 band.
        let r = p
            .compute(
                "feeder voltage drop voltage 208 resistance 5 reactance 2 load 500 kvar 200 \
                 min 0.99 max 1.01",
                &[],
            )
            .unwrap();
        assert!(r.answer.contains("FAIL"), "{}", r.answer);
    }

    #[test]
    fn needs_branch_params() {
        let p = GridPhysicsDomainPlugin;
        // Missing reactance -> cannot build the branch.
        assert!(
            p.compute("feeder voltage 12470 resistance 2 load 50", &[])
                .is_none()
        );
    }

    #[test]
    fn out_of_domain_is_ignored() {
        let p = GridPhysicsDomainPlugin;
        assert!(p.compute("what is the capital of France?", &[]).is_none());
    }
}
