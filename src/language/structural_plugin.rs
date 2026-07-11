// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Structural / civil domain plugin — closed-form beam bending (stress,
//! deflection, factor of safety) via `symthaea-structural`.
//!
//! Fills the facade-routing gap for the Civil discipline: `EngineeringManager`
//! already exposes `evaluate_structural`, but until now no `DomainPlugin` routed
//! a natural-language structural query to the real solver in the LLM-facing
//! pipeline (unlike control/circuits/optics/… which already have plugins).
//!
//! Envelope: closed-form, linear-elastic, single-span 2D bending. No 3D frames,
//! indeterminate structures, or dynamics — for those the Phase-3 FEA bridge
//! (CalculiX/code_aster) is the answer, not this plugin.

use crate::language::domain_plugin::{ComputedResult, DomainPlugin, Entity};
use crate::language::plugin_parse::labeled;
use crate::mind::structured_thought::{ETier, EpistemicCube, MTier, NTier};
use symthaea_structural::material::steel_a36;
use symthaea_structural::{Beam, LoadCase, Section};

pub struct StructuralDomainPlugin;

const CUES: &[&str] = &[
    "beam",
    "cantilever",
    "bending stress",
    "deflection",
    "factor of safety",
    "simply supported",
    "simply-supported",
];

impl StructuralDomainPlugin {
    fn has_cue(text: &str) -> bool {
        let t = text.to_lowercase();
        CUES.iter().any(|c| t.contains(c))
    }
}

impl DomainPlugin for StructuralDomainPlugin {
    fn domain_name(&self) -> &str {
        "structural"
    }

    fn extract_entities(&self, _text: &str) -> Vec<Entity> {
        Vec::new()
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        if Self::has_cue(topic) { 0.9 } else { 0.1 }
    }

    fn vocabulary(&self) -> Vec<String> {
        [
            "beam",
            "cantilever",
            "bending",
            "stress",
            "deflection",
            "load",
            "span",
            "safety",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn compute(&self, input: &str, _entities: &[Entity]) -> Option<ComputedResult> {
        if !Self::has_cue(input) {
            return None;
        }
        // Geometry + load, each a single-token label followed by a number (m / N).
        let length = labeled(input, &["length", "span"])?;
        let load = labeled(input, &["load", "force"])?;
        let width = labeled(input, &["width", "breadth"])?;
        let height = labeled(input, &["height", "depth"])?;
        if length <= 0.0 || width <= 0.0 || height <= 0.0 {
            return None;
        }
        // Default support is a cantilever with an end point load; "simply
        // supported" switches to a mid-span point load. Material defaults to
        // A36 structural steel.
        let t = input.to_lowercase();
        let (config, load_case) = if t.contains("simply") {
            (
                "simply supported",
                LoadCase::SimplySupportedCenterPoint(load),
            )
        } else {
            ("cantilever", LoadCase::CantileverEndPoint(load))
        };
        let beam = Beam {
            length,
            section: Section::rectangular(width, height),
            material: steel_a36(),
        };
        let r = beam.analyze(load_case);
        Some(ComputedResult {
            answer: format!(
                "A {config} A36-steel beam (L={length} m, {width}×{height} m section) under a \
                 {load} N point load: max bending stress {:.3e} Pa, max deflection {:.3e} m, \
                 factor of safety {:.2}.",
                r.max_bending_stress, r.max_deflection, r.factor_of_safety
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
    fn cantilever_bending_stress_is_known() {
        let p = StructuralDomainPlugin;
        // Cantilever, L=1 m, P=1000 N, rectangular 0.05×0.1 m section.
        // M = P·L = 1000 N·m; I = b·h³/12 = 4.1667e-6 m⁴; c = h/2 = 0.05 m;
        // σ = M·c/I = 1.200e7 Pa (12 MPa). A36 (fy≈250 MPa) → FoS ≫ 2.
        let r = p
            .compute(
                "cantilever beam length 1 load 1000 width 0.05 height 0.1",
                &[],
            )
            .unwrap();
        assert!(r.answer.contains("1.200e7"), "{}", r.answer);
        assert!(r.answer.contains("factor of safety"));
    }

    #[test]
    fn needs_full_geometry() {
        let p = StructuralDomainPlugin;
        // Missing height → cannot build a section → no computed answer.
        assert!(
            p.compute("cantilever beam length 1 load 1000 width 0.05", &[])
                .is_none()
        );
    }

    #[test]
    fn out_of_domain_is_ignored() {
        let p = StructuralDomainPlugin;
        assert!(p.compute("what is the capital of France?", &[]).is_none());
        assert!(p.is_in_domain("what is the capital of France?") < 0.5);
    }
}
