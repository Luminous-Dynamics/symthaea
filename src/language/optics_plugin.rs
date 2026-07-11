// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Optics domain plugin — thin-lens imaging.

use crate::language::domain_plugin::{ComputedResult, DomainPlugin, Entity};
use crate::language::plugin_parse::labeled;
use crate::mind::structured_thought::{ETier, EpistemicCube, MTier, NTier};
use symthaea_optics::{image_distance, magnification};

pub struct OpticsDomainPlugin;

const CUES: &[&str] = &["thin lens", "image distance", "focal length", "lens"];

impl OpticsDomainPlugin {
    fn has_cue(text: &str) -> bool {
        let t = text.to_lowercase();
        CUES.iter().any(|c| t.contains(c))
    }
}

impl DomainPlugin for OpticsDomainPlugin {
    fn domain_name(&self) -> &str {
        "optics"
    }

    fn extract_entities(&self, _text: &str) -> Vec<Entity> {
        Vec::new()
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        if Self::has_cue(topic) { 0.9 } else { 0.1 }
    }

    fn vocabulary(&self) -> Vec<String> {
        [
            "lens",
            "focal",
            "image",
            "object",
            "magnification",
            "optics",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn compute(&self, input: &str, _entities: &[Entity]) -> Option<ComputedResult> {
        if !Self::has_cue(input) {
            return None;
        }
        let f = labeled(input, &["f", "focal"])?;
        let object = labeled(input, &["object", "do", "distance"])?;
        if object == 0.0 {
            return None;
        }
        let di = image_distance(f, object);
        let mag = magnification(di, object);
        Some(ComputedResult {
            answer: format!(
                "A thin lens with focal length {f} and object distance {object} forms an image at \
                 {di:.2} (magnification {mag:.2})."
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
    fn thin_lens_image() {
        let p = OpticsDomainPlugin;
        // f=10, object=30 → image at 15, magnification −0.5.
        let r = p.compute("thin lens with f=10 and object=30", &[]).unwrap();
        assert!(r.answer.contains("15.00"), "{}", r.answer);
        assert!(r.answer.contains("-0.50"));
    }

    #[test]
    fn needs_both_params() {
        let p = OpticsDomainPlugin;
        assert!(p.compute("thin lens with f=10", &[]).is_none());
    }
}
