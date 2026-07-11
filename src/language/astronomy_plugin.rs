// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Astronomy domain plugin — Wien's law, Kepler's third law, Schwarzschild radius.

use crate::language::domain_plugin::{ComputedResult, DomainPlugin, Entity};
use crate::language::plugin_parse::value_for_unit;
use crate::mind::structured_thought::{ETier, EpistemicCube, MTier, NTier};
use symthaea_astronomy::{orbital_period_years, schwarzschild_radius, wien_peak_wavelength_nm};

pub struct AstronomyDomainPlugin;

const CUES: &[&str] = &[
    "wien",
    "blackbody",
    "peak wavelength",
    "orbital period",
    "kepler",
    "schwarzschild",
    "event horizon",
];

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

impl AstronomyDomainPlugin {
    fn has_cue(text: &str) -> bool {
        let t = text.to_lowercase();
        CUES.iter().any(|c| t.contains(c))
    }
}

impl DomainPlugin for AstronomyDomainPlugin {
    fn domain_name(&self) -> &str {
        "astronomy"
    }

    fn extract_entities(&self, _text: &str) -> Vec<Entity> {
        Vec::new()
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        if Self::has_cue(topic) { 0.9 } else { 0.1 }
    }

    fn vocabulary(&self) -> Vec<String> {
        [
            "wien",
            "blackbody",
            "kepler",
            "orbital",
            "schwarzschild",
            "magnitude",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn compute(&self, input: &str, _entities: &[Entity]) -> Option<ComputedResult> {
        let lower = input.to_lowercase();
        if !Self::has_cue(input) {
            return None;
        }
        if (lower.contains("wien") || lower.contains("blackbody") || lower.contains("peak"))
            && let Some(t) = value_for_unit(input, &["k", "kelvin"])
        {
            let nm = wien_peak_wavelength_nm(t);
            return Some(result(format!(
                "A blackbody at {t} K peaks at {nm:.1} nm (Wien)."
            )));
        }
        if (lower.contains("orbital period") || lower.contains("kepler"))
            && let Some(a) = value_for_unit(input, &["au"])
        {
            let p = orbital_period_years(a);
            return Some(result(format!(
                "A body with semi-major axis {a} AU has an orbital period of {p:.3} years (Kepler)."
            )));
        }
        if (lower.contains("schwarzschild") || lower.contains("event horizon"))
            && let Some(m) = value_for_unit(input, &["kg"])
        {
            let rs = schwarzschild_radius(m);
            return Some(result(format!(
                "The Schwarzschild radius for {m} kg is {rs:.1} m."
            )));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wien_peak() {
        let p = AstronomyDomainPlugin;
        let r = p.compute("wien peak wavelength for 5778 K", &[]).unwrap();
        assert!(
            r.answer.contains("501") || r.answer.contains("502"),
            "{}",
            r.answer
        );
    }

    #[test]
    fn kepler_period() {
        let p = AstronomyDomainPlugin;
        let r = p.compute("orbital period for 1 AU", &[]).unwrap();
        assert!(r.answer.contains("1.000"));
    }

    #[test]
    fn no_cue_none() {
        let p = AstronomyDomainPlugin;
        assert!(p.compute("the star is bright", &[]).is_none());
    }
}
