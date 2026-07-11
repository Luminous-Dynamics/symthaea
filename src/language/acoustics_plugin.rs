// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Acoustics domain plugin — speed of sound in air at a temperature.

use crate::language::domain_plugin::{ComputedResult, DomainPlugin, Entity};
use crate::language::plugin_parse::value_for_unit;
use crate::mind::structured_thought::{ETier, EpistemicCube, MTier, NTier};
use symthaea_acoustics::{speed_of_sound_air, wavelength};

pub struct AcousticsDomainPlugin;

const CUES: &[&str] = &["speed of sound", "sound speed", "wavelength of sound"];

impl AcousticsDomainPlugin {
    fn has_cue(text: &str) -> bool {
        let t = text.to_lowercase();
        CUES.iter().any(|c| t.contains(c))
    }
}

impl DomainPlugin for AcousticsDomainPlugin {
    fn domain_name(&self) -> &str {
        "acoustics"
    }

    fn extract_entities(&self, _text: &str) -> Vec<Entity> {
        Vec::new()
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        if Self::has_cue(topic) { 0.9 } else { 0.1 }
    }

    fn vocabulary(&self) -> Vec<String> {
        ["sound", "speed", "wavelength", "frequency", "acoustics"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn compute(&self, input: &str, _entities: &[Entity]) -> Option<ComputedResult> {
        if !Self::has_cue(input) {
            return None;
        }
        // Default to 20 °C if no temperature is given.
        let temp = value_for_unit(input, &["c", "celsius"]).unwrap_or(20.0);
        let c = speed_of_sound_air(temp);
        // If a frequency is given, also report the wavelength.
        if let Some(f) = value_for_unit(input, &["hz", "hertz"]) {
            let lambda = wavelength(f, c);
            return Some(result(format!(
                "At {temp} °C the speed of sound is {c:.1} m/s; a {f} Hz tone has wavelength {lambda:.3} m."
            )));
        }
        Some(result(format!(
            "The speed of sound in air at {temp} °C is {c:.1} m/s."
        )))
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_at_temperature() {
        let p = AcousticsDomainPlugin;
        let r = p.compute("speed of sound at 20 C", &[]).unwrap();
        assert!(r.answer.contains("343"), "{}", r.answer);
    }

    #[test]
    fn defaults_to_room_temperature() {
        let p = AcousticsDomainPlugin;
        let r = p.compute("what is the speed of sound?", &[]).unwrap();
        assert!(r.answer.contains("343"));
    }

    #[test]
    fn no_cue_none() {
        let p = AcousticsDomainPlugin;
        assert!(p.compute("the music is loud", &[]).is_none());
    }
}
