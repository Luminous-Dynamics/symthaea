// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Physiology domain plugin.
//!
//! Wires `symthaea-physiology` into the facade's domain-plugin path so Symthaea
//! answers body-metric questions **deterministically** — it parses mass and
//! height from the query and computes BMI (with the standard WHO category) in
//! Rust, bypassing LLM generation. Mirrors [`super::chemistry_plugin`].

use crate::language::domain_plugin::{ComputedResult, DomainPlugin, Entity};
use crate::mind::structured_thought::{ETier, EpistemicCube, MTier, NTier};
use symthaea_physiology::nutrition::bmi;

/// Domain plugin for human physiology / body metrics.
pub struct PhysiologyDomainPlugin;

const CUES: &[&str] = &["bmi", "body mass index", "body-mass"];

impl PhysiologyDomainPlugin {
    fn has_cue(text: &str) -> bool {
        let t = text.to_lowercase();
        CUES.iter().any(|c| t.contains(c))
    }

    /// Parse `(value, unit)` measurements, tolerating "70 kg" and "70kg" forms.
    /// Returns `(mass_kg, height_m)` when both are present.
    fn parse_body_metrics(text: &str) -> (Option<f64>, Option<f64>) {
        let (mut mass, mut height) = (None, None);
        let lower = text.to_lowercase();
        // Tokenize; for each numeric token, read the unit from the same token
        // ("70kg") or the following one ("70 kg"), stripping any surrounding
        // punctuation so "1.75 m?" still resolves.
        let toks: Vec<&str> = lower
            .split(|c: char| c.is_whitespace() || c == ',')
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .collect();
        let clean_unit = |s: &str| {
            s.trim_matches(|c: char| !c.is_ascii_alphabetic())
                .to_string()
        };
        for (idx, tok) in toks.iter().enumerate() {
            let split = tok.find(|c: char| !(c.is_ascii_digit() || c == '.'));
            let (num_str, inline) = match split {
                Some(0) | None => (*tok, ""),
                Some(i) => (&tok[..i], &tok[i..]),
            };
            let Ok(value) = num_str.parse::<f64>() else {
                continue;
            };
            let unit = if !inline.is_empty() {
                clean_unit(inline)
            } else if let Some(next) = toks.get(idx + 1) {
                clean_unit(next)
            } else {
                continue;
            };
            match unit.as_str() {
                "kg" | "kgs" | "kilogram" | "kilograms" => mass = mass.or(Some(value)),
                "m" | "meter" | "meters" | "metre" | "metres" => height = height.or(Some(value)),
                "cm" | "centimeter" | "centimeters" => height = height.or(Some(value / 100.0)),
                _ => {}
            }
        }
        (mass, height)
    }

    fn category(bmi: f64) -> &'static str {
        if bmi < 18.5 {
            "underweight"
        } else if bmi < 25.0 {
            "normal weight"
        } else if bmi < 30.0 {
            "overweight"
        } else {
            "obese"
        }
    }
}

impl DomainPlugin for PhysiologyDomainPlugin {
    fn domain_name(&self) -> &str {
        "physiology"
    }

    fn extract_entities(&self, text: &str) -> Vec<Entity> {
        if !Self::has_cue(text) {
            return Vec::new();
        }
        let (mass, height) = Self::parse_body_metrics(text);
        let mut ents = Vec::new();
        if let Some(m) = mass {
            ents.push(Entity::new("mass_kg", format!("{m}"), 0, 0));
        }
        if let Some(h) = height {
            ents.push(Entity::new("height_m", format!("{h}"), 0, 0));
        }
        ents
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        if Self::has_cue(topic) { 0.9 } else { 0.1 }
    }

    fn vocabulary(&self) -> Vec<String> {
        [
            "bmi", "body", "mass", "index", "weight", "height", "kg", "meters",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn compute(&self, input: &str, entities: &[Entity]) -> Option<ComputedResult> {
        if !Self::has_cue(input) {
            return None;
        }
        let mass: f64 = entities
            .iter()
            .find(|e| e.entity_type == "mass_kg")?
            .value
            .parse()
            .ok()?;
        let height: f64 = entities
            .iter()
            .find(|e| e.entity_type == "height_m")?
            .value
            .parse()
            .ok()?;
        if height <= 0.0 {
            return None;
        }
        let b = bmi(mass, height);
        Some(ComputedResult {
            answer: format!(
                "BMI for {mass} kg at {height} m is {b:.1} ({}).",
                Self::category(b)
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
    fn parses_and_computes_bmi() {
        let p = PhysiologyDomainPlugin;
        let ents = p.extract_entities("what is the BMI for 70 kg and 1.75 m?");
        let r = p.compute("BMI for 70 kg and 1.75 m", &ents).unwrap();
        assert!(r.answer.contains("22.9"));
        assert!(r.answer.contains("normal weight"));
    }

    #[test]
    fn handles_cm_and_no_space_units() {
        let p = PhysiologyDomainPlugin;
        let ents = p.extract_entities("BMI 90kg 180cm");
        let r = p.compute("BMI 90kg 180cm", &ents).unwrap();
        // 90 / 1.8^2 = 27.8 → overweight.
        assert!(r.answer.contains("27.8"));
        assert!(r.answer.contains("overweight"));
    }

    #[test]
    fn no_cue_no_extraction() {
        let p = PhysiologyDomainPlugin;
        assert!(p.extract_entities("I weigh 70 kg").is_empty());
        assert!(p.compute("tell me about health", &[]).is_none());
    }
}
