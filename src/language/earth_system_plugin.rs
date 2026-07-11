// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Earth-system domain plugin — CO₂ forcing and TCRE cumulative-emissions warming.

use crate::language::domain_plugin::{ComputedResult, DomainPlugin, Entity};
use crate::language::plugin_parse::value_for_unit;
use crate::mind::structured_thought::{ETier, EpistemicCube, MTier, NTier};
use symthaea_earth_system::{
    co2_radiative_forcing, warming_from_cumulative_carbon, warming_from_cumulative_co2,
};

pub struct EarthSystemDomainPlugin;

const CUES: &[&str] = &[
    "warming from",
    "cumulative emissions",
    "carbon budget",
    "co2 forcing",
    "radiative forcing",
    "tcre",
    "gtc",
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

impl EarthSystemDomainPlugin {
    fn has_cue(text: &str) -> bool {
        let t = text.to_lowercase();
        CUES.iter().any(|c| t.contains(c))
    }
}

impl DomainPlugin for EarthSystemDomainPlugin {
    fn domain_name(&self) -> &str {
        "earth-system"
    }

    fn extract_entities(&self, _text: &str) -> Vec<Entity> {
        Vec::new()
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        if Self::has_cue(topic) { 0.9 } else { 0.1 }
    }

    fn vocabulary(&self) -> Vec<String> {
        [
            "warming",
            "carbon",
            "emissions",
            "co2",
            "forcing",
            "climate",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn compute(&self, input: &str, _entities: &[Entity]) -> Option<ComputedResult> {
        if !Self::has_cue(input) {
            return None;
        }
        let lower = input.to_lowercase();
        if lower.contains("doubling") && lower.contains("co2") {
            let df = co2_radiative_forcing(560.0, 280.0);
            return Some(result(format!(
                "Doubling CO₂ gives a radiative forcing of {df:.2} W/m²."
            )));
        }
        if let Some(gtc) = value_for_unit(input, &["gtc", "gtcarbon"]) {
            let dt = warming_from_cumulative_carbon(gtc);
            return Some(result(format!(
                "Cumulative emissions of {gtc} GtC cause about {dt:.2} °C of warming (TCRE)."
            )));
        }
        if let Some(gtco2) = value_for_unit(input, &["gtco2"]) {
            let dt = warming_from_cumulative_co2(gtco2);
            return Some(result(format!(
                "Cumulative emissions of {gtco2} GtCO₂ cause about {dt:.2} °C of warming (TCRE)."
            )));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tcre_warming() {
        let p = EarthSystemDomainPlugin;
        let r = p.compute("warming from 1000 GtC", &[]).unwrap();
        assert!(r.answer.contains("1.65"), "{}", r.answer);
    }

    #[test]
    fn co2_doubling_forcing() {
        let p = EarthSystemDomainPlugin;
        let r = p
            .compute("what is the radiative forcing from doubling CO2?", &[])
            .unwrap();
        assert!(r.answer.contains("3.7"), "{}", r.answer);
    }

    #[test]
    fn no_cue_none() {
        let p = EarthSystemDomainPlugin;
        assert!(p.compute("it is warm today", &[]).is_none());
    }
}
