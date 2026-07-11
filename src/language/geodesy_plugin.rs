// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Geodesy domain plugin — deterministic great-circle distance.
//!
//! Wires `symthaea-geodesy` into the facade: given two lat/lon coordinate pairs
//! in the query, computes the haversine distance in km, bypassing the LLM.

use crate::language::domain_plugin::{ComputedResult, DomainPlugin, Entity};
use crate::mind::structured_thought::{ETier, EpistemicCube, MTier, NTier};
use symthaea_geodesy::haversine_distance;

/// Domain plugin for great-circle distance between coordinates.
pub struct GeodesyDomainPlugin;

const CUES: &[&str] = &[
    "great circle",
    "great-circle",
    "distance between",
    "distance from",
    "how far",
    "geodesic distance",
    "km between",
];

impl GeodesyDomainPlugin {
    fn has_cue(text: &str) -> bool {
        let t = text.to_lowercase();
        CUES.iter().any(|c| t.contains(c))
    }

    /// Extract the first four signed decimal numbers as `(lat1, lon1, lat2, lon2)`.
    fn parse_coords(text: &str) -> Option<(f64, f64, f64, f64)> {
        let nums: Vec<f64> = text
            .split(|c: char| c.is_whitespace() || c == ',' || c == ';')
            .filter_map(|t| {
                t.trim_matches(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
                    .parse::<f64>()
                    .ok()
            })
            .collect();
        if nums.len() >= 4 {
            Some((nums[0], nums[1], nums[2], nums[3]))
        } else {
            None
        }
    }
}

impl DomainPlugin for GeodesyDomainPlugin {
    fn domain_name(&self) -> &str {
        "geodesy"
    }

    fn extract_entities(&self, text: &str) -> Vec<Entity> {
        if !Self::has_cue(text) {
            return Vec::new();
        }
        match Self::parse_coords(text) {
            Some((a, b, c, d)) => vec![
                Entity::new("coord", format!("{a},{b}"), 0, 0),
                Entity::new("coord", format!("{c},{d}"), 0, 0),
            ],
            None => Vec::new(),
        }
    }

    fn is_in_domain(&self, topic: &str) -> f64 {
        if Self::has_cue(topic) { 0.9 } else { 0.1 }
    }

    fn vocabulary(&self) -> Vec<String> {
        ["distance", "great", "circle", "latitude", "longitude", "km"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn compute(&self, input: &str, _entities: &[Entity]) -> Option<ComputedResult> {
        if !Self::has_cue(input) {
            return None;
        }
        let (lat1, lon1, lat2, lon2) = Self::parse_coords(input)?;
        let km = haversine_distance(lat1, lon1, lat2, lon2);
        Some(ComputedResult {
            answer: format!(
                "The great-circle distance from ({lat1}, {lon1}) to ({lat2}, {lon2}) is {km:.0} km."
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
    fn computes_distance() {
        let p = GeodesyDomainPlugin;
        let r = p
            .compute("distance from 51.5074 -0.1278 to 48.8566 2.3522", &[])
            .unwrap();
        // London → Paris ≈ 343 km.
        let expected = format!(
            "{:.0}",
            haversine_distance(51.5074, -0.1278, 48.8566, 2.3522)
        );
        assert!(r.answer.contains(&expected), "answer: {}", r.answer);
        assert!(r.answer.contains("km"));
        // London → Paris is ~343.5 km.
        let d = haversine_distance(51.5074, -0.1278, 48.8566, 2.3522);
        assert!((340.0..346.0).contains(&d), "distance {d}");
    }

    #[test]
    fn needs_two_coordinate_pairs() {
        let p = GeodesyDomainPlugin;
        assert!(p.compute("distance from 51.5 -0.13", &[]).is_none());
    }

    #[test]
    fn no_cue_no_computation() {
        let p = GeodesyDomainPlugin;
        assert!(p.compute("51.5 -0.13 48.86 2.35", &[]).is_none());
    }
}
