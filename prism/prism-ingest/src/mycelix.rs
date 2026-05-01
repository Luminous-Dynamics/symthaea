// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Load claims from mycelix-knowledge seed data (embedded at compile time).

use crate::RawClaim;
use prism_common::{EmpiricalLevel, MaterialityLevel, NormativeLevel};
use serde::Deserialize;

// Embed JSON files at compile time — no filesystem access needed in WASM
const ENERGY_JSON: &str =
    include_str!("../../../mycelix-knowledge/seed-data/claims/energy-facts.json");
const GEO_JSON: &str =
    include_str!("../../../mycelix-knowledge/seed-data/claims/geonames-major.json");
const WHO_JSON: &str = include_str!("../../../mycelix-knowledge/seed-data/claims/who-health.json");
const NOAA_JSON: &str =
    include_str!("../../../mycelix-knowledge/seed-data/claims/noaa-climate.json");
const FAO_JSON: &str =
    include_str!("../../../mycelix-knowledge/seed-data/claims/fao-agriculture.json");

#[derive(Deserialize)]
struct MycelixClaim {
    content: String,
    classification: Option<Classification>,
    sources: Option<Vec<String>>,
    tags: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct Classification {
    empirical: Option<f64>,
    normative: Option<f64>,
    #[serde(alias = "mythic")]
    materiality: Option<f64>,
}

/// Load and parse all mycelix-knowledge seed claims.
pub fn load_mycelix_claims() -> Vec<RawClaim> {
    let mut claims = Vec::new();

    for json_str in [ENERGY_JSON, GEO_JSON, WHO_JSON, NOAA_JSON, FAO_JSON] {
        let data: Vec<MycelixClaim> = match serde_json::from_str(json_str) {
            Ok(d) => d,
            Err(e) => {
                log::warn!("Failed to parse mycelix JSON: {}", e);
                continue;
            }
        };

        for c in data {
            let content = c.content.trim().to_string();

            // Filter: too short, too long, or low-semantic-value content
            if content.len() < 20 || content.len() > 250 {
                continue;
            }
            let low = content.to_lowercase();

            // Coordinate-only claims
            if low.contains("latitude") && low.contains("longitude") {
                continue;
            }
            if low.matches('°').count() >= 2 {
                continue;
            }

            // City population boilerplate ("X is a major city in Y with a metropolitan population")
            // These flood search results with low-value geographic noise
            if low.contains("is a major city") && low.contains("metropolitan population") {
                continue;
            }

            let e_val = c
                .classification
                .as_ref()
                .and_then(|cl| cl.empirical)
                .unwrap_or(0.0);

            let e_level = if e_val >= 0.75 {
                EmpiricalLevel::E4
            } else if e_val >= 0.5 {
                EmpiricalLevel::E3
            } else if e_val >= 0.25 {
                EmpiricalLevel::E2
            } else {
                EmpiricalLevel::E1
            };

            // Simplify source URLs to domain names
            let sources: Vec<String> = c
                .sources
                .unwrap_or_default()
                .into_iter()
                .take(2)
                .map(|s| s.split('/').nth(2).unwrap_or(&s).to_string())
                .collect();

            let tags: Vec<String> = c.tags.unwrap_or_default().into_iter().take(3).collect();

            let n_val = c
                .classification
                .as_ref()
                .and_then(|cl| cl.normative)
                .unwrap_or(0.5);
            let n_level = if n_val >= 0.75 {
                NormativeLevel::N3
            } else if n_val >= 0.5 {
                NormativeLevel::N2
            } else if n_val >= 0.25 {
                NormativeLevel::N1
            } else {
                NormativeLevel::N0
            };

            let m_val = c
                .classification
                .as_ref()
                .and_then(|cl| cl.materiality)
                .unwrap_or(0.5);
            let m_level = if m_val >= 0.75 {
                MaterialityLevel::M3
            } else if m_val >= 0.5 {
                MaterialityLevel::M2
            } else if m_val >= 0.25 {
                MaterialityLevel::M1
            } else {
                MaterialityLevel::M0
            };

            claims.push(RawClaim {
                content,
                empirical_level: e_level,
                normative_level: n_level,
                materiality_level: m_level,
                sources,
                tags,
            });
        }
    }

    claims
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_mycelix_claims() {
        let claims = load_mycelix_claims();
        assert!(!claims.is_empty(), "Should load mycelix claims");
        // Should have filtered out coordinate-only claims
        for c in &claims {
            let low = c.content.to_lowercase();
            assert!(
                !(low.contains("latitude") && low.contains("longitude")),
                "Should filter coordinates: {}",
                c.content
            );
        }
    }
}
