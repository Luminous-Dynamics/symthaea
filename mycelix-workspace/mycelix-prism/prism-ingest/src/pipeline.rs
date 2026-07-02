// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bulk claim ingestion pipeline.
//!
//! Loads claims from JSON files in the claims/ directory.
//! Each file contains an array of claims with a simple schema:
//!
//! ```json
//! [
//!   {
//!     "claim": "Factual statement here",
//!     "level": "E4",
//!     "sources": ["Source 1"],
//!     "tags": ["domain", "subdomain"]
//!   }
//! ]
//! ```
//!
//! E-levels: E4 (established), E3 (replicated), E2 (tested), E1 (preliminary)

use crate::RawClaim;
use prism_common::{EmpiricalLevel, MaterialityLevel, NormativeLevel};
use serde::Deserialize;

#[derive(Deserialize)]
struct JsonClaim {
    claim: String,
    level: String,
    sources: Vec<String>,
    tags: Vec<String>,
}

fn parse_level(s: &str) -> EmpiricalLevel {
    match s {
        "E4" => EmpiricalLevel::E4,
        "E3" => EmpiricalLevel::E3,
        "E2" => EmpiricalLevel::E2,
        _ => EmpiricalLevel::E1,
    }
}

fn infer_nm(e: EmpiricalLevel, tags: &[String]) -> (NormativeLevel, MaterialityLevel) {
    let n = match e {
        EmpiricalLevel::E4 => NormativeLevel::N3,
        EmpiricalLevel::E3 => NormativeLevel::N2,
        _ => NormativeLevel::N1,
    };
    let applied = tags.iter().any(|t| {
        matches!(
            t.as_str(),
            "climate"
                | "energy"
                | "ecology"
                | "health"
                | "medicine"
                | "agriculture"
                | "engineering"
                | "nutrition"
                | "water"
                | "environment"
                | "public-health"
        )
    });
    let abstract_ = tags.iter().any(|t| {
        matches!(
            t.as_str(),
            "mathematics" | "physics" | "quantum" | "philosophy" | "logic" | "epistemology"
        )
    });
    let m = if applied {
        MaterialityLevel::M3
    } else if abstract_ {
        MaterialityLevel::M1
    } else {
        MaterialityLevel::M2
    };
    (n, m)
}

fn load_json(json: &str) -> Vec<RawClaim> {
    let entries: Vec<JsonClaim> = match serde_json::from_str(json) {
        Ok(v) => v,
        Err(e) => {
            log::warn!("Failed to parse pipeline JSON: {}", e);
            return vec![];
        }
    };
    entries
        .into_iter()
        .filter_map(|entry| {
            let content = entry.claim.trim().to_string();
            if content.len() < 20 || content.len() > 300 {
                return None;
            }
            let e = parse_level(&entry.level);
            let (n, m) = infer_nm(e, &entry.tags);
            Some(RawClaim {
                content,
                empirical_level: e,
                normative_level: n,
                materiality_level: m,
                sources: entry.sources,
                tags: entry.tags,
            })
        })
        .collect()
}

// Embed all claim JSON files at compile time.
// To add a new domain: create claims/<domain>.json and add include_str! here.
const SOURCES: &[&str] = &[
    include_str!("../claims/physics-astronomy.json"),
    include_str!("../claims/chemistry-materials.json"),
    include_str!("../claims/biology-medicine.json"),
    include_str!("../claims/cs-ai-engineering.json"),
    include_str!("../claims/earth-environment.json"),
    include_str!("../claims/psychology-education.json"),
    include_str!("../claims/economics-society.json"),
    include_str!("../claims/philosophy-humanities.json"),
    include_str!("../claims/health-nutrition.json"),
    include_str!("../claims/energy-sustainability.json"),
    include_str!("../claims/history-civilization.json"),
    include_str!("../claims/mathematics-logic.json"),
    include_str!("../claims/technology-internet.json"),
    include_str!("../claims/politics-governance.json"),
    include_str!("../claims/architecture-design.json"),
    include_str!("../claims/communication-media.json"),
    include_str!("../claims/food-agriculture.json"),
    include_str!("../claims/anthropology-culture.json"),
    include_str!("../claims/wikipedia-extracts.json"),
];

/// Load all pipeline claims from embedded JSON files.
pub fn load_pipeline_claims() -> Vec<RawClaim> {
    let mut all = Vec::new();
    for src in SOURCES {
        all.extend(load_json(src));
    }
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_pipeline_claims() {
        let claims = load_pipeline_claims();
        assert!(
            claims.len() >= 100,
            "Expected 100+ pipeline claims, got {}",
            claims.len()
        );
    }

    #[test]
    fn all_claims_have_tags() {
        for c in load_pipeline_claims() {
            assert!(
                !c.tags.is_empty(),
                "Claim has no tags: {}",
                &c.content[..c.content.len().min(60)]
            );
        }
    }

    #[test]
    fn all_claims_have_sources() {
        for c in load_pipeline_claims() {
            assert!(
                !c.sources.is_empty(),
                "Claim has no sources: {}",
                &c.content[..c.content.len().min(60)]
            );
        }
    }

    #[test]
    fn no_duplicate_claims() {
        let claims = load_pipeline_claims();
        let mut seen = std::collections::HashSet::new();
        for c in &claims {
            let key: String = c
                .content
                .chars()
                .take(80)
                .collect::<String>()
                .to_lowercase();
            assert!(
                seen.insert(key),
                "Duplicate pipeline claim: {}",
                &c.content[..c.content.len().min(60)]
            );
        }
    }
}
