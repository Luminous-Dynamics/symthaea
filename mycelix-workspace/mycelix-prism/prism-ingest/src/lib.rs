// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pure Rust data ingestion for Prism.
//!
//! Loads claims from curated sources, mycelix-knowledge JSON,
//! and Wikidata SPARQL results. Deduplicates and quality-filters.

pub mod curated;
pub mod mycelix;
pub mod pipeline;
pub mod wikidata_claims;

use prism_common::{EmpiricalLevel, MaterialityLevel, NormativeLevel};

/// A raw claim before HDC encoding — full E/N/M classification.
#[derive(Debug, Clone)]
pub struct RawClaim {
    pub content: String,
    pub empirical_level: EmpiricalLevel,
    pub normative_level: NormativeLevel,
    pub materiality_level: MaterialityLevel,
    pub sources: Vec<String>,
    pub tags: Vec<String>,
}

/// Safely truncate a string to at most `max` bytes at a char boundary.
fn safe_truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut i = max;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    &s[..i]
}

/// Check if a claim passes quality filters.
fn passes_quality(content: &str) -> bool {
    let low = content.to_lowercase();

    // Too short to be useful
    if content.len() < 20 {
        return false;
    }

    // Coordinate-only data
    if low.contains("latitude") && low.contains("longitude") {
        return false;
    }
    if low.matches('°').count() >= 2 {
        return false;
    }

    // City population boilerplate
    if low.contains("is a major city") && low.contains("metropolitan population") {
        return false;
    }

    // Wikidata description boilerplate (too generic)
    if low.starts_with("described as ") && low.len() < 50 {
        return false;
    }

    // Pure "Wikimedia" meta entries
    if low.contains("wikimedia") || low.contains("disambiguation") {
        return false;
    }

    true
}

/// Load curated claims only (high-quality, ~140 claims).
pub fn curated_claims() -> Vec<RawClaim> {
    curated::curated_claims()
        .into_iter()
        .filter(|c| passes_quality(&c.content))
        .collect()
}

/// Load pipeline claims from JSON files (~400+ claims).
pub fn pipeline_claims() -> Vec<RawClaim> {
    pipeline::load_pipeline_claims()
        .into_iter()
        .filter(|c| passes_quality(&c.content))
        .collect()
}

/// Load mycelix JSON claims only (~100-200 claims).
pub fn mycelix_claims() -> Vec<RawClaim> {
    mycelix::load_mycelix_claims()
        .into_iter()
        .filter(|c| passes_quality(&c.content))
        .collect()
}

/// Load all claims from all sources, deduplicated and quality-filtered.
pub fn load_all_claims() -> Vec<RawClaim> {
    let mut claims = Vec::new();
    let mut seen = std::collections::HashSet::new();

    let mut add = |c: RawClaim| {
        if !passes_quality(&c.content) {
            return;
        }
        let key = safe_truncate(&c.content.to_lowercase(), 80).to_string();
        if seen.insert(key) {
            claims.push(c);
        }
    };

    // Curated (highest quality, loaded first)
    for c in curated::curated_claims() {
        add(c);
    }

    // Pipeline (bulk JSON claims, second priority)
    for c in pipeline::load_pipeline_claims() {
        add(c);
    }

    // Mycelix-knowledge
    for c in mycelix::load_mycelix_claims() {
        add(c);
    }

    // Wikidata
    for c in wikidata_claims::wikidata_claims() {
        add(c);
    }

    log::info!("Ingested {} total claims", claims.len());
    claims
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_claims() {
        let claims = load_all_claims();
        assert!(
            claims.len() >= 50,
            "Should load at least 50 claims, got {}",
            claims.len()
        );
    }

    #[test]
    fn no_duplicates() {
        let claims = load_all_claims();
        let mut seen = std::collections::HashSet::new();
        for c in &claims {
            let key = safe_truncate(&c.content.to_lowercase(), 80).to_string();
            assert!(
                seen.insert(key),
                "Duplicate: {}",
                safe_truncate(&c.content, 60)
            );
        }
    }

    #[test]
    fn no_coordinate_claims() {
        let claims = load_all_claims();
        for c in &claims {
            let low = c.content.to_lowercase();
            assert!(
                !(low.contains("latitude") && low.contains("longitude")),
                "Coordinate claim should be filtered: {}",
                safe_truncate(&c.content, 60)
            );
        }
    }

    #[test]
    fn no_boilerplate_claims() {
        let claims = load_all_claims();
        for c in &claims {
            let low = c.content.to_lowercase();
            assert!(
                !low.contains("wikimedia"),
                "Wikimedia meta claim: {}",
                safe_truncate(&c.content, 60)
            );
        }
    }
}
