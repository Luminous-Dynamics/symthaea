// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cosine similarity search engine.

use crate::catalog::{Domain, PhysicsCatalog};
use crate::hv::HV;
use serde::{Deserialize, Serialize};

/// Search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub name: String,
    pub domain: Domain,
    pub latex: String,
    pub score: f32,
}

/// Search the catalog by query string.
///
/// Encodes the query as an HV and finds nearest neighbors.
pub fn search_by_text(
    catalog: &PhysicsCatalog,
    query: &str,
    max_results: usize,
) -> Vec<SearchResult> {
    let query_hv = HV::from_name(query);
    search_by_hv(catalog, &query_hv, max_results)
}

/// Search by pre-computed HV.
pub fn search_by_hv(
    catalog: &PhysicsCatalog,
    query_hv: &HV,
    max_results: usize,
) -> Vec<SearchResult> {
    let mut results: Vec<SearchResult> = catalog
        .entries()
        .iter()
        .map(|entry| SearchResult {
            name: entry.name.clone(),
            domain: entry.domain,
            latex: entry.latex.clone(),
            score: entry.hv.similarity(query_hv),
        })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(max_results);
    results
}

/// Search filtered by domain.
pub fn search_in_domain(
    catalog: &PhysicsCatalog,
    query: &str,
    domain: Domain,
    max_results: usize,
) -> Vec<SearchResult> {
    let query_hv = HV::from_name(query);
    let mut results: Vec<SearchResult> = catalog
        .entries()
        .iter()
        .filter(|e| e.domain == domain)
        .map(|entry| SearchResult {
            name: entry.name.clone(),
            domain: entry.domain,
            latex: entry.latex.clone(),
            score: entry.hv.similarity(&query_hv),
        })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(max_results);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_returns_results() {
        let catalog = PhysicsCatalog::new();
        let results = search_by_text(&catalog, "Newton gravitation", 5);
        assert_eq!(results.len(), 5);
        assert!(results[0].score >= results[4].score);
    }

    #[test]
    fn domain_filter_works() {
        let catalog = PhysicsCatalog::new();
        let results = search_in_domain(&catalog, "wave", Domain::Optics, 10);
        for r in &results {
            assert_eq!(r.domain, Domain::Optics);
        }
    }
}
