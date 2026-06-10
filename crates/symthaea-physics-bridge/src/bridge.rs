// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cognitive loop bridge.
//!
//! Thin wrapper around `PhysicsSearchEngine` that produces `ContinuousHV`
//! compatible with `CognitiveLoopService::cycle_with_hv()`.
//!
//! Follows the vision-manifold bridge pattern.

use crate::query::PhysicsSearchEngine;
use crate::types::{PhysicsBridgeTelemetry, PhysicsEquation, SearchResult};
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

/// Bridge from physics search engine to cognitive loop.
///
/// Wraps `PhysicsSearchEngine` and produces `ContinuousHV` outputs
/// suitable for feeding into `cycle_with_hv()`.
pub struct PhysicsBridge {
    engine: PhysicsSearchEngine,
    last_query_hv: Option<ContinuousHV>,
    last_results: Vec<SearchResult>,
    query_count: u64,
}

impl PhysicsBridge {
    /// Create a new bridge with the default catalog.
    pub fn new() -> Self {
        Self {
            engine: PhysicsSearchEngine::new(),
            last_query_hv: None,
            last_results: Vec::new(),
            query_count: 0,
        }
    }

    /// Query the catalog with a physics equation and return a ContinuousHV
    /// representing the search context (top matches bundled).
    ///
    /// The returned HV captures the "semantic neighborhood" of the query
    /// in physics-equation space, suitable for `cycle_with_hv()`.
    pub fn query_equation(
        &mut self,
        equation: &PhysicsEquation,
        max_results: usize,
    ) -> ContinuousHV {
        self.last_results = self.engine.search_equation(equation, max_results);
        self.query_count += 1;

        let hv = self.bundle_results();
        self.last_query_hv = Some(hv.clone());
        hv
    }

    /// Query by raw HV from the cognitive loop.
    ///
    /// Useful when the cognitive loop produces a thought_vector that
    /// should be matched against the physics catalog.
    pub fn query_hv(&mut self, query: &ContinuousHV, max_results: usize) -> ContinuousHV {
        self.last_results = self.engine.search_by_hv(query, max_results);
        self.query_count += 1;

        let hv = self.bundle_results();
        self.last_query_hv = Some(hv.clone());
        hv
    }

    /// Get the last search results.
    pub fn last_results(&self) -> &[SearchResult] {
        &self.last_results
    }

    /// Get telemetry about the last query.
    pub fn telemetry(&self) -> PhysicsBridgeTelemetry {
        PhysicsBridgeTelemetry {
            catalog_size: self.engine.catalog_size(),
            results_returned: self.last_results.len(),
            top_match: self.last_results.first().map(|r| r.name.clone()),
            top_score: self.last_results.first().map_or(0.0, |r| r.score),
        }
    }

    /// Access the underlying search engine.
    pub fn engine(&self) -> &PhysicsSearchEngine {
        &self.engine
    }

    /// Total number of queries performed.
    pub fn query_count(&self) -> u64 {
        self.query_count
    }

    /// Bundle top search results into a single ContinuousHV.
    ///
    /// Uses score-weighted bundling so higher-similarity matches
    /// contribute more to the output.
    fn bundle_results(&self) -> ContinuousHV {
        if self.last_results.is_empty() {
            return ContinuousHV::zero(HDC_DIMENSION);
        }

        // Get the catalog entries for each result
        let catalog = self.engine.catalog();
        let mut hvs = Vec::new();
        let mut weights = Vec::new();

        for result in &self.last_results {
            if let Some(entry) = catalog.find_by_name(&result.name) {
                hvs.push(&entry.full_hv);
                // Use score as weight (higher match → more influence)
                weights.push(result.score.max(0.01)); // Prevent zero weight
            }
        }

        if hvs.is_empty() {
            return ContinuousHV::zero(HDC_DIMENSION);
        }

        ContinuousHV::weighted_bundle(&hvs, &weights).normalize()
    }
}

impl Default for PhysicsBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::HDC_DIMENSION;

    #[test]
    fn bridge_creates_successfully() {
        let bridge = PhysicsBridge::new();
        assert!(bridge.engine().catalog_size() >= 27);
        assert_eq!(bridge.query_count(), 0);
    }

    #[test]
    fn query_equation_returns_valid_hv() {
        let mut bridge = PhysicsBridge::new();
        let eq = bridge
            .engine()
            .catalog()
            .find_by_name("Maxwell Gauss Law")
            .unwrap()
            .equation
            .clone();
        let hv = bridge.query_equation(&eq, 5);
        assert_eq!(hv.dim(), HDC_DIMENSION);
        assert!(hv.values.iter().all(|v| v.is_finite()));
        // Normalized output
        let norm: f32 = hv.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Expected normalized, got norm = {norm}"
        );
    }

    #[test]
    fn query_hv_returns_valid_hv() {
        let mut bridge = PhysicsBridge::new();
        let query = ContinuousHV::random(HDC_DIMENSION, 42);
        let hv = bridge.query_hv(&query, 5);
        assert_eq!(hv.dim(), HDC_DIMENSION);
        assert!(hv.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn telemetry_populated() {
        let mut bridge = PhysicsBridge::new();
        let eq = bridge
            .engine()
            .catalog()
            .find_by_name("Wave Equation")
            .unwrap()
            .equation
            .clone();
        bridge.query_equation(&eq, 3);

        let telemetry = bridge.telemetry();
        assert!(telemetry.catalog_size >= 27);
        assert_eq!(telemetry.results_returned, 3);
        assert!(telemetry.top_match.is_some());
        assert!(telemetry.top_score > 0.0);
    }

    #[test]
    fn query_count_increments() {
        let mut bridge = PhysicsBridge::new();
        assert_eq!(bridge.query_count(), 0);
        let query = ContinuousHV::random(HDC_DIMENSION, 42);
        bridge.query_hv(&query, 3);
        assert_eq!(bridge.query_count(), 1);
        bridge.query_hv(&query, 3);
        assert_eq!(bridge.query_count(), 2);
    }

    #[test]
    fn last_results_accessible() {
        let mut bridge = PhysicsBridge::new();
        let query = ContinuousHV::random(HDC_DIMENSION, 42);
        bridge.query_hv(&query, 5);
        let results = bridge.last_results();
        assert_eq!(results.len(), 5);
    }
}
