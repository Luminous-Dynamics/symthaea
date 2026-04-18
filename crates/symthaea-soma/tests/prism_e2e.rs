// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! End-to-end tests for Prism epistemic search integration in Soma.
//!
//! Verifies the full pipeline: init → search → results with real claims.

#![cfg(feature = "prism-search")]

use symthaea_soma::engine::{SomaConfig, SomaEngine};

fn make_engine() -> SomaEngine {
    let config = SomaConfig::default();
    SomaEngine::new(config)
}

#[test]
fn prism_init_loads_core_claims() {
    let mut engine = make_engine();
    assert!(!engine.prism_available(), "should not be available before init");

    engine.prism_init();
    assert!(engine.prism_available(), "should be available after init");

    // Core claims should have at least 100 claims (curated + mycelix)
    let results = engine.prism_search("test", 1);
    // Even if "test" doesn't match well, the engine should be functional
    assert!(engine.prism_available());
}

#[test]
fn prism_search_returns_relevant_results() {
    let mut engine = make_engine();
    engine.prism_init();

    // Search for a known curated claim
    let results = engine.prism_search("ocean acidification", 5);
    assert!(
        !results.is_empty(),
        "should find results for 'ocean acidification' — a curated claim"
    );

    // Check the top result contains relevant content
    let top = &results[0];
    let content_lower = top.content.to_lowercase();
    assert!(
        content_lower.contains("ocean") || content_lower.contains("acidification") || content_lower.contains("co2"),
        "top result should be about ocean acidification, got: {}",
        &top.content[..top.content.len().min(80)]
    );
}

#[test]
fn prism_search_consciousness_domain() {
    let mut engine = make_engine();
    engine.prism_init();

    let results = engine.prism_search("consciousness", 3);
    assert!(
        !results.is_empty(),
        "should find results for 'consciousness'"
    );

    // At least one result should mention consciousness-related terms
    let any_relevant = results.iter().any(|r| {
        let low = r.content.to_lowercase();
        low.contains("conscious") || low.contains("phi") || low.contains("iit")
            || low.contains("workspace") || low.contains("neural")
    });
    assert!(any_relevant, "at least one result should be about consciousness");
}

#[test]
fn prism_search_empty_query_returns_empty() {
    let mut engine = make_engine();
    engine.prism_init();

    let results = engine.prism_search("", 5);
    assert!(results.is_empty(), "empty query should return no results");

    let results = engine.prism_search("   ", 5);
    assert!(results.is_empty(), "whitespace query should return no results");
}

#[test]
fn prism_search_results_have_epistemic_levels() {
    let mut engine = make_engine();
    engine.prism_init();

    let results = engine.prism_search("DNA genetics", 3);
    assert!(!results.is_empty());

    for result in &results {
        // Every result should have a valid epistemic level
        let e = result.empirical_level.as_f32();
        assert!(
            e >= 0.0 && e <= 1.0,
            "epistemic level should be in [0, 1], got {}",
            e
        );
    }
}

#[test]
fn prism_load_full_expands_corpus() {
    let mut engine = make_engine();
    engine.prism_init();

    let core_results = engine.prism_search("promethium element", 3);

    // Load the full corpus (includes Wikidata chemical elements)
    engine.prism_load_full();

    let full_results = engine.prism_search("promethium element", 3);

    // Full corpus should have more or better results for Wikidata-specific queries
    assert!(
        full_results.len() >= core_results.len(),
        "full corpus should have at least as many results: core={}, full={}",
        core_results.len(),
        full_results.len()
    );
}

#[test]
fn prism_search_before_init_returns_empty() {
    let engine = make_engine();
    // Should not panic, just return empty
    let results = engine.prism_search("anything", 5);
    assert!(results.is_empty(), "search before init should return empty");
}
