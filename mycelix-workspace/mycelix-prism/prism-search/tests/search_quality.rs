// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Search quality integration tests.
//!
//! These tests verify that the epistemic search engine returns
//! relevant results for various query types.

use prism_search::{encode_text, SearchEngine};

fn engine() -> SearchEngine {
    SearchEngine::with_seed_claims()
}

// ── Relevance tests: top result must be on-topic ──

#[test]
fn search_ocean_acidification_top_result() {
    let results = engine().search("ocean acidification", 5);
    assert!(!results.is_empty());
    let top = results[0].content.to_lowercase();
    assert!(
        top.contains("ocean")
            || top.contains("acidification")
            || top.contains("co2")
            || top.contains("ph"),
        "Top result for 'ocean acidification' should be about oceans/CO2, got: {}",
        results[0].content
    );
}

#[test]
fn search_rust_programming_top_result() {
    let results = engine().search("rust programming language", 5);
    assert!(!results.is_empty());
    let top = results[0].content.to_lowercase();
    assert!(
        top.contains("rust"),
        "Top result for 'rust programming' should mention Rust, got: {}",
        results[0].content
    );
}

#[test]
fn search_consciousness_top_result() {
    let results = engine().search("consciousness awareness", 5);
    assert!(!results.is_empty());
    let top = results[0].content.to_lowercase();
    assert!(
        top.contains("consciousness")
            || top.contains("awareness")
            || top.contains("iit")
            || top.contains("global workspace"),
        "Top result for 'consciousness' should be relevant, got: {}",
        results[0].content
    );
}

#[test]
fn search_quantum_physics_top_result() {
    let results = engine().search("quantum physics entanglement", 5);
    assert!(!results.is_empty());
    let top = results[0].content.to_lowercase();
    assert!(
        top.contains("quantum") || top.contains("entanglement") || top.contains("heisenberg"),
        "Top result for 'quantum physics' should be relevant, got: {}",
        results[0].content
    );
}

#[test]
fn search_dna_genetics_top_result() {
    let results = engine().search("DNA genetics double helix", 5);
    assert!(!results.is_empty());
    let top = results[0].content.to_lowercase();
    assert!(
        top.contains("dna") || top.contains("gene") || top.contains("helix"),
        "Top result for 'DNA genetics' should be relevant, got: {}",
        results[0].content
    );
}

#[test]
fn search_climate_change_top_result() {
    let results = engine().search("climate change greenhouse gas", 5);
    assert!(!results.is_empty());
    let top = results[0].content.to_lowercase();
    assert!(
        top.contains("climate")
            || top.contains("greenhouse")
            || top.contains("co2")
            || top.contains("temperature"),
        "Top result for 'climate change' should be relevant, got: {}",
        results[0].content
    );
}

#[test]
fn search_solar_energy_top_result() {
    let results = engine().search("solar energy photovoltaic", 5);
    assert!(!results.is_empty());
    let top = results[0].content.to_lowercase();
    assert!(
        top.contains("solar") || top.contains("photovoltaic") || top.contains("energy"),
        "Top result for 'solar energy' should be relevant, got: {}",
        results[0].content
    );
}

// ── Encoding quality tests ──

#[test]
fn encoding_same_meaning_high_similarity() {
    let hv1 = encode_text("ocean acidification causes");
    let hv2 = encode_text("causes of ocean acidification");
    let sim = hv1.similarity(&hv2);
    assert!(sim > 0.5, "Reordered words should be similar: {}", sim);
}

#[test]
fn encoding_synonyms_moderate_similarity() {
    let hv1 = encode_text("rust memory safe programming");
    let hv2 = encode_text("rust safe memory programming language");
    let sim = hv1.similarity(&hv2);
    assert!(sim > 0.5, "Overlapping words should be similar: {}", sim);
}

#[test]
fn encoding_unrelated_low_similarity() {
    let hv1 = encode_text("quantum entanglement physics");
    let hv2 = encode_text("chocolate cake recipe baking");
    let sim = hv1.similarity(&hv2);
    assert!(sim < 0.52, "Unrelated topics should be dissimilar: {}", sim);
}

// ── Ranking quality tests ──

#[test]
fn ranking_relevant_above_irrelevant() {
    let results = engine().search("programming language memory safety", 10);
    if results.len() >= 2 {
        // First result should have higher score than last
        assert!(
            results[0].rank_score() > results.last().unwrap().rank_score(),
            "First result should rank higher than last"
        );
    }
}

#[test]
fn ranking_scores_decrease() {
    let results = engine().search("ocean temperature climate", 10);
    for i in 1..results.len() {
        assert!(
            results[i].rank_score() <= results[i - 1].rank_score() + 0.001,
            "Results should be sorted by decreasing score at positions {} and {}",
            i - 1,
            i
        );
    }
}
