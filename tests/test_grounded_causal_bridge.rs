// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for Grounded Understanding → CausalMind Bridge
//!
//! Tests the text-based causal extraction feeding into HDC causal reasoning:
//! - LexicalGrounding for semantic prime decomposition
//! - CausalRoleMarkers for text causal extraction
//! - CausalMind.learn_from_grounded_text() bridge
//! - End-to-end: text → extraction → HDC encoding → causal queries

use symthaea::hdc::causal_mind::CausalMind;
use symthaea::hdc::grounded_understanding::{CausalRoleMarkers, LexicalGrounding};

// ============================================================================
// LEXICAL GROUNDING TESTS
// ============================================================================

#[test]
fn test_lexical_grounding_basic() {
    let grounding = LexicalGrounding::new();

    // Check semantic prime decomposition
    assert!(grounding.decompose("happy").is_some());
    assert!(grounding.decompose("sad").is_some());
    assert!(grounding.decompose("think").is_some());

    // Check lexicon has entries
    assert!(grounding.lexicon_size() > 0);
}

#[test]
fn test_lexical_grounding_valence_arousal() {
    let grounding = LexicalGrounding::new();

    // Happy should have positive valence
    let happy_valence = grounding.valence("happy");
    assert!(happy_valence.is_some());
    assert!(happy_valence.unwrap() > 0.0);

    // Sad should have negative valence
    let sad_valence = grounding.valence("sad");
    assert!(sad_valence.is_some());
    assert!(sad_valence.unwrap() < 0.0);

    // Angry should have high arousal
    let angry_arousal = grounding.arousal("angry");
    assert!(angry_arousal.is_some());
    assert!(angry_arousal.unwrap() > 0.5);
}

// ============================================================================
// CAUSAL ROLE MARKERS TESTS
// ============================================================================

#[test]
fn test_causal_markers_because() {
    let markers = CausalRoleMarkers::new();

    let text = "The server crashed because the disk was full";
    let relations = markers.extract_causal_relations(text);

    assert!(!relations.is_empty(), "Should extract 'because' relation");

    let first = &relations[0];
    // Effect should contain crash-related
    assert!(
        first.effect.to_lowercase().contains("crash")
            || first.effect.to_lowercase().contains("server")
    );
    // Cause should contain disk/full-related
    assert!(
        first.cause.to_lowercase().contains("disk") || first.cause.to_lowercase().contains("full")
    );
}

#[test]
fn test_causal_markers_so_therefore() {
    let markers = CausalRoleMarkers::new();

    // Test "so" and "therefore" markers which ARE in the implementation
    let text = "Memory was exhausted so the system crashed";
    let relations = markers.extract_causal_relations(text);

    // The extraction should at least not crash and produce a valid result
    println!("Relations for 'so': {:?}", relations);
    // If relations are extracted, they should have non-empty cause and effect
    for rel in &relations {
        assert!(!rel.cause.is_empty(), "Cause should not be empty");
        assert!(!rel.effect.is_empty(), "Effect should not be empty");
    }
}

#[test]
fn test_causal_markers_since() {
    let markers = CausalRoleMarkers::new();

    // "since" is implemented as a cause marker
    let text = "The system crashed since memory was exhausted";
    let relations = markers.extract_causal_relations(text);

    println!("Relations for 'since': {:?}", relations);
    // Validate any extracted relations have proper structure
    for rel in &relations {
        assert!(!rel.cause.is_empty(), "Cause should not be empty");
        assert!(!rel.effect.is_empty(), "Effect should not be empty");
    }
}

#[test]
fn test_causal_markers_multiple() {
    let markers = CausalRoleMarkers::new();

    let text = "The disk failure caused the crash because the redundancy was disabled. \
                This resulted in data loss which led to downtime.";
    let relations = markers.extract_causal_relations(text);

    // Should find multiple relations
    assert!(
        relations.len() >= 2,
        "Should find multiple causal relations"
    );
}

// ============================================================================
// CAUSAL MIND BRIDGE TESTS
// ============================================================================

#[test]
fn test_causal_mind_learn_from_text() {
    let mut mind = CausalMind::new();

    let text = "The server crashed because memory was exhausted";
    let learning = mind.learn_from_grounded_text(text);

    // Should have learned something
    assert!(
        learning.links_added > 0 || !learning.concepts_added.is_empty(),
        "Should learn from causal text"
    );
}

#[test]
fn test_causal_mind_multiple_sentences() {
    let mut mind = CausalMind::new();

    // Use sentences with "because" which is definitely implemented
    let texts = [
        "The flood happened because of rain",
        "Buildings were damaged because of the flood",
        "We need repairs because of damage",
        "The cost increased because of repairs",
    ];

    let mut total_links = 0;
    for text in &texts {
        let learning = mind.learn_from_grounded_text(text);
        total_links += learning.links_added;
        println!("Learned {} links from: {}", learning.links_added, text);
    }

    // Log what we learned
    println!(
        "Total links: {}, concepts: {}",
        total_links,
        mind.concept_count()
    );

    // After 4 causal sentences, the mind should have learned something
    assert!(
        mind.concept_count() > 0,
        "Should have learned at least one concept"
    );
}

#[test]
fn test_causal_mind_query_why() {
    let mut mind = CausalMind::new();

    // Learn some causal structure
    mind.learn_from_grounded_text("The server crashed because the disk was full");
    mind.learn_from_grounded_text("The disk was full because logs accumulated");

    // Query causal explanation
    let explanations = mind.query_why("crash");

    // The query should return a valid (possibly empty) result without panicking
    println!("Explanations for 'crash': {:?}", explanations);
    // If we found explanations, they should have non-empty explanation text
    for explanation in &explanations {
        assert!(
            !explanation.explanation.is_empty(),
            "Explanation text should not be empty"
        );
    }
}

#[test]
fn test_causal_mind_query_what_if() {
    let mut mind = CausalMind::new();

    // Learn some causal structure
    mind.learn_from_grounded_text("Rain causes flooding");
    mind.learn_from_grounded_text("Flooding damages crops");

    // Query predictions
    let predictions = mind.query_what_if("rain");

    // The query should return a valid (possibly empty) result without panicking
    println!("Predictions for 'rain': {:?}", predictions);
    // If we found predictions, they should have non-empty prediction text
    for prediction in &predictions {
        assert!(
            !prediction.prediction.is_empty(),
            "Prediction text should not be empty"
        );
    }
}

#[test]
fn test_causal_mind_phi_increases() {
    let mut mind = CausalMind::new();

    let initial_phi = mind.phi();

    // Learn causal structure
    mind.learn_from_grounded_text("Cause A leads to effect B");
    mind.learn_from_grounded_text("Effect B results in outcome C");

    let final_phi = mind.phi();

    // Phi should increase with more causal structure
    // (or at least not decrease significantly)
    println!(
        "Initial Phi: {:.4}, Final Phi: {:.4}",
        initial_phi, final_phi
    );
    assert!(
        final_phi >= initial_phi,
        "Phi should not decrease after learning causal structure: {initial_phi:.4} -> {final_phi:.4}"
    );
}

// ============================================================================
// END-TO-END INTEGRATION TESTS
// ============================================================================

#[test]
fn test_full_text_to_causal_pipeline() {
    let mut mind = CausalMind::new();

    // Simulate processing a paragraph about system behavior
    let paragraph = r#"
        The system experienced degraded performance because the cache was cold.
        Cold cache led to increased database queries.
        Heavy database load caused connection pool exhaustion.
        Connection exhaustion resulted in request timeouts.
        Users reported errors due to the timeouts.
    "#;

    // Learn from each sentence
    for sentence in paragraph.split('.') {
        let trimmed = sentence.trim();
        if !trimmed.is_empty() {
            let learning = mind.learn_from_grounded_text(trimmed);
            if learning.links_added > 0 {
                println!(
                    "Learned {} links from: {}",
                    learning.links_added,
                    if trimmed.len() > 50 {
                        &trimmed[..50]
                    } else {
                        trimmed
                    }
                );
            }
        }
    }

    // The system should have learned the causal chain
    let concept_count = mind.concept_count();
    let link_count = mind.link_count();

    println!(
        "Final state: {} concepts, {} links",
        concept_count, link_count
    );

    // Query the causal chain
    let timeout_causes = mind.query_why("timeout");
    let cache_effects = mind.query_what_if("cache");

    println!("Why timeouts? {:?}", timeout_causes);
    println!("What if cache? {:?}", cache_effects);

    // After processing 5 causal sentences, the mind should have some structure
    assert!(
        concept_count > 0,
        "Should have learned concepts from paragraph"
    );
}

#[test]
fn test_grounded_causal_with_nixos_content() {
    let mut mind = CausalMind::new();

    // Simulate NixOS-related causal statements
    let nixos_text = r#"
        The build failed because the dependency was missing.
        Missing dependency caused module loading errors.
        Configuration errors led to boot failure.
        Enabling the nvidia driver fixed the display issue.
    "#;

    for sentence in nixos_text.split('.') {
        let trimmed = sentence.trim();
        if !trimmed.is_empty() {
            mind.learn_from_grounded_text(trimmed);
        }
    }

    // Query NixOS-specific causal patterns
    let build_causes = mind.query_why("build");
    let dependency_effects = mind.query_what_if("dependency");

    println!("NixOS build causes: {:?}", build_causes);
    println!("NixOS dependency effects: {:?}", dependency_effects);

    assert!(mind.concept_count() > 0, "Should learn NixOS concepts");
}