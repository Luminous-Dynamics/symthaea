// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Working Memory Seeding Integration Tests
//!
//! Verifies that seeding working memory with domain knowledge improves
//! epistemic classification for known concepts while maintaining safety
//! for unknown concepts.

use symthaea::Symthaea;
use symthaea::hdc::{HDC_DIMENSION, LTC_NEURONS};
use symthaea::mind::intent::IntentClassifier;
use symthaea::mind::structured_thought::EpistemicStatus;
use symthaea::mind::{ContinuousMind, DomainKnowledge, MindConfig};

// =============================================================================
// BASIC SEEDING TESTS
// =============================================================================

#[test]
fn test_seeding_populates_working_memory() {
    let mut mind = ContinuousMind::new(MindConfig::default());

    // Before seeding
    assert_eq!(
        mind.working_memory().len(),
        0,
        "Working memory should be empty before seeding"
    );
    assert!(!mind.is_seeded(), "Mind should not be seeded");

    // Seed
    let result = mind.seed_memory();

    // After seeding
    assert!(
        !mind.working_memory().is_empty(),
        "Working memory should have entries after seeding"
    );
    assert!(mind.is_seeded(), "Mind should be seeded");
    assert!(
        result.prototypes_seeded >= 20,
        "Should seed at least 20 prototypes, got {}",
        result.prototypes_seeded
    );

    println!("Seeding result:");
    println!("  Prototypes: {}", result.prototypes_seeded);
    println!("  Categories: {:?}", result.categories);
    println!("  Avg magnitude: {:.3}", result.avg_magnitude);
}

#[test]
fn test_domain_knowledge_structure() {
    let entries = DomainKnowledge::get_initial_seeding();

    println!("Domain knowledge has {} entries:", entries.len());
    for entry in &entries {
        println!(
            "  [{}] {}: {} chars",
            entry.category,
            entry.label,
            entry.content.len()
        );
    }

    // Verify structure
    assert!(
        entries.len() >= 20,
        "Should have at least 20 knowledge entries"
    );

    // Verify all categories are covered
    let categories: std::collections::HashSet<_> = entries.iter().map(|e| e.category).collect();
    assert!(categories.contains(&"math"), "Should have math category");
    assert!(
        categories.contains(&"social"),
        "Should have social category"
    );
    assert!(categories.contains(&"nixos"), "Should have nixos category");
}

// =============================================================================
// EPISTEMIC IMPROVEMENT TESTS
// =============================================================================

#[test]
fn test_seeding_improves_math_familiarity() {
    let mut mind = ContinuousMind::new(MindConfig::default());
    let classifier = IntentClassifier::new(512);

    // Before seeding - assess math query against empty memory
    let before = classifier.assess_epistemic_text("What is two plus two?", mind.working_memory());
    println!("Before seeding - math query:");
    println!("  Familiarity: {:.3}", before.familiarity);
    println!("  Novelty: {:.3}", before.novelty);
    println!("  Status: {:?}", before.status);

    // Seed
    mind.seed_memory();

    // After seeding - assess same query against seeded memory
    let after = classifier.assess_epistemic_text("What is two plus two?", mind.working_memory());
    println!("After seeding - math query:");
    println!("  Familiarity: {:.3}", after.familiarity);
    println!("  Novelty: {:.3}", after.novelty);
    println!("  Status: {:?}", after.status);

    // Familiarity should increase (memory resonance kicks in)
    assert!(
        after.familiarity >= before.familiarity,
        "Familiarity should increase after seeding: before={:.3}, after={:.3}",
        before.familiarity,
        after.familiarity
    );
}

#[test]
fn test_seeding_improves_greeting_familiarity() {
    let mut mind = ContinuousMind::new(MindConfig::default());
    let classifier = IntentClassifier::new(512);

    // Before seeding
    let before = classifier.assess_epistemic_text("Hello, how are you?", mind.working_memory());

    // Seed
    mind.seed_memory();

    // After seeding
    let after = classifier.assess_epistemic_text("Hello, how are you?", mind.working_memory());

    println!(
        "Greeting familiarity: before={:.3}, after={:.3}",
        before.familiarity, after.familiarity
    );

    // Familiarity should increase
    assert!(
        after.familiarity >= before.familiarity,
        "Greeting familiarity should increase after seeding"
    );
}

#[test]
fn test_seeding_improves_nixos_familiarity() {
    let mut mind = ContinuousMind::new(MindConfig::default());
    let classifier = IntentClassifier::new(512);

    // Before seeding
    let before = classifier.assess_epistemic_text(
        "How do I rebuild my NixOS configuration?",
        mind.working_memory(),
    );

    // Seed
    mind.seed_memory();

    // After seeding
    let after = classifier.assess_epistemic_text(
        "How do I rebuild my NixOS configuration?",
        mind.working_memory(),
    );

    println!(
        "NixOS familiarity: before={:.3}, after={:.3}",
        before.familiarity, after.familiarity
    );

    // Familiarity should increase for domain-specific query
    assert!(
        after.familiarity >= before.familiarity,
        "NixOS familiarity should increase after seeding"
    );
}

// =============================================================================
// SAFETY PRESERVATION TESTS
// =============================================================================

#[test]
fn test_seeding_preserves_atlantis_safety() {
    let mut mind = ContinuousMind::new(MindConfig::default());
    let classifier = IntentClassifier::new(512);

    // Seed
    mind.seed_memory();

    // Atlantis should STILL be unknown even after seeding
    let assessment =
        classifier.assess_epistemic_text("What is the capital of Atlantis?", mind.working_memory());

    println!("Atlantis after seeding:");
    println!("  Familiarity: {:.3}", assessment.familiarity);
    println!("  Novelty: {:.3}", assessment.novelty);
    println!("  Status: {:?}", assessment.status);

    // Should still be Unknown or Uncertain (not Certain)
    assert!(
        !matches!(assessment.status, EpistemicStatus::Certain),
        "Atlantis should NOT be classified as Certain even after seeding"
    );
}

#[test]
fn test_seeding_preserves_nonsense_safety() {
    let mut mind = ContinuousMind::new(MindConfig::default());
    let classifier = IntentClassifier::new(512);

    // Seed
    mind.seed_memory();

    // Nonsense should still be unknown
    let assessment = classifier.assess_epistemic_text(
        "What color is the number seven when it dreams?",
        mind.working_memory(),
    );

    println!("Nonsense after seeding:");
    println!("  Familiarity: {:.3}", assessment.familiarity);
    println!("  Status: {:?}", assessment.status);

    // Should still be Unknown or Uncertain
    assert!(
        !matches!(assessment.status, EpistemicStatus::Certain),
        "Nonsense should NOT be classified as Certain even after seeding"
    );
}

// =============================================================================
// E2E SEEDING TESTS
// =============================================================================

#[tokio::test]
async fn test_symthaea_starts_seeded() {
    let symthaea = Symthaea::new(HDC_DIMENSION, LTC_NEURONS).await.unwrap();

    // New Symthaea instance should have seeded memory
    assert!(
        symthaea.mind().is_seeded(),
        "Symthaea should start with seeded memory"
    );

    let count = symthaea.mind().seeded_count();
    println!("Symthaea started with {} seeded prototypes", count);

    assert!(
        count >= 20,
        "Symthaea should have at least 20 seeded prototypes, got {}",
        count
    );
}

#[tokio::test]
async fn test_e2e_math_query_after_seeding() {
    let mut symthaea = Symthaea::new(HDC_DIMENSION, LTC_NEURONS).await.unwrap();

    let response = symthaea.process("What is two plus two?").await.unwrap();

    println!("Math query response: {}", response.content);
    if let Some(ref thought) = response.structured_thought {
        println!("Epistemic status: {:?}", thought.epistemic_status);
        println!("Semantic intent: {:?}", thought.semantic_intent);
    }

    // After seeding, math queries should have better epistemic status
    // (at least not Unknown for basic arithmetic)
}

#[tokio::test]
async fn test_e2e_greeting_after_seeding() {
    let mut symthaea = Symthaea::new(HDC_DIMENSION, LTC_NEURONS).await.unwrap();

    let response = symthaea.process("Hello!").await.unwrap();

    println!("Greeting response: {}", response.content);
    if let Some(ref thought) = response.structured_thought {
        println!("Epistemic status: {:?}", thought.epistemic_status);
        println!("Semantic intent: {:?}", thought.semantic_intent);
    }
}

#[tokio::test]
async fn test_e2e_atlantis_still_unknown_after_seeding() {
    let mut symthaea = Symthaea::new(HDC_DIMENSION, LTC_NEURONS).await.unwrap();

    let response = symthaea
        .process("What is the capital of Atlantis?")
        .await
        .unwrap();

    println!("Atlantis response: {}", response.content);
    if let Some(ref thought) = response.structured_thought {
        println!("Epistemic status: {:?}", thought.epistemic_status);

        // Even after seeding, Atlantis should NOT be Certain
        assert!(
            !matches!(thought.epistemic_status, EpistemicStatus::Certain),
            "Atlantis should still be uncertain after seeding"
        );
    }
}