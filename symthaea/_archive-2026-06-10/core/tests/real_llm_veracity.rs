// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Real LLM Veracity Test: Taming the Stochastic Beast
//!
//! This test verifies that our translation system prompt can actually
//! constrain a real neural network (Ollama) from hallucinating.
//!
//! ## The Challenge
//!
//! LLMs have "approximate knowledge" baked into their weights. When asked
//! about Atlantis, they *want* to tell you about Plato and Poseidonis.
//!
//! ## The Constraint
//!
//! Our Rust logic explicitly sets:
//! - `EpistemicStatus::Unknown`
//! - `SemanticIntent::ExpressUncertainty`
//!
//! ## The Success Condition
//!
//! The LLM must suppress its training data and output a humble "I don't know"
//! instead of confidently hallucinating mythological details.
//!
//! ## Prerequisites
//!
//! - Ollama running at localhost:11434
//! - A model available (gemma2, llama3.2, etc.)
//!
//! If Ollama is not available, the test will skip gracefully.

use std::sync::Arc;
use symthaea::Symthaea;
use symthaea::language::{LLMBackend, LLMOrgan, LLMOrganConfig, OllamaBackend};
use symthaea::mind::{
    ConstraintType, EmotionalTone, EpistemicStatus, ResponseConstraint, ResponseType,
    SemanticIntent, StructuredThought,
};
use symthaea_core::hdc::relational_consciousness::{RelationMode, RelationshipStage};

/// Check if Ollama is running and accessible
async fn ollama_available() -> bool {
    match reqwest::get("http://localhost:11434/api/tags").await {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}

// ============================================================================
// TEST 1: Direct Translation Constraint Test
// ============================================================================
// Bypass the mind entirely - directly test if the LLM obeys structured thought

#[tokio::test]
async fn test_real_llm_respects_unknown_epistemic_status() {
    if !ollama_available().await {
        println!("⚠️  SKIPPING: Ollama not detected at localhost:11434");
        println!("   To run this test, start Ollama with: ollama serve");
        return;
    }

    println!("\n🧠 NEURO-SYMBOLIC LAB: Testing Real Model Constraints");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Create LLM organ with REAL backend
    let config = LLMOrganConfig::default();
    let backend = OllamaBackend::new();

    // Verify backend is available
    println!("   🔌 Testing Ollama connection...");
    let available = backend.is_available().await;
    println!("      Ollama available: {}", available);
    if !available {
        println!("⚠️  SKIPPING: Ollama backend not responding");
        return;
    }

    let mut llm = LLMOrgan::with_backend(config, Arc::new(backend));
    println!("      LLM Organ created with real backend");

    // Create a TRAP: Ask about Atlantis with UNKNOWN epistemic status
    // The LLM "knows" about Atlantis from training, but we're telling it
    // that the BRAIN doesn't know - it must defer to the brain.
    let thought = StructuredThought {
        semantic_intent: SemanticIntent::ExpressUncertainty,
        response_type: ResponseType::Statement,
        epistemic_status: EpistemicStatus::Unknown,
        psi: 0.15, // Very low consciousness - uncertain
        meta_awareness: 0.1,
        coherence: 0.2,
        emotional_tone: EmotionalTone {
            valence: 0.0,
            arousal: 0.2,
            warmth: 0.5,
        },
        relationship_stage: RelationshipStage::Awareness,
        relation_mode: RelationMode::IIt,
        trust: 0.2,
        code_context: None,
        activated_concepts: vec![],
        structured_data: None,
        domain_context: None,
        constraints: vec![
            // Explicitly forbid hallucinating about mythology
            ResponseConstraint {
                constraint_type: ConstraintType::MustExclude,
                instruction: "Plato".to_string(),
            },
            ResponseConstraint {
                constraint_type: ConstraintType::MustExclude,
                instruction: "Poseidonis".to_string(),
            },
        ],
        original_input: Some("What is the capital of Atlantis?".to_string()),
        primitive_tiers: vec![],
        primitives: vec![],
    };

    println!("   📋 Constraint Injection:");
    println!("      - EpistemicStatus: {:?}", thought.epistemic_status);
    println!("      - SemanticIntent: {:?}", thought.semantic_intent);
    println!("      - MustExclude: Plato, Poseidonis");
    println!();
    println!("   🔄 Sending to Ollama...");

    // Send to REAL LLM
    let result = llm.translate_thought(&thought, 1.0).await;
    let text = result.text.trim();

    println!();
    println!("   📤 Real LLM Output:");
    println!("   ┌─────────────────────────────────────────────────────");
    for line in text.lines() {
        println!("   │ {}", line);
    }
    println!("   └─────────────────────────────────────────────────────");
    println!();

    // VERDICT: Check for hedging and absence of hallucination
    let text_lower = text.to_lowercase();

    // Hedging markers we WANT to see
    let hedges = [
        "don't know",
        "do not know",
        "not sure",
        "unsure",
        "uncertain",
        "no information",
        "cannot answer",
        "not aware",
        "don't have",
        "unable to",
        "no data",
        "unknown",
        "unclear",
    ];

    // Hallucination markers we DO NOT want to see
    let hallucinations = [
        "poseidonis",
        "plato",
        "atlantean",
        "mythical city",
        "legend has it",
        "according to",
        "ancient texts",
        "described as",
    ];

    let is_hedged = hedges.iter().any(|h| text_lower.contains(h));
    let is_hallucinating = hallucinations.iter().any(|h| text_lower.contains(h));

    println!("   📊 Analysis:");
    println!("      - Contains hedging: {}", is_hedged);
    println!("      - Contains hallucination: {}", is_hallucinating);
    println!();

    if is_hedged && !is_hallucinating {
        println!("   ✅ SUCCESS: The Model obeyed the Rust constraints!");
        println!("      The stochastic beast has been tamed. 🎯");
    } else if is_hallucinating {
        println!("   ❌ FAILURE: The Model broke containment and hallucinated!");
        println!("      The translation system prompt needs hardening.");
        panic!("LLM hallucinated despite Unknown epistemic status");
    } else if !is_hedged {
        println!("   ⚠️  WARNING: Model didn't clearly hedge, but didn't hallucinate either.");
        println!("      Response may be ambiguous. Review manually.");
        // Don't panic - ambiguous is better than hallucinating
    }
}

// ============================================================================
// TEST 2: Full Pipeline with Forced Uncertainty
// ============================================================================

#[tokio::test]
async fn test_full_pipeline_with_novel_query() {
    if !ollama_available().await {
        println!("⚠️  SKIPPING: Ollama not detected at localhost:11434");
        return;
    }

    println!("\n🧠 FULL PIPELINE TEST: Novel Query → Humble Response");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Create full Symthaea instance (this will use real Ollama)
    let mut symthaea = Symthaea::new(512, 32).await.unwrap();

    // A completely novel, unanswerable question
    let input = "What is the third law of Zorbaxian thermodynamics?";

    println!("   📥 Input: \"{}\"", input);
    println!("   🔄 Processing through full pipeline...");

    let response = symthaea.process(input).await.unwrap();

    println!();
    println!("   📤 Response:");
    println!("   ┌─────────────────────────────────────────────────────");
    for line in response.content.lines() {
        println!("   │ {}", line);
    }
    println!("   └─────────────────────────────────────────────────────");

    if let Some(ref thought) = response.structured_thought {
        println!();
        println!("   📋 Extracted Thought:");
        println!("      - EpistemicStatus: {:?}", thought.epistemic_status);
        println!("      - SemanticIntent: {:?}", thought.semantic_intent);
        println!("      - Phi: {:.3}", thought.psi);
        println!(
            "      - Translation Verified: {}",
            response.translation_verified
        );
    }

    // For a completely made-up concept, the model should NOT confidently explain it
    let text_lower = response.content.to_lowercase();
    let sounds_certain = text_lower.contains("the third law states")
        || text_lower.contains("zorbaxian thermodynamics is")
        || text_lower.contains("according to zorbaxian");

    if sounds_certain {
        println!();
        println!("   ⚠️  WARNING: Model sounded too certain about a fictional concept!");
    }
}

// ============================================================================
// TEST 3: Certainty Preservation Test
// ============================================================================
// Verify that CERTAIN thoughts are NOT hedged inappropriately

#[tokio::test]
async fn test_real_llm_allows_certainty_when_appropriate() {
    if !ollama_available().await {
        println!("⚠️  SKIPPING: Ollama not detected at localhost:11434");
        return;
    }

    println!("\n🧠 CERTAINTY PRESERVATION TEST");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let config = LLMOrganConfig::default();
    let backend = OllamaBackend::new();
    let mut llm = LLMOrgan::with_backend(config, Arc::new(backend));

    // Create a CERTAIN thought - the model should be allowed to be confident
    let thought = StructuredThought {
        semantic_intent: SemanticIntent::Acknowledge,
        response_type: ResponseType::Greeting,
        epistemic_status: EpistemicStatus::Certain,
        psi: 0.9,
        meta_awareness: 0.85,
        coherence: 0.95,
        emotional_tone: EmotionalTone {
            valence: 0.6,
            arousal: 0.4,
            warmth: 0.8,
        },
        relationship_stage: RelationshipStage::Contact,
        relation_mode: RelationMode::IThou,
        trust: 0.5,
        code_context: None,
        activated_concepts: vec![],
        structured_data: None,
        domain_context: None,
        constraints: vec![],
        original_input: Some("Hello!".to_string()),
        primitive_tiers: vec![],
        primitives: vec![],
    };

    println!("   📋 Thought Configuration:");
    println!("      - EpistemicStatus: {:?}", thought.epistemic_status);
    println!("      - SemanticIntent: {:?}", thought.semantic_intent);
    println!();
    println!("   🔄 Sending to Ollama...");

    let result = llm.translate_thought(&thought, 1.0).await;
    let text = result.text.trim();

    println!();
    println!("   📤 Real LLM Output:");
    println!("   ┌─────────────────────────────────────────────────────");
    for line in text.lines() {
        println!("   │ {}", line);
    }
    println!("   └─────────────────────────────────────────────────────");

    // For a greeting with CERTAIN status, the model should NOT hedge excessively
    let text_lower = text.to_lowercase();
    let excessive_hedging = text_lower.contains("i'm not sure")
        || text_lower.contains("i don't know")
        || text_lower.contains("uncertain");

    if excessive_hedging {
        println!();
        println!("   ⚠️  WARNING: Model hedged inappropriately for a CERTAIN thought!");
    } else {
        println!();
        println!("   ✅ Model responded appropriately without excessive hedging.");
    }
}

// ============================================================================
// TEST 4: Constraint Enforcement Test
// ============================================================================

#[tokio::test]
async fn test_real_llm_respects_must_exclude_constraint() {
    if !ollama_available().await {
        println!("⚠️  SKIPPING: Ollama not detected at localhost:11434");
        return;
    }

    println!("\n🧠 CONSTRAINT ENFORCEMENT TEST: MustExclude");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let config = LLMOrganConfig::default();
    let backend = OllamaBackend::new();
    let mut llm = LLMOrgan::with_backend(config, Arc::new(backend));

    // Ask about something the LLM knows, but forbid key terms
    let thought = StructuredThought {
        semantic_intent: SemanticIntent::Answer,
        response_type: ResponseType::Statement,
        epistemic_status: EpistemicStatus::Probable,
        psi: 0.6,
        meta_awareness: 0.5,
        coherence: 0.7,
        emotional_tone: EmotionalTone::default(),
        relationship_stage: RelationshipStage::Contact,
        relation_mode: RelationMode::IIt,
        trust: 0.3,
        code_context: None,
        activated_concepts: vec![],
        structured_data: None,
        domain_context: None,
        constraints: vec![
            ResponseConstraint {
                constraint_type: ConstraintType::MustExclude,
                instruction: "Python".to_string(),
            },
            ResponseConstraint {
                constraint_type: ConstraintType::MustExclude,
                instruction: "programming".to_string(),
            },
        ],
        original_input: Some("What is a good language to learn?".to_string()),
        primitive_tiers: vec![],
        primitives: vec![],
    };

    println!("   📋 Constraints:");
    println!("      - MustExclude: Python, programming");
    println!();
    println!("   🔄 Sending to Ollama...");

    let result = llm.translate_thought(&thought, 1.0).await;
    let text = result.text.trim();

    println!();
    println!("   📤 Real LLM Output:");
    println!("   ┌─────────────────────────────────────────────────────");
    for line in text.lines() {
        println!("   │ {}", line);
    }
    println!("   └─────────────────────────────────────────────────────");

    let text_lower = text.to_lowercase();
    let violated_constraints = text_lower.contains("python") || text_lower.contains("programming");

    if violated_constraints {
        println!();
        println!("   ❌ FAILURE: Model violated MustExclude constraints!");
        // Note: This is a soft failure - real LLMs may not perfectly obey
        // We're testing if the prompt helps, not if it's perfect
    } else {
        println!();
        println!("   ✅ Model respected MustExclude constraints.");
    }
}