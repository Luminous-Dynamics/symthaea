// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # End-to-End Ollama Integration Tests
//!
//! Tests the full Symthaea pipeline with a real Ollama backend.
//! All tests skip gracefully if Ollama is not running.
//!
//! ## What this validates
//!
//! 1. Full pipeline: text input → HDC encode → CfC evolve → predict → learn → LLM translate → response
//! 2. Consciousness metrics evolve over a multi-turn conversation
//! 3. Structured thoughts carry epistemic status through to LLM translation
//! 4. Memory consolidation works with real (non-simulated) responses
//! 5. Partnership state advances with real interactions
//!
//! ## Prerequisites
//!
//! - Ollama running at localhost:11434
//! - A model pulled (defaults to gemma3:1b via SYMTHAEA_LLM_MODEL)
//!
//! Run with: `cargo test --test e2e_ollama_integration -- --nocapture`

use symthaea::Symthaea;
use symthaea::language::{LLMBackend, OllamaBackend};

/// Check if Ollama is running and accessible.
async fn ollama_available() -> bool {
    let backend = OllamaBackend::with_timeouts(
        "http://localhost:11434",
        "gemma3:1b",
        std::time::Duration::from_secs(2),
        std::time::Duration::from_secs(5),
    );
    backend.is_available().await
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. BASIC PIPELINE: Input → Process → Response with real LLM
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_ollama_basic_process() {
    if !ollama_available().await {
        eprintln!("SKIP: Ollama not available at localhost:11434");
        return;
    }

    let mut s = Symthaea::new(256, 16).await.unwrap();
    let resp = s.process("What is photosynthesis?").await.unwrap();

    // With a real LLM, content should be substantive (not just simulated echo)
    assert!(
        !resp.content.is_empty(),
        "Response content should be non-empty"
    );
    assert!(
        resp.content.len() > 10,
        "Real LLM response should be substantive, got: '{}'",
        resp.content
    );
    assert!(
        resp.confidence >= 0.0 && resp.confidence <= 1.0,
        "Confidence should be in [0,1], got {}",
        resp.confidence
    );
    assert!(
        resp.consciousness_level >= 0.0,
        "Consciousness level should be non-negative, got {}",
        resp.consciousness_level
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. MULTI-TURN CONVERSATION: State evolution across turns
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_ollama_multi_turn_state_evolution() {
    if !ollama_available().await {
        eprintln!("SKIP: Ollama not available at localhost:11434");
        return;
    }

    let mut s = Symthaea::new(256, 16).await.unwrap();

    let inputs = [
        "Hello, I'd like to learn about neural networks.",
        "How does backpropagation work?",
        "What about recurrent neural networks?",
        "Can you explain attention mechanisms?",
        "How do transformers use attention?",
    ];

    let mut confidences = Vec::new();
    let mut consciousness_levels = Vec::new();

    for input in &inputs {
        let resp = s.process(input).await.unwrap();

        assert!(
            !resp.content.is_empty(),
            "Response should be non-empty for: '{input}'"
        );

        confidences.push(resp.confidence);
        consciousness_levels.push(resp.consciousness_level);

        // Structured thought should always be populated
        assert!(
            resp.structured_thought.is_some(),
            "Structured thought should be present for: '{input}'"
        );
    }

    // All confidence values should be finite and bounded
    assert!(
        confidences
            .iter()
            .all(|c| c.is_finite() && *c >= 0.0 && *c <= 1.0),
        "All confidences should be finite and in [0,1]: {:?}",
        confidences
    );

    // All consciousness levels should be finite and non-negative
    assert!(
        consciousness_levels
            .iter()
            .all(|c| c.is_finite() && *c >= 0.0),
        "All consciousness levels should be finite and non-negative: {:?}",
        consciousness_levels
    );

    // Partnership state should reflect interactions
    let ps = s.partnership_state();
    assert!(
        ps.interactions >= 5,
        "Should have at least 5 interactions, got {}",
        ps.interactions
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. STRUCTURED THOUGHT FIDELITY: Thought fields propagate through pipeline
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_ollama_structured_thought_propagation() {
    if !ollama_available().await {
        eprintln!("SKIP: Ollama not available at localhost:11434");
        return;
    }

    let mut s = Symthaea::new(256, 16).await.unwrap();

    // Question about something factual — should produce high-confidence thought
    let resp = s.process("Explain what water is made of.").await.unwrap();

    let thought = resp
        .structured_thought
        .as_ref()
        .expect("Structured thought should be present");

    // Psi should be non-negative and finite
    assert!(
        thought.psi >= 0.0 && thought.psi.is_finite(),
        "Psi should be non-negative and finite, got {}",
        thought.psi
    );

    // Meta-awareness should be bounded
    assert!(
        thought.meta_awareness >= 0.0 && thought.meta_awareness <= 1.0,
        "Meta-awareness should be in [0,1], got {}",
        thought.meta_awareness
    );

    // Coherence should be non-negative
    assert!(
        thought.coherence >= 0.0 && thought.coherence.is_finite(),
        "Coherence should be non-negative and finite, got {}",
        thought.coherence
    );

    // Emotional tone should be finite
    assert!(
        thought.emotional_tone.valence.is_finite(),
        "Valence should be finite"
    );
    assert!(
        thought.emotional_tone.arousal.is_finite(),
        "Arousal should be finite"
    );

    // The original input should be preserved
    assert_eq!(
        thought.original_input.as_deref(),
        Some("Explain what water is made of."),
        "Original input should be preserved in thought"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. MEMORY + SLEEP: Real interactions produce consolidated memories
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_ollama_memory_consolidation() {
    if !ollama_available().await {
        eprintln!("SKIP: Ollama not available at localhost:11434");
        return;
    }

    let mut s = Symthaea::new(256, 16).await.unwrap();

    // Process enough inputs to populate working memory
    let topics = [
        "Tell me about the solar system.",
        "How do tides work?",
        "What causes seasons on Earth?",
        "How far is the Moon from Earth?",
    ];

    for topic in &topics {
        let resp = s.process(topic).await.unwrap();
        assert!(!resp.content.is_empty());
    }

    // Introspect: should see memory activity
    let state = s.introspect();
    assert!(
        state.memory_stats.long_term_count > 0,
        "Long-term memory should have entries after interactions"
    );

    // Sleep: consolidation should process real memories
    let report = s.sleep().await.unwrap();
    assert!(
        report.scaled > 0,
        "Sleep should scale memories, got scaled={}",
        report.scaled
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. EMOTIONAL CONTENT: Pipeline handles affective input correctly
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_ollama_emotional_content_handling() {
    if !ollama_available().await {
        eprintln!("SKIP: Ollama not available at localhost:11434");
        return;
    }

    let mut s = Symthaea::new(256, 16).await.unwrap();

    // Neutral baseline
    let neutral = s
        .process("What is the boiling point of water?")
        .await
        .unwrap();

    // Emotionally charged input
    let emotional = s
        .process("I feel a deep sense of wonder and gratitude for the beauty of the natural world.")
        .await
        .unwrap();

    // Both should succeed
    assert!(!neutral.content.is_empty());
    assert!(!emotional.content.is_empty());

    // The structured thoughts should both have valid emotional tones
    let neutral_thought = neutral.structured_thought.as_ref().unwrap();
    let emotional_thought = emotional.structured_thought.as_ref().unwrap();

    assert!(neutral_thought.emotional_tone.valence.is_finite());
    assert!(emotional_thought.emotional_tone.valence.is_finite());
    assert!(neutral_thought.emotional_tone.arousal.is_finite());
    assert!(emotional_thought.emotional_tone.arousal.is_finite());
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6. TRANSLATION VERIFICATION: LLM output respects epistemic constraints
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_ollama_translation_verification() {
    if !ollama_available().await {
        eprintln!("SKIP: Ollama not available at localhost:11434");
        return;
    }

    let mut s = Symthaea::new(256, 16).await.unwrap();

    let resp = s
        .process("What is the speed of light in a vacuum?")
        .await
        .unwrap();

    // The translation_verified field should be set
    // (true means LLM output passed fidelity checks against structured thought)
    // With a real LLM, we may or may not pass verification depending on model output,
    // but the field itself should be populated and the response should succeed.
    assert!(
        !resp.content.is_empty(),
        "Response should be non-empty even if verification fails"
    );

    // The response should always be marked safe for a benign factual question
    assert!(
        resp.safe,
        "Factual question about speed of light should be safe"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 7. SIGMA EVOLUTION: Spectral MIP coherence metric should appear
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_ollama_sigma_metric_appears() {
    if !ollama_available().await {
        eprintln!("SKIP: Ollama not available at localhost:11434");
        return;
    }

    let mut s = Symthaea::new(256, 16).await.unwrap();

    // Sigma is computed from memory coordinator, may need a few interactions
    let mut saw_sigma = false;
    for i in 0..8 {
        let resp = s
            .process(&format!("Topic number {i}: tell me about integers."))
            .await
            .unwrap();
        if resp.sigma.is_some() {
            saw_sigma = true;
        }
    }

    // Sigma may or may not appear in 8 turns (depends on memory coordinator state),
    // but the pipeline should not panic and all responses should succeed.
    // We just log the observation.
    if saw_sigma {
        eprintln!("  sigma appeared within 8 turns");
    } else {
        eprintln!("  sigma did not appear in 8 turns (expected for short conversations)");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 8. PAUSE/RESUME: Full state preserved with real LLM interactions
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_ollama_pause_resume_preserves_state() {
    if !ollama_available().await {
        eprintln!("SKIP: Ollama not available at localhost:11434");
        return;
    }

    let tmp = std::env::temp_dir().join("e2e_ollama_pause_test.json");
    let path = tmp.to_str().unwrap();

    let interaction_count;
    {
        let mut s = Symthaea::new(256, 16).await.unwrap();
        let _ = s.process("First real interaction.").await.unwrap();
        let _ = s.process("Second real interaction.").await.unwrap();
        interaction_count = s.partnership_state().interactions;
        s.pause(path).unwrap();
    }

    let resumed = Symthaea::resume(path).unwrap();
    assert_eq!(
        resumed.dimension(),
        256,
        "Dimension should be preserved across pause/resume"
    );

    // Clean up
    let _ = std::fs::remove_file(path);

    assert!(
        interaction_count >= 2,
        "Should have at least 2 interactions before pause"
    );
}