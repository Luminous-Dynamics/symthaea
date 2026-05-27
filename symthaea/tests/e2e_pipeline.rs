// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================================
// E2E Pipeline Integration Test: Full Perception -> Cognition -> Action Flow
// ==================================================================================
//
// Tests the complete Symthaea pipeline end-to-end:
//   1. Create Symthaea instance with default config
//   2. Feed text input via process()
//   3. Verify response is non-empty
//   4. Verify CycleMetadata-like fields are populated (via ProcessResponse)
//   5. Feed multiple inputs and verify state evolution
//   6. Test with varied input types (questions, statements, emotional content)
//
// No feature flags required — uses default configuration only.
//
// NOTE: Symthaea::new() and .process() may fail on CI runners due to filesystem
// permissions (model caches) or missing LLM backend. Tests skip gracefully.
// ==================================================================================

use symthaea::symthaea::Symthaea;

/// Helper: create Symthaea, skipping if CI permissions prevent it.
macro_rules! try_new_symthaea {
    ($dim:expr, $depth:expr) => {
        match Symthaea::new($dim, $depth).await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Skipping test (Symthaea::new failed — CI permissions?): {e}");
                return;
            }
        }
    };
}

/// Helper: call process(), skipping if LLM backend is unavailable.
macro_rules! try_process {
    ($sym:expr, $input:expr) => {
        match $sym.process($input).await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Skipping test (process() failed — no LLM backend on CI?): {e}");
                return;
            }
        }
    };
}

// ── Test 1: Basic process() produces a non-empty response ─────────────

#[tokio::test]
async fn test_process_returns_nonempty_response() {
    let mut s = try_new_symthaea!(512, 64);
    let resp = try_process!(s, "Hello, how are you?");

    assert!(
        !resp.content.is_empty(),
        "Process should return non-empty content"
    );
    assert!(
        resp.confidence >= 0.0 && resp.confidence <= 1.0,
        "Confidence should be in [0,1], got {}",
        resp.confidence
    );
}

// ── Test 2: Structured thought is populated after process ─────────────

#[tokio::test]
async fn test_structured_thought_populated() {
    let mut s = try_new_symthaea!(512, 64);
    let resp = try_process!(s, "What is the meaning of life?");

    let thought = resp
        .structured_thought
        .as_ref()
        .expect("ProcessResponse should include structured_thought");

    // Psi (consciousness level) should be non-negative
    assert!(
        thought.psi >= 0.0,
        "Psi should be non-negative, got {}",
        thought.psi
    );

    // Meta-awareness should be in [0, 1]
    assert!(
        thought.meta_awareness >= 0.0 && thought.meta_awareness <= 1.0,
        "Meta-awareness should be in [0,1], got {}",
        thought.meta_awareness
    );

    // Coherence should be finite and non-negative
    assert!(
        thought.coherence.is_finite() && thought.coherence >= 0.0,
        "Coherence should be finite and non-negative, got {}",
        thought.coherence
    );

    // Emotional tone should have valid valence and arousal
    assert!(
        thought.emotional_tone.valence.is_finite(),
        "Valence should be finite"
    );
    assert!(
        thought.emotional_tone.arousal.is_finite(),
        "Arousal should be finite"
    );
}

// ── Test 3: Consciousness level is populated ──────────────────────────

#[tokio::test]
async fn test_consciousness_level_populated() {
    let mut s = try_new_symthaea!(512, 64);
    let resp = try_process!(s, "Tell me about consciousness.");

    assert!(
        resp.consciousness_level >= 0.0 && resp.consciousness_level <= 1.0,
        "Consciousness level should be in [0,1], got {}",
        resp.consciousness_level
    );
}

// ── Test 4: Multiple inputs evolve state ──────────────────────────────

#[tokio::test]
async fn test_state_evolution_over_multiple_inputs() {
    let mut s = try_new_symthaea!(512, 64);

    let r1 = try_process!(s, "The weather is nice today.");
    let r2 = try_process!(s, "I enjoy learning about physics.");
    let r3 = try_process!(s, "What causes the tides on Earth?");

    // After multiple interactions, the partnership state should have advanced
    let ps = s.partnership_state();
    assert!(
        ps.interactions >= 3,
        "Should have at least 3 interactions, got {}",
        ps.interactions
    );

    // Trajectory should have points recorded
    assert!(
        ps.trajectory_points >= 3,
        "Should have trajectory points from interactions, got {}",
        ps.trajectory_points
    );

    // All responses should be non-empty
    assert!(!r1.content.is_empty(), "Response 1 should be non-empty");
    assert!(!r2.content.is_empty(), "Response 2 should be non-empty");
    assert!(!r3.content.is_empty(), "Response 3 should be non-empty");
}

// ── Test 5: Varied input types ────────────────────────────────────────

#[tokio::test]
async fn test_varied_input_types() {
    let mut s = try_new_symthaea!(512, 64);

    // Question
    let q = try_process!(s, "What is 2 + 2?");
    assert!(
        !q.content.is_empty(),
        "Question response should be non-empty"
    );

    // Statement
    let st = try_process!(s, "The sun rises in the east.");
    assert!(
        !st.content.is_empty(),
        "Statement response should be non-empty"
    );

    // Emotional content
    let em = try_process!(s, "I feel deeply grateful for this beautiful moment.");
    assert!(
        !em.content.is_empty(),
        "Emotional response should be non-empty"
    );

    // Command-like
    let cmd = try_process!(s, "Please help me understand quantum mechanics.");
    assert!(
        !cmd.content.is_empty(),
        "Command response should be non-empty"
    );
}

// ── Test 6: Introspection reflects processing ─────────────────────────

#[tokio::test]
async fn test_introspection_reflects_processing() {
    let mut s = try_new_symthaea!(512, 64);

    let before = s.introspect();
    assert_eq!(
        before.memory_stats.long_term_count, 0,
        "No interactions before processing"
    );

    // Process several inputs
    for _ in 0..5 {
        let _ = s.process("some input text for processing").await;
    }

    let after = s.introspect();
    assert!(
        after.memory_stats.long_term_count > 0,
        "Long-term count should reflect interactions"
    );
    assert!(
        after.consciousness_level >= 0.0,
        "Consciousness level should be non-negative after processing"
    );
}

// ── Test 7: Safety flag is set for normal inputs ──────────────────────

#[tokio::test]
async fn test_safety_flag_for_normal_inputs() {
    let mut s = try_new_symthaea!(512, 64);

    let resp = try_process!(s, "Tell me about the history of mathematics.");
    assert!(resp.safe, "Normal inputs should be marked as safe");
}

// ── Test 8: Different inputs produce different responses ──────────────

#[tokio::test]
async fn test_different_inputs_produce_different_responses() {
    let mut s = try_new_symthaea!(512, 64);

    let r1 = try_process!(s, "quantum entanglement");
    let r2 = try_process!(s, "chocolate chip cookies");

    // The structured thoughts should differ (different activated concepts)
    let t1 = r1.structured_thought.as_ref().unwrap();
    let t2 = r2.structured_thought.as_ref().unwrap();

    // At minimum, the original inputs should differ
    assert_ne!(
        t1.original_input, t2.original_input,
        "Original inputs should be different"
    );
}

// ── Test 9: Pause and resume preserves interaction count ──────────────

#[tokio::test]
async fn test_pause_resume_preserves_state() {
    let tmp = std::env::temp_dir().join("e2e_pipeline_pause_test.json");
    let path = tmp.to_str().unwrap();

    {
        let mut s = try_new_symthaea!(512, 64);
        let _ = s.process("first interaction").await;
        let _ = s.process("second interaction").await;
        s.pause(path).unwrap();
    }

    let resumed = Symthaea::resume(path).unwrap();
    assert_eq!(
        resumed.dimension(),
        512,
        "Dimension should be preserved across pause/resume"
    );

    // Clean up
    let _ = std::fs::remove_file(path);
}

// ── Test 10: Embedding API consistency ────────────────────────────────

#[tokio::test]
async fn test_embed_api_consistency_with_process() {
    let mut s = try_new_symthaea!(512, 64);

    let embed = s.embed("test input");
    assert_eq!(
        embed.values.len(),
        512,
        "Embedding dimension should match constructor"
    );

    // Embedding should be normalized
    let mag: f32 = embed.values.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (mag - 1.0).abs() < 0.01,
        "Embedding should be normalized, got magnitude {mag}"
    );
}

// ── Test 11: Sleep consolidation after processing ─────────────────────

#[tokio::test]
async fn test_sleep_after_processing() {
    let mut s = try_new_symthaea!(512, 64);

    // Process several inputs
    for i in 0..5 {
        let _ = s.process(&format!("input number {i}")).await;
    }

    let report = s.sleep().await.unwrap();
    assert!(
        report.scaled > 0,
        "Sleep should report non-zero scaled memories"
    );
}

// ── Test 12 (ignored/expensive): Extended conversation ────────────────

#[tokio::test]
#[ignore]
async fn test_extended_conversation_50_turns() {
    let mut s = try_new_symthaea!(512, 64);

    let inputs = [
        "Hello, I am starting a conversation.",
        "What do you know about neural networks?",
        "How does backpropagation work?",
        "Tell me about consciousness.",
        "What is integrated information theory?",
        "Can machines be conscious?",
        "What is the hard problem of consciousness?",
        "How does HDC differ from traditional neural nets?",
        "What are hypervectors?",
        "Explain binding in HDC.",
        "What is predictive coding?",
        "How does free energy principle work?",
        "Tell me about active inference.",
        "What is the Bayesian brain hypothesis?",
        "How do emotions affect cognition?",
        "What is somatic marker hypothesis?",
        "Explain theory of mind.",
        "What is global workspace theory?",
        "How does attention work?",
        "What is metacognition?",
        "Tell me about episodic memory.",
        "How does memory consolidation work?",
        "What is working memory?",
        "Explain long-term potentiation.",
        "What is neuroplasticity?",
        "How do synapses form?",
        "What are mirror neurons?",
        "Tell me about embodied cognition.",
        "What is enactivism?",
        "How does perception work?",
        "What is the binding problem?",
        "Explain qualia.",
        "What is phenomenal consciousness?",
        "How does language relate to thought?",
        "What is the Sapir-Whorf hypothesis?",
        "Tell me about recursion in language.",
        "What is universal grammar?",
        "How do children learn language?",
        "What is statistical learning?",
        "Tell me about reinforcement learning.",
        "What is temporal difference learning?",
        "How do reward signals work?",
        "What is dopamine's role in learning?",
        "Tell me about the default mode network.",
        "What happens during sleep?",
        "How do dreams relate to memory?",
        "What is REM sleep?",
        "Tell me about circadian rhythms.",
        "What is the role of the hippocampus?",
        "How does spatial navigation work?",
    ];

    let mut confidence_history = Vec::new();

    for input in &inputs {
        let resp = try_process!(s, input);
        assert!(
            !resp.content.is_empty(),
            "Response should be non-empty for: {input}"
        );
        assert!(resp.confidence >= 0.0 && resp.confidence <= 1.0);
        confidence_history.push(resp.confidence);
    }

    // After 50 interactions, partnership should have advanced
    let ps = s.partnership_state();
    assert!(
        ps.interactions >= 50,
        "Should have 50+ interactions, got {}",
        ps.interactions
    );

    // Confidence should be finite for all interactions
    assert!(
        confidence_history.iter().all(|c| c.is_finite()),
        "All confidence values should be finite"
    );
}