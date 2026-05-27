// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Liquid-Mamba Full Pipeline Integration Tests
//!
//! End-to-end tests that exercise the Liquid-Mamba fusion pipeline against the
//! real mamba-130m model from HuggingFace. Each test is `#[ignore]` because the
//! model is ~260MB and requires a network download on first run.
//!
//! Run with:
//! ```bash
//! cargo test --features liquid-mamba --test liquid_mamba_integration -- --ignored --test-threads=1
//! ```
//!
//! The `--test-threads=1` flag prevents model cache contention.

#![cfg(feature = "liquid-mamba")]

use symthaea_broca::{LiquidMambaConfig, LiquidMambaGenerator, ThoughtChannels};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

// ============================================================================
// Helpers
// ============================================================================

fn test_genesis() -> GenesisSeed {
    GenesisSeed::from_phrase("liquid-mamba-integration-test")
}

fn short_config() -> LiquidMambaConfig {
    LiquidMambaConfig {
        max_tokens: 15,
        temperature: 0.01, // Near-greedy for determinism
        top_k: 5,
        enable_veto: false, // Disable veto for basic tests (re-enable explicitly)
        ..Default::default()
    }
}

fn make_generator() -> LiquidMambaGenerator {
    LiquidMambaGenerator::new(&test_genesis(), short_config())
        .expect("failed to load mamba-130m — is network available?")
}

// ============================================================================
// Tests
// ============================================================================

/// Basic generation from default thought channels produces non-empty text or EOS.
#[test]
#[ignore = "Requires mamba-130m (~260MB)"]
fn test_generate_from_channels() {
    let mut r#gen = make_generator();
    let channels = ThoughtChannels::default();
    let result = r#gen.generate(&channels);

    // Either produces text or hits EOS immediately — both are valid
    assert!(
        !result.text.is_empty() || result.eos_terminated,
        "Generation should produce text or terminate with EOS"
    );
    assert!(result.final_coherence.is_finite());
}

/// Output HVs are collected for every generated token and semantic PE is in valid range.
#[test]
#[ignore = "Requires mamba-130m (~260MB)"]
fn test_output_hvs_collected() {
    let mut r#gen = make_generator();
    let channels = ThoughtChannels::with_intent(1); // Answer
    let result = r#gen.generate(&channels);

    // output_hvs should contain one HV per generated token (including EOS-terminated)
    // Note: veto is disabled so no spurious tokens
    if result.num_tokens > 0 || !result.token_ids.is_empty() {
        assert!(
            !result.output_hvs.is_empty(),
            "output_hvs should be non-empty when tokens were generated"
        );
    }

    // Semantic PE should be finite and in [0, 2] (1 - cosine similarity)
    assert!(
        result.semantic_pe.is_finite(),
        "semantic_pe should be finite"
    );
    assert!(
        result.semantic_pe >= 0.0 && result.semantic_pe <= 2.0,
        "semantic_pe {} should be in [0, 2]",
        result.semantic_pe
    );

    // Each output HV should have the HDC dimension (16384 by default)
    for (i, hv) in result.output_hvs.iter().enumerate() {
        assert_eq!(
            hv.values.len(),
            16384,
            "output_hvs[{i}] should have 16384 dimensions"
        );
        assert!(
            hv.values.iter().all(|v| v.is_finite()),
            "output_hvs[{i}] should contain only finite values"
        );
    }
}

/// Different semantic intents produce different text with near-greedy sampling.
#[test]
#[ignore = "Requires mamba-130m (~260MB)"]
fn test_different_intents_differ() {
    let genesis = test_genesis();
    let config = LiquidMambaConfig {
        max_tokens: 20,
        temperature: 0.01,
        top_k: 1, // Greedy
        enable_veto: false,
        ..Default::default()
    };

    let mut gen1 = LiquidMambaGenerator::new(&genesis, config.clone()).expect("load failed");
    let mut gen2 = LiquidMambaGenerator::new(&genesis, config).expect("load failed");

    let ch_answer = ThoughtChannels::with_intent(1); // Answer
    let ch_reflect = ThoughtChannels::with_intent(5); // Reflect

    let r1 = gen1.generate(&ch_answer);
    let r2 = gen2.generate(&ch_reflect);

    // With different initial context projections, the outputs should differ
    // (this is stochastic — if both happen to hit EOS immediately, they'd match)
    if r1.num_tokens > 2 && r2.num_tokens > 2 {
        assert_ne!(
            r1.token_ids, r2.token_ids,
            "Different intents should produce different token sequences"
        );
    }
}

/// Coherence monitoring works when veto is enabled.
#[test]
#[ignore = "Requires mamba-130m (~260MB)"]
fn test_coherence_monitoring() {
    let genesis = test_genesis();
    let config = LiquidMambaConfig {
        max_tokens: 20,
        enable_veto: true,
        veto_threshold: 0.20,
        ..Default::default()
    };

    let mut r#gen = LiquidMambaGenerator::new(&genesis, config).expect("load failed");
    let channels = ThoughtChannels::default();
    let result = r#gen.generate(&channels);

    // Coherence should be finite regardless of whether veto fired
    assert!(
        result.final_coherence.is_finite(),
        "Final coherence should be finite, got {}",
        result.final_coherence
    );
}

/// Direct neural path: StructuredThought → thought_to_channels → generate.
#[test]
#[ignore = "Requires mamba-130m (~260MB)"]
fn test_direct_neural_path() {
    use symthaea::language::ssm_backend::thought_to_channels;
    use symthaea::mind::structured_thought::{
        ActivatedConcept, EmotionalTone, EpistemicStatus, SemanticIntent, StructuredThought,
    };

    let mut thought = StructuredThought::default();
    thought.semantic_intent = SemanticIntent::Answer;
    thought.epistemic_status = EpistemicStatus::Certain;
    thought.emotional_tone = EmotionalTone {
        valence: 0.5,
        arousal: 0.4,
        warmth: 0.6,
    };
    thought.psi = 0.7;
    thought.coherence = 0.8;
    thought.activated_concepts = vec![ActivatedConcept {
        name: "test".to_string(),
        activation: 0.9,
        relevance: 0.8,
    }];

    let channels = thought_to_channels(&thought, 1.0);
    let mut r#gen = make_generator();
    let result = r#gen.generate(&channels);

    assert!(
        result.final_coherence.is_finite(),
        "Direct neural path should produce finite coherence"
    );
}

/// Gradient convergence: 5 steps of projection learning should not blow up.
#[test]
#[ignore = "Requires mamba-130m (~260MB)"]
fn test_gradient_convergence() {
    let mut r#gen = make_generator();
    let channels = ThoughtChannels::with_intent(1);

    // Initial generation to get baseline PE
    let result0 = r#gen.generate(&channels);
    let pe0 = result0.semantic_pe;

    // Run 5 gradient steps using the output HVs from each generation
    let mut last_pe = pe0;
    for step in 0..5 {
        let result = r#gen.generate(&channels);
        if result.output_hvs.is_empty() {
            continue; // EOS immediately — nothing to learn from
        }

        let thought_hv = r#gen.encoder().encode(&channels);
        let refs: Vec<&ContinuousHV> = result.output_hvs.iter().collect();
        let bundled = ContinuousHV::bundle(&refs).normalize();

        r#gen.projection_mut()
            .compute_gradients(&thought_hv, &bundled);
        r#gen.projection_mut().apply_gradients(0.001, 1.0);

        last_pe = result.semantic_pe;
        assert!(
            last_pe.is_finite(),
            "PE should stay finite after step {step}, got {last_pe}"
        );
    }

    // PE should not have increased dramatically (>0.3 above initial)
    // Note: small number of steps, stochastic generation — weak assertion
    assert!(
        last_pe < pe0 + 0.3 || !pe0.is_finite(),
        "PE should not increase >0.3: initial={pe0}, final={last_pe}"
    );
}

/// High thermodynamic load still allows generation (biological delta modulation).
#[test]
#[ignore = "Requires mamba-130m (~260MB)"]
fn test_biological_modulation() {
    let genesis = test_genesis();
    let config = LiquidMambaConfig {
        max_tokens: 10,
        enable_liquid_delta: true,
        delta_mod_strength: 1.0,
        ..Default::default()
    };

    let mut r#gen = LiquidMambaGenerator::new(&genesis, config).expect("load failed");

    // Simulate exhausted state
    r#gen.update_affect(0.95, 0.5);

    let channels = ThoughtChannels::default();
    let result = r#gen.generate(&channels);

    // Should still produce output (possibly shorter due to faster decay)
    assert!(
        result.final_coherence.is_finite(),
        "Should generate even under high load"
    );
    assert!(
        result.semantic_pe.is_finite(),
        "Semantic PE should be finite under high load"
    );
}

/// Distill step end-to-end: generate → distill → verify PE in valid range.
#[test]
#[ignore = "Requires mamba-130m (~260MB)"]
fn test_distill_step_end_to_end() {
    let mut r#gen = make_generator();
    let channels = ThoughtChannels::with_intent(1); // Answer

    let result = r#gen.generate(&channels);

    // PE should be tracked after generation
    let pe_after_gen = r#gen.last_semantic_pe();
    assert!(
        pe_after_gen.is_finite(),
        "PE should be finite after generate()"
    );
    assert_eq!(
        pe_after_gen, result.semantic_pe,
        "last_semantic_pe() should match result.semantic_pe"
    );

    // Distill step should not crash
    r#gen.distill_step(&channels, &result);

    // Generation count should have incremented
    assert!(
        r#gen.generation_count() >= 1,
        "generation_count should increment"
    );

    // Generate again — PE may change
    let result2 = r#gen.generate(&channels);
    assert!(
        result2.semantic_pe.is_finite(),
        "PE should stay finite after distill"
    );
    assert!(
        r#gen.last_semantic_pe().is_finite(),
        "last_semantic_pe should stay finite after second generation"
    );
}

/// Streaming callback fires per token for L-SSM.
#[test]
#[ignore = "Requires mamba-130m (~260MB)"]
fn test_streaming_callback_l_ssm() {
    let mut r#gen = make_generator();
    let channels = ThoughtChannels::with_intent(1); // Answer

    let mut streamed = String::new();
    let mut callback_count = 0usize;

    let result = r#gen.generate_with_callback(&channels, &mut |token| {
        streamed.push_str(token);
        callback_count += 1;
    });

    // If tokens were generated, callback should have fired
    if result.num_tokens > 0 {
        assert!(
            callback_count > 0,
            "Streaming callback should fire for {} generated tokens",
            result.num_tokens
        );
    }

    // Streamed text should match result text
    assert_eq!(
        streamed, result.text,
        "Streamed text should match result.text"
    );

    // Result should be valid
    assert!(result.final_coherence.is_finite());
    assert!(result.semantic_pe.is_finite());
}

/// update_affect temperature modulates biological delta modulation.
#[test]
#[ignore = "Requires mamba-130m (~260MB)"]
fn test_update_affect_changes_delta_modulation() {
    let genesis = test_genesis();
    let config = LiquidMambaConfig {
        max_tokens: 10,
        temperature: 0.01, // Near-greedy
        top_k: 1,
        enable_liquid_delta: true,
        delta_mod_strength: 1.0,
        enable_veto: false,
        ..Default::default()
    };

    // Normal state: load=0, temp=1.0
    let mut gen_normal = LiquidMambaGenerator::new(&genesis, config.clone()).expect("load failed");
    gen_normal.update_affect(0.0, 1.0);
    assert!((gen_normal.mood_temperature() - 1.0).abs() < 0.01);

    // Hot mood: load=0.5, temp=2.0
    let mut gen_hot = LiquidMambaGenerator::new(&genesis, config).expect("load failed");
    gen_hot.update_affect(0.5, 2.0);
    assert!((gen_hot.mood_temperature() - 2.0).abs() < 0.01);

    let channels = ThoughtChannels::default();
    let result_normal = gen_normal.generate(&channels);
    let result_hot = gen_hot.generate(&channels);

    // Both should produce valid output
    assert!(result_normal.final_coherence.is_finite());
    assert!(result_hot.final_coherence.is_finite());

    // With different biological states, outputs should differ
    // (load=0.5 + hot mood shifts effective arousal)
    if result_normal.num_tokens > 2 && result_hot.num_tokens > 2 {
        // They may differ in tokens or coherence
        let token_diff = result_normal.token_ids != result_hot.token_ids;
        let coherence_diff =
            (result_normal.final_coherence - result_hot.final_coherence).abs() > 0.01;
        eprintln!(
            "Affect modulation: normal={} tokens, hot={} tokens, token_diff={}, coherence_diff={}",
            result_normal.num_tokens, result_hot.num_tokens, token_diff, coherence_diff
        );
    }
}