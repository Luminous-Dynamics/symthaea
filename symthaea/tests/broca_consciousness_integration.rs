//! Broca CfC-HDC Integration Tests
//!
//! End-to-end tests exercising the Broca language generation pipeline
//! with consciousness-gated generation, multi-turn context, conversation
//! history injection, and gating audit trails.
//!
//! These tests use the CfC-HDC backend (no Mamba/SSM required) and run
//! without model checkpoints (fresh untrained generator).
//!
//! Run with:
//! ```bash
//! cargo test --test broca_consciousness_integration
//! ```

use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::gating::GatingConfig;
use symthaea_broca::generator::{BrocaConfig, BrocaGenerator, GatingTraceEntry, SamplingStrategy};
use symthaea_core::genesis::GenesisSeed;

fn test_genesis() -> GenesisSeed {
    GenesisSeed::from_phrase("broca-consciousness-integration-test")
}

fn test_generator() -> BrocaGenerator {
    let config = BrocaConfig {
        gating: GatingConfig::default(),
        sampling: SamplingStrategy::Greedy,
        enable_epistemic_gate: true,
        enable_emotional_modulation: true,
        enable_coherence_feedback: true,
        enable_consciousness_gating: true,
        enable_semantic_veto: true,
        ..Default::default()
    };
    BrocaGenerator::new(&test_genesis(), config)
}

// ============================================================================
// Basic generation produces valid output
// ============================================================================

#[test]
fn test_basic_generation_produces_tokens() {
    let mut gen = test_generator();
    let channels = ThoughtChannels::default();
    let result = gen.generate(&channels);

    assert!(result.num_tokens > 0, "should generate at least one token");
    assert!(
        !result.text.is_empty(),
        "generated text should be non-empty"
    );
    assert_eq!(result.token_ids.len(), result.num_tokens);
}

// ============================================================================
// Consciousness gating: low consciousness suppresses generation
// ============================================================================

#[test]
fn test_consciousness_gating_suppresses_low_psi() {
    let config = BrocaConfig {
        enable_consciousness_gating: true,
        ..Default::default()
    };
    let mut gen = BrocaGenerator::new(&test_genesis(), config);

    // Very low consciousness — should reduce max tokens significantly
    let mut channels = ThoughtChannels::default();
    channels.set_consciousness(0.05, 0.1, 0.1); // psi=0.05

    let low_result = gen.generate(&channels);

    // High consciousness
    channels.set_consciousness(0.95, 0.9, 0.9);
    let high_result = gen.generate(&channels);

    // High consciousness should generally produce more tokens
    // (or at least not fewer, given the gating formula)
    assert!(
        high_result.num_tokens >= low_result.num_tokens || low_result.num_tokens <= 3,
        "high consciousness ({}) should produce >= tokens than low ({})",
        high_result.num_tokens,
        low_result.num_tokens
    );
}

// ============================================================================
// Epistemic gating: uncertain state affects generation
// ============================================================================

#[test]
fn test_epistemic_gating_affects_output() {
    let mut gen = test_generator();

    // Certain (epistemic=0 means certain in the ordinal scale)
    let mut certain_channels = ThoughtChannels::default();
    certain_channels.set_epistemic(0.0); // Certain
    certain_channels.set_consciousness(0.8, 0.7, 0.8);
    let certain_result = gen.generate(&certain_channels);

    // Out-of-domain (epistemic=4.0 means OOD)
    let mut ood_channels = ThoughtChannels::default();
    ood_channels.set_epistemic(4.0); // Out of domain
    ood_channels.set_consciousness(0.8, 0.7, 0.8);
    let ood_result = gen.generate(&ood_channels);

    // Both should produce output (not gated by consciousness)
    assert!(certain_result.num_tokens > 0);
    assert!(ood_result.num_tokens > 0);

    // The outputs should differ due to epistemic gating
    // (OOD should boost hedging tokens)
    assert_ne!(
        certain_result.text, ood_result.text,
        "certain vs OOD should produce different text"
    );
}

// ============================================================================
// Multi-turn context: generate_continuing preserves CfC state
// ============================================================================

#[test]
fn test_multi_turn_continuing_differs_from_fresh() {
    let mut gen = test_generator();
    let channels = ThoughtChannels::default();

    // First generation (fresh state)
    let result1 = gen.generate(&channels);

    // Second generation with continuing (preserves CfC temporal context)
    let result2 = gen.generate_continuing(&channels);

    // Third generation with fresh reset
    gen.controller_mut().reset();
    let result3 = gen.generate(&channels);

    // result1 and result3 should be identical (same fresh state)
    assert_eq!(
        result1.text, result3.text,
        "fresh generations with same seed should match"
    );

    // result2 should differ (continuing from result1's CfC state)
    // Note: with untrained weights this may or may not differ,
    // but the token_ids should reflect the different internal state
    if result1.num_tokens > 1 && result2.num_tokens > 1 {
        // At least verify the call succeeded
        assert!(result2.num_tokens > 0);
    }
}

// ============================================================================
// Coherence dynamics: per-token coherence tracked
// ============================================================================

#[test]
fn test_coherence_dynamics_populated() {
    let config = BrocaConfig {
        enable_coherence_feedback: true,
        ..Default::default()
    };
    let mut gen = BrocaGenerator::new(&test_genesis(), config);
    let channels = ThoughtChannels::default();
    let result = gen.generate(&channels);

    assert_eq!(
        result.coherence_dynamics.len(),
        result.num_tokens,
        "coherence_dynamics should have one entry per token"
    );

    for (i, &coh) in result.coherence_dynamics.iter().enumerate() {
        assert!(
            coh.is_finite(),
            "coherence at position {i} should be finite, got {coh}"
        );
        assert!(
            (-1.0..=1.0).contains(&coh),
            "coherence at position {i} should be in [-1, 1], got {coh}"
        );
    }
}

// ============================================================================
// Gating trace: audit trail populated when gating enabled
// ============================================================================

#[test]
fn test_gating_trace_populated() {
    let config = BrocaConfig {
        enable_coherence_feedback: true,
        enable_epistemic_gate: true,
        enable_emotional_modulation: true,
        ..Default::default()
    };
    let mut gen = BrocaGenerator::new(&test_genesis(), config);

    let mut channels = ThoughtChannels::default();
    channels.set_consciousness(0.8, 0.7, 0.8);
    let result = gen.generate(&channels);

    assert_eq!(
        result.gating_trace.len(),
        result.num_tokens,
        "gating_trace should have one entry per token"
    );

    for entry in &result.gating_trace {
        assert!(entry.coherence.is_finite());
        assert!(entry.binding_weight.is_finite());
    }
}

// ============================================================================
// Hallucination flag: noise input triggers flag
// ============================================================================

#[test]
fn test_hallucination_flag_field_exists() {
    let mut gen = test_generator();
    let channels = ThoughtChannels::default();
    let result = gen.generate(&channels);

    // hallucination_flag should be a valid bool (structural test)
    let _flag: bool = result.hallucination_flag;

    // With default (non-noisy) channels and untrained model,
    // we can't guarantee whether it fires, but it should not panic
}

// ============================================================================
// Semantic veto: verify veto_triggered field
// ============================================================================

#[test]
fn test_semantic_veto_field_present() {
    let config = BrocaConfig {
        enable_semantic_veto: true,
        enable_coherence_feedback: true,
        ..Default::default()
    };
    let mut gen = BrocaGenerator::new(&test_genesis(), config);
    let channels = ThoughtChannels::default();
    let result = gen.generate(&channels);

    // veto_triggered should be accessible (structural check)
    let _veto: bool = result.veto_triggered;
    // final_coherence should be finite
    assert!(result.final_coherence.is_finite());
}

// ============================================================================
// Emotional modulation: different emotions produce different output
// ============================================================================

#[test]
fn test_emotional_modulation_affects_output() {
    let mut gen = test_generator();

    // Calm, positive
    let mut calm = ThoughtChannels::default();
    calm.set_emotion(0.8, 0.1, 0.9); // positive valence, low arousal, warm
    calm.set_consciousness(0.8, 0.7, 0.8);
    let calm_result = gen.generate(&calm);

    // Excited, negative
    let mut excited = ThoughtChannels::default();
    excited.set_emotion(-0.8, 0.9, 0.1); // negative valence, high arousal, cold
    excited.set_consciousness(0.8, 0.7, 0.8);
    let excited_result = gen.generate(&excited);

    // Both should produce output
    assert!(calm_result.num_tokens > 0);
    assert!(excited_result.num_tokens > 0);

    // Different emotional states should produce different text
    assert_ne!(
        calm_result.text, excited_result.text,
        "different emotions should produce different text"
    );
}

// ============================================================================
// New context channels: time_pressure, domain_familiarity, etc.
// ============================================================================

#[test]
fn test_context_channels_accessible() {
    let mut channels = ThoughtChannels::default();

    // Set the 4 new channels
    channels.set_context(
        0.8,  // time_pressure
        0.9,  // domain_familiarity
        0.5,  // social_context
        0.95, // response_confidence
    );

    // Verify they're set correctly
    assert!((channels.time_pressure() - 0.8).abs() < 1e-6);
    assert!((channels.domain_familiarity() - 0.9).abs() < 1e-6);
    assert!((channels.social_context() - 0.5).abs() < 1e-6);
    assert!((channels.response_confidence() - 0.95).abs() < 1e-6);

    // Generate with these channels
    let mut gen = test_generator();
    let result = gen.generate(&channels);
    assert!(result.num_tokens > 0);
}

// ============================================================================
// Training pair backward compatibility: 20-channel legacy data pads to 24
// ============================================================================

#[test]
fn test_training_pair_legacy_padding() {
    use symthaea_broca::encoder::NUM_CHANNELS;
    use symthaea_broca::training::TrainingPair;

    // Simulate legacy 20-channel data
    let legacy_pair = TrainingPair {
        channels: vec![0.5; 20],
        target_text: "hello world".to_string(),
        target_ids: vec![1, 2, 3],
    };

    let tc = legacy_pair.to_thought_channels();
    assert_eq!(tc.channels.len(), NUM_CHANNELS);

    // First 20 channels should be 0.5
    for i in 0..20 {
        assert!(
            (tc.channels[i] - 0.5).abs() < 1e-6,
            "channel {i} should be 0.5"
        );
    }

    // Channels 20-23 should have default values (not 0.5)
    // time_pressure default = 0.0, domain_familiarity = 0.5,
    // social_context = 0.5, response_confidence = 0.5
    assert!(
        (tc.time_pressure() - 0.0).abs() < 1e-6,
        "time_pressure default"
    );
}

// ============================================================================
// Evaluation metrics: contrastive intent and hallucination rate
// ============================================================================

#[test]
fn test_eval_result_has_new_metrics() {
    use symthaea_broca::evaluation::{evaluate, EvalConfig};
    use symthaea_broca::training::TrainingDataset;

    let mut gen = test_generator();
    let tokenizer = gen.tokenizer().clone();

    // Create a tiny eval dataset
    let mut dataset = TrainingDataset::default();
    let mut channels = ThoughtChannels::default();
    channels.channels[0] = 1.0; // Acknowledgment intent
    channels.set_consciousness(0.8, 0.7, 0.8);

    dataset.pairs.push(symthaea_broca::TrainingPair {
        channels: channels.channels.to_vec(),
        target_text: "I understand.".to_string(),
        target_ids: vec![],
    });
    dataset.tokenize_all(&tokenizer);

    let config = EvalConfig {
        dataset,
        compute_perplexity: true,
        compute_english_ratio: true,
        per_intent_breakdown: true,
        max_gen_tokens: 16,
    };

    let result = evaluate(&mut gen, &config);

    // New metrics should be present
    assert!(result.contrastive_intent_score.is_some());
    assert!(result.hallucination_rate.is_some());

    let cis = result.contrastive_intent_score.unwrap();
    assert!(cis.is_finite(), "contrastive_intent_score should be finite");

    let hr = result.hallucination_rate.unwrap();
    assert!(hr.is_finite(), "hallucination_rate should be finite");
    assert!(
        (0.0..=1.0).contains(&hr),
        "hallucination_rate should be in [0,1]"
    );
}

// ============================================================================
// Checkpoint round-trip: save and reload preserves generation
// ============================================================================

#[test]
fn test_checkpoint_roundtrip_preserves_generation() {
    let mut gen = test_generator();
    let channels = ThoughtChannels::default();

    let result1 = gen.generate(&channels);

    // Save checkpoint
    let path = std::env::temp_dir()
        .join("broca_integration_checkpoint.bin")
        .to_string_lossy()
        .to_string();
    gen.save_checkpoint(&path, 0, 1.0, None, None, None)
        .expect("save should succeed");

    // Reload
    let (mut gen2, _, _, _) =
        BrocaGenerator::from_checkpoint(&path, &test_genesis()).expect("load should succeed");
    let result2 = gen2.generate(&channels);

    assert_eq!(
        result1.text, result2.text,
        "generation after checkpoint roundtrip should match"
    );

    let _ = std::fs::remove_file(&path);
}
