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

use symthaea_broca::{
    LiquidMambaConfig, LiquidMambaGenerator, ThoughtChannels,
};
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
    let mut gen = make_generator();
    let channels = ThoughtChannels::default();
    let result = gen.generate(&channels);

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
    let mut gen = make_generator();
    let channels = ThoughtChannels::with_intent(1); // Answer
    let result = gen.generate(&channels);

    // output_hvs should contain one HV per generated token (including EOS-terminated)
    // Note: veto is disabled so no spurious tokens
    if result.num_tokens > 0 || !result.token_ids.is_empty() {
        assert!(
            !result.output_hvs.is_empty(),
            "output_hvs should be non-empty when tokens were generated"
        );
    }

    // Semantic PE should be finite and in [0, 2] (1 - cosine similarity)
    assert!(result.semantic_pe.is_finite(), "semantic_pe should be finite");
    assert!(
        result.semantic_pe >= 0.0 && result.semantic_pe <= 2.0,
        "semantic_pe {} should be in [0, 2]",
        result.semantic_pe
    );

    // Each output HV should have the HDC dimension (16384 by default)
    for (i, hv) in result.output_hvs.iter().enumerate() {
        assert_eq!(
            hv.values.len(), 16384,
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

    let mut gen1 = LiquidMambaGenerator::new(&genesis, config.clone())
        .expect("load failed");
    let mut gen2 = LiquidMambaGenerator::new(&genesis, config)
        .expect("load failed");

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

    let mut gen = LiquidMambaGenerator::new(&genesis, config)
        .expect("load failed");
    let channels = ThoughtChannels::default();
    let result = gen.generate(&channels);

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
        ActivatedConcept, EmotionalTone, StructuredThought, SemanticIntent,
        EpistemicStatus,
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
    thought.activated_concepts = vec![
        ActivatedConcept {
            name: "test".to_string(),
            activation: 0.9,
            relevance: 0.8,
        },
    ];

    let channels = thought_to_channels(&thought, 1.0);
    let mut gen = make_generator();
    let result = gen.generate(&channels);

    assert!(
        result.final_coherence.is_finite(),
        "Direct neural path should produce finite coherence"
    );
}

/// Gradient convergence: 5 steps of projection learning should not blow up.
#[test]
#[ignore = "Requires mamba-130m (~260MB)"]
fn test_gradient_convergence() {
    let mut gen = make_generator();
    let channels = ThoughtChannels::with_intent(1);

    // Initial generation to get baseline PE
    let result0 = gen.generate(&channels);
    let pe0 = result0.semantic_pe;

    // Run 5 gradient steps using the output HVs from each generation
    let mut last_pe = pe0;
    for step in 0..5 {
        let result = gen.generate(&channels);
        if result.output_hvs.is_empty() {
            continue; // EOS immediately — nothing to learn from
        }

        let thought_hv = gen.encoder().encode(&channels);
        let refs: Vec<&ContinuousHV> = result.output_hvs.iter().collect();
        let bundled = ContinuousHV::bundle(&refs).normalize();

        gen.projection_mut().compute_gradients(&thought_hv, &bundled);
        gen.projection_mut().apply_gradients(0.001, 1.0);

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

    let mut gen = LiquidMambaGenerator::new(&genesis, config)
        .expect("load failed");

    // Simulate exhausted state
    gen.update_affect(0.95, 0.5);

    let channels = ThoughtChannels::default();
    let result = gen.generate(&channels);

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
