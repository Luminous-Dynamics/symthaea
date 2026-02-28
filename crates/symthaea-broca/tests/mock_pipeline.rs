//! Offline Liquid-Mamba integration tests using MockMamba backend.
//!
//! These tests exercise the full generate → distill → PE pipeline without
//! network access, using the mock Mamba backend.

#![cfg(feature = "mamba")]

use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::liquid_mamba::{LiquidMambaConfig, LiquidMambaGenerator};
use symthaea_core::genesis::GenesisSeed;

fn test_genesis() -> GenesisSeed {
    GenesisSeed::from_phrase("test-mock-pipeline")
}

fn mock_gen(config: LiquidMambaConfig) -> LiquidMambaGenerator {
    LiquidMambaGenerator::with_mock(&test_genesis(), config)
}

fn default_config() -> LiquidMambaConfig {
    LiquidMambaConfig {
        max_tokens: 16,
        enable_consciousness_gating: false,
        ..Default::default()
    }
}

/// 10 generate+distill rounds — all PE values must be finite and in [0, 2].
#[test]
fn test_generate_distill_pe_finite() {
    let mut gen = mock_gen(default_config());
    let channels = ThoughtChannels::with_intent(1);

    for _ in 0..10 {
        let result = gen.generate(&channels);
        gen.distill_step(&channels, &result);

        assert!(
            result.semantic_pe.is_finite(),
            "PE should be finite, got: {}",
            result.semantic_pe
        );
        assert!(
            result.semantic_pe >= 0.0 && result.semantic_pe <= 2.0,
            "PE should be in [0, 2], got: {}",
            result.semantic_pe
        );
    }
}

/// After 5+ rounds with warmup=0 and accumulation=1, projection weights
/// should have changed from their initial values.
#[test]
fn test_distillation_modifies_weights() {
    let config = LiquidMambaConfig {
        max_tokens: 16,
        warmup_steps: 0,
        accumulation_steps: 1,
        enable_consciousness_gating: false,
        ..Default::default()
    };
    let mut gen = mock_gen(config);
    let channels = ThoughtChannels::with_intent(1);

    let initial_weights = gen.projection().flatten_weights();

    for _ in 0..6 {
        let result = gen.generate(&channels);
        gen.distill_step(&channels, &result);
    }

    let final_weights = gen.projection().flatten_weights();
    assert_eq!(initial_weights.len(), final_weights.len());

    // At least some weights should have changed
    let changed = initial_weights
        .iter()
        .zip(final_weights.iter())
        .filter(|(a, b)| (*a - *b).abs() > 1e-10)
        .count();

    assert!(
        changed > 0,
        "Expected weights to change after distillation, but all {} weights are identical",
        initial_weights.len()
    );
}

/// After 20 rounds, pe_stats() returns finite values.
#[test]
fn test_pe_trend_after_distillation() {
    let mut gen = mock_gen(default_config());
    let channels = ThoughtChannels::with_intent(2);

    for _ in 0..20 {
        let result = gen.generate(&channels);
        gen.distill_step(&channels, &result);
    }

    let (mean, std_dev, trend) = gen.pe_stats();
    assert!(mean.is_finite(), "PE mean should be finite: {mean}");
    assert!(
        std_dev.is_finite(),
        "PE std_dev should be finite: {std_dev}"
    );
    assert!(trend.is_finite(), "PE trend should be finite: {trend}");
    assert!(mean >= 0.0, "PE mean should be non-negative: {mean}");
    assert!(
        std_dev >= 0.0,
        "PE std_dev should be non-negative: {std_dev}"
    );
}

/// With veto enabled, final_coherence should be finite.
#[test]
fn test_coherence_monitoring_works() {
    let config = LiquidMambaConfig {
        max_tokens: 16,
        enable_veto: true,
        enable_consciousness_gating: false,
        ..Default::default()
    };
    let mut gen = mock_gen(config);
    let channels = ThoughtChannels::with_intent(0);

    let result = gen.generate(&channels);
    assert!(
        result.final_coherence.is_finite(),
        "final_coherence should be finite: {}",
        result.final_coherence
    );
}

/// set_fep_modulation() with various values should not panic,
/// and current_lr() should remain finite.
#[test]
fn test_fep_modulation_no_panic() {
    let mut gen = mock_gen(default_config());

    for &fep in &[0.0, 0.3, 0.5, 0.7, 1.0] {
        gen.set_fep_modulation(fep);
        let lr = gen.current_lr();
        assert!(lr.is_finite(), "LR should be finite for fep={fep}: {lr}");
    }
}

/// 55 rounds triggers check_projection_health at gen 50.
/// last_cached_rank() should be finite and positive afterward.
#[test]
fn test_collapse_recovery_health_check() {
    let config = LiquidMambaConfig {
        max_tokens: 8,
        warmup_steps: 0,
        accumulation_steps: 1,
        enable_consciousness_gating: false,
        ..Default::default()
    };
    let mut gen = mock_gen(config);
    let channels = ThoughtChannels::with_intent(1);

    for _ in 0..55 {
        let result = gen.generate(&channels);
        gen.distill_step(&channels, &result);
    }

    let rank = gen.last_cached_rank();
    assert!(rank.is_finite(), "Cached rank should be finite: {rank}");
    assert!(rank > 0.0, "Cached rank should be positive: {rank}");
}

/// Consciousness gating with high and low psi should complete
/// and produce finite PE.
#[test]
fn test_consciousness_gating_high_low_psi() {
    let config = LiquidMambaConfig {
        max_tokens: 12,
        enable_consciousness_gating: true,
        ..Default::default()
    };

    // High psi
    let mut gen = mock_gen(config.clone());
    let mut channels = ThoughtChannels::with_intent(0);
    channels.channels[12] = 0.95; // psi = high
    let result = gen.generate(&channels);
    assert!(
        result.semantic_pe.is_finite(),
        "PE should be finite with high psi: {}",
        result.semantic_pe
    );

    // Low psi
    let mut gen = mock_gen(config);
    let mut channels = ThoughtChannels::with_intent(0);
    channels.channels[12] = 0.05; // psi = low
    let result = gen.generate(&channels);
    assert!(
        result.semantic_pe.is_finite(),
        "PE should be finite with low psi: {}",
        result.semantic_pe
    );
}

/// EMA inference swap: after 10 rounds with enable_ema, EMA should be
/// active and all weights finite.
#[test]
fn test_ema_inference_swap() {
    let config = LiquidMambaConfig {
        max_tokens: 8,
        warmup_steps: 0,
        accumulation_steps: 1,
        enable_ema: true,
        ema_decay: 0.99,
        enable_consciousness_gating: false,
        ..Default::default()
    };
    let mut gen = mock_gen(config);
    let channels = ThoughtChannels::with_intent(1);

    for _ in 0..10 {
        let result = gen.generate(&channels);
        gen.distill_step(&channels, &result);
    }

    assert!(gen.projection().has_ema(), "EMA should be active");

    let weights = gen.projection().flatten_weights();
    assert!(
        weights.iter().all(|w| w.is_finite()),
        "All projection weights should be finite after EMA inference cycles"
    );
}
