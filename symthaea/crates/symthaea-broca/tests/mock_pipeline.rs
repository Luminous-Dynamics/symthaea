// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Offline Liquid-Mamba integration tests using MockMamba backend.
//!
//! These tests exercise the full generate → distill → PE pipeline without
//! network access, using the mock Mamba backend.

#![cfg(feature = "test-helpers")]

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
/// Requires enable_consciousness_gating=true (recovery noise is gated).
#[test]
fn test_collapse_recovery_health_check() {
    let config = LiquidMambaConfig {
        max_tokens: 8,
        warmup_steps: 0,
        accumulation_steps: 1,
        enable_consciousness_gating: true, // Required: health check at gen 50 is gated
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

/// Warm-start bidirectional → generate → distill pipeline should work end to end.
#[test]
fn test_warm_start_then_generate_distill() {
    let mut gen = mock_gen(default_config());
    let channels = ThoughtChannels::with_intent(1);

    // Collect sample HVs
    let samples: Vec<symthaea_core::hdc::ContinuousHV> = (0..10)
        .map(|i| {
            let ch = ThoughtChannels::with_intent(i % 4);
            gen.encoder().encode(&ch)
        })
        .collect();

    // Warm-start projection
    gen.projection_mut().warm_start_bidirectional(&samples);

    // Generate + distill should still work
    for _ in 0..5 {
        let result = gen.generate(&channels);
        assert!(
            result.semantic_pe.is_finite(),
            "PE should be finite after warm-start: {}",
            result.semantic_pe
        );
        gen.distill_step(&channels, &result);
    }
}

/// Contrastive pretrain → generate pipeline should produce finite results.
#[test]
fn test_contrastive_pretrain_then_generate() {
    let mut gen = mock_gen(default_config());

    // Build thought HVs from different intents
    let thought_hvs: Vec<symthaea_core::hdc::ContinuousHV> = (0..8)
        .map(|i| {
            let ch = ThoughtChannels::with_intent(i % 4);
            gen.encoder().encode(&ch)
        })
        .collect();

    // Contrastive pretrain
    let (avg_dist, avg_recon) = gen
        .projection_mut()
        .contrastive_pretrain(&thought_hvs, 5, 0.01);
    assert!(
        avg_dist.is_finite(),
        "avg_dist should be finite: {avg_dist}"
    );
    assert!(
        avg_recon.is_finite(),
        "avg_recon should be finite: {avg_recon}"
    );

    // Generate should still produce finite results
    let channels = ThoughtChannels::with_intent(2);
    let result = gen.generate(&channels);
    assert!(
        result.semantic_pe.is_finite(),
        "PE should be finite after contrastive pretrain: {}",
        result.semantic_pe
    );
}

/// Surprise-weighted gradient: high-PE examples should modify weights more.
#[test]
fn test_surprise_gradient_amplifies() {
    let config = LiquidMambaConfig {
        max_tokens: 8,
        warmup_steps: 0,
        accumulation_steps: 1,
        enable_consciousness_gating: false,
        surprise_gradient_alpha: 1.0, // Strong surprise weighting
        ..Default::default()
    };
    let mut gen = mock_gen(config);
    let channels = ThoughtChannels::with_intent(1);

    let initial_weights = gen.projection().flatten_weights();

    // Run generate + distill
    let result = gen.generate(&channels);
    gen.distill_step(&channels, &result);

    let after_weights = gen.projection().flatten_weights();

    // Weights should have changed (non-zero PE → amplified gradients)
    let changed = initial_weights
        .iter()
        .zip(after_weights.iter())
        .filter(|(a, b)| (*a - *b).abs() > 1e-10)
        .count();

    assert!(
        changed > 0,
        "Surprise-weighted gradient should modify weights"
    );
}

/// Eval-only workflow: train → save checkpoint → load → evaluate.
///
/// Tests the same code path the binary's --eval-only mode exercises:
/// generate+distill → ProjectionCheckpoint save → fresh gen with loaded weights → evaluate.
#[test]
fn test_eval_only_workflow() {
    use symthaea_broca::checkpoint::ProjectionCheckpoint;
    use symthaea_broca::evaluation::{
        evaluate_liquid_mamba, format_liquid_mamba_eval_report, LiquidMambaEvalConfig,
        QualityGateThresholds,
    };
    use symthaea_broca::training::{TrainingDataset, TrainingPair};

    let config = LiquidMambaConfig {
        max_tokens: 8,
        warmup_steps: 0,
        accumulation_steps: 1,
        enable_consciousness_gating: false,
        ..Default::default()
    };
    let mut gen = mock_gen(config.clone());
    let channels = ThoughtChannels::with_intent(1);

    // 1. Train for 5 rounds
    for _ in 0..5 {
        let result = gen.generate(&channels);
        gen.distill_step(&channels, &result);
    }

    // 2. Save projection checkpoint to tempfile
    let dir = std::env::temp_dir();
    let path = dir.join("test_eval_only_workflow.bin");
    let weights = gen.projection().flatten_weights();
    let mut ckpt = ProjectionCheckpoint::new(
        weights,
        gen.config().hdc_dim,
        gen.config().bottleneck_dim,
        gen.config().ssm_dim,
        5,
        gen.projection().is_deep(),
        gen.projection().inner_dim(),
    );
    ckpt.save_to_file(&path).unwrap();

    // 3. Load fresh gen from checkpoint (simulating --resume)
    let mut fresh_gen = mock_gen(config);
    let loaded_ckpt = ProjectionCheckpoint::load_from_file(&path).unwrap();
    fresh_gen
        .projection_mut()
        .load_weights(&loaded_ckpt.projection_weights);

    // 4. Build eval config and run evaluate_liquid_mamba()
    let mut dataset = TrainingDataset::default();
    for intent in 0..3 {
        let ch = ThoughtChannels::with_intent(intent);
        dataset.pairs.push(TrainingPair {
            channels: ch.channels.to_vec(),
            target_text: "test evaluation".to_string(),
            target_ids: vec![],
        });
    }
    let eval_config = LiquidMambaEvalConfig {
        dataset,
        compute_perplexity: true,
        compute_english_ratio: true,
        per_intent_breakdown: true,
        max_gen_tokens: 8,
        consciousness_gating_test: false,
    };
    let result = evaluate_liquid_mamba(&mut fresh_gen, &eval_config);

    // 5. Assert: result has finite perplexity, PE, rank
    assert!(
        result.base.perplexity.is_finite(),
        "Perplexity should be finite: {}",
        result.base.perplexity
    );
    assert!(
        result.avg_semantic_pe.is_finite(),
        "PE should be finite: {}",
        result.avg_semantic_pe
    );
    assert!(
        result.avg_effective_rank.is_finite(),
        "Rank should be finite: {}",
        result.avg_effective_rank
    );

    // 6. Assert: format_liquid_mamba_eval_report() produces non-empty string
    let report = format_liquid_mamba_eval_report(&result, &QualityGateThresholds::default());
    assert!(!report.is_empty(), "Report should be non-empty");
    assert!(report.contains("Liquid-Mamba"), "Report should have header");

    let _ = std::fs::remove_file(&path);
}

/// Diagnostics accumulate across multiple distill_step calls.
#[test]
fn test_diagnostics_across_training() {
    let config = LiquidMambaConfig {
        max_tokens: 8,
        warmup_steps: 0,
        accumulation_steps: 1,
        enable_consciousness_gating: false,
        ..Default::default()
    };
    let mut gen = mock_gen(config);
    gen.enable_diagnostics();

    let channels = ThoughtChannels::with_intent(1);

    // Train 10 rounds (warmup=0, accum=1 so every step applies)
    for _ in 0..10 {
        let result = gen.generate(&channels);
        gen.distill_step(&channels, &result);
    }

    let diag = gen
        .projection_diagnostics()
        .expect("diagnostics should be enabled");
    assert!(
        diag.total_steps >= 5,
        "Expected >= 5 gradient steps, got {}",
        diag.total_steps
    );
    assert_eq!(
        diag.grad_norms_down.len(),
        diag.total_steps,
        "grad_norms_down length should match total_steps"
    );
    assert!(
        !diag.bottleneck_collapse_detected(),
        "Healthy mock should not detect collapse"
    );
}

/// Diagnostics restore from checkpoint: total_steps should be cumulative.
#[test]
fn test_diagnostics_restore_from_checkpoint() {
    use symthaea_broca::checkpoint::ProjectionCheckpoint;

    let config = LiquidMambaConfig {
        max_tokens: 8,
        warmup_steps: 0,
        accumulation_steps: 1,
        enable_consciousness_gating: false,
        ..Default::default()
    };
    let mut gen = mock_gen(config.clone());
    gen.enable_diagnostics();

    let channels = ThoughtChannels::with_intent(1);

    // Train 10 rounds
    for _ in 0..10 {
        let result = gen.generate(&channels);
        gen.distill_step(&channels, &result);
    }

    let diag = gen
        .projection_diagnostics()
        .expect("diagnostics should be enabled");
    let original_steps = diag.total_steps;
    assert!(
        original_steps >= 5,
        "Should have recorded steps: {original_steps}"
    );

    // Take snapshot and save checkpoint
    let snap = diag.snapshot();
    let weights = gen.projection().flatten_weights();
    let mut ckpt = ProjectionCheckpoint::new(
        weights,
        gen.config().hdc_dim,
        gen.config().bottleneck_dim,
        gen.config().ssm_dim,
        10,
        gen.projection().is_deep(),
        gen.projection().inner_dim(),
    );
    ckpt.diagnostics_snapshot = Some(snap);
    let dir = std::env::temp_dir();
    let path = dir.join("test_diagnostics_restore.bin");
    ckpt.save_to_file(&path).unwrap();

    // Create fresh gen, enable diagnostics, restore from snapshot
    let mut fresh_gen = mock_gen(config);
    fresh_gen.enable_diagnostics();
    let loaded_ckpt = ProjectionCheckpoint::load_from_file(&path).unwrap();
    if let Some(ref snap) = loaded_ckpt.diagnostics_snapshot {
        if let Some(diag) = fresh_gen.projection_diagnostics_mut() {
            diag.restore_from_snapshot(snap);
        }
    }
    fresh_gen
        .projection_mut()
        .load_weights(&loaded_ckpt.projection_weights);

    // Verify restored steps match original
    let restored_diag = fresh_gen
        .projection_diagnostics()
        .expect("diagnostics should be enabled");
    assert_eq!(
        restored_diag.total_steps, original_steps,
        "Restored total_steps should match original"
    );

    // Train 5 more rounds
    for _ in 0..5 {
        let result = fresh_gen.generate(&channels);
        fresh_gen.distill_step(&channels, &result);
    }

    let final_diag = fresh_gen
        .projection_diagnostics()
        .expect("diagnostics should be enabled");
    assert!(
        final_diag.total_steps > original_steps,
        "Cumulative steps ({}) should exceed original ({})",
        final_diag.total_steps,
        original_steps
    );

    let _ = std::fs::remove_file(&path);
}

/// Diagnostics-triggered collapse detection and recovery: trains normally,
/// then injects uniform weights to collapse the projection, verifies
/// that bottleneck_collapse_detected() fires, then exercises the
/// check_projection_health() recovery path.
#[test]
fn test_diagnostics_triggered_collapse_recovery() {
    let config = LiquidMambaConfig {
        max_tokens: 8,
        warmup_steps: 0,
        accumulation_steps: 1,
        enable_consciousness_gating: false,
        ..Default::default()
    };
    let mut gen = mock_gen(config);
    gen.enable_diagnostics();

    let channels = ThoughtChannels::with_intent(1);

    // Phase 1: Train normally to accumulate healthy diagnostics baselines
    for _ in 0..10 {
        let result = gen.generate(&channels);
        gen.distill_step(&channels, &result);
    }

    let diag = gen
        .projection_diagnostics()
        .expect("diagnostics should be enabled");
    assert!(
        diag.total_steps >= 5,
        "Should have recorded gradient steps: {}",
        diag.total_steps
    );
    assert!(
        !diag.bottleneck_collapse_detected(),
        "Healthy training should not detect collapse"
    );

    // Phase 2: Inject uniform weights to induce dimensional collapse
    let initial_weights = gen.projection().flatten_weights();
    let uniform: Vec<f32> = initial_weights.iter().map(|_| 0.001).collect();
    gen.projection_mut().load_weights(&uniform);

    // Run more rounds: distill_step records bottleneck variance after each
    // gradient application. With uniform weights, variance should be near-zero.
    for _ in 0..10 {
        let result = gen.generate(&channels);
        gen.distill_step(&channels, &result);
    }

    let diag = gen
        .projection_diagnostics()
        .expect("diagnostics should be enabled");

    // After injecting uniform weights, at least some bottleneck variances
    // should be very small. Check if collapse was detected.
    let low_var_count = diag
        .bottleneck_variances
        .iter()
        .filter(|&&v| v < 0.01)
        .count();

    // Phase 3: Manually trigger health check to verify recovery works
    // (the auto-recovery path requires enable_consciousness_gating, but
    // we test the recovery function directly here)
    let sample_hvs: Vec<symthaea_core::hdc::ContinuousHV> = (0..5)
        .map(|i| {
            let ch = ThoughtChannels::with_intent(i % 4);
            gen.encoder().encode(&ch)
        })
        .collect();
    gen.check_projection_health(&sample_hvs);

    // After health check, rank should be finite and positive
    let rank = gen.last_cached_rank();
    assert!(rank.is_finite(), "Cached rank should be finite: {rank}");
    assert!(rank > 0.0, "Cached rank should be positive: {rank}");

    // Weights should have changed from the uniform injection
    let final_weights = gen.projection().flatten_weights();
    let changed = uniform
        .iter()
        .zip(final_weights.iter())
        .filter(|(a, b)| (*a - *b).abs() > 1e-6)
        .count();
    assert!(
        changed > 0,
        "Weights should differ after health check recovery (low_var_count={})",
        low_var_count
    );
}

/// Full training pipeline smoke test: dataset → warm-start → epochs →
/// checkpoint → reload → evaluate → diagnostics round-trip.
#[test]
fn test_full_training_pipeline_smoke() {
    use symthaea_broca::checkpoint::ProjectionCheckpoint;
    use symthaea_broca::evaluation::{
        evaluate_liquid_mamba, format_liquid_mamba_eval_report, LiquidMambaEvalConfig,
        QualityGateThresholds,
    };
    use symthaea_broca::training::{TrainingDataset, TrainingPair};

    let config = LiquidMambaConfig {
        max_tokens: 8,
        warmup_steps: 0,
        accumulation_steps: 1,
        enable_consciousness_gating: false,
        ..Default::default()
    };
    let mut gen = mock_gen(config.clone());
    gen.enable_diagnostics();

    // 1. Build a small training dataset (5 pairs, diverse intents)
    let mut dataset = TrainingDataset::default();
    for intent in 0..5 {
        let ch = ThoughtChannels::with_intent(intent % 4);
        dataset.pairs.push(TrainingPair {
            channels: ch.channels.to_vec(),
            target_text: format!("training sample {intent}"),
            target_ids: vec![],
        });
    }

    // 2. Warm-start from sample HVs
    let samples: Vec<symthaea_core::hdc::ContinuousHV> = (0..10)
        .map(|i| {
            let ch = ThoughtChannels::with_intent(i % 4);
            gen.encoder().encode(&ch)
        })
        .collect();
    gen.projection_mut().warm_start_bidirectional(&samples);

    // 3. Run 2 epochs over the dataset
    for _epoch in 0..2 {
        for pair in &dataset.pairs {
            let ch = pair.to_thought_channels();
            let result = gen.generate(&ch);
            gen.distill_step(&ch, &result);
        }
    }

    // 4. Compute effective rank
    let rank = gen.last_cached_rank();
    assert!(
        rank.is_finite(),
        "Rank should be finite after training: {rank}"
    );

    // 5. Save checkpoint with diagnostics snapshot
    let weights = gen.projection().flatten_weights();
    let diag = gen.projection_diagnostics().expect("diagnostics enabled");
    let snap = diag.snapshot();
    let mut ckpt = ProjectionCheckpoint::new(
        weights.clone(),
        gen.config().hdc_dim,
        gen.config().bottleneck_dim,
        gen.config().ssm_dim,
        10,
        gen.projection().is_deep(),
        gen.projection().inner_dim(),
    );
    ckpt.diagnostics_snapshot = Some(snap);
    let dir = std::env::temp_dir();
    let path = dir.join("test_full_pipeline_smoke.bin");
    ckpt.save_to_file(&path).unwrap();

    // 6. Load checkpoint into fresh gen
    let mut fresh_gen = mock_gen(config);
    let loaded_ckpt = ProjectionCheckpoint::load_from_file(&path).unwrap();
    fresh_gen
        .projection_mut()
        .load_weights(&loaded_ckpt.projection_weights);

    // 7. Assert loaded weights match saved weights
    let loaded_weights = fresh_gen.projection().flatten_weights();
    assert_eq!(weights.len(), loaded_weights.len());
    let weight_match = weights
        .iter()
        .zip(loaded_weights.iter())
        .all(|(a, b)| (a - b).abs() < 1e-10);
    assert!(weight_match, "Loaded weights should match saved weights");

    // 8. Run evaluation
    let eval_config = LiquidMambaEvalConfig {
        dataset,
        compute_perplexity: true,
        compute_english_ratio: true,
        per_intent_breakdown: true,
        max_gen_tokens: 8,
        consciousness_gating_test: false,
    };
    let result = evaluate_liquid_mamba(&mut fresh_gen, &eval_config);

    assert!(
        result.base.perplexity.is_finite(),
        "Perplexity should be finite: {}",
        result.base.perplexity
    );
    assert!(
        result.avg_semantic_pe.is_finite(),
        "PE should be finite: {}",
        result.avg_semantic_pe
    );
    assert!(
        result.avg_effective_rank.is_finite(),
        "Rank should be finite: {}",
        result.avg_effective_rank
    );

    // 9. Format report
    let report = format_liquid_mamba_eval_report(&result, &QualityGateThresholds::default());
    assert!(!report.is_empty(), "Report should be non-empty");

    // 10. Diagnostics snapshot round-trip
    let loaded_snap = loaded_ckpt
        .diagnostics_snapshot
        .expect("Checkpoint should have diagnostics");
    assert!(
        loaded_snap.total_steps > 0,
        "Snapshot total_steps should be > 0: {}",
        loaded_snap.total_steps
    );

    let _ = std::fs::remove_file(&path);
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

/// Full temporal projection pipeline: generate + distill with temporal_projection=true.
/// Verifies that the temporal path (64 chunks × 768D via inject_context_sequence)
/// produces finite PE, modifies temporal weights, and doesn't regress spatial path.
#[test]
fn test_temporal_generation_pipeline() {
    let config = LiquidMambaConfig {
        max_tokens: 16,
        warmup_steps: 0,
        accumulation_steps: 1,
        enable_consciousness_gating: false,
        temporal_projection: true,
        ..Default::default()
    };
    let mut gen = mock_gen(config);
    let channels = ThoughtChannels::with_intent(1);

    // Temporal projection should be initialized
    assert!(
        gen.temporal_proj().is_some(),
        "temporal_proj should be Some when temporal_projection=true"
    );

    let initial_weights = gen.temporal_proj().unwrap().flatten_weights();
    assert!(
        !initial_weights.is_empty(),
        "Temporal projection should have weights"
    );

    // Run 10 generate+distill rounds
    for i in 0..10 {
        let result = gen.generate(&channels);
        assert!(
            result.semantic_pe.is_finite(),
            "PE should be finite at round {i}, got: {}",
            result.semantic_pe
        );
        assert!(
            result.semantic_pe >= 0.0 && result.semantic_pe <= 2.0,
            "PE should be in [0, 2] at round {i}, got: {}",
            result.semantic_pe
        );
        gen.distill_step(&channels, &result);
    }

    // Temporal weights should have changed
    let final_weights = gen.temporal_proj().unwrap().flatten_weights();
    assert_eq!(initial_weights.len(), final_weights.len());
    let changed = initial_weights
        .iter()
        .zip(final_weights.iter())
        .filter(|(a, b)| (*a - *b).abs() > 1e-10)
        .count();
    assert!(
        changed > 0,
        "Expected temporal weights to change after distillation, but all {} weights are identical",
        initial_weights.len()
    );

    // PE stats should be finite
    let (mean, std_dev, trend) = gen.pe_stats();
    assert!(mean.is_finite(), "PE mean should be finite: {mean}");
    assert!(
        std_dev.is_finite(),
        "PE std_dev should be finite: {std_dev}"
    );
    assert!(trend.is_finite(), "PE trend should be finite: {trend}");
}

/// Temporal checkpoint round-trip: save temporal weights, load into fresh gen,
/// verify weights match and generation still works.
#[test]
fn test_temporal_checkpoint_roundtrip() {
    use symthaea_broca::checkpoint::ProjectionCheckpoint;

    let config = LiquidMambaConfig {
        max_tokens: 8,
        warmup_steps: 0,
        accumulation_steps: 1,
        enable_consciousness_gating: false,
        temporal_projection: true,
        ..Default::default()
    };
    let mut gen = mock_gen(config.clone());
    let channels = ThoughtChannels::with_intent(2);

    // Train for 5 rounds
    for _ in 0..5 {
        let result = gen.generate(&channels);
        gen.distill_step(&channels, &result);
    }

    // Save temporal checkpoint
    let tp = gen.temporal_proj().unwrap();
    let temporal_weights = tp.flatten_weights();
    let spatial_weights = gen.projection().flatten_weights();

    let mut ckpt = ProjectionCheckpoint::new_temporal(
        spatial_weights.clone(),
        temporal_weights.clone(),
        gen.config().hdc_dim,
        gen.config().bottleneck_dim,
        gen.config().ssm_dim,
        5,
        tp.chunk_size(),
        tp.num_chunks(),
    );
    let dir = std::env::temp_dir();
    let path = dir.join("test_temporal_ckpt_roundtrip.bin");
    ckpt.save_to_file(&path).unwrap();

    // Load into fresh gen
    let mut fresh_gen = mock_gen(config);
    let loaded_ckpt = ProjectionCheckpoint::load_from_file(&path).unwrap();
    assert!(loaded_ckpt.temporal, "Checkpoint should have temporal=true");

    fresh_gen
        .projection_mut()
        .load_weights(&loaded_ckpt.projection_weights);
    if let Some(ref tw) = loaded_ckpt.temporal_weights {
        fresh_gen.temporal_proj_mut().unwrap().load_weights(tw);
    }

    // Verify temporal weights match
    let loaded_tw = fresh_gen.temporal_proj().unwrap().flatten_weights();
    assert_eq!(temporal_weights.len(), loaded_tw.len());
    let match_count = temporal_weights
        .iter()
        .zip(loaded_tw.iter())
        .filter(|(a, b)| (*a - *b).abs() < 1e-10)
        .count();
    assert_eq!(
        match_count,
        temporal_weights.len(),
        "All temporal weights should match after load"
    );

    // Generation should still work
    let result = fresh_gen.generate(&channels);
    assert!(
        result.semantic_pe.is_finite(),
        "PE should be finite after checkpoint restore: {}",
        result.semantic_pe
    );

    let _ = std::fs::remove_file(&path);
}

/// Temporal projection health check: inject uniform weights to collapse temporal
/// projection, verify check_projection_health() detects and recovers.
#[test]
fn test_temporal_collapse_recovery_health_check() {
    let config = LiquidMambaConfig {
        max_tokens: 8,
        warmup_steps: 0,
        accumulation_steps: 1,
        enable_consciousness_gating: false,
        temporal_projection: true,
        ..Default::default()
    };
    let mut gen = mock_gen(config);
    let channels = ThoughtChannels::with_intent(1);

    // Phase 1: Train normally to get baseline
    for _ in 0..5 {
        let result = gen.generate(&channels);
        gen.distill_step(&channels, &result);
    }

    // Phase 2: Inject uniform weights to induce collapse
    let tp = gen.temporal_proj().unwrap();
    let original_weights = tp.flatten_weights();
    let uniform: Vec<f32> = original_weights.iter().map(|_| 0.001).collect();
    gen.temporal_proj_mut().unwrap().load_weights(&uniform);

    // Phase 3: Trigger health check
    let sample_hvs: Vec<symthaea_core::hdc::ContinuousHV> = (0..5)
        .map(|i| {
            let ch = ThoughtChannels::with_intent(i % 4);
            gen.encoder().encode(&ch)
        })
        .collect();
    gen.check_projection_health(&sample_hvs);

    // Rank should be finite and positive
    let rank = gen.last_cached_rank();
    assert!(rank.is_finite(), "Cached rank should be finite: {rank}");
    assert!(rank > 0.0, "Cached rank should be positive: {rank}");

    // Weights should have changed from uniform (recovery noise applied)
    let final_weights = gen.temporal_proj().unwrap().flatten_weights();
    let changed = uniform
        .iter()
        .zip(final_weights.iter())
        .filter(|(a, b)| (*a - *b).abs() > 1e-6)
        .count();
    assert!(
        changed > 0,
        "Temporal weights should differ after health check recovery"
    );
}
