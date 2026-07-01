// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Ablation Baselines for Paper Claims
//!
//! Compares three configurations to demonstrate HDC+CfC synergy:
//! - **HDC-only**: Encoder without temporal CfC prediction (always predict from last)
//! - **CfC-only**: CfC network without HDC encoding (raw chars → CfC)
//! - **Full system**: HDC+CfC (standard cognitive loop)
//!
//! Usage: `cargo run --example ablation_baselines`

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, TemporalBackend};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Symthaea v0.5.0 — Ablation Baseline Comparison");
    println!("═══════════════════════════════════════════════════════════════\n");

    let words = ["alpha beta", "gamma delta", "epsilon zeta", "eta theta"];
    let n_cycles = 200;
    let warmup = 20;
    let window = 10;

    // Full system (HDC encoder + CfC temporal)
    println!("Running Full HDC+CfC system ({} cycles)...", n_cycles);
    let full_result = run_ablation(
        "Full (HDC+CfC)",
        &words,
        n_cycles,
        warmup,
        window,
        AblationMode::Full,
    );

    // HDC-only (no temporal learning — high learning threshold disables CfC training)
    println!("Running HDC-only ({} cycles)...", n_cycles);
    let hdc_result = run_ablation(
        "HDC-only",
        &words,
        n_cycles,
        warmup,
        window,
        AblationMode::HdcOnly,
    );

    // Minimal CfC (reduced HDC contribution via minimal encoder)
    println!("Running Minimal-HDC+CfC ({} cycles)...", n_cycles);
    let cfc_result = run_ablation(
        "Minimal-HDC+CfC",
        &words,
        n_cycles,
        warmup,
        window,
        AblationMode::MinimalHdc,
    );

    // HdcLtcUnified backend
    println!("Running HDC-LTC Unified ({} cycles)...", n_cycles);
    let unified_result = run_ablation(
        "HDC-LTC Unified",
        &words,
        n_cycles,
        warmup,
        window,
        AblationMode::HdcLtc,
    );

    // Print comparison table
    println!("\n── Comparison Table ─────────────────────────────────────────");
    println!(
        "  {:<20} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "Configuration", "Init Err", "Final Err", "Reduction%", "Phi_final", "Learn%"
    );
    println!("  {}", "─".repeat(70));

    for r in [&full_result, &hdc_result, &cfc_result, &unified_result] {
        println!(
            "  {:<20} {:>10.4} {:>10.4} {:>10.1} {:>10.4} {:>8.1}",
            r.name, r.initial_error, r.final_error, r.reduction_pct, r.final_phi, r.learning_pct
        );
    }

    println!("\n── Key Finding ─────────────────────────────────────────────");
    let full_reduction = full_result.reduction_pct;
    let hdc_only_reduction = hdc_result.reduction_pct;
    let synergy = full_reduction - hdc_only_reduction;
    println!("  Full system reduction:  {:.1}%", full_reduction);
    println!("  HDC-only reduction:     {:.1}%", hdc_only_reduction);
    println!("  CfC temporal synergy:   {:+.1}pp", synergy);
    println!(
        "  Conclusion: {}",
        if synergy > 0.0 {
            "CfC temporal prediction adds measurable value to HDC encoding"
        } else {
            "HDC encoding alone achieves comparable error reduction (CfC marginal)"
        }
    );
    println!();
}

#[derive(Debug, Clone, Copy)]
enum AblationMode {
    Full,
    HdcOnly,
    MinimalHdc,
    HdcLtc,
}

struct AblationResult {
    name: String,
    initial_error: f32,
    final_error: f32,
    reduction_pct: f32,
    final_phi: f64,
    learning_pct: f32,
}

fn run_ablation(
    name: &str,
    words: &[&str],
    n_cycles: usize,
    warmup: usize,
    window: usize,
    mode: AblationMode,
) -> AblationResult {
    let config = match mode {
        AblationMode::Full => CognitiveLoopConfig {
            learning_threshold: 0.0,
            ..CognitiveLoopConfig::with_cfc()
        },
        AblationMode::HdcOnly => CognitiveLoopConfig {
            learning_threshold: 10.0, // Effectively disables CfC learning
            ..CognitiveLoopConfig::with_cfc()
        },
        AblationMode::MinimalHdc => CognitiveLoopConfig {
            learning_threshold: 0.0,
            encoder_config: symthaea_core::hdc::predictive_encoder::PredictiveEncoderConfig {
                attention_lr: 0.0, // No attention adaptation
                ..Default::default()
            },
            ..CognitiveLoopConfig::with_cfc()
        },
        AblationMode::HdcLtc => CognitiveLoopConfig {
            learning_threshold: 0.0,
            temporal_backend: TemporalBackend::HdcLtcUnified,
            ..Default::default()
        },
    };

    let mut service = CognitiveLoopService::new(config).expect("Failed to create service");
    let mut errors = Vec::with_capacity(n_cycles);

    for i in 0..n_cycles {
        let result = service.cycle(words[i % words.len()]);
        errors.push(result.prediction_error);
    }

    let initial_start = warmup;
    let initial_end = (warmup + window).min(errors.len());
    let initial_error = if initial_end > initial_start {
        errors[initial_start..initial_end].iter().sum::<f32>()
            / (initial_end - initial_start) as f32
    } else {
        1.0
    };

    let final_start = errors.len().saturating_sub(window);
    let final_error =
        errors[final_start..].iter().sum::<f32>() / (errors.len() - final_start) as f32;

    let reduction_pct = if initial_error > 0.0 {
        ((initial_error - final_error) / initial_error) * 100.0
    } else {
        0.0
    };

    let stats = service.stats();
    let learning_pct = if stats.total_cycles > 0 {
        stats.learning_cycles as f32 / stats.total_cycles as f32 * 100.0
    } else {
        0.0
    };

    AblationResult {
        name: name.to_string(),
        initial_error,
        final_error,
        reduction_pct,
        final_phi: service.unified_psi(),
        learning_pct,
    }
}
