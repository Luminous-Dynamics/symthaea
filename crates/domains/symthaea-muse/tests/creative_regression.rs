// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Creative quality regression gate.
//!
//! Locks in the minimum acceptable quality for Symthaea's music synthesis.
//! If any metric drops below `BASELINE - TOLERANCE`, the test fails, alerting
//! that a code change has degraded creative output.
//!
//! # Updating baselines
//!
//! After a deliberate quality improvement, run:
//!
//! ```bash
//! cargo test -p symthaea-muse -- print_creative_benchmark --nocapture
//! ```
//!
//! Copy the printed scores into the BASELINE_* constants below.
//!
//! # Design philosophy
//!
//! These metrics are *proxy* measures, not ground truth. They catch regressions
//! (e.g., a synthesis bug that flattens all notes to the same pitch would
//! destroy melodic coherence). They are NOT a substitute for listening tests.

use symthaea_muse::{MuseConfig, MusicalState, creative_bench};

/// Regression tolerance: a metric may decline by at most this much before failing.
/// Allows for minor floating-point variance across platforms.
const TOLERANCE: f32 = 0.05;

// ─── Baselines ────────────────────────────────────────────────────────────────
// Established from the initial production implementation (Apr 2026).
// Run `print_creative_benchmark` to regenerate.

// Measured 2026-04-11 (post-rhythm+sparsity fix):
//   melodic=0.755, rhythm=0.933, emotion=0.463, form=0.606, composite=0.704
// Baselines set ~15% below measured to absorb cross-platform + stochastic variance.
const BASELINE_MELODIC_COHERENCE: f32 = 0.64;
const BASELINE_RHYTHMIC_REGULARITY: f32 = 0.78;
const BASELINE_EMOTIONAL_ALIGNMENT: f32 = 0.38;
const BASELINE_FORM_COMPLIANCE: f32 = 0.50;
const BASELINE_COMPOSITE: f32 = 0.59;

// ─── Standard benchmark config ────────────────────────────────────────────────

fn bench_config() -> MuseConfig {
    MuseConfig {
        duration_secs: 4.0,
        max_notes: 16,
        ..MuseConfig::default()
    }
}

// ─── Regression gate ──────────────────────────────────────────────────────────

#[test]
fn creative_quality_regression() {
    let result = creative_bench::run_benchmark(&bench_config(), &MusicalState::default());

    assert!(
        result.mean_melodic_coherence >= BASELINE_MELODIC_COHERENCE - TOLERANCE,
        "melodic coherence REGRESSION: {:.3} < baseline {:.3} - tolerance {:.3}\n{}",
        result.mean_melodic_coherence,
        BASELINE_MELODIC_COHERENCE,
        TOLERANCE,
        result.report(),
    );
    assert!(
        result.mean_rhythmic_regularity >= BASELINE_RHYTHMIC_REGULARITY - TOLERANCE,
        "rhythmic regularity REGRESSION: {:.3} < baseline {:.3} - tolerance {:.3}\n{}",
        result.mean_rhythmic_regularity,
        BASELINE_RHYTHMIC_REGULARITY,
        TOLERANCE,
        result.report(),
    );
    assert!(
        result.mean_emotional_alignment >= BASELINE_EMOTIONAL_ALIGNMENT - TOLERANCE,
        "emotional alignment REGRESSION: {:.3} < baseline {:.3} - tolerance {:.3}\n{}",
        result.mean_emotional_alignment,
        BASELINE_EMOTIONAL_ALIGNMENT,
        TOLERANCE,
        result.report(),
    );
    assert!(
        result.mean_form_compliance >= BASELINE_FORM_COMPLIANCE - TOLERANCE,
        "form compliance REGRESSION: {:.3} < baseline {:.3} - tolerance {:.3}\n{}",
        result.mean_form_compliance,
        BASELINE_FORM_COMPLIANCE,
        TOLERANCE,
        result.report(),
    );
    assert!(
        result.mean_composite >= BASELINE_COMPOSITE - TOLERANCE,
        "composite score REGRESSION: {:.3} < baseline {:.3} - tolerance {:.3}\n{}",
        result.mean_composite,
        BASELINE_COMPOSITE,
        TOLERANCE,
        result.report(),
    );
}

/// Regression gate for the neural melody mode (CfC-driven generation).
#[test]
fn neural_melody_regression() {
    let config = MuseConfig {
        duration_secs: 4.0,
        max_notes: 16,
        melody_mode: symthaea_muse::MelodyMode::Neural,
        ..MuseConfig::default()
    };
    let result = creative_bench::run_benchmark(&config, &MusicalState::default());

    // Neural mode has its own baseline (lower than classic due to CfC timing).
    const NEURAL_BASELINE_COMPOSITE: f32 = 0.35;
    assert!(
        result.mean_composite >= NEURAL_BASELINE_COMPOSITE - TOLERANCE,
        "neural melody composite REGRESSION: {:.3} < {:.3}\n{}",
        result.mean_composite,
        NEURAL_BASELINE_COMPOSITE - TOLERANCE,
        result.report(),
    );
}

/// Print current benchmark scores. Use this to update baselines after improvements.
///
/// Run with:  cargo test -p symthaea-muse -- print_creative_benchmark --nocapture
#[test]
fn print_creative_benchmark() {
    let result = creative_bench::run_benchmark(&bench_config(), &MusicalState::default());
    println!("\n{}", result.report());
    println!(
        "\nUpdate constants:\n\
         const BASELINE_MELODIC_COHERENCE: f32    = {:.3};\n\
         const BASELINE_RHYTHMIC_REGULARITY: f32  = {:.3};\n\
         const BASELINE_EMOTIONAL_ALIGNMENT: f32  = {:.3};\n\
         const BASELINE_FORM_COMPLIANCE: f32      = {:.3};\n\
         const BASELINE_COMPOSITE: f32            = {:.3};",
        result.mean_melodic_coherence,
        result.mean_rhythmic_regularity,
        result.mean_emotional_alignment,
        result.mean_form_compliance,
        result.mean_composite,
    );
}
