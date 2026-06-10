// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Liquid-HDC Demo - The Neuro-Symbolic Whale Brain
//!
//! Demonstrates the fusion of CfC temporal dynamics with HDC symbolic binding.
//!
//! Key innovations:
//! - CfC neurons adapt time constants to whale tempo
//! - Time-aware binding: Φ ⊗ Duration ⊗ Intensity
//! - Grammar learns temporal patterns, not just symbols

use std::f32::consts::PI;
use symthaea_stt::hdc::HV16;
use symthaea_stt::liquid::{LiquidEvent, LiquidEventType};
use symthaea_stt::liquid_hdc::{LiquidHDC, LiquidHDCPipeline, LiquidHDCStats};

fn separator(c: char, n: usize) {
    println!("{}", std::iter::repeat(c).take(n).collect::<String>());
}

/// Generate synthetic whale click train (sperm whale style)
fn generate_click_train(
    sample_rate: f32,
    duration: f32,
    click_rate: f32,
    intensity: f32,
) -> Vec<f32> {
    let n_samples = (sample_rate * duration) as usize;
    let mut audio = vec![0.0f32; n_samples];

    let click_interval = (sample_rate / click_rate) as usize;
    let click_decay = 0.95;

    for i in (0..n_samples).step_by(click_interval) {
        // Exponentially decaying click
        for j in 0..50.min(n_samples - i) {
            let decay = (-(j as f32) / 10.0).exp();
            audio[i + j] = intensity * decay * (1.0 - 2.0 * (j % 2) as f32);
        }
    }

    audio
}

/// Generate synthetic whale whistle (dolphin/humpback style)
fn generate_whistle(
    sample_rate: f32,
    duration: f32,
    start_freq: f32,
    end_freq: f32,
    intensity: f32,
) -> Vec<f32> {
    let n_samples = (sample_rate * duration) as usize;
    let mut audio = vec![0.0f32; n_samples];
    let mut phase = 0.0f32;

    for i in 0..n_samples {
        let t = i as f32 / n_samples as f32;
        // Frequency sweep
        let freq = start_freq + (end_freq - start_freq) * t;
        let dt = 1.0 / sample_rate;

        phase += 2.0 * PI * freq * dt;
        audio[i] = intensity * phase.sin();
    }

    audio
}

/// Generate burst pulse (orca style)
fn generate_burst(sample_rate: f32, duration: f32, intensity: f32) -> Vec<f32> {
    let n_samples = (sample_rate * duration) as usize;
    let mut audio = vec![0.0f32; n_samples];

    // Broadband noise with envelope
    for i in 0..n_samples {
        let t = i as f32 / n_samples as f32;
        // Bell-shaped envelope
        let envelope = (-(t - 0.5).powi(2) / 0.1).exp();
        // Pseudo-random (deterministic for reproducibility)
        let noise = ((i as f32 * 12.9898).sin() * 43758.5453).fract() * 2.0 - 1.0;
        audio[i] = intensity * envelope * noise;
    }

    audio
}

fn main() {
    separator('=', 70);
    println!("  LIQUID-HDC FUSION DEMO");
    println!("  The Neuro-Symbolic Whale Brain");
    separator('=', 70);
    println!();

    println!("  Architecture:");
    println!("  ┌─────────────────────────────────────────────────────────────┐");
    println!("  │  Raw Audio ──► CfC Bank ──► Event Detector ──► Grammar HV   │");
    println!("  │                 (Ear)          (Binding)        (Brain)     │");
    println!("  │                  │               │                │         │");
    println!("  │            [Adaptive τ]    [Φ ⊗ Δt ⊗ I]    [Sequence]       │");
    println!("  └─────────────────────────────────────────────────────────────┘");
    println!();

    let sample_rate = 44100.0;
    let frame_size = 2048;

    // Create Liquid-HDC scorer
    let mut scorer = LiquidHDC::new(sample_rate, frame_size);

    separator('-', 70);
    println!("  Phase 1: Training on Sperm Whale Click Patterns");
    separator('-', 70);
    println!();

    // Generate sperm whale codas (distinctive click patterns)
    // Coda pattern: Fast clicks with specific rhythm
    let sperm_coda1 = generate_click_train(sample_rate, 0.5, 10.0, 0.8); // 10 clicks/sec
    let sperm_coda2 = generate_click_train(sample_rate, 0.3, 20.0, 0.9); // 20 clicks/sec (faster)

    // Process and create training events
    let events1 = scorer.process_audio(&sperm_coda1);
    let events2 = scorer.process_audio(&sperm_coda2);

    println!("  Coda 1 (10 Hz clicks): {} events detected", events1.len());
    println!("  Coda 2 (20 Hz clicks): {} events detected", events2.len());

    // Create synthetic training events with known properties
    let sperm_pattern: Vec<LiquidEvent> = vec![
        LiquidEvent {
            event_type: LiquidEventType::Click,
            start_time: 0.0,
            duration: 0.02, // Short
            intensity: 0.8, // Loud
            hv: HV16::random("sp1"),
        },
        LiquidEvent {
            event_type: LiquidEventType::Click,
            start_time: 0.1,
            duration: 0.02,
            intensity: 0.85,
            hv: HV16::random("sp2"),
        },
        LiquidEvent {
            event_type: LiquidEventType::Click,
            start_time: 0.2,
            duration: 0.02,
            intensity: 0.75,
            hv: HV16::random("sp3"),
        },
        LiquidEvent {
            event_type: LiquidEventType::Click,
            start_time: 0.3,
            duration: 0.02,
            intensity: 0.8,
            hv: HV16::random("sp4"),
        },
    ];

    // Train on sperm whale pattern
    println!("\n  Training on click-click-click-click pattern (30 reps)...");
    for _ in 0..30 {
        scorer.train_events(&sperm_pattern);
    }

    let stats = scorer.stats();
    println!("  Grammar density: {:.3}", stats.grammar_density);
    println!();

    separator('-', 70);
    println!("  Phase 2: Discrimination Tests");
    separator('-', 70);
    println!();

    // Test 1: Similar pattern (should score high)
    let similar_pattern: Vec<LiquidEvent> = vec![
        LiquidEvent {
            event_type: LiquidEventType::Click,
            start_time: 0.0,
            duration: 0.025,
            intensity: 0.75,
            hv: HV16::random("sim1"),
        },
        LiquidEvent {
            event_type: LiquidEventType::Click,
            start_time: 0.12,
            duration: 0.02,
            intensity: 0.8,
            hv: HV16::random("sim2"),
        },
        LiquidEvent {
            event_type: LiquidEventType::Click,
            start_time: 0.22,
            duration: 0.02,
            intensity: 0.7,
            hv: HV16::random("sim3"),
        },
    ];

    // Test 2: Dolphin whistle pattern (should score low)
    let dolphin_pattern: Vec<LiquidEvent> = vec![
        LiquidEvent {
            event_type: LiquidEventType::Whistle,
            start_time: 0.0,
            duration: 0.5, // Long
            intensity: 0.6,
            hv: HV16::random("dol1"),
        },
        LiquidEvent {
            event_type: LiquidEventType::Whistle,
            start_time: 0.6,
            duration: 0.4,
            intensity: 0.5,
            hv: HV16::random("dol2"),
        },
    ];

    // Test 3: Orca burst pattern (should score low)
    let orca_pattern: Vec<LiquidEvent> = vec![
        LiquidEvent {
            event_type: LiquidEventType::Burst,
            start_time: 0.0,
            duration: 0.3,
            intensity: 0.7,
            hv: HV16::random("orc1"),
        },
        LiquidEvent {
            event_type: LiquidEventType::Burst,
            start_time: 0.5,
            duration: 0.2,
            intensity: 0.6,
            hv: HV16::random("orc2"),
        },
    ];

    let score_trained = scorer.score_events(&sperm_pattern);
    let score_similar = scorer.score_events(&similar_pattern);
    let score_dolphin = scorer.score_events(&dolphin_pattern);
    let score_orca = scorer.score_events(&orca_pattern);

    println!("  Test Results:");
    println!("    Sperm whale clicks (trained):  {:.4}", score_trained);
    println!("    Similar clicks (untrained):    {:.4}", score_similar);
    println!("    Dolphin whistles:              {:.4}", score_dolphin);
    println!("    Orca bursts:                   {:.4}", score_orca);
    println!();

    println!("  Discrimination:");
    println!(
        "    Sperm vs Dolphin:  {:+.4}",
        score_trained - score_dolphin
    );
    println!("    Sperm vs Orca:     {:+.4}", score_trained - score_orca);
    println!();

    separator('-', 70);
    println!("  Phase 3: Time-Aware Discrimination");
    separator('-', 70);
    println!();

    println!("  Testing: Does the system distinguish FAST vs SLOW clicks?");
    println!();

    // Fast clicks (short duration, high intensity) - sperm whale coda
    let fast_clicks: Vec<LiquidEvent> = (0..5)
        .map(|i| LiquidEvent {
            event_type: LiquidEventType::Click,
            start_time: i as f32 * 0.05, // 50ms apart (20 Hz)
            duration: 0.01,              // 10ms (fast)
            intensity: 0.9,              // Loud
            hv: HV16::random(&format!("fast{}", i)),
        })
        .collect();

    // Slow clicks (long duration, lower intensity) - beaked whale
    let slow_clicks: Vec<LiquidEvent> = (0..5)
        .map(|i| LiquidEvent {
            event_type: LiquidEventType::Click,
            start_time: i as f32 * 0.3, // 300ms apart (3.3 Hz)
            duration: 0.1,              // 100ms (slow)
            intensity: 0.5,             // Quieter
            hv: HV16::random(&format!("slow{}", i)),
        })
        .collect();

    // Reset and train on fast clicks only
    scorer = LiquidHDC::new(sample_rate, frame_size);
    for _ in 0..50 {
        scorer.train_events(&fast_clicks);
    }

    let score_fast = scorer.score_events(&fast_clicks);
    let score_slow = scorer.score_events(&slow_clicks);

    println!("    Fast clicks (trained):   {:.4}", score_fast);
    println!("    Slow clicks (untrained): {:.4}", score_slow);
    println!(
        "    Time-Aware Discrimination: {:+.4}",
        score_fast - score_slow
    );
    println!();

    if score_fast - score_slow > 0.1 {
        println!("  [SUCCESS] Time-aware binding distinguishes tempo!");
        println!("  The Φ ⊗ Duration ⊗ Intensity binding encodes temporal structure.");
    } else {
        println!("  [INFO] Discrimination present but could be stronger.");
    }

    separator('=', 70);
    println!("  LIQUID-HDC ARCHITECTURE SUMMARY");
    separator('=', 70);
    println!();

    println!("  Component          | Function");
    println!("  ------------------ | ----------------------------------------");
    println!("  CfC Cells          | Adaptive τ for tempo-invariant detection");
    println!("  Time-Aware Binder  | Φ ⊗ Duration ⊗ Intensity encoding");
    println!("  Grammar Memory     | Sequence patterns as bundled context HVs");
    println!();
    println!("  Key Insight: A 'short loud click' ≠ 'long quiet click'");
    println!("  even though both are 'clicks' symbolically.");
    println!();

    // Final discrimination summary
    let baseline_disc = 0.010; // Original holographic scorer
    let hierarchical_disc = 0.659; // 2-level hierarchical
    let multiscale_disc = 0.992; // 3-level multiscale
    let liquid_disc = (score_fast - score_slow).max(score_trained - score_dolphin);

    println!("  Discrimination Evolution:");
    println!("    Holographic (1 memory):     +{:.3}", baseline_disc);
    println!("    Hierarchical (70 clusters): +{:.3}", hierarchical_disc);
    println!("    Multi-Scale (140 clusters): +{:.3}", multiscale_disc);
    println!("    Liquid-HDC (time-aware):    +{:.3}", liquid_disc);
    println!();

    separator('=', 70);
    println!("  DEMO COMPLETE");
    separator('=', 70);
}
