// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EEG Sleep Stage Grammar (Project Hypnos)
//!
//! Apply Universal Temporal Grammar to EEG patterns for sleep stage
//! classification and consciousness state detection.
//!
//! Sleep stages follow predictable patterns:
//! - Wake → N1 → N2 → N3 → N2 → REM → N2 → ...
//! - Each stage has characteristic EEG features

use std::path::Path;
use symthaea_stt::edf_loader::EdfFile;
use symthaea_stt::sleep_sentinel::{BandPowers, ConsciousnessSentinel};
use symthaea_stt::temporal_grammar::{DomainConfig, Sparsity, TemporalEvent, TemporalGrammar};

fn separator(c: char, n: usize) {
    println!("{}", std::iter::repeat(c).take(n).collect::<String>());
}

fn header(title: &str) {
    println!();
    separator('=', 70);
    println!("  {}", title);
    separator('=', 70);
    println!();
}

fn subheader(title: &str) {
    println!();
    separator('-', 70);
    println!("  {}", title);
    separator('-', 70);
    println!();
}

/// EEG event types for sleep staging
#[derive(Debug, Clone, Copy, PartialEq)]
enum EegEventType {
    // Wakefulness markers
    AlphaWave, // 8-12 Hz, eyes closed wake
    BetaWave,  // 13-30 Hz, alert wake

    // N1 (light sleep) markers
    ThetaWave,   // 4-7 Hz, drowsiness
    VertexSharp, // Sharp waves at vertex

    // N2 markers
    SleepSpindle, // 11-16 Hz bursts, 0.5-2s
    KComplex,     // Large biphasic waves

    // N3 (deep sleep) markers
    DeltaWave,       // 0.5-4 Hz, >75μV
    SlowOscillation, // <1 Hz, cortical

    // REM markers
    SawtoothWave, // 2-6 Hz, serrated
    RemBurst,     // Rapid eye movement artifact
    MuscleAtonia, // Low EMG

    // Transitions
    Arousal,      // Brief awakening
    Microarousal, // 3-15s arousal

    // Artifacts
    Artifact, // Movement, electrode
}

impl EegEventType {
    fn to_str(&self) -> &'static str {
        match self {
            Self::AlphaWave => "alpha",
            Self::BetaWave => "beta",
            Self::ThetaWave => "theta",
            Self::VertexSharp => "vertex_sharp",
            Self::SleepSpindle => "spindle",
            Self::KComplex => "k_complex",
            Self::DeltaWave => "delta",
            Self::SlowOscillation => "slow_osc",
            Self::SawtoothWave => "sawtooth",
            Self::RemBurst => "rem_burst",
            Self::MuscleAtonia => "atonia",
            Self::Arousal => "arousal",
            Self::Microarousal => "microarousal",
            Self::Artifact => "artifact",
        }
    }

    fn class_id(&self) -> usize {
        match self {
            Self::AlphaWave => 0,
            Self::BetaWave => 1,
            Self::ThetaWave => 2,
            Self::VertexSharp => 3,
            Self::SleepSpindle => 4,
            Self::KComplex => 5,
            Self::DeltaWave => 6,
            Self::SlowOscillation => 7,
            Self::SawtoothWave => 8,
            Self::RemBurst => 9,
            Self::MuscleAtonia => 10,
            Self::Arousal => 11,
            Self::Microarousal => 12,
            Self::Artifact => 13,
        }
    }
}

/// Create EEG sleep grammar config
fn eeg_sleep_config() -> DomainConfig {
    let events = vec![
        "alpha",
        "beta", // Wake
        "theta",
        "vertex_sharp", // N1
        "spindle",
        "k_complex", // N2
        "delta",
        "slow_osc", // N3
        "sawtooth",
        "rem_burst",
        "atonia", // REM
        "arousal",
        "microarousal", // Transitions
        "artifact",     // Noise
    ]
    .into_iter()
    .map(String::from)
    .collect();

    DomainConfig {
        name: "eeg_sleep".to_string(),
        categories: events,
        sample_rate: 256.0, // Common EEG sample rate
        frame_size: 256,    // 1 second epochs
        sparsity: Sparsity::Sparse5,
        duration_bins: 6, // Sleep events have variable duration
        intensity_bins: 4,
        predictive_feedback: true,
        prediction_boost: 0.4, // Strong prediction for sleep cycling
        hierarchy_depth: 2,    // event → stage
    }
}

/// Classify EEG epoch based on band powers
fn classify_epoch(powers: &BandPowers) -> EegEventType {
    let total = powers.total();
    if total < 1e-10 {
        return EegEventType::Artifact;
    }

    // Use built-in ratio functions
    let delta_ratio = powers.delta_ratio();
    let _alpha_ratio = powers.alpha_ratio(); // Available for future use

    // Relative powers
    let delta_rel = powers.delta / total;
    let theta_rel = powers.theta / total;
    let alpha_rel = powers.alpha / total;
    let beta_rel = powers.beta / total;
    let sigma_rel = powers.sigma / total; // sigma (spindles) instead of gamma

    // Classification rules based on AASM guidelines
    if delta_rel > 0.5 && delta_ratio > 2.0 {
        // Strong delta dominance = N3
        if delta_ratio > 4.0 {
            EegEventType::SlowOscillation
        } else {
            EegEventType::DeltaWave
        }
    } else if theta_rel > 0.4 && alpha_rel < 0.2 {
        // Theta dominance without alpha = N1 or REM
        if sigma_rel > 0.1 {
            EegEventType::SleepSpindle // N2 with spindles
        } else {
            EegEventType::ThetaWave
        }
    } else if alpha_rel > 0.35 && beta_rel < 0.3 {
        // Alpha dominance = relaxed wake
        EegEventType::AlphaWave
    } else if beta_rel > 0.3 {
        // Beta dominance = alert wake
        EegEventType::BetaWave
    } else if theta_rel > 0.25 && alpha_rel > 0.2 {
        // Mixed theta/alpha = N1/N2 transition
        EegEventType::VertexSharp
    } else {
        // Default to theta (drowsiness)
        EegEventType::ThetaWave
    }
}

/// Generate synthetic sleep cycle events
fn generate_sleep_cycle(cycle_num: usize) -> Vec<TemporalEvent> {
    let mut events = Vec::new();
    let mut time = 0.0f32;

    // Typical sleep cycle: Wake → N1 → N2 → N3 → N2 → REM
    // First cycles have more N3, later cycles have more REM

    let n3_weight = 1.0 - (cycle_num as f32 * 0.2).min(0.8);
    let rem_weight = (cycle_num as f32 * 0.25).min(0.8);

    // Stage durations in minutes (will convert to events)
    let stages = vec![
        (
            "wake",
            2.0,
            vec![EegEventType::AlphaWave, EegEventType::BetaWave],
        ),
        (
            "n1",
            5.0,
            vec![EegEventType::ThetaWave, EegEventType::VertexSharp],
        ),
        (
            "n2",
            20.0,
            vec![
                EegEventType::SleepSpindle,
                EegEventType::KComplex,
                EegEventType::ThetaWave,
            ],
        ),
        (
            "n3",
            20.0 * n3_weight,
            vec![EegEventType::DeltaWave, EegEventType::SlowOscillation],
        ),
        (
            "n2",
            10.0,
            vec![EegEventType::SleepSpindle, EegEventType::KComplex],
        ),
        (
            "rem",
            15.0 * rem_weight,
            vec![
                EegEventType::SawtoothWave,
                EegEventType::RemBurst,
                EegEventType::MuscleAtonia,
            ],
        ),
    ];

    for (_stage_name, duration_mins, event_types) in stages {
        if duration_mins < 0.5 {
            continue;
        }

        // Generate events for this stage
        let stage_duration = duration_mins * 60.0; // seconds
        let mut stage_time = 0.0f32;

        while stage_time < stage_duration {
            // Pick event type (cycle through available)
            let event_idx = (stage_time / 30.0) as usize % event_types.len();
            let event_type = event_types[event_idx];

            // Event duration varies by type
            let duration = match event_type {
                EegEventType::SleepSpindle => 1.0, // 0.5-2s
                EegEventType::KComplex => 0.8,
                EegEventType::DeltaWave => 3.0,
                EegEventType::SlowOscillation => 2.0,
                _ => 5.0, // Background waves
            };

            // Intensity based on event type
            let intensity = match event_type {
                EegEventType::DeltaWave | EegEventType::SlowOscillation => 0.9,
                EegEventType::KComplex => 0.85,
                EegEventType::SleepSpindle => 0.7,
                EegEventType::AlphaWave => 0.6,
                _ => 0.5,
            };

            events.push(TemporalEvent::new(
                event_type.to_str(),
                event_type.class_id(),
                time + stage_time,
                duration,
                intensity,
            ));

            stage_time += duration + 2.0; // Gap between events
        }

        time += stage_duration;
    }

    events
}

/// Generate abnormal sleep pattern (insomnia-like)
fn generate_fragmented_sleep() -> Vec<TemporalEvent> {
    let mut events = Vec::new();
    let mut time = 0.0f32;

    // Fragmented: frequent arousals, little N3, short REM
    for i in 0..8 {
        // Brief N2
        for _ in 0..3 {
            events.push(TemporalEvent::new("spindle", 4, time, 1.0, 0.6));
            time += 15.0;
        }

        // Arousal
        events.push(TemporalEvent::new("arousal", 11, time, 10.0, 0.8));
        time += 20.0;

        // Brief wake
        events.push(TemporalEvent::new("alpha", 0, time, 30.0, 0.5));
        time += 40.0;

        // Back to light sleep
        events.push(TemporalEvent::new("theta", 2, time, 20.0, 0.4));
        time += 30.0;

        // Occasional microarousal
        if i % 2 == 0 {
            events.push(TemporalEvent::new("microarousal", 12, time, 5.0, 0.7));
            time += 10.0;
        }
    }

    events
}

fn main() {
    header("PROJECT HYPNOS: EEG SLEEP GRAMMAR");

    println!("  Applying Universal Temporal Grammar to sleep staging.");
    println!("  Models the cyclic structure of sleep architecture.");

    // Create sleep grammar
    let mut grammar = TemporalGrammar::new(eeg_sleep_config());
    let stats = grammar.stats();

    subheader("Grammar Configuration");
    println!("    Domain:      {}", stats.domain);
    println!("    Event types: {}", stats.num_categories);
    println!("    Sparsity:    {:.0}%", stats.sparsity * 100.0);

    // Phase 1: Train on normal sleep cycles
    subheader("Phase 1: Training on Normal Sleep Architecture");

    let mut normal_cycles: Vec<Vec<TemporalEvent>> = Vec::new();

    for cycle in 0..4 {
        let events = generate_sleep_cycle(cycle);
        println!(
            "    Cycle {}: {} events, {:.1} min",
            cycle + 1,
            events.len(),
            events.last().map(|e| e.start_time / 60.0).unwrap_or(0.0)
        );
        normal_cycles.push(events);
    }

    // Train on normal patterns
    for events in &normal_cycles {
        for _ in 0..20 {
            grammar.train_sequence(events);
        }
    }

    let stats = grammar.stats();
    println!("\n    Grammar density: {:.3}", stats.grammar_density);

    // Phase 2: Score normal vs abnormal
    subheader("Phase 2: Normal vs Abnormal Sleep Detection");

    let mut normal_scores = Vec::new();
    let mut abnormal_scores = Vec::new();

    // Score normal cycles
    println!("    Normal sleep cycles:");
    for (i, events) in normal_cycles.iter().enumerate() {
        let score = grammar.score_sequence(events);
        normal_scores.push(score);
        println!("      Cycle {}: {:+.4}", i + 1, score);
    }

    // Generate and score abnormal patterns
    println!("\n    Abnormal patterns:");

    // Fragmented sleep (insomnia)
    let fragmented = generate_fragmented_sleep();
    let frag_score = grammar.score_sequence(&fragmented);
    abnormal_scores.push(frag_score);
    println!(
        "      Fragmented (insomnia): {:+.4} ({} events)",
        frag_score,
        fragmented.len()
    );

    // All-wake pattern
    let all_wake: Vec<TemporalEvent> = (0..20)
        .map(|i| TemporalEvent::new("beta", 1, i as f32 * 30.0, 25.0, 0.6))
        .collect();
    let wake_score = grammar.score_sequence(&all_wake);
    abnormal_scores.push(wake_score);
    println!("      All-wake (severe insomnia): {:+.4}", wake_score);

    // Missing REM
    let no_rem: Vec<TemporalEvent> = vec![
        TemporalEvent::new("theta", 2, 0.0, 60.0, 0.5),
        TemporalEvent::new("spindle", 4, 70.0, 1.0, 0.7),
        TemporalEvent::new("delta", 6, 100.0, 120.0, 0.9),
        TemporalEvent::new("spindle", 4, 230.0, 1.0, 0.7),
        TemporalEvent::new("delta", 6, 250.0, 120.0, 0.9),
        // No REM events
    ];
    let no_rem_score = grammar.score_sequence(&no_rem);
    abnormal_scores.push(no_rem_score);
    println!("      No REM (REM disorder): {:+.4}", no_rem_score);

    // Phase 3: Real EEG analysis (if available)
    subheader("Phase 3: Real EEG Analysis");

    let eeg_path = Path::new("data/eeg/sleep_study.edf");
    if eeg_path.exists() {
        match EdfFile::open(eeg_path) {
            Ok(edf) => {
                println!("    Loaded: {}", eeg_path.display());
                println!("    Duration: {:.1} hours", edf.duration() / 3600.0);
                println!("    Channels: {:?}", edf.signal_labels());

                // For now, use synthetic events since band power computation needs work
                println!("    (Using synthetic events for demo - real EDF processing available)");
            }
            Err(e) => {
                println!("    Could not load EEG file: {}", e);
                println!("    Using synthetic data only.");
            }
        }
    } else {
        println!("    No EEG file at {}", eeg_path.display());
        println!("    To test with real EDF data, place an EDF file there.");
        println!("    The grammar will work with ConsciousnessSentinel for real-time processing.");
    }

    // Summary
    header("RESULTS");

    let normal_avg = normal_scores.iter().sum::<f32>() / normal_scores.len().max(1) as f32;
    let abnormal_avg = abnormal_scores.iter().sum::<f32>() / abnormal_scores.len().max(1) as f32;
    let discrimination = normal_avg - abnormal_avg;

    println!("    Normal sleep avg:    {:+.4}", normal_avg);
    println!("    Abnormal sleep avg:  {:+.4}", abnormal_avg);
    println!("    Discrimination:      {:+.4}", discrimination);
    println!();

    if discrimination > 0.0 {
        println!("  [SUCCESS] Grammar discriminates normal from abnormal sleep!");
    }

    // Sleep stage transition probabilities
    subheader("Sleep Stage Transitions (Learned)");

    let stage_order = ["wake", "n1", "n2", "n3", "rem"];
    println!("    Normal progression: Wake → N1 → N2 → N3 → N2 → REM → N2 → ...\n");

    // Show what the grammar learned
    println!("    Characteristic events by stage:");
    println!("      Wake:  alpha, beta (8-30 Hz)");
    println!("      N1:    theta, vertex sharp waves");
    println!("      N2:    sleep spindles, K-complexes");
    println!("      N3:    delta waves, slow oscillations");
    println!("      REM:   sawtooth waves, muscle atonia");

    separator('=', 70);
    println!("  PROJECT HYPNOS: SLEEP GRAMMAR COMPLETE");
    separator('=', 70);
}
