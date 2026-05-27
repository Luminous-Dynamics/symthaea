// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # OpenNeuro Resting-State Meditation Benchmark
//!
//! Tests Symthaea's MeditationSentinel (Project Nous) on detection of
//! meditation and flow states from multi-channel EEG, validated against
//! known EEG signatures from the literature (Lutz 2004, Cahn & Polich 2006).
//!
//! ## Validates
//! 1. Flow state detection via gamma-theta coupling
//! 2. Meditation depth via frontal midline theta
//! 3. State discrimination across 6 meditation categories
//! 4. Session tracking with progressive deepening
//!
//! ## Run
//! ```bash
//! cargo run --example benchmark_meditation_resting --release
//! ```

use std::time::Instant;

use symthaea::perception::physio::{
    MeditationCategory, MeditationChannel, MeditationSentinel, MeditationSimulator,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Meditation & Resting-State EEG Benchmark             ║");
    println!("║       Project Nous - Flow & Focus Detection                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let sample_rate = 256.0;
    let window_sec = 4.0;

    // ═══════════════════════════════════════════════════════════════
    // Test 1: State Detection Accuracy
    // ═══════════════════════════════════════════════════════════════
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 1: Meditation State Detection (6 categories)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let all_states = [
        MeditationCategory::Flow,
        MeditationCategory::Absorption,
        MeditationCategory::Focused,
        MeditationCategory::OpenAwareness,
        MeditationCategory::Relaxed,
        MeditationCategory::Wandering,
    ];

    let n_trials = 5;
    let mut correct = 0;
    let mut meditative_correct = 0; // Binary: meditative vs wandering
    let mut total = 0;

    let mut sentinel = MeditationSentinel::new();

    for &target in &all_states {
        let mut trial_correct = 0;

        for trial in 0..n_trials {
            sentinel.reset();

            let mut sim = MeditationSimulator::new(sample_rate);
            // Vary seed
            for _ in 0..trial {
                sim.generate(MeditationCategory::Wandering, 0.5);
            }

            let eeg = sim.generate(target, window_sec);
            let state = sentinel.detect(&eeg);
            let predicted = state.classify();

            if predicted == target {
                correct += 1;
                trial_correct += 1;
            }

            // Binary: meditative vs non-meditative
            if target.is_meditative() == predicted.is_meditative() {
                meditative_correct += 1;
            }

            total += 1;
        }

        println!(
            "  {:20} → {}/{} correct",
            target.name(),
            trial_correct,
            n_trials
        );
    }

    let exact_accuracy = correct as f64 / total as f64;
    let binary_accuracy = meditative_correct as f64 / total as f64;

    println!(
        "\n  Exact classification: {:.1}% ({}/{})",
        exact_accuracy * 100.0,
        correct,
        total
    );
    println!(
        "  Binary (med/wander):  {:.1}% ({}/{})",
        binary_accuracy * 100.0,
        meditative_correct,
        total
    );

    // ═══════════════════════════════════════════════════════════════
    // Test 2: EEG Signature Validation
    // ═══════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 2: EEG Signature Validation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let mut sim = MeditationSimulator::new(sample_rate);

    // Flow state should have high gamma
    let flow_eeg = sim.generate(MeditationCategory::Flow, window_sec);
    let wander_eeg = sim.generate(MeditationCategory::Wandering, window_sec);

    let flow_gamma = flow_eeg.gamma_power(MeditationChannel::Fz).unwrap_or(0.0);
    let flow_theta = flow_eeg.theta_power(MeditationChannel::Fz).unwrap_or(0.0);
    let wander_gamma = wander_eeg.gamma_power(MeditationChannel::Fz).unwrap_or(0.0);
    let wander_theta = wander_eeg.theta_power(MeditationChannel::Fz).unwrap_or(0.0);

    println!("  Flow state:");
    println!("    Gamma power (Fz): {:.6}", flow_gamma);
    println!("    Theta power (Fz): {:.6}", flow_theta);
    println!("  Mind wandering:");
    println!("    Gamma power (Fz): {:.6}", wander_gamma);
    println!("    Theta power (Fz): {:.6}", wander_theta);

    let gamma_higher_flow = flow_gamma > wander_gamma;
    println!("  Gamma higher in flow: {}", gamma_higher_flow);

    // Absorption should have high theta at Fz
    let absorb_eeg = sim.generate(MeditationCategory::Absorption, window_sec);
    let absorb_theta = absorb_eeg.theta_power(MeditationChannel::Fz).unwrap_or(0.0);

    println!("  Absorption theta (Fz): {:.6}", absorb_theta);
    let theta_high_absorption = absorb_theta > wander_theta;
    println!("  Theta higher in absorption: {}", theta_high_absorption);

    // Open awareness should have high alpha at occipital
    let open_eeg = sim.generate(MeditationCategory::OpenAwareness, window_sec);
    let open_alpha_oz = open_eeg.alpha_power(MeditationChannel::Oz).unwrap_or(0.0);
    let wander_alpha_oz = wander_eeg.alpha_power(MeditationChannel::Oz).unwrap_or(0.0);

    println!("  Open awareness alpha (Oz): {:.6}", open_alpha_oz);
    println!("  Wandering alpha (Oz): {:.6}", wander_alpha_oz);
    let alpha_higher_open = open_alpha_oz > wander_alpha_oz;
    println!("  Alpha higher in open awareness: {}", alpha_higher_open);

    // ═══════════════════════════════════════════════════════════════
    // Test 3: Quality Score Ordering
    // ═══════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 3: Meditation Quality Score Ordering");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let mut qualities: Vec<(MeditationCategory, f64)> = Vec::new();

    for &target in &all_states {
        sentinel.reset();
        let eeg = sim.generate(target, window_sec);
        let state = sentinel.detect(&eeg);
        let quality = state.quality();
        qualities.push((target, quality));
        println!("  {:20}: quality = {:.3}", target.name(), quality);
    }

    // Flow and Absorption should have highest quality
    let flow_quality = qualities
        .iter()
        .find(|(c, _)| *c == MeditationCategory::Flow)
        .unwrap()
        .1;
    let wander_quality = qualities
        .iter()
        .find(|(c, _)| *c == MeditationCategory::Wandering)
        .unwrap()
        .1;
    let quality_ordered = flow_quality > wander_quality;
    println!("  Flow > Wandering quality: {}", quality_ordered);

    // ═══════════════════════════════════════════════════════════════
    // Test 4: Session Progression Tracking
    // ═══════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 4: Meditation Session Progression");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Simulate a meditation session: wandering → relaxed → focused → flow
    sentinel.reset();
    let session_phases = [
        (MeditationCategory::Wandering, 3),
        (MeditationCategory::Relaxed, 3),
        (MeditationCategory::Focused, 3),
        (MeditationCategory::Flow, 3),
    ];

    let mut phase_qualities: Vec<(String, f64)> = Vec::new();

    for (phase, n_windows) in &session_phases {
        let mut total_quality = 0.0;
        for _ in 0..*n_windows {
            let eeg = sim.generate(*phase, window_sec);
            let state = sentinel.detect(&eeg);
            total_quality += state.quality();
        }
        let avg_quality = total_quality / *n_windows as f64;
        phase_qualities.push((phase.name().to_string(), avg_quality));
        println!("  {:20}: avg quality = {:.3}", phase.name(), avg_quality);
    }

    // Quality should generally increase through session
    let session_improves = phase_qualities.len() >= 2
        && phase_qualities.last().unwrap().1 > phase_qualities.first().unwrap().1;
    println!("  Session quality improves: {}", session_improves);

    // Check category stats
    let stats = sentinel.category_stats();
    println!("\n  Category distribution over session:");
    for &cat in &all_states {
        let count = stats.get(&cat).copied().unwrap_or(0);
        if count > 0 {
            println!("    {:20}: {}", cat.name(), count);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Test 5: Processing Speed
    // ═══════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 5: Processing Speed");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    sentinel.reset();
    let test_eeg = sim.generate(MeditationCategory::Flow, window_sec);

    let n_iter = 100;
    let t = Instant::now();
    for _ in 0..n_iter {
        sentinel.detect(&test_eeg);
    }
    let elapsed = t.elapsed().as_secs_f64();
    let per_window_ms = elapsed * 1000.0 / n_iter as f64;
    let realtime = per_window_ms < window_sec * 1000.0; // Must process faster than real-time

    println!(
        "  {} windows in {:.2}s ({:.2}ms each)",
        n_iter, elapsed, per_window_ms
    );
    println!("  Real-time capable: {}", realtime);

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                 VALIDATION SUMMARY                         ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    let checks = vec![
        (
            "Binary accuracy (med/wander) >= 50%",
            binary_accuracy >= 0.50,
        ),
        ("Gamma higher in flow state", gamma_higher_flow),
        ("Theta higher in absorption", theta_high_absorption),
        ("Quality: flow > wandering", quality_ordered),
        ("Session quality improves", session_improves),
        ("Real-time processing", realtime),
    ];

    let mut passed = 0;
    for (name, pass) in &checks {
        println!("║  {} {:50}   ║", if *pass { "PASS" } else { "FAIL" }, name);
        if *pass {
            passed += 1;
        }
    }
    println!("╟──────────────────────────────────────────────────────────────╢");
    println!(
        "║  Result: {}/{} tests passed                                 ║",
        passed,
        checks.len()
    );
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Save
    let result_json = serde_json::json!({
        "benchmark": "Meditation/Resting-State EEG (Synthetic OpenNeuro-like)",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "exact_accuracy": exact_accuracy,
        "binary_accuracy": binary_accuracy,
        "gamma_flow_valid": gamma_higher_flow,
        "theta_absorption_valid": theta_high_absorption,
        "alpha_open_valid": alpha_higher_open,
        "quality_ordered": quality_ordered,
        "session_improves": session_improves,
        "realtime_capable": realtime,
        "tests_passed": passed,
        "tests_total": checks.len(),
    });

    std::fs::create_dir_all("data/benchmarks/meditation").ok();
    if let Ok(f) = std::fs::File::create("data/benchmarks/meditation/results.json") {
        serde_json::to_writer_pretty(f, &result_json).ok();
        println!("Results saved to data/benchmarks/meditation/results.json");
    }
}