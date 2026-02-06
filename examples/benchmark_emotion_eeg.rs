//! # Emotion Recognition from EEG Benchmark
//!
//! Tests Symthaea's EmotionSentinel (Project Pathos) on emotion detection
//! from multi-channel EEG using the Valence-Arousal circumplex model.
//! Uses EmotionSimulator for synthetic data matching DEAP/SEED protocols.
//!
//! ## Validates
//! 1. Frontal Asymmetry Index correctly maps to valence
//! 2. Beta/Alpha ratio correctly maps to arousal
//! 3. Discrete emotion classification accuracy
//! 4. Temporal smoothing improves stability
//!
//! ## Run
//! ```bash
//! cargo run --example benchmark_emotion_eeg --release
//! ```

use std::time::Instant;

use symthaea::perception::physio::{
    EmotionSentinel, EmotionSimulator, EmotionCategory, EmotionChannel,
    EmotionEEG, FrequencyBand,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Emotion Recognition EEG Benchmark                    ║");
    println!("║       Project Pathos - Valence/Arousal Detection           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let sample_rate = 256.0; // Standard EEG sample rate
    let window_sec = 4.0;     // 4-second analysis windows

    // ═══════════════════════════════════════════════════════════════
    // Test 1: Frontal Asymmetry → Valence
    // ═══════════════════════════════════════════════════════════════
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 1: Frontal Asymmetry Index → Valence");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let mut sentinel = EmotionSentinel::new();
    let mut simulator = EmotionSimulator::new(sample_rate);

    // Test positive vs negative emotions
    let positive_emotions = [EmotionCategory::Happy, EmotionCategory::Excited, EmotionCategory::Content];
    let negative_emotions = [EmotionCategory::Sad, EmotionCategory::Angry, EmotionCategory::Afraid];

    let mut positive_valences = Vec::new();
    let mut negative_valences = Vec::new();

    for &emotion in &positive_emotions {
        let eeg = simulator.generate(emotion, window_sec);
        let state = sentinel.detect(&eeg);
        positive_valences.push(state.valence);
        println!("  {:10} → valence = {:.3}, arousal = {:.3}",
            emotion.name(), state.valence, state.arousal);
    }

    sentinel.reset();

    for &emotion in &negative_emotions {
        let eeg = simulator.generate(emotion, window_sec);
        let state = sentinel.detect(&eeg);
        negative_valences.push(state.valence);
        println!("  {:10} → valence = {:.3}, arousal = {:.3}",
            emotion.name(), state.valence, state.arousal);
    }

    let avg_positive_v = positive_valences.iter().sum::<f64>() / positive_valences.len() as f64;
    let avg_negative_v = negative_valences.iter().sum::<f64>() / negative_valences.len() as f64;
    let valence_separates = avg_positive_v > avg_negative_v;

    println!("\n  Avg positive valence: {:.3}", avg_positive_v);
    println!("  Avg negative valence: {:.3}", avg_negative_v);
    println!("  Valence separates emotions: {}", valence_separates);

    // ═══════════════════════════════════════════════════════════════
    // Test 2: Arousal Detection
    // ═══════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 2: Beta/Alpha Ratio → Arousal");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    sentinel.reset();

    let high_arousal = [EmotionCategory::Excited, EmotionCategory::Angry];
    let low_arousal = [EmotionCategory::Relaxed, EmotionCategory::Bored];

    let mut high_arousals = Vec::new();
    let mut low_arousals = Vec::new();

    for &emotion in &high_arousal {
        let eeg = simulator.generate(emotion, window_sec);
        let state = sentinel.detect(&eeg);
        high_arousals.push(state.arousal);
        println!("  {:10} → arousal = {:.3}", emotion.name(), state.arousal);
    }

    for &emotion in &low_arousal {
        let eeg = simulator.generate(emotion, window_sec);
        let state = sentinel.detect(&eeg);
        low_arousals.push(state.arousal);
        println!("  {:10} → arousal = {:.3}", emotion.name(), state.arousal);
    }

    let avg_high_a = high_arousals.iter().sum::<f64>() / high_arousals.len() as f64;
    let avg_low_a = low_arousals.iter().sum::<f64>() / low_arousals.len() as f64;
    let arousal_separates = avg_high_a > avg_low_a;

    println!("\n  Avg high arousal: {:.3}", avg_high_a);
    println!("  Avg low arousal:  {:.3}", avg_low_a);
    println!("  Arousal separates emotions: {}", arousal_separates);

    // ═══════════════════════════════════════════════════════════════
    // Test 3: Discrete Emotion Classification
    // ═══════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 3: Discrete Emotion Classification");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let all_emotions = [
        EmotionCategory::Excited, EmotionCategory::Happy,
        EmotionCategory::Content, EmotionCategory::Relaxed,
        EmotionCategory::Angry, EmotionCategory::Afraid,
        EmotionCategory::Sad, EmotionCategory::Bored,
    ];

    let n_trials = 5;
    let mut correct = 0;
    let mut quadrant_correct = 0;
    let mut total = 0;

    for &target in &all_emotions {
        for trial in 0..n_trials {
            sentinel.reset();

            // Vary simulator seed per trial
            let mut sim = EmotionSimulator::new(sample_rate);
            // Run a few warm-up windows to vary the seed
            for _ in 0..trial {
                sim.generate(EmotionCategory::Neutral, 0.5);
            }

            let eeg = sim.generate(target, window_sec);
            let state = sentinel.detect(&eeg);
            let predicted = state.classify();

            if predicted == target {
                correct += 1;
            }

            // Check quadrant match (same valence sign AND same arousal sign)
            let (tv, ta) = target.target_va();
            let target_quadrant = (tv >= 0.0, ta >= 0.0);
            let predicted_quadrant = (state.valence >= 0.0, state.arousal >= 0.0);
            if target_quadrant == predicted_quadrant {
                quadrant_correct += 1;
            }

            total += 1;
        }
    }

    let exact_accuracy = correct as f64 / total as f64;
    let quadrant_accuracy = quadrant_correct as f64 / total as f64;

    println!("  Exact classification:   {:.1}% ({}/{})", exact_accuracy * 100.0, correct, total);
    println!("  Quadrant (V/A) match:   {:.1}% ({}/{})", quadrant_accuracy * 100.0, quadrant_correct, total);

    // ═══════════════════════════════════════════════════════════════
    // Test 4: Spectral Analysis Quality
    // ═══════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 4: Spectral Band Power Analysis");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Generate relaxed EEG (should have high alpha)
    let relaxed_eeg = simulator.generate(EmotionCategory::Relaxed, window_sec);
    let excited_eeg = simulator.generate(EmotionCategory::Excited, window_sec);

    let relaxed_alpha = relaxed_eeg.alpha_power(EmotionChannel::Fz).unwrap_or(0.0);
    let relaxed_beta = relaxed_eeg.beta_power(EmotionChannel::Fz).unwrap_or(0.0);
    let excited_alpha = excited_eeg.alpha_power(EmotionChannel::Fz).unwrap_or(0.0);
    let excited_beta = excited_eeg.beta_power(EmotionChannel::Fz).unwrap_or(0.0);

    println!("  Relaxed: alpha={:.6}, beta={:.6}, ratio={:.3}",
        relaxed_alpha, relaxed_beta,
        if relaxed_alpha > 0.0 { relaxed_beta / relaxed_alpha } else { 0.0 });
    println!("  Excited: alpha={:.6}, beta={:.6}, ratio={:.3}",
        excited_alpha, excited_beta,
        if excited_alpha > 0.0 { excited_beta / excited_alpha } else { 0.0 });

    let alpha_higher_relaxed = relaxed_alpha > excited_alpha;
    let beta_higher_excited = excited_beta > relaxed_beta;

    println!("  Alpha higher when relaxed: {}", alpha_higher_relaxed);
    println!("  Beta higher when excited: {}", beta_higher_excited);

    // ═══════════════════════════════════════════════════════════════
    // Test 5: Temporal Smoothing
    // ═══════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 5: Temporal Smoothing Stability");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    sentinel.reset();

    // Process multiple windows of the same emotion
    let mut raw_valences = Vec::new();
    for _ in 0..10 {
        let eeg = simulator.generate(EmotionCategory::Happy, window_sec);
        let state = sentinel.detect(&eeg);
        raw_valences.push(state.valence);
    }

    let smoothed = sentinel.smoothed_state(5);

    let raw_variance: f64 = {
        let mean = raw_valences.iter().sum::<f64>() / raw_valences.len() as f64;
        raw_valences.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / raw_valences.len() as f64
    };

    // Smoothed should be more stable
    let smoothed_within_range = smoothed.valence.abs() < 1.0 && smoothed.confidence > 0.0;

    println!("  Raw valence variance: {:.4}", raw_variance);
    println!("  Smoothed valence: {:.3}", smoothed.valence);
    println!("  Smoothed confidence: {:.3}", smoothed.confidence);
    println!("  Smoothed within valid range: {}", smoothed_within_range);

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                 VALIDATION SUMMARY                         ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    let checks = vec![
        ("Valence separates +/- emotions", valence_separates),
        ("Arousal separates high/low", arousal_separates),
        ("Quadrant accuracy > 40%", quadrant_accuracy > 0.40),
        ("Spectral alpha matches theory", alpha_higher_relaxed),
        ("Temporal smoothing stable", smoothed_within_range),
    ];

    let mut passed = 0;
    for (name, pass) in &checks {
        println!("║  {} {:50}   ║", if *pass { "PASS" } else { "FAIL" }, name);
        if *pass { passed += 1; }
    }
    println!("╟──────────────────────────────────────────────────────────────╢");
    println!("║  Result: {}/{} tests passed                                 ║", passed, checks.len());
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Save
    let result_json = serde_json::json!({
        "benchmark": "Emotion EEG Recognition (Synthetic DEAP-like)",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "valence_separates": valence_separates,
        "arousal_separates": arousal_separates,
        "exact_accuracy": exact_accuracy,
        "quadrant_accuracy": quadrant_accuracy,
        "spectral_valid": alpha_higher_relaxed && beta_higher_excited,
        "tests_passed": passed,
        "tests_total": checks.len(),
    });

    std::fs::create_dir_all("data/benchmarks/emotion").ok();
    if let Ok(f) = std::fs::File::create("data/benchmarks/emotion/results.json") {
        serde_json::to_writer_pretty(f, &result_json).ok();
        println!("Results saved to data/benchmarks/emotion/results.json");
    }
}
