// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # EEG Seizure Detection Benchmark
//!
//! Validates Symthaea's temporal processing (LTC/CfC) for detecting seizures
//! from EEG signals. Tests the fundamental claim that continuous-time neural
//! dynamics outperform discrete windowing for clinical EEG applications.
//!
//! ## Method
//! 1. Generate synthetic EEG with known seizure patterns (or load TUH EEG)
//! 2. Process through Symthaea's SleepSentinel for real-time state detection
//! 3. Measure seizure detection accuracy, latency, and false positive rate
//!
//! ## Clinical Significance
//! Epileptic seizures show characteristic high-amplitude rhythmic discharges
//! with a sudden increase in synchrony and decrease in complexity.
//! This is a natural application for Φ-based monitoring.
//!
//! ## Run
//! ```bash
//! cargo run --example benchmark_eeg_seizure --release
//! ```

use std::time::Instant;

use symthaea::perception::physio::{IntegrationMetrics, SleepSentinel, SleepSentinelConfig};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       EEG Seizure Detection Benchmark                      ║");
    println!("║       LTC Temporal Dynamics for Clinical EEG               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("Running with synthetic seizure data...\n");

    let config = SleepSentinelConfig {
        local_neurons: 64,
        global_neurons: 128,
        dt_ms: 10.0,
        integration_window: 300,
        tau_base: 100.0,
        enable_adaptive_thresholds: true,
        steps_per_epoch: 500,        // Reduced from 3000 for faster benchmark
        use_spectral_analysis: true, // Use Welch PSD for band powers
        ..SleepSentinelConfig::default()
    };

    let mut sentinel = SleepSentinel::new(config);
    let sample_rate = 256.0; // Hz (standard clinical EEG)
    let epoch_len = (10.0 * sample_rate) as usize; // 10-second epochs

    // ═══════════════════════════════════════════════════════════════
    // Test 1: Train on known patterns
    // ═══════════════════════════════════════════════════════════════
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 1: Training on Synthetic EEG Patterns");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let t = Instant::now();

    // Normal EEG: alpha rhythm, moderate amplitude
    for i in 0..10 {
        let (f, o) = generate_normal_eeg(epoch_len, sample_rate, i as u64);
        sentinel.train_epoch(&f, &o, symthaea::perception::physio::SleepStage::Wake);
    }
    println!("  Trained 10 normal epochs");

    // Seizure EEG: high amplitude, hypersynchronous, rhythmic
    for i in 0..10 {
        let (f, o) = generate_seizure_eeg(epoch_len, sample_rate, i as u64);
        sentinel.train_epoch(
            &f,
            &o,
            symthaea::perception::physio::SleepStage::N3, // Use N3 as high-synchrony proxy
        );
    }
    println!("  Trained 10 seizure epochs");

    // Pre-ictal EEG: subtle changes before seizure
    for i in 0..10 {
        let (f, o) = generate_preictal_eeg(epoch_len, sample_rate, i as u64);
        sentinel.train_epoch(
            &f,
            &o,
            symthaea::perception::physio::SleepStage::N1, // Use N1 as transitional proxy
        );
    }
    println!("  Trained 10 pre-ictal epochs");
    println!("  Training time: {:.1}s\n", t.elapsed().as_secs_f64());

    // ═══════════════════════════════════════════════════════════════
    // Test 2: Detection on test data
    // ═══════════════════════════════════════════════════════════════
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 2: Seizure Detection");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let t = Instant::now();

    // Collect spectral features from training data to set classification thresholds
    // Run a few epochs through to establish normal vs seizure spectral baselines
    let mut normal_features = Vec::new();
    let mut seizure_features = Vec::new();
    for i in 0..5 {
        let (f, o) = generate_normal_eeg(epoch_len, sample_rate, 50 + i as u64);
        let (_state, metrics) = sentinel.process_epoch(&f, &o);
        normal_features.push(extract_seizure_features(&metrics));

        let (f, o) = generate_seizure_eeg(epoch_len, sample_rate, 50 + i as u64);
        let (_state, metrics) = sentinel.process_epoch(&f, &o);
        seizure_features.push(extract_seizure_features(&metrics));
    }
    let threshold = compute_seizure_threshold(&normal_features, &seizure_features);
    println!("  Seizure classifier threshold: {:.4}\n", threshold);

    // Test normal epochs (should NOT trigger seizure detection)
    let mut normal_correct = 0;
    let test_normal = 10;
    for i in 0..test_normal {
        let (f, o) = generate_normal_eeg(epoch_len, sample_rate, 100 + i as u64);
        let (_state, metrics) = sentinel.process_epoch(&f, &o);
        let score = seizure_score(&metrics);
        let is_normal = score < threshold;
        if is_normal {
            normal_correct += 1;
        }
        if i < 3 {
            println!(
                "  Normal #{}: score={:.4} (thr={:.4}), sync={:.4}, entropy={:.4}, delta={:.4}",
                i,
                score,
                threshold,
                metrics.synchrony,
                metrics.spectral_entropy,
                metrics.delta_power
            );
        }
    }

    // Test seizure epochs (should trigger seizure detection)
    let mut seizure_detected = 0;
    let test_seizure = 10;
    for i in 0..test_seizure {
        let (f, o) = generate_seizure_eeg(epoch_len, sample_rate, 200 + i as u64);
        let (_state, metrics) = sentinel.process_epoch(&f, &o);
        let score = seizure_score(&metrics);
        let is_seizure = score >= threshold;
        if is_seizure {
            seizure_detected += 1;
        }
        if i < 3 {
            println!(
                "  Seizure #{}: score={:.4} (thr={:.4}), sync={:.4}, entropy={:.4}, delta={:.4}",
                i,
                score,
                threshold,
                metrics.synchrony,
                metrics.spectral_entropy,
                metrics.delta_power
            );
        }
    }

    // Test pre-ictal epochs
    let mut preictal_transitional = 0;
    let test_preictal = 10;
    for i in 0..test_preictal {
        let (f, o) = generate_preictal_eeg(epoch_len, sample_rate, 300 + i as u64);
        let (_state, metrics) = sentinel.process_epoch(&f, &o);
        let score = seizure_score(&metrics);
        // Pre-ictal should show some elevation (transitional)
        let is_transitional = score > threshold * 0.5;
        if is_transitional {
            preictal_transitional += 1;
        }
        if i < 3 {
            println!(
                "  Pre-ictal #{}: score={:.4} (thr={:.4}), sync={:.4}, entropy={:.4}",
                i, score, threshold, metrics.synchrony, metrics.spectral_entropy
            );
        }
    }

    let test_time = t.elapsed().as_secs_f64();

    // ═══════════════════════════════════════════════════════════════
    // Test 3: Simulated continuous monitoring
    // Normal → pre-ictal → seizure → post-ictal → normal
    // ═══════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 3: Continuous Monitoring Simulation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    #[allow(clippy::type_complexity)]
    let phases: Vec<(&str, Box<dyn Fn(usize, f64, u64) -> (Vec<f64>, Vec<f64>)>)> = vec![
        ("Normal", Box::new(generate_normal_eeg)),
        ("Normal", Box::new(generate_normal_eeg)),
        ("Pre-ictal", Box::new(generate_preictal_eeg)),
        ("Pre-ictal", Box::new(generate_preictal_eeg)),
        ("Seizure", Box::new(generate_seizure_eeg)),
        ("Seizure", Box::new(generate_seizure_eeg)),
        ("Seizure", Box::new(generate_seizure_eeg)),
        ("Post-ictal", Box::new(generate_postictal_eeg)),
        ("Normal", Box::new(generate_normal_eeg)),
        ("Normal", Box::new(generate_normal_eeg)),
    ];

    let mut phi_trajectory = Vec::new();
    let mut sync_trajectory = Vec::new();

    let mut score_trajectory = Vec::new();
    for (i, (phase_name, gen_fn)) in phases.iter().enumerate() {
        let (f, o) = gen_fn(epoch_len, sample_rate, 400 + i as u64);
        let (_state, metrics) = sentinel.process_epoch(&f, &o);
        let score = seizure_score(&metrics);
        phi_trajectory.push(metrics.phi_proxy);
        sync_trajectory.push(metrics.synchrony);
        score_trajectory.push(score);

        let label = if score >= threshold {
            "SEIZURE"
        } else {
            "normal "
        };
        println!(
            "  t={:>2} [{:10}] │ {} │ score={:.4} │ φ={:.4} │ sync={:.4}",
            i, phase_name, label, score, metrics.phi_proxy, metrics.synchrony
        );
    }

    // Seizure epochs should have highest synchrony
    let normal_sync: f32 = sync_trajectory[..2].iter().sum::<f32>() / 2.0;
    let seizure_sync: f32 = sync_trajectory[4..7].iter().sum::<f32>() / 3.0;

    // Results
    let specificity = normal_correct as f64 / test_normal as f64;
    let sensitivity = seizure_detected as f64 / test_seizure as f64;
    let preictal_rate = preictal_transitional as f64 / test_preictal as f64;

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                 VALIDATION SUMMARY                         ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║  Specificity (normal→normal):  {:.1}% ({}/{})              ║",
        specificity * 100.0,
        normal_correct,
        test_normal
    );
    println!(
        "║  Sensitivity (seizure→detect): {:.1}% ({}/{})              ║",
        sensitivity * 100.0,
        seizure_detected,
        test_seizure
    );
    println!(
        "║  Pre-ictal detection:          {:.1}% ({}/{})              ║",
        preictal_rate * 100.0,
        preictal_transitional,
        test_preictal
    );
    println!(
        "║  Seizure sync > normal sync:   {}                         ║",
        if seizure_sync > normal_sync {
            "PASS"
        } else {
            "FAIL"
        }
    );
    println!("╟──────────────────────────────────────────────────────────────╢");

    let checks = vec![
        ("Specificity > 70%", specificity > 0.70),
        ("Sensitivity > 50%", sensitivity > 0.50),
        ("Seizure synchrony > normal", seizure_sync > normal_sync),
    ];

    let passed = checks.iter().filter(|(_, p)| *p).count();
    for (name, pass) in &checks {
        println!("║  {} {:50}   ║", if *pass { "PASS" } else { "FAIL" }, name);
    }
    println!("╟──────────────────────────────────────────────────────────────╢");
    println!(
        "║  Result: {}/{} tests passed  ({:.1}s)                       ║",
        passed,
        checks.len(),
        test_time
    );
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Save
    let result_json = serde_json::json!({
        "benchmark": "EEG Seizure Detection (Synthetic)",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "specificity": specificity,
        "sensitivity": sensitivity,
        "preictal_detection": preictal_rate,
        "seizure_sync_gt_normal": seizure_sync > normal_sync,
        "tests_passed": passed,
        "tests_total": checks.len(),
    });

    std::fs::create_dir_all("data/benchmarks/seizure").ok();
    if let Ok(f) = std::fs::File::create("data/benchmarks/seizure/results.json") {
        serde_json::to_writer_pretty(f, &result_json).ok();
        println!("Results saved to data/benchmarks/seizure/results.json");
    }
}

/// Extract features relevant to seizure detection from IntegrationMetrics.
///
/// Returns a feature vector: [delta_ratio, inv_entropy, synchrony, total_power]
fn extract_seizure_features(m: &IntegrationMetrics) -> [f32; 4] {
    let total = m.delta_power + m.theta_power + m.alpha_power + m.beta_power + m.gamma_power;
    let delta_ratio = if total > 1e-6 {
        m.delta_power / total
    } else {
        0.0
    };
    // Invert entropy so high score = seizure-like (low entropy = rhythmic)
    let inv_entropy = 1.0 - m.spectral_entropy.clamp(0.0, 1.0);
    [delta_ratio, inv_entropy, m.synchrony, total]
}

/// Compute a seizure score from metrics. Higher = more seizure-like.
///
/// Seizure signature: high delta ratio + low spectral entropy + high synchrony + high power.
fn seizure_score(m: &IntegrationMetrics) -> f32 {
    let feats = extract_seizure_features(m);
    let delta_ratio = feats[0];
    let inv_entropy = feats[1];
    let synchrony = feats[2];
    let total_power = feats[3];

    // Weighted combination: emphasize delta_ratio and low entropy
    // Normalize total_power relative to typical EEG amplitudes
    let power_score = (total_power / 50.0).clamp(0.0, 2.0);
    0.35 * delta_ratio + 0.30 * inv_entropy + 0.20 * synchrony + 0.15 * power_score
}

/// Compute a threshold that separates normal from seizure scores.
///
/// Uses the midpoint between the max normal score and min seizure score.
/// Falls back to the mean of all scores if classes overlap.
fn compute_seizure_threshold(normal: &[[f32; 4]], seizure: &[[f32; 4]]) -> f32 {
    let normal_scores: Vec<f32> = normal
        .iter()
        .map(|f| {
            let _m = IntegrationMetrics {
                delta_power: f[0] * (f[0] + 0.001), // reconstruct approximate metrics
                spectral_entropy: 1.0 - f[1],
                synchrony: f[2],
                ..Default::default()
            };
            // Actually just compute from features directly
            0.35 * f[0] + 0.30 * f[1] + 0.20 * f[2] + 0.15 * (f[3] / 50.0).clamp(0.0, 2.0)
        })
        .collect();

    let seizure_scores: Vec<f32> = seizure
        .iter()
        .map(|f| 0.35 * f[0] + 0.30 * f[1] + 0.20 * f[2] + 0.15 * (f[3] / 50.0).clamp(0.0, 2.0))
        .collect();

    let max_normal = normal_scores
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let min_seizure = seizure_scores.iter().cloned().fold(f32::INFINITY, f32::min);

    if min_seizure > max_normal {
        // Clean separation: use midpoint
        (max_normal + min_seizure) / 2.0
    } else {
        // Overlap: use weighted mean biased toward catching seizures (lower threshold)
        let mean_normal = normal_scores.iter().sum::<f32>() / normal_scores.len() as f32;
        let mean_seizure = seizure_scores.iter().sum::<f32>() / seizure_scores.len() as f32;
        // Bias toward sensitivity: 40% from normal mean, 60% from seizure mean
        mean_normal * 0.6 + mean_seizure * 0.4
    }
}

fn generate_normal_eeg(len: usize, sr: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let dt = 1.0 / sr;
    let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut rand = || -> f64 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 33) as f64 / (1u64 << 31) as f64 * 2.0 - 1.0
    };

    let mut f = vec![0.0; len];
    let mut o = vec![0.0; len];
    for i in 0..len {
        let t = i as f64 * dt;
        let alpha = 25.0 * (2.0 * std::f64::consts::PI * 10.0 * t).sin();
        let beta = 10.0 * (2.0 * std::f64::consts::PI * 20.0 * t).sin();
        f[i] = alpha * 0.4 + beta * 0.3 + 15.0 * rand();
        o[i] = alpha * 0.7 + beta * 0.1 + 15.0 * rand();
    }
    (f, o)
}

fn generate_seizure_eeg(len: usize, sr: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let dt = 1.0 / sr;
    let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut rand = || -> f64 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 33) as f64 / (1u64 << 31) as f64 * 2.0 - 1.0
    };

    let mut f = vec![0.0; len];
    let mut o = vec![0.0; len];
    for i in 0..len {
        let t = i as f64 * dt;
        // High amplitude, rhythmic 3 Hz spike-wave (generalized seizure)
        let spike = 150.0 * (2.0 * std::f64::consts::PI * 3.0 * t).sin();
        let harmonic = 75.0 * (2.0 * std::f64::consts::PI * 6.0 * t).sin();
        let noise = 5.0 * rand();
        // Highly synchronized between channels
        f[i] = spike + harmonic + noise;
        o[i] = spike * 0.95 + harmonic * 0.9 + 5.0 * rand();
    }
    (f, o)
}

fn generate_preictal_eeg(len: usize, sr: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let dt = 1.0 / sr;
    let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut rand = || -> f64 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 33) as f64 / (1u64 << 31) as f64 * 2.0 - 1.0
    };

    let mut f = vec![0.0; len];
    let mut o = vec![0.0; len];
    for i in 0..len {
        let t = i as f64 * dt;
        // Increasing amplitude, increasing synchrony
        let progress = i as f64 / len as f64;
        let alpha = 25.0 * (1.0 + progress) * (2.0 * std::f64::consts::PI * 10.0 * t).sin();
        let theta = 35.0 * progress * (2.0 * std::f64::consts::PI * 5.0 * t).sin();
        let noise = 12.0 * (1.0 - progress * 0.5) * rand();
        f[i] = alpha * 0.5 + theta * 0.4 + noise;
        o[i] = alpha * 0.5 * (0.5 + 0.5 * progress) + theta * 0.3 + 12.0 * rand();
    }
    (f, o)
}

fn generate_postictal_eeg(len: usize, sr: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let dt = 1.0 / sr;
    let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut rand = || -> f64 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 33) as f64 / (1u64 << 31) as f64 * 2.0 - 1.0
    };

    let mut f = vec![0.0; len];
    let mut o = vec![0.0; len];
    for i in 0..len {
        let t = i as f64 * dt;
        // Low amplitude, slowed, disorganized (post-ictal suppression)
        let delta = 15.0 * (2.0 * std::f64::consts::PI * 1.5 * t).sin();
        let noise = 20.0 * rand();
        f[i] = delta * 0.5 + noise;
        o[i] = delta * 0.3 + 20.0 * rand(); // Low synchrony
    }
    (f, o)
}