// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Sleep Stage Classification Benchmark
//!
//! Validates Symthaea's Sleep Sentinel (dual-channel LTC) against the Sleep-EDF
//! dataset for automated sleep staging from EEG signals.
//!
//! ## Method
//! Uses Symthaea's built-in SleepSentinel which processes frontal (Fpz-Cz) and
//! occipital (Pz-Oz) EEG channels through dual LTC networks with adaptive
//! thresholds to classify 5 sleep stages (Wake, N1, N2, N3, REM).
//!
//! ## Dataset
//! Sleep-EDF Expanded (Kemp et al. 2000, Goldberger et al. 2000)
//! - 197 whole-night polysomnographic recordings
//! - 2-channel EEG at 100 Hz
//! - 30-second epoch annotations per AASM standard
//!
//! ## Expected Results
//! - LTC-based: 70-80% overall accuracy (5-class)
//! - Wake detection: >85%
//! - Deep sleep (N3): >80%
//! - N1 (hardest): 30-50%
//!
//! ## Run
//! ```bash
//! cargo run --example benchmark_sleepstage --release
//! ```

use std::path::Path;
use std::time::Instant;

use symthaea::dynamics::hmm::HiddenMarkovModel;
use symthaea::dynamics::spectral_analysis::{SpectralAnalyzer, SpectralConfig, WindowType};
use symthaea::dynamics::wavelet::{DwtConfig, ExtensionMode, WaveletAnalyzer, WaveletFamily};
use symthaea::perception::physio::{
    ConsciousnessState, EdfFile, IntegrationMetrics, SleepSentinel, SleepSentinelConfig, SleepStage,
};

const DATA_DIR: &str = "data/benchmarks/sleep-edf";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Sleep Stage Classification Benchmark                 ║");
    println!("║       Sleep-EDF + Symthaea Sleep Sentinel                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let data_path = Path::new(DATA_DIR);
    if !data_path.exists() {
        eprintln!("Sleep-EDF data not found at {}", DATA_DIR);
        eprintln!("Download with: python scripts/download_sleep_edf.py");
        eprintln!("\nRunning with synthetic data instead...\n");
        run_synthetic_benchmark();
        return;
    }

    // Find all PSG files
    let mut psg_files: Vec<_> = std::fs::read_dir(data_path)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .to_str()
                .map(|s| s.ends_with("-PSG.edf"))
                .unwrap_or(false)
        })
        .map(|e| e.path())
        .collect();
    psg_files.sort();

    if psg_files.is_empty() {
        eprintln!("No PSG .edf files found in {}", DATA_DIR);
        eprintln!("Running with synthetic data instead...\n");
        run_synthetic_benchmark();
        return;
    }

    println!("Found {} PSG recordings\n", psg_files.len());

    // Use first 10 for training, rest for testing
    // Use 2/3 for training, 1/3 for testing (minimum 1 each)
    let n_train = (psg_files.len() * 2 / 3).max(1).min(psg_files.len() - 1);
    let n_test = psg_files.len() - n_train;

    let config = SleepSentinelConfig {
        local_neurons: 64,
        global_neurons: 128,
        dt_ms: 10.0,
        integration_window: 300,
        tau_base: 100.0,
        enable_adaptive_thresholds: true,
        steps_per_epoch: 100,        // Reduced for faster benchmarking
        use_spectral_analysis: true, // Welch PSD with recalibrated thresholds for proper spectral ratios
        ..SleepSentinelConfig::default()
    };

    let mut sentinel = SleepSentinel::new(config);

    // Spectral analyzer for feature extraction (same config as emission model)
    let train_spectral = SpectralAnalyzer::new(SpectralConfig {
        window_size: 512,
        overlap: 0.5,
        window_type: WindowType::Hann,
        sample_rate: 100.0, // Sleep-EDF is 100 Hz
    });
    let mut training_features: Vec<(usize, [f64; N_FEAT])> = Vec::new();

    // Training phase
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Training Phase ({} recordings)", n_train);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let train_start = Instant::now();
    let mut total_train_epochs = 0;

    for (i, psg_path) in psg_files[..n_train].iter().enumerate() {
        // Sleep-EDF naming: SC4001E0-PSG.edf -> SC4001E*-Hypnogram.edf
        let psg_str = psg_path.to_str().unwrap();
        let hyp_path = find_hypnogram(psg_str)
            .unwrap_or_else(|| psg_str.replace("-PSG.edf", "-Hypnogram.edf"));

        let mut edf = match EdfFile::load(psg_path) {
            Ok(e) => e,
            Err(err) => {
                println!("  Skip {}: {}", psg_path.display(), err);
                continue;
            }
        };

        if let Err(err) = edf.load_hypnogram(&hyp_path) {
            println!("  Skip {}: {}", hyp_path, err);
            continue;
        }

        let n_epochs = edf.num_epochs();
        println!(
            "  [{}/{}] {} - {} epochs",
            i + 1,
            n_train,
            psg_path.file_name().unwrap().to_str().unwrap(),
            n_epochs
        );

        // Collect raw features for this recording (will z-normalize per recording)
        let mut recording_features: Vec<(usize, [f64; N_FEAT])> = Vec::new();

        for epoch_idx in 0..n_epochs {
            if let Some((frontal, occipital, stage)) = edf.get_labeled_epoch(epoch_idx) {
                let stage_idx = stage_to_idx(&stage);
                sentinel.train_epoch(&frontal, &occipital, stage);
                total_train_epochs += 1;
                if stage_idx < 5 {
                    let features = extract_spectral_features(&frontal, &train_spectral);
                    recording_features.push((stage_idx, features));
                }
            }
        }

        // Z-normalize features within this recording to remove subject-specific
        // baseline and amplify within-recording stage differences
        if !recording_features.is_empty() {
            let nr = recording_features.len() as f64;
            let mut feat_mean = [0.0f64; N_FEAT];
            let mut feat_sq = [0.0f64; N_FEAT];
            for (_, feats) in &recording_features {
                for f in 0..N_FEAT {
                    feat_mean[f] += feats[f];
                    feat_sq[f] += feats[f] * feats[f];
                }
            }
            for f in 0..N_FEAT {
                feat_mean[f] /= nr;
                feat_sq[f] = ((feat_sq[f] / nr - feat_mean[f] * feat_mean[f]).max(1e-10)).sqrt();
            }
            for (stage_idx, feats) in recording_features {
                let mut z = [0.0f64; N_FEAT];
                for f in 0..N_FEAT {
                    z[f] = (feats[f] - feat_mean[f]) / feat_sq[f];
                }
                training_features.push((stage_idx, z));
            }
        }
    }

    let train_time = train_start.elapsed().as_secs_f64();
    println!(
        "\n  Training: {} epochs in {:.1}s ({:.0} epochs/s)",
        total_train_epochs,
        train_time,
        total_train_epochs as f64 / train_time
    );

    // Build learned emission model from training spectral features
    let stage_stats = StageFeatureStats::from_training_data(&training_features);
    let feat_stage_names = ["Wake", "N1", "N2", "N3", "REM"];
    println!("\n  Learned feature distributions (mean):");
    println!(
        "  {:>5} {:>4} {:>6} {:>6} {:>6} {:>6} {:>8} {:>7} {:>6}",
        "", "n", "delta", "theta", "alpha", "beta", "logPow", "logRMS", "ZCR"
    );
    for (s, feat_stage_name) in feat_stage_names.iter().enumerate() {
        println!(
            "  {:>5} {:>4} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>8.2} {:>7.2} {:>6.4}",
            feat_stage_name,
            stage_stats.counts[s],
            stage_stats.means[s][0],
            stage_stats.means[s][1],
            stage_stats.means[s][2],
            stage_stats.means[s][3],
            stage_stats.means[s][4],
            stage_stats.means[s][5],
            stage_stats.means[s][6]
        );
    }

    // Testing phase
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Testing Phase ({} recordings)", n_test);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let test_start = Instant::now();
    let mut confusion_raw = vec![vec![0usize; 5]; 5]; // Raw (no HMM)
    let mut confusion_hmm = vec![vec![0usize; 5]; 5]; // HMM-smoothed
    let mut total_test_epochs = 0;

    // Build HMM with physiological sleep transition constraints
    let hmm = build_sleep_hmm();

    for (i, psg_path) in psg_files[n_train..n_train + n_test].iter().enumerate() {
        // Sleep-EDF naming: SC4001E0-PSG.edf -> SC4001E*-Hypnogram.edf
        let psg_str = psg_path.to_str().unwrap();
        let hyp_path = find_hypnogram(psg_str)
            .unwrap_or_else(|| psg_str.replace("-PSG.edf", "-Hypnogram.edf"));

        let mut edf = match EdfFile::load(psg_path) {
            Ok(e) => e,
            Err(err) => {
                println!("  Skip {}: {}", psg_path.display(), err);
                continue;
            }
        };

        if let Err(err) = edf.load_hypnogram(&hyp_path) {
            println!("  Skip {}: {}", hyp_path, err);
            continue;
        }

        let n_epochs = edf.num_epochs();

        // Collect all predictions + metrics for this recording
        let mut actual_indices = Vec::new();
        let mut raw_predictions = Vec::new();
        let mut epoch_metrics = Vec::new();
        let mut epoch_signals: Vec<Vec<f64>> = Vec::new();

        for epoch_idx in 0..n_epochs {
            if let Some((frontal, occipital, stage)) = edf.get_labeled_epoch(epoch_idx) {
                let actual_idx = stage_to_idx(&stage);
                if actual_idx >= 5 {
                    continue; // skip Movement/Unknown
                }

                let (predicted_state, metrics) = sentinel.process_epoch(&frontal, &occipital);
                let predicted_idx = consciousness_to_idx(&predicted_state);

                actual_indices.push(actual_idx);
                raw_predictions.push(predicted_idx);
                epoch_metrics.push(metrics);
                epoch_signals.push(frontal);
            }
        }

        // Ensemble: combine learned Gaussian model + rule-based emission probs
        // Gaussian model is good at Wake detection (z-normalized features)
        // Rule-based model is good at N2/N3 separation (sigmoid thresholds)
        let emission_gaussian = compute_emission_probs_learned(&epoch_signals, &stage_stats, 100.0);
        let emission_rules = compute_emission_probs_enhanced(
            &epoch_metrics,
            &epoch_signals,
            100.0,
            Some(&stage_stats),
        );
        let emission_sequence: Vec<Vec<f64>> = emission_gaussian
            .iter()
            .zip(emission_rules.iter())
            .map(|(g, r)| {
                // Geometric mean gives balanced combination
                let mut combined: Vec<f64> = g
                    .iter()
                    .zip(r.iter())
                    .map(|(&gp, &rp)| (gp * rp).sqrt())
                    .collect();
                let sum: f64 = combined.iter().sum();
                if sum > 0.0 {
                    for p in combined.iter_mut() {
                        *p = (*p / sum).max(0.01);
                    }
                    let sum2: f64 = combined.iter().sum();
                    for p in combined.iter_mut() {
                        *p /= sum2;
                    }
                }
                combined
            })
            .collect();

        // Run Viterbi smoothing on the full recording
        let smoothed = hmm.viterbi(&emission_sequence);

        // Score both raw and HMM-smoothed
        let mut correct_raw = 0;
        let mut correct_hmm = 0;

        for j in 0..actual_indices.len() {
            let actual = actual_indices[j];
            let raw_pred = raw_predictions[j];
            let hmm_pred = smoothed[j];

            confusion_raw[actual][raw_pred] += 1;
            confusion_hmm[actual][hmm_pred] += 1;

            if actual == raw_pred {
                correct_raw += 1;
            }
            if actual == hmm_pred {
                correct_hmm += 1;
            }
        }

        total_test_epochs += actual_indices.len();

        let n_valid = actual_indices.len();
        let acc_raw = if n_valid > 0 {
            correct_raw as f64 / n_valid as f64 * 100.0
        } else {
            0.0
        };
        let acc_hmm = if n_valid > 0 {
            correct_hmm as f64 / n_valid as f64 * 100.0
        } else {
            0.0
        };
        println!(
            "  [{}/{}] {} - raw {:.1}% / HMM {:.1}% ({} epochs)",
            i + 1,
            n_test,
            psg_path.file_name().unwrap().to_str().unwrap(),
            acc_raw,
            acc_hmm,
            n_valid
        );
    }

    let test_time = test_start.elapsed().as_secs_f64();

    println!("\n--- Raw (no HMM) ---");
    print_results(&confusion_raw, total_test_epochs, test_time);
    println!("--- HMM Viterbi Smoothed ---");
    print_results(&confusion_hmm, total_test_epochs, test_time);
}

fn stage_to_idx(stage: &SleepStage) -> usize {
    match stage {
        SleepStage::Wake => 0,
        SleepStage::N1 => 1,
        SleepStage::N2 => 2,
        SleepStage::N3 => 3,
        SleepStage::REM => 4,
        _ => 5, // Movement/Unknown
    }
}

fn consciousness_to_idx(state: &ConsciousnessState) -> usize {
    match state {
        ConsciousnessState::Awake => 0,
        ConsciousnessState::Transitional => 1, // maps to N1
        ConsciousnessState::LightSleep => 2,   // maps to N2
        ConsciousnessState::DeepSleep => 3,    // maps to N3
        ConsciousnessState::REM => 4,
    }
}

fn print_results(confusion: &[Vec<usize>], total: usize, test_time: f64) {
    let stage_names = ["Wake", "N1", "N2", "N3", "REM"];

    // Per-class metrics
    let mut total_correct = 0;
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    RESULTS                                 ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    for i in 0..5 {
        let class_total: usize = confusion[i].iter().sum();
        let correct = confusion[i][i];
        total_correct += correct;
        let acc = if class_total > 0 {
            correct as f64 / class_total as f64 * 100.0
        } else {
            0.0
        };
        println!(
            "║  {:5} │ {:>6.1}% ({:>5}/{:>5})                             ║",
            stage_names[i], acc, correct, class_total
        );
    }

    let overall = if total > 0 {
        total_correct as f64 / total as f64 * 100.0
    } else {
        0.0
    };

    println!("╟──────────────────────────────────────────────────────────────╢");
    println!(
        "║  Overall: {:.1}% ({}/{})  Time: {:.1}s                  ║",
        overall, total_correct, total, test_time
    );
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("VALIDATION");
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "  Overall > 50% (above chance): {}",
        if overall > 50.0 { "PASS" } else { "FAIL" }
    );
    println!(
        "  Overall > 65%:                {}",
        if overall > 65.0 { "PASS" } else { "FAIL" }
    );
    println!(
        "  Overall > 75%:                {}",
        if overall > 75.0 { "PASS" } else { "FAIL" }
    );
}

/// Synthetic benchmark when real EEG data is not available.
/// Generates EEG-like signals for each sleep stage and validates
/// the SleepSentinel's discrimination ability.
fn run_synthetic_benchmark() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Synthetic EEG Sleep Stage Benchmark");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let config = SleepSentinelConfig {
        local_neurons: 64,
        global_neurons: 128,
        dt_ms: 10.0,
        integration_window: 300,
        tau_base: 100.0,
        enable_adaptive_thresholds: true,
        steps_per_epoch: 500,        // Reduced from 3000 for faster benchmark
        use_spectral_analysis: true, // Welch PSD with recalibrated thresholds for proper spectral ratios
        ..SleepSentinelConfig::default()
    };

    let mut sentinel = SleepSentinel::new(config);
    let sample_rate = 100.0; // Hz
    let epoch_len = (30.0 * sample_rate) as usize; // 30-second epochs

    // Generate synthetic EEG for each stage
    let stages = vec![
        (SleepStage::Wake, "Wake"),
        (SleepStage::N1, "N1"),
        (SleepStage::N2, "N2"),
        (SleepStage::N3, "N3"),
        (SleepStage::REM, "REM"),
    ];

    // Training: generate 10 epochs per stage (reduced for benchmark speed)
    println!("\nTraining with synthetic EEG...");
    let train_start = Instant::now();
    let train_epochs_per_stage = 10;

    for (stage, name) in &stages {
        for epoch_i in 0..train_epochs_per_stage {
            let (frontal, occipital) =
                generate_synthetic_eeg(stage, epoch_len, sample_rate, epoch_i as u64);
            sentinel.train_epoch(&frontal, &occipital, *stage);
        }
        println!("  Trained {} x {} epochs", name, train_epochs_per_stage);
    }
    println!(
        "  Training time: {:.1}s\n",
        train_start.elapsed().as_secs_f64()
    );

    // Testing: 10 epochs per stage
    println!("Testing...");
    let test_start = Instant::now();
    let test_epochs_per_stage = 10;
    let mut confusion_raw = vec![vec![0usize; 5]; 5];
    let mut confusion_hmm = vec![vec![0usize; 5]; 5];

    // Build HMM for Viterbi smoothing
    let hmm = build_sleep_hmm();

    // Collect all predictions and metrics first (simulating a full night recording)
    let mut all_actual = Vec::new();
    let mut all_raw_pred = Vec::new();
    let mut all_metrics = Vec::new();
    let mut all_signals: Vec<Vec<f64>> = Vec::new();

    for (stage_idx, (stage, _name)) in stages.iter().enumerate() {
        for epoch_i in 0..test_epochs_per_stage {
            let seed = 10000 + epoch_i as u64;
            let (frontal, occipital) = generate_synthetic_eeg(stage, epoch_len, sample_rate, seed);
            let (predicted, metrics) = sentinel.process_epoch(&frontal, &occipital);
            let pred_idx = consciousness_to_idx(&predicted);

            all_actual.push(stage_idx);
            all_raw_pred.push(pred_idx);
            all_metrics.push(metrics);
            all_signals.push(frontal);
        }
    }

    // Compute emission probabilities with wavelet/PAC enhancement
    let emission_sequence =
        compute_emission_probs_enhanced(&all_metrics, &all_signals, sample_rate, None);
    let smoothed = hmm.viterbi(&emission_sequence);

    // Score both raw and HMM-smoothed
    for j in 0..all_actual.len() {
        confusion_raw[all_actual[j]][all_raw_pred[j]] += 1;
        confusion_hmm[all_actual[j]][smoothed[j]] += 1;
    }

    let total = test_epochs_per_stage * 5;
    let test_time = test_start.elapsed().as_secs_f64();

    println!("\n--- Raw (no HMM) ---");
    print_results(&confusion_raw, total, test_time);
    println!("--- HMM Viterbi Smoothed ---");
    print_results(&confusion_hmm, total, test_time);

    // Save results (use HMM-smoothed accuracy)
    let mut total_correct = 0;
    for (i, row) in confusion_hmm.iter().enumerate().take(5) {
        total_correct += row[i];
    }
    let overall = total_correct as f64 / total as f64;

    let result_json = serde_json::json!({
        "benchmark": "Sleep Stage Classification (Synthetic)",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "overall_accuracy": overall,
        "total_epochs": total,
        "mode": "synthetic",
        "note": "Synthetic data - download Sleep-EDF for real validation",
    });

    let result_path = "data/benchmarks/sleep-edf/results.json";
    std::fs::create_dir_all("data/benchmarks/sleep-edf").ok();
    if let Ok(f) = std::fs::File::create(result_path) {
        serde_json::to_writer_pretty(f, &result_json).ok();
        println!("Results saved to {}", result_path);
    }
}

/// Generate synthetic EEG signals characteristic of each sleep stage.
///
/// Signal characteristics:
/// - Wake: Alpha (8-12 Hz) dominant, low amplitude, low synchrony
/// - N1: Theta (4-7 Hz), reduced alpha, transitional
/// - N2: Sleep spindles (12-14 Hz bursts), K-complexes
/// - N3: Delta (0.5-2 Hz) dominant, high amplitude, high synchrony
/// - REM: Mixed frequency, low amplitude, low synchrony (paradoxical)
fn generate_synthetic_eeg(
    stage: &SleepStage,
    epoch_len: usize,
    sample_rate: f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    let mut frontal = vec![0.0f64; epoch_len];
    let mut occipital = vec![0.0f64; epoch_len];

    // Simple LCG PRNG for reproducibility
    let mut rng_state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut rand_f64 = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng_state >> 33) as f64 / (1u64 << 31) as f64 - 1.0
    };

    let dt = 1.0 / sample_rate;

    match stage {
        SleepStage::Wake => {
            // Alpha waves (10 Hz), beta (20 Hz), low amplitude, moderate synchrony
            for i in 0..epoch_len {
                let t = i as f64 * dt;
                let alpha = 30.0 * (2.0 * std::f64::consts::PI * 10.0 * t).sin();
                let beta = 15.0 * (2.0 * std::f64::consts::PI * 20.0 * t).sin();
                let noise_f = 10.0 * rand_f64();
                let noise_o = 10.0 * rand_f64();
                frontal[i] = alpha * 0.3 + beta * 0.5 + noise_f;
                occipital[i] = alpha * 0.8 + beta * 0.2 + noise_o; // Alpha dominant in occipital
            }
        }
        SleepStage::N1 => {
            // Theta (5 Hz), reduced alpha, vertex sharp waves
            for i in 0..epoch_len {
                let t = i as f64 * dt;
                let theta = 40.0 * (2.0 * std::f64::consts::PI * 5.0 * t).sin();
                let alpha = 10.0 * (2.0 * std::f64::consts::PI * 10.0 * t).sin();
                let noise_f = 15.0 * rand_f64();
                let noise_o = 15.0 * rand_f64();
                frontal[i] = theta * 0.6 + alpha * 0.2 + noise_f;
                occipital[i] = theta * 0.4 + alpha * 0.3 + noise_o;
            }
        }
        SleepStage::N2 => {
            // Sleep spindles (12 Hz bursts) + K-complexes
            for i in 0..epoch_len {
                let t = i as f64 * dt;
                let theta = 30.0 * (2.0 * std::f64::consts::PI * 5.0 * t).sin();
                // Spindle bursts every ~5 seconds
                let spindle_env = ((2.0 * std::f64::consts::PI * 0.2 * t).sin()).max(0.0);
                let spindle = 25.0 * spindle_env * (2.0 * std::f64::consts::PI * 13.0 * t).sin();
                let noise_f = 12.0 * rand_f64();
                let noise_o = 12.0 * rand_f64();
                frontal[i] = theta * 0.4 + spindle * 0.8 + noise_f;
                occipital[i] = theta * 0.3 + spindle * 0.5 + noise_o;
            }
        }
        SleepStage::N3 => {
            // Delta waves (1 Hz), high amplitude, high synchrony
            for i in 0..epoch_len {
                let t = i as f64 * dt;
                let delta = 100.0 * (2.0 * std::f64::consts::PI * 1.0 * t).sin();
                let delta2 = 50.0 * (2.0 * std::f64::consts::PI * 0.5 * t).sin();
                let noise_f = 8.0 * rand_f64();
                let noise_o = 8.0 * rand_f64();
                // High synchrony between channels
                frontal[i] = delta + delta2 + noise_f;
                occipital[i] = delta * 0.9 + delta2 * 0.85 + noise_o;
            }
        }
        SleepStage::REM => {
            // Mixed frequency, low amplitude, desynchronized (like wake but without alpha)
            for i in 0..epoch_len {
                let t = i as f64 * dt;
                let theta = 20.0 * (2.0 * std::f64::consts::PI * 6.0 * t).sin();
                let beta = 10.0 * (2.0 * std::f64::consts::PI * 18.0 * t).sin();
                let saw = 8.0 * (2.0 * std::f64::consts::PI * 3.0 * t).sin();
                let noise_f = 20.0 * rand_f64();
                let noise_o = 20.0 * rand_f64();
                // Desynchronized between channels
                frontal[i] = theta * 0.5 + beta * 0.3 + saw + noise_f;
                occipital[i] = theta * 0.3 + beta * 0.5 - saw * 0.5 + noise_o;
            }
        }
        _ => {
            // Unknown/Movement - random noise
            for i in 0..epoch_len {
                frontal[i] = 50.0 * rand_f64();
                occipital[i] = 50.0 * rand_f64();
            }
        }
    }

    (frontal, occipital)
}

/// Build a physiologically-constrained HMM for sleep staging.
///
/// States: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
/// Transition constraints encode AASM sleep physiology:
/// - Sleep progresses Wake→N1→N2→N3 (descent) and reverses (ascent)
/// - REM follows light sleep (N1/N2), not deep sleep (N3 rarely→REM)
/// - Adjacent stage transitions are most probable
/// - No jumps across >1 stage (e.g., Wake→N3 is near-zero)
fn build_sleep_hmm() -> HiddenMarkovModel {
    // Transition matrix: A[i][j] = P(state j | state i)
    // States: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
    // Encodes AASM sleep physiology:
    // - Sleep progresses Wake→N1→N2→N3 (descent) and reverses (ascent)
    // - REM follows light sleep (N1/N2), not deep sleep (N3 rarely→REM)
    // - Adjacent stage transitions are most probable
    // - No jumps across >1 stage (e.g., Wake→N3 is near-zero)
    #[rustfmt::skip]
    let transitions = vec![
        //  Wake    N1      N2      N3      REM
        vec![0.70,  0.20,   0.05,   0.00,   0.05],   // Wake → mostly stays, can enter N1
        vec![0.10,  0.50,   0.30,   0.02,   0.08],   // N1 → can go to Wake, N2, or REM
        vec![0.02,  0.10,   0.60,   0.20,   0.08],   // N2 → mostly stays, descend to N3 or ascend
        vec![0.00,  0.02,   0.20,   0.75,   0.03],   // N3 → mostly stays, ascend through N2
        vec![0.05,  0.15,   0.10,   0.00,   0.70],   // REM → can go to Wake/N1/N2, not N3
    ];

    // Initial state: most likely Wake or N2 (depends on recording start)
    let initial = vec![0.40, 0.15, 0.25, 0.10, 0.10];

    let state_names = vec![
        "Wake".into(),
        "N1".into(),
        "N2".into(),
        "N3".into(),
        "REM".into(),
    ];

    HiddenMarkovModel::with_params(initial, transitions, state_names)
}

/// Number of spectral features per epoch.
const N_FEAT: usize = 7;

/// Extract discriminative EEG features from a raw signal using Welch PSD.
///
/// Returns [delta_ratio, theta_ratio, alpha_ratio, beta_ratio, log_total_power, log_rms, spectral_slope]
/// - Band ratios are normalized within 0.5-30 Hz only (not full spectrum)
/// - log_total_power: absolute power level (differs greatly across stages)
/// - log_rms: signal amplitude (N3 >> Wake)
/// - spectral_slope: 1/f slope (more negative = more low-freq dominated = deeper sleep)
fn extract_spectral_features(signal: &[f64], spectral: &SpectralAnalyzer) -> [f64; N_FEAT] {
    if signal.len() < 256 {
        return [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0];
    }
    let spectrum = spectral.welch(signal);

    // Compute band powers (only 0.5-30 Hz for EEG)
    let mut delta = 0.0f64;
    let mut theta = 0.0f64;
    let mut alpha = 0.0f64;
    let mut beta = 0.0f64;
    // For spectral slope: collect log(f), log(PSD) pairs in 1-30 Hz
    let mut log_f_sum = 0.0f64;
    let mut log_p_sum = 0.0f64;
    let mut log_f_sq_sum = 0.0f64;
    let mut log_fp_sum = 0.0f64;
    let mut slope_n = 0usize;

    for (i, &p) in spectrum.psd.iter().enumerate() {
        let f = spectrum.frequencies[i];
        let pv = p.max(0.0);
        if (0.5..4.0).contains(&f) {
            delta += pv;
        } else if (4.0..8.0).contains(&f) {
            theta += pv;
        } else if (8.0..12.0).contains(&f) {
            alpha += pv;
        } else if (12.0..30.0).contains(&f) {
            beta += pv;
        }

        // Spectral slope: linear regression of log(PSD) vs log(f) in 1-30 Hz
        if (1.0..=30.0).contains(&f) && pv > 1e-20 {
            let lf = f.ln();
            let lp = pv.ln();
            log_f_sum += lf;
            log_p_sum += lp;
            log_f_sq_sum += lf * lf;
            log_fp_sum += lf * lp;
            slope_n += 1;
        }
    }

    // Normalize within EEG bands only (0.5-30 Hz), not full spectrum
    let total_eeg = (delta + theta + alpha + beta).max(1e-10);

    // Log total power — captures amplitude differences between stages
    let log_power = total_eeg.max(1e-20).ln();

    // RMS amplitude (N3: high amplitude delta; Wake: low amplitude fast)
    let rms = (signal.iter().map(|&x| x * x).sum::<f64>() / signal.len() as f64).sqrt();
    let log_rms = rms.max(1e-20).ln();

    // Spectral slope via linear regression: log(PSD) = slope * log(f) + intercept
    // More negative slope = steeper 1/f = deeper sleep (delta dominant)
    // Flatter/positive slope = more broadband = Wake/REM
    let spectral_slope = if slope_n >= 3 {
        let n = slope_n as f64;
        let denom = n * log_f_sq_sum - log_f_sum * log_f_sum;
        if denom.abs() > 1e-10 {
            (n * log_fp_sum - log_f_sum * log_p_sum) / denom
        } else {
            0.0
        }
    } else {
        0.0
    };

    [
        delta / total_eeg,
        theta / total_eeg,
        alpha / total_eeg,
        beta / total_eeg,
        log_power,
        log_rms,
        spectral_slope,
    ]
}

/// Per-stage feature statistics learned from training data.
/// Uses diagonal Gaussian model for emission probability computation.
struct StageFeatureStats {
    means: [[f64; N_FEAT]; 5], // [stage][feature]
    vars: [[f64; N_FEAT]; 5],  // [stage][feature]
    counts: [usize; 5],
}

impl StageFeatureStats {
    fn from_training_data(features: &[(usize, [f64; N_FEAT])]) -> Self {
        let mut sums = [[0.0f64; N_FEAT]; 5];
        let mut sq_sums = [[0.0f64; N_FEAT]; 5];
        let mut counts = [0usize; 5];

        for &(stage, ref feats) in features {
            if stage >= 5 {
                continue;
            }
            counts[stage] += 1;
            for f in 0..N_FEAT {
                sums[stage][f] += feats[f];
                sq_sums[stage][f] += feats[f] * feats[f];
            }
        }

        let mut means = [[0.0; N_FEAT]; 5];
        let mut vars = [[0.0; N_FEAT]; 5];
        for s in 0..5 {
            let n = counts[s].max(1) as f64;
            for f in 0..N_FEAT {
                means[s][f] = sums[s][f] / n;
                vars[s][f] = (sq_sums[s][f] / n - means[s][f] * means[s][f]).max(1e-6);
            }
        }

        Self {
            means,
            vars,
            counts,
        }
    }

    /// Compute emission probability vector for all 5 stages given observed features.
    fn emission_probs(&self, features: &[f64; N_FEAT]) -> Vec<f64> {
        let mut log_likes = [0.0f64; 5];
        for (s, log_like) in log_likes.iter_mut().enumerate() {
            if self.counts[s] == 0 {
                *log_like = -100.0;
                continue;
            }
            for (f, feature) in features.iter().enumerate() {
                let diff = feature - self.means[s][f];
                *log_like += -0.5 * diff * diff / self.vars[s][f];
                *log_like += -0.5 * self.vars[s][f].ln();
            }
        }
        // Softmax with numerical stability
        let max_ll = log_likes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut probs: Vec<f64> = log_likes.iter().map(|&ll| (ll - max_ll).exp()).collect();
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for p in probs.iter_mut() {
                *p = (*p / sum).max(0.01);
            }
            let sum2: f64 = probs.iter().sum();
            for p in probs.iter_mut() {
                *p /= sum2;
            }
        } else {
            probs = vec![0.2; 5];
        }
        probs
    }
}

/// Compute emission probabilities using learned per-stage Gaussian distributions.
/// Applies per-recording z-normalization to match the training normalization.
fn compute_emission_probs_learned(
    signals: &[Vec<f64>],
    stats: &StageFeatureStats,
    sample_rate: f64,
) -> Vec<Vec<f64>> {
    let spectral = SpectralAnalyzer::new(SpectralConfig {
        window_size: 512,
        overlap: 0.5,
        window_type: WindowType::Hann,
        sample_rate,
    });

    // First pass: extract raw features for all epochs
    let raw_features: Vec<[f64; N_FEAT]> = signals
        .iter()
        .map(|signal| extract_spectral_features(signal, &spectral))
        .collect();

    // Compute per-recording mean and std for z-normalization
    let n = raw_features.len() as f64;
    let mut mean = [0.0f64; N_FEAT];
    let mut sq = [0.0f64; N_FEAT];
    for feats in &raw_features {
        for f in 0..N_FEAT {
            mean[f] += feats[f];
            sq[f] += feats[f] * feats[f];
        }
    }
    for f in 0..N_FEAT {
        mean[f] /= n;
        sq[f] = ((sq[f] / n - mean[f] * mean[f]).max(1e-10)).sqrt();
    }

    // Z-normalize and compute emission probabilities
    raw_features
        .iter()
        .map(|feats| {
            let mut z = [0.0f64; N_FEAT];
            for f in 0..N_FEAT {
                z[f] = (feats[f] - mean[f]) / sq[f];
            }
            stats.emission_probs(&z)
        })
        .collect()
}

/// Enhanced emission probabilities combining spectral, wavelet, and PAC features.
///
/// Adds to the base spectral ratios:
/// 1. Wavelet spindle detection → boosts N2 confidence
/// 2. Wavelet entropy → multi-scale complexity measure
/// 3. PAC delta-beta coupling → distinguishes N3 from lighter stages
/// 4. PAC theta-gamma coupling → distinguishes REM from other stages
/// 5. Adaptive thresholds from training statistics (not hardcoded)
fn compute_emission_probs_enhanced(
    metrics: &[IntegrationMetrics],
    signals: &[Vec<f64>],
    sample_rate: f64,
    stage_stats: Option<&StageFeatureStats>,
) -> Vec<Vec<f64>> {
    // Initialize wavelet analyzer for EEG
    let wavelet_config = DwtConfig {
        wavelet: WaveletFamily::Db4,
        max_level: None,
        extension: ExtensionMode::Symmetric,
    };
    let wavelet = WaveletAnalyzer::new(wavelet_config, sample_rate);

    // Compute band powers DIRECTLY from raw EEG via FFT (not from IntegrationMetrics)
    use symthaea::dynamics::spectral_analysis::{SpectralAnalyzer, SpectralConfig, WindowType};
    let spectral = SpectralAnalyzer::new(SpectralConfig {
        window_size: 512,
        overlap: 0.5,
        window_type: WindowType::Hann,
        sample_rate,
    });

    metrics
        .iter()
        .zip(signals.iter())
        .map(|(m, signal)| {
            // Compute band powers from raw EEG signal via Welch PSD.
            // IntegrationMetrics come from CfC network state and don't reflect
            // epoch-to-epoch frequency variation needed for classification.
            let (delta_ratio, theta_ratio, alpha_ratio, beta_ratio, _entropy) = if signal.len()
                >= 256
            {
                let spectrum = spectral.welch(signal);

                let mut delta = 0.0f64;
                let mut theta = 0.0f64;
                let mut alpha = 0.0f64;
                let mut beta = 0.0f64;
                let mut total = 0.0f64;
                for (i, &p) in spectrum.psd.iter().enumerate() {
                    let f = spectrum.frequencies[i];
                    let pv = p.max(0.0);
                    total += pv;
                    if (0.5..4.0).contains(&f) {
                        delta += pv;
                    } else if (4.0..8.0).contains(&f) {
                        theta += pv;
                    } else if (8.0..12.0).contains(&f) {
                        alpha += pv;
                    } else if (12.0..30.0).contains(&f) {
                        beta += pv;
                    }
                }
                let t = total.max(1e-10);
                // Spectral entropy from PSD
                let ent: f64 = spectrum
                    .psd
                    .iter()
                    .map(|&p| {
                        let n = (p.max(0.0) / t).max(1e-20);
                        -n * n.ln()
                    })
                    .sum::<f64>()
                    / (spectrum.psd.len() as f64).ln().max(1.0);

                (delta / t, theta / t, alpha / t, beta / t, ent)
            } else {
                // Fallback to IntegrationMetrics for short signals
                let total_power =
                    (m.delta_power + m.theta_power + m.alpha_power + m.beta_power + m.gamma_power)
                        .max(1e-10);
                (
                    m.delta_power as f64 / total_power as f64,
                    m.theta_power as f64 / total_power as f64,
                    m.alpha_power as f64 / total_power as f64,
                    m.beta_power as f64 / total_power as f64,
                    m.spectral_entropy as f64,
                )
            };

            // --- Wavelet features ---
            // Spindle count (12-14 Hz bursts characteristic of N2)
            let spindles = wavelet.detect_spindles(signal);
            let spindle_count = spindles.len() as f64;
            // Normalized: 0-3+ spindles per 30s epoch
            let spindle_score = (spindle_count / 3.0).min(1.0);

            // Wavelet entropy (multi-scale complexity)
            let w_entropy = wavelet.wavelet_entropy(signal);

            // --- Cross-frequency proxies (fast approximation of PAC) ---
            // Delta-beta co-activation: strong in deep sleep (delta dominant + low beta)
            // Using band power product as PAC proxy (avoids expensive FIR+Hilbert)
            let pac_score = (delta_ratio * (1.0 - beta_ratio) * 4.0).min(1.0);

            // Theta-gamma proxy: theta dominant with some high-freq activity = REM
            // True PAC requires FIR filtering (~30ms/epoch), proxy is O(1)
            let theta_gamma_score = (theta_ratio * beta_ratio * 20.0).min(1.0);

            // --- Adaptive thresholds from training statistics ---
            // Use training means as sigmoid centers instead of hardcoded values.
            // Fallback to hardcoded if no stats provided.
            let (
                wake_alpha_c,
                wake_delta_c,
                n1_theta_c,
                n1_delta_c,
                n2_delta_c,
                n3_delta_c,
                rem_theta_c,
                rem_delta_c,
            ) = if let Some(ss) = stage_stats {
                // means[stage][feature]: 0=delta, 1=theta, 2=alpha, 3=beta
                (
                    ss.means[0][2].max(0.05), // Wake alpha center
                    ss.means[0][0].min(0.45), // Wake delta center (upper bound)
                    ss.means[1][1].max(0.08), // N1 theta center
                    ss.means[1][0],           // N1 delta center
                    ss.means[2][0],           // N2 delta center
                    ss.means[3][0].max(0.50), // N3 delta center
                    ss.means[4][1].max(0.08), // REM theta center
                    ss.means[4][0].min(0.55),
                ) // REM delta center (upper bound)
            } else {
                (0.08, 0.40, 0.10, 0.35, 0.42, 0.55, 0.12, 0.50)
            };

            // --- Compute enhanced scores ---
            // Use thresholded features, not linear scaling.
            // Delta dominates in most EEG epochs; linear weighting biases everything to N3.
            let mut scores = [0.0f64; 5];

            // Sigmoid-like threshold: sharp transition at cutoff
            let sigmoid = |x: f64, center: f64, sharpness: f64| -> f64 {
                1.0 / (1.0 + (-(x - center) * sharpness).exp())
            };

            // Wake: alpha relatively high, delta NOT dominant, high wavelet entropy
            let wake_alpha = sigmoid(alpha_ratio, wake_alpha_c, 30.0);
            let wake_low_delta = sigmoid(-delta_ratio, -wake_delta_c, 15.0);
            let wake_high_wentropy = sigmoid(w_entropy, 1.5, 8.0); // broadband = Wake
            scores[0] = wake_alpha * 2.0
                + wake_low_delta * 1.5
                + wake_high_wentropy * 1.2
                + beta_ratio * 1.0
                + (1.0 - spindle_score) * 0.3;

            // N1: theta dominant relative to alpha, moderate delta
            let n1_theta = sigmoid(theta_ratio, n1_theta_c, 25.0);
            let n1_mod_delta = 1.0 - (delta_ratio - n1_delta_c).abs() * 3.0;
            let n1_low_alpha = sigmoid(-alpha_ratio, -0.10, 20.0);
            let n1_mid_wentropy = 1.0 - (w_entropy - 1.3).abs() * 2.0; // peaks mid-range
            scores[1] = n1_theta * 2.0
                + n1_mod_delta.max(0.0) * 1.0
                + n1_low_alpha * 0.8
                + n1_mid_wentropy.max(0.0) * 0.8
                + (1.0 - spindle_score) * 0.3;

            // N2: moderate delta, spindles are the hallmark
            let n2_mod_delta = 1.0 - (delta_ratio - n2_delta_c).abs() * 2.5;
            scores[2] = n2_mod_delta.max(0.0) * 1.5 + theta_ratio * 1.0
            + spindle_score * 2.5 // Spindles are THE N2 marker
            + (1.0 - wake_alpha) * 0.3;

            // N3: delta MUST be dominant, low wavelet entropy, low alpha/beta
            let n3_high_delta = sigmoid(delta_ratio, n3_delta_c, 20.0);
            let n3_low_fast = sigmoid(-(alpha_ratio + beta_ratio), -0.10, 20.0);
            let n3_low_wentropy = sigmoid(-w_entropy, -0.8, 10.0); // concentrated = N3
            scores[3] =
                n3_high_delta * 3.0 + n3_low_fast * 1.0 + n3_low_wentropy * 1.0 + pac_score * 0.8;

            // REM: theta high, delta NOT dominant, theta-gamma coupling, no spindles
            let rem_theta = sigmoid(theta_ratio, rem_theta_c, 25.0);
            let rem_low_delta = sigmoid(-delta_ratio, -rem_delta_c, 15.0);
            let rem_beta = sigmoid(beta_ratio, 0.05, 30.0);
            scores[4] = rem_theta * 1.5 + rem_low_delta * 1.5 + rem_beta * 1.0
            + theta_gamma_score * 1.2 // Theta-gamma coupling = REM marker
            + (1.0 - spindle_score) * 0.3
            + (1.0 - pac_score) * 0.3;

            // Normalize to proper probability distribution
            let sum: f64 = scores.iter().sum();
            if sum > 0.0 {
                scores.iter().map(|&s| (s / sum).max(0.01)).collect()
            } else {
                vec![0.2; 5] // uniform fallback
            }
        })
        .collect()
}

/// Find the hypnogram file for a PSG file in Sleep-EDF format.
/// PSG: SC4001E0-PSG.edf -> Hypnogram: SC4001E*-Hypnogram.edf
fn find_hypnogram(psg_path: &str) -> Option<String> {
    use std::path::Path;

    let path = Path::new(psg_path);
    let dir = path.parent()?;
    let filename = path.file_name()?.to_str()?;

    // Extract the subject ID (e.g., "SC4001" from "SC4001E0-PSG.edf")
    let subject_id = filename.get(..6)?;

    // Search for matching hypnogram file
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_str()?;
            if name_str.starts_with(subject_id) && name_str.ends_with("-Hypnogram.edf") {
                return Some(entry.path().to_str()?.to_string());
            }
        }
    }
    None
}