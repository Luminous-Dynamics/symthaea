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
use symthaea::dynamics::wavelet::{WaveletAnalyzer, DwtConfig, WaveletFamily, ExtensionMode};
use symthaea::dynamics::phase_amplitude_coupling::{PacAnalyzer, PacConfig};
use symthaea::perception::physio::{
    EdfFile, IntegrationMetrics, SleepSentinel, SleepSentinelConfig, SleepStage,
    ConsciousnessState,
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
        steps_per_epoch: 100, // Reduced for faster benchmarking
        use_spectral_analysis: true, // Welch PSD with recalibrated thresholds for proper spectral ratios
        ..SleepSentinelConfig::default()
    };

    let mut sentinel = SleepSentinel::new(config);

    // Training phase
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Training Phase ({} recordings)", n_train);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let train_start = Instant::now();
    let mut total_train_epochs = 0;

    for (i, psg_path) in psg_files[..n_train].iter().enumerate() {
        // Sleep-EDF naming: SC4001E0-PSG.edf -> SC4001E*-Hypnogram.edf
        let psg_str = psg_path.to_str().unwrap();
        let hyp_path = find_hypnogram(psg_str).unwrap_or_else(|| {
            psg_str.replace("-PSG.edf", "-Hypnogram.edf")
        });

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

        for epoch_idx in 0..n_epochs {
            if let Some((frontal, occipital, stage)) = edf.get_labeled_epoch(epoch_idx) {
                sentinel.train_epoch(&frontal, &occipital, stage);
                total_train_epochs += 1;
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
        let hyp_path = find_hypnogram(psg_str).unwrap_or_else(|| {
            psg_str.replace("-PSG.edf", "-Hypnogram.edf")
        });

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

        // Compute emission probabilities from spectral metrics + wavelet/PAC features
        let emission_sequence = compute_emission_probs_enhanced(&epoch_metrics, &epoch_signals, 100.0);

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

            if actual == raw_pred { correct_raw += 1; }
            if actual == hmm_pred { correct_hmm += 1; }
        }

        total_test_epochs += actual_indices.len();

        let n_valid = actual_indices.len();
        let acc_raw = if n_valid > 0 { correct_raw as f64 / n_valid as f64 * 100.0 } else { 0.0 };
        let acc_hmm = if n_valid > 0 { correct_hmm as f64 / n_valid as f64 * 100.0 } else { 0.0 };
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
        steps_per_epoch: 500, // Reduced from 3000 for faster benchmark
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
            sentinel.train_epoch(&frontal, &occipital, stage.clone());
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
            let (frontal, occipital) =
                generate_synthetic_eeg(stage, epoch_len, sample_rate, seed);
            let (predicted, metrics) = sentinel.process_epoch(&frontal, &occipital);
            let pred_idx = consciousness_to_idx(&predicted);

            all_actual.push(stage_idx);
            all_raw_pred.push(pred_idx);
            all_metrics.push(metrics);
            all_signals.push(frontal);
        }
    }

    // Compute emission probabilities with wavelet/PAC enhancement
    let emission_sequence = compute_emission_probs_enhanced(&all_metrics, &all_signals, sample_rate);
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
    for i in 0..5 {
        total_correct += confusion_hmm[i][i];
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
        "Wake".into(), "N1".into(), "N2".into(), "N3".into(), "REM".into(),
    ];

    HiddenMarkovModel::with_params(initial, transitions, state_names)
}

/// Enhanced emission probabilities combining spectral, wavelet, and PAC features.
///
/// Adds to the base spectral ratios:
/// 1. Wavelet spindle detection → boosts N2 confidence
/// 2. Wavelet entropy → multi-scale complexity measure
/// 3. PAC delta-beta coupling → distinguishes N3 from lighter stages
fn compute_emission_probs_enhanced(
    metrics: &[IntegrationMetrics],
    signals: &[Vec<f64>],
    sample_rate: f64,
) -> Vec<Vec<f64>> {
    // Initialize wavelet analyzer for EEG
    let wavelet_config = DwtConfig {
        wavelet: WaveletFamily::Db4,
        max_level: None,
        extension: ExtensionMode::Symmetric,
    };
    let wavelet = WaveletAnalyzer::new(wavelet_config, sample_rate);

    // Initialize PAC analyzer (fast: no surrogates for emission probs)
    let pac_config = PacConfig {
        sample_rate,
        n_phase_bins: 18,
        n_surrogates: 0, // Skip surrogates for speed (not testing significance here)
        significance_level: 0.05,
        filter_order: 128, // Shorter filter for 100 Hz data
    };
    let pac = PacAnalyzer::new(pac_config);

    // Compute band powers DIRECTLY from raw EEG via FFT (not from IntegrationMetrics)
    use symthaea::dynamics::spectral_analysis::{SpectralAnalyzer, SpectralConfig, WindowType};
    let spectral = SpectralAnalyzer::new(SpectralConfig {
        window_size: 512,
        overlap: 0.5,
        window_type: WindowType::Hann,
        sample_rate,
    });

    metrics.iter().zip(signals.iter()).map(|(m, signal)| {
        // Compute band powers from raw EEG signal via Welch PSD.
        // IntegrationMetrics come from CfC network state and don't reflect
        // epoch-to-epoch frequency variation needed for classification.
        let (delta_ratio, theta_ratio, alpha_ratio, beta_ratio, entropy) = if signal.len() >= 256 {
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
                if f >= 0.5 && f < 4.0 { delta += pv; }
                else if f >= 4.0 && f < 8.0 { theta += pv; }
                else if f >= 8.0 && f < 12.0 { alpha += pv; }
                else if f >= 12.0 && f < 30.0 { beta += pv; }
            }
            let t = total.max(1e-10);
            // Spectral entropy from PSD
            let ent: f64 = spectrum.psd.iter().map(|&p| {
                let n = (p.max(0.0) / t).max(1e-20);
                -n * n.ln()
            }).sum::<f64>() / (spectrum.psd.len() as f64).ln().max(1.0);

            (delta / t, theta / t, alpha / t, beta / t, ent)
        } else {
            // Fallback to IntegrationMetrics for short signals
            let total_power = (m.delta_power + m.theta_power + m.alpha_power + m.beta_power + m.gamma_power).max(1e-10);
            (m.delta_power as f64 / total_power as f64,
             m.theta_power as f64 / total_power as f64,
             m.alpha_power as f64 / total_power as f64,
             m.beta_power as f64 / total_power as f64,
             m.spectral_entropy as f64)
        };

        // --- Wavelet features ---
        // Spindle count (12-14 Hz bursts characteristic of N2)
        let spindles = wavelet.detect_spindles(signal);
        let spindle_count = spindles.len() as f64;
        // Normalized: 0-3+ spindles per 30s epoch
        let spindle_score = (spindle_count / 3.0).min(1.0);

        // Wavelet entropy (multi-scale complexity)
        let w_entropy = wavelet.wavelet_entropy(signal);

        // --- PAC features ---
        // Delta-beta coupling: strong in deep sleep, weak in light/REM
        let delta_beta_mi = if signal.len() >= 256 {
            let pac_result = pac.compute_pac(signal, (0.5, 4.0), (13.0, 30.0));
            pac_result.modulation_index
        } else {
            0.0
        };
        // Normalize MI (typically 0-0.3 range for EEG)
        let pac_score = (delta_beta_mi / 0.3).min(1.0);

        // --- Compute enhanced scores ---
        // Use thresholded features, not linear scaling.
        // Delta dominates in most EEG epochs; linear weighting biases everything to N3.
        let mut scores = [0.0f64; 5];

        // Sigmoid-like threshold: sharp transition at cutoff
        let sigmoid = |x: f64, center: f64, sharpness: f64| -> f64 {
            1.0 / (1.0 + (-(x - center) * sharpness).exp())
        };

        // Wake: alpha relatively high, delta NOT dominant, high entropy
        // Alpha power should be noticeable (>10%), delta should be low (<40%)
        let wake_alpha = sigmoid(alpha_ratio, 0.08, 30.0); // sharp at 8%
        let wake_low_delta = sigmoid(-delta_ratio, -0.40, 15.0); // penalize delta > 40%
        scores[0] = wake_alpha * 2.0 + wake_low_delta * 1.5 + entropy * 0.8
            + beta_ratio * 1.0 + (1.0 - spindle_score) * 0.3;

        // N1: theta dominant relative to alpha, moderate delta (25-50%)
        let n1_theta = sigmoid(theta_ratio, 0.10, 25.0);
        let n1_mod_delta = 1.0 - (delta_ratio - 0.35).abs() * 3.0; // peaks at delta=35%
        let n1_low_alpha = sigmoid(-alpha_ratio, -0.10, 20.0); // penalize alpha > 10%
        scores[1] = n1_theta * 2.0 + n1_mod_delta.max(0.0) * 1.0 + n1_low_alpha * 0.8
            + (1.0 - spindle_score) * 0.3;

        // N2: moderate delta (30-55%), spindles are the hallmark
        let n2_mod_delta = 1.0 - (delta_ratio - 0.42).abs() * 2.5;
        scores[2] = n2_mod_delta.max(0.0) * 1.5 + theta_ratio * 1.0
            + spindle_score * 2.5 // Spindles are THE N2 marker
            + (1.0 - wake_alpha) * 0.3; // Not awake

        // N3: delta MUST be dominant (>55%), low entropy, low alpha/beta
        let n3_high_delta = sigmoid(delta_ratio, 0.55, 20.0); // sharp at 55%
        let n3_low_fast = sigmoid(-(alpha_ratio + beta_ratio), -0.10, 20.0);
        scores[3] = n3_high_delta * 3.0 + n3_low_fast * 1.0
            + (1.0 - entropy) * 0.5
            + pac_score * 0.8
            + (1.0 - w_entropy) * 0.3;

        // REM: theta high, delta NOT dominant (<50%), beta present, no spindles
        let rem_theta = sigmoid(theta_ratio, 0.12, 25.0);
        let rem_low_delta = sigmoid(-delta_ratio, -0.50, 15.0);
        let rem_beta = sigmoid(beta_ratio, 0.05, 30.0);
        scores[4] = rem_theta * 1.5 + rem_low_delta * 1.5 + rem_beta * 1.0
            + entropy * 0.5
            + (1.0 - spindle_score) * 0.3
            + (1.0 - pac_score) * 0.3;

        // Normalize to proper probability distribution
        let sum: f64 = scores.iter().sum();
        if sum > 0.0 {
            scores.iter().map(|&s| (s / sum).max(0.01)).collect()
        } else {
            vec![0.2; 5] // uniform fallback
        }
    }).collect()
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
