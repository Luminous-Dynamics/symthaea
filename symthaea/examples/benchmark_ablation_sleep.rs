// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Sleep Staging Ablation Study
//!
//! Systematically removes components from the full HDC+CfC+Phi sleep staging
//! pipeline to measure each component's contribution. This proves the
//! combination matters — not just any single technique.
//!
//! ## Ablation Conditions
//! | Condition      | LTC/CfC | Spectral (Phi proxy) | Adaptive Thresholds | HMM Smoothing |
//! |----------------|---------|---------------------|---------------------|---------------|
//! | Full System    | Yes     | Yes                 | Yes                 | Yes           |
//! | No Spectral    | Yes     | No                  | Yes                 | Yes           |
//! | No Adaptive    | Yes     | Yes                 | No                  | Yes           |
//! | No HMM         | Yes     | Yes                 | Yes                 | No            |
//! | Spectral Only  | No*     | Yes                 | No                  | No            |
//! | Random         | No      | No                  | No                  | No            |
//!
//! *Spectral Only still processes through LTC but ignores LTC-derived features.
//!
//! ## Run
//! ```bash
//! cargo run --example benchmark_ablation_sleep --release
//! ```

use std::time::Instant;

use symthaea::dynamics::hmm::HiddenMarkovModel;
use symthaea::perception::physio::{
    ConsciousnessState, IntegrationMetrics, SleepSentinel, SleepSentinelConfig, SleepStage,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Sleep Staging Ablation Study                         ║");
    println!("║       Component Contribution Analysis                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let sample_rate = 100.0;
    let epoch_len = (30.0 * sample_rate) as usize;
    let train_epochs_per_stage = 10;
    let test_epochs_per_stage = 10;

    // Define ablation conditions
    let conditions: Vec<AblationCondition> = vec![
        AblationCondition {
            name: "Full System",
            use_spectral: true,
            use_adaptive: true,
            use_hmm: true,
            use_ltc: true,
        },
        AblationCondition {
            name: "No Spectral",
            use_spectral: false,
            use_adaptive: true,
            use_hmm: true,
            use_ltc: true,
        },
        AblationCondition {
            name: "No Adaptive",
            use_spectral: true,
            use_adaptive: false,
            use_hmm: true,
            use_ltc: true,
        },
        AblationCondition {
            name: "No HMM",
            use_spectral: true,
            use_adaptive: true,
            use_hmm: false,
            use_ltc: true,
        },
        AblationCondition {
            name: "Spectral Only",
            use_spectral: true,
            use_adaptive: false,
            use_hmm: false,
            use_ltc: false,
        },
        AblationCondition {
            name: "Random Baseline",
            use_spectral: false,
            use_adaptive: false,
            use_hmm: false,
            use_ltc: false,
        },
    ];

    let stages = [
        (SleepStage::Wake, "Wake"),
        (SleepStage::N1, "N1"),
        (SleepStage::N2, "N2"),
        (SleepStage::N3, "N3"),
        (SleepStage::REM, "REM"),
    ];

    let hmm = build_sleep_hmm();

    let mut all_results: Vec<AblationResult> = Vec::new();

    for condition in &conditions {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Condition: {}", condition.name);
        println!(
            "  LTC={}, Spectral={}, Adaptive={}, HMM={}",
            condition.use_ltc, condition.use_spectral, condition.use_adaptive, condition.use_hmm
        );
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let start = Instant::now();

        if !condition.use_ltc && !condition.use_spectral {
            // Random baseline: assign random predictions
            let result = run_random_baseline(test_epochs_per_stage);
            let elapsed = start.elapsed().as_secs_f64();
            print_condition_results(condition.name, &result, elapsed);
            all_results.push(result);
            continue;
        }

        let config = SleepSentinelConfig {
            local_neurons: 64,
            global_neurons: 128,
            dt_ms: 10.0,
            integration_window: 300,
            tau_base: 100.0,
            enable_adaptive_thresholds: condition.use_adaptive,
            steps_per_epoch: if condition.use_ltc { 500 } else { 50 },
            use_spectral_analysis: condition.use_spectral,
            ..SleepSentinelConfig::default()
        };

        let mut sentinel = SleepSentinel::new(config);

        // Training phase
        for (stage, _name) in &stages {
            for epoch_i in 0..train_epochs_per_stage {
                let (frontal, occipital) =
                    generate_synthetic_eeg(stage, epoch_len, sample_rate, epoch_i as u64);
                sentinel.train_epoch(&frontal, &occipital, *stage);
            }
        }

        // Testing phase
        let mut confusion_raw = vec![vec![0usize; 5]; 5];
        let mut confusion_hmm = vec![vec![0usize; 5]; 5];
        let mut all_actual = Vec::new();
        let mut all_raw_pred = Vec::new();
        let mut all_metrics = Vec::new();

        for (stage_idx, (stage, _name)) in stages.iter().enumerate() {
            for epoch_i in 0..test_epochs_per_stage {
                let seed = 10000 + epoch_i as u64;
                let (frontal, occipital) =
                    generate_synthetic_eeg(stage, epoch_len, sample_rate, seed);

                if condition.use_ltc {
                    let (predicted, metrics) = sentinel.process_epoch(&frontal, &occipital);
                    let pred_idx = consciousness_to_idx(&predicted);
                    all_actual.push(stage_idx);
                    all_raw_pred.push(pred_idx);
                    all_metrics.push(metrics);
                } else {
                    // Spectral-only: process through sentinel but classify purely on spectral
                    let (_predicted, metrics) = sentinel.process_epoch(&frontal, &occipital);
                    let pred_idx = classify_spectral_only(&metrics);
                    all_actual.push(stage_idx);
                    all_raw_pred.push(pred_idx);
                    all_metrics.push(metrics);
                }
            }
        }

        // Score raw predictions
        for j in 0..all_actual.len() {
            confusion_raw[all_actual[j]][all_raw_pred[j]] += 1;
        }

        // HMM smoothing (if enabled)
        if condition.use_hmm {
            let emission_sequence = compute_emission_probs(&all_metrics);
            let smoothed = hmm.viterbi(&emission_sequence);
            for j in 0..all_actual.len() {
                confusion_hmm[all_actual[j]][smoothed[j]] += 1;
            }
        } else {
            confusion_hmm = confusion_raw.clone();
        }

        let elapsed = start.elapsed().as_secs_f64();

        let result = AblationResult {
            name: condition.name,
            raw_confusion: confusion_raw,
            hmm_confusion: confusion_hmm,
            use_hmm: condition.use_hmm,
            elapsed_s: elapsed,
        };

        print_condition_results(condition.name, &result, elapsed);
        all_results.push(result);
    }

    // Summary table
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                     ABLATION STUDY RESULTS                             ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ {:18} │ {:>8} │ {:>8} │ {:>6} {:>6} {:>6} {:>6} {:>6} ║",
        "Condition", "Raw Acc", "HMM Acc", "Wake", "N1", "N2", "N3", "REM"
    );
    println!("╟────────────────────┼──────────┼──────────┼────────────────────────────────╢");

    for result in &all_results {
        let raw_acc = compute_accuracy(&result.raw_confusion);
        let hmm_acc = compute_accuracy(&result.hmm_confusion);
        let per_class = compute_per_class_accuracy(if result.use_hmm {
            &result.hmm_confusion
        } else {
            &result.raw_confusion
        });
        println!(
            "║ {:18} │ {:>7.1}% │ {:>7.1}% │ {:>5.1}% {:>5.1}% {:>5.1}% {:>5.1}% {:>5.1}% ║",
            result.name,
            raw_acc * 100.0,
            hmm_acc * 100.0,
            per_class[0] * 100.0,
            per_class[1] * 100.0,
            per_class[2] * 100.0,
            per_class[3] * 100.0,
            per_class[4] * 100.0,
        );
    }
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    // Validation: Full system should beat ablated versions
    println!("ABLATION VALIDATION");
    println!("═══════════════════════════════════════════════════════════════");

    let full_acc = compute_accuracy(if all_results[0].use_hmm {
        &all_results[0].hmm_confusion
    } else {
        &all_results[0].raw_confusion
    });

    for result in &all_results[1..] {
        let acc = compute_accuracy(if result.use_hmm {
            &result.hmm_confusion
        } else {
            &result.raw_confusion
        });
        let diff = (full_acc - acc) * 100.0;
        let pass = full_acc >= acc;
        println!(
            "  Full >= {:18}: {:>+6.1}pp  {}",
            result.name,
            diff,
            if pass { "PASS" } else { "FAIL" }
        );
    }

    let random_acc = compute_accuracy(&all_results[all_results.len() - 1].raw_confusion);
    println!(
        "\n  Full > Random:                    {:.1}% vs {:.1}%  {}",
        full_acc * 100.0,
        random_acc * 100.0,
        if full_acc > random_acc {
            "PASS"
        } else {
            "FAIL"
        }
    );
    println!(
        "  Full > 20% (chance):              {:.1}%  {}",
        full_acc * 100.0,
        if full_acc > 0.20 { "PASS" } else { "FAIL" }
    );

    // Save results
    let result_json = serde_json::json!({
        "benchmark": "Sleep Staging Ablation Study",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "conditions": all_results.iter().map(|r| {
            let raw_acc = compute_accuracy(&r.raw_confusion);
            let hmm_acc = compute_accuracy(&r.hmm_confusion);
            let per_class = compute_per_class_accuracy(if r.use_hmm { &r.hmm_confusion } else { &r.raw_confusion });
            serde_json::json!({
                "name": r.name,
                "raw_accuracy": raw_acc,
                "hmm_accuracy": hmm_acc,
                "best_accuracy": if r.use_hmm { hmm_acc } else { raw_acc },
                "per_class_accuracy": {
                    "Wake": per_class[0],
                    "N1": per_class[1],
                    "N2": per_class[2],
                    "N3": per_class[3],
                    "REM": per_class[4],
                },
                "elapsed_s": r.elapsed_s,
            })
        }).collect::<Vec<_>>(),
        "validation": {
            "full_beats_all_ablated": all_results[1..].iter().all(|r| {
                let acc = compute_accuracy(if r.use_hmm { &r.hmm_confusion } else { &r.raw_confusion });
                full_acc >= acc
            }),
            "full_above_chance": full_acc > 0.20,
        },
    });

    std::fs::create_dir_all("data/benchmarks/ablation-sleep").ok();
    if let Ok(f) = std::fs::File::create("data/benchmarks/ablation-sleep/results.json") {
        serde_json::to_writer_pretty(f, &result_json).ok();
        println!("\nResults saved to data/benchmarks/ablation-sleep/results.json");
    }
}

struct AblationCondition {
    name: &'static str,
    use_spectral: bool,
    use_adaptive: bool,
    use_hmm: bool,
    use_ltc: bool,
}

struct AblationResult {
    name: &'static str,
    raw_confusion: Vec<Vec<usize>>,
    hmm_confusion: Vec<Vec<usize>>,
    use_hmm: bool,
    elapsed_s: f64,
}

fn consciousness_to_idx(state: &ConsciousnessState) -> usize {
    match state {
        ConsciousnessState::Awake => 0,
        ConsciousnessState::Transitional => 1,
        ConsciousnessState::LightSleep => 2,
        ConsciousnessState::DeepSleep => 3,
        ConsciousnessState::REM => 4,
    }
}

/// Classify purely from spectral band power ratios, ignoring LTC dynamics.
fn classify_spectral_only(metrics: &IntegrationMetrics) -> usize {
    let total = (metrics.delta_power
        + metrics.theta_power
        + metrics.alpha_power
        + metrics.beta_power
        + metrics.gamma_power)
        .max(1e-10);
    let delta_r = metrics.delta_power / total;
    let theta_r = metrics.theta_power / total;
    let alpha_r = metrics.alpha_power / total;
    let beta_r = metrics.beta_power / total;

    // Simple decision tree based on band power ratios
    if delta_r > 0.45 {
        3 // N3: delta dominant
    } else if alpha_r + beta_r > 0.40 {
        0 // Wake: alpha+beta dominant
    } else if theta_r > delta_r && theta_r > alpha_r {
        // Theta dominant — could be N1 or REM
        let theta_to_delta = if delta_r > 0.01 {
            theta_r / delta_r
        } else {
            10.0
        };
        if theta_to_delta > 1.5 && beta_r > 0.08 {
            4 // REM: theta dominant with some beta, low delta
        } else {
            1 // N1: theta dominant
        }
    } else {
        2 // N2: moderate everything
    }
}

/// Random baseline: uniform random predictions.
fn run_random_baseline(test_epochs_per_stage: usize) -> AblationResult {
    let mut confusion = vec![vec![0usize; 5]; 5];
    let mut seed: u64 = 42;

    for row in &mut confusion {
        for _ in 0..test_epochs_per_stage {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let pred = (seed >> 33) as usize % 5;
            row[pred] += 1;
        }
    }

    AblationResult {
        name: "Random Baseline",
        raw_confusion: confusion.clone(),
        hmm_confusion: confusion,
        use_hmm: false,
        elapsed_s: 0.0,
    }
}

fn compute_accuracy(confusion: &[Vec<usize>]) -> f64 {
    let correct: usize = (0..5).map(|i| confusion[i][i]).sum();
    let total: usize = confusion.iter().flat_map(|r| r.iter()).sum();
    if total > 0 {
        correct as f64 / total as f64
    } else {
        0.0
    }
}

fn compute_per_class_accuracy(confusion: &[Vec<usize>]) -> Vec<f64> {
    (0..5)
        .map(|i| {
            let class_total: usize = confusion[i].iter().sum();
            if class_total > 0 {
                confusion[i][i] as f64 / class_total as f64
            } else {
                0.0
            }
        })
        .collect()
}

fn print_condition_results(name: &str, result: &AblationResult, elapsed: f64) {
    let stage_names = ["Wake", "N1", "N2", "N3", "REM"];
    let confusion = if result.use_hmm {
        &result.hmm_confusion
    } else {
        &result.raw_confusion
    };

    println!("\n  Results ({}):", name);
    for i in 0..5 {
        let class_total: usize = confusion[i].iter().sum();
        let correct = confusion[i][i];
        let acc = if class_total > 0 {
            correct as f64 / class_total as f64 * 100.0
        } else {
            0.0
        };
        println!(
            "    {:5}: {:>5.1}% ({}/{})",
            stage_names[i], acc, correct, class_total
        );
    }
    let overall = compute_accuracy(confusion);
    println!("    Overall: {:.1}% ({:.1}s)\n", overall * 100.0, elapsed);
}

/// Generate synthetic EEG signals characteristic of each sleep stage.
fn generate_synthetic_eeg(
    stage: &SleepStage,
    epoch_len: usize,
    sample_rate: f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    let mut frontal = vec![0.0f64; epoch_len];
    let mut occipital = vec![0.0f64; epoch_len];

    let mut rng_state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut rand_f64 = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng_state >> 33) as f64 / (1u64 << 31) as f64 - 1.0
    };

    let dt = 1.0 / sample_rate;

    match stage {
        SleepStage::Wake => {
            for i in 0..epoch_len {
                let t = i as f64 * dt;
                let alpha = 30.0 * (2.0 * std::f64::consts::PI * 10.0 * t).sin();
                let beta = 15.0 * (2.0 * std::f64::consts::PI * 20.0 * t).sin();
                let noise_f = 10.0 * rand_f64();
                let noise_o = 10.0 * rand_f64();
                frontal[i] = alpha * 0.3 + beta * 0.5 + noise_f;
                occipital[i] = alpha * 0.8 + beta * 0.2 + noise_o;
            }
        }
        SleepStage::N1 => {
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
            for i in 0..epoch_len {
                let t = i as f64 * dt;
                let theta = 30.0 * (2.0 * std::f64::consts::PI * 5.0 * t).sin();
                let spindle_env = ((2.0 * std::f64::consts::PI * 0.2 * t).sin()).max(0.0);
                let spindle = 25.0 * spindle_env * (2.0 * std::f64::consts::PI * 13.0 * t).sin();
                let noise_f = 12.0 * rand_f64();
                let noise_o = 12.0 * rand_f64();
                frontal[i] = theta * 0.4 + spindle * 0.8 + noise_f;
                occipital[i] = theta * 0.3 + spindle * 0.5 + noise_o;
            }
        }
        SleepStage::N3 => {
            for i in 0..epoch_len {
                let t = i as f64 * dt;
                let delta = 100.0 * (2.0 * std::f64::consts::PI * 1.0 * t).sin();
                let delta2 = 50.0 * (2.0 * std::f64::consts::PI * 0.5 * t).sin();
                let noise_f = 8.0 * rand_f64();
                let noise_o = 8.0 * rand_f64();
                frontal[i] = delta + delta2 + noise_f;
                occipital[i] = delta * 0.9 + delta2 * 0.85 + noise_o;
            }
        }
        SleepStage::REM => {
            for i in 0..epoch_len {
                let t = i as f64 * dt;
                let theta = 20.0 * (2.0 * std::f64::consts::PI * 6.0 * t).sin();
                let beta = 10.0 * (2.0 * std::f64::consts::PI * 18.0 * t).sin();
                let saw = 8.0 * (2.0 * std::f64::consts::PI * 3.0 * t).sin();
                let noise_f = 20.0 * rand_f64();
                let noise_o = 20.0 * rand_f64();
                frontal[i] = theta * 0.5 + beta * 0.3 + saw + noise_f;
                occipital[i] = theta * 0.3 + beta * 0.5 - saw * 0.5 + noise_o;
            }
        }
        _ => {
            for i in 0..epoch_len {
                frontal[i] = 50.0 * rand_f64();
                occipital[i] = 50.0 * rand_f64();
            }
        }
    }

    (frontal, occipital)
}

/// Build a physiologically-constrained HMM for sleep staging.
fn build_sleep_hmm() -> HiddenMarkovModel {
    #[rustfmt::skip]
    let transitions = vec![
        vec![0.70, 0.20, 0.05, 0.00, 0.05],
        vec![0.10, 0.50, 0.30, 0.02, 0.08],
        vec![0.02, 0.10, 0.60, 0.20, 0.08],
        vec![0.00, 0.02, 0.20, 0.75, 0.03],
        vec![0.05, 0.15, 0.10, 0.00, 0.70],
    ];

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

/// Convert IntegrationMetrics into emission probability vectors for HMM.
fn compute_emission_probs(metrics: &[IntegrationMetrics]) -> Vec<Vec<f64>> {
    metrics
        .iter()
        .map(|m| {
            let total_power =
                (m.delta_power + m.theta_power + m.alpha_power + m.beta_power + m.gamma_power)
                    .max(1e-10);
            let delta_ratio = m.delta_power as f64 / total_power as f64;
            let theta_ratio = m.theta_power as f64 / total_power as f64;
            let alpha_ratio = m.alpha_power as f64 / total_power as f64;
            let beta_ratio = m.beta_power as f64 / total_power as f64;
            let entropy = m.spectral_entropy as f64;

            let mut scores = [0.0f64; 5];
            scores[0] = (alpha_ratio + beta_ratio) * 2.0 + entropy * 0.5;
            scores[1] = theta_ratio * 2.5 + (1.0 - delta_ratio) * 0.5;
            scores[2] = theta_ratio * 1.5 + delta_ratio * 0.8 + beta_ratio * 0.5;
            scores[3] = delta_ratio * 3.0 + (1.0 - entropy) * 0.5;
            let theta_to_delta = if delta_ratio > 0.01 {
                theta_ratio / delta_ratio
            } else {
                theta_ratio * 10.0
            };
            scores[4] = theta_to_delta.min(3.0) * 0.8 + beta_ratio * 1.5 + entropy * 0.3;

            let sum: f64 = scores.iter().sum();
            if sum > 0.0 {
                scores.iter().map(|&s| (s / sum).max(0.01)).collect()
            } else {
                vec![0.2; 5]
            }
        })
        .collect()
}