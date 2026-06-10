// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unified Grammar Demo - Comparing Combined vs Individual Architectures
//!
//! This demo compares the unified architecture (Hierarchical + CfC + Grammar)
//! against individual components to measure the improvement.

use std::collections::HashMap;
use std::f32::consts::PI;
use std::path::Path;
use symthaea_stt::audio::AudioFrontend;
use symthaea_stt::temporal_grammar::{DomainConfig, Sparsity, TemporalEvent, TemporalGrammar};
use symthaea_stt::unified_grammar::UnifiedGrammar;

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

// Event extraction (reused from whale validation)
fn compute_stats(audio: &[f32], sample_rate: f32, frame_size: usize) -> (f32, f32, f32) {
    let hop = frame_size / 2;
    let mut energies = Vec::new();

    let mut i = 0;
    while i + frame_size <= audio.len() {
        let frame = &audio[i..i + frame_size];
        let energy = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt();
        energies.push(energy);
        i += hop;
    }

    let mut sorted = energies.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let p25 = sorted.get(sorted.len() / 4).copied().unwrap_or(0.0);
    let p75 = sorted.get(3 * sorted.len() / 4).copied().unwrap_or(0.0);
    let std = {
        let mean = energies.iter().sum::<f32>() / energies.len().max(1) as f32;
        (energies.iter().map(|e| (e - mean).powi(2)).sum::<f32>() / energies.len().max(1) as f32)
            .sqrt()
    };

    (p25, p75, std)
}

fn extract_events(audio: &[f32], sample_rate: f32, frame_size: usize) -> Vec<TemporalEvent> {
    let (e_p25, e_p75, _) = compute_stats(audio, sample_rate, frame_size);
    let hop = frame_size / 2;
    let dt = hop as f32 / sample_rate;
    let n_fft = frame_size.min(512);

    let mut events = Vec::new();
    let mut current: Option<(String, usize, f32, f32)> = None;
    let mut time = 0.0f32;
    let mut prev_centroid = 0.0f32;

    let mut i = 0;
    while i + frame_size <= audio.len() {
        let frame = &audio[i..i + frame_size];

        // Energy
        let energy = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt();

        // TK energy
        let tk = if frame.len() >= 3 {
            let mut tk_sum = 0.0f32;
            for j in 1..frame.len() - 1 {
                tk_sum += (frame[j] * frame[j] - frame[j - 1] * frame[j + 1]).abs();
            }
            (tk_sum / (frame.len() - 2) as f32).sqrt()
        } else {
            0.0
        };

        // Spectral centroid
        let mut spectrum = vec![0.0f32; n_fft / 2];
        for k in 0..n_fft / 2 {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;
            for (idx, &s) in frame.iter().take(n_fft).enumerate() {
                let angle = -2.0 * PI * k as f32 * idx as f32 / n_fft as f32;
                real += s * angle.cos();
                imag += s * angle.sin();
            }
            spectrum[k] = (real * real + imag * imag).sqrt();
        }

        let freq_res = sample_rate / n_fft as f32;
        let total: f32 = spectrum.iter().sum();
        let centroid = if total > 1e-10 {
            spectrum
                .iter()
                .enumerate()
                .map(|(k, &m)| k as f32 * freq_res * m)
                .sum::<f32>()
                / total
        } else {
            0.0
        };

        let bandwidth = if total > 1e-10 {
            (spectrum
                .iter()
                .enumerate()
                .map(|(k, &m)| (k as f32 * freq_res - centroid).powi(2) * m)
                .sum::<f32>()
                / total)
                .sqrt()
        } else {
            0.0
        };

        // Crest factor
        let peak = frame.iter().map(|s| s.abs()).fold(0.0f32, |a, b| a.max(b));
        let crest = if energy > 1e-10 { peak / energy } else { 0.0 };

        // FM slope
        let fm_slope = (centroid - prev_centroid) / dt;
        prev_centroid = centroid;

        // Classify
        let (etype, cid) = {
            let silence = e_p25 * 0.7;
            let loud = e_p75;
            let tk_ratio = if energy > 1e-10 { tk / energy } else { 0.0 };

            if energy < silence {
                ("silence", 8)
            } else if crest > 4.0 && tk_ratio > 0.3 {
                if energy > loud {
                    ("click_loud", 0)
                } else {
                    ("click_soft", 1)
                }
            } else {
                let bw_ratio = if centroid > 100.0 {
                    bandwidth / centroid
                } else {
                    1.0
                };
                if bw_ratio < 0.4 {
                    if fm_slope > 200.0 {
                        ("whistle_up", 2)
                    } else if fm_slope < -200.0 {
                        ("whistle_down", 3)
                    } else {
                        ("whistle_flat", 4)
                    }
                } else if bandwidth > 800.0 {
                    ("burst_rapid", 5)
                } else {
                    ("burst_slow", 6)
                }
            }
        };

        // State machine
        match &current {
            None => {
                if etype != "silence" {
                    current = Some((etype.to_string(), cid, time, energy));
                }
            }
            Some((ct, cc, st, pk)) => {
                if etype != ct {
                    let dur = time - st;
                    if dur >= 0.015 && ct != "silence" {
                        events.push(TemporalEvent::new(ct, *cc, *st, dur, *pk));
                    }
                    current = if etype != "silence" {
                        Some((etype.to_string(), cid, time, energy))
                    } else {
                        None
                    };
                } else {
                    current = Some((ct.clone(), *cc, *st, pk.max(energy)));
                }
            }
        }

        time += dt;
        i += hop;
    }

    if let Some((ct, cc, st, pk)) = current {
        let dur = time - st;
        if dur >= 0.015 && ct != "silence" {
            events.push(TemporalEvent::new(&ct, cc, st, dur, pk));
        }
    }

    events
}

/// Create baseline temporal grammar for comparison
fn baseline_cetacean_config() -> DomainConfig {
    let calls = vec![
        "click_loud",
        "click_soft",
        "whistle_up",
        "whistle_down",
        "whistle_flat",
        "burst_rapid",
        "burst_slow",
        "moan",
        "silence",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    DomainConfig {
        name: "cetacean_baseline".to_string(),
        categories: calls,
        sample_rate: 44100.0,
        frame_size: 2048,
        sparsity: Sparsity::Sparse5,
        duration_bins: 8,
        intensity_bins: 4,
        predictive_feedback: true,
        prediction_boost: 0.3,
        hierarchy_depth: 2,
    }
}

fn main() -> std::io::Result<()> {
    header("UNIFIED GRAMMAR: ARCHITECTURE COMPARISON");

    println!("  Comparing unified (Hierarchical + CfC + Grammar) vs baseline.");
    println!();
    println!("  Unified architecture combines:");
    println!("    - Hierarchical clustering (prevents saturation)");
    println!("    - CfC temporal dynamics (adaptive time constants)");
    println!("    - Sparse grammar learning (pattern recognition)");

    let whale_dir = Path::new("data/whales/sperm");

    let whale_files: Vec<_> = std::fs::read_dir(whale_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "wav")
                .unwrap_or(false)
        })
        .collect();

    if whale_files.is_empty() {
        println!("\n  No whale audio files found. Running synthetic tests only.");
        run_synthetic_tests();
        return Ok(());
    }

    println!("\n  Found {} whale recordings", whale_files.len());

    // Extract events from all recordings
    subheader("Phase 1: Event Extraction");

    let mut all_events: Vec<(String, Vec<TemporalEvent>)> = Vec::new();

    for entry in &whale_files {
        let path = entry.path();
        let filename = path.file_name().unwrap().to_string_lossy().to_string();

        if let Ok((audio, sr)) = AudioFrontend::load_wav(&path) {
            let events = extract_events(&audio, sr as f32, 2048);
            println!("    {}: {} events", filename, events.len());
            all_events.push((filename, events));
        }
    }

    // Split train/test
    let mid = all_events.len() / 2;
    let (train_set, test_set) = all_events.split_at(mid);

    // Create both grammars
    let mut unified = UnifiedGrammar::cetacean();
    let mut baseline = TemporalGrammar::new(baseline_cetacean_config());

    // Train both
    subheader("Phase 2: Training");

    println!("    Training unified and baseline grammars...\n");

    for (filename, events) in train_set {
        if events.len() >= 3 {
            for _ in 0..25 {
                unified.train_sequence(events);
                baseline.train_sequence(events);
            }
            println!("    Trained: {} ({} events)", filename, events.len());
        }
    }

    let unified_stats = unified.stats();
    let baseline_stats = baseline.stats();

    println!(
        "\n    Unified:  {} clusters, {:.1}% global density, {:.1}% cluster density",
        unified_stats.num_clusters,
        unified_stats.global_density * 100.0,
        unified_stats.avg_cluster_density * 100.0
    );
    println!(
        "    Baseline: {:.1}% density",
        baseline_stats.grammar_density * 100.0
    );

    // Score both
    subheader("Phase 3: Scoring Comparison");

    let mut unified_train = Vec::new();
    let mut unified_test = Vec::new();
    let mut baseline_train = Vec::new();
    let mut baseline_test = Vec::new();

    println!("    Training set:");
    for (filename, events) in train_set {
        if events.len() >= 3 {
            let u_score = unified.score_sequence(events);
            let b_score = baseline.score_sequence(events);
            unified_train.push(u_score);
            baseline_train.push(b_score);
            println!(
                "      {}: unified={:+.4} baseline={:+.4} (Δ={:+.4})",
                filename,
                u_score,
                b_score,
                u_score - b_score
            );
        }
    }

    println!("\n    Test set:");
    for (filename, events) in test_set {
        if events.len() >= 3 {
            let u_score = unified.score_sequence(events);
            let b_score = baseline.score_sequence(events);
            unified_test.push(u_score);
            baseline_test.push(b_score);
            println!(
                "      {}: unified={:+.4} baseline={:+.4} (Δ={:+.4})",
                filename,
                u_score,
                b_score,
                u_score - b_score
            );
        }
    }

    // Calculate averages
    let u_train_avg = unified_train.iter().sum::<f32>() / unified_train.len().max(1) as f32;
    let u_test_avg = unified_test.iter().sum::<f32>() / unified_test.len().max(1) as f32;
    let b_train_avg = baseline_train.iter().sum::<f32>() / baseline_train.len().max(1) as f32;
    let b_test_avg = baseline_test.iter().sum::<f32>() / baseline_test.len().max(1) as f32;

    // Results
    header("COMPARISON RESULTS");

    println!("    UNIFIED (Hierarchical + CfC + Grammar):");
    println!("      Train avg:        {:+.4}", u_train_avg);
    println!("      Test avg:         {:+.4}", u_test_avg);
    println!("      Generalization:   {:+.4}", u_train_avg - u_test_avg);
    println!();

    println!("    BASELINE (Temporal Grammar only):");
    println!("      Train avg:        {:+.4}", b_train_avg);
    println!("      Test avg:         {:+.4}", b_test_avg);
    println!("      Generalization:   {:+.4}", b_train_avg - b_test_avg);
    println!();

    let improvement = u_train_avg - b_train_avg;
    println!("    IMPROVEMENT:");
    println!("      Unified vs Baseline: {:+.4}", improvement);

    if improvement > 0.0 {
        let pct = if b_train_avg.abs() > 0.001 {
            (improvement / b_train_avg.abs()) * 100.0
        } else {
            f32::INFINITY
        };
        println!("      Percentage gain:     {:.1}%", pct);
        println!();
        println!("  [SUCCESS] Unified architecture outperforms baseline!");
    } else {
        println!();
        println!("  [INFO] Baseline performed better on this dataset.");
    }

    separator('=', 70);

    Ok(())
}

/// Run synthetic tests when no whale data available
fn run_synthetic_tests() {
    subheader("Synthetic Pattern Tests");

    // Create grammars
    let mut unified = UnifiedGrammar::cetacean();
    let mut baseline = TemporalGrammar::new(baseline_cetacean_config());

    // Sperm whale click train pattern
    let click_train: Vec<TemporalEvent> = vec![
        TemporalEvent::new("click_loud", 0, 0.0, 0.01, 0.9),
        TemporalEvent::new("click_soft", 1, 0.05, 0.01, 0.5),
        TemporalEvent::new("click_soft", 1, 0.10, 0.01, 0.5),
        TemporalEvent::new("click_soft", 1, 0.15, 0.01, 0.4),
        TemporalEvent::new("click_loud", 0, 0.20, 0.01, 0.85),
    ];

    // Humpback song pattern (different)
    let song: Vec<TemporalEvent> = vec![
        TemporalEvent::new("moan", 7, 0.0, 2.0, 0.7),
        TemporalEvent::new("whistle_up", 2, 2.5, 1.0, 0.6),
        TemporalEvent::new("whistle_down", 3, 4.0, 1.5, 0.65),
        TemporalEvent::new("moan", 7, 6.0, 2.5, 0.75),
    ];

    // Dolphin burst pattern (novel)
    let bursts: Vec<TemporalEvent> = vec![
        TemporalEvent::new("burst_rapid", 5, 0.0, 0.1, 0.6),
        TemporalEvent::new("whistle_flat", 4, 0.2, 0.5, 0.55),
        TemporalEvent::new("burst_rapid", 5, 0.8, 0.1, 0.65),
        TemporalEvent::new("burst_slow", 6, 1.0, 0.3, 0.5),
    ];

    println!("    Training on click train pattern (sperm whale)...\n");

    // Train both on click train
    for _ in 0..50 {
        unified.train_sequence(&click_train);
        baseline.train_sequence(&click_train);
    }

    // Score all patterns
    let patterns = vec![
        ("Click train (trained)", &click_train),
        ("Humpback song (different)", &song),
        ("Dolphin bursts (novel)", &bursts),
    ];

    println!("    Scoring patterns:\n");
    println!(
        "    {:30} {:>10} {:>10} {:>10}",
        "Pattern", "Unified", "Baseline", "Δ"
    );
    println!("    {}", "-".repeat(62));

    let mut unified_trained = 0.0f32;
    let mut unified_other = 0.0f32;
    let mut baseline_trained = 0.0f32;
    let mut baseline_other = 0.0f32;

    for (i, (name, events)) in patterns.iter().enumerate() {
        let u = unified.score_sequence(events);
        let b = baseline.score_sequence(events);

        println!("    {:30} {:+10.4} {:+10.4} {:+10.4}", name, u, b, u - b);

        if i == 0 {
            unified_trained = u;
            baseline_trained = b;
        } else {
            unified_other += u;
            baseline_other += b;
        }
    }

    unified_other /= 2.0;
    baseline_other /= 2.0;

    println!();
    println!("    Discrimination (trained - other):");
    println!("      Unified:  {:+.4}", unified_trained - unified_other);
    println!("      Baseline: {:+.4}", baseline_trained - baseline_other);

    let u_disc = unified_trained - unified_other;
    let b_disc = baseline_trained - baseline_other;

    header("SYNTHETIC TEST RESULTS");

    println!("    Unified discrimination:  {:+.4}", u_disc);
    println!("    Baseline discrimination: {:+.4}", b_disc);
    println!("    Improvement:             {:+.4}", u_disc - b_disc);
    println!();

    if u_disc > b_disc {
        println!("  [SUCCESS] Unified architecture shows better discrimination!");
    } else if u_disc == b_disc {
        println!("  [INFO] Both architectures perform similarly.");
    } else {
        println!("  [INFO] Baseline performed better on synthetic data.");
    }

    // Test CfC temporal adaptation
    subheader("CfC Temporal Adaptation Test");

    // Rapid events vs slow events
    let rapid_events: Vec<TemporalEvent> = (0..10)
        .map(|i| TemporalEvent::new("click_soft", 1, i as f32 * 0.02, 0.01, 0.5))
        .collect();

    let slow_events: Vec<TemporalEvent> = (0..5)
        .map(|i| TemporalEvent::new("moan", 7, i as f32 * 1.0, 0.8, 0.6))
        .collect();

    // Train on rapid
    let mut unified_rapid = UnifiedGrammar::cetacean();
    for _ in 0..30 {
        unified_rapid.train_sequence(&rapid_events);
    }

    let rapid_score = unified_rapid.score_sequence(&rapid_events);
    let slow_on_rapid = unified_rapid.score_sequence(&slow_events);

    println!("    Trained on rapid click events:");
    println!("      Rapid events score: {:+.4}", rapid_score);
    println!("      Slow events score:  {:+.4}", slow_on_rapid);
    println!(
        "      Temporal discrimination: {:+.4}",
        rapid_score - slow_on_rapid
    );

    if rapid_score > slow_on_rapid {
        println!("\n  [SUCCESS] CfC adapts to temporal dynamics!");
    }

    separator('=', 70);
}
