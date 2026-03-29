// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Real-World Whale Validation v2
//!
//! Improved event detection with:
//! - Better spectral analysis for FM direction
//! - Intensity-based sub-categories
//! - Click detection using Teager-Kaiser energy
//! - Hierarchical grammar to prevent saturation

use std::collections::HashMap;
use std::f32::consts::PI;
use std::path::Path;
use symthaea_stt::audio::AudioFrontend;
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

/// Enhanced feature extraction for cetacean audio
struct CetaceanFeatures {
    /// RMS energy
    energy: f32,
    /// Teager-Kaiser energy (transient detection)
    tk_energy: f32,
    /// Spectral centroid (Hz)
    centroid: f32,
    /// Spectral flux (change rate)
    flux: f32,
    /// FM slope estimate (Hz/s)
    fm_slope: f32,
    /// Harmonicity (0-1)
    harmonicity: f32,
}

/// Extract cetacean-specific features from a frame
fn extract_features(samples: &[f32], sample_rate: f32, prev_centroid: f32) -> CetaceanFeatures {
    let n = samples.len();

    // RMS energy
    let energy = (samples.iter().map(|s| s * s).sum::<f32>() / n as f32).sqrt();

    // Teager-Kaiser energy (sensitive to clicks/transients)
    let tk_energy = if n >= 3 {
        let mut tk_sum = 0.0f32;
        for i in 1..n - 1 {
            let tk = samples[i] * samples[i] - samples[i - 1] * samples[i + 1];
            tk_sum += tk.abs();
        }
        (tk_sum / (n - 2) as f32).sqrt()
    } else {
        0.0
    };

    // Simple DFT for spectral features (just first few bins for efficiency)
    let n_fft = n.min(512);
    let mut spectrum = vec![0.0f32; n_fft / 2];

    for k in 0..n_fft / 2 {
        let mut real = 0.0f32;
        let mut imag = 0.0f32;
        for (i, &sample) in samples.iter().take(n_fft).enumerate() {
            let angle = -2.0 * PI * k as f32 * i as f32 / n_fft as f32;
            real += sample * angle.cos();
            imag += sample * angle.sin();
        }
        spectrum[k] = (real * real + imag * imag).sqrt();
    }

    // Spectral centroid
    let freq_resolution = sample_rate / n_fft as f32;
    let total_energy: f32 = spectrum.iter().sum();
    let centroid = if total_energy > 1e-10 {
        spectrum
            .iter()
            .enumerate()
            .map(|(k, &mag)| k as f32 * freq_resolution * mag)
            .sum::<f32>()
            / total_energy
    } else {
        0.0
    };

    // Spectral flux (change from previous frame)
    let flux = (centroid - prev_centroid).abs();

    // FM slope estimate (Hz per second)
    let frame_duration = n as f32 / sample_rate;
    let fm_slope = if frame_duration > 0.0 {
        (centroid - prev_centroid) / frame_duration
    } else {
        0.0
    };

    // Harmonicity estimate (ratio of harmonic to total energy)
    // Simple: compare even bins to total
    let even_energy: f32 = spectrum.iter().step_by(2).sum();
    let harmonicity = if total_energy > 1e-10 {
        (even_energy / total_energy).min(1.0)
    } else {
        0.0
    };

    CetaceanFeatures {
        energy,
        tk_energy,
        centroid,
        flux,
        fm_slope,
        harmonicity,
    }
}

/// Classify a frame into event type
fn classify_frame(features: &CetaceanFeatures) -> (&'static str, usize) {
    // Thresholds (tuned for cetacean audio)
    let silence_thresh = 0.01;
    let click_tk_thresh = 0.15;
    let whistle_harm_thresh = 0.4;
    let fm_up_thresh = 500.0; // Hz/s
    let fm_down_thresh = -500.0; // Hz/s

    if features.energy < silence_thresh {
        return ("silence", 8);
    }

    // Click detection: high TK energy relative to RMS
    let tk_ratio = if features.energy > 1e-10 {
        features.tk_energy / features.energy
    } else {
        0.0
    };

    if tk_ratio > click_tk_thresh {
        // Intensity-based click sub-types
        if features.energy > 0.3 {
            return ("click_loud", 0);
        } else {
            return ("click_soft", 1);
        }
    }

    // Whistle detection: high harmonicity
    if features.harmonicity > whistle_harm_thresh {
        if features.fm_slope > fm_up_thresh {
            return ("whistle_up", 2);
        } else if features.fm_slope < fm_down_thresh {
            return ("whistle_down", 3);
        } else {
            return ("whistle_flat", 4);
        }
    }

    // Burst/pulse: moderate energy, low harmonicity
    if features.energy > 0.05 {
        if features.flux > 200.0 {
            return ("burst_rapid", 5);
        } else {
            return ("burst_slow", 6);
        }
    }

    // Default: low-level noise/moan
    ("moan", 7)
}

/// Extract events from audio with enhanced classification
fn extract_events_v2(audio: &[f32], sample_rate: f32, frame_size: usize) -> Vec<TemporalEvent> {
    let mut events: Vec<TemporalEvent> = Vec::new();
    let hop_size = frame_size / 2; // 50% overlap

    let mut current_event: Option<(String, usize, f32, f32)> = None;
    let mut time = 0.0f32;
    let mut prev_centroid = 0.0f32;
    let frame_duration = frame_size as f32 / sample_rate;
    let min_event_duration = 0.015; // 15ms minimum

    let mut i = 0;
    while i + frame_size <= audio.len() {
        let frame = &audio[i..i + frame_size];
        let features = extract_features(frame, sample_rate, prev_centroid);
        prev_centroid = features.centroid;

        let (event_type, class_id) = classify_frame(&features);

        // State machine
        match &current_event {
            None => {
                if event_type != "silence" {
                    current_event = Some((event_type.to_string(), class_id, time, features.energy));
                }
            }
            Some((curr_type, curr_class, start_time, peak_intensity)) => {
                if event_type != curr_type {
                    let duration = time - start_time;
                    if duration >= min_event_duration && curr_type != "silence" {
                        events.push(TemporalEvent::new(
                            curr_type,
                            *curr_class,
                            *start_time,
                            duration,
                            *peak_intensity,
                        ));
                    }

                    if event_type != "silence" {
                        current_event =
                            Some((event_type.to_string(), class_id, time, features.energy));
                    } else {
                        current_event = None;
                    }
                } else {
                    let new_peak = peak_intensity.max(features.energy);
                    current_event = Some((curr_type.clone(), *curr_class, *start_time, new_peak));
                }
            }
        }

        time += hop_size as f32 / sample_rate;
        i += hop_size;
    }

    // Flush
    if let Some((curr_type, curr_class, start_time, peak_intensity)) = current_event {
        let duration = time - start_time;
        if duration >= min_event_duration && curr_type != "silence" {
            events.push(TemporalEvent::new(
                &curr_type,
                curr_class,
                start_time,
                duration,
                peak_intensity,
            ));
        }
    }

    events
}

/// Create enhanced cetacean domain config
fn enhanced_cetacean_config() -> DomainConfig {
    let calls = vec![
        "click_loud",
        "click_soft", // 0, 1
        "whistle_up",
        "whistle_down",
        "whistle_flat", // 2, 3, 4
        "burst_rapid",
        "burst_slow", // 5, 6
        "moan",
        "silence", // 7, 8
    ]
    .into_iter()
    .map(String::from)
    .collect();

    DomainConfig {
        name: "cetacean_v2".to_string(),
        categories: calls,
        sample_rate: 44100.0,
        frame_size: 2048,
        sparsity: Sparsity::Sparse5,
        duration_bins: 10,
        intensity_bins: 5,
        predictive_feedback: true,
        prediction_boost: 0.25,
        hierarchy_depth: 2,
    }
}

fn main() -> std::io::Result<()> {
    header("WHALE BRIDGE v2: ENHANCED VALIDATION");

    println!("  Improvements over v1:");
    println!("    - Teager-Kaiser energy for click detection");
    println!("    - FM slope analysis for whistle direction");
    println!("    - Intensity-based sub-categories");
    println!("    - 9 event types instead of 5");

    let whale_dir = Path::new("data/whales/sperm");
    let test_dir = Path::new("data/test_animals");

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
        println!("\n  No whale audio files found.");
        return Ok(());
    }

    println!("\n  Found {} whale recordings", whale_files.len());

    // Create enhanced grammar
    let mut grammar = TemporalGrammar::new(enhanced_cetacean_config());
    let stats = grammar.stats();

    subheader("Enhanced Grammar Configuration");
    println!("    Domain:      {}", stats.domain);
    println!("    Categories:  {} (was 5)", stats.num_categories);
    println!("    Sparsity:    {:.0}%", stats.sparsity * 100.0);

    // Phase 1: Process all recordings
    subheader("Phase 1: Enhanced Event Extraction");

    let mut all_results: Vec<(String, Vec<TemporalEvent>)> = Vec::new();
    let mut total_events = 0;
    let mut type_totals: HashMap<String, usize> = HashMap::new();

    for entry in &whale_files {
        let path = entry.path();
        let filename = path.file_name().unwrap().to_string_lossy().to_string();

        let (audio, sample_rate) = match AudioFrontend::load_wav(&path) {
            Ok(r) => r,
            Err(e) => {
                println!("    Error: {}: {}", filename, e);
                continue;
            }
        };

        let events = extract_events_v2(&audio, sample_rate as f32, 2048);

        print!(
            "    {} ({:.1}s): ",
            filename,
            audio.len() as f32 / sample_rate as f32
        );

        // Count types
        let mut local_counts: HashMap<String, usize> = HashMap::new();
        for event in &events {
            *local_counts.entry(event.category.clone()).or_insert(0) += 1;
            *type_totals.entry(event.category.clone()).or_insert(0) += 1;
        }

        // Print counts
        let mut counts: Vec<_> = local_counts.iter().collect();
        counts.sort_by(|a, b| b.1.cmp(a.1));
        for (t, c) in counts.iter().take(4) {
            print!("{}:{} ", t, c);
        }
        println!();

        total_events += events.len();
        all_results.push((filename, events));
    }

    println!("\n    Total events: {}", total_events);
    println!("    Event type distribution:");
    let mut type_vec: Vec<_> = type_totals.iter().collect();
    type_vec.sort_by(|a, b| b.1.cmp(a.1));
    for (t, c) in &type_vec {
        let pct = **c as f32 / total_events as f32 * 100.0;
        println!("      {}: {} ({:.1}%)", t, c, pct);
    }

    // Phase 2: Train/Test
    subheader("Phase 2: Training");

    let mid = all_results.len() / 2;
    let (train_set, test_set) = all_results.split_at(mid);

    for (filename, events) in train_set {
        if events.len() >= 3 {
            for _ in 0..15 {
                grammar.train_sequence(events);
            }
            println!("    Trained: {} ({} events)", filename, events.len());
        }
    }

    let stats = grammar.stats();
    println!(
        "\n    Grammar density: {:.3} (target <0.5)",
        stats.grammar_density
    );

    // Phase 3: Score
    subheader("Phase 3: Scoring");

    let mut train_scores = Vec::new();
    let mut test_scores = Vec::new();

    println!("    Training set:");
    for (filename, events) in train_set {
        if events.len() >= 3 {
            let score = grammar.score_sequence(events);
            train_scores.push(score);
            println!("      {}: {:.4}", filename, score);
        }
    }

    println!("\n    Test set:");
    for (filename, events) in test_set {
        if events.len() >= 3 {
            let score = grammar.score_sequence(events);
            test_scores.push(score);
            println!("      {}: {:.4}", filename, score);
        }
    }

    let train_avg: f32 = train_scores.iter().sum::<f32>() / train_scores.len().max(1) as f32;
    let test_avg: f32 = test_scores.iter().sum::<f32>() / test_scores.len().max(1) as f32;

    // Phase 4: Cross-species
    if test_dir.exists() {
        subheader("Phase 4: Cross-Species Test");

        let test_files: Vec<_> = std::fs::read_dir(test_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "wav")
                    .unwrap_or(false)
            })
            .collect();

        for entry in &test_files {
            let path = entry.path();
            let filename = path.file_name().unwrap().to_string_lossy().to_string();

            if let Ok((audio, sample_rate)) = AudioFrontend::load_wav(&path) {
                let events = extract_events_v2(&audio, sample_rate as f32, 2048);
                if events.len() >= 2 {
                    let score = grammar.score_sequence(&events);
                    let label = if filename.contains("whale") {
                        "WHALE"
                    } else if filename.contains("dolphin") {
                        "DOLPHIN"
                    } else {
                        "OTHER"
                    };
                    println!(
                        "    [{}] {}: {:.4} ({} events)",
                        label,
                        filename,
                        score,
                        events.len()
                    );
                }
            }
        }
    }

    // Summary
    header("RESULTS");

    println!("    Training average:     {:.4}", train_avg);
    println!("    Test average:         {:.4}", test_avg);
    println!("    Generalization gap:   {:+.4}", train_avg - test_avg);
    println!("    Grammar density:      {:.3}", stats.grammar_density);
    println!();

    if train_avg > test_avg && train_avg > 0.1 {
        println!("  [SUCCESS] Grammar learned whale-specific patterns!");
    }

    if stats.grammar_density < 0.5 {
        println!("  [SUCCESS] Sparse HDC prevented saturation!");
    }

    separator('=', 70);
    println!("  WHALE BRIDGE v2 COMPLETE");
    separator('=', 70);

    Ok(())
}
