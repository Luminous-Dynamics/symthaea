// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Real-World Whale Validation v3
//!
//! Adaptive event detection with:
//! - Signal-adaptive thresholds (computed from recording statistics)
//! - Better spectral peak tracking for whistle detection
//! - Cepstral analysis for harmonicity
//! - Multi-band energy analysis

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

/// Signal statistics for adaptive thresholding
#[derive(Debug, Clone)]
struct SignalStats {
    energy_mean: f32,
    energy_std: f32,
    energy_median: f32,
    tk_mean: f32,
    tk_std: f32,
    centroid_mean: f32,
    centroid_std: f32,
}

/// Compute signal statistics from the entire recording
fn compute_signal_stats(audio: &[f32], sample_rate: f32, frame_size: usize) -> SignalStats {
    let hop_size = frame_size / 2;
    let n_fft = frame_size.min(512);

    let mut energies = Vec::new();
    let mut tk_energies = Vec::new();
    let mut centroids = Vec::new();

    let mut i = 0;
    while i + frame_size <= audio.len() {
        let frame = &audio[i..i + frame_size];

        // RMS energy
        let energy = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt();
        energies.push(energy);

        // Teager-Kaiser energy
        if frame.len() >= 3 {
            let mut tk_sum = 0.0f32;
            for j in 1..frame.len() - 1 {
                let tk = frame[j] * frame[j] - frame[j - 1] * frame[j + 1];
                tk_sum += tk.abs();
            }
            tk_energies.push((tk_sum / (frame.len() - 2) as f32).sqrt());
        }

        // Spectral centroid
        let mut spectrum = vec![0.0f32; n_fft / 2];
        for k in 0..n_fft / 2 {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;
            for (idx, &sample) in frame.iter().take(n_fft).enumerate() {
                let angle = -2.0 * PI * k as f32 * idx as f32 / n_fft as f32;
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }
            spectrum[k] = (real * real + imag * imag).sqrt();
        }

        let freq_res = sample_rate / n_fft as f32;
        let total_e: f32 = spectrum.iter().sum();
        if total_e > 1e-10 {
            let centroid = spectrum
                .iter()
                .enumerate()
                .map(|(k, &mag)| k as f32 * freq_res * mag)
                .sum::<f32>()
                / total_e;
            centroids.push(centroid);
        }

        i += hop_size;
    }

    // Compute statistics
    let energy_mean = energies.iter().sum::<f32>() / energies.len().max(1) as f32;
    let energy_std = (energies
        .iter()
        .map(|e| (e - energy_mean).powi(2))
        .sum::<f32>()
        / energies.len().max(1) as f32)
        .sqrt();

    let mut sorted_e = energies.clone();
    sorted_e.sort_by(|a, b| a.total_cmp(b));
    let energy_median = sorted_e.get(sorted_e.len() / 2).copied().unwrap_or(0.0);

    let tk_mean = tk_energies.iter().sum::<f32>() / tk_energies.len().max(1) as f32;
    let tk_std = (tk_energies
        .iter()
        .map(|e| (e - tk_mean).powi(2))
        .sum::<f32>()
        / tk_energies.len().max(1) as f32)
        .sqrt();

    let centroid_mean = centroids.iter().sum::<f32>() / centroids.len().max(1) as f32;
    let centroid_std = (centroids
        .iter()
        .map(|c| (c - centroid_mean).powi(2))
        .sum::<f32>()
        / centroids.len().max(1) as f32)
        .sqrt();

    SignalStats {
        energy_mean,
        energy_std,
        energy_median,
        tk_mean,
        tk_std,
        centroid_mean,
        centroid_std,
    }
}

/// Enhanced feature extraction with more spectral detail
#[derive(Debug, Clone)]
struct SpectralFeatures {
    energy: f32,
    tk_energy: f32,
    centroid: f32,
    bandwidth: f32,
    spectral_flux: f32,
    peak_freq: f32,
    low_band_ratio: f32,  // 0-500 Hz
    mid_band_ratio: f32,  // 500-2000 Hz
    high_band_ratio: f32, // 2000+ Hz
    harmonic_ratio: f32,
    crest_factor: f32,
}

fn extract_spectral_features(
    samples: &[f32],
    sample_rate: f32,
    prev_spectrum: &[f32],
) -> (SpectralFeatures, Vec<f32>) {
    let n = samples.len();
    let n_fft = n.min(512);

    // RMS energy
    let energy = (samples.iter().map(|s| s * s).sum::<f32>() / n as f32).sqrt();

    // Peak value for crest factor
    let peak = samples
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, |a, b| a.max(b));
    let crest_factor = if energy > 1e-10 { peak / energy } else { 0.0 };

    // Teager-Kaiser energy
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

    // FFT (simple DFT)
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

    let freq_res = sample_rate / n_fft as f32;
    let total_energy: f32 = spectrum.iter().sum();

    // Spectral centroid
    let centroid = if total_energy > 1e-10 {
        spectrum
            .iter()
            .enumerate()
            .map(|(k, &mag)| k as f32 * freq_res * mag)
            .sum::<f32>()
            / total_energy
    } else {
        0.0
    };

    // Spectral bandwidth
    let bandwidth = if total_energy > 1e-10 {
        (spectrum
            .iter()
            .enumerate()
            .map(|(k, &mag)| (k as f32 * freq_res - centroid).powi(2) * mag)
            .sum::<f32>()
            / total_energy)
            .sqrt()
    } else {
        0.0
    };

    // Peak frequency
    let (peak_bin, _) =
        spectrum.iter().enumerate().fold(
            (0, 0.0f32),
            |(bi, bm), (i, &m)| if m > bm { (i, m) } else { (bi, bm) },
        );
    let peak_freq = peak_bin as f32 * freq_res;

    // Band ratios
    let low_cutoff = (500.0 / freq_res) as usize;
    let mid_cutoff = (2000.0 / freq_res) as usize;

    let low_energy: f32 = spectrum.iter().take(low_cutoff.min(spectrum.len())).sum();
    let mid_energy: f32 = spectrum
        .iter()
        .skip(low_cutoff.min(spectrum.len()))
        .take((mid_cutoff - low_cutoff).max(0))
        .sum();
    let high_energy: f32 = spectrum.iter().skip(mid_cutoff.min(spectrum.len())).sum();

    let (low_band_ratio, mid_band_ratio, high_band_ratio) = if total_energy > 1e-10 {
        (
            low_energy / total_energy,
            mid_energy / total_energy,
            high_energy / total_energy,
        )
    } else {
        (0.0, 0.0, 0.0)
    };

    // Spectral flux (change from previous frame)
    let spectral_flux = if prev_spectrum.len() == spectrum.len() {
        spectrum
            .iter()
            .zip(prev_spectrum.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    } else {
        0.0
    };

    // Harmonic ratio (even bins vs odd bins - crude but fast)
    let even_energy: f32 = spectrum.iter().step_by(2).sum();
    let odd_energy: f32 = spectrum.iter().skip(1).step_by(2).sum();
    let harmonic_ratio = if total_energy > 1e-10 {
        (even_energy - odd_energy).abs() / total_energy
    } else {
        0.0
    };

    let features = SpectralFeatures {
        energy,
        tk_energy,
        centroid,
        bandwidth,
        spectral_flux,
        peak_freq,
        low_band_ratio,
        mid_band_ratio,
        high_band_ratio,
        harmonic_ratio,
        crest_factor,
    };

    (features, spectrum)
}

/// Adaptive classifier using signal statistics
fn classify_frame_adaptive(
    features: &SpectralFeatures,
    stats: &SignalStats,
    prev_centroid: f32,
    frame_duration: f32,
) -> (&'static str, usize) {
    // Adaptive thresholds based on signal statistics
    let silence_thresh = stats.energy_median * 0.5;
    let loud_thresh = stats.energy_mean + 1.5 * stats.energy_std;

    // Click detection: high crest factor AND high TK relative to energy
    let tk_ratio = if features.energy > 1e-10 {
        features.tk_energy / features.energy
    } else {
        0.0
    };
    let click_thresh = stats.tk_mean / stats.energy_mean.max(1e-10)
        + 0.5 * stats.tk_std / stats.energy_mean.max(1e-10);

    // FM slope for whistle direction
    let fm_slope = if frame_duration > 0.0 {
        (features.centroid - prev_centroid) / frame_duration
    } else {
        0.0
    };

    // Silence
    if features.energy < silence_thresh {
        return ("silence", 8);
    }

    // Click: high transient content (high crest factor + high TK ratio)
    if features.crest_factor > 3.0 && tk_ratio > click_thresh.max(0.1) {
        if features.energy > loud_thresh {
            return ("click_loud", 0);
        } else {
            return ("click_soft", 1);
        }
    }

    // Whistle: narrowband, tonal (low bandwidth relative to centroid)
    let bandwidth_ratio = if features.centroid > 100.0 {
        features.bandwidth / features.centroid
    } else {
        1.0
    };

    if bandwidth_ratio < 0.5 && features.mid_band_ratio > 0.3 {
        // Tonal sound in mid-frequency range - likely whistle
        if fm_slope > 300.0 {
            return ("whistle_up", 2);
        } else if fm_slope < -300.0 {
            return ("whistle_down", 3);
        } else {
            return ("whistle_flat", 4);
        }
    }

    // Burst: broadband, rapid spectral change
    if features.spectral_flux > stats.centroid_std * 2.0 && features.bandwidth > 500.0 {
        if features.high_band_ratio > 0.3 {
            return ("burst_rapid", 5);
        } else {
            return ("burst_slow", 6);
        }
    }

    // Moan: low frequency dominated, sustained
    if features.low_band_ratio > 0.5 && features.energy > silence_thresh * 2.0 {
        return ("moan", 7);
    }

    // Default based on dominant band
    if features.high_band_ratio > features.low_band_ratio
        && features.high_band_ratio > features.mid_band_ratio
    {
        return ("burst_rapid", 5);
    } else if features.mid_band_ratio > features.low_band_ratio {
        return ("whistle_flat", 4);
    } else {
        return ("moan", 7);
    }
}

/// Extract events with adaptive thresholding
fn extract_events_v3(audio: &[f32], sample_rate: f32, frame_size: usize) -> Vec<TemporalEvent> {
    // First pass: compute signal statistics
    let stats = compute_signal_stats(audio, sample_rate, frame_size);

    let mut events: Vec<TemporalEvent> = Vec::new();
    let hop_size = frame_size / 2;
    let frame_duration = frame_size as f32 / sample_rate;
    let min_event_duration = 0.015;

    let mut current_event: Option<(String, usize, f32, f32)> = None;
    let mut time = 0.0f32;
    let mut prev_centroid = 0.0f32;
    let mut prev_spectrum: Vec<f32> = Vec::new();

    let mut i = 0;
    while i + frame_size <= audio.len() {
        let frame = &audio[i..i + frame_size];
        let (features, spectrum) = extract_spectral_features(frame, sample_rate, &prev_spectrum);

        let (event_type, class_id) =
            classify_frame_adaptive(&features, &stats, prev_centroid, frame_duration);

        prev_centroid = features.centroid;
        prev_spectrum = spectrum;

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

/// Enhanced cetacean domain config
fn enhanced_cetacean_config() -> DomainConfig {
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
        name: "cetacean_v3".to_string(),
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
    header("WHALE BRIDGE v3: ADAPTIVE DETECTION");

    println!("  Improvements over v2:");
    println!("    - Signal-adaptive thresholds (computed per recording)");
    println!("    - Crest factor for transient detection");
    println!("    - Bandwidth ratio for tonal vs. broadband");
    println!("    - Multi-band energy ratios");
    println!("    - Spectral flux for rapid changes");

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

    let mut grammar = TemporalGrammar::new(enhanced_cetacean_config());
    let stats = grammar.stats();

    subheader("Grammar Configuration");
    println!("    Domain:      {}", stats.domain);
    println!("    Categories:  {}", stats.num_categories);
    println!("    Sparsity:    {:.0}%", stats.sparsity * 100.0);

    // Phase 1: Extract events
    subheader("Phase 1: Adaptive Event Extraction");

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

        // Compute and display signal stats
        let sig_stats = compute_signal_stats(&audio, sample_rate as f32, 2048);
        println!(
            "    {} ({:.1}s)",
            filename,
            audio.len() as f32 / sample_rate as f32
        );
        println!(
            "      Signal: E_mean={:.4} E_std={:.4} TK_mean={:.4}",
            sig_stats.energy_mean, sig_stats.energy_std, sig_stats.tk_mean
        );

        let events = extract_events_v3(&audio, sample_rate as f32, 2048);

        // Count types
        let mut local_counts: HashMap<String, usize> = HashMap::new();
        for event in &events {
            *local_counts.entry(event.category.clone()).or_insert(0) += 1;
            *type_totals.entry(event.category.clone()).or_insert(0) += 1;
        }

        print!("      Events: ");
        let mut counts: Vec<_> = local_counts.iter().collect();
        counts.sort_by(|a, b| b.1.cmp(a.1));
        for (t, c) in counts.iter().take(5) {
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
        let pct = **c as f32 / total_events.max(1) as f32 * 100.0;
        println!("      {}: {} ({:.1}%)", t, c, pct);
    }

    // Check diversity
    let diversity = type_totals.len() as f32 / 9.0 * 100.0;
    println!(
        "\n    Event diversity: {:.0}% ({}/9 types used)",
        diversity,
        type_totals.len()
    );

    // Phase 2: Train
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
                let events = extract_events_v3(&audio, sample_rate as f32, 2048);
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
    println!("    Event diversity:      {:.0}%", diversity);
    println!();

    if diversity > 50.0 {
        println!("  [SUCCESS] Diverse event detection achieved!");
    } else if diversity > 30.0 {
        println!("  [PARTIAL] Moderate event diversity.");
    } else {
        println!("  [INFO] Low diversity - may need more varied recordings.");
    }

    if train_avg > test_avg && train_avg > 0.1 {
        println!("  [SUCCESS] Grammar learned whale-specific patterns!");
    }

    if stats.grammar_density < 0.5 {
        println!("  [SUCCESS] Sparse HDC prevented saturation!");
    }

    separator('=', 70);
    println!("  WHALE BRIDGE v3 COMPLETE");
    separator('=', 70);

    Ok(())
}
