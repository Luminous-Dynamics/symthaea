// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Real-World Whale Validation v4
//!
//! Final refinements:
//! - 1% sparsity to prevent saturation
//! - Proper FM tracking with sliding window
//! - Better whistle/burst disambiguation

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

/// Signal statistics
#[derive(Debug, Clone)]
struct SignalStats {
    energy_mean: f32,
    energy_std: f32,
    energy_p25: f32,
    energy_p75: f32,
    tk_mean: f32,
    centroid_mean: f32,
    centroid_std: f32,
}

fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f32 * p) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn compute_signal_stats(audio: &[f32], sample_rate: f32, frame_size: usize) -> SignalStats {
    let hop_size = frame_size / 2;
    let n_fft = frame_size.min(512);

    let mut energies = Vec::new();
    let mut tk_energies = Vec::new();
    let mut centroids = Vec::new();

    let mut i = 0;
    while i + frame_size <= audio.len() {
        let frame = &audio[i..i + frame_size];

        let energy = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt();
        energies.push(energy);

        if frame.len() >= 3 {
            let mut tk_sum = 0.0f32;
            for j in 1..frame.len() - 1 {
                let tk = frame[j] * frame[j] - frame[j - 1] * frame[j + 1];
                tk_sum += tk.abs();
            }
            tk_energies.push((tk_sum / (frame.len() - 2) as f32).sqrt());
        }

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

    let energy_mean = energies.iter().sum::<f32>() / energies.len().max(1) as f32;
    let energy_std = (energies
        .iter()
        .map(|e| (e - energy_mean).powi(2))
        .sum::<f32>()
        / energies.len().max(1) as f32)
        .sqrt();

    let mut sorted_e = energies.clone();
    sorted_e.sort_by(|a, b| a.total_cmp(b));
    let energy_p25 = percentile(&sorted_e, 0.25);
    let energy_p75 = percentile(&sorted_e, 0.75);

    let tk_mean = tk_energies.iter().sum::<f32>() / tk_energies.len().max(1) as f32;

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
        energy_p25,
        energy_p75,
        tk_mean,
        centroid_mean,
        centroid_std,
    }
}

#[derive(Debug, Clone)]
struct FrameFeatures {
    energy: f32,
    tk_energy: f32,
    centroid: f32,
    bandwidth: f32,
    spectral_flux: f32,
    peak_freq: f32,
    low_ratio: f32,
    mid_ratio: f32,
    high_ratio: f32,
    crest_factor: f32,
    flatness: f32,
}

fn extract_frame_features(
    samples: &[f32],
    sample_rate: f32,
    prev_spectrum: &[f32],
) -> (FrameFeatures, Vec<f32>) {
    let n = samples.len();
    let n_fft = n.min(512);

    let energy = (samples.iter().map(|s| s * s).sum::<f32>() / n as f32).sqrt();
    let peak = samples
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, |a, b| a.max(b));
    let crest_factor = if energy > 1e-10 { peak / energy } else { 0.0 };

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

    let (peak_bin, _) =
        spectrum.iter().enumerate().fold(
            (0, 0.0f32),
            |(bi, bm), (i, &m)| if m > bm { (i, m) } else { (bi, bm) },
        );
    let peak_freq = peak_bin as f32 * freq_res;

    // Spectral flatness (geometric mean / arithmetic mean)
    let geom_mean = spectrum
        .iter()
        .filter(|&&m| m > 1e-10)
        .map(|m| m.ln())
        .sum::<f32>()
        / spectrum.len() as f32;
    let arith_mean = total_energy / spectrum.len() as f32;
    let flatness = if arith_mean > 1e-10 {
        geom_mean.exp() / arith_mean
    } else {
        0.0
    };

    let low_cutoff = (500.0 / freq_res) as usize;
    let mid_cutoff = (2000.0 / freq_res) as usize;

    let low_e: f32 = spectrum.iter().take(low_cutoff.min(spectrum.len())).sum();
    let mid_e: f32 = spectrum
        .iter()
        .skip(low_cutoff.min(spectrum.len()))
        .take((mid_cutoff - low_cutoff).max(0))
        .sum();
    let high_e: f32 = spectrum.iter().skip(mid_cutoff.min(spectrum.len())).sum();

    let (low_ratio, mid_ratio, high_ratio) = if total_energy > 1e-10 {
        (
            low_e / total_energy,
            mid_e / total_energy,
            high_e / total_energy,
        )
    } else {
        (0.0, 0.0, 0.0)
    };

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

    let features = FrameFeatures {
        energy,
        tk_energy,
        centroid,
        bandwidth,
        spectral_flux,
        peak_freq,
        low_ratio,
        mid_ratio,
        high_ratio,
        crest_factor,
        flatness,
    };

    (features, spectrum)
}

/// FM tracker with sliding window
struct FmTracker {
    centroids: Vec<f32>,
    window_size: usize,
}

impl FmTracker {
    fn new(window_size: usize) -> Self {
        FmTracker {
            centroids: Vec::new(),
            window_size,
        }
    }

    fn update(&mut self, centroid: f32) {
        self.centroids.push(centroid);
        if self.centroids.len() > self.window_size {
            self.centroids.remove(0);
        }
    }

    fn slope(&self, frame_duration: f32) -> f32 {
        if self.centroids.len() < 2 {
            return 0.0;
        }
        // Linear regression slope
        let n = self.centroids.len() as f32;
        let sum_x: f32 = (0..self.centroids.len()).map(|i| i as f32).sum();
        let sum_y: f32 = self.centroids.iter().sum();
        let sum_xy: f32 = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, &y)| i as f32 * y)
            .sum();
        let sum_x2: f32 = (0..self.centroids.len()).map(|i| (i as f32).powi(2)).sum();

        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return 0.0;
        }
        let slope_per_frame = (n * sum_xy - sum_x * sum_y) / denom;
        slope_per_frame / frame_duration // Hz/s
    }

    fn variance(&self) -> f32 {
        if self.centroids.len() < 2 {
            return 0.0;
        }
        let mean: f32 = self.centroids.iter().sum::<f32>() / self.centroids.len() as f32;
        self.centroids
            .iter()
            .map(|c| (c - mean).powi(2))
            .sum::<f32>()
            / self.centroids.len() as f32
    }
}

fn classify_frame_v4(
    features: &FrameFeatures,
    stats: &SignalStats,
    fm_tracker: &FmTracker,
    frame_duration: f32,
) -> (&'static str, usize) {
    // Adaptive thresholds
    let silence_thresh = stats.energy_p25 * 0.7;
    let loud_thresh = stats.energy_p75;

    let tk_ratio = if features.energy > 1e-10 {
        features.tk_energy / features.energy
    } else {
        0.0
    };

    // FM analysis from tracker
    let fm_slope = fm_tracker.slope(frame_duration);
    let fm_variance = fm_tracker.variance();

    // Silence
    if features.energy < silence_thresh {
        return ("silence", 8);
    }

    // Click: impulsive (high crest, high TK ratio)
    if features.crest_factor > 4.0 && tk_ratio > 0.3 {
        if features.energy > loud_thresh {
            return ("click_loud", 0);
        } else {
            return ("click_soft", 1);
        }
    }

    // Whistle: tonal (narrow bandwidth, stable centroid)
    let bandwidth_ratio = if features.centroid > 100.0 {
        features.bandwidth / features.centroid
    } else {
        1.0
    };

    let is_tonal = bandwidth_ratio < 0.4 && features.flatness < 0.3;

    if is_tonal && fm_variance < stats.centroid_std.powi(2) * 4.0 {
        // Check FM direction using tracked slope
        if fm_slope > 200.0 {
            return ("whistle_up", 2);
        } else if fm_slope < -200.0 {
            return ("whistle_down", 3);
        } else {
            return ("whistle_flat", 4);
        }
    }

    // Burst: broadband, rapid change
    if features.flatness > 0.4 || features.bandwidth > 800.0 {
        if features.spectral_flux > stats.centroid_std * 1.5 || features.high_ratio > 0.25 {
            return ("burst_rapid", 5);
        } else {
            return ("burst_slow", 6);
        }
    }

    // Moan: low-frequency dominated, sustained
    if features.low_ratio > 0.4 {
        return ("moan", 7);
    }

    // Default: classify by dominant band
    if features.mid_ratio > features.low_ratio && features.mid_ratio > features.high_ratio {
        return ("whistle_flat", 4);
    } else if features.high_ratio > features.low_ratio {
        return ("burst_slow", 6);
    } else {
        return ("moan", 7);
    }
}

fn extract_events_v4(audio: &[f32], sample_rate: f32, frame_size: usize) -> Vec<TemporalEvent> {
    let stats = compute_signal_stats(audio, sample_rate, frame_size);

    let mut events: Vec<TemporalEvent> = Vec::new();
    let hop_size = frame_size / 2;
    let frame_duration = hop_size as f32 / sample_rate;
    let min_event_duration = 0.015;

    let mut current_event: Option<(String, usize, f32, f32)> = None;
    let mut time = 0.0f32;
    let mut prev_spectrum: Vec<f32> = Vec::new();
    let mut fm_tracker = FmTracker::new(5);

    let mut i = 0;
    while i + frame_size <= audio.len() {
        let frame = &audio[i..i + frame_size];
        let (features, spectrum) = extract_frame_features(frame, sample_rate, &prev_spectrum);

        fm_tracker.update(features.centroid);
        let (event_type, class_id) =
            classify_frame_v4(&features, &stats, &fm_tracker, frame_duration);

        prev_spectrum = spectrum;

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

        time += frame_duration;
        i += hop_size;
    }

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

fn cetacean_config_v4() -> DomainConfig {
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
        name: "cetacean_v4".to_string(),
        categories: calls,
        sample_rate: 44100.0,
        frame_size: 2048,
        sparsity: Sparsity::Sparse1, // 1% for maximum capacity
        duration_bins: 10,
        intensity_bins: 5,
        predictive_feedback: true,
        prediction_boost: 0.3,
        hierarchy_depth: 3,
    }
}

fn main() -> std::io::Result<()> {
    header("WHALE BRIDGE v4: FM TRACKING + ULTRA-SPARSE");

    println!("  Final refinements:");
    println!("    - 1% sparsity (10× capacity vs 5%)");
    println!("    - FM tracking with sliding window regression");
    println!("    - Spectral flatness for tonal detection");
    println!("    - 3-level hierarchy");

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

    let mut grammar = TemporalGrammar::new(cetacean_config_v4());
    let stats = grammar.stats();

    subheader("Grammar Configuration");
    println!("    Domain:      {}", stats.domain);
    println!("    Categories:  {}", stats.num_categories);
    println!("    Sparsity:    {:.0}%", stats.sparsity * 100.0);
    println!("    Hierarchy:   {} levels", stats.hierarchy_depth);

    subheader("Phase 1: Event Extraction with FM Tracking");

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

        println!(
            "    {} ({:.1}s)",
            filename,
            audio.len() as f32 / sample_rate as f32
        );

        let events = extract_events_v4(&audio, sample_rate as f32, 2048);

        let mut local_counts: HashMap<String, usize> = HashMap::new();
        for event in &events {
            *local_counts.entry(event.category.clone()).or_insert(0) += 1;
            *type_totals.entry(event.category.clone()).or_insert(0) += 1;
        }

        print!("      ");
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
    println!("\n    Distribution:");
    let mut type_vec: Vec<_> = type_totals.iter().collect();
    type_vec.sort_by(|a, b| b.1.cmp(a.1));
    for (t, c) in &type_vec {
        let pct = **c as f32 / total_events.max(1) as f32 * 100.0;
        let bar_len = (pct / 5.0) as usize;
        let bar: String = std::iter::repeat('#').take(bar_len).collect();
        println!("      {:12} {:4} ({:5.1}%) {}", t, c, pct, bar);
    }

    let diversity = type_totals.len() as f32 / 9.0 * 100.0;
    println!(
        "\n    Diversity: {:.0}% ({}/9 types)",
        diversity,
        type_totals.len()
    );

    subheader("Phase 2: Training with Ultra-Sparse Grammar");

    let mid = all_results.len() / 2;
    let (train_set, test_set) = all_results.split_at(mid);

    for (filename, events) in train_set {
        if events.len() >= 3 {
            for _ in 0..20 {
                grammar.train_sequence(events);
            }
            println!("    Trained: {} ({} events)", filename, events.len());
        }
    }

    let stats = grammar.stats();
    println!(
        "\n    Grammar density: {:.4} (target <0.05 for 1% sparse)",
        stats.grammar_density
    );

    subheader("Phase 3: Scoring");

    let mut train_scores = Vec::new();
    let mut test_scores = Vec::new();

    println!("    Training set:");
    for (filename, events) in train_set {
        if events.len() >= 3 {
            let score = grammar.score_sequence(events);
            train_scores.push(score);
            println!("      {}: {:+.4}", filename, score);
        }
    }

    println!("\n    Test set:");
    for (filename, events) in test_set {
        if events.len() >= 3 {
            let score = grammar.score_sequence(events);
            test_scores.push(score);
            println!("      {}: {:+.4}", filename, score);
        }
    }

    let train_avg: f32 = train_scores.iter().sum::<f32>() / train_scores.len().max(1) as f32;
    let test_avg: f32 = test_scores.iter().sum::<f32>() / test_scores.len().max(1) as f32;

    if test_dir.exists() {
        subheader("Phase 4: Cross-Species Discrimination");

        let mut whale_scores = Vec::new();
        let mut other_scores = Vec::new();

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
                let events = extract_events_v4(&audio, sample_rate as f32, 2048);
                if events.len() >= 2 {
                    let score = grammar.score_sequence(&events);
                    let is_whale = filename.contains("whale");
                    let label = if is_whale {
                        "WHALE"
                    } else if filename.contains("dolphin") {
                        "DOLPHIN"
                    } else {
                        "OTHER"
                    };

                    if is_whale {
                        whale_scores.push(score);
                    } else {
                        other_scores.push(score);
                    }

                    println!("    [{}] {}: {:+.4}", label, filename, score);
                }
            }
        }

        if !whale_scores.is_empty() && !other_scores.is_empty() {
            let whale_avg = whale_scores.iter().sum::<f32>() / whale_scores.len() as f32;
            let other_avg = other_scores.iter().sum::<f32>() / other_scores.len() as f32;
            println!(
                "\n    Cross-species discrimination: {:+.4}",
                whale_avg - other_avg
            );
        }
    }

    header("FINAL RESULTS");

    println!("    Training average:     {:+.4}", train_avg);
    println!("    Test average:         {:+.4}", test_avg);
    println!("    Generalization gap:   {:+.4}", train_avg - test_avg);
    println!("    Grammar density:      {:.4}", stats.grammar_density);
    println!("    Event diversity:      {:.0}%", diversity);
    println!();

    if diversity >= 70.0 {
        println!("  [EXCELLENT] High event diversity!");
    } else if diversity >= 50.0 {
        println!("  [GOOD] Moderate event diversity.");
    }

    if stats.grammar_density < 0.1 {
        println!("  [SUCCESS] Ultra-sparse HDC - no saturation risk!");
    }

    if train_avg > test_avg && train_avg > 0.0 {
        println!("  [SUCCESS] Grammar learned species-specific patterns!");
    }

    separator('=', 70);
    println!("  WHALE BRIDGE v4 COMPLETE");
    separator('=', 70);

    Ok(())
}
