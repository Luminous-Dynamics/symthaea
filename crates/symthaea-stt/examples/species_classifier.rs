// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Species-Specific Whale Grammars
//!
//! Train separate grammars for each cetacean species, then use
//! grammar scores as a species classifier.

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

// Reuse event extraction from v4
#[derive(Debug, Clone)]
struct SignalStats {
    energy_p25: f32,
    energy_p75: f32,
    centroid_std: f32,
}

fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f32 * p) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn compute_stats(audio: &[f32], sample_rate: f32, frame_size: usize) -> SignalStats {
    let hop = frame_size / 2;
    let n_fft = frame_size.min(512);
    let mut energies = Vec::new();
    let mut centroids = Vec::new();

    let mut i = 0;
    while i + frame_size <= audio.len() {
        let frame = &audio[i..i + frame_size];
        let energy = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt();
        energies.push(energy);

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
        let total: f32 = spectrum.iter().sum();
        if total > 1e-10 {
            let centroid = spectrum
                .iter()
                .enumerate()
                .map(|(k, &m)| k as f32 * freq_res * m)
                .sum::<f32>()
                / total;
            centroids.push(centroid);
        }
        i += hop;
    }

    let mut sorted = energies.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let centroid_mean = centroids.iter().sum::<f32>() / centroids.len().max(1) as f32;
    let centroid_std = (centroids
        .iter()
        .map(|c| (c - centroid_mean).powi(2))
        .sum::<f32>()
        / centroids.len().max(1) as f32)
        .sqrt();

    SignalStats {
        energy_p25: percentile(&sorted, 0.25),
        energy_p75: percentile(&sorted, 0.75),
        centroid_std,
    }
}

#[derive(Debug, Clone)]
struct FrameFeatures {
    energy: f32,
    tk_energy: f32,
    centroid: f32,
    bandwidth: f32,
    crest_factor: f32,
    flatness: f32,
    low_ratio: f32,
    mid_ratio: f32,
    high_ratio: f32,
    spectral_flux: f32,
}

fn extract_features(
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
        let mut tk = 0.0f32;
        for i in 1..n - 1 {
            tk += (samples[i] * samples[i] - samples[i - 1] * samples[i + 1]).abs();
        }
        (tk / (n - 2) as f32).sqrt()
    } else {
        0.0
    };

    let mut spectrum = vec![0.0f32; n_fft / 2];
    for k in 0..n_fft / 2 {
        let mut real = 0.0f32;
        let mut imag = 0.0f32;
        for (i, &s) in samples.iter().take(n_fft).enumerate() {
            let angle = -2.0 * PI * k as f32 * i as f32 / n_fft as f32;
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

    let geom = spectrum
        .iter()
        .filter(|&&m| m > 1e-10)
        .map(|m| m.ln())
        .sum::<f32>()
        / spectrum.len() as f32;
    let arith = total / spectrum.len() as f32;
    let flatness = if arith > 1e-10 {
        geom.exp() / arith
    } else {
        0.0
    };

    let low_cut = (500.0 / freq_res) as usize;
    let mid_cut = (2000.0 / freq_res) as usize;
    let low_e: f32 = spectrum.iter().take(low_cut.min(spectrum.len())).sum();
    let mid_e: f32 = spectrum
        .iter()
        .skip(low_cut)
        .take((mid_cut - low_cut).max(0))
        .sum();
    let high_e: f32 = spectrum.iter().skip(mid_cut.min(spectrum.len())).sum();

    let (low_ratio, mid_ratio, high_ratio) = if total > 1e-10 {
        (low_e / total, mid_e / total, high_e / total)
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

    (
        FrameFeatures {
            energy,
            tk_energy,
            centroid,
            bandwidth,
            crest_factor,
            flatness,
            low_ratio,
            mid_ratio,
            high_ratio,
            spectral_flux,
        },
        spectrum,
    )
}

struct FmTracker {
    centroids: Vec<f32>,
    window: usize,
}

impl FmTracker {
    fn new(window: usize) -> Self {
        Self {
            centroids: Vec::new(),
            window,
        }
    }

    fn update(&mut self, c: f32) {
        self.centroids.push(c);
        if self.centroids.len() > self.window {
            self.centroids.remove(0);
        }
    }

    fn slope(&self, dt: f32) -> f32 {
        if self.centroids.len() < 2 {
            return 0.0;
        }
        let n = self.centroids.len() as f32;
        let sx: f32 = (0..self.centroids.len()).map(|i| i as f32).sum();
        let sy: f32 = self.centroids.iter().sum();
        let sxy: f32 = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, &y)| i as f32 * y)
            .sum();
        let sx2: f32 = (0..self.centroids.len()).map(|i| (i as f32).powi(2)).sum();
        let denom = n * sx2 - sx * sx;
        if denom.abs() < 1e-10 {
            return 0.0;
        }
        (n * sxy - sx * sy) / denom / dt
    }
}

fn classify(
    f: &FrameFeatures,
    stats: &SignalStats,
    fm: &FmTracker,
    dt: f32,
) -> (&'static str, usize) {
    let silence = stats.energy_p25 * 0.7;
    let loud = stats.energy_p75;
    let tk_ratio = if f.energy > 1e-10 {
        f.tk_energy / f.energy
    } else {
        0.0
    };
    let fm_slope = fm.slope(dt);

    if f.energy < silence {
        return ("silence", 8);
    }

    if f.crest_factor > 4.0 && tk_ratio > 0.3 {
        return if f.energy > loud {
            ("click_loud", 0)
        } else {
            ("click_soft", 1)
        };
    }

    let bw_ratio = if f.centroid > 100.0 {
        f.bandwidth / f.centroid
    } else {
        1.0
    };
    if bw_ratio < 0.4 && f.flatness < 0.3 {
        return if fm_slope > 200.0 {
            ("whistle_up", 2)
        } else if fm_slope < -200.0 {
            ("whistle_down", 3)
        } else {
            ("whistle_flat", 4)
        };
    }

    if f.flatness > 0.4 || f.bandwidth > 800.0 {
        return if f.high_ratio > 0.25 {
            ("burst_rapid", 5)
        } else {
            ("burst_slow", 6)
        };
    }

    if f.low_ratio > 0.4 {
        return ("moan", 7);
    }

    if f.mid_ratio > f.low_ratio && f.mid_ratio > f.high_ratio {
        ("whistle_flat", 4)
    } else if f.high_ratio > f.low_ratio {
        ("burst_slow", 6)
    } else {
        ("moan", 7)
    }
}

fn extract_events(audio: &[f32], sample_rate: f32, frame_size: usize) -> Vec<TemporalEvent> {
    let stats = compute_stats(audio, sample_rate, frame_size);
    let hop = frame_size / 2;
    let dt = hop as f32 / sample_rate;
    let min_dur = 0.015;

    let mut events = Vec::new();
    let mut current: Option<(String, usize, f32, f32)> = None;
    let mut time = 0.0f32;
    let mut prev_spectrum: Vec<f32> = Vec::new();
    let mut fm = FmTracker::new(5);

    let mut i = 0;
    while i + frame_size <= audio.len() {
        let frame = &audio[i..i + frame_size];
        let (features, spectrum) = extract_features(frame, sample_rate, &prev_spectrum);
        fm.update(features.centroid);
        let (etype, cid) = classify(&features, &stats, &fm, dt);
        prev_spectrum = spectrum;

        match &current {
            None => {
                if etype != "silence" {
                    current = Some((etype.to_string(), cid, time, features.energy));
                }
            }
            Some((ct, cc, st, peak)) => {
                if etype != ct {
                    let dur = time - st;
                    if dur >= min_dur && ct != "silence" {
                        events.push(TemporalEvent::new(ct, *cc, *st, dur, *peak));
                    }
                    current = if etype != "silence" {
                        Some((etype.to_string(), cid, time, features.energy))
                    } else {
                        None
                    };
                } else {
                    current = Some((ct.clone(), *cc, *st, peak.max(features.energy)));
                }
            }
        }
        time += dt;
        i += hop;
    }

    if let Some((ct, cc, st, peak)) = current {
        let dur = time - st;
        if dur >= min_dur && ct != "silence" {
            events.push(TemporalEvent::new(&ct, cc, st, dur, peak));
        }
    }
    events
}

fn species_config(name: &str) -> DomainConfig {
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
        name: format!("{}_grammar", name),
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

/// Identify species from filename
fn identify_species(filename: &str) -> &'static str {
    let lower = filename.to_lowercase();
    if lower.contains("beluga") {
        "beluga"
    } else if lower.contains("humpback") {
        "humpback"
    } else if lower.contains("orca") || lower.contains("killer") {
        "orca"
    } else if lower.contains("sperm") {
        "sperm"
    } else if lower.contains("dolphin") {
        "dolphin"
    } else {
        "unknown"
    }
}

fn main() -> std::io::Result<()> {
    header("SPECIES-SPECIFIC WHALE GRAMMARS");

    println!("  Training separate grammars per species for better discrimination.");

    let whale_dir = Path::new("data/whales/sperm");
    let test_dir = Path::new("data/test_animals");

    // Collect all recordings by species
    let mut by_species: HashMap<&str, Vec<(String, Vec<TemporalEvent>)>> = HashMap::new();

    let whale_files: Vec<_> = std::fs::read_dir(whale_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "wav")
                .unwrap_or(false)
        })
        .collect();

    subheader("Phase 1: Extract Events by Species");

    for entry in &whale_files {
        let path = entry.path();
        let filename = path.file_name().unwrap().to_string_lossy().to_string();
        let species = identify_species(&filename);

        if let Ok((audio, sr)) = AudioFrontend::load_wav(&path) {
            let events = extract_events(&audio, sr as f32, 2048);
            println!(
                "    [{}] {}: {} events",
                species.to_uppercase(),
                filename,
                events.len()
            );
            by_species
                .entry(species)
                .or_default()
                .push((filename, events));
        }
    }

    // Also check test_animals
    if test_dir.exists() {
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
            let species = identify_species(&filename);

            if let Ok((audio, sr)) = AudioFrontend::load_wav(&path) {
                let events = extract_events(&audio, sr as f32, 2048);
                println!(
                    "    [{}] {}: {} events",
                    species.to_uppercase(),
                    filename,
                    events.len()
                );
                by_species
                    .entry(species)
                    .or_default()
                    .push((filename, events));
            }
        }
    }

    println!(
        "\n    Species found: {:?}",
        by_species.keys().collect::<Vec<_>>()
    );

    // Phase 2: Train species-specific grammars
    subheader("Phase 2: Train Species-Specific Grammars");

    let mut grammars: HashMap<&str, TemporalGrammar> = HashMap::new();

    for (species, recordings) in &by_species {
        if *species == "unknown" {
            continue;
        }

        let mut grammar = TemporalGrammar::new(species_config(species));

        let mut total_events = 0;
        for (filename, events) in recordings {
            if events.len() >= 3 {
                for _ in 0..25 {
                    grammar.train_sequence(events);
                }
                total_events += events.len();
                println!(
                    "    Trained {} on {} ({} events)",
                    species,
                    filename,
                    events.len()
                );
            }
        }

        let stats = grammar.stats();
        println!(
            "    {} grammar: density={:.3}, trained on {} events\n",
            species, stats.grammar_density, total_events
        );

        grammars.insert(species, grammar);
    }

    // Phase 3: Cross-species classification
    subheader("Phase 3: Species Classification Test");

    println!("    Scoring each recording against all grammars:\n");

    let mut correct = 0;
    let mut total = 0;

    for (true_species, recordings) in &by_species {
        for (filename, events) in recordings {
            if events.len() < 3 {
                continue;
            }

            let mut scores: Vec<(&str, f32)> = Vec::new();

            for (species, grammar) in &mut grammars {
                let score = grammar.score_sequence(events);
                scores.push((species, score));
            }

            scores.sort_by(|a, b| b.1.total_cmp(&a.1));

            let predicted = scores.first().map(|(s, _)| *s).unwrap_or("unknown");
            let is_correct = predicted == *true_species;

            if is_correct {
                correct += 1;
            }
            total += 1;

            let marker = if is_correct { "✓" } else { "✗" };
            print!(
                "    {} [{}] {} → predicted: {} ",
                marker, true_species, filename, predicted
            );
            print!("(");
            for (s, sc) in scores.iter().take(3) {
                print!("{}:{:+.3} ", s, sc);
            }
            println!(")");
        }
    }

    // Summary
    header("CLASSIFICATION RESULTS");

    let accuracy = correct as f32 / total.max(1) as f32 * 100.0;
    println!("    Correct: {}/{}", correct, total);
    println!("    Accuracy: {:.1}%", accuracy);
    println!();

    if accuracy >= 70.0 {
        println!("  [SUCCESS] Species-specific grammars achieve good discrimination!");
    } else if accuracy >= 50.0 {
        println!("  [PARTIAL] Better than random, but needs more training data.");
    } else {
        println!("  [INFO] Need more species-specific recordings for training.");
    }

    separator('=', 70);

    Ok(())
}
