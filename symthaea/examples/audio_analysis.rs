// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Audio Analysis Tool
//!
//! Analyzes the generated WAV files to evaluate synthesis quality:
//! - Waveform statistics (RMS, peak, dynamic range)
//! - Spectral analysis (dominant frequencies)
//! - Quality metrics
//!
//! Run with: cargo run --example audio_analysis

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

fn main() {
    println!("=== Audio Synthesis Quality Analysis ===\n");

    let audio_dir = Path::new("audio_output");

    let files = [
        ("hello_world.wav", "Basic TTS: 'Hello World'"),
        ("vowels.wav", "Vowel sequence: AA EH IY OW UW"),
        ("emotion_calm.wav", "Emotional: Calm pacing"),
        ("emotion_excited.wav", "Emotional: Excited pacing"),
        ("emotion_focused.wav", "Emotional: Focused pacing"),
        ("rap_boom_bap.wav", "Rap: Boom Bap flow (90 BPM)"),
        ("rap_triplet.wav", "Rap: Triplet flow (140 BPM)"),
        ("rap_double_time.wav", "Rap: Double-time flow (100 BPM)"),
    ];

    println!(
        "{:<25} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "File", "Duration", "RMS", "Peak", "Crest", "Dom. Freq"
    );
    println!("{:-<87}", "");

    for (filename, _description) in &files {
        let path = audio_dir.join(filename);
        if path.exists() {
            match analyze_wav(&path) {
                Ok(stats) => {
                    println!(
                        "{:<25} {:>9.2}s {:>10.4} {:>10.4} {:>9.2}dB {:>10.0}Hz",
                        filename,
                        stats.duration,
                        stats.rms,
                        stats.peak,
                        stats.crest_factor_db,
                        stats.dominant_freq
                    );
                }
                Err(e) => {
                    println!("{:<25} Error: {}", filename, e);
                }
            }
        } else {
            println!("{:<25} File not found", filename);
        }
    }

    println!("\n--- Detailed Analysis ---\n");

    for (filename, description) in &files {
        let path = audio_dir.join(filename);
        if path.exists() {
            if let Ok(stats) = analyze_wav(&path) {
                println!("{}:", description);
                println!("  File: {}", filename);
                println!(
                    "  Duration: {:.2}s ({} samples @ {}Hz)",
                    stats.duration, stats.sample_count, stats.sample_rate
                );
                println!(
                    "  Amplitude: RMS={:.4}, Peak={:.4}, Crest={:.2}dB",
                    stats.rms, stats.peak, stats.crest_factor_db
                );
                println!(
                    "  Dynamics: Min={:.4}, Max={:.4}, Range={:.2}dB",
                    stats.min_sample, stats.max_sample, stats.dynamic_range_db
                );
                println!(
                    "  Zero crossings: {} ({:.1}/s = ~{:.0}Hz fundamental)",
                    stats.zero_crossings,
                    stats.zero_crossings as f32 / stats.duration,
                    stats.zero_crossings as f32 / stats.duration / 2.0
                );
                println!(
                    "  Spectral: Dominant freq ~{:.0}Hz, Energy centroid ~{:.0}Hz",
                    stats.dominant_freq, stats.spectral_centroid
                );

                // Quality assessment
                let quality = assess_quality(&stats);
                println!("  Quality: {}", quality);
                println!();
            }
        }
    }

    // Compare emotional variations
    println!("--- Emotional Pacing Comparison ---\n");

    let emotional_files = [
        "emotion_calm.wav",
        "emotion_excited.wav",
        "emotion_focused.wav",
    ];
    let mut emotional_stats: Vec<(&str, AudioStats)> = Vec::new();

    for filename in &emotional_files {
        let path = audio_dir.join(filename);
        if let Ok(stats) = analyze_wav(&path) {
            emotional_stats.push((filename, stats));
        }
    }

    if emotional_stats.len() == 3 {
        println!(
            "{:<20} {:>12} {:>12} {:>12}",
            "Metric", "Calm", "Excited", "Focused"
        );
        println!("{:-<58}", "");

        println!(
            "{:<20} {:>12.4} {:>12.4} {:>12.4}",
            "RMS Energy",
            emotional_stats[0].1.rms,
            emotional_stats[1].1.rms,
            emotional_stats[2].1.rms
        );

        println!(
            "{:<20} {:>12.0} {:>12.0} {:>12.0}",
            "Dom. Freq (Hz)",
            emotional_stats[0].1.dominant_freq,
            emotional_stats[1].1.dominant_freq,
            emotional_stats[2].1.dominant_freq
        );

        println!(
            "{:<20} {:>12.0} {:>12.0} {:>12.0}",
            "Zero Cross/s",
            emotional_stats[0].1.zero_crossings as f32 / emotional_stats[0].1.duration,
            emotional_stats[1].1.zero_crossings as f32 / emotional_stats[1].1.duration,
            emotional_stats[2].1.zero_crossings as f32 / emotional_stats[2].1.duration
        );
    }

    // Compare rap flows
    println!("\n--- Rap Flow Comparison ---\n");

    let rap_files = [
        ("rap_boom_bap.wav", "Boom Bap (90 BPM)"),
        ("rap_triplet.wav", "Triplet (140 BPM)"),
        ("rap_double_time.wav", "Double-time (100 BPM)"),
    ];

    println!(
        "{:<25} {:>10} {:>12} {:>12}",
        "Flow", "Duration", "Density", "Avg Pitch"
    );
    println!("{:-<61}", "");

    for (filename, flow_name) in &rap_files {
        let path = audio_dir.join(filename);
        if let Ok(stats) = analyze_wav(&path) {
            let syllable_density = stats.zero_crossings as f32 / stats.duration / 100.0;
            println!(
                "{:<25} {:>9.2}s {:>11.1}/s {:>11.0}Hz",
                flow_name, stats.duration, syllable_density, stats.dominant_freq
            );
        }
    }

    println!("\n=== Analysis Complete ===");
}

#[derive(Default)]
struct AudioStats {
    sample_rate: u32,
    sample_count: usize,
    duration: f32,
    rms: f32,
    peak: f32,
    min_sample: f32,
    max_sample: f32,
    crest_factor_db: f32,
    dynamic_range_db: f32,
    zero_crossings: usize,
    dominant_freq: f32,
    spectral_centroid: f32,
}

fn analyze_wav(path: &Path) -> Result<AudioStats, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut reader = BufReader::new(file);

    // Read WAV header
    let mut header = [0u8; 44];
    reader.read_exact(&mut header).map_err(|e| e.to_string())?;

    // Parse header
    if &header[0..4] != b"RIFF" || &header[8..12] != b"WAVE" {
        return Err("Invalid WAV file".to_string());
    }

    let sample_rate = u32::from_le_bytes([header[24], header[25], header[26], header[27]]);
    let bits_per_sample = u16::from_le_bytes([header[34], header[35]]);
    let data_size = u32::from_le_bytes([header[40], header[41], header[42], header[43]]);

    if bits_per_sample != 16 {
        return Err(format!("Unsupported bits per sample: {}", bits_per_sample));
    }

    // Read samples
    let num_samples = data_size as usize / 2;
    let mut samples = Vec::with_capacity(num_samples);
    let mut sample_buf = [0u8; 2];

    for _ in 0..num_samples {
        if reader.read_exact(&mut sample_buf).is_err() {
            break;
        }
        let sample = i16::from_le_bytes(sample_buf);
        samples.push(sample as f32 / 32768.0);
    }

    if samples.is_empty() {
        return Err("No samples read".to_string());
    }

    // Calculate statistics
    let duration = samples.len() as f32 / sample_rate as f32;

    // RMS
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    let rms = (sum_sq / samples.len() as f32).sqrt();

    // Peak and min/max
    let max_sample = samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_sample = samples.iter().cloned().fold(f32::INFINITY, f32::min);
    let peak = max_sample.abs().max(min_sample.abs());

    // Crest factor (peak/RMS in dB)
    let crest_factor_db = if rms > 0.0 {
        20.0 * (peak / rms).log10()
    } else {
        0.0
    };

    // Dynamic range
    let dynamic_range_db = if min_sample.abs() > 0.0001 && max_sample.abs() > 0.0001 {
        20.0 * (max_sample.abs() / min_sample.abs().max(0.0001)).log10()
    } else {
        0.0
    };

    // Zero crossings
    let zero_crossings = samples
        .windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count();

    // Simple frequency estimation via autocorrelation
    let dominant_freq = estimate_pitch(&samples, sample_rate);

    // Spectral centroid (simplified)
    let spectral_centroid = estimate_spectral_centroid(&samples, sample_rate);

    Ok(AudioStats {
        sample_rate,
        sample_count: samples.len(),
        duration,
        rms,
        peak,
        min_sample,
        max_sample,
        crest_factor_db,
        dynamic_range_db,
        zero_crossings,
        dominant_freq,
        spectral_centroid,
    })
}

fn estimate_pitch(samples: &[f32], sample_rate: u32) -> f32 {
    // Simple autocorrelation-based pitch estimation
    let max_lag = (sample_rate as usize / 50).min(samples.len() / 2); // Min 50Hz
    let min_lag = sample_rate as usize / 500; // Max 500Hz

    if samples.len() < max_lag * 2 {
        return 0.0;
    }

    let mut best_lag = min_lag;
    let mut best_corr = 0.0f32;

    // Use a portion of the signal
    let analysis_len = (samples.len() / 4).min(4096);
    let start = samples.len() / 4;

    for lag in min_lag..max_lag {
        let mut corr = 0.0f32;
        for i in 0..analysis_len {
            if start + i + lag < samples.len() {
                corr += samples[start + i] * samples[start + i + lag];
            }
        }
        corr /= analysis_len as f32;

        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }

    if best_lag > 0 {
        sample_rate as f32 / best_lag as f32
    } else {
        0.0
    }
}

fn estimate_spectral_centroid(samples: &[f32], sample_rate: u32) -> f32 {
    // Simplified spectral centroid using zero-crossing rate
    // A proper implementation would use FFT
    let zc_rate = samples
        .windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count() as f32
        / samples.len() as f32;

    // ZC rate roughly correlates with spectral centroid
    zc_rate * sample_rate as f32 / 2.0
}

fn assess_quality(stats: &AudioStats) -> &'static str {
    // Simple quality heuristics
    if stats.rms < 0.01 {
        "Very quiet - may need amplitude boost"
    } else if stats.rms > 0.5 {
        "Very loud - may be clipping"
    } else if stats.crest_factor_db < 3.0 {
        "Low dynamic range - sounds compressed"
    } else if stats.crest_factor_db > 20.0 {
        "High dynamic range - may have silence gaps"
    } else if stats.dominant_freq < 80.0 || stats.dominant_freq > 400.0 {
        "Unusual pitch range for speech"
    } else {
        "Good - within expected ranges for speech synthesis"
    }
}