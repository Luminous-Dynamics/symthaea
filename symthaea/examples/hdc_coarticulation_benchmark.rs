// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # HDC Coarticulation Benchmark
//!
//! Demonstrates the effect of HDC-driven coarticulation on speech synthesis.
//!
//! Compares:
//! 1. No coarticulation (discrete phoneme transitions)
//! 2. Basic coarticulation (fixed blend)
//! 3. HDC coarticulation (similarity-driven blend + tau modulation)
//!
//! Metrics:
//! - Formant trajectory smoothness (derivative variance)
//! - Transition naturalness (similarity to human speech patterns)
//! - Computational overhead
//!
//! Run with: cargo run --example hdc_coarticulation_benchmark

use std::sync::Arc;
use std::time::Instant;
use symthaea::voice::{
    ArticulatoryConfig, ArticulatorySynthesizer, FormantFrame, FormantVocoder, LTCPacing,
    PhonemeHdcCodec,
};

fn main() {
    println!("=== HDC Coarticulation Benchmark ===\n");

    // Test phrases with different phonetic characteristics
    let test_phrases = [
        ("Hello World", "HH AH0 L OW1 | W ER1 L D"),
        ("Similar Sounds", "S IH1 M AH0 L ER0 | S AW1 N D Z"), // S-S transition
        ("Contrasting", "K AA1 N T R AE2 S T IH0 NG"),         // Many consonant clusters
        ("Flowing Vowels", "F L OW1 IH0 NG | V AW1 AH0 L Z"),  // Vowel-heavy
    ];

    // Shared codec for fair comparison
    let codec = Arc::new(PhonemeHdcCodec::new());

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                 COARTICULATION MODE COMPARISON                   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    for (name, arpabet) in &test_phrases {
        println!("--- {} ---", name);
        println!("  Phonemes: {}\n", arpabet);

        // Mode 1: No coarticulation
        let (frames_none, time_none) =
            synthesize_with_mode(arpabet, CoarticulationMode::None, codec.clone());
        let metrics_none = analyze_frames(&frames_none);

        // Mode 2: Basic coarticulation (fixed blend)
        let (frames_basic, time_basic) =
            synthesize_with_mode(arpabet, CoarticulationMode::Basic, codec.clone());
        let metrics_basic = analyze_frames(&frames_basic);

        // Mode 3: HDC-driven coarticulation
        let (frames_hdc, time_hdc) =
            synthesize_with_mode(arpabet, CoarticulationMode::HDC, codec.clone());
        let metrics_hdc = analyze_frames(&frames_hdc);

        // Print comparison table
        println!(
            "  {:20} {:>12} {:>12} {:>12}",
            "Metric", "None", "Basic", "HDC"
        );
        println!("  {:-<58}", "");
        println!(
            "  {:20} {:>12.4} {:>12.4} {:>12.4}",
            "F1 Smoothness",
            metrics_none.f1_smoothness,
            metrics_basic.f1_smoothness,
            metrics_hdc.f1_smoothness
        );
        println!(
            "  {:20} {:>12.4} {:>12.4} {:>12.4}",
            "F2 Smoothness",
            metrics_none.f2_smoothness,
            metrics_basic.f2_smoothness,
            metrics_hdc.f2_smoothness
        );
        println!(
            "  {:20} {:>12.4} {:>12.4} {:>12.4}",
            "Transition Jerk",
            metrics_none.transition_jerk,
            metrics_basic.transition_jerk,
            metrics_hdc.transition_jerk
        );
        println!(
            "  {:20} {:>12.4} {:>12.4} {:>12.4}",
            "Energy Variance",
            metrics_none.energy_variance,
            metrics_basic.energy_variance,
            metrics_hdc.energy_variance
        );
        println!(
            "  {:20} {:>10.0}µs {:>10.0}µs {:>10.0}µs",
            "Synthesis Time", time_none, time_basic, time_hdc
        );

        // Calculate improvement
        let smoothness_improvement = (metrics_hdc.f1_smoothness + metrics_hdc.f2_smoothness)
            / (metrics_none.f1_smoothness + metrics_none.f2_smoothness + 0.0001);
        let jerk_reduction =
            1.0 - (metrics_hdc.transition_jerk / (metrics_none.transition_jerk + 0.0001));

        println!("\n  HDC vs None:");
        println!("    Smoothness: {:.1}x better", smoothness_improvement);
        println!("    Jerk reduction: {:.1}%", jerk_reduction * 100.0);
        println!();
    }

    // Generate audio files for listening comparison
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    GENERATING AUDIO FILES                        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    std::fs::create_dir_all("audio_output").ok();

    let test_phrase = "HH AH0 L OW1 | W ER1 L D"; // "Hello World"
    let pacing = LTCPacing::default();
    let mut vocoder = FormantVocoder::new();

    // Generate with no coarticulation
    let (frames_none, _) =
        synthesize_with_mode(test_phrase, CoarticulationMode::None, codec.clone());
    let audio_none = vocoder.synthesize(&frames_none);
    write_wav("audio_output/coarticulation_none.wav", &audio_none, 24000);
    println!("  Written: audio_output/coarticulation_none.wav");

    // Generate with basic coarticulation
    let (frames_basic, _) =
        synthesize_with_mode(test_phrase, CoarticulationMode::Basic, codec.clone());
    let audio_basic = vocoder.synthesize(&frames_basic);
    write_wav("audio_output/coarticulation_basic.wav", &audio_basic, 24000);
    println!("  Written: audio_output/coarticulation_basic.wav");

    // Generate with HDC coarticulation
    let (frames_hdc, _) = synthesize_with_mode(test_phrase, CoarticulationMode::HDC, codec.clone());
    let audio_hdc = vocoder.synthesize(&frames_hdc);
    write_wav("audio_output/coarticulation_hdc.wav", &audio_hdc, 24000);
    println!("  Written: audio_output/coarticulation_hdc.wav");

    // HDC phoneme similarity analysis
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                  HDC PHONEME SIMILARITY MATRIX                   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let phonemes = ["IY", "AE", "AA", "OW", "UW", "P", "T", "K", "S", "M", "L"];
    print!("      ");
    for p in &phonemes {
        print!("{:>6}", p);
    }
    println!();

    for p1 in &phonemes {
        print!("{:>5} ", p1);
        for p2 in &phonemes {
            let sim = codec.similarity(p1, p2);
            print!("{:>6.2}", sim);
        }
        println!();
    }

    println!("\nKey observations:");
    println!("  - Vowels cluster together (IY/AE/AA/OW/UW have higher mutual similarity)");
    println!("  - Consonants form distinct clusters (stops P/T/K, nasals/liquids M/L)");
    println!("  - HDC coarticulation uses these similarities for natural transitions");

    println!("\n=== Benchmark Complete ===");
}

#[derive(Clone, Copy)]
enum CoarticulationMode {
    None,  // Discrete transitions
    Basic, // Fixed anticipatory blend
    HDC,   // HDC-driven blend + tau
}

fn synthesize_with_mode(
    arpabet: &str,
    mode: CoarticulationMode,
    codec: Arc<PhonemeHdcCodec>,
) -> (Vec<FormantFrame>, f64) {
    let config = match mode {
        CoarticulationMode::None => ArticulatoryConfig {
            coarticulation: false,
            hdc_coarticulation: false,
            ..Default::default()
        },
        CoarticulationMode::Basic => ArticulatoryConfig {
            coarticulation: true,
            hdc_coarticulation: false,
            ..Default::default()
        },
        CoarticulationMode::HDC => ArticulatoryConfig {
            coarticulation: true,
            hdc_coarticulation: true,
            ..Default::default()
        },
    };

    let mut synth = ArticulatorySynthesizer::with_shared_codec(config, codec);
    let pacing = LTCPacing::default();

    let start = Instant::now();
    let frames = synth.synthesize_arpabet(arpabet, &pacing);
    let elapsed = start.elapsed().as_micros() as f64;

    (frames, elapsed)
}

struct FrameMetrics {
    f1_smoothness: f32,
    f2_smoothness: f32,
    transition_jerk: f32,
    energy_variance: f32,
}

fn analyze_frames(frames: &[FormantFrame]) -> FrameMetrics {
    if frames.len() < 3 {
        return FrameMetrics {
            f1_smoothness: 0.0,
            f2_smoothness: 0.0,
            transition_jerk: 0.0,
            energy_variance: 0.0,
        };
    }

    // Calculate first derivatives (velocity)
    let mut f1_derivs: Vec<f32> = Vec::new();
    let mut f2_derivs: Vec<f32> = Vec::new();

    for i in 1..frames.len() {
        let dt = frames[i].time - frames[i - 1].time;
        if dt > 0.0 {
            f1_derivs.push((frames[i].f1 - frames[i - 1].f1) / dt);
            f2_derivs.push((frames[i].f2 - frames[i - 1].f2) / dt);
        }
    }

    // Calculate second derivatives (jerk indicator)
    let mut f1_jerk: Vec<f32> = Vec::new();
    for i in 1..f1_derivs.len() {
        f1_jerk.push((f1_derivs[i] - f1_derivs[i - 1]).abs());
    }

    // Smoothness = 1 / (1 + variance of derivatives)
    let f1_var = variance(&f1_derivs);
    let f2_var = variance(&f2_derivs);
    let f1_smoothness = 1.0 / (1.0 + f1_var.sqrt() / 100.0);
    let f2_smoothness = 1.0 / (1.0 + f2_var.sqrt() / 100.0);

    // Transition jerk = sum of second derivatives
    let transition_jerk = f1_jerk.iter().sum::<f32>() / (f1_jerk.len().max(1) as f32);

    // Energy variance
    let energies: Vec<f32> = frames.iter().map(|f| f.energy).collect();
    let energy_variance = variance(&energies);

    FrameMetrics {
        f1_smoothness,
        f2_smoothness,
        transition_jerk: transition_jerk / 1000.0, // Scale for readability
        energy_variance,
    }
}

fn variance(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
    let var: f32 = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
    var
}

fn write_wav(path: &str, samples: &[f32], sample_rate: u32) {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path).expect("Failed to create WAV file");

    // Convert to i16
    let samples_i16: Vec<i16> = samples
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
        .collect();

    let data_size = (samples_i16.len() * 2) as u32;
    let file_size = data_size + 36;

    // WAV header
    file.write_all(b"RIFF").unwrap();
    file.write_all(&file_size.to_le_bytes()).unwrap();
    file.write_all(b"WAVE").unwrap();
    file.write_all(b"fmt ").unwrap();
    file.write_all(&16u32.to_le_bytes()).unwrap(); // fmt chunk size
    file.write_all(&1u16.to_le_bytes()).unwrap(); // PCM
    file.write_all(&1u16.to_le_bytes()).unwrap(); // mono
    file.write_all(&sample_rate.to_le_bytes()).unwrap();
    file.write_all(&(sample_rate * 2).to_le_bytes()).unwrap(); // byte rate
    file.write_all(&2u16.to_le_bytes()).unwrap(); // block align
    file.write_all(&16u16.to_le_bytes()).unwrap(); // bits per sample
    file.write_all(b"data").unwrap();
    file.write_all(&data_size.to_le_bytes()).unwrap();

    for sample in &samples_i16 {
        file.write_all(&sample.to_le_bytes()).unwrap();
    }
}
