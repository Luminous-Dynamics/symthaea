// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # LibriSpeech HDC Audio Encoding Benchmark
//!
//! Tests Symthaea's HDC encoding capability on speech features.
//! Since LibriSpeech requires large downloads (~60GB), this benchmark uses
//! synthetic MFCC-like features to validate the HDC encoding pipeline
//! for audio classification tasks.
//!
//! ## Method
//! 1. Generate synthetic MFCC feature vectors (13 coefficients per frame)
//! 2. Encode temporal sequences via HDC bind(position, level)
//! 3. Classify speakers or phonemes using prototype matching
//!
//! ## Run
//! ```bash
//! cargo run --example benchmark_librispeech_hdc --release
//! ```

use std::time::Instant;

use symthaea::hdc::unified_hv::ContinuousHV;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       LibriSpeech HDC Audio Encoding Benchmark             ║");
    println!("║       Temporal HDC for Speech Classification               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let dim = 4096;
    let n_mfcc = 13;
    let n_levels = 32;
    let n_speakers = 10;
    let frames_per_utterance = 100; // ~1 second at 100 fps
    let train_per_speaker = 50;
    let test_per_speaker = 20;

    // Pre-generate codebook
    let level_hvs: Vec<ContinuousHV> = (0..n_levels)
        .map(|l| ContinuousHV::random(dim, 90000 + l as u64))
        .collect();
    let coeff_hvs: Vec<ContinuousHV> = (0..n_mfcc)
        .map(|c| ContinuousHV::random(dim, 91000 + c as u64))
        .collect();
    let frame_hvs: Vec<ContinuousHV> = (0..frames_per_utterance)
        .map(|f| ContinuousHV::random(dim, 92000 + f as u64))
        .collect();

    // ═══════════════════════════════════════════════════════════════
    // Test 1: Speaker Identification
    // ═══════════════════════════════════════════════════════════════
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 1: Speaker Identification ({} speakers)", n_speakers);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let t = Instant::now();

    // Generate speaker-specific utterances and train prototypes
    let mut prototypes: Vec<Vec<f32>> = (0..n_speakers).map(|_| vec![0.0f32; dim]).collect();
    let mut counts = vec![0usize; n_speakers];

    for speaker in 0..n_speakers {
        for utt in 0..train_per_speaker {
            let mfccs = generate_speaker_mfccs(speaker, frames_per_utterance, n_mfcc, utt as u64);
            let encoded =
                encode_utterance(&mfccs, &level_hvs, &coeff_hvs, &frame_hvs, n_levels, dim);

            for (acc, &val) in prototypes[speaker].iter_mut().zip(encoded.values.iter()) {
                *acc += val;
            }
            counts[speaker] += 1;
        }
    }

    // Normalize prototypes
    let speaker_prototypes: Vec<ContinuousHV> = prototypes
        .iter()
        .map(|proto| {
            let norm: f32 = proto.iter().map(|x| x * x).sum::<f32>().sqrt();
            let normed: Vec<f32> = if norm > 0.0 {
                proto.iter().map(|x| x / norm).collect()
            } else {
                proto.clone()
            };
            ContinuousHV::from_vec(normed)
        })
        .collect();

    let train_time = t.elapsed().as_secs_f64();
    println!(
        "  Training: {} utterances in {:.1}s",
        n_speakers * train_per_speaker,
        train_time
    );

    // Test
    let t = Instant::now();
    let mut correct = 0;
    let total = n_speakers * test_per_speaker;

    for speaker in 0..n_speakers {
        for utt in 0..test_per_speaker {
            let seed = 1000 + utt as u64;
            let mfccs = generate_speaker_mfccs(speaker, frames_per_utterance, n_mfcc, seed);
            let encoded =
                encode_utterance(&mfccs, &level_hvs, &coeff_hvs, &frame_hvs, n_levels, dim);

            let predicted = speaker_prototypes
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    encoded
                        .similarity(a)
                        .partial_cmp(&encoded.similarity(b))
                        .unwrap()
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            if predicted == speaker {
                correct += 1;
            }
        }
    }

    let test_time = t.elapsed().as_secs_f64();
    let accuracy = correct as f64 / total as f64;
    println!(
        "  Testing: {:.2}% ({}/{}) in {:.1}s",
        accuracy * 100.0,
        correct,
        total,
        test_time
    );

    // ═══════════════════════════════════════════════════════════════
    // Test 2: Encoding Quality - Temporal vs Non-temporal
    // ═══════════════════════════════════════════════════════════════
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 2: Temporal vs Bag-of-frames Encoding");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Encode same utterance with and without temporal position
    let test_mfccs = generate_speaker_mfccs(0, frames_per_utterance, n_mfcc, 42);

    let temporal_enc = encode_utterance(
        &test_mfccs,
        &level_hvs,
        &coeff_hvs,
        &frame_hvs,
        n_levels,
        dim,
    );

    // Bag-of-frames: same but without position binding
    let bag_enc = encode_utterance_bag(&test_mfccs, &level_hvs, &coeff_hvs, n_levels, dim);

    // Reversed utterance
    let mut reversed_mfccs = test_mfccs.clone();
    reversed_mfccs.reverse();
    let reversed_enc = encode_utterance(
        &reversed_mfccs,
        &level_hvs,
        &coeff_hvs,
        &frame_hvs,
        n_levels,
        dim,
    );
    let reversed_bag = encode_utterance_bag(&reversed_mfccs, &level_hvs, &coeff_hvs, n_levels, dim);

    let temporal_rev_sim = temporal_enc.similarity(&reversed_enc);
    let bag_rev_sim = bag_enc.similarity(&reversed_bag);

    println!(
        "  Temporal: forward↔reversed similarity = {:.4}",
        temporal_rev_sim
    );
    println!(
        "  Bag-of-frames: forward↔reversed similarity = {:.4}",
        bag_rev_sim
    );
    println!(
        "  Temporal encodes order: {}",
        if temporal_rev_sim < bag_rev_sim {
            "YES"
        } else {
            "NO"
        }
    );

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                 VALIDATION SUMMARY                         ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    let checks = vec![
        ("Speaker ID accuracy > 50%", accuracy > 0.50),
        ("Speaker ID accuracy > 70%", accuracy > 0.70),
        (
            "Temporal encoding preserves order",
            temporal_rev_sim < bag_rev_sim,
        ),
    ];

    let mut passed = 0;
    for (name, pass) in &checks {
        println!("║  {} {:50}   ║", if *pass { "PASS" } else { "FAIL" }, name);
        if *pass {
            passed += 1;
        }
    }
    println!("╟──────────────────────────────────────────────────────────────╢");
    println!(
        "║  Result: {}/{} tests passed                                 ║",
        passed,
        checks.len()
    );
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Save
    let result_json = serde_json::json!({
        "benchmark": "LibriSpeech HDC Audio Encoding (Synthetic)",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "speaker_accuracy": accuracy,
        "n_speakers": n_speakers,
        "temporal_order_preserved": temporal_rev_sim < bag_rev_sim,
        "tests_passed": passed,
        "tests_total": checks.len(),
    });

    std::fs::create_dir_all("data/benchmarks/librispeech").ok();
    if let Ok(f) = std::fs::File::create("data/benchmarks/librispeech/results.json") {
        serde_json::to_writer_pretty(f, &result_json).ok();
        println!("Results saved to data/benchmarks/librispeech/results.json");
    }
}

/// Generate synthetic MFCC-like features for a speaker
fn generate_speaker_mfccs(
    speaker: usize,
    n_frames: usize,
    n_mfcc: usize,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(speaker as u64 * 12345 + 1);
    let mut rand = || -> f32 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0
    };

    // Speaker-specific base MFCC values (formant structure)
    let base: Vec<f32> = (0..n_mfcc)
        .map(|c| (speaker as f32 * 0.3 + c as f32 * 0.1).sin() * 0.5)
        .collect();

    (0..n_frames)
        .map(|_| (0..n_mfcc).map(|c| base[c] + rand() * 0.2).collect())
        .collect()
}

/// Encode an utterance (sequence of MFCC frames) into a single HDC vector
/// Uses temporal binding: bind(frame_pos, bind(coeff_pos, level))
fn encode_utterance(
    mfccs: &[Vec<f32>],
    level_hvs: &[ContinuousHV],
    coeff_hvs: &[ContinuousHV],
    frame_hvs: &[ContinuousHV],
    n_levels: usize,
    dim: usize,
) -> ContinuousHV {
    let mut acc = vec![0.0f32; dim];

    for (frame_idx, frame) in mfccs.iter().enumerate() {
        if frame_idx >= frame_hvs.len() {
            break;
        }
        for (coeff_idx, &val) in frame.iter().enumerate() {
            if coeff_idx >= coeff_hvs.len() {
                break;
            }
            let level = ((val + 1.0) / 2.0 * n_levels as f32)
                .max(0.0)
                .min((n_levels - 1) as f32) as usize;
            let bound = frame_hvs[frame_idx]
                .bind(&coeff_hvs[coeff_idx])
                .bind(&level_hvs[level]);
            for (a, &v) in acc.iter_mut().zip(bound.values.iter()) {
                *a += v;
            }
        }
    }

    let norm: f32 = acc.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut acc {
            *v /= norm;
        }
    }
    ContinuousHV::from_vec(acc)
}

/// Bag-of-frames encoding (no temporal position)
fn encode_utterance_bag(
    mfccs: &[Vec<f32>],
    level_hvs: &[ContinuousHV],
    coeff_hvs: &[ContinuousHV],
    n_levels: usize,
    dim: usize,
) -> ContinuousHV {
    let mut acc = vec![0.0f32; dim];

    for frame in mfccs.iter() {
        for (coeff_idx, &val) in frame.iter().enumerate() {
            if coeff_idx >= coeff_hvs.len() {
                break;
            }
            let level = ((val + 1.0) / 2.0 * n_levels as f32)
                .max(0.0)
                .min((n_levels - 1) as f32) as usize;
            let bound = coeff_hvs[coeff_idx].bind(&level_hvs[level]);
            for (a, &v) in acc.iter_mut().zip(bound.values.iter()) {
                *a += v;
            }
        }
    }

    let norm: f32 = acc.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut acc {
            *v /= norm;
        }
    }
    ContinuousHV::from_vec(acc)
}