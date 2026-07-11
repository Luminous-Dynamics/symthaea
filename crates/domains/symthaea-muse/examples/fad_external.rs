// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! First externally-grounded FAD evaluation of Symthaea's music.
//!
//! `FadScore` had only ever been fed Symthaea output as both generated AND
//! reference sets (self-vs-self), so it had never measured distance to real
//! music. This harness uses MAESTRO v3.0.0 performance audio (real virtuoso
//! piano recordings) as the reference distribution.
//!
//! To make the single FAD number interpretable, two anchors bracket it:
//! - NOISE FLOOR: MAESTRO half-A vs MAESTRO half-B (same distribution —
//!   the best any generator could possibly score against this reference)
//! - CEILING: white noise vs MAESTRO (maximally unmusical audio)
//!
//! Honest scope: the embedding is the crate's 24-band pseudo-MFCC, not
//! VGGish/CLAP, so absolute values are not comparable to published FAD
//! numbers. Within this embedding, the floor/ceiling anchors give the muse
//! score meaning. Also note the reference is solo piano — timbre distance
//! from Symthaea's additive synth is expected and part of what FAD sees.
//!
//! Usage:
//! ```bash
//! cargo run --release -p symthaea-muse --example fad_external -- \
//!     /opt/datasets/maestro/maestro-v3.0.0
//! ```

use std::path::{Path, PathBuf};
use symthaea_muse::creative_bench::FadScore;
use symthaea_muse::{AudioData, MuseConfig, MusicalState, compose};

const SAMPLE_RATE: u32 = 44100;
const EXCERPT_SECS: usize = 8;
const SKIP_SECS: usize = 30; // skip into each piece (past the opening)
const N_REFERENCE: usize = 60;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let maestro_root = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("/opt/datasets/maestro/maestro-v3.0.0");

    println!("=== FAD vs real music (MAESTRO reference) ===\n");

    // 1. Reference set: excerpts from real performances
    let mut wavs = Vec::new();
    collect_wavs(Path::new(maestro_root), &mut wavs);
    wavs.sort();
    if wavs.len() < N_REFERENCE {
        eprintln!(
            "Need at least {N_REFERENCE} WAVs under {maestro_root}, found {}",
            wavs.len()
        );
        std::process::exit(1);
    }
    // Spread picks across the corpus rather than taking one year's block
    let stride = wavs.len() / N_REFERENCE;
    let mut reference: Vec<Vec<[f32; 2]>> = Vec::new();
    for i in 0..N_REFERENCE {
        if let Some(excerpt) = load_excerpt(&wavs[i * stride]) {
            reference.push(excerpt);
        }
    }
    println!(
        "Loaded {} reference excerpts ({EXCERPT_SECS}s each) from MAESTRO",
        reference.len()
    );

    // 2. Generated set: compositions across the consciousness grid
    let mut generated: Vec<Vec<[f32; 2]>> = Vec::new();
    let config = MuseConfig {
        duration_secs: EXCERPT_SECS as f32,
        max_notes: 32,
        ..Default::default()
    };
    let mut seed = 1u64;
    for &psi in &[0.2f32, 0.5, 0.8] {
        for &valence in &[-0.5f32, 0.0, 0.5] {
            for &arousal in &[0.2f32, 0.5, 0.8] {
                let state = MusicalState {
                    consciousness_level: psi,
                    valence,
                    arousal,
                    ..Default::default()
                };
                let comp = compose(&config, &state, seed);
                seed += 1;
                if let AudioData::StereoF32(frames) = comp.audio {
                    generated.push(frames);
                }
            }
        }
    }
    println!(
        "Generated {} Symthaea compositions across the V/A/psi grid\n",
        generated.len()
    );

    // 3. Anchors
    let half = reference.len() / 2;
    let (ref_a, ref_b) = reference.split_at(half);
    let floor = FadScore::compute(ref_a, ref_b, SAMPLE_RATE);

    let noise: Vec<Vec<[f32; 2]>> = (0..generated.len())
        .map(|i| white_noise_excerpt(0x9e3779b9u64.wrapping_mul(i as u64 + 1)))
        .collect();
    let ceiling = FadScore::compute(&noise, &reference, SAMPLE_RATE);

    // 4. The measurement
    let muse = FadScore::compute(&generated, &reference, SAMPLE_RATE);

    println!("── FAD against MAESTRO reference (24-band pseudo-MFCC embedding) ──");
    println!("  noise floor (MAESTRO vs MAESTRO): {:>10.3}", floor.fad);
    println!("  Symthaea compositions:            {:>10.3}", muse.fad);
    println!("  ceiling (white noise):            {:>10.3}", ceiling.fad);
    if ceiling.fad > floor.fad {
        let position = (muse.fad - floor.fad) / (ceiling.fad - floor.fad);
        println!(
            "\n  Normalized position: {position:.3}  (0.0 = indistinguishable from \
             real piano within\n  this embedding, 1.0 = as far from real music as white noise)"
        );
    }
    println!(
        "\n  n_generated={} n_reference={} — record these numbers with the date;\n  \
         they are the baseline future generator work must beat.",
        muse.n_generated, muse.n_reference
    );
}

fn collect_wavs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_wavs(&path, out);
        } else if path.extension().map(|e| e == "wav").unwrap_or(false) {
            out.push(path);
        }
    }
}

/// Load an EXCERPT_SECS stereo excerpt starting SKIP_SECS into the file.
fn load_excerpt(path: &Path) -> Option<Vec<[f32; 2]>> {
    let mut reader = hound::WavReader::open(path).ok()?;
    let spec = reader.spec();
    if spec.channels != 2 || spec.sample_rate != SAMPLE_RATE || spec.bits_per_sample != 16 {
        eprintln!(
            "  skip {} (format {}ch/{}Hz/{}bit)",
            path.file_name().unwrap_or_default().to_string_lossy(),
            spec.channels,
            spec.sample_rate,
            spec.bits_per_sample
        );
        return None;
    }
    let skip = (SKIP_SECS as u32) * SAMPLE_RATE;
    let take = (EXCERPT_SECS as u32) * SAMPLE_RATE;
    if reader.duration() < skip + take {
        return None; // piece too short
    }
    reader.seek(skip).ok()?;
    let mut frames = Vec::with_capacity(take as usize);
    let mut samples = reader.samples::<i16>();
    for _ in 0..take {
        let l = samples.next()?.ok()? as f32 / 32768.0;
        let r = samples.next()?.ok()? as f32 / 32768.0;
        frames.push([l, r]);
    }
    Some(frames)
}

/// Deterministic white-noise excerpt at typical music RMS (~-20 dBFS).
fn white_noise_excerpt(mut state: u64) -> Vec<[f32; 2]> {
    let n = EXCERPT_SECS * SAMPLE_RATE as usize;
    let mut frames = Vec::with_capacity(n);
    let mut next = || {
        // xorshift64*
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let v = state.wrapping_mul(0x2545F4914F6CDD1D);
        ((v >> 40) as f32 / 8388608.0 - 1.0) * 0.1
    };
    for _ in 0..n {
        frames.push([next(), next()]);
    }
    frames
}
