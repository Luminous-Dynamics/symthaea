// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compares the hand-crafted 24-band proxy FAD against real CLAP-embedding
//! FAD on the same generated audio, to sanity-check that the real metric
//! agrees with the proxy on the easy case (self vs. self ~= 0, calm vs.
//! intense clearly > 0) before trusting it for anything subtler.
//!
//! Run: cargo run --release -p symthaea-muse --features clap-fad --example clap_fad_compare
//! (downloads the ~117MB CLAP ONNX model on first run, cached by hf-hub;
//! set CLAP_AUDIO_MODEL_PATH to reuse an already-downloaded copy)

use symthaea_muse::creative_bench::FadScore;
use symthaea_muse::{AudioData, MuseConfig, MusicalState, compose};

const SAMPLE_RATE: u32 = symthaea_muse::clap_mel::SAMPLE_RATE;

fn render_set(state: &MusicalState, seeds: std::ops::Range<u64>) -> Vec<Vec<[f32; 2]>> {
    let config = MuseConfig {
        duration_secs: 2.0,
        max_notes: 12,
        sample_rate: SAMPLE_RATE,
        ..Default::default()
    };
    seeds
        .filter_map(|seed| {
            let comp = compose(&config, state, seed);
            match comp.audio {
                AudioData::StereoF32(s) => Some(s),
                _ => None,
            }
        })
        .collect()
}

fn main() {
    println!("Rendering calm set (5 seeds)...");
    let calm = MusicalState {
        arousal: 0.2,
        valence: 0.5,
        ..Default::default()
    };
    let calm_set = render_set(&calm, 0..5);

    println!("Rendering intense set (5 seeds)...");
    let intense = MusicalState {
        arousal: 0.9,
        valence: -0.5,
        dopamine: 0.8,
        ..Default::default()
    };
    let intense_set = render_set(&intense, 10..15);

    println!("\n=== Proxy FAD (24-band spectral heuristic) ===");
    let proxy_self = FadScore::compute(&calm_set, &calm_set, SAMPLE_RATE);
    let proxy_cross = FadScore::compute(&calm_set, &intense_set, SAMPLE_RATE);
    println!("  calm vs calm (self):     {:.3}", proxy_self.fad);
    println!("  calm vs intense (cross): {:.3}", proxy_cross.fad);

    println!("\n=== Real FAD (CLAP audio-tower embedding) ===");
    println!("Loading CLAP audio tower (downloads on first run)...");
    match FadScore::compute_with_clap(&calm_set, &calm_set, SAMPLE_RATE) {
        Ok(clap_self) => {
            let clap_cross = FadScore::compute_with_clap(&calm_set, &intense_set, SAMPLE_RATE)
                .expect("cross-set CLAP FAD");
            println!("  calm vs calm (self):     {:.3}", clap_self.fad);
            println!("  calm vs intense (cross): {:.3}", clap_cross.fad);

            println!("\n=== Agreement check ===");
            let proxy_discriminates = proxy_cross.fad > proxy_self.fad;
            let clap_discriminates = clap_cross.fad > clap_self.fad;
            println!("  proxy: cross > self? {proxy_discriminates}");
            println!("  clap:  cross > self? {clap_discriminates}");
            if proxy_discriminates && clap_discriminates {
                println!("  Both metrics agree calm and intense sets are distinguishable.");
            } else {
                println!("  DISAGREEMENT — investigate before trusting either metric further.");
            }
        }
        Err(e) => {
            eprintln!("CLAP FAD failed: {e}");
            eprintln!(
                "(Network access or model download issue — proxy FAD numbers above are still valid.)"
            );
        }
    }
}
