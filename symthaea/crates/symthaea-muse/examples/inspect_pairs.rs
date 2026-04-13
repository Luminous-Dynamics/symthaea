// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Inspect a .pairs.bin file produced by extract_maestro_pairs.
//!
//! ```sh
//! cargo run --release -p symthaea-muse --example inspect_pairs -- \
//!     /opt/datasets/maestro/training_pairs/2004/<some>.pairs.bin
//! ```

use std::path::PathBuf;
use symthaea_muse::training_pairs::load_pairs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: inspect_pairs <path.pairs.bin>");
        std::process::exit(1);
    }
    let path = PathBuf::from(&args[1]);
    let (states, mels) = load_pairs(&path).expect("load");
    println!("Pairs:     {}", states.len());
    println!("State dim: {} (fixed)", 17);
    println!("Mel dim:   {}", mels.first().map(|m| m.len()).unwrap_or(0));

    // Print first pair
    if let (Some(s), Some(m)) = (states.first(), mels.first()) {
        println!("\nFirst state:");
        let names = [
            "consciousness", "arousal", "valence", "dopamine", "serotonin",
            "noradrenaline", "pred_error", "harmony[0]", "harmony[1]",
            "harmony[2]", "harmony[3]", "harmony[4]", "harmony[5]",
            "harmony[6]", "harmony[7]", "time_secs", "mel_dim",
        ];
        for (i, name) in names.iter().enumerate() {
            println!("  {:16} {:+.4}", name, s[i]);
        }
        let min = m.iter().copied().fold(f32::INFINITY, f32::min);
        let max = m.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mean: f32 = m.iter().sum::<f32>() / m.len() as f32;
        println!("\nFirst mel: min={:.3} max={:.3} mean={:.3}", min, max, mean);
    }

    // Sanity: scan a few more
    println!("\nScanning all pairs for NaN/Inf...");
    let mut nan_count = 0;
    for (s, m) in states.iter().zip(mels.iter()) {
        if s.iter().any(|v| !v.is_finite()) || m.iter().any(|v| !v.is_finite()) {
            nan_count += 1;
        }
    }
    println!("  NaN/Inf pairs: {}", nan_count);
}
