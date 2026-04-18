// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compute the honest global mean-baseline MSE over the whole dataset.
//!
//! The per-file mean MSE reported by `train_hdc_decoder` on its probe file is
//! misleading — it's an oracle that knows per-file statistics. The inference-
//! time baseline is: predict the global mean mel frame for every input. The
//! MSE of that prediction equals the dataset-wide mel variance — the fraction
//! of variance the decoder explains is 1 − (val_mse / variance).
//!
//! Single-pass, streaming, memory-bounded (~few MB regardless of dataset size).

use std::path::{Path, PathBuf};
use symthaea_muse::training_pairs::load_pairs;

fn find_bin_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() { out.extend(find_bin_files(&p)); }
            else if p.extension().and_then(|s| s.to_str()) == Some("bin")
                && !p.to_string_lossy().ends_with(".pred.bin") { out.push(p); }
        }
    }
    out.sort();
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: mel_baseline <pairs_dir>");
        std::process::exit(1);
    }
    let dir = PathBuf::from(&args[0]);
    let files = find_bin_files(&dir);
    println!("Scanning {} .pairs.bin files under {}", files.len(), dir.display());

    // Two-pass: first file to get dim, then streaming mean/M2 (Welford).
    let (_, probe_mels) = load_pairs(&files[0]).expect("load first");
    let n_mels = probe_mels[0].len();
    println!("  n_mels: {}", n_mels);

    let mut mean = vec![0.0f64; n_mels];
    let mut m2 = vec![0.0f64; n_mels];
    let mut count: u64 = 0;

    for (idx, path) in files.iter().enumerate() {
        let (_, mels) = match load_pairs(path) {
            Ok(p) => p,
            Err(_) => continue,
        };
        for frame in &mels {
            count += 1;
            let inv = 1.0 / count as f64;
            for j in 0..n_mels {
                let x = frame[j] as f64;
                let delta = x - mean[j];
                mean[j] += delta * inv;
                m2[j] += delta * (x - mean[j]);
            }
        }
        if (idx + 1) % 100 == 0 {
            println!("  [{}/{}] frames={}", idx + 1, files.len(), count);
        }
    }

    // Per-bin variance and aggregates
    let variance: Vec<f64> = m2.iter().map(|&s| s / (count - 1) as f64).collect();
    let mean_variance: f64 = variance.iter().sum::<f64>() / n_mels as f64;
    let total_var: f64 = variance.iter().sum::<f64>();
    let mean_mel: f64 = mean.iter().sum::<f64>() / n_mels as f64;

    println!("\n═══ Global Mel Statistics ═══");
    println!("  total frames:      {}", count);
    println!("  global mean:       {:.4}", mean_mel);
    println!("  mean per-bin var:  {:.4}  ← MSE of 'predict global mean' baseline", mean_variance);
    println!("  total var (sum):   {:.4}", total_var);
    println!("\n  Variance explained by a model with val_mse X:  1 − X / {:.4}", mean_variance);

    // Save the global mean so trainers can subtract it from targets
    use std::io::Write;
    let out = dir.join("global_mel_mean.bin");
    let mut f = std::fs::File::create(&out).unwrap();
    f.write_all(&(n_mels as u32).to_le_bytes()).unwrap();
    for &v in &mean {
        f.write_all(&(v as f32).to_le_bytes()).unwrap();
    }
    println!("\n  Saved global mean → {}", out.display());
}
