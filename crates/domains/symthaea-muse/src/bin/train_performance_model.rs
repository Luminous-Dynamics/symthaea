// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fit the `expressive::ExpressiveModel` on MAESTRO v3 performances.
//!
//! Learns melodic-line VELOCITY DEVIATION (accent texture around the local
//! rolling mean) and ARTICULATION (sounded duration / inter-onset interval)
//! from real virtuoso playing — the two expressive dimensions that survive
//! performance-capture tempo drift without beat tracking (see
//! `expressive.rs`'s module docs for why grid-based micro-timing is out of
//! scope).
//!
//! Same discipline as `train_melody_predictor`: streaming normal equations,
//! deterministic file-level split (FNV bucket 0 of 10 = test), closed-form
//! ridge solve, HONEST held-out metrics against baselines, weights +
//! provenance written as one JSON artifact.
//!
//! Usage:
//! ```bash
//! cargo run --release -p symthaea-muse --bin train_performance_model -- \
//!     /opt/datasets/maestro/maestro-v3.0.0 \
//!     crates/domains/symthaea-muse/data/performance_model_weights.json
//! ```

use std::path::{Path, PathBuf};
use symthaea_muse::expressive::{N_FEATURES, PerformancePair, extract_pairs};
use symthaea_muse::midi_trainer::parse_midi;

const DIM: usize = N_FEATURES + 1; // + bias

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let midi_root = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("/opt/datasets/maestro/maestro-v3.0.0");
    let out_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("crates/domains/symthaea-muse/data/performance_model_weights.json");

    println!("=== ExpressiveModel trainer (velocity deviation + articulation) ===");
    println!("MIDI root: {midi_root}");

    let mut files = Vec::new();
    collect_midi_files(Path::new(midi_root), &mut files);
    files.sort();
    if files.is_empty() {
        eprintln!("No MIDI files found under {midi_root}");
        std::process::exit(1);
    }
    println!("Found {} MIDI files", files.len());

    let mut xtx = vec![[0.0f64; DIM]; DIM];
    let mut xty_vel = [0.0f64; DIM];
    let mut xty_art = [0.0f64; DIM];
    let mut test_pairs: Vec<PerformancePair> = Vec::new();
    let (mut n_train_files, mut n_test_files) = (0usize, 0usize);
    let (mut n_train_pairs, mut n_skipped) = (0usize, 0usize);

    for (i, path) in files.iter().enumerate() {
        let melody = match parse_midi(path) {
            Ok(m) => m,
            Err(_) => {
                n_skipped += 1;
                continue;
            }
        };
        let pairs = extract_pairs(&melody);
        if pairs.is_empty() {
            n_skipped += 1;
            continue;
        }
        if file_hash(path) % 10 == 0 {
            n_test_files += 1;
            test_pairs.extend(pairs);
        } else {
            n_train_files += 1;
            for p in &pairs {
                let x = features_with_bias(p);
                for r in 0..DIM {
                    for c in r..DIM {
                        xtx[r][c] += x[r] * x[c];
                    }
                    xty_vel[r] += x[r] * p.velocity_dev as f64;
                    xty_art[r] += x[r] * p.articulation as f64;
                }
                n_train_pairs += 1;
            }
        }
        if (i + 1) % 100 == 0 {
            println!("  parsed {}/{} files...", i + 1, files.len());
        }
    }
    for r in 0..DIM {
        for c in 0..r {
            xtx[r][c] = xtx[c][r];
        }
    }
    println!(
        "Train: {n_train_files} files / {n_train_pairs} pairs. \
         Test: {n_test_files} files / {} pairs. Skipped: {n_skipped}.",
        test_pairs.len()
    );
    if n_train_pairs < 10_000 || test_pairs.len() < 1_000 {
        eprintln!("Not enough data for a trustworthy fit — aborting.");
        std::process::exit(1);
    }

    let ridge = 1e-3 * n_train_pairs as f64;
    let w_vel = solve_ridge(&xtx, &xty_vel, ridge);
    let w_art = solve_ridge(&xtx, &xty_art, ridge);

    // Held-out evaluation vs honest baselines:
    // - velocity_dev baseline: predict 0 (i.e. "play the local mean") — MAE
    //   equals the mean |dev| in the data; the model must beat it.
    // - articulation baseline: predict the train-set mean articulation.
    let art_mean_train = xty_art[DIM - 1] / n_train_pairs as f64; // bias row = Σ target
    let (mut vel_mae_model, mut vel_mae_zero) = (0.0f64, 0.0f64);
    let (mut art_mae_model, mut art_mae_mean) = (0.0f64, 0.0f64);
    let mut vel_dir_hits = 0usize;
    let mut vel_dir_total = 0usize;
    for p in &test_pairs {
        let x = features_with_bias(p);
        let pred_vel: f64 = (0..DIM).map(|k| w_vel[k] as f64 * x[k]).sum();
        let pred_art: f64 = (0..DIM).map(|k| w_art[k] as f64 * x[k]).sum();
        vel_mae_model += (pred_vel - p.velocity_dev as f64).abs();
        vel_mae_zero += (p.velocity_dev as f64).abs();
        art_mae_model += (pred_art - p.articulation as f64).abs();
        art_mae_mean += (art_mean_train - p.articulation as f64).abs();
        // Accent-direction accuracy on clearly-accented notes (|dev|>0.05):
        // does the model at least know WHICH WAY the player leans?
        if p.velocity_dev.abs() > 0.05 {
            vel_dir_total += 1;
            if (pred_vel > 0.0) == (p.velocity_dev > 0.0) {
                vel_dir_hits += 1;
            }
        }
    }
    let n = test_pairs.len() as f64;
    let (vel_mae_model, vel_mae_zero) = (vel_mae_model / n, vel_mae_zero / n);
    let (art_mae_model, art_mae_mean) = (art_mae_model / n, art_mae_mean / n);
    let dir_acc = vel_dir_hits as f64 / vel_dir_total.max(1) as f64;

    println!("\n── Held-out evaluation ({} pairs) ──", test_pairs.len());
    println!("  velocity-dev MAE:  model {vel_mae_model:.4}  vs  zero-baseline {vel_mae_zero:.4}");
    println!("  accent direction:  {dir_acc:.3} on {vel_dir_total} clearly-accented notes");
    println!("  articulation MAE:  model {art_mae_model:.4}  vs  mean-baseline {art_mae_mean:.4}");
    if vel_mae_model >= vel_mae_zero && art_mae_model >= art_mae_mean {
        eprintln!(
            "\nModel beats NEITHER baseline — refusing to write weights that \
             would ship a no-better-than-nothing model."
        );
        std::process::exit(1);
    }

    let json = serde_json::json!({
        "w_velocity": w_vel,
        "w_articulation": w_art,
        "provenance": {
            "dataset": midi_root,
            "trained": "ridge least squares (closed form), lambda = 1e-3 * n",
            "split": "file-level FNV hash bucket 0/10 = test; no augmentation",
            "train_files": n_train_files,
            "train_pairs": n_train_pairs,
            "test_files": n_test_files,
            "test_pairs": test_pairs.len(),
            "targets": "velocity deviation from ±8-note local mean; articulation = duration/IOI (grid-free — see expressive.rs docs for why onset micro-timing is out of scope)",
            "held_out_metrics": {
                "velocity_dev_mae_model": vel_mae_model,
                "velocity_dev_mae_zero_baseline": vel_mae_zero,
                "accent_direction_accuracy": dir_acc,
                "accent_direction_n": vel_dir_total,
                "articulation_mae_model": art_mae_model,
                "articulation_mae_mean_baseline": art_mae_mean,
            },
        },
    });
    if let Some(parent) = Path::new(out_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::write(out_path, serde_json::to_string_pretty(&json).unwrap()) {
        Ok(()) => println!("\nWrote {out_path}"),
        Err(e) => {
            eprintln!("Failed to write {out_path}: {e}");
            std::process::exit(1);
        }
    }
    println!(
        "Rebuild symthaea-muse so expressive.rs's include_str! picks up the new weights,\n\
         then re-run `cargo test -p symthaea-muse --lib expressive`."
    );
}

fn features_with_bias(p: &PerformancePair) -> [f64; DIM] {
    let mut x = [0.0f64; DIM];
    for (i, &f) in p.features.iter().enumerate() {
        x[i] = f as f64;
    }
    x[DIM - 1] = 1.0;
    x
}

fn collect_midi_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_midi_files(&path, out);
        } else if path
            .extension()
            .map(|e| e == "mid" || e == "midi")
            .unwrap_or(false)
        {
            out.push(path);
        }
    }
}

/// Deterministic per-file FNV-1a hash (same scheme as train_melody_predictor,
/// so the two models share the identical train/test file partition — no
/// cross-model leakage if they're ever evaluated jointly).
fn file_hash(path: &Path) -> u64 {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let mut h: u64 = 0xcbf29ce484222325;
    for b in name.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn solve_ridge(xtx: &[[f64; DIM]], xty: &[f64; DIM], ridge: f64) -> [f32; DIM] {
    let mut a = vec![[0.0f64; DIM + 1]; DIM];
    for r in 0..DIM {
        for c in 0..DIM {
            a[r][c] = xtx[r][c] + if r == c { ridge } else { 0.0 };
        }
        a[r][DIM] = xty[r];
    }
    for col in 0..DIM {
        let pivot = (col..DIM)
            .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
            .unwrap();
        a.swap(col, pivot);
        let pv = a[col][col];
        assert!(pv.abs() > 1e-12, "singular normal equations at col {col}");
        for r in 0..DIM {
            if r != col {
                let factor = a[r][col] / pv;
                for c in col..=DIM {
                    a[r][c] -= factor * a[col][c];
                }
            }
        }
    }
    let mut w = [0.0f32; DIM];
    for r in 0..DIM {
        w[r] = (a[r][DIM] / a[r][r]) as f32;
    }
    w
}
