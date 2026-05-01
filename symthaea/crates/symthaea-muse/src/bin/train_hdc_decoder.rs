// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Baseline trainer for the HDC mel decoder.
//!
//! Streams MAESTRO `(state, mel_frame)` pairs from disk, trains a tiny
//! pure-Rust MLP (state → hidden → mel), reports loss periodically, and
//! saves checkpoints. This is the Phase 2 baseline — a concrete artifact
//! that proves the end-to-end data flow before investing in candle/GPU.
//!
//! ```sh
//! cargo run --release -p symthaea-muse --bin train_hdc_decoder -- \
//!     /opt/datasets/maestro/training_pairs \
//!     /opt/datasets/maestro/checkpoints/baseline.bin \
//!     --epochs 1 --hidden 256 --lr 1e-3 --max-files 32
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;
use symthaea_muse::hdc_mel_decoder::{DecoderConfig, MelDecoder};
use symthaea_muse::training_pairs::load_pairs;

/// Normalize a 17-field state vector for training.
///
/// The raw save layout puts two awkward values in fields 15 and 16:
///   - field 15: time_secs (0..~600s, unbounded)
///   - field 16: mel_dim as f32 (constant 128, pure noise for the model)
///
/// We clip, rescale, and zero them so the network sees only meaningful inputs.
fn normalize_state(raw: &[f32; 17]) -> [f32; 17] {
    let mut s = *raw;
    // Clip time_secs to [0, 600] then scale to [0, 1]
    s[15] = (s[15].clamp(0.0, 600.0)) / 600.0;
    // mel_dim is constant per dataset — zero it out (and keep field count stable)
    s[16] = 0.0;
    s
}

struct Args {
    pairs_dir: PathBuf,
    ckpt_path: PathBuf,
    epochs: usize,
    hidden: usize,
    lr: f32,
    max_files: Option<usize>,
    log_every: usize,
    /// Half-window for temporal context. 0 = no context (17D input),
    /// 2 = 5-frame window (85D input), etc.
    context: usize,
    adam: bool,
    /// Path to global_mel_mean.bin (from mel_baseline). If set, targets are
    /// centered by subtracting the mean — the decoder learns deviations.
    global_mean: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("Usage: train_hdc_decoder <pairs_dir> <ckpt_path> [--epochs N] [--hidden N] [--lr F] [--max-files N]");
        std::process::exit(1);
    }
    let pairs_dir = PathBuf::from(args.remove(0));
    let ckpt_path = PathBuf::from(args.remove(0));

    let mut epochs = 1;
    let mut hidden = 256;
    let mut lr = 1e-3;
    let mut max_files = None;
    let mut log_every = 50_000;
    let mut context = 0usize;
    let mut adam = false;
    let mut global_mean: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--epochs" => {
                epochs = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--hidden" => {
                hidden = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--lr" => {
                lr = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--max-files" => {
                max_files = Some(args[i + 1].parse().unwrap());
                i += 2;
            }
            "--log-every" => {
                log_every = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--context" => {
                context = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--adam" => {
                adam = true;
                i += 1;
            }
            "--global-mean" => {
                global_mean = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            other => {
                eprintln!("Unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }

    Args {
        pairs_dir,
        ckpt_path,
        epochs,
        hidden,
        lr,
        max_files,
        log_every,
        context,
        adam,
        global_mean,
    }
}

/// Load a `global_mel_mean.bin` file: u32 n_mels, then n_mels f32s.
fn load_global_mean(path: &std::path::Path) -> std::io::Result<Vec<f32>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut h = [0u8; 4];
    f.read_exact(&mut h)?;
    let n = u32::from_le_bytes(h) as usize;
    let mut buf = vec![0u8; n * 4];
    f.read_exact(&mut buf)?;
    Ok(buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

/// Build a temporal-context input vector: concatenate normalized states
/// from [t - ctx ..= t + ctx]. Frames outside the file clamp to the nearest
/// valid frame (replicate-boundary, standard for time-series).
fn build_context_input(states: &[[f32; 17]], t: usize, ctx: usize) -> Vec<f32> {
    let window = 2 * ctx + 1;
    let mut out = Vec::with_capacity(window * 17);
    for k in 0..window {
        let idx =
            (t as isize + k as isize - ctx as isize).clamp(0, states.len() as isize - 1) as usize;
        out.extend_from_slice(&normalize_state(&states[idx]));
    }
    out
}

fn find_bin_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                out.extend(find_bin_files(&p));
            } else if p.extension().and_then(|s| s.to_str()) == Some("bin") {
                let name = p.to_string_lossy();
                if name.ends_with(".pred.bin") || name.ends_with("global_mel_mean.bin") {
                    continue;
                }
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

fn main() {
    let args = parse_args();
    let files = {
        let mut f = find_bin_files(&args.pairs_dir);
        if let Some(n) = args.max_files {
            f.truncate(n);
        }
        f
    };
    if files.is_empty() {
        eprintln!(
            "No .pairs.bin files found under {}",
            args.pairs_dir.display()
        );
        std::process::exit(1);
    }

    println!("═══ HDC Mel Decoder — Baseline Trainer ═══");
    println!("  pairs_dir:  {}", args.pairs_dir.display());
    println!("  checkpoint: {}", args.ckpt_path.display());
    println!("  files:      {}", files.len());
    println!("  epochs:     {}", args.epochs);
    println!("  hidden:     {}", args.hidden);
    println!("  lr:         {}", args.lr);

    // Random 90/10 train/val split (deterministic from seed).
    // Chronological split biases val toward the newest year's recording gear,
    // which produces misleading val_mse dominated by dataset shift.
    let mut split_seed = 0xC0DE_u64;
    let mut shuffled_files = files.clone();
    for i in (1..shuffled_files.len()).rev() {
        split_seed ^= split_seed << 13;
        split_seed ^= split_seed >> 7;
        split_seed ^= split_seed << 17;
        let j = (split_seed as usize) % (i + 1);
        shuffled_files.swap(i, j);
    }
    let split = (shuffled_files.len() * 9 / 10).max(1);
    let train_files: Vec<PathBuf> = shuffled_files[..split].to_vec();
    let val_files: Vec<PathBuf> = shuffled_files[split..].to_vec();
    println!("  train files: {}", train_files.len());
    println!("  val files:   {}", val_files.len());

    // Probe the first file to get mel_dim
    let (probe_states, probe_mels) = load_pairs(&train_files[0]).expect("load first file");
    let n_mels = probe_mels[0].len();
    let base_state_dim = probe_states[0].len();
    let context_window = 2 * args.context + 1;
    let state_dim = base_state_dim * context_window;
    println!(
        "  base_state: {}  context_half: {}  window: {}",
        base_state_dim, args.context, context_window
    );
    println!("  state_dim:  {}", state_dim);
    println!("  n_mels:     {}", n_mels);

    let cfg = DecoderConfig {
        state_dim,
        hidden: args.hidden,
        n_mels,
        lr: args.lr,
        seed: 0xBADC0FFEE,
    };
    let mut decoder = MelDecoder::new(cfg);
    if args.adam {
        decoder.enable_adam();
        println!("  optimizer:  Adam");
    } else {
        println!("  optimizer:  SGD");
    }

    // Optional mean-centering
    let mean_vec: Option<Vec<f32>> = args.global_mean.as_ref().map(|p| {
        let v = load_global_mean(p).expect("load global mean");
        assert_eq!(v.len(), n_mels, "global mean n_mels mismatch");
        println!("  centering:  global_mel_mean from {}", p.display());
        v
    });

    // Simple xorshift for shuffle indices
    let mut seed = 0xA11CE_u64;

    let t_start = Instant::now();
    let mut global_step = 0usize;
    let mut window_loss = 0.0f32;
    let mut window_count = 0usize;

    // Precompute mel mean as the trivial baseline to beat
    let mut mel_sum = vec![0.0f32; n_mels];
    let mut total_frames = 0usize;
    for (_, mel) in probe_states.iter().zip(probe_mels.iter()) {
        for j in 0..n_mels {
            mel_sum[j] += mel[j];
        }
        total_frames += 1;
    }
    let mel_mean: Vec<f32> = mel_sum.iter().map(|&s| s / total_frames as f32).collect();
    let mut mean_mse = 0.0;
    for mel in &probe_mels {
        for j in 0..n_mels {
            let d = mel[j] - mel_mean[j];
            mean_mse += d * d;
        }
    }
    mean_mse /= (probe_mels.len() * n_mels) as f32;
    println!("  mean-baseline MSE (probe file): {:.4}", mean_mse);
    println!();

    for epoch in 0..args.epochs {
        for (file_idx, path) in train_files.iter().enumerate() {
            let (states, mels) = match load_pairs(path) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("  SKIP {}: {}", path.display(), e);
                    continue;
                }
            };

            // Shuffle indices (Fisher-Yates with xorshift)
            let mut idx: Vec<usize> = (0..states.len()).collect();
            for i in (1..idx.len()).rev() {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let j = (seed as usize) % (i + 1);
                idx.swap(i, j);
            }

            for &i in &idx {
                let input = build_context_input(&states, i, args.context);
                let target: Vec<f32> = if let Some(mu) = &mean_vec {
                    mels[i].iter().zip(mu).map(|(x, m)| x - m).collect()
                } else {
                    mels[i].clone()
                };
                let loss = decoder.step(&input, &target);
                window_loss += loss;
                window_count += 1;
                global_step += 1;

                if global_step % args.log_every == 0 {
                    let avg = window_loss / window_count as f32;
                    let rate = global_step as f64 / t_start.elapsed().as_secs_f64();
                    println!(
                        "  [epoch {}] step {:>10} file {:>4}/{:<4}  mse={:.4}  ({:.0} samples/s)",
                        epoch,
                        global_step,
                        file_idx + 1,
                        train_files.len(),
                        avg,
                        rate
                    );
                    window_loss = 0.0;
                    window_count = 0;
                }
            }
        }

        // Validation pass
        let mut val_loss = 0.0f64;
        let mut val_count = 0usize;
        for vpath in &val_files {
            if let Ok((vs, vm)) = load_pairs(vpath) {
                for (i, m) in vm.iter().enumerate() {
                    let input = build_context_input(&vs, i, args.context);
                    let pred = decoder.predict(&input);
                    // Add mean back for apples-to-apples comparison with
                    // uncentered val_mse.
                    let mut e = 0.0;
                    for j in 0..n_mels {
                        let pred_full = if let Some(mu) = &mean_vec {
                            pred[j] + mu[j]
                        } else {
                            pred[j]
                        };
                        let d = pred_full - m[j];
                        e += d * d;
                    }
                    val_loss += (e / n_mels as f32) as f64;
                    val_count += 1;
                }
            }
        }
        let val_mse = if val_count > 0 {
            val_loss / val_count as f64
        } else {
            0.0
        };

        // Save checkpoint at end of each epoch
        if let Some(parent) = args.ckpt_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        decoder.save(&args.ckpt_path).expect("save checkpoint");
        // R² vs the 2.1538 global-mean baseline (hard-coded; true variance
        // comes from mel_baseline). This gives a dimensionless quality score.
        const GLOBAL_VAR: f64 = 2.1538;
        let r2 = 1.0 - val_mse / GLOBAL_VAR;
        println!(
            "  [epoch {} DONE] val_mse={:.4}  R²={:.3}  ({} frames)  checkpoint → {}",
            epoch,
            val_mse,
            r2,
            val_count,
            args.ckpt_path.display()
        );
    }

    let elapsed = t_start.elapsed();
    println!("\n═══ Training Complete ═══");
    println!("  total steps: {}", global_step);
    println!("  elapsed:     {:.1}s", elapsed.as_secs_f64());
    println!(
        "  throughput:  {:.0} samples/s",
        global_step as f64 / elapsed.as_secs_f64()
    );
    println!("  checkpoint:  {}", args.ckpt_path.display());
}
