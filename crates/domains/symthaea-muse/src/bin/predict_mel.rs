// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Predict mel frames for a .pairs.bin file using a trained decoder checkpoint.
//!
//! Consumes a .pairs.bin (ground truth) + a checkpoint, emits per-frame MSE,
//! per-mel-bin RMSE, and saves predicted mel frames next to the source for
//! qualitative inspection.
//!
//! ```sh
//! cargo run --release -p symthaea-muse --bin predict_mel -- \
//!     /opt/datasets/maestro/checkpoints/baseline_full.bin \
//!     /opt/datasets/maestro/training_pairs/2004/<some>.pairs.bin
//! ```

use std::path::PathBuf;
use symthaea_muse::hdc_mel_decoder::MelDecoder;
use symthaea_muse::training_pairs::load_pairs;

fn normalize_state(raw: &[f32; 17]) -> [f32; 17] {
    let mut s = *raw;
    s[15] = (s[15].clamp(0.0, 600.0)) / 600.0;
    s[16] = 0.0;
    s
}

/// Build the same temporal context input the trainer uses.
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("Usage: predict_mel <checkpoint> <pairs.bin> [--out path.pred.bin]");
        std::process::exit(1);
    }
    let ckpt_path = PathBuf::from(&args[0]);
    let pairs_path = PathBuf::from(&args[1]);
    let out_path = if args.len() >= 4 && args[2] == "--out" {
        PathBuf::from(&args[3])
    } else {
        pairs_path.with_extension("pred.bin")
    };

    let decoder = MelDecoder::load(&ckpt_path).expect("load checkpoint");
    // Infer the context half-window from state_dim / 17.
    let ctx = (decoder.cfg.state_dim / 17).saturating_sub(1) / 2;
    println!("Checkpoint: {}", ckpt_path.display());
    println!(
        "  state_dim: {}  (ctx half-window = {})",
        decoder.cfg.state_dim, ctx
    );
    println!("  hidden:    {}", decoder.cfg.hidden);
    println!("  n_mels:    {}", decoder.cfg.n_mels);

    let (states, mels) = load_pairs(&pairs_path).expect("load pairs");
    println!("\nGround truth: {}", pairs_path.display());
    println!("  pairs: {}", states.len());

    let n_mels = decoder.cfg.n_mels;
    let mut per_frame_mse = Vec::with_capacity(states.len());
    let mut per_bin_sse = vec![0.0f64; n_mels];
    let mut preds: Vec<Vec<f32>> = Vec::with_capacity(states.len());

    for (i, m) in mels.iter().enumerate() {
        let input = build_context_input(&states, i, ctx);
        let pred = decoder.predict(&input);
        let mut frame_mse = 0.0f32;
        for j in 0..n_mels {
            let d = pred[j] - m[j];
            frame_mse += d * d;
            per_bin_sse[j] += (d * d) as f64;
        }
        per_frame_mse.push(frame_mse / n_mels as f32);
        preds.push(pred);
    }

    let total: f64 = per_frame_mse.iter().map(|&x| x as f64).sum();
    let avg = total / per_frame_mse.len() as f64;
    let max = per_frame_mse.iter().copied().fold(0.0f32, f32::max);
    let min = per_frame_mse.iter().copied().fold(f32::INFINITY, f32::min);

    println!("\nFrame MSE:");
    println!("  mean: {:.4}", avg);
    println!("  min:  {:.4}", min);
    println!("  max:  {:.4}", max);

    println!("\nPer-mel-bin RMSE (first 16 bins):");
    for j in 0..16.min(n_mels) {
        let rmse = (per_bin_sse[j] / per_frame_mse.len() as f64).sqrt();
        println!("  bin[{:3}]: {:.4}", j, rmse);
    }

    // Save predictions in same binary format: u32 count, u32 n_mels, then frames.
    use std::io::Write;
    let mut f = std::fs::File::create(&out_path).expect("create out");
    f.write_all(&(preds.len() as u32).to_le_bytes()).unwrap();
    f.write_all(&(n_mels as u32).to_le_bytes()).unwrap();
    for pred in &preds {
        for &v in pred {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    println!("\nPredictions saved → {}", out_path.display());
}
