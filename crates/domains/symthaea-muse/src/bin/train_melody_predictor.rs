// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fit the `learned_melody::MelodyPredictor` linear model on real MIDI data.
//!
//! This is the in-repo, reproducible trainer that `learned_melody.rs`'s
//! provenance note demanded: it fits the exact 20-weight + bias linear model
//! that `MelodyPredictor::predict()` evaluates, using `midi_trainer`'s
//! feature pipeline, on a file-level held-out split, and reports HONEST
//! metrics (including baselines) before writing the weights + provenance.
//!
//! Fitting: `predict()` decodes interval = tanh(5·(Wx+b))·12 and duration =
//! (tanh(Wx+b)·0.5+0.5)·4, so we least-squares Wx+b against the
//! inverse-decoded targets (atanh transform), with a small ridge term.
//! Closed-form normal equations — no iterative training to babysit.
//!
//! Split: files whose name hashes to bucket 0 of 10 form the test set
//! (deterministic, augmentation-free, no window leakage across the split).
//!
//! Usage:
//! ```bash
//! cargo run --release -p symthaea-muse --bin train_melody_predictor -- \
//!     /opt/datasets/maestro/maestro-v3.0.0 \
//!     crates/domains/symthaea-muse/data/melody_predictor_weights.json
//! ```

use std::path::{Path, PathBuf};
use symthaea_muse::midi_trainer::{MelodyTrainingPair, melody_to_training_pairs, parse_midi};

const DIM: usize = 21; // 20 features + bias

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let midi_root = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("/opt/datasets/maestro/maestro-v3.0.0");
    let out_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("crates/domains/symthaea-muse/data/melody_predictor_weights.json");

    println!("=== MelodyPredictor linear-model trainer ===");
    println!("MIDI root: {midi_root}");

    // 1. Collect MIDI files recursively
    let mut files = Vec::new();
    collect_midi_files(Path::new(midi_root), &mut files);
    files.sort(); // deterministic order
    if files.is_empty() {
        eprintln!("No MIDI files found under {midi_root}");
        std::process::exit(1);
    }
    println!("Found {} MIDI files", files.len());

    // 2. Stream files: accumulate normal equations on train, keep test pairs
    let mut xtx_iv = vec![[0.0f64; DIM]; DIM];
    let mut xty_iv = [0.0f64; DIM];
    let mut xtx_dur = vec![[0.0f64; DIM]; DIM];
    let mut xty_dur = [0.0f64; DIM];
    let mut test_pairs: Vec<MelodyTrainingPair> = Vec::new();
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
        let pairs = melody_to_training_pairs(&melody);
        if pairs.is_empty() {
            n_skipped += 1;
            continue;
        }
        let is_test = file_hash(path) % 10 == 0;
        if is_test {
            n_test_files += 1;
            test_pairs.extend(pairs);
        } else {
            n_train_files += 1;
            for p in &pairs {
                let x = features_with_bias(p);
                let (y_iv, y_dur) = inverse_decoded_targets(p);
                for r in 0..DIM {
                    for c in r..DIM {
                        xtx_iv[r][c] += x[r] * x[c];
                        xtx_dur[r][c] += x[r] * x[c];
                    }
                    xty_iv[r] += x[r] * y_iv;
                    xty_dur[r] += x[r] * y_dur;
                }
                n_train_pairs += 1;
            }
        }
        if (i + 1) % 100 == 0 {
            println!("  parsed {}/{} files...", i + 1, files.len());
        }
    }
    // Mirror the upper triangle
    for r in 0..DIM {
        for c in 0..r {
            xtx_iv[r][c] = xtx_iv[c][r];
            xtx_dur[r][c] = xtx_dur[c][r];
        }
    }
    println!(
        "Train: {n_train_files} files / {n_train_pairs} pairs. \
         Test: {n_test_files} files / {} pairs. Skipped: {n_skipped}.",
        test_pairs.len()
    );
    if n_train_pairs < 1000 || test_pairs.len() < 200 {
        eprintln!("Not enough data for a trustworthy fit — aborting.");
        std::process::exit(1);
    }

    // 3. Solve ridge-regularized normal equations
    let ridge = 1e-3 * n_train_pairs as f64;
    let w_iv = solve_ridge(&xtx_iv, &xty_iv, ridge);
    let w_dur = solve_ridge(&xtx_dur, &xty_dur, ridge);

    // 4. Honest held-out evaluation: trained vs old constants vs baselines
    let trained = eval_weights(&test_pairs, &w_iv, &w_dur);
    let old = eval_old_constants(&test_pairs);
    let naive = eval_naive_baselines(&test_pairs);

    println!("\n── Held-out evaluation ({} pairs) ──", test_pairs.len());
    println!("                     direction-acc  interval-MAE  duration-MAE");
    println!(
        "  trained (this run)      {:.3}         {:.2} st       {:.2} beats",
        trained.dir_acc, trained.iv_mae, trained.dur_mae
    );
    println!(
        "  old hand-tuned consts   {:.3}         {:.2} st       {:.2} beats",
        old.dir_acc, old.iv_mae, old.dur_mae
    );
    println!(
        "  naive continue-direction {:.3}        (majority-dir {:.3})",
        naive.0, naive.1
    );
    println!(
        "  direction-acc counts moves >= 1 semitone only ({} of {} test pairs)",
        trained.n_directional,
        test_pairs.len()
    );

    // 5. Write weights + embedded provenance
    let json = weights_json(
        &w_iv,
        &w_dur,
        midi_root,
        n_train_files,
        n_train_pairs,
        n_test_files,
        test_pairs.len(),
        &trained,
        &old,
        naive,
    );
    if let Some(parent) = Path::new(out_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::write(out_path, &json) {
        Ok(()) => println!("\nWrote {out_path}"),
        Err(e) => {
            eprintln!("Failed to write {out_path}: {e}");
            std::process::exit(1);
        }
    }
    println!(
        "Rebuild symthaea-muse so learned_melody's include_str! picks up the new weights,\n\
         then re-run `cargo test -p symthaea-muse --lib learned_melody`."
    );
}

// ─── Data plumbing ──────────────────────────────────────────────────────────

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

/// Deterministic per-file hash for the train/test split (FNV-1a over the
/// file name — stable across runs and machines).
fn file_hash(path: &Path) -> u64 {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let mut h: u64 = 0xcbf29ce484222325;
    for b in name.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// 20 features (matching `MelodyPredictor::build_features` exactly) + bias.
///
/// Alignment matters: `build_features` RIGHT-aligns histories (newest interval
/// at slot 7, newest duration at slot 15, zeros on the left when short). The
/// leakage fix leaves 7 context intervals, so slot 0 is always empty here —
/// training must place them in the same slots inference will.
fn features_with_bias(p: &MelodyTrainingPair) -> [f64; DIM] {
    let mut x = [0.0f64; DIM];
    // Intervals /12 clamped, newest at slot 7
    for (i, &iv) in p.interval_context.iter().rev().take(8).enumerate() {
        x[7 - i] = ((iv / 12.0).clamp(-1.0, 1.0)) as f64;
    }
    // Durations /4 clamped, newest at slot 15
    for (i, &dur) in p.duration_context.iter().rev().take(8).enumerate() {
        x[8 + 7 - i] = ((dur / 4.0).clamp(0.0, 1.0)) as f64;
    }
    x[16] = (p.beat_position / 4.0) as f64;
    x[17] = p.phrase_position as f64;
    x[18] = p.valence as f64;
    x[19] = p.arousal as f64;
    x[20] = 1.0; // bias
    x
}

/// Invert `predict()`'s decode so the linear fit targets raw pre-activation.
/// interval: pred = tanh(5·raw)·12  →  raw = atanh(t/12)/5
/// duration: pred = (tanh(raw)·0.5+0.5)·4  →  raw = atanh(t/2 − 1)
fn inverse_decoded_targets(p: &MelodyTrainingPair) -> (f64, f64) {
    let t_iv = (p.target_interval as f64 / 12.0).clamp(-0.98, 0.98);
    let y_iv = t_iv.atanh() / 5.0;
    let t_dur = (p.target_duration as f64 / 2.0 - 1.0).clamp(-0.98, 0.98);
    let y_dur = t_dur.atanh();
    (y_iv, y_dur)
}

// ─── Ridge solve (Gaussian elimination, partial pivoting) ───────────────────

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
                let f = a[r][col] / pv;
                for c in col..=DIM {
                    a[r][c] -= f * a[col][c];
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

// ─── Evaluation ─────────────────────────────────────────────────────────────

struct EvalResult {
    dir_acc: f64,
    iv_mae: f64,
    dur_mae: f64,
    n_directional: usize,
}

fn decode(raw_iv: f64, raw_dur: f64) -> (f64, f64) {
    let iv = (raw_iv * 5.0).tanh() * 12.0;
    let dur = (raw_dur.tanh() * 0.5 + 0.5) * 4.0;
    (iv, dur)
}

fn eval_linear(
    pairs: &[MelodyTrainingPair],
    predict: impl Fn(&[f64; DIM]) -> (f64, f64),
) -> EvalResult {
    let (mut dir_hits, mut n_directional) = (0usize, 0usize);
    let (mut iv_err, mut dur_err) = (0.0f64, 0.0f64);
    for p in pairs {
        let x = features_with_bias(p);
        let (raw_iv, raw_dur) = predict(&x);
        let (pred_iv, pred_dur) = decode(raw_iv, raw_dur);
        iv_err += (pred_iv - p.target_interval as f64).abs();
        dur_err += (pred_dur - p.target_duration as f64).abs();
        if p.target_interval.abs() >= 1.0 {
            n_directional += 1;
            if (pred_iv >= 0.0) == (p.target_interval >= 0.0) {
                dir_hits += 1;
            }
        }
    }
    EvalResult {
        dir_acc: dir_hits as f64 / n_directional.max(1) as f64,
        iv_mae: iv_err / pairs.len().max(1) as f64,
        dur_mae: dur_err / pairs.len().max(1) as f64,
        n_directional,
    }
}

fn eval_weights(pairs: &[MelodyTrainingPair], w_iv: &[f32; DIM], w_dur: &[f32; DIM]) -> EvalResult {
    eval_linear(pairs, |x| {
        let mut riv = 0.0f64;
        let mut rdur = 0.0f64;
        for i in 0..DIM {
            riv += w_iv[i] as f64 * x[i];
            rdur += w_dur[i] as f64 * x[i];
        }
        (riv, rdur)
    })
}

/// The pre-2026-07-06 hand-tuned constants, kept here so every training run
/// reports how the shipped model compares against them on the same split.
#[rustfmt::skip]
const OLD_INTERVAL_WEIGHTS: [f32; 20] = [
    0.0012, -0.0018, 0.0065, -0.0025, 0.0008, -0.0015, -0.0839, 0.1854,
    0.0003, -0.0005, 0.0002, -0.0008, 0.0004, -0.0003, 0.0011, 0.0029,
    0.0015, -0.0022, 0.0018, 0.0008,
];
const OLD_INTERVAL_BIAS: f32 = -0.002;
#[rustfmt::skip]
const OLD_DURATION_WEIGHTS: [f32; 20] = [
    -0.0005, 0.0003, -0.0002, 0.0001, -0.0004, 0.0002, 0.0008, -0.0012,
    0.0015, -0.0008, 0.0010, -0.0005, 0.0012, -0.0006, 0.0280, 0.0850,
    0.0030, 0.0120, -0.0005, -0.0180,
];
const OLD_DURATION_BIAS: f32 = 0.25;

fn eval_old_constants(pairs: &[MelodyTrainingPair]) -> EvalResult {
    eval_linear(pairs, |x| {
        let mut riv = OLD_INTERVAL_BIAS as f64;
        let mut rdur = OLD_DURATION_BIAS as f64;
        for i in 0..20 {
            riv += OLD_INTERVAL_WEIGHTS[i] as f64 * x[i];
            rdur += OLD_DURATION_WEIGHTS[i] as f64 * x[i];
        }
        (riv, rdur)
    })
}

/// (continue-previous-direction accuracy, majority-direction accuracy)
fn eval_naive_baselines(pairs: &[MelodyTrainingPair]) -> (f64, f64) {
    let (mut cont_hits, mut down_hits, mut n) = (0usize, 0usize, 0usize);
    for p in pairs {
        if p.target_interval.abs() < 1.0 {
            continue;
        }
        n += 1;
        let last_iv = p.interval_context.last().copied().unwrap_or(0.0);
        if (last_iv >= 0.0) == (p.target_interval >= 0.0) {
            cont_hits += 1;
        }
        if p.target_interval < 0.0 {
            down_hits += 1; // melodic gravity: descent is the majority class
        }
    }
    (
        cont_hits as f64 / n.max(1) as f64,
        down_hits as f64 / n.max(1) as f64,
    )
}

// ─── Output ─────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn weights_json(
    w_iv: &[f32; DIM],
    w_dur: &[f32; DIM],
    midi_root: &str,
    n_train_files: usize,
    n_train_pairs: usize,
    n_test_files: usize,
    n_test_pairs: usize,
    trained: &EvalResult,
    old: &EvalResult,
    naive: (f64, f64),
) -> String {
    let date = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!(
        "{{\n  \"provenance\": {{\n    \"trainer\": \"src/bin/train_melody_predictor.rs\",\n    \
         \"dataset\": \"{midi_root}\",\n    \"unix_time\": {date},\n    \
         \"train_files\": {n_train_files},\n    \"train_pairs\": {n_train_pairs},\n    \
         \"test_files\": {n_test_files},\n    \"test_pairs\": {n_test_pairs},\n    \
         \"split\": \"file-level FNV-1a bucket 0 of 10, no augmentation\",\n    \
         \"fit\": \"ridge least squares (lambda=1e-3*n) on atanh-inverse-decoded targets\",\n    \
         \"held_out_direction_accuracy\": {:.4},\n    \
         \"held_out_interval_mae_semitones\": {:.4},\n    \
         \"held_out_duration_mae_beats\": {:.4},\n    \
         \"old_hand_tuned_direction_accuracy\": {:.4},\n    \
         \"naive_continue_direction_accuracy\": {:.4},\n    \
         \"naive_majority_direction_accuracy\": {:.4}\n  }},\n  \
         \"interval_weights\": {:?},\n  \"interval_bias\": {},\n  \
         \"duration_weights\": {:?},\n  \"duration_bias\": {}\n}}\n",
        trained.dir_acc,
        trained.iv_mae,
        trained.dur_mae,
        old.dir_acc,
        naive.0,
        naive.1,
        &w_iv[..20],
        w_iv[20],
        &w_dur[..20],
        w_dur[20],
    )
}
