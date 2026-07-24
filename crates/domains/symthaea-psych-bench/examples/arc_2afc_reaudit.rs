// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Re-audit of the ARC-AGI "2-AFC" scoring used by `arc_dataset.rs`/`arc_strict.rs`
//! and siblings (`arc_fewshot`, `arc_noise`, `arc_chain`, `arc_scaling`,
//! `arc_staircase`).
//!
//! **Why this exists**: the 2026-07-18 pixel-ARC vision experiment found that an
//! HDC rule-vector 2-AFC protocol collapses to chance (46.3%, margins ~1e-4) once
//! the wrong-answer distractor is a FAIR one (same structural class as the
//! correct answer) rather than an arbitrary one. The ARC-AGI benchmarks in this
//! crate score "2-AFC" by comparing the predicted output HV's similarity to the
//! actual output HV against its similarity to a **literally random `BinaryHV`**
//! (see `arc_dataset.rs:208`, `arc_strict.rs:239`, and the same pattern in
//! `arc_fewshot`/`arc_noise`/`arc_chain`/`arc_scaling`/`arc_staircase`). Because
//! `BinaryGridEncoder` builds every grid HV as a majority-vote bundle over a
//! *shared* small basis (per-task row/col/color HVs), ANY two grids encoded by
//! the same encoder share structure that a true-random HV does not — so this
//! comparison is structured-vs-noise, not correct-vs-plausible-wrong. This
//! example reruns the real 400-task ARC-AGI training set already checked into
//! `data/benchmarks/arc/repo/data/training` and reports 2-AFC accuracy under:
//!
//!   1. **random**: the existing method (distractor = random BinaryHV)
//!   2. **identity**: distractor = the test input itself, unchanged (a
//!      "no transformation happened" wrong guess — same encoder, same basis)
//!   3. **reflect_x** / **reflect_y**: distractor = a generic wrong transform of
//!      the test input (skipped per-task if it's identical to the true output)
//!
//! Fair distractors 2-4 are exactly as "grid-structured" as the true answer;
//! only their answer to the *specific rule* is wrong. If accuracy collapses
//! under these relative to `random`, the "100% 2-AFC" claim in
//! `book/src/research/validation.md`/`publications.md` is inflated the same way
//! the retracted ETHICS 94.5% figure was — by an easy discriminability
//! shortcut, not real rule-transfer performance.
//!
//! ## Run
//! ```bash
//! cargo run -p symthaea-psych-bench --example arc_2afc_reaudit --release
//! ```

use std::path::PathBuf;

use symthaea_core::hdc::BinaryHV;
use symthaea_core::hdc::binary_grid_encoder::BinaryGridEncoder;
use symthaea_core::hdc::grid_encoder::GridEncoder;
use symthaea_psych_bench::benchmarks::reasoning::arc_dataset::{ArcTask, load_arc_tasks};

fn arc_data_dir() -> PathBuf {
    std::env::var("SYMTHAEA_ARC_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data/benchmarks/arc/repo/data/training"))
}

struct TaskResult {
    id: String,
    random_correct: bool,
    identity_correct: Option<bool>,
    reflect_x_correct: Option<bool>,
    reflect_y_correct: Option<bool>,
    pred_actual_sim: f64,
    pred_random_sim: f64,
}

fn eval_task(task_id: &str, task: &ArcTask, seed: u64) -> Option<TaskResult> {
    if task.train.is_empty() || task.test.is_empty() {
        return None;
    }

    let max_rows = task
        .train
        .iter()
        .chain(task.test.iter())
        .flat_map(|p| [p.input.len(), p.output.len()])
        .max()
        .unwrap_or(30);
    let max_cols = task
        .train
        .iter()
        .chain(task.test.iter())
        .flat_map(|p| {
            [
                p.input.iter().map(|r| r.len()).max().unwrap_or(0),
                p.output.iter().map(|r| r.len()).max().unwrap_or(0),
            ]
        })
        .max()
        .unwrap_or(30);
    let num_colors = 10;

    let encoder = BinaryGridEncoder::new(max_rows.max(1), max_cols.max(1), num_colors, seed);

    let rules: Vec<BinaryHV> = task
        .train
        .iter()
        .map(|pair| {
            let in_hv = encoder.encode_grid(&pair.input);
            let out_hv = encoder.encode_grid(&pair.output);
            encoder.encode_rule(&in_hv, &out_hv)
        })
        .collect();
    let consensus = encoder.bundle_rules(&rules);

    let test_pair = &task.test[0];
    let test_in_hv = encoder.encode_grid(&test_pair.input);
    let test_out_hv = encoder.encode_grid(&test_pair.output);
    let predicted = encoder.apply_rule(&test_in_hv, &consensus);

    let pred_actual_sim = predicted.similarity(&test_out_hv) as f64;

    // Existing method: random BinaryHV distractor.
    let distractor = BinaryHV::random(seed ^ 0xDEAD);
    let pred_random_sim = predicted.similarity(&distractor) as f64;
    let random_correct = pred_actual_sim > pred_random_sim;

    // Fair distractor #1: identity (test input unchanged) — only valid if it
    // actually differs from the true output (else there's no wrong answer to test).
    let identity_correct = if test_pair.input != test_pair.output {
        let id_hv = encoder.encode_grid(&test_pair.input);
        let id_sim = predicted.similarity(&id_hv) as f64;
        Some(pred_actual_sim > id_sim)
    } else {
        None
    };

    // Fair distractor #2/#3: generic wrong transform, same dims as input (safe
    // for the per-task encoder's row/col basis — reflects never change shape).
    let reflect_x_grid = GridEncoder::reflect_x(&test_pair.input);
    let reflect_x_correct = if reflect_x_grid != test_pair.output {
        let hv = encoder.encode_grid(&reflect_x_grid);
        let sim = predicted.similarity(&hv) as f64;
        Some(pred_actual_sim > sim)
    } else {
        None
    };
    let reflect_y_grid = GridEncoder::reflect_y(&test_pair.input);
    let reflect_y_correct = if reflect_y_grid != test_pair.output {
        let hv = encoder.encode_grid(&reflect_y_grid);
        let sim = predicted.similarity(&hv) as f64;
        Some(pred_actual_sim > sim)
    } else {
        None
    };

    Some(TaskResult {
        id: task_id.to_string(),
        random_correct,
        identity_correct,
        reflect_x_correct,
        reflect_y_correct,
        pred_actual_sim,
        pred_random_sim,
    })
}

fn rate(results: &[TaskResult], f: impl Fn(&TaskResult) -> Option<bool>) -> (f64, usize) {
    let vals: Vec<bool> = results.iter().filter_map(&f).collect();
    if vals.is_empty() {
        return (0.0, 0);
    }
    let correct = vals.iter().filter(|&&v| v).count();
    (correct as f64 / vals.len() as f64, vals.len())
}

fn main() {
    let dir = arc_data_dir();
    if !dir.is_dir() {
        eprintln!("ARC data not found at {}", dir.display());
        eprintln!("Set SYMTHAEA_ARC_DATA_DIR to override.");
        return;
    }

    let tasks = load_arc_tasks(&dir).expect("failed to load ARC tasks");
    println!(
        "Loaded {} real ARC-AGI tasks from {}\n",
        tasks.len(),
        dir.display()
    );

    let seed_base = 42u64;
    let mut results = Vec::new();
    for (i, (task_id, task)) in tasks.iter().enumerate() {
        if let Some(r) = eval_task(task_id, task, seed_base ^ (i as u64)) {
            results.push(r);
        }
    }

    println!(
        "Evaluated {} tasks (train.len()>0 && test.len()>0)\n",
        results.len()
    );

    let (random_acc, random_n) = rate(&results, |r| Some(r.random_correct));
    let (identity_acc, identity_n) = rate(&results, |r| r.identity_correct);
    let (rx_acc, rx_n) = rate(&results, |r| r.reflect_x_correct);
    let (ry_acc, ry_n) = rate(&results, |r| r.reflect_y_correct);

    let mean_pred_actual: f64 =
        results.iter().map(|r| r.pred_actual_sim).sum::<f64>() / results.len().max(1) as f64;
    let mean_pred_random: f64 =
        results.iter().map(|r| r.pred_random_sim).sum::<f64>() / results.len().max(1) as f64;

    println!("═══════════════════════════════════════════════════════════");
    println!(" 2-AFC accuracy by distractor type (chance = 50%)");
    println!("═══════════════════════════════════════════════════════════");
    println!(
        "  random distractor (EXISTING METHOD):  {:.1}%  (n={})",
        random_acc * 100.0,
        random_n
    );
    println!(
        "  identity distractor (fair):           {:.1}%  (n={})",
        identity_acc * 100.0,
        identity_n
    );
    println!(
        "  reflect_x distractor (fair):          {:.1}%  (n={})",
        rx_acc * 100.0,
        rx_n
    );
    println!(
        "  reflect_y distractor (fair):          {:.1}%  (n={})",
        ry_acc * 100.0,
        ry_n
    );
    println!();
    println!(
        "  mean sim(predicted, actual output):   {:.4}",
        mean_pred_actual
    );
    println!(
        "  mean sim(predicted, random HV):        {:.4}",
        mean_pred_random
    );
    println!();

    let mut worst: Vec<(&str, f64)> = results
        .iter()
        .map(|r| (r.id.as_str(), r.pred_actual_sim))
        .collect();
    worst.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("  Worst 5 tasks by sim(predicted, actual) — beaten by identity distractor:");
    for (id, sim) in worst.iter().take(5) {
        println!("    {} sim={:.4}", id, sim);
    }
    println!();
    println!("  NOTE: 'random' compares a structured grid encoding against an unrelated");
    println!("  random vector. If mean sim(predicted,actual) >> mean sim(predicted,random)");
    println!("  regardless of whether the RULE was learned correctly, the random-distractor");
    println!("  metric mostly measures 'is this a valid grid encoding', not rule transfer.");
}
