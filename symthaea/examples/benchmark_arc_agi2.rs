// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # ARC-AGI-2 Benchmark (honest baseline)
//!
//! Runs Symthaea's current `GridEncoder`-based rule-vector pipeline over
//! the ARC-AGI-2 public evaluation set (120 tasks) and emits a reproducible
//! CSV. This is the Phase 1 **honest baseline** — no output-grid generator
//! is wired in yet, so `predicted_correct` is always 0. The purpose of the
//! baseline is to publish reproducible *per-task* measurements that future
//! work (stretch `grid_macro_discovery.rs` integration at Week-3 go/no-go)
//! will improve.
//!
//! ## Dataset
//!
//! Not vendored. See `docs/arc-agi-2-dataset.md` for upstream source and
//! licensing. The env var `SYMTHAEA_ARC2_DATA_DIR` must point at the
//! directory containing the 120 evaluation JSON files.
//!
//! ## Output
//!
//! CSV to stdout with columns:
//! - `task_id`: filename stem
//! - `num_train`: training example count in the task
//! - `num_test`: test input count
//! - `intra_rule_consistency`: mean pairwise similarity of training rule
//!   hypervectors (signal: how self-consistent is the task's rule?)
//! - `predicted_correct`: 1 iff at least one test output was reconstructed
//!   correctly (always 0 at baseline — no generator yet)
//! - `notes`: free-text diagnostic
//!
//! ## Run
//!
//! ```bash
//! SYMTHAEA_ARC2_DATA_DIR=/path/to/arc-agi-2/evaluation \
//!   cargo run --release --example benchmark_arc_agi2 > arc_agi2_results.csv
//! ```

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use symthaea::hdc::grid_encoder::GridEncoder;

fn arc2_data_dir() -> PathBuf {
    env::var("SYMTHAEA_ARC2_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data/benchmarks/arc-agi-2/evaluation"))
}

fn parse_grid(value: &serde_json::Value) -> Option<Vec<Vec<u8>>> {
    let arr = value.as_array()?;
    arr.iter()
        .map(|row| {
            row.as_array().map(|r| {
                r.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as u8))
                    .collect()
            })
        })
        .collect()
}

struct ArcTask {
    train: Vec<(Vec<Vec<u8>>, Vec<Vec<u8>>)>,
    test_inputs: Vec<Vec<Vec<u8>>>,
    test_outputs: Vec<Option<Vec<Vec<u8>>>>,
}

fn parse_task(path: &Path) -> Option<ArcTask> {
    let content = fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;

    let train = json["train"]
        .as_array()?
        .iter()
        .filter_map(|example| {
            let input = parse_grid(&example["input"])?;
            let output = parse_grid(&example["output"])?;
            Some((input, output))
        })
        .collect::<Vec<_>>();

    // Test inputs: always present. Outputs: present in public eval with
    // ground truth, absent in held-out evaluation. Handle both.
    let test_entries = json["test"].as_array()?;
    let mut test_inputs: Vec<Vec<Vec<u8>>> = Vec::new();
    let mut test_outputs: Vec<Option<Vec<Vec<u8>>>> = Vec::new();
    for entry in test_entries {
        let input = parse_grid(&entry["input"])?;
        test_inputs.push(input);
        test_outputs.push(parse_grid(&entry["output"]));
    }

    if train.is_empty() || test_inputs.is_empty() {
        return None;
    }

    Some(ArcTask {
        train,
        test_inputs,
        test_outputs,
    })
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn main() -> ExitCode {
    let data_dir = arc2_data_dir();

    if !data_dir.exists() {
        eprintln!("# ARC-AGI-2 dataset not found at {}", data_dir.display());
        eprintln!("# Set SYMTHAEA_ARC2_DATA_DIR to the evaluation directory.");
        eprintln!("# See docs/arc-agi-2-dataset.md for upstream source.");
        // Emit an empty CSV with just the header so downstream tooling
        // can distinguish "ran, no data" from "panic".
        println!("task_id,num_train,num_test,intra_rule_consistency,predicted_correct,notes");
        return ExitCode::SUCCESS;
    }

    let mut task_files: Vec<_> = match fs::read_dir(&data_dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false))
            .map(|e| e.path())
            .collect(),
        Err(e) => {
            eprintln!("# Failed to read {}: {}", data_dir.display(), e);
            println!("task_id,num_train,num_test,intra_rule_consistency,predicted_correct,notes");
            return ExitCode::SUCCESS;
        }
    };
    task_files.sort();

    // Encoder: same conventions as ARC-AGI-1 benchmark for comparability.
    let dim = 4096;
    let encoder = GridEncoder::new(dim, 30, 30, 10, 0);

    // Emit CSV header
    println!("task_id,num_train,num_test,intra_rule_consistency,predicted_correct,notes");

    let mut tasks_seen = 0usize;
    let mut tasks_parsed = 0usize;

    for task_path in &task_files {
        tasks_seen += 1;
        let task_id = task_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let task = match parse_task(task_path) {
            Some(t) => t,
            None => {
                println!(
                    "{},0,0,0.0000,0,{}",
                    csv_escape(&task_id),
                    csv_escape("parse_failed")
                );
                continue;
            }
        };
        tasks_parsed += 1;

        // Encode training rule vectors.
        let rule_hvs: Vec<_> = task
            .train
            .iter()
            .map(|(inp, out)| {
                let in_hv = encoder.encode_grid(inp);
                let out_hv = encoder.encode_grid(out);
                encoder.encode_rule(&in_hv, &out_hv)
            })
            .collect();

        // Intra-task rule consistency: mean pairwise similarity.
        let mut total_sim = 0.0f32;
        let mut n_pairs = 0usize;
        for i in 0..rule_hvs.len() {
            for j in i + 1..rule_hvs.len() {
                total_sim += rule_hvs[i].similarity(&rule_hvs[j]);
                n_pairs += 1;
            }
        }
        let intra_consistency = if n_pairs > 0 {
            total_sim / n_pairs as f32
        } else {
            0.0
        };

        // Honest: no generator yet, so predicted_correct is always 0.
        // Whether ground-truth outputs are present affects the note, not
        // the score (we're not running a predictor to compare against).
        let has_ground_truth = task.test_outputs.iter().any(|o| o.is_some());
        let notes = if has_ground_truth {
            "no_generator"
        } else {
            "no_generator;no_gt"
        };

        println!(
            "{},{},{},{:.4},0,{}",
            csv_escape(&task_id),
            task.train.len(),
            task.test_inputs.len(),
            intra_consistency,
            csv_escape(notes)
        );
    }

    eprintln!(
        "# summary: {} files seen, {} tasks parsed, 0 solved (no generator wired)",
        tasks_seen, tasks_parsed
    );

    ExitCode::SUCCESS
}
