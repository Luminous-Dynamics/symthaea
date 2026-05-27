// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//!
//! HellaSwag Commonsense NLI Benchmark (HDC baseline)
//!
//! Evaluates Symthaea's HDC BinaryHV encoding on the HellaSwag validation set
//! (Zellers et al., 2019). Each item has a context and 4 candidate endings;
//! the model picks the ending whose "context + ending" vector is most similar
//! to the context vector alone.
//!
//! Encoding: deterministic word→BinaryHV via `BinaryHV::random(hash(word))`,
//! sentence = `BinaryHV::bundle(&word_hvs)`.
//!
//! Random baseline: 25% (4-way multiple choice).
//!
//! Run with:
//!   cargo run --release --example benchmark_hellaswag
//!
//! Data expected at: data/benchmarks/hellaswag/validation.jsonl
//! Results written to: data/benchmarks/hellaswag/results.json

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::time::Instant;
use symthaea::hdc::binary_hv::BinaryHV;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Maximum samples to evaluate (caps runtime).
const MAX_SAMPLES: usize = 2000;

/// Path to HellaSwag JSONL validation data.
const DATA_PATH: &str = "data/benchmarks/hellaswag/validation.jsonl";

/// Path to write results JSON.
const RESULTS_PATH: &str = "data/benchmarks/hellaswag/results.json";

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// One HellaSwag item as stored in the JSONL file.
#[derive(Debug, Deserialize)]
struct HellaSwagItem {
    ctx: String,
    endings: Vec<String>,
    label: serde_json::Value,
    activity_label: String,
}

impl HellaSwagItem {
    /// Parse the label field, which may be an integer or a string representation.
    fn label_index(&self) -> usize {
        match &self.label {
            serde_json::Value::Number(n) => n.as_u64().unwrap_or(0) as usize,
            serde_json::Value::String(s) => s.parse::<usize>().unwrap_or(0),
            _ => 0,
        }
    }
}

/// Per-activity accuracy record.
#[derive(Debug, Default, Clone, Serialize)]
struct ActivityStats {
    total: usize,
    correct: usize,
    accuracy: f64,
}

/// Full results structure written to JSON.
#[derive(Debug, Serialize)]
struct BenchmarkResults {
    benchmark: String,
    method: String,
    total_samples: usize,
    correct: usize,
    overall_accuracy: f64,
    random_baseline: f64,
    elapsed_secs: f64,
    per_activity: HashMap<String, ActivityStats>,
}

// ---------------------------------------------------------------------------
// HDC encoding helpers
// ---------------------------------------------------------------------------

/// Hash a word string to a deterministic u64 seed for `BinaryHV::random`.
fn word_seed(word: &str) -> u64 {
    // Use a simple but effective hash: FNV-1a 64-bit.
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in word.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Tokenize text into lowercase words (alphanumeric sequences).
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

/// Encode a text string as a BinaryHV by bundling per-word random vectors.
fn encode_text(text: &str) -> BinaryHV {
    let words = tokenize(text);
    if words.is_empty() {
        return BinaryHV::random(0);
    }
    let word_hvs: Vec<BinaryHV> = words
        .iter()
        .map(|w| BinaryHV::random(word_seed(w)))
        .collect();
    BinaryHV::bundle(&word_hvs)
}

// ---------------------------------------------------------------------------
// Main benchmark
// ---------------------------------------------------------------------------

fn main() {
    println!("=============================================================");
    println!("  HellaSwag Commonsense NLI Benchmark (HDC BinaryHV)");
    println!("=============================================================\n");

    // --- Load data ---
    let data_path = Path::new(DATA_PATH);
    if !data_path.exists() {
        eprintln!(
            "ERROR: Data file not found at {}\n\
             Download HellaSwag validation JSONL and place it there.\n\
             See: https://github.com/rowanz/hellaswag",
            DATA_PATH
        );
        std::process::exit(1);
    }

    let file = File::open(data_path).expect("Failed to open data file");
    let reader = BufReader::new(file);

    let mut items: Vec<HellaSwagItem> = Vec::new();
    for (line_no, line) in reader.lines().enumerate() {
        if items.len() >= MAX_SAMPLES {
            break;
        }
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("WARNING: Skipping line {}: {}", line_no + 1, e);
                continue;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<HellaSwagItem>(&line) {
            Ok(item) => {
                if item.endings.len() == 4 {
                    items.push(item);
                } else {
                    eprintln!(
                        "WARNING: Skipping line {} (expected 4 endings, got {})",
                        line_no + 1,
                        item.endings.len()
                    );
                }
            }
            Err(e) => {
                eprintln!("WARNING: Skipping line {}: parse error: {}", line_no + 1, e);
            }
        }
    }

    let n = items.len();
    println!("Loaded {} items (cap: {})\n", n, MAX_SAMPLES);
    if n == 0 {
        eprintln!("ERROR: No valid items loaded.");
        std::process::exit(1);
    }

    // --- Evaluate ---
    let start = Instant::now();
    let mut correct = 0usize;
    let mut per_activity: HashMap<String, ActivityStats> = HashMap::new();

    for (i, item) in items.iter().enumerate() {
        let ctx_hv = encode_text(&item.ctx);
        let gold = item.label_index();

        // Score each ending: encode "context + ending" and compare to context
        let mut best_idx = 0usize;
        let mut best_sim = f32::NEG_INFINITY;

        for (j, ending) in item.endings.iter().enumerate() {
            let full_text = format!("{} {}", &item.ctx, ending);
            let full_hv = encode_text(&full_text);
            let sim = ctx_hv.similarity(&full_hv);
            if sim > best_sim {
                best_sim = sim;
                best_idx = j;
            }
        }

        let is_correct = best_idx == gold;
        if is_correct {
            correct += 1;
        }

        let stats = per_activity.entry(item.activity_label.clone()).or_default();
        stats.total += 1;
        if is_correct {
            stats.correct += 1;
        }

        // Progress every 200 items
        if (i + 1) % 200 == 0 || i + 1 == n {
            let running_acc = correct as f64 / (i + 1) as f64 * 100.0;
            println!(
                "  [{:>5}/{}] running accuracy: {:.1}%",
                i + 1,
                n,
                running_acc
            );
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let overall_accuracy = correct as f64 / n as f64;

    // Compute per-activity accuracy
    for stats in per_activity.values_mut() {
        stats.accuracy = if stats.total > 0 {
            stats.correct as f64 / stats.total as f64
        } else {
            0.0
        };
    }

    // --- Report ---
    println!("\n=============================================================");
    println!("  Results");
    println!("=============================================================");
    println!("  Total samples:     {}", n);
    println!("  Correct:           {}", correct);
    println!("  Overall accuracy:  {:.2}%", overall_accuracy * 100.0);
    println!("  Random baseline:   25.00%");
    println!("  Elapsed:           {:.2}s", elapsed);
    println!(
        "  Throughput:        {:.1} items/sec",
        n as f64 / elapsed.max(0.001)
    );

    // Top/bottom activities
    let mut activities: Vec<_> = per_activity.iter().collect();
    activities.sort_by(|a, b| {
        b.1.accuracy
            .partial_cmp(&a.1.accuracy)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!(
        "\n  Per-activity breakdown ({} activities):",
        activities.len()
    );
    println!(
        "  {:>50}  {:>6}  {:>6}  {:>7}",
        "Activity", "Total", "Corr", "Acc"
    );
    println!("  {}", "-".repeat(75));

    for (name, stats) in &activities {
        println!(
            "  {:>50}  {:>6}  {:>6}  {:>6.1}%",
            truncate_str(name, 50),
            stats.total,
            stats.correct,
            stats.accuracy * 100.0,
        );
    }
    println!();

    // --- Save results ---
    let results = BenchmarkResults {
        benchmark: "HellaSwag".to_string(),
        method: "HDC BinaryHV (16384D, word-level bag-of-words bundle, cosine similarity)"
            .to_string(),
        total_samples: n,
        correct,
        overall_accuracy,
        random_baseline: 0.25,
        elapsed_secs: elapsed,
        per_activity,
    };

    let results_path = Path::new(RESULTS_PATH);
    if let Some(parent) = results_path.parent() {
        fs::create_dir_all(parent).expect("Failed to create results directory");
    }

    let json = serde_json::to_string_pretty(&results).expect("Failed to serialize results");
    let mut out = File::create(results_path).expect("Failed to create results file");
    out.write_all(json.as_bytes())
        .expect("Failed to write results");

    println!("  Results saved to: {}", RESULTS_PATH);
    println!("=============================================================");
}

/// Truncate a string to at most `max_len` characters, appending "..." if truncated.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}