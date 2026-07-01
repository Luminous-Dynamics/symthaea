// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # TruthfulQA Full Benchmark (HDC BinaryHV)
//!
//! Evaluates Symthaea's HDC BinaryHV encoding on the complete TruthfulQA dataset
//! (Lin et al., 2022) — 817 questions across 38 categories.
//!
//! ## Method
//! 1. Encode each question as a BinaryHV (hash each word, bundle)
//! 2. Encode correct answer(s) and incorrect answer(s) as BinaryHVs
//! 3. MC1 accuracy: is similarity(Q, best_correct) > similarity(Q, best_incorrect)?
//! 4. Track per-category breakdown
//!
//! ## Encoding
//! Deterministic word->BinaryHV via `BinaryHV::random(fnv1a(word))`,
//! sentence = `BinaryHV::bundle(&word_hvs)`.
//!
//! Random baseline: 50% (binary comparison).
//!
//! ## Dataset
//! TruthfulQA (Lin et al., 2022): 817 questions, 38 categories
//! CSV: Type, Category, Question, Best Answer, Best Incorrect Answer,
//!       Correct Answers (;-separated), Incorrect Answers (;-separated), Source
//!
//! ## Run
//! ```bash
//! cargo run --release --example benchmark_truthfulqa_full
//! ```
//!
//! Data: data/benchmarks/truthfulqa/repo/TruthfulQA.csv
//! Results: data/benchmarks/truthfulqa/results.json

use serde::Serialize;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::time::Instant;
use symthaea::hdc::binary_hv::BinaryHV;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const DATA_PATH: &str = "data/benchmarks/truthfulqa/repo/TruthfulQA.csv";
const RESULTS_PATH: &str = "data/benchmarks/truthfulqa/results.json";

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// One TruthfulQA question parsed from CSV.
#[derive(Debug, Clone)]
struct TruthfulQAItem {
    question_type: String,
    category: String,
    question: String,
    best_answer: String,
    best_incorrect_answer: String,
    correct_answers: Vec<String>,
    incorrect_answers: Vec<String>,
}

/// Per-category accuracy record.
#[derive(Debug, Default, Clone, Serialize)]
struct CategoryStats {
    total: usize,
    correct: usize,
    accuracy: f64,
}

/// Per-type accuracy record.
#[derive(Debug, Default, Clone, Serialize)]
struct TypeStats {
    total: usize,
    correct: usize,
    accuracy: f64,
}

/// Full results structure written to JSON.
#[derive(Debug, Serialize)]
struct BenchmarkResults {
    benchmark: String,
    method: String,
    total_questions: usize,
    mc1_correct: usize,
    mc1_accuracy: f64,
    mc1_any_correct: usize,
    mc1_any_accuracy: f64,
    random_baseline: f64,
    elapsed_secs: f64,
    avg_correct_similarity: f64,
    avg_incorrect_similarity: f64,
    similarity_gap: f64,
    per_category: HashMap<String, CategoryStats>,
    per_type: HashMap<String, TypeStats>,
}

// ---------------------------------------------------------------------------
// CSV parsing
// ---------------------------------------------------------------------------

/// Parse a single CSV field, handling quoted fields with embedded commas.
/// Returns (field, rest_of_line).
fn parse_csv_field(input: &str) -> (String, &str) {
    let input = input.trim_start();
    if input.starts_with('"') {
        // Quoted field — find the closing quote (handles "" escapes)
        let content = &input[1..];
        let mut field = String::new();
        let mut chars = content.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '"' {
                if chars.peek() == Some(&'"') {
                    // Escaped quote
                    field.push('"');
                    chars.next();
                } else {
                    // End of quoted field
                    break;
                }
            } else {
                field.push(ch);
            }
        }
        // Skip comma separator if present
        let rest: String = chars.collect();
        let rest_trimmed = rest.trim_start_matches(',');
        // We need to return a reference, so find position in original
        let consumed = input.len() - rest.len();
        let remainder = &input[consumed..];
        let remainder = remainder.trim_start_matches(',');
        (field, remainder)
    } else {
        // Unquoted field — find next comma
        match input.find(',') {
            Some(pos) => (input[..pos].to_string(), &input[pos + 1..]),
            None => (input.to_string(), ""),
        }
    }
}

/// Parse all 8 CSV fields from a line.
fn parse_csv_line(line: &str) -> Option<TruthfulQAItem> {
    let (question_type, rest) = parse_csv_field(line);
    let (category, rest) = parse_csv_field(rest);
    let (question, rest) = parse_csv_field(rest);
    let (best_answer, rest) = parse_csv_field(rest);
    let (best_incorrect_answer, rest) = parse_csv_field(rest);
    let (correct_answers_raw, rest) = parse_csv_field(rest);
    let (incorrect_answers_raw, _rest) = parse_csv_field(rest);

    if question.is_empty() {
        return None;
    }

    let correct_answers: Vec<String> = correct_answers_raw
        .split(';')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let incorrect_answers: Vec<String> = incorrect_answers_raw
        .split(';')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    Some(TruthfulQAItem {
        question_type,
        category,
        question,
        best_answer,
        best_incorrect_answer,
        correct_answers,
        incorrect_answers,
    })
}

// ---------------------------------------------------------------------------
// HDC encoding helpers
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit hash of a word string.
fn word_seed(word: &str) -> u64 {
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
    println!("  TruthfulQA Full Benchmark (HDC BinaryHV)");
    println!("  817 questions, 38 categories");
    println!("=============================================================\n");

    // --- Load data ---
    let data_path = Path::new(DATA_PATH);
    if !data_path.exists() {
        eprintln!(
            "ERROR: Data file not found at {}\n\
             Download TruthfulQA and place CSV at that path.\n\
             See: https://github.com/sylinrl/TruthfulQA",
            DATA_PATH
        );
        std::process::exit(1);
    }

    let file = File::open(data_path).expect("Failed to open TruthfulQA CSV");
    let reader = BufReader::new(file);

    let mut items: Vec<TruthfulQAItem> = Vec::new();
    for (line_no, line) in reader.lines().enumerate() {
        // Skip header
        if line_no == 0 {
            continue;
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
        match parse_csv_line(&line) {
            Some(item) => items.push(item),
            None => {
                eprintln!("WARNING: Skipping line {} (parse failed)", line_no + 1);
            }
        }
    }

    let n = items.len();
    println!("Loaded {} questions from TruthfulQA CSV\n", n);
    if n == 0 {
        eprintln!("ERROR: No questions loaded. Check CSV format.");
        std::process::exit(1);
    }

    // --- Run benchmark ---
    let start = Instant::now();

    let mut mc1_correct = 0usize;
    let mut mc1_any_correct = 0usize;
    let mut total_correct_sim = 0.0f64;
    let mut total_incorrect_sim = 0.0f64;
    let mut per_category: HashMap<String, CategoryStats> = HashMap::new();
    let mut per_type: HashMap<String, TypeStats> = HashMap::new();

    for (i, item) in items.iter().enumerate() {
        let q_hv = encode_text(&item.question);

        // MC1: best correct vs best incorrect
        let best_correct_hv = encode_text(&item.best_answer);
        let best_incorrect_hv = encode_text(&item.best_incorrect_answer);

        let sim_correct = q_hv.similarity(&best_correct_hv) as f64;
        let sim_incorrect = q_hv.similarity(&best_incorrect_hv) as f64;

        total_correct_sim += sim_correct;
        total_incorrect_sim += sim_incorrect;

        let mc1_hit = sim_correct > sim_incorrect;
        if mc1_hit {
            mc1_correct += 1;
        }

        // MC1-any: does Q match ANY correct answer better than best incorrect?
        let max_correct_sim = item
            .correct_answers
            .iter()
            .map(|a| q_hv.similarity(&encode_text(a)) as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        if max_correct_sim > sim_incorrect {
            mc1_any_correct += 1;
        }

        // Per-category stats
        let cat = per_category.entry(item.category.clone()).or_default();
        cat.total += 1;
        if mc1_hit {
            cat.correct += 1;
        }

        // Per-type stats
        let typ = per_type.entry(item.question_type.clone()).or_default();
        typ.total += 1;
        if mc1_hit {
            typ.correct += 1;
        }

        // Progress
        if (i + 1) % 100 == 0 || i + 1 == n {
            let acc = mc1_correct as f64 / (i + 1) as f64 * 100.0;
            println!("[{:>4}/{:>4}] MC1 running accuracy: {:.1}%", i + 1, n, acc);
        }
    }

    let elapsed = start.elapsed().as_secs_f64();

    // Compute final accuracies
    let mc1_accuracy = mc1_correct as f64 / n as f64;
    let mc1_any_accuracy = mc1_any_correct as f64 / n as f64;
    let avg_correct_sim = total_correct_sim / n as f64;
    let avg_incorrect_sim = total_incorrect_sim / n as f64;
    let similarity_gap = avg_correct_sim - avg_incorrect_sim;

    for stats in per_category.values_mut() {
        stats.accuracy = if stats.total > 0 {
            stats.correct as f64 / stats.total as f64
        } else {
            0.0
        };
    }
    for stats in per_type.values_mut() {
        stats.accuracy = if stats.total > 0 {
            stats.correct as f64 / stats.total as f64
        } else {
            0.0
        };
    }

    // --- Print results ---
    println!("\n=============================================================");
    println!("  RESULTS");
    println!("=============================================================\n");
    println!("Total questions:        {}", n);
    println!(
        "MC1 accuracy (best):    {:.1}% ({}/{})",
        mc1_accuracy * 100.0,
        mc1_correct,
        n
    );
    println!(
        "MC1 accuracy (any):     {:.1}% ({}/{})",
        mc1_any_accuracy * 100.0,
        mc1_any_correct,
        n
    );
    println!("Random baseline:        50.0%");
    println!("Avg correct sim:        {:.4}", avg_correct_sim);
    println!("Avg incorrect sim:      {:.4}", avg_incorrect_sim);
    println!("Similarity gap:         {:.4}", similarity_gap);
    println!("Elapsed:                {:.2}s", elapsed);

    // Per-type breakdown
    println!("\n--- Per-Type Breakdown ---");
    let mut types_sorted: Vec<_> = per_type.iter().collect();
    types_sorted.sort_by_key(|(k, _)| k.clone());
    for (typ, stats) in &types_sorted {
        println!(
            "  {:<20} {:.1}% ({}/{})",
            typ,
            stats.accuracy * 100.0,
            stats.correct,
            stats.total
        );
    }

    // Per-category breakdown (top and bottom 5)
    let mut cats_sorted: Vec<_> = per_category.iter().collect();
    cats_sorted.sort_by(|a, b| b.1.accuracy.partial_cmp(&a.1.accuracy).unwrap());

    println!("\n--- Top 5 Categories ---");
    for (cat, stats) in cats_sorted.iter().take(5) {
        println!(
            "  {:<30} {:.1}% ({}/{})",
            cat,
            stats.accuracy * 100.0,
            stats.correct,
            stats.total
        );
    }

    println!("\n--- Bottom 5 Categories ---");
    for (cat, stats) in cats_sorted.iter().rev().take(5) {
        println!(
            "  {:<30} {:.1}% ({}/{})",
            cat,
            stats.accuracy * 100.0,
            stats.correct,
            stats.total
        );
    }

    // --- Save results JSON ---
    let results = BenchmarkResults {
        benchmark: "TruthfulQA".to_string(),
        method: "HDC BinaryHV (16,384-bit, word-bundle encoding)".to_string(),
        total_questions: n,
        mc1_correct,
        mc1_accuracy,
        mc1_any_correct,
        mc1_any_accuracy,
        random_baseline: 0.5,
        elapsed_secs: elapsed,
        avg_correct_similarity: avg_correct_sim,
        avg_incorrect_similarity: avg_incorrect_sim,
        similarity_gap,
        per_category: per_category.clone(),
        per_type: per_type.clone(),
    };

    let results_path = Path::new(RESULTS_PATH);
    if let Some(parent) = results_path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let json = serde_json::to_string_pretty(&results).expect("Failed to serialize results");
    let mut f = File::create(results_path).expect("Failed to create results file");
    f.write_all(json.as_bytes())
        .expect("Failed to write results");

    println!("\nResults saved to {}", RESULTS_PATH);
    println!(
        "\nConclusion: HDC bag-of-words achieves {:.1}% MC1 on TruthfulQA (random = 50%).",
        mc1_accuracy * 100.0
    );
    println!("This measures lexical overlap between question and answer representations.");
    println!("Higher performance requires semantic understanding beyond word-level encoding.");
}