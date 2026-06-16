// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # MMLU (Massive Multitask Language Understanding) Benchmark
//!
//! Tests Symthaea's HDC encoding on the MMLU benchmark (Hendrycks et al. 2021),
//! which evaluates knowledge and reasoning across 57 subjects.
//!
//! ## Method
//! 1. Hash each unique word/token to a deterministic basis BinaryHV (BLAKE3 seed)
//! 2. Encode sentences as `BinaryHV::bundle(&word_hvs)` (bag-of-words)
//! 3. For each 4-choice question: encode Q, encode each choice, pick argmax similarity(Q, A)
//! 4. Report per-subject and overall accuracy
//!
//! ## Significance
//! This is an honest baseline measuring whether HDC bag-of-words encoding captures
//! enough semantic structure for multiple-choice QA. Random baseline = 25% (4 choices).
//! Any accuracy significantly above 25% indicates that word-overlap and distributional
//! similarity in HDC space carry signal for this task.
//!
//! ## Dataset
//! MMLU (Hendrycks 2021): 14,042 test questions across 57 subjects
//! Format: JSONL, each line `{"question":"...","subject":"...","choices":["A","B","C","D"],"answer":1}`
//!
//! ## Run
//! ```bash
//! cargo run --example benchmark_mmlu --release
//! # Run all 14,042 questions:
//! SYMTHAEA_MMLU_ALL=1 cargo run --example benchmark_mmlu --release
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter};
use std::path::PathBuf;
use std::time::Instant;
use symthaea::hdc::binary_hv::BinaryHV;

/// Maximum samples when not running the full dataset
const MAX_SAMPLES: usize = 2000;

/// Path to MMLU test data
fn mmlu_data_path() -> PathBuf {
    env::var("SYMTHAEA_MMLU_DATA_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data/benchmarks/mmlu/test.jsonl"))
}

/// Path to write results
fn mmlu_results_path() -> PathBuf {
    env::var("SYMTHAEA_MMLU_RESULTS_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data/benchmarks/mmlu/results.json"))
}

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Deserialize)]
struct MmluQuestion {
    question: String,
    subject: String,
    choices: Vec<String>,
    answer: usize,
}

#[derive(Debug, Serialize)]
struct SubjectResult {
    subject: String,
    correct: usize,
    total: usize,
    accuracy: f64,
}

#[derive(Debug, Serialize)]
struct BenchmarkResults {
    benchmark: String,
    method: String,
    total_questions: usize,
    total_correct: usize,
    overall_accuracy: f64,
    random_baseline: f64,
    elapsed_seconds: f64,
    per_subject: Vec<SubjectResult>,
}

// ============================================================================
// HDC Encoding
// ============================================================================

/// Token-to-HV cache using BLAKE3 hash as deterministic seed.
struct WordEncoder {
    cache: HashMap<String, BinaryHV>,
}

impl WordEncoder {
    fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Get or create a basis HV for a word. Uses BLAKE3 hash of the lowercase
    /// token as a u64 seed for deterministic, content-addressed generation.
    fn encode_word(&mut self, word: &str) -> BinaryHV {
        let lower = word.to_lowercase();
        if let Some(hv) = self.cache.get(&lower) {
            return hv.clone();
        }
        let hash = blake3::hash(lower.as_bytes());
        let seed = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
        let hv = BinaryHV::random(seed);
        self.cache.insert(lower, hv.clone());
        hv
    }

    /// Encode a sentence as a bag-of-words bundle of word HVs.
    /// Returns None if the text has no alphanumeric tokens.
    fn encode_sentence(&mut self, text: &str) -> Option<BinaryHV> {
        let word_hvs: Vec<BinaryHV> = tokenize(text).iter().map(|w| self.encode_word(w)).collect();
        if word_hvs.is_empty() {
            return None;
        }
        Some(BinaryHV::bundle(&word_hvs))
    }
}

/// Simple whitespace + punctuation tokenizer. Strips non-alphanumeric chars,
/// lowercases, and filters empty tokens.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| c.is_whitespace() || c == '/' || c == '-')
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .collect::<String>()
        })
        .map(|w| w.to_lowercase())
        .filter(|w| !w.is_empty())
        .collect()
}

// ============================================================================
// Benchmark Logic
// ============================================================================

/// For a given question, encode Q and each choice, return the index of the
/// choice with highest similarity to the question HV.
fn predict_answer(encoder: &mut WordEncoder, question: &MmluQuestion) -> Option<usize> {
    let q_hv = encoder.encode_sentence(&question.question)?;

    let mut best_idx = 0;
    let mut best_sim = f32::NEG_INFINITY;

    for (i, choice) in question.choices.iter().enumerate() {
        // Encode the choice text
        let choice_hv = match encoder.encode_sentence(choice) {
            Some(hv) => hv,
            None => continue,
        };
        // Score = similarity between question and choice encodings
        let sim = q_hv.similarity(&choice_hv);
        if sim > best_sim {
            best_sim = sim;
            best_idx = i;
        }
    }

    Some(best_idx)
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let run_all = env::var("SYMTHAEA_MMLU_ALL").is_ok();
    let max_samples = if run_all { usize::MAX } else { MAX_SAMPLES };

    println!("=== MMLU Benchmark (HDC Bag-of-Words Baseline) ===");
    println!();

    // --- Load data ---
    let data_path = mmlu_data_path();
    if !data_path.exists() {
        eprintln!("ERROR: MMLU data not found at {}", data_path.display());
        eprintln!("Place test.jsonl in data/benchmarks/mmlu/ or set SYMTHAEA_MMLU_DATA_PATH");
        std::process::exit(1);
    }

    println!("Loading data from {} ...", data_path.display());
    let file = File::open(&data_path).expect("Failed to open MMLU data file");
    let reader = BufReader::new(file);

    let mut questions: Vec<MmluQuestion> = Vec::new();
    let mut parse_errors = 0usize;

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => {
                parse_errors += 1;
                continue;
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<MmluQuestion>(trimmed) {
            Ok(q) => {
                if q.answer < q.choices.len() {
                    questions.push(q);
                } else {
                    parse_errors += 1;
                }
            }
            Err(_) => {
                parse_errors += 1;
            }
        }
        if questions.len() >= max_samples {
            break;
        }
    }

    if questions.is_empty() {
        eprintln!("ERROR: No valid questions loaded (parse errors: {parse_errors})");
        std::process::exit(1);
    }

    let total_available = questions.len();
    println!(
        "Loaded {total_available} questions ({parse_errors} parse errors, cap={}{}).",
        if run_all { "none" } else { "" },
        if run_all {
            String::new()
        } else {
            format!("{MAX_SAMPLES}")
        },
    );

    // --- Collect subjects ---
    let subjects: Vec<String> = {
        let mut s: Vec<String> = questions.iter().map(|q| q.subject.clone()).collect();
        s.sort();
        s.dedup();
        s
    };
    println!("Subjects: {}", subjects.len());
    println!();

    // --- Run benchmark ---
    let start = Instant::now();
    let mut encoder = WordEncoder::new();

    // Per-subject tracking
    let mut subject_correct: HashMap<String, usize> = HashMap::new();
    let mut subject_total: HashMap<String, usize> = HashMap::new();

    let mut total_correct = 0usize;
    let mut total_evaluated = 0usize;

    for (i, q) in questions.iter().enumerate() {
        if let Some(predicted) = predict_answer(&mut encoder, q) {
            *subject_total.entry(q.subject.clone()).or_insert(0) += 1;
            total_evaluated += 1;
            if predicted == q.answer {
                *subject_correct.entry(q.subject.clone()).or_insert(0) += 1;
                total_correct += 1;
            }
        }

        // Progress reporting every 500 questions
        if (i + 1) % 500 == 0 || i + 1 == questions.len() {
            let elapsed = start.elapsed().as_secs_f64();
            let qps = (i + 1) as f64 / elapsed;
            print!(
                "\r  [{}/{}] {:.0} q/s, running accuracy: {:.1}%    ",
                i + 1,
                questions.len(),
                qps,
                if total_evaluated > 0 {
                    100.0 * total_correct as f64 / total_evaluated as f64
                } else {
                    0.0
                },
            );
        }
    }
    println!();

    let elapsed = start.elapsed().as_secs_f64();

    // --- Compile results ---
    let overall_accuracy = if total_evaluated > 0 {
        total_correct as f64 / total_evaluated as f64
    } else {
        0.0
    };

    let mut per_subject: Vec<SubjectResult> = subjects
        .iter()
        .filter_map(|s| {
            let total = *subject_total.get(s)?;
            let correct = *subject_correct.get(s).unwrap_or(&0);
            Some(SubjectResult {
                subject: s.clone(),
                correct,
                total,
                accuracy: if total > 0 {
                    correct as f64 / total as f64
                } else {
                    0.0
                },
            })
        })
        .collect();

    // Sort by accuracy descending
    per_subject.sort_by(|a, b| b.accuracy.partial_cmp(&a.accuracy).unwrap());

    let results = BenchmarkResults {
        benchmark: "MMLU".to_string(),
        method: "HDC bag-of-words (BLAKE3 random projection, argmax cosine similarity)".to_string(),
        total_questions: total_evaluated,
        total_correct,
        overall_accuracy,
        random_baseline: 0.25,
        elapsed_seconds: elapsed,
        per_subject,
    };

    // --- Print summary ---
    println!();
    println!("=== Results ===");
    println!(
        "Overall: {}/{} correct = {:.2}% (random baseline: 25.00%)",
        total_correct,
        total_evaluated,
        overall_accuracy * 100.0,
    );
    println!(
        "Elapsed: {elapsed:.1}s ({:.0} questions/sec)",
        total_evaluated as f64 / elapsed
    );
    println!("Vocabulary size: {} unique tokens", encoder.cache.len());
    println!();

    // Top and bottom 5 subjects
    println!("Top 5 subjects:");
    for sr in results.per_subject.iter().take(5) {
        println!(
            "  {:<40} {}/{} = {:.1}%",
            sr.subject,
            sr.correct,
            sr.total,
            sr.accuracy * 100.0,
        );
    }
    println!();
    println!("Bottom 5 subjects:");
    for sr in results.per_subject.iter().rev().take(5) {
        println!(
            "  {:<40} {}/{} = {:.1}%",
            sr.subject,
            sr.correct,
            sr.total,
            sr.accuracy * 100.0,
        );
    }

    // --- Save results ---
    let results_path = mmlu_results_path();
    if let Some(parent) = results_path.parent() {
        fs::create_dir_all(parent).ok();
    }
    match File::create(&results_path) {
        Ok(f) => {
            let writer = BufWriter::new(f);
            if let Err(e) = serde_json::to_writer_pretty(writer, &results) {
                eprintln!("WARNING: Failed to write results JSON: {e}");
            } else {
                println!();
                println!("Results saved to {}", results_path.display());
            }
        }
        Err(e) => {
            eprintln!(
                "WARNING: Could not create results file {}: {e}",
                results_path.display()
            );
        }
    }
}