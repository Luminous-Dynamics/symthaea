// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # MMLU Trained-Prototype Benchmark
//!
//! Trains HDC prototypes on MMLU training data and evaluates on held-out questions.
//!
//! ## Method
//! 1. Load MMLU data (auxiliary_train parquet or test.jsonl with 80/20 split)
//! 2. For each subject in the training set:
//!    - Encode `question + " " + correct_answer` as a BinaryHV for each question
//!    - Bundle all correct-answer encodings → per-subject prototype
//! 3. At evaluation time, for each question:
//!    - Encode `question + " " + choice_i` for each of the 4 choices
//!    - Pick the choice whose encoding has highest similarity to the subject prototype
//! 4. Report per-subject and overall accuracy
//!
//! ## Significance
//! This demonstrates HDC's associative memory: by bundling many question+answer pairs
//! for a subject, the prototype captures the "semantic fingerprint" of correct answers
//! in that domain. At test time, the correct answer choice should resonate more strongly
//! with the prototype than distractors.
//!
//! ## Dataset
//! MMLU (Hendrycks 2021): 14,042 test questions across 57 subjects
//! Format: JSONL, each line `{"question":"...","subject":"...","choices":["A","B","C","D"],"answer":1}`
//!
//! ## Run
//! ```bash
//! cargo run --example benchmark_mmlu_trained --release
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter};
use std::path::PathBuf;
use std::time::Instant;
use symthaea::hdc::binary_hv::BinaryHV;

/// Maximum questions to process
const MAX_SAMPLES: usize = 2000;

/// Train/eval split ratio (fraction used for training)
const TRAIN_FRACTION: f64 = 0.8;

/// Path to MMLU test data
fn mmlu_data_path() -> PathBuf {
    env::var("SYMTHAEA_MMLU_DATA_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data/benchmarks/mmlu/test.jsonl"))
}

/// Path to write results
fn mmlu_results_path() -> PathBuf {
    PathBuf::from("data/benchmarks/mmlu/results_trained.json")
}

/// Path to save/load prototypes
fn prototypes_path() -> PathBuf {
    PathBuf::from("data/benchmarks/mmlu/prototypes.json")
}

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Deserialize, Clone)]
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
    train_examples: usize,
}

#[derive(Debug, Serialize)]
struct BenchmarkResults {
    benchmark: String,
    method: String,
    total_questions: usize,
    total_correct: usize,
    overall_accuracy: f64,
    random_baseline: f64,
    train_count: usize,
    eval_count: usize,
    subjects_trained: usize,
    elapsed_seconds: f64,
    per_subject: Vec<SubjectResult>,
}

// ============================================================================
// HDC Encoding (FNV-1a deterministic word seeding)
// ============================================================================

/// FNV-1a hash for deterministic word-to-seed mapping.
fn word_to_seed(word: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in word.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Encode text as a bag-of-words BinaryHV bundle using FNV-1a seeding.
/// Filters tokens shorter than 2 characters.
fn encode_text(text: &str) -> Option<BinaryHV> {
    let words: Vec<BinaryHV> = text
        .split_whitespace()
        .filter(|w| w.len() > 1)
        .map(|w| BinaryHV::random(word_to_seed(&w.to_lowercase())))
        .collect();
    if words.is_empty() {
        return None;
    }
    Some(BinaryHV::bundle(&words))
}

/// Encode a question concatenated with a specific answer choice.
fn encode_question_answer(question: &str, answer: &str) -> Option<BinaryHV> {
    let combined = format!("{} {}", question, answer);
    encode_text(&combined)
}

// ============================================================================
// Prototype Training
// ============================================================================

/// Train per-subject prototypes from training questions.
/// Returns a map from subject name to (prototype HV, training count).
fn train_prototypes(train_questions: &[MmluQuestion]) -> HashMap<String, (BinaryHV, usize)> {
    // Group questions by subject
    let mut subject_hvs: HashMap<String, Vec<BinaryHV>> = HashMap::new();

    for q in train_questions {
        if q.answer >= q.choices.len() {
            continue;
        }
        let correct_answer = &q.choices[q.answer];
        if let Some(hv) = encode_question_answer(&q.question, correct_answer) {
            subject_hvs.entry(q.subject.clone()).or_default().push(hv);
        }
    }

    // Bundle all question+correct_answer encodings per subject into a prototype
    let mut prototypes: HashMap<String, (BinaryHV, usize)> = HashMap::new();
    for (subject, hvs) in &subject_hvs {
        let count = hvs.len();
        if count == 0 {
            continue;
        }
        let prototype = BinaryHV::bundle(hvs);
        prototypes.insert(subject.clone(), (prototype, count));
    }

    prototypes
}

/// Predict the answer for a question using the trained subject prototype.
/// Returns None if the subject has no prototype or encoding fails.
fn predict_with_prototype(
    question: &MmluQuestion,
    prototypes: &HashMap<String, (BinaryHV, usize)>,
) -> Option<usize> {
    let (prototype, _) = prototypes.get(&question.subject)?;

    let mut best_idx = 0;
    let mut best_sim = f32::NEG_INFINITY;

    for (i, choice) in question.choices.iter().enumerate() {
        let candidate_hv = encode_question_answer(&question.question, choice)?;
        let sim = candidate_hv.similarity(prototype);
        if sim > best_sim {
            best_sim = sim;
            best_idx = i;
        }
    }

    Some(best_idx)
}

// ============================================================================
// Prototype Persistence
// ============================================================================

/// Save prototype metadata (subjects and training counts) to JSON.
/// Since BinaryHV doesn't implement Serialize, we store enough info to
/// rebuild prototypes from the training data.
fn save_prototype_metadata(
    prototypes: &HashMap<String, (BinaryHV, usize)>,
    train_questions: &[MmluQuestion],
    path: &PathBuf,
) {
    // Store per-subject training question indices for reproducibility
    #[derive(Serialize)]
    struct PrototypeEntry {
        subject: String,
        train_count: usize,
        /// Indices into the training data for this subject
        question_indices: Vec<usize>,
    }

    #[derive(Serialize)]
    struct PrototypesManifest {
        method: String,
        encoding: String,
        total_train_questions: usize,
        subjects: Vec<PrototypeEntry>,
    }

    let mut entries: Vec<PrototypeEntry> = Vec::new();
    for (subject, (_, count)) in prototypes {
        let indices: Vec<usize> = train_questions
            .iter()
            .enumerate()
            .filter(|(_, q)| q.subject == *subject)
            .map(|(i, _)| i)
            .collect();
        entries.push(PrototypeEntry {
            subject: subject.clone(),
            train_count: *count,
            question_indices: indices,
        });
    }
    entries.sort_by(|a, b| a.subject.cmp(&b.subject));

    let manifest = PrototypesManifest {
        method: "HDC trained prototypes (FNV-1a bag-of-words, per-subject bundle)".to_string(),
        encoding: "BinaryHV 16384-bit, FNV-1a word seeding, bundle(question + correct_answer)"
            .to_string(),
        total_train_questions: train_questions.len(),
        subjects: entries,
    };

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    match File::create(path) {
        Ok(f) => {
            let writer = BufWriter::new(f);
            if let Err(e) = serde_json::to_writer_pretty(writer, &manifest) {
                eprintln!("WARNING: Failed to write prototypes JSON: {e}");
            } else {
                println!("Prototypes saved to {}", path.display());
            }
        }
        Err(e) => {
            eprintln!(
                "WARNING: Could not create prototypes file {}: {e}",
                path.display()
            );
        }
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("=== MMLU Trained-Prototype Benchmark ===");
    println!("Method: HDC per-subject prototypes (FNV-1a bag-of-words)");
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
        if questions.len() >= MAX_SAMPLES {
            break;
        }
    }

    if questions.is_empty() {
        eprintln!("ERROR: No valid questions loaded (parse errors: {parse_errors})");
        std::process::exit(1);
    }

    println!(
        "Loaded {} questions ({} parse errors, cap={}).",
        questions.len(),
        parse_errors,
        MAX_SAMPLES,
    );

    // --- Split into train (80%) and eval (20%) ---
    let train_count = (questions.len() as f64 * TRAIN_FRACTION) as usize;
    let train_questions = &questions[..train_count];
    let eval_questions = &questions[train_count..];

    println!(
        "Split: {} train, {} eval ({:.0}%/{:.0}%)",
        train_questions.len(),
        eval_questions.len(),
        TRAIN_FRACTION * 100.0,
        (1.0 - TRAIN_FRACTION) * 100.0,
    );

    // Collect unique subjects
    let subjects: Vec<String> = {
        let mut s: Vec<String> = questions.iter().map(|q| q.subject.clone()).collect();
        s.sort();
        s.dedup();
        s
    };
    println!("Subjects: {}", subjects.len());
    println!();

    // --- Train prototypes ---
    println!("Training per-subject prototypes...");
    let train_start = Instant::now();
    let prototypes = train_prototypes(train_questions);
    let train_elapsed = train_start.elapsed().as_secs_f64();

    println!(
        "Trained {} subject prototypes in {:.2}s",
        prototypes.len(),
        train_elapsed,
    );

    // Report training stats per subject
    let mut train_counts: Vec<(&String, usize)> =
        prototypes.iter().map(|(s, (_, c))| (s, *c)).collect();
    train_counts.sort_by(|a, b| b.1.cmp(&a.1));

    println!("Training examples per subject (top 5):");
    for (subject, count) in train_counts.iter().take(5) {
        println!("  {:<40} {}", subject, count);
    }
    if train_counts.len() > 5 {
        println!("  ... and {} more subjects", train_counts.len() - 5);
    }
    println!();

    // --- Save prototypes ---
    let proto_path = prototypes_path();
    save_prototype_metadata(&prototypes, train_questions, &proto_path);
    println!();

    // --- Evaluate ---
    println!(
        "Evaluating on {} held-out questions...",
        eval_questions.len()
    );
    let eval_start = Instant::now();

    let mut subject_correct: HashMap<String, usize> = HashMap::new();
    let mut subject_total: HashMap<String, usize> = HashMap::new();
    let mut total_correct = 0usize;
    let mut total_evaluated = 0usize;
    let mut no_prototype_count = 0usize;

    for (i, q) in eval_questions.iter().enumerate() {
        if !prototypes.contains_key(&q.subject) {
            no_prototype_count += 1;
            continue;
        }

        if let Some(predicted) = predict_with_prototype(q, &prototypes) {
            *subject_total.entry(q.subject.clone()).or_insert(0) += 1;
            total_evaluated += 1;
            if predicted == q.answer {
                *subject_correct.entry(q.subject.clone()).or_insert(0) += 1;
                total_correct += 1;
            }
        }

        // Progress reporting
        if (i + 1) % 100 == 0 || i + 1 == eval_questions.len() {
            let elapsed = eval_start.elapsed().as_secs_f64();
            let qps = (i + 1) as f64 / elapsed;
            print!(
                "\r  [{}/{}] {:.0} q/s, running accuracy: {:.1}%    ",
                i + 1,
                eval_questions.len(),
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

    let eval_elapsed = eval_start.elapsed().as_secs_f64();
    let total_elapsed = train_elapsed + eval_elapsed;

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
            let train_ex = prototypes.get(s).map(|(_, c)| *c).unwrap_or(0);
            Some(SubjectResult {
                subject: s.clone(),
                correct,
                total,
                accuracy: if total > 0 {
                    correct as f64 / total as f64
                } else {
                    0.0
                },
                train_examples: train_ex,
            })
        })
        .collect();

    // Sort by accuracy descending
    per_subject.sort_by(|a, b| b.accuracy.partial_cmp(&a.accuracy).unwrap());

    let results = BenchmarkResults {
        benchmark: "MMLU (trained prototypes)".to_string(),
        method: "HDC per-subject prototype (FNV-1a bag-of-words, bundle question+correct_answer)"
            .to_string(),
        total_questions: total_evaluated,
        total_correct,
        overall_accuracy,
        random_baseline: 0.25,
        train_count: train_questions.len(),
        eval_count: eval_questions.len(),
        subjects_trained: prototypes.len(),
        elapsed_seconds: total_elapsed,
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
        "Train: {} questions, Eval: {} questions ({} skipped — no prototype)",
        train_questions.len(),
        total_evaluated,
        no_prototype_count,
    );
    println!(
        "Elapsed: {:.1}s total ({:.1}s train, {:.1}s eval, {:.0} eval q/s)",
        total_elapsed,
        train_elapsed,
        eval_elapsed,
        if eval_elapsed > 0.0 {
            total_evaluated as f64 / eval_elapsed
        } else {
            0.0
        },
    );
    println!();

    // Top and bottom 5 subjects
    println!("Top 5 subjects:");
    for sr in results.per_subject.iter().take(5) {
        println!(
            "  {:<40} {}/{} = {:.1}% (trained on {} examples)",
            sr.subject,
            sr.correct,
            sr.total,
            sr.accuracy * 100.0,
            sr.train_examples,
        );
    }
    println!();
    println!("Bottom 5 subjects:");
    for sr in results.per_subject.iter().rev().take(5) {
        println!(
            "  {:<40} {}/{} = {:.1}% (trained on {} examples)",
            sr.subject,
            sr.correct,
            sr.total,
            sr.accuracy * 100.0,
            sr.train_examples,
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