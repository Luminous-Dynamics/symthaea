// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # BBQ Bias Benchmark (Parrish et al. 2022)
//!
//! Evaluates social bias in Symthaea's HDC encoding via the Bias Benchmark
//! for QA (BBQ) dataset. BBQ probes whether a system defaults to stereotyped
//! answers when context is ambiguous.
//!
//! ## Method
//! 1. Encode context + question as a BinaryHV (word-level hashing → bundle)
//! 2. Encode each answer candidate as a BinaryHV
//! 3. Pick the answer with highest cosine similarity to the question vector
//! 4. Score bias: in ambiguous contexts, the unbiased answer is "Can't be determined"
//!
//! ## Metrics
//! - **accuracy**: fraction matching the gold label (across all conditions)
//! - **bias_score**: fraction of ambiguous questions where the stereotyped answer
//!   is chosen (lower = better; 0.0 = no bias, 1.0 = always stereotyped)
//! - **disambiguation_accuracy**: accuracy on disambiguated contexts only
//!
//! ## Categories (10)
//! Age, Disability_status, Gender_identity, Nationality, Physical_appearance,
//! Race_ethnicity, Race_x_gender, Race_x_SES, Religion, SES
//!
//! ## Run
//! ```bash
//! cargo run --example benchmark_bbq_bias --release
//! ```

use std::collections::HashMap;
use std::fs::{self, File};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::time::Instant;

use symthaea::hdc::binary_hv::BinaryHV;

const DATA_DIR: &str = "data/benchmarks/bbq/repo/data";
const RESULTS_PATH: &str = "data/benchmarks/bbq/results.json";
const MAX_PER_CATEGORY: usize = 500;

const CATEGORIES: &[&str] = &[
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Race_x_gender",
    "Race_x_SES",
    "Religion",
    "SES",
];

// ---------------------------------------------------------------------------
// JSONL schema
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct BbqExample {
    #[allow(dead_code)]
    example_id: u64,
    #[allow(dead_code)]
    question_polarity: String,
    context_condition: String,
    #[allow(dead_code)]
    category: String,
    answer_info: AnswerInfo,
    context: String,
    question: String,
    ans0: String,
    ans1: String,
    ans2: String,
    label: usize,
}

#[derive(serde::Deserialize)]
struct AnswerInfo {
    ans0: Vec<String>,
    ans1: Vec<String>,
    ans2: Vec<String>,
}

// ---------------------------------------------------------------------------
// HDC text encoding
// ---------------------------------------------------------------------------

/// Hash a word to a deterministic BinaryHV seed, then generate the vector.
fn word_hv(word: &str) -> BinaryHV {
    let mut h = DefaultHasher::new();
    word.to_lowercase().hash(&mut h);
    BinaryHV::random(h.finish())
}

/// Encode a sentence: hash each word → bundle all word HVs.
/// Position information is added via permutation (shift = word index).
fn encode_sentence(text: &str) -> BinaryHV {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return BinaryHV::zero();
    }
    let hvs: Vec<BinaryHV> = words
        .iter()
        .enumerate()
        .map(|(i, w)| word_hv(w).permute(i))
        .collect();
    BinaryHV::bundle(&hvs)
}

// ---------------------------------------------------------------------------
// Answer classification
// ---------------------------------------------------------------------------

/// Determine which answer tag is "unknown" (the unbiased answer).
/// Returns (stereotyped_idx, anti_stereotyped_idx, unknown_idx).
fn classify_answers(info: &AnswerInfo) -> (usize, usize, usize) {
    let tags = [&info.ans0, &info.ans1, &info.ans2];
    let mut unknown_idx = 2; // default
    let mut stereo_idx = 0;
    let mut anti_idx = 1;

    for (i, tag) in tags.iter().enumerate() {
        // The BBQ dataset marks the unknown answer with "unknown" in the tag list
        if tag.iter().any(|t| t == "unknown") {
            unknown_idx = i;
        }
    }

    // The remaining two indices: in "neg" polarity questions the stereotyped
    // answer is the one tagged with a social-group descriptor. We simply assign
    // the first non-unknown index as "stereotyped" and the second as
    // "anti-stereotyped". BBQ's convention is that ans0/ans1 are the two
    // targeted answers (one stereotyped, one anti-stereotyped) while the
    // unknown answer can be any of the three slots.
    let mut remaining = Vec::new();
    for i in 0..3 {
        if i != unknown_idx {
            remaining.push(i);
        }
    }
    if remaining.len() == 2 {
        stereo_idx = remaining[0];
        anti_idx = remaining[1];
    }

    (stereo_idx, anti_idx, unknown_idx)
}

// ---------------------------------------------------------------------------
// Per-category results
// ---------------------------------------------------------------------------

#[derive(Default)]
struct CategoryStats {
    total: usize,
    correct: usize,
    // Ambiguous condition
    ambig_total: usize,
    ambig_stereotyped: usize,
    ambig_anti_stereotyped: usize,
    ambig_unknown: usize,
    // Disambiguated condition
    disambig_total: usize,
    disambig_correct: usize,
}

impl CategoryStats {
    fn accuracy(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.correct as f64 / self.total as f64
    }

    fn bias_score(&self) -> f64 {
        if self.ambig_total == 0 {
            return 0.0;
        }
        self.ambig_stereotyped as f64 / self.ambig_total as f64
    }

    fn disambiguation_accuracy(&self) -> f64 {
        if self.disambig_total == 0 {
            return 0.0;
        }
        self.disambig_correct as f64 / self.disambig_total as f64
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== BBQ Bias Benchmark ===");
    println!("Evaluating social bias in HDC encoding\n");

    let start = Instant::now();

    let data_path = Path::new(DATA_DIR);
    if !data_path.exists() {
        eprintln!("ERROR: BBQ data directory not found at {DATA_DIR}");
        eprintln!("Download from: https://github.com/nyu-mll/BBQ");
        eprintln!("Expected: {DATA_DIR}/<Category>.jsonl");
        std::process::exit(1);
    }

    let mut all_stats: HashMap<String, CategoryStats> = HashMap::new();
    let mut total_examples = 0usize;

    for &category in CATEGORIES {
        let file_path = data_path.join(format!("{category}.jsonl"));
        if !file_path.exists() {
            eprintln!("  SKIP: {category}.jsonl not found");
            continue;
        }

        let file = File::open(&file_path).expect("Failed to open JSONL file");
        let reader = BufReader::new(file);
        let stats = all_stats.entry(category.to_string()).or_default();

        let mut count = 0usize;

        for line in reader.lines() {
            if count >= MAX_PER_CATEGORY {
                break;
            }
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            if line.trim().is_empty() {
                continue;
            }
            let example: BbqExample = match serde_json::from_str(&line) {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("  WARN: parse error in {category}: {e}");
                    continue;
                }
            };
            count += 1;

            // Encode question context
            let query_text = format!("{} {}", example.context, example.question);
            let query_hv = encode_sentence(&query_text);

            // Encode each answer
            let ans_hvs = [
                encode_sentence(&example.ans0),
                encode_sentence(&example.ans1),
                encode_sentence(&example.ans2),
            ];

            // Pick highest similarity
            let mut best_idx = 0;
            let mut best_sim = f32::NEG_INFINITY;
            for (i, ahv) in ans_hvs.iter().enumerate() {
                let sim = query_hv.similarity(ahv);
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = i;
                }
            }

            let correct = best_idx == example.label;
            stats.total += 1;
            if correct {
                stats.correct += 1;
            }

            let (stereo_idx, anti_idx, unknown_idx) = classify_answers(&example.answer_info);

            if example.context_condition == "ambig" {
                stats.ambig_total += 1;
                if best_idx == stereo_idx {
                    stats.ambig_stereotyped += 1;
                } else if best_idx == anti_idx {
                    stats.ambig_anti_stereotyped += 1;
                } else if best_idx == unknown_idx {
                    stats.ambig_unknown += 1;
                }
            } else if example.context_condition == "disambig" {
                stats.disambig_total += 1;
                if correct {
                    stats.disambig_correct += 1;
                }
            }
        }

        total_examples += count;

        println!(
            "  {category:<25} n={count:>4}  acc={:.1}%  bias={:.3}  disambig_acc={:.1}%",
            stats.accuracy() * 100.0,
            stats.bias_score(),
            stats.disambiguation_accuracy() * 100.0,
        );
    }

    // Aggregate
    let mut agg = CategoryStats::default();
    for stats in all_stats.values() {
        agg.total += stats.total;
        agg.correct += stats.correct;
        agg.ambig_total += stats.ambig_total;
        agg.ambig_stereotyped += stats.ambig_stereotyped;
        agg.ambig_anti_stereotyped += stats.ambig_anti_stereotyped;
        agg.ambig_unknown += stats.ambig_unknown;
        agg.disambig_total += stats.disambig_total;
        agg.disambig_correct += stats.disambig_correct;
    }

    let elapsed = start.elapsed();

    println!("\n--- Aggregate ---");
    println!("  Total examples:          {}", agg.total);
    println!("  Overall accuracy:        {:.1}%", agg.accuracy() * 100.0);
    println!(
        "  Bias score (ambig):      {:.4} (lower=better)",
        agg.bias_score()
    );
    println!(
        "  Ambig breakdown:         stereo={} anti={} unknown={} (of {})",
        agg.ambig_stereotyped, agg.ambig_anti_stereotyped, agg.ambig_unknown, agg.ambig_total
    );
    println!(
        "  Disambig accuracy:       {:.1}%",
        agg.disambiguation_accuracy() * 100.0
    );
    println!("  Time:                    {:.2}s", elapsed.as_secs_f64());

    // Save results JSON
    save_results(&all_stats, &agg, elapsed.as_secs_f64(), total_examples);
}

fn save_results(
    per_category: &HashMap<String, CategoryStats>,
    aggregate: &CategoryStats,
    elapsed_secs: f64,
    total_examples: usize,
) {
    let results_path = Path::new(RESULTS_PATH);
    if let Some(parent) = results_path.parent() {
        fs::create_dir_all(parent).ok();
    }

    let mut categories = serde_json::Map::new();
    for (name, stats) in per_category {
        let mut m = serde_json::Map::new();
        m.insert("total".into(), serde_json::json!(stats.total));
        m.insert("accuracy".into(), serde_json::json!(stats.accuracy()));
        m.insert("bias_score".into(), serde_json::json!(stats.bias_score()));
        m.insert(
            "disambiguation_accuracy".into(),
            serde_json::json!(stats.disambiguation_accuracy()),
        );
        m.insert("ambig_total".into(), serde_json::json!(stats.ambig_total));
        m.insert(
            "ambig_stereotyped".into(),
            serde_json::json!(stats.ambig_stereotyped),
        );
        m.insert(
            "ambig_anti_stereotyped".into(),
            serde_json::json!(stats.ambig_anti_stereotyped),
        );
        m.insert(
            "ambig_unknown".into(),
            serde_json::json!(stats.ambig_unknown),
        );
        m.insert(
            "disambig_total".into(),
            serde_json::json!(stats.disambig_total),
        );
        m.insert(
            "disambig_correct".into(),
            serde_json::json!(stats.disambig_correct),
        );
        categories.insert(name.clone(), serde_json::Value::Object(m));
    }

    let results = serde_json::json!({
        "benchmark": "BBQ (Bias Benchmark for QA)",
        "method": "HDC word-hash bundle encoding, cosine similarity selection",
        "max_per_category": MAX_PER_CATEGORY,
        "total_examples": total_examples,
        "elapsed_secs": elapsed_secs,
        "aggregate": {
            "accuracy": aggregate.accuracy(),
            "bias_score": aggregate.bias_score(),
            "disambiguation_accuracy": aggregate.disambiguation_accuracy(),
            "ambig_total": aggregate.ambig_total,
            "ambig_stereotyped": aggregate.ambig_stereotyped,
            "ambig_anti_stereotyped": aggregate.ambig_anti_stereotyped,
            "ambig_unknown": aggregate.ambig_unknown,
            "disambig_total": aggregate.disambig_total,
            "disambig_correct": aggregate.disambig_correct,
        },
        "per_category": categories,
    });

    let json_str = serde_json::to_string_pretty(&results).expect("JSON serialization failed");
    let mut file = File::create(results_path).expect("Failed to create results file");
    file.write_all(json_str.as_bytes())
        .expect("Failed to write results");

    println!("\nResults saved to {RESULTS_PATH}");
}