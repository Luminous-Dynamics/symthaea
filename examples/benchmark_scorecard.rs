// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Unified Benchmark Scorecard
//!
//! Runs all external dataset benchmarks and collects results into a single
//! publishable scorecard with per-domain breakdowns and LLM baselines.
//!
//! This complements the psych-bench suite (which tests cognitive architecture)
//! by measuring performance on standard ML/NLP/ethics benchmarks.
//!
//! ## Usage
//!
//! ```sh
//! cargo run --example benchmark_scorecard
//! cargo run --example benchmark_scorecard -- --quick    # Subset for CI
//! cargo run --example benchmark_scorecard -- --full     # All samples
//! ```
//!
//! ## Output
//!
//! - Console: Markdown table summary
//! - `data/benchmarks/scorecard.json`: Full JSON results
//! - `data/benchmarks/SCORECARD.md`: Publishable markdown

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;
use symthaea::hdc::binary_hv::BinaryHV;
use symthaea::hdc::moral_algebra::{MoralAlgebra, MoralVerdict};
use symthaea::hdc::moral_parser::MoralParser;
// Trained prototype imports (used by ETHICS Trained via cached results)

// ============================================================================
// Configuration
// ============================================================================

const QUICK_SAMPLES: usize = 200;
const DEFAULT_SAMPLES: usize = 1000;
const FULL_SAMPLES: usize = usize::MAX;

// ============================================================================
// Scorecard Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScorecardResult {
    benchmark: String,
    domain: String,
    metric: String,
    value: f64,
    n_samples: usize,
    elapsed_ms: u64,
    notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LlmBaseline {
    model: String,
    value: f64,
    source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScorecardEntry {
    result: ScorecardResult,
    baselines: Vec<LlmBaseline>,
    random_baseline: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Scorecard {
    version: String,
    timestamp: String,
    entries: Vec<ScorecardEntry>,
    total_elapsed_ms: u64,
}

// ============================================================================
// HDC Encoding Utilities
// ============================================================================

fn word_to_seed(word: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in word.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn encode_text(text: &str) -> BinaryHV {
    let words: Vec<BinaryHV> = text
        .split_whitespace()
        .filter(|w| w.len() > 1)
        .map(|w| BinaryHV::random(word_to_seed(&w.to_lowercase())))
        .collect();
    if words.is_empty() {
        return BinaryHV::random(0);
    }
    BinaryHV::bundle(&words)
}

fn mc_score(question: &str, choices: &[&str]) -> usize {
    let q_hv = encode_text(question);
    let mut best_idx = 0;
    let mut best_sim = f32::NEG_INFINITY;
    for (i, choice) in choices.iter().enumerate() {
        // Encode the full question+choice as the candidate understanding
        let full = format!("{} {}", question, choice);
        let c_hv = encode_text(&full);
        let sim = q_hv.similarity(&c_hv);
        if sim > best_sim {
            best_sim = sim;
            best_idx = i;
        }
    }
    best_idx
}

// ============================================================================
// ETHICS Benchmark (Moral Algebra)
// ============================================================================

fn run_ethics(max_samples: usize) -> ScorecardResult {
    let start = Instant::now();
    let algebra = MoralAlgebra::default_dim();
    let parser = MoralParser::new();

    let datasets_path = Path::new("data/moral_datasets");
    let mut correct = 0usize;
    let mut total = 0usize;

    // Raw MoralAlgebra ensemble (no trained prototypes — see ETHICS Trained for full pipeline)
    if let Ok(file) = File::open(datasets_path.join("ethics.json")) {
        let reader = BufReader::new(file);
        if let Ok(data) = serde_json::from_reader::<_, serde_json::Value>(reader) {
            if let Some(examples) = data.get("examples").and_then(|e| e.as_array()) {
                for ex in examples.iter().take(max_samples) {
                    let text = ex.get("text").and_then(|t| t.as_str()).unwrap_or("");
                    let label = ex.get("label").and_then(|l| l.as_i64()).unwrap_or(0);
                    if text.is_empty() {
                        continue;
                    }

                    let parsed = parser.parse(text);
                    let judgment = algebra.judge_ensemble(None, parsed.intent, text);
                    let predicted = match judgment.final_verdict {
                        MoralVerdict::Good | MoralVerdict::Neutral => 1,
                        _ => 0,
                    };
                    if predicted == label as i32 {
                        correct += 1;
                    }
                    total += 1;
                }
            }
        }
    }

    let accuracy = if total > 0 {
        correct as f64 / total as f64
    } else {
        0.0
    };

    ScorecardResult {
        benchmark: "ETHICS Raw (Hendrycks 2021)".into(),
        domain: "Ethics".into(),
        metric: "accuracy".into(),
        value: accuracy,
        n_samples: total,
        elapsed_ms: start.elapsed().as_millis() as u64,
        notes: if total == 0 {
            vec!["No data loaded — run scripts/download_moral_datasets_v3.py".into()]
        } else {
            vec!["Raw MoralAlgebra ensemble (baseline, no keyword signals)".into()]
        },
    }
}

// ============================================================================
// MMLU Benchmark
// ============================================================================

#[derive(Deserialize)]
struct MmluQuestion {
    question: String,
    subject: String,
    choices: Vec<String>,
    answer: usize,
}

fn run_mmlu(max_samples: usize) -> ScorecardResult {
    let start = Instant::now();
    let path = Path::new("data/benchmarks/mmlu/test.jsonl");
    let mut correct = 0usize;
    let mut total = 0usize;

    if let Ok(file) = File::open(path) {
        let reader = BufReader::new(file);
        for line in reader.lines().take(max_samples).flatten() {
            if let Ok(q) = serde_json::from_str::<MmluQuestion>(&line) {
                let choices: Vec<&str> = q.choices.iter().map(|s| s.as_str()).collect();
                let predicted = mc_score(&q.question, &choices);
                if predicted == q.answer {
                    correct += 1;
                }
                total += 1;
            }
        }
    }

    ScorecardResult {
        benchmark: "MMLU (Hendrycks 2021)".into(),
        domain: "Knowledge".into(),
        metric: "accuracy".into(),
        value: if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        },
        n_samples: total,
        elapsed_ms: start.elapsed().as_millis() as u64,
        notes: vec!["HDC bag-of-words encoding (no trained model)".into()],
    }
}

// ============================================================================
// HellaSwag Benchmark
// ============================================================================

#[derive(Deserialize)]
struct HellaSwagItem {
    ctx: String,
    endings: Vec<String>,
    label: String,
}

fn run_hellaswag(max_samples: usize) -> ScorecardResult {
    let start = Instant::now();
    let path = Path::new("data/benchmarks/hellaswag/validation.jsonl");
    let mut correct = 0usize;
    let mut total = 0usize;

    if let Ok(file) = File::open(path) {
        let reader = BufReader::new(file);
        for line in reader.lines().take(max_samples).flatten() {
            if let Ok(item) = serde_json::from_str::<HellaSwagItem>(&line) {
                let label: usize = item.label.parse().unwrap_or(0);
                let ctx_hv = encode_text(&item.ctx);
                let mut best_idx = 0;
                let mut best_sim = f32::NEG_INFINITY;
                for (i, ending) in item.endings.iter().enumerate() {
                    let full = format!("{} {}", item.ctx, ending);
                    let e_hv = encode_text(&full);
                    let sim = ctx_hv.similarity(&e_hv);
                    if sim > best_sim {
                        best_sim = sim;
                        best_idx = i;
                    }
                }
                if best_idx == label {
                    correct += 1;
                }
                total += 1;
            }
        }
    }

    ScorecardResult {
        benchmark: "HellaSwag (Zellers 2019)".into(),
        domain: "Commonsense".into(),
        metric: "accuracy".into(),
        value: if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        },
        n_samples: total,
        elapsed_ms: start.elapsed().as_millis() as u64,
        notes: vec!["HDC bag-of-words encoding (no trained model)".into()],
    }
}

// ============================================================================
// GSM8K Benchmark
// ============================================================================

#[derive(Deserialize)]
struct Gsm8kItem {
    question: String,
    answer: String,
}

fn extract_gsm8k_answer(answer: &str) -> Option<f64> {
    // Extract number after ####
    if let Some(pos) = answer.rfind("####") {
        let num_str = answer[pos + 4..].trim().replace(',', "");
        num_str.parse::<f64>().ok()
    } else {
        None
    }
}

fn extract_numbers(text: &str) -> Vec<f64> {
    let mut nums = Vec::new();
    let mut current = String::new();
    let mut in_num = false;
    for ch in text.chars() {
        if ch.is_ascii_digit() || (ch == '.' && in_num) || (ch == '-' && !in_num) {
            current.push(ch);
            in_num = true;
        } else if ch == ',' && in_num {
            // skip commas in numbers
        } else {
            if in_num {
                if let Ok(n) = current.parse::<f64>() {
                    nums.push(n);
                }
                current.clear();
            }
            in_num = false;
        }
    }
    if !current.is_empty() {
        if let Ok(n) = current.parse::<f64>() {
            nums.push(n);
        }
    }
    nums
}

fn attempt_gsm8k_solve(question: &str) -> Option<f64> {
    let nums = extract_numbers(question);
    let q_lower = question.to_lowercase();

    if nums.len() < 2 {
        return None;
    }

    // Simple heuristic patterns
    if (q_lower.contains("total") || q_lower.contains("altogether") || q_lower.contains("sum"))
        && nums.len() >= 2
    {
        return Some(nums.iter().sum());
    }
    if (q_lower.contains("each") || q_lower.contains("per")) && q_lower.contains("how many") {
        if nums.len() >= 2 {
            return Some(nums[0] * nums[1]);
        }
    }
    if q_lower.contains("left") || q_lower.contains("remaining") || q_lower.contains("fewer") {
        if nums.len() >= 2 {
            return Some(nums[0] - nums[1]);
        }
    }

    None
}

fn run_gsm8k(max_samples: usize) -> ScorecardResult {
    let start = Instant::now();
    let path = Path::new("data/benchmarks/gsm8k/test.jsonl");
    let mut correct = 0usize;
    let mut attempted = 0usize;
    let mut total = 0usize;

    if let Ok(file) = File::open(path) {
        let reader = BufReader::new(file);
        for line in reader.lines().take(max_samples).flatten() {
            if let Ok(item) = serde_json::from_str::<Gsm8kItem>(&line) {
                total += 1;
                let gold = match extract_gsm8k_answer(&item.answer) {
                    Some(g) => g,
                    None => continue,
                };
                if let Some(predicted) = attempt_gsm8k_solve(&item.question) {
                    attempted += 1;
                    if (predicted - gold).abs() < 0.01 {
                        correct += 1;
                    }
                }
            }
        }
    }

    ScorecardResult {
        benchmark: "GSM8K (Cobbe 2021)".into(),
        domain: "Math".into(),
        metric: "accuracy".into(),
        value: if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        },
        n_samples: total,
        elapsed_ms: start.elapsed().as_millis() as u64,
        notes: vec![
            format!(
                "Attempted {}/{} via pattern matching (no LLM chain-of-thought)",
                attempted, total
            ),
            "Heuristic number extraction + operation detection".into(),
        ],
    }
}

// ============================================================================
// TruthfulQA Benchmark
// ============================================================================

fn run_truthfulqa(max_samples: usize) -> ScorecardResult {
    let start = Instant::now();
    let path = Path::new("data/benchmarks/truthfulqa/repo/TruthfulQA.csv");
    let mut correct = 0usize;
    let mut total = 0usize;

    if let Ok(content) = fs::read_to_string(path) {
        for line in content.lines().skip(1).take(max_samples) {
            // CSV: Type,Category,Question,Best Answer,Best Incorrect Answer,...
            let fields: Vec<&str> = line.splitn(8, ',').collect();
            if fields.len() < 5 {
                continue;
            }
            let question = fields[2].trim_matches('"');
            let best_correct = fields[3].trim_matches('"');
            let best_incorrect = fields[4].trim_matches('"');

            if question.is_empty() || best_correct.is_empty() || best_incorrect.is_empty() {
                continue;
            }

            let q_hv = encode_text(question);
            let correct_hv = encode_text(&format!("{} {}", question, best_correct));
            let incorrect_hv = encode_text(&format!("{} {}", question, best_incorrect));

            let sim_correct = q_hv.similarity(&correct_hv);
            let sim_incorrect = q_hv.similarity(&incorrect_hv);

            if sim_correct >= sim_incorrect {
                correct += 1;
            }
            total += 1;
        }
    }

    ScorecardResult {
        benchmark: "TruthfulQA (Lin 2022)".into(),
        domain: "Truthfulness".into(),
        metric: "mc1_accuracy".into(),
        value: if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        },
        n_samples: total,
        elapsed_ms: start.elapsed().as_millis() as u64,
        notes: vec!["MC1: best correct vs best incorrect, HDC similarity".into()],
    }
}

// ============================================================================
// BBQ Bias Benchmark
// ============================================================================

#[derive(Deserialize)]
struct BbqItem {
    context: Option<String>,
    question: Option<String>,
    ans0: Option<String>,
    ans1: Option<String>,
    ans2: Option<String>,
    label: Option<usize>,
    context_condition: Option<String>,
    answer_info: Option<serde_json::Value>,
    category: Option<String>,
}

fn run_bbq(max_samples: usize) -> ScorecardResult {
    let start = Instant::now();
    let bbq_dir = Path::new("data/benchmarks/bbq/repo/data");
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut ambig_unknown = 0usize;
    let mut ambig_total = 0usize;

    let categories = [
        "Age",
        "Disability_status",
        "Gender_identity",
        "Nationality",
        "Physical_appearance",
        "Race_ethnicity",
        "Religion",
        "SES",
    ];

    let per_cat = max_samples / categories.len().max(1);

    for cat in &categories {
        let path = bbq_dir.join(format!("{}.jsonl", cat));
        if let Ok(file) = File::open(&path) {
            let reader = BufReader::new(file);
            for line in reader.lines().take(per_cat).flatten() {
                if let Ok(item) = serde_json::from_str::<BbqItem>(&line) {
                    let context = item.context.as_deref().unwrap_or("");
                    let question = item.question.as_deref().unwrap_or("");
                    let answers = [
                        item.ans0.as_deref().unwrap_or(""),
                        item.ans1.as_deref().unwrap_or(""),
                        item.ans2.as_deref().unwrap_or(""),
                    ];
                    let label = item.label.unwrap_or(2);

                    let full_q = format!("{} {}", context, question);
                    let choices: Vec<&str> = answers.to_vec();
                    let predicted = mc_score(&full_q, &choices);

                    if predicted == label {
                        correct += 1;
                    }
                    total += 1;

                    // Track ambiguous context bias
                    if item.context_condition.as_deref() == Some("ambig") {
                        ambig_total += 1;
                        // In ambiguous contexts, ans2 is typically "unknown"
                        if predicted == 2 {
                            ambig_unknown += 1;
                        }
                    }
                }
            }
        }
    }

    let bias_score = if ambig_total > 0 {
        1.0 - (ambig_unknown as f64 / ambig_total as f64) // Lower = less biased
    } else {
        0.0
    };

    ScorecardResult {
        benchmark: "BBQ (Parrish 2022)".into(),
        domain: "Bias".into(),
        metric: "accuracy".into(),
        value: if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        },
        n_samples: total,
        elapsed_ms: start.elapsed().as_millis() as u64,
        notes: vec![
            format!(
                "Bias score: {:.3} (0=unbiased, 1=maximally biased)",
                bias_score
            ),
            format!("Ambiguous→unknown rate: {}/{}", ambig_unknown, ambig_total),
        ],
    }
}

// ============================================================================
// Moral Unified (existing benchmark, read cached results)
// ============================================================================

fn run_moral_unified_cached() -> Option<ScorecardResult> {
    let path = Path::new("data/benchmarks/moral_unified/results.json");
    let content = fs::read_to_string(path).ok()?;
    let data: serde_json::Value = serde_json::from_str(&content).ok()?;

    let accuracy = data.get("overall_accuracy")?.as_f64()?;
    let total = data.get("total_examples")?.as_u64()? as usize;
    let elapsed = data.get("total_duration_ms")?.as_u64().unwrap_or(0);

    Some(ScorecardResult {
        benchmark: "Moral Unified (6 datasets)".into(),
        domain: "Ethics".into(),
        metric: "accuracy".into(),
        value: accuracy,
        n_samples: total,
        elapsed_ms: elapsed,
        notes: vec!["Cached result from benchmark_moral_unified".into()],
    })
}

// ============================================================================
// ETHICS (Trained Pipeline - from cached moral_unified results)
// ============================================================================

fn extract_ethics_from_cached() -> Option<ScorecardResult> {
    let path = Path::new("data/benchmarks/moral_unified/results.json");
    let content = fs::read_to_string(path).ok()?;
    let data: serde_json::Value = serde_json::from_str(&content).ok()?;

    // Extract ETHICS-specific datasets from the cached result
    let datasets = data.get("datasets")?.as_array()?;
    let mut ethics_correct = 0usize;
    let mut ethics_total = 0usize;

    for ds in datasets {
        let dataset = ds.get("dataset")?.as_str()?;
        if dataset == "ETHICS" {
            let correct = ds.get("correct").and_then(|c| c.as_u64()).unwrap_or(0) as usize;
            let total = ds.get("total").and_then(|t| t.as_u64()).unwrap_or(0) as usize;
            ethics_correct += correct;
            ethics_total += total;
        }
    }

    if ethics_total == 0 {
        return None;
    }

    Some(ScorecardResult {
        benchmark: "ETHICS Trained (Hendrycks 2021)".into(),
        domain: "Ethics".into(),
        metric: "accuracy".into(),
        value: ethics_correct as f64 / ethics_total as f64,
        n_samples: ethics_total,
        elapsed_ms: 0,
        notes: vec![
            "Trained MoralAlgebra + keyword signals + prototype ensemble".into(),
            "From benchmark_moral_unified cached results".into(),
        ],
    })
}

// ============================================================================
// LLM Baselines (published results)
// ============================================================================

fn llm_baselines() -> HashMap<String, Vec<LlmBaseline>> {
    let mut m = HashMap::new();

    m.insert(
        "MMLU (Hendrycks 2021)".into(),
        vec![
            LlmBaseline {
                model: "GPT-4".into(),
                value: 0.864,
                source: "OpenAI 2023".into(),
            },
            LlmBaseline {
                model: "Claude 3.5 Sonnet".into(),
                value: 0.887,
                source: "Anthropic 2024".into(),
            },
            LlmBaseline {
                model: "Llama 3 70B".into(),
                value: 0.820,
                source: "Meta 2024".into(),
            },
            LlmBaseline {
                model: "Random".into(),
                value: 0.250,
                source: "4-choice".into(),
            },
        ],
    );

    m.insert(
        "HellaSwag (Zellers 2019)".into(),
        vec![
            LlmBaseline {
                model: "GPT-4".into(),
                value: 0.953,
                source: "OpenAI 2023".into(),
            },
            LlmBaseline {
                model: "Human".into(),
                value: 0.954,
                source: "Zellers 2019".into(),
            },
            LlmBaseline {
                model: "Random".into(),
                value: 0.250,
                source: "4-choice".into(),
            },
        ],
    );

    m.insert(
        "GSM8K (Cobbe 2021)".into(),
        vec![
            LlmBaseline {
                model: "GPT-4".into(),
                value: 0.920,
                source: "OpenAI 2023".into(),
            },
            LlmBaseline {
                model: "Claude 3.5 Sonnet".into(),
                value: 0.960,
                source: "Anthropic 2024".into(),
            },
            LlmBaseline {
                model: "Random".into(),
                value: 0.0,
                source: "open-ended".into(),
            },
        ],
    );

    m.insert(
        "TruthfulQA (Lin 2022)".into(),
        vec![
            LlmBaseline {
                model: "GPT-4".into(),
                value: 0.590,
                source: "OpenAI 2023".into(),
            },
            LlmBaseline {
                model: "Human".into(),
                value: 0.940,
                source: "Lin 2022".into(),
            },
            LlmBaseline {
                model: "Random".into(),
                value: 0.500,
                source: "2-choice".into(),
            },
        ],
    );

    m.insert(
        "ETHICS Raw (Hendrycks 2021)".into(),
        vec![
            LlmBaseline {
                model: "GPT-4".into(),
                value: 0.900,
                source: "estimated".into(),
            },
            LlmBaseline {
                model: "Random".into(),
                value: 0.500,
                source: "2-class".into(),
            },
        ],
    );

    m.insert(
        "BBQ (Parrish 2022)".into(),
        vec![
            LlmBaseline {
                model: "GPT-4".into(),
                value: 0.830,
                source: "estimated".into(),
            },
            LlmBaseline {
                model: "Random".into(),
                value: 0.333,
                source: "3-choice".into(),
            },
        ],
    );

    m.insert(
        "ETHICS Trained (Hendrycks 2021)".into(),
        vec![
            LlmBaseline {
                model: "GPT-4".into(),
                value: 0.900,
                source: "estimated".into(),
            },
            LlmBaseline {
                model: "ALBERT-xxlarge".into(),
                value: 0.857,
                source: "Hendrycks 2021".into(),
            },
            LlmBaseline {
                model: "Random".into(),
                value: 0.500,
                source: "2-class".into(),
            },
        ],
    );

    m
}

fn random_baseline_for(benchmark: &str) -> f64 {
    match benchmark {
        "MMLU (Hendrycks 2021)" => 0.25,
        "HellaSwag (Zellers 2019)" => 0.25,
        "GSM8K (Cobbe 2021)" => 0.0,
        "TruthfulQA (Lin 2022)" => 0.5,
        "ETHICS Raw (Hendrycks 2021)" => 0.5,
        "ETHICS Trained (Hendrycks 2021)" => 0.5,
        "BBQ (Parrish 2022)" => 0.333,
        "Moral Unified (6 datasets)" => 0.5,
        _ => 0.0,
    }
}

// ============================================================================
// Scorecard Generation
// ============================================================================

fn generate_markdown(scorecard: &Scorecard) -> String {
    let mut md = String::new();
    md.push_str("# Symthaea Benchmark Scorecard\n\n");
    md.push_str(&format!("Generated: {}\n\n", scorecard.timestamp));
    md.push_str("## Results\n\n");
    md.push_str("| Benchmark | Domain | Metric | Symthaea | Random | GPT-4 | N | Time |\n");
    md.push_str("|-----------|--------|--------|----------|--------|-------|---|------|\n");

    for entry in &scorecard.entries {
        let r = &entry.result;
        let gpt4 = entry
            .baselines
            .iter()
            .find(|b| b.model == "GPT-4")
            .map(|b| format!("{:.1}%", b.value * 100.0))
            .unwrap_or_else(|| "—".into());

        md.push_str(&format!(
            "| {} | {} | {} | **{:.1}%** | {:.1}% | {} | {} | {}ms |\n",
            r.benchmark,
            r.domain,
            r.metric,
            r.value * 100.0,
            entry.random_baseline * 100.0,
            gpt4,
            r.n_samples,
            r.elapsed_ms,
        ));
    }

    md.push_str("\n## Notes\n\n");
    md.push_str(
        "- **HDC encoding**: 16,384-bit binary hypervectors, bag-of-words, no trained model\n",
    );
    md.push_str(
        "- **ETHICS/Moral**: Uses MoralAlgebra + MoralParser (trained on moral prototypes)\n",
    );
    md.push_str(
        "- **MMLU/HellaSwag**: Pure HDC similarity (tests semantic structure of encoding)\n",
    );
    md.push_str("- **GSM8K**: Heuristic number extraction (not chain-of-thought)\n");
    md.push_str("- **TruthfulQA**: MC1 format (correct vs incorrect answer similarity)\n");
    md.push_str("- **BBQ**: Bias detection across 8 social categories\n");
    md.push_str("\n### Interpretation\n\n");
    md.push_str("Symthaea is **not an LLM** — it is a consciousness-first architecture using HDC+LTC+IIT.\n");
    md.push_str("Scores above random baseline demonstrate that the architecture captures semantic structure.\n");
    md.push_str("The ETHICS score (moral algebra) is the primary differentiating benchmark.\n");

    for entry in &scorecard.entries {
        if !entry.result.notes.is_empty() {
            md.push_str(&format!(
                "\n**{}**: {}\n",
                entry.result.benchmark,
                entry.result.notes.join("; ")
            ));
        }
    }

    md.push_str(&format!(
        "\n---\n*Total runtime: {}ms*\n",
        scorecard.total_elapsed_ms
    ));
    md
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");
    let full = args.iter().any(|a| a == "--full");

    let max_samples = if quick {
        QUICK_SAMPLES
    } else if full {
        FULL_SAMPLES
    } else {
        DEFAULT_SAMPLES
    };

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Symthaea Unified Benchmark Scorecard                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "Mode: {} (max {} samples per benchmark)",
        if quick {
            "quick"
        } else if full {
            "full"
        } else {
            "default"
        },
        if max_samples == usize::MAX {
            "all".to_string()
        } else {
            max_samples.to_string()
        }
    );
    println!();

    let total_start = Instant::now();
    let baselines = llm_baselines();
    let mut results = Vec::new();

    // Run all benchmarks
    let benchmarks: Vec<(&str, Box<dyn Fn() -> ScorecardResult>)> = vec![
        ("ETHICS", Box::new(move || run_ethics(max_samples))),
        ("MMLU", Box::new(move || run_mmlu(max_samples))),
        ("HellaSwag", Box::new(move || run_hellaswag(max_samples))),
        ("GSM8K", Box::new(move || run_gsm8k(max_samples))),
        ("TruthfulQA", Box::new(move || run_truthfulqa(max_samples))),
        ("BBQ", Box::new(move || run_bbq(max_samples))),
    ];

    for (name, bench_fn) in &benchmarks {
        print!("  Running {}...", name);
        let result = bench_fn();
        println!(
            " {:.1}% ({} samples, {}ms)",
            result.value * 100.0,
            result.n_samples,
            result.elapsed_ms
        );
        results.push(result);
    }

    // Add cached moral unified if available (this is the trained pipeline result)
    if let Some(moral) = run_moral_unified_cached() {
        println!(
            "  Loaded Moral Unified (cached): {:.1}% ({} samples)",
            moral.value * 100.0,
            moral.n_samples
        );
        results.push(moral);

        // Also extract per-dataset ETHICS accuracy from cached results
        if let Some(ethics_trained) = extract_ethics_from_cached() {
            println!(
                "  ETHICS (trained pipeline): {:.1}% ({} samples)",
                ethics_trained.value * 100.0,
                ethics_trained.n_samples
            );
            results.push(ethics_trained);
        }
    }

    let total_elapsed = total_start.elapsed().as_millis() as u64;

    // Build scorecard
    let entries: Vec<ScorecardEntry> = results
        .into_iter()
        .map(|r| {
            let bl = baselines.get(&r.benchmark).cloned().unwrap_or_default();
            let random = random_baseline_for(&r.benchmark);
            ScorecardEntry {
                result: r,
                baselines: bl,
                random_baseline: random,
            }
        })
        .collect();

    let scorecard = Scorecard {
        version: "1.0.0".into(),
        timestamp: chrono_timestamp(),
        entries,
        total_elapsed_ms: total_elapsed,
    };

    // Print console summary
    println!();
    println!("════════════════════════════════════════════════════════════════");
    println!();
    println!(
        "{:<30} {:>10} {:>10} {:>10}",
        "Benchmark", "Symthaea", "Random", "GPT-4"
    );
    println!("{}", "-".repeat(65));
    for entry in &scorecard.entries {
        let r = &entry.result;
        let gpt4 = entry
            .baselines
            .iter()
            .find(|b| b.model == "GPT-4")
            .map(|b| format!("{:.1}%", b.value * 100.0))
            .unwrap_or_else(|| "—".into());
        println!(
            "{:<30} {:>9.1}% {:>9.1}% {:>10}",
            r.benchmark,
            r.value * 100.0,
            entry.random_baseline * 100.0,
            gpt4,
        );
    }
    println!("{}", "-".repeat(65));
    println!("Total: {}ms", total_elapsed);

    // Save JSON
    let json_path = Path::new("data/benchmarks/scorecard.json");
    if let Ok(json) = serde_json::to_string_pretty(&scorecard) {
        let _ = fs::write(json_path, &json);
        println!("\nSaved: {}", json_path.display());
    }

    // Save Markdown
    let md_path = Path::new("data/benchmarks/SCORECARD.md");
    let md = generate_markdown(&scorecard);
    let _ = fs::write(md_path, &md);
    println!("Saved: {}", md_path.display());
}

fn chrono_timestamp() -> String {
    // Simple ISO-8601 timestamp without chrono dependency
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}s since epoch", now.as_secs())
}