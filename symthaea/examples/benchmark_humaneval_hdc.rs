// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # HumanEval HDC Code Understanding Benchmark
//!
//! Tests Symthaea's HDC BinaryHV encoding on code understanding using the
//! OpenAI HumanEval dataset (Chen et al., 2021) — 164 Python coding problems.
//!
//! Since HDC cannot execute Python, this benchmark evaluates **code understanding**
//! rather than code generation:
//!
//! 1. **Solution discrimination**: Given a function prompt, does HDC pick the
//!    real canonical solution over shuffled/corrupted distractors?
//! 2. **Prompt clustering**: Do similar task prompts (e.g., list operations,
//!    string manipulation) cluster together in HDC space?
//!
//! ## Encoding
//! Deterministic token->BinaryHV via `BinaryHV::random(fnv1a(token))`,
//! code = `BinaryHV::bundle(&token_hvs)`.
//!
//! The prompt is bound with each candidate solution via `prompt_hv.bind(&solution_hv)`
//! to create a "prompt-solution pair" vector. The pair most similar to the
//! prompt's own encoding is selected.
//!
//! ## Distractor Generation
//! - **Line-shuffled**: Randomly permute lines of the canonical solution
//! - **Token-corrupted**: Replace 30% of tokens with random substitutes
//! - **Reversed**: Reverse the line order entirely
//!
//! ## Run
//! ```bash
//! cargo run --release --example benchmark_humaneval_hdc
//! ```
//!
//! Data: data/benchmarks/humaneval/repo/data/HumanEval.jsonl.gz
//! Results: data/benchmarks/humaneval/results.json

use serde::Serialize;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::time::Instant;
use symthaea::hdc::binary_hv::BinaryHV;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const DATA_PATH: &str = "data/benchmarks/humaneval/repo/data/HumanEval.jsonl.gz";
const RESULTS_PATH: &str = "data/benchmarks/humaneval/results.json";

/// Number of distractors per problem.
const N_DISTRACTORS: usize = 3;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// One HumanEval problem parsed from JSONL.
#[derive(Debug, Clone)]
struct HumanEvalProblem {
    task_id: String,
    prompt: String,
    canonical_solution: String,
    entry_point: String,
}

/// Per-problem evaluation result.
#[derive(Debug, Clone, Serialize)]
struct ProblemResult {
    task_id: String,
    entry_point: String,
    correct: bool,
    canonical_rank: usize,
    canonical_similarity: f64,
    best_distractor_similarity: f64,
    margin: f64,
}

/// Cluster analysis record.
#[derive(Debug, Clone, Serialize)]
struct ClusterStats {
    category: String,
    count: usize,
    avg_intra_similarity: f64,
    avg_inter_similarity: f64,
    separation: f64,
}

/// Full results structure written to JSON.
#[derive(Debug, Serialize)]
struct BenchmarkResults {
    benchmark: String,
    method: String,
    total_problems: usize,
    discrimination_correct: usize,
    discrimination_accuracy: f64,
    random_baseline: f64,
    avg_canonical_similarity: f64,
    avg_distractor_similarity: f64,
    avg_margin: f64,
    elapsed_secs: f64,
    cluster_analysis: Vec<ClusterStats>,
    per_problem: Vec<ProblemResult>,
}

// ---------------------------------------------------------------------------
// Data loading (via zcat)
// ---------------------------------------------------------------------------

fn load_humaneval(path: &Path) -> Vec<HumanEvalProblem> {
    let output = std::process::Command::new("zcat")
        .arg(path)
        .output()
        .expect("Failed to run zcat -- is gzip installed?");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("zcat failed: {}", stderr);
    }

    let content = String::from_utf8_lossy(&output.stdout);
    let mut problems = Vec::new();

    for (line_no, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("WARNING: Skipping line {}: {}", line_no + 1, e);
                continue;
            }
        };
        problems.push(HumanEvalProblem {
            task_id: v["task_id"].as_str().unwrap_or("").to_string(),
            prompt: v["prompt"].as_str().unwrap_or("").to_string(),
            canonical_solution: v["canonical_solution"].as_str().unwrap_or("").to_string(),
            entry_point: v["entry_point"].as_str().unwrap_or("").to_string(),
        });
    }
    problems
}

// ---------------------------------------------------------------------------
// HDC encoding helpers
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit hash of a token string.
fn token_seed(token: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in token.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Tokenize code into tokens: identifiers, keywords, operators, literals.
/// Preserves Python-relevant tokens better than pure alphanumeric splitting.
fn tokenize_code(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            current.push(ch);
        } else {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            // Keep meaningful operators/punctuation as single-char tokens
            if !ch.is_whitespace() {
                tokens.push(ch.to_string());
            }
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// Encode a code string as a BinaryHV by bundling per-token random vectors.
fn encode_code(text: &str) -> BinaryHV {
    let tokens = tokenize_code(text);
    if tokens.is_empty() {
        return BinaryHV::random(0);
    }
    let token_hvs: Vec<BinaryHV> = tokens
        .iter()
        .map(|t| BinaryHV::random(token_seed(t)))
        .collect();
    BinaryHV::bundle(&token_hvs)
}

// ---------------------------------------------------------------------------
// Distractor generation
// ---------------------------------------------------------------------------

/// Simple deterministic PRNG for reproducible shuffling.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x6c62272e07bb0142,
        }
    }

    fn next_u64(&mut self) -> u64 {
        // SplitMix64
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

/// Generate a line-shuffled distractor from canonical solution.
fn distractor_shuffled(canonical: &str, seed: u64) -> String {
    let mut lines: Vec<&str> = canonical.lines().collect();
    if lines.len() <= 1 {
        // Can't shuffle a single line; corrupt tokens instead
        return distractor_corrupted(canonical, seed);
    }
    let mut rng = SimpleRng::new(seed);
    // Fisher-Yates shuffle
    for i in (1..lines.len()).rev() {
        let j = rng.next_usize(i + 1);
        lines.swap(i, j);
    }
    lines.join("\n")
}

/// Generate a token-corrupted distractor (replace ~30% of tokens).
fn distractor_corrupted(canonical: &str, seed: u64) -> String {
    let tokens = tokenize_code(canonical);
    if tokens.is_empty() {
        return canonical.to_string();
    }
    let mut rng = SimpleRng::new(seed);
    let replacement_words = [
        "foo", "bar", "baz", "qux", "tmp", "val", "res", "buf", "idx", "cnt", "flag", "node",
        "data", "elem", "prev", "next", "curr", "size", "len", "key",
    ];

    let result_tokens: Vec<String> = tokens
        .iter()
        .map(|t| {
            if t.len() > 1 && rng.next_u64() % 100 < 30 {
                replacement_words[rng.next_usize(replacement_words.len())].to_string()
            } else {
                t.clone()
            }
        })
        .collect();

    // Reconstruct with spaces between alphanumeric tokens
    let mut out = String::new();
    for (i, tok) in result_tokens.iter().enumerate() {
        if i > 0 {
            let prev = &result_tokens[i - 1];
            let prev_is_word = prev.chars().next().map_or(false, |c| c.is_alphanumeric());
            let curr_is_word = tok.chars().next().map_or(false, |c| c.is_alphanumeric());
            if prev_is_word && curr_is_word {
                out.push(' ');
            }
        }
        out.push_str(tok);
    }
    out
}

/// Generate a reversed-lines distractor.
fn distractor_reversed(canonical: &str) -> String {
    let lines: Vec<&str> = canonical.lines().collect();
    let reversed: Vec<&str> = lines.into_iter().rev().collect();
    reversed.join("\n")
}

// ---------------------------------------------------------------------------
// Prompt clustering heuristic
// ---------------------------------------------------------------------------

/// Categorize a HumanEval task by its entry point name into rough categories.
fn categorize_task(entry_point: &str) -> &'static str {
    let name = entry_point.to_lowercase();
    if name.contains("string")
        || name.contains("str")
        || name.contains("char")
        || name.contains("vowel")
        || name.contains("palindrome")
        || name.contains("encode")
        || name.contains("decode")
        || name.contains("encrypt")
        || name.contains("word")
        || name.contains("letter")
        || name.contains("bracket")
        || name.contains("paren")
    {
        "string_ops"
    } else if name.contains("sort")
        || name.contains("max")
        || name.contains("min")
        || name.contains("list")
        || name.contains("array")
        || name.contains("filter")
        || name.contains("remove")
        || name.contains("unique")
        || name.contains("reverse")
        || name.contains("flatten")
        || name.contains("interleave")
        || name.contains("zip")
    {
        "list_ops"
    } else if name.contains("prime")
        || name.contains("fib")
        || name.contains("factorial")
        || name.contains("gcd")
        || name.contains("sum")
        || name.contains("count")
        || name.contains("digit")
        || name.contains("even")
        || name.contains("odd")
        || name.contains("number")
        || name.contains("math")
        || name.contains("median")
        || name.contains("mean")
        || name.contains("triangle")
    {
        "math_numeric"
    } else if name.contains("valid")
        || name.contains("check")
        || name.contains("is_")
        || name.contains("has_")
        || name.contains("can_")
        || name.contains("match")
        || name.contains("correct")
        || name.contains("monoton")
    {
        "validation"
    } else {
        "other"
    }
}

// ---------------------------------------------------------------------------
// Main benchmark
// ---------------------------------------------------------------------------

fn main() {
    println!("=============================================================");
    println!("  HumanEval HDC Code Understanding Benchmark");
    println!("  164 problems, solution discrimination + clustering");
    println!("=============================================================\n");

    // --- Load data ---
    let data_path = Path::new(DATA_PATH);
    if !data_path.exists() {
        eprintln!(
            "ERROR: Data file not found at {}\n\
             Download HumanEval and place the gzipped JSONL at that path.\n\
             See: https://github.com/openai/human-eval",
            DATA_PATH
        );
        std::process::exit(1);
    }

    let problems = load_humaneval(data_path);
    let n = problems.len();
    println!("Loaded {} HumanEval problems\n", n);
    if n == 0 {
        eprintln!("ERROR: No problems loaded. Check data file.");
        std::process::exit(1);
    }

    // --- Run discrimination benchmark ---
    let start = Instant::now();

    let mut discrimination_correct = 0usize;
    let mut total_canonical_sim = 0.0f64;
    let mut total_distractor_sim = 0.0f64;
    let mut total_margin = 0.0f64;
    let mut per_problem: Vec<ProblemResult> = Vec::new();

    // Also collect prompt HVs for clustering analysis
    let mut prompt_hvs: Vec<(String, &'static str, BinaryHV)> = Vec::new();

    for (i, problem) in problems.iter().enumerate() {
        let prompt_hv = encode_code(&problem.prompt);
        let canonical_hv = encode_code(&problem.canonical_solution);

        let category = categorize_task(&problem.entry_point);
        prompt_hvs.push((problem.task_id.clone(), category, prompt_hv));

        // Create prompt-solution pair vector for canonical
        let canonical_pair = prompt_hv.bind(&canonical_hv);

        // Generate distractors and their pair vectors
        let task_seed = token_seed(&problem.task_id);
        let distractors = [
            distractor_shuffled(&problem.canonical_solution, task_seed),
            distractor_corrupted(&problem.canonical_solution, task_seed.wrapping_add(1)),
            distractor_reversed(&problem.canonical_solution),
        ];

        let distractor_pairs: Vec<BinaryHV> = distractors
            .iter()
            .map(|d| {
                let d_hv = encode_code(d);
                prompt_hv.bind(&d_hv)
            })
            .collect();

        // Score: similarity of each pair to the prompt
        let canonical_sim = prompt_hv.similarity(&canonical_pair) as f64;
        let distractor_sims: Vec<f64> = distractor_pairs
            .iter()
            .map(|d| prompt_hv.similarity(d) as f64)
            .collect();
        let best_distractor_sim = distractor_sims
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Rank the canonical among all candidates
        let mut all_sims: Vec<(usize, f64)> = std::iter::once((0, canonical_sim))
            .chain(distractor_sims.iter().enumerate().map(|(i, &s)| (i + 1, s)))
            .collect();
        all_sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let canonical_rank = all_sims
            .iter()
            .position(|(idx, _)| *idx == 0)
            .unwrap_or(N_DISTRACTORS)
            + 1;

        let correct = canonical_rank == 1;
        if correct {
            discrimination_correct += 1;
        }

        let margin = canonical_sim - best_distractor_sim;
        total_canonical_sim += canonical_sim;
        total_distractor_sim += best_distractor_sim;
        total_margin += margin;

        per_problem.push(ProblemResult {
            task_id: problem.task_id.clone(),
            entry_point: problem.entry_point.clone(),
            correct,
            canonical_rank,
            canonical_similarity: canonical_sim,
            best_distractor_similarity: best_distractor_sim,
            margin,
        });

        // Progress
        if (i + 1) % 20 == 0 || i + 1 == n {
            let acc = discrimination_correct as f64 / (i + 1) as f64 * 100.0;
            println!(
                "[{:>3}/{:>3}] Discrimination accuracy: {:.1}%",
                i + 1,
                n,
                acc
            );
        }
    }

    // --- Clustering analysis ---
    println!("\nRunning prompt clustering analysis...");

    let mut category_hvs: HashMap<&'static str, Vec<usize>> = HashMap::new();
    for (idx, (_, cat, _)) in prompt_hvs.iter().enumerate() {
        category_hvs.entry(cat).or_default().push(idx);
    }

    let mut cluster_analysis: Vec<ClusterStats> = Vec::new();

    for (&category, indices) in &category_hvs {
        if indices.len() < 2 {
            continue;
        }

        // Intra-cluster similarity (average pairwise within category)
        let mut intra_sum = 0.0f64;
        let mut intra_count = 0usize;
        for i in 0..indices.len() {
            for j in (i + 1)..indices.len() {
                let sim = prompt_hvs[indices[i]]
                    .2
                    .similarity(&prompt_hvs[indices[j]].2) as f64;
                intra_sum += sim;
                intra_count += 1;
            }
        }
        let avg_intra = if intra_count > 0 {
            intra_sum / intra_count as f64
        } else {
            0.0
        };

        // Inter-cluster similarity (average to all other-category prompts)
        let mut inter_sum = 0.0f64;
        let mut inter_count = 0usize;
        for &idx_a in indices {
            for (idx_b, (_, cat_b, _)) in prompt_hvs.iter().enumerate() {
                if *cat_b != category {
                    let sim = prompt_hvs[idx_a].2.similarity(&prompt_hvs[idx_b].2) as f64;
                    inter_sum += sim;
                    inter_count += 1;
                }
            }
        }
        let avg_inter = if inter_count > 0 {
            inter_sum / inter_count as f64
        } else {
            0.0
        };

        cluster_analysis.push(ClusterStats {
            category: category.to_string(),
            count: indices.len(),
            avg_intra_similarity: avg_intra,
            avg_inter_similarity: avg_inter,
            separation: avg_intra - avg_inter,
        });
    }

    cluster_analysis.sort_by(|a, b| b.separation.partial_cmp(&a.separation).unwrap());

    let elapsed = start.elapsed().as_secs_f64();

    // --- Print results ---
    let discrimination_accuracy = discrimination_correct as f64 / n as f64;
    let avg_canonical_sim = total_canonical_sim / n as f64;
    let avg_distractor_sim = total_distractor_sim / n as f64;
    let avg_margin = total_margin / n as f64;

    println!("\n=============================================================");
    println!("  RESULTS");
    println!("=============================================================\n");

    println!("--- Solution Discrimination ---");
    println!("Total problems:         {}", n);
    println!(
        "Discrimination acc:     {:.1}% ({}/{})",
        discrimination_accuracy * 100.0,
        discrimination_correct,
        n
    );
    println!(
        "Random baseline:        {:.1}% (1/{} choices)",
        100.0 / (N_DISTRACTORS + 1) as f64,
        N_DISTRACTORS + 1
    );
    println!("Avg canonical sim:      {:.4}", avg_canonical_sim);
    println!("Avg best distractor:    {:.4}", avg_distractor_sim);
    println!("Avg margin:             {:.4}", avg_margin);

    println!("\n--- Prompt Clustering ---");
    for stats in &cluster_analysis {
        println!(
            "  {:<16} n={:<3}  intra={:.4}  inter={:.4}  sep={:+.4}",
            stats.category,
            stats.count,
            stats.avg_intra_similarity,
            stats.avg_inter_similarity,
            stats.separation
        );
    }

    println!("\nElapsed: {:.2}s", elapsed);

    // --- Save results JSON ---
    let results = BenchmarkResults {
        benchmark: "HumanEval-HDC".to_string(),
        method: "HDC BinaryHV (16,384-bit, code-token-bundle, bind-pair discrimination)"
            .to_string(),
        total_problems: n,
        discrimination_correct,
        discrimination_accuracy,
        random_baseline: 1.0 / (N_DISTRACTORS + 1) as f64,
        avg_canonical_similarity: avg_canonical_sim,
        avg_distractor_similarity: avg_distractor_sim,
        avg_margin,
        elapsed_secs: elapsed,
        cluster_analysis,
        per_problem,
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
        "\nConclusion: HDC code-token encoding achieves {:.1}% solution discrimination",
        discrimination_accuracy * 100.0
    );
    println!(
        "on HumanEval (random = {:.1}%).",
        100.0 / (N_DISTRACTORS + 1) as f64
    );
    println!(
        "Clustering separation indicates whether code with similar purpose groups in HDC space."
    );
}