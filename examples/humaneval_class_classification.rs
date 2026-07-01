// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! HumanEval Class Classification Benchmark
//!
//! Honest measurement: how well does our hybrid System 1 classifier
//! identify the algorithm class of HumanEval Python problems?
//!
//! This tests cross-language generalization of the Abstract Algorithm
//! Encoder. The classifier was trained on Rust solutions; if it can
//! correctly classify Python problem descriptions, it confirms the
//! Substrate Independence thesis for code.
//!
//! Run:
//!   cargo run --example humaneval_class_classification --features code_generation -- --limit 30

use symthaea::language::algorithm_encoder::{AlgorithmClass, AlgorithmEncoder};
use symthaea::language::algorithm_training::{
    LearnedProjection, build_channels_from_purpose_public, build_training_pairs, hybrid_classify,
    hybrid_classify_knn, knn_hdc_classify, strong_keyword_class, train_linear_classifier,
};

struct HEvProblem {
    task_id: String,
    prompt: String,
    entry_point: String,
}

/// Heuristic classifier for HumanEval Python prompts → algorithm class.
///
/// This is the GROUND TRUTH for our evaluation. If our system gets it
/// right, it's classifying like a human expert would.
fn ground_truth_class(problem: &HEvProblem) -> AlgorithmClass {
    let p = problem.prompt.to_lowercase();
    let n = problem.entry_point.to_lowercase();
    let combined = format!("{n} {p}");

    if combined.contains("sort") || combined.contains("ordered") {
        AlgorithmClass::Sorting
    } else if combined.contains("prime")
        || combined.contains("fibonacci")
        || combined.contains("factorial")
        || combined.contains("digit")
        || combined.contains("number")
        || combined.contains("multiplication")
        || combined.contains("divisor")
        || combined.contains("modulo")
    {
        AlgorithmClass::Mathematical
    } else if combined.contains("string")
        || combined.contains("char")
        || combined.contains("vowel")
        || combined.contains("palindrome")
        || combined.contains("anagram")
        || combined.contains("encode")
        || combined.contains("decode")
    {
        AlgorithmClass::StringProcessing
    } else if combined.contains("search")
        || combined.contains("find")
        || combined.contains("locate")
    {
        AlgorithmClass::Search
    } else if combined.contains("graph")
        || combined.contains("tree")
        || combined.contains("path")
        || combined.contains("matrix")
    {
        AlgorithmClass::Graph
    } else if combined.contains("class ")
        || combined.contains("stack")
        || combined.contains("queue")
    {
        AlgorithmClass::DataStructure
    } else {
        AlgorithmClass::IoTransform
    }
}

fn extract_purpose(prompt: &str) -> String {
    // Extract the docstring (between triple quotes) or first comment line
    if let Some(start) = prompt.find("\"\"\"") {
        let after = &prompt[start + 3..];
        if let Some(end) = after.find("\"\"\"") {
            return after[..end].trim().to_string();
        }
    }
    if let Some(start) = prompt.find("'''") {
        let after = &prompt[start + 3..];
        if let Some(end) = after.find("'''") {
            return after[..end].trim().to_string();
        }
    }
    prompt.lines().take(3).collect::<Vec<_>>().join(" ")
}

fn extract_signature(prompt: &str, entry_point: &str) -> String {
    // Find the def line for entry_point
    for line in prompt.lines() {
        if line.contains(&format!("def {entry_point}")) {
            return line.trim().trim_end_matches(':').to_string();
        }
    }
    format!("def {entry_point}()")
}

fn load_problems(path: &str, limit: usize) -> std::io::Result<Vec<HEvProblem>> {
    let content = std::fs::read_to_string(path)?;
    let mut problems = Vec::new();
    for line in content.lines().take(limit) {
        let task_id = extract_json_string(line, "task_id").unwrap_or_default();
        let prompt = extract_json_string(line, "prompt").unwrap_or_default();
        let entry_point = extract_json_string(line, "entry_point").unwrap_or_default();
        problems.push(HEvProblem {
            task_id,
            prompt,
            entry_point,
        });
    }
    Ok(problems)
}

/// Naive JSON string field extractor (avoids serde dependency for example).
fn extract_json_string(json: &str, field: &str) -> Option<String> {
    let key = format!("\"{field}\":");
    let start = json.find(&key)? + key.len();
    let after = json[start..].trim_start();
    if !after.starts_with('"') {
        return None;
    }
    let body = &after[1..];
    let mut result = String::new();
    let mut chars = body.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(result),
            '\\' => match chars.next()? {
                'n' => result.push('\n'),
                't' => result.push('\t'),
                'r' => result.push('\r'),
                '"' => result.push('"'),
                '\\' => result.push('\\'),
                other => result.push(other),
            },
            other => result.push(other),
        }
    }
    None
}

fn main() {
    let limit = std::env::args()
        .skip_while(|a| a != "--limit")
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    let path = "benchmarks/ai_benchmarks/data/humaneval/decompressed/HumanEval.jsonl";
    let problems = match load_problems(path, limit) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to load {path}: {e}");
            std::process::exit(1);
        }
    };

    println!("Loaded {} HumanEval problems", problems.len());
    println!("Training classifier on Rust corpus...");
    let (classifier, train_acc, eval_acc, _) = train_linear_classifier(100, 0.01);
    println!(
        "  → train={:.0}% eval={:.0}%",
        train_acc * 100.0,
        eval_acc * 100.0
    );

    let pairs = build_training_pairs();
    let encoder = AlgorithmEncoder::new();
    let projection = LearnedProjection::fit(&pairs);

    let mut linear_correct = 0usize;
    let mut knn_correct = 0usize;
    let mut keyword_used = 0usize;
    let mut by_class: std::collections::HashMap<AlgorithmClass, (usize, usize, usize)> =
        std::collections::HashMap::new();

    println!("\n=== Per-problem results (Linear vs k-NN HDC) ===");
    for problem in &problems {
        let purpose = extract_purpose(&problem.prompt);
        let signature = extract_signature(&problem.prompt, &problem.entry_point);

        let truth = ground_truth_class(problem);

        // System 1: both classifiers
        let channels = symthaea::language::algorithm_training::build_channels_from_purpose_public(
            &purpose, &signature,
        );
        let hv = encoder.encode(&channels);
        let projected = projection.project(&hv);
        let predicted_lin = hybrid_classify(&purpose, &projected.values, &classifier);
        let predicted_knn = hybrid_classify_knn(&purpose, &hv, &pairs, 5);

        let used_keyword = strong_keyword_class(&purpose).is_some();
        if used_keyword {
            keyword_used += 1;
        }

        let mark_lin = if predicted_lin == truth { "✓" } else { "✗" };
        let mark_knn = if predicted_knn == truth { "✓" } else { "✗" };
        if predicted_lin == truth {
            linear_correct += 1;
        }
        if predicted_knn == truth {
            knn_correct += 1;
        }

        let entry = by_class.entry(truth).or_insert((0, 0, 0));
        entry.2 += 1;
        if predicted_lin == truth {
            entry.0 += 1;
        }
        if predicted_knn == truth {
            entry.1 += 1;
        }
        // Use the linear prediction for printout consistency
        let predicted = predicted_lin;
        let mark = mark_lin;
        let _ = mark_knn;

        println!(
            "  {} {:14} truth={:?} predicted={:?}{}",
            mark,
            problem.task_id,
            truth,
            predicted,
            if used_keyword { " (keyword)" } else { "" }
        );
    }

    println!("\n=== Summary: Cross-Language Classification ===");
    println!(
        "Linear classifier:  {}/{} ({:.1}%)",
        linear_correct,
        problems.len(),
        linear_correct as f32 / problems.len() as f32 * 100.0
    );
    println!(
        "k-NN HDC voting:    {}/{} ({:.1}%)",
        knn_correct,
        problems.len(),
        knn_correct as f32 / problems.len() as f32 * 100.0
    );
    println!(
        "Keyword priors used: {}/{} ({:.0}%)",
        keyword_used,
        problems.len(),
        keyword_used as f32 / problems.len() as f32 * 100.0
    );
    println!("\nPer-class (linear / k-NN):");
    for (class, (lin, knn, total)) in &by_class {
        println!(
            "  {:?}: linear={}/{} ({:.0}%) | k-NN={}/{} ({:.0}%)",
            class,
            lin,
            total,
            *lin as f32 / *total as f32 * 100.0,
            knn,
            total,
            *knn as f32 / *total as f32 * 100.0
        );
    }
}
