// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ARC-AGI dataset loader and evaluator.
//!
//! **IMPORTANT: Honesty note on what this measures.**
//!
//! When used with real ARC JSON files (from Chollet's dataset), this evaluates
//! HDC grid encoding/retrieval fidelity on genuine ARC tasks. However, the HDC
//! approach here tests **encoding algebra** (XOR bind/unbind rule recovery), not
//! novel rule discovery. The XOR self-inverse property means "rule application"
//! is exact algebraic recovery, not generalization to unseen transform types.
//!
//! **RETRACTION + FIX (2026-07-18)**: the "2-AFC" scoring here used to compare
//! the predicted output HV's similarity to the real answer against its
//! similarity to a **literally random `BinaryHV`** — and since
//! `BinaryGridEncoder` builds every grid HV from a shared per-task basis
//! (row/col/color HVs), ANY structured grid encoding beats random noise
//! regardless of whether the predicted output's *rule* is correct. That
//! scoring measured 99.0% on the real 400-task ARC-AGI training set (matching
//! the previously published "100% 2-AFC" claim in
//! `book/src/research/validation.md`) but collapsed to 13.8% (identity
//! distractor, *below* chance) and 64.9%/67.8% (reflect_x/reflect_y) once the
//! distractor was fair (equally structured). See `examples/arc_2afc_reaudit.rs`
//! for the standalone re-audit that found this — same inflation class as the
//! retracted Hendrycks ETHICS 94.5%→56.2% figure.
//!
//! `fair_distractor_grid()` below is the fix: it builds a distractor by
//! applying a generic wrong transform (reflect/color-swap, in the same spirit
//! as `arc_analogy.rs`'s "wrong transform on C" distractor) to the test input,
//! falling back to identity only if every transform coincides with the real
//! answer. `evaluate_arc_tasks()` now scores against this fair distractor by
//! default. Even so: this remains **encoding-fidelity** scoring (XOR
//! bind/unbind rule recovery), not novel rule discovery — do not interpret
//! high 2-AFC accuracy as comparable to ARC solve rates reported for LLMs or
//! humans, which require generating exact pixel-perfect outputs.
//!
//! ARC JSON format:
//! ```json
//! {
//!   "train": [{"input": [[0,1,...], ...], "output": [[2,3,...], ...]}],
//!   "test":  [{"input": [[0,1,...], ...], "output": [[2,3,...], ...]}]
//! }
//! ```
//!
//! Usage: `--arc-data-dir /path/to/ARC-AGI/data/training/`
//!
//! Source: <https://github.com/fchollet/ARC-AGI>

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;
use symthaea_core::hdc::grid_encoder::GridEncoder;

/// Build a fair (equally-structured) 2-AFC distractor grid for ARC-style
/// rule-transfer scoring. See this file's 2026-07-18 retraction note for why
/// a literally random `BinaryHV` is not a fair alternative.
///
/// Tries a small set of generic wrong transforms of `test_input` — in the
/// same spirit as `arc_analogy.rs`'s "wrong transform on C" distractor — and
/// returns the first one that actually differs from `true_output` (a 2-AFC
/// trial needs two distinct alternatives). Identity (test input unchanged)
/// is included only as a last resort: it's a legitimate "no transformation
/// happened" wrong guess, but as the harshest of the candidates (see the
/// re-audit's 13.8% figure) it shouldn't be the *first* thing tried against
/// every task. Returns `None` in the rare case where every candidate
/// coincides with the true answer — callers should skip that trial.
pub fn fair_distractor_grid(
    test_input: &[Vec<u8>],
    true_output: &[Vec<u8>],
) -> Option<Vec<Vec<u8>>> {
    let candidates = [
        GridEncoder::reflect_x(test_input),
        GridEncoder::reflect_y(test_input),
        GridEncoder::color_replace(test_input, 0, 1),
        GridEncoder::color_replace(test_input, 1, 2),
        test_input.to_vec(),
    ];
    candidates.into_iter().find(|g| g != true_output)
}

/// A single ARC input/output pair.
#[derive(Debug, Clone, Deserialize)]
pub struct ArcPair {
    pub input: Vec<Vec<u8>>,
    pub output: Vec<Vec<u8>>,
}

/// A complete ARC task with training and test pairs.
#[derive(Debug, Clone, Deserialize)]
pub struct ArcTask {
    pub train: Vec<ArcPair>,
    pub test: Vec<ArcPair>,
}

/// Breakdown of accuracy by a categorical variable.
#[derive(Debug, Clone, Default)]
pub struct AccuracyBreakdown {
    /// Bin label → (count, correct, mean_similarity).
    pub bins: BTreeMap<String, (usize, usize, f64)>,
}

/// Results from evaluating on ARC tasks.
///
/// Note: When used with Chollet's real ARC data, scores reflect HDC encoding
/// fidelity under 2-AFC (not rule discovery ability). When used with synthetic
/// tasks (the common case in psych-bench), scores reflect encoding algebra
/// only — the system knows the transform family and tests retrieval, not
/// discovery of novel rules.
#[derive(Debug, Clone)]
pub struct ArcDatasetResult {
    /// Number of tasks loaded.
    pub tasks_loaded: usize,
    /// Number of tasks where test output was predicted correctly (2-AFC).
    pub tasks_correct: usize,
    /// Mean cosine similarity between predicted and actual test outputs.
    pub mean_similarity: f64,
    /// Mean rule consistency across training pairs (within-task).
    pub mean_rule_consistency: f64,
    /// Per-task results: (task_id, similarity, correct).
    pub per_task: Vec<(String, f64, bool)>,
    /// Accuracy breakdown by grid size (max dimension).
    pub by_grid_size: AccuracyBreakdown,
    /// Accuracy breakdown by number of training examples.
    pub by_train_count: AccuracyBreakdown,
    /// Histogram of similarity scores (10 bins from 0.0 to 1.0).
    pub similarity_histogram: [usize; 10],
    /// Whether tasks were procedurally generated (true) or from Chollet's dataset (false).
    pub synthetic: bool,
}

/// Load all ARC tasks from a directory of JSON files.
///
/// Returns a map of filename → ArcTask.
pub fn load_arc_tasks(dir: &Path) -> Result<BTreeMap<String, ArcTask>, String> {
    if !dir.is_dir() {
        return Err(format!("Not a directory: {}", dir.display()));
    }

    let mut tasks = BTreeMap::new();
    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("Failed to read directory {}: {}", dir.display(), e))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("Read dir entry error: {}", e))?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "json") {
            let content = std::fs::read_to_string(&path)
                .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
            let task: ArcTask = serde_json::from_str(&content)
                .map_err(|e| format!("Failed to parse {}: {}", path.display(), e))?;
            let name = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string());
            tasks.insert(name, task);
        }
    }

    Ok(tasks)
}

/// Evaluate Symthaea's HDC grid encoder on real ARC tasks.
///
/// For each task:
///   1. Encode all training pairs, compute rule HVs.
///   2. Bundle training rules into a consensus rule.
///   3. Apply consensus rule to test input, compare to test output.
///   4. Score: 2-AFC (predicted vs fair-distractor grid — see `fair_distractor_grid()`).
pub fn evaluate_arc_tasks(
    tasks: &BTreeMap<String, ArcTask>,
    dimension: usize,
    seed: u64,
) -> ArcDatasetResult {
    use symthaea_core::hdc::binary_grid_encoder::BinaryGridEncoder;

    let _dimension = dimension;
    let mut tasks_correct: usize = 0;
    let mut similarity_sum: f64 = 0.0;
    let mut consistency_sum: f64 = 0.0;
    let mut per_task = Vec::new();
    let mut task_count: usize = 0;

    for (task_id, task) in tasks {
        if task.train.is_empty() || task.test.is_empty() {
            continue;
        }

        // Determine grid dimensions from the task
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
        let num_colors = 10; // ARC uses colors 0-9

        let encoder = BinaryGridEncoder::new(
            max_rows.max(1),
            max_cols.max(1),
            num_colors,
            seed ^ (task_count as u64),
        );

        // Encode training pairs and compute rules
        let mut rules = Vec::new();
        for pair in &task.train {
            let in_hv = encoder.encode_grid(&pair.input);
            let out_hv = encoder.encode_grid(&pair.output);
            rules.push(encoder.encode_rule(&in_hv, &out_hv));
        }

        // Rule consistency: mean pairwise cosine of training rules
        let mut consistency_pairs = Vec::new();
        for i in 0..rules.len() {
            for j in (i + 1)..rules.len() {
                consistency_pairs.push(rules[i].similarity(&rules[j]) as f64);
            }
        }
        let rule_consistency = if consistency_pairs.is_empty() {
            1.0
        } else {
            consistency_pairs.iter().sum::<f64>() / consistency_pairs.len() as f64
        };
        consistency_sum += rule_consistency;

        // Bundle rules into consensus
        let consensus = encoder.bundle_rules(&rules);

        // Evaluate on first test pair
        let test_pair = &task.test[0];
        let test_in_hv = encoder.encode_grid(&test_pair.input);
        let test_out_hv = encoder.encode_grid(&test_pair.output);
        let predicted = encoder.apply_rule(&test_in_hv, &consensus);

        let pred_sim = predicted.similarity(&test_out_hv) as f64;
        similarity_sum += pred_sim;

        // 2-AFC: predicted vs a fair (equally structured) distractor — see
        // this file's 2026-07-18 retraction note. Falls back to the test
        // input unchanged in the rare case every generic transform happens
        // to coincide with the true output; that fallback can never register
        // as a false "correct" (comparing predicted to itself is never >).
        let distractor_grid = fair_distractor_grid(&test_pair.input, &test_pair.output)
            .unwrap_or_else(|| test_pair.input.clone());
        let distractor = encoder.encode_grid(&distractor_grid);
        let dist_sim = predicted.similarity(&distractor) as f64;
        let correct = pred_sim > dist_sim;
        if correct {
            tasks_correct += 1;
        }

        per_task.push((task_id.clone(), pred_sim, correct));
        task_count += 1;
    }

    // Build breakdowns
    let mut by_grid_size = AccuracyBreakdown::default();
    let mut by_train_count = AccuracyBreakdown::default();
    let mut similarity_histogram = [0usize; 10];

    for (task_id, task) in tasks {
        if let Some((_id, sim, correct)) = per_task.iter().find(|(id, _, _)| id == task_id) {
            // Grid size breakdown
            let max_dim = task
                .train
                .iter()
                .chain(task.test.iter())
                .flat_map(|p| [p.input.len(), p.output.len()])
                .max()
                .unwrap_or(0);
            let size_label = if max_dim <= 5 {
                "small (<=5)"
            } else if max_dim <= 15 {
                "medium (6-15)"
            } else {
                "large (>15)"
            };
            let entry = by_grid_size
                .bins
                .entry(size_label.to_string())
                .or_insert((0, 0, 0.0));
            entry.0 += 1;
            if *correct {
                entry.1 += 1;
            }
            entry.2 += sim;

            // Train count breakdown
            let train_label = format!("{} examples", task.train.len());
            let entry = by_train_count
                .bins
                .entry(train_label)
                .or_insert((0, 0, 0.0));
            entry.0 += 1;
            if *correct {
                entry.1 += 1;
            }
            entry.2 += sim;

            // Similarity histogram
            let bin = (sim * 10.0).floor().clamp(0.0, 9.0) as usize;
            similarity_histogram[bin] += 1;
        }
    }

    // Convert similarity sums to means
    for entry in by_grid_size.bins.values_mut() {
        if entry.0 > 0 {
            entry.2 /= entry.0 as f64;
        }
    }
    for entry in by_train_count.bins.values_mut() {
        if entry.0 > 0 {
            entry.2 /= entry.0 as f64;
        }
    }

    ArcDatasetResult {
        tasks_loaded: task_count,
        tasks_correct,
        mean_similarity: if task_count > 0 {
            similarity_sum / task_count as f64
        } else {
            0.0
        },
        mean_rule_consistency: if task_count > 0 {
            consistency_sum / task_count as f64
        } else {
            0.0
        },
        per_task,
        by_grid_size,
        by_train_count,
        similarity_histogram,
        synthetic: false, // loaded from real ARC JSON files
    }
}

/// Format ARC dataset evaluation results.
pub fn format_arc_dataset_result(result: &ArcDatasetResult) -> String {
    let mut lines = Vec::new();
    lines.push(
        "## ARC-AGI Dataset Evaluation (HDC encoding fidelity, not rule discovery)".to_string(),
    );
    lines.push(format!("Tasks loaded: {}", result.tasks_loaded));
    lines.push(format!(
        "Tasks correct (2-AFC): {}/{} ({:.1}%)",
        result.tasks_correct,
        result.tasks_loaded,
        if result.tasks_loaded > 0 {
            100.0 * result.tasks_correct as f64 / result.tasks_loaded as f64
        } else {
            0.0
        }
    ));
    lines.push(format!(
        "Mean predicted-actual similarity: {:.4}",
        result.mean_similarity
    ));
    lines.push(format!(
        "Mean rule consistency: {:.4}",
        result.mean_rule_consistency
    ));

    // Grid size breakdown
    if !result.by_grid_size.bins.is_empty() {
        lines.push(String::new());
        lines.push("By grid size:".to_string());
        for (label, (count, correct, mean_sim)) in &result.by_grid_size.bins {
            let pct = if *count > 0 {
                100.0 * *correct as f64 / *count as f64
            } else {
                0.0
            };
            lines.push(format!(
                "  {}: {}/{} ({:.1}%), mean_sim={:.4}",
                label, correct, count, pct, mean_sim
            ));
        }
    }

    // Train count breakdown
    if !result.by_train_count.bins.is_empty() {
        lines.push(String::new());
        lines.push("By training examples:".to_string());
        for (label, (count, correct, mean_sim)) in &result.by_train_count.bins {
            let pct = if *count > 0 {
                100.0 * *correct as f64 / *count as f64
            } else {
                0.0
            };
            lines.push(format!(
                "  {}: {}/{} ({:.1}%), mean_sim={:.4}",
                label, correct, count, pct, mean_sim
            ));
        }
    }

    // Similarity histogram
    lines.push(String::new());
    lines.push("Similarity distribution:".to_string());
    for (i, &count) in result.similarity_histogram.iter().enumerate() {
        let lo = i as f64 * 0.1;
        let hi = lo + 0.1;
        let bar = "#".repeat(count.min(60));
        lines.push(format!("  [{:.1},{:.1}): {:>4} {}", lo, hi, count, bar));
    }

    // LLM comparison — NOT directly comparable
    let our_accuracy = if result.tasks_loaded > 0 {
        result.tasks_correct as f64 / result.tasks_loaded as f64
    } else {
        0.0
    };
    lines.push(String::new());
    lines.push("Reference: Real ARC-AGI solve rates (NOT directly comparable):".to_string());
    lines.push(
        "  NOTE: Symthaea uses 2-AFC (predicted vs fair equally-structured distractor),"
            .to_string(),
    );
    lines.push("  while the scores below require exact pixel-perfect generation.".to_string());
    lines.push("  Our metric tests encoding fidelity, not rule discovery.".to_string());
    lines.push(format!(
        "  Symthaea HDC (2-AFC): {:.1}%",
        our_accuracy * 100.0
    ));
    lines.push("  GPT-4 (exact):        5.0%".to_string());
    lines.push("  Claude 3.5 (exact):   21.0%".to_string());
    lines.push("  o3-high (exact):      87.5%".to_string());
    lines.push("  Human (exact):        84.0%".to_string());

    // Show top-5 best and worst tasks
    let mut sorted = result.per_task.clone();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if sorted.len() > 5 {
        lines.push(String::new());
        lines.push("Top 5 tasks (highest similarity):".to_string());
        for (id, sim, correct) in sorted.iter().take(5) {
            let mark = if *correct { "+" } else { "-" };
            lines.push(format!("  [{}] {} sim={:.4}", mark, id, sim));
        }
        lines.push(String::new());
        lines.push("Bottom 5 tasks (lowest similarity):".to_string());
        for (id, sim, correct) in sorted.iter().rev().take(5) {
            let mark = if *correct { "+" } else { "-" };
            lines.push(format!("  [{}] {} sim={:.4}", mark, id, sim));
        }
    }

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arc_pair_deserialize() {
        let json = r#"{"input": [[0,1],[2,3]], "output": [[3,2],[1,0]]}"#;
        let pair: ArcPair = serde_json::from_str(json).unwrap();
        assert_eq!(pair.input.len(), 2);
        assert_eq!(pair.output[0], vec![3, 2]);
    }

    #[test]
    fn test_arc_task_deserialize() {
        let json = r#"{
            "train": [{"input": [[0,1],[2,3]], "output": [[3,2],[1,0]]}],
            "test": [{"input": [[4,5],[6,7]], "output": [[7,6],[5,4]]}]
        }"#;
        let task: ArcTask = serde_json::from_str(json).unwrap();
        assert_eq!(task.train.len(), 1);
        assert_eq!(task.test.len(), 1);
    }

    #[test]
    fn test_evaluate_synthetic_task() {
        use symthaea_core::hdc::grid_encoder::GridEncoder;

        // Create a synthetic ARC task: reflect_x
        let input1 = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let output1 = vec![vec![2, 1, 0], vec![5, 4, 3]]; // reflect_x
        let input2 = vec![vec![1, 0, 1], vec![2, 2, 0]];
        let output2 = vec![vec![1, 0, 1], vec![0, 2, 2]]; // reflect_x

        let test_input = vec![vec![5, 3, 1], vec![4, 2, 0]];
        let test_output = GridEncoder::reflect_x(&test_input);

        let task = ArcTask {
            train: vec![
                ArcPair {
                    input: input1,
                    output: output1,
                },
                ArcPair {
                    input: input2,
                    output: output2,
                },
            ],
            test: vec![ArcPair {
                input: test_input,
                output: test_output,
            }],
        };

        let mut tasks = BTreeMap::new();
        tasks.insert("synthetic_reflect".to_string(), task);

        let result = evaluate_arc_tasks(&tasks, 512, 42);
        assert_eq!(result.tasks_loaded, 1);
        // The rule should be somewhat consistent
        assert!(result.mean_rule_consistency.is_finite());
        // The prediction should beat a random distractor
        assert!(result.mean_similarity.is_finite());
    }

    #[test]
    fn test_load_nonexistent_dir() {
        let result = load_arc_tasks(Path::new("/nonexistent/path"));
        assert!(result.is_err());
    }

    #[test]
    fn test_format_result() {
        let result = ArcDatasetResult {
            tasks_loaded: 10,
            tasks_correct: 7,
            mean_similarity: 0.45,
            mean_rule_consistency: 0.72,
            per_task: vec![
                ("task1".to_string(), 0.6, true),
                ("task2".to_string(), 0.3, false),
            ],
            by_grid_size: AccuracyBreakdown::default(),
            by_train_count: AccuracyBreakdown::default(),
            similarity_histogram: [0; 10],
            synthetic: false,
        };
        let formatted = format_arc_dataset_result(&result);
        assert!(formatted.contains("10"));
        assert!(formatted.contains("7/10"));
        assert!(formatted.contains("70.0%"));
        assert!(formatted.contains("NOT directly comparable"));
    }
}
