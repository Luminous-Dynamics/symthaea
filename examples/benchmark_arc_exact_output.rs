// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ARC exact-output qualification lane.
//!
//! This is intentionally distinct from `benchmark_arc_reasoning.rs`, which is an HDC
//! representation/rule-vector probe. Success here means one thing only: the solver produced
//! the exact held-out output grid.
//!
//! The initial solver is deliberately modest and auditable. It searches a finite canonical
//! grammar of exact grid transformations (identity, reflections, rotations, translations,
//! color replacement, and geometry+color composition). A transform is eligible only if it
//! exactly reproduces every supplied training output. If no candidate survives, the solver
//! abstains. If surviving candidates disagree on the test output, the solver also abstains.
//! The test target is never passed to the solver.
//!
//! Required environment:
//! - `SYMTHAEA_SUBJECT_REVISION`: exact code revision being qualified
//! - `SYMTHAEA_ARC_DATASET_VERSION`: exact ARC dataset/version identity
//!
//! Optional environment:
//! - `SYMTHAEA_ARC_DATA_DIR`: ARC repository data directory
//! - `SYMTHAEA_ARC_SPLIT`: `training` (default) or `evaluation`
//! - `SYMTHAEA_ARC_EXACT_RESULTS_PATH`: JSON output path
//! - `SYMTHAEA_ARC_MAX_TASKS`: deterministic prefix limit for smoke runs

use serde::Serialize;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use symthaea::hdc::grid_encoder::GridEncoder;
use symthaea::intelligence::{
    aggregate_receipts, evaluate_episode, AbstentionReason, CapabilitySlice, EvidenceRef,
    EpisodeJudgment, ReasoningDecisionRecord, ReasoningDomain, ReasoningEpisode, ReasoningOutcome,
    ReasoningProblemRef, ReasoningQualificationReceipt, ResourceUsage,
};

type Grid = Vec<Vec<u8>>;

const CONFIGURATION_ID: &str = "arc-exact-transform-search-v1";
const RESULT_SCHEMA_VERSION: u32 = 1;

#[derive(Clone)]
struct GridPair {
    input: Grid,
    output: Grid,
}

struct ArcTestCase {
    input: Grid,
    expected: Grid,
}

struct ArcTask {
    train: Vec<GridPair>,
    test: Vec<ArcTestCase>,
}

#[derive(Debug, Clone, Copy)]
enum Geometry {
    Identity,
    ReflectX,
    ReflectY,
    Rotate90,
    Rotate180,
    Rotate270,
    Translate { dx: i32, dy: i32 },
}

#[derive(Debug, Clone, Copy)]
enum ColorOp {
    None,
    Replace { from: u8, to: u8 },
}

#[derive(Debug, Clone, Copy)]
struct CandidateTransform {
    geometry: Geometry,
    color: ColorOp,
}

struct SolveResult {
    prediction: Option<Grid>,
    matching_candidates: usize,
    distinct_predictions: usize,
    selected: Option<CandidateTransform>,
    candidates_checked: u64,
}

#[derive(Serialize)]
struct TaskResult {
    problem_id: String,
    exact_correct: bool,
    asserted: bool,
    matching_candidates: usize,
    distinct_predictions: usize,
    selected_transform: Option<String>,
    episode_id: String,
    receipt_id: String,
}

#[derive(Serialize)]
struct ArcExactReport {
    schema_version: u32,
    subject_revision: String,
    dataset_version: String,
    split: String,
    configuration_id: String,
    task_limit: Option<usize>,
    task_files_seen: usize,
    test_cases_evaluated: usize,
    aggregate: CapabilitySlice,
    tasks: Vec<TaskResult>,
    receipts: Vec<ReasoningQualificationReceipt>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("ARC exact-output qualification failed: {err}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let subject_revision = required_env("SYMTHAEA_SUBJECT_REVISION")?;
    let dataset_version = required_env("SYMTHAEA_ARC_DATASET_VERSION")?;
    let split = env::var("SYMTHAEA_ARC_SPLIT").unwrap_or_else(|_| "training".into());
    if split != "training" && split != "evaluation" {
        return Err("SYMTHAEA_ARC_SPLIT must be `training` or `evaluation`".into());
    }

    let data_dir = env::var("SYMTHAEA_ARC_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data/benchmarks/arc/repo/data"));
    let split_dir = data_dir.join(&split);
    if !split_dir.exists() {
        return Err(format!("ARC split directory not found: {}", split_dir.display()));
    }

    let results_path = env::var("SYMTHAEA_ARC_EXACT_RESULTS_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data/benchmarks/arc/exact-output-results.json"));
    let max_tasks = env::var("SYMTHAEA_ARC_MAX_TASKS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());

    let mut task_files: Vec<PathBuf> = fs::read_dir(&split_dir)
        .map_err(|err| format!("failed to list {}: {err}", split_dir.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    task_files.sort();
    if let Some(limit) = max_tasks {
        task_files.truncate(limit);
    }

    println!("ARC exact-output qualification");
    println!("subject:       {subject_revision}");
    println!("dataset:       {dataset_version}");
    println!("split:         {split}");
    println!("configuration: {CONFIGURATION_ID}");
    println!("task files:    {}", task_files.len());

    let candidates = canonical_candidates();
    println!("candidate grammar size: {}", candidates.len());

    let mut receipts = Vec::new();
    let mut task_results = Vec::new();

    for path in &task_files {
        // Qualification fails closed: a selected task may not silently disappear because its
        // file could not be read or parsed.
        let raw = fs::read(path)
            .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        let task = parse_task(&raw)
            .map_err(|err| format!("failed to parse {}: {err}", path.display()))?;

        let file_stem = path
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or_else(|| format!("task filename is not valid UTF-8: {}", path.display()))?;
        let raw_hash = blake3::hash(&raw).to_hex().to_string();
        let evidence = training_evidence(&task.train, file_stem)?;

        for (test_index, test_case) in task.test.iter().enumerate() {
            let started = Instant::now();
            // Critical qualification boundary: solve() receives training pairs and test INPUT,
            // never `test_case.expected`.
            let solve = solve(&task.train, &test_case.input, &candidates);
            let wall_time_us = started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
            let problem_id = format!("{file_stem}#test-{test_index}");
            let problem_hash = scoped_problem_hash(&raw_hash, test_index);

            let (outcome, asserted, selected_name, operation, output_refs) =
                if let Some(prediction) = &solve.prediction {
                    let value = serde_json::to_string(prediction)
                        .map_err(|err| format!("failed to encode prediction: {err}"))?;
                    // Conservative initial proxy: hypothesis ambiguity lowers confidence even
                    // when all surviving hypotheses happen to agree on this particular test grid.
                    // RQ-006 is responsible for empirical confidence calibration.
                    let confidence = 1.0 / solve.matching_candidates.max(1) as f64;
                    let selected_name = solve.selected.map(transform_name);
                    let operation = format!(
                        "exact-transform-search: {} training-consistent candidates, one distinct test prediction; assert {}",
                        solve.matching_candidates,
                        selected_name.as_deref().unwrap_or("canonical candidate")
                    );
                    (
                        ReasoningOutcome::Asserted { value, confidence },
                        true,
                        selected_name,
                        operation,
                        vec!["predicted-grid".into()],
                    )
                } else if solve.matching_candidates == 0 {
                    (
                        ReasoningOutcome::Abstained {
                            reason: AbstentionReason::Unidentified,
                            answerability: 0.0,
                        },
                        false,
                        None,
                        "exact-transform-search: no training-consistent candidate; abstain"
                            .into(),
                        vec![],
                    )
                } else {
                    (
                        ReasoningOutcome::Abstained {
                            reason: AbstentionReason::ConflictingEvidence,
                            answerability: 0.0,
                        },
                        false,
                        None,
                        format!(
                            "exact-transform-search: {} training-consistent candidates imply {} distinct test predictions; abstain",
                            solve.matching_candidates, solve.distinct_predictions
                        ),
                        vec![],
                    )
                };

            let episode = ReasoningEpisode::new(
                &subject_revision,
                CONFIGURATION_ID,
                ReasoningDomain::Abstraction,
                ReasoningProblemRef {
                    benchmark: "ARC-AGI exact output".into(),
                    benchmark_version: dataset_version.clone(),
                    split: split.clone(),
                    problem_id: problem_id.clone(),
                    problem_hash: format!("blake3:{problem_hash}"),
                },
                evidence.clone(),
                vec![],
                vec![ReasoningDecisionRecord {
                    operation,
                    input_refs: evidence.iter().map(|item| item.id.clone()).collect(),
                    output_refs,
                    verifier: Some("exact-grid-equality-v1".into()),
                }],
                outcome,
                ResourceUsage {
                    wall_time_us,
                    deliberation_steps: solve.candidates_checked,
                    tool_calls: 0,
                    model_tokens: 0,
                },
            )
            .map_err(|err| format!("invalid episode {problem_id}: {err}"))?;

            let exact_correct = solve
                .prediction
                .as_ref()
                .is_some_and(|prediction| prediction == &test_case.expected);
            let evaluation_lineage = format!(
                "arc-exact-v1:{dataset_version}:{split}:{problem_hash}:{}",
                blake3::hash(b"exact-grid-equality-v1").to_hex()
            );
            let receipt = evaluate_episode(
                &episode,
                &evaluation_lineage,
                &EpisodeJudgment {
                    exact_correct: Some(exact_correct),
                    task_score: None,
                },
            )
            .map_err(|err| format!("evaluation failed for {problem_id}: {err}"))?;
            let episode_id = episode
                .id()
                .map_err(|err| format!("episode identity failed for {problem_id}: {err}"))?;

            task_results.push(TaskResult {
                problem_id,
                exact_correct,
                asserted,
                matching_candidates: solve.matching_candidates,
                distinct_predictions: solve.distinct_predictions,
                selected_transform: selected_name,
                episode_id: episode_id.0,
                receipt_id: receipt.receipt_id.clone(),
            });
            receipts.push(receipt);
        }
    }

    let aggregate = aggregate_receipts(&receipts);
    println!("test cases:     {}", aggregate.episodes);
    println!("coverage:       {:.3}", aggregate.coverage);
    match aggregate.exact_accuracy {
        Some(value) => println!("exact accuracy: {:.3}", value),
        None => println!("exact accuracy: n/a"),
    }
    if let Some(value) = aggregate.selective_accuracy {
        println!("selective acc:  {:.3}", value);
    }

    let report = ArcExactReport {
        schema_version: RESULT_SCHEMA_VERSION,
        subject_revision,
        dataset_version,
        split,
        configuration_id: CONFIGURATION_ID.into(),
        task_limit: max_tasks,
        task_files_seen: task_files.len(),
        test_cases_evaluated: aggregate.episodes,
        aggregate,
        tasks: task_results,
        receipts,
    };
    if let Some(parent) = results_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    let encoded = serde_json::to_string_pretty(&report)
        .map_err(|err| format!("failed to encode result report: {err}"))?;
    fs::write(&results_path, encoded)
        .map_err(|err| format!("failed to write {}: {err}", results_path.display()))?;
    println!("receipt report: {}", results_path.display());
    Ok(())
}

fn required_env(name: &str) -> Result<String, String> {
    match env::var(name) {
        Ok(value) if !value.trim().is_empty() => Ok(value),
        _ => Err(format!("required environment variable {name} is missing")),
    }
}

fn parse_task(raw: &[u8]) -> Result<ArcTask, String> {
    let value: serde_json::Value =
        serde_json::from_slice(raw).map_err(|err| format!("invalid ARC JSON: {err}"))?;
    let train = value
        .get("train")
        .and_then(|value| value.as_array())
        .ok_or_else(|| "ARC task has no train array".to_string())?
        .iter()
        .map(parse_pair)
        .collect::<Result<Vec<_>, _>>()?;
    let test = value
        .get("test")
        .and_then(|value| value.as_array())
        .ok_or_else(|| "ARC task has no test array".to_string())?
        .iter()
        .map(|pair| {
            let parsed = parse_pair(pair)?;
            Ok(ArcTestCase {
                input: parsed.input,
                expected: parsed.output,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    if train.is_empty() || test.is_empty() {
        return Err("ARC task must contain at least one train and one test pair".into());
    }
    Ok(ArcTask { train, test })
}

fn parse_pair(value: &serde_json::Value) -> Result<GridPair, String> {
    Ok(GridPair {
        input: parse_grid(
            value
                .get("input")
                .ok_or_else(|| "pair has no input grid".to_string())?,
        )?,
        output: parse_grid(
            value
                .get("output")
                .ok_or_else(|| "pair has no output grid".to_string())?,
        )?,
    })
}

fn parse_grid(value: &serde_json::Value) -> Result<Grid, String> {
    let rows = value
        .as_array()
        .ok_or_else(|| "grid must be an array".to_string())?;
    if rows.is_empty() {
        return Err("grid must contain at least one row".into());
    }

    let mut grid = Vec::with_capacity(rows.len());
    let mut width = None;
    for row in rows {
        let cells = row
            .as_array()
            .ok_or_else(|| "grid row must be an array".to_string())?;
        if cells.is_empty() {
            return Err("grid rows must contain at least one cell".into());
        }
        if let Some(expected) = width {
            if cells.len() != expected {
                return Err("grid must be rectangular".into());
            }
        } else {
            width = Some(cells.len());
        }
        let parsed = cells
            .iter()
            .map(|cell| {
                let value = cell
                    .as_u64()
                    .ok_or_else(|| "grid cell must be an integer".to_string())?;
                if value > 9 {
                    return Err("ARC colors must be within 0..=9".into());
                }
                Ok(value as u8)
            })
            .collect::<Result<Vec<_>, String>>()?;
        grid.push(parsed);
    }
    Ok(grid)
}

fn training_evidence(train: &[GridPair], task_id: &str) -> Result<Vec<EvidenceRef>, String> {
    train
        .iter()
        .enumerate()
        .map(|(index, pair)| {
            let encoded = serde_json::to_vec(&(pair.input.clone(), pair.output.clone()))
                .map_err(|err| format!("failed to encode training evidence: {err}"))?;
            Ok(EvidenceRef {
                id: format!("train-{index}"),
                content_hash: format!("blake3:{}", blake3::hash(&encoded).to_hex()),
                provenance: format!("ARC task {task_id} training pair {index}"),
                independence_group: Some(format!("ARC-task:{task_id}")),
            })
        })
        .collect()
}

fn scoped_problem_hash(raw_hash: &str, test_index: usize) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(raw_hash.as_bytes());
    hasher.update(&u64::try_from(test_index).unwrap_or(u64::MAX).to_le_bytes());
    hasher.finalize().to_hex().to_string()
}

fn canonical_candidates() -> Vec<CandidateTransform> {
    let mut geometry = vec![
        Geometry::Identity,
        Geometry::ReflectX,
        Geometry::ReflectY,
        Geometry::Rotate90,
        Geometry::Rotate180,
        Geometry::Rotate270,
    ];
    for dy in -3..=3 {
        for dx in -3..=3 {
            if dx != 0 || dy != 0 {
                geometry.push(Geometry::Translate { dx, dy });
            }
        }
    }

    let mut colors = vec![ColorOp::None];
    for from in 0..=9 {
        for to in 0..=9 {
            if from != to {
                colors.push(ColorOp::Replace { from, to });
            }
        }
    }

    let mut out = Vec::with_capacity(geometry.len() * colors.len());
    for geometry in geometry {
        for color in &colors {
            out.push(CandidateTransform {
                geometry,
                color: *color,
            });
        }
    }
    out
}

fn solve(train: &[GridPair], test_input: &Grid, candidates: &[CandidateTransform]) -> SolveResult {
    let mut matching = Vec::new();
    let mut candidates_checked = 0u64;
    for candidate in candidates {
        let mut matches_all = true;
        for pair in train {
            candidates_checked = candidates_checked.saturating_add(1);
            if apply_transform(&pair.input, *candidate) != pair.output {
                matches_all = false;
                break;
            }
        }
        if matches_all {
            matching.push(*candidate);
        }
    }

    // Training-equivalent hypotheses are allowed to remain distinct, but they only justify an
    // asserted answer when they are prediction-equivalent on the held-out input. This prevents
    // canonical ordering from silently resolving genuine epistemic ambiguity.
    let mut predictions: Vec<(Grid, CandidateTransform)> = Vec::new();
    for candidate in &matching {
        let prediction = apply_transform(test_input, *candidate);
        if !predictions.iter().any(|(known, _)| known == &prediction) {
            predictions.push((prediction, *candidate));
        }
    }

    let (prediction, selected) = if predictions.len() == 1 {
        let (grid, candidate) = predictions.remove(0);
        (Some(grid), Some(candidate))
    } else {
        (None, None)
    };

    SolveResult {
        prediction,
        matching_candidates: matching.len(),
        distinct_predictions: predictions.len().max(usize::from(selected.is_some())),
        selected,
        candidates_checked,
    }
}

fn apply_transform(grid: &Grid, candidate: CandidateTransform) -> Grid {
    let geometry = match candidate.geometry {
        Geometry::Identity => grid.clone(),
        Geometry::ReflectX => GridEncoder::reflect_x(grid),
        Geometry::ReflectY => GridEncoder::reflect_y(grid),
        Geometry::Rotate90 => GridEncoder::rotate_90(grid),
        Geometry::Rotate180 => GridEncoder::rotate_90(&GridEncoder::rotate_90(grid)),
        Geometry::Rotate270 => {
            GridEncoder::rotate_90(&GridEncoder::rotate_90(&GridEncoder::rotate_90(grid)))
        }
        Geometry::Translate { dx, dy } => GridEncoder::translate_grid(grid, dx, dy, 0),
    };
    match candidate.color {
        ColorOp::None => geometry,
        ColorOp::Replace { from, to } => GridEncoder::color_replace(&geometry, from, to),
    }
}

fn transform_name(candidate: CandidateTransform) -> String {
    let geometry = match candidate.geometry {
        Geometry::Identity => "identity".into(),
        Geometry::ReflectX => "reflect-x".into(),
        Geometry::ReflectY => "reflect-y".into(),
        Geometry::Rotate90 => "rotate-90".into(),
        Geometry::Rotate180 => "rotate-180".into(),
        Geometry::Rotate270 => "rotate-270".into(),
        Geometry::Translate { dx, dy } => format!("translate({dx},{dy})"),
    };
    match candidate.color {
        ColorOp::None => geometry,
        ColorOp::Replace { from, to } => format!("{geometry}+color({from}->{to})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(input: Grid, output: Grid) -> GridPair {
        GridPair { input, output }
    }

    #[test]
    fn solver_never_needs_test_target_to_predict_reflection() {
        let train_input = vec![vec![1, 0, 2], vec![0, 3, 0]];
        let train_output = GridEncoder::reflect_x(&train_input);
        let test_input = vec![vec![4, 0, 1], vec![2, 3, 0]];
        let train = vec![pair(train_input, train_output)];
        let candidates = canonical_candidates();
        let result = solve(&train, &test_input, &candidates);
        assert_eq!(result.prediction, Some(GridEncoder::reflect_x(&test_input)));
        assert!(result.matching_candidates >= 1);
        assert_eq!(result.distinct_predictions, 1);
    }

    #[test]
    fn solver_abstains_when_output_is_outside_initial_grammar() {
        let input = vec![vec![1, 2], vec![3, 4]];
        let output = vec![vec![1, 2, 1, 2], vec![3, 4, 3, 4]];
        let train = vec![pair(input.clone(), output)];
        let candidates = canonical_candidates();
        let result = solve(&train, &input, &candidates);
        assert!(result.prediction.is_none());
        assert_eq!(result.matching_candidates, 0);
        assert_eq!(result.distinct_predictions, 0);
    }

    #[test]
    fn solver_abstains_when_training_consistent_hypotheses_disagree_on_test() {
        let symmetric = vec![vec![1, 0, 1], vec![2, 3, 2]];
        let train = vec![pair(symmetric.clone(), symmetric)];
        let test_input = vec![vec![1, 2, 0], vec![3, 4, 5]];
        let candidates = canonical_candidates();
        let result = solve(&train, &test_input, &candidates);
        assert!(result.matching_candidates > 1);
        assert!(result.distinct_predictions > 1);
        assert!(result.prediction.is_none());
        assert!(result.selected.is_none());
    }

    #[test]
    fn candidate_order_is_deterministic() {
        let left = canonical_candidates()
            .into_iter()
            .map(transform_name)
            .collect::<Vec<_>>();
        let right = canonical_candidates()
            .into_iter()
            .map(transform_name)
            .collect::<Vec<_>>();
        assert_eq!(left, right);
    }

    #[test]
    fn exact_grid_equality_is_strict() {
        let expected = vec![vec![1, 2], vec![3, 4]];
        assert_eq!(expected, vec![vec![1, 2], vec![3, 4]]);
        assert_ne!(expected, vec![vec![1, 2], vec![4, 3]]);
    }
}
