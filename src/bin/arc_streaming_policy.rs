// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! RQ-006Z-G: budget-stable streaming ARC policy baselines.
//!
//! This binary consumes only the target-stripped solver view produced by RQ-006Z-F. It creates one
//! complete semantic action trajectory per task/policy and then seals preregistered nested prefixes
//! of that trajectory. The policy never observes the stopping budget.

use serde::Serialize;
use serde_json::{Map, Value};
use std::collections::{BTreeSet, HashSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use symthaea::hdc::grid_encoder::GridEncoder;

type Grid = Vec<Vec<u8>>;

const SCHEMA_VERSION: u32 = 1;
const CONFIGURATION_ID: &str = "arc-streaming-baselines-v1";
const PROTOCOL_VERSION: &str = "rq006z-streaming-v1";
const ACTION_SPACE_VERSION: &str = "rq003-candidate-transform-v1";
const ACTION_SPACE_DOMAIN: &[u8] = b"symthaea/reasoning/arc-action-space/v1";
const PROBLEM_VISIBLE_DOMAIN: &[u8] = b"symthaea/reasoning/arc-stream-problem-visible/v1";
const EXECUTION_CONTEXT_DOMAIN: &[u8] = b"symthaea/reasoning/arc-stream-execution/v1";
const RANDOM_PRIORITY_DOMAIN: &[u8] = b"symthaea/reasoning/arc-stream-random-priority/v1";
const ACTION_DOMAIN: &[u8] = b"symthaea/reasoning/arc-stream-native-action/v1";
const TRACE_DOMAIN: &[u8] = b"symthaea/reasoning/arc-stream-trace/v1";
const PREFIX_DOMAIN: &[u8] = b"symthaea/reasoning/arc-stream-prefix/v1";
const GRID_DOMAIN: &[u8] = b"symthaea/reasoning/arc-grid/v1";
const SENTINEL: [[u8; 1]; 1] = [[0]];
const PREFIX_BUDGETS: [usize; 9] = [16, 32, 64, 128, 256, 512, 1024, 2048, 4914];

#[derive(Debug, Clone)]
struct GridPair {
    input: Grid,
    output: Grid,
}

#[derive(Debug, Clone)]
struct SolverTask {
    train: Vec<GridPair>,
    test_inputs: Vec<Grid>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Geometry {
    Identity,
    ReflectX,
    ReflectY,
    Rotate90,
    Rotate180,
    Rotate270,
    Translate { dx: i32, dy: i32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColorOp {
    None,
    Replace { from: u8, to: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CandidateTransform {
    canonical_id: usize,
    geometry: Geometry,
    color: ColorOp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PolicyKind {
    Canonical,
    SemanticHashRandom,
}

impl PolicyKind {
    const fn id(self) -> &'static str {
        match self {
            Self::Canonical => "canonical-stream-v1",
            Self::SemanticHashRandom => "semantic-hash-random-v2",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum LocalDecision {
    Asserted(String),
    Abstained(AbstentionKind),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AbstentionKind {
    NoTrainingConsistentCandidate,
    ConflictingPredictions,
}

#[derive(Debug, Clone, Serialize)]
struct StreamAction {
    step: usize,
    candidate_id: usize,
    candidate_name: String,
    pairs_checked: usize,
    first_mismatch_index: Option<usize>,
    training_consistent: bool,
    test_prediction_digest: Option<String>,
    action_commitment: String,
}

#[derive(Debug, Clone, Serialize)]
struct PrefixReceipt {
    budget: usize,
    candidates_evaluated: usize,
    training_pair_checks: usize,
    asserted: bool,
    prediction_digest: Option<String>,
    distinct_prediction_count: usize,
    prefix_commitment: String,
}

#[derive(Debug, Serialize)]
struct StreamEpisode {
    problem_id: String,
    policy_id: String,
    seed_root: Option<u64>,
    problem_visible_commitment: String,
    execution_context_commitment: String,
    trace_commitment: String,
    actions: Vec<StreamAction>,
    prefixes: Vec<PrefixReceipt>,
}

#[derive(Debug, Serialize)]
struct StreamingPolicyReport {
    schema_version: u32,
    configuration_id: String,
    protocol_version: String,
    subject_revision: String,
    action_space_version: String,
    action_space_commitment: String,
    grammar_size: usize,
    prefix_budgets: Vec<usize>,
    random_seed_root: u64,
    task_files_seen: usize,
    test_cases_evaluated: usize,
    episodes: Vec<StreamEpisode>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("ARC streaming policy qualification failed: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let subject_revision = required_full_revision("SYMTHAEA_SUBJECT_REVISION")?;
    let solver_root = PathBuf::from(required_env("SYMTHAEA_ARC_SOLVER_VIEW_ROOT")?);
    let output_path = PathBuf::from(required_env("SYMTHAEA_ARC_STREAM_RESULTS_PATH")?);
    let random_seed_root = required_u64_env("SYMTHAEA_ARC_POLICY_SEED")?;
    let max_tasks = optional_usize_env("SYMTHAEA_ARC_MAX_TASKS")?;

    let candidates = canonical_candidates();
    validate_action_space(&candidates)?;
    if candidates.len() != *PREFIX_BUDGETS.last().expect("fixed prefix list is nonempty") {
        return Err("full prefix budget does not equal frozen grammar size".into());
    }
    let action_space_commitment = action_space_commitment(&candidates);
    let split_dir = solver_root.join("training");
    let task_paths = selected_task_files(&split_dir, max_tasks)?;

    let mut episodes = Vec::new();
    let mut test_cases_evaluated = 0usize;
    for path in &task_paths {
        let bytes = fs::read(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let task = parse_solver_task(&bytes)?;
        let stem = path
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or_else(|| format!("task filename is not UTF-8: {}", path.display()))?;

        for (test_index, test_input) in task.test_inputs.iter().enumerate() {
            let problem_id = format!("{stem}#test-{test_index}");
            let visible = problem_visible_commitment(
                &problem_id,
                &task.train,
                test_input,
                &action_space_commitment,
            );
            episodes.push(run_stream(
                &problem_id,
                &task.train,
                test_input,
                &candidates,
                &visible,
                PolicyKind::Canonical,
                None,
            )?);
            episodes.push(run_stream(
                &problem_id,
                &task.train,
                test_input,
                &candidates,
                &visible,
                PolicyKind::SemanticHashRandom,
                Some(random_seed_root),
            )?);
            test_cases_evaluated = test_cases_evaluated.saturating_add(1);
        }
    }

    let report = StreamingPolicyReport {
        schema_version: SCHEMA_VERSION,
        configuration_id: CONFIGURATION_ID.into(),
        protocol_version: PROTOCOL_VERSION.into(),
        subject_revision,
        action_space_version: ACTION_SPACE_VERSION.into(),
        action_space_commitment,
        grammar_size: candidates.len(),
        prefix_budgets: PREFIX_BUDGETS.to_vec(),
        random_seed_root,
        task_files_seen: task_paths.len(),
        test_cases_evaluated,
        episodes,
    };

    if let Some(parent) = output_path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    fs::write(
        &output_path,
        serde_json::to_string_pretty(&report)
            .map_err(|error| format!("failed to encode streaming report: {error}"))?,
    )
    .map_err(|error| format!("failed to write {}: {error}", output_path.display()))?;

    println!("ARC budget-stable streaming baselines");
    println!("subject:      {}", report.subject_revision);
    println!("tasks:        {}", report.task_files_seen);
    println!("test cases:   {}", report.test_cases_evaluated);
    println!("grammar:      {}", report.grammar_size);
    println!("prefixes:     {:?}", report.prefix_budgets);
    println!("report:       {}", output_path.display());
    Ok(())
}

fn run_stream(
    problem_id: &str,
    train: &[GridPair],
    test_input: &Grid,
    candidates: &[CandidateTransform],
    visible_commitment: &str,
    policy: PolicyKind,
    seed_root: Option<u64>,
) -> Result<StreamEpisode, String> {
    let ordered = ordered_candidates(candidates, policy, visible_commitment, seed_root)?;
    if ordered.len() != candidates.len() {
        return Err("stream policy did not produce the complete action space".into());
    }

    let execution_context = execution_context_commitment(
        visible_commitment,
        policy,
        seed_root,
        ordered.len(),
    );
    let mut actions = Vec::with_capacity(ordered.len());
    for (step, candidate) in ordered.into_iter().enumerate() {
        actions.push(evaluate_action(train, test_input, candidate, step));
    }
    validate_trace(&actions, candidates.len())?;
    let trace_commitment = trace_commitment(&execution_context, &actions);
    let prefixes = PREFIX_BUDGETS
        .iter()
        .map(|budget| build_prefix(&execution_context, &actions, *budget))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(StreamEpisode {
        problem_id: problem_id.into(),
        policy_id: policy.id().into(),
        seed_root,
        problem_visible_commitment: visible_commitment.into(),
        execution_context_commitment: execution_context,
        trace_commitment,
        actions,
        prefixes,
    })
}

fn ordered_candidates(
    candidates: &[CandidateTransform],
    policy: PolicyKind,
    visible_commitment: &str,
    seed_root: Option<u64>,
) -> Result<Vec<CandidateTransform>, String> {
    validate_action_space(candidates)?;
    let mut ordered = candidates.to_vec();
    match policy {
        PolicyKind::Canonical => {
            if seed_root.is_some() {
                return Err("canonical streaming policy must not receive a seed".into());
            }
            ordered.sort_by_key(|candidate| candidate.canonical_id);
        }
        PolicyKind::SemanticHashRandom => {
            let seed = seed_root.ok_or_else(|| "semantic-hash random policy requires a seed".to_string())?;
            ordered.sort_by(|left, right| {
                random_priority(visible_commitment, seed, *left)
                    .cmp(&random_priority(visible_commitment, seed, *right))
                    .then_with(|| transform_name(*left).cmp(&transform_name(*right)))
            });
        }
    }
    Ok(ordered)
}

fn random_priority(
    visible_commitment: &str,
    seed_root: u64,
    candidate: CandidateTransform,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, RANDOM_PRIORITY_DOMAIN);
    hash_str(&mut hasher, visible_commitment);
    hash_u64(&mut hasher, seed_root);
    hash_str(&mut hasher, &transform_name(candidate));
    *hasher.finalize().as_bytes()
}

fn evaluate_action(
    train: &[GridPair],
    test_input: &Grid,
    candidate: CandidateTransform,
    step: usize,
) -> StreamAction {
    let mut pairs_checked = 0usize;
    let mut first_mismatch_index = None;
    for (index, pair) in train.iter().enumerate() {
        pairs_checked = pairs_checked.saturating_add(1);
        if apply_transform(&pair.input, candidate) != pair.output {
            first_mismatch_index = Some(index);
            break;
        }
    }
    let training_consistent = first_mismatch_index.is_none();
    let test_prediction_digest = training_consistent
        .then(|| grid_digest(&apply_transform(test_input, candidate)));
    let candidate_name = transform_name(candidate);
    let action_commitment = action_commitment(
        step,
        candidate.canonical_id,
        &candidate_name,
        pairs_checked,
        first_mismatch_index,
        training_consistent,
        test_prediction_digest.as_deref(),
    );
    StreamAction {
        step,
        candidate_id: candidate.canonical_id,
        candidate_name,
        pairs_checked,
        first_mismatch_index,
        training_consistent,
        test_prediction_digest,
        action_commitment,
    }
}

fn validate_trace(actions: &[StreamAction], grammar_size: usize) -> Result<(), String> {
    if actions.len() != grammar_size {
        return Err("stream trace does not cover the full grammar".into());
    }
    let mut ids = HashSet::with_capacity(grammar_size);
    for (expected_step, action) in actions.iter().enumerate() {
        if action.step != expected_step {
            return Err(format!("stream step {} is out of order", action.step));
        }
        if action.candidate_id >= grammar_size || !ids.insert(action.candidate_id) {
            return Err(format!("invalid or repeated candidate {}", action.candidate_id));
        }
        if action.training_consistent != action.first_mismatch_index.is_none() {
            return Err(format!("action {} mismatch semantics are inconsistent", action.step));
        }
        if action.training_consistent != action.test_prediction_digest.is_some() {
            return Err(format!("action {} prediction semantics are inconsistent", action.step));
        }
        let expected = action_commitment(
            action.step,
            action.candidate_id,
            &action.candidate_name,
            action.pairs_checked,
            action.first_mismatch_index,
            action.training_consistent,
            action.test_prediction_digest.as_deref(),
        );
        if action.action_commitment != expected {
            return Err(format!("action {} commitment mismatch", action.step));
        }
    }
    Ok(())
}

fn build_prefix(
    execution_context: &str,
    actions: &[StreamAction],
    budget: usize,
) -> Result<PrefixReceipt, String> {
    if budget == 0 || budget > actions.len() {
        return Err(format!("invalid prefix budget {budget}"));
    }
    let prefix = &actions[..budget];
    let mut predictions = Vec::<String>::new();
    let mut training_pair_checks = 0usize;
    for action in prefix {
        training_pair_checks = training_pair_checks.saturating_add(action.pairs_checked);
        if let Some(digest) = &action.test_prediction_digest {
            if !predictions.contains(digest) {
                predictions.push(digest.clone());
            }
        }
    }
    let distinct_prediction_count = predictions.len();
    let decision = decision_from_prediction_digests(predictions);
    let (asserted, prediction_digest) = match &decision {
        LocalDecision::Asserted(digest) => (true, Some(digest.clone())),
        LocalDecision::Abstained(_) => (false, None),
    };
    let commitment = prefix_commitment(execution_context, budget, prefix, &decision);
    Ok(PrefixReceipt {
        budget,
        candidates_evaluated: prefix.len(),
        training_pair_checks,
        asserted,
        prediction_digest,
        distinct_prediction_count,
        prefix_commitment: commitment,
    })
}

fn decision_from_prediction_digests(mut predictions: Vec<String>) -> LocalDecision {
    match predictions.len() {
        0 => LocalDecision::Abstained(AbstentionKind::NoTrainingConsistentCandidate),
        1 => LocalDecision::Asserted(predictions.remove(0)),
        _ => LocalDecision::Abstained(AbstentionKind::ConflictingPredictions),
    }
}

fn problem_visible_commitment(
    problem_id: &str,
    train: &[GridPair],
    test_input: &Grid,
    action_space_commitment: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, PROBLEM_VISIBLE_DOMAIN);
    hash_str(&mut hasher, problem_id);
    hash_str(&mut hasher, action_space_commitment);
    hash_u64(&mut hasher, train.len() as u64);
    for pair in train {
        hash_grid(&mut hasher, &pair.input);
        hash_grid(&mut hasher, &pair.output);
    }
    hash_grid(&mut hasher, test_input);
    hasher.finalize().to_hex().to_string()
}

fn execution_context_commitment(
    visible_commitment: &str,
    policy: PolicyKind,
    seed_root: Option<u64>,
    maximum_trace_length: usize,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, EXECUTION_CONTEXT_DOMAIN);
    hash_str(&mut hasher, PROTOCOL_VERSION);
    hash_str(&mut hasher, visible_commitment);
    hash_str(&mut hasher, policy.id());
    match seed_root {
        Some(seed) => {
            hasher.update(&[1]);
            hash_u64(&mut hasher, seed);
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hash_u64(&mut hasher, maximum_trace_length as u64);
    hasher.finalize().to_hex().to_string()
}

fn action_commitment(
    step: usize,
    candidate_id: usize,
    candidate_name: &str,
    pairs_checked: usize,
    first_mismatch_index: Option<usize>,
    training_consistent: bool,
    prediction_digest: Option<&str>,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, ACTION_DOMAIN);
    hash_u64(&mut hasher, step as u64);
    hash_u64(&mut hasher, candidate_id as u64);
    hash_str(&mut hasher, candidate_name);
    hash_u64(&mut hasher, pairs_checked as u64);
    match first_mismatch_index {
        Some(index) => {
            hasher.update(&[1]);
            hash_u64(&mut hasher, index as u64);
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hasher.update(&[u8::from(training_consistent)]);
    match prediction_digest {
        Some(digest) => {
            hasher.update(&[1]);
            hash_str(&mut hasher, digest);
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hasher.finalize().to_hex().to_string()
}

fn trace_commitment(execution_context: &str, actions: &[StreamAction]) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, TRACE_DOMAIN);
    hash_str(&mut hasher, execution_context);
    hash_u64(&mut hasher, actions.len() as u64);
    for action in actions {
        hash_str(&mut hasher, &action.action_commitment);
    }
    hasher.finalize().to_hex().to_string()
}

fn prefix_commitment(
    execution_context: &str,
    budget: usize,
    actions: &[StreamAction],
    decision: &LocalDecision,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, PREFIX_DOMAIN);
    hash_str(&mut hasher, execution_context);
    hash_u64(&mut hasher, budget as u64);
    hash_u64(&mut hasher, actions.len() as u64);
    for action in actions {
        hash_str(&mut hasher, &action.action_commitment);
    }
    match decision {
        LocalDecision::Asserted(digest) => {
            hasher.update(&[1]);
            hash_str(&mut hasher, digest);
        }
        LocalDecision::Abstained(reason) => {
            hasher.update(&[0]);
            hash_u64(
                &mut hasher,
                match reason {
                    AbstentionKind::NoTrainingConsistentCandidate => 0,
                    AbstentionKind::ConflictingPredictions => 1,
                },
            );
        }
    }
    hasher.finalize().to_hex().to_string()
}

fn action_space_commitment(candidates: &[CandidateTransform]) -> String {
    let mut canonical = candidates.to_vec();
    canonical.sort_by_key(|candidate| candidate.canonical_id);
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, ACTION_SPACE_DOMAIN);
    hash_str(&mut hasher, ACTION_SPACE_VERSION);
    hash_u64(&mut hasher, canonical.len() as u64);
    for candidate in canonical {
        hash_u64(&mut hasher, candidate.canonical_id as u64);
        hash_str(&mut hasher, &transform_name(candidate));
    }
    hasher.finalize().to_hex().to_string()
}

fn validate_action_space(candidates: &[CandidateTransform]) -> Result<(), String> {
    if candidates.is_empty() {
        return Err("candidate grammar must not be empty".into());
    }
    let mut ids = BTreeSet::new();
    let mut names = BTreeSet::new();
    for candidate in candidates {
        if !ids.insert(candidate.canonical_id) {
            return Err(format!("duplicate canonical candidate id {}", candidate.canonical_id));
        }
        let name = transform_name(*candidate);
        if !names.insert(name.clone()) {
            return Err(format!("duplicate semantic candidate identity {name}"));
        }
    }
    for (expected, actual) in ids.into_iter().enumerate() {
        if expected != actual {
            return Err(format!("candidate ids are not contiguous: expected {expected}, got {actual}"));
        }
    }
    Ok(())
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
    let mut candidates = Vec::with_capacity(geometry.len() * colors.len());
    for geometry in geometry {
        for color in &colors {
            let canonical_id = candidates.len();
            candidates.push(CandidateTransform {
                canonical_id,
                geometry,
                color: *color,
            });
        }
    }
    candidates
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

fn parse_solver_task(bytes: &[u8]) -> Result<SolverTask, String> {
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("invalid solver-view task JSON: {error}"))?;
    let object = object(&value, "solver-view task")?;
    let train = require_array(object, "train", "solver-view task")?
        .iter()
        .map(parse_pair)
        .collect::<Result<Vec<_>, _>>()?;
    if train.is_empty() {
        return Err("solver-view task requires at least one training pair".into());
    }
    let tests = require_array(object, "test", "solver-view task")?;
    if tests.is_empty() {
        return Err("solver-view task requires at least one test pair".into());
    }
    let mut test_inputs = Vec::with_capacity(tests.len());
    for value in tests {
        let pair = object(value, "solver-view test pair")?;
        let input = parse_grid(
            pair.get("input")
                .ok_or_else(|| "solver-view test pair missing input".to_string())?,
        )?;
        let output = parse_grid(
            pair.get("output")
                .ok_or_else(|| "solver-view test pair missing sentinel output".to_string())?,
        )?;
        if output != SENTINEL {
            return Err("solver-view test output is not the fixed public sentinel".into());
        }
        test_inputs.push(input);
    }
    Ok(SolverTask { train, test_inputs })
}

fn parse_pair(value: &Value) -> Result<GridPair, String> {
    let object = object(value, "training pair")?;
    Ok(GridPair {
        input: parse_grid(
            object
                .get("input")
                .ok_or_else(|| "training pair missing input".to_string())?,
        )?,
        output: parse_grid(
            object
                .get("output")
                .ok_or_else(|| "training pair missing output".to_string())?,
        )?,
    })
}

fn parse_grid(value: &Value) -> Result<Grid, String> {
    let rows = value
        .as_array()
        .ok_or_else(|| "grid must be an array".to_string())?;
    if rows.is_empty() || rows.len() > 30 {
        return Err("grid height must be within 1..=30".into());
    }
    let mut result = Vec::with_capacity(rows.len());
    let mut width = None;
    for row in rows {
        let cells = row
            .as_array()
            .ok_or_else(|| "grid row must be an array".to_string())?;
        if cells.is_empty() || cells.len() > 30 {
            return Err("grid width must be within 1..=30".into());
        }
        if let Some(expected) = width {
            if cells.len() != expected {
                return Err("grid must be rectangular".into());
            }
        } else {
            width = Some(cells.len());
        }
        let mut parsed = Vec::with_capacity(cells.len());
        for cell in cells {
            let value = cell
                .as_u64()
                .ok_or_else(|| "grid cell must be an unsigned integer".to_string())?;
            if value > 9 {
                return Err("grid colors must be within 0..=9".into());
            }
            parsed.push(value as u8);
        }
        result.push(parsed);
    }
    Ok(result)
}

fn selected_task_files(split_dir: &Path, limit: Option<usize>) -> Result<Vec<PathBuf>, String> {
    let mut files = fs::read_dir(split_dir)
        .map_err(|error| format!("failed to list {}: {error}", split_dir.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("failed to enumerate {}: {error}", split_dir.display()))?;
    files.sort();
    if files.iter().any(|path| path.extension().and_then(|value| value.to_str()) != Some("json")) {
        return Err("solver-view training directory contains a non-JSON entry".into());
    }
    if let Some(limit) = limit {
        if limit == 0 {
            return Err("SYMTHAEA_ARC_MAX_TASKS must be positive when set".into());
        }
        files.truncate(limit);
    }
    if files.is_empty() {
        return Err(format!("no solver-view tasks selected from {}", split_dir.display()));
    }
    Ok(files)
}

fn grid_digest(grid: &Grid) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, GRID_DOMAIN);
    hash_grid(&mut hasher, grid);
    hasher.finalize().to_hex().to_string()
}

fn hash_grid(hasher: &mut blake3::Hasher, grid: &Grid) {
    hash_u64(hasher, grid.len() as u64);
    for row in grid {
        hash_u64(hasher, row.len() as u64);
        for cell in row {
            hasher.update(&[*cell]);
        }
    }
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    hash_bytes(hasher, value.as_bytes());
}

fn hash_bytes(hasher: &mut blake3::Hasher, value: &[u8]) {
    hash_u64(hasher, value.len() as u64);
    hasher.update(value);
}

fn hash_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))
}

fn require_array<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<&'a Vec<Value>, String> {
    object
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label} field {key} must be an array"))
}

fn required_env(name: &str) -> Result<String, String> {
    match env::var(name) {
        Ok(value) if !value.trim().is_empty() => Ok(value.trim().to_string()),
        Ok(_) => Err(format!("required environment variable {name} is empty")),
        Err(error) => Err(format!("required environment variable {name} is missing: {error}")),
    }
}

fn required_full_revision(name: &str) -> Result<String, String> {
    let value = required_env(name)?;
    if value.len() != 40 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{name} must be an exact 40-hex Git revision"));
    }
    Ok(value.to_ascii_lowercase())
}

fn required_u64_env(name: &str) -> Result<u64, String> {
    required_env(name)?
        .parse::<u64>()
        .map_err(|error| format!("{name} must be an unsigned integer: {error}"))
}

fn optional_usize_env(name: &str) -> Result<Option<usize>, String> {
    match env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map(Some)
            .map_err(|error| format!("{name} must be an unsigned integer: {error}")),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(format!("failed to read {name}: {error}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(input: Grid, output: Grid) -> GridPair {
        GridPair { input, output }
    }

    fn small_candidates() -> Vec<CandidateTransform> {
        vec![
            CandidateTransform {
                canonical_id: 0,
                geometry: Geometry::Identity,
                color: ColorOp::None,
            },
            CandidateTransform {
                canonical_id: 1,
                geometry: Geometry::ReflectX,
                color: ColorOp::None,
            },
            CandidateTransform {
                canonical_id: 2,
                geometry: Geometry::ReflectY,
                color: ColorOp::None,
            },
            CandidateTransform {
                canonical_id: 3,
                geometry: Geometry::Rotate180,
                color: ColorOp::None,
            },
        ]
    }

    #[test]
    fn frozen_grammar_size_and_anchor_order_match_rq003() {
        let candidates = canonical_candidates();
        assert_eq!(candidates.len(), 54 * 91);
        assert_eq!(transform_name(candidates[0]), "identity");
        assert_eq!(transform_name(candidates[91]), "reflect-x");
        assert_eq!(transform_name(candidates[546]), "translate(-3,-3)");
        assert_eq!(
            transform_name(*candidates.last().unwrap()),
            "translate(3,3)+color(9->8)"
        );
        validate_action_space(&candidates).unwrap();
    }

    #[test]
    fn problem_visible_commitment_has_no_budget_parameter() {
        let train = vec![pair(vec![vec![1]], vec![vec![1]])];
        let input = vec![vec![2]];
        let action_space = action_space_commitment(&small_candidates());
        let a = problem_visible_commitment("x#test-0", &train, &input, &action_space);
        let b = problem_visible_commitment("x#test-0", &train, &input, &action_space);
        assert_eq!(a, b);
    }

    #[test]
    fn semantic_random_order_is_candidate_vector_permutation_invariant() {
        let candidates = canonical_candidates();
        let visible = "0123456789abcdef";
        let a = ordered_candidates(
            &candidates,
            PolicyKind::SemanticHashRandom,
            visible,
            Some(2797608998),
        )
        .unwrap();
        let mut permuted = candidates.clone();
        permuted.reverse();
        let b = ordered_candidates(
            &permuted,
            PolicyKind::SemanticHashRandom,
            visible,
            Some(2797608998),
        )
        .unwrap();
        let names_a = a.into_iter().map(transform_name).collect::<Vec<_>>();
        let names_b = b.into_iter().map(transform_name).collect::<Vec<_>>();
        assert_eq!(names_a, names_b);
    }

    #[test]
    fn semantic_random_prefix_is_stable_across_larger_stopping_points() {
        let candidates = canonical_candidates();
        let ordered = ordered_candidates(
            &candidates,
            PolicyKind::SemanticHashRandom,
            "visible",
            Some(1234),
        )
        .unwrap();
        assert_eq!(&ordered[..16], &ordered[..32][..16]);
        assert_eq!(&ordered[..128], &ordered[..256][..128]);
    }

    #[test]
    fn semantic_random_seed_changes_trajectory() {
        let candidates = canonical_candidates();
        let a = ordered_candidates(
            &candidates,
            PolicyKind::SemanticHashRandom,
            "visible",
            Some(1),
        )
        .unwrap();
        let b = ordered_candidates(
            &candidates,
            PolicyKind::SemanticHashRandom,
            "visible",
            Some(2),
        )
        .unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn duplicate_candidate_identity_fails_closed() {
        let mut candidates = small_candidates();
        candidates[3].canonical_id = 2;
        assert!(validate_action_space(&candidates).is_err());
    }

    #[test]
    fn prefix_receipts_are_nested_over_one_fixed_trace() {
        let train = vec![pair(
            vec![vec![1, 0], vec![0, 0]],
            vec![vec![1, 0], vec![0, 0]],
        )];
        let input = vec![vec![2, 0], vec![3, 0]];
        let candidates = canonical_candidates();
        let visible = problem_visible_commitment(
            "fixture#test-0",
            &train,
            &input,
            &action_space_commitment(&candidates),
        );
        let episode = run_stream(
            "fixture#test-0",
            &train,
            &input,
            &candidates,
            &visible,
            PolicyKind::SemanticHashRandom,
            Some(44),
        )
        .unwrap();
        for receipt in &episode.prefixes {
            assert_eq!(receipt.candidates_evaluated, receipt.budget);
            assert_eq!(
                receipt.prefix_commitment,
                build_prefix(
                    &episode.execution_context_commitment,
                    &episode.actions,
                    receipt.budget,
                )
                .unwrap()
                .prefix_commitment
            );
        }
    }

    #[test]
    fn full_prefix_matches_decision_from_complete_action_set() {
        let train_grid = vec![vec![0, 0, 0], vec![0, 0, 0]];
        let train = vec![pair(train_grid.clone(), train_grid)];
        let input = vec![vec![1, 0, 2], vec![3, 4, 0]];
        let candidates = canonical_candidates();
        let ordered = ordered_candidates(&candidates, PolicyKind::Canonical, "visible", None).unwrap();
        let actions = ordered
            .into_iter()
            .enumerate()
            .map(|(step, candidate)| evaluate_action(&train, &input, candidate, step))
            .collect::<Vec<_>>();
        let prefix = build_prefix("context", &actions, actions.len()).unwrap();
        let predictions = actions
            .iter()
            .filter_map(|action| action.test_prediction_digest.clone())
            .collect::<BTreeSet<_>>();
        assert_eq!(prefix.distinct_prediction_count, predictions.len());
        assert_eq!(prefix.asserted, predictions.len() == 1);
    }

    #[test]
    fn prefix_commitment_changes_with_budget() {
        let train = vec![pair(vec![vec![1]], vec![vec![1]])];
        let input = vec![vec![2]];
        let candidates = canonical_candidates();
        let actions = candidates
            .iter()
            .copied()
            .enumerate()
            .map(|(step, candidate)| evaluate_action(&train, &input, candidate, step))
            .collect::<Vec<_>>();
        let a = build_prefix("context", &actions, 16).unwrap();
        let b = build_prefix("context", &actions, 32).unwrap();
        assert_ne!(a.prefix_commitment, b.prefix_commitment);
    }
}
