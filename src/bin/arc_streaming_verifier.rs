// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Independent verifier for RQ-006Z-G budget-stable ARC streaming traces.
//!
//! This executable intentionally duplicates the frozen streaming semantics rather than importing
//! producer helpers. It receives only target-stripped solver-view bytes plus a sealed streaming
//! report and independently reconstructs every action trajectory and prefix receipt.

use serde::Serialize;
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use symthaea::hdc::grid_encoder::GridEncoder;

type Grid = Vec<Vec<u8>>;

const SCHEMA_VERSION: u32 = 1;
const VERIFIER_DOMAIN: &str = "symthaea/reasoning/arc-stream-verifier/v1";
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

#[derive(Debug, Clone)]
struct ReplayAction {
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
struct VerifiedEpisode {
    problem_id: String,
    policy_id: String,
    problem_visible_commitment: String,
    execution_context_commitment: String,
    trace_commitment: String,
    prefixes_verified: usize,
    full_prefix_matches_exhaustive: bool,
}

#[derive(Debug, Serialize)]
struct StreamingVerificationReport {
    schema_version: u32,
    domain: String,
    subject_revision: String,
    source_report_blake3: String,
    action_space_commitment: String,
    grammar_size: usize,
    prefix_budgets: Vec<usize>,
    expected_problem_count: usize,
    episodes_verified: usize,
    episodes: Vec<VerifiedEpisode>,
    commitment: String,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("ARC streaming verification failed: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let subject_revision = required_full_revision("SYMTHAEA_SUBJECT_REVISION")?;
    let solver_root = PathBuf::from(required_env("SYMTHAEA_ARC_SOLVER_VIEW_ROOT")?);
    let source_path = PathBuf::from(required_env("SYMTHAEA_ARC_STREAM_RESULTS_PATH")?);
    let output_path = PathBuf::from(required_env("SYMTHAEA_ARC_STREAM_VERIFY_PATH")?);

    let source_bytes = fs::read(&source_path)
        .map_err(|error| format!("failed to read {}: {error}", source_path.display()))?;
    let source: Value = serde_json::from_slice(&source_bytes)
        .map_err(|error| format!("invalid streaming report JSON: {error}"))?;
    let source_obj = object(&source, "streaming report")?;

    if require_u64(source_obj, "schema_version", "streaming report")? != 1 {
        return Err("unexpected streaming report schema".into());
    }
    require_eq_str(
        source_obj,
        "configuration_id",
        CONFIGURATION_ID,
        "streaming report",
    )?;
    require_eq_str(
        source_obj,
        "protocol_version",
        PROTOCOL_VERSION,
        "streaming report",
    )?;
    require_eq_str(
        source_obj,
        "subject_revision",
        &subject_revision,
        "streaming report",
    )?;
    require_eq_str(
        source_obj,
        "action_space_version",
        ACTION_SPACE_VERSION,
        "streaming report",
    )?;

    let candidates = canonical_candidates();
    validate_action_space(&candidates)?;
    let action_space = action_space_commitment(&candidates);
    require_eq_str(
        source_obj,
        "action_space_commitment",
        &action_space,
        "streaming report",
    )?;
    if require_usize(source_obj, "grammar_size", "streaming report")? != candidates.len() {
        return Err("streaming report grammar size mismatch".into());
    }
    let published_budgets = require_array(source_obj, "prefix_budgets", "streaming report")?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| "prefix budget must be an unsigned integer".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if published_budgets != PREFIX_BUDGETS {
        return Err(format!("unexpected prefix budgets: {published_budgets:?}"));
    }
    let random_seed_root = require_u64(source_obj, "random_seed_root", "streaming report")?;

    let tasks = load_solver_tasks(&solver_root)?;
    if require_usize(source_obj, "task_files_seen", "streaming report")? != tasks.len() {
        return Err("streaming report task file count mismatch".into());
    }
    let expected_problem_ids = expected_problem_ids(&tasks);
    if require_usize(source_obj, "test_cases_evaluated", "streaming report")?
        != expected_problem_ids.len()
    {
        return Err("streaming report test-case count mismatch".into());
    }

    let episode_values = require_array(source_obj, "episodes", "streaming report")?;
    if episode_values.len() != expected_problem_ids.len().saturating_mul(2) {
        return Err("streaming report must contain exactly two policy episodes per problem".into());
    }

    let mut seen_pairs = BTreeSet::new();
    let mut policies_by_problem: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut verified = Vec::with_capacity(episode_values.len());

    for episode_value in episode_values {
        let episode = object(episode_value, "stream episode")?;
        let problem_id = require_str(episode, "problem_id", "stream episode")?.to_string();
        if !expected_problem_ids.contains(&problem_id) {
            return Err(format!("stream episode references unexpected problem {problem_id}"));
        }
        let policy_id = require_str(episode, "policy_id", "stream episode")?;
        let policy = match policy_id {
            "canonical-stream-v1" => PolicyKind::Canonical,
            "semantic-hash-random-v2" => PolicyKind::SemanticHashRandom,
            other => return Err(format!("unsupported streaming policy {other}")),
        };
        if !seen_pairs.insert((problem_id.clone(), policy_id.to_string())) {
            return Err(format!("duplicate stream episode {problem_id}/{policy_id}"));
        }
        policies_by_problem
            .entry(problem_id.clone())
            .or_default()
            .insert(policy_id.to_string());

        let seed_root = optional_u64(episode, "seed_root", "stream episode")?;
        match policy {
            PolicyKind::Canonical if seed_root.is_some() => {
                return Err("canonical stream episode must not carry a seed".into());
            }
            PolicyKind::SemanticHashRandom if seed_root != Some(random_seed_root) => {
                return Err("random-v2 stream episode seed does not match report seed root".into());
            }
            _ => {}
        }

        let (stem, test_index) = parse_problem_id(&problem_id)?;
        let task = tasks
            .get(stem)
            .ok_or_else(|| format!("unknown solver-view task stem {stem}"))?;
        let test_input = task
            .test_inputs
            .get(test_index)
            .ok_or_else(|| format!("test index out of range for {problem_id}"))?;
        let visible = problem_visible_commitment(&problem_id, &task.train, test_input, &action_space);
        require_eq_str(
            episode,
            "problem_visible_commitment",
            &visible,
            "stream episode",
        )?;

        let expected_execution = execution_context_commitment(
            &visible,
            policy,
            seed_root,
            candidates.len(),
        );
        require_eq_str(
            episode,
            "execution_context_commitment",
            &expected_execution,
            "stream episode",
        )?;

        let ordered = ordered_candidates(&candidates, policy, &visible, seed_root)?;
        let action_values = require_array(episode, "actions", "stream episode")?;
        if action_values.len() != candidates.len() {
            return Err(format!("{problem_id}/{policy_id} did not publish a full grammar trace"));
        }

        let mut replayed = Vec::with_capacity(candidates.len());
        let mut seen_candidate_ids = HashSet::with_capacity(candidates.len());
        for (step, value) in action_values.iter().enumerate() {
            let published = object(value, "stream action")?;
            let expected_candidate = ordered[step];
            let replay = replay_action(&task.train, test_input, expected_candidate, step);
            validate_published_action(published, &replay)?;
            if !seen_candidate_ids.insert(replay.candidate_id) {
                return Err(format!("repeated candidate in {problem_id}/{policy_id}"));
            }
            replayed.push(replay);
        }
        if seen_candidate_ids.len() != candidates.len() {
            return Err(format!("incomplete candidate coverage in {problem_id}/{policy_id}"));
        }

        let expected_trace = trace_commitment(&expected_execution, &replayed);
        require_eq_str(
            episode,
            "trace_commitment",
            &expected_trace,
            "stream episode",
        )?;

        let prefix_values = require_array(episode, "prefixes", "stream episode")?;
        if prefix_values.len() != PREFIX_BUDGETS.len() {
            return Err(format!("incorrect prefix count for {problem_id}/{policy_id}"));
        }
        for (index, budget) in PREFIX_BUDGETS.iter().enumerate() {
            let published = object(&prefix_values[index], "prefix receipt")?;
            validate_prefix(published, &expected_execution, &replayed, *budget)?;
        }

        let full_decision = decision_from_actions(&replayed);
        let exhaustive = exhaustive_reference(&task.train, test_input, &candidates);
        if full_decision != exhaustive {
            return Err(format!(
                "full streaming prefix does not match exhaustive decision for {problem_id}/{policy_id}"
            ));
        }

        verified.push(VerifiedEpisode {
            problem_id,
            policy_id: policy.id().into(),
            problem_visible_commitment: visible,
            execution_context_commitment: expected_execution,
            trace_commitment: expected_trace,
            prefixes_verified: PREFIX_BUDGETS.len(),
            full_prefix_matches_exhaustive: true,
        });
    }

    let expected_policies = BTreeSet::from([
        PolicyKind::Canonical.id().to_string(),
        PolicyKind::SemanticHashRandom.id().to_string(),
    ]);
    for problem_id in &expected_problem_ids {
        let actual = policies_by_problem
            .get(problem_id)
            .ok_or_else(|| format!("missing policy episodes for {problem_id}"))?;
        if actual != &expected_policies {
            return Err(format!("policy set mismatch for {problem_id}: {actual:?}"));
        }
    }

    verified.sort_by(|left, right| {
        left.problem_id
            .cmp(&right.problem_id)
            .then_with(|| left.policy_id.cmp(&right.policy_id))
    });
    let source_report_blake3 = blake3::hash(&source_bytes).to_hex().to_string();
    let mut output = StreamingVerificationReport {
        schema_version: SCHEMA_VERSION,
        domain: VERIFIER_DOMAIN.into(),
        subject_revision,
        source_report_blake3,
        action_space_commitment: action_space,
        grammar_size: candidates.len(),
        prefix_budgets: PREFIX_BUDGETS.to_vec(),
        expected_problem_count: expected_problem_ids.len(),
        episodes_verified: verified.len(),
        episodes: verified,
        commitment: String::new(),
    };
    output.commitment = verification_commitment(&output);
    write_json(&output_path, &output)?;

    println!("ARC streaming verification");
    println!("problems:  {}", output.expected_problem_count);
    println!("episodes:  {}", output.episodes_verified);
    println!("commitment: {}", output.commitment);
    Ok(())
}

fn validate_prefix(
    published: &Map<String, Value>,
    execution_context: &str,
    actions: &[ReplayAction],
    budget: usize,
) -> Result<(), String> {
    if require_usize(published, "budget", "prefix receipt")? != budget
        || require_usize(published, "candidates_evaluated", "prefix receipt")? != budget
    {
        return Err(format!("prefix budget/accounting mismatch at {budget}"));
    }
    let prefix = &actions[..budget];
    let training_pair_checks = prefix
        .iter()
        .fold(0usize, |acc, action| acc.saturating_add(action.pairs_checked));
    if require_usize(published, "training_pair_checks", "prefix receipt")?
        != training_pair_checks
    {
        return Err(format!("training-pair accounting mismatch at prefix {budget}"));
    }

    let mut predictions = Vec::<String>::new();
    for action in prefix {
        if let Some(digest) = &action.test_prediction_digest {
            if !predictions.contains(digest) {
                predictions.push(digest.clone());
            }
        }
    }
    let distinct_count = predictions.len();
    let decision = decision_from_prediction_digests(predictions);
    if require_usize(published, "distinct_prediction_count", "prefix receipt")? != distinct_count {
        return Err(format!("distinct-prediction count mismatch at prefix {budget}"));
    }
    let asserted = require_bool(published, "asserted", "prefix receipt")?;
    let prediction = optional_string(published, "prediction_digest", "prefix receipt")?;
    match &decision {
        LocalDecision::Asserted(digest) => {
            if !asserted || prediction.as_deref() != Some(digest.as_str()) {
                return Err(format!("asserted prefix decision mismatch at {budget}"));
            }
        }
        LocalDecision::Abstained(_) => {
            if asserted || prediction.is_some() {
                return Err(format!("abstained prefix decision mismatch at {budget}"));
            }
        }
    }
    let expected = prefix_commitment(execution_context, budget, prefix, &decision);
    require_eq_str(published, "prefix_commitment", &expected, "prefix receipt")?;
    Ok(())
}

fn replay_action(
    train: &[GridPair],
    test_input: &Grid,
    candidate: CandidateTransform,
    step: usize,
) -> ReplayAction {
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
    let test_prediction_digest = if training_consistent {
        Some(grid_digest(&apply_transform(test_input, candidate)))
    } else {
        None
    };
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
    ReplayAction {
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

fn validate_published_action(
    published: &Map<String, Value>,
    replay: &ReplayAction,
) -> Result<(), String> {
    if require_usize(published, "step", "stream action")? != replay.step
        || require_usize(published, "candidate_id", "stream action")? != replay.candidate_id
        || require_str(published, "candidate_name", "stream action")? != replay.candidate_name
        || require_usize(published, "pairs_checked", "stream action")? != replay.pairs_checked
        || optional_usize(published, "first_mismatch_index", "stream action")?
            != replay.first_mismatch_index
        || require_bool(published, "training_consistent", "stream action")?
            != replay.training_consistent
        || optional_string(published, "test_prediction_digest", "stream action")?
            != replay.test_prediction_digest
        || require_str(published, "action_commitment", "stream action")?
            != replay.action_commitment
    {
        return Err(format!("published stream action {} failed replay", replay.step));
    }
    Ok(())
}

fn decision_from_actions(actions: &[ReplayAction]) -> LocalDecision {
    let mut predictions = Vec::<String>::new();
    for action in actions {
        if let Some(digest) = &action.test_prediction_digest {
            if !predictions.contains(digest) {
                predictions.push(digest.clone());
            }
        }
    }
    decision_from_prediction_digests(predictions)
}

fn exhaustive_reference(
    train: &[GridPair],
    test_input: &Grid,
    candidates: &[CandidateTransform],
) -> LocalDecision {
    let mut predictions = Vec::<String>::new();
    for candidate in candidates {
        if train
            .iter()
            .all(|pair| apply_transform(&pair.input, *candidate) == pair.output)
        {
            let digest = grid_digest(&apply_transform(test_input, *candidate));
            if !predictions.contains(&digest) {
                predictions.push(digest);
            }
        }
    }
    decision_from_prediction_digests(predictions)
}

fn decision_from_prediction_digests(mut predictions: Vec<String>) -> LocalDecision {
    match predictions.len() {
        0 => LocalDecision::Abstained(AbstentionKind::NoTrainingConsistentCandidate),
        1 => LocalDecision::Asserted(predictions.remove(0)),
        _ => LocalDecision::Abstained(AbstentionKind::ConflictingPredictions),
    }
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
                return Err("canonical policy must not have a seed".into());
            }
            ordered.sort_by_key(|candidate| candidate.canonical_id);
        }
        PolicyKind::SemanticHashRandom => {
            let seed = seed_root.ok_or_else(|| "random-v2 policy requires a seed".to_string())?;
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

fn trace_commitment(execution_context: &str, actions: &[ReplayAction]) -> String {
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
    actions: &[ReplayAction],
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
    if candidates.len() != 4914 {
        return Err(format!("unexpected grammar size {}", candidates.len()));
    }
    let mut ids = BTreeSet::new();
    let mut names = BTreeSet::new();
    for candidate in candidates {
        if !ids.insert(candidate.canonical_id) {
            return Err(format!("duplicate candidate id {}", candidate.canonical_id));
        }
        let name = transform_name(*candidate);
        if !names.insert(name.clone()) {
            return Err(format!("duplicate candidate identity {name}"));
        }
    }
    for (expected, actual) in ids.into_iter().enumerate() {
        if expected != actual {
            return Err(format!("non-contiguous candidate id: expected {expected}, got {actual}"));
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

fn load_solver_tasks(root: &Path) -> Result<BTreeMap<String, SolverTask>, String> {
    let split = root.join("training");
    let mut paths = fs::read_dir(&split)
        .map_err(|error| format!("failed to list {}: {error}", split.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("failed to enumerate solver-view tasks: {error}"))?;
    paths.sort();
    if paths.is_empty() {
        return Err("solver-view training directory is empty".into());
    }
    let mut tasks = BTreeMap::new();
    for path in paths {
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            return Err(format!("non-JSON solver-view entry: {}", path.display()));
        }
        let bytes = fs::read(&path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let task = parse_solver_task(&bytes)?;
        let stem = path
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or_else(|| format!("solver-view filename is not UTF-8: {}", path.display()))?
            .to_string();
        if tasks.insert(stem.clone(), task).is_some() {
            return Err(format!("duplicate solver-view task stem {stem}"));
        }
    }
    Ok(tasks)
}

fn expected_problem_ids(tasks: &BTreeMap<String, SolverTask>) -> BTreeSet<String> {
    let mut result = BTreeSet::new();
    for (stem, task) in tasks {
        for index in 0..task.test_inputs.len() {
            result.insert(format!("{stem}#test-{index}"));
        }
    }
    result
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
        return Err("solver-view task requires training demonstrations".into());
    }
    let tests = require_array(object, "test", "solver-view task")?;
    if tests.is_empty() {
        return Err("solver-view task requires test inputs".into());
    }
    let sentinel = vec![vec![0]];
    let mut test_inputs = Vec::with_capacity(tests.len());
    for value in tests {
        let pair = object(value, "solver-view test pair")?;
        let input = parse_grid(
            pair.get("input")
                .ok_or_else(|| "solver-view test pair missing input".to_string())?,
        )?;
        let output = parse_grid(
            pair.get("output")
                .ok_or_else(|| "solver-view test pair missing sentinel".to_string())?,
        )?;
        if output != sentinel {
            return Err("solver-view test output is not the public sentinel".into());
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

fn parse_problem_id(value: &str) -> Result<(&str, usize), String> {
    let (stem, suffix) = value
        .rsplit_once("#test-")
        .ok_or_else(|| format!("invalid problem id {value}"))?;
    let index = suffix
        .parse::<usize>()
        .map_err(|error| format!("invalid test index in {value}: {error}"))?;
    if stem.is_empty() {
        return Err("problem id has empty task stem".into());
    }
    Ok((stem, index))
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

fn verification_commitment(value: &StreamingVerificationReport) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_str(&mut hasher, VERIFIER_DOMAIN);
    hash_u64(&mut hasher, u64::from(value.schema_version));
    hash_str(&mut hasher, &value.subject_revision);
    hash_str(&mut hasher, &value.source_report_blake3);
    hash_str(&mut hasher, &value.action_space_commitment);
    hash_u64(&mut hasher, value.grammar_size as u64);
    hash_u64(&mut hasher, value.prefix_budgets.len() as u64);
    for budget in &value.prefix_budgets {
        hash_u64(&mut hasher, *budget as u64);
    }
    hash_u64(&mut hasher, value.expected_problem_count as u64);
    hash_u64(&mut hasher, value.episodes_verified as u64);
    for episode in &value.episodes {
        hash_str(&mut hasher, &episode.problem_id);
        hash_str(&mut hasher, &episode.policy_id);
        hash_str(&mut hasher, &episode.problem_visible_commitment);
        hash_str(&mut hasher, &episode.execution_context_commitment);
        hash_str(&mut hasher, &episode.trace_commitment);
        hash_u64(&mut hasher, episode.prefixes_verified as u64);
        hasher.update(&[u8::from(episode.full_prefix_matches_exhaustive)]);
    }
    hasher.finalize().to_hex().to_string()
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

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    fs::write(
        path,
        serde_json::to_string_pretty(value)
            .map_err(|error| format!("failed to encode {}: {error}", path.display()))?,
    )
    .map_err(|error| format!("failed to write {}: {error}", path.display()))
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

fn require_str<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<&'a str, String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label} field {key} must be a string"))
}

fn require_eq_str(
    object: &Map<String, Value>,
    key: &str,
    expected: &str,
    label: &str,
) -> Result<(), String> {
    let actual = require_str(object, key, label)?;
    if actual != expected {
        return Err(format!("{label} field {key} is {actual:?}, expected {expected:?}"));
    }
    Ok(())
}

fn require_u64(object: &Map<String, Value>, key: &str, label: &str) -> Result<u64, String> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label} field {key} must be an unsigned integer"))
}

fn require_usize(
    object: &Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<usize, String> {
    let value = require_u64(object, key, label)?;
    usize::try_from(value).map_err(|_| format!("{label} field {key} does not fit usize"))
}

fn require_bool(object: &Map<String, Value>, key: &str, label: &str) -> Result<bool, String> {
    object
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{label} field {key} must be a boolean"))
}

fn optional_u64(object: &Map<String, Value>, key: &str, label: &str) -> Result<Option<u64>, String> {
    match object.get(key) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("{label} field {key} must be null or unsigned integer")),
    }
}

fn optional_usize(
    object: &Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<Option<usize>, String> {
    optional_u64(object, key, label)?
        .map(|value| usize::try_from(value).map_err(|_| format!("{label} field {key} overflows usize")))
        .transpose()
}

fn optional_string(
    object: &Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<Option<String>, String> {
    match object.get(key) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => value
            .as_str()
            .map(|value| Some(value.to_string()))
            .ok_or_else(|| format!("{label} field {key} must be null or string")),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_grammar_matches_frozen_size_and_anchors() {
        let candidates = canonical_candidates();
        validate_action_space(&candidates).unwrap();
        assert_eq!(transform_name(candidates[0]), "identity");
        assert_eq!(transform_name(candidates[91]), "reflect-x");
        assert_eq!(transform_name(candidates[546]), "translate(-3,-3)");
        assert_eq!(
            transform_name(*candidates.last().unwrap()),
            "translate(3,3)+color(9->8)"
        );
    }

    #[test]
    fn independent_random_order_ignores_candidate_vector_position() {
        let candidates = canonical_candidates();
        let mut reversed = candidates.clone();
        reversed.reverse();
        let a = ordered_candidates(
            &candidates,
            PolicyKind::SemanticHashRandom,
            "visible",
            Some(2797608998),
        )
        .unwrap();
        let b = ordered_candidates(
            &reversed,
            PolicyKind::SemanticHashRandom,
            "visible",
            Some(2797608998),
        )
        .unwrap();
        assert_eq!(
            a.into_iter().map(transform_name).collect::<Vec<_>>(),
            b.into_iter().map(transform_name).collect::<Vec<_>>()
        );
    }

    #[test]
    fn prefix_commitment_binds_budget_and_decision() {
        let actions = vec![ReplayAction {
            step: 0,
            candidate_id: 0,
            candidate_name: "identity".into(),
            pairs_checked: 1,
            first_mismatch_index: None,
            training_consistent: true,
            test_prediction_digest: Some("abc".into()),
            action_commitment: "commit".into(),
        }];
        let decision = LocalDecision::Asserted("abc".into());
        let a = prefix_commitment("ctx", 1, &actions, &decision);
        let b = prefix_commitment("ctx", 2, &actions, &decision);
        assert_ne!(a, b);
    }
}
