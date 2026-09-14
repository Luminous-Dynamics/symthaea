// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Independent evaluator for the process-isolated ARC native-action lane.
//!
//! The frozen policy binary runs only on target-stripped solver-view files. This evaluator runs
//! afterward and receives the true target bundle. It independently replays native actions against
//! solver-visible bytes, recomputes action commitments and policy seals, then scores prediction
//! digests against the real targets.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::Serialize;
use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use symthaea::hdc::grid_encoder::GridEncoder;

type Grid = Vec<Vec<u8>>;

const SCHEMA_VERSION: u32 = 1;
const EVALUATION_DOMAIN: &str = "symthaea/reasoning/arc-process-isolated-evaluation/v1";
const ACTION_SPACE_VERSION: &str = "rq003-candidate-transform-v1";
const ACTION_SPACE_DOMAIN: &[u8] = b"symthaea/reasoning/arc-action-space/v1";
const SOLVER_VISIBLE_DOMAIN: &[u8] = b"symthaea/reasoning/arc-policy-visible/v2";
const POLICY_SEAL_DOMAIN: &[u8] = b"symthaea/reasoning/arc-budgeted-policy-seal/v2";
const ACTION_DOMAIN: &[u8] = b"symthaea/reasoning/arc-native-action/v1";
const GRID_DOMAIN: &[u8] = b"symthaea/reasoning/arc-grid/v1";
const SOLVER_VIEW_DOMAIN: &str = "symthaea/reasoning/arc-solver-view/v1";
const TARGET_BUNDLE_DOMAIN: &str = "symthaea/reasoning/arc-evaluator-targets/v1";
const SENTINEL_DOMAIN: &str = "public-fixed-zero-grid-v1";

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
    geometry: Geometry,
    color: ColorOp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PolicyKind {
    CanonicalOrder,
    UniformRandom,
}

impl PolicyKind {
    const fn id(self) -> &'static str {
        match self {
            Self::CanonicalOrder => "canonical-order-v1",
            Self::UniformRandom => "uniform-random-without-replacement-v1",
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

#[derive(Debug, Clone)]
struct TargetFact {
    expected: Grid,
    target_digest: String,
}

#[derive(Debug, Default, Clone, Serialize)]
struct EvaluatedAggregate {
    episodes: usize,
    asserted: usize,
    exact_correct: usize,
    certified_assertions: usize,
    false_certainty: usize,
    missed_reference_assertions: usize,
    exact_accuracy: Option<f64>,
    coverage: Option<f64>,
    selective_accuracy: Option<f64>,
    certified_coverage: Option<f64>,
    false_certainty_rate: Option<f64>,
}

impl EvaluatedAggregate {
    fn observe(&mut self, row: &EvaluatedPolicyRow) {
        self.episodes = self.episodes.saturating_add(1);
        self.asserted = self.asserted.saturating_add(usize::from(row.asserted));
        self.exact_correct = self
            .exact_correct
            .saturating_add(usize::from(row.exact_correct));
        self.certified_assertions = self
            .certified_assertions
            .saturating_add(usize::from(row.exhaustive_certified_assertion));
        self.false_certainty = self
            .false_certainty
            .saturating_add(usize::from(row.false_certainty));
        self.missed_reference_assertions = self
            .missed_reference_assertions
            .saturating_add(usize::from(row.missed_reference_assertion));
    }

    fn finalize(&mut self) {
        self.exact_accuracy = ratio(self.exact_correct, self.episodes);
        self.coverage = ratio(self.asserted, self.episodes);
        self.selective_accuracy = ratio(self.exact_correct, self.asserted);
        self.certified_coverage = ratio(self.certified_assertions, self.episodes);
        self.false_certainty_rate = ratio(self.false_certainty, self.asserted);
    }
}

#[derive(Debug, Clone, Serialize)]
struct EvaluatedPolicyRow {
    problem_id: String,
    policy_id: String,
    policy_seed: Option<u64>,
    asserted: bool,
    local_prediction_digest: Option<String>,
    true_target_digest: String,
    exact_correct: bool,
    exhaustive_asserted: bool,
    exhaustive_prediction_digest: Option<String>,
    exhaustive_exact_correct: bool,
    exhaustive_certified_assertion: bool,
    false_certainty: bool,
    missed_reference_assertion: bool,
    action_trace_commitments_valid: bool,
    policy_seal_valid: bool,
    solver_visible_commitment_valid: bool,
    policy_order_valid: bool,
    sentinel_target_digest_seen: bool,
}

#[derive(Debug, Serialize)]
struct ProcessIsolatedEvaluation {
    schema_version: u32,
    domain: String,
    subject_revision: String,
    dataset_revision: String,
    dataset_tree: String,
    solver_view_commitment: String,
    target_bundle_commitment: String,
    policy_report_blake3: String,
    candidate_budget: usize,
    random_seed_root: u64,
    canonical: EvaluatedAggregate,
    uniform_random: EvaluatedAggregate,
    rows: Vec<EvaluatedPolicyRow>,
    commitment: String,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("ARC process-isolated evaluation failed: {err}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let subject_revision = required_env("SYMTHAEA_SUBJECT_REVISION")?;
    let solver_root = PathBuf::from(required_env("SYMTHAEA_ARC_SOLVER_VIEW_ROOT")?);
    let solver_manifest_path = PathBuf::from(required_env("SYMTHAEA_ARC_SOLVER_VIEW_MANIFEST_PATH")?);
    let target_bundle_path = PathBuf::from(required_env("SYMTHAEA_ARC_TARGET_BUNDLE_PATH")?);
    let policy_report_path = PathBuf::from(required_env("SYMTHAEA_ARC_POLICY_RESULTS_PATH")?);
    let output_path = PathBuf::from(required_env("SYMTHAEA_ARC_ISOLATED_EVALUATION_PATH")?);

    let solver_manifest_bytes = fs::read(&solver_manifest_path)
        .map_err(|err| format!("failed to read {}: {err}", solver_manifest_path.display()))?;
    let target_bundle_bytes = fs::read(&target_bundle_path)
        .map_err(|err| format!("failed to read {}: {err}", target_bundle_path.display()))?;
    let report_bytes = fs::read(&policy_report_path)
        .map_err(|err| format!("failed to read {}: {err}", policy_report_path.display()))?;
    let solver_manifest: Value = serde_json::from_slice(&solver_manifest_bytes)
        .map_err(|err| format!("invalid solver-view manifest JSON: {err}"))?;
    let target_bundle: Value = serde_json::from_slice(&target_bundle_bytes)
        .map_err(|err| format!("invalid target bundle JSON: {err}"))?;
    let report: Value = serde_json::from_slice(&report_bytes)
        .map_err(|err| format!("invalid policy report JSON: {err}"))?;

    let (solver_view_commitment, tasks) = validate_solver_view(&solver_root, &solver_manifest)?;
    let (target_bundle_commitment, dataset_revision, dataset_tree, targets) =
        validate_target_bundle(&target_bundle)?;
    let report_obj = object(&report, "policy report")?;
    require_eq_str(report_obj, "subject_revision", &subject_revision, "policy report")?;
    require_eq_str(report_obj, "dataset_version", &dataset_revision, "policy report")?;
    require_eq_str(report_obj, "split", "training", "policy report")?;
    require_eq_str(
        report_obj,
        "configuration_id",
        "arc-native-budgeted-policy-v2",
        "policy report",
    )?;
    require_eq_str(
        report_obj,
        "action_space_version",
        ACTION_SPACE_VERSION,
        "policy report",
    )?;

    let candidates = canonical_candidates();
    let grammar_commitment = action_space_commitment(&candidates);
    require_eq_str(
        report_obj,
        "action_space_commitment",
        &grammar_commitment,
        "policy report",
    )?;
    if require_usize(report_obj, "candidate_grammar_size", "policy report")? != candidates.len() {
        return Err("policy report grammar size mismatch".into());
    }
    let budget = require_usize(report_obj, "budget", "policy report")?;
    let random_seed_root = require_u64(report_obj, "random_seed_root", "policy report")?;

    let sentinel_digest = grid_digest(&vec![vec![0]])?;
    let rows_json = require_array(report_obj, "tasks", "policy report")?;
    let mut canonical = EvaluatedAggregate::default();
    let mut random = EvaluatedAggregate::default();
    let mut rows = Vec::with_capacity(rows_json.len());

    for row in rows_json {
        let row_obj = object(row, "policy task result")?;
        let problem_id = require_str(row_obj, "problem_id", "policy task result")?.to_string();
        let (file_stem, test_index) = parse_problem_id(&problem_id)?;
        let task = tasks
            .get(file_stem)
            .ok_or_else(|| format!("policy result references unknown solver-view task {file_stem}"))?;
        let test_input = task
            .test_inputs
            .get(test_index)
            .ok_or_else(|| format!("policy result test index is out of range: {problem_id}"))?;
        let target = targets
            .get(&problem_id)
            .ok_or_else(|| format!("policy result has no evaluator target: {problem_id}"))?;
        let recomputed_target = grid_digest(&target.expected)?;
        if recomputed_target != target.target_digest {
            return Err(format!("target bundle digest mismatch for {problem_id}"));
        }

        let policy_id = require_str(row_obj, "policy_id", "policy task result")?;
        let policy = match policy_id {
            "canonical-order-v1" => PolicyKind::CanonicalOrder,
            "uniform-random-without-replacement-v1" => PolicyKind::UniformRandom,
            other => return Err(format!("unsupported policy id {other}")),
        };
        let policy_seed = optional_u64(row_obj, "policy_seed", "policy task result")?;
        let visible = solver_visible_commitment(
            &problem_id,
            &task.train,
            test_input,
            &grammar_commitment,
            budget,
        );
        require_eq_str(
            row_obj,
            "solver_visible_commitment",
            &visible,
            "policy task result",
        )?;

        let expected_seed = match policy {
            PolicyKind::CanonicalOrder => None,
            PolicyKind::UniformRandom => Some(derive_policy_seed(random_seed_root, &visible)),
        };
        if policy_seed != expected_seed {
            return Err(format!("policy seed mismatch for {problem_id}/{policy_id}"));
        }

        let action_values = require_array(row_obj, "actions", "policy task result")?;
        if action_values.len() != budget {
            return Err(format!("action count mismatch for {problem_id}/{policy_id}"));
        }
        let expected_order = policy_order(candidates.len(), policy, policy_seed)?;
        let mut replayed = Vec::with_capacity(budget);
        let mut seen = HashSet::with_capacity(budget);
        let mut predictions = Vec::<String>::new();
        for (step, action_value) in action_values.iter().enumerate() {
            let action_obj = object(action_value, "native action")?;
            let candidate_id = require_usize(action_obj, "candidate_id", "native action")?;
            if candidate_id != expected_order[step] {
                return Err(format!("policy order mismatch at {problem_id}/{policy_id} step {step}"));
            }
            if !seen.insert(candidate_id) {
                return Err(format!("repeated candidate at {problem_id}/{policy_id}"));
            }
            let candidate = *candidates
                .get(candidate_id)
                .ok_or_else(|| format!("candidate id out of range: {candidate_id}"))?;
            let replay = replay_action(&task.train, test_input, candidate_id, candidate, step)?;
            validate_published_action(action_obj, &replay)?;
            if replay.training_consistent {
                let digest = replay
                    .test_prediction_digest
                    .as_ref()
                    .expect("consistent replay always has prediction digest")
                    .clone();
                if !predictions.contains(&digest) {
                    predictions.push(digest);
                }
            }
            replayed.push(replay);
        }

        let decision = decision_from_prediction_digests(predictions);
        let published_asserted = require_bool(row_obj, "asserted", "policy task result")?;
        let published_local = optional_string(row_obj, "local_prediction_digest", "policy task result")?;
        match &decision {
            LocalDecision::Asserted(digest) => {
                if !published_asserted || published_local.as_deref() != Some(digest.as_str()) {
                    return Err(format!("published local decision mismatch for {problem_id}/{policy_id}"));
                }
            }
            LocalDecision::Abstained(_) => {
                if published_asserted || published_local.is_some() {
                    return Err(format!("published abstention mismatch for {problem_id}/{policy_id}"));
                }
            }
        }

        let expected_seal = policy_seal(
            policy,
            policy_seed,
            budget,
            candidates.len(),
            &grammar_commitment,
            &visible,
            &replayed,
            &decision,
        );
        let published_seal = require_hex64(row_obj, "policy_sealed_commitment", "policy task result")?;
        if published_seal != expected_seal {
            return Err(format!("policy seal mismatch for {problem_id}/{policy_id}"));
        }

        let sentinel_seen = require_str(
            row_obj,
            "evaluator_target_digest",
            "policy task result",
        )? == sentinel_digest;
        if !sentinel_seen {
            return Err(format!(
                "policy process did not evaluate against the fixed public sentinel for {problem_id}/{policy_id}"
            ));
        }

        let local_digest = match &decision {
            LocalDecision::Asserted(digest) => Some(digest.clone()),
            LocalDecision::Abstained(_) => None,
        };
        let exact_correct = local_digest.as_deref() == Some(target.target_digest.as_str());
        let exhaustive_asserted = require_bool(row_obj, "exhaustive_asserted", "policy task result")?;
        let exhaustive_digest = optional_string(
            row_obj,
            "exhaustive_prediction_digest",
            "policy task result",
        )?;
        if exhaustive_asserted != exhaustive_digest.is_some() {
            return Err(format!("exhaustive assertion/digest mismatch for {problem_id}/{policy_id}"));
        }
        let exhaustive_exact_correct =
            exhaustive_digest.as_deref() == Some(target.target_digest.as_str());
        let certified = match (local_digest.as_deref(), exhaustive_digest.as_deref()) {
            (Some(local), Some(full)) => local == full,
            _ => false,
        };
        let false_certainty = local_digest.is_some() && !certified;
        let missed_reference_assertion = local_digest.is_none() && exhaustive_asserted;

        let evaluated = EvaluatedPolicyRow {
            problem_id,
            policy_id: policy.id().into(),
            policy_seed,
            asserted: local_digest.is_some(),
            local_prediction_digest: local_digest,
            true_target_digest: target.target_digest.clone(),
            exact_correct,
            exhaustive_asserted,
            exhaustive_prediction_digest: exhaustive_digest,
            exhaustive_exact_correct,
            exhaustive_certified_assertion: certified,
            false_certainty,
            missed_reference_assertion,
            action_trace_commitments_valid: true,
            policy_seal_valid: true,
            solver_visible_commitment_valid: true,
            policy_order_valid: true,
            sentinel_target_digest_seen: true,
        };
        match policy {
            PolicyKind::CanonicalOrder => canonical.observe(&evaluated),
            PolicyKind::UniformRandom => random.observe(&evaluated),
        }
        rows.push(evaluated);
    }

    if canonical.episodes == 0 || canonical.episodes != random.episodes {
        return Err("process-isolated evaluation requires matched canonical/random episodes".into());
    }
    canonical.finalize();
    random.finalize();

    let mut output = ProcessIsolatedEvaluation {
        schema_version: SCHEMA_VERSION,
        domain: EVALUATION_DOMAIN.into(),
        subject_revision,
        dataset_revision,
        dataset_tree,
        solver_view_commitment,
        target_bundle_commitment,
        policy_report_blake3: blake3::hash(&report_bytes).to_hex().to_string(),
        candidate_budget: budget,
        random_seed_root,
        canonical,
        uniform_random: random,
        rows,
        commitment: String::new(),
    };
    output.commitment = evaluation_commitment(&output);
    write_json(&output_path, &output)?;

    println!("ARC process-isolated evaluation");
    println!("solver view: {}", output.solver_view_commitment);
    println!("targets:     {}", output.target_bundle_commitment);
    println!("canonical:   {:?}", output.canonical.exact_accuracy);
    println!("random:      {:?}", output.uniform_random.exact_accuracy);
    println!("commitment:  {}", output.commitment);
    Ok(())
}

fn validate_solver_view(
    root: &Path,
    manifest: &Value,
) -> Result<(String, BTreeMap<String, SolverTask>), String> {
    let obj = object(manifest, "solver-view manifest")?;
    if require_u64(obj, "schema_version", "solver-view manifest")? != 1 {
        return Err("unexpected solver-view manifest schema".into());
    }
    require_eq_str(obj, "domain", SOLVER_VIEW_DOMAIN, "solver-view manifest")?;
    require_eq_str(obj, "sentinel", SENTINEL_DOMAIN, "solver-view manifest")?;
    require_eq_str(obj, "split", "training", "solver-view manifest")?;
    let files = require_array(obj, "files", "solver-view manifest")?;
    if require_usize(obj, "task_count", "solver-view manifest")? != files.len() {
        return Err("solver-view task_count mismatch".into());
    }

    let mut commitment_files = Vec::with_capacity(files.len());
    let mut tasks = BTreeMap::new();
    for file in files {
        let file_obj = object(file, "solver-view file")?;
        let relative = require_str(file_obj, "relative_path", "solver-view file")?;
        if !relative.starts_with("training/") || !relative.ends_with(".json") {
            return Err(format!("invalid solver-view path {relative}"));
        }
        let path = root.join(relative);
        let bytes = fs::read(&path)
            .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        let byte_length = require_u64(file_obj, "byte_length", "solver-view file")?;
        if bytes.len() as u64 != byte_length {
            return Err(format!("solver-view byte length mismatch for {relative}"));
        }
        let digest = blake3::hash(&bytes).to_hex().to_string();
        require_eq_str(file_obj, "blake3", &digest, "solver-view file")?;
        commitment_files.push((relative.to_string(), byte_length, digest));

        let stem = path
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or_else(|| format!("solver-view filename not UTF-8: {relative}"))?
            .to_string();
        if tasks.insert(stem.clone(), parse_solver_task(&bytes)?).is_some() {
            return Err(format!("duplicate solver-view task stem {stem}"));
        }
    }

    let commitment = solver_view_commitment(&commitment_files);
    require_eq_str(obj, "commitment", &commitment, "solver-view manifest")?;
    Ok((commitment, tasks))
}

fn validate_target_bundle(
    value: &Value,
) -> Result<(String, String, String, BTreeMap<String, TargetFact>), String> {
    let obj = object(value, "target bundle")?;
    if require_u64(obj, "schema_version", "target bundle")? != 1 {
        return Err("unexpected target bundle schema".into());
    }
    require_eq_str(obj, "domain", TARGET_BUNDLE_DOMAIN, "target bundle")?;
    require_eq_str(obj, "split", "training", "target bundle")?;
    let repository = require_str(obj, "dataset_repository", "target bundle")?;
    let revision = require_git_hex(obj, "dataset_revision", "target bundle")?;
    let tree = require_git_hex(obj, "dataset_tree", "target bundle")?;
    let manifest_blake3 = require_hex64(obj, "dataset_manifest_blake3", "target bundle")?;
    let manifest_sha256 = require_hex64(obj, "dataset_manifest_sha256", "target bundle")?;
    let selected_paths = require_array(obj, "selected_task_paths", "target bundle")?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| "target selected_task_paths entry must be string".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let target_values = require_array(obj, "targets", "target bundle")?;
    let mut targets = BTreeMap::new();
    let mut commitment_targets = Vec::with_capacity(target_values.len());
    for value in target_values {
        let target = object(value, "target record")?;
        let problem_id = require_str(target, "problem_id", "target record")?.to_string();
        let expected = parse_grid(
            target
                .get("expected")
                .ok_or_else(|| "target record missing expected grid".to_string())?,
        )?;
        let digest = grid_digest(&expected)?;
        require_eq_str(target, "target_digest", &digest, "target record")?;
        if targets
            .insert(
                problem_id.clone(),
                TargetFact {
                    expected,
                    target_digest: digest.clone(),
                },
            )
            .is_some()
        {
            return Err(format!("duplicate target problem id {problem_id}"));
        }
        commitment_targets.push((problem_id, digest));
    }
    let commitment = target_bundle_commitment(
        repository,
        &revision,
        &tree,
        &manifest_blake3,
        &manifest_sha256,
        &selected_paths,
        &commitment_targets,
    );
    require_eq_str(obj, "commitment", &commitment, "target bundle")?;
    Ok((commitment, revision, tree, targets))
}

fn parse_solver_task(bytes: &[u8]) -> Result<SolverTask, String> {
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|err| format!("invalid solver-view task JSON: {err}"))?;
    let obj = object(&value, "solver-view task")?;
    let train = require_array(obj, "train", "solver-view task")?
        .iter()
        .map(parse_pair)
        .collect::<Result<Vec<_>, _>>()?;
    let tests = require_array(obj, "test", "solver-view task")?;
    let sentinel = vec![vec![0]];
    let mut test_inputs = Vec::with_capacity(tests.len());
    for (index, value) in tests.iter().enumerate() {
        let pair = object(value, "solver test pair")?;
        let input = parse_grid(
            pair.get("input")
                .ok_or_else(|| format!("test[{index}] missing input"))?,
        )?;
        let output = parse_grid(
            pair.get("output")
                .ok_or_else(|| format!("test[{index}] missing sentinel output"))?,
        )?;
        if output != sentinel {
            return Err(format!("test[{index}] does not contain fixed public sentinel"));
        }
        test_inputs.push(input);
    }
    Ok(SolverTask { train, test_inputs })
}

fn replay_action(
    train: &[GridPair],
    test_input: &Grid,
    candidate_id: usize,
    candidate: CandidateTransform,
    step: usize,
) -> Result<ReplayAction, String> {
    let mut pairs_checked = 0usize;
    let mut mismatch = None;
    for (index, pair) in train.iter().enumerate() {
        pairs_checked = pairs_checked.saturating_add(1);
        if apply_transform(&pair.input, candidate) != pair.output {
            mismatch = Some(index);
            break;
        }
    }
    let training_consistent = mismatch.is_none();
    let test_prediction_digest = if training_consistent {
        Some(grid_digest(&apply_transform(test_input, candidate))?)
    } else {
        None
    };
    let candidate_name = transform_name(candidate);
    let action_commitment = native_action_commitment(
        step,
        candidate_id,
        &candidate_name,
        pairs_checked,
        mismatch,
        training_consistent,
        test_prediction_digest.as_deref(),
    );
    Ok(ReplayAction {
        step,
        candidate_id,
        candidate_name,
        pairs_checked,
        first_mismatch_index: mismatch,
        training_consistent,
        test_prediction_digest,
        action_commitment,
    })
}

fn validate_published_action(published: &Map<String, Value>, replay: &ReplayAction) -> Result<(), String> {
    if require_usize(published, "step", "native action")? != replay.step
        || require_usize(published, "candidate_id", "native action")? != replay.candidate_id
        || require_str(published, "candidate_name", "native action")? != replay.candidate_name
        || require_usize(published, "pairs_checked", "native action")? != replay.pairs_checked
        || optional_usize(published, "first_mismatch_index", "native action")?
            != replay.first_mismatch_index
        || require_bool(published, "training_consistent", "native action")?
            != replay.training_consistent
        || optional_string(published, "test_prediction_digest", "native action")?
            != replay.test_prediction_digest
        || require_str(published, "action_commitment", "native action")?
            != replay.action_commitment
    {
        return Err(format!("published native action {} does not match replay", replay.step));
    }
    Ok(())
}

fn decision_from_prediction_digests(mut predictions: Vec<String>) -> LocalDecision {
    match predictions.len() {
        0 => LocalDecision::Abstained(AbstentionKind::NoTrainingConsistentCandidate),
        1 => LocalDecision::Asserted(predictions.remove(0)),
        _ => LocalDecision::Abstained(AbstentionKind::ConflictingPredictions),
    }
}

#[allow(clippy::too_many_arguments)]
fn policy_seal(
    policy: PolicyKind,
    seed: Option<u64>,
    budget: usize,
    grammar_size: usize,
    grammar_commitment: &str,
    visible_commitment: &str,
    actions: &[ReplayAction],
    decision: &LocalDecision,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, POLICY_SEAL_DOMAIN);
    hash_str(&mut hasher, policy.id());
    match seed {
        Some(seed) => {
            hasher.update(&[1]);
            hash_u64(&mut hasher, seed);
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hash_u64(&mut hasher, budget as u64);
    hash_u64(&mut hasher, grammar_size as u64);
    hash_str(&mut hasher, grammar_commitment);
    hash_str(&mut hasher, visible_commitment);
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

#[allow(clippy::too_many_arguments)]
fn native_action_commitment(
    step: usize,
    candidate_id: usize,
    candidate_name: &str,
    pairs_checked: usize,
    mismatch: Option<usize>,
    training_consistent: bool,
    prediction_digest: Option<&str>,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, ACTION_DOMAIN);
    hash_u64(&mut hasher, step as u64);
    hash_u64(&mut hasher, candidate_id as u64);
    hash_str(&mut hasher, candidate_name);
    hash_u64(&mut hasher, pairs_checked as u64);
    match mismatch {
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

fn action_space_commitment(candidates: &[CandidateTransform]) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, ACTION_SPACE_DOMAIN);
    hash_str(&mut hasher, ACTION_SPACE_VERSION);
    hash_u64(&mut hasher, candidates.len() as u64);
    for (candidate_id, candidate) in candidates.iter().enumerate() {
        hash_u64(&mut hasher, candidate_id as u64);
        hash_str(&mut hasher, &transform_name(*candidate));
    }
    hasher.finalize().to_hex().to_string()
}

fn solver_visible_commitment(
    problem_id: &str,
    train: &[GridPair],
    test_input: &Grid,
    grammar_commitment: &str,
    budget: usize,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, SOLVER_VISIBLE_DOMAIN);
    hash_str(&mut hasher, problem_id);
    hash_str(&mut hasher, grammar_commitment);
    hash_u64(&mut hasher, budget as u64);
    hash_u64(&mut hasher, train.len() as u64);
    for pair in train {
        hash_grid(&mut hasher, &pair.input);
        hash_grid(&mut hasher, &pair.output);
    }
    hash_grid(&mut hasher, test_input);
    hasher.finalize().to_hex().to_string()
}

fn derive_policy_seed(root: u64, visible_commitment: &str) -> u64 {
    let digest = blake3::hash(visible_commitment.as_bytes());
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest.as_bytes()[..8]);
    root ^ u64::from_le_bytes(bytes)
}

fn policy_order(
    grammar_size: usize,
    policy: PolicyKind,
    seed: Option<u64>,
) -> Result<Vec<usize>, String> {
    let mut order = (0..grammar_size).collect::<Vec<_>>();
    match policy {
        PolicyKind::CanonicalOrder => {
            if seed.is_some() {
                return Err("canonical-order policy must not receive seed".into());
            }
        }
        PolicyKind::UniformRandom => {
            let seed = seed.ok_or_else(|| "random policy requires seed".to_string())?;
            let mut rng = StdRng::seed_from_u64(seed);
            order.shuffle(&mut rng);
        }
    }
    Ok(order)
}

fn parse_pair(value: &Value) -> Result<GridPair, String> {
    let obj = object(value, "train pair")?;
    Ok(GridPair {
        input: parse_grid(obj.get("input").ok_or_else(|| "train pair missing input".to_string())?)?,
        output: parse_grid(obj.get("output").ok_or_else(|| "train pair missing output".to_string())?)?,
    })
}

fn parse_grid(value: &Value) -> Result<Grid, String> {
    let rows = value
        .as_array()
        .ok_or_else(|| "grid must be array".to_string())?;
    if rows.is_empty() {
        return Err("grid must not be empty".into());
    }
    let mut grid = Vec::with_capacity(rows.len());
    let mut width = None;
    for row in rows {
        let cells = row
            .as_array()
            .ok_or_else(|| "grid row must be array".to_string())?;
        if cells.is_empty() {
            return Err("grid row must not be empty".into());
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
                .ok_or_else(|| "grid cell must be unsigned integer".to_string())?;
            if value > 9 {
                return Err("grid color must be within 0..=9".into());
            }
            parsed.push(value as u8);
        }
        grid.push(parsed);
    }
    Ok(grid)
}

fn grid_digest(grid: &Grid) -> Result<String, String> {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, GRID_DOMAIN);
    hash_grid(&mut hasher, grid);
    Ok(hasher.finalize().to_hex().to_string())
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

fn solver_view_commitment(files: &[(String, u64, String)]) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_str(&mut hasher, SOLVER_VIEW_DOMAIN);
    hash_str(&mut hasher, SENTINEL_DOMAIN);
    hash_u64(&mut hasher, files.len() as u64);
    for (path, len, digest) in files {
        hash_str(&mut hasher, path);
        hash_u64(&mut hasher, *len);
        hash_str(&mut hasher, digest);
    }
    hasher.finalize().to_hex().to_string()
}

#[allow(clippy::too_many_arguments)]
fn target_bundle_commitment(
    repository: &str,
    revision: &str,
    tree: &str,
    manifest_blake3: &str,
    manifest_sha256: &str,
    selected_paths: &[String],
    targets: &[(String, String)],
) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_str(&mut hasher, TARGET_BUNDLE_DOMAIN);
    hash_str(&mut hasher, repository);
    hash_str(&mut hasher, revision);
    hash_str(&mut hasher, tree);
    hash_str(&mut hasher, manifest_blake3);
    hash_str(&mut hasher, manifest_sha256);
    hash_u64(&mut hasher, selected_paths.len() as u64);
    for path in selected_paths {
        hash_str(&mut hasher, path);
    }
    hash_u64(&mut hasher, targets.len() as u64);
    for (problem_id, digest) in targets {
        hash_str(&mut hasher, problem_id);
        hash_str(&mut hasher, digest);
    }
    hasher.finalize().to_hex().to_string()
}

fn evaluation_commitment(value: &ProcessIsolatedEvaluation) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_str(&mut hasher, EVALUATION_DOMAIN);
    hash_u64(&mut hasher, u64::from(value.schema_version));
    hash_str(&mut hasher, &value.subject_revision);
    hash_str(&mut hasher, &value.dataset_revision);
    hash_str(&mut hasher, &value.dataset_tree);
    hash_str(&mut hasher, &value.solver_view_commitment);
    hash_str(&mut hasher, &value.target_bundle_commitment);
    hash_str(&mut hasher, &value.policy_report_blake3);
    hash_u64(&mut hasher, value.candidate_budget as u64);
    hash_u64(&mut hasher, value.random_seed_root);
    hash_u64(&mut hasher, value.rows.len() as u64);
    for row in &value.rows {
        hash_str(&mut hasher, &row.problem_id);
        hash_str(&mut hasher, &row.policy_id);
        match row.policy_seed {
            Some(seed) => {
                hasher.update(&[1]);
                hash_u64(&mut hasher, seed);
            }
            None => {
                hasher.update(&[0]);
            }
        }
        hasher.update(&[u8::from(row.asserted)]);
        match &row.local_prediction_digest {
            Some(digest) => {
                hasher.update(&[1]);
                hash_str(&mut hasher, digest);
            }
            None => {
                hasher.update(&[0]);
            }
        }
        hash_str(&mut hasher, &row.true_target_digest);
        hasher.update(&[
            u8::from(row.exact_correct),
            u8::from(row.exhaustive_asserted),
            u8::from(row.exhaustive_exact_correct),
            u8::from(row.exhaustive_certified_assertion),
            u8::from(row.false_certainty),
            u8::from(row.missed_reference_assertion),
            u8::from(row.action_trace_commitments_valid),
            u8::from(row.policy_seal_valid),
            u8::from(row.solver_visible_commitment_valid),
            u8::from(row.policy_order_valid),
            u8::from(row.sentinel_target_digest_seen),
        ]);
    }
    hasher.finalize().to_hex().to_string()
}

fn parse_problem_id(value: &str) -> Result<(&str, usize), String> {
    let (stem, index) = value
        .rsplit_once("#test-")
        .ok_or_else(|| format!("invalid ARC problem id {value}"))?;
    let index = index
        .parse::<usize>()
        .map_err(|err| format!("invalid ARC test index in {value}: {err}"))?;
    Ok((stem, index))
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be object"))
}

fn require_array<'a>(object: &'a Map<String, Value>, key: &str, label: &str) -> Result<&'a Vec<Value>, String> {
    object
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label} field {key} must be array"))
}

fn require_str<'a>(object: &'a Map<String, Value>, key: &str, label: &str) -> Result<&'a str, String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label} field {key} must be string"))
}

fn require_eq_str(object: &Map<String, Value>, key: &str, expected: &str, label: &str) -> Result<(), String> {
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
        .ok_or_else(|| format!("{label} field {key} must be unsigned integer"))
}

fn require_usize(object: &Map<String, Value>, key: &str, label: &str) -> Result<usize, String> {
    usize::try_from(require_u64(object, key, label)?)
        .map_err(|_| format!("{label} field {key} does not fit usize"))
}

fn require_bool(object: &Map<String, Value>, key: &str, label: &str) -> Result<bool, String> {
    object
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{label} field {key} must be bool"))
}

fn optional_string(object: &Map<String, Value>, key: &str, label: &str) -> Result<Option<String>, String> {
    match object.get(key) {
        Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.to_ascii_lowercase())),
        Some(_) => Err(format!("{label} field {key} must be string or null")),
        None => Err(format!("{label} missing field {key}")),
    }
}

fn optional_u64(object: &Map<String, Value>, key: &str, label: &str) -> Result<Option<u64>, String> {
    match object.get(key) {
        Some(Value::Null) => Ok(None),
        Some(Value::Number(value)) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("{label} field {key} must be unsigned integer or null")),
        Some(_) => Err(format!("{label} field {key} must be unsigned integer or null")),
        None => Err(format!("{label} missing field {key}")),
    }
}

fn optional_usize(object: &Map<String, Value>, key: &str, label: &str) -> Result<Option<usize>, String> {
    optional_u64(object, key, label)?
        .map(usize::try_from)
        .transpose()
        .map_err(|_| format!("{label} field {key} does not fit usize"))
}

fn require_hex64(object: &Map<String, Value>, key: &str, label: &str) -> Result<String, String> {
    let value = require_str(object, key, label)?;
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{label} field {key} must be 64-hex digest"));
    }
    Ok(value.to_ascii_lowercase())
}

fn require_git_hex(object: &Map<String, Value>, key: &str, label: &str) -> Result<String, String> {
    let value = require_str(object, key, label)?;
    if value.len() != 40 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{label} field {key} must be 40-hex Git id"));
    }
    Ok(value.to_ascii_lowercase())
}

fn required_env(name: &str) -> Result<String, String> {
    match env::var(name) {
        Ok(value) if !value.trim().is_empty() => Ok(value.trim().to_string()),
        Ok(_) => Err(format!("required environment variable {name} is empty")),
        Err(err) => Err(format!("required environment variable {name} is missing: {err}")),
    }
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    fs::write(
        path,
        serde_json::to_string_pretty(value)
            .map_err(|err| format!("failed to encode {}: {err}", path.display()))?,
    )
    .map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator != 0).then_some(numerator as f64 / denominator as f64)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_seed_depends_only_on_visible_commitment() {
        let visible = "abcd";
        assert_eq!(derive_policy_seed(42, visible), derive_policy_seed(42, visible));
    }

    #[test]
    fn canonical_grammar_anchors_match_frozen_policy() {
        let candidates = canonical_candidates();
        assert_eq!(candidates.len(), 4_914);
        assert_eq!(transform_name(candidates[0]), "identity");
        assert_eq!(transform_name(candidates[91]), "reflect-x");
        assert_eq!(transform_name(candidates[546]), "translate(-3,-3)");
        assert_eq!(
            transform_name(*candidates.last().unwrap()),
            "translate(3,3)+color(9->8)"
        );
    }
}
