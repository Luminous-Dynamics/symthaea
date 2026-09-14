// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! RQ-006Z-C development lane: resource-bounded policy control over the native RQ-003 ARC grammar.
//!
//! The policy may choose only which existing `CandidateTransform` to evaluate next. Candidate
//! evaluation uses training demonstrations plus the held-out test INPUT. The hidden expected test
//! grid is not accepted by `run_policy` and is used only after the policy episode has been sealed.
//!
//! This lane does not replace the exhaustive RQ-003 exact-output qualifier. It measures search
//! control under a fixed candidate-evaluation budget and separately reports whether any policy-local
//! assertion is certified by the full grammar after the policy episode is sealed.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::Serialize;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use symthaea::hdc::grid_encoder::GridEncoder;

type Grid = Vec<Vec<u8>>;

const SCHEMA_VERSION: u32 = 1;
const CONFIGURATION_ID: &str = "arc-native-budgeted-policy-v1";
const ACTION_SPACE_VERSION: &str = "rq003-candidate-transform-v1";
const POLICY_SEAL_DOMAIN: &[u8] = b"symthaea/reasoning/arc-budgeted-policy-seal/v1";
const ACTION_DOMAIN: &[u8] = b"symthaea/reasoning/arc-native-action/v1";
const DEFAULT_BUDGET: usize = 128;
const DEFAULT_RANDOM_SEED: u64 = 0xA6C0_2026;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum Geometry {
    Identity,
    ReflectX,
    ReflectY,
    Rotate90,
    Rotate180,
    Rotate270,
    Translate { dx: i32, dy: i32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum ColorOp {
    None,
    Replace { from: u8, to: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct CandidateTransform {
    geometry: Geometry,
    color: ColorOp,
}

#[derive(Debug, Clone, Serialize)]
struct ArcActionManifestEntry {
    candidate_id: usize,
    transform: CandidateTransform,
    name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
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

#[derive(Debug, Clone, Serialize)]
struct NativeActionRecord {
    step: usize,
    candidate_id: usize,
    candidate_name: String,
    pairs_checked: usize,
    first_mismatch_index: Option<usize>,
    training_consistent: bool,
    test_prediction_digest: Option<String>,
    action_commitment: String,
}

#[derive(Debug, Clone, PartialEq)]
enum LocalDecision {
    Asserted(Grid),
    Abstained(AbstentionKind),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum AbstentionKind {
    NoTrainingConsistentCandidate,
    ConflictingPredictions,
}

#[derive(Debug, Clone)]
struct PolicyRun {
    policy: PolicyKind,
    policy_seed: Option<u64>,
    budget: usize,
    action_space_commitment: String,
    solver_visible_commitment: String,
    actions: Vec<NativeActionRecord>,
    decision: LocalDecision,
    candidates_evaluated: usize,
    training_pair_checks: usize,
    sealed_commitment: String,
}

#[derive(Debug, Clone)]
struct ExhaustiveReference {
    decision: LocalDecision,
    candidates_evaluated: usize,
    training_pair_checks: usize,
}

#[derive(Debug, Clone, Serialize)]
struct PolicyTaskResult {
    problem_id: String,
    policy_id: String,
    policy_seed: Option<u64>,
    budget: usize,
    grammar_size: usize,
    action_space_commitment: String,
    solver_visible_commitment: String,
    policy_sealed_commitment: String,
    action_sequence: Vec<usize>,
    candidates_evaluated: usize,
    training_pair_checks: usize,
    asserted: bool,
    local_prediction_digest: Option<String>,
    exact_correct: bool,
    exhaustive_asserted: bool,
    exhaustive_prediction_digest: Option<String>,
    exhaustive_exact_correct: bool,
    exhaustive_certified_assertion: bool,
    false_certainty: bool,
    exhaustive_candidates_evaluated: usize,
    exhaustive_training_pair_checks: usize,
    wall_time_us: u64,
}

#[derive(Debug, Default, Clone, Serialize)]
struct PolicyAggregate {
    episodes: usize,
    asserted: usize,
    exact_correct: usize,
    exhaustive_certified_assertions: usize,
    false_certainty: usize,
    candidates_evaluated: u64,
    training_pair_checks: u64,
    exact_accuracy: Option<f64>,
    coverage: Option<f64>,
    selective_accuracy: Option<f64>,
    certified_coverage: Option<f64>,
}

impl PolicyAggregate {
    fn observe(&mut self, result: &PolicyTaskResult) {
        self.episodes = self.episodes.saturating_add(1);
        self.asserted = self.asserted.saturating_add(usize::from(result.asserted));
        self.exact_correct = self
            .exact_correct
            .saturating_add(usize::from(result.exact_correct));
        self.exhaustive_certified_assertions = self
            .exhaustive_certified_assertions
            .saturating_add(usize::from(result.exhaustive_certified_assertion));
        self.false_certainty = self
            .false_certainty
            .saturating_add(usize::from(result.false_certainty));
        self.candidates_evaluated = self
            .candidates_evaluated
            .saturating_add(result.candidates_evaluated as u64);
        self.training_pair_checks = self
            .training_pair_checks
            .saturating_add(result.training_pair_checks as u64);
    }

    fn finalize(&mut self) {
        self.exact_accuracy = ratio(self.exact_correct, self.episodes);
        self.coverage = ratio(self.asserted, self.episodes);
        self.selective_accuracy = ratio(self.exact_correct, self.asserted);
        self.certified_coverage = ratio(self.exhaustive_certified_assertions, self.episodes);
    }
}

#[derive(Debug, Serialize)]
struct ArcBudgetedPolicyReport {
    schema_version: u32,
    subject_revision: String,
    dataset_version: String,
    split: String,
    configuration_id: String,
    action_space_version: String,
    action_space_commitment: String,
    candidate_grammar_size: usize,
    budget: usize,
    random_seed_root: u64,
    task_limit: Option<usize>,
    task_files_seen: usize,
    test_cases_evaluated: usize,
    canonical: PolicyAggregate,
    uniform_random: PolicyAggregate,
    paired_exact_delta_random_minus_canonical: i64,
    tasks: Vec<PolicyTaskResult>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("ARC budgeted policy qualification failed: {err}");
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

    let results_path = env::var("SYMTHAEA_ARC_POLICY_RESULTS_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data/benchmarks/arc/budgeted-policy-results.json"));
    let max_tasks = env::var("SYMTHAEA_ARC_MAX_TASKS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    let requested_budget = env::var("SYMTHAEA_ARC_CANDIDATE_BUDGET")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_BUDGET);
    if requested_budget == 0 {
        return Err("SYMTHAEA_ARC_CANDIDATE_BUDGET must be greater than zero".into());
    }
    let random_seed_root = env::var("SYMTHAEA_ARC_POLICY_SEED")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(DEFAULT_RANDOM_SEED);

    let candidates = canonical_candidates();
    let budget = requested_budget.min(candidates.len());
    let action_manifest = action_space_manifest(&candidates);
    let grammar_commitment = action_space_commitment(&action_manifest)?;

    let task_files = selected_task_files(&split_dir, max_tasks)?;
    println!("ARC native-action budgeted policy lane");
    println!("subject:        {subject_revision}");
    println!("dataset:        {dataset_version}");
    println!("split:          {split}");
    println!("grammar size:   {}", candidates.len());
    println!("budget:         {budget}");
    println!("random seed:    {random_seed_root}");
    println!("task files:     {}", task_files.len());

    let mut canonical_aggregate = PolicyAggregate::default();
    let mut random_aggregate = PolicyAggregate::default();
    let mut task_results = Vec::new();
    let mut paired_exact_delta = 0i64;
    let mut test_cases_evaluated = 0usize;

    for path in &task_files {
        let raw = fs::read(path)
            .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        let task = parse_task(&raw)
            .map_err(|err| format!("failed to parse {}: {err}", path.display()))?;
        let file_stem = path
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or_else(|| format!("task filename is not valid UTF-8: {}", path.display()))?;
        let raw_hash = blake3::hash(&raw).to_hex().to_string();

        for (test_index, test_case) in task.test.iter().enumerate() {
            let problem_id = format!("{file_stem}#test-{test_index}");
            let visible = solver_visible_commitment(
                &raw_hash,
                test_index,
                &task.train,
                &test_case.input,
                &grammar_commitment,
                budget,
            )?;
            let derived_seed = derive_policy_seed(random_seed_root, &visible);

            let canonical = evaluate_policy_episode(
                &problem_id,
                &task.train,
                &test_case.input,
                &test_case.expected,
                &candidates,
                &grammar_commitment,
                &visible,
                budget,
                PolicyKind::CanonicalOrder,
                None,
            )?;
            let random = evaluate_policy_episode(
                &problem_id,
                &task.train,
                &test_case.input,
                &test_case.expected,
                &candidates,
                &grammar_commitment,
                &visible,
                budget,
                PolicyKind::UniformRandom,
                Some(derived_seed),
            )?;

            paired_exact_delta += i64::from(random.exact_correct) - i64::from(canonical.exact_correct);
            canonical_aggregate.observe(&canonical);
            random_aggregate.observe(&random);
            task_results.push(canonical);
            task_results.push(random);
            test_cases_evaluated = test_cases_evaluated.saturating_add(1);
        }
    }

    canonical_aggregate.finalize();
    random_aggregate.finalize();

    let report = ArcBudgetedPolicyReport {
        schema_version: SCHEMA_VERSION,
        subject_revision,
        dataset_version,
        split,
        configuration_id: CONFIGURATION_ID.into(),
        action_space_version: ACTION_SPACE_VERSION.into(),
        action_space_commitment: grammar_commitment,
        candidate_grammar_size: candidates.len(),
        budget,
        random_seed_root,
        task_limit: max_tasks,
        task_files_seen: task_files.len(),
        test_cases_evaluated,
        canonical: canonical_aggregate,
        uniform_random: random_aggregate,
        paired_exact_delta_random_minus_canonical: paired_exact_delta,
        tasks: task_results,
    };

    if let Some(parent) = results_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    fs::write(
        &results_path,
        serde_json::to_string_pretty(&report)
            .map_err(|err| format!("failed to encode policy report: {err}"))?,
    )
    .map_err(|err| format!("failed to write {}: {err}", results_path.display()))?;

    println!("canonical exact: {:?}", report.canonical.exact_accuracy);
    println!("random exact:    {:?}", report.uniform_random.exact_accuracy);
    println!("canonical coverage: {:?}", report.canonical.coverage);
    println!("random coverage:    {:?}", report.uniform_random.coverage);
    println!("canonical certified coverage: {:?}", report.canonical.certified_coverage);
    println!("random certified coverage:    {:?}", report.uniform_random.certified_coverage);
    println!("report: {}", results_path.display());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn evaluate_policy_episode(
    problem_id: &str,
    train: &[GridPair],
    test_input: &Grid,
    expected: &Grid,
    candidates: &[CandidateTransform],
    grammar_commitment: &str,
    visible_commitment: &str,
    budget: usize,
    policy: PolicyKind,
    policy_seed: Option<u64>,
) -> Result<PolicyTaskResult, String> {
    let started = Instant::now();

    // Critical leakage boundary: `run_policy` receives no expected target.
    let run = run_policy(
        train,
        test_input,
        candidates,
        grammar_commitment,
        visible_commitment,
        budget,
        policy,
        policy_seed,
    )?;

    // The policy episode is sealed before either the hidden target or full-grammar reference is
    // consulted. Everything below this line is evaluator-side information.
    let exhaustive = exhaustive_reference(train, test_input, candidates);
    let local_prediction = asserted_grid(&run.decision);
    let exhaustive_prediction = asserted_grid(&exhaustive.decision);
    let asserted = local_prediction.is_some();
    let exact_correct = local_prediction.is_some_and(|prediction| prediction == expected);
    let exhaustive_asserted = exhaustive_prediction.is_some();
    let exhaustive_exact_correct =
        exhaustive_prediction.is_some_and(|prediction| prediction == expected);
    let certified = match (local_prediction, exhaustive_prediction) {
        (Some(local), Some(full)) => local == full,
        _ => false,
    };
    let false_certainty = asserted && !certified;
    let wall_time_us = started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;

    Ok(PolicyTaskResult {
        problem_id: problem_id.into(),
        policy_id: run.policy.id().into(),
        policy_seed: run.policy_seed,
        budget: run.budget,
        grammar_size: candidates.len(),
        action_space_commitment: run.action_space_commitment,
        solver_visible_commitment: run.solver_visible_commitment,
        policy_sealed_commitment: run.sealed_commitment,
        action_sequence: run.actions.iter().map(|action| action.candidate_id).collect(),
        candidates_evaluated: run.candidates_evaluated,
        training_pair_checks: run.training_pair_checks,
        asserted,
        local_prediction_digest: local_prediction.map(grid_digest),
        exact_correct,
        exhaustive_asserted,
        exhaustive_prediction_digest: exhaustive_prediction.map(grid_digest),
        exhaustive_exact_correct,
        exhaustive_certified_assertion: certified,
        false_certainty,
        exhaustive_candidates_evaluated: exhaustive.candidates_evaluated,
        exhaustive_training_pair_checks: exhaustive.training_pair_checks,
        wall_time_us,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_policy(
    train: &[GridPair],
    test_input: &Grid,
    candidates: &[CandidateTransform],
    grammar_commitment: &str,
    visible_commitment: &str,
    budget: usize,
    policy: PolicyKind,
    policy_seed: Option<u64>,
) -> Result<PolicyRun, String> {
    let budget = budget.min(candidates.len());
    if budget == 0 || candidates.is_empty() {
        return Err("policy run requires a non-empty grammar and positive budget".into());
    }

    let order = policy_order(candidates.len(), policy, policy_seed)?;
    let mut seen = HashSet::new();
    let mut actions = Vec::with_capacity(budget);
    let mut consistent_predictions: Vec<Grid> = Vec::new();
    let mut training_pair_checks = 0usize;

    for (step, candidate_id) in order.into_iter().take(budget).enumerate() {
        if candidate_id >= candidates.len() {
            return Err(format!("policy emitted out-of-range candidate id {candidate_id}"));
        }
        if !seen.insert(candidate_id) {
            return Err(format!("policy repeated candidate id {candidate_id}"));
        }
        let candidate = candidates[candidate_id];
        let evaluation = evaluate_native_action(train, test_input, candidate_id, candidate, step)?;
        training_pair_checks = training_pair_checks.saturating_add(evaluation.pairs_checked);
        if evaluation.training_consistent {
            let prediction = apply_transform(test_input, candidate);
            if !consistent_predictions.iter().any(|known| known == &prediction) {
                consistent_predictions.push(prediction);
            }
        }
        actions.push(evaluation);
    }

    let decision = decision_from_predictions(consistent_predictions);
    let sealed_commitment = policy_seal(
        policy,
        policy_seed,
        budget,
        grammar_commitment,
        visible_commitment,
        &actions,
        &decision,
    )?;

    Ok(PolicyRun {
        policy,
        policy_seed,
        budget,
        action_space_commitment: grammar_commitment.into(),
        solver_visible_commitment: visible_commitment.into(),
        candidates_evaluated: actions.len(),
        training_pair_checks,
        actions,
        decision,
        sealed_commitment,
    })
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
                return Err("canonical-order policy must not receive a random seed".into());
            }
        }
        PolicyKind::UniformRandom => {
            let seed = seed.ok_or_else(|| "uniform-random policy requires a seed".to_string())?;
            let mut rng = StdRng::seed_from_u64(seed);
            order.shuffle(&mut rng);
        }
    }
    Ok(order)
}

fn evaluate_native_action(
    train: &[GridPair],
    test_input: &Grid,
    candidate_id: usize,
    candidate: CandidateTransform,
    step: usize,
) -> Result<NativeActionRecord, String> {
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
    let prediction_digest = training_consistent.then(|| grid_digest(&apply_transform(test_input, candidate)));
    let candidate_name = transform_name(candidate);
    let action_commitment = native_action_commitment(
        step,
        candidate_id,
        &candidate_name,
        pairs_checked,
        mismatch,
        training_consistent,
        prediction_digest.as_deref(),
    );

    Ok(NativeActionRecord {
        step,
        candidate_id,
        candidate_name,
        pairs_checked,
        first_mismatch_index: mismatch,
        training_consistent,
        test_prediction_digest: prediction_digest,
        action_commitment,
    })
}

fn exhaustive_reference(
    train: &[GridPair],
    test_input: &Grid,
    candidates: &[CandidateTransform],
) -> ExhaustiveReference {
    let mut predictions = Vec::<Grid>::new();
    let mut pair_checks = 0usize;
    for candidate in candidates {
        let mut consistent = true;
        for pair in train {
            pair_checks = pair_checks.saturating_add(1);
            if apply_transform(&pair.input, *candidate) != pair.output {
                consistent = false;
                break;
            }
        }
        if consistent {
            let prediction = apply_transform(test_input, *candidate);
            if !predictions.iter().any(|known| known == &prediction) {
                predictions.push(prediction);
            }
        }
    }
    ExhaustiveReference {
        decision: decision_from_predictions(predictions),
        candidates_evaluated: candidates.len(),
        training_pair_checks: pair_checks,
    }
}

fn decision_from_predictions(mut predictions: Vec<Grid>) -> LocalDecision {
    match predictions.len() {
        0 => LocalDecision::Abstained(AbstentionKind::NoTrainingConsistentCandidate),
        1 => LocalDecision::Asserted(predictions.remove(0)),
        _ => LocalDecision::Abstained(AbstentionKind::ConflictingPredictions),
    }
}

fn asserted_grid(decision: &LocalDecision) -> Option<&Grid> {
    match decision {
        LocalDecision::Asserted(grid) => Some(grid),
        LocalDecision::Abstained(_) => None,
    }
}

fn action_space_manifest(candidates: &[CandidateTransform]) -> Vec<ArcActionManifestEntry> {
    candidates
        .iter()
        .enumerate()
        .map(|(candidate_id, candidate)| ArcActionManifestEntry {
            candidate_id,
            transform: *candidate,
            name: transform_name(*candidate),
        })
        .collect()
}

fn action_space_commitment(manifest: &[ArcActionManifestEntry]) -> Result<String, String> {
    let encoded = serde_json::to_vec(manifest)
        .map_err(|err| format!("failed to encode ARC action manifest: {err}"))?;
    let mut hasher = blake3::Hasher::new();
    hash_str(&mut hasher, ACTION_SPACE_VERSION);
    hash_bytes(&mut hasher, &encoded);
    Ok(hasher.finalize().to_hex().to_string())
}

fn solver_visible_commitment(
    raw_task_hash: &str,
    test_index: usize,
    train: &[GridPair],
    test_input: &Grid,
    grammar_commitment: &str,
    budget: usize,
) -> Result<String, String> {
    let train_visible = train
        .iter()
        .map(|pair| (&pair.input, &pair.output))
        .collect::<Vec<_>>();
    let encoded = serde_json::to_vec(&(train_visible, test_input))
        .map_err(|err| format!("failed to encode solver-visible ARC material: {err}"))?;
    let mut hasher = blake3::Hasher::new();
    hash_str(&mut hasher, "symthaea/reasoning/arc-policy-visible/v1");
    hash_str(&mut hasher, raw_task_hash);
    hash_u64(&mut hasher, test_index as u64);
    hash_str(&mut hasher, grammar_commitment);
    hash_u64(&mut hasher, budget as u64);
    hash_bytes(&mut hasher, &encoded);
    Ok(hasher.finalize().to_hex().to_string())
}

fn policy_seal(
    policy: PolicyKind,
    seed: Option<u64>,
    budget: usize,
    grammar_commitment: &str,
    visible_commitment: &str,
    actions: &[NativeActionRecord],
    decision: &LocalDecision,
) -> Result<String, String> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(POLICY_SEAL_DOMAIN);
    hash_str(&mut hasher, policy.id());
    match seed {
        Some(seed) => {
            hasher.update(&[1]);
            hash_u64(&mut hasher, seed);
        }
        None => hasher.update(&[0]),
    }
    hash_u64(&mut hasher, budget as u64);
    hash_str(&mut hasher, grammar_commitment);
    hash_str(&mut hasher, visible_commitment);
    hash_u64(&mut hasher, actions.len() as u64);
    for action in actions {
        hash_str(&mut hasher, &action.action_commitment);
    }
    match decision {
        LocalDecision::Asserted(grid) => {
            hasher.update(&[1]);
            hash_str(&mut hasher, &grid_digest(grid));
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
    Ok(hasher.finalize().to_hex().to_string())
}

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
    hasher.update(ACTION_DOMAIN);
    hash_u64(&mut hasher, step as u64);
    hash_u64(&mut hasher, candidate_id as u64);
    hash_str(&mut hasher, candidate_name);
    hash_u64(&mut hasher, pairs_checked as u64);
    match mismatch {
        Some(index) => {
            hasher.update(&[1]);
            hash_u64(&mut hasher, index as u64);
        }
        None => hasher.update(&[0]),
    }
    hasher.update(&[u8::from(training_consistent)]);
    match prediction_digest {
        Some(digest) => {
            hasher.update(&[1]);
            hash_str(&mut hasher, digest);
        }
        None => hasher.update(&[0]),
    }
    hasher.finalize().to_hex().to_string()
}

fn derive_policy_seed(root: u64, visible_commitment: &str) -> u64 {
    let digest = blake3::hash(visible_commitment.as_bytes());
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest.as_bytes()[..8]);
    root ^ u64::from_le_bytes(bytes)
}

fn selected_task_files(split_dir: &Path, max_tasks: Option<usize>) -> Result<Vec<PathBuf>, String> {
    let mut task_files = Vec::new();
    for entry in fs::read_dir(split_dir)
        .map_err(|err| format!("failed to list {}: {err}", split_dir.display()))?
    {
        let entry = entry
            .map_err(|err| format!("failed to read entry in {}: {err}", split_dir.display()))?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "json") {
            task_files.push(path);
        }
    }
    task_files.sort();
    if let Some(limit) = max_tasks {
        task_files.truncate(limit);
    }
    if task_files.is_empty() {
        return Err(format!("no ARC JSON task files selected from {}", split_dir.display()));
    }
    Ok(task_files)
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
        grid.push(
            cells
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
                .collect::<Result<Vec<_>, String>>()?,
        );
    }
    Ok(grid)
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

fn grid_digest(grid: &Grid) -> String {
    let encoded = serde_json::to_vec(grid).expect("ARC grid serialization is infallible");
    blake3::hash(&encoded).to_hex().to_string()
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

    fn pair(input: Grid, output: Grid) -> GridPair {
        GridPair { input, output }
    }

    fn two_candidate_ambiguous_fixture() -> (Vec<GridPair>, Grid, Vec<CandidateTransform>) {
        // Identity and reflect-X are indistinguishable on the all-zero training input/output,
        // but produce different predictions on the asymmetric held-out input.
        let train_grid = vec![vec![0, 0, 0], vec![0, 0, 0]];
        let train = vec![pair(train_grid.clone(), train_grid)];
        let test_input = vec![vec![1, 0, 2], vec![3, 4, 0]];
        let candidates = vec![
            CandidateTransform {
                geometry: Geometry::Identity,
                color: ColorOp::None,
            },
            CandidateTransform {
                geometry: Geometry::ReflectX,
                color: ColorOp::None,
            },
        ];
        (train, test_input, candidates)
    }

    #[test]
    fn canonical_grammar_matches_rq003_size() {
        assert_eq!(canonical_candidates().len(), 54 * 91);
    }

    #[test]
    fn same_random_seed_reproduces_candidate_sequence() {
        let a = policy_order(100, PolicyKind::UniformRandom, Some(44)).unwrap();
        let b = policy_order(100, PolicyKind::UniformRandom, Some(44)).unwrap();
        assert_eq!(a, b);
        let unique = a.iter().copied().collect::<HashSet<_>>();
        assert_eq!(unique.len(), 100);
    }

    #[test]
    fn policy_run_has_no_target_parameter_and_is_target_invariant() {
        let (train, test_input, candidates) = two_candidate_ambiguous_fixture();
        let manifest = action_space_manifest(&candidates);
        let grammar = action_space_commitment(&manifest).unwrap();
        let visible = "solver-visible-fixture";
        let a = run_policy(
            &train,
            &test_input,
            &candidates,
            &grammar,
            visible,
            1,
            PolicyKind::CanonicalOrder,
            None,
        )
        .unwrap();
        // Two incompatible evaluator targets cannot affect the already-sealed policy episode,
        // because neither target is an argument to run_policy().
        let expected_a = test_input.clone();
        let expected_b = GridEncoder::reflect_x(&test_input);
        assert_ne!(expected_a, expected_b);
        let b = run_policy(
            &train,
            &test_input,
            &candidates,
            &grammar,
            visible,
            1,
            PolicyKind::CanonicalOrder,
            None,
        )
        .unwrap();
        assert_eq!(
            a.actions.iter().map(|x| x.candidate_id).collect::<Vec<_>>(),
            b.actions.iter().map(|x| x.candidate_id).collect::<Vec<_>>()
        );
        assert_eq!(a.sealed_commitment, b.sealed_commitment);
    }

    #[test]
    fn bounded_local_assertion_can_fail_exhaustive_certification() {
        let (train, test_input, candidates) = two_candidate_ambiguous_fixture();
        let grammar = action_space_commitment(&action_space_manifest(&candidates)).unwrap();
        let run = run_policy(
            &train,
            &test_input,
            &candidates,
            &grammar,
            "visible",
            1,
            PolicyKind::CanonicalOrder,
            None,
        )
        .unwrap();
        assert!(matches!(run.decision, LocalDecision::Asserted(_)));
        let full = exhaustive_reference(&train, &test_input, &candidates);
        assert!(matches!(
            full.decision,
            LocalDecision::Abstained(AbstentionKind::ConflictingPredictions)
        ));
    }

    #[test]
    fn full_budget_matches_exhaustive_decision() {
        let (train, test_input, candidates) = two_candidate_ambiguous_fixture();
        let grammar = action_space_commitment(&action_space_manifest(&candidates)).unwrap();
        let run = run_policy(
            &train,
            &test_input,
            &candidates,
            &grammar,
            "visible",
            candidates.len(),
            PolicyKind::CanonicalOrder,
            None,
        )
        .unwrap();
        let full = exhaustive_reference(&train, &test_input, &candidates);
        assert_eq!(run.decision, full.decision);
    }

    #[test]
    fn native_action_sequence_never_repeats_candidate() {
        let candidates = canonical_candidates();
        let grammar = action_space_commitment(&action_space_manifest(&candidates)).unwrap();
        let input = vec![vec![1, 2], vec![3, 4]];
        let train = vec![pair(input.clone(), input.clone())];
        let run = run_policy(
            &train,
            &input,
            &candidates,
            &grammar,
            "visible",
            128,
            PolicyKind::UniformRandom,
            Some(99),
        )
        .unwrap();
        let ids = run
            .actions
            .iter()
            .map(|action| action.candidate_id)
            .collect::<HashSet<_>>();
        assert_eq!(ids.len(), run.actions.len());
    }
}
