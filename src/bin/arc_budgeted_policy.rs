// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! RQ-006Z-C development lane: resource-bounded policy control over the native RQ-003 ARC grammar.
//!
//! The policy can choose only which existing `CandidateTransform` to evaluate next. Candidate
//! evaluation uses training demonstrations plus the held-out test INPUT. The hidden expected test
//! grid is not accepted by `run_policy`; both policy episodes are sealed before target evaluation
//! or exhaustive-reference certification begins.
//!
//! This lane does not replace RQ-003. It measures search control under an equal candidate budget
//! and reports policy-local answers separately from full-grammar certification.

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

const SCHEMA_VERSION: u32 = 2;
const CONFIGURATION_ID: &str = "arc-native-budgeted-policy-v2";
const ACTION_SPACE_VERSION: &str = "rq003-candidate-transform-v1";
const ACTION_SPACE_DOMAIN: &[u8] = b"symthaea/reasoning/arc-action-space/v1";
const SOLVER_VISIBLE_DOMAIN: &[u8] = b"symthaea/reasoning/arc-policy-visible/v2";
const POLICY_SEAL_DOMAIN: &[u8] = b"symthaea/reasoning/arc-budgeted-policy-seal/v2";
const ACTION_DOMAIN: &[u8] = b"symthaea/reasoning/arc-native-action/v1";
const GRID_DOMAIN: &[u8] = b"symthaea/reasoning/arc-grid/v1";
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
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

#[derive(Debug, Clone, PartialEq, Eq)]
enum LocalDecision {
    Asserted(Grid),
    Abstained(AbstentionKind),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AbstentionKind {
    NoTrainingConsistentCandidate,
    ConflictingPredictions,
}

#[derive(Debug, Clone)]
struct PolicyRun {
    policy: PolicyKind,
    policy_seed: Option<u64>,
    budget: usize,
    grammar_size: usize,
    action_space_commitment: String,
    solver_visible_commitment: String,
    actions: Vec<NativeActionRecord>,
    decision: LocalDecision,
    candidates_evaluated: usize,
    training_pair_checks: usize,
    policy_wall_time_us: u64,
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
    actions: Vec<NativeActionRecord>,
    candidates_evaluated: usize,
    training_pair_checks: usize,
    policy_wall_time_us: u64,
    asserted: bool,
    local_prediction_digest: Option<String>,
    evaluator_target_digest: String,
    exact_correct: bool,
    exhaustive_asserted: bool,
    exhaustive_prediction_digest: Option<String>,
    exhaustive_exact_correct: bool,
    exhaustive_certified_assertion: bool,
    exhaustive_decision_match: bool,
    false_certainty: bool,
    missed_reference_assertion: bool,
    exhaustive_candidates_evaluated: usize,
    exhaustive_training_pair_checks: usize,
}

#[derive(Debug, Default, Clone, Serialize)]
struct PolicyAggregate {
    episodes: usize,
    asserted: usize,
    exact_correct: usize,
    exhaustive_certified_assertions: usize,
    exhaustive_decision_matches: usize,
    false_certainty: usize,
    missed_reference_assertions: usize,
    candidates_evaluated: u64,
    training_pair_checks: u64,
    policy_wall_time_us: u64,
    exact_accuracy: Option<f64>,
    coverage: Option<f64>,
    selective_accuracy: Option<f64>,
    certified_coverage: Option<f64>,
    exhaustive_decision_match_rate: Option<f64>,
    false_certainty_rate: Option<f64>,
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
        self.exhaustive_decision_matches = self
            .exhaustive_decision_matches
            .saturating_add(usize::from(result.exhaustive_decision_match));
        self.false_certainty = self
            .false_certainty
            .saturating_add(usize::from(result.false_certainty));
        self.missed_reference_assertions = self
            .missed_reference_assertions
            .saturating_add(usize::from(result.missed_reference_assertion));
        self.candidates_evaluated = self
            .candidates_evaluated
            .saturating_add(result.candidates_evaluated as u64);
        self.training_pair_checks = self
            .training_pair_checks
            .saturating_add(result.training_pair_checks as u64);
        self.policy_wall_time_us = self
            .policy_wall_time_us
            .saturating_add(result.policy_wall_time_us);
    }

    fn finalize(&mut self) {
        self.exact_accuracy = ratio(self.exact_correct, self.episodes);
        self.coverage = ratio(self.asserted, self.episodes);
        self.selective_accuracy = ratio(self.exact_correct, self.asserted);
        self.certified_coverage = ratio(self.exhaustive_certified_assertions, self.episodes);
        self.exhaustive_decision_match_rate =
            ratio(self.exhaustive_decision_matches, self.episodes);
        self.false_certainty_rate = ratio(self.false_certainty, self.asserted);
    }
}

#[derive(Debug, Default, Clone, Serialize)]
struct ReferenceAggregate {
    episodes: usize,
    asserted: usize,
    exact_correct: usize,
    candidates_evaluated: u64,
    training_pair_checks: u64,
    exact_accuracy: Option<f64>,
    coverage: Option<f64>,
}

impl ReferenceAggregate {
    fn observe(&mut self, reference: &ExhaustiveReference, expected: &Grid) {
        let prediction = asserted_grid(&reference.decision);
        self.episodes = self.episodes.saturating_add(1);
        self.asserted = self.asserted.saturating_add(usize::from(prediction.is_some()));
        self.exact_correct = self.exact_correct.saturating_add(usize::from(
            prediction.is_some_and(|prediction| prediction == expected),
        ));
        self.candidates_evaluated = self
            .candidates_evaluated
            .saturating_add(reference.candidates_evaluated as u64);
        self.training_pair_checks = self
            .training_pair_checks
            .saturating_add(reference.training_pair_checks as u64);
    }

    fn finalize(&mut self) {
        self.exact_accuracy = ratio(self.exact_correct, self.episodes);
        self.coverage = ratio(self.asserted, self.episodes);
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
    exhaustive_reference: ReferenceAggregate,
    paired_exact_delta_random_minus_canonical: i64,
    paired_exact_delta_exhaustive_minus_canonical: i64,
    paired_exact_delta_exhaustive_minus_random: i64,
    tasks: Vec<PolicyTaskResult>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("ARC budgeted policy qualification failed: {err}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let subject_revision = required_full_revision("SYMTHAEA_SUBJECT_REVISION")?;
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
    let max_tasks = optional_usize_env("SYMTHAEA_ARC_MAX_TASKS")?;
    let requested_budget =
        optional_usize_env("SYMTHAEA_ARC_CANDIDATE_BUDGET")?.unwrap_or(DEFAULT_BUDGET);
    let random_seed_root =
        optional_u64_env("SYMTHAEA_ARC_POLICY_SEED")?.unwrap_or(DEFAULT_RANDOM_SEED);

    let candidates = canonical_candidates();
    if requested_budget == 0 || requested_budget > candidates.len() {
        return Err(format!(
            "SYMTHAEA_ARC_CANDIDATE_BUDGET must be within 1..={} (got {requested_budget})",
            candidates.len()
        ));
    }
    let budget = requested_budget;
    let grammar_commitment = action_space_commitment(&candidates);
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
    let mut reference_aggregate = ReferenceAggregate::default();
    let mut task_results = Vec::new();
    let mut paired_exact_delta = 0i64;
    let mut canonical_reference_delta = 0i64;
    let mut random_reference_delta = 0i64;
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

        for (test_index, test_case) in task.test.iter().enumerate() {
            let problem_id = format!("{file_stem}#test-{test_index}");
            let visible_commitment = solver_visible_commitment(
                &problem_id,
                &task.train,
                &test_case.input,
                &grammar_commitment,
                budget,
            );
            let derived_seed = derive_policy_seed(random_seed_root, &visible_commitment);

            // Both solver episodes seal before the hidden target or exhaustive reference is used.
            let canonical_run = run_policy(
                &task.train,
                &test_case.input,
                &candidates,
                &grammar_commitment,
                &visible_commitment,
                budget,
                PolicyKind::CanonicalOrder,
                None,
            )?;
            let random_run = run_policy(
                &task.train,
                &test_case.input,
                &candidates,
                &grammar_commitment,
                &visible_commitment,
                budget,
                PolicyKind::UniformRandom,
                Some(derived_seed),
            )?;

            validate_policy_run(&canonical_run)?;
            validate_policy_run(&random_run)?;

            // Evaluator-only boundary begins here.
            let reference = exhaustive_reference(&task.train, &test_case.input, &candidates);
            reference_aggregate.observe(&reference, &test_case.expected);
            let reference_exact = decision_exact(&reference.decision, &test_case.expected);

            let canonical = evaluate_sealed_run(
                &problem_id,
                canonical_run,
                &test_case.expected,
                &reference,
            )?;
            let random = evaluate_sealed_run(
                &problem_id,
                random_run,
                &test_case.expected,
                &reference,
            )?;

            paired_exact_delta += bool_i64(random.exact_correct) - bool_i64(canonical.exact_correct);
            canonical_reference_delta +=
                bool_i64(reference_exact) - bool_i64(canonical.exact_correct);
            random_reference_delta += bool_i64(reference_exact) - bool_i64(random.exact_correct);

            canonical_aggregate.observe(&canonical);
            random_aggregate.observe(&random);
            task_results.push(canonical);
            task_results.push(random);
            test_cases_evaluated = test_cases_evaluated.saturating_add(1);
        }
    }

    canonical_aggregate.finalize();
    random_aggregate.finalize();
    reference_aggregate.finalize();

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
        exhaustive_reference: reference_aggregate,
        paired_exact_delta_random_minus_canonical: paired_exact_delta,
        paired_exact_delta_exhaustive_minus_canonical: canonical_reference_delta,
        paired_exact_delta_exhaustive_minus_random: random_reference_delta,
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

    println!("canonical exact:      {:?}", report.canonical.exact_accuracy);
    println!("random exact:         {:?}", report.uniform_random.exact_accuracy);
    println!("exhaustive exact:     {:?}", report.exhaustive_reference.exact_accuracy);
    println!("canonical coverage:   {:?}", report.canonical.coverage);
    println!("random coverage:      {:?}", report.uniform_random.coverage);
    println!("canonical certified:  {:?}", report.canonical.certified_coverage);
    println!("random certified:     {:?}", report.uniform_random.certified_coverage);
    println!("report: {}", results_path.display());
    Ok(())
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
    if candidates.is_empty() || budget == 0 || budget > candidates.len() {
        return Err(format!(
            "policy run requires budget within 1..={} (got {budget})",
            candidates.len()
        ));
    }

    let started = Instant::now();
    let order = policy_order(candidates.len(), policy, policy_seed)?;
    let mut seen = HashSet::new();
    let mut actions = Vec::with_capacity(budget);
    let mut consistent_predictions = Vec::<Grid>::new();
    let mut training_pair_checks = 0usize;

    for (step, candidate_id) in order.into_iter().take(budget).enumerate() {
        if candidate_id >= candidates.len() {
            return Err(format!("policy emitted out-of-range candidate id {candidate_id}"));
        }
        if !seen.insert(candidate_id) {
            return Err(format!("policy repeated candidate id {candidate_id}"));
        }

        let candidate = candidates[candidate_id];
        let action = evaluate_native_action(train, test_input, candidate_id, candidate, step);
        training_pair_checks = training_pair_checks.saturating_add(action.pairs_checked);
        if action.training_consistent {
            let prediction = apply_transform(test_input, candidate);
            if !consistent_predictions.iter().any(|known| known == &prediction) {
                consistent_predictions.push(prediction);
            }
        }
        actions.push(action);
    }

    let decision = decision_from_predictions(consistent_predictions);
    let candidates_evaluated = actions.len();
    let policy_wall_time_us = started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
    let sealed_commitment = policy_seal(
        policy,
        policy_seed,
        budget,
        candidates.len(),
        grammar_commitment,
        visible_commitment,
        &actions,
        &decision,
    );

    Ok(PolicyRun {
        policy,
        policy_seed,
        budget,
        grammar_size: candidates.len(),
        action_space_commitment: grammar_commitment.into(),
        solver_visible_commitment: visible_commitment.into(),
        actions,
        decision,
        candidates_evaluated,
        training_pair_checks,
        policy_wall_time_us,
        sealed_commitment,
    })
}

fn validate_policy_run(run: &PolicyRun) -> Result<(), String> {
    if run.budget == 0 || run.budget > run.grammar_size {
        return Err("sealed policy run has invalid budget".into());
    }
    if run.actions.len() != run.budget || run.candidates_evaluated != run.actions.len() {
        return Err("sealed policy run did not consume its exact candidate budget".into());
    }

    let mut seen = HashSet::new();
    let mut pair_checks = 0usize;
    for (index, action) in run.actions.iter().enumerate() {
        if action.step != index {
            return Err(format!("sealed action step {} is out of order", action.step));
        }
        if action.candidate_id >= run.grammar_size || !seen.insert(action.candidate_id) {
            return Err(format!(
                "sealed action candidate {} is invalid or repeated",
                action.candidate_id
            ));
        }
        if action.training_consistent != action.first_mismatch_index.is_none() {
            return Err(format!(
                "sealed action {} has inconsistent mismatch semantics",
                action.step
            ));
        }
        if action.training_consistent != action.test_prediction_digest.is_some() {
            return Err(format!(
                "sealed action {} has inconsistent prediction semantics",
                action.step
            ));
        }
        let expected_commitment = native_action_commitment_from_record(action);
        if action.action_commitment != expected_commitment {
            return Err(format!("sealed action {} commitment mismatch", action.step));
        }
        pair_checks = pair_checks.saturating_add(action.pairs_checked);
    }
    if pair_checks != run.training_pair_checks {
        return Err("sealed policy run training-pair accounting mismatch".into());
    }

    let expected_seal = policy_seal(
        run.policy,
        run.policy_seed,
        run.budget,
        run.grammar_size,
        &run.action_space_commitment,
        &run.solver_visible_commitment,
        &run.actions,
        &run.decision,
    );
    if run.sealed_commitment != expected_seal {
        return Err("sealed policy commitment mismatch".into());
    }
    Ok(())
}

fn evaluate_sealed_run(
    problem_id: &str,
    run: PolicyRun,
    expected: &Grid,
    reference: &ExhaustiveReference,
) -> Result<PolicyTaskResult, String> {
    validate_policy_run(&run)?;

    let local_prediction = asserted_grid(&run.decision);
    let exhaustive_prediction = asserted_grid(&reference.decision);
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
    let missed_reference_assertion = !asserted && exhaustive_asserted;
    let exhaustive_decision_match = decisions_equivalent(&run.decision, &reference.decision);
    let local_prediction_digest = local_prediction.map(grid_digest);
    let exhaustive_prediction_digest = exhaustive_prediction.map(grid_digest);
    let evaluator_target_digest = grid_digest(expected);

    Ok(PolicyTaskResult {
        problem_id: problem_id.into(),
        policy_id: run.policy.id().into(),
        policy_seed: run.policy_seed,
        budget: run.budget,
        grammar_size: run.grammar_size,
        action_space_commitment: run.action_space_commitment,
        solver_visible_commitment: run.solver_visible_commitment,
        policy_sealed_commitment: run.sealed_commitment,
        actions: run.actions,
        candidates_evaluated: run.candidates_evaluated,
        training_pair_checks: run.training_pair_checks,
        policy_wall_time_us: run.policy_wall_time_us,
        asserted,
        local_prediction_digest,
        evaluator_target_digest,
        exact_correct,
        exhaustive_asserted,
        exhaustive_prediction_digest,
        exhaustive_exact_correct,
        exhaustive_certified_assertion: certified,
        exhaustive_decision_match,
        false_certainty,
        missed_reference_assertion,
        exhaustive_candidates_evaluated: reference.candidates_evaluated,
        exhaustive_training_pair_checks: reference.training_pair_checks,
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
) -> NativeActionRecord {
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
    let prediction_digest = if training_consistent {
        Some(grid_digest(&apply_transform(test_input, candidate)))
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
        prediction_digest.as_deref(),
    );

    NativeActionRecord {
        step,
        candidate_id,
        candidate_name,
        pairs_checked,
        first_mismatch_index: mismatch,
        training_consistent,
        test_prediction_digest: prediction_digest,
        action_commitment,
    }
}

fn native_action_commitment_from_record(action: &NativeActionRecord) -> String {
    native_action_commitment(
        action.step,
        action.candidate_id,
        &action.candidate_name,
        action.pairs_checked,
        action.first_mismatch_index,
        action.training_consistent,
        action.test_prediction_digest.as_deref(),
    )
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

fn decisions_equivalent(left: &LocalDecision, right: &LocalDecision) -> bool {
    match (left, right) {
        (LocalDecision::Asserted(a), LocalDecision::Asserted(b)) => a == b,
        (LocalDecision::Abstained(a), LocalDecision::Abstained(b)) => a == b,
        _ => false,
    }
}

fn decision_exact(decision: &LocalDecision, expected: &Grid) -> bool {
    asserted_grid(decision).is_some_and(|prediction| prediction == expected)
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

#[allow(clippy::too_many_arguments)]
fn policy_seal(
    policy: PolicyKind,
    seed: Option<u64>,
    budget: usize,
    grammar_size: usize,
    grammar_commitment: &str,
    visible_commitment: &str,
    actions: &[NativeActionRecord],
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
        return Err(format!(
            "no ARC JSON task files selected from {}",
            split_dir.display()
        ));
    }
    Ok(task_files)
}

fn required_env(name: &str) -> Result<String, String> {
    match env::var(name) {
        Ok(value) if !value.trim().is_empty() => Ok(value.trim().to_string()),
        Ok(_) => Err(format!("required environment variable {name} is empty")),
        Err(err) => Err(format!("required environment variable {name} is missing: {err}")),
    }
}

fn required_full_revision(name: &str) -> Result<String, String> {
    let revision = required_env(name)?;
    let valid_len = revision.len() == 40 || revision.len() == 64;
    if !valid_len || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!(
            "{name} must be a full 40- or 64-hex revision, got `{revision}`"
        ));
    }
    Ok(revision.to_ascii_lowercase())
}

fn optional_usize_env(name: &str) -> Result<Option<usize>, String> {
    match env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map(Some)
            .map_err(|err| format!("{name} must be an unsigned integer: {err}")),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(err) => Err(format!("failed to read {name}: {err}")),
    }
}

fn optional_u64_env(name: &str) -> Result<Option<u64>, String> {
    match env::var(name) {
        Ok(value) => value
            .parse::<u64>()
            .map(Some)
            .map_err(|err| format!("{name} must be an unsigned integer: {err}")),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(err) => Err(format!("failed to read {name}: {err}")),
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

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator != 0).then_some(numerator as f64 / denominator as f64)
}

const fn bool_i64(value: bool) -> i64 {
    if value { 1 } else { 0 }
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

    fn ambiguous_fixture() -> (Vec<GridPair>, Grid, Vec<CandidateTransform>) {
        // Identity and reflect-X are indistinguishable on training data but disagree on test input.
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
    fn canonical_grammar_matches_rq003_size_order_and_anchors() {
        let candidates = canonical_candidates();
        assert_eq!(candidates.len(), 54 * 91);
        assert_eq!(transform_name(candidates[0]), "identity");
        assert_eq!(transform_name(candidates[1]), "identity+color(0->1)");
        assert_eq!(transform_name(candidates[91]), "reflect-x");
        assert_eq!(transform_name(candidates[546]), "translate(-3,-3)");
        assert_eq!(
            transform_name(*candidates.last().unwrap()),
            "translate(3,3)+color(9->8)"
        );
        let names = candidates
            .iter()
            .map(|candidate| transform_name(*candidate))
            .collect::<HashSet<_>>();
        assert_eq!(names.len(), candidates.len());
    }

    #[test]
    fn same_random_seed_reproduces_unique_candidate_sequence() {
        let a = policy_order(100, PolicyKind::UniformRandom, Some(44)).unwrap();
        let b = policy_order(100, PolicyKind::UniformRandom, Some(44)).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.iter().copied().collect::<HashSet<_>>().len(), 100);
    }

    #[test]
    fn hidden_target_change_cannot_change_visible_lineage_or_policy_trace() {
        let (train, test_input, candidates) = ambiguous_fixture();
        let grammar = action_space_commitment(&candidates);
        let visible = solver_visible_commitment("fixture#0", &train, &test_input, &grammar, 1);
        let seed = derive_policy_seed(1234, &visible);
        let run = run_policy(
            &train,
            &test_input,
            &candidates,
            &grammar,
            &visible,
            1,
            PolicyKind::CanonicalOrder,
            None,
        )
        .unwrap();
        let reference = exhaustive_reference(&train, &test_input, &candidates);

        let target_a = test_input.clone();
        let target_b = GridEncoder::reflect_x(&test_input);
        assert_ne!(target_a, target_b);
        assert_eq!(seed, derive_policy_seed(1234, &visible));

        let a = evaluate_sealed_run("fixture#0", run.clone(), &target_a, &reference).unwrap();
        let b = evaluate_sealed_run("fixture#0", run, &target_b, &reference).unwrap();
        assert_eq!(a.solver_visible_commitment, b.solver_visible_commitment);
        assert_eq!(a.policy_sealed_commitment, b.policy_sealed_commitment);
        assert_eq!(a.actions, b.actions);
        assert_ne!(a.evaluator_target_digest, b.evaluator_target_digest);
        assert_ne!(a.exact_correct, b.exact_correct);
    }

    #[test]
    fn bounded_local_assertion_is_flagged_when_full_grammar_is_ambiguous() {
        let (train, test_input, candidates) = ambiguous_fixture();
        let grammar = action_space_commitment(&candidates);
        let visible = solver_visible_commitment("fixture#0", &train, &test_input, &grammar, 1);
        let run = run_policy(
            &train,
            &test_input,
            &candidates,
            &grammar,
            &visible,
            1,
            PolicyKind::CanonicalOrder,
            None,
        )
        .unwrap();
        let reference = exhaustive_reference(&train, &test_input, &candidates);
        let result = evaluate_sealed_run("fixture#0", run, &test_input, &reference).unwrap();
        assert!(result.asserted);
        assert!(result.exact_correct);
        assert!(result.false_certainty);
        assert!(!result.exhaustive_certified_assertion);
        assert!(!result.exhaustive_decision_match);
    }

    #[test]
    fn full_budget_matches_exhaustive_decision() {
        let (train, test_input, candidates) = ambiguous_fixture();
        let grammar = action_space_commitment(&candidates);
        let visible = solver_visible_commitment(
            "fixture#0",
            &train,
            &test_input,
            &grammar,
            candidates.len(),
        );
        let run = run_policy(
            &train,
            &test_input,
            &candidates,
            &grammar,
            &visible,
            candidates.len(),
            PolicyKind::CanonicalOrder,
            None,
        )
        .unwrap();
        let reference = exhaustive_reference(&train, &test_input, &candidates);
        assert_eq!(run.decision, reference.decision);
    }

    #[test]
    fn nested_action_tampering_invalidates_policy_run() {
        let candidates = canonical_candidates();
        let input = vec![vec![1, 2], vec![3, 4]];
        let train = vec![pair(input.clone(), input.clone())];
        let grammar = action_space_commitment(&candidates);
        let visible = solver_visible_commitment("fixture#0", &train, &input, &grammar, 4);
        let mut run = run_policy(
            &train,
            &input,
            &candidates,
            &grammar,
            &visible,
            4,
            PolicyKind::CanonicalOrder,
            None,
        )
        .unwrap();
        validate_policy_run(&run).unwrap();
        run.actions[0].candidate_name.push_str("-tampered");
        assert!(validate_policy_run(&run).is_err());
    }

    #[test]
    fn policy_budget_is_exact_and_never_silently_clipped() {
        let candidates = canonical_candidates();
        let input = vec![vec![1]];
        let train = vec![pair(input.clone(), input.clone())];
        let grammar = action_space_commitment(&candidates);
        let visible = solver_visible_commitment("fixture#0", &train, &input, &grammar, 1);
        assert!(run_policy(
            &train,
            &input,
            &candidates,
            &grammar,
            &visible,
            candidates.len() + 1,
            PolicyKind::CanonicalOrder,
            None,
        )
        .is_err());
    }
}
