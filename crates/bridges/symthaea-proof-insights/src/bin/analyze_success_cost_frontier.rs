// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Analyze a frozen target x arm x budget schedule as a verified-success/cost frontier.
//!
//! This is a descriptive evaluator, not a proof checker or semantic auditor. It joins
//! each result to the preregistered plan, counts non-attempts in the assigned denominator,
//! requires exact statement binding for execution, and never emits a confirmatory pass.

use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const PLAN_VERSION: &str = "pitb.success-cost-frontier-plan.v0";
const OUTCOME_VERSION: &str = "pitb.success-cost-frontier-outcome.v0";
const REPORT_VERSION: &str = "pitb.success-cost-frontier-analysis.v0";
const DEFAULT_BOOTSTRAP_REPLICATES: usize = 10_000;
const DEFAULT_SEED: u64 = 0x5049_5442_5343_4630;

#[derive(Debug, Clone)]
struct Target {
    id: String,
    component_id: String,
    statement_sha256: String,
}

#[derive(Debug, Clone)]
struct Arm {
    id: String,
    method_class: String,
    model_id: String,
    model_version: String,
    checker_id: String,
    checker_version: String,
    config_sha256: String,
}

#[derive(Debug, Clone)]
struct Budget {
    id: String,
    max_model_calls: u64,
    max_tokens: u64,
    max_wall_ms: u64,
    max_cost_microusd: u64,
}

#[derive(Debug, Clone)]
struct Assignment {
    run_id: String,
    target_id: String,
    component_id: String,
    arm_id: String,
    budget_id: String,
    input_sha256: String,
}

#[derive(Debug, Clone)]
struct Plan {
    id: String,
    dataset_kind: String,
    split: String,
    targets: Vec<Target>,
    arms: Vec<Arm>,
    budgets: Vec<Budget>,
    assignments: Vec<Assignment>,
}

#[derive(Debug, Clone)]
struct Outcome {
    target_id: String,
    component_id: String,
    arm_id: String,
    budget_id: String,
    status: String,
    success: bool,
    model_calls: u64,
    tokens: u64,
    wall_ms: u64,
    cost_microusd: u64,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4
        || args.len() > 6
        || args.get(1).is_some_and(|x| x == "--help" || x == "-h")
    {
        eprintln!(
            "usage: {} <frozen-plan.json> <outcomes.jsonl> <report.json> [bootstrap_replicates] [seed]",
            args.first()
                .map(String::as_str)
                .unwrap_or("analyze_success_cost_frontier")
        );
        if args.get(1).is_some_and(|x| x == "--help" || x == "-h") {
            return Ok(());
        }
        bail!("invalid command-line arguments");
    }

    let plan_bytes = fs::read(&args[1]).with_context(|| format!("read plan {}", args[1]))?;
    let outcome_bytes =
        fs::read(&args[2]).with_context(|| format!("read outcomes {}", args[2]))?;
    let plan_value: Value =
        serde_json::from_slice(&plan_bytes).context("parse frozen plan JSON")?;
    let plan = parse_plan(&plan_value)?;
    let bootstrap_replicates = args
        .get(4)
        .map(|x| x.parse::<usize>().context("parse bootstrap replicate count"))
        .transpose()?
        .unwrap_or(DEFAULT_BOOTSTRAP_REPLICATES);
    if bootstrap_replicates < 1_000 {
        bail!("at least 1000 component-bootstrap replicates are required");
    }
    let seed = args
        .get(5)
        .map(|x| x.parse::<u64>().context("parse bootstrap seed"))
        .transpose()?
        .unwrap_or(DEFAULT_SEED);

    let outcomes = parse_and_join_outcomes(&plan, &outcome_bytes)?;
    let report = analyze(
        &plan,
        &outcomes,
        &plan_bytes,
        &outcome_bytes,
        bootstrap_replicates,
        seed,
    )?;
    fs::write(
        &args[3],
        serde_json::to_vec_pretty(&report).context("serialize analysis report")?,
    )
    .with_context(|| format!("write report {}", args[3]))?;

    let counts = report
        .get("counts")
        .context("internal report has no counts")?;
    eprintln!(
        "PITB_SUCCESS_COST_FRONTIER_ANALYSIS: dataset_kind={} planned_cells={} result_rows={} executed_rows={} verified_successes={} decision={}",
        plan.dataset_kind,
        counts["planned_cells"],
        counts["outcome_rows"],
        counts["executed_rows"],
        counts["verified_successes"],
        report["confirmatory_decision"].as_str().unwrap_or("invalid")
    );
    Ok(())
}

fn parse_plan(value: &Value) -> Result<Plan> {
    if string(value, "schema_version", "plan")? != PLAN_VERSION {
        bail!("unsupported plan schema_version");
    }
    let id = string(value, "plan_id", "plan")?.to_owned();
    let dataset_kind = string(value, "dataset_kind", "plan")?.to_owned();
    if !matches!(dataset_kind.as_str(), "synthetic_fixture" | "reviewed_benchmark") {
        bail!("unsupported dataset_kind {dataset_kind:?}");
    }
    let split = string(value, "split", "plan")?.to_owned();
    if !matches!(split.as_str(), "validation" | "test" | "external_challenge") {
        bail!("unsupported split {split:?}");
    }
    if dataset_kind == "synthetic_fixture" && split != "validation" {
        bail!("synthetic frontier inputs are permitted only on validation");
    }
    validate_sha(string(value, "target_manifest_sha256", "plan")?, "target manifest")?;
    validate_sha(string(value, "plan_receipt_sha256", "plan")?, "plan receipt")?;

    let policy = value.get("analysis_policy").context("plan.analysis_policy missing")?;
    if string(policy, "primary_endpoint", "analysis_policy")?
        != "exact_target_semantic_gate_success"
        || string(policy, "success_gate", "analysis_policy")?
            != "target_kernel_and_semantic_gate_and_applicable_successor_receipts"
        || string(policy, "resampling_unit", "analysis_policy")? != "target_component_id"
        || !policy
            .get("require_complete_cartesian_coverage")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        || string(policy, "confirmatory_decision", "analysis_policy")?
            != "not_evaluated_until_real_outcomes_and_receipts_are_validated"
    {
        bail!("frontier plan analysis_policy differs from the required exact-target contract");
    }
    let minimum_levels = policy
        .get("minimum_budget_levels")
        .and_then(Value::as_u64)
        .context("analysis_policy.minimum_budget_levels must be an integer")? as usize;

    let mut targets = Vec::new();
    let mut target_ids = BTreeSet::new();
    for (index, item) in array(value, "targets", "plan")?.iter().enumerate() {
        let context = format!("targets[{index}]");
        let target_id = string(item, "target_id", &context)?.to_owned();
        if !target_ids.insert(target_id.clone()) {
            bail!("duplicate target_id {target_id:?}");
        }
        let component_id = string(item, "component_id", &context)?.to_owned();
        if component_id.is_empty() {
            bail!("{context}: component_id cannot be empty");
        }
        let path = string(item, "artifact_path", &context)?;
        if path.starts_with('/')
            || path.split('/').any(|part| part == ".." || part.is_empty())
        {
            bail!("{context}: artifact_path must be repository-relative and cannot escape the root");
        }
        validate_commit(string(item, "source_commit", &context)?, "target source commit")?;
        // Source commits are SHA-1 identifiers; artifact/statement content digests are SHA-256.
        validate_sha(string(item, "artifact_sha256", &context)?, "target artifact")?;
        let statement_sha256 = string(item, "statement_sha256", &context)?.to_owned();
        validate_sha(&statement_sha256, "target statement")?;
        targets.push(Target {
            id: target_id,
            component_id,
            statement_sha256,
        });
    }
    if targets.is_empty() {
        bail!("plan has no targets");
    }

    let mut arms = Vec::new();
    let mut arm_ids = BTreeSet::new();
    let mut method_classes = BTreeSet::new();
    for (index, item) in array(value, "arms", "plan")?.iter().enumerate() {
        let context = format!("arms[{index}]");
        let arm_id = string(item, "arm_id", &context)?.to_owned();
        if !arm_ids.insert(arm_id.clone()) {
            bail!("duplicate arm_id {arm_id:?}");
        }
        let method_class = string(item, "method_class", &context)?.to_owned();
        method_classes.insert(method_class.clone());
        let prompt_sha256 = string(item, "prompt_sha256", &context)?;
        validate_sha(prompt_sha256, "arm prompt")?;
        let config_sha256 = string(item, "config_sha256", &context)?.to_owned();
        validate_sha(&config_sha256, "arm config")?;
        arms.push(Arm {
            id: arm_id,
            method_class,
            model_id: string(item, "model_id", &context)?.to_owned(),
            model_version: string(item, "model_version", &context)?.to_owned(),
            checker_id: string(item, "checker_id", &context)?.to_owned(),
            checker_version: string(item, "checker_version", &context)?.to_owned(),
            config_sha256,
        });
    }
    if arms.len() < 3 || method_classes.len() < 3 {
        bail!("frontier plan requires at least three arms and three distinct method classes");
    }

    let mut budgets = Vec::new();
    let mut budget_ids = BTreeSet::new();
    for (index, item) in array(value, "budget_levels", "plan")?.iter().enumerate() {
        let context = format!("budget_levels[{index}]");
        let budget_id = string(item, "budget_id", &context)?.to_owned();
        if !budget_ids.insert(budget_id.clone()) {
            bail!("duplicate budget_id {budget_id:?}");
        }
        let budget = Budget {
            id: budget_id,
            max_model_calls: positive_u64(item, "max_model_calls", &context)?,
            max_tokens: positive_u64(item, "max_tokens", &context)?,
            max_wall_ms: positive_u64(item, "max_wall_ms", &context)?,
            max_cost_microusd: positive_u64(item, "max_cost_microusd", &context)?,
        };
        if let Some(previous) = budgets.last() {
            let previous: &Budget = previous;
            if budget.max_model_calls <= previous.max_model_calls
                || budget.max_tokens <= previous.max_tokens
                || budget.max_wall_ms <= previous.max_wall_ms
                || budget.max_cost_microusd <= previous.max_cost_microusd
            {
                bail!("every ordered budget level must strictly increase all resource caps");
            }
        }
        budgets.push(budget);
    }
    if budgets.len() < minimum_levels || minimum_levels < 3 {
        bail!("plan has fewer budget levels than its policy requires");
    }

    let mut assignments = Vec::new();
    let mut run_ids = BTreeSet::new();
    let mut cells = BTreeSet::new();
    for (index, item) in array(value, "assignments", "plan")?.iter().enumerate() {
        let context = format!("assignments[{index}]");
        let assignment = Assignment {
            run_id: string(item, "run_id", &context)?.to_owned(),
            target_id: string(item, "target_id", &context)?.to_owned(),
            component_id: string(item, "component_id", &context)?.to_owned(),
            arm_id: string(item, "arm_id", &context)?.to_owned(),
            budget_id: string(item, "budget_id", &context)?.to_owned(),
            input_sha256: string(item, "input_sha256", &context)?.to_owned(),
        };
        validate_sha(&assignment.input_sha256, "assignment input")?;
        if !run_ids.insert(assignment.run_id.clone()) {
            bail!("duplicate assignment run_id {:?}", assignment.run_id);
        }
        let cell = (
            assignment.target_id.clone(),
            assignment.arm_id.clone(),
            assignment.budget_id.clone(),
        );
        if !cells.insert(cell) {
            bail!("duplicate target/arm/budget assignment");
        }
        let target = targets
            .iter()
            .find(|target| target.id == assignment.target_id)
            .with_context(|| format!("assignment references unknown target {:?}", assignment.target_id))?;
        if target.component_id != assignment.component_id {
            bail!("assignment component differs from frozen target component");
        }
        if !arm_ids.contains(&assignment.arm_id) || !budget_ids.contains(&assignment.budget_id) {
            bail!("assignment references an unknown arm or budget");
        }
        assignments.push(assignment);
    }
    let expected_cells = targets.len() * arms.len() * budgets.len();
    if cells.len() != expected_cells || assignments.len() != expected_cells {
        bail!(
            "incomplete Cartesian plan: expected {expected_cells} target/arm/budget cells, got {}",
            cells.len()
        );
    }

    Ok(Plan {
        id,
        dataset_kind,
        split,
        targets,
        arms,
        budgets,
        assignments,
    })
}

fn parse_and_join_outcomes(plan: &Plan, bytes: &[u8]) -> Result<Vec<Outcome>> {
    let text = std::str::from_utf8(bytes).context("outcomes JSONL must be UTF-8")?;
    let assignment_by_run: BTreeMap<&str, &Assignment> = plan
        .assignments
        .iter()
        .map(|assignment| (assignment.run_id.as_str(), assignment))
        .collect();
    let target_by_id: BTreeMap<&str, &Target> = plan
        .targets
        .iter()
        .map(|target| (target.id.as_str(), target))
        .collect();
    let arm_by_id: BTreeMap<&str, &Arm> = plan
        .arms
        .iter()
        .map(|arm| (arm.id.as_str(), arm))
        .collect();
    let budget_by_id: BTreeMap<&str, &Budget> = plan
        .budgets
        .iter()
        .map(|budget| (budget.id.as_str(), budget))
        .collect();

    let mut outcomes = Vec::new();
    let mut seen_runs = BTreeSet::new();
    let mut seen_cells = BTreeSet::new();
    for (line_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            bail!("outcomes JSONL has empty line {}", line_index + 1);
        }
        let context = format!("outcomes line {}", line_index + 1);
        let value: Value =
            serde_json::from_str(line).with_context(|| format!("parse {context}"))?;
        if string(&value, "schema_version", &context)? != OUTCOME_VERSION {
            bail!("{context}: unsupported schema_version");
        }
        if string(&value, "plan_id", &context)? != plan.id {
            bail!("{context}: plan_id does not match the frozen plan");
        }
        let run_id = string(&value, "run_id", &context)?.to_owned();
        if !seen_runs.insert(run_id.clone()) {
            bail!("{context}: duplicate run_id {run_id:?}");
        }
        let assignment = assignment_by_run
            .get(run_id.as_str())
            .with_context(|| format!("{context}: unknown run_id {run_id:?}"))?;
        let target_id = string(&value, "target_id", &context)?.to_owned();
        let component_id = string(&value, "component_id", &context)?.to_owned();
        let arm_id = string(&value, "arm_id", &context)?.to_owned();
        let budget_id = string(&value, "budget_id", &context)?.to_owned();
        let input_sha256 = string(&value, "input_sha256", &context)?;
        if target_id != assignment.target_id
            || component_id != assignment.component_id
            || arm_id != assignment.arm_id
            || budget_id != assignment.budget_id
            || input_sha256 != assignment.input_sha256
        {
            bail!("{context}: outcome identity does not exactly match its assigned plan cell");
        }
        let cell = (target_id.clone(), arm_id.clone(), budget_id.clone());
        if !seen_cells.insert(cell) {
            bail!("{context}: duplicate target/arm/budget outcome");
        }
        let target = target_by_id[target_id.as_str()];
        let arm = arm_by_id[arm_id.as_str()];
        let budget = budget_by_id[budget_id.as_str()];
        if string(&value, "model_id", &context)? != arm.model_id
            || string(&value, "model_version", &context)? != arm.model_version
            || string(&value, "checker_id", &context)? != arm.checker_id
            || string(&value, "checker_version", &context)? != arm.checker_version
            || string(&value, "config_sha256", &context)? != arm.config_sha256
        {
            bail!("{context}: model/checker/config identity differs from frozen arm");
        }

        let status = string(&value, "status", &context)?.to_owned();
        if !matches!(
            status.as_str(),
            "verified_success"
                | "proof_only"
                | "failed"
                | "timeout"
                | "not_attempted"
                | "tool_unavailable"
                | "out_of_scope"
        ) {
            bail!("{context}: unsupported status {status:?}");
        }
        let checker_passed = boolean(&value, "checker_passed", &context)?;
        let semantic_gate_passed = boolean(&value, "semantic_gate_passed", &context)?;
        let successor_status = string(&value, "successor_gate_status", &context)?;
        if !matches!(successor_status, "passed" | "not_applicable" | "failed" | "not_run") {
            bail!("{context}: unsupported successor_gate_status {successor_status:?}");
        }
        let checked_statement = optional_string(&value, "checked_target_statement_sha256", &context)?;
        if let Some(digest) = checked_statement {
            validate_sha(digest, "checked target statement")?;
            if digest != target.statement_sha256 {
                bail!("{context}: checked statement digest differs from the exact planned target");
            }
        }
        let calls = unsigned(&value, "model_calls_used", &context)?;
        let tokens = unsigned(&value, "tokens_used", &context)?;
        let wall_ms = unsigned(&value, "wall_ms", &context)?;
        let cost = unsigned(&value, "cost_microusd", &context)?;
        if calls > budget.max_model_calls
            || tokens > budget.max_tokens
            || wall_ms > budget.max_wall_ms
            || cost > budget.max_cost_microusd
        {
            bail!("{context}: observed resource usage exceeds its assigned budget cap");
        }

        let unexecuted = matches!(status.as_str(), "not_attempted" | "tool_unavailable" | "out_of_scope");
        let checker_receipt = optional_string(&value, "checker_receipt_sha256", &context)?;
        let semantic_receipt = optional_string(&value, "semantic_gate_receipt_sha256", &context)?;
        let successor_receipt = optional_string(&value, "successor_gate_receipt_sha256", &context)?;
        let run_receipt = optional_string(&value, "run_receipt_sha256", &context)?;
        for (name, digest) in [
            ("checker receipt", checker_receipt),
            ("semantic-gate receipt", semantic_receipt),
            ("successor receipt", successor_receipt),
            ("run receipt", run_receipt),
        ] {
            if let Some(digest) = digest {
                validate_sha(digest, name)?;
            }
        }
        let receipts_absent = [
            checker_receipt,
            semantic_receipt,
            successor_receipt,
            run_receipt,
        ]
        .iter()
        .all(|item| item.is_none());
        if unexecuted {
            if checker_passed
                || semantic_gate_passed
                || checked_statement.is_some()
                || !receipts_absent
                || calls != 0
                || tokens != 0
                || wall_ms != 0
                || cost != 0
                || successor_status != "not_run"
            {
                bail!("{context}: unexecuted assignment must have zero usage, null check evidence, and no success flags");
            }
        } else {
            if checked_statement.is_none()
                || checker_receipt.is_none()
                || semantic_receipt.is_none()
                || successor_receipt.is_none()
                || run_receipt.is_none()
            {
                bail!("{context}: attempted outcome requires exact statement binding and all four receipts");
            }
        }

        if status == "verified_success" {
            if !checker_passed
                || !semantic_gate_passed
                || !matches!(successor_status, "passed" | "not_applicable")
                || checked_statement != Some(target.statement_sha256.as_str())
            {
                bail!("{context}: verified_success contradicts target/checker/semantic/successor evidence");
            }
        } else if status == "proof_only" {
            if !checker_passed || semantic_gate_passed {
                bail!("{context}: proof_only requires kernel success but semantic-gate failure");
            }
        } else if !unexecuted && (checker_passed || semantic_gate_passed) {
            bail!("{context}: non-success attempted status cannot assert both correctness gates");
        }

        if plan.dataset_kind == "synthetic_fixture" && !unexecuted {
            bail!("{context}: synthetic fixture cannot claim an executed model outcome");
        }

        outcomes.push(Outcome {
            target_id,
            component_id,
            arm_id,
            budget_id,
            status,
            success: is_verified_success(
                string(&value, "status", &context)?,
                checker_passed,
                semantic_gate_passed,
                successor_status,
                checked_statement,
                &target.statement_sha256,
            ),
            model_calls: calls,
            tokens,
            wall_ms,
            cost_microusd: cost,
        });
    }
    if outcomes.len() != plan.assignments.len()
        || seen_runs.len() != plan.assignments.len()
        || seen_cells.len() != plan.assignments.len()
    {
        bail!(
            "outcome coverage mismatch: {} plan assignments but {} unique outcome rows",
            plan.assignments.len(),
            outcomes.len()
        );
    }
    Ok(outcomes)
}

fn analyze(
    plan: &Plan,
    outcomes: &[Outcome],
    plan_bytes: &[u8],
    outcome_bytes: &[u8],
    bootstrap_replicates: usize,
    seed: u64,
) -> Result<Value> {
    let mut statuses = BTreeMap::<String, usize>::new();
    let mut executed_rows = 0usize;
    let mut verified_successes = 0usize;
    let unexecuted_count = outcomes
        .iter()
        .filter(|outcome| matches!(outcome.status.as_str(), "not_attempted" | "tool_unavailable" | "out_of_scope"))
        .count();
    for outcome in outcomes {
        *statuses.entry(outcome.status.clone()).or_default() += 1;
        if !matches!(outcome.status.as_str(), "not_attempted" | "tool_unavailable" | "out_of_scope") {
            executed_rows += 1;
        }
        if outcome.success {
            verified_successes += 1;
        }
    }

    let mut curves = Vec::new();
    let mut paired_effects = Vec::new();
    let mut all_points = Vec::new();
    let mut frontier = Vec::new();

    if plan.dataset_kind != "synthetic_fixture" {
        for budget in &plan.budgets {
            for arm in &plan.arms {
                let selected: Vec<&Outcome> = outcomes
                    .iter()
                    .filter(|outcome| outcome.arm_id == arm.id && outcome.budget_id == budget.id)
                    .collect();
                let successes = selected.iter().filter(|outcome| outcome.success).count();
                let calls: u64 = selected.iter().map(|outcome| outcome.model_calls).sum();
                let tokens: u64 = selected.iter().map(|outcome| outcome.tokens).sum();
                let wall_ms: u64 = selected.iter().map(|outcome| outcome.wall_ms).sum();
                let cost: u64 = selected.iter().map(|outcome| outcome.cost_microusd).sum();
                let executed = selected
                    .iter()
                    .filter(|outcome| !matches!(outcome.status.as_str(), "not_attempted" | "tool_unavailable" | "out_of_scope"))
                    .count();
                let rate = ratio(successes, plan.targets.len());
                let groups = group_rate_counts(&selected);
                let ci = bootstrap_rate(&groups, bootstrap_replicates, stable_seed(seed, &format!("rate:{}:{}", arm.id, budget.id)));
                let curve = json!({
                    "arm_id": arm.id,
                    "method_class": arm.method_class,
                    "budget_id": budget.id,
                    "budget_caps": {
                        "max_model_calls": budget.max_model_calls,
                        "max_tokens": budget.max_tokens,
                        "max_wall_ms": budget.max_wall_ms,
                        "max_cost_microusd": budget.max_cost_microusd
                    },
                    "assigned_targets": plan.targets.len(),
                    "executed_rows": executed,
                    "unexecuted_rows": selected.len().saturating_sub(executed),
                    "verified_successes": successes,
                    "verified_success_rate": rate,
                    "verified_success_rate_ci_95": ci.map(|interval| json!({
                        "method": "component_clustered_percentile_bootstrap",
                        "lower": interval[0],
                        "upper": interval[1]
                    })),
                    "total_model_calls": calls,
                    "total_tokens": tokens,
                    "total_wall_ms": wall_ms,
                    "total_cost_microusd": cost,
                    "mean_cost_microusd_per_assigned_target": (plan.targets.len() > 0).then_some(cost as f64 / plan.targets.len() as f64),
                    "cost_microusd_per_verified_solve": (successes > 0).then_some(cost as f64 / successes as f64)
                });
                let cost_per_target = cost as f64 / plan.targets.len() as f64;
                all_points.push((
                    format!("{}:{}", arm.id, budget.id),
                    arm.id.clone(),
                    arm.method_class.clone(),
                    budget.id.clone(),
                    rate,
                    cost_per_target,
                    executed,
                ));
                curves.push(curve);
            }
        }

        // Paired target-level differences; resampling occurs at the connected-component unit.
        for budget in &plan.budgets {
            for left_index in 0..plan.arms.len() {
                for right_index in (left_index + 1)..plan.arms.len() {
                    let left = &plan.arms[left_index];
                    let right = &plan.arms[right_index];
                    let mut left_successes = 0usize;
                    let mut right_successes = 0usize;
                    let mut cluster_deltas = BTreeMap::<String, (i64, usize)>::new();
                    for target in &plan.targets {
                        let left_observation = outcomes.iter().find(|outcome| {
                            outcome.target_id == target.id
                                && outcome.arm_id == left.id
                                && outcome.budget_id == budget.id
                        }).context("left arm result disappeared after full-coverage validation")?;
                        let right_observation = outcomes.iter().find(|outcome| {
                            outcome.target_id == target.id
                                && outcome.arm_id == right.id
                                && outcome.budget_id == budget.id
                        }).context("right arm result disappeared after full-coverage validation")?;
                        left_successes += usize::from(left_observation.success);
                        right_successes += usize::from(right_observation.success);
                        let entry = cluster_deltas.entry(target.component_id.clone()).or_default();
                        entry.0 += i64::from(left_observation.success) - i64::from(right_observation.success);
                        entry.1 += 1;
                    }
                    let difference = (left_successes as f64 - right_successes as f64) / plan.targets.len() as f64;
                    let interval = bootstrap_paired_delta(
                        &cluster_deltas,
                        bootstrap_replicates,
                        stable_seed(seed, &format!("paired:{}:{}:{}", left.id, right.id, budget.id)),
                    );
                    paired_effects.push(json!({
                        "budget_id": budget.id,
                        "treatment_arm_id": left.id,
                        "comparator_arm_id": right.id,
                        "matched_targets": plan.targets.len(),
                        "component_count": cluster_deltas.len(),
                        "treatment_successes": left_successes,
                        "comparator_successes": right_successes,
                        "success_rate_difference_treatment_minus_comparator": difference,
                        "confidence_interval_95": interval.map(|ci| json!({
                            "method": "component_clustered_percentile_bootstrap",
                            "lower": ci[0],
                            "upper": ci[1]
                        }))
                    }));
                }
            }
        }

        for candidate in &all_points {
            if candidate.6 == 0 {
                continue;
            }
            let dominated_by: Vec<String> = all_points
                .iter()
                .filter(|other| other.0 != candidate.0 && other.6 > 0)
                .filter(|other| dominates(other.4, other.5, candidate.4, candidate.5))
                .map(|other| other.0.clone())
                .collect();
            if dominated_by.is_empty() {
                frontier.push(json!({
                    "point_id": candidate.0,
                    "arm_id": candidate.1,
                    "method_class": candidate.2,
                    "budget_id": candidate.3,
                    "verified_success_rate": candidate.4,
                    "mean_cost_microusd_per_assigned_target": candidate.5
                }));
            }
        }
    }

    let decision = if plan.dataset_kind == "synthetic_fixture" {
        "not_evaluated_synthetic_fixture"
    } else {
        "not_evaluated_confirmatory_gate_requires_primary_analyzer_and_independent_receipts"
    };
    Ok(json!({
        "schema_version": REPORT_VERSION,
        "plan_id": plan.id,
        "dataset_kind": plan.dataset_kind,
        "split": plan.split,
        "plan_sha256": sha256(plan_bytes),
        "outcomes_sha256": sha256(outcome_bytes),
        "analyzer_source_sha256": sha256(include_str!("analyze_success_cost_frontier.rs").as_bytes()),
        "bootstrap": {
            "unit": "target_component_id",
            "replicates": bootstrap_replicates,
            "seed": seed,
            "interval": "percentile_95",
            "note": "Component clusters, not repeated budget-tier rows, are resampled. The frontier is descriptive and never clears the primary confirmatory gate."
        },
        "counts": {
            "planned_cells": plan.assignments.len(),
            "outcome_rows": outcomes.len(),
            "executed_rows": executed_rows,
            "unexecuted_rows": unexecuted_count,
            "verified_successes": verified_successes,
            "outcome_status_counts": statuses
        },
        "curves": curves,
        "paired_effects": paired_effects,
        "pareto_frontier": frontier,
        "limitations": [
            "This analyzer validates structure, joins, identity, budgets, and outcome consistency; it does not inspect proof/checker receipt contents or independently adjudicate mathematical statement correspondence.",
            "A verified success requires exact planned statement identity, kernel/checker success, semantic-gate success, and a passed or independently documented not-applicable successor gate.",
            "The synthetic fixture intentionally reports no curves, paired effects, or frontier points and cannot yield model-performance metrics.",
            "A curve is target-weighted over all assigned targets, including not-attempted and unavailable rows as non-solves. Cost per assigned target uses actual recorded usage, not the nominal budget cap.",
            "Pareto points are computed from observed success rate and mean actual cost per assigned target; no interpolation between unmeasured budget caps is performed.",
            "The confirmatory decision is always unassessed here; use the pre-registered primary analyzer and independent receipts for any utility claim."
        ],
        "confirmatory_decision": decision
    }))
}

fn is_verified_success(
    status: &str,
    checker_passed: bool,
    semantic_gate_passed: bool,
    successor_status: &str,
    checked_statement: Option<&str>,
    target_statement: &str,
) -> bool {
    status == "verified_success"
        && checker_passed
        && semantic_gate_passed
        && matches!(successor_status, "passed" | "not_applicable")
        && checked_statement == Some(target_statement)
}

fn group_rate_counts(selected: &[&Outcome]) -> BTreeMap<String, (u64, u64)> {
    let mut groups = BTreeMap::<String, (u64, u64)>::new();
    for outcome in selected {
        let entry = groups.entry(outcome.component_id.clone()).or_default();
        entry.0 += u64::from(outcome.success);
        entry.1 += 1;
    }
    groups
}

fn bootstrap_rate(
    groups: &BTreeMap<String, (u64, u64)>,
    replicates: usize,
    seed: u64,
) -> Option<[f64; 2]> {
    if groups.len() < 2 || replicates == 0 {
        return None;
    }
    let values: Vec<(u64, u64)> = groups.values().copied().collect();
    let mut rng = SplitMix64::new(seed);
    let mut estimates = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut success = 0u64;
        let mut total = 0u64;
        for _ in 0..values.len() {
            let (cluster_success, cluster_total) = values[rng.next() as usize % values.len()];
            success = success.saturating_add(cluster_success);
            total = total.saturating_add(cluster_total);
        }
        if total > 0 {
            estimates.push(success as f64 / total as f64);
        }
    }
    percentile_interval(estimates)
}

fn bootstrap_paired_delta(
    groups: &BTreeMap<String, (i64, usize)>,
    replicates: usize,
    seed: u64,
) -> Option<[f64; 2]> {
    if groups.len() < 2 || replicates == 0 {
        return None;
    }
    let values: Vec<(i64, usize)> = groups.values().copied().collect();
    let mut rng = SplitMix64::new(seed);
    let mut estimates = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut delta_sum = 0i64;
        let mut target_count = 0usize;
        for _ in 0..values.len() {
            let (cluster_delta, cluster_targets) = values[rng.next() as usize % values.len()];
            delta_sum += cluster_delta;
            target_count += cluster_targets;
        }
        if target_count > 0 {
            estimates.push(delta_sum as f64 / target_count as f64);
        }
    }
    percentile_interval(estimates)
}

fn percentile_interval(mut samples: Vec<f64>) -> Option<[f64; 2]> {
    if samples.is_empty() {
        return None;
    }
    samples.sort_by(f64::total_cmp);
    Some([
        percentile(&samples, 0.025)?,
        percentile(&samples, 0.975)?,
    ])
}

fn percentile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() || !(0.0..=1.0).contains(&p) {
        return None;
    }
    let position = p * (sorted.len() - 1) as f64;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    let weight = position - low as f64;
    Some(sorted[low] * (1.0 - weight) + sorted[high] * weight)
}

fn dominates(rate_a: f64, cost_a: f64, rate_b: f64, cost_b: f64) -> bool {
    rate_a >= rate_b && cost_a <= cost_b && (rate_a > rate_b || cost_a < cost_b)
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
}

fn stable_seed(base: u64, label: &str) -> u64 {
    let mut value = 0xcbf29ce484222325u64 ^ base;
    for byte in label.as_bytes() {
        value ^= u64::from(*byte);
        value = value.wrapping_mul(0x100000001b3);
    }
    value
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }
}

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn validate_commit(value: &str, context: &str) -> Result<()> {
    if value.len() != 40
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{context}: expected 40 lowercase hexadecimal characters");
    }
    Ok(())
}

fn validate_sha(value: &str, context: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{context}: expected 64 lowercase hexadecimal characters");
    }
    Ok(())
}

fn string<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|item| !item.is_empty())
        .with_context(|| format!("{context}.{key} must be a non-empty string"))
}

fn optional_string<'a>(value: &'a Value, key: &str, context: &str) -> Result<Option<&'a str>> {
    match value.get(key) {
        Some(Value::String(value)) if !value.is_empty() => Ok(Some(value.as_str())),
        Some(Value::Null) => Ok(None),
        _ => bail!("{context}.{key} must be a non-empty string or null"),
    }
}

fn boolean(value: &Value, key: &str, context: &str) -> Result<bool> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .with_context(|| format!("{context}.{key} must be a boolean"))
}

fn unsigned(value: &Value, key: &str, context: &str) -> Result<u64> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .with_context(|| format!("{context}.{key} must be a non-negative integer"))
}

fn positive_u64(value: &Value, key: &str, context: &str) -> Result<u64> {
    let result = unsigned(value, key, context)?;
    if result == 0 {
        bail!("{context}.{key} must be positive");
    }
    Ok(result)
}

fn array<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a Vec<Value>> {
    value
        .get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("{context}.{key} must be an array"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_target_success_requires_both_gates_and_matching_digest() {
        assert!(is_verified_success(
            "verified_success",
            true,
            true,
            "passed",
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ));
        assert!(!is_verified_success(
            "verified_success",
            true,
            true,
            "passed",
            Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ));
        assert!(!is_verified_success(
            "verified_success",
            true,
            false,
            "passed",
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ));
        assert!(!is_verified_success(
            "verified_success",
            true,
            true,
            "failed",
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ));
    }

    #[test]
    fn pareto_dominance_requires_strict_improvement() {
        assert!(dominates(0.8, 10.0, 0.7, 10.0));
        assert!(dominates(0.8, 9.0, 0.8, 10.0));
        assert!(!dominates(0.8, 10.0, 0.8, 10.0));
        assert!(!dominates(0.7, 9.0, 0.8, 10.0));
    }

    #[test]
    fn component_bootstrap_is_deterministic_and_needs_multiple_clusters() {
        let one = BTreeMap::from([("only".to_owned(), (1u64, 1u64))]);
        assert_eq!(bootstrap_rate(&one, 1_000, 7), None);
        let groups = BTreeMap::from([
            ("a".to_owned(), (2u64, 2u64)),
            ("b".to_owned(), (0u64, 2u64)),
            ("c".to_owned(), (1u64, 2u64)),
        ]);
        assert_eq!(
            bootstrap_rate(&groups, 1_000, 123),
            bootstrap_rate(&groups, 1_000, 123)
        );
        assert!(bootstrap_rate(&groups, 1_000, 123).is_some());
    }

    #[test]
    fn paired_bootstrap_is_deterministic_and_target_weighted() {
        let groups = BTreeMap::from([
            ("a".to_owned(), (1i64, 1usize)),
            ("b".to_owned(), (-1i64, 1usize)),
            ("c".to_owned(), (0i64, 2usize)),
        ]);
        assert_eq!(
            bootstrap_paired_delta(&groups, 1_000, 99),
            bootstrap_paired_delta(&groups, 1_000, 99)
        );
        assert!(bootstrap_paired_delta(&groups, 1_000, 99).is_some());
    }

    #[test]
    fn percentile_interpolates_and_handles_boundaries() {
        assert_eq!(percentile(&[0.0, 10.0, 20.0], 0.0), Some(0.0));
        assert_eq!(percentile(&[0.0, 10.0, 20.0], 0.5), Some(10.0));
        assert_eq!(percentile(&[0.0, 10.0, 20.0], 1.0), Some(20.0));
        assert_eq!(percentile(&[], 0.5), None);
    }
}
