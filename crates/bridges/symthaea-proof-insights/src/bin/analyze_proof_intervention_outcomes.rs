// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Analyze paired proof-intervention outcomes under a frozen multi-arm plan.
//!
//! The analyzer joins every pair x arm, treats non-attempts as failures, checks
//! exact plan identities and budget caps, authenticates measured artifact bytes
//! beneath a caller-supplied evidence root, and reports paired ITT contrasts.
//! Receipt-specific semantic parsing remains a separate verifier boundary.

#[path = "../proof_intervention_trial.rs"]
mod proof_intervention_trial;
#[path = "../benchmark_integrity.rs"]
mod benchmark_integrity;
#[path = "../proof_intervention_evidence.rs"]
mod proof_intervention_evidence;

use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::{Component, Path, PathBuf},
    process::ExitCode,
};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const OUTCOME_VERSION: &str = "pitb.proof-intervention-trial-outcome.v0";
const BOOTSTRAP_REPLICATES: usize = 10_000;
const RESOURCE_FIELDS: [&str; 6] = [
    "tokens",
    "model_calls",
    "tool_calls",
    "wall_ms",
    "lean_cpu_ms",
    "cost_microusd",
];

fn field<'a>(value: &'a Value, pointer: &str) -> Result<&'a Value> {
    value
        .pointer(pointer)
        .with_context(|| format!("missing field {pointer}"))
}

fn string<'a>(value: &'a Value, pointer: &str) -> Result<&'a str> {
    field(value, pointer)?
        .as_str()
        .with_context(|| format!("{pointer} must be a string"))
}

fn integer(value: &Value, pointer: &str) -> Result<u64> {
    field(value, pointer)?
        .as_u64()
        .with_context(|| format!("{pointer} must be a nonnegative integer"))
}

fn bool_value(value: &Value, pointer: &str) -> Result<bool> {
    field(value, pointer)?
        .as_bool()
        .with_context(|| format!("{pointer} must be boolean"))
}

fn digest(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{label} must be a 64-character lowercase SHA-256 digest");
    }
    Ok(())
}

fn optional_digest(value: &Value, pointer: &str) -> Result<Option<String>> {
    match field(value, pointer)? {
        Value::Null => Ok(None),
        Value::String(value) => {
            digest(value, pointer)?;
            Ok(Some(value.clone()))
        }
        _ => bail!("{pointer} must be a SHA-256 digest or null"),
    }
}

fn safe_artifact_path(root: &Path, relative: &str) -> Result<PathBuf> {
    let path = Path::new(relative);
    if relative.trim().is_empty()
        || path.is_absolute()
        || relative.contains('\\')
        || relative.contains('\0')
        || path.components().any(|part| {
            matches!(
                part,
                Component::ParentDir
                    | Component::CurDir
                    | Component::RootDir
                    | Component::Prefix(_)
            )
        })
    {
        bail!("receipt artifact path must be safe and relative to the evidence root");
    }
    let root = fs::canonicalize(root)
        .with_context(|| format!("canonicalize evidence root {}", root.display()))?;
    let resolved = fs::canonicalize(root.join(path))
        .with_context(|| format!("resolve receipt artifact {relative}"))?;
    if !resolved.starts_with(&root) || !resolved.is_file() {
        bail!("receipt artifact {relative} escapes the evidence root or is not a regular file");
    }
    Ok(resolved)
}

fn authenticate_artifact(
    root: Option<&Path>,
    evidence_class: &str,
    path: Option<&str>,
    expected_sha256: Option<&str>,
    label: &str,
) -> Result<bool> {
    if evidence_class == "synthetic_outcome_fixture" {
        if path.is_some() {
            bail!("synthetic outcome fixtures cannot claim measured evidence artifact files");
        }
        return Ok(false);
    }
    match (path, expected_sha256) {
        (Some(_), None) | (None, Some(_)) => {
            bail!("{label} path and digest must either both be present or both be null")
        }
        (None, None) => bail!("{label} is required for measured outcomes"),
        (Some(relative), Some(expected)) => {
            digest(expected, label)?;
            if evidence_class == "synthetic_outcome_fixture" {
                bail!("synthetic outcome fixtures cannot claim measured evidence artifact files");
            }
            let root = root.context("measured outcomes require an evidence root")?;
            let resolved = safe_artifact_path(root, relative)?;
            let bytes = fs::read(&resolved)
                .with_context(|| format!("read {label} {}", resolved.display()))?;
            let actual = hex::encode(Sha256::digest(&bytes));
            if actual != expected {
                bail!("{label} bytes do not match the outcome receipt SHA-256");
            }
            Ok(true)
        }
    }
}

fn read_verified_artifact(
    root: &Path,
    relative: &str,
    expected_sha256: &str,
    label: &str,
) -> Result<Vec<u8>> {
    digest(expected_sha256, label)?;
    let resolved = safe_artifact_path(root, relative)?;
    let bytes = fs::read(&resolved)
        .with_context(|| format!("read {label} {}", resolved.display()))?;
    let actual = hex::encode(Sha256::digest(&bytes));
    if actual != expected_sha256 {
        bail!("{label} bytes do not match the paired outcome SHA-256");
    }
    Ok(bytes)
}

fn resource_values(value: &Value, prefix: &str) -> Result<Vec<u64>> {
    RESOURCE_FIELDS
        .iter()
        .map(|name| integer(value, &format!("{prefix}/{name}")))
        .collect()
}

fn outcome_success(value: &Value, target_statement: &str) -> Result<bool> {
    let status = string(value, "/status")?;
    let checker = bool_value(value, "/checker_passed")?;
    let target_integrity_passed = bool_value(value, "/target_integrity_passed")?;
    let successor = string(value, "/successor_gate_status")?;
    let checked_statement = match field(value, "/checked_target_statement_sha256")? {
        Value::Null => None,
        Value::String(value) => Some(value.as_str()),
        _ => bail!("checked_target_statement_sha256 must be a digest or null"),
    };

    let proof = optional_digest(value, "/proof_sha256")?;
    let checker_receipt = optional_digest(value, "/checker_receipt_sha256")?;
    let target_integrity_receipt = optional_digest(value, "/target_integrity_receipt_sha256")?;
    let successor_receipt = optional_digest(value, "/successor_gate_receipt_sha256")?;

    if !matches!(
        status,
        "verified_success"
            | "proof_only"
            | "failed"
            | "timeout"
            | "not_attempted"
            | "tool_unavailable"
            | "invalid_candidate"
    ) {
        bail!("unknown intervention outcome status {status}");
    }
    if !matches!(successor, "passed" | "not_applicable" | "failed" | "not_run") {
        bail!("unknown successor gate status {successor}");
    }

    if status == "not_attempted" {
        if checker
            || !target_integrity_passed
            || successor != "not_run"
            || checked_statement.is_some()
            || proof.is_some()
            || checker_receipt.is_some()
            || successor_receipt.is_some()
        {
            bail!("not_attempted outcome contains proof/checker/semantic evidence");
        }
        return Ok(false);
    }

    if status == "verified_success" {
        if !checker
            || !target_integrity_passed
            || !matches!(successor, "passed" | "not_applicable")
            || checked_statement != Some(target_statement)
            || proof.is_none()
            || checker_receipt.is_none()
            || target_integrity_receipt.is_none()
            || (successor == "passed" && successor_receipt.is_none())
            || (successor == "not_applicable" && successor_receipt.is_some())
        {
            bail!("verified_success status is inconsistent with exact-target verification receipts");
        }
        return Ok(true);
    }

    Ok(false)
}

#[derive(Debug, Clone)]
struct PairOutcome {
    component_id: String,
    successes: BTreeMap<String, bool>,
    costs_microusd: BTreeMap<String, u64>,
}

fn validate_and_analyze(
    plan: &Value,
    outcomes: &[Value],
    evidence_root: Option<&Path>,
    reviewer_registry_path: Option<&Path>,
    expected_registry_sha256: Option<&str>,
) -> Result<Value> {
    let expected_pairs = proof_intervention_trial::validate_plan(plan)?;
    let plan_evidence = string(plan, "/evidence_class")?;
    let expected_outcome_evidence = if plan_evidence == "synthetic_contract_fixture" {
        "synthetic_outcome_fixture"
    } else {
        "measured_outcome"
    };
    if plan_evidence == "preregistered_trial"
        && (evidence_root.is_none()
            || reviewer_registry_path.is_none()
            || expected_registry_sha256.is_none())
    {
        bail!("measured intervention analysis requires an evidence root, reviewer-key registry, and externally pinned registry SHA-256");
    }
    let trial_id = string(plan, "/trial_id")?;
    let horizon = string(plan, "/evaluation/horizon_id")?;
    let continuation = string(plan, "/evaluation/continuation_policy_sha256")?;
    let plan_seed = string(plan, "/randomization/seed_sha256")?;

    let arms = field(plan, "/arms")?
        .as_array()
        .context("plan arms must be an array")?;
    let mut arm_ids = Vec::new();
    let mut arm_roles = BTreeMap::<String, String>::new();
    let mut arm_successes = BTreeMap::<String, u64>::new();
    let mut arm_costs = BTreeMap::<String, u128>::new();
    for arm in arms {
        let arm_id = string(arm, "/arm_id")?.to_owned();
        let role = string(arm, "/role")?.to_owned();
        arm_ids.push(arm_id.clone());
        let _ = arm_roles.insert(arm_id.clone(), role);
        let _ = arm_successes.insert(arm_id.clone(), 0);
        let _ = arm_costs.insert(arm_id, 0);
    }
    let control_id = arm_roles
        .iter()
        .find(|(_, role)| role.as_str() == "control")
        .map(|(id, _)| id.clone())
        .context("plan has no unique control arm")?;

    let pairs = field(plan, "/pairs")?
        .as_array()
        .context("plan pairs must be an array")?;
    if outcomes.len() != expected_pairs * arm_ids.len() {
        bail!(
            "outcome row count {} does not equal complete pair x arm coverage {}",
            outcomes.len(),
            expected_pairs * arm_ids.len()
        );
    }

    let mut pair_by_id = BTreeMap::<String, &Value>::new();
    for pair in pairs {
        let _ = pair_by_id.insert(string(pair, "/pair_id")?.to_owned(), pair);
    }
    let mut seen = BTreeSet::<(String, String)>::new();
    let mut run_ids = BTreeSet::<String>::new();
    let mut pair_rows = BTreeMap::<String, BTreeMap<String, (bool, u64)>>::new();
    let mut measured_artifacts_authenticated = true;
    let mut signed_target_integrity_receipts_verified = 0_u64;

    for outcome in outcomes {
        if string(outcome, "/schema_version")? != OUTCOME_VERSION {
            bail!("unsupported intervention outcome schema version");
        }
        let evidence_class = string(outcome, "/evidence_class")?;
        if evidence_class != expected_outcome_evidence {
            bail!("outcome evidence class does not match the preregistered plan");
        }
        if string(outcome, "/trial_id")? != trial_id {
            bail!("outcome trial_id differs from plan");
        }
        let pair_id = string(outcome, "/pair_id")?.to_owned();
        let arm_id = string(outcome, "/arm_id")?.to_owned();
        let run_id = string(outcome, "/run_id")?.to_owned();
        if run_id.trim().is_empty() || !run_ids.insert(run_id.clone()) {
            bail!("run IDs must be nonempty and unique");
        }
        if !arm_roles.contains_key(&arm_id) {
            bail!("outcome references unregistered arm {arm_id}");
        }
        if !seen.insert((pair_id.clone(), arm_id.clone())) {
            bail!("duplicate outcome row for pair {pair_id} arm {arm_id}");
        }
        let pair = pair_by_id
            .get(&pair_id)
            .with_context(|| format!("outcome references unknown pair {pair_id}"))?;

        for (outcome_pointer, pair_pointer, label) in [
            ("/task_id", "/task_id", "task identity"),
            ("/component_id", "/component_id", "component identity"),
            ("/repository_commit", "/repository_commit", "repository commit"),
            ("/lean_toolchain", "/lean_toolchain", "Lean toolchain"),
            ("/environment_sha256", "/environment_sha256", "environment"),
            ("/initial_proof_state_sha256", "/initial_proof_state_sha256", "initial proof state"),
            ("/initial_state_receipt_sha256", "/initial_state_receipt_sha256", "initial state receipt"),
            ("/target_statement_sha256", "/target_statement_sha256", "target statement"),
            ("/model_seed_sha256", "/model_seed_sha256", "model seed"),
        ] {
            if string(outcome, outcome_pointer)? != string(pair, pair_pointer)? {
                bail!("outcome {run_id} has {label} drift from its paired assignment");
            }
        }
        if string(outcome, "/evaluation_horizon_id")? != horizon
            || string(outcome, "/continuation_policy_sha256")? != continuation
        {
            bail!("outcome {run_id} has horizon or continuation-policy drift");
        }

        let order = field(pair, "/execution_order")?
            .as_array()
            .context("pair execution_order must be an array")?;
        let expected_index = order
            .iter()
            .position(|item| item.as_str() == Some(&arm_id))
            .context("outcome arm missing from pair execution order")?;
        if integer(outcome, "/execution_order_index")? as usize != expected_index {
            bail!("outcome {run_id} execution_order_index differs from frozen randomized order");
        }

        let actual = resource_values(outcome, "/actual_resources")?;
        let budget = resource_values(pair, "/resource_budget")?;
        for (index, (used, cap)) in actual.iter().zip(budget.iter()).enumerate() {
            if used > cap {
                bail!(
                    "outcome {run_id} exceeded {} budget: actual={} cap={}",
                    RESOURCE_FIELDS[index],
                    used,
                    cap
                );
            }
        }

        let status = string(outcome, "/status")?;
        let successful = outcome_success(outcome, string(pair, "/target_statement_sha256")?)?;
        if status == "not_attempted" && actual.iter().any(|amount| *amount != 0) {
            bail!("not_attempted outcome {run_id} must have zero actual resources");
        }

        let receipt_specs = [
            ("/proof_path", "/proof_sha256", "proof artifact", status == "verified_success"),
            ("/checker_receipt_path", "/checker_receipt_sha256", "checker receipt", status == "verified_success"),
            ("/target_integrity_receipt_path", "/target_integrity_receipt_sha256", "target integrity receipt", true),
            (
                "/successor_gate_receipt_path",
                "/successor_gate_receipt_sha256",
                "successor gate receipt",
                status == "verified_success"
                    && string(outcome, "/successor_gate_status")? == "passed",
            ),
            ("/run_receipt_path", "/run_receipt_sha256", "run receipt", true),
        ];
        for (path_pointer, digest_pointer, label, required) in receipt_specs {
            let path = match field(outcome, path_pointer)? {
                Value::Null => None,
                Value::String(value) => Some(value.as_str()),
                _ => bail!("{path_pointer} must be a relative path or null"),
            };
            let expected = match field(outcome, digest_pointer)? {
                Value::Null => None,
                Value::String(value) => Some(value.as_str()),
                _ => bail!("{digest_pointer} must be a SHA-256 digest or null"),
            };
            if required && (path.is_none() || expected.is_none()) {
                bail!("outcome {run_id} is missing required {label} path/digest");
            }
            let authenticated =
                authenticate_artifact(evidence_root, evidence_class, path, expected, label)?;
            if evidence_class == "measured_outcome" && !authenticated {
                measured_artifacts_authenticated = false;
            }
        }

        if evidence_class == "measured_outcome" {
            let root = evidence_root.context("measured evidence requires a canonical evidence root")?;
            let expected_identity = proof_intervention_evidence::ExpectedIdentity {
                trial_id,
                pair_id: &pair_id,
                arm_id: &arm_id,
                run_id: &run_id,
                task_id: string(outcome, "/task_id")?,
                component_id: string(outcome, "/component_id")?,
                repository_commit: string(outcome, "/repository_commit")?,
                lean_toolchain: string(outcome, "/lean_toolchain")?,
                environment_sha256: string(outcome, "/environment_sha256")?,
                initial_proof_state_sha256: string(outcome, "/initial_proof_state_sha256")?,
                target_statement_sha256: string(outcome, "/target_statement_sha256")?,
                model_seed_sha256: string(outcome, "/model_seed_sha256")?,
                evaluation_horizon_id: horizon,
                continuation_policy_sha256: continuation,
                execution_order_index: expected_index as u64,
            };

            let checker_path = match field(outcome, "/checker_receipt_path")? {
                Value::Null => None,
                Value::String(value) => Some(value.as_str()),
                _ => bail!("checker_receipt_path must be a relative path or null"),
            };
            let checker_digest = optional_digest(outcome, "/checker_receipt_sha256")?;
            match (checker_path, checker_digest.as_deref()) {
                (Some(relative), Some(expected_sha)) => {
                    let bytes = read_verified_artifact(root, relative, expected_sha, "checker receipt")?;
                    let proof_sha = optional_digest(outcome, "/proof_sha256")?;
                    let checked_target = match field(outcome, "/checked_target_statement_sha256")? {
                        Value::Null => None,
                        Value::String(value) => Some(value.as_str()),
                        _ => bail!("checked target statement must be a digest or null"),
                    };
                    proof_intervention_evidence::verify_checker_receipt(
                        &bytes,
                        expected_sha,
                        expected_identity,
                        status,
                        proof_sha.as_deref(),
                        checked_target,
                    )?;
                }
                (None, None) if status != "verified_success" => {}
                _ => bail!("checker receipt path/digest is incomplete or missing for a verified success"),
            }

            let run_relative = match field(outcome, "/run_receipt_path")? {
                Value::String(value) => value.as_str(),
                _ => bail!("measured outcome requires a run_receipt_path"),
            };
            let run_digest = string(outcome, "/run_receipt_sha256")?;
            let run_bytes = read_verified_artifact(root, run_relative, run_digest, "run receipt")?;
            proof_intervention_evidence::verify_run_receipt(
                &run_bytes,
                run_digest,
                expected_identity,
                outcome,
                field(pair, "/resource_budget")?,
            )?;

            let successor_status = string(outcome, "/successor_gate_status")?;
            if successor_status == "passed" {
                let relative = match field(outcome, "/successor_gate_receipt_path")? {
                    Value::String(value) => value.as_str(),
                    _ => bail!("passed successor gate requires a successor receipt path"),
                };
                let expected = string(outcome, "/successor_gate_receipt_sha256")?;
                let proof_sha = string(outcome, "/proof_sha256")?;
                let bytes = read_verified_artifact(root, relative, expected, "successor receipt")?;
                proof_intervention_evidence::verify_successor_receipt(
                    &bytes,
                    expected,
                    expected_identity,
                    proof_sha,
                )?;
            }
        }

        if evidence_class == "measured_outcome" {
            let root = evidence_root.context("measured semantic receipt requires evidence root")?;
            let target_integrity_path = match field(outcome, "/target_integrity_receipt_path")? {
                Value::String(value) => value,
                _ => bail!("verified success requires a target integrity receipt path"),
            };
            let target_integrity_digest = string(outcome, "/target_integrity_receipt_sha256")?;
            let resolved_target_integrity_path = safe_artifact_path(root, target_integrity_path)?;
            let registry_path = reviewer_registry_path
                .context("measured semantic receipt requires reviewer-key registry")?;
            let registry_digest = expected_registry_sha256
                .context("measured semantic receipt requires externally pinned registry SHA-256")?;
            let integrity = benchmark_integrity::verify_file_with_evidence_root_and_reviewer_registry(
                &resolved_target_integrity_path,
                Some(target_integrity_digest),
                root,
                registry_path,
                registry_digest,
            )?;
            if integrity.integrity_status != "eligible"
                || !integrity.reviewer_signature_verified
                || integrity.verified_evidence_artifact_count != 11
                || integrity.task_id != string(outcome, "/task_id")?
                || integrity.statement_sha256 != string(outcome, "/target_statement_sha256")?
                || integrity.repository_commit != string(outcome, "/repository_commit")?
                || integrity.lean_toolchain != string(outcome, "/lean_toolchain")?
                || integrity.cohort_id != string(plan, "/dataset/cohort_id")?
                || integrity.snapshot_id != string(plan, "/dataset/snapshot_id")?
            {
                bail!("signed benchmark-integrity receipt does not bind this exact paired proof outcome");
            }

            signed_target_integrity_receipts_verified = signed_target_integrity_receipts_verified.saturating_add(1);
        }

        let run_digest = string(outcome, "/run_receipt_sha256")?;
        digest(run_digest, "run_receipt_sha256")?;

        *arm_successes.get_mut(&arm_id).context("arm count missing")? +=
            if successful { 1 } else { 0 };
        let cost = integer(outcome, "/actual_resources/cost_microusd")?;
        *arm_costs.get_mut(&arm_id).context("arm cost missing")? += u128::from(cost);

        let _ = pair_rows
            .entry(pair_id)
            .or_default()
            .insert(arm_id, (successful, cost));
    }

    if seen.len() != expected_pairs * arm_ids.len() {
        bail!("incomplete pair x arm outcome matrix");
    }
    let mut paired_outcomes = Vec::with_capacity(pairs.len());
    let mut component_set = BTreeSet::new();
    for pair in pairs {
        let pair_id = string(pair, "/pair_id")?.to_owned();
        let component_id = string(pair, "/component_id")?.to_owned();
        component_set.insert(component_id.clone());
        let rows = pair_rows
            .get(&pair_id)
            .with_context(|| format!("missing outcomes for pair {pair_id}"))?;
        if rows.len() != arm_ids.len() {
            bail!("pair {pair_id} does not have complete arm coverage");
        }
        let mut successes = BTreeMap::new();
        let mut costs = BTreeMap::new();
        for arm_id in &arm_ids {
            let (success, cost) = rows
                .get(arm_id)
                .with_context(|| format!("pair {pair_id} missing arm {arm_id}"))?;
            let _ = successes.insert(arm_id.clone(), *success);
            let _ = costs.insert(arm_id.clone(), *cost);
        }
        paired_outcomes.push(PairOutcome {
            component_id,
            successes,
            costs_microusd: costs,
        });
    }

    if component_set.len() < 4 {
        bail!("paired analysis requires at least four independent components");
    }
    let n = expected_pairs as u64;
    let mut arm_summaries = Vec::new();
    for arm_id in &arm_ids {
        let success_count = arm_successes[arm_id];
        let cost_total = arm_costs[arm_id];
        arm_summaries.push(json!({
            "arm_id":arm_id,
            "role":arm_roles[arm_id],
            "assigned_pairs":n,
            "verified_successes":success_count,
            "intention_to_treat_success_rate":success_count as f64 / n as f64,
            "total_cost_microusd":cost_total.to_string(),
            "cost_per_verified_success_microusd":if success_count > 0 {
                Some((cost_total / u128::from(success_count)).to_string())
            } else {
                None::<String>
            },
            "non_attempts_and_unavailable_are_failures":true
        }));
    }

    let mut comparisons = Vec::new();
    let bootstrap_seed_bytes = hex::decode(plan_seed)?;
    let bootstrap_seed = u64::from_le_bytes(
        bootstrap_seed_bytes[0..8]
            .try_into()
            .context("randomization seed prefix")?,
    );
    for arm_id in &arm_ids {
        if arm_id == &control_id {
            continue;
        }
        let role = &arm_roles[arm_id];
        let mut wins = 0_u64;
        let mut losses = 0_u64;
        let mut ties = 0_u64;
        let mut difference_sum = 0_i64;
        let mut by_component = BTreeMap::<String, Vec<i64>>::new();
        let mut paired_cost_difference_microusd = 0_i128;
        for pair in &paired_outcomes {
            let treatment = if pair.successes[arm_id] { 1_i64 } else { 0_i64 };
            let control = if pair.successes[&control_id] { 1_i64 } else { 0_i64 };
            let difference = treatment - control;
            difference_sum += difference;
            if difference > 0 {
                wins += 1;
            } else if difference < 0 {
                losses += 1;
            } else {
                ties += 1;
            }
            paired_cost_difference_microusd +=
                i128::from(pair.costs_microusd[arm_id]) - i128::from(pair.costs_microusd[&control_id]);
            by_component
                .entry(pair.component_id.clone())
                .or_default()
                .push(difference);
        }
        let paired_risk_difference = difference_sum as f64 / n as f64;
        let paired_mean_cost_difference_microusd =
            paired_cost_difference_microusd as f64 / n as f64;
        let (lower, upper) = cluster_bootstrap(
            &by_component,
            BOOTSTRAP_REPLICATES,
            bootstrap_seed ^ stable_hash(arm_id),
        );
        let contrast_eligible = role == "intervention";
        comparisons.push(json!({
            "control_arm_id":control_id,
            "comparison_arm_id":arm_id,
            "comparison_role":role,
            "confirmatory_intervention_contrast":contrast_eligible,
            "paired_units":n,
            "treatment_only_successes":wins,
            "control_only_successes":losses,
            "paired_ties":ties,
            "paired_itt_risk_difference":paired_risk_difference,
            "paired_itt_risk_difference_ppm":(paired_risk_difference * 1_000_000.0).round() as i64,
            "paired_mean_cost_difference_microusd":paired_mean_cost_difference_microusd,
            "component_cluster_bootstrap_95_ci_ppm":[
                (lower * 1_000_000.0).round() as i64,
                (upper * 1_000_000.0).round() as i64
            ],
            "bootstrap_replicates":BOOTSTRAP_REPLICATES,
            "resampling_unit":"component_id",
            "oracle_and_negative_control_are_diagnostic_only":role != "intervention"
        }));
    }

    let all_synthetic = plan_evidence == "synthetic_contract_fixture";
    let measured_outcome_rows = if all_synthetic { 0 } else { outcomes.len() as u64 };
    let all_measured_rows_have_signed_target_integrity =
        !all_synthetic
            && signed_target_integrity_receipts_verified == measured_outcome_rows;

    Ok(json!({
        "schema_version":"pitb.proof-intervention-trial-analysis.v0",
        "trial_id":trial_id,
        "evidence_class":if all_synthetic {"synthetic_analysis_fixture"} else {"measured_analysis"},
        "pair_count":n,
        "component_count":component_set.len(),
        "arm_summaries":arm_summaries,
        "paired_comparisons":comparisons,
        "all_pair_arm_rows_present":true,
        "resource_caps_respected":true,
        "non_attempts_and_unavailable_counted_as_failures":true,
        "measured_artifact_bytes_authenticated":measured_artifacts_authenticated,
        "measured_outcome_rows":measured_outcome_rows,
        "signed_target_integrity_receipt_count":signed_target_integrity_receipts_verified,
        "all_measured_rows_have_signed_target_integrity":all_measured_rows_have_signed_target_integrity,
        "checker_run_successor_receipt_semantics_validated":true,
        "checker_run_successor_runner_origin_attested":false,
        "confirmatory_result_allowed":false,
        "confirmatory_blocker":"checker, run and successor receipt structures and identity joins are validated, but their origin lacks a trusted runner/verifier signature or independently attested execution environment",
        "bootstrap_seed_sha256":plan_seed
    }))
}

fn stable_hash(value: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
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

fn percentile(sorted: &[f64], probability: f64) -> f64 {
    let position = probability * (sorted.len().saturating_sub(1)) as f64;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    let fraction = position - low as f64;
    sorted[low] * (1.0 - fraction) + sorted[high] * fraction
}

fn cluster_bootstrap(
    groups: &BTreeMap<String, Vec<i64>>,
    replicates: usize,
    seed: u64,
) -> (f64, f64) {
    let clusters: Vec<&Vec<i64>> = groups.values().collect();
    let mut rng = SplitMix64::new(seed);
    let mut estimates = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut sum = 0_i64;
        let mut count = 0_u64;
        for _ in 0..clusters.len() {
            let index = rng.next() as usize % clusters.len();
            for difference in clusters[index] {
                sum += *difference;
                count += 1;
            }
        }
        if count > 0 {
            estimates.push(sum as f64 / count as f64);
        }
    }
    estimates.sort_by(f64::total_cmp);
    (percentile(&estimates, 0.025), percentile(&estimates, 0.975))
}

fn parse_jsonl(path: &Path) -> Result<Vec<Value>> {
    let bytes = fs::read(path).with_context(|| format!("read outcomes {}", path.display()))?;
    let text = std::str::from_utf8(&bytes).context("outcomes JSONL is not UTF-8")?;
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        rows.push(
            serde_json::from_str(line)
                .with_context(|| format!("parse outcomes JSONL line {}", index + 1))?,
        );
    }
    Ok(rows)
}

fn synthesize_outcomes(plan: &Value) -> Result<Vec<Value>> {
    let trial_id = string(plan, "/trial_id")?;
    let horizon = string(plan, "/evaluation/horizon_id")?;
    let continuation = string(plan, "/evaluation/continuation_policy_sha256")?;
    let arms = field(plan, "/arms")?
        .as_array()
        .context("plan arms must be array")?;
    let pairs = field(plan, "/pairs")?
        .as_array()
        .context("plan pairs must be array")?;
    let mut rows = Vec::new();
    let mut run_index = 0_u64;
    for (pair_index, pair) in pairs.iter().enumerate() {
        let order = field(pair, "/execution_order")?
            .as_array()
            .context("execution_order array")?;
        for (arm_index, arm) in arms.iter().enumerate() {
            let arm_id = string(arm, "/arm_id")?;
            let position = order
                .iter()
                .position(|value| value.as_str() == Some(arm_id))
                .context("synthetic arm missing from execution order")?;
            let success = match arm_id {
                "A0_full_context" => pair_index % 2 == 0,
                "A1_predicted_support" => pair_index < 5,
                "A2_oracle_support" => pair_index < 7,
                "A3_random_support" => pair_index % 3 == 0,
                _ => false,
            };
            let not_attempted = arm_id == "A1_predicted_support" && pair_index == 7;
            let status = if not_attempted {
                "not_attempted"
            } else if success {
                "verified_success"
            } else {
                "failed"
            };
            let proof = if success { Some("4".repeat(64)) } else { None };
            let checker = if success { Some("5".repeat(64)) } else { None };
            let semantic = if success { Some("6".repeat(64)) } else { None };
            let successor = if success { Some("7".repeat(64)) } else { None };
            let resources = if not_attempted {
                json!({"tokens":0,"model_calls":0,"tool_calls":0,"wall_ms":0,"lean_cpu_ms":0,"cost_microusd":0})
            } else {
                json!({"tokens":1000 + arm_index as u64 * 10,"model_calls":1,"tool_calls":4,"wall_ms":800,"lean_cpu_ms":400,"cost_microusd":1000 + arm_index as u64 * 100})
            };
            rows.push(json!({
                "schema_version":OUTCOME_VERSION,
                "evidence_class":"synthetic_outcome_fixture",
                "trial_id":trial_id,
                "pair_id":pair["pair_id"],
                "arm_id":arm_id,
                "run_id":format!("synthetic-run-{run_index:04}"),
                "task_id":pair["task_id"],
                "component_id":pair["component_id"],
                "repository_commit":pair["repository_commit"],
                "lean_toolchain":pair["lean_toolchain"],
                "environment_sha256":pair["environment_sha256"],
                "initial_proof_state_sha256":pair["initial_proof_state_sha256"],
                "initial_state_receipt_sha256":pair["initial_state_receipt_sha256"],
                "target_statement_sha256":pair["target_statement_sha256"],
                "model_seed_sha256":pair["model_seed_sha256"],
                "evaluation_horizon_id":horizon,
                "continuation_policy_sha256":continuation,
                "execution_order_index":position,
                "status":status,
                "checker_passed":success,
                "target_integrity_passed":true,
                "successor_gate_status":if success {"passed"} else if not_attempted {"not_run"} else {"failed"},
                "checked_target_statement_sha256":if success {Some(pair["target_statement_sha256"].clone())} else {None::<Value>},
                "proof_sha256":proof,
                "checker_receipt_sha256":checker,
                "target_integrity_receipt_sha256":Some("6".repeat(64)),
                "successor_gate_receipt_sha256":successor,
                "run_receipt_sha256":"8".repeat(64),
                "proof_path":null,
                "checker_receipt_path":null,
                "target_integrity_receipt_path":null,
                "successor_gate_receipt_path":null,
                "run_receipt_path":null,
                "actual_resources":resources
            }));
            run_index += 1;
        }
    }
    Ok(rows)
}

fn must_reject_outcomes(
    label: &str,
    plan: &Value,
    outcomes: &[Value],
) -> Result<()> {
    if validate_and_analyze(plan, outcomes, None).is_ok() {
        bail!("negative intervention outcome control unexpectedly accepted: {label}");
    }
    println!("PITB_INTERVENTION_OUTCOME_NEGATIVE_PASS: {label}");
    Ok(())
}

fn self_test(path: &PathBuf) -> Result<()> {
    let plan_bytes = fs::read(path).with_context(|| format!("read plan fixture {}", path.display()))?;
    let plan: Value = serde_json::from_slice(&plan_bytes).context("parse plan fixture")?;
    let outcomes = synthesize_outcomes(&plan)?;
    let report = validate_and_analyze(&plan, &outcomes, None, None, None)?;
    if report["evidence_class"] != "synthetic_analysis_fixture"
        || report["pair_count"] != json!(8)
        || report["confirmatory_result_allowed"] != Value::Bool(false)
    {
        bail!("synthetic paired analyzer summary differs from its contract");
    }
    println!("PITB_INTERVENTION_OUTCOME_FIXTURE_PASS: complete synthetic pair x arm matrix analyzed");
    
    let mut missing = outcomes.clone();
    let _ = missing.pop();
    must_reject_outcomes("missing outcome from a preregistered arm", &plan, &missing)?;

    let mut duplicate = outcomes.clone();
    duplicate.push(outcomes[0].clone());
    must_reject_outcomes("duplicate pair x arm outcome", &plan, &duplicate)?;

    let mut drift = outcomes.clone();
    drift[0]["target_statement_sha256"] = json!("f".repeat(64));
    must_reject_outcomes("target statement drift", &plan, &drift)?;

    let mut over_budget = outcomes.clone();
    over_budget[0]["actual_resources"]["tokens"] = json!(40001);
    must_reject_outcomes("resource cap overrun", &plan, &over_budget)?;

    let mut false_success = outcomes.clone();
    let successful_index = false_success
        .iter()
        .position(|row| row["status"] == "verified_success")
        .context("synthetic fixture has no success row")?;
    false_success[successful_index]["target_integrity_receipt_sha256"] = Value::Null;
    must_reject_outcomes("verified-success claim without semantic receipt", &plan, &false_success)?;

    let mut order_drift = outcomes.clone();
    order_drift[0]["execution_order_index"] = json!(3);
    must_reject_outcomes("outcome execution order drift", &plan, &order_drift)?;

    let mut not_attempted_miscount = outcomes.clone();
    let not_attempted_index = not_attempted_miscount
        .iter()
        .position(|row| row["status"] == "not_attempted")
        .context("synthetic fixture has no not_attempted row")?;
    not_attempted_miscount[not_attempted_index]["checker_passed"] = Value::Bool(true);
    must_reject_outcomes("not-attempted row carries checker success", &plan, &not_attempted_miscount)?;

    println!("PITB_INTERVENTION_OUTCOME_PASS: exact plan joins, complete denominators, gates and budgets");
    Ok(())
}

fn run() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.as_slice() {
        [flag] if flag == "--help" || flag == "-h" => {
            println!("Usage: analyze_proof_intervention_outcomes --analyze PLAN.json OUTCOMES.jsonl EVIDENCE_ROOT REVIEWER_KEY_REGISTRY EXPECTED_REGISTRY_SHA256 REPORT.json | --self-test PLAN_FIXTURE.json");
            Ok(())
        }
        [flag, path] if flag == "--self-test" => self_test(&PathBuf::from(path)),
        [flag, plan_path, outcomes_path, evidence_root, reviewer_registry, expected_registry_sha256, report_path]
            if flag == "--analyze" =>
        {
            let plan_bytes = fs::read(plan_path).with_context(|| format!("read plan {plan_path}"))?;
            let plan: Value = serde_json::from_slice(&plan_bytes).context("parse plan JSON")?;
            let outcomes = parse_jsonl(Path::new(outcomes_path))?;
            let report = validate_and_analyze(
                &plan,
                &outcomes,
                Some(Path::new(evidence_root)),
                Some(Path::new(reviewer_registry)),
                Some(expected_registry_sha256),
            )?;
            fs::write(report_path, serde_json::to_vec_pretty(&report)?)
                .with_context(|| format!("write analysis report {report_path}"))?;
            println!(
                "PITB_INTERVENTION_OUTCOME_ANALYSIS_PASS: trial={} pairs={} report_sha256={}",
                report["trial_id"].as_str().unwrap_or("unknown"),
                report["pair_count"],
                hex::encode(Sha256::digest(serde_json::to_vec(&report)?))
            );
            Ok(())
        }
        _ => bail!("expected --analyze PLAN OUTCOMES EVIDENCE_ROOT REVIEWER_KEY_REGISTRY EXPECTED_REGISTRY_SHA256 REPORT, --self-test PLAN_FIXTURE, or --help"),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("PITB_INTERVENTION_OUTCOME_FAIL: {error:#}");
            ExitCode::FAILURE
        }
    }
}
