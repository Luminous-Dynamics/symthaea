// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed analysis for PITB paired proof-state intervention outcomes.
//!
//! This binary checks suite/outcome joins, distinguishes insight applicability
//! from the action taken, and reports component-clustered descriptive metrics.
//! Synthetic data is never scored. It never declares a confirmatory gate passed.

use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
};

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

const SUITE_VERSION: &str = "pitb.proof-state-intervention-suite.v0";
const OUTCOME_VERSION: &str = "pitb.proof-state-intervention-outcomes.v0";
const REPORT_VERSION: &str = "pitb.proof-state-intervention-analysis.v0";
const ANALYZER_SOURCE: &str = include_str!("analyze_proof_state_intervention.rs");
const DEFAULT_BOOTSTRAP_REPLICATES: usize = 2_000;
const DEFAULT_SEED: u64 = 0x5053_494E_5445_5256;
const MIN_BOOTSTRAP_REPLICATES: usize = 1_000;
// A percentile bootstrap over very few clusters gives unstable, low-information intervals.
// This is a descriptive reporting floor, not a guarantee of adequate statistical power.
const MIN_COMPONENTS_FOR_INTERVAL: usize = 20;

#[derive(Debug, Clone)]
struct StateRow {
    case_id: String,
    pair_id: String,
    component_id: String,
    split: String,
    state_role: String,
    intervention_kind: String,
    expected_applicability: String,
    expected_action_policy: String,
    execution_status: String,
    model_id: Option<String>,
    model_version: Option<String>,
    config_sha256: Option<String>,
    budget_caps: Option<Usage>,
    semantic_gate_status: Option<String>,
    applicability_judgment: Option<String>,
    decision: Option<String>,
    original_move_applied: Option<bool>,
    checker_result: String,
    model_calls: u64,
    input_tokens: u64,
    output_tokens: u64,
    wall_time_ms: u64,
    cost_microusd: u64,
}

#[derive(Debug, Clone)]
struct Point {
    component_id: String,
    success: bool,
}

#[derive(Debug, Clone)]
struct Metric {
    numerator: usize,
    denominator: usize,
    estimate: Option<f64>,
    independent_components: usize,
    interval_status: &'static str,
    lower_95: Option<f64>,
    upper_95: Option<f64>,
}

#[derive(Debug, Clone)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        // Xorshift64*: small deterministic PRNG for reproducible bootstrap indices.
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next_u64() % upper as u64) as usize
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 || args.len() > 7 || args.get(1).is_some_and(|x| x == "--help" || x == "-h") {
        eprintln!(
            concat!(
                "usage: {} <proof-state-suite.json> <outcomes.json> ",
                "<train|validation|test|external_challenge> <report.json> ",
                "[bootstrap_replicates] [seed]"
            ),
            args.first().map(String::as_str).unwrap_or("analyze_proof_state_intervention")
        );
        if args.get(1).is_some_and(|x| x == "--help" || x == "-h") {
            return Ok(());
        }
        bail!("invalid command-line arguments");
    }

    let suite_bytes = fs::read(&args[1]).with_context(|| format!("read suite {}", args[1]))?;
    let outcome_bytes = fs::read(&args[2]).with_context(|| format!("read outcomes {}", args[2]))?;
    let split = args[3].as_str();
    if !matches!(split, "train" | "validation" | "test" | "external_challenge") {
        bail!("unsupported split {split:?}");
    }
    let replicates = args
        .get(5)
        .map(|value| value.parse::<usize>().context("parse bootstrap replicate count"))
        .transpose()?
        .unwrap_or(DEFAULT_BOOTSTRAP_REPLICATES);
    if replicates < MIN_BOOTSTRAP_REPLICATES {
        bail!("bootstrap replicate count must be at least {MIN_BOOTSTRAP_REPLICATES}");
    }
    let seed = args
        .get(6)
        .map(|value| value.parse::<u64>().context("parse bootstrap seed"))
        .transpose()?
        .unwrap_or(DEFAULT_SEED);

    let suite: Value =
        serde_json::from_slice(&suite_bytes).context("parse proof-state suite JSON")?;
    let outcomes: Value =
        serde_json::from_slice(&outcome_bytes).context("parse proof-state outcome JSON")?;
    let report = analyze(
        &suite,
        &outcomes,
        split,
        &hex_digest(&suite_bytes),
        &hex_digest(&outcome_bytes),
        replicates,
        seed,
    )?;
    let encoded = serde_json::to_string_pretty(&report)?;
    fs::write(&args[4], format!("{encoded}\n"))
        .with_context(|| format!("write report {}", args[4]))?;

    let counts = &report["counts"];
    println!(
        "PITB_PROOF_STATE_ANALYSIS_WRITTEN: split={split} rows={} executed={} report={}",
        counts["rows_in_split"].as_u64().unwrap_or(0),
        counts["executed_rows_in_split"].as_u64().unwrap_or(0),
        args[4]
    );
    println!(
        "confirmatory_decision={}",
        report["confirmatory_decision"].as_str().unwrap_or("missing")
    );
    Ok(())
}

fn validate_case_state_contract(case: &Value, context: &str) -> Result<()> {
    let target = field(case, "target", context)?;
    let base = field(case, "base_state", context)?;
    let variant = field(case, "variant_state", context)?;

    // A proof-state intervention changes the local context, not the theorem,
    // elaborated goal, source identity, toolchain, or environment.
    for (state_role, state) in [("base_state", base), ("variant_state", variant)] {
        for key in [
            "source_commit",
            "statement_sha256",
            "goal_type_sha256",
            "environment_sha256",
            "lean_toolchain",
        ] {
            let state_value = string(state, key, &format!("{context}.{state_role}"))?;
            let target_value = string(target, key_for_target(key), &format!("{context}.target"))?;
            if state_value != target_value {
                bail!(
                    "{context}.{state_role}.{key} differs from target identity/evaluation contract"
                );
            }
        }
    }

    let base_goal_count = unsigned(base, "goal_count", &format!("{context}.base_state"))?;
    let variant_goal_count = unsigned(variant, "goal_count", &format!("{context}.variant_state"))?;
    if base_goal_count != variant_goal_count {
        bail!("{context}: proof-state intervention changed goal_count");
    }
    Ok(())
}

fn key_for_target(state_key: &str) -> &str {
    // State and target contracts use the same names for all invariant fields.
    state_key
}

fn validate_intervention_contract(case: &Value, context: &str) -> Result<()> {
    let intervention = field(case, "intervention", context)?;
    let contract = field(case, "contract", context)?;
    let kind = string(intervention, "kind", context)?;
    let operation = string(intervention, "operation", context)?;
    let expected_operation = match kind {
        "required_hypothesis_cleared" => "clear_local_decl",
        "hypothesis_alpha_renamed" => "rename_local_decl",
        "irrelevant_declaration_removed" => "remove_irrelevant_decl",
        "independent_declarations_reordered" => "reorder_independent_decls",
        _ => bail!("{context}: unsupported intervention kind {kind:?}"),
    };
    if operation != expected_operation {
        bail!("{context}: operation {operation:?} does not match intervention kind {kind:?}");
    }
    if field(intervention, "single_factor", context)?.as_bool() != Some(true) {
        bail!("{context}: confirmatory intervention must declare single_factor=true");
    }

    let before = field(intervention, "before_declaration", context)?;
    let after = field(intervention, "after_declaration", context)?;
    let before_key = string(before, "stable_key", context)?;
    let after_key = string(after, "stable_key", context)?;
    let before_role = string(before, "role", context)?;
    let after_role = string(after, "role", context)?;
    if before_key != after_key || before_role != after_role {
        bail!("{context}: before/after declarations do not identify the same declaration and role");
    }
    let before_present = field(before, "present", context)?
        .as_bool()
        .context("before_declaration.present must be boolean")?;
    let after_present = field(after, "present", context)?
        .as_bool()
        .context("after_declaration.present must be boolean")?;
    let before_type = field(before, "type_sha256", context)?.as_str();
    let after_type = field(after, "type_sha256", context)?.as_str();
    let before_position = field(before, "position", context)?.as_u64();
    let after_position = field(after, "position", context)?.as_u64();
    let before_name = field(before, "user_name", context)?.as_str();
    let after_name = field(after, "user_name", context)?.as_str();

    if before_present != before_type.is_some() || before_present != before_position.is_some() {
        bail!("{context}: before declaration presence, type digest, and position disagree");
    }
    if after_present != after_type.is_some() || after_present != after_position.is_some() {
        bail!("{context}: after declaration presence, type digest, and position disagree");
    }
    if !before_present {
        bail!("{context}: intervention source declaration must be present before the operation");
    }

    let base_status = string(contract, "base_precondition_status", context)?;
    let variant_status = string(contract, "variant_precondition_status", context)?;
    let action_policy = string(contract, "expected_action_policy", context)?;
    match kind {
        "required_hypothesis_cleared" => {
            if before_role != "required_precondition"
                || after_present
                || after_name.is_some()
                || before_type != Some(string(contract, "required_type_sha256", context)?)
                || base_status != "available"
                || variant_status != "absent"
                || action_policy != "move_forbidden_or_replan"
            {
                bail!("{context}: cleared required hypothesis contradicts its declaration or precondition contract");
            }
        }
        "hypothesis_alpha_renamed" => {
            if before_role != "required_precondition"
                || !after_present
                || before_type != after_type
                || before_type != Some(string(contract, "required_type_sha256", context)?)
                || before_name.is_none()
                || after_name.is_none()
                || before_name == after_name
                || before_position != after_position
                || base_status != "available"
                || variant_status != "available"
                || action_policy != "preserve_eligibility"
            {
                bail!("{context}: alpha-renaming contradicts its declaration or precondition contract");
            }
        }
        "irrelevant_declaration_removed" => {
            if before_role != "irrelevant"
                || after_present
                || after_name.is_some()
                || base_status != "available"
                || variant_status != "available"
                || action_policy != "preserve_eligibility"
            {
                bail!("{context}: irrelevant-declaration removal contradicts its declaration or precondition contract");
            }
        }
        "independent_declarations_reordered" => {
            if !after_present
                || before_type != after_type
                || before_position.is_none()
                || after_position.is_none()
                || before_position == after_position
                || before_name != after_name
                || base_status != "available"
                || variant_status != "available"
                || action_policy != "preserve_eligibility"
            {
                bail!("{context}: declaration reordering contradicts its declaration or precondition contract");
            }
        }
        _ => unreachable!("intervention kind was checked above"),
    }
    Ok(())
}

fn analyze(
    suite: &Value,
    bundle: &Value,
    requested_split: &str,
    suite_digest: &str,
    outcomes_digest: &str,
    bootstrap_replicates: usize,
    seed: u64,
) -> Result<Value> {
    require_string(suite, "schema_version", "suite", SUITE_VERSION)?;
    require_string(bundle, "schema_version", "outcomes", OUTCOME_VERSION)?;
    let suite_id = string(suite, "suite_id", "suite")?.to_owned();
    if string(bundle, "suite_id", "outcomes")? != suite_id {
        bail!("outcome bundle suite_id does not match suite");
    }
    let dataset_kind = string(suite, "dataset_kind", "suite")?;
    if !matches!(dataset_kind, "synthetic_fixture" | "reviewed_benchmark") {
        bail!("unsupported suite dataset_kind {dataset_kind:?}");
    }
    if string(bundle, "dataset_kind", "outcomes")? != dataset_kind {
        bail!("suite and outcome bundle dataset_kind values disagree");
    }
    let cases = array(suite, "cases", "suite")?;
    if cases.is_empty() {
        bail!("proof-state suite must contain at least one case");
    }

    let mut case_by_id: BTreeMap<String, &Value> = BTreeMap::new();
    let mut expected_outcome_ids = BTreeSet::new();
    let mut pair_components = BTreeMap::<String, String>::new();
    for (index, case) in cases.iter().enumerate() {
        let context = format!("suite.cases[{index}]");
        let case_id = string(case, "case_id", &context)?.to_owned();
        let pair_id = string(case, "pair_id", &context)?.to_owned();
        let component_id = string(case, "component_id", &context)?.to_owned();
        let split = string(case, "split", &context)?;
        if !matches!(split, "train" | "validation" | "test" | "external_challenge") {
            bail!("{context}: unsupported split {split:?}");
        }
        if case_by_id.insert(case_id.clone(), case).is_some() {
            bail!("duplicate proof-state case_id {case_id:?}");
        }
        match pair_components.insert(pair_id.clone(), component_id.clone()) {
            Some(previous) if previous != component_id => {
                bail!("pair {pair_id:?} spans multiple connected components")
            }
            _ => {}
        }

        let intervention = field(case, "intervention", &context)?;
        let kind = string(intervention, "kind", &format!("{context}.intervention"))?;
        if !matches!(
            kind,
            "required_hypothesis_cleared"
                | "hypothesis_alpha_renamed"
                | "irrelevant_declaration_removed"
                | "independent_declarations_reordered"
        ) {
            bail!("{context}: unsupported intervention kind {kind:?}");
        }
        validate_case_state_contract(case, &context)?;
        validate_intervention_contract(case, &context)?;
        let links = field(case, "outcome_links", &context)?;
        for role in ["base", "variant"] {
            let outcome_id = string(links, &format!("{role}_outcome_id"), &context)?.to_owned();
            if !expected_outcome_ids.insert(outcome_id.clone()) {
                bail!("duplicate suite outcome ID {outcome_id:?}");
            }
        }
    }

    let outcome_values = array(bundle, "outcomes", "outcomes")?;
    if outcome_values.is_empty() {
        bail!("outcome bundle must contain at least one row");
    }
    let mut rows = Vec::with_capacity(outcome_values.len());
    let mut seen_ids = BTreeSet::new();

    for (index, value) in outcome_values.iter().enumerate() {
        let context = format!("outcomes[{index}]");
        let outcome_id = string(value, "outcome_id", &context)?.to_owned();
        if !seen_ids.insert(outcome_id.clone()) {
            bail!("{context}: duplicate outcome_id {outcome_id:?}");
        }
        let case_id = string(value, "case_id", &context)?.to_owned();
        let case = case_by_id
            .get(&case_id)
            .copied()
            .with_context(|| format!("{context}: unknown case_id {case_id:?}"))?;
        let pair_id = string(value, "pair_id", &context)?.to_owned();
        let component_id = string(value, "component_id", &context)?.to_owned();
        let split = string(value, "split", &context)?.to_owned();
        let role = string(value, "state_role", &context)?.to_owned();
        if role != "base" && role != "variant" {
            bail!("{context}: state_role must be base or variant");
        }
        if pair_id != string(case, "pair_id", &case_id)? {
            bail!("{context}: pair_id differs from the suite");
        }
        if component_id != string(case, "component_id", &case_id)? {
            bail!("{context}: component_id differs from the suite");
        }
        if split != string(case, "split", &case_id)? {
            bail!("{context}: split differs from the suite");
        }
        if string(value, "source_insight_record_id", &context)?
            != string(field(case, "source_insight", &case_id)?, "record_id", &case_id)?
        {
            bail!("{context}: source insight record ID mismatch");
        }
        let insight_sha = string(value, "source_insight_sha256", &context)?;
        let expected_insight_sha = string(
            field(case, "source_insight", &case_id)?,
            "insight_sha256",
            &case_id,
        )?;
        validate_sha256(insight_sha, &format!("{context}.source_insight_sha256"))?;
        if insight_sha != expected_insight_sha {
            bail!("{context}: source insight digest mismatch");
        }

        let intervention_kind = string(field(case, "intervention", &case_id)?, "kind", &case_id)?;
        let (expected_applicability, expected_action_policy) =
            expected_state_labels(intervention_kind, &role)?;
        if string(value, "expected_applicability", &context)? != expected_applicability {
            bail!("{context}: expected_applicability contradicts the intervention/state role");
        }
        if string(value, "expected_action_policy", &context)? != expected_action_policy {
            bail!("{context}: expected_action_policy contradicts the intervention/state role");
        }

        let links = field(case, "outcome_links", &case_id)?;
        if string(links, &format!("{role}_outcome_id"), &case_id)? != outcome_id {
            bail!("{context}: outcome_id does not match its base/variant link");
        }
        let state_field = if role == "base" { "base_state" } else { "variant_state" };
        let paired_field = if role == "base" { "variant_state" } else { "base_state" };
        let state = field(case, state_field, &case_id)?;
        let paired_state = field(case, paired_field, &case_id)?;
        let target = field(case, "target", &case_id)?;
        let proof_state = field(value, "proof_state", &context)?;
        for (row_key, source_value, source_key) in [
            ("target_statement_sha256", target, "statement_sha256"),
            ("goal_type_sha256", state, "goal_type_sha256"),
            ("environment_sha256", state, "environment_sha256"),
            ("snapshot_sha256", state, "snapshot_sha256"),
            ("local_context_sha256", state, "local_context_sha256"),
        ] {
            let actual = string(proof_state, row_key, &context)?;
            validate_sha256(actual, &format!("{context}.proof_state.{row_key}"))?;
            if actual != string(source_value, source_key, &format!("{context}.{source_key}"))? {
                bail!("{context}: {row_key} does not match suite state/target evidence");
            }
        }
        for (row_key, source_key) in [
            ("paired_snapshot_sha256", "snapshot_sha256"),
            ("paired_local_context_sha256", "local_context_sha256"),
        ] {
            let actual = string(proof_state, row_key, &context)?;
            if actual != string(paired_state, source_key, &format!("{context}.{source_key}"))? {
                bail!("{context}: {row_key} does not match the opposite paired state");
            }
        }

        let execution = field(value, "execution", &context)?;
        let expected_checked_target = string(proof_state, "target_statement_sha256", &context)?;
        let parsed = parse_execution(
            execution,
            &context,
            dataset_kind,
            expected_checked_target,
        )?;
        rows.push(StateRow {
            case_id,
            pair_id,
            component_id,
            split,
            state_role: role,
            intervention_kind: intervention_kind.to_owned(),
            expected_applicability: expected_applicability.to_owned(),
            expected_action_policy: expected_action_policy.to_owned(),
            execution_status: parsed.status,
            model_id: parsed.model_id,
            model_version: parsed.model_version,
            config_sha256: parsed.config_sha256,
            budget_caps: parsed.budget_caps,
            semantic_gate_status: parsed.semantic_gate_status,
            applicability_judgment: parsed.applicability_judgment,
            decision: parsed.decision,
            original_move_applied: parsed.original_move_applied,
            checker_result: parsed.checker_result,
            model_calls: parsed.model_calls,
            input_tokens: parsed.input_tokens,
            output_tokens: parsed.output_tokens,
            wall_time_ms: parsed.wall_time_ms,
            cost_microusd: parsed.cost_microusd,
        });
    }

    if seen_ids != expected_outcome_ids {
        let missing: Vec<_> = expected_outcome_ids.difference(&seen_ids).cloned().collect();
        let extra: Vec<_> = seen_ids.difference(&expected_outcome_ids).cloned().collect();
        bail!("outcome coverage mismatch; missing={missing:?}, extra={extra:?}");
    }
    validate_pair_configurations(&rows)?;

    let selected: Vec<_> = rows.iter().filter(|row| row.split == requested_split).collect();
    if selected.is_empty() {
        bail!("no outcome rows for requested split {requested_split:?}");
    }
    let pair_count = selected
        .iter()
        .map(|row| row.pair_id.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let executed_count = selected.iter().filter(|row| row.execution_status == "executed").count();
    let not_run_count = selected.len() - executed_count;

    let mut counts = Map::new();
    counts.insert("cases_in_split".into(), json!(pair_count));
    counts.insert("rows_in_split".into(), json!(selected.len()));
    counts.insert("executed_rows_in_split".into(), json!(executed_count));
    counts.insert("not_run_rows_in_split".into(), json!(not_run_count));
    counts.insert(
        "expected_eligible_rows".into(),
        json!(selected.iter().filter(|row| row.expected_applicability == "eligible").count()),
    );
    counts.insert(
        "expected_not_applicable_rows".into(),
        json!(selected.iter().filter(|row| row.expected_applicability == "not_applicable").count()),
    );
    counts.insert(
        "uncertain_judgments".into(),
        json!(
            selected
                .iter()
                .filter(|row| row.applicability_judgment.as_deref() == Some("uncertain"))
                .count()
        ),
    );
    counts.insert(
        "checked_success_rows".into(),
        json!(selected.iter().filter(|row| row.checker_result == "checked_success").count()),
    );
    counts.insert(
        "checked_failure_rows".into(),
        json!(selected.iter().filter(|row| row.checker_result == "checked_failure").count()),
    );
    counts.insert(
        "semantic_gate_pass_rows".into(),
        json!(
            selected
                .iter()
                .filter(|row| row.semantic_gate_status.as_deref() == Some("passed"))
                .count()
        ),
    );
    counts.insert(
        "semantic_gate_failed_rows".into(),
        json!(
            selected
                .iter()
                .filter(|row| row.semantic_gate_status.as_deref() == Some("failed"))
                .count()
        ),
    );
    counts.insert(
        "semantic_gate_unknown_rows".into(),
        json!(
            selected
                .iter()
                .filter(|row| row.semantic_gate_status.as_deref() == Some("unknown"))
                .count()
        ),
    );

    let mut resource_usage = Map::new();
    let totals = Usage {
        model_calls: selected.iter().map(|row| row.model_calls).sum(),
        input_tokens: selected.iter().map(|row| row.input_tokens).sum(),
        output_tokens: selected.iter().map(|row| row.output_tokens).sum(),
        wall_time_ms: selected.iter().map(|row| row.wall_time_ms).sum(),
        cost_microusd: selected.iter().map(|row| row.cost_microusd).sum(),
    };
    resource_usage.insert("executed_rows_with_usage".into(), json!(executed_count));
    resource_usage.insert("totals".into(), json!(totals));
    resource_usage.insert(
        "mean_per_executed_row".into(),
        if executed_count == 0 {
            Value::Null
        } else {
            json!({
                "model_calls": totals.model_calls as f64 / executed_count as f64,
                "input_tokens": totals.input_tokens as f64 / executed_count as f64,
                "output_tokens": totals.output_tokens as f64 / executed_count as f64,
                "wall_time_ms": totals.wall_time_ms as f64 / executed_count as f64,
                "cost_microusd": totals.cost_microusd as f64 / executed_count as f64
            })
        },
    );

    let metrics = if dataset_kind == "synthetic_fixture" {
        Map::new()
    } else {
        compute_metrics(&rows, requested_split, bootstrap_replicates, seed)?
    };

    let confirmatory_decision = if dataset_kind == "synthetic_fixture" {
        "not_evaluated_synthetic_fixture"
    } else if not_run_count > 0 {
        "not_evaluated_incomplete_execution_coverage"
    } else {
        "not_evaluated_requires_preregistered_protocol_independent_receipts_and_review"
    };

    Ok(json!({
        "schema_version": REPORT_VERSION,
        "suite_id": suite_id,
        "dataset_kind": dataset_kind,
        "split": requested_split,
        "suite_sha256": suite_digest,
        "outcomes_sha256": outcomes_digest,
        "analyzer_source_sha256": hex_digest(ANALYZER_SOURCE.as_bytes()),
        "bootstrap": {
            "unit": "component_id",
            "replicates": bootstrap_replicates,
            "seed": seed,
            "interval": "percentile_95",
            "note": concat!(
                "Intervals are component-clustered percentile-bootstrap summaries emitted only when at least ",
                "20 independent components contribute. This is a descriptive reporting floor, not a ",
                "prospective power analysis; the analyzer never certifies the independent truth of supplied labels."
            ),
        },
        "counts": Value::Object(counts),
        "metrics": Value::Object(metrics),
        "resource_usage": Value::Object(resource_usage),
        "limitations": [
            concat!(
                "The analyzer validates suite-to-outcome joins but does not ",
                "establish the mathematical ",
                "correctness of reviewer labels."
            ),
            concat!(
                "A structurally valid run receipt digest does not prove the receipt content was ",
                "independently audited."
            ),
            "Synthetic fixtures are never scored; their model outcomes must remain not_run.",
            "The overall confirmatory decision is never set to pass by this binary."
        ],
        "confirmatory_decision": confirmatory_decision
    }))
}

#[derive(Debug, Clone)]
struct ParsedExecution {
    status: String,
    model_id: Option<String>,
    model_version: Option<String>,
    config_sha256: Option<String>,
    budget_caps: Option<Usage>,
    semantic_gate_status: Option<String>,
    applicability_judgment: Option<String>,
    decision: Option<String>,
    original_move_applied: Option<bool>,
    checker_result: String,
    model_calls: u64,
    input_tokens: u64,
    output_tokens: u64,
    wall_time_ms: u64,
    cost_microusd: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
struct Usage {
    model_calls: u64,
    input_tokens: u64,
    output_tokens: u64,
    wall_time_ms: u64,
    cost_microusd: u64,
}

fn validate_semantic_gate(
    gate: &Value,
    context: &str,
    expected_target_statement_sha256: &str,
) -> Result<String> {
    let gate_context = format!("{context}.semantic_gate");
    let status = string(gate, "status", &gate_context)?.to_owned();
    if !matches!(status.as_str(), "passed" | "failed" | "unknown") {
        bail!("{context}: unsupported semantic gate status {status:?}");
    }

    let target_digest = string(gate, "checked_target_statement_sha256", &gate_context)?;
    validate_sha256(target_digest, &format!("{gate_context}.checked_target_statement_sha256"))?;
    if target_digest != expected_target_statement_sha256 {
        bail!("{context}: semantic gate is bound to a different target statement");
    }

    let correspondence = field(gate, "statement_correspondence", &gate_context)?;
    let correspondence_status = string(correspondence, "status", &gate_context)?;
    if !matches!(correspondence_status, "matches_intended_claim" | "mismatch" | "unknown") {
        bail!("{context}: invalid semantic-gate statement-correspondence status");
    }
    let reviewer_rows = array(correspondence, "reviewers", &gate_context)?;
    let mut reviewer_ids = BTreeSet::new();
    let mut reviewer_decisions = Vec::with_capacity(reviewer_rows.len());
    for (index, reviewer) in reviewer_rows.iter().enumerate() {
        let reviewer_context = format!(
            "{gate_context}.statement_correspondence.reviewers[{index}]"
        );
        let reviewer_id = string(reviewer, "reviewer_id", &reviewer_context)?;
        if reviewer_id.trim().is_empty() {
            bail!("{context}: semantic-gate reviewer ID must not be blank");
        }
        if !reviewer_ids.insert(reviewer_id.to_owned()) {
            bail!("{context}: duplicate semantic-gate reviewer identity {reviewer_id:?}");
        }
        let decision = string(reviewer, "decision", &reviewer_context)?;
        if !matches!(decision, "matches_intended_claim" | "mismatch" | "unknown") {
            bail!("{context}: unsupported semantic-gate reviewer decision {decision:?}");
        }
        validate_sha256(
            string(reviewer, "evidence_sha256", &reviewer_context)?,
            &format!("{reviewer_context}.evidence_sha256"),
        )?;
        reviewer_decisions.push(decision);
    }
    let reviewers = reviewer_rows.len();
    let derived_correspondence = if reviewers < 2 {
        "unknown"
    } else if reviewer_decisions.iter().all(|decision| *decision == "matches_intended_claim") {
        "matches_intended_claim"
    } else if reviewer_decisions.iter().all(|decision| *decision == "mismatch") {
        "mismatch"
    } else {
        // Disagreement or any unresolved reviewer finding must not be majority-voted
        // into a binary label without a separately preregistered adjudication process.
        "unknown"
    };
    if correspondence_status != derived_correspondence {
        bail!(
            "{context}: statement-correspondence status contradicts individual reviewer decisions; expected {derived_correspondence:?}"
        );
    }
    validate_sha256(
        string(correspondence, "receipt_sha256", &gate_context)?,
        &format!("{gate_context}.statement_correspondence.receipt_sha256"),
    )?;

    let axiom_policy = field(gate, "axiom_policy", &gate_context)?;
    let axiom_status = string(axiom_policy, "status", &gate_context)?;
    if !matches!(axiom_status, "compliant" | "violation" | "unknown") {
        bail!("{context}: invalid semantic-gate axiom-policy status");
    }
    if string(axiom_policy, "policy_id", &gate_context)?.trim().is_empty() {
        bail!("{context}: semantic-gate axiom policy ID must not be blank");
    }
    validate_sha256(
        string(axiom_policy, "receipt_sha256", &gate_context)?,
        &format!("{gate_context}.axiom_policy.receipt_sha256"),
    )?;

    let successor = field(gate, "successor_check", &gate_context)?;
    let successor_status = string(successor, "status", &gate_context)?;
    if !matches!(successor_status, "passed" | "failed" | "not_applicable" | "unknown") {
        bail!("{context}: invalid semantic-gate dependent-successor status");
    }
    validate_sha256(
        string(successor, "receipt_sha256", &gate_context)?,
        &format!("{gate_context}.successor_check.receipt_sha256"),
    )?;
    let successor_reason = field(successor, "reason", &gate_context)?;
    if successor_status == "not_applicable" {
        if !successor_reason.as_str().is_some_and(|value| !value.trim().is_empty()) {
            bail!("{context}: not_applicable successor check requires a reason");
        }
    } else if !successor_reason.is_null() {
        bail!("{context}: successor-check reason is only allowed when status is not_applicable");
    }

    let mutation = field(gate, "mutation_controls", &gate_context)?;
    let mutation_status = string(mutation, "status", &gate_context)?;
    if !matches!(mutation_status, "passed" | "failed" | "unknown") {
        bail!("{context}: invalid semantic-gate mutation-control status");
    }
    validate_sha256(
        string(mutation, "suite_sha256", &gate_context)?,
        &format!("{gate_context}.mutation_controls.suite_sha256"),
    )?;
    validate_sha256(
        string(mutation, "receipt_sha256", &gate_context)?,
        &format!("{gate_context}.mutation_controls.receipt_sha256"),
    )?;

    let hard_failure = correspondence_status == "mismatch"
        || axiom_status == "violation"
        || successor_status == "failed"
        || mutation_status == "failed";
    let unresolved_evidence = correspondence_status == "unknown"
        || reviewers < 2
        || axiom_status == "unknown"
        || successor_status == "unknown"
        || mutation_status == "unknown";
    let derived_status = if hard_failure {
        "failed"
    } else if unresolved_evidence {
        "unknown"
    } else {
        "passed"
    };
    if status != derived_status {
        bail!(
            "{context}: declared semantic gate status {status:?} contradicts component evidence; expected {derived_status:?}"
        );
    }
    Ok(status)
}

fn parse_execution(
    execution: &Value,
    context: &str,
    dataset_kind: &str,
    expected_target_statement_sha256: &str,
) -> Result<ParsedExecution> {
    let status = string(execution, "status", context)?.to_owned();
    let checker_result = string(execution, "checker_result", context)?.to_owned();
    if !matches!(status.as_str(), "not_run" | "executed") {
        bail!("{context}: unsupported execution status {status:?}");
    }
    if status == "not_run" {
        for key in [
            "model",
            "decision",
            "applicability_judgment",
            "original_move_applied",
            "checked_target_statement_sha256",
            "proof_sha256",
            "run_receipt_sha256",
            "resource_usage",
            "budget_caps",
            "semantic_gate",
        ] {
            if !field(execution, key, context)?.is_null() {
                bail!("{context}: not_run execution must set {key} to null");
            }
        }
        if checker_result != "not_run" {
            bail!("{context}: not_run execution cannot claim a checker outcome");
        }
        return Ok(ParsedExecution {
            status,
            model_id: None,
            model_version: None,
            config_sha256: None,
            budget_caps: None,
            semantic_gate_status: None,
            applicability_judgment: None,
            decision: None,
            original_move_applied: None,
            checker_result,
            model_calls: 0,
            input_tokens: 0,
            output_tokens: 0,
            wall_time_ms: 0,
            cost_microusd: 0,
        });
    }
    if dataset_kind == "synthetic_fixture" {
        bail!("{context}: synthetic fixture cannot contain an executed model outcome");
    }

    let model = field(execution, "model", context)?;
    let model_id = string(model, "model_id", &format!("{context}.model"))?.to_owned();
    let model_version = string(model, "model_version", &format!("{context}.model"))?.to_owned();
    if model_id.trim().is_empty() || model_version.trim().is_empty() {
        bail!("{context}: model identity must not be blank");
    }
    let config_sha256 = string(model, "config_sha256", &format!("{context}.model"))?.to_owned();
    validate_sha256(&config_sha256, &format!("{context}.model.config_sha256"))?;
    let checked_target_sha = string(execution, "checked_target_statement_sha256", context)?;
    validate_sha256(
        checked_target_sha,
        &format!("{context}.checked_target_statement_sha256"),
    )?;
    if checked_target_sha != expected_target_statement_sha256 {
        bail!("{context}: checker target statement digest does not match the paired suite target");
    }

    let semantic_gate = field(execution, "semantic_gate", context)?;
    let semantic_gate_status =
        validate_semantic_gate(semantic_gate, context, expected_target_statement_sha256)?;

    let applicability_judgment = string(execution, "applicability_judgment", context)?.to_owned();
    if !matches!(applicability_judgment.as_str(), "eligible" | "not_applicable" | "uncertain") {
        bail!("{context}: unknown applicability judgment");
    }
    let decision = string(execution, "decision", context)?.to_owned();
    if !matches!(
        decision.as_str(),
        "apply_original_move" | "adapt_move" | "replan" | "abstain" | "reject_insight"
    ) {
        bail!("{context}: unknown action decision {decision:?}");
    }
    let original_move_applied = field(execution, "original_move_applied", context)?
        .as_bool()
        .with_context(|| format!("{context}: original_move_applied must be a boolean"))?;
    if checker_result != "checked_success" && checker_result != "checked_failure" {
        bail!("{context}: executed row must contain an actual checker result");
    }
    if decision == "apply_original_move" && !original_move_applied {
        bail!("{context}: apply_original_move conflicts with original_move_applied=false");
    }
    if original_move_applied && decision != "apply_original_move" {
        bail!("{context}: original_move_applied=true requires decision=apply_original_move");
    }
    if original_move_applied && applicability_judgment != "eligible" {
        bail!("{context}: original move cannot be applied when judged not_applicable or uncertain");
    }
    if applicability_judgment == "not_applicable" && original_move_applied {
        bail!("{context}: original move was applied despite not_applicable judgment");
    }

    let proof_sha = field(execution, "proof_sha256", context)?;
    if checker_result == "checked_success" {
        let digest = proof_sha
            .as_str()
            .with_context(|| format!("{context}: checked success requires proof_sha256"))?;
        validate_sha256(digest, &format!("{context}.proof_sha256"))?;
    } else if !proof_sha.is_null() {
        validate_sha256(
            proof_sha
                .as_str()
                .context("proof_sha256 must be a string or null")?,
            &format!("{context}.proof_sha256"),
        )?;
    }
    let receipt = string(execution, "run_receipt_sha256", context)?;
    validate_sha256(receipt, &format!("{context}.run_receipt_sha256"))?;
    let declared_caps = field(execution, "budget_caps", context)?;
    let budget_caps = Usage {
        model_calls: unsigned(declared_caps, "model_calls", context)?,
        input_tokens: unsigned(declared_caps, "input_tokens", context)?,
        output_tokens: unsigned(declared_caps, "output_tokens", context)?,
        wall_time_ms: unsigned(declared_caps, "wall_time_ms", context)?,
        cost_microusd: unsigned(declared_caps, "cost_microusd", context)?,
    };
    let resources = field(execution, "resource_usage", context)?;
    let model_calls = unsigned(resources, "model_calls", context)?;
    let input_tokens = unsigned(resources, "input_tokens", context)?;
    let output_tokens = unsigned(resources, "output_tokens", context)?;
    let wall_time_ms = unsigned(resources, "wall_time_ms", context)?;
    let cost_microusd = unsigned(resources, "cost_microusd", context)?;
    for (name, actual, cap) in [
        ("model_calls", model_calls, budget_caps.model_calls),
        ("input_tokens", input_tokens, budget_caps.input_tokens),
        ("output_tokens", output_tokens, budget_caps.output_tokens),
        ("wall_time_ms", wall_time_ms, budget_caps.wall_time_ms),
        ("cost_microusd", cost_microusd, budget_caps.cost_microusd),
    ] {
        if actual > cap {
            bail!("{context}: reported {name} usage {actual} exceeds declared cap {cap}");
        }
    }

    Ok(ParsedExecution {
        status,
        model_id: Some(model_id),
        model_version: Some(model_version),
        config_sha256: Some(config_sha256),
        budget_caps: Some(budget_caps),
        semantic_gate_status: Some(semantic_gate_status),
        applicability_judgment: Some(applicability_judgment),
        decision: Some(decision),
        original_move_applied: Some(original_move_applied),
        checker_result,
        model_calls,
        input_tokens,
        output_tokens,
        wall_time_ms,
        cost_microusd,
    })
}

fn expected_state_labels(
    intervention_kind: &str,
    role: &str,
) -> Result<(&'static str, &'static str)> {
    if role == "base" {
        return Ok(("eligible", "preserve_eligibility"));
    }
    match intervention_kind {
        "required_hypothesis_cleared" => Ok(("not_applicable", "move_forbidden_or_replan")),
        "hypothesis_alpha_renamed"
        | "irrelevant_declaration_removed"
        | "independent_declarations_reordered" => {
            Ok(("eligible", "preserve_eligibility"))
        }
        _ => bail!("unsupported intervention kind {intervention_kind:?}"),
    }
}

fn validate_pair_configurations(rows: &[StateRow]) -> Result<()> {
    let mut by_pair = BTreeMap::<&str, Vec<&StateRow>>::new();
    for row in rows {
        by_pair.entry(row.pair_id.as_str()).or_default().push(row);
    }
    for (pair_id, pair_rows) in by_pair {
        if pair_rows.len() != 2 {
            bail!(
                "pair {pair_id:?} must have exactly one base and one variant outcome; found {}",
                pair_rows.len()
            );
        }
        let base = pair_rows
            .iter()
            .find(|row| row.state_role == "base")
            .context("paired outcomes are missing base row")?;
        let variant = pair_rows
            .iter()
            .find(|row| row.state_role == "variant")
            .context("paired outcomes are missing variant row")?;
        if base.case_id != variant.case_id
            || base.component_id != variant.component_id
            || base.split != variant.split
        {
            bail!("pair {pair_id:?} has inconsistent case/component/split metadata");
        }
        if base.execution_status == "executed" && variant.execution_status == "executed" {
            if base.model_id != variant.model_id
                || base.model_version != variant.model_version
                || base.config_sha256 != variant.config_sha256
            {
                bail!(
                    concat!(
                        "pair {pair_id:?} uses mismatched model/configuration across base ",
                        "and variant states"
                    )
                );
            }
            if base.budget_caps != variant.budget_caps {
                bail!("pair {pair_id:?} uses mismatched declared budget caps across paired states");
            }
        }
    }
    Ok(())
}

fn is_qualified_target_success(
    execution_status: &str,
    checker_result: &str,
    semantic_gate_status: Option<&str>,
) -> bool {
    execution_status == "executed"
        && checker_result == "checked_success"
        && semantic_gate_status == Some("passed")
}

fn compute_metrics(
    rows: &[StateRow],
    split: &str,
    bootstrap_replicates: usize,
    seed: u64,
) -> Result<Map<String, Value>> {
    let selected: Vec<&StateRow> = rows.iter().filter(|row| row.split == split).collect();
    let mut metrics = Map::new();

    let raw_checker_success = selected.iter().map(|row| Point {
        component_id: row.component_id.clone(),
        success: row.execution_status == "executed" && row.checker_result == "checked_success",
    }).collect();
    metrics.insert(
        "raw_checker_success_rate_on_assigned_rows".into(),
        metric_json(estimate(raw_checker_success, bootstrap_replicates, seed)),
    );

    // The primary exact-target endpoint requires both kernel checking and the
    // independent semantic gate; raw compilation success remains separately visible.
    let qualified_target_success = selected.iter().map(|row| Point {
        component_id: row.component_id.clone(),
        success: is_qualified_target_success(
            &row.execution_status,
            &row.checker_result,
            row.semantic_gate_status.as_deref(),
        ),
    }).collect();
    metrics.insert(
        "exact_target_solve_rate_on_assigned_rows".into(),
        metric_json(estimate(
            qualified_target_success,
            bootstrap_replicates,
            seed.wrapping_add(20),
        )),
    );

    let coverage = selected.iter().map(|row| Point {
        component_id: row.component_id.clone(),
        success: row.execution_status == "executed"
            && row
                .applicability_judgment
                .as_deref()
                .is_some_and(|value| value != "uncertain"),
    }).collect();
    metrics.insert(
        "decisive_applicability_judgment_coverage".into(),
        metric_json(estimate(coverage, bootstrap_replicates, seed.wrapping_add(1))),
    );

    let overall_accuracy = selected.iter().map(|row| Point {
        component_id: row.component_id.clone(),
        success: row.execution_status == "executed"
            && row.applicability_judgment.as_deref()
                == Some(row.expected_applicability.as_str()),
    }).collect();
    metrics.insert(
        "applicability_correct_rate_on_assigned_rows".into(),
        metric_json(estimate(overall_accuracy, bootstrap_replicates, seed.wrapping_add(2))),
    );

    let eligible_recall = selected
        .iter()
        .filter(|row| row.expected_applicability == "eligible")
        .map(|row| Point {
        component_id: row.component_id.clone(),
        success: row.execution_status == "executed"
            && row.applicability_judgment.as_deref() == Some("eligible"),
    }).collect();
    metrics.insert(
        "eligible_insight_recall_on_assigned_rows".into(),
        metric_json(estimate(eligible_recall, bootstrap_replicates, seed.wrapping_add(3))),
    );

    let inapplicable_sensitivity = selected
        .iter()
        .filter(|row| row.expected_applicability == "not_applicable")
        .map(|row| Point {
        component_id: row.component_id.clone(),
        success: row.execution_status == "executed"
            && row.applicability_judgment.as_deref() == Some("not_applicable"),
    }).collect();
    metrics.insert(
        "inapplicability_sensitivity_on_assigned_rows".into(),
        metric_json(estimate(inapplicable_sensitivity, bootstrap_replicates, seed.wrapping_add(4))),
    );

    let false_eligibility = selected.iter().filter(|row| {
        row.expected_applicability == "not_applicable"
            && row.execution_status == "executed"
            && row.applicability_judgment.as_deref().is_some_and(|x| x != "uncertain")
    }).map(|row| Point {
        component_id: row.component_id.clone(),
        success: row.applicability_judgment.as_deref() == Some("eligible"),
    }).collect();
    metrics.insert(
        "false_eligibility_rate_among_decisive_invalid_state_judgments".into(),
        metric_json(estimate(false_eligibility, bootstrap_replicates, seed.wrapping_add(5))),
    );

    let unsafe_reuse = selected
        .iter()
        .filter(|row| {
            row.expected_applicability == "not_applicable"
                && row.execution_status == "executed"
        })
        .map(|row| Point {
        component_id: row.component_id.clone(),
        success: row.original_move_applied == Some(true),
    }).collect();
    metrics.insert(
        "unsafe_original_move_application_rate".into(),
        metric_json(estimate(unsafe_reuse, bootstrap_replicates, seed.wrapping_add(6))),
    );

    let forbidden_move_compliance = selected
        .iter()
        .filter(|row| row.expected_action_policy == "move_forbidden_or_replan")
        .map(|row| Point {
            component_id: row.component_id.clone(),
            success: row.execution_status == "executed"
                && row.decision.as_deref() != Some("apply_original_move")
                && row.original_move_applied == Some(false),
        })
        .collect();
    metrics.insert(
        "forbidden_move_compliance_on_assigned_rows".into(),
        metric_json(estimate(
            forbidden_move_compliance,
            bootstrap_replicates,
            seed.wrapping_add(10),
        )),
    );

    let pairs = selected_pairs(&selected);
    let equivalence_pairs: Vec<_> = pairs.iter().filter(|(_, base, variant)| {
        base.expected_applicability == "eligible"
            && variant.expected_applicability == "eligible"
            && base.intervention_kind != "required_hypothesis_cleared"
    }).collect();
    let pair_stability = equivalence_pairs.iter().map(|(_, base, variant)| Point {
        component_id: base.component_id.clone(),
        success: base.execution_status == "executed"
            && variant.execution_status == "executed"
            && base.applicability_judgment.as_deref().is_some_and(|x| x != "uncertain")
            && base.applicability_judgment.as_deref() == variant.applicability_judgment.as_deref(),
    }).collect();
    metrics.insert(
        "semantics_preserving_applicability_stability_on_assigned_pairs".into(),
        metric_json(estimate(pair_stability, bootstrap_replicates, seed.wrapping_add(7))),
    );

    let withdrawal_pairs: Vec<_> = pairs.iter().filter(|(_, base, variant)| {
        base.intervention_kind == "required_hypothesis_cleared"
            && base.expected_applicability == "eligible"
            && variant.expected_applicability == "not_applicable"
    }).collect();
    let withdrawal_accuracy = withdrawal_pairs.iter().map(|(_, base, variant)| Point {
        component_id: base.component_id.clone(),
        success: base.execution_status == "executed"
            && variant.execution_status == "executed"
            && base.applicability_judgment.as_deref() == Some("eligible")
            && variant.applicability_judgment.as_deref() == Some("not_applicable")
            && variant.original_move_applied == Some(false),
    }).collect();
    metrics.insert(
        "required_precondition_withdrawal_pair_accuracy".into(),
        metric_json(estimate(withdrawal_accuracy, bootstrap_replicates, seed.wrapping_add(8))),
    );

    let decisive = selected
        .iter()
        .filter(|row| {
            row.execution_status == "executed"
                && row
                    .applicability_judgment
                    .as_deref()
                    .is_some_and(|value| value != "uncertain")
        })
        .map(|row| Point {
        component_id: row.component_id.clone(),
        success: row.applicability_judgment.as_deref() == Some(row.expected_applicability.as_str()),
    }).collect();
    metrics.insert(
        "applicability_accuracy_among_decisive_executed_rows".into(),
        metric_json(estimate(decisive, bootstrap_replicates, seed.wrapping_add(9))),
    );

    let matched_config = selected
        .iter()
        .filter(|row| row.execution_status == "executed")
        .collect::<Vec<_>>();
    let executed_configurations: BTreeSet<_> = matched_config
        .iter()
        .filter_map(|row| {
            Some((
                row.model_id.as_deref()?,
                row.model_version.as_deref()?,
                row.config_sha256.as_deref()?,
            ))
        })
        .collect();
    if executed_configurations.len() > 1 {
        bail!(
            concat!(
                "requested split contains multiple model/configuration identities; analyze one ",
                "frozen comparison arm at a time"
            )
        );
    }
    Ok(metrics)
}

fn selected_pairs<'a>(rows: &[&'a StateRow]) -> Vec<(String, &'a StateRow, &'a StateRow)> {
    let mut pairs = BTreeMap::<String, Vec<&StateRow>>::new();
    for row in rows {
        pairs.entry(row.pair_id.clone()).or_default().push(*row);
    }
    let mut result = Vec::new();
    for (pair_id, members) in pairs {
        if members.len() != 2 {
            continue;
        }
        if let (Some(base), Some(variant)) = (
            members.iter().find(|row| row.state_role == "base"),
            members.iter().find(|row| row.state_role == "variant"),
        ) {
            result.push((pair_id, *base, *variant));
        }
    }
    result
}

fn estimate(points: Vec<Point>, replicates: usize, seed: u64) -> Metric {
    let denominator = points.len();
    let numerator = points.iter().filter(|point| point.success).count();
    if denominator == 0 {
        return Metric {
            numerator,
            denominator,
            estimate: None,
            independent_components: 0,
            interval_status: "not_reported_no_observations",
            lower_95: None,
            upper_95: None,
        };
    }
    let estimate = numerator as f64 / denominator as f64;
    let mut grouped = BTreeMap::<String, Vec<bool>>::new();
    for point in &points {
        grouped.entry(point.component_id.clone()).or_default().push(point.success);
    }
    let components: Vec<_> = grouped.keys().cloned().collect();
    if components.len() < MIN_COMPONENTS_FOR_INTERVAL {
        // Keep the descriptive point estimate, but do not manufacture a confidence
        // interval from a handful of nominally independent clusters. The 20-cluster
        // threshold is only a reporting floor; prospective power analysis is still required.
        return Metric {
            numerator,
            denominator,
            estimate: Some(estimate),
            independent_components: components.len(),
            interval_status: "suppressed_insufficient_independent_components",
            lower_95: None,
            upper_95: None,
        };
    }
    let mut rng = DeterministicRng::new(seed);
    let mut samples = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut success_count = 0usize;
        let mut row_count = 0usize;
        for _ in 0..components.len() {
            let component = &components[rng.index(components.len())];
            if let Some(values) = grouped.get(component) {
                row_count += values.len();
                success_count += values.iter().filter(|&&value| value).count();
            }
        }
        if row_count > 0 {
            samples.push(success_count as f64 / row_count as f64);
        }
    }
    samples.sort_by(f64::total_cmp);
    let lower = percentile(&samples, 0.025);
    let upper = percentile(&samples, 0.975);
    Metric {
        numerator,
        denominator,
        estimate: Some(estimate),
        independent_components: components.len(),
        interval_status: "reported_descriptive_interval",
        lower_95: lower,
        upper_95: upper,
    }
}

fn percentile(sorted_values: &[f64], probability: f64) -> Option<f64> {
    if sorted_values.is_empty() {
        return None;
    }
    let index = ((sorted_values.len() - 1) as f64 * probability).floor() as usize;
    sorted_values.get(index).copied()
}

fn metric_json(metric: Metric) -> Value {
    json!({
        "numerator": metric.numerator,
        "denominator": metric.denominator,
        "estimate": metric.estimate,
        "independent_components": metric.independent_components,
        "confidence_interval_95": {
            "method": "component_clustered_percentile_bootstrap",
            "status": metric.interval_status,
            "lower": metric.lower_95,
            "upper": metric.upper_95
        }
    })
}

fn require_string(value: &Value, key: &str, context: &str, expected: &str) -> Result<()> {
    let actual = string(value, key, context)?;
    if actual != expected {
        bail!("{context}.{key}: expected {expected:?}, found {actual:?}");
    }
    Ok(())
}

fn field<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a Value> {
    value.get(key).with_context(|| format!("{context}: missing field {key:?}"))
}

fn string<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str> {
    field(value, key, context)?
        .as_str()
        .with_context(|| format!("{context}.{key} must be a string"))
}

fn array<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a Vec<Value>> {
    field(value, key, context)?
        .as_array()
        .with_context(|| format!("{context}.{key} must be an array"))
}

fn unsigned(value: &Value, key: &str, context: &str) -> Result<u64> {
    field(value, key, context)?
        .as_u64()
        .with_context(|| format!("{context}.{key} must be a nonnegative integer"))
}

fn validate_sha256(value: &str, context: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        bail!("{context} must be lowercase 64-character SHA-256 hex");
    }
    Ok(())
}

fn hex_digest(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applicability_labels_differ_before_and_after_required_premise_removal() {
        assert_eq!(
            expected_state_labels("required_hypothesis_cleared", "base").unwrap(),
            ("eligible", "preserve_eligibility")
        );
        assert_eq!(
            expected_state_labels("required_hypothesis_cleared", "variant").unwrap(),
            ("not_applicable", "move_forbidden_or_replan")
        );
    }

    #[test]
    fn semantics_preserving_context_changes_keep_insight_eligible() {
        for kind in [
            "hypothesis_alpha_renamed",
            "irrelevant_declaration_removed",
            "independent_declarations_reordered",
        ] {
            assert_eq!(
                expected_state_labels(kind, "base").unwrap(),
                ("eligible", "preserve_eligibility")
            );
            assert_eq!(
                expected_state_labels(kind, "variant").unwrap(),
                ("eligible", "preserve_eligibility")
            );
        }
    }

    #[test]
    fn component_bootstrap_is_deterministic() {
        let points = vec![
            Point { component_id: "a".into(), success: true },
            Point { component_id: "a".into(), success: false },
            Point { component_id: "b".into(), success: true },
        ];
        let first = estimate(points.clone(), 1000, 71);
        let second = estimate(points, 1000, 71);
        assert_eq!(first.estimate, second.estimate);
        assert_eq!(first.lower_95, second.lower_95);
        assert_eq!(first.upper_95, second.upper_95);
        assert_eq!(first.numerator, 2);
        assert_eq!(first.denominator, 3);
    }

    #[test]
    fn percentile_selects_lower_index_for_evenly_sorted_samples() {
        assert_eq!(percentile(&[0.0, 0.5, 1.0], 0.5), Some(0.5));
        assert_eq!(percentile(&[], 0.5), None);
    }

    #[test]
    fn one_component_does_not_get_a_fictitious_confidence_interval() {
        let points = vec![
            Point { component_id: "single".into(), success: true },
            Point { component_id: "single".into(), success: false },
            Point { component_id: "single".into(), success: true },
        ];
        let result = estimate(points, 1000, 9);
        assert_eq!(result.estimate, Some(2.0 / 3.0));
        assert_eq!(result.independent_components, 1);
        assert_eq!(result.interval_status, "suppressed_insufficient_independent_components");
        assert_eq!(result.lower_95, None);
        assert_eq!(result.upper_95, None);
    }

    #[test]
    fn bootstrap_interval_is_suppressed_below_twenty_independent_components() {
        let points = (0..19)
            .map(|index| Point {
                component_id: format!("component-{index}"),
                success: index % 2 == 0,
            })
            .collect();
        let result = estimate(points, 1_000, 23);
        assert_eq!(result.independent_components, 19);
        assert_eq!(result.interval_status, "suppressed_insufficient_independent_components");
        assert!(result.estimate.is_some());
        assert_eq!(result.lower_95, None);
        assert_eq!(result.upper_95, None);
    }

    #[test]
    fn bootstrap_interval_is_reported_at_twenty_independent_components() {
        let points = (0..20)
            .map(|index| Point {
                component_id: format!("component-{index}"),
                success: index % 2 == 0,
            })
            .collect();
        let result = estimate(points, 1_000, 29);
        assert_eq!(result.independent_components, 20);
        assert_eq!(result.interval_status, "reported_descriptive_interval");
        assert!(result.lower_95.is_some());
        assert!(result.upper_95.is_some());
    }

    fn checked_in_synthetic_inputs() -> (Value, Value) {
        // The dedicated workflow stages this crate under RUNNER_TEMP. Resolve
        // fixtures from an explicit repository root there; ordinary workspace
        // tests fall back to the real repo root three levels above the crate.
        let repo_root = std::env::var_os("PITB_REPO_ROOT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
            });
        let suite_path = repo_root.join(
            "data/benchmarks/proof-insight-transfer-v0/fixtures/proof-state-intervention-suite.synthetic.json",
        );
        let outcomes_path = repo_root.join(
            "data/benchmarks/proof-insight-transfer-v0/fixtures/proof-state-intervention-outcomes.synthetic.json",
        );
        let suite_bytes = std::fs::read(&suite_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", suite_path.display()));
        let outcomes_bytes = std::fs::read(&outcomes_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", outcomes_path.display()));
        let suite: Value = serde_json::from_slice(&suite_bytes)
            .expect("checked-in proof-state suite fixture must parse");
        let outcomes: Value = serde_json::from_slice(&outcomes_bytes)
            .expect("checked-in proof-state outcomes fixture must parse");
        (suite, outcomes)
    }

    #[test]
    fn paired_state_must_preserve_target_goal_environment_and_source_identity() {
        let (suite, _) = checked_in_synthetic_inputs();
        let original = suite["cases"][0].clone();
        validate_case_state_contract(&original, "test-case")
            .expect("fixture state pair should preserve its target contract");

        for (field_name, wrong_value) in [
            ("goal_type_sha256", "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"),
            ("environment_sha256", "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"),
            ("source_commit", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
            ("statement_sha256", "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"),
            ("lean_toolchain", "leanprover/lean4:v0.0.0-invalid"),
        ] {
            let mut mutant = original.clone();
            mutant["variant_state"][field_name] = Value::String(wrong_value.into());
            let error = validate_case_state_contract(&mutant, "test-mutant")
                .expect_err("pair with changed invariant target identity must fail closed");
            assert!(
                error.to_string().contains("differs from target identity/evaluation contract"),
                "unexpected diagnostic for {field_name}: {error}"
            );
        }

        let mut changed_goal_count = original;
        changed_goal_count["variant_state"]["goal_count"] = json!(2);
        let error = validate_case_state_contract(&changed_goal_count, "test-mutant")
            .expect_err("pair with changed goal count must fail closed");
        assert!(error.to_string().contains("changed goal_count"));
    }

    #[test]
    fn intervention_kind_operation_and_precondition_contract_must_agree() {
        let (suite, _) = checked_in_synthetic_inputs();
        let original = suite["cases"][0].clone();
        validate_intervention_contract(&original, "test-case")
            .expect("fixture intervention contract should be internally consistent");

        let mut wrong_operation = original.clone();
        wrong_operation["intervention"]["operation"] = Value::String("rename_local_decl".into());
        let error = validate_intervention_contract(&wrong_operation, "test-mutant")
            .expect_err("operation/kind mismatch must fail closed");
        assert!(error.to_string().contains("does not match intervention kind"));

        let mut wrong_required_type = original.clone();
        wrong_required_type["contract"]["required_type_sha256"] =
            Value::String("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into());
        let error = validate_intervention_contract(&wrong_required_type, "test-mutant")
            .expect_err("required type mismatch must fail closed");
        assert!(error.to_string().contains("contradicts its declaration or precondition contract"));

        let mut false_clear = original.clone();
        false_clear["intervention"]["after_declaration"]["present"] = Value::Bool(true);
        let error = validate_intervention_contract(&false_clear, "test-mutant")
            .expect_err("cleared premise falsely marked present must fail closed");
        assert!(
            error.to_string().contains("after declaration presence")
                || error.to_string().contains("contradicts its declaration or precondition contract")
        );
    }

    #[test]
    fn analyzer_rejects_a_pair_with_a_mismatched_intervention_receipt() {
        let (mut suite, outcomes) = checked_in_synthetic_inputs();
        suite["cases"][0]["intervention"]["operation"] = Value::String("rename_local_decl".into());
        let error = analyze(
            &suite,
            &outcomes,
            "test",
            &"a".repeat(64),
            &"b".repeat(64),
            1000,
            17,
        ).expect_err("internally inconsistent intervention must fail before scoring");
        assert!(error.to_string().contains("does not match intervention kind"));
    }

    #[test]
    fn checked_in_synthetic_fixture_is_joined_but_never_scored() {
        let (suite, outcomes) = checked_in_synthetic_inputs();
        let report = analyze(
            &suite,
            &outcomes,
            "test",
            &"a".repeat(64),
            &"b".repeat(64),
            1000,
            17,
        )
        .expect("checked-in synthetic outcome joins must validate");

        assert_eq!(report["dataset_kind"], "synthetic_fixture");
        assert_eq!(report["counts"]["cases_in_split"], 4);
        assert_eq!(report["counts"]["rows_in_split"], 8);
        assert_eq!(report["counts"]["executed_rows_in_split"], 0);
        assert_eq!(report["counts"]["not_run_rows_in_split"], 8);
        assert_eq!(report["counts"]["expected_eligible_rows"], 7);
        assert_eq!(report["counts"]["expected_not_applicable_rows"], 1);
        assert!(report["metrics"].as_object().expect("metrics object").is_empty());
        assert_eq!(report["confirmatory_decision"], "not_evaluated_synthetic_fixture");
    }

    #[test]
    fn missing_state_side_fails_closed() {
        let (suite, mut outcomes) = checked_in_synthetic_inputs();
        outcomes["outcomes"]
            .as_array_mut()
            .expect("outcome array")
            .pop();

        let error = analyze(
            &suite,
            &outcomes,
            "test",
            &"a".repeat(64),
            &"b".repeat(64),
            1000,
            17,
        )
        .expect_err("missing paired state side must be rejected");
        assert!(error.to_string().contains("outcome coverage mismatch"));
    }

    #[test]
    fn removed_required_hypothesis_variant_cannot_be_labeled_eligible() {
        let (suite, mut outcomes) = checked_in_synthetic_inputs();
        let rows = outcomes["outcomes"].as_array_mut().expect("outcome array");
        let variant = rows
            .iter_mut()
            .find(|row| row["outcome_id"] == "state-outcome-0-variant")
            .expect("required-hypothesis-cleared variant exists");
        variant["expected_applicability"] = Value::String("eligible".into());
        variant["expected_action_policy"] = Value::String("preserve_eligibility".into());

        let error = analyze(
            &suite,
            &outcomes,
            "test",
            &"a".repeat(64),
            &"b".repeat(64),
            1000,
            17,
        )
        .expect_err("removed-premise variant must not inherit the base applicability label");
        assert!(error.to_string().contains("expected_applicability"));
    }

    #[test]
    fn synthetic_fixture_cannot_claim_a_model_execution() {
        let (suite, mut outcomes) = checked_in_synthetic_inputs();
        outcomes["outcomes"][0]["execution"]["status"] = Value::String("executed".into());

        let error = analyze(
            &suite,
            &outcomes,
            "test",
            &"a".repeat(64),
            &"b".repeat(64),
            1000,
            17,
        )
        .expect_err("synthetic fixture must never claim an executed model result");
        assert!(
            error
                .to_string()
                .contains("synthetic fixture cannot contain an executed model outcome")
        );
    }

    fn passing_semantic_gate(target_digest: &str) -> Value {
        json!({
            "status": "passed",
            "checked_target_statement_sha256": target_digest,
            "statement_correspondence": {
                "status": "matches_intended_claim",
                "reviewers": [
                    {
                        "reviewer_id": "reviewer-a",
                        "decision": "matches_intended_claim",
                        "evidence_sha256": "6".repeat(64)
                    },
                    {
                        "reviewer_id": "reviewer-b",
                        "decision": "matches_intended_claim",
                        "evidence_sha256": "7".repeat(64)
                    }
                ],
                "receipt_sha256": "1".repeat(64)
            },
            "axiom_policy": {
                "status": "compliant",
                "policy_id": "test-allowed-axioms-v0",
                "receipt_sha256": "2".repeat(64)
            },
            "successor_check": {
                "status": "not_applicable",
                "receipt_sha256": "3".repeat(64),
                "reason": "The fixture target has no meaningful dependent successor."
            },
            "mutation_controls": {
                "status": "passed",
                "suite_sha256": "4".repeat(64),
                "receipt_sha256": "5".repeat(64)
            }
        })
    }

    #[test]
    fn checker_result_must_bind_to_exact_expected_statement_digest() {
        let target_digest = "d".repeat(64);
        let make_execution = |checked_target: &str| {
            json!({
                "status": "executed",
                "model": {
                    "model_id": "test-model",
                    "model_version": "test-version",
                    "config_sha256": "a".repeat(64)
                },
                "budget_caps": {
                    "model_calls": 2,
                    "input_tokens": 16,
                    "output_tokens": 8,
                    "wall_time_ms": 100,
                    "cost_microusd": 10
                },
                "applicability_judgment": "eligible",
                "decision": "apply_original_move",
                "original_move_applied": true,
                "checked_target_statement_sha256": checked_target,
                "semantic_gate": passing_semantic_gate(checked_target),
                "checker_result": "checked_success",
                "proof_sha256": "b".repeat(64),
                "run_receipt_sha256": "c".repeat(64),
                "resource_usage": {
                    "model_calls": 1,
                    "input_tokens": 8,
                    "output_tokens": 3,
                    "wall_time_ms": 12,
                    "cost_microusd": 4
                }
            })
        };
        assert!(
            parse_execution(
                &make_execution(&target_digest),
                "test-execution",
                "reviewed_benchmark",
                &target_digest,
            )
            .is_ok()
        );
        let wrong_target = "e".repeat(64);
        let error = parse_execution(
            &make_execution(&wrong_target),
            "test-execution",
            "reviewed_benchmark",
            &target_digest,
        )
        .expect_err("checker receipt bound to a different target must be rejected");
        assert!(error.to_string().contains("does not match the paired suite target"));
    }

    #[test]
    fn primary_success_requires_checker_success_and_semantic_gate_pass() {
        assert!(is_qualified_target_success("executed", "checked_success", Some("passed")));
        assert!(!is_qualified_target_success("executed", "checked_success", Some("unknown")));
        assert!(!is_qualified_target_success("executed", "checked_success", Some("failed")));
        assert!(!is_qualified_target_success("executed", "checked_failure", Some("passed")));
        assert!(!is_qualified_target_success("not_run", "not_run", None));
    }

    #[test]
    fn semantic_gate_must_bind_to_the_exact_target() {
        let target_digest = "d".repeat(64);
        let make_execution = |gate: Value| json!({
            "status": "executed",
            "model": {
                "model_id": "test-model",
                "model_version": "test-version",
                "config_sha256": "a".repeat(64)
            },
            "applicability_judgment": "eligible",
            "decision": "apply_original_move",
            "original_move_applied": true,
            "checked_target_statement_sha256": target_digest,
            "semantic_gate": gate,
            "checker_result": "checked_success",
            "proof_sha256": "b".repeat(64),
            "run_receipt_sha256": "c".repeat(64),
            "budget_caps": {
                "model_calls": 2,
                "input_tokens": 16,
                "output_tokens": 8,
                "wall_time_ms": 100,
                "cost_microusd": 10
            },
            "resource_usage": {
                "model_calls": 1,
                "input_tokens": 8,
                "output_tokens": 3,
                "wall_time_ms": 12,
                "cost_microusd": 4
            }
        });
        let wrong = make_execution(passing_semantic_gate(&"e".repeat(64)));
        let error = parse_execution(&wrong, "test-execution", "reviewed_benchmark", &target_digest)
            .expect_err("semantic gate bound to the wrong target must fail closed");
        assert!(
            error
                .to_string()
                .contains("semantic gate is bound to a different target statement")
        );
        let valid = make_execution(passing_semantic_gate(&target_digest));
        assert!(
            parse_execution(&valid, "test-execution", "reviewed_benchmark", &target_digest).is_ok()
        );
    }

    #[test]
    fn semantic_gate_cannot_pass_with_an_axiom_violation() {
        let target_digest = "d".repeat(64);
        let mut gate = passing_semantic_gate(&target_digest);
        gate["axiom_policy"]["status"] = Value::String("violation".into());
        let error = validate_semantic_gate(&gate, "test-execution", &target_digest)
            .expect_err("passed semantic gate with an axiom violation must be rejected");
        assert!(error.to_string().contains("contradicts component evidence"));
    }

    #[test]
    fn semantic_gate_requires_two_independent_statement_reviews_to_pass() {
        let target_digest = "d".repeat(64);
        let mut gate = passing_semantic_gate(&target_digest);
        gate["statement_correspondence"]["reviewers"].as_array_mut().unwrap().pop();
        let error = validate_semantic_gate(&gate, "test-execution", &target_digest)
            .expect_err("one review cannot qualify the semantic gate");
        assert!(
            error
                .to_string()
                .contains("statement-correspondence status contradicts individual reviewer decisions")
        );
    }

    #[test]
    fn semantic_gate_does_not_vote_away_reviewer_disagreement() {
        let target_digest = "d".repeat(64);
        let mut gate = passing_semantic_gate(&target_digest);
        gate["statement_correspondence"]["reviewers"][1]["decision"] =
            Value::String("mismatch".into());
        let error = validate_semantic_gate(&gate, "test-execution", &target_digest)
            .expect_err("reviewer disagreement must not become a passing correspondence label");
        assert!(
            error
                .to_string()
                .contains("statement-correspondence status contradicts individual reviewer decisions")
        );
    }

    #[test]
    fn semantic_gate_rejects_duplicate_reviewer_ids() {
        let target_digest = "d".repeat(64);
        let mut gate = passing_semantic_gate(&target_digest);
        gate["statement_correspondence"]["reviewers"][1]["reviewer_id"] =
            Value::String("reviewer-a".into());
        let error = validate_semantic_gate(&gate, "test-execution", &target_digest)
            .expect_err("duplicate reviewer identities must be rejected");
        assert!(error.to_string().contains("duplicate semantic-gate reviewer identity"));
    }

    #[test]
    fn executed_outcome_cannot_exceed_declared_resource_caps() {
        let target_digest = "d".repeat(64);
        let mut execution = json!({
            "status": "executed",
            "model": {
                "model_id": "test-model",
                "model_version": "test-version",
                "config_sha256": "a".repeat(64)
            },
            "applicability_judgment": "eligible",
            "decision": "apply_original_move",
            "original_move_applied": true,
            "checked_target_statement_sha256": target_digest,
            "semantic_gate": passing_semantic_gate(&target_digest),
            "checker_result": "checked_success",
            "proof_sha256": "b".repeat(64),
            "run_receipt_sha256": "c".repeat(64),
            "budget_caps": {
                "model_calls": 1,
                "input_tokens": 8,
                "output_tokens": 3,
                "wall_time_ms": 12,
                "cost_microusd": 4
            },
            "resource_usage": {
                "model_calls": 2,
                "input_tokens": 8,
                "output_tokens": 3,
                "wall_time_ms": 12,
                "cost_microusd": 4
            }
        });
        let error = parse_execution(
            &execution,
            "test-execution",
            "reviewed_benchmark",
            &target_digest,
        ).expect_err("usage above the declared model-call cap must fail");
        assert!(error.to_string().contains("exceeds declared cap"));

        execution["resource_usage"]["model_calls"] = json!(1);
        assert!(parse_execution(
            &execution,
            "test-execution",
            "reviewed_benchmark",
            &target_digest,
        ).is_ok());
    }
}
