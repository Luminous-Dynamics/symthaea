// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Conservative, deterministic controller for allocating proof-search actions.
//!
//! It ranks actions by a lower confidence bound on incremental verified-success
//! probability divided by a preregistered shadow-priced fraction of remaining
//! resources. It does not verify Lean proofs or authenticate calibration receipts.

use std::{
    collections::BTreeSet,
    env,
    fs,
    path::PathBuf,
    process::ExitCode,
};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const SCHEMA_VERSION: &str = "pitb.proof-search-allocation-decision.v0";
const POLICY_ID: &str = "pitb-lcb-shadow-price-v0";
const CONFIDENCE_Z_HUNDREDTHS: u64 = 196;
const WEIGHT_TOTAL_PPM: u64 = 1_000_000;
const MAX_ACTIONS: usize = 64;
const RESOURCE_FIELDS: [&str; 6] = [
    "tokens",
    "model_calls",
    "tool_calls",
    "wall_ms",
    "lean_cpu_ms",
    "cost_microusd",
];

#[derive(Debug, Clone)]
struct Candidate {
    action_id: String,
    kind: String,
    predicted_ppm: u64,
    standard_error_ppm: u64,
    lcb_ppm: u64,
    shadow_cost_ppm: f64,
    score: f64,
    resources: Value,
}

#[derive(Debug)]
enum Mode {
    Help,
    Select { input: PathBuf, output: PathBuf },
    SelfTest(PathBuf),
}

fn string<'a>(value: &'a Value, path: &str) -> Result<&'a str> {
    value
        .pointer(path)
        .and_then(Value::as_str)
        .with_context(|| format!("{path} must be a string"))
}

fn integer(value: &Value, path: &str) -> Result<u64> {
    value
        .pointer(path)
        .and_then(Value::as_u64)
        .with_context(|| format!("{path} must be a nonnegative integer"))
}

fn boolean(value: &Value, path: &str) -> Result<bool> {
    value
        .pointer(path)
        .and_then(Value::as_bool)
        .with_context(|| format!("{path} must be boolean"))
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn digest(value: &Value, path: &str) -> Result<&str> {
    let candidate = string(value, path)?;
    if !is_lower_hex(candidate, 64) {
        bail!("{path} must be a 64-character lowercase SHA-256 digest");
    }
    Ok(candidate)
}

fn resource(value: &Value, path: &str) -> Result<u64> {
    integer(value, path)
}

fn lcb_ppm(predicted: u64, standard_error: u64) -> u64 {
    // Round the uncertainty penalty upward: a conservative integer lower bound.
    let penalty = CONFIDENCE_Z_HUNDREDTHS
        .saturating_mul(standard_error)
        .saturating_add(99)
        / 100;
    predicted.saturating_sub(penalty)
}

fn validate_input(value: &Value) -> Result<()> {
    if string(value, "/schema_version")? != SCHEMA_VERSION {
        bail!("unsupported proof-search allocation schema version");
    }
    let split = string(value, "/evaluation_split")?;
    if !matches!(split, "validation" | "test" | "external_challenge") {
        bail!("unsupported evaluation_split");
    }
    let task_id = digest(value, "/target/task_id")?;
    let _ = task_id;
    let commit = string(value, "/target/repository_commit")?;
    if !is_lower_hex(commit, 40) {
        bail!("target.repository_commit must be a 40-character lowercase Git SHA-1");
    }
    let proof_state = digest(value, "/target/proof_state_sha256")?;
    let _ = digest(value, "/target/statement_sha256")?;

    if string(value, "/policy/policy_id")? != POLICY_ID {
        bail!("unsupported allocation policy");
    }
    let _ = digest(value, "/policy/policy_sha256")?;
    let _ = digest(value, "/policy/calibration_manifest_sha256")?;
    if string(value, "/policy/calibration_split")? != "validation" {
        bail!("calibration data must come from the validation split");
    }
    if integer(value, "/policy/calibration_sample_count")? < 200 {
        bail!("calibration sample count is below the qualification minimum");
    }
    if integer(value, "/policy/calibration_ece_ppm")?
        > integer(value, "/policy/max_calibration_ece_ppm")?
    {
        bail!("calibration ECE exceeds the registered maximum");
    }
    if integer(value, "/policy/confidence_z_hundredths")? != CONFIDENCE_Z_HUNDREDTHS {
        bail!("confidence multiplier differs from the fixed 1.96 policy");
    }

    let mut weight_sum = 0_u64;
    for field in RESOURCE_FIELDS {
        weight_sum = weight_sum
            .checked_add(integer(value, &format!("/policy/shadow_weights_ppm/{field}"))?)
            .context("shadow weight sum overflow")?;
        let _ = resource(value, &format!("/remaining_budget/{field}"))?;
    }
    if weight_sum != WEIGHT_TOTAL_PPM {
        bail!("shadow weights must sum to exactly 1,000,000 ppm");
    }

    let actions = value
        .get("actions")
        .and_then(Value::as_array)
        .context("actions must be an array")?;
    if actions.is_empty() || actions.len() > MAX_ACTIONS {
        bail!("action count must be in 1..={MAX_ACTIONS}");
    }
    let mut ids = BTreeSet::new();
    for (index, action) in actions.iter().enumerate() {
        let prefix = format!("/actions/{index}");
        let action_id = string(action, &format!("{prefix}/action_id"))?;
        if action_id.trim().is_empty() || !ids.insert(action_id) {
            bail!("action IDs must be nonempty and unique");
        }
        let _ = string(action, &format!("{prefix}/kind"))?;
        if digest(action, &format!("{prefix}/proof_state_sha256"))? != proof_state {
            bail!("action {action_id} is based on a different proof-state identity");
        }
        let predicted = integer(action, &format!("{prefix}/predicted_incremental_success_ppm"))?;
        let standard_error = integer(action, &format!("{prefix}/standard_error_ppm"))?;
        if predicted > WEIGHT_TOTAL_PPM || standard_error > WEIGHT_TOTAL_PPM {
            bail!("action {action_id} probability estimate exceeds 1,000,000 ppm");
        }
        let _ = digest(action, &format!("{prefix}/prediction_evidence_sha256"))?;
        let _ = digest(action, &format!("{prefix}/calibration_receipt_sha256"))?;
        if !boolean(action, &format!("{prefix}/preconditions_met"))? {
            bail!("action {action_id} has unmet preconditions; omit it rather than proposing it");
        }
        let mut has_nonzero_cost = false;
        for field in RESOURCE_FIELDS {
            let amount = resource(action, &format!("{prefix}/expected_resources/{field}"))?;
            has_nonzero_cost |= amount > 0;
        }
        if !has_nonzero_cost {
            bail!("action {action_id} has zero expected resource cost");
        }
    }
    Ok(())
}

fn build_candidate(value: &Value, index: usize) -> Result<Option<Candidate>> {
    let action = &value["actions"][index];
    let action_id = string(action, "/action_id")?.to_owned();
    let kind = string(action, "/kind")?.to_owned();
    let predicted = integer(action, "/predicted_incremental_success_ppm")?;
    let standard_error = integer(action, "/standard_error_ppm")?;
    let lcb = lcb_ppm(predicted, standard_error);
    let minimum_lcb = integer(value, "/policy/minimum_lcb_delta_ppm")?;

    let mut shadow_cost = 0.0_f64;
    let mut resources = serde_json::Map::new();
    for field in RESOURCE_FIELDS {
        let used = integer(action, &format!("/expected_resources/{field}"))?;
        let remaining = integer(value, &format!("/remaining_budget/{field}"))?;
        if used > remaining {
            return Ok(None);
        }
        resources.insert(field.to_owned(), json!(used));
        let weight = integer(value, &format!("/policy/shadow_weights_ppm/{field}"))?;
        if weight > 0 {
            if remaining == 0 {
                return Ok(None);
            }
            shadow_cost += weight as f64 * used as f64 / remaining as f64;
        }
    }
    if lcb < minimum_lcb || lcb == 0 || !shadow_cost.is_finite() || shadow_cost <= 0.0 {
        return Ok(None);
    }
    let score = lcb as f64 / shadow_cost;
    if !score.is_finite() || score <= 0.0 {
        return Ok(None);
    }
    Ok(Some(Candidate {
        action_id,
        kind,
        predicted_ppm: predicted,
        standard_error_ppm: standard_error,
        lcb_ppm: lcb,
        shadow_cost_ppm: shadow_cost,
        score,
        resources: Value::Object(resources),
    }))
}

fn choose(value: &Value, input_sha256: &str) -> Result<Value> {
    validate_input(value)?;
    let actions = value["actions"]
        .as_array()
        .context("actions must be an array")?;
    let mut candidates = Vec::new();
    let mut rejected = Vec::new();
    for index in 0..actions.len() {
        match build_candidate(value, index)? {
            Some(candidate) => candidates.push(candidate),
            None => rejected.push(string(&actions[index], "/action_id")?.to_owned()),
        }
    }
    candidates.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.action_id.cmp(&right.action_id))
    });

    let weights_sha256 = {
        let encoded = serde_json::to_vec(&value["policy"]["shadow_weights_ppm"])?;
        hex::encode(Sha256::digest(encoded))
    };
    let selected = candidates.first();
    let decision = match selected {
        Some(candidate) => json!({
            "schema_version":"pitb.proof-search-allocation-result.v0",
            "decision_id":value["decision_id"],
            "input_sha256":input_sha256,
            "policy_id":POLICY_ID,
            "policy_sha256":value["policy"]["policy_sha256"],
            "calibration_manifest_sha256":value["policy"]["calibration_manifest_sha256"],
            "shadow_weights_sha256":weights_sha256,
            "decision":"execute",
            "selected_action":{
                "action_id":candidate.action_id,
                "kind":candidate.kind,
                "predicted_incremental_success_ppm":candidate.predicted_ppm,
                "standard_error_ppm":candidate.standard_error_ppm,
                "lower_confidence_bound_ppm":candidate.lcb_ppm,
                "shadow_cost_ppm":candidate.shadow_cost_ppm,
                "lcb_per_shadow_cost":candidate.score,
                "expected_resources":candidate.resources
            },
            "rejected_or_ineligible_action_ids":rejected,
            "selection_rule":"max_lcb_incremental_verified_success_per_shadow_priced_remaining_budget; lexical action_id tie-break",
            "confidence_z_hundredths":CONFIDENCE_Z_HUNDREDTHS,
            "semantic_success_claimed":false
        }),
        None => json!({
            "schema_version":"pitb.proof-search-allocation-result.v0",
            "decision_id":value["decision_id"],
            "input_sha256":input_sha256,
            "policy_id":POLICY_ID,
            "policy_sha256":value["policy"]["policy_sha256"],
            "calibration_manifest_sha256":value["policy"]["calibration_manifest_sha256"],
            "shadow_weights_sha256":weights_sha256,
            "decision":"stop",
            "selected_action":null,
            "rejected_or_ineligible_action_ids":rejected,
            "selection_rule":"no feasible action has a positive lower-confidence gain above the registered threshold",
            "confidence_z_hundredths":CONFIDENCE_Z_HUNDREDTHS,
            "semantic_success_claimed":false
        }),
    };
    Ok(decision)
}

fn select_file(input: &PathBuf, output: &PathBuf) -> Result<()> {
    let bytes = fs::read(input).with_context(|| format!("read input {}", input.display()))?;
    let input_sha256 = hex::encode(Sha256::digest(&bytes));
    let value: Value = serde_json::from_slice(&bytes).context("parse allocation decision JSON")?;
    let decision = choose(&value, &input_sha256)?;
    fs::write(output, serde_json::to_vec_pretty(&decision)?)
        .with_context(|| format!("write decision {}", output.display()))?;
    println!(
        "PITB_PROOF_SEARCH_ALLOCATION_PASS: decision={} selected={} input_sha256={}",
        decision["decision"],
        decision["selected_action"]["action_id"].as_str().unwrap_or("none"),
        input_sha256
    );
    Ok(())
}

fn must_reject(label: &str, value: Value) -> Result<()> {
    let bytes = serde_json::to_vec(&value)?;
    let digest = hex::encode(Sha256::digest(&bytes));
    if choose(&value, &digest).is_ok() {
        bail!("negative allocation control unexpectedly accepted: {label}");
    }
    println!("PITB_PROOF_SEARCH_ALLOCATION_NEGATIVE_PASS: {label}");
    Ok(())
}

fn self_test(path: &PathBuf) -> Result<()> {
    let bytes = fs::read(path).with_context(|| format!("read fixture {}", path.display()))?;
    let mut value: Value = serde_json::from_slice(&bytes).context("parse synthetic fixture")?;
    let digest = hex::encode(Sha256::digest(&bytes));
    let selected = choose(&value, &digest)?;
    if selected["decision"] != "execute"
        || selected["selected_action"]["action_id"] != "retrieve-01"
    {
        bail!("synthetic controller did not select the best conservative value/cost action");
    }
    println!("PITB_PROOF_SEARCH_ALLOCATION_FIXTURE_PASS: retrieve-01 selected by conservative value/cost");

    // A high raw prediction must lose when its uncertainty makes the lower bound weak.
    let mut bad = value.clone();
    bad["actions"][1]["standard_error_ppm"] = json!(120_000);
    let altered = serde_json::to_vec(&bad)?;
    let decision = choose(&bad, &hex::encode(Sha256::digest(altered)))?;
    if decision["selected_action"]["action_id"] == "retrieve-01" {
        bail!("high-uncertainty action retained the winning rank");
    }
    println!("PITB_PROOF_SEARCH_ALLOCATION_PASS: uncertainty-sensitive ranking");

    // A cheaper action that is not feasible under one remaining cap cannot be selected.
    let mut bad = value.clone();
    bad["actions"][1]["expected_resources"]["tool_calls"] = json!(9);
    bad["remaining_budget"]["tool_calls"] = json!(8);
    let decision = choose(&bad, &hex::encode(Sha256::digest(serde_json::to_vec(&bad)?)))?;
    if decision["selected_action"]["action_id"] != "continue-01" {
        bail!("action exceeding one remaining resource cap was not excluded before ranking");
    }
    println!("PITB_PROOF_SEARCH_ALLOCATION_PASS: multidimensional budget feasibility");

    let mut bad = value.clone();
    bad["policy"]["calibration_ece_ppm"] = json!(80_000);
    must_reject("calibration error above preregistered maximum", bad)?;

    let mut bad = value.clone();
    bad["actions"][0]["proof_state_sha256"] = json!("f".repeat(64));
    must_reject("proof-state identity drift", bad)?;

    let mut bad = value.clone();
    bad["actions"][1]["action_id"] = json!("continue-01");
    must_reject("duplicate action ID", bad)?;

    let mut bad = value.clone();
    bad["policy"]["shadow_weights_ppm"]["tokens"] = json!(0);
    must_reject("shadow weights do not sum to one", bad)?;

    let mut bad = value.clone();
    for action in bad["actions"].as_array_mut().context("actions array")? {
        action["predicted_incremental_success_ppm"] = json!(10_000);
        action["standard_error_ppm"] = json!(10_000);
    }
    let stopped = choose(&bad, &hex::encode(Sha256::digest(serde_json::to_vec(&bad)?)))?;
    if stopped["decision"] != "stop" || !stopped["selected_action"].is_null() {
        bail!("controller did not stop when every action's lower bound fell below threshold");
    }
    println!("PITB_PROOF_SEARCH_ALLOCATION_PASS: conservative stop rule");

    value["remaining_budget"]["tokens"] = json!(0);
    let no_tokens = choose(&value, &hex::encode(Sha256::digest(serde_json::to_vec(&value)?)))?;
    if no_tokens["decision"] != "stop" {
        bail!("controller did not stop after all actions became resource-infeasible");
    }
    println!("PITB_PROOF_SEARCH_ALLOCATION_PASS: exhausted budget stop rule");
    Ok(())
}

fn parse_mode() -> Result<Mode> {
    let mut args = env::args().skip(1);
    let Some(action) = args.next() else {
        bail!("expected --select INPUT OUTPUT, --self-test INPUT, or --help");
    };
    if matches!(action.as_str(), "--help" | "-h") {
        return Ok(Mode::Help);
    }
    match action.as_str() {
        "--select" => {
            let input = PathBuf::from(args.next().context("missing input path")?);
            let output = PathBuf::from(args.next().context("missing output path")?);
            if args.next().is_some() {
                bail!("too many --select arguments");
            }
            Ok(Mode::Select { input, output })
        }
        "--self-test" => {
            let input = PathBuf::from(args.next().context("missing fixture path")?);
            if args.next().is_some() {
                bail!("too many --self-test arguments");
            }
            Ok(Mode::SelfTest(input))
        }
        _ => bail!("unknown action {action}"),
    }
}

fn run() -> Result<()> {
    match parse_mode()? {
        Mode::Help => {
            println!("Usage: allocate_proof_search_budget --select INPUT OUTPUT | --self-test INPUT");
            Ok(())
        }
        Mode::Select { input, output } => select_file(&input, &output),
        Mode::SelfTest(path) => self_test(&path),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("PITB_PROOF_SEARCH_ALLOCATION_FAIL: {error:#}");
            ExitCode::FAILURE
        }
    }
}
