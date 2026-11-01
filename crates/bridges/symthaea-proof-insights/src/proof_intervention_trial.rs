// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Shared validation for preregistered paired proof-intervention trial plans.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const VERSION: &str = "pitb.proof-intervention-trial-plan.v0";
const PAIR_DOMAIN: &[u8] = b"PITB-INTERVENTION-PAIR-V0\0";
const ORDER_DOMAIN: &[u8] = b"PITB-INTERVENTION-ORDER-V0\0";
const ORDER_RULE: &str =
    "sort by SHA256(domain || seed || pair_id || arm_id), ascending; arm ID breaks ties";
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

fn lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn digest_bytes(value: &str, label: &str) -> Result<Vec<u8>> {
    if !lower_hex(value, 64) {
        bail!("{label} must be a 64-character lowercase SHA-256 digest");
    }
    hex::decode(value).with_context(|| format!("decode {label}"))
}

fn commit(value: &str, label: &str) -> Result<()> {
    if !lower_hex(value, 40) {
        bail!("{label} must be a 40-character lowercase Git SHA-1");
    }
    Ok(())
}

fn sha256(parts: &[&[u8]]) -> String {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update(part);
    }
    hex::encode(hash.finalize())
}

fn expected_pair_id(pair: &Value) -> Result<String> {
    let task = digest_bytes(string(pair, "/task_id")?, "pair.task_id")?;
    let component = string(pair, "/component_id")?;
    if component.trim().is_empty() {
        bail!("pair.component_id cannot be empty");
    }
    let state = digest_bytes(
        string(pair, "/initial_proof_state_sha256")?,
        "pair.initial_proof_state_sha256",
    )?;
    let statement = digest_bytes(
        string(pair, "/target_statement_sha256")?,
        "pair.target_statement_sha256",
    )?;
    let seed = digest_bytes(
        string(pair, "/model_seed_sha256")?,
        "pair.model_seed_sha256",
    )?;
    Ok(sha256(&[
        PAIR_DOMAIN,
        &task,
        component.as_bytes(),
        b"\0",
        &state,
        &statement,
        &seed,
    ]))
}

fn expected_execution_order(
    seed_hex: &str,
    pair_id_hex: &str,
    arm_ids: &[String],
) -> Result<Vec<String>> {
    let seed = digest_bytes(seed_hex, "randomization.seed_sha256")?;
    let pair_id = digest_bytes(pair_id_hex, "pair.pair_id")?;
    let mut keys = Vec::with_capacity(arm_ids.len());
    for arm_id in arm_ids {
        let key = sha256(&[
            ORDER_DOMAIN,
            &seed,
            &pair_id,
            arm_id.as_bytes(),
        ]);
        keys.push((key, arm_id.clone()));
    }
    keys.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    Ok(keys.into_iter().map(|(_, arm_id)| arm_id).collect())
}

fn parse_arm_ids(value: &Value) -> Result<(Vec<String>, BTreeMap<String, String>)> {
    let arms = field(value, "/arms")?
        .as_array()
        .context("arms must be an array")?;
    if !(2..=8).contains(&arms.len()) {
        bail!("v0 requires 2..=8 preregistered arms");
    }
    let mut ids = BTreeSet::new();
    let mut role_by_id = BTreeMap::new();
    let mut arm_ids = Vec::with_capacity(arms.len());
    let mut control_count = 0usize;
    let mut intervention_count = 0usize;
    for (index, arm) in arms.iter().enumerate() {
        let pointer = format!("/arms/{index}");
        let id = string(arm, &format!("{pointer}/arm_id"))?.to_owned();
        if id.trim().is_empty() || !ids.insert(id.clone()) {
            bail!("arm IDs must be nonempty and unique");
        }
        let role = string(arm, &format!("{pointer}/role"))?;
        if !matches!(
            role,
            "control" | "intervention" | "oracle_upper_bound" | "negative_control"
        ) {
            bail!("arm {id} has unknown role {role}");
        }
        control_count += usize::from(role == "control");
        intervention_count += usize::from(role == "intervention");
        let method = string(arm, &format!("{pointer}/method"))?;
        if method.trim().is_empty() {
            bail!("arm {id} method cannot be empty");
        }
        let _ = digest_bytes(
            string(arm, &format!("{pointer}/policy_sha256"))?,
            "arm.policy_sha256",
        )?;
        let _ = digest_bytes(
            string(arm, &format!("{pointer}/config_sha256"))?,
            "arm.config_sha256",
        )?;
        role_by_id.insert(id.clone(), role.to_owned());
        arm_ids.push(id);
    }
    if control_count != 1 || intervention_count == 0 {
        bail!("trial requires exactly one control arm and at least one intervention arm");
    }
    arm_ids.sort();
    Ok((arm_ids, role_by_id))
}

fn budget_signature(pair: &Value) -> Result<Vec<u64>> {
    RESOURCE_FIELDS
        .iter()
        .map(|field| {
            let amount = integer(
                pair,
                &format!("/resource_budget/{field}"),
            )?;
            if amount == 0 {
                bail!("matched resource budgets must be positive for {field}");
            }
            Ok(amount)
        })
        .collect()
}

pub fn validate_plan(value: &Value) -> Result<usize> {
    if string(value, "/schema_version")? != VERSION {
        bail!("unsupported proof-intervention trial plan version");
    }
    if !matches!(
        string(value, "/evidence_class")?,
        "synthetic_contract_fixture" | "preregistered_trial"
    ) {
        bail!("unknown intervention plan evidence class");
    }
    if string(value, "/trial_id")?.trim().is_empty()
        || string(value, "/dataset/cohort_id")?.trim().is_empty()
        || string(value, "/dataset/snapshot_id")?.trim().is_empty()
        || string(value, "/model/model_id")?.trim().is_empty()
        || string(value, "/model/model_version")?.trim().is_empty()
        || string(value, "/evaluation/horizon_id")?.trim().is_empty()
    {
        bail!("trial, cohort, snapshot, model and horizon identities cannot be empty");
    }
    if !matches!(
        string(value, "/evaluation_split")?,
        "validation" | "test" | "external_challenge"
    ) {
        bail!("unsupported evaluation split");
    }
    let _ = digest_bytes(
        string(value, "/dataset/dataset_manifest_sha256")?,
        "dataset manifest",
    )?;
    let _ = digest_bytes(
        string(value, "/evaluation/continuation_policy_sha256")?,
        "continuation policy",
    )?;
    let _ = digest_bytes(
        string(value, "/randomization/seed_sha256")?,
        "randomization seed",
    )?;
    if string(value, "/randomization/algorithm")? != "sha256-pair-arm-order-v0"
        || string(value, "/randomization/order_rule")? != ORDER_RULE
    {
        bail!("randomization algorithm/order rule differs from v0");
    }

    for pointer in [
        "/execution_controls/fresh_process_per_arm",
        "/execution_controls/independent_copy_of_initial_state",
        "/execution_controls/discard_mutable_caches_between_arms",
        "/execution_controls/same_environment_within_pair",
        "/execution_controls/same_model_seed_within_pair",
        "/execution_controls/same_resource_budget_within_pair",
    ] {
        if field(value, pointer)?.as_bool() != Some(true) {
            bail!("{pointer} must be true; paired arms require isolation and matched inputs");
        }
    }
    if string(value, "/analysis_policy/primary_estimand")?
        != "paired_intention_to_treat_risk_difference"
        || string(value, "/analysis_policy/resampling_unit")? != "component_id"
        || !matches!(
            string(value, "/analysis_policy/confirmatory_decision")?,
            "not_evaluated_until_measured_outcome_receipts_are_validated"
        )
    {
        bail!("analysis policy differs from the preregistered paired ITT contract");
    }
    for pointer in [
        "/analysis_policy/include_not_attempted_as_failure",
        "/analysis_policy/include_tool_unavailable_as_failure",
        "/analysis_policy/require_complete_arm_coverage",
        "/analysis_policy/oracle_arm_is_diagnostic_only",
    ] {
        if field(value, pointer)?.as_bool() != Some(true) {
            bail!("{pointer} must be true");
        }
    }
    let minimum_components =
        integer(value, "/analysis_policy/minimum_components")? as usize;
    if minimum_components < 4 {
        bail!("component-cluster analysis requires at least four components");
    }

    let (arm_ids, _) = parse_arm_ids(value)?;
    let pairs = field(value, "/pairs")?
        .as_array()
        .context("pairs must be an array")?;
    if pairs.len() < minimum_components {
        bail!("pair count is below the registered minimum component count");
    }
    let random_seed = string(value, "/randomization/seed_sha256")?;
    let mut pair_ids = BTreeSet::new();
    let mut pair_units = BTreeSet::new();
    let mut components = BTreeSet::new();
    let mut common_budget: Option<Vec<u64>> = None;

    for (index, pair) in pairs.iter().enumerate() {
        let prefix = format!("/pairs/{index}");
        let claimed_pair_id = string(pair, &format!("{prefix}/pair_id"))?;
        let task_id = string(pair, &format!("{prefix}/task_id"))?;
        let component_id = string(pair, &format!("{prefix}/component_id"))?;
        let repository_commit = string(pair, &format!("{prefix}/repository_commit"))?;
        let lean_toolchain = string(pair, &format!("{prefix}/lean_toolchain"))?;
        let environment = string(pair, &format!("{prefix}/environment_sha256"))?;
        let state = string(pair, &format!("{prefix}/initial_proof_state_sha256"))?;
        let state_receipt = string(pair, &format!("{prefix}/initial_state_receipt_sha256"))?;
        let statement = string(pair, &format!("{prefix}/target_statement_sha256"))?;
        let model_seed = string(pair, &format!("{prefix}/model_seed_sha256"))?;

        commit(repository_commit, "pair.repository_commit")?;
        for (digest, label) in [
            (task_id, "pair.task_id"),
            (environment, "pair.environment_sha256"),
            (state, "pair.initial_proof_state_sha256"),
            (state_receipt, "pair.initial_state_receipt_sha256"),
            (statement, "pair.target_statement_sha256"),
            (model_seed, "pair.model_seed_sha256"),
        ] {
            let _ = digest_bytes(digest, label)?;
        }
        if lean_toolchain.trim().is_empty() {
            bail!("pair {index} Lean toolchain cannot be empty");
        }
        if component_id.trim().is_empty() {
            bail!("pair {index} component_id cannot be empty");
        }

        let expected_id = expected_pair_id(pair)?;
        if claimed_pair_id != expected_id {
            bail!("pair {index} pair_id does not bind task, component, initial state, statement and model seed");
        }
        if !pair_ids.insert(claimed_pair_id.to_owned()) {
            bail!("duplicate pair_id {claimed_pair_id}");
        }

        let unit_key = format!("{task_id}:{state}:{model_seed}");
        if !pair_units.insert(unit_key) {
            bail!("duplicate task x initial-state x model-seed pair unit");
        }
        components.insert(component_id.to_owned());

        let budget = budget_signature(pair)?;
        if let Some(expected) = &common_budget {
            if expected != &budget {
                bail!("all v0 pairs must use the same preregistered multi-resource budget");
            }
        } else {
            common_budget = Some(budget);
        }

        let supplied_order = field(pair, &format!("{prefix}/execution_order"))?
            .as_array()
            .context("execution_order must be an array")?;
        let supplied_order: Vec<String> = supplied_order
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_owned)
                    .context("execution_order entries must be strings")
            })
            .collect::<Result<_>>()?;
        let order_set: BTreeSet<&str> = supplied_order.iter().map(String::as_str).collect();
        let arm_set: BTreeSet<&str> = arm_ids.iter().map(String::as_str).collect();
        if order_set.len() != supplied_order.len() || order_set != arm_set {
            bail!("pair {index} execution_order must contain every arm exactly once");
        }
        let expected_order =
            expected_execution_order(random_seed, claimed_pair_id, &arm_ids)?;
        if supplied_order != expected_order {
            bail!("pair {index} execution_order does not match deterministic randomization");
        }
    }

    if components.len() < minimum_components {
        bail!(
            "trial has {} distinct components; at least {} required",
            components.len(),
            minimum_components
        );
    }
    Ok(pairs.len())
}

fn refresh_pair_ids_and_orders(value: &mut Value) -> Result<()> {
    let seed = string(value, "/randomization/seed_sha256")?.to_owned();
    let (arm_ids, _) = parse_arm_ids(value)?;
    let pairs = value["pairs"].as_array_mut().context("pairs array")?;
    for pair in pairs {
        let pair_id = expected_pair_id(pair)?;
        pair["pair_id"] = json!(pair_id);
        let order = expected_execution_order(&seed, &pair_id, &arm_ids)?;
        pair["execution_order"] = json!(order);
    }
    Ok(())
}

