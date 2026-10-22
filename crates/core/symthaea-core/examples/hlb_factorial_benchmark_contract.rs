// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Validates the HLB HDC × CfC/LTC × FEP benchmark design contract.
//!
//! This executable validates a plan, not an experiment. It does not instantiate
//! cognitive components, run tasks, measure performance, or qualify SOTA claims.
//! A run must be bound to the exact checked-out source SHA.

use std::collections::BTreeSet;
use std::env;

use serde_json::{json, Value};

const MANIFEST: &str =
    include_str!("../../../../docs/research/hlb-factorial-benchmark-v1.json");
const EXPECTED_FACTORS: [&str; 3] = ["native_hdc", "cfc_ltc", "fep_epistemic_policy"];

fn string_field<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing or non-string field: {key}"))
}

fn array_field<'a>(value: &'a Value, key: &str) -> Result<&'a Vec<Value>, String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing or non-array field: {key}"))
}

fn bool_field(value: &Value, key: &str) -> Result<bool, String> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("missing or non-boolean factor: {key}"))
}

fn validate_manifest(manifest: &Value) -> Result<(), String> {
    if string_field(manifest, "schema_version")? != "hlb-factorial-benchmark-v1" {
        return Err("unexpected schema_version".into());
    }
    if string_field(manifest, "status")? != "protocol_only" {
        return Err("protocol status must remain protocol_only until experiments exist".into());
    }
    let claim_ceiling = string_field(manifest, "claim_ceiling")?;
    if !claim_ceiling.contains("No runtime component activation")
        || !claim_ceiling.contains("No runtime")
    {
        return Err("claim ceiling must prohibit runtime/performance claims".into());
    }

    let factors = array_field(manifest, "factors")?;
    if factors.len() != EXPECTED_FACTORS.len() {
        return Err(format!(
            "expected {} factorial definitions, found {}",
            EXPECTED_FACTORS.len(),
            factors.len()
        ));
    }
    let mut factor_ids = BTreeSet::new();
    for factor in factors {
        let id = string_field(factor, "id")?;
        if !factor_ids.insert(id.to_owned()) {
            return Err(format!("duplicate factor id: {id}"));
        }
        if string_field(factor, "enabled_mechanism")?.trim().is_empty()
            || string_field(factor, "disabled_control")?.trim().is_empty()
            || string_field(factor, "adapter_status")?.trim().is_empty()
        {
            return Err(format!("factor {id} must name a mechanism, control, and adapter status"));
        }
    }
    let expected_factor_ids = EXPECTED_FACTORS
        .iter()
        .map(|id| (*id).to_owned())
        .collect::<BTreeSet<_>>();
    if factor_ids != expected_factor_ids {
        return Err("factor IDs do not match the registered HDC/CfC/FEP design".into());
    }

    let profiles = array_field(manifest, "profiles")?;
    if profiles.len() != 8 {
        return Err(format!("complete 2x2x2 factorial requires 8 profiles, found {}", profiles.len()));
    }
    let mut profile_ids = BTreeSet::new();
    let mut combinations = BTreeSet::new();
    for profile in profiles {
        let id = string_field(profile, "id")?;
        if !profile_ids.insert(id.to_owned()) {
            return Err(format!("duplicate profile id: {id}"));
        }
        if string_field(profile, "execution_status")? != "planned_not_executed" {
            return Err(format!(
                "profile {id} cannot claim execution from a plan contract"
            ));
        }

        let h = bool_field(profile, "native_hdc")?;
        let c = bool_field(profile, "cfc_ltc")?;
        let f = bool_field(profile, "fep_epistemic_policy")?;
        let expected_id = format!(
            "H{}-C{}-F{}",
            u8::from(h),
            u8::from(c),
            u8::from(f)
        );
        if id != expected_id {
            return Err(format!("profile {id} does not match factor values ({expected_id})"));
        }
        if !combinations.insert((h, c, f)) {
            return Err(format!("duplicate factorial combination: {id}"));
        }
    }
    let expected_combinations = (0_u8..8)
        .map(|bits| {
            (
                bits & 0b100 != 0,
                bits & 0b010 != 0,
                bits & 0b001 != 0,
            )
        })
        .collect::<BTreeSet<_>>();
    if combinations != expected_combinations {
        return Err("factorial matrix is incomplete".into());
    }

    let tasks = array_field(manifest, "tasks")?;
    if tasks.len() < 3 {
        return Err("at least three task families are required".into());
    }
    let mut task_ids = BTreeSet::new();
    for task in tasks {
        let id = string_field(task, "id")?;
        if !task_ids.insert(id.to_owned()) {
            return Err(format!("duplicate task id: {id}"));
        }
        if string_field(task, "environment")?.trim().is_empty() {
            return Err(format!("task {id} has no environment contract"));
        }
        if array_field(task, "primary_outcomes")?.is_empty() {
            return Err(format!("task {id} has no primary outcome"));
        }
    }

    let baselines = array_field(manifest, "baselines")?;
    let baseline_ids = baselines
        .iter()
        .map(|baseline| string_field(baseline, "id").map(str::to_owned))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let required_baselines = [
        "hdc_prototype_or_online",
        "fixed_seed_random_projection",
        "svm_shared_input_features",
        "rvfl_shared_input_features",
        "parameter_matched_gru",
        "simple_leaky_state",
        "memoryless_model",
        "pragmatic_greedy_policy",
        "random_exploration_policy",
        "bounded_information_gain_policy",
        "perfect_information_oracle",
    ];
    for baseline in required_baselines {
        if !baseline_ids.contains(baseline) {
            return Err(format!("missing required comparator identity: {baseline}"));
        }
    }

    let seeds = manifest
        .get("seeds")
        .ok_or_else(|| "missing seeds".to_owned())?;
    let development = array_field(seeds, "development")?
        .iter()
        .map(|v| v.as_u64().ok_or_else(|| "development seed must be an integer".to_owned()))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let held_out = array_field(seeds, "held_out")?
        .iter()
        .map(|v| v.as_u64().ok_or_else(|| "held-out seed must be an integer".to_owned()))
        .collect::<Result<BTreeSet<_>, _>>()?;
    if development.len() != 10 || held_out.len() != 20 {
        return Err(format!(
            "expected 10 unique development seeds and 20 unique held-out seeds; found {} and {}",
            development.len(),
            held_out.len()
        ));
    }
    if !development.is_disjoint(&held_out) {
        return Err("development and held-out seeds overlap".into());
    }

    let budget = manifest
        .get("tuning_budget")
        .ok_or_else(|| "missing tuning_budget".to_owned())?;
    if budget
        .get("max_candidate_configs_per_profile_task")
        .and_then(Value::as_u64)
        .filter(|n| *n > 0)
        .is_none()
        || budget.get("same_candidate_count_for_all_profiles").and_then(Value::as_bool)
            != Some(true)
        || budget.get("aggregate_validation_compute_must_be_recorded").and_then(Value::as_bool)
            != Some(true)
    {
        return Err("tuning budget must be bounded, matched, and recorded".into());
    }

    let hard_gates = array_field(manifest, "hard_gates")?;
    if hard_gates.len() < 5 {
        return Err("required fail-closed integrity gates are missing".into());
    }
    for reference in array_field(manifest, "research_references")? {
        let url = string_field(reference, "url")?;
        if !url.starts_with("https://") {
            return Err(format!("research reference must use HTTPS: {url}"));
        }
    }

    Ok(())
}

fn valid_source_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn main() {
    let source_sha = env::var("SYMTHAEA_SOURCE_SHA")
        .expect("SYMTHAEA_SOURCE_SHA must bind the receipt to the exact Git commit");
    assert!(
        valid_source_sha(&source_sha),
        "SYMTHAEA_SOURCE_SHA must be a 40-character hexadecimal Git SHA"
    );

    let manifest: Value = serde_json::from_str(MANIFEST).expect("manifest JSON must parse");
    validate_manifest(&manifest).unwrap_or_else(|err| panic!("manifest contract failed: {err}"));

    let receipt = json!({
        "receipt_version": "hlb-factorial-benchmark-contract-receipt-v1",
        "source_sha": source_sha,
        "manifest_blake3": blake3::hash(MANIFEST.as_bytes()).to_hex().to_string(),
        "factorial_profiles_validated": 8,
        "tasks_in_contract": manifest["tasks"].as_array().map_or(0, Vec::len),
        "baseline_identities": manifest["baselines"].as_array().map_or(0, Vec::len),
        "status": "protocol_contract_validated",
        "runtime_profiles_executed": false,
        "performance_results_available": false,
        "sota_claim_supported": false
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&receipt).expect("receipt serialization must succeed")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        serde_json::from_str(MANIFEST).expect("fixture JSON")
    }

    #[test]
    fn checked_in_factorial_contract_is_complete() {
        validate_manifest(&fixture()).expect("valid contract");
    }

    #[test]
    fn missing_profile_fails_closed() {
        let mut value = fixture();
        value["profiles"].as_array_mut().unwrap().pop();
        assert!(validate_manifest(&value).is_err());
    }

    #[test]
    fn duplicate_profile_fails_closed() {
        let mut value = fixture();
        let first = value["profiles"][0].clone();
        value["profiles"].as_array_mut().unwrap().push(first);
        assert!(validate_manifest(&value).is_err());
    }

    #[test]
    fn profile_name_must_match_factor_values() {
        let mut value = fixture();
        value["profiles"][0]["fep_epistemic_policy"] = Value::Bool(true);
        assert!(validate_manifest(&value).is_err());
    }

    #[test]
    fn development_and_held_out_seeds_must_be_disjoint() {
        let mut value = fixture();
        value["seeds"]["held_out"][0] = json!(11);
        assert!(validate_manifest(&value).is_err());
    }

    #[test]
    fn missing_strong_baseline_fails_closed() {
        let mut value = fixture();
        let remaining = value["baselines"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|baseline| baseline["id"] != "parameter_matched_gru")
            .cloned()
            .collect();
        value["baselines"] = Value::Array(remaining);
        assert!(validate_manifest(&value).is_err());
    }

    #[test]
    fn protocol_cannot_claim_measured_performance() {
        let mut value = fixture();
        value["status"] = Value::String("benchmark_complete".into());
        assert!(validate_manifest(&value).is_err());
    }

    #[test]
    fn source_binding_requires_exact_sha_shape() {
        assert!(valid_source_sha("a3ba137d4c3900e2ef6731a4ca5065dc2c8288c3"));
        assert!(!valid_source_sha("queued"));
        assert!(!valid_source_sha("a3ba"));
    }
}
