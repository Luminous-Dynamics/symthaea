// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Validates the HLB HDC × CfC/LTC × FEP × WSP protocol contract.
//!
//! This example validates experiment design only. It does not instantiate or
//! train a model, run a task, apply a weight edit, measure performance, or
//! qualify a capability/SOTA claim. Runtime receipts must be produced by a
//! separately reviewed experiment runner.

use std::collections::BTreeSet;
use std::env;

use serde_json::{Value, json};

const MANIFEST: &str = include_str!("../../../../docs/research/hlb-fep-wsp-factorial-v1.json");
const EXPECTED_FACTORS: [&str; 4] = [
    "native_hdc",
    "cfc_ltc",
    "fep_epistemic_policy",
    "weight_space_programming",
];

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
    if string_field(manifest, "schema_version")? != "hlb-fep-wsp-factorial-v1" {
        return Err("unexpected schema_version".into());
    }
    if string_field(manifest, "status")? != "protocol_only" {
        return Err("protocol status must remain protocol_only until experiments exist".into());
    }
    if !string_field(manifest, "claim_ceiling")?.contains("No runtime activation") {
        return Err("claim ceiling must prohibit runtime/performance claims".into());
    }

    let factors = array_field(manifest, "factors")?;
    if factors.len() != EXPECTED_FACTORS.len() {
        return Err(format!(
            "expected {} factor definitions, found {}",
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
        for field in ["enabled_mechanism", "disabled_control", "adapter_status"] {
            if string_field(factor, field)?.trim().is_empty() {
                return Err(format!("factor {id} has empty {field}"));
            }
        }
    }
    let expected_factor_ids = EXPECTED_FACTORS
        .iter()
        .map(|id| (*id).to_owned())
        .collect::<BTreeSet<_>>();
    if factor_ids != expected_factor_ids {
        return Err("factor IDs do not match the registered HDC/CfC/FEP/WSP design".into());
    }

    let profiles = array_field(manifest, "profiles")?;
    if profiles.len() != 16 {
        return Err(format!(
            "complete 2x2x2x2 factorial requires 16 profiles, found {}",
            profiles.len()
        ));
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
                "profile {id} cannot claim execution from a protocol contract"
            ));
        }

        let h = bool_field(profile, "native_hdc")?;
        let c = bool_field(profile, "cfc_ltc")?;
        let f = bool_field(profile, "fep_epistemic_policy")?;
        let w = bool_field(profile, "weight_space_programming")?;
        let expected_id = format!(
            "H{}-C{}-F{}-W{}",
            u8::from(h),
            u8::from(c),
            u8::from(f),
            u8::from(w)
        );
        if id != expected_id {
            return Err(format!(
                "profile {id} does not match factor values ({expected_id})"
            ));
        }
        if !combinations.insert((h, c, f, w)) {
            return Err(format!("duplicate factorial combination: {id}"));
        }
    }
    let expected_combinations = (0_u8..16)
        .map(|bits| {
            (
                bits & 0b1000 != 0,
                bits & 0b0100 != 0,
                bits & 0b0010 != 0,
                bits & 0b0001 != 0,
            )
        })
        .collect::<BTreeSet<_>>();
    if combinations != expected_combinations {
        return Err("factorial matrix is incomplete".into());
    }

    let tasks = array_field(manifest, "tasks")?;
    if tasks.len() < 4 {
        return Err("at least four task families are required".into());
    }
    let mut task_ids = BTreeSet::new();
    for task in tasks {
        let id = string_field(task, "id")?;
        if !task_ids.insert(id.to_owned()) {
            return Err(format!("duplicate task id: {id}"));
        }
        if string_field(task, "environment")?.trim().is_empty()
            || array_field(task, "primary_outcomes")?.is_empty()
        {
            return Err(format!("task {id} lacks an environment or primary outcome"));
        }
    }

    let baselines = array_field(manifest, "baselines")?;
    let baseline_ids = baselines
        .iter()
        .map(|baseline| string_field(baseline, "id").map(str::to_owned))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let required_baselines = [
        "original_ltc_cfc",
        "parameter_matched_gru",
        "nialim_dfa_cfn_prdfa_cfn",
        "hdc_onlinehd_or_equivalent",
        "svm_shared_features",
        "bounded_bayesian_value_of_information",
        "matched_rl_exploration",
        "no_edit",
        "random_matched_sparsity_edit",
        "analytic_oracle_repair",
        "gradient_adam_repair",
        "black_box_coordinate_or_evolutionary_search",
        "wsp_weights_only",
        "wsp_trace_assisted",
        "swi_constrained_wsp_with_verifier",
        "perfect_information_oracle",
        "bayesian_model_reduction_structure_selection",
        "raw_flat_weight_distance_editor",
        "activation_probe_alignment_editor",
        "permutation_equivariant_weight_space_editor",
        "oracle_permutation_alignment",
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
        .map(|v| {
            v.as_u64()
                .ok_or_else(|| "development seed must be an integer".to_owned())
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    let held_out = array_field(seeds, "held_out")?
        .iter()
        .map(|v| {
            v.as_u64()
                .ok_or_else(|| "held-out seed must be an integer".to_owned())
        })
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
    for field in [
        "max_candidate_configs_per_profile_task",
        "max_weight_program_candidates_per_fault_instance",
        "forward_pass_budget_per_candidate",
    ] {
        if budget
            .get(field)
            .and_then(Value::as_u64)
            .filter(|n| *n > 0)
            .is_none()
        {
            return Err(format!(
                "tuning budget field {field} must be a positive integer"
            ));
        }
    }
    if budget
        .get("same_candidate_count_for_all_profiles")
        .and_then(Value::as_bool)
        != Some(true)
        || budget
            .get("aggregate_validation_compute_must_be_recorded")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err("candidate and validation budgets must be matched and recorded".into());
    }

    let hard_gates = array_field(manifest, "hard_gates")?;
    if hard_gates.len() < 12 {
        return Err("required fail-closed integrity gates are missing".into());
    }
    let joined_gates = hard_gates
        .iter()
        .map(|gate| gate.as_str().unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n");
    for phrase in [
        "held-out",
        "fail closed",
        "Healthy networks",
        "information gain",
        "transactional",
        "No SOTA",
    ] {
        if !joined_gates.to_lowercase().contains(&phrase.to_lowercase()) {
            return Err(format!("hard gates omit required concept: {phrase}"));
        }
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
        .expect("SYMTHAEA_SOURCE_SHA must bind the protocol receipt to the exact Git commit");
    assert!(
        valid_source_sha(&source_sha),
        "SYMTHAEA_SOURCE_SHA must be exactly 40 hexadecimal characters"
    );

    let manifest: Value =
        serde_json::from_str(MANIFEST).expect("HLB FEP/WSP factorial manifest must be valid JSON");
    validate_manifest(&manifest).expect("HLB FEP/WSP factorial contract is invalid");

    let receipt = json!({
        "schema_version": "hlb-fep-wsp-factorial-v1",
        "source_sha": source_sha,
        "protocol_status": "protocol_only",
        "profiles_validated": 16,
        "factor_count": 4,
        "development_seed_count": 10,
        "held_out_seed_count": 20,
        "runtime_execution": false,
        "performance_claims_permitted": false
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&receipt).expect("receipt serializes")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_manifest_is_a_complete_four_factor_design() {
        let manifest: Value = serde_json::from_str(MANIFEST).expect("manifest parses");
        validate_manifest(&manifest).expect("manifest validates");
    }

    #[test]
    fn protocol_never_marks_profiles_as_executed() {
        let mut manifest: Value = serde_json::from_str(MANIFEST).expect("manifest parses");
        manifest["profiles"][0]["execution_status"] = Value::String("executed".into());
        assert!(validate_manifest(&manifest).is_err());
    }

    #[test]
    fn duplicate_profile_combination_is_rejected() {
        let mut manifest: Value = serde_json::from_str(MANIFEST).expect("manifest parses");
        manifest["profiles"][15] = manifest["profiles"][0].clone();
        assert!(validate_manifest(&manifest).is_err());
    }

    #[test]
    fn development_and_holdout_seed_overlap_is_rejected() {
        let mut manifest: Value = serde_json::from_str(MANIFEST).expect("manifest parses");
        manifest["seeds"]["held_out"][0] = manifest["seeds"]["development"][0].clone();
        assert!(validate_manifest(&manifest).is_err());
    }

    #[test]
    fn source_sha_must_be_exact_hex() {
        assert!(valid_source_sha("0123456789abcdef0123456789abcdef01234567"));
        assert!(!valid_source_sha("0123456789abcdef"));
        assert!(!valid_source_sha(
            "z123456789abcdef0123456789abcdef01234567"
        ));
    }
}
