// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::{json, Map, Value};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use symthaea_energy_benchmark_zero::BandgapTarget;
use symthaea_energy_benchmark_zero_promotion::{
    run_promotion_assessment_from_official_bytes, PromotionCriteria, PromotionPlan,
    CHRONOLOGY_DISCLOSURE,
};

const CRITERIA_FIELDS: [&str; 9] = [
    "min_micro_precision_delta",
    "min_micro_recall_delta",
    "max_weighted_mae_delta_ev",
    "max_macro_regret_delta_ev",
    "max_per_fold_top_k_hit_drop",
    "max_per_fold_mae_increase_ev",
    "max_per_fold_regret_increase_ev",
    "min_folds_hits_ge_blind_mean",
    "min_folds_regret_le_blind_median",
];

fn main() {
    if let Err(error) = run_cli() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

fn run_cli() -> Result<(), String> {
    let mut args = env::args();
    let program = args
        .next()
        .unwrap_or_else(|| "energy-benchmark-zero-promotion".into());
    let command = args.next().ok_or_else(|| usage(&program))?;

    match command.as_str() {
        "plan" => {
            let target_min = args.next().ok_or_else(|| usage(&program))?;
            let target_max = args.next().ok_or_else(|| usage(&program))?;
            let top_k = args.next().ok_or_else(|| usage(&program))?;
            let criteria_path = args.next().ok_or_else(|| usage(&program))?;
            if args.next().is_some() {
                return Err(usage(&program));
            }

            let target = parse_target(&target_min, &target_max)?;
            let top_k = parse_top_k(&top_k)?;
            let criteria = read_criteria(&criteria_path)?;
            let plan = PromotionPlan::new(target, top_k, criteria)
                .map_err(|error| error.to_string())?;
            let plan_sha256 = plan.sha256().map_err(|error| error.to_string())?;
            let output = json!({
                "plan": plan,
                "plan_sha256": plan_sha256,
                "chronology_disclosure": CHRONOLOGY_DISCLOSURE,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&output).map_err(|error| error.to_string())?
            );
            Ok(())
        }
        "run" => {
            let artifact_path = args.next().ok_or_else(|| usage(&program))?;
            let target_min = args.next().ok_or_else(|| usage(&program))?;
            let target_max = args.next().ok_or_else(|| usage(&program))?;
            let top_k = args.next().ok_or_else(|| usage(&program))?;
            let criteria_path = args.next().ok_or_else(|| usage(&program))?;
            let registration_evidence_ref = args.next();
            if args.next().is_some() {
                return Err(usage(&program));
            }

            let target = parse_target(&target_min, &target_max)?;
            let top_k = parse_top_k(&top_k)?;
            let criteria = read_criteria(&criteria_path)?;
            let plan = PromotionPlan::new(target, top_k, criteria)
                .map_err(|error| error.to_string())?;
            let artifact = fs::read(&artifact_path)
                .map_err(|error| format!("failed to read benchmark artifact: {error}"))?;
            let receipt = run_promotion_assessment_from_official_bytes(
                &artifact,
                plan,
                registration_evidence_ref,
            )
            .map_err(|error| error.to_string())?;

            // Deliberately omit artifact_path and criteria_path from evidence.
            println!(
                "{}",
                receipt.to_json_pretty().map_err(|error| error.to_string())?
            );
            Ok(())
        }
        _ => Err(usage(&program)),
    }
}

fn parse_target(min: &str, max: &str) -> Result<BandgapTarget, String> {
    let min_ev: f64 = min
        .parse()
        .map_err(|_| "target-min-eV must be a finite number".to_owned())?;
    let max_ev: f64 = max
        .parse()
        .map_err(|_| "target-max-eV must be a finite number".to_owned())?;
    BandgapTarget::new(min_ev, max_ev).map_err(|error| error.to_string())
}

fn parse_top_k(value: &str) -> Result<usize, String> {
    let top_k: usize = value
        .parse()
        .map_err(|_| "top-k must be a positive integer".to_owned())?;
    if top_k == 0 {
        return Err("top-k must be positive".into());
    }
    Ok(top_k)
}

fn read_criteria(path: &str) -> Result<PromotionCriteria, String> {
    let bytes = fs::read(path).map_err(|error| format!("failed to read criteria JSON: {error}"))?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid criteria JSON: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "criteria JSON must be one object".to_owned())?;
    validate_exact_fields(object)?;

    Ok(PromotionCriteria {
        min_micro_precision_delta: required_f64(object, "min_micro_precision_delta")?,
        min_micro_recall_delta: required_f64(object, "min_micro_recall_delta")?,
        max_weighted_mae_delta_ev: required_f64(object, "max_weighted_mae_delta_ev")?,
        max_macro_regret_delta_ev: required_f64(object, "max_macro_regret_delta_ev")?,
        max_per_fold_top_k_hit_drop: required_usize(object, "max_per_fold_top_k_hit_drop")?,
        max_per_fold_mae_increase_ev: required_f64(object, "max_per_fold_mae_increase_ev")?,
        max_per_fold_regret_increase_ev: required_f64(
            object,
            "max_per_fold_regret_increase_ev",
        )?,
        min_folds_hits_ge_blind_mean: required_usize(object, "min_folds_hits_ge_blind_mean")?,
        min_folds_regret_le_blind_median: required_usize(
            object,
            "min_folds_regret_le_blind_median",
        )?,
    })
}

fn validate_exact_fields(object: &Map<String, Value>) -> Result<(), String> {
    let expected: BTreeSet<&str> = CRITERIA_FIELDS.into_iter().collect();
    let actual: BTreeSet<&str> = object.keys().map(String::as_str).collect();
    if actual != expected {
        let missing: Vec<&str> = expected.difference(&actual).copied().collect();
        let unknown: Vec<&str> = actual.difference(&expected).copied().collect();
        return Err(format!(
            "criteria JSON field mismatch; missing={missing:?}, unknown={unknown:?}"
        ));
    }
    Ok(())
}

fn required_f64(object: &Map<String, Value>, key: &str) -> Result<f64, String> {
    object
        .get(key)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| format!("criteria field {key:?} must be a finite JSON number"))
}

fn required_usize(object: &Map<String, Value>, key: &str) -> Result<usize, String> {
    let value = object
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("criteria field {key:?} must be a non-negative integer"))?;
    usize::try_from(value).map_err(|_| format!("criteria field {key:?} exceeds usize range"))
}

fn usage(program: &str) -> String {
    format!(
        "usage:\n  {program} plan <target-min-eV> <target-max-eV> <top-k> <criteria.json>\n  {program} run <matbench_expt_gap.json.gz> <target-min-eV> <target-max-eV> <top-k> <criteria.json> [registration-evidence-ref]"
    )
}
