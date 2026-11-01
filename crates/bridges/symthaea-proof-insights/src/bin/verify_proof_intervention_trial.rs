// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! CLI for validating preregistered paired proof-intervention trial plans.

#[path = "../proof_intervention_trial.rs"]
mod proof_intervention_trial;

use std::{
    env,
    fs,
    path::PathBuf,
    process::ExitCode,
};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn must_reject(label: &str, value: Value) -> Result<()> {
    if proof_intervention_trial::validate_plan(&value).is_ok() {
        bail!("negative intervention-trial control unexpectedly accepted: {label}");
    }
    println!("PITB_INTERVENTION_TRIAL_NEGATIVE_PASS: {label}");
    Ok(())
}

fn self_test(path: &PathBuf) -> Result<()> {
    let bytes = fs::read(path).with_context(|| format!("read fixture {}", path.display()))?;
    let mut value: Value =
        serde_json::from_slice(&bytes).context("parse synthetic intervention plan")?;
    let pairs = proof_intervention_trial::validate_plan(&value)?;
    let arms = value["arms"]
        .as_array()
        .context("fixture arms missing")?
        .len();
    println!(
        "PITB_INTERVENTION_TRIAL_FIXTURE_PASS: {pairs} matched pairs x {arms} arms; synthetic only"
    );

    let mut bad = value.clone();
    bad["pairs"][0]["pair_id"] = json!("0".repeat(64));
    must_reject("pair ID does not bind exact task/state/seed", bad)?;

    let mut bad = value.clone();
    let duplicated_arm = bad["pairs"][0]["execution_order"][1].clone();
    bad["pairs"][0]["execution_order"][0] = duplicated_arm;
    must_reject("execution order is not a permutation of all arms", bad)?;

    let mut bad = value.clone();
    bad["pairs"][0]["execution_order"].as_array_mut()
        .context("execution order array")?
        let _ = bad["pairs"][0]["execution_order"].as_array_mut()
            .context("execution order array")?
            .pop();
    must_reject("missing arm from paired execution", bad)?;

    let mut bad = value.clone();
    bad["execution_controls"]["discard_mutable_caches_between_arms"] = Value::Bool(false);
    must_reject("shared mutable caches across paired arms", bad)?;

    let mut bad = value.clone();
    let substituted_seed = bad["pairs"][0]["model_seed_sha256"].clone();
    bad["pairs"][1]["model_seed_sha256"] = substituted_seed;
    must_reject("pair ID fails after model seed substitution", bad)?;

    let mut bad = value.clone();
    bad["pairs"][1]["resource_budget"]["tokens"] = json!(39999);
    must_reject("resource budget differs across matched pairs", bad)?;

    let mut bad = value.clone();
    for pair in bad["pairs"].as_array_mut().context("pairs array")? {
        pair["component_id"] = json!("single-cluster");
    }
    must_reject("component identity mutation is detected by pair identity", bad)?;

    let mut bad = value.clone();
    bad["analysis_policy"]["include_not_attempted_as_failure"] = Value::Bool(false);
    must_reject("survivorship bias by excluding non-attempts", bad)?;

    let mut bad = value.clone();
    bad["arms"][0]["role"] = json!("intervention");
    must_reject("trial no longer has exactly one control", bad)?;

    let mut bad = value.clone();
    bad["analysis_policy"]["oracle_arm_is_diagnostic_only"] = Value::Bool(false);
    must_reject("oracle arm cannot enter confirmatory treatment contrast", bad)?;

    value["evaluation_split"] = json!("test");
    proof_intervention_trial::validate_plan(&value)?;
    println!("PITB_INTERVENTION_TRIAL_PASS: frozen test split accepted with unchanged predeclared assignments");
    Ok(())
}

fn run() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.as_slice() {
        [flag] if flag == "--help" || flag == "-h" => {
            println!("Usage: verify_proof_intervention_trial --verify PLAN.json | --self-test FIXTURE.json");
            Ok(())
        }
        [flag, path] if flag == "--self-test" => self_test(&PathBuf::from(path)),
        [flag, path] if flag == "--verify" => {
            let bytes = fs::read(path).with_context(|| format!("read trial plan {path}"))?;
            let value: Value = serde_json::from_slice(&bytes).context("parse trial plan JSON")?;
            let pairs = proof_intervention_trial::validate_plan(&value)?;
            println!(
                "PITB_INTERVENTION_TRIAL_PASS: trial={} pairs={} arms={} plan_sha256={}",
                value["trial_id"].as_str().unwrap_or("unknown"),
                pairs,
                value["arms"].as_array().map_or(0, Vec::len),
                hex::encode(Sha256::digest(bytes))
            );
            Ok(())
        }
        _ => bail!("expected --verify PLAN.json, --self-test FIXTURE.json, or --help"),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("PITB_INTERVENTION_TRIAL_FAIL: {error:#}");
            ExitCode::FAILURE
        }
    }
}
