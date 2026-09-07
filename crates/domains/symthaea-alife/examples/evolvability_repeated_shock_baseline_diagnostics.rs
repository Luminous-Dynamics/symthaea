// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Descriptive baseline-drift analysis downstream of a verified repeated-shock result capsule.
//!
//! This example does not rerun the simulation, alter recovery metrics, exclude seeds, or compute
//! p-values. It exposes whether each condition entered shock 2 from a substantially different
//! observed population baseline than shock 1, which is context needed when interpreting metrics
//! normalized to each shock's own baseline.

use sha2::{Digest, Sha256};
use std::{env, fs};
use symthaea_alife::{
    RepeatedShockBaselineShiftV1, compare_relative_baseline_shift,
    repeated_shock_baseline_shift,
};

const EXPECTED_SEEDS: &[u64] = &[1, 2, 3, 4, 5, 6, 7, 8];

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn digest_json(value: &serde_json::Value) -> (String, String) {
    let bytes = serde_json::to_vec(value).expect("diagnostic JSON must serialize");
    let digest = sha256_hex(&bytes);
    let json = String::from_utf8(bytes).expect("serde_json output is UTF-8");
    (digest, json)
}

fn shift_json(shift: RepeatedShockBaselineShiftV1) -> serde_json::Value {
    serde_json::json!({
        "status": "ok",
        "first_baseline_mean_observed_population": shift.first_baseline_mean_observed_population,
        "second_baseline_mean_observed_population": shift.second_baseline_mean_observed_population,
        "second_to_first_ratio": shift.second_to_first_ratio,
        "fractional_change": shift.fractional_change,
        "log_ratio": shift.log_ratio,
    })
}

fn shift_from_transfer(transfer: &serde_json::Value) -> Result<RepeatedShockBaselineShiftV1, String> {
    let object = transfer
        .as_object()
        .ok_or_else(|| "transfer must be an object".to_owned())?;
    match object.get("status").and_then(serde_json::Value::as_str) {
        Some("ok") => {
            let first = object
                .get("first_baseline_mean_observed_population")
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| "missing finite first baseline".to_owned())?;
            let second = object
                .get("second_baseline_mean_observed_population")
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| "missing finite second baseline".to_owned())?;
            repeated_shock_baseline_shift(first, second)
                .map_err(|error| format!("baseline diagnostic unavailable: {error:?}"))
        }
        Some("unavailable") => Err("source repeated-shock transfer unavailable".to_owned()),
        other => Err(format!("unexpected transfer status: {other:?}")),
    }
}

fn shift_result_json(result: &Result<RepeatedShockBaselineShiftV1, String>) -> serde_json::Value {
    match result {
        Ok(shift) => shift_json(*shift),
        Err(error) => serde_json::json!({"status": "unavailable", "error": error}),
    }
}

fn relative_result_json(
    reference: &Result<RepeatedShockBaselineShiftV1, String>,
    candidate: &Result<RepeatedShockBaselineShiftV1, String>,
) -> serde_json::Value {
    match (reference, candidate) {
        (Ok(reference), Ok(candidate)) => match compare_relative_baseline_shift(reference, candidate) {
            Ok(relative) => serde_json::json!({
                "status": "ok",
                "candidate_to_reference_ratio_of_ratios": relative.candidate_to_reference_ratio_of_ratios,
                "log_ratio_advantage": relative.log_ratio_advantage,
            }),
            Err(error) => serde_json::json!({
                "status": "unavailable",
                "error": format!("relative baseline diagnostic unavailable: {error:?}"),
            }),
        },
        _ => serde_json::json!({
            "status": "unavailable",
            "error": "one or both source baseline diagnostics unavailable",
        }),
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let results_path = args
        .next()
        .expect("usage: baseline-diagnostics <results.json> <results.sha256>");
    let results_sha_path = args
        .next()
        .expect("usage: baseline-diagnostics <results.json> <results.sha256>");
    assert!(args.next().is_none(), "unexpected extra arguments");

    let results_bytes = fs::read(&results_path).expect("read results JSON bytes");
    let supplied_results_sha256 = fs::read_to_string(&results_sha_path)
        .expect("read verified result digest")
        .trim()
        .to_owned();
    let observed_results_sha256 = sha256_hex(&results_bytes);
    assert_eq!(
        observed_results_sha256, supplied_results_sha256,
        "diagnostic input bytes must match the supplied verified result digest"
    );

    let results: serde_json::Value =
        serde_json::from_slice(&results_bytes).expect("parse results JSON");
    assert_eq!(
        results.get("schema").and_then(serde_json::Value::as_str),
        Some("symthaea.alife.repeated-shock.results.v1")
    );
    let seed_results = results
        .get("seed_results")
        .and_then(serde_json::Value::as_array)
        .expect("seed_results array");
    assert_eq!(seed_results.len(), EXPECTED_SEEDS.len());

    let mut diagnostics = Vec::with_capacity(EXPECTED_SEEDS.len());
    for (&expected_seed, entry) in EXPECTED_SEEDS.iter().zip(seed_results) {
        let seed = entry
            .get("seed")
            .and_then(serde_json::Value::as_u64)
            .expect("numeric seed");
        assert_eq!(seed, expected_seed, "fixed seed order must not drift");

        let frozen = shift_from_transfer(
            entry
                .get("frozen_transfer")
                .expect("frozen transfer result"),
        );
        let selected = shift_from_transfer(
            entry
                .get("selected_transfer")
                .expect("selected transfer result"),
        );
        let random_peer = shift_from_transfer(
            entry
                .get("random_peer_transfer")
                .expect("RandomPeer transfer result"),
        );

        diagnostics.push(serde_json::json!({
            "seed": seed,
            "frozen": shift_result_json(&frozen),
            "selected": shift_result_json(&selected),
            "random_peer": shift_result_json(&random_peer),
            "selected_vs_frozen": relative_result_json(&frozen, &selected),
            "random_peer_vs_frozen": relative_result_json(&frozen, &random_peer),
            "selected_vs_random_peer": relative_result_json(&random_peer, &selected),
        }));
    }

    let evidence = serde_json::json!({
        "schema": "symthaea.alife.repeated-shock.baseline-diagnostics.v1",
        "results_sha256": supplied_results_sha256,
        "diagnostic_contract": {
            "descriptive_only": true,
            "changes_primary_metric": false,
            "seed_exclusion_rule": false,
            "p_values": false,
            "absolute_population_generalization": false,
            "interpretation": "context_for_metrics_normalized_to_each_shocks_own_pre_shock_baseline",
        },
        "seed_diagnostics": diagnostics,
    });

    let (diagnostics_sha256, diagnostics_json) = digest_json(&evidence);
    println!("baseline_diagnostics_sha256={diagnostics_sha256}");
    println!("baseline_diagnostics_json={diagnostics_json}");
}
