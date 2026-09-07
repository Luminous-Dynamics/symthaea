// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Preregistered inferential analysis of the protocol/result-bound repeated-shock capsule.
//!
//! This example does not rerun the simulation. It consumes the independently verified results
//! JSON emitted by `evolvability_repeated_shock_sweep` and applies exactly one primary
//! inferential test:
//!
//! ```text
//! selected FromParent vs mutation-enabled RandomPeer
//! primary outcome = deficit_area_advantage_candidate_minus_reference
//! alternative     = positive directional tendency
//! test            = exact one-sided paired sign test
//! alpha           = 0.05
//! panel           = exact seeds [1,2,3,4,5,6,7,8]
//! missingness     = no inferential conclusion unless all eight primary outcomes are available
//! ties            = exact zeroes excluded from the binomial denominator
//! ```
//!
//! The four-dimensional Pareto reports remain descriptive. No secondary p-values are produced.

use sha2::{Digest, Sha256};
use std::{env, fs};
use symthaea_alife::exact_positive_sign_test;

const EXPECTED_SEEDS: &[u64] = &[1, 2, 3, 4, 5, 6, 7, 8];
const PRIMARY_METRIC: &str = "deficit_area_advantage_candidate_minus_reference";

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()
}

fn digest_json(value: &serde_json::Value) -> (String, String) {
    let bytes = serde_json::to_vec(value).expect("inference JSON must serialize");
    let digest = sha256_hex(&bytes);
    let json = String::from_utf8(bytes).expect("serde_json output is UTF-8");
    (digest, json)
}

fn main() {
    let mut args = env::args().skip(1);
    let results_path = args.next().expect("usage: inference <results.json> <results.sha256>");
    let results_sha_path = args.next().expect("usage: inference <results.json> <results.sha256>");
    assert!(args.next().is_none(), "unexpected extra arguments");

    let results_bytes = fs::read(&results_path).expect("read results JSON");
    let results_sha256 = fs::read_to_string(&results_sha_path)
        .expect("read verified result digest")
        .trim()
        .to_owned();
    let observed_results_sha256 = sha256_hex(&results_bytes);
    assert_eq!(
        observed_results_sha256, results_sha256,
        "inference input bytes must match the supplied verified result digest"
    );
    let results: serde_json::Value =
        serde_json::from_slice(&results_bytes).expect("parse results JSON");

    assert_eq!(
        results.get("schema").and_then(serde_json::Value::as_str),
        Some("symthaea.alife.repeated-shock.results.v1"),
        "inference accepts only the v1 verified result schema"
    );

    let seed_results = results
        .get("seed_results")
        .and_then(serde_json::Value::as_array)
        .expect("seed_results array");
    assert_eq!(seed_results.len(), EXPECTED_SEEDS.len(), "exact fixed panel size");

    let mut primary_values = Vec::with_capacity(EXPECTED_SEEDS.len());
    let mut unavailable_seeds = Vec::new();

    for (&expected_seed, entry) in EXPECTED_SEEDS.iter().zip(seed_results) {
        let seed = entry
            .get("seed")
            .and_then(serde_json::Value::as_u64)
            .expect("numeric seed");
        assert_eq!(seed, expected_seed, "fixed seed order must not drift");

        let relative = entry
            .get("selected_vs_random_peer_relative")
            .and_then(serde_json::Value::as_object)
            .expect("selected-vs-random-peer result object");
        match relative.get("status").and_then(serde_json::Value::as_str) {
            Some("ok") => {
                let value = relative
                    .get(PRIMARY_METRIC)
                    .and_then(serde_json::Value::as_f64)
                    .expect("finite primary metric when status=ok");
                assert!(value.is_finite(), "primary metric must be finite");
                primary_values.push(value);
            }
            Some("unavailable") => unavailable_seeds.push(seed),
            other => panic!("unexpected relative-result status for seed {seed}: {other:?}"),
        }
    }

    let inference = if unavailable_seeds.is_empty() {
        let test = exact_positive_sign_test(&primary_values).expect("finite eight-seed primary panel");
        serde_json::json!({
            "status": "available",
            "test": {
                "name": "exact_one_sided_paired_sign_test",
                "alternative": "selected_from_parent_primary_metric_gt_random_peer_primary_metric",
                "total_observations": test.total_observations,
                "positive": test.positive,
                "negative": test.negative,
                "ties": test.ties,
                "non_ties": test.non_ties,
                "tail_numerator": test.tail_numerator.to_string(),
                "denominator": test.denominator.to_string(),
                "one_sided_p_value": test.one_sided_p_value,
                "alpha_numerator": 1,
                "alpha_denominator": 20,
                "reject_at_alpha_0_05": test.reject_at_alpha_0_05,
            }
        })
    } else {
        serde_json::json!({
            "status": "unavailable",
            "reason": "predeclared_missingness_policy_requires_all_eight_primary_outcomes",
            "unavailable_seeds": unavailable_seeds,
        })
    };

    let evidence = serde_json::json!({
        "schema": "symthaea.alife.repeated-shock.inference.v1",
        "results_sha256": results_sha256,
        "preregistration": {
            "comparison": "selected_from_parent_vs_mutation_enabled_random_peer",
            "primary_metric": PRIMARY_METRIC,
            "primary_metric_orientation": "positive_favors_selected_from_parent",
            "test": "exact_one_sided_paired_sign_test",
            "alternative": "positive_directional_tendency",
            "alpha_numerator": 1,
            "alpha_denominator": 20,
            "expected_seed_panel": EXPECTED_SEEDS,
            "missingness_policy": "require_all_eight_primary_outcomes",
            "tie_policy": "exact_zero_excluded_from_binomial_denominator",
            "secondary_metric_p_values": false,
        },
        "inference": inference,
        "claim_boundary": {
            "tests_fixed_stochastic_seed_panel": true,
            "biological_population_generalization": false,
            "lineage_mechanism_established": false,
            "individual_learning_or_memory_established": false,
            "random_peer_exact_common_random_number_replay": false,
        }
    });

    let (inference_sha256, inference_json) = digest_json(&evidence);
    println!("inference_sha256={inference_sha256}");
    println!("inference_json={inference_json}");
}
