// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explicit-policy snapshot comparison command.
//!
//! No metric direction or threshold is inferred. The caller must supply:
//! baseline snapshot, current snapshot, policy manifest, warning fraction, and
//! critical fraction.
//!
//! Exit status:
//! - 0: comparison completed with no blocking integrity failure;
//! - 1: comparison completed and contains a blocking regression/integrity state;
//! - 2: setup/input/manifest/comparison-contract failure.

use serde_json::json;
use std::{env, fs, path::Path};
use symthaea_psych_bench::harness::snapshot::RegressionSnapshot;
use symthaea_psych_bench::regression_contract::RegressionThresholds;
use symthaea_psych_bench::regression_policy_manifest::{
    MetricPolicyManifest, compare_snapshots_with_manifest,
};

fn main() {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 6 {
        setup_failure(
            "usage",
            "usage: policy_regression_compare <baseline.json> <current.json> <policy-manifest.json> <warning-fraction> <critical-fraction>",
        );
    }

    let baseline_text = read_text(&args[1], "baseline_read");
    let current_text = read_text(&args[2], "current_read");
    let manifest_text = read_text(&args[3], "manifest_read");

    let baseline = RegressionSnapshot::from_json(&baseline_text)
        .unwrap_or_else(|error| setup_failure("baseline_parse", &error.to_string()));
    let current = RegressionSnapshot::from_json(&current_text)
        .unwrap_or_else(|error| setup_failure("current_parse", &error.to_string()));
    let manifest: MetricPolicyManifest = serde_json::from_str(&manifest_text)
        .unwrap_or_else(|error| setup_failure("manifest_parse", &error.to_string()));

    let warning_fraction = parse_fraction(&args[4], "warning_fraction");
    let critical_fraction = parse_fraction(&args[5], "critical_fraction");
    let thresholds = RegressionThresholds::new(warning_fraction, critical_fraction)
        .unwrap_or_else(|error| setup_failure("threshold_policy", &error));

    let semantic_manifest_digest = manifest
        .digest_hex()
        .unwrap_or_else(|error| setup_failure("manifest_structure", &format!("{error:?}")));

    let report = compare_snapshots_with_manifest(&baseline, &current, &manifest, thresholds)
        .unwrap_or_else(|error| setup_failure("comparison_contract", &format!("{error:?}")));

    let blocking = report.has_blocking_integrity_failure();
    let receipt = json!({
        "schema_version": "psych-policy-regression-cli-v1",
        "baseline_file_digest": blake3::hash(baseline_text.as_bytes()).to_hex().to_string(),
        "current_file_digest": blake3::hash(current_text.as_bytes()).to_hex().to_string(),
        "manifest_file_digest": blake3::hash(manifest_text.as_bytes()).to_hex().to_string(),
        "semantic_manifest_digest": semantic_manifest_digest,
        "manifest_id": manifest.manifest_id,
        "manifest_revision": manifest.revision,
        "blocking_integrity_failure": blocking,
        "report": report,
    });

    println!(
        "{}",
        serde_json::to_string_pretty(&receipt).expect("comparison receipt must serialize")
    );
    std::process::exit(if blocking { 1 } else { 0 });
}

fn read_text(path: &str, kind: &str) -> String {
    fs::read_to_string(Path::new(path))
        .unwrap_or_else(|error| setup_failure(kind, &format!("{path}: {error}")))
}

fn parse_fraction(value: &str, kind: &str) -> f64 {
    value
        .parse::<f64>()
        .unwrap_or_else(|error| setup_failure(kind, &error.to_string()))
}

fn setup_failure(kind: &str, detail: &str) -> ! {
    eprintln!("POLICY_REGRESSION_SETUP_ERROR kind={kind} detail={detail}");
    std::process::exit(2);
}
