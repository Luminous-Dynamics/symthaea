// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Network-free plan/run CLI for controlled Energy Benchmark Zero.

use serde_json::json;
use std::env;
use std::fs;
use std::process::ExitCode;
use symthaea_energy_benchmark_zero::BandgapTarget;
use symthaea_energy_benchmark_zero_controls::{
    run_controlled_benchmark_from_official_bytes, BenchmarkPlan, PLAN_CHRONOLOGY_DISCLOSURE,
};
use symthaea_matbench_folds::FoldIndex;

fn usage(program: &str) -> String {
    format!(
        "usage:\n  {program} plan <fold:0..4> <target-min-eV> <target-max-eV> <top-k>\n  {program} run <matbench_expt_gap.json.gz> <fold:0..4> <target-min-eV> <target-max-eV> <top-k> [registration-evidence-ref]"
    )
}

fn parse_plan(
    fold_raw: &str,
    min_raw: &str,
    max_raw: &str,
    top_k_raw: &str,
) -> Result<BenchmarkPlan, String> {
    let fold_value = fold_raw
        .parse::<u8>()
        .map_err(|_| format!("invalid fold {fold_raw:?}; expected integer 0..4"))?;
    let fold = FoldIndex::new(fold_value).map_err(|error| error.to_string())?;
    let min_ev = min_raw
        .parse::<f64>()
        .map_err(|_| format!("invalid target minimum {min_raw:?}"))?;
    let max_ev = max_raw
        .parse::<f64>()
        .map_err(|_| format!("invalid target maximum {max_raw:?}"))?;
    let target = BandgapTarget::new(min_ev, max_ev).map_err(|error| error.to_string())?;
    let top_k = top_k_raw
        .parse::<usize>()
        .map_err(|_| format!("invalid top-k {top_k_raw:?}; expected positive integer"))?;
    BenchmarkPlan::new(fold, target, top_k).map_err(|error| error.to_string())
}

fn run_cli() -> Result<(), String> {
    let mut args = env::args();
    let program = args
        .next()
        .unwrap_or_else(|| "energy-benchmark-zero-controls".to_owned());
    let mode = args.next().ok_or_else(|| usage(&program))?;

    match mode.as_str() {
        "plan" => {
            let fold = args.next().ok_or_else(|| usage(&program))?;
            let min = args.next().ok_or_else(|| usage(&program))?;
            let max = args.next().ok_or_else(|| usage(&program))?;
            let top_k = args.next().ok_or_else(|| usage(&program))?;
            if args.next().is_some() {
                return Err(usage(&program));
            }
            let plan = parse_plan(&fold, &min, &max, &top_k)?;
            let output = json!({
                "plan": plan,
                "plan_sha256": plan.sha256().map_err(|error| error.to_string())?,
                "chronology_disclosure": PLAN_CHRONOLOGY_DISCLOSURE,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&output)
                    .map_err(|error| format!("failed to encode plan: {error}"))?
            );
            Ok(())
        }
        "run" => {
            let artifact_path = args.next().ok_or_else(|| usage(&program))?;
            let fold = args.next().ok_or_else(|| usage(&program))?;
            let min = args.next().ok_or_else(|| usage(&program))?;
            let max = args.next().ok_or_else(|| usage(&program))?;
            let top_k = args.next().ok_or_else(|| usage(&program))?;
            let registration_evidence_ref = args.next();
            if args.next().is_some() {
                return Err(usage(&program));
            }

            let plan = parse_plan(&fold, &min, &max, &top_k)?;
            let bytes = fs::read(&artifact_path)
                .map_err(|error| format!("failed to read {artifact_path:?}: {error}"))?;
            let receipt = run_controlled_benchmark_from_official_bytes(
                &bytes,
                plan,
                registration_evidence_ref,
            )
            .map_err(|error| error.to_string())?;

            // Deliberately omit the local input path. Artifact identity is
            // established by the pinned source digest in the evidence chain.
            println!(
                "{}",
                receipt
                    .to_json_pretty()
                    .map_err(|error| format!("failed to encode receipt: {error}"))?
            );
            Ok(())
        }
        _ => Err(usage(&program)),
    }
}

fn main() -> ExitCode {
    match run_cli() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("energy-benchmark-zero-controls: {error}");
            ExitCode::FAILURE
        }
    }
}
