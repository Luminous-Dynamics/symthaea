// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Network-free CLI for one leakage-qualified Benchmark Zero baseline run.

use std::env;
use std::fs;
use std::process::ExitCode;
use symthaea_energy_benchmark_zero::BandgapTarget;
use symthaea_energy_benchmark_zero_baseline::run_baseline_benchmark_from_official_bytes;
use symthaea_matbench_folds::FoldIndex;

fn usage(program: &str) -> String {
    format!(
        "usage: {program} <matbench_expt_gap.json.gz> <fold:0..4> <target-min-eV> <target-max-eV> <top-k>"
    )
}

fn run() -> Result<(), String> {
    let mut args = env::args();
    let program = args
        .next()
        .unwrap_or_else(|| "energy-benchmark-zero-baseline".to_owned());
    let artifact_path = args.next().ok_or_else(|| usage(&program))?;
    let fold_raw = args.next().ok_or_else(|| usage(&program))?;
    let min_raw = args.next().ok_or_else(|| usage(&program))?;
    let max_raw = args.next().ok_or_else(|| usage(&program))?;
    let top_k_raw = args.next().ok_or_else(|| usage(&program))?;
    if args.next().is_some() {
        return Err(usage(&program));
    }

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

    let bytes = fs::read(&artifact_path)
        .map_err(|error| format!("failed to read {artifact_path:?}: {error}"))?;
    let receipt = run_baseline_benchmark_from_official_bytes(&bytes, fold, target, top_k)
        .map_err(|error| error.to_string())?;

    // Deliberately omit the local input path. Scientific identity is bound to
    // accepted artifact bytes/digests, not host-specific filesystem layout.
    println!(
        "{}",
        receipt
            .to_json_pretty()
            .map_err(|error| format!("failed to encode receipt: {error}"))?
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("energy-benchmark-zero-baseline: {error}");
            ExitCode::FAILURE
        }
    }
}
