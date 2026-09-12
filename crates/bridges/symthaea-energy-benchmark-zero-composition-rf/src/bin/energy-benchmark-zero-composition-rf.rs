// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::env;
use std::fs;
use symthaea_energy_benchmark_zero::BandgapTarget;
use symthaea_energy_benchmark_zero_composition_rf::run_composition_rf_benchmark_from_official_bytes;
use symthaea_matbench_folds::FoldIndex;

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
        .unwrap_or_else(|| "energy-benchmark-zero-composition-rf".into());

    let artifact_path = args.next().ok_or_else(|| usage(&program))?;
    let fold: u8 = args
        .next()
        .ok_or_else(|| usage(&program))?
        .parse()
        .map_err(|_| "fold must be an integer in 0..4".to_owned())?;
    let target_min_ev: f64 = args
        .next()
        .ok_or_else(|| usage(&program))?
        .parse()
        .map_err(|_| "target-min-eV must be a finite number".to_owned())?;
    let target_max_ev: f64 = args
        .next()
        .ok_or_else(|| usage(&program))?
        .parse()
        .map_err(|_| "target-max-eV must be a finite number".to_owned())?;
    let top_k: usize = args
        .next()
        .ok_or_else(|| usage(&program))?
        .parse()
        .map_err(|_| "top-k must be a positive integer".to_owned())?;

    if args.next().is_some() {
        return Err(usage(&program));
    }

    let fold = FoldIndex::new(fold).map_err(|error| error.to_string())?;
    let target = BandgapTarget::new(target_min_ev, target_max_ev)
        .map_err(|error| error.to_string())?;
    let bytes = fs::read(&artifact_path)
        .map_err(|error| format!("failed to read benchmark artifact: {error}"))?;

    let receipt = run_composition_rf_benchmark_from_official_bytes(
        &bytes,
        fold,
        target,
        top_k,
    )
    .map_err(|error| error.to_string())?;

    // Deliberately omit host-local artifact_path from the emitted evidence.
    println!(
        "{}",
        receipt.to_json_pretty().map_err(|error| error.to_string())?
    );
    Ok(())
}

fn usage(program: &str) -> String {
    format!(
        "usage: {program} <matbench_expt_gap.json.gz> <fold:0..4> <target-min-eV> <target-max-eV> <top-k>"
    )
}
