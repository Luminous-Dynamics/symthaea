// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Minimal two-phase CLI for MAG-QUAL-001 sealed-target qualification.

use std::{env, fs, path::Path, process::ExitCode};
use symthaea_materials_benchmarks::BlindBenchmarkManifest;
use symthaea_materials_qualification::{
    BlindGenerationReceipt, EvaluationEnvironment, GenerationIsolationAttestation,
    evaluate_frozen_generation, freeze_generation_receipt,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("MAG-QUAL-001 error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let Some(command) = args.first().map(String::as_str) else {
        return Err(usage().into());
    };
    match command {
        "freeze" if args.len() == 5 => freeze(&args[1], &args[2], &args[3], &args[4]),
        "evaluate" if args.len() == 7 => evaluate(
            &args[1], &args[2], &args[3], &args[4], &args[5], &args[6],
        ),
        _ => Err(usage().into()),
    }
}

fn freeze(
    manifest_path: &str,
    submission_path: &str,
    isolation_path: &str,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let manifest: BlindBenchmarkManifest = read_json(manifest_path)?;
    let submission_bytes = fs::read(submission_path)?;
    let isolation: GenerationIsolationAttestation = read_json(isolation_path)?;
    let receipt = freeze_generation_receipt(&manifest, &submission_bytes, isolation)?;
    let receipt_sha = receipt.receipt_sha256()?;
    write_json(output_path, &receipt)?;
    println!("MAG-QUAL-001 Phase A frozen");
    println!("submission_sha256={}", receipt.submission_sha256);
    println!("generation_receipt_sha256={receipt_sha}");
    println!("sealed_targets_sha256={}", receipt.sealed_targets_sha256);
    Ok(())
}

fn evaluate(
    manifest_path: &str,
    submission_path: &str,
    generation_receipt_path: &str,
    sealed_targets_path: &str,
    evaluation_environment_path: &str,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let manifest: BlindBenchmarkManifest = read_json(manifest_path)?;
    let submission_bytes = fs::read(submission_path)?;
    let generation: BlindGenerationReceipt = read_json(generation_receipt_path)?;
    let sealed_targets = fs::read(sealed_targets_path)?;
    let environment: EvaluationEnvironment = read_json(evaluation_environment_path)?;
    let receipt = evaluate_frozen_generation(
        &manifest,
        &submission_bytes,
        &generation,
        &sealed_targets,
        &environment,
    )?;
    let receipt_sha = receipt.receipt_sha256()?;
    write_json(output_path, &receipt)?;
    println!("MAG-QUAL-001 Phase B evaluated");
    println!("generation_receipt_sha256={}", receipt.generation_receipt_sha256);
    println!("evaluation_receipt_sha256={receipt_sha}");
    println!("scorecard_sha256={}", receipt.scorecard_sha256);
    println!("coverage_fraction={:.6}", receipt.scorecard.coverage_fraction);
    if let Some(mae) = receipt.scorecard.scalar_mae {
        println!("scalar_mae={mae:.12}");
    }
    if let Some(rmse) = receipt.scorecard.scalar_rmse {
        println!("scalar_rmse={rmse:.12}");
    }
    if let Some(f1) = receipt.scorecard.f1 {
        println!("recovery_f1={f1:.12}");
    }
    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &str) -> Result<T, Box<dyn std::error::Error>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn write_json<T: serde::Serialize>(
    path: &str,
    value: &T,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    fs::write(path, bytes)?;
    Ok(())
}

fn usage() -> &'static str {
    "usage:\n  mag_qual_001 freeze <manifest.json> <submission.json> <isolation.json> <generation-receipt.json>\n  mag_qual_001 evaluate <manifest.json> <submission.json> <generation-receipt.json> <sealed-targets.json> <evaluation-environment.json> <evaluation-receipt.json>"
}
