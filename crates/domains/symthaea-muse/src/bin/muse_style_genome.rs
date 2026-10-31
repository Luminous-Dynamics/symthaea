// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evaluate a layered StyleGenome across fixed seeds and verify its receipt.

use std::env;
use std::fs;
use std::io::{self, Write};
use std::process::ExitCode;

use serde::Deserialize;
use symthaea_muse::{MusicalIntent, style_genome::{
    GenomeEvaluation, GenomeEvaluationConfig, StyleGenome, evaluate_genome,
    reference_population_digest,
}};
use symthaea_muse_protocol::catalog::HybridStyleSpec;
use symthaea_music_theory::fingerprint::STRUCT_DIMS;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvaluationRequest {
    recipe: HybridStyleSpec,
    intent: MusicalIntent,
    #[serde(default)]
    config: GenomeEvaluationConfig,
    /// Rows are exact structural fingerprints with STRUCT_DIMS values each.
    #[serde(default)]
    reference_population: Vec<Vec<f64>>,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("muse_style_genome: {error}");
            ExitCode::FAILURE
        }
    }
}

fn read_json<T: for<'de> serde::Deserialize<'de>>(path: &str) -> Result<T, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {path}: {error}"))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("parse {path}: {error}"))
}

fn write_json<T: serde::Serialize>(value: &T, path: Option<&str>) -> Result<(), String> {
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("serialize output: {error}"))?;
    bytes.push(b'\n');
    match path {
        Some(path) => {
            fs::write(path, bytes).map_err(|error| format!("write {path}: {error}"))?;
            eprintln!("wrote {path}");
        }
        None => {
            let stdout = io::stdout();
            let mut lock = stdout.lock();
            lock.write_all(&bytes)
                .map_err(|error| format!("write stdout: {error}"))?;
            lock.flush().map_err(|error| format!("flush stdout: {error}"))?;
        }
    }
    Ok(())
}

fn parse_reference_population(rows: Vec<Vec<f64>>) -> Result<Vec<[f64; STRUCT_DIMS]>, String> {
    rows.into_iter()
        .enumerate()
        .map(|(index, row)| {
            if row.len() != STRUCT_DIMS {
                return Err(format!(
                    "reference fingerprint {index} has {} dimensions; expected {STRUCT_DIMS}",
                    row.len()
                ));
            }
            let array: [f64; STRUCT_DIMS] = row.try_into()
                .map_err(|_| format!("reference fingerprint {index} dimension conversion failed"))?;
            if array.iter().any(|value| !value.is_finite()) {
                return Err(format!("reference fingerprint {index} contains a non-finite value"));
            }
            Ok(array)
        })
        .collect()
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let command = args.next().ok_or_else(usage)?;
    if command == "--help" || command == "-h" {
        println!("{}", usage());
        return Ok(());
    }
    let mut input = None::<String>;
    let mut output = None::<String>;
    let mut references = None::<String>;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" | "-i" => {
                let value = args.next().ok_or_else(|| format!("{arg} requires a path"))?;
                if input.replace(value).is_some() {
                    return Err("--input was supplied more than once".into());
                }
            }
            "--output" | "-o" => {
                let value = args.next().ok_or_else(|| format!("{arg} requires a path"))?;
                if output.replace(value).is_some() {
                    return Err("--output was supplied more than once".into());
                }
            }
            "--references" | "-r" => {
                let value = args.next().ok_or_else(|| format!("{arg} requires a path"))?;
                if references.replace(value).is_some() {
                    return Err("--references was supplied more than once".into());
                }
            }
            "--help" | "-h" => {
                println!("{}", usage());
                return Ok(());
            }
            other => return Err(format!("unknown argument {other:?}\n{}", usage())),
        }
    }
    let input = input.ok_or_else(|| "--input is required".to_string())?;

    match command.as_str() {
        "evaluate" => {
            if references.is_some() {
                return Err(
                    "reference fingerprints belong in the EvaluationRequest JSON; --references is for verify".into()
                );
            }
            let request: EvaluationRequest = read_json(&input)?;
            let reference_population = parse_reference_population(request.reference_population)?;
            let genome = StyleGenome::new(request.recipe)
                .map_err(|issue| format!("genome creation failed: {}: {}", issue.code, issue.message))?;
            let evaluation = evaluate_genome(
                &genome,
                &request.intent,
                &request.config,
                &reference_population,
            ).map_err(|issues| {
                issues.iter()
                    .map(|issue| format!("{}: {}", issue.code, issue.message))
                    .collect::<Vec<_>>()
                    .join("\n")
            })?;
            if !evaluation.verify_digest() {
                return Err("newly generated evaluation receipt failed digest verification".into());
            }
            eprintln!(
                "genome={} eligible={} seeds={} objective_axes=6 evaluation_sha256={}",
                evaluation.genome.genome_id,
                evaluation.eligible_for_archive,
                evaluation.seed_count,
                evaluation.evaluation_sha256,
            );
            if !evaluation.eligible_for_archive {
                eprintln!(
                    "abstained: {}",
                    evaluation.abstention_reasons.iter()
                        .map(|issue| format!("{}: {}", issue.code, issue.message))
                        .collect::<Vec<_>>()
                        .join("; ")
                );
            }
            write_json(&evaluation, output.as_deref())?;
        }
        "verify" => {
            let evaluation: GenomeEvaluation = read_json(&input)?;
            if !evaluation.verify_digest() {
                return Err(
                    "evaluation schema, genome identity, config commitment, score trials or report digest failed".into()
                );
            }
            match (&evaluation.reference_population_sha256, references.as_deref()) {
                (Some(expected), Some(path)) => {
                    let rows: Vec<Vec<f64>> = read_json(path)?;
                    let population = parse_reference_population(rows)?;
                    let actual = reference_population_digest(&population)
                        .map_err(|issues| issues.into_iter()
                            .map(|issue| format!("{}: {}", issue.code, issue.message))
                            .collect::<Vec<_>>()
                            .join("\n"))?
                        .ok_or_else(|| "reference population file was empty".to_string())?;
                    if &actual != expected
                        || population.len() != evaluation.reference_population_member_count
                    {
                        return Err("reference population digest/count does not match the evaluation receipt".into());
                    }
                }
                (Some(_), None) => {
                    return Err(
                        "this evaluation used a reference population; pass --references with the exact fingerprint JSON".into()
                    );
                }
                (None, Some(_)) => {
                    return Err(
                        "evaluation has no reference population commitment, but --references was supplied".into()
                    );
                }
                (None, None) => {}
            }
            println!(
                "valid StyleGenome evaluation: genome={} seeds={} eligible={} sha256={}",
                evaluation.genome.genome_id,
                evaluation.seed_count,
                evaluation.eligible_for_archive,
                evaluation.evaluation_sha256,
            );
        }
        other => return Err(format!("unknown command {other:?}\n{}", usage())),
    }
    Ok(())
}

fn usage() -> String {
    format!(
        "Usage:\n  muse_style_genome evaluate --input REQUEST.json [--output EVALUATION.json]\n  muse_style_genome verify --input EVALUATION.json [--references FINGERPRINTS.json]\n\nREQUEST.json fields:\n  recipe: HybridStyleSpec\n  intent: MusicalIntent\n  config: GenomeEvaluationConfig (optional)\n  reference_population: array of {}-value fingerprint arrays (optional)\n\nEvaluation uses deterministic seeds, hard score invariants, native trace reference checks, Muse Analyst's independent trace verifier, and a six-axis structural and grammar-native Pareto receipt. It does not use human preference scores.",
        STRUCT_DIMS,
    )
}
