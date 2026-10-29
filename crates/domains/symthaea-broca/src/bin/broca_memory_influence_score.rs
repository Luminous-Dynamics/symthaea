// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Offline scorer for paired memory-atom ablation observations.
//!
//! This evaluator intentionally requires both actual run observations and the separate
//! evaluator-only labels file. Never include evaluator labels in the production reader
//! or memory-query path.

use std::env;
use std::fs;
use std::process::ExitCode;

use symthaea_broca::counterfactual_evidence_audit::score_atom_ablation_jsonl;

fn main() -> ExitCode {
    let mut args = env::args_os();
    let program = args
        .next()
        .and_then(|arg| arg.into_string().ok())
        .unwrap_or_else(|| "broca-memory-influence-score".to_string());
    let Some(observations_path) = args.next() else {
        eprintln!("usage: {program} <observations.jsonl> <evaluator-only-labels.jsonl>");
        return ExitCode::from(2);
    };
    let Some(labels_path) = args.next() else {
        eprintln!("usage: {program} <observations.jsonl> <evaluator-only-labels.jsonl>");
        return ExitCode::from(2);
    };
    if args.next().is_some() {
        eprintln!("usage: {program} <observations.jsonl> <evaluator-only-labels.jsonl>");
        return ExitCode::from(2);
    }

    let observations_path = std::path::PathBuf::from(observations_path);
    let labels_path = std::path::PathBuf::from(labels_path);
    let observations = match fs::read_to_string(&observations_path) {
        Ok(contents) => contents,
        Err(error) => {
            eprintln!("cannot read observations {}: {error}", observations_path.display());
            return ExitCode::FAILURE;
        }
    };
    let labels = match fs::read_to_string(&labels_path) {
        Ok(contents) => contents,
        Err(error) => {
            eprintln!("cannot read evaluator labels {}: {error}", labels_path.display());
            return ExitCode::FAILURE;
        }
    };
    let score = match score_atom_ablation_jsonl(&observations, &labels) {
        Ok(score) => score,
        Err(error) => {
            eprintln!("atom-ablation evaluation rejected: {error}");
            return ExitCode::FAILURE;
        }
    };

    match serde_json::to_string_pretty(&score) {
        Ok(encoded) => {
            println!("{encoded}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("cannot serialize atom-ablation score: {error}");
            ExitCode::FAILURE
        }
    }
}
