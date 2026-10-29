// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Validate the structural contract of a counterfactual evidence JSONL fixture.
//! This tool does not evaluate semantic support and intentionally accepts only the
//! input fixture path supplied by the caller; it never opens evaluator-only labels.

use std::env;
use std::fs;
use std::process::ExitCode;

use symthaea_broca::counterfactual_evidence_audit::validate_counterfactual_fixture_jsonl;

fn main() -> ExitCode {
    let mut args = env::args_os();
    let program = args
        .next()
        .and_then(|arg| arg.into_string().ok())
        .unwrap_or_else(|| "broca-counterfactual-audit".to_string());
    let Some(path) = args.next() else {
        eprintln!("usage: {program} <counterfactual-input.jsonl>");
        return ExitCode::from(2);
    };
    if args.next().is_some() {
        eprintln!("usage: {program} <counterfactual-input.jsonl>");
        return ExitCode::from(2);
    }

    let path = std::path::PathBuf::from(path);
    let input = match fs::read_to_string(&path) {
        Ok(input) => input,
        Err(error) => {
            eprintln!("cannot read fixture {}: {error}", path.display());
            return ExitCode::FAILURE;
        }
    };
    let pairs = match validate_counterfactual_fixture_jsonl(&input) {
        Ok(pairs) => pairs,
        Err(error) => {
            eprintln!("counterfactual fixture validation failed: {error}");
            return ExitCode::FAILURE;
        }
    };

    let families: std::collections::BTreeSet<_> =
        pairs.iter().map(|pair| pair.family.as_str()).collect();
    println!(
        "STRUCTURAL_FIXTURE_VALIDATION=PASS pairs={} families={} cases={} semantic_support=NOT_EVALUATED",
        pairs.len(),
        families.len(),
        pairs.len() * 2
    );
    println!("fixture_sha256={}", sha256_hex(input.as_bytes()));
    ExitCode::SUCCESS
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}
