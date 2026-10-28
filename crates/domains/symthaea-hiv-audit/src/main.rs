// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use std::env;
use std::path::PathBuf;
use symthaea_hiv_audit::{capture_receipt, print_receipt_summary, verify_receipt, write_receipt_atomic};

fn usage() -> &'static str {
    "Symthaea HIV reservoir dataset audit

Commands:
  capture --manifest PATH --dataset ID --data-dir DIR --output RECEIPT.json --source-commit HEX
  verify  --manifest PATH --dataset ID --data-dir DIR --receipt RECEIPT.json

capture hashes each registered source file and validates CSV shape, unique participant IDs,
and pairwise ID overlap. It does not download source data and does not store participant IDs.
verify fails if files, metadata, CSV headers/rows, participant-ID sets, or joins have changed."
}

fn main() {
    if let Err(error) = run() {
        eprintln!("ERROR: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let command = args.next().ok_or_else(|| usage().to_owned())?;
    if command == "--help" || command == "-h" || command == "help" {
        println!("{}", usage());
        return Ok(());
    }

    let mut values = std::collections::HashMap::<String, String>::new();
    while let Some(key) = args.next() {
        if !key.starts_with("--") {
            return Err(format!("expected a --flag, received {key:?}\n{}", usage()));
        }
        let value = args.next().ok_or_else(|| format!("missing value for {key}"))?;
        if values.insert(key.clone(), value).is_some() {
            return Err(format!("duplicate argument {key}"));
        }
    }

    let required = |key: &str| -> Result<String, String> {
        values.get(key).cloned().ok_or_else(|| format!("required argument {key} is missing"))
    };

    let manifest = PathBuf::from(required("--manifest")?);
    let dataset_id = required("--dataset")?;
    let data_dir = PathBuf::from(required("--data-dir")?);

    match command.as_str() {
        "capture" => {
            let output = PathBuf::from(required("--output")?);
            let source_commit = required("--source-commit")?;
            let canonical_data_dir = data_dir.canonicalize()
                .map_err(|e| format!("canonicalize data directory {}: {e}", data_dir.display()))?;
            let output_parent = output.parent().filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(std::path::Path::new("."));
            std::fs::create_dir_all(output_parent)
                .map_err(|e| format!("create output directory: {e}"))?;
            let canonical_parent = output_parent.canonicalize()
                .map_err(|e| format!("canonicalize receipt directory: {e}"))?;
            let canonical_output = canonical_parent.join(output.file_name().unwrap_or_default());
            if canonical_output.starts_with(&canonical_data_dir) {
                return Err("receipt must be stored outside the source data directory".to_owned());
            }

            let receipt = capture_receipt(&manifest, &dataset_id, &data_dir, &source_commit)?;
            write_receipt_atomic(&output, &receipt)?;
            print_receipt_summary(&receipt);
            println!("receipt_written: {}", output.display());
            println!("claim_boundary: local integrity snapshot only; no biological result is inferred");
            Ok(())
        }
        "verify" => {
            if values.contains_key("--output") || values.contains_key("--source-commit") {
                return Err("verify accepts --receipt, not --output or --source-commit".to_owned());
            }
            let receipt_path = PathBuf::from(required("--receipt")?);
            let receipt = verify_receipt(&manifest, &dataset_id, &data_dir, &receipt_path)?;
            print_receipt_summary(&receipt);
            println!("verification: PASS (files match captured local receipt)");
            println!("claim_boundary: file integrity and structural consistency only");
            Ok(())
        }
        _ => Err(format!("unknown command {command:?}\n{}", usage())),
    }
}
