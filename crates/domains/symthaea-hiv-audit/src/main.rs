// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use std::env;
use std::path::PathBuf;
use symthaea_hiv_audit::{
    capture_receipt, print_receipt_summary, verify_receipt, write_receipt_atomic,
};

fn usage() -> &'static str {
    "Symthaea HIV reservoir dataset audit

Commands:
  capture --manifest PATH --dataset ID --data-dir DIR --source-archive ZIP --output RECEIPT.json --source-commit HEX
  verify  --manifest PATH --dataset ID --data-dir DIR --source-archive ZIP --receipt RECEIPT.json

capture hashes each registered source file and validates CSV shape, unique participant IDs,
and pairwise ID overlap. It does not download source data and does not store participant IDs.
verify fails if files, metadata, CSV headers/rows, participant-ID sets, or joins have changed."
}

fn ensure_outside_data_dir(
    data_dir: &std::path::Path,
    path: &std::path::Path,
    label: &str,
) -> Result<(), String> {
    let canonical_data_dir = data_dir.canonicalize()
        .map_err(|e| format!("canonicalize data directory {}: {e}", data_dir.display()))?;
    let canonical_path = if path.exists() {
        path.canonicalize().map_err(|e| format!("canonicalize {} {}: {e}", label, path.display()))?
    } else {
        let parent = path.parent().filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(std::path::Path::new("."));
        parent.canonicalize()
            .map_err(|e| format!("canonicalize {} parent: {e}", label))?
            .join(path.file_name().ok_or_else(|| format!("{} has no filename", label))?)
    };
    if canonical_path.starts_with(&canonical_data_dir) {
        return Err(format!("{label} must be stored outside the source data directory"));
    }
    Ok(())
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

    let allowed: &[&str] = match command.as_str() {
        "capture" => &[
            "--manifest",
            "--dataset",
            "--data-dir",
            "--source-archive",
            "--output",
            "--source-commit",
        ],
        "verify" => &[
            "--manifest",
            "--dataset",
            "--data-dir",
            "--source-archive",
            "--receipt",
        ],
        _ => return Err(format!("unknown command {command:?}\n{}", usage())),
    };
    if let Some(unexpected) = values.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(format!("unexpected argument {unexpected:?} for command {command:?}"));
    }

    let required = |key: &str| -> Result<String, String> {
        values.get(key).cloned().ok_or_else(|| format!("required argument {key} is missing"))
    };

    let manifest = PathBuf::from(required("--manifest")?);
    let dataset_id = required("--dataset")?;
    let data_dir = PathBuf::from(required("--data-dir")?);
    let source_archive = PathBuf::from(required("--source-archive")?);

    match command.as_str() {
        "capture" => {
            let output = PathBuf::from(required("--output")?);
            let source_commit = required("--source-commit")?;
            std::fs::create_dir_all(output.parent().filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(std::path::Path::new(".")))
                .map_err(|e| format!("create output directory: {e}"))?;
            ensure_outside_data_dir(&data_dir, &output, "receipt")?;
            ensure_outside_data_dir(&data_dir, &source_archive, "source archive")?;

            let receipt = capture_receipt(
                &manifest,
                &dataset_id,
                &data_dir,
                &source_commit,
                &source_archive,
            )?;
            write_receipt_atomic(&output, &receipt)?;
            print_receipt_summary(&receipt);
            println!("receipt_written: {}", output.display());
            println!("claim_boundary: local integrity snapshot only; no biological result is inferred");
            Ok(())
        }
        "verify" => {
            if values.contains_key("--output") || values.contains_key("--source-commit") {
                return Err(
                    "verify accepts --receipt and --source-archive, not --output or --source-commit"
                        .to_owned(),
                );
            }
            let receipt_path = PathBuf::from(required("--receipt")?);
            ensure_outside_data_dir(&data_dir, &receipt_path, "receipt")?;
            ensure_outside_data_dir(&data_dir, &source_archive, "source archive")?;
            let receipt = verify_receipt(
                &manifest,
                &dataset_id,
                &data_dir,
                &receipt_path,
                &source_archive,
            )?;
            print_receipt_summary(&receipt);
            println!("verification: PASS (files match captured local receipt)");
            println!("claim_boundary: file integrity and structural consistency only");
            Ok(())
        }
        _ => Err(format!("unknown command {command:?}\n{}", usage())),
    }
}
