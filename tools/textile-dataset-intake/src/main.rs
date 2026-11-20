// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use std::env;
use std::path::PathBuf;
use textile_dataset_intake::nir_sort::parse_nir_sort_filename;
use textile_dataset_intake::{build_receipt, verify_receipt, write_receipt};

fn usage() -> &'static str {
    "Usage:\n  textile-dataset-intake verify --manifest FILE --root DIR --operation USE --out RECEIPT.json\n  textile-dataset-intake verify-receipt --manifest FILE --root DIR --receipt RECEIPT.json\n  textile-dataset-intake inspect-nir-sort-name --file PATH\n\nNo command downloads data. The verifier only reads files already present under --root."
}

fn option(args: &[String], name: &str) -> Result<String, String> {
    let mut matches = args
        .windows(2)
        .filter(|pair| pair[0] == name)
        .map(|pair| pair[1].clone());
    let first = matches
        .next()
        .ok_or_else(|| format!("missing required option {name}"))?;
    if matches.next().is_some() {
        return Err(format!("option {name} was supplied more than once"));
    }
    Ok(first)
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    let command = args.get(1).ok_or_else(|| usage().to_owned())?;
    match command.as_str() {
        "verify" => {
            let manifest = PathBuf::from(option(&args, "--manifest")?);
            let root = PathBuf::from(option(&args, "--root")?);
            let operation = option(&args, "--operation")?;
            let output = PathBuf::from(option(&args, "--out")?);
            let receipt = build_receipt(&manifest, &root, &operation)
                .map_err(|error| error.to_string())?;
            write_receipt(&output, &receipt).map_err(|error| error.to_string())?;
            println!(
                "PASS {}: {} files, manifest sha256 {}",
                receipt.dataset_id,
                receipt.verified_files.len(),
                receipt.manifest_sha256
            );
            Ok(())
        }
        "verify-receipt" => {
            let manifest = PathBuf::from(option(&args, "--manifest")?);
            let root = PathBuf::from(option(&args, "--root")?);
            let receipt = PathBuf::from(option(&args, "--receipt")?);
            let verified = verify_receipt(&manifest, &root, &receipt)
                .map_err(|error| error.to_string())?;
            println!(
                "PASS {}: receipt matches {} verified files",
                verified.dataset_id,
                verified.verified_files.len()
            );
            Ok(())
        }
        "inspect-nir-sort-name" => {
            let file = option(&args, "--file")?;
            let identity = parse_nir_sort_filename(&file)
                .map_err(|error| error.to_string())?;
            let json = serde_json::to_string_pretty(&identity)
                .map_err(|error| format!("cannot encode identity as JSON: {error}"))?;
            println!("{json}");
            Ok(())
        }
        _ => Err(usage().to_owned()),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("FAIL: {error}");
        eprintln!("{}", usage());
        std::process::exit(2);
    }
}
