// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use std::env;
use std::fs::File;
use std::io::BufReader;
use std::process::ExitCode;

use symthaea_cosmology_bench::{
    sha256_bytes, sha256_file, verify_sha256_digest, verify_sha256_file_against_manifest, ChainTable,
    CHAIN_AUDIT_PROTOCOL, CHAIN_AUDIT_VERSION,
};

fn json_string(value: &str) -> String {
    let mut output = String::with_capacity(value.len() + 2);
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character.is_control() => {
                output.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => output.push(character),
        }
    }
    output.push('"');
    output
}

fn json_string_array(values: &[String]) -> String {
    let items = values.iter().map(|value| json_string(value)).collect::<Vec<_>>();
    format!("[{}]", items.join(", "))
}

fn json_input_artifacts(values: &[(String, String, u64)]) -> String {
    let items = values
        .iter()
        .map(|(path, digest, size)| {
            format!(
                "{{\"path\": {}, \"sha256\": {}, \"size_bytes\": {}}}",
                json_string(path),
                json_string(digest),
                size
            )
        })
        .collect::<Vec<_>>();
    format!("[{}]", items.join(", "))
}

fn run() -> Result<bool, String> {
    let args = env::args().collect::<Vec<_>>();
    if args.get(1).map(String::as_str) == Some("--verify-manifest") {
        if args.len() != 5 {
            return Err(format!(
                "usage: {} --verify-manifest SHA256SUMS RELATIVE_PATH FILE",
                args.first().map(String::as_str).unwrap_or("symthaea-cosmology-chain-audit")
            ));
        }
        let manifest_path = &args[2];
        let relative_path = &args[3];
        let file_path = &args[4];
        let manifest_text = std::fs::read_to_string(manifest_path)
            .map_err(|error| format!("cannot read manifest {manifest_path}: {error}"))?;
        let (expected, actual, matches) =
            verify_sha256_file_against_manifest(file_path, &manifest_text, relative_path)
                .map_err(|error| error.to_string())?;
        println!("{{");
        println!("  \"protocol\": {},", json_string(CHAIN_AUDIT_PROTOCOL));
        println!("  \"version\": {},", CHAIN_AUDIT_VERSION);
        println!("  \"mode\": \"sha256_manifest_verification\",");
        println!("  \"manifest_path\": {},", json_string(manifest_path));
        println!("  \"manifest_relative_path\": {},", json_string(relative_path));
        println!("  \"input_path\": {},", json_string(file_path));
        println!("  \"expected_sha256\": {},", json_string(&expected));
        println!("  \"actual_sha256\": {},", json_string(&actual));
        println!("  \"matches\": {}", matches);
        println!("}}");
        return Ok(matches);
    }
    if args.get(1).map(String::as_str) == Some("--verify-sha256") {
        if args.len() != 4 {
            return Err(format!(
                "usage: {} --verify-sha256 FILE EXPECTED_SHA256",
                args.first().map(String::as_str).unwrap_or("symthaea-cosmology-chain-audit")
            ));
        }
        let path = &args[2];
        let expected = &args[3];
        let actual = sha256_file(path).map_err(|error| error.to_string())?;
        let matches = verify_sha256_digest(&actual, expected).map_err(|error| error.to_string())?;
        println!("{{");
        println!("  \"protocol\": {},", json_string(CHAIN_AUDIT_PROTOCOL));
        println!("  \"version\": {},", CHAIN_AUDIT_VERSION);
        println!("  \"mode\": \"sha256_verification\",");
        println!("  \"input_path\": {},", json_string(path));
        println!("  \"expected_sha256\": {},", json_string(expected));
        println!("  \"actual_sha256\": {},", json_string(&actual));
        println!("  \"matches\": {}", matches);
        println!("}}");
        return Ok(matches);
    }

    if args.len() < 3 {
        return Err(format!(
            "usage: {} COLUMN_NAME CHAIN_TXT [CHAIN_TXT ...] OR --verify-sha256 FILE EXPECTED_SHA256 OR --verify-manifest SHA256SUMS RELATIVE_PATH FILE",
            args.first().map(String::as_str).unwrap_or("symthaea-cosmology-chain-audit")
        ));
    }

    let requested_column = &args[1];
    let paths = &args[2..];
    let mut tables = Vec::with_capacity(paths.len());
    let mut input_artifacts = Vec::with_capacity(paths.len());
    for path in paths {
        // Hash the exact byte buffer that is parsed, preventing a race where
        // the file could change between parsing and a later hash operation.
        let bytes = std::fs::read(path).map_err(|error| format!("cannot read {path}: {error}"))?;
        let digest = sha256_bytes(&bytes);
        let size_bytes = u64::try_from(bytes.len())
            .map_err(|_| format!("input {path} is too large to describe in the audit manifest"))?;
        let table = ChainTable::parse(std::io::Cursor::new(bytes.as_slice()))
            .map_err(|error| format!("{path}: {error}"))?;
        tables.push(table);
        input_artifacts.push((path.to_string(), digest, size_bytes));
    }
    let chain = ChainTable::combine(tables).map_err(|error| error.to_string())?;
    let summary = chain.summarize(requested_column).map_err(|error| error.to_string())?;

    println!("{{");
    println!("  \"protocol\": {},", json_string(CHAIN_AUDIT_PROTOCOL));
    println!("  \"version\": {},", CHAIN_AUDIT_VERSION);
    println!("  \"mode\": \"weighted_chain_summary\",");
    println!("  \"input_paths\": {},", json_string_array(paths));
    println!("  \"input_chain_count\": {},", paths.len());
    println!("  \"input_artifacts\": {},", json_input_artifacts(&input_artifacts));
    println!("  \"selected_column\": {},", json_string(&summary.name));
    println!("  \"sample_rows\": {},", summary.sample_rows);
    println!("  \"positive_weight_rows\": {},", summary.positive_weight_rows);
    println!("  \"total_weight\": {:.17},", summary.total_weight);
    println!("  \"weight_effective_sample_size\": {:.17},", summary.weight_effective_sample_size);
    println!("  \"weighted_mean\": {:.17},", summary.weighted_mean);
    println!("  \"weighted_variance\": {:.17},", summary.weighted_variance);
    println!("  \"weighted_empirical_q16\": {:.17},", summary.q16);
    println!("  \"weighted_empirical_q50\": {:.17},", summary.q50);
    println!("  \"weighted_empirical_q84\": {:.17}", summary.q84);
    println!("}}");
    Ok(true)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}
