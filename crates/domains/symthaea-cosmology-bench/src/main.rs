// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use std::env;
use std::fs;
use std::process::ExitCode;

use symthaea_cosmology_bench::{
    expected_sha256_from_manifest, sha256_bytes, sha256_file, verify_sha256_digest,
    verify_sha256_file_against_manifest, ChainTable, CHAIN_AUDIT_PROTOCOL,
    CHAIN_AUDIT_VERSION,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct InputArtifactAudit {
    path: String,
    manifest_relative_path: Option<String>,
    expected_sha256: Option<String>,
    sha256: String,
    size_bytes: u64,
    checksum_verified: bool,
}

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

fn json_optional_string(value: Option<&str>) -> String {
    value.map(json_string).unwrap_or_else(|| "null".to_string())
}

fn json_string_array(values: &[String]) -> String {
    let items = values.iter().map(|value| json_string(value)).collect::<Vec<_>>();
    format!("[{}]", items.join(", "))
}

fn json_input_artifacts(values: &[InputArtifactAudit]) -> String {
    let items = values
        .iter()
        .map(|artifact| {
            format!(
                "{{\"path\": {}, \"manifest_relative_path\": {}, \"expected_sha256\": {}, \"sha256\": {}, \"size_bytes\": {}, \"checksum_verified\": {}}}",
                json_string(&artifact.path),
                json_optional_string(artifact.manifest_relative_path.as_deref()),
                json_optional_string(artifact.expected_sha256.as_deref()),
                json_string(&artifact.sha256),
                artifact.size_bytes,
                artifact.checksum_verified
            )
        })
        .collect::<Vec<_>>();
    format!("[{}]", items.join(", "))
}

fn summarize_loaded_chains(
    requested_column: &str,
    tables: Vec<ChainTable>,
    input_artifacts: &[InputArtifactAudit],
    mode: &str,
    checksum_manifest: Option<(&str, &str)>,
) -> Result<bool, String> {
    let chain = ChainTable::combine(tables).map_err(|error| error.to_string())?;
    let summary = chain.summarize(requested_column).map_err(|error| error.to_string())?;
    let paths = input_artifacts
        .iter()
        .map(|artifact| artifact.path.clone())
        .collect::<Vec<_>>();
    let manifest_json = checksum_manifest
        .map(|(path, digest)| {
            format!(
                "{{\"path\": {}, \"sha256\": {}}}",
                json_string(path),
                json_string(digest)
            )
        })
        .unwrap_or_else(|| "null".to_string());

    println!("{{");
    println!("  \"protocol\": {},", json_string(CHAIN_AUDIT_PROTOCOL));
    println!("  \"version\": {},", CHAIN_AUDIT_VERSION);
    println!("  \"mode\": {},", json_string(mode));
    println!("  \"input_paths\": {},", json_string_array(&paths));
    println!("  \"input_chain_count\": {},", paths.len());
    println!("  \"checksum_manifest\": {},", manifest_json);
    println!("  \"input_artifacts\": {},", json_input_artifacts(input_artifacts));
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

fn load_unverified_chains(paths: &[String]) -> Result<(Vec<ChainTable>, Vec<InputArtifactAudit>), String> {
    let mut tables = Vec::with_capacity(paths.len());
    let mut artifacts = Vec::with_capacity(paths.len());
    for path in paths {
        let bytes = fs::read(path).map_err(|error| format!("cannot read {path}: {error}"))?;
        let digest = sha256_bytes(&bytes);
        let size_bytes = u64::try_from(bytes.len())
            .map_err(|_| format!("input {path} is too large to describe in the audit manifest"))?;
        let table = ChainTable::parse(std::io::Cursor::new(bytes.as_slice()))
            .map_err(|error| format!("{path}: {error}"))?;
        tables.push(table);
        artifacts.push(InputArtifactAudit {
            path: path.clone(),
            manifest_relative_path: None,
            expected_sha256: None,
            sha256: digest,
            size_bytes,
            checksum_verified: false,
        });
    }
    Ok((tables, artifacts))
}

fn load_manifest_verified_chains(
    manifest_text: &str,
    manifest_sha256: &str,
    mappings: &[String],
) -> Result<(Vec<ChainTable>, Vec<InputArtifactAudit>), String> {
    let mut tables = Vec::with_capacity(mappings.len());
    let mut artifacts = Vec::with_capacity(mappings.len());

    for mapping in mappings {
        let (relative_path, local_path) = mapping.split_once('=').ok_or_else(|| {
            format!(
                "invalid input mapping {mapping:?}; expected MANIFEST_RELATIVE_PATH=LOCAL_FILE_PATH"
            )
        })?;
        if relative_path.is_empty() || local_path.is_empty() {
            return Err(format!(
                "invalid input mapping {mapping:?}; both sides of '=' are required"
            ));
        }

        let expected = expected_sha256_from_manifest(manifest_text, relative_path)
            .map_err(|error| format!("manifest entry {relative_path:?}: {error}"))?;
        let bytes = fs::read(local_path)
            .map_err(|error| format!("cannot read local artifact {local_path}: {error}"))?;
        let actual = sha256_bytes(&bytes);
        if !verify_sha256_digest(&actual, &expected)
            .map_err(|error| error.to_string())?
        {
            return Err(format!(
                "checksum mismatch for {local_path:?} (manifest path {relative_path:?}): expected {expected}, actual {actual}"
            ));
        }

        // Hash and parse exactly the same bytes. The manifest check completes
        // before these bytes can be included in any reported scientific summary.
        let size_bytes = u64::try_from(bytes.len())
            .map_err(|_| format!("input {local_path} is too large to describe in the audit manifest"))?;
        let table = ChainTable::parse(std::io::Cursor::new(bytes.as_slice()))
            .map_err(|error| format!("{local_path}: {error}"))?;
        tables.push(table);
        artifacts.push(InputArtifactAudit {
            path: local_path.to_string(),
            manifest_relative_path: Some(relative_path.to_string()),
            expected_sha256: Some(expected),
            sha256: actual,
            size_bytes,
            checksum_verified: true,
        });
    }

    // Mention the manifest digest in the report so it can be retained with the
    // result. This does not independently authenticate the manifest itself; the
    // caller must acquire it from the official publisher over a trusted channel.
    let _ = manifest_sha256;
    Ok((tables, artifacts))
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
        let manifest_bytes = fs::read(manifest_path)
            .map_err(|error| format!("cannot read manifest {manifest_path}: {error}"))?;
        let manifest_text = std::str::from_utf8(&manifest_bytes)
            .map_err(|error| format!("checksum manifest is not UTF-8: {error}"))?;
        let (expected, actual, matches) =
            verify_sha256_file_against_manifest(file_path, manifest_text, relative_path)
                .map_err(|error| error.to_string())?;
        println!("{{");
        println!("  \"protocol\": {},", json_string(CHAIN_AUDIT_PROTOCOL));
        println!("  \"version\": {},", CHAIN_AUDIT_VERSION);
        println!("  \"mode\": \"sha256_manifest_verification\",");
        println!("  \"manifest_path\": {},", json_string(manifest_path));
        println!("  \"manifest_sha256\": {},", json_string(&sha256_bytes(&manifest_bytes)));
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

    if args.get(1).map(String::as_str) == Some("--audit-manifest") {
        if args.len() < 6 {
            return Err(format!(
                "usage: {} --audit-manifest SHA256SUMS COLUMN RELATIVE_PATH=LOCAL_FILE [RELATIVE_PATH=LOCAL_FILE ...]",
                args.first().map(String::as_str).unwrap_or("symthaea-cosmology-chain-audit")
            ));
        }
        let manifest_path = &args[2];
        let requested_column = &args[3];
        let manifest_bytes = fs::read(manifest_path)
            .map_err(|error| format!("cannot read manifest {manifest_path}: {error}"))?;
        let manifest_text = std::str::from_utf8(&manifest_bytes)
            .map_err(|error| format!("checksum manifest is not UTF-8: {error}"))?;
        let manifest_sha256 = sha256_bytes(&manifest_bytes);
        let mappings = &args[4..];
        let (tables, artifacts) = load_manifest_verified_chains(manifest_text, &manifest_sha256, mappings)?;
        summarize_loaded_chains(
            requested_column,
            tables,
            &artifacts,
            "manifest_verified_weighted_chain_summary",
            Some((manifest_path, &manifest_sha256)),
        )
    } else if args.get(1).map(String::as_str) == Some("--summary-unverified") {
        if args.len() < 4 {
            return Err(format!(
                "usage: {} --summary-unverified COLUMN_NAME CHAIN_TXT [CHAIN_TXT ...]",
                args.first().map(String::as_str).unwrap_or("symthaea-cosmology-chain-audit")
            ));
        }
        let requested_column = &args[2];
        let paths = args[3..].to_vec();
        let (tables, artifacts) = load_unverified_chains(&paths)?;
        summarize_loaded_chains(
            requested_column,
            tables,
            &artifacts,
            "unverified_weighted_chain_summary",
            None,
        )
    } else {
        Err(format!(
            "usage: {} --audit-manifest SHA256SUMS COLUMN RELATIVE_PATH=LOCAL_FILE [...] OR --summary-unverified COLUMN_NAME CHAIN_TXT [...] OR --verify-manifest SHA256SUMS RELATIVE_PATH FILE OR --verify-sha256 FILE EXPECTED_SHA256",
            args.first().map(String::as_str).unwrap_or("symthaea-cosmology-chain-audit")
        ))
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_artifact_json_binds_path_digest_and_byte_size() {
        let artifacts = vec![InputArtifactAudit {
            path: "chain.1.txt".to_string(),
            manifest_relative_path: Some("cobaya/base/desi-bao-all/chain.1.txt".to_string()),
            expected_sha256: Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".to_string()),
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".to_string(),
            size_bytes: 3,
            checksum_verified: true,
        }];
        assert_eq!(
            json_input_artifacts(&artifacts),
            "[{\"path\": \"chain.1.txt\", \"manifest_relative_path\": \"cobaya/base/desi-bao-all/chain.1.txt\", \"expected_sha256\": \"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\", \"sha256\": \"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\", \"size_bytes\": 3, \"checksum_verified\": true}]"
        );
    }

    #[test]
    fn input_artifact_json_escapes_untrusted_paths() {
        let artifacts = vec![InputArtifactAudit {
            path: "chain\n1.txt".to_string(),
            manifest_relative_path: None,
            expected_sha256: None,
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".to_string(),
            size_bytes: 3,
            checksum_verified: false,
        }];
        assert!(json_input_artifacts(&artifacts).contains("chain\\n1.txt"));
    }
}
