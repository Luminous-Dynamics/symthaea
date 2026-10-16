// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Offline integrity verifier for a retained Living City run package.
//!
//! Usage: living_city_verify <run-directory>
//!
//! The directory contains bundle.json, a typed evidence manifest, a typed run
//! receipt, a receipt digest sidecar, and the source/output files referenced by
//! bundle.json. File paths must remain within the canonical run directory.
//!
//! A successful report means only that the package is internally consistent and
//! its retained bytes match the declared digests. It does not authenticate the
//! author, validate physics, certify code compliance, or approve construction.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io;
use std::path::{Component, Path, PathBuf};
use std::process;

use symthaea_scenarios::living_city_evidence::{
    EvidenceManifest, ScenarioRunReceipt, evidence_manifest_sha256,
    sha256_reader_hex, scenario_run_receipt_sha256, verify_manifest_source_digests,
    verify_receipt_manifest_link, verify_receipt_output_digests,
    verify_scenario_run_receipt_digest,
};

const BUNDLE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VerificationBundle {
    schema_version: u32,
    manifest_path: String,
    receipt_path: String,
    receipt_sha256_path: String,
    source_files: Vec<FileBinding>,
    output_files: Vec<FileBinding>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileBinding {
    id: String,
    path: String,
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn safe_file_path(root: &Path, relative_path: &str) -> io::Result<PathBuf> {
    let relative = Path::new(relative_path);
    if relative.as_os_str().is_empty()
        || relative.is_absolute()
        || relative.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(invalid_input(
            "package file path must be a non-empty relative path without parent traversal",
        ));
    }

    let candidate = root.join(relative);
    let resolved = fs::canonicalize(candidate)?;
    if !resolved.starts_with(root) {
        return Err(invalid_input(
            "package file resolves outside the run directory (path traversal or external symlink)",
        ));
    }
    if !resolved.is_file() {
        return Err(invalid_input("package path must resolve to a regular file"));
    }
    Ok(resolved)
}

fn read_package_file(root: &Path, relative_path: &str) -> io::Result<Vec<u8>> {
    fs::read(safe_file_path(root, relative_path)?)
}

fn read_json<T: for<'de> Deserialize<'de>>(
    root: &Path,
    relative_path: &str,
) -> Result<T, Box<dyn Error>> {
    let bytes = read_package_file(root, relative_path)?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn compute_file_digest_map(
    root: &Path,
    bindings: &[FileBinding],
    collection_name: &'static str,
) -> Result<BTreeMap<String, String>, Box<dyn Error>> {
    let mut digests = BTreeMap::new();
    for binding in bindings {
        if binding.id.trim().is_empty() {
            return Err(invalid_input(format!(
                "{collection_name} contains an empty artifact ID"
            ))
            .into());
        }
        if digests.contains_key(&binding.id) {
            return Err(invalid_input(format!(
                "{collection_name} contains duplicate artifact ID {:?}",
                binding.id
            ))
            .into());
        }

        let path = safe_file_path(root, &binding.path)?;
        let file = File::open(path)?;
        let actual_sha256 = sha256_reader_hex(file)?;
        digests.insert(binding.id.clone(), actual_sha256);
    }
    Ok(digests)
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os();
    let _program = args.next();
    let run_directory = match (args.next(), args.next()) {
        (Some(path), None) => PathBuf::from(path),
        _ => {
            return Err(invalid_input("usage: living_city_verify <run-directory>").into());
        }
    };

    let root = fs::canonicalize(run_directory)?;
    if !root.is_dir() {
        return Err(invalid_input("run-directory must be a directory").into());
    }

    let bundle: VerificationBundle = read_json(&root, "bundle.json")?;
    if bundle.schema_version != BUNDLE_SCHEMA_VERSION {
        return Err(invalid_input("unsupported verification bundle schema version").into());
    }

    let manifest: EvidenceManifest = read_json(&root, &bundle.manifest_path)?;
    let receipt: ScenarioRunReceipt = read_json(&root, &bundle.receipt_path)?;
    manifest.validate()?;
    receipt.validate()?;

    let receipt_digest_sidecar =
        read_package_file(&root, &bundle.receipt_sha256_path)?;
    let expected_receipt_sha256 = std::str::from_utf8(&receipt_digest_sidecar)?.trim();
    verify_scenario_run_receipt_digest(&receipt, expected_receipt_sha256)?;
    verify_receipt_manifest_link(&receipt, &manifest)?;

    let source_digests =
        compute_file_digest_map(&root, &bundle.source_files, "source_files")?;
    verify_manifest_source_digests(&manifest, &source_digests)?;

    let output_digests =
        compute_file_digest_map(&root, &bundle.output_files, "output_files")?;
    verify_receipt_output_digests(&receipt, &output_digests)?;

    let manifest_sha256 = evidence_manifest_sha256(&manifest)?;
    let receipt_sha256 = scenario_run_receipt_sha256(&receipt)?;
    let missing_sources: Vec<&str> = manifest
        .items
        .iter()
        .filter(|item| {
            item.kind
                == symthaea_scenarios::living_city_evidence::EvidenceKind::Missing
        })
        .map(|item| item.id.as_str())
        .collect();

    let report = serde_json::json!({
        "schema_version": 1,
        "package_integrity": "verified",
        "run_id": receipt.run_id,
        "scenario_id": receipt.scenario_id,
        "repository": receipt.repository,
        "commit_sha": receipt.commit_sha,
        "manifest_sha256": manifest_sha256,
        "receipt_sha256": receipt_sha256,
        "verified_source_count": source_digests.len(),
        "verified_output_count": output_digests.len(),
        "explicit_missing_source_ids": missing_sources,
        "scope": "Package integrity and digest linkage only. This report does not authenticate receipt authorship, validate the physical models or solver claims, establish jurisdictional code compliance, replace professional review, or authorize construction."
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("living_city_verify: verification failed: {error}");
        process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verifier_rejects_empty_absolute_and_parent_traversal_paths() {
        let root = Path::new("/does/not/need/to/exist/for-this-guard-test");
        assert!(safe_file_path(root, "").is_err());
        assert!(safe_file_path(root, "/etc/passwd").is_err());
        assert!(safe_file_path(root, "../outside.txt").is_err());
        assert!(safe_file_path(root, "subdir/../../outside.txt").is_err());
    }

    #[test]
    fn bundle_parser_rejects_unknown_fields() {
        let bundle = r#"{
            "schema_version": 1,
            "manifest_path": "manifest.json",
            "receipt_path": "receipt.json",
            "receipt_sha256_path": "receipt.sha256",
            "source_files": [],
            "output_files": [],
            "unexpected_authority": "trusted"
        }"#;
        assert!(serde_json::from_str::<VerificationBundle>(bundle).is_err());
    }

    #[test]
    fn bundle_parser_accepts_the_declared_schema() {
        let bundle = r#"{
            "schema_version": 1,
            "manifest_path": "manifest.json",
            "receipt_path": "receipt.json",
            "receipt_sha256_path": "receipt.sha256",
            "source_files": [{"id": "survey", "path": "sources/survey.geojson"}],
            "output_files": [{"id": "metrics", "path": "outputs/metrics.json"}]
        }"#;
        let parsed = serde_json::from_str::<VerificationBundle>(bundle).unwrap();
        assert_eq!(parsed.schema_version, BUNDLE_SCHEMA_VERSION);
        assert_eq!(parsed.source_files.len(), 1);
        assert_eq!(parsed.output_files.len(), 1);
    }
}
