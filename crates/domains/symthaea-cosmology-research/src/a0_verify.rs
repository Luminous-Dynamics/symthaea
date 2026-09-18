// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! DE-001A0 byte-integrity verifier.
//!
//! This program is deliberately local-only. It does not download data and it
//! does not evaluate a cosmological likelihood. It verifies that externally
//! retrieved files are byte-identical to preregistered authoritative inputs.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

const MAX_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    protocol: String,
    scientific_claim: String,
    mirror_policy: String,
    artifacts: Vec<ArtifactSpec>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactSpec {
    role: String,
    authority: String,
    locator: String,
    expected_size: u64,
    sha256: String,
}

#[derive(Debug, Serialize)]
struct ArtifactReceipt {
    role: String,
    authority: String,
    locator: String,
    path: String,
    expected_size: u64,
    actual_size: u64,
    expected_sha256: String,
    actual_sha256: String,
    status: &'static str,
}

#[derive(Debug, Serialize)]
struct Receipt {
    protocol: String,
    manifest_sha256: String,
    verdict: &'static str,
    scientific_claim: &'static str,
    artifacts: Vec<ArtifactReceipt>,
    errors: Vec<String>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn read_manifest_bytes(path: &Path) -> Result<Vec<u8>, String> {
    let meta = fs::symlink_metadata(path)
        .map_err(|error| format!("{}: manifest metadata failed: {error}", path.display()))?;
    if meta.file_type().is_symlink() {
        return Err(format!("{}: manifest symlinks are forbidden", path.display()));
    }
    if !meta.file_type().is_file() {
        return Err(format!("{}: manifest is not a regular file", path.display()));
    }
    if meta.len() > MAX_MANIFEST_BYTES {
        return Err(format!(
            "{}: manifest size {} exceeds limit {MAX_MANIFEST_BYTES}",
            path.display(),
            meta.len()
        ));
    }
    let bytes = fs::read(path)
        .map_err(|error| format!("{}: manifest read failed: {error}", path.display()))?;
    if bytes.len() as u64 != meta.len() {
        return Err(format!("{}: manifest size changed while reading", path.display()));
    }
    Ok(bytes)
}

fn load_regular_file(path: &Path, expected_size: u64) -> Result<Vec<u8>, String> {
    let link_meta = fs::symlink_metadata(path)
        .map_err(|error| format!("{}: metadata failed: {error}", path.display()))?;
    if link_meta.file_type().is_symlink() {
        return Err(format!("{}: symlinks are forbidden", path.display()));
    }
    if !link_meta.file_type().is_file() {
        return Err(format!("{}: not a regular file", path.display()));
    }
    if expected_size > MAX_ARTIFACT_BYTES {
        return Err(format!(
            "{}: expected size {expected_size} exceeds verifier limit {MAX_ARTIFACT_BYTES}",
            path.display()
        ));
    }
    if link_meta.len() != expected_size {
        return Err(format!(
            "{}: size mismatch before read: expected {expected_size}, got {}",
            path.display(),
            link_meta.len()
        ));
    }

    let mut file = File::open(path)
        .map_err(|error| format!("{}: open failed: {error}", path.display()))?;
    let opened_meta = file
        .metadata()
        .map_err(|error| format!("{}: opened-file metadata failed: {error}", path.display()))?;
    if opened_meta.len() != expected_size {
        return Err(format!(
            "{}: size changed between metadata and open: expected {expected_size}, got {}",
            path.display(),
            opened_meta.len()
        ));
    }

    let capacity = usize::try_from(expected_size)
        .map_err(|_| format!("{}: expected size does not fit usize", path.display()))?;
    let mut bytes = Vec::with_capacity(capacity);
    file.read_to_end(&mut bytes)
        .map_err(|error| format!("{}: read failed: {error}", path.display()))?;
    if bytes.len() as u64 != expected_size {
        return Err(format!(
            "{}: size changed while reading: expected {expected_size}, got {}",
            path.display(),
            bytes.len()
        ));
    }
    Ok(bytes)
}

fn parse_bindings(args: impl IntoIterator<Item = String>) -> Result<BTreeMap<String, PathBuf>, String> {
    let mut bindings = BTreeMap::new();
    for arg in args {
        let (role, path) = arg
            .split_once('=')
            .ok_or_else(|| format!("binding must be ROLE=PATH, got {arg:?}"))?;
        if role.trim().is_empty() || path.trim().is_empty() {
            return Err(format!("binding must contain non-empty ROLE and PATH: {arg:?}"));
        }
        if bindings.insert(role.to_owned(), PathBuf::from(path)).is_some() {
            return Err(format!("duplicate role binding: {role}"));
        }
    }
    Ok(bindings)
}

fn verify(manifest_path: &Path, bindings: BTreeMap<String, PathBuf>) -> Result<Receipt, String> {
    let manifest_bytes = read_manifest_bytes(manifest_path)?;
    let manifest_sha256 = sha256_hex(&manifest_bytes);
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("{}: invalid manifest JSON: {error}", manifest_path.display()))?;

    if manifest.schema_version != 1 {
        return Err(format!("unsupported manifest schema_version={}", manifest.schema_version));
    }
    if manifest.protocol.trim().is_empty() {
        return Err("manifest protocol must not be empty".into());
    }
    if manifest.scientific_claim != "NONE" {
        return Err(format!(
            "A0 manifest must declare scientific_claim=NONE, got {:?}",
            manifest.scientific_claim
        ));
    }
    if manifest.mirror_policy.trim().is_empty() {
        return Err("manifest mirror_policy must not be empty".into());
    }

    let mut roles = BTreeSet::new();
    for spec in &manifest.artifacts {
        if !roles.insert(spec.role.as_str()) {
            return Err(format!("duplicate manifest role: {}", spec.role));
        }
        if spec.role.trim().is_empty() || spec.authority.trim().is_empty() || spec.locator.trim().is_empty() {
            return Err(format!("manifest artifact {:?} has empty identity fields", spec.role));
        }
        if !is_lower_hex_sha256(&spec.sha256) {
            return Err(format!("manifest role {} has invalid SHA-256", spec.role));
        }
        if spec.expected_size > MAX_ARTIFACT_BYTES {
            return Err(format!("manifest role {} exceeds verifier size limit", spec.role));
        }
    }

    let expected_roles: BTreeSet<&str> = manifest.artifacts.iter().map(|spec| spec.role.as_str()).collect();
    let supplied_roles: BTreeSet<&str> = bindings.keys().map(String::as_str).collect();
    if expected_roles != supplied_roles {
        let missing: Vec<_> = expected_roles.difference(&supplied_roles).copied().collect();
        let unexpected: Vec<_> = supplied_roles.difference(&expected_roles).copied().collect();
        return Err(format!("role set mismatch: missing={missing:?}, unexpected={unexpected:?}"));
    }

    let mut receipts = Vec::with_capacity(manifest.artifacts.len());
    let mut errors = Vec::new();

    for spec in manifest.artifacts {
        let path = bindings
            .get(&spec.role)
            .expect("role-set equality guarantees a binding");
        match load_regular_file(path, spec.expected_size) {
            Ok(bytes) => {
                let actual_sha256 = sha256_hex(&bytes);
                let matches = actual_sha256 == spec.sha256;
                if !matches {
                    errors.push(format!("{}: SHA-256 mismatch", spec.role));
                }
                receipts.push(ArtifactReceipt {
                    role: spec.role,
                    authority: spec.authority,
                    locator: spec.locator,
                    path: path.display().to_string(),
                    expected_size: spec.expected_size,
                    actual_size: bytes.len() as u64,
                    expected_sha256: spec.sha256,
                    actual_sha256,
                    status: if matches { "PASS" } else { "INVALID" },
                });
            }
            Err(error) => {
                errors.push(format!("{}: {error}", spec.role));
                receipts.push(ArtifactReceipt {
                    role: spec.role,
                    authority: spec.authority,
                    locator: spec.locator,
                    path: path.display().to_string(),
                    expected_size: spec.expected_size,
                    actual_size: 0,
                    expected_sha256: spec.sha256,
                    actual_sha256: "UNAVAILABLE".into(),
                    status: "INVALID",
                });
            }
        }
    }

    Ok(Receipt {
        protocol: manifest.protocol,
        manifest_sha256,
        verdict: if errors.is_empty() { "PASS" } else { "INVALID" },
        scientific_claim: "NONE",
        artifacts: receipts,
        errors,
    })
}

fn write_receipt(receipt: &Receipt) -> io::Result<()> {
    serde_json::to_writer_pretty(io::stdout().lock(), receipt)?;
    println!();
    Ok(())
}

fn run() -> Result<i32, String> {
    let mut args = env::args().skip(1);
    let manifest = args.next().ok_or_else(|| {
        "usage: de001a-a0-verify MANIFEST.json ROLE=PATH [ROLE=PATH ...]".to_owned()
    })?;
    let bindings = parse_bindings(args)?;
    let receipt = verify(Path::new(&manifest), bindings)?;
    let exit = if receipt.verdict == "PASS" { 0 } else { 2 };
    write_receipt(&receipt).map_err(|error| format!("failed to write receipt: {error}"))?;
    Ok(exit)
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("DE-001A0 INVALID: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_sha256_vector_is_correct() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn digest_validator_is_strict_lower_hex() {
        assert!(is_lower_hex_sha256(&"a".repeat(64)));
        assert!(!is_lower_hex_sha256(&"A".repeat(64)));
        assert!(!is_lower_hex_sha256("abc"));
        assert!(!is_lower_hex_sha256(&"g".repeat(64)));
    }

    #[test]
    fn duplicate_cli_bindings_are_rejected() {
        let args = vec!["data=/tmp/a".to_owned(), "data=/tmp/b".to_owned()];
        assert!(parse_bindings(args).is_err());
    }
}
