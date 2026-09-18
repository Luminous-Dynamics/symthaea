// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! DE-001A1E reusable environment-evidence verifier.
//!
//! This program performs no cosmology. It verifies that a qualified environment
//! receipt remains reusable for the current checkout by recomputing the exact
//! environment definition and independently realizing the same Nix environment.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

const MAX_TEXT_BYTES: u64 = 4 * 1024 * 1024;
const PROTOCOL: &str = "DE-001A1E-ENVIRONMENT-EVIDENCE-v1";
const AUTHORITY: &str = "environment-evidence-reuse-only";
const NIX_VERSION: &str = "nix (Nix) 2.34.7";
const DEFINITION_PROTOCOL: &str = "DE-001A-ENVIRONMENT-DEFINITION-v1";
const RECEIPT_PROTOCOL: &str = "DE-001A-ENVIRONMENT-QUALIFICATION-v3";
const REUSE_POLICY: &str = "current-checkout-must-match-environment-definition-sha256";
const FLAKE_RELATIVE: &str = "crates/domains/symthaea-cosmology-research/nix/flake.nix";
const LOCK_RELATIVE: &str = "crates/domains/symthaea-cosmology-research/nix/flake.lock";
const CLOSURE_RELATIVE: &str =
    "crates/domains/symthaea-cosmology-research/references/de001a_execution_closure_v2.json";
const FLAKE_DIR_RELATIVE: &str = "crates/domains/symthaea-cosmology-research/nix";

const RECEIPT_KEYS: &[&str] = &[
    "protocol",
    "verdict",
    "authority",
    "scientific_claim",
    "reuse_policy",
    "source_head",
    "source_tree",
    "source_base",
    "environment_definition_sha256",
    "flake_nix_sha256",
    "flake_lock_sha256",
    "closure_manifest_sha256",
    "flake_metadata_sha256",
    "nix_version",
    "environment_store_path",
    "environment_nar_hash",
    "environment_closure_size",
    "environment_versions_sha256",
    "workflow_run_id",
    "workflow_attempt",
    "job_status",
];

#[derive(Debug)]
struct DefinitionIdentity {
    sha256: String,
    flake_nix_sha256: String,
    flake_lock_sha256: String,
    closure_manifest_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Versions {
    python: String,
    packages: BTreeMap<String, String>,
    scientific_claim: String,
}

#[derive(Debug, Serialize)]
struct Receipt {
    protocol: &'static str,
    verdict: &'static str,
    scientific_claim: &'static str,
    authority: &'static str,
    environment_reuse_authorized: bool,
    a1c_execution_authorized: bool,
    qualification_receipt_sha256: String,
    versions_sha256: String,
    current_environment_definition_sha256: String,
    qualifying_source_head: String,
    realized_environment_store_path: String,
    realized_environment_nar_hash: String,
    realized_environment_closure_size: u64,
    nix_version: &'static str,
}

#[derive(Debug, Serialize)]
struct InvalidReceipt {
    protocol: &'static str,
    verdict: &'static str,
    scientific_claim: &'static str,
    authority: &'static str,
    environment_reuse_authorized: bool,
    a1c_execution_authorized: bool,
    error: String,
}

fn read_regular_file(path: &Path, max_bytes: u64) -> Result<Vec<u8>, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("{}: metadata failed: {error}", path.display()))?;
    if metadata.file_type().is_symlink() {
        return Err(format!("{}: symlinks are forbidden", path.display()));
    }
    if !metadata.file_type().is_file() {
        return Err(format!("{}: not a regular file", path.display()));
    }
    if metadata.len() > max_bytes {
        return Err(format!("{}: file exceeds {max_bytes} bytes", path.display()));
    }
    let bytes = fs::read(path).map_err(|error| format!("{}: read failed: {error}", path.display()))?;
    if bytes.len() as u64 != metadata.len() {
        return Err(format!("{}: file changed size while reading", path.display()));
    }
    Ok(bytes)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_sha256(value: &str) -> bool {
    is_lower_hex(value, 64)
}

fn is_git_oid(value: &str) -> bool {
    (value.len() == 40 || value.len() == 64)
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn run_text(program: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| format!("failed to run {program}: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("{program} {:?} failed: {}", args, stderr.trim()));
    }
    String::from_utf8(output.stdout)
        .map(|text| text.trim().to_owned())
        .map_err(|error| format!("{program} output was not UTF-8: {error}"))
}

fn repository_root() -> Result<PathBuf, String> {
    let root = run_text("git", &["rev-parse", "--show-toplevel"])?;
    if root.is_empty() {
        return Err("git returned an empty repository root".into());
    }
    Ok(PathBuf::from(root))
}

fn parse_receipt(bytes: &[u8]) -> Result<BTreeMap<String, String>, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| format!("environment receipt is not UTF-8: {error}"))?;
    let allowed: BTreeSet<&str> = RECEIPT_KEYS.iter().copied().collect();
    let mut values = BTreeMap::new();
    for (index, line) in text.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| format!("receipt line {} is not key=value", index + 1))?;
        if !allowed.contains(key) {
            return Err(format!("unknown environment receipt key: {key}"));
        }
        if values.insert(key.to_owned(), value.to_owned()).is_some() {
            return Err(format!("duplicate environment receipt key: {key}"));
        }
    }
    for key in RECEIPT_KEYS {
        if !values.contains_key(*key) {
            return Err(format!("missing environment receipt key: {key}"));
        }
    }
    if values.len() != RECEIPT_KEYS.len() {
        return Err("environment receipt has unexpected key cardinality".into());
    }
    Ok(values)
}

fn field<'a>(receipt: &'a BTreeMap<String, String>, key: &str) -> Result<&'a str, String> {
    receipt
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("missing environment receipt key: {key}"))
}

fn compute_definition(root: &Path) -> Result<DefinitionIdentity, String> {
    let flake_nix = read_regular_file(&root.join(FLAKE_RELATIVE), MAX_TEXT_BYTES)?;
    let flake_lock = read_regular_file(&root.join(LOCK_RELATIVE), MAX_TEXT_BYTES)?;
    let closure_manifest = read_regular_file(&root.join(CLOSURE_RELATIVE), MAX_TEXT_BYTES)?;

    let flake_nix_sha256 = sha256_hex(&flake_nix);
    let flake_lock_sha256 = sha256_hex(&flake_lock);
    let closure_manifest_sha256 = sha256_hex(&closure_manifest);
    let definition = format!(
        "protocol={DEFINITION_PROTOCOL}\nplatform=x86_64-linux\nnix_version={NIX_VERSION}\nflake_nix_sha256={flake_nix_sha256}\nflake_lock_sha256={flake_lock_sha256}\nclosure_manifest_sha256={closure_manifest_sha256}\n"
    );

    Ok(DefinitionIdentity {
        sha256: sha256_hex(definition.as_bytes()),
        flake_nix_sha256,
        flake_lock_sha256,
        closure_manifest_sha256,
    })
}

fn validate_receipt(
    receipt: &BTreeMap<String, String>,
    definition: &DefinitionIdentity,
) -> Result<(), String> {
    for (key, expected) in [
        ("protocol", RECEIPT_PROTOCOL),
        ("verdict", "PASS"),
        ("authority", "environment-qualification-only"),
        ("scientific_claim", "NONE"),
        ("reuse_policy", REUSE_POLICY),
        ("nix_version", NIX_VERSION),
        ("job_status", "success"),
    ] {
        if field(receipt, key)? != expected {
            return Err(format!("unexpected environment receipt {key}"));
        }
    }

    if field(receipt, "environment_definition_sha256")? != definition.sha256
        || field(receipt, "flake_nix_sha256")? != definition.flake_nix_sha256
        || field(receipt, "flake_lock_sha256")? != definition.flake_lock_sha256
        || field(receipt, "closure_manifest_sha256")? != definition.closure_manifest_sha256
    {
        return Err("qualified receipt does not match the current environment definition".into());
    }

    for key in [
        "environment_definition_sha256",
        "flake_nix_sha256",
        "flake_lock_sha256",
        "closure_manifest_sha256",
        "flake_metadata_sha256",
        "environment_versions_sha256",
    ] {
        if !is_sha256(field(receipt, key)?) {
            return Err(format!("malformed SHA-256 in environment receipt: {key}"));
        }
    }

    if !is_git_oid(field(receipt, "source_head")?) || !is_git_oid(field(receipt, "source_tree")?) {
        return Err("environment qualification source HEAD/TREE is malformed".into());
    }
    let source_base = field(receipt, "source_base")?;
    if !source_base.is_empty() && !is_git_oid(source_base) {
        return Err("environment qualification source base is malformed".into());
    }

    let store_path = field(receipt, "environment_store_path")?;
    if !store_path.starts_with("/nix/store/") || store_path.contains('\n') {
        return Err("invalid qualified environment store path".into());
    }
    let nar_hash = field(receipt, "environment_nar_hash")?;
    if !nar_hash.starts_with("sha256-") || nar_hash.len() <= "sha256-".len() {
        return Err("invalid qualified environment NAR hash".into());
    }
    let closure_size = field(receipt, "environment_closure_size")?
        .parse::<u64>()
        .map_err(|_| "environment closure size is not an unsigned integer".to_owned())?;
    if closure_size == 0 {
        return Err("environment closure size must be positive".into());
    }
    for key in ["workflow_run_id", "workflow_attempt"] {
        let value = field(receipt, key)?
            .parse::<u64>()
            .map_err(|_| format!("{key} is not an unsigned integer"))?;
        if value == 0 {
            return Err(format!("{key} must be positive"));
        }
    }
    Ok(())
}

fn validate_versions(bytes: &[u8], expected_sha256: &str) -> Result<(), String> {
    if sha256_hex(bytes) != expected_sha256 {
        return Err("package-version evidence SHA-256 mismatch".into());
    }
    let versions: Versions = serde_json::from_slice(bytes)
        .map_err(|error| format!("invalid package-version JSON: {error}"))?;
    if versions.scientific_claim != "NONE" || !versions.python.starts_with("3.11.") {
        return Err("unexpected Python environment identity".into());
    }
    let expected = BTreeMap::from([
        ("Py-BOBYQA".to_owned(), "1.4.1".to_owned()),
        ("camb".to_owned(), "1.6.6".to_owned()),
        ("cobaya".to_owned(), "3.6.2".to_owned()),
        ("getdist".to_owned(), "1.7.4".to_owned()),
        ("iminuit".to_owned(), "2.32.0".to_owned()),
    ]);
    if versions.packages != expected {
        return Err("qualified package-version set does not match the frozen environment".into());
    }
    Ok(())
}

fn realize_current_environment(root: &Path) -> Result<(String, String, u64), String> {
    let nix_version = run_text("nix", &["--version"])?;
    if nix_version != NIX_VERSION {
        return Err(format!("unexpected live Nix version: {nix_version}"));
    }

    let flake = root.join(FLAKE_DIR_RELATIVE);
    let target = format!("{}#environment", flake.display());
    let store_path = run_text(
        "nix",
        &[
            "build",
            &target,
            "--no-update-lock-file",
            "--print-out-paths",
            "--no-link",
        ],
    )?;
    if store_path.is_empty() || store_path.lines().count() != 1 || !store_path.starts_with("/nix/store/") {
        return Err("Nix did not realize exactly one environment store path".into());
    }

    let nar_hash = run_text("nix", &["hash", "path", &store_path])?;
    let path_info = run_text("nix", &["path-info", "-S", &store_path])?;
    let mut lines = path_info.lines();
    let line = lines
        .next()
        .ok_or_else(|| "nix path-info returned no rows".to_owned())?;
    if lines.next().is_some() {
        return Err("nix path-info returned multiple rows".into());
    }
    let fields: Vec<_> = line.split_whitespace().collect();
    if fields.len() < 2 || fields[0] != store_path {
        return Err("unexpected nix path-info output".into());
    }
    let closure_size = fields[1]
        .parse::<u64>()
        .map_err(|_| "nix path-info closure size is not an unsigned integer".to_owned())?;
    if closure_size == 0 {
        return Err("realized environment closure size must be positive".into());
    }
    Ok((store_path, nar_hash, closure_size))
}

fn execute(receipt_path: &Path, versions_path: &Path) -> Result<Receipt, String> {
    let receipt_bytes = read_regular_file(receipt_path, MAX_TEXT_BYTES)?;
    let versions_bytes = read_regular_file(versions_path, MAX_TEXT_BYTES)?;
    let parsed = parse_receipt(&receipt_bytes)?;
    let root = repository_root()?;
    let definition = compute_definition(&root)?;
    validate_receipt(&parsed, &definition)?;
    validate_versions(
        &versions_bytes,
        field(&parsed, "environment_versions_sha256")?,
    )?;

    let (store_path, nar_hash, closure_size) = realize_current_environment(&root)?;
    if store_path != field(&parsed, "environment_store_path")?
        || nar_hash != field(&parsed, "environment_nar_hash")?
        || closure_size.to_string() != field(&parsed, "environment_closure_size")?
    {
        return Err("current environment realization does not match the qualified receipt".into());
    }

    Ok(Receipt {
        protocol: PROTOCOL,
        verdict: "PASS",
        scientific_claim: "NONE",
        authority: AUTHORITY,
        environment_reuse_authorized: true,
        a1c_execution_authorized: false,
        qualification_receipt_sha256: sha256_hex(&receipt_bytes),
        versions_sha256: sha256_hex(&versions_bytes),
        current_environment_definition_sha256: definition.sha256,
        qualifying_source_head: field(&parsed, "source_head")?.to_owned(),
        realized_environment_store_path: store_path,
        realized_environment_nar_hash: nar_hash,
        realized_environment_closure_size: closure_size,
        nix_version: NIX_VERSION,
    })
}

fn write_json<T: Serialize>(value: &T) -> Result<(), String> {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    serde_json::to_writer_pretty(&mut out, value)
        .map_err(|error| format!("failed to serialize receipt: {error}"))?;
    writeln!(out).map_err(|error| format!("failed to write receipt: {error}"))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        let invalid = InvalidReceipt {
            protocol: PROTOCOL,
            verdict: "INVALID",
            scientific_claim: "NONE",
            authority: AUTHORITY,
            environment_reuse_authorized: false,
            a1c_execution_authorized: false,
            error: "usage: de001a-a1e-environment <environment-receipt.txt> <environment-versions.json>".into(),
        };
        let _ = write_json(&invalid);
        std::process::exit(2);
    }

    match execute(Path::new(&args[1]), Path::new(&args[2])) {
        Ok(receipt) => {
            if let Err(error) = write_json(&receipt) {
                eprintln!("{error}");
                std::process::exit(2);
            }
        }
        Err(error) => {
            let invalid = InvalidReceipt {
                protocol: PROTOCOL,
                verdict: "INVALID",
                scientific_claim: "NONE",
                authority: AUTHORITY,
                environment_reuse_authorized: false,
                a1c_execution_authorized: false,
                error,
            };
            let _ = write_json(&invalid);
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_receipt_text() -> String {
        [
            ("protocol", RECEIPT_PROTOCOL),
            ("verdict", "PASS"),
            ("authority", "environment-qualification-only"),
            ("scientific_claim", "NONE"),
            ("reuse_policy", REUSE_POLICY),
            ("source_head", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            ("source_tree", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
            ("source_base", "cccccccccccccccccccccccccccccccccccccccc"),
            ("environment_definition_sha256", &"d".repeat(64)),
            ("flake_nix_sha256", &"e".repeat(64)),
            ("flake_lock_sha256", &"f".repeat(64)),
            ("closure_manifest_sha256", &"a".repeat(64)),
            ("flake_metadata_sha256", &"b".repeat(64)),
            ("nix_version", NIX_VERSION),
            ("environment_store_path", "/nix/store/example-environment"),
            ("environment_nar_hash", "sha256-example"),
            ("environment_closure_size", "42"),
            ("environment_versions_sha256", &"c".repeat(64)),
            ("workflow_run_id", "1"),
            ("workflow_attempt", "1"),
            ("job_status", "success"),
        ]
        .into_iter()
        .map(|(key, value)| format!("{key}={value}\n"))
        .collect()
    }

    #[test]
    fn parser_rejects_unknown_and_duplicate_keys() {
        let base = valid_receipt_text();
        assert!(parse_receipt(format!("{base}unknown=value\n").as_bytes()).is_err());
        assert!(parse_receipt(format!("{base}protocol=again\n").as_bytes()).is_err());
    }

    #[test]
    fn version_set_is_exact() {
        let bytes = br#"{"packages":{"Py-BOBYQA":"1.4.1","camb":"1.6.6","cobaya":"3.6.2","getdist":"1.7.4","iminuit":"2.32.0"},"python":"3.11.9","scientific_claim":"NONE"}"#;
        assert!(validate_versions(bytes, &sha256_hex(bytes)).is_ok());

        let wrong = br#"{"packages":{"Py-BOBYQA":"1.4.1","camb":"1.6.6","cobaya":"3.6.1","getdist":"1.7.4","iminuit":"2.32.0"},"python":"3.11.9","scientific_claim":"NONE"}"#;
        assert!(validate_versions(wrong, &sha256_hex(wrong)).is_err());
    }
}
