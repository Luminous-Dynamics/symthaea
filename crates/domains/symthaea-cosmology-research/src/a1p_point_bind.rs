// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! DE-001A1P fixed-point provenance binding.
//!
//! This program performs no cosmology. It proves that the fixed Omega_m,
//! h*r_d, and chi2_BAO values in the A1R manifest are actually present in the
//! hash-bound official DESI best-fit text authenticated by A0.

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

const MAX_SMALL_FILE_BYTES: u64 = 1024 * 1024;
const PROTOCOL: &str = "DE-001A1P-POINT-BINDING-v1";
const AUTHORITY: &str = "parameter-provenance-binding-only";

#[derive(Debug, Serialize)]
struct FieldBinding {
    field: &'static str,
    manifest_value: f64,
    upstream_value: f64,
    exact_f64_match: bool,
}

#[derive(Debug, Serialize)]
struct Receipt {
    protocol: &'static str,
    verdict: &'static str,
    scientific_claim: &'static str,
    authority: &'static str,
    a1r_manifest_sha256: String,
    a0_receipt_sha256: String,
    bestfit_sha256: String,
    bestfit_role: String,
    bindings: Vec<FieldBinding>,
}

#[derive(Debug, Serialize)]
struct InvalidReceipt {
    protocol: &'static str,
    verdict: &'static str,
    scientific_claim: &'static str,
    authority: &'static str,
    error: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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
        return Err(format!(
            "{}: file size {} exceeds limit {max_bytes}",
            path.display(),
            metadata.len()
        ));
    }
    let bytes = fs::read(path).map_err(|error| format!("{}: read failed: {error}", path.display()))?;
    if bytes.len() as u64 != metadata.len() {
        return Err(format!("{}: file changed size while reading", path.display()));
    }
    Ok(bytes)
}

fn require_object<'a>(value: &'a Value, key: &str) -> Result<&'a serde_json::Map<String, Value>, String> {
    value
        .get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("missing or non-object field {key:?}"))
}

fn require_str<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing or non-string field {key:?}"))
}

fn require_u64(value: &Value, key: &str) -> Result<u64, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("missing or non-u64 field {key:?}"))
}

fn require_f64(value: &Value, key: &str) -> Result<f64, String> {
    let number = value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("missing or non-number field {key:?}"))?;
    if !number.is_finite() {
        return Err(format!("field {key:?} must be finite"));
    }
    Ok(number)
}

fn parse_a1r_manifest(bytes: &[u8]) -> Result<(String, u64, String, f64, f64, f64), String> {
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("invalid A1R manifest JSON: {error}"))?;
    if value.get("schema_version").and_then(Value::as_u64) != Some(1) {
        return Err("A1R manifest schema_version must be 1".into());
    }
    if require_str(&value, "protocol")? != "DE-001A1R-RUST-ORACLE-v1" {
        return Err("unexpected A1R protocol".into());
    }
    if require_str(&value, "scientific_claim")? != "NONE" {
        return Err("A1R manifest must declare scientific_claim=NONE".into());
    }

    let source = Value::Object(require_object(&value, "source")?.clone());
    let role = require_str(&source, "bestfit_role")?.to_owned();
    let size = require_u64(&source, "bestfit_size")?;
    let digest = require_str(&source, "bestfit_sha256")?.to_owned();
    if role != "reference-bestfit-text" || size != 902 || !is_lower_hex_sha256(&digest) {
        return Err("A1R best-fit source identity is not frozen as expected".into());
    }

    let parameters = Value::Object(require_object(&value, "parameters")?.clone());
    let omega_m = require_f64(&parameters, "omega_m")?;
    let h_r_d_mpc = require_f64(&parameters, "h_r_d_mpc")?;
    let reference = Value::Object(require_object(&value, "reference")?.clone());
    let chi2_bao = require_f64(&reference, "chi2_bao")?;
    Ok((role, size, digest, omega_m, h_r_d_mpc, chi2_bao))
}

fn validate_a0_receipt(
    bytes: &[u8],
    role: &str,
    expected_size: u64,
    expected_sha256: &str,
) -> Result<(), String> {
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("invalid A0 receipt JSON: {error}"))?;
    if require_str(&value, "protocol")? != "DE-001A0-BYTE-INTEGRITY-v1"
        || require_str(&value, "verdict")? != "PASS"
        || require_str(&value, "scientific_claim")? != "NONE"
    {
        return Err("A1P requires a clean A0 PASS receipt".into());
    }
    let errors = value
        .get("errors")
        .and_then(Value::as_array)
        .ok_or_else(|| "A0 receipt errors must be an array".to_owned())?;
    if !errors.is_empty() {
        return Err("A0 receipt contains errors".into());
    }
    let artifacts = value
        .get("artifacts")
        .and_then(Value::as_array)
        .ok_or_else(|| "A0 receipt artifacts must be an array".to_owned())?;
    let matching: Vec<_> = artifacts
        .iter()
        .filter(|artifact| artifact.get("role").and_then(Value::as_str) == Some(role))
        .collect();
    if matching.len() != 1 {
        return Err(format!("A0 receipt must contain exactly one {role:?} artifact"));
    }
    let artifact = matching[0];
    let expected_digest = artifact
        .get("expected_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| "A0 artifact missing expected_sha256".to_owned())?;
    let actual_digest = artifact
        .get("actual_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| "A0 artifact missing actual_sha256".to_owned())?;
    let status = artifact
        .get("status")
        .and_then(Value::as_str)
        .ok_or_else(|| "A0 artifact missing status".to_owned())?;
    let expected_len = artifact
        .get("expected_size")
        .and_then(Value::as_u64)
        .ok_or_else(|| "A0 artifact missing expected_size".to_owned())?;
    let actual_len = artifact
        .get("actual_size")
        .and_then(Value::as_u64)
        .ok_or_else(|| "A0 artifact missing actual_size".to_owned())?;
    if status != "PASS"
        || expected_len != expected_size
        || actual_len != expected_size
        || expected_digest != expected_sha256
        || actual_digest != expected_sha256
    {
        return Err("A0 best-fit artifact does not match the A1R source identity".into());
    }
    Ok(())
}

fn parse_bestfit_table(bytes: &[u8]) -> Result<BTreeMap<String, f64>, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| format!("best-fit text is not UTF-8: {error}"))?;
    let lines: Vec<_> = text.lines().collect();
    let mut header_index = None;
    let mut headers = Vec::new();

    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if !trimmed.starts_with('#') {
            continue;
        }
        let fields: Vec<_> = trimmed.trim_start_matches('#').split_whitespace().collect();
        let field_set: BTreeSet<_> = fields.iter().copied().collect();
        if field_set.contains("omm") && field_set.contains("hrdrag") && field_set.contains("chi2__BAO") {
            if field_set.len() != fields.len() {
                return Err("best-fit header contains duplicate field names".into());
            }
            header_index = Some(index);
            headers = fields;
            break;
        }
    }

    let header_index = header_index.ok_or_else(|| {
        "best-fit text has no header containing exact fields omm, hrdrag, chi2__BAO".to_owned()
    })?;
    let data_line = lines
        .iter()
        .skip(header_index + 1)
        .map(|line| line.trim())
        .find(|line| !line.is_empty() && !line.starts_with('#'))
        .ok_or_else(|| "best-fit table has no data row after its header".to_owned())?;
    let values: Vec<_> = data_line.split_whitespace().collect();
    if values.len() != headers.len() {
        return Err(format!(
            "best-fit row/header width mismatch: {} values for {} fields",
            values.len(),
            headers.len()
        ));
    }

    let mut parsed = BTreeMap::new();
    for (field, token) in headers.into_iter().zip(values) {
        let number: f64 = token
            .parse()
            .map_err(|_| format!("best-fit field {field:?} is not a finite float"))?;
        if !number.is_finite() {
            return Err(format!("best-fit field {field:?} is not finite"));
        }
        parsed.insert(field.to_owned(), number);
    }
    Ok(parsed)
}

fn exact_binding(field: &'static str, manifest_value: f64, table: &BTreeMap<String, f64>) -> Result<FieldBinding, String> {
    let upstream_value = *table
        .get(field)
        .ok_or_else(|| format!("best-fit table is missing field {field:?}"))?;
    let exact_f64_match = upstream_value.to_bits() == manifest_value.to_bits();
    Ok(FieldBinding {
        field,
        manifest_value,
        upstream_value,
        exact_f64_match,
    })
}

fn execute(manifest_path: &Path, a0_path: &Path, bestfit_path: &Path) -> Result<Receipt, String> {
    let manifest_bytes = read_regular_file(manifest_path, MAX_SMALL_FILE_BYTES)?;
    let manifest_sha256 = sha256_hex(&manifest_bytes);
    let (role, expected_size, expected_sha256, omega_m, h_r_d_mpc, chi2_bao) =
        parse_a1r_manifest(&manifest_bytes)?;

    let a0_bytes = read_regular_file(a0_path, MAX_SMALL_FILE_BYTES)?;
    let a0_sha256 = sha256_hex(&a0_bytes);
    validate_a0_receipt(&a0_bytes, &role, expected_size, &expected_sha256)?;

    let bestfit_bytes = read_regular_file(bestfit_path, MAX_SMALL_FILE_BYTES)?;
    if bestfit_bytes.len() as u64 != expected_size {
        return Err(format!(
            "best-fit byte count mismatch: expected {expected_size}, got {}",
            bestfit_bytes.len()
        ));
    }
    let bestfit_sha256 = sha256_hex(&bestfit_bytes);
    if bestfit_sha256 != expected_sha256 {
        return Err(format!(
            "best-fit SHA-256 mismatch: expected {expected_sha256}, got {bestfit_sha256}"
        ));
    }

    let table = parse_bestfit_table(&bestfit_bytes)?;
    let bindings = vec![
        exact_binding("omm", omega_m, &table)?,
        exact_binding("hrdrag", h_r_d_mpc, &table)?,
        exact_binding("chi2__BAO", chi2_bao, &table)?,
    ];
    if bindings.iter().any(|binding| !binding.exact_f64_match) {
        let mismatches: Vec<_> = bindings
            .iter()
            .filter(|binding| !binding.exact_f64_match)
            .map(|binding| binding.field)
            .collect();
        return Err(format!(
            "A1R manifest values are not exact transcriptions of the official best-fit row: {mismatches:?}"
        ));
    }

    Ok(Receipt {
        protocol: PROTOCOL,
        verdict: "PASS",
        scientific_claim: "NONE",
        authority: AUTHORITY,
        a1r_manifest_sha256: manifest_sha256,
        a0_receipt_sha256: a0_sha256,
        bestfit_sha256,
        bestfit_role: role,
        bindings,
    })
}

fn write_json<T: Serialize>(value: &T) -> io::Result<()> {
    let stdout = io::stdout();
    let mut lock = stdout.lock();
    serde_json::to_writer_pretty(&mut lock, value)?;
    writeln!(&mut lock)?;
    Ok(())
}

fn run() -> Result<i32, String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: de001a-a1p-point-bind A1R_MANIFEST.json A0_RECEIPT.json BESTFIT.txt".into());
    }
    let receipt = execute(
        Path::new(&args[0]),
        Path::new(&args[1]),
        Path::new(&args[2]),
    )?;
    write_json(&receipt).map_err(|error| format!("failed to write receipt: {error}"))?;
    Ok(0)
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            let receipt = InvalidReceipt {
                protocol: PROTOCOL,
                verdict: "INVALID",
                scientific_claim: "NONE",
                authority: AUTHORITY,
                error,
            };
            let _ = write_json(&receipt);
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_required_fields_from_reordered_table() {
        let input = b"# chi2__BAO other hrdrag omm\n10.282299 5 101.54786 0.29717787\n";
        let table = parse_bestfit_table(input).unwrap();
        assert_eq!(table["omm"].to_bits(), 0.29717787_f64.to_bits());
        assert_eq!(table["hrdrag"].to_bits(), 101.54786_f64.to_bits());
        assert_eq!(table["chi2__BAO"].to_bits(), 10.282299_f64.to_bits());
    }

    #[test]
    fn rejects_missing_exact_header_names() {
        let input = b"# chi2__BAO H0rdrag omm\n10.282299 10154.786 0.29717787\n";
        assert!(parse_bestfit_table(input).is_err());
    }

    #[test]
    fn exact_binding_rejects_additional_upstream_precision() {
        let table = BTreeMap::from([("omm".to_owned(), 0.297177871_f64)]);
        let binding = exact_binding("omm", 0.29717787, &table).unwrap();
        assert!(!binding.exact_f64_match);
    }
}
