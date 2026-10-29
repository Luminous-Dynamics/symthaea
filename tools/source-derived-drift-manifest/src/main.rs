//! Fail-closed validator for source-derived semisynthetic drift benchmark manifests.
//!
//! The manifest describes provenance and experimental design. Dataset payloads are deliberately
//! not embedded in this tool. Output bytes and the canonical manifest are hashed with the host's
//! standard `sha256sum` utility when a receipt is validated.

use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::Path,
    process::{Command, ExitCode},
};

const MANIFEST_KEYS: &[&str] = &[
    "schema_version",
    "run_id",
    "source_id",
    "source_url",
    "source_version",
    "source_license_spdx",
    "source_license_evidence_url",
    "source_sha256",
    "row_order_semantics",
    "feature_columns",
    "target_column",
    "label_semantics",
    "transform_family",
    "transform_severity",
    "transition_shape",
    "change_start",
    "change_end",
    "seed",
    "calibration_sources",
    "heldout_source",
    "detector_id",
    "detector_version",
    "detector_config_sha256",
    "policy_version",
    "label_reveal_policy",
    "label_delay_steps",
    "label_observation_fraction",
];

const RECEIPT_KEYS: &[&str] = &[
    "schema_version",
    "run_id",
    "manifest_sha256",
    "source_sha256",
    "output_sha256",
    "artifact_path",
];

const HASH_LENGTH: usize = 64;

type Fields = BTreeMap<String, String>;

fn fail(message: impl AsRef<str>) -> String {
    message.as_ref().to_owned()
}

fn parse_fields(input: &str, keys: &[&str]) -> Result<Fields, String> {
    if !input.is_ascii() {
        return Err(fail("manifest must be ASCII; encode non-ASCII source metadata separately"));
    }
    if !input.ends_with('\n') || input.contains('\r') {
        return Err(fail("manifest must use LF line endings and end with exactly a newline"));
    }

    let allowed: BTreeSet<&str> = keys.iter().copied().collect();
    let mut fields = Fields::new();
    for (index, line) in input.lines().enumerate() {
        if line.is_empty() {
            return Err(fail(format!("blank line at line {}", index + 1)));
        }
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| fail(format!("expected key=value at line {}", index + 1)))?;
        if !allowed.contains(key) {
            return Err(fail(format!("unknown key at line {}: {key}", index + 1)));
        }
        if value.is_empty() || value.trim() != value || value.contains('=') {
            return Err(fail(format!("empty, padded, or ambiguous value for {key}")));
        }
        if fields.insert(key.to_owned(), value.to_owned()).is_some() {
            return Err(fail(format!("duplicate key: {key}")));
        }
    }

    if fields.len() != keys.len() {
        let missing: Vec<&str> = keys
            .iter()
            .copied()
            .filter(|key| !fields.contains_key(*key))
            .collect();
        return Err(fail(format!("missing required keys: {}", missing.join(", "))));
    }
    Ok(fields)
}

fn canonical_bytes(fields: &Fields, keys: &[&str]) -> Result<String, String> {
    let mut output = String::new();
    for key in keys {
        let value = fields
            .get(*key)
            .ok_or_else(|| fail(format!("missing required key while canonicalizing: {key}")))?;
        output.push_str(key);
        output.push('=');
        output.push_str(value);
        output.push('\n');
    }
    Ok(output)
}

fn required<'a>(fields: &'a Fields, key: &str) -> Result<&'a str, String> {
    fields
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| fail(format!("missing field: {key}")))
}

fn is_token(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-:".contains(&byte))
}

fn is_spdx_token(value: &str) -> bool {
    is_token(value)
}

fn validate_sha256(value: &str, field: &str) -> Result<(), String> {
    if value.len() != HASH_LENGTH
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(fail(format!("{field} must be 64 lowercase hexadecimal characters")));
    }
    Ok(())
}

fn canonical_u64(value: &str, field: &str) -> Result<u64, String> {
    let parsed = value
        .parse::<u64>()
        .map_err(|_| fail(format!("{field} must be a non-negative integer")))?;
    if parsed.to_string() != value {
        return Err(fail(format!("{field} is not canonically formatted")));
    }
    Ok(parsed)
}

fn canonical_unit_float(value: &str, field: &str) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| fail(format!("{field} must be a decimal in [0, 1]")))?;
    if !parsed.is_finite() || !(0.0..=1.0).contains(&parsed) {
        return Err(fail(format!("{field} must be finite and in [0, 1]")));
    }
    if format!("{parsed:.6}") != value {
        return Err(fail(format!("{field} must use canonical six-decimal formatting")));
    }
    Ok(parsed)
}

fn comma_list(value: &str, field: &str) -> Result<Vec<&str>, String> {
    let items: Vec<&str> = value.split(',').collect();
    if items.is_empty() || items.iter().any(|item| !is_token(item)) {
        return Err(fail(format!("{field} must be a comma-separated list of canonical tokens")));
    }
    let unique: BTreeSet<&str> = items.iter().copied().collect();
    if unique.len() != items.len() {
        return Err(fail(format!("{field} contains duplicate entries")));
    }
    let sorted: Vec<&str> = unique.iter().copied().collect();
    if items != sorted {
        return Err(fail(format!("{field} must be sorted lexicographically")));
    }
    Ok(items)
}

fn validate_manifest(fields: &Fields) -> Result<(), String> {
    if required(fields, "schema_version")? != "1" {
        return Err(fail("unsupported manifest schema_version (expected 1)"));
    }

    for key in [
        "run_id",
        "source_id",
        "source_version",
        "detector_id",
        "detector_version",
        "policy_version",
    ] {
        if !is_token(required(fields, key)?) {
            return Err(fail(format!("{key} must be a canonical identifier token")));
        }
    }

    let source_url = required(fields, "source_url")?;
    if !source_url.starts_with("https://")
        || source_url.contains(' ')
        || source_url.contains('#')
    {
        return Err(fail("source_url must be an HTTPS source URL without a fragment or spaces"));
    }

    let license = required(fields, "source_license_spdx")?;
    if !is_spdx_token(license)
        || ["UNKNOWN", "UNVERIFIED", "NOASSERTION"].contains(&license)
    {
        return Err(fail("source_license_spdx must be a verified SPDX identifier, not an unknown placeholder"));
    }

    let license_url = required(fields, "source_license_evidence_url")?;
    if !license_url.starts_with("https://")
        || license_url.contains(' ')
        || license_url.contains('#')
    {
        return Err(fail("source_license_evidence_url must be an HTTPS evidence URL without a fragment or spaces"));
    }

    validate_sha256(required(fields, "source_sha256")?, "source_sha256")?;
    validate_sha256(
        required(fields, "detector_config_sha256")?,
        "detector_config_sha256",
    )?;

    match required(fields, "row_order_semantics")? {
        "temporal" | "source_ordered" | "seeded_permutation" => {}
        other => return Err(fail(format!("unsupported row_order_semantics: {other}"))),
    }

    let features = comma_list(required(fields, "feature_columns")?, "feature_columns")?;
    let target = required(fields, "target_column")?;
    if target != "NA" && features.contains(&target) {
        return Err(fail("target_column must not also appear in feature_columns"));
    }

    match required(fields, "label_semantics")? {
        "none" if target != "NA" => {
            return Err(fail("label_semantics=none requires target_column=NA"))
        }
        "binary" | "categorical" | "continuous" if target == "NA" => {
            return Err(fail("a declared label_semantics requires a non-NA target_column"))
        }
        "none" | "binary" | "categorical" | "continuous" => {}
        other => return Err(fail(format!("unsupported label_semantics: {other}"))),
    }

    let transform = required(fields, "transform_family")?;
    let severity = canonical_unit_float(required(fields, "transform_severity")?, "transform_severity")?;
    let transition = required(fields, "transition_shape")?;
    let start = required(fields, "change_start")?;
    let end = required(fields, "change_end")?;
    if transform == "none" {
        if severity != 0.0 || transition != "none" || start != "NA" || end != "NA" {
            return Err(fail("zero-shift sham requires transform_family=none, severity=0.000000, transition_shape=none, and NA change bounds"));
        }
    } else {
        match transform {
            "class_prior" | "label_swap" | "feature_permutation" | "feature_filtering"
            | "covariate_affine" | "sensor_noise" => {}
            other => return Err(fail(format!("unsupported transform_family: {other}"))),
        }
        if severity == 0.0 {
            return Err(fail("non-sham transform requires transform_severity greater than zero"));
        }
        match transition {
            "abrupt" => {
                let start_value = canonical_u64(start, "change_start")?;
                let end_value = canonical_u64(end, "change_end")?;
                if start_value == 0 || start_value != end_value {
                    return Err(fail("abrupt transitions require identical positive change_start and change_end"));
                }
            }
            "gradual" => {
                let start_value = canonical_u64(start, "change_start")?;
                let end_value = canonical_u64(end, "change_end")?;
                if start_value == 0 || end_value <= start_value {
                    return Err(fail("gradual transitions require 0 < change_start < change_end"));
                }
            }
            other => return Err(fail(format!("non-sham transform requires abrupt or gradual transition_shape, got {other}"))),
        }
        if matches!(transform, "class_prior" | "label_swap")
            && !matches!(required(fields, "label_semantics")?, "binary" | "categorical")
        {
            return Err(fail("class_prior and label_swap require binary or categorical labels"));
        }
    }

    canonical_u64(required(fields, "seed")?, "seed")?;
    let calibration_sources = comma_list(required(fields, "calibration_sources")?, "calibration_sources")?;
    let heldout_source = required(fields, "heldout_source")?;
    if !is_token(heldout_source) {
        return Err(fail("heldout_source must be a canonical source identifier"));
    }
    if heldout_source != required(fields, "source_id")? {
        return Err(fail("heldout_source must equal source_id for the evaluated source"));
    }
    if calibration_sources.contains(&heldout_source) {
        return Err(fail("held-out source leaks into calibration_sources"));
    }

    let reveal = required(fields, "label_reveal_policy")?;
    let delay = canonical_u64(required(fields, "label_delay_steps")?, "label_delay_steps")?;
    let observation_fraction = canonical_unit_float(
        required(fields, "label_observation_fraction")?,
        "label_observation_fraction",
    )?;
    match reveal {
        "all_immediate" => {
            if observation_fraction != 1.0 || delay != 0 {
                return Err(fail("all_immediate requires label_observation_fraction=1.000000 and label_delay_steps=0"));
            }
        }
        "delayed_all" => {
            if observation_fraction != 1.0 || delay == 0 {
                return Err(fail("delayed_all requires full label observation and positive label_delay_steps"));
            }
        }
        "randomized_sentinel" => {
            if target == "NA" || observation_fraction <= 0.0 || observation_fraction >= 1.0 {
                return Err(fail("randomized_sentinel requires a target and label_observation_fraction strictly between 0 and 1"));
            }
        }
        "feature_only" => {
            if observation_fraction != 0.0 || delay != 0 {
                return Err(fail("feature_only requires zero observed labels and zero label_delay_steps"));
            }
        }
        other => return Err(fail(format!("unsupported label_reveal_policy: {other}"))),
    }
    if required(fields, "label_semantics")? == "none" && reveal != "feature_only" {
        return Err(fail("label_semantics=none requires label_reveal_policy=feature_only"));
    }

    Ok(())
}

fn validate_receipt(fields: &Fields, manifest: &Fields) -> Result<(), String> {
    if required(fields, "schema_version")? != "1" {
        return Err(fail("unsupported receipt schema_version (expected 1)"));
    }
    if required(fields, "run_id")? != required(manifest, "run_id")? {
        return Err(fail("receipt run_id does not match manifest"));
    }
    validate_sha256(required(fields, "manifest_sha256")?, "manifest_sha256")?;
    validate_sha256(required(fields, "source_sha256")?, "receipt source_sha256")?;
    validate_sha256(required(fields, "output_sha256")?, "output_sha256")?;
    if required(fields, "source_sha256")? != required(manifest, "source_sha256")? {
        return Err(fail("receipt source_sha256 does not match manifest source_sha256"));
    }
    let path = required(fields, "artifact_path")?;
    if path.starts_with('/')
        || path.contains('\\')
        || path.split('/').any(|part| part.is_empty() || part == "." || part == "..")
        || path.bytes().any(|byte| !(byte.is_ascii_alphanumeric() || b"._-/".contains(&byte)))
    {
        return Err(fail("artifact_path must be a canonical safe relative path"));
    }
    Ok(())
}

fn parse_canonical(input: &str, keys: &[&str], label: &str) -> Result<Fields, String> {
    let fields = parse_fields(input, keys)?;
    let canonical = canonical_bytes(&fields, keys)?;
    if canonical != input {
        return Err(fail(format!("{label} is not in canonical key order/format")));
    }
    Ok(fields)
}

fn hash_file(path: &Path) -> Result<String, String> {
    let output = Command::new("sha256sum")
        .arg("--")
        .arg(path)
        .output()
        .map_err(|error| fail(format!("cannot run sha256sum: {error}")))?;
    if !output.status.success() {
        return Err(fail(format!("sha256sum failed for {}", path.display())));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|_| fail("sha256sum produced non-UTF-8 output"))?;
    let digest = stdout
        .split_whitespace()
        .next()
        .ok_or_else(|| fail("sha256sum returned no digest"))?;
    validate_sha256(digest, "computed SHA-256")?;
    Ok(digest.to_owned())
}

fn usage() {
    eprintln!("Usage:");
    eprintln!("  source-derived-drift-manifest validate MANIFEST");
    eprintln!("  source-derived-drift-manifest canonicalize INPUT OUTPUT");
    eprintln!("  source-derived-drift-manifest validate-receipt MANIFEST RECEIPT OUTPUT");
}

fn execute(args: &[String]) -> Result<(), String> {
    match args {
        [command, manifest_path] if command == "validate" => {
            let input = fs::read_to_string(manifest_path)
                .map_err(|error| fail(format!("cannot read manifest: {error}")))?;
            let fields = parse_canonical(&input, MANIFEST_KEYS, "manifest")?;
            validate_manifest(&fields)?;
            let digest = hash_file(Path::new(manifest_path))?;
            println!(
                "PASS: canonical source-derived manifest run_id={} manifest_sha256={digest}",
                required(&fields, "run_id")?
            );
        }
        [command, input_path, output_path] if command == "canonicalize" => {
            let input = fs::read_to_string(input_path)
                .map_err(|error| fail(format!("cannot read input manifest: {error}")))?;
            let fields = parse_fields(&input, MANIFEST_KEYS)?;
            validate_manifest(&fields)?;
            let canonical = canonical_bytes(&fields, MANIFEST_KEYS)?;
            fs::write(output_path, canonical)
                .map_err(|error| fail(format!("cannot write canonical manifest: {error}")))?;
            let digest = hash_file(Path::new(output_path))?;
            println!("PASS: canonicalized manifest manifest_sha256={digest}");
        }
        [command, manifest_path, receipt_path, output_path] if command == "validate-receipt" => {
            let manifest_text = fs::read_to_string(manifest_path)
                .map_err(|error| fail(format!("cannot read manifest: {error}")))?;
            let manifest = parse_canonical(&manifest_text, MANIFEST_KEYS, "manifest")?;
            validate_manifest(&manifest)?;
            let receipt_text = fs::read_to_string(receipt_path)
                .map_err(|error| fail(format!("cannot read receipt: {error}")))?;
            let receipt = parse_canonical(&receipt_text, RECEIPT_KEYS, "receipt")?;
            validate_receipt(&receipt, &manifest)?;
            let manifest_digest = hash_file(Path::new(manifest_path))?;
            if required(&receipt, "manifest_sha256")? != manifest_digest {
                return Err(fail("receipt manifest_sha256 does not match the manifest bytes"));
            }
            let output_digest = hash_file(Path::new(output_path))?;
            if required(&receipt, "output_sha256")? != output_digest {
                return Err(fail("receipt output_sha256 does not match the output bytes"));
            }
            println!(
                "PASS: receipt run_id={} manifest_sha256={} output_sha256={}",
                required(&receipt, "run_id")?,
                manifest_digest,
                output_digest
            );
        }
        _ => {
            usage();
            return Err(fail("invalid command arguments"));
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match execute(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("FAIL: {error}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_fields() -> Fields {
        let mut fields = Fields::new();
        let values = [
            ("schema_version", "1"),
            ("run_id", "run_001"),
            ("source_id", "electricity_v1"),
            ("source_url", "https://example.org/datasets/electricity"),
            ("source_version", "2026-01"),
            ("source_license_spdx", "CC-BY-4.0"),
            ("source_license_evidence_url", "https://example.org/fixture-license"),
            ("source_sha256", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            ("row_order_semantics", "temporal"),
            ("feature_columns", "demand,price,temperature"),
            ("target_column", "label"),
            ("label_semantics", "binary"),
            ("transform_family", "feature_permutation"),
            ("transform_severity", "0.500000"),
            ("transition_shape", "gradual"),
            ("change_start", "100"),
            ("change_end", "200"),
            ("seed", "42"),
            ("calibration_sources", "asfault_v1,covtype_v1,noaa_v1"),
            ("heldout_source", "electricity_v1"),
            ("detector_id", "cusum_upper"),
            ("detector_version", "0.1.0"),
            ("detector_config_sha256", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
            ("policy_version", "randomized_audit_v1"),
            ("label_reveal_policy", "randomized_sentinel"),
            ("label_delay_steps", "5"),
            ("label_observation_fraction", "0.100000"),
        ];
        for (key, value) in values {
            fields.insert(key.to_owned(), value.to_owned());
        }
        fields
    }

    fn valid_manifest() -> String {
        canonical_bytes(&valid_fields(), MANIFEST_KEYS).unwrap()
    }

    fn set(fields: &mut Fields, key: &str, value: &str) {
        fields.insert(key.to_owned(), value.to_owned());
    }

    #[test]
    fn accepts_canonical_source_heldout_manifest() {
        let text = valid_manifest();
        let fields = parse_canonical(&text, MANIFEST_KEYS, "manifest").unwrap();
        validate_manifest(&fields).unwrap();
    }

    #[test]
    fn canonicalizer_is_deterministic_and_removes_order_ambiguity() {
        let fields = valid_fields();
        let shuffled = fields
            .iter()
            .rev()
            .map(|(key, value)| format!("{key}={value}\n"))
            .collect::<String>();
        let parsed = parse_fields(&shuffled, MANIFEST_KEYS).unwrap();
        validate_manifest(&parsed).unwrap();
        let canonical = canonical_bytes(&parsed, MANIFEST_KEYS).unwrap();
        assert_eq!(canonical, valid_manifest());
        assert!(parse_canonical(&shuffled, MANIFEST_KEYS, "manifest").is_err());
    }

    #[test]
    fn rejects_heldout_source_leakage_and_duplicate_sources() {
        let mut fields = valid_fields();
        set(&mut fields, "calibration_sources", "asfault_v1,electricity_v1,noaa_v1");
        assert!(validate_manifest(&fields).unwrap_err().contains("leaks"));
        set(&mut fields, "calibration_sources", "asfault_v1,asfault_v1,noaa_v1");
        assert!(validate_manifest(&fields).unwrap_err().contains("duplicate"));
    }

    #[test]
    fn rejects_missing_or_unverified_license_and_bad_hashes() {
        let mut fields = valid_fields();
        set(&mut fields, "source_license_spdx", "UNVERIFIED");
        assert!(validate_manifest(&fields).unwrap_err().contains("verified SPDX"));
        set(&mut fields, "source_license_spdx", "CC-BY-4.0");
        set(&mut fields, "source_sha256", "aaaaaaaa");
        assert!(validate_manifest(&fields).unwrap_err().contains("source_sha256"));
    }

    #[test]
    fn sham_requires_explicit_zero_transform_and_no_change_interval() {
        let mut fields = valid_fields();
        set(&mut fields, "transform_family", "none");
        set(&mut fields, "transform_severity", "0.000000");
        set(&mut fields, "transition_shape", "none");
        set(&mut fields, "change_start", "NA");
        set(&mut fields, "change_end", "NA");
        validate_manifest(&fields).unwrap();
        set(&mut fields, "transform_severity", "0.100000");
        assert!(validate_manifest(&fields).unwrap_err().contains("zero-shift sham"));
    }

    #[test]
    fn abrupt_and_gradual_bounds_are_enforced() {
        let mut fields = valid_fields();
        set(&mut fields, "transition_shape", "abrupt");
        set(&mut fields, "change_end", "100");
        validate_manifest(&fields).unwrap();
        set(&mut fields, "change_end", "101");
        assert!(validate_manifest(&fields).unwrap_err().contains("abrupt"));
        set(&mut fields, "transition_shape", "gradual");
        set(&mut fields, "change_start", "200");
        set(&mut fields, "change_end", "100");
        assert!(validate_manifest(&fields).unwrap_err().contains("gradual"));
    }

    #[test]
    fn feature_only_and_randomized_sentinel_budgets_are_explicit() {
        let mut fields = valid_fields();
        set(&mut fields, "label_reveal_policy", "feature_only");
        set(&mut fields, "label_delay_steps", "0");
        set(&mut fields, "label_observation_fraction", "0.000000");
        validate_manifest(&fields).unwrap();
        set(&mut fields, "label_observation_fraction", "0.010000");
        assert!(validate_manifest(&fields).unwrap_err().contains("feature_only"));
        set(&mut fields, "label_reveal_policy", "randomized_sentinel");
        set(&mut fields, "label_observation_fraction", "1.000000");
        assert!(validate_manifest(&fields).unwrap_err().contains("randomized_sentinel"));
    }

    #[test]
    fn rejects_target_as_a_feature_and_unverified_spdx_syntax() {
        let mut fields = valid_fields();
        set(&mut fields, "feature_columns", "demand,label,price");
        assert!(validate_manifest(&fields).unwrap_err().contains("target_column"));
        set(&mut fields, "feature_columns", "demand,price,temperature");
        set(&mut fields, "source_license_spdx", "CC BY 4.0");
        assert!(validate_manifest(&fields).unwrap_err().contains("verified SPDX"));
    }

    #[test]
    fn receipt_binds_run_source_and_safe_artifact_path() {
        let manifest = valid_fields();
        let receipt_pairs = [
            ("schema_version", "1"),
            ("run_id", "run_001"),
            ("manifest_sha256", "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"),
            ("source_sha256", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            ("output_sha256", "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"),
            ("artifact_path", "artifacts/run_001/results.csv"),
        ];
        let mut receipt = Fields::new();
        for (key, value) in receipt_pairs {
            receipt.insert(key.to_owned(), value.to_owned());
        }
        validate_receipt(&receipt, &manifest).unwrap();
        set(&mut receipt, "run_id", "run_002");
        assert!(validate_receipt(&receipt, &manifest).unwrap_err().contains("run_id"));
        set(&mut receipt, "run_id", "run_001");
        set(&mut receipt, "artifact_path", "../escape.csv");
        assert!(validate_receipt(&receipt, &manifest).unwrap_err().contains("safe relative path"));
    }

    #[test]
    fn rejects_noncanonical_number_format_and_unknown_keys() {
        let mut fields = valid_fields();
        set(&mut fields, "transform_severity", "5e-1");
        assert!(validate_manifest(&fields).unwrap_err().contains("six-decimal"));
        let text = format!("{}unknown=x\n", valid_manifest());
        assert!(parse_fields(&text, MANIFEST_KEYS).unwrap_err().contains("unknown key"));
    }
}
