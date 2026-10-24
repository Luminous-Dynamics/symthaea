// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed checks for the Open Audio Evidence manifest.
//!
//! This crate checks project-specific manifest invariants and the integrity of
//! referenced artifact bytes. It is not a complete JSON Schema validator or a
//! standards-compliance tool.

use chrono::DateTime;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Component, Path, PathBuf};

const EMBEDDED_SCHEMA: &str =
    include_str!("../../../../schemas/audio-evidence-manifest.schema.json");
const EXPECTED_SCHEMA_DRAFT: &str = "https://json-schema.org/draft/2020-12/schema";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactCheck {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
}

/// Check project-specific invariants. This does not replace a full Draft 2020-12 validator.
pub fn validate_manifest(manifest: &Value) -> Result<(), String> {
    let schema: Value = serde_json::from_str(EMBEDDED_SCHEMA)
        .map_err(|error| format!("embedded manifest schema is not valid JSON: {error}"))?;
    if schema.get("$schema").and_then(Value::as_str) != Some(EXPECTED_SCHEMA_DRAFT) {
        return Err("embedded schema draft is not Draft 2020-12".to_owned());
    }
    let embedded_version = schema
        .pointer("/properties/schema_version/const")
        .and_then(Value::as_str)
        .ok_or_else(|| "embedded schema has no schema_version const".to_owned())?;

    let root = object_at(manifest, "manifest")?;
    let allowed = schema
        .get("properties")
        .and_then(Value::as_object)
        .ok_or_else(|| "embedded schema has no root properties object".to_owned())?;
    for key in root.keys() {
        if !allowed.contains_key(key) {
            return Err(format!("manifest: unknown root field {key:?}"));
        }
    }
    require_fields(root, "manifest", &[
        "schema_version", "manifest_id", "data_status", "created_at",
        "product", "build", "protocol", "decision", "limitations",
    ])?;
    if string_at(manifest, "/schema_version")? != embedded_version {
        return Err(format!("manifest schema_version must be {embedded_version}"));
    }

    let manifest_id = string_at(manifest, "/manifest_id")?;
    if manifest_id.len() < 3 || manifest_id.len() > 128
        || !manifest_id
            .chars()
            .all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || "._-".contains(ch))
        || !manifest_id
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit())
    {
        return Err(
            concat!(
                "manifest_id must be 3-128 ASCII uppercase/digit/._- characters ",
                "and start with an uppercase letter or digit"
            )
            .to_owned(),
        );
    }

    let status = string_at(manifest, "/data_status")?;
    if !matches!(
        status,
        "planned" | "captured" | "analyzed" | "validated" | "invalid" | "excluded"
    ) {
        return Err(format!("unknown data_status {status:?}"));
    }
    validate_timestamp(string_at(manifest, "/created_at")?, "/created_at")?;
    if let Some(value) = manifest.get("started_at") {
        let timestamp = value
            .as_str()
            .ok_or_else(|| "started_at must be a string".to_owned())?;
        validate_timestamp(timestamp, "/started_at")?;
    }
    if let Some(value) = manifest.get("completed_at") {
        let timestamp = value
            .as_str()
            .ok_or_else(|| "completed_at must be a string".to_owned())?;
        validate_timestamp(timestamp, "/completed_at")?;
    }

    validate_required_object(
        manifest,
        "/product",
        &["kind", "brand", "model", "hardware_revision"],
    )?;
    validate_required_object(manifest, "/build", &["build_id", "source_revision"])?;
    validate_required_object(
        manifest,
        "/protocol",
        &["protocol_id", "version", "measurement_type"],
    )?;
    for pointer in [
        "/product/kind", "/product/brand", "/product/model", "/product/hardware_revision",
        "/build/build_id", "/build/source_revision", "/protocol/protocol_id",
        "/protocol/version", "/protocol/measurement_type",
    ] {
        validate_nonempty_string(manifest, pointer)?;
    }

    let decision = object_at_pointer(manifest, "/decision")?;
    require_fields(decision, "/decision", &["disposition"])?;
    let disposition = string_at(manifest, "/decision/disposition")?;
    let limitations = manifest
        .get("limitations")
        .and_then(Value::as_array)
        .ok_or_else(|| "limitations must be an array".to_owned())?;
    if limitations.is_empty()
        || limitations
            .iter()
            .any(|item| item.as_str().is_none_or(|s| s.trim().is_empty()))
    {
        return Err("limitations must contain at least one non-empty string".to_owned());
    }

    let captures = manifest.get("captures").and_then(Value::as_array);
    let results = manifest.get("results").and_then(Value::as_array);
    match status {
        "planned" => {
            if captures.is_some_and(|items| !items.is_empty()) {
                return Err("planned manifest must not contain captures".to_owned());
            }
            if results.is_some_and(|items| !items.is_empty()) {
                return Err("planned manifest must not contain results".to_owned());
            }
            if disposition != "not_evaluated" {
                return Err("planned manifest decision must be not_evaluated".to_owned());
            }
        }
        "captured" | "analyzed" => {
            validate_started_run(manifest, captures)?;
            if disposition != "not_evaluated" {
                return Err(format!("{status} manifest cannot claim a gate pass/fail"));
            }
            if status == "analyzed" {
                validate_analysis_and_results(manifest, results)?;
            }
        }
        "validated" => {
            validate_started_run(manifest, captures)?;
            validate_analysis_and_results(manifest, results)?;
            if !matches!(disposition, "pass" | "fail" | "inconclusive") {
                return Err(
                    "validated manifest decision must be pass, fail, or inconclusive".to_owned(),
                );
            }
        }
        "invalid" | "excluded" => {
            validate_started_run(manifest, captures)?;
            if !matches!(disposition, "not_evaluated" | "fail" | "inconclusive" | "excluded") {
                return Err(format!(
                    "invalid/excluded run has incompatible decision {disposition:?}"
                ));
            }
        }
        _ => unreachable!("data_status was checked above"),
    }
    validate_artifact_shapes(manifest)?;
    Ok(())
}

/// Verify every referenced artifact is inside the bundle and has the declared
/// SHA-256 and optional byte length. Symlink escapes are rejected.
pub fn verify_artifacts(
    manifest: &Value,
    bundle_root: &Path,
) -> Result<Vec<ArtifactCheck>, String> {
    validate_manifest(manifest)?;
    let root = fs::canonicalize(bundle_root)
        .map_err(|error| format!("cannot resolve bundle root {}: {error}", bundle_root.display()))?;
    let mut objects = Vec::new();
    collect_artifacts(manifest, &mut objects);
    let mut checked = Vec::with_capacity(objects.len());
    for artifact in objects {
        let relative = string_member(artifact, "path", "artifact")?;
        let expected = string_member(artifact, "sha256", relative)?;
        let full_path = safe_artifact_path(&root, relative)?;
        let bytes = fs::read(&full_path)
            .map_err(|error| format!("cannot read artifact {relative:?}: {error}"))?;
        let actual = hex::encode(Sha256::digest(&bytes));
        if !actual.eq_ignore_ascii_case(expected) {
            return Err(format!(
                "SHA-256 mismatch for {relative:?}: expected {expected}, got {actual}"
            ));
        }
        let actual_size = u64::try_from(bytes.len())
            .map_err(|_| format!("artifact {relative:?} is too large to represent"))?;
        if let Some(size_value) = artifact.get("size_bytes") {
            let expected_size = size_value.as_u64()
                .ok_or_else(|| {
                    format!("artifact {relative:?} size_bytes must be an unsigned integer")
                })?;
            if expected_size != actual_size {
                return Err(format!(
                    "size mismatch for {relative:?}: expected {expected_size}, got {actual_size}"
                ));
            }
        }
        checked.push(ArtifactCheck {
            path: relative.to_owned(),
            size_bytes: actual_size,
            sha256: actual,
        });
    }
    Ok(checked)
}

/// Returns the number of successfully verified artifact references.
pub fn check_manifest_bundle(manifest: &Value, bundle_root: &Path) -> Result<usize, String> {
    verify_artifacts(manifest, bundle_root).map(|items| items.len())
}

fn validate_started_run(manifest: &Value, captures: Option<&Vec<Value>>) -> Result<(), String> {
    let started = string_at(manifest, "/started_at")
        .map_err(|_| "started run must provide started_at".to_owned())?;
    validate_timestamp(started, "/started_at")?;
    let fixture = object_at_pointer(manifest, "/fixture")
        .map_err(|_| "started run must provide a fixture object".to_owned())?;
    require_fields(
        fixture,
        "/fixture",
        &["fixture_id", "fixture_type", "model_or_design", "revision"],
    )?;
    let instruments = manifest
        .get("instrumentation")
        .and_then(Value::as_array)
        .ok_or_else(|| "started run must provide instrumentation array".to_owned())?;
    if instruments.is_empty() {
        return Err("started run must identify at least one instrument".to_owned());
    }
    for (index, instrument) in instruments.iter().enumerate() {
        let label = format!("/instrumentation/{index}");
        let object = instrument
            .as_object()
            .ok_or_else(|| format!("{label} must be an object"))?;
        require_fields(
            object,
            &label,
            &["instrument_id", "kind", "manufacturer", "model", "calibration_state"],
        )?;
    }
    if !manifest.get("environment").is_some_and(Value::is_object) {
        return Err(
            "started run must provide an environment object (unknowns may be omitted)".to_owned(),
        );
    }
    let captures = captures.ok_or_else(|| "started run must provide captures array".to_owned())?;
    if captures.is_empty() {
        return Err("started run must provide at least one capture".to_owned());
    }
    if !captures.iter().any(|capture| {
        capture.get("capture_role").and_then(Value::as_str) == Some("raw")
    }) {
        return Err("started run must contain at least one raw capture".to_owned());
    }
    for (index, capture) in captures.iter().enumerate() {
        let label = format!("/captures/{index}");
        let object = capture
            .as_object()
            .ok_or_else(|| format!("{label} must be an object"))?;
        require_fields(
            object,
            &label,
            &["capture_id", "capture_role", "artifact", "channel_or_angle"],
        )?;
        let role = string_member(object, "capture_role", &label)?;
        if !matches!(
            role,
            "raw" | "reference_control" | "calibration_check" | "processed" | "plot" | "report"
                | "photo" | "other"
        ) {
            return Err(format!("{label}/capture_role has unknown value {role:?}"));
        }
        validate_nonempty_member(object, "capture_id", &label)?;
        validate_nonempty_member(object, "channel_or_angle", &label)?;
    }
    Ok(())
}

fn validate_analysis_and_results(
    manifest: &Value,
    results: Option<&Vec<Value>>,
) -> Result<(), String> {
    let analysis = object_at_pointer(manifest, "/analysis")
        .map_err(|_| "analyzed/validated manifest must provide analysis object".to_owned())?;
    require_fields(analysis, "/analysis", &["tool_id", "tool_version", "source_revision"])?;
    for key in ["tool_id", "tool_version", "source_revision"] {
        validate_nonempty_member(analysis, key, "/analysis")?;
    }
    let results = results
        .ok_or_else(|| "analyzed/validated manifest must provide results array".to_owned())?;
    if results.is_empty() {
        return Err("analyzed/validated manifest must contain at least one result".to_owned());
    }
    for (index, result) in results.iter().enumerate() {
        let label = format!("/results/{index}");
        let object = result
            .as_object()
            .ok_or_else(|| format!("{label} must be an object"))?;
        require_fields(
            object,
            &label,
            &["metric_id", "value", "unit", "statistic", "frequency_or_condition"],
        )?;
        for key in ["metric_id", "unit", "frequency_or_condition"] {
            validate_nonempty_member(object, key, &label)?;
        }
        if !object.get("value").is_some_and(Value::is_number) {
            return Err(format!("{label}/value must be numeric"));
        }
    }
    Ok(())
}

fn validate_artifact_shapes(manifest: &Value) -> Result<(), String> {
    validate_artifact_references(manifest, "manifest")
}

fn validate_artifact_references(value: &Value, label: &str) -> Result<(), String> {
    match value {
        Value::Object(object) => {
            for (key, child) in object {
                let child_label = format!("{label}/{key}");
                if key == "artifact" {
                    validate_artifact_object(child, &child_label)?;
                }
                if key.ends_with("_artifacts") {
                    let items = child.as_array().ok_or_else(|| {
                        format!("{child_label} must be an array of artifact records")
                    })?;
                    for (index, item) in items.iter().enumerate() {
                        validate_artifact_object(item, &format!("{child_label}/{index}"))?;
                    }
                }
                validate_artifact_references(child, &child_label)?;
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                validate_artifact_references(child, &format!("{label}/{index}"))?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_artifact_object(value: &Value, label: &str) -> Result<(), String> {
    let artifact = value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    require_fields(artifact, label, &["path", "sha256", "media_type"])?;
    let path = string_member(artifact, "path", label)?;
    validate_relative_path(path).map_err(|error| format!("{label}: {error}"))?;
    let digest = string_member(artifact, "sha256", path)?;
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("artifact {path:?} has malformed SHA-256 digest"));
    }
    if string_member(artifact, "media_type", path)?.trim().is_empty() {
        return Err(format!("artifact {path:?} media_type must be non-empty"));
    }
    if let Some(size) = artifact.get("size_bytes") {
        if size.as_u64().is_none() {
            return Err(format!("artifact {path:?} size_bytes must be an unsigned integer"));
        }
    }
    Ok(())
}

fn collect_artifacts<'a>(value: &'a Value, output: &mut Vec<&'a Map<String, Value>>) {
    match value {
        Value::Object(object) => {
            if object.contains_key("path")
                && object.contains_key("sha256")
                && object.contains_key("media_type")
            {
                output.push(object);
            }
            for child in object.values() {
                collect_artifacts(child, output);
            }
        }
        Value::Array(items) => {
            for child in items {
                collect_artifacts(child, output);
            }
        }
        _ => {}
    }
}

fn validate_relative_path(raw: &str) -> Result<PathBuf, String> {
    if raw.is_empty() || raw.starts_with('/') || raw.ends_with('/')
        || !raw.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"._-/".contains(&byte))
    {
        return Err(format!("artifact path {raw:?} must be a non-empty repository-relative path"));
    }
    let path = Path::new(raw);
    for component in path.components() {
        match component {
            Component::Normal(_) => {}
            Component::CurDir
            | Component::ParentDir
            | Component::RootDir
            | Component::Prefix(_) => {
                return Err(format!("artifact path {raw:?} contains an unsafe path component"));
            }
        }
    }
    Ok(path.to_path_buf())
}

fn safe_artifact_path(root: &Path, relative: &str) -> Result<PathBuf, String> {
    let full_path = fs::canonicalize(root.join(validate_relative_path(relative)?))
        .map_err(|error| format!("cannot resolve artifact {relative:?}: {error}"))?;
    if !full_path.starts_with(root) {
        return Err(format!("artifact {relative:?} resolves outside the bundle root"));
    }
    if !full_path.is_file() {
        return Err(format!("artifact {relative:?} is not a regular file"));
    }
    Ok(full_path)
}

fn validate_timestamp(raw: &str, label: &str) -> Result<(), String> {
    DateTime::parse_from_rfc3339(raw).map(|_| ())
        .map_err(|_| format!("{label} must be an RFC 3339 date-time"))
}

fn validate_required_object(value: &Value, pointer: &str, fields: &[&str]) -> Result<(), String> {
    require_fields(object_at_pointer(value, pointer)?, pointer, fields)
}

fn validate_nonempty_string(value: &Value, pointer: &str) -> Result<(), String> {
    let text = string_at(value, pointer)?;
    if text.trim().is_empty() {
        return Err(format!("{pointer} must be a non-empty string"));
    }
    Ok(())
}

fn validate_nonempty_member(
    object: &Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<(), String> {
    let value = object
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label}/{key} must be a string"))?;
    if value.trim().is_empty() {
        return Err(format!("{label}/{key} must be a non-empty string"));
    }
    Ok(())
}

fn object_at<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))
}

fn object_at_pointer<'a>(
    value: &'a Value,
    pointer: &str,
) -> Result<&'a Map<String, Value>, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{pointer} must be an object"))
}

fn string_at<'a>(value: &'a Value, pointer: &str) -> Result<&'a str, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{pointer} must be a string"))
}

fn string_member<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<&'a str, String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label}/{key} must be a string"))
}

fn require_fields(object: &Map<String, Value>, label: &str, fields: &[&str]) -> Result<(), String> {
    for field in fields {
        if !object.contains_key(*field) {
            return Err(format!("{label} is missing required field {field:?}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    const PLANNED_EXAMPLE: &str =
        include_str!("../../../../schemas/examples/h1-fit-repeatability-plan.json");
    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    fn planned() -> Value {
        serde_json::from_str(PLANNED_EXAMPLE).expect("example JSON must parse")
    }

    fn started_manifest(raw_bytes: &[u8]) -> Value {
        let digest = hex::encode(Sha256::digest(raw_bytes));
        json!({
            "schema_version": "1.0.0",
            "manifest_id": "TEST-CAPTURE-001",
            "data_status": "captured",
            "created_at": "2026-10-17T16:35:00Z",
            "started_at": "2026-10-17T16:36:00Z",
            "product": {
                "kind": "headphone",
                "brand": "Example",
                "model": "H1",
                "hardware_revision": "r1",
                "condition": "prototype"
            },
            "build": {"build_id": "test-build", "source_revision": "deadbeef"},
            "protocol": {
                "protocol_id": "test",
                "version": "1.0",
                "measurement_type": "headphone_response"
            },
            "fixture": {
                "fixture_id": "fixture-1",
                "fixture_type": "headphone_coupler",
                "model_or_design": "test coupler",
                "revision": "r1"
            },
            "instrumentation": [{
                "instrument_id": "mic-1",
                "kind": "measurement_microphone",
                "manufacturer": "Example",
                "model": "M1",
                "calibration_state": "unknown"
            }],
            "environment": {},
            "captures": [{
                "capture_id": "raw-L-1",
                "capture_role": "raw",
                "artifact": {
                    "path": "captures/raw.bin",
                    "sha256": digest,
                    "media_type": "application/octet-stream",
                    "size_bytes": raw_bytes.len()
                },
                "channel_or_angle": "L",
                "repeat_index": 1
            }],
            "decision": {"disposition": "not_evaluated"},
            "limitations": ["Test fixture only; not a real acoustic capture."]
        })
    }

    fn temp_dir() -> PathBuf {
        let serial = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "symthaea-audio-evidence-{}-{serial}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("temporary directory");
        path
    }

    #[test]
    fn accepts_planned_example_without_faking_measurements() {
        let manifest = planned();
        assert_eq!(manifest["data_status"], "planned");
        assert!(manifest.get("captures").is_none());
        assert!(manifest.get("results").is_none());
        assert!(validate_manifest(&manifest).is_ok());
    }

    #[test]
    fn planned_manifest_cannot_claim_pass() {
        let mut manifest = planned();
        manifest["decision"]["disposition"] = json!("pass");
        assert!(
            validate_manifest(&manifest)
                .expect_err("planned PASS must fail closed")
                .contains("must be not_evaluated")
        );
    }

    #[test]
    fn planned_manifest_cannot_contain_measurement_captures() {
        let mut manifest = planned();
        manifest["captures"] = json!([{"capture_role": "raw"}]);
        assert!(
            validate_manifest(&manifest)
                .expect_err("planned data must not have captures")
                .contains("must not contain captures")
        );
    }

    #[test]
    fn started_run_requires_fixture_instruments_environment_and_raw_capture() {
        let mut manifest = started_manifest(b"test bytes");
        manifest.as_object_mut().expect("root").remove("fixture");
        assert!(
            validate_manifest(&manifest)
                .expect_err("fixture is required")
                .contains("fixture object")
        );
    }

    #[test]
    fn started_run_cannot_claim_pass_before_analysis() {
        let mut manifest = started_manifest(b"test bytes");
        manifest["decision"]["disposition"] = json!("pass");
        assert!(
            validate_manifest(&manifest)
                .expect_err("captured data is not validated")
                .contains("cannot claim a gate pass/fail")
        );
    }

    #[test]
    fn unsafe_artifact_paths_are_rejected() {
        let mut manifest = started_manifest(b"test bytes");
        manifest["captures"][0]["artifact"]["path"] = json!("../escape.wav");
        assert!(
            validate_manifest(&manifest)
                .expect_err("parent traversal must fail")
                .contains("unsafe path component")
        );
    }

    #[test]
    fn rejects_capture_artifact_that_is_not_an_object() {
        let mut manifest = started_manifest(b"test bytes");
        manifest["captures"][0]["artifact"] = json!("not an artifact object");
        assert!(validate_manifest(&manifest)
            .expect_err("capture artifact must be an object")
            .contains("must be an object"));
    }

    #[test]
    fn rejects_artifact_missing_digest() {
        let mut manifest = started_manifest(b"test bytes");
        manifest["captures"][0]["artifact"]
            .as_object_mut()
            .expect("artifact object")
            .remove("sha256");
        assert!(validate_manifest(&manifest)
            .expect_err("digest is mandatory")
            .contains("missing required field"));
    }

    #[test]
    fn verifies_actual_artifact_bytes_and_detects_mutation() {
        let root = temp_dir();
        let captures = root.join("captures");
        fs::create_dir_all(&captures).expect("capture dir");
        let path = captures.join("raw.bin");
        fs::write(&path, b"test bytes").expect("write capture");
        let manifest = started_manifest(b"test bytes");
        assert_eq!(check_manifest_bundle(&manifest, &root).expect("matching bytes"), 1);
        fs::write(path, b"mutated bytes").expect("mutate capture");
        assert!(
            verify_artifacts(&manifest, &root)
                .expect_err("mutation must fail")
                .contains("SHA-256 mismatch")
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_escape_from_bundle() {
        let root = temp_dir();
        let outside = temp_dir();
        let captures = root.join("captures");
        fs::create_dir_all(&captures).expect("capture dir");
        let outside_file = outside.join("outside.bin");
        fs::write(&outside_file, b"test bytes").expect("outside file");
        std::os::unix::fs::symlink(&outside_file, captures.join("raw.bin"))
            .expect("create symlink");
        let manifest = started_manifest(b"test bytes");
        assert!(
            verify_artifacts(&manifest, &root)
                .expect_err("symlink escape must fail")
                .contains("resolves outside the bundle root")
        );
        fs::remove_dir_all(root).expect("cleanup root");
        fs::remove_dir_all(outside).expect("cleanup outside");
    }
}
