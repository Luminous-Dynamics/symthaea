//! Independent verifier for a finite, explicitly declared Cartesian universe of
//! structural connectome hypotheses.
//!
//! This tool hashes referenced local artifacts, regenerates every admissible world
//! from finite axes, and checks exact equality with matched before/after evaluations.
//! It does not prove that the axes are biologically exhaustive or that the freeze
//! attestation is authentic. Those claims require an operator-pinned trust policy.

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Component, Path, PathBuf};

const PROTOCOL: &str = "CONNECTOME-STRUCTURAL-WORLD-UNIVERSE-V1";
const RECEIPT_PROTOCOL: &str = "CONNECTOME-STRUCTURAL-WORLD-VERIFICATION-RECEIPT-V1";
const POLICY: &str = "cartesian_product_all_combinations_v1";
const MAX_WORLDS: usize = 100_000;
const MAX_AXES: usize = 64;
const TOLERANCE: f64 = 1.0e-12;

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// serde_json's default map representation is key ordered (preserve_order is not enabled).
/// This is the pinned SERDE_JSON_ORDERED_CANONICAL_V1 wire encoding, not a claim of RFC 8785.
fn canonical_bytes(value: &Value) -> Result<Vec<u8>, String> {
    serde_json::to_vec(value)
        .map_err(|error| format!("canonical JSON serialization failed: {error}"))
}

fn canonical_sha256(value: &Value) -> Result<String, String> {
    Ok(sha256(&canonical_bytes(value)?))
}

fn without_field(value: &Value, field: &str) -> Value {
    let mut clone = value.clone();
    if let Some(object) = clone.as_object_mut() {
        object.remove(field);
    }
    clone
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_sha256(value: &str) -> bool {
    is_lower_hex(value, 64)
}

fn nonempty_str<'a>(value: &'a Value, key: &str, errors: &mut Vec<String>) -> Option<&'a str> {
    match value.get(key).and_then(Value::as_str) {
        Some(text) if !text.trim().is_empty() => Some(text),
        _ => {
            errors.push(format!("{key}: missing or empty string"));
            None
        }
    }
}

fn finite_number(value: &Value, key: &str, errors: &mut Vec<String>) -> Option<f64> {
    match value.get(key).and_then(Value::as_f64) {
        Some(number) if number.is_finite() => Some(number),
        _ => {
            errors.push(format!("{key}: missing or non-finite number"));
            None
        }
    }
}

fn validate_interval(value: &Value, key: &str, errors: &mut Vec<String>) -> Option<(f64, f64)> {
    let lower = finite_number(value, "lower", errors);
    let upper = finite_number(value, "upper", errors);
    match (lower, upper) {
        (Some(lower), Some(upper)) if lower <= upper => Some((lower, upper)),
        (Some(_), Some(_)) => {
            errors.push(format!("{key}: lower bound exceeds upper bound"));
            None
        }
        _ => None,
    }
}

fn contains(interval: (f64, f64), value: f64) -> bool {
    value >= interval.0 - TOLERANCE && value <= interval.1 + TOLERANCE
}

fn safe_artifact_path(root: &Path, relative: &str) -> Result<PathBuf, String> {
    let candidate = Path::new(relative);
    if candidate.is_absolute()
        || candidate.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(format!(
            "artifact path must be relative and cannot traverse directories: {relative}"
        ));
    }
    let root = root
        .canonicalize()
        .map_err(|error| format!("artifact root cannot be canonicalized: {error}"))?;
    let resolved = root
        .join(candidate)
        .canonicalize()
        .map_err(|error| format!("artifact {relative} cannot be read: {error}"))?;
    if !resolved.starts_with(&root) {
        return Err(format!(
            "artifact path escapes manifest directory: {relative}"
        ));
    }
    Ok(resolved)
}

fn file_sha256(path: &Path) -> Result<String, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    Ok(sha256(&bytes))
}

fn world_id(assignment: &Map<String, Value>) -> Result<String, String> {
    let assignment = Value::Object(assignment.clone());
    let mut bytes = b"CONNECTOME-STRUCTURAL-WORLD-V1\n".to_vec();
    bytes.extend(canonical_bytes(&assignment)?);
    Ok(format!("world-{}", sha256(&bytes)))
}

#[derive(Clone, Debug)]
struct Alternative {
    id: String,
    evidence_artifact_id: String,
    evidence_sha256: String,
}

#[derive(Clone, Debug)]
struct Axis {
    id: String,
    alternatives: Vec<Alternative>,
}

/// Generate all worlds in lexical axis/alternative order. The only admissibility
/// rule in V1 is the full Cartesian product; there are no opaque predicates/filters.
fn expand_world_ids(
    axes: &[Axis],
    index: usize,
    assignment: &mut Map<String, Value>,
    output: &mut Vec<String>,
) -> Result<(), String> {
    if index == axes.len() {
        output.push(world_id(assignment)?);
        return Ok(());
    }
    let axis = &axes[index];
    for alternative in &axis.alternatives {
        assignment.insert(axis.id.clone(), Value::String(alternative.id.clone()));
        expand_world_ids(axes, index + 1, assignment, output)?;
    }
    assignment.remove(&axis.id);
    Ok(())
}

fn read_axes(manifest: &Value, errors: &mut Vec<String>) -> Vec<Axis> {
    let Some(items) = manifest.get("axes").and_then(Value::as_array) else {
        errors.push("axes: expected a non-empty array".to_owned());
        return Vec::new();
    };
    if items.is_empty() {
        errors.push("axes: expected a non-empty array".to_owned());
        return Vec::new();
    }
    if items.len() > MAX_AXES {
        errors.push(format!(
            "axes: {} axes exceed the V1 safety cap of {MAX_AXES}",
            items.len()
        ));
        return Vec::new();
    }

    let mut axes = Vec::new();
    let mut seen_axes = BTreeSet::new();
    for (axis_index, item) in items.iter().enumerate() {
        let id = match item.get("axis_id").and_then(Value::as_str) {
            Some(id) if !id.trim().is_empty() => id.to_owned(),
            _ => {
                errors.push(format!("axes[{axis_index}].axis_id: missing or empty"));
                continue;
            }
        };
        if !seen_axes.insert(id.clone()) {
            errors.push(format!("axes: duplicate axis_id {id}"));
        }
        let Some(alternatives_json) = item.get("alternatives").and_then(Value::as_array) else {
            errors.push(format!("axes[{axis_index}].alternatives: expected array"));
            continue;
        };
        if alternatives_json.is_empty() {
            errors.push(format!(
                "axes[{axis_index}].alternatives: must not be empty"
            ));
        }
        let mut alternatives = Vec::new();
        let mut seen_alternatives = BTreeSet::new();
        for (alternative_index, alternative) in alternatives_json.iter().enumerate() {
            let alternative_id = match alternative.get("alternative_id").and_then(Value::as_str) {
                Some(value) if !value.trim().is_empty() => value.to_owned(),
                _ => {
                    errors.push(format!(
                        "axes[{axis_index}].alternatives[{alternative_index}].alternative_id: missing"
                    ));
                    continue;
                }
            };
            if !seen_alternatives.insert(alternative_id.clone()) {
                errors.push(format!(
                    "axes[{axis_index}]: duplicate alternative_id {alternative_id}"
                ));
            }
            let artifact_id = match alternative
                .get("evidence_artifact_id")
                .and_then(Value::as_str)
            {
                Some(value) if !value.trim().is_empty() => value.to_owned(),
                _ => {
                    errors.push(format!(
                        "axes[{axis_index}].alternatives[{alternative_index}].evidence_artifact_id: missing"
                    ));
                    continue;
                }
            };
            let evidence_sha256 = match alternative.get("evidence_sha256").and_then(Value::as_str) {
                Some(digest) if is_sha256(digest) => digest.to_owned(),
                _ => {
                    errors.push(format!(
                        "axes[{axis_index}].alternatives[{alternative_index}].evidence_sha256: invalid SHA-256"
                    ));
                    continue;
                }
            };
            alternatives.push(Alternative {
                id: alternative_id,
                evidence_artifact_id: artifact_id,
                evidence_sha256,
            });
        }
        alternatives.sort_by(|left, right| left.id.cmp(&right.id));
        axes.push(Axis { id, alternatives });
    }
    axes.sort_by(|left, right| left.id.cmp(&right.id));
    axes
}

fn validate_artifacts(
    manifest: &Value,
    root: &Path,
    errors: &mut Vec<String>,
) -> BTreeMap<String, String> {
    let mut digests = BTreeMap::new();
    let mut paths = BTreeSet::new();
    let Some(items) = manifest.get("artifact_files").and_then(Value::as_array) else {
        errors.push("artifact_files: expected an array".to_owned());
        return digests;
    };
    for (index, item) in items.iter().enumerate() {
        let id = match item.get("artifact_id").and_then(Value::as_str) {
            Some(value) if !value.trim().is_empty() => value.to_owned(),
            _ => {
                errors.push(format!("artifact_files[{index}].artifact_id: missing"));
                continue;
            }
        };
        let relative = match item.get("relative_path").and_then(Value::as_str) {
            Some(value) if !value.trim().is_empty() => value,
            _ => {
                errors.push(format!("artifact_files[{index}].relative_path: missing"));
                continue;
            }
        };
        let declared = match item.get("sha256").and_then(Value::as_str) {
            Some(value) if is_sha256(value) => value,
            _ => {
                errors.push(format!("artifact_files[{index}].sha256: invalid SHA-256"));
                continue;
            }
        };
        if digests.contains_key(&id) {
            errors.push(format!("artifact_files: duplicate artifact_id {id}"));
            continue;
        }
        if !paths.insert(relative.to_owned()) {
            errors.push(format!(
                "artifact_files: duplicate relative_path {relative}"
            ));
            continue;
        }
        match safe_artifact_path(root, relative).and_then(|path| file_sha256(&path)) {
            Ok(actual) if actual == declared => {
                digests.insert(id, actual);
            }
            Ok(actual) => errors.push(format!(
                "artifact {id}: SHA-256 mismatch; declared {declared}, computed {actual}"
            )),
            Err(error) => errors.push(format!("artifact {id}: {error}")),
        }
    }

    for required in [
        "prequery_freeze_receipt",
        "source_universe_snapshot",
        "base_graph_snapshot",
        "hypothetical_result_graph_snapshot",
        "candidate_patch",
    ] {
        if !digests.contains_key(required) {
            errors.push(format!(
                "artifact_files: missing required artifact_id {required}"
            ));
        }
    }
    digests
}

fn verify_manifest(manifest_path: &Path, verifier_sha256: &str) -> Result<Value, Vec<String>> {
    let mut errors = Vec::new();
    if !is_sha256(verifier_sha256) {
        return Err(vec![
            "verifier_sha256: verifier binary digest is invalid".to_owned(),
        ]);
    }
    let bytes = fs::read(manifest_path).map_err(|error| {
        vec![format!(
            "manifest cannot be read at {}: {error}",
            manifest_path.display()
        )]
    })?;
    let manifest: Value = serde_json::from_slice(&bytes)
        .map_err(|error| vec![format!("manifest JSON is invalid: {error}")])?;
    let root = manifest_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));

    if manifest.get("protocol").and_then(Value::as_str) != Some(PROTOCOL) {
        errors.push(format!("protocol: expected {PROTOCOL}"));
    }
    if manifest.get("schema_version").and_then(Value::as_str) != Some("0.1") {
        errors.push("schema_version: expected 0.1".to_owned());
    }

    let declared_manifest_sha = manifest.get("manifest_sha256").and_then(Value::as_str);
    let computed_manifest_sha = canonical_sha256(&without_field(&manifest, "manifest_sha256"))
        .unwrap_or_else(|error| {
            errors.push(error);
            String::new()
        });
    match declared_manifest_sha {
        Some(declared) if declared == computed_manifest_sha && is_sha256(declared) => {}
        Some(declared) => errors.push(format!(
            "manifest_sha256: declared {declared}, computed {computed_manifest_sha}"
        )),
        None => errors.push("manifest_sha256: missing".to_owned()),
    }

    let universe_id = nonempty_str(&manifest, "universe_id", &mut errors).unwrap_or_default();
    if manifest.get("admissibility_policy").and_then(Value::as_str) != Some(POLICY) {
        errors.push(format!("admissibility_policy: V1 requires {POLICY}"));
    }

    let context = manifest.get("context").cloned().unwrap_or(Value::Null);
    let mut binding = Map::new();
    for field in [
        "candidate_id",
        "operation_id",
        "task_route",
        "query_id",
        "scenario_id",
        "metric_id",
        "unit_id",
        "source_volume_id",
    ] {
        if let Some(value) = nonempty_str(&context, field, &mut errors) {
            binding.insert(field.to_owned(), Value::String(value.to_owned()));
        }
    }
    let sample_hash = nonempty_str(&context, "sample_hash", &mut errors).unwrap_or_default();
    if !is_lower_hex(sample_hash, 32) {
        errors.push("context.sample_hash: expected 32 lowercase hexadecimal characters".to_owned());
    }
    binding.insert(
        "sample_hash".to_owned(),
        Value::String(sample_hash.to_owned()),
    );
    for field in [
        "query_definition_sha256",
        "base_graph_snapshot_sha256",
        "hypothetical_result_graph_snapshot_sha256",
        "candidate_patch_sha256",
    ] {
        if let Some(value) = nonempty_str(&context, field, &mut errors) {
            if !is_sha256(value) {
                errors.push(format!("context.{field}: expected lowercase SHA-256"));
            }
            binding.insert(field.to_owned(), Value::String(value.to_owned()));
        }
    }
    let task_route = context.get("task_route").and_then(Value::as_str);
    if !matches!(
        task_route,
        Some("false_split_correction" | "false_merge_identification" | "split_mask_generation")
    ) {
        errors.push("context.task_route: unsupported task route".to_owned());
    }
    let decision_threshold = finite_number(&context, "decision_threshold", &mut errors);
    let materiality_threshold = finite_number(&context, "materiality_threshold", &mut errors);
    if materiality_threshold.is_some_and(|value| value <= 0.0) {
        errors.push("context.materiality_threshold: must be positive".to_owned());
    }
    if let Some(value) = decision_threshold {
        binding.insert("decision_threshold".to_owned(), json!(value));
    }
    if let Some(value) = materiality_threshold {
        binding.insert("materiality_threshold".to_owned(), json!(value));
    }

    let baseline_interval = match context.get("baseline_interval") {
        Some(value) => validate_interval(value, "context.baseline_interval", &mut errors),
        None => {
            errors.push("context.baseline_interval: missing".to_owned());
            None
        }
    };
    let after_interval = match context.get("after_edit_interval") {
        Some(value) => validate_interval(value, "context.after_edit_interval", &mut errors),
        None => {
            errors.push("context.after_edit_interval: missing".to_owned());
            None
        }
    };
    let paired_effect_interval = match context.get("paired_effect_interval") {
        Some(value) => validate_interval(value, "context.paired_effect_interval", &mut errors),
        None => {
            errors.push("context.paired_effect_interval: missing".to_owned());
            None
        }
    };
    if let (Some(baseline), Some(after), Some(effect)) =
        (baseline_interval, after_interval, paired_effect_interval)
    {
        let rectangular = (after.0 - baseline.1, after.1 - baseline.0);
        if effect.0 < rectangular.0 - TOLERANCE || effect.1 > rectangular.1 + TOLERANCE {
            errors.push(
                "context.paired_effect_interval: exceeds the marginal rectangular outer bound"
                    .to_owned(),
            );
        }
    }

    let freeze = manifest.get("freeze").cloned().unwrap_or(Value::Null);
    if freeze
        .get("universe_frozen_before_query_scoring")
        .and_then(Value::as_bool)
        != Some(true)
    {
        errors.push("freeze.universe_frozen_before_query_scoring: must be true".to_owned());
    }
    if freeze
        .get("query_scoring_opened_before_freeze")
        .and_then(Value::as_bool)
        != Some(false)
    {
        errors.push("freeze.query_scoring_opened_before_freeze: must be false".to_owned());
    }
    if freeze
        .get("unresolved_frontier_count")
        .and_then(Value::as_u64)
        != Some(0)
    {
        errors.push("freeze.unresolved_frontier_count: must be exactly zero".to_owned());
    }
    let freeze_receipt_digest = freeze
        .get("prequery_freeze_receipt_sha256")
        .and_then(Value::as_str);
    if !freeze_receipt_digest.is_some_and(is_sha256) {
        errors.push("freeze.prequery_freeze_receipt_sha256: invalid SHA-256".to_owned());
    }

    let artifact_digests = validate_artifacts(&manifest, root, &mut errors);
    if let Some(receipt_sha) = freeze_receipt_digest
        && artifact_digests
            .get("prequery_freeze_receipt")
            .map(String::as_str)
            != Some(receipt_sha)
    {
        errors.push(
            "freeze.prequery_freeze_receipt_sha256: does not match verified artifact".to_owned(),
        );
    }
    for (context_field, artifact_id) in [
        ("base_graph_snapshot_sha256", "base_graph_snapshot"),
        (
            "hypothetical_result_graph_snapshot_sha256",
            "hypothetical_result_graph_snapshot",
        ),
        ("candidate_patch_sha256", "candidate_patch"),
    ] {
        if let Some(value) = context.get(context_field).and_then(Value::as_str)
            && artifact_digests.get(artifact_id).map(String::as_str) != Some(value)
        {
            errors.push(format!(
                "context.{context_field}: does not match verified artifact"
            ));
        }
    }

    let axes = read_axes(&manifest, &mut errors);
    for axis in &axes {
        for alternative in &axis.alternatives {
            match artifact_digests.get(&alternative.evidence_artifact_id) {
                Some(actual) if actual == &alternative.evidence_sha256 => {}
                Some(_) => errors.push(format!(
                    "axis {}/{}: evidence digest mismatch for artifact {}",
                    axis.id, alternative.id, alternative.evidence_artifact_id
                )),
                None => errors.push(format!(
                    "axis {}/{}: evidence artifact {} was not independently verified",
                    axis.id, alternative.id, alternative.evidence_artifact_id
                )),
            }
        }
    }

    let product_count = axes.iter().try_fold(1usize, |total, axis| {
        total.checked_mul(axis.alternatives.len())
    });
    let mut expected_world_ids = Vec::new();
    match product_count {
        Some(count) if count > 0 && count <= MAX_WORLDS && axes.len() == manifest.get("axes").and_then(Value::as_array).map_or(0, |values| values.len()) => {
            let mut assignment = Map::new();
            if let Err(error) = expand_world_ids(&axes, 0, &mut assignment, &mut expected_world_ids) {
                errors.push(error);
            }
            expected_world_ids.sort();
        }
        Some(count) if count > MAX_WORLDS => errors.push(format!(
            "axes: Cartesian universe contains {count} worlds, exceeding the V1 safety cap {MAX_WORLDS}"
        )),
        _ => errors.push("axes: invalid or overflowing Cartesian universe size".to_owned()),
    }

    let mut evaluated_ids = BTreeSet::new();
    let mut evaluated_assignments = BTreeSet::new();
    let Some(paired_worlds) = manifest.get("paired_worlds").and_then(Value::as_array) else {
        errors.push("paired_worlds: expected an array".to_owned());
        return Err(errors);
    };
    if paired_worlds.is_empty() {
        errors.push("paired_worlds: must not be empty".to_owned());
    }
    let axis_ids: BTreeSet<String> = axes.iter().map(|axis| axis.id.clone()).collect();
    let axis_values: BTreeMap<String, BTreeSet<String>> = axes
        .iter()
        .map(|axis| {
            (
                axis.id.clone(),
                axis.alternatives
                    .iter()
                    .map(|alternative| alternative.id.clone())
                    .collect(),
            )
        })
        .collect();
    for (index, world) in paired_worlds.iter().enumerate() {
        let Some(assignment_json) = world.get("assignment").and_then(Value::as_object) else {
            errors.push(format!(
                "paired_worlds[{index}].assignment: expected object"
            ));
            continue;
        };
        let assignment_keys: BTreeSet<String> = assignment_json.keys().cloned().collect();
        if assignment_keys != axis_ids {
            errors.push(format!(
                "paired_worlds[{index}].assignment: keys do not exactly match axes"
            ));
            continue;
        }
        let mut valid_assignment = Map::new();
        let mut assignment_valid = true;
        for axis in &axes {
            match assignment_json.get(&axis.id).and_then(Value::as_str) {
                Some(value)
                    if axis_values
                        .get(&axis.id)
                        .is_some_and(|values| values.contains(value)) =>
                {
                    valid_assignment.insert(axis.id.clone(), Value::String(value.to_owned()));
                }
                _ => {
                    errors.push(format!(
                        "paired_worlds[{index}].assignment.{}: unknown alternative",
                        axis.id
                    ));
                    assignment_valid = false;
                }
            }
        }
        if !assignment_valid {
            continue;
        }
        let assignment_digest =
            canonical_sha256(&Value::Object(valid_assignment.clone())).unwrap_or_default();
        if !evaluated_assignments.insert(assignment_digest) {
            errors.push(format!("paired_worlds[{index}]: duplicate assignment"));
        }
        let computed_id = world_id(&valid_assignment).unwrap_or_default();
        let supplied_id = world
            .get("world_id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if supplied_id != computed_id {
            errors.push(format!(
                "paired_worlds[{index}].world_id: expected {computed_id}, found {supplied_id}"
            ));
        }
        if !evaluated_ids.insert(supplied_id.to_owned()) {
            errors.push(format!(
                "paired_worlds[{index}].world_id: duplicate world ID"
            ));
        }

        if let (Some(baseline_interval), Some(after_interval), Some(effect_interval)) =
            (baseline_interval, after_interval, paired_effect_interval)
        {
            let baseline = finite_number(world, "baseline_value", &mut errors);
            let after = finite_number(world, "after_edit_value", &mut errors);
            if let (Some(baseline), Some(after)) = (baseline, after) {
                if !contains(baseline_interval, baseline) {
                    errors.push(format!(
                        "paired_worlds[{index}]: baseline value lies outside marginal interval"
                    ));
                }
                if !contains(after_interval, after) {
                    errors.push(format!(
                        "paired_worlds[{index}]: after-edit value lies outside marginal interval"
                    ));
                }
                if !contains(effect_interval, after - baseline) {
                    errors.push(format!(
                        "paired_worlds[{index}]: paired difference lies outside effect interval"
                    ));
                }
            }
        } else {
            let _ = finite_number(world, "baseline_value", &mut errors);
            let _ = finite_number(world, "after_edit_value", &mut errors);
        }
    }

    let expected: BTreeSet<String> = expected_world_ids.iter().cloned().collect();
    let missing: Vec<String> = expected.difference(&evaluated_ids).cloned().collect();
    let extra: Vec<String> = evaluated_ids.difference(&expected).cloned().collect();
    if !missing.is_empty() || !extra.is_empty() {
        errors.push(format!(
            "paired_worlds: exact Cartesian coverage mismatch; missing={}, extra={}",
            missing.len(),
            extra.len()
        ));
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    let axes_value = manifest.get("axes").cloned().unwrap_or(Value::Null);
    let enumeration_input = json!({
        "protocol": PROTOCOL,
        "admissibility_policy": POLICY,
        "axes": axes_value,
        "source_universe_snapshot_sha256": artifact_digests.get("source_universe_snapshot")
    });
    let enumeration_input_sha256 =
        canonical_sha256(&enumeration_input).map_err(|error| vec![error])?;
    let expected_world_manifest = Value::Array(
        expected_world_ids
            .iter()
            .map(|id| Value::String(id.clone()))
            .collect(),
    );
    let expected_world_manifest_sha256 =
        canonical_sha256(&expected_world_manifest).map_err(|error| vec![error])?;
    let admissibility_policy_sha256 = sha256(POLICY.as_bytes());

    let closure_proof = json!({
        "protocol": "CARTESIAN-CLOSURE-PROOF-V1",
        "universe_id": universe_id,
        "manifest_sha256": computed_manifest_sha,
        "enumeration_input_sha256": enumeration_input_sha256,
        "expected_world_manifest_sha256": expected_world_manifest_sha256,
        "expected_world_count": expected_world_ids.len(),
        "evaluated_world_count": evaluated_ids.len(),
        "missing_world_count": 0,
        "extra_world_count": 0,
        "unresolved_frontier_count": 0,
        "rule": POLICY
    });
    let closure_proof_sha256 = canonical_sha256(&closure_proof).map_err(|error| vec![error])?;

    let mut coverage = json!({
        "universe_id": universe_id,
        "candidate_id": binding.get("candidate_id"),
        "operation_id": binding.get("operation_id"),
        "sample_hash": binding.get("sample_hash"),
        "task_route": binding.get("task_route"),
        "query_id": binding.get("query_id"),
        "scenario_id": binding.get("scenario_id"),
        "query_definition_sha256": binding.get("query_definition_sha256"),
        "metric_id": binding.get("metric_id"),
        "unit_id": binding.get("unit_id"),
        "decision_threshold": binding.get("decision_threshold"),
        "materiality_threshold": binding.get("materiality_threshold"),
        "source_volume_id": binding.get("source_volume_id"),
        "base_graph_snapshot_sha256": binding.get("base_graph_snapshot_sha256"),
        "hypothetical_result_graph_snapshot_sha256": binding.get("hypothetical_result_graph_snapshot_sha256"),
        "candidate_patch_sha256": binding.get("candidate_patch_sha256"),
        "universe_manifest_sha256": computed_manifest_sha,
        "admissibility_policy_sha256": admissibility_policy_sha256,
        "enumeration_input_sha256": enumeration_input_sha256,
        "enumerator_sha256": verifier_sha256,
        "expected_world_manifest_sha256": expected_world_manifest_sha256,
        "closure_proof_sha256": closure_proof_sha256,
        "verifier_sha256": verifier_sha256,
        "expected_world_ids": expected_world_ids,
        "closure_verified": false,
        "finite_cartesian_enumeration_verified": true
    });
    let mut receipt = json!({
        "protocol": RECEIPT_PROTOCOL,
        "disposition": "VERIFIED",
        "coverage": coverage,
        "expected_world_count": expected_world_ids.len(),
        "evaluated_world_count": evaluated_ids.len(),
        "trust_boundary": {
            "prequery_freeze_authentication": "NOT_PERFORMED_EXTERNAL_TRUST_REQUIRED",
            "universe_selection_independence": "NOT_ASSESSED",
            "alternative_biological_admissibility": "NOT_ASSESSED"
        },
        "biological_exhaustiveness": false,
        "errors": []
    });
    let mut receipt_for_hash = receipt.clone();
    if let Some(object) = receipt_for_hash.as_object_mut() {
        object.remove("verification_receipt_sha256");
    }
    if let Some(coverage_object) = receipt_for_hash
        .get_mut("coverage")
        .and_then(Value::as_object_mut)
    {
        coverage_object.remove("verification_receipt_sha256");
    }
    let verification_receipt_sha256 =
        canonical_sha256(&receipt_for_hash).map_err(|error| vec![error])?;
    if let Some(object) = receipt.as_object_mut() {
        object.insert(
            "verification_receipt_sha256".to_owned(),
            Value::String(verification_receipt_sha256.clone()),
        );
    }
    coverage = receipt.get("coverage").cloned().unwrap_or(Value::Null);
    if let Some(object) = coverage.as_object_mut() {
        object.insert(
            "verification_receipt_sha256".to_owned(),
            Value::String(verification_receipt_sha256),
        );
    }
    if let Some(object) = receipt.as_object_mut() {
        object.insert("coverage".to_owned(), coverage);
    }
    Ok(receipt)
}

fn executable_sha256() -> Result<String, String> {
    let path = env::current_exe()
        .map_err(|error| format!("cannot locate verifier executable: {error}"))?;
    file_sha256(&path)
}

fn main() {
    let arguments: Vec<String> = env::args().collect();
    if arguments.len() < 2 || arguments.len() > 3 {
        eprintln!(
            "usage: connectome-worldset-verifier <universe-manifest.json> [receipt-output.json]"
        );
        std::process::exit(64);
    }
    let verifier_sha = match executable_sha256() {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(70);
        }
    };
    match verify_manifest(Path::new(&arguments[1]), &verifier_sha) {
        Ok(receipt) => {
            let encoded = serde_json::to_string_pretty(&receipt).expect("receipt is serializable");
            if arguments.len() == 3
                && let Err(error) = fs::write(&arguments[2], &encoded)
            {
                eprintln!("cannot write receipt: {error}");
                std::process::exit(74);
            }
            println!("{encoded}");
        }
        Err(errors) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "protocol": RECEIPT_PROTOCOL,
                    "disposition": "INVALID",
                    "closure_verified": false,
                    "errors": errors,
                    "biological_exhaustiveness": false
                }))
                .expect("error report is serializable")
            );
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

    fn test_root() -> PathBuf {
        let path = env::temp_dir().join(format!(
            "connectome-worldset-verifier-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).expect("test directory");
        path
    }

    fn add_artifact(root: &Path, id: &str, filename: &str, bytes: &[u8]) -> Value {
        fs::write(root.join(filename), bytes).expect("write test artifact");
        json!({
            "artifact_id": id,
            "relative_path": filename,
            "sha256": sha256(bytes)
        })
    }

    fn seal_manifest(mut manifest: Value) -> Value {
        let digest = canonical_sha256(&without_field(&manifest, "manifest_sha256"))
            .expect("manifest digest");
        manifest
            .as_object_mut()
            .expect("object")
            .insert("manifest_sha256".to_owned(), Value::String(digest));
        manifest
    }

    fn fixture(root: &Path) -> Value {
        let artifacts = vec![
            add_artifact(
                root,
                "prequery_freeze_receipt",
                "freeze.json",
                br#"{"frozen":true}"#,
            ),
            add_artifact(
                root,
                "source_universe_snapshot",
                "source.json",
                br#"{"source":"synthetic"}"#,
            ),
            add_artifact(root, "base_graph_snapshot", "base.graph", b"base-graph"),
            add_artifact(
                root,
                "hypothetical_result_graph_snapshot",
                "after.graph",
                b"after-graph",
            ),
            add_artifact(root, "candidate_patch", "patch.bin", b"patch"),
            add_artifact(root, "a1_evidence", "a1.json", b"axis-a-one"),
            add_artifact(root, "a2_evidence", "a2.json", b"axis-a-two"),
            add_artifact(root, "b1_evidence", "b1.json", b"axis-b-one"),
            add_artifact(root, "b2_evidence", "b2.json", b"axis-b-two"),
        ];
        let digest_for = |id: &str| {
            artifacts
                .iter()
                .find(|artifact| artifact["artifact_id"] == id)
                .expect("artifact")["sha256"]
                .as_str()
                .expect("digest")
                .to_owned()
        };
        let axes = json!([
            {
                "axis_id": "detector_state",
                "alternatives": [
                    {"alternative_id":"low","evidence_artifact_id":"a1_evidence","evidence_sha256":digest_for("a1_evidence")},
                    {"alternative_id":"high","evidence_artifact_id":"a2_evidence","evidence_sha256":digest_for("a2_evidence")}
                ]
            },
            {
                "axis_id": "structural_hypothesis",
                "alternatives": [
                    {"alternative_id":"merge","evidence_artifact_id":"b1_evidence","evidence_sha256":digest_for("b1_evidence")},
                    {"alternative_id":"separate","evidence_artifact_id":"b2_evidence","evidence_sha256":digest_for("b2_evidence")}
                ]
            }
        ]);
        let mut worlds = Vec::new();
        for detector in ["high", "low"] {
            for structure in ["merge", "separate"] {
                let mut assignment = Map::new();
                assignment.insert(
                    "detector_state".to_owned(),
                    Value::String(detector.to_owned()),
                );
                assignment.insert(
                    "structural_hypothesis".to_owned(),
                    Value::String(structure.to_owned()),
                );
                worlds.push(json!({
                    "world_id": world_id(&assignment).expect("world ID"),
                    "assignment": Value::Object(assignment),
                    "baseline_value": 0.2,
                    "after_edit_value": if detector == "high" { 0.8 } else { 0.3 }
                }));
            }
        }
        let artifact_digest = |id: &str| digest_for(id);
        seal_manifest(json!({
            "schema_version": "0.1",
            "protocol": PROTOCOL,
            "universe_id": "synthetic-universe-01",
            "admissibility_policy": POLICY,
            "manifest_sha256": "",
            "context": {
                "candidate_id": "candidate-01",
                "operation_id": "operation-01",
                "sample_hash": "0123456789abcdef0123456789abcdef",
                "task_route": "false_split_correction",
                "query_id": "query-01",
                "scenario_id": "scenario-01",
                "query_definition_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "metric_id": "connection_count",
                "unit_id": "count",
                "decision_threshold": 0.5,
                "materiality_threshold": 1.0,
                "source_volume_id": "synthetic-volume-01",
                "base_graph_snapshot_sha256": artifact_digest("base_graph_snapshot"),
                "hypothetical_result_graph_snapshot_sha256": artifact_digest("hypothetical_result_graph_snapshot"),
                "candidate_patch_sha256": artifact_digest("candidate_patch"),
                "baseline_interval": {"lower":0.0,"upper":1.0},
                "after_edit_interval": {"lower":0.0,"upper":1.0},
                "paired_effect_interval": {"lower":-1.0,"upper":1.0}
            },
            "freeze": {
                "universe_frozen_before_query_scoring": true,
                "query_scoring_opened_before_freeze": false,
                "unresolved_frontier_count": 0,
                "prequery_freeze_receipt_sha256": artifact_digest("prequery_freeze_receipt")
            },
            "artifact_files": artifacts,
            "axes": axes,
            "paired_worlds": worlds
        }))
    }

    #[test]
    fn full_cartesian_product_yields_adapter_ready_coverage_receipt() {
        let root = test_root();
        let path = root.join("universe.json");
        let manifest = fixture(&root);
        fs::write(
            &path,
            serde_json::to_vec_pretty(&manifest).expect("serialize"),
        )
        .expect("write manifest");
        let verifier = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
        let receipt = verify_manifest(&path, verifier).expect("verified finite universe");
        assert_eq!(receipt["disposition"], "VERIFIED");
        assert!(
            !receipt["coverage"]["closure_verified"]
                .as_bool()
                .expect("trusted closure flag must remain false without external trust")
        );
        assert!(
            receipt["coverage"]["finite_cartesian_enumeration_verified"]
                .as_bool()
                .expect("finite product enumeration result")
        );
        assert_eq!(receipt["expected_world_count"], 4);
        assert_eq!(receipt["evaluated_world_count"], 4);
        assert_eq!(
            receipt["trust_boundary"]["prequery_freeze_authentication"],
            "NOT_PERFORMED_EXTERNAL_TRUST_REQUIRED"
        );
        assert_eq!(
            receipt["trust_boundary"]["universe_selection_independence"],
            "NOT_ASSESSED"
        );
        assert_eq!(
            receipt["coverage"]["expected_world_ids"]
                .as_array()
                .expect("IDs")
                .len(),
            4
        );
        assert!(is_sha256(
            receipt["coverage"]["verification_receipt_sha256"]
                .as_str()
                .expect("receipt digest")
        ));
    }

    #[test]
    fn omitted_world_is_rejected_even_after_manifest_is_resealed() {
        let root = test_root();
        let path = root.join("universe.json");
        let mut manifest = fixture(&root);
        manifest["paired_worlds"]
            .as_array_mut()
            .expect("worlds")
            .pop();
        manifest = seal_manifest(manifest);
        fs::write(
            &path,
            serde_json::to_vec_pretty(&manifest).expect("serialize"),
        )
        .expect("write manifest");
        let errors = verify_manifest(&path, &"c".repeat(64)).expect_err("missing world");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("exact Cartesian coverage mismatch"))
        );
    }

    #[test]
    fn duplicated_world_is_rejected() {
        let root = test_root();
        let path = root.join("universe.json");
        let mut manifest = fixture(&root);
        let duplicate = manifest["paired_worlds"][0].clone();
        manifest["paired_worlds"]
            .as_array_mut()
            .expect("worlds")
            .push(duplicate);
        manifest = seal_manifest(manifest);
        fs::write(
            &path,
            serde_json::to_vec_pretty(&manifest).expect("serialize"),
        )
        .expect("write manifest");
        let errors = verify_manifest(&path, &"c".repeat(64)).expect_err("duplicate world");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("duplicate world ID"))
        );
    }

    #[test]
    fn changed_artifact_bytes_are_rejected() {
        let root = test_root();
        let path = root.join("universe.json");
        let manifest = fixture(&root);
        fs::write(root.join("base.graph"), b"tampered").expect("mutate artifact");
        fs::write(
            &path,
            serde_json::to_vec_pretty(&manifest).expect("serialize"),
        )
        .expect("write manifest");
        let errors = verify_manifest(&path, &"c".repeat(64)).expect_err("artifact mismatch");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("base_graph_snapshot"))
        );
    }

    #[test]
    fn paired_values_outside_marginal_interval_are_rejected() {
        let root = test_root();
        let path = root.join("universe.json");
        let mut manifest = fixture(&root);
        manifest["paired_worlds"][0]["baseline_value"] = json!(2.0);
        manifest = seal_manifest(manifest);
        fs::write(
            &path,
            serde_json::to_vec_pretty(&manifest).expect("serialize"),
        )
        .expect("write manifest");
        let errors = verify_manifest(&path, &"c".repeat(64)).expect_err("out of interval");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("baseline value lies outside"))
        );
    }

    #[test]
    fn unresolved_frontier_cannot_be_called_complete() {
        let root = test_root();
        let path = root.join("universe.json");
        let mut manifest = fixture(&root);
        manifest["freeze"]["unresolved_frontier_count"] = json!(1);
        manifest = seal_manifest(manifest);
        fs::write(
            &path,
            serde_json::to_vec_pretty(&manifest).expect("serialize"),
        )
        .expect("write manifest");
        let errors = verify_manifest(&path, &"c".repeat(64)).expect_err("frontier remains");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("unresolved_frontier_count"))
        );
    }

    #[test]
    fn manifest_digest_covers_the_complete_declared_document() {
        let root = test_root();
        let path = root.join("universe.json");
        let mut manifest = fixture(&root);
        manifest["universe_id"] = json!("tampered-universe");
        fs::write(
            &path,
            serde_json::to_vec_pretty(&manifest).expect("serialize"),
        )
        .expect("write manifest");
        let errors = verify_manifest(&path, &"c".repeat(64)).expect_err("stale digest");
        assert!(errors.iter().any(|error| error.contains("manifest_sha256")));
    }
    #[test]
    fn parent_traversal_in_artifact_path_is_rejected() {
        let root = test_root();
        let path = root.join("universe.json");
        let mut manifest = fixture(&root);
        manifest["artifact_files"][0]["relative_path"] = json!("../freeze.json");
        manifest = seal_manifest(manifest);
        fs::write(
            &path,
            serde_json::to_vec_pretty(&manifest).expect("serialize"),
        )
        .expect("write manifest");
        let errors = verify_manifest(&path, &"c".repeat(64)).expect_err("path traversal");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("cannot traverse directories"))
        );
    }

    #[test]
    fn alternative_must_bind_exact_evidence_artifact_id_and_digest() {
        let root = test_root();
        let path = root.join("universe.json");
        let mut manifest = fixture(&root);
        manifest["axes"][0]["alternatives"][0]["evidence_artifact_id"] =
            json!("base_graph_snapshot");
        manifest = seal_manifest(manifest);
        fs::write(
            &path,
            serde_json::to_vec_pretty(&manifest).expect("serialize"),
        )
        .expect("write manifest");
        let errors = verify_manifest(&path, &"c".repeat(64)).expect_err("wrong evidence artifact");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("evidence digest mismatch"))
        );
    }

    #[test]
    fn axis_count_is_bounded_before_recursive_enumeration() {
        let root = test_root();
        let path = root.join("universe.json");
        let mut manifest = fixture(&root);
        let mut axes = manifest["axes"].as_array().expect("axes").clone();
        while axes.len() <= MAX_AXES {
            axes.push(json!({
                "axis_id": format!("extra-{}", axes.len()),
                "alternatives": [{
                    "alternative_id": "only",
                    "evidence_artifact_id": "a1_evidence",
                    "evidence_sha256": manifest["artifact_files"][5]["sha256"]
                }]
            }));
        }
        manifest["axes"] = Value::Array(axes);
        manifest = seal_manifest(manifest);
        fs::write(
            &path,
            serde_json::to_vec_pretty(&manifest).expect("serialize"),
        )
        .expect("write manifest");
        let errors = verify_manifest(&path, &"c".repeat(64)).expect_err("axis cap");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("exceed the V1 safety cap"))
        );
    }
}
