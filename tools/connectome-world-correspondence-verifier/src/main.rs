#![forbid(unsafe_code)]

//! Fail-closed semantic verifier for versioned connectome correspondence receipts.
//!
//! V0.1 verifies structural consistency, exact sidecar bytes, object membership,
//! relation/cardinality semantics, summary counters and query-output declarations.
//! It deliberately does not authenticate evidence artifacts, edit-history receipts,
//! expert adjudication, coordinate-registration anchors, signer identities or keys.
//! Therefore it never promotes a biological correspondence claim.

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::process;

const SCHEMA_VERSION: &str = "0.1.0";
const REPORT_SCHEMA_SHA256: &str =
    "10c3eb5ab6deb5d4044897501a97336e8ae3ec2a6a3520e9d19e5f313dccad9c";
const WORLD_SCHEMA_SHA256: &str =
    "b8bbc20a6e581232b7439a1fbbbecf65bbcb6b6b702732c98720a2dce8575f06";
const PROTOCOL_SHA256: &str =
    "935643661ebacd206cee3dc153806deba6dd245dbf20c5d4b402207c16b53c92";
const POLICY_SHA256: &str =
    "2d9c68e6c835e266225a680601d79e15c1aed79ff53204b67386187b001f45eb";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cardinality {
    OneToOne,
    OneToMany,
    ManyToOne,
    ManyToMany,
    Unmapped,
}

impl Cardinality {
    fn as_str(self) -> &'static str {
        match self {
            Self::OneToOne => "ONE_TO_ONE",
            Self::OneToMany => "ONE_TO_MANY",
            Self::ManyToOne => "MANY_TO_ONE",
            Self::ManyToMany => "MANY_TO_MANY",
            Self::Unmapped => "UNMAPPED",
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Summary {
    mapping_count: u64,
    verified_count: u64,
    ambiguous_count: u64,
    not_estimable_count: u64,
    rejected_count: u64,
    unmapped_source_object_count: u64,
    one_to_many_count: u64,
    many_to_one_count: u64,
    many_to_many_count: u64,
    query_same_count: u64,
    query_different_count: u64,
    query_not_estimable_count: u64,
}

#[derive(Debug)]
struct WorldManifest {
    world: Value,
    manifest: Value,
    objects: BTreeMap<String, String>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn object<'a>(value: &'a Value, context: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{context}: expected object"))
}

fn array<'a>(value: &'a Value, context: &str) -> Result<&'a Vec<Value>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("{context}: expected array"))
}

fn string_field(value: &Value, key: &str, context: &str) -> Result<String, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("{context}.{key}: expected non-empty string"))
}

fn optional_string_field(value: &Value, key: &str, context: &str) -> Result<Option<String>, String> {
    match value.get(key) {
        Some(Value::Null) => Ok(None),
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(Some(value.clone())),
        _ => Err(format!("{context}.{key}: expected non-empty string or null")),
    }
}

fn digest_field(value: &Value, key: &str, context: &str) -> Result<String, String> {
    let digest = string_field(value, key, context)?;
    if !valid_sha256(&digest) {
        return Err(format!("{context}.{key}: expected lowercase SHA-256 hex"));
    }
    Ok(digest)
}

fn optional_digest_field(value: &Value, key: &str, context: &str) -> Result<Option<String>, String> {
    match value.get(key) {
        Some(Value::Null) => Ok(None),
        Some(Value::String(value)) if valid_sha256(value) => Ok(Some(value.clone())),
        _ => Err(format!("{context}.{key}: expected lowercase SHA-256 hex or null")),
    }
}

fn require_keys(
    value: &Value,
    expected: &[&str],
    context: &str,
) -> Result<(), String> {
    let map = object(value, context)?;
    if map.len() != expected.len() || expected.iter().any(|key| !map.contains_key(*key)) {
        return Err(format!(
            "{context}: unexpected or missing key; exact keys required: {}",
            expected.join(", ")
        ));
    }
    Ok(())
}

fn json_number(value: &Value, context: &str) -> Result<f64, String> {
    let number = value
        .as_f64()
        .ok_or_else(|| format!("{context}: expected finite JSON number"))?;
    if !number.is_finite() {
        return Err(format!("{context}: non-finite number"));
    }
    Ok(number)
}

fn read_world_manifest(bytes: &[u8], context: &str) -> Result<WorldManifest, String> {
    let root: Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("{context}: invalid JSON: {error}"))?;
    require_keys(&root, &["schema_version", "world", "manifest"], context)?;
    if string_field(&root, "schema_version", context)? != SCHEMA_VERSION {
        return Err(format!("{context}: unsupported schema version"));
    }

    let world = root.get("world").ok_or_else(|| format!("{context}.world missing"))?;
    require_keys(
        world,
        &["world_id", "source_dataset_id", "source_volume_id", "release_id", "objects"],
        &format!("{context}.world"),
    )?;
    for key in ["world_id", "source_dataset_id", "source_volume_id", "release_id"] {
        string_field(world, key, &format!("{context}.world"))?;
    }

    let object_values = array(
        world.get("objects").ok_or_else(|| format!("{context}.world.objects missing"))?,
        &format!("{context}.world.objects"),
    )?;
    if object_values.is_empty() {
        return Err(format!("{context}: world object inventory must not be empty"));
    }
    let mut objects = BTreeMap::new();
    for (index, object_value) in object_values.iter().enumerate() {
        let item_context = format!("{context}.world.objects[{index}]");
        require_keys(
            object_value,
            &["object_id", "object_type", "lineage_id", "stable_anchor_id"],
            &item_context,
        )?;
        let object_id = string_field(object_value, "object_id", &item_context)?;
        let object_type = string_field(object_value, "object_type", &item_context)?;
        if !matches!(
            object_type.as_str(),
            "SEGMENT" | "NEURON" | "SYNAPSE" | "ARTIFACT"
        ) {
            return Err(format!("{item_context}: unsupported object_type {object_type}"));
        }
        optional_string_field(object_value, "lineage_id", &item_context)?;
        optional_string_field(object_value, "stable_anchor_id", &item_context)?;
        if objects.insert(object_id.clone(), object_type).is_some() {
            return Err(format!("{context}: duplicate object_id {object_id}"));
        }
    }

    let manifest = root
        .get("manifest")
        .ok_or_else(|| format!("{context}.manifest missing"))?;
    require_keys(
        manifest,
        &[
            "segmentation_sha256",
            "mesh_sha256",
            "skeleton_sha256",
            "synapse_annotations_sha256",
            "coordinate_frame_id",
            "voxel_size_nm",
        ],
        &format!("{context}.manifest"),
    )?;
    digest_field(manifest, "segmentation_sha256", context)?;
    optional_digest_field(manifest, "mesh_sha256", context)?;
    optional_digest_field(manifest, "skeleton_sha256", context)?;
    optional_digest_field(manifest, "synapse_annotations_sha256", context)?;
    string_field(manifest, "coordinate_frame_id", context)?;
    let voxel = array(
        manifest.get("voxel_size_nm").ok_or_else(|| format!("{context}.manifest.voxel_size_nm missing"))?,
        &format!("{context}.manifest.voxel_size_nm"),
    )?;
    if voxel.len() != 3 {
        return Err(format!("{context}: voxel_size_nm must contain exactly three axes"));
    }
    for (i, item) in voxel.iter().enumerate() {
        if json_number(item, &format!("{context}.voxel_size_nm[{i}]"))? <= 0.0 {
            return Err(format!("{context}: voxel sizes must be positive"));
        }
    }

    Ok(WorldManifest {
        world: world.clone(),
        manifest: manifest.clone(),
        objects,
    })
}

fn compare_world_identity(
    receipt_world: &Value,
    manifest_world: &Value,
    context: &str,
) -> Result<(), String> {
    require_keys(
        receipt_world,
        &["world_id", "source_dataset_id", "source_volume_id", "release_id"],
        context,
    )?;
    for key in ["world_id", "source_dataset_id", "source_volume_id", "release_id"] {
        if string_field(receipt_world, key, context)? != string_field(manifest_world, key, context)? {
            return Err(format!("{context}.{key}: receipt and sidecar identity disagree"));
        }
    }
    Ok(())
}

fn compare_artifact_manifest(
    receipt_artifact: &Value,
    manifest_sidecar: &WorldManifest,
    raw_bytes: &[u8],
    context: &str,
) -> Result<(), String> {
    require_keys(
        receipt_artifact,
        &[
            "manifest_sha256",
            "segmentation_sha256",
            "mesh_sha256",
            "skeleton_sha256",
            "synapse_annotations_sha256",
            "coordinate_frame_id",
            "voxel_size_nm",
        ],
        context,
    )?;
    let raw_digest = sha256_hex(raw_bytes);
    if digest_field(receipt_artifact, "manifest_sha256", context)? != raw_digest {
        return Err(format!("{context}.manifest_sha256: sidecar raw-byte digest mismatch"));
    }
    for key in [
        "segmentation_sha256",
        "mesh_sha256",
        "skeleton_sha256",
        "synapse_annotations_sha256",
        "coordinate_frame_id",
        "voxel_size_nm",
    ] {
        if receipt_artifact.get(key) != manifest_sidecar.manifest.get(key) {
            return Err(format!("{context}.{key}: receipt and sidecar manifest disagree"));
        }
    }
    Ok(())
}

fn parse_cardinality(source_count: usize, target_count: usize) -> Result<Cardinality, String> {
    if source_count == 0 {
        return Err("mapping source_object_ids must not be empty".to_owned());
    }
    Ok(match (source_count, target_count) {
        (_, 0) => Cardinality::Unmapped,
        (1, 1) => Cardinality::OneToOne,
        (1, _) => Cardinality::OneToMany,
        (_, 1) => Cardinality::ManyToOne,
        (_, _) => Cardinality::ManyToMany,
    })
}

fn expected_object_type(relation_type: &str) -> Result<&'static str, String> {
    match relation_type {
        "ARTIFACT_IDENTITY" => Ok("ARTIFACT"),
        "SEGMENT_LINEAGE" => Ok("SEGMENT"),
        "NEURON_CORRESPONDENCE" => Ok("NEURON"),
        "SYNAPSE_CORRESPONDENCE" => Ok("SYNAPSE"),
        _ => Err(format!("unsupported relation_type {relation_type}")),
    }
}

fn string_array(value: &Value, context: &str) -> Result<Vec<String>, String> {
    let values = array(value, context)?;
    let mut output = Vec::with_capacity(values.len());
    let mut seen = BTreeSet::new();
    for item in values {
        let id = item
            .as_str()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| format!("{context}: expected non-empty string items"))?
            .to_owned();
        if !seen.insert(id.clone()) {
            return Err(format!("{context}: duplicate ID {id}"));
        }
        output.push(id);
    }
    Ok(output)
}

fn validate_registration(receipt: &Value) -> Result<&'static str, String> {
    let registration = receipt
        .get("registration")
        .ok_or_else(|| "registration missing".to_owned())?;
    require_keys(
        registration,
        &[
            "status",
            "transform_source_to_target_4x4",
            "max_residual_nm",
            "uncertainty_bound_nm",
            "coordinate_frame_policy_sha256",
        ],
        "registration",
    )?;
    let status = string_field(registration, "status", "registration")?;
    if !matches!(
        status.as_str(),
        "VERIFIED" | "UNCERTAIN" | "NOT_REGISTERED" | "REJECTED"
    ) {
        return Err("registration.status invalid".to_owned());
    }
    digest_field(registration, "coordinate_frame_policy_sha256", "registration")?;

    let rows = array(
        registration
            .get("transform_source_to_target_4x4")
            .ok_or_else(|| "registration transform missing".to_owned())?,
        "registration.transform_source_to_target_4x4",
    )?;
    if rows.len() != 4 {
        return Err("registration transform must have four rows".to_owned());
    }
    let mut matrix = [[0.0_f64; 4]; 4];
    for (row_index, row_value) in rows.iter().enumerate() {
        let row = array(row_value, &format!("registration.transform[{row_index}]"))?;
        if row.len() != 4 {
            return Err("registration transform must be 4x4".to_owned());
        }
        for (column_index, value) in row.iter().enumerate() {
            matrix[row_index][column_index] =
                json_number(value, &format!("registration.transform[{row_index}][{column_index}]"))?;
        }
    }
    let bottom = matrix[3];
    if bottom[0].abs() > 1e-9
        || bottom[1].abs() > 1e-9
        || bottom[2].abs() > 1e-9
        || (bottom[3] - 1.0).abs() > 1e-9
    {
        return Err("registration transform violates affine column-vector convention".to_owned());
    }
    let a = matrix[0][0];
    let b = matrix[0][1];
    let c = matrix[0][2];
    let d = matrix[1][0];
    let e = matrix[1][1];
    let f = matrix[1][2];
    let g = matrix[2][0];
    let h = matrix[2][1];
    let i = matrix[2][2];
    let determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if !determinant.is_finite() || determinant.abs() <= 1e-12 {
        return Err("registration transform upper-left 3x3 matrix is singular".to_owned());
    }

    let residual = registration.get("max_residual_nm")
        .ok_or_else(|| "registration.max_residual_nm missing".to_owned())?;
    let uncertainty = registration.get("uncertainty_bound_nm")
        .ok_or_else(|| "registration.uncertainty_bound_nm missing".to_owned())?;
    for (name, value) in [("max_residual_nm", residual), ("uncertainty_bound_nm", uncertainty)] {
        if !value.is_null() && json_number(value, &format!("registration.{name}"))? < 0.0 {
            return Err(format!("registration.{name} must be non-negative"));
        }
    }
    if status == "VERIFIED" && (residual.is_null() || uncertainty.is_null()) {
        return Err("VERIFIED registration requires residual and uncertainty values".to_owned());
    }
    Ok(match status.as_str() {
        "VERIFIED" => "CLAIMED_VERIFIED_NOT_AUTHENTICATED",
        "UNCERTAIN" => "CLAIMED_UNCERTAIN",
        "NOT_REGISTERED" => "CLAIMED_NOT_REGISTERED",
        "REJECTED" => "CLAIMED_REJECTED_NOT_AUTHENTICATED",
        _ => unreachable!(),
    })
}

fn count_summary(value: &Value, expected: &Summary) -> Result<(), String> {
    require_keys(
        value,
        &[
            "mapping_count",
            "claimed_verified_count",
            "ambiguous_count",
            "not_estimable_count",
            "rejected_count",
            "unmapped_source_object_count",
            "one_to_many_count",
            "many_to_one_count",
            "many_to_many_count",
            "query_same_count",
            "query_different_count",
            "query_not_estimable_count",
        ],
        "summary",
    )?;
    let expected_fields = [
        ("mapping_count", expected.mapping_count),
        ("claimed_verified_count", expected.verified_count),
        ("ambiguous_count", expected.ambiguous_count),
        ("not_estimable_count", expected.not_estimable_count),
        ("rejected_count", expected.rejected_count),
        ("unmapped_source_object_count", expected.unmapped_source_object_count),
        ("one_to_many_count", expected.one_to_many_count),
        ("many_to_one_count", expected.many_to_one_count),
        ("many_to_many_count", expected.many_to_many_count),
        ("query_same_count", expected.query_same_count),
        ("query_different_count", expected.query_different_count),
        ("query_not_estimable_count", expected.query_not_estimable_count),
    ];
    for (key, expected_value) in expected_fields {
        let actual = value
            .get(key)
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("summary.{key}: expected non-negative integer"))?;
        if actual != expected_value {
            return Err(format!(
                "summary.{key}: reported {actual}, independently recomputed {expected_value}"
            ));
        }
    }
    Ok(())
}

fn verify_receipt(
    receipt_bytes: &[u8],
    source_manifest_bytes: &[u8],
    target_manifest_bytes: &[u8],
) -> Result<Value, String> {
    let receipt: Value = serde_json::from_slice(receipt_bytes)
        .map_err(|error| format!("receipt JSON invalid: {error}"))?;
    require_keys(
        &receipt,
        &[
            "schema_version",
            "receipt_id",
            "source_world",
            "target_world",
            "artifact_manifests",
            "registration",
            "mappings",
            "query_comparisons",
            "summary",
            "disposition",
            "provenance",
        ],
        "receipt",
    )?;
    if string_field(&receipt, "schema_version", "receipt")? != SCHEMA_VERSION {
        return Err("unsupported receipt schema_version".to_owned());
    }
    let receipt_id = string_field(&receipt, "receipt_id", "receipt")?;

    let source = read_world_manifest(source_manifest_bytes, "source_manifest")?;
    let target = read_world_manifest(target_manifest_bytes, "target_manifest")?;
    compare_world_identity(
        receipt.get("source_world").ok_or_else(|| "source_world missing".to_owned())?,
        &source.world,
        "source_world",
    )?;
    compare_world_identity(
        receipt.get("target_world").ok_or_else(|| "target_world missing".to_owned())?,
        &target.world,
        "target_world",
    )?;

    let artifact_manifests = receipt
        .get("artifact_manifests")
        .ok_or_else(|| "artifact_manifests missing".to_owned())?;
    require_keys(artifact_manifests, &["source", "target"], "artifact_manifests")?;
    compare_artifact_manifest(
        artifact_manifests.get("source").ok_or_else(|| "artifact_manifests.source missing".to_owned())?,
        &source,
        source_manifest_bytes,
        "artifact_manifests.source",
    )?;
    compare_artifact_manifest(
        artifact_manifests.get("target").ok_or_else(|| "artifact_manifests.target missing".to_owned())?,
        &target,
        target_manifest_bytes,
        "artifact_manifests.target",
    )?;

    let provenance = receipt.get("provenance").ok_or_else(|| "provenance missing".to_owned())?;
    require_keys(
        provenance,
        &[
            "source_manifest_sha256",
            "target_manifest_sha256",
            "lineage_receipt_sha256",
            "mapping_policy_sha256",
            "correspondence_engine_sha256",
            "verifier_protocol_sha256",
            "verifier_sha256",
            "report_schema_sha256",
            "world_manifest_schema_sha256",
        ],
        "provenance",
    )?;
    let source_digest = digest_field(provenance, "source_manifest_sha256", "provenance")?;
    let target_digest = digest_field(provenance, "target_manifest_sha256", "provenance")?;
    if source_digest != sha256_hex(source_manifest_bytes)
        || target_digest != sha256_hex(target_manifest_bytes)
    {
        return Err("provenance manifest digest does not match exact sidecar bytes".to_owned());
    }
    if source_digest != digest_field(artifact_manifests.get("source").unwrap(), "manifest_sha256", "artifact_manifests.source")?
        || target_digest != digest_field(artifact_manifests.get("target").unwrap(), "manifest_sha256", "artifact_manifests.target")?
    {
        return Err("artifact_manifests and provenance manifest digests disagree".to_owned());
    }
    optional_digest_field(provenance, "lineage_receipt_sha256", "provenance")?;
    if digest_field(provenance, "mapping_policy_sha256", "provenance")? != POLICY_SHA256 {
        return Err("mapping_policy_sha256 does not match pinned evidence policy".to_owned());
    }
    if digest_field(provenance, "verifier_protocol_sha256", "provenance")? != PROTOCOL_SHA256 {
        return Err("verifier_protocol_sha256 does not match pinned correspondence protocol".to_owned());
    }
    if digest_field(provenance, "report_schema_sha256", "provenance")? != REPORT_SCHEMA_SHA256 {
        return Err("report_schema_sha256 does not match pinned receipt schema".to_owned());
    }
    if digest_field(provenance, "world_manifest_schema_sha256", "provenance")? != WORLD_SCHEMA_SHA256 {
        return Err("world_manifest_schema_sha256 does not match pinned world manifest schema".to_owned());
    }
    let engine_digest = optional_digest_field(provenance, "correspondence_engine_sha256", "provenance")?;
    let verifier_digest = optional_digest_field(provenance, "verifier_sha256", "provenance")?;

    let registration_claim = validate_registration(&receipt)?;

    let mappings = array(
        receipt.get("mappings").ok_or_else(|| "mappings missing".to_owned())?,
        "mappings",
    )?;
    if mappings.is_empty() {
        return Err("mappings must not be empty".to_owned());
    }

    let mut summary = Summary {
        mapping_count: mappings.len() as u64,
        ..Summary::default()
    };
    let mut mapping_ids = BTreeSet::new();
    let mut evidence_ids = BTreeSet::new();
    let mut source_coverage = BTreeSet::new();
    let mut target_coverage = BTreeSet::new();
    let mut unmapped_sources = BTreeSet::new();
    let mut source_mapping_kinds = BTreeMap::<String, (String, Option<String>)>::new();
    let mut alternative_groups =
        BTreeMap::<String, (String, Vec<String>, BTreeSet<Vec<String>>)>::new();

    for (index, mapping) in mappings.iter().enumerate() {
        let context = format!("mappings[{index}]");
        require_keys(
            mapping,
            &[
                "mapping_id",
                "relation_type",
                "source_object_ids",
                "target_object_ids",
                "cardinality",
                "support_status",
                "evidence",
                "rationale_code",
                "alternative_set_id",
            ],
            &context,
        )?;
        let mapping_id = string_field(mapping, "mapping_id", &context)?;
        if !mapping_ids.insert(mapping_id.clone()) {
            return Err(format!("duplicate mapping_id {mapping_id}"));
        }
        let relation_type = string_field(mapping, "relation_type", &context)?;
        let required_type = expected_object_type(&relation_type)?;
        let source_ids = string_array(
            mapping.get("source_object_ids").ok_or_else(|| format!("{context}.source_object_ids missing"))?,
            &format!("{context}.source_object_ids"),
        )?;
        let target_ids = string_array(
            mapping.get("target_object_ids").ok_or_else(|| format!("{context}.target_object_ids missing"))?,
            &format!("{context}.target_object_ids"),
        )?;
        let cardinality = parse_cardinality(source_ids.len(), target_ids.len())?;
        let reported_cardinality = string_field(mapping, "cardinality", &context)?;
        if reported_cardinality != cardinality.as_str() {
            return Err(format!(
                "{context}.cardinality: reported {reported_cardinality}, computed {}",
                cardinality.as_str()
            ));
        }
        match cardinality {
            Cardinality::OneToMany => summary.one_to_many_count += 1,
            Cardinality::ManyToOne => summary.many_to_one_count += 1,
            Cardinality::ManyToMany => summary.many_to_many_count += 1,
            Cardinality::OneToOne | Cardinality::Unmapped => {}
        }

        let support_status = string_field(mapping, "support_status", &context)?;
        match support_status.as_str() {
            "CLAIMED_VERIFIED" => summary.verified_count += 1,
            "AMBIGUOUS" => summary.ambiguous_count += 1,
            "NOT_ESTIMABLE" => summary.not_estimable_count += 1,
            "REJECTED" => summary.rejected_count += 1,
            _ => return Err(format!("{context}.support_status invalid")),
        }
        let alternative_set_id = optional_string_field(mapping, "alternative_set_id", &context)?;
        if support_status == "AMBIGUOUS" && (alternative_set_id.is_none() || target_ids.is_empty()) {
            return Err(format!("{context}: ambiguous mapping requires a non-empty target and alternative_set_id"));
        }
        if support_status != "AMBIGUOUS" && alternative_set_id.is_some() {
            return Err(format!("{context}: alternative_set_id only applies to AMBIGUOUS mappings"));
        }

        for source_id in &source_ids {
            let object_type = source
                .objects
                .get(source_id)
                .ok_or_else(|| format!("{context}: source object {source_id} absent from source inventory"))?;
            if object_type != required_type {
                return Err(format!(
                    "{context}: source object {source_id} is {object_type}, expected {required_type}"
                ));
            }
            source_coverage.insert(source_id.clone());
            let current = (relation_type.clone(), alternative_set_id.clone());
            if let Some(previous) = source_mapping_kinds.get(source_id) {
                if previous != &current || current.1.is_none() {
                    return Err(format!(
                        "{context}: source object {source_id} appears in repeated mapping rows without one shared, explicit alternative set"
                    ));
                }
            } else {
                source_mapping_kinds.insert(source_id.clone(), current);
            }
            if cardinality == Cardinality::Unmapped {
                unmapped_sources.insert(source_id.clone());
            }
        }
        for target_id in &target_ids {
            if !target_coverage.insert(target_id.clone()) {
                return Err(format!("{context}: target object {target_id} is claimed by more than one mapping row"));
            }
            let object_type = target
                .objects
                .get(target_id)
                .ok_or_else(|| format!("{context}: target object {target_id} absent from target inventory"))?;
            if object_type != required_type {
                return Err(format!(
                    "{context}: target object {target_id} is {object_type}, expected {required_type}"
                ));
            }
        }

        let evidence = array(
            mapping.get("evidence").ok_or_else(|| format!("{context}.evidence missing"))?,
            &format!("{context}.evidence"),
        )?;
        if evidence.is_empty() {
            return Err(format!("{context}: evidence list must not be empty"));
        }
        for (evidence_index, evidence_value) in evidence.iter().enumerate() {
            let evidence_context = format!("{context}.evidence[{evidence_index}]");
            require_keys(
                evidence_value,
                &["evidence_id", "kind", "artifact_sha256", "locator", "authority"],
                &evidence_context,
            )?;
            let evidence_id = string_field(evidence_value, "evidence_id", &evidence_context)?;
            if !evidence_ids.insert(evidence_id.clone()) {
                return Err(format!("duplicate evidence_id {evidence_id}"));
            }
            let kind = string_field(evidence_value, "kind", &evidence_context)?;
            if !matches!(
                kind.as_str(),
                "SOURCE_ANCHOR" | "EDIT_TRANSACTION" | "LINEAGE_RECEIPT"
                    | "PHYSICAL_REGISTRATION" | "MORPHOLOGY" | "SYNAPSE_PARTNER"
                    | "EXPERT_ADJUDICATION" | "OTHER"
            ) {
                return Err(format!("{evidence_context}: invalid evidence kind"));
            }
            digest_field(evidence_value, "artifact_sha256", &evidence_context)?;
            string_field(evidence_value, "locator", &evidence_context)?;
            let authority = string_field(evidence_value, "authority", &evidence_context)?;
            if !matches!(
                authority.as_str(),
                "SOURCE_DERIVED" | "INDEPENDENTLY_VERIFIED" | "EXPERT_ADJUDICATED"
                    | "MODEL_DERIVED" | "UNKNOWN"
            ) {
                return Err(format!("{evidence_context}: invalid evidence authority"));
            }
        }

        if let Some(group_id) = alternative_set_id {
            let group = alternative_groups
                .entry(group_id)
                .or_insert_with(|| (relation_type.clone(), source_ids.clone(), BTreeSet::new()));
            if group.0 != relation_type || group.1 != source_ids {
                return Err(format!("alternative set {group_id} mixes relation types or source ID sets"));
            }
            if target_ids.iter().any(|target_id| {
                group.2.iter().any(|existing| existing.contains(target_id))
            }) {
                return Err(format!("alternative set {group_id} reuses a target object across mutually exclusive alternatives"));
            }
            if !group.2.insert(target_ids.clone()) {
                return Err(format!("alternative set {group_id} contains duplicate target alternative"));
            }
        }
        string_field(mapping, "rationale_code", &context)?;
    }

    for (group_id, (_, _, alternatives)) in &alternative_groups {
        if alternatives.len() < 2 {
            return Err(format!("alternative set {group_id} must contain at least two distinct alternatives"));
        }
    }

    let source_inventory_ids: BTreeSet<String> =
        source.objects.keys().cloned().collect();
    if source_coverage != source_inventory_ids {
        let missing: Vec<String> = source
            .objects
            .keys()
            .filter(|id| !source_coverage.contains(*id))
            .cloned()
            .collect();
        return Err(format!("source inventory objects missing from correspondence receipt: {}", missing.join(", ")));
    }
    for unmapped_id in &unmapped_sources {
        if source_mapping_kinds
            .get(unmapped_id)
            .is_some_and(|(_, alternative)| alternative.is_some())
        {
            return Err(format!("unmapped source object {unmapped_id} cannot also be an ambiguous alternative"));
        }
    }
    summary.unmapped_source_object_count = unmapped_sources.len() as u64;

    let query_comparisons = array(
        receipt.get("query_comparisons").ok_or_else(|| "query_comparisons missing".to_owned())?,
        "query_comparisons",
    )?;
    let mut query_ids = BTreeSet::new();
    for (index, query) in query_comparisons.iter().enumerate() {
        let context = format!("query_comparisons[{index}]");
        require_keys(
            query,
            &[
                "query_id",
                "query_definition_sha256",
                "analysis_code_sha256",
                "source_output_sha256",
                "target_output_sha256",
                "equivalence",
            ],
            &context,
        )?;
        let query_id = string_field(query, "query_id", &context)?;
        if !query_ids.insert(query_id) {
            return Err(format!("{context}: duplicate query_id"));
        }
        digest_field(query, "query_definition_sha256", &context)?;
        digest_field(query, "analysis_code_sha256", &context)?;
        let source_output = optional_digest_field(query, "source_output_sha256", &context)?;
        let target_output = optional_digest_field(query, "target_output_sha256", &context)?;
        let equivalence = string_field(query, "equivalence", &context)?;
        match equivalence.as_str() {
            "SAME" => {
                if source_output.is_none() || source_output != target_output {
                    return Err(format!("{context}: SAME requires equal, non-null source and target output digests"));
                }
                summary.query_same_count += 1;
            }
            "DIFFERENT" => {
                if source_output.is_none() || target_output.is_none() || source_output == target_output {
                    return Err(format!("{context}: DIFFERENT requires distinct, non-null output digests"));
                }
                summary.query_different_count += 1;
            }
            "NOT_ESTIMABLE" => {
                if source_output.is_some() && target_output.is_some() {
                    return Err(format!("{context}: NOT_ESTIMABLE requires at least one missing output digest"));
                }
                summary.query_not_estimable_count += 1;
            }
            _ => return Err(format!("{context}: invalid equivalence")),
        }
    }

    count_summary(
        receipt.get("summary").ok_or_else(|| "summary missing".to_owned())?,
        &summary,
    )?;

    let claimed_disposition = string_field(&receipt, "disposition", "receipt")?;
    if !matches!(
        claimed_disposition.as_str(),
        "CORRESPONDENCE_VERIFIED" | "PARTIAL" | "AMBIGUOUS" | "NOT_ESTIMABLE" | "REJECTED"
    ) {
        return Err("receipt.disposition invalid".to_owned());
    }

    // V0.1 does not authenticate evidence bytes, lineage receipts, signer authority,
    // expert adjudication or physical registration anchors. Even a self-declared
    // VERIFIED mapping and a matching hash string are not a trust root.
    let recomputed_disposition = "NOT_ESTIMABLE";
    let engine_present = engine_digest.is_some();
    let verifier_present = verifier_digest.is_some();

    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "receipt_id": receipt_id,
        "status": "COUNTERS_RECOMPUTED",
        "source_report_sha256": sha256_hex(receipt_bytes),
        "source_manifest_sha256": source_digest,
        "target_manifest_sha256": target_digest,
        "claimed_disposition": claimed_disposition,
        "recomputed_disposition": recomputed_disposition,
        "trusted_promotion": false,
        "support_status_claim_counts": {
            "claimed_verified_mapping_count": summary.verified_count,
            "ambiguous_mapping_count": summary.ambiguous_count,
            "not_estimable_mapping_count": summary.not_estimable_count,
            "rejected_mapping_count": summary.rejected_count
        },
        "trust_boundaries": {
            "evidence_artifact_bytes": "NOT_AUTHENTICATED",
            "edit_history_and_lineage": "NOT_AUTHENTICATED",
            "expert_adjudication": "NOT_AUTHENTICATED",
            "physical_registration_residuals": "NOT_RECOMPUTED_FROM_ANCHORS",
            "signer_identity_and_revocation": "NOT_AUTHENTICATED",
            "correspondence_engine_digest_present": engine_present,
            "verifier_binary_digest_present": verifier_present
        },
        "registration_claim": registration_claim,
        "recomputed_summary": {
            "mapping_count": summary.mapping_count,
            "ambiguous_count": summary.ambiguous_count,
            "not_estimable_count": summary.not_estimable_count,
            "rejected_count": summary.rejected_count,
            "unmapped_source_object_count": summary.unmapped_source_object_count,
            "one_to_many_count": summary.one_to_many_count,
            "many_to_one_count": summary.many_to_one_count,
            "many_to_many_count": summary.many_to_many_count,
            "query_same_count": summary.query_same_count,
            "query_different_count": summary.query_different_count,
            "query_not_estimable_count": summary.query_not_estimable_count
        }
    }))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!(
            "usage: {} <receipt.json> <source-world.json> <target-world.json>",
            args.first().map(String::as_str).unwrap_or("connectome-world-correspondence-verifier")
        );
        process::exit(64);
    }
    let receipt = fs::read(&args[1]).unwrap_or_else(|error| {
        eprintln!("cannot read receipt: {error}");
        process::exit(66);
    });
    let source = fs::read(&args[2]).unwrap_or_else(|error| {
        eprintln!("cannot read source manifest: {error}");
        process::exit(66);
    });
    let target = fs::read(&args[3]).unwrap_or_else(|error| {
        eprintln!("cannot read target manifest: {error}");
        process::exit(66);
    });

    match verify_receipt(&receipt, &source, &target) {
        Ok(result) => println!("{}", serde_json::to_string(&result).expect("serialize verifier result")),
        Err(error) => {
            eprintln!("REJECTED: {error}");
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RECEIPT: &str = include_str!(
        "../../../research/connectomics/fixtures/connectome-correspondence-split-merge-ambiguity-known-answer-v0.1.json"
    );
    const SOURCE: &str = include_str!(
        "../../../research/connectomics/fixtures/connectome-correspondence-source-world-v0.1.json"
    );
    const TARGET: &str = include_str!(
        "../../../research/connectomics/fixtures/connectome-correspondence-target-world-v0.1.json"
    );

    fn verify(receipt: &str, source: &str, target: &str) -> Result<Value, String> {
        verify_receipt(receipt.as_bytes(), source.as_bytes(), target.as_bytes())
    }

    #[test]
    fn recomputes_split_merge_and_ambiguity_counters() {
        let output = verify(RECEIPT, SOURCE, TARGET).expect("valid synthetic receipt");
        let counts = &output["recomputed_summary"];
        assert_eq!(counts["mapping_count"], 8);
        assert_eq!(output["support_status_claim_counts"]["claimed_verified_mapping_count"], 4);
        assert_eq!(counts["ambiguous_count"], 2);
        assert_eq!(counts["not_estimable_count"], 2);
        assert_eq!(counts["unmapped_source_object_count"], 1);
        assert_eq!(counts["one_to_many_count"], 1);
        assert_eq!(counts["many_to_one_count"], 1);
        assert_eq!(counts["query_same_count"], 1);
        assert_eq!(counts["query_different_count"], 1);
        assert_eq!(counts["query_not_estimable_count"], 1);
    }

    #[test]
    fn refuses_to_promote_claimed_verified_mappings_without_evidence_authentication() {
        let output = verify(RECEIPT, SOURCE, TARGET).expect("synthetic structure is valid");
        assert_eq!(output["status"], "COUNTERS_RECOMPUTED");
        assert_eq!(output["trusted_promotion"], false);
        assert_eq!(output["recomputed_disposition"], "NOT_ESTIMABLE");
        assert_eq!(output["trust_boundaries"]["evidence_artifact_bytes"], "NOT_AUTHENTICATED");
    }

    #[test]
    fn rejects_source_manifest_byte_mutation() {
        let mut source = SOURCE.to_owned();
        source.push(' ');
        assert!(verify(RECEIPT, &source, TARGET).is_err());
    }

    #[test]
    fn rejects_cardinality_label_mismatch() {
        let mut receipt: Value = serde_json::from_str(RECEIPT).expect("fixture JSON");
        receipt["mappings"][0]["cardinality"] = Value::String("ONE_TO_ONE".to_owned());
        let bytes = serde_json::to_string(&receipt).expect("serialize altered receipt");
        assert!(verify(&bytes, SOURCE, TARGET).is_err());
    }

    #[test]
    fn rejects_unknown_source_or_target_object() {
        let mut receipt: Value = serde_json::from_str(RECEIPT).expect("fixture JSON");
        receipt["mappings"][0]["target_object_ids"][0] =
            Value::String("not-in-target-world".to_owned());
        let bytes = serde_json::to_string(&receipt).expect("serialize altered receipt");
        assert!(verify(&bytes, SOURCE, TARGET).is_err());
    }

    #[test]
    fn rejects_query_output_equivalence_contradiction() {
        let mut receipt: Value = serde_json::from_str(RECEIPT).expect("fixture JSON");
        receipt["query_comparisons"][0]["equivalence"] = Value::String("DIFFERENT".to_owned());
        let bytes = serde_json::to_string(&receipt).expect("serialize altered receipt");
        assert!(verify(&bytes, SOURCE, TARGET).is_err());
    }

    #[test]
    fn rejects_non_affine_registration_matrix() {
        let mut receipt: Value = serde_json::from_str(RECEIPT).expect("fixture JSON");
        receipt["registration"]["transform_source_to_target_4x4"][3][0] = Value::from(0.5);
        let bytes = serde_json::to_string(&receipt).expect("serialize altered receipt");
        assert!(verify(&bytes, SOURCE, TARGET).is_err());
    }

    #[test]
    fn rejects_duplicate_mapping_ids() {
        let mut receipt: Value = serde_json::from_str(RECEIPT).expect("fixture JSON");
        let first_mapping_id = receipt["mappings"][0]["mapping_id"].clone();
        receipt["mappings"][1]["mapping_id"] = first_mapping_id;
        let bytes = serde_json::to_string(&receipt).expect("serialize altered receipt");
        assert!(verify(&bytes, SOURCE, TARGET).is_err());
    }

    #[test]
    fn rejects_missing_source_inventory_coverage() {
        let mut receipt: Value = serde_json::from_str(RECEIPT).expect("fixture JSON");
        receipt["mappings"].as_array_mut().expect("mappings").remove(7);
        receipt["summary"]["mapping_count"] = Value::from(7);
        receipt["summary"]["claimed_verified_count"] = Value::from(3);
        let bytes = serde_json::to_string(&receipt).expect("serialize altered receipt");
        assert!(verify(&bytes, SOURCE, TARGET).is_err());
    }

    #[test]
    fn rejects_inconsistent_alternative_set() {
        let mut receipt: Value = serde_json::from_str(RECEIPT).expect("fixture JSON");
        receipt["mappings"][4]["alternative_set_id"] = Value::Null;
        let bytes = serde_json::to_string(&receipt).expect("serialize altered receipt");
        assert!(verify(&bytes, SOURCE, TARGET).is_err());
    }

    #[test]
    fn rejects_corrupted_artifact_manifest_digest() {
        let mut receipt: Value = serde_json::from_str(RECEIPT).expect("fixture JSON");
        receipt["provenance"]["source_manifest_sha256"] =
            Value::String("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned());
        let bytes = serde_json::to_string(&receipt).expect("serialize altered receipt");
        assert!(verify(&bytes, SOURCE, TARGET).is_err());
    }
    #[test]
    fn rejects_repeated_source_object_without_explicit_alternative_set() {
        let mut receipt: Value = serde_json::from_str(RECEIPT).expect("fixture JSON");
        let mut duplicate = receipt["mappings"][3].clone();
        duplicate["mapping_id"] = Value::String("map-neuron-2-without-alt-set".to_owned());
        duplicate["target_object_ids"] = json!(["neuron-2-candidate-b"]);
        duplicate["alternative_set_id"] = Value::Null;
        duplicate["support_status"] = Value::String("NOT_ESTIMABLE".to_owned());
        receipt["mappings"].as_array_mut().expect("mappings").push(duplicate);
        receipt["summary"]["mapping_count"] = Value::from(9);
        receipt["summary"]["not_estimable_count"] = Value::from(3);
        let bytes = serde_json::to_string(&receipt).expect("serialize altered receipt");
        assert!(verify(&bytes, SOURCE, TARGET).is_err());
    }

    #[test]
    fn rejects_alternative_set_with_overlapping_target_objects() {
        let mut receipt: Value = serde_json::from_str(RECEIPT).expect("fixture JSON");
        receipt["mappings"][4]["target_object_ids"] =
            json!(["neuron-2-candidate-a"]);
        let bytes = serde_json::to_string(&receipt).expect("serialize altered receipt");
        assert!(verify(&bytes, SOURCE, TARGET).is_err());
    }

    #[test]
    fn rejects_singular_registration_transform() {
        let mut receipt: Value = serde_json::from_str(RECEIPT).expect("fixture JSON");
        receipt["registration"]["transform_source_to_target_4x4"] =
            json!([[1,0,0,20],[0,0,0,-10],[0,0,1,5],[0,0,0,1]]);
        let bytes = serde_json::to_string(&receipt).expect("serialize altered receipt");
        assert!(verify(&bytes, SOURCE, TARGET).is_err());
    }

    #[test]
    fn rejects_relation_object_type_mismatch() {
        let mut receipt: Value = serde_json::from_str(RECEIPT).expect("fixture JSON");
        receipt["mappings"][2]["source_object_ids"] = json!(["seg-A"]);
        let bytes = serde_json::to_string(&receipt).expect("serialize altered receipt");
        assert!(verify(&bytes, SOURCE, TARGET).is_err());
    }

    #[test]
    fn rejects_not_estimable_query_with_two_output_digests() {
        let mut receipt: Value = serde_json::from_str(RECEIPT).expect("fixture JSON");
        receipt["query_comparisons"][2]["source_output_sha256"] =
            Value::String("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned());
        receipt["query_comparisons"][2]["target_output_sha256"] =
            Value::String("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned());
        let bytes = serde_json::to_string(&receipt).expect("serialize altered receipt");
        assert!(verify(&bytes, SOURCE, TARGET).is_err());
    }


    #[test]
    fn rejects_target_object_reused_across_mapping_rows() {
        let mut receipt: Value = serde_json::from_str(RECEIPT).expect("fixture JSON");
        receipt["mappings"][0]["target_object_ids"][1] = Value::String("seg-BC".to_owned());
        let bytes = serde_json::to_string(&receipt).expect("serialize altered receipt");
        assert!(verify(&bytes, SOURCE, TARGET).is_err());
    }


}
