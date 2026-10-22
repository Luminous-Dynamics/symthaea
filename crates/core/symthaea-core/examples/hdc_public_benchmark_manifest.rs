// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fail-closed contract checker for the public HDC/VSA dataset manifest.
//!
//! This validates provenance metadata and experiment declarations. It does not
//! download data, perform a full JSON Schema validation, execute classifiers,
//! or certify license rights. A receipt digest is not an authenticated signature.

use std::collections::BTreeSet;
use std::error::Error;

use serde_json::Value;

const MANIFEST_TEXT: &str =
    include_str!("../../../../docs/research/hdc-public-benchmark-manifest.json");
const SCHEMA_TEXT: &str =
    include_str!("../../../../docs/research/schemas/hdc-public-benchmark-manifest-v1.schema.json");

type CheckResult<T> = Result<T, String>;

const EXPECTED_DATASETS: [&str; 7] = [
    "cifar10",
    "emg_hand_gestures",
    "european_languages",
    "isolet",
    "mnist",
    "pamap2",
    "ucihar",
];

const REQUIRED_METHODS: [&str; 16] = [
    "centroid_prototype",
    "map_bipolar_prototype",
    "bsc_binary_prototype",
    "hrr_real_prototype",
    "onlinehd",
    "adapthd",
    "disthd",
    "sparsehd",
    "svm_raw",
    "lda_raw",
    "ridge_raw",
    "rvfl_raw",
    "svm_encoded",
    "lda_encoded",
    "ridge_encoded",
    "rvfl_encoded",
];

fn member<'a>(value: &'a Value, key: &str, path: &str) -> CheckResult<&'a Value> {
    value
        .get(key)
        .ok_or_else(|| format!("missing required field: {path}.{key}"))
}

fn as_str<'a>(value: &'a Value, path: &str) -> CheckResult<&'a str> {
    value
        .as_str()
        .ok_or_else(|| format!("expected string at {path}"))
}

fn as_array<'a>(value: &'a Value, path: &str) -> CheckResult<&'a [Value]> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| format!("expected array at {path}"))
}

fn string_field<'a>(value: &'a Value, key: &str, path: &str) -> CheckResult<&'a str> {
    as_str(member(value, key, path)?, &format!("{path}.{key}"))
}

fn validate_document(manifest: &Value, schema: &Value) -> CheckResult<()> {
    let schema_uri = string_field(schema, "$schema", "schema")?;
    if schema_uri != "https://json-schema.org/draft/2020-12/schema" {
        return Err(format!("unsupported JSON Schema declaration: {schema_uri}"));
    }

    let schema_title = string_field(schema, "title", "schema")?;
    if schema_title != "Symthaea public HDC/VSA benchmark manifest v1" {
        return Err("schema title does not identify manifest v1".to_owned());
    }

    let schema_properties = member(schema, "properties", "schema")?;
    for key in [
        "schema_version",
        "benchmark_id",
        "status",
        "created_date",
        "reference_protocol",
        "policies",
        "metrics",
        "comparison_tracks",
        "methods",
        "datasets",
    ] {
        if schema_properties.get(key).is_none() {
            return Err(format!("JSON schema is missing root property: {key}"));
        }
    }

    if string_field(manifest, "schema_version", "manifest")?
        != "hdc-public-benchmark-manifest-v1"
    {
        return Err("unsupported manifest schema_version".to_owned());
    }
    if string_field(manifest, "benchmark_id", "manifest")?
        != "symthaea-hdc-vsa-public-suite"
    {
        return Err("unexpected benchmark_id".to_owned());
    }
    let manifest_status = string_field(manifest, "status", "manifest")?;
    if manifest_status != "metadata_only" {
        return Err(format!(
            "this PR only qualifies metadata: expected status=metadata_only, got {manifest_status}"
        ));
    }

    let reference = member(manifest, "reference_protocol", "manifest")?;
    if member(reference, "benchmark_dimensions", "reference_protocol")?.as_u64() != Some(10_000) {
        return Err("reference benchmark_dimensions must remain 10000".to_owned());
    }
    if member(reference, "repetitions", "reference_protocol")?.as_u64() != Some(5) {
        return Err("reference repetitions must remain 5 until protocol audit changes it".to_owned());
    }
    if member(reference, "reference_hdc_task_count", "reference_protocol")?.as_u64() != Some(7) {
        return Err("reference HDC task count must be seven".to_owned());
    }
    if member(reference, "reference_uci_task_count", "reference_protocol")?.as_u64()
        != Some(121)
    {
        return Err("reference UCI task count must be 121".to_owned());
    }
    let reference_url = string_field(reference, "paper_url", "reference_protocol")?;
    if !reference_url.starts_with("https://") {
        return Err("reference paper URL must use HTTPS".to_owned());
    }
    let code_url = string_field(reference, "reference_code_url", "reference_protocol")?;
    if !code_url.starts_with("https://") {
        return Err("reference code URL must use HTTPS".to_owned());
    }
    if string_field(reference, "reproduction_status", "reference_protocol")?
        == "reproduced"
    {
        return Err(
            "metadata-only manifest cannot claim the reference benchmark is reproduced".to_owned(),
        );
    }

    let datasets = as_array(member(manifest, "datasets", "manifest")?, "manifest.datasets")?;
    if datasets.len() != EXPECTED_DATASETS.len() {
        return Err(format!(
            "expected {} reference datasets; found {}",
            EXPECTED_DATASETS.len(),
            datasets.len()
        ));
    }

    let mut observed_ids = BTreeSet::new();
    for (index, dataset) in datasets.iter().enumerate() {
        let path = format!("manifest.datasets[{index}]");
        let id = string_field(dataset, "id", &path)?;
        if !observed_ids.insert(id.to_owned()) {
            return Err(format!("duplicate dataset id: {id}"));
        }
        if !EXPECTED_DATASETS.contains(&id) {
            return Err(format!("dataset is not part of the canonical seven-task suite: {id}"));
        }
        if string_field(dataset, "benchmark_group", &path)? != "hdc_application_suite" {
            return Err(format!("{id}: unexpected benchmark group"));
        }

        let sources = as_array(member(dataset, "sources", &path)?, &format!("{path}.sources"))?;
        if sources.is_empty() {
            return Err(format!("{id}: at least one source record is required"));
        }
        for (source_index, source) in sources.iter().enumerate() {
            let source_path = format!("{path}.sources[{source_index}]");
            for field in ["url", "license_evidence_url"] {
                let url = string_field(source, field, &source_path)?;
                if !url.starts_with("https://") {
                    return Err(format!("{source_path}.{field} must use HTTPS"));
                }
            }
            let license_status =
                string_field(source, "license_review_status", &source_path)?;
            if ![
                "upstream_metadata",
                "review_required",
                "reviewed_for_declared_use",
            ]
            .contains(&license_status)
            {
                return Err(format!("{source_path}: invalid license review status"));
            }
        }

        let artifact = member(dataset, "artifact", &path)?;
        if string_field(artifact, "status", &format!("{path}.artifact"))? != "not_materialized" {
            return Err(format!(
                "{id}: metadata-only PR must not claim downloaded/verified dataset artifacts"
            ));
        }
        if !member(artifact, "sha256", &format!("{path}.artifact"))?.is_null()
            || !member(artifact, "bytes", &format!("{path}.artifact"))?.is_null()
        {
            return Err(format!(
                "{id}: an unmaterialized artifact must have null SHA-256 and byte count"
            ));
        }
    }
    let expected_ids: BTreeSet<String> =
        EXPECTED_DATASETS.iter().map(|id| (*id).to_owned()).collect();
    if observed_ids != expected_ids {
        return Err("canonical dataset inventory mismatch".to_owned());
    }

    for (id, expected_representation) in [
        ("cifar10", "pretrained_two_layer_cnn_features"),
        ("mnist", "pretrained_two_layer_cnn_features"),
        ("european_languages", "character_trigram_counts"),
        ("isolet", "617_real_valued_features"),
        ("ucihar", "561_engineered_sensor_features"),
    ] {
        let dataset = datasets
            .iter()
            .find(|dataset| dataset.get("id").and_then(Value::as_str) == Some(id))
            .ok_or_else(|| format!("missing required dataset: {id}"))?;
        let representation = string_field(
            member(dataset, "reference_preprocessing", id)?,
            "representation",
            &format!("{id}.reference_preprocessing"),
        )?;
        if representation != expected_representation {
            return Err(format!(
                "{id}: expected reference representation {expected_representation}, got {representation}"
            ));
        }
    }

    let methods = as_array(member(manifest, "methods", "manifest")?, "manifest.methods")?;
    let mut method_ids = BTreeSet::new();
    for (index, method) in methods.iter().enumerate() {
        let method_id = string_field(method, "id", &format!("manifest.methods[{index}]"))?;
        if !method_ids.insert(method_id.to_owned()) {
            return Err(format!("duplicate method id: {method_id}"));
        }
        if string_field(method, "status", &format!("manifest.methods[{index}]"))? == "qualified" {
            return Err(format!(
                "metadata-only manifest cannot qualify unexecuted method: {method_id}"
            ));
        }
    }
    for required in REQUIRED_METHODS {
        if !method_ids.contains(required) {
            return Err(format!("required fair-comparison baseline missing: {required}"));
        }
    }

    let tracks = as_array(
        member(manifest, "comparison_tracks", "manifest")?,
        "manifest.comparison_tracks",
    )?;
    let mut track_ids = BTreeSet::new();
    for (index, track) in tracks.iter().enumerate() {
        let track_id = string_field(track, "id", &format!("manifest.comparison_tracks[{index}]"))?;
        if !track_ids.insert(track_id.to_owned()) {
            return Err(format!("duplicate comparison track: {track_id}"));
        }
    }
    for required in ["reference_defaults", "equal_budget_tuning", "resource_matched"] {
        if !track_ids.contains(required) {
            return Err(format!("required comparison track missing: {required}"));
        }
    }

    let metrics = as_array(member(manifest, "metrics", "manifest")?, "manifest.metrics")?;
    let metric_set: BTreeSet<&str> = metrics
        .iter()
        .filter_map(Value::as_str)
        .collect();
    for required in [
        "accuracy",
        "macro_f1",
        "per_class_recall",
        "train_wall_ms",
        "single_query_latency_p95_ms",
        "model_payload_bytes",
        "peak_rss_bytes",
    ] {
        if !metric_set.contains(required) {
            return Err(format!("required reporting metric missing: {required}"));
        }
    }

    Ok(())
}

fn parse_and_validate() -> CheckResult<(Value, Value)> {
    let manifest: Value = serde_json::from_str(MANIFEST_TEXT)
        .map_err(|error| format!("manifest is not valid JSON: {error}"))?;
    let schema: Value = serde_json::from_str(SCHEMA_TEXT)
        .map_err(|error| format!("schema is not valid JSON: {error}"))?;
    validate_document(&manifest, &schema)?;
    Ok((manifest, schema))
}

fn main() -> Result<(), Box<dyn Error>> {
    let source_sha =
        std::env::var("SYMTHAEA_SOURCE_SHA").unwrap_or_else(|_| "UNBOUND_LOCAL".to_owned());
    let (manifest, schema) = parse_and_validate().map_err(std::io::Error::other)?;
    let manifest_digest = blake3::hash(MANIFEST_TEXT.as_bytes()).to_hex();
    let schema_digest = blake3::hash(SCHEMA_TEXT.as_bytes()).to_hex();
    let dataset_count = as_array(&manifest["datasets"], "manifest.datasets")?.len();
    let method_count = as_array(&manifest["methods"], "manifest.methods")?.len();
    let track_count = as_array(&manifest["comparison_tracks"], "manifest.comparison_tracks")?.len();
    let status = string_field(&manifest, "status", "manifest")?;

    println!(
        "# hdc_public_benchmark_manifest_receipt=protocol:hdc-public-benchmark-manifest-v1,source_sha:{source_sha},status:{status},dataset_count:{dataset_count},method_count:{method_count},comparison_track_count:{track_count},manifest_blake3:{manifest_digest},schema_blake3:{schema_digest}"
    );
    // Read the schema URI from the parsed object so the compiler catches accidental
    // loss of the schema linkage even on local runs.
    println!(
        "# schema_uri={}",
        string_field(&schema, "$schema", "schema")?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Value, Value) {
        let manifest: Value =
            serde_json::from_str(MANIFEST_TEXT).expect("manifest JSON parses");
        let schema: Value = serde_json::from_str(SCHEMA_TEXT).expect("schema JSON parses");
        (manifest, schema)
    }

    #[test]
    fn metadata_manifest_passes_declared_contract() {
        let (manifest, schema) = fixture();
        validate_document(&manifest, &schema).expect("metadata contract should pass");
    }

    #[test]
    fn duplicate_dataset_ids_fail_closed() {
        let (mut manifest, schema) = fixture();
        let first_id = manifest["datasets"][0]["id"].clone();
        manifest["datasets"][1]["id"] = first_id;
        assert!(validate_document(&manifest, &schema).is_err());
    }

    #[test]
    fn missing_conventional_baseline_fails_closed() {
        let (mut manifest, schema) = fixture();
        let methods = manifest["methods"].as_array_mut().expect("methods array");
        methods.retain(|method| method["id"].as_str() != Some("svm_raw"));
        assert!(validate_document(&manifest, &schema).is_err());
    }

    #[test]
    fn metadata_only_manifest_cannot_claim_materialized_artifact() {
        let (mut manifest, schema) = fixture();
        manifest["datasets"][0]["artifact"]["status"] =
            Value::String("verified".to_owned());
        manifest["datasets"][0]["artifact"]["sha256"] =
            Value::String("a".repeat(64));
        manifest["datasets"][0]["artifact"]["bytes"] = Value::from(123_u64);
        assert!(validate_document(&manifest, &schema).is_err());
    }

    #[test]
    fn reference_preprocessing_mismatch_fails_closed() {
        let (mut manifest, schema) = fixture();
        manifest["datasets"][0]["reference_preprocessing"]["representation"] =
            Value::String("raw_rgb_pixels".to_owned());
        assert!(validate_document(&manifest, &schema).is_err());
    }

    #[test]
    fn insecure_source_url_fails_closed() {
        let (mut manifest, schema) = fixture();
        manifest["datasets"][0]["sources"][0]["url"] =
            Value::String("http://example.org/data".to_owned());
        assert!(validate_document(&manifest, &schema).is_err());
    }
}
