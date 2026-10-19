// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeMap;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use symthaea_culinary::culinary_evidence_bundle::{
    analyze_paired_outcomes_with_verified_bundle_v1,
    verify_culinary_evidence_bundle_v1,
};
use symthaea_culinary::culinary_ro_crate::export_culinary_evidence_bundle_ro_crate_v1;

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../fixtures/culinary-experiment-v1/synthetic-paired-cooking-study.json"
    ))
    .unwrap()
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

// The names/bytes created by this helper are synthetic test artifacts. They
// exercise byte verification and do not authenticate any source or physical run.
fn test_only_bundle() -> (Value, Vec<u8>, String, BTreeMap<String, Vec<u8>>) {
    let mut experiment = fixture();
    let protocol_bytes = b"test-only frozen protocol bytes\n".to_vec();
    let protocol_sha = sha256_hex(&protocol_bytes);
    experiment["protocol_sha256"] = Value::String(protocol_sha.clone());

    let mut artifacts = vec![json!({
        "artifact_id":"protocol-artifact-v1",
        "artifact_kind":"protocol",
        "path":"protocols/test-protocol-v1.md",
        "source_ref":"urn:test-evidence:protocol",
        "source_version":"protocol-v1",
        "sha256":protocol_sha,
        "byte_length":protocol_bytes.len()
    })];
    let mut payloads = BTreeMap::new();
    payloads.insert("protocol-artifact-v1".to_owned(), protocol_bytes);

    for block in experiment["blocks"].as_array_mut().unwrap() {
        for run in block["runs"].as_array_mut().unwrap() {
            let run_id = run["run_id"].as_str().unwrap().to_owned();
            let trace_id = run["process_trace_id"].as_str().unwrap().to_owned();
            let mut trace_payload_value: Value = serde_json::from_str(include_str!(
                "../fixtures/process-nutrient-binding-v1/synthetic-rice-pea-trace.json"
            )).unwrap();
            trace_payload_value["trace_id"] = Value::String(trace_id.clone());
            let trace_payload = serde_json::to_vec(&trace_payload_value).unwrap();
            artifacts.push(json!({
                "artifact_id":trace_id,
                "artifact_kind":"process_trace",
                "path":format!("traces/{run_id}.json"),
                "source_ref":format!("urn:test-evidence:trace:{run_id}"),
                "source_version":"trace-v1",
                "sha256":sha256_hex(&trace_payload),
                "byte_length":trace_payload.len()
            }));
            payloads.insert(trace_id.clone(), trace_payload);

            let started_at = run["started_at"].as_str().unwrap().to_owned();
            for outcome in run["outcomes"].as_array_mut().unwrap() {
                let outcome_id = outcome["outcome_id"].as_str().unwrap().to_owned();
                let artifact_id = format!("measurement-{run_id}-{outcome_id}");
                let source_ref = format!("urn:test-evidence:measurement:{run_id}:{outcome_id}");
                let source_version = "measurement-fixture-v1";
                outcome["status"] = Value::String("observed".into());
                outcome["evidence"]["source_ref"] = Value::String(source_ref.clone());
                outcome["evidence"]["source_version"] = Value::String(source_version.into());
                outcome["evidence"]["known_at"] = Value::String(started_at.clone());
                outcome["evidence"]["evidence_status"] = Value::String("qualified".into());
                let payload_value = json!({
                    "schema_version":"1.0.0",
                    "run_id":run_id,
                    "outcome_id":outcome_id,
                    "status":outcome["status"].clone(),
                    "value":outcome["value"].clone(),
                    "lower":outcome["lower"].clone(),
                    "upper":outcome["upper"].clone(),
                    "unit":outcome["unit"].clone(),
                    "method":outcome["method"].clone(),
                    "unknown_reason":outcome["unknown_reason"].clone()
                });
                let payload = serde_json::to_vec(&payload_value).unwrap();
                artifacts.push(json!({
                    "artifact_id":artifact_id,
                    "artifact_kind":"measurement_record",
                    "path":format!("measurements/{run_id}/{outcome_id}.json"),
                    "source_ref":source_ref,
                    "source_version":source_version,
                    "sha256":sha256_hex(&payload),
                    "byte_length":payload.len()
                }));
                payloads.insert(artifact_id, payload);
            }
        }
    }

    let manifest = json!({
        "schema_version":"1.0.0",
        "bundle_id":"test-only-bundle-v1",
        "experiment_id":experiment["experiment_id"].clone(),
        "protocol_sha256":protocol_sha,
        "artifacts":artifacts
    });
    let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
    let manifest_sha = sha256_hex(&manifest_bytes);
    (experiment, manifest_bytes, manifest_sha, payloads)
}

fn verify_bundle(
    experiment: &Value,
    manifest_bytes: &[u8],
    manifest_sha: &str,
    payloads: &BTreeMap<String, Vec<u8>>,
) -> symthaea_culinary::culinary_evidence_bundle::VerifiedCulinaryEvidenceBundleV1 {
    verify_culinary_evidence_bundle_v1(experiment, manifest_bytes, manifest_sha, payloads).unwrap()
}

#[test]
fn valid_test_only_bundle_verifies_all_artifact_bytes_and_bindings() {
    let (experiment, manifest_bytes, manifest_sha, payloads) = test_only_bundle();
    let verified = verify_bundle(&experiment, &manifest_bytes, &manifest_sha, &payloads);
    assert_eq!(verified.experiment_id(), experiment["experiment_id"].as_str().unwrap());
    assert_eq!(verified.protocol_sha256(), experiment["protocol_sha256"].as_str().unwrap());
    assert_eq!(verified.artifact_count(), 17);
    assert_eq!(verified.process_trace_count(), 4);
    assert_eq!(verified.outcome_source_count(), 12);
    assert_eq!(verified.manifest_sha256(), manifest_sha);
}

#[test]
fn payload_byte_tampering_is_rejected() {
    let (experiment, manifest_bytes, manifest_sha, mut payloads) = test_only_bundle();
    payloads.get_mut("protocol-artifact-v1").unwrap().push(b'X');
    let error = verify_culinary_evidence_bundle_v1(
        &experiment,&manifest_bytes,&manifest_sha,&payloads
    ).unwrap_err();
    assert!(error.iter().any(|item| item.contains("byte_length mismatch")));
    assert!(error.iter().any(|item| item.contains("SHA-256 mismatch")));
}

#[test]
fn manifest_bytes_must_match_separately_pinned_digest() {
    let (experiment, mut manifest_bytes, manifest_sha, payloads) = test_only_bundle();
    manifest_bytes.push(b' ');
    let error = verify_culinary_evidence_bundle_v1(
        &experiment,&manifest_bytes,&manifest_sha,&payloads
    ).unwrap_err();
    assert!(error.iter().any(|item| item.contains("separately pinned manifest SHA-256")));
}

#[test]
fn protocol_bytes_must_match_preregistered_protocol_digest() {
    let (mut experiment, manifest_bytes, manifest_sha, payloads) = test_only_bundle();
    experiment["protocol_sha256"] = Value::String("b".repeat(64));
    let error = verify_culinary_evidence_bundle_v1(
        &experiment,&manifest_bytes,&manifest_sha,&payloads
    ).unwrap_err();
    assert!(error.iter().any(|item| item.contains("preregistered protocol digest")));
}

#[test]
fn missing_artifact_payload_fails_closed() {
    let (experiment, manifest_bytes, manifest_sha, mut payloads) = test_only_bundle();
    payloads.remove("protocol-artifact-v1");
    let error=verify_culinary_evidence_bundle_v1(&experiment,&manifest_bytes,&manifest_sha,&payloads).unwrap_err();
    assert!(error.iter().any(|item| item.contains("payload bytes missing")));
}

#[test]
fn undeclared_extra_payload_fails_closed() {
    let (experiment, manifest_bytes, manifest_sha, mut payloads) = test_only_bundle();
    payloads.insert("not-declared".into(), b"extra".to_vec());
    let error=verify_culinary_evidence_bundle_v1(&experiment,&manifest_bytes,&manifest_sha,&payloads).unwrap_err();
    assert!(error.iter().any(|item| item.contains("undeclared payload bytes")));
}

#[test]
fn unsafe_artifact_paths_are_rejected() {
    let (experiment, manifest_bytes, _, payloads) = test_only_bundle();
    let mut manifest: Value = serde_json::from_slice(&manifest_bytes).unwrap();
    manifest["artifacts"][0]["path"] = Value::String("../outside/protocol.md".into());
    let bytes = serde_json::to_vec(&manifest).unwrap();
    let digest = sha256_hex(&bytes);
    let error=verify_culinary_evidence_bundle_v1(&experiment,&bytes,&digest,&payloads).unwrap_err();
    assert!(error.iter().any(|item| item.contains("safe normalized relative path")));
}

#[test]
fn every_process_trace_id_must_resolve_to_process_trace_artifact() {
    let (mut experiment, manifest_bytes, manifest_sha, payloads) = test_only_bundle();
    experiment["blocks"][0]["runs"][0]["process_trace_id"] = Value::String("missing-trace".into());
    let error=verify_culinary_evidence_bundle_v1(&experiment,&manifest_bytes,&manifest_sha,&payloads).unwrap_err();
    assert!(error.iter().any(|item| item.contains("must resolve to a declared process_trace")));
}

#[test]
fn outcome_evidence_version_must_match_manifest_artifact() {
    let (mut experiment, manifest_bytes, manifest_sha, payloads) = test_only_bundle();
    experiment["blocks"][0]["runs"][0]["outcomes"][0]["evidence"]["source_version"] =
        Value::String("version-drift".into());
    let error=verify_culinary_evidence_bundle_v1(&experiment,&manifest_bytes,&manifest_sha,&payloads).unwrap_err();
    assert!(error.iter().any(|item| item.contains("source_version does not match")));
}

#[test]
fn changed_experiment_cannot_reuse_verified_bundle_capability() {
    let (experiment, manifest_bytes, manifest_sha, payloads) = test_only_bundle();
    let verified = verify_bundle(&experiment,&manifest_bytes,&manifest_sha,&payloads);
    let mut altered = experiment.clone();
    altered["blocks"][0]["runs"][0]["outcomes"][0]["value"] = json!(999.0);
    let error = analyze_paired_outcomes_with_verified_bundle_v1(&altered,&verified).unwrap_err();
    assert!(error.iter().any(|item| item.contains("contents changed after evidence bundle verification")));
}

#[test]
fn verified_bundle_analyzes_only_bound_experiment_value() {
    let (experiment, manifest_bytes, manifest_sha, payloads) = test_only_bundle();
    let verified=verify_bundle(&experiment,&manifest_bytes,&manifest_sha,&payloads);
    let analysis=analyze_paired_outcomes_with_verified_bundle_v1(&experiment,&verified).unwrap();
    assert_eq!(analysis.block_count,2);
    assert_eq!(analysis.summaries.len(),3);
}

#[test]
fn duplicate_manifest_fields_are_rejected_by_typed_deserialization() {
    let (experiment, manifest_bytes, _, payloads)=test_only_bundle();
    let text=String::from_utf8(manifest_bytes).unwrap();
    let injected=text.replacen(
        "{",
        "{\"bundle_id\":\"first\",\"bundle_id\":\"second\",",
        1
    );
    let bytes=injected.into_bytes();
    let digest=sha256_hex(&bytes);
    let error=verify_culinary_evidence_bundle_v1(&experiment,&bytes,&digest,&payloads).unwrap_err();
    assert!(error.iter().any(|item|item.contains("duplicate field")));
}

#[test]
fn qualified_evidence_must_not_point_to_synthetic_source_uri() {
    let (experiment, mut manifest_bytes, _, payloads)=test_only_bundle();
    let mut manifest:Value=serde_json::from_slice(&manifest_bytes).unwrap();
    manifest["artifacts"][2]["source_ref"]=Value::String("urn:synthetic:qualified-spoof".into());
    manifest_bytes=serde_json::to_vec(&manifest).unwrap();
    let digest=sha256_hex(&manifest_bytes);
    let mut altered=experiment;
    altered["blocks"][0]["runs"][0]["outcomes"][0]["evidence"]["source_ref"]=
        Value::String("urn:synthetic:qualified-spoof".into());
    let error=verify_culinary_evidence_bundle_v1(&altered,&manifest_bytes,&digest,&payloads).unwrap_err();
    assert!(error.iter().any(|item|item.contains("cannot be qualified from a synthetic source")));
}

#[test]
fn duplicate_artifact_paths_are_rejected() {
    let (experiment, manifest_bytes, _, payloads)=test_only_bundle();
    let mut manifest:Value=serde_json::from_slice(&manifest_bytes).unwrap();
    let path=manifest["artifacts"][1]["path"].clone();
    manifest["artifacts"][2]["path"]=path;
    let bytes=serde_json::to_vec(&manifest).unwrap();
    let digest=sha256_hex(&bytes);
    let error=verify_culinary_evidence_bundle_v1(&experiment,&bytes,&digest,&payloads).unwrap_err();
    assert!(error.iter().any(|item|item.contains("duplicate artifact path")));
}


#[test]
fn unreferenced_manifest_artifact_is_rejected() {
    let (experiment, manifest_bytes, _, mut payloads) = test_only_bundle();
    let mut manifest: Value = serde_json::from_slice(&manifest_bytes).unwrap();
    let extra_payload = b"declared but unreachable artifact".to_vec();
    let extra_id = "unreferenced-artifact";
    manifest["artifacts"].as_array_mut().unwrap().push(json!({
        "artifact_id":extra_id,
        "artifact_kind":"other",
        "path":"extra/unreferenced.bin",
        "source_ref":"urn:test-evidence:unreferenced",
        "source_version":"v1",
        "sha256":sha256_hex(&extra_payload),
        "byte_length":extra_payload.len()
    }));
    payloads.insert(extra_id.into(), extra_payload);
    let bytes = serde_json::to_vec(&manifest).unwrap();
    let digest = sha256_hex(&bytes);
    let error = verify_culinary_evidence_bundle_v1(&experiment,&bytes,&digest,&payloads).unwrap_err();
    assert!(error.iter().any(|item| item.contains("is not referenced")));
}


#[test]
fn measurement_artifact_bytes_must_match_consumed_outcome_values() {
    let (experiment, manifest_bytes, _, mut payloads) = test_only_bundle();
    let artifact_id = "measurement-synthetic-run-001-prepared_yield_g";
    let mut record: Value = serde_json::from_slice(payloads.get(artifact_id).unwrap()).unwrap();
    record["value"] = json!(777.0);
    let altered_payload = serde_json::to_vec(&record).unwrap();
    payloads.insert(artifact_id.into(), altered_payload.clone());

    let mut manifest: Value = serde_json::from_slice(&manifest_bytes).unwrap();
    let item = manifest["artifacts"].as_array_mut().unwrap().iter_mut()
        .find(|artifact| artifact["artifact_id"] == artifact_id)
        .unwrap();
    item["sha256"] = Value::String(sha256_hex(&altered_payload));
    item["byte_length"] = json!(altered_payload.len());
    let bytes = serde_json::to_vec(&manifest).unwrap();
    let digest = sha256_hex(&bytes);
    let error = verify_culinary_evidence_bundle_v1(&experiment,&bytes,&digest,&payloads).unwrap_err();
    assert!(error.iter().any(|item| item.contains("does not exactly match experiment outcome")));
}

#[test]
fn process_trace_payload_identity_must_match_manifest_artifact_id() {
    let (experiment, manifest_bytes, _, mut payloads) = test_only_bundle();
    let artifact_id = "synthetic-process-trace-001-control";
    let mut trace: Value = serde_json::from_slice(payloads.get(artifact_id).unwrap()).unwrap();
    trace["trace_id"] = Value::String("different-trace-id".into());
    let altered_payload = serde_json::to_vec(&trace).unwrap();
    payloads.insert(artifact_id.into(), altered_payload.clone());

    let mut manifest: Value = serde_json::from_slice(&manifest_bytes).unwrap();
    let item = manifest["artifacts"].as_array_mut().unwrap().iter_mut()
        .find(|artifact| artifact["artifact_id"] == artifact_id)
        .unwrap();
    item["sha256"] = Value::String(sha256_hex(&altered_payload));
    item["byte_length"] = json!(altered_payload.len());
    let bytes = serde_json::to_vec(&manifest).unwrap();
    let digest = sha256_hex(&bytes);
    let error = verify_culinary_evidence_bundle_v1(&experiment,&bytes,&digest,&payloads).unwrap_err();
    assert!(error.iter().any(|item| item.contains("internal trace_id does not match artifact_id")));
}


#[test]
fn duplicate_measurement_record_fields_are_rejected() {
    let (experiment, manifest_bytes, _, mut payloads) = test_only_bundle();
    let artifact_id = "measurement-synthetic-run-001-prepared_yield_g";
    let original = String::from_utf8(payloads.get(artifact_id).unwrap().clone()).unwrap();
    let injected = original.replacen("\"value\":246", "\"value\":246,\"value\":246", 1);
    assert_ne!(original, injected);
    let payload = injected.into_bytes();
    payloads.insert(artifact_id.into(), payload.clone());

    let mut manifest: Value = serde_json::from_slice(&manifest_bytes).unwrap();
    let item = manifest["artifacts"].as_array_mut().unwrap().iter_mut()
        .find(|artifact| artifact["artifact_id"] == artifact_id)
        .unwrap();
    item["sha256"] = Value::String(sha256_hex(&payload));
    item["byte_length"] = json!(payload.len());
    let bytes = serde_json::to_vec(&manifest).unwrap();
    let digest = sha256_hex(&bytes);
    let error = verify_culinary_evidence_bundle_v1(&experiment,&bytes,&digest,&payloads).unwrap_err();
    assert!(error.iter().any(|item| item.contains("duplicate field")));
}


#[test]
fn duplicate_artifact_ids_are_rejected() {
    let (experiment, manifest_bytes, _, payloads) = test_only_bundle();
    let mut manifest: Value = serde_json::from_slice(&manifest_bytes).unwrap();
    let duplicated_id = manifest["artifacts"][1]["artifact_id"].clone();
    manifest["artifacts"][2]["artifact_id"] = duplicated_id;
    let bytes = serde_json::to_vec(&manifest).unwrap();
    let digest = sha256_hex(&bytes);
    let error = verify_culinary_evidence_bundle_v1(&experiment,&bytes,&digest,&payloads).unwrap_err();
    assert!(error.iter().any(|item| item.contains("duplicate artifact_id")));
}

#[test]
fn duplicate_source_references_are_rejected() {
    let (experiment, manifest_bytes, _, payloads) = test_only_bundle();
    let mut manifest: Value = serde_json::from_slice(&manifest_bytes).unwrap();
    let source_ref = manifest["artifacts"][1]["source_ref"].clone();
    manifest["artifacts"][2]["source_ref"] = source_ref;
    let bytes = serde_json::to_vec(&manifest).unwrap();
    let digest = sha256_hex(&bytes);
    let error = verify_culinary_evidence_bundle_v1(&experiment,&bytes,&digest,&payloads).unwrap_err();
    assert!(error.iter().any(|item| item.contains("duplicate artifact source_ref")));
}

#[test]
fn unbound_outcome_source_reference_is_rejected() {
    let (mut experiment, manifest_bytes, manifest_sha, payloads) = test_only_bundle();
    experiment["blocks"][0]["runs"][0]["outcomes"][0]["evidence"]["source_ref"] =
        Value::String("urn:test-evidence:absent-source".into());
    let error = verify_culinary_evidence_bundle_v1(&experiment,&manifest_bytes,&manifest_sha,&payloads).unwrap_err();
    assert!(error.iter().any(|item| item.contains("has no manifest artifact")));
}


#[test]
fn ro_crate_export_preserves_exact_manifest_and_payload_bytes() {
    let (experiment, manifest_bytes, manifest_sha, payloads) = test_only_bundle();
    let exported = export_culinary_evidence_bundle_ro_crate_v1(
        &experiment, &manifest_bytes, &manifest_sha, &payloads
    ).unwrap();

    assert_eq!(exported.manifest_sha256(), manifest_sha);
    assert_eq!(exported.files().len(), payloads.len() + 2);
    assert_eq!(
        exported.files().get("culinary-evidence-manifest.json").unwrap(),
        &manifest_bytes
    );

    let manifest: Value = serde_json::from_slice(&manifest_bytes).unwrap();
    for artifact in manifest["artifacts"].as_array().unwrap() {
        let id = artifact["artifact_id"].as_str().unwrap();
        let path = artifact["path"].as_str().unwrap();
        assert_eq!(exported.files().get(path).unwrap(), payloads.get(id).unwrap());
    }

    let parsed_metadata: Value = serde_json::from_slice(
        exported.files().get("ro-crate-metadata.json").unwrap()
    ).unwrap();
    assert_eq!(parsed_metadata, exported.metadata().clone());

    let graph = exported.metadata()["@graph"].as_array().unwrap();
    let root = graph.iter().find(|node| node["@id"] == "./").unwrap();
    assert_eq!(root["@type"], "Dataset");
    let parts = root["hasPart"].as_array().unwrap();
    assert_eq!(parts.len(), payloads.len() + 1);
    let referenced: std::collections::BTreeSet<String> = parts.iter()
        .map(|item| item["@id"].as_str().unwrap().to_owned()).collect();
    assert_eq!(referenced.len(), parts.len());
    assert!(referenced.contains("culinary-evidence-manifest.json"));
    for artifact in manifest["artifacts"].as_array().unwrap() {
        assert!(referenced.contains(artifact["path"].as_str().unwrap()));
    }

    let repeated = export_culinary_evidence_bundle_ro_crate_v1(
        &experiment, &manifest_bytes, &manifest_sha, &payloads
    ).unwrap();
    assert_eq!(exported.files(), repeated.files());
}

#[test]
fn ro_crate_export_rejects_payload_tampering_before_packaging() {
    let (experiment, manifest_bytes, manifest_sha, mut payloads) = test_only_bundle();
    let id = payloads.keys().next().unwrap().clone();
    payloads.get_mut(&id).unwrap().push(b'X');
    let error = export_culinary_evidence_bundle_ro_crate_v1(
        &experiment, &manifest_bytes, &manifest_sha, &payloads
    ).unwrap_err();
    assert!(!error.is_empty());
}

#[test]
fn ro_crate_export_rejects_reserved_metadata_path_collisions() {
    let (experiment, manifest_bytes, _, payloads) = test_only_bundle();
    let mut manifest: Value = serde_json::from_slice(&manifest_bytes).unwrap();
    manifest["artifacts"][0]["path"] = Value::String("ro-crate-metadata.json".into());
    let bytes = serde_json::to_vec(&manifest).unwrap();
    let digest = sha256_hex(&bytes);
    let error = export_culinary_evidence_bundle_ro_crate_v1(
        &experiment, &bytes, &digest, &payloads
    ).unwrap_err();
    assert!(error.iter().any(|item| item.contains("reserved file")));
}
