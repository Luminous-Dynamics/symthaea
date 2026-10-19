// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! RO-Crate 1.3 export for byte-verified Culinary evidence bundles.
//!
//! Export is deliberately downstream of the content-addressed verifier: no
//! RO-Crate is produced unless the manifest, payload bytes, process-trace IDs,
//! measurement records and experiment bindings pass the existing verifier.
//! This adapter preserves exact source bytes and adds machine-readable metadata;
//! it does not add signatures, custody attestations, or claims of scientific truth.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::culinary_evidence_bundle::verify_culinary_evidence_bundle_v1;

const CRATE_METADATA_PATH: &str = "ro-crate-metadata.json";
const EVIDENCE_MANIFEST_PATH: &str = "culinary-evidence-manifest.json";
const RO_CRATE_PROFILE: &str = "https://w3id.org/ro/crate/1.3";
const RO_CRATE_CONTEXT: &str = "https://w3id.org/ro/crate/1.3/context";
const CULINARY_CONTEXT: &str = "https://w3id.org/luminous-dynamics/culinary-evidence#";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExportManifestV1 {
    schema_version: String,
    bundle_id: String,
    experiment_id: String,
    protocol_sha256: String,
    artifacts: Vec<ExportArtifactV1>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExportArtifactV1 {
    artifact_id: String,
    artifact_kind: String,
    path: String,
    source_ref: String,
    source_version: String,
    sha256: String,
    byte_length: u64,
}

/// A deterministic in-memory RO-Crate package. File keys are normalized crate
/// paths; values are the exact original manifest and payload bytes, plus the
/// generated `ro-crate-metadata.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct CulinaryRoCrateV1 {
    metadata: Value,
    files: BTreeMap<String, Vec<u8>>,
    manifest_sha256: String,
}

impl CulinaryRoCrateV1 {
    /// RO-Crate 1.3 JSON-LD metadata document.
    pub fn metadata(&self) -> &Value {
        &self.metadata
    }

    /// Complete crate file map, including exact source bytes and metadata JSON.
    pub fn files(&self) -> &BTreeMap<String, Vec<u8>> {
        &self.files
    }

    /// SHA-256 of the exact original evidence-manifest byte sequence.
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }
}

/// Verify a Culinary evidence bundle and export it as an RO-Crate 1.3 package.
///
/// The returned file map contains every input payload at its manifest-declared
/// relative path, the unmodified manifest bytes at
/// `culinary-evidence-manifest.json`, and generated RO-Crate JSON-LD metadata.
/// The function fails closed if a payload would shadow either reserved filename.
///
/// SHA-256 is an integrity binding to a separately supplied expected digest. It
/// does not authenticate who captured an artifact, prove custody, establish that
/// a physical process followed its recorded trace, or validate scientific truth.
pub fn export_culinary_evidence_bundle_ro_crate_v1(
    experiment: &Value,
    manifest_bytes: &[u8],
    expected_manifest_sha256: &str,
    artifact_bytes: &BTreeMap<String, Vec<u8>>,
) -> Result<CulinaryRoCrateV1, Vec<String>> {
    let verified = verify_culinary_evidence_bundle_v1(
        experiment,
        manifest_bytes,
        expected_manifest_sha256,
        artifact_bytes,
    )?;

    // The verifier has already rejected duplicate/unknown manifest keys and
    // verified all byte bindings. This typed read supplies fields for the export.
    let manifest: ExportManifestV1 = serde_json::from_slice(manifest_bytes).map_err(|error| {
        vec![format!("verified evidence manifest could not be decoded for export: {error}")]
    })?;
    if manifest.schema_version != "1.0.0" {
        return Err(vec!["unsupported Culinary evidence manifest version for RO-Crate export".into()]);
    }

    let reserved: [&str; 2] = [CRATE_METADATA_PATH, EVIDENCE_MANIFEST_PATH];
    if let Some(artifact) = manifest.artifacts.iter().find(|artifact| {
        reserved.iter().any(|reserved_path| artifact.path.eq_ignore_ascii_case(reserved_path))
    }) {
        return Err(vec![format!(
            "artifact path {:?} collides with an RO-Crate reserved file",
            artifact.path
        )]);
    }

    let protocol_path = manifest.artifacts.iter()
        .find(|artifact| artifact.artifact_kind == "protocol")
        .map(|artifact| artifact.path.as_str())
        .ok_or_else(|| vec!["verified manifest lost its protocol artifact before export".to_owned()])?;

    let mut files = BTreeMap::new();
    let mut part_ids = vec![json!({ "@id": EVIDENCE_MANIFEST_PATH })];
    let mut graph = vec![
        json!({
            "@id": CRATE_METADATA_PATH,
            "@type": "CreativeWork",
            "about": { "@id": "./" },
            "conformsTo": { "@id": RO_CRATE_PROFILE }
        }),
        json!({
            "@id": "./",
            "@type": "Dataset",
            "name": format!("Culinary experiment evidence {}", manifest.experiment_id),
            "identifier": manifest.bundle_id,
            "hasPart": part_ids,
            "isBasedOn": { "@id": protocol_path },
            "manifestSha256": verified.manifest_sha256(),
            "experimentId": manifest.experiment_id,
            "protocolSha256": manifest.protocol_sha256,
            "verificationProfile": "culinary-evidence-bundle-v1-byte-integrity-only"
        }),
    ];

    files.insert(EVIDENCE_MANIFEST_PATH.to_owned(), manifest_bytes.to_vec());
    graph.push(json!({
        "@id": EVIDENCE_MANIFEST_PATH,
        "@type": "File",
        "name": "Original Culinary evidence bundle manifest",
        "encodingFormat": "application/json",
        "contentSize": format!("{} bytes", manifest_bytes.len()),
        "byteLength": manifest_bytes.len(),
        "sha256": verified.manifest_sha256(),
        "artifactKind": "evidence_manifest_v1",
        "isPartOf": { "@id": "./" }
    }));

    for artifact in &manifest.artifacts {
        let payload = artifact_bytes.get(&artifact.artifact_id).ok_or_else(|| {
            vec![format!("verified payload {:?} unexpectedly absent during export", artifact.artifact_id)]
        })?;
        // Re-check export mapping invariants so a future refactor cannot silently
        // decouple the packaged bytes from their declared content identifiers.
        if payload.len() as u64 != artifact.byte_length {
            return Err(vec![format!("payload byte length changed during export for {:?}", artifact.artifact_id)]);
        }

        let media_type = media_type_for_path(&artifact.path);
        files.insert(artifact.path.clone(), payload.clone());
        part_ids.push(json!({ "@id": artifact.path }));
        graph.push(json!({
            "@id": artifact.path,
            "@type": "File",
            "name": artifact.artifact_id,
            "identifier": artifact.source_ref,
            "encodingFormat": media_type,
            "contentSize": format!("{} bytes", artifact.byte_length),
            "byteLength": artifact.byte_length,
            "sha256": artifact.sha256,
            "artifactId": artifact.artifact_id,
            "artifactKind": artifact.artifact_kind,
            "sourceRef": artifact.source_ref,
            "sourceVersion": artifact.source_version,
            "isPartOf": { "@id": "./" }
        }));
    }

    // The root's hasPart was initially created before the artifacts were walked.
    // Replace its value now so every source artifact and the original manifest
    // are reachable from the dataset node.
    graph[1]["hasPart"] = Value::Array(part_ids);

    let metadata = json!({
        "@context": [
            RO_CRATE_CONTEXT,
            {
                "culinary": CULINARY_CONTEXT,
                "artifactId": "culinary:artifactId",
                "artifactKind": "culinary:artifactKind",
                "sourceRef": "culinary:sourceRef",
                "sourceVersion": "culinary:sourceVersion",
                "byteLength": "culinary:byteLength",
                "manifestSha256": "culinary:manifestSha256",
                "experimentId": "culinary:experimentId",
                "protocolSha256": "culinary:protocolSha256",
                "verificationProfile": "culinary:verificationProfile",
                "sha256": "https://schema.org/sha256"
            }
        ],
        "@graph": graph
    });
    let mut metadata_bytes = serde_json::to_vec_pretty(&metadata)
        .map_err(|error| vec![format!("could not encode RO-Crate JSON-LD: {error}")])?;
    metadata_bytes.push(b'\n');
    files.insert(CRATE_METADATA_PATH.to_owned(), metadata_bytes);

    Ok(CulinaryRoCrateV1 {
        metadata,
        files,
        manifest_sha256: verified.manifest_sha256().to_owned(),
    })
}

fn media_type_for_path(path: &str) -> &'static str {
    match path.rsplit('/').next().unwrap_or(path).rsplit_once('.') {
        Some((_, extension)) if extension.eq_ignore_ascii_case("json") => "application/json",
        Some((_, extension)) if extension.eq_ignore_ascii_case("md") => "text/markdown",
        Some((_, extension)) if extension.eq_ignore_ascii_case("txt") => "text/plain",
        _ => "application/octet-stream",
    }
}
