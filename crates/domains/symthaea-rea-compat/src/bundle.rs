//! REA EvidenceBundle and residual-unknown graph compatibility.
//!
//! This module verifies internal consistency: deterministic manifests,
//! individual Evidence identities, unknown revision digests, reference
//! closure, verified-resolution qualification, and dependency-cycle freedom.
//! It does not authenticate the producer or prove observations true.

use crate::{
    EvidenceError, EvidenceErrorKind, MAX_EVIDENCE_JSON_BYTES, canonical_jcs_bytes,
    parse_strict_json_bytes, sha256_hex, verify_evidence_value,
};
use serde_json::{Map, Value, json};
use std::collections::{HashMap, HashSet};
use std::fmt;

pub const MAX_EVIDENCE_BUNDLE_JSON_BYTES: usize = MAX_EVIDENCE_JSON_BYTES;
const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;

const BUNDLE_KEYS: &[&str] = &[
    "artifacts",
    "providers",
    "environments",
    "scenarios",
    "captures",
    "unknowns",
    "records",
];
const UNKNOWN_REQUIRED_KEYS: &[&str] = &[
    "unknown_id",
    "revision",
    "previous_revision_digest",
    "revision_digest",
    "scope_digest",
    "question",
    "status",
    "severity",
    "domain",
    "supporting_evidence_ids",
    "contradicting_evidence_ids",
    "required_authority",
    "required_confidence",
    "required_environment",
    "recommended_probes",
    "relationships",
    "resolution",
    "mutation_evidence_ids",
];
const UNKNOWN_OPTIONAL_KEYS: &[&str] = &["comparison_conclusion_scope"];
const SUBJECT_FORMATS: &[&str] = &[
    "hopper",
    "analysis-database",
    "mach-o",
    "elf",
    "pe",
    "dos-mz",
    "dos-com",
    "raw-binary",
    "gamecube-dol",
    "zip",
    "ipa",
    "apk",
    "msix",
    "appx",
    "hap",
    "hsp",
    "app-pack",
    "asar",
    "ines",
    "godot-pck",
    "unityfs",
    "uimage",
    "dmg",
    "pkg",
    "plist",
    "javascript",
    "source-map",
    "directory",
    "file",
    "unknown",
    "mach-o-universal",
    "javascript-bundle",
    "entitlements",
    "pyinstaller",
    "node-sea",
    "dotnet-bundle",
    "bun-standalone",
    "intel-hex",
    "srec",
    "esp-image",
    "uf2",
    "dfuse",
    "esp-flash-image",
    "fit",
];
const ARCHITECTURES: &[&str] = &[
    "x86", "x86_64", "arm", "arm64", "arm64_32", "mips", "6502", "riscv", "powerpc", "m68k",
    "hexagon",
];
const COMPARISON_DIMENSIONS: &[&str] = &[
    "identity",
    "pseudocode",
    "assembly",
    "comments",
    "calls",
    "references",
    "strings_names",
    "cfg",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundleErrorKind {
    InvalidJson,
    InputTooLarge,
    SchemaViolation,
    EvidenceInvalid,
    DuplicateEvidenceId,
    NonCanonicalBundle,
    UnknownGraphInvalid,
    Canonicalization,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundleError {
    pub kind: BundleErrorKind,
    pub message: String,
}

impl BundleError {
    fn new(kind: BundleErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl fmt::Display for BundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for BundleError {}

impl From<EvidenceError> for BundleError {
    fn from(error: EvidenceError) -> Self {
        let kind = match error.kind {
            EvidenceErrorKind::InvalidJson => BundleErrorKind::InvalidJson,
            EvidenceErrorKind::InputTooLarge => BundleErrorKind::InputTooLarge,
            EvidenceErrorKind::SchemaViolation | EvidenceErrorKind::IdentityMismatch => {
                BundleErrorKind::EvidenceInvalid
            }
            EvidenceErrorKind::Canonicalization => BundleErrorKind::Canonicalization,
        };
        Self::new(kind, error.to_string())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedEvidenceBundle {
    pub bundle: Value,
    pub canonical_sha256: String,
    pub raw_file_sha256: Option<String>,
    pub record_count: usize,
    pub unknown_revision_count: usize,
}

/// Verify a parsed EvidenceBundle, including its canonical manifests and the
/// complete residual-unknown history graph.
pub fn verify_evidence_bundle_value(input: &Value) -> Result<VerifiedEvidenceBundle, BundleError> {
    let mut parsed = input.clone();
    let root = as_object(&parsed, "EvidenceBundle")?;
    strict_keys(root, BUNDLE_KEYS, &[], "EvidenceBundle")?;

    let records = require_array(field(root, "records", "EvidenceBundle")?, "records")?.clone();
    let mut records_by_id = HashMap::<String, Value>::with_capacity(records.len());
    for (index, record) in records.iter().enumerate() {
        let verified = verify_evidence_value(record).map_err(|error| {
            BundleError::new(
                BundleErrorKind::EvidenceInvalid,
                format!("records[{index}] failed Evidence verification: {error}"),
            )
        })?;
        if records_by_id
            .insert(verified.evidence_id.clone(), verified.value)
            .is_some()
        {
            return Err(BundleError::new(
                BundleErrorKind::DuplicateEvidenceId,
                format!(
                    "EvidenceBundle contains duplicate Evidence ID {}",
                    verified.evidence_id
                ),
            ));
        }
    }

    let unknowns = require_array(field(root, "unknowns", "EvidenceBundle")?, "unknowns")?.clone();
    let normalized_unknowns = unknowns
        .iter()
        .enumerate()
        .map(|(index, unknown)| {
            normalize_and_verify_unknown(unknown).map_err(|error| {
                BundleError::new(
                    error.kind,
                    format!("unknowns[{index}] failed validation: {}", error.message),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    validate_unknown_graph(&normalized_unknowns, &records_by_id)?;

    if let Some(root) = parsed.as_object_mut() {
        root.insert(
            "unknowns".to_owned(),
            Value::Array(normalized_unknowns.clone()),
        );
    }
    validate_manifest_shapes(&parsed)?;
    let canonical = create_evidence_bundle(&records, &normalized_unknowns)?;
    if parsed != canonical {
        return Err(BundleError::new(
            BundleErrorKind::NonCanonicalBundle,
            "EvidenceBundle manifests or record/unknown ordering are not canonical",
        ));
    }

    let bytes = canonical_jcs_bytes(&canonical).map_err(BundleError::from)?;
    Ok(VerifiedEvidenceBundle {
        bundle: canonical,
        canonical_sha256: sha256_hex(&bytes),
        raw_file_sha256: None,
        record_count: records.len(),
        unknown_revision_count: normalized_unknowns.len(),
    })
}

/// Parse bounded raw bundle JSON and reject duplicate object keys before schema
/// admission. The raw-file digest remains distinct from the canonical digest.
pub fn verify_evidence_bundle_json(input: &[u8]) -> Result<VerifiedEvidenceBundle, BundleError> {
    if input.len() > MAX_EVIDENCE_BUNDLE_JSON_BYTES {
        return Err(BundleError::new(
            BundleErrorKind::InputTooLarge,
            format!(
                "EvidenceBundle JSON is {} bytes; importer limit is {} bytes",
                input.len(),
                MAX_EVIDENCE_BUNDLE_JSON_BYTES
            ),
        ));
    }
    let parsed = parse_strict_json_bytes(input, MAX_EVIDENCE_BUNDLE_JSON_BYTES)
        .map_err(BundleError::from)?;
    let mut verified = verify_evidence_bundle_value(&parsed)?;
    verified.raw_file_sha256 = Some(sha256_hex(input));
    Ok(verified)
}

/// Return JCS bytes for a bundle that passes the same upstream compatibility checks.
pub fn canonical_evidence_bundle_bytes(bundle: &Value) -> Result<Vec<u8>, BundleError> {
    let verified = verify_evidence_bundle_value(bundle)?;
    canonical_jcs_bytes(&verified.bundle).map_err(BundleError::from)
}

/// Select the evidence and complete residual-unknown histories for one artifact.
///
/// This mirrors upstream REA's `evidenceBundleForTarget` projection: every
/// revision in a retained history must be scoped to `sha256`, and every
/// referenced Evidence record must either belong to that artifact or be a
/// mutation record explicitly listed in `mutation_evidence_ids`. Relationship
/// closure is preserved by removing histories whose related histories are not
/// retained, then recursively removing their dependents.
pub fn evidence_bundle_for_target(bundle: &Value, sha256: &str) -> Result<Value, BundleError> {
    let verified = verify_evidence_bundle_value(bundle)?;
    let bundle = verified.bundle;
    let root = as_object(&bundle, "EvidenceBundle")?;
    let records = require_array(field(root, "records", "EvidenceBundle")?, "records")?;
    let unknowns = require_array(field(root, "unknowns", "EvidenceBundle")?, "unknowns")?;

    let records_by_id: HashMap<&str, &Value> = records
        .iter()
        .filter_map(|record| {
            record
                .get("evidence_id")
                .and_then(Value::as_str)
                .map(|id| (id, record))
        })
        .collect();

    let mut histories = HashMap::<String, Vec<&Value>>::new();
    for unknown in unknowns {
        let id = string_field(unknown, "unknown_id")
            .ok_or_else(|| schema_error("unknown_id is required"))?;
        histories.entry(id.to_owned()).or_default().push(unknown);
    }

    let mut retained = HashSet::<String>::new();
    let mut dependents = HashMap::<String, HashSet<String>>::new();
    let mut relationships = HashMap::<String, HashSet<String>>::new();

    for (id, history) in &histories {
        let eligible = history.iter().all(|unknown| {
            if unknown.get("scope_digest").and_then(Value::as_str) != Some(sha256) {
                return false;
            }
            let mutation_ids: HashSet<&str> = unknown
                .get("mutation_evidence_ids")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect();
            let mut referenced = Vec::<&str>::new();
            for key in [
                "supporting_evidence_ids",
                "contradicting_evidence_ids",
                "mutation_evidence_ids",
            ] {
                referenced.extend(
                    unknown
                        .get(key)
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                        .filter_map(Value::as_str),
                );
            }
            referenced.extend(
                unknown
                    .get("resolution")
                    .and_then(|resolution| resolution.get("evidence_ids"))
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str),
            );
            referenced.into_iter().all(|evidence_id| {
                let Some(record) = records_by_id.get(evidence_id) else {
                    return false;
                };
                let subject_matches = record
                    .get("subject")
                    .and_then(|subject| subject.get("digest"))
                    .and_then(|digest| digest.get("sha256"))
                    .and_then(Value::as_str)
                    == Some(sha256);
                let is_target_mutation = mutation_ids.contains(evidence_id)
                    && record.get("predicate_type").and_then(Value::as_str)
                        == Some("rea.residual-unknown-mutation");
                subject_matches || is_target_mutation
            })
        });
        if !eligible {
            continue;
        }

        retained.insert(id.clone());
        let related_ids: HashSet<String> = history
            .iter()
            .flat_map(|unknown| {
                unknown
                    .get("relationships")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
            })
            .filter_map(|relationship| string_field(relationship, "unknown_id"))
            .map(str::to_owned)
            .collect();
        relationships.insert(id.clone(), related_ids.clone());
        for related_id in related_ids {
            dependents.entry(related_id).or_default().insert(id.clone());
        }
    }

    let mut pending = Vec::<String>::new();
    for (id, related_ids) in &relationships {
        if related_ids
            .iter()
            .any(|related_id| !retained.contains(related_id))
        {
            pending.push(id.clone());
        }
    }
    let mut index = 0;
    while index < pending.len() {
        let id = pending[index].clone();
        index += 1;
        if !retained.remove(&id) {
            continue;
        }
        if let Some(children) = dependents.get(&id) {
            pending.extend(children.iter().cloned());
        }
    }

    let selected_unknowns: Vec<Value> = unknowns
        .iter()
        .filter(|unknown| {
            string_field(unknown, "unknown_id").is_some_and(|id| retained.contains(id))
        })
        .cloned()
        .collect();
    let mutation_ids: HashSet<String> = selected_unknowns
        .iter()
        .flat_map(|unknown| {
            unknown
                .get("mutation_evidence_ids")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();

    let selected_records: Vec<Value> = records
        .iter()
        .filter(|record| {
            let subject_matches = record
                .get("subject")
                .and_then(|subject| subject.get("digest"))
                .and_then(|digest| digest.get("sha256"))
                .and_then(Value::as_str)
                == Some(sha256);
            let is_mutation = record
                .get("evidence_id")
                .and_then(Value::as_str)
                .is_some_and(|id| mutation_ids.contains(id));
            subject_matches || is_mutation
        })
        .cloned()
        .collect();

    create_evidence_bundle(&selected_records, &selected_unknowns)
}

/// Upstream REA bundle authentication does not reject Evidence links pointing
/// outside the bundle. This separate stricter audit reports them without
/// changing compatibility parser disposition.
pub fn audit_evidence_link_closure(bundle: &Value) -> Result<Vec<(String, String)>, BundleError> {
    let root = as_object(bundle, "EvidenceBundle")?;
    let records = require_array(field(root, "records", "EvidenceBundle")?, "records")?;
    let known: HashSet<&str> = records
        .iter()
        .filter_map(|record| record.get("evidence_id").and_then(Value::as_str))
        .collect();
    let mut missing = Vec::new();
    for record in records {
        let Some(source_id) = record.get("evidence_id").and_then(Value::as_str) else {
            continue;
        };
        if let Some(links) = record.get("evidence_links").and_then(Value::as_array) {
            for link in links.iter().filter_map(Value::as_str) {
                if !known.contains(link) {
                    missing.push((source_id.to_owned(), link.to_owned()));
                }
            }
        }
    }
    missing.sort();
    missing.dedup();
    Ok(missing)
}

fn create_evidence_bundle(records: &[Value], unknowns: &[Value]) -> Result<Value, BundleError> {
    let mut sorted_records = records.to_vec();
    sorted_records.sort_by(|a, b| {
        string_field(a, "evidence_id")
            .unwrap_or_default()
            .cmp(string_field(b, "evidence_id").unwrap_or_default())
    });

    let mut artifacts = Vec::new();
    let mut providers = Vec::new();
    let mut environments = Vec::new();
    let mut scenarios = Vec::new();
    let mut captures = Vec::new();

    for record in &sorted_records {
        let map = as_object(record, "Evidence record")?;
        let subject = field(map, "subject", "Evidence")?;
        if !subject.is_null() {
            let subject = as_object(subject, "Evidence.subject")?;
            let artifact = json!({
                "digest": field(subject, "digest", "Evidence.subject")?,
                "format": field(subject, "format", "Evidence.subject")?,
                "architecture": field(subject, "architecture", "Evidence.subject")?,
            });
            artifacts.push((artifact_manifest_key(&artifact)?, artifact));
        }

        let provider = field(map, "provider", "Evidence")?.clone();
        providers.push((provider_manifest_key(&provider)?, provider));

        let environment = field(map, "environment", "Evidence")?;
        if !environment.is_null() {
            environments.push((environment_manifest_key(environment)?, environment.clone()));
        }

        let evidence_id = field(map, "evidence_id", "Evidence")?.clone();
        if field(map, "authority", "Evidence")? == "controlled-replay" {
            scenarios.push(json!({
                "evidence_id": evidence_id,
                "operation": field(map, "operation", "Evidence")?,
                "authority": "controlled-replay",
            }));
        }
        if field(map, "predicate_type", "Evidence")? == "rea.process-capture" {
            captures.push(json!({
                "evidence_id": field(map, "evidence_id", "Evidence")?,
                "predicate_type": field(map, "predicate_type", "Evidence")?,
            }));
        }
    }

    let mut sorted_unknowns = unknowns.to_vec();
    sorted_unknowns.sort_by(|a, b| {
        string_field(a, "unknown_id")
            .unwrap_or_default()
            .cmp(string_field(b, "unknown_id").unwrap_or_default())
            .then_with(|| {
                safe_integer_field(a, "revision")
                    .unwrap_or(0.0)
                    .total_cmp(&safe_integer_field(b, "revision").unwrap_or(0.0))
            })
    });

    Ok(json!({
        "artifacts": unique_sorted_manifest_values(artifacts),
        "providers": unique_sorted_manifest_values(providers),
        "environments": unique_sorted_manifest_values(environments),
        "scenarios": scenarios,
        "captures": captures,
        "unknowns": sorted_unknowns,
        "records": sorted_records,
    }))
}

fn unique_sorted_manifest_values(values: Vec<(String, Value)>) -> Vec<Value> {
    let mut values = values;
    values.sort_by(|a, b| a.0.cmp(&b.0));
    let mut output = Vec::with_capacity(values.len());
    let mut previous: Option<String> = None;
    for (key, value) in values {
        if previous.as_deref() != Some(key.as_str()) {
            previous = Some(key);
            output.push(value);
        }
    }
    output
}

fn artifact_manifest_key(value: &Value) -> Result<String, BundleError> {
    let map = as_object(value, "artifact manifest")?;
    let digest = as_object(
        field(map, "digest", "artifact manifest")?,
        "artifact digest",
    )?;
    let sha = json_string(field(digest, "sha256", "artifact digest")?)?;
    let format = json_string(field(map, "format", "artifact manifest")?)?;
    let architecture = json_optional_string(field(map, "architecture", "artifact manifest")?)?;
    Ok(format!(
        "{{\"digest\":{{\"sha256\":{sha}}},\"format\":{format},\"architecture\":{architecture}}}"
    ))
}

fn provider_manifest_key(value: &Value) -> Result<String, BundleError> {
    let map = as_object(value, "provider manifest")?;
    let id = json_string(field(map, "id", "provider manifest")?)?;
    let name = json_string(field(map, "name", "provider manifest")?)?;
    let version = json_optional_string(field(map, "version", "provider manifest")?)?;
    Ok(format!(
        "{{\"id\":{id},\"name\":{name},\"version\":{version}}}"
    ))
}

fn environment_manifest_key(value: &Value) -> Result<String, BundleError> {
    let map = as_object(value, "environment manifest")?;
    let id = json_string(field(map, "id", "environment manifest")?)?;
    let platform = json_string(field(map, "platform", "environment manifest")?)?;
    let architecture = json_string(field(map, "architecture", "environment manifest")?)?;
    let isolation = json_string(field(map, "isolation", "environment manifest")?)?;
    Ok(format!(
        "{{\"id\":{id},\"platform\":{platform},\"architecture\":{architecture},\"isolation\":{isolation}}}"
    ))
}

fn json_string(value: &Value) -> Result<String, BundleError> {
    let text = value
        .as_str()
        .ok_or_else(|| schema_error("manifest field must be a string"))?;
    serde_json::to_string(text).map_err(|error| {
        BundleError::new(
            BundleErrorKind::Canonicalization,
            format!("manifest string serialization failed: {error}"),
        )
    })
}

fn json_optional_string(value: &Value) -> Result<String, BundleError> {
    if value.is_null() {
        Ok("null".to_owned())
    } else {
        json_string(value)
    }
}

fn normalize_and_verify_unknown(value: &Value) -> Result<Value, BundleError> {
    let map = as_object(value, "ResidualUnknown")?;
    strict_keys(
        map,
        UNKNOWN_REQUIRED_KEYS,
        UNKNOWN_OPTIONAL_KEYS,
        "ResidualUnknown",
    )?;

    let mut normalized = value.clone();
    let question = trimmed_text(field(map, "question", "ResidualUnknown")?, "question")?;
    let domain = trimmed_text(field(map, "domain", "ResidualUnknown")?, "domain")?;
    normalized["question"] = Value::String(question.clone());
    normalized["domain"] = Value::String(domain.clone());

    let unknown_id = prefixed_digest(
        field(map, "unknown_id", "ResidualUnknown")?,
        "unk_",
        "unknown_id",
    )?;
    let revision = safe_integer(field(map, "revision", "ResidualUnknown")?, "revision")?;
    if revision < 1.0 {
        return Err(schema_error("ResidualUnknown.revision must be at least 1"));
    }
    validate_optional_digest(
        field(map, "previous_revision_digest", "ResidualUnknown")?,
        "previous_revision_digest",
    )?;
    validate_sha256(
        field(map, "revision_digest", "ResidualUnknown")?,
        "revision_digest",
    )?;
    validate_optional_digest(
        field(map, "scope_digest", "ResidualUnknown")?,
        "scope_digest",
    )?;

    let status = string_value(field(map, "status", "ResidualUnknown")?, "status")?;
    if ![
        "open",
        "investigating",
        "blocked",
        "contradicted",
        "resolved",
    ]
    .contains(&status)
    {
        return Err(schema_error(format!(
            "unsupported ResidualUnknown.status '{status}'"
        )));
    }
    let severity = string_value(field(map, "severity", "ResidualUnknown")?, "severity")?;
    if !["low", "medium", "high", "critical"].contains(&severity) {
        return Err(schema_error(format!(
            "unsupported ResidualUnknown.severity '{severity}'"
        )));
    }

    if let Some(comparison) = map.get("comparison_conclusion_scope") {
        validate_comparison_conclusion_scope(comparison)?;
    }
    let supporting = validate_id_array(
        field(map, "supporting_evidence_ids", "ResidualUnknown")?,
        "supporting_evidence_ids",
        "ev_",
        false,
    )?;
    let contradicting = validate_id_array(
        field(map, "contradicting_evidence_ids", "ResidualUnknown")?,
        "contradicting_evidence_ids",
        "ev_",
        false,
    )?;
    validate_id_array(
        field(map, "mutation_evidence_ids", "ResidualUnknown")?,
        "mutation_evidence_ids",
        "ev_",
        true,
    )?;
    let supporting_set: HashSet<&str> = supporting.iter().map(String::as_str).collect();
    let contradicting_set: HashSet<&str> = contradicting.iter().map(String::as_str).collect();
    if supporting
        .iter()
        .any(|id| contradicting_set.contains(id.as_str()))
    {
        return Err(schema_error(
            "Evidence cannot simultaneously support and contradict",
        ));
    }

    let required_authority = field(map, "required_authority", "ResidualUnknown")?;
    if !required_authority.is_null() {
        let authority = string_value(required_authority, "required_authority")?;
        if ![
            "shipped-artifact",
            "controlled-replay",
            "historical-reference",
            "external-service",
            "analyst-inference",
        ]
        .contains(&authority)
        {
            return Err(schema_error("unsupported required_authority"));
        }
    }
    let required_confidence = string_value(
        field(map, "required_confidence", "ResidualUnknown")?,
        "required_confidence",
    )?;
    if !["observed", "derived", "inferred"].contains(&required_confidence) {
        return Err(schema_error("unsupported required_confidence"));
    }

    validate_environment_requirement(field(map, "required_environment", "ResidualUnknown")?)?;
    normalize_probes(
        &mut normalized,
        field(map, "recommended_probes", "ResidualUnknown")?,
    )?;
    validate_relationships(field(map, "relationships", "ResidualUnknown")?, &unknown_id)?;

    let resolution = field(map, "resolution", "ResidualUnknown")?;
    normalize_resolution(&mut normalized, resolution)?;
    if (status == "resolved") != !resolution.is_null() {
        return Err(schema_error(
            "Only resolved residual unknowns carry a resolution",
        ));
    }
    if status == "contradicted" && contradicting.is_empty() {
        return Err(schema_error(
            "Contradicted status requires contradicting evidence",
        ));
    }

    if revision == 1.0 {
        if !map["previous_revision_digest"].is_null() {
            return Err(schema_error("Only revision 1 has no predecessor digest"));
        }
    } else if map["previous_revision_digest"].is_null() {
        return Err(schema_error(
            "Noninitial revisions require a predecessor digest",
        ));
    }

    if let Some(comparison) = map.get("comparison_conclusion_scope") {
        let evidence_id = comparison["comparison_evidence_id"]
            .as_str()
            .expect("validated comparison Evidence ID");
        if !supporting_set.contains(evidence_id) && !contradicting_set.contains(evidence_id) {
            return Err(schema_error(
                "Scoped comparison Evidence must support or contradict the unknown",
            ));
        }
        if domain != "function-comparison" {
            return Err(schema_error(
                "Comparison conclusion scope requires function-comparison domain",
            ));
        }
    }

    let mut identity = Map::new();
    identity.insert("domain".to_owned(), Value::String(domain));
    identity.insert(
        "question".to_owned(),
        Value::String(collapse_ecmascript_whitespace(&question)),
    );
    identity.insert(
        "scope_digest".to_owned(),
        field(map, "scope_digest", "ResidualUnknown")?.clone(),
    );
    if let Some(comparison) = map.get("comparison_conclusion_scope") {
        identity.insert("comparison_conclusion_scope".to_owned(), comparison.clone());
    }
    let expected_unknown_id = format!("unk_{}", hash_canonical(&Value::Object(identity))?);
    if expected_unknown_id != unknown_id {
        return Err(graph_error(
            "ResidualUnknown.unknown_id does not match its stable immutable identity",
        ));
    }

    let mut revision_payload = normalized.clone();
    revision_payload
        .as_object_mut()
        .expect("unknown is an object")
        .remove("revision_digest");
    let expected_revision_digest = hash_canonical(&revision_payload)?;
    let actual_revision_digest = string_field(value, "revision_digest").unwrap_or_default();
    if expected_revision_digest != actual_revision_digest {
        return Err(graph_error(
            "ResidualUnknown.revision_digest does not match its canonical revision payload",
        ));
    }
    Ok(normalized)
}

fn validate_comparison_conclusion_scope(value: &Value) -> Result<(), BundleError> {
    let map = as_object(value, "comparison_conclusion_scope")?;
    strict_keys(
        map,
        &["comparison_evidence_id", "dimensions"],
        &[],
        "comparison_conclusion_scope",
    )?;
    prefixed_digest(
        field(map, "comparison_evidence_id", "comparison_conclusion_scope")?,
        "ev_",
        "comparison_evidence_id",
    )?;
    let dimensions = require_array(
        field(map, "dimensions", "comparison_conclusion_scope")?,
        "dimensions",
    )?;
    if dimensions.is_empty() {
        return Err(schema_error("comparison dimensions must not be empty"));
    }
    let mut previous: Option<&str> = None;
    let mut seen = HashSet::new();
    for item in dimensions {
        let dimension = string_value(item, "comparison dimension")?;
        if dimension != "overall" && !COMPARISON_DIMENSIONS.contains(&dimension) {
            return Err(schema_error(format!(
                "unsupported comparison dimension '{dimension}'"
            )));
        }
        if !seen.insert(dimension) || previous.is_some_and(|last| last >= dimension) {
            return Err(schema_error(
                "comparison dimensions must be sorted and unique",
            ));
        }
        previous = Some(dimension);
    }
    Ok(())
}

fn validate_environment_requirement(value: &Value) -> Result<(), BundleError> {
    if value.is_null() {
        return Ok(());
    }
    let map = as_object(value, "required_environment")?;
    strict_keys(
        map,
        &["id", "platform", "architecture", "isolation"],
        &[],
        "required_environment",
    )?;
    for key in ["id", "platform", "architecture"] {
        let value = field(map, key, "required_environment")?;
        if !value.is_null() {
            nonempty_string(value, &format!("required_environment.{key}"))?;
        }
    }
    let isolation = field(map, "isolation", "required_environment")?;
    if !isolation.is_null() {
        let isolation = string_value(isolation, "required_environment.isolation")?;
        if !["none", "process", "container", "virtual-machine"].contains(&isolation) {
            return Err(schema_error("unsupported required_environment.isolation"));
        }
    }
    Ok(())
}

fn normalize_probes(normalized: &mut Value, value: &Value) -> Result<(), BundleError> {
    let probes = require_array(value, "recommended_probes")?;
    let mut result = Vec::with_capacity(probes.len());
    for (index, probe) in probes.iter().enumerate() {
        let context = format!("recommended_probes[{index}]");
        let map = as_object(probe, &context)?;
        strict_keys(map, &["operation", "rationale"], &[], &context)?;
        nonempty_string(
            field(map, "operation", &context)?,
            &format!("{context}.operation"),
        )?;
        let rationale = trimmed_text(
            field(map, "rationale", &context)?,
            &format!("{context}.rationale"),
        )?;
        let mut object = probe.clone();
        object["rationale"] = Value::String(rationale);
        result.push(object);
    }
    normalized["recommended_probes"] = Value::Array(result);
    Ok(())
}

fn normalize_resolution(normalized: &mut Value, value: &Value) -> Result<(), BundleError> {
    if value.is_null() {
        return Ok(());
    }
    let map = as_object(value, "resolution")?;
    strict_keys(
        map,
        &["disposition", "rationale", "evidence_ids"],
        &[],
        "resolution",
    )?;
    let disposition = string_value(
        field(map, "disposition", "resolution")?,
        "resolution.disposition",
    )?;
    let require_nonempty = if disposition == "verified" {
        true
    } else if disposition == "withdrawn" || disposition == "out-of-scope" {
        false
    } else {
        return Err(schema_error(format!(
            "unsupported resolution disposition '{disposition}'"
        )));
    };
    let rationale = trimmed_text(
        field(map, "rationale", "resolution")?,
        "resolution.rationale",
    )?;
    validate_id_array_allow_duplicates(
        field(map, "evidence_ids", "resolution")?,
        "resolution.evidence_ids",
        "ev_",
        require_nonempty,
    )?;
    normalized["resolution"]["rationale"] = Value::String(rationale);
    Ok(())
}

fn validate_relationships(value: &Value, own_id: &str) -> Result<(), BundleError> {
    let relationships = require_array(value, "relationships")?;
    let mut seen = HashSet::new();
    for (index, relationship) in relationships.iter().enumerate() {
        let context = format!("relationships[{index}]");
        let map = as_object(relationship, &context)?;
        strict_keys(map, &["type", "unknown_id"], &[], &context)?;
        let kind = string_value(field(map, "type", &context)?, &format!("{context}.type"))?;
        if !["depends-on", "related-to", "contradicts"].contains(&kind) {
            return Err(schema_error(format!(
                "unsupported relationship type '{kind}'"
            )));
        }
        let id = prefixed_digest(
            field(map, "unknown_id", &context)?,
            "unk_",
            &format!("{context}.unknown_id"),
        )?;
        if id == own_id {
            return Err(schema_error("An unknown cannot relate to itself"));
        }
        if !seen.insert(format!("{kind}:{id}")) {
            return Err(schema_error("ResidualUnknown relationships must be unique"));
        }
    }
    Ok(())
}

fn validate_id_array_allow_duplicates(
    value: &Value,
    context: &str,
    prefix: &str,
    nonempty: bool,
) -> Result<Vec<String>, BundleError> {
    let values = require_array(value, context)?;
    if nonempty && values.is_empty() {
        return Err(schema_error(format!("{context} must not be empty")));
    }
    values
        .iter()
        .enumerate()
        .map(|(index, item)| prefixed_digest(item, prefix, &format!("{context}[{index}]")))
        .collect()
}

fn validate_id_array(
    value: &Value,
    context: &str,
    prefix: &str,
    nonempty: bool,
) -> Result<Vec<String>, BundleError> {
    let values = require_array(value, context)?;
    if nonempty && values.is_empty() {
        return Err(schema_error(format!("{context} must not be empty")));
    }
    let mut output = Vec::with_capacity(values.len());
    let mut seen = HashSet::new();
    for (index, item) in values.iter().enumerate() {
        let id = prefixed_digest(item, prefix, &format!("{context}[{index}]"))?;
        if !seen.insert(id.clone()) {
            return Err(schema_error(format!("{context} IDs must be unique")));
        }
        output.push(id);
    }
    Ok(output)
}

fn validate_unknown_graph(
    unknowns: &[Value],
    evidence_by_id: &HashMap<String, Value>,
) -> Result<(), BundleError> {
    let mut histories = HashMap::<String, Vec<Value>>::new();
    for unknown in unknowns {
        let id = string_field(unknown, "unknown_id")
            .ok_or_else(|| schema_error("unknown_id is required"))?;
        histories
            .entry(id.to_owned())
            .or_default()
            .push(unknown.clone());
    }

    let mut heads = HashMap::<String, Value>::with_capacity(histories.len());
    for (id, history) in &mut histories {
        history.sort_by(|a, b| {
            safe_integer_field(a, "revision")
                .unwrap_or(0.0)
                .total_cmp(&safe_integer_field(b, "revision").unwrap_or(0.0))
        });
        for (index, revision) in history.iter().enumerate() {
            if safe_integer_field(revision, "revision") != Some((index + 1) as f64) {
                return Err(graph_error(format!(
                    "Residual unknown {id} revision history has a gap"
                )));
            }
            if index > 0
                && revision.get("previous_revision_digest")
                    != history[index - 1].get("revision_digest")
            {
                return Err(graph_error(format!(
                    "Residual unknown {id} revision chain is broken"
                )));
            }
        }
        let head = history
            .last()
            .ok_or_else(|| graph_error("Residual unknown history is empty"))?;
        heads.insert(id.clone(), head.clone());
    }

    for unknown in unknowns {
        let id = string_field(unknown, "unknown_id").unwrap_or_default();
        let mut referenced = Vec::new();
        for key in [
            "supporting_evidence_ids",
            "contradicting_evidence_ids",
            "mutation_evidence_ids",
        ] {
            if let Some(ids) = unknown.get(key).and_then(Value::as_array) {
                referenced.extend(ids.iter().filter_map(Value::as_str));
            }
        }
        if let Some(ids) = unknown
            .get("resolution")
            .and_then(|r| r.get("evidence_ids"))
            .and_then(Value::as_array)
        {
            referenced.extend(ids.iter().filter_map(Value::as_str));
        }
        for evidence_id in referenced {
            if !evidence_by_id.contains_key(evidence_id) {
                return Err(graph_error(format!(
                    "Residual unknown {id} references missing Evidence {evidence_id}"
                )));
            }
        }
        if let Some(relationships) = unknown.get("relationships").and_then(Value::as_array) {
            for relationship in relationships {
                let related = string_field(relationship, "unknown_id").unwrap_or_default();
                if !heads.contains_key(related) {
                    return Err(graph_error(format!(
                        "Residual unknown {id} references missing unknown {related}"
                    )));
                }
            }
        }
        if unknown
            .get("resolution")
            .and_then(|r| r.get("disposition"))
            .and_then(Value::as_str)
            == Some("verified")
        {
            validate_verified_resolution(unknown, evidence_by_id)?;
        }
    }
    reject_dependency_cycles(&heads)
}

fn validate_verified_resolution(
    unknown: &Value,
    evidence_by_id: &HashMap<String, Value>,
) -> Result<(), BundleError> {
    let id = string_field(unknown, "unknown_id").unwrap_or_default();
    if unknown
        .get("contradicting_evidence_ids")
        .and_then(Value::as_array)
        .is_some_and(|v| !v.is_empty())
    {
        return Err(graph_error(format!(
            "Residual unknown {id} retains contradicting evidence"
        )));
    }
    let supporting: HashSet<&str> = unknown
        .get("supporting_evidence_ids")
        .and_then(Value::as_array)
        .map(|v| v.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let required_authority = unknown.get("required_authority").and_then(Value::as_str);
    let required_confidence = unknown
        .get("required_confidence")
        .and_then(Value::as_str)
        .unwrap_or("inferred");
    let resolution_ids = unknown
        .get("resolution")
        .and_then(|r| r.get("evidence_ids"))
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let qualifies = resolution_ids
        .iter()
        .filter_map(Value::as_str)
        .any(|evidence_id| {
            if !supporting.contains(evidence_id) {
                return false;
            }
            let Some(evidence) = evidence_by_id.get(evidence_id) else {
                return false;
            };
            if evidence.get("predicate_type").and_then(Value::as_str)
                == Some("rea.residual-unknown-mutation")
            {
                return false;
            }
            if required_authority.is_some_and(|required| {
                evidence.get("authority").and_then(Value::as_str) != Some(required)
            }) {
                return false;
            }
            let confidence = evidence
                .get("confidence")
                .and_then(Value::as_str)
                .unwrap_or("inferred");
            confidence_rank(confidence) >= confidence_rank(required_confidence)
                && environment_satisfies(
                    unknown.get("required_environment"),
                    evidence.get("environment"),
                )
        });
    if !qualifies {
        return Err(graph_error(format!(
            "Residual unknown {id} has no qualifying resolution Evidence"
        )));
    }
    Ok(())
}

fn confidence_rank(value: &str) -> u8 {
    match value {
        "inferred" => 1,
        "derived" => 2,
        "observed" => 3,
        _ => 0,
    }
}

fn environment_satisfies(requirement: Option<&Value>, actual: Option<&Value>) -> bool {
    let Some(requirement) = requirement.filter(|v| !v.is_null()) else {
        return true;
    };
    let Some(actual) = actual.filter(|v| !v.is_null()) else {
        return false;
    };
    ["id", "platform", "architecture", "isolation"]
        .into_iter()
        .all(|key| {
            requirement.get(key).is_none_or(Value::is_null)
                || requirement.get(key) == actual.get(key)
        })
}

fn reject_dependency_cycles(unknowns: &HashMap<String, Value>) -> Result<(), BundleError> {
    struct Frame {
        id: String,
        dependencies: Vec<String>,
        next: usize,
    }
    let mut visited = HashSet::<String>::new();
    let mut active = HashSet::<String>::new();

    for root in unknowns.keys() {
        if visited.contains(root) {
            continue;
        }
        active.insert(root.clone());
        let mut stack = vec![Frame {
            id: root.clone(),
            dependencies: dependency_ids(unknowns.get(root).expect("root exists")),
            next: 0,
        }];
        while let Some(frame) = stack.last_mut() {
            if frame.next >= frame.dependencies.len() {
                let finished = stack.pop().expect("nonempty stack").id;
                active.remove(&finished);
                visited.insert(finished);
                continue;
            }
            let dependency = frame.dependencies[frame.next].clone();
            frame.next += 1;
            if active.contains(&dependency) {
                return Err(graph_error(format!(
                    "Residual unknown dependency cycle through {dependency}"
                )));
            }
            if visited.contains(&dependency) {
                continue;
            }
            let value = unknowns
                .get(&dependency)
                .ok_or_else(|| graph_error(format!("Missing dependency {dependency}")))?;
            active.insert(dependency.clone());
            stack.push(Frame {
                id: dependency,
                dependencies: dependency_ids(value),
                next: 0,
            });
        }
    }
    Ok(())
}

fn dependency_ids(unknown: &Value) -> Vec<String> {
    unknown
        .get("relationships")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|r| r.get("type").and_then(Value::as_str) == Some("depends-on"))
        .filter_map(|r| string_field(r, "unknown_id").map(str::to_owned))
        .collect()
}

fn validate_manifest_shapes(bundle: &Value) -> Result<(), BundleError> {
    let root = as_object(bundle, "EvidenceBundle")?;
    for v in require_array(field(root, "artifacts", "EvidenceBundle")?, "artifacts")? {
        validate_artifact_manifest(v)?;
    }
    for v in require_array(field(root, "providers", "EvidenceBundle")?, "providers")? {
        validate_provider_manifest(v)?;
    }
    for v in require_array(
        field(root, "environments", "EvidenceBundle")?,
        "environments",
    )? {
        validate_environment_manifest(v)?;
    }
    for v in require_array(field(root, "scenarios", "EvidenceBundle")?, "scenarios")? {
        validate_scenario_manifest(v)?;
    }
    for v in require_array(field(root, "captures", "EvidenceBundle")?, "captures")? {
        validate_capture_manifest(v)?;
    }
    Ok(())
}

fn validate_artifact_manifest(value: &Value) -> Result<(), BundleError> {
    let map = as_object(value, "artifact manifest")?;
    strict_keys(
        map,
        &["digest", "format", "architecture"],
        &[],
        "artifact manifest",
    )?;
    let digest = as_object(
        field(map, "digest", "artifact manifest")?,
        "artifact digest",
    )?;
    strict_keys(digest, &["sha256"], &[], "artifact digest")?;
    validate_sha256(
        field(digest, "sha256", "artifact digest")?,
        "artifact digest.sha256",
    )?;
    let format = string_value(
        field(map, "format", "artifact manifest")?,
        "artifact.format",
    )?;
    if !SUBJECT_FORMATS.contains(&format) {
        return Err(schema_error("unsupported artifact format"));
    }
    let architecture = field(map, "architecture", "artifact manifest")?;
    if !architecture.is_null() {
        let architecture = string_value(architecture, "artifact.architecture")?;
        if !ARCHITECTURES.contains(&architecture) {
            return Err(schema_error("unsupported artifact architecture"));
        }
        if format == "gamecube-dol" && architecture != "powerpc" {
            return Err(schema_error("gamecube-dol requires powerpc"));
        }
        if format != "raw-binary"
            && format != "gamecube-dol"
            && matches!(architecture, "6502" | "m68k")
        {
            return Err(schema_error("6502 and m68k are only valid for raw-binary"));
        }
    } else if format == "gamecube-dol" {
        return Err(schema_error("gamecube-dol requires powerpc"));
    }
    Ok(())
}

fn validate_provider_manifest(value: &Value) -> Result<(), BundleError> {
    let map = as_object(value, "provider manifest")?;
    strict_keys(map, &["id", "name", "version"], &[], "provider manifest")?;
    nonempty_string(field(map, "id", "provider manifest")?, "provider.id")?;
    nonempty_string(field(map, "name", "provider manifest")?, "provider.name")?;
    let version = field(map, "version", "provider manifest")?;
    if !version.is_null() && version.as_str().is_none() {
        return Err(schema_error("provider.version must be string or null"));
    }
    Ok(())
}

fn validate_environment_manifest(value: &Value) -> Result<(), BundleError> {
    let map = as_object(value, "environment manifest")?;
    strict_keys(
        map,
        &["id", "platform", "architecture", "isolation"],
        &[],
        "environment manifest",
    )?;
    for key in ["id", "platform", "architecture"] {
        nonempty_string(field(map, key, "environment manifest")?, key)?;
    }
    let isolation = string_value(
        field(map, "isolation", "environment manifest")?,
        "isolation",
    )?;
    if !["none", "process", "container", "virtual-machine"].contains(&isolation) {
        return Err(schema_error("unsupported isolation"));
    }
    Ok(())
}

fn validate_scenario_manifest(value: &Value) -> Result<(), BundleError> {
    let map = as_object(value, "scenario manifest")?;
    strict_keys(
        map,
        &["evidence_id", "operation", "authority"],
        &[],
        "scenario manifest",
    )?;
    prefixed_digest(
        field(map, "evidence_id", "scenario manifest")?,
        "ev_",
        "scenario.evidence_id",
    )?;
    nonempty_string(
        field(map, "operation", "scenario manifest")?,
        "scenario.operation",
    )?;
    if field(map, "authority", "scenario manifest")? != "controlled-replay" {
        return Err(schema_error("scenario authority must be controlled-replay"));
    }
    Ok(())
}

fn validate_capture_manifest(value: &Value) -> Result<(), BundleError> {
    let map = as_object(value, "capture manifest")?;
    strict_keys(
        map,
        &["evidence_id", "predicate_type"],
        &[],
        "capture manifest",
    )?;
    prefixed_digest(
        field(map, "evidence_id", "capture manifest")?,
        "ev_",
        "capture.evidence_id",
    )?;
    nonempty_string(
        field(map, "predicate_type", "capture manifest")?,
        "capture.predicate_type",
    )?;
    Ok(())
}

fn hash_canonical(value: &Value) -> Result<String, BundleError> {
    let bytes = canonical_jcs_bytes(value).map_err(BundleError::from)?;
    Ok(sha256_hex(&bytes))
}

fn as_object<'a>(value: &'a Value, context: &str) -> Result<&'a Map<String, Value>, BundleError> {
    value
        .as_object()
        .ok_or_else(|| schema_error(format!("{context} must be an object")))
}

fn require_array<'a>(value: &'a Value, context: &str) -> Result<&'a Vec<Value>, BundleError> {
    value
        .as_array()
        .ok_or_else(|| schema_error(format!("{context} must be an array")))
}

fn field<'a>(
    map: &'a Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<&'a Value, BundleError> {
    map.get(key)
        .ok_or_else(|| schema_error(format!("{context}.{key} is required")))
}

fn strict_keys(
    map: &Map<String, Value>,
    required: &[&str],
    optional: &[&str],
    context: &str,
) -> Result<(), BundleError> {
    for key in required {
        if !map.contains_key(*key) {
            return Err(schema_error(format!("{context}.{key} is required")));
        }
    }
    for key in map.keys() {
        if !required.contains(&key.as_str()) && !optional.contains(&key.as_str()) {
            return Err(schema_error(format!(
                "{context} contains unknown field '{key}'"
            )));
        }
    }
    Ok(())
}

fn string_value<'a>(value: &'a Value, context: &str) -> Result<&'a str, BundleError> {
    value
        .as_str()
        .ok_or_else(|| schema_error(format!("{context} must be a string")))
}

fn nonempty_string<'a>(value: &'a Value, context: &str) -> Result<&'a str, BundleError> {
    let value = string_value(value, context)?;
    if value.is_empty() {
        return Err(schema_error(format!("{context} must not be empty")));
    }
    Ok(value)
}

fn trimmed_text(value: &Value, context: &str) -> Result<String, BundleError> {
    let value = string_value(value, context)?;
    let value = trim_ecmascript_whitespace(value);
    if value.is_empty() {
        return Err(schema_error(format!("{context} must not be empty")));
    }
    Ok(value.to_owned())
}

fn trim_ecmascript_whitespace(value: &str) -> &str {
    value.trim_matches(is_ecmascript_whitespace)
}

fn collapse_ecmascript_whitespace(value: &str) -> String {
    let mut output = String::new();
    let mut pending = false;
    for ch in value.chars() {
        if is_ecmascript_whitespace(ch) {
            pending = !output.is_empty();
        } else {
            if pending {
                output.push(' ');
                pending = false;
            }
            output.push(ch);
        }
    }
    output
}

fn is_ecmascript_whitespace(ch: char) -> bool {
    matches!(
        ch,
        '\u{0009}'
            | '\u{000A}'
            | '\u{000B}'
            | '\u{000C}'
            | '\u{000D}'
            | '\u{0020}'
            | '\u{1680}'
            | '\u{2000}'
            ..='\u{200A}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202F}'
                | '\u{205F}'
                | '\u{3000}'
                | '\u{FEFF}'
    )
}

fn prefixed_digest(value: &Value, prefix: &str, context: &str) -> Result<String, BundleError> {
    let value = string_value(value, context)?;
    let digest = value
        .strip_prefix(prefix)
        .ok_or_else(|| schema_error(format!("{context} must start with '{prefix}'")))?;
    validate_hex_digest(digest, context)?;
    Ok(value.to_owned())
}

fn validate_optional_digest(value: &Value, context: &str) -> Result<(), BundleError> {
    if value.is_null() {
        return Ok(());
    }
    validate_sha256(value, context)
}

fn validate_sha256(value: &Value, context: &str) -> Result<(), BundleError> {
    validate_hex_digest(string_value(value, context)?, context)
}

fn validate_hex_digest(value: &str, context: &str) -> Result<(), BundleError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(schema_error(format!(
            "{context} must be 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn safe_integer(value: &Value, context: &str) -> Result<f64, BundleError> {
    let n = value
        .as_f64()
        .ok_or_else(|| schema_error(format!("{context} must be a safe integer")))?;
    if !n.is_finite() || n.fract() != 0.0 || n < 0.0 || n > MAX_SAFE_INTEGER {
        return Err(schema_error(format!(
            "{context} must be a non-negative safe integer"
        )));
    }
    Ok(n)
}

fn safe_integer_field(value: &Value, name: &str) -> Option<f64> {
    value.get(name).and_then(Value::as_f64)
}
fn string_field<'a>(value: &'a Value, name: &str) -> Option<&'a str> {
    value.get(name).and_then(Value::as_str)
}
fn graph_error(message: impl Into<String>) -> BundleError {
    BundleError::new(BundleErrorKind::UnknownGraphInvalid, message)
}
fn schema_error(message: impl Into<String>) -> BundleError {
    BundleError::new(BundleErrorKind::SchemaViolation, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_evidence_id;

    fn record() -> Value {
        serde_json::from_str(include_str!("../tests/fixtures/rea-evidence-v1.json"))
            .expect("source-derived Evidence fixture is valid")
    }

    fn unknown_id(question: &str, domain: &str) -> String {
        let identity = json!({
            "domain": domain.trim(),
            "question": collapse_ecmascript_whitespace(trim_ecmascript_whitespace(question)),
            "scope_digest": null
        });
        format!("unk_{}", hash_canonical(&identity).unwrap())
    }

    #[allow(clippy::too_many_arguments)]
    fn unknown_revision(
        question: &str,
        domain: &str,
        revision: u64,
        previous_revision_digest: Option<&str>,
        relationships: Value,
        status: &str,
        supporting_evidence_ids: Vec<String>,
        contradicting_evidence_ids: Vec<String>,
        resolution: Value,
        required_authority: Value,
        required_confidence: &str,
    ) -> Value {
        let evidence_id = record()["evidence_id"].as_str().unwrap().to_owned();
        let mut unknown = json!({
            "unknown_id": unknown_id(question, domain),
            "revision": revision,
            "previous_revision_digest": previous_revision_digest,
            "revision_digest": "0".repeat(64),
            "scope_digest": null,
            "question": question,
            "status": status,
            "severity": "medium",
            "domain": domain,
            "supporting_evidence_ids": supporting_evidence_ids,
            "contradicting_evidence_ids": contradicting_evidence_ids,
            "required_authority": required_authority,
            "required_confidence": required_confidence,
            "required_environment": null,
            "recommended_probes": [],
            "relationships": relationships,
            "resolution": resolution,
            "mutation_evidence_ids": [evidence_id]
        });
        let mut revision_payload = unknown.clone();
        revision_payload
            .as_object_mut()
            .unwrap()
            .remove("revision_digest");
        unknown["revision_digest"] = json!(hash_canonical(&revision_payload).unwrap());
        unknown
    }

    fn open_unknown(question: &str, relationships: Value) -> Value {
        unknown_revision(
            question,
            "archive",
            1,
            None,
            relationships,
            "open",
            vec![],
            vec![],
            Value::Null,
            Value::Null,
            "inferred",
        )
    }

    fn scoped_open_unknown(question: &str, scope_sha256: &str) -> Value {
        let mut unknown = open_unknown(question, json!([]));
        let normalized_question =
            collapse_ecmascript_whitespace(trim_ecmascript_whitespace(question));
        let identity = json!({
            "domain": "archive",
            "question": normalized_question,
            "scope_digest": scope_sha256,
        });
        unknown["scope_digest"] = json!(scope_sha256);
        unknown["unknown_id"] = json!(format!("unk_{}", hash_canonical(&identity).unwrap()));
        let mut revision_payload = unknown.clone();
        revision_payload
            .as_object_mut()
            .unwrap()
            .remove("revision_digest");
        unknown["revision_digest"] = json!(hash_canonical(&revision_payload).unwrap());
        unknown
    }

    fn verified_unknown(question: &str, required_authority: Value) -> Value {
        let evidence_id = record()["evidence_id"].as_str().unwrap().to_owned();
        unknown_revision(
            question,
            "archive",
            1,
            None,
            json!([]),
            "resolved",
            vec![evidence_id.clone()],
            vec![],
            json!({
                "disposition": "verified",
                "rationale": "The observation satisfies the declared requirement",
                "evidence_ids": [evidence_id]
            }),
            required_authority,
            "observed",
        )
    }

    fn bundle(records: Vec<Value>, unknowns: Vec<Value>) -> Value {
        create_evidence_bundle(&records, &unknowns).unwrap()
    }

    #[test]
    fn verifies_canonical_manifest_projection_and_separate_digests() {
        let evidence = record();
        let unknown = open_unknown("Archive member metadata is complete", json!([]));
        let input = bundle(vec![evidence], vec![unknown]);
        let verified = verify_evidence_bundle_value(&input).unwrap();

        assert_eq!(verified.record_count, 1);
        assert_eq!(verified.unknown_revision_count, 1);
        assert_eq!(verified.raw_file_sha256, None);
        assert_eq!(
            verified.canonical_sha256,
            sha256_hex(&canonical_jcs_bytes(&verified.bundle).unwrap())
        );

        let raw = serde_json::to_vec(&input).unwrap();
        let from_json = verify_evidence_bundle_json(&raw).unwrap();
        let expected_raw_sha256 = sha256_hex(&raw);
        assert_eq!(
            from_json.raw_file_sha256.as_deref(),
            Some(expected_raw_sha256.as_str())
        );
        assert_eq!(from_json.canonical_sha256, verified.canonical_sha256);
        assert_ne!(
            from_json.raw_file_sha256.as_deref(),
            Some(from_json.canonical_sha256.as_str())
        );
    }

    #[test]
    fn projects_only_unknown_histories_scoped_to_the_selected_artifact() {
        let evidence = record();
        let target_sha = evidence["subject"]["digest"]["sha256"]
            .as_str()
            .unwrap()
            .to_owned();
        let unknown = scoped_open_unknown(
            "APK signing block region needs independent inspection",
            &target_sha,
        );
        let input = bundle(vec![evidence], vec![unknown]);

        let selected = evidence_bundle_for_target(&input, &target_sha).unwrap();
        assert_eq!(selected["records"].as_array().unwrap().len(), 1);
        assert_eq!(selected["unknowns"].as_array().unwrap().len(), 1);
        assert_eq!(
            selected["unknowns"][0]["scope_digest"].as_str(),
            Some(target_sha.as_str())
        );
        assert!(verify_evidence_bundle_value(&selected).is_ok());

        let other_target = "b".repeat(64);
        let empty = evidence_bundle_for_target(&input, &other_target).unwrap();
        assert!(empty["records"].as_array().unwrap().is_empty());
        assert!(empty["unknowns"].as_array().unwrap().is_empty());
        assert!(verify_evidence_bundle_value(&empty).is_ok());
    }

    #[test]
    fn target_projection_rejects_histories_with_cross_artifact_support_evidence() {
        let target_evidence = record();
        let target_sha = target_evidence["subject"]["digest"]["sha256"]
            .as_str()
            .unwrap()
            .to_owned();

        let mut other_evidence = record();
        other_evidence["subject"]["digest"]["sha256"] = json!("b".repeat(64));
        other_evidence["evidence_id"] = json!(compute_evidence_id(&other_evidence).unwrap());
        let other_id = other_evidence["evidence_id"].as_str().unwrap().to_owned();

        let mut unknown = scoped_open_unknown(
            "APK support evidence must remain scoped to the target artifact",
            &target_sha,
        );
        unknown["supporting_evidence_ids"] = json!([other_id]);
        let mut revision_payload = unknown.clone();
        revision_payload
            .as_object_mut()
            .unwrap()
            .remove("revision_digest");
        unknown["revision_digest"] = json!(hash_canonical(&revision_payload).unwrap());

        let input = bundle(vec![target_evidence, other_evidence], vec![unknown]);
        let selected = evidence_bundle_for_target(&input, &target_sha).unwrap();
        assert_eq!(selected["records"].as_array().unwrap().len(), 1);
        assert_eq!(selected["unknowns"].as_array().unwrap().len(), 0);
        assert_eq!(
            selected["records"][0]["subject"]["digest"]["sha256"].as_str(),
            Some(target_sha.as_str())
        );
    }

    #[test]
    fn rejects_duplicate_evidence_records() {
        let evidence = record();
        let duplicate = bundle(vec![evidence.clone(), evidence], vec![]);
        let error = verify_evidence_bundle_value(&duplicate).unwrap_err();
        assert_eq!(error.kind, BundleErrorKind::DuplicateEvidenceId);
    }

    #[test]
    fn rejects_noncanonical_manifest_projection() {
        let mut input = bundle(vec![record()], vec![]);
        input["artifacts"][0]["format"] = json!("zip");
        let error = verify_evidence_bundle_value(&input).unwrap_err();
        assert_eq!(error.kind, BundleErrorKind::NonCanonicalBundle);
    }

    #[test]
    fn rejects_missing_evidence_references_from_unknown_history() {
        let mut unknown = open_unknown("A claim with missing support", json!([]));
        unknown["supporting_evidence_ids"] =
            json!(["ev_ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"]);
        let mut payload = unknown.clone();
        payload.as_object_mut().unwrap().remove("revision_digest");
        unknown["revision_digest"] = json!(hash_canonical(&payload).unwrap());

        let error =
            verify_evidence_bundle_value(&bundle(vec![record()], vec![unknown])).unwrap_err();
        assert_eq!(error.kind, BundleErrorKind::UnknownGraphInvalid);
        assert!(error.message.contains("references missing Evidence"));
    }

    #[test]
    fn rejects_revision_gaps_and_broken_predecessor_chains() {
        let first = open_unknown("A versioned unresolved claim", json!([]));
        let first_digest = first["revision_digest"].as_str().unwrap().to_owned();
        let third = unknown_revision(
            "A versioned unresolved claim",
            "archive",
            3,
            Some(&first_digest),
            json!([]),
            "investigating",
            vec![],
            vec![],
            Value::Null,
            Value::Null,
            "inferred",
        );
        let error =
            verify_evidence_bundle_value(&bundle(vec![record()], vec![first, third])).unwrap_err();
        assert_eq!(error.kind, BundleErrorKind::UnknownGraphInvalid);
        assert!(error.message.contains("revision history has a gap"));

        let first = open_unknown("A second revision chain", json!([]));
        let second = unknown_revision(
            "A second revision chain",
            "archive",
            2,
            Some(&"a".repeat(64)),
            json!([]),
            "investigating",
            vec![],
            vec![],
            Value::Null,
            Value::Null,
            "inferred",
        );
        let error =
            verify_evidence_bundle_value(&bundle(vec![record()], vec![first, second])).unwrap_err();
        assert_eq!(error.kind, BundleErrorKind::UnknownGraphInvalid);
        assert!(error.message.contains("revision chain is broken"));
    }

    #[test]
    fn rejects_dangling_relationships_and_dependency_cycles() {
        let dangling = open_unknown(
            "Unknown with a dangling relation",
            json!([{
                "type": "related-to",
                "unknown_id": format!("unk_{}", "f".repeat(64))
            }]),
        );
        let error =
            verify_evidence_bundle_value(&bundle(vec![record()], vec![dangling])).unwrap_err();
        assert_eq!(error.kind, BundleErrorKind::UnknownGraphInvalid);
        assert!(error.message.contains("references missing unknown"));

        let left_id = unknown_id("Left dependency", "archive");
        let right_id = unknown_id("Right dependency", "archive");
        let left = open_unknown(
            "Left dependency",
            json!([{"type": "depends-on", "unknown_id": right_id}]),
        );
        let right = open_unknown(
            "Right dependency",
            json!([{"type": "depends-on", "unknown_id": left_id}]),
        );
        let error =
            verify_evidence_bundle_value(&bundle(vec![record()], vec![left, right])).unwrap_err();
        assert_eq!(error.kind, BundleErrorKind::UnknownGraphInvalid);
        assert!(error.message.contains("dependency cycle"));
    }

    #[test]
    fn accepts_only_evidence_qualified_verified_resolutions() {
        let valid = verified_unknown("A claim that can be verified", Value::Null);
        assert!(verify_evidence_bundle_value(&bundle(vec![record()], vec![valid])).is_ok());

        let invalid = verified_unknown(
            "A claim requiring a different authority",
            json!("controlled-replay"),
        );
        let error =
            verify_evidence_bundle_value(&bundle(vec![record()], vec![invalid])).unwrap_err();
        assert_eq!(error.kind, BundleErrorKind::UnknownGraphInvalid);
        assert!(error.message.contains("no qualifying resolution Evidence"));
    }

    #[test]
    fn evidence_link_closure_is_a_separate_stricter_audit() {
        let mut evidence = record();
        evidence["evidence_links"] =
            json!(["ev_ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"]);
        evidence["evidence_id"] = json!(compute_evidence_id(&evidence).unwrap());
        let input = bundle(vec![evidence], vec![]);

        // Upstream REA bundle authentication does not require links to resolve
        // inside the same bundle; the stricter graph audit reports them.
        assert!(verify_evidence_bundle_value(&input).is_ok());
        let missing = audit_evidence_link_closure(&input).unwrap();
        assert_eq!(missing.len(), 1);
        assert_eq!(
            missing[0].1,
            "ev_ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        );
    }

    #[test]
    fn rejects_duplicate_keys_in_raw_bundle_json() {
        let error = verify_evidence_bundle_json(br#"{"records":[],"records":[]}"#).unwrap_err();
        assert_eq!(error.kind, BundleErrorKind::InvalidJson);
    }
}
