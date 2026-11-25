//! Deterministic structural, temporal and artifact-integrity checks for the
//! Claim-Evidence Integrity Benchmark v0.1.
//!
//! This validator checks fixture contracts and exact evidence bytes. It does
//! not decide whether a natural-language passage entails a claim.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;

const FIXTURE_PACK: &str = include_str!("../fixtures/claim-evidence-integrity-v0.1.json");

const VERDICTS: &[&str] = &[
    "SUPPORTED",
    "REFUTED",
    "MIXED",
    "UNDECIDABLE",
    "OUT_OF_SCOPE",
];
const RELATIONS: &[&str] = &[
    "supports",
    "refutes",
    "qualifies",
    "irrelevant",
    "unresolved",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationIssue {
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationReport {
    pub schema_version: String,
    pub fixture_count: usize,
    pub evidence_count: usize,
    pub verified_digest_count: usize,
}

fn issue(issues: &mut Vec<ValidationIssue>, path: impl Into<String>, message: impl Into<String>) {
    issues.push(ValidationIssue {
        path: path.into(),
        message: message.into(),
    });
}

fn object<'a>(
    value: &'a Value,
    path: &str,
    issues: &mut Vec<ValidationIssue>,
) -> Option<&'a serde_json::Map<String, Value>> {
    match value.as_object() {
        Some(map) => Some(map),
        None => {
            issue(issues, path, "expected an object");
            None
        }
    }
}

fn required_string<'a>(
    map: &'a serde_json::Map<String, Value>,
    key: &str,
    path: &str,
    issues: &mut Vec<ValidationIssue>,
) -> Option<&'a str> {
    match map.get(key).and_then(Value::as_str) {
        Some(value) if !value.trim().is_empty() => Some(value),
        _ => {
            issue(issues, format!("{path}.{key}"), "required non-empty string");
            None
        }
    }
}

fn required_array<'a>(
    map: &'a serde_json::Map<String, Value>,
    key: &str,
    path: &str,
    issues: &mut Vec<ValidationIssue>,
) -> Option<&'a Vec<Value>> {
    match map.get(key).and_then(Value::as_array) {
        Some(value) => Some(value),
        None => {
            issue(issues, format!("{path}.{key}"), "required array");
            None
        }
    }
}

fn validate_exact_keys(
    map: &serde_json::Map<String, Value>,
    required: &[&str],
    path: &str,
    issues: &mut Vec<ValidationIssue>,
) {
    for key in required {
        if !map.contains_key(*key) {
            issue(issues, format!("{path}.{key}"), "required field is missing");
        }
    }
    for key in map.keys() {
        if !required.contains(&key.as_str()) {
            issue(issues, format!("{path}.{key}"), "unexpected field");
        }
    }
}

fn valid_time(value: &str) -> bool {
    value.contains('T')
        && (value.ends_with('Z')
            || value.as_bytes().get(value.len().saturating_sub(6)) == Some(&b'+')
            || value.as_bytes().get(value.len().saturating_sub(6)) == Some(&b'-'))
}

fn sha256_hex(text: &str) -> String {
    let digest = Sha256::digest(text.as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn validate_evidence(
    value: &Value,
    path: &str,
    issues: &mut Vec<ValidationIssue>,
) -> bool {
    let Some(map) = object(value, path, issues) else {
        return false;
    };
    const KEYS: &[&str] = &[
        "evidence_id",
        "source_id",
        "source_family_id",
        "source_role",
        "canonical_url",
        "publication_time",
        "retrieval_time",
        "content_integrity",
        "artifact_text",
        "artifact_sha256",
        "quote_span",
        "relation_to_claim",
        "correction_status",
        "relationship_notes",
    ];
    validate_exact_keys(map, KEYS, path, issues);

    let evidence_id = required_string(map, "evidence_id", path, issues);
    required_string(map, "source_id", path, issues);
    required_string(map, "source_family_id", path, issues);
    let source_role = required_string(map, "source_role", path, issues);
    let url = required_string(map, "canonical_url", path, issues);
    let publication = required_string(map, "publication_time", path, issues);
    let retrieval = required_string(map, "retrieval_time", path, issues);
    let integrity = required_string(map, "content_integrity", path, issues);
    let artifact = required_string(map, "artifact_text", path, issues);
    let digest = required_string(map, "artifact_sha256", path, issues);
    let relation = required_string(map, "relation_to_claim", path, issues);
    let correction = required_string(map, "correction_status", path, issues);
    required_string(map, "relationship_notes", path, issues);

    if let Some(role) = source_role {
        if !["primary", "secondary", "registry", "syndicated", "unknown"].contains(&role) {
            issue(issues, format!("{path}.source_role"), "unknown source role");
        }
    }
    if let Some(url) = url {
        if !(url.starts_with("https://") || url.starts_with("http://")) {
            issue(issues, format!("{path}.canonical_url"), "URL must use HTTP(S)");
        }
    }
    if let Some(time) = publication {
        if !valid_time(time) {
            issue(issues, format!("{path}.publication_time"), "expected RFC3339 timestamp");
        }
    }
    if let Some(time) = retrieval {
        if !valid_time(time) {
            issue(issues, format!("{path}.retrieval_time"), "expected RFC3339 timestamp");
        }
    }
    if let Some(integrity) = integrity {
        if !["clean", "compromised", "missing", "unknown"].contains(&integrity) {
            issue(issues, format!("{path}.content_integrity"), "unknown integrity state");
        }
    }
    if let Some(relation) = relation {
        if !RELATIONS.contains(&relation) {
            issue(issues, format!("{path}.relation_to_claim"), "unknown evidence relation");
        }
    }
    if let Some(correction) = correction {
        if !["none", "corrected", "retracted", "superseded"].contains(&correction) {
            issue(issues, format!("{path}.correction_status"), "unknown correction state");
        }
    }

    let mut digest_ok = false;
    if let (Some(artifact), Some(digest)) = (artifact, digest) {
        let actual = sha256_hex(artifact);
        if digest != actual {
            issue(
                issues,
                format!("{path}.artifact_sha256"),
                format!("digest mismatch; computed {actual}"),
            );
        } else {
            digest_ok = true;
        }
    }

    if let Some(span_value) = map.get("quote_span") {
        if let Some(span) = object(span_value, &format!("{path}.quote_span"), issues) {
            const SPAN_KEYS: &[&str] = &["byte_start", "byte_end", "quote"];
            validate_exact_keys(span, SPAN_KEYS, &format!("{path}.quote_span"), issues);
            let start = span.get("byte_start").and_then(Value::as_u64).map(|n| n as usize);
            let end = span.get("byte_end").and_then(Value::as_u64).map(|n| n as usize);
            let quote = required_string(span, "quote", &format!("{path}.quote_span"), issues);
            match (artifact, start, end, quote) {
                (Some(text), Some(start), Some(end), Some(quote)) => {
                    let bytes = text.as_bytes();
                    if start >= end || end > bytes.len() {
                        issue(issues, format!("{path}.quote_span"), "byte range is invalid");
                    } else if std::str::from_utf8(&bytes[start..end]).ok() != Some(quote) {
                        issue(
                            issues,
                            format!("{path}.quote_span"),
                            "quoted span does not match exact artifact bytes",
                        );
                    }
                }
                _ => issue(
                    issues,
                    format!("{path}.quote_span"),
                    "byte offsets and artifact text are required",
                ),
            }
        }
    } else {
        issue(issues, format!("{path}.quote_span"), "required quote span is missing");
    }

    let _ = evidence_id;
    digest_ok
}

fn validate_fixture(
    value: &Value,
    index: usize,
    seen_ids: &mut HashSet<String>,
    evidence_count: &mut usize,
    verified_digests: &mut usize,
    issues: &mut Vec<ValidationIssue>,
) {
    let path = format!("fixtures[{index}]");
    let Some(map) = object(value, &path, issues) else {
        return;
    };
    const KEYS: &[&str] = &[
        "fixture_id",
        "family",
        "claim",
        "claim_scope",
        "label_provenance",
        "claim_reference_time",
        "verification_time",
        "model_knowledge_cutoff",
        "evidence",
        "expected_verdict",
        "required_invariants",
        "description",
        "status",
    ];
    validate_exact_keys(map, KEYS, &path, issues);

    if let Some(id) = required_string(map, "fixture_id", &path, issues) {
        if !seen_ids.insert(id.to_owned()) {
            issue(issues, format!("{path}.fixture_id"), "duplicate fixture ID");
        }
    }
    required_string(map, "family", &path, issues);
    required_string(map, "claim", &path, issues);
    required_string(map, "description", &path, issues);

    if let Some(scope_value) = map.get("claim_scope") {
        if let Some(scope) = object(scope_value, &format!("{path}.claim_scope"), issues) {
            const SCOPE_KEYS: &[&str] = &[
                "population",
                "outcome_or_measure",
                "quantifier_or_effect",
                "domain_or_reference_boundary",
                "scope_note",
            ];
            validate_exact_keys(scope, SCOPE_KEYS, &format!("{path}.claim_scope"), issues);
            for key in SCOPE_KEYS {
                required_string(scope, key, &format!("{path}.claim_scope"), issues);
            }
        }
    } else {
        issue(issues, format!("{path}.claim_scope"), "required claim scope object is missing");
    }

    if let Some(provenance_value) = map.get("label_provenance") {
        if let Some(provenance) = object(
            provenance_value,
            &format!("{path}.label_provenance"),
            issues,
        ) {
            const PROVENANCE_KEYS: &[&str] = &[
                "label_status",
                "authority_id",
                "evidence_reference",
                "independent_of_candidate",
                "permitted_use",
            ];
            validate_exact_keys(
                provenance,
                PROVENANCE_KEYS,
                &format!("{path}.label_provenance"),
                issues,
            );
            if required_string(provenance, "label_status", &format!("{path}.label_provenance"), issues)
                != Some("SYNTHETIC_KNOWN_ANSWER")
            {
                issue(
                    issues,
                    format!("{path}.label_provenance.label_status"),
                    "only synthetic known-answer labels are allowed in v0.1",
                );
            }
            required_string(provenance, "authority_id", &format!("{path}.label_provenance"), issues);
            required_string(provenance, "evidence_reference", &format!("{path}.label_provenance"), issues);
            required_string(provenance, "permitted_use", &format!("{path}.label_provenance"), issues);
            if provenance.get("independent_of_candidate").and_then(Value::as_bool) != Some(true) {
                issue(
                    issues,
                    format!("{path}.label_provenance.independent_of_candidate"),
                    "known-answer labels must remain evaluator-only",
                );
            }
        }
    } else {
        issue(
            issues,
            format!("{path}.label_provenance"),
            "synthetic label provenance is required",
        );
    }

    for key in [
        "claim_reference_time",
        "verification_time",
        "model_knowledge_cutoff",
    ] {
        if let Some(time) = required_string(map, key, &path, issues) {
            if !valid_time(time) {
                issue(issues, format!("{path}.{key}"), "expected RFC3339 timestamp");
            }
        }
    }

    if let Some(verdict) = required_string(map, "expected_verdict", &path, issues) {
        if !VERDICTS.contains(&verdict) {
            issue(issues, format!("{path}.expected_verdict"), "unknown verdict");
        }
    }
    if required_string(map, "status", &path, issues)
        != Some("SYNTHETIC_KNOWN_ANSWER_NOT_EXPERT_GOLD")
    {
        issue(issues, format!("{path}.status"), "fixture must retain synthetic-only status");
    }

    if let Some(invariants) = required_array(map, "required_invariants", &path, issues) {
        if invariants.is_empty() {
            issue(issues, format!("{path}.required_invariants"), "at least one invariant is required");
        }
        for (i, invariant) in invariants.iter().enumerate() {
            if invariant.as_str().is_none_or(|s| s.trim().is_empty()) {
                issue(
                    issues,
                    format!("{path}.required_invariants[{i}]"),
                    "invariant must be a non-empty string",
                );
            }
        }
    }

    if let Some(evidence) = required_array(map, "evidence", &path, issues) {
        *evidence_count += evidence.len();
        let mut evidence_ids = HashSet::new();
        for (i, item) in evidence.iter().enumerate() {
            if let Some(item_map) = item.as_object() {
                if let Some(id) = item_map.get("evidence_id").and_then(Value::as_str) {
                    if !evidence_ids.insert(id.to_owned()) {
                        issue(
                            issues,
                            format!("{path}.evidence[{i}].evidence_id"),
                            "duplicate evidence ID within fixture",
                        );
                    }
                }
            }
            if validate_evidence(item, &format!("{path}.evidence[{i}]"), issues) {
                *verified_digests += 1;
            }
        }
    }
}

/// Validate a serialized v0.1 fixture pack.
pub fn validate_pack(input: &str) -> Result<ValidationReport, Vec<ValidationIssue>> {
    let value: Value = serde_json::from_str(input).map_err(|error| {
        vec![ValidationIssue {
            path: "$".to_owned(),
            message: format!("invalid JSON: {error}"),
        }]
    })?;
    validate_value(&value)
}

/// Validate a parsed fixture pack.
pub fn validate_value(value: &Value) -> Result<ValidationReport, Vec<ValidationIssue>> {
    let mut issues = Vec::new();
    let Some(map) = object(value, "$", &mut issues) else {
        return Err(issues);
    };
    const KEYS: &[&str] = &[
        "schema_version",
        "status",
        "created_at",
        "source_issue",
        "purpose",
        "verdicts",
        "relation_labels",
        "temporal_axes",
        "evidence_digest_algorithm",
        "fixtures",
        "evaluation_policy",
        "limitations",
    ];
    validate_exact_keys(map, KEYS, "$", &mut issues);

    let schema_version = required_string(map, "schema_version", "$", &mut issues)
        .unwrap_or_default()
        .to_owned();
    if schema_version != "symthaea-claim-evidence-integrity-v0.1" {
        issue(&mut issues, "$.schema_version", "unsupported schema version");
    }
    if required_string(map, "status", "$", &mut issues)
        != Some("SYNTHETIC_KNOWN_ANSWER_FIXTURES_NOT_EXPERT_GOLD")
    {
        issue(&mut issues, "$.status", "pack must retain synthetic-only status");
    }
    if let Some(created_at) = required_string(map, "created_at", "$", &mut issues) {
        if !valid_time(created_at) {
            issue(&mut issues, "$.created_at", "expected RFC3339 timestamp");
        }
    }
    required_string(map, "purpose", "$", &mut issues);
    if map.get("source_issue").and_then(Value::as_u64) != Some(13635) {
        issue(&mut issues, "$.source_issue", "unexpected source issue");
    }
    if required_string(map, "evidence_digest_algorithm", "$", &mut issues)
        != Some("SHA-256 over artifact_text UTF-8 bytes; fixture strings are ASCII so byte offsets are unambiguous.")
    {
        issue(
            &mut issues,
            "$.evidence_digest_algorithm",
            "unsupported or missing artifact digest contract",
        );
    }

    let expected_verdicts: HashSet<&str> = VERDICTS.iter().copied().collect();
    if let Some(values) = required_array(map, "verdicts", "$", &mut issues) {
        let actual: HashSet<&str> = values.iter().filter_map(Value::as_str).collect();
        if actual != expected_verdicts || actual.len() != values.len() {
            issue(&mut issues, "$.verdicts", "declared verdicts must exactly match supported verdict enum");
        }
    }
    let expected_relations: HashSet<&str> = RELATIONS.iter().copied().collect();
    if let Some(values) = required_array(map, "relation_labels", "$", &mut issues) {
        let actual: HashSet<&str> = values.iter().filter_map(Value::as_str).collect();
        if actual != expected_relations || actual.len() != values.len() {
            issue(&mut issues, "$.relation_labels", "declared relation labels must exactly match supported relation enum");
        }
    }
    let expected_axes: HashSet<&str> = [
        "claim_reference_time",
        "evidence_publication_time",
        "evidence_retrieval_time",
        "model_knowledge_cutoff",
    ]
    .into_iter()
    .collect();
    if let Some(values) = required_array(map, "temporal_axes", "$", &mut issues) {
        let actual: HashSet<&str> = values.iter().filter_map(Value::as_str).collect();
        if actual != expected_axes || actual.len() != values.len() {
            issue(&mut issues, "$.temporal_axes", "all four temporal axes must be declared exactly once");
        }
    }

    if let Some(policy_value) = map.get("evaluation_policy") {
        if let Some(policy) = object(policy_value, "$.evaluation_policy", &mut issues) {
            const POLICY_KEYS: &[&str] = &[
                "denominator",
                "score_separation",
                "primary_contract",
                "no_single_scalar",
                "oracle_custody",
            ];
            validate_exact_keys(policy, POLICY_KEYS, "$.evaluation_policy", &mut issues);
            for key in ["denominator", "primary_contract", "no_single_scalar", "oracle_custody"] {
                required_string(policy, key, "$.evaluation_policy", &mut issues);
            }
            if let Some(scores) = required_array(policy, "score_separation", "$.evaluation_policy", &mut issues) {
                let names: Vec<&str> = scores.iter().filter_map(Value::as_str).collect();
                let unique: HashSet<&str> = names.iter().copied().collect();
                if names.len() < 8 || unique.len() != names.len() || names.len() != scores.len() {
                    issue(
                        &mut issues,
                        "$.evaluation_policy.score_separation",
                        "at least eight unique non-empty score dimensions are required",
                    );
                }
                if names.iter().any(|name| name.trim().is_empty()) {
                    issue(
                        &mut issues,
                        "$.evaluation_policy.score_separation",
                        "score dimensions cannot be empty",
                    );
                }
            }
        }
    } else {
        issue(&mut issues, "$.evaluation_policy", "required object is missing");
    }
    if let Some(limitations) = required_array(map, "limitations", "$", &mut issues) {
        if limitations.len() < 3
            || limitations
                .iter()
                .any(|item| item.as_str().is_none_or(|value| value.trim().is_empty()))
        {
            issue(&mut issues, "$.limitations", "at least three non-empty limitations are required");
        }
    }

    let mut seen_ids = HashSet::new();
    let mut evidence_count = 0;
    let mut verified_digests = 0;
    if let Some(fixtures) = required_array(map, "fixtures", "$", &mut issues) {
        if fixtures.len() < 12 {
            issue(&mut issues, "$.fixtures", "at least twelve fixtures are required");
        }
        for (index, fixture) in fixtures.iter().enumerate() {
            validate_fixture(
                fixture,
                index,
                &mut seen_ids,
                &mut evidence_count,
                &mut verified_digests,
                &mut issues,
            );
        }
    }
    if !issues.is_empty() {
        return Err(issues);
    }

    Ok(ValidationReport {
        schema_version,
        fixture_count: seen_ids.len(),
        evidence_count,
        verified_digest_count: verified_digests,
    })
}

/// Run deterministic negative controls against the checked-in pack.
pub fn self_test() -> Result<ValidationReport, String> {
    let valid = validate_pack(FIXTURE_PACK).map_err(|issues| format!("fixture pack invalid: {issues:?}"))?;
    let mut tampered: Value =
        serde_json::from_str(FIXTURE_PACK).map_err(|error| error.to_string())?;
    tampered["fixtures"][0]["evidence"][0]["artifact_text"] =
        Value::String("mutated evidence that no longer matches its digest".to_owned());
    if validate_value(&tampered).is_ok() {
        return Err("tampered artifact unexpectedly passed validation".to_owned());
    }

    let mut bad_span: Value =
        serde_json::from_str(FIXTURE_PACK).map_err(|error| error.to_string())?;
    bad_span["fixtures"][0]["evidence"][0]["quote_span"]["byte_start"] = Value::from(1);
    if validate_value(&bad_span).is_ok() {
        return Err("tampered quote span unexpectedly passed validation".to_owned());
    }

    let mut duplicate: Value =
        serde_json::from_str(FIXTURE_PACK).map_err(|error| error.to_string())?;
    duplicate["fixtures"][1]["fixture_id"] = duplicate["fixtures"][0]["fixture_id"].clone();
    if validate_value(&duplicate).is_ok() {
        return Err("duplicate fixture ID unexpectedly passed validation".to_owned());
    }

    Ok(valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_in_pack_passes_digest_and_span_validation() {
        let report = validate_pack(FIXTURE_PACK).unwrap();
        assert_eq!(report.fixture_count, 15);
        assert_eq!(report.evidence_count, 23);
        assert_eq!(report.verified_digest_count, 23);
    }

    #[test]
    fn corrupted_artifact_digest_fails_closed() {
        let mut value: Value = serde_json::from_str(FIXTURE_PACK).unwrap();
        value["fixtures"][0]["evidence"][0]["artifact_text"] =
            Value::String("tampered".to_owned());
        assert!(validate_value(&value).is_err());
    }

    #[test]
    fn mismatched_quote_span_fails_closed() {
        let mut value: Value = serde_json::from_str(FIXTURE_PACK).unwrap();
        value["fixtures"][0]["evidence"][0]["quote_span"]["byte_start"] = Value::from(1);
        assert!(validate_value(&value).is_err());
    }

    #[test]
    fn duplicate_fixture_ids_fail_closed() {
        let mut value: Value = serde_json::from_str(FIXTURE_PACK).unwrap();
        value["fixtures"][1]["fixture_id"] = value["fixtures"][0]["fixture_id"].clone();
        assert!(validate_value(&value).is_err());
    }

    #[test]
    fn removing_a_required_context_axis_fails_closed() {
        let mut value: Value = serde_json::from_str(FIXTURE_PACK).unwrap();
        value["fixtures"][0].as_object_mut().unwrap().remove("claim_reference_time");
        assert!(validate_value(&value).is_err());
    }
    #[test]
    fn unsupported_relation_enum_fails_closed() {
        let mut value: Value = serde_json::from_str(FIXTURE_PACK).unwrap();
        value["fixtures"][0]["evidence"][0]["relation_to_claim"] =
            Value::String("weakly_relevant".to_owned());
        assert!(validate_value(&value).is_err());
    }

    #[test]
    fn missing_top_level_temporal_axis_fails_closed() {
        let mut value: Value = serde_json::from_str(FIXTURE_PACK).unwrap();
        let axes = value["temporal_axes"].as_array_mut().unwrap();
        axes.retain(|axis| axis.as_str() != Some("model_knowledge_cutoff"));
        assert!(validate_value(&value).is_err());
    }

    #[test]
    fn duplicate_evidence_ids_fail_closed() {
        let mut value: Value = serde_json::from_str(FIXTURE_PACK).unwrap();
        let duplicate = value["fixtures"][0]["evidence"][0].clone();
        value["fixtures"][0]["evidence"]
            .as_array_mut()
            .unwrap()
            .push(duplicate);
        assert!(validate_value(&value).is_err());
    }
}
