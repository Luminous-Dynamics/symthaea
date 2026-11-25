//! Deterministic structural, temporal and artifact-integrity checks for the
//! Claim-Evidence Integrity Benchmark v0.1.
//!
//! This validator checks fixture contracts and exact evidence bytes. It does
//! not decide whether a natural-language passage entails a claim.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;

const FIXTURE_PACK: &str = include_str!("../fixtures/claim-evidence-integrity-v0.1.json");
const CANDIDATE_INPUTS: &str = include_str!("../fixtures/candidate-inputs-v0.1.json");
const EVALUATOR_KEY: &str = include_str!("../fixtures/evaluator-key-v0.1.json");

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
    let bytes = value.as_bytes();
    if bytes.len() != 20 && bytes.len() != 25 {
        return false;
    }
    if bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
    {
        return false;
    }
    let number = |range: std::ops::Range<usize>| -> Option<u32> {
        let slice = bytes.get(range)?;
        if !slice.iter().all(u8::is_ascii_digit) {
            return None;
        }
        std::str::from_utf8(slice).ok()?.parse().ok()
    };
    let (Some(year), Some(month), Some(day), Some(hour), Some(minute), Some(second)) = (
        number(0..4),
        number(5..7),
        number(8..10),
        number(11..13),
        number(14..16),
        number(17..19),
    ) else {
        return false;
    };
    if year == 0 || !(1..=12).contains(&month) || !(0..=23).contains(&hour)
        || !(0..=59).contains(&minute) || !(0..=59).contains(&second)
    {
        return false;
    }
    let leap_year = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap_year => 29,
        2 => 28,
        _ => return false,
    };
    if day == 0 || day > days_in_month {
        return false;
    }

    if bytes.len() == 20 {
        return bytes[19] == b'Z';
    }
    if bytes[19] != b'+' && bytes[19] != b'-' || bytes[22] != b':' {
        return false;
    }
    let (Some(offset_hour), Some(offset_minute)) = (number(20..22), number(23..25)) else {
        return false;
    };
    offset_hour <= 23 && offset_minute <= 59
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

    required_string(map, "evidence_id", path, issues);
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
    if !map.get("relationship_notes").is_some_and(Value::is_string) {
        issue(
            issues,
            format!("{path}.relationship_notes"),
            "relationship_notes must be a string (empty is allowed)",
        );
    }

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeparatedViewReport {
    pub fixture_count: usize,
    pub evidence_count: usize,
    pub verified_digest_count: usize,
    pub verified_quote_count: usize,
}

/// Validate the candidate-visible inputs and evaluator-only key as a pair.
///
/// Exact field allowlists prevent evaluator labels from entering candidate
/// inputs; ID/digest binding prevents the key from being detached from the
/// artifact bytes that the candidate actually received.
pub fn validate_separated_views(
    candidate_json: &str,
    evaluator_json: &str,
) -> Result<SeparatedViewReport, Vec<ValidationIssue>> {
    let candidate: Value = serde_json::from_str(candidate_json).map_err(|error| {
        vec![ValidationIssue {
            path: "candidate".to_owned(),
            message: format!("invalid candidate JSON: {error}"),
        }]
    })?;
    let evaluator: Value = serde_json::from_str(evaluator_json).map_err(|error| {
        vec![ValidationIssue {
            path: "evaluator".to_owned(),
            message: format!("invalid evaluator JSON: {error}"),
        }]
    })?;
    let mut issues = Vec::new();

    let Some(candidate_map) = object(&candidate, "candidate", &mut issues) else {
        return Err(issues);
    };
    const CANDIDATE_KEYS: &[&str] = &["schema_version", "status", "fixtures"];
    validate_exact_keys(candidate_map, CANDIDATE_KEYS, "candidate", &mut issues);
    if required_string(candidate_map, "schema_version", "candidate", &mut issues)
        != Some("symthaea-claim-evidence-candidate-input-v0.1")
    {
        issue(&mut issues, "candidate.schema_version", "unsupported candidate schema");
    }
    if required_string(candidate_map, "status", "candidate", &mut issues)
        != Some("SYNTHETIC_CANDIDATE_INPUT_NOT_EXPERT_DATA")
    {
        issue(&mut issues, "candidate.status", "candidate view must retain synthetic-only status");
    }

    let Some(evaluator_map) = object(&evaluator, "evaluator", &mut issues) else {
        return Err(issues);
    };
    const EVALUATOR_KEYS: &[&str] = &["schema_version", "status", "fixtures"];
    validate_exact_keys(evaluator_map, EVALUATOR_KEYS, "evaluator", &mut issues);
    if required_string(evaluator_map, "schema_version", "evaluator", &mut issues)
        != Some("symthaea-claim-evidence-evaluator-key-v0.1")
    {
        issue(&mut issues, "evaluator.schema_version", "unsupported evaluator schema");
    }
    if required_string(evaluator_map, "status", "evaluator", &mut issues)
        != Some("SYNTHETIC_EVALUATOR_ONLY_NOT_EXPERT_GOLD")
    {
        issue(&mut issues, "evaluator.status", "evaluator key must remain evaluator-only");
    }

    let candidate_fixtures = required_array(candidate_map, "fixtures", "candidate", &mut issues);
    let evaluator_fixtures = required_array(evaluator_map, "fixtures", "evaluator", &mut issues);
    let (Some(candidate_fixtures), Some(evaluator_fixtures)) =
        (candidate_fixtures, evaluator_fixtures)
    else {
        return Err(issues);
    };
    if candidate_fixtures.len() < 12 || candidate_fixtures.len() != evaluator_fixtures.len() {
        issue(
            &mut issues,
            "fixtures",
            "candidate and evaluator views must contain the same 12+ fixtures",
        );
    }

    const CANDIDATE_FIXTURE_KEYS: &[&str] = &[
        "fixture_id",
        "claim",
        "claim_scope",
        "claim_reference_time",
        "verification_time",
        "model_knowledge_cutoff",
        "evidence",
    ];
    const KEY_FIXTURE_KEYS: &[&str] = &[
        "fixture_id",
        "expected_verdict",
        "required_invariants",
        "label_provenance",
        "evidence_labels",
    ];
    const CANDIDATE_EVIDENCE_KEYS: &[&str] = &[
        "evidence_id",
        "source_id",
        "canonical_url",
        "publication_time",
        "retrieval_time",
        "artifact_text",
        "artifact_sha256",
    ];
    const KEY_EVIDENCE_KEYS: &[&str] = &[
        "evidence_id",
        "artifact_sha256",
        "source_family_id",
        "source_role",
        "content_integrity",
        "quote_span",
        "relation_to_claim",
        "correction_status",
        "relationship_notes",
    ];

    let mut seen_fixture_ids = HashSet::new();
    let mut fixture_count = 0;
    let mut evidence_count = 0;
    let mut verified_digest_count = 0;
    let mut verified_quote_count = 0;

    for index in 0..candidate_fixtures.len().min(evaluator_fixtures.len()) {
        let path = format!("fixtures[{index}]");
        let Some(candidate_fixture) = object(
            &candidate_fixtures[index],
            &format!("candidate.{path}"),
            &mut issues,
        ) else {
            continue;
        };
        let Some(evaluator_fixture) = object(
            &evaluator_fixtures[index],
            &format!("evaluator.{path}"),
            &mut issues,
        ) else {
            continue;
        };
        validate_exact_keys(
            candidate_fixture,
            CANDIDATE_FIXTURE_KEYS,
            &format!("candidate.{path}"),
            &mut issues,
        );
        validate_exact_keys(
            evaluator_fixture,
            KEY_FIXTURE_KEYS,
            &format!("evaluator.{path}"),
            &mut issues,
        );
        let candidate_id = required_string(
            candidate_fixture,
            "fixture_id",
            &format!("candidate.{path}"),
            &mut issues,
        );
        let evaluator_id = required_string(
            evaluator_fixture,
            "fixture_id",
            &format!("evaluator.{path}"),
            &mut issues,
        );
        if candidate_id != evaluator_id {
            issue(
                &mut issues,
                &path,
                "candidate/evaluator fixture IDs or ordering do not match",
            );
        }
        if let Some(id) = candidate_id {
            if !seen_fixture_ids.insert(id.to_owned()) {
                issue(
                    &mut issues,
                    format!("candidate.{path}.fixture_id"),
                    "duplicate fixture ID",
                );
            }
        }
        required_string(candidate_fixture, "claim", &format!("candidate.{path}"), &mut issues);
        if let Some(scope_value) = candidate_fixture.get("claim_scope") {
            if let Some(scope) = object(scope_value, &format!("candidate.{path}.claim_scope"), &mut issues) {
                const SCOPE_KEYS: &[&str] = &[
                    "population", "outcome_or_measure", "quantifier_or_effect",
                    "domain_or_reference_boundary", "scope_note",
                ];
                validate_exact_keys(scope, SCOPE_KEYS, &format!("candidate.{path}.claim_scope"), &mut issues);
                for key in SCOPE_KEYS {
                    required_string(scope, key, &format!("candidate.{path}.claim_scope"), &mut issues);
                }
            }
        } else {
            issue(&mut issues, format!("candidate.{path}.claim_scope"), "missing claim scope");
        }
        for key in ["claim_reference_time", "verification_time", "model_knowledge_cutoff"] {
            if let Some(time) = required_string(candidate_fixture, key, &format!("candidate.{path}"), &mut issues) {
                if !valid_time(time) {
                    issue(&mut issues, format!("candidate.{path}.{key}"), "invalid RFC3339 timestamp");
                }
            }
        }
        if let Some(invariants) = required_array(evaluator_fixture, "required_invariants", &format!("evaluator.{path}"), &mut issues) {
            if invariants.is_empty() || invariants.iter().any(|item| item.as_str().is_none_or(|value| value.trim().is_empty())) {
                issue(&mut issues, format!("evaluator.{path}.required_invariants"), "required invariants must be non-empty strings");
            }
        }
        if let Some(verdict) =
            required_string(evaluator_fixture, "expected_verdict", &format!("evaluator.{path}"), &mut issues)
        {
            if !VERDICTS.contains(&verdict) {
                issue(
                    &mut issues,
                    format!("evaluator.{path}.expected_verdict"),
                    "unsupported expected verdict",
                );
            }
        }
        if let Some(provenance_value) = evaluator_fixture.get("label_provenance") {
            if let Some(provenance) = object(
                provenance_value,
                &format!("evaluator.{path}.label_provenance"),
                &mut issues,
            ) {
                if provenance.get("independent_of_candidate").and_then(Value::as_bool) != Some(true)
                    || provenance.get("label_status").and_then(Value::as_str)
                        != Some("SYNTHETIC_KNOWN_ANSWER")
                {
                    issue(
                        &mut issues,
                        format!("evaluator.{path}.label_provenance"),
                        "synthetic labels must remain evaluator-only",
                    );
                }
            }
        } else {
            issue(
                &mut issues,
                format!("evaluator.{path}.label_provenance"),
                "missing evaluator label provenance",
            );
        }

        let Some(candidate_evidence) = required_array(
            candidate_fixture,
            "evidence",
            &format!("candidate.{path}"),
            &mut issues,
        ) else {
            continue;
        };
        let Some(key_evidence) = required_array(
            evaluator_fixture,
            "evidence_labels",
            &format!("evaluator.{path}"),
            &mut issues,
        ) else {
            continue;
        };
        if candidate_evidence.len() != key_evidence.len() {
            issue(
                &mut issues,
                &path,
                "candidate and evaluator evidence counts do not match",
            );
        }
        let mut seen_evidence_ids = HashSet::new();

        for evidence_index in 0..candidate_evidence.len().min(key_evidence.len()) {
            let evidence_path = format!("{path}.evidence[{evidence_index}]");
            let Some(candidate_item) = object(
                &candidate_evidence[evidence_index],
                &format!("candidate.{evidence_path}"),
                &mut issues,
            ) else {
                continue;
            };
            let Some(key_item) = object(
                &key_evidence[evidence_index],
                &format!("evaluator.{evidence_path}"),
                &mut issues,
            ) else {
                continue;
            };
            validate_exact_keys(
                candidate_item,
                CANDIDATE_EVIDENCE_KEYS,
                &format!("candidate.{evidence_path}"),
                &mut issues,
            );
            validate_exact_keys(
                key_item,
                KEY_EVIDENCE_KEYS,
                &format!("evaluator.{evidence_path}"),
                &mut issues,
            );

            let candidate_evidence_id = required_string(
                candidate_item,
                "evidence_id",
                &format!("candidate.{evidence_path}"),
                &mut issues,
            );
            let key_evidence_id = required_string(
                key_item,
                "evidence_id",
                &format!("evaluator.{evidence_path}"),
                &mut issues,
            );
            if candidate_evidence_id != key_evidence_id {
                issue(
                    &mut issues,
                    &evidence_path,
                    "candidate/evaluator evidence IDs or ordering do not match",
                );
            }
            if let Some(id) = candidate_evidence_id {
                if !seen_evidence_ids.insert(id.to_owned()) {
                    issue(
                        &mut issues,
                        format!("candidate.{evidence_path}.evidence_id"),
                        "duplicate evidence ID within fixture",
                    );
                }
            }
            required_string(candidate_item, "source_id", &format!("candidate.{evidence_path}"), &mut issues);
            if let Some(url) = required_string(candidate_item, "canonical_url", &format!("candidate.{evidence_path}"), &mut issues) {
                if !(url.starts_with("https://") || url.starts_with("http://")) {
                    issue(&mut issues, format!("candidate.{evidence_path}.canonical_url"), "URL must use HTTP(S)");
                }
            }
            for key in ["publication_time", "retrieval_time"] {
                if let Some(time) = required_string(candidate_item, key, &format!("candidate.{evidence_path}"), &mut issues) {
                    if !valid_time(time) {
                        issue(&mut issues, format!("candidate.{evidence_path}.{key}"), "invalid RFC3339 timestamp");
                    }
                }
            }
            let artifact = required_string(
                candidate_item,
                "artifact_text",
                &format!("candidate.{evidence_path}"),
                &mut issues,
            );
            let candidate_digest = required_string(
                candidate_item,
                "artifact_sha256",
                &format!("candidate.{evidence_path}"),
                &mut issues,
            );
            let key_digest = required_string(
                key_item,
                "artifact_sha256",
                &format!("evaluator.{evidence_path}"),
                &mut issues,
            );
            if candidate_digest != key_digest {
                issue(
                    &mut issues,
                    &evidence_path,
                    "candidate artifact digest and evaluator binding differ",
                );
            }
            if let (Some(artifact), Some(digest)) = (artifact, candidate_digest) {
                if sha256_hex(artifact) == digest {
                    verified_digest_count += 1;
                } else {
                    issue(
                        &mut issues,
                        format!("candidate.{evidence_path}.artifact_sha256"),
                        "candidate artifact SHA-256 mismatch",
                    );
                }
            }

            required_string(key_item, "source_family_id", &format!("evaluator.{evidence_path}"), &mut issues);
            if let Some(role) = required_string(key_item, "source_role", &format!("evaluator.{evidence_path}"), &mut issues) {
                if !["primary", "secondary", "registry", "syndicated", "unknown"].contains(&role) {
                    issue(&mut issues, format!("evaluator.{evidence_path}.source_role"), "unknown source role");
                }
            }
            if let Some(integrity) = required_string(key_item, "content_integrity", &format!("evaluator.{evidence_path}"), &mut issues) {
                if !["clean", "compromised", "missing", "unknown"].contains(&integrity) {
                    issue(&mut issues, format!("evaluator.{evidence_path}.content_integrity"), "unknown integrity state");
                }
            }
            if let Some(correction) = required_string(key_item, "correction_status", &format!("evaluator.{evidence_path}"), &mut issues) {
                if !["none", "corrected", "retracted", "superseded"].contains(&correction) {
                    issue(&mut issues, format!("evaluator.{evidence_path}.correction_status"), "unknown correction status");
                }
            }
            if !key_item.get("relationship_notes").is_some_and(Value::is_string) {
                issue(&mut issues, format!("evaluator.{evidence_path}.relationship_notes"), "relationship_notes must be a string");
            }
            if let Some(relation) = required_string(
                key_item,
                "relation_to_claim",
                &format!("evaluator.{evidence_path}"),
                &mut issues,
            ) {
                if !RELATIONS.contains(&relation) {
                    issue(
                        &mut issues,
                        format!("evaluator.{evidence_path}.relation_to_claim"),
                        "unsupported evidence relation",
                    );
                }
            }
            if let Some(span_value) = key_item.get("quote_span") {
                if let Some(span) = object(
                    span_value,
                    &format!("evaluator.{evidence_path}.quote_span"),
                    &mut issues,
                ) {
                    const SPAN_KEYS: &[&str] = &["byte_start", "byte_end", "quote"];
                    validate_exact_keys(
                        span,
                        SPAN_KEYS,
                        &format!("evaluator.{evidence_path}.quote_span"),
                        &mut issues,
                    );
                    let start = span.get("byte_start").and_then(Value::as_u64).map(|n| n as usize);
                    let end = span.get("byte_end").and_then(Value::as_u64).map(|n| n as usize);
                    let quote = required_string(
                        span,
                        "quote",
                        &format!("evaluator.{evidence_path}.quote_span"),
                        &mut issues,
                    );
                    match (artifact, start, end, quote) {
                        (Some(text), Some(start), Some(end), Some(quote)) => {
                            let bytes = text.as_bytes();
                            if start >= end || end > bytes.len() {
                                issue(
                                    &mut issues,
                                    format!("evaluator.{evidence_path}.quote_span"),
                                    "invalid quote byte range",
                                );
                            } else if std::str::from_utf8(&bytes[start..end]).ok() != Some(quote) {
                                issue(
                                    &mut issues,
                                    format!("evaluator.{evidence_path}.quote_span"),
                                    "evaluator quote does not match candidate-visible artifact",
                                );
                            } else {
                                verified_quote_count += 1;
                            }
                        }
                        _ => issue(
                            &mut issues,
                            format!("evaluator.{evidence_path}.quote_span"),
                            "quote span, candidate artifact and offsets are required",
                        ),
                    }
                }
            } else {
                issue(
                    &mut issues,
                    format!("evaluator.{evidence_path}.quote_span"),
                    "missing evaluator quote span",
                );
            }
            evidence_count += 1;
        }
        fixture_count += 1;
    }

    if !issues.is_empty() {
        return Err(issues);
    }
    Ok(SeparatedViewReport {
        fixture_count,
        evidence_count,
        verified_digest_count,
        verified_quote_count,
    })
}

/// Validate the checked-in split views, including answer-key isolation and
/// cross-binding to the exact candidate-visible evidence bytes.
pub fn validate_checked_in_views() -> Result<SeparatedViewReport, Vec<ValidationIssue>> {
    validate_separated_views(CANDIDATE_INPUTS, EVALUATOR_KEY)
}

/// Run deterministic negative controls against the checked-in pack.
pub fn self_test() -> Result<ValidationReport, String> {
    let valid = validate_pack(FIXTURE_PACK).map_err(|issues| format!("fixture pack invalid: {issues:?}"))?;
    validate_checked_in_views()
        .map_err(|issues| format!("separated candidate/evaluator views invalid: {issues:?}"))?;
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
    let duplicate_id = duplicate["fixtures"][0]["fixture_id"].clone();
    duplicate["fixtures"][1]["fixture_id"] = duplicate_id;
    if validate_value(&duplicate).is_ok() {
        return Err("duplicate fixture ID unexpectedly passed validation".to_owned());
    }

    Ok(valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_and_evaluator_views_cross_bind_without_label_leakage() {
        let report = validate_checked_in_views().unwrap();
        assert_eq!(report.fixture_count, 15);
        assert_eq!(report.evidence_count, 23);
        assert_eq!(report.verified_digest_count, 23);
        assert_eq!(report.verified_quote_count, 23);
    }

    #[test]
    fn candidate_view_rejects_evaluator_label_leakage() {
        let mut candidate: Value = serde_json::from_str(CANDIDATE_INPUTS).unwrap();
        let evaluator: Value = serde_json::from_str(EVALUATOR_KEY).unwrap();
        candidate["fixtures"][0]["expected_verdict"] = Value::String("SUPPORTED".to_owned());
        assert!(validate_separated_views(
            &candidate.to_string(),
            &evaluator.to_string()
        )
        .is_err());
    }

    #[test]
    fn evaluator_key_cannot_be_detached_from_candidate_artifact() {
        let candidate: Value = serde_json::from_str(CANDIDATE_INPUTS).unwrap();
        let mut evaluator: Value = serde_json::from_str(EVALUATOR_KEY).unwrap();
        evaluator["fixtures"][0]["evidence_labels"][0]["artifact_sha256"] =
            Value::String("0".repeat(64));
        assert!(validate_separated_views(
            &candidate.to_string(),
            &evaluator.to_string()
        )
        .is_err());
    }

    #[test]
    fn timestamps_require_valid_calendar_and_timezone_fields() {
        assert!(valid_time("2024-02-29T12:30:45Z"));
        assert!(valid_time("2026-11-25T13:07:00+02:00"));
        assert!(!valid_time("2025-02-29T12:30:45Z"));
        assert!(!valid_time("2026-13-01T12:30:45Z"));
        assert!(!valid_time("2026-11-25T25:30:45Z"));
        assert!(!valid_time("not-a-dateTyesZ"));
        assert!(!valid_time("2026-11-25T13:07:00"));
    }

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
        let duplicate_id = value["fixtures"][0]["fixture_id"].clone();
        value["fixtures"][1]["fixture_id"] = duplicate_id;
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
    fn missing_claim_scope_fails_closed() {
        let mut value: Value = serde_json::from_str(FIXTURE_PACK).unwrap();
        value["fixtures"][0].as_object_mut().unwrap().remove("claim_scope");
        assert!(validate_value(&value).is_err());
    }

    #[test]
    fn candidate_visible_synthetic_label_provenance_fails_closed() {
        let mut value: Value = serde_json::from_str(FIXTURE_PACK).unwrap();
        value["fixtures"][0]["label_provenance"]["independent_of_candidate"] =
            Value::Bool(false);
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
