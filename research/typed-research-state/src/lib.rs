//! Conformance checks for the public ResearchState evidence projection.
//!
//! This crate validates state shape, cross-references and the non-promotion
//! boundaries between retrieval, claim classification, truth adjudication and
//! decision usability. It does not perform semantic entailment or authenticate
//! upstream evidence authorities.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

const EXAMPLES: &str = include_str!("../fixtures/research-state-examples-v0.1.json");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationIssue {
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationReport {
    pub research_episode_id: String,
    pub claim_count: usize,
    pub evidence_count: usize,
    pub relation_count: usize,
    /// Digest of the exact serialized candidate state bytes.
    pub state_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExamplesReport {
    pub example_count: usize,
    pub claim_count: usize,
    pub evidence_count: usize,
    pub relation_count: usize,
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
            issue(issues, path, "expected object");
            None
        }
    }
}

fn exact_keys(
    map: &serde_json::Map<String, Value>,
    expected: &[&str],
    path: &str,
    issues: &mut Vec<ValidationIssue>,
) {
    for key in expected {
        if !map.contains_key(*key) {
            issue(issues, format!("{path}.{key}"), "required field missing");
        }
    }
    for key in map.keys() {
        if !expected.contains(&key.as_str()) {
            issue(issues, format!("{path}.{key}"), "unexpected field");
        }
    }
}

fn string<'a>(
    map: &'a serde_json::Map<String, Value>,
    key: &str,
    path: &str,
    issues: &mut Vec<ValidationIssue>,
) -> Option<&'a str> {
    match map.get(key).and_then(Value::as_str) {
        Some(value) if !value.trim().is_empty() => Some(value),
        _ => {
            issue(issues, format!("{path}.{key}"), "expected non-empty string");
            None
        }
    }
}

fn array<'a>(
    map: &'a serde_json::Map<String, Value>,
    key: &str,
    path: &str,
    issues: &mut Vec<ValidationIssue>,
) -> Option<&'a Vec<Value>> {
    match map.get(key).and_then(Value::as_array) {
        Some(value) => Some(value),
        None => {
            issue(issues, format!("{path}.{key}"), "expected array");
            None
        }
    }
}

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
}

fn digest_string(
    map: &serde_json::Map<String, Value>,
    key: &str,
    path: &str,
    issues: &mut Vec<ValidationIssue>,
) -> Option<String> {
    let value = string(map, key, path, issues)?;
    if !is_digest(value) {
        issue(
            issues,
            format!("{path}.{key}"),
            "expected lowercase SHA-256 hex",
        );
        None
    } else {
        Some(value.to_owned())
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
    if year == 0
        || !(1..=12).contains(&month)
        || !(0..=23).contains(&hour)
        || !(0..=59).contains(&minute)
        || !(0..=59).contains(&second)
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
    if (bytes[19] != b'+' && bytes[19] != b'-') || bytes[22] != b':' {
        return false;
    }
    let (Some(offset_hour), Some(offset_minute)) = (number(20..22), number(23..25)) else {
        return false;
    };
    offset_hour <= 23 && offset_minute <= 59
}

fn validate_evidence(
    value: &Value,
    index: usize,
    ids: &mut HashMap<String, Option<String>>,
    issues: &mut Vec<ValidationIssue>,
) {
    let path = format!("evidence[{index}]");
    let Some(map) = object(value, &path, issues) else {
        return;
    };
    const KEYS: &[&str] = &[
        "evidence_id",
        "artifact_ref",
        "artifact_sha256",
        "canonical_source_ref",
        "published_at",
        "captured_at",
        "source_lineage",
        "source_role",
        "integrity_status",
        "correction_status",
        "retrieval_state",
    ];
    exact_keys(map, KEYS, &path, issues);

    let id = string(map, "evidence_id", &path, issues).map(str::to_owned);
    string(map, "artifact_ref", &path, issues);
    digest_string(map, "artifact_sha256", &path, issues);
    if let Some(source) = string(map, "canonical_source_ref", &path, issues)
        && !(source.starts_with("https://") || source.starts_with("http://"))
    {
        issue(
            issues,
            format!("{path}.canonical_source_ref"),
            "source URI must use HTTP(S)",
        );
    }
    for key in ["published_at", "captured_at"] {
        if let Some(time) = string(map, key, &path, issues)
            && !valid_time(time)
        {
            issue(issues, format!("{path}.{key}"), "invalid RFC3339 timestamp");
        }
    }

    let mut family = None;
    if let Some(lineage_value) = map.get("source_lineage") {
        if let Some(lineage) = object(lineage_value, &format!("{path}.source_lineage"), issues) {
            match lineage.get("status").and_then(Value::as_str) {
                Some("UNKNOWN") => {
                    exact_keys(
                        lineage,
                        &["status"],
                        &format!("{path}.source_lineage"),
                        issues,
                    );
                }
                Some("KNOWN") => {
                    exact_keys(
                        lineage,
                        &["status", "family_ref"],
                        &format!("{path}.source_lineage"),
                        issues,
                    );
                    family = string(
                        lineage,
                        "family_ref",
                        &format!("{path}.source_lineage"),
                        issues,
                    )
                    .map(str::to_owned);
                }
                _ => issue(
                    issues,
                    format!("{path}.source_lineage.status"),
                    "lineage status must be KNOWN or UNKNOWN",
                ),
            }
        }
    } else {
        issue(
            issues,
            format!("{path}.source_lineage"),
            "missing source lineage state",
        );
    }

    if let Some(role) = string(map, "source_role", &path, issues)
        && !["UNKNOWN", "PRIMARY", "SECONDARY", "REGISTRY", "SYNDICATED"].contains(&role)
    {
        issue(issues, format!("{path}.source_role"), "unknown source role");
    }
    if let Some(state) = string(map, "integrity_status", &path, issues)
        && !["UNKNOWN", "CLEAN", "COMPROMISED", "MISSING"].contains(&state)
    {
        issue(
            issues,
            format!("{path}.integrity_status"),
            "unknown integrity state",
        );
    }
    if let Some(state) = string(map, "correction_status", &path, issues)
        && !["UNKNOWN", "NONE", "CORRECTED", "RETRACTED", "SUPERSEDED"].contains(&state)
    {
        issue(
            issues,
            format!("{path}.correction_status"),
            "unknown correction state",
        );
    }
    if let Some(state) = string(map, "retrieval_state", &path, issues)
        && ![
            "NOT_FOUND",
            "FETCHED",
            "CAPTURED",
            "DIGEST_MISMATCH",
            "UNATTRIBUTED",
        ]
        .contains(&state)
    {
        issue(
            issues,
            format!("{path}.retrieval_state"),
            "unknown retrieval state",
        );
    }
    if let Some(id) = id
        && ids.insert(id.clone(), family).is_some()
    {
        issue(
            issues,
            format!("{path}.evidence_id"),
            "duplicate evidence ID",
        );
    }
}

fn validate_classifier(
    value: &Value,
    path: &str,
    prefix_sha256: &str,
    claim_sha256: &str,
    issues: &mut Vec<ValidationIssue>,
) -> Option<String> {
    let map = object(value, path, issues)?;
    const KEYS: &[&str] = &[
        "classifier_id",
        "version",
        "output_digest",
        "confidence",
        "emitted_at",
        "evidence_prefix_sha256",
        "claim_sha256",
    ];
    exact_keys(map, KEYS, path, issues);
    let classifier_id = string(map, "classifier_id", path, issues).map(str::to_owned);
    string(map, "version", path, issues);
    digest_string(map, "output_digest", path, issues);
    let confidence = map.get("confidence").and_then(Value::as_f64);
    if !confidence.is_some_and(|value| value.is_finite() && (0.0..=1.0).contains(&value)) {
        issue(
            issues,
            format!("{path}.confidence"),
            "classifier confidence must be finite and in [0,1]",
        );
    }
    if let Some(time) = string(map, "emitted_at", path, issues)
        && !valid_time(time)
    {
        issue(
            issues,
            format!("{path}.emitted_at"),
            "invalid RFC3339 timestamp",
        );
    }
    let actual_prefix = digest_string(map, "evidence_prefix_sha256", path, issues);
    if actual_prefix.as_deref() != Some(prefix_sha256) {
        issue(
            issues,
            format!("{path}.evidence_prefix_sha256"),
            "classifier output was produced against a different evidence prefix",
        );
    }
    let actual_claim = digest_string(map, "claim_sha256", path, issues);
    if actual_claim.as_deref() != Some(claim_sha256) {
        issue(
            issues,
            format!("{path}.claim_sha256"),
            "classifier output was produced for a different scoped claim",
        );
    }
    classifier_id
}

fn validate_quote_span(value: &Value, path: &str, issues: &mut Vec<ValidationIssue>) {
    let Some(map) = object(value, path, issues) else {
        return;
    };
    const KEYS: &[&str] = &["byte_start", "byte_end", "quote_sha256"];
    exact_keys(map, KEYS, path, issues);
    let start = map.get("byte_start").and_then(Value::as_u64);
    let end = map.get("byte_end").and_then(Value::as_u64);
    if !matches!((start, end), (Some(a), Some(b)) if a < b) {
        issue(issues, path, "quote span must have byte_start < byte_end");
    }
    digest_string(map, "quote_sha256", path, issues);
}

struct RelationValidationContext<'a> {
    claim_id: &'a str,
    evidence_ids: &'a HashMap<String, Option<String>>,
    prefix_sha256: &'a str,
    claim_sha256: &'a str,
    classified_authorities: &'a mut HashSet<String>,
    supports: &'a mut HashSet<String>,
    refutes: &'a mut HashSet<String>,
}

fn validate_relation(
    value: &Value,
    index: usize,
    context: &mut RelationValidationContext<'_>,
    issues: &mut Vec<ValidationIssue>,
) {
    let path = format!("claims[{}].evidence_relations[{index}]", context.claim_id);
    let Some(map) = object(value, &path, issues) else {
        return;
    };
    const KEYS: &[&str] = &[
        "claim_id",
        "evidence_id",
        "relation",
        "classifier_provenance",
        "quote_span",
    ];
    exact_keys(map, KEYS, &path, issues);
    if string(map, "claim_id", &path, issues) != Some(context.claim_id) {
        issue(
            issues,
            format!("{path}.claim_id"),
            "relation claim_id must match containing claim",
        );
    }
    let evidence_id = string(map, "evidence_id", &path, issues).map(str::to_owned);
    let family = evidence_id
        .as_ref()
        .and_then(|id| context.evidence_ids.get(id))
        .cloned();
    if family.is_none() {
        issue(
            issues,
            format!("{path}.evidence_id"),
            "relation references missing evidence",
        );
    }
    let relation = string(map, "relation", &path, issues);
    if let Some(relation) = relation {
        if ![
            "UNCLASSIFIED",
            "SUPPORTS",
            "REFUTES",
            "QUALIFIES",
            "IRRELEVANT",
            "UNRESOLVED",
        ]
        .contains(&relation)
        {
            issue(
                issues,
                format!("{path}.relation"),
                "unknown claim-source relation",
            );
        }
        // Known lineage uses the authoritative source-family reference.
        // Unknown lineage still remains a distinct observed source for conflict
        // detection, but its synthetic key is never evidence of independence.
        let known_family = match family.as_ref() {
            Some(Some(value)) => Some(value.clone()),
            _ => None,
        };
        let origin = known_family.or_else(|| {
            evidence_id
                .as_ref()
                .map(|id| format!("unknown-origin:{id}"))
        });
        if relation == "SUPPORTS"
            && let Some(origin) = origin.as_ref()
        {
            context.supports.insert(origin.clone());
        }
        if relation == "REFUTES"
            && let Some(origin) = origin.as_ref()
        {
            context.refutes.insert(origin.clone());
        }
    }

    match (
        relation,
        map.get("classifier_provenance"),
        map.get("quote_span"),
    ) {
        (Some("UNCLASSIFIED"), Some(Value::Null), Some(Value::Null)) => {}
        (Some("UNCLASSIFIED"), _, _) => issue(
            issues,
            &path,
            "UNCLASSIFIED relation must not carry classifier provenance or a gold quote span",
        ),
        (Some(valid_relation), Some(Value::Object(_)), Some(Value::Object(_)))
            if valid_relation != "UNCLASSIFIED" =>
        {
            let classifier = validate_classifier(
                map.get("classifier_provenance").unwrap(),
                &format!("{path}.classifier_provenance"),
                context.prefix_sha256,
                context.claim_sha256,
                issues,
            );
            if let Some(classifier) = classifier {
                context.classified_authorities.insert(classifier);
            }
            validate_quote_span(
                map.get("quote_span").unwrap(),
                &format!("{path}.quote_span"),
                issues,
            );
        }
        (Some(valid_relation), _, _) if valid_relation != "UNCLASSIFIED" => issue(
            issues,
            &path,
            "classified relation requires classifier provenance and exact quote digest",
        ),
        _ => {}
    }
}

fn validate_adjudication(
    value: &Value,
    path: &str,
    classifier_ids: &HashSet<String>,
    issues: &mut Vec<ValidationIssue>,
) -> (bool, Option<String>) {
    let Some(map) = object(value, path, issues) else {
        return (false, None);
    };
    const KEYS: &[&str] = &[
        "status",
        "result",
        "authority_id",
        "evidence_reference",
        "adjudicated_at",
    ];
    exact_keys(map, KEYS, path, issues);
    match map.get("status").and_then(Value::as_str) {
        Some("NOT_ADJUDICATED") => {
            for key in [
                "result",
                "authority_id",
                "evidence_reference",
                "adjudicated_at",
            ] {
                if map.get(key) != Some(&Value::Null) {
                    issue(
                        issues,
                        format!("{path}.{key}"),
                        "NOT_ADJUDICATED requires null fields",
                    );
                }
            }
            (false, None)
        }
        Some("ADJUDICATED") => {
            let result = string(map, "result", path, issues).map(str::to_owned);
            if !result.as_deref().is_some_and(|value| {
                ["SUPPORTED", "REFUTED", "MIXED", "UNDECIDABLE"].contains(&value)
            }) {
                issue(
                    issues,
                    format!("{path}.result"),
                    "unknown adjudication result",
                );
            }
            let authority = string(map, "authority_id", path, issues).map(str::to_owned);
            if let Some(authority) = authority.as_ref()
                && classifier_ids.contains(authority)
            {
                issue(
                    issues,
                    format!("{path}.authority_id"),
                    "claim-source classifier cannot serve as independent adjudicator",
                );
            }
            string(map, "evidence_reference", path, issues);
            if let Some(time) = string(map, "adjudicated_at", path, issues)
                && !valid_time(time)
            {
                issue(
                    issues,
                    format!("{path}.adjudicated_at"),
                    "invalid RFC3339 timestamp",
                );
            }
            (true, result)
        }
        _ => {
            issue(
                issues,
                format!("{path}.status"),
                "adjudication status must be ADJUDICATED or NOT_ADJUDICATED",
            );
            (false, None)
        }
    }
}

struct UsabilityValidationContext<'a> {
    decision_id: &'a str,
    risk_class: &'a str,
    adjudicated: bool,
    adjudication_result: Option<&'a str>,
    has_conflict: bool,
    has_support: bool,
}

fn validate_usability(
    value: &Value,
    path: &str,
    context: &UsabilityValidationContext<'_>,
    issues: &mut Vec<ValidationIssue>,
) {
    let Some(map) = object(value, path, issues) else {
        return;
    };
    const KEYS: &[&str] = &["status", "decision_scope_id", "policy_id", "reason_codes"];
    exact_keys(map, KEYS, path, issues);
    let status = string(map, "status", path, issues);
    if !status.is_some_and(|value| {
        [
            "UNASSESSED",
            "SUFFICIENT",
            "INSUFFICIENT",
            "CONFLICTED",
            "STALE",
            "OUT_OF_SCOPE",
        ]
        .contains(&value)
    }) {
        issue(
            issues,
            format!("{path}.status"),
            "unknown decision-usability status",
        );
    }
    if string(map, "decision_scope_id", path, issues) != Some(context.decision_id) {
        issue(
            issues,
            format!("{path}.decision_scope_id"),
            "usability must be scoped to the active decision",
        );
    }
    string(map, "policy_id", path, issues);
    if let Some(codes) = array(map, "reason_codes", path, issues) {
        for (index, code) in codes.iter().enumerate() {
            if code.as_str().is_none_or(|value| value.trim().is_empty()) {
                issue(
                    issues,
                    format!("{path}.reason_codes[{index}]"),
                    "reason code must be non-empty",
                );
            }
        }
    }
    if status == Some("SUFFICIENT") && !context.has_support {
        issue(
            issues,
            format!("{path}.status"),
            "decision sufficiency requires at least one explicit claim-specific SUPPORTS relation",
        );
    }
    if status == Some("SUFFICIENT") && context.has_conflict {
        issue(
            issues,
            format!("{path}.status"),
            "conflicting support/refutation cannot be decision-sufficient",
        );
    }
    if status == Some("SUFFICIENT")
        && ["HIGH", "CRITICAL"].contains(&context.risk_class)
        && (!context.adjudicated || context.adjudication_result != Some("SUPPORTED"))
    {
        issue(
            issues,
            format!("{path}.status"),
            "high-risk SUFFICIENT requires independent SUPPORTED adjudication",
        );
    }
}

pub fn scoped_claim_sha256(
    claim_id: &str,
    statement: &str,
    scope: &Value,
    reference_time: &str,
) -> String {
    // This ordered manifest binds a classifier relation to the exact claim
    // text, scope and reference time, not merely to a reusable claim ID.
    let manifest = serde_json::json!({
        "claim_id": claim_id,
        "statement": statement,
        "scope": scope,
        "reference_time": reference_time,
    });
    let bytes = serde_json::to_vec(&manifest)
        .expect("serializing a serde_json::Value to JSON bytes is infallible");
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Compute the canonical SHA-256 digest for the ordered public evidence prefix.
///
/// The manifest is compact JSON for the ordered object
/// { research_episode_id, sequence_number, evidence }. Callers must preserve
/// evidence order and serialize the exact public evidence records.
pub fn evidence_prefix_sha256(
    research_episode_id: &str,
    sequence_number: u64,
    evidence: &[Value],
) -> String {
    // serde_json preserve_order is enabled in this standalone crate. This
    // manifest field order is canonical and is shared with the fixture writer.
    let manifest = serde_json::json!({
        "research_episode_id": research_episode_id,
        "sequence_number": sequence_number,
        "evidence": evidence,
    });
    let bytes = serde_json::to_vec(&manifest)
        .expect("serializing a serde_json::Value to JSON bytes is infallible");
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn validate_state(input: &str) -> Result<ValidationReport, Vec<ValidationIssue>> {
    let value: Value = serde_json::from_str(input).map_err(|error| {
        vec![ValidationIssue {
            path: "$".to_owned(),
            message: format!("invalid JSON: {error}"),
        }]
    })?;
    let mut issues = Vec::new();
    let Some(root) = object(&value, "$", &mut issues) else {
        return Err(issues);
    };
    const ROOT_KEYS: &[&str] = &[
        "schema_version",
        "status",
        "research_episode_id",
        "evidence_prefix",
        "decision_context",
        "claims",
        "evidence",
    ];
    exact_keys(root, ROOT_KEYS, "$", &mut issues);
    if string(root, "schema_version", "$", &mut issues)
        != Some("symthaea-public-research-state-v0.1")
    {
        issue(
            &mut issues,
            "$.schema_version",
            "unsupported schema version",
        );
    }
    if !string(root, "status", "$", &mut issues).is_some_and(|status| {
        [
            "SYNTHETIC_CONFORMANCE_EXAMPLE_NOT_PRODUCTION_STATE",
            "PRODUCTION_PUBLIC_STATE",
        ]
        .contains(&status)
    }) {
        issue(&mut issues, "$.status", "unknown ResearchState origin mode");
    }
    let episode = string(root, "research_episode_id", "$", &mut issues)
        .unwrap_or_default()
        .to_owned();

    let Some(prefix_value) = root.get("evidence_prefix") else {
        issue(&mut issues, "$.evidence_prefix", "missing evidence prefix");
        return Err(issues);
    };
    let Some(prefix) = object(prefix_value, "$.evidence_prefix", &mut issues) else {
        return Err(issues);
    };
    exact_keys(
        prefix,
        &[
            "prefix_id",
            "prefix_sha256",
            "lineage_snapshot_ref",
            "sequence_number",
        ],
        "$.evidence_prefix",
        &mut issues,
    );
    string(prefix, "prefix_id", "$.evidence_prefix", &mut issues);
    let prefix_sha256 = digest_string(prefix, "prefix_sha256", "$.evidence_prefix", &mut issues)
        .unwrap_or_default();
    string(
        prefix,
        "lineage_snapshot_ref",
        "$.evidence_prefix",
        &mut issues,
    );
    let sequence_number = prefix.get("sequence_number").and_then(Value::as_u64);
    if sequence_number.is_none() {
        issue(
            &mut issues,
            "$.evidence_prefix.sequence_number",
            "expected non-negative sequence number",
        );
    }

    let Some(decision_value) = root.get("decision_context") else {
        issue(
            &mut issues,
            "$.decision_context",
            "missing decision context",
        );
        return Err(issues);
    };
    let Some(decision) = object(decision_value, "$.decision_context", &mut issues) else {
        return Err(issues);
    };
    exact_keys(
        decision,
        &["decision_id", "question", "reference_time", "risk_class"],
        "$.decision_context",
        &mut issues,
    );
    let decision_id = string(decision, "decision_id", "$.decision_context", &mut issues)
        .unwrap_or_default()
        .to_owned();
    string(decision, "question", "$.decision_context", &mut issues);
    if let Some(time) = string(
        decision,
        "reference_time",
        "$.decision_context",
        &mut issues,
    ) && !valid_time(time)
    {
        issue(
            &mut issues,
            "$.decision_context.reference_time",
            "invalid RFC3339 timestamp",
        );
    }
    let risk_class = string(decision, "risk_class", "$.decision_context", &mut issues)
        .unwrap_or_default()
        .to_owned();
    if !["LOW", "MODERATE", "HIGH", "CRITICAL"].contains(&risk_class.as_str()) {
        issue(
            &mut issues,
            "$.decision_context.risk_class",
            "unknown risk class",
        );
    }

    let Some(evidence) = array(root, "evidence", "$", &mut issues) else {
        return Err(issues);
    };
    if let Some(sequence_number) = sequence_number {
        let expected = evidence_prefix_sha256(&episode, sequence_number, evidence);
        if prefix_sha256 != expected {
            issue(
                &mut issues,
                "$.evidence_prefix.prefix_sha256",
                "prefix digest does not match the exact public evidence manifest",
            );
        }
    }
    let mut evidence_ids: HashMap<String, Option<String>> = HashMap::new();
    for (index, item) in evidence.iter().enumerate() {
        validate_evidence(item, index, &mut evidence_ids, &mut issues);
    }

    let Some(claims) = array(root, "claims", "$", &mut issues) else {
        return Err(issues);
    };
    if claims.is_empty() {
        issue(&mut issues, "$.claims", "at least one claim is required");
    }
    let mut claim_ids = HashSet::new();
    let mut relation_count = 0;
    for (index, item) in claims.iter().enumerate() {
        let path = format!("claims[{index}]");
        let Some(claim) = object(item, &path, &mut issues) else {
            continue;
        };
        exact_keys(
            claim,
            &[
                "claim_id",
                "statement",
                "scope",
                "reference_time",
                "evidence_relations",
                "truth_adjudication",
                "decision_usability",
            ],
            &path,
            &mut issues,
        );
        let claim_id = string(claim, "claim_id", &path, &mut issues)
            .unwrap_or_default()
            .to_owned();
        if !claim_ids.insert(claim_id.clone()) {
            issue(
                &mut issues,
                format!("{path}.claim_id"),
                "duplicate claim ID",
            );
        }
        let statement = string(claim, "statement", &path, &mut issues).unwrap_or_default();
        if let Some(scope_value) = claim.get("scope") {
            if let Some(scope) = object(scope_value, &format!("{path}.scope"), &mut issues) {
                const SCOPE_KEYS: &[&str] = &[
                    "population",
                    "outcome_or_measure",
                    "quantifier_or_effect",
                    "domain_or_reference_boundary",
                    "scope_note",
                ];
                exact_keys(scope, SCOPE_KEYS, &format!("{path}.scope"), &mut issues);
                for key in SCOPE_KEYS {
                    string(scope, key, &format!("{path}.scope"), &mut issues);
                }
            }
        } else {
            issue(
                &mut issues,
                format!("{path}.scope"),
                "claim scope is required",
            );
        }
        let reference_time =
            string(claim, "reference_time", &path, &mut issues).unwrap_or_default();
        if !valid_time(reference_time) {
            issue(
                &mut issues,
                format!("{path}.reference_time"),
                "invalid RFC3339 timestamp",
            );
        }
        let null_scope = Value::Null;
        let scope_value = claim.get("scope").unwrap_or(&null_scope);
        let claim_sha256 = scoped_claim_sha256(&claim_id, statement, scope_value, reference_time);

        let mut pair_ids = HashSet::new();
        let mut classifier_ids = HashSet::new();
        let mut support_families = HashSet::new();
        let mut refute_families = HashSet::new();
        if let Some(relations) = array(claim, "evidence_relations", &path, &mut issues) {
            relation_count += relations.len();
            for (relation_index, relation) in relations.iter().enumerate() {
                let mut relation_context = RelationValidationContext {
                    claim_id: &claim_id,
                    evidence_ids: &evidence_ids,
                    prefix_sha256: &prefix_sha256,
                    claim_sha256: &claim_sha256,
                    classified_authorities: &mut classifier_ids,
                    supports: &mut support_families,
                    refutes: &mut refute_families,
                };
                validate_relation(relation, relation_index, &mut relation_context, &mut issues);
                if let Some(relation_map) = relation.as_object()
                    && let (Some(claim_ref), Some(evidence_ref)) = (
                        relation_map.get("claim_id").and_then(Value::as_str),
                        relation_map.get("evidence_id").and_then(Value::as_str),
                    )
                {
                    let key = format!("{claim_ref}|{evidence_ref}");
                    if !pair_ids.insert(key) {
                        issue(
                            &mut issues,
                            format!("{path}.evidence_relations[{relation_index}]"),
                            "duplicate claim/evidence relation",
                        );
                    }
                }
            }
        }

        let adjudication_path = format!("{path}.truth_adjudication");
        let adjudication = claim.get("truth_adjudication");
        let (adjudicated, result) = match adjudication {
            Some(value) => {
                validate_adjudication(value, &adjudication_path, &classifier_ids, &mut issues)
            }
            None => {
                issue(
                    &mut issues,
                    &adjudication_path,
                    "truth adjudication field is required",
                );
                (false, None)
            }
        };
        let has_conflict = !support_families.is_empty() && !refute_families.is_empty();
        let has_support = !support_families.is_empty();
        if let Some(usability) = claim.get("decision_usability") {
            let usability_context = UsabilityValidationContext {
                decision_id: &decision_id,
                risk_class: &risk_class,
                adjudicated,
                adjudication_result: result.as_deref(),
                has_conflict,
                has_support,
            };
            validate_usability(
                usability,
                &format!("{path}.decision_usability"),
                &usability_context,
                &mut issues,
            );
        } else {
            issue(
                &mut issues,
                format!("{path}.decision_usability"),
                "decision usability is required",
            );
        }
    }

    if !issues.is_empty() {
        return Err(issues);
    }
    Ok(ValidationReport {
        research_episode_id: episode,
        claim_count: claims.len(),
        evidence_count: evidence.len(),
        relation_count,
        state_sha256: Sha256::digest(input.as_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
    })
}

/// Validate all checked-in synthetic ResearchState examples.
pub fn validate_examples() -> Result<ExamplesReport, Vec<ValidationIssue>> {
    let wrapper: Value = serde_json::from_str(EXAMPLES).map_err(|error| {
        vec![ValidationIssue {
            path: "examples".to_owned(),
            message: format!("invalid examples JSON: {error}"),
        }]
    })?;
    let mut issues = Vec::new();
    let Some(root) = object(&wrapper, "examples", &mut issues) else {
        return Err(issues);
    };
    exact_keys(
        root,
        &[
            "schema_version",
            "status",
            "created_at",
            "issue",
            "owner",
            "examples",
        ],
        "examples",
        &mut issues,
    );
    if string(root, "schema_version", "examples", &mut issues)
        != Some("symthaea-public-research-state-v0.1")
    {
        issue(
            &mut issues,
            "examples.schema_version",
            "unexpected example schema",
        );
    }
    if string(root, "status", "examples", &mut issues)
        != Some("SYNTHETIC_CONFORMANCE_EXAMPLES_NOT_PRODUCTION_STATE")
    {
        issue(
            &mut issues,
            "examples.status",
            "examples must remain synthetic",
        );
    }
    string(root, "created_at", "examples", &mut issues);
    string(root, "owner", "examples", &mut issues);
    if root.get("issue").and_then(Value::as_u64) != Some(13663) {
        issue(&mut issues, "examples.issue", "unexpected owner issue");
    }
    let Some(examples) = array(root, "examples", "examples", &mut issues) else {
        return Err(issues);
    };
    if examples.len() < 4 {
        issue(
            &mut issues,
            "examples.examples",
            "at least four conformance examples are required",
        );
    }

    let mut ids = HashSet::new();
    let mut claim_count = 0;
    let mut evidence_count = 0;
    let mut relation_count = 0;
    for (index, example) in examples.iter().enumerate() {
        let path = format!("examples[{index}]");
        let Some(map) = object(example, &path, &mut issues) else {
            continue;
        };
        exact_keys(map, &["fixture_id", "state"], &path, &mut issues);
        if let Some(id) = string(map, "fixture_id", &path, &mut issues)
            && !ids.insert(id.to_owned())
        {
            issue(
                &mut issues,
                format!("{path}.fixture_id"),
                "duplicate example ID",
            );
        }
        if let Some(state) = map.get("state") {
            match validate_state(&state.to_string()) {
                Ok(report) => {
                    claim_count += report.claim_count;
                    evidence_count += report.evidence_count;
                    relation_count += report.relation_count;
                }
                Err(state_issues) => {
                    for state_issue in state_issues {
                        issue(
                            &mut issues,
                            format!("{path}.state.{}", state_issue.path),
                            state_issue.message,
                        );
                    }
                }
            }
        } else {
            issue(
                &mut issues,
                format!("{path}.state"),
                "state object is required",
            );
        }
    }

    if !issues.is_empty() {
        return Err(issues);
    }
    Ok(ExamplesReport {
        example_count: examples.len(),
        claim_count,
        evidence_count,
        relation_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn first_state() -> Value {
        let pack: Value = serde_json::from_str(EXAMPLES).unwrap();
        pack["examples"][0]["state"].clone()
    }

    fn second_state() -> Value {
        let pack: Value = serde_json::from_str(EXAMPLES).unwrap();
        pack["examples"][1]["state"].clone()
    }

    fn third_state() -> Value {
        let pack: Value = serde_json::from_str(EXAMPLES).unwrap();
        pack["examples"][2]["state"].clone()
    }

    #[test]
    fn all_four_separated_layer_examples_validate() {
        let report = validate_examples().unwrap();
        assert_eq!(report.example_count, 4);
        assert_eq!(report.claim_count, 4);
        assert_eq!(report.evidence_count, 5);
        assert_eq!(report.relation_count, 5);
    }

    #[test]
    fn state_digest_binds_exact_serialized_bytes() {
        let state = first_state();
        let text = state.to_string();
        let report = validate_state(&text).unwrap();
        assert_eq!(
            report.state_sha256,
            Sha256::digest(text.as_bytes())
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        );
    }

    #[test]
    fn evidence_prefix_digest_rejects_modified_manifest() {
        let mut state = first_state();
        state["evidence"][0]["artifact_ref"] =
            Value::String("urn:synthetic:artifact:tampered".to_owned());
        assert!(validate_state(&state.to_string()).is_err());
    }

    #[test]
    fn global_claim_confidence_is_not_an_allowed_field() {
        let mut state = first_state();
        state["confidence"] = Value::from(0.99);
        assert!(validate_state(&state.to_string()).is_err());
    }

    #[test]
    fn production_public_state_marker_is_accepted() {
        let mut state = first_state();
        state["status"] = Value::String("PRODUCTION_PUBLIC_STATE".to_owned());
        assert!(validate_state(&state.to_string()).is_ok());
    }

    #[test]
    fn claim_level_global_confidence_is_not_an_allowed_field() {
        let mut state = first_state();
        state["claims"][0]["confidence"] = Value::from(0.99);
        assert!(validate_state(&state.to_string()).is_err());
    }

    #[test]
    fn unclassified_relation_cannot_carry_classifier_or_quote() {
        let mut state = first_state();
        let prefix_sha256 = state["evidence_prefix"]["prefix_sha256"].clone();
        state["claims"][0]["evidence_relations"][0]["classifier_provenance"] = serde_json::json!({
            "classifier_id": "classifier",
            "version": "1",
            "output_digest": "a".repeat(64),
            "confidence": 0.9,
            "emitted_at": "2026-11-25T10:00:00Z",
            "evidence_prefix_sha256": prefix_sha256
        });
        assert!(validate_state(&state.to_string()).is_err());
    }

    #[test]
    fn classifier_output_is_bound_to_exact_claim_text_and_scope() {
        let mut state = second_state();
        state["claims"][0]["statement"] =
            Value::String("A materially changed claim with the same claim ID".to_owned());
        assert!(validate_state(&state.to_string()).is_err());
    }

    #[test]
    fn classifier_must_bind_the_current_evidence_prefix() {
        let mut state = serde_json::from_str::<Value>(EXAMPLES).unwrap();
        let state = &mut state["examples"][1]["state"];
        state["claims"][0]["evidence_relations"][0]["classifier_provenance"]["evidence_prefix_sha256"] =
            Value::String("0".repeat(64));
        assert!(validate_state(&state.to_string()).is_err());
    }

    #[test]
    fn unknown_source_lineage_does_not_hide_a_claim_conflict() {
        let mut state = first_state();

        let second = serde_json::json!({
            "evidence_id": "ev-a2",
            "artifact_ref": "urn:synthetic:artifact:rs-unknown-second",
            "artifact_sha256": "b".repeat(64),
            "canonical_source_ref": "https://example.org/second",
            "published_at": "2026-01-10T00:00:00Z",
            "captured_at": "2026-11-25T10:00:00Z",
            "source_lineage": { "status": "UNKNOWN" },
            "source_role": "UNKNOWN",
            "integrity_status": "CLEAN",
            "correction_status": "NONE",
            "retrieval_state": "CAPTURED"
        });
        state["evidence"].as_array_mut().unwrap().push(second);
        let evidence = state["evidence"].as_array().unwrap().clone();
        let updated_prefix = evidence_prefix_sha256("synthetic-episode", 1, &evidence);
        state["evidence_prefix"]["prefix_sha256"] = Value::String(updated_prefix.clone());

        let claim_value = &state["claims"][0];
        let claim_scope = claim_value["scope"].clone();
        let claim_statement = claim_value["statement"].as_str().unwrap().to_owned();
        let claim_reference_time = claim_value["reference_time"].as_str().unwrap().to_owned();
        let claim_digest = scoped_claim_sha256(
            "claim-a",
            &claim_statement,
            &claim_scope,
            &claim_reference_time,
        );
        let first_relation = &mut state["claims"][0]["evidence_relations"][0];
        first_relation["relation"] = Value::String("SUPPORTS".to_owned());
        first_relation["classifier_provenance"] = serde_json::json!({
            "classifier_id": "synthetic-classifier",
            "version": "0.1",
            "output_digest": "e".repeat(64),
            "confidence": 0.9,
            "emitted_at": "2026-11-25T10:00:00Z",
            "evidence_prefix_sha256": updated_prefix,
            "claim_sha256": claim_digest
        });
        first_relation["quote_span"] = serde_json::json!({
            "byte_start": 0,
            "byte_end": 5,
            "quote_sha256": "f".repeat(64)
        });
        let second_relation = serde_json::json!({
            "claim_id": "claim-a",
            "evidence_id": "ev-a2",
            "relation": "REFUTES",
            "classifier_provenance": {
                "classifier_id": "synthetic-classifier",
                "version": "0.1",
                "output_digest": "c".repeat(64),
                "confidence": 0.9,
                "emitted_at": "2026-11-25T10:00:00Z",
                "evidence_prefix_sha256": updated_prefix,
                "claim_sha256": claim_digest
            },
            "quote_span": {
                "byte_start": 0,
                "byte_end": 5,
                "quote_sha256": "d".repeat(64)
            }
        });
        state["claims"][0]["evidence_relations"]
            .as_array_mut()
            .unwrap()
            .push(second_relation);
        state["claims"][0]["decision_usability"]["status"] = Value::String("SUFFICIENT".to_owned());

        assert!(validate_state(&state.to_string()).is_err());
    }

    #[test]
    fn unclassified_retrieval_cannot_be_marked_sufficient() {
        let mut state = first_state();
        state["claims"][0]["decision_usability"]["status"] = Value::String("SUFFICIENT".to_owned());
        assert!(validate_state(&state.to_string()).is_err());
    }

    #[test]
    fn conflict_cannot_be_marked_sufficient() {
        let mut state = third_state();
        state["claims"][0]["decision_usability"]["status"] = Value::String("SUFFICIENT".to_owned());
        assert!(validate_state(&state.to_string()).is_err());
    }

    #[test]
    fn high_risk_sufficiency_requires_independent_supported_adjudication() {
        let mut state = second_state();
        state["decision_context"]["risk_class"] = Value::String("HIGH".to_owned());
        state["claims"][0]["decision_usability"]["status"] = Value::String("SUFFICIENT".to_owned());
        assert!(validate_state(&state.to_string()).is_err());
    }

    #[test]
    fn classifier_cannot_be_its_own_truth_adjudicator() {
        let mut pack: Value = serde_json::from_str(EXAMPLES).unwrap();
        let state = &mut pack["examples"][1]["state"];
        state["claims"][0]["truth_adjudication"] = serde_json::json!({
            "status": "ADJUDICATED",
            "result": "SUPPORTED",
            "authority_id": "synthetic-claim-pair-classifier",
            "evidence_reference": "urn:test:adjudication",
            "adjudicated_at": "2026-11-25T10:08:00Z"
        });
        assert!(validate_state(&state.to_string()).is_err());
    }

    #[test]
    fn dangling_claim_evidence_link_fails_closed() {
        let mut state = first_state();
        state["claims"][0]["evidence_relations"][0]["evidence_id"] =
            Value::String("does-not-exist".to_owned());
        assert!(validate_state(&state.to_string()).is_err());
    }

    #[test]
    fn unknown_source_lineage_is_not_promoted_to_a_family() {
        let mut state = first_state();
        state["evidence"][0]["source_lineage"]["family_ref"] =
            Value::String("guessed-family".to_owned());
        assert!(validate_state(&state.to_string()).is_err());
    }

    #[test]
    fn stale_historical_adjudication_remains_valid_but_not_usable() {
        let pack: Value = serde_json::from_str(EXAMPLES).unwrap();
        let state = &pack["examples"][3]["state"];
        let report = validate_state(&state.to_string()).unwrap();
        assert_eq!(report.claim_count, 1);
    }
}
