//! Read-only structural and temporal contract checks for EGDDM snapshots.
//!
//! This module intentionally does not execute repository-controlled code,
//! resolve external source URIs, authenticate signatures, or implement the
//! full JSON Schema 2020-12 specification.

use std::collections::{HashMap, HashSet, VecDeque};

use chrono::{DateTime, FixedOffset};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ValidationReport {
    pub diagnostics: Vec<Diagnostic>,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.diagnostics.is_empty()
    }

    fn error(&mut self, code: &'static str, path: impl Into<String>, message: impl Into<String>) {
        self.diagnostics.push(Diagnostic {
            code,
            path: path.into(),
            message: message.into(),
        });
    }

    fn sort(&mut self) {
        self.diagnostics.sort_by(|a, b| {
            (&a.path, a.code, &a.message).cmp(&(&b.path, b.code, &b.message))
        });
    }
}

#[derive(Debug, Clone)]
struct NodeMeta {
    node_type: String,
    claim_state: String,
    authority: String,
    attributes: Value,
    lifecycle_state: String,
    invalidated_by: Option<String>,
}

const NODE_TYPES: &[&str] = &[
    "project", "workspace", "package", "crate", "module", "symbol", "interface",
    "configuration_world", "dependency", "test", "benchmark", "runtime", "deployment",
    "objective", "capability", "requirement", "invariant", "decision", "hypothesis",
    "milestone", "acceptance_criterion", "risk", "evidence", "change_event",
    "development_action", "external_system", "authority_boundary", "domain_cluster",
    "application",
];

const CLAIM_STATES: &[&str] = &[
    "observed", "inferred", "proposed", "attempted", "passed_local", "ci_pending",
    "ci_passed", "independently_qualified", "rejected", "blocked", "unknown",
];

const AUTHORITIES: &[&str] = &[
    "source_declared", "syntax_extracted", "manifest_resolved", "compiler_resolved",
    "runtime_observed", "benchmark_observed", "independently_adjudicated", "inferred",
    "proposed",
];

const LIFECYCLE_STATES: &[&str] = &[
    "current", "stale", "superseded", "revoked", "unknown", "missing", "disputed",
];

const SOURCE_KINDS: &[&str] = &[
    "repository_file", "commit", "issue", "pull_request", "workflow_run",
    "benchmark_result", "external_document", "human_review", "other",
];

const RELATIONS: &[&str] = &[
    "contains", "exports", "imports", "depends_on", "calls", "implements", "requires",
    "validates", "tested_by", "observed_by", "introduced_by", "removed_by", "supersedes",
    "invalidates", "affects", "blocked_by", "contradicts", "derived_from",
    "compatible_with", "configured_by", "built_by", "deployed_as",
];

const EVENT_KINDS: &[&str] = &[
    "repository_revision_observed", "source_introduced", "source_changed", "source_removed",
    "contract_changed", "configuration_world_changed", "test_evidence_recorded",
    "benchmark_evidence_recorded", "evidence_invalidated", "evidence_superseded",
    "capability_link_changed", "decision_recorded", "authority_evidence_recorded",
    "development_action_proposed", "development_action_observed", "risk_opened", "risk_closed",
];

const EVENT_EFFECTS: &[&str] = &[
    "introduces", "changes", "invalidates", "supersedes", "observes", "proposes",
    "no_state_change",
];

fn object<'a>(value: &'a Value, path: &str, report: &mut ValidationReport) -> Option<&'a serde_json::Map<String, Value>> {
    match value.as_object() {
        Some(value) => Some(value),
        None => {
            report.error("EGDDM-TYPE-001", path, "expected an object");
            None
        }
    }
}

fn array<'a>(value: &'a Value, path: &str, report: &mut ValidationReport) -> Option<&'a Vec<Value>> {
    match value.as_array() {
        Some(value) => Some(value),
        None => {
            report.error("EGDDM-TYPE-002", path, "expected an array");
            None
        }
    }
}

fn string<'a>(value: &'a Value, path: &str, report: &mut ValidationReport) -> Option<&'a str> {
    match value.as_str() {
        Some(value) if !value.trim().is_empty() => Some(value),
        Some(_) => {
            report.error("EGDDM-STRING-001", path, "string must not be empty");
            None
        }
        None => {
            report.error("EGDDM-TYPE-003", path, "expected a non-empty string");
            None
        }
    }
}

fn field<'a>(value: &'a Value, key: &str, path: &str, report: &mut ValidationReport) -> Option<&'a Value> {
    match value.get(key) {
        Some(value) => Some(value),
        None => {
            report.error("EGDDM-REQUIRED-001", format!("{path}.{key}"), "required field is missing");
            None
        }
    }
}

fn enum_value(value: &Value, path: &str, allowed: &[&str], report: &mut ValidationReport) -> Option<String> {
    let value = string(value, path, report)?;
    if allowed.contains(&value) {
        Some(value.to_owned())
    } else {
        report.error(
            "EGDDM-ENUM-001",
            path,
            format!("unsupported value {value:?}; allowed values are {}", allowed.join(", ")),
        );
        None
    }
}

fn hex_string(value: &str) -> bool {
    value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_revision(value: &str) -> bool {
    (7..=64).contains(&value.len()) && hex_string(value)
}

fn valid_digest(value: &str) -> bool {
    matches!(value.len(), 40 | 64) && hex_string(value)
}

fn optional_string(value: &Value, path: &str, report: &mut ValidationReport) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(value) if !value.trim().is_empty() => Some(value.clone()),
        Value::String(_) => {
            report.error("EGDDM-STRING-001", path, "string must not be empty");
            None
        }
        _ => {
            report.error("EGDDM-TYPE-003", path, "expected a string or null");
            None
        }
    }
}

fn check_allowed_keys(
    value: &serde_json::Map<String, Value>,
    allowed: &[&str],
    path: &str,
    report: &mut ValidationReport,
) {
    for key in value.keys() {
        if !allowed.contains(&key.as_str()) {
            report.error(
                "EGDDM-UNKNOWN-FIELD-001",
                format!("{path}.{key}"),
                "field is not defined by the v0.1 contract",
            );
        }
    }
}

fn validate_source(value: &Value, path: &str, report: &mut ValidationReport) {
    let Some(source) = object(value, path, report) else {
        return;
    };
    check_allowed_keys(
        source,
        &["kind", "uri", "path", "line_start", "line_end", "digest"],
        path,
        report,
    );
    if let Some(kind) = field(value, "kind", path, report) {
        enum_value(kind, &format!("{path}.kind"), SOURCE_KINDS, report);
    }
    if let Some(uri) = field(value, "uri", path, report) {
        string(uri, &format!("{path}.uri"), report);
    }
    if let Some(digest) = field(value, "digest", path, report) {
        match digest {
            Value::Null => {}
            Value::String(digest) if valid_digest(digest) => {}
            Value::String(_) => report.error(
                "EGDDM-DIGEST-001",
                format!("{path}.digest"),
                "source digest must be 40- or 64-character hexadecimal, or null",
            ),
            _ => report.error("EGDDM-TYPE-004", format!("{path}.digest"), "expected string or null"),
        }
    }
    let line_start = source.get("line_start").and_then(Value::as_u64);
    let line_end = source.get("line_end").and_then(Value::as_u64);
    if source.get("line_start").is_some() && line_start.is_none() {
        report.error("EGDDM-LINE-001", format!("{path}.line_start"), "line_start must be a positive integer");
    }
    if source.get("line_end").is_some() && line_end.is_none() {
        report.error("EGDDM-LINE-001", format!("{path}.line_end"), "line_end must be a positive integer");
    }
    if let (Some(start), Some(end)) = (line_start, line_end) {
        if start == 0 || end == 0 || start > end {
            report.error("EGDDM-LINE-002", path, "source line range must satisfy 1 <= line_start <= line_end");
        }
    }
}

fn validate_provenance(value: &Value, path: &str, report: &mut ValidationReport) -> String {
    let Some(provenance) = object(value, path, report) else {
        return String::new();
    };
    check_allowed_keys(
        provenance,
        &[
            "authority", "sources", "extracted_by", "observed_at",
            "valid_for_revision", "valid_for_world", "confidence", "limitations",
        ],
        path,
        report,
    );

    let authority = field(value, "authority", path, report)
        .and_then(|v| enum_value(v, &format!("{path}.authority"), AUTHORITIES, report))
        .unwrap_or_default();

    if let Some(sources) = field(value, "sources", path, report)
        .and_then(|v| array(v, &format!("{path}.sources"), report))
    {
        if sources.is_empty() {
            report.error("EGDDM-PROVENANCE-001", format!("{path}.sources"), "at least one provenance source is required");
        }
        for (index, source) in sources.iter().enumerate() {
            validate_source(source, &format!("{path}.sources[{index}]"), report);
        }
    }

    if let Some(extracted_by) = field(value, "extracted_by", path, report) {
        if let Some(extractor) = object(extracted_by, &format!("{path}.extracted_by"), report) {
            check_allowed_keys(extractor, &["name", "version", "digest"], &format!("{path}.extracted_by"), report);
            for key in ["name", "version"] {
                if let Some(item) = field(extracted_by, key, &format!("{path}.extracted_by"), report) {
                    string(item, &format!("{path}.extracted_by.{key}"), report);
                }
            }
            if let Some(digest) = extractor.get("digest") {
                match digest {
                    Value::Null => {}
                    Value::String(digest) if digest.len() == 64 && hex_string(digest) => {}
                    _ => report.error(
                        "EGDDM-DIGEST-002",
                        format!("{path}.extracted_by.digest"),
                        "extractor digest must be 64-character hexadecimal, or null",
                    ),
                }
            }
        }
    }

    if let Some(observed_at) = field(value, "observed_at", path, report) {
        string(observed_at, &format!("{path}.observed_at"), report);
    }

    for key in ["valid_for_revision", "valid_for_world"] {
        if let Some(item) = provenance.get(key) {
            optional_string(item, &format!("{path}.{key}"), report);
        }
    }

    if let Some(confidence) = provenance.get("confidence") {
        match confidence {
            Value::Null => {}
            Value::Number(number) => {
                if let Some(number) = number.as_f64() {
                    if !(0.0..=1.0).contains(&number) {
                        report.error("EGDDM-CONFIDENCE-001", format!("{path}.confidence"), "confidence must be within [0, 1]");
                    }
                } else {
                    report.error("EGDDM-CONFIDENCE-001", format!("{path}.confidence"), "confidence must be finite");
                }
            }
            _ => report.error("EGDDM-TYPE-005", format!("{path}.confidence"), "expected number or null"),
        }
    }
    authority
}

fn validate_lifecycle(value: &Value, path: &str, report: &mut ValidationReport) -> (String, Option<String>) {
    let Some(lifecycle) = object(value, path, report) else {
        return (String::new(), None);
    };
    check_allowed_keys(lifecycle, &["state", "supersedes", "invalidated_by", "reason"], path, report);
    let state = field(value, "state", path, report)
        .and_then(|v| enum_value(v, &format!("{path}.state"), LIFECYCLE_STATES, report))
        .unwrap_or_default();
    let invalidated_by = lifecycle
        .get("invalidated_by")
        .and_then(|v| optional_string(v, &format!("{path}.invalidated_by"), report));
    let reason = lifecycle
        .get("reason")
        .and_then(|v| optional_string(v, &format!("{path}.reason"), report));

    if matches!(state.as_str(), "stale" | "superseded" | "revoked") {
        if invalidated_by.is_none() {
            report.error(
                "EGDDM-LIFECYCLE-001",
                path,
                "stale, superseded and revoked facts must reference the event that changed applicability",
            );
        }
        if reason.is_none() {
            report.error("EGDDM-LIFECYCLE-002", path, "non-current lifecycle state requires a reason");
        }
    }
    if state == "current" && invalidated_by.is_some() {
        report.error("EGDDM-LIFECYCLE-003", path, "current facts cannot carry invalidated_by");
    }
    (state, invalidated_by)
}

fn validate_node(
    value: &Value,
    index: usize,
    report: &mut ValidationReport,
    ids: &mut HashSet<String>,
    nodes: &mut HashMap<String, NodeMeta>,
) {
    let path = format!("nodes[{index}]");
    let Some(node) = object(value, &path, report) else {
        return;
    };
    check_allowed_keys(node, &["id", "type", "claim_state", "label", "attributes", "provenance", "lifecycle"], &path, report);

    let id = field(value, "id", &path, report).and_then(|v| string(v, &format!("{path}.id"), report));
    let node_type = field(value, "type", &path, report)
        .and_then(|v| enum_value(v, &format!("{path}.type"), NODE_TYPES, report));
    let claim_state = field(value, "claim_state", &path, report)
        .and_then(|v| enum_value(v, &format!("{path}.claim_state"), CLAIM_STATES, report));
    if let Some(label) = field(value, "label", &path, report) {
        string(label, &format!("{path}.label"), report);
    }
    if let Some(attributes) = node.get("attributes") {
        if !attributes.is_object() {
            report.error("EGDDM-TYPE-001", format!("{path}.attributes"), "attributes must be an object");
        }
    }

    let authority = field(value, "provenance", &path, report)
        .map(|v| validate_provenance(v, &format!("{path}.provenance"), report))
        .unwrap_or_default();
    let (lifecycle_state, invalidated_by) = field(value, "lifecycle", &path, report)
        .map(|v| validate_lifecycle(v, &format!("{path}.lifecycle"), report))
        .unwrap_or_default();

    if let Some(state) = claim_state.as_deref() {
        let permitted = match state {
            "passed_local" | "ci_passed" => matches!(authority.as_str(), "runtime_observed" | "benchmark_observed" | "independently_adjudicated"),
            "independently_qualified" => authority == "independently_adjudicated",
            _ => true,
        };
        if !permitted {
            report.error(
                "EGDDM-AUTHORITY-001",
                format!("{path}.claim_state"),
                format!("claim state {state:?} cannot be supported by authority {authority:?}"),
            );
        }
    }

    if let (Some(id), Some(node_type), Some(claim_state)) = (id, node_type, claim_state) {
        if !ids.insert(id.to_owned()) {
            report.error("EGDDM-ID-001", format!("{path}.id"), format!("duplicate graph identifier {id:?}"));
        }
        nodes.insert(
            id.to_owned(),
            NodeMeta {
                node_type,
                claim_state,
                authority,
                attributes: node.get("attributes").cloned().unwrap_or(Value::Null),
                lifecycle_state,
                invalidated_by,
            },
        );
    }
}

fn validate_edge(
    value: &Value,
    index: usize,
    report: &mut ValidationReport,
    ids: &mut HashSet<String>,
    node_ids: &HashSet<String>,
    contains_edges: &mut Vec<(String, String)>,
) {
    let path = format!("edges[{index}]");
    let Some(edge) = object(value, &path, report) else {
        return;
    };
    check_allowed_keys(edge, &["id", "from", "relation", "to", "attributes", "provenance", "lifecycle"], &path, report);
    let id = field(value, "id", &path, report).and_then(|v| string(v, &format!("{path}.id"), report));
    let from = field(value, "from", &path, report).and_then(|v| string(v, &format!("{path}.from"), report));
    let relation = field(value, "relation", &path, report)
        .and_then(|v| enum_value(v, &format!("{path}.relation"), RELATIONS, report));
    let to = field(value, "to", &path, report).and_then(|v| string(v, &format!("{path}.to"), report));

    if let Some(attributes) = edge.get("attributes") {
        if !attributes.is_object() {
            report.error("EGDDM-TYPE-001", format!("{path}.attributes"), "attributes must be an object");
        }
    }
    if let Some(provenance) = field(value, "provenance", &path, report) {
        validate_provenance(provenance, &format!("{path}.provenance"), report);
    }
    if let Some(lifecycle) = field(value, "lifecycle", &path, report) {
        validate_lifecycle(lifecycle, &format!("{path}.lifecycle"), report);
    }

    if let Some(from) = from.as_deref() {
        if !node_ids.contains(from) {
            report.error("EGDDM-EDGE-001", format!("{path}.from"), format!("unknown node identifier {from:?}"));
        }
    }
    if let Some(to) = to.as_deref() {
        if !node_ids.contains(to) {
            report.error("EGDDM-EDGE-002", format!("{path}.to"), format!("unknown node identifier {to:?}"));
        }
    }
    if relation.as_deref() == Some("contains") {
        if let (Some(from), Some(to)) = (from.as_deref(), to.as_deref()) {
            contains_edges.push((from.to_owned(), to.to_owned()));
        }
    }
    if let Some(id) = id {
        if !ids.insert(id.to_owned()) {
            report.error("EGDDM-ID-001", format!("{path}.id"), format!("duplicate graph identifier {id:?}"));
        }
    }
}


fn parse_rfc3339(value: &Value, path: &str, report: &mut ValidationReport) -> Option<DateTime<FixedOffset>> {
    let raw = string(value, path, report)?;
    match DateTime::parse_from_rfc3339(raw) {
        Ok(parsed) => Some(parsed),
        Err(_) => {
            report.error("EGDDM-TIME-001", path, "timestamp must be valid RFC 3339");
            None
        }
    }
}

fn validate_decision_evidence_list(
    value: &Value,
    path: &str,
    expect_admissible: bool,
    cutoff: &DateTime<FixedOffset>,
    decision_at: &DateTime<FixedOffset>,
    nodes: &HashMap<String, NodeMeta>,
    seen: &mut HashSet<String>,
    report: &mut ValidationReport,
) {
    let Some(entries) = array(value, path, report) else {
        return;
    };
    for (index, entry) in entries.iter().enumerate() {
        let entry_path = format!("{path}[{index}]");
        let Some(entry_object) = object(entry, &entry_path, report) else {
            continue;
        };
        check_allowed_keys(
            entry_object,
            &[
                "evidence_id", "available_to_actor_at", "source_observed_at",
                "valid_from", "valid_through", "admissibility", "reason",
            ],
            &entry_path,
            report,
        );

        let evidence_id = field(entry, "evidence_id", &entry_path, report)
            .and_then(|v| string(v, &format!("{entry_path}.evidence_id"), report))
            .map(str::to_owned);
        if let Some(evidence_id) = evidence_id.as_deref() {
            if !seen.insert(evidence_id.to_owned()) {
                report.error(
                    "DTEA-EVIDENCE-001",
                    format!("{entry_path}.evidence_id"),
                    format!("evidence {evidence_id:?} is duplicated across decision evidence sets"),
                );
            }
            match nodes.get(evidence_id) {
                Some(node) if node.node_type == "evidence" => {}
                Some(node) => report.error(
                    "DTEA-EVIDENCE-002",
                    format!("{entry_path}.evidence_id"),
                    format!("decision evidence reference must target an evidence node, not {}", node.node_type),
                ),
                None => report.error(
                    "DTEA-EVIDENCE-003",
                    format!("{entry_path}.evidence_id"),
                    format!("unknown evidence node {evidence_id:?}"),
                ),
            }
        }

        let available = field(entry, "available_to_actor_at", &entry_path, report)
            .and_then(|v| parse_rfc3339(v, &format!("{entry_path}.available_to_actor_at"), report));
        if expect_admissible {
            if let Some(available) = available.as_ref() {
                if available > cutoff {
                    report.error(
                        "DTEA-EVIDENCE-004",
                        format!("{entry_path}.available_to_actor_at"),
                        "consulted evidence was not available to the actor by the decision knowledge cutoff",
                    );
                }
                if available > decision_at {
                    report.error(
                        "DTEA-EVIDENCE-005",
                        format!("{entry_path}.available_to_actor_at"),
                        "consulted evidence became available after the decision",
                    );
                }
            }
        }

        if let Some(source_observed) = entry.get("source_observed_at") {
            if let Some(source_observed) = optional_string(source_observed, &format!("{entry_path}.source_observed_at"), report) {
                let parsed = DateTime::parse_from_rfc3339(&source_observed);
                match parsed {
                    Ok(source_observed) => {
                        if let Some(available) = available.as_ref() {
                            if &source_observed > available {
                                report.error(
                                    "DTEA-EVIDENCE-006",
                                    format!("{entry_path}.source_observed_at"),
                                    "source observation cannot occur after evidence became available to the actor",
                                );
                            }
                        }
                    }
                    Err(_) => report.error(
                        "EGDDM-TIME-001",
                        format!("{entry_path}.source_observed_at"),
                        "timestamp must be valid RFC 3339",
                    ),
                }
            }
        }

        let valid_from = entry
            .get("valid_from")
            .filter(|value| !value.is_null())
            .and_then(|value| parse_rfc3339(value, &format!("{entry_path}.valid_from"), report));
        let valid_through = entry
            .get("valid_through")
            .filter(|value| !value.is_null())
            .and_then(|value| parse_rfc3339(value, &format!("{entry_path}.valid_through"), report));

        if let (Some(valid_from), Some(valid_through)) = (valid_from.as_ref(), valid_through.as_ref()) {
            if valid_from > valid_through {
                report.error(
                    "DTEA-EVIDENCE-009",
                    &entry_path,
                    "evidence validity interval must satisfy valid_from <= valid_through",
                );
            }
        }

        if expect_admissible {
            if valid_from.as_ref().is_some_and(|valid_from| valid_from > decision_at) {
                report.error(
                    "DTEA-EVIDENCE-010",
                    format!("{entry_path}.valid_from"),
                    "consulted evidence is not yet effective at the decision time",
                );
            }
            if valid_through.as_ref().is_some_and(|valid_through| valid_through < decision_at) {
                report.error(
                    "DTEA-EVIDENCE-011",
                    format!("{entry_path}.valid_through"),
                    "consulted evidence expired before the decision time",
                );
            }
        }

        let admissibility = field(entry, "admissibility", &entry_path, report)
            .and_then(|v| enum_value(v, &format!("{entry_path}.admissibility"), &["admissible", "inadmissible", "indeterminate"], report));
        if let Some(admissibility) = admissibility.as_deref() {
            if expect_admissible && admissibility != "admissible" {
                report.error(
                    "DTEA-EVIDENCE-007",
                    format!("{entry_path}.admissibility"),
                    "consulted evidence must be marked admissible at decision time",
                );
            }
            if !expect_admissible && admissibility == "admissible" {
                report.error(
                    "DTEA-EVIDENCE-008",
                    format!("{entry_path}.admissibility"),
                    "excluded evidence cannot be marked admissible",
                );
            }
        }

        if let Some(reason) = field(entry, "reason", &entry_path, report) {
            string(reason, &format!("{entry_path}.reason"), report);
        }
    }
}


fn validate_decision_authority(
    value: &Value,
    path: &str,
    actor_id: Option<&str>,
    authority_scope: Option<&str>,
    policy_version: Option<&str>,
    outcome: Option<&str>,
    cutoff: &DateTime<FixedOffset>,
    decision_at: &DateTime<FixedOffset>,
    nodes: &HashMap<String, NodeMeta>,
    report: &mut ValidationReport,
) {
    let Some(authority) = object(value, path, report) else {
        return;
    };
    check_allowed_keys(
        authority,
        &[
            "evidence_id", "owner_repository", "owner_record_ref", "owner_record_type",
            "actor_id", "authority_scope", "policy_version", "source_observed_at",
            "available_to_actor_at", "valid_from", "valid_through",
            "owner_record_disposition", "reason",
        ],
        path,
        report,
    );

    let mut values = HashMap::new();
    for key in [
        "evidence_id", "owner_repository", "owner_record_ref", "owner_record_type",
        "actor_id", "authority_scope", "policy_version", "reason",
    ] {
        if let Some(item) = field(value, key, path, report)
            .and_then(|v| string(v, &format!("{path}.{key}"), report))
        {
            values.insert(key, item.to_owned());
        }
    }
    let evidence_id = values.get("evidence_id").map(String::as_str);
    let status = field(value, "owner_record_disposition", path, report)
        .and_then(|v| enum_value(
            v,
            &format!("{path}.owner_record_disposition"),
            &[
                "source_declares_authorized", "source_declares_unauthorized",
                "expired", "revoked", "contested", "indeterminate",
            ],
            report,
        ));

    let source_observed = field(value, "source_observed_at", path, report)
        .and_then(|v| parse_rfc3339(v, &format!("{path}.source_observed_at"), report));
    let available = field(value, "available_to_actor_at", path, report)
        .and_then(|v| parse_rfc3339(v, &format!("{path}.available_to_actor_at"), report));

    let valid_from = field(value, "valid_from", path, report).and_then(|v| {
        if v.is_null() {
            None
        } else {
            parse_rfc3339(v, &format!("{path}.valid_from"), report)
        }
    });
    let valid_through = field(value, "valid_through", path, report).and_then(|v| {
        if v.is_null() {
            None
        } else {
            parse_rfc3339(v, &format!("{path}.valid_through"), report)
        }
    });

    if let (Some(source_observed), Some(available)) = (source_observed.as_ref(), available.as_ref()) {
        if source_observed > available {
            report.error(
                "DTEA-AUTH-001",
                format!("{path}.source_observed_at"),
                "authority source observation cannot occur after it became available to the actor",
            );
        }
        if available > cutoff || available > decision_at {
            report.error(
                "DTEA-AUTH-002",
                format!("{path}.available_to_actor_at"),
                "authority evidence was not available to the actor by the decision knowledge cutoff",
            );
        }
    }

    if let Some(valid_from) = valid_from.as_ref() {
        if valid_from > decision_at {
            report.error(
                "DTEA-AUTH-003",
                format!("{path}.valid_from"),
                "authority evidence is not yet effective at the decision time",
            );
        }
    }
    if let Some(valid_through) = valid_through.as_ref() {
        if valid_through < decision_at {
            report.error(
                "DTEA-AUTH-004",
                format!("{path}.valid_through"),
                "authority evidence expired before the decision time",
            );
        }
    }

    if status.as_deref() == Some("source_declares_authorized") {
        if valid_from.is_none() || valid_through.is_none() {
            report.error(
                "DTEA-AUTH-005",
                format!("{path}.valid_from"),
                "a source-declared authorized disposition requires an explicit bounded validity interval; missing bounds remain indeterminate",
            );
        }
    }

    if let (Some(expected), Some(actual)) = (actor_id, values.get("actor_id").map(String::as_str)) {
        if expected != actual {
            report.error("DTEA-AUTH-006", format!("{path}.actor_id"), "authority evidence actor_id does not match the decision actor_id");
        }
    }
    if let (Some(expected), Some(actual)) = (authority_scope, values.get("authority_scope").map(String::as_str)) {
        if expected != actual {
            report.error("DTEA-AUTH-007", format!("{path}.authority_scope"), "authority evidence scope does not match the decision authority_scope");
        }
    }
    if let (Some(expected), Some(actual)) = (policy_version, values.get("policy_version").map(String::as_str)) {
        if expected != actual {
            report.error("DTEA-AUTH-008", format!("{path}.policy_version"), "authority evidence policy_version does not match the decision policy_version");
        }
    }

    if let Some(evidence_id) = evidence_id {
        match nodes.get(evidence_id) {
            Some(node) if node.node_type == "evidence" => {
                if node.lifecycle_state != "current" {
                    report.error("DTEA-AUTH-009", format!("{path}.evidence_id"), "authority evidence node is not current in the resulting snapshot");
                }
                let attributes = &node.attributes;
                for (authority_key, node_key) in [
                    ("owner_repository", "owner_repository"),
                    ("owner_record_ref", "owner_record_ref"),
                    ("owner_record_type", "owner_record_type"),
                    ("actor_id", "actor_id"),
                    ("authority_scope", "authority_scope"),
                    ("policy_version", "policy_version"),
                    ("owner_record_disposition", "owner_record_disposition"),
                ] {
                    let from_record = values.get(authority_key).map(String::as_str)
                        .or_else(|| status.as_deref().filter(|_| authority_key == "owner_record_disposition"));
                    let from_node = attributes.get(node_key).and_then(Value::as_str);
                    if from_record.is_some() && from_node != from_record {
                        report.error(
                            "DTEA-AUTH-010",
                            format!("{path}.{authority_key}"),
                            format!("authority record {authority_key} does not match the referenced evidence node"),
                        );
                    }
                }
                for key in ["source_observed_at", "available_to_actor_at", "valid_from", "valid_through"] {
                    let record_time = value.get(key).and_then(Value::as_str);
                    let node_time = attributes.get(key).and_then(Value::as_str);
                    if record_time != node_time {
                        report.error(
                            "DTEA-AUTH-011",
                            format!("{path}.{key}"),
                            format!("authority record {key} does not match the referenced evidence node"),
                        );
                    }
                }
                if !matches!(node.authority.as_str(), "source_declared" | "manifest_resolved" | "runtime_observed" | "independently_adjudicated") {
                    report.error("DTEA-AUTH-012", format!("{path}.evidence_id"), "authority evidence node carries an unsupported provenance authority");
                }
            }
            Some(node) => report.error(
                "DTEA-AUTH-013",
                format!("{path}.evidence_id"),
                format!("authority evidence reference must target an evidence node, not {}", node.node_type),
            ),
            None => report.error(
                "DTEA-AUTH-014",
                format!("{path}.evidence_id"),
                format!("unknown authority evidence node {evidence_id:?}"),
            ),
        }
    }

    let action_requires_authority = matches!(
        outcome,
        Some("approved" | "accepted" | "released" | "executed" | "authorized")
    );
    if action_requires_authority && status.as_deref() != Some("source_declares_authorized") {
        report.error(
            "DTEA-AUTH-015",
            format!("{path}.owner_record_disposition"),
            "the recorded outcome requires source-declared authorization; unauthorized, expired, revoked, contested or indeterminate authority cannot support it",
        );
    }
}

fn validate_decision_payload(
    payload: &Value,
    path: &str,
    occurred_at: Option<&str>,
    state_effect: &str,
    targets: &[String],
    nodes: &HashMap<String, NodeMeta>,
    report: &mut ValidationReport,
) {
    let Some(payload_object) = object(payload, path, report) else {
        return;
    };
    check_allowed_keys(
        payload_object,
        &[
            "decision_at", "knowledge_cutoff_at", "actor_id", "authority_scope",
            "policy_version", "outcome", "rationale_summary", "consulted_evidence",
            "excluded_evidence", "later_corrections", "abstention_reason", "actor_authority",
        ],
        path,
        report,
    );

    let decision_at = field(payload, "decision_at", path, report)
        .and_then(|v| parse_rfc3339(v, &format!("{path}.decision_at"), report));
    let cutoff = field(payload, "knowledge_cutoff_at", path, report)
        .and_then(|v| parse_rfc3339(v, &format!("{path}.knowledge_cutoff_at"), report));

    let mut decision_fields = HashMap::new();
    for key in ["actor_id", "authority_scope", "policy_version", "outcome"] {
        if let Some(value) = field(payload, key, path, report)
            .and_then(|v| string(v, &format!("{path}.{key}"), report))
        {
            decision_fields.insert(key, value.to_owned());
        }
    }

    if let (Some(decision_at), Some(cutoff)) = (decision_at.as_ref(), cutoff.as_ref()) {
        if cutoff > decision_at {
            report.error(
                "DTEA-TIME-001",
                format!("{path}.knowledge_cutoff_at"),
                "knowledge cutoff cannot be later than the decision time",
            );
        }
        match occurred_at.and_then(|value| DateTime::parse_from_rfc3339(value).ok()) {
            Some(event_time) if &event_time == decision_at => {}
            Some(_) => report.error(
                "DTEA-TIME-002",
                format!("{path}.decision_at"),
                "decision_at must match the decision_recorded event's occurred_at",
            ),
            None => report.error(
                "DTEA-TIME-003",
                format!("{path}.decision_at"),
                "decision_recorded requires a valid occurred_at timestamp",
            ),
        }

        if let Some(authority) = field(payload, "actor_authority", path, report) {
            validate_decision_authority(
                authority,
                &format!("{path}.actor_authority"),
                decision_fields.get("actor_id").map(String::as_str),
                decision_fields.get("authority_scope").map(String::as_str),
                decision_fields.get("policy_version").map(String::as_str),
                decision_fields.get("outcome").map(String::as_str),
                cutoff,
                decision_at,
                nodes,
                report,
            );
        }

        if let Some(consulted) = field(payload, "consulted_evidence", path, report) {
            let mut seen = HashSet::new();
            validate_decision_evidence_list(
                consulted,
                &format!("{path}.consulted_evidence"),
                true,
                cutoff,
                decision_at,
                nodes,
                &mut seen,
                report,
            );
            if let Some(excluded) = field(payload, "excluded_evidence", path, report) {
                validate_decision_evidence_list(
                    excluded,
                    &format!("{path}.excluded_evidence"),
                    false,
                    cutoff,
                    decision_at,
                    nodes,
                    &mut seen,
                    report,
                );
            }
        } else if let Some(excluded) = field(payload, "excluded_evidence", path, report) {
            let mut seen = HashSet::new();
            validate_decision_evidence_list(
                excluded,
                &format!("{path}.excluded_evidence"),
                false,
                cutoff,
                decision_at,
                nodes,
                &mut seen,
                report,
            );
        }
    }

    if let Some(later) = payload.get("later_corrections") {
        if let Some(later) = array(later, &format!("{path}.later_corrections"), report) {
            let mut unique = HashSet::new();
            for (index, correction) in later.iter().enumerate() {
                if let Some(correction) = string(correction, &format!("{path}.later_corrections[{index}]"), report) {
                    if !unique.insert(correction.to_owned()) {
                        report.error("DTEA-EVENT-001", format!("{path}.later_corrections[{index}]"), "duplicate correction event reference");
                    }
                }
            }
        }
    }

    if state_effect != "introduces" && state_effect != "observes" {
        report.error(
            "DTEA-EVENT-002",
            format!("{path}"),
            "decision_recorded must use state_effect=introduces or observes",
        );
    }
    if !targets.iter().any(|target| nodes.get(target).is_some_and(|node| node.node_type == "decision")) {
        report.error(
            "DTEA-EVENT-003",
            format!("{path}"),
            "decision_recorded must target at least one decision node",
        );
    }
}

fn validate_event(
    value: &Value,
    index: usize,
    report: &mut ValidationReport,
    event_ids: &mut HashSet<String>,
    nodes: &HashMap<String, NodeMeta>,
    invalidation_targets: &mut HashMap<String, String>,
    authority_evidence_targets: &mut HashSet<String>,
) {
    let path = format!("events[{index}]");
    let Some(event) = object(value, &path, report) else {
        return;
    };
    check_allowed_keys(
        event,
        &[
            "id", "kind", "occurred_at", "observed_at", "effective_revision",
            "prior_revision", "targets", "state_effect", "provenance", "payload", "validity",
        ],
        &path,
        report,
    );

    let id = field(value, "id", &path, report).and_then(|v| string(v, &format!("{path}.id"), report));
    let kind = field(value, "kind", &path, report)
        .and_then(|v| enum_value(v, &format!("{path}.kind"), EVENT_KINDS, report));
    let effect = field(value, "state_effect", &path, report)
        .and_then(|v| enum_value(v, &format!("{path}.state_effect"), EVENT_EFFECTS, report));

    let occurred_at = event
        .get("occurred_at")
        .and_then(|value| optional_string(value, &format!("{path}.occurred_at"), report));
    if let Some(occurred_at) = occurred_at.as_deref() {
        if DateTime::parse_from_rfc3339(occurred_at).is_err() {
            report.error("EGDDM-TIME-001", format!("{path}.occurred_at"), "timestamp must be valid RFC 3339");
        }
    }
    if let Some(observed_at) = field(value, "observed_at", &path, report) {
        parse_rfc3339(observed_at, &format!("{path}.observed_at"), report);
    }

    if let Some(revision) = field(value, "effective_revision", &path, report)
        .and_then(|v| string(v, &format!("{path}.effective_revision"), report))
    {
        if !valid_revision(revision) {
            report.error("EGDDM-REVISION-001", format!("{path}.effective_revision"), "revision must be 7-64 hexadecimal characters");
        }
    }
    if let Some(prior) = event.get("prior_revision") {
        if let Some(prior) = optional_string(prior, &format!("{path}.prior_revision"), report) {
            if !valid_revision(&prior) {
                report.error("EGDDM-REVISION-002", format!("{path}.prior_revision"), "prior revision must be 7-64 hexadecimal characters");
            }
        }
    }

    let mut targets_for_event = Vec::new();
    if let Some(targets) = field(value, "targets", &path, report)
        .and_then(|v| array(v, &format!("{path}.targets"), report))
    {
        if targets.is_empty() {
            report.error("EGDDM-EVENT-001", format!("{path}.targets"), "event must identify at least one target");
        }
        let mut unique_targets = HashSet::new();
        for (target_index, target) in targets.iter().enumerate() {
            if let Some(target) = string(target, &format!("{path}.targets[{target_index}]"), report) {
                if !nodes.contains_key(target) {
                    report.error("EGDDM-EVENT-002", format!("{path}.targets[{target_index}]"), format!("unknown graph target {target:?}"));
                }
                if !unique_targets.insert(target.to_owned()) {
                    report.error("EGDDM-EVENT-003", format!("{path}.targets[{target_index}]"), "duplicate target in event");
                }
                targets_for_event.push(target.to_owned());
            }
        }
    }

    if let Some(payload) = field(value, "payload", &path, report) {
        if !payload.is_object() {
            report.error("EGDDM-TYPE-001", format!("{path}.payload"), "event payload must be an object");
        }
    }
    if let Some(provenance) = field(value, "provenance", &path, report) {
        validate_provenance(provenance, &format!("{path}.provenance"), report);
    }

    if let (Some(id), Some(kind), Some(effect)) = (id, kind, effect) {
        if !event_ids.insert(id.to_owned()) {
            report.error("EGDDM-ID-002", format!("{path}.id"), format!("duplicate event identifier {id:?}"));
        }
        if kind == "evidence_invalidated" || kind == "evidence_superseded" {
            let expected_state = if kind == "evidence_invalidated" { "stale" } else { "superseded" };
            for target in &targets_for_event {
                if let Some(node) = nodes.get(target) {
                    if node.node_type != "evidence" {
                        report.error(
                            "EGDDM-EVENT-004",
                            format!("{path}.targets"),
                            format!("{kind} must target evidence nodes; {target:?} is {}", node.node_type),
                        );
                    }
                    invalidation_targets.insert(target.clone(), id.to_owned());
                    if node.lifecycle_state != expected_state {
                        report.error(
                            "EGDDM-EVENT-005",
                            format!("{path}.targets"),
                            format!("target {target:?} must have lifecycle state {expected_state:?} in the resulting snapshot"),
                        );
                    }
                    if node.invalidated_by.as_deref() != Some(id.as_str()) {
                        report.error(
                            "EGDDM-EVENT-006",
                            format!("{path}.targets"),
                            format!("target {target:?} lifecycle.invalidated_by must reference event {id:?}"),
                        );
                    }
                }
            }
            let expected_effect = if kind == "evidence_invalidated" { "invalidates" } else { "supersedes" };
            if effect != expected_effect {
                report.error("EGDDM-EVENT-007", format!("{path}.state_effect"), format!("{kind} must use state_effect={expected_effect}"));
            }
        }
        if kind == "authority_evidence_recorded" {
            for target in &targets_for_event {
                authority_evidence_targets.insert(target.clone());
            }
            if effect != "observes" {
                report.error(
                    "DTEA-AUTH-016",
                    format!("{path}.state_effect"),
                    "authority_evidence_recorded must use state_effect=observes",
                );
            }
            let payload = event.get("payload");
            let evidence_id = payload
                .and_then(|value| value.get("evidence_id"))
                .and_then(Value::as_str);
            if evidence_id.is_none_or(|id| !targets_for_event.iter().any(|target| target == id)) {
                report.error(
                    "DTEA-AUTH-017",
                    format!("{path}.payload.evidence_id"),
                    "authority evidence event payload evidence_id must be one of its exact targets",
                );
            }
            let event_observed_at = event.get("observed_at")
                .and_then(Value::as_str)
                .and_then(|value| DateTime::parse_from_rfc3339(value).ok());
            for target in &targets_for_event {
                match nodes.get(target) {
                    Some(node) if node.node_type == "evidence" => {
                        if let (Some(payload), Some(observed)) = (payload, event_observed_at.as_ref()) {
                            let actor = payload.get("actor_id").and_then(Value::as_str);
                            let scope = payload.get("authority_scope").and_then(Value::as_str);
                            let policy = payload.get("policy_version").and_then(Value::as_str);
                            for (key, actual) in [
                                ("owner_repository", payload.get("owner_repository").and_then(Value::as_str)),
                                ("owner_record_ref", payload.get("owner_record_ref").and_then(Value::as_str)),
                                ("owner_record_type", payload.get("owner_record_type").and_then(Value::as_str)),
                                ("actor_id", actor),
                                ("authority_scope", scope),
                                ("policy_version", policy),
                                ("source_observed_at", payload.get("source_observed_at").and_then(Value::as_str)),
                                ("available_to_actor_at", payload.get("available_to_actor_at").and_then(Value::as_str)),
                                ("valid_from", payload.get("valid_from").and_then(Value::as_str)),
                                ("valid_through", payload.get("valid_through").and_then(Value::as_str)),
                                ("owner_record_disposition", payload.get("owner_record_disposition").and_then(Value::as_str)),
                            ] {
                                if actual != node.attributes.get(key).and_then(Value::as_str) {
                                    report.error(
                                        "DTEA-AUTH-018",
                                        format!("{path}.payload.{key}"),
                                        format!("authority event {key} does not match its evidence node"),
                                    );
                                }
                            }
                            let available = payload.get("available_to_actor_at")
                                .and_then(Value::as_str)
                                .and_then(|value| DateTime::parse_from_rfc3339(value).ok());
                            if available.as_ref().is_none_or(|available| available < observed) {
                                report.error(
                                    "DTEA-AUTH-019",
                                    format!("{path}.payload.available_to_actor_at"),
                                    "authority evidence cannot be available to the actor before the event was observed",
                                );
                            }
                            let source_observed = payload.get("source_observed_at")
                                .and_then(Value::as_str)
                                .and_then(|value| DateTime::parse_from_rfc3339(value).ok());
                            let event_occurred = occurred_at.as_deref()
                                .and_then(|value| DateTime::parse_from_rfc3339(value).ok());
                            if source_observed.as_ref().is_none_or(|source| event_occurred.as_ref() != Some(source)) {
                                report.error(
                                    "DTEA-AUTH-021",
                                    format!("{path}.occurred_at"),
                                    "authority evidence event occurred_at must match the referenced owner's source_observed_at",
                                );
                            }
                        }
                    }
                    Some(node) => report.error(
                        "DTEA-AUTH-020",
                        format!("{path}.targets"),
                        format!("authority evidence event must target evidence nodes, not {}", node.node_type),
                    ),
                    None => {}
                }
            }
        }
        if kind == "decision_recorded" {
            let decision_payload = event.get("payload");
            if let Some(decision_payload) = decision_payload {
                validate_decision_payload(
                    decision_payload,
                    &format!("{path}.payload"),
                    occurred_at.as_deref(),
                    &effect,
                    &targets_for_event,
                    nodes,
                    report,
                );
            }
        }
        if kind == "development_action_proposed" {
            if effect != "proposes" {
                report.error("EGDDM-EVENT-008", format!("{path}.state_effect"), "development_action_proposed must use state_effect=proposes");
            }
            for target in &targets_for_event {
                if let Some(node) = nodes.get(target) {
                    if node.node_type != "development_action" || node.claim_state != "proposed" {
                        report.error(
                            "EGDDM-EVENT-009",
                            format!("{path}.targets"),
                            format!("proposed event target {target:?} must be a proposed development_action"),
                        );
                    }
                }
            }
        }
    }
}

fn validate_contains_acyclic(
    node_ids: &HashSet<String>,
    edges: &[(String, String)],
    report: &mut ValidationReport,
) {
    let mut indegree: HashMap<&str, usize> = node_ids.iter().map(|id| (id.as_str(), 0)).collect();
    let mut outgoing: HashMap<&str, Vec<&str>> = HashMap::new();

    for (from, to) in edges {
        if let (Some(degree), true) = (indegree.get_mut(to.as_str()), node_ids.contains(from)) {
            *degree += 1;
            outgoing.entry(from.as_str()).or_default().push(to.as_str());
        }
    }

    let mut queue = VecDeque::new();
    for (node, degree) in &indegree {
        if *degree == 0 {
            queue.push_back(*node);
        }
    }

    let mut visited = 0usize;
    while let Some(node) = queue.pop_front() {
        visited += 1;
        if let Some(children) = outgoing.get(node) {
            for child in children {
                if let Some(degree) = indegree.get_mut(child) {
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(child);
                    }
                }
            }
        }
    }

    if visited != node_ids.len() {
        report.error("EGDDM-GRAPH-001", "edges", "contains relationships must be acyclic");
    }
}

/// Validate one v0.1 snapshot from its parsed JSON representation.
pub fn validate_value(value: &Value) -> ValidationReport {
    let mut report = ValidationReport::default();
    let Some(root) = object(value, "$", &mut report) else {
        return report;
    };
    check_allowed_keys(
        root,
        &["schema_version", "map_id", "repository", "snapshot", "extractor", "nodes", "edges", "limitations", "events"],
        "$",
        &mut report,
    );

    if let Some(version) = field(value, "schema_version", "$", &mut report) {
        if version.as_str() != Some("0.1.0") {
            report.error("EGDDM-VERSION-001", "$.schema_version", "only schema_version 0.1.0 is supported");
        }
    }
    if let Some(map_id) = field(value, "map_id", "$", &mut report) {
        string(map_id, "$.map_id", &mut report);
    }

    if let Some(repository_value) = field(value, "repository", "$", &mut report) {
        if let Some(repository) = object(repository_value, "$.repository", &mut report) {
            check_allowed_keys(repository, &["canonical_name", "visibility", "default_branch"], "$.repository", &mut report);
            if let Some(name) = field(repository_value, "canonical_name", "$.repository", &mut report) {
                string(name, "$.repository.canonical_name", &mut report);
            }
            if let Some(visibility) = field(repository_value, "visibility", "$.repository", &mut report) {
                enum_value(visibility, "$.repository.visibility", &["public", "private", "internal", "unknown"], &mut report);
            }
        }
    }

    if let Some(snapshot_value) = field(value, "snapshot", "$", &mut report) {
        if let Some(snapshot) = object(snapshot_value, "$.snapshot", &mut report) {
            check_allowed_keys(
                snapshot,
                &["revision", "tree_digest", "tree_digest_algorithm", "observed_at", "parent_snapshot", "source_event_high_watermark", "configuration_world"],
                "$.snapshot",
                &mut report,
            );
            if let Some(revision) = snapshot.get("revision").and_then(Value::as_str) {
                if !valid_revision(revision) {
                    report.error("EGDDM-REVISION-001", "$.snapshot.revision", "revision must be 7-64 hexadecimal characters");
                }
            } else {
                report.error("EGDDM-REVISION-001", "$.snapshot.revision", "revision must be a non-empty hexadecimal string");
            }
            let algorithm = snapshot.get("tree_digest_algorithm").and_then(Value::as_str);
            let digest = snapshot.get("tree_digest").and_then(Value::as_str);
            match (algorithm, digest) {
                (Some("git_tree_oid_sha1"), Some(digest)) if digest.len() == 40 && hex_string(digest) => {}
                (Some("sha256"), Some(digest)) if digest.len() == 64 && hex_string(digest) => {}
                (Some("git_tree_oid_sha1"), _) => report.error("EGDDM-DIGEST-003", "$.snapshot.tree_digest", "git_tree_oid_sha1 requires a 40-character hexadecimal digest"),
                (Some("sha256"), _) => report.error("EGDDM-DIGEST-003", "$.snapshot.tree_digest", "sha256 requires a 64-character hexadecimal digest"),
                _ => report.error("EGDDM-DIGEST-004", "$.snapshot.tree_digest_algorithm", "unsupported or missing tree digest algorithm"),
            }
            if let Some(observed) = field(snapshot_value, "observed_at", "$.snapshot", &mut report) {
                string(observed, "$.snapshot.observed_at", &mut report);
            }
            if let Some(world) = snapshot.get("configuration_world") {
                if !world.is_object() {
                    report.error("EGDDM-TYPE-001", "$.snapshot.configuration_world", "configuration_world must be an object");
                }
            }
        }
    }

    if let Some(extractor_value) = field(value, "extractor", "$", &mut report) {
        if let Some(extractor) = object(extractor_value, "$.extractor", &mut report) {
            check_allowed_keys(extractor, &["name", "version", "build_digest"], "$.extractor", &mut report);
            for key in ["name", "version"] {
                if let Some(item) = field(extractor_value, key, "$.extractor", &mut report) {
                    string(item, &format!("$.extractor.{key}"), &mut report);
                }
            }
            if let Some(digest) = extractor.get("build_digest") {
                match digest {
                    Value::Null => {}
                    Value::String(digest) if digest.len() == 64 && hex_string(digest) => {}
                    _ => report.error("EGDDM-DIGEST-005", "$.extractor.build_digest", "build_digest must be 64-character hexadecimal, or null"),
                }
            }
        }
    }

    let mut ids = HashSet::new();
    let mut nodes = HashMap::new();
    if let Some(node_values) = field(value, "nodes", "$", &mut report)
        .and_then(|v| array(v, "$.nodes", &mut report))
    {
        if node_values.is_empty() {
            report.error("EGDDM-GRAPH-002", "$.nodes", "snapshot must contain at least one node");
        }
        for (index, node) in node_values.iter().enumerate() {
            validate_node(node, index, &mut report, &mut ids, &mut nodes);
        }
    }
    let node_ids: HashSet<String> = nodes.keys().cloned().collect();

    let mut contains_edges = Vec::new();
    if let Some(edge_values) = field(value, "edges", "$", &mut report)
        .and_then(|v| array(v, "$.edges", &mut report))
    {
        for (index, edge) in edge_values.iter().enumerate() {
            validate_edge(edge, index, &mut report, &mut ids, &node_ids, &mut contains_edges);
        }
    }

    let mut event_ids = HashSet::new();
    let mut invalidation_targets = HashMap::new();
    let mut authority_evidence_targets = HashSet::new();
    if let Some(event_values) = value.get("events") {
        if let Some(event_values) = array(event_values, "$.events", &mut report) {
            for (index, event) in event_values.iter().enumerate() {
                validate_event(
                    event, index, &mut report, &mut event_ids, &nodes,
                    &mut invalidation_targets, &mut authority_evidence_targets,
                );
            }
        }
    }

    if let Some(event_values) = value.get("events").and_then(Value::as_array) {
        for (index, event) in event_values.iter().enumerate() {
            if event.get("kind").and_then(Value::as_str) != Some("decision_recorded") {
                continue;
            }
            let event_path = format!("events[{index}]");
            let Some(payload) = event.get("payload") else {
                continue;
            };
            let Some(decision_at) = payload
                .get("decision_at")
                .and_then(Value::as_str)
                .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
            else {
                continue;
            };
            if let Some(corrections) = payload.get("later_corrections").and_then(Value::as_array) {
                for (correction_index, correction) in corrections.iter().enumerate() {
                    let Some(correction_id) = correction.as_str() else {
                        continue;
                    };
                    if !event_ids.contains(correction_id) {
                        report.error(
                            "DTEA-EVENT-004",
                            format!("{event_path}.payload.later_corrections[{correction_index}]"),
                            format!("unknown later-correction event {correction_id:?}"),
                        );
                        continue;
                    }
                    let correction_event = event_values.iter().find(|candidate| {
                        candidate.get("id").and_then(Value::as_str) == Some(correction_id)
                    });
                    let correction_time = correction_event
                        .and_then(|candidate| candidate.get("occurred_at"))
                        .and_then(Value::as_str)
                        .and_then(|value| DateTime::parse_from_rfc3339(value).ok());
                    match correction_time {
                        Some(correction_time) if correction_time > decision_at => {}
                        Some(_) => report.error(
                            "DTEA-EVENT-005",
                            format!("{event_path}.payload.later_corrections[{correction_index}]"),
                            "a later correction must have an effective occurred_at after decision_at",
                        ),
                        None => report.error(
                            "DTEA-EVENT-006",
                            format!("{event_path}.payload.later_corrections[{correction_index}]"),
                            "later correction event requires a valid occurred_at timestamp",
                        ),
                    }
                }
            }
        }
    }

    if let Some(event_values) = value.get("events").and_then(Value::as_array) {
        for (index, event) in event_values.iter().enumerate() {
            if event.get("kind").and_then(Value::as_str) != Some("decision_recorded") {
                continue;
            }
            let authority_id = event
                .get("payload")
                .and_then(|payload| payload.get("actor_authority"))
                .and_then(|authority| authority.get("evidence_id"))
                .and_then(Value::as_str);
            if let Some(authority_id) = authority_id {
                if !authority_evidence_targets.contains(authority_id) {
                    report.error(
                        "DTEA-AUTH-018",
                        format!("events[{index}].payload.actor_authority.evidence_id"),
                        "decision authority evidence must have a corresponding authority_evidence_recorded event in the map event slice",
                    );
                }
            }
        }
    }

    for (node_id, meta) in &nodes {
        if let Some(event_id) = &meta.invalidated_by {
            if !event_ids.contains(event_id) {
                report.error(
                    "EGDDM-LIFECYCLE-004",
                    format!("nodes[{node_id}].lifecycle.invalidated_by"),
                    format!("lifecycle references unknown event {event_id:?}"),
                );
            }
        }
        if meta.lifecycle_state == "stale" && meta.node_type == "evidence" && !invalidation_targets.contains_key(node_id) {
            report.error(
                "EGDDM-LIFECYCLE-005",
                format!("nodes[{node_id}].lifecycle"),
                "stale evidence must be explained by an evidence_invalidated event in the snapshot event slice",
            );
        }
    }

    if let Some(snapshot) = value.get("snapshot") {
        if let Some(watermark) = snapshot.get("source_event_high_watermark") {
            if let Some(watermark) = optional_string(watermark, "$.snapshot.source_event_high_watermark", &mut report) {
                if !event_ids.contains(&watermark) {
                    report.error(
                        "EGDDM-EVENT-010",
                        "$.snapshot.source_event_high_watermark",
                        format!("event watermark {watermark:?} is not present in the snapshot event slice"),
                    );
                }
            }
        }
    }

    validate_contains_acyclic(&node_ids, &contains_edges, &mut report);
    report.sort();
    report
}

/// Parse and validate a JSON snapshot. A parse error is distinct from a
/// parseable map that violates the v0.1 contract.
pub fn validate_json(input: &str) -> Result<ValidationReport, serde_json::Error> {
    let value: Value = serde_json::from_str(input)?;
    Ok(validate_value(&value))
}


const FEDERATION_RELATIONS: &[&str] = &[
    "source_declared_integration",
    "declares_role_for",
    "shared_schema_candidate",
    "benchmark_composition",
];

const FEDERATION_STATUSES: &[&str] = &[
    "source_declared_unverified",
    "contract_versioned_unqualified",
    "independently_qualified",
];

fn validate_federation_snapshot(
    value: &Value,
    path: &str,
    report: &mut ValidationReport,
) {
    let Some(snapshot) = object(value, path, report) else {
        return;
    };
    check_allowed_keys(
        snapshot,
        &[
            "revision", "tree_digest", "tree_digest_algorithm",
            "observed_at", "configuration_world",
        ],
        path,
        report,
    );
    if let Some(revision) = field(value, "revision", path, report)
        .and_then(|v| string(v, &format!("{path}.revision"), report))
    {
        if revision.len() != 40 || !hex_string(revision) {
            report.error("EGDDM-REVISION-001", format!("{path}.revision"), "federation subject revision must be a 40-character Git SHA-1 OID");
        }
    }
    if let Some(tree) = field(value, "tree_digest", path, report)
        .and_then(|v| string(v, &format!("{path}.tree_digest"), report))
    {
        if tree.len() != 40 || !hex_string(tree) {
            report.error("EGDDM-DIGEST-003", format!("{path}.tree_digest"), "federation tree digest must be a 40-character Git tree OID");
        }
    }
    if let Some(algorithm) = field(value, "tree_digest_algorithm", path, report) {
        if algorithm.as_str() != Some("git_tree_oid_sha1") {
            report.error("EGDDM-DIGEST-004", format!("{path}.tree_digest_algorithm"), "federation v0.1 requires git_tree_oid_sha1");
        }
    }
    if let Some(observed) = field(value, "observed_at", path, report) {
        parse_rfc3339(observed, &format!("{path}.observed_at"), report);
    }
    if let Some(world) = field(value, "configuration_world", path, report) {
        if !world.is_object() {
            report.error("EGDDM-TYPE-001", format!("{path}.configuration_world"), "configuration_world must be an object");
        }
    }
}


fn validate_federation_manifest_contract(
    value: &Value,
    path: &str,
    report: &mut ValidationReport,
) {
    let Some(contract) = object(value, path, report) else {
        return;
    };
    check_allowed_keys(
        contract,
        &["schema_version", "schema_path", "schema_blob_oid", "schema_artifact_commit_sha"],
        path,
        report,
    );
    if field(value, "schema_version", path, report).and_then(Value::as_str) != Some("0.1.1") {
        report.error(
            "EGDDM-FEDERATION-040",
            format!("{path}.schema_version"),
            "manifest_contract schema_version must be 0.1.1",
        );
    }
    if let Some(schema_path) = field(value, "schema_path", path, report) {
        string(schema_path, &format!("{path}.schema_path"), report);
    }
    for key in ["schema_blob_oid", "schema_artifact_commit_sha"] {
        if let Some(oid) = field(value, key, path, report)
            .and_then(|v| string(v, &format!("{path}.{key}"), report))
        {
            if oid.len() != 40 || !hex_string(oid) {
                report.error(
                    "EGDDM-FEDERATION-041",
                    format!("{path}.{key}"),
                    "manifest schema pin must be a 40-character Git object OID",
                );
            }
        }
    }
}

fn validate_federation_contract(
    value: &Value,
    path: &str,
    report: &mut ValidationReport,
) {
    let Some(contract) = object(value, path, report) else {
        return;
    };
    check_allowed_keys(
        contract,
        &[
            "schema_version", "schema_path", "schema_blob_oid",
            "schema_artifact_commit_sha", "status", "claim_ceiling",
        ],
        path,
        report,
    );
    if field(value, "schema_version", path, report).and_then(Value::as_str) != Some("0.1.0") {
        report.error("EGDDM-FEDERATION-001", format!("{path}.schema_version"), "shared contract schema_version must be 0.1.0");
    }
    for key in ["schema_path", "claim_ceiling"] {
        if let Some(item) = field(value, key, path, report) {
            string(item, &format!("{path}.{key}"), report);
        }
    }
    for key in ["schema_blob_oid", "schema_artifact_commit_sha"] {
        if let Some(item) = field(value, key, path, report)
            .and_then(|v| string(v, &format!("{path}.{key}"), report))
        {
            if item.len() != 40 || !hex_string(item) {
                report.error("EGDDM-DIGEST-006", format!("{path}.{key}"), "field must be a 40-character Git object OID");
            }
        }
    }
    if let Some(status) = field(value, "status", path, report) {
        enum_value(status, &format!("{path}.status"), &[
            "provisional_unqualified", "schema_shape_checked", "independently_qualified",
        ], report);
    }
}

fn validate_federation_member(
    value: &Value,
    index: usize,
    report: &mut ValidationReport,
    project_ids: &mut HashSet<String>,
    repository_names: &mut HashSet<String>,
    map_ids: &mut HashSet<String>,
    member_bindings: &mut HashMap<String, (String, String, String)>,
) {
    let path = format!("members[{index}]");
    let Some(member) = object(value, &path, report) else {
        return;
    };
    check_allowed_keys(
        member,
        &[
            "project_id", "repository_full_name", "visibility",
            "default_branch", "role_summaries", "map",
        ],
        &path,
        report,
    );

    let project_id = field(value, "project_id", &path, report)
        .and_then(|v| string(v, &format!("{path}.project_id"), report))
        .map(str::to_owned);
    let repository = field(value, "repository_full_name", &path, report)
        .and_then(|v| string(v, &format!("{path}.repository_full_name"), report))
        .map(str::to_owned);
    if let Some(visibility) = field(value, "visibility", &path, report) {
        enum_value(visibility, &format!("{path}.visibility"), &["public", "private", "internal", "unknown"], report);
    }
    if let Some(branch) = field(value, "default_branch", &path, report) {
        string(branch, &format!("{path}.default_branch"), report);
    }
    if let Some(roles) = field(value, "role_summaries", &path, report)
        .and_then(|v| array(v, &format!("{path}.role_summaries"), report))
    {
        if roles.is_empty() {
            report.error("EGDDM-FEDERATION-002", format!("{path}.role_summaries"), "at least one bounded role summary is required");
        }
        for (role_index, role) in roles.iter().enumerate() {
            string(role, &format!("{path}.role_summaries[{role_index}]"), report);
        }
    }

    let Some(map_value) = field(value, "map", &path, report) else {
        return;
    };
    let map_path = format!("{path}.map");
    let Some(map) = object(map_value, &map_path, report) else {
        return;
    };
    check_allowed_keys(
        map,
        &[
            "map_id", "path", "blob_oid", "artifact_branch",
            "artifact_commit_sha", "artifact_tree_oid", "artifact_url",
            "subject_snapshot", "authority_summary", "node_count",
            "edge_count", "claim_ceiling",
        ],
        &map_path,
        report,
    );
    let map_id = field(map_value, "map_id", &map_path, report)
        .and_then(|v| string(v, &format!("{map_path}.map_id"), report))
        .map(str::to_owned);
    for key in ["path", "artifact_branch", "artifact_url", "claim_ceiling"] {
        if let Some(item) = field(map_value, key, &map_path, report) {
            string(item, &format!("{map_path}.{key}"), report);
        }
    }
    if let Some(url) = map.get("artifact_url").and_then(Value::as_str) {
        let member_repo = value.get("repository_full_name").and_then(Value::as_str).unwrap_or_default();
        let artifact_path = map.get("path").and_then(Value::as_str).unwrap_or_default();
        let artifact_commit = map.get("artifact_commit_sha").and_then(Value::as_str).unwrap_or_default();
        let expected_url = format!("https://github.com/{member_repo}/blob/{artifact_commit}/{artifact_path}");
        if url != expected_url {
            report.error("EGDDM-FEDERATION-003", format!("{map_path}.artifact_url"), "artifact_url must pin the exact repository, artifact commit and map path");
        }
    }
    for key in ["blob_oid", "artifact_commit_sha", "artifact_tree_oid"] {
        if let Some(item) = field(map_value, key, &map_path, report)
            .and_then(|v| string(v, &format!("{map_path}.{key}"), report))
        {
            if item.len() != 40 || !hex_string(item) {
                report.error("EGDDM-DIGEST-006", format!("{map_path}.{key}"), "field must be a 40-character Git object OID");
            }
        }
    }
    if let Some(authority) = field(map_value, "authority_summary", &map_path, report) {
        enum_value(authority, &format!("{map_path}.authority_summary"), &[
            "source_declared_projection", "mixed_source_authority_projection", "unknown",
        ], report);
    }
    if let Some(count) = field(map_value, "node_count", &map_path, report) {
        if count.as_u64().is_none_or(|value| value == 0) {
            report.error("EGDDM-FEDERATION-004", format!("{map_path}.node_count"), "node_count must be a positive integer");
        }
    }
    if let Some(count) = field(map_value, "edge_count", &map_path, report) {
        if count.as_u64().is_none() {
            report.error("EGDDM-FEDERATION-005", format!("{map_path}.edge_count"), "edge_count must be a non-negative integer");
        }
    }
    if let Some(snapshot) = field(map_value, "subject_snapshot", &map_path, report) {
        validate_federation_snapshot(snapshot, &format!("{map_path}.subject_snapshot"), report);
    }

    if let (Some(project_id), Some(repository), Some(map_id)) = (project_id, repository, map_id) {
        if !project_ids.insert(project_id.clone()) {
            report.error("EGDDM-FEDERATION-006", format!("{path}.project_id"), format!("duplicate project_id {project_id:?}"));
        }
        if !repository_names.insert(repository.clone()) {
            report.error("EGDDM-FEDERATION-007", format!("{path}.repository_full_name"), format!("repository {repository:?} is listed more than once"));
        }
        if !map_ids.insert(map_id.clone()) {
            report.error("EGDDM-FEDERATION-008", format!("{map_path}.map_id"), format!("duplicate map_id {map_id:?}"));
        }
        let snapshot_revision = map_value
            .get("subject_snapshot")
            .and_then(|value| value.get("revision"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let snapshot_tree = map_value
            .get("subject_snapshot")
            .and_then(|value| value.get("tree_digest"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        member_bindings.insert(project_id, (repository, map_id, format!("{snapshot_revision}:{snapshot_tree}")));
    }
}


fn validate_federation_compatibility_receipt(
    value: &Value,
    path: &str,
    from_project: Option<&str>,
    to_project: Option<&str>,
    members: &HashMap<String, (String, String, String)>,
    contract_id: Option<&str>,
    contract_version: Option<&str>,
    contract_blob_oid: Option<&str>,
    report: &mut ValidationReport,
) {
    let Some(receipt) = object(value, path, report) else {
        return;
    };
    check_allowed_keys(
        receipt,
        &[
            "receipt_id", "receipt_sha256", "verifier_build_sha256",
            "source_project_id", "source_revision", "source_tree_oid",
            "target_project_id", "target_revision", "target_tree_oid",
            "contract_id", "contract_version", "contract_blob_oid",
            "configuration_world_digest", "outcome", "tested_at",
        ],
        path,
        report,
    );

    for key in ["receipt_id", "contract_id", "contract_version"] {
        if let Some(item) = field(value, key, path, report) {
            string(item, &format!("{path}.{key}"), report);
        }
    }
    for key in ["receipt_sha256", "verifier_build_sha256", "configuration_world_digest"] {
        if let Some(item) = field(value, key, path, report)
            .and_then(|v| string(v, &format!("{path}.{key}"), report))
        {
            if item.len() != 64 || !hex_string(item) {
                report.error(
                    "EGDDM-FEDERATION-029",
                    format!("{path}.{key}"),
                    "compatibility receipt digest must be 64-character hexadecimal SHA-256",
                );
            }
        }
    }

    let source_project = field(value, "source_project_id", path, report)
        .and_then(|v| string(v, &format!("{path}.source_project_id"), report));
    let target_project = field(value, "target_project_id", path, report)
        .and_then(|v| string(v, &format!("{path}.target_project_id"), report));

    if source_project != from_project {
        report.error(
            "EGDDM-FEDERATION-030",
            format!("{path}.source_project_id"),
            "compatibility receipt source_project_id must match the directed relationship source",
        );
    }
    if target_project != to_project {
        report.error(
            "EGDDM-FEDERATION-031",
            format!("{path}.target_project_id"),
            "compatibility receipt target_project_id must match the directed relationship target",
        );
    }

    for (project_id, revision_key, tree_key) in [
        (source_project, "source_revision", "source_tree_oid"),
        (target_project, "target_revision", "target_tree_oid"),
    ] {
        let revision = field(value, revision_key, path, report)
            .and_then(|v| string(v, &format!("{path}.{revision_key}"), report));
        let tree = field(value, tree_key, path, report)
            .and_then(|v| string(v, &format!("{path}.{tree_key}"), report));
        if let Some(revision) = revision {
            if revision.len() != 40 || !hex_string(revision) {
                report.error("EGDDM-FEDERATION-032", format!("{path}.{revision_key}"), "revision must be a 40-character Git commit OID");
            }
        }
        if let Some(tree) = tree {
            if tree.len() != 40 || !hex_string(tree) {
                report.error("EGDDM-FEDERATION-033", format!("{path}.{tree_key}"), "tree must be a 40-character Git tree OID");
            }
        }
        if let (Some(project_id), Some(revision), Some(tree)) = (project_id, revision, tree) {
            match members.get(project_id) {
                Some((_, _, snapshot_pin)) => {
                    let expected = format!("{revision}:{tree}");
                    if snapshot_pin != &expected {
                        report.error(
                            "EGDDM-FEDERATION-034",
                            format!("{path}.{revision_key}"),
                            format!("receipt source snapshot for {project_id:?} does not match the pinned federation member map"),
                        );
                    }
                }
                None => report.error(
                    "EGDDM-FEDERATION-009",
                    format!("{path}.{revision_key}"),
                    format!("unknown receipt project_id {project_id:?}"),
                ),
            }
        }
    }


    let receipt_contract_id = value.get("contract_id").and_then(Value::as_str);
    let receipt_contract_version = value.get("contract_version").and_then(Value::as_str);
    if receipt_contract_id != contract_id {
        report.error(
            "EGDDM-FEDERATION-043",
            format!("{path}.contract_id"),
            "receipt contract_id must match the relationship contract binding",
        );
    }
    if receipt_contract_version != contract_version {
        report.error(
            "EGDDM-FEDERATION-044",
            format!("{path}.contract_version"),
            "receipt contract_version must match the relationship contract binding",
        );
    }

    let receipt_contract_blob = field(value, "contract_blob_oid", path, report)
        .and_then(|v| string(v, &format!("{path}.contract_blob_oid"), report));
    if let Some(blob) = receipt_contract_blob {
        if blob.len() != 40 || !hex_string(blob) {
            report.error("EGDDM-FEDERATION-018", format!("{path}.contract_blob_oid"), "contract_blob_oid must be a 40-character Git object OID");
        }
        if contract_blob_oid != Some(blob) {
            report.error("EGDDM-FEDERATION-035", format!("{path}.contract_blob_oid"), "receipt contract blob must match the relationship's exact contract binding");
        }
    }

    if let Some(outcome) = field(value, "outcome", path, report) {
        if outcome.as_str() != Some("passed") {
            report.error("EGDDM-FEDERATION-036", format!("{path}.outcome"), "independent compatibility receipt must record outcome=passed");
        }
    }
    if let Some(tested_at) = field(value, "tested_at", path, report) {
        parse_rfc3339(tested_at, &format!("{path}.tested_at"), report);
    }
}

fn validate_federation_relationship(
    value: &Value,
    index: usize,
    report: &mut ValidationReport,
    members: &HashMap<String, (String, String, String)>,
    shared_schema_blob_oid: Option<&str>,
) {
    let path = format!("relationships[{index}]");
    let Some(relationship) = object(value, &path, report) else {
        return;
    };
    check_allowed_keys(
        relationship,
        &[
            "from_project_id", "relation", "to_project_id", "status",
            "authority", "source_refs", "contract_binding", "compatibility_receipt", "claim_limitations",
        ],
        &path,
        report,
    );
    let from = field(value, "from_project_id", &path, report)
        .and_then(|v| string(v, &format!("{path}.from_project_id"), report));
    let to = field(value, "to_project_id", &path, report)
        .and_then(|v| string(v, &format!("{path}.to_project_id"), report));
    if from.is_some() && from == to {
        report.error("EGDDM-FEDERATION-027", &path, "federation relationship cannot be a self-edge");
    }
    if let Some(from) = from {
        if !members.contains_key(from) {
            report.error("EGDDM-FEDERATION-009", format!("{path}.from_project_id"), format!("unknown project_id {from:?}"));
        }
    }
    if let Some(to) = to {
        if !members.contains_key(to) {
            report.error("EGDDM-FEDERATION-010", format!("{path}.to_project_id"), format!("unknown project_id {to:?}"));
        }
    }
    let relation = field(value, "relation", &path, report)
        .and_then(|v| enum_value(v, &format!("{path}.relation"), FEDERATION_RELATIONS, report));
    let status = field(value, "status", &path, report)
        .and_then(|v| enum_value(v, &format!("{path}.status"), FEDERATION_STATUSES, report));
    let authority = field(value, "authority", &path, report)
        .and_then(|v| enum_value(v, &format!("{path}.authority"), &[
            "source_declared", "manifest_resolved", "compiler_resolved",
            "runtime_observed", "independently_adjudicated",
        ], report));

    if status.as_deref() == Some("independently_qualified")
        && authority.as_deref() != Some("independently_adjudicated")
    {
        report.error("EGDDM-FEDERATION-011", format!("{path}.authority"), "independently_qualified relationship requires independently_adjudicated authority");
    }
    if status.as_deref() == Some("contract_versioned_unqualified")
        && value.get("contract_binding").is_none_or(Value::is_null)
    {
        report.error("EGDDM-FEDERATION-012", format!("{path}.contract_binding"), "contract_versioned_unqualified relationship requires an exact contract binding");
    }
    if status.as_deref() == Some("independently_qualified")
        && value.get("compatibility_receipt").is_none_or(Value::is_null)
    {
        report.error(
            "EGDDM-FEDERATION-037",
            format!("{path}.compatibility_receipt"),
            "independently_qualified relationship requires a two-sided compatibility receipt",
        );
    }
    if relation.as_deref() == Some("shared_schema_candidate") {
        let bound_blob = value
            .get("contract_binding")
            .and_then(|binding| binding.get("contract_blob_oid"))
            .and_then(Value::as_str);
        if bound_blob != shared_schema_blob_oid {
            report.error(
                "EGDDM-FEDERATION-038",
                format!("{path}.contract_binding.contract_blob_oid"),
                "shared_schema_candidate must bind the exact schema blob pinned by shared_contract",
            );
        }
    }

    if let Some(sources) = field(value, "source_refs", &path, report)
        .and_then(|v| array(v, &format!("{path}.source_refs"), report))
    {
        if sources.is_empty() {
            report.error("EGDDM-FEDERATION-013", format!("{path}.source_refs"), "relationship requires at least one source reference");
        }
        for (source_index, source) in sources.iter().enumerate() {
            let source_path = format!("{path}.source_refs[{source_index}]");
            let Some(source) = object(source, &source_path, report) else {
                continue;
            };
            check_allowed_keys(
                source,
                &["repository_full_name", "path", "blob_oid", "line_start", "line_end"],
                &source_path,
                report,
            );
            let source_repo = source
                .get("repository_full_name")
                .and_then(|v| string(v, &format!("{source_path}.repository_full_name"), report));
            if let Some(source_repo) = source_repo {
                if !members.values().any(|(repository, _, _)| repository.as_str() == source_repo) {
                    report.error("EGDDM-FEDERATION-014", format!("{source_path}.repository_full_name"), format!("source repository {source_repo:?} is not a federation member"));
                }
                let endpoint_repositories = [from, to]
                    .into_iter()
                    .flatten()
                    .filter_map(|project_id| members.get(project_id).map(|(repository, _, _)| repository.as_str()))
                    .collect::<HashSet<_>>();
                if !endpoint_repositories.contains(source_repo) {
                    report.error(
                        "EGDDM-FEDERATION-039",
                        format!("{source_path}.repository_full_name"),
                        "relationship source evidence must originate from one of its two endpoint repositories",
                    );
                }
            } else if !source.contains_key("repository_full_name") {
                report.error("EGDDM-REQUIRED-001", format!("{source_path}.repository_full_name"), "required field is missing");
            }
            if let Some(source_file) = source.get("path") {
                string(source_file, &format!("{source_path}.path"), report);
            } else {
                report.error("EGDDM-REQUIRED-001", format!("{source_path}.path"), "required field is missing");
            }
            if let Some(blob_value) = source.get("blob_oid") {
                if let Some(blob) = string(blob_value, &format!("{source_path}.blob_oid"), report) {
                    if blob.len() != 40 || !hex_string(blob) {
                        report.error("EGDDM-FEDERATION-015", format!("{source_path}.blob_oid"), "source blob_oid must be a 40-character Git OID");
                    }
                }
            } else {
                report.error("EGDDM-REQUIRED-001", format!("{source_path}.blob_oid"), "required field is missing");
            }
            let start = source.get("line_start").and_then(Value::as_u64);
            let end = source.get("line_end").and_then(Value::as_u64);
            if start.is_none_or(|value| value == 0) || end.is_none_or(|value| value == 0) {
                report.error("EGDDM-FEDERATION-016", source_path.clone(), "source line range requires positive line_start and line_end");
            } else if start > end {
                report.error("EGDDM-FEDERATION-017", source_path.clone(), "source line_start must not exceed line_end");
            }
        }
    }

    if let Some(receipt) = relationship.get("compatibility_receipt") {
        if !receipt.is_null() {
            let binding = relationship.get("contract_binding");
            let bound_contract_id = binding
                .and_then(|binding| binding.get("contract_id"))
                .and_then(Value::as_str);
            let bound_contract_version = binding
                .and_then(|binding| binding.get("version"))
                .and_then(Value::as_str);
            let bound_contract = binding
                .and_then(|binding| binding.get("contract_blob_oid"))
                .and_then(Value::as_str);
            validate_federation_compatibility_receipt(
                receipt,
                &format!("{path}.compatibility_receipt"),
                from,
                to,
                members,
                bound_contract_id,
                bound_contract_version,
                bound_contract,
                report,
            );
        }
    }
    if let Some(binding) = relationship.get("contract_binding") {
        if !binding.is_null() {
            if let Some(binding) = object(binding, &format!("{path}.contract_binding"), report) {
                check_allowed_keys(binding, &["contract_id", "version", "contract_blob_oid", "status"], &format!("{path}.contract_binding"), report);
                for key in ["contract_id", "version"] {
                    if let Some(item) = binding.get(key) {
                        string(item, &format!("{path}.contract_binding.{key}"), report);
                    } else {
                        report.error("EGDDM-REQUIRED-001", format!("{path}.contract_binding.{key}"), "required field is missing");
                    }
                }
                if let Some(oid_value) = binding.get("contract_blob_oid") {
                    if let Some(oid) = string(oid_value, &format!("{path}.contract_binding.contract_blob_oid"), report) {
                        if oid.len() != 40 || !hex_string(oid) {
                            report.error("EGDDM-FEDERATION-018", format!("{path}.contract_binding.contract_blob_oid"), "contract_blob_oid must be a 40-character Git OID");
                        }
                    }
                } else {
                    report.error("EGDDM-REQUIRED-001", format!("{path}.contract_binding.contract_blob_oid"), "required field is missing");
                }
                if let Some(binding_status) = binding.get("status") {
                    let binding_status = enum_value(binding_status, &format!("{path}.contract_binding.status"), &[
                        "declared", "two_sided_match", "independently_conformance_tested",
                    ], report);
                    if status.as_deref() == Some("independently_qualified")
                        && binding_status.as_deref() != Some("independently_conformance_tested")
                    {
                        report.error("EGDDM-FEDERATION-028", format!("{path}.contract_binding.status"), "independently_qualified relationship requires independently_conformance_tested contract binding");
                    }
                } else {
                    report.error("EGDDM-REQUIRED-001", format!("{path}.contract_binding.status"), "required field is missing");
                }
            }
        }
    }
    if let Some(limitations) = field(value, "claim_limitations", &path, report)
        .and_then(|v| array(v, &format!("{path}.claim_limitations"), report))
    {
        for (limitation_index, limitation) in limitations.iter().enumerate() {
            string(limitation, &format!("{path}.claim_limitations[{limitation_index}]"), report);
        }
    }

    if relation.is_none() {
        report.error("EGDDM-FEDERATION-019", path, "relationship has no valid relation");
    }
}

fn validate_federation_join_policy(
    value: &Value,
    path: &str,
    report: &mut ValidationReport,
) {
    let Some(policy) = object(value, path, report) else {
        return;
    };
    check_allowed_keys(
        policy,
        &[
            "map_identity", "cross_project_join", "authority_non_transfer",
            "claim_promotion", "staleness",
        ],
        path,
        report,
    );
    for key in [
        "map_identity", "cross_project_join", "authority_non_transfer",
        "claim_promotion", "staleness",
    ] {
        if let Some(item) = field(value, key, path, report) {
            string(item, &format!("{path}.{key}"), report);
        }
    }
}

/// Validate a federation directory. This checks only the manifest's local
/// structure and identity pins; it does not fetch or authenticate member maps.
pub fn validate_federation_value(value: &Value) -> ValidationReport {
    let mut report = ValidationReport::default();
    let Some(root) = object(value, "$", &mut report) else {
        return report;
    };
    check_allowed_keys(
        root,
        &[
            "schema_version", "federation_id", "observed_at", "hub",
            "shared_contract", "manifest_contract", "members", "relationships",
            "join_policy", "limitations",
        ],
        "$",
        &mut report,
    );
    let federation_version = field(value, "schema_version", "$", &mut report)
        .and_then(Value::as_str);
    if !matches!(federation_version, Some("0.1.0" | "0.1.1")) {
        report.error("EGDDM-FEDERATION-020", "$.schema_version", "supported federation schema versions are 0.1.0 and 0.1.1");
    }
    if federation_version == Some("0.1.1") {
        if let Some(contract) = field(value, "manifest_contract", "$", &mut report) {
            validate_federation_manifest_contract(contract, "$.manifest_contract", &mut report);
        }
    } else if value.get("manifest_contract").is_some() {
        report.error("EGDDM-FEDERATION-042", "$.manifest_contract", "manifest_contract is introduced in federation schema_version 0.1.1");
    }
    if let Some(id) = field(value, "federation_id", "$", &mut report) {
        string(id, "$.federation_id", &mut report);
    }
    if let Some(observed) = field(value, "observed_at", "$", &mut report) {
        parse_rfc3339(observed, "$.observed_at", &mut report);
    }
    if let Some(contract) = field(value, "shared_contract", "$", &mut report) {
        validate_federation_contract(contract, "$.shared_contract", &mut report);
    }
    if let Some(hub) = field(value, "hub", "$", &mut report) {
        if let Some(hub) = object(hub, "$.hub", &mut report) {
            check_allowed_keys(
                hub,
                &[
                    "repository_full_name", "default_branch", "subject_revision",
                    "subject_tree_oid", "self_map_id", "self_map_path", "self_map_blob_oid",
                ],
                "$.hub",
                &mut report,
            );
            for key in ["repository_full_name", "default_branch", "self_map_id", "self_map_path"] {
                if let Some(item) = hub.get(key) {
                    string(item, &format!("$.hub.{key}"), &mut report);
                } else {
                    report.error("EGDDM-REQUIRED-001", format!("$.hub.{key}"), "required field is missing");
                }
            }
            for key in ["subject_revision", "subject_tree_oid", "self_map_blob_oid"] {
                if let Some(value) = hub.get(key) {
                    if let Some(item) = string(value, &format!("$.hub.{key}"), &mut report) {
                        if item.len() != 40 || !hex_string(item) {
                            report.error("EGDDM-FEDERATION-021", format!("$.hub.{key}"), "hub identity must be a 40-character Git OID");
                        }
                    }
                } else {
                    report.error("EGDDM-REQUIRED-001", format!("$.hub.{key}"), "required field is missing");
                }
            }
        }
    }

    let mut project_ids = HashSet::new();
    let mut repository_names = HashSet::new();
    let mut map_ids = HashSet::new();
    let mut members = HashMap::new();
    if let Some(member_values) = field(value, "members", "$", &mut report)
        .and_then(|v| array(v, "$.members", &mut report))
    {
        if member_values.is_empty() {
            report.error("EGDDM-FEDERATION-022", "$.members", "federation must contain at least one member");
        }
        for (index, member) in member_values.iter().enumerate() {
            validate_federation_member(
                member, index, &mut report,
                &mut project_ids, &mut repository_names, &mut map_ids, &mut members,
            );
        }
    }

    if let Some(hub) = value.get("hub").and_then(Value::as_object) {
        let hub_repo = hub.get("repository_full_name").and_then(Value::as_str);
        let hub_map_id = hub.get("self_map_id").and_then(Value::as_str);
        let hub_path = hub.get("self_map_path").and_then(Value::as_str);
        let hub_blob = hub.get("self_map_blob_oid").and_then(Value::as_str);
        let hub_member = members.values().find(|(repository, _, _)| Some(repository.as_str()) == hub_repo);
        match hub_member {
            Some((_, map_id, subject_pin)) if Some(map_id.as_str()) == hub_map_id => {
                let member_map = value.get("members")
                    .and_then(Value::as_array)
                    .and_then(|items| items.iter().find(|member| member.get("repository_full_name").and_then(Value::as_str) == hub_repo))
                    .and_then(|member| member.get("map"));
                if member_map.and_then(|map| map.get("path")).and_then(Value::as_str) != hub_path
                    || member_map.and_then(|map| map.get("blob_oid")).and_then(Value::as_str) != hub_blob
                {
                    report.error("EGDDM-FEDERATION-023", "$.hub", "hub self-map path/blob must match the hub repository's member map pin");
                }
                let expected_subject = format!(
                    "{}:{}",
                    hub.get("subject_revision").and_then(Value::as_str).unwrap_or_default(),
                    hub.get("subject_tree_oid").and_then(Value::as_str).unwrap_or_default()
                );
                if subject_pin != &expected_subject {
                    report.error("EGDDM-FEDERATION-026", "$.hub", "hub subject revision/tree must match the hub member map's source snapshot");
                }
            }
            _ => report.error("EGDDM-FEDERATION-024", "$.hub", "hub must exactly match one member map by repository and map_id"),
        }
    }

    if let Some(relationship_values) = field(value, "relationships", "$", &mut report)
        .and_then(|v| array(v, "$.relationships", &mut report))
    {
        for (index, relationship) in relationship_values.iter().enumerate() {
            let shared_schema_blob = value
                .get("shared_contract")
                .and_then(|contract| contract.get("schema_blob_oid"))
                .and_then(Value::as_str);
            validate_federation_relationship(
                relationship, index, &mut report, &members, shared_schema_blob,
            );
        }
    }
    if let Some(policy) = field(value, "join_policy", "$", &mut report) {
        validate_federation_join_policy(policy, "$.join_policy", &mut report);
    }
    if let Some(limitations) = field(value, "limitations", "$", &mut report)
        .and_then(|v| array(v, "$.limitations", &mut report))
    {
        if limitations.is_empty() {
            report.error("EGDDM-FEDERATION-025", "$.limitations", "federation must preserve explicit limitations");
        }
        for (index, limitation) in limitations.iter().enumerate() {
            string(limitation, &format!("$.limitations[{index}]"), &mut report);
        }
    }

    report.sort();
    report
}

pub fn validate_federation_json(input: &str) -> Result<ValidationReport, serde_json::Error> {
    let value: Value = serde_json::from_str(input)?;
    Ok(validate_federation_value(&value))
}


fn read_nonnegative_number(value: &Value, path: &str, report: &mut ValidationReport) -> Option<f64> {
    match value.as_f64() {
        Some(number) if number.is_finite() && number >= 0.0 => Some(number),
        _ => {
            report.error("DYNPLAN-NUMBER-001", path, "expected a finite non-negative number");
            None
        }
    }
}

fn validate_numeric_range(value: &Value, path: &str, report: &mut ValidationReport) -> Option<f64> {
    let Some(range) = object(value, path, report) else {
        return None;
    };
    check_allowed_keys(range, &["min", "expected", "max"], path, report);
    let min = field(value, "min", path, report)
        .and_then(|v| read_nonnegative_number(v, &format!("{path}.min"), report));
    let expected = field(value, "expected", path, report)
        .and_then(|v| read_nonnegative_number(v, &format!("{path}.expected"), report));
    let max = field(value, "max", path, report)
        .and_then(|v| read_nonnegative_number(v, &format!("{path}.max"), report));
    if let (Some(min), Some(expected), Some(max)) = (min, expected, max) {
        if min > expected || expected > max {
            report.error("DYNPLAN-RANGE-001", path, "numeric range must satisfy min <= expected <= max");
        }
    }
    expected
}

fn validate_plan_string_array(
    value: &Value,
    path: &str,
    report: &mut ValidationReport,
) -> Vec<String> {
    let mut output = Vec::new();
    if let Some(items) = array(value, path, report) {
        let mut seen = HashSet::new();
        for (index, item) in items.iter().enumerate() {
            if let Some(item) = string(item, &format!("{path}[{index}]"), report) {
                if !seen.insert(item.to_owned()) {
                    report.error("DYNPLAN-ID-001", format!("{path}[{index}]"), "duplicate identifier in array");
                }
                output.push(item.to_owned());
            }
        }
    }
    output
}

fn validate_development_plan_fixture_value(value: &Value) -> ValidationReport {
    let mut report = ValidationReport::default();
    let Some(root) = object(value, "$", &mut report) else {
        return report;
    };
    check_allowed_keys(
        root,
        &[
            "document_type", "schema_version", "benchmark_id", "task_id", "status",
            "observed_at", "scenario_clock", "objective", "resource_budget",
            "federation_snapshot", "hard_constraints", "evidence_state", "actions",
            "candidate_plans", "claim_ceiling",
        ],
        "$",
        &mut report,
    );

    if field(value, "document_type", "$", &mut report).and_then(Value::as_str)
        != Some("development_trajectory_planning_case")
    {
        report.error("DYNPLAN-DOCUMENT-001", "$.document_type", "unexpected DYNPLAN case document type");
    }
    if field(value, "schema_version", "$", &mut report).and_then(Value::as_str) != Some("0.1.0") {
        report.error("DYNPLAN-DOCUMENT-002", "$.schema_version", "only DYNPLAN schema_version 0.1.0 is supported");
    }
    if field(value, "benchmark_id", "$", &mut report).and_then(Value::as_str) != Some("DYNPLAN-001") {
        report.error("DYNPLAN-DOCUMENT-003", "$.benchmark_id", "benchmark_id must be DYNPLAN-001");
    }
    for key in ["task_id", "scenario_clock", "claim_ceiling"] {
        if let Some(item) = field(value, key, "$", &mut report) {
            string(item, &format!("$.{key}"), &mut report);
        }
    }
    if let Some(status) = field(value, "status", "$", &mut report) {
        enum_value(status, "$.status", &["public_synthetic_known_answer", "development", "validation", "confirmatory"], &mut report);
    }
    if let Some(observed) = field(value, "observed_at", "$", &mut report) {
        parse_rfc3339(observed, "$.observed_at", &mut report);
    }

    if let Some(objective_value) = field(value, "objective", "$", &mut report) {
        if let Some(objective) = object(objective_value, "$.objective", &mut report) {
            check_allowed_keys(objective, &["id", "description", "required_capability", "acceptance_evidence"], "$.objective", &mut report);
            for key in ["id", "description", "required_capability"] {
                if let Some(item) = field(objective_value, key, "$.objective", &mut report) {
                    string(item, &format!("$.objective.{key}"), &mut report);
                }
            }
            if let Some(items) = field(objective_value, "acceptance_evidence", "$.objective", &mut report) {
                validate_plan_string_array(items, "$.objective.acceptance_evidence", &mut report);
            }
        }
    }

    let mut budget_total = None;
    if let Some(budget_value) = field(value, "resource_budget", "$", &mut report) {
        if let Some(budget) = object(budget_value, "$.resource_budget", &mut report) {
            check_allowed_keys(
                budget,
                &["unit", "total", "time_budget_hours", "execution_authority", "mutation_authority"],
                "$.resource_budget",
                &mut report,
            );
            for key in ["unit", "execution_authority", "mutation_authority"] {
                if let Some(item) = field(budget_value, key, "$.resource_budget", &mut report) {
                    string(item, &format!("$.resource_budget.{key}"), &mut report);
                }
            }
            budget_total = field(budget_value, "total", "$.resource_budget", &mut report)
                .and_then(|v| read_nonnegative_number(v, "$.resource_budget.total", &mut report));
            if let Some(hours) = field(budget_value, "time_budget_hours", "$.resource_budget", &mut report) {
                read_nonnegative_number(hours, "$.resource_budget.time_budget_hours", &mut report);
            }
        }
    }

    if let Some(federation_value) = field(value, "federation_snapshot", "$", &mut report) {
        if let Some(federation) = object(federation_value, "$.federation_snapshot", &mut report) {
            check_allowed_keys(
                federation,
                &[
                    "manifest_repository", "manifest_path", "manifest_artifact_branch",
                    "manifest_blob_oid", "shared_schema", "members", "cross_project_contract",
                ],
                "$.federation_snapshot",
                &mut report,
            );
            for key in ["manifest_repository", "manifest_path", "manifest_artifact_branch"] {
                if let Some(item) = field(federation_value, key, "$.federation_snapshot", &mut report) {
                    string(item, &format!("$.federation_snapshot.{key}"), &mut report);
                }
            }
            if let Some(blob) = field(federation_value, "manifest_blob_oid", "$.federation_snapshot", &mut report)
                .and_then(|v| string(v, "$.federation_snapshot.manifest_blob_oid", &mut report))
            {
                if !valid_digest(&blob) || blob.len() != 40 {
                    report.error("DYNPLAN-PIN-001", "$.federation_snapshot.manifest_blob_oid", "manifest_blob_oid must be a 40-character Git OID");
                }
            }
            if let Some(shared) = field(federation_value, "shared_schema", "$.federation_snapshot", &mut report) {
                if let Some(shared) = object(shared, "$.federation_snapshot.shared_schema", &mut report) {
                    check_allowed_keys(shared, &["contract_id", "version", "schema_path", "schema_blob_oid", "status"], "$.federation_snapshot.shared_schema", &mut report);
                    for key in ["contract_id", "version", "schema_path", "status"] {
                        if let Some(item) = field(&Value::Object(shared.clone()), key, "$.federation_snapshot.shared_schema", &mut report) {
                            string(item, &format!("$.federation_snapshot.shared_schema.{key}"), &mut report);
                        }
                    }
                    if let Some(blob) = field(&Value::Object(shared.clone()), "schema_blob_oid", "$.federation_snapshot.shared_schema", &mut report)
                        .and_then(|v| string(v, "$.federation_snapshot.shared_schema.schema_blob_oid", &mut report))
                    {
                        if blob.len() != 40 || !hex_string(&blob) {
                            report.error("DYNPLAN-PIN-002", "$.federation_snapshot.shared_schema.schema_blob_oid", "schema_blob_oid must be a 40-character Git OID");
                        }
                    }
                }
            }
            if let Some(members) = field(federation_value, "members", "$.federation_snapshot", &mut report)
                .and_then(|v| array(v, "$.federation_snapshot.members", &mut report))
            {
                if members.len() < 2 {
                    report.error("DYNPLAN-FEDERATION-001", "$.federation_snapshot.members", "cross-project planning task requires at least two pinned members");
                }
                let mut member_ids = HashSet::new();
                for (index, member) in members.iter().enumerate() {
                    let member_path = format!("$.federation_snapshot.members[{index}]");
                    if let Some(member) = object(member, &member_path, &mut report) {
                        check_allowed_keys(member, &["project_id", "repository", "subject_revision", "map_path", "map_blob_oid"], &member_path, &mut report);
                        for key in ["project_id", "repository", "subject_revision", "map_path", "map_blob_oid"] {
                            if let Some(item) = field(&Value::Object(member.clone()), key, &member_path, &mut report) {
                                string(item, &format!("{member_path}.{key}"), &mut report);
                            }
                        }
                        if let Some(id) = member.get("project_id").and_then(Value::as_str) {
                            if !member_ids.insert(id.to_owned()) {
                                report.error("DYNPLAN-FEDERATION-002", format!("{member_path}.project_id"), "duplicate federation project_id");
                            }
                        }
                        for key in ["subject_revision", "map_blob_oid"] {
                            if let Some(oid) = member.get(key).and_then(Value::as_str) {
                                if oid.len() != 40 || !hex_string(oid) {
                                    report.error("DYNPLAN-PIN-003", format!("{member_path}.{key}"), "member identity must be a 40-character Git OID");
                                }
                            }
                        }
                    }
                }
            }
            if let Some(contract) = field(federation_value, "cross_project_contract", "$.federation_snapshot", &mut report) {
                if let Some(contract) = object(contract, "$.federation_snapshot.cross_project_contract", &mut report) {
                    check_allowed_keys(
                        contract,
                        &[
                            "relation", "status", "contract_binding", "two_sided_owner_conformance",
                            "independent_consumer_conformance", "critical_unknowns",
                        ],
                        "$.federation_snapshot.cross_project_contract",
                        &mut report,
                    );
                    for key in ["relation", "status", "two_sided_owner_conformance", "independent_consumer_conformance"] {
                        if let Some(item) = contract.get(key) {
                            string(item, &format!("$.federation_snapshot.cross_project_contract.{key}"), &mut report);
                        } else {
                            report.error("EGDDM-REQUIRED-001", format!("$.federation_snapshot.cross_project_contract.{key}"), "required field is missing");
                        }
                    }
                    if let Some(unknowns) = contract.get("critical_unknowns") {
                        validate_plan_string_array(unknowns, "$.federation_snapshot.cross_project_contract.critical_unknowns", &mut report);
                    }
                    if let Some(binding) = contract.get("contract_binding") {
                        if let Some(binding) = object(binding, "$.federation_snapshot.cross_project_contract.contract_binding", &mut report) {
                            check_allowed_keys(binding, &["contract_id", "version", "contract_blob_oid", "status"], "$.federation_snapshot.cross_project_contract.contract_binding", &mut report);
                            for key in ["contract_id", "version", "contract_blob_oid", "status"] {
                                if let Some(item) = binding.get(key) {
                                    string(item, &format!("$.federation_snapshot.cross_project_contract.contract_binding.{key}"), &mut report);
                                } else {
                                    report.error("EGDDM-REQUIRED-001", format!("$.federation_snapshot.cross_project_contract.contract_binding.{key}"), "required field is missing");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some(constraints) = field(value, "hard_constraints", "$", &mut report)
        .and_then(|v| array(v, "$.hard_constraints", &mut report))
    {
        if constraints.is_empty() {
            report.error("DYNPLAN-CONSTRAINT-001", "$.hard_constraints", "at least one hard constraint is required");
        }
        let mut ids = HashSet::new();
        for (index, constraint) in constraints.iter().enumerate() {
            let path = format!("$.hard_constraints[{index}]");
            if let Some(constraint) = object(constraint, &path, &mut report) {
                check_allowed_keys(constraint, &["id", "rule", "failure"], &path, &mut report);
                for key in ["id", "rule", "failure"] {
                    if let Some(item) = constraint.get(key).and_then(Value::as_str) {
                        string(&Value::String(item.to_owned()), &format!("{path}.{key}"), &mut report);
                        if key == "id" && !ids.insert(item.to_owned()) {
                            report.error("DYNPLAN-ID-001", format!("{path}.id"), "duplicate hard constraint ID");
                        }
                    } else {
                        report.error("EGDDM-REQUIRED-001", format!("{path}.{key}"), "required string is missing");
                    }
                }
            }
        }
    }

    let mut evidence_ids = HashSet::new();
    if let Some(evidence_values) = field(value, "evidence_state", "$", &mut report)
        .and_then(|v| array(v, "$.evidence_state", &mut report))
    {
        for (index, evidence) in evidence_values.iter().enumerate() {
            let path = format!("$.evidence_state[{index}]");
            if let Some(evidence) = object(evidence, &path, &mut report) {
                check_allowed_keys(evidence, &["id", "status", "scope", "limitation"], &path, &mut report);
                for key in ["id", "status", "scope", "limitation"] {
                    if let Some(item) = evidence.get(key).and_then(Value::as_str) {
                        if key == "id" && !evidence_ids.insert(item.to_owned()) {
                            report.error("DYNPLAN-ID-002", format!("{path}.id"), "duplicate evidence ID");
                        }
                        string(&Value::String(item.to_owned()), &format!("{path}.{key}"), &mut report);
                    } else {
                        report.error("EGDDM-REQUIRED-001", format!("{path}.{key}"), "required string is missing");
                    }
                }
            }
        }
    }

    let mut action_ids = HashSet::new();
    let mut action_costs = HashMap::new();
    let action_values = field(value, "actions", "$", &mut report)
        .and_then(|v| array(v, "$.actions", &mut report))
        .cloned()
        .unwrap_or_default();
    for (index, action) in action_values.iter().enumerate() {
        let path = format!("$.actions[{index}]");
        if let Some(action) = object(action, &path, &mut report) {
            check_allowed_keys(
                action,
                &[
                    "id", "kind", "description", "cost", "duration_hours",
                    "prerequisites", "resolves", "produces", "risk", "reversible",
                    "execution_authority_required", "acceptance",
                ],
                &path,
                &mut report,
            );
            if let Some(id) = action.get("id").and_then(Value::as_str) {
                if !action_ids.insert(id.to_owned()) {
                    report.error("DYNPLAN-ID-003", format!("{path}.id"), "duplicate action ID");
                }
                string(&Value::String(id.to_owned()), &format!("{path}.id"), &mut report);
            } else {
                report.error("EGDDM-REQUIRED-001", format!("{path}.id"), "action id is required");
            }
            if let Some(kind) = action.get("kind") {
                enum_value(kind, &format!("{path}.kind"), &[
                    "evidence_acquisition", "owner_review", "validation_design",
                    "implementation_proposal", "analysis", "benchmark_proposal", "defer",
                ], &mut report);
            }
            for key in ["description", "acceptance"] {
                if let Some(item) = action.get(key) {
                    string(item, &format!("{path}.{key}"), &mut report);
                } else {
                    report.error("EGDDM-REQUIRED-001", format!("{path}.{key}"), "required field is missing");
                }
            }
            let cost = action.get("cost")
                .and_then(|v| validate_numeric_range(v, &format!("{path}.cost"), &mut report));
            if let (Some(id), Some(cost)) = (action.get("id").and_then(Value::as_str), cost) {
                action_costs.insert(id.to_owned(), cost);
            }
            if let Some(duration) = action.get("duration_hours") {
                validate_numeric_range(duration, &format!("{path}.duration_hours"), &mut report);
            }
            if let Some(risk) = action.get("risk") {
                enum_value(risk, &format!("{path}.risk"), &["low", "medium", "high", "unknown"], &mut report);
            }
            for key in ["reversible", "execution_authority_required"] {
                if !action.get(key).is_some_and(Value::is_boolean) {
                    report.error("DYNPLAN-TYPE-001", format!("{path}.{key}"), "expected a boolean");
                }
            }
            for key in ["prerequisites", "resolves", "produces"] {
                if let Some(items) = action.get(key) {
                    validate_plan_string_array(items, &format!("{path}.{key}"), &mut report);
                } else {
                    report.error("EGDDM-REQUIRED-001", format!("{path}.{key}"), "required array is missing");
                }
            }
        }
    }

    for (index, action) in action_values.iter().enumerate() {
        let path = format!("$.actions[{index}]");
        let action_id = action.get("id").and_then(Value::as_str).unwrap_or_default();
        if let Some(prerequisites) = action.get("prerequisites").and_then(Value::as_array) {
            for (pre_index, prerequisite) in prerequisites.iter().enumerate() {
                if let Some(prerequisite) = prerequisite.as_str() {
                    if !action_ids.contains(prerequisite) && !evidence_ids.contains(prerequisite) {
                        report.error(
                            "DYNPLAN-REF-001",
                            format!("{path}.prerequisites[{pre_index}]"),
                            format!("unknown action/evidence prerequisite {prerequisite:?}"),
                        );
                    }
                }
            }
        }
        let _ = action_id;
    }

    if let Some(plans) = field(value, "candidate_plans", "$", &mut report)
        .and_then(|v| array(v, "$.candidate_plans", &mut report))
    {
        let mut plan_ids = HashSet::new();
        for (index, plan) in plans.iter().enumerate() {
            let path = format!("$.candidate_plans[{index}]");
            if let Some(plan) = object(plan, &path, &mut report) {
                check_allowed_keys(
                    plan,
                    &[
                        "id", "actions", "declared_cost", "claimed_benefit",
                        "declared_status", "plan_rationale", "claim_state",
                        "implementation_contingency",
                    ],
                    &path,
                    &mut report,
                );
                let id = plan.get("id").and_then(Value::as_str);
                if let Some(id) = id {
                    if !plan_ids.insert(id.to_owned()) {
                        report.error("DYNPLAN-ID-004", format!("{path}.id"), "duplicate plan ID");
                    }
                } else {
                    report.error("EGDDM-REQUIRED-001", format!("{path}.id"), "plan id is required");
                }
                let action_refs = plan.get("actions")
                    .map(|v| validate_plan_string_array(v, &format!("{path}.actions"), &mut report))
                    .unwrap_or_default();
                let mut sum = 0.0;
                for action_id in action_refs {
                    match action_costs.get(&action_id) {
                        Some(cost) => sum += cost,
                        None => report.error("DYNPLAN-REF-002", format!("{path}.actions"), format!("unknown action {action_id:?}")),
                    }
                }
                let declared_cost = plan.get("declared_cost")
                    .and_then(|v| read_nonnegative_number(v, &format!("{path}.declared_cost"), &mut report));
                if let Some(declared_cost) = declared_cost {
                    if (declared_cost - sum).abs() > 0.000_001 {
                        report.error("DYNPLAN-COST-001", format!("{path}.declared_cost"), format!("declared plan cost {declared_cost} does not equal expected action cost sum {sum}"));
                    }
                }
                for key in ["claimed_benefit", "plan_rationale"] {
                    if let Some(item) = plan.get(key) {
                        string(item, &format!("{path}.{key}"), &mut report);
                    } else {
                        report.error("EGDDM-REQUIRED-001", format!("{path}.{key}"), "required field is missing");
                    }
                }
                if let Some(status) = plan.get("declared_status") {
                    enum_value(status, &format!("{path}.declared_status"), &["ready", "conditional-feasible", "infeasible", "defer"], &mut report);
                }
                if plan.get("claim_state").and_then(Value::as_str) != Some("proposed") {
                    report.error("DYNPLAN-CLAIM-001", format!("{path}.claim_state"), "candidate development plans must remain proposed");
                }
            }
        }
    }

    if budget_total.is_none() {
        report.error("DYNPLAN-BUDGET-001", "$.resource_budget.total", "resource budget total is required");
    }
    report.sort();
    report
}

fn validate_development_plan_oracle_value(value: &Value) -> ValidationReport {
    let mut report = ValidationReport::default();
    let Some(root) = object(value, "$", &mut report) else {
        return report;
    };
    check_allowed_keys(
        root,
        &[
            "document_type", "benchmark_id", "fixture_id", "status",
            "candidate_plan_ids", "action_ids", "selected_plan",
            "selected_first_action", "conditional_next_actions",
            "implementation_authorization", "implementation_status",
            "plan_dispositions", "hard_invariants", "negative_controls",
            "metrics", "claim_ceiling",
        ],
        "$",
        &mut report,
    );
    if value.get("document_type").and_then(Value::as_str) != Some("development_trajectory_planning_oracle") {
        report.error("DYNPLAN-ORACLE-001", "$.document_type", "unexpected DYNPLAN oracle document type");
    }
    if value.get("benchmark_id").and_then(Value::as_str) != Some("DYNPLAN-001") {
        report.error("DYNPLAN-ORACLE-002", "$.benchmark_id", "benchmark_id must be DYNPLAN-001");
    }
    for key in ["fixture_id", "claim_ceiling"] {
        if let Some(item) = field(value, key, "$", &mut report) {
            string(item, &format!("$.{key}"), &mut report);
        }
    }
    if let Some(status) = field(value, "status", "$", &mut report) {
        enum_value(status, "$.status", &["public_synthetic_known_answer", "development", "validation", "confirmatory"], &mut report);
    }

    let candidate_ids = value.get("candidate_plan_ids")
        .map(|v| validate_plan_string_array(v, "$.candidate_plan_ids", &mut report))
        .unwrap_or_default();
    let action_ids = value.get("action_ids")
        .map(|v| validate_plan_string_array(v, "$.action_ids", &mut report))
        .unwrap_or_default();
    let selected_plan = field(value, "selected_plan", "$", &mut report)
        .and_then(|v| string(v, "$.selected_plan", &mut report));
    let selected_first_action = field(value, "selected_first_action", "$", &mut report)
        .and_then(|v| string(v, "$.selected_first_action", &mut report));
    if let Some(selected_plan) = selected_plan {
        if !candidate_ids.iter().any(|id| id == selected_plan) {
            report.error("DYNPLAN-ORACLE-003", "$.selected_plan", "selected_plan is absent from candidate_plan_ids");
        }
    }
    if let Some(selected_action) = selected_first_action {
        if !action_ids.iter().any(|id| id == selected_action) {
            report.error("DYNPLAN-ORACLE-004", "$.selected_first_action", "selected_first_action is absent from action_ids");
        }
    }
    if value.get("implementation_authorization").and_then(Value::as_str) != Some("not_granted") {
        report.error("DYNPLAN-ORACLE-005", "$.implementation_authorization", "benchmark oracle must not grant implementation authority");
    }
    for key in ["implementation_status"] {
        if let Some(item) = value.get(key) {
            string(item, &format!("$.{key}"), &mut report);
        } else {
            report.error("EGDDM-REQUIRED-001", format!("$.{key}"), "required field is missing");
        }
    }

    if let Some(dispositions) = field(value, "plan_dispositions", "$", &mut report)
        .and_then(|v| array(v, "$.plan_dispositions", &mut report))
    {
        let mut ids = HashSet::new();
        for (index, disposition) in dispositions.iter().enumerate() {
            let path = format!("$.plan_dispositions[{index}]");
            if let Some(disposition) = object(disposition, &path, &mut report) {
                check_allowed_keys(disposition, &["plan_id", "disposition", "reason_codes"], &path, &mut report);
                let id = disposition.get("plan_id").and_then(Value::as_str);
                if let Some(id) = id {
                    if !candidate_ids.iter().any(|candidate| candidate == id) {
                        report.error("DYNPLAN-ORACLE-006", format!("{path}.plan_id"), "disposition references unknown candidate plan");
                    }
                    if !ids.insert(id.to_owned()) {
                        report.error("DYNPLAN-ORACLE-007", format!("{path}.plan_id"), "duplicate plan disposition");
                    }
                } else {
                    report.error("EGDDM-REQUIRED-001", format!("{path}.plan_id"), "plan_id is required");
                }
                if let Some(status) = disposition.get("disposition") {
                    enum_value(status, &format!("{path}.disposition"), &["feasible", "conditional_feasible", "infeasible", "dominated", "defer", "indeterminate"], &mut report);
                }
                if let Some(codes) = disposition.get("reason_codes") {
                    validate_plan_string_array(codes, &format!("{path}.reason_codes"), &mut report);
                }
            }
        }
        if ids.len() != candidate_ids.len() {
            report.error("DYNPLAN-ORACLE-008", "$.plan_dispositions", "oracle must have exactly one disposition for every candidate plan");
        }
    }
    for key in ["hard_invariants", "metrics"] {
        if let Some(items) = value.get(key) {
            validate_plan_string_array(items, &format!("$.{key}"), &mut report);
        } else {
            report.error("EGDDM-REQUIRED-001", format!("$.{key}"), "required array is missing");
        }
    }
    if let Some(controls) = field(value, "negative_controls", "$", &mut report)
        .and_then(|v| array(v, "$.negative_controls", &mut report))
    {
        let mut ids = HashSet::new();
        for (index, control) in controls.iter().enumerate() {
            let path = format!("$.negative_controls[{index}]");
            if let Some(control) = object(control, &path, &mut report) {
                check_allowed_keys(control, &["id", "mutation", "expected"], &path, &mut report);
                let id = control.get("id").and_then(Value::as_str);
                if let Some(id) = id {
                    if !ids.insert(id.to_owned()) {
                        report.error("DYNPLAN-ORACLE-009", format!("{path}.id"), "duplicate negative-control ID");
                    }
                } else {
                    report.error("EGDDM-REQUIRED-001", format!("{path}.id"), "negative control ID is required");
                }
                for key in ["mutation", "expected"] {
                    if let Some(item) = control.get(key) {
                        string(item, &format!("{path}.{key}"), &mut report);
                    } else {
                        report.error("EGDDM-REQUIRED-001", format!("{path}.{key}"), "required string is missing");
                    }
                }
            }
        }
        if controls.is_empty() {
            report.error("DYNPLAN-ORACLE-010", "$.negative_controls", "oracle must include negative controls");
        }
    }
    report.sort();
    report
}


fn validate_retrieval_case_value(value: &Value) -> ValidationReport {
    let mut report = ValidationReport::default();
    let Some(root) = object(value, "$", &mut report) else {
        return report;
    };
    check_allowed_keys(
        root,
        &[
            "document_type", "schema_version", "benchmark_id", "task_id",
            "status", "observed_at", "evaluation_contract",
            "repository_snapshot", "cases", "claim_ceiling",
        ],
        "$",
        &mut report,
    );
    if value.get("document_type").and_then(Value::as_str) != Some("retrieval_exploration_benchmark_case") {
        report.error("DYNRETR-DOC-001", "$.document_type", "unexpected retrieval benchmark case document type");
    }
    if value.get("schema_version").and_then(Value::as_str) != Some("0.1.0")
        || value.get("benchmark_id").and_then(Value::as_str) != Some("DYNRETR-001")
    {
        report.error("DYNRETR-DOC-002", "$", "DYNRETR case requires schema_version 0.1.0 and benchmark_id DYNRETR-001");
    }
    for key in ["task_id", "claim_ceiling"] {
        if let Some(item) = field(value, key, "$", &mut report) {
            string(item, &format!("$.{key}"), &mut report);
        }
    }
    if let Some(status) = field(value, "status", "$", &mut report) {
        enum_value(status, "$.status", &["public_synthetic_known_answer", "design_only", "sealed_holdout"], &mut report);
    }
    if let Some(observed_at) = field(value, "observed_at", "$", &mut report) {
        parse_rfc3339(observed_at, "$.observed_at", &mut report);
    }

    if let Some(contract_value) = field(value, "evaluation_contract", "$", &mut report) {
        if let Some(contract) = object(contract_value, "$.evaluation_contract", &mut report) {
            check_allowed_keys(
                contract,
                &["tokenizer_id", "token_budget", "line_budget", "region_unit", "ranking_direction", "no_gold_policy"],
                "$.evaluation_contract",
                &mut report,
            );
            if let Some(tokenizer) = field(contract_value, "tokenizer_id", "$.evaluation_contract", &mut report) {
                string(tokenizer, "$.evaluation_contract.tokenizer_id", &mut report);
            }
            for key in ["token_budget", "line_budget"] {
                if let Some(item) = field(contract_value, key, "$.evaluation_contract", &mut report) {
                    if item.as_u64().is_none_or(|n| n == 0) {
                        report.error("DYNRETR-BUDGET-001", format!("$.evaluation_contract.{key}"), "budget must be a positive integer");
                    }
                }
            }
            if let Some(unit) = field(contract_value, "region_unit", "$.evaluation_contract", &mut report) {
                enum_value(unit, "$.evaluation_contract.region_unit", &["file", "symbol", "line_span", "mixed"], &mut report);
            }
            if field(contract_value, "ranking_direction", "$.evaluation_contract", &mut report).and_then(Value::as_str) != Some("lower_rank_is_better") {
                report.error("DYNRETR-CONTRACT-001", "$.evaluation_contract.ranking_direction", "ranking_direction must be lower_rank_is_better");
            }
            if let Some(policy) = field(contract_value, "no_gold_policy", "$.evaluation_contract", &mut report) {
                enum_value(policy, "$.evaluation_contract.no_gold_policy", &["explicit_abstain", "abstain_or_indeterminate"], &mut report);
            }
        }
    }

    if let Some(snapshot_value) = field(value, "repository_snapshot", "$", &mut report) {
        if let Some(snapshot) = object(snapshot_value, "$.repository_snapshot", &mut report) {
            check_allowed_keys(snapshot, &["repository_id", "revision", "tree_oid", "configuration_world"], "$.repository_snapshot", &mut report);
            if let Some(repository_id) = field(snapshot_value, "repository_id", "$.repository_snapshot", &mut report) {
                string(repository_id, "$.repository_snapshot.repository_id", &mut report);
            }
            for key in ["revision", "tree_oid"] {
                if let Some(hash) = field(snapshot_value, key, "$.repository_snapshot", &mut report)
                    .and_then(|v| string(v, &format!("$.repository_snapshot.{key}"), &mut report))
                {
                    if hash.len() != 40 || !hex_string(hash) {
                        report.error("DYNRETR-SNAPSHOT-001", format!("$.repository_snapshot.{key}"), "snapshot identity must be a 40-character hexadecimal Git OID");
                    }
                }
            }
            if let Some(world) = field(snapshot_value, "configuration_world", "$.repository_snapshot", &mut report) {
                if !world.is_object() {
                    report.error("DYNRETR-WORLD-001", "$.repository_snapshot.configuration_world", "configuration world must be an object");
                }
            }
        }
    }

    let mut case_ids = HashSet::new();
    if let Some(cases) = field(value, "cases", "$", &mut report)
        .and_then(|v| array(v, "$.cases", &mut report))
    {
        if cases.is_empty() {
            report.error("DYNRETR-CASES-001", "$.cases", "retrieval corpus must contain at least one case");
        }
        for (index, case_value) in cases.iter().enumerate() {
            let path = format!("$.cases[{index}]");
            let Some(case) = object(case_value, &path, &mut report) else {
                continue;
            };
            check_allowed_keys(
                case,
                &[
                    "case_id", "case_kind", "repository_id", "repository_revision", "repository_tree_oid", "counterfactual_pair_of",
                    "query", "configuration_world", "token_budget", "line_budget",
                    "changed_entities", "candidate_regions", "expected_disposition", "notes",
                ],
                &path,
                &mut report,
            );
            let case_id = field(case_value, "case_id", &path, &mut report)
                .and_then(|v| string(v, &format!("{path}.case_id"), &mut report));
            if let Some(case_id) = case_id {
                if !case_ids.insert(case_id.to_owned()) {
                    report.error("DYNRETR-ID-001", format!("{path}.case_id"), "duplicate retrieval case ID");
                }
            }
            if let Some(kind) = field(case_value, "case_kind", &path, &mut report) {
                enum_value(kind, &format!("{path}.case_kind"), &[
                    "positive_context_retrieval", "natural_no_gold",
                    "wrong_repository_counterfactual", "configuration_world_mismatch",
                    "temporal_revision_mismatch", "nonlocal_change_impact", "lexical_decoy",
                ], &mut report);
            }
            for key in ["repository_id", "query"] {
                if let Some(item) = field(case_value, key, &path, &mut report) {
                    string(item, &format!("{path}.{key}"), &mut report);
                }
            }

            let case_revision = field(case_value, "repository_revision", &path, &mut report)
                .and_then(|v| string(v, &format!("{path}.repository_revision"), &mut report));
            let case_tree = field(case_value, "repository_tree_oid", &path, &mut report)
                .and_then(|v| string(v, &format!("{path}.repository_tree_oid"), &mut report));
            for (key, value) in [("repository_revision", case_revision), ("repository_tree_oid", case_tree)] {
                if let Some(value) = value {
                    if value.len() != 40 || !hex_string(value) {
                        report.error("DYNRETR-SNAPSHOT-002", format!("{path}.{key}"), "case snapshot identity must be a 40-character Git OID");
                    }
                }
            }
            if let Some(pair) = case.get("counterfactual_pair_of") {
                optional_string(pair, &format!("{path}.counterfactual_pair_of"), &mut report);
            }
            if let Some(world) = field(case_value, "configuration_world", &path, &mut report) {
                if !world.is_object() {
                    report.error("DYNRETR-WORLD-001", format!("{path}.configuration_world"), "configuration world must be an object");
                }
            }
            for key in ["token_budget", "line_budget"] {
                if let Some(item) = field(case_value, key, &path, &mut report) {
                    if item.as_u64().is_none_or(|n| n == 0) {
                        report.error("DYNRETR-BUDGET-002", format!("{path}.{key}"), "case budget must be a positive integer");
                    }
                }
            }
            for key in ["changed_entities"] {
                if let Some(items) = field(case_value, key, &path, &mut report) {
                    validate_plan_string_array(items, &format!("{path}.{key}"), &mut report);
                }
            }
            if let Some(disposition) = field(case_value, "expected_disposition", &path, &mut report) {
                enum_value(disposition, &format!("{path}.expected_disposition"), &["retrieve", "abstain", "indeterminate"], &mut report);
            }
            if let Some(notes) = case.get("notes") {
                string(notes, &format!("{path}.notes"), &mut report);
            }

            if let Some(regions) = field(case_value, "candidate_regions", &path, &mut report)
                .and_then(|v| array(v, &format!("{path}.candidate_regions"), &mut report))
            {
                if regions.is_empty() {
                    report.error("DYNRETR-REGION-001", format!("{path}.candidate_regions"), "case must include at least one candidate region");
                }
                let mut region_ids = HashSet::new();
                for (region_index, region_value) in regions.iter().enumerate() {
                    let region_path = format!("{path}.candidate_regions[{region_index}]");
                    let Some(region) = object(region_value, &region_path, &mut report) else {
                        continue;
                    };
                    check_allowed_keys(
                        region,
                        &[
                            "region_id", "path", "line_start", "line_end", "token_estimate",
                            "authority", "source_revision", "availability",
                            "configuration_requirements", "lexical_terms",
                        ],
                        &region_path,
                        &mut report,
                    );
                    if let Some(id) = field(region_value, "region_id", &region_path, &mut report)
                        .and_then(|v| string(v, &format!("{region_path}.region_id"), &mut report))
                    {
                        if !region_ids.insert(id.to_owned()) {
                            report.error("DYNRETR-ID-002", format!("{region_path}.region_id"), "duplicate region ID within case");
                        }
                    }
                    if let Some(source_path) = field(region_value, "path", &region_path, &mut report) {
                        string(source_path, &format!("{region_path}.path"), &mut report);
                    }
                    let line_start = field(region_value, "line_start", &region_path, &mut report).and_then(Value::as_u64);
                    let line_end = field(region_value, "line_end", &region_path, &mut report).and_then(Value::as_u64);
                    if line_start.is_none_or(|n| n == 0) || line_end.is_none_or(|n| n == 0) || line_start > line_end {
                        report.error("DYNRETR-REGION-002", region_path.clone(), "region line span must satisfy 1 <= line_start <= line_end");
                    }
                    if let Some(tokens) = field(region_value, "token_estimate", &region_path, &mut report) {
                        if tokens.as_u64().is_none_or(|n| n == 0) {
                            report.error("DYNRETR-BUDGET-003", format!("{region_path}.token_estimate"), "token estimate must be positive");
                        }
                    }
                    if let Some(authority) = field(region_value, "authority", &region_path, &mut report) {
                        enum_value(authority, &format!("{region_path}.authority"), AUTHORITIES, &mut report);
                    }
                    let region_revision = field(region_value, "source_revision", &region_path, &mut report)
                        .and_then(|v| string(v, &format!("{region_path}.source_revision"), &mut report));
                    if let Some(revision) = region_revision {
                        if revision.len() != 40 || !hex_string(revision) {
                            report.error("DYNRETR-REVISION-001", format!("{region_path}.source_revision"), "region revision must be a 40-character Git OID");
                        }
                    }
                    let availability = field(region_value, "availability", &region_path, &mut report)
                        .and_then(|v| enum_value(v, &format!("{region_path}.availability"), &[
                            "available", "inactive_configuration", "stale_revision", "missing", "unauthorized", "unknown",
                        ], &mut report));
                    match (availability.as_deref(), region_revision, case_revision) {
                        (Some("available" | "inactive_configuration"), Some(region_rev), Some(case_rev)) if region_rev != case_rev => {
                            report.error("DYNRETR-SNAPSHOT-003", format!("{region_path}.source_revision"), "available/inactive evidence must belong to the case's exact pinned repository revision");
                        }
                        (Some("stale_revision"), Some(region_rev), Some(case_rev)) if region_rev == case_rev => {
                            report.error("DYNRETR-SNAPSHOT-004", format!("{region_path}.source_revision"), "stale_revision evidence must differ from the case's pinned repository revision");
                        }
                        _ => {}
                    }
                    let requirements = region_value.get("configuration_requirements")
                        .map(|items| validate_plan_string_array(items, &format!("{region_path}.configuration_requirements"), &mut report))
                        .unwrap_or_default();
                    if let Some(lexical) = field(region_value, "lexical_terms", &region_path, &mut report) {
                        validate_plan_string_array(lexical, &format!("{region_path}.lexical_terms"), &mut report);
                    }
                    let active_features = case_value.get("configuration_world")
                        .and_then(|world| world.get("features"))
                        .and_then(Value::as_array)
                        .map(|items| items.iter().filter_map(Value::as_str).collect::<HashSet<_>>())
                        .unwrap_or_default();
                    let requirements_satisfied = requirements.iter().all(|requirement| active_features.contains(requirement.as_str()));
                    if availability.as_deref() == Some("inactive_configuration") && (requirements.is_empty() || requirements_satisfied) {
                        report.error("DYNRETR-WORLD-002", format!("{region_path}.configuration_requirements"), "inactive_configuration region must require at least one feature absent from the case world");
                    }
                    if availability.as_deref() == Some("available") && !requirements_satisfied {
                        report.error("DYNRETR-WORLD-003", format!("{region_path}.configuration_requirements"), "available region has configuration requirements not satisfied by the case world");
                    }
                }
            }
        }
    }
    report.sort();
    report
}

fn validate_retrieval_oracle_value(value: &Value) -> ValidationReport {
    let mut report = ValidationReport::default();
    let Some(root) = object(value, "$", &mut report) else {
        return report;
    };
    check_allowed_keys(
        root,
        &[
            "document_type", "schema_version", "benchmark_id", "fixture_id", "status",
            "expected_dispositions", "negative_controls", "metrics", "claim_ceiling",
        ],
        "$",
        &mut report,
    );
    if value.get("document_type").and_then(Value::as_str) != Some("retrieval_exploration_benchmark_oracle") {
        report.error("DYNRETR-ORACLE-001", "$.document_type", "unexpected retrieval oracle document type");
    }
    if value.get("schema_version").and_then(Value::as_str) != Some("0.1.0")
        || value.get("benchmark_id").and_then(Value::as_str) != Some("DYNRETR-001")
    {
        report.error("DYNRETR-ORACLE-002", "$", "retrieval oracle requires schema_version 0.1.0 and benchmark_id DYNRETR-001");
    }
    for key in ["fixture_id", "claim_ceiling"] {
        if let Some(item) = field(value, key, "$", &mut report) {
            string(item, &format!("$.{key}"), &mut report);
        }
    }
    if let Some(status) = field(value, "status", "$", &mut report) {
        enum_value(status, "$.status", &["public_synthetic_known_answer", "design_only", "sealed_holdout"], &mut report);
    }

    let mut case_ids = HashSet::new();
    if let Some(entries) = field(value, "expected_dispositions", "$", &mut report)
        .and_then(|v| array(v, "$.expected_dispositions", &mut report))
    {
        for (index, entry_value) in entries.iter().enumerate() {
            let path = format!("$.expected_dispositions[{index}]");
            let Some(entry) = object(entry_value, &path, &mut report) else {
                continue;
            };
            check_allowed_keys(entry, &["case_id", "disposition", "relevant_regions", "forbidden_regions"], &path, &mut report);
            if let Some(id) = field(entry_value, "case_id", &path, &mut report)
                .and_then(|v| string(v, &format!("{path}.case_id"), &mut report))
            {
                if !case_ids.insert(id.to_owned()) {
                    report.error("DYNRETR-ORACLE-003", format!("{path}.case_id"), "duplicate case_id in oracle");
                }
            }
            if let Some(disposition) = field(entry_value, "disposition", &path, &mut report) {
                enum_value(disposition, &format!("{path}.disposition"), &["retrieve", "abstain", "indeterminate"], &mut report);
            }
            let relevant = entry_value.get("relevant_regions")
                .map(|v| validate_plan_string_array(v, &format!("{path}.relevant_regions"), &mut report))
                .unwrap_or_default();
            let forbidden = entry_value.get("forbidden_regions")
                .map(|v| validate_plan_string_array(v, &format!("{path}.forbidden_regions"), &mut report))
                .unwrap_or_default();
            for id in &relevant {
                if forbidden.contains(id) {
                    report.error("DYNRETR-ORACLE-004", &path, format!("region {id:?} cannot be both relevant and forbidden"));
                }
            }
            if entry_value.get("disposition").and_then(Value::as_str) == Some("abstain") && !relevant.is_empty() {
                report.error("DYNRETR-ORACLE-005", &path, "abstain disposition must not contain relevant retrieval regions");
            }
        }
    }

    if let Some(controls) = field(value, "negative_controls", "$", &mut report)
        .and_then(|v| array(v, "$.negative_controls", &mut report))
    {
        let mut ids = HashSet::new();
        for (index, control_value) in controls.iter().enumerate() {
            let path = format!("$.negative_controls[{index}]");
            let Some(control) = object(control_value, &path, &mut report) else {
                continue;
            };
            check_allowed_keys(control, &["id", "mutation", "expected"], &path, &mut report);
            if let Some(id) = field(control_value, "id", &path, &mut report)
                .and_then(|v| string(v, &format!("{path}.id"), &mut report))
            {
                if !ids.insert(id.to_owned()) {
                    report.error("DYNRETR-ORACLE-006", format!("{path}.id"), "duplicate negative-control ID");
                }
            }
            for key in ["mutation", "expected"] {
                if let Some(item) = field(control_value, key, &path, &mut report) {
                    string(item, &format!("{path}.{key}"), &mut report);
                }
            }
        }
    }
    if let Some(metrics) = field(value, "metrics", "$", &mut report) {
        validate_plan_string_array(metrics, "$.metrics", &mut report);
    }
    report.sort();
    report
}

/// Autodetect and validate an EGDDM snapshot or federation directory.
pub fn validate_document_json(input: &str) -> Result<ValidationReport, serde_json::Error> {
    let value: Value = serde_json::from_str(input)?;
    match value.get("document_type").and_then(Value::as_str) {
        Some("development_trajectory_planning_case") => Ok(validate_development_plan_fixture_value(&value)),
        Some("development_trajectory_planning_oracle") => Ok(validate_development_plan_oracle_value(&value)),
        Some("retrieval_exploration_benchmark_case") => Ok(validate_retrieval_case_value(&value)),
        Some("retrieval_exploration_benchmark_oracle") => Ok(validate_retrieval_oracle_value(&value)),
        _ if value.get("federation_id").is_some() => Ok(validate_federation_value(&value)),
        _ => Ok(validate_value(&value)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEMPORAL_FIXTURE: &str = include_str!(
        "../../../docs/development-map/fixtures/temporal-evidence-invalidation-v0.1.map.json"
    );
    const TEMPORAL_EXPECTED: &str = include_str!(
        "../../../docs/development-map/fixtures/temporal-evidence-invalidation-v0.1.expected.json"
    );

    #[test]
    fn temporal_fixture_satisfies_v01_contract() {
        let report = validate_json(TEMPORAL_FIXTURE).expect("fixture JSON must parse");
        assert!(report.is_valid(), "{:#?}", report.diagnostics);
    }

    #[test]
    fn stale_evidence_cannot_become_current_while_invalidation_event_remains() {
        let mut value: Value = serde_json::from_str(TEMPORAL_FIXTURE).expect("fixture JSON");
        let node = value["nodes"]
            .as_array_mut()
            .expect("nodes")
            .iter_mut()
            .find(|node| node["id"] == "fixture:evidence-v1")
            .expect("evidence-v1 node");
        node["lifecycle"]["state"] = Value::String("current".to_owned());
        node["lifecycle"]["invalidated_by"] = Value::Null;
        let report = validate_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "EGDDM-EVENT-005"));
    }

    #[test]
    fn compiler_or_ci_claim_cannot_be_supported_by_readme_declaration() {
        let mut value: Value = serde_json::from_str(TEMPORAL_FIXTURE).expect("fixture JSON");
        let node = value["nodes"]
            .as_array_mut()
            .expect("nodes")
            .iter_mut()
            .find(|node| node["id"] == "fixture:evidence-v2")
            .expect("evidence-v2 node");
        node["claim_state"] = Value::String("ci_passed".to_owned());
        node["provenance"]["authority"] = Value::String("source_declared".to_owned());
        let report = validate_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "EGDDM-AUTHORITY-001"));
    }

    #[test]
    fn unknown_event_target_is_rejected() {
        let mut value: Value = serde_json::from_str(TEMPORAL_FIXTURE).expect("fixture JSON");
        value["events"][0]["targets"] = Value::Array(vec![Value::String("missing:node".to_owned())]);
        let report = validate_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "EGDDM-EVENT-002"));
    }

    #[test]
    fn proposed_action_cannot_be_relabelled_as_observed() {
        let mut value: Value = serde_json::from_str(TEMPORAL_FIXTURE).expect("fixture JSON");
        let node = value["nodes"]
            .as_array_mut()
            .expect("nodes")
            .iter_mut()
            .find(|node| node["id"] == "fixture:action-v3")
            .expect("action-v3 node");
        node["claim_state"] = Value::String("observed".to_owned());
        value["events"][8]["state_effect"] = Value::String("observes".to_owned());
        let report = validate_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "EGDDM-EVENT-008"));
    }

    #[test]
    fn fixture_oracle_is_parseable_and_has_seven_controls() {
        let oracle: Value = serde_json::from_str(TEMPORAL_EXPECTED).expect("oracle JSON");
        assert_eq!(oracle["negative_controls"].as_array().map(Vec::len), Some(7));
        assert_eq!(oracle["expected_snapshots"].as_array().map(Vec::len), Some(3));
    }




    const FEDERATION_FIXTURE: &str = include_str!(
        "../../../docs/development-map/federation-v0.1.json"
    );

    #[test]
    fn four_project_federation_manifest_satisfies_local_contract() {
        let report = validate_federation_json(FEDERATION_FIXTURE).expect("federation JSON");
        assert!(report.is_valid(), "{:#?}", report.diagnostics);
    }

    const FEDERATION_V011_FIXTURE: &str = include_str!(
        "../../../docs/development-map/federation-v0.1.1.json"
    );

    #[test]
    fn federation_v011_manifest_satisfies_local_contract() {
        let report = validate_federation_json(FEDERATION_V011_FIXTURE).expect("v0.1.1 federation JSON");
        assert!(report.is_valid(), "{:#?}", report.diagnostics);
    }

    #[test]
    fn federation_v011_independent_relationship_requires_two_sided_receipt() {
        let mut value: Value = serde_json::from_str(FEDERATION_V011_FIXTURE).expect("v0.1.1 federation JSON");
        let relationship = value["relationships"]
            .as_array_mut().expect("relationships")
            .iter_mut()
            .find(|item| item["relation"] == "shared_schema_candidate")
            .expect("shared schema candidate");
        relationship["status"] = Value::String("independently_qualified".to_owned());
        relationship["authority"] = Value::String("independently_adjudicated".to_owned());
        relationship["contract_binding"]["status"] =
            Value::String("independently_conformance_tested".to_owned());
        relationship["compatibility_receipt"] = Value::Null;
        let report = validate_federation_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "EGDDM-FEDERATION-037"));
    }

    #[test]
    fn federation_v011_receipt_must_match_contract_id_and_version() {
        let mut value: Value = serde_json::from_str(FEDERATION_V011_FIXTURE).expect("v0.1.1 federation JSON");
        let relationship = value["relationships"]
            .as_array_mut().expect("relationships")
            .iter_mut()
            .find(|item| item["relation"] == "shared_schema_candidate")
            .expect("shared schema candidate");
        relationship["status"] = Value::String("independently_qualified".to_owned());
        relationship["authority"] = Value::String("independently_adjudicated".to_owned());
        relationship["contract_binding"]["status"] =
            Value::String("independently_conformance_tested".to_owned());
        relationship["compatibility_receipt"] = serde_json::json!({
            "receipt_id":"synthetic:receipt",
            "receipt_sha256":"a".repeat(64),
            "verifier_build_sha256":"b".repeat(64),
            "source_project_id":"mycelix",
            "source_revision":"6089a68430a501c51934c2c4107061c2a2cb5565",
            "source_tree_oid":"79283620f83cdfa155fc86a2362335390628913d",
            "target_project_id":"symthaea",
            "target_revision":"dcffe1060ece4815cf42548a8f646cec4f075535",
            "target_tree_oid":"74972b6d6d1d3c67a8050cac76e46af49f8b0cd9",
            "contract_id":"WRONG-CONTRACT",
            "contract_version":"0.1.0",
            "contract_blob_oid":"f4abb87b24fc00d30e323bffd3a774b08d0740d4",
            "configuration_world_digest":"c".repeat(64),
            "outcome":"passed",
            "tested_at":"2026-11-14T12:00:00Z"
        });
        let report = validate_federation_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "EGDDM-FEDERATION-043"));

        let relationship = value["relationships"]
            .as_array_mut().expect("relationships")
            .iter_mut()
            .find(|item| item["relation"] == "shared_schema_candidate")
            .expect("shared schema candidate");
        relationship["compatibility_receipt"]["contract_id"] =
            Value::String("EGDDM-SNAPSHOT".to_owned());
        relationship["compatibility_receipt"]["contract_version"] =
            Value::String("9.9.9".to_owned());
        let report = validate_federation_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "EGDDM-FEDERATION-044"));
    }

    #[test]
    fn federation_v011_receipt_must_match_both_pinned_member_snapshots() {
        let mut value: Value = serde_json::from_str(FEDERATION_V011_FIXTURE).expect("v0.1.1 federation JSON");
        let relationship = value["relationships"]
            .as_array_mut().expect("relationships")
            .iter_mut()
            .find(|item| item["relation"] == "shared_schema_candidate")
            .expect("shared schema candidate");
        relationship["status"] = Value::String("independently_qualified".to_owned());
        relationship["authority"] = Value::String("independently_adjudicated".to_owned());
        relationship["contract_binding"]["status"] =
            Value::String("independently_conformance_tested".to_owned());
        relationship["compatibility_receipt"] = serde_json::json!({
            "receipt_id":"synthetic:receipt",
            "receipt_sha256":"a".repeat(64),
            "verifier_build_sha256":"b".repeat(64),
            "source_project_id":"mycelix",
            "source_revision":"6089a68430a501c51934c2c4107061c2a2cb5565",
            "source_tree_oid":"79283620f83cdfa155fc86a2362335390628913d",
            "target_project_id":"symthaea",
            "target_revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "target_tree_oid":"74972b6d6d1d3c67a8050cac76e46af49f8b0cd9",
            "contract_id":"EGDDM-SNAPSHOT",
            "contract_version":"0.1.0",
            "contract_blob_oid":"f4abb87b24fc00d30e323bffd3a774b08d0740d4",
            "configuration_world_digest":"c".repeat(64),
            "outcome":"passed",
            "tested_at":"2026-11-14T12:00:00Z"
        });
        let report = validate_federation_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "EGDDM-FEDERATION-034"));
    }

    #[test]
    fn federation_relationship_cannot_reference_unknown_project() {
        let mut value: Value = serde_json::from_str(FEDERATION_FIXTURE).expect("federation JSON");
        value["relationships"][0]["to_project_id"] = Value::String("missing-project".to_owned());
        let report = validate_federation_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "EGDDM-FEDERATION-010"));
    }

    #[test]
    fn federation_artifact_identity_must_be_a_git_oid() {
        let mut value: Value = serde_json::from_str(FEDERATION_FIXTURE).expect("federation JSON");
        value["members"][2]["map"]["artifact_commit_sha"] = Value::String("not-a-git-oid".to_owned());
        let report = validate_federation_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "EGDDM-DIGEST-006"));
    }

    #[test]
    fn federation_hub_must_match_exact_member_map() {
        let mut value: Value = serde_json::from_str(FEDERATION_FIXTURE).expect("federation JSON");
        value["hub"]["self_map_blob_oid"] = Value::String("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned());
        let report = validate_federation_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "EGDDM-FEDERATION-023"));
    }

    #[test]
    fn federation_map_url_must_pin_artifact_commit_and_path() {
        let mut value: Value = serde_json::from_str(FEDERATION_FIXTURE).expect("federation JSON");
        value["members"][1]["map"]["artifact_url"] =
            Value::String("https://github.com/Luminous-Dynamics/mycelix/blob/main/docs/development-map/source-declared-maturity-v0.1.json".to_owned());
        let report = validate_federation_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "EGDDM-FEDERATION-003"));
    }

    #[test]
    fn federation_hub_subject_snapshot_must_match_hub_map() {
        let mut value: Value = serde_json::from_str(FEDERATION_FIXTURE).expect("federation JSON");
        value["hub"]["subject_revision"] =
            Value::String("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned());
        let report = validate_federation_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "EGDDM-FEDERATION-026"));
    }

    #[test]
    fn federation_shared_schema_binding_must_match_current_contract_blob() {
        let mut value: Value = serde_json::from_str(FEDERATION_FIXTURE).expect("federation JSON");
        let relationship = value["relationships"]
            .as_array_mut().expect("relationships")
            .iter_mut()
            .find(|item| item["relation"] == "shared_schema_candidate")
            .expect("shared schema relationship");
        relationship["contract_binding"]["contract_blob_oid"] =
            Value::String("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned());
        let report = validate_federation_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "EGDDM-FEDERATION-038"));
    }

    #[test]
    fn federation_independent_qualification_requires_two_sided_receipt() {
        let mut value: Value = serde_json::from_str(FEDERATION_FIXTURE).expect("federation JSON");
        let relationship = value["relationships"]
            .as_array_mut().expect("relationships")
            .iter_mut()
            .find(|item| item["relation"] == "shared_schema_candidate")
            .expect("shared schema relationship");
        relationship["status"] = Value::String("independently_qualified".to_owned());
        relationship["authority"] = Value::String("independently_adjudicated".to_owned());
        relationship["contract_binding"]["status"] =
            Value::String("independently_conformance_tested".to_owned());
        let report = validate_federation_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "EGDDM-FEDERATION-037"));
    }

    #[test]
    fn federation_relationship_source_must_belong_to_an_endpoint() {
        let mut value: Value = serde_json::from_str(FEDERATION_FIXTURE).expect("federation JSON");
        value["relationships"][0]["source_refs"][0]["repository_full_name"] =
            Value::String("Luminous-Dynamics/mycelix".to_owned());
        let report = validate_federation_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "EGDDM-FEDERATION-039"));
    }

    #[test]
    fn federation_qualified_relationship_requires_independent_authority() {
        let mut value: Value = serde_json::from_str(FEDERATION_FIXTURE).expect("federation JSON");
        value["relationships"][0]["status"] = Value::String("independently_qualified".to_owned());
        let report = validate_federation_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "EGDDM-FEDERATION-011"));
    }


    const DYNRETR_CASE: &str = include_str!(
        "../../../docs/development-map/fixtures/retrieval-exploration-v0.1.json"
    );
    const DYNRETR_ORACLE: &str = include_str!(
        "../../../docs/development-map/fixtures/retrieval-exploration-v0.1.expected.json"
    );

    #[test]
    fn retrieval_exploration_fixture_satisfies_local_contract() {
        let report = validate_document_json(DYNRETR_CASE).expect("DYNRETR case JSON");
        assert!(report.is_valid(), "{:#?}", report.diagnostics);
    }

    #[test]
    fn retrieval_available_region_must_match_case_revision() {
        let mut value: Value = serde_json::from_str(DYNRETR_CASE).expect("DYNRETR case JSON");
        value["cases"][0]["candidate_regions"][0]["source_revision"] =
            Value::String("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned());
        let report = validate_retrieval_case_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "DYNRETR-SNAPSHOT-003"));
    }

    #[test]
    fn retrieval_stale_region_must_not_match_case_revision() {
        let mut value: Value = serde_json::from_str(DYNRETR_CASE).expect("DYNRETR case JSON");
        value["cases"][4]["candidate_regions"][1]["source_revision"] =
            value["cases"][4]["repository_revision"].clone();
        let report = validate_retrieval_case_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "DYNRETR-SNAPSHOT-004"));
    }

    #[test]
    fn retrieval_inactive_configuration_must_be_inactive_in_case_world() {
        let mut value: Value = serde_json::from_str(DYNRETR_CASE).expect("DYNRETR case JSON");
        value["cases"][3]["configuration_world"]["features"] =
            Value::Array(vec![Value::String("payments-v2".to_owned())]);
        let report = validate_retrieval_case_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "DYNRETR-WORLD-002"));
    }

    #[test]
    fn retrieval_exploration_oracle_satisfies_local_contract() {
        let report = validate_document_json(DYNRETR_ORACLE).expect("DYNRETR oracle JSON");
        assert!(report.is_valid(), "{:#?}", report.diagnostics);
    }

    #[test]
    fn retrieval_natural_no_gold_and_wrong_repository_stay_separate() {
        let case: Value = serde_json::from_str(DYNRETR_CASE).expect("DYNRETR case JSON");
        let cases = case["cases"].as_array().expect("cases");
        let natural = cases.iter().find(|item| item["case_id"] == "DYNRETR-SYN-002").expect("natural no-gold");
        let counterfactual = cases.iter().find(|item| item["case_id"] == "DYNRETR-SYN-003").expect("counterfactual no-gold");
        assert_eq!(natural["case_kind"], "natural_no_gold");
        assert_eq!(counterfactual["case_kind"], "wrong_repository_counterfactual");
        assert_eq!(natural["expected_disposition"], "abstain");
        assert_eq!(counterfactual["expected_disposition"], "abstain");
    }

    #[test]
    fn retrieval_configuration_and_temporal_decoys_are_explicit() {
        let case: Value = serde_json::from_str(DYNRETR_CASE).expect("DYNRETR case JSON");
        let cases = case["cases"].as_array().expect("cases");
        let config = cases.iter().find(|item| item["case_id"] == "DYNRETR-SYN-004").expect("config case");
        assert!(config["candidate_regions"].as_array().unwrap().iter().any(|r| r["availability"] == "inactive_configuration"));
        let temporal = cases.iter().find(|item| item["case_id"] == "DYNRETR-SYN-005").expect("temporal case");
        assert!(temporal["candidate_regions"].as_array().unwrap().iter().any(|r| r["availability"] == "stale_revision"));
    }

    #[test]
    fn retrieval_oracle_cannot_mark_a_region_both_relevant_and_forbidden() {
        let mut oracle: Value = serde_json::from_str(DYNRETR_ORACLE).expect("DYNRETR oracle JSON");
        oracle["expected_dispositions"][0]["forbidden_regions"] =
            Value::Array(vec![Value::String("r1".to_owned())]);
        let report = validate_retrieval_oracle_value(&oracle);
        assert!(report.diagnostics.iter().any(|d| d.code == "DYNRETR-ORACLE-004"));
    }

    const DYNPLAN_CASE: &str = include_str!(
        "../../../docs/development-map/fixtures/development-trajectory-planning-v0.1.json"
    );
    const DYNPLAN_ORACLE: &str = include_str!(
        "../../../docs/development-map/fixtures/development-trajectory-planning-v0.1.expected.json"
    );

    #[test]
    fn development_trajectory_fixture_has_closed_references_and_plan_costs() {
        let report = validate_document_json(DYNPLAN_CASE).expect("DYNPLAN case JSON");
        assert!(report.is_valid(), "{:#?}", report.diagnostics);
    }

    #[test]
    fn development_trajectory_oracle_references_every_candidate_plan() {
        let report = validate_document_json(DYNPLAN_ORACLE).expect("DYNPLAN oracle JSON");
        assert!(report.is_valid(), "{:#?}", report.diagnostics);
    }

    #[test]
    fn development_plan_unknown_action_reference_is_rejected() {
        let mut value: Value = serde_json::from_str(DYNPLAN_CASE).expect("DYNPLAN case JSON");
        value["candidate_plans"][0]["actions"] =
            Value::Array(vec![Value::String("missing:action".to_owned())]);
        let report = validate_development_plan_fixture_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "DYNPLAN-REF-002"));
    }

    #[test]
    fn development_plan_cost_must_match_action_expected_costs() {
        let mut value: Value = serde_json::from_str(DYNPLAN_CASE).expect("DYNPLAN case JSON");
        value["candidate_plans"][1]["declared_cost"] = Value::Number(serde_json::Number::from_f64(3.0).expect("finite"));
        let report = validate_development_plan_fixture_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "DYNPLAN-COST-001"));
    }

    #[test]
    fn development_plan_oracle_cannot_select_an_unknown_plan() {
        let mut value: Value = serde_json::from_str(DYNPLAN_ORACLE).expect("DYNPLAN oracle JSON");
        value["selected_plan"] = Value::String("missing:plan".to_owned());
        let report = validate_development_plan_oracle_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "DYNPLAN-ORACLE-003"));
    }

    const DTEA_FIXTURE: &str = include_str!(
        "../../../docs/development-map/fixtures/decision-time-admissibility-v0.1.map.json"
    );
    const DTEA_EXPECTED: &str = include_str!(
        "../../../docs/development-map/fixtures/decision-time-admissibility-v0.1.expected.json"
    );

    #[test]
    fn decision_time_fixture_preserves_late_arrival_and_later_correction() {
        let report = validate_json(DTEA_FIXTURE).expect("DTEA fixture JSON must parse");
        assert!(report.is_valid(), "{:#?}", report.diagnostics);
    }

    #[test]
    fn backdated_but_late_evidence_cannot_be_consulted() {
        let mut value: Value = serde_json::from_str(DTEA_FIXTURE).expect("DTEA fixture JSON");
        let late = value["events"][3]["payload"]["excluded_evidence"][0].clone();
        value["events"][3]["payload"]["consulted_evidence"] = Value::Array(vec![late]);
        value["events"][3]["payload"]["excluded_evidence"] = Value::Array(vec![]);
        let report = validate_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "DTEA-EVIDENCE-004"));
    }

    #[test]
    fn excluded_evidence_may_arrive_after_decision() {
        let report = validate_json(DTEA_FIXTURE).expect("DTEA fixture JSON");
        assert!(report.is_valid(), "{:#?}", report.diagnostics);
    }

    #[test]
    fn source_cannot_be_observed_after_actor_availability() {
        let mut value: Value = serde_json::from_str(DTEA_FIXTURE).expect("DTEA fixture JSON");
        value["events"][3]["payload"]["consulted_evidence"][0]["source_observed_at"] =
            Value::String("2026-11-14T09:10:00Z".to_owned());
        let report = validate_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "DTEA-EVIDENCE-006"));
    }

    #[test]
    fn decision_later_correction_must_resolve_and_follow_decision_time() {
        let mut value: Value = serde_json::from_str(DTEA_FIXTURE).expect("DTEA fixture JSON");
        value["events"][3]["payload"]["later_corrections"][0] = Value::String("missing:event".to_owned());
        let report = validate_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "DTEA-EVENT-004"));
    }

    #[test]
    fn correction_effective_before_decision_is_not_a_later_correction() {
        let mut value: Value = serde_json::from_str(DTEA_FIXTURE).expect("DTEA fixture JSON");
        value["events"][5]["occurred_at"] = Value::String("2026-11-14T09:59:00Z".to_owned());
        let report = validate_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "DTEA-EVENT-005"));
    }

    #[test]
    fn knowledge_cutoff_cannot_be_after_decision() {
        let mut value: Value = serde_json::from_str(DTEA_FIXTURE).expect("DTEA fixture JSON");
        value["events"][3]["payload"]["knowledge_cutoff_at"] =
            Value::String("2026-11-14T10:01:00Z".to_owned());
        let report = validate_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "DTEA-TIME-001"));
    }

    #[test]
    fn decision_authority_actor_must_match() {
        let mut value: Value = serde_json::from_str(DTEA_FIXTURE).expect("DTEA fixture JSON");
        value["events"][3]["payload"]["actor_authority"]["actor_id"] =
            Value::String("another-actor".to_owned());
        let report = validate_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "DTEA-AUTH-006"));
    }

    #[test]
    fn decision_authority_scope_must_match() {
        let mut value: Value = serde_json::from_str(DTEA_FIXTURE).expect("DTEA fixture JSON");
        value["events"][3]["payload"]["actor_authority"]["authority_scope"] =
            Value::String("release:production".to_owned());
        let report = validate_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "DTEA-AUTH-007"));
    }

    #[test]
    fn authority_evidence_must_be_available_by_decision_cutoff() {
        let mut value: Value = serde_json::from_str(DTEA_FIXTURE).expect("DTEA fixture JSON");
        value["events"][3]["payload"]["actor_authority"]["available_to_actor_at"] =
            Value::String("2026-11-14T10:15:00Z".to_owned());
        let report = validate_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "DTEA-AUTH-002"));
    }

    #[test]
    fn expired_authority_cannot_support_a_decision() {
        let mut value: Value = serde_json::from_str(DTEA_FIXTURE).expect("DTEA fixture JSON");
        value["events"][3]["payload"]["actor_authority"]["valid_through"] =
            Value::String("2026-11-14T09:59:00Z".to_owned());
        value["nodes"][5]["attributes"]["valid_through"] =
            Value::String("2026-11-14T09:59:00Z".to_owned());
        let report = validate_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "DTEA-AUTH-004"));
    }

    #[test]
    fn authority_evidence_reference_must_resolve_to_evidence_node() {
        let mut value: Value = serde_json::from_str(DTEA_FIXTURE).expect("DTEA fixture JSON");
        value["events"][3]["payload"]["actor_authority"]["evidence_id"] =
            Value::String("missing:authority".to_owned());
        let report = validate_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "DTEA-AUTH-014"));
    }

    #[test]
    fn decision_policy_must_match_owner_authority_record() {
        let mut value: Value = serde_json::from_str(DTEA_FIXTURE).expect("DTEA fixture JSON");
        value["events"][3]["payload"]["actor_authority"]["policy_version"] =
            Value::String("release-policy-v2".to_owned());
        let report = validate_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "DTEA-AUTH-008"));
    }

    #[test]
    fn authority_recorded_event_must_match_owner_evidence_node() {
        let mut value: Value = serde_json::from_str(DTEA_FIXTURE).expect("DTEA fixture JSON");
        value["events"][1]["payload"]["owner_record_ref"] =
            Value::String("synthetic://mycelix/authority-grant/OTHER".to_owned());
        let report = validate_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "DTEA-AUTH-018"));
    }

    #[test]
    fn decision_cannot_reference_authority_without_recorded_event() {
        let mut value: Value = serde_json::from_str(DTEA_FIXTURE).expect("DTEA fixture JSON");
        value["events"][1]["kind"] = Value::String("source_changed".to_owned());
        let report = validate_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "DTEA-AUTH-018"));
    }

    #[test]
    fn consulted_evidence_cannot_become_effective_after_the_decision() {
        let mut value: Value = serde_json::from_str(DTEA_FIXTURE).expect("DTEA fixture JSON");
        value["events"][3]["payload"]["consulted_evidence"][0]["valid_from"] =
            Value::String("2026-11-14T10:01:00Z".to_owned());
        let report = validate_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "DTEA-EVIDENCE-010"));
    }

    #[test]
    fn consulted_evidence_cannot_expire_before_the_decision() {
        let mut value: Value = serde_json::from_str(DTEA_FIXTURE).expect("DTEA fixture JSON");
        value["events"][3]["payload"]["consulted_evidence"][0]["valid_through"] =
            Value::String("2026-11-14T09:59:00Z".to_owned());
        let report = validate_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "DTEA-EVIDENCE-011"));
    }

    #[test]
    fn consulted_evidence_rejects_reversed_validity_intervals() {
        let mut value: Value = serde_json::from_str(DTEA_FIXTURE).expect("DTEA fixture JSON");
        value["events"][3]["payload"]["consulted_evidence"][0]["valid_from"] =
            Value::String("2026-11-14T11:00:00Z".to_owned());
        value["events"][3]["payload"]["consulted_evidence"][0]["valid_through"] =
            Value::String("2026-11-14T09:00:00Z".to_owned());
        let report = validate_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "DTEA-EVIDENCE-009"));
    }

    #[test]
    fn evidence_validity_bounds_are_inclusive_at_the_decision_time() {
        let mut value: Value = serde_json::from_str(DTEA_FIXTURE).expect("DTEA fixture JSON");
        value["events"][3]["payload"]["consulted_evidence"][0]["valid_from"] =
            Value::String("2026-11-14T10:00:00Z".to_owned());
        value["events"][3]["payload"]["consulted_evidence"][0]["valid_through"] =
            Value::String("2026-11-14T10:00:00Z".to_owned());
        let report = validate_value(&value);
        assert!(!report.diagnostics.iter().any(|d| matches!(
            d.code,
            "DTEA-EVIDENCE-009" | "DTEA-EVIDENCE-010" | "DTEA-EVIDENCE-011"
        )));
    }

    #[test]
    fn decision_time_manifest_survives_current_evidence_invalidation() {
        let value: Value = serde_json::from_str(DTEA_FIXTURE).expect("DTEA fixture JSON");
        assert_eq!(value["nodes"][2]["lifecycle"]["state"], "stale");
        assert_eq!(
            value["events"][3]["payload"]["consulted_evidence"][0]["evidence_id"],
            "dtea:evidence:E1"
        );
        let oracle: Value = serde_json::from_str(DTEA_EXPECTED).expect("DTEA expected JSON");
        assert_eq!(oracle["expected_decision"]["decision_time_state_of_E1"], "admissible");
        assert_eq!(oracle["expected_decision"]["current_state_of_E1"], "stale");
    }


    #[test]
    fn containment_cycles_are_rejected() {
        let mut value: Value = serde_json::from_str(TEMPORAL_FIXTURE).expect("fixture JSON");
        let provenance = value["edges"][0]["provenance"].clone();
        let edges = value["edges"].as_array_mut().expect("edges");
        edges.push(serde_json::json!({
            "id":"edge-cycle",
            "from":"fixture:capability",
            "relation":"contains",
            "to":"fixture:project",
            "provenance":provenance,
            "lifecycle":{"state":"current","supersedes":null,"invalidated_by":null,"reason":null}
        }));
        let report = validate_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "EGDDM-GRAPH-001"));
    }
}
