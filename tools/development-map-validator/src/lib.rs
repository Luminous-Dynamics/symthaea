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
    "capability_link_changed", "decision_recorded", "development_action_proposed",
    "development_action_observed", "risk_opened", "risk_closed",
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

        for key in ["valid_from", "valid_through"] {
            if let Some(value) = entry.get(key) {
                if let Some(value) = optional_string(value, &format!("{entry_path}.{key}"), report) {
                    if DateTime::parse_from_rfc3339(&value).is_err() {
                        report.error("EGDDM-TIME-001", format!("{entry_path}.{key}"), "timestamp must be valid RFC 3339");
                    }
                }
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
            "excluded_evidence", "later_corrections", "abstention_reason",
        ],
        path,
        report,
    );

    let decision_at = field(payload, "decision_at", path, report)
        .and_then(|v| parse_rfc3339(v, &format!("{path}.decision_at"), report));
    let cutoff = field(payload, "knowledge_cutoff_at", path, report)
        .and_then(|v| parse_rfc3339(v, &format!("{path}.knowledge_cutoff_at"), report));

    for key in ["actor_id", "authority_scope", "policy_version", "outcome"] {
        if let Some(value) = field(payload, key, path, report) {
            string(value, &format!("{path}.{key}"), report);
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
    if let Some(observed_at) = field(value, "observed_at", &path, report) {
        string(observed_at, &format!("{path}.observed_at"), report);
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
    if let Some(event_values) = value.get("events") {
        if let Some(event_values) = array(event_values, "$.events", &mut report) {
            for (index, event) in event_values.iter().enumerate() {
                validate_event(event, index, &mut report, &mut event_ids, &nodes, &mut invalidation_targets);
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

fn validate_federation_relationship(
    value: &Value,
    index: usize,
    report: &mut ValidationReport,
    members: &HashMap<String, (String, String, String)>,
) {
    let path = format!("relationships[{index}]");
    let Some(relationship) = object(value, &path, report) else {
        return;
    };
    check_allowed_keys(
        relationship,
        &[
            "from_project_id", "relation", "to_project_id", "status",
            "authority", "source_refs", "contract_binding", "claim_limitations",
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
            "shared_contract", "members", "relationships", "join_policy",
            "limitations",
        ],
        "$",
        &mut report,
    );
    if field(value, "schema_version", "$", &mut report).and_then(Value::as_str) != Some("0.1.0") {
        report.error("EGDDM-FEDERATION-020", "$.schema_version", "only federation schema_version 0.1.0 is supported");
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
            validate_federation_relationship(relationship, index, &mut report, &members);
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

/// Autodetect and validate an EGDDM snapshot or federation directory.
pub fn validate_document_json(input: &str) -> Result<ValidationReport, serde_json::Error> {
    let value: Value = serde_json::from_str(input)?;
    if value.get("federation_id").is_some() {
        Ok(validate_federation_value(&value))
    } else {
        Ok(validate_value(&value))
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
    fn federation_qualified_relationship_requires_independent_authority() {
        let mut value: Value = serde_json::from_str(FEDERATION_FIXTURE).expect("federation JSON");
        value["relationships"][0]["status"] = Value::String("independently_qualified".to_owned());
        let report = validate_federation_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "EGDDM-FEDERATION-011"));
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
        let late = value["events"][2]["payload"]["excluded_evidence"][0].clone();
        value["events"][2]["payload"]["consulted_evidence"] = Value::Array(vec![late]);
        value["events"][2]["payload"]["excluded_evidence"] = Value::Array(vec![]);
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
        value["events"][2]["payload"]["consulted_evidence"][0]["source_observed_at"] =
            Value::String("2026-11-14T09:10:00Z".to_owned());
        let report = validate_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "DTEA-EVIDENCE-006"));
    }

    #[test]
    fn decision_later_correction_must_resolve_and_follow_decision_time() {
        let mut value: Value = serde_json::from_str(DTEA_FIXTURE).expect("DTEA fixture JSON");
        value["events"][2]["payload"]["later_corrections"][0] = Value::String("missing:event".to_owned());
        let report = validate_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "DTEA-EVENT-004"));
    }

    #[test]
    fn correction_effective_before_decision_is_not_a_later_correction() {
        let mut value: Value = serde_json::from_str(DTEA_FIXTURE).expect("DTEA fixture JSON");
        value["events"][4]["occurred_at"] = Value::String("2026-11-14T09:59:00Z".to_owned());
        let report = validate_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "DTEA-EVENT-005"));
    }

    #[test]
    fn knowledge_cutoff_cannot_be_after_decision() {
        let mut value: Value = serde_json::from_str(DTEA_FIXTURE).expect("DTEA fixture JSON");
        value["events"][2]["payload"]["knowledge_cutoff_at"] =
            Value::String("2026-11-14T10:01:00Z".to_owned());
        let report = validate_value(&value);
        assert!(report.diagnostics.iter().any(|d| d.code == "DTEA-TIME-001"));
    }

    #[test]
    fn decision_time_manifest_survives_current_evidence_invalidation() {
        let value: Value = serde_json::from_str(DTEA_FIXTURE).expect("DTEA fixture JSON");
        assert_eq!(value["nodes"][2]["lifecycle"]["state"], "stale");
        assert_eq!(
            value["events"][2]["payload"]["consulted_evidence"][0]["evidence_id"],
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
