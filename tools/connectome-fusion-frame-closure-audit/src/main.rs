#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::process;

const SCHEMA_VERSION: &str = "0.1.0";
const EXECUTION_ROLE: &str = "FUSION_EXECUTION_LEDGER";
const CLAIM_ROLE: &str = "CLAIM_CERTIFICATE_LEDGER";

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema_version: String,
    audit_id: String,
    scope: Scope,
    scope_sha256: String,
    fusion_policy_projection: Projection,
    source_exports: Vec<SourceExport>,
    declared_frame: Frame,
    reconciliation_policy: ReconciliationPolicy,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Scope {
    scope_id: String,
    start_at: String,
    end_at: String,
    event_time_basis: String,
    query_ids: Vec<String>,
    release_ids: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Projection {
    projection_id: String,
    projection_sha256: String,
    rule: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SourceExport {
    source_id: String,
    source_role: String,
    custodian_ref: String,
    scope_sha256: String,
    extraction_sha256: String,
    sequence_start: u64,
    sequence_end: u64,
    record_count: u64,
    unique_use_count: u64,
    ordered_events_sha256: String,
    coverage_disposition: String,
    shared_controls: Vec<String>,
    events: Vec<Event>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
struct Event {
    event_id: String,
    sequence: u64,
    event_at: String,
    ingested_at: String,
    canonical_use_key: String,
    operation_id: String,
    claim_id: String,
    query_id: String,
    certificate_id: String,
    fusion_policy_sha256: String,
    verifier_sha256: String,
    dependence_bundle_sha256: String,
    source_release_ids: Vec<String>,
    evidence_ids: Vec<String>,
    outcome: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Frame {
    frame_sha256: String,
    assertion_count: u64,
    assertions: Vec<Assertion>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
struct Assertion {
    assertion_id: String,
    canonical_use_key: String,
    claim_id: String,
    query_id: String,
    left_evidence_id: String,
    right_evidence_id: String,
    fusion_policy_sha256: String,
    dependence_bundle_sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ReconciliationPolicy {
    required_source_roles: Vec<String>,
    shared_control_policy: String,
}

#[derive(Debug, Serialize)]
struct Report {
    report_version: &'static str,
    audit_id: String,
    disposition: String,
    promotion_authorized: bool,
    input_sha256: String,
    scope_sha256: String,
    projection_sha256: String,
    source_checks: Vec<SourceCheck>,
    source_population_match: bool,
    required_assertion_count: usize,
    frame_assertion_count: usize,
    missing_assertions: Vec<AssertionKey>,
    orphan_assertions: Vec<AssertionKey>,
    execution_without_claim: Vec<String>,
    claim_without_execution: Vec<String>,
    errors: Vec<String>,
    limitations: Vec<String>,
}

#[derive(Debug, Serialize)]
struct SourceCheck {
    source_id: String,
    source_role: String,
    record_count_recomputed: usize,
    unique_use_count_recomputed: usize,
    sequence_contiguous: bool,
    event_digest_valid: bool,
    extraction_digest_valid: bool,
    scope_match: bool,
    coverage_qualified: bool,
    custodian_unique: bool,
    shared_controls_empty: bool,
}

#[derive(Debug, Serialize, Clone, Eq, PartialEq, Ord, PartialOrd)]
struct AssertionKey {
    canonical_use_key: String,
    claim_id: String,
    query_id: String,
    left_evidence_id: String,
    right_evidence_id: String,
    fusion_policy_sha256: String,
    dependence_bundle_sha256: String,
}

#[derive(Debug, Serialize)]
struct UseKeyMaterial {
    scope_sha256: String,
    query_id: String,
    claim_id: String,
    certificate_id: String,
    operation_id: String,
    fusion_policy_sha256: String,
    verifier_sha256: String,
    dependence_bundle_sha256: String,
    source_release_ids: Vec<String>,
    evidence_ids: Vec<String>,
}

#[derive(Debug, Eq, PartialEq)]
struct EventSemantic {
    operation_id: String,
    claim_id: String,
    query_id: String,
    certificate_id: String,
    fusion_policy_sha256: String,
    verifier_sha256: String,
    dependence_bundle_sha256: String,
    source_release_ids: Vec<String>,
    evidence_ids: Vec<String>,
}

fn canonical_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, String> {
    // serde_json::Value uses lexicographically ordered object keys by default.
    let value = serde_json::to_value(value).map_err(|e| format!("canonicalize: {e}"))?;
    serde_json::to_vec(&value).map_err(|e| format!("serialize canonical JSON: {e}"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn domain_hash<T: Serialize>(domain: &str, value: &T) -> Result<String, String> {
    let canonical = canonical_bytes(value)?;
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update([0_u8]);
    hasher.update(canonical);
    Ok(hex::encode(hasher.finalize()))
}

fn domain_hash_string(domain: &str, value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update([0_u8]);
    hasher.update(value.as_bytes());
    hex::encode(hasher.finalize())
}

fn sorted_unique(values: &[String]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn use_key(input: &Input, event: &Event) -> Result<String, String> {
    let mut releases = event.source_release_ids.clone();
    releases.sort();
    let mut evidence = event.evidence_ids.clone();
    evidence.sort();
    let material = UseKeyMaterial {
        scope_sha256: input.scope_sha256.clone(),
        query_id: event.query_id.clone(),
        claim_id: event.claim_id.clone(),
        certificate_id: event.certificate_id.clone(),
        operation_id: event.operation_id.clone(),
        fusion_policy_sha256: event.fusion_policy_sha256.clone(),
        verifier_sha256: event.verifier_sha256.clone(),
        dependence_bundle_sha256: event.dependence_bundle_sha256.clone(),
        source_release_ids: releases,
        evidence_ids: evidence,
    };
    domain_hash("connectome-fusion-use-key-v0.1", &material)
}

fn semantic(event: &Event) -> EventSemantic {
    let mut source_release_ids = event.source_release_ids.clone();
    source_release_ids.sort();
    let mut evidence_ids = event.evidence_ids.clone();
    evidence_ids.sort();
    EventSemantic {
        operation_id: event.operation_id.clone(),
        claim_id: event.claim_id.clone(),
        query_id: event.query_id.clone(),
        certificate_id: event.certificate_id.clone(),
        fusion_policy_sha256: event.fusion_policy_sha256.clone(),
        verifier_sha256: event.verifier_sha256.clone(),
        dependence_bundle_sha256: event.dependence_bundle_sha256.clone(),
        source_release_ids,
        evidence_ids,
    }
}

fn assertion_keys(event: &Event) -> Vec<AssertionKey> {
    let mut ids = event.evidence_ids.clone();
    ids.sort();
    let mut result = Vec::new();
    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            result.push(AssertionKey {
                canonical_use_key: event.canonical_use_key.clone(),
                claim_id: event.claim_id.clone(),
                query_id: event.query_id.clone(),
                left_evidence_id: ids[i].clone(),
                right_evidence_id: ids[j].clone(),
                fusion_policy_sha256: event.fusion_policy_sha256.clone(),
                dependence_bundle_sha256: event.dependence_bundle_sha256.clone(),
            });
        }
    }
    result
}

fn frame_key(assertion: &Assertion) -> AssertionKey {
    AssertionKey {
        canonical_use_key: assertion.canonical_use_key.clone(),
        claim_id: assertion.claim_id.clone(),
        query_id: assertion.query_id.clone(),
        left_evidence_id: assertion.left_evidence_id.clone(),
        right_evidence_id: assertion.right_evidence_id.clone(),
        fusion_policy_sha256: assertion.fusion_policy_sha256.clone(),
        dependence_bundle_sha256: assertion.dependence_bundle_sha256.clone(),
    }
}

fn verify(input: &Input, raw: &[u8]) -> Report {
    let mut errors = Vec::new();
    if input.schema_version != SCHEMA_VERSION {
        errors.push("UNSUPPORTED_SCHEMA_VERSION".to_string());
    }
    if input.audit_id.trim().is_empty() || input.scope.scope_id.trim().is_empty() {
        errors.push("MISSING_STABLE_AUDIT_OR_SCOPE_ID".to_string());
    }
    if input.scope.start_at >= input.scope.end_at || !input.scope.start_at.ends_with("Z") || !input.scope.end_at.ends_with("Z") {
        errors.push("INVALID_SCOPE_INTERVAL".to_string());
    }
    if !matches!(input.scope.event_time_basis.as_str(), "EVENT_TIME" | "INGESTION_TIME") {
        errors.push("UNSUPPORTED_EVENT_TIME_BASIS".to_string());
    }
    if input.scope.query_ids.is_empty() || !sorted_unique(&input.scope.query_ids) {
        errors.push("QUERY_SCOPE_IDS_EMPTY_DUPLICATED_OR_UNSORTED".to_string());
    }
    if input.scope.release_ids.is_empty() || !sorted_unique(&input.scope.release_ids) {
        errors.push("RELEASE_SCOPE_IDS_EMPTY_DUPLICATED_OR_UNSORTED".to_string());
    }
    if input.reconciliation_policy.shared_control_policy != "FAIL_CLOSED_ON_DECLARED_SHARED_CONTROL" {
        errors.push("UNSUPPORTED_SHARED_CONTROL_POLICY".to_string());
    }

    let computed_scope = domain_hash("connectome-fusion-scope-v0.1", &input.scope);
    if !computed_scope.as_ref().is_ok_and(|digest| digest == &input.scope_sha256) {
        errors.push("SCOPE_DIGEST_MISMATCH".to_string());
    }

    let projection_material = ProjectionMaterial {
        projection_id: input.fusion_policy_projection.projection_id.clone(),
        rule: input.fusion_policy_projection.rule.clone(),
    };
    let computed_projection = domain_hash("connectome-fusion-projection-v0.1", &projection_material);
    if !computed_projection.as_ref().is_ok_and(|digest| digest == &input.fusion_policy_projection.projection_sha256) {
        errors.push("PROJECTION_DIGEST_MISMATCH".to_string());
    }
    if input.fusion_policy_projection.rule != "ALL_UNORDERED_EVIDENCE_PAIRS" {
        errors.push("UNSUPPORTED_FUSION_POLICY_PROJECTION".to_string());
    }

    let required_roles: BTreeSet<&str> = input.reconciliation_policy.required_source_roles.iter().map(String::as_str).collect();
    let expected_roles: BTreeSet<&str> = [EXECUTION_ROLE, CLAIM_ROLE].into_iter().collect();
    if required_roles != expected_roles || input.reconciliation_policy.required_source_roles.len() != 2 {
        errors.push("REQUIRED_SOURCE_ROLE_SET_UNSUPPORTED".to_string());
    }

    let mut source_checks = Vec::new();
    let mut source_ids = BTreeSet::new();
    let mut custodian_refs = BTreeSet::new();
    let mut shared_custodian_detected = false;
    let mut role_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut exports_by_role: BTreeMap<String, &SourceExport> = BTreeMap::new();

    for source in &input.source_exports {
        if !source_ids.insert(source.source_id.as_str()) {
            errors.push(format!("DUPLICATE_SOURCE_ID:{}", source.source_id));
        }
        if source.source_role != EXECUTION_ROLE && source.source_role != CLAIM_ROLE {
            errors.push(format!("UNSUPPORTED_SOURCE_ROLE:{}", source.source_role));
        }
        *role_counts.entry(source.source_role.clone()).or_default() += 1;
        if exports_by_role.insert(source.source_role.clone(), source).is_some() {
            errors.push(format!("DUPLICATE_SOURCE_ROLE:{}", source.source_role));
        }

        let computed_events = domain_hash("connectome-fusion-events-v0.1", &source.events);
        let computed_extraction = domain_hash_string("connectome-fusion-extraction-v0.1", &source.source_id);
        let event_ids: BTreeSet<&str> = source.events.iter().map(|e| e.event_id.as_str()).collect();
        let seqs: BTreeSet<u64> = source.events.iter().map(|e| e.sequence).collect();
        let unique_use_keys: BTreeSet<&str> = source.events.iter().map(|e| e.canonical_use_key.as_str()).collect();
        let sequence_contiguous = source.events.len() as u64 == source.record_count
            && seqs.len() == source.events.len()
            && source.events.iter().enumerate().all(|(i, e)| source.sequence_start.checked_add(i as u64) == Some(e.sequence))
            && if source.events.is_empty() {
                source.sequence_end == source.sequence_start
            } else {
                source.events.last().map(|e| e.sequence) == Some(source.sequence_end)
                    && source.sequence_end.saturating_sub(source.sequence_start).saturating_add(1) == source.record_count
            };
        let event_digest_valid = computed_events.as_ref().is_ok_and(|digest| digest == &source.ordered_events_sha256);
        let extraction_digest_valid = computed_extraction == source.extraction_sha256;
        let scope_match = source.scope_sha256 == input.scope_sha256;
        let custodian_unique = custodian_refs.insert(source.custodian_ref.as_str());
        if !custodian_unique {
            shared_custodian_detected = true;
        }

        if source.source_id.trim().is_empty() || source.custodian_ref.trim().is_empty() {
            errors.push(format!("SOURCE_ID_OR_CUSTODIAN_MISSING:{}", source.source_id));
        }
        if event_ids.len() != source.events.len() {
            errors.push(format!("DUPLICATE_EVENT_ID:{}", source.source_id));
        }
        if source.record_count != source.events.len() as u64 {
            errors.push(format!("RECORD_COUNT_MISMATCH:{}", source.source_id));
        }
        if source.unique_use_count != unique_use_keys.len() as u64 {
            errors.push(format!("UNIQUE_USE_COUNT_MISMATCH:{}", source.source_id));
        }
        if !sequence_contiguous {
            errors.push(format!("SEQUENCE_GAP_OR_ORDER_ERROR:{}", source.source_id));
        }
        if !event_digest_valid {
            errors.push(format!("EVENT_EXPORT_DIGEST_MISMATCH:{}", source.source_id));
        }
        if !extraction_digest_valid {
            errors.push(format!("EXTRACTION_DIGEST_MISMATCH:{}", source.source_id));
        }
        if !scope_match {
            errors.push(format!("SOURCE_SCOPE_DIGEST_MISMATCH:{}", source.source_id));
        }
        for event in &source.events {
            if event.event_id.trim().is_empty()
                || event.operation_id.trim().is_empty()
                || event.claim_id.trim().is_empty()
                || event.query_id.trim().is_empty()
                || event.certificate_id.trim().is_empty()
                || event.evidence_ids.len() < 2
                || event.source_release_ids.is_empty()
                || !event.event_at.ends_with("Z")
                || !event.ingested_at.ends_with("Z")
                || event.ingested_at < event.event_at
                || !sorted_unique(&event.source_release_ids)
                || !sorted_unique(&event.evidence_ids)
            {
                errors.push(format!("INVALID_EVENT_IDENTITY_OR_EVIDENCE_SET:{}:{}", source.source_id, event.event_id));
            }
            let scoped_time = if input.scope.event_time_basis == "EVENT_TIME" {
                &event.event_at
            } else {
                &event.ingested_at
            };
            if scoped_time < &input.scope.start_at || scoped_time >= &input.scope.end_at
                || !input.scope.query_ids.contains(&event.query_id)
                || event.source_release_ids.iter().any(|r| !input.scope.release_ids.contains(r))
            {
                errors.push(format!("EVENT_OUTSIDE_DECLARED_SCOPE:{}:{}", source.source_id, event.event_id));
            }
            if !matches!(event.outcome.as_str(), "CONTRIBUTED" | "FAILED_BEFORE_CLAIM" | "CANCELLED" | "RETRY_OF_PRIOR_EVENT") {
                errors.push(format!("UNSUPPORTED_EVENT_OUTCOME:{}:{}", source.source_id, event.event_id));
            }
            match use_key(input, event) {
                Ok(expected) if expected == event.canonical_use_key => {}
                Ok(_) => errors.push(format!("CANONICAL_USE_KEY_MISMATCH:{}:{}", source.source_id, event.event_id)),
                Err(e) => errors.push(format!("USE_KEY_CANONICALIZATION_FAILED:{}:{e}", source.source_id)),
            }
        }

        source_checks.push(SourceCheck {
            source_id: source.source_id.clone(),
            source_role: source.source_role.clone(),
            record_count_recomputed: source.events.len(),
            unique_use_count_recomputed: unique_use_keys.len(),
            sequence_contiguous,
            event_digest_valid,
            extraction_digest_valid,
            scope_match,
            coverage_qualified: source.coverage_disposition == "QUALIFIED_COMPLETE_WITHIN_DECLARED_SCOPE",
            custodian_unique,
            shared_controls_empty: source.shared_controls.is_empty(),
        });
    }

    for role in [EXECUTION_ROLE, CLAIM_ROLE] {
        if role_counts.get(role).copied().unwrap_or(0) != 1 {
            errors.push(format!("REQUIRED_SOURCE_ROLE_MISSING_OR_DUPLICATED:{role}"));
        }
    }

    if input.declared_frame.assertion_count != input.declared_frame.assertions.len() as u64 {
        errors.push("FRAME_ASSERTION_COUNT_MISMATCH".to_string());
    }
    let computed_frame = domain_hash("connectome-fusion-frame-v0.1", &input.declared_frame.assertions);
    if !computed_frame.as_ref().is_ok_and(|digest| digest == &input.declared_frame.frame_sha256) {
        errors.push("FRAME_DIGEST_MISMATCH".to_string());
    }

    let mut frame_ids = BTreeSet::new();
    let mut frame_keys = BTreeSet::new();
    for assertion in &input.declared_frame.assertions {
        if assertion.assertion_id.trim().is_empty() || !frame_ids.insert(assertion.assertion_id.as_str()) {
            errors.push("FRAME_ASSERTION_ID_EMPTY_OR_DUPLICATED".to_string());
        }
        if assertion.left_evidence_id >= assertion.right_evidence_id {
            errors.push(format!("FRAME_PAIR_NOT_CANONICALLY_ORDERED:{}", assertion.assertion_id));
        }
        if !frame_keys.insert(frame_key(assertion)) {
            errors.push(format!("DUPLICATE_FRAME_ASSERTION:{}", assertion.assertion_id));
        }
    }

    if !errors.is_empty() {
        return Report {
            report_version: SCHEMA_VERSION,
            audit_id: input.audit_id.clone(),
            disposition: "NOT_ESTIMABLE".to_string(),
            promotion_authorized: false,
            input_sha256: sha256_hex(raw),
            scope_sha256: input.scope_sha256.clone(),
            projection_sha256: input.fusion_policy_projection.projection_sha256.clone(),
            source_checks,
            source_population_match: false,
            required_assertion_count: 0,
            frame_assertion_count: input.declared_frame.assertions.len(),
            missing_assertions: Vec::new(),
            orphan_assertions: Vec::new(),
            execution_without_claim: Vec::new(),
            claim_without_execution: Vec::new(),
            errors,
            limitations: vec![
                "Structural and digest checks qualify only the supplied records.".to_string(),
                "This report cannot establish that an event source recorded every real-world evidence use.".to_string(),
                "This report does not establish biological independence or biological truth.".to_string(),
            ],
        };
    }

    let mut coverage_unknown = false;
    let mut shared_control = shared_custodian_detected;
    for source in &input.source_exports {
        if source.coverage_disposition != "QUALIFIED_COMPLETE_WITHIN_DECLARED_SCOPE" {
            coverage_unknown = true;
        }
        if !source.shared_controls.is_empty() {
            shared_control = true;
        }
    }

    let execution = exports_by_role.get(EXECUTION_ROLE).copied();
    let claim = exports_by_role.get(CLAIM_ROLE).copied();
    let mut execution_map: BTreeMap<String, EventSemantic> = BTreeMap::new();
    let mut claim_map: BTreeMap<String, EventSemantic> = BTreeMap::new();

    if let Some(source) = execution {
        for event in &source.events {
            if event.outcome == "CONTRIBUTED" {
                if execution_map.insert(event.canonical_use_key.clone(), semantic(event)).is_some() {
                    errors.push(format!("DUPLICATE_CONTRIBUTING_USE_IN_EXECUTION:{}", event.canonical_use_key));
                }
            }
        }
    }
    if let Some(source) = claim {
        for event in &source.events {
            if event.outcome == "CONTRIBUTED" {
                if claim_map.insert(event.canonical_use_key.clone(), semantic(event)).is_some() {
                    errors.push(format!("DUPLICATE_CONTRIBUTING_USE_IN_CLAIM_LEDGER:{}", event.canonical_use_key));
                }
            } else {
                errors.push(format!("NON_CONTRIBUTING_EVENT_IN_CLAIM_LEDGER:{}", event.event_id));
            }
        }
    }

    let execution_keys: BTreeSet<String> = execution_map.keys().cloned().collect();
    let claim_keys: BTreeSet<String> = claim_map.keys().cloned().collect();
    let execution_without_claim: Vec<String> = execution_keys.difference(&claim_keys).cloned().collect();
    let claim_without_execution: Vec<String> = claim_keys.difference(&execution_keys).cloned().collect();
    let source_population_match = execution_keys == claim_keys
        && execution_map.iter().all(|(key, value)| claim_map.get(key) == Some(value));
    if !source_population_match {
        errors.push("SOURCE_POPULATION_RECONCILIATION_MISMATCH".to_string());
    }

    let mut required_keys = BTreeSet::new();
    if let Some(source) = execution {
        for event in &source.events {
            if event.outcome == "CONTRIBUTED" {
                for key in assertion_keys(event) {
                    required_keys.insert(key);
                }
            }
        }
    }
    let missing_assertions: Vec<AssertionKey> = required_keys.difference(&frame_keys).cloned().collect();
    let orphan_assertions: Vec<AssertionKey> = frame_keys.difference(&required_keys).cloned().collect();

    let disposition = if !errors.is_empty() {
        "RECONCILIATION_CONFLICT"
    } else if coverage_unknown || shared_control {
        "SOURCE_COVERAGE_UNKNOWN"
    } else if !missing_assertions.is_empty() {
        "FRAME_GAP_DETECTED"
    } else if !orphan_assertions.is_empty() {
        "RECONCILIATION_CONFLICT"
    } else if required_keys.is_empty() {
        "NOT_ESTIMABLE"
    } else {
        "FRAME_CLOSED_WITHIN_DECLARED_SCOPE"
    };

    let closed = disposition == "FRAME_CLOSED_WITHIN_DECLARED_SCOPE";
    let mut limitations = vec![
        "Closure is bounded to the named event exports, custodians, query/release scope, and event-time convention.".to_string(),
        "Agreement between two exports does not prove that every real-world operation was logged.".to_string(),
        "Shared or undisclosed common controls may invalidate apparent source separation.".to_string(),
        "A closed frame is not proof of biological independence, biological truth, or causal identification.".to_string(),
    ];
    if shared_control {
        limitations.push("At least one source declares shared control; promotion is blocked.".to_string());
    }

    Report {
        report_version: SCHEMA_VERSION,
        audit_id: input.audit_id.clone(),
        disposition: disposition.to_string(),
        promotion_authorized: closed,
        input_sha256: sha256_hex(raw),
        scope_sha256: input.scope_sha256.clone(),
        projection_sha256: input.fusion_policy_projection.projection_sha256.clone(),
        source_checks,
        source_population_match,
        required_assertion_count: required_keys.len(),
        frame_assertion_count: frame_keys.len(),
        missing_assertions,
        orphan_assertions,
        execution_without_claim,
        claim_without_execution,
        errors,
        limitations,
    }
}

#[derive(Serialize)]
struct ProjectionMaterial {
    projection_id: String,
    rule: String,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: connectome-fusion-frame-closure-audit <input.json>");
        process::exit(2);
    }
    let path = &args[1];
    let raw = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("cannot read {path}: {error}");
            process::exit(2);
        }
    };
    let input: Input = match serde_json::from_slice(&raw) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("invalid input JSON: {error}");
            process::exit(2);
        }
    };
    let report = verify(&input, &raw);
    match serde_json::to_string_pretty(&report) {
        Ok(json) => println!("{json}"),
        Err(error) => {
            eprintln!("cannot serialize report: {error}");
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../../../research/connectomics/fixtures/connectome-fusion-frame-closure-omitted-use-v0.1.json");

    fn parse_fixture() -> (Input, Vec<u8>) {
        let raw = FIXTURE.as_bytes().to_vec();
        let input: Input = serde_json::from_slice(&raw).expect("fixture decodes");
        (input, raw)
    }

    #[test]
    fn omitted_use_is_reported_as_frame_gap() {
        let (input, raw) = parse_fixture();
        let report = verify(&input, &raw);
        assert_eq!(report.disposition, "FRAME_GAP_DETECTED");
        assert!(!report.promotion_authorized);
        assert_eq!(report.required_assertion_count, 2);
        assert_eq!(report.frame_assertion_count, 1);
        assert_eq!(report.missing_assertions.len(), 1);
        assert_eq!(report.missing_assertions[0].left_evidence_id, "evidence-B");
        assert_eq!(report.missing_assertions[0].right_evidence_id, "evidence-C");
        assert!(report.orphan_assertions.is_empty());
        assert!(report.errors.is_empty());
    }

    #[test]
    fn unknown_source_coverage_never_promotes() {
        let (mut input, raw) = parse_fixture();
        input.source_exports[0].coverage_disposition = "UNKNOWN".to_string();
        let report = verify(&input, &raw);
        assert_eq!(report.disposition, "SOURCE_COVERAGE_UNKNOWN");
        assert!(!report.promotion_authorized);
    }

    #[test]
    fn shared_source_control_never_promotes() {
        let (mut input, raw) = parse_fixture();
        input.source_exports[0].shared_controls.push("shared-event-bus".to_string());
        let report = verify(&input, &raw);
        assert_eq!(report.disposition, "SOURCE_COVERAGE_UNKNOWN");
        assert!(!report.promotion_authorized);
    }

    #[test]
    fn tampered_frame_digest_is_not_estimable() {
        let (mut input, raw) = parse_fixture();
        input.declared_frame.frame_sha256 = "0".repeat(64);
        let report = verify(&input, &raw);
        assert_eq!(report.disposition, "NOT_ESTIMABLE");
        assert!(report.errors.iter().any(|e| e == "FRAME_DIGEST_MISMATCH"));
    }

    #[test]
    fn unknown_fields_are_rejected_by_strict_decoder() {
        let raw = FIXTURE.replace("\"audit_id\":", "\"unrecognized\": \"x\",\n  \"audit_id\":");
        let parsed = serde_json::from_str::<Input>(&raw);
        assert!(parsed.is_err());
    }
}
