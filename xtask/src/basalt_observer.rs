//! Structural auditor for the BASALT observer-challenge catalog.
//!
//! This validates the challenge specification, not an observer, coverage receipt,
//! or benchmark result. Qualification remains a separate execution-and-oracle task.

use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;

const MAX_CATALOG_BYTES: u64 = 16 * 1024 * 1024;
const MAX_CASES: usize = 10_000;
const MAX_TEXT_BYTES: usize = 8_192;
const MAX_FACTS_PER_CASE: usize = 64;
const MAX_PAIRS_PER_CASE: usize = 64;

const FAMILIES: &[&str] = &[
    "filesystem", "process", "environment", "generated_artifact", "toolchain",
    "cache", "network", "observer_integrity", "snapshot_consistency", "positive_control",
];
const CHANNELS: &[&str] = &[
    "filesystem", "process", "environment", "generated_artifact",
    "toolchain", "cache", "network", "observer_integrity",
];
const MUTATION_KINDS: &[&str] = &[
    "read_content", "read_metadata", "directory_membership", "symlink_target",
    "mmap_bytes", "rename_delete_create", "untracked_read", "root_escape",
    "generated_source", "build_script_input", "proc_macro_input", "env_value",
    "child_env", "inherited_fd", "invocation_metadata", "dynamic_library",
    "nondeterministic_source", "pseudo_filesystem_read", "toolchain_closure",
    "dependency_resolution", "cache_hit", "network_dns", "network_artifact",
    "registry_artifact", "event_overflow", "observer_restart", "unmonitored_child",
    "concurrent_hash_mutation", "irrelevant_mutation", "unsupported_channel",
];
const DISPOSITIONS: &[&str] = &[
    "capture_dependency", "report_unresolved", "invalidate_receipt", "preserve_unrelated",
];
const RECEIPT_FACTS: &[&str] = &[
    "resource_identity", "content_digest", "read_event", "metadata_digest",
    "metadata_read_event", "directory_identity", "sorted_membership_digest",
    "enumeration_event", "link_identity", "resolved_target_identity", "target_digest",
    "mapping_observation", "path_identity", "namespace_change", "tree_digest",
    "untracked_classification", "unresolved_channel", "access_path_classification",
    "generator_identity", "generator_input_digest", "generated_artifact_digest",
    "process_lineage", "argv_cwd", "toolchain_digest", "macro_host_identity",
    "environment_binding", "value_commitment", "inherited_fd_provenance",
    "invocation_identity", "executable_identity", "loaded_object_digest",
    "virtualization_status", "mount_identity", "closure_identity", "target_configuration",
    "lockfile_digest", "config_digest", "feature_resolution", "cache_entry_identity",
    "artifact_digest", "cache_producer_provenance", "network_policy",
    "resolution_observation", "endpoint_identity", "protocol", "response_artifact_digest",
    "drop_counter", "overflow_status", "receipt_integrity", "observer_lifecycle",
    "capture_interval", "supervision_status", "stable_handle_identity",
    "snapshot_consistency", "root_boundary", "mutation_identity",
    "non_dependency_witness", "coverage_scope", "registry_identity",
];

/// The result of checking catalog structure and internal cross-references.
/// This report is deliberately never confirmatory-eligible.
#[derive(Debug, Serialize)]
pub struct CatalogAuditReport {
    pub report_version: u32,
    pub input_digest: String,
    pub structurally_valid: bool,
    pub qualification_eligible: bool,
    pub profile_id: Option<String>,
    pub profile_status: Option<String>,
    pub execution_status: Option<String>,
    pub counts: CatalogAuditCounts,
    pub family_counts: BTreeMap<String, usize>,
    pub channel_counts: BTreeMap<String, usize>,
    pub disposition_counts: BTreeMap<String, usize>,
    pub cases: Vec<CaseAssessment>,
    pub errors: Vec<CatalogAuditError>,
}

#[derive(Debug, Default, Serialize)]
pub struct CatalogAuditCounts {
    pub cases_total: usize,
    pub cases_capture_dependency: usize,
    pub cases_report_unresolved: usize,
    pub cases_invalidate_receipt: usize,
    pub cases_preserve_unrelated: usize,
    pub cases_with_pairs: usize,
    pub pair_links: usize,
}

#[derive(Debug, Serialize)]
pub struct CaseAssessment {
    pub case_id: String,
    pub expected_disposition: String,
    pub paired_case_ids: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct CatalogAuditError {
    pub code: String,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Catalog {
    format_version: u32,
    suite_id: String,
    profile_id: String,
    profile_status: String,
    execution_status: String,
    scoped_claim: String,
    oracle_policy: OraclePolicy,
    expected_dispositions: Vec<String>,
    cases: Vec<ChallengeCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OraclePolicy {
    ground_truth_source: String,
    observer_under_test_must_not_author_the_oracle: bool,
    independent_replay_required_for_confirmatory_use: bool,
    score_unit: String,
    repeated_events_are_not_independent_samples: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChallengeCase {
    case_id: String,
    family: String,
    channel: String,
    mutation_kind: String,
    #[serde(default)]
    ground_truth_resource_id: RequiredNullable<String>,
    expected_disposition: String,
    required_receipt_facts: Vec<String>,
    paired_case_ids: Vec<String>,
    description: String,
    safety_property: String,
}

/// Like Option<T>, but also records whether the JSON property was present.
/// The schema requires this property even when its value is null.
#[derive(Debug)]
struct RequiredNullable<T> {
    present: bool,
    value: Option<T>,
}

impl<T> Default for RequiredNullable<T> {
    fn default() -> Self {
        Self { present: false, value: None }
    }
}

impl<'de, T> Deserialize<'de> for RequiredNullable<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<T>::deserialize(deserializer)
            .map(|value| Self { present: true, value })
    }
}

pub fn audit_file(path: &Path) -> anyhow::Result<CatalogAuditReport> {
    let file = File::open(path)?;
    let mut bytes = Vec::new();
    file.take(MAX_CATALOG_BYTES + 1).read_to_end(&mut bytes)?;
    Ok(audit_bytes(&bytes))
}

pub fn audit_bytes(bytes: &[u8]) -> CatalogAuditReport {
    let input_digest = format!("sha256:{}", sha256_hex(bytes));
    let mut report = CatalogAuditReport {
        report_version: 1,
        input_digest,
        structurally_valid: false,
        qualification_eligible: false,
        profile_id: None,
        profile_status: None,
        execution_status: None,
        counts: CatalogAuditCounts::default(),
        family_counts: BTreeMap::new(),
        channel_counts: BTreeMap::new(),
        disposition_counts: BTreeMap::new(),
        cases: Vec::new(),
        errors: Vec::new(),
    };

    if bytes.len() as u64 > MAX_CATALOG_BYTES {
        push_error(&mut report, "E_CATALOG_TOO_LARGE", "$", "observer challenge catalog exceeds the 16 MiB input limit");
        return report;
    }

    let catalog: Catalog = match serde_json::from_slice(bytes) {
        Ok(value) => value,
        Err(error) => {
            push_error(&mut report, "E_CATALOG_JSON", "$", &format!("invalid catalog JSON, duplicate field, unknown field, or missing field: {error}"));
            return report;
        }
    };

    report.profile_id = Some(catalog.profile_id.clone());
    report.profile_status = Some(catalog.profile_status.clone());
    report.execution_status = Some(catalog.execution_status.clone());

    validate_catalog(&catalog, &mut report);
    report.structurally_valid = report.errors.is_empty();
    report
}

fn validate_catalog(catalog: &Catalog, report: &mut CatalogAuditReport) {
    if catalog.format_version != 1 {
        push_error(report, "E_FORMAT_VERSION", "format_version", "only format_version=1 is supported");
    }
    if catalog.suite_id != "basalt-observer-challenges-v1" {
        push_error(report, "E_SUITE_ID", "suite_id", "suite_id must be basalt-observer-challenges-v1");
    }
    if !valid_identifier(&catalog.profile_id) {
        push_error(report, "E_PROFILE_ID", "profile_id", "profile_id has invalid identifier syntax");
    }
    if catalog.profile_status != "candidate_not_qualified" {
        push_error(report, "E_PROFILE_STATUS", "profile_status", "catalog fixtures must remain candidate_not_qualified; execution results belong in a separate report");
    }
    if catalog.execution_status != "not_executed" {
        push_error(report, "E_EXECUTION_STATUS", "execution_status", "a challenge catalog cannot assert execution; execution_status must be not_executed");
    }
    check_text(report, "scoped_claim", &catalog.scoped_claim);

    let oracle = &catalog.oracle_policy;
    if oracle.ground_truth_source != "challenge_harness_injected_resources_and_independently_checked_mutation_pairs" {
        push_error(report, "E_ORACLE_SOURCE", "oracle_policy.ground_truth_source", "ground truth must be injected and independently checked");
    }
    if !oracle.observer_under_test_must_not_author_the_oracle {
        push_error(report, "E_ORACLE_INDEPENDENCE", "oracle_policy.observer_under_test_must_not_author_the_oracle", "the observer under test must not author its own oracle");
    }
    if !oracle.independent_replay_required_for_confirmatory_use {
        push_error(report, "E_REPLAY_REQUIRED", "oracle_policy.independent_replay_required_for_confirmatory_use", "independent replay must be required");
    }
    if oracle.score_unit != "independent_task_scenario" {
        push_error(report, "E_SCORE_UNIT", "oracle_policy.score_unit", "score unit must be independent_task_scenario");
    }
    if !oracle.repeated_events_are_not_independent_samples {
        push_error(report, "E_PSEUDOREPLICATION", "oracle_policy.repeated_events_are_not_independent_samples", "repeated events within one task must not be counted as independent samples");
    }

    if catalog.expected_dispositions.len() != DISPOSITIONS.len() {
        push_error(report, "E_DISPOSITION_SET", "expected_dispositions", "expected_dispositions must contain each of the four supported dispositions exactly once");
    }
    let expected: BTreeSet<&str> = catalog.expected_dispositions.iter().map(String::as_str).collect();
    let supported: BTreeSet<&str> = DISPOSITIONS.iter().copied().collect();
    if expected.len() != catalog.expected_dispositions.len() || expected != supported {
        push_error(report, "E_DISPOSITION_SET", "expected_dispositions", "expected_dispositions does not exactly match the supported disposition set");
    }

    if catalog.cases.is_empty() {
        push_error(report, "E_CASES_EMPTY", "cases", "at least one challenge case is required");
    }
    if catalog.cases.len() > MAX_CASES {
        push_error(report, "E_CASES_LIMIT", "cases", "challenge case count exceeds 10000");
        return;
    }

    report.counts.cases_total = catalog.cases.len();
    let mut case_indices = BTreeMap::<&str, usize>::new();

    for (index, case) in catalog.cases.iter().enumerate() {
        let path = format!("cases[{index}]");
        validate_case(case, &path, report);

        if case_indices.insert(case.case_id.as_str(), index).is_some() {
            push_error(report, "E_CASE_ID_DUPLICATE", &format!("{path}.case_id"), "case IDs must be unique");
        }

        *report.family_counts.entry(case.family.clone()).or_insert(0) += 1;
        *report.channel_counts.entry(case.channel.clone()).or_insert(0) += 1;
        *report.disposition_counts.entry(case.expected_disposition.clone()).or_insert(0) += 1;

        match case.expected_disposition.as_str() {
            "capture_dependency" => report.counts.cases_capture_dependency += 1,
            "report_unresolved" => report.counts.cases_report_unresolved += 1,
            "invalidate_receipt" => report.counts.cases_invalidate_receipt += 1,
            "preserve_unrelated" => report.counts.cases_preserve_unrelated += 1,
            _ => {}
        }

        if !case.paired_case_ids.is_empty() {
            report.counts.cases_with_pairs += 1;
            report.counts.pair_links += case.paired_case_ids.len();
        }

        report.cases.push(CaseAssessment {
            case_id: case.case_id.clone(),
            expected_disposition: case.expected_disposition.clone(),
            paired_case_ids: case.paired_case_ids.clone(),
        });
    }

    for (index, case) in catalog.cases.iter().enumerate() {
        let path = format!("cases[{index}]");
        validate_pairs(case, &path, catalog, &case_indices, report);
    }
}

fn validate_case(case: &ChallengeCase, path: &str, report: &mut CatalogAuditReport) {
    if !valid_case_id(&case.case_id) {
        push_error(report, "E_CASE_ID", &format!("{path}.case_id"), "case_id must match ^[a-z0-9][a-z0-9-]{1,95}$");
    }
    if !FAMILIES.contains(&case.family.as_str()) {
        push_error(report, "E_CASE_FAMILY", &format!("{path}.family"), "unsupported challenge family");
    }
    if !CHANNELS.contains(&case.channel.as_str()) {
        push_error(report, "E_CASE_CHANNEL", &format!("{path}.channel"), "unsupported observation channel");
    }
    if !MUTATION_KINDS.contains(&case.mutation_kind.as_str()) {
        push_error(report, "E_MUTATION_KIND", &format!("{path}.mutation_kind"), "unsupported mutation kind");
    }
    if !DISPOSITIONS.contains(&case.expected_disposition.as_str()) {
        push_error(report, "E_EXPECTED_DISPOSITION", &format!("{path}.expected_disposition"), "unsupported expected disposition");
    }

    if !case.ground_truth_resource_id.present {
        push_error(report, "E_GROUND_TRUTH_REQUIRED", &format!("{path}.ground_truth_resource_id"), "ground_truth_resource_id is required, even when null");
    } else if let Some(resource_id) = &case.ground_truth_resource_id.value {
        if !valid_identifier(resource_id) {
            push_error(report, "E_GROUND_TRUTH_ID", &format!("{path}.ground_truth_resource_id"), "ground-truth resource ID has invalid identifier syntax");
        }
    }

    if case.required_receipt_facts.is_empty() || case.required_receipt_facts.len() > MAX_FACTS_PER_CASE {
        push_error(report, "E_RECEIPT_FACTS_LIMIT", &format!("{path}.required_receipt_facts"), "required_receipt_facts must contain 1..=64 entries");
    }
    let mut facts = BTreeSet::new();
    for (fact_index, fact) in case.required_receipt_facts.iter().enumerate() {
        if !RECEIPT_FACTS.contains(&fact.as_str()) {
            push_error(report, "E_RECEIPT_FACT", &format!("{path}.required_receipt_facts[{fact_index}]"), "unsupported required receipt fact");
        }
        if !facts.insert(fact) {
            push_error(report, "E_RECEIPT_FACT_DUPLICATE", &format!("{path}.required_receipt_facts[{fact_index}]"), "required receipt facts must be unique");
        }
    }

    if case.paired_case_ids.len() > MAX_PAIRS_PER_CASE {
        push_error(report, "E_PAIRS_LIMIT", &format!("{path}.paired_case_ids"), "a case may name at most 64 paired cases");
    }
    let mut pairs = BTreeSet::new();
    for (pair_index, pair_id) in case.paired_case_ids.iter().enumerate() {
        if !valid_case_id(pair_id) {
            push_error(report, "E_PAIR_ID", &format!("{path}.paired_case_ids[{pair_index}]"), "paired case ID has invalid syntax");
        }
        if pair_id == &case.case_id {
            push_error(report, "E_PAIR_SELF", &format!("{path}.paired_case_ids[{pair_index}]"), "a case cannot pair with itself");
        }
        if !pairs.insert(pair_id) {
            push_error(report, "E_PAIR_DUPLICATE", &format!("{path}.paired_case_ids[{pair_index}]"), "paired case IDs must be unique");
        }
    }

    check_text(report, &format!("{path}.description"), &case.description);
    check_text(report, &format!("{path}.safety_property"), &case.safety_property);

    match case.expected_disposition.as_str() {
        "capture_dependency" | "preserve_unrelated" => {
            if case.ground_truth_resource_id.value.is_none() {
                push_error(report, "E_DISPOSITION_RESOURCE_REQUIRED", &format!("{path}.ground_truth_resource_id"), "capture_dependency and preserve_unrelated require a concrete ground-truth resource ID");
            }
        }
        _ => {}
    }

    if case.expected_disposition == "preserve_unrelated" && case.paired_case_ids.len() != 1 {
        push_error(report, "E_PRESERVE_PAIR_REQUIRED", &format!("{path}.paired_case_ids"), "preserve_unrelated requires exactly one paired capture_dependency case");
    }
}

fn validate_pairs(
    case: &ChallengeCase,
    path: &str,
    catalog: &Catalog,
    case_indices: &BTreeMap<&str, usize>,
    report: &mut CatalogAuditReport,
) {
    for (pair_index, pair_id) in case.paired_case_ids.iter().enumerate() {
        let pair_path = format!("{path}.paired_case_ids[{pair_index}]");
        let Some(other_index) = case_indices.get(pair_id.as_str()).copied() else {
            push_error(report, "E_PAIR_UNKNOWN", &pair_path, "paired case ID does not exist in this catalog");
            continue;
        };
        let other = &catalog.cases[other_index];
        if !other.paired_case_ids.iter().any(|candidate| candidate == &case.case_id) {
            push_error(report, "E_PAIR_NOT_RECIPROCAL", &pair_path, "paired case relationship must be reciprocal");
        }
    }

    if case.expected_disposition == "preserve_unrelated" {
        if let Some(pair_id) = case.paired_case_ids.first() {
            if let Some(other_index) = case_indices.get(pair_id.as_str()).copied() {
                let other = &catalog.cases[other_index];
                if other.expected_disposition != "capture_dependency" {
                    push_error(report, "E_PRESERVE_PAIR_DISPOSITION", path, "preserve_unrelated must pair with capture_dependency");
                }
                if other.ground_truth_resource_id.value != case.ground_truth_resource_id.value {
                    push_error(report, "E_PRESERVE_PAIR_RESOURCE", path, "preserve_unrelated must pair against the same injected resource as its capture_dependency control");
                }
            }
        }
    }
}

fn check_text(report: &mut CatalogAuditReport, path: &str, value: &str) {
    if value.len() > MAX_TEXT_BYTES || value.trim().is_empty() {
        push_error(report, "E_TEXT_BOUNDS", path, "text must be non-empty and no longer than 8192 UTF-8 bytes");
    }
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.trim() == value
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':' | b'/')
        })
}

fn valid_case_id(value: &str) -> bool {
    let bytes = value.as_bytes();
    (2..=96).contains(&bytes.len())
        && (bytes[0].is_ascii_lowercase() || bytes[0].is_ascii_digit())
        && bytes[1..].iter().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'-'
        })
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn push_error(report: &mut CatalogAuditReport, code: &str, path: &str, message: &str) {
    report.errors.push(CatalogAuditError {
        code: code.to_string(),
        path: path.to_string(),
        message: message.to_string(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn example_value() -> Value {
        serde_json::from_slice(include_bytes!(
            "../../benchmarks/basalt/coverage/observer-challenges.v1.json"
        ))
        .expect("checked-in observer challenge example must be valid JSON")
    }

    fn audit_value(value: &Value) -> CatalogAuditReport {
        audit_bytes(&serde_json::to_vec(value).expect("fixture serializes"))
    }

    fn has_error(report: &CatalogAuditReport, code: &str) -> bool {
        report.errors.iter().any(|error| error.code == code)
    }

    #[test]
    fn checked_in_catalog_is_structurally_valid_but_not_qualified() {
        let report = audit_bytes(include_bytes!(
            "../../benchmarks/basalt/coverage/observer-challenges.v1.json"
        ));
        assert!(report.structurally_valid, "{:?}", report.errors);
        assert!(!report.qualification_eligible);
        assert_eq!(report.counts.cases_total, 33);
        assert_eq!(report.counts.cases_capture_dependency, 19);
        assert_eq!(report.counts.cases_report_unresolved, 7);
        assert_eq!(report.counts.cases_invalidate_receipt, 5);
        assert_eq!(report.counts.cases_preserve_unrelated, 2);
        assert_eq!(report.counts.cases_with_pairs, 2);
        assert_eq!(report.counts.pair_links, 2);
    }

    #[test]
    fn duplicate_case_ids_are_rejected() {
        let mut value = example_value();
        let first = value["cases"][0]["case_id"].clone();
        value["cases"][1]["case_id"] = first;
        let report = audit_value(&value);
        assert!(!report.structurally_valid);
        assert!(has_error(&report, "E_CASE_ID_DUPLICATE"));
    }

    #[test]
    fn missing_nullable_ground_truth_property_is_rejected() {
        let mut value = example_value();
        value["cases"][0].as_object_mut().unwrap().remove("ground_truth_resource_id");
        let report = audit_value(&value);
        assert!(!report.structurally_valid);
        assert!(has_error(&report, "E_GROUND_TRUTH_REQUIRED"));
    }

    #[test]
    fn unknown_catalog_fields_are_rejected() {
        let mut value = example_value();
        value["observer_claim"] = Value::String("complete".into());
        let report = audit_value(&value);
        assert!(!report.structurally_valid);
        assert!(has_error(&report, "E_CATALOG_JSON"));
    }

    #[test]
    fn unknown_receipt_fact_is_rejected() {
        let mut value = example_value();
        value["cases"][0]["required_receipt_facts"].as_array_mut().unwrap()
            .push(Value::String("trust_me".into()));
        let report = audit_value(&value);
        assert!(!report.structurally_valid);
        assert!(has_error(&report, "E_RECEIPT_FACT"));
    }

    #[test]
    fn unresolved_disposition_may_have_no_resource_identity() {
        let mut value = example_value();
        let case = value["cases"].as_array_mut().unwrap().iter_mut()
            .find(|item| item["expected_disposition"] == "report_unresolved")
            .unwrap();
        case["ground_truth_resource_id"] = Value::Null;
        let report = audit_value(&value);
        assert!(report.structurally_valid, "{:?}", report.errors);
    }

    #[test]
    fn preserve_unrelated_requires_same_resource_capture_pair() {
        let mut value = example_value();
        let case = value["cases"].as_array_mut().unwrap().iter_mut()
            .find(|item| item["expected_disposition"] == "preserve_unrelated")
            .unwrap();
        case["ground_truth_resource_id"] = Value::String("fixture:wrong-resource".into());
        let report = audit_value(&value);
        assert!(!report.structurally_valid);
        assert!(has_error(&report, "E_PRESERVE_PAIR_RESOURCE"));
    }

    #[test]
    fn pair_relationships_must_be_reciprocal() {
        let mut value = example_value();
        value["cases"].as_array_mut().unwrap().iter_mut()
            .find(|item| item["case_id"] == "unused-neighbor-change")
            .unwrap()["paired_case_ids"] = serde_json::json!([]);
        let report = audit_value(&value);
        assert!(!report.structurally_valid);
        assert!(has_error(&report, "E_PRESERVE_PAIR_REQUIRED"));
        assert!(has_error(&report, "E_PAIR_NOT_RECIPROCAL"));
    }

    #[test]
    fn candidate_catalog_cannot_claim_execution_or_qualification() {
        let mut value = example_value();
        value["profile_status"] = Value::String("qualified".into());
        value["execution_status"] = Value::String("passed".into());
        let report = audit_value(&value);
        assert!(!report.structurally_valid);
        assert!(has_error(&report, "E_PROFILE_STATUS"));
        assert!(has_error(&report, "E_EXECUTION_STATUS"));
        assert!(!report.qualification_eligible);
    }

    #[test]
    fn input_limit_is_fail_closed() {
        let bytes = vec![b' '; (MAX_CATALOG_BYTES as usize) + 1];
        let report = audit_bytes(&bytes);
        assert!(!report.structurally_valid);
        assert!(has_error(&report, "E_CATALOG_TOO_LARGE"));
    }
}
