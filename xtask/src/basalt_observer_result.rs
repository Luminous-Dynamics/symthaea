//! Independent adjudicator for BASALT observer challenge executions.
//!
//! The catalog, oracle ledger, and observer receipts are three separate inputs.
//! Observer receipts are untrusted. Digests bind the exact files, but this tool
//! does not authenticate the oracle author or qualify an observation profile.
//! A successful adjudication is a conformance result for one run, never a
//! confirmatory-eligibility decision.

use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::Read;
use std::path::Path;

const MAX_INPUT_BYTES: u64 = 16 * 1024 * 1024;
const MAX_CASES: usize = 10_000;
const MAX_RECORDS: usize = 10_000;
const MAX_FACTS: usize = 128;
const MAX_OBSERVED_RESOURCES: usize = 4_096;
const MAX_UNRESOLVED_CHANNELS: usize = 64;

#[derive(Debug, Serialize)]
pub struct ResultAuditReport {
    pub report_version: u32,
    pub catalog_digest: String,
    pub oracle_digest: String,
    pub observer_receipts_digest: String,
    pub run_id: Option<String>,
    pub adjudication_complete: bool,
    pub adjudication_passed: bool,
    /// Permanently false: this command adjudicates one run, not a profile.
    pub qualification_eligible: bool,
    /// Hashes identify bytes; this tool does not authenticate who authored them.
    pub oracle_authenticity_established: bool,
    /// Replay is asserted by the supplied oracle ledger, not executed by this auditor.
    pub oracle_replay_executed_by_adjudicator: bool,
    /// Raw observer receipts are parsed as untrusted claims; their signer is not verified.
    pub observer_receipt_authenticity_established: bool,
    pub counts: ResultAuditCounts,
    pub metrics: ResultAuditMetrics,
    pub cases: Vec<CaseAssessment>,
    pub errors: Vec<ResultAuditError>,
}

#[derive(Debug, Default, Serialize)]
pub struct ResultAuditCounts {
    pub challenge_cases: usize,
    pub oracle_records: usize,
    pub observer_receipts: usize,
    pub adjudicated_cases: usize,
    pub passed_cases: usize,
    pub failed_cases: usize,
}

#[derive(Debug, Default, Serialize)]
pub struct ResultAuditMetrics {
    pub hidden_dependency_misses: usize,
    pub stale_evidence_false_acceptances: usize,
    pub false_invalidations: usize,
    pub unrelated_evidence_not_retained: usize,
    pub unresolved_channel_misses: usize,
    pub integrity_failures_not_invalidated: usize,
    pub observed_resource_attributions: usize,
}

#[derive(Debug, Serialize)]
pub struct CaseAssessment {
    pub case_id: String,
    pub expected_disposition: String,
    pub oracle_ground_truth: Option<String>,
    pub observer_disposition: Option<String>,
    pub reuse_decision: Option<String>,
    pub passed: bool,
    pub error_codes: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ResultAuditError {
    pub code: String,
    pub case_id: Option<String>,
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

#[derive(Debug)]
struct RequiredNullable<T> {
    present: bool,
    value: Option<T>,
}

impl<T> Default for RequiredNullable<T> {
    fn default() -> Self {
        Self {
            present: false,
            value: None,
        }
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
        Option::<T>::deserialize(deserializer).map(|value| Self {
            present: true,
            value,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OracleLedger {
    format_version: u32,
    suite_id: String,
    run_id: String,
    catalog_digest: String,
    profile_id: String,
    harness: HarnessIdentity,
    records: Vec<OracleRecord>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HarnessIdentity {
    harness_id: String,
    binary_digest: String,
    fixture_digest: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OracleRecord {
    case_id: String,
    ground_truth: String,
    #[serde(default)]
    injected_resource_id: RequiredNullable<String>,
    mutation: MutationWitness,
    witness_kind: String,
    independent_replay: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MutationWitness {
    applied: bool,
    #[serde(default)]
    before_digest: RequiredNullable<String>,
    #[serde(default)]
    after_digest: RequiredNullable<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ObserverReceipts {
    format_version: u32,
    suite_id: String,
    run_id: String,
    catalog_digest: String,
    observer_profile_id: String,
    observer_binary_digest: String,
    records: Vec<ObserverReceipt>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ObserverReceipt {
    case_id: String,
    disposition: String,
    observed_resources: Vec<ObservedResource>,
    facts: Vec<String>,
    unresolved_channels: Vec<String>,
    reuse_decision: String,
    integrity: IntegritySignals,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservedResource {
    resource_id: String,
    digest: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct IntegritySignals {
    event_loss: bool,
    observer_restart: bool,
    unmonitored_process: bool,
    snapshot_race: bool,
    root_escape: bool,
}

pub fn audit_files(
    catalog_path: &Path,
    oracle_path: &Path,
    receipts_path: &Path,
) -> anyhow::Result<ResultAuditReport> {
    let catalog_bytes = read_bounded(catalog_path)?;
    let oracle_bytes = read_bounded(oracle_path)?;
    let receipts_bytes = read_bounded(receipts_path)?;
    Ok(audit_bytes(
        &catalog_bytes,
        &oracle_bytes,
        &receipts_bytes,
    ))
}

pub fn audit_bytes(
    catalog_bytes: &[u8],
    oracle_bytes: &[u8],
    receipts_bytes: &[u8],
) -> ResultAuditReport {
    let mut report = ResultAuditReport {
        report_version: 1,
        catalog_digest: sha256_tagged(catalog_bytes),
        oracle_digest: sha256_tagged(oracle_bytes),
        observer_receipts_digest: sha256_tagged(receipts_bytes),
        run_id: None,
        adjudication_complete: false,
        adjudication_passed: false,
        qualification_eligible: false,
        oracle_authenticity_established: false,
        oracle_replay_executed_by_adjudicator: false,
        observer_receipt_authenticity_established: false,
        counts: ResultAuditCounts::default(),
        metrics: ResultAuditMetrics::default(),
        cases: Vec::new(),
        errors: Vec::new(),
    };

    if catalog_bytes.len() as u64 > MAX_INPUT_BYTES
        || oracle_bytes.len() as u64 > MAX_INPUT_BYTES
        || receipts_bytes.len() as u64 > MAX_INPUT_BYTES
    {
        push_error(
            &mut report,
            None,
            "E_INPUT_TOO_LARGE",
            "$",
            "each BASALT adjudication input is limited to 16 MiB",
        );
        return report;
    }

    let catalog: Catalog = match serde_json::from_slice(catalog_bytes) {
        Ok(value) => value,
        Err(error) => {
            push_error(
                &mut report,
                None,
                "E_CATALOG_JSON",
                "catalog",
                &format!("invalid or non-strict challenge catalog: {error}"),
            );
            return report;
        }
    };
    let catalog_audit = crate::basalt_observer::audit_bytes(catalog_bytes);
    if !catalog_audit.structurally_valid {
        for error in &catalog_audit.errors {
            push_error(
                &mut report,
                None,
                &format!("E_CATALOG_{}", error.code),
                &error.path,
                &error.message,
            );
        }
        return report;
    }

    let oracle: OracleLedger = match serde_json::from_slice(oracle_bytes) {
        Ok(value) => value,
        Err(error) => {
            push_error(
                &mut report,
                None,
                "E_ORACLE_JSON",
                "oracle",
                &format!("invalid or non-strict independent oracle ledger: {error}"),
            );
            return report;
        }
    };
    let receipts: ObserverReceipts = match serde_json::from_slice(receipts_bytes) {
        Ok(value) => value,
        Err(error) => {
            push_error(
                &mut report,
                None,
                "E_RECEIPTS_JSON",
                "observer_receipts",
                &format!("invalid or non-strict observer receipt ledger: {error}"),
            );
            return report;
        }
    };

    validate_envelopes(
        &catalog,
        &oracle,
        &receipts,
        &mut report,
    );

    report.run_id = Some(oracle.run_id.clone());
    report.counts.challenge_cases = catalog.cases.len();
    report.counts.oracle_records = oracle.records.len();
    report.counts.observer_receipts = receipts.records.len();

    if catalog.cases.is_empty() || catalog.cases.len() > MAX_CASES {
        push_error(
            &mut report,
            None,
            "E_CASE_COUNT",
            "catalog.cases",
            "challenge catalog must contain 1..=10000 cases",
        );
        return report;
    }
    if oracle.records.len() > MAX_RECORDS || receipts.records.len() > MAX_RECORDS {
        push_error(
            &mut report,
            None,
            "E_RECORD_COUNT",
            "$",
            "oracle and receipt ledgers are limited to 10000 records each",
        );
        return report;
    }

    let mut catalog_ids = BTreeSet::new();
    for case in &catalog.cases {
        if !catalog_ids.insert(case.case_id.as_str()) {
            push_error(
                &mut report,
                Some(&case.case_id),
                "E_CATALOG_CASE_DUPLICATE",
                "catalog.cases",
                "catalog case IDs must be unique",
            );
        }
    }

    let mut oracle_by_id = BTreeMap::<&str, &OracleRecord>::new();
    for record in &oracle.records {
        if oracle_by_id.insert(&record.case_id, record).is_some() {
            push_error(
                &mut report,
                Some(&record.case_id),
                "E_ORACLE_CASE_DUPLICATE",
                "oracle.records",
                "oracle must contain exactly one record per challenge case",
            );
        }
        if !catalog_ids.contains(record.case_id.as_str()) {
            push_error(
                &mut report,
                Some(&record.case_id),
                "E_ORACLE_CASE_UNKNOWN",
                "oracle.records",
                "oracle contains a case ID that is absent from the catalog",
            );
        }
    }

    let mut receipts_by_id = BTreeMap::<&str, &ObserverReceipt>::new();
    for record in &receipts.records {
        if receipts_by_id.insert(&record.case_id, record).is_some() {
            push_error(
                &mut report,
                Some(&record.case_id),
                "E_RECEIPT_CASE_DUPLICATE",
                "observer_receipts.records",
                "observer receipts must contain exactly one record per challenge case",
            );
        }
        if !catalog_ids.contains(record.case_id.as_str()) {
            push_error(
                &mut report,
                Some(&record.case_id),
                "E_RECEIPT_CASE_UNKNOWN",
                "observer_receipts.records",
                "observer receipts contain a case ID that is absent from the catalog",
            );
        }
    }

    let mut complete = oracle_by_id.len() == catalog.cases.len()
        && receipts_by_id.len() == catalog.cases.len();

    for case in &catalog.cases {
        let before_errors = report.errors.len();
        let oracle_record = oracle_by_id.get(case.case_id.as_str()).copied();
        let receipt_record = receipts_by_id.get(case.case_id.as_str()).copied();

        match (oracle_record, receipt_record) {
            (Some(oracle_record), Some(receipt_record)) => {
                adjudicate_case(
                    case,
                    oracle_record,
                    receipt_record,
                    &mut report,
                );
                report.counts.adjudicated_cases += 1;
            }
            (None, Some(_)) => {
                complete = false;
                push_error(
                    &mut report,
                    Some(&case.case_id),
                    "E_ORACLE_CASE_MISSING",
                    "oracle.records",
                    "the independent oracle has no record for this challenge case",
                );
            }
            (Some(_), None) => {
                complete = false;
                push_error(
                    &mut report,
                    Some(&case.case_id),
                    "E_RECEIPT_CASE_MISSING",
                    "observer_receipts.records",
                    "the observer receipt ledger has no record for this challenge case",
                );
            }
            (None, None) => {
                complete = false;
                push_error(
                    &mut report,
                    Some(&case.case_id),
                    "E_ORACLE_AND_RECEIPT_MISSING",
                    "records",
                    "both the independent oracle and observer receipt are missing",
                );
            }
        }

        let error_codes = report.errors[before_errors..]
            .iter()
            .filter(|error| error.case_id.as_deref() == Some(case.case_id.as_str()))
            .map(|error| error.code.clone())
            .collect::<Vec<_>>();
        let passed = error_codes.is_empty()
            && oracle_record.is_some()
            && receipt_record.is_some();
        if passed {
            report.counts.passed_cases += 1;
        } else {
            report.counts.failed_cases += 1;
        }

        report.cases.push(CaseAssessment {
            case_id: case.case_id.clone(),
            expected_disposition: case.expected_disposition.clone(),
            oracle_ground_truth: oracle_record.map(|record| record.ground_truth.clone()),
            observer_disposition: receipt_record.map(|record| record.disposition.clone()),
            reuse_decision: receipt_record.map(|record| record.reuse_decision.clone()),
            passed,
            error_codes,
        });
    }

    validate_pairs(
        &catalog.cases,
        &oracle_by_id,
        &receipts_by_id,
        &mut report,
    );

    report.counts.passed_cases = 0;
    report.counts.failed_cases = 0;
    for assessment in &mut report.cases {
        let codes = report
            .errors
            .iter()
            .filter(|error| error.case_id.as_deref() == Some(assessment.case_id.as_str()))
            .map(|error| error.code.clone())
            .collect::<BTreeSet<_>>();
        assessment.error_codes = codes.into_iter().collect();
        assessment.passed = assessment.error_codes.is_empty()
            && assessment.oracle_ground_truth.is_some()
            && assessment.observer_disposition.is_some();
        if assessment.passed {
            report.counts.passed_cases += 1;
        } else {
            report.counts.failed_cases += 1;
        }
    }

    report.adjudication_complete = complete
        && report.counts.adjudicated_cases == catalog.cases.len();
    report.adjudication_passed = report.adjudication_complete
        && report.counts.passed_cases == catalog.cases.len()
        && report.errors.is_empty();

    // These are policy constants, not values inferred from a green run.
    report.qualification_eligible = false;
    report.oracle_authenticity_established = false;
    report
}

fn validate_envelopes(
    catalog: &Catalog,
    oracle: &OracleLedger,
    receipts: &ObserverReceipts,
    report: &mut ResultAuditReport,
) {
    if catalog.format_version != 1
        || catalog.suite_id != "basalt-observer-challenges-v1"
    {
        push_error(
            report,
            None,
            "E_CATALOG_IDENTITY",
            "catalog",
            "unsupported challenge catalog format or suite ID",
        );
    }
    if catalog.profile_status != "candidate_not_qualified"
        || catalog.execution_status != "not_executed"
    {
        push_error(
            report,
            None,
            "E_CATALOG_STATUS",
            "catalog",
            "catalog must remain candidate_not_qualified/not_executed; execution belongs in separate ledgers",
        );
    }
    if catalog.profile_id.trim().is_empty()
        || catalog.scoped_claim.trim().is_empty()
        || catalog.oracle_policy.ground_truth_source.trim().is_empty()
        || !catalog.oracle_policy.observer_under_test_must_not_author_the_oracle
        || !catalog.oracle_policy.independent_replay_required_for_confirmatory_use
        || catalog.oracle_policy.score_unit != "independent_task_scenario"
        || !catalog.oracle_policy.repeated_events_are_not_independent_samples
        || catalog.expected_dispositions.is_empty()
    {
        push_error(
            report,
            None,
            "E_CATALOG_ORACLE_POLICY",
            "catalog.oracle_policy",
            "catalog must declare an independent scenario-level oracle and replay requirement",
        );
    }

    if oracle.format_version != 1
        || oracle.suite_id != "basalt-observer-oracle-v1"
    {
        push_error(
            report,
            None,
            "E_ORACLE_IDENTITY",
            "oracle",
            "unsupported oracle ledger format or suite ID",
        );
    }
    if receipts.format_version != 1
        || receipts.suite_id != "basalt-observer-receipts-v1"
    {
        push_error(
            report,
            None,
            "E_RECEIPTS_IDENTITY",
            "observer_receipts",
            "unsupported observer receipt format or suite ID",
        );
    }

    if oracle.catalog_digest != report.catalog_digest
        || receipts.catalog_digest != report.catalog_digest
    {
        push_error(
            report,
            None,
            "E_CATALOG_DIGEST_MISMATCH",
            "catalog_digest",
            "oracle and observer receipt ledgers must bind the exact catalog bytes supplied to the auditor",
        );
    }
    if !valid_digest(&oracle.catalog_digest)
        || !valid_digest(&oracle.harness.binary_digest)
        || !valid_digest(&oracle.harness.fixture_digest)
        || !valid_digest(&receipts.catalog_digest)
        || !valid_digest(&receipts.observer_binary_digest)
    {
        push_error(
            report,
            None,
            "E_DIGEST_FORMAT",
            "$",
            "all catalog, harness, fixture, and observer digests must be sha256:<64 lowercase hex characters>",
        );
    }
    if !valid_identifier(&oracle.run_id)
        || oracle.harness.harness_id.trim().is_empty()
    {
        push_error(
            report,
            None,
            "E_ORACLE_PROVENANCE_FIELDS",
            "oracle.harness",
            "run ID and harness identity must be non-empty bounded identifiers",
        );
    }
    if oracle.profile_id != catalog.profile_id
        || receipts.observer_profile_id != catalog.profile_id
    {
        push_error(
            report,
            None,
            "E_PROFILE_MISMATCH",
            "profile_id",
            "catalog, oracle ledger, and observer receipts must name the same bounded profile",
        );
    }
    if receipts.run_id != oracle.run_id {
        push_error(
            report,
            None,
            "E_RUN_ID_MISMATCH",
            "run_id",
            "oracle and observer receipt ledgers must refer to the same run",
        );
    }
}

fn adjudicate_case(
    case: &ChallengeCase,
    oracle: &OracleRecord,
    receipt: &ObserverReceipt,
    report: &mut ResultAuditReport,
) {
    let case_id = case.case_id.as_str();
    let path = format!("cases[{case_id}]");
    let expected_truth = match case.expected_disposition.as_str() {
        "capture_dependency" => "consumed",
        "preserve_unrelated" => "not_consumed",
        "report_unresolved" => "unsupported",
        "invalidate_receipt" => "integrity_failure",
        _ => {
            push_error(
                report,
                Some(case_id),
                "E_EXPECTED_DISPOSITION",
                &format!("{path}.expected_disposition"),
                "unsupported expected disposition",
            );
            "invalid"
        }
    };
    let expected_witness = match expected_truth {
        "consumed" => "controlled_read",
        "not_consumed" => "controlled_non_read",
        "unsupported" => "unsupported_channel",
        "integrity_failure" => "integrity_fault",
        _ => "invalid",
    };

    if oracle.ground_truth != expected_truth {
        push_error(
            report,
            Some(case_id),
            "E_ORACLE_GROUND_TRUTH",
            &format!("{path}.oracle.ground_truth"),
            "independent oracle outcome does not match the catalog's expected scenario class",
        );
    }
    if !oracle.injected_resource_id.present
        || oracle.injected_resource_id.value != case.ground_truth_resource_id.value
    {
        push_error(
            report,
            Some(case_id),
            "E_ORACLE_RESOURCE_ID",
            &format!("{path}.oracle.injected_resource_id"),
            "oracle must explicitly bind the catalog's ground-truth resource ID, including null",
        );
    }
    if oracle.witness_kind != expected_witness {
        push_error(
            report,
            Some(case_id),
            "E_ORACLE_WITNESS_KIND",
            &format!("{path}.oracle.witness_kind"),
            "oracle witness kind is inconsistent with the expected scenario class",
        );
    }
    if oracle.independent_replay != "confirmed" {
        push_error(
            report,
            Some(case_id),
            "E_ORACLE_REPLAY_UNCONFIRMED",
            &format!("{path}.oracle.independent_replay"),
            "independent fixture replay must confirm the injected ground-truth condition",
        );
    }

    let needs_mutation = matches!(
        case.expected_disposition.as_str(),
        "capture_dependency" | "preserve_unrelated"
    );
    if needs_mutation && !oracle.mutation.applied {
        push_error(
            report,
            Some(case_id),
            "E_MUTATION_NOT_APPLIED",
            &format!("{path}.oracle.mutation"),
            "dependency and unrelated-control cases require an applied resource mutation",
        );
    }
    if !oracle.mutation.before_digest.present || !oracle.mutation.after_digest.present {
        push_error(
            report,
            Some(case_id),
            "E_MUTATION_DIGEST_FIELDS_REQUIRED",
            &format!("{path}.oracle.mutation"),
            "before_digest and after_digest must be present, using null when no mutation was applied",
        );
    }
    match (
        oracle.mutation.applied,
        oracle.mutation.before_digest.value.as_deref(),
        oracle.mutation.after_digest.value.as_deref(),
    ) {
        (true, Some(before), Some(after)) => {
            if !valid_digest(before) || !valid_digest(after) || before == after {
                push_error(
                    report,
                    Some(case_id),
                    "E_MUTATION_DIGEST",
                    &format!("{path}.oracle.mutation"),
                    "an applied mutation requires distinct, valid before/after SHA-256 digests",
                );
            }
        }
        (true, _, _) => push_error(
            report,
            Some(case_id),
            "E_MUTATION_DIGEST_MISSING",
            &format!("{path}.oracle.mutation"),
            "an applied mutation must include both before and after digests",
        ),
        (false, None, None) => {}
        (false, _, _) => push_error(
            report,
            Some(case_id),
            "E_MUTATION_DIGEST_WITHOUT_MUTATION",
            &format!("{path}.oracle.mutation"),
            "before/after digests must be null when no mutation was applied",
        ),
    }

    if receipt.facts.len() > MAX_FACTS {
        push_error(
            report,
            Some(case_id),
            "E_RECEIPT_FACT_LIMIT",
            &format!("{path}.receipt.facts"),
            "receipt fact count exceeds 128",
        );
    }
    let mut fact_set = BTreeSet::new();
    for fact in &receipt.facts {
        if fact.trim().is_empty() || !fact_set.insert(fact.as_str()) {
            push_error(
                report,
                Some(case_id),
                "E_RECEIPT_FACT_DUPLICATE_OR_EMPTY",
                &format!("{path}.receipt.facts"),
                "receipt facts must be non-empty and unique",
            );
        }
    }
    let mut hidden_dependency_missed = false;
    for required in &case.required_receipt_facts {
        if !fact_set.contains(required.as_str()) {
            push_error(
                report,
                Some(case_id),
                "E_RECEIPT_FACT_MISSING",
                &format!("{path}.receipt.facts"),
                "observer receipt is missing a catalog-required fact",
            );
            hidden_dependency_missed |= case.expected_disposition == "capture_dependency";
        }
    }

    if receipt.observed_resources.len() > MAX_OBSERVED_RESOURCES {
        push_error(
            report,
            Some(case_id),
            "E_OBSERVED_RESOURCE_LIMIT",
            &format!("{path}.receipt.observed_resources"),
            "observed resource count exceeds 4096",
        );
    }
    let mut observed_ids = BTreeSet::new();
    for observed in &receipt.observed_resources {
        report.metrics.observed_resource_attributions += 1;
        if !valid_identifier(&observed.resource_id)
            || !valid_digest(&observed.digest)
            || !observed_ids.insert(observed.resource_id.as_str())
        {
            push_error(
                report,
                Some(case_id),
                "E_OBSERVED_RESOURCE_INVALID",
                &format!("{path}.receipt.observed_resources"),
                "observed resources require unique IDs and valid SHA-256 digests",
            );
        }
    }

    if receipt.unresolved_channels.len() > MAX_UNRESOLVED_CHANNELS {
        push_error(
            report,
            Some(case_id),
            "E_UNRESOLVED_CHANNEL_LIMIT",
            &format!("{path}.receipt.unresolved_channels"),
            "unresolved channel count exceeds 64",
        );
    }
    let mut unresolved_set = BTreeSet::new();
    for channel in &receipt.unresolved_channels {
        if channel.trim().is_empty() || !unresolved_set.insert(channel.as_str()) {
            push_error(
                report,
                Some(case_id),
                "E_UNRESOLVED_CHANNEL_DUPLICATE_OR_EMPTY",
                &format!("{path}.receipt.unresolved_channels"),
                "unresolved channels must be non-empty and unique",
            );
        }
    }

    let resource_id = case.ground_truth_resource_id.value.as_deref();
    let matching_observations = resource_id
        .map(|id| {
            receipt
                .observed_resources
                .iter()
                .filter(|observed| observed.resource_id == id)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    match case.expected_disposition.as_str() {
        "capture_dependency" => {
            if matching_observations.len() != 1 {
                hidden_dependency_missed = true;
                push_error(
                    report,
                    Some(case_id),
                    "E_HIDDEN_DEPENDENCY_MISSED",
                    &format!("{path}.receipt.observed_resources"),
                    "consumed ground-truth resource must be represented exactly once in the observer receipt",
                );
            } else if let Some(expected_digest) = oracle.mutation.after_digest.value.as_deref() {
                if matching_observations[0].digest != expected_digest {
                    hidden_dependency_missed = true;
                    push_error(
                        report,
                        Some(case_id),
                        "E_DEPENDENCY_DIGEST_MISMATCH",
                        &format!("{path}.receipt.observed_resources"),
                        "observed resource digest does not match the independently injected post-mutation state",
                    );
                }
            }
        }
        "preserve_unrelated" => {
            if !matching_observations.is_empty() {
                push_error(
                    report,
                    Some(case_id),
                    "E_UNRELATED_RESOURCE_ATTRIBUTED",
                    &format!("{path}.receipt.observed_resources"),
                    "independently proven unconsumed resource must not be attributed as a dependency",
                );
            }
        }
        _ => {}
    }

    let integrity_fault = receipt.integrity.event_loss
        || receipt.integrity.observer_restart
        || receipt.integrity.unmonitored_process
        || receipt.integrity.snapshot_race
        || receipt.integrity.root_escape;
    let mut integrity_failure_not_invalidated = false;

    if integrity_fault
        && (receipt.disposition != "invalidated"
            || receipt.reuse_decision != "invalidated")
    {
        integrity_failure_not_invalidated = true;
        push_error(
            report,
            Some(case_id),
            "E_INTEGRITY_FAILURE_NOT_INVALIDATED",
            &format!("{path}.receipt.integrity"),
            "event loss, restart, unmonitored process, snapshot race, or root escape must invalidate both the receipt and its reuse decision",
        );
    }

    match case.expected_disposition.as_str() {
        "capture_dependency" => {
            if receipt.disposition != "captured" {
                push_error(
                    report,
                    Some(case_id),
                    "E_CAPTURE_DISPOSITION",
                    &format!("{path}.receipt.disposition"),
                    "captured dependency scenario must produce captured disposition",
                );
            }
            match receipt.reuse_decision.as_str() {
                "invalidated" | "reverified" => {}
                "reused" => {
                    report.metrics.stale_evidence_false_acceptances += 1;
                    push_error(
                        report,
                        Some(case_id),
                        "E_STALE_EVIDENCE_FALSE_ACCEPTED",
                        &format!("{path}.receipt.reuse_decision"),
                        "prior evidence was reused after a consumed dependency changed",
                    );
                }
                _ => push_error(
                    report,
                    Some(case_id),
                    "E_REVERIFICATION_REQUIRED",
                    &format!("{path}.receipt.reuse_decision"),
                    "changed consumed dependency requires invalidation or re-verification",
                ),
            }
        }
        "preserve_unrelated" => {
            if receipt.disposition != "preserved" {
                push_error(
                    report,
                    Some(case_id),
                    "E_PRESERVE_DISPOSITION",
                    &format!("{path}.receipt.disposition"),
                    "unrelated positive control must retain its evidence",
                );
            }
            if receipt.reuse_decision != "reused" {
                report.metrics.unrelated_evidence_not_retained += 1;
                if matches!(
                    receipt.reuse_decision.as_str(),
                    "invalidated" | "reverified"
                ) {
                    report.metrics.false_invalidations += 1;
                }
                push_error(
                    report,
                    Some(case_id),
                    "E_UNRELATED_EVIDENCE_NOT_RETAINED",
                    &format!("{path}.receipt.reuse_decision"),
                    "independently proven unaffected evidence must be reused, not invalidated or reverified",
                );
            }
        }
        "report_unresolved" => {
            let mut unsupported_channel_violation = false;
            if !unresolved_set.contains(case.channel.as_str()) {
                unsupported_channel_violation = true;
                push_error(
                    report,
                    Some(case_id),
                    "E_UNRESOLVED_CHANNEL_NOT_DISCLOSED",
                    &format!("{path}.receipt.unresolved_channels"),
                    "unsupported observation channel must be explicitly reported unresolved",
                );
            }
            if !matches!(receipt.disposition.as_str(), "unresolved" | "invalidated") {
                unsupported_channel_violation = true;
                push_error(
                    report,
                    Some(case_id),
                    "E_UNSUPPORTED_CHANNEL_ACCEPTED",
                    &format!("{path}.receipt.disposition"),
                    "unsupported channel must be unresolved or invalidated, never captured or complete",
                );
            }
            if !matches!(
                receipt.reuse_decision.as_str(),
                "unresolved" | "invalidated"
            ) {
                unsupported_channel_violation = true;
                if receipt.reuse_decision == "reused" {
                    report.metrics.stale_evidence_false_acceptances += 1;
                }
                push_error(
                    report,
                    Some(case_id),
                    "E_UNRESOLVED_EVIDENCE_REUSED",
                    &format!("{path}.receipt.reuse_decision"),
                    "evidence depending on an unsupported channel cannot be reused",
                );
            }
            if unsupported_channel_violation {
                report.metrics.unresolved_channel_misses += 1;
            }
        }
        "invalidate_receipt" => {
            if !integrity_fault {
                push_error(
                    report,
                    Some(case_id),
                    "E_INTEGRITY_SIGNAL_MISSING",
                    &format!("{path}.receipt.integrity"),
                    "integrity-failure control must be independently visible in receipt integrity signals",
                );
            }
            if receipt.disposition != "invalidated"
                || receipt.reuse_decision != "invalidated"
            {
                integrity_failure_not_invalidated = true;
                push_error(
                    report,
                    Some(case_id),
                    "E_INVALID_RECEIPT_ACCEPTED",
                    &format!("{path}.receipt"),
                    "integrity-failure challenge must invalidate the receipt and prohibit reuse",
                );
            }
        }
        _ => {}
    }

    if hidden_dependency_missed {
        report.metrics.hidden_dependency_misses += 1;
    }
    if integrity_failure_not_invalidated {
        report.metrics.integrity_failures_not_invalidated += 1;
    }
}

fn validate_pairs(
    cases: &[ChallengeCase],
    oracle_by_id: &BTreeMap<&str, &OracleRecord>,
    receipts_by_id: &BTreeMap<&str, &ObserverReceipt>,
    report: &mut ResultAuditReport,
) {
    let case_by_id = cases
        .iter()
        .map(|case| (case.case_id.as_str(), case))
        .collect::<BTreeMap<_, _>>();

    for case in cases {
        for paired_id in &case.paired_case_ids {
            // Reciprocal links are adjudicated once in stable lexical order.
            if case.case_id.as_str() > paired_id.as_str() {
                continue;
            }
            let Some(paired_case) = case_by_id.get(paired_id.as_str()).copied() else {
                push_error(
                    report,
                    Some(&case.case_id),
                    "E_PAIR_CASE_MISSING",
                    "catalog.cases.paired_case_ids",
                    "paired case is missing from the challenge catalog",
                );
                continue;
            };
            if !paired_case
                .paired_case_ids
                .iter()
                .any(|candidate| candidate == &case.case_id)
            {
                push_error(
                    report,
                    Some(&case.case_id),
                    "E_PAIR_NOT_RECIPROCAL",
                    "catalog.cases.paired_case_ids",
                    "paired challenge relationship must be reciprocal",
                );
            }

            let Some(left_oracle) = oracle_by_id.get(case.case_id.as_str()).copied() else {
                continue;
            };
            let Some(right_oracle) = oracle_by_id.get(paired_id.as_str()).copied() else {
                continue;
            };
            if left_oracle.injected_resource_id.value
                != right_oracle.injected_resource_id.value
                || left_oracle.mutation.before_digest.value != right_oracle.mutation.before_digest.value
                || left_oracle.mutation.after_digest.value != right_oracle.mutation.after_digest.value
                || !left_oracle.mutation.applied
                || !right_oracle.mutation.applied
            {
                push_error(
                    report,
                    Some(&case.case_id),
                    "E_PAIR_GROUND_TRUTH_MISMATCH",
                    "oracle.records",
                    "paired recall/selectivity controls must mutate the same resource from identical before/after states",
                );
            }

            let dispositions = [
                case.expected_disposition.as_str(),
                paired_case.expected_disposition.as_str(),
            ];
            let is_recall_selectivity_pair = dispositions.contains(&"capture_dependency")
                && dispositions.contains(&"preserve_unrelated");
            if !is_recall_selectivity_pair {
                push_error(
                    report,
                    Some(&case.case_id),
                    "E_PAIR_DISPOSITION_MISMATCH",
                    "catalog.cases.paired_case_ids",
                    "paired controls must include one capture_dependency and one preserve_unrelated case",
                );
            }

            if let (Some(left), Some(right)) = (
                receipts_by_id.get(case.case_id.as_str()).copied(),
                receipts_by_id.get(paired_id.as_str()).copied(),
            ) {
                let capture_case = if case.expected_disposition == "capture_dependency" {
                    Some((case, left))
                } else if paired_case.expected_disposition == "capture_dependency" {
                    Some((paired_case, right))
                } else {
                    None
                };
                let preserve_case = if case.expected_disposition == "preserve_unrelated" {
                    Some((case, left))
                } else if paired_case.expected_disposition == "preserve_unrelated" {
                    Some((paired_case, right))
                } else {
                    None
                };
                if let (Some((capture, capture_receipt)), Some((preserve, preserve_receipt))) =
                    (capture_case, preserve_case)
                {
                    let resource = capture.ground_truth_resource_id.value.as_deref();
                    let capture_has_resource = resource.is_some_and(|id| {
                        capture_receipt
                            .observed_resources
                            .iter()
                            .any(|observed| observed.resource_id == id)
                    });
                    let preserve_has_resource = resource.is_some_and(|id| {
                        preserve_receipt
                            .observed_resources
                            .iter()
                            .any(|observed| observed.resource_id == id)
                    });
                    if !capture_has_resource || preserve_has_resource {
                        push_error(
                            report,
                            Some(&case.case_id),
                            "E_PAIR_SELECTIVITY_CONTRADICTION",
                            "observer_receipts.records",
                            "the same resource must be captured when consumed and absent when independently proven unconsumed",
                        );
                    }
                }
            }
        }
    }
}

fn read_bounded(path: &Path) -> anyhow::Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut bytes = Vec::new();
    file.take(MAX_INPUT_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_INPUT_BYTES {
        anyhow::bail!("BASALT adjudication input exceeds 16 MiB: {}", path.display());
    }
    Ok(bytes)
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.trim() == value
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(byte, b'-' | b'_' | b'.' | b':' | b'/')
        })
}

fn valid_digest(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return false;
    };
    hex.len() == 64
        && hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_tagged(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(64);
    for byte in digest {
        // Writing to a String is infallible.
        let _ = write!(&mut hex, "{byte:02x}");
    }
    format!("sha256:{hex}")
}

fn push_error(
    report: &mut ResultAuditReport,
    case_id: Option<&str>,
    code: &str,
    path: &str,
    message: &str,
) {
    report.errors.push(ResultAuditError {
        code: code.to_string(),
        case_id: case_id.map(str::to_string),
        path: path.to_string(),
        message: message.to_string(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn fixture_inputs() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let before = format!("sha256:{}", "a".repeat(64));
        let after = format!("sha256:{}", "b".repeat(64));
        let resource_id = "fixture:paired-secret.txt";

        let catalog = json!({
            "format_version": 1,
            "suite_id": "basalt-observer-challenges-v1",
            "profile_id": "linux-hermetic-candidate-v1",
            "profile_status": "candidate_not_qualified",
            "execution_status": "not_executed",
            "scoped_claim": "four-case paired, unresolved, and integrity-fault oracle/adjudicator test",
            "oracle_policy": {
                "ground_truth_source": "challenge_harness_injected_resources_and_independently_checked_mutation_pairs",
                "observer_under_test_must_not_author_the_oracle": true,
                "independent_replay_required_for_confirmatory_use": true,
                "score_unit": "independent_task_scenario",
                "repeated_events_are_not_independent_samples": true
            },
            "expected_dispositions": [
                "capture_dependency", "report_unresolved",
                "invalidate_receipt", "preserve_unrelated"
            ],
            "cases": [
                {
                    "case_id": "hidden-input-read",
                    "family": "filesystem",
                    "channel": "filesystem",
                    "mutation_kind": "untracked_read",
                    "ground_truth_resource_id": resource_id,
                    "expected_disposition": "capture_dependency",
                    "required_receipt_facts": [
                        "resource_identity", "content_digest", "untracked_classification"
                    ],
                    "paired_case_ids": ["unused-neighbor"],
                    "description": "The task consumes the changed untracked resource.",
                    "safety_property": "Changed consumed inputs must be observed."
                },
                {
                    "case_id": "unused-neighbor",
                    "family": "positive_control",
                    "channel": "filesystem",
                    "mutation_kind": "irrelevant_mutation",
                    "ground_truth_resource_id": resource_id,
                    "expected_disposition": "preserve_unrelated",
                    "required_receipt_facts": [
                        "mutation_identity", "non_dependency_witness"
                    ],
                    "paired_case_ids": ["hidden-input-read"],
                    "description": "The task does not consume the same changed resource.",
                    "safety_property": "Unchanged result evidence must be retained."
                },
                {
                    "case_id": "unresolved-env-channel",
                    "family": "environment",
                    "channel": "environment",
                    "mutation_kind": "nondeterministic_source",
                    "ground_truth_resource_id": null,
                    "expected_disposition": "report_unresolved",
                    "required_receipt_facts": ["unresolved_channel", "virtualization_status"],
                    "paired_case_ids": [],
                    "description": "The candidate profile cannot observe a required environment source.",
                    "safety_property": "Unsupported observation must be disclosed and cannot be reused as confirmatory."
                },
                {
                    "case_id": "event-loss",
                    "family": "observer_integrity",
                    "channel": "observer_integrity",
                    "mutation_kind": "event_overflow",
                    "ground_truth_resource_id": null,
                    "expected_disposition": "invalidate_receipt",
                    "required_receipt_facts": ["drop_counter", "overflow_status", "receipt_integrity"],
                    "paired_case_ids": [],
                    "description": "The observer event queue overflows during the capture interval.",
                    "safety_property": "Event loss invalidates the receipt and prohibits reuse."
                }
            ]
        });
        let catalog_bytes = serde_json::to_vec(&catalog).expect("catalog fixture serializes");
        let catalog_digest = sha256_tagged(&catalog_bytes);

        let oracle = json!({
            "format_version": 1,
            "suite_id": "basalt-observer-oracle-v1",
            "run_id": "test-run-001",
            "catalog_digest": catalog_digest,
            "profile_id": "linux-hermetic-candidate-v1",
            "harness": {
                "harness_id": "basalt-fixture-harness-v1",
                "binary_digest": format!("sha256:{}", "c".repeat(64)),
                "fixture_digest": format!("sha256:{}", "d".repeat(64))
            },
            "records": [
                {
                    "case_id": "hidden-input-read",
                    "ground_truth": "consumed",
                    "injected_resource_id": resource_id,
                    "mutation": {
                        "applied": true,
                        "before_digest": before,
                        "after_digest": after
                    },
                    "witness_kind": "controlled_read",
                    "independent_replay": "confirmed"
                },
                {
                    "case_id": "unused-neighbor",
                    "ground_truth": "not_consumed",
                    "injected_resource_id": resource_id,
                    "mutation": {
                        "applied": true,
                        "before_digest": before,
                        "after_digest": after
                    },
                    "witness_kind": "controlled_non_read",
                    "independent_replay": "confirmed"
                },
                {
                    "case_id": "unresolved-env-channel",
                    "ground_truth": "unsupported",
                    "injected_resource_id": null,
                    "mutation": {
                        "applied": false,
                        "before_digest": null,
                        "after_digest": null
                    },
                    "witness_kind": "unsupported_channel",
                    "independent_replay": "confirmed"
                },
                {
                    "case_id": "event-loss",
                    "ground_truth": "integrity_failure",
                    "injected_resource_id": null,
                    "mutation": {
                        "applied": false,
                        "before_digest": null,
                        "after_digest": null
                    },
                    "witness_kind": "integrity_fault",
                    "independent_replay": "confirmed"
                }
            ]
        });
        let oracle_bytes = serde_json::to_vec(&oracle).expect("oracle fixture serializes");

        let receipts = json!({
            "format_version": 1,
            "suite_id": "basalt-observer-receipts-v1",
            "run_id": "test-run-001",
            "catalog_digest": catalog_digest,
            "observer_profile_id": "linux-hermetic-candidate-v1",
            "observer_binary_digest": format!("sha256:{}", "e".repeat(64)),
            "records": [
                {
                    "case_id": "hidden-input-read",
                    "disposition": "captured",
                    "observed_resources": [{
                        "resource_id": resource_id,
                        "digest": after
                    }],
                    "facts": [
                        "resource_identity", "content_digest", "untracked_classification"
                    ],
                    "unresolved_channels": [],
                    "reuse_decision": "reverified",
                    "integrity": {
                        "event_loss": false,
                        "observer_restart": false,
                        "unmonitored_process": false,
                        "snapshot_race": false,
                        "root_escape": false
                    }
                },
                {
                    "case_id": "unused-neighbor",
                    "disposition": "preserved",
                    "observed_resources": [],
                    "facts": ["mutation_identity", "non_dependency_witness"],
                    "unresolved_channels": [],
                    "reuse_decision": "reused",
                    "integrity": {
                        "event_loss": false,
                        "observer_restart": false,
                        "unmonitored_process": false,
                        "snapshot_race": false,
                        "root_escape": false
                    }
                },
                {
                    "case_id": "unresolved-env-channel",
                    "disposition": "unresolved",
                    "observed_resources": [],
                    "facts": ["unresolved_channel", "virtualization_status"],
                    "unresolved_channels": ["environment"],
                    "reuse_decision": "unresolved",
                    "integrity": {
                        "event_loss": false,
                        "observer_restart": false,
                        "unmonitored_process": false,
                        "snapshot_race": false,
                        "root_escape": false
                    }
                },
                {
                    "case_id": "event-loss",
                    "disposition": "invalidated",
                    "observed_resources": [],
                    "facts": ["drop_counter", "overflow_status", "receipt_integrity"],
                    "unresolved_channels": [],
                    "reuse_decision": "invalidated",
                    "integrity": {
                        "event_loss": true,
                        "observer_restart": false,
                        "unmonitored_process": false,
                        "snapshot_race": false,
                        "root_escape": false
                    }
                }
            ]
        });
        let receipts_bytes = serde_json::to_vec(&receipts).expect("receipt fixture serializes");
        (catalog_bytes, oracle_bytes, receipts_bytes)
    }

    #[test]
    fn accepts_paired_recall_and_selectivity_control_without_qualifying_profile() {
        let (catalog, oracle, receipts) = fixture_inputs();
        let report = audit_bytes(&catalog, &oracle, &receipts);
        assert!(report.adjudication_complete, "{:?}", report.errors);
        assert!(report.adjudication_passed, "{:?}", report.errors);
        assert_eq!(report.counts.passed_cases, 4);
        assert_eq!(report.metrics.hidden_dependency_misses, 0);
        assert_eq!(report.metrics.false_invalidations, 0);
        assert_eq!(report.metrics.unrelated_evidence_not_retained, 0);
        assert!(!report.qualification_eligible);
        assert!(!report.oracle_authenticity_established);
        assert!(!report.oracle_replay_executed_by_adjudicator);
        assert!(!report.observer_receipt_authenticity_established);
    }

    #[test]
    fn rejects_reuse_after_a_consumed_dependency_changes() {
        let (catalog, oracle, receipts) = fixture_inputs();
        let mut receipt_value: Value =
            serde_json::from_slice(&receipts).expect("receipt fixture parses");
        receipt_value["records"][0]["reuse_decision"] = json!("reused");
        let receipts = serde_json::to_vec(&receipt_value).expect("mutated receipt serializes");
        let report = audit_bytes(&catalog, &oracle, &receipts);
        assert!(!report.adjudication_passed);
        assert_eq!(report.metrics.stale_evidence_false_acceptances, 1);
        assert!(report.errors.iter().any(|error| {
            error.code == "E_STALE_EVIDENCE_FALSE_ACCEPTED"
        }));
    }

    #[test]
    fn rejects_event_loss_disguised_as_a_successful_capture() {
        let (catalog, oracle, receipts) = fixture_inputs();
        let mut receipt_value: Value =
            serde_json::from_slice(&receipts).expect("receipt fixture parses");
        receipt_value["records"][0]["integrity"]["event_loss"] = json!(true);
        let receipts = serde_json::to_vec(&receipt_value).expect("mutated receipt serializes");
        let report = audit_bytes(&catalog, &oracle, &receipts);
        assert!(!report.adjudication_passed);
        assert_eq!(report.metrics.integrity_failures_not_invalidated, 1);
        assert!(report.errors.iter().any(|error| {
            error.code == "E_INTEGRITY_FAILURE_NOT_INVALIDATED"
        }));
    }

    #[test]
    fn rejects_an_unrelated_resource_that_is_attributed_as_a_dependency() {
        let (catalog, oracle, receipts) = fixture_inputs();
        let mut receipt_value: Value =
            serde_json::from_slice(&receipts).expect("receipt fixture parses");
        receipt_value["records"][1]["observed_resources"] = json!([{
            "resource_id": "fixture:paired-secret.txt",
            "digest": format!("sha256:{}", "b".repeat(64))
        }]);
        let receipts = serde_json::to_vec(&receipt_value).expect("mutated receipt serializes");
        let report = audit_bytes(&catalog, &oracle, &receipts);
        assert!(!report.adjudication_passed);
        assert!(report.errors.iter().any(|error| {
            error.code == "E_UNRELATED_RESOURCE_ATTRIBUTED"
        }));
    }

    #[test]
    fn rejects_unsupported_channel_when_receipt_claims_capture() {
        let (catalog, oracle, receipts) = fixture_inputs();
        let mut receipt_value: Value =
            serde_json::from_slice(&receipts).expect("receipt fixture parses");
        receipt_value["records"][2]["disposition"] = json!("captured");
        receipt_value["records"][2]["unresolved_channels"] = json!([]);
        receipt_value["records"][2]["reuse_decision"] = json!("reused");
        let receipts = serde_json::to_vec(&receipt_value).expect("mutated receipt serializes");
        let report = audit_bytes(&catalog, &oracle, &receipts);
        assert!(!report.adjudication_passed);
        assert_eq!(report.metrics.unresolved_channel_misses, 1);
        assert!(report.errors.iter().any(|error| {
            error.code == "E_UNRESOLVED_CHANNEL_NOT_DISCLOSED"
        }));
    }

    #[test]
    fn rejects_integrity_loss_on_a_capture_case_even_when_receipt_is_invalidated() {
        let (catalog, oracle, receipts) = fixture_inputs();
        let mut receipt_value: Value =
            serde_json::from_slice(&receipts).expect("receipt fixture parses");
        receipt_value["records"][0]["integrity"]["event_loss"] = json!(true);
        receipt_value["records"][0]["disposition"] = json!("invalidated");
        receipt_value["records"][0]["reuse_decision"] = json!("reverified");
        let receipts = serde_json::to_vec(&receipt_value).expect("mutated receipt serializes");
        let report = audit_bytes(&catalog, &oracle, &receipts);
        assert!(!report.adjudication_passed);
        assert_eq!(report.metrics.integrity_failures_not_invalidated, 1);
        assert!(report.errors.iter().any(|error| {
            error.code == "E_INTEGRITY_FAILURE_NOT_INVALIDATED"
        }));
    }

    #[test]
    fn rejects_integrity_failure_when_reuse_is_not_invalidated() {
        let (catalog, oracle, receipts) = fixture_inputs();
        let mut receipt_value: Value =
            serde_json::from_slice(&receipts).expect("receipt fixture parses");
        receipt_value["records"][3]["reuse_decision"] = json!("reused");
        let receipts = serde_json::to_vec(&receipt_value).expect("mutated receipt serializes");
        let report = audit_bytes(&catalog, &oracle, &receipts);
        assert!(!report.adjudication_passed);
        assert_eq!(report.metrics.integrity_failures_not_invalidated, 1);
        assert!(report.errors.iter().any(|error| {
            error.code == "E_INVALID_RECEIPT_ACCEPTED"
        }));
    }

    #[test]
    fn rejects_false_invalidation_of_unrelated_evidence() {
        let (catalog, oracle, receipts) = fixture_inputs();
        let mut receipt_value: Value =
            serde_json::from_slice(&receipts).expect("receipt fixture parses");
        receipt_value["records"][1]["reuse_decision"] = json!("invalidated");
        let receipts = serde_json::to_vec(&receipt_value).expect("mutated receipt serializes");
        let report = audit_bytes(&catalog, &oracle, &receipts);
        assert!(!report.adjudication_passed);
        assert_eq!(report.metrics.false_invalidations, 1);
        assert_eq!(report.metrics.unrelated_evidence_not_retained, 1);
    }

    #[test]
    fn rejects_cross_run_receipts() {
        let (catalog, oracle, receipts) = fixture_inputs();
        let mut receipt_value: Value =
            serde_json::from_slice(&receipts).expect("receipt fixture parses");
        receipt_value["run_id"] = json!("different-run");
        let receipts = serde_json::to_vec(&receipt_value).expect("mutated receipt serializes");
        let report = audit_bytes(&catalog, &oracle, &receipts);
        assert!(!report.adjudication_passed);
        assert!(report.errors.iter().any(|error| {
            error.code == "E_RUN_ID_MISMATCH"
        }));
    }

    #[test]
    fn rejects_cross_catalog_receipts() {
        let (catalog, oracle, receipts) = fixture_inputs();
        let mut receipt_value: Value =
            serde_json::from_slice(&receipts).expect("receipt fixture parses");
        receipt_value["catalog_digest"] = json!(format!("sha256:{}", "f".repeat(64)));
        let receipts = serde_json::to_vec(&receipt_value).expect("mutated receipt serializes");
        let report = audit_bytes(&catalog, &oracle, &receipts);
        assert!(!report.adjudication_passed);
        assert!(report.errors.iter().any(|error| {
            error.code == "E_CATALOG_DIGEST_MISMATCH"
        }));
    }

    #[test]
    fn rejects_incomplete_receipt_coverage() {
        let (catalog, oracle, receipts) = fixture_inputs();
        let mut receipt_value: Value =
            serde_json::from_slice(&receipts).expect("receipt fixture parses");
        receipt_value["records"].as_array_mut().expect("records array").remove(0);
        let receipts = serde_json::to_vec(&receipt_value).expect("mutated receipt serializes");
        let report = audit_bytes(&catalog, &oracle, &receipts);
        assert!(!report.adjudication_complete);
        assert!(!report.adjudication_passed);
        assert!(report.errors.iter().any(|error| {
            error.code == "E_RECEIPT_CASE_MISSING"
        }));
    }
}
