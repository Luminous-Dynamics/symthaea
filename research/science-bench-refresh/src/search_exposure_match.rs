//! Post-run evaluator-side classification of retrieval exposure.
//!
//! This validates full event coverage and report integrity against a verified
//! receipt. It does not implement semantic search matching itself: the evaluator
//! matcher must be independently qualified, version-pinned, and kept outside the
//! policy-visible process.

use crate::{
    search_exposure::{
        SEARCH_RECEIPT_SCHEMA, SearchExposureReceiptV1, verify_search_exposure_receipt,
    },
    sha256_hex,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const MATCH_REPORT_SCHEMA: &str =
    "symthaea.science_bench_refresh_search_exposure_match_report.v1";
pub const MATCH_VERIFICATION_REPORT_SCHEMA: &str =
    "symthaea.science_bench_refresh_search_exposure_match_verification.v1";

const HASH_PREFIX: &[u8] = b"symthaea.science_bench_refresh.search_match.v1\0";
const MAX_MATCH_RECORDS: usize = 250_000;
const CLASSIFICATIONS: &[&str] = &[
    "no_match",
    "benign_related",
    "benchmark_metadata",
    "question_context",
    "explicit_answer",
    "review_required",
    "not_evaluable",
];
const MATCH_METHODS: &[&str] = &[
    "none",
    "exact_digest",
    "lexical_overlap",
    "semantic_model",
    "manual_review",
];
const MAX_SCORE_MICROS: u32 = 1_000_000;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchExposureMatchReportV1 {
    pub schema: String,
    pub run_id: String,
    pub generation_id: String,
    pub receipt_sha256: String,
    pub receipt_commitment_sha256: String,
    pub matcher_revision_sha256: String,
    pub decision_rules_sha256: String,
    pub expected_event_count: u64,
    pub records: Vec<SearchExposureMatchRecordV1>,
    pub report_commitment_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchExposureMatchRecordV1 {
    pub event_sha256: String,
    pub classification: String,
    pub match_method: String,
    pub match_score_micros: u32,
    pub decision_threshold_micros: u32,
    pub protected_reference_sha256: Option<String>,
    pub evidence_artifact_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct MatchCommitmentPayloadV1<'a> {
    schema: &'a str,
    run_id: &'a str,
    generation_id: &'a str,
    receipt_sha256: &'a str,
    receipt_commitment_sha256: &'a str,
    matcher_revision_sha256: &'a str,
    decision_rules_sha256: &'a str,
    expected_event_count: u64,
    records: &'a [SearchExposureMatchRecordV1],
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchExposureMatchVerificationReportV1 {
    pub schema: String,
    pub receipt_sha256: String,
    pub match_report_sha256: String,
    pub run_id: String,
    pub generation_id: String,
    pub receipt_integrity_valid: bool,
    pub receipt_policy_compliant: bool,
    pub receipt_commitment_matches: bool,
    pub report_commitment_expected: String,
    pub report_commitment_recomputed: String,
    pub report_commitment_matches: bool,
    pub expected_event_count: u64,
    pub observed_event_count: u64,
    pub event_coverage_valid: bool,
    pub classification_contract_valid: bool,
    pub integrity_valid: bool,
    pub classification_counts: BTreeMap<String, u64>,
    pub contaminated_event_count: u64,
    pub review_required_event_count: u64,
    pub admission: String,
    pub issues: Vec<String>,
}

fn digest_domain_framed(domain: &[u8], payload: &[u8]) -> Result<String, String> {
    let domain_len = u64::try_from(domain.len())
        .map_err(|_| "match hash domain exceeds u64 length".to_owned())?;
    let payload_len = u64::try_from(payload.len())
        .map_err(|_| "match hash payload exceeds u64 length".to_owned())?;
    let mut hasher = Sha256::new();
    hasher.update(HASH_PREFIX);
    hasher.update(domain_len.to_be_bytes());
    hasher.update(domain);
    hasher.update(payload_len.to_be_bytes());
    hasher.update(payload);
    Ok(hex_digest(&hasher.finalize()))
}

fn hex_digest(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn match_commitment(report: &SearchExposureMatchReportV1) -> Result<String, String> {
    let payload = MatchCommitmentPayloadV1 {
        schema: &report.schema,
        run_id: &report.run_id,
        generation_id: &report.generation_id,
        receipt_sha256: &report.receipt_sha256,
        receipt_commitment_sha256: &report.receipt_commitment_sha256,
        matcher_revision_sha256: &report.matcher_revision_sha256,
        decision_rules_sha256: &report.decision_rules_sha256,
        expected_event_count: report.expected_event_count,
        records: &report.records,
    };
    let encoded = serde_json::to_vec(&payload)
        .map_err(|error| format!("cannot serialize match-report commitment payload: {error}"))?;
    digest_domain_framed(b"match-report.v1", &encoded)
}

fn has_protected_evidence(record: &SearchExposureMatchRecordV1) -> bool {
    record
        .protected_reference_sha256
        .as_deref()
        .is_some_and(valid_digest)
        && record
            .evidence_artifact_sha256
            .as_deref()
            .is_some_and(valid_digest)
}

fn validate_record_contract(
    record: &SearchExposureMatchRecordV1,
    event: Option<&crate::search_exposure::SearchExposureEventV1>,
    index: usize,
    issues: &mut Vec<String>,
) -> bool {
    let mut valid = true;
    if !valid_digest(&record.event_sha256) {
        issues.push(format!("records[{index}].event_sha256 is not a valid digest"));
        valid = false;
    }
    if !CLASSIFICATIONS.contains(&record.classification.as_str()) {
        issues.push(format!("records[{index}] has an unsupported classification"));
        return false;
    }
    if !MATCH_METHODS.contains(&record.match_method.as_str()) {
        issues.push(format!("records[{index}] has an unsupported match_method"));
        return false;
    }
    if record.match_score_micros > MAX_SCORE_MICROS
        || record.decision_threshold_micros > MAX_SCORE_MICROS
    {
        issues.push(format!("records[{index}] score/threshold exceeds 1,000,000"));
        valid = false;
    }
    for (label, digest) in [
        ("protected_reference_sha256", record.protected_reference_sha256.as_deref()),
        ("evidence_artifact_sha256", record.evidence_artifact_sha256.as_deref()),
    ] {
        if digest.is_some_and(|value| !valid_digest(value)) {
            issues.push(format!("records[{index}].{label} is not a valid digest"));
            valid = false;
        }
    }

    if let Some(event) = event {
        if record.event_sha256 != event.event_sha256 {
            issues.push(format!(
                "records[{index}] event_sha256 does not match receipt event at the same position"
            ));
            valid = false;
        }
        if event.response_sha256.is_none() && record.classification != "not_evaluable" {
            issues.push(format!(
                "records[{index}] must be not_evaluable because the event has no response digest"
            ));
            valid = false;
        }
        if event.response_sha256.is_some() && record.classification == "not_evaluable" {
            issues.push(format!(
                "records[{index}] cannot be not_evaluable when the receipt has a response digest"
            ));
            valid = false;
        }
    }

    match record.classification.as_str() {
        "no_match" => {
            if record.match_method != "none"
                || record.match_score_micros != 0
                || record.protected_reference_sha256.is_some()
                || record.evidence_artifact_sha256.is_some()
            {
                issues.push(format!(
                    "records[{index}] no_match requires method none, score zero, and no protected evidence"
                ));
                valid = false;
            }
        }
        "not_evaluable" => {
            if record.match_method != "none"
                || record.match_score_micros != 0
                || record.protected_reference_sha256.is_some()
                || record.evidence_artifact_sha256.is_some()
            {
                issues.push(format!(
                    "records[{index}] not_evaluable requires method none, score zero, and no protected evidence"
                ));
                valid = false;
            }
        }
        "benign_related" | "review_required" => {
            if record.match_method == "none" || !has_protected_evidence(record) {
                issues.push(format!(
                    "records[{index}] {0} requires a method and protected/evidence digests",
                    record.classification
                ));
                valid = false;
            }
            if record.match_method == "exact_digest"
                && record.match_score_micros != MAX_SCORE_MICROS
            {
                issues.push(format!(
                    "records[{index}] exact_digest must carry score 1,000,000"
                ));
                valid = false;
            }
        }
        "benchmark_metadata" | "question_context" | "explicit_answer" => {
            if record.match_method == "none" || !has_protected_evidence(record) {
                issues.push(format!(
                    "records[{index}] confirmed exposure classification requires a method and protected/evidence digests"
                ));
                valid = false;
            }
            if record.match_score_micros < record.decision_threshold_micros {
                issues.push(format!(
                    "records[{index}] exposure score is below its declared threshold"
                ));
                valid = false;
            }
            if record.match_method == "exact_digest"
                && record.match_score_micros != MAX_SCORE_MICROS
            {
                issues.push(format!(
                    "records[{index}] exact_digest must carry score 1,000,000"
                ));
                valid = false;
            }
        }
        _ => {}
    }
    valid
}

/// Verify a post-run evaluator match report against the exact receipt bytes.
///
/// This validates the report's linkage, completeness, commitment, and declared
/// classification contract. It does not independently execute the semantic
/// matcher; matcher and decision-rule revisions must be separately qualified.
pub fn verify_search_exposure_match_report(
    receipt_bytes: &[u8],
    match_report_bytes: &[u8],
) -> Result<SearchExposureMatchVerificationReportV1, String> {
    let receipt_verification = verify_search_exposure_receipt(receipt_bytes)?;
    if !receipt_verification.integrity_valid {
        return Err("search receipt must pass integrity verification before matching".to_owned());
    }
    let receipt: SearchExposureReceiptV1 = serde_json::from_slice(receipt_bytes)
        .map_err(|error| format!("receipt does not match strict v1 shape: {error}"))?;
    if receipt.schema != SEARCH_RECEIPT_SCHEMA {
        return Err(format!("receipt schema must be {SEARCH_RECEIPT_SCHEMA}"));
    }
    let report: SearchExposureMatchReportV1 = serde_json::from_slice(match_report_bytes)
        .map_err(|error| format!("match report does not match strict v1 shape: {error}"))?;
    if report.schema != MATCH_REPORT_SCHEMA {
        return Err(format!("match report schema must be {MATCH_REPORT_SCHEMA}"));
    }
    if report.run_id.trim().is_empty() || report.run_id.len() > 128 {
        return Err("match report run_id must be 1..=128 characters".to_owned());
    }
    if report.generation_id.trim().is_empty() || report.generation_id.len() > 128 {
        return Err("match report generation_id must be 1..=128 characters".to_owned());
    }
    for (label, digest) in [
        ("receipt_sha256", report.receipt_sha256.as_str()),
        ("receipt_commitment_sha256", report.receipt_commitment_sha256.as_str()),
        ("matcher_revision_sha256", report.matcher_revision_sha256.as_str()),
        ("decision_rules_sha256", report.decision_rules_sha256.as_str()),
        ("report_commitment_sha256", report.report_commitment_sha256.as_str()),
    ] {
        if !valid_digest(digest) {
            return Err(format!("{label} must be 64 lowercase hex characters"));
        }
    }
    if report.records.len() > MAX_MATCH_RECORDS {
        return Err(format!("match report exceeds safety limit of {MAX_MATCH_RECORDS} records"));
    }
    if report.expected_event_count > MAX_MATCH_RECORDS as u64 {
        return Err(format!("match expected_event_count exceeds {MAX_MATCH_RECORDS}"));
    }

    let mut issues = Vec::new();
    if report.run_id != receipt.run_id {
        issues.push("match report run_id does not match receipt".to_owned());
    }
    if report.generation_id != receipt.generation_id {
        issues.push("match report generation_id does not match receipt".to_owned());
    }
    let receipt_sha256 = sha256_hex(receipt_bytes);
    if report.receipt_sha256 != receipt_sha256 {
        issues.push("match report receipt_sha256 does not match exact receipt bytes".to_owned());
    }
    if report.receipt_commitment_sha256 != receipt.receipt_commitment_sha256 {
        issues.push("match report does not bind the receipt commitment".to_owned());
    }
    if report.expected_event_count != receipt.expected_event_count {
        issues.push("match report expected_event_count does not match receipt".to_owned());
    }

    let event_coverage_valid = report.records.len() == receipt.events.len()
        && report.expected_event_count == u64::try_from(receipt.events.len()).unwrap_or(u64::MAX);
    if !event_coverage_valid {
        issues.push(format!(
            "match records ({}) do not cover receipt events ({}) and expected denominator",
            report.records.len(),
            receipt.events.len()
        ));
    }

    let mut classification_contract_valid = true;
    let mut classification_counts = BTreeMap::<String, u64>::new();
    let mut contaminated_event_count = 0_u64;
    let mut review_required_event_count = 0_u64;

    for (index, record) in report.records.iter().enumerate() {
        let event = receipt.events.get(index);
        if !validate_record_contract(record, event, index, &mut issues) {
            classification_contract_valid = false;
        }
        let count = classification_counts
            .entry(record.classification.clone())
            .or_default();
        *count = count
            .checked_add(1)
            .ok_or_else(|| "classification counter overflows u64".to_owned())?;
        if matches!(
            record.classification.as_str(),
            "benchmark_metadata" | "question_context" | "explicit_answer"
        ) {
            contaminated_event_count = contaminated_event_count
                .checked_add(1)
                .ok_or_else(|| "contaminated event count overflows u64".to_owned())?;
        }
        if record.classification == "review_required" {
            review_required_event_count = review_required_event_count
                .checked_add(1)
                .ok_or_else(|| "review-required event count overflows u64".to_owned())?;
        }
    }

    let report_commitment_recomputed = match_commitment(&report)?;
    let report_commitment_matches =
        report_commitment_recomputed == report.report_commitment_sha256;
    if !report_commitment_matches {
        issues.push("match report commitment does not match contents".to_owned());
    }

    let receipt_commitment_matches =
        report.receipt_commitment_sha256 == receipt.receipt_commitment_sha256;
    let integrity_valid = receipt_commitment_matches
        && report_commitment_matches
        && report.run_id == receipt.run_id
        && report.generation_id == receipt.generation_id
        && report.receipt_sha256 == receipt_sha256
        && report.expected_event_count == receipt.expected_event_count
        && event_coverage_valid
        && classification_contract_valid;

    let count = |classification: &str| {
        classification_counts
            .get(classification)
            .copied()
            .unwrap_or(0)
    };
    let admission = if !integrity_valid {
        "match_report_integrity_or_coverage_invalid"
    } else if !receipt_verification.policy_compliant {
        "blocked_receipt_network_policy_violation"
    } else if count("explicit_answer") > 0 {
        "blocked_explicit_answer_exposure"
    } else if count("question_context") > 0 {
        "blocked_question_context_exposure"
    } else if count("benchmark_metadata") > 0 {
        "blocked_benchmark_metadata_exposure"
    } else if review_required_event_count > 0 {
        "manual_review_required"
    } else {
        "no_detected_retrieval_contamination_under_declared_matcher"
    };

    Ok(SearchExposureMatchVerificationReportV1 {
        schema: MATCH_VERIFICATION_REPORT_SCHEMA.to_owned(),
        receipt_sha256,
        match_report_sha256: sha256_hex(match_report_bytes),
        run_id: receipt.run_id,
        generation_id: receipt.generation_id,
        receipt_integrity_valid: receipt_verification.integrity_valid,
        receipt_policy_compliant: receipt_verification.policy_compliant,
        receipt_commitment_matches,
        report_commitment_expected: report.report_commitment_sha256,
        report_commitment_recomputed,
        report_commitment_matches,
        expected_event_count: report.expected_event_count,
        observed_event_count: u64::try_from(report.records.len())
            .map_err(|_| "match record count exceeds u64".to_owned())?,
        event_coverage_valid,
        classification_contract_valid,
        integrity_valid,
        classification_counts,
        contaminated_event_count,
        review_required_event_count,
        admission: admission.to_owned(),
        issues,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search_exposure::{
        SEARCH_RECEIPT_SCHEMA, SearchExposureEventV1, event_hash, receipt_hash,
    };

    const ZERO: &str =
        "0000000000000000000000000000000000000000000000000000000000000000";

    fn sealed_receipt() -> SearchExposureReceiptV1 {
        let mut event = SearchExposureEventV1 {
            sequence: 0,
            event_kind: "network_request".to_owned(),
            decision: "block".to_owned(),
            request_sha256: sha256_hex(b"query"),
            host: "search.example.org".to_owned(),
            response_sha256: None,
            previous_event_sha256: ZERO.to_owned(),
            event_sha256: ZERO.to_owned(),
        };
        event.event_sha256 = event_hash(&event).unwrap();
        let mut receipt = SearchExposureReceiptV1 {
            schema: SEARCH_RECEIPT_SCHEMA.to_owned(),
            run_id: "match-test-run".to_owned(),
            generation_id: "match-test-generation".to_owned(),
            policy_revision_sha256: sha256_hex(b"policy"),
            task_ref_sha256: sha256_hex(b"task"),
            network_policy: "isolated".to_owned(),
            allowed_hosts: vec![],
            expected_event_count: 1,
            final_event_sha256: event.event_sha256.clone(),
            events: vec![event],
            receipt_commitment_sha256: ZERO.to_owned(),
        };
        receipt.receipt_commitment_sha256 = receipt_hash(&receipt).unwrap();
        receipt
    }

    fn match_report(
        receipt_bytes: &[u8],
        classification: &str,
        method: &str,
        score: u32,
        threshold: u32,
        protected: Option<String>,
        evidence: Option<String>,
    ) -> SearchExposureMatchReportV1 {
        let receipt: SearchExposureReceiptV1 = serde_json::from_slice(receipt_bytes).unwrap();
        let mut report = SearchExposureMatchReportV1 {
            schema: MATCH_REPORT_SCHEMA.to_owned(),
            run_id: receipt.run_id,
            generation_id: receipt.generation_id,
            receipt_sha256: sha256_hex(receipt_bytes),
            receipt_commitment_sha256: receipt.receipt_commitment_sha256,
            matcher_revision_sha256: sha256_hex(b"matcher"),
            decision_rules_sha256: sha256_hex(b"frozen-decisions"),
            expected_event_count: 1,
            records: vec![SearchExposureMatchRecordV1 {
                event_sha256: receipt.events[0].event_sha256.clone(),
                classification: classification.to_owned(),
                match_method: method.to_owned(),
                match_score_micros: score,
                decision_threshold_micros: threshold,
                protected_reference_sha256: protected,
                evidence_artifact_sha256: evidence,
            }],
            report_commitment_sha256: ZERO.to_owned(),
        };
        report.report_commitment_sha256 = match_commitment(&report).unwrap();
        report
    }

    #[test]
    fn complete_no_match_report_passes_with_full_event_coverage() {
        let receipt = sealed_receipt();
        let receipt_bytes = serde_json::to_vec(&receipt).unwrap();
        let report = match_report(
            &receipt_bytes,
            "not_evaluable",
            "none",
            0,
            0,
            None,
            None,
        );
        let report_bytes = serde_json::to_vec(&report).unwrap();
        let verified =
            verify_search_exposure_match_report(&receipt_bytes, &report_bytes).unwrap();
        assert!(verified.integrity_valid);
        assert_eq!(
            verified.admission,
            "no_detected_retrieval_contamination_under_declared_matcher"
        );
    }

    #[test]
    fn explicit_answer_match_blocks_and_records_evidence_digests() {
        let receipt = sealed_receipt();
        let receipt_bytes = serde_json::to_vec(&receipt).unwrap();
        let report = match_report(
            &receipt_bytes,
            "explicit_answer",
            "exact_digest",
            MAX_SCORE_MICROS,
            MAX_SCORE_MICROS,
            Some(sha256_hex(b"protected-answer")),
            Some(sha256_hex(b"evidence-snippet")),
        );
        let verified = verify_search_exposure_match_report(
            &receipt_bytes,
            &serde_json::to_vec(&report).unwrap(),
        )
        .unwrap();
        assert!(verified.integrity_valid);
        assert_eq!(verified.contaminated_event_count, 1);
        assert_eq!(verified.admission, "blocked_explicit_answer_exposure");
    }

    #[test]
    fn missing_records_and_mutated_match_report_fail_closed() {
        let receipt = sealed_receipt();
        let receipt_bytes = serde_json::to_vec(&receipt).unwrap();
        let report = match_report(
            &receipt_bytes,
            "not_evaluable",
            "none",
            0,
            0,
            None,
            None,
        );
        let mut missing = report.clone();
        missing.records.clear();
        missing.report_commitment_sha256 = match_commitment(&missing).unwrap();
        let verified = verify_search_exposure_match_report(
            &receipt_bytes,
            &serde_json::to_vec(&missing).unwrap(),
        )
        .unwrap();
        assert!(!verified.integrity_valid);
        assert!(!verified.event_coverage_valid);

        let mut mutated = report;
        mutated.records[0].classification = "explicit_answer".to_owned();
        let verified = verify_search_exposure_match_report(
            &receipt_bytes,
            &serde_json::to_vec(&mutated).unwrap(),
        )
        .unwrap();
        assert!(!verified.report_commitment_matches);
        assert!(!verified.integrity_valid);
    }

    #[test]
    fn a_match_report_cannot_rebind_to_another_receipt() {
        let receipt = sealed_receipt();
        let receipt_bytes = serde_json::to_vec(&receipt).unwrap();
        let report = match_report(
            &receipt_bytes,
            "not_evaluable",
            "none",
            0,
            0,
            None,
            None,
        );
        let mut altered_receipt = receipt;
        altered_receipt.run_id = "different-run".to_owned();
        let altered_bytes = serde_json::to_vec(&altered_receipt).unwrap();
        let verified = verify_search_exposure_match_report(
            &altered_bytes,
            &serde_json::to_vec(&report).unwrap(),
        );
        assert!(verified.is_err() || !verified.unwrap().integrity_valid);
    }
}
