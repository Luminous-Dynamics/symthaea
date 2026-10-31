//! Deterministic exact-digest matcher for evaluator-protected retrieval artifacts.
//!
//! This is a narrow, auditable baseline: it matches an observed response digest
//! against precommitted artifact digests. It cannot detect substrings, paraphrase,
//! semantic equivalents, or a response whose collector logged the wrong digest.
//! The protected index must be kept evaluator-side and its commitment anchored
//! independently before the trial.

use crate::{
    search_exposure::{SearchExposureReceiptV1, verify_search_exposure_receipt},
    search_exposure_match::{
        MATCH_REPORT_SCHEMA, SearchExposureMatchRecordV1, SearchExposureMatchReportV1,
        SearchExposureMatchVerificationReportV1, match_commitment,
        verify_search_exposure_match_report,
    },
    sha256_hex,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const PROTECTED_INDEX_SCHEMA: &str =
    "symthaea.science_bench_refresh_protected_search_artifact_index.v1";

const HASH_PREFIX: &[u8] = b"symthaea.science_bench_refresh.protected_search_index.v1\0";
const MAX_ARTIFACTS: usize = 250_000;
const MAX_DIGEST: u32 = 1_000_000;
const CLASSIFICATIONS: &[&str] = &["benchmark_metadata", "question_context", "explicit_answer"];

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProtectedSearchArtifactIndexV1 {
    pub schema: String,
    pub generation_id: String,
    pub matcher_revision_sha256: String,
    pub decision_rules_sha256: String,
    pub artifacts: Vec<ProtectedSearchArtifactV1>,
    pub index_commitment_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProtectedSearchArtifactV1 {
    pub reference_id_sha256: String,
    pub classification: String,
    pub content_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
struct ProtectedIndexCommitmentPayloadV1<'a> {
    schema: &'a str,
    generation_id: &'a str,
    matcher_revision_sha256: &'a str,
    decision_rules_sha256: &'a str,
    artifacts: &'a [ProtectedSearchArtifactV1],
}

fn digest_domain_framed(domain: &[u8], payload: &[u8]) -> Result<String, String> {
    let domain_len = u64::try_from(domain.len())
        .map_err(|_| "protected-index hash domain exceeds u64 length".to_owned())?;
    let payload_len = u64::try_from(payload.len())
        .map_err(|_| "protected-index hash payload exceeds u64 length".to_owned())?;
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

fn index_commitment(index: &ProtectedSearchArtifactIndexV1) -> Result<String, String> {
    let payload = ProtectedIndexCommitmentPayloadV1 {
        schema: &index.schema,
        generation_id: &index.generation_id,
        matcher_revision_sha256: &index.matcher_revision_sha256,
        decision_rules_sha256: &index.decision_rules_sha256,
        artifacts: &index.artifacts,
    };
    let encoded = serde_json::to_vec(&payload)
        .map_err(|error| format!("cannot serialize protected-index commitment: {error}"))?;
    digest_domain_framed(b"protected-index.v1", &encoded)
}

fn validate_protected_index_bytes(
    bytes: &[u8],
    expected_generation: &str,
) -> Result<ProtectedSearchArtifactIndexV1, String> {
    let index: ProtectedSearchArtifactIndexV1 = serde_json::from_slice(bytes)
        .map_err(|error| format!("protected index does not match strict v1 shape: {error}"))?;
    if index.schema != PROTECTED_INDEX_SCHEMA {
        return Err(format!(
            "protected index schema must be {PROTECTED_INDEX_SCHEMA}"
        ));
    }
    if index.generation_id.trim().is_empty() || index.generation_id.len() > 128 {
        return Err("protected index generation_id must be 1..=128 characters".to_owned());
    }
    if index.generation_id != expected_generation {
        return Err("protected index generation_id does not match receipt".to_owned());
    }
    for (field, value) in [
        (
            "matcher_revision_sha256",
            index.matcher_revision_sha256.as_str(),
        ),
        (
            "decision_rules_sha256",
            index.decision_rules_sha256.as_str(),
        ),
        (
            "index_commitment_sha256",
            index.index_commitment_sha256.as_str(),
        ),
    ] {
        if !valid_digest(value) {
            return Err(format!("{field} must be 64 lowercase hex characters"));
        }
    }
    if index.artifacts.len() > MAX_ARTIFACTS {
        return Err(format!(
            "protected index exceeds {MAX_ARTIFACTS} artifact records"
        ));
    }

    let mut reference_ids = BTreeSet::new();
    let mut previous_key: Option<(&str, &str, &str)> = None;
    for (position, artifact) in index.artifacts.iter().enumerate() {
        if !valid_digest(&artifact.reference_id_sha256) {
            return Err(format!(
                "artifacts[{position}].reference_id_sha256 must be 64 lowercase hex characters"
            ));
        }
        if !reference_ids.insert(artifact.reference_id_sha256.as_str()) {
            return Err(format!(
                "protected index repeats reference_id_sha256 at artifacts[{position}]"
            ));
        }
        if !CLASSIFICATIONS.contains(&artifact.classification.as_str()) {
            return Err(format!(
                "artifacts[{position}] has unsupported protected classification"
            ));
        }
        if !valid_digest(&artifact.content_sha256) {
            return Err(format!(
                "artifacts[{position}].content_sha256 must be 64 lowercase hex characters"
            ));
        }
        let key = (
            artifact.content_sha256.as_str(),
            artifact.classification.as_str(),
            artifact.reference_id_sha256.as_str(),
        );
        if previous_key.is_some_and(|previous| previous >= key) {
            return Err("protected artifacts are not in canonical order".to_owned());
        }
        previous_key = Some(key);
    }

    let actual_commitment = index_commitment(&index)?;
    if actual_commitment != index.index_commitment_sha256 {
        return Err("protected index commitment does not match index contents".to_owned());
    }
    Ok(index)
}

fn classification_rank(classification: &str) -> u8 {
    match classification {
        "benchmark_metadata" => 1,
        "question_context" => 2,
        "explicit_answer" => 3,
        _ => 0,
    }
}

/// Compute a stable commitment over the index's typed fields (excluding the
/// commitment field itself). The returned commitment still requires an external
/// preregistration/append-only anchor to authenticate its value.
pub fn compute_protected_index_commitment(
    index: &ProtectedSearchArtifactIndexCommitmentInput,
) -> Result<String, String> {
    let typed = ProtectedSearchArtifactIndexV1 {
        schema: PROTECTED_INDEX_SCHEMA.to_owned(),
        generation_id: index.generation_id.clone(),
        matcher_revision_sha256: index.matcher_revision_sha256.clone(),
        decision_rules_sha256: index.decision_rules_sha256.clone(),
        artifacts: index.artifacts.clone(),
        index_commitment_sha256: String::new(),
    };
    index_commitment(&typed)
}

/// Input helper for generating a protected index commitment.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProtectedSearchArtifactIndexCommitmentInput {
    pub generation_id: String,
    pub matcher_revision_sha256: String,
    pub decision_rules_sha256: String,
    pub artifacts: Vec<ProtectedSearchArtifactV1>,
}

/// Classify each receipt event by exact response-digest match against the
/// evaluator-only index. If the same digest appears at several severities, use
/// the most severe classification. Responses with no match remain no_match;
/// blocked/no-response events remain not_evaluable.
pub fn match_search_exposure_exact(
    receipt_bytes: &[u8],
    protected_index_bytes: &[u8],
) -> Result<SearchExposureMatchReportV1, String> {
    let receipt_verification = verify_search_exposure_receipt(receipt_bytes)?;
    if !receipt_verification.integrity_valid {
        return Err("search receipt must pass integrity verification before matching".to_owned());
    }
    let receipt: SearchExposureReceiptV1 = serde_json::from_slice(receipt_bytes)
        .map_err(|error| format!("receipt does not match strict v1 shape: {error}"))?;
    let index = validate_protected_index_bytes(protected_index_bytes, &receipt.generation_id)?;

    let mut protected_by_digest = BTreeMap::<String, (String, String)>::new();
    for artifact in &index.artifacts {
        protected_by_digest
            .entry(artifact.content_sha256.clone())
            .and_modify(|current| {
                let new_rank = classification_rank(&artifact.classification);
                let current_rank = classification_rank(&current.0);
                if new_rank > current_rank
                    || (new_rank == current_rank && artifact.reference_id_sha256 < current.1)
                {
                    *current = (
                        artifact.classification.clone(),
                        artifact.reference_id_sha256.clone(),
                    );
                }
            })
            .or_insert_with(|| {
                (
                    artifact.classification.clone(),
                    artifact.reference_id_sha256.clone(),
                )
            });
    }

    let mut records = Vec::with_capacity(receipt.events.len());
    for event in &receipt.events {
        let record = match event.response_sha256.as_deref() {
            None => SearchExposureMatchRecordV1 {
                event_sha256: event.event_sha256.clone(),
                classification: "not_evaluable".to_owned(),
                match_method: "none".to_owned(),
                match_score_micros: 0,
                decision_threshold_micros: 0,
                protected_reference_sha256: None,
                evidence_artifact_sha256: None,
            },
            Some(response_digest) => match protected_by_digest.get(response_digest) {
                Some((classification, reference_id_sha256)) => SearchExposureMatchRecordV1 {
                    event_sha256: event.event_sha256.clone(),
                    classification: classification.clone(),
                    match_method: "exact_digest".to_owned(),
                    match_score_micros: MAX_DIGEST,
                    decision_threshold_micros: MAX_DIGEST,
                    // Report only the opaque reference ID digest, not the protected
                    // content digest, which could enable dictionary attacks.
                    protected_reference_sha256: Some(reference_id_sha256.clone()),
                    evidence_artifact_sha256: Some(sha256_hex(
                        format!(
                            "symthaea.search-evidence.v1\0{}\0{}",
                            event.event_sha256, response_digest
                        )
                        .as_bytes(),
                    )),
                },
                None => SearchExposureMatchRecordV1 {
                    event_sha256: event.event_sha256.clone(),
                    classification: "no_match".to_owned(),
                    match_method: "none".to_owned(),
                    match_score_micros: 0,
                    decision_threshold_micros: MAX_DIGEST,
                    protected_reference_sha256: None,
                    evidence_artifact_sha256: None,
                },
            },
        };
        records.push(record);
    }

    let mut report = SearchExposureMatchReportV1 {
        schema: MATCH_REPORT_SCHEMA.to_owned(),
        run_id: receipt.run_id,
        generation_id: receipt.generation_id,
        receipt_sha256: sha256_hex(receipt_bytes),
        receipt_commitment_sha256: receipt.receipt_commitment_sha256,
        protected_index_sha256: sha256_hex(protected_index_bytes),
        matcher_revision_sha256: index.matcher_revision_sha256,
        decision_rules_sha256: index.decision_rules_sha256,
        expected_event_count: receipt.expected_event_count,
        records,
        report_commitment_sha256: String::new(),
    };
    report.report_commitment_sha256 = match_commitment(&report)?;
    Ok(report)
}

/// Verify a report against the exact receipt and protected-index bytes. This
/// recomputes the deterministic exact-digest classifications rather than merely
/// checking that a caller supplied internally plausible labels.
pub fn verify_search_exposure_exact_match_report(
    receipt_bytes: &[u8],
    protected_index_bytes: &[u8],
    match_report_bytes: &[u8],
) -> Result<SearchExposureMatchVerificationReportV1, String> {
    let expected = match_search_exposure_exact(receipt_bytes, protected_index_bytes)?;
    let supplied: SearchExposureMatchReportV1 = serde_json::from_slice(match_report_bytes)
        .map_err(|error| format!("match report does not match strict v1 shape: {error}"))?;

    let expected_value: Value = serde_json::to_value(&expected)
        .map_err(|error| format!("cannot serialize expected match report: {error}"))?;
    let supplied_value: Value = serde_json::to_value(&supplied)
        .map_err(|error| format!("cannot serialize supplied match report: {error}"))?;
    if expected_value != supplied_value {
        return Err("match report differs from exact matcher output".to_owned());
    }
    if supplied.protected_index_sha256 != sha256_hex(protected_index_bytes) {
        return Err("match report does not bind exact protected index bytes".to_owned());
    }
    verify_search_exposure_match_report(receipt_bytes, match_report_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search_exposure::{
        SEARCH_RECEIPT_SCHEMA, SearchExposureEventV1, event_hash, receipt_hash,
    };

    const ZERO: &str = "0000000000000000000000000000000000000000000000000000000000000000";

    fn receipt_with_response(content: Option<&[u8]>) -> SearchExposureReceiptV1 {
        let response_sha = content.map(sha256_hex);
        let mut event = SearchExposureEventV1 {
            sequence: 0,
            event_kind: if content.is_some() {
                "fetch"
            } else {
                "network_request"
            }
            .to_owned(),
            decision: if content.is_some() { "allow" } else { "block" }.to_owned(),
            request_sha256: sha256_hex(b"opaque-request"),
            host: "search.example.org".to_owned(),
            response_sha256: response_sha,
            previous_event_sha256: ZERO.to_owned(),
            event_sha256: ZERO.to_owned(),
        };
        event.event_sha256 = event_hash(&event).unwrap();
        let mut receipt = SearchExposureReceiptV1 {
            schema: SEARCH_RECEIPT_SCHEMA.to_owned(),
            run_id: "exact-matcher-test-run".to_owned(),
            generation_id: "exact-matcher-test-generation".to_owned(),
            policy_revision_sha256: sha256_hex(b"policy-revision"),
            task_ref_sha256: sha256_hex(b"opaque-task-ref"),
            network_policy: if content.is_some() {
                "audited_open_web"
            } else {
                "isolated"
            }
            .to_owned(),
            allowed_hosts: vec![],
            expected_event_count: 1,
            final_event_sha256: event.event_sha256.clone(),
            events: vec![event],
            receipt_commitment_sha256: ZERO.to_owned(),
        };
        receipt.receipt_commitment_sha256 = receipt_hash(&receipt).unwrap();
        receipt
    }

    fn protected_index(
        generation: &str,
        artifacts: Vec<ProtectedSearchArtifactV1>,
    ) -> ProtectedSearchArtifactIndexV1 {
        let mut index = ProtectedSearchArtifactIndexV1 {
            schema: PROTECTED_INDEX_SCHEMA.to_owned(),
            generation_id: generation.to_owned(),
            matcher_revision_sha256: sha256_hex(b"exact matcher revision v1"),
            decision_rules_sha256: sha256_hex(b"exact-match-thresholds-v1"),
            artifacts,
            index_commitment_sha256: ZERO.to_owned(),
        };
        index.artifacts.sort_by(|left, right| {
            (
                &left.content_sha256,
                &left.classification,
                &left.reference_id_sha256,
            )
                .cmp(&(
                    &right.content_sha256,
                    &right.classification,
                    &right.reference_id_sha256,
                ))
        });
        index.index_commitment_sha256 = index_commitment(&index).unwrap();
        index
    }

    fn artifact(
        reference: &str,
        classification: &str,
        content: &[u8],
    ) -> ProtectedSearchArtifactV1 {
        ProtectedSearchArtifactV1 {
            reference_id_sha256: sha256_hex(reference.as_bytes()),
            classification: classification.to_owned(),
            content_sha256: sha256_hex(content),
        }
    }

    #[test]
    fn exact_answer_is_detected_and_overrides_lower_severity_duplicates() {
        let body = b"exact protected answer bytes";
        let receipt = receipt_with_response(Some(body));
        let receipt_bytes = serde_json::to_vec(&receipt).unwrap();
        let index = protected_index(
            &receipt.generation_id,
            vec![
                artifact("meta-ref", "benchmark_metadata", body),
                artifact("question-ref", "question_context", body),
                artifact("answer-ref", "explicit_answer", body),
            ],
        );
        let index_bytes = serde_json::to_vec(&index).unwrap();
        let report = match_search_exposure_exact(&receipt_bytes, &index_bytes).unwrap();
        assert_eq!(report.records[0].classification, "explicit_answer");
        assert_eq!(report.records[0].match_method, "exact_digest");
        assert_eq!(report.protected_index_sha256, sha256_hex(&index_bytes));
        assert_eq!(
            report.records[0].protected_reference_sha256,
            Some(sha256_hex(b"answer-ref"))
        );
        assert_ne!(
            report.records[0].protected_reference_sha256,
            report.records[0].evidence_artifact_sha256
        );
        assert_ne!(
            report.records[0].protected_reference_sha256,
            Some(sha256_hex(body))
        );
    }

    #[test]
    fn metadata_and_question_context_are_classified_separately() {
        let meta_body = b"metadata exact bytes";
        let meta_receipt = receipt_with_response(Some(meta_body));
        let meta_index = protected_index(
            &meta_receipt.generation_id,
            vec![artifact("meta", "benchmark_metadata", meta_body)],
        );
        let meta_report = match_search_exposure_exact(
            &serde_json::to_vec(&meta_receipt).unwrap(),
            &serde_json::to_vec(&meta_index).unwrap(),
        )
        .unwrap();
        assert_eq!(meta_report.records[0].classification, "benchmark_metadata");

        let context_body = b"context exact bytes";
        let context_receipt = receipt_with_response(Some(context_body));
        let context_index = protected_index(
            &context_receipt.generation_id,
            vec![artifact("context", "question_context", context_body)],
        );
        let context_report = match_search_exposure_exact(
            &serde_json::to_vec(&context_receipt).unwrap(),
            &serde_json::to_vec(&context_index).unwrap(),
        )
        .unwrap();
        assert_eq!(context_report.records[0].classification, "question_context");
    }

    #[test]
    fn unrelated_content_is_no_match_and_blocked_events_are_not_evaluable() {
        let receipt = receipt_with_response(Some(b"unrelated response"));
        let index = protected_index(
            &receipt.generation_id,
            vec![artifact("answer", "explicit_answer", b"answer bytes")],
        );
        let report = match_search_exposure_exact(
            &serde_json::to_vec(&receipt).unwrap(),
            &serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();
        assert_eq!(report.records[0].classification, "no_match");

        let blocked = receipt_with_response(None);
        let blocked_index = protected_index(&blocked.generation_id, vec![]);
        let report = match_search_exposure_exact(
            &serde_json::to_vec(&blocked).unwrap(),
            &serde_json::to_vec(&blocked_index).unwrap(),
        )
        .unwrap();
        assert_eq!(report.records[0].classification, "not_evaluable");
    }

    #[test]
    fn altered_or_unanchored_protected_index_is_rejected() {
        let body = b"protected answer";
        let receipt = receipt_with_response(Some(body));
        let mut index = protected_index(
            &receipt.generation_id,
            vec![artifact("answer", "explicit_answer", body)],
        );
        index.artifacts.clear();
        let receipt_bytes = serde_json::to_vec(&receipt).unwrap();
        assert!(
            match_search_exposure_exact(&receipt_bytes, &serde_json::to_vec(&index).unwrap(),)
                .is_err()
        );

        let mut wrong_generation = protected_index(
            "other-generation",
            vec![artifact("answer", "explicit_answer", body)],
        );
        wrong_generation.index_commitment_sha256 = index_commitment(&wrong_generation).unwrap();
        assert!(
            match_search_exposure_exact(
                &receipt_bytes,
                &serde_json::to_vec(&wrong_generation).unwrap(),
            )
            .is_err()
        );
    }

    #[test]
    fn generated_report_is_recomputed_when_verified() {
        let body = b"answer bytes";
        let receipt = receipt_with_response(Some(body));
        let receipt_bytes = serde_json::to_vec(&receipt).unwrap();
        let index = protected_index(
            &receipt.generation_id,
            vec![artifact("answer", "explicit_answer", body)],
        );
        let index_bytes = serde_json::to_vec(&index).unwrap();
        let report = match_search_exposure_exact(&receipt_bytes, &index_bytes).unwrap();
        let report_bytes = serde_json::to_vec(&report).unwrap();
        let verified =
            verify_search_exposure_exact_match_report(&receipt_bytes, &index_bytes, &report_bytes)
                .unwrap();
        assert_eq!(verified.contaminated_event_count, 1);

        let mut altered = report;
        altered.records[0].classification = "no_match".to_owned();
        assert!(
            verify_search_exposure_exact_match_report(
                &receipt_bytes,
                &index_bytes,
                &serde_json::to_vec(&altered).unwrap()
            )
            .is_err()
        );
    }
}
