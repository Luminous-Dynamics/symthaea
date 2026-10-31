//! Evaluator-side evidence for conclusions inferable only by composing
//! multiple individually insufficient information-access events.
//!
//! This module records and audits minimal supporting event sets. It does not
//! inspect raw tool-return content or independently validate semantic claims.
//! All content matching and insufficiency/sufficiency evidence remains in an
//! evaluator-controlled evidence store.

use crate::{
    hash_domain,
    search_time::{
        audit_search_time_trace, SearchEventDispositionV1, SearchTimeExposureTraceV1,
        SearchTimeEventV1,
    },
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const COMPOSITION_FINDINGS_SCHEMA: &str =
    "symthaea.compositional_exposure_findings.v1";
pub const COMPOSITION_AUDIT_SCHEMA: &str =
    "symthaea.compositional_exposure_audit.v1";
pub const COMPOSITION_MAX_FINDINGS: usize = 100_000;
pub const COMPOSITION_MAX_EVENTS_PER_FINDING: usize = 256;
pub const COMPOSITION_MAX_ITEM_REFERENCES: usize = 100_000;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CompositionalExposureClassV1 {
    ContextBearing,
    AnswerBearing,
    Unknown,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CompositionalEvidenceMethodV1 {
    ExactMatch,
    FuzzyText,
    SemanticReview,
    HumanAdjudication,
    Other,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SingleEventInsufficiencyV1 {
    pub event_sequence: u64,
    pub evidence_sha256: String,
    pub method: CompositionalEvidenceMethodV1,
    pub confidence_permille: u16,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LeaveOneOutInsufficiencyV1 {
    pub omitted_event_sequence: u64,
    /// Canonically sorted supporting set after removing omitted_event_sequence.
    pub tested_event_sequences: Vec<u64>,
    pub tested_subset_sha256: String,
    pub evidence_sha256: String,
    pub method: CompositionalEvidenceMethodV1,
    pub confidence_permille: u16,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JointSufficiencyEvidenceV1 {
    pub evidence_sha256: String,
    pub method: CompositionalEvidenceMethodV1,
    pub confidence_permille: u16,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CompositionalExposureFindingV1 {
    pub finding_id: String,
    pub exposure_class: CompositionalExposureClassV1,
    /// Commitment only. The protected conclusion itself never leaves evaluator custody.
    pub protected_conclusion_commitment_sha256: String,
    pub matched_item_ids: Vec<String>,
    /// Canonically sorted event sequences forming the proposed support set.
    pub supporting_event_sequences: Vec<u64>,
    /// One evaluator attestation per supporting event that it is insufficient alone.
    pub single_event_insufficiency: Vec<SingleEventInsufficiencyV1>,
    /// One evaluator attestation per source event that the remaining set is insufficient.
    pub leave_one_out_insufficiency: Vec<LeaveOneOutInsufficiencyV1>,
    pub joint_sufficiency: JointSufficiencyEvidenceV1,
    /// Minimality is certified only for monotone inference predicates with evidence.
    pub monotonicity_assumed: bool,
    pub monotonicity_evidence_sha256: Option<String>,
    pub confidence_permille: u16,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CompositionalExposureFindingsV1 {
    pub schema: String,
    pub run_id: String,
    pub trace_root_sha256: String,
    pub findings: Vec<CompositionalExposureFindingV1>,
    pub findings_sha256: String,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CompositionalExposureAuditV1 {
    pub schema: String,
    pub valid: bool,
    pub trace_integrity_replayed: bool,
    pub findings_digest_matches: bool,
    pub every_source_event_was_individually_insufficient_attested: bool,
    pub every_leave_one_out_set_was_insufficient_attested: bool,
    pub every_joint_set_was_sufficient_attested: bool,
    pub minimality_certified_only_under_monotonicity: bool,
    pub finding_count: u64,
    pub context_bearing_count: u64,
    pub answer_bearing_count: u64,
    pub unknown_count: u64,
    pub minimality_certified_count: u64,
    pub single_source_attestation_count: u64,
    pub leave_one_out_attestation_count: u64,
    pub matched_item_reference_count: u64,
    pub run_id: String,
    pub trace_root_sha256: String,
    pub findings_sha256: String,
    pub independently_verified: bool,
    pub semantic_evidence_independently_verified: bool,
    pub claim_boundary: String,
    pub report_sha256: String,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_digest(value: &str, label: &str) -> Result<(), String> {
    if valid_digest(value) {
        Ok(())
    } else {
        Err(format!("{label} must be lowercase SHA-256 hex"))
    }
}

fn sorted_unique(values: &[u64]) -> bool {
    values.len() >= 2
        && values.windows(2).all(|pair| pair[0] < pair[1])
}

pub fn canonical_event_subset_digest(event_sequences: &[u64]) -> Result<String, String> {
    if event_sequences.is_empty()
        || event_sequences.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err("event subset must be nonempty, sorted and unique".to_owned());
    }
    let bytes = serde_json::to_vec(event_sequences)
        .map_err(|error| format!("cannot serialize event subset: {error}"))?;
    Ok(hash_domain(b"compositional-event-subset.v1", &bytes))
}

pub fn compositional_findings_digest(
    findings: &CompositionalExposureFindingsV1,
) -> Result<String, String> {
    let mut normalized = findings.clone();
    normalized.findings_sha256.clear();
    let bytes = serde_json::to_vec(&normalized)
        .map_err(|error| format!("cannot serialize compositional findings: {error}"))?;
    Ok(hash_domain(b"compositional-exposure-findings.v1", &bytes))
}

fn audit_report_digest(report: &CompositionalExposureAuditV1) -> Result<String, String> {
    let mut normalized = report.clone();
    normalized.report_sha256.clear();
    let bytes = serde_json::to_vec(&normalized)
        .map_err(|error| format!("cannot serialize compositional audit report: {error}"))?;
    Ok(hash_domain(b"compositional-exposure-audit.v1", &bytes))
}

fn validate_event_reference(
    event_by_sequence: &BTreeMap<u64, &SearchTimeEventV1>,
    sequence: u64,
) -> Result<(), String> {
    let event = event_by_sequence
        .get(&sequence)
        .ok_or_else(|| format!("composition references unknown trace event {sequence}"))?;
    if event.disposition != SearchEventDispositionV1::Allowed {
        return Err(format!("composition references blocked trace event {sequence}"));
    }
    if event.response_bytes == 0 || event.response_sha256.is_none() {
        return Err(format!("composition event {sequence} has no received response body"));
    }
    Ok(())
}

fn validate_finding(
    finding: &CompositionalExposureFindingV1,
    event_by_sequence: &BTreeMap<u64, &SearchTimeEventV1>,
) -> Result<bool, String> {
    if finding.finding_id.trim().is_empty() || finding.finding_id.len() > 256 {
        return Err("compositional finding ID must be nonempty and at most 256 bytes".to_owned());
    }
    validate_digest(
        &finding.protected_conclusion_commitment_sha256,
        "protected_conclusion_commitment_sha256",
    )?;
    if finding.matched_item_ids.is_empty()
        || finding.matched_item_ids.len() > COMPOSITION_MAX_ITEM_REFERENCES
    {
        return Err("compositional finding must reference between 1 and 100000 benchmark items".to_owned());
    }
    let item_ids: BTreeSet<&str> = finding.matched_item_ids.iter().map(String::as_str).collect();
    if item_ids.len() != finding.matched_item_ids.len()
        || item_ids.iter().any(|item| item.trim().is_empty())
    {
        return Err("matched benchmark item IDs must be unique and nonempty".to_owned());
    }
    if finding.supporting_event_sequences.len() < 2
        || finding.supporting_event_sequences.len() > COMPOSITION_MAX_EVENTS_PER_FINDING
        || !sorted_unique(&finding.supporting_event_sequences)
    {
        return Err("supporting event set must contain 2..=256 sorted unique sequences".to_owned());
    }
    for sequence in &finding.supporting_event_sequences {
        validate_event_reference(event_by_sequence, *sequence)?;
    }
    validate_digest(
        &finding.joint_sufficiency.evidence_sha256,
        "joint_sufficiency.evidence_sha256",
    )?;
    if finding.joint_sufficiency.confidence_permille > 1_000
        || finding.confidence_permille > 1_000
    {
        return Err("compositional evidence confidence must be in [0, 1000]".to_owned());
    }

    let mut single_by_sequence = BTreeMap::new();
    for evidence in &finding.single_event_insufficiency {
        validate_digest(&evidence.evidence_sha256, "single_event_insufficiency.evidence_sha256")?;
        if evidence.confidence_permille > 1_000
            || single_by_sequence.insert(evidence.event_sequence, evidence).is_some()
        {
            return Err("single-event insufficiency evidence must be unique with valid confidence".to_owned());
        }
    }
    let support: BTreeSet<u64> = finding.supporting_event_sequences.iter().copied().collect();
    if single_by_sequence.keys().copied().collect::<BTreeSet<_>>() != support {
        return Err("each supporting event must have exactly one single-event insufficiency attestation".to_owned());
    }

    let mut leave_by_omitted = BTreeMap::new();
    for evidence in &finding.leave_one_out_insufficiency {
        validate_digest(&evidence.evidence_sha256, "leave_one_out.evidence_sha256")?;
        validate_digest(&evidence.tested_subset_sha256, "leave_one_out.tested_subset_sha256")?;
        if evidence.confidence_permille > 1_000
            || leave_by_omitted.insert(evidence.omitted_event_sequence, evidence).is_some()
        {
            return Err("leave-one-out evidence must be unique with valid confidence".to_owned());
        }
        let expected_subset: Vec<u64> = finding.supporting_event_sequences.iter()
            .copied()
            .filter(|sequence| *sequence != evidence.omitted_event_sequence)
            .collect();
        if expected_subset.len() + 1 != finding.supporting_event_sequences.len()
            || evidence.tested_event_sequences != expected_subset
            || canonical_event_subset_digest(&expected_subset)? != evidence.tested_subset_sha256
        {
            return Err("leave-one-out evidence does not bind the exact supporting set minus its omitted event".to_owned());
        }
    }
    if leave_by_omitted.keys().copied().collect::<BTreeSet<_>>() != support {
        return Err("each supporting event must have exactly one leave-one-out insufficiency attestation".to_owned());
    }

    match (
        finding.monotonicity_assumed,
        finding.monotonicity_evidence_sha256.as_deref(),
    ) {
        (true, Some(digest)) => validate_digest(digest, "monotonicity_evidence_sha256")?,
        (false, None) => {}
        _ => return Err("monotonicity assumption must be paired with exactly one evidence digest or explicit absence".to_owned()),
    }
    Ok(finding.monotonicity_assumed)
}

/// Audit evaluator-supplied compositional exposure evidence against an exact
/// search-time trace. Semantic truth and executor trust remain external.
pub fn audit_compositional_exposure(
    trace: &SearchTimeExposureTraceV1,
    findings: &CompositionalExposureFindingsV1,
) -> Result<CompositionalExposureAuditV1, String> {
    if findings.schema != COMPOSITION_FINDINGS_SCHEMA {
        return Err(format!("findings schema must be {COMPOSITION_FINDINGS_SCHEMA}"));
    }
    if findings.run_id != trace.run_id || findings.trace_root_sha256 != trace.trace_root_sha256 {
        return Err("compositional findings do not bind to the exact search trace and run".to_owned());
    }
    if findings.findings.len() > COMPOSITION_MAX_FINDINGS {
        return Err(format!("compositional findings exceed {COMPOSITION_MAX_FINDINGS}"));
    }
    let trace_audit = audit_search_time_trace(trace)?;
    if !trace_audit.hash_chain_replayed
        || !trace_audit.trace_root_matches
        || !trace_audit.sequence_complete
        || !trace_audit.timestamps_valid_and_monotonic
    {
        return Err("compositional findings require an integrity-valid search-time trace".to_owned());
    }
    validate_digest(&findings.findings_sha256, "findings_sha256")?;
    let findings_digest_matches = compositional_findings_digest(findings)?
        == findings.findings_sha256;
    if !findings_digest_matches {
        return Err("compositional findings digest does not match contents".to_owned());
    }
    let event_by_sequence: BTreeMap<u64, &SearchTimeEventV1> =
        trace.events.iter().map(|event| (event.sequence, event)).collect();
    let mut finding_ids = BTreeSet::new();
    let mut support_sets = BTreeSet::new();
    let mut context_count = 0_u64;
    let mut answer_count = 0_u64;
    let mut unknown_count = 0_u64;
    let mut minimality_count = 0_u64;
    let mut single_count = 0_u64;
    let mut leave_count = 0_u64;
    let mut item_reference_count = 0_u64;

    for finding in &findings.findings {
        if !finding_ids.insert(finding.finding_id.as_str()) {
            return Err("compositional finding IDs must be unique".to_owned());
        }
        if !support_sets.insert(finding.supporting_event_sequences.clone()) {
            return Err("duplicate minimal supporting event sets are not allowed within one findings artifact".to_owned());
        }
        let minimality_certified = validate_finding(finding, &event_by_sequence)?;
        match finding.exposure_class {
            CompositionalExposureClassV1::ContextBearing => context_count += 1,
            CompositionalExposureClassV1::AnswerBearing => answer_count += 1,
            CompositionalExposureClassV1::Unknown => unknown_count += 1,
        }
        if minimality_certified {
            minimality_count += 1;
        }
        single_count = single_count.checked_add(finding.single_event_insufficiency.len() as u64)
            .ok_or_else(|| "single-event attestation count overflowed".to_owned())?;
        leave_count = leave_count.checked_add(finding.leave_one_out_insufficiency.len() as u64)
            .ok_or_else(|| "leave-one-out attestation count overflowed".to_owned())?;
        item_reference_count = item_reference_count.checked_add(finding.matched_item_ids.len() as u64)
            .ok_or_else(|| "matched item reference count overflowed".to_owned())?;
    }

    let mut report = CompositionalExposureAuditV1 {
        schema: COMPOSITION_AUDIT_SCHEMA.to_owned(),
        valid: true,
        trace_integrity_replayed: true,
        findings_digest_matches,
        every_source_event_was_individually_insufficient_attested: true,
        every_leave_one_out_set_was_insufficient_attested: true,
        every_joint_set_was_sufficient_attested: true,
        minimality_certified_only_under_monotonicity: true,
        finding_count: u64::try_from(findings.findings.len())
            .map_err(|_| "finding count exceeds u64")?,
        context_bearing_count: context_count,
        answer_bearing_count: answer_count,
        unknown_count,
        minimality_certified_count: minimality_count,
        single_source_attestation_count: single_count,
        leave_one_out_attestation_count: leave_count,
        matched_item_reference_count: item_reference_count,
        run_id: trace.run_id.clone(),
        trace_root_sha256: trace.trace_root_sha256.clone(),
        findings_sha256: findings.findings_sha256.clone(),
        independently_verified: false,
        semantic_evidence_independently_verified: false,
        claim_boundary: "audits_evaluator_supplied_compositional_evidence_bindings_and_minimal_set_logic;does_not_verify_semantic_truth_or_model_influence".to_owned(),
        report_sha256: String::new(),
    };
    report.report_sha256 = audit_report_digest(&report)?;
    Ok(report)
}

pub fn verify_compositional_audit_integrity(
    report: &CompositionalExposureAuditV1,
) -> Result<(), String> {
    if report.schema != COMPOSITION_AUDIT_SCHEMA
        || !report.valid
        || !report.trace_integrity_replayed
        || !report.findings_digest_matches
        || !report.every_source_event_was_individually_insufficient_attested
        || !report.every_leave_one_out_set_was_insufficient_attested
        || !report.every_joint_set_was_sufficient_attested
        || !report.minimality_certified_only_under_monotonicity
        || report.independently_verified
        || report.semantic_evidence_independently_verified
    {
        return Err("compositional audit report invariants or trust boundary are invalid".to_owned());
    }
    validate_digest(&report.trace_root_sha256, "trace_root_sha256")?;
    validate_digest(&report.findings_sha256, "findings_sha256")?;
    validate_digest(&report.report_sha256, "report_sha256")?;
    let expected = audit_report_digest(report)?;
    if expected != report.report_sha256 {
        return Err("compositional audit report digest mismatch".to_owned());
    }
    if report.finding_count
        != report.context_bearing_count
            .checked_add(report.answer_bearing_count)
            .and_then(|value| value.checked_add(report.unknown_count))
            .ok_or_else(|| "compositional finding count overflowed".to_owned())?
        || report.minimality_certified_count > report.finding_count
    {
        return Err("compositional audit report summary counts are inconsistent".to_owned());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        hash_domain,
        search_time::{
            seal_search_time_trace, SearchAccessModeV1, SearchEventDispositionV1,
            SearchInformationChannelV1, SearchTimeEventV1, SearchTimeExposureTraceV1,
            SEARCH_TRACE_SCHEMA,
        },
    };

    fn digest(label: &str) -> String {
        hash_domain(b"compositional-exposure-test", label.as_bytes())
    }

    fn event(sequence: u64) -> SearchTimeEventV1 {
        SearchTimeEventV1 {
            sequence,
            invocation_id: format!("tool-call-{sequence}"),
            channel: SearchInformationChannelV1::RemoteMcp,
            provider_or_tool: "synthetic-tool".to_owned(),
            destination_sha256: digest("destination"),
            request_sha256: digest(&format!("request-{sequence}")),
            response_sha256: Some(digest(&format!("response-{sequence}"))),
            response_bytes: 10,
            occurred_at_utc: format!("2026-10-31T18:00:{:02}Z", sequence),
            disposition: SearchEventDispositionV1::Allowed,
            previous_event_sha256: String::new(),
            event_sha256: String::new(),
        }
    }

    fn trace() -> SearchTimeExposureTraceV1 {
        let mut trace = SearchTimeExposureTraceV1 {
            schema: SEARCH_TRACE_SCHEMA.to_owned(),
            run_id: "composition-run-001".to_owned(),
            benchmark_id: "synthetic-composition-benchmark".to_owned(),
            benchmark_version: "v1".to_owned(),
            model_checkpoint_sha256: digest("checkpoint"),
            challenge_packet_sha256: digest("challenge"),
            network_policy_sha256: digest("policy"),
            enforcement_build_sha256: digest("enforcement"),
            policy_id: "allowlisted-v1".to_owned(),
            access_mode: SearchAccessModeV1::Allowlisted,
            allowlisted_destination_sha256s: vec![digest("destination")],
            instrumented_channels: vec![
                SearchInformationChannelV1::WebSearch,
                SearchInformationChannelV1::BrowserNavigation,
                SearchInformationChannelV1::HttpApi,
                SearchInformationChannelV1::RemoteMcp,
                SearchInformationChannelV1::VectorStore,
                SearchInformationChannelV1::LocalCache,
                SearchInformationChannelV1::MountedFile,
                SearchInformationChannelV1::AgentMemory,
                SearchInformationChannelV1::Other,
            ],
            unobserved_channels: Vec::new(),
            events: vec![event(1), event(2), event(3)],
            event_count: 0,
            trace_root_sha256: String::new(),
            claim_boundary: "synthetic trace fixture".to_owned(),
        };
        seal_search_time_trace(&mut trace).unwrap();
        trace
    }

    fn finding() -> CompositionalExposureFindingV1 {
        let support = vec![1, 2, 3];
        let single = support.iter().map(|sequence| SingleEventInsufficiencyV1 {
            event_sequence: *sequence,
            evidence_sha256: digest(&format!("single-insufficient-{sequence}")),
            method: CompositionalEvidenceMethodV1::HumanAdjudication,
            confidence_permille: 1_000,
        }).collect();
        let leave = support.iter().map(|omitted| {
            let subset: Vec<u64> = support.iter().copied().filter(|sequence| sequence != omitted).collect();
            LeaveOneOutInsufficiencyV1 {
                omitted_event_sequence: *omitted,
                tested_subset_sha256: canonical_event_subset_digest(&subset).unwrap(),
                tested_event_sequences: subset,
                evidence_sha256: digest(&format!("leave-one-out-{omitted}")),
                method: CompositionalEvidenceMethodV1::HumanAdjudication,
                confidence_permille: 1_000,
            }
        }).collect();
        CompositionalExposureFindingV1 {
            finding_id: "composition-001".to_owned(),
            exposure_class: CompositionalExposureClassV1::AnswerBearing,
            protected_conclusion_commitment_sha256: digest("protected-conclusion"),
            matched_item_ids: vec!["item-001".to_owned()],
            supporting_event_sequences: support,
            single_event_insufficiency: single,
            leave_one_out_insufficiency: leave,
            joint_sufficiency: JointSufficiencyEvidenceV1 {
                evidence_sha256: digest("joint-sufficient"),
                method: CompositionalEvidenceMethodV1::HumanAdjudication,
                confidence_permille: 1_000,
            },
            monotonicity_assumed: true,
            monotonicity_evidence_sha256: Some(digest("monotone-predicate")),
            confidence_permille: 1_000,
        }
    }

    fn findings(trace: &SearchTimeExposureTraceV1) -> CompositionalExposureFindingsV1 {
        let mut findings = CompositionalExposureFindingsV1 {
            schema: COMPOSITION_FINDINGS_SCHEMA.to_owned(),
            run_id: trace.run_id.clone(),
            trace_root_sha256: trace.trace_root_sha256.clone(),
            findings: vec![finding()],
            findings_sha256: String::new(),
            claim_boundary: "synthetic evaluator fixture".to_owned(),
        };
        findings.findings_sha256 = compositional_findings_digest(&findings).unwrap();
        findings
    }

    #[test]
    fn valid_three_event_minimal_composition_is_audited() {
        let trace = trace();
        let findings = findings(&trace);
        let report = audit_compositional_exposure(&trace, &findings).unwrap();
        assert!(report.valid);
        assert_eq!(report.answer_bearing_count, 1);
        assert_eq!(report.minimality_certified_count, 1);
        assert_eq!(report.single_source_attestation_count, 3);
        assert_eq!(report.leave_one_out_attestation_count, 3);
        assert!(!report.independently_verified);
        assert!(verify_compositional_audit_integrity(&report).is_ok());
    }

    #[test]
    fn non_monotone_predicate_is_audited_but_not_minimality_certified() {
        let trace = trace();
        let mut findings = findings(&trace);
        findings.findings[0].monotonicity_assumed = false;
        findings.findings[0].monotonicity_evidence_sha256 = None;
        findings.findings_sha256 = compositional_findings_digest(&findings).unwrap();
        let report = audit_compositional_exposure(&trace, &findings).unwrap();
        assert_eq!(report.minimality_certified_count, 0);
        assert_eq!(report.answer_bearing_count, 1);
    }

    #[test]
    fn incomplete_single_source_insufficiency_attestations_fail() {
        let trace = trace();
        let mut findings = findings(&trace);
        findings.findings[0].single_event_insufficiency.pop();
        findings.findings_sha256 = compositional_findings_digest(&findings).unwrap();
        assert!(audit_compositional_exposure(&trace, &findings).unwrap_err()
            .contains("each supporting event"));
    }

    #[test]
    fn incomplete_or_wrong_leave_one_out_evidence_fails() {
        let trace = trace();
        let mut findings = findings(&trace);
        findings.findings[0].leave_one_out_insufficiency[0].tested_event_sequences = vec![2, 3];
        findings.findings[0].leave_one_out_insufficiency[0].tested_subset_sha256 =
            canonical_event_subset_digest(&[2, 3]).unwrap();
        findings.findings_sha256 = compositional_findings_digest(&findings).unwrap();
        assert!(audit_compositional_exposure(&trace, &findings).unwrap_err()
            .contains("exact supporting set"));

        let mut findings = findings(&trace);
        findings.findings[0].leave_one_out_insufficiency.pop();
        findings.findings_sha256 = compositional_findings_digest(&findings).unwrap();
        assert!(audit_compositional_exposure(&trace, &findings).unwrap_err()
            .contains("each supporting event"));
    }

    #[test]
    fn blocked_or_zero_byte_source_events_are_rejected() {
        let mut trace = trace();
        trace.events[2].disposition = SearchEventDispositionV1::Blocked;
        trace.events[2].response_sha256 = None;
        trace.events[2].response_bytes = 0;
        seal_search_time_trace(&mut trace).unwrap();
        let findings = findings(&trace);
        assert!(audit_compositional_exposure(&trace, &findings).unwrap_err()
            .contains("blocked trace event"));

        let mut trace = trace();
        trace.events[1].response_bytes = 0;
        trace.events[1].response_sha256 = None;
        seal_search_time_trace(&mut trace).unwrap();
        let findings = findings(&trace);
        assert!(audit_compositional_exposure(&trace, &findings).unwrap_err()
            .contains("no received response body"));
    }

    #[test]
    fn source_support_must_be_sorted_unique_and_contain_at_least_two_events() {
        let trace = trace();
        let mut findings = findings(&trace);
        findings.findings[0].supporting_event_sequences = vec![2, 1, 3];
        findings.findings_sha256 = compositional_findings_digest(&findings).unwrap();
        assert!(audit_compositional_exposure(&trace, &findings).unwrap_err()
            .contains("sorted unique sequences"));

        let mut findings = findings(&trace);
        findings.findings[0].supporting_event_sequences = vec![1];
        findings.findings_sha256 = compositional_findings_digest(&findings).unwrap();
        assert!(audit_compositional_exposure(&trace, &findings).is_err());
    }

    #[test]
    fn stale_trace_findings_and_unknown_events_fail() {
        let trace = trace();
        let mut findings = findings(&trace);
        findings.trace_root_sha256 = digest("another-trace");
        findings.findings_sha256 = compositional_findings_digest(&findings).unwrap();
        assert!(audit_compositional_exposure(&trace, &findings).unwrap_err()
            .contains("do not bind"));

        let mut findings = findings(&trace);
        findings.findings[0].supporting_event_sequences = vec![1, 2, 99];
        findings.findings_sha256 = compositional_findings_digest(&findings).unwrap();
        assert!(audit_compositional_exposure(&trace, &findings).unwrap_err()
            .contains("unknown trace event"));
    }

    #[test]
    fn duplicate_findings_ids_or_support_sets_fail() {
        let trace = trace();
        let mut findings = findings(&trace);
        findings.findings.push(findings.findings[0].clone());
        findings.findings_sha256 = compositional_findings_digest(&findings).unwrap();
        assert!(audit_compositional_exposure(&trace, &findings).unwrap_err()
            .contains("finding IDs must be unique"));

        let mut findings = findings(&trace);
        findings.findings[0].finding_id = "other-id".to_owned();
        findings.findings.push(findings.findings[0].clone());
        findings.findings_sha256 = compositional_findings_digest(&findings).unwrap();
        assert!(audit_compositional_exposure(&trace, &findings).unwrap_err()
            .contains("duplicate minimal supporting"));
    }

    #[test]
    fn digest_tampering_and_invalid_monotonicity_binding_fail_closed() {
        let trace = trace();
        let mut findings = findings(&trace);
        findings.findings[0].joint_sufficiency.evidence_sha256 = "bad".to_owned();
        findings.findings_sha256 = compositional_findings_digest(&findings).unwrap();
        assert!(audit_compositional_exposure(&trace, &findings).is_err());

        let mut findings = findings(&trace);
        findings.findings[0].monotonicity_assumed = true;
        findings.findings[0].monotonicity_evidence_sha256 = None;
        findings.findings_sha256 = compositional_findings_digest(&findings).unwrap();
        assert!(audit_compositional_exposure(&trace, &findings).unwrap_err()
            .contains("monotonicity assumption"));
    }

    #[test]
    fn duplicate_event_subset_sequences_and_invalid_confidence_are_rejected() {
        let trace = trace();
        let mut findings = findings(&trace);
        findings.findings[0].single_event_insufficiency[0].confidence_permille = 1_001;
        findings.findings_sha256 = compositional_findings_digest(&findings).unwrap();
        assert!(audit_compositional_exposure(&trace, &findings).is_err());

        let mut findings = findings(&trace);
        findings.findings[0].leave_one_out_insufficiency[0].tested_event_sequences = vec![2, 2];
        findings.findings_sha256 = compositional_findings_digest(&findings).unwrap();
        assert!(audit_compositional_exposure(&trace, &findings).is_err());
    }

    #[test]
    fn audit_report_tampering_is_detected() {
        let trace = trace();
        let findings = findings(&trace);
        let mut report = audit_compositional_exposure(&trace, &findings).unwrap();
        report.answer_bearing_count = 0;
        assert!(verify_compositional_audit_integrity(&report)
            .unwrap_err().contains("digest mismatch"));
    }
}
