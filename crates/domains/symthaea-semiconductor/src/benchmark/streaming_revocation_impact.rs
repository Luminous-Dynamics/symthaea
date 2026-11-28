// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only requalification impact analysis for online adapter update journals.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

use super::gated_streaming_adapter::{
    GATED_STREAMING_ADAPTER_SCHEMA_V1, GatedStreamingAdapterEngineV1,
    StreamingAcceptedUpdateJournalEntryV1, StreamingUpdateDispositionV1,
    verify_streaming_prediction_receipt_v1, verify_streaming_update_receipt_v1,
};
use super::label_consumption_guard::{
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1,
    LabelEvidenceFrontierV1,
};

pub const STREAMING_REVOCATION_IMPACT_INPUT_SCHEMA_V1: &str =
    "luminous.semi-bench.streaming-revocation-impact-input.v1";
pub const STREAMING_REVOCATION_IMPACT_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.streaming-revocation-impact-report.v1";
pub const STREAMING_UPDATE_JOURNAL_ENTRY_SCHEMA_V1: &str =
    "luminous.semi-bench.streaming-update-journal-entry.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingRevocationImpactStatusV1 {
    CompleteNoReplayRequired,
    RebuildRequired,
    Unknown,
    Rejected,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingUpdateImpactStateV1 {
    Unaffected,
    DirectReplayRequired,
    InheritedReplayRequired,
    DirectIndeterminate,
    InheritedIndeterminate,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StreamingRevocationImpactInputV1 {
    pub schema: String,
    pub input_id: String,
    pub engine: GatedStreamingAdapterEngineV1,
    pub current_label_reports: Vec<LabelConsumptionEligibilityReportV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StreamingUpdateImpactEntryV1 {
    pub update_sequence: u64,
    pub label_id: String,
    pub label_revision: u64,
    pub state: StreamingUpdateImpactStateV1,
    pub direct_reason: Option<String>,
    pub inherited_from_label_ids: Vec<String>,
    pub requires_new_generation_replay: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StreamingRevocationImpactReportV1 {
    pub schema: String,
    pub input_id: String,
    pub engine_id: String,
    pub generation_id: String,
    pub journal_head_sha256: String,
    pub status: StreamingRevocationImpactStatusV1,
    pub reason_codes: Vec<String>,
    pub first_affected_update_sequence: Option<u64>,
    pub directly_invalidated_update_sequences: Vec<u64>,
    pub directly_indeterminate_update_sequences: Vec<u64>,
    pub inherited_impacted_update_sequences: Vec<u64>,
    pub unresolved_label_keys: Vec<String>,
    pub entries: Vec<StreamingUpdateImpactEntryV1>,
    pub historical_receipts_rewritten: bool,
    pub running_adapter_mutated: bool,
    pub replay_generation_executed: bool,
    pub retraining_executed: bool,
    pub equipment_control_authority: bool,
    pub report_sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum CurrentRisk {
    Clear,
    Unknown,
    Blocked,
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| format!("canonical JSON serialization failed: {error}"))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn state_digest(
    state: &super::gated_streaming_adapter::StreamingAdapterStateV1,
) -> Result<String, String> {
    json_sha256(state)
}

fn label_report_digest(report: &LabelConsumptionEligibilityReportV1) -> Result<String, String> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn journal_entry_digest(entry: &StreamingAcceptedUpdateJournalEntryV1) -> Result<String, String> {
    let mut canonical = entry.clone();
    canonical.entry_sha256.clear();
    json_sha256(&canonical)
}

fn report_digest(report: &StreamingRevocationImpactReportV1) -> Result<String, String> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn label_key(label_id: &str, revision: u64) -> String {
    format!("{label_id}:{revision}")
}

fn genesis_journal_digest(engine: &GatedStreamingAdapterEngineV1) -> Result<String, String> {
    json_sha256(&(
        GATED_STREAMING_ADAPTER_SCHEMA_V1,
        engine.engine_id.as_str(),
        engine.generation_id.as_str(),
        engine.global_model_artifact_sha256.as_str(),
        engine.feature_schema_sha256.as_str(),
        engine.policy_sha256.as_str(),
        engine.genesis_state_sha256.as_str(),
    ))
}

fn rejected_report(
    input: &StreamingRevocationImpactInputV1,
    reason: &str,
) -> Result<StreamingRevocationImpactReportV1, String> {
    let mut report = StreamingRevocationImpactReportV1 {
        schema: STREAMING_REVOCATION_IMPACT_REPORT_SCHEMA_V1.into(),
        input_id: input.input_id.clone(),
        engine_id: input.engine.engine_id.clone(),
        generation_id: input.engine.generation_id.clone(),
        journal_head_sha256: input.engine.journal_head_sha256.clone(),
        status: StreamingRevocationImpactStatusV1::Rejected,
        reason_codes: vec![reason.into()],
        first_affected_update_sequence: None,
        directly_invalidated_update_sequences: Vec::new(),
        directly_indeterminate_update_sequences: Vec::new(),
        inherited_impacted_update_sequences: Vec::new(),
        unresolved_label_keys: Vec::new(),
        entries: Vec::new(),
        historical_receipts_rewritten: false,
        running_adapter_mutated: false,
        replay_generation_executed: false,
        retraining_executed: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

fn verify_journal(engine: &GatedStreamingAdapterEngineV1) -> Result<(), String> {
    if engine.schema != GATED_STREAMING_ADAPTER_SCHEMA_V1 {
        return Err("engine_schema_unsupported".into());
    }
    if !valid_sha256(&engine.global_model_artifact_sha256)
        || !valid_sha256(&engine.feature_schema_sha256)
        || !valid_sha256(&engine.policy_sha256)
        || !valid_sha256(&engine.genesis_state_sha256)
        || !valid_sha256(&engine.journal_head_sha256)
    {
        return Err("engine_identity_digest_malformed".into());
    }
    if state_digest(&engine.genesis_adapter_state)? != engine.genesis_state_sha256 {
        return Err("genesis_state_digest_mismatch".into());
    }
    if engine.accepted_update_journal.len() as u64 != engine.adapter_state.accepted_update_count {
        return Err("accepted_update_journal_count_mismatch".into());
    }

    let mut previous_state = engine.genesis_state_sha256.clone();
    let mut previous_journal = genesis_journal_digest(engine)?;
    let mut accepted_keys = BTreeSet::new();
    let mut measurement_events = BTreeSet::new();
    let mut sample_ids = BTreeSet::new();

    for (index, entry) in engine.accepted_update_journal.iter().enumerate() {
        let sequence = index as u64 + 1;
        if entry.schema != STREAMING_UPDATE_JOURNAL_ENTRY_SCHEMA_V1
            || entry.engine_id != engine.engine_id
            || entry.generation_id != engine.generation_id
            || entry.update_sequence != sequence
            || entry.arrival_sequence != sequence
        {
            return Err(format!("journal_identity_or_sequence_invalid:{sequence}"));
        }
        if entry.parent_state_sha256 != previous_state
            || entry.parent_journal_sha256 != previous_journal
        {
            return Err(format!("journal_parent_chain_mismatch:{sequence}"));
        }
        if !valid_sha256(&entry.resulting_state_sha256)
            || !valid_sha256(&entry.entry_sha256)
            || journal_entry_digest(entry)? != entry.entry_sha256
        {
            return Err(format!("journal_entry_digest_invalid:{sequence}"));
        }
        verify_streaming_prediction_receipt_v1(&entry.prediction_receipt)
            .map_err(|_| format!("prediction_receipt_invalid:{sequence}"))?;
        verify_streaming_update_receipt_v1(&entry.update_receipt)
            .map_err(|_| format!("update_receipt_invalid:{sequence}"))?;

        let delivery = &entry.delivery;
        let report = &delivery.eligibility_report;
        if entry.update_receipt.disposition != StreamingUpdateDispositionV1::Updated
            || entry.update_receipt.update_sequence != Some(sequence)
            || entry.update_receipt.adapter_state_before_sha256 != entry.parent_state_sha256
            || entry.update_receipt.adapter_state_after_sha256 != entry.resulting_state_sha256
            || entry.update_receipt.prediction_receipt_sha256.as_deref()
                != Some(entry.prediction_receipt.receipt_sha256.as_str())
            || entry.update_receipt.eligibility_report_sha256.as_deref()
                != Some(report.report_sha256.as_str())
            || entry.update_receipt.label_id != delivery.label_id
            || entry.update_receipt.label_revision != delivery.label_revision
            || entry.update_receipt.label_payload_sha256 != delivery.label_payload_sha256
            || entry.update_receipt.measurement_event_id != delivery.measurement_event_id
            || entry.prediction_receipt.sample_id != delivery.sample_id
            || entry.prediction_receipt.measurement_event_id != delivery.measurement_event_id
            || report.label_id != delivery.label_id
            || report.label_revision != delivery.label_revision
            || report.label_sha256 != delivery.label_payload_sha256
            || report.measurement_event_id != delivery.measurement_event_id
        {
            return Err(format!("journal_evidence_binding_mismatch:{sequence}"));
        }
        if label_report_digest(report)? != report.report_sha256
            || report.frontier != LabelEvidenceFrontierV1::CurrentAtK1
            || report.eligibility != LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy
            || !report.downstream_consumption_allowed
            || (engine.policy.require_source_authenticity_assertion && !report.source_authenticity_proven)
        {
            return Err(format!("journal_contains_nonqualified_update:{sequence}"));
        }
        if !accepted_keys.insert(label_key(&delivery.label_id, delivery.label_revision))
            || !measurement_events.insert(delivery.measurement_event_id.clone())
            || !sample_ids.insert(delivery.sample_id.clone())
        {
            return Err(format!("journal_duplicate_identity:{sequence}"));
        }
        previous_state = entry.resulting_state_sha256.clone();
        previous_journal = entry.entry_sha256.clone();
    }

    if state_digest(&engine.adapter_state)? != previous_state {
        return Err("live_adapter_state_does_not_match_journal_head".into());
    }
    if engine.journal_head_sha256 != previous_journal {
        return Err("journal_head_digest_mismatch".into());
    }
    if engine.accepted_label_revisions != accepted_keys {
        return Err("accepted_label_revision_index_mismatch".into());
    }
    Ok(())
}

fn classify_current_report(
    report: &LabelConsumptionEligibilityReportV1,
    require_source_authenticity: bool,
) -> CurrentRisk {
    if report.frontier != LabelEvidenceFrontierV1::CurrentAtK1 {
        return CurrentRisk::Unknown;
    }
    if require_source_authenticity && !report.source_authenticity_proven {
        return CurrentRisk::Blocked;
    }
    match report.eligibility {
        LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy
            if report.downstream_consumption_allowed => CurrentRisk::Clear,
        LabelConsumptionEligibilityV1::Unknown => CurrentRisk::Unknown,
        LabelConsumptionEligibilityV1::Blocked => CurrentRisk::Blocked,
        LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy => CurrentRisk::Blocked,
    }
}

/// Computes which accepted adapter updates need a new generation replay after
/// current K1 requalification. The function is read-only.
pub fn assess_streaming_revocation_impact_v1(
    input: &StreamingRevocationImpactInputV1,
) -> Result<StreamingRevocationImpactReportV1, String> {
    if input.schema != STREAMING_REVOCATION_IMPACT_INPUT_SCHEMA_V1 {
        return rejected_report(input, "impact_input_schema_unsupported");
    }
    if input.input_id.trim().is_empty() {
        return rejected_report(input, "impact_input_id_missing");
    }
    if let Err(reason) = verify_journal(&input.engine) {
        return rejected_report(input, &reason);
    }

    let mut current_reports = BTreeMap::new();
    for report in &input.current_label_reports {
        if report.label_id.trim().is_empty()
            || report.label_revision == 0
            || !valid_sha256(&report.label_sha256)
            || !valid_sha256(&report.report_sha256)
            || label_report_digest(report)? != report.report_sha256
        {
            return rejected_report(input, "current_k1_report_identity_or_digest_invalid");
        }
        let key = label_key(&report.label_id, report.label_revision);
        if current_reports.insert(key, report).is_some() {
            return rejected_report(input, "duplicate_current_k1_report");
        }
    }

    let mut risks = BTreeMap::new();
    let mut reasons = BTreeMap::new();
    let mut unresolved = Vec::new();
    let mut direct_blocked = Vec::new();
    let mut direct_unknown = Vec::new();
    let mut first_problem: Option<(u64, CurrentRisk)> = None;

    for entry in &input.engine.accepted_update_journal {
        let key = label_key(&entry.delivery.label_id, entry.delivery.label_revision);
        let risk = match current_reports.get(&key) {
            None => {
                unresolved.push(key.clone());
                reasons.insert(entry.update_sequence, "current_k1_report_missing".to_string());
                CurrentRisk::Unknown
            }
            Some(current) => {
                if current.label_sha256 != entry.delivery.label_payload_sha256
                    || current.measurement_event_id != entry.delivery.measurement_event_id
                {
                    return rejected_report(input, "current_k1_report_does_not_match_historical_payload");
                }
                classify_current_report(current, input.engine.policy.require_source_authenticity_assertion)
            }
        };
        risks.insert(entry.update_sequence, risk);
        match risk {
            CurrentRisk::Clear => {}
            CurrentRisk::Unknown => {
                direct_unknown.push(entry.update_sequence);
                reasons.entry(entry.update_sequence)
                    .or_insert_with(|| "current_k1_status_unknown_or_stale".into());
                if first_problem.is_none() {
                    first_problem = Some((entry.update_sequence, risk));
                }
            }
            CurrentRisk::Blocked => {
                direct_blocked.push(entry.update_sequence);
                reasons.insert(entry.update_sequence, "historical_update_evidence_now_blocked".into());
                if first_problem.is_none() {
                    first_problem = Some((entry.update_sequence, risk));
                }
            }
        }
    }

    let mut inherited = Vec::new();
    let mut entries = Vec::with_capacity(input.engine.accepted_update_journal.len());
    for journal_entry in &input.engine.accepted_update_journal {
        let sequence = journal_entry.update_sequence;
        let own = risks.get(&sequence).copied().unwrap_or(CurrentRisk::Unknown);
        let state = match own {
            CurrentRisk::Blocked => StreamingUpdateImpactStateV1::DirectReplayRequired,
            CurrentRisk::Unknown => StreamingUpdateImpactStateV1::DirectIndeterminate,
            CurrentRisk::Clear => match first_problem {
                Some((first, CurrentRisk::Blocked)) if sequence > first => {
                    inherited.push(sequence);
                    StreamingUpdateImpactStateV1::InheritedReplayRequired
                }
                Some((first, CurrentRisk::Unknown)) if sequence > first => {
                    inherited.push(sequence);
                    StreamingUpdateImpactStateV1::InheritedIndeterminate
                }
                _ => StreamingUpdateImpactStateV1::Unaffected,
            },
        };
        let inherited_from_label_ids = match first_problem {
            Some((first, _)) if sequence > first => input.engine.accepted_update_journal
                .iter()
                .filter(|candidate| candidate.update_sequence <= first)
                .filter(|candidate| risks.get(&candidate.update_sequence) != Some(&CurrentRisk::Clear))
                .map(|candidate| candidate.delivery.label_id.clone())
                .collect(),
            _ => Vec::new(),
        };
        entries.push(StreamingUpdateImpactEntryV1 {
            update_sequence: sequence,
            label_id: journal_entry.delivery.label_id.clone(),
            label_revision: journal_entry.delivery.label_revision,
            state,
            direct_reason: reasons.get(&sequence).cloned(),
            inherited_from_label_ids,
            requires_new_generation_replay: state != StreamingUpdateImpactStateV1::Unaffected,
        });
    }

    let status = match first_problem {
        None => StreamingRevocationImpactStatusV1::CompleteNoReplayRequired,
        Some((_, CurrentRisk::Blocked)) => StreamingRevocationImpactStatusV1::RebuildRequired,
        Some((_, CurrentRisk::Unknown)) => StreamingRevocationImpactStatusV1::Unknown,
        Some((_, CurrentRisk::Clear)) => StreamingRevocationImpactStatusV1::CompleteNoReplayRequired,
    };
    let mut reason_codes = Vec::new();
    match status {
        StreamingRevocationImpactStatusV1::RebuildRequired => {
            reason_codes.push("accepted_update_uses_now_blocked_label_evidence".into());
        }
        StreamingRevocationImpactStatusV1::Unknown => {
            reason_codes.push("accepted_update_has_missing_or_unknown_current_k1_evidence".into());
        }
        _ => {}
    }
    unresolved.sort();
    unresolved.dedup();

    let mut report = StreamingRevocationImpactReportV1 {
        schema: STREAMING_REVOCATION_IMPACT_REPORT_SCHEMA_V1.into(),
        input_id: input.input_id.clone(),
        engine_id: input.engine.engine_id.clone(),
        generation_id: input.engine.generation_id.clone(),
        journal_head_sha256: input.engine.journal_head_sha256.clone(),
        status,
        reason_codes,
        first_affected_update_sequence: first_problem.map(|value| value.0),
        directly_invalidated_update_sequences: direct_blocked,
        directly_indeterminate_update_sequences: direct_unknown,
        inherited_impacted_update_sequences: inherited,
        unresolved_label_keys: unresolved,
        entries,
        historical_receipts_rewritten: false,
        running_adapter_mutated: false,
        replay_generation_executed: false,
        retraining_executed: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}
