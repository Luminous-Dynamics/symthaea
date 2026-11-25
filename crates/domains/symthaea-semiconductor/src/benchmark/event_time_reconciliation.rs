// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dual-clock event-time reconstruction for streaming semiconductor VM.
//!
//! Arrival-order receipts remain immutable. This module reconstructs a new
//! generation from an append-only event log ordered by declared event time
//! and source sequence, while preserving ingestion time as audit evidence.
//! A late eligibility correction never rewrites a prior update; it marks the
//! affected label and requires a distinct rebuild/requalification generation.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use super::gated_streaming_adapter::{
    GatedStreamingAdapterEngineV1, StreamingLabelDeliveryV1, StreamingPredictionInputV1,
    StreamingUpdateDispositionV1, adapter_state_digest_for_reconciliation_v1,
    deliver_qualified_stream_label_v1, predict_and_register_stream_sample_v1,
};
use super::label_consumption_guard::{
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1,
};

pub const EVENT_RECONSTRUCTION_SCHEMA_V1: &str =
    "luminous.semi-bench.event-reconstruction.v1";
pub const RECONSTRUCTION_EVENT_ENVELOPE_SCHEMA_V1: &str =
    "luminous.semi-bench.reconstruction-event-envelope.v1";
pub const RECONSTRUCTION_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.reconstruction-report.v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StreamingReconstructionPayloadV1 {
    Prediction {
        input: StreamingPredictionInputV1,
    },
    LabelDelivery {
        delivery: StreamingLabelDeliveryV1,
    },
    EligibilityCorrection {
        sample_id: String,
        label_id: String,
        label_revision: u64,
        measurement_event_id: String,
        /// Valid-time frontier from which this K1 correction applies.
        effective_at_unix_seconds: i64,
        /// Time at which the corrected K1 report became available to the system.
        report_available_at_unix_seconds: i64,
        report: LabelConsumptionEligibilityReportV1,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReconstructionEventEnvelopeV1 {
    pub schema: String,
    pub event_id: String,
    pub event_time_unix_seconds: i64,
    pub ingestion_time_unix_seconds: i64,
    pub clock_domain_id: String,
    /// Monotonic source sequence. Required; no hash/ID fallback tie-break is allowed.
    pub source_sequence: Option<u64>,
    pub payload: StreamingReconstructionPayloadV1,
    pub payload_sha256: String,
    pub envelope_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EventTimeReconstructionInputV1 {
    pub schema: String,
    pub generation_id: String,
    pub clock_domain_id: String,
    pub knowledge_cutoff_unix_seconds: i64,
    pub initial_engine: GatedStreamingAdapterEngineV1,
    pub events: Vec<ReconstructionEventEnvelopeV1>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReconstructionStatusV1 {
    Complete,
    Unknown,
    RebuildRequired,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReconstructionEventDispositionV1 {
    PredictionRegistered,
    LabelUpdated,
    LabelHeldUnknown,
    LabelBlocked,
    LabelRejected,
    EligibilityCorrectionAppliedToFutureEvents,
    LateCorrectionAffectsPriorUpdate,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReconstructionEventReceiptV1 {
    pub event_id: String,
    pub event_time_unix_seconds: i64,
    pub ingestion_time_unix_seconds: i64,
    pub source_sequence: u64,
    pub payload_sha256: String,
    pub original_event_receipt_sha256: Option<String>,
    pub disposition: ReconstructionEventDispositionV1,
    pub sample_id: String,
    pub label_id: Option<String>,
    pub label_revision: Option<u64>,
    pub adapter_state_before_sha256: String,
    pub adapter_state_after_sha256: String,
    pub referenced_original_receipt_unchanged: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EventTimeReconstructionReportV1 {
    pub schema: String,
    pub generation_id: String,
    pub clock_domain_id: String,
    pub knowledge_cutoff_unix_seconds: i64,
    pub status: ReconstructionStatusV1,
    pub event_log_sha256: String,
    /// Excludes ingestion-time differences; this is the canonical causal replay sequence.
    pub canonical_replay_sha256: String,
    pub initial_adapter_state_sha256: String,
    pub final_adapter_state_sha256: String,
    pub accepted_update_count: u64,
    pub held_unknown_count: u64,
    pub blocked_count: u64,
    pub rejected_count: u64,
    pub late_corrected_label_ids: Vec<String>,
    pub affected_measurement_event_ids: Vec<String>,
    pub ordered_event_ids: Vec<String>,
    pub event_receipts: Vec<ReconstructionEventReceiptV1>,
    pub source_events_mutated: bool,
    pub prior_receipts_mutated: bool,
    pub automatic_rebuild_executed: bool,
    pub equipment_control_authority: bool,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct EventTimeReconstructionErrorV1(pub String);

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn json_sha256<T: Serialize + ?Sized>(
    value: &T,
) -> Result<String, EventTimeReconstructionErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        EventTimeReconstructionErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn payload_digest(
    payload: &StreamingReconstructionPayloadV1,
) -> Result<String, EventTimeReconstructionErrorV1> {
    json_sha256(payload)
}

fn envelope_digest(
    envelope: &ReconstructionEventEnvelopeV1,
) -> Result<String, EventTimeReconstructionErrorV1> {
    let mut canonical = envelope.clone();
    canonical.envelope_sha256.clear();
    json_sha256(&canonical)
}

fn report_digest(
    report: &EventTimeReconstructionReportV1,
) -> Result<String, EventTimeReconstructionErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn label_key(label_id: &str, revision: u64) -> String {
    format!("{label_id}:{revision}")
}

fn validate_payload_time(
    envelope: &ReconstructionEventEnvelopeV1,
) -> Result<(), EventTimeReconstructionErrorV1> {
    let expected = match &envelope.payload {
        StreamingReconstructionPayloadV1::Prediction { input } => {
            if input.clock_domain_id != envelope.clock_domain_id {
                return Err(EventTimeReconstructionErrorV1(
                    "prediction payload and event envelope clock domains differ".into(),
                ));
            }
            input.prediction_at_unix_seconds
        }
        StreamingReconstructionPayloadV1::LabelDelivery { delivery } => {
            if delivery.clock_domain_id != envelope.clock_domain_id {
                return Err(EventTimeReconstructionErrorV1(
                    "delivery payload and event envelope clock domains differ".into(),
                ));
            }
            delivery.delivered_at_unix_seconds
        }
        StreamingReconstructionPayloadV1::EligibilityCorrection {
            label_id,
            label_revision,
            measurement_event_id,
            effective_at_unix_seconds,
            report_available_at_unix_seconds,
            report,
            ..
        } => {
            if report.label_id != *label_id
                || report.label_revision != *label_revision
                || report.measurement_event_id != *measurement_event_id
                || report.frontier
                    != super::label_consumption_guard::LabelEvidenceFrontierV1::CurrentAtK1
                || !valid_sha256(&report.report_sha256)
            {
                return Err(EventTimeReconstructionErrorV1(
                    "correction report identity, revision, frontier or digest is invalid".into(),
                ));
            }
            if *report_available_at_unix_seconds < *effective_at_unix_seconds
                || *report_available_at_unix_seconds < report.qualified_corrected_value as i64
                    && false
            {
                return Err(EventTimeReconstructionErrorV1(
                    "correction report availability precedes its valid-time frontier".into(),
                ));
            }
            *report_available_at_unix_seconds
        }
    };
    if expected != envelope.event_time_unix_seconds {
        return Err(EventTimeReconstructionErrorV1(
            "event envelope time does not match its payload's declared event time".into(),
        ));
    }
    Ok(())
}

fn unknown_report(
    input: &EventTimeReconstructionInputV1,
    reason_event_ids: Vec<String>,
) -> Result<EventTimeReconstructionReportV1, EventTimeReconstructionErrorV1> {
    let mut report = EventTimeReconstructionReportV1 {
        schema: RECONSTRUCTION_REPORT_SCHEMA_V1.into(),
        generation_id: input.generation_id.clone(),
        clock_domain_id: input.clock_domain_id.clone(),
        knowledge_cutoff_unix_seconds: input.knowledge_cutoff_unix_seconds,
        status: ReconstructionStatusV1::Unknown,
        event_log_sha256: json_sha256(&input.events)?,
        canonical_replay_sha256: String::new(),
        initial_adapter_state_sha256: adapter_state_digest_for_reconciliation_v1(
            &input.initial_engine,
        )
        .map_err(|error| EventTimeReconstructionErrorV1(error.to_string()))?,
        final_adapter_state_sha256: adapter_state_digest_for_reconciliation_v1(
            &input.initial_engine,
        )
        .map_err(|error| EventTimeReconstructionErrorV1(error.to_string()))?,
        accepted_update_count: 0,
        held_unknown_count: 0,
        blocked_count: 0,
        rejected_count: 0,
        late_corrected_label_ids: Vec::new(),
        affected_measurement_event_ids: Vec::new(),
        ordered_event_ids: reason_event_ids,
        event_receipts: Vec::new(),
        source_events_mutated: false,
        prior_receipts_mutated: false,
        automatic_rebuild_executed: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

/// Creates a sealed event envelope. The caller is responsible for supplying
/// real source sequence and clock values; this helper does not invent either.
pub fn seal_reconstruction_event_v1(
    envelope: &mut ReconstructionEventEnvelopeV1,
) -> Result<(), EventTimeReconstructionErrorV1> {
    envelope.payload_sha256 = payload_digest(&envelope.payload)?;
    envelope.envelope_sha256.clear();
    envelope.envelope_sha256 = envelope_digest(envelope)?;
    Ok(())
}

fn correction_report_digest(
    report: &LabelConsumptionEligibilityReportV1,
) -> Result<String, EventTimeReconstructionErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn event_sequence(
    event: &ReconstructionEventEnvelopeV1,
) -> Result<u64, EventTimeReconstructionErrorV1> {
    event.source_sequence.ok_or_else(|| {
        EventTimeReconstructionErrorV1(format!(
            "event {} has no source sequence for total ordering",
            event.event_id
        ))
    })
}

/// Reconstructs a new model-state generation by event time, then source sequence.
/// Ingestion times remain bound to the audit manifest but do not change causal
/// replay ordering. Events ingested after the knowledge cut-off are excluded.
pub fn reconstruct_streaming_adapter_event_time_v1(
    input: &EventTimeReconstructionInputV1,
) -> Result<EventTimeReconstructionReportV1, EventTimeReconstructionErrorV1> {
    if input.schema != EVENT_RECONSTRUCTION_SCHEMA_V1
        || input.generation_id.trim().is_empty()
        || input.clock_domain_id.trim().is_empty()
    {
        return Err(EventTimeReconstructionErrorV1(
            "event-time reconstruction schema or identity is invalid".into(),
        ));
    }
    let initial_state = adapter_state_digest_for_reconciliation_v1(&input.initial_engine)
        .map_err(|error| EventTimeReconstructionErrorV1(error.to_string()))?;
    let mut eligible_events = Vec::new();
    let mut event_ids = BTreeSet::new();
    let mut source_sequences = BTreeSet::new();
    let mut unknown_order_ids = Vec::new();

    for event in &input.events {
        if event.schema != RECONSTRUCTION_EVENT_ENVELOPE_SCHEMA_V1
            || event.event_id.trim().is_empty()
            || !event_ids.insert(event.event_id.as_str())
        {
            return Err(EventTimeReconstructionErrorV1(
                "event envelope schema or event identity is invalid/duplicated".into(),
            ));
        }
        if event.clock_domain_id != input.clock_domain_id {
            return unknown_report(input, vec![event.event_id.clone()]);
        }
        if event.ingestion_time_unix_seconds < event.event_time_unix_seconds {
            return Err(EventTimeReconstructionErrorV1(format!(
                "event {} was ingested before its declared event time",
                event.event_id
            )));
        }
        let sequence = match event_sequence(event) {
            Ok(value) => value,
            Err(_) => {
                unknown_order_ids.push(event.event_id.clone());
                continue;
            }
        };
        if !source_sequences.insert(sequence) {
            return Err(EventTimeReconstructionErrorV1(
                "source sequence is duplicated within the reconstruction clock domain".into(),
            ));
        }
        if payload_digest(&event.payload)? != event.payload_sha256
            || envelope_digest(event)? != event.envelope_sha256
        {
            return Err(EventTimeReconstructionErrorV1(format!(
                "event {} payload/envelope digest mismatch",
                event.event_id
            )));
        }
        validate_payload_time(event)?;
        if event.ingestion_time_unix_seconds <= input.knowledge_cutoff_unix_seconds {
            eligible_events.push(event.clone());
        }
    }
    if !unknown_order_ids.is_empty() {
        return unknown_report(input, unknown_order_ids);
    }

    eligible_events.sort_by(|left, right| {
        left.event_time_unix_seconds
            .cmp(&right.event_time_unix_seconds)
            .then_with(|| left.source_sequence.cmp(&right.source_sequence))
    });

    let mut engine = input.initial_engine.clone();
    let mut latest_corrections: BTreeMap<String, LabelConsumptionEligibilityReportV1> =
        BTreeMap::new();
    let mut accepted_updates: BTreeMap<String, String> = BTreeMap::new();
    let mut late_corrected_label_ids = BTreeSet::new();
    let mut affected_events = BTreeSet::new();
    let mut event_receipts = Vec::new();
    let mut ordered_event_ids = Vec::new();
    let mut held_unknown_count = 0_u64;
    let mut blocked_count = 0_u64;
    let mut rejected_count = 0_u64;
    let mut replay_projection = Vec::new();

    for event in &eligible_events {
        let sequence = event_sequence(event)?;
        let before = adapter_state_digest_for_reconciliation_v1(&engine)
            .map_err(|error| EventTimeReconstructionErrorV1(error.to_string()))?;
        let mut original_receipt = None;
        let (sample_id, label_id, label_revision, disposition) = match &event.payload {
            StreamingReconstructionPayloadV1::Prediction { input: prediction } => {
                let receipt = predict_and_register_stream_sample_v1(&mut engine, prediction)
                    .map_err(|error| EventTimeReconstructionErrorV1(error.to_string()))?;
                original_receipt = Some(receipt.receipt_sha256);
                (
                    prediction.sample_id.clone(),
                    None,
                    None,
                    ReconstructionEventDispositionV1::PredictionRegistered,
                )
            }
            StreamingReconstructionPayloadV1::LabelDelivery { delivery } => {
                let key = label_key(&delivery.label_id, delivery.label_revision);
                let mut replay_delivery = delivery.clone();
                if let Some(corrected) = latest_corrections.get(&key) {
                    if corrected.requalification_report_sha256
                        == delivery.eligibility_report.requalification_report_sha256
                    {
                        replay_delivery.eligibility_report = corrected.clone();
                        replay_delivery.eligibility_report_available_at_unix_seconds = event
                            .event_time_unix_seconds;
                    }
                }
                let receipt = deliver_qualified_stream_label_v1(&mut engine, &replay_delivery)
                    .map_err(|error| EventTimeReconstructionErrorV1(error.to_string()))?;
                original_receipt = Some(receipt.receipt_sha256);
                let result = match receipt.disposition {
                    StreamingUpdateDispositionV1::Updated => {
                        accepted_updates.insert(key, delivery.measurement_event_id.clone());
                        ReconstructionEventDispositionV1::LabelUpdated
                    }
                    StreamingUpdateDispositionV1::HeldUnknown => {
                        held_unknown_count += 1;
                        ReconstructionEventDispositionV1::LabelHeldUnknown
                    }
                    StreamingUpdateDispositionV1::Blocked => {
                        blocked_count += 1;
                        ReconstructionEventDispositionV1::LabelBlocked
                    }
                    _ => {
                        rejected_count += 1;
                        ReconstructionEventDispositionV1::LabelRejected
                    }
                };
                (
                    delivery.sample_id.clone(),
                    Some(delivery.label_id.clone()),
                    Some(delivery.label_revision),
                    result,
                )
            }
            StreamingReconstructionPayloadV1::EligibilityCorrection {
                sample_id,
                label_id,
                label_revision,
                measurement_event_id,
                report,
                ..
            } => {
                let key = label_key(label_id, *label_revision);
                if correction_report_digest(report)? != report.report_sha256 {
                    return Err(EventTimeReconstructionErrorV1(
                        "eligibility correction report digest mismatch".into(),
                    ));
                }
                let old_report = latest_corrections.insert(key.clone(), report.clone());
                if accepted_updates.contains_key(&key) {
                    late_corrected_label_ids.insert(label_id.clone());
                    affected_events.insert(measurement_event_id.clone());
                    (
                        sample_id.clone(),
                        Some(label_id.clone()),
                        Some(*label_revision),
                        ReconstructionEventDispositionV1::LateCorrectionAffectsPriorUpdate,
                    )
                } else {
                    (
                        sample_id.clone(),
                        Some(label_id.clone()),
                        Some(*label_revision),
                        ReconstructionEventDispositionV1::EligibilityCorrectionAppliedToFutureEvents,
                    )
                }
            }
        };
        let after = adapter_state_digest_for_reconciliation_v1(&engine)
            .map_err(|error| EventTimeReconstructionErrorV1(error.to_string()))?;
        let receipt = ReconstructionEventReceiptV1 {
            event_id: event.event_id.clone(),
            event_time_unix_seconds: event.event_time_unix_seconds,
            ingestion_time_unix_seconds: event.ingestion_time_unix_seconds,
            source_sequence: sequence,
            payload_sha256: event.payload_sha256.clone(),
            original_event_receipt_sha256: original_receipt,
            disposition,
            sample_id,
            label_id,
            label_revision,
            adapter_state_before_sha256: before,
            adapter_state_after_sha256: after,
            referenced_original_receipt_unchanged: true,
        };
        replay_projection.push((
            event.event_id.clone(),
            event.event_time_unix_seconds,
            sequence,
            event.payload_sha256.clone(),
            disposition,
        ));
        ordered_event_ids.push(event.event_id.clone());
        event_receipts.push(receipt);
    }

    let final_state = adapter_state_digest_for_reconciliation_v1(&engine)
        .map_err(|error| EventTimeReconstructionErrorV1(error.to_string()))?;
    let event_log_sha256 = json_sha256(&input.events)?;
    let canonical_replay_sha256 = json_sha256(&replay_projection)?;
    let mut report = EventTimeReconstructionReportV1 {
        schema: RECONSTRUCTION_REPORT_SCHEMA_V1.into(),
        generation_id: input.generation_id.clone(),
        clock_domain_id: input.clock_domain_id.clone(),
        knowledge_cutoff_unix_seconds: input.knowledge_cutoff_unix_seconds,
        status: if late_corrected_label_ids.is_empty() {
            ReconstructionStatusV1::Complete
        } else {
            ReconstructionStatusV1::RebuildRequired
        },
        event_log_sha256,
        canonical_replay_sha256,
        initial_adapter_state_sha256: initial_state,
        final_adapter_state_sha256: final_state,
        accepted_update_count: engine.adapter_state.accepted_update_count,
        held_unknown_count,
        blocked_count,
        rejected_count,
        late_corrected_label_ids: late_corrected_label_ids.into_iter().collect(),
        affected_measurement_event_ids: affected_events.into_iter().collect(),
        ordered_event_ids,
        event_receipts,
        source_events_mutated: false,
        prior_receipts_mutated: false,
        automatic_rebuild_executed: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::gated_streaming_adapter::{
        STREAMING_ADAPTER_POLICY_SCHEMA_V1, STREAMING_PREDICTION_SCHEMA_V1,
        STREAMING_DELIVERY_SCHEMA_V1, create_gated_streaming_adapter_engine_v1,
        StreamingAdapterPolicyV1,
    };
    use super::super::label_consumption_guard::{
        LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1,
        LabelEvidenceFrontierV1,
    };

    fn digest(ch: char) -> String { std::iter::repeat_n(ch, 64).collect() }

    fn engine() -> GatedStreamingAdapterEngineV1 {
        create_gated_streaming_adapter_engine_v1(
            "replay-engine", "fab-clock", &digest('g'), &digest('a'), vec![0.0],
            StreamingAdapterPolicyV1 {
                schema: STREAMING_ADAPTER_POLICY_SCHEMA_V1.into(),
                policy_id: "replay-policy".into(),
                revision: 1,
                feature_learning_rate: 0.2,
                feature_norm_floor: 1e-6,
                maximum_abs_weight_delta: 0.2,
                maximum_abs_feature_value: 100.0,
                maximum_abs_prediction_error: 100.0,
                global_context_ema_alpha: 0.25,
                local_context_ema_alpha: 0.5,
                minimum_context_support: 2,
                temporal_ema_alpha: 0.2,
                temporal_activation_threshold: 0.1,
            },
        ).unwrap()
    }

    fn prediction(sample: &str, event: &str, time: i64) -> StreamingPredictionInputV1 {
        StreamingPredictionInputV1 {
            schema: STREAMING_PREDICTION_SCHEMA_V1.into(),
            sample_id: sample.into(),
            measurement_event_id: event.into(),
            event_time_unix_seconds: time - 1,
            feature_available_at_unix_seconds: time,
            prediction_at_unix_seconds: time,
            clock_domain_id: "fab-clock".into(),
            global_model_artifact_sha256: digest('g'),
            feature_schema_sha256: digest('a'),
            feature_vector: vec![1.0],
            selected_feature_indices: vec![0],
            global_baseline_prediction: 10.0,
            context_key: None,
        }
    }

    fn eligibility(label: &str, revision: u64, event: &str, value: f64,
        status: LabelConsumptionEligibilityV1, available_at: i64) -> LabelConsumptionEligibilityReportV1 {
        let mut report = LabelConsumptionEligibilityReportV1 {
            schema: "luminous.semi-bench.label-consumption-report.v1".into(),
            label_id: label.into(),
            label_revision: revision,
            measurement_event_id: event.into(),
            calibration_certificate_id: "cert-1".into(),
            frontier: LabelEvidenceFrontierV1::CurrentAtK1,
            eligibility: status,
            downstream_consumption_allowed: status == LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            reasons: Vec::new(),
            label_sha256: digest('b'),
            measurement_sha256: digest('c'),
            qualified_corrected_value: value,
            qualified_standard_uncertainty_upper_bound: Some(0.1),
            qualification_report_sha256: digest('d'),
            requalification_report_sha256: digest('e'),
            requalification_frontier_sha256: digest('f'),
            source_authenticity_proven: false,
            training_snapshot_mutated: false,
            equipment_control_authority: false,
            report_sha256: String::new(),
        };
        let mut canonical = report.clone();
        canonical.report_sha256.clear();
        report.report_sha256 = json_sha256(&canonical).unwrap();
        let _ = available_at;
        report
    }

    fn delivery(sample: &str, label: &str, event: &str, delivered: i64) -> StreamingLabelDeliveryV1 {
        let report = eligibility(label, 1, event, 11.0,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, delivered - 1);
        StreamingLabelDeliveryV1 {
            schema: STREAMING_DELIVERY_SCHEMA_V1.into(),
            sample_id: sample.into(),
            label_id: label.into(),
            label_revision: 1,
            measurement_event_id: event.into(),
            label_payload_sha256: report.label_sha256.clone(),
            label_value: 11.0,
            label_available_at_unix_seconds: delivered - 3,
            eligibility_report_available_at_unix_seconds: delivered - 1,
            delivered_at_unix_seconds: delivered,
            clock_domain_id: "fab-clock".into(),
            idempotency_key: format!("{label}:1"),
            eligibility_report: report,
        }
    }

    fn envelope(id: &str, time: i64, ingested: i64, seq: u64,
        payload: StreamingReconstructionPayloadV1) -> ReconstructionEventEnvelopeV1 {
        let mut value = ReconstructionEventEnvelopeV1 {
            schema: RECONSTRUCTION_EVENT_ENVELOPE_SCHEMA_V1.into(),
            event_id: id.into(),
            event_time_unix_seconds: time,
            ingestion_time_unix_seconds: ingested,
            clock_domain_id: "fab-clock".into(),
            source_sequence: Some(seq),
            payload,
            payload_sha256: String::new(),
            envelope_sha256: String::new(),
        };
        seal_reconstruction_event_v1(&mut value).unwrap();
        value
    }

    fn input(events: Vec<ReconstructionEventEnvelopeV1>) -> EventTimeReconstructionInputV1 {
        EventTimeReconstructionInputV1 {
            schema: EVENT_RECONSTRUCTION_SCHEMA_V1.into(),
            generation_id: "rebuild-001".into(),
            clock_domain_id: "fab-clock".into(),
            knowledge_cutoff_unix_seconds: 1000,
            initial_engine: engine(),
            events,
        }
    }

    #[test]
    fn event_time_replay_is_invariant_to_transport_input_order() {
        let mut events = vec![
            envelope("p1", 100, 110, 1, StreamingReconstructionPayloadV1::Prediction {
                input: prediction("s1", "m1", 100),
            }),
            envelope("d1", 120, 130, 2, StreamingReconstructionPayloadV1::LabelDelivery {
                delivery: delivery("s1", "l1", "m1", 120),
            }),
            envelope("p2", 140, 150, 3, StreamingReconstructionPayloadV1::Prediction {
                input: prediction("s2", "m2", 140),
            }),
        ];
        let a = reconstruct_streaming_adapter_event_time_v1(&input(events.clone())).unwrap();
        events.reverse();
        let b = reconstruct_streaming_adapter_event_time_v1(&input(events)).unwrap();
        assert_eq!(a.canonical_replay_sha256, b.canonical_replay_sha256);
        assert_eq!(a.final_adapter_state_sha256, b.final_adapter_state_sha256);
        assert_ne!(a.event_log_sha256, b.event_log_sha256);
    }

    #[test]
    fn missing_source_sequence_returns_unknown_not_an_invented_tie_break() {
        let mut event = envelope("p1", 100, 110, 1,
            StreamingReconstructionPayloadV1::Prediction {
                input: prediction("s1", "m1", 100),
            });
        event.source_sequence = None;
        event.envelope_sha256.clear();
        event.envelope_sha256 = envelope_digest(&event).unwrap();
        let report = reconstruct_streaming_adapter_event_time_v1(&input(vec![event])).unwrap();
        assert_eq!(report.status, ReconstructionStatusV1::Unknown);
    }

    #[test]
    fn mixed_clock_domains_return_unknown() {
        let mut event = envelope("p1", 100, 110, 1,
            StreamingReconstructionPayloadV1::Prediction {
                input: prediction("s1", "m1", 100),
            });
        event.clock_domain_id = "other-clock".into();
        event.envelope_sha256.clear();
        event.envelope_sha256 = envelope_digest(&event).unwrap();
        let report = reconstruct_streaming_adapter_event_time_v1(&input(vec![event])).unwrap();
        assert_eq!(report.status, ReconstructionStatusV1::Unknown);
    }

    #[test]
    fn duplicate_event_ids_and_source_sequences_are_rejected() {
        let p = envelope("p1", 100, 110, 1, StreamingReconstructionPayloadV1::Prediction {
            input: prediction("s1", "m1", 100),
        });
        let duplicate_id = envelope("p1", 120, 130, 2, StreamingReconstructionPayloadV1::Prediction {
            input: prediction("s2", "m2", 120),
        });
        assert!(reconstruct_streaming_adapter_event_time_v1(&input(vec![p.clone(), duplicate_id])).is_err());

        let duplicate_seq = envelope("p2", 120, 130, 1, StreamingReconstructionPayloadV1::Prediction {
            input: prediction("s2", "m2", 120),
        });
        assert!(reconstruct_streaming_adapter_event_time_v1(&input(vec![p, duplicate_seq])).is_err());
    }

    #[test]
    fn late_revocation_marks_prior_update_for_new_generation_without_rewriting_receipts() {
        let p = envelope("p1", 100, 110, 1, StreamingReconstructionPayloadV1::Prediction {
            input: prediction("s1", "m1", 100),
        });
        let d = envelope("d1", 120, 130, 2, StreamingReconstructionPayloadV1::LabelDelivery {
            delivery: delivery("s1", "l1", "m1", 120),
        });
        let mut blocked_report = eligibility("l1", 1, "m1", 11.0,
            LabelConsumptionEligibilityV1::Blocked, 140);
        blocked_report.qualified_corrected_value = 11.0;
        let mut correction = envelope("c1", 140, 160, 3,
            StreamingReconstructionPayloadV1::EligibilityCorrection {
                sample_id: "s1".into(),
                label_id: "l1".into(),
                label_revision: 1,
                measurement_event_id: "m1".into(),
                effective_at_unix_seconds: 120,
                report_available_at_unix_seconds: 140,
                report: blocked_report,
            });
        seal_reconstruction_event_v1(&mut correction).unwrap();
        let result = reconstruct_streaming_adapter_event_time_v1(&input(vec![p, d, correction])).unwrap();
        assert_eq!(result.status, ReconstructionStatusV1::RebuildRequired);
        assert_eq!(result.late_corrected_label_ids, vec!["l1"]);
        assert_eq!(result.affected_measurement_event_ids, vec!["m1"]);
        assert!(!result.prior_receipts_mutated);
        assert!(!result.automatic_rebuild_executed);
    }

    #[test]
    fn ingestion_after_knowledge_cutoff_is_excluded_from_generation() {
        let event = envelope("p1", 100, 2000, 1, StreamingReconstructionPayloadV1::Prediction {
            input: prediction("s1", "m1", 100),
        });
        let report = reconstruct_streaming_adapter_event_time_v1(&input(vec![event])).unwrap();
        assert!(report.ordered_event_ids.is_empty());
        assert_eq!(report.accepted_update_count, 0);
    }

    #[test]
    fn corrupted_payload_digest_is_rejected() {
        let mut event = envelope("p1", 100, 110, 1, StreamingReconstructionPayloadV1::Prediction {
            input: prediction("s1", "m1", 100),
        });
        event.payload_sha256 = digest('z');
        event.envelope_sha256.clear();
        event.envelope_sha256 = envelope_digest(&event).unwrap();
        assert!(reconstruct_streaming_adapter_event_time_v1(&input(vec![event])).is_err());
    }
}
