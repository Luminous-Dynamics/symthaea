// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bitemporal calibration-change impact assessment for SEMI-BENCH-001.
//!
//! This is a semiconductor-specific adapter over the K0/K1 reference semantics
//! in the engineering decision-requalification benchmark. It is not an event
//! ingestion service, trust root, certificate authenticator, or control path.
//! A digest binds bytes; it does not prove issuer identity or event authority.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

use super::metrology_qualification_gate::{
    CalibrationCertificateV1, MeasurementEvidenceV1, MeasurementQualificationPolicyV1,
    MeasurementQualificationReportV1, MeasurementQualificationStatusV1,
    qualify_measurement_evidence_v1,
};

pub const CALIBRATION_REQUALIFICATION_INPUT_SCHEMA_V1: &str =
    "luminous.semi-bench.calibration-requalification-input.v1";
pub const CALIBRATION_REQUALIFICATION_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.calibration-requalification-report.v1";
pub const CALIBRATION_CHANGE_EVENT_SCHEMA_V1: &str =
    "luminous.semi-bench.calibration-change-event.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationChangeKindV1 {
    Revoked,
    Superseded,
    Corrected,
    ScopeChanged,
    Reinstated,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationEventQualificationV1 {
    Qualified,
    Unknown,
    Rejected,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationEventInventoryStateV1 {
    CompleteFrozen,
    Incomplete,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationEventOrderingStateV1 {
    Total,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationChangeTargetV1 {
    Certificate {
        certificate_id: String,
        certificate_sha256: String,
    },
    ReferenceStandard {
        reference_standard_id: String,
        reference_certificate_sha256: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationOperatingRangeV1 {
    EntireDeclaredCertificateRange,
    ClosedInterval { lower: f64, upper: f64 },
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationChangeScopeV1 {
    pub target: CalibrationChangeTargetV1,
    pub instrument_id: String,
    pub measurand_id: String,
    pub unit: String,
    pub method_revision: String,
    pub operating_range: CalibrationOperatingRangeV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationChangeEventV1 {
    pub schema: String,
    pub event_id: String,
    pub event_kind: CalibrationChangeKindV1,
    pub source_id: String,
    pub source_payload_sha256: String,
    /// A declared issuer/authority reference; not authenticated by this module.
    pub authority_ref: String,
    /// Owner sequence used only when the source ordering contract is qualified.
    pub source_sequence: Option<u64>,
    /// Valid/world time: when the change applies.
    pub effective_at_unix_seconds: i64,
    /// Knowledge time: when this consumer could legitimately know the change.
    pub observed_at_unix_seconds: i64,
    pub qualification: CalibrationEventQualificationV1,
    /// None means unknown scope, never an empty/no-impact scope.
    pub scope: Option<CalibrationChangeScopeV1>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CalibrationEventFrontierV1 {
    pub inventory_state: CalibrationEventInventoryStateV1,
    pub ordering_state: CalibrationEventOrderingStateV1,
    /// Event IDs actually materialized into the candidate-visible context.
    pub materialized_event_ids: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MeasurementTimeBoundsV1 {
    pub earliest_measured_at_unix_seconds: i64,
    pub latest_measured_at_unix_seconds: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationRequalificationInputV1 {
    pub schema: String,
    pub input_id: String,
    pub measurement: MeasurementEvidenceV1,
    pub calibration_at_k0: CalibrationCertificateV1,
    pub qualification_policy: MeasurementQualificationPolicyV1,
    /// Immutable gate output captured at K0, before later events are applied.
    pub original_qualification_at_k0: MeasurementQualificationReportV1,
    pub measurement_time_bounds: MeasurementTimeBoundsV1,
    pub k0_unix_seconds: i64,
    pub k1_unix_seconds: i64,
    /// Digest of the complete owner-frozen event inventory, not an authenticity proof.
    pub owner_event_inventory_sha256: String,
    /// The full event inventory is evaluator/owner input; candidate visibility is
    /// separately represented by the K0/K1 materialized ID sets below.
    pub owner_events: Vec<CalibrationChangeEventV1>,
    pub k0: CalibrationEventFrontierV1,
    pub k1: CalibrationEventFrontierV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationRequalificationReasonV1 {
    NoApplicableChange,
    OriginalQualificationNotQualified,
    EventInventoryIncomplete,
    EventOrderingUnknown,
    UnknownEventScope,
    MeasurementTimeBoundaryUncertain,
    RelevantEventNotMaterialized,
    EventSourceNotQualified,
    CalibrationRevoked,
    CalibrationSuperseded,
    CalibrationCorrected,
    CalibrationScopeChanged,
    CalibrationReinstatedRequiresNewEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationRequalificationFrontierReportV1 {
    /// Digest of only the evidence admissible at this knowledge frontier.
    pub frontier_input_sha256: String,
    pub status: MeasurementQualificationStatusV1,
    pub downstream_consumption_allowed_at_frontier: bool,
    pub reason_codes: Vec<CalibrationRequalificationReasonV1>,
    pub potentially_applicable_event_ids: Vec<String>,
    pub applied_event_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationRequalificationReportV1 {
    pub schema: String,
    pub input_id: String,
    pub measurement_event_id: String,
    pub calibration_certificate_id: String,
    pub original_k0_qualification_sha256: String,
    pub owner_event_inventory_sha256: String,
    pub canonical_input_sha256: String,
    pub k0_unix_seconds: i64,
    pub k1_unix_seconds: i64,
    /// Reconstructed only from K0-visible information; never rewritten by K1.
    pub as_known_at_k0: CalibrationRequalificationFrontierReportV1,
    /// Current eligibility after K1-visible qualified events and inventory checks.
    pub current_at_k1: CalibrationRequalificationFrontierReportV1,
    /// Digests establish content identity, not issuer authenticity.
    pub source_authenticity_proven: bool,
    pub equipment_control_authority: bool,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct CalibrationRequalificationErrorV1(pub String);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ApplicabilityV1 {
    No,
    Yes,
    Unknown,
}

fn require_text(value: &str, field: &str) -> Result<(), CalibrationRequalificationErrorV1> {
    if value.trim().is_empty() {
        return Err(CalibrationRequalificationErrorV1(format!(
            "{field}: expected non-empty text"
        )));
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(
    value: &T,
) -> Result<String, CalibrationRequalificationErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        CalibrationRequalificationErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

/// Seals the complete owner event inventory in canonical event-ID order.
/// This is a deterministic content digest, not an owner signature.
pub fn seal_calibration_event_inventory_v1(
    events: &[CalibrationChangeEventV1],
) -> Result<String, CalibrationRequalificationErrorV1> {
    let mut canonical = events.to_vec();
    canonical.sort_by(|left, right| left.event_id.cmp(&right.event_id));
    json_sha256(&canonical)
}

fn validate_scope(
    scope: &CalibrationChangeScopeV1,
) -> Result<(), CalibrationRequalificationErrorV1> {
    match &scope.target {
        CalibrationChangeTargetV1::Certificate {
            certificate_id,
            certificate_sha256,
        } => {
            require_text(certificate_id, "scope.target.certificate_id")?;
            if !valid_sha256(certificate_sha256) {
                return Err(CalibrationRequalificationErrorV1(
                    "scope.target.certificate_sha256: expected lowercase SHA-256".into(),
                ));
            }
        }
        CalibrationChangeTargetV1::ReferenceStandard {
            reference_standard_id,
            reference_certificate_sha256,
        } => {
            require_text(reference_standard_id, "scope.target.reference_standard_id")?;
            if !valid_sha256(reference_certificate_sha256) {
                return Err(CalibrationRequalificationErrorV1(
                    "scope.target.reference_certificate_sha256: expected lowercase SHA-256".into(),
                ));
            }
        }
    }

    require_text(&scope.instrument_id, "scope.instrument_id")?;
    require_text(&scope.measurand_id, "scope.measurand_id")?;
    require_text(&scope.unit, "scope.unit")?;
    require_text(&scope.method_revision, "scope.method_revision")?;

    if let CalibrationOperatingRangeV1::ClosedInterval { lower, upper } = scope.operating_range {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(CalibrationRequalificationErrorV1(
                "scope.operating_range: invalid finite closed interval".into(),
            ));
        }
    }
    Ok(())
}

fn validate_event(
    event: &CalibrationChangeEventV1,
) -> Result<(), CalibrationRequalificationErrorV1> {
    if event.schema != CALIBRATION_CHANGE_EVENT_SCHEMA_V1 {
        return Err(CalibrationRequalificationErrorV1(format!(
            "event {} has unsupported schema",
            event.event_id
        )));
    }
    require_text(&event.event_id, "event.event_id")?;
    require_text(&event.source_id, "event.source_id")?;
    require_text(&event.authority_ref, "event.authority_ref")?;
    if !valid_sha256(&event.source_payload_sha256) {
        return Err(CalibrationRequalificationErrorV1(format!(
            "event {} source_payload_sha256: expected lowercase SHA-256",
            event.event_id
        )));
    }
    if let Some(scope) = &event.scope {
        validate_scope(scope)?;
    }
    Ok(())
}

fn validate_frontier(
    frontier: &CalibrationEventFrontierV1,
    cutoff: i64,
    events_by_id: &std::collections::BTreeMap<String, &CalibrationChangeEventV1>,
    name: &str,
) -> Result<(), CalibrationRequalificationErrorV1> {
    let mut seen = BTreeSet::new();
    for event_id in &frontier.materialized_event_ids {
        require_text(event_id, &format!("{name}.materialized_event_id"))?;
        if !seen.insert(event_id) {
            return Err(CalibrationRequalificationErrorV1(format!(
                "{name}: duplicate materialized event ID"
            )));
        }
        let event = events_by_id.get(event_id).ok_or_else(|| {
            CalibrationRequalificationErrorV1(format!(
                "{name}: materialized event {event_id} is absent from owner inventory"
            ))
        })?;
        if event.observed_at_unix_seconds > cutoff {
            return Err(CalibrationRequalificationErrorV1(format!(
                "{name}: event {event_id} is materialized before its observed_at time"
            )));
        }
    }
    Ok(())
}

fn validate_input(
    input: &CalibrationRequalificationInputV1,
) -> Result<(), CalibrationRequalificationErrorV1> {
    if input.schema != CALIBRATION_REQUALIFICATION_INPUT_SCHEMA_V1 {
        return Err(CalibrationRequalificationErrorV1(
            "input.schema: unsupported".into(),
        ));
    }
    require_text(&input.input_id, "input.input_id")?;

    if input.k0_unix_seconds > input.k1_unix_seconds {
        return Err(CalibrationRequalificationErrorV1(
            "knowledge frontiers must satisfy K0 <= K1".into(),
        ));
    }

    let bounds = input.measurement_time_bounds;
    if bounds.earliest_measured_at_unix_seconds > bounds.latest_measured_at_unix_seconds
        || input.measurement.measured_at_unix_seconds < bounds.earliest_measured_at_unix_seconds
        || input.measurement.measured_at_unix_seconds > bounds.latest_measured_at_unix_seconds
    {
        return Err(CalibrationRequalificationErrorV1(
            "measurement time bounds must be ordered and contain measured_at".into(),
        ));
    }
    if input.k0_unix_seconds < bounds.latest_measured_at_unix_seconds {
        return Err(CalibrationRequalificationErrorV1(
            "K0 must not precede the latest possible measurement time".into(),
        ));
    }
    if !valid_sha256(&input.owner_event_inventory_sha256) {
        return Err(CalibrationRequalificationErrorV1(
            "owner_event_inventory_sha256: expected lowercase SHA-256".into(),
        ));
    }

    let mut event_ids = BTreeSet::new();
    let mut events_by_id = std::collections::BTreeMap::new();
    for event in &input.owner_events {
        validate_event(event)?;
        if !event_ids.insert(event.event_id.as_str()) {
            return Err(CalibrationRequalificationErrorV1(
                "owner event inventory contains duplicate event IDs".into(),
            ));
        }
        events_by_id.insert(event.event_id.clone(), event);
    }

    let computed_inventory_sha = seal_calibration_event_inventory_v1(&input.owner_events)?;
    if computed_inventory_sha != input.owner_event_inventory_sha256 {
        return Err(CalibrationRequalificationErrorV1(
            "owner event inventory digest does not match canonical event inventory".into(),
        ));
    }

    validate_frontier(&input.k0, input.k0_unix_seconds, &events_by_id, "k0")?;
    validate_frontier(&input.k1, input.k1_unix_seconds, &events_by_id, "k1")?;

    // Recreate the original gate report from the immutable K0 inputs. A caller
    // cannot substitute a different report while retaining the same event set.
    let recreated = qualify_measurement_evidence_v1(
        &input.measurement,
        Some(&input.calibration_at_k0),
        &input.qualification_policy,
    )
    .map_err(|error| CalibrationRequalificationErrorV1(error.to_string()))?;
    if recreated != input.original_qualification_at_k0 {
        return Err(CalibrationRequalificationErrorV1(
            "original K0 qualification receipt does not replay from its frozen inputs".into(),
        ));
    }
    Ok(())
}

fn scope_applicability(
    event_scope: Option<&CalibrationChangeScopeV1>,
    input: &CalibrationRequalificationInputV1,
) -> ApplicabilityV1 {
    let Some(scope) = event_scope else {
        return ApplicabilityV1::Unknown;
    };

    let certificate_target = matches!(&scope.target, CalibrationChangeTargetV1::Certificate { .. });
    let target_matches = match &scope.target {
        CalibrationChangeTargetV1::Certificate {
            certificate_id,
            certificate_sha256,
        } => {
            certificate_id == &input.calibration_at_k0.certificate_id
                && certificate_sha256 == &input.calibration_at_k0.certificate_sha256
        }
        CalibrationChangeTargetV1::ReferenceStandard {
            reference_standard_id,
            reference_certificate_sha256,
        } => {
            reference_standard_id == &input.calibration_at_k0.reference_standard_id
                && input
                    .calibration_at_k0
                    .reference_certificate_sha256
                    .as_ref()
                    == Some(reference_certificate_sha256)
        }
    };
    if !target_matches {
        return ApplicabilityV1::No;
    }

    let dimensions_match = scope.instrument_id == input.measurement.instrument_id
        && scope.measurand_id == input.measurement.measurand_id
        && scope.unit == input.measurement.unit
        && scope.method_revision == input.measurement.method_revision;
    if !dimensions_match {
        // A direct certificate-ID match with contradictory dimensions is an
        // inconsistent scope declaration. A shared reference standard may
        // legitimately cover other exact measurement dimensions.
        return if certificate_target {
            ApplicabilityV1::Unknown
        } else {
            ApplicabilityV1::No
        };
    }

    match scope.operating_range {
        CalibrationOperatingRangeV1::EntireDeclaredCertificateRange => ApplicabilityV1::Yes,
        CalibrationOperatingRangeV1::ClosedInterval { lower, upper } => {
            if input.measurement.corrected_value >= lower
                && input.measurement.corrected_value <= upper
            {
                ApplicabilityV1::Yes
            } else {
                ApplicabilityV1::No
            }
        }
        CalibrationOperatingRangeV1::Unknown => ApplicabilityV1::Unknown,
    }
}

fn temporal_applicability(
    event: &CalibrationChangeEventV1,
    input: &CalibrationRequalificationInputV1,
) -> ApplicabilityV1 {
    let bounds = input.measurement_time_bounds;
    if event.effective_at_unix_seconds > bounds.latest_measured_at_unix_seconds {
        ApplicabilityV1::No
    } else if event.effective_at_unix_seconds <= bounds.earliest_measured_at_unix_seconds {
        ApplicabilityV1::Yes
    } else {
        ApplicabilityV1::Unknown
    }
}

fn set_unknown(status: &mut MeasurementQualificationStatusV1) {
    if *status != MeasurementQualificationStatusV1::Rejected {
        *status = MeasurementQualificationStatusV1::Unknown;
    }
}

fn frontier_input_sha256(
    input: &CalibrationRequalificationInputV1,
    frontier: &CalibrationEventFrontierV1,
    cutoff: i64,
) -> Result<String, CalibrationRequalificationErrorV1> {
    let mut visible_events: Vec<CalibrationChangeEventV1> = input
        .owner_events
        .iter()
        .filter(|event| event.observed_at_unix_seconds <= cutoff)
        .cloned()
        .collect();
    visible_events.sort_by(|left, right| left.event_id.cmp(&right.event_id));
    let mut materialized_event_ids = frontier.materialized_event_ids.clone();
    materialized_event_ids.sort();
    json_sha256(&(
        &input.input_id,
        &input.original_qualification_at_k0.report_sha256,
        input.measurement_time_bounds,
        cutoff,
        frontier.inventory_state,
        frontier.ordering_state,
        materialized_event_ids,
        visible_events,
    ))
}

fn assess_frontier(
    input: &CalibrationRequalificationInputV1,
    frontier: &CalibrationEventFrontierV1,
    cutoff: i64,
) -> Result<CalibrationRequalificationFrontierReportV1, CalibrationRequalificationErrorV1> {
    let initial_status = input.original_qualification_at_k0.status;
    let mut status = initial_status;
    let mut reasons = BTreeSet::new();
    let mut potentially_applicable = BTreeSet::new();
    let mut applied = Vec::new();

    if status != MeasurementQualificationStatusV1::Qualified {
        reasons.insert(CalibrationRequalificationReasonV1::OriginalQualificationNotQualified);
    }
    if frontier.inventory_state != CalibrationEventInventoryStateV1::CompleteFrozen {
        reasons.insert(CalibrationRequalificationReasonV1::EventInventoryIncomplete);
        set_unknown(&mut status);
    }
    if frontier.ordering_state != CalibrationEventOrderingStateV1::Total {
        reasons.insert(CalibrationRequalificationReasonV1::EventOrderingUnknown);
        set_unknown(&mut status);
    }

    let materialized: BTreeSet<&str> = frontier
        .materialized_event_ids
        .iter()
        .map(String::as_str)
        .collect();

    let mut relevant_events: Vec<&CalibrationChangeEventV1> = Vec::new();

    for event in input
        .owner_events
        .iter()
        .filter(|event| event.observed_at_unix_seconds <= cutoff)
    {
        let temporal = temporal_applicability(event, input);
        if temporal == ApplicabilityV1::No {
            continue;
        }

        let scope = scope_applicability(event.scope.as_ref(), input);
        if scope == ApplicabilityV1::No {
            continue;
        }

        potentially_applicable.insert(event.event_id.clone());

        if temporal == ApplicabilityV1::Unknown {
            reasons.insert(CalibrationRequalificationReasonV1::MeasurementTimeBoundaryUncertain);
            set_unknown(&mut status);
            continue;
        }
        if scope == ApplicabilityV1::Unknown {
            reasons.insert(CalibrationRequalificationReasonV1::UnknownEventScope);
            set_unknown(&mut status);
            continue;
        }
        if !materialized.contains(event.event_id.as_str()) {
            reasons.insert(CalibrationRequalificationReasonV1::RelevantEventNotMaterialized);
            set_unknown(&mut status);
            continue;
        }
        if event.qualification != CalibrationEventQualificationV1::Qualified {
            reasons.insert(CalibrationRequalificationReasonV1::EventSourceNotQualified);
            set_unknown(&mut status);
            continue;
        }
        relevant_events.push(event);
    }

    // Multiple changes to the same calibration lineage require a qualified
    // total owner order. A single applicable event needs no relative ordering.
    if relevant_events.len() > 1 {
        let same_source = relevant_events
            .iter()
            .all(|event| event.source_id == relevant_events[0].source_id);
        let mut sequences = BTreeSet::new();
        let all_sequenced = relevant_events.iter().all(|event| {
            event
                .source_sequence
                .is_some_and(|sequence| sequences.insert(sequence))
        });
        if !same_source || !all_sequenced {
            reasons.insert(CalibrationRequalificationReasonV1::EventOrderingUnknown);
            set_unknown(&mut status);
            relevant_events.clear();
        }
    }

    relevant_events.sort_by_key(|event| event.source_sequence.unwrap_or(u64::MAX));
    for event in relevant_events {
        applied.push(event.event_id.clone());
        match event.event_kind {
            CalibrationChangeKindV1::Revoked => {
                status = MeasurementQualificationStatusV1::Rejected;
                reasons.insert(CalibrationRequalificationReasonV1::CalibrationRevoked);
            }
            CalibrationChangeKindV1::Superseded => {
                set_unknown(&mut status);
                reasons.insert(CalibrationRequalificationReasonV1::CalibrationSuperseded);
            }
            CalibrationChangeKindV1::Corrected => {
                set_unknown(&mut status);
                reasons.insert(CalibrationRequalificationReasonV1::CalibrationCorrected);
            }
            CalibrationChangeKindV1::ScopeChanged => {
                set_unknown(&mut status);
                reasons.insert(CalibrationRequalificationReasonV1::CalibrationScopeChanged);
            }
            CalibrationChangeKindV1::Reinstated => {
                // A qualified reinstatement supersedes the earlier revocation
                // state for current status, but does not recreate a certificate
                // or requalify this historical measurement. Preserve a separate
                // original K0 rejection when one already existed.
                status = if initial_status == MeasurementQualificationStatusV1::Rejected {
                    MeasurementQualificationStatusV1::Rejected
                } else {
                    MeasurementQualificationStatusV1::Unknown
                };
                reasons.insert(
                    CalibrationRequalificationReasonV1::CalibrationReinstatedRequiresNewEvidence,
                );
            }
        }
    }

    if reasons.is_empty() {
        reasons.insert(CalibrationRequalificationReasonV1::NoApplicableChange);
    }

    Ok(CalibrationRequalificationFrontierReportV1 {
        frontier_input_sha256: frontier_input_sha256(input, frontier, cutoff)?,
        status,
        downstream_consumption_allowed_at_frontier: status
            == MeasurementQualificationStatusV1::Qualified,
        reason_codes: reasons.into_iter().collect(),
        potentially_applicable_event_ids: potentially_applicable.into_iter().collect(),
        applied_event_ids: applied,
    })
}

fn canonical_input_sha256(
    input: &CalibrationRequalificationInputV1,
) -> Result<String, CalibrationRequalificationErrorV1> {
    let mut canonical = input.clone();
    canonical
        .owner_events
        .sort_by(|left, right| left.event_id.cmp(&right.event_id));
    canonical.k0.materialized_event_ids.sort();
    canonical.k1.materialized_event_ids.sort();
    json_sha256(&canonical)
}

fn report_digest(
    report: &CalibrationRequalificationReportV1,
) -> Result<String, CalibrationRequalificationErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

/// Replays K0 from K0-visible events, then independently computes K1 current
/// eligibility. Later events never mutate the as-known K0 assessment.
pub fn requalify_calibration_measurement_v1(
    input: &CalibrationRequalificationInputV1,
) -> Result<CalibrationRequalificationReportV1, CalibrationRequalificationErrorV1> {
    validate_input(input)?;

    let as_known_at_k0 = assess_frontier(input, &input.k0, input.k0_unix_seconds)?;
    let current_at_k1 = assess_frontier(input, &input.k1, input.k1_unix_seconds)?;

    let mut report = CalibrationRequalificationReportV1 {
        schema: CALIBRATION_REQUALIFICATION_REPORT_SCHEMA_V1.into(),
        input_id: input.input_id.clone(),
        measurement_event_id: input.measurement.measurement_event_id.clone(),
        calibration_certificate_id: input.calibration_at_k0.certificate_id.clone(),
        original_k0_qualification_sha256: input.original_qualification_at_k0.report_sha256.clone(),
        owner_event_inventory_sha256: input.owner_event_inventory_sha256.clone(),
        canonical_input_sha256: canonical_input_sha256(input)?,
        k0_unix_seconds: input.k0_unix_seconds,
        k1_unix_seconds: input.k1_unix_seconds,
        as_known_at_k0,
        current_at_k1,
        source_authenticity_proven: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::metrology_qualification_gate::{
        CALIBRATION_CERTIFICATE_SCHEMA_V1, MEASUREMENT_EVIDENCE_SCHEMA_V1,
        MeasurementUncertaintyComponentV1, QUALIFICATION_POLICY_SCHEMA_V1,
        TraceabilityDeclarationV1, UncertaintyComponentKindV1,
    };

    fn policy() -> MeasurementQualificationPolicyV1 {
        MeasurementQualificationPolicyV1 {
            schema: QUALIFICATION_POLICY_SCHEMA_V1.into(),
            policy_id: "strict-metrology-evidence.v1".into(),
            revision: 1,
            maximum_standard_uncertainty: 0.20,
            required_coverage_factor: 2.0,
            require_claimed_traceability: true,
            require_reference_certificate_digest: true,
        }
    }

    fn certificate() -> CalibrationCertificateV1 {
        CalibrationCertificateV1 {
            schema: CALIBRATION_CERTIFICATE_SCHEMA_V1.into(),
            certificate_id: "cal-cert-001".into(),
            certificate_sha256: "a".repeat(64),
            issuing_authority_ref: "metrology-lab-01".into(),
            instrument_id: "ellipsometer-07".into(),
            measurand_id: "film-thickness".into(),
            unit: "nm".into(),
            method_revision: "film-method-3".into(),
            reference_standard_id: "traceable-step-height-std-02".into(),
            reference_certificate_sha256: Some("b".repeat(64)),
            traceability_declaration: TraceabilityDeclarationV1::ClaimedTraceable,
            calibration_performed_at_unix_seconds: 1_800_000_000,
            issued_at_unix_seconds: 1_800_000_100,
            valid_from_unix_seconds: 1_800_000_100,
            valid_until_unix_seconds: 1_900_000_000,
            revoked_at_unix_seconds: None,
            correction: -0.05,
            calibration_standard_uncertainty: 0.04,
            coverage_factor: 2.0,
        }
    }

    fn measurement() -> MeasurementEvidenceV1 {
        MeasurementEvidenceV1 {
            schema: MEASUREMENT_EVIDENCE_SCHEMA_V1.into(),
            measurement_event_id: "measurement-event-001".into(),
            source_record_sha256: "c".repeat(64),
            instrument_id: "ellipsometer-07".into(),
            measurand_id: "film-thickness".into(),
            unit: "nm".into(),
            method_revision: "film-method-3".into(),
            measured_at_unix_seconds: 1_850_000_000,
            raw_value: 12.50,
            correction_applied: -0.05,
            corrected_value: 12.45,
            uncertainty_components: vec![
                MeasurementUncertaintyComponentV1 {
                    component_id: "repeatability".into(),
                    source_ref: "msa-study-2026-11".into(),
                    source_sha256: "d".repeat(64),
                    kind: UncertaintyComponentKindV1::TypeA,
                    standard_uncertainty: 0.03,
                },
                MeasurementUncertaintyComponentV1 {
                    component_id: "resolution".into(),
                    source_ref: "instrument-spec-rev-7".into(),
                    source_sha256: "e".repeat(64),
                    kind: UncertaintyComponentKindV1::TypeB,
                    standard_uncertainty: 0.02,
                },
            ],
        }
    }

    fn scope() -> CalibrationChangeScopeV1 {
        CalibrationChangeScopeV1 {
            target: CalibrationChangeTargetV1::Certificate {
                certificate_id: "cal-cert-001".into(),
                certificate_sha256: "a".repeat(64),
            },
            instrument_id: "ellipsometer-07".into(),
            measurand_id: "film-thickness".into(),
            unit: "nm".into(),
            method_revision: "film-method-3".into(),
            operating_range: CalibrationOperatingRangeV1::EntireDeclaredCertificateRange,
        }
    }

    fn event(
        event_id: &str,
        kind: CalibrationChangeKindV1,
        effective_at: i64,
        observed_at: i64,
        source_sequence: u64,
    ) -> CalibrationChangeEventV1 {
        CalibrationChangeEventV1 {
            schema: CALIBRATION_CHANGE_EVENT_SCHEMA_V1.into(),
            event_id: event_id.into(),
            event_kind: kind,
            source_id: "metrology-lab-01".into(),
            source_payload_sha256: "f".repeat(64),
            authority_ref: "calibration-owner-01".into(),
            source_sequence: Some(source_sequence),
            effective_at_unix_seconds: effective_at,
            observed_at_unix_seconds: observed_at,
            qualification: CalibrationEventQualificationV1::Qualified,
            scope: Some(scope()),
        }
    }

    fn base_input() -> CalibrationRequalificationInputV1 {
        let measured = measurement();
        let calibration = certificate();
        let policy_value = policy();
        let original =
            qualify_measurement_evidence_v1(&measured, Some(&calibration), &policy_value).unwrap();
        CalibrationRequalificationInputV1 {
            schema: CALIBRATION_REQUALIFICATION_INPUT_SCHEMA_V1.into(),
            input_id: "cal-requal-001".into(),
            measurement: measured,
            calibration_at_k0: calibration,
            qualification_policy: policy_value,
            original_qualification_at_k0: original,
            measurement_time_bounds: MeasurementTimeBoundsV1 {
                earliest_measured_at_unix_seconds: 1_850_000_000,
                latest_measured_at_unix_seconds: 1_850_000_000,
            },
            k0_unix_seconds: 1_860_000_000,
            k1_unix_seconds: 1_880_000_000,
            owner_event_inventory_sha256: seal_calibration_event_inventory_v1(&[]).unwrap(),
            owner_events: vec![],
            k0: CalibrationEventFrontierV1 {
                inventory_state: CalibrationEventInventoryStateV1::CompleteFrozen,
                ordering_state: CalibrationEventOrderingStateV1::Total,
                materialized_event_ids: vec![],
            },
            k1: CalibrationEventFrontierV1 {
                inventory_state: CalibrationEventInventoryStateV1::CompleteFrozen,
                ordering_state: CalibrationEventOrderingStateV1::Total,
                materialized_event_ids: vec![],
            },
        }
    }

    fn install_events(
        input: &mut CalibrationRequalificationInputV1,
        events: Vec<CalibrationChangeEventV1>,
    ) {
        input.owner_event_inventory_sha256 = seal_calibration_event_inventory_v1(&events).unwrap();
        input.k0.materialized_event_ids = events
            .iter()
            .filter(|event| event.observed_at_unix_seconds <= input.k0_unix_seconds)
            .map(|event| event.event_id.clone())
            .collect();
        input.k1.materialized_event_ids = events
            .iter()
            .filter(|event| event.observed_at_unix_seconds <= input.k1_unix_seconds)
            .map(|event| event.event_id.clone())
            .collect();
        input.owner_events = events;
    }

    #[test]
    fn late_retroactive_revocation_preserves_k0_and_rejects_current_use() {
        let mut input = base_input();
        install_events(
            &mut input,
            vec![event(
                "revoke-001",
                CalibrationChangeKindV1::Revoked,
                1_840_000_000,
                1_870_000_000,
                1,
            )],
        );

        let report = requalify_calibration_measurement_v1(&input).unwrap();
        assert_eq!(
            report.as_known_at_k0.status,
            MeasurementQualificationStatusV1::Qualified
        );
        assert_eq!(
            report.current_at_k1.status,
            MeasurementQualificationStatusV1::Rejected
        );
        assert!(
            !report
                .current_at_k1
                .downstream_consumption_allowed_at_frontier
        );
        assert!(!report.source_authenticity_proven);
        assert!(!report.equipment_control_authority);
    }

    #[test]
    fn post_k0_events_do_not_change_the_k0_frontier_digest() {
        let baseline = base_input();
        let baseline_report = requalify_calibration_measurement_v1(&baseline).unwrap();

        let mut with_late_event = base_input();
        install_events(
            &mut with_late_event,
            vec![event(
                "late-revocation",
                CalibrationChangeKindV1::Revoked,
                1_840_000_000,
                1_870_000_000,
                1,
            )],
        );
        let late_report = requalify_calibration_measurement_v1(&with_late_event).unwrap();

        assert_eq!(
            baseline_report.as_known_at_k0.frontier_input_sha256,
            late_report.as_known_at_k0.frontier_input_sha256
        );
        assert_eq!(
            baseline_report.as_known_at_k0.status,
            late_report.as_known_at_k0.status
        );
        assert_ne!(
            baseline_report.current_at_k1.frontier_input_sha256,
            late_report.current_at_k1.frontier_input_sha256
        );
        assert_eq!(
            late_report.current_at_k1.status,
            MeasurementQualificationStatusV1::Rejected
        );
        assert_ne!(baseline_report.report_sha256, late_report.report_sha256);
    }

    #[test]
    fn revocation_observed_before_k0_blocks_both_frontiers() {
        let mut input = base_input();
        install_events(
            &mut input,
            vec![event(
                "revoke-002",
                CalibrationChangeKindV1::Revoked,
                1_840_000_000,
                1_850_000_500,
                1,
            )],
        );

        let report = requalify_calibration_measurement_v1(&input).unwrap();
        assert_eq!(
            report.as_known_at_k0.status,
            MeasurementQualificationStatusV1::Rejected
        );
        assert_eq!(
            report.current_at_k1.status,
            MeasurementQualificationStatusV1::Rejected
        );
    }

    #[test]
    fn revocation_effective_after_measurement_does_not_invalidate_it() {
        let mut input = base_input();
        install_events(
            &mut input,
            vec![event(
                "revoke-future",
                CalibrationChangeKindV1::Revoked,
                1_860_000_000,
                1_861_000_000,
                1,
            )],
        );

        let report = requalify_calibration_measurement_v1(&input).unwrap();
        assert_eq!(
            report.as_known_at_k0.status,
            MeasurementQualificationStatusV1::Qualified
        );
        assert_eq!(
            report.current_at_k1.status,
            MeasurementQualificationStatusV1::Qualified
        );
        assert!(
            report
                .current_at_k1
                .downstream_consumption_allowed_at_frontier
        );
    }

    #[test]
    fn different_certificate_scope_does_not_invalidate_measurement() {
        let mut input = base_input();
        let mut unrelated = event(
            "revoke-other",
            CalibrationChangeKindV1::Revoked,
            1_840_000_000,
            1_870_000_000,
            1,
        );
        unrelated.scope.as_mut().unwrap().target = CalibrationChangeTargetV1::Certificate {
            certificate_id: "other-certificate".into(),
            certificate_sha256: "9".repeat(64),
        };
        install_events(&mut input, vec![unrelated]);

        let report = requalify_calibration_measurement_v1(&input).unwrap();
        assert_eq!(
            report.current_at_k1.status,
            MeasurementQualificationStatusV1::Qualified
        );
        assert!(report.current_at_k1.applied_event_ids.is_empty());
    }

    #[test]
    fn corrected_calibration_requires_requalification() {
        let mut input = base_input();
        install_events(
            &mut input,
            vec![event(
                "certificate-corrected",
                CalibrationChangeKindV1::Corrected,
                1_840_000_000,
                1_870_000_000,
                1,
            )],
        );

        let report = requalify_calibration_measurement_v1(&input).unwrap();
        assert_eq!(
            report.current_at_k1.status,
            MeasurementQualificationStatusV1::Unknown
        );
        assert!(
            report
                .current_at_k1
                .reason_codes
                .contains(&CalibrationRequalificationReasonV1::CalibrationCorrected)
        );
    }

    #[test]
    fn direct_certificate_scope_with_contradictory_dimensions_is_unknown() {
        let mut input = base_input();
        let mut conflicting = event(
            "certificate-scope-conflict",
            CalibrationChangeKindV1::Revoked,
            1_840_000_000,
            1_870_000_000,
            1,
        );
        conflicting.scope.as_mut().unwrap().unit = "um".into();
        install_events(&mut input, vec![conflicting]);

        let report = requalify_calibration_measurement_v1(&input).unwrap();
        assert_eq!(
            report.current_at_k1.status,
            MeasurementQualificationStatusV1::Unknown
        );
        assert!(
            report
                .current_at_k1
                .reason_codes
                .contains(&CalibrationRequalificationReasonV1::UnknownEventScope)
        );
    }

    #[test]
    fn unknown_scope_is_not_treated_as_empty_scope() {
        let mut input = base_input();
        let mut unknown = event(
            "revoke-unknown-scope",
            CalibrationChangeKindV1::Revoked,
            1_840_000_000,
            1_870_000_000,
            1,
        );
        unknown.scope = None;
        install_events(&mut input, vec![unknown]);

        let report = requalify_calibration_measurement_v1(&input).unwrap();
        assert_eq!(
            report.current_at_k1.status,
            MeasurementQualificationStatusV1::Unknown
        );
        assert!(
            report.current_at_k1.reason_codes.contains(
                &CalibrationRequalificationReasonV1::UnknownEventScope
            )
        );
    }

    #[test]
    fn relevant_event_missing_from_candidate_context_is_indeterminate() {
        let mut input = base_input();
        install_events(
            &mut input,
            vec![event(
                "revoke-not-materialized",
                CalibrationChangeKindV1::Revoked,
                1_840_000_000,
                1_870_000_000,
                1,
            )],
        );
        input.k1.materialized_event_ids.clear();

        let report = requalify_calibration_measurement_v1(&input).unwrap();
        assert_eq!(
            report.current_at_k1.status,
            MeasurementQualificationStatusV1::Unknown
        );
        assert!(
            report
                .current_at_k1
                .reason_codes
                .contains(&CalibrationRequalificationReasonV1::RelevantEventNotMaterialized)
        );
    }

    #[test]
    fn incomplete_owner_inventory_blocks_current_support() {
        let mut input = base_input();
        input.k1.inventory_state = CalibrationEventInventoryStateV1::Incomplete;

        let report = requalify_calibration_measurement_v1(&input).unwrap();
        assert_eq!(
            report.current_at_k1.status,
            MeasurementQualificationStatusV1::Unknown
        );
    }

    #[test]
    fn measurement_time_interval_straddling_effective_time_is_unknown() {
        let mut input = base_input();
        input.measurement_time_bounds = MeasurementTimeBoundsV1 {
            earliest_measured_at_unix_seconds: 1_849_999_900,
            latest_measured_at_unix_seconds: 1_850_000_100,
        };
        install_events(
            &mut input,
            vec![event(
                "revoke-boundary",
                CalibrationChangeKindV1::Revoked,
                1_850_000_000,
                1_870_000_000,
                1,
            )],
        );

        let report = requalify_calibration_measurement_v1(&input).unwrap();
        assert_eq!(
            report.current_at_k1.status,
            MeasurementQualificationStatusV1::Unknown
        );
        assert!(
            report
                .current_at_k1
                .reason_codes
                .contains(&CalibrationRequalificationReasonV1::MeasurementTimeBoundaryUncertain)
        );
    }

    #[test]
    fn unknown_source_qualification_cannot_revoke_or_preserve_support() {
        let mut input = base_input();
        let mut unqualified = event(
            "revoke-unqualified",
            CalibrationChangeKindV1::Revoked,
            1_840_000_000,
            1_870_000_000,
            1,
        );
        unqualified.qualification = CalibrationEventQualificationV1::Unknown;
        install_events(&mut input, vec![unqualified]);

        let report = requalify_calibration_measurement_v1(&input).unwrap();
        assert_eq!(
            report.current_at_k1.status,
            MeasurementQualificationStatusV1::Unknown
        );
        assert!(
            report
                .current_at_k1
                .reason_codes
                .contains(&CalibrationRequalificationReasonV1::EventSourceNotQualified)
        );
    }

    #[test]
    fn reinstatement_does_not_automatically_requalify_old_measurement() {
        let mut input = base_input();
        install_events(
            &mut input,
            vec![
                event(
                    "revoke-before-reinstate",
                    CalibrationChangeKindV1::Revoked,
                    1_840_000_000,
                    1_841_000_000,
                    1,
                ),
                event(
                    "reinstate-later",
                    CalibrationChangeKindV1::Reinstated,
                    1_842_000_000,
                    1_843_000_000,
                    2,
                ),
            ],
        );

        let report = requalify_calibration_measurement_v1(&input).unwrap();
        assert_eq!(
            report.current_at_k1.status,
            MeasurementQualificationStatusV1::Unknown
        );
        assert!(report.current_at_k1.reason_codes.contains(
            &CalibrationRequalificationReasonV1::CalibrationReinstatedRequiresNewEvidence
        ));
    }

    #[test]
    fn reference_standard_revocation_propagates_to_matching_certificate() {
        let mut input = base_input();
        let mut shared = event(
            "reference-standard-revoke",
            CalibrationChangeKindV1::Revoked,
            1_840_000_000,
            1_870_000_000,
            1,
        );
        shared.scope.as_mut().unwrap().target = CalibrationChangeTargetV1::ReferenceStandard {
            reference_standard_id: "traceable-step-height-std-02".into(),
            reference_certificate_sha256: "b".repeat(64),
        };
        install_events(&mut input, vec![shared]);

        let report = requalify_calibration_measurement_v1(&input).unwrap();
        assert_eq!(
            report.current_at_k1.status,
            MeasurementQualificationStatusV1::Rejected
        );
    }

    #[test]
    fn multiple_changes_need_unique_ordered_owner_sequences() {
        let mut input = base_input();
        let mut second = event(
            "revoke-second",
            CalibrationChangeKindV1::Revoked,
            1_841_000_000,
            1_842_000_000,
            1,
        );
        second.source_sequence = Some(1);
        install_events(
            &mut input,
            vec![
                event(
                    "supersede-first",
                    CalibrationChangeKindV1::Superseded,
                    1_840_000_000,
                    1_841_000_000,
                    1,
                ),
                second,
            ],
        );

        let report = requalify_calibration_measurement_v1(&input).unwrap();
        assert_eq!(
            report.current_at_k1.status,
            MeasurementQualificationStatusV1::Unknown
        );
        assert!(report.current_at_k1.applied_event_ids.is_empty());
    }

    #[test]
    fn event_input_order_does_not_change_receipts() {
        let first = event(
            "revoke-ordered",
            CalibrationChangeKindV1::Revoked,
            1_840_000_000,
            1_841_000_000,
            1,
        );
        let second = event(
            "reinstate-ordered",
            CalibrationChangeKindV1::Reinstated,
            1_842_000_000,
            1_843_000_000,
            2,
        );

        let mut a = base_input();
        install_events(&mut a, vec![first.clone(), second.clone()]);
        let mut b = base_input();
        install_events(&mut b, vec![second, first]);

        let report_a = requalify_calibration_measurement_v1(&a).unwrap();
        let report_b = requalify_calibration_measurement_v1(&b).unwrap();
        assert_eq!(
            report_a.canonical_input_sha256,
            report_b.canonical_input_sha256
        );
        assert_eq!(report_a.report_sha256, report_b.report_sha256);
    }

    #[test]
    fn tampered_original_k0_report_is_rejected() {
        let mut input = base_input();
        input.original_qualification_at_k0.report_sha256 = "0".repeat(64);
        assert!(requalify_calibration_measurement_v1(&input).is_err());
    }
}
