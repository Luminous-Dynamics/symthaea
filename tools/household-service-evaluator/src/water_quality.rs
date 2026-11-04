//! Source-to-use water qualification record validation.
//!
//! This module checks internal identity, scope, currentness and invalidation
//! consistency. It does not test water against SANS limits, authenticate a
//! laboratory, validate a treatment process, or establish physical safety.
//! All timestamps used for temporal comparisons must be UTC RFC3339 strings
//! ending in Z; callers should normalize offsets before evaluation.

use crate::EvaluationError;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WaterSourceClass { Municipal, Borehole, Tanker, Rainwater, Greywater, SurfaceWater, Other, Unknown }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WaterStreamKind { DiscreteBatch, ContinuousStream, DerivedBlend, Unknown }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TreatmentOperatingState { QualifiedState, UnqualifiedState, Bypassed, Faulted, Maintenance, Unknown, NotApplicable }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WaterUseClass { Drinking, FoodPreparation, Bathing, Laundry, ToiletFlushing, Irrigation, Cleaning, IndustrialNonpotable, OtherDeclared }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplePointRelationship { SamePhysicalPoint, ValidatedTransfer, DifferentPointNoTransfer, Unknown }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MonitoringStatus { Current, Due, Overdue, Unknown, NotApplicable }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum WaterQualificationStatus {
    QualifiedForDeclaredUse, KnownNonCompliance, UnknownEvidenceInsufficient,
    ExpiredOrMonitoringOverdue, SourceOrSamplePointMismatch,
    TreatmentStateUnqualified, CorrectiveActionPending,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterResultState { Numeric, BelowDetection, AboveRange, Qualitative, NotTested, Invalid }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WaterEventType {
    SourceSwitch, BlendCreated, TreatmentBypass, TreatmentFault, Maintenance,
    MediaReplacement, StorageStagnation, SamplePointChange, MonitoringOverdue,
    NonCompliantResult, CorrectiveActionStarted, CorrectiveActionCompleted,
    RiskProfileChanged, Other,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WaterSource {
    pub source_id: String,
    pub source_class: WaterSourceClass,
    pub system_id: String,
    pub system_revision: String,
    #[serde(default)]
    pub upstream_source_ids: Vec<String>,
    pub provider_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WaterStream {
    pub stream_id: String,
    pub stream_kind: WaterStreamKind,
    pub observed_from: String,
    pub observed_until: Option<String>,
    pub storage_vessel_id: Option<String>,
    pub residence_seconds: Option<u64>,
    pub storage_state: String,
    #[serde(default)]
    pub blend_parent_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WaterTreatment {
    pub train_id: String,
    pub configuration_revision: String,
    pub operating_state: TreatmentOperatingState,
    pub media_or_component_revision: Option<String>,
    pub last_maintenance_at: Option<String>,
    #[serde(default)]
    pub active_alarm_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IntendedWaterUse {
    pub use_class: WaterUseClass,
    pub use_profile_id: String,
    pub use_profile_revision: String,
    pub jurisdiction: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WaterDeliveryBoundary {
    pub boundary_id: String,
    pub boundary_kind: String,
    pub sample_point_id: String,
    pub sample_point_relationship: SamplePointRelationship,
    pub transfer_validation_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WaterQualityProfile {
    pub standard_id: String,
    pub standard_edition: String,
    pub risk_assessment_id: String,
    pub risk_assessment_revision: String,
    pub monitoring_programme_id: String,
    pub monitoring_programme_revision: String,
    pub monitoring_status: MonitoringStatus,
    #[serde(default)]
    pub required_parameter_ids: Vec<String>,
    pub next_required_review_at: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WaterParameterResult {
    pub parameter_id: String,
    pub result_state: ParameterResultState,
    pub value: Option<f64>,
    pub unit: String,
    pub detection_or_reporting_limit: Option<f64>,
    pub uncertainty: Option<f64>,
    pub qualifier: Option<String>,
    #[serde(default)]
    pub quality_control_flags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WaterSampleObservation {
    pub sample_id: String,
    pub sample_point_id: String,
    pub collected_at: String,
    pub collector_id: Option<String>,
    pub laboratory_or_provider_id: Option<String>,
    pub method_id: String,
    pub method_revision: Option<String>,
    pub chain_of_custody_digest_sha256: Option<String>,
    pub parameter_results: Vec<WaterParameterResult>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WaterQualificationDecision {
    pub status: WaterQualificationStatus,
    pub reason_codes: Vec<String>,
    pub decided_at: String,
    pub valid_from: Option<String>,
    pub valid_until: Option<String>,
    pub supporting_sample_ids: Vec<String>,
    pub incident_id: Option<String>,
    pub corrective_action_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WaterQualificationEvent {
    pub event_id: String,
    pub event_type: WaterEventType,
    pub occurred_at: String,
    #[serde(default)]
    pub affected_stream_ids: Vec<String>,
    pub details_digest_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WaterEvidenceLineage {
    pub source_record_ids: Vec<String>,
    pub evaluator_id: String,
    pub evaluator_revision: String,
    pub source_repository_commit: Option<String>,
    pub synthetic_only: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WaterUseQualificationV1 {
    pub schema_version: String,
    pub qualification_id: String,
    pub source: WaterSource,
    pub stream: WaterStream,
    pub treatment: WaterTreatment,
    pub intended_use: IntendedWaterUse,
    pub delivery_boundary: WaterDeliveryBoundary,
    pub quality_profile: WaterQualityProfile,
    pub sample_observations: Vec<WaterSampleObservation>,
    pub decision: WaterQualificationDecision,
    pub event_history: Vec<WaterQualificationEvent>,
    pub evidence_lineage: WaterEvidenceLineage,
}

/// Summary of record consistency only. It does not establish water safety.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct WaterUseQualificationValidation {
    pub qualification_id: String,
    pub declared_status: WaterQualificationStatus,
    pub intended_use: WaterUseClass,
    pub supporting_sample_count: usize,
    pub synthetic_only: bool,
    pub qualification_record_consistent: bool,
    pub physical_water_safety_established: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct UtcInstant { year: u16, month: u8, day: u8, hour: u8, minute: u8, second: u8, nanosecond: u32 }

fn invalid(path: &'static str, message: impl Into<String>) -> EvaluationError {
    EvaluationError::new(path, message)
}

fn nonempty(path: &'static str, value: &str) -> Result<(), EvaluationError> {
    if value.trim().is_empty() { return Err(invalid(path, "must not be empty")); }
    Ok(())
}

fn is_leap_year(year: u16) -> bool { year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) }

fn parse_utc_instant(value: &str, path: &'static str) -> Result<UtcInstant, EvaluationError> {
    if !value.ends_with('Z') { return Err(invalid(path, "timestamp must be normalized to UTC and end in Z")); }
    let no_z = &value[..value.len() - 1];
    let (date, time) = no_z.split_once('T').ok_or_else(|| invalid(path, "expected RFC3339 date-time"))?;
    let date_bytes = date.as_bytes();
    if date_bytes.len() != 10
        || date_bytes[4] != b'-'
        || date_bytes[7] != b'-'
        || !date_bytes.iter().enumerate().all(|(i, b)| matches!(i, 4 | 7) || b.is_ascii_digit())
    {
        return Err(invalid(path, "date must use ASCII YYYY-MM-DD"));
    }
    let year = date[0..4].parse::<u16>().map_err(|_| invalid(path, "invalid year"))?;
    let month = date[5..7].parse::<u8>().map_err(|_| invalid(path, "invalid month"))?;
    let day = date[8..10].parse::<u8>().map_err(|_| invalid(path, "invalid day"))?;
    let (clock, fraction) = match time.split_once('.') { Some((c, f)) => (c, Some(f)), None => (time, None) };
    let clock_bytes = clock.as_bytes();
    if clock_bytes.len() != 8
        || clock_bytes[2] != b':'
        || clock_bytes[5] != b':'
        || !clock_bytes.iter().enumerate().all(|(i, b)| matches!(i, 2 | 5) || b.is_ascii_digit())
    {
        return Err(invalid(path, "time must use ASCII HH:MM:SS[.fraction]Z"));
    }
    let hour = clock[0..2].parse::<u8>().map_err(|_| invalid(path, "invalid hour"))?;
    let minute = clock[3..5].parse::<u8>().map_err(|_| invalid(path, "invalid minute"))?;
    let second = clock[6..8].parse::<u8>().map_err(|_| invalid(path, "invalid second"))?;
    let nanosecond = if let Some(f) = fraction {
        if f.is_empty() || f.len() > 9 || !f.bytes().all(|b| b.is_ascii_digit()) {
            return Err(invalid(path, "fraction must contain one to nine digits"));
        }
        format!("{f:0<9}").parse::<u32>().map_err(|_| invalid(path, "invalid fractional second"))?
    } else { 0 };
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => return Err(invalid(path, "month outside 1..=12")),
    };
    if year == 0 || day == 0 || day > days || hour > 23 || minute > 59 || second > 59 {
        return Err(invalid(path, "date or time component outside valid range"));
    }
    Ok(UtcInstant { year, month, day, hour, minute, second, nanosecond })
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn event_invalidates_claim(kind: WaterEventType) -> bool {
    matches!(kind,
        WaterEventType::SourceSwitch | WaterEventType::BlendCreated |
        WaterEventType::TreatmentBypass | WaterEventType::TreatmentFault |
        WaterEventType::Maintenance | WaterEventType::MediaReplacement |
        WaterEventType::StorageStagnation | WaterEventType::SamplePointChange |
        WaterEventType::MonitoringOverdue | WaterEventType::NonCompliantResult |
        WaterEventType::CorrectiveActionStarted | WaterEventType::CorrectiveActionCompleted |
        WaterEventType::RiskProfileChanged | WaterEventType::Other)
}

fn requires_drinking_quality(use_class: WaterUseClass) -> bool {
    matches!(use_class, WaterUseClass::Drinking | WaterUseClass::FoodPreparation)
}

fn validate_parameter_result(result: &WaterParameterResult) -> Result<(), EvaluationError> {
    nonempty("sample_observations.parameter_results.parameter_id", &result.parameter_id)?;
    nonempty("sample_observations.parameter_results.unit", &result.unit)?;
    if result.result_state == ParameterResultState::Numeric {
        let value = result.value.ok_or_else(|| invalid("sample_observations.parameter_results.value", "numeric result requires a value"))?;
        if !value.is_finite() { return Err(invalid("sample_observations.parameter_results.value", "must be finite")); }
    }
    for (path, value) in [
        ("sample_observations.parameter_results.detection_or_reporting_limit", result.detection_or_reporting_limit),
        ("sample_observations.parameter_results.uncertainty", result.uncertainty),
    ] {
        if let Some(value) = value {
            if !value.is_finite() || value < 0.0 { return Err(invalid(path, "must be finite and non-negative")); }
        }
    }
    Ok(())
}

/// Check internal consistency of a WaterUseQualificationV1 record.
///
/// No SANS numerical comparison occurs here. Consumers still need to verify
/// the source records, laboratory authority, methods, current standard and
/// monitoring programme independently.
pub fn validate_water_use_qualification(
    input: &WaterUseQualificationV1,
) -> Result<WaterUseQualificationValidation, EvaluationError> {
    if input.schema_version != "1.0.0" { return Err(invalid("schema_version", "unsupported schema version")); }
    for (path, value) in [
        ("qualification_id", input.qualification_id.as_str()),
        ("source.source_id", input.source.source_id.as_str()),
        ("source.system_id", input.source.system_id.as_str()),
        ("source.system_revision", input.source.system_revision.as_str()),
        ("stream.stream_id", input.stream.stream_id.as_str()),
        ("stream.storage_state", input.stream.storage_state.as_str()),
        ("treatment.train_id", input.treatment.train_id.as_str()),
        ("treatment.configuration_revision", input.treatment.configuration_revision.as_str()),
        ("intended_use.use_profile_id", input.intended_use.use_profile_id.as_str()),
        ("intended_use.use_profile_revision", input.intended_use.use_profile_revision.as_str()),
        ("delivery_boundary.boundary_id", input.delivery_boundary.boundary_id.as_str()),
        ("delivery_boundary.boundary_kind", input.delivery_boundary.boundary_kind.as_str()),
        ("delivery_boundary.sample_point_id", input.delivery_boundary.sample_point_id.as_str()),
        ("quality_profile.standard_id", input.quality_profile.standard_id.as_str()),
        ("quality_profile.standard_edition", input.quality_profile.standard_edition.as_str()),
        ("quality_profile.risk_assessment_id", input.quality_profile.risk_assessment_id.as_str()),
        ("quality_profile.risk_assessment_revision", input.quality_profile.risk_assessment_revision.as_str()),
        ("quality_profile.monitoring_programme_id", input.quality_profile.monitoring_programme_id.as_str()),
        ("quality_profile.monitoring_programme_revision", input.quality_profile.monitoring_programme_revision.as_str()),
        ("evidence_lineage.evaluator_id", input.evidence_lineage.evaluator_id.as_str()),
        ("evidence_lineage.evaluator_revision", input.evidence_lineage.evaluator_revision.as_str()),
    ] { nonempty(path, value)?; }

    let stream_start = parse_utc_instant(&input.stream.observed_from, "stream.observed_from")?;
    if let Some(until) = &input.stream.observed_until {
        if parse_utc_instant(until, "stream.observed_until")? <= stream_start {
            return Err(invalid("stream.observed_until", "must be later than observed_from"));
        }
    }
    let unique_parents: BTreeSet<_> = input.stream.blend_parent_ids.iter().collect();
    if unique_parents.len() != input.stream.blend_parent_ids.len() {
        return Err(invalid("stream.blend_parent_ids", "contains duplicate parent IDs"));
    }
    if input.stream.blend_parent_ids.iter().any(|id| id.trim().is_empty()) {
        return Err(invalid("stream.blend_parent_ids", "parent IDs must not be empty"));
    }
    if input.stream.blend_parent_ids.iter().any(|id| id == &input.stream.stream_id) {
        return Err(invalid("stream.blend_parent_ids", "derived stream cannot be its own parent"));
    }
    let upstream_ids: BTreeSet<_> = input.source.upstream_source_ids.iter().collect();
    if upstream_ids.len() != input.source.upstream_source_ids.len()
        || input.source.upstream_source_ids.iter().any(|id| id.trim().is_empty() || id == &input.source.source_id)
    {
        return Err(invalid("source.upstream_source_ids", "upstream source IDs must be non-empty, unique and not self-referential"));
    }
    match input.stream.stream_kind {
        WaterStreamKind::DerivedBlend if input.stream.blend_parent_ids.len() < 2 =>
            return Err(invalid("stream.blend_parent_ids", "derived blend requires at least two distinct parents")),
        WaterStreamKind::DerivedBlend => {},
        _ if !input.stream.blend_parent_ids.is_empty() =>
            return Err(invalid("stream.blend_parent_ids", "parents are only valid for a derived blend")),
        _ => {},
    }

    let mut sample_by_id: BTreeMap<&str, &WaterSampleObservation> = BTreeMap::new();
    let mut sample_times: BTreeMap<&str, UtcInstant> = BTreeMap::new();
    for sample in &input.sample_observations {
        nonempty("sample_observations.sample_id", &sample.sample_id)?;
        nonempty("sample_observations.sample_point_id", &sample.sample_point_id)?;
        nonempty("sample_observations.method_id", &sample.method_id)?;
        let time = parse_utc_instant(&sample.collected_at, "sample_observations.collected_at")?;
        if let Some(digest) = &sample.chain_of_custody_digest_sha256 {
            if !valid_sha256(digest) { return Err(invalid("sample_observations.chain_of_custody_digest_sha256", "must be 64 hexadecimal characters")); }
        }
        if sample_by_id.insert(sample.sample_id.as_str(), sample).is_some() {
            return Err(invalid("sample_observations.sample_id", "duplicate sample ID"));
        }
        sample_times.insert(sample.sample_id.as_str(), time);
        let mut parameters = BTreeSet::new();
        for result in &sample.parameter_results {
            validate_parameter_result(result)?;
            if !parameters.insert(result.parameter_id.as_str()) {
                return Err(invalid("sample_observations.parameter_results", "duplicate parameter ID in one sample"));
            }
        }
    }
    if input.quality_profile.required_parameter_ids.iter().collect::<BTreeSet<_>>().len()
        != input.quality_profile.required_parameter_ids.len() {
        return Err(invalid("quality_profile.required_parameter_ids", "contains duplicate parameter IDs"));
    }

    let decision_time = parse_utc_instant(&input.decision.decided_at, "decision.decided_at")?;
    if input.evidence_lineage.source_record_ids.is_empty()
        || input.evidence_lineage.source_record_ids.iter().any(|id| id.trim().is_empty())
        || input.evidence_lineage.source_record_ids.iter().collect::<BTreeSet<_>>().len()
            != input.evidence_lineage.source_record_ids.len()
    {
        return Err(invalid("evidence_lineage.source_record_ids", "requires non-empty, unique source record IDs"));
    }
    let mut event_ids = BTreeSet::new();
    let mut parsed_events = Vec::with_capacity(input.event_history.len());
    for event in &input.event_history {
        nonempty("event_history.event_id", &event.event_id)?;
        if !event_ids.insert(&event.event_id) { return Err(invalid("event_history.event_id", "duplicate event ID")); }
        let time = parse_utc_instant(&event.occurred_at, "event_history.occurred_at")?;
        if let Some(digest) = &event.details_digest_sha256 {
            if !valid_sha256(digest) { return Err(invalid("event_history.details_digest_sha256", "must be 64 hexadecimal characters")); }
        }
        let affects_stream = event.affected_stream_ids.is_empty()
            || event.affected_stream_ids.iter().any(|id| id == &input.stream.stream_id);
        parsed_events.push((event.event_type, time, affects_stream));
    }

    let mut support_ids = BTreeSet::new();
    let mut latest_supporting_sample: Option<UtcInstant> = None;
    let mut supported_parameters = BTreeSet::new();
    for sample_id in &input.decision.supporting_sample_ids {
        if !support_ids.insert(sample_id.as_str()) {
            return Err(invalid("decision.supporting_sample_ids", "contains duplicate sample IDs"));
        }
        let sample = sample_by_id.get(sample_id.as_str()).ok_or_else(||
            invalid("decision.supporting_sample_ids", "references a missing sample observation"))?;
        let sample_time = *sample_times.get(sample_id.as_str()).expect("sample time inserted with sample");
        if sample_time > decision_time {
            return Err(invalid("decision.supporting_sample_ids", "supporting sample occurs after decision time"));
        }
        latest_supporting_sample = Some(latest_supporting_sample.map_or(sample_time, |latest| latest.max(sample_time)));
        for result in &sample.parameter_results {
            if !matches!(result.result_state, ParameterResultState::NotTested | ParameterResultState::Invalid) {
                supported_parameters.insert(result.parameter_id.as_str());
            }
        }
    }

    if let Some(valid_from) = &input.decision.valid_from {
        let valid_from = parse_utc_instant(valid_from, "decision.valid_from")?;
        if valid_from > decision_time {
            return Err(invalid("decision.valid_from", "must not occur after decision time"));
        }
        if let Some(latest_sample) = latest_supporting_sample {
            if valid_from < latest_sample {
                return Err(invalid("decision.valid_from", "qualification validity cannot begin before its latest supporting sample"));
            }
        }
    }
    if let Some(valid_until) = &input.decision.valid_until {
        let expiry = parse_utc_instant(valid_until, "decision.valid_until")?;
        if let Some(valid_from) = &input.decision.valid_from {
            if expiry <= parse_utc_instant(valid_from, "decision.valid_from")? {
                return Err(invalid("decision.valid_until", "must be later than valid_from"));
            }
        }
        if decision_time >= expiry { return Err(invalid("decision.valid_until", "decision time is at or after declared expiry")); }
    }
    if let Some(next_review) = &input.quality_profile.next_required_review_at {
        if parse_utc_instant(next_review, "quality_profile.next_required_review_at")? <= decision_time
            && input.quality_profile.monitoring_status == MonitoringStatus::Current {
            return Err(invalid("quality_profile.monitoring_status", "cannot be current at or after next required review time"));
        }
    }

    if input.decision.status != WaterQualificationStatus::QualifiedForDeclaredUse
        && input.decision.reason_codes.is_empty() {
        return Err(invalid("decision.reason_codes", "non-qualified states require at least one reason code"));
    }
    if input.decision.status == WaterQualificationStatus::KnownNonCompliance
        && input.decision.supporting_sample_ids.is_empty() {
        return Err(invalid("decision.supporting_sample_ids", "known non-compliance requires supporting evidence"));
    }
    if input.decision.status == WaterQualificationStatus::CorrectiveActionPending
        && input.decision.incident_id.as_deref().is_none_or(str::is_empty) {
        return Err(invalid("decision.incident_id", "corrective-action-pending state requires an incident ID"));
    }
    if input.decision.status == WaterQualificationStatus::TreatmentStateUnqualified
        && matches!(input.treatment.operating_state, TreatmentOperatingState::QualifiedState | TreatmentOperatingState::NotApplicable) {
        return Err(invalid("treatment.operating_state", "unqualified decision conflicts with qualified/not-applicable treatment state"));
    }

    if input.decision.status == WaterQualificationStatus::QualifiedForDeclaredUse {
        if input.source.source_class == WaterSourceClass::Unknown {
            return Err(invalid("source.source_class", "unknown source cannot support qualified-use claim"));
        }
        if input.stream.stream_kind == WaterStreamKind::Unknown {
            return Err(invalid("stream.stream_kind", "unknown stream kind cannot support qualified-use claim"));
        }
        if !input.treatment.active_alarm_ids.is_empty()
            || matches!(input.treatment.operating_state,
                TreatmentOperatingState::UnqualifiedState | TreatmentOperatingState::Bypassed |
                TreatmentOperatingState::Faulted | TreatmentOperatingState::Maintenance |
                TreatmentOperatingState::Unknown) {
            return Err(invalid("treatment.operating_state", "active alarms or unqualified treatment state block qualified-use claim"));
        }
        if input.quality_profile.monitoring_status != MonitoringStatus::Current {
            return Err(invalid("quality_profile.monitoring_status", "qualified-use claim requires current monitoring programme"));
        }
        if input.quality_profile.required_parameter_ids.is_empty() {
            return Err(invalid("quality_profile.required_parameter_ids", "qualified-use claim requires a declared parameter set"));
        }
        if input.decision.supporting_sample_ids.is_empty() {
            return Err(invalid("decision.supporting_sample_ids", "qualified-use claim requires supporting sample evidence"));
        }
        if input.decision.valid_from.is_none() {
            return Err(invalid("decision.valid_from", "qualified-use claim requires a declared validity start"));
        }
        match input.delivery_boundary.sample_point_relationship {
            SamplePointRelationship::SamePhysicalPoint => {
                for id in &input.decision.supporting_sample_ids {
                    let sample = sample_by_id.get(id.as_str()).expect("support sample checked above");
                    if sample.sample_point_id != input.delivery_boundary.sample_point_id {
                        return Err(invalid("delivery_boundary.sample_point_relationship", "supporting sample does not match declared physical sample point"));
                    }
                }
            }
            SamplePointRelationship::ValidatedTransfer => {
                if input.delivery_boundary.transfer_validation_id.as_deref().is_none_or(str::is_empty) {
                    return Err(invalid("delivery_boundary.transfer_validation_id", "validated transfer requires a validation reference"));
                }
            }
            SamplePointRelationship::DifferentPointNoTransfer | SamplePointRelationship::Unknown =>
                return Err(invalid("delivery_boundary.sample_point_relationship", "sample-point relationship cannot support qualified-use claim")),
        }
        for required in &input.quality_profile.required_parameter_ids {
            if !supported_parameters.contains(required.as_str()) {
                return Err(invalid("quality_profile.required_parameter_ids", format!("supporting samples do not cover required parameter {required}")));
            }
        }
        if requires_drinking_quality(input.intended_use.use_class) {
            if input.source.source_class == WaterSourceClass::Greywater {
                return Err(invalid("source.source_class", "greywater cannot be qualified for drinking or food preparation"));
            }
            for id in &input.decision.supporting_sample_ids {
                let sample = sample_by_id.get(id.as_str()).expect("support sample checked above");
                if sample.laboratory_or_provider_id.as_deref().is_none_or(str::is_empty) {
                    return Err(invalid("sample_observations.laboratory_or_provider_id", "drinking/food-preparation use requires laboratory or qualified provider identity"));
                }
                if sample.chain_of_custody_digest_sha256.as_deref().is_none_or(|digest| !valid_sha256(digest)) {
                    return Err(invalid("sample_observations.chain_of_custody_digest_sha256", "drinking/food-preparation use requires a valid chain-of-custody digest"));
                }
            }
        }
        if let Some(latest_sample) = latest_supporting_sample {
            if parsed_events.iter().any(|(kind, when, affects)| *affects && event_invalidates_claim(*kind) && *when > latest_sample) {
                return Err(invalid("event_history", "an invalidating event occurred after the latest supporting sample"));
            }
        }
    }

    if let Some(commit) = &input.evidence_lineage.source_repository_commit {
        if commit.len() != 40 || !commit.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(invalid("evidence_lineage.source_repository_commit", "must be a full 40-character Git object ID"));
        }
    }

    Ok(WaterUseQualificationValidation {
        qualification_id: input.qualification_id.clone(),
        declared_status: input.decision.status,
        intended_use: input.intended_use.use_class,
        supporting_sample_count: input.decision.supporting_sample_ids.len(),
        synthetic_only: input.evidence_lineage.synthetic_only,
        qualification_record_consistent: true,
        physical_water_safety_established: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_record() -> WaterUseQualificationV1 {
        WaterUseQualificationV1 {
            schema_version: "1.0.0".into(),
            qualification_id: "qual-001".into(),
            source: WaterSource {
                source_id: "municipal-main".into(), source_class: WaterSourceClass::Municipal,
                system_id: "supply-zone-01".into(), system_revision: "rev-3".into(),
                upstream_source_ids: vec![], provider_id: Some("water-provider-01".into()),
            },
            stream: WaterStream {
                stream_id: "stream-01".into(), stream_kind: WaterStreamKind::ContinuousStream,
                observed_from: "2026-11-04T08:00:00Z".into(), observed_until: None,
                storage_vessel_id: None, residence_seconds: None, storage_state: "not_applicable".into(),
                blend_parent_ids: vec![],
            },
            treatment: WaterTreatment {
                train_id: "municipal-treatment".into(), configuration_revision: "treatment-rev-1".into(),
                operating_state: TreatmentOperatingState::NotApplicable, media_or_component_revision: None,
                last_maintenance_at: None, active_alarm_ids: vec![],
            },
            intended_use: IntendedWaterUse {
                use_class: WaterUseClass::Drinking, use_profile_id: "sa-drinking-water".into(),
                use_profile_revision: "1".into(), jurisdiction: Some("South Africa".into()),
            },
            delivery_boundary: WaterDeliveryBoundary {
                boundary_id: "kitchen-tap".into(), boundary_kind: "household_fixture".into(),
                sample_point_id: "kitchen-tap".into(), sample_point_relationship: SamplePointRelationship::SamePhysicalPoint,
                transfer_validation_id: None,
            },
            quality_profile: WaterQualityProfile {
                standard_id: "SANS 241".into(), standard_edition: "2025".into(),
                risk_assessment_id: "risk-01".into(), risk_assessment_revision: "r1".into(),
                monitoring_programme_id: "monitor-01".into(), monitoring_programme_revision: "m1".into(),
                monitoring_status: MonitoringStatus::Current, required_parameter_ids: vec!["ecoli".into()],
                next_required_review_at: Some("2026-11-05T08:00:00Z".into()),
            },
            sample_observations: vec![WaterSampleObservation {
                sample_id: "sample-001".into(), sample_point_id: "kitchen-tap".into(),
                collected_at: "2026-11-04T09:00:00Z".into(), collector_id: Some("collector-01".into()),
                laboratory_or_provider_id: Some("lab-01".into()), method_id: "method-ecoli-01".into(),
                method_revision: Some("2".into()), chain_of_custody_digest_sha256: Some("a".repeat(64)),
                parameter_results: vec![WaterParameterResult {
                    parameter_id: "ecoli".into(), result_state: ParameterResultState::Numeric,
                    value: Some(0.0), unit: "CFU/100mL".into(), detection_or_reporting_limit: Some(1.0),
                    uncertainty: None, qualifier: None, quality_control_flags: vec![],
                }],
            }],
            decision: WaterQualificationDecision {
                status: WaterQualificationStatus::QualifiedForDeclaredUse, reason_codes: vec![],
                decided_at: "2026-11-04T10:00:00Z".into(), valid_from: Some("2026-11-04T09:30:00Z".into()),
                valid_until: None, supporting_sample_ids: vec!["sample-001".into()],
                incident_id: None, corrective_action_id: None,
            },
            event_history: vec![],
            evidence_lineage: WaterEvidenceLineage {
                source_record_ids: vec!["source-record-01".into()], evaluator_id: "water-qualifier".into(),
                evaluator_revision: "0.1.0".into(), source_repository_commit: Some("b".repeat(40)), synthetic_only: false,
            },
        }
    }

    #[test]
    fn accepts_consistent_declared_candidate_without_claiming_safety() {
        let result = validate_water_use_qualification(&valid_record()).unwrap();
        assert!(result.qualification_record_consistent);
        assert!(!result.physical_water_safety_established);
        assert_eq!(result.supporting_sample_count, 1);
    }

    #[test]
    fn synthetic_record_never_establishes_physical_safety() {
        let mut record = valid_record();
        record.evidence_lineage.synthetic_only = true;
        let result = validate_water_use_qualification(&record).unwrap();
        assert!(result.synthetic_only);
        assert!(!result.physical_water_safety_established);
    }

    #[test]
    fn rejects_non_current_monitoring() {
        let mut record = valid_record();
        record.quality_profile.monitoring_status = MonitoringStatus::Overdue;
        assert!(validate_water_use_qualification(&record).is_err());
    }

    #[test]
    fn rejects_source_switch_after_supporting_sample() {
        let mut record = valid_record();
        record.event_history.push(WaterQualificationEvent {
            event_id: "evt-01".into(), event_type: WaterEventType::SourceSwitch,
            occurred_at: "2026-11-04T09:30:00Z".into(), affected_stream_ids: vec!["stream-01".into()],
            details_digest_sha256: None,
        });
        assert!(validate_water_use_qualification(&record).is_err());
    }

    #[test]
    fn rejects_mismatched_physical_sample_point() {
        let mut record = valid_record();
        record.sample_observations[0].sample_point_id = "tank-outlet".into();
        assert!(validate_water_use_qualification(&record).is_err());
    }

    #[test]
    fn rejects_unvalidated_different_sample_point() {
        let mut record = valid_record();
        record.delivery_boundary.sample_point_relationship = SamplePointRelationship::DifferentPointNoTransfer;
        assert!(validate_water_use_qualification(&record).is_err());
    }

    #[test]
    fn rejects_greywater_as_drinking_water() {
        let mut record = valid_record();
        record.source.source_class = WaterSourceClass::Greywater;
        assert!(validate_water_use_qualification(&record).is_err());
    }

    #[test]
    fn rejects_treatment_bypass_after_sample() {
        let mut record = valid_record();
        record.event_history.push(WaterQualificationEvent {
            event_id: "evt-bypass".into(), event_type: WaterEventType::TreatmentBypass,
            occurred_at: "2026-11-04T09:45:00Z".into(), affected_stream_ids: vec!["stream-01".into()],
            details_digest_sha256: None,
        });
        assert!(validate_water_use_qualification(&record).is_err());
    }

    #[test]
    fn rejects_required_parameter_not_covered_by_supporting_samples() {
        let mut record = valid_record();
        record.quality_profile.required_parameter_ids.push("turbidity".into());
        assert!(validate_water_use_qualification(&record).is_err());
    }

    #[test]
    fn rejects_derived_blend_without_two_parent_streams() {
        let mut record = valid_record();
        record.stream.stream_kind = WaterStreamKind::DerivedBlend;
        assert!(validate_water_use_qualification(&record).is_err());
    }

    #[test]
    fn unknown_state_is_valid_record_but_never_a_qualified_claim() {
        let mut record = valid_record();
        record.decision.status = WaterQualificationStatus::UnknownEvidenceInsufficient;
        record.decision.reason_codes = vec!["sample_missing".into()];
        record.decision.supporting_sample_ids.clear();
        record.sample_observations.clear();
        let result = validate_water_use_qualification(&record).unwrap();
        assert_eq!(result.declared_status, WaterQualificationStatus::UnknownEvidenceInsufficient);
        assert!(!result.physical_water_safety_established);
    }

    #[test]
    fn rejects_non_utc_timestamp_instead_of_ambiguous_ordering() {
        let mut record = valid_record();
        record.decision.decided_at = "2026-11-04T12:00:00+02:00".into();
        assert!(validate_water_use_qualification(&record).is_err());
    }

    #[test]
    fn rejects_duplicate_sample_ids() {
        let mut record = valid_record();
        record.sample_observations.push(record.sample_observations[0].clone());
        assert!(validate_water_use_qualification(&record).is_err());
    }

    #[test]
    fn rejects_duplicate_event_ids() {
        let mut record = valid_record();
        let event = WaterQualificationEvent {
            event_id: "evt-01".into(), event_type: WaterEventType::Maintenance,
            occurred_at: "2026-11-04T09:30:00Z".into(), affected_stream_ids: vec!["stream-01".into()],
            details_digest_sha256: None,
        };
        record.event_history = vec![event.clone(), event];
        assert!(validate_water_use_qualification(&record).is_err());
    }
}
