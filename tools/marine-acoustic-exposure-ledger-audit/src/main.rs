use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::{env, fs, process};

const SCHEMA_VERSION: &str = "0.1.0";
const EARTH_RADIUS_M: f64 = 6_371_000.0;
const TIME_TOLERANCE_S: f64 = 1.0e-6;

#[derive(Debug, Deserialize, PartialEq, Eq, Copy, Clone)]
#[serde(rename_all = "snake_case")]
enum EvidenceKind { Synthetic, Measured }

#[derive(Debug, Deserialize, PartialEq, Eq, Copy, Clone)]
#[serde(rename_all = "snake_case")]
enum RightsStatus { PublicOrPermitted, Unknown, Restricted }

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    schema_version: String,
    case_id: String,
    period: ExposurePeriod,
    source_events: Vec<SourceEvent>,
    receivers: Vec<Receiver>,
    contributions: Vec<AcousticContribution>,
    ambient_observations: Vec<AmbientObservation>,
    exposure_limits: Vec<ExposureLimit>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExposurePeriod {
    period_id: String,
    start_time_s: f64,
    end_time_s: f64,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceEvent {
    event_id: String,
    evidence_kind: EvidenceKind,
    dataset_id: String,
    citation: String,
    rights_status: RightsStatus,
    independently_reviewed: bool,
    vessel_id: String,
    vessel_revision: String,
    propeller_revision: String,
    source_state_id: String,
    mission_id: String,
    route_id: String,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    source_track_sha256: String,
    source_level_method_id: String,
    source_level_calibration_sha256: String,
    source_level_raw_trace_sha256: String,
    source_level_timebase_receipt_sha256: String,
    operating_point_receipt_sha256: String,
    source_levels: Vec<SourceLevelBand>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceLevelBand {
    band_id: String,
    quantity_id: String,
    center_frequency_hz: f64,
    source_level_db: f64,
    expanded_uncertainty_db: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Receiver {
    receiver_id: String,
    latitude_deg: f64,
    longitude_deg: f64,
    depth_m: f64,
    location_evidence_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AcousticContribution {
    contribution_id: String,
    event_id: String,
    receiver_id: String,
    band_id: String,
    start_time_s: f64,
    end_time_s: f64,
    duration_s: f64,
    source_latitude_deg: f64,
    source_longitude_deg: f64,
    source_depth_m: f64,
    range_m: f64,
    range_tolerance_m: f64,
    transmission_loss_db: f64,
    transmission_loss_expanded_uncertainty_db: f64,
    propagation_model_id: String,
    propagation_model_version: String,
    propagation_validation_sha256: String,
    bathymetry_sha256: String,
    sound_speed_profile_sha256: String,
    seabed_model_sha256: String,
    environment_fingerprint_sha256: String,
    timebase_receipt_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AmbientObservation {
    observation_id: String,
    receiver_id: String,
    band_id: String,
    start_time_s: f64,
    end_time_s: f64,
    duration_s: f64,
    ambient_spl_db: f64,
    expanded_uncertainty_db: f64,
    measurement_method_id: String,
    instrument_id: String,
    calibration_certificate_sha256: String,
    raw_trace_sha256: String,
    timebase_receipt_sha256: String,
    ambient_only_basis: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExposureLimit {
    limit_id: String,
    receiver_id: String,
    band_id: String,
    upper_sel_threshold_db: f64,
    requirement_reference: String,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    schema_version: &'static str,
    case_id: String,
    period_id: String,
    disposition: &'static str,
    partitions: Vec<ExposurePartitionReport>,
    limit_results: Vec<LimitResult>,
    claim_ceiling: &'static str,
    reasons: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ExposurePartitionReport {
    receiver_id: String,
    band_id: String,
    contribution_count: usize,
    unique_source_event_count: usize,
    unique_temporal_coverage_s: f64,
    summed_source_contribution_duration_s: f64,
    ship_only_cumulative_sel_db: f64,
    ship_only_cumulative_sel_lower_db: f64,
    ship_only_cumulative_sel_upper_db: f64,
    maximum_received_level_db: f64,
    maximum_received_level_lower_db: f64,
    maximum_received_level_upper_db: f64,
    contribution_reports: Vec<ContributionReport>,
    ambient_observation_count: usize,
    ambient_reference_sel_db: Option<f64>,
    ambient_reference_sel_lower_db: Option<f64>,
    ambient_reference_sel_upper_db: Option<f64>,
    ambient_only_basis: &'static str,
}

#[derive(Debug, Serialize)]
struct ContributionReport {
    contribution_id: String,
    event_id: String,
    receiver_id: String,
    band_id: String,
    source_level_db: f64,
    transmission_loss_db: f64,
    received_level_db: f64,
    received_level_lower_db: f64,
    received_level_upper_db: f64,
    duration_s: f64,
    sel_db: f64,
    sel_lower_db: f64,
    sel_upper_db: f64,
    propagation_model_id: String,
    propagation_model_version: String,
    propagation_validation_sha256: String,
    bathymetry_sha256: String,
    sound_speed_profile_sha256: String,
    seabed_model_sha256: String,
    environment_fingerprint_sha256: String,
    timebase_receipt_sha256: String,
}

#[derive(Debug, Serialize)]
struct LimitResult {
    limit_id: String,
    receiver_id: String,
    band_id: String,
    threshold_db: f64,
    observed_lower_db: Option<f64>,
    observed_upper_db: Option<f64>,
    status: &'static str,
    requirement_reference: String,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn valid_lat_lon(latitude: f64, longitude: f64) -> bool {
    latitude.is_finite() && (-90.0..=90.0).contains(&latitude)
        && longitude.is_finite() && (-180.0..=180.0).contains(&longitude)
}

fn haversine_distance_m(lat_a: f64, lon_a: f64, lat_b: f64, lon_b: f64) -> f64 {
    let to_radians = std::f64::consts::PI / 180.0;
    let lat_a = lat_a * to_radians;
    let lat_b = lat_b * to_radians;
    let delta_lat = (lat_b - lat_a) / 2.0;
    let delta_lon = (lon_b - lon_a) * to_radians / 2.0;
    let a = delta_lat.sin().powi(2)
        + lat_a.cos() * lat_b.cos() * delta_lon.sin().powi(2);
    2.0 * EARTH_RADIUS_M * a.sqrt().asin()
}

fn log_energy_sum(levels_db: &[f64]) -> Option<f64> {
    if levels_db.is_empty() || levels_db.iter().any(|level| !level.is_finite()) {
        return None;
    }
    let max_level = levels_db.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum_relative_energy = levels_db.iter()
        .map(|level| 10.0_f64.powf((level - max_level) / 10.0))
        .sum::<f64>();
    let result = max_level + 10.0 * sum_relative_energy.log10();
    result.is_finite().then_some(result)
}

fn union_duration(intervals: &[(f64, f64)]) -> f64 {
    if intervals.is_empty() { return 0.0; }
    let mut sorted = intervals.to_vec();
    sorted.sort_by(|a,b| a.0.total_cmp(&b.0));
    let mut total = 0.0;
    let (mut start, mut end) = sorted[0];
    for (next_start, next_end) in sorted.into_iter().skip(1) {
        if next_start <= end {
            end = end.max(next_end);
        } else {
            total += end - start;
            start = next_start;
            end = next_end;
        }
    }
    total + end - start
}

fn validate_case(case: &Case) -> Vec<String> {
    let mut errors = Vec::new();
    if case.schema_version != SCHEMA_VERSION {
        errors.push(format!("unsupported schema_version {}; expected {SCHEMA_VERSION}", case.schema_version));
    }
    if case.case_id.trim().is_empty() || case.period.period_id.trim().is_empty()
        || !case.period.start_time_s.is_finite()
        || !case.period.end_time_s.is_finite()
        || case.period.end_time_s <= case.period.start_time_s
    {
        errors.push("case/period identity or interval is invalid".to_string());
    }
    for (label,digest) in [
        ("period service fingerprint",&case.period.service_fingerprint_sha256),
        ("period environment fingerprint",&case.period.environment_fingerprint_sha256),
    ] {
        if !is_sha256(digest) { errors.push(format!("{label} must be a SHA-256 digest")); }
    }

    let mut event_ids = BTreeSet::new();
    for event in &case.source_events {
        if event.event_id.trim().is_empty() || !event_ids.insert(event.event_id.clone()) {
            errors.push(format!("empty or duplicate source event {}", event.event_id));
        }
        if event.dataset_id.trim().is_empty() || event.citation.trim().is_empty()
            || event.vessel_id.trim().is_empty() || event.vessel_revision.trim().is_empty()
            || event.propeller_revision.trim().is_empty() || event.source_state_id.trim().is_empty()
            || event.mission_id.trim().is_empty() || event.route_id.trim().is_empty()
            || event.source_level_method_id.trim().is_empty()
        {
            errors.push(format!("source event {} lacks required identity/method metadata", event.event_id));
        }
        if event.service_fingerprint_sha256 != case.period.service_fingerprint_sha256
            || event.environment_fingerprint_sha256 != case.period.environment_fingerprint_sha256
        {
            errors.push(format!("source event {} does not match the frozen service/environment fingerprint", event.event_id));
        }
        for (label,digest) in [
            ("source track",&event.source_track_sha256),
            ("source-level calibration",&event.source_level_calibration_sha256),
            ("source-level raw trace",&event.source_level_raw_trace_sha256),
            ("source-level timebase",&event.source_level_timebase_receipt_sha256),
            ("operating-point receipt",&event.operating_point_receipt_sha256),
        ] {
            if !is_sha256(digest) { errors.push(format!("source event {} has invalid {label} digest", event.event_id)); }
        }
        let mut bands = BTreeSet::new();
        for band in &event.source_levels {
            if band.band_id.trim().is_empty() || !bands.insert(band.band_id.clone())
                || band.quantity_id != "source_level_db_re_1upa_at_1m"
                || !finite_positive(band.center_frequency_hz)
                || !band.source_level_db.is_finite()
                || !finite_positive(band.expanded_uncertainty_db)
                || !(band.source_level_db - band.expanded_uncertainty_db).is_finite()
                || !(band.source_level_db + band.expanded_uncertainty_db).is_finite()
            {
                errors.push(format!("source event {} has invalid/duplicate source-level band", event.event_id));
            }
        }
        if event.source_levels.is_empty() {
            errors.push(format!("source event {} has no source-level spectrum", event.event_id));
        }
    }

    let mut receivers = BTreeSet::new();
    for receiver in &case.receivers {
        if receiver.receiver_id.trim().is_empty() || !receivers.insert(receiver.receiver_id.clone())
            || !valid_lat_lon(receiver.latitude_deg,receiver.longitude_deg)
            || !finite_positive(receiver.depth_m)
            || !is_sha256(&receiver.location_evidence_sha256)
        {
            errors.push(format!("receiver {} has invalid/duplicate identity, coordinates, depth or location evidence", receiver.receiver_id));
        }
    }

    let mut contribution_ids = BTreeSet::new();
    for contribution in &case.contributions {
        if contribution.contribution_id.trim().is_empty()
            || !contribution_ids.insert(contribution.contribution_id.clone())
        {
            errors.push(format!("empty or duplicate acoustic contribution {}", contribution.contribution_id));
        }
        let event = case.source_events.iter().find(|event| event.event_id == contribution.event_id);
        let receiver = case.receivers.iter().find(|receiver| receiver.receiver_id == contribution.receiver_id);
        if event.is_none() || receiver.is_none() {
            errors.push(format!("contribution {} references an unknown source event or receiver", contribution.contribution_id));
            continue;
        }
        let event = event.unwrap();
        let receiver = receiver.unwrap();
        if !event.source_levels.iter().any(|band| band.band_id == contribution.band_id) {
            errors.push(format!("contribution {} has no matching source-level band", contribution.contribution_id));
        }
        if contribution.environment_fingerprint_sha256 != case.period.environment_fingerprint_sha256 {
            errors.push(format!("contribution {} environment fingerprint differs from the frozen period", contribution.contribution_id));
        }
        if !contribution.start_time_s.is_finite() || !contribution.end_time_s.is_finite()
            || !finite_positive(contribution.duration_s)
            || contribution.end_time_s <= contribution.start_time_s
            || ((contribution.end_time_s-contribution.start_time_s)-contribution.duration_s).abs() > TIME_TOLERANCE_S
            || contribution.start_time_s < case.period.start_time_s
            || contribution.end_time_s > case.period.end_time_s
        {
            errors.push(format!("contribution {} has invalid/out-of-period time interval or duration", contribution.contribution_id));
        }
        if !valid_lat_lon(contribution.source_latitude_deg,contribution.source_longitude_deg)
            || !finite_positive(contribution.source_depth_m)
            || !finite_positive(contribution.range_m)
            || !finite_positive(contribution.range_tolerance_m)
        {
            errors.push(format!("contribution {} has invalid source geometry", contribution.contribution_id));
        } else {
            let calculated = haversine_distance_m(
                contribution.source_latitude_deg,contribution.source_longitude_deg,
                receiver.latitude_deg,receiver.longitude_deg
            );
            if (calculated-contribution.range_m).abs() > contribution.range_tolerance_m {
                errors.push(format!("contribution {} declared source/receiver range differs from coordinates", contribution.contribution_id));
            }
        }
        if !contribution.transmission_loss_db.is_finite()
            || !finite_positive(contribution.transmission_loss_expanded_uncertainty_db)
            || contribution.propagation_model_id.trim().is_empty()
            || contribution.propagation_model_version.trim().is_empty()
        {
            errors.push(format!("contribution {} has invalid transmission loss or propagation model identity", contribution.contribution_id));
        }
        for (label,digest) in [
            ("propagation validation",&contribution.propagation_validation_sha256),
            ("bathymetry",&contribution.bathymetry_sha256),
            ("sound-speed profile",&contribution.sound_speed_profile_sha256),
            ("seabed model",&contribution.seabed_model_sha256),
            ("timebase",&contribution.timebase_receipt_sha256),
        ] {
            if !is_sha256(digest) { errors.push(format!("contribution {} has invalid {label} digest", contribution.contribution_id)); }
        }
        let source_band = event.source_levels.iter().find(|band| band.band_id == contribution.band_id);
        if let Some(band) = source_band {
            let combined_u = band.expanded_uncertainty_db + contribution.transmission_loss_expanded_uncertainty_db;
            let received = band.source_level_db - contribution.transmission_loss_db;
            if !combined_u.is_finite() || !(received-combined_u).is_finite() || !(received+combined_u).is_finite() {
                errors.push(format!("contribution {} received-level interval overflows", contribution.contribution_id));
            }
        }
    }

    let mut ambient_ids = BTreeSet::new();
    for observation in &case.ambient_observations {
        if observation.observation_id.trim().is_empty() || !ambient_ids.insert(observation.observation_id.clone())
            || !receivers.iter().any(|receiver| receiver == &observation.receiver_id)
            || observation.band_id.trim().is_empty()
            || !observation.ambient_only_basis.eq("independently_measured_ambient_reference")
            || !observation.measurement_method_id.trim().is_empty()
            || !observation.instrument_id.trim().is_empty()
            || !observation.start_time_s.is_finite()
            || !observation.end_time_s.is_finite()
            || observation.end_time_s <= observation.start_time_s
            || !finite_positive(observation.duration_s)
            || ((observation.end_time_s-observation.start_time_s)-observation.duration_s).abs() > TIME_TOLERANCE_S
            || observation.start_time_s < case.period.start_time_s
            || observation.end_time_s > case.period.end_time_s
            || !observation.ambient_spl_db.is_finite()
            || !finite_positive(observation.expanded_uncertainty_db)
        {
            errors.push(format!("ambient observation {} has invalid identity, timing, method or level", observation.observation_id));
        }
        for (label,digest) in [
            ("ambient calibration",&observation.calibration_certificate_sha256),
            ("ambient raw trace",&observation.raw_trace_sha256),
            ("ambient timebase",&observation.timebase_receipt_sha256),
        ] {
            if !is_sha256(digest) { errors.push(format!("ambient observation {} has invalid {label} digest", observation.observation_id)); }
        }
    }

    let mut limits = BTreeSet::new();
    for limit in &case.exposure_limits {
        if limit.limit_id.trim().is_empty() || !limits.insert(limit.limit_id.clone())
            || !receivers.contains(&limit.receiver_id)
            || limit.band_id.trim().is_empty()
            || !limit.upper_sel_threshold_db.is_finite()
            || limit.requirement_reference.trim().is_empty()
        {
            errors.push(format!("exposure limit {} is malformed", limit.limit_id));
        }
    }

    let mut grouped: BTreeMap<(String,String,String),Vec<&AcousticContribution>> = BTreeMap::new();
    for contribution in &case.contributions {
        grouped.entry((contribution.event_id.clone(),contribution.receiver_id.clone(),contribution.band_id.clone()))
            .or_default().push(contribution);
    }
    for ((event_id,receiver_id,band_id),mut records) in grouped {
        records.sort_by(|a,b| a.start_time_s.total_cmp(&b.start_time_s));
        for pair in records.windows(2) {
            if pair[1].start_time_s < pair[0].end_time_s - TIME_TOLERANCE_S {
                errors.push(format!("source event {event_id}, receiver {receiver_id}, band {band_id} has overlapping duplicate exposure intervals"));
            }
        }
    }
    errors
}

fn main() {
    eprintln!("Implement audit");
}
