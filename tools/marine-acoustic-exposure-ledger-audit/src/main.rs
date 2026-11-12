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
    uncertainty_confidence_level: f64,
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
    transmission_loss_uncertainty_confidence_level: f64,
    source_to_propagation_sync_receipt_sha256: String,
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
    evidence_kind: EvidenceKind,
    rights_status: RightsStatus,
    independently_reviewed: bool,
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
    maximum_individual_contribution_received_level_db: f64,
    maximum_individual_contribution_received_level_lower_db: f64,
    maximum_individual_contribution_received_level_upper_db: f64,
    maximum_simultaneous_ship_received_level_db: f64,
    maximum_simultaneous_ship_received_level_lower_db: f64,
    maximum_simultaneous_ship_received_level_upper_db: f64,
    contribution_reports: Vec<ContributionReport>,
    ambient_observation_count: usize,
    ambient_reference_sel_db: Option<f64>,
    ambient_reference_sel_lower_db: Option<f64>,
    ambient_reference_sel_upper_db: Option<f64>,
    ambient_observation_refs: Vec<AmbientObservationRef>,
    ambient_only_basis: &'static str,
}

#[derive(Debug, Serialize)]
struct AmbientObservationRef {
    observation_id: String,
    raw_trace_sha256: String,
    calibration_certificate_sha256: String,
    measurement_method_id: String,
    instrument_id: String,
    timebase_receipt_sha256: String,
}

#[derive(Debug, Serialize)]
struct ContributionReport {
    contribution_id: String,
    event_id: String,
    receiver_id: String,
    band_id: String,
    vessel_id: String,
    vessel_revision: String,
    propeller_revision: String,
    source_state_id: String,
    source_track_sha256: String,
    operating_point_receipt_sha256: String,
    source_level_method_id: String,
    source_level_raw_trace_sha256: String,
    source_level_calibration_sha256: String,
    source_level_db: f64,
    source_level_expanded_uncertainty_db: f64,
    source_level_uncertainty_confidence_level: f64,
    transmission_loss_db: f64,
    transmission_loss_expanded_uncertainty_db: f64,
    transmission_loss_uncertainty_confidence_level: f64,
    source_to_propagation_sync_receipt_sha256: String,
    received_level_db: f64,
    received_level_lower_db: f64,
    received_level_upper_db: f64,
    source_latitude_deg: f64,
    source_longitude_deg: f64,
    source_depth_m: f64,
    range_m: f64,
    receiver_latitude_deg: f64,
    receiver_longitude_deg: f64,
    receiver_depth_m: f64,
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

fn haversine_slant_distance_m(
    lat_a: f64, lon_a: f64, depth_a_m: f64,
    lat_b: f64, lon_b: f64, depth_b_m: f64,
) -> f64 {
    let to_radians = std::f64::consts::PI / 180.0;
    let lat_a = lat_a * to_radians;
    let lat_b = lat_b * to_radians;
    let delta_lat = (lat_b - lat_a) / 2.0;
    let delta_lon = (lon_b - lon_a) * to_radians / 2.0;
    let a = delta_lat.sin().powi(2)
        + lat_a.cos() * lat_b.cos() * delta_lon.sin().powi(2);
    let horizontal = 2.0 * EARTH_RADIUS_M * a.sqrt().asin();
    horizontal.hypot(depth_b_m - depth_a_m)
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
    if case.source_events.is_empty() || case.receivers.is_empty() || case.contributions.is_empty() {
        errors.push("at least one source event, receiver and acoustic contribution are required".to_string());
    }
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
                || !band.uncertainty_confidence_level.is_finite()
                || band.uncertainty_confidence_level <= 0.0
                || band.uncertainty_confidence_level >= 1.0
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
            let calculated = haversine_slant_distance_m(
                contribution.source_latitude_deg,
                contribution.source_longitude_deg,
                contribution.source_depth_m,
                receiver.latitude_deg,
                receiver.longitude_deg,
                receiver.depth_m,
            );
            if (calculated-contribution.range_m).abs() > contribution.range_tolerance_m {
                errors.push(format!("contribution {} declared source/receiver range differs from coordinates", contribution.contribution_id));
            }
        }
        if !contribution.transmission_loss_db.is_finite()
            || !finite_positive(contribution.transmission_loss_expanded_uncertainty_db)
            || !contribution.transmission_loss_uncertainty_confidence_level.is_finite()
            || contribution.transmission_loss_uncertainty_confidence_level <= 0.0
            || contribution.transmission_loss_uncertainty_confidence_level >= 1.0
            || contribution.propagation_model_id.trim().is_empty()
            || contribution.propagation_model_version.trim().is_empty()
        {
            errors.push(format!("contribution {} has invalid transmission loss or propagation model identity", contribution.contribution_id));
        }
        for (label,digest) in [
            ("source/propagation synchronization receipt",&contribution.source_to_propagation_sync_receipt_sha256),
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
            if (band.uncertainty_confidence_level - contribution.transmission_loss_uncertainty_confidence_level).abs() > 1.0e-12 {
                errors.push(format!("contribution {} source-level and transmission-loss uncertainty confidence levels differ", contribution.contribution_id));
            }
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
            || observation.measurement_method_id.trim().is_empty()
            || observation.instrument_id.trim().is_empty()
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

    let mut ambient_grouped: BTreeMap<(String,String),Vec<&AmbientObservation>> = BTreeMap::new();
    for observation in &case.ambient_observations {
        ambient_grouped.entry((observation.receiver_id.clone(),observation.band_id.clone()))
            .or_default().push(observation);
    }
    for ((receiver_id,band_id),mut records) in ambient_grouped {
        records.sort_by(|a,b|a.start_time_s.total_cmp(&b.start_time_s));
        for pair in records.windows(2) {
            if pair[1].start_time_s < pair[0].end_time_s - TIME_TOLERANCE_S {
                errors.push(format!("ambient reference for receiver {receiver_id}, band {band_id} contains overlapping samples that would double-count ambient SEL"));
            }
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


#[derive(Debug, Serialize)]
struct AmbientObservationRef {
    observation_id: String,
    raw_trace_sha256: String,
    calibration_certificate_sha256: String,
    measurement_method_id: String,
    timebase_receipt_sha256: String,
}

#[derive(Debug)]
struct CalculatedContribution<'a> {
    input: &'a AcousticContribution,
    source_event: &'a SourceEvent,
    receiver: &'a Receiver,
    source_level_db: f64,
    source_level_uncertainty_db: f64,
    source_level_confidence_level: f64,
    transmission_loss_confidence_level: f64,
    received_level_db: f64,
    received_level_lower_db: f64,
    received_level_upper_db: f64,
    sel_db: f64,
    sel_lower_db: f64,
    sel_upper_db: f64,
}

fn calculate_contribution<'a>(
    contribution: &'a AcousticContribution,
    event: &'a SourceEvent,
    receiver: &'a Receiver,
) -> Option<CalculatedContribution<'a>> {
    let band = event.source_levels.iter().find(|band| band.band_id == contribution.band_id)?;
    if (band.uncertainty_confidence_level - contribution.transmission_loss_uncertainty_confidence_level).abs() > 1.0e-12 {
        return None;
    }
    let uncertainty = band.expanded_uncertainty_db
        + contribution.transmission_loss_expanded_uncertainty_db;
    let received = band.source_level_db - contribution.transmission_loss_db;
    let lower = received - uncertainty;
    let upper = received + uncertainty;
    let duration_adjustment = 10.0 * contribution.duration_s.log10();
    let sel = received + duration_adjustment;
    let sel_lower = lower + duration_adjustment;
    let sel_upper = upper + duration_adjustment;
    if ![received,lower,upper,sel,sel_lower,sel_upper].iter().all(|value| value.is_finite()) {
        return None;
    }
    Some(CalculatedContribution {
        input: contribution,
        source_event: event,
        receiver,
        source_level_db: band.source_level_db,
        source_level_uncertainty_db: band.expanded_uncertainty_db,
        source_level_confidence_level: band.uncertainty_confidence_level,
        transmission_loss_confidence_level: contribution.transmission_loss_uncertainty_confidence_level,
        received_level_db: received,
        received_level_lower_db: lower,
        received_level_upper_db: upper,
        sel_db: sel,
        sel_lower_db: sel_lower,
        sel_upper_db: sel_upper,
    })
}

fn maximum_simultaneous_level<F>(
    contributions: &[CalculatedContribution<'_>],
    level: F,
) -> Option<f64>
where
    F: Fn(&CalculatedContribution<'_>) -> f64,
{
    let mut boundaries: Vec<(f64, bool, usize, f64)> = Vec::new();
    for (index, contribution) in contributions.iter().enumerate() {
        boundaries.push((contribution.input.start_time_s, true, index, level(contribution)));
        boundaries.push((contribution.input.end_time_s, false, index, level(contribution)));
    }
    boundaries.sort_by(|a,b| a.0.total_cmp(&b.0).then_with(||a.1.cmp(&b.1)));
    let mut active:BTreeMap<usize,f64>=BTreeMap::new();
    let mut maximum:Option<f64>=None;
    let mut index=0;
    while index<boundaries.len() {
        let time=boundaries[index].0;
        let mut end=index;
        while end<boundaries.len() && boundaries[end].0==time { end+=1; }
        // Half-open intervals [start,end): remove ended contributions first.
        for (_,is_start,id,_) in &boundaries[index..end] {
            if !*is_start { active.remove(id); }
        }
        for (_,is_start,id,level) in &boundaries[index..end] {
            if *is_start { active.insert(*id,*level); }
        }
        if !active.is_empty() {
            let levels:Vec<f64>=active.values().copied().collect();
            if let Some(total)=log_energy_sum(&levels) {
                maximum=Some(maximum.map_or(total,|current|current.max(total)));
            }
        }
        index=end;
    }
    maximum
}

fn calculate_ambient_sel(
    observations: &[&AmbientObservation],
) -> (Option<f64>,Option<f64>,Option<f64>,Vec<AmbientObservationRef>) {
    let mut central=Vec::new();
    let mut lower=Vec::new();
    let mut upper=Vec::new();
    let mut references=Vec::new();
    for observation in observations {
        let adjustment=10.0*observation.duration_s.log10();
        central.push(observation.ambient_spl_db+adjustment);
        lower.push(observation.ambient_spl_db-observation.expanded_uncertainty_db+adjustment);
        upper.push(observation.ambient_spl_db+observation.expanded_uncertainty_db+adjustment);
        references.push(AmbientObservationRef {
            observation_id:observation.observation_id.clone(),
            raw_trace_sha256:observation.raw_trace_sha256.clone(),
            calibration_certificate_sha256:observation.calibration_certificate_sha256.clone(),
            measurement_method_id:observation.measurement_method_id.clone(),
            timebase_receipt_sha256:observation.timebase_receipt_sha256.clone(),
        });
    }
    (
        log_energy_sum(&central),
        log_energy_sum(&lower),
        log_energy_sum(&upper),
        references,
    )
}

fn audit(case:&Case)->AuditReport {
    let validation_errors=validate_case(case);
    let mut reasons=validation_errors.clone();
    let mut grouped:BTreeMap<(String,String),Vec<CalculatedContribution>>=BTreeMap::new();
    if validation_errors.is_empty() {
        for contribution in &case.contributions {
            let event=case.source_events.iter().find(|event|event.event_id==contribution.event_id).unwrap();
            let receiver=case.receivers.iter().find(|receiver|receiver.receiver_id==contribution.receiver_id).unwrap();
            if let Some(calculated)=calculate_contribution(contribution,event,receiver) {
                grouped.entry((contribution.receiver_id.clone(),contribution.band_id.clone()))
                    .or_default().push(calculated);
            } else {
                reasons.push(format!("contribution {} could not be converted to finite received-level/SEL intervals",contribution.contribution_id));
            }
        }
    }

    let mut partitions=Vec::new();
    for ((receiver_id,band_id),mut contributions) in grouped {
        contributions.sort_by(|a,b|a.input.start_time_s.total_cmp(&b.input.start_time_s)
            .then_with(||a.input.contribution_id.cmp(&b.input.contribution_id)));
        let central:Vec<f64>=contributions.iter().map(|item|item.sel_db).collect();
        let lower:Vec<f64>=contributions.iter().map(|item|item.sel_lower_db).collect();
        let upper:Vec<f64>=contributions.iter().map(|item|item.sel_upper_db).collect();
        let Some(ship_sel)=log_energy_sum(&central) else {
            reasons.push(format!("receiver {receiver_id} band {band_id} ship SEL could not be accumulated"));
            continue;
        };
        let Some(ship_lower)=log_energy_sum(&lower) else { continue; };
        let Some(ship_upper)=log_energy_sum(&upper) else { continue; };
        let intervals:Vec<(f64,f64)>=contributions.iter()
            .map(|item|(item.input.start_time_s,item.input.end_time_s)).collect();
        let unique_coverage=union_duration(&intervals);
        let summed_duration: f64=contributions.iter().map(|item|item.input.duration_s).sum();
        let unique_events:BTreeSet<&str>=contributions.iter().map(|item|item.input.event_id.as_str()).collect();
        let max_individual=contributions.iter().map(|item|item.received_level_db).fold(f64::NEG_INFINITY,f64::max);
        let max_individual_lower=contributions.iter().map(|item|item.received_level_lower_db).fold(f64::NEG_INFINITY,f64::max);
        let max_individual_upper=contributions.iter().map(|item|item.received_level_upper_db).fold(f64::NEG_INFINITY,f64::max);
        let max_simultaneous=maximum_simultaneous_level(&contributions,|item|item.received_level_db).unwrap_or(f64::NEG_INFINITY);
        let max_simultaneous_lower=maximum_simultaneous_level(&contributions,|item|item.received_level_lower_db).unwrap_or(f64::NEG_INFINITY);
        let max_simultaneous_upper=maximum_simultaneous_level(&contributions,|item|item.received_level_upper_db).unwrap_or(f64::NEG_INFINITY);
        let contribution_reports=contributions.iter().map(|item|ContributionReport {
            contribution_id:item.input.contribution_id.clone(),
            event_id:item.input.event_id.clone(),
            receiver_id:item.input.receiver_id.clone(),
            band_id:item.input.band_id.clone(),
            vessel_id:item.source_event.vessel_id.clone(),
            vessel_revision:item.source_event.vessel_revision.clone(),
            propeller_revision:item.source_event.propeller_revision.clone(),
            source_state_id:item.source_event.source_state_id.clone(),
            source_track_sha256:item.source_event.source_track_sha256.clone(),
            operating_point_receipt_sha256:item.source_event.operating_point_receipt_sha256.clone(),
            source_level_method_id:item.source_event.source_level_method_id.clone(),
            source_level_raw_trace_sha256:item.source_event.source_level_raw_trace_sha256.clone(),
            source_level_calibration_sha256:item.source_event.source_level_calibration_sha256.clone(),
            source_level_db:item.source_level_db,
            source_level_expanded_uncertainty_db:item.source_level_uncertainty_db,
            source_level_uncertainty_confidence_level:item.source_level_confidence_level,
            transmission_loss_db:item.input.transmission_loss_db,
            transmission_loss_expanded_uncertainty_db:item.input.transmission_loss_expanded_uncertainty_db,
            transmission_loss_uncertainty_confidence_level:item.transmission_loss_confidence_level,
            source_to_propagation_sync_receipt_sha256:item.input.source_to_propagation_sync_receipt_sha256.clone(),
            received_level_db:item.received_level_db,
            received_level_lower_db:item.received_level_lower_db,
            received_level_upper_db:item.received_level_upper_db,
            duration_s:item.input.duration_s,
            sel_db:item.sel_db,
            sel_lower_db:item.sel_lower_db,
            sel_upper_db:item.sel_upper_db,
            source_latitude_deg:item.input.source_latitude_deg,
            source_longitude_deg:item.input.source_longitude_deg,
            source_depth_m:item.input.source_depth_m,
            range_m:item.input.range_m,
            receiver_latitude_deg:item.receiver.latitude_deg,
            receiver_longitude_deg:item.receiver.longitude_deg,
            receiver_depth_m:item.receiver.depth_m,
            propagation_model_id:item.input.propagation_model_id.clone(),
            propagation_model_version:item.input.propagation_model_version.clone(),
            propagation_validation_sha256:item.input.propagation_validation_sha256.clone(),
            bathymetry_sha256:item.input.bathymetry_sha256.clone(),
            sound_speed_profile_sha256:item.input.sound_speed_profile_sha256.clone(),
            seabed_model_sha256:item.input.seabed_model_sha256.clone(),
            environment_fingerprint_sha256:item.input.environment_fingerprint_sha256.clone(),
            timebase_receipt_sha256:item.input.timebase_receipt_sha256.clone(),
        }).collect();

        let ambient_refs:Vec<&AmbientObservation>=case.ambient_observations.iter()
            .filter(|obs|obs.receiver_id==receiver_id&&obs.band_id==band_id).collect();
        let (ambient_sel,ambient_lower,ambient_upper,ambient_observation_refs)=calculate_ambient_sel(&ambient_refs);
        partitions.push(ExposurePartitionReport {
            receiver_id:receiver_id.clone(),
            band_id:band_id.clone(),
            contribution_count:contributions.len(),
            unique_source_event_count:unique_events.len(),
            unique_temporal_coverage_s:unique_coverage,
            summed_source_contribution_duration_s:summed_duration,
            ship_only_cumulative_sel_db:ship_sel,
            ship_only_cumulative_sel_lower_db:ship_lower,
            ship_only_cumulative_sel_upper_db:ship_upper,
            maximum_individual_contribution_received_level_db:max_individual,
            maximum_individual_contribution_received_level_lower_db:max_individual_lower,
            maximum_individual_contribution_received_level_upper_db:max_individual_upper,
            maximum_simultaneous_ship_received_level_db:max_simultaneous,
            maximum_simultaneous_ship_received_level_lower_db:max_simultaneous_lower,
            maximum_simultaneous_ship_received_level_upper_db:max_simultaneous_upper,
            contribution_reports,
            ambient_observation_count:ambient_refs.len(),
            ambient_reference_sel_db:ambient_sel,
            ambient_reference_sel_lower_db:ambient_lower,
            ambient_reference_sel_upper_db:ambient_upper,
            ambient_observation_refs,
            ambient_only_basis:"separate_ambient_reference_not_added_to_ship_energy",
        });
    }

    let mut limit_results=Vec::new();
    let mut any_limit_failed=false;
    let mut any_limit_unresolved=false;
    for limit in &case.exposure_limits {
        let partition=partitions.iter().find(|item|item.receiver_id==limit.receiver_id&&item.band_id==limit.band_id);
        let (lower,upper,status)=match partition {
            Some(partition)=>{
                let lower=partition.ship_only_cumulative_sel_lower_db;
                let upper=partition.ship_only_cumulative_sel_upper_db;
                let status=if upper<=limit.upper_sel_threshold_db {
                    "pass"
                } else if lower>limit.upper_sel_threshold_db {
                    any_limit_failed=true;
                    "fail"
                } else {
                    any_limit_unresolved=true;
                    "unresolved"
                };
                (Some(lower),Some(upper),status)
            }
            None=>{
                any_limit_unresolved=true;
                (None,None,"missing")
            }
        };
        limit_results.push(LimitResult {
            limit_id:limit.limit_id.clone(),
            receiver_id:limit.receiver_id.clone(),
            band_id:limit.band_id.clone(),
            threshold_db:limit.upper_sel_threshold_db,
            observed_lower_db:lower,
            observed_upper_db:upper,
            status,
            requirement_reference:limit.requirement_reference.clone(),
        });
    }

    if case.source_events.iter().any(|event|event.evidence_kind!=EvidenceKind::Measured
        || event.rights_status!=RightsStatus::PublicOrPermitted
        || !event.independently_reviewed) {
        reasons.push("one or more source events are synthetic, unreviewed or not rights-cleared".to_string());
    }
    if case.ambient_observations.iter().any(|obs|obs.evidence_kind!=EvidenceKind::Measured
        || obs.rights_status!=RightsStatus::PublicOrPermitted
        || !obs.independently_reviewed) {
        reasons.push("one or more ambient observations are synthetic, unreviewed or not rights-cleared".to_string());
    }

    let evidence_incomplete=case.source_events.iter().any(|event|event.evidence_kind!=EvidenceKind::Measured
        || event.rights_status!=RightsStatus::PublicOrPermitted
        || !event.independently_reviewed)
        || case.ambient_observations.iter().any(|obs|obs.evidence_kind!=EvidenceKind::Measured
            || obs.rights_status!=RightsStatus::PublicOrPermitted
            || !obs.independently_reviewed);
    let disposition=if !validation_errors.is_empty() {
        "INVALID"
    } else if any_limit_failed {
        "EXPOSURE_LIMIT_FAILED"
    } else if evidence_incomplete||any_limit_unresolved {
        "INCOMPLETE"
    } else {
        "EXPOSURE_LEDGER_COMPUTED"
    };

    AuditReport {
        schema_version:SCHEMA_VERSION,
        case_id:case.case_id.clone(),
        period_id:case.period.period_id.clone(),
        disposition,
        partitions,
        limit_results,
        claim_ceiling:"frequency_resolved_source_to_receiver_ship_only_cumulative_sel_and_separate_ambient_reference_only_no_propagation_solver_ecological_effect_or_regulatory_compliance_claim",
        reasons,
    }
}

fn main() {
    let mut args=env::args().skip(1);
    let Some(path)=args.next() else {
        eprintln!("usage: marine-acoustic-exposure-ledger-audit <case.json>");
        process::exit(64);
    };
    if args.next().is_some() {
        eprintln!("usage: marine-acoustic-exposure-ledger-audit <case.json>");
        process::exit(64);
    }
    let input=match fs::read_to_string(&path) {
        Ok(value)=>value,
        Err(error)=>{
            eprintln!("could not read {path}: {error}");
            process::exit(64);
        }
    };
    let case:Case=match serde_json::from_str(&input) {
        Ok(value)=>value,
        Err(error)=>{
            eprintln!("invalid case JSON: {error}");
            process::exit(64);
        }
    };
    let report=audit(&case);
    println!("{}",serde_json::to_string_pretty(&report).expect("report serializes"));
    match report.disposition {
        "EXPOSURE_LEDGER_COMPUTED"|"EXPOSURE_LIMIT_FAILED"=>{}
        "INCOMPLETE"=>process::exit(3),
        _=>process::exit(2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(c:char)->String { std::iter::repeat_n(c,64).collect() }

    fn source_event(id:&str,level:f64,uncertainty:f64)->SourceEvent {
        SourceEvent {
            event_id:id.to_string(),
            evidence_kind:EvidenceKind::Measured,
            dataset_id:format!("dataset-{id}"),
            citation:"controlled ship trial".to_string(),
            rights_status:RightsStatus::PublicOrPermitted,
            independently_reviewed:true,
            vessel_id:"vessel-1".to_string(),
            vessel_revision:"hull-r1".to_string(),
            propeller_revision:"prop-r1".to_string(),
            source_state_id:"propeller-operating-state-1".to_string(),
            mission_id:"mission-1".to_string(),
            route_id:"route-1".to_string(),
            service_fingerprint_sha256:digest('1'),
            environment_fingerprint_sha256:digest('2'),
            source_track_sha256:digest(if id=="event-a"{'a'}else{'b'}),
            source_level_method_id:"measured_source_spectrum_v1".to_string(),
            source_level_calibration_sha256:digest('c'),
            source_level_raw_trace_sha256:digest(if id=="event-a"{'d'}else{'e'}),
            source_level_timebase_receipt_sha256:digest('f'),
            operating_point_receipt_sha256:digest('9'),
            source_levels:vec![SourceLevelBand {
                band_id:"one-third-octave-125Hz".to_string(),
                quantity_id:"source_level_db_re_1upa_at_1m".to_string(),
                center_frequency_hz:125.0,
                source_level_db:level,
                expanded_uncertainty_db:uncertainty,
                uncertainty_confidence_level:0.95,
            }],
        }
    }

    fn receiver()->Receiver {
        Receiver {
            receiver_id:"receiver-1".to_string(),
            latitude_deg:0.0,
            longitude_deg:0.001,
            depth_m:20.0,
            location_evidence_sha256:digest('7'),
        }
    }

    fn contribution(id:&str,event_id:&str,start:f64,end:f64,tl:f64,tl_u:f64)->AcousticContribution {
        AcousticContribution {
            contribution_id:id.to_string(),
            event_id:event_id.to_string(),
            receiver_id:"receiver-1".to_string(),
            band_id:"one-third-octave-125Hz".to_string(),
            start_time_s:start,
            end_time_s:end,
            duration_s:end-start,
            source_latitude_deg:0.0,
            source_longitude_deg:0.0,
            source_depth_m:10.0,
            range_m:111.195,
            range_tolerance_m:2.0,
            transmission_loss_db:tl,
            transmission_loss_expanded_uncertainty_db:tl_u,
            transmission_loss_uncertainty_confidence_level:0.95,
            source_to_propagation_sync_receipt_sha256:digest('6'),
            propagation_model_id:"bellhop".to_string(),
            propagation_model_version:"test-v1".to_string(),
            propagation_validation_sha256:digest('a'),
            bathymetry_sha256:digest('b'),
            sound_speed_profile_sha256:digest('c'),
            seabed_model_sha256:digest('d'),
            environment_fingerprint_sha256:digest('2'),
            timebase_receipt_sha256:digest('e'),
        }
    }

    fn ambient()->AmbientObservation {
        AmbientObservation {
            observation_id:"ambient-1".to_string(),
            evidence_kind:EvidenceKind::Measured,
            rights_status:RightsStatus::PublicOrPermitted,
            independently_reviewed:true,
            receiver_id:"receiver-1".to_string(),
            band_id:"one-third-octave-125Hz".to_string(),
            start_time_s:0.0,
            end_time_s:100.0,
            duration_s:100.0,
            ambient_spl_db:50.0,
            expanded_uncertainty_db:1.0,
            measurement_method_id:"ambient-monitoring-v1".to_string(),
            instrument_id:"hydrophone-1".to_string(),
            calibration_certificate_sha256:digest('f'),
            raw_trace_sha256:digest('8'),
            timebase_receipt_sha256:digest('9'),
            ambient_only_basis:"independently_measured_ambient_reference".to_string(),
        }
    }

    fn case()->Case {
        Case {
            schema_version:SCHEMA_VERSION.to_string(),
            case_id:"synthetic-test-case".to_string(),
            period:ExposurePeriod {
                period_id:"one-hour-window".to_string(),
                start_time_s:0.0,
                end_time_s:3600.0,
                service_fingerprint_sha256:digest('1'),
                environment_fingerprint_sha256:digest('2'),
            },
            source_events:vec![source_event("event-a",120.0,0.5),source_event("event-b",119.0,0.5)],
            receivers:vec![receiver()],
            contributions:vec![
                contribution("contrib-a","event-a",0.0,10.0,60.0,0.5),
                contribution("contrib-b","event-b",0.0,10.0,60.0,0.5),
            ],
            ambient_observations:vec![ambient()],
            exposure_limits:vec![ExposureLimit {
                limit_id:"ship-sel-limit".to_string(),
                receiver_id:"receiver-1".to_string(),
                band_id:"one-third-octave-125Hz".to_string(),
                upper_sel_threshold_db:80.0,
                requirement_reference:"scenario-limit:receiver-1-125hz".to_string(),
            }],
        }
    }

    #[test]
    fn uncertainty_confidence_level_mismatch_is_invalid() {
        let mut input=case();
        input.contributions[0].transmission_loss_uncertainty_confidence_level=0.99;
        assert_eq!(audit(&input).disposition,"INVALID");
    }

    #[test]
    fn source_to_propagation_synchronization_receipt_is_retained() {
        let report=audit(&case());
        assert_eq!(
            report.partitions[0].contribution_reports[0].source_to_propagation_sync_receipt_sha256,
            digest('6')
        );
    }

    #[test]
    fn simultaneous_received_level_energy_sums_only_overlapping_events() {
        let report=audit(&case());
        let partition=&report.partitions[0];
        let expected=log_energy_sum(&[60.0,59.0]).unwrap();
        assert!((partition.maximum_simultaneous_ship_received_level_db-expected).abs()<1e-12);
        assert!((partition.maximum_individual_contribution_received_level_db-60.0).abs()<1e-12);
    }

    #[test]
    fn sequential_events_do_not_get_added_to_instantaneous_peak() {
        let mut input=case();
        input.contributions[1].start_time_s=10.0;
        input.contributions[1].end_time_s=20.0;
        assert!((audit(&input).partitions[0].maximum_simultaneous_ship_received_level_db-60.0).abs()<1e-12);
        assert!((audit(&input).partitions[0].unique_temporal_coverage_s-20.0).abs()<1e-12);
    }

    #[test]
    fn received_level_is_source_level_minus_transmission_loss() {
        let input=case();
        let report=audit(&input);
        let contribution=&report.partitions[0].contribution_reports[0];
        assert!((contribution.received_level_db-60.0).abs()<1e-12);
    }

    #[test]
    fn single_segment_sel_adds_ten_log_duration() {
        let mut input=case();
        input.contributions.truncate(1);
        let report=audit(&input);
        assert!((report.partitions[0].ship_only_cumulative_sel_db-70.0).abs()<1e-12);
    }

    #[test]
    fn overlapping_independent_ship_events_are_energy_summed_not_db_added() {
        let report=audit(&case());
        let expected=log_energy_sum(&[70.0,69.0]).unwrap();
        assert!((report.partitions[0].ship_only_cumulative_sel_db-expected).abs()<1e-12);
        assert!(report.partitions[0].ship_only_cumulative_sel_db<100.0);
        assert!((report.partitions[0].unique_temporal_coverage_s-10.0).abs()<1e-12);
        assert!((report.partitions[0].summed_source_contribution_duration_s-20.0).abs()<1e-12);
    }

    #[test]
    fn ambient_reference_is_not_added_to_ship_only_sel() {
        let report=audit(&case());
        let partition=&report.partitions[0];
        assert!(partition.ambient_reference_sel_db.is_some());
        assert_eq!(partition.ambient_only_basis,"separate_ambient_reference_not_added_to_ship_energy");
        assert!((partition.ship_only_cumulative_sel_db-log_energy_sum(&[70.0,69.0]).unwrap()).abs()<1e-12);
    }

    #[test]
    fn source_and_propagation_uncertainties_expand_received_sel_bounds() {
        let report=audit(&case());
        let contribution=&report.partitions[0].contribution_reports[0];
        assert!((contribution.received_level_lower_db-59.0).abs()<1e-12);
        assert!((contribution.received_level_upper_db-61.0).abs()<1e-12);
        assert!((contribution.sel_lower_db-69.0).abs()<1e-12);
        assert!((contribution.sel_upper_db-71.0).abs()<1e-12);
    }

    #[test]
    fn upper_exposure_limit_passes_only_when_full_interval_is_below() {
        let report=audit(&case());
        assert_eq!(report.limit_results[0].status,"pass");
    }

    #[test]
    fn uncertainty_overlapping_exposure_limit_is_unresolved() {
        let mut input=case();
        input.exposure_limits[0].upper_sel_threshold_db=72.0;
        let report=audit(&input);
        assert_eq!(report.limit_results[0].status,"unresolved");
        assert_eq!(report.disposition,"INCOMPLETE");
    }

    #[test]
    fn exposure_limit_fails_only_when_lower_interval_is_above() {
        let mut input=case();
        input.exposure_limits[0].upper_sel_threshold_db=68.0;
        let report=audit(&input);
        assert_eq!(report.limit_results[0].status,"fail");
        assert_eq!(report.disposition,"EXPOSURE_LIMIT_FAILED");
    }

    #[test]
    fn synthetic_source_events_never_qualify_exposure_ledger() {
        let mut input=case();
        input.source_events[0].evidence_kind=EvidenceKind::Synthetic;
        let report=audit(&input);
        assert_eq!(report.disposition,"INCOMPLETE");
    }

    #[test]
    fn unknown_source_rights_hold_exposure_claim() {
        let mut input=case();
        input.source_events[0].rights_status=RightsStatus::Unknown;
        assert_eq!(audit(&input).disposition,"INCOMPLETE");
    }

    #[test]
    fn duplicate_contribution_id_is_invalid() {
        let mut input=case();
        input.contributions[1].contribution_id=input.contributions[0].contribution_id.clone();
        assert_eq!(audit(&input).disposition,"INVALID");
    }

    #[test]
    fn overlapping_same_event_receiver_band_intervals_are_invalid() {
        let mut input=case();
        input.contributions[1].event_id="event-a".to_string();
        input.contributions[1].start_time_s=5.0;
        input.contributions[1].end_time_s=15.0;
        input.contributions[1].duration_s=10.0;
        assert_eq!(audit(&input).disposition,"INVALID");
    }

    #[test]
    fn range_mismatch_against_receiver_coordinates_is_invalid() {
        let mut input=case();
        input.contributions[0].range_m=500.0;
        assert_eq!(audit(&input).disposition,"INVALID");
    }

    #[test]
    fn missing_propagation_validation_reference_is_invalid() {
        let mut input=case();
        input.contributions[0].propagation_validation_sha256="bad".to_string();
        assert_eq!(audit(&input).disposition,"INVALID");
    }

    #[test]
    fn source_environment_mismatch_is_invalid() {
        let mut input=case();
        input.contributions[0].environment_fingerprint_sha256=digest('3');
        assert_eq!(audit(&input).disposition,"INVALID");
    }

    #[test]
    fn ambient_method_and_instrument_are_required() {
        let mut input=case();
        input.ambient_observations[0].measurement_method_id.clear();
        assert_eq!(audit(&input).disposition,"INVALID");
    }

    #[test]
    fn slant_range_includes_source_receiver_depth_difference() {
        let expected = (111.195_f64.powi(2) + 10.0_f64.powi(2)).sqrt();
        let calculated = haversine_slant_distance_m(0.0,0.0,10.0,0.0,0.001,20.0);
        assert!((calculated-expected).abs()<1.0e-6);
    }

    #[test]
    fn overlapping_ambient_reference_windows_are_invalid() {
        let mut input=case();
        let mut duplicate=ambient();
        duplicate.observation_id="ambient-2".to_string();
        duplicate.start_time_s=50.0;
        duplicate.end_time_s=150.0;
        duplicate.duration_s=100.0;
        input.ambient_observations.push(duplicate);
        assert_eq!(audit(&input).disposition,"INVALID");
    }

    #[test]
    fn exposure_receipt_preserves_source_and_receiver_lineage() {
        let report=audit(&case());
        let receipt=&report.partitions[0].contribution_reports[0];
        assert_eq!(receipt.vessel_revision,"hull-r1");
        assert_eq!(receipt.propeller_revision,"prop-r1");
        assert_eq!(receipt.source_state_id,"propeller-operating-state-1");
        assert_eq!(receipt.receiver_depth_m,20.0);
        assert_eq!(receipt.operating_point_receipt_sha256,digest('9'));
        assert_eq!(report.partitions[0].ambient_observation_refs[0].instrument_id,"hydrophone-1");
    }

    #[test]
    fn unmatched_source_frequency_band_is_invalid() {
        let mut input=case();
        input.contributions[0].band_id="one-third-octave-250Hz".to_string();
        assert_eq!(audit(&input).disposition,"INVALID");
    }
}

