use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::{env, fs, process};

const SCHEMA_VERSION: &str = "0.1.0";
const DEEP_WATER_MIN_M: f64 = 150.0;
const DEEP_WATER_LENGTH_MULTIPLE: f64 = 1.5;
const IACS_MIN_CPA_M: f64 = 100.0;
const IACS_MIN_DATA_WINDOW_S: f64 = 10.0;
const BOUNDARY_EPSILON_M: f64 = 1.0e-6;

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum EvidenceKind { Synthetic, Measured }

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RightsStatus { PublicOrPermitted, Unknown, Restricted }

#[derive(Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum Substrate {
    Sand,
    Rock,
    Mud,
    Gravel,
    Mixed,
    Other,
    Unknown,
}

impl Substrate {
    fn name(self) -> &'static str {
        match self {
            Self::Sand => "sand",
            Self::Rock => "rock",
            Self::Mud => "mud",
            Self::Gravel => "gravel",
            Self::Mixed => "mixed",
            Self::Other => "other",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum RunAspect { Port, Starboard }

impl RunAspect {
    fn name(self) -> &'static str {
        match self { Self::Port => "port", Self::Starboard => "starboard" }
    }
}

#[derive(Debug, Deserialize, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum AcousticQuantity {
    ReceivedSpl,
    RadiatedNoiseLevel,
    SourceLevel,
    DeepWaterEquivalentRadiatedNoiseLevel,
}

impl AcousticQuantity {
    fn name(self) -> &'static str {
        match self {
            Self::ReceivedSpl => "received_spl",
            Self::RadiatedNoiseLevel => "radiated_noise_level",
            Self::SourceLevel => "source_level",
            Self::DeepWaterEquivalentRadiatedNoiseLevel => "deep_water_equivalent_radiated_noise_level",
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    schema_version: String,
    case_id: String,
    evidence_kind: EvidenceKind,
    source: SourceEvidence,
    minimum_runs: usize,
    maximum_depth_range_fraction: f64,
    maximum_seabed_sound_speed_range_fraction: f64,
    maximum_ctd_sound_speed_range_fraction: f64,
    maximum_stw_relative_deviation: f64,
    maximum_cross_site_seabed_sound_speed_difference_fraction: f64,
    campaigns: Vec<Campaign>,
    propagation_transfer_receipts: Vec<PropagationTransferReceipt>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceEvidence {
    dataset_id: String,
    citation: String,
    raw_data_sha256: String,
    rights_status: RightsStatus,
    independently_reviewed: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Campaign {
    campaign_id: String,
    site: Site,
    vessel: Vessel,
    hydrophones: Vec<Hydrophone>,
    in_situ_pre_deployment_calibration_sha256: String,
    in_situ_post_recovery_calibration_sha256: String,
    runs: Vec<Run>,
    bands: Vec<Band>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Site {
    site_id: String,
    site_revision: String,
    bathymetry_source_sha256: String,
    bathymetry_method_id: String,
    water_depth_samples: Vec<WaterDepthSample>,
    seabed_profile_source_sha256: String,
    seabed_method_id: String,
    seabed_samples: Vec<SeabedSample>,
    ctd_profile: Option<CtdProfile>,
    current: Option<CurrentObservation>,
    ambient_background: Option<AmbientBackground>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WaterDepthSample {
    sample_id: String,
    along_track_position_m: f64,
    cross_track_position_m: f64,
    depth_m: f64,
    expanded_uncertainty_m: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SeabedSample {
    sample_id: String,
    along_track_position_m: f64,
    cross_track_position_m: f64,
    substrate: Substrate,
    sound_speed_m_s: f64,
    expanded_uncertainty_m_s: f64,
    evidence_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CtdProfile {
    instrument_id: String,
    method_id: String,
    calibration_certificate_sha256: String,
    evidence_sha256: String,
    samples: Vec<CtdSample>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CtdSample {
    depth_m: f64,
    sound_speed_m_s: f64,
    expanded_uncertainty_m_s: f64,
    temperature_c: f64,
    salinity_psu: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CurrentObservation {
    instrument_id: String,
    method_id: String,
    evidence_sha256: String,
    speed_m_s: f64,
    direction_deg_true: f64,
    observed_at_utc: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AmbientBackground {
    measurement_id: String,
    method_id: String,
    raw_trace_sha256: String,
    measured_at_utc: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Vessel {
    vessel_id: String,
    overall_ship_length_m: f64,
    acoustic_ship_length_m: f64,
    acoustic_reference_point_id: String,
    gps_to_acoustic_centre_offset_m: f64,
    propulsion_configuration_sha256: String,
    service_fingerprint_sha256: String,
    loading_fingerprint_sha256: String,
    draft_m: f64,
    hull_condition: String,
    propeller_condition: String,
    last_dry_dock_date: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Hydrophone {
    instrument_id: String,
    x_m: f64,
    y_m: f64,
    depth_m: f64,
    orientation_azimuth_deg: f64,
    orientation_elevation_deg: f64,
    calibration_certificate_sha256: String,
    calibrated_min_frequency_hz: f64,
    calibrated_max_frequency_hz: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Run {
    run_id: String,
    aspect: RunAspect,
    timebase_receipt_sha256: String,
    operating_point_receipt_sha256: String,
    background_trace_sha256: String,
    propulsion_configuration_sha256: String,
    service_fingerprint_sha256: String,
    loading_fingerprint_sha256: String,
    draft_m: f64,
    hull_condition: String,
    propeller_condition: String,
    speed_through_water_knots: Option<f64>,
    speed_over_ground_knots: Option<f64>,
    shaft_rotation_rate_rpm: f64,
    propulsion_power_kw: f64,
    heading_deg_true: f64,
    cpa_distance_m: f64,
    data_window_duration_s: f64,
    run_start_utc: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Band {
    band_id: String,
    quantity: AcousticQuantity,
    center_frequency_hz: f64,
    site_cutoff_frequency_hz: Option<f64>,
    propagation_method_id: String,
    propagation_method_revision: String,
    expanded_uncertainty_db: f64,
    expanded_uncertainty_confidence_level: f64,
    evidence_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PropagationTransferReceipt {
    from_site_id: String,
    to_site_id: String,
    contract_id: String,
    receipt_sha256: String,
    independently_reviewed: bool,
    per_band_validated: bool,
    validated_band_ids: Vec<String>,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    schema_version: &'static str,
    case_id: String,
    source_dataset_id: String,
    evidence_kind: &'static str,
    claim_ceiling: &'static str,
    campaign_reports: Vec<CampaignReport>,
    site_comparisons: Vec<SiteComparisonReport>,
    disposition: &'static str,
    reasons: Vec<String>,
}

#[derive(Debug, Serialize)]
struct CampaignReport {
    campaign_id: String,
    site_id: String,
    site_revision: String,
    depth_boundary_m: f64,
    mean_water_depth_m: Option<f64>,
    mean_water_depth_lower_m: Option<f64>,
    mean_water_depth_upper_m: Option<f64>,
    water_depth_range_fraction: Option<f64>,
    depth_category: &'static str,
    seabed_substrates: Vec<&'static str>,
    mean_seabed_sound_speed_m_s: Option<f64>,
    seabed_sound_speed_range_fraction: Option<f64>,
    mean_ctd_sound_speed_m_s: Option<f64>,
    ctd_sound_speed_range_fraction: Option<f64>,
    hydrophone_count: usize,
    recommendations_unmet: Vec<String>,
    three_hydrophone_recommendation_met: bool,
    reciprocal_run_count: usize,
    port_run_count: usize,
    starboard_run_count: usize,
    band_reports: Vec<BandReport>,
    disposition: &'static str,
    reasons: Vec<String>,
}

#[derive(Debug, Serialize)]
struct BandReport {
    band_id: String,
    quantity: &'static str,
    center_frequency_hz: f64,
    site_cutoff_frequency_hz: Option<f64>,
    eligible_for_declared_site_campaign: bool,
    exclusion_reason: Option<&'static str>,
    propagation_method_id: String,
    propagation_method_revision: String,
    expanded_uncertainty_db: f64,
}

#[derive(Debug, Serialize)]
struct SiteComparisonReport {
    from_site_id: String,
    to_site_id: String,
    same_depth_category: bool,
    same_substrate_set: bool,
    seabed_sound_speed_difference_fraction: Option<f64>,
    ctd_sound_speed_difference_fraction: Option<f64>,
    same_band_quantity_contract: bool,
    result: &'static str,
    transfer_receipt_sha256: Option<String>,
    reasons: Vec<String>,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}
fn finite_positive(value: f64) -> bool { value.is_finite() && value > 0.0 }
fn finite_nonnegative(value: f64) -> bool { value.is_finite() && value >= 0.0 }
fn digest_errors(label: &str, values: &[(&str, &str)], out: &mut Vec<String>) {
    for (name, value) in values {
        if !is_sha256(value) { out.push(format!("{label}: {name} must be a 64-character SHA-256 hex digest")); }
    }
}
fn mean(values: impl Iterator<Item=f64>) -> Option<f64> {
    let v: Vec<f64> = values.collect();
    if v.is_empty() || v.iter().any(|x| !x.is_finite()) { None } else { Some(v.iter().sum::<f64>() / v.len() as f64) }
}
fn mean_interval(values: &[f64], expanded: &[f64]) -> Option<(f64, f64, f64)> {
    if values.is_empty() || values.len() != expanded.len()
        || values.iter().any(|v| !finite_positive(*v))
        || expanded.iter().any(|u| !finite_positive(*u))
    { return None; }
    let count = values.len() as f64;
    let average = values.iter().sum::<f64>() / count;
    let lower = values.iter().zip(expanded).map(|(v, u)| v - u).sum::<f64>() / count;
    let upper = values.iter().zip(expanded).map(|(v, u)| v + u).sum::<f64>() / count;
    Some((lower, average, upper))
}

fn range_fraction_with_uncertainty(values: &[f64], expanded: &[f64]) -> Option<f64> {
    if values.is_empty() || values.len() != expanded.len()
        || values.iter().any(|v| !finite_positive(*v))
        || expanded.iter().any(|u| !finite_positive(*u))
    { return None; }
    let lower = values.iter().zip(expanded).map(|(v, u)| v - u).fold(f64::INFINITY, f64::min);
    let upper = values.iter().zip(expanded).map(|(v, u)| v + u).fold(f64::NEG_INFINITY, f64::max);
    let average = values.iter().sum::<f64>() / values.len() as f64;
    Some((upper - lower) / average)
}

fn depth_category_from_interval(lower: Option<f64>, upper: Option<f64>, boundary: f64) -> &'static str {
    match (lower, upper) {
        (Some(low), Some(_)) if low > boundary => "deep",
        (Some(_), Some(high)) if high < boundary => "shallow",
        (Some(_), Some(_)) => "boundary_undefined",
        _ => "unknown",
    }
}

fn mean_difference_fraction(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(x), Some(y)) if x.max(y) > 0.0 => Some((x - y).abs() / x.max(y)),
        _ => None,
    }
}
fn depth_boundary_m(length_m: f64) -> f64 {
    DEEP_WATER_MIN_M.max(DEEP_WATER_LENGTH_MULTIPLE * length_m)
}
fn depth_category(mean_depth: Option<f64>, boundary: f64) -> &'static str {
    match mean_depth {
        None => "unknown",
        Some(depth) if (depth - boundary).abs() <= BOUNDARY_EPSILON_M => "boundary_undefined",
        Some(depth) if depth > boundary => "deep",
        Some(_) => "shallow",
    }
}
fn quantity_name(quantity: AcousticQuantity) -> &'static str { quantity.name() }

fn validate_case(case: &Case) -> Vec<String> {
    let mut errors = Vec::new();
    if case.schema_version != SCHEMA_VERSION { errors.push(format!("unsupported schema_version {}; expected {SCHEMA_VERSION}", case.schema_version)); }
    if case.case_id.trim().is_empty() || case.source.dataset_id.trim().is_empty() || case.source.citation.trim().is_empty() {
        errors.push("case_id, source dataset_id and citation are required".to_string());
    }
    digest_errors("case", &[("source raw data", &case.source.raw_data_sha256)], &mut errors);
    for (name, v) in [
        ("maximum_depth_range_fraction", case.maximum_depth_range_fraction),
        ("maximum_seabed_sound_speed_range_fraction", case.maximum_seabed_sound_speed_range_fraction),
        ("maximum_ctd_sound_speed_range_fraction", case.maximum_ctd_sound_speed_range_fraction),
        ("maximum_stw_relative_deviation", case.maximum_stw_relative_deviation),
        ("maximum_cross_site_seabed_sound_speed_difference_fraction", case.maximum_cross_site_seabed_sound_speed_difference_fraction),
    ] {
        if !finite_nonnegative(v) || v >= 1.0 { errors.push(format!("{name} must be finite in [0, 1)")); }
    }
    if case.minimum_runs < 2 { errors.push("minimum_runs must be at least two reciprocal runs".to_string()); }
    if case.campaigns.is_empty() { errors.push("at least one campaign is required".to_string()); }
    let mut ids = BTreeSet::new();
    for campaign in &case.campaigns {
        if campaign.campaign_id.trim().is_empty() || !ids.insert(campaign.campaign_id.clone()) {
            errors.push("campaign IDs must be non-empty and unique".to_string());
        }
        validate_campaign(campaign, case, &mut errors);
    }
    for receipt in &case.propagation_transfer_receipts {
        if receipt.from_site_id.trim().is_empty() || receipt.to_site_id.trim().is_empty() || receipt.contract_id.trim().is_empty() {
            errors.push("propagation transfer receipt requires from/to site and contract IDs".to_string());
        }
        if !is_sha256(&receipt.receipt_sha256) {
            errors.push("propagation transfer receipt digest is malformed".to_string());
        }
        if receipt.validated_band_ids.iter().collect::<BTreeSet<_>>().len() != receipt.validated_band_ids.len() {
            errors.push("propagation transfer receipt has duplicate band IDs".to_string());
        }
    }
    errors
}

fn validate_campaign(c: &Campaign, case: &Case, errors: &mut Vec<String>) {
    if c.site.site_id.trim().is_empty() || c.site.site_revision.trim().is_empty() { errors.push(format!("{}: site identity/revision is required", c.campaign_id)); }
    if c.vessel.vessel_id.trim().is_empty() || c.vessel.acoustic_reference_point_id.trim().is_empty()
        || c.vessel.hull_condition.trim().is_empty() || c.vessel.propeller_condition.trim().is_empty()
        || c.vessel.last_dry_dock_date.trim().is_empty()
    { errors.push(format!("{}: vessel identity, acoustic reference, hull/propeller condition and dry-dock date are required", c.campaign_id)); }
    for (name, v) in [
        ("overall ship length", c.vessel.overall_ship_length_m),
        ("acoustic ship length", c.vessel.acoustic_ship_length_m),
        ("draft", c.vessel.draft_m),
    ] {
        if !finite_positive(v) { errors.push(format!("{}: {name} must be finite and positive", c.campaign_id)); }
    }
    if c.vessel.acoustic_ship_length_m > c.vessel.overall_ship_length_m {
        errors.push(format!("{}: acoustic ship length cannot exceed overall ship length", c.campaign_id));
    }
    if !c.vessel.gps_to_acoustic_centre_offset_m.is_finite() { errors.push(format!("{}: GPS-to-acoustic-centre offset must be finite", c.campaign_id)); }
    digest_errors(&c.campaign_id, &[
        ("propulsion configuration", &c.vessel.propulsion_configuration_sha256),
        ("service fingerprint", &c.vessel.service_fingerprint_sha256),
        ("loading fingerprint", &c.vessel.loading_fingerprint_sha256),
        ("bathymetry source", &c.site.bathymetry_source_sha256),
        ("seabed profile source", &c.site.seabed_profile_source_sha256),
        ("pre-deployment field calibration", &c.in_situ_pre_deployment_calibration_sha256),
        ("post-recovery field calibration", &c.in_situ_post_recovery_calibration_sha256),
    ], errors);
    if c.site.bathymetry_method_id.trim().is_empty() || c.site.seabed_method_id.trim().is_empty() {
        errors.push(format!("{}: bathymetry and seabed characterization methods are required", c.campaign_id));
    }
    if c.site.water_depth_samples.len() < 2 { errors.push(format!("{}: at least two bathymetry samples are required", c.campaign_id)); }
    let mut depth_ids = BTreeSet::new();
    for sample in &c.site.water_depth_samples {
        if sample.sample_id.trim().is_empty() || !depth_ids.insert(sample.sample_id.clone()) { errors.push(format!("{}: bathymetry sample IDs must be unique", c.campaign_id)); }
        if !sample.along_track_position_m.is_finite() || !sample.cross_track_position_m.is_finite()
            || !finite_positive(sample.depth_m) || !finite_positive(sample.expanded_uncertainty_m)
        { errors.push(format!("{}: invalid bathymetry sample {}", c.campaign_id, sample.sample_id)); }
    }
    if c.site.seabed_samples.len() < 2 { errors.push(format!("{}: at least two seabed characterization samples are required", c.campaign_id)); }
    let mut seabed_ids = BTreeSet::new();
    for sample in &c.site.seabed_samples {
        if sample.sample_id.trim().is_empty() || !seabed_ids.insert(sample.sample_id.clone()) { errors.push(format!("{}: seabed sample IDs must be unique", c.campaign_id)); }
        if !sample.along_track_position_m.is_finite() || !sample.cross_track_position_m.is_finite()
            || !finite_positive(sample.sound_speed_m_s) || !finite_positive(sample.expanded_uncertainty_m_s)
        { errors.push(format!("{}: invalid seabed sample {}", c.campaign_id, sample.sample_id)); }
        if !is_sha256(&sample.evidence_sha256) { errors.push(format!("{}: seabed evidence digest malformed", c.campaign_id)); }
    }
    if let Some(ctd) = &c.site.ctd_profile {
        if ctd.instrument_id.trim().is_empty() || ctd.method_id.trim().is_empty() || ctd.samples.len() < 2 {
            errors.push(format!("{}: CTD profile needs instrument/method identity and at least two samples", c.campaign_id));
        }
        digest_errors(&c.campaign_id, &[
            ("CTD calibration", &ctd.calibration_certificate_sha256),
            ("CTD evidence", &ctd.evidence_sha256),
        ], errors);
        for sample in &ctd.samples {
            if !finite_nonnegative(sample.depth_m) || !finite_positive(sample.sound_speed_m_s)
                || !finite_positive(sample.expanded_uncertainty_m_s) || !sample.temperature_c.is_finite()
                || !finite_nonnegative(sample.salinity_psu)
            { errors.push(format!("{}: invalid CTD sample", c.campaign_id)); }
        }
    }
    if let Some(current) = &c.site.current {
        if current.instrument_id.trim().is_empty() || current.method_id.trim().is_empty()
            || !finite_nonnegative(current.speed_m_s) || !current.direction_deg_true.is_finite()
            || !(0.0..360.0).contains(&current.direction_deg_true) || current.observed_at_utc.trim().is_empty()
        { errors.push(format!("{}: invalid current observation", c.campaign_id)); }
        if !is_sha256(&current.evidence_sha256) { errors.push(format!("{}: current evidence digest malformed", c.campaign_id)); }
    }
    if let Some(background) = &c.site.ambient_background {
        if background.measurement_id.trim().is_empty() || background.method_id.trim().is_empty()
            || background.measured_at_utc.trim().is_empty() || !is_sha256(&background.raw_trace_sha256)
        { errors.push(format!("{}: invalid ambient background record", c.campaign_id)); }
    }
    if c.hydrophones.is_empty() { errors.push(format!("{}: at least one hydrophone is required", c.campaign_id)); }
    let mut hydrophone_ids = BTreeSet::new();
    for h in &c.hydrophones {
        if h.instrument_id.trim().is_empty() || !hydrophone_ids.insert(h.instrument_id.clone()) {
            errors.push(format!("{}: hydrophone IDs must be non-empty and unique", c.campaign_id));
        }
        if !h.x_m.is_finite() || !h.y_m.is_finite() || !finite_positive(h.depth_m)
            || !h.orientation_azimuth_deg.is_finite() || !(0.0..360.0).contains(&h.orientation_azimuth_deg)
            || !h.orientation_elevation_deg.is_finite() || !finite_positive(h.calibrated_min_frequency_hz)
            || !finite_positive(h.calibrated_max_frequency_hz) || h.calibrated_min_frequency_hz >= h.calibrated_max_frequency_hz
        { errors.push(format!("{}: invalid hydrophone geometry or calibrated frequency range", c.campaign_id)); }
        if !is_sha256(&h.calibration_certificate_sha256) { errors.push(format!("{}: hydrophone calibration digest malformed", c.campaign_id)); }
    }
    if c.runs.is_empty() { errors.push(format!("{}: at least one run is required", c.campaign_id)); }
    let mut run_ids = BTreeSet::new();
    for run in &c.runs {
        if run.run_id.trim().is_empty() || !run_ids.insert(run.run_id.clone()) { errors.push(format!("{}: run IDs must be unique", c.campaign_id)); }
        digest_errors(&c.campaign_id, &[
            ("run timebase", &run.timebase_receipt_sha256),
            ("run operating point", &run.operating_point_receipt_sha256),
            ("run background trace", &run.background_trace_sha256),
            ("run propulsion configuration", &run.propulsion_configuration_sha256),
            ("run service fingerprint", &run.service_fingerprint_sha256),
            ("run loading fingerprint", &run.loading_fingerprint_sha256),
        ], errors);
        if run.propulsion_configuration_sha256 != c.vessel.propulsion_configuration_sha256
            || run.service_fingerprint_sha256 != c.vessel.service_fingerprint_sha256
            || run.loading_fingerprint_sha256 != c.vessel.loading_fingerprint_sha256
            || (run.draft_m - c.vessel.draft_m).abs() > 1.0e-6
            || run.hull_condition != c.vessel.hull_condition
            || run.propeller_condition != c.vessel.propeller_condition
        { errors.push(format!("{} / {}: run does not match the frozen vessel/service state", c.campaign_id, run.run_id)); }
        if !run.heading_deg_true.is_finite() || !(0.0..360.0).contains(&run.heading_deg_true)
            || !finite_positive(run.shaft_rotation_rate_rpm) || !finite_positive(run.propulsion_power_kw)
            || !finite_positive(run.cpa_distance_m) || !finite_positive(run.data_window_duration_s)
            || run.run_start_utc.trim().is_empty()
        { errors.push(format!("{} / {}: invalid run geometry or propulsion telemetry", c.campaign_id, run.run_id)); }
        if run.speed_through_water_knots.is_some_and(|v| !finite_positive(v))
            || run.speed_over_ground_knots.is_some_and(|v| !finite_positive(v))
        { errors.push(format!("{} / {}: invalid speed telemetry", c.campaign_id, run.run_id)); }
    }
    if c.bands.is_empty() { errors.push(format!("{}: at least one acoustic band is required", c.campaign_id)); }
    let mut band_ids = BTreeSet::new();
    for band in &c.bands {
        if band.band_id.trim().is_empty() || !band_ids.insert(band.band_id.clone()) { errors.push(format!("{}: band IDs must be unique", c.campaign_id)); }
        if !finite_positive(band.center_frequency_hz)
            || band.site_cutoff_frequency_hz.is_some_and(|v| !finite_positive(v))
            || band.propagation_method_id.trim().is_empty() || band.propagation_method_revision.trim().is_empty()
            || !finite_positive(band.expanded_uncertainty_db)
            || !band.expanded_uncertainty_confidence_level.is_finite()
            || band.expanded_uncertainty_confidence_level <= 0.0
            || band.expanded_uncertainty_confidence_level >= 1.0
            || !is_sha256(&band.evidence_sha256)
        { errors.push(format!("{} / {}: invalid band/propagation evidence", c.campaign_id, band.band_id)); }
    }
}

fn campaign_report(c: &Campaign, case: &Case) -> CampaignReport {
    let mut reasons = Vec::new();
    let depths: Vec<f64> = c.site.water_depth_samples.iter().map(|s| s.depth_m).collect();
    let depth_uncertainties: Vec<f64> = c.site.water_depth_samples.iter().map(|s| s.expanded_uncertainty_m).collect();
    let depth_interval = mean_interval(&depths, &depth_uncertainties);
    let (depth_lower, mean_depth, depth_upper) = depth_interval
        .map(|(lower, average, upper)| (Some(lower), Some(average), Some(upper)))
        .unwrap_or((None, None, None));
    let depth_range = range_fraction_with_uncertainty(&depths, &depth_uncertainties);
    let boundary = depth_boundary_m(c.vessel.overall_ship_length_m);
    let category = depth_category_from_interval(depth_lower, depth_upper, boundary);
    if category == "boundary_undefined" { reasons.push("mean site depth lies exactly on the ISO deep/shallow classification boundary".to_string()); }
    if depth_range.is_none_or(|v| v > case.maximum_depth_range_fraction) {
        reasons.push("bathymetry coverage is missing or water-depth variation exceeds the declared site tolerance".to_string());
    }
    let substrates: BTreeSet<Substrate> = c.site.seabed_samples.iter().map(|s| s.substrate).collect();
    let seabed_speeds: Vec<f64> = c.site.seabed_samples.iter().map(|s| s.sound_speed_m_s).collect();
    let seabed_uncertainties: Vec<f64> = c.site.seabed_samples.iter().map(|s| s.expanded_uncertainty_m_s).collect();
    let mean_seabed_speed = mean(seabed_speeds.iter().copied());
    let seabed_range = range_fraction_with_uncertainty(&seabed_speeds, &seabed_uncertainties);
    if seabed_range.is_none_or(|v| v > case.maximum_seabed_sound_speed_range_fraction) {
        reasons.push("seabed sound-speed coverage is missing or spatial variation exceeds the declared tolerance".to_string());
    }
    if substrates.contains(&Substrate::Unknown) { reasons.push("one or more seabed samples have unknown substrate classification".to_string()); }
    let ctd_speeds: Vec<f64> = c.site.ctd_profile.as_ref().map(|p| p.samples.iter().map(|s| s.sound_speed_m_s).collect()).unwrap_or_default();
    let ctd_uncertainties: Vec<f64> = c.site.ctd_profile.as_ref().map(|p| p.samples.iter().map(|s| s.expanded_uncertainty_m_s).collect()).unwrap_or_default();
    let mean_ctd_speed = mean(ctd_speeds.iter().copied());
    let ctd_range = range_fraction_with_uncertainty(&ctd_speeds, &ctd_uncertainties);
    if ctd_range.is_none() || ctd_range.is_some_and(|v| v > case.maximum_ctd_sound_speed_range_fraction) {
        reasons.push("CTD/sound-speed profile is missing or exceeds the declared within-profile uniformity tolerance".to_string());
    }
    if c.site.current.is_none() { reasons.push("observed current direction and speed are missing".to_string()); }
    if c.site.ambient_background.is_none() { reasons.push("ambient background measurement record is missing".to_string()); }
    if c.hydrophones.iter().any(|h| !is_sha256(&h.calibration_certificate_sha256)) {
        reasons.push("one or more hydrophone calibration records are invalid".to_string());
    }
    if !is_sha256(&c.in_situ_pre_deployment_calibration_sha256) || !is_sha256(&c.in_situ_post_recovery_calibration_sha256) {
        reasons.push("pre-deployment and post-recovery in-situ calibration checks are both required".to_string());
    }
    let mut recommendations_unmet = Vec::new();
    if c.hydrophones.len() < 3 {
        recommendations_unmet.push("IACS recommends three hydrophones for reduced uncertainty and channel redundancy".to_string());
    }
    let mut port = 0usize;
    let mut starboard = 0usize;
    let mut stw_values = Vec::new();
    for run in &c.runs {
        match run.aspect { RunAspect::Port => port += 1, RunAspect::Starboard => starboard += 1 }
        if let Some(stw) = run.speed_through_water_knots { stw_values.push(stw); }
        else { reasons.push(format!("run {} lacks speed-through-water evidence; speed over ground is not a substitute", run.run_id)); }
        if run.data_window_duration_s < IACS_MIN_DATA_WINDOW_S {
            recommendations_unmet.push(format!("run {} data window is shorter than the IACS 10 s recommendation", run.run_id));
        }
        if run.cpa_distance_m + 1.0e-9 < IACS_MIN_CPA_M.max(c.vessel.acoustic_ship_length_m) {
            recommendations_unmet.push(format!("run {} CPA distance is below the IACS recommended greater of 100 m or acoustic ship length", run.run_id));
        }
    }
    if port == 0 || starboard == 0 { reasons.push("both port and starboard reciprocal aspects are required for this campaign contract".to_string()); }
    if c.runs.len() < case.minimum_runs { reasons.push("campaign has fewer than the declared minimum number of runs".to_string()); }
    if let Some(median) = mean(stw_values.iter().copied()) {
        if stw_values.len() != c.runs.len() || stw_values.iter().any(|v| (v - median).abs() / median > case.maximum_stw_relative_deviation) {
            reasons.push("speed-through-water is missing or varies beyond the declared matched-run tolerance".to_string());
        }
    }
    for run in &c.runs {
        if run.propulsion_configuration_sha256 != c.vessel.propulsion_configuration_sha256
            || run.service_fingerprint_sha256 != c.vessel.service_fingerprint_sha256
            || run.loading_fingerprint_sha256 != c.vessel.loading_fingerprint_sha256
        { reasons.push(format!("run {} is not in the declared matched operating/service stratum", run.run_id)); }
    }
    if c.site.ctd_profile.is_none() { reasons.push("CTD/sound-speed profile not supplied".to_string()); }
    if c.site.current.is_none() { reasons.push("current observation not supplied".to_string()); }

    let band_reports: Vec<BandReport> = c.bands.iter().map(|band| {
        let hydrophone_covers = c.hydrophones.iter().all(|h| {
            band.center_frequency_hz >= h.calibrated_min_frequency_hz
                && band.center_frequency_hz <= h.calibrated_max_frequency_hz
        });
        let cutoff = band.site_cutoff_frequency_hz;
        let below_cutoff = cutoff.is_some_and(|v| band.center_frequency_hz < v);
        let exclusion = if cutoff.is_none() { Some("unknown_site_cutoff") }
            else if below_cutoff { Some("below_declared_site_cutoff") }
            else if !hydrophone_covers { Some("outside_calibrated_hydrophone_frequency_range") }
            else { None };
        BandReport {
            band_id: band.band_id.clone(),
            quantity: quantity_name(band.quantity),
            center_frequency_hz: band.center_frequency_hz,
            site_cutoff_frequency_hz: cutoff,
            eligible_for_declared_site_campaign: exclusion.is_none() && c.site.ctd_profile.is_some(),
            exclusion_reason: exclusion,
            propagation_method_id: band.propagation_method_id.clone(),
            propagation_method_revision: band.propagation_method_revision.clone(),
            expanded_uncertainty_db: band.expanded_uncertainty_db,
        }
    }).collect();

    if c.bands.iter().any(|band| band.site_cutoff_frequency_hz.is_none()) {
        reasons.push("one or more bands have an unknown site cutoff frequency".to_string());
    }
    if band_reports.iter().all(|b| !b.eligible_for_declared_site_campaign) {
        reasons.push("no acoustic band is eligible under the declared site cutoff and sensor response".to_string());
    }

    let three_hydrophones = c.hydrophones.len() >= 3;
    let hard_incomplete = category == "boundary_undefined" || category == "unknown"
        || reasons.iter().any(|r| {
            r.contains("missing") || r.contains("exceeds") || r.contains("unknown")
                || r.contains("below") || r.contains("shorter") || r.contains("lacks")
                || r.contains("fewer") || r.contains("both port") || r.contains("not supplied")
                || r.contains("not a substitute") || r.contains("varies beyond")
                || r.contains("classification boundary") || r.contains("no acoustic band is eligible")
        });
    let invalid = !validate_campaign_errors_empty(c, case);
    let disposition = if invalid { "INVALID" }
        else if hard_incomplete { "INCOMPLETE" }
        else if !recommendations_unmet.is_empty() { "CONDITIONAL" }
        else { "SITE_ELIGIBLE_FOR_DECLARED_MEASUREMENT" };

    CampaignReport {
        campaign_id: c.campaign_id.clone(),
        site_id: c.site.site_id.clone(),
        site_revision: c.site.site_revision.clone(),
        depth_boundary_m: boundary,
        mean_water_depth_m: mean_depth,
        mean_water_depth_lower_m: depth_lower,
        mean_water_depth_upper_m: depth_upper,
        water_depth_range_fraction: depth_range,
        depth_category: category,
        seabed_substrates: substrates.into_iter().map(Substrate::name).collect(),
        mean_seabed_sound_speed_m_s: mean_seabed_speed,
        seabed_sound_speed_range_fraction: seabed_range,
        mean_ctd_sound_speed_m_s: mean_ctd_speed,
        ctd_sound_speed_range_fraction: ctd_range,
        hydrophone_count: c.hydrophones.len(),
        recommendations_unmet,
        three_hydrophone_recommendation_met: three_hydrophones,
        reciprocal_run_count: port + starboard,
        port_run_count: port,
        starboard_run_count: starboard,
        band_reports,
        disposition,
        reasons,
    }
}

fn validate_campaign_errors_empty(c: &Campaign, case: &Case) -> bool {
    let mut errors = Vec::new();
    validate_campaign(c, case, &mut errors);
    errors.is_empty()
}

fn site_comparisons(case: &Case, reports: &[CampaignReport]) -> Vec<SiteComparisonReport> {
    let mut out = Vec::new();
    for i in 0..case.campaigns.len() {
        for j in (i + 1)..case.campaigns.len() {
            let a = &case.campaigns[i];
            let b = &case.campaigns[j];
            let ra = &reports[i];
            let rb = &reports[j];
            let mut reasons = Vec::new();
            if a.vessel.vessel_id != b.vessel.vessel_id
                || a.vessel.service_fingerprint_sha256 != b.vessel.service_fingerprint_sha256
                || a.vessel.propulsion_configuration_sha256 != b.vessel.propulsion_configuration_sha256
                || a.vessel.loading_fingerprint_sha256 != b.vessel.loading_fingerprint_sha256
                || (a.vessel.draft_m - b.vessel.draft_m).abs() > 1.0e-6
                || a.vessel.hull_condition != b.vessel.hull_condition
                || a.vessel.propeller_condition != b.vessel.propeller_condition
            {
                reasons.push("cross-site campaigns do not share an exact vessel/service/propulsion/loading/hull-state identity".to_string());
            }
            let same_depth = ra.depth_category == rb.depth_category && ra.depth_category != "boundary_undefined";
            let same_substrate = ra.seabed_substrates == rb.seabed_substrates;
            let speed_delta = mean_difference_fraction(ra.mean_seabed_sound_speed_m_s, rb.mean_seabed_sound_speed_m_s);
            let ctd_delta = mean_difference_fraction(ra.mean_ctd_sound_speed_m_s, rb.mean_ctd_sound_speed_m_s);
            let a_band_map: BTreeMap<&str, &Band> = a.bands.iter().map(|band| (band.band_id.as_str(), band)).collect();
            let b_band_map: BTreeMap<&str, &Band> = b.bands.iter().map(|band| (band.band_id.as_str(), band)).collect();
            let same_band_ids = a_band_map.keys().copied().collect::<BTreeSet<_>>() == b_band_map.keys().copied().collect::<BTreeSet<_>>();
            let same_band_quantity_contract = same_band_ids && a_band_map.iter().all(|(id, band_a)| {
                b_band_map.get(id).is_some_and(|band_b| {
                    band_a.quantity == band_b.quantity
                        && (band_a.center_frequency_hz - band_b.center_frequency_hz).abs() <= 1.0e-6
                        && band_a.propagation_method_id == band_b.propagation_method_id
                        && band_a.propagation_method_revision == band_b.propagation_method_revision
                        && (band_a.expanded_uncertainty_confidence_level - band_b.expanded_uncertainty_confidence_level).abs() <= 1.0e-9
                })
            });
            if !same_band_quantity_contract {
                reasons.push("cross-site campaigns do not have identical band IDs, frequencies and acoustic quantity semantics".to_string());
            }
            let environment_differs = !same_depth || !same_substrate
                || speed_delta.is_none_or(|v| v > case.maximum_cross_site_seabed_sound_speed_difference_fraction)
                || ctd_delta.is_none_or(|v| v > case.maximum_cross_site_seabed_sound_speed_difference_fraction);
            if environment_differs {
                reasons.push("site depth category, seabed substrate or seabed sound-speed class differs".to_string());
            }
            let mut transfer_hash = None;
            let all_band_ids: BTreeSet<String> = a.bands.iter().map(|b| b.band_id.clone()).collect();
            let transfer = case.propagation_transfer_receipts.iter().find(|receipt| {
                receipt.from_site_id == a.site.site_id && receipt.to_site_id == b.site.site_id
                    && receipt.independently_reviewed && receipt.per_band_validated
                    && is_sha256(&receipt.receipt_sha256)
                    && same_band_quantity_contract
                    && all_band_ids.iter().all(|id| receipt.validated_band_ids.contains(id))
            });
            let result = if !reasons.iter().any(|r| r.contains("vessel/service"))
                && !environment_differs && same_band_quantity_contract {
                "SAME_DECLARED_PROPAGATION_STRATUM"
            } else if transfer.is_some() && !reasons.iter().any(|r| r.contains("vessel/service")) {
                transfer_hash = transfer.map(|r| r.receipt_sha256.clone());
                reasons.push("propagation-transfer receipt is referenced but its physical validity is not independently recalculated by this audit".to_string());
                "TRANSFER_RECEIPT_PRESENT_NOT_VALIDATED_HERE"
            } else {
                "INCOMPARABLE"
            };
            out.push(SiteComparisonReport {
                from_site_id: a.site.site_id.clone(),
                to_site_id: b.site.site_id.clone(),
                same_depth_category: same_depth,
                same_substrate_set: same_substrate,
                seabed_sound_speed_difference_fraction: speed_delta,
                ctd_sound_speed_difference_fraction: ctd_delta,
                same_band_quantity_contract,
                result,
                transfer_receipt_sha256: transfer_hash,
                reasons,
            });
        }
    }
    out
}

fn audit(case: &Case) -> AuditReport {
    let validation_errors = validate_case(case);
    let mut reasons = validation_errors.clone();
    let campaign_reports: Vec<CampaignReport> = if validation_errors.is_empty() {
        case.campaigns.iter().map(|c| campaign_report(c, case)).collect()
    } else {
        Vec::new()
    };
    let comparisons = if validation_errors.is_empty() { site_comparisons(case, &campaign_reports) } else { Vec::new() };
    if case.evidence_kind == EvidenceKind::Synthetic { reasons.push("synthetic evidence cannot qualify a physical measurement site".to_string()); }
    if case.source.rights_status != RightsStatus::PublicOrPermitted { reasons.push("source rights are not established as public_or_permitted".to_string()); }
    if !case.source.independently_reviewed { reasons.push("source metadata has not been independently reviewed".to_string()); }
    if campaign_reports.iter().any(|r| r.disposition == "INCOMPLETE") { reasons.push("one or more campaigns are incomplete".to_string()); }
    if campaign_reports.iter().any(|r| r.disposition == "INVALID") { reasons.push("one or more campaigns are invalid".to_string()); }
    if comparisons.iter().any(|r| r.result == "INCOMPARABLE") { reasons.push("one or more cross-site comparisons are incomparable".to_string()); }

    let disposition = if !validation_errors.is_empty() || campaign_reports.iter().any(|r| r.disposition == "INVALID") {
        "INVALID"
    } else if case.evidence_kind == EvidenceKind::Synthetic
        || case.source.rights_status != RightsStatus::PublicOrPermitted
        || !case.source.independently_reviewed
        || campaign_reports.iter().any(|r| r.disposition == "INCOMPLETE")
    {
        "INCOMPLETE"
    } else if campaign_reports.iter().any(|r| r.disposition == "CONDITIONAL") {
        "CONDITIONAL"
    } else {
        "SITE_ELIGIBLE_FOR_DECLARED_MEASUREMENT"
    };

    AuditReport {
        schema_version: SCHEMA_VERSION,
        case_id: case.case_id.clone(),
        source_dataset_id: case.source.dataset_id.clone(),
        evidence_kind: match case.evidence_kind { EvidenceKind::Synthetic => "synthetic", EvidenceKind::Measured => "measured" },
        claim_ceiling: "site_and_campaign_evidence_admission_only_no_iso_conformity_source_level_solver_propagation_correction_ecological_or_regulatory_qualification",
        campaign_reports,
        site_comparisons: comparisons,
        disposition,
        reasons,
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else { eprintln!("usage: marine-shallow-water-site-audit <case.json>"); process::exit(64); };
    if args.next().is_some() { eprintln!("usage: marine-shallow-water-site-audit <case.json>"); process::exit(64); }
    let input = match fs::read_to_string(&path) { Ok(v) => v, Err(e) => { eprintln!("could not read {path}: {e}"); process::exit(64); } };
    let case: Case = match serde_json::from_str(&input) { Ok(v) => v, Err(e) => { eprintln!("invalid case JSON: {e}"); process::exit(64); } };
    let report = audit(&case);
    println!("{}", serde_json::to_string_pretty(&report).expect("report serializes"));
    match report.disposition { "SITE_ELIGIBLE_FOR_DECLARED_MEASUREMENT" | "CONDITIONAL" => {}, "INVALID" => process::exit(2), _ => process::exit(3) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(c: char) -> String { std::iter::repeat_n(c, 64).collect() }

    fn sample_campaign(id: &str, site_id: &str, substrate: Substrate, seabed_speed: f64, hydrophones: usize) -> Campaign {
        let hydrophone_list = (0..hydrophones).map(|i| Hydrophone {
            instrument_id: format!("hydrophone-{i}"),
            x_m: i as f64 * 5.0,
            y_m: 0.0,
            depth_m: 18.0,
            orientation_azimuth_deg: 180.0,
            orientation_elevation_deg: 0.0,
            calibration_certificate_sha256: digest('a'),
            calibrated_min_frequency_hz: 10.0,
            calibrated_max_frequency_hz: 20_000.0,
        }).collect();
        let runs = (0..4).map(|i| Run {
            run_id: format!("{id}-run-{i}"),
            aspect: if i % 2 == 0 { RunAspect::Port } else { RunAspect::Starboard },
            timebase_receipt_sha256: digest('b'),
            operating_point_receipt_sha256: digest('c'),
            background_trace_sha256: digest('d'),
            propulsion_configuration_sha256: digest('e'),
            service_fingerprint_sha256: digest('f'),
            loading_fingerprint_sha256: digest('1'),
            draft_m: 4.0,
            hull_condition: "clean".to_string(),
            propeller_condition: "polished".to_string(),
            speed_through_water_knots: Some(10.0 + i as f64 * 0.05),
            speed_over_ground_knots: Some(10.1 + i as f64 * 0.05),
            shaft_rotation_rate_rpm: 120.0,
            propulsion_power_kw: 800.0,
            heading_deg_true: if i % 2 == 0 { 90.0 } else { 270.0 },
            cpa_distance_m: 120.0,
            data_window_duration_s: 20.0,
            run_start_utc: format!("2026-10-01T10:{i:02}:00Z"),
        }).collect();
        Campaign {
            campaign_id: id.to_string(),
            site: Site {
                site_id: site_id.to_string(),
                site_revision: "v1".to_string(),
                bathymetry_source_sha256: digest('2'),
                bathymetry_method_id: "multibeam".to_string(),
                water_depth_samples: vec![
                    WaterDepthSample { sample_id:"d1".to_string(), along_track_position_m:0.0,cross_track_position_m:0.0,depth_m:30.0,expanded_uncertainty_m:0.2 },
                    WaterDepthSample { sample_id:"d2".to_string(), along_track_position_m:100.0,cross_track_position_m:10.0,depth_m:31.0,expanded_uncertainty_m:0.2 },
                ],
                seabed_profile_source_sha256: digest('3'),
                seabed_method_id: "geophysical-and-core".to_string(),
                seabed_samples: vec![
                    SeabedSample { sample_id:"s1".to_string(),along_track_position_m:0.0,cross_track_position_m:0.0,substrate,sound_speed_m_s:seabed_speed,expanded_uncertainty_m_s:10.0,evidence_sha256:digest('4') },
                    SeabedSample { sample_id:"s2".to_string(),along_track_position_m:100.0,cross_track_position_m:10.0,substrate,sound_speed_m_s:seabed_speed+5.0,expanded_uncertainty_m_s:10.0,evidence_sha256:digest('5') },
                ],
                ctd_profile: Some(CtdProfile {
                    instrument_id:"ctd-1".to_string(),method_id:"sound-velocity-profile".to_string(),
                    calibration_certificate_sha256:digest('6'),evidence_sha256:digest('7'),
                    samples:vec![
                        CtdSample{depth_m:0.0,sound_speed_m_s:1500.0,expanded_uncertainty_m_s:0.5,temperature_c:18.0,salinity_psu:35.0},
                        CtdSample{depth_m:20.0,sound_speed_m_s:1502.0,expanded_uncertainty_m_s:0.5,temperature_c:17.0,salinity_psu:35.1},
                    ]
                }),
                current:Some(CurrentObservation{instrument_id:"adcp-1".to_string(),method_id:"adcp".to_string(),evidence_sha256:digest('8'),speed_m_s:0.5,direction_deg_true:180.0,observed_at_utc:"2026-10-01T09:00:00Z".to_string()}),
                ambient_background:Some(AmbientBackground{measurement_id:"bg-1".to_string(),method_id:"pre-run-hydrophone".to_string(),raw_trace_sha256:digest('9'),measured_at_utc:"2026-10-01T09:00:00Z".to_string()}),
            },
            vessel:Vessel{
                vessel_id:"vessel-1".to_string(),overall_ship_length_m:80.0,acoustic_ship_length_m:25.0,
                acoustic_reference_point_id:"acoustic-centre".to_string(),gps_to_acoustic_centre_offset_m:3.0,
                propulsion_configuration_sha256:digest('e'),service_fingerprint_sha256:digest('f'),loading_fingerprint_sha256:digest('1'),
                draft_m:4.0,hull_condition:"clean".to_string(),propeller_condition:"polished".to_string(),last_dry_dock_date:"2026-06-01".to_string()
            },
            hydrophones:hydrophone_list,
            in_situ_pre_deployment_calibration_sha256:digest('a'),
            in_situ_post_recovery_calibration_sha256:digest('b'),
            runs,
            bands:vec![
                Band{band_id:"rnl-100hz".to_string(),quantity:AcousticQuantity::RadiatedNoiseLevel,center_frequency_hz:100.0,site_cutoff_frequency_hz:Some(20.0),propagation_method_id:"iso-17208-3".to_string(),propagation_method_revision:"2025".to_string(),expanded_uncertainty_db:2.2,expanded_uncertainty_confidence_level:0.95,evidence_sha256:digest('c')},
                Band{band_id:"sl-1000hz".to_string(),quantity:AcousticQuantity::SourceLevel,center_frequency_hz:1000.0,site_cutoff_frequency_hz:Some(20.0),propagation_method_id:"iso-17208-3".to_string(),propagation_method_revision:"2025".to_string(),expanded_uncertainty_db:2.2,expanded_uncertainty_confidence_level:0.95,evidence_sha256:digest('d')},
            ],
        }
    }

    fn case() -> Case {
        Case{
            schema_version:SCHEMA_VERSION.to_string(),case_id:"synthetic-site-001".to_string(),evidence_kind:EvidenceKind::Synthetic,
            source:SourceEvidence{dataset_id:"synthetic-site".to_string(),citation:"synthetic known answer".to_string(),raw_data_sha256:digest('a'),rights_status:RightsStatus::PublicOrPermitted,independently_reviewed:true},
            minimum_runs:4,maximum_depth_range_fraction:0.10,maximum_seabed_sound_speed_range_fraction:0.10,
            maximum_ctd_sound_speed_range_fraction:0.10,maximum_stw_relative_deviation:0.03,
            maximum_cross_site_seabed_sound_speed_difference_fraction:0.08,
            campaigns:vec![sample_campaign("campaign-1","site-sand",Substrate::Sand,1650.0,3)],
            propagation_transfer_receipts:vec![],
        }
    }

    fn measured(input:&mut Case){input.evidence_kind=EvidenceKind::Measured;}

    #[test] fn synthetic_fixture_never_qualifies_physical_site(){let r=audit(&case());assert_eq!(r.disposition,"INCOMPLETE");assert_eq!(r.campaign_reports[0].depth_category,"shallow");}
    #[test] fn deep_shallow_boundary_uses_max_150m_or_1p5_length(){assert_eq!(depth_boundary_m(100.0),150.0);assert_eq!(depth_boundary_m(200.0),300.0);assert_eq!(depth_category(Some(250.0),300.0),"shallow");assert_eq!(depth_category(Some(301.0),300.0),"deep");}
    #[test] fn exact_deep_shallow_boundary_is_not_classified(){assert_eq!(depth_category(Some(150.0),150.0),"boundary_undefined");}
    #[test]
    fn exact_depth_classification_boundary_blocks_site_eligibility() {
        let mut input = case();
        measured(&mut input);
        for sample in &mut input.campaigns[0].site.water_depth_samples {
            sample.depth_m = 150.0;
        }
        let report = audit(&input);
        assert_eq!(report.campaign_reports[0].depth_category, "boundary_undefined");
        assert_eq!(report.disposition, "INCOMPLETE");
    }
    #[test] fn complete_three_hydrophone_campaign_is_eligible_when_measured(){let mut x=case();measured(&mut x);assert_eq!(audit(&x).disposition,"SITE_ELIGIBLE_FOR_DECLARED_MEASUREMENT");}
    #[test] fn single_hydrophone_is_conditional_not_invalid(){let mut x=case();measured(&mut x);x.campaigns[0].hydrophones.truncate(1);assert_eq!(audit(&x).disposition,"CONDITIONAL");}
    #[test] fn missing_ctd_is_incomplete(){let mut x=case();measured(&mut x);x.campaigns[0].site.ctd_profile=None;assert_eq!(audit(&x).disposition,"INCOMPLETE");}
    #[test] fn missing_current_is_incomplete(){let mut x=case();measured(&mut x);x.campaigns[0].site.current=None;assert_eq!(audit(&x).disposition,"INCOMPLETE");}
    #[test] fn nonuniform_bathymetry_exceeds_site_tolerance(){let mut x=case();measured(&mut x);x.campaigns[0].site.water_depth_samples[1].depth_m=60.0;assert_eq!(audit(&x).disposition,"INCOMPLETE");}
    #[test] fn nonuniform_seabed_sound_speed_exceeds_tolerance(){let mut x=case();measured(&mut x);x.campaigns[0].site.seabed_samples[1].sound_speed_m_s=2100.0;assert_eq!(audit(&x).disposition,"INCOMPLETE");}
    #[test] fn missing_post_recovery_field_calibration_is_incomplete(){let mut x=case();measured(&mut x);x.campaigns[0].in_situ_post_recovery_calibration_sha256="missing".to_string();assert_eq!(audit(&x).disposition,"INVALID");}
    #[test] fn speed_over_ground_is_not_a_substitute_for_stw(){let mut x=case();measured(&mut x);x.campaigns[0].runs[0].speed_through_water_knots=None;assert_eq!(audit(&x).disposition,"INCOMPLETE");}
    #[test] fn short_data_window_is_conditional_advisory(){let mut x=case();measured(&mut x);x.campaigns[0].runs[0].data_window_duration_s=8.0;assert_eq!(audit(&x).disposition,"CONDITIONAL");}
    #[test] fn cpa_below_iacs_recommendation_is_conditional_advisory(){let mut x=case();measured(&mut x);x.campaigns[0].runs[0].cpa_distance_m=50.0;assert_eq!(audit(&x).disposition,"CONDITIONAL");}
    #[test] fn unknown_site_cutoff_is_incomplete(){let mut x=case();measured(&mut x);x.campaigns[0].bands[0].site_cutoff_frequency_hz=None;assert_eq!(audit(&x).disposition,"INCOMPLETE");}
    #[test] fn both_port_and_starboard_runs_are_required(){let mut x=case();measured(&mut x);for r in &mut x.campaigns[0].runs{r.aspect=RunAspect::Port;}assert_eq!(audit(&x).disposition,"INCOMPLETE");}
    #[test] fn quantity_ids_remain_separate_in_output(){let r=audit(&case());let b=&r.campaign_reports[0].band_reports;assert_eq!(b[0].quantity,"radiated_noise_level");assert_eq!(b[1].quantity,"source_level");}
    #[test] fn bands_below_site_cutoff_are_excluded(){let mut x=case();x.campaigns[0].bands[0].center_frequency_hz=10.0;let r=audit(&x);assert_eq!(r.campaign_reports[0].band_reports[0].eligible_for_declared_site_campaign,false);assert_eq!(r.campaign_reports[0].band_reports[0].exclusion_reason,Some("below_declared_site_cutoff"));}
    #[test] fn cross_site_substrate_difference_is_incomparable_without_transfer(){let mut x=case();measured(&mut x);let mut b=x.campaigns[0].clone();b.campaign_id="campaign-2".to_string();b.site.site_id="site-rock".to_string();for s in &mut b.site.seabed_samples{s.substrate=Substrate::Rock;s.sound_speed_m_s=2200.0;}x.campaigns.push(b);let r=audit(&x);assert_eq!(r.site_comparisons[0].result,"INCOMPARABLE");}
    #[test] fn reviewed_per_band_transfer_receipt_is_preserved_not_recalculated(){let mut x=case();measured(&mut x);let mut b=x.campaigns[0].clone();b.campaign_id="campaign-2".to_string();b.site.site_id="site-rock".to_string();for s in &mut b.site.seabed_samples{s.substrate=Substrate::Rock;s.sound_speed_m_s=2200.0;}x.campaigns.push(b);x.propagation_transfer_receipts.push(PropagationTransferReceipt{from_site_id:"site-sand".to_string(),to_site_id:"site-rock".to_string(),contract_id:"urn-prop-transfer-v1".to_string(),receipt_sha256:digest('e'),independently_reviewed:true,per_band_validated:true,validated_band_ids:vec!["rnl-100hz".to_string(),"sl-1000hz".to_string()]});let r=audit(&x);assert_eq!(r.site_comparisons[0].result,"TRANSFER_RECEIPT_PRESENT_NOT_VALIDATED_HERE");}
    #[test] fn cross_site_vessel_or_service_mismatch_stays_incomparable_even_with_transfer(){let mut x=case();measured(&mut x);let mut b=x.campaigns[0].clone();b.campaign_id="campaign-2".to_string();b.site.site_id="site-rock".to_string();b.vessel.vessel_id="other-vessel".to_string();x.campaigns.push(b);x.propagation_transfer_receipts.push(PropagationTransferReceipt{from_site_id:"site-sand".to_string(),to_site_id:"site-rock".to_string(),contract_id:"urn-prop-transfer-v1".to_string(),receipt_sha256:digest('e'),independently_reviewed:true,per_band_validated:true,validated_band_ids:vec!["rnl-100hz".to_string(),"sl-1000hz".to_string()]});assert_eq!(audit(&x).site_comparisons[0].result,"INCOMPARABLE");}
    #[test]
    fn depth_uncertainty_straddling_deep_shallow_boundary_is_incomplete() {
        let mut input = case();
        measured(&mut input);
        for sample in &mut input.campaigns[0].site.water_depth_samples {
            sample.depth_m = 151.0;
            sample.expanded_uncertainty_m = 2.0;
        }
        let report = audit(&input);
        assert_eq!(report.campaign_reports[0].depth_category, "boundary_undefined");
        assert_eq!(report.disposition, "INCOMPLETE");
    }

    #[test]
    fn cross_site_method_revision_mismatch_blocks_comparison() {
        let mut input = case();
        measured(&mut input);
        let mut second = input.campaigns[0].clone();
        second.campaign_id = "campaign-2".to_string();
        second.site.site_id = "site-2".to_string();
        second.bands[0].propagation_method_revision = "other-revision".to_string();
        input.campaigns.push(second);
        assert_eq!(audit(&input).site_comparisons[0].same_band_quantity_contract, false);
        assert_eq!(audit(&input).site_comparisons[0].result, "INCOMPARABLE");
    }

    #[test]
    fn all_bands_below_cutoff_prevent_site_eligibility() {
        let mut input = case();
        measured(&mut input);
        for band in &mut input.campaigns[0].bands {
            band.site_cutoff_frequency_hz = Some(20_000.0);
        }
        assert_eq!(audit(&input).disposition, "INCOMPLETE");
    }

    #[test]
    fn cross_site_quantity_mismatch_blocks_comparison() {
        let mut input = case();
        measured(&mut input);
        let mut second = input.campaigns[0].clone();
        second.campaign_id = "campaign-2".to_string();
        second.site.site_id = "site-2".to_string();
        second.bands[0].quantity = AcousticQuantity::SourceLevel;
        input.campaigns.push(second);
        assert_eq!(audit(&input).site_comparisons[0].same_band_quantity_contract, false);
        assert_eq!(audit(&input).site_comparisons[0].result, "INCOMPARABLE");
    }

    #[test] fn invalid_digest_is_rejected(){let mut x=case();x.campaigns[0].site.bathymetry_source_sha256="bad".to_string();assert_eq!(audit(&x).disposition,"INVALID");}
    #[test] fn duplicate_run_ids_are_invalid(){let mut x=case();x.campaigns[0].runs[1].run_id=x.campaigns[0].runs[0].run_id.clone();assert_eq!(audit(&x).disposition,"INVALID");}
    #[test] fn no_blind_free_field_correction_is_emitted(){let r=audit(&case());assert!(r.claim_ceiling.contains("no_iso_conformity_source_level_solver_propagation_correction"));}
    #[test] fn stw_mismatch_between_reciprocal_runs_is_incomplete(){let mut x=case();measured(&mut x);x.campaigns[0].runs[3].speed_through_water_knots=Some(14.0);assert_eq!(audit(&x).disposition,"INCOMPLETE");}
}
