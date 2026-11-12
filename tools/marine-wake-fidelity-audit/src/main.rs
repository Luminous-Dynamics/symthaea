use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::{env, fs, process};

const SCHEMA_VERSION: &str = "0.1.0";
const GRID_TOLERANCE: f64 = 1.0e-10;
const WEIGHT_SUM_TOLERANCE: f64 = 1.0e-6;
const ANGLE_TOLERANCE_DEG: f64 = 1.0e-6;
const ITTC_BLOCKAGE_RECOMMENDATION: f64 = 0.20;
const ZERO_AMPLITUDE_EPSILON: f64 = 1.0e-10;

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum EvidenceKind { Synthetic, Measured }

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RightsStatus { PublicOrPermitted, Unknown, Restricted }

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum VelocityMethod { FiveHolePitot, OtherPitot, Rans, Cfd, WireScreenDerived, OtherMeasured }

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum WakeBasis { ModelScale, FullScale }

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum CoordinateFrame { PropellerLookingForwardFromAft }

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum AngularOrigin { TopDeadCentre }

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PositiveAzimuth { ClockwiseViewedFromAft, CounterClockwiseViewedFromAft }

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum AxialPositiveDirection { TowardBow, TowardStern }

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum BlockageMethod { WireScreen, DummyHull, FullHull, OtherDeclared }

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum BlockageReviewState { NotRequired, Reviewed, Unknown }

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    schema_version: String,
    case_id: String,
    evidence_kind: EvidenceKind,
    source: SourceEvidence,
    propeller: Propeller,
    condition_fingerprint_sha256: String,
    reference_velocity_mps: f64,
    coordinate_frame: CoordinateFrame,
    angular_origin: AngularOrigin,
    positive_azimuth: PositiveAzimuth,
    axial_positive_direction: AxialPositiveDirection,
    tolerances: ComponentTolerances,
    max_fourier_harmonic: usize,
    blockage: BlockageEvidence,
    target: WakeField,
    realized: WakeField,
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
struct Propeller {
    revision: String,
    diameter_m: f64,
    hub_ratio: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ComponentTolerances {
    axial: f64,
    radial: f64,
    tangential: f64,
    nominal_wake_fraction: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BlockageEvidence {
    section_shape: String,
    section_width_m: Option<f64>,
    section_height_m: Option<f64>,
    section_diameter_m: Option<f64>,
    method: BlockageMethod,
    maximum_hull_frontal_area_m2: Option<f64>,
    other_blocking_area_m2: Option<f64>,
    other_basis_description: Option<String>,
    review_state: BlockageReviewState,
    review_evidence_sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WakeField {
    field_id: String,
    method: VelocityMethod,
    scale_basis: WakeBasis,
    propeller_revision: String,
    condition_fingerprint_sha256: String,
    reference_velocity_mps: f64,
    instrument_or_solver_id: String,
    calibration_or_solver_validation_sha256: String,
    raw_data_sha256: String,
    coordinate_frame: CoordinateFrame,
    angular_origin: AngularOrigin,
    positive_azimuth: PositiveAzimuth,
    axial_positive_direction: AxialPositiveDirection,
    area_weight_method_id: String,
    area_weight_evidence_sha256: String,
    points: Vec<WakePoint>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WakePoint {
    r_over_r: f64,
    azimuth_deg: f64,
    area_weight: f64,
    vx_over_vref: Option<VelocityComponent>,
    vr_over_vref: Option<VelocityComponent>,
    vt_over_vref: Option<VelocityComponent>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VelocityComponent {
    value: f64,
    expanded_uncertainty: f64,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    schema_version: &'static str,
    case_id: String,
    source_dataset_id: String,
    source_data_sha256: String,
    evidence_kind: &'static str,
    disposition: &'static str,
    claim_ceiling: &'static str,
    blockage: BlockageReport,
    target_summary: FieldSummary,
    realized_summary: FieldSummary,
    component_metrics: Vec<ComponentMetric>,
    cell_comparisons: Vec<CellComparison>,
    nominal_wake_fraction: WakeFractionComparison,
    fourier_comparisons: Vec<FourierComparison>,
    reasons: Vec<String>,
}

#[derive(Debug, Serialize)]
struct BlockageReport {
    method: &'static str,
    blocking_area_basis: &'static str,
    blocking_area_m2: f64,
    test_section_area_m2: f64,
    area_blockage_ratio: f64,
    below_ittc_twenty_percent_recommendation: bool,
    review_state: &'static str,
    review_required: bool,
    review_assessment_eligible: bool,
    review_evidence_sha256: Option<String>,
    interpretation: &'static str,
}

#[derive(Debug, Serialize)]
struct FieldSummary {
    field_id: String,
    method: &'static str,
    scale_basis: &'static str,
    point_count: usize,
    axial_coverage_fraction: f64,
    radial_coverage_fraction: f64,
    tangential_coverage_fraction: f64,
    area_weight_sum: f64,
    nominal_wake_fraction: Option<f64>,
    nominal_wake_fraction_expanded_bound: Option<f64>,
}

#[derive(Debug, Serialize)]
struct ComponentMetric {
    component: &'static str,
    point_count: usize,
    bias: Option<f64>,
    mean_absolute_error: Option<f64>,
    rmse: Option<f64>,
    p95_absolute_error: Option<f64>,
    maximum_absolute_error: Option<f64>,
    tolerance: f64,
    area_weight_sum: f64,
    unweighted_bias: Option<f64>,
    unweighted_mean_absolute_error: Option<f64>,
    unweighted_rmse: Option<f64>,
    area_weight_sum: f64,
    unweighted_bias: Option<f64>,
    unweighted_mean_absolute_error: Option<f64>,
    unweighted_rmse: Option<f64>,
    definitely_outside_cells: usize,
    unresolved_cells: usize,
    within_tolerance_cells: usize,
}

#[derive(Debug, Serialize)]
struct CellComparison {
    r_over_r: f64,
    azimuth_deg: f64,
    component: &'static str,
    target_value: f64,
    realized_value: f64,
    signed_residual: f64,
    conservative_residual_lower: f64,
    conservative_residual_upper: f64,
    tolerance: f64,
    disposition: &'static str,
}

#[derive(Debug, Serialize)]
struct WakeFractionComparison {
    target: Option<f64>,
    target_expanded_bound: Option<f64>,
    realized: Option<f64>,
    realized_expanded_bound: Option<f64>,
    signed_difference: Option<f64>,
    conservative_difference_lower: Option<f64>,
    conservative_difference_upper: Option<f64>,
    tolerance: f64,
    disposition: &'static str,
}

#[derive(Debug, Serialize)]
struct FourierComparison {
    r_over_r: f64,
    component: &'static str,
    harmonic: usize,
    target_amplitude: f64,
    realized_amplitude: f64,
    amplitude_difference: f64,
    target_phase_deg: Option<f64>,
    realized_phase_deg: Option<f64>,
    circular_phase_difference_deg: Option<f64>,
}

#[derive(Debug)]
struct MetricAccumulator {
    residuals: Vec<f64>,
    lower_abs: Vec<f64>,
    upper_abs: Vec<f64>,
    weighted_values: Vec<(f64, f64)>,
    weighted_residual_sum: f64,
    weighted_absolute_sum: f64,
    weighted_square_sum: f64,
    weight_sum: f64,
    bias_sum: f64,
    outside: usize,
    unresolved: usize,
    within: usize,
}

impl MetricAccumulator {
    fn new() -> Self {
        Self {
            residuals: Vec::new(),
            lower_abs: Vec::new(),
            upper_abs: Vec::new(),
            weighted_values: Vec::new(),
            weighted_residual_sum: 0.0,
            weighted_absolute_sum: 0.0,
            weighted_square_sum: 0.0,
            weight_sum: 0.0,
            bias_sum: 0.0,
            outside: 0,
            unresolved: 0,
            within: 0,
        }
    }

    fn push(&mut self, delta: f64, uncertainty: f64, tolerance: f64, area_weight: f64) -> &'static str {
        let absolute = delta.abs();
        let lower = (absolute - uncertainty).max(0.0);
        let upper = absolute + uncertainty;
        self.residuals.push(delta);
        self.lower_abs.push(lower);
        self.upper_abs.push(upper);
        self.weighted_values.push((absolute, area_weight));
        self.weighted_residual_sum += delta * area_weight;
        self.weighted_absolute_sum += absolute * area_weight;
        self.weighted_square_sum += delta * delta * area_weight;
        self.weight_sum += area_weight;
        self.bias_sum += delta;
        if lower > tolerance {
            self.outside += 1;
            "outside_tolerance"
        } else if upper <= tolerance {
            self.within += 1;
            "within_tolerance"
        } else {
            self.unresolved += 1;
            "uncertainty_overlaps_tolerance"
        }
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn method_name(value: VelocityMethod) -> &'static str {
    match value {
        VelocityMethod::FiveHolePitot => "five_hole_pitot",
        VelocityMethod::OtherPitot => "other_pitot",
        VelocityMethod::Rans => "rans",
        VelocityMethod::Cfd => "cfd",
        VelocityMethod::WireScreenDerived => "wire_screen_derived",
        VelocityMethod::OtherMeasured => "other_measured",
    }
}

fn scale_name(value: WakeBasis) -> &'static str {
    match value { WakeBasis::ModelScale => "model_scale", WakeBasis::FullScale => "full_scale" }
}

fn blockage_method_name(value: BlockageMethod) -> &'static str {
    match value {
        BlockageMethod::WireScreen => "wire_screen",
        BlockageMethod::DummyHull => "dummy_hull",
        BlockageMethod::FullHull => "full_hull",
        BlockageMethod::OtherDeclared => "other_declared",
    }
}

fn review_name(value: BlockageReviewState) -> &'static str {
    match value {
        BlockageReviewState::NotRequired => "not_required",
        BlockageReviewState::Reviewed => "reviewed",
        BlockageReviewState::Unknown => "unknown",
    }
}

fn component_name(index: usize) -> &'static str {
    match index { 0 => "axial", 1 => "radial", _ => "tangential" }
}

fn component<'a>(point: &'a WakePoint, index: usize) -> Option<&'a VelocityComponent> {
    match index { 0 => point.vx_over_vref.as_ref(), 1 => point.vr_over_vref.as_ref(), _ => point.vt_over_vref.as_ref() }
}

fn component_tolerance(tolerances: &ComponentTolerances, index: usize) -> f64 {
    match index { 0 => tolerances.axial, 1 => tolerances.radial, _ => tolerances.tangential }
}

fn section_area(blockage: &BlockageEvidence) -> Option<f64> {
    match blockage.section_shape.as_str() {
        "rectangular" => Some(blockage.section_width_m? * blockage.section_height_m?),
        "circular" => Some(std::f64::consts::PI * blockage.section_diameter_m?.powi(2) / 4.0),
        _ => None,
    }
}

fn calculate_blockage(case: &Case) -> Result<BlockageReport, String> {
    let area = section_area(&case.blockage).ok_or_else(|| "unknown or incomplete test-section geometry".to_string())?;
    if !finite_positive(area) {
        return Err("test-section area must be finite and positive".to_string());
    }
    let (blocking_area, basis) = match case.blockage.method {
        BlockageMethod::WireScreen => {
            (std::f64::consts::PI * case.propeller.diameter_m.powi(2) / 4.0, "propeller_disk_area_per_ittc_rev03_table_5")
        }
        BlockageMethod::DummyHull | BlockageMethod::FullHull => {
            (case.blockage.maximum_hull_frontal_area_m2.ok_or_else(|| "dummy/full hull requires maximum_hull_frontal_area_m2".to_string())?, "maximum_hull_frontal_area_per_ittc_rev03_table_5")
        }
        BlockageMethod::OtherDeclared => {
            if case.blockage.other_basis_description.as_deref().unwrap_or("").trim().is_empty() {
                return Err("other blockage method requires an explicit basis description".to_string());
            }
            (case.blockage.other_blocking_area_m2.ok_or_else(|| "other blockage method requires other_blocking_area_m2".to_string())?, "explicit_other_declared_projected_area")
        }
    };
    if !finite_positive(blocking_area) {
        return Err("blocking area must be finite and positive".to_string());
    }
    let ratio = blocking_area / area;
    if !ratio.is_finite() || ratio >= 1.0 {
        return Err("blockage ratio must be finite and below 100%".to_string());
    }
    let below_recommendation = ratio < ITTC_BLOCKAGE_RECOMMENDATION;
    let review_required = !below_recommendation;
    if let Some(digest) = case.blockage.review_evidence_sha256.as_deref() {
        if !is_sha256(digest) {
            return Err("blockage review evidence digest is malformed".to_string());
        }
    }
    Ok(BlockageReport {
        method: blockage_method_name(case.blockage.method),
        blocking_area_basis: basis,
        blocking_area_m2: blocking_area,
        test_section_area_m2: area,
        area_blockage_ratio: ratio,
        below_ittc_twenty_percent_recommendation: below_recommendation,
        review_state: review_name(case.blockage.review_state),
        review_required,
        review_assessment_eligible: !review_required
            || (case.blockage.review_state == BlockageReviewState::Reviewed
                && case.blockage.review_evidence_sha256.as_deref().is_some_and(is_sha256)),
        review_evidence_sha256: case.blockage.review_evidence_sha256.clone(),
        interpretation: "area_ratio_is_an_ittc_recommendation_marker_not_a_correction_or_proof_of_zero_acoustic_blockage_effect",
    })
}

fn point_key(point: &WakePoint) -> (i64, i64) {
    ((point.r_over_r / GRID_TOLERANCE).round() as i64, (point.azimuth_deg / GRID_TOLERANCE).round() as i64)
}

fn validate_field(field: &WakeField, case: &Case, label: &str, errors: &mut Vec<String>) {
    if field.field_id.trim().is_empty()
        || field.instrument_or_solver_id.trim().is_empty()
        || field.area_weight_method_id.trim().is_empty()
    {
        errors.push(format!("{label}: field, instrument/solver and area-weight method IDs are required"));
    }
    for (name, digest) in [
        ("condition fingerprint", &field.condition_fingerprint_sha256),
        ("calibration/solver validation", &field.calibration_or_solver_validation_sha256),
        ("raw data", &field.raw_data_sha256),
        ("area quadrature evidence", &field.area_weight_evidence_sha256),
    ] {
        if !is_sha256(digest) {
            errors.push(format!("{label}: {name} digest must be SHA-256 hex"));
        }
    }
    if field.propeller_revision != case.propeller.revision
        || field.condition_fingerprint_sha256 != case.condition_fingerprint_sha256
        || (field.reference_velocity_mps - case.reference_velocity_mps).abs() > GRID_TOLERANCE
    {
        errors.push(format!("{label}: propeller revision, condition fingerprint or reference velocity differs from frozen case"));
    }
    if !finite_positive(field.reference_velocity_mps) || field.points.is_empty() {
        errors.push(format!("{label}: reference velocity must be positive and polar grid non-empty"));
    }
    let mut keys = BTreeSet::new();
    let mut sum = 0.0;
    for point in &field.points {
        if !point.r_over_r.is_finite()
            || point.r_over_r < case.propeller.hub_ratio
            || point.r_over_r > 1.0
            || !point.azimuth_deg.is_finite()
            || !(0.0..360.0).contains(&point.azimuth_deg)
            || !finite_positive(point.area_weight)
        {
            errors.push(format!("{label}: invalid polar coordinate or area weight"));
        }
        if !keys.insert(point_key(point)) {
            errors.push(format!("{label}: duplicate polar grid coordinate"));
        }
        sum += point.area_weight;
        for (index, value) in [point.vx_over_vref.as_ref(), point.vr_over_vref.as_ref(), point.vt_over_vref.as_ref()].into_iter().enumerate() {
            if let Some(value) = value {
                if !value.value.is_finite() || !finite_positive(value.expanded_uncertainty) {
                    errors.push(format!("{label}: {} component value/uncertainty invalid", component_name(index)));
                }
            }
        }
        if field.method == VelocityMethod::WireScreenDerived
            && (point.vr_over_vref.is_some() || point.vt_over_vref.is_some())
        {
            errors.push(format!("{label}: wire-screen-derived method cannot declare measured radial/tangential fidelity without separate measurement evidence"));
        }
    }
    if (sum - 1.0).abs() > WEIGHT_SUM_TOLERANCE {
        errors.push(format!("{label}: disc area weights must sum to 1 (observed {sum})"));
    }
}

fn validate_case(case: &Case) -> Vec<String> {
    let mut errors = Vec::new();
    if case.schema_version != SCHEMA_VERSION {
        errors.push(format!("unsupported schema_version {}; expected {SCHEMA_VERSION}", case.schema_version));
    }
    if case.case_id.trim().is_empty()
        || case.source.dataset_id.trim().is_empty()
        || case.source.citation.trim().is_empty()
        || case.propeller.revision.trim().is_empty()
    {
        errors.push("case, dataset, citation and propeller revision are required".to_string());
    }
    if !is_sha256(&case.source.raw_data_sha256)
        || !is_sha256(&case.condition_fingerprint_sha256)
    {
        errors.push("source and condition digests must be 64-character SHA-256".to_string());
    }
    if !finite_positive(case.propeller.diameter_m)
        || !case.propeller.hub_ratio.is_finite()
        || case.propeller.hub_ratio < 0.0
        || case.propeller.hub_ratio >= 1.0
        || !finite_positive(case.reference_velocity_mps)
    {
        errors.push("invalid propeller diameter, hub ratio or reference velocity".to_string());
    }
    for (name, value) in [
        ("axial", case.tolerances.axial),
        ("radial", case.tolerances.radial),
        ("tangential", case.tolerances.tangential),
        ("nominal wake fraction", case.tolerances.nominal_wake_fraction),
    ] {
        if !finite_positive(value) {
            errors.push(format!("{name} tolerance must be positive"));
        }
    }
    if case.max_fourier_harmonic == 0 {
        errors.push("max_fourier_harmonic must be positive".to_string());
    }
    validate_field(&case.target, case, "target", &mut errors);
    validate_field(&case.realized, case, "realized", &mut errors);
    if let Err(error) = calculate_blockage(case) {
        errors.push(error);
    }
    errors
}

fn same_grid(target: &WakeField, realized: &WakeField) -> bool {
    if target.points.len() != realized.points.len() {
        return false;
    }
    let mut t = target.points.iter().map(point_key).collect::<Vec<_>>();
    let mut r = realized.points.iter().map(point_key).collect::<Vec<_>>();
    t.sort_unstable();
    r.sort_unstable();
    t == r
}

fn same_quadrature(target: &WakeField, realized: &WakeField) -> bool {
    if target.area_weight_method_id != realized.area_weight_method_id {
        return false;
    }
    let target_points = grid_index(target);
    let realized_points = grid_index(realized);
    target_points.iter().all(|(key, target_point)| {
        realized_points.get(key).is_some_and(|realized_point| {
            (target_point.area_weight - realized_point.area_weight).abs() <= WEIGHT_SUM_TOLERANCE
        })
    })
}

fn field_coordinates_match_case(field: &WakeField, case: &Case) -> bool {
    field.coordinate_frame == case.coordinate_frame
        && field.angular_origin == case.angular_origin
        && field.positive_azimuth == case.positive_azimuth
        && field.axial_positive_direction == case.axial_positive_direction
}

fn field_coordinates_match_each_other(target: &WakeField, realized: &WakeField) -> bool {
    target.coordinate_frame == realized.coordinate_frame
        && target.angular_origin == realized.angular_origin
        && target.positive_azimuth == realized.positive_azimuth
        && target.axial_positive_direction == realized.axial_positive_direction
}

fn grid_index(field: &WakeField) -> BTreeMap<(i64, i64), &WakePoint> {
    field.points.iter().map(|point| (point_key(point), point)).collect()
}

fn empirical_quantile(sorted: &[f64], probability: f64) -> Option<f64> {
    if sorted.is_empty() || !(0.0..=1.0).contains(&probability) { return None; }
    let position = probability * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    Some(sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64))
}

fn weighted_quantile(values: &[(f64, f64)], probability: f64) -> Option<f64> {
    if values.is_empty() || !(0.0..=1.0).contains(&probability) {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.0.total_cmp(&right.0));
    let total_weight: f64 = sorted.iter().map(|(_, weight)| *weight).sum();
    if !finite_positive(total_weight)
        || sorted.iter().any(|(value, weight)| !value.is_finite() || !finite_positive(*weight))
    {
        return None;
    }
    let target_weight = probability * total_weight;
    let mut accumulated = 0.0;
    for (value, weight) in sorted {
        accumulated += weight;
        if accumulated >= target_weight {
            return Some(value);
        }
    }
    None
}

fn compute_field_summary(field: &WakeField) -> FieldSummary {
    let total = field.points.len() as f64;
    let (axial, radial, tangential, weight_sum) = if total > 0.0 {
        (
            field.points.iter().filter(|p| p.vx_over_vref.is_some()).count() as f64 / total,
            field.points.iter().filter(|p| p.vr_over_vref.is_some()).count() as f64 / total,
            field.points.iter().filter(|p| p.vt_over_vref.is_some()).count() as f64 / total,
            field.points.iter().filter(|p| p.area_weight.is_finite()).map(|p| p.area_weight).sum::<f64>(),
        )
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };
    let valid_axial_quadrature = total > 0.0
        && field.points.iter().all(|point| {
            point.area_weight.is_finite()
                && point.area_weight > 0.0
                && point.vx_over_vref.as_ref().is_some_and(|component| {
                    component.value.is_finite() && component.expanded_uncertainty.is_finite()
                })
        });
    let wake_fraction = if valid_axial_quadrature {
        let value = 1.0 - field.points.iter()
            .map(|point| point.area_weight * point.vx_over_vref.as_ref().unwrap().value)
            .sum::<f64>();
        value.is_finite().then_some(value)
    } else {
        None
    };
    let wake_bound = if valid_axial_quadrature {
        let value = field.points.iter()
            .map(|point| point.area_weight * point.vx_over_vref.as_ref().unwrap().expanded_uncertainty)
            .sum::<f64>();
        value.is_finite().then_some(value)
    } else {
        None
    };
    FieldSummary {
        field_id: field.field_id.clone(),
        method: method_name(field.method),
        scale_basis: scale_name(field.scale_basis),
        point_count: field.points.len(),
        axial_coverage_fraction: axial,
        radial_coverage_fraction: radial,
        tangential_coverage_fraction: tangential,
        area_weight_sum: weight_sum,
        nominal_wake_fraction: wake_fraction,
        nominal_wake_fraction_expanded_bound: wake_bound,
    }
}

fn summarize_component(
    index: usize,
    target: &WakeField,
    realized: &WakeField,
    tolerance: f64,
) -> (ComponentMetric, Vec<CellComparison>) {
    let tmap = grid_index(target);
    let rmap = grid_index(realized);
    let mut acc = MetricAccumulator::new();
    let mut comparisons = Vec::new();
    let mut keys = tmap.keys().copied().collect::<Vec<_>>();
    keys.sort_unstable();
    for key in keys {
        let tp = tmap[&key];
        let rp = rmap[&key];
        let (Some(t), Some(r)) = (component(tp, index), component(rp, index)) else { continue; };
        let delta = r.value - t.value;
        let uncertainty = t.expanded_uncertainty + r.expanded_uncertainty;
        let disposition = acc.push(delta, uncertainty, tolerance, tp.area_weight);
        let lower = (delta.abs() - uncertainty).max(0.0);
        let upper = delta.abs() + uncertainty;
        comparisons.push(CellComparison {
            r_over_r: tp.r_over_r,
            azimuth_deg: tp.azimuth_deg,
            component: component_name(index),
            target_value: t.value,
            realized_value: r.value,
            signed_residual: delta,
            conservative_residual_lower: lower,
            conservative_residual_upper: upper,
            tolerance,
            disposition,
        });
    }
    let n = acc.residuals.len();
    let mut absolute = acc.residuals.iter().map(|x| x.abs()).collect::<Vec<_>>();
    absolute.sort_by(f64::total_cmp);
    let unweighted_rmse = if n == 0 { None } else { Some((acc.residuals.iter().map(|x| x*x).sum::<f64>() / n as f64).sqrt()) };
    let metric = ComponentMetric {
        component: component_name(index),
        point_count: n,
        bias: if acc.weight_sum > 0.0 { Some(acc.weighted_residual_sum / acc.weight_sum) } else { None },
        mean_absolute_error: if acc.weight_sum > 0.0 { Some(acc.weighted_absolute_sum / acc.weight_sum) } else { None },
        rmse: if acc.weight_sum > 0.0 { Some((acc.weighted_square_sum / acc.weight_sum).sqrt()) } else { None },
        p95_absolute_error: weighted_quantile(&acc.weighted_values, 0.95),
        maximum_absolute_error: absolute.last().copied(),
        tolerance,
        area_weight_sum: acc.weight_sum,
        unweighted_bias: if n == 0 { None } else { Some(acc.bias_sum / n as f64) },
        unweighted_mean_absolute_error: if n == 0 { None } else { Some(absolute.iter().sum::<f64>() / n as f64) },
        unweighted_rmse,
        definitely_outside_cells: acc.outside,
        unresolved_cells: acc.unresolved,
        within_tolerance_cells: acc.within,
    };
    (metric, comparisons)
}

fn nominal_wake_comparison(
    target: &FieldSummary,
    realized: &FieldSummary,
    tolerance: f64,
) -> WakeFractionComparison {
    match (
        target.nominal_wake_fraction,
        target.nominal_wake_fraction_expanded_bound,
        realized.nominal_wake_fraction,
        realized.nominal_wake_fraction_expanded_bound,
    ) {
        (Some(t), Some(tu), Some(r), Some(ru)) => {
            let delta = r - t;
            let uncertainty = tu + ru;
            let lower = (delta.abs() - uncertainty).max(0.0);
            let upper = delta.abs() + uncertainty;
            let status = if lower > tolerance { "outside_tolerance" }
                else if upper <= tolerance { "within_tolerance" }
                else { "uncertainty_overlaps_tolerance" };
            WakeFractionComparison {
                target: Some(t), target_expanded_bound: Some(tu),
                realized: Some(r), realized_expanded_bound: Some(ru),
                signed_difference: Some(delta),
                conservative_difference_lower: Some(lower),
                conservative_difference_upper: Some(upper),
                tolerance, disposition: status,
            }
        }
        _ => WakeFractionComparison {
            target: target.nominal_wake_fraction,
            target_expanded_bound: target.nominal_wake_fraction_expanded_bound,
            realized: realized.nominal_wake_fraction,
            realized_expanded_bound: realized.nominal_wake_fraction_expanded_bound,
            signed_difference: None, conservative_difference_lower: None,
            conservative_difference_upper: None, tolerance,
            disposition: "incomplete_missing_axial_coverage",
        }
    }
}

fn sorted_unique(values: &mut Vec<f64>) -> bool {
    values.sort_by(f64::total_cmp);
    if values.len() < 4 { return false; }
    for pair in values.windows(2) {
        if (pair[1] - pair[0]).abs() <= ANGLE_TOLERANCE_DEG { return false; }
    }
    let n = values.len();
    let spacing = 360.0 / n as f64;
    values.iter().enumerate().all(|(i, value)| {
        let expected = i as f64 * spacing;
        (*value - expected).abs() <= ANGLE_TOLERANCE_DEG
    })
}

fn harmonic_coefficients(points: &[&WakePoint], index: usize, max_harmonic: usize) -> Option<Vec<(f64, f64)>> {
    if points.is_empty() { return None; }
    let mut angles = points.iter().map(|point| point.azimuth_deg).collect::<Vec<_>>();
    if !sorted_unique(&mut angles) || max_harmonic >= (angles.len() + 1) / 2 {
        return None;
    }
    let map = points.iter().map(|point| (point_key(point), *point)).collect::<BTreeMap<_, _>>();
    let mut coefficients = Vec::new();
    let dc = angles.iter().map(|angle| {
        let key = ((points[0].r_over_r / GRID_TOLERANCE).round() as i64, (*angle / GRID_TOLERANCE).round() as i64);
        component(map.get(&key).copied()?, index).map(|component| component.value)
    }).collect::<Option<Vec<_>>>()?;
    let n = dc.len() as f64;
    coefficients.push((dc.iter().sum::<f64>() / n, 0.0));
    for harmonic in 1..=max_harmonic {
        let mut a = 0.0;
        let mut b = 0.0;
        for angle in &angles {
            let key = ((points[0].r_over_r / GRID_TOLERANCE).round() as i64, (*angle / GRID_TOLERANCE).round() as i64);
            let value = component(map.get(&key).copied()?, index)?.value;
            let phase = (harmonic as f64) * angle.to_radians();
            a += value * phase.cos();
            b += value * phase.sin();
        }
        coefficients.push((2.0 * a / n, 2.0 * b / n));
    }
    Some(coefficients)
}

fn circular_difference_deg(left: f64, right: f64) -> f64 {
    ((left - right + 180.0).rem_euclid(360.0)) - 180.0
}

fn fourier_metrics(
    target: &WakeField,
    realized: &WakeField,
    max_harmonic: usize,
) -> Vec<FourierComparison> {
    let mut radii = target.points.iter().map(|p| p.r_over_r).collect::<Vec<_>>();
    radii.sort_by(f64::total_cmp);
    radii.dedup_by(|a,b| (*a-*b).abs() <= GRID_TOLERANCE);
    let mut output = Vec::new();
    let tmap = grid_index(target);
    let rmap = grid_index(realized);
    for radius in radii {
        let mut target_ring = target.points.iter().filter(|p| (p.r_over_r-radius).abs()<=GRID_TOLERANCE).collect::<Vec<_>>();
        let mut realized_ring = realized.points.iter().filter(|p| (p.r_over_r-radius).abs()<=GRID_TOLERANCE).collect::<Vec<_>>();
        target_ring.sort_by(|a,b| a.azimuth_deg.total_cmp(&b.azimuth_deg));
        realized_ring.sort_by(|a,b| a.azimuth_deg.total_cmp(&b.azimuth_deg));
        if target_ring.len()!=realized_ring.len()
            || target_ring.iter().zip(&realized_ring).any(|(a,b)|(a.azimuth_deg-b.azimuth_deg).abs()>ANGLE_TOLERANCE_DEG)
        { continue; }
        for component_index in 0..3 {
            let (Some(tc),Some(rc)) = (
                harmonic_coefficients(&target_ring,component_index,max_harmonic),
                harmonic_coefficients(&realized_ring,component_index,max_harmonic)
            ) else { continue; };
            for harmonic in 0..=max_harmonic {
                let (ta,tb)=tc[harmonic];
                let (ra,rb)=rc[harmonic];
                let ta_amp=if harmonic==0 { ta.abs() } else { ta.hypot(tb) };
                let ra_amp=if harmonic==0 { ra.abs() } else { ra.hypot(rb) };
                let target_phase=if harmonic>0 && ta_amp>ZERO_AMPLITUDE_EPSILON { Some(tb.atan2(ta).to_degrees().rem_euclid(360.0)) } else { None };
                let realized_phase=if harmonic>0 && ra_amp>ZERO_AMPLITUDE_EPSILON { Some(rb.atan2(ra).to_degrees().rem_euclid(360.0)) } else { None };
                let phase_difference=match (target_phase,realized_phase) {
                    (Some(t),Some(r))=>Some(circular_difference_deg(r,t)), _=>None
                };
                output.push(FourierComparison {
                    r_over_r:radius,
                    component:component_name(component_index),
                    harmonic,
                    target_amplitude:ta_amp,
                    realized_amplitude:ra_amp,
                    amplitude_difference:ra_amp-ta_amp,
                    target_phase_deg:target_phase,
                    realized_phase_deg:realized_phase,
                    circular_phase_difference_deg:phase_difference,
                });
            }
        }
    }
    output
}

fn audit(case: &Case) -> AuditReport {
    let errors = validate_case(case);
    let mut reasons = errors.clone();
    let blockage = calculate_blockage(case).ok();
    let target_summary = compute_field_summary(&case.target);
    let realized_summary = compute_field_summary(&case.realized);
    let mut component_metrics = Vec::new();
    let mut cell_comparisons = Vec::new();
    let mut fourier_comparisons = Vec::new();
    let mut has_definite_failure = false;
    let mut unresolved = !errors.is_empty();
    let grid_matches = same_grid(&case.target,&case.realized);

    if !grid_matches {
        reasons.push("target and realized wake fields do not have exactly the same polar grid; interpolation is prohibited in v0.1".to_string());
        unresolved = true;
    }
    let coordinate_mismatch = case.target.reference_velocity_mps != case.realized.reference_velocity_mps
        || case.target.propeller_revision != case.realized.propeller_revision
        || case.target.condition_fingerprint_sha256 != case.realized.condition_fingerprint_sha256
        || !field_coordinates_match_each_other(&case.target, &case.realized)
        || !field_coordinates_match_case(&case.target, case)
        || !field_coordinates_match_case(&case.realized, case);
    if coordinate_mismatch {
        reasons.push("target and realized operating-point, reference-velocity or coordinate-frame identities differ".to_string());
        unresolved = true;
    }
    let scale_mismatch = case.target.scale_basis != case.realized.scale_basis;
    if scale_mismatch {
        reasons.push("model-scale and full-scale wake fields cannot be compared without an explicit scale-transfer method".to_string());
        unresolved = true;
    }
    let quadrature_mismatch = grid_matches && !same_quadrature(&case.target, &case.realized);
    if quadrature_mismatch {
        reasons.push("target and realized polar grids use different area-weight methods or weights; area-integrated comparisons are withheld".to_string());
        unresolved = true;
    }

    if grid_matches && !coordinate_mismatch && !scale_mismatch && !quadrature_mismatch && errors.is_empty() {
        let coverage=[0,1,2].map(|index| {
            case.target.points.iter().zip(&case.realized.points).all(|(t,r)| component(t,index).is_some()&&component(r,index).is_some())
        });
        for index in 0..3 {
            let tol=component_tolerance(&case.tolerances,index);
            let (metric,cells)=summarize_component(index,&case.target,&case.realized,tol);
            if metric.point_count>0 {
                has_definite_failure |= metric.definitely_outside_cells>0;
                unresolved |= metric.unresolved_cells>0;
                component_metrics.push(metric);
                cell_comparisons.extend(cells);
            }
        }
        if !coverage[1] || !coverage[2] {
            reasons.push("one or more wake velocity components are missing; axial-only results cannot qualify full-vector wake similarity".to_string());
            unresolved = true;
        }
        let t = &target_summary;
        let r = &realized_summary;
        let wake=nominal_wake_comparison(t,r,case.tolerances.nominal_wake_fraction);
        if wake.disposition=="outside_tolerance" { has_definite_failure=true; }
        if wake.disposition!="within_tolerance" { unresolved=true; }
        fourier_comparisons=fourier_metrics(&case.target,&case.realized,case.max_fourier_harmonic);
        if fourier_comparisons.is_empty() {
            reasons.push("Fourier harmonics withheld: incomplete/nonuniform azimuth samples or insufficient anti-aliasing coverage".to_string());
            unresolved=true;
        }
    } else {
        reasons.push("component and Fourier residuals withheld because input validation, grid or coordinate identity failed".to_string());
        unresolved=true;
    }

    if let Some(blockage)=&blockage {
        if blockage.review_required && !blockage.review_assessment_eligible {
            reasons.push("area blockage is >=20%; a reviewed blockage assessment digest is required before comparison".to_string());
            unresolved = true;
        } else if !blockage.below_ittc_twenty_percent_recommendation {
            reasons.push("area blockage is >=20%; a reviewed assessment is present, but this does not constitute an acoustic blockage correction".to_string());
        }
    }
    if case.evidence_kind==EvidenceKind::Synthetic {
        reasons.push("synthetic wake fields cannot qualify measured wake similarity".to_string());
        unresolved=true;
    }
    if case.source.rights_status!=RightsStatus::PublicOrPermitted || !case.source.independently_reviewed {
        reasons.push("source rights or independent review are not eligible for physical evidence".to_string());
        unresolved=true;
    }
    let full_vector=case.target.points.iter().all(|p|p.vx_over_vref.is_some()&&p.vr_over_vref.is_some()&&p.vt_over_vref.is_some())
        &&case.realized.points.iter().all(|p|p.vx_over_vref.is_some()&&p.vr_over_vref.is_some()&&p.vt_over_vref.is_some());
    let disposition=if !errors.is_empty() {
        "INVALID"
    } else if has_definite_failure || coordinate_mismatch || scale_mismatch || quadrature_mismatch || !grid_matches {
        "INCOMPARABLE"
    } else if case.evidence_kind==EvidenceKind::Synthetic || unresolved {
        if !full_vector && component_metrics.iter().any(|m|m.component=="axial") {
            "AXIAL_ONLY_DESCRIPTIVE"
        } else {
            "INCOMPLETE"
        }
    } else if !full_vector {
        "AXIAL_ONLY_DESCRIPTIVE"
    } else {
        "FULL_VECTOR_COMPARABLE"
    };
    let wake = nominal_wake_comparison(&target_summary,&realized_summary,case.tolerances.nominal_wake_fraction);
    AuditReport {
        schema_version:SCHEMA_VERSION,
        case_id:case.case_id.clone(),
        source_dataset_id:case.source.dataset_id.clone(),
        source_data_sha256:case.source.raw_data_sha256.clone(),
        evidence_kind:match case.evidence_kind {EvidenceKind::Synthetic=>"synthetic",EvidenceKind::Measured=>"measured"},
        disposition,
        claim_ceiling:"wake_field_arithmetic_and_declared_component_tolerance_comparison_only_no_cfd_validation_reynolds_froude_similarity_cavitation_noise_prediction_propeller_performance_or_ecological_qualification",
        blockage:blockage.unwrap_or(BlockageReport {
            method:blockage_method_name(case.blockage.method),
            blocking_area_basis:"unresolved",
            blocking_area_m2:0.0,
            test_section_area_m2:0.0,
            area_blockage_ratio:0.0,
            below_ittc_twenty_percent_recommendation:false,
            review_state:review_name(case.blockage.review_state),
            review_required:false,
            review_assessment_eligible:false,
            review_evidence_sha256:case.blockage.review_evidence_sha256.clone(),
            interpretation:"invalid_blockage_geometry",
        }),
        target_summary,
        realized_summary,
        component_metrics,
        cell_comparisons,
        nominal_wake_fraction:wake,
        fourier_comparisons,
        reasons,
    }
}

fn main() {
    let mut args=env::args().skip(1);
    let Some(path)=args.next() else {
        eprintln!("usage: marine-wake-fidelity-audit <case.json>");
        process::exit(64);
    };
    if args.next().is_some() { eprintln!("usage: marine-wake-fidelity-audit <case.json>"); process::exit(64); }
    let input=match fs::read_to_string(&path){Ok(x)=>x,Err(e)=>{eprintln!("could not read {path}: {e}");process::exit(64)}};
    let case:Case=match serde_json::from_str(&input){Ok(x)=>x,Err(e)=>{eprintln!("invalid case JSON: {e}");process::exit(64)}};
    let report=audit(&case);
    println!("{}",serde_json::to_string_pretty(&report).expect("report serializes"));
    match report.disposition {
        "FULL_VECTOR_COMPARABLE"=>{},
        "INVALID"|"INCOMPARABLE"=>process::exit(2),
        _=>process::exit(3),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(c:char)->String { std::iter::repeat_n(c,64).collect() }

    fn make_points(perturb:f64, axial_only:bool)->Vec<WakePoint>{
        let mut points=Vec::new();
        let radii=[0.3,0.6,0.9];
        let radial_weights=[0.21875,0.3255208333333333,0.4557291666666667];
        for (ri,r) in radii.iter().enumerate(){
            for ai in 0..8{
                let theta=ai as f64*45.0;
                let rad=theta.to_radians();
                let vx=0.82+0.03*rad.cos()+0.01*(2.0*rad).cos()+perturb;
                let vr=0.01*rad.sin();
                let vt=0.025*(2.0*rad).cos();
                points.push(WakePoint{
                    r_over_r:*r,azimuth_deg:theta,area_weight:radial_weights[ri]/8.0,
                    vx_over_vref:Some(VelocityComponent{value:vx,expanded_uncertainty:0.005}),
                    vr_over_vref:if axial_only{None}else{Some(VelocityComponent{value:vr,expanded_uncertainty:0.003})},
                    vt_over_vref:if axial_only{None}else{Some(VelocityComponent{value:vt,expanded_uncertainty:0.003})},
                });
            }
        }
        points
    }

    fn field(id:&str,method:VelocityMethod,perturb:f64,axial_only:bool)->WakeField{
        WakeField{
            field_id:id.into(),method,scale_basis:WakeBasis::ModelScale,
            propeller_revision:"prop-r7".into(),condition_fingerprint_sha256:digest('1'),
            reference_velocity_mps:5.0,instrument_or_solver_id:format!("{id}-sensor"),
            calibration_or_solver_validation_sha256:digest('2'),raw_data_sha256:digest(if id=="target"{'3'}else{'4'}),
            coordinate_frame:CoordinateFrame::PropellerLookingForwardFromAft,
            angular_origin:AngularOrigin::TopDeadCentre,
            positive_azimuth:PositiveAzimuth::ClockwiseViewedFromAft,
            axial_positive_direction:AxialPositiveDirection::TowardStern,
            area_weight_method_id:"annular-sector-midpoint-v1".into(),
            area_weight_evidence_sha256:digest('7'),
            points:make_points(perturb,axial_only),
        }
    }

    fn case()->Case{
        Case{
            schema_version:SCHEMA_VERSION.into(),case_id:"synthetic-wake-001".into(),
            evidence_kind:EvidenceKind::Synthetic,
            source:SourceEvidence{dataset_id:"synthetic-wake".into(),citation:"Synthetic analytic fixture".into(),
                raw_data_sha256:digest('5'),rights_status:RightsStatus::PublicOrPermitted,independently_reviewed:true},
            propeller:Propeller{revision:"prop-r7".into(),diameter_m:0.25,hub_ratio:0.2},
            condition_fingerprint_sha256:digest('1'),reference_velocity_mps:5.0,
            coordinate_frame:CoordinateFrame::PropellerLookingForwardFromAft,
            angular_origin:AngularOrigin::TopDeadCentre,
            positive_azimuth:PositiveAzimuth::ClockwiseViewedFromAft,
            axial_positive_direction:AxialPositiveDirection::TowardStern,
            tolerances:ComponentTolerances{axial:0.04,radial:0.02,tangential:0.02,nominal_wake_fraction:0.04},
            max_fourier_harmonic:2,
            blockage:BlockageEvidence{
                section_shape:"rectangular".into(),section_width_m:Some(0.5),section_height_m:Some(0.5),section_diameter_m:None,
                method:BlockageMethod::WireScreen,maximum_hull_frontal_area_m2:None,other_blocking_area_m2:None,other_basis_description:None,
                review_state:BlockageReviewState::NotRequired,review_evidence_sha256:None,
            },
            target:field("target",VelocityMethod::FiveHolePitot,0.0,false),
            realized:field("realized",VelocityMethod::Rans,0.005,false),
        }
    }

    #[test]
    fn analytic_wake_fields_are_full_vector_comparable_arithmetically(){
        let mut input=case();input.evidence_kind=EvidenceKind::Measured;
        let report=audit(&input);
        assert_eq!(report.disposition,"FULL_VECTOR_COMPARABLE");
        assert_eq!(report.component_metrics.len(),3);
        assert_eq!(report.cell_comparisons.len(),72);
    }

    #[test]
    fn synthetic_field_never_becomes_physical_qualification(){
        let report=audit(&case());
        assert_eq!(report.disposition,"INCOMPLETE");
        assert!(report.reasons.iter().any(|r|r.contains("synthetic wake fields")));
    }

    #[test]
    fn axial_only_wire_screen_never_becomes_full_vector_pass(){
        let mut input=case();input.evidence_kind=EvidenceKind::Measured;
        input.target=field("target",VelocityMethod::FiveHolePitot,0.0,true);
        input.realized=field("realized",VelocityMethod::WireScreenDerived,0.0,true);
        let report=audit(&input);
        assert_eq!(report.disposition,"AXIAL_ONLY_DESCRIPTIVE");
        assert!(report.component_metrics.iter().any(|m|m.component=="axial"));
        assert!(!report.component_metrics.iter().any(|m|m.component=="radial"));
    }

    #[test]
    fn uncertainty_overlapping_component_tolerance_is_incomplete(){
        let mut input=case();input.evidence_kind=EvidenceKind::Measured;
        input.tolerances.axial=0.007;
        input.realized=field("realized",VelocityMethod::Rans,0.006,false);
        assert_eq!(audit(&input).disposition,"INCOMPLETE");
    }

    #[test]
    fn definite_component_error_outside_tolerance_is_incomparable(){
        let mut input=case();input.evidence_kind=EvidenceKind::Measured;
        input.realized=field("realized",VelocityMethod::Rans,0.10,false);
        assert_eq!(audit(&input).disposition,"INCOMPARABLE");
    }

    #[test]
    fn area_weighted_nominal_wake_fraction_has_known_answer(){
        let report=audit(&case());
        let wake=report.nominal_wake_fraction.target.unwrap();
        assert!((wake-(1.0-0.82)).abs()<0.03);
    }

    #[test]
    fn fourier_harmonics_recover_analytic_first_and_second_modes(){
        let report=audit(&case());
        let first=report.fourier_comparisons.iter().find(|x|x.component=="axial"&&x.harmonic==1).unwrap();
        let second=report.fourier_comparisons.iter().find(|x|x.component=="axial"&&x.harmonic==2).unwrap();
        assert!((first.target_amplitude-0.03).abs()<1.0e-10);
        assert!((second.target_amplitude-0.01).abs()<1.0e-10);
    }

    #[test]
    fn fourier_phase_uses_circular_difference(){
        assert!((circular_difference_deg(1.0,359.0)-2.0).abs()<1.0e-12);
        assert!((circular_difference_deg(359.0,1.0)+2.0).abs()<1.0e-12);
    }

    #[test]
    fn nonuniform_azimuth_grid_withholds_fourier_metrics(){
        let mut input=case();input.target.points[1].azimuth_deg=40.0;
        let report=audit(&input);
        assert!(report.fourier_comparisons.is_empty());
    }

    #[test]
    fn missing_or_mismatched_grid_is_incomplete_not_interpolated(){
        let mut input=case();input.realized.points.pop();
        assert_eq!(audit(&input).disposition,"INVALID");
        let mut input=case();input.realized.points[0].azimuth_deg=22.5;
        assert_eq!(audit(&input).disposition,"INCOMPLETE");
    }

    #[test]
    fn coordinate_convention_mismatch_is_incomparable(){
        let mut input=case();
        input.evidence_kind=EvidenceKind::Measured;
        input.realized.positive_azimuth=PositiveAzimuth::CounterClockwiseViewedFromAft;
        assert_eq!(audit(&input).disposition,"INCOMPARABLE");
    }

    #[test]
    fn different_area_quadrature_methods_are_incomparable() {
        let mut input=case();
        input.evidence_kind=EvidenceKind::Measured;
        input.realized.area_weight_method_id="different-quadrature".into();
        assert_eq!(audit(&input).disposition,"INCOMPARABLE");
    }

    #[test]
    fn area_weighted_component_bias_uses_polar_cell_weights() {
        let mut input=case();
        input.evidence_kind=EvidenceKind::Measured;
        input.target.points[0].area_weight += 0.01;
        input.target.points[1].area_weight -= 0.01;
        input.realized.points[0].area_weight += 0.01;
        input.realized.points[1].area_weight -= 0.01;
        input.realized.points[0].vx_over_vref.as_mut().unwrap().value += 0.05;
        let report=audit(&input);
        let axial=report.component_metrics.iter().find(|m|m.component=="axial").unwrap();
        assert!(axial.area_weight_sum > 0.999);
        assert!((axial.bias.unwrap() - 0.0068671875).abs() < 1.0e-9);
        assert_ne!(axial.bias,axial.unweighted_bias);
    }

    #[test]
    fn area_weight_evidence_digest_is_mandatory() {
        let mut input=case();
        input.target.area_weight_evidence_sha256="bad".into();
        assert_eq!(audit(&input).disposition,"INVALID");
    }

    #[test]
    fn area_weights_must_sum_to_one(){
        let mut input=case();input.target.points[0].area_weight+=0.1;
        assert_eq!(audit(&input).disposition,"INVALID");
    }

    #[test]
    fn wire_screen_uses_propeller_disk_area_for_blockage(){
        let report=audit(&case());
        assert_eq!(report.blockage.blocking_area_basis,"propeller_disk_area_per_ittc_rev03_table_5");
        let expected=std::f64::consts::PI*0.25_f64.powi(2)/4.0/0.25;
        assert!((report.blockage.area_blockage_ratio-expected).abs()<1.0e-12);
    }

    #[test]
    fn dummy_and_full_hull_use_maximum_frontal_area(){
        let mut input=case();
        input.blockage.method=BlockageMethod::DummyHull;
        input.blockage.maximum_hull_frontal_area_m2=Some(0.04);
        let report=audit(&input);
        assert_eq!(report.blockage.blocking_area_basis,"maximum_hull_frontal_area_per_ittc_rev03_table_5");
        assert!((report.blockage.blocking_area_ratio-0.16).abs()<1.0e-12);
    }

    #[test]
    fn high_blockage_requires_reviewed_assessment_digest(){
        let mut input=case();
        input.blockage.section_width_m=Some(0.4);
        input.blockage.section_height_m=Some(0.4);
        assert_eq!(audit(&input).disposition,"INCOMPLETE");
        input.blockage.review_state=BlockageReviewState::Reviewed;
        input.blockage.review_evidence_sha256=Some(digest('6'));
        let report=audit(&input);
        assert!(report.blockage.review_required);
        assert!(report.blockage.review_assessment_eligible);
        assert!(report.reasons.iter().any(|r|r.contains(">=20%")));
    }

    #[test]
    fn wrong_blockage_basis_is_not_silently_accepted(){
        let mut input=case();
        input.blockage.method=BlockageMethod::DummyHull;
        assert_eq!(audit(&input).disposition,"INVALID");
        input.blockage.method=BlockageMethod::WireScreen;
        assert!(audit(&input).blockage.blocking_area_m2>0.0);
    }

    #[test]
    fn scale_basis_mismatch_is_not_compared(){
        let mut input=case();input.realized.scale_basis=WakeBasis::FullScale;
        assert_eq!(audit(&input).disposition,"INCOMPLETE");
    }

    #[test]
    fn invalid_reference_velocity_and_geometry_are_rejected(){
        let mut input=case();input.reference_velocity_mps=0.0;
        assert_eq!(audit(&input).disposition,"INVALID");
        let mut input=case();input.propeller.hub_ratio=1.0;
        assert_eq!(audit(&input).disposition,"INVALID");
    }

    #[test]
    fn malformed_provenance_digest_is_invalid(){
        let mut input=case();input.realized.raw_data_sha256="bad".into();
        assert_eq!(audit(&input).disposition,"INVALID");
    }

    #[test]
    fn component_coverage_is_reported_without_fabricating_missing_cells(){
        let mut input=case();
        input.target.points[0].vr_over_vref=None;
        let report=audit(&input);
        assert!(report.component_metrics.iter().find(|m|m.component=="radial").unwrap().point_count<72);
        assert_eq!(report.disposition,"INCOMPLETE");
    }

    #[test]
    fn unsupported_other_blockage_requires_explicit_basis(){
        let mut input=case();input.blockage.method=BlockageMethod::OtherDeclared;
        assert_eq!(audit(&input).disposition,"INVALID");
        input.blockage.other_blocking_area_m2=Some(0.02);
        input.blockage.other_basis_description=Some("measured projected fixture".into());
        assert!(audit(&input).blockage.area_blockage_ratio>0.0);
    }
}
