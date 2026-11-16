//! Independent sectional-integration reference for full-envelope loading curves.
//!
//! This is a deliberately separate check from G2c's polyhedral mesh
//! integration. It integrates submerged section areas and waterline breadths
//! from the declared station profiles. The longitudinal rule is piecewise
//! linear (trapezoidal sectional integration); no trim, heel, free-surface,
//! resistance, stability approval, or geometry generation is implied.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use crate::full_envelope::{
    FullEnvelopeDisposition, FullEnvelopeSpec, ProfilePoint, SymmetryPolicy,
    compute_full_envelope_spec_digest, validate_full_envelope_spec,
};

pub const SECTIONAL_REFERENCE_SCHEMA_VERSION: u32 = 1;
pub const SECTIONAL_REFERENCE_ALGORITHM: &str = "sectional-area-trapezoid-v1";
const GEOMETRY_TOLERANCE_M: f64 = 1.0e-9;
const MAX_LOADING_CONDITIONS: usize = 128;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoadingCondition {
    pub condition_id: String,
    /// Horizontal waterplane elevation in the source vessel coordinate frame.
    /// The source convention is z-down; values below zero are above the design
    /// waterplane, and values above zero are below it.
    pub waterplane_z_m: f64,
    /// Explicit density. No universal seawater/freshwater value is assumed.
    pub water_density_kg_m3: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SectionalReferenceRequest {
    pub schema_version: u32,
    pub source_spec: FullEnvelopeSpec,
    pub loading_conditions: Vec<LoadingCondition>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SectionalReferenceIssue {
    pub code: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SectionalHydrostaticResult {
    pub algorithm: String,
    pub source_spec_digest_blake3: String,
    pub loading_condition_id: String,
    pub waterplane_z_m: f64,
    pub water_density_kg_m3: f64,
    pub displacement_volume_m3: f64,
    pub displaced_mass_tonnes: f64,
    pub waterplane_area_m2: f64,
    pub tonnes_per_cm_immersion: f64,
    pub longitudinal_center_of_buoyancy_m: f64,
    pub transverse_center_of_buoyancy_m: f64,
    pub vertical_center_of_buoyancy_m: f64,
    pub longitudinal_center_of_flotation_m: f64,
    pub result_digest_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadingSweepReceipt {
    pub receipt_schema_version: u32,
    pub algorithm: String,
    pub source_spec_digest_blake3: String,
    pub results: Vec<SectionalHydrostaticResult>,
    pub sweep_digest_blake3: String,
}

#[derive(Debug, Clone, Copy)]
struct Point2 {
    y: f64,
    z: f64,
}

#[derive(Debug, Clone, Copy)]
struct SectionSample {
    x_m: f64,
    area_m2: f64,
    centroid_y_m: f64,
    centroid_z_m: f64,
    waterline_breadth_m: f64,
}

#[derive(Debug, Clone, Copy)]
struct PolygonProperties {
    area_m2: f64,
    centroid_y_m: f64,
    centroid_z_m: f64,
}

#[derive(Serialize)]
struct ResultDigestIdentity<'a> {
    algorithm: &'a str,
    source_spec_digest_blake3: &'a str,
    loading_condition_id: &'a str,
    waterplane_z_m: f64,
    water_density_kg_m3: f64,
    displacement_volume_m3: f64,
    displaced_mass_tonnes: f64,
    waterplane_area_m2: f64,
    tonnes_per_cm_immersion: f64,
    longitudinal_center_of_buoyancy_m: f64,
    transverse_center_of_buoyancy_m: f64,
    vertical_center_of_buoyancy_m: f64,
    longitudinal_center_of_flotation_m: f64,
}

#[derive(Serialize)]
struct SweepDigestIdentity<'a> {
    receipt_schema_version: u32,
    algorithm: &'a str,
    source_spec_digest_blake3: &'a str,
    results: &'a [SectionalHydrostaticResult],
}

fn push_issue(issues: &mut Vec<SectionalReferenceIssue>, code: &str, detail: impl Into<String>) {
    issues.push(SectionalReferenceIssue {
        code: code.to_owned(),
        detail: detail.into(),
    });
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= GEOMETRY_TOLERANCE_M
}

fn kahan_add(sum: &mut f64, correction: &mut f64, value: f64) {
    let adjusted = value - *correction;
    let next = *sum + adjusted;
    *correction = (next - *sum) - adjusted;
    *sum = next;
}

fn polygon_properties(points: &[Point2]) -> Option<PolygonProperties> {
    if points.len() < 3 {
        return None;
    }
    let mut twice_area = 0.0;
    let mut area_correction = 0.0;
    let mut centroid_y_numerator = 0.0;
    let mut y_correction = 0.0;
    let mut centroid_z_numerator = 0.0;
    let mut z_correction = 0.0;

    for index in 0..points.len() {
        let current = points[index];
        let next = points[(index + 1) % points.len()];
        let cross = current.y * next.z - next.y * current.z;
        kahan_add(&mut twice_area, &mut area_correction, cross);
        kahan_add(
            &mut centroid_y_numerator,
            &mut y_correction,
            (current.y + next.y) * cross,
        );
        kahan_add(
            &mut centroid_z_numerator,
            &mut z_correction,
            (current.z + next.z) * cross,
        );
    }
    if !twice_area.is_finite() || twice_area.abs() <= GEOMETRY_TOLERANCE_M.powi(2) {
        return None;
    }
    let signed_area = 0.5 * twice_area;
    let centroid_y = centroid_y_numerator / (3.0 * twice_area);
    let centroid_z = centroid_z_numerator / (3.0 * twice_area);
    let area = signed_area.abs();
    if !area.is_finite() || !centroid_y.is_finite() || !centroid_z.is_finite() {
        return None;
    }
    Some(PolygonProperties {
        area_m2: area,
        centroid_y_m: centroid_y,
        centroid_z_m: centroid_z,
    })
}

fn interpolate_intersection(a: Point2, b: Point2, plane_z_m: f64) -> Point2 {
    let denominator = b.z - a.z;
    let fraction = if denominator.abs() <= f64::EPSILON {
        0.0
    } else {
        (plane_z_m - a.z) / denominator
    };
    Point2 {
        y: a.y + fraction * (b.y - a.y),
        z: plane_z_m,
    }
}

/// Clip a closed section polygon to the submerged half-plane z >= waterplane.
fn clip_submerged_polygon(points: &[Point2], plane_z_m: f64) -> Vec<Point2> {
    if points.len() < 3 {
        return Vec::new();
    }
    let mut output = Vec::with_capacity(points.len() + 2);
    let mut previous = points[points.len() - 1];
    let mut previous_inside = previous.z >= plane_z_m;
    for current in points {
        let current_inside = current.z >= plane_z_m;
        match (previous_inside, current_inside) {
            (true, true) => output.push(*current),
            (true, false) => output.push(interpolate_intersection(previous, *current, plane_z_m)),
            (false, true) => {
                output.push(interpolate_intersection(previous, *current, plane_z_m));
                output.push(*current);
            }
            (false, false) => {}
        }
        previous = *current;
        previous_inside = current_inside;
    }

    let mut cleaned = Vec::with_capacity(output.len());
    for point in output {
        if cleaned.last().is_none_or(|last: &Point2| {
            (last.y - point.y).abs() > GEOMETRY_TOLERANCE_M
                || (last.z - point.z).abs() > GEOMETRY_TOLERANCE_M
        }) {
            cleaned.push(point);
        }
    }
    if cleaned.len() > 1 {
        let first = cleaned[0];
        let last = *cleaned.last().unwrap();
        if (first.y - last.y).abs() <= GEOMETRY_TOLERANCE_M
            && (first.z - last.z).abs() <= GEOMETRY_TOLERANCE_M
        {
            cleaned.pop();
        }
    }
    cleaned
}

fn section_profile_polygon(
    starboard: &[ProfilePoint],
    port: Option<&[ProfilePoint]>,
    symmetry: SymmetryPolicy,
) -> Vec<Point2> {
    let mut polygon = Vec::with_capacity(starboard.len() * 2);
    polygon.extend(starboard.iter().map(|point| Point2 {
        y: point.y_m,
        z: point.z_m,
    }));
    match symmetry {
        SymmetryPolicy::SymmetricAboutY0 => {
            for point in starboard
                .iter()
                .rev()
                .take(starboard.len().saturating_sub(1))
            {
                polygon.push(Point2 {
                    y: -point.y_m,
                    z: point.z_m,
                });
            }
        }
        SymmetryPolicy::ExplicitPortAndStarboard => {
            if let Some(port_profile) = port {
                for point in port_profile
                    .iter()
                    .rev()
                    .take(port_profile.len().saturating_sub(1))
                {
                    polygon.push(Point2 {
                        y: point.y_m,
                        z: point.z_m,
                    });
                }
            }
        }
    }
    polygon
}

fn profile_waterline_bounds(profile: &[ProfilePoint], plane_z_m: f64) -> Option<(f64, f64)> {
    if profile.len() < 2 {
        return None;
    }
    let mut values = Vec::new();
    for pair in profile.windows(2) {
        let a = Point2 {
            y: pair[0].y_m,
            z: pair[0].z_m,
        };
        let b = Point2 {
            y: pair[1].y_m,
            z: pair[1].z_m,
        };
        let a_on = close(a.z, plane_z_m);
        let b_on = close(b.z, plane_z_m);
        if a_on {
            values.push(a.y);
        }
        if b_on {
            values.push(b.y);
        }
        if !a_on && !b_on && (a.z < plane_z_m) != (b.z < plane_z_m) {
            values.push(interpolate_intersection(a, b, plane_z_m).y);
        }
    }
    let minimum = values.iter().copied().reduce(f64::min)?;
    let maximum = values.iter().copied().reduce(f64::max)?;
    Some((minimum, maximum))
}

fn station_sample(
    spec: &FullEnvelopeSpec,
    station_index: usize,
    plane_z_m: f64,
) -> Result<SectionSample, SectionalReferenceIssue> {
    let station = &spec.stations[station_index];
    let keel_z = station.starboard_profile[0].z_m;
    let sheer_z = station
        .starboard_profile
        .last()
        .map(|point| point.z_m)
        .unwrap_or(0.0);
    if plane_z_m < sheer_z - GEOMETRY_TOLERANCE_M || plane_z_m >= keel_z - GEOMETRY_TOLERANCE_M {
        return Err(SectionalReferenceIssue {
            code: "waterplane_outside_supported_section".into(),
            detail: format!(
                "condition waterplane z={} m does not intersect station {} between its sheer ({} m) and keel ({} m)",
                plane_z_m, station.station_id, sheer_z, keel_z
            ),
        });
    }

    let x_m = station.x_over_length * spec.principal_dimensions.length_m;
    let polygon = section_profile_polygon(
        &station.starboard_profile,
        station.port_profile.as_deref(),
        spec.symmetry,
    );
    let clipped = clip_submerged_polygon(&polygon, plane_z_m);
    let properties = polygon_properties(&clipped);

    let starboard_bounds = profile_waterline_bounds(&station.starboard_profile, plane_z_m)
        .ok_or_else(|| SectionalReferenceIssue {
            code: "starboard_waterline_intersection".into(),
            detail: format!(
                "condition waterplane does not intersect station {} starboard profile",
                station.station_id
            ),
        })?;
    let port_bounds = match spec.symmetry {
        SymmetryPolicy::SymmetricAboutY0 => (-starboard_bounds.1, -starboard_bounds.0),
        SymmetryPolicy::ExplicitPortAndStarboard => {
            let profile =
                station
                    .port_profile
                    .as_deref()
                    .ok_or_else(|| SectionalReferenceIssue {
                        code: "explicit_port_profile_missing".into(),
                        detail: format!("station {} has no port profile", station.station_id),
                    })?;
            profile_waterline_bounds(profile, plane_z_m).ok_or_else(|| SectionalReferenceIssue {
                code: "port_waterline_intersection".into(),
                detail: format!(
                    "condition waterplane does not intersect station {} port profile",
                    station.station_id
                ),
            })?
        }
    };
    let waterline_breadth = (starboard_bounds.1 - port_bounds.0).max(0.0);
    let (area_m2, centroid_y_m, centroid_z_m) = properties
        .map(|item| (item.area_m2, item.centroid_y_m, item.centroid_z_m))
        .unwrap_or((0.0, 0.0, 0.0));

    if !area_m2.is_finite() || !waterline_breadth.is_finite() {
        return Err(SectionalReferenceIssue {
            code: "non_finite_sectional_property".into(),
            detail: format!(
                "station {} produced a non-finite clipped section",
                station.station_id
            ),
        });
    }

    Ok(SectionSample {
        x_m,
        area_m2,
        centroid_y_m,
        centroid_z_m,
        waterline_breadth_m: waterline_breadth,
    })
}

fn integrate_linear_curve(samples: &[SectionSample], value: impl Fn(&SectionSample) -> f64) -> f64 {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for pair in samples.windows(2) {
        let dx = pair[1].x_m - pair[0].x_m;
        let segment = dx * 0.5 * (value(&pair[0]) + value(&pair[1]));
        kahan_add(&mut sum, &mut correction, segment);
    }
    sum
}

/// Exact first moment of a piecewise-linear scalar curve f(x) over each station
/// interval. This avoids applying a trapezoid to x*f(x), which is quadratic.
fn integrate_linear_first_moment(
    samples: &[SectionSample],
    value: impl Fn(&SectionSample) -> f64,
) -> f64 {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for pair in samples.windows(2) {
        let x0 = pair[0].x_m;
        let dx = pair[1].x_m - x0;
        let f0 = value(&pair[0]);
        let f1 = value(&pair[1]);
        let delta = f1 - f0;
        let segment = dx * (x0 * (f0 + 0.5 * delta) + dx * (0.5 * f0 + delta / 3.0));
        kahan_add(&mut sum, &mut correction, segment);
    }
    sum
}

fn result_digest(result: &SectionalHydrostaticResult) -> Result<String, serde_json::Error> {
    let identity = ResultDigestIdentity {
        algorithm: &result.algorithm,
        source_spec_digest_blake3: &result.source_spec_digest_blake3,
        loading_condition_id: &result.loading_condition_id,
        waterplane_z_m: result.waterplane_z_m,
        water_density_kg_m3: result.water_density_kg_m3,
        displacement_volume_m3: result.displacement_volume_m3,
        displaced_mass_tonnes: result.displaced_mass_tonnes,
        waterplane_area_m2: result.waterplane_area_m2,
        tonnes_per_cm_immersion: result.tonnes_per_cm_immersion,
        longitudinal_center_of_buoyancy_m: result.longitudinal_center_of_buoyancy_m,
        transverse_center_of_buoyancy_m: result.transverse_center_of_buoyancy_m,
        vertical_center_of_buoyancy_m: result.vertical_center_of_buoyancy_m,
        longitudinal_center_of_flotation_m: result.longitudinal_center_of_flotation_m,
    };
    Ok(blake3::hash(&serde_json::to_vec(&identity)?)
        .to_hex()
        .to_string())
}

fn evaluate_one(
    spec: &FullEnvelopeSpec,
    source_digest: &str,
    condition: &LoadingCondition,
) -> Result<SectionalHydrostaticResult, SectionalReferenceIssue> {
    if condition.condition_id.trim().is_empty() {
        return Err(SectionalReferenceIssue {
            code: "loading_condition_identity".into(),
            detail: "loading-condition ID must not be empty".into(),
        });
    }
    if !condition.waterplane_z_m.is_finite() {
        return Err(SectionalReferenceIssue {
            code: "waterplane_non_finite".into(),
            detail: "waterplane elevation must be finite".into(),
        });
    }
    if !condition.water_density_kg_m3.is_finite() || condition.water_density_kg_m3 <= 0.0 {
        return Err(SectionalReferenceIssue {
            code: "water_density".into(),
            detail: "water density must be finite and explicitly greater than zero".into(),
        });
    }

    let mut samples = Vec::with_capacity(spec.stations.len());
    for station_index in 0..spec.stations.len() {
        let sample = station_sample(spec, station_index, condition.waterplane_z_m)?;
        samples.push(sample);
    }
    if samples.len() < 2 {
        return Err(SectionalReferenceIssue {
            code: "station_count".into(),
            detail: "sectional integration requires at least two valid stations".into(),
        });
    }

    let displacement_volume = integrate_linear_curve(&samples, |sample| sample.area_m2);
    let waterplane_area = integrate_linear_curve(&samples, |sample| sample.waterline_breadth_m);
    let lcb_moment = integrate_linear_first_moment(&samples, |sample| sample.area_m2);
    let lcf_moment = integrate_linear_first_moment(&samples, |sample| sample.waterline_breadth_m);
    let transverse_moment =
        integrate_linear_curve(&samples, |sample| sample.area_m2 * sample.centroid_y_m);
    let vertical_moment =
        integrate_linear_curve(&samples, |sample| sample.area_m2 * sample.centroid_z_m);

    if !displacement_volume.is_finite()
        || displacement_volume <= 0.0
        || !waterplane_area.is_finite()
        || waterplane_area <= 0.0
    {
        return Err(SectionalReferenceIssue {
            code: "non_positive_integral".into(),
            detail: "loading condition produced non-positive displacement or waterplane area"
                .into(),
        });
    }

    let density = condition.water_density_kg_m3;
    let mut result = SectionalHydrostaticResult {
        algorithm: SECTIONAL_REFERENCE_ALGORITHM.into(),
        source_spec_digest_blake3: source_digest.into(),
        loading_condition_id: condition.condition_id.clone(),
        waterplane_z_m: condition.waterplane_z_m,
        water_density_kg_m3: density,
        displacement_volume_m3: displacement_volume,
        displaced_mass_tonnes: displacement_volume * density / 1000.0,
        waterplane_area_m2: waterplane_area,
        tonnes_per_cm_immersion: density * waterplane_area / 100_000.0,
        longitudinal_center_of_buoyancy_m: lcb_moment / displacement_volume,
        transverse_center_of_buoyancy_m: transverse_moment / displacement_volume,
        vertical_center_of_buoyancy_m: vertical_moment / displacement_volume,
        longitudinal_center_of_flotation_m: lcf_moment / waterplane_area,
        result_digest_blake3: String::new(),
    };
    result.result_digest_blake3 =
        result_digest(&result).map_err(|error| SectionalReferenceIssue {
            code: "result_digest".into(),
            detail: format!("could not serialize sectional result identity: {error}"),
        })?;
    Ok(result)
}

pub fn compute_loading_sweep_digest(
    receipt: &LoadingSweepReceipt,
) -> Result<String, serde_json::Error> {
    let identity = SweepDigestIdentity {
        receipt_schema_version: receipt.receipt_schema_version,
        algorithm: &receipt.algorithm,
        source_spec_digest_blake3: &receipt.source_spec_digest_blake3,
        results: &receipt.results,
    };
    Ok(blake3::hash(&serde_json::to_vec(&identity)?)
        .to_hex()
        .to_string())
}

/// Evaluate multiple upright horizontal waterplanes from one validated full
/// external hull. Results are independent sectional-area references; they do
/// not create displacement meshes and do not support trim or heel.
pub fn evaluate_loading_sweep(
    request: &SectionalReferenceRequest,
) -> Result<LoadingSweepReceipt, Vec<SectionalReferenceIssue>> {
    let mut issues = Vec::new();
    if request.schema_version != SECTIONAL_REFERENCE_SCHEMA_VERSION {
        push_issue(
            &mut issues,
            "schema_version",
            format!(
                "unsupported loading-sweep schema {}; expected {}",
                request.schema_version, SECTIONAL_REFERENCE_SCHEMA_VERSION
            ),
        );
    }
    if request.loading_conditions.is_empty()
        || request.loading_conditions.len() > MAX_LOADING_CONDITIONS
    {
        push_issue(
            &mut issues,
            "loading_condition_count",
            format!("loading-condition count must be within 1..={MAX_LOADING_CONDITIONS}"),
        );
    }

    let source_validation = validate_full_envelope_spec(&request.source_spec);
    match source_validation.disposition {
        FullEnvelopeDisposition::ValidInputContract => {}
        FullEnvelopeDisposition::Incomplete => push_issue(
            &mut issues,
            "source_spec_incomplete",
            "source full-envelope specification contains unsupported/incomplete features",
        ),
        FullEnvelopeDisposition::Rejected => push_issue(
            &mut issues,
            "source_spec_rejected",
            "source full-envelope specification fails its input contract",
        ),
    }
    if !issues.is_empty() {
        return Err(issues);
    }

    let source_digest = match compute_full_envelope_spec_digest(&request.source_spec) {
        Ok(value) => value,
        Err(error) => {
            push_issue(
                &mut issues,
                "source_spec_digest",
                format!("could not compute source full-envelope digest: {error}"),
            );
            String::new()
        }
    };

    let mut ids = BTreeSet::new();
    let mut previous_plane: Option<f64> = None;
    let mut results = Vec::with_capacity(request.loading_conditions.len());
    for condition in &request.loading_conditions {
        if condition.condition_id.trim().is_empty() || !ids.insert(condition.condition_id.as_str())
        {
            push_issue(
                &mut issues,
                "loading_condition_identity",
                "loading-condition IDs must be non-empty and unique",
            );
            continue;
        }
        if let Some(previous) = previous_plane
            && condition.waterplane_z_m <= previous
        {
            push_issue(
                &mut issues,
                "loading_condition_order",
                "loading conditions must be strictly ordered by ascending waterplane z",
            );
        }
        previous_plane = Some(condition.waterplane_z_m);

        match evaluate_one(&request.source_spec, &source_digest, condition) {
            Ok(result) => results.push(result),
            Err(issue) => issues.push(issue),
        }
    }

    if !issues.is_empty() || results.len() != request.loading_conditions.len() {
        return Err(issues);
    }

    let mut receipt = LoadingSweepReceipt {
        receipt_schema_version: 1,
        algorithm: SECTIONAL_REFERENCE_ALGORITHM.into(),
        source_spec_digest_blake3: source_digest,
        results,
        sweep_digest_blake3: String::new(),
    };
    receipt.sweep_digest_blake3 = compute_loading_sweep_digest(&receipt).map_err(|error| {
        vec![SectionalReferenceIssue {
            code: "sweep_digest".into(),
            detail: format!("could not serialize loading sweep identity: {error}"),
        }]
    })?;
    Ok(receipt)
}

pub fn verify_loading_sweep_receipt(
    request: &SectionalReferenceRequest,
    receipt: &LoadingSweepReceipt,
) -> Result<(), Vec<SectionalReferenceIssue>> {
    let expected = evaluate_loading_sweep(request)?;
    if &expected != receipt {
        return Err(vec![SectionalReferenceIssue {
            code: "loading_sweep_receipt_mismatch".into(),
            detail: "receipt does not match a fresh independent sectional integration of the exact request".into(),
        }]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::full_envelope::{
        DeckPoint, DeckPointRole, DeckSection, FullEnvelopeStation, ProfilePointRole,
        WaterlinePortSource, WaterlineStationIntersection,
    };

    fn fixture_request() -> SectionalReferenceRequest {
        serde_json::from_str(include_str!(
            "../../../../docs/research/marine/fixtures/marine-full-envelope-loading-sweep-001.json"
        ))
        .unwrap()
    }

    fn tapered_fixture_request() -> SectionalReferenceRequest {
        serde_json::from_str(include_str!(
            "../../../../docs/research/marine/fixtures/marine-full-envelope-tapered-loading-sweep-001.json"
        ))
        .unwrap()
    }

    fn tapered_request_with_station_count(station_count: usize) -> SectionalReferenceRequest {
        let mut request = tapered_fixture_request();
        if station_count == 2 {
            return request;
        }
        let first = request.source_spec.stations[0].clone();
        let last = request.source_spec.stations[1].clone();
        let mut stations = Vec::with_capacity(station_count);
        let mut deck_sections = Vec::with_capacity(station_count);
        let mut intersections = Vec::with_capacity(station_count);

        for index in 0..station_count {
            let fraction = index as f64 / (station_count - 1) as f64;
            let station_id = format!("REFINED-{index:03}");
            let mut starboard_profile = first.starboard_profile.clone();
            for (point_index, point) in starboard_profile.iter_mut().enumerate() {
                let a = first.starboard_profile[point_index].y_m;
                let b = last.starboard_profile[point_index].y_m;
                point.y_m = a + fraction * (b - a);
                point.point_id = format!("{station_id}-P{point_index:02}");
            }
            let waterline_point_id = starboard_profile
                .iter()
                .find(|point| point.role == ProfilePointRole::Waterline)
                .unwrap()
                .point_id
                .clone();
            let sheer = starboard_profile.last().unwrap();
            let sheer_point_id = sheer.point_id.clone();
            let deck_edge_z_m = sheer.z_m;
            let half_beam = sheer.y_m;
            let station = FullEnvelopeStation {
                station_id: station_id.clone(),
                x_over_length: fraction,
                starboard_profile,
                port_profile: None,
            };
            deck_sections.push(DeckSection {
                station_id: station_id.clone(),
                points: vec![
                    DeckPoint {
                        point_id: format!("MIRROR:{sheer_point_id}"),
                        y_m: -half_beam,
                        z_m: deck_edge_z_m,
                        role: DeckPointRole::PortDeckEdge,
                    },
                    DeckPoint {
                        point_id: format!("{station_id}-DECK-CROWN"),
                        y_m: 0.0,
                        z_m: -4.5,
                        role: DeckPointRole::DeckCrown,
                    },
                    DeckPoint {
                        point_id: sheer_point_id,
                        y_m: half_beam,
                        z_m: deck_edge_z_m,
                        role: DeckPointRole::StarboardDeckEdge,
                    },
                ],
            });
            intersections.push(WaterlineStationIntersection {
                station_id: station_id.clone(),
                starboard_point_id: waterline_point_id,
                port_source: WaterlinePortSource::MirroredFromStarboard,
                port_point_id: None,
            });
            stations.push(station);
        }
        request.source_spec.stations = stations;
        request.source_spec.deck_surface.sections = deck_sections;
        request.source_spec.design_waterline.station_intersections = intersections;
        request
    }


    fn quadratic_beam_request(station_count: usize) -> SectionalReferenceRequest {
        assert!(station_count >= 2);
        let mut request = fixture_request();
        let mut stations = Vec::with_capacity(station_count);
        let mut deck_sections = Vec::with_capacity(station_count);
        let mut intersections = Vec::with_capacity(station_count);

        for index in 0..station_count {
            let fraction = index as f64 / (station_count - 1) as f64;
            let half_beam_m = 8.0 + 4.0 * fraction * (1.0 - fraction);
            let station_id = format!("QUADRATIC-{index:03}");
            let keel_id = format!("{station_id}-KEEL");
            let bottom_id = format!("{station_id}-BOTTOM");
            let waterline_id = format!("{station_id}-WATERLINE");
            let sheer_id = format!("{station_id}-SHEER");

            let starboard_profile = vec![
                ProfilePoint {
                    point_id: keel_id,
                    y_m: 0.0,
                    z_m: 6.0,
                    role: ProfilePointRole::Keel,
                    region_to_next: crate::full_envelope::ProfileRegion::Bottom,
                },
                ProfilePoint {
                    point_id: bottom_id,
                    y_m: half_beam_m,
                    z_m: 6.0,
                    role: ProfilePointRole::Bottom,
                    region_to_next: crate::full_envelope::ProfileRegion::Bottom,
                },
                ProfilePoint {
                    point_id: waterline_id.clone(),
                    y_m: half_beam_m,
                    z_m: 0.0,
                    role: ProfilePointRole::Waterline,
                    region_to_next: crate::full_envelope::ProfileRegion::Topsides,
                },
                ProfilePoint {
                    point_id: sheer_id.clone(),
                    y_m: half_beam_m,
                    z_m: -4.0,
                    role: ProfilePointRole::Sheer,
                    region_to_next: crate::full_envelope::ProfileRegion::None,
                },
            ];
            stations.push(FullEnvelopeStation {
                station_id: station_id.clone(),
                x_over_length: fraction,
                starboard_profile,
                port_profile: None,
            });
            deck_sections.push(DeckSection {
                station_id: station_id.clone(),
                points: vec![
                    DeckPoint {
                        point_id: format!("MIRROR:{sheer_id}"),
                        y_m: -half_beam_m,
                        z_m: -4.0,
                        role: DeckPointRole::PortDeckEdge,
                    },
                    DeckPoint {
                        point_id: format!("{station_id}-DECK-CROWN"),
                        y_m: 0.0,
                        z_m: -4.5,
                        role: DeckPointRole::DeckCrown,
                    },
                    DeckPoint {
                        point_id: sheer_id,
                        y_m: half_beam_m,
                        z_m: -4.0,
                        role: DeckPointRole::StarboardDeckEdge,
                    },
                ],
            });
            intersections.push(WaterlineStationIntersection {
                station_id,
                starboard_point_id: waterline_id,
                port_source: WaterlinePortSource::MirroredFromStarboard,
                port_point_id: None,
            });
        }

        request.source_spec.candidate_id = "quadratic-half-beam-known-answer".into();
        request.source_spec.principal_dimensions.maximum_beam_m = 20.0;
        request.source_spec.stations = stations;
        request.source_spec.deck_surface.sections = deck_sections;
        request.source_spec.design_waterline.station_intersections = intersections;
        request.loading_conditions = vec![LoadingCondition {
            condition_id: "QUADRATIC-DESIGN-Z0".into(),
            waterplane_z_m: 0.0,
            water_density_kg_m3: 1025.0,
        }];
        request
    }

    fn result_by_id<'a>(
        receipt: &'a LoadingSweepReceipt,
        id: &str,
    ) -> &'a SectionalHydrostaticResult {
        receipt
            .results
            .iter()
            .find(|result| result.loading_condition_id == id)
            .unwrap()
    }

    #[test]
    fn box_fixture_matches_three_analytic_loading_conditions() {
        let receipt = evaluate_loading_sweep(&fixture_request()).unwrap();
        let overload = result_by_id(&receipt, "OVERLOAD-Z-NEG2");
        let design = result_by_id(&receipt, "DESIGN-WL-Z0");
        let light = result_by_id(&receipt, "LIGHT-Z-POS2");

        assert!((overload.displacement_volume_m3 - 16_000.0).abs() < 1.0e-9);
        assert!((design.displacement_volume_m3 - 12_000.0).abs() < 1.0e-9);
        assert!((light.displacement_volume_m3 - 8_000.0).abs() < 1.0e-9);
        for result in [overload, design, light] {
            assert!((result.waterplane_area_m2 - 2_000.0).abs() < 1.0e-9);
            assert!((result.longitudinal_center_of_buoyancy_m - 50.0).abs() < 1.0e-9);
            assert!((result.longitudinal_center_of_flotation_m - 50.0).abs() < 1.0e-9);
            assert!(result.transverse_center_of_buoyancy_m.abs() < 1.0e-9);
            assert!((result.tonnes_per_cm_immersion - 20.5).abs() < 1.0e-9);
        }
        assert!((overload.vertical_center_of_buoyancy_m - 2.0).abs() < 1.0e-9);
        assert!((design.vertical_center_of_buoyancy_m - 3.0).abs() < 1.0e-9);
        assert!((light.vertical_center_of_buoyancy_m - 4.0).abs() < 1.0e-9);
        assert!((design.displaced_mass_tonnes - 12_300.0).abs() < 1.0e-9);
    }

    #[test]
    fn linear_taper_fixture_matches_analytic_loading_curves() {
        let receipt = evaluate_loading_sweep(&tapered_fixture_request()).unwrap();
        let overload = result_by_id(&receipt, "TAPER-OVERLOAD-Z-NEG2");
        let design = result_by_id(&receipt, "TAPER-DESIGN-Z0");
        let light = result_by_id(&receipt, "TAPER-LIGHT-Z-POS2");
        let expected_lcb = 500_000.0 / 9_000.0;

        assert!((overload.displacement_volume_m3 - 12_000.0).abs() < 1.0e-9);
        assert!((design.displacement_volume_m3 - 9_000.0).abs() < 1.0e-9);
        assert!((light.displacement_volume_m3 - 6_000.0).abs() < 1.0e-9);
        for result in [overload, design, light] {
            assert!((result.waterplane_area_m2 - 1_500.0).abs() < 1.0e-9);
            assert!((result.longitudinal_center_of_buoyancy_m - expected_lcb).abs() < 1.0e-9);
            assert!((result.longitudinal_center_of_flotation_m - expected_lcb).abs() < 1.0e-9);
            assert!(result.transverse_center_of_buoyancy_m.abs() < 1.0e-9);
            assert!((result.tonnes_per_cm_immersion - 15.375).abs() < 1.0e-9);
        }
        assert!((overload.vertical_center_of_buoyancy_m - 2.0).abs() < 1.0e-9);
        assert!((design.vertical_center_of_buoyancy_m - 3.0).abs() < 1.0e-9);
        assert!((light.vertical_center_of_buoyancy_m - 4.0).abs() < 1.0e-9);
        assert!((design.displaced_mass_tonnes - 9_225.0).abs() < 1.0e-9);
    }

    #[test]
    fn linear_taper_loading_curve_is_invariant_to_station_refinement() {
        let two = evaluate_loading_sweep(&tapered_request_with_station_count(2)).unwrap();
        let three = evaluate_loading_sweep(&tapered_request_with_station_count(3)).unwrap();
        let five = evaluate_loading_sweep(&tapered_request_with_station_count(5)).unwrap();
        assert_ne!(
            two.source_spec_digest_blake3,
            five.source_spec_digest_blake3
        );

        for condition_id in [
            "TAPER-OVERLOAD-Z-NEG2",
            "TAPER-DESIGN-Z0",
            "TAPER-LIGHT-Z-POS2",
        ] {
            let a = result_by_id(&two, condition_id);
            let b = result_by_id(&three, condition_id);
            let c = result_by_id(&five, condition_id);
            for (left, middle, right) in [
                (
                    a.displacement_volume_m3,
                    b.displacement_volume_m3,
                    c.displacement_volume_m3,
                ),
                (
                    a.waterplane_area_m2,
                    b.waterplane_area_m2,
                    c.waterplane_area_m2,
                ),
                (
                    a.longitudinal_center_of_buoyancy_m,
                    b.longitudinal_center_of_buoyancy_m,
                    c.longitudinal_center_of_buoyancy_m,
                ),
                (
                    a.vertical_center_of_buoyancy_m,
                    b.vertical_center_of_buoyancy_m,
                    c.vertical_center_of_buoyancy_m,
                ),
            ] {
                assert!((left - middle).abs() < 1.0e-9);
                assert!((left - right).abs() < 1.0e-9);
            }
        }
    }


    #[test]
    fn quadratic_beam_loading_converges_at_second_order_for_displacement() {
        let coarse = evaluate_loading_sweep(&quadratic_beam_request(3)).unwrap();
        let medium = evaluate_loading_sweep(&quadratic_beam_request(5)).unwrap();
        let fine = evaluate_loading_sweep(&quadratic_beam_request(9)).unwrap();
        let exact_volume_m3 = 10_400.0;

        let coarse_error = (result_by_id(&coarse, "QUADRATIC-DESIGN-Z0").displacement_volume_m3
            - exact_volume_m3)
            .abs();
        let medium_error = (result_by_id(&medium, "QUADRATIC-DESIGN-Z0").displacement_volume_m3
            - exact_volume_m3)
            .abs();
        let fine_error = (result_by_id(&fine, "QUADRATIC-DESIGN-Z0").displacement_volume_m3
            - exact_volume_m3)
            .abs();

        assert!((coarse_error - 200.0).abs() < 1.0e-9);
        assert!((medium_error - 50.0).abs() < 1.0e-9);
        assert!((fine_error - 12.5).abs() < 1.0e-9);
        assert!((coarse_error / medium_error - 4.0).abs() < 1.0e-9);
        assert!((medium_error / fine_error - 4.0).abs() < 1.0e-9);
    }

    #[test]
    fn quadratic_beam_loading_converges_at_second_order_for_waterplane_area() {
        let coarse = evaluate_loading_sweep(&quadratic_beam_request(3)).unwrap();
        let medium = evaluate_loading_sweep(&quadratic_beam_request(5)).unwrap();
        let fine = evaluate_loading_sweep(&quadratic_beam_request(9)).unwrap();
        let exact_area_m2 = 5_200.0 / 3.0;

        let coarse_error = (result_by_id(&coarse, "QUADRATIC-DESIGN-Z0").waterplane_area_m2
            - exact_area_m2)
            .abs();
        let medium_error = (result_by_id(&medium, "QUADRATIC-DESIGN-Z0").waterplane_area_m2
            - exact_area_m2)
            .abs();
        let fine_error = (result_by_id(&fine, "QUADRATIC-DESIGN-Z0").waterplane_area_m2
            - exact_area_m2)
            .abs();

        assert!((coarse_error - 100.0 / 3.0).abs() < 1.0e-9);
        assert!((medium_error - 25.0 / 3.0).abs() < 1.0e-9);
        assert!((fine_error - 25.0 / 12.0).abs() < 1.0e-9);
        assert!((coarse_error / medium_error - 4.0).abs() < 1.0e-9);
        assert!((medium_error / fine_error - 4.0).abs() < 1.0e-9);
    }

    #[test]
    fn loading_sweep_digest_is_deterministic_and_verifiable() {
        let request = fixture_request();
        let first = evaluate_loading_sweep(&request).unwrap();
        let second = evaluate_loading_sweep(&request).unwrap();
        assert_eq!(first, second);
        verify_loading_sweep_receipt(&request, &first).unwrap();
        let mut tampered = first;
        tampered.results[0].displacement_volume_m3 += 1.0;
        assert!(verify_loading_sweep_receipt(&request, &tampered).is_err());
    }

    #[test]
    fn conditions_must_be_sorted_by_waterplane_elevation() {
        let mut request = fixture_request();
        request.loading_conditions.swap(0, 2);
        let errors = evaluate_loading_sweep(&request).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|issue| issue.code == "loading_condition_order")
        );
    }

    #[test]
    fn zero_density_is_rejected_without_a_default() {
        let mut request = fixture_request();
        request.loading_conditions[1].water_density_kg_m3 = 0.0;
        let errors = evaluate_loading_sweep(&request).unwrap_err();
        assert!(errors.iter().any(|issue| issue.code == "water_density"));
    }

    #[test]
    fn waterplane_above_sheer_is_rejected_instead_of_filling_the_deck() {
        let mut request = fixture_request();
        request.loading_conditions[0].waterplane_z_m = -4.25;
        request.loading_conditions[1].waterplane_z_m = 0.0;
        request.loading_conditions[2].waterplane_z_m = 2.0;
        let errors = evaluate_loading_sweep(&request).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|issue| issue.code == "waterplane_outside_supported_section")
        );
    }

    #[test]
    fn invalid_source_geometry_is_rejected_without_indexing_missing_stations() {
        let mut request = fixture_request();
        request.source_spec.stations.clear();
        let errors = evaluate_loading_sweep(&request).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|issue| issue.code == "source_spec_rejected")
        );
    }

    #[test]
    fn explicit_asymmetric_port_side_uses_its_real_profile() {
        let mut request = fixture_request();
        request.source_spec.symmetry = SymmetryPolicy::ExplicitPortAndStarboard;
        for station in &mut request.source_spec.stations {
            let mut port = station.starboard_profile.clone();
            for point in &mut port {
                if point.role != ProfilePointRole::Keel {
                    point.point_id = format!("PORT-{}", point.point_id);
                }
                point.y_m = -point.y_m * 0.9;
            }
            station.port_profile = Some(port);
        }
        for intersection in &mut request.source_spec.design_waterline.station_intersections {
            intersection.port_source =
                crate::full_envelope::WaterlinePortSource::ExplicitPortProfile;
            intersection.port_point_id = Some(format!("PORT-{}", intersection.starboard_point_id));
        }
        for (station, deck_section) in request
            .source_spec
            .stations
            .iter()
            .zip(request.source_spec.deck_surface.sections.iter_mut())
        {
            let port_sheer = station.port_profile.as_ref().unwrap().last().unwrap();
            deck_section.points[0].point_id = port_sheer.point_id.clone();
            deck_section.points[0].y_m = port_sheer.y_m;
        }
        let receipt = evaluate_loading_sweep(&request).unwrap();
        let design = result_by_id(&receipt, "DESIGN-WL-Z0");
        assert!((design.displacement_volume_m3 - 11_400.0).abs() < 1.0e-8);
        assert!((design.transverse_center_of_buoyancy_m - 0.5).abs() < 1.0e-8);
    }

    #[test]
    fn duplicate_loading_condition_ids_are_rejected() {
        let mut request = fixture_request();
        request.loading_conditions[2].condition_id =
            request.loading_conditions[1].condition_id.clone();
        let errors = evaluate_loading_sweep(&request).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|issue| issue.code == "loading_condition_identity")
        );
    }
}
