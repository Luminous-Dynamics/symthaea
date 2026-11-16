//! A0/A1 static-attitude waterplane clipping and sectional force/moment evaluation.
//!
//! The body frame is right-handed: x-forward, y-starboard, z-down. Pitch and
//! roll transform world-down gravity into body coordinates. A horizontal
//! world waterplane is therefore oblique in the body frame. The implementation
//! clips each complete section polygon (including the declared deck profile)
//! against that plane and integrates sectional area and first moments along x.
//!
//! This module evaluates one trial attitude. It does not solve the equilibrium
//! root, calculate GZ/GM, or approve intact/damage stability.

use serde::{Deserialize, Serialize};

use crate::full_envelope::{
    FullEnvelopeDisposition, FullEnvelopeSpec, SymmetryPolicy,
    compute_full_envelope_spec_digest, validate_full_envelope_spec,
};

pub const STATIC_ATTITUDE_EVALUATION_SCHEMA_VERSION: u32 = 1;
pub const STATIC_ATTITUDE_EVALUATION_ALGORITHM: &str =
    "oblique-waterplane-sectional-cut-trapezoid-v1";
pub const MAX_STATIC_ATTITUDE_ABS_RAD: f64 = 0.523_598_775_598_298_8;
const MIN_GEOMETRY_TOLERANCE_M: f64 = 1.0e-12;
const MAX_GEOMETRY_TOLERANCE_M: f64 = 1.0e-5;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BodyPoint3 {
    pub x_m: f64,
    pub y_m: f64,
    pub z_m: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticAttitude {
    /// Body-frame offset in the plane equation g_body · r_body = offset_m.
    /// Positive is deeper for the upright z-down case.
    pub waterplane_offset_m: f64,
    /// Positive pitch raises the bow in the declared right-handed body frame.
    pub pitch_rad: f64,
    /// Positive roll lowers the starboard side in the declared body frame.
    pub roll_rad: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticAttitudeEvaluationRequest {
    pub schema_version: u32,
    pub source_spec: FullEnvelopeSpec,
    pub load_case_id: String,
    pub total_mass_tonnes: f64,
    pub water_density_kg_m3: f64,
    pub gravity_m_s2: f64,
    pub center_of_gravity_body_m: BodyPoint3,
    pub attitude: StaticAttitude,
    /// Explicit geometric tolerance used for clipping and section predicates.
    pub geometry_tolerance_m: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StaticAttitudeIssue {
    pub code: String,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BodyGravityDirection {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StaticAttitudeResiduals {
    /// Displaced mass minus declared mass (tonnes).
    pub mass_residual_tonnes: f64,
    /// Two independent components of buoyancy moment about the CG (N m).
    /// The basis is orthonormal and perpendicular to body-frame gravity.
    pub moment_residual_1_nm: f64,
    pub moment_residual_2_nm: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StaticAttitudeEvaluationReceipt {
    pub receipt_schema_version: u32,
    pub algorithm: String,
    pub source_spec_digest_blake3: String,
    pub load_case_id: String,
    pub total_mass_tonnes: f64,
    pub water_density_kg_m3: f64,
    pub gravity_m_s2: f64,
    pub center_of_gravity_body_m: BodyPoint3,
    pub attitude: StaticAttitude,
    pub geometry_tolerance_m: f64,
    pub body_gravity_direction: BodyGravityDirection,
    pub displacement_volume_m3: f64,
    pub displaced_mass_tonnes: f64,
    pub center_of_buoyancy_body_m: BodyPoint3,
    pub residuals: StaticAttitudeResiduals,
    pub station_count: usize,
    pub receipt_digest_blake3: String,
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
    first_moment_y_m3: f64,
    first_moment_z_m3: f64,
}

#[derive(Debug, Clone, Copy)]
struct PolygonProperties {
    area_m2: f64,
    centroid_y_m: f64,
    centroid_z_m: f64,
}

#[derive(Serialize)]
struct ReceiptDigestIdentity<'a> {
    receipt_schema_version: u32,
    algorithm: &'a str,
    source_spec_digest_blake3: &'a str,
    load_case_id: &'a str,
    total_mass_tonnes: f64,
    water_density_kg_m3: f64,
    gravity_m_s2: f64,
    center_of_gravity_body_m: BodyPoint3,
    attitude: StaticAttitude,
    geometry_tolerance_m: f64,
    body_gravity_direction: BodyGravityDirection,
    displacement_volume_m3: f64,
    displaced_mass_tonnes: f64,
    center_of_buoyancy_body_m: BodyPoint3,
    residuals: StaticAttitudeResiduals,
    station_count: usize,
}

fn push_issue(issues: &mut Vec<StaticAttitudeIssue>, code: &str, detail: impl Into<String>) {
    issues.push(StaticAttitudeIssue {
        code: code.to_owned(),
        detail: detail.into(),
    });
}

fn finite_point(point: BodyPoint3) -> bool {
    point.x_m.is_finite() && point.y_m.is_finite() && point.z_m.is_finite()
}

fn dot2(point: Point2, gy: f64, gz: f64) -> f64 {
    gy * point.y + gz * point.z
}

fn cross3(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn dot3(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn norm3(value: [f64; 3]) -> f64 {
    dot3(value, value).sqrt()
}

fn normalize3(value: [f64; 3]) -> Option<[f64; 3]> {
    let norm = norm3(value);
    (norm.is_finite() && norm > f64::EPSILON).then(|| {
        [value[0] / norm, value[1] / norm, value[2] / norm]
    })
}

fn body_gravity_direction(attitude: StaticAttitude) -> BodyGravityDirection {
    let (sin_pitch, cos_pitch) = attitude.pitch_rad.sin_cos();
    let (sin_roll, cos_roll) = attitude.roll_rad.sin_cos();

    // R_body_to_world = R_y(pitch) * R_x(roll), so
    // R^T * (0, 0, 1) = (-sin(pitch), sin(roll)cos(pitch), cos(roll)cos(pitch)).
    let gravity = [
        -sin_pitch,
        sin_roll * cos_pitch,
        cos_roll * cos_pitch,
    ];
    let normalized = normalize3(gravity).unwrap_or([0.0, 0.0, 1.0]);
    BodyGravityDirection {
        x: normalized[0],
        y: normalized[1],
        z: normalized[2],
    }
}

fn deduplicate_polygon(points: Vec<Point2>, tolerance: f64) -> Vec<Point2> {
    let mut cleaned = Vec::with_capacity(points.len());
    for point in points {
        if cleaned.last().is_none_or(|last: &Point2| {
            (last.y - point.y).abs() > tolerance || (last.z - point.z).abs() > tolerance
        }) {
            cleaned.push(point);
        }
    }
    if cleaned.len() > 1 {
        let first = cleaned[0];
        let last = *cleaned.last().unwrap();
        if (first.y - last.y).abs() <= tolerance && (first.z - last.z).abs() <= tolerance {
            cleaned.pop();
        }
    }
    cleaned
}

fn section_polygon(
    spec: &FullEnvelopeSpec,
    station_index: usize,
    tolerance: f64,
) -> Vec<Point2> {
    let station = &spec.stations[station_index];
    let deck = &spec.deck_surface.sections[station_index];
    let mut polygon = Vec::with_capacity(
        station.starboard_profile.len()
            + deck.points.len()
            + station
                .port_profile
                .as_ref()
                .map_or(station.starboard_profile.len(), Vec::len),
    );

    polygon.extend(station.starboard_profile.iter().map(|point| Point2 {
        y: point.y_m,
        z: point.z_m,
    }));

    // The starboard sheer is already present. Walk across the actual deck
    // profile to the port sheer; never substitute a straight sheer-to-sheer
    // chord when the oblique waterplane enters the deck/freeboard region.
    polygon.extend(deck.points.iter().rev().skip(1).map(|point| Point2 {
        y: point.y_m,
        z: point.z_m,
    }));

    match spec.symmetry {
        SymmetryPolicy::SymmetricAboutY0 => {
            polygon.extend(
                station
                    .starboard_profile
                    .iter()
                    .rev()
                    .skip(1)
                    .map(|point| Point2 {
                        y: -point.y_m,
                        z: point.z_m,
                    }),
            );
        }
        SymmetryPolicy::ExplicitPortAndStarboard => {
            if let Some(port_profile) = station.port_profile.as_ref() {
                polygon.extend(port_profile.iter().rev().skip(1).map(|point| Point2 {
                    y: point.y_m,
                    z: point.z_m,
                }));
            }
        }
    }

    deduplicate_polygon(polygon, tolerance)
}

fn orientation(a: Point2, b: Point2, c: Point2) -> f64 {
    (b.y - a.y) * (c.z - a.z) - (b.z - a.z) * (c.y - a.y)
}

fn point_on_segment(
    point: Point2,
    a: Point2,
    b: Point2,
    coordinate_tolerance: f64,
    area_tolerance: f64,
) -> bool {
    orientation(a, b, point).abs() <= area_tolerance
        && point.y >= a.y.min(b.y) - coordinate_tolerance
        && point.y <= a.y.max(b.y) + coordinate_tolerance
        && point.z >= a.z.min(b.z) - coordinate_tolerance
        && point.z <= a.z.max(b.z) + coordinate_tolerance
}

fn segments_intersect(
    a: Point2,
    b: Point2,
    c: Point2,
    d: Point2,
    coordinate_tolerance: f64,
    area_tolerance: f64,
) -> bool {
    let o1 = orientation(a, b, c);
    let o2 = orientation(a, b, d);
    let o3 = orientation(c, d, a);
    let o4 = orientation(c, d, b);

    if ((o1 > area_tolerance && o2 < -area_tolerance)
        || (o1 < -area_tolerance && o2 > area_tolerance))
        && ((o3 > area_tolerance && o4 < -area_tolerance)
            || (o3 < -area_tolerance && o4 > area_tolerance))
    {
        return true;
    }
    (o1.abs() <= area_tolerance
        && point_on_segment(c, a, b, coordinate_tolerance, area_tolerance))
        || (o2.abs() <= area_tolerance
            && point_on_segment(d, a, b, coordinate_tolerance, area_tolerance))
        || (o3.abs() <= area_tolerance
            && point_on_segment(a, c, d, coordinate_tolerance, area_tolerance))
        || (o4.abs() <= area_tolerance
            && point_on_segment(b, c, d, coordinate_tolerance, area_tolerance))
}

fn polygon_is_simple(points: &[Point2], tolerance: f64) -> bool {
    if points.len() < 3 {
        return true;
    }
    let scale = points
        .iter()
        .map(|point| point.y.abs().max(point.z.abs()))
        .fold(1.0_f64, f64::max);
    let area_tolerance = tolerance * scale;
    let edge_count = points.len();
    for first in 0..edge_count {
        let a = points[first];
        let b = points[(first + 1) % edge_count];
        if (a.y - b.y).abs() <= tolerance && (a.z - b.z).abs() <= tolerance {
            continue;
        }
        for second in first + 1..edge_count {
            if second == first + 1 || (first == 0 && second + 1 == edge_count) {
                continue;
            }
            let c = points[second];
            let d = points[(second + 1) % edge_count];
            if segments_intersect(a, b, c, d, tolerance, area_tolerance) {
                return false;
            }
        }
    }
    true
}

fn clip_section_to_oblique_waterplane(
    points: &[Point2],
    gy: f64,
    gz: f64,
    section_plane_offset_m: f64,
    tolerance: f64,
) -> Vec<Point2> {
    if points.len() < 3 {
        return Vec::new();
    }
    let mut output = Vec::with_capacity(points.len() + 2);
    let mut previous = points[points.len() - 1];
    let mut previous_signed = dot2(previous, gy, gz) - section_plane_offset_m;
    let mut previous_inside = previous_signed >= 0.0;

    for current in points {
        let current_signed = dot2(*current, gy, gz) - section_plane_offset_m;
        let current_inside = current_signed >= 0.0;
        match (previous_inside, current_inside) {
            (true, true) => output.push(*current),
            (true, false) => {
                let denominator = previous_signed - current_signed;
                let fraction = if denominator.abs() <= f64::EPSILON {
                    0.0
                } else {
                    previous_signed / denominator
                };
                output.push(Point2 {
                    y: previous.y + fraction * (current.y - previous.y),
                    z: previous.z + fraction * (current.z - previous.z),
                });
            }
            (false, true) => {
                let denominator = previous_signed - current_signed;
                let fraction = if denominator.abs() <= f64::EPSILON {
                    0.0
                } else {
                    previous_signed / denominator
                };
                output.push(Point2 {
                    y: previous.y + fraction * (current.y - previous.y),
                    z: previous.z + fraction * (current.z - previous.z),
                });
                output.push(*current);
            }
            (false, false) => {}
        }
        previous = *current;
        previous_signed = current_signed;
    }

    deduplicate_polygon(output, tolerance)
}

fn polygon_properties(points: &[Point2], tolerance: f64) -> Option<PolygonProperties> {
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

    if !twice_area.is_finite() || twice_area.abs() <= tolerance * tolerance {
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

fn kahan_add(sum: &mut f64, correction: &mut f64, value: f64) {
    let adjusted = value - *correction;
    let next = *sum + adjusted;
    *correction = (next - *sum) - adjusted;
    *sum = next;
}

fn integrate_linear_curve(samples: &[SectionSample], value: impl Fn(&SectionSample) -> f64) -> f64 {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for pair in samples.windows(2) {
        let dx = pair[1].x_m - pair[0].x_m;
        kahan_add(
            &mut sum,
            &mut correction,
            dx * 0.5 * (value(&pair[0]) + value(&pair[1])),
        );
    }
    sum
}

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

fn moment_basis(gravity: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let reference = if gravity[2].abs() < 0.9 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let first = normalize3(cross3(gravity, reference)).unwrap_or([1.0, 0.0, 0.0]);
    let second = normalize3(cross3(gravity, first)).unwrap_or([0.0, 1.0, 0.0]);
    (first, second)
}

fn validate_request(request: &StaticAttitudeEvaluationRequest) -> Vec<StaticAttitudeIssue> {
    let mut issues = Vec::new();
    if request.schema_version != STATIC_ATTITUDE_EVALUATION_SCHEMA_VERSION {
        push_issue(
            &mut issues,
            "schema_version",
            format!(
                "unsupported static-attitude evaluation schema {}; expected {}",
                request.schema_version, STATIC_ATTITUDE_EVALUATION_SCHEMA_VERSION
            ),
        );
    }
    if request.load_case_id.trim().is_empty() {
        push_issue(&mut issues, "load_case_id", "load-case ID must not be empty");
    }
    if !request.total_mass_tonnes.is_finite() || request.total_mass_tonnes <= 0.0 {
        push_issue(
            &mut issues,
            "total_mass",
            "total vessel mass must be finite and greater than zero",
        );
    }
    if !request.water_density_kg_m3.is_finite() || request.water_density_kg_m3 <= 0.0 {
        push_issue(
            &mut issues,
            "water_density",
            "water density must be finite and explicitly greater than zero",
        );
    }
    if !request.gravity_m_s2.is_finite() || request.gravity_m_s2 <= 0.0 {
        push_issue(
            &mut issues,
            "gravity",
            "gravity magnitude must be finite and explicitly greater than zero",
        );
    }
    if !finite_point(request.center_of_gravity_body_m) {
        push_issue(
            &mut issues,
            "center_of_gravity",
            "all centre-of-gravity coordinates must be finite",
        );
    }
    let attitude = request.attitude;
    if !attitude.waterplane_offset_m.is_finite()
        || !attitude.pitch_rad.is_finite()
        || !attitude.roll_rad.is_finite()
    {
        push_issue(
            &mut issues,
            "attitude_non_finite",
            "waterplane offset, pitch and roll must be finite",
        );
    }
    if attitude.pitch_rad.abs() > MAX_STATIC_ATTITUDE_ABS_RAD
        || attitude.roll_rad.abs() > MAX_STATIC_ATTITUDE_ABS_RAD
    {
        push_issue(
            &mut issues,
            "attitude_domain",
            format!(
                "A0/A1 currently supports absolute pitch and roll no greater than {:.6} rad",
                MAX_STATIC_ATTITUDE_ABS_RAD
            ),
        );
    }
    if !request.geometry_tolerance_m.is_finite()
        || !(MIN_GEOMETRY_TOLERANCE_M..=MAX_GEOMETRY_TOLERANCE_M)
            .contains(&request.geometry_tolerance_m)
    {
        push_issue(
            &mut issues,
            "geometry_tolerance",
            format!(
                "geometry tolerance must be within {MIN_GEOMETRY_TOLERANCE_M:e}..={MAX_GEOMETRY_TOLERANCE_M:e} metres"
            ),
        );
    }
    issues
}

pub fn compute_static_attitude_receipt_digest(
    receipt: &StaticAttitudeEvaluationReceipt,
) -> Result<String, serde_json::Error> {
    let identity = ReceiptDigestIdentity {
        receipt_schema_version: receipt.receipt_schema_version,
        algorithm: &receipt.algorithm,
        source_spec_digest_blake3: &receipt.source_spec_digest_blake3,
        load_case_id: &receipt.load_case_id,
        total_mass_tonnes: receipt.total_mass_tonnes,
        water_density_kg_m3: receipt.water_density_kg_m3,
        gravity_m_s2: receipt.gravity_m_s2,
        center_of_gravity_body_m: receipt.center_of_gravity_body_m,
        attitude: receipt.attitude,
        geometry_tolerance_m: receipt.geometry_tolerance_m,
        body_gravity_direction: receipt.body_gravity_direction,
        displacement_volume_m3: receipt.displacement_volume_m3,
        displaced_mass_tonnes: receipt.displaced_mass_tonnes,
        center_of_buoyancy_body_m: receipt.center_of_buoyancy_body_m,
        residuals: receipt.residuals,
        station_count: receipt.station_count,
    };
    Ok(blake3::hash(&serde_json::to_vec(&identity)?)
        .to_hex()
        .to_string())
}

/// Evaluate one upright or tilted calm-water attitude.
///
/// The accepted source must be a valid G2f full-envelope profile. Every
/// station's complete section is clipped by the oblique world-horizontal
/// waterplane expressed in body coordinates. The returned moments are the
/// physical buoyancy moment about the supplied CG; zero residuals are
/// necessary equilibrium conditions but do not establish stability.
pub fn evaluate_static_attitude(
    request: &StaticAttitudeEvaluationRequest,
) -> Result<StaticAttitudeEvaluationReceipt, Vec<StaticAttitudeIssue>> {
    let mut issues = validate_request(request);
    let source_report = validate_full_envelope_spec(&request.source_spec);
    match source_report.disposition {
        FullEnvelopeDisposition::ValidInputContract => {}
        FullEnvelopeDisposition::Incomplete => push_issue(
            &mut issues,
            "source_spec_incomplete",
            "full-envelope source explicitly requires unsupported geometry/boundary handling",
        ),
        FullEnvelopeDisposition::Rejected => push_issue(
            &mut issues,
            "source_spec_rejected",
            "full-envelope source fails its typed semantic contract",
        ),
    }
    if !issues.is_empty() {
        return Err(issues);
    }

    let source_digest = compute_full_envelope_spec_digest(&request.source_spec).map_err(|error| {
        vec![StaticAttitudeIssue {
            code: "source_spec_digest".into(),
            detail: format!("could not serialize full-envelope source identity: {error}"),
        }]
    })?;
    let gravity = body_gravity_direction(request.attitude);
    let gravity_vector = [gravity.x, gravity.y, gravity.z];
    if gravity.z <= 0.0 {
        return Err(vec![StaticAttitudeIssue {
            code: "body_gravity_orientation".into(),
            detail: "supported A0/A1 attitude must retain positive body-frame down component".into(),
        }]);
    }

    let tolerance = request.geometry_tolerance_m;
    let mut samples = Vec::with_capacity(request.source_spec.stations.len());
    for (station_index, station) in request.source_spec.stations.iter().enumerate() {
        let x_m = station.x_over_length * request.source_spec.principal_dimensions.length_m;
        let section = section_polygon(&request.source_spec, station_index, tolerance);
        if section.len() < 3 {
            return Err(vec![StaticAttitudeIssue {
                code: "section_polygon_degenerate".into(),
                detail: format!(
                    "station {} does not define a complete section polygon",
                    station.station_id
                ),
            }]);
        }
        let is_pointed_endpoint = (station_index == 0
            && request.source_spec.end_conditions.bow
                == crate::full_envelope::HullEndCondition::PointedStem)
            || (station_index + 1 == request.source_spec.stations.len()
                && request.source_spec.end_conditions.stern
                    == crate::full_envelope::HullEndCondition::PointedStem);
        if !is_pointed_endpoint && !polygon_is_simple(&section, tolerance) {
            return Err(vec![StaticAttitudeIssue {
                code: "section_self_intersection".into(),
                detail: format!(
                    "station {} full section polygon self-intersects",
                    station.station_id
                ),
            }]);
        }

        let section_plane_offset = request.attitude.waterplane_offset_m - gravity.x * x_m;
        let signed_values = section
            .iter()
            .map(|point| dot2(*point, gravity.y, gravity.z))
            .collect::<Vec<_>>();
        let min_value = signed_values.iter().copied().reduce(f64::min).unwrap_or(f64::NAN);
        let max_value = signed_values.iter().copied().reduce(f64::max).unwrap_or(f64::NAN);
        if section_plane_offset < min_value - tolerance
            || section_plane_offset > max_value + tolerance
        {
            return Err(vec![StaticAttitudeIssue {
                code: "waterplane_outside_section".into(),
                detail: format!(
                    "oblique waterplane does not intersect station {} section (station-plane offset {:.9} m outside [{:.9}, {:.9}] m)",
                    station.station_id, section_plane_offset, min_value, max_value
                ),
            }]);
        }

        let clipped = clip_section_to_oblique_waterplane(
            &section,
            gravity.y,
            gravity.z,
            section_plane_offset,
            tolerance,
        );
        let properties = polygon_properties(&clipped, tolerance);
        let (area_m2, first_moment_y_m3, first_moment_z_m3) = match properties {
            Some(item) => (
                item.area_m2,
                item.area_m2 * item.centroid_y_m,
                item.area_m2 * item.centroid_z_m,
            ),
            None => (0.0, 0.0, 0.0),
        };
        if !area_m2.is_finite()
            || !first_moment_y_m3.is_finite()
            || !first_moment_z_m3.is_finite()
        {
            return Err(vec![StaticAttitudeIssue {
                code: "section_non_finite_property".into(),
                detail: format!(
                    "station {} produced a non-finite clipped section",
                    station.station_id
                ),
            }]);
        }
        samples.push(SectionSample {
            x_m,
            area_m2,
            first_moment_y_m3,
            first_moment_z_m3,
        });
    }

    let displacement_volume_m3 = integrate_linear_curve(&samples, |sample| sample.area_m2);
    if !displacement_volume_m3.is_finite() || displacement_volume_m3 <= 0.0 {
        return Err(vec![StaticAttitudeIssue {
            code: "non_positive_displacement".into(),
            detail: "trial attitude produced non-positive displacement volume".into(),
        }]);
    }
    let x_moment = integrate_linear_first_moment(&samples, |sample| sample.area_m2);
    let y_moment = integrate_linear_curve(&samples, |sample| sample.first_moment_y_m3);
    let z_moment = integrate_linear_curve(&samples, |sample| sample.first_moment_z_m3);
    let center_of_buoyancy = BodyPoint3 {
        x_m: x_moment / displacement_volume_m3,
        y_m: y_moment / displacement_volume_m3,
        z_m: z_moment / displacement_volume_m3,
    };
    if !finite_point(center_of_buoyancy) {
        return Err(vec![StaticAttitudeIssue {
            code: "center_of_buoyancy_non_finite".into(),
            detail: "sectional integration produced a non-finite centre of buoyancy".into(),
        }]);
    }

    let displaced_mass_tonnes =
        displacement_volume_m3 * request.water_density_kg_m3 / 1000.0;
    let mass_residual_tonnes = displaced_mass_tonnes - request.total_mass_tonnes;

    let displacement_from_cg = [
        center_of_buoyancy.x_m - request.center_of_gravity_body_m.x_m,
        center_of_buoyancy.y_m - request.center_of_gravity_body_m.y_m,
        center_of_buoyancy.z_m - request.center_of_gravity_body_m.z_m,
    ];
    let buoyancy_force_n =
        displacement_volume_m3 * request.water_density_kg_m3 * request.gravity_m_s2;
    let buoyancy_force_vector = [
        -buoyancy_force_n * gravity.x,
        -buoyancy_force_n * gravity.y,
        -buoyancy_force_n * gravity.z,
    ];
    let moment_n_m = cross3(displacement_from_cg, buoyancy_force_vector);
    let (basis_1, basis_2) = moment_basis(gravity_vector);
    let residuals = StaticAttitudeResiduals {
        mass_residual_tonnes,
        moment_residual_1_nm: dot3(moment_n_m, basis_1),
        moment_residual_2_nm: dot3(moment_n_m, basis_2),
    };
    if !residuals.mass_residual_tonnes.is_finite()
        || !residuals.moment_residual_1_nm.is_finite()
        || !residuals.moment_residual_2_nm.is_finite()
    {
        return Err(vec![StaticAttitudeIssue {
            code: "residual_non_finite".into(),
            detail: "static-attitude force/moment residual is non-finite".into(),
        }]);
    }

    let mut receipt = StaticAttitudeEvaluationReceipt {
        receipt_schema_version: 1,
        algorithm: STATIC_ATTITUDE_EVALUATION_ALGORITHM.into(),
        source_spec_digest_blake3: source_digest,
        load_case_id: request.load_case_id.clone(),
        total_mass_tonnes: request.total_mass_tonnes,
        water_density_kg_m3: request.water_density_kg_m3,
        gravity_m_s2: request.gravity_m_s2,
        center_of_gravity_body_m: request.center_of_gravity_body_m,
        attitude: request.attitude,
        geometry_tolerance_m: tolerance,
        body_gravity_direction: gravity,
        displacement_volume_m3,
        displaced_mass_tonnes,
        center_of_buoyancy_body_m: center_of_buoyancy,
        residuals,
        station_count: samples.len(),
        receipt_digest_blake3: String::new(),
    };
    receipt.receipt_digest_blake3 =
        compute_static_attitude_receipt_digest(&receipt).map_err(|error| {
            vec![StaticAttitudeIssue {
                code: "receipt_digest".into(),
                detail: format!(
                    "could not serialize static-attitude receipt identity: {error}"
                ),
            }]
        })?;
    Ok(receipt)
}

pub fn verify_static_attitude_evaluation_receipt(
    request: &StaticAttitudeEvaluationRequest,
    receipt: &StaticAttitudeEvaluationReceipt,
) -> Result<(), Vec<StaticAttitudeIssue>> {
    let expected = evaluate_static_attitude(request)?;
    if &expected != receipt {
        return Err(vec![StaticAttitudeIssue {
            code: "static_attitude_receipt_mismatch".into(),
            detail: "receipt does not reproduce the exact full-envelope/load/attitude evaluation".into(),
        }]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sectional_reference::{SectionalReferenceRequest, evaluate_loading_sweep};

    fn box_source() -> FullEnvelopeSpec {
        serde_json::from_str(include_str!(
            "../../../../docs/research/marine/fixtures/marine-full-envelope-box-001.json"
        ))
        .unwrap()
    }

    fn box_request() -> StaticAttitudeEvaluationRequest {
        StaticAttitudeEvaluationRequest {
            schema_version: 1,
            source_spec: box_source(),
            load_case_id: "BOX-CENTRAL-CG".into(),
            total_mass_tonnes: 12_300.0,
            water_density_kg_m3: 1025.0,
            gravity_m_s2: 9.80665,
            center_of_gravity_body_m: BodyPoint3 {
                x_m: 50.0,
                y_m: 0.0,
                z_m: 3.0,
            },
            attitude: StaticAttitude {
                waterplane_offset_m: 0.0,
                pitch_rad: 0.0,
                roll_rad: 0.0,
            },
            geometry_tolerance_m: 1.0e-9,
        }
    }

    #[test]
    fn upright_oblique_cut_reproduces_independent_horizontal_loading_reference() {
        let request = box_request();
        let receipt = evaluate_static_attitude(&request).unwrap();
        assert!((receipt.displacement_volume_m3 - 12_000.0).abs() < 1.0e-8);
        assert!((receipt.center_of_buoyancy_body_m.x_m - 50.0).abs() < 1.0e-8);
        assert!(receipt.center_of_buoyancy_body_m.y_m.abs() < 1.0e-8);
        assert!((receipt.center_of_buoyancy_body_m.z_m - 3.0).abs() < 1.0e-8);
        assert!(receipt.residuals.mass_residual_tonnes.abs() < 1.0e-8);
        assert!(receipt.residuals.moment_residual_1_nm.abs() < 1.0e-7);
        assert!(receipt.residuals.moment_residual_2_nm.abs() < 1.0e-7);

        let horizontal_request: SectionalReferenceRequest = serde_json::from_value(
            serde_json::json!({
                "schema_version": 1,
                "source_spec": request.source_spec,
                "loading_conditions": [{
                    "condition_id": "UPRIGHT",
                    "waterplane_z_m": 0.0,
                    "water_density_kg_m3": 1025.0
                }]
            }),
        )
        .unwrap();
        let horizontal = evaluate_loading_sweep(&horizontal_request).unwrap();
        assert!(
            (receipt.displacement_volume_m3 - horizontal.results[0].displacement_volume_m3).abs()
                < 1.0e-8
        );
        assert!(
            (receipt.center_of_buoyancy_body_m.x_m
                - horizontal.results[0].longitudinal_center_of_buoyancy_m)
                .abs()
                < 1.0e-8
        );
        assert!(
            (receipt.center_of_buoyancy_body_m.y_m
                - horizontal.results[0].transverse_center_of_buoyancy_m)
                .abs()
                < 1.0e-8
        );
        assert!(
            (receipt.center_of_buoyancy_body_m.z_m
                - horizontal.results[0].vertical_center_of_buoyancy_m)
                .abs()
                < 1.0e-8
        );
    }

    #[test]
    fn pitch_and_roll_generate_the_expected_body_gravity_direction() {
        let attitude = StaticAttitude {
            waterplane_offset_m: 0.0,
            pitch_rad: 0.1,
            roll_rad: -0.2,
        };
        let gravity = body_gravity_direction(attitude);
        assert!((gravity.x + 0.1_f64.sin()).abs() < 1.0e-12);
        assert!((gravity.y + 0.2_f64.sin() * 0.1_f64.cos()).abs() < 1.0e-12);
        assert!((gravity.z - 0.2_f64.cos() * 0.1_f64.cos()).abs() < 1.0e-12);
        let norm = (gravity.x.powi(2) + gravity.y.powi(2) + gravity.z.powi(2)).sqrt();
        assert!((norm - 1.0).abs() < 1.0e-12);
    }


    #[test]
    fn pure_roll_clipping_matches_the_analytic_prismatic_section() {
        let mut request = box_request();
        request.attitude.roll_rad = 0.1;
        let receipt = evaluate_static_attitude(&request).unwrap();
        let expected_tcb = (2000.0 / 3.0) * 0.1_f64.tan() / 120.0;
        let expected_vcb = 3.0 - (1000.0 / 360.0) * 0.1_f64.tan().powi(2);
        assert!((receipt.displacement_volume_m3 - 12_000.0).abs() < 1.0e-7);
        assert!((receipt.center_of_buoyancy_body_m.y_m - expected_tcb).abs() < 1.0e-8);
        assert!((receipt.center_of_buoyancy_body_m.z_m - expected_vcb).abs() < 1.0e-8);
    }

    #[test]
    fn pure_pitch_clipping_reproduces_the_linear_prismatic_displacement() {
        let mut request = box_request();
        request.attitude.waterplane_offset_m = -1.0;
        request.attitude.pitch_rad = 0.02;
        let receipt = evaluate_static_attitude(&request).unwrap();
        assert!((receipt.displacement_volume_m3 - 12_000.0).abs() < 1.0e-7);
        assert!((receipt.center_of_buoyancy_body_m.x_m - 47.222_222_222_222_22).abs() < 1.0e-8);
        assert!(receipt.center_of_buoyancy_body_m.y_m.abs() < 1.0e-8);
    }

    #[test]
    fn explicit_cg_offset_produces_a_nonzero_physical_moment_residual() {
        let mut request = box_request();
        request.center_of_gravity_body_m.y_m = 1.0;
        let receipt = evaluate_static_attitude(&request).unwrap();
        assert!(receipt.residuals.moment_residual_1_nm.abs() > 1.0e6);
        assert!(receipt.residuals.moment_residual_2_nm.abs() < 1.0e-7);
    }

    #[test]
    fn moment_residual_vanishes_when_cg_is_on_the_buoyancy_gravity_line() {
        let mut request = box_request();
        request.attitude.pitch_rad = 0.08;
        request.attitude.roll_rad = -0.05;
        let first = evaluate_static_attitude(&request).unwrap();
        let gravity = first.body_gravity_direction;
        let displacement = 2.0;
        request.center_of_gravity_body_m = BodyPoint3 {
            x_m: first.center_of_buoyancy_body_m.x_m - displacement * gravity.x,
            y_m: first.center_of_buoyancy_body_m.y_m - displacement * gravity.y,
            z_m: first.center_of_buoyancy_body_m.z_m - displacement * gravity.z,
        };
        let second = evaluate_static_attitude(&request).unwrap();
        assert!(second.residuals.moment_residual_1_nm.abs() < 1.0e-6);
        assert!(second.residuals.moment_residual_2_nm.abs() < 1.0e-6);
    }

    #[test]
    fn asymmetric_section_geometry_is_not_mirrored_during_oblique_clipping() {
        let mut request = box_request();
        request.source_spec.symmetry = SymmetryPolicy::ExplicitPortAndStarboard;
        for station in &mut request.source_spec.stations {
            let mut port = station.starboard_profile.clone();
            for point in &mut port {
                if point.role != crate::full_envelope::ProfilePointRole::Keel {
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
        let receipt = evaluate_static_attitude(&request).unwrap();
        assert!((receipt.displacement_volume_m3 - 11_400.0).abs() < 1.0e-8);
        assert!((receipt.center_of_buoyancy_body_m.y_m - 0.5).abs() < 1.0e-8);
    }

    #[test]
    fn waterplane_outside_any_section_is_rejected() {
        let mut request = box_request();
        request.attitude.waterplane_offset_m = 7.0;
        let issues = evaluate_static_attitude(&request).unwrap_err();
        assert!(issues.iter().any(|issue| issue.code == "waterplane_outside_section"));
    }

    #[test]
    fn unsupported_geometry_never_becomes_a_static_attitude_result() {
        let mut request = box_request();
        request.source_spec.boundary_semantics.exterior_shell =
            crate::full_envelope::ExteriorShellSemantics::ExplicitBoundaryLoops;
        request.source_spec.boundary_semantics.openings.push(
            crate::full_envelope::OpeningBoundary {
                opening_id: "UNSUPPORTED-DOWNFLOODING-OPENING".into(),
                boundary_point_refs: vec![
                    "BOW-KEEL".into(),
                    "BOW-WL-S".into(),
                    "BOW-SHEER-S".into(),
                ],
                status: crate::full_envelope::OpeningStatus::UnsupportedRequiresReview,
            },
        );
        let issues = evaluate_static_attitude(&request).unwrap_err();
        assert!(issues.iter().any(|issue| issue.code == "source_spec_incomplete"));
    }

    #[test]
    fn static_attitude_receipt_is_deterministic_and_detects_rebound_mutation() {
        let request = box_request();
        let first = evaluate_static_attitude(&request).unwrap();
        let second = evaluate_static_attitude(&request).unwrap();
        assert_eq!(first, second);
        verify_static_attitude_evaluation_receipt(&request, &first).unwrap();

        let mut tampered = first;
        tampered.displacement_volume_m3 += 10.0;
        tampered.receipt_digest_blake3 =
            compute_static_attitude_receipt_digest(&tampered).unwrap();
        assert!(verify_static_attitude_evaluation_receipt(&request, &tampered).is_err());
    }

    #[test]
    fn static_attitude_receipt_binds_attitude_and_geometric_tolerance() {
        let mut request = box_request();
        let first = evaluate_static_attitude(&request).unwrap();
        request.geometry_tolerance_m = 2.0e-9;
        let second = evaluate_static_attitude(&request).unwrap();
        assert_ne!(first.receipt_digest_blake3, second.receipt_digest_blake3);
        request.geometry_tolerance_m = 1.0e-9;
        request.attitude.roll_rad = 0.01;
        let third = evaluate_static_attitude(&request).unwrap();
        assert_ne!(first.receipt_digest_blake3, third.receipt_digest_blake3);
    }

    #[test]
    fn static_attitude_rejects_non_finite_and_out_of_domain_attitudes() {
        let mut request = box_request();
        request.attitude.pitch_rad = f64::NAN;
        let issues = evaluate_static_attitude(&request).unwrap_err();
        assert!(issues.iter().any(|issue| issue.code == "attitude_non_finite"));

        request.attitude.pitch_rad = MAX_STATIC_ATTITUDE_ABS_RAD + 0.01;
        let issues = evaluate_static_attitude(&request).unwrap_err();
        assert!(issues.iter().any(|issue| issue.code == "attitude_domain"));
    }
}
