//! Independent sectional hydrostatics cross-check for the exact G2b triangle mesh.
//!
//! Unlike G2c, this module does not integrate tetrahedral mesh volume or sum
//! waterplane triangles. It intersects the actual side/waterplane facets at
//! each bay midpoint, computes section area/first moment from the resulting
//! polygon, and applies per-bay Simpson integration. It is a bounded
//! cross-check for the convex G2b displacement mesh, not a second production
//! hydrostatics or stability solver.

use serde::{Deserialize, Serialize};

use crate::hydrostatics::HydrostaticProperties;
use crate::loft::{verify_loft_artifact_integrity, LoftedHullArtifact};
use crate::section::{compute_sampled_sections_digest, SectionFamilyArtifact};

const RESULT_SCHEMA_VERSION: u32 = 1;
const METHOD_ID: &str = "triangle-slice-sectional-simpson-1-3-v1";
const WATERPLANE_TOLERANCE_M: f64 = 1.0e-9;
const MINIMUM_SECTION_AREA_M2: f64 = 1.0e-14;
const GEOMETRY_EPSILON_FACTOR: f64 = 128.0;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SectionalViolation {
    pub code: String,
    pub detail: String,
}

fn violation(code: &str, detail: impl Into<String>) -> SectionalViolation {
    SectionalViolation {
        code: code.to_owned(),
        detail: detail.into(),
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    fn add(&mut self, value: f64) {
        let adjusted = value - self.correction;
        let next = self.sum + adjusted;
        self.correction = (next - self.sum) - adjusted;
        self.sum = next;
    }

    fn value(self) -> f64 {
        self.sum
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SectionalHydrostaticsReference {
    pub result_schema_version: u32,
    pub method_id: String,
    pub candidate_id: String,
    pub source_spec_digest_blake3: String,
    pub sampled_sections_digest_blake3: String,
    pub geometry_digest_blake3: String,
    pub loft_manifest_digest_blake3: String,
    pub water_density_kg_m3: f64,
    pub displacement_volume_m3: f64,
    pub displacement_mass_kg: f64,
    pub displacement_tonnes: f64,
    /// Coordinates in the declared x-forward, y-starboard, z-down frame.
    pub centre_of_buoyancy_m: [f64; 3],
    pub longitudinal_centre_of_buoyancy_m: f64,
    pub transverse_centre_of_buoyancy_m: f64,
    /// Positive z is below the exact z=0 design waterplane.
    pub vertical_centre_of_buoyancy_below_waterplane_m: f64,
    pub waterplane_area_m2: f64,
    pub longitudinal_centre_of_flotation_m: f64,
    pub transverse_centre_of_flotation_m: f64,
    pub transverse_waterplane_second_moment_m4: f64,
    pub longitudinal_waterplane_second_moment_m4: f64,
    pub waterplane_product_of_inertia_m4: f64,
    pub transverse_metacentric_radius_bm_t_m: f64,
    pub longitudinal_metacentric_radius_bm_l_m: f64,
    pub tonnes_per_centimetre_immersion: f64,
    pub bay_count: usize,
    pub bay_lengths_m: Vec<f64>,
    pub midpoint_intersection_vertex_counts: Vec<usize>,
    pub result_digest_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricTolerance {
    pub absolute: f64,
    pub relative: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SectionalCrossCheckTolerance {
    pub profile_id: String,
    pub displacement_volume_m3: MetricTolerance,
    pub mass_kg: MetricTolerance,
    pub position_m: MetricTolerance,
    pub waterplane_area_m2: MetricTolerance,
    pub waterplane_second_moment_m4: MetricTolerance,
    pub metacentric_radius_m: MetricTolerance,
    pub tonnes_per_centimetre: MetricTolerance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricDisposition {
    Pass,
    Fail,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricComparison {
    pub metric_id: String,
    pub sectional_value: f64,
    pub polyhedral_value: f64,
    pub absolute_error: f64,
    pub allowed_error: f64,
    pub disposition: MetricDisposition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CrossCheckDisposition {
    Consistent,
    Inconsistent,
    Incomparable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SectionalCrossCheckReport {
    pub report_schema_version: u32,
    pub candidate_id: String,
    pub source_spec_digest_blake3: String,
    pub sampled_sections_digest_blake3: String,
    pub geometry_digest_blake3: String,
    pub loft_manifest_digest_blake3: String,
    pub tolerance_profile_id: String,
    pub disposition: CrossCheckDisposition,
    pub identity_errors: Vec<String>,
    pub metrics: Vec<MetricComparison>,
    pub report_digest_blake3: String,
}

#[derive(Debug, Clone, Copy)]
struct SectionSlice {
    area_m2: f64,
    vertical_first_moment_m3: f64,
    half_breadth_m: f64,
    intersection_vertex_count: usize,
}

fn close_enough(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance
}

fn geometry_tolerance(points: &[[f64; 2]]) -> f64 {
    let scale = points
        .iter()
        .flat_map(|point| point.iter())
        .fold(1.0_f64, |current, value| current.max(value.abs()));
    GEOMETRY_EPSILON_FACTOR * f64::EPSILON * scale
}

fn signed_polygon_area_and_centroid_z(points: &[[f64; 2]]) -> Result<(f64, f64), SectionalViolation> {
    if points.len() < 3 {
        return Err(violation(
            "section_polygon_too_small",
            "a submerged section requires at least three distinct boundary points",
        ));
    }

    // Translate the polygon in y before applying the shoelace formula. z is
    // already measured from the design waterplane and remains positive down.
    let y_reference = points.iter().map(|point| point[0]).sum::<f64>() / points.len() as f64;
    let mut twice_area = CompensatedSum::default();
    let mut vertical_centroid_numerator = CompensatedSum::default();

    for index in 0..points.len() {
        let next = (index + 1) % points.len();
        let y0 = points[index][0] - y_reference;
        let z0 = points[index][1];
        let y1 = points[next][0] - y_reference;
        let z1 = points[next][1];
        let cross = y0 * z1 - y1 * z0;
        twice_area.add(cross);
        vertical_centroid_numerator.add((z0 + z1) * cross);
    }

    let signed_twice_area = twice_area.value();
    if !signed_twice_area.is_finite() || signed_twice_area.abs() <= 2.0 * MINIMUM_SECTION_AREA_M2 {
        return Err(violation(
            "section_area_degenerate",
            "section polygon has zero, non-finite, or numerically unresolved area",
        ));
    }
    let area = signed_twice_area.abs() * 0.5;
    let centroid_z = vertical_centroid_numerator.value() / (3.0 * signed_twice_area);
    if !area.is_finite() || !centroid_z.is_finite() || centroid_z < -WATERPLANE_TOLERANCE_M {
        return Err(violation(
            "section_centroid_invalid",
            "section area or vertical centroid is non-finite or above the design waterplane",
        ));
    }
    Ok((area, centroid_z.max(0.0)))
}

fn ensure_convex_polygon(points: &[[f64; 2]]) -> Result<(), SectionalViolation> {
    let scale = points
        .iter()
        .flat_map(|point| point.iter())
        .fold(1.0_f64, |current, value| current.max(value.abs()));
    let tolerance = GEOMETRY_EPSILON_FACTOR * f64::EPSILON * scale * scale;
    let mut orientation = 0.0_f64;

    for index in 0..points.len() {
        let a = points[index];
        let b = points[(index + 1) % points.len()];
        let c = points[(index + 2) % points.len()];
        let cross = (b[0] - a[0]) * (c[1] - b[1])
            - (b[1] - a[1]) * (c[0] - b[0]);
        if cross.abs() <= tolerance {
            continue;
        }
        if orientation == 0.0 {
            orientation = cross.signum();
        } else if cross.signum() != orientation {
            return Err(violation(
                "section_not_convex",
                "the sectional cross-check currently admits convex section polygons only",
            ));
        }
    }

    if orientation == 0.0 {
        return Err(violation(
            "section_polygon_degenerate",
            "section polygon has no resolvable turn",
        ));
    }
    Ok(())
}

fn slice_metrics_from_polygon(
    mut points: Vec<[f64; 2]>,
) -> Result<SectionSlice, SectionalViolation> {
    if points.len() < 3 {
        return Err(violation(
            "section_intersection_incomplete",
            "the plane/mesh intersection did not produce a closed section polygon",
        ));
    }

    let dedupe_tolerance = geometry_tolerance(&points);
    points.sort_by(|left, right| {
        left[0]
            .total_cmp(&right[0])
            .then_with(|| left[1].total_cmp(&right[1]))
    });
    points.dedup_by(|left, right| {
        (left[0] - right[0]).abs() <= dedupe_tolerance
            && (left[1] - right[1]).abs() <= dedupe_tolerance
    });
    if points.len() < 3 {
        return Err(violation(
            "section_intersection_duplicate",
            "intersection points collapse to fewer than three distinct vertices",
        ));
    }

    let centre_y = points.iter().map(|point| point[0]).sum::<f64>() / points.len() as f64;
    let centre_z = points.iter().map(|point| point[1]).sum::<f64>() / points.len() as f64;
    points.sort_by(|left, right| {
        (left[1] - centre_z)
            .atan2(left[0] - centre_y)
            .total_cmp(&(right[1] - centre_z).atan2(right[0] - centre_y))
    });

    ensure_convex_polygon(&points)?;

    let (area_m2, centroid_z_m) = signed_polygon_area_and_centroid_z(&points)?;
    let mut minimum_y = f64::INFINITY;
    let mut maximum_y = f64::NEG_INFINITY;
    let waterline_tolerance = geometry_tolerance(&points).max(WATERPLANE_TOLERANCE_M);

    for point in &points {
        if point[1] < -WATERPLANE_TOLERANCE_M {
            return Err(violation(
                "section_above_waterplane",
                "section-plane intersection contains a point above z=0",
            ));
        }
        if point[1].abs() <= waterline_tolerance {
            minimum_y = minimum_y.min(point[0]);
            maximum_y = maximum_y.max(point[0]);
        }
    }

    if !minimum_y.is_finite() || !maximum_y.is_finite() || maximum_y <= minimum_y {
        return Err(violation(
            "waterline_intersection_missing",
            "the section-plane intersection has no finite waterline chord",
        ));
    }
    if (maximum_y + minimum_y).abs() > 10.0 * waterline_tolerance {
        return Err(violation(
            "section_not_symmetric",
            "G2b sectional cross-check requires port/starboard symmetry about y=0",
        ));
    }

    let half_breadth_m = (maximum_y - minimum_y) * 0.5;
    Ok(SectionSlice {
        area_m2,
        vertical_first_moment_m3: area_m2 * centroid_z_m,
        half_breadth_m,
        intersection_vertex_count: points.len(),
    })
}

fn section_slice_from_source(
    section: &crate::section::SampledSection,
) -> Result<SectionSlice, SectionalViolation> {
    let starboard = &section.keel_to_sheer_m;
    if starboard.len() < 2 {
        return Err(violation(
            "source_section_too_small",
            "source half-section requires at least keel and waterline points",
        ));
    }
    let mut polygon = Vec::with_capacity(starboard.len() * 2);
    for point in starboard {
        polygon.push([point[1], point[2]]);
    }
    polygon.push([-starboard.last().expect("non-empty section")[1],
                  starboard.last().expect("non-empty section")[2]]);
    for point in starboard.iter().rev().skip(1).take(starboard.len().saturating_sub(2)) {
        polygon.push([-point[1], point[2]]);
    }

    let mut slice = slice_metrics_from_polygon(polygon)?;
    slice.intersection_vertex_count = starboard.len() * 2 - 1;
    Ok(slice)
}

fn edge_intersection(
    left: [f64; 3],
    right: [f64; 3],
    plane_x_m: f64,
) -> Option<[f64; 2]> {
    let left_delta = left[0] - plane_x_m;
    let right_delta = right[0] - plane_x_m;
    if left_delta == 0.0 || right_delta == 0.0 || left_delta.signum() == right_delta.signum() {
        return None;
    }
    let denominator = right[0] - left[0];
    if denominator == 0.0 {
        return None;
    }
    let fraction = (plane_x_m - left[0]) / denominator;
    Some([
        left[1] + fraction * (right[1] - left[1]),
        left[2] + fraction * (right[2] - left[2]),
    ])
}

fn intersect_mesh_at_x(
    artifact: &LoftedHullArtifact,
    plane_x_m: f64,
) -> Result<SectionSlice, SectionalViolation> {
    let geometry = &artifact.candidate.geometry;
    let faces = &geometry.faces;
    let vertices = &geometry.vertices;
    let mut intersections = Vec::new();
    let ranges = [artifact.side_face_range, artifact.waterplane_face_range];

    for range in ranges {
        let Some(end) = range.first_face.checked_add(range.face_count) else {
            return Err(violation(
                "face_range_overflow",
                "side or waterplane face range overflows usize",
            ));
        };
        if end > faces.len() {
            return Err(violation(
                "face_range_out_of_bounds",
                "side or waterplane face range exceeds the mesh face array",
            ));
        }

        for face_index in range.first_face..end {
            let face = faces[face_index];
            let mut triangle_points = [[0.0; 3]; 3];
            for index in 0..3 {
                let Some(point) = vertices.get(face[index] as usize).copied() else {
                    return Err(violation(
                        "face_index_invalid",
                        format!("mesh face {face_index} has an out-of-range vertex index"),
                    ));
                };
                if point.iter().any(|value| !value.is_finite()) {
                    return Err(violation(
                        "mesh_coordinate_non_finite",
                        format!("mesh face {face_index} contains a non-finite coordinate"),
                    ));
                }
                triangle_points[index] = point;
            }

            let mut face_intersections = Vec::with_capacity(2);
            for (left, right) in [(0, 1), (1, 2), (2, 0)] {
                if let Some(point) =
                    edge_intersection(triangle_points[left], triangle_points[right], plane_x_m)
                {
                    face_intersections.push(point);
                }
            }

            match face_intersections.len() {
                0 => {}
                2 => intersections.extend(face_intersections),
                count => {
                    return Err(violation(
                        "triangle_slice_topology",
                        format!(
                            "mesh face {face_index} produced {count} intersections at an interior bay plane; expected zero or two"
                        ),
                    ));
                }
            }
        }
    }

    slice_metrics_from_polygon(intersections)
}

fn validate_inputs(
    sections: &SectionFamilyArtifact,
    loft: &LoftedHullArtifact,
    water_density_kg_m3: f64,
) -> Result<(), Vec<SectionalViolation>> {
    let mut errors = Vec::new();
    if !water_density_kg_m3.is_finite() || water_density_kg_m3 <= 0.0 {
        errors.push(violation(
            "water_density",
            "water density must be finite and strictly positive in kg/m^3",
        ));
    }
    if sections.artifact_schema_version != 1
        || sections.units != "m"
        || loft.candidate.geometry.units != "m"
    {
        errors.push(violation(
            "schema_or_units",
            "sectional cross-check requires schema version 1 and metres",
        ));
    }
    if sections.candidate_id != loft.candidate.candidate_id {
        errors.push(violation(
            "candidate_identity",
            "section family and loft candidate identities differ",
        ));
    }
    if sections.source_spec_digest_blake3 != loft.source_spec_digest_blake3 {
        errors.push(violation(
            "source_spec_digest",
            "section family and loft source-spec digests differ",
        ));
    }
    if sections.sampled_sections_digest_blake3 != loft.sampled_sections_digest_blake3 {
        errors.push(violation(
            "sampled_sections_digest",
            "section family and loft sampled-section digests differ",
        ));
    }
    if sections.coordinate_frame != loft.candidate.geometry.coordinate_frame {
        errors.push(violation(
            "coordinate_frame",
            "section family and loft coordinate frames differ",
        ));
    }
    let frame = &sections.coordinate_frame;
    if frame.handedness != "right-handed"
        || !frame.axis_convention.contains("x-forward")
        || !frame.axis_convention.contains("y-starboard")
        || !frame.axis_convention.contains("z-down")
    {
        errors.push(violation(
            "coordinate_frame_unsupported",
            "sectional cross-check requires right-handed x-forward/y-starboard/z-down coordinates",
        ));
    }

    match compute_sampled_sections_digest(&sections.sections) {
        Ok(actual) if actual.eq_ignore_ascii_case(&sections.sampled_sections_digest_blake3) => {}
        Ok(_) => errors.push(violation(
            "sample_digest_recomputed",
            "sampled section coordinates do not match the declared digest",
        )),
        Err(error) => errors.push(violation(
            "sample_digest_serialization",
            format!("sampled sections cannot be serialized: {error}"),
        )),
    }

    if sections.sections.len() < 2 {
        errors.push(violation(
            "station_count",
            "at least two longitudinal stations are required",
        ));
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    let point_count = sections.sections[0].keel_to_sheer_m.len();
    if point_count < 2 {
        errors.push(violation(
            "section_point_count",
            "each section needs at least two points",
        ));
    }
    let mut previous_x = f64::NEG_INFINITY;
    for (station_index, section) in sections.sections.iter().enumerate() {
        if !section.x_m.is_finite() || section.x_m <= previous_x {
            errors.push(violation(
                "station_order",
                format!("station {station_index} is non-finite or not strictly increasing"),
            ));
        }
        previous_x = section.x_m;
        if section.keel_to_sheer_m.len() != point_count {
            errors.push(violation(
                "section_point_count_mismatch",
                format!("station {station_index} has a different point count"),
            ));
            continue;
        }
        for (point_index, point) in section.keel_to_sheer_m.iter().enumerate() {
            if point.iter().any(|value| !value.is_finite()) {
                errors.push(violation(
                    "section_coordinate_non_finite",
                    format!("station {station_index} point {point_index} is non-finite"),
                ));
            }
            if !close_enough(point[0], section.x_m, WATERPLANE_TOLERANCE_M) {
                errors.push(violation(
                    "section_station_x_mismatch",
                    format!("station {station_index} point {point_index} has an inconsistent x coordinate"),
                ));
            }
            if point[1] < -WATERPLANE_TOLERANCE_M
                || point[2] < -WATERPLANE_TOLERANCE_M
            {
                errors.push(violation(
                    "section_coordinate_outside_domain",
                    format!("station {station_index} point {point_index} is outside the starboard/submerged domain"),
                ));
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[derive(Debug, Clone, Serialize)]
struct ReferenceDigestIdentity<'a> {
    result_schema_version: u32,
    method_id: &'a str,
    candidate_id: &'a str,
    source_spec_digest_blake3: &'a str,
    sampled_sections_digest_blake3: &'a str,
    geometry_digest_blake3: &'a str,
    loft_manifest_digest_blake3: &'a str,
    water_density_kg_m3: f64,
    displacement_volume_m3: f64,
    displacement_mass_kg: f64,
    displacement_tonnes: f64,
    centre_of_buoyancy_m: [f64; 3],
    waterplane_area_m2: f64,
    longitudinal_centre_of_flotation_m: f64,
    transverse_waterplane_second_moment_m4: f64,
    longitudinal_waterplane_second_moment_m4: f64,
    waterplane_product_of_inertia_m4: f64,
    transverse_metacentric_radius_bm_t_m: f64,
    longitudinal_metacentric_radius_bm_l_m: f64,
    tonnes_per_centimetre_immersion: f64,
    bay_lengths_m: &'a [f64],
    midpoint_intersection_vertex_counts: &'a [usize],
}

fn reference_digest(
    result: &SectionalHydrostaticsReference,
) -> Result<String, serde_json::Error> {
    let identity = ReferenceDigestIdentity {
        result_schema_version: result.result_schema_version,
        method_id: &result.method_id,
        candidate_id: &result.candidate_id,
        source_spec_digest_blake3: &result.source_spec_digest_blake3,
        sampled_sections_digest_blake3: &result.sampled_sections_digest_blake3,
        geometry_digest_blake3: &result.geometry_digest_blake3,
        loft_manifest_digest_blake3: &result.loft_manifest_digest_blake3,
        water_density_kg_m3: result.water_density_kg_m3,
        displacement_volume_m3: result.displacement_volume_m3,
        displacement_mass_kg: result.displacement_mass_kg,
        displacement_tonnes: result.displacement_tonnes,
        centre_of_buoyancy_m: result.centre_of_buoyancy_m,
        waterplane_area_m2: result.waterplane_area_m2,
        longitudinal_centre_of_flotation_m: result.longitudinal_centre_of_flotation_m,
        transverse_waterplane_second_moment_m4: result.transverse_waterplane_second_moment_m4,
        longitudinal_waterplane_second_moment_m4: result.longitudinal_waterplane_second_moment_m4,
        waterplane_product_of_inertia_m4: result.waterplane_product_of_inertia_m4,
        transverse_metacentric_radius_bm_t_m: result.transverse_metacentric_radius_bm_t_m,
        longitudinal_metacentric_radius_bm_l_m: result.longitudinal_metacentric_radius_bm_l_m,
        tonnes_per_centimetre_immersion: result.tonnes_per_centimetre_immersion,
        bay_lengths_m: &result.bay_lengths_m,
        midpoint_intersection_vertex_counts: &result.midpoint_intersection_vertex_counts,
    };
    Ok(blake3::hash(&serde_json::to_vec(&identity)?).to_hex().to_string())
}

pub fn verify_sectional_reference_integrity(
    result: &SectionalHydrostaticsReference,
) -> Result<(), SectionalViolation> {
    if result.result_schema_version != RESULT_SCHEMA_VERSION || result.method_id != METHOD_ID {
        return Err(violation(
            "reference_schema",
            "unsupported sectional reference schema or method identity",
        ));
    }
    let actual = reference_digest(result).map_err(|error| {
        violation(
            "reference_digest_serialization",
            format!("sectional result cannot be serialized: {error}"),
        )
    })?;
    if !actual.eq_ignore_ascii_case(&result.result_digest_blake3) {
        return Err(violation(
            "reference_digest",
            "sectional reference result digest does not match its content",
        ));
    }
    Ok(())
}

/// Independently integrates the exact G2b piecewise-planar displacement mesh.
///
/// Each station bay is sliced at its midpoint by intersecting the actual
/// side/waterplane triangles. The section area and first moments are then
/// integrated using Simpson's 1/3 rule separately over that bay. Because the
/// cross-section vertices move linearly within each planar triangle and the
/// bay topology is fixed, area is at most quadratic and its first moments at
/// most cubic in x; Simpson's rule is exact for those polynomials, up to
/// floating-point arithmetic.
pub fn calculate_sectional_reference(
    sections: &SectionFamilyArtifact,
    loft: &LoftedHullArtifact,
    water_density_kg_m3: f64,
) -> Result<SectionalHydrostaticsReference, Vec<SectionalViolation>> {
    validate_inputs(sections, loft, water_density_kg_m3)?;
    verify_loft_artifact_integrity(loft).map_err(|errors| {
        errors
            .into_iter()
            .map(|error| violation(
                "loft_integrity",
                format!("{}: {}", error.code, error.detail),
            ))
            .collect::<Vec<_>>()
    })?;

    let mut displacement = CompensatedSum::default();
    let mut first_x = CompensatedSum::default();
    let mut first_z = CompensatedSum::default();
    let mut waterplane_area = CompensatedSum::default();
    let mut waterplane_first_x = CompensatedSum::default();
    let mut transverse_second = CompensatedSum::default();

    let mut bay_lengths_m = Vec::with_capacity(sections.sections.len() - 1);
    let mut midpoint_intersection_vertex_counts =
        Vec::with_capacity(sections.sections.len() - 1);

    for index in 0..sections.sections.len() - 1 {
        let left = &sections.sections[index];
        let right = &sections.sections[index + 1];
        let dx = right.x_m - left.x_m;
        if !dx.is_finite() || dx <= 0.0 {
            return Err(vec![violation(
                "bay_length",
                format!("bay {index} has non-positive or non-finite length"),
            )]);
        }
        let midpoint_x = left.x_m + dx * 0.5;
        let left_slice = section_slice_from_source(left).map_err(|error| vec![error])?;
        let midpoint_slice =
            intersect_mesh_at_x(loft, midpoint_x).map_err(|error| vec![error])?;
        let right_slice = section_slice_from_source(right).map_err(|error| vec![error])?;

        let simpson = |f0: f64, fm: f64, f1: f64| dx * (f0 + 4.0 * fm + f1) / 6.0;
        let volume_bay = simpson(
            left_slice.area_m2,
            midpoint_slice.area_m2,
            right_slice.area_m2,
        );
        let x_moment_bay = simpson(
            left.x_m * left_slice.area_m2,
            midpoint_x * midpoint_slice.area_m2,
            right.x_m * right_slice.area_m2,
        );
        let z_moment_bay = simpson(
            left_slice.vertical_first_moment_m3,
            midpoint_slice.vertical_first_moment_m3,
            right_slice.vertical_first_moment_m3,
        );

        let left_breadth = left_slice.half_breadth_m;
        let middle_breadth = midpoint_slice.half_breadth_m;
        let right_breadth = right_slice.half_breadth_m;
        let wp_area_bay = simpson(
            2.0 * left_breadth,
            2.0 * middle_breadth,
            2.0 * right_breadth,
        );
        let wp_x_moment_bay = simpson(
            2.0 * left.x_m * left_breadth,
            2.0 * midpoint_x * middle_breadth,
            2.0 * right.x_m * right_breadth,
        );
        let transverse_second_bay = simpson(
            (2.0 / 3.0) * left_breadth.powi(3),
            (2.0 / 3.0) * middle_breadth.powi(3),
            (2.0 / 3.0) * right_breadth.powi(3),
        );

        for value in [
            volume_bay,
            x_moment_bay,
            z_moment_bay,
            wp_area_bay,
            wp_x_moment_bay,
            transverse_second_bay,
        ] {
            if !value.is_finite() {
                return Err(vec![violation(
                    "non_finite_integral",
                    format!("bay {index} produced a non-finite sectional integral"),
                )]);
            }
        }

        displacement.add(volume_bay);
        first_x.add(x_moment_bay);
        first_z.add(z_moment_bay);
        waterplane_area.add(wp_area_bay);
        waterplane_first_x.add(wp_x_moment_bay);
        transverse_second.add(transverse_second_bay);
        bay_lengths_m.push(dx);
        midpoint_intersection_vertex_counts.push(midpoint_slice.intersection_vertex_count);
    }

    let volume = displacement.value();
    let aw = waterplane_area.value();
    if !volume.is_finite() || volume <= 0.0 || !aw.is_finite() || aw <= MINIMUM_SECTION_AREA_M2 {
        return Err(vec![violation(
            "positive_integrals",
            "sectional displacement and waterplane area must be finite and positive",
        )]);
    }

    let lcb = first_x.value() / volume;
    let vcb = first_z.value() / volume;
    let lcf = waterplane_first_x.value() / aw;
    let longitudinal_second = {
        let mut sum = CompensatedSum::default();
        for index in 0..sections.sections.len() - 1 {
            let left = &sections.sections[index];
            let right = &sections.sections[index + 1];
            let dx = right.x_m - left.x_m;
            let midpoint_x = left.x_m + dx * 0.5;
            let left_slice = section_slice_from_source(left).map_err(|error| vec![error])?;
            let midpoint_slice =
                intersect_mesh_at_x(loft, midpoint_x).map_err(|error| vec![error])?;
            let right_slice = section_slice_from_source(right).map_err(|error| vec![error])?;
            let f0 = 2.0 * (left.x_m - lcf).powi(2) * left_slice.half_breadth_m;
            let fm = 2.0 * (midpoint_x - lcf).powi(2) * midpoint_slice.half_breadth_m;
            let f1 = 2.0 * (right.x_m - lcf).powi(2) * right_slice.half_breadth_m;
            sum.add(dx * (f0 + 4.0 * fm + f1) / 6.0);
        }
        sum.value()
    };

    let mass = water_density_kg_m3 * volume;
    let tonnes = mass / 1_000.0;
    let bm_t = transverse_second.value() / volume;
    let bm_l = longitudinal_second / volume;
    let tpc = water_density_kg_m3 * aw * 0.01 / 1_000.0;
    let values = [lcb, vcb, lcf, mass, tonnes, bm_t, bm_l, tpc, longitudinal_second];
    if values.iter().any(|value| !value.is_finite()) {
        return Err(vec![violation(
            "non_finite_properties",
            "one or more derived sectional hydrostatic properties are non-finite",
        )]);
    }

    let mut result = SectionalHydrostaticsReference {
        result_schema_version: RESULT_SCHEMA_VERSION,
        method_id: METHOD_ID.to_owned(),
        candidate_id: loft.candidate.candidate_id.clone(),
        source_spec_digest_blake3: sections.source_spec_digest_blake3.clone(),
        sampled_sections_digest_blake3: sections.sampled_sections_digest_blake3.clone(),
        geometry_digest_blake3: loft.mesh_digest_blake3.clone(),
        loft_manifest_digest_blake3: loft.loft_manifest_digest_blake3.clone(),
        water_density_kg_m3,
        displacement_volume_m3: volume,
        displacement_mass_kg: mass,
        displacement_tonnes: tonnes,
        centre_of_buoyancy_m: [lcb, 0.0, vcb],
        longitudinal_centre_of_buoyancy_m: lcb,
        transverse_centre_of_buoyancy_m: 0.0,
        vertical_centre_of_buoyancy_below_waterplane_m: vcb,
        waterplane_area_m2: aw,
        longitudinal_centre_of_flotation_m: lcf,
        transverse_centre_of_flotation_m: 0.0,
        transverse_waterplane_second_moment_m4: transverse_second.value(),
        longitudinal_waterplane_second_moment_m4: longitudinal_second,
        waterplane_product_of_inertia_m4: 0.0,
        transverse_metacentric_radius_bm_t_m: bm_t,
        longitudinal_metacentric_radius_bm_l_m: bm_l,
        tonnes_per_centimetre_immersion: tpc,
        bay_count: bay_lengths_m.len(),
        bay_lengths_m,
        midpoint_intersection_vertex_counts,
        result_digest_blake3: String::new(),
    };
    result.result_digest_blake3 = reference_digest(&result)
        .map_err(|error| vec![violation(
            "result_digest",
            format!("sectional result digest cannot be computed: {error}"),
        )])?;
    Ok(result)
}

fn compare_metric(
    metric_id: &str,
    sectional_value: f64,
    polyhedral_value: f64,
    tolerance: &MetricTolerance,
) -> MetricComparison {
    let absolute_error = (sectional_value - polyhedral_value).abs();
    let allowed_error =
        tolerance.absolute + tolerance.relative * sectional_value.abs().max(polyhedral_value.abs());
    MetricComparison {
        metric_id: metric_id.to_owned(),
        sectional_value,
        polyhedral_value,
        absolute_error,
        allowed_error,
        disposition: if absolute_error <= allowed_error {
            MetricDisposition::Pass
        } else {
            MetricDisposition::Fail
        },
    }
}

#[derive(Debug, Clone, Serialize)]
struct CrossCheckDigestIdentity<'a> {
    report_schema_version: u32,
    candidate_id: &'a str,
    source_spec_digest_blake3: &'a str,
    sampled_sections_digest_blake3: &'a str,
    geometry_digest_blake3: &'a str,
    loft_manifest_digest_blake3: &'a str,
    tolerance_profile_id: &'a str,
    disposition: CrossCheckDisposition,
    identity_errors: &'a [String],
    metrics: &'a [MetricComparison],
}

fn report_digest(report: &SectionalCrossCheckReport) -> Result<String, serde_json::Error> {
    let identity = CrossCheckDigestIdentity {
        report_schema_version: report.report_schema_version,
        candidate_id: &report.candidate_id,
        source_spec_digest_blake3: &report.source_spec_digest_blake3,
        sampled_sections_digest_blake3: &report.sampled_sections_digest_blake3,
        geometry_digest_blake3: &report.geometry_digest_blake3,
        loft_manifest_digest_blake3: &report.loft_manifest_digest_blake3,
        tolerance_profile_id: &report.tolerance_profile_id,
        disposition: report.disposition,
        identity_errors: &report.identity_errors,
        metrics: &report.metrics,
    };
    Ok(blake3::hash(&serde_json::to_vec(&identity)?).to_hex().to_string())
}

/// Compares the independent sectional reference with the G2c polyhedral result.
/// A mismatched source/mesh/density identity is INCOMPARABLE, never a pass.
pub fn compare_with_polyhedral_hydrostatics(
    sectional: &SectionalHydrostaticsReference,
    polyhedral: &HydrostaticProperties,
    tolerance: &SectionalCrossCheckTolerance,
) -> Result<SectionalCrossCheckReport, SectionalViolation> {
    verify_sectional_reference_integrity(sectional)?;
    if tolerance.profile_id.trim().is_empty() {
        return Err(violation(
            "tolerance_profile_id",
            "cross-check tolerance profile ID must not be empty",
        ));
    }
    let all_tolerances = [
        &tolerance.displacement_volume_m3,
        &tolerance.mass_kg,
        &tolerance.position_m,
        &tolerance.waterplane_area_m2,
        &tolerance.waterplane_second_moment_m4,
        &tolerance.metacentric_radius_m,
        &tolerance.tonnes_per_centimetre,
    ];
    if all_tolerances.iter().any(|item| {
        !item.absolute.is_finite()
            || !item.relative.is_finite()
            || item.absolute < 0.0
            || item.relative < 0.0
    }) {
        return Err(violation(
            "tolerance_value",
            "all absolute/relative tolerances must be finite and non-negative",
        ));
    }

    let mut identity_errors = Vec::new();
    if sectional.candidate_id != polyhedral.candidate_id {
        identity_errors.push("candidate IDs differ".to_owned());
    }
    if sectional.geometry_digest_blake3 != polyhedral.geometry_digest_blake3 {
        identity_errors.push("mesh geometry digests differ".to_owned());
    }
    if sectional.loft_manifest_digest_blake3 != polyhedral.loft_manifest_digest_blake3 {
        identity_errors.push("loft manifest digests differ".to_owned());
    }
    if sectional.water_density_kg_m3 != polyhedral.water_density_kg_m3 {
        identity_errors.push("water densities differ".to_owned());
    }

    let mut report = SectionalCrossCheckReport {
        report_schema_version: 1,
        candidate_id: sectional.candidate_id.clone(),
        source_spec_digest_blake3: sectional.source_spec_digest_blake3.clone(),
        sampled_sections_digest_blake3: sectional.sampled_sections_digest_blake3.clone(),
        geometry_digest_blake3: sectional.geometry_digest_blake3.clone(),
        loft_manifest_digest_blake3: sectional.loft_manifest_digest_blake3.clone(),
        tolerance_profile_id: tolerance.profile_id.clone(),
        disposition: CrossCheckDisposition::Incomparable,
        identity_errors,
        metrics: Vec::new(),
        report_digest_blake3: String::new(),
    };

    if report.identity_errors.is_empty() {
        let mut metrics = vec![
            compare_metric(
                "displacement_volume_m3",
                sectional.displacement_volume_m3,
                polyhedral.displacement_volume_m3,
                &tolerance.displacement_volume_m3,
            ),
            compare_metric(
                "displacement_mass_kg",
                sectional.displacement_mass_kg,
                polyhedral.displacement_mass_kg,
                &tolerance.mass_kg,
            ),
            compare_metric(
                "lcb_m",
                sectional.longitudinal_centre_of_buoyancy_m,
                polyhedral.longitudinal_centre_of_buoyancy_m,
                &tolerance.position_m,
            ),
            compare_metric(
                "tcb_m",
                sectional.transverse_centre_of_buoyancy_m,
                polyhedral.transverse_centre_of_buoyancy_m,
                &tolerance.position_m,
            ),
            compare_metric(
                "vcb_below_waterplane_m",
                sectional.vertical_centre_of_buoyancy_below_waterplane_m,
                polyhedral.vertical_centre_of_buoyancy_below_waterplane_m,
                &tolerance.position_m,
            ),
            compare_metric(
                "waterplane_area_m2",
                sectional.waterplane_area_m2,
                polyhedral.waterplane_area_m2,
                &tolerance.waterplane_area_m2,
            ),
            compare_metric(
                "lcf_m",
                sectional.longitudinal_centre_of_flotation_m,
                polyhedral.longitudinal_centre_of_flotation_m,
                &tolerance.position_m,
            ),
            compare_metric(
                "transverse_waterplane_second_moment_m4",
                sectional.transverse_waterplane_second_moment_m4,
                polyhedral.transverse_waterplane_second_moment_m4,
                &tolerance.waterplane_second_moment_m4,
            ),
            compare_metric(
                "longitudinal_waterplane_second_moment_m4",
                sectional.longitudinal_waterplane_second_moment_m4,
                polyhedral.longitudinal_waterplane_second_moment_m4,
                &tolerance.waterplane_second_moment_m4,
            ),
            compare_metric(
                "bm_t_m",
                sectional.transverse_metacentric_radius_bm_t_m,
                polyhedral.transverse_metacentric_radius_bm_t_m,
                &tolerance.metacentric_radius_m,
            ),
            compare_metric(
                "bm_l_m",
                sectional.longitudinal_metacentric_radius_bm_l_m,
                polyhedral.longitudinal_metacentric_radius_bm_l_m,
                &tolerance.metacentric_radius_m,
            ),
            compare_metric(
                "tpc_tonnes_per_cm",
                sectional.tonnes_per_centimetre_immersion,
                polyhedral.tonnes_per_centimetre_immersion,
                &tolerance.tonnes_per_centimetre,
            ),
        ];
        report.metrics.append(&mut metrics);
        report.disposition = if report
            .metrics
            .iter()
            .all(|metric| metric.disposition == MetricDisposition::Pass)
        {
            CrossCheckDisposition::Consistent
        } else {
            CrossCheckDisposition::Inconsistent
        };
    }

    report.report_digest_blake3 = report_digest(&report).map_err(|error| {
        violation(
            "report_digest",
            format!("cross-check report digest cannot be computed: {error}"),
        )
    })?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydrostatics::calculate_hydrostatic_properties;
    use crate::loft::loft_waterplane_capped_hull;
    use crate::section::{compute_sampled_sections_digest, SampledSection};
    use crate::{AdmissionProfile, CoordinateFrame};

    fn frame() -> CoordinateFrame {
        CoordinateFrame {
            id: "ship-local".into(),
            axis_convention: "x-forward,y-starboard,z-down".into(),
            handedness: "right-handed".into(),
        }
    }

    fn family(
        stations: &[(f64, f64, f64)],
        section_shape: &[[f64; 2]],
        candidate_id: &str,
    ) -> SectionFamilyArtifact {
        let sections = stations
            .iter()
            .map(|(x, half_breadth, draft)| SampledSection {
                station_index: 0,
                x_m: *x,
                keel_to_sheer_m: section_shape
                    .iter()
                    .map(|point| [*x, point[0] * half_breadth, point[1] * draft])
                    .collect(),
            })
            .enumerate()
            .map(|(index, mut section)| {
                section.station_index = index;
                section
            })
            .collect::<Vec<_>>();
        let mut artifact = SectionFamilyArtifact {
            artifact_schema_version: 1,
            candidate_id: candidate_id.into(),
            source_spec_digest_blake3: "a".repeat(64),
            sampled_sections_digest_blake3: String::new(),
            length_m: stations.last().unwrap().0 - stations[0].0,
            beam_m: 2.0 * stations.iter().map(|station| station.1).fold(0.0, f64::max),
            draft_m: stations.iter().map(|station| station.2).fold(0.0, f64::max),
            units: "m".into(),
            coordinate_frame: frame(),
            sections,
        };
        artifact.sampled_sections_digest_blake3 =
            compute_sampled_sections_digest(&artifact.sections).unwrap();
        artifact
    }

    fn build_loft(sections: &SectionFamilyArtifact) -> LoftedHullArtifact {
        let profile = AdmissionProfile {
            id: "sectional-cross-check-test-v1".into(),
            require_closed_surface: true,
            minimum_face_area_m2: 1.0e-12,
            minimum_enclosed_volume_m3: 1.0e-12,
            additional_required_checks: vec![],
        };
        loft_waterplane_capped_hull(sections, profile).unwrap()
    }

    fn triangular_prism() -> (SectionFamilyArtifact, LoftedHullArtifact) {
        let sections = family(
            &[(0.0, 10.0, 6.0), (100.0, 10.0, 6.0)],
            &[[0.0, 1.0], [1.0, 0.0]],
            "triangular-prism-sectional-001",
        );
        let loft = build_loft(&sections);
        (sections, loft)
    }

    fn rectangular_prism() -> (SectionFamilyArtifact, LoftedHullArtifact) {
        let sections = family(
            &[(0.0, 10.0, 6.0), (100.0, 10.0, 6.0)],
            &[[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
            "rectangular-prism-sectional-001",
        );
        let loft = build_loft(&sections);
        (sections, loft)
    }

    fn test_tolerance() -> SectionalCrossCheckTolerance {
        let tight = |absolute| MetricTolerance {
            absolute,
            relative: 1.0e-10,
        };
        SectionalCrossCheckTolerance {
            profile_id: "known-answer-sectional-cross-check-v1".into(),
            displacement_volume_m3: tight(1.0e-8),
            mass_kg: tight(1.0e-5),
            position_m: tight(1.0e-9),
            waterplane_area_m2: tight(1.0e-8),
            waterplane_second_moment_m4: tight(1.0e-5),
            metacentric_radius_m: tight(1.0e-9),
            tonnes_per_centimetre: tight(1.0e-9),
        }
    }

    #[test]
    fn triangular_prism_sectional_oracle_matches_polyhedral_hydrostatics() {
        let (sections, loft) = triangular_prism();
        let reference = calculate_sectional_reference(&sections, &loft, 1_025.0).unwrap();
        assert!((reference.displacement_volume_m3 - 6_000.0).abs() < 1.0e-8);
        assert!((reference.longitudinal_centre_of_buoyancy_m - 50.0).abs() < 1.0e-9);
        assert!((reference.vertical_centre_of_buoyancy_below_waterplane_m - 2.0).abs() < 1.0e-9);
        assert!((reference.waterplane_area_m2 - 2_000.0).abs() < 1.0e-8);
        let polyhedral = calculate_hydrostatic_properties(&loft, 1_025.0).unwrap();
        let report =
            compare_with_polyhedral_hydrostatics(&reference, &polyhedral, &test_tolerance()).unwrap();
        assert_eq!(report.disposition, CrossCheckDisposition::Consistent);
        assert!(report.metrics.iter().all(|metric| metric.disposition == MetricDisposition::Pass));
        verify_sectional_reference_integrity(&reference).unwrap();
    }

    #[test]
    fn rectangular_prism_sectional_oracle_matches_polyhedral_hydrostatics() {
        let (sections, loft) = rectangular_prism();
        let reference = calculate_sectional_reference(&sections, &loft, 1_025.0).unwrap();
        assert!((reference.displacement_volume_m3 - 12_000.0).abs() < 1.0e-8);
        assert!((reference.vertical_centre_of_buoyancy_below_waterplane_m - 3.0).abs() < 1.0e-9);
        let polyhedral = calculate_hydrostatic_properties(&loft, 1_025.0).unwrap();
        let report =
            compare_with_polyhedral_hydrostatics(&reference, &polyhedral, &test_tolerance()).unwrap();
        assert_eq!(report.disposition, CrossCheckDisposition::Consistent);
    }

    #[test]
    fn nonuniform_tapered_bays_are_integrated_independently() {
        let sections = family(
            &[(0.0, 8.0, 4.0), (40.0, 9.0, 6.0), (120.0, 12.0, 8.0)],
            &[[0.0, 1.0], [1.0, 0.0]],
            "nonuniform-tapered-triangle-001",
        );
        let loft = build_loft(&sections);
        let reference = calculate_sectional_reference(&sections, &loft, 1_025.0).unwrap();
        let polyhedral = calculate_hydrostatic_properties(&loft, 1_025.0).unwrap();
        let report =
            compare_with_polyhedral_hydrostatics(&reference, &polyhedral, &test_tolerance()).unwrap();
        assert_eq!(report.disposition, CrossCheckDisposition::Consistent);
        assert_eq!(reference.bay_count, 2);
        assert_eq!(reference.bay_lengths_m, vec![40.0, 80.0]);
        assert!((reference.displacement_volume_m3 - 7_626.666_666_666_667).abs() < 1.0e-8);
    }

    #[test]
    fn a_mutated_polyhedral_result_is_inconsistent_not_a_pass() {
        let (sections, loft) = triangular_prism();
        let reference = calculate_sectional_reference(&sections, &loft, 1_025.0).unwrap();
        let mut polyhedral = calculate_hydrostatic_properties(&loft, 1_025.0).unwrap();
        polyhedral.displacement_volume_m3 += 0.01;
        let report =
            compare_with_polyhedral_hydrostatics(&reference, &polyhedral, &test_tolerance()).unwrap();
        assert_eq!(report.disposition, CrossCheckDisposition::Inconsistent);
        assert!(report.metrics.iter().any(|metric| {
            metric.metric_id == "displacement_volume_m3"
                && metric.disposition == MetricDisposition::Fail
        }));
    }

    #[test]
    fn wrong_mesh_identity_is_incomparable_not_a_pass() {
        let (sections, loft) = triangular_prism();
        let reference = calculate_sectional_reference(&sections, &loft, 1_025.0).unwrap();
        let mut polyhedral = calculate_hydrostatic_properties(&loft, 1_025.0).unwrap();
        polyhedral.geometry_digest_blake3 = "b".repeat(64);
        let report =
            compare_with_polyhedral_hydrostatics(&reference, &polyhedral, &test_tolerance()).unwrap();
        assert_eq!(report.disposition, CrossCheckDisposition::Incomparable);
        assert!(!report.identity_errors.is_empty());
    }

    #[test]
    fn wrong_density_is_incomparable() {
        let (sections, loft) = triangular_prism();
        let reference = calculate_sectional_reference(&sections, &loft, 1_025.0).unwrap();
        let polyhedral = calculate_hydrostatic_properties(&loft, 1_000.0).unwrap();
        let report =
            compare_with_polyhedral_hydrostatics(&reference, &polyhedral, &test_tolerance()).unwrap();
        assert_eq!(report.disposition, CrossCheckDisposition::Incomparable);
    }

    #[test]
    fn tampered_section_digest_is_rejected() {
        let (mut sections, loft) = triangular_prism();
        sections.sections[1].keel_to_sheer_m[1][1] += 0.1;
        let errors = calculate_sectional_reference(&sections, &loft, 1_025.0).unwrap_err();
        assert!(errors.iter().any(|error| error.code == "sample_digest_recomputed"));
    }

    #[test]
    fn tampered_reference_receipt_is_rejected() {
        let (sections, loft) = triangular_prism();
        let mut reference = calculate_sectional_reference(&sections, &loft, 1_025.0).unwrap();
        reference.displacement_volume_m3 += 1.0;
        assert_eq!(
            verify_sectional_reference_integrity(&reference).unwrap_err().code,
            "reference_digest"
        );
    }
}
