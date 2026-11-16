//! G2b deterministic, convex-section waterplane-capped surface construction.
//!
//! The result is a closed displacement-volume mesh bounded by the design
//! waterplane. It is not a deck/freeboard model or a general-purpose CAD hull.

use serde::{Deserialize, Serialize};

use crate::section::{
    LongitudinalEndCondition, LongitudinalEndConditions, SectionFamilyArtifact,
    compute_sampled_sections_digest, verify_section_family_artifact_integrity,
};
use crate::{
    AdmissionProfile, AdmissionReport, CandidateInput, MeshGeometry, admit, compute_geometry_digest,
};

const LOFT_SCHEMA_VERSION: u32 = 2;
const MAX_STATIONS: usize = 512;
const MAX_SECTION_POINTS: usize = 8_193;
const MAX_MESH_VERTICES: usize = 1_000_000;
const MAX_MESH_FACES: usize = 2_000_000;
const GEOMETRY_TOLERANCE: f64 = 1.0e-10;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoftViolation {
    pub code: String,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FaceRange {
    pub first_face: usize,
    pub face_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoftedHullArtifact {
    pub artifact_schema_version: u32,
    pub generator: String,
    /// This cap closes the submerged displacement volume at the design
    /// waterplane. It is not a deck surface or a freeboard representation.
    pub cap_semantics: String,
    pub source_spec_digest_blake3: String,
    pub sampled_sections_digest_blake3: String,
    pub mesh_digest_blake3: String,
    pub loft_manifest_digest_blake3: String,
    pub side_face_range: FaceRange,
    pub waterplane_face_range: FaceRange,
    pub end_cap_face_range: FaceRange,
    #[serde(default)]
    pub bow_end_cap_face_range: FaceRange,
    #[serde(default)]
    pub stern_end_cap_face_range: FaceRange,
    #[serde(default)]
    pub station_count: usize,
    #[serde(default)]
    pub section_point_count: usize,
    #[serde(default)]
    pub longitudinal_end_conditions: LongitudinalEndConditions,
    pub candidate: CandidateInput,
    pub admission_report: AdmissionReport,
}

#[derive(Serialize)]
struct LoftArtifactIdentity<'a> {
    artifact_schema_version: u32,
    generator: &'a str,
    cap_semantics: &'a str,
    source_spec_digest_blake3: &'a str,
    sampled_sections_digest_blake3: &'a str,
    mesh_digest_blake3: &'a str,
    side_face_range: FaceRange,
    waterplane_face_range: FaceRange,
    end_cap_face_range: FaceRange,
    bow_end_cap_face_range: FaceRange,
    stern_end_cap_face_range: FaceRange,
    station_count: usize,
    section_point_count: usize,
    longitudinal_end_conditions: LongitudinalEndConditions,
    candidate_schema_version: u32,
    candidate_id: &'a str,
    geometry_digest_blake3: &'a str,
    admission_profile: &'a AdmissionProfile,
}

fn compute_loft_manifest_digest(
    artifact: &LoftedHullArtifact,
) -> Result<String, serde_json::Error> {
    let identity = LoftArtifactIdentity {
        artifact_schema_version: artifact.artifact_schema_version,
        generator: &artifact.generator,
        cap_semantics: &artifact.cap_semantics,
        source_spec_digest_blake3: &artifact.source_spec_digest_blake3,
        sampled_sections_digest_blake3: &artifact.sampled_sections_digest_blake3,
        mesh_digest_blake3: &artifact.mesh_digest_blake3,
        side_face_range: artifact.side_face_range,
        waterplane_face_range: artifact.waterplane_face_range,
        end_cap_face_range: artifact.end_cap_face_range,
        bow_end_cap_face_range: artifact.bow_end_cap_face_range,
        stern_end_cap_face_range: artifact.stern_end_cap_face_range,
        station_count: artifact.station_count,
        section_point_count: artifact.section_point_count,
        longitudinal_end_conditions: artifact.longitudinal_end_conditions,
        candidate_schema_version: artifact.candidate.schema_version,
        candidate_id: &artifact.candidate.candidate_id,
        geometry_digest_blake3: &artifact.candidate.geometry_digest_blake3,
        admission_profile: &artifact.candidate.profile,
    };
    let bytes = serde_json::to_vec(&identity)?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn violation(violations: &mut Vec<LoftViolation>, code: &str, detail: impl Into<String>) {
    violations.push(LoftViolation {
        code: code.to_owned(),
        detail: detail.into(),
    });
}

fn close_enough(left: f64, right: f64, tolerance: f64) -> bool {
    (left - right).abs() <= tolerance
}

fn cross_2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn section_ring(section: &crate::section::SampledSection) -> Vec<[f64; 3]> {
    let mut ring = section.keel_to_sheer_m.clone();
    for point in section.keel_to_sheer_m.iter().skip(1).rev() {
        ring.push([point[0], -point[1], point[2]]);
    }
    ring
}

fn validate_section_family(artifact: &SectionFamilyArtifact) -> Result<usize, Vec<LoftViolation>> {
    let mut violations = Vec::new();

    if artifact.artifact_schema_version != 2 {
        violation(
            &mut violations,
            "section_artifact_version",
            "only section artifact schema version 2 is supported by G2e",
        );
    }
    if let Err(errors) = verify_section_family_artifact_integrity(artifact) {
        for error in errors {
            violation(
                &mut violations,
                &format!("section_family_{}", error.code),
                error.detail,
            );
        }
    }
    if artifact.candidate_id.trim().is_empty() {
        violation(
            &mut violations,
            "candidate_id",
            "candidate_id must not be empty",
        );
    }
    if artifact.units != "m" {
        violation(
            &mut violations,
            "units",
            "G2b accepts metres only; no implicit conversion is performed",
        );
    }

    for (name, value) in [
        ("length_m", artifact.length_m),
        ("beam_m", artifact.beam_m),
        ("draft_m", artifact.draft_m),
    ] {
        if !value.is_finite() || value <= 0.0 {
            violation(
                &mut violations,
                name,
                format!("{name} must be finite and positive"),
            );
        }
    }

    let frame = &artifact.coordinate_frame;
    if frame.id.trim().is_empty()
        || frame.handedness != "right-handed"
        || !frame.axis_convention.contains("x-forward")
        || !frame.axis_convention.contains("y-starboard")
        || !frame.axis_convention.contains("z-down")
    {
        violation(
            &mut violations,
            "coordinate_frame",
            "G2b requires explicit right-handed x-forward, y-starboard, z-down coordinates",
        );
    }

    if artifact.sections.len() < 2 || artifact.sections.len() > MAX_STATIONS {
        violation(
            &mut violations,
            "station_count",
            format!("station count must be between 2 and {MAX_STATIONS}"),
        );
    }

    if artifact.sections.len() == 2
        && artifact.longitudinal_end_conditions.bow == LongitudinalEndCondition::CollapsedCentreline
        && artifact.longitudinal_end_conditions.stern
            == LongitudinalEndCondition::CollapsedCentreline
    {
        violation(
            &mut violations,
            "adjacent_collapsed_ends",
            "two adjacent collapsed endpoints have no positive interior station",
        );
    }

    if artifact.source_spec_digest_blake3.len() != 64
        || !artifact
            .source_spec_digest_blake3
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        violation(
            &mut violations,
            "source_spec_digest",
            "source specification digest must be a 64-character hexadecimal BLAKE3 digest",
        );
    }

    match compute_sampled_sections_digest(&artifact.sections) {
        Ok(actual) if actual.eq_ignore_ascii_case(&artifact.sampled_sections_digest_blake3) => {}
        Ok(_) => violation(
            &mut violations,
            "sample_digest",
            "sampled section coordinates do not match sampled_sections_digest_blake3",
        ),
        Err(error) => violation(
            &mut violations,
            "sample_digest",
            format!("sampled sections cannot be serialized: {error}"),
        ),
    }

    let Some(first) = artifact.sections.first() else {
        return Err(violations);
    };
    let point_count = first.keel_to_sheer_m.len();
    if point_count < 2 || point_count > MAX_SECTION_POINTS {
        violation(
            &mut violations,
            "section_point_count",
            format!("each section must have 2..={MAX_SECTION_POINTS} points"),
        );
        return Err(violations);
    }

    let total_samples = artifact.sections.len().checked_mul(point_count);
    match total_samples {
        Some(total) if total <= MAX_MESH_VERTICES / 2 => {}
        _ => violation(
            &mut violations,
            "mesh_vertex_budget",
            "section family exceeds the supported mirrored mesh vertex budget",
        ),
    }

    let coordinate_tolerance = artifact.length_m.max(1.0) * GEOMETRY_TOLERANCE;
    let mut previous_x: Option<f64> = None;

    for (station_index, section) in artifact.sections.iter().enumerate() {
        if section.station_index != station_index {
            violation(
                &mut violations,
                "station_identity",
                format!("station index {station_index} is not preserved in the section artifact"),
            );
        }
        if section.keel_to_sheer_m.len() != point_count {
            violation(
                &mut violations,
                "section_point_count_mismatch",
                format!("station {station_index} has a different number of sampled points"),
            );
            continue;
        }
        if !section.x_m.is_finite() {
            violation(
                &mut violations,
                "station_x",
                format!("station {station_index} x coordinate is non-finite"),
            );
            continue;
        }
        if let Some(previous) = previous_x {
            if section.x_m <= previous {
                violation(
                    &mut violations,
                    "station_order",
                    format!("station {station_index} is not strictly after its predecessor"),
                );
            }
        }
        previous_x = Some(section.x_m);

        for (point_index, point) in section.keel_to_sheer_m.iter().enumerate() {
            if point.iter().any(|coordinate| !coordinate.is_finite()) {
                violation(
                    &mut violations,
                    "finite_coordinates",
                    format!("station {station_index}, point {point_index} is non-finite"),
                );
                continue;
            }
            if !close_enough(point[0], section.x_m, coordinate_tolerance) {
                violation(
                    &mut violations,
                    "station_plane",
                    format!(
                        "station {station_index} contains a point outside its declared x-plane"
                    ),
                );
                break;
            }
            if point[1] < -coordinate_tolerance
                || point[1] > artifact.beam_m * 0.5 + coordinate_tolerance
                || point[2] < -coordinate_tolerance
                || point[2] > artifact.draft_m + coordinate_tolerance
            {
                violation(
                    &mut violations,
                    "section_bounds",
                    format!(
                        "station {station_index}, point {point_index} lies outside the declared beam/draft envelope"
                    ),
                );
                break;
            }
            if point_index > 0 {
                let previous = section.keel_to_sheer_m[point_index - 1];
                if point[1] < previous[1] - coordinate_tolerance
                    || point[2] > previous[2] + coordinate_tolerance
                {
                    violation(
                        &mut violations,
                        "section_monotonicity",
                        format!("station {station_index} is not monotonic from keel to waterline"),
                    );
                    break;
                }
                if close_enough(point[1], previous[1], coordinate_tolerance)
                    && close_enough(point[2], previous[2], coordinate_tolerance)
                {
                    violation(
                        &mut violations,
                        "duplicate_section_point",
                        format!("station {station_index} contains duplicate consecutive points"),
                    );
                    break;
                }
            }
        }

        let keel = section.keel_to_sheer_m[0];
        let waterline = section.keel_to_sheer_m[point_count - 1];
        if !close_enough(keel[1], 0.0, coordinate_tolerance) || keel[2] <= coordinate_tolerance {
            violation(
                &mut violations,
                "keel_endpoint",
                format!(
                    "station {station_index} must begin on the centreline below the waterplane"
                ),
            );
        }
        if !close_enough(waterline[2], 0.0, coordinate_tolerance) {
            violation(
                &mut violations,
                "waterline_endpoint",
                format!("station {station_index} must end on z=0"),
            );
        }

        let endpoint_condition = if station_index == 0 {
            Some(artifact.longitudinal_end_conditions.bow)
        } else if station_index + 1 == artifact.sections.len() {
            Some(artifact.longitudinal_end_conditions.stern)
        } else {
            None
        };
        let collapsed = endpoint_condition == Some(LongitudinalEndCondition::CollapsedCentreline);

        if collapsed {
            if section.keel_to_sheer_m.iter().any(|point| point[1] != 0.0) {
                violation(
                    &mut violations,
                    "collapsed_end_not_centreline",
                    format!(
                        "collapsed endpoint station {station_index} must have exact y=0 for every sample"
                    ),
                );
            }
            for pair in section.keel_to_sheer_m.windows(2) {
                if pair[0][2] <= pair[1][2] {
                    violation(
                        &mut violations,
                        "collapsed_end_sample_order",
                        format!(
                            "collapsed endpoint station {station_index} must have strictly descending z samples from keel to waterline"
                        ),
                    );
                    break;
                }
            }
            // A collapsed endpoint is a line, not an area-bearing section ring.
            continue;
        }

        if waterline[1] <= coordinate_tolerance {
            violation(
                &mut violations,
                "waterline_endpoint",
                format!("station {station_index} must end at positive starboard breadth on z=0"),
            );
        }

        let ring = section_ring(section);
        if ring.len() < 3 {
            violation(
                &mut violations,
                "section_ring",
                format!("station {station_index} does not form a closed section ring"),
            );
            continue;
        }

        let signed_area_twice: f64 = (0..ring.len())
            .map(|index| {
                let current = ring[index];
                let next = ring[(index + 1) % ring.len()];
                current[1] * next[2] - current[2] * next[1]
            })
            .sum();
        // Cross products have area units. Scale the coordinate tolerance
        // by the section length scale; squaring a length tolerance would be
        // dimensionally inconsistent and can reject exactly collinear
        // linearly sampled points due to floating-point roundoff.
        let section_scale = artifact.beam_m.max(artifact.draft_m).max(1.0);
        let area_tolerance = coordinate_tolerance * section_scale;
        if !signed_area_twice.is_finite() || signed_area_twice >= -area_tolerance {
            violation(
                &mut violations,
                "section_area_orientation",
                format!("station {station_index} must enclose a non-zero clockwise yz-section"),
            );
        }

        for index in 0..ring.len() {
            let previous = ring[(index + ring.len() - 1) % ring.len()];
            let current = ring[index];
            let next = ring[(index + 1) % ring.len()];
            let turn = cross_2d(
                [previous[1], previous[2]],
                [current[1], current[2]],
                [next[1], next[2]],
            );
            if !turn.is_finite() || turn > area_tolerance {
                violation(
                    &mut violations,
                    "non_convex_section",
                    format!(
                        "station {station_index} is outside the supported convex-section loft profile"
                    ),
                );
                break;
            }
        }
    }

    if artifact.sections.len() >= 2 {
        if !close_enough(artifact.sections[0].x_m, 0.0, coordinate_tolerance) {
            violation(
                &mut violations,
                "first_station",
                "first section must be at x=0",
            );
        }
        if let Some(last) = artifact.sections.last() {
            if !close_enough(last.x_m, artifact.length_m, coordinate_tolerance) {
                violation(
                    &mut violations,
                    "last_station",
                    "last section must be at x=length_m",
                );
            }
        }
    }

    let first_signature = &first.keel_to_sheer_m;
    if first_signature.len() == point_count {
        for (station_index, section) in artifact.sections.iter().enumerate().skip(1) {
            if section.keel_to_sheer_m.len() != point_count {
                continue;
            }
            if section.keel_to_sheer_m.len() != first_signature.len() {
                violation(
                    &mut violations,
                    "section_point_count_mismatch",
                    format!("station {station_index} has incompatible section sampling"),
                );
            }
        }
    }

    if violations.is_empty() {
        Ok(point_count)
    } else {
        Err(violations)
    }
}

fn section_centre(section: &crate::section::SampledSection) -> [f64; 3] {
    let ring = section_ring(section);
    let count = ring.len() as f64;
    [
        section.x_m,
        ring.iter().map(|point| point[1]).sum::<f64>() / count,
        ring.iter().map(|point| point[2]).sum::<f64>() / count,
    ]
}

/// Verify that the mesh bytes and the face-region metadata are bound to
/// the stored geometry and loft-manifest digests.
pub fn verify_loft_artifact_integrity(
    artifact: &LoftedHullArtifact,
) -> Result<(), Vec<LoftViolation>> {
    let mut violations = Vec::new();
    match compute_geometry_digest(&artifact.candidate.geometry) {
        Ok(actual)
            if actual.eq_ignore_ascii_case(&artifact.candidate.geometry_digest_blake3)
                && actual.eq_ignore_ascii_case(&artifact.mesh_digest_blake3) => {}
        Ok(_) => violation(
            &mut violations,
            "mesh_digest",
            "candidate geometry does not match the mesh digest references",
        ),
        Err(error) => violation(
            &mut violations,
            "mesh_digest",
            format!("candidate geometry cannot be serialized: {error}"),
        ),
    }
    match compute_loft_manifest_digest(artifact) {
        Ok(actual) if actual.eq_ignore_ascii_case(&artifact.loft_manifest_digest_blake3) => {}
        Ok(_) => violation(
            &mut violations,
            "loft_manifest_digest",
            "loft metadata or face-region references do not match the manifest digest",
        ),
        Err(error) => violation(
            &mut violations,
            "loft_manifest_digest",
            format!("loft manifest cannot be serialized: {error}"),
        ),
    }

    let face_count = artifact.candidate.geometry.faces.len();
    let vertex_count = artifact.candidate.geometry.vertices.len();
    if face_count > MAX_MESH_FACES {
        violation(
            &mut violations,
            "mesh_face_budget",
            format!("candidate has {face_count} faces; maximum is {MAX_MESH_FACES}"),
        );
    } else {
        let ranges = [
            artifact.side_face_range,
            artifact.waterplane_face_range,
            artifact.end_cap_face_range,
        ];
        let mut covered = vec![false; face_count];
        for range in ranges {
            match range.first_face.checked_add(range.face_count) {
                Some(end) if end <= face_count => {
                    for index in range.first_face..end {
                        if covered[index] {
                            violation(
                                &mut violations,
                                "face_region_overlap",
                                format!("face {index} appears in multiple semantic ranges"),
                            );
                            break;
                        }
                        covered[index] = true;
                    }
                }
                _ => violation(
                    &mut violations,
                    "face_region_bounds",
                    "a semantic face range exceeds the candidate mesh",
                ),
            }
        }
        if covered.iter().any(|value| !value) {
            violation(
                &mut violations,
                "face_region_coverage",
                "semantic face ranges do not cover every mesh face",
            );
        }
    }
    if vertex_count > MAX_MESH_VERTICES {
        violation(
            &mut violations,
            "mesh_vertex_budget",
            format!("candidate has {vertex_count} vertices; maximum is {MAX_MESH_VERTICES}"),
        );
    }

    if artifact.artifact_schema_version != LOFT_SCHEMA_VERSION
        || artifact.generator != "symthaea-marine-waterplane-loft-v2"
        || artifact.cap_semantics != "closed-displacement-volume-at-design-waterplane-z0"
        || artifact.candidate.schema_version != 1
    {
        violation(
            &mut violations,
            "loft_artifact_version",
            "loft artifact, generator, cap semantics or candidate schema is unsupported",
        );
    }
    if !artifact.candidate.profile.require_closed_surface {
        violation(
            &mut violations,
            "profile_requires_closure",
            "loft artifact profile must require a closed surface",
        );
    }

    let fresh_admission = admit(&artifact.candidate);
    if fresh_admission != artifact.admission_report {
        violation(
            &mut violations,
            "admission_report_mismatch",
            "stored geometry admission report does not match a fresh evaluation of the candidate and profile",
        );
    }

    let side_end = artifact
        .side_face_range
        .first_face
        .checked_add(artifact.side_face_range.face_count);
    let waterplane_end = artifact
        .waterplane_face_range
        .first_face
        .checked_add(artifact.waterplane_face_range.face_count);
    let end_cap_end = artifact
        .end_cap_face_range
        .first_face
        .checked_add(artifact.end_cap_face_range.face_count);
    if artifact.side_face_range.first_face != 0
        || side_end != Some(artifact.waterplane_face_range.first_face)
        || waterplane_end != Some(artifact.end_cap_face_range.first_face)
        || end_cap_end != Some(face_count)
    {
        violation(
            &mut violations,
            "face_region_order",
            "side, waterplane and end-cap face ranges must be contiguous in generator order",
        );
    }

    let bow_collapsed =
        artifact.longitudinal_end_conditions.bow == LongitudinalEndCondition::CollapsedCentreline;
    let stern_collapsed =
        artifact.longitudinal_end_conditions.stern == LongitudinalEndCondition::CollapsedCentreline;
    let point_count = artifact.section_point_count;
    let station_count = artifact.station_count;
    let mut expected_vertices = 0_usize;
    let mut station_bases = Vec::with_capacity(station_count.min(MAX_STATIONS));
    if point_count < 2
        || point_count > MAX_SECTION_POINTS
        || station_count < 2
        || station_count > MAX_STATIONS
        || (station_count == 2 && bow_collapsed && stern_collapsed)
    {
        violation(
            &mut violations,
            "end_semantics_station_budget",
            "unsupported station/section counts for declared longitudinal end conditions",
        );
    } else {
        let mut vertex_layout_valid = true;
        for station_index in 0..station_count {
            station_bases.push(expected_vertices);
            let collapsed = (station_index == 0 && bow_collapsed)
                || (station_index + 1 == station_count && stern_collapsed);
            let station_vertices = point_count + if collapsed { 0 } else { point_count - 1 };
            let Some(next) = expected_vertices.checked_add(station_vertices) else {
                vertex_layout_valid = false;
                break;
            };
            let end = next.min(artifact.candidate.geometry.vertices.len());
            if next > artifact.candidate.geometry.vertices.len() {
                vertex_layout_valid = false;
                break;
            }
            let vertices = &artifact.candidate.geometry.vertices[expected_vertices..end];
            let starboard = &vertices[..point_count];
            if collapsed {
                if starboard.iter().any(|point| point[1] != 0.0) {
                    violation(
                        &mut violations,
                        "collapsed_end_geometry",
                        format!(
                            "collapsed endpoint station {station_index} has a non-zero transverse coordinate"
                        ),
                    );
                    vertex_layout_valid = false;
                }
            } else {
                let port = &vertices[point_count..];
                for point_index in 1..point_count {
                    let source = starboard[point_index];
                    let mirrored = port[point_index - 1];
                    if source[0] != mirrored[0]
                        || source[1] != -mirrored[1]
                        || source[2] != mirrored[2]
                    {
                        violation(
                            &mut violations,
                            "port_starboard_mirror",
                            format!(
                                "station {station_index} port sample {point_index} is not the exact mirror of its starboard sample"
                            ),
                        );
                        vertex_layout_valid = false;
                        break;
                    }
                }
            }
            expected_vertices = next;
        }
        let planar_cap_count = usize::from(!bow_collapsed) + usize::from(!stern_collapsed);
        let expected_total_vertices = expected_vertices.checked_add(planar_cap_count);
        if !vertex_layout_valid
            || expected_total_vertices != Some(artifact.candidate.geometry.vertices.len())
        {
            violation(
                &mut violations,
                "vertex_layout",
                "mesh vertex layout does not match the versioned station/end-condition budget",
            );
        }
        if expected_vertices > artifact.candidate.geometry.vertices.len() {
            vertex_layout_valid = false;
        }
        if vertex_layout_valid {
            let mut centre_offset = expected_vertices;
            if !bow_collapsed {
                let centre = artifact.candidate.geometry.vertices[centre_offset];
                if (centre[0] - artifact.candidate.geometry.vertices[0][0]).abs()
                    > GEOMETRY_TOLERANCE
                    || centre[1].abs() > GEOMETRY_TOLERANCE
                {
                    violation(
                        &mut violations,
                        "bow_cap_centre",
                        "planar bow cap centre must lie in the bow section plane and centreline",
                    );
                }
                centre_offset += 1;
            }
            if !stern_collapsed {
                let centre = artifact.candidate.geometry.vertices[centre_offset];
                let last_station_base =
                    (0..station_count - 1).fold(0_usize, |offset, station_index| {
                        let collapsed = station_index == 0 && bow_collapsed;
                        offset + point_count + if collapsed { 0 } else { point_count - 1 }
                    });
                let stern_x = artifact.candidate.geometry.vertices[last_station_base][0];
                if (centre[0] - stern_x).abs() > GEOMETRY_TOLERANCE
                    || centre[1].abs() > GEOMETRY_TOLERANCE
                {
                    violation(
                        &mut violations,
                        "stern_cap_centre",
                        "planar stern cap centre must lie in the stern section plane and centreline",
                    );
                }
            }
        }
    }

    let expected_side_face_budget = station_count
        .checked_sub(1)
        .and_then(|count| count.checked_mul(4))
        .and_then(|count| count.checked_mul(point_count.saturating_sub(1)));
    if station_count >= 2
        && station_count <= MAX_STATIONS
        && point_count >= 2
        && point_count <= MAX_SECTION_POINTS
        && expected_side_face_budget.is_some_and(|count| count <= MAX_MESH_FACES)
        && station_bases.len() == station_count
        && artifact
            .side_face_range
            .first_face
            .checked_add(artifact.side_face_range.face_count)
            == Some(artifact.waterplane_face_range.first_face)
    {
        let faces = &artifact.candidate.geometry.faces;
        let mut expected_side = Vec::<[u32; 3]>::new();
        let mut expected_waterplane = Vec::<[u32; 3]>::new();
        let port_vertex = |station_index: usize, point_index: usize| -> u32 {
            let base = station_bases[station_index];
            let collapsed = (station_index == 0 && bow_collapsed)
                || (station_index + 1 == station_count && stern_collapsed);
            if point_index == 0 || collapsed {
                (base + point_index) as u32
            } else {
                (base + point_count + point_index - 1) as u32
            }
        };
        let starboard_vertex = |station_index: usize, point_index: usize| -> u32 {
            (station_bases[station_index] + point_index) as u32
        };
        for interval in 0..station_count - 1 {
            let right_collapsed = interval + 1 == station_count - 1 && stern_collapsed;
            for point_index in 0..point_count - 1 {
                let a = starboard_vertex(interval, point_index);
                let b = starboard_vertex(interval + 1, point_index);
                let c = starboard_vertex(interval + 1, point_index + 1);
                let d = starboard_vertex(interval, point_index + 1);
                let port_a = port_vertex(interval, point_index);
                let port_b = port_vertex(interval + 1, point_index);
                let port_c = port_vertex(interval + 1, point_index + 1);
                let port_d = port_vertex(interval, point_index + 1);
                if right_collapsed {
                    expected_side.extend([
                        [a, b, d],
                        [b, c, d],
                        [port_a, port_d, port_b],
                        [port_d, port_c, port_b],
                    ]);
                } else {
                    expected_side.extend([
                        [a, b, c],
                        [a, c, d],
                        [port_a, port_c, port_b],
                        [port_a, port_d, port_c],
                    ]);
                }
            }

            let left_collapsed = interval == 0 && bow_collapsed;
            let port_here = port_vertex(interval, point_count - 1);
            let starboard_here = starboard_vertex(interval, point_count - 1);
            let starboard_next = starboard_vertex(interval + 1, point_count - 1);
            let port_next = port_vertex(interval + 1, point_count - 1);
            match (left_collapsed, right_collapsed) {
                (true, false) => expected_waterplane.push([port_here, starboard_next, port_next]),
                (false, true) => {
                    expected_waterplane.push([port_here, starboard_here, starboard_next])
                }
                (false, false) => {
                    expected_waterplane.push([port_here, starboard_here, starboard_next]);
                    expected_waterplane.push([port_here, starboard_next, port_next]);
                }
                (true, true) => {}
            }
        }
        let side_first = artifact.side_face_range.first_face;
        let side_end = side_first.saturating_add(artifact.side_face_range.face_count);
        if side_end <= faces.len()
            && faces[side_first..side_end]
                .iter()
                .copied()
                .ne(expected_side.iter().copied())
        {
            violation(
                &mut violations,
                "side_closure_semantics",
                "side triangle indices do not match the declared shared-vertex endpoint and diagonal semantics",
            );
        }

        let first = artifact.waterplane_face_range.first_face;
        let end = first.saturating_add(artifact.waterplane_face_range.face_count);
        if end <= faces.len()
            && faces[first..end]
                .iter()
                .copied()
                .ne(expected_waterplane.iter().copied())
        {
            violation(
                &mut violations,
                "waterplane_closure_semantics",
                "waterplane triangle indices do not match the declared collapsed/planar endpoint semantics",
            );
        }

        let expected_cap_faces = |is_bow: bool, centre_index: u32| -> Vec<[u32; 3]> {
            let station_index = if is_bow { 0 } else { station_count - 1 };
            let mut ring = (0..point_count)
                .map(|point_index| starboard_vertex(station_index, point_index))
                .collect::<Vec<_>>();
            for point_index in (1..point_count).rev() {
                ring.push(port_vertex(station_index, point_index));
            }
            let mut cap_faces = Vec::with_capacity(ring.len());
            for index in 0..ring.len() {
                let next = (index + 1) % ring.len();
                if is_bow {
                    cap_faces.push([centre_index, ring[index], ring[next]]);
                } else {
                    cap_faces.push([centre_index, ring[next], ring[index]]);
                }
            }
            cap_faces
        };

        let planar_bow = !bow_collapsed;
        let planar_stern = !stern_collapsed;
        let first_centre_index = expected_vertices as u32;
        if planar_bow {
            let expected = expected_cap_faces(true, first_centre_index);
            let range = artifact.bow_end_cap_face_range;
            let end = range.first_face.saturating_add(range.face_count);
            if end <= faces.len()
                && faces[range.first_face..end]
                    .iter()
                    .copied()
                    .ne(expected.iter().copied())
            {
                violation(
                    &mut violations,
                    "bow_cap_topology",
                    "bow end-cap triangles do not match the declared planar bow ring and centre",
                );
            }
        }
        if planar_stern {
            let stern_centre_index = first_centre_index + (if planar_bow { 1_u32 } else { 0_u32 });
            let expected = expected_cap_faces(false, stern_centre_index);
            let range = artifact.stern_end_cap_face_range;
            let end = range.first_face.saturating_add(range.face_count);
            if end <= faces.len()
                && faces[range.first_face..end]
                    .iter()
                    .copied()
                    .ne(expected.iter().copied())
            {
                violation(
                    &mut violations,
                    "stern_cap_topology",
                    "stern end-cap triangles do not match the declared planar stern ring and centre",
                );
            }
        }
    }

    let expected_side_faces = station_count
        .checked_sub(1)
        .and_then(|count| count.checked_mul(4))
        .and_then(|count| count.checked_mul(point_count.saturating_sub(1)));
    let expected_waterplane_faces = station_count
        .checked_sub(1)
        .and_then(|count| count.checked_mul(2))
        .and_then(|count| count.checked_sub(if bow_collapsed { 1_usize } else { 0_usize }))
        .and_then(|count| count.checked_sub(if stern_collapsed { 1_usize } else { 0_usize }));
    let cap_faces_per_planar_end = point_count
        .checked_mul(2)
        .and_then(|count| count.checked_sub(1));
    let expected_bow_cap_faces = if bow_collapsed {
        Some(0)
    } else {
        cap_faces_per_planar_end
    };
    let expected_stern_cap_faces = if stern_collapsed {
        Some(0)
    } else {
        cap_faces_per_planar_end
    };
    if expected_side_faces != Some(artifact.side_face_range.face_count)
        || expected_waterplane_faces != Some(artifact.waterplane_face_range.face_count)
        || expected_bow_cap_faces != Some(artifact.bow_end_cap_face_range.face_count)
        || expected_stern_cap_faces != Some(artifact.stern_end_cap_face_range.face_count)
    {
        violation(
            &mut violations,
            "end_face_budget",
            "semantic face counts do not match the declared endpoint closure conditions",
        );
    }

    let expected_bow_range_end = artifact
        .bow_end_cap_face_range
        .first_face
        .checked_add(artifact.bow_end_cap_face_range.face_count);
    let expected_stern_range_end = artifact
        .stern_end_cap_face_range
        .first_face
        .checked_add(artifact.stern_end_cap_face_range.face_count);
    if artifact.bow_end_cap_face_range.first_face != artifact.end_cap_face_range.first_face
        || expected_bow_range_end != Some(artifact.stern_end_cap_face_range.first_face)
        || expected_stern_range_end != end_cap_end
        || artifact
            .bow_end_cap_face_range
            .face_count
            .checked_add(artifact.stern_end_cap_face_range.face_count)
            != Some(artifact.end_cap_face_range.face_count)
    {
        violation(
            &mut violations,
            "end_cap_range_semantics",
            "bow/stern subranges must exactly partition the combined end-cap range",
        );
    }

    if violations.is_empty() {
        Ok(())
    } else {
        Err(violations)
    }
}

/// Loft a validated section family into a deterministic closed displacement
/// volume capped at z=0 (the design waterplane).
///
/// The profile is restricted to convex sections so each end cap can use a
/// centre fan with a known interior point. No geometry healing or resampling
/// is performed.
pub fn loft_waterplane_capped_hull(
    sections: &SectionFamilyArtifact,
    profile: AdmissionProfile,
) -> Result<LoftedHullArtifact, Vec<LoftViolation>> {
    let point_count = validate_section_family(sections)?;
    let station_count = sections.sections.len();

    if !profile.require_closed_surface {
        return Err(vec![LoftViolation {
            code: "profile_requires_closure".into(),
            detail:
                "waterplane-capped displacement meshes require a closed-surface admission profile"
                    .into(),
        }]);
    }

    let mirrored_point_count = point_count.checked_sub(1);
    let Some(mirrored_point_count) = mirrored_point_count else {
        return Err(vec![LoftViolation {
            code: "mesh_vertex_budget".into(),
            detail: "section point count is too small to mirror".into(),
        }]);
    };
    let bow_collapsed =
        sections.longitudinal_end_conditions.bow == LongitudinalEndCondition::CollapsedCentreline;
    let stern_collapsed =
        sections.longitudinal_end_conditions.stern == LongitudinalEndCondition::CollapsedCentreline;
    let bow_planar = !bow_collapsed;
    let stern_planar = !stern_collapsed;

    // Preflight every station independently: collapsed endpoints share every
    // port/starboard sample ID; all other stations add mirrored samples except keel.
    let mut vertex_count = 0_usize;
    for station_index in 0..station_count {
        let collapsed = (station_index == 0 && bow_collapsed)
            || (station_index + 1 == station_count && stern_collapsed);
        let station_vertices =
            point_count.checked_add(if collapsed { 0 } else { mirrored_point_count });
        let Some(station_vertices) = station_vertices else {
            return Err(vec![LoftViolation {
                code: "mesh_vertex_budget".into(),
                detail: "per-station vertex count overflowed".into(),
            }]);
        };
        let Some(total) = vertex_count.checked_add(station_vertices) else {
            return Err(vec![LoftViolation {
                code: "mesh_vertex_budget".into(),
                detail: "total station vertex count overflowed".into(),
            }]);
        };
        vertex_count = total;
    }
    let planar_cap_count = (if bow_planar { 1_usize } else { 0_usize })
        + (if stern_planar { 1_usize } else { 0_usize });
    let Some(vertex_count) = vertex_count
        .checked_add(planar_cap_count)
        .filter(|count| *count <= MAX_MESH_VERTICES)
    else {
        return Err(vec![LoftViolation {
            code: "mesh_vertex_budget".into(),
            detail: format!("loft exceeds the {MAX_MESH_VERTICES}-vertex budget"),
        }]);
    };

    let side_faces_per_interval = mirrored_point_count.checked_mul(4);
    let Some(side_faces_per_interval) = side_faces_per_interval else {
        return Err(vec![LoftViolation {
            code: "mesh_face_budget".into(),
            detail: "side face count overflowed".into(),
        }]);
    };
    let side_face_count = (station_count - 1).checked_mul(side_faces_per_interval);
    let Some(side_face_count) = side_face_count else {
        return Err(vec![LoftViolation {
            code: "mesh_face_budget".into(),
            detail: "side face count overflowed".into(),
        }]);
    };
    let waterplane_face_count = (station_count - 1)
        .checked_mul(2)
        .and_then(|count| count.checked_sub(if bow_collapsed { 1_usize } else { 0_usize }))
        .and_then(|count| count.checked_sub(if stern_collapsed { 1_usize } else { 0_usize }));
    let Some(waterplane_face_count) = waterplane_face_count else {
        return Err(vec![LoftViolation {
            code: "mesh_face_budget".into(),
            detail: "waterplane face count underflowed or overflowed".into(),
        }]);
    };
    let cap_faces_per_end = point_count
        .checked_mul(2)
        .and_then(|count| count.checked_sub(1));
    let Some(cap_faces_per_end) = cap_faces_per_end else {
        return Err(vec![LoftViolation {
            code: "mesh_face_budget".into(),
            detail: "end-cap face count overflowed".into(),
        }]);
    };
    let bow_end_cap_face_count = if bow_planar { cap_faces_per_end } else { 0 };
    let stern_end_cap_face_count = if stern_planar { cap_faces_per_end } else { 0 };
    let Some(end_cap_face_count) = bow_end_cap_face_count.checked_add(stern_end_cap_face_count)
    else {
        return Err(vec![LoftViolation {
            code: "mesh_face_budget".into(),
            detail: "combined end-cap face count overflowed".into(),
        }]);
    };
    let Some(face_count) = side_face_count
        .checked_add(waterplane_face_count)
        .and_then(|count| count.checked_add(end_cap_face_count))
        .filter(|count| *count <= MAX_MESH_FACES)
    else {
        return Err(vec![LoftViolation {
            code: "mesh_face_budget".into(),
            detail: format!("loft exceeds the {MAX_MESH_FACES}-face budget"),
        }]);
    };

    if vertex_count > u32::MAX as usize || face_count == 0 {
        return Err(vec![LoftViolation {
            code: "mesh_index_budget".into(),
            detail: "generated mesh cannot be represented by the supported u32 index format".into(),
        }]);
    }

    let mut vertices = Vec::with_capacity(vertex_count);
    let mut starboard = vec![vec![0_u32; point_count]; station_count];
    let mut port = vec![vec![0_u32; point_count]; station_count];

    for (station_index, section) in sections.sections.iter().enumerate() {
        for (point_index, point) in section.keel_to_sheer_m.iter().enumerate() {
            let index = u32::try_from(vertices.len()).map_err(|_| {
                vec![LoftViolation {
                    code: "mesh_index_budget".into(),
                    detail: "vertex index exceeds u32".into(),
                }]
            })?;
            starboard[station_index][point_index] = index;
            vertices.push(*point);
        }
        port[station_index][0] = starboard[station_index][0];
        let collapsed = (station_index == 0 && bow_collapsed)
            || (station_index + 1 == station_count && stern_collapsed);
        if collapsed {
            for point_index in 1..point_count {
                port[station_index][point_index] = starboard[station_index][point_index];
            }
        } else {
            for (point_index, point) in section.keel_to_sheer_m.iter().enumerate().skip(1) {
                let index = u32::try_from(vertices.len()).map_err(|_| {
                    vec![LoftViolation {
                        code: "mesh_index_budget".into(),
                        detail: "vertex index exceeds u32".into(),
                    }]
                })?;
                port[station_index][point_index] = index;
                vertices.push([point[0], -point[1], point[2]]);
            }
        }
    }

    let mut side_faces = Vec::with_capacity(side_face_count);
    let mut waterplane_faces = Vec::with_capacity(waterplane_face_count);

    for station_index in 0..station_count - 1 {
        let interval_enters_collapsed_stern =
            station_index + 1 == station_count - 1 && stern_collapsed;
        for point_index in 0..point_count - 1 {
            let a = starboard[station_index][point_index];
            let b = starboard[station_index + 1][point_index];
            let c = starboard[station_index + 1][point_index + 1];
            let d = starboard[station_index][point_index + 1];

            let port_a = port[station_index][point_index];
            let port_b = port[station_index + 1][point_index];
            let port_c = port[station_index + 1][point_index + 1];
            let port_d = port[station_index][point_index + 1];

            if interval_enters_collapsed_stern {
                // At a collapsed right endpoint the port and starboard
                // station columns share IDs. Flip both side-quad diagonals
                // so their shared seam edge has exactly two incident faces.
                side_faces.push([a, b, d]);
                side_faces.push([b, c, d]);
                side_faces.push([port_a, port_d, port_b]);
                side_faces.push([port_d, port_c, port_b]);
            } else {
                side_faces.push([a, b, c]);
                side_faces.push([a, c, d]);
                side_faces.push([port_a, port_c, port_b]);
                side_faces.push([port_a, port_d, port_c]);
            }
        }

        let port_here = port[station_index][point_count - 1];
        let starboard_here = starboard[station_index][point_count - 1];
        let starboard_next = starboard[station_index + 1][point_count - 1];
        let port_next = port[station_index + 1][point_count - 1];
        let left_collapsed = station_index == 0 && bow_collapsed;
        let right_collapsed = station_index + 1 == station_count - 1 && stern_collapsed;
        match (left_collapsed, right_collapsed) {
            (true, false) => waterplane_faces.push([port_here, starboard_next, port_next]),
            (false, true) => waterplane_faces.push([port_here, starboard_here, starboard_next]),
            (false, false) => {
                waterplane_faces.push([port_here, starboard_here, starboard_next]);
                waterplane_faces.push([port_here, starboard_next, port_next]);
            }
            (true, true) => {
                return Err(vec![LoftViolation {
                    code: "adjacent_collapsed_ends".into(),
                    detail: "an interval cannot connect two collapsed longitudinal endpoints"
                        .into(),
                }]);
            }
        }
    }

    let bow_centre = if bow_planar {
        let index = u32::try_from(vertices.len()).map_err(|_| {
            vec![LoftViolation {
                code: "mesh_index_budget".into(),
                detail: "bow centre index exceeds u32".into(),
            }]
        })?;
        vertices.push(section_centre(&sections.sections[0]));
        Some(index)
    } else {
        None
    };
    let stern_centre = if stern_planar {
        let index = u32::try_from(vertices.len()).map_err(|_| {
            vec![LoftViolation {
                code: "mesh_index_budget".into(),
                detail: "stern centre index exceeds u32".into(),
            }]
        })?;
        vertices.push(section_centre(
            sections
                .sections
                .last()
                .expect("validated sections are non-empty"),
        ));
        Some(index)
    } else {
        None
    };

    let side_face_range = FaceRange {
        first_face: 0,
        face_count: side_faces.len(),
    };
    let waterplane_face_range = FaceRange {
        first_face: side_faces.len(),
        face_count: waterplane_faces.len(),
    };
    let mut faces = Vec::with_capacity(face_count);
    faces.extend(side_faces);
    faces.extend(waterplane_faces);

    let bow_end_cap_face_range = FaceRange {
        first_face: faces.len(),
        face_count: 0,
    };
    if let Some(centre) = bow_centre {
        let ring = {
            let mut points = starboard[0].clone();
            points.extend(port[0].iter().skip(1).rev().copied());
            points
        };
        for index in 0..ring.len() {
            let next = (index + 1) % ring.len();
            faces.push([centre, ring[index], ring[next]]);
        }
    }
    let bow_end_cap_face_range = FaceRange {
        face_count: faces.len() - bow_end_cap_face_range.first_face,
        ..bow_end_cap_face_range
    };

    let stern_end_cap_face_range = FaceRange {
        first_face: faces.len(),
        face_count: 0,
    };
    if let Some(centre) = stern_centre {
        let last = station_count - 1;
        let ring = {
            let mut points = starboard[last].clone();
            points.extend(port[last].iter().skip(1).rev().copied());
            points
        };
        for index in 0..ring.len() {
            let next = (index + 1) % ring.len();
            faces.push([centre, ring[next], ring[index]]);
        }
    }
    let stern_end_cap_face_range = FaceRange {
        face_count: faces.len() - stern_end_cap_face_range.first_face,
        ..stern_end_cap_face_range
    };
    let end_cap_face_range = FaceRange {
        first_face: bow_end_cap_face_range.first_face,
        face_count: bow_end_cap_face_range.face_count + stern_end_cap_face_range.face_count,
    };

    if vertices.len() != vertex_count || faces.len() != face_count {
        return Err(vec![LoftViolation {
            code: "mesh_count_invariant".into(),
            detail: "generated vertex/face counts differ from preflight budget calculation".into(),
        }]);
    }

    let geometry = MeshGeometry {
        units: "m".into(),
        coordinate_frame: sections.coordinate_frame.clone(),
        vertices,
        faces,
    };
    let geometry_digest_blake3 = compute_geometry_digest(&geometry).map_err(|error| {
        vec![LoftViolation {
            code: "geometry_digest".into(),
            detail: format!("could not compute normalized geometry digest: {error}"),
        }]
    })?;
    let candidate = CandidateInput {
        schema_version: 1,
        candidate_id: sections.candidate_id.clone(),
        geometry_digest_blake3: geometry_digest_blake3.clone(),
        geometry,
        profile,
    };
    let admission_report = admit(&candidate);

    let mut artifact = LoftedHullArtifact {
        artifact_schema_version: LOFT_SCHEMA_VERSION,
        generator: "symthaea-marine-waterplane-loft-v2".into(),
        cap_semantics: "closed-displacement-volume-at-design-waterplane-z0".into(),
        source_spec_digest_blake3: sections.source_spec_digest_blake3.clone(),
        sampled_sections_digest_blake3: sections.sampled_sections_digest_blake3.clone(),
        mesh_digest_blake3: geometry_digest_blake3,
        loft_manifest_digest_blake3: String::new(),
        side_face_range,
        waterplane_face_range,
        end_cap_face_range,
        bow_end_cap_face_range,
        stern_end_cap_face_range,
        station_count,
        section_point_count: point_count,
        longitudinal_end_conditions: sections.longitudinal_end_conditions,
        candidate,
        admission_report,
    };
    artifact.loft_manifest_digest_blake3 =
        compute_loft_manifest_digest(&artifact).map_err(|error| {
            vec![LoftViolation {
                code: "loft_manifest_digest".into(),
                detail: format!("could not serialize loft artifact identity: {error}"),
            }]
        })?;
    Ok(artifact)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::section::{
        LongitudinalEndCondition, LongitudinalEndConditions, SectionControlPoint, SectionHullSpec,
        SectionStation, generate_section_artifact,
    };
    use crate::{CoordinateFrame, Disposition};

    fn convex_spec() -> SectionHullSpec {
        let shape = vec![
            SectionControlPoint {
                breadth_fraction: 0.0,
                depth_fraction: 1.0,
            },
            SectionControlPoint {
                breadth_fraction: 0.45,
                depth_fraction: 0.82,
            },
            SectionControlPoint {
                breadth_fraction: 0.82,
                depth_fraction: 0.36,
            },
            SectionControlPoint {
                breadth_fraction: 1.0,
                depth_fraction: 0.0,
            },
        ];
        SectionHullSpec {
            schema_version: 1,
            candidate_id: "loft-fixture-001".into(),
            length_m: 100.0,
            beam_m: 20.0,
            draft_m: 6.0,
            coordinate_frame: CoordinateFrame {
                id: "ship-local".into(),
                axis_convention: "x-forward,y-starboard,z-down".into(),
                handedness: "right-handed".into(),
            },
            samples_per_segment: 4,
            longitudinal_end_conditions: LongitudinalEndConditions::default(),
            stations: vec![
                SectionStation {
                    x_over_length: 0.0,
                    half_breadth_fraction: 0.05,
                    draft_fraction: 0.2,
                    control_points: shape.clone(),
                },
                SectionStation {
                    x_over_length: 0.5,
                    half_breadth_fraction: 1.0,
                    draft_fraction: 1.0,
                    control_points: shape.clone(),
                },
                SectionStation {
                    x_over_length: 1.0,
                    half_breadth_fraction: 0.05,
                    draft_fraction: 0.2,
                    control_points: shape,
                },
            ],
        }
    }

    fn profile() -> AdmissionProfile {
        AdmissionProfile {
            id: "waterplane-capped-displacement-v1".into(),
            require_closed_surface: true,
            minimum_face_area_m2: 1.0e-12,
            minimum_enclosed_volume_m3: 1.0e-12,
            additional_required_checks: vec![],
        }
    }

    fn end_condition_spec(
        bow: LongitudinalEndCondition,
        stern: LongitudinalEndCondition,
    ) -> SectionHullSpec {
        let mut spec = convex_spec();
        spec.candidate_id = format!("end-closure-{bow:?}-{stern:?}");
        spec.schema_version = 2;
        spec.longitudinal_end_conditions = LongitudinalEndConditions { bow, stern };
        if bow == LongitudinalEndCondition::CollapsedCentreline {
            spec.stations[0].half_breadth_fraction = 0.0;
        }
        if stern == LongitudinalEndCondition::CollapsedCentreline {
            let last_station = spec.stations.len() - 1;
            spec.stations[last_station].half_breadth_fraction = 0.0;
        }
        spec
    }

    fn assert_admissible_end_conditions(
        bow: LongitudinalEndCondition,
        stern: LongitudinalEndCondition,
    ) -> LoftedHullArtifact {
        let sections = generate_section_artifact(&end_condition_spec(bow, stern)).unwrap();
        let loft = loft_waterplane_capped_hull(&sections, profile()).unwrap();
        assert_eq!(
            loft.admission_report.disposition,
            Disposition::AdmissibleForProfile
        );
        assert_eq!(loft.admission_report.metrics.boundary_edge_count, 0);
        assert_eq!(loft.admission_report.metrics.non_manifold_edge_count, 0);
        assert_eq!(
            loft.admission_report.metrics.non_manifold_vertex_link_count,
            0
        );
        verify_loft_artifact_integrity(&loft).unwrap();

        let expected_bow_caps = if bow == LongitudinalEndCondition::PlanarEndCap {
            2 * loft.section_point_count - 1
        } else {
            0
        };
        let expected_stern_caps = if stern == LongitudinalEndCondition::PlanarEndCap {
            2 * loft.section_point_count - 1
        } else {
            0
        };
        assert_eq!(loft.bow_end_cap_face_range.face_count, expected_bow_caps);
        assert_eq!(
            loft.stern_end_cap_face_range.face_count,
            expected_stern_caps
        );
        loft
    }

    fn rectangular_spec() -> SectionHullSpec {
        let shape = vec![
            SectionControlPoint {
                breadth_fraction: 0.0,
                depth_fraction: 1.0,
            },
            SectionControlPoint {
                breadth_fraction: 1.0,
                depth_fraction: 1.0,
            },
            SectionControlPoint {
                breadth_fraction: 1.0,
                depth_fraction: 0.0,
            },
        ];
        SectionHullSpec {
            schema_version: 1,
            candidate_id: "rectangular-prism-known-answer-001".into(),
            length_m: 100.0,
            beam_m: 20.0,
            draft_m: 6.0,
            coordinate_frame: CoordinateFrame {
                id: "ship-local".into(),
                axis_convention: "x-forward,y-starboard,z-down".into(),
                handedness: "right-handed".into(),
            },
            samples_per_segment: 1,
            longitudinal_end_conditions: LongitudinalEndConditions::default(),
            stations: vec![
                SectionStation {
                    x_over_length: 0.0,
                    half_breadth_fraction: 1.0,
                    draft_fraction: 1.0,
                    control_points: shape.clone(),
                },
                SectionStation {
                    x_over_length: 1.0,
                    half_breadth_fraction: 1.0,
                    draft_fraction: 1.0,
                    control_points: shape,
                },
            ],
        }
    }

    fn prismatic_artifact() -> SectionFamilyArtifact {
        let spec = SectionHullSpec {
            schema_version: 1,
            candidate_id: "prismatic-known-answer-001".into(),
            length_m: 100.0,
            beam_m: 20.0,
            draft_m: 6.0,
            coordinate_frame: CoordinateFrame {
                id: "ship-local".into(),
                axis_convention: "x-forward,y-starboard,z-down".into(),
                handedness: "right-handed".into(),
            },
            samples_per_segment: 1,
            longitudinal_end_conditions: LongitudinalEndConditions::default(),
            stations: vec![
                SectionStation {
                    x_over_length: 0.0,
                    half_breadth_fraction: 1.0,
                    draft_fraction: 1.0,
                    control_points: vec![
                        SectionControlPoint {
                            breadth_fraction: 0.0,
                            depth_fraction: 1.0,
                        },
                        SectionControlPoint {
                            breadth_fraction: 1.0,
                            depth_fraction: 0.0,
                        },
                    ],
                },
                SectionStation {
                    x_over_length: 1.0,
                    half_breadth_fraction: 1.0,
                    draft_fraction: 1.0,
                    control_points: vec![
                        SectionControlPoint {
                            breadth_fraction: 0.0,
                            depth_fraction: 1.0,
                        },
                        SectionControlPoint {
                            breadth_fraction: 1.0,
                            depth_fraction: 0.0,
                        },
                    ],
                },
            ],
        };
        generate_section_artifact(&spec).unwrap()
    }

    fn artifact() -> SectionFamilyArtifact {
        generate_section_artifact(&convex_spec()).unwrap()
    }

    fn rebind_loft_after_test_mutation(loft: &mut LoftedHullArtifact) {
        let digest = crate::compute_geometry_digest(&loft.candidate.geometry).unwrap();
        loft.candidate.geometry_digest_blake3 = digest.clone();
        loft.mesh_digest_blake3 = digest;
        loft.admission_report = crate::admit(&loft.candidate);
        loft.loft_manifest_digest_blake3 = compute_loft_manifest_digest(loft).unwrap();
    }

    #[test]
    fn convex_section_family_lofts_to_closed_admissible_mesh() {
        let result = loft_waterplane_capped_hull(&artifact(), profile()).unwrap();
        assert_eq!(
            result.admission_report.disposition,
            Disposition::AdmissibleForProfile
        );
        assert_eq!(result.candidate.geometry.units, "m");
        assert_eq!(
            result.cap_semantics,
            "closed-displacement-volume-at-design-waterplane-z0"
        );
        assert_eq!(result.admission_report.metrics.boundary_edge_count, 0);
        assert_eq!(result.admission_report.metrics.non_manifold_edge_count, 0);
        assert_eq!(
            result
                .admission_report
                .metrics
                .non_manifold_vertex_link_count,
            0
        );
        assert_eq!(result.side_face_range.first_face, 0);
        assert_eq!(
            result.waterplane_face_range.first_face,
            result.side_face_range.face_count
        );
        assert_eq!(
            result.end_cap_face_range.first_face,
            result.waterplane_face_range.first_face + result.waterplane_face_range.face_count
        );
        for face in result.candidate.geometry.faces[result.waterplane_face_range.first_face
            ..result.waterplane_face_range.first_face + result.waterplane_face_range.face_count]
            .iter()
        {
            assert!(face.iter().all(|index| {
                result.candidate.geometry.vertices[*index as usize][2].abs() < 1.0e-12
            }));
        }
        assert!(result.admission_report.metrics.signed_volume_m3.unwrap() > 0.0);
        assert_eq!(
            result.mesh_digest_blake3,
            result.candidate.geometry_digest_blake3
        );
    }

    #[test]
    fn triangular_prismatic_displacement_volume_matches_known_answer() {
        let result = loft_waterplane_capped_hull(&prismatic_artifact(), profile()).unwrap();
        assert_eq!(result.candidate.geometry.vertices.len(), 8);
        assert_eq!(result.candidate.geometry.faces.len(), 12);
        let volume = result.admission_report.metrics.signed_volume_m3.unwrap();
        assert!((volume - 6_000.0).abs() < 1.0e-9);
        assert_eq!(
            result.admission_report.disposition,
            Disposition::AdmissibleForProfile
        );
    }

    #[test]
    fn legacy_v1_loft_parses_but_fails_v2_integrity_admission() {
        let loft = loft_waterplane_capped_hull(&artifact(), profile()).unwrap();
        let mut legacy = serde_json::to_value(&loft).unwrap();
        let object = legacy.as_object_mut().unwrap();
        object.insert("artifact_schema_version".into(), serde_json::json!(1));
        object.insert(
            "generator".into(),
            serde_json::json!("symthaea-marine-waterplane-loft-v1"),
        );
        object.remove("bow_end_cap_face_range");
        object.remove("stern_end_cap_face_range");
        object.remove("station_count");
        object.remove("section_point_count");
        object.remove("longitudinal_end_conditions");
        let parsed: LoftedHullArtifact = serde_json::from_value(legacy).unwrap();
        let errors = verify_loft_artifact_integrity(&parsed).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|item| item.code == "loft_artifact_version")
        );
    }

    #[test]
    fn face_ranges_are_bound_to_the_loft_manifest_digest() {
        let mut result = loft_waterplane_capped_hull(&artifact(), profile()).unwrap();
        assert!(verify_loft_artifact_integrity(&result).is_ok());
        result.waterplane_face_range.face_count += 1;
        let error = verify_loft_artifact_integrity(&result).unwrap_err();
        assert!(error.iter().any(|item| item.code == "loft_manifest_digest"));
    }

    #[test]
    fn admission_profile_and_report_are_bound_to_the_loft_manifest() {
        let mut result = loft_waterplane_capped_hull(&artifact(), profile()).unwrap();
        result.candidate.profile.minimum_face_area_m2 *= 2.0;
        result.admission_report = admit(&result.candidate);
        let error = verify_loft_artifact_integrity(&result).unwrap_err();
        assert!(error.iter().any(|item| item.code == "loft_manifest_digest"));
    }

    #[test]
    fn pointed_bow_with_planar_stern_is_manifold_and_admissible() {
        let loft = assert_admissible_end_conditions(
            LongitudinalEndCondition::CollapsedCentreline,
            LongitudinalEndCondition::PlanarEndCap,
        );
        assert_eq!(loft.waterplane_face_range.face_count, 3);
        assert_eq!(loft.bow_end_cap_face_range.face_count, 0);
        assert!(loft.end_cap_face_range.face_count > 0);
    }

    #[test]
    fn planar_bow_with_pointed_stern_is_manifold_and_admissible() {
        let loft = assert_admissible_end_conditions(
            LongitudinalEndCondition::PlanarEndCap,
            LongitudinalEndCondition::CollapsedCentreline,
        );
        assert_eq!(loft.waterplane_face_range.face_count, 3);
        assert_eq!(loft.stern_end_cap_face_range.face_count, 0);
        assert!(loft.end_cap_face_range.face_count > 0);
    }

    #[test]
    fn both_pointed_ends_share_centreline_vertices_without_end_caps() {
        let loft = assert_admissible_end_conditions(
            LongitudinalEndCondition::CollapsedCentreline,
            LongitudinalEndCondition::CollapsedCentreline,
        );
        assert_eq!(loft.waterplane_face_range.face_count, 2);
        assert_eq!(loft.end_cap_face_range.face_count, 0);

        let point_count = loft.section_point_count;
        for station_index in [0, loft.station_count - 1] {
            let base = if station_index == 0 {
                0
            } else {
                (0..loft.station_count - 1).fold(0_usize, |offset, index| {
                    let collapsed = index == 0;
                    offset + point_count + if collapsed { 0 } else { point_count - 1 }
                })
            };
            let endpoint_vertices = &loft.candidate.geometry.vertices[base..base + point_count];
            assert!(endpoint_vertices.iter().all(|point| point[1] == 0.0));
        }
    }

    #[test]
    fn planar_planar_legacy_end_conditions_remain_supported() {
        let loft = assert_admissible_end_conditions(
            LongitudinalEndCondition::PlanarEndCap,
            LongitudinalEndCondition::PlanarEndCap,
        );
        assert_eq!(
            loft.waterplane_face_range.face_count,
            2 * (loft.station_count - 1)
        );
        assert_eq!(
            loft.end_cap_face_range.face_count,
            2 * (2 * loft.section_point_count - 1)
        );
    }

    #[test]
    fn collapsed_interior_station_is_rejected() {
        let mut spec = convex_spec();
        spec.stations[1].half_breadth_fraction = 0.0;
        let error = generate_section_artifact(&spec).unwrap_err();
        assert!(
            error
                .iter()
                .any(|item| item.code == "interior_station_breadth")
        );
    }

    #[test]
    fn zero_breadth_endpoint_with_planar_cap_label_is_rejected() {
        let mut spec = convex_spec();
        spec.stations[0].half_breadth_fraction = 0.0;
        let error = generate_section_artifact(&spec).unwrap_err();
        assert!(error.iter().any(|item| item.code == "planar_end_breadth"));
    }

    #[test]
    fn positive_breadth_endpoint_with_collapsed_label_is_rejected() {
        let mut spec = end_condition_spec(
            LongitudinalEndCondition::CollapsedCentreline,
            LongitudinalEndCondition::PlanarEndCap,
        );
        spec.stations[0].half_breadth_fraction = 0.01;
        let error = generate_section_artifact(&spec).unwrap_err();
        assert!(
            error
                .iter()
                .any(|item| item.code == "collapsed_end_breadth")
        );
    }

    #[test]
    fn two_adjacent_collapsed_ends_are_rejected() {
        let mut spec = end_condition_spec(
            LongitudinalEndCondition::CollapsedCentreline,
            LongitudinalEndCondition::CollapsedCentreline,
        );
        spec.stations.truncate(2);
        spec.stations[1].x_over_length = 1.0;
        spec.stations[1].half_breadth_fraction = 0.0;
        let error = generate_section_artifact(&spec).unwrap_err();
        assert!(
            error
                .iter()
                .any(|item| item.code == "adjacent_collapsed_ends")
        );
    }

    #[test]
    fn mutated_end_condition_without_manifest_rebinding_is_rejected() {
        let mut loft = assert_admissible_end_conditions(
            LongitudinalEndCondition::CollapsedCentreline,
            LongitudinalEndCondition::PlanarEndCap,
        );
        loft.longitudinal_end_conditions.bow = LongitudinalEndCondition::PlanarEndCap;
        let errors = verify_loft_artifact_integrity(&loft).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|item| item.code == "loft_manifest_digest")
        );
    }

    #[test]
    fn rehashed_wrong_side_face_is_rejected_by_shared_vertex_semantics() {
        let mut loft = assert_admissible_end_conditions(
            LongitudinalEndCondition::PlanarEndCap,
            LongitudinalEndCondition::CollapsedCentreline,
        );
        let face_index = loft.side_face_range.first_face;
        loft.candidate.geometry.faces[face_index][0] = loft.candidate.geometry.faces[face_index][1];
        rebind_loft_after_test_mutation(&mut loft);
        let errors = verify_loft_artifact_integrity(&loft).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|item| item.code == "side_closure_semantics")
        );
    }

    #[test]
    fn rehashed_wrong_waterplane_triangle_is_rejected_by_closure_semantics() {
        let mut loft = assert_admissible_end_conditions(
            LongitudinalEndCondition::CollapsedCentreline,
            LongitudinalEndCondition::PlanarEndCap,
        );
        let face_index = loft.waterplane_face_range.first_face;
        loft.candidate.geometry.faces[face_index][0] = loft.candidate.geometry.faces[face_index][1];
        rebind_loft_after_test_mutation(&mut loft);
        let errors = verify_loft_artifact_integrity(&loft).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|item| item.code == "waterplane_closure_semantics")
        );
    }

    #[test]
    fn rehashed_wrong_planar_end_cap_triangle_is_rejected() {
        let mut loft = assert_admissible_end_conditions(
            LongitudinalEndCondition::PlanarEndCap,
            LongitudinalEndCondition::PlanarEndCap,
        );
        let face_index = loft.bow_end_cap_face_range.first_face;
        loft.candidate.geometry.faces[face_index][0] = loft.candidate.geometry.faces[face_index][1];
        rebind_loft_after_test_mutation(&mut loft);
        let errors = verify_loft_artifact_integrity(&loft).unwrap_err();
        assert!(errors.iter().any(|item| item.code == "bow_cap_topology"));
    }

    #[test]
    fn collapsed_endpoint_must_not_duplicate_transverse_vertices() {
        let mut loft = assert_admissible_end_conditions(
            LongitudinalEndCondition::CollapsedCentreline,
            LongitudinalEndCondition::PlanarEndCap,
        );
        let starboard_keel = loft.candidate.geometry.vertices[0];
        loft.candidate.geometry.vertices[0][1] = 0.25;
        assert_ne!(loft.candidate.geometry.vertices[0], starboard_keel);
        // Rebinding the geometry digest cannot make invalid collapsed-end semantics pass.
        let digest = crate::compute_geometry_digest(&loft.candidate.geometry).unwrap();
        loft.candidate.geometry_digest_blake3 = digest.clone();
        loft.mesh_digest_blake3 = digest;
        loft.admission_report = crate::admit(&loft.candidate);
        loft.loft_manifest_digest_blake3 = compute_loft_manifest_digest(&loft).unwrap();
        let errors = verify_loft_artifact_integrity(&loft).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|item| item.code == "collapsed_end_geometry")
        );
    }

    #[test]
    fn rectangular_prism_mesh_matches_known_volume_and_waterplane() {
        let source = generate_section_artifact(&rectangular_spec()).unwrap();
        let result = loft_waterplane_capped_hull(&source, profile()).unwrap();
        assert_eq!(
            result.admission_report.disposition,
            Disposition::AdmissibleForProfile
        );
        assert!(
            (result.admission_report.metrics.signed_volume_m3.unwrap() - 12_000.0).abs() < 1.0e-9
        );
        assert_eq!(result.admission_report.metrics.boundary_edge_count, 0);
        assert_eq!(result.admission_report.metrics.non_manifold_edge_count, 0);
        assert_eq!(
            result
                .admission_report
                .metrics
                .non_manifold_vertex_link_count,
            0
        );
        assert!(verify_loft_artifact_integrity(&result).is_ok());
    }

    #[test]
    fn loft_is_deterministic_for_identical_section_artifacts() {
        let source = artifact();
        let first = loft_waterplane_capped_hull(&source, profile()).unwrap();
        let second = loft_waterplane_capped_hull(&source, profile()).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn sample_digest_mismatch_is_rejected() {
        let mut source = artifact();
        source.sections[1].keel_to_sheer_m[1][1] += 0.1;
        let error = loft_waterplane_capped_hull(&source, profile()).unwrap_err();
        assert!(error.iter().any(|item| item.code == "sample_digest"));
    }

    #[test]
    fn inconsistent_section_point_counts_are_rejected() {
        let mut source = artifact();
        source.sections[1].keel_to_sheer_m.pop();
        source.sampled_sections_digest_blake3 =
            compute_sampled_sections_digest(&source.sections).unwrap();
        let error = loft_waterplane_capped_hull(&source, profile()).unwrap_err();
        assert!(
            error
                .iter()
                .any(|item| item.code == "section_point_count_mismatch")
        );
    }

    #[test]
    fn non_convex_section_is_rejected_without_healing() {
        let mut source = artifact();
        source.sections[1].keel_to_sheer_m[5][1] = 4.6;
        source.sampled_sections_digest_blake3 =
            compute_sampled_sections_digest(&source.sections).unwrap();
        let error = loft_waterplane_capped_hull(&source, profile()).unwrap_err();
        assert!(error.iter().any(|item| item.code == "non_convex_section"));
    }

    #[test]
    fn non_monotonic_station_coordinates_are_rejected() {
        let mut source = artifact();
        source.sections[1].x_m = source.sections[0].x_m;
        let station_x = source.sections[1].x_m;
        for point in &mut source.sections[1].keel_to_sheer_m {
            point[0] = station_x;
        }
        source.sampled_sections_digest_blake3 =
            compute_sampled_sections_digest(&source.sections).unwrap();
        let error = loft_waterplane_capped_hull(&source, profile()).unwrap_err();
        assert!(error.iter().any(|item| item.code == "station_order"));
    }

    #[test]
    fn open_surface_admission_profile_is_rejected() {
        let mut invalid_profile = profile();
        invalid_profile.require_closed_surface = false;
        let error = loft_waterplane_capped_hull(&artifact(), invalid_profile).unwrap_err();
        assert!(
            error
                .iter()
                .any(|item| item.code == "profile_requires_closure")
        );
    }
}
