//! G2b deterministic, convex-section waterplane-capped surface construction.
//!
//! The result is a closed displacement-volume mesh bounded by the design
//! waterplane. It is not a deck/freeboard model or a general-purpose CAD hull.

use serde::{Deserialize, Serialize};

use crate::section::{compute_sampled_sections_digest, SectionFamilyArtifact};
use crate::{
    admit, compute_geometry_digest, AdmissionProfile, AdmissionReport, CandidateInput,
    MeshGeometry,
};

const LOFT_SCHEMA_VERSION: u32 = 1;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
        candidate_schema_version: artifact.candidate.schema_version,
        candidate_id: &artifact.candidate.candidate_id,
        geometry_digest_blake3: &artifact.candidate.geometry_digest_blake3,
        admission_profile: &artifact.candidate.profile,
    };
    let bytes = serde_json::to_vec(&identity)?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}


fn violation(
    violations: &mut Vec<LoftViolation>,
    code: &str,
    detail: impl Into<String>,
) {
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

fn validate_section_family(
    artifact: &SectionFamilyArtifact,
) -> Result<usize, Vec<LoftViolation>> {
    let mut violations = Vec::new();

    if artifact.artifact_schema_version != 1 {
        violation(
            &mut violations,
            "section_artifact_version",
            "only section artifact schema version 1 is supported",
        );
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
                    format!("station {station_index} contains a point outside its declared x-plane"),
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
                    format!("station {station_index}, point {point_index} lies outside the declared beam/draft envelope"),
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
                format!("station {station_index} must begin on the centreline below the waterplane"),
            );
        }
        if waterline[1] <= coordinate_tolerance
            || !close_enough(waterline[2], 0.0, coordinate_tolerance)
        {
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
        let area_tolerance = coordinate_tolerance * coordinate_tolerance;
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

    if artifact.artifact_schema_version != LOFT_SCHEMA_VERSION
        || artifact.generator != "symthaea-marine-waterplane-loft-v1"
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
            detail: "waterplane-capped displacement meshes require a closed-surface admission profile".into(),
        }]);
    }

    let vertices_per_station = point_count
        .checked_mul(2)
        .and_then(|count| count.checked_sub(1));
    let Some(vertices_per_station) = vertices_per_station else {
        return Err(vec![LoftViolation {
            code: "mesh_vertex_budget".into(),
            detail: "mirrored vertex count overflowed".into(),
        }]);
    };
    let vertex_count = station_count
        .checked_mul(vertices_per_station)
        .and_then(|count| count.checked_add(2));
    let Some(vertex_count) = vertex_count.filter(|count| *count <= MAX_MESH_VERTICES) else {
        return Err(vec![LoftViolation {
            code: "mesh_vertex_budget".into(),
            detail: format!("loft exceeds the {MAX_MESH_VERTICES}-vertex budget"),
        }]);
    };

    let interval_face_count = point_count
        .checked_sub(1)
        .and_then(|count| count.checked_mul(4))
        .and_then(|count| count.checked_add(2));
    let Some(interval_face_count) = interval_face_count else {
        return Err(vec![LoftViolation {
            code: "mesh_face_budget".into(),
            detail: "face count overflowed".into(),
        }]);
    };
    let end_cap_face_count = point_count.checked_mul(4).and_then(|count| count.checked_sub(2));
    let Some(end_cap_face_count) = end_cap_face_count else {
        return Err(vec![LoftViolation {
            code: "mesh_face_budget".into(),
            detail: "end-cap face count overflowed".into(),
        }]);
    };
    let face_count = (station_count - 1)
        .checked_mul(interval_face_count)
        .and_then(|count| count.checked_add(end_cap_face_count));
    let Some(face_count) = face_count.filter(|count| *count <= MAX_MESH_FACES) else {
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

    let side_face_count = (station_count - 1) * 4 * (point_count - 1);
    let waterplane_face_count = (station_count - 1) * 2;
    let mut side_faces = Vec::with_capacity(side_face_count);
    let mut waterplane_faces = Vec::with_capacity(waterplane_face_count);

    for station_index in 0..station_count - 1 {
        for point_index in 0..point_count - 1 {
            let a = starboard[station_index][point_index];
            let b = starboard[station_index + 1][point_index];
            let c = starboard[station_index + 1][point_index + 1];
            let d = starboard[station_index][point_index + 1];
            side_faces.push([a, b, c]);
            side_faces.push([a, c, d]);

            let a = port[station_index][point_index];
            let b = port[station_index + 1][point_index];
            let c = port[station_index + 1][point_index + 1];
            let d = port[station_index][point_index + 1];
            side_faces.push([a, c, b]);
            side_faces.push([a, d, c]);
        }

        let port_here = port[station_index][point_count - 1];
        let starboard_here = starboard[station_index][point_count - 1];
        let starboard_next = starboard[station_index + 1][point_count - 1];
        let port_next = port[station_index + 1][point_count - 1];
        waterplane_faces.push([port_here, starboard_here, starboard_next]);
        waterplane_faces.push([port_here, starboard_next, port_next]);
    }

    let bow_ring = {
        let mut ring = starboard[0].clone();
        ring.extend(port[0].iter().skip(1).rev().copied());
        ring
    };
    let stern_ring = {
        let last = station_count - 1;
        let mut ring = starboard[last].clone();
        ring.extend(port[last].iter().skip(1).rev().copied());
        ring
    };

    let bow_centre = u32::try_from(vertices.len()).map_err(|_| {
        vec![LoftViolation {
            code: "mesh_index_budget".into(),
            detail: "bow centre index exceeds u32".into(),
        }]
    })?;
    vertices.push(section_centre(&sections.sections[0]));
    let stern_centre = u32::try_from(vertices.len()).map_err(|_| {
        vec![LoftViolation {
            code: "mesh_index_budget".into(),
            detail: "stern centre index exceeds u32".into(),
        }]
    })?;
    vertices.push(section_centre(
        sections.sections.last().expect("validated sections are non-empty"),
    ));

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

    let end_cap_face_range = FaceRange {
        first_face: faces.len(),
        face_count: end_cap_face_count,
    };
    for index in 0..bow_ring.len() {
        let next = (index + 1) % bow_ring.len();
        faces.push([bow_centre, bow_ring[index], bow_ring[next]]);
    }
    for index in 0..stern_ring.len() {
        let next = (index + 1) % stern_ring.len();
        faces.push([stern_centre, stern_ring[next], stern_ring[index]]);
    }

    if vertices.len() != vertex_count || faces.len() != face_count {
        return Err(vec![LoftViolation {
            code: "mesh_count_invariant",
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
            code: "geometry_digest",
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
        generator: "symthaea-marine-waterplane-loft-v1".into(),
        cap_semantics: "closed-displacement-volume-at-design-waterplane-z0".into(),
        source_spec_digest_blake3: sections.source_spec_digest_blake3.clone(),
        sampled_sections_digest_blake3: sections.sampled_sections_digest_blake3.clone(),
        mesh_digest_blake3: geometry_digest_blake3,
        loft_manifest_digest_blake3: String::new(),
        side_face_range,
        waterplane_face_range,
        end_cap_face_range,
        candidate,
        admission_report,
    };
    artifact.loft_manifest_digest_blake3 = compute_loft_manifest_digest(&artifact)
        .map_err(|error| {
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
        generate_section_artifact, SectionControlPoint, SectionHullSpec, SectionStation,
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

    #[test]
    fn convex_section_family_lofts_to_closed_admissible_mesh() {
        let result = loft_waterplane_capped_hull(&artifact(), profile()).unwrap();
        assert_eq!(
            result.admission_report.disposition,
            Disposition::AdmissibleForProfile
        );
        assert_eq!(result.candidate.geometry.units, "m");
        assert_eq!(result.cap_semantics, "closed-displacement-volume-at-design-waterplane-z0");
        assert_eq!(result.admission_report.metrics.boundary_edge_count, 0);
        assert_eq!(result.admission_report.metrics.non_manifold_edge_count, 0);
        assert_eq!(result.admission_report.metrics.non_manifold_vertex_link_count, 0);
        assert_eq!(result.side_face_range.first_face, 0);
        assert_eq!(
            result.waterplane_face_range.first_face,
            result.side_face_range.face_count
        );
        assert_eq!(
            result.end_cap_face_range.first_face,
            result.waterplane_face_range.first_face + result.waterplane_face_range.face_count
        );
        for face in result.candidate.geometry.faces
            [result.waterplane_face_range.first_face
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
        assert!(error
            .iter()
            .any(|item| item.code == "section_point_count_mismatch"));
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
        for point in &mut source.sections[1].keel_to_sheer_m {
            point[0] = source.sections[1].x_m;
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
        assert!(error
            .iter()
            .any(|item| item.code == "profile_requires_closure"));
    }
}
