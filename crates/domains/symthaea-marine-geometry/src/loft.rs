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
    pub candidate: CandidateInput,
    pub admission_report: AdmissionReport,
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

    let mut faces = Vec::with_capacity(face_count);

    for station_index in 0..station_count - 1 {
        for point_index in 0..point_count - 1 {
            let a = starboard[station_index][point_index];
            let b = starboard[station_index + 1][point_index];
            let c = starboard[station_index + 1][point_index + 1];
            let d = starboard[station_index][point_index + 1];
            faces.push([a, b, c]);
            faces.push([a, c, d]);

            let a = port[station_index][point_index];
            let b = port[station_index + 1][point_index];
            let c = port[station_index + 1][point_index + 1];
            let d = port[station_index][point_index + 1];
            faces.push([a, c, b]);
            faces.push([a, d, c]);
        }

        let port_here = port[station_index][point_count - 1];
        let starboard_here = starboard[station_index][point_count - 1];
        let starboard_next = starboard[station_index + 1][point_count - 1];
        let port_next = port[station_index + 1][point_count - 1];
        faces.push([port_here, starboard_here, starboard_next]);
        faces.push([port_here, starboard_next, port_next]);
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

    Ok(LoftedHullArtifact {
        artifact_schema_version: LOFT_SCHEMA_VERSION,
        generator: "symthaea-marine-waterplane-loft-v1".into(),
        cap_semantics: "closed-displacement-volume-at-design-waterplane-z0".into(),
        source_spec_digest_blake3: sections.source_spec_digest_blake3.clone(),
        sampled_sections_digest_blake3: sections.sampled_sections_digest_blake3.clone(),
        mesh_digest_blake3: geometry_digest_blake3,
        candidate,
        admission_report,
    })
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
        assert!(result.admission_report.metrics.signed_volume_m3.unwrap() > 0.0);
        assert_eq!(
            result.mesh_digest_blake3,
            result.candidate.geometry_digest_blake3
        );
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
        source.sections[1].keel_to_sheer_m[5][2] = 5.9;
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
