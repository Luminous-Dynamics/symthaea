//! Deterministic, deliberately scoped marine geometry admission.
//!
//! This crate validates a versioned triangle-mesh interchange artifact. It is
//! not a CAD kernel, hydrostatics solver, self-intersection detector, DFM
//! checker, CFD solver, or vessel-approval system.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const SCHEMA_VERSION: u32 = 1;
const ADMISSION_SCOPE: &str = "triangle-mesh-integrity-v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateInput {
    pub schema_version: u32,
    pub candidate_id: String,
    /// BLAKE3 of the canonical serde_json serialization of the geometry struct.
    /// Use the CLI digest-only mode to compute it.
    pub geometry_digest_blake3: String,
    pub geometry: MeshGeometry,
    pub profile: AdmissionProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshGeometry {
    /// This first adapter accepts metres only; no implicit conversion occurs.
    pub units: String,
    pub coordinate_frame: CoordinateFrame,
    /// Vertices in the declared coordinate frame, in metres.
    pub vertices: Vec<[f64; 3]>,
    /// Zero-based, consistently wound triangle indices.
    pub faces: Vec<[u32; 3]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoordinateFrame {
    pub id: String,
    /// Human-readable axis semantics. No axis transform is inferred.
    pub axis_convention: String,
    /// Signed-volume interpretation currently requires right-handed.
    pub handedness: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmissionProfile {
    pub id: String,
    pub require_closed_surface: bool,
    pub minimum_face_area_m2: f64,
    pub minimum_enclosed_volume_m3: f64,
    /// Checks not implemented by this adapter remain UNKNOWN and force
    /// INCOMPLETE; they are never silently treated as passes.
    pub additional_required_checks: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CheckStatus {
    Pass,
    Fail,
    NotRequired,
    NotEvaluated,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Check {
    pub code: String,
    pub status: CheckStatus,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Disposition {
    AdmissibleForProfile,
    Invalid,
    Incomplete,
    Incomparable,
    Blocked,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometryMetrics {
    pub vertex_count: usize,
    pub face_count: usize,
    pub boundary_edge_count: usize,
    pub non_manifold_edge_count: usize,
    pub inconsistent_winding_edge_count: usize,
    pub duplicate_face_count: usize,
    pub degenerate_face_count: usize,
    pub signed_volume_m3: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmissionReport {
    pub report_schema_version: u32,
    pub admission_scope: String,
    pub candidate_id: String,
    pub geometry_digest_blake3: String,
    pub profile_id: String,
    pub disposition: Disposition,
    pub checks: Vec<Check>,
    pub metrics: GeometryMetrics,
}

/// Hashes normalized geometry data, not the original JSON whitespace.
/// Struct field order is fixed by the Rust schema; unknown input fields are denied.
pub fn compute_geometry_digest(geometry: &MeshGeometry) -> Result<String, serde_json::Error> {
    let canonical = serde_json::to_vec(geometry)?;
    Ok(blake3::hash(&canonical).to_hex().to_string())
}

fn push(
    checks: &mut Vec<Check>,
    code: &str,
    status: CheckStatus,
    detail: impl Into<String>,
) {
    checks.push(Check {
        code: code.to_owned(),
        status,
        detail: detail.into(),
    });
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn triangle_area(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let normal = cross(sub(b, a), sub(c, a));
    0.5 * dot(normal, normal).sqrt()
}

fn signed_volume(geometry: &MeshGeometry) -> Option<f64> {
    let mut six_volume = 0.0_f64;
    for face in &geometry.faces {
        let a = *geometry.vertices.get(face[0] as usize)?;
        let b = *geometry.vertices.get(face[1] as usize)?;
        let c = *geometry.vertices.get(face[2] as usize)?;
        six_volume += dot(a, cross(b, c));
    }
    let volume = six_volume / 6.0;
    volume.is_finite().then_some(volume)
}

/// Validates the candidate against its declared mesh-integrity profile.
///
/// AdmissibleForProfile means only that all checks implemented by this
/// triangle-mesh adapter passed and no additional unsupported check was
/// required. It does not imply hull feasibility, self-intersection freedom,
/// hydrostatic suitability, manufacturability, structural adequacy, or
/// hydrodynamic performance.
pub fn admit(candidate: &CandidateInput) -> AdmissionReport {
    let mut checks = Vec::new();
    let mut invalid = false;
    let mut incomparable = false;
    let mut blocked = false;
    let mut incomplete = false;

    if candidate.candidate_id.trim().is_empty() {
        invalid = true;
        push(
            &mut checks,
            "candidate_id",
            CheckStatus::Fail,
            "candidate_id is empty",
        );
    } else {
        push(
            &mut checks,
            "candidate_id",
            CheckStatus::Pass,
            "candidate_id is present",
        );
    }

    if candidate.schema_version == SCHEMA_VERSION {
        push(
            &mut checks,
            "schema_version",
            CheckStatus::Pass,
            "schema version 1",
        );
    } else {
        incomparable = true;
        push(
            &mut checks,
            "schema_version",
            CheckStatus::Fail,
            format!("schema version {} is not supported", candidate.schema_version),
        );
    }

    let frame = &candidate.geometry.coordinate_frame;
    if candidate.geometry.units != "m"
        || frame.id.trim().is_empty()
        || frame.axis_convention.trim().is_empty()
        || frame.handedness != "right-handed"
    {
        incomparable = true;
        push(
            &mut checks,
            "units_and_frame",
            CheckStatus::Fail,
            "requires units=m, non-empty frame id and axis convention, and right-handed coordinates; no implicit conversion is performed",
        );
    } else {
        push(
            &mut checks,
            "units_and_frame",
            CheckStatus::Pass,
            "metres and an explicit right-handed coordinate frame are declared",
        );
    }

    let profile_valid = candidate.profile.minimum_face_area_m2.is_finite()
        && candidate.profile.minimum_face_area_m2 > 0.0
        && candidate.profile.minimum_enclosed_volume_m3.is_finite()
        && candidate.profile.minimum_enclosed_volume_m3 >= 0.0
        && !candidate.profile.id.trim().is_empty();
    if profile_valid {
        push(
            &mut checks,
            "profile_parameters",
            CheckStatus::Pass,
            "profile parameters are finite and within supported ranges",
        );
    } else {
        invalid = true;
        push(
            &mut checks,
            "profile_parameters",
            CheckStatus::Fail,
            "profile id or geometric thresholds are invalid",
        );
    }

    let finite_vertices = candidate
        .geometry
        .vertices
        .iter()
        .all(|v| v.iter().all(|x| x.is_finite()));
    if finite_vertices {
        push(
            &mut checks,
            "finite_coordinates",
            CheckStatus::Pass,
            "all vertex coordinates are finite",
        );
    } else {
        invalid = true;
        push(
            &mut checks,
            "finite_coordinates",
            CheckStatus::Fail,
            "one or more vertex coordinates are non-finite",
        );
    }

    match compute_geometry_digest(&candidate.geometry) {
        Ok(actual) if actual.eq_ignore_ascii_case(&candidate.geometry_digest_blake3) => {
            push(
                &mut checks,
                "geometry_digest",
                CheckStatus::Pass,
                "geometry digest matches normalized geometry",
            );
        }
        Ok(actual) => {
            invalid = true;
            push(
                &mut checks,
                "geometry_digest",
                CheckStatus::Fail,
                format!("digest mismatch; computed digest is {actual}"),
            );
        }
        Err(error) => {
            invalid = true;
            push(
                &mut checks,
                "geometry_digest",
                CheckStatus::Fail,
                format!("geometry cannot be serialized canonically: {error}"),
            );
        }
    }

    let vertices = &candidate.geometry.vertices;
    let faces = &candidate.geometry.faces;
    let mut indices_valid = !faces.is_empty();
    let mut duplicate_faces = 0_usize;
    let mut degenerate_faces = 0_usize;
    let mut seen_faces = BTreeSet::new();
    let mut edges: BTreeMap<(u32, u32), Vec<i8>> = BTreeMap::new();

    for face in faces {
        let [a, b, c] = *face;
        if a as usize >= vertices.len()
            || b as usize >= vertices.len()
            || c as usize >= vertices.len()
            || a == b
            || b == c
            || c == a
        {
            indices_valid = false;
            continue;
        }

        let mut key = [a, b, c];
        key.sort_unstable();
        if !seen_faces.insert(key) {
            duplicate_faces += 1;
        }

        let area = triangle_area(
            vertices[a as usize],
            vertices[b as usize],
            vertices[c as usize],
        );
        if !area.is_finite() || area < candidate.profile.minimum_face_area_m2 {
            degenerate_faces += 1;
        }

        for (from, to) in [(a, b), (b, c), (c, a)] {
            let (edge, direction) = if from < to {
                ((from, to), 1_i8)
            } else {
                ((to, from), -1_i8)
            };
            edges.entry(edge).or_default().push(direction);
        }
    }

    if indices_valid {
        push(
            &mut checks,
            "face_indices",
            CheckStatus::Pass,
            format!(
                "{} non-empty faces reference valid, distinct vertices",
                faces.len()
            ),
        );
    } else {
        invalid = true;
        push(
            &mut checks,
            "face_indices",
            CheckStatus::Fail,
            "faces are empty or contain out-of-range/repeated vertex indices",
        );
    }

    if duplicate_faces == 0 {
        push(
            &mut checks,
            "duplicate_faces",
            CheckStatus::Pass,
            "no duplicate triangles",
        );
    } else {
        invalid = true;
        push(
            &mut checks,
            "duplicate_faces",
            CheckStatus::Fail,
            format!("{duplicate_faces} duplicate triangles"),
        );
    }

    if degenerate_faces == 0 {
        push(
            &mut checks,
            "non_degenerate_faces",
            CheckStatus::Pass,
            "all triangle areas meet the profile threshold",
        );
    } else {
        invalid = true;
        push(
            &mut checks,
            "non_degenerate_faces",
            CheckStatus::Fail,
            format!("{degenerate_faces} degenerate or below-threshold triangles"),
        );
    }

    let boundary_edges = edges.values().filter(|dirs| dirs.len() == 1).count();
    let non_manifold_edges = edges.values().filter(|dirs| dirs.len() > 2).count();
    let inconsistent_winding_edges = edges
        .values()
        .filter(|dirs| dirs.len() == 2 && dirs[0] + dirs[1] != 0)
        .count();

    if non_manifold_edges == 0 {
        push(
            &mut checks,
            "edge_manifoldness",
            CheckStatus::Pass,
            "no edge is incident to more than two faces",
        );
    } else {
        invalid = true;
        push(
            &mut checks,
            "edge_manifoldness",
            CheckStatus::Fail,
            format!("{non_manifold_edges} non-manifold edges"),
        );
    }

    if inconsistent_winding_edges == 0 {
        push(
            &mut checks,
            "consistent_winding",
            CheckStatus::Pass,
            "shared edges have opposing directed incidence",
        );
    } else {
        invalid = true;
        push(
            &mut checks,
            "consistent_winding",
            CheckStatus::Fail,
            format!("{inconsistent_winding_edges} shared edges have inconsistent winding"),
        );
    }

    if candidate.profile.require_closed_surface {
        if boundary_edges == 0 && !faces.is_empty() {
            push(
                &mut checks,
                "closed_surface",
                CheckStatus::Pass,
                "no boundary edges detected",
            );
        } else {
            invalid = true;
            push(
                &mut checks,
                "closed_surface",
                CheckStatus::Fail,
                format!("{boundary_edges} boundary edges detected"),
            );
        }
    } else {
        push(
            &mut checks,
            "closed_surface",
            CheckStatus::NotRequired,
            "profile does not require a closed surface",
        );
    }

    let volume = if indices_valid && finite_vertices && !faces.is_empty() {
        signed_volume(&candidate.geometry)
    } else {
        None
    };

    let volume_evaluable = candidate.profile.require_closed_surface
        && boundary_edges == 0
        && non_manifold_edges == 0
        && inconsistent_winding_edges == 0
        && indices_valid
        && finite_vertices
        && duplicate_faces == 0
        && degenerate_faces == 0
        && candidate.geometry.units == "m"
        && frame.handedness == "right-handed";

    if volume_evaluable {
        match volume {
            Some(value) if value > 0.0 => {
                push(
                    &mut checks,
                    "positive_signed_volume",
                    CheckStatus::Pass,
                    format!(
                        "signed volume is {value:.12e} m^3 in the declared right-handed frame"
                    ),
                );
                if value >= candidate.profile.minimum_enclosed_volume_m3 {
                    push(
                        &mut checks,
                        "minimum_enclosed_volume",
                        CheckStatus::Pass,
                        format!(
                            "signed volume meets the profile minimum of {:.12e} m^3",
                            candidate.profile.minimum_enclosed_volume_m3
                        ),
                    );
                } else {
                    blocked = true;
                    push(
                        &mut checks,
                        "minimum_enclosed_volume",
                        CheckStatus::Fail,
                        format!(
                            "signed volume {value:.12e} m^3 is below the profile minimum {:.12e} m^3",
                            candidate.profile.minimum_enclosed_volume_m3
                        ),
                    );
                }
            }
            Some(value) => {
                invalid = true;
                push(
                    &mut checks,
                    "positive_signed_volume",
                    CheckStatus::Fail,
                    format!(
                        "signed volume must be positive in the declared right-handed frame; observed {value:.12e} m^3"
                    ),
                );
                push(
                    &mut checks,
                    "minimum_enclosed_volume",
                    CheckStatus::NotEvaluated,
                    "requires a positive, closed, consistently oriented surface",
                );
            }
            None => {
                invalid = true;
                push(
                    &mut checks,
                    "positive_signed_volume",
                    CheckStatus::Fail,
                    "signed volume could not be calculated as a finite value",
                );
                push(
                    &mut checks,
                    "minimum_enclosed_volume",
                    CheckStatus::NotEvaluated,
                    "requires a finite signed volume",
                );
            }
        }
    } else {
        push(
            &mut checks,
            "positive_signed_volume",
            CheckStatus::NotEvaluated,
            "requires a closed, valid, consistently oriented surface in a supported frame",
        );
        push(
            &mut checks,
            "minimum_enclosed_volume",
            CheckStatus::NotEvaluated,
            "requires a valid signed-volume result",
        );
    }

    for required in &candidate.profile.additional_required_checks {
        incomplete = true;
        push(
            &mut checks,
            &format!("unsupported_required:{required}"),
            CheckStatus::Unknown,
            "this check is not implemented by triangle-mesh-integrity-v1; admission cannot be promoted",
        );
    }

    let disposition = if invalid {
        Disposition::Invalid
    } else if incomparable {
        Disposition::Incomparable
    } else if blocked {
        Disposition::Blocked
    } else if incomplete {
        Disposition::Incomplete
    } else {
        Disposition::AdmissibleForProfile
    };

    AdmissionReport {
        report_schema_version: 1,
        admission_scope: ADMISSION_SCOPE.to_owned(),
        candidate_id: candidate.candidate_id.clone(),
        geometry_digest_blake3: candidate.geometry_digest_blake3.clone(),
        profile_id: candidate.profile.id.clone(),
        disposition,
        checks,
        metrics: GeometryMetrics {
            vertex_count: vertices.len(),
            face_count: faces.len(),
            boundary_edge_count: boundary_edges,
            non_manifold_edge_count: non_manifold_edges,
            inconsistent_winding_edge_count: inconsistent_winding_edges,
            duplicate_face_count: duplicate_faces,
            degenerate_face_count: degenerate_faces,
            signed_volume_m3: volume.filter(|v| v.is_finite()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tetrahedron() -> CandidateInput {
        let geometry = MeshGeometry {
            units: "m".into(),
            coordinate_frame: CoordinateFrame {
                id: "ship-local".into(),
                axis_convention: "x-forward,y-starboard,z-down".into(),
                handedness: "right-handed".into(),
            },
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            faces: vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        };
        let digest = compute_geometry_digest(&geometry).unwrap();
        CandidateInput {
            schema_version: 1,
            candidate_id: "fixture-tetrahedron-001".into(),
            geometry_digest_blake3: digest,
            geometry,
            profile: AdmissionProfile {
                id: "mesh-integrity-v1".into(),
                require_closed_surface: true,
                minimum_face_area_m2: 1.0e-12,
                minimum_enclosed_volume_m3: 1.0e-12,
                additional_required_checks: vec![],
            },
        }
    }

    #[test]
    fn outward_closed_tetrahedron_is_admissible_only_for_mesh_profile() {
        let report = admit(&tetrahedron());
        assert_eq!(report.disposition, Disposition::AdmissibleForProfile);
        assert_eq!(report.metrics.boundary_edge_count, 0);
        assert!((report.metrics.signed_volume_m3.unwrap() - 1.0 / 6.0).abs() < 1.0e-12);
        assert_eq!(report.admission_scope, ADMISSION_SCOPE);
    }

    #[test]
    fn open_surface_is_rejected_when_closure_is_required() {
        let mut candidate = tetrahedron();
        candidate.geometry.faces.pop();
        candidate.geometry_digest_blake3 =
            compute_geometry_digest(&candidate.geometry).unwrap();
        let report = admit(&candidate);
        assert_eq!(report.disposition, Disposition::Invalid);
        assert!(report.metrics.boundary_edge_count > 0);
    }

    #[test]
    fn globally_reversed_winding_is_rejected() {
        let mut candidate = tetrahedron();
        for face in &mut candidate.geometry.faces {
            face.swap(1, 2);
        }
        candidate.geometry_digest_blake3 =
            compute_geometry_digest(&candidate.geometry).unwrap();
        let report = admit(&candidate);
        assert_eq!(report.disposition, Disposition::Invalid);
        assert!(report.checks.iter().any(|c| {
            c.code == "positive_signed_volume" && c.status == CheckStatus::Fail
        }));
    }

    #[test]
    fn geometry_mutation_without_digest_update_is_rejected() {
        let mut candidate = tetrahedron();
        candidate.geometry.vertices[1][0] = 1.25;
        let report = admit(&candidate);
        assert_eq!(report.disposition, Disposition::Invalid);
        assert!(report.checks.iter().any(|c| {
            c.code == "geometry_digest" && c.status == CheckStatus::Fail
        }));
    }

    #[test]
    fn unsupported_required_check_forces_incomplete() {
        let mut candidate = tetrahedron();
        candidate
            .profile
            .additional_required_checks
            .push("self_intersection_free".into());
        let report = admit(&candidate);
        assert_eq!(report.disposition, Disposition::Incomplete);
        assert!(report.checks.iter().any(|c| c.status == CheckStatus::Unknown));
    }

    #[test]
    fn non_metric_units_are_incomparable_not_converted() {
        let mut candidate = tetrahedron();
        candidate.geometry.units = "ft".into();
        candidate.geometry_digest_blake3 =
            compute_geometry_digest(&candidate.geometry).unwrap();
        let report = admit(&candidate);
        assert_eq!(report.disposition, Disposition::Incomparable);
    }

    #[test]
    fn below_profile_volume_is_blocked_not_silently_rescaled() {
        let mut candidate = tetrahedron();
        candidate.profile.minimum_enclosed_volume_m3 = 0.2;
        let report = admit(&candidate);
        assert_eq!(report.disposition, Disposition::Blocked);
    }

    #[test]
    fn repeated_vertex_index_is_rejected() {
        let mut candidate = tetrahedron();
        candidate.geometry.faces.push([0, 0, 1]);
        candidate.geometry_digest_blake3 =
            compute_geometry_digest(&candidate.geometry).unwrap();
        let report = admit(&candidate);
        assert_eq!(report.disposition, Disposition::Invalid);
    }

    #[test]
    fn out_of_range_vertex_index_is_rejected() {
        let mut candidate = tetrahedron();
        candidate.geometry.faces.push([0, 1, 99]);
        candidate.geometry_digest_blake3 =
            compute_geometry_digest(&candidate.geometry).unwrap();
        let report = admit(&candidate);
        assert_eq!(report.disposition, Disposition::Invalid);
    }

    #[test]
    fn duplicate_triangle_is_rejected() {
        let mut candidate = tetrahedron();
        candidate.geometry.faces.push([0, 2, 1]);
        candidate.geometry_digest_blake3 =
            compute_geometry_digest(&candidate.geometry).unwrap();
        let report = admit(&candidate);
        assert_eq!(report.disposition, Disposition::Invalid);
        assert!(report.metrics.duplicate_face_count > 0);
    }

    #[test]
    fn non_finite_vertex_is_rejected() {
        let mut candidate = tetrahedron();
        candidate.geometry.vertices[0][0] = f64::NAN;
        let report = admit(&candidate);
        assert_eq!(report.disposition, Disposition::Invalid);
        assert!(report.checks.iter().any(|c| {
            c.code == "finite_coordinates" && c.status == CheckStatus::Fail
        }));
    }

    #[test]
    fn report_is_deterministic_for_identical_input() {
        let candidate = tetrahedron();
        let first = serde_json::to_vec(&admit(&candidate)).unwrap();
        let second = serde_json::to_vec(&admit(&candidate)).unwrap();
        assert_eq!(first, second);
    }
}
