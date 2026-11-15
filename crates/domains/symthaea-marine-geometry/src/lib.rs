//! Deterministic, deliberately scoped marine geometry admission.

//!
//! This crate validates a versioned triangle-mesh interchange artifact. It is
//! not a CAD kernel, hydrostatics solver, self-intersection detector, DFM
//! checker, CFD solver, or vessel-approval system.

pub mod hydrostatics;
pub mod sectional_hydrostatics;
pub mod loft;
pub mod section;

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

const SCHEMA_VERSION: u32 = 1;
const ADMISSION_SCOPE: &str = "triangle-mesh-integrity-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoordinateFrame {
    pub id: String,
    /// Human-readable axis semantics. No axis transform is inferred.
    pub axis_convention: String,
    /// Signed-volume interpretation currently requires right-handed.
    pub handedness: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeometryMetrics {
    pub vertex_count: usize,
    pub face_count: usize,
    pub boundary_edge_count: usize,
    pub non_manifold_edge_count: usize,
    pub inconsistent_winding_edge_count: usize,
    pub duplicate_face_count: usize,
    pub degenerate_face_count: usize,
    pub non_manifold_vertex_link_count: usize,
    pub component_signed_volumes_m3: Vec<f64>,
    pub signed_volume_m3: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

fn push(checks: &mut Vec<Check>, code: &str, status: CheckStatus, detail: impl Into<String>) {
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

/// Counts vertices whose link is not a single cycle, as required for a
/// closed triangular 2-manifold. This detects disconnected shells pinched
/// together at a vertex, which edge-incidence checks alone cannot detect.
fn count_non_manifold_vertex_links(geometry: &MeshGeometry) -> Option<usize> {
    let mut links: Vec<BTreeMap<u32, BTreeSet<u32>>> =
        vec![BTreeMap::new(); geometry.vertices.len()];

    for face in &geometry.faces {
        let [a, b, c] = *face;
        if a as usize >= links.len() || b as usize >= links.len() || c as usize >= links.len() {
            return None;
        }
        for (center, left, right) in [(a, b, c), (b, c, a), (c, a, b)] {
            links[center as usize]
                .entry(left)
                .or_default()
                .insert(right);
            links[center as usize]
                .entry(right)
                .or_default()
                .insert(left);
        }
    }

    let mut non_manifold = 0_usize;
    for link in links {
        if link.len() < 3 || link.values().any(|neighbors| neighbors.len() != 2) {
            non_manifold += 1;
            continue;
        }

        let Some(start) = link.keys().next().copied() else {
            non_manifold += 1;
            continue;
        };
        let mut visited = BTreeSet::new();
        let mut queue = VecDeque::from([start]);
        while let Some(vertex) = queue.pop_front() {
            if !visited.insert(vertex) {
                continue;
            }
            if let Some(neighbors) = link.get(&vertex) {
                queue.extend(neighbors.iter().copied());
            }
        }
        if visited.len() != link.len() {
            non_manifold += 1;
        }
    }
    Some(non_manifold)
}

/// Calculates signed volume independently for every edge-connected closed
/// component. Each component uses a local reference vertex to reduce
/// cancellation from large global coordinate offsets.
fn component_signed_volumes(geometry: &MeshGeometry) -> Option<Vec<f64>> {
    if geometry.faces.is_empty() {
        return None;
    }

    let mut edge_faces: BTreeMap<(u32, u32), Vec<usize>> = BTreeMap::new();
    for (face_index, face) in geometry.faces.iter().enumerate() {
        let [a, b, c] = *face;
        if a as usize >= geometry.vertices.len()
            || b as usize >= geometry.vertices.len()
            || c as usize >= geometry.vertices.len()
        {
            return None;
        }
        for (from, to) in [(a, b), (b, c), (c, a)] {
            let edge = if from < to { (from, to) } else { (to, from) };
            edge_faces.entry(edge).or_default().push(face_index);
        }
    }

    let mut adjacency = vec![Vec::new(); geometry.faces.len()];
    for incident_faces in edge_faces.values() {
        if incident_faces.len() != 2 {
            return None;
        }
        adjacency[incident_faces[0]].push(incident_faces[1]);
        adjacency[incident_faces[1]].push(incident_faces[0]);
    }

    let mut visited = vec![false; geometry.faces.len()];
    let mut volumes = Vec::new();
    for start in 0..geometry.faces.len() {
        if visited[start] {
            continue;
        }
        let mut queue = VecDeque::from([start]);
        let mut component = Vec::new();
        visited[start] = true;
        while let Some(face_index) = queue.pop_front() {
            component.push(face_index);
            for neighbor in &adjacency[face_index] {
                if !visited[*neighbor] {
                    visited[*neighbor] = true;
                    queue.push_back(*neighbor);
                }
            }
        }

        let reference_index = geometry.faces[start][0] as usize;
        let origin = *geometry.vertices.get(reference_index)?;
        let mut six_volume = 0.0_f64;
        for face_index in component {
            let [a, b, c] = geometry.faces[face_index];
            let a = sub(*geometry.vertices.get(a as usize)?, origin);
            let b = sub(*geometry.vertices.get(b as usize)?, origin);
            let c = sub(*geometry.vertices.get(c as usize)?, origin);
            six_volume += dot(a, cross(b, c));
        }
        let volume = six_volume / 6.0;
        if !volume.is_finite() {
            return None;
        }
        volumes.push(volume);
    }
    Some(volumes)
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
            format!(
                "schema version {} is not supported",
                candidate.schema_version
            ),
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

    let mut non_manifold_vertex_link_count = 0_usize;
    let vertex_links_evaluable = candidate.profile.require_closed_surface
        && indices_valid
        && finite_vertices
        && boundary_edges == 0
        && non_manifold_edges == 0
        && duplicate_faces == 0
        && degenerate_faces == 0;
    if vertex_links_evaluable {
        match count_non_manifold_vertex_links(&candidate.geometry) {
            Some(0) => push(
                &mut checks,
                "vertex_link_manifoldness",
                CheckStatus::Pass,
                "each vertex link is one connected cycle",
            ),
            Some(count) => {
                non_manifold_vertex_link_count = count;
                invalid = true;
                push(
                    &mut checks,
                    "vertex_link_manifoldness",
                    CheckStatus::Fail,
                    format!("{count} vertices have a disconnected or non-cyclic link"),
                );
            }
            None => {
                invalid = true;
                push(
                    &mut checks,
                    "vertex_link_manifoldness",
                    CheckStatus::Fail,
                    "vertex-link analysis could not be completed",
                );
            }
        }
    } else if candidate.profile.require_closed_surface {
        push(
            &mut checks,
            "vertex_link_manifoldness",
            CheckStatus::NotEvaluated,
            "requires valid indices, finite coordinates, and edge-closed topology",
        );
    } else {
        push(
            &mut checks,
            "vertex_link_manifoldness",
            CheckStatus::NotRequired,
            "profile does not require a closed surface",
        );
    }

    let volume_evaluable = vertex_links_evaluable
        && non_manifold_vertex_link_count == 0
        && inconsistent_winding_edges == 0
        && candidate.geometry.units == "m"
        && frame.handedness == "right-handed";

    let component_volumes = if volume_evaluable {
        component_signed_volumes(&candidate.geometry)
    } else {
        None
    };
    let all_components_outward = component_volumes
        .as_ref()
        .is_some_and(|volumes| !volumes.is_empty() && volumes.iter().all(|volume| *volume > 0.0));
    let total_volume = component_volumes.as_ref().and_then(|volumes| {
        let total: f64 = volumes.iter().sum();
        total.is_finite().then_some(total)
    });

    if volume_evaluable {
        match (
            component_volumes.as_ref(),
            total_volume,
            all_components_outward,
        ) {
            (Some(volumes), Some(total), true) => {
                push(
                    &mut checks,
                    "positive_component_signed_volumes",
                    CheckStatus::Pass,
                    format!(
                        "all {} closed components have positive signed volume in the declared right-handed frame",
                        volumes.len()
                    ),
                );
                if total >= candidate.profile.minimum_enclosed_volume_m3 {
                    push(
                        &mut checks,
                        "minimum_enclosed_volume",
                        CheckStatus::Pass,
                        format!(
                            "total component volume {total:.12e} m^3 meets the profile minimum {:.12e} m^3",
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
                            "total component volume {total:.12e} m^3 is below the profile minimum {:.12e} m^3",
                            candidate.profile.minimum_enclosed_volume_m3
                        ),
                    );
                }
            }
            (Some(volumes), Some(total), false) => {
                invalid = true;
                let inward = volumes.iter().filter(|volume| **volume <= 0.0).count();
                push(
                    &mut checks,
                    "positive_component_signed_volumes",
                    CheckStatus::Fail,
                    format!(
                        "{inward} of {} components have non-positive signed volume; total signed volume is {total:.12e} m^3",
                        volumes.len()
                    ),
                );
                push(
                    &mut checks,
                    "minimum_enclosed_volume",
                    CheckStatus::NotEvaluated,
                    "requires every closed component to have positive signed volume",
                );
            }
            _ => {
                invalid = true;
                push(
                    &mut checks,
                    "positive_component_signed_volumes",
                    CheckStatus::Fail,
                    "per-component signed volume could not be calculated",
                );
                push(
                    &mut checks,
                    "minimum_enclosed_volume",
                    CheckStatus::NotEvaluated,
                    "requires finite per-component signed volumes",
                );
            }
        }
    } else {
        push(
            &mut checks,
            "positive_component_signed_volumes",
            CheckStatus::NotEvaluated,
            "requires a closed, edge-manifold, vertex-manifold, consistently oriented surface in a supported frame",
        );
        push(
            &mut checks,
            "minimum_enclosed_volume",
            CheckStatus::NotEvaluated,
            "requires valid per-component signed-volume results",
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
            non_manifold_vertex_link_count,
            component_signed_volumes_m3: component_volumes.unwrap_or_default(),
            signed_volume_m3: total_volume,
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
        assert_eq!(report.metrics.non_manifold_vertex_link_count, 0);
        assert_eq!(report.metrics.component_signed_volumes_m3.len(), 1);
        assert!((report.metrics.signed_volume_m3.unwrap() - 1.0 / 6.0).abs() < 1.0e-12);
        assert_eq!(report.admission_scope, ADMISSION_SCOPE);
    }

    #[test]
    fn open_surface_is_rejected_when_closure_is_required() {
        let mut candidate = tetrahedron();
        candidate.geometry.faces.pop();
        candidate.geometry_digest_blake3 = compute_geometry_digest(&candidate.geometry).unwrap();
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
        candidate.geometry_digest_blake3 = compute_geometry_digest(&candidate.geometry).unwrap();
        let report = admit(&candidate);
        assert_eq!(report.disposition, Disposition::Invalid);
        assert!(report.checks.iter().any(|c| {
            c.code == "positive_component_signed_volumes" && c.status == CheckStatus::Fail
        }));
    }

    #[test]
    fn geometry_mutation_without_digest_update_is_rejected() {
        let mut candidate = tetrahedron();
        candidate.geometry.vertices[1][0] = 1.25;
        let report = admit(&candidate);
        assert_eq!(report.disposition, Disposition::Invalid);
        assert!(
            report
                .checks
                .iter()
                .any(|c| { c.code == "geometry_digest" && c.status == CheckStatus::Fail })
        );
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
        assert!(
            report
                .checks
                .iter()
                .any(|c| c.status == CheckStatus::Unknown)
        );
    }

    #[test]
    fn non_metric_units_are_incomparable_not_converted() {
        let mut candidate = tetrahedron();
        candidate.geometry.units = "ft".into();
        candidate.geometry_digest_blake3 = compute_geometry_digest(&candidate.geometry).unwrap();
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
        candidate.geometry_digest_blake3 = compute_geometry_digest(&candidate.geometry).unwrap();
        let report = admit(&candidate);
        assert_eq!(report.disposition, Disposition::Invalid);
    }

    #[test]
    fn out_of_range_vertex_index_is_rejected() {
        let mut candidate = tetrahedron();
        candidate.geometry.faces.push([0, 1, 99]);
        candidate.geometry_digest_blake3 = compute_geometry_digest(&candidate.geometry).unwrap();
        let report = admit(&candidate);
        assert_eq!(report.disposition, Disposition::Invalid);
    }

    #[test]
    fn duplicate_triangle_is_rejected() {
        let mut candidate = tetrahedron();
        candidate.geometry.faces.push([0, 2, 1]);
        candidate.geometry_digest_blake3 = compute_geometry_digest(&candidate.geometry).unwrap();
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
        assert!(
            report
                .checks
                .iter()
                .any(|c| { c.code == "finite_coordinates" && c.status == CheckStatus::Fail })
        );
    }

    #[test]
    fn inward_small_disconnected_component_cannot_hide_behind_positive_total_volume() {
        let mut candidate = tetrahedron();
        candidate.geometry.vertices.extend([
            [3.0, 0.0, 0.0],
            [3.5, 0.0, 0.0],
            [3.0, 0.5, 0.0],
            [3.0, 0.0, 0.5],
        ]);
        let offset_faces = [[0_u32, 2_u32, 1_u32], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        for [a, b, c] in offset_faces {
            candidate.geometry.faces.push([a + 4, c + 4, b + 4]);
        }
        candidate.geometry_digest_blake3 = compute_geometry_digest(&candidate.geometry).unwrap();
        let report = admit(&candidate);
        assert!(report.metrics.signed_volume_m3.unwrap() > 0.0);
        assert_eq!(report.metrics.component_signed_volumes_m3.len(), 2);
        assert!(
            report
                .metrics
                .component_signed_volumes_m3
                .iter()
                .any(|v| *v < 0.0)
        );
        assert_eq!(report.disposition, Disposition::Invalid);
    }

    #[test]
    fn closed_shells_touching_only_at_one_vertex_are_rejected() {
        let mut candidate = tetrahedron();
        candidate
            .geometry
            .vertices
            .extend([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]);
        // The second tetrahedron shares the first tetrahedron's vertex 0,
        // but shares no edges. Its local winding is outward.
        candidate
            .geometry
            .faces
            .extend([[0, 5, 4], [0, 4, 6], [0, 6, 5], [4, 5, 6]]);
        candidate.geometry_digest_blake3 = compute_geometry_digest(&candidate.geometry).unwrap();
        let report = admit(&candidate);
        assert_eq!(report.disposition, Disposition::Invalid);
        assert!(report.metrics.non_manifold_vertex_link_count > 0);
    }

    #[test]
    fn report_is_deterministic_for_identical_input() {
        let candidate = tetrahedron();
        let first = serde_json::to_vec(&admit(&candidate)).unwrap();
        let second = serde_json::to_vec(&admit(&candidate)).unwrap();
        assert_eq!(first, second);
    }
}
