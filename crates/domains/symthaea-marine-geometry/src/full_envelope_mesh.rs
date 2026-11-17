//! G2f-b deterministic full external hull surface loft.
//!
//! The input is the G2f-a full-envelope contract. The output is a separate
//! full external shell; it is not the G2b waterplane-capped displacement mesh.
//! This first implementation supports closed shells with compatible station
//! profiles, explicit/mirrored port geometry, a ruled deck, pointed endpoints,
//! and convex transom caps. It does not heal, weld, fill openings, or infer
//! missing geometry.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::full_envelope::{
    DeckPoint, FullEnvelopeDisposition, FullEnvelopeSpec, HullEndCondition, ProfilePoint,
    ProfileRegion, SurfaceRegion, SymmetryPolicy, compute_full_envelope_spec_digest,
    validate_full_envelope_spec,
};
use crate::{
    AdmissionProfile, AdmissionReport, CandidateInput, CoordinateFrame, Disposition, MeshGeometry,
    admit, compute_geometry_digest,
};

pub const FULL_ENVELOPE_MESH_SCHEMA_VERSION: u32 = 1;
pub const FULL_ENVELOPE_MESH_GENERATOR: &str = "symthaea-marine-full-envelope-loft-v1";
const MAX_MESH_VERTICES: usize = 1_000_000;
const MAX_MESH_FACES: usize = 2_000_000;
const GEOMETRY_TOLERANCE_M: f64 = 1.0e-9;
const CAP_AREA_TOLERANCE_M2: f64 = 1.0e-10;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FullEnvelopeMeshIssueLevel {
    Rejected,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullEnvelopeMeshIssue {
    pub code: String,
    pub level: FullEnvelopeMeshIssueLevel,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticFaceRange {
    pub region: SurfaceRegion,
    pub first_face: usize,
    pub face_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VertexIdentityBinding {
    pub station_index: usize,
    pub station_id: String,
    pub source_point_id: String,
    pub vertex_index: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FullEnvelopeMeshArtifact {
    pub artifact_schema_version: u32,
    pub generator: String,
    pub source_spec_schema_version: u32,
    pub source_spec_digest_blake3: String,
    pub vertex_identity_digest_blake3: String,
    pub semantic_face_ranges: Vec<SemanticFaceRange>,
    pub vertex_identities: Vec<VertexIdentityBinding>,
    pub mesh_digest_blake3: String,
    pub manifest_digest_blake3: String,
    pub candidate: CandidateInput,
    pub admission_report: AdmissionReport,
}

#[derive(Debug, Clone)]
struct StationMeshIndices {
    station_id: String,
    starboard: Vec<u32>,
    port: Vec<u32>,
    deck: Vec<u32>,
}

#[derive(Serialize)]
struct VertexIdentityDigest<'a> {
    bindings: &'a [VertexIdentityBinding],
}

#[derive(Serialize)]
struct FullEnvelopeMeshManifest<'a> {
    artifact_schema_version: u32,
    generator: &'a str,
    source_spec_schema_version: u32,
    source_spec_digest_blake3: &'a str,
    vertex_identity_digest_blake3: &'a str,
    semantic_face_ranges: &'a [SemanticFaceRange],
    vertex_count: usize,
    face_count: usize,
    mesh_digest_blake3: &'a str,
    candidate_id: &'a str,
    admission_profile: &'a AdmissionProfile,
}

fn push_issue(
    issues: &mut Vec<FullEnvelopeMeshIssue>,
    level: FullEnvelopeMeshIssueLevel,
    code: &str,
    detail: impl Into<String>,
) {
    issues.push(FullEnvelopeMeshIssue {
        code: code.to_owned(),
        level,
        detail: detail.into(),
    });
}

fn reject(
    issues: &mut Vec<FullEnvelopeMeshIssue>,
    code: &str,
    detail: impl Into<String>,
) {
    push_issue(
        issues,
        FullEnvelopeMeshIssueLevel::Rejected,
        code,
        detail,
    );
}

fn incomplete(
    issues: &mut Vec<FullEnvelopeMeshIssue>,
    code: &str,
    detail: impl Into<String>,
) {
    push_issue(
        issues,
        FullEnvelopeMeshIssueLevel::Incomplete,
        code,
        detail,
    );
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= GEOMETRY_TOLERANCE_M
}

fn composite_identity(station_index: usize, station_id: &str, point_id: &str) -> String {
    format!("{station_index:04}/{station_id}/{point_id}")
}

fn ensure_vertex(
    station_index: usize,
    station_id: &str,
    point_id: &str,
    position: [f64; 3],
    vertices: &mut Vec<[f64; 3]>,
    identity_map: &mut BTreeMap<String, u32>,
    bindings: &mut Vec<VertexIdentityBinding>,
) -> Result<u32, FullEnvelopeMeshIssue> {
    if !position.iter().all(|coordinate| coordinate.is_finite()) {
        return Err(FullEnvelopeMeshIssue {
            code: "non_finite_vertex".into(),
            level: FullEnvelopeMeshIssueLevel::Rejected,
            detail: format!(
                "station {station_id} point {point_id} contains a non-finite coordinate"
            ),
        });
    }
    let identity = composite_identity(station_index, station_id, point_id);
    if let Some(existing) = identity_map.get(&identity).copied() {
        let existing_position = vertices[existing as usize];
        if existing_position
            .iter()
            .zip(position.iter())
            .any(|(left, right)| !close(*left, *right))
        {
            return Err(FullEnvelopeMeshIssue {
                code: "shared_identity_coordinate_mismatch".into(),
                level: FullEnvelopeMeshIssueLevel::Rejected,
                detail: format!(
                    "station {station_id} reuses point identity {point_id} at different coordinates"
                ),
            });
        }
        return Ok(existing);
    }
    if vertices.len() >= MAX_MESH_VERTICES || vertices.len() >= u32::MAX as usize {
        return Err(FullEnvelopeMeshIssue {
            code: "mesh_vertex_budget".into(),
            level: FullEnvelopeMeshIssueLevel::Rejected,
            detail: format!("vertex count reached the configured limit {MAX_MESH_VERTICES}"),
        });
    }
    let index = vertices.len() as u32;
    vertices.push(position);
    identity_map.insert(identity, index);
    bindings.push(VertexIdentityBinding {
        station_index,
        station_id: station_id.to_owned(),
        source_point_id: point_id.to_owned(),
        vertex_index: index,
    });
    Ok(index)
}

fn source_profile_position(
    x_m: f64,
    point: &ProfilePoint,
    mirrored: bool,
) -> [f64; 3] {
    [
        x_m,
        if mirrored { -point.y_m } else { point.y_m },
        point.z_m,
    ]
}

fn source_deck_position(x_m: f64, point: &DeckPoint) -> [f64; 3] {
    [x_m, point.y_m, point.z_m]
}

fn checked_add(total: usize, amount: usize) -> Option<usize> {
    total.checked_add(amount)
}

fn checked_mul(left: usize, right: usize) -> Option<usize> {
    left.checked_mul(right)
}

fn preflight_mesh_budget(spec: &FullEnvelopeSpec) -> Result<(), Vec<FullEnvelopeMeshIssue>> {
    let station_count = spec.stations.len();
    let Some(profile_point_count) = spec.stations.first().map(|station| station.starboard_profile.len()) else {
        return Err(vec![FullEnvelopeMeshIssue {
            code: "station_count".into(),
            level: FullEnvelopeMeshIssueLevel::Rejected,
            detail: "at least two non-empty stations are required".into(),
        }]);
    };
    let Some(deck_point_count) = spec
        .deck_surface
        .sections
        .iter()
        .map(|section| section.points.len())
        .max()
    else {
        return Err(vec![FullEnvelopeMeshIssue {
            code: "deck_station_count".into(),
            level: FullEnvelopeMeshIssueLevel::Rejected,
            detail: "every station requires a deck section".into(),
        }]);
    };

    let vertex_upper_per_station = profile_point_count
        .checked_mul(2)
        .and_then(|count| count.checked_add(deck_point_count));
    let vertex_upper = vertex_upper_per_station
        .and_then(|per_station| per_station.checked_mul(station_count))
        .and_then(|count| count.checked_add(2));
    let side_faces = station_count
        .checked_sub(1)
        .and_then(|intervals| intervals.checked_mul(profile_point_count.saturating_sub(1)))
        .and_then(|count| count.checked_mul(4));
    let deck_faces = station_count
        .checked_sub(1)
        .and_then(|intervals| {
            let max_segments = deck_point_count.saturating_sub(1);
            intervals.checked_mul(max_segments)
        })
        .and_then(|count| count.checked_mul(2));
    let cap_faces_per_end = profile_point_count
        .checked_mul(2)
        .and_then(|count| count.checked_add(deck_point_count))
        .and_then(|count| count.checked_sub(3));
    let cap_face_upper = cap_faces_per_end.and_then(|count| count.checked_mul(2));
    let face_upper = side_faces
        .and_then(|count| checked_add(count, deck_faces?))
        .and_then(|count| checked_add(count, cap_face_upper?));

    let Some(vertex_upper) = vertex_upper else {
        return Err(vec![FullEnvelopeMeshIssue {
            code: "vertex_budget_overflow".into(),
            level: FullEnvelopeMeshIssueLevel::Rejected,
            detail: "checked vertex-count preflight overflowed".into(),
        }]);
    };
    let Some(face_upper) = face_upper else {
        return Err(vec![FullEnvelopeMeshIssue {
            code: "face_budget_overflow".into(),
            level: FullEnvelopeMeshIssueLevel::Rejected,
            detail: "checked face-count preflight overflowed".into(),
        }]);
    };
    if vertex_upper > MAX_MESH_VERTICES || face_upper > MAX_MESH_FACES {
        return Err(vec![FullEnvelopeMeshIssue {
            code: "mesh_budget_exceeded".into(),
            level: FullEnvelopeMeshIssueLevel::Rejected,
            detail: format!(
                "preflight upper bound is {vertex_upper} vertices and {face_upper} faces; limits are {MAX_MESH_VERTICES} and {MAX_MESH_FACES}"
            ),
        }]);
    }
    Ok(())
}

fn validate_generation_profile(
    spec: &FullEnvelopeSpec,
) -> Result<(), Vec<FullEnvelopeMeshIssue>> {
    let mut issues = Vec::new();
    let source_report = validate_full_envelope_spec(spec);
    match source_report.disposition {
        FullEnvelopeDisposition::ValidInputContract => {}
        FullEnvelopeDisposition::Incomplete => incomplete(
            &mut issues,
            "source_spec_incomplete",
            "full-envelope source contains features outside the G2f-b supported profile",
        ),
        FullEnvelopeDisposition::Rejected => reject(
            &mut issues,
            "source_spec_rejected",
            "full-envelope source does not pass the G2f-a input contract",
        ),
    }

    if spec.boundary_semantics.exterior_shell
        != crate::full_envelope::ExteriorShellSemantics::ClosedShell
    {
        incomplete(
            &mut issues,
            "boundary_loop_shell_unsupported",
            "G2f-b v1 generates closed shells only; explicit opening loops are not yet meshed",
        );
    }
    if !spec.boundary_semantics.openings.is_empty() {
        incomplete(
            &mut issues,
            "openings_unsupported",
            "G2f-b v1 does not generate hatch, intake, shaft or other openings",
        );
    }
    if !spec.deck_surface.sections.is_empty()
        && spec.deck_surface.construction
            == crate::full_envelope::DeckConstruction::Unsupported
    {
        incomplete(
            &mut issues,
            "unsupported_deck",
            "the selected deck construction is not supported by the G2f-b loft",
        );
    }
    if spec.symmetry == SymmetryPolicy::ExplicitPortAndStarboard {
        for station in &spec.stations {
            let Some(port) = station.port_profile.as_ref() else {
                continue;
            };
            if let (Some(starboard_keel), Some(port_keel)) =
                (station.starboard_profile.first(), port.first())
                && (starboard_keel.point_id != port_keel.point_id
                    || !close(starboard_keel.y_m, 0.0)
                    || !close(port_keel.y_m, 0.0)
                    || !close(starboard_keel.z_m, port_keel.z_m))
            {
                reject(
                    &mut issues,
                    "shared_keel_identity",
                    format!(
                        "asymmetric station {} must use one shared centreline keel point identity and coordinate"
                    , station.station_id),
                );
            }
        }
    }

    for (station_index, station) in spec.stations.iter().enumerate() {
        for point in &station.starboard_profile {
            if point.region_to_next == ProfileRegion::Unsupported {
                incomplete(
                    &mut issues,
                    "unsupported_profile_segment",
                    format!(
                        "station {} point {} declares an unsupported segment",
                        station.station_id, point.point_id
                    ),
                );
            }
        }
        if station_index + 1 < spec.stations.len() {
            let next = &spec.stations[station_index + 1];
            if station.starboard_profile.len() != next.starboard_profile.len() {
                reject(
                    &mut issues,
                    "station_profile_count_mismatch",
                    format!(
                        "stations {} and {} have different starboard profile point counts",
                        station.station_id, next.station_id
                    ),
                );
                continue;
            }
            for (point_index, (left, right)) in station
                .starboard_profile
                .iter()
                .zip(next.starboard_profile.iter())
                .enumerate()
            {
                if left.role != right.role || left.region_to_next != right.region_to_next {
                    reject(
                        &mut issues,
                        "station_profile_semantics_mismatch",
                        format!(
                            "stations {} and {} disagree at profile point {point_index}",
                            station.station_id, next.station_id
                        ),
                    );
                }
            }
            let left_port = station.port_profile.as_ref();
            let right_port = next.port_profile.as_ref();
            if let (Some(left), Some(right)) = (left_port, right_port) {
                if left.len() != right.len()
                    || left.iter().zip(right.iter()).any(|(a, b)| {
                        a.role != b.role || a.region_to_next != b.region_to_next
                    })
                {
                    reject(
                        &mut issues,
                        "station_port_semantics_mismatch",
                        format!(
                            "stations {} and {} have incompatible port profile semantics",
                            station.station_id, next.station_id
                        ),
                    );
                }
            }
        }
    }

    if spec.stations.len() >= 2 {
        let deck_count = spec
            .deck_surface
            .sections
            .iter()
            .map(|section| section.points.len())
            .max()
            .unwrap_or(0);
        for (station, section) in spec
            .stations
            .iter()
            .zip(spec.deck_surface.sections.iter())
        {
            if section.points.len() != 1 && section.points.len() != deck_count {
                reject(
                    &mut issues,
                    "deck_point_count_mismatch",
                    format!(
                        "deck section {} must either be a pointed endpoint or match the common deck point count",
                        station.station_id
                    ),
                );
            }
        }
        for pair in spec.deck_surface.sections.windows(2) {
            if pair[0].points.len() > 1 && pair[1].points.len() > 1 {
                if pair[0].points.len() != pair[1].points.len()
                    || pair[0]
                        .points
                        .iter()
                        .zip(pair[1].points.iter())
                        .any(|(a, b)| a.role != b.role)
                {
                    reject(
                        &mut issues,
                        "deck_semantics_mismatch",
                        format!(
                            "adjacent deck sections {} and {} have incompatible point semantics",
                            pair[0].station_id, pair[1].station_id
                        ),
                    );
                }
            }
        }
    }

    if !issues.is_empty() {
        return Err(issues);
    }
    preflight_mesh_budget(spec)
}

fn add_profile_vertices(
    station_index: usize,
    station_id: &str,
    x_m: f64,
    profile: &[ProfilePoint],
    mirrored: bool,
    vertices: &mut Vec<[f64; 3]>,
    identity_map: &mut BTreeMap<String, u32>,
    bindings: &mut Vec<VertexIdentityBinding>,
) -> Result<Vec<u32>, FullEnvelopeMeshIssue> {
    let mut indices = Vec::with_capacity(profile.len());
    for point in profile {
        let is_centerline = close(point.y_m, 0.0);
        let point_id = if mirrored && !is_centerline {
            format!("MIRROR:{}", point.point_id)
        } else {
            point.point_id.clone()
        };
        let position = source_profile_position(x_m, point, mirrored);
        let index = ensure_vertex(
            station_index,
            station_id,
            &point_id,
            position,
            vertices,
            identity_map,
            bindings,
        )?;
        indices.push(index);
    }
    Ok(indices)
}

fn add_deck_vertices(
    station_index: usize,
    station_id: &str,
    x_m: f64,
    deck_points: &[DeckPoint],
    vertices: &mut Vec<[f64; 3]>,
    identity_map: &mut BTreeMap<String, u32>,
    bindings: &mut Vec<VertexIdentityBinding>,
) -> Result<Vec<u32>, FullEnvelopeMeshIssue> {
    let mut indices = Vec::with_capacity(deck_points.len());
    for point in deck_points {
        indices.push(ensure_vertex(
            station_index,
            station_id,
            &point.point_id,
            source_deck_position(x_m, point),
            vertices,
            identity_map,
            bindings,
        )?);
    }
    Ok(indices)
}

fn cap_ring(
    station_id: &str,
    station: &StationMeshIndices,
) -> Result<Vec<u32>, FullEnvelopeMeshIssue> {
    let mut ring = station.starboard.to_vec();
    if station.deck.len() < 2 || station.port.len() < 2 {
        return Err(FullEnvelopeMeshIssue {
            code: "transom_cap_section_incomplete".into(),
            level: FullEnvelopeMeshIssueLevel::Rejected,
            detail: format!("transom station {station_id} requires full deck and port profiles"),
        });
    }
    ring.extend(station.deck.iter().rev().skip(1).copied());
    ring.extend(station.port.iter().rev().skip(1).copied());
    if ring.len() > 1 && ring.first() == ring.last() {
        ring.pop();
    } else {
        return Err(FullEnvelopeMeshIssue {
            code: "transom_keel_seam".into(),
            level: FullEnvelopeMeshIssueLevel::Rejected,
            detail: format!(
                "transom station {station_id} does not close through a shared keel vertex identity"
            ),
        });
    }
    if ring.len() < 3 {
        return Err(FullEnvelopeMeshIssue {
            code: "transom_ring_size".into(),
            level: FullEnvelopeMeshIssueLevel::Rejected,
            detail: format!("transom station {station_id} has fewer than three unique perimeter vertices"),
        });
    }
    if ring.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(FullEnvelopeMeshIssue {
            code: "transom_duplicate_edge_vertex".into(),
            level: FullEnvelopeMeshIssueLevel::Rejected,
            detail: format!("transom station {station_id} contains a repeated consecutive vertex"),
        });
    }
    Ok(ring)
}

fn signed_yz_area_twice(ring: &[u32], vertices: &[[f64; 3]]) -> f64 {
    let mut sum = 0.0;
    for index in 0..ring.len() {
        let a = vertices[ring[index] as usize];
        let b = vertices[ring[(index + 1) % ring.len()] as usize];
        sum += a[1] * b[2] - b[1] * a[2];
    }
    sum
}

fn transom_ring_is_convex_clockwise(
    station_id: &str,
    ring: &[u32],
    vertices: &[[f64; 3]],
) -> Result<(), FullEnvelopeMeshIssue> {
    let twice_area = signed_yz_area_twice(ring, vertices);
    if !twice_area.is_finite() || twice_area >= -CAP_AREA_TOLERANCE_M2 {
        return Err(FullEnvelopeMeshIssue {
            code: "transom_cap_orientation".into(),
            level: FullEnvelopeMeshIssueLevel::Rejected,
            detail: format!(
                "transom station {station_id} perimeter must be a non-zero clockwise yz polygon"
            ),
        });
    }
    let mut has_strict_turn = false;
    for index in 0..ring.len() {
        let previous = vertices[ring[(index + ring.len() - 1) % ring.len()] as usize];
        let current = vertices[ring[index] as usize];
        let next = vertices[ring[(index + 1) % ring.len()] as usize];
        let first = [current[1] - previous[1], current[2] - previous[2]];
        let second = [next[1] - current[1], next[2] - current[2]];
        let turn = first[0] * second[1] - first[1] * second[0];
        if !turn.is_finite() || turn > CAP_AREA_TOLERANCE_M2 {
            return Err(FullEnvelopeMeshIssue {
                code: "non_convex_transom_cap".into(),
                level: FullEnvelopeMeshIssueLevel::Incomplete,
                detail: format!(
                    "transom station {station_id} is outside the v1 convex-cap triangulation profile"
                ),
            });
        }
        has_strict_turn |= turn < -CAP_AREA_TOLERANCE_M2;
    }
    if !has_strict_turn {
        return Err(FullEnvelopeMeshIssue {
            code: "degenerate_transom_cap".into(),
            level: FullEnvelopeMeshIssueLevel::Rejected,
            detail: format!("transom station {station_id} perimeter has no non-zero corner"),
        });
    }
    Ok(())
}

fn append_transom_cap(
    station_index: usize,
    station_id: &str,
    is_bow: bool,
    ring: &[u32],
    vertices: &mut Vec<[f64; 3]>,
    identity_map: &mut BTreeMap<String, u32>,
    bindings: &mut Vec<VertexIdentityBinding>,
) -> Result<Vec<[u32; 3]>, FullEnvelopeMeshIssue> {
    transom_ring_is_convex_clockwise(station_id, ring, vertices)?;
    let mut centre = [0.0; 3];
    for index in ring {
        let point = vertices[*index as usize];
        for axis in 0..3 {
            centre[axis] += point[axis] / ring.len() as f64;
        }
    }
    let centre_id = if is_bow {
        "@generated/bow-transom-centre"
    } else {
        "@generated/stern-transom-centre"
    };
    let centre_index = ensure_vertex(
        station_index,
        station_id,
        centre_id,
        centre,
        vertices,
        identity_map,
        bindings,
    )?;
    let mut faces = Vec::with_capacity(ring.len());
    for index in 0..ring.len() {
        let next = (index + 1) % ring.len();
        if is_bow {
            faces.push([centre_index, ring[index], ring[next]]);
        } else {
            faces.push([centre_index, ring[next], ring[index]]);
        }
    }
    Ok(faces)
}

fn append_side_interval(
    left: &StationMeshIndices,
    right: &StationMeshIndices,
    left_spec: &crate::full_envelope::FullEnvelopeStation,
    right_spec: &crate::full_envelope::FullEnvelopeStation,
    right_collapsed: bool,
    bottom_faces: &mut Vec<[u32; 3]>,
    topsides_faces: &mut Vec<[u32; 3]>,
    issues: &mut Vec<FullEnvelopeMeshIssue>,
) {
    for point_index in 0..left.starboard.len().saturating_sub(1) {
        let region = left_spec.starboard_profile[point_index].region_to_next;
        if region != right_spec.starboard_profile[point_index].region_to_next {
            issues.push(FullEnvelopeMeshIssue {
                code: "station_segment_semantics_mismatch".into(),
                level: FullEnvelopeMeshIssueLevel::Rejected,
                detail: format!(
                    "stations {} and {} disagree on profile segment {point_index}",
                    left.station_id, right.station_id
                ),
            });
            continue;
        }
        let target = match region {
            ProfileRegion::Bottom => bottom_faces,
            ProfileRegion::Topsides => topsides_faces,
            _ => {
                issues.push(FullEnvelopeMeshIssue {
                    code: "unsupported_surface_segment".into(),
                    level: FullEnvelopeMeshIssueLevel::Incomplete,
                    detail: format!(
                        "profile segment {point_index} from station {} to {} is not a supported bottom/topside region",
                        left.station_id, right.station_id
                    ),
                });
                continue;
            }
        };
        let a = left.starboard[point_index];
        let b = right.starboard[point_index];
        let c = right.starboard[point_index + 1];
        let d = left.starboard[point_index + 1];
        let port_a = left.port[point_index];
        let port_b = right.port[point_index];
        let port_c = right.port[point_index + 1];
        let port_d = left.port[point_index + 1];

        if right_collapsed {
            target.extend([[a, b, d], [b, c, d], [port_a, port_d, port_b], [port_d, port_c, port_b]]);
        } else {
            target.extend([[a, b, c], [a, c, d], [port_a, port_c, port_b], [port_a, port_d, port_c]]);
        }
    }
}

fn append_deck_interval(
    left: &StationMeshIndices,
    right: &StationMeshIndices,
    deck_faces: &mut Vec<[u32; 3]>,
) {
    if left.deck.len() == 1 {
        let tip = left.deck[0];
        for segment in 0..right.deck.len().saturating_sub(1) {
            deck_faces.push([tip, right.deck[segment + 1], right.deck[segment]]);
        }
    } else if right.deck.len() == 1 {
        let tip = right.deck[0];
        for segment in 0..left.deck.len().saturating_sub(1) {
            deck_faces.push([tip, left.deck[segment], left.deck[segment + 1]]);
        }
    } else {
        for segment in 0..left.deck.len().saturating_sub(1) {
            let a = left.deck[segment];
            let b = right.deck[segment];
            let c = right.deck[segment + 1];
            let d = left.deck[segment + 1];
            deck_faces.extend([[a, d, b], [b, d, c]]);
        }
    }
}

pub fn compute_full_envelope_mesh_manifest_digest(
    artifact: &FullEnvelopeMeshArtifact,
) -> Result<String, serde_json::Error> {
    let identity = FullEnvelopeMeshManifest {
        artifact_schema_version: artifact.artifact_schema_version,
        generator: &artifact.generator,
        source_spec_schema_version: artifact.source_spec_schema_version,
        source_spec_digest_blake3: &artifact.source_spec_digest_blake3,
        vertex_identity_digest_blake3: &artifact.vertex_identity_digest_blake3,
        semantic_face_ranges: &artifact.semantic_face_ranges,
        vertex_count: artifact.candidate.geometry.vertices.len(),
        face_count: artifact.candidate.geometry.faces.len(),
        mesh_digest_blake3: &artifact.mesh_digest_blake3,
        candidate_id: &artifact.candidate.candidate_id,
        admission_profile: &artifact.candidate.profile,
    };
    Ok(blake3::hash(&serde_json::to_vec(&identity)?).to_hex().to_string())
}

pub fn compute_vertex_identity_digest(
    bindings: &[VertexIdentityBinding],
) -> Result<String, serde_json::Error> {
    Ok(blake3::hash(&serde_json::to_vec(&VertexIdentityDigest { bindings })?)
        .to_hex()
        .to_string())
}

/// Generate a closed full external hull mesh and freshly qualify its triangle
/// topology against the caller's exact G0/G1 admission profile.
pub fn generate_full_envelope_mesh(
    spec: &FullEnvelopeSpec,
    admission_profile: &AdmissionProfile,
) -> Result<FullEnvelopeMeshArtifact, Vec<FullEnvelopeMeshIssue>> {
    validate_generation_profile(spec)?;
    if !admission_profile.require_closed_surface {
        return Err(vec![FullEnvelopeMeshIssue {
            code: "admission_profile_open_shell".into(),
            level: FullEnvelopeMeshIssueLevel::Rejected,
            detail: "G2f-b v1 requires an admission profile that requires a closed surface".into(),
        }]);
    }

    let station_count = spec.stations.len();
    let mut vertices = Vec::<[f64; 3]>::new();
    let mut identity_map = BTreeMap::<String, u32>::new();
    let mut bindings = Vec::<VertexIdentityBinding>::new();
    let mut stations = Vec::<StationMeshIndices>::with_capacity(station_count);

    for (station_index, station) in spec.stations.iter().enumerate() {
        let x_m = station.x_over_length * spec.principal_dimensions.length_m;
        let starboard = add_profile_vertices(
            station_index,
            &station.station_id,
            x_m,
            &station.starboard_profile,
            false,
            &mut vertices,
            &mut identity_map,
            &mut bindings,
        )
        .map_err(|issue| vec![issue])?;

        let port = match spec.symmetry {
            SymmetryPolicy::SymmetricAboutY0 => add_profile_vertices(
                station_index,
                &station.station_id,
                x_m,
                &station.starboard_profile,
                true,
                &mut vertices,
                &mut identity_map,
                &mut bindings,
            )
            .map_err(|issue| vec![issue])?,
            SymmetryPolicy::ExplicitPortAndStarboard => {
                let Some(port_profile) = station.port_profile.as_ref() else {
                    return Err(vec![FullEnvelopeMeshIssue {
                        code: "explicit_port_profile_missing".into(),
                        level: FullEnvelopeMeshIssueLevel::Rejected,
                        detail: format!("station {} is missing its explicit port profile", station.station_id),
                    }]);
                };
                add_profile_vertices(
                    station_index,
                    &station.station_id,
                    x_m,
                    port_profile,
                    false,
                    &mut vertices,
                    &mut identity_map,
                    &mut bindings,
                )
                .map_err(|issue| vec![issue])?
            }
        };

        let deck_section = spec
            .deck_surface
            .sections
            .get(station_index)
            .ok_or_else(|| {
                vec![FullEnvelopeMeshIssue {
                    code: "deck_station_missing".into(),
                    level: FullEnvelopeMeshIssueLevel::Rejected,
                    detail: format!("station {} has no matching deck section", station.station_id),
                }]
            })?;
        let deck = add_deck_vertices(
            station_index,
            &station.station_id,
            x_m,
            &deck_section.points,
            &mut vertices,
            &mut identity_map,
            &mut bindings,
        )
        .map_err(|issue| vec![issue])?;

        stations.push(StationMeshIndices {
            station_id: station.station_id.clone(),
            starboard,
            port,
            deck,
        });
    }

    let mut bottom_faces = Vec::new();
    let mut topsides_faces = Vec::new();
    let mut deck_faces = Vec::new();
    let mut bow_faces = Vec::new();
    let mut stern_faces = Vec::new();
    let mut issues = Vec::new();

    for interval in 0..station_count - 1 {
        let right_collapsed = interval + 1 == station_count - 1
            && spec.end_conditions.stern == HullEndCondition::PointedStem;
        append_side_interval(
            &stations[interval],
            &stations[interval + 1],
            &spec.stations[interval],
            &spec.stations[interval + 1],
            right_collapsed,
            &mut bottom_faces,
            &mut topsides_faces,
            &mut issues,
        );
        append_deck_interval(
            &stations[interval],
            &stations[interval + 1],
            &mut deck_faces,
        );
    }

    if spec.end_conditions.bow == HullEndCondition::Transom {
        let ring = cap_ring(spec.stations[0].station_id.as_str(), &stations[0])
            .map_err(|issue| vec![issue])?;
        bow_faces = append_transom_cap(
            0,
            &spec.stations[0].station_id,
            true,
            &ring,
            &mut vertices,
            &mut identity_map,
            &mut bindings,
        )
        .map_err(|issue| vec![issue])?;
    }
    if spec.end_conditions.stern == HullEndCondition::Transom {
        let last_index = station_count - 1;
        let ring = cap_ring(
            spec.stations[last_index].station_id.as_str(),
            &stations[last_index],
        )
        .map_err(|issue| vec![issue])?;
        stern_faces = append_transom_cap(
            last_index,
            &spec.stations[last_index].station_id,
            false,
            &ring,
            &mut vertices,
            &mut identity_map,
            &mut bindings,
        )
        .map_err(|issue| vec![issue])?;
    }

    if !issues.is_empty() {
        return Err(issues);
    }

    let mut faces = Vec::new();
    let mut face_ranges = Vec::new();
    for (region, region_faces) in [
        (SurfaceRegion::Bottom, bottom_faces),
        (SurfaceRegion::Topsides, topsides_faces),
        (SurfaceRegion::Deck, deck_faces),
        (SurfaceRegion::BowEndClosure, bow_faces),
        (SurfaceRegion::SternEndClosure, stern_faces),
    ] {
        let first_face = faces.len();
        faces.extend(region_faces);
        face_ranges.push(SemanticFaceRange {
            region,
            first_face,
            face_count: faces.len() - first_face,
        });
    }

    if vertices.len() > MAX_MESH_VERTICES || faces.len() > MAX_MESH_FACES {
        return Err(vec![FullEnvelopeMeshIssue {
            code: "mesh_budget_exceeded".into(),
            level: FullEnvelopeMeshIssueLevel::Rejected,
            detail: format!(
                "generated mesh has {} vertices and {} faces; configured limits are {MAX_MESH_VERTICES} and {MAX_MESH_FACES}",
                vertices.len(),
                faces.len()
            ),
        }]);
    }

    let geometry = MeshGeometry {
        units: spec.units.clone(),
        coordinate_frame: CoordinateFrame {
            id: spec.coordinate_frame.id.clone(),
            axis_convention: spec.coordinate_frame.axis_convention.clone(),
            handedness: spec.coordinate_frame.handedness.clone(),
        },
        vertices,
        faces,
    };
    let mesh_digest_blake3 = compute_geometry_digest(&geometry).map_err(|error| {
        vec![FullEnvelopeMeshIssue {
            code: "mesh_digest".into(),
            level: FullEnvelopeMeshIssueLevel::Rejected,
            detail: format!("could not compute geometry digest: {error}"),
        }]
    })?;
    let candidate_id = format!("{}-full-envelope", spec.candidate_id);
    let candidate = CandidateInput {
        schema_version: 1,
        candidate_id,
        geometry_digest_blake3: mesh_digest_blake3.clone(),
        geometry,
        profile: admission_profile.clone(),
    };
    let admission_report = admit(&candidate);
    if admission_report.disposition != Disposition::AdmissibleForProfile {
        return Err(vec![FullEnvelopeMeshIssue {
            code: "g0_g1_admission_not_pass".into(),
            level: if admission_report.disposition == Disposition::Incomplete {
                FullEnvelopeMeshIssueLevel::Incomplete
            } else {
                FullEnvelopeMeshIssueLevel::Rejected
            },
            detail: format!(
                "fresh mesh admission returned {:?}; G2f-b does not promote an incomplete or rejected shell",
                admission_report.disposition
            ),
        }]);
    }

    let mut vertex_identities = bindings;
    vertex_identities.sort_by(|left, right| {
        (left.station_index, left.source_point_id.as_str())
            .cmp(&(right.station_index, right.source_point_id.as_str()))
    });
    let vertex_identity_digest_blake3 =
        compute_vertex_identity_digest(&vertex_identities).map_err(|error| {
            vec![FullEnvelopeMeshIssue {
                code: "vertex_identity_digest".into(),
                level: FullEnvelopeMeshIssueLevel::Rejected,
                detail: format!("could not serialize vertex identity bindings: {error}"),
            }]
        })?;

    let mut artifact = FullEnvelopeMeshArtifact {
        artifact_schema_version: FULL_ENVELOPE_MESH_SCHEMA_VERSION,
        generator: FULL_ENVELOPE_MESH_GENERATOR.into(),
        source_spec_schema_version: spec.schema_version,
        source_spec_digest_blake3: compute_full_envelope_spec_digest(spec).map_err(|error| {
            vec![FullEnvelopeMeshIssue {
                code: "source_spec_digest".into(),
                level: FullEnvelopeMeshIssueLevel::Rejected,
                detail: format!("could not compute source specification digest: {error}"),
            }]
        })?,
        vertex_identity_digest_blake3,
        semantic_face_ranges: face_ranges,
        vertex_identities,
        mesh_digest_blake3,
        manifest_digest_blake3: String::new(),
        candidate,
        admission_report,
    };
    artifact.manifest_digest_blake3 =
        compute_full_envelope_mesh_manifest_digest(&artifact).map_err(|error| {
            vec![FullEnvelopeMeshIssue {
                code: "manifest_digest".into(),
                level: FullEnvelopeMeshIssueLevel::Rejected,
                detail: format!("could not compute full-envelope mesh manifest: {error}"),
            }]
        })?;
    Ok(artifact)
}

pub fn verify_full_envelope_mesh_artifact(
    spec: &FullEnvelopeSpec,
    admission_profile: &AdmissionProfile,
    artifact: &FullEnvelopeMeshArtifact,
) -> Result<(), Vec<FullEnvelopeMeshIssue>> {
    let expected = generate_full_envelope_mesh(spec, admission_profile)?;
    if &expected != artifact {
        return Err(vec![FullEnvelopeMeshIssue {
            code: "full_envelope_mesh_replay_mismatch".into(),
            level: FullEnvelopeMeshIssueLevel::Rejected,
            detail: "artifact differs from a fresh generation of the exact full-envelope source and admission profile".into(),
        }]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_fixture() -> FullEnvelopeSpec {
        serde_json::from_str(include_str!(
            "../../../../docs/research/marine/fixtures/marine-full-envelope-box-001.json"
        ))
        .unwrap()
    }

    fn admission_profile() -> AdmissionProfile {
        AdmissionProfile {
            id: "marine-full-envelope-closed-shell-test-v1".into(),
            require_closed_surface: true,
            minimum_face_area_m2: 1.0e-12,
            minimum_enclosed_volume_m3: 1.0e-9,
            additional_required_checks: Vec::new(),
        }
    }

    fn assert_issue(
        spec: &FullEnvelopeSpec,
        expected_code: &str,
        level: FullEnvelopeMeshIssueLevel,
    ) {
        let errors = generate_full_envelope_mesh(spec, &admission_profile()).unwrap_err();
        assert!(errors.iter().any(|issue| {
            issue.code == expected_code && issue.level == level
        }));
    }

    fn make_pointed_end(spec: &mut FullEnvelopeSpec, is_bow: bool) {
        let station_index = if is_bow { 0 } else { spec.stations.len() - 1 };
        spec.end_conditions.bow = if is_bow {
            HullEndCondition::PointedStem
        } else {
            spec.end_conditions.bow
        };
        spec.end_conditions.stern = if is_bow {
            spec.end_conditions.stern
        } else {
            HullEndCondition::PointedStem
        };
        for point in &mut spec.stations[station_index].starboard_profile {
            point.y_m = 0.0;
        }
        let station_id = spec.stations[station_index].station_id.clone();
        let sheer = spec.stations[station_index]
            .starboard_profile
            .last()
            .unwrap()
            .clone();
        spec.deck_surface.sections[station_index].points = vec![DeckPoint {
            point_id: sheer.point_id,
            y_m: 0.0,
            z_m: sheer.z_m,
            role: crate::full_envelope::DeckPointRole::DeckCrown,
        }];
        let intersection = spec
            .design_waterline
            .station_intersections
            .iter_mut()
            .find(|item| item.station_id == station_id)
            .unwrap();
        intersection.port_source =
            crate::full_envelope::WaterlinePortSource::MirroredFromStarboard;
        intersection.port_point_id = None;
    }

    fn make_asymmetric(spec: &mut FullEnvelopeSpec) {
        spec.symmetry = SymmetryPolicy::ExplicitPortAndStarboard;
        for station in &mut spec.stations {
            let mut port = station.starboard_profile.clone();
            for point in &mut port {
                if point.role != crate::full_envelope::ProfilePointRole::Keel {
                    point.point_id = format!("PORT-{}", point.point_id);
                    point.y_m = -0.9 * point.y_m;
                }
            }
            station.port_profile = Some(port);
        }
        for intersection in &mut spec.design_waterline.station_intersections {
            intersection.port_source =
                crate::full_envelope::WaterlinePortSource::ExplicitPortProfile;
            intersection.port_point_id =
                Some(format!("PORT-{}", intersection.starboard_point_id));
        }
        for (station, deck) in spec
            .stations
            .iter()
            .zip(spec.deck_surface.sections.iter_mut())
        {
            let port_sheer = station.port_profile.as_ref().unwrap().last().unwrap();
            deck.points[0].point_id = port_sheer.point_id.clone();
            deck.points[0].y_m = port_sheer.y_m;
        }
    }

    #[test]
    fn box_fixture_generates_admissible_full_external_shell_with_known_volume() {
        let spec = box_fixture();
        let artifact =
            generate_full_envelope_mesh(&spec, &admission_profile()).unwrap();
        assert_eq!(
            artifact.admission_report.disposition,
            Disposition::AdmissibleForProfile
        );
        assert_eq!(artifact.admission_report.metrics.boundary_edge_count, 0);
        assert_eq!(artifact.admission_report.metrics.non_manifold_edge_count, 0);
        assert_eq!(artifact.admission_report.metrics.non_manifold_vertex_link_count, 0);
        assert_eq!(artifact.admission_report.metrics.degenerate_face_count, 0);
        assert!(
            (artifact.admission_report.metrics.signed_volume_m3.unwrap() - 20_500.0).abs()
                < 1.0e-8
        );
        assert_ne!(
            artifact.admission_report.metrics.signed_volume_m3.unwrap(),
            12_000.0
        );
    }

    #[test]
    fn identical_source_and_profile_generate_identical_manifest_and_mesh() {
        let spec = box_fixture();
        let first = generate_full_envelope_mesh(&spec, &admission_profile()).unwrap();
        let second = generate_full_envelope_mesh(&spec, &admission_profile()).unwrap();
        assert_eq!(first, second);
        verify_full_envelope_mesh_artifact(&spec, &admission_profile(), &first).unwrap();
    }

    #[test]
    fn transom_bow_and_pointed_stern_generate_one_open_end_closure() {
        let mut spec = box_fixture();
        make_pointed_end(&mut spec, false);
        let artifact =
            generate_full_envelope_mesh(&spec, &admission_profile()).unwrap();
        assert_eq!(artifact.semantic_face_ranges[3].face_count, 0);
        assert!(artifact.semantic_face_ranges[4].face_count > 0);
        assert_eq!(artifact.admission_report.metrics.boundary_edge_count, 0);
        assert_eq!(artifact.admission_report.metrics.non_manifold_edge_count, 0);
        assert_eq!(artifact.admission_report.metrics.non_manifold_vertex_link_count, 0);
    }

    #[test]
    fn pointed_bow_and_transom_stern_generate_one_open_end_closure() {
        let mut spec = box_fixture();
        make_pointed_end(&mut spec, true);
        let artifact =
            generate_full_envelope_mesh(&spec, &admission_profile()).unwrap();
        assert_eq!(artifact.semantic_face_ranges[3].face_count, 0);
        assert!(artifact.semantic_face_ranges[4].face_count > 0);
        assert_eq!(artifact.admission_report.metrics.boundary_edge_count, 0);
        assert_eq!(artifact.admission_report.metrics.non_manifold_edge_count, 0);
        assert_eq!(artifact.admission_report.metrics.non_manifold_vertex_link_count, 0);
    }

    #[test]
    fn explicit_asymmetric_port_profile_is_not_mirrored() {
        let mut spec = box_fixture();
        make_asymmetric(&mut spec);
        let artifact =
            generate_full_envelope_mesh(&spec, &admission_profile()).unwrap();
        assert!(
            (artifact.admission_report.metrics.signed_volume_m3.unwrap() - 12_350.0).abs()
                < 1.0e-8
        );
        assert_eq!(artifact.admission_report.metrics.boundary_edge_count, 0);
        assert_eq!(artifact.admission_report.metrics.non_manifold_edge_count, 0);
    }

    #[test]
    fn non_convex_transom_cap_is_incomplete() {
        let mut spec = box_fixture();
        spec.deck_surface.sections[0].points[1].z_m = -3.0;
        assert_issue(
            &spec,
            "non_convex_transom_cap",
            FullEnvelopeMeshIssueLevel::Incomplete,
        );
    }

    #[test]
    fn incompatible_station_profile_counts_are_rejected() {
        let mut spec = box_fixture();
        spec.stations[1].starboard_profile.pop();
        assert_issue(
            &spec,
            "station_profile_count_mismatch",
            FullEnvelopeMeshIssueLevel::Rejected,
        );
    }

    #[test]
    fn explicit_asymmetric_hull_requires_one_shared_centreline_keel_identity() {
        let mut spec = box_fixture();
        make_asymmetric(&mut spec);
        for station in &mut spec.stations {
            station.port_profile.as_mut().unwrap()[0].point_id = "PORT-KEEL".into();
        }
        assert_issue(
            &spec,
            "shared_keel_identity",
            FullEnvelopeMeshIssueLevel::Rejected,
        );
    }

    #[test]
    fn opening_boundary_loops_are_incomplete_not_silently_capped() {
        let mut spec = box_fixture();
        spec.boundary_semantics.exterior_shell =
            crate::full_envelope::ExteriorShellSemantics::ExplicitBoundaryLoops;
        spec.boundary_semantics.openings.push(crate::full_envelope::OpeningBoundary {
            opening_id: "HATCH-01".into(),
            boundary_point_refs: vec![
                "BOW-KEEL".into(),
                "BOW-WL-S".into(),
                "BOW-SHEER-S".into(),
            ],
            status: crate::full_envelope::OpeningStatus::Supported,
        });
        assert_issue(
            &spec,
            "boundary_loop_shell_unsupported",
            FullEnvelopeMeshIssueLevel::Incomplete,
        );
    }

    #[test]
    fn semantic_face_ranges_cover_every_generated_face_once() {
        let artifact =
            generate_full_envelope_mesh(&box_fixture(), &admission_profile()).unwrap();
        let mut next = 0;
        for range in &artifact.semantic_face_ranges {
            assert_eq!(range.first_face, next);
            next += range.face_count;
        }
        assert_eq!(next, artifact.candidate.geometry.faces.len());
    }

    #[test]
    fn rehashed_face_mutation_is_rejected_by_fresh_generation_verifier() {
        let spec = box_fixture();
        let mut artifact =
            generate_full_envelope_mesh(&spec, &admission_profile()).unwrap();
        artifact.candidate.geometry.faces[0][0] = artifact.candidate.geometry.faces[0][1];
        artifact.mesh_digest_blake3 =
            compute_geometry_digest(&artifact.candidate.geometry).unwrap();
        artifact.candidate.geometry_digest_blake3 =
            artifact.mesh_digest_blake3.clone();
        artifact.admission_report = admit(&artifact.candidate);
        artifact.manifest_digest_blake3 =
            compute_full_envelope_mesh_manifest_digest(&artifact).unwrap();
        assert!(verify_full_envelope_mesh_artifact(
            &spec,
            &admission_profile(),
            &artifact
        )
        .is_err());
    }

    #[test]
    fn unsupported_mesh_admission_checks_do_not_become_passes() {
        let mut profile = admission_profile();
        profile.additional_required_checks = vec!["manufacturing-feasibility".into()];
        let errors = generate_full_envelope_mesh(&box_fixture(), &profile).unwrap_err();
        assert!(errors.iter().any(|item| item.code == "g0_g1_admission_not_pass"));
    }

    #[test]
    fn face_and_vertex_identity_digests_are_distinct_from_geometry_digest() {
        let artifact =
            generate_full_envelope_mesh(&box_fixture(), &admission_profile()).unwrap();
        assert_ne!(
            artifact.vertex_identity_digest_blake3,
            artifact.mesh_digest_blake3
        );
        assert_ne!(
            artifact.manifest_digest_blake3,
            artifact.mesh_digest_blake3
        );
        assert_eq!(
            artifact.vertex_identities.len(),
            artifact.candidate.geometry.vertices.len()
        );
    }
}
