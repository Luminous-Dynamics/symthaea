//! G2f-b deterministic full external hull mesh generation.
//!
//! This generator is intentionally bounded to closed, opening-free profiles
//! with compatible station topology. It creates a full external surface
//! (bottom, topsides, deck and declared bow/stern closures), not a G2b
//! waterplane-capped displacement volume and not a general-purpose B-rep.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::full_envelope::{
    DeckConstruction, FullEnvelopeDisposition, FullEnvelopeSpec, HullEndCondition,
    ProfileRegion, SurfaceRegion, SymmetryPolicy, validate_full_envelope_spec,
    compute_full_envelope_spec_digest,
};
use crate::{
    AdmissionProfile, AdmissionReport, CandidateInput, CoordinateFrame, Disposition,
    MeshGeometry, admit, compute_geometry_digest,
};

pub const FULL_ENVELOPE_MESH_SCHEMA_VERSION: u32 = 1;
pub const FULL_ENVELOPE_MESH_GENERATOR: &str = "full-envelope-station-loft-v1";
const MAX_MESH_VERTICES: usize = 1_000_000;
const MAX_MESH_FACES: usize = 2_000_000;
const GEOMETRY_TOLERANCE_M: f64 = 1.0e-10;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullEnvelopeMeshViolation {
    pub code: String,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullEnvelopeFaceRange {
    pub first_face: usize,
    pub face_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourcePointVertex {
    pub point_id: String,
    pub vertex_index: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FullEnvelopeMeshArtifact {
    pub artifact_schema_version: u32,
    pub generator: String,
    pub source_spec_digest_blake3: String,
    pub point_vertex_mapping_digest_blake3: String,
    pub mesh_digest_blake3: String,
    pub manifest_digest_blake3: String,
    pub bottom_face_range: FullEnvelopeFaceRange,
    pub topsides_face_range: FullEnvelopeFaceRange,
    pub deck_face_range: FullEnvelopeFaceRange,
    pub bow_end_face_range: FullEnvelopeFaceRange,
    pub stern_end_face_range: FullEnvelopeFaceRange,
    pub point_vertex_mapping: Vec<SourcePointVertex>,
    pub candidate: CandidateInput,
    pub admission_report: AdmissionReport,
}

#[derive(Serialize)]
struct MeshManifestIdentity<'a> {
    artifact_schema_version: u32,
    generator: &'a str,
    source_spec_digest_blake3: &'a str,
    point_vertex_mapping_digest_blake3: &'a str,
    mesh_digest_blake3: &'a str,
    bottom_face_range: FullEnvelopeFaceRange,
    topsides_face_range: FullEnvelopeFaceRange,
    deck_face_range: FullEnvelopeFaceRange,
    bow_end_face_range: FullEnvelopeFaceRange,
    stern_end_face_range: FullEnvelopeFaceRange,
    point_vertex_mapping: &'a [SourcePointVertex],
    candidate_schema_version: u32,
    candidate_id: &'a str,
    geometry_digest_blake3: &'a str,
    admission_profile: &'a AdmissionProfile,
}

#[derive(Debug, Clone)]
struct StationVertexLayout {
    starboard: Vec<u32>,
    port: Vec<u32>,
    deck: Vec<u32>,
}

#[derive(Default)]
struct FaceGroups {
    bottom: Vec<[u32; 3]>,
    topsides: Vec<[u32; 3]>,
    deck: Vec<[u32; 3]>,
    bow: Vec<[u32; 3]>,
    stern: Vec<[u32; 3]>,
}

fn violation(
    violations: &mut Vec<FullEnvelopeMeshViolation>,
    code: &str,
    detail: impl Into<String>,
) {
    violations.push(FullEnvelopeMeshViolation {
        code: code.to_owned(),
        detail: detail.into(),
    });
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= GEOMETRY_TOLERANCE_M
}

fn coordinate_for_station(spec: &FullEnvelopeSpec, station_index: usize) -> f64 {
    spec.stations[station_index].x_over_length * spec.principal_dimensions.length_m
}

fn register_point(
    point_id: &str,
    coordinate: [f64; 3],
    vertices: &mut Vec<[f64; 3]>,
    indices: &mut BTreeMap<String, u32>,
    violations: &mut Vec<FullEnvelopeMeshViolation>,
) -> Option<u32> {
    if let Some(existing_index) = indices.get(point_id).copied() {
        let existing = vertices.get(existing_index as usize).copied()?;
        if existing != coordinate {
            violation(
                violations,
                "point_identity_coordinate_conflict",
                format!(
                    "source point identity {point_id} is reused at different exact coordinates"
                ),
            );
            return None;
        }
        return Some(existing_index);
    }

    let next_index = u32::try_from(vertices.len()).ok()?;
    vertices.push(coordinate);
    indices.insert(point_id.to_owned(), next_index);
    Some(next_index)
}

fn checked_face_budget(spec: &FullEnvelopeSpec) -> Option<usize> {
    let mut total = 0_usize;
    for interval in spec.stations.windows(2) {
        let starboard_segments = interval[0].starboard_profile.len().checked_sub(1)?;
        let port_segments = match spec.symmetry {
            SymmetryPolicy::SymmetricAboutY0 => starboard_segments,
            SymmetryPolicy::ExplicitPortAndStarboard => {
                interval[0].port_profile.as_ref()?.len().checked_sub(1)?
            }
        };
        let deck_segments = interval[0].deck_profile_segments()?;
        let side_faces = starboard_segments
            .checked_add(port_segments)?
            .checked_mul(2)?;
        let deck_faces = deck_segments.checked_mul(2)?;
        total = total.checked_add(side_faces)?.checked_add(deck_faces)?;
    }
    if spec.end_conditions.bow == HullEndCondition::Transom {
        total = total.checked_add(cap_face_budget(spec, 0)?)?;
    }
    if spec.end_conditions.stern == HullEndCondition::Transom {
        total = total.checked_add(cap_face_budget(spec, spec.stations.len() - 1)?)?;
    }
    Some(total)
}

fn cap_face_budget(spec: &FullEnvelopeSpec, station_index: usize) -> Option<usize> {
    let station = spec.stations.get(station_index)?;
    let port_count = match spec.symmetry {
        SymmetryPolicy::SymmetricAboutY0 => station.starboard_profile.len(),
        SymmetryPolicy::ExplicitPortAndStarboard => station.port_profile.as_ref()?.len(),
    };
    let perimeter_count = station
        .starboard_profile
        .len()
        .checked_add(station.deck_section_count()?)?
        .checked_add(port_count)?
        .checked_sub(3)?;
    perimeter_count.checked_sub(2)
}

trait FullEnvelopeStationBudget {
    fn deck_profile_segments(&self) -> Option<usize>;
    fn deck_section_count(&self) -> Option<usize>;
}
impl FullEnvelopeStationBudget for crate::full_envelope::FullEnvelopeStation {
    fn deck_profile_segments(&self) -> Option<usize> {
        // Filled by the caller using the matching deck section count; this
        // placeholder is replaced in preflight_face_budget below.
        None
    }

    fn deck_section_count(&self) -> Option<usize> {
        Some(self.starboard_profile.len())
    }
}

fn preflight_face_budget(spec: &FullEnvelopeSpec) -> Option<usize> {
    let mut total = 0_usize;
    for interval_index in 0..spec.stations.len().checked_sub(1)? {
        let left = &spec.stations[interval_index];
        let right = &spec.stations[interval_index + 1];
        let starboard_segments = left.starboard_profile.len().checked_sub(1)?;
        let port_segments = match spec.symmetry {
            SymmetryPolicy::SymmetricAboutY0 => starboard_segments,
            SymmetryPolicy::ExplicitPortAndStarboard => {
                left.port_profile.as_ref()?.len().checked_sub(1)?
            }
        };
        let deck_left = &spec.deck_surface.sections[interval_index].points;
        let deck_right = &spec.deck_surface.sections[interval_index + 1].points;
        let deck_faces = if deck_left.len() == 1 || deck_right.len() == 1 {
            deck_left.len().max(deck_right.len()).checked_sub(1)?
        } else {
            deck_left.len().checked_sub(1)?.checked_mul(2)?
        };
        let side_faces = starboard_segments
            .checked_add(port_segments)?
            .checked_mul(2)?;
        total = total.checked_add(side_faces)?.checked_add(deck_faces)?;
        let _ = right;
    }
    if spec.end_conditions.bow == HullEndCondition::Transom {
        total = total.checked_add(cap_face_budget_for_spec(spec, 0)?)?;
    }
    if spec.end_conditions.stern == HullEndCondition::Transom {
        total = total.checked_add(cap_face_budget_for_spec(spec, spec.stations.len() - 1)?)?;
    }
    Some(total)
}

fn cap_face_budget_for_spec(spec: &FullEnvelopeSpec, station_index: usize) -> Option<usize> {
    let station = &spec.stations[station_index];
    let deck_count = spec.deck_surface.sections[station_index].points.len();
    let port_count = match spec.symmetry {
        SymmetryPolicy::SymmetricAboutY0 => station.starboard_profile.len(),
        SymmetryPolicy::ExplicitPortAndStarboard => station.port_profile.as_ref()?.len(),
    };
    let ring_count = station
        .starboard_profile
        .len()
        .checked_add(deck_count)?
        .checked_add(port_count)?
        .checked_sub(2)?;
    ring_count.checked_sub(2)
}

fn register_station_vertices(
    spec: &FullEnvelopeSpec,
    station_index: usize,
    vertices: &mut Vec<[f64; 3]>,
    indices: &mut BTreeMap<String, u32>,
    violations: &mut Vec<FullEnvelopeMeshViolation>,
) -> Option<StationVertexLayout> {
    let station = &spec.stations[station_index];
    let x_m = coordinate_for_station(spec, station_index);
    let mut starboard = Vec::with_capacity(station.starboard_profile.len());
    for point in &station.starboard_profile {
        starboard.push(register_point(
            &point.point_id,
            [x_m, point.y_m, point.z_m],
            vertices,
            indices,
            violations,
        )?);
    }

    let mut port = Vec::new();
    match spec.symmetry {
        SymmetryPolicy::SymmetricAboutY0 => {
            port.reserve(station.starboard_profile.len());
            for point in &station.starboard_profile {
                let id = if close(point.y_m, 0.0) {
                    point.point_id.clone()
                } else {
                    format!("MIRROR:{}", point.point_id)
                };
                port.push(register_point(
                    &id,
                    [x_m, -point.y_m, point.z_m],
                    vertices,
                    indices,
                    violations,
                )?);
            }
        }
        SymmetryPolicy::ExplicitPortAndStarboard => {
            let profile = station.port_profile.as_ref()?;
            port.reserve(profile.len());
            for point in profile {
                port.push(register_point(
                    &point.point_id,
                    [x_m, point.y_m, point.z_m],
                    vertices,
                    indices,
                    violations,
                )?);
            }
        }
    }

    let deck_section = &spec.deck_surface.sections[station_index];
    let mut deck = Vec::with_capacity(deck_section.points.len());
    for point in &deck_section.points {
        deck.push(register_point(
            &point.point_id,
            [x_m, point.y_m, point.z_m],
            vertices,
            indices,
            violations,
        )?);
    }

    Some(StationVertexLayout {
        starboard,
        port,
        deck,
    })
}

fn push_side_faces(
    spec: &FullEnvelopeSpec,
    left: &StationVertexLayout,
    right: &StationVertexLayout,
    interval_index: usize,
    groups: &mut FaceGroups,
) -> Result<(), FullEnvelopeMeshViolation> {
    let left_station = &spec.stations[interval_index];
    let profile = &left_station.starboard_profile;
    for point_index in 0..profile.len() - 1 {
        let region = profile[point_index].region_to_next;
        let a = left.starboard[point_index];
        let b = right.starboard[point_index];
        let c = right.starboard[point_index + 1];
        let d = left.starboard[point_index + 1];
        match region {
            ProfileRegion::Bottom => groups.bottom.extend([[a, b, c], [a, c, d]]),
            ProfileRegion::Topsides => groups.topsides.extend([[a, b, c], [a, c, d]]),
            unsupported => {
                return Err(FullEnvelopeMeshViolation {
                    code: "unsupported_starboard_profile_region".into(),
                    detail: format!(
                        "station {} segment {} has unsupported region {:?}",
                        left_station.station_id, point_index, unsupported
                    ),
                });
            }
        }
    }

    let port_profile = match spec.symmetry {
        SymmetryPolicy::SymmetricAboutY0 => &left_station.starboard_profile,
        SymmetryPolicy::ExplicitPortAndStarboard => left_station.port_profile.as_ref().ok_or_else(|| {
            FullEnvelopeMeshViolation {
                code: "explicit_port_profile_missing".into(),
                detail: format!("station {} has no explicit port profile", left_station.station_id),
            }
        })?,
    };
    for point_index in 0..port_profile.len() - 1 {
        let region = port_profile[point_index].region_to_next;
        let a = left.port[point_index];
        let b = right.port[point_index];
        let c = right.port[point_index + 1];
        let d = left.port[point_index + 1];
        match region {
            ProfileRegion::Bottom => groups.bottom.extend([[a, c, b], [a, d, c]]),
            ProfileRegion::Topsides => groups.topsides.extend([[a, c, b], [a, d, c]]),
            unsupported => {
                return Err(FullEnvelopeMeshViolation {
                    code: "unsupported_port_profile_region".into(),
                    detail: format!(
                        "station {} segment {} has unsupported region {:?}",
                        left_station.station_id, point_index, unsupported
                    ),
                });
            }
        }
    }
    Ok(())
}

fn push_deck_faces(
    left: &StationVertexLayout,
    right: &StationVertexLayout,
    interval_index: usize,
    groups: &mut FaceGroups,
) -> Result<(), FullEnvelopeMeshViolation> {
    match (left.deck.as_slice(), right.deck.as_slice()) {
        (left_points, right_points) if left_points.len() >= 2 && left_points.len() == right_points.len() => {
            for point_index in 0..left_points.len() - 1 {
                let a = left_points[point_index];
                let b = right_points[point_index];
                let c = right_points[point_index + 1];
                let d = left_points[point_index + 1];
                groups.deck.extend([[a, c, b], [a, d, c]]);
            }
        }
        ([point], right_points) if right_points.len() >= 2 => {
            for point_index in 0..right_points.len() - 1 {
                let b = right_points[point_index];
                let c = right_points[point_index + 1];
                groups.deck.push([*point, c, b]);
            }
        }
        (left_points, [point]) if left_points.len() >= 2 => {
            for point_index in 0..left_points.len() - 1 {
                let a = left_points[point_index];
                let b = left_points[point_index + 1];
                groups.deck.push([*point, a, b]);
            }
        }
        _ => {
            return Err(FullEnvelopeMeshViolation {
                code: "deck_station_topology".into(),
                detail: format!(
                    "station interval {interval_index} has incompatible deck-section point counts"
                ),
            });
        }
    }
    Ok(())
}

fn end_perimeter(
    spec: &FullEnvelopeSpec,
    station_index: usize,
    layout: &StationVertexLayout,
) -> Result<Vec<u32>, FullEnvelopeMeshViolation> {
    let mut ring = Vec::new();
    ring.extend(layout.starboard.iter().copied());
    for index in layout.deck.iter().rev().skip(1) {
        ring.push(*index);
    }
    for index in layout.port.iter().rev().skip(1) {
        ring.push(*index);
    }
    if ring.len() < 3 {
        return Err(FullEnvelopeMeshViolation {
            code: "end_perimeter_size".into(),
            detail: format!("station {} end perimeter has fewer than three vertices", spec.stations[station_index].station_id),
        });
    }
    if ring.first() == ring.last() {
        ring.pop();
    }
    Ok(ring)
}

fn cross_yz(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1])
}

fn is_convex_end_perimeter(
    ring: &[u32],
    vertices: &[[f64; 3]],
) -> bool {
    if ring.len() < 3 {
        return false;
    }
    let scale = vertices
        .iter()
        .map(|point| point[1].abs().max(point[2].abs()))
        .fold(1.0_f64, f64::max);
    let tolerance = GEOMETRY_TOLERANCE_M * scale;
    let mut sign = 0_i8;
    for index in 0..ring.len() {
        let a = vertices[ring[index] as usize];
        let b = vertices[ring[(index + 1) % ring.len()] as usize];
        let c = vertices[ring[(index + 2) % ring.len()] as usize];
        let turn = cross_yz(a, b, c);
        if turn.abs() <= tolerance {
            continue;
        }
        let current_sign = if turn > 0.0 { 1 } else { -1 };
        if sign != 0 && current_sign != sign {
            return false;
        }
        sign = current_sign;
    }
    sign != 0
}

fn push_end_cap(
    ring: &[u32],
    bow: bool,
    groups: &mut FaceGroups,
) {
    let target = if bow { &mut groups.bow } else { &mut groups.stern };
    for index in 1..ring.len() - 1 {
        if bow {
            target.push([ring[0], ring[index], ring[index + 1]]);
        } else {
            target.push([ring[0], ring[index + 1], ring[index]]);
        }
    }
}

fn build_point_vertex_mapping(
    indices: &BTreeMap<String, u32>,
) -> Vec<SourcePointVertex> {
    indices
        .iter()
        .map(|(point_id, vertex_index)| SourcePointVertex {
            point_id: point_id.clone(),
            vertex_index: *vertex_index,
        })
        .collect()
}

fn digest_mapping(mapping: &[SourcePointVertex]) -> Result<String, serde_json::Error> {
    Ok(blake3::hash(&serde_json::to_vec(mapping)?).to_hex().to_string())
}

fn compute_manifest_digest(
    artifact: &FullEnvelopeMeshArtifact,
) -> Result<String, serde_json::Error> {
    let identity = MeshManifestIdentity {
        artifact_schema_version: artifact.artifact_schema_version,
        generator: &artifact.generator,
        source_spec_digest_blake3: &artifact.source_spec_digest_blake3,
        point_vertex_mapping_digest_blake3: &artifact.point_vertex_mapping_digest_blake3,
        mesh_digest_blake3: &artifact.mesh_digest_blake3,
        bottom_face_range: artifact.bottom_face_range,
        topsides_face_range: artifact.topsides_face_range,
        deck_face_range: artifact.deck_face_range,
        bow_end_face_range: artifact.bow_end_face_range,
        stern_end_face_range: artifact.stern_end_face_range,
        point_vertex_mapping: &artifact.point_vertex_mapping,
        candidate_schema_version: artifact.candidate.schema_version,
        candidate_id: &artifact.candidate.candidate_id,
        geometry_digest_blake3: &artifact.candidate.geometry_digest_blake3,
        admission_profile: &artifact.candidate.profile,
    };
    Ok(blake3::hash(&serde_json::to_vec(&identity)?).to_hex().to_string())
}

fn range_for(faces: &mut Vec<[u32; 3]>, source: &[[u32; 3]]) -> FullEnvelopeFaceRange {
    let range = FullEnvelopeFaceRange {
        first_face: faces.len(),
        face_count: source.len(),
    };
    faces.extend_from_slice(source);
    range
}

/// Generate a deterministic full external hull surface from a validated
/// G2f-a source specification. The supplied admission profile is applied
/// afresh. This generator supports closed shells only; it does not heal or
/// weld geometry.
pub fn generate_full_envelope_mesh(
    spec: &FullEnvelopeSpec,
    profile: AdmissionProfile,
) -> Result<FullEnvelopeMeshArtifact, Vec<FullEnvelopeMeshViolation>> {
    let mut violations = Vec::new();
    let validation = validate_full_envelope_spec(spec);
    match validation.disposition {
        FullEnvelopeDisposition::ValidInputContract => {}
        FullEnvelopeDisposition::Incomplete => {
            violation(&mut violations, "source_spec_incomplete", "source full-envelope spec contains unsupported features");
        }
        FullEnvelopeDisposition::Rejected => {
            violation(&mut violations, "source_spec_rejected", "source full-envelope spec fails its typed input contract");
        }
    }
    if spec.boundary_semantics.exterior_shell != crate::full_envelope::ExteriorShellSemantics::ClosedShell
        || !spec.boundary_semantics.openings.is_empty()
    {
        violation(&mut violations, "openings_not_supported", "G2f-b v1 generates closed shells only; openings and cut-outs are not yet supported");
    }
    if spec.deck_surface.construction == DeckConstruction::Unsupported {
        violation(&mut violations, "deck_construction_unsupported", "G2f-b requires a supported deck construction");
    }
    if spec.stations.len() > crate::full_envelope::MAX_FULL_ENVELOPE_STATIONS {
        violation(&mut violations, "station_budget", "station count exceeds the G2f-b generation budget");
    }

    for interval in 0..spec.stations.len().saturating_sub(1) {
        let left = &spec.stations[interval];
        let right = &spec.stations[interval + 1];
        if left.starboard_profile.len() != right.starboard_profile.len()
            || left.starboard_profile.iter().zip(&right.starboard_profile).any(|(a, b)| {
                a.role != b.role || a.region_to_next != b.region_to_next
            })
        {
            violation(
                &mut violations,
                "starboard_station_topology",
                format!("stations {} and {} have incompatible starboard profile semantics", left.station_id, right.station_id),
            );
        }
        if spec.symmetry == SymmetryPolicy::ExplicitPortAndStarboard {
            let left_port = left.port_profile.as_ref();
            let right_port = right.port_profile.as_ref();
            if left_port.is_none() || right_port.is_none()
                || left_port.unwrap().len() != right_port.unwrap().len()
                || left_port.unwrap().iter().zip(right_port.unwrap()).any(|(a, b)| {
                    a.role != b.role || a.region_to_next != b.region_to_next
                })
            {
                violation(
                    &mut violations,
                    "port_station_topology",
                    format!("stations {} and {} have incompatible explicit port profile semantics", left.station_id, right.station_id),
                );
            }
        }
        let left_deck = &spec.deck_surface.sections[interval].points;
        let right_deck = &spec.deck_surface.sections[interval + 1].points;
        let pointed_interval = left_deck.len() == 1 || right_deck.len() == 1;
        if (!pointed_interval && left_deck.len() != right_deck.len())
            || (pointed_interval && left_deck.len() != 1 && right_deck.len() != 1)
        {
            violation(
                &mut violations,
                "deck_station_topology",
                format!("stations {} and {} have incompatible deck section counts", left.station_id, right.station_id),
            );
        }
        if !pointed_interval && left_deck.iter().zip(right_deck).any(|(a, b)| a.role != b.role) {
            violation(
                &mut violations,
                "deck_station_semantics",
                format!("stations {} and {} have incompatible deck point roles", left.station_id, right.station_id),
            );
        }
    }

    if let Some(face_budget) = preflight_face_budget(spec) {
        if face_budget > MAX_MESH_FACES {
            violation(&mut violations, "mesh_face_budget", format!("estimated face count {face_budget} exceeds {MAX_MESH_FACES}"));
        }
    } else {
        violation(&mut violations, "mesh_face_budget_overflow", "face-count preflight overflowed");
    }
    let vertex_upper_bound = spec.stations.iter().try_fold(0_usize, |total, station| {
        let port_count = match spec.symmetry {
            SymmetryPolicy::SymmetricAboutY0 => station.starboard_profile.len(),
            SymmetryPolicy::ExplicitPortAndStarboard => station.port_profile.as_ref()?.len(),
        };
        let deck_count = spec.deck_surface.sections.iter().find(|section| section.station_id == station.station_id)?.points.len();
        total.checked_add(station.starboard_profile.len())?
            .checked_add(port_count)?
            .checked_add(deck_count)
    });
    match vertex_upper_bound {
        Some(count) if count <= MAX_MESH_VERTICES => {}
        Some(count) => violation(&mut violations, "mesh_vertex_budget", format!("estimated vertex count {count} exceeds {MAX_MESH_VERTICES}")),
        None => violation(&mut violations, "mesh_vertex_budget_overflow", "vertex-count preflight overflowed or a port/deck station is missing"),
    }
    if !violations.is_empty() {
        return Err(violations);
    }

    let mut vertices = Vec::<[f64; 3]>::new();
    let mut point_indices = BTreeMap::<String, u32>::new();
    let mut station_layouts = Vec::with_capacity(spec.stations.len());
    for station_index in 0..spec.stations.len() {
        let layout = register_station_vertices(
            spec,
            station_index,
            &mut vertices,
            &mut point_indices,
            &mut violations,
        );
        match layout {
            Some(layout) => station_layouts.push(layout),
            None => violation(&mut violations, "point_vertex_mapping", format!("could not register source point identities for station {}", spec.stations[station_index].station_id)),
        }
    }
    if !violations.is_empty() {
        return Err(violations);
    }

    let mut groups = FaceGroups::default();
    for interval in 0..station_layouts.len() - 1 {
        if let Err(error) = push_side_faces(spec, &station_layouts[interval], &station_layouts[interval + 1], interval, &mut groups) {
            violations.push(error);
        }
        if let Err(error) = push_deck_faces(&station_layouts[interval], &station_layouts[interval + 1], interval, &mut groups) {
            violations.push(error);
        }
    }
    if !violations.is_empty() {
        return Err(violations);
    }

    if spec.end_conditions.bow == HullEndCondition::Transom {
        let ring = match end_perimeter(spec, 0, &station_layouts[0]) {
            Ok(ring) => ring,
            Err(error) => { violations.push(error); Vec::new() }
        };
        if !ring.is_empty() && !is_convex_end_perimeter(&ring, &vertices) {
            violation(&mut violations, "bow_transom_non_convex", "bow transom perimeter is not convex under the supported G2f-b v1 cap policy");
        } else if !ring.is_empty() {
            push_end_cap(&ring, true, &mut groups);
        }
    }
    let stern_index = spec.stations.len() - 1;
    if spec.end_conditions.stern == HullEndCondition::Transom {
        let ring = match end_perimeter(spec, stern_index, &station_layouts[stern_index]) {
            Ok(ring) => ring,
            Err(error) => { violations.push(error); Vec::new() }
        };
        if !ring.is_empty() && !is_convex_end_perimeter(&ring, &vertices) {
            violation(&mut violations, "stern_transom_non_convex", "stern transom perimeter is not convex under the supported G2f-b v1 cap policy");
        } else if !ring.is_empty() {
            push_end_cap(&ring, false, &mut groups);
        }
    }
    if !violations.is_empty() {
        return Err(violations);
    }

    let mut faces = Vec::<[u32; 3]>::new();
    let bottom_face_range = range_for(&mut faces, &groups.bottom);
    let topsides_face_range = range_for(&mut faces, &groups.topsides);
    let deck_face_range = range_for(&mut faces, &groups.deck);
    let bow_end_face_range = range_for(&mut faces, &groups.bow);
    let stern_end_face_range = range_for(&mut faces, &groups.stern);

    let coordinate_frame = CoordinateFrame {
        id: spec.coordinate_frame.id.clone(),
        axis_convention: spec.coordinate_frame.axis_convention.clone(),
        handedness: spec.coordinate_frame.handedness.clone(),
    };
    let geometry = MeshGeometry {
        units: spec.units.clone(),
        coordinate_frame,
        vertices,
        faces,
    };
    let geometry_digest_blake3 = compute_geometry_digest(&geometry)
        .map_err(|error| vec![FullEnvelopeMeshViolation {
            code: "geometry_digest".into(),
            detail: format!("could not compute mesh digest: {error}"),
        }])?;
    let candidate = CandidateInput {
        schema_version: 1,
        candidate_id: spec.candidate_id.clone(),
        geometry_digest_blake3: geometry_digest_blake3.clone(),
        geometry,
        profile,
    };
    let admission_report = admit(&candidate);
    if admission_report.disposition != Disposition::AdmissibleForProfile {
        return Err(vec![FullEnvelopeMeshViolation {
            code: "mesh_admission_rejected".into(),
            detail: format!(
                "fresh G0/G1 admission returned {:?}; boundary={}, nonmanifold-edge={}, vertex-link={}, degenerate={}, duplicate={}, volume={:?}",
                admission_report.disposition,
                admission_report.metrics.boundary_edge_count,
                admission_report.metrics.non_manifold_edge_count,
                admission_report.metrics.non_manifold_vertex_link_count,
                admission_report.metrics.degenerate_face_count,
                admission_report.metrics.duplicate_face_count,
                admission_report.metrics.signed_volume_m3
            ),
        }]);
    }

    let point_vertex_mapping = build_point_vertex_mapping(&point_indices);
    let point_vertex_mapping_digest_blake3 = digest_mapping(&point_vertex_mapping)
        .map_err(|error| vec![FullEnvelopeMeshViolation {
            code: "point_vertex_mapping_digest".into(),
            detail: format!("could not serialize point-vertex mapping: {error}"),
        }])?;
    let source_spec_digest_blake3 = compute_full_envelope_spec_digest(spec)
        .map_err(|error| vec![FullEnvelopeMeshViolation {
            code: "source_spec_digest".into(),
            detail: format!("could not serialize source specification: {error}"),
        }])?;
    let mut artifact = FullEnvelopeMeshArtifact {
        artifact_schema_version: FULL_ENVELOPE_MESH_SCHEMA_VERSION,
        generator: FULL_ENVELOPE_MESH_GENERATOR.into(),
        source_spec_digest_blake3,
        point_vertex_mapping_digest_blake3,
        mesh_digest_blake3: geometry_digest_blake3,
        manifest_digest_blake3: String::new(),
        bottom_face_range,
        topsides_face_range,
        deck_face_range,
        bow_end_face_range,
        stern_end_face_range,
        point_vertex_mapping,
        candidate,
        admission_report,
    };
    artifact.manifest_digest_blake3 = compute_manifest_digest(&artifact)
        .map_err(|error| vec![FullEnvelopeMeshViolation {
            code: "manifest_digest".into(),
            detail: format!("could not serialize full-envelope mesh manifest: {error}"),
        }])?;
    Ok(artifact)
}

/// Rebuild the expected mesh from the exact source specification and compare
/// the complete artifact. Rehashing modified geometry or face metadata cannot
/// make the mutation pass because semantic expectations are freshly rebuilt.
pub fn verify_full_envelope_mesh_artifact(
    spec: &FullEnvelopeSpec,
    artifact: &FullEnvelopeMeshArtifact,
) -> Result<(), Vec<FullEnvelopeMeshViolation>> {
    let expected = generate_full_envelope_mesh(spec, artifact.candidate.profile.clone())?;
    if expected != *artifact {
        return Err(vec![FullEnvelopeMeshViolation {
            code: "full_envelope_mesh_artifact_mismatch".into(),
            detail: "artifact differs from a fresh full-envelope mesh reconstruction of the exact source specification and admission profile".into(),
        }]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> FullEnvelopeSpec {
        let contents = match name {
            "box" => include_str!("../../../../docs/research/marine/fixtures/marine-full-envelope-box-001.json"),
            "asymmetric" => include_str!("../../../../docs/research/marine/fixtures/marine-full-envelope-asymmetric-001.json"),
            "pointed-bow-asymmetric" => include_str!("../../../../docs/research/marine/fixtures/marine-full-envelope-pointed-bow-asymmetric-001.json"),
            _ => panic!("unknown fixture {name}"),
        };
        serde_json::from_str::<FullEnvelopeSpec>(contents).unwrap()
    }

    fn profile() -> AdmissionProfile {
        AdmissionProfile {
            id: "marine-full-external-shell-v1".into(),
            require_closed_surface: true,
            minimum_face_area_m2: 1.0e-12,
            minimum_enclosed_volume_m3: 1.0e-12,
            additional_required_checks: Vec::new(),
        }
    }

    #[test]
    fn box_hull_generates_a_closed_full_external_shell_with_20500_cubic_metres() {
        let artifact = generate_full_envelope_mesh(&fixture("box"), profile()).unwrap();
        assert_eq!(artifact.admission_report.disposition, Disposition::AdmissibleForProfile);
        assert_eq!(artifact.admission_report.metrics.boundary_edge_count, 0);
        assert_eq!(artifact.admission_report.metrics.non_manifold_edge_count, 0);
        assert_eq!(artifact.admission_report.metrics.non_manifold_vertex_link_count, 0);
        assert!(
            (artifact.admission_report.metrics.signed_volume_m3.unwrap() - 20_500.0).abs()
                < 1.0e-8
        );
        assert!(artifact.bottom_face_range.face_count > 0);
        assert!(artifact.topsides_face_range.face_count > 0);
        assert!(artifact.deck_face_range.face_count > 0);
        assert!(artifact.bow_end_face_range.face_count > 0);
        assert!(artifact.stern_end_face_range.face_count > 0);
        verify_full_envelope_mesh_artifact(&fixture("box"), &artifact).unwrap();
    }

    #[test]
    fn asymmetric_fixture_uses_real_port_geometry_and_is_manifold() {
        let source = fixture("asymmetric");
        let artifact = generate_full_envelope_mesh(&source, profile()).unwrap();
        assert_eq!(artifact.admission_report.disposition, Disposition::AdmissibleForProfile);
        assert_eq!(artifact.admission_report.metrics.boundary_edge_count, 0);
        assert_eq!(artifact.admission_report.metrics.non_manifold_edge_count, 0);
        assert_eq!(artifact.admission_report.metrics.non_manifold_vertex_link_count, 0);
        assert!(
            artifact
                .point_vertex_mapping
                .iter()
                .any(|identity| identity.point_id.starts_with("PORT-"))
        );
    }

    #[test]
    fn pointed_bow_with_transom_stern_has_no_bow_cap_and_one_stern_cap() {
        let source = fixture("pointed-bow-asymmetric");
        let artifact = generate_full_envelope_mesh(&source, profile()).unwrap();
        assert_eq!(artifact.bow_end_face_range.face_count, 0);
        assert!(artifact.stern_end_face_range.face_count > 0);
        assert_eq!(artifact.admission_report.metrics.boundary_edge_count, 0);
        verify_full_envelope_mesh_artifact(&source, &artifact).unwrap();
    }

    #[test]
    fn identical_source_and_profile_reproduce_identical_artifact() {
        let source = fixture("box");
        let first = generate_full_envelope_mesh(&source, profile()).unwrap();
        let second = generate_full_envelope_mesh(&source, profile()).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn rehashed_face_mutation_is_rejected_by_full_reconstruction() {
        let source = fixture("box");
        let mut artifact = generate_full_envelope_mesh(&source, profile()).unwrap();
        let face_index = artifact.deck_face_range.first_face;
        artifact.candidate.geometry.faces[face_index][0] =
            artifact.candidate.geometry.faces[face_index][1];
        let digest = compute_geometry_digest(&artifact.candidate.geometry).unwrap();
        artifact.candidate.geometry_digest_blake3 = digest.clone();
        artifact.mesh_digest_blake3 = digest;
        artifact.admission_report = admit(&artifact.candidate);
        artifact.manifest_digest_blake3 = compute_manifest_digest(&artifact).unwrap();
        assert!(verify_full_envelope_mesh_artifact(&source, &artifact).is_err());
    }

    #[test]
    fn unsupported_openings_return_incomplete_instead_of_filling_them() {
        let mut source = fixture("box");
        source.boundary_semantics.exterior_shell =
            crate::full_envelope::ExteriorShellSemantics::ExplicitBoundaryLoops;
        source.boundary_semantics.openings.push(crate::full_envelope::OpeningBoundary {
            opening_id: "HATCH-001".into(),
            boundary_point_refs: vec![
                "BOW-KEEL".into(),
                "BOW-WL-S".into(),
                "BOW-SHEER-S".into(),
            ],
            status: crate::full_envelope::OpeningStatus::Supported,
        });
        let errors = generate_full_envelope_mesh(&source, profile()).unwrap_err();
        assert!(errors.iter().any(|item| item.code == "openings_not_supported"));
    }
}
