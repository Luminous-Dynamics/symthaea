// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic CPU tessellation from canonical 3D Mycelial Presence curves.
//!
//! This module deliberately has no Bevy, wgpu, or GPU dependency. It produces
//! indexed triangle geometry that any renderer can upload or transform.

use std::error::Error;
use std::fmt;

use crate::mycelial_model::{
    MYCELIAL_MODEL_VERSION, MycelialFilament, MycelialModel, SpatialPoint3,
};

pub const MAX_MYCELIAL_FILAMENTS: usize = 512;
pub const MAX_TUBE_MESH_VERTICES: usize = 1_000_000;
const MAX_FILAMENT_COORDINATE: f32 = 10_000.0;
const MAX_FILAMENT_RADIUS: f32 = 100.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TubeTessellationConfig {
    /// Number of Bézier intervals along each filament.
    pub axial_segments: u8,
    /// Number of vertices around each cross-section.
    pub radial_segments: u8,
}

impl Default for TubeTessellationConfig {
    fn default() -> Self {
        Self {
            axial_segments: 12,
            radial_segments: 6,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TubeTessellationError {
    AxialResolutionOutOfRange(u8),
    RadialResolutionOutOfRange(u8),
    UnsupportedModelVersion(u16),
    FilamentCountExceeded(usize),
    MeshVertexLimitExceeded(usize),
    InvalidFilamentGeometry(u32),
    InvalidFilamentTopology,
    VertexCountOverflow,
}

impl fmt::Display for TubeTessellationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AxialResolutionOutOfRange(value) => {
                write!(f, "axial segments must be in 2..=128, got {value}")
            }
            Self::RadialResolutionOutOfRange(value) => {
                write!(f, "radial segments must be in 3..=32, got {value}")
            }
            Self::UnsupportedModelVersion(version) => {
                write!(f, "unsupported MycelialModel version {version}")
            }
            Self::FilamentCountExceeded(count) => {
                write!(
                    f,
                    "model has {count} filaments; limit is {MAX_MYCELIAL_FILAMENTS}"
                )
            }
            Self::MeshVertexLimitExceeded(count) => {
                write!(
                    f,
                    "mesh requires {count} vertices; limit is {MAX_TUBE_MESH_VERTICES}"
                )
            }
            Self::InvalidFilamentGeometry(id) => {
                write!(f, "filament {id} has non-finite points or invalid radii")
            }
            Self::InvalidFilamentTopology => {
                write!(
                    f,
                    "filament IDs must be unique and parent references must exist"
                )
            }
            Self::VertexCountOverflow => write!(f, "tube mesh exceeds u32 vertex indexing"),
        }
    }
}

impl Error for TubeTessellationError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilamentMeshRange {
    pub filament_id: u32,
    pub parent_id: Option<u32>,
    pub generation: u8,
    pub terminal: bool,
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub first_index: u32,
    pub index_count: u32,
}

/// Combined indexed triangle mesh. Each filament has a range so renderers can
/// select, recolor, cull, or animate individual semantic branches.
#[derive(Debug, Clone, PartialEq)]
pub struct MycelialTubeMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    pub filaments: Vec<FilamentMeshRange>,
}

impl MycelialTubeMesh {
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Basic structural validation suitable before a renderer uploads buffers.
    pub fn validate(&self) -> bool {
        self.positions.len() == self.normals.len()
            && self.indices.len().is_multiple_of(3)
            && self
                .positions
                .iter()
                .all(|point| point.iter().all(|v| v.is_finite()))
            && self
                .normals
                .iter()
                .all(|normal| normal.iter().all(|v| v.is_finite()))
            && self
                .indices
                .iter()
                .all(|index| (*index as usize) < self.positions.len())
            && self.filaments.iter().all(|range| {
                (range.first_vertex as usize)
                    .checked_add(range.vertex_count as usize)
                    .is_some_and(|end| end <= self.positions.len())
                    && (range.first_index as usize)
                        .checked_add(range.index_count as usize)
                        .is_some_and(|end| end <= self.indices.len())
            })
    }
}

/// Turn every cubic filament into a tapered tube using parallel-transport
/// frames. This avoids the sudden twists common with independently rebuilt
/// Frenet frames at low-curvature points.
pub fn tessellate_mycelial_model(
    model: &MycelialModel,
    config: TubeTessellationConfig,
) -> Result<MycelialTubeMesh, TubeTessellationError> {
    validate_config(config)?;
    if model.schema_version != MYCELIAL_MODEL_VERSION {
        return Err(TubeTessellationError::UnsupportedModelVersion(
            model.schema_version,
        ));
    }
    if model.filaments.len() > MAX_MYCELIAL_FILAMENTS {
        return Err(TubeTessellationError::FilamentCountExceeded(
            model.filaments.len(),
        ));
    }

    // Parent-before-child order rejects cycles and makes ranges safe to replay.
    let mut seen_ids = std::collections::HashSet::with_capacity(model.filaments.len());
    for filament in &model.filaments {
        if filament
            .parent_id
            .is_some_and(|parent_id| parent_id == filament.id || !seen_ids.contains(&parent_id))
        {
            return Err(TubeTessellationError::InvalidFilamentTopology);
        }
        if !seen_ids.insert(filament.id) {
            return Err(TubeTessellationError::InvalidFilamentTopology);
        }
        if !valid_filament_geometry(filament) {
            return Err(TubeTessellationError::InvalidFilamentGeometry(filament.id));
        }
    }

    let vertices_per_filament =
        (config.axial_segments as usize + 1) * config.radial_segments as usize;
    let estimated_vertices = model
        .filaments
        .len()
        .checked_mul(vertices_per_filament)
        .ok_or(TubeTessellationError::VertexCountOverflow)?;
    if estimated_vertices > MAX_TUBE_MESH_VERTICES {
        return Err(TubeTessellationError::MeshVertexLimitExceeded(
            estimated_vertices,
        ));
    }

    let mut mesh = MycelialTubeMesh {
        positions: Vec::with_capacity(estimated_vertices),
        normals: Vec::with_capacity(estimated_vertices),
        indices: Vec::new(),
        filaments: Vec::with_capacity(model.filaments.len()),
    };

    for filament in &model.filaments {
        tessellate_filament(filament, config, &mut mesh)?;
    }

    debug_assert!(mesh.validate());
    Ok(mesh)
}

fn valid_filament_geometry(filament: &MycelialFilament) -> bool {
    let bounded_point = |point: SpatialPoint3| {
        point.is_finite()
            && point.x.abs() <= MAX_FILAMENT_COORDINATE
            && point.y.abs() <= MAX_FILAMENT_COORDINATE
            && point.z.abs() <= MAX_FILAMENT_COORDINATE
    };
    bounded_point(filament.start)
        && bounded_point(filament.control_a)
        && bounded_point(filament.control_b)
        && bounded_point(filament.end)
        && filament.radius_start.is_finite()
        && filament.radius_end.is_finite()
        && filament.radius_start > 0.0
        && filament.radius_end > 0.0
        && filament.radius_start <= MAX_FILAMENT_RADIUS
        && filament.radius_end <= MAX_FILAMENT_RADIUS
}

fn validate_config(config: TubeTessellationConfig) -> Result<(), TubeTessellationError> {
    if !(2..=128).contains(&config.axial_segments) {
        return Err(TubeTessellationError::AxialResolutionOutOfRange(
            config.axial_segments,
        ));
    }
    if !(3..=32).contains(&config.radial_segments) {
        return Err(TubeTessellationError::RadialResolutionOutOfRange(
            config.radial_segments,
        ));
    }
    Ok(())
}

fn tessellate_filament(
    filament: &MycelialFilament,
    config: TubeTessellationConfig,
    mesh: &mut MycelialTubeMesh,
) -> Result<(), TubeTessellationError> {
    let axial = config.axial_segments as usize;
    let radial = config.radial_segments as usize;
    let filament_vertex_count = (axial + 1) * radial;
    let filament_index_count = axial * radial * 6;

    let first_vertex = u32::try_from(mesh.positions.len())
        .map_err(|_| TubeTessellationError::VertexCountOverflow)?;
    let first_index = u32::try_from(mesh.indices.len())
        .map_err(|_| TubeTessellationError::VertexCountOverflow)?;
    let final_vertex_count = mesh
        .positions
        .len()
        .checked_add(filament_vertex_count)
        .and_then(|count| u32::try_from(count).ok())
        .ok_or(TubeTessellationError::VertexCountOverflow)?;
    let final_index_count = mesh
        .indices
        .len()
        .checked_add(filament_index_count)
        .and_then(|count| u32::try_from(count).ok())
        .ok_or(TubeTessellationError::VertexCountOverflow)?;

    let mut previous_normal: Option<SpatialPoint3> = None;
    for axial_index in 0..=axial {
        let t = axial_index as f32 / axial as f32;
        let center = cubic_bezier(
            filament.start,
            filament.control_a,
            filament.control_b,
            filament.end,
            t,
        );
        let tangent = cubic_bezier_tangent(
            filament.start,
            filament.control_a,
            filament.control_b,
            filament.end,
            t,
        )
        .normalized();

        let normal = transport_normal(tangent, previous_normal);
        let binormal = tangent.cross(normal).normalized();
        previous_normal = Some(normal);

        let radius = filament.radius_start + (filament.radius_end - filament.radius_start) * t;
        for radial_index in 0..radial {
            let angle = radial_index as f32 * std::f32::consts::TAU / radial as f32;
            let radial_normal = normal
                .scale(angle.cos())
                .add(binormal.scale(angle.sin()))
                .normalized();
            let position = center.add(radial_normal.scale(radius));
            mesh.positions.push([position.x, position.y, position.z]);
            mesh.normals
                .push([radial_normal.x, radial_normal.y, radial_normal.z]);
        }
    }

    for axial_index in 0..axial {
        for radial_index in 0..radial {
            let next_radial = (radial_index + 1) % radial;
            let a = first_vertex + (axial_index * radial + radial_index) as u32;
            let b = first_vertex + (axial_index * radial + next_radial) as u32;
            let c = first_vertex + ((axial_index + 1) * radial + radial_index) as u32;
            let d = first_vertex + ((axial_index + 1) * radial + next_radial) as u32;
            // Circumferential-then-axial winding matches the generated outward normals.
            mesh.indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }

    mesh.filaments.push(FilamentMeshRange {
        filament_id: filament.id,
        parent_id: filament.parent_id,
        generation: filament.generation,
        terminal: filament.terminal,
        first_vertex,
        vertex_count: u32::try_from(filament_vertex_count)
            .map_err(|_| TubeTessellationError::VertexCountOverflow)?,
        first_index,
        index_count: u32::try_from(filament_index_count)
            .map_err(|_| TubeTessellationError::VertexCountOverflow)?,
    });

    debug_assert_eq!(mesh.positions.len(), final_vertex_count as usize);
    debug_assert_eq!(mesh.indices.len(), final_index_count as usize);
    Ok(())
}

fn cubic_bezier(
    p0: SpatialPoint3,
    p1: SpatialPoint3,
    p2: SpatialPoint3,
    p3: SpatialPoint3,
    t: f32,
) -> SpatialPoint3 {
    let u = 1.0 - t;
    p0.scale(u * u * u)
        .add(p1.scale(3.0 * u * u * t))
        .add(p2.scale(3.0 * u * t * t))
        .add(p3.scale(t * t * t))
}

fn cubic_bezier_tangent(
    p0: SpatialPoint3,
    p1: SpatialPoint3,
    p2: SpatialPoint3,
    p3: SpatialPoint3,
    t: f32,
) -> SpatialPoint3 {
    let u = 1.0 - t;
    p1.sub(p0)
        .scale(3.0 * u * u)
        .add(p2.sub(p1).scale(6.0 * u * t))
        .add(p3.sub(p2).scale(3.0 * t * t))
}

fn transport_normal(
    tangent: SpatialPoint3,
    previous_normal: Option<SpatialPoint3>,
) -> SpatialPoint3 {
    if let Some(previous) = previous_normal {
        let projected = previous.sub(tangent.scale(previous.dot(tangent)));
        if projected.length() > 1.0e-5 {
            return projected.normalized();
        }
    }

    let reference = if tangent.z.abs() > 0.82 {
        SpatialPoint3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        }
    } else {
        SpatialPoint3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        }
    };
    tangent.cross(reference).normalized()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embodiment::EmbodimentSnapshot;
    use crate::mycelial_model::{PresenceMode, build_mycelial_model};

    #[test]
    fn tessellation_has_predictable_topology_and_valid_indices() {
        let model = build_mycelial_model(&EmbodimentSnapshot::default(), PresenceMode::Ambient);
        let config = TubeTessellationConfig {
            axial_segments: 8,
            radial_segments: 5,
        };
        let mesh = tessellate_mycelial_model(&model, config).expect("valid tessellation");
        let per_filament_vertices = 9 * 5;
        let per_filament_indices = 8 * 5 * 6;
        assert_eq!(mesh.filaments.len(), model.filaments.len());
        assert_eq!(
            mesh.vertex_count(),
            model.filaments.len() * per_filament_vertices
        );
        assert_eq!(
            mesh.indices.len(),
            model.filaments.len() * per_filament_indices
        );
        assert_eq!(mesh.triangle_count(), mesh.indices.len() / 3);
        assert!(mesh.validate());
    }

    #[test]
    fn tessellation_is_deterministic_and_normals_are_unit_length() {
        let model =
            build_mycelial_model(&EmbodimentSnapshot::default(), PresenceMode::Conversational);
        let config = TubeTessellationConfig::default();
        let first = tessellate_mycelial_model(&model, config).expect("valid mesh");
        let second = tessellate_mycelial_model(&model, config).expect("valid mesh");
        assert_eq!(first, second);
        assert!(first.normals.iter().all(|normal| {
            let length =
                (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            (length - 1.0).abs() < 1.0e-3
        }));
    }

    #[test]
    fn tessellation_rejects_unbounded_resolution_requests() {
        let model = build_mycelial_model(&EmbodimentSnapshot::default(), PresenceMode::Ambient);
        assert_eq!(
            tessellate_mycelial_model(
                &model,
                TubeTessellationConfig {
                    axial_segments: 1,
                    radial_segments: 6
                },
            )
            .unwrap_err(),
            TubeTessellationError::AxialResolutionOutOfRange(1)
        );
        assert_eq!(
            tessellate_mycelial_model(
                &model,
                TubeTessellationConfig {
                    axial_segments: 12,
                    radial_segments: 33
                },
            )
            .unwrap_err(),
            TubeTessellationError::RadialResolutionOutOfRange(33)
        );
    }
    #[test]
    fn malformed_geometry_and_unsupported_versions_fail_closed() {
        let mut model = build_mycelial_model(&EmbodimentSnapshot::default(), PresenceMode::Ambient);
        model.filaments[0].radius_start = f32::NAN;
        assert_eq!(
            tessellate_mycelial_model(&model, TubeTessellationConfig::default()).unwrap_err(),
            TubeTessellationError::InvalidFilamentGeometry(0)
        );

        let mut model = build_mycelial_model(&EmbodimentSnapshot::default(), PresenceMode::Ambient);
        model.schema_version = MYCELIAL_MODEL_VERSION + 1;
        assert_eq!(
            tessellate_mycelial_model(&model, TubeTessellationConfig::default()).unwrap_err(),
            TubeTessellationError::UnsupportedModelVersion(MYCELIAL_MODEL_VERSION + 1)
        );
    }

    #[test]
    fn mesh_resource_limits_are_checked_before_allocation() {
        let mut model = build_mycelial_model(&EmbodimentSnapshot::default(), PresenceMode::Ambient);
        let seed = model.filaments[0].clone();
        while model.filaments.len() < MAX_MYCELIAL_FILAMENTS {
            let mut filament = seed.clone();
            filament.id = model.filaments.len() as u32;
            filament.parent_id = None;
            model.filaments.push(filament);
        }
        assert_eq!(
            tessellate_mycelial_model(
                &model,
                TubeTessellationConfig {
                    axial_segments: 128,
                    radial_segments: 32,
                },
            )
            .unwrap_err(),
            TubeTessellationError::MeshVertexLimitExceeded(MAX_MYCELIAL_FILAMENTS * 129 * 32)
        );

        model.filaments.push(seed);
        assert_eq!(
            tessellate_mycelial_model(&model, TubeTessellationConfig::default()).unwrap_err(),
            TubeTessellationError::FilamentCountExceeded(MAX_MYCELIAL_FILAMENTS + 1)
        );
    }

    #[test]
    fn nonfinite_or_unbounded_spatial_values_are_rejected() {
        let mut model = build_mycelial_model(&EmbodimentSnapshot::default(), PresenceMode::Ambient);
        model.filaments[0].end.x = MAX_FILAMENT_COORDINATE + 1.0;
        assert_eq!(
            tessellate_mycelial_model(&model, TubeTessellationConfig::default()).unwrap_err(),
            TubeTessellationError::InvalidFilamentGeometry(0)
        );
    }

    #[test]
    fn cyclic_or_out_of_order_parent_references_are_rejected() {
        let mut model = build_mycelial_model(&EmbodimentSnapshot::default(), PresenceMode::Ambient);
        model.filaments[0].parent_id = Some(model.filaments[0].id);
        assert_eq!(
            tessellate_mycelial_model(&model, TubeTessellationConfig::default()).unwrap_err(),
            TubeTessellationError::InvalidFilamentTopology
        );

        let mut model = build_mycelial_model(&EmbodimentSnapshot::default(), PresenceMode::Ambient);
        model.filaments[1].parent_id = Some(u32::MAX);
        assert_eq!(
            tessellate_mycelial_model(&model, TubeTessellationConfig::default()).unwrap_err(),
            TubeTessellationError::InvalidFilamentTopology
        );
    }

    #[test]
    fn tube_triangle_winding_agrees_with_vertex_normals() {
        let filament = MycelialFilament {
            id: 7,
            parent_id: None,
            generation: 0,
            terminal: true,
            start: SpatialPoint3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            control_a: SpatialPoint3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            control_b: SpatialPoint3 {
                x: 2.0,
                y: 0.0,
                z: 0.0,
            },
            end: SpatialPoint3 {
                x: 3.0,
                y: 0.0,
                z: 0.0,
            },
            radius_start: 1.0,
            radius_end: 1.0,
        };
        let config = TubeTessellationConfig {
            axial_segments: 2,
            radial_segments: 4,
        };
        let mut mesh = MycelialTubeMesh {
            positions: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
            filaments: Vec::new(),
        };
        tessellate_filament(&filament, config, &mut mesh).expect("valid straight tube");

        let [a, b, c] = [
            mesh.indices[0] as usize,
            mesh.indices[1] as usize,
            mesh.indices[2] as usize,
        ];
        let edge_ab = sub3(mesh.positions[b], mesh.positions[a]);
        let edge_ac = sub3(mesh.positions[c], mesh.positions[a]);
        let face_normal = cross3(edge_ab, edge_ac);
        assert!(
            dot3(face_normal, mesh.normals[a]) > 0.0,
            "triangle face normal must point with the generated outward vertex normal"
        );
    }

    fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
}
